# Ferret frozen gate — Mirage MPK decode GEMM (Blackwell)

**The optimizer must not edit anything under `gate/`. The gate is
hash-locked: `../gate.sha256` records the sha256 of every file here, and
`check.py` refuses to run (emits a failing `GATE_RESULT`) if any gate file is
edited, added, or removed. Passing this gate by modifying the gate is a
FAIL.**

## Target

Repo: `mirage-det`, branch `fix-deterministic-decode`
(pod clone `/root/.cache/mirage-det` @ `5e7d6a525519c7af4b8da80aea0231e9d88e35c2`;
the two kernel headers, `tma.cuh` and the sm100_linear harness are
byte-identical to local `29fb80857c488c6e5dce55fbbd8f33ce7320631f`).

Kernels under optimization:
1. `linear_sm100_mpk_task_impl` —
   `include/mirage/persistent_kernel/tasks/blackwell/linear_sm100_mpk.cuh`
   (TASK_LINEAR_SM100; and with SplitK=false + per-split K,
   TASK_SPLITK_PARTIAL_LINEAR_SM100).
2. `splitk_reduce_task_impl` —
   `include/mirage/persistent_kernel/tasks/blackwell/splitk_reduce_sm100.cuh`
   (TASK_SPLITK_REDUCE_SM100).

Baseline = these CURRENT kernels, benched exactly as the (adapted) harness
calls them. Goal: speed at small-M decode shapes with UNCHANGED numerics.

## Real-math contract (bit-for-bit)

- **linear**: `out[M, N_task] = act[M, K] @ W[N_task, K]^T`; bf16 inputs and
  outputs, fp32 accumulation, K reduced in a FIXED sequential k-tile order
  (current kernel: single MMA-warp chain over k tiles of bK=64,
  ScaleOut::Zero on the first tile then ScaleOut::One — the deterministic
  order). NOBIAS=true for the LINEAR and SPLITK_PARTIAL registrations
  gate-tested here; the optional bias/residual add is fp32 before the single
  fp32→bf16 convert. No atomics; no `tma_reduce_add` on the deterministic
  path; output via plain TMA store.
- **splitk_reduce**:
  `out[r,c] = bf16( fp32(residual[r,c]) + Σ_{s=0..7} fp32(partials[(s*M+r)*PARTIAL_STRIDE + c]) )`
  — residual first, splits in ascending order, fp32 accumulator, single
  final bf16 round.
- **THE GATE IS BITWISE**: candidate outputs must be bitwise identical
  (`torch.equal` on the raw bf16 tensors) to the CURRENT kernel's outputs on
  identical inputs. Timing may change; values may not.
- **Split-K partials are part of the contract**: `check.py` bitwise-checks
  BOTH the partials buffer `[8*M, 128]` AND the combined output. A candidate
  that fuses away or reformats the partials FAILS by default, even if the
  final combined output matches.

## Shapes (production Qwen3-1.7B decode; M = max_num_batched_tokens ∈ {8,16})

| case family      | per-task shape                       | K stride (gmem)  |
|------------------|--------------------------------------|------------------|
| qkv-like         | K=2048, N_task=64                    | 2048 (contig)    |
| gateup-like      | K=2048, N_task=128                   | 2048 (contig)    |
| lm_head-like     | K=2048, N_task=256                   | 2048 (contig)    |
| o_proj splitk    | per-split K=256, N=128, 8 splits     | 2048 (full K)    |
| down splitk      | per-split K=768, N=128, 8 splits     | 6144 (full K)    |
| splitk_reduce    | M×128, 8 splits, with residual       | 128              |

Each shape runs under three input distributions: `normal` (randn),
`small` (randn × 1e-20 → products land in the fp32-denormal /
bf16-denormal range) and `large` (randn × 1e4) to exercise rounding.
36 correctness cases total (see `reference.py all_cases()`).

Kernel-instantiation constants (fixed): MMA_M=128, MMA_N=16, TILE_SIZE=64,
OUTPUT_ATOM_SIZE=128, NUM_AB_STAGE=8, NUM_ACC_STAGE=2, NUM_C_STAGE=4,
256-thread CTA, 224KB dynamic smem, cluster (1,1,1).
TMA descriptors mirror production `fill_tma_desc_by_task` exactly, including
the `min(MMA_N, batch)` box clamp on input/output and full-K
`GMEM_STRIDE_ROW` for the split-K partial's strided K-slice views.

## Harness (../harness — frozen wrapper, shared by reference and candidates)

`gate_common.cuh` + `gate_cases_m{8,16}.cu` + `gate_bind.cu` + `setup.py`.
The ONLY differences vs `tests/runtime_python/blackwell/sm100_linear/`:
- setup.py: nvcc from `CUDA_HOME=/usr/local/cuda` (pod has CUDA 13.0; the
  original hardcoded /usr/local/cuda-12.8), arch `compute_103a/sm_103a`
  (torch reports capability (10,3); nvidia-smi's name string is wrong),
  include dirs from `MIRAGE_ROOT`, module name from `GATE_EXT_NAME`, plus
  `../compat_include` (symlinks to ONLY the pip-cu13 cusparse/cusolver
  headers the pod's toolkit lacks — never the whole pip include dir, whose
  newer crt/host_runtime.h breaks nvcc 13.0 host stubs).
- gate_common.cuh includes `<cute/tensor.hpp>` before the kernel headers
  (the kernel header's own include order trips a cute copy_atom/prefetch
  include cycle under this cutlass+nvcc combo) and includes only the two
  kernels under test rather than task_header.cuh (which drags non-inline
  definitions that break multi-TU linking).
- host plumbing: TMA descriptors cached per (in,w,out) pointer tuple instead
  of re-cudaMalloc'ed per call, and launches go to the current torch stream
  without a per-call device sync — so CUDA-event timing measures kernel
  time. Reference and candidate both go through this identical wrapper.

**Candidate build interface** (the only sanctioned way to enter the gate):
```
cd /root/.cache/ferret-ws1/harness
CUDA_HOME=/usr/local/cuda MAX_JOBS=4 \
  GATE_EXT_NAME=candidate_kernels \
  MIRAGE_ROOT=/path/to/tree/with/optimized/headers \
  python3 setup.py build_ext --inplace   # (or a copy of this dir; do not edit it)
CUDA_VISIBLE_DEVICES=3 python3 /root/.cache/ferret-ws1/gate/check.py \
  --module candidate_kernels --module-path /path/to/build/dir
```
The candidate module must expose `linear(x,w,out)`,
`splitk_partial(x,w,partials)`, `splitk_reduce(partials,residual,out)` with
the exact shapes above. Unsupported shapes / exceptions = case FAIL.

## Gate checks (check.py)

1. **Integrity**: every file under `gate/` must match `../gate.sha256`.
2. **Bitwise vs golden**: all 36 cases, `torch.equal` on raw bf16; each case
   runs TWICE with different output-buffer prefills (0.0 and −777.0) — this
   proves run-to-run determinism and that the kernel writes every checked
   output location (no pass-by-leaving-prefill).
3. **Fresh-input cross-check (anti-replay)**: for the 12 normal-dist shape
   families, new random inputs are drawn at check time (`os.urandom` salt)
   and the candidate must match the FROZEN reference extension
   (`gate/refext/`, hash-locked) bitwise. A candidate that memorizes the
   goldens cannot pass.
4. **Perf**: see `perf_spec.md`.

Output: exactly one line
`GATE_RESULT {"pass": bool, "bitwise": {...}, "perf": {...}, "first_failing_case": ...}`.

## Oracle validation (proves harness wiring; run on B300-class GPU, sm_103)

Golden outputs from the current kernel were compared against an independent
fp32 torch oracle (`matmul`, TF32 disabled). For `normal`/`small` cases:
strict per-element `assert_close(rtol=1e-2, atol=1e-2)` — all pass. For
`large` cases a per-element rtol is not meaningful under cancellation (fp32
accumulation-order noise vs torch exceeds 1% on isolated near-cancelled
elements), so they are validated by scale-normalized inf-norm:
`max|got-ref| <= 1e-2 * max|ref|` — all pass. The split-K combined oracle
follows the contract formula (per-split fp32 matmul → bf16 round →
ascending-order fp32 sum on residual); a naive full-fp32 matmul oracle is
NOT comparable at these tolerances because the real pipeline rounds each
partial to bf16. Per-case numbers: `oracle_validation.json`. Summary:
- worst per-element rel err (elements with abs err > atol): 2.06e-2 —
  `linear_m16_n128_k2048_large`, 1/2048 elements, heavy cancellation
  (covered by the inf-norm criterion; all other cases < 1e-2).
- worst scale-normalized inf-norm err ratio across ALL 36 cases: 3.45e-3
  (`linear_m8_n256_k2048_large`), bound 1e-2.
- worst abs err among normal-dist cases: 0.582 (`splitk_m16_ks768_normal`,
  output magnitudes O(80); consistent with bf16 output rounding).

Run-to-run determinism: all 36 cases produced bitwise-identical outputs
across two runs with different output-buffer prefills (0.0 / -777.0), which
also proves every checked output element is written by the kernel.
Environment: B300-class GPU (sm_103), torch 2.11.0+cu130, CUDA 13.0.

## Edge cases covered

- M=8 < MMA_N=16 (clamped TMA boxes + kClampedBN transaction bytes).
- N_task=64 < OUTPUT_ATOM_SIZE=128 (weight TMA OOB zero-fill rows; output
  TMA store col-clamp) and N_task=256 (two m_tiles).
- Strided K-slice TMA views for split-K partials (stride ≠ per-split K).
- Denormal-range and large-magnitude inputs (bf16 rounding paths).
- Partials buffer checked as a first-class output (format is contractual).
- Output buffers prefilled with sentinels (kernels must write, not inherit).

## Production flags note

The production megakernel compiles with `-rdc=false` (no NVSHMEM; see
`python/mirage/mpk/persistent_kernel.py:227`), 256-thread worker CTAs, no
CUDA graphs, single stream. The standalone harness numbers here are
DIAGNOSTIC. **Final acceptance (run by L1, not by this gate) = the optimized
kernel compiled into the full megakernel, a 20-step matched-accounting
rollout improvement, plus the greedy 16-member bitwise gate re-pass.**

## Freeze

```
cd /root/.cache/ferret-ws1
find gate -type f | sort | xargs sha256sum > gate.sha256
chmod -R a-w gate
```
