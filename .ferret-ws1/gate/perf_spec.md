# Ferret gate — performance specification

## Metric

CUDA-event timing through the frozen harness wrapper (identical for
reference and candidate): per case, 200 warmup iterations, then 3 rounds of
1000 timed iterations on the current torch stream; the reported number is
the best round's mean **µs/iter**. Single GPU (CUDA_VISIBLE_DEVICES=3, B300
class, sm_103), single stream, no CUDA graphs.

- `linear_*` cases time one kernel launch.
- `splitk_*` cases time the FULL 8-launch deterministic partial pass
  (8 sequential TASK_SPLITK_PARTIAL kernels).
- `reduce_*` cases time one splitk_reduce launch (1 CTA × 256 threads).

## Baseline (current in-tree kernel, this pod, 2026-08-05, GPU 3)

`baseline_perf.json` is the machine-readable copy used by check.py.

| perf case                   | baseline µs/iter |
|-----------------------------|------------------|
| linear_m8_n64_k2048_normal  | 8.192  |
| linear_m8_n128_k2048_normal | 8.192  |
| linear_m8_n256_k2048_normal | 10.240 |
| linear_m16_n64_k2048_normal | 8.192  |
| linear_m16_n128_k2048_normal| 8.193  |
| linear_m16_n256_k2048_normal| 10.239 |
| splitk_m8_ks256_normal      | 32.787 |
| splitk_m8_ks768_normal      | 49.133 |
| splitk_m16_ks256_normal     | 32.788 |
| splitk_m16_ks768_normal     | 49.134 |
| reduce_m8_normal            | 3.912  |
| reduce_m16_normal           | 4.096  |

Sanity: per-launch time scales with K (K=2048 -> 8.2 µs; splitk per-split
K=768 -> 49.13/8 = 6.14 µs; K=256 -> 32.79/8 = 4.10 µs); N=256 runs two
m_tiles (10.24 µs); M=8 vs M=16 identical (same 16-row MMA tile). The tiny
1-CTA reduce sustains ~4 µs/iter back-to-back, so the linear numbers are
kernel-execution-dominated, not launch-throughput-bound.

## Target (what check.py enforces as `perf_pass`)

- **Geomean speedup ≥ 1.20×** over the three M=16 LINEAR cases:
  `linear_m16_n64_k2048_normal`, `linear_m16_n128_k2048_normal`,
  `linear_m16_n256_k2048_normal`.
- **No perf case may regress more than 5%** (candidate µs ≤ 1.05 × baseline
  µs) across ALL 12 perf cases, including the M=8 linears, the split-K
  partial passes and the reduces.
- Split-K cases are reported (and regression-bounded) but not part of the
  geomean target.

## Final acceptance (out of scope for this standalone gate)

The standalone harness number is DIAGNOSTIC ONLY. The production megakernel
compiles with `-rdc=false` (no NVSHMEM; `python/mirage/mpk/persistent_kernel.py:227`),
256-thread worker CTAs, no CUDA graphs, single stream. FINAL acceptance is
run by L1, not by this gate:

1. the optimized kernel compiled into the FULL megakernel, showing a
   20-step matched-accounting rollout improvement, and
2. the greedy 16-member bitwise gate re-pass.

A candidate that wins here but loses inside the megakernel does not ship.
