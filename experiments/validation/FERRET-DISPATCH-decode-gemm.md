# Ferret dispatch spec: decode-GEMM task optimization (direction 3)

Prepared 2026-08-04; launch via the `ferret-kernel-system` skill when a
B200 frees up. This is the third leg of closing the 2.56× rollout gap
(paper §6 "matched E2E accounting"), alongside shared-prefix group
prefill (e53, implemented) and mbt-specialized graphs (e53 runner A/B).

## Target

`TASK_LINEAR_SM100` (`include/mirage/persistent_kernel/tasks/blackwell/
linear_sm100_mpk.cuh`) and `TASK_SPLITK_PARTIAL_LINEAR_SM100` at the
decode shapes that dominate the matched group-16 workload:

- e37 trace attribution: LINEAR 51% + SPLITK_PARTIAL 37% of all task
  time; post-E56 the residual gap "is in decode GEMMs after sampling,
  capture, and episode setup were reduced" (paper §9).
- Decode shapes (Qwen3-1.7B, mbt=8/16 window): M = mbt rows,
  K/N ∈ {2048×6144 qkv (fused), 2048×2048 o_proj, 2048×12288 gatedup,
  6144×2048 down (split-K)}; lm_head M=mbt, N=153600, K=2048.
- Per-task avgs from e37: LINEAR 14.6 µs (36,861 execs), SPLITK 6.8 µs
  (57,339 execs). cuBLAS-equivalent for these shapes is the bar (the
  MPK paper's own Fig. 12 shows their pipelined task at 1.15–1.29× over
  cuBLAS — we appear well short of that on this branch/shape).

## Frozen-gate constraints (why ferret, not a bare optimizer agent)

- Determinism conditions are non-negotiable: (b) intra-task reduction
  order must remain a pure function of compile-time params; (c) no
  arrival-order combines (no atomics/tma_reduce_add on outputs). The
  independent test-writer must include a rerun-bitwise gate and a
  vs-reference numerical gate (bitwise against the CURRENT kernel on
  identical inputs is the cleanest: optimization must not change values,
  only speed — same-order reductions preserved).
- Test harness exists: tests/runtime_python/blackwell/sm100_linear/
  (linear_sm100_mpk + linear_splitk_sm100 bindings) — the test-writer
  should extend it with timing + bitwise gates at the decode shapes
  above.

## Acceptance

- ≥1.2× on the M≤16 decode shapes, bitwise-identical outputs, twins/
  rerun/rescore ladder re-passes (validation/e40, e42 scripts), and the
  matched group-16 rollout (e52/e53) improves accordingly.
