# MPK-Det experiment suite

Layout:

- `*.py`, `*.cu` — measurement clients and experiment drivers (details below)
- `runners/` — the exact orchestration shell scripts used on the B200 pods
  (server lifecycle, arm sequencing, output collection). `e21b`/`e22b`
  are matched-length / default-backend reruns of `e21`/`e22`;
  `tp2_loop2.sh` is the TP=2 upstream crash repro (3/3 SIGABRT with
  coredumps); `moe_rescore.sh` is the MoE bitwise rescore check.
- `analysis/` — post-processing: `dt_histogram.py` (Figure 2 binning),
  `e26_curves.py` (training-arm eval curves).
- `hopper/h100_pod.yaml` — self-contained H100 validation pod: clones
  this branch, builds, runs the flagged-vs-stock split-K arms, prints
  text md5s to stdout (the linter's Hopper falsifiability test).
- Seed replicates: `e26b` = `e26_train.py` with seeds 20260728→20260729.

Scripts backing the paper's evaluation. Each maps to a research question;
all were run on B200 (sm_100a) unless noted. Model: Qwen/Qwen3-8B unless
noted. SGLang baselines use a dedicated venv (`sglang.launch_server`).

## Determinism & rescore consistency (RQ1/RQ2)

- `../demo/qwen3/rescore_consistency.py` — decode-vs-chunked-prefill
  bitwise rescore harness (K-sweep, full rescore, sampled mode). Drives
  `demo.py --deterministic --capture-probs --dump-tokens-file`.
- The MoE variant uses the same flags on `demo_30B_A3B.py`
  (Qwen3-30B-A3B; expect 332/332 bitwise at the default prompt).
- `../tools/determinism_check.py` — static linter for conditions
  (a)/(b)/(c); run against `include/mirage/persistent_kernel/tasks`.

## Mismatch measurement (RQ5, Figure 2)

- `e20_dt.py` — SGLang same-engine per-token δt: sample with
  `return_logprob`, rescore the identical ids via prefill
  (`logprob_start_len`). Run once per server mode (normal / det).
- `e20_cross.py`, `e14_sgl_dt.py` — cross-engine δt: MPK rollout ref
  (from `--dump-tokens-file`) rescored by SGLang.
- `e27_vllm_dt.py` — same, rescored by vLLM `prompt_logprobs`
  (offline API; run under the vLLM venv).

## Performance (RQ3/RQ3b/RQ3c)

- `e13_throughput.py` — legacy client (counts requested tokens; kept for
  provenance only — superseded by e21).
- `e21_throughput.py` — closed-loop concurrency client with
  actual-token accounting (`usage.completion_tokens`); C in {1,2,4,8}.
- `e22_len.py` — per-token decode latency vs context length (radix-warm
  protocol); run per server mode. Compare det against the engine's
  *fastest* baseline (see paper RQ3c for the triton-baseline artifact).
- `sampler_stat_test.cu` — statistical validation of the position-keyed
  Gumbel sampler (marginal + lag-1 χ²); `-DOLD_SCHEME` reproduces the
  upstream constant-offset failure.

## RL-engine baselines and step-time

- `e32_sgl.py`, `e38_sgl.py` — the recompute-deployment baseline arm:
  generate a GRPO group on SGLang, then acquire pi_old via the
  miles-style prefill rescore (`logprob_start_len`). e32 = Qwen3-1.7B,
  e38 = Qwen3-8B. Paired with the MPK arm (demo `--grpo-*`, see below).
- `e43_miles_baseline.py` — proves the baseline IS miles' code: imports
  miles' own `_build_prefill_scoring_payload`, drives it against SGLang,
  and checks the payload is field-identical to ours (verifier for the
  paper's baseline claim). Run under the miles source on PYTHONPATH.
- `e44_weight_sync_bench.py` — trainer→rollout synchronization benchmark
  for the reusable `mirage.mpk.weight_sync` plan. `--target synthetic`
  sanity-checks padding and TP slicing without model download;
  `--target mpk` loads a real MPK rollout target and reports per-step
  sync bytes, latency, and effective bandwidth.
- `runners/e32_run.sh`, `runners/e38_run.sh` — orchestrate both arms
  (MPK GRPO + SGLang server + baseline) on one pod.
- `runners/e43_run.sh` — brings up SGLang and runs the miles-baseline
  verifier.
- `runners/e44_run.sh` — runs synthetic and real-MPK weight-sync
  benchmarks; this is the bridge experiment toward the disaggregated
  miles-class comparison.
- `e45_e2e_accounting.py` — same-trainer E2E projection and speedup gate.
  It removes E19's optional old-logprob diagnostic from MPK wall-clock,
  holds reward/trainer/optimizer/sync time fixed, and swaps only MPK rollout
  against SGLang generation + miles-style rescore. The script warns when
  generated-token totals are not matched; it is an optimization gate, not a
  replacement for the final full-miles wall-clock run.
- `e50_sglang_miles.py` — matched-token SGLang arm for the E2E gate. It
  uses the same GSM8K prompts and total sequence cap as MPK, ignores EOS so
  both arms process exactly the same token count, and invokes miles' own
  prefill-scoring payload builder for the old-policy rescore phase.
- `e52_online_mpk.py` — matched-token MPK serving arm. It submits the same
  raw token ids through the completion API, requests deterministic sampled
  rollouts and captured logprobs, ignores EOS, and verifies every response
  reaches the requested length. Together with E50 it measures the rollout
  boundary without changing the shared reward/backward/update tail.

The E51/E52 matched artifacts process 135,616 generated tokens in each
arm. The current B200 result is a negative performance result: the offline
MPK step is 2.495 s and the online projection is 2.454 s, versus 1.147 s
for normal SGLang generation plus miles-style prefill rescore and the same
trainer tail. The zero-TIM path is therefore implemented but does not yet
win E2E at Qwen3-1.7B/group-16; the remaining gap is MPK decode GEMM
throughput, not probability capture or serving setup.

Example with the archived 1.7B data:

```bash
python experiments/e45_e2e_accounting.py \
  --model Qwen3-1.7B \
  --mpk-log experiments/data/e39_full17b.jsonl \
  --baseline-log experiments/data/e32_sgl.jsonl
```

## RL experiments (RQ6/RQ7 + dose-response)

- `../demo/qwen3/e19_grpo.py` — GRPO loop with MPK rollouts. The MPK arm
  uses rollout-captured values in the loss while differentiable trainer
  logprobs provide the gradient path; the `hf` arm uses trainer recompute
  values as well as its gradient path.
- `--grpo-trainer-backend` selects `hf`, `torchtitan`, `megatron`, or a
  project-specific `<module>:<factory>`. The built-in TorchTitan path loads
  TorchTitan's native Qwen3 model, maps HF weights/names with its state-dict
  adapter, and uses TorchTitan gradient clipping and optimizer containers.
  The built-in Megatron path uses Megatron Bridge for Qwen3 conversion,
  gathers TP logits for selected-token log-softmax, finalizes distributed
  gradients, and steps the Megatron optimizer. Both adapters accept one
  non-pipeline model part: prebuilt TorchTitan stacks may carry TP/FSDP2,
  while the Megatron builder configures TP/DDP and its distributed optimizer.
  PP is rejected until the RL loss is hosted inside the framework's coupled
  pipeline forward/backward schedule.
- `e23_noise.py` — one-step dose-response: inject the measured empirical
  δt distribution (from e20) into the GRPO ratio at scale f; reports
  clip fraction, ratio tails, gradient cosine.
- `e25_train.py` — 300-step training version (Qwen3-1.7B).
- `e26_train.py` — at-scale version (Qwen3-8B, thousands of steps,
  held-out greedy eval every 100 steps).

Install optional stacks with `pip install -e '.[torchtitan]'` or
`pip install -e '.[megatron]'`. The Megatron extra is pinned to the
Megatron Bridge 0.3/Core 0.16 line compatible with this repository's
Transformers 4.x dependency. A single-node TorchTitan run is selected by:

```bash
python demo/qwen3/demo.py --use-mirage --model Qwen/Qwen3-1.7B \
  --deterministic --sampling-seed 42 --capture-probs \
  --grpo-steps 1 --grpo-trainer-backend torchtitan
```

The built-in Megatron path initializes a single-process NCCL/model-parallel
runtime when needed and reuses an existing distributed runtime when launched
under `torchrun`. Distributed launchers can alternatively pass prebuilt model
chunks/optimizers through a `<module>:<factory>` integration.

The archived B200/Qwen3-1.7B runs are `e54` (TorchTitan), `e55`
(Megatron), and `e56` (Megatron with the eliminated old-policy replay timed
as a paired counterfactual). They use TorchTitan 0.2.0 and Megatron-Core
0.16.1 + Megatron Bridge 0.3.1, respectively.

## TP crash repro (RQ4)

Upstream split-K under TP=2 (deterministic SIGABRT; coredump pins the
NVLS `multimem.ld_reduce` in `allreduce.cuh`):

```bash
mpirun -np 2 --allow-run-as-root \
  python demo/qwen3/demo.py --use-mirage --model-path <qwen3-8b-mp2-dir>
# with CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 CUDA_ENABLE_LIGHTWEIGHT_COREDUMP=1
```

The deterministic path (`--deterministic`) on the same setup completes
and is bitwise reproducible.

## Hopper prediction (linter falsifiability)

On H100 (sm_90a): enable `use_splitk` for cc90 in `demo.py` (the gate is
`use_splitk = (target_cc == 100)`), which selects
`splitk_linear_swapAB_hopper` — the linter-flagged `tma_reduce_add`
path. Prediction: rerun divergence with split-K on, bitwise-identical
reruns with the stock (tma_store) path.
