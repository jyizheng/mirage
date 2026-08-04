# Raw experiment data

All Qwen3-8B on B200 unless noted. See `../README.md` for the scripts
that produced each file and the paper section each backs.

## Mismatch (δt) distributions — Figure 2 / RQ5 table
- `e20_normal.json`, `e20_det.json` — SGLang same-engine per-token
  signed δt (decode logprobs vs same-server prefill rescore), 8×512
  sampled tokens; `normal` = triton backend, `det` = deterministic mode.
- `e20_cross_greedy.json`, `e20_cross_sampled.json` — MPK rollout refs
  rescored by SGLang prefill (cross-engine deployment).
- `e27_vllm_greedy.json`, `e27_vllm_sampled.json` — same refs rescored
  by vLLM `prompt_logprobs`.

## Performance
- `e21_throughput_runs.log` — closed-loop concurrency (C=1/2/4/8),
  MPK-Det server + SGLang normal/det, matched ~260-token greedy,
  actual-token accounting (also contains the earlier 480-token run).
- `e22_length_sweep.log` — per-token decode latency vs context length
  (SGLang normal-triton / det-triton / default backend).

## Dose–response and training arms (GSM8K)
- `e23_noise.json` — one-step GRPO distortion vs injected mismatch
  scale f (Qwen3-1.7B; empirical e20-det deltas).
- `e25_f{0,1,3}.jsonl` — 300-step training arms, Qwen3-1.7B.
- `e26_f{0,1,3}.jsonl`, `e26b_f{0,1,3}.jsonl` — 3,000-step arms,
  Qwen3-8B, seeds 20260728/20260729; `eval_acc` records every 100
  steps (held-out 32-prompt greedy).

## RL engine step time (Qwen3-1.7B, G=8)
- `e32_mpk.jsonl` — e19 GRPO loop with timing decomposition
  (`t_rollout_s` / `t_recompute_s` (measured, unused) / `t_update_s`).
- `e32_mpk64.jsonl` — same with max_num_batched_tokens=64.
- `e32_sgl.jsonl` — SGLang arm (`t_gen_s` + miles-style prefill
  rescore `t_rescore_s`).
- `e33_mpk.jsonl` — after parallelizing the capture (+sampling) tasks
  (superseded: this run's sampler read wrong logits rows, see 28577d8).
- `e39_full17b.jsonl`, `e39_full8b.jsonl` — corrected runs after the
  sampling dmap fix (1.7B and 8B, 20 steps each).
- `e38_sgl.jsonl` — SGLang arm at Qwen3-8B.
- `e46_capture_fusion_17b.json` — group-8 sampled probability-capture
  fusion A/B on Qwen3-1.7B: three runs per variant, with token ids and
  captured float32 probability bits checked identical.
- `e47_mpk_optimized_17b.jsonl` — 20-step MPK GRPO run after capture
  fusion, full-group trainer replay, synchronized timing, and removal of
  the eliminated old-logprob diagnostic from engine wall-clock.
- `e47_e2e_projection_17b.json` — E45 same-trainer accounting over the
  optimized E47 MPK log and archived SGLang+miles-style rescore arm.
- `e48_one_pass_capture_17b.json` — one-pass sampled-probability capture
  A/B; token ids, probability bits, and the co-compiled reference check
  remain bitwise identical.
- `e49_mpk_one_pass_17b.jsonl`, `e49_e2e_projection_17b.json` — 20-step
  GRPO run and same-trainer projection after the one-pass capture change.
- `e50_mpk_group16_17b.jsonl` — preliminary group-16 MPK run retained for
  provenance; use the matched E50/E51 files below for comparisons.
- `e50_mpk_group16_matched_17b.jsonl`,
  `e50_sgl_group16_matched_17b.jsonl` — fixed-length, ignore-EOS MPK and
  SGLang+miles-style arms. Both process exactly 135,616 generated tokens.
- `e51_mpk_parallel_group16_matched_17b.jsonl`,
  `e51_e2e_matched_17b.json` — matched MPK arm after position-keyed
  vocabulary sampling was partitioned across 128 tasks, plus the E2E gate.
- `e52_online_mpk_group16_matched_17b.jsonl`,
  `e52_e2e_online_projection_17b.json` — the same workload through the
  OpenAI-compatible completion server, including captured logprobs. This
  isolates offline episode setup from the persistent online path.

## Serving-level checks
- `e35_serving_rescore.log` — online-engine rollout vs teacher-forcing
  resubmission: 260/260 bitwise with deterministic kernels; 69/259
  with upstream kernels (nondet control).
- `moe_rescore.log` — Qwen3-30B-A3B rescore: 332/332 bitwise.
