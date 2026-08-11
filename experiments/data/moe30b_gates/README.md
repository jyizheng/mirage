# Qwen3-30B-A3B MoE online determinism gate artifacts

Generated at 58b91b7a on ajiao-dev-pod-b200-0 (B300, cc 10.3, TP=1) by
`experiments/validation/moe30b_run.sh 0`, 2026-08-11.  Engine config for
every run: `--deterministic --capture-logprobs --ignore-eos
--max-seq-length 512 --max-num-batched-requests 16
--max-num-batched-tokens 16 --pinned-ring-capacity 32`,
`MPK_DET_NUM_SPLITS=4`; seeded runs add `--sampling-seed 42`.

All six design-phase-(c) gates PASS (see RESULTS.txt):

| gate | check | result |
|---|---|---|
| i | greedy rerun bitwise, same + fresh server process | PASS |
| ii | seeded (42) rerun bitwise across server restart | PASS |
| iii | 16-member greedy group_completions == single reference | PASS |
| iv | target request bitwise unchanged alone vs +3 concurrent | PASS |
| v | rescore == rollout, 343/343 captured probs bitwise (rescore.log) | PASS |
| vi | dense Qwen3-1.7B greedy rerun bitwise on the same build | PASS |

Perf (rough, one B300): 30B-A3B single-stream decode ~70 tok/s;
16-wide greedy group ~1390 tok/s aggregate; dense 1.7B ~383 tok/s.

These runs exercise the phase-(b) kernel fixes (a961f2ef): fixed-order
active-expert compaction in `topk_softmax_sm100.cuh` and zero-filled
unrouted B lanes in `moe_linear_sm100.cuh`.
