# Option C validation: 2-GPU disaggregated GRPO (--trainer-device)

Pod ajiao-dev-pod-b200-0 (B300s, NV18), Qwen3-1.7B, HF backend, group 16,
seed 42, --deterministic, MPK_DET_NUM_SPLITS=4, mbt 16, max-seq 512,
5 outer steps each. Branch fix-deterministic-decode @ 799e51b7.

| run | config | eos | notes |
|-----|--------|-----|-------|
| a   | colocated (GPU0) | ignore | pre trainer-determinism fix |
| a2  | colocated rerun  | ignore | proves HF backward run-to-run nondeterminism: gn differs at step 0 vs a, SHAs diverge from step 1 |
| a3  | colocated | ignore | post fix (use_deterministic_algorithms) |
| b   | disagg + per-burst streaming (13-15 waves) | ignore | first cut: step-0 bitwise, later steps diverge via fragmented-backward grads; t_train 0.84-0.97s vs 0.26s |
| b2  | disagg, batched fwd | ignore | pre determinism fix: loss matches step 0, gn noise |
| b3  | disagg, batched fwd | ignore | **bitwise == a3 on all 5 steps (sha, loss, grad_norm); r==1 all steps** |
| c1  | colocated | real | EOS-realistic control |
| c2  | disagg, batched | real | **bitwise == c1 all 5 steps**; step parity (1.098 vs 1.104 s mean) |
| c3  | disagg + --stream-replay-fwd (wave floor R/4) | real | streaming LOSES: mean step 1.358s (+24% vs c2); fragmented backward + non-overlapping final flush exceed the ~0.1s tail harvest (decode idle 13.8%) |
| d   | disagg, --inner-epochs 3, --no-fused-sampling-capture | ignore | ep1 rdev==0 all steps; ep2/3 rescore 2.26s each; multi-epoch + cross-device sync (11.8-12.3ms) work |

Cross-device weight sync: 3.20 GiB / 310 tensors in 11.8 ms (271 GiB/s over
NVLink; colocated same-device copy is 2.9 ms). Matched-arm overhead of
disaggregation: +0.9% step time (b3 vs a3).
