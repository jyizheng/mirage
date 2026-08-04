# Engine validation harnesses

In-process / on-pod checks that validate the RL-engine deliverables. Run
on a B200 pod with the fork built (`pip install -e .`). These produce the
bitwise results cited in the paper's §7 / RQ2 / Table 3.

- `e35_serving_rescore.py` — serving-level rescore consistency. Rolls out
  through the online engine (capture → API logprobs), resubmits the full
  token sequence as a teacher-forcing prompt, compares float32 bit
  patterns. Result: 260/260 bitwise with `MPK_DETERMINISTIC=1`; 69/259
  with upstream kernels (nondet control). Run in-process (only
  copy-engine ops while the MPK kernel is resident).

- `e31_capture_diag.py` — capture-buffer diagnostic used to root-cause the
  online slot-vs-buffer-row aliasing bug (commits 735e8d1/49dca53/d5d32a8).
  Prints per-row nonzero counts and the in-kernel diag sentinel.

- `e40_sampling_params.sh` — deterministic temperature/top-k/top-p
  validation (commit f508b4e). Checks: default path bitwise-stable; top-k
  rerun identical AND differs from default; top-p/temperature run.

- `e42_reference_cocompile.sh` — reference-model co-compilation (commits
  db3d78a..255c667). ref==policy → reference logprobs must equal the
  policy's bit-for-bit. Result: 334/334 (38 teacher-forced prompt + 296
  decoded). Raw dump archived at `../data/reference_cocompile_ref.json`.

Note: `../e43_miles_baseline.py` (proves the recompute baseline is miles'
own `_build_prefill_scoring_payload`) and the paper-figure experiments
live one level up in `experiments/`.
