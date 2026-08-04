#!/usr/bin/env python3
"""E45: same-trainer E2E accounting for MPK vs recompute rollout paths.

This is deliberately a projection, not a substitute for a full miles run.
It holds the measured MPK reward/trainer/optimizer/sync portion constant and
swaps only the inference path:

  MPK:     rollout with authoritative captured logprobs
  baseline: SGLang rollout + miles-style old-logprob prefill rescore

Legacy E19 logs measured the eliminated trainer-side recompute inside every
MPK step.  E45 removes that diagnostic before comparing engine time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"{path}: no JSONL records")
    return rows


def mean(rows: list[dict], key: str) -> float:
    try:
        return statistics.fmean(float(row[key]) for row in rows)
    except KeyError as exc:
        raise ValueError(f"missing required field {key!r}") from exc


def old_recompute_time(row: dict) -> float:
    return float(row.get("t_old_recompute_s", row.get("t_recompute_s", 0.0)))


def generated_tokens(rows: list[dict]) -> int:
    return sum(sum(int(length) for length in row["gen_lens"]) for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpk-log", type=Path, required=True)
    parser.add_argument("--baseline-log", type=Path, required=True)
    parser.add_argument("--model", default="unknown")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--require-speedup",
        action="store_true",
        help="Exit nonzero unless projected baseline_time / MPK_time > 1.",
    )
    args = parser.parse_args()

    mpk = load_jsonl(args.mpk_log)
    baseline = load_jsonl(args.baseline_log)

    mpk_step_raw = mean(mpk, "t_step_s")
    mpk_old_diagnostic = statistics.fmean(old_recompute_time(row) for row in mpk)
    mpk_step = mpk_step_raw - mpk_old_diagnostic
    mpk_rollout = mean(mpk, "t_rollout_s")
    shared_tail = mpk_step - mpk_rollout

    baseline_generate = mean(baseline, "t_gen_s")
    baseline_rescore = mean(baseline, "t_rescore_s")
    baseline_inference = baseline_generate + baseline_rescore
    projected_baseline_step = baseline_inference + shared_tail

    speedup = projected_baseline_step / mpk_step
    rollout_gap = mpk_rollout - baseline_inference
    rollout_target = baseline_inference
    required_reduction = max(0.0, rollout_gap)

    mpk_tokens = generated_tokens(mpk)
    baseline_tokens = generated_tokens(baseline)
    token_ratio = mpk_tokens / baseline_tokens if baseline_tokens else float("inf")

    result = {
        "experiment": "E45 same-trainer projection",
        "model": args.model,
        "mpk_steps": len(mpk),
        "baseline_steps": len(baseline),
        "mpk_step_raw_s": mpk_step_raw,
        "eliminated_diagnostic_s": mpk_old_diagnostic,
        "mpk_engine_step_s": mpk_step,
        "mpk_rollout_s": mpk_rollout,
        "shared_reward_train_sync_s": shared_tail,
        "baseline_generate_s": baseline_generate,
        "baseline_old_rescore_s": baseline_rescore,
        "projected_baseline_step_s": projected_baseline_step,
        "projected_speedup_baseline_over_mpk": speedup,
        "rollout_gap_s": rollout_gap,
        "mpk_rollout_target_s": rollout_target,
        "required_rollout_reduction_s": required_reduction,
        "mpk_generated_tokens": mpk_tokens,
        "baseline_generated_tokens": baseline_tokens,
        "generated_token_ratio": token_ratio,
        "passes_speedup_gate": speedup > 1.0,
    }

    print(f"E45 same-trainer projection: {args.model}")
    print(
        f"  MPK engine step:       {mpk_step:.4f}s "
        f"(raw {mpk_step_raw:.4f}s, removed diagnostic "
        f"{mpk_old_diagnostic:.4f}s)"
    )
    print(
        f"  Baseline projected:    {projected_baseline_step:.4f}s "
        f"(generate {baseline_generate:.4f}s + old-rescore "
        f"{baseline_rescore:.4f}s + shared tail {shared_tail:.4f}s)"
    )
    print(f"  Projected speedup:     {speedup:.3f}x")
    print(
        f"  Rollout target:        {mpk_rollout:.4f}s -> "
        f"{rollout_target:.4f}s (reduce {required_reduction:.4f}s)"
    )
    if not 0.95 <= token_ratio <= 1.05:
        print(
            f"  WARNING: generated-token totals differ ({mpk_tokens} vs "
            f"{baseline_tokens}, ratio {token_ratio:.3f}); rerun with matched "
            "token traces before making a paper claim."
        )

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2) + "\n")

    if args.require_speedup and speedup <= 1.0:
        print("E45 SPEEDUP GATE: FAIL", file=sys.stderr)
        return 1
    print("E45 SPEEDUP GATE: PASS" if speedup > 1.0 else "E45 SPEEDUP GATE: OPEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
