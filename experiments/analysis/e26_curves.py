# Summarize e25/e26 training-arm jsonl logs: held-out eval curve,
# train-reward halves, sustained clip fraction.
# Usage: python e26_curves.py e26_f0.jsonl e26_f1.jsonl [...]
import json
import sys

for path in sys.argv[1:]:
    recs = [json.loads(l) for l in open(path)]
    evals = [(r["step"], r["eval_acc"]) for r in recs if "eval_acc" in r]
    tr = [r for r in recs if "reward" in r and not r.get("skipped")]
    clip = sum(r.get("clip_frac", 0) for r in tr) / max(len(tr), 1)
    if evals:
        mid = evals[len(evals) // 2][0]
        e1 = [a for s, a in evals if s <= mid]
        e2 = [a for s, a in evals if s > mid]
        print(f"{path}: updates={len(tr)} clip={clip:.4f} "
              f"eval first-half {sum(e1)/len(e1):.3f} "
              f"second-half {sum(e2)/len(e2):.3f}")
        print("  curve:", " ".join(f"{a:.2f}" for _, a in evals))
    else:
        r1 = [r["reward"] for r in tr[: len(tr) // 2]]
        r2 = [r["reward"] for r in tr[len(tr) // 2:]]
        print(f"{path}: updates={len(tr)} clip={clip:.4f} "
              f"reward {sum(r1)/len(r1):.3f} -> {sum(r2)/len(r2):.3f}")
