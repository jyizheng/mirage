# Bin per-token |delta_t| into the log-decade histogram of the paper's
# Figure 2. Input: one or more e20/e27-style JSONs ({"deltas": [...]}).
import json
import sys

EDGES = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
LABELS = ["=0", "<=1e-6", "<=1e-5", "<=1e-4", "<=1e-3", "<=1e-2",
          "<=1e-1", "<=1"]

for path in sys.argv[1:]:
    d = json.load(open(path))
    deltas = d["deltas"] if isinstance(d, dict) else d
    n = len(deltas)
    counts = [0] * len(EDGES)
    for x in deltas:
        a = abs(x)
        if a == 0.0:
            counts[0] += 1
            continue
        for i in range(1, len(EDGES)):
            if a <= EDGES[i]:
                counts[i] += 1
                break
        else:
            counts[-1] += 1
    pct = [c / n * 100 for c in counts]
    print(path, "n=", n)
    for lab, p in zip(LABELS, pct):
        print(f"  {lab:>8}: {p:6.2f}%")
