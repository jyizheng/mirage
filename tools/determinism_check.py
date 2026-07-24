#!/usr/bin/env python3
"""Static determinism linter for MPK task kernels.

Encodes the sufficient conditions for schedule-independent, bitwise
deterministic and batch-invariant execution of a task-graph runtime:

  (a) each task writes a disjoint output region (no cross-task write
      overlap, no read of another task's in-flight output);
  (b) every intra-task reduction order is a pure function of
      compile-time parameters (not of runtime batch composition, window
      row index, or scheduling state);
  (c) every cross-task combine is an explicit task whose operand order
      is fixed (never atomic accumulation in completion order).

This is a lint, not a verifier: it flags source patterns that can only
be sound if a human argument accompanies them, and classifies known
categories. Ground truth: run against MPK upstream, it flags exactly
the defects found empirically (split-K tma_reduce_add -> (c); sampler
slot-keyed noise -> (b)), plus the scheduling-only integer atomics in
MoE routing (classified, not failed).

Usage: python tools/determinism_check.py [task_dir]
"""
import re
import sys
from pathlib import Path

TASK_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else
                "include/mirage/persistent_kernel/tasks")

# (pattern, condition, severity, note)
RULES = [
    # -- condition (c): atomic / in-place-reduce combines --
    (r"tma_reduce_add|cp\.reduce\.async\.bulk", "c", "VIOLATION",
     "TMA reduction into global memory: partials combine in task-completion "
     "order. Replace with partial stores + a fixed-order reduce task."),
    (r"multimem\.(red|st)\b", "c", "REVIEW",
     "NVLS in-switch reduction: combine tree is fixed for a fixed team "
     "(deterministic run-to-run) but changes with team membership."),
    (r"\batomicAdd\s*\(\s*[^,]*(float|double|half|bfloat|__nv_bfloat)", "c",
     "VIOLATION",
     "Floating-point atomic accumulation: value depends on arrival order."),
    (r"\batomic(Add|Sub|Exch|CAS|Min|Max)\b", "c", "CLASSIFY",
     "Atomic on integer data: benign iff it orders scheduling metadata "
     "only and no floating-point value depends on the resulting order."),
    (r"\bred\.(global|relaxed|release)", "c", "VIOLATION",
     "PTX red.* reduction: arrival-order combine."),
    # -- condition (b): runtime-state-keyed numerics --
    (r"curand_init\s*\(", "b", "REVIEW",
     "RNG stream key: must be a pure function of (seed, request identity, "
     "sequence position); keying by window row / batch slot / step-counter "
     "alone breaks batch invariance and replay."),
    (r"__shfl_(up|down)_sync", "b", "INFO",
     "Directional shuffle reduction: order is fixed per warp; sound iff "
     "lane->element mapping is compile-time."),
    # -- condition (a): slot-keyed shared-state indexing (reader side) --
    (r"\[\s*batch_idx\s*\]|\[\s*row_idx?\s*\]|\[\s*b\s*\*\s*MAX_", "a", "INFO",
     "Slot-indexed shared buffer access: sound iff slot->request mapping "
     "is derived from request metadata (qo_indptr), not assumed."),
]

SKIP = {"cutlass", "cute"}


def scan(path: Path):
    findings = []
    for f in sorted(path.rglob("*.cuh")) + sorted(path.rglob("*.h")):
        if any(part in SKIP for part in f.parts):
            continue
        text = f.read_text(errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("*"):
                continue
            for pat, cond, sev, note in RULES:
                if re.search(pat, line):
                    findings.append((sev, cond, f, i, stripped[:90], note))
    return findings


def main():
    findings = scan(TASK_DIR)
    order = {"VIOLATION": 0, "CLASSIFY": 1, "REVIEW": 2, "INFO": 3}
    findings.sort(key=lambda x: (order[x[0]], str(x[2])))
    counts = {}
    shown_info = 0
    for sev, cond, f, i, line, note in findings:
        counts[sev] = counts.get(sev, 0) + 1
        if sev == "INFO":
            shown_info += 1
            if shown_info > 6:
                continue
        rel = f.relative_to(TASK_DIR.parent) if TASK_DIR.parent in f.parents \
            else f
        print(f"[{sev}] ({cond}) {rel}:{i}\n    {line}\n    -> {note}\n")
    print("summary:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 1 if counts.get("VIOLATION", 0) else 0


if __name__ == "__main__":
    sys.exit(main())
