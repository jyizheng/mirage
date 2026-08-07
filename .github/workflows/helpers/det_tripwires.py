#!/usr/bin/env python3
"""Determinism-critical regression tripwires (no GPU / no build required).

Cheap, low-false-positive greps that guard invariants the deterministic
decode work depends on:

  1. No `torch.cuda.synchronize(` call in python/mirage/engine/.
     Synchronizing the host against the device while the megakernel is
     resident deadlocks the persistent-kernel scheduler. Matches inside
     comments are ignored; an intentional call can be whitelisted by
     putting `# ci-allow-sync` on the same line.

  2. The env hooks `MPK_DET_NUM_SPLITS` and `MPK_LINEAR_TASKS_` must
     still exist under python/mirage/ -- the experiments/ harness sets
     them to control deterministic split-K partitioning, so silently
     renaming/removing them breaks every runner script.

Exit code 0 = all tripwires pass, 1 = violation (details on stdout).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

failures: list[str] = []


def check_no_engine_sync() -> None:
    engine_dir = REPO / "python" / "mirage" / "engine"
    for path in sorted(engine_dir.rglob("*.py")):
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if "ci-allow-sync" in line:
                continue
            # Ignore matches that only appear in a trailing comment.
            # (Crude comment stripping; good enough since '#' inside a
            # string literal on the same line as a sync call is unlikely.)
            code = line.split("#", 1)[0]
            if "torch.cuda.synchronize(" in code:
                failures.append(
                    f"{path.relative_to(REPO)}:{lineno}: "
                    "torch.cuda.synchronize() call in mirage.engine "
                    "(megakernel deadlock rule). If truly intentional, "
                    "append `# ci-allow-sync` to the line."
                )


def check_env_hooks_exist() -> None:
    pkg = REPO / "python" / "mirage"
    text = "\n".join(
        p.read_text(encoding="utf-8", errors="replace")
        for p in sorted(pkg.rglob("*.py"))
    )
    for hook in ("MPK_DET_NUM_SPLITS", "MPK_LINEAR_TASKS_"):
        if hook not in text:
            failures.append(
                f"env hook `{hook}` no longer referenced anywhere under "
                "python/mirage/ -- experiments/runners depend on it."
            )


def main() -> int:
    check_no_engine_sync()
    check_env_hooks_exist()
    if failures:
        print("Determinism tripwire FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All determinism tripwires passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
