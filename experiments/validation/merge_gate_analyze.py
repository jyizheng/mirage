#!/usr/bin/env python3
"""Gate analysis for merge-head runs (merge_gate_run.sh).

Hard gates (exit nonzero on failure):
  1. seeded run1 == run2 bitwise (token ids + logprob bit patterns)
     across a full server restart;
  2. greedy run1 == run2 bitwise across a full server restart;
  3. all 16 greedy group members bitwise match greedy run1;
  4. no spike victims (logprob < -20) in the seeded run.

Informational (never fails): whether the new trajectories differ from the
committed references_v2 JSONs (--refs). Expected to differ after a
numerics-changing merge; the result is recorded in RESULTS.txt so the
references_v2 update commit can cite it.
"""
import argparse
import json
import struct

SPIKE_THRESHOLD = -20.0


def load_choices(path):
    with open(path) as f:
        return json.load(f)["response"]["choices"]


def bits(lps):
    return [struct.pack("<f", x) for x in lps]


def traj(choice):
    lp = choice.get("logprobs")
    lps = lp["token_logprobs"] if lp else []
    return (choice["token_ids"], bits(lps))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--refs", required=True,
                        help="committed references_v2 directory")
    args = parser.parse_args()
    d = args.dir

    seeded1 = load_choices(f"{d}/seeded_run1.json")[0]
    seeded2 = load_choices(f"{d}/seeded_run2.json")[0]
    greedy1 = load_choices(f"{d}/greedy_run1.json")[0]
    greedy2 = load_choices(f"{d}/greedy_run2.json")[0]
    group = load_choices(f"{d}/group16.json")

    results = []
    ok = True

    def gate(name, cond, detail=""):
        nonlocal ok
        results.append(f"gate {name}: {'PASS' if cond else 'FAIL'} {detail}")
        ok = ok and cond

    gate("1_seeded_bitwise_restart", traj(seeded1) == traj(seeded2),
         f"({len(seeded1['token_ids'])} tokens)")
    gate("2_greedy_bitwise_restart", traj(greedy1) == traj(greedy2),
         f"({len(greedy1['token_ids'])} tokens)")
    mismatches = [i for i, c in enumerate(group) if traj(c) != traj(greedy1)]
    gate("3_group16_matches_greedy", len(group) == 16 and not mismatches,
         f"(members={len(group)}, mismatches={mismatches})")
    lp = seeded1.get("logprobs")
    victims = [x for x in (lp["token_logprobs"] if lp else []) if
               x < SPIKE_THRESHOLD]
    gate("4_no_seeded_spikes", not victims, f"(victims={len(victims)})")
    # Deterministic garbage passes bitwise gates: also require that no
    # run ever emits a pad token (id >= true vocab size). Caught the
    # upstream #755 -1e4 pad regression (2026-08-27).
    vocab = 151936  # Qwen3 true vocab (padded to 153600)
    pads = {name: sum(1 for t in run["token_ids"] if t >= vocab)
            for name, run in (("seeded1", seeded1), ("seeded2", seeded2),
                              ("greedy1", greedy1), ("greedy2", greedy2))}
    gate("5_no_pad_tokens", not any(pads.values()), f"({pads})")

    for name, run in (("seeded_1p7b", seeded1), ("greedy_1p7b", greedy1)):
        try:
            old = load_choices(f"{args.refs}/{name}.json")[0]
            changed = traj(old) != traj(run)
            results.append(f"info {name}_vs_committed_refs: "
                           f"{'CHANGED' if changed else 'UNCHANGED'}")
        except FileNotFoundError:
            results.append(f"info {name}_vs_committed_refs: NO_REF")

    with open(f"{d}/RESULTS.txt", "w") as f:
        f.write("\n".join(results) + "\n")
    print("\n".join(results))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
