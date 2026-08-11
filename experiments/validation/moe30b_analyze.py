#!/usr/bin/env python3
"""Gate analysis for the Qwen3-30B-A3B MoE online determinism suite.

Reads the runs produced by moe30b_run.sh from --dir and checks:
  (i)   greedy /v1/completions is bitwise repeatable: run1 == run2 on one
        server process AND == run3 on a fresh server process;
  (ii)  seeded (--sampling-seed 42) sampling is bitwise repeatable across
        server restarts;
  (iii) all 16 greedy group_completions members bitwise match the single
        greedy reference;
  (iv)  batch-composition invariance: the target request's tokens and
        logprobs are bitwise unchanged when it runs alongside 3 different
        concurrent requests vs alone;
  (vi)  dense Qwen3-1.7B greedy is bitwise repeatable on the same build
        (regression guard for the shared kernels).
Gate (v), rescore == rollout, is a separate in-process script
(moe30b_rescore.py); its verdict is folded in by the driver.

Bitwise: tokens are ints; logprobs travel as JSON floats, and Python's
repr round-trip is exact for float64, so list equality == bit equality.
Exit 0 iff every gate passes.  Writes RESULTS.txt next to the inputs.
"""
import argparse
import json
import sys


def load(dirpath, name):
    with open(f"{dirpath}/{name}.json") as f:
        return json.load(f)


def choices(rec):
    return rec["response"]["choices"]


def logprobs(choice):
    lp = choice.get("logprobs")
    return lp["token_logprobs"] if lp else None


def same(a, b):
    return (a["token_ids"] == b["token_ids"]
            and logprobs(a) == logprobs(b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="/workspace/moe30b")
    args = parser.parse_args()
    d = args.dir

    g1 = choices(load(d, "moe_greedy_run1"))[0]
    g2 = choices(load(d, "moe_greedy_run2"))[0]
    g3 = choices(load(d, "moe_greedy_run3"))[0]
    s1 = choices(load(d, "moe_seeded_run1"))[0]
    s2 = choices(load(d, "moe_seeded_run2"))[0]
    group = choices(load(d, "moe_group16"))
    solo = choices(load(d, "moe_batch_solo"))[0]
    conc = choices(load(d, "moe_batch_concurrent"))[0]
    d1 = choices(load(d, "dense_greedy_run1"))[0]
    d2 = choices(load(d, "dense_greedy_run2"))[0]

    lines = []
    gates = {}

    lines.append(f"greedy gen_len={len(g1['token_ids'])} "
                 f"seeded gen_len={len(s1['token_ids'])} "
                 f"dense gen_len={len(d1['token_ids'])}")

    gates["i_greedy_rerun_same_server"] = same(g1, g2)
    gates["i_greedy_rerun_fresh_server"] = same(g1, g3)
    gates["ii_seeded_rerun_across_restart"] = same(s1, s2)
    member_ok = [same(c, g1) for c in group]
    lines.append(f"group16 members matching single greedy: "
                 f"{sum(member_ok)}/{len(member_ok)}")
    gates["iii_group16_matches_reference"] = (
        len(member_ok) == 16 and all(member_ok))
    gates["iv_batch_composition_invariant"] = same(solo, conc)
    gates["vi_dense_1p7b_greedy_rerun"] = same(d1, d2)

    # seeded and greedy must actually differ, otherwise the seeded gate
    # silently degenerated into the greedy one
    gates["sanity_seeded_differs_from_greedy"] = (
        s1["token_ids"] != g1["token_ids"])

    for k, v in gates.items():
        lines.append(f"gate {k}: {'PASS' if v else 'FAIL'}")
    ok = all(gates.values())
    lines.append("HTTP GATES: " + ("PASS" if ok else "FAIL"))

    report = "\n".join(lines) + "\n"
    with open(f"{d}/RESULTS.txt", "w") as f:
        f.write(report)
    print(report, end="")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
