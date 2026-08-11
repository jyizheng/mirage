#!/usr/bin/env python3
"""Gate analysis for the Gumbel-spike clamp (references_v2 regeneration).

Reads the six runs produced by refv2_run.sh from --dir and checks:
  1. pre-fix seeded run has spike victims (logprob < -20), post-fix has 0;
  2. post-fix seeded run is bitwise repeatable across server restarts;
  3. post-fix seeded trajectory differs from pre-fix (proves the JIT
     actually recompiled the clamp in);
  4. greedy output is bitwise UNCHANGED pre->post (greedy never draws the
     uniform);
  5. all 16 greedy group members bitwise match the single greedy run
     (shared-prefix path unaffected).
Exit 0 iff every gate passes. Writes RESULTS.txt next to the inputs.
"""
import argparse
import json
import sys

SPIKE_THRESHOLD = -20.0


def load_choices(dirpath, name):
    with open(f"{dirpath}/{name}.json") as f:
        return json.load(f)["response"]["choices"]


def logprobs(choice):
    lp = choice.get("logprobs")
    return lp["token_logprobs"] if lp else None


def spikes(choice):
    lp = logprobs(choice)
    toks = choice["token_ids"]
    return [(i, toks[i], lp[i]) for i in range(len(lp))
            if lp[i] < SPIKE_THRESHOLD]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="/workspace/refv2")
    args = parser.parse_args()
    d = args.dir

    pre = load_choices(d, "pre_seeded")[0]
    run1 = load_choices(d, "post_seeded_run1")[0]
    run2 = load_choices(d, "post_seeded_run2")[0]
    pre_g = load_choices(d, "pre_greedy")[0]
    post_g = load_choices(d, "post_greedy")[0]
    group = load_choices(d, "post_group16")

    lines = []
    gates = {}

    sp_pre, sp_post = spikes(pre), spikes(run1)
    lines.append(f"pre-fix seeded gen_len={len(pre['token_ids'])} "
                 f"spike victims (logprob<{SPIKE_THRESHOLD}): {len(sp_pre)}")
    for i, t, l in sp_pre:
        lines.append(f"  pre spike: pos={i} token={t} logprob={l:.3f}")
    lines.append(f"post-fix seeded gen_len={len(run1['token_ids'])} "
                 f"spike victims: {len(sp_post)} {sp_post}")
    gates["pre_has_spikes"] = len(sp_pre) > 0
    gates["post_zero_spikes"] = len(sp_post) == 0

    gates["seeded_repeatable_tokens"] = (
        run1["token_ids"] == run2["token_ids"])
    gates["seeded_repeatable_logprobs"] = (
        logprobs(run1) == logprobs(run2))
    gates["seeded_trajectory_changed"] = (
        run1["token_ids"] != pre["token_ids"])

    gates["greedy_tokens_unchanged"] = (
        pre_g["token_ids"] == post_g["token_ids"])
    gates["greedy_logprobs_unchanged"] = (
        logprobs(pre_g) == logprobs(post_g))

    member_ok = [c["token_ids"] == post_g["token_ids"]
                 and logprobs(c) == logprobs(post_g) for c in group]
    lines.append(f"group16 members matching single greedy: "
                 f"{sum(member_ok)}/{len(member_ok)}")
    gates["group16_all_match"] = (
        len(member_ok) == 16 and all(member_ok))

    for k, v in gates.items():
        lines.append(f"gate {k}: {'PASS' if v else 'FAIL'}")
    ok = all(gates.values())
    lines.append("ALL GATES: " + ("PASS" if ok else "FAIL"))

    report = "\n".join(lines) + "\n"
    with open(f"{d}/RESULTS.txt", "w") as f:
        f.write(report)
    print(report, end="")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
