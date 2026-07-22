# Decode-vs-rescore consistency harness (RL zero-logprob-diff, necessary
# condition at token level).
#
# Run 1 decodes a trajectory token-by-token. Runs 2..n replay a prefix of
# that trajectory (prompt + first K generated tokens, fed as raw token ids)
# through chunked prefill, then continue decoding. If the KV cache built by
# prefill is bitwise-identical to the one built by decode — the property an
# RL trainer relies on when rescoring rollout tokens — every continuation
# must reproduce the reference suffix exactly. Each K probes one
# prefill/decode boundary position; disagreement at any position would
# surface as a diverging continuation (greedy decoding amplifies any logits
# mismatch at near-ties).
#
# Usage (on a B200 machine, from demo/qwen3):
#   python rescore_consistency.py --ks 8 16 24 32 48 [--deterministic]
import argparse
import json
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--ks", type=int, nargs="+", default=[8, 16, 24, 32, 48])
parser.add_argument("--full-rescore", action="store_true")
parser.add_argument("--deterministic", action="store_true")
parser.add_argument("--max-new-tokens", type=int, default=96)
parser.add_argument("--workdir", default="/tmp/rescore")
args = parser.parse_args()

import os

os.makedirs(args.workdir, exist_ok=True)
det = ["--deterministic"] if args.deterministic else []


def run_demo(extra, log_name):
    cmd = (
        [sys.executable, "demo.py", "--use-mirage", "--capture-probs"]
        + det
        + ["--max-new-tokens", str(args.max_new_tokens)]
        + extra
    )
    log = os.path.join(args.workdir, log_name)
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        print(f"FAILED ({log_name}): see {log}")
        sys.exit(1)


# Run 1: reference trajectory
ref_file = os.path.join(args.workdir, "ref.json")
run_demo(["--dump-tokens-file", ref_file], "ref.log")
ref = json.load(open(ref_file))
ref_ids, p0 = ref["token_ids"], ref["prompt_length"]
gen_len = len(ref_ids) - p0
print(f"reference: prompt={p0} generated={gen_len}")

# Runs 2..n: replay prompt + K generated tokens as prompt, continue
failures = 0
for k in args.ks:
    if k >= gen_len:
        print(f"K={k}: skipped (>= generated length {gen_len})")
        continue
    prefix = ref_ids[: p0 + k]
    pf = os.path.join(args.workdir, f"prompt_k{k}.json")
    json.dump(prefix, open(pf, "w"))
    of = os.path.join(args.workdir, f"out_k{k}.json")
    run_demo(
        ["--prompt-ids-file", pf, "--dump-tokens-file", of], f"k{k}.log"
    )
    out = json.load(open(of))
    got = out["token_ids"]
    # compare the continuation region (positions p0+k .. end of shorter run)
    n = min(len(ref_ids), len(got)) - (p0 + k)
    ref_cont = ref_ids[p0 + k : p0 + k + n]
    got_cont = got[p0 + k : p0 + k + n]
    if ref_cont == got_cont:
        print(f"K={k}: token MATCH over {n} continuation tokens")
    else:
        first = next(i for i in range(n) if ref_cont[i] != got_cont[i])
        failures += 1
        print(
            f"K={k}: token MISMATCH at continuation offset {first} "
            f"(ref={ref_cont[first]} got={got_cont[first]})"
        )
        continue
    # bitwise probability comparison over the decode region of run-K.
    # buffer indices track the runtime step counter, which advances the same
    # way in both runs for the same absolute position, so slots align.
    if "prob_bits" in ref and "prob_bits" in out:
        rb, gb = ref["prob_bits"], out["prob_bits"]
        m = min(len(rb), len(gb))
        # run-K's decode-region capture starts after its own prompt
        start = p0 + k
        diffs = [
            i for i in range(start, m) if rb[i] != gb[i] and gb[i] != 0
        ]
        checked = sum(1 for i in range(start, m) if gb[i] != 0)
        if not diffs:
            print(f"K={k}: prob BITWISE MATCH over {checked} positions")
        else:
            failures += 1
            i = diffs[0]
            print(
                f"K={k}: prob BITWISE MISMATCH at pos {i} "
                f"(ref_bits={rb[i] & 0xFFFFFFFF:#010x} "
                f"got_bits={gb[i] & 0xFFFFFFFF:#010x}, "
                f"{len(diffs)}/{checked} positions differ)"
            )

# Full rescore: feed the ENTIRE trajectory as prompt; the
# prefill_prob_capture task then computes the teacher-forcing probability
# of every trajectory token during chunked prefill. Compare bitwise against
# the reference run's decode-time capture — this is exactly the
# rollout-vs-rescore logprob comparison an RL trainer performs
# (miles' recompute_logprobs_via_prefill / input_token_logprobs).
if args.full_rescore:
    pf = os.path.join(args.workdir, "prompt_full.json")
    json.dump(ref_ids, open(pf, "w"))
    of = os.path.join(args.workdir, "out_full.json")
    run_demo(["--prompt-ids-file", pf, "--dump-tokens-file", of], "full.log")
    out = json.load(open(of))
    rb, gb = ref["prob_bits"], out["prob_bits"]
    # decode capture in the reference run covers the generated region; find
    # the slot alignment (should be shift 0) and compare bitwise
    lo = p0 + 1
    hi = min(len(rb), len(gb)) - 2
    best = None
    for shift in range(-2, 3):
        m = sum(
            1
            for i in range(lo, hi)
            if rb[i] != 0 and 0 <= i + shift < len(gb) and rb[i] == gb[i + shift]
        )
        if best is None or m > best[1]:
            best = (shift, m)
    shift, matched = best
    total = sum(1 for i in range(lo, hi) if rb[i] != 0)
    mism = [
        i
        for i in range(lo, hi)
        if rb[i] != 0 and rb[i] != gb[i + shift]
    ]
    if not mism and total > 0:
        print(
            f"FULL RESCORE: BITWISE MATCH over {matched}/{total} positions "
            f"(slot shift {shift})"
        )
    else:
        failures += 1
        print(
            f"FULL RESCORE: {len(mism)}/{total} positions differ "
            f"(best shift {shift}); first diff at {mism[0] if mism else '-'}"
        )

print("RESULT:", "ALL MATCH" if failures == 0 else f"{failures} mismatching checks")
sys.exit(1 if failures else 0)
