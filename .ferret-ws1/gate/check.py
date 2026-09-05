#!/usr/bin/env python3
"""Ferret gate: frozen correctness+perf gate for MPK decode GEMM candidates.

Runs ALL cases against a built candidate extension and emits exactly one line

  GATE_RESULT {"pass": bool, "bitwise": {...}, "perf": {...},
               "first_failing_case": name-or-none}

Correctness (BITWISE): candidate outputs must be bitwise identical
(torch.equal on raw bf16 tensors) to the goldens produced by the CURRENT
in-tree kernel. For the split-K family BOTH the partials buffer and the
combined output are checked; a candidate that changes the partial format
FAILS. Each case runs twice with different output prefills (determinism +
full-output-coverage), and every shape family is also cross-checked against
the frozen reference extension on FRESH random inputs drawn at check time
(anti-replay: goldens cannot be memorized).

Perf: CUDA-event timing, 200 warmup + 1000 iters (best of 3 rounds), us/iter,
compared against gate/baseline_perf.json. Target (see perf_spec.md):
geomean speedup >= 1.2x over the M=16 LINEAR cases, no perf case regressing
more than 5%.

Candidate interface: a torch extension module exposing
  linear(x, w, out), splitk_partial(x, w, partials),
  splitk_reduce(partials, residual, out)
built from the frozen harness wrapper (../harness) with MIRAGE_ROOT pointing
at the candidate source tree and GATE_EXT_NAME=<candidate module name>.

Usage:
  CUDA_VISIBLE_DEVICES=3 python3 check.py --module candidate_kernels \
      --module-path /path/to/candidate/build
"""

import argparse
import hashlib
import json
import math
import os
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

import torch  # noqa: E402

GATE_DIR = os.path.dirname(os.path.abspath(__file__))
WS_ROOT = os.path.dirname(GATE_DIR)
sys.path.insert(0, GATE_DIR)
import reference as ref  # noqa: E402

BASELINE_PATH = os.path.join(GATE_DIR, "baseline_perf.json")
HASH_FILE = os.path.join(WS_ROOT, "gate.sha256")
WARMUP = 200
ITERS = 1000
ROUNDS = 3
GEOMEAN_TARGET = 1.20
MAX_REGRESSION = 1.05


def verify_gate_integrity():
    """If the freeze manifest exists, every file under gate/ must match it
    exactly (no edits, no additions, no deletions). __pycache__ is ignored."""
    if not os.path.isfile(HASH_FILE):
        return None  # pre-freeze (baseline generation)
    want = {}
    with open(HASH_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            digest, path = line.split(None, 1)
            want[path.strip()] = digest
    found = {}
    for root, dirs, files in os.walk(GATE_DIR):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fn in files:
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, WS_ROOT)
            h = hashlib.sha256()
            with open(full, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            found[rel] = h.hexdigest()
    if want != found:
        missing = sorted(set(want) - set(found))
        extra = sorted(set(found) - set(want))
        changed = sorted(
            k for k in set(want) & set(found) if want[k] != found[k]
        )
        return (
            f"gate integrity FAIL: missing={missing} extra={extra} "
            f"changed={changed}"
        )
    return "ok"


def bitwise_check(mod, case, golden):
    """Two prefill-differing runs, both must equal golden on every tensor."""
    inputs = ref.make_inputs(case)
    for prefill in ref.PREFILLS:
        got = ref.run_case(mod, case, inputs, prefill=prefill)
        if set(got) != set(golden):
            return False, f"output keys {sorted(got)} != {sorted(golden)}"
        for key in golden:
            g = golden[key].cuda()
            if got[key].shape != g.shape or got[key].dtype != g.dtype:
                return False, f"{key}: shape/dtype mismatch"
            if not torch.equal(got[key], g):
                n_bad = (
                    (got[key].view(torch.int16) != g.view(torch.int16))
                    .sum()
                    .item()
                )
                return (
                    False,
                    f"{key}: {n_bad}/{g.numel()} elements differ bitwise "
                    f"(prefill={prefill})",
                )
    return True, "ok"


def fresh_cross_check(mod, refmod, case, salt):
    """Fresh random inputs at check time: candidate must match the frozen
    reference extension bitwise. Defeats golden-replay candidates."""
    inputs = ref.make_inputs(case, seed_salt=salt)
    got = ref.run_case(mod, case, inputs, prefill=0.0)
    want = ref.run_case(refmod, case, inputs, prefill=0.0)
    for key in want:
        if key not in got or not torch.equal(got[key], want[key]):
            return False, f"{key}: differs from reference on fresh inputs"
    return True, "ok"


def perf_call(mod, case, inputs, out_bufs):
    if case["kind"] == "linear":
        mod.linear(inputs["x"], inputs["w"], out_bufs["out"])
    elif case["kind"] == "splitk":
        mod.splitk_partial(inputs["x"], inputs["w"], out_bufs["partials"])
    elif case["kind"] == "reduce":
        mod.splitk_reduce(
            inputs["partials"], inputs["residual"], out_bufs["out"]
        )


def measure_case(mod, case):
    dev = "cuda"
    inputs = ref.make_inputs(case)
    m, n = case["m"], case["n"]
    out_bufs = {
        "out": torch.empty(m, n, dtype=torch.bfloat16, device=dev),
        "partials": torch.empty(
            ref.NUM_SPLITS * m, n, dtype=torch.bfloat16, device=dev
        ),
    }
    for _ in range(WARMUP):
        perf_call(mod, case, inputs, out_bufs)
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(ROUNDS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(ITERS):
            perf_call(mod, case, inputs, out_bufs)
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) * 1000.0 / ITERS)  # us
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--module-path", default=None)
    ap.add_argument(
        "--write-baseline",
        action="store_true",
        help="record baseline_perf.json from this module (gate author only)",
    )
    ap.add_argument("--skip-perf", action="store_true")
    args = ap.parse_args()

    result = {
        "pass": False,
        "bitwise": {},
        "perf": {},
        "first_failing_case": None,
    }
    failures = []

    def fail(name, why):
        failures.append((name, why))
        if result["first_failing_case"] is None:
            result["first_failing_case"] = name
        print(f"[FAIL] {name}: {why}", file=sys.stderr)

    integrity = verify_gate_integrity()
    if integrity not in (None, "ok"):
        result["first_failing_case"] = "gate_integrity"
        result["bitwise"]["gate_integrity"] = False
        print("GATE_RESULT " + json.dumps(result))
        print(integrity, file=sys.stderr)
        sys.exit(1)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    refext_dir = os.path.join(GATE_DIR, "refext")
    refmod = ref.load_module("gate_ref_kernels", refext_dir)
    if args.module == "gate_ref_kernels":
        mod = refmod
    else:
        mod = ref.load_module(args.module, args.module_path)

    # --- bitwise vs golden ---
    for case in ref.all_cases():
        name = case["name"]
        golden_path = os.path.join(ref.GOLDEN_DIR, f"{name}.pt")
        golden = torch.load(golden_path, weights_only=True)
        try:
            ok, why = bitwise_check(mod, case, golden)
        except Exception as e:  # candidate crashed / rejected shape
            ok, why = False, f"exception: {e}"
        result["bitwise"][name] = ok
        if not ok:
            fail(name, why)

    # --- fresh-input cross-check vs frozen reference extension ---
    salt = ":fresh:" + os.urandom(8).hex()
    for case in ref.all_cases():
        if case["dist"] != "normal":
            continue
        name = case["name"] + "_fresh"
        try:
            ok, why = fresh_cross_check(mod, refmod, case, salt)
        except Exception as e:
            ok, why = False, f"exception: {e}"
        result["bitwise"][name] = ok
        if not ok:
            fail(name, why)

    bitwise_pass = all(result["bitwise"].values())

    # --- perf ---
    perf_pass = True
    if not args.skip_perf:
        cases_by_name = {c["name"]: c for c in ref.all_cases()}
        timings = {}
        for name in ref.perf_case_names():
            timings[name] = measure_case(mod, cases_by_name[name])
        if args.write_baseline:
            with open(BASELINE_PATH, "w") as f:
                json.dump(
                    {
                        "note": "us/iter, current in-tree kernel via frozen "
                        "harness; CUDA events, 200 warmup + 1000 iters, "
                        "best of 3 rounds; splitk_* times the full 8-launch "
                        "partial pass",
                        "device": torch.cuda.get_device_name(0),
                        "us": timings,
                    },
                    f,
                    indent=1,
                    sort_keys=True,
                )
            result["perf"] = {k: round(v, 3) for k, v in timings.items()}
        else:
            with open(BASELINE_PATH) as f:
                baseline = json.load(f)["us"]
            speedups = []
            worst_reg = 0.0
            for name, us in timings.items():
                base = baseline[name]
                result["perf"][name] = {
                    "us": round(us, 3),
                    "baseline_us": round(base, 3),
                    "speedup": round(base / us, 4),
                }
                worst_reg = max(worst_reg, us / base)
                if name in ref.m16_linear_perf_names():
                    speedups.append(base / us)
            geomean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
            perf_pass = geomean >= GEOMEAN_TARGET and worst_reg <= MAX_REGRESSION
            result["perf"]["_summary"] = {
                "geomean_m16_linear_speedup": round(geomean, 4),
                "geomean_target": GEOMEAN_TARGET,
                "worst_case_slowdown": round(worst_reg, 4),
                "max_allowed_slowdown": MAX_REGRESSION,
                "perf_pass": perf_pass,
            }
            if not perf_pass:
                fail("_perf", f"geomean={geomean:.4f} worst_reg={worst_reg:.4f}")

    result["pass"] = bool(bitwise_pass and perf_pass)
    print("GATE_RESULT " + json.dumps(result))
    sys.exit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
