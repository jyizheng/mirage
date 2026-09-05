#!/usr/bin/env python3
"""Ferret gate: reference/golden generation for the Mirage MPK decode GEMM gate.

Regenerates the fixed seeded inputs and the golden outputs from the CURRENT
(unmodified) in-tree kernels:
  - kernel::linear_sm100_mpk_task_impl  (TASK_LINEAR_SM100 and, with
    SplitK=false + per-split K, TASK_SPLITK_PARTIAL_LINEAR_SM100)
  - kernel::splitk_reduce_task_impl     (TASK_SPLITK_REDUCE_SM100)

Golden = the exact bf16 bits produced by the current kernel. Any candidate
must reproduce these bits exactly (torch.equal); see check.py.

This file is also the shared case library imported by check.py. Do not edit:
the gate directory is hash-locked (see ../gate.sha256).

Usage (on the pod, from /root/.cache/ferret-ws1/gate):
  CUDA_VISIBLE_DEVICES=3 python3 reference.py \
      --module gate_ref_kernels --module-path ../harness
"""

import argparse
import importlib
import json
import os
import sys
import zlib

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

import torch  # noqa: E402

GATE_DIR = os.path.dirname(os.path.abspath(__file__))
GOLDEN_DIR = os.path.join(GATE_DIR, "golden")
NUM_SPLITS = 8

# Input distributions: non-trivial randn plus denormal-ish small values and
# large magnitudes to exercise rounding paths.
DISTS = {"normal": 1.0, "small": 1e-20, "large": 1e4}

# Prefill sentinels: every case is run twice with different output-buffer
# prefill values; identical results prove run-to-run bitwise determinism AND
# that the kernel writes every checked output location.
PREFILLS = (0.0, -777.0)


def case_seed(tag):
    return zlib.crc32(tag.encode("utf-8")) & 0x7FFFFFFF


def gen_bf16(shape, tag, scale):
    """Deterministic bf16 tensor. Generated on CPU so the bits do not depend
    on GPU model or CUDA RNG implementation."""
    g = torch.Generator(device="cpu").manual_seed(case_seed(tag))
    t = torch.randn(shape, generator=g, dtype=torch.float32) * scale
    return t.to(torch.bfloat16)


def all_cases():
    """Production Qwen3-1.7B decode matrix; M = max_num_batched_tokens."""
    cases = []
    for m in (8, 16):
        for n, label in ((64, "qkv"), (128, "gateup"), (256, "lmhead")):
            for dist in DISTS:
                cases.append(
                    {
                        "name": f"linear_m{m}_n{n}_k2048_{dist}",
                        "kind": "linear",
                        "family": label,
                        "m": m,
                        "n": n,
                        "k": 2048,
                        "dist": dist,
                    }
                )
        for ks, kf, label in ((256, 2048, "o_proj"), (768, 6144, "down")):
            for dist in DISTS:
                cases.append(
                    {
                        "name": f"splitk_m{m}_ks{ks}_{dist}",
                        "kind": "splitk",
                        "family": label,
                        "m": m,
                        "n": 128,
                        "ks": ks,
                        "kf": kf,
                        "dist": dist,
                    }
                )
        for dist in DISTS:
            cases.append(
                {
                    "name": f"reduce_m{m}_{dist}",
                    "kind": "reduce",
                    "family": "reduce",
                    "m": m,
                    "n": 128,
                    "dist": dist,
                }
            )
    return cases


def perf_case_names():
    return [c["name"] for c in all_cases() if c["dist"] == "normal"]


def m16_linear_perf_names():
    return [
        c["name"]
        for c in all_cases()
        if c["dist"] == "normal" and c["kind"] == "linear" and c["m"] == 16
    ]


def make_inputs(case, device="cuda", seed_salt=""):
    """Inputs for a case. seed_salt="" gives the FIXED golden inputs;
    check.py passes a random salt for fresh-input cross-checks."""
    scale = DISTS[case["dist"]]
    tag = case["name"] + seed_salt
    if case["kind"] == "linear":
        return {
            "x": gen_bf16((case["m"], case["k"]), tag + ":x", scale).to(device),
            "w": gen_bf16((case["n"], case["k"]), tag + ":w", scale).to(device),
        }
    if case["kind"] == "splitk":
        return {
            "x": gen_bf16((case["m"], case["kf"]), tag + ":x", scale).to(device),
            "w": gen_bf16((case["n"], case["kf"]), tag + ":w", scale).to(device),
            "residual": gen_bf16(
                (case["m"], case["n"]), tag + ":res", 1.0
            ).to(device),
        }
    if case["kind"] == "reduce":
        return {
            "partials": gen_bf16(
                (NUM_SPLITS * case["m"], case["n"]), tag + ":p", scale
            ).to(device),
            "residual": gen_bf16(
                (case["m"], case["n"]), tag + ":res", 1.0
            ).to(device),
        }
    raise ValueError(case["kind"])


def run_case(mod, case, inputs, prefill=0.0):
    """Run one case through a built extension module. Returns dict of output
    tensors (still on GPU). For splitk, BOTH the partials and the combined
    output are gate-checked intermediates/outputs."""
    dev = next(iter(inputs.values())).device
    m, n = case["m"], case["n"]
    if case["kind"] == "linear":
        out = torch.full((m, n), prefill, dtype=torch.bfloat16, device=dev)
        mod.linear(inputs["x"], inputs["w"], out)
        torch.cuda.synchronize()
        return {"out": out}
    if case["kind"] == "splitk":
        partials = torch.full(
            (NUM_SPLITS * m, n), prefill, dtype=torch.bfloat16, device=dev
        )
        mod.splitk_partial(inputs["x"], inputs["w"], partials)
        out = torch.full((m, n), prefill, dtype=torch.bfloat16, device=dev)
        mod.splitk_reduce(partials, inputs["residual"], out)
        torch.cuda.synchronize()
        return {"partials": partials, "out": out}
    if case["kind"] == "reduce":
        out = torch.full((m, n), prefill, dtype=torch.bfloat16, device=dev)
        mod.splitk_reduce(inputs["partials"], inputs["residual"], out)
        torch.cuda.synchronize()
        return {"out": out}
    raise ValueError(case["kind"])


def oracle_fp32(case, inputs):
    """Independent fp32 torch oracle (validates harness wiring, NOT bitwise)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    if case["kind"] == "linear":
        return {
            "out": inputs["x"].float() @ inputs["w"].float().t()
        }
    if case["kind"] == "splitk":
        x, w = inputs["x"].float(), inputs["w"].float()
        ks = case["ks"]
        parts = []
        for s in range(NUM_SPLITS):
            parts.append(
                x[:, s * ks : (s + 1) * ks] @ w[:, s * ks : (s + 1) * ks].t()
            )
        partials = torch.cat(parts, dim=0)  # [8*m, n], fp32
        # Combined oracle follows the contract formula: partials are rounded
        # to bf16 (that is what the partial task stores), then summed in
        # ascending split order in fp32 on top of the residual. A full-fp32
        # matmul oracle would NOT be comparable at atol=1e-2 under
        # cancellation, because the real pipeline rounds each partial.
        acc = inputs["residual"].float()
        for s in range(NUM_SPLITS):
            acc = acc + parts[s].to(torch.bfloat16).float()
        return {"partials": partials, "out": acc}
    if case["kind"] == "reduce":
        p = inputs["partials"].float().view(NUM_SPLITS, case["m"], case["n"])
        return {"out": inputs["residual"].float() + p.sum(dim=0)}
    raise ValueError(case["kind"])


def load_module(name, path):
    if path:
        sys.path.insert(0, os.path.abspath(path))
    return importlib.import_module(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="gate_ref_kernels")
    ap.add_argument(
        "--module-path",
        default=None,
        help="dir containing the built reference extension "
        "(default: ./refext if present, else ../harness)",
    )
    args = ap.parse_args()

    module_path = args.module_path
    if module_path is None:
        refext = os.path.join(GATE_DIR, "refext")
        module_path = (
            refext
            if os.path.isdir(refext)
            else os.path.join(GATE_DIR, "..", "harness")
        )
    mod = load_module(args.module, module_path)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.makedirs(GOLDEN_DIR, exist_ok=True)

    validation = {}
    for case in all_cases():
        name = case["name"]
        inputs = make_inputs(case)

        # Two runs with different output prefills: bitwise determinism +
        # full-coverage-of-output proof.
        r0 = run_case(mod, case, inputs, prefill=PREFILLS[0])
        r1 = run_case(mod, case, inputs, prefill=PREFILLS[1])
        for key in r0:
            assert torch.equal(r0[key], r1[key]), (
                f"{name}/{key}: NOT run-to-run bitwise deterministic "
                f"(or output not fully written)"
            )

        # Oracle validation: golden vs independent fp32 torch math.
        # normal/small: strict per-element rtol=1e-2, atol=1e-2.
        # large: per-element rtol is not meaningful under cancellation
        # (terms ~1e8 canceling to ~1e5; fp32 accumulation-order noise vs
        # torch exceeds 1% on isolated elements), so validate with a
        # scale-normalized inf-norm: max|got-ref| <= 1e-2 * max|ref|.
        oracle = oracle_fp32(case, inputs)
        stats = {"deterministic": True}
        for key in r0:
            got = r0[key].float()
            ref = oracle[key]
            if case["dist"] != "large":
                torch.testing.assert_close(got, ref, rtol=1e-2, atol=1e-2)
            abs_err = (got - ref).abs()
            rel_err = abs_err / ref.abs().clamp_min(1e-30)
            scale = max(ref.abs().max().item(), 1e-30)
            inf_ratio = abs_err.max().item() / scale
            assert inf_ratio <= 1e-2, (
                f"{name}/{key}: inf-norm ratio {inf_ratio} > 1e-2"
            )
            stats[f"{key}_max_abs_err"] = abs_err.max().item()
            stats[f"{key}_max_rel_err_at_abs_gt_atol"] = (
                rel_err[abs_err > 1e-2].max().item()
                if (abs_err > 1e-2).any()
                else 0.0
            )
            stats[f"{key}_oracle_max_abs"] = scale
            stats[f"{key}_inf_norm_err_ratio"] = inf_ratio
        validation[name] = stats

        torch.save(
            {k: v.cpu() for k, v in r0.items()},
            os.path.join(GOLDEN_DIR, f"{name}.pt"),
        )
        print(f"[golden] {name}: ok ({', '.join(sorted(r0))})")

    with open(os.path.join(GATE_DIR, "oracle_validation.json"), "w") as f:
        json.dump(
            {
                "note": "golden (current in-tree kernel) vs independent fp32 "
                "torch oracle, rtol=1e-2 atol=1e-2; deterministic = two runs "
                "with different output prefills were bitwise identical",
                "torch": torch.__version__,
                "device": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "cases": validation,
            },
            f,
            indent=1,
            sort_keys=True,
        )
    print(f"[golden] wrote {len(validation)} cases to {GOLDEN_DIR}")


if __name__ == "__main__":
    main()
