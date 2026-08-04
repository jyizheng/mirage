#!/usr/bin/env python3
"""E44: trainer -> MPK rollout weight-sync benchmark.

This is the missing systems experiment between the single-node E19 loop and a
miles-class disaggregated engine.  It measures the reusable synchronization
plan introduced in ``mirage.mpk.weight_sync``:

  * ``--synthetic``: no model download; validates the Qwen3 mapping shape,
    padding, TP slicing, and benchmark output schema.
  * ``--target hf-pair``: loads two HF modules and measures a same-name copy
    baseline.
  * ``--target mpk``: builds a real MPK rollout engine and measures copying a
    trainer state dict into the MPK attached tensors that the persistent
    kernel reads.

Output is one JSON record containing bytes, timing distribution, and plan
coverage.  The transport for disaggregated deployment can wrap the same fitted
source slices; this benchmark pins the mapping and copy cost first.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

try:
    import torch
except ImportError as exc:  # pragma: no cover - cluster dependency
    raise SystemExit("E44 requires torch in the experiment environment") from exc

from mirage.mpk.weight_sync import (
    build_name_matching_sync_plan,
    build_qwen3_mpk_sync_plan,
    tensor_map,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--target", choices=("synthetic", "hf-pair", "mpk"),
                        default="synthetic")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--intermediate", type=int, default=128)
    parser.add_argument("--vocab", type=int, default=512)
    parser.add_argument("--padded-vocab", type=int, default=640)
    parser.add_argument("--max-num-batched-requests", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--max-num-pages", type=int, default=16)
    parser.add_argument("--page-size", type=int, default=4096)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    sources, targets, plan_name = build_inputs(args)
    if args.target == "hf-pair":
        plan = build_name_matching_sync_plan(
            sources, targets, rank=args.rank, world_size=args.world_size)
    else:
        plan = build_qwen3_mpk_sync_plan(
            sources, targets, rank=args.rank, world_size=args.world_size,
            num_layers=args.layers if args.target == "synthetic" else None)

    times = []
    last_report = None
    for it in range(args.warmup + args.steps):
        sync_device(args.device)
        t0 = time.perf_counter()
        last_report = plan.sync(sources, targets, strict=args.strict)
        sync_device(args.device)
        dt = time.perf_counter() - t0
        if it >= args.warmup:
            times.append(dt)

    assert last_report is not None
    mean_s = statistics.mean(times) if times else 0.0
    bandwidth = (
        (last_report.bytes / float(1 << 30)) / mean_s if mean_s > 0 else 0.0
    )
    out = {
        "experiment": "e44_weight_sync_bench",
        "target": args.target,
        "plan": plan.name,
        "input": plan_name,
        "rank": args.rank,
        "world_size": args.world_size,
        "specs": len(plan.specs),
        "synced_tensors": last_report.tensors,
        "sync_bytes": last_report.bytes,
        "sync_gib": round(last_report.gib, 6),
        "missing_sources": list(last_report.missing_sources),
        "missing_targets": list(last_report.missing_targets),
        "steps": args.steps,
        "mean_ms": round(mean_s * 1e3, 4),
        "median_ms": round(statistics.median(times) * 1e3, 4),
        "p95_ms": round(percentile(times, 0.95) * 1e3, 4),
        "bandwidth_gib_s": round(bandwidth, 4),
    }
    print(json.dumps(out, indent=2))


def build_inputs(args):
    if args.target == "synthetic":
        return synthetic_qwen3(args), synthetic_mpk_targets(args), "synthetic-qwen3"

    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16
    trainer = AutoModelForCausalLM.from_pretrained(
        args.model_path or args.model, dtype=dtype).to(args.device)
    sources = tensor_map(trainer)

    if args.target == "hf-pair":
        rollout = AutoModelForCausalLM.from_pretrained(
            args.model_path or args.model, dtype=dtype).to(args.device)
        return sources, tensor_map(rollout), args.model_path or args.model

    from mirage.engine.model_runner import ModelRunner, RunnerConfig

    runner = ModelRunner(
        RunnerConfig(
            model=args.model,
            model_path=args.model_path,
            max_num_batched_requests=args.max_num_batched_requests,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_seq_length=args.max_seq_length,
            max_num_pages=args.max_num_pages,
            page_size=args.page_size,
            capture_logprobs=True,
            deterministic=True,
            output_dir=args.output_dir,
        )
    )
    return sources, tensor_map(runner.mpk), args.model_path or args.model


def synthetic_qwen3(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    src = {
        "model.embed_tokens.weight": torch.randn(
            args.vocab, args.hidden, dtype=dtype, device=device),
        "lm_head.weight": torch.randn(
            args.vocab, args.hidden, dtype=dtype, device=device),
        "model.norm.weight": torch.randn(args.hidden, dtype=dtype, device=device),
    }
    for i in range(args.layers):
        p = f"model.layers.{i}."
        src[p + "input_layernorm.weight"] = torch.randn(
            args.hidden, dtype=dtype, device=device)
        src[p + "post_attention_layernorm.weight"] = torch.randn(
            args.hidden, dtype=dtype, device=device)
        src[p + "self_attn.q_proj.weight"] = torch.randn(
            args.hidden, args.hidden, dtype=dtype, device=device)
        src[p + "self_attn.k_proj.weight"] = torch.randn(
            args.hidden // 4, args.hidden, dtype=dtype, device=device)
        src[p + "self_attn.v_proj.weight"] = torch.randn(
            args.hidden // 4, args.hidden, dtype=dtype, device=device)
        src[p + "self_attn.o_proj.weight"] = torch.randn(
            args.hidden, args.hidden, dtype=dtype, device=device)
        src[p + "self_attn.q_norm.weight"] = torch.randn(
            args.hidden // 4, dtype=dtype, device=device)
        src[p + "self_attn.k_norm.weight"] = torch.randn(
            args.hidden // 4, dtype=dtype, device=device)
        src[p + "mlp.gate_proj.weight"] = torch.randn(
            args.intermediate, args.hidden, dtype=dtype, device=device)
        src[p + "mlp.up_proj.weight"] = torch.randn(
            args.intermediate, args.hidden, dtype=dtype, device=device)
        src[p + "mlp.down_proj.weight"] = torch.randn(
            args.hidden, args.intermediate, dtype=dtype, device=device)
    return src


def synthetic_mpk_targets(args):
    src = synthetic_qwen3(args)
    device = next(iter(src.values())).device
    dtype = next(iter(src.values())).dtype

    def target_like(name: str, tensor: torch.Tensor) -> torch.Tensor:
        shape = list(tensor.shape)
        if args.world_size > 1:
            if any(x in name for x in ("q_proj", "k_proj", "v_proj",
                                       "gate_proj", "up_proj")):
                shape[0] //= args.world_size
            if any(x in name for x in ("o_proj", "down_proj")):
                shape[1] //= args.world_size
        return torch.empty(*shape, dtype=dtype, device=device)

    dst = {
        "embed_tokens": torch.empty_like(src["model.embed_tokens.weight"]),
        "lm_head": torch.empty(
            args.padded_vocab, args.hidden, dtype=dtype, device=device),
        "model_norm_weight": torch.empty_like(src["model.norm.weight"]),
    }
    for i in range(args.layers):
        p = f"model.layers.{i}."
        for target, source in (
            (f"layer_{i}_input_layernorm", p + "input_layernorm.weight"),
            (f"layer_{i}_post_attn_layernorm",
             p + "post_attention_layernorm.weight"),
            (f"layer_{i}_q_proj", p + "self_attn.q_proj.weight"),
            (f"layer_{i}_k_proj", p + "self_attn.k_proj.weight"),
            (f"layer_{i}_v_proj", p + "self_attn.v_proj.weight"),
            (f"layer_{i}_q_norm", p + "self_attn.q_norm.weight"),
            (f"layer_{i}_k_norm", p + "self_attn.k_norm.weight"),
            (f"layer_{i}_o_proj", p + "self_attn.o_proj.weight"),
            (f"layer_{i}_gate_proj", p + "mlp.gate_proj.weight"),
            (f"layer_{i}_up_proj", p + "mlp.up_proj.weight"),
            (f"layer_{i}_down_proj", p + "mlp.down_proj.weight"),
        ):
            dst[target] = target_like(target, src[source])
    return dst


def sync_device(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


if __name__ == "__main__":
    main()
