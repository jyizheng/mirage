#!/usr/bin/env python3
"""Matched-token SGLang + miles inference arm for the E2E speed gate."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
from pathlib import Path
import sys
import time
import urllib.request

from datasets import load_dataset
from transformers import AutoTokenizer


def post(port: int, body: dict, timeout: int = 600) -> dict:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--port", type=int, default=8322)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--max-total-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--miles-root", type=Path, default=Path("/workspace/miles"))
    args = parser.parse_args()

    sys.path.insert(0, str(args.miles_root))
    from miles.rollout.generate_utils.prefill_logprobs import (
        _build_prefill_scoring_payload,
    )
    from miles.utils.types import Sample

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    prompts = []
    for row in dataset:
        messages = [{
            "role": "user",
            "content": row["question"] + "\nThink briefly, then give the "
            "final numeric answer after '####'.",
        }]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_length = len(tokenizer(text).input_ids)
        if prompt_length <= 320:
            prompts.append((text, prompt_length))
        if len(prompts) >= args.steps:
            break

    def generate_one(text: str, prompt_length: int, seed: int):
        max_new_tokens = args.max_total_tokens - prompt_length
        output = post(args.port, {
            "text": text,
            "sampling_params": {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
                "sampling_seed": seed,
            },
            "return_logprob": True,
            "logprob_start_len": 0,
        })
        metadata = output["meta_info"]
        prompt_ids = [entry[1] for entry in metadata["input_token_logprobs"]]
        output_ids = [entry[1] for entry in metadata["output_token_logprobs"]]
        if len(output_ids) != max_new_tokens:
            raise RuntimeError(
                f"matched-token generation expected {max_new_tokens} tokens, "
                f"got {len(output_ids)}"
            )
        return prompt_ids + output_ids, len(prompt_ids)

    def rescore_one(trajectory):
        token_ids, prompt_length = trajectory
        sample = Sample(
            tokens=token_ids,
            response_length=len(token_ids) - prompt_length,
        )
        try:
            payload = _build_prefill_scoring_payload(
                args=None, sample=sample, sampling_params={}
            )
        except TypeError:
            payload = _build_prefill_scoring_payload(None, sample, {})
        post(args.port, payload)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pool = futures.ThreadPoolExecutor(max_workers=args.group_size)
    generate_one(prompts[0][0], prompts[0][1], args.seed)

    with args.output.open("w") as log:
        for step, (text, prompt_length) in enumerate(prompts):
            start = time.perf_counter()
            trajectories = list(pool.map(
                lambda request_id: generate_one(
                    text,
                    prompt_length,
                    args.seed + step * args.group_size + request_id,
                ),
                range(args.group_size),
            ))
            generate_seconds = time.perf_counter() - start

            start = time.perf_counter()
            list(pool.map(rescore_one, trajectories))
            rescore_seconds = time.perf_counter() - start
            record = {
                "step": step,
                "t_gen_s": round(generate_seconds, 6),
                "t_rescore_s": round(rescore_seconds, 6),
                "gen_lens": [
                    len(token_ids) - trajectory_prompt_length
                    for token_ids, trajectory_prompt_length in trajectories
                ],
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(record, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
