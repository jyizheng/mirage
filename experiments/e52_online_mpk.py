#!/usr/bin/env python3
"""Matched-token online MPK rollout benchmark for the E2E speed gate."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
from pathlib import Path
import time
import urllib.request

from datasets import load_dataset
from transformers import AutoTokenizer


def post(port: int, body: dict, timeout: int = 600) -> dict:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--port", type=int, default=8332)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--max-total-tokens", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

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

    def generate_one(text: str, prompt_length: int):
        response = post(args.port, {
            "prompt": text,
            "use_template": False,
        })
        choice = response["choices"][0]
        token_ids = choice["token_ids"]
        expected = args.max_total_tokens - prompt_length
        if len(token_ids) != expected:
            raise RuntimeError(
                f"matched-token generation expected {expected} tokens, "
                f"got {len(token_ids)}"
            )
        logprobs = choice.get("logprobs", {}).get("token_logprobs", [])
        if len(logprobs) != expected or any(x is None for x in logprobs):
            raise RuntimeError("online MPK did not return every rollout logprob")
        return len(token_ids)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pool = futures.ThreadPoolExecutor(max_workers=args.group_size)
    generate_one(prompts[0][0], prompts[0][1])

    with args.output.open("w") as log:
        for step, (text, prompt_length) in enumerate(prompts):
            start = time.perf_counter()
            gen_lens = list(pool.map(
                lambda _: generate_one(text, prompt_length),
                range(args.group_size),
            ))
            rollout_seconds = time.perf_counter() - start
            record = {
                "step": step,
                "t_rollout_s": round(rollout_seconds, 6),
                "gen_lens": gen_lens,
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(record, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
