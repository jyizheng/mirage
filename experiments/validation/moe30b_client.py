#!/usr/bin/env python3
"""MoE-online gate client: one fixed greedy/seeded request against a running
mirage engine server (single completion or group_completions), saved with
full metadata plus wall time for a rough tokens/s figure.

Same fixed prompt as refv2_client.py (GSM8K train[0] + e52 instruction
suffix, templated with enable_thinking=False) so the MoE gates are directly
comparable with the dense references_v2 runs.
"""
import argparse
import json
import time
import urllib.request

from transformers import AutoTokenizer

QUESTION = (
    "Natalia sold clips to 48 of her friends in April, and then she sold "
    "half as many clips in May. How many clips did Natalia sell altogether "
    "in April and May?\n"
    "Think briefly, then give the final numeric answer after '####'."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--group-size", type=int, default=0,
                        help="0 = /v1/completions, N>0 = group_completions")
    parser.add_argument("--commit", default="", help="engine commit, recorded")
    parser.add_argument("--mode", default="", help="seeded|greedy, recorded")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": QUESTION}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)

    if args.group_size > 0:
        url = f"http://127.0.0.1:{args.port}/v1/group_completions"
        body = {"prompt": text, "group_size": args.group_size,
                "use_template": False}
    else:
        url = f"http://127.0.0.1:{args.port}/v1/completions"
        body = {"prompt": text, "use_template": False}

    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.monotonic()
    response = json.load(urllib.request.urlopen(req, timeout=1800))
    elapsed = time.monotonic() - t0

    record = {
        "commit": args.commit,
        "mode": args.mode,
        "model": args.model,
        "prompt": text,
        "request": body,
        "elapsed_s": elapsed,
        "response": response,
    }
    with open(args.output, "w") as f:
        json.dump(record, f, indent=1)
    gen = sum(len(c["token_ids"]) for c in response["choices"])
    print(f"saved {args.output}: choices={len(response['choices'])} "
          f"gen_tokens={gen} elapsed={elapsed:.1f}s "
          f"({gen / elapsed:.1f} tok/s)")


if __name__ == "__main__":
    main()
