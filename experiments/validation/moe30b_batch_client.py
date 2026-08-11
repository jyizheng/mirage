#!/usr/bin/env python3
"""Gate (iv): batch-composition invariance for the MoE online path.

Phase ``solo``:  submit the fixed target request against an idle server.
Phase ``batch``: start three different decoy requests, wait for them to be
mid-decode, then submit the same target request -- its prefill chunks and
every decode step now share the batch with the decoys.

Proposition-1 condition (a) says the target's tokens and captured logprobs
must be bitwise identical in both phases (routing, expert GEMM tiling and
the combine are all batch-slot-indexed, never compacted by batch content).
"""
import argparse
import json
import threading
import time
import urllib.request

from transformers import AutoTokenizer

TARGET_QUESTION = (
    "Natalia sold clips to 48 of her friends in April, and then she sold "
    "half as many clips in May. How many clips did Natalia sell altogether "
    "in April and May?\n"
    "Think briefly, then give the final numeric answer after '####'."
)
DECOY_QUESTIONS = [
    "Explain the difference between TCP and UDP in two sentences.",
    "Write a haiku about mountains in early winter.",
    "List three prime numbers greater than 100 and explain briefly why "
    "each is prime.",
]


def post_completion(port, text, timeout=1800):
    body = {"prompt": text, "use_template": False}
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=["solo", "batch"], required=True)
    parser.add_argument("--commit", default="", help="engine commit, recorded")
    parser.add_argument("--decoy-lead", type=float, default=3.0,
                        help="seconds decoys run before the target submits")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def templ(q):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=False)

    target_text = templ(TARGET_QUESTION)

    decoy_results = [None] * len(DECOY_QUESTIONS)
    threads = []
    if args.phase == "batch":
        def run_decoy(i, text):
            decoy_results[i] = post_completion(args.port, text)

        for i, q in enumerate(DECOY_QUESTIONS):
            t = threading.Thread(target=run_decoy, args=(i, templ(q)))
            t.start()
            threads.append(t)
        time.sleep(args.decoy_lead)

    t0 = time.monotonic()
    target_response = post_completion(args.port, target_text)
    elapsed = time.monotonic() - t0
    for t in threads:
        t.join()

    record = {
        "commit": args.commit,
        "model": args.model,
        "phase": args.phase,
        "prompt": target_text,
        "elapsed_s": elapsed,
        "response": target_response,
        "decoy_gen_tokens": [
            len(r["choices"][0]["token_ids"]) if r else None
            for r in decoy_results],
    }
    with open(args.output, "w") as f:
        json.dump(record, f, indent=1)
    choice = target_response["choices"][0]
    print(f"saved {args.output} [{args.phase}]: "
          f"gen_tokens={len(choice['token_ids'])} elapsed={elapsed:.1f}s "
          f"decoys={record['decoy_gen_tokens']}")


if __name__ == "__main__":
    main()
