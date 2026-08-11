#!/usr/bin/env python3
"""references_v2 client: one fixed seeded/greedy request against a running
mirage engine server, saved with full metadata.

Used to (re)generate experiments/data/references_v2 after the Gumbel-spike
clamp (sampling.cuh uniform2gumbel) changed every seeded trajectory. The
prompt is embedded verbatim so references regenerate without dataset access
(it is GSM8K train[0] plus the e52 instruction suffix, templated with
enable_thinking=False, matching the e52/e53 request shape).
"""
import argparse
import json
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
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
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
    response = json.load(urllib.request.urlopen(req, timeout=900))

    record = {
        "commit": args.commit,
        "mode": args.mode,
        "model": args.model,
        "prompt": text,
        "request": body,
        "response": response,
    }
    with open(args.output, "w") as f:
        json.dump(record, f, indent=1)
    choice = response["choices"][0]
    print(f"saved {args.output}: choices={len(response['choices'])} "
          f"gen_tokens={len(choice['token_ids'])} "
          f"logprobs={'logprobs' in choice}")


if __name__ == "__main__":
    main()
