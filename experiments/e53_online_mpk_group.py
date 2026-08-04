# E53: e52's matched online MPK rollout, but through /v1/group_completions
# -- one call per step, shared-prefix prefill (prompt KV computed once,
# replicated to the group's pages; members admit via initial_step).
# Compare t_rollout_s against e52 at the same server config for the
# prefix-sharing speedup; token/logprob outputs must match e52 bitwise
# under greedy decoding.
import argparse
import json
import time
import urllib.request
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def post(port: int, payload: dict, timeout: float = 900.0) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/group_completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--port", type=int, default=8332)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--max-total-tokens", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dump-first-step", type=Path, default=None,
                        help="Dump step-0 token_ids+logprobs for bitwise "
                        "A/B against the unshared (e52-style) path.")
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
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        prompt_length = len(tokenizer(text).input_ids)
        if prompt_length <= 320:
            prompts.append((text, prompt_length))
        if len(prompts) >= args.steps:
            break

    def generate_group(text: str, prompt_length: int):
        response = post(args.port, {
            "prompt": text,
            "group_size": args.group_size,
            "use_template": False,
        })
        choices = response["choices"]
        expected = args.max_total_tokens - prompt_length
        for c in choices:
            if len(c["token_ids"]) != expected:
                raise RuntimeError(
                    f"matched-token generation expected {expected}, got "
                    f"{len(c['token_ids'])} (member {c['index']})")
            lp = c.get("logprobs", {}).get("token_logprobs", [])
            if len(lp) != expected or any(x is None for x in lp):
                raise RuntimeError("missing rollout logprobs")
        return choices

    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate_group(prompts[0][0], prompts[0][1])  # warmup

    with args.output.open("w") as log:
        for step, (text, prompt_length) in enumerate(prompts):
            start = time.perf_counter()
            choices = generate_group(text, prompt_length)
            rollout_seconds = time.perf_counter() - start
            record = {
                "step": step,
                "t_rollout_s": round(rollout_seconds, 6),
                "gen_lens": [len(c["token_ids"]) for c in choices],
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(record, flush=True)
            if step == 0 and args.dump_first_step:
                json.dump(
                    [{"token_ids": c["token_ids"],
                      "logprobs": c["logprobs"]["token_logprobs"]}
                     for c in choices],
                    args.dump_first_step.open("w"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
