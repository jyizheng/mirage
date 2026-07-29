# E20: per-token delta_t histogram data.
# Same-engine TIM for SGLang: sample a trajectory (decode-time logprobs via
# return_logprob), then rescore the identical token sequence via prefill
# (max_new_tokens=0, logprob_start_len) -- exactly the trainer-side
# recompute path. delta_t[k] = lp_decode[k] - lp_rescore[k].
# Dumps raw per-position deltas for the paper figure.
import json
import sys
import urllib.request

PORT = 8322
MODE = sys.argv[1]  # tag for output file
OUT = f"/tmp/e20_{MODE}.json"

SYS = ("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. "
       "You are a helpful assistant.<|im_end|>\n")
QUESTIONS = [
    "Give me a short introduction to large language model.",
    "Natalia sold clips to 48 of her friends in April, and then she sold "
    "half as many clips in May. How many clips did Natalia sell altogether?",
    "Explain the difference between a process and a thread.",
    "Write a short story about a lighthouse keeper who finds a message "
    "in a bottle.",
    "What is the capital of Australia, and why is it not Sydney?",
    "A train travels 60 miles per hour for 2.5 hours. How far does it go?",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "Describe how photosynthesis works at a high level.",
]
MAX_NEW = 512


def post(body):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/generate", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


all_deltas = []
per_prompt = []
for qi, q in enumerate(QUESTIONS):
    prompt = SYS + f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    gen = post({
        "text": prompt,
        "sampling_params": {"temperature": 1.0, "top_p": 1.0, "top_k": -1,
                            "max_new_tokens": MAX_NEW, "ignore_eos": True},
        "return_logprob": True,
    })
    meta = gen["meta_info"]
    out_lp = meta["output_token_logprobs"]  # (logprob, token_id, ...)
    dec_lp = [e[0] for e in out_lp]
    out_ids = [e[1] for e in out_lp]

    # recover prompt ids from input_token_logprobs? not returned without
    # logprob_start_len; instead rescore with text+decoded tokens is lossy.
    # Use input_ids path: tokenize server-side by sending the same text and
    # asking for prompt logprobs is unnecessary -- meta has prompt_tokens
    # count and we can rescore with text=prompt plus token ids appended via
    # input_ids only. So fetch prompt ids with a 1-token probe.
    probe = post({
        "text": prompt,
        "sampling_params": {"temperature": 0, "max_new_tokens": 1},
        "return_logprob": True, "logprob_start_len": 0,
    })
    prompt_ids = [e[1] for e in probe["meta_info"]["input_token_logprobs"]]
    ids = prompt_ids + out_ids
    p0 = len(prompt_ids)

    res = post({
        "input_ids": ids,
        "sampling_params": {"temperature": 0, "max_new_tokens": 0},
        "return_logprob": True,
        "logprob_start_len": p0 - 1,
    })
    itl = res["meta_info"]["input_token_logprobs"]
    deltas = []
    for j, entry in enumerate(itl):
        lp_res, tok = entry[0], entry[1]
        t = p0 - 1 + j
        if lp_res is None or t < p0:
            continue
        k = t - p0
        assert tok == ids[t], f"align {tok}!={ids[t]} @ {t}"
        deltas.append(dec_lp[k] - lp_res)
    all_deltas.extend(deltas)
    nz = sum(1 for d in deltas if d != 0.0)
    per_prompt.append({"q": qi, "n": len(deltas), "nonzero": nz})
    print(f"prompt {qi}: n={len(deltas)} nonzero={nz} "
          f"max|dt|={max(abs(d) for d in deltas):.6g}", flush=True)

ad = sorted(abs(d) for d in all_deltas)
n = len(ad)
nz = sum(1 for d in ad if d != 0.0)
summary = {"mode": MODE, "n": n, "nonzero": nz,
           "max": ad[-1], "mean": sum(ad) / n,
           "p50": ad[n // 2], "p99": ad[int(n * 0.99) - 1]}
print("SUMMARY", json.dumps(summary), flush=True)
json.dump({"summary": summary, "per_prompt": per_prompt,
           "deltas": all_deltas}, open(OUT, "w"))
print("wrote", OUT)
