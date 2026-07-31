# E32 SGLang arm: per-RL-step inference cost in the recompute deployment.
# Each step: generate a GRPO group of G sampled trajectories (concurrent
# requests), then acquire pi_old via the miles-style prefill rescore
# (logprob_start_len) over every trajectory. Times both phases.
import concurrent.futures as cf
import json
import time
import urllib.request

PORT = 8323
STEPS = 20
G = 8
NEW = 200

from datasets import load_dataset
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
ds = load_dataset("openai/gsm8k", "main", split="train")
prompts = []
for row in ds:
    msgs = [{"role": "user",
             "content": row["question"] + "\nThink briefly, then give the "
                        "final numeric answer after '####'."}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True,
                                   enable_thinking=False)
    ids = tok(text).input_ids
    if len(ids) <= 320:
        prompts.append((text, len(ids)))
    if len(prompts) >= STEPS:
        break


def post(body, timeout=600):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/generate", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def gen_one(text):
    out = post({"text": text,
                "sampling_params": {"temperature": 1.0, "top_p": 1.0,
                                    "top_k": -1, "max_new_tokens": NEW},
                "return_logprob": True, "logprob_start_len": 0})
    meta = out["meta_info"]
    prompt_ids = [e[1] for e in meta["input_token_logprobs"]]
    out_ids = [e[1] for e in meta["output_token_logprobs"]]
    return prompt_ids + out_ids, len(prompt_ids)


def rescore_one(ids, p0):
    post({"input_ids": ids,
          "sampling_params": {"temperature": 0, "max_new_tokens": 0},
          "return_logprob": True, "logprob_start_len": p0 - 1})


log = open("/tmp/e38_sgl.jsonl", "w")
pool = cf.ThreadPoolExecutor(max_workers=G)
# warmup
gen_one(prompts[0][0])
for it, (text, plen) in enumerate(prompts):
    t0 = time.time()
    trajs = list(pool.map(lambda _: gen_one(text), range(G)))
    t_gen = time.time() - t0
    t1 = time.time()
    list(pool.map(lambda tr: rescore_one(tr[0], tr[1]), trajs))
    t_rescore = time.time() - t1
    rec = {"step": it, "t_gen_s": round(t_gen, 4),
           "t_rescore_s": round(t_rescore, 4),
           "gen_lens": [len(t[0]) - t[1] for t in trajs]}
    log.write(json.dumps(rec) + "\n")
    log.flush()
    print(rec, flush=True)
print("E32 SGL DONE")
