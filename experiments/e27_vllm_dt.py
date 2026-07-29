# E27: conformance-suite row -- delta_t between MPK rollout logprobs and
# vLLM prompt_logprobs rescoring (the vLLM-trainer deployment pairing).
# Same math as e14/e20_cross but against vLLM offline API.
#
# Usage: vllm-venv/bin/python e27_vllm_dt.py ref.json tag [out.json]
import json
import math
import sys

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

ref = json.load(open(sys.argv[1]))
tag = sys.argv[2]
out_path = sys.argv[3] if len(sys.argv) > 3 else f"/tmp/e27_{tag}.json"
ids = ref["token_ids"]
p0 = ref["prompt_length"]

llm = LLM(model="Qwen/Qwen3-8B", max_model_len=4096,
          gpu_memory_utilization=0.85, enforce_eager=False)
sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)
out = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp)[0]

plp = out.prompt_logprobs  # list, entry per position (None at pos 0)
deltas = []
for t in range(p0, len(ids)):
    entry = plp[t]
    if entry is None:
        continue
    lp_obj = entry.get(ids[t])
    if lp_obj is None:
        continue
    lp_vllm = lp_obj.logprob
    p_mpk = ref["probs"][t - 1]
    if p_mpk <= 0:
        continue
    deltas.append(math.log(p_mpk) - lp_vllm)

ad = sorted(abs(d) for d in deltas)
n = len(ad)
summary = {"tag": tag, "n": n, "max": ad[-1], "mean": sum(ad) / n,
           "p99": ad[int(n * 0.99) - 1]}
print("E27", json.dumps(summary))
json.dump({"summary": summary, "deltas": deltas}, open(out_path, "w"))
