# E20-cross: raw per-position delta_t between an MPK rollout ref
# (token_ids + probs) and SGLang prefill rescoring. Same math as e14 but
# dumps signed raw deltas for the histogram.
import json
import math
import sys
import urllib.request

ref = json.load(open(sys.argv[1]))
tag = sys.argv[2]
ids = ref["token_ids"]
p0 = ref["prompt_length"]

body = json.dumps({
    "input_ids": ids,
    "sampling_params": {"max_new_tokens": 0, "temperature": 0},
    "return_logprob": True,
    "logprob_start_len": p0 - 1,
}).encode()
req = urllib.request.Request("http://127.0.0.1:8322/generate", data=body,
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=600) as r:
    out = json.loads(r.read())

itl = out["meta_info"]["input_token_logprobs"]
deltas = []
for j, entry in enumerate(itl):
    lp_sgl, tok = entry[0], entry[1]
    t = p0 - 1 + j
    if t >= len(ids) or lp_sgl is None:
        continue
    assert tok == ids[t]
    p_mpk = ref["probs"][t - 1]
    if p_mpk <= 0:
        continue
    deltas.append(math.log(p_mpk) - lp_sgl)

ad = sorted(abs(d) for d in deltas)
n = len(ad)
print(f"cross {tag}: n={n} max={ad[-1]:.6g} mean={sum(ad)/n:.6g}")
json.dump({"tag": tag, "deltas": deltas},
          open(f"/tmp/e20_{tag}.json", "w"))
