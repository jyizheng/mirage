# E14: delta_t between MPK rollout logprobs and SGLang prefill rescoring
# (the miles/true-on-policy deployment combination: MPK rolls out, SGLang
# rescores — or vice versa). Uses sglang's input_ids + return_logprob +
# logprob_start_len, i.e. exactly miles' recompute_logprobs_via_prefill
# payload shape.
import json
import math
import sys
import urllib.request

ref = json.load(open(sys.argv[1]))
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

# input_token_logprobs: list of (logprob, token_id, text?) starting at
# logprob_start_len+1 — logprob of ids[t] given prefix, aligned per sglang.
itl = out["meta_info"]["input_token_logprobs"]
deltas = []
checked = 0
for j, entry in enumerate(itl):
    lp_sgl, tok = entry[0], entry[1]
    t = p0 - 1 + j  # first entry is the token at logprob_start_len
    if t >= len(ids) or lp_sgl is None:
        continue
    assert tok == ids[t], f"alignment: {tok} != {ids[t]} at {t}"
    p_mpk = ref["probs"][t - 1]
    if p_mpk <= 0:
        continue
    deltas.append(abs(math.log(p_mpk) - lp_sgl))
    checked += 1

deltas.sort()
n = len(deltas)
print(f"positions={n}  max|dt|={deltas[-1]:.6f}  "
      f"mean={sum(deltas)/n:.6f}  p99={deltas[int(n*0.99)-1]:.6f}")
