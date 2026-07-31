# E43: the RL-engine baseline IS miles' recompute path, not a mimic.
# Build a Sample from an MPK rollout ref, call miles' OWN
# _build_prefill_scoring_payload, POST to SGLang, and (1) confirm the
# payload equals what our e32 baseline sends, (2) recover the per-token
# reference logprobs miles would feed the trainer.
import json
import math
import sys
import urllib.request

sys.path.insert(0, "/workspace/miles")
from miles.rollout.generate_utils.prefill_logprobs import (
    _build_prefill_scoring_payload)
from miles.utils.types import Sample

ref = json.load(open(sys.argv[1]))       # MPK rollout: token_ids, probs, prompt_length
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8322
ids = ref["token_ids"]
p0 = ref["prompt_length"]
resp_len = len(ids) - p0

# miles Sample: full token sequence + response length
s = Sample(tokens=ids, response_length=resp_len)
try:
    payload = _build_prefill_scoring_payload(args=None, sample=s,
                                             sampling_params={})
except TypeError:
    # older signature
    payload = _build_prefill_scoring_payload(None, s, {})

# (1) payload equivalence vs our e32 baseline
ours = {"input_ids": ids,
        "sampling_params": {"max_new_tokens": 0, "temperature": 0},
        "return_logprob": True, "logprob_start_len": p0 - 1}
same_core = (payload["input_ids"] == ours["input_ids"] and
             payload["logprob_start_len"] == ours["logprob_start_len"] and
             payload["return_logprob"] is True and
             payload["sampling_params"]["max_new_tokens"] == 0 and
             payload["sampling_params"]["temperature"] == 0)
print("miles payload == our baseline payload (core fields):", same_core)
print("miles logprob_start_len:", payload["logprob_start_len"],
      "max_new_tokens:", payload["sampling_params"]["max_new_tokens"])

# (2) drive miles' payload against SGLang, recover response logprobs, dt vs MPK
req = urllib.request.Request(f"http://127.0.0.1:{PORT}/generate",
                             data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=600) as r:
    out = json.loads(r.read())
itl = out["meta_info"]["input_token_logprobs"]
deltas = []
for j, e in enumerate(itl):
    lp, tok = e[0], e[1]
    t = p0 - 1 + j
    if t >= len(ids) or lp is None or t < p0:
        continue
    p_mpk = ref["probs"][t - 1]
    if p_mpk > 0:
        deltas.append(abs(math.log(p_mpk) - lp))
ad = sorted(deltas)
n = len(ad)
print(f"miles-baseline rescore: n={n} max|dt|={ad[-1]:.6f} "
      f"mean={sum(ad)/n:.6f} p99={ad[int(n*0.99)-1]:.6f}")
print("MILES BASELINE: OK" if same_core and n > 0 else "CHECK")
