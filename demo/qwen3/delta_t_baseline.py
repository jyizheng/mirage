# M0 delta-t measurement harness (design doc: mpk-rl-design-doc.md §9).
#
# Quantifies Training-Inference Mismatch between the MPK rollout engine and
# a trainer-style forward (HF transformers, bf16, teacher forcing) — the
# quantity TIM/VeXact call delta_t. Contrast lines:
#   (a) MPK rollout  vs MPK rescore          -> expected max|dt| == 0 (proven)
#   (b) MPK rollout  vs HF/torch forward     -> expected nonzero (this is M0)
#
# Input: a --capture-probs --dump-tokens-file dump from demo.py (the rollout),
# e.g. produced by rescore_consistency.py as ref.json.
#
# Usage: python delta_t_baseline.py --ref /tmp/rescore_sampled3/ref.json
import argparse
import json
import math

import torch
from transformers import AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--ref", required=True, help="rollout dump json")
parser.add_argument("--model", default="Qwen/Qwen3-8B")
args = parser.parse_args()

ref = json.load(open(args.ref))
ids = ref["token_ids"]
p0 = ref["prompt_length"]
mpk_probs = ref["probs"]  # slot t holds P(token_{t+1} | <=t), rollout capture

model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.bfloat16
).cuda()
model.eval()

with torch.inference_mode():
    input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
    logits = model(input_ids=input_ids).logits[0]  # [L, vocab] bf16
    logprobs = torch.log_softmax(logits.float(), dim=-1)

# generated region: tokens at positions p0..L-1, predicted from rows p0-1..L-2
rows, deltas, flips = [], [], 0
for pos in range(p0, len(ids)):
    p_mpk = mpk_probs[pos - 1]
    if p_mpk <= 0.0:
        continue  # unwritten slot
    lp_hf = logprobs[pos - 1, ids[pos]].item()
    lp_mpk = math.log(p_mpk)
    deltas.append(abs(lp_mpk - lp_hf))
    if logits[pos - 1].argmax().item() != ids[pos]:
        flips += 1
    rows.append((pos, lp_mpk, lp_hf))

deltas_t = torch.tensor(deltas)
print(f"model={args.model}  trajectory_len={len(ids)}  prompt={p0}  "
      f"compared_positions={len(deltas)}")
print(f"delta_t (|logp_mpk - logp_hf|):  max={deltas_t.max():.6f}  "
      f"mean={deltas_t.mean():.6f}  p99={deltas_t.quantile(0.99):.6f}")
print(f"argmax flips (HF argmax != rollout token): {flips}/{len(deltas)}")
print("(contrast: MPK rollout vs MPK rescore is bitwise 0 — "
      "see rescore_consistency.py FULL RESCORE)")
