# E23: TIM dose-response. Inject measured rollout-rescore mismatch
# (empirical delta_t distribution from E20, SGLang det mode) into the
# GRPO importance ratio at scale factors f, and measure the optimization
# distortion: clip fraction, |ratio-1| tail, and gradient direction
# rotation vs the zero-noise (true on-policy) gradient.
#
# lp_old := lp_theta.detach() + f * delta,  delta ~ empirical E20 deltas.
# f=0 is exact zero-mismatch (what MPK-Det provides); f=1 is "what SGLang
# actually exhibits"; other f values trace the dose-response curve.
import json
import math
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-1.7B"
DELTAS_FILE = sys.argv[1] if len(sys.argv) > 1 else "/tmp/e20_det.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/e23_noise.json"
GROUP = 8
NEW = 128
PROMPTS = [
    "Natalia sold clips to 48 of her friends in April, and then she sold "
    "half as many clips in May. How many clips did Natalia sell altogether "
    "in April and May?",
    "A robe takes 2 bolts of blue fiber and half that much white fiber. "
    "How many bolts in total does it take?",
    "Josh decides to try flipping a house. He buys a house for $80,000 and "
    "then puts in $50,000 in repairs. This increased the value of the "
    "house by 150%. How much profit did he make?",
    "James decides to run 3 sprints 3 times a week. He runs 60 meters each "
    "sprint. How many total meters does he run a week?",
]
SCALES = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]
CLIP_EPS = 0.2

torch.manual_seed(20260723)
random.seed(20260723)

deltas = json.load(open(DELTAS_FILE))["deltas"]
print(f"empirical deltas: n={len(deltas)} mean|d|="
      f"{sum(abs(d) for d in deltas)/len(deltas):.4g}", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16).cuda()
model.gradient_checkpointing_enable()

# ── generate trajectories once (fixed across scales) ─────────────────────
trajs = []  # (ids, plen)
model.eval()
for p in PROMPTS:
    msgs = [{"role": "user", "content": p}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids.cuda()
    plen = ids.shape[1]
    with torch.no_grad():
        out = model.generate(ids, do_sample=True, temperature=1.0,
                             top_p=1.0, top_k=0, max_new_tokens=NEW,
                             num_return_sequences=GROUP,
                             pad_token_id=tok.eos_token_id)
    for r in range(GROUP):
        seq = out[r]
        # trim trailing pads
        end = len(seq)
        trajs.append((seq[:end].tolist(), plen))
print(f"trajectories: {len(trajs)}", flush=True)

# advantages: fixed random +-1 pattern (we measure mechanics, not reward)
advs = [1.0 if i % 2 == 0 else -1.0 for i in range(len(trajs))]

model.train()


def token_logprobs(ids_list, plen):
    ids = torch.tensor([ids_list], device="cuda")
    logits = model(input_ids=ids).logits[0]
    rows = torch.arange(plen - 1, len(ids_list) - 1, device="cuda")
    tg = ids[0][rows + 1]
    lp = torch.log_softmax(logits[rows].float(), dim=-1)
    return lp.gather(-1, tg.unsqueeze(-1)).squeeze(-1)


def grpo_grad(scale):
    """One GRPO loss+backward over all trajectories; returns metrics and
    the flattened gradient vector."""
    model.zero_grad(set_to_none=True)
    clip_hits, n_tok = 0, 0
    ratio_abs = []
    for i, (ids_list, plen) in enumerate(trajs):
        lp_theta = token_logprobs(ids_list, plen)
        d = torch.tensor([deltas[random.randrange(len(deltas))]
                          for _ in range(lp_theta.numel())],
                         dtype=torch.float32, device="cuda")
        lp_old = lp_theta.detach() + scale * d
        ratio = torch.exp(lp_theta - lp_old)
        ratio_abs.extend((ratio - 1).abs().detach().tolist())
        clip_hits += int(((ratio < 1 - CLIP_EPS) |
                          (ratio > 1 + CLIP_EPS)).sum().item())
        n_tok += ratio.numel()
        un = ratio * advs[i]
        cl = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advs[i]
        loss = -torch.min(un, cl).mean() / len(trajs)
        loss.backward()
    g = torch.cat([p.grad.float().flatten() for p in model.parameters()
                   if p.grad is not None])
    ratio_abs.sort()
    n = len(ratio_abs)
    return {
        "scale": scale,
        "clip_frac": clip_hits / n_tok,
        "ratio_dev_mean": sum(ratio_abs) / n,
        "ratio_dev_p99": ratio_abs[int(n * 0.99) - 1],
        "ratio_dev_max": ratio_abs[-1],
    }, g


results = []
g0 = None
for f in SCALES:
    torch.manual_seed(777)      # same noise stream shape across scales
    random.seed(777)
    m, g = grpo_grad(f)
    if f == 0.0:
        g0 = g.clone()
        m["grad_cos_vs_clean"] = 1.0
    else:
        m["grad_cos_vs_clean"] = float(
            torch.nn.functional.cosine_similarity(g, g0, dim=0).item())
    print(json.dumps(m), flush=True)
    results.append(m)

json.dump(results, open(OUT, "w"))
print("wrote", OUT)
