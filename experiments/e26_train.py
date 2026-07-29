# E26: reward-separation at scale. Same two-arm injected-mismatch GRPO
# as E25 but Qwen3-8B, thousands of steps, plus a fixed 32-prompt
# held-out eval (greedy) every 100 steps for a low-noise accuracy curve.
#
# Usage: python e26_train.py <scale_f> <steps> <out.jsonl> [deltas.json]
import json
import random
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-8B"
SCALE = float(sys.argv[1])
STEPS = int(sys.argv[2])
OUT = sys.argv[3]
DELTAS_FILE = sys.argv[4] if len(sys.argv) > 4 else "/tmp/e20_det.json"
GROUP = 8
NEW = 200
CLIP_EPS = 0.2
LR = 2e-6

torch.manual_seed(20260728)
random.seed(20260728)

deltas = json.load(open(DELTAS_FILE))["deltas"]


def extract_answer(text):
    m = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return m[-1] if m else None


tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16).cuda()
model.gradient_checkpointing_enable()
opt = torch.optim.AdamW(model.parameters(), lr=LR)

from datasets import load_dataset


def build_items(split, cap, max_prompt=320):
    ds = load_dataset("openai/gsm8k", "main", split=split)
    items = []
    for row in ds:
        q = row["question"]
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        msgs = [{"role": "user",
                 "content": q + "\nThink briefly, then give the final "
                                "numeric answer after '####'."}]
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
        ids = tok(text).input_ids
        if len(ids) <= max_prompt:
            items.append((ids, gold))
        if len(items) >= cap:
            break
    return items


data = build_items("train", STEPS)
eval_set = build_items("test", 32)
log_f = open(OUT, "w")


def run_eval():
    model.eval()
    correct = 0
    with torch.no_grad():
        for pids, gold in eval_set:
            ids = torch.tensor([pids], device="cuda")
            out = model.generate(ids, do_sample=False, max_new_tokens=NEW,
                                 pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0][len(pids):], skip_special_tokens=True)
            correct += int(extract_answer(text) == gold)
    return correct / len(eval_set)


for it, (prompt_ids, gold) in enumerate(data):
    if it % 100 == 0:
        acc = run_eval()
        log_f.write(json.dumps({"step": it, "eval_acc": acc}) + "\n")
        log_f.flush()
        print(f"[f={SCALE}] step {it}: EVAL {acc:.3f}", flush=True)

    # ── rollout (sampled, temperature 1) ──
    model.eval()
    ids = torch.tensor([prompt_ids], device="cuda")
    plen = ids.shape[1]
    with torch.no_grad():
        out = model.generate(ids, do_sample=True, temperature=1.0,
                             top_p=1.0, top_k=0, max_new_tokens=NEW,
                             num_return_sequences=GROUP,
                             pad_token_id=tok.eos_token_id)
    trajs = []
    rewards = []
    for r in range(GROUP):
        seq = out[r].tolist()
        end = len(seq)
        while end > plen and seq[end - 1] == tok.eos_token_id:
            end -= 1
        end = min(end + 1, len(seq))
        trajs.append(seq[:end])
        text = tok.decode(seq[plen:end], skip_special_tokens=True)
        rewards.append(1.0 if extract_answer(text) == gold else 0.0)

    rw = torch.tensor(rewards, device="cuda")
    if rw.std() < 1e-6:
        log_f.write(json.dumps({"step": it, "reward": rw.mean().item(),
                                "skipped": True}) + "\n")
        log_f.flush()
        continue
    adv = (rw - rw.mean()) / (rw.std() + 1e-6)

    # ── GRPO update ──
    model.train()
    opt.zero_grad(set_to_none=True)
    clip_hits, n_tok = 0, 0
    ratio_devs = []
    ent_sum, ent_n = 0.0, 0
    for i, seq in enumerate(trajs):
        if adv[i].abs() < 1e-8 or len(seq) <= plen:
            continue
        ids_t = torch.tensor([seq], device="cuda")
        logits = model(input_ids=ids_t).logits[0]
        rows = torch.arange(plen - 1, len(seq) - 1, device="cuda")
        tg = ids_t[0][rows + 1]
        lsm = torch.log_softmax(logits[rows].float(), dim=-1)
        lp_theta = lsm.gather(-1, tg.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            p = lsm.exp()
            ent_sum += float(-(p * lsm).sum(-1).mean().item())
            ent_n += 1
        d = torch.tensor([deltas[random.randrange(len(deltas))]
                          for _ in range(lp_theta.numel())],
                         dtype=torch.float32, device="cuda")
        lp_old = lp_theta.detach() + SCALE * d
        ratio = torch.exp(lp_theta - lp_old)
        ratio_devs.append(float((ratio - 1).abs().max().item()))
        clip_hits += int(((ratio < 1 - CLIP_EPS) |
                          (ratio > 1 + CLIP_EPS)).sum().item())
        n_tok += ratio.numel()
        un = ratio * adv[i]
        cl = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv[i]
        loss = -torch.min(un, cl).mean() / GROUP
        loss.backward()
    gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    opt.step()

    rec = {"step": it, "reward": rw.mean().item(),
           "clip_frac": clip_hits / max(n_tok, 1),
           "ratio_dev_max": max(ratio_devs) if ratio_devs else 0.0,
           "entropy": ent_sum / max(ent_n, 1),
           "grad_norm": gn,
           "gen_len": sum(len(t) - plen for t in trajs) / GROUP}
    log_f.write(json.dumps(rec) + "\n")
    log_f.flush()
    if it % 10 == 0:
        print(f"[f={SCALE}] step {it}: reward={rec['reward']:.3f} "
              f"clip={rec['clip_frac']:.4f}", flush=True)

acc = run_eval()
log_f.write(json.dumps({"step": STEPS, "eval_acc": acc}) + "\n")
print("E26 DONE", SCALE, "final eval", acc)
