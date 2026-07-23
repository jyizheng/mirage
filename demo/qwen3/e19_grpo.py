# E19: scaled-down GRPO stability experiment (TIM-paper replication shape).
#
# Two arms differing ONLY in where the trainer's logprob VALUES come from:
#   arm "mpk": theta-logprobs enter the loss through an autograd.Function
#              whose forward returns the MPK rescore values (bit-exact w.r.t.
#              the rollout engine) and whose backward recomputes them
#              differentiably on the trainer stack (design doc §5.3).
#              On-policy, ratio == 1 bitwise -> clipping never spuriously
#              activates.
#   arm "hf":  theta-logprobs are the trainer-stack (HF) forward values —
#              the standard "recomputation" convention. delta_t noise makes
#              ratios != 1, randomly engaging the clip (the TIM pathology).
# Rollouts, rewards, advantages, backward machinery, optimizer, and weight
# sync are identical across arms.
#
# Called from demo.py (--grpo-steps > 0) with the compiled MPK and its meta
# tensors in scope. Single GPU; group sampling uses the batch's request
# slots (position-keyed Gumbel noise differs per request slot, so a group
# on the same prompt yields distinct trajectories).
import json
import math
import re
import time

import torch


def extract_answer(text):
    m = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return m[-1] if m else None


def load_gsm8k(tokenizer, n):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    items = []
    for row in ds:
        q = row["question"]
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        msgs = [
            {"role": "user",
             "content": q + "\nThink briefly, then give the final numeric "
                            "answer after '####'."}
        ]
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tokenizer(text).input_ids
        if len(ids) <= 320:
            items.append((ids, gold))
        if len(items) >= n:
            break
    return items


def run(
    args,
    mpk,
    model_demo,           # demo Qwen3ForCausalLM (params attached to MPK)
    tokenizer,
    tokens,               # [R, max_seq] meta tensor
    step,                 # [R]
    prompt_lengths,       # [R]
    num_new_tokens,       # [R]
    prob_buffer,          # [mbt, max_seq] float32 (attached capture buffer)
    eos_token_id,
):
    from transformers import AutoModelForCausalLM

    R = tokens.shape[0]           # group size = request slots
    max_seq = tokens.shape[1]
    dev = "cuda"
    arm = args.grpo_arm
    out_path = args.grpo_log or f"/tmp/e19_{arm}.jsonl"
    log_f = open(out_path, "w")

    trainer = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16
    ).cuda()
    trainer.gradient_checkpointing_enable()
    trainer.train()
    opt = torch.optim.AdamW(trainer.parameters(), lr=args.grpo_lr)

    # weight sync map: demo model param names match HF checkpoint names
    demo_sd = dict(model_demo.named_parameters())
    hf_sd = dict(trainer.named_parameters())
    sync_keys = [k for k in hf_sd if k in demo_sd]
    print(f"[e19] weight-sync keys: {len(sync_keys)}/{len(hf_sd)}")

    data = load_gsm8k(tokenizer, args.grpo_steps * 1)
    print(f"[e19] arm={arm} steps={args.grpo_steps} group={R} "
          f"lr={args.grpo_lr} data={len(data)}")

    def rollout(prompt_ids):
        plen = len(prompt_ids)
        with torch.no_grad():
            tokens.zero_()
            for r in range(R):
                tokens[r, :plen] = torch.tensor(prompt_ids, device=dev)
            prompt_lengths.fill_(plen)
            step.zero_()
            num_new_tokens.fill_(1)
            prob_buffer.zero_()
        meta_ptrs = [t.data_ptr() for t in mpk.meta_tensors.values()]
        prof_ptr = (mpk.profiler_tensor.data_ptr()
                    if mpk.profiler_tensor is not None else 0)
        names = list(mpk._model_tensors.keys())
        ptrs = [t.data_ptr() for t in mpk._model_tensors.values()]
        mpk.init_func(
            meta_ptrs, prof_ptr, mpk.mpi_rank, mpk.num_workers,
            mpk.num_local_schedulers, mpk.num_remote_schedulers,
            mpk.max_seq_length, mpk.total_num_requests, mpk.eos_token_id,
            mpk.allocate_nvshmem_teams, names, ptrs, "",
        )
        mpk()
        torch.cuda.synchronize()
        outs = []
        for r in range(R):
            end = int(step[r].item()) + 1
            ids = tokens[r, :end].tolist()
            # generated region and its rollout logprobs (slot t-1 -> P(tok_t))
            lp = []
            pos = []
            for t in range(plen, end):
                p = float(prob_buffer[r, t - 1].item())
                if p > 0.0:
                    lp.append(math.log(p))
                    pos.append(t)
            outs.append({"ids": ids, "plen": plen, "pos": pos, "lp_old": lp})
        return outs

    def trainer_logprobs(sample):
        ids = torch.tensor([sample["ids"]], dtype=torch.long, device=dev)
        rows = torch.tensor([t - 1 for t in sample["pos"]], device=dev)
        tg = torch.tensor([sample["ids"][t] for t in sample["pos"]], device=dev)
        logits = trainer(input_ids=ids).logits[0]
        lp = torch.log_softmax(logits[rows].float(), dim=-1)
        return lp.gather(-1, tg.unsqueeze(-1)).squeeze(-1)

    class MPKForward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, anchor, sample_idx):
            s = samples[sample_idx]
            ctx.sample_idx = sample_idx
            return torch.tensor(s["lp_old"], dtype=torch.float32, device=dev)

        @staticmethod
        def backward(ctx, grad_out):
            with torch.enable_grad():
                lp = trainer_logprobs(samples[ctx.sample_idx])
                lp.backward(grad_out)
            return None, None

    anchor = torch.zeros(1, requires_grad=True, device=dev)
    clip_eps = 0.2

    for it, (prompt_ids, gold) in enumerate(data[: args.grpo_steps]):
        t0 = time.time()
        samples = rollout(prompt_ids)
        rewards = []
        for s in samples:
            text = tokenizer.decode(s["ids"][s["plen"]:],
                                    skip_special_tokens=True)
            rewards.append(1.0 if extract_answer(text) == gold else 0.0)
        rw = torch.tensor(rewards, device=dev)
        adv = (rw - rw.mean()) / (rw.std() + 1e-6)

        opt.zero_grad(set_to_none=True)
        losses, ratio_devs = [], []
        for i, s in enumerate(samples):
            if not s["lp_old"] or adv[i].abs() < 1e-8:
                continue
            lp_old = torch.tensor(s["lp_old"], dtype=torch.float32, device=dev)
            if arm == "mpk":
                lp_theta = MPKForward.apply(anchor, i)
            else:
                lp_theta = trainer_logprobs(s)
            ratio = torch.exp(lp_theta - lp_old)
            ratio_devs.append((ratio - 1).abs().max().item())
            un = ratio * adv[i]
            cl = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv[i]
            losses.append(-torch.minimum(un, cl).mean())
        if losses:
            loss = torch.stack(losses).mean()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(trainer.parameters(), 1.0)
            opt.step()
            # sync trainer -> rollout weights (MPK reads them next rollout)
            with torch.no_grad():
                for k in sync_keys:
                    demo_sd[k].copy_(hf_sd[k])
        rec = {
            "step": it,
            "reward_mean": float(rw.mean()),
            "ratio_dev_max": max(ratio_devs) if ratio_devs else 0.0,
            "loss": float(loss.item()) if losses else None,
            "grad_norm": float(gn) if losses else None,
            "gen_lens": [len(s["pos"]) for s in samples],
            "sec": time.time() - t0,
        }
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        if it % 5 == 0:
            print(f"[e19] {rec}")
    log_f.close()
    print(f"[e19] DONE arm={arm} log={out_path}")
