# E19: scaled-down GRPO stability experiment (TIM-paper replication shape).
#
# Two arms differing ONLY in where the trainer's logprob VALUES come from:
#   arm "mpk": theta-logprob values come from MPK (bit-exact w.r.t. the
#              rollout engine), while a selectable trainer backend supplies
#              the differentiable replay and backward (design doc §5.3).
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

from mirage.mpk.trainer_backend import (
    bind_forward_values,
    create_trainer_backend,
)
from mirage.mpk.weight_sync import build_name_matching_sync_plan


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
    R = tokens.shape[0]           # group size = request slots
    max_seq = tokens.shape[1]
    dev = "cuda"
    arm = args.grpo_arm
    out_path = args.grpo_log or f"/tmp/e19_{arm}.jsonl"
    log_f = open(out_path, "w")

    backend = create_trainer_backend(
        args.grpo_trainer_backend,
        model_name=args.model,
        tokenizer=tokenizer,
        learning_rate=args.grpo_lr,
        micro_batch_size=getattr(args, "grpo_trainer_micro_batch_size", 0),
        factory_kwargs={"engine_args": args}
        if args.grpo_trainer_backend != "hf" else None,
    )

    demo_sd = dict(model_demo.named_parameters())
    trainer_sd = dict(backend.named_parameters())
    sync_plan = build_name_matching_sync_plan(
        trainer_sd, demo_sd, tie_lm_head_to_embeddings=True)
    print(f"[e19] weight-sync plan: {len(sync_plan.specs)} tensors")

    def sync_weights():
        report = sync_plan.sync(trainer_sd, demo_sd, strict=True)
        return report

    sync_report = sync_weights()  # start from identical weights on both sides
    print(f"[e19] initial weight-sync: {sync_report.tensors} tensors, "
          f"{sync_report.gib:.2f} GiB")
    data = load_gsm8k(tokenizer, args.grpo_steps * 1)
    print(f"[e19] arm={arm} steps={args.grpo_steps} group={R} "
          f"lr={args.grpo_lr} data={len(data)} "
          f"trainer_backend={args.grpo_trainer_backend} trainer_micro_batch="
          f"{getattr(backend, 'micro_batch_size', 0) or R}")

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
        # re-arm the in-kernel runtime state over the same meta tensor
        # pointers (queues/events/step bookkeeping); the task graph and
        # resource registrations from the initial init are reused
        mpk.init_request_func()
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

    clip_eps = 0.2
    measure_old_recompute = getattr(args, "grpo_measure_old_recompute", False)

    for it, (prompt_ids, gold) in enumerate(data[: args.grpo_steps]):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        samples = rollout(prompt_ids)
        t_rollout = time.perf_counter() - t0
        rewards = []
        for s in samples:
            text = tokenizer.decode(s["ids"][s["plen"]:],
                                    skip_special_tokens=True)
            rewards.append(1.0 if extract_answer(text) == gold else 0.0)
        rw = torch.tensor(rewards, device=dev)
        adv = (rw - rw.mean()) / (rw.std() + 1e-6)

        # This pass is a counterfactual baseline, not part of the zero-TIM
        # engine.  Keep it opt-in so E2E timing does not charge MPK for work
        # that the design eliminates.
        t_old_recompute = 0.0
        if measure_old_recompute:
            old_recompute_samples = [s for s in samples if s["lp_old"]]
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            with torch.no_grad():
                backend.selected_token_logprobs(old_recompute_samples)
            torch.cuda.synchronize()
            t_old_recompute = time.perf_counter() - t1

        torch.cuda.synchronize()
        t2 = time.perf_counter()
        backend.zero_grad()
        losses, ratio_devs, clip_hits, n_tok = [], [], 0, 0
        worst = {}
        active = [
            i for i, s in enumerate(samples)
            if s["lp_old"] and float(adv[i].abs()) >= 1e-8
        ]
        trainer_lps = backend.selected_token_logprobs(
            [samples[i] for i in active]
        )
        for active_idx, i in enumerate(active):
            s = samples[i]
            lp_old = torch.tensor(s["lp_old"], dtype=torch.float32, device=dev)
            lp_trainer = trainer_lps[active_idx]
            if arm == "mpk":
                lp_theta = bind_forward_values(lp_old, lp_trainer)
            else:
                lp_theta = lp_trainer
            ratio = torch.exp(lp_theta - lp_old)
            ratio_devs.append((ratio - 1).abs().max().item())
            with torch.no_grad():
                devi = (lp_theta.detach() - lp_old).abs()
                j = int(devi.argmax())
                if float(devi[j]) > 5.0:
                    json.dump(
                        {"ids": s["ids"], "plen": s["plen"],
                         "pos": s["pos"][j], "lp_old": float(lp_old[j]),
                         "lp_theta": float(lp_theta[j].detach())},
                        open(f"/tmp/e19_offender_{it}_{i}.json", "w"))
                if float(devi[j]) > worst.get("dlp", 0):
                    worst.update({
                        "dlp": float(devi[j]),
                        "pos": s["pos"][j],
                        "tok": s["ids"][s["pos"][j]],
                        "lp_old": float(lp_old[j]),
                        "lp_theta": float(lp_theta[j]),
                        "gen_off": j,
                        "gen_len": len(s["pos"]),
                    })
            clip_hits += int(((ratio < 1 - clip_eps) |
                              (ratio > 1 + clip_eps)).sum().item())
            n_tok += ratio.numel()
            un = ratio * adv[i]
            cl = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv[i]
            losses.append(-torch.minimum(un, cl).mean())
        if losses:
            loss = torch.stack(losses).mean()
            gn = backend.backward_and_step(loss)
            torch.cuda.synchronize()
            # sync trainer -> rollout weights (MPK reads them next rollout)
            t_sync_start = time.perf_counter()
            sync_report = sync_weights()
            torch.cuda.synchronize()
            t_sync = time.perf_counter() - t_sync_start
        else:
            t_sync = 0.0
        torch.cuda.synchronize()
        t_train = time.perf_counter() - t2
        t_step = time.perf_counter() - t0
        rec = {
            "step": it,
            "reward_mean": float(rw.mean()),
            "ratio_dev_max": max(ratio_devs) if ratio_devs else 0.0,
            "clip_frac": (clip_hits / n_tok) if n_tok else 0.0,
            "worst": worst or None,
            "loss": float(loss.item()) if losses else None,
            "grad_norm": float(gn) if losses else None,
            "gen_lens": [len(s["pos"]) for s in samples],
            "t_rollout_s": round(t_rollout, 4),
            "t_old_recompute_s": round(t_old_recompute, 4),
            # Deprecated alias retained for existing analysis scripts.
            "t_recompute_s": round(t_old_recompute, 4),
            "t_train_s": round(t_train, 4),
            "t_update_s": round(t_train, 4),
            "sync_tensors": sync_report.tensors if losses else 0,
            "sync_bytes": sync_report.bytes if losses else 0,
            "t_sync_s": round(t_sync, 6),
            "t_step_s": round(t_step, 4),
            "sec": t_step,
        }
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        if it % 5 == 0:
            print(f"[e19] {rec}")
    log_f.close()
    print(f"[e19] DONE arm={arm} log={out_path}")
