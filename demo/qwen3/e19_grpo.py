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
# Multi-epoch (off-policy inner-epoch) reuse (--inner-epochs N, default 1 =
# exact single-epoch behavior): per outer step the group is rolled out ONCE
# and pi_old logprobs are frozen from the rollout capture; advantages are
# frozen from the rollout rewards. Then N clipped updates are taken. Before
# every epoch after the first, the frozen trajectories are teacher-forced
# through the SAME MPK engine (full trajectory resubmitted as a prompt; the
# prefill probability-capture task emits P(token_t | prefix) under the
# weights synced after the previous epoch's update) to obtain the current
# pi_theta logprobs. Because rescore == rollout is bitwise on this engine
# (rescore_consistency --full-rescore; serving-level e35, 260/260), epoch-1
# ratios are exactly 1 and later epochs' ratios reflect ONLY real parameter
# drift — no trainer/inference mismatch enters the objective.
#
# Called from demo.py (--grpo-steps > 0) with the compiled MPK and its meta
# tensors in scope. Single GPU; group sampling uses the batch's request
# slots (position-keyed Gumbel noise differs per request slot, so a group
# on the same prompt yields distinct trajectories).
import atexit
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
    sampling_capture_fused=False,
):
    R = tokens.shape[0]           # group size = request slots
    max_seq = tokens.shape[1]
    inner_epochs = max(1, int(getattr(args, "inner_epochs", 1) or 1))
    if inner_epochs > 1 and args.grpo_arm == "mpk" and sampling_capture_fused:
        raise ValueError(
            "--inner-epochs > 1 rescores the frozen trajectories by "
            "teacher-forcing them through the prefill probability-capture "
            "task, which is not in the task graph when sampling capture is "
            "fused; rerun with --no-fused-sampling-capture (or the default "
            "parallel sampling path)"
        )
    if args.max_num_batched_tokens < R:
        raise ValueError(
            "fixed-group GRPO requires --max-num-batched-tokens >= "
            f"--max-num-batched-requests ({args.max_num_batched_tokens} < {R}); "
            "otherwise the rollout group is silently split into request waves"
        )
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
    close_backend = getattr(backend, "close", None)
    if close_backend is not None:
        atexit.register(close_backend)

    demo_sd = dict(model_demo.named_parameters())
    sync_plan = None
    sync_source_ids = None

    def sync_weights():
        nonlocal sync_plan, sync_source_ids
        trainer_sd = dict(backend.named_parameters())
        source_ids = tuple((name, id(value)) for name, value in trainer_sd.items())
        if sync_plan is None or source_ids != sync_source_ids:
            sync_plan = build_name_matching_sync_plan(
                trainer_sd, demo_sd, tie_lm_head_to_embeddings=True)
            sync_source_ids = source_ids
        report = sync_plan.sync(trainer_sd, demo_sd, strict=True)
        return report

    sync_report = sync_weights()  # start from identical weights on both sides
    print(f"[e19] weight-sync plan: {len(sync_plan.specs)} tensors")
    print(f"[e19] initial weight-sync: {sync_report.tensors} tensors, "
          f"{sync_report.gib:.2f} GiB")
    data = load_gsm8k(tokenizer, args.grpo_steps * 1)
    print(f"[e19] arm={arm} steps={args.grpo_steps} group={R} "
          f"inner_epochs={inner_epochs} "
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

    def mpk_rescore(samples):
        # Teacher-force the FROZEN trajectories through the SAME MPK engine
        # that produced them: each full trajectory is resubmitted as a
        # prompt, and the prefill probability-capture task writes
        # P(token_t | prefix) under the CURRENT (post-sync) weights into
        # prob_buffer[r, t-1] — the serving rescore path validated bitwise
        # against rollout capture on unchanged weights (rescore_consistency
        # --full-rescore; e35). The whole group is rescored in ONE batched
        # engine pass.
        with torch.no_grad():
            tokens.zero_()
            prob_buffer.zero_()
            lengths = []
            for r, s in enumerate(samples):
                # A trajectory that fills the sequence buffer (no eos before
                # the cap) cannot be fully teacher-forced: prefill needs one
                # free decode slot. Truncate the resubmitted prompt by one
                # token; the uncovered final position falls back to the
                # frozen lp_old below (its ratio contribution is exactly 1,
                # so no bias enters the surrogate).
                length = min(len(s["ids"]), max_seq - 1)
                lengths.append(length)
                tokens[r, :length] = torch.tensor(s["ids"][:length],
                                                  device=dev)
                prompt_lengths[r] = length
            step.zero_()
            num_new_tokens.fill_(1)
        mpk.init_request_func()
        mpk()
        torch.cuda.synchronize()
        lps = []
        for r, s in enumerate(samples):
            # prefill capture covers P(token_t | prefix) for t in
            # [1, lengths[r]); positions beyond keep their frozen value
            lp = []
            for k, t in enumerate(s["pos"]):
                if t < lengths[r]:
                    p = float(prob_buffer[r, t - 1].item())
                    if p <= 0.0:
                        raise RuntimeError(
                            f"rescore capture missing at request {r} slot "
                            f"{t - 1} (prob={p}); prefill probability "
                            "capture did not cover the teacher-forced region"
                        )
                    lp.append(math.log(p))
                else:
                    lp.append(s["lp_old"][k])
            lps.append(torch.tensor(lp, dtype=torch.float32, device=dev))
        return lps

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
        active = [
            i for i, s in enumerate(samples)
            if s["lp_old"] and float(adv[i].abs()) >= 1e-8
        ]
        active_samples = [samples[i] for i in active]
        # Inner-epoch loop: pi_old (lp_old) and the advantages stay FROZEN
        # from the rollout; only the pi_theta side of the ratio moves.
        epochs_log = []
        ep1 = None
        t_sync_total = 0.0
        for ep in range(1, inner_epochs + 1):
            # (a) current pi_theta logprob VALUES for the frozen
            # trajectories. Epoch 1: the rollout capture IS pi_theta at
            # capture time (rescore == rollout bitwise on this engine), so
            # it is used directly and the ratio is asserted to be exactly
            # 1. Epochs > 1: one batched MPK rescore pass under the weights
            # synced after the previous epoch's update.
            t_rescore = 0.0
            rescored = None
            if arm == "mpk" and ep > 1:
                torch.cuda.synchronize()
                tr0 = time.perf_counter()
                rescored = mpk_rescore(samples)
                torch.cuda.synchronize()
                t_rescore = time.perf_counter() - tr0
            # (b) clipped surrogate on the frozen trajectories/advantages,
            # backward, optimizer step, trainer -> rollout weight sync.
            backend.zero_grad()
            losses, ratio_devs, clip_hits, n_tok = [], [], 0, 0
            ratio_dev_sum = 0.0
            if ep == 1:
                worst = {}
            trainer_lps = backend.selected_token_logprobs(active_samples)
            for active_idx, i in enumerate(active):
                s = samples[i]
                lp_old = torch.tensor(s["lp_old"], dtype=torch.float32,
                                      device=dev)
                lp_trainer = trainer_lps[active_idx]
                if arm == "mpk":
                    authoritative = lp_old if rescored is None else rescored[i]
                    lp_theta = bind_forward_values(authoritative, lp_trainer)
                else:
                    lp_theta = lp_trainer
                ratio = torch.exp(lp_theta - lp_old)
                ratio_dev = (ratio - 1).abs()
                ratio_devs.append(ratio_dev.max().item())
                ratio_dev_sum += float(ratio_dev.sum().item())
                if ep == 1:
                    # capture-vs-trainer deviation diagnostics (TIM noise on
                    # the hf arm); epochs > 1 deviate by REAL drift, which
                    # is reported through the per-epoch ratio stats instead
                    with torch.no_grad():
                        devi = (lp_theta.detach() - lp_old).abs()
                        j = int(devi.argmax())
                        if float(devi[j]) > 5.0:
                            json.dump(
                                {"ids": s["ids"], "plen": s["plen"],
                                 "pos": s["pos"][j],
                                 "lp_old": float(lp_old[j]),
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
            if arm == "mpk" and ep == 1 and ratio_devs:
                dev_max = max(ratio_devs)
                assert dev_max == 0.0, (
                    "epoch-1 ratio must be exactly 1 (pi_theta values are "
                    f"the rollout capture itself); max |ratio-1| = {dev_max}"
                )
            if losses:
                loss = torch.stack(losses).mean()
                gn = backend.backward_and_step(loss)
                torch.cuda.synchronize()
                # sync trainer -> rollout weights (MPK reads them at the
                # next epoch's rescore / the next rollout)
                t_sync_start = time.perf_counter()
                sync_report = sync_weights()
                torch.cuda.synchronize()
                t_sync = time.perf_counter() - t_sync_start
            else:
                t_sync = 0.0
            t_sync_total += t_sync
            ep_rec = {
                "epoch": ep,
                "ratio_dev_mean": (ratio_dev_sum / n_tok) if n_tok else 0.0,
                "ratio_dev_max": max(ratio_devs) if ratio_devs else 0.0,
                "clip_frac": (clip_hits / n_tok) if n_tok else 0.0,
                "t_rescore_s": round(t_rescore, 4),
                "loss": float(loss.item()) if losses else None,
                "grad_norm": float(gn) if losses else None,
                "t_sync_s": round(t_sync, 6),
            }
            epochs_log.append(ep_rec)
            if ep == 1:
                ep1 = ep_rec
            if inner_epochs > 1:
                print(f"[e19] step {it} epoch {ep}/{inner_epochs} "
                      f"ratio_dev mean={ep_rec['ratio_dev_mean']:.3e} "
                      f"max={ep_rec['ratio_dev_max']:.3e} "
                      f"clip_frac={ep_rec['clip_frac']:.4f} "
                      f"rescore={t_rescore:.3f}s loss={ep_rec['loss']}")
            if not losses:
                # no active trajectory (e.g. uniform rewards): no update
                # happened, so further epochs would rescore unchanged
                # weights — skip them
                break
        torch.cuda.synchronize()
        t_train = time.perf_counter() - t2
        t_step = time.perf_counter() - t0
        rec = {
            "step": it,
            "reward_mean": float(rw.mean()),
            # top-level ratio/clip keep their historical meaning: the
            # rollout-capture vs pi_theta comparison of the FIRST update
            # (identically the whole story when inner_epochs == 1);
            # per-epoch stats live under "epochs"
            "ratio_dev_max": ep1["ratio_dev_max"] if ep1 else 0.0,
            "clip_frac": ep1["clip_frac"] if ep1 else 0.0,
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
            "t_sync_s": round(t_sync_total, 6),
            "inner_epochs": inner_epochs,
            "epochs": epochs_log,
            "t_zero_tim_step_s": round(t_step - t_old_recompute, 4),
            "old_recompute_overhead_frac": (
                t_old_recompute / t_step if t_step else 0.0
            ),
            "t_step_s": round(t_step, 4),
            "sec": t_step,
            "trainer_backend": args.grpo_trainer_backend,
            "model": args.model,
            "max_num_batched_requests": R,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_seq_length": max_seq,
            "deterministic": args.deterministic,
            "sampling_seed": args.sampling_seed,
            "capture_probs": args.capture_probs,
        }
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        if it % 5 == 0:
            print(f"[e19] {rec}")
    log_f.close()
    if close_backend is not None:
        close_backend()
        atexit.unregister(close_backend)
    print(f"[e19] DONE arm={arm} log={out_path}")
