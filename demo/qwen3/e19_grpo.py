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
# tensors in scope. Group sampling uses the batch's request slots
# (position-keyed Gumbel noise differs per request slot, so a group on the
# same prompt yields distinct trajectories).
#
# Device placement (--trainer-device, design doc option C): by default the
# trainer backend is colocated with the engine (single GPU, exact historical
# behavior). With --trainer-device cuda:1 the loop disaggregates: the
# engine/megakernel keeps the current device, the trainer (master weights +
# AdamW) lives on cuda:1, the trainer->engine weight sync becomes a
# cross-device P2P copy, and by default the update is bitwise comparable to
# the colocated loop (single batched epoch-1 replay forward, now on cuda:1).
# With --stream-replay-fwd additionally set (mpk arm only), the epoch-1
# replay forward is STREAMED: as trajectories retire inside the rollout
# (the offline kernel's per-request completion is host-observable through
# step[]/tokens[]), completion waves of selected-token replay forwards run
# on cuda:1 while the surviving decodes drain the rollout tail on cuda:0;
# rewards are computed on the CPU concurrently. Backward + optimizer still
# wait for all rewards (GRPO advantages are group-normalized) and the
# fenced weight sync still precedes the next rollout, so strict on-policy
# semantics -- including the epoch-1 ratio == 1 assert -- are
# device-placement-invariant either way; streaming only changes gradient
# micro-batching (like --grpo-trainer-micro-batch-size), so its update is
# not bitwise comparable to the colocated one.
import atexit
import hashlib
import json
import math
import os
import re
import time

import torch

from mirage.mpk.checkpoint import (
    capture_rng_state,
    config_echo,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
    verify_config,
)
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

    # --- trainer determinism ---------------------------------------------
    # The MPK engine is deterministic by construction, but the HF trainer's
    # backward is NOT bit-reproducible by default (embedding/SDPA backward
    # atomics): measured on B300, two identical colocated runs produce
    # different grad norms at step 0 and their rollouts diverge from step 1
    # (through the synced weights). Under --deterministic, pin the trainer
    # ops too, so the whole loop is a pure function of (weights, data,
    # seeds) and checkpoint resume / device placement are bitwise
    # invariants.
    if getattr(args, "deterministic", False):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)

    # --- device placement (option C: 2-GPU disaggregation) ---------------
    # The engine/megakernel always lives on the current device; the trainer
    # backend is colocated by default and moves to --trainer-device when set.
    engine_dev = torch.device("cuda", torch.cuda.current_device())
    trainer_device = getattr(args, "trainer_device", None)
    if trainer_device:
        td = torch.device(trainer_device)
        if td.type != "cuda":
            raise ValueError(
                f"--trainer-device must be a CUDA device, got {trainer_device!r}")
        td_index = td.index if td.index is not None else torch.cuda.current_device()
        if td_index >= torch.cuda.device_count():
            raise ValueError(
                f"--trainer-device {trainer_device} needs {td_index + 1} visible "
                f"CUDA device(s), found {torch.cuda.device_count()}")
        train_dev = torch.device("cuda", td_index)
    else:
        train_dev = engine_dev
    disagg = train_dev != engine_dev
    # Streamed replay-forward is opt-in (--stream-replay-fwd) and needs the
    # value bridge (arm mpk): the loss forward uses MPK-authoritative
    # values, so per-wave trainer batching only affects gradient
    # micro-batching. On the hf arm the trainer forward VALUES enter the
    # ratio, so streaming would change the objective with batch shape.
    # Default disagg keeps the single batched post-rollout forward, whose
    # update is bitwise comparable to the colocated loop. Measured on B300
    # (Qwen3-1.7B, group 16, mbt 16): chunked prefill staggers each row's
    # decode start, so rows retire in 13-15 bursts even under --ignore-eos,
    # and backward over that many fragmented replay graphs cost 0.84-0.97 s
    # vs 0.26 s batched - far more than the ~0.1 s of forward it hides.
    stream_replay = (disagg and arm == "mpk"
                     and bool(getattr(args, "stream_replay_fwd", False)))
    if getattr(args, "stream_replay_fwd", False) and not stream_replay:
        raise ValueError(
            "--stream-replay-fwd requires a disaggregated --trainer-device "
            "and --grpo-arm mpk")

    def sync_devices():
        torch.cuda.synchronize(engine_dev)
        if disagg:
            torch.cuda.synchronize(train_dev)

    save_every = int(getattr(args, "save_every", 0) or 0)
    ckpt_dir = getattr(args, "checkpoint_dir", None)
    resume_from = getattr(args, "resume_from", None)
    if save_every < 0:
        raise ValueError("--save-every must be >= 0")
    if save_every > 0 and not ckpt_dir:
        raise ValueError("--save-every requires --checkpoint-dir")

    out_path = args.grpo_log or f"/tmp/e19_{arm}.jsonl"
    # A resumed run appends so a shared log keeps one contiguous history.
    log_f = open(out_path, "a" if resume_from else "w")

    backend = create_trainer_backend(
        args.grpo_trainer_backend,
        model_name=args.model,
        tokenizer=tokenizer,
        learning_rate=args.grpo_lr,
        micro_batch_size=getattr(args, "grpo_trainer_micro_batch_size", 0),
        device=str(train_dev),
        factory_kwargs={"engine_args": args}
        if args.grpo_trainer_backend != "hf" else None,
    )
    if disagg:
        backend_dev = next(iter(backend.named_parameters()))[1].device
        if backend_dev != train_dev:
            raise ValueError(
                f"--trainer-device {train_dev} requested but the "
                f"{args.grpo_trainer_backend!r} backend placed parameters on "
                f"{backend_dev}; this backend is not device-parameterized")
        print(f"[e19] disaggregated: engine on {engine_dev}, trainer on "
              f"{train_dev}, streamed replay-fwd={stream_replay}")
    close_backend = getattr(backend, "close", None)
    if close_backend is not None:
        atexit.register(close_backend)

    if (save_every > 0 or resume_from) and not (
        hasattr(backend, "state_dict") and hasattr(backend, "load_state_dict")
    ):
        raise TypeError(
            f"trainer backend {args.grpo_trainer_backend!r} does not "
            "implement the optional checkpoint contract "
            "(state_dict/load_state_dict); checkpointing is unavailable"
        )

    start_step = 0
    if resume_from:
        t_load0 = time.perf_counter()
        payload = load_checkpoint(resume_from)
        verify_config(payload["config"], args)
        if payload.get("trainer_backend") != args.grpo_trainer_backend:
            raise ValueError(
                f"checkpoint trainer state is {payload.get('trainer_backend')!r} "
                f"but this run uses {args.grpo_trainer_backend!r}"
            )
        backend.load_state_dict(payload["trainer"])
        restore_rng_state(payload["rng"])
        start_step = int(payload["outer_step"])
        del payload
        t_load = time.perf_counter() - t_load0
        print(f"[e19] resumed from {resume_from}: {start_step} outer steps "
              f"completed, restore took {t_load:.2f}s; the weight sync below "
              "re-arms the MPK engine from the restored trainer weights")
        if start_step >= args.grpo_steps:
            raise ValueError(
                f"checkpoint already contains {start_step} completed steps; "
                f"--grpo-steps {args.grpo_steps} is the TOTAL step count and "
                "must exceed it for the run to continue"
            )

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
    sync_devices()
    print(f"[e19] weight-sync plan: {len(sync_plan.specs)} tensors")
    print(f"[e19] initial weight-sync: {sync_report.tensors} tensors, "
          f"{sync_report.gib:.2f} GiB")
    data = load_gsm8k(tokenizer, args.grpo_steps * 1)
    print(f"[e19] arm={arm} steps={args.grpo_steps} group={R} "
          f"inner_epochs={inner_epochs} "
          f"lr={args.grpo_lr} data={len(data)} "
          f"trainer_backend={args.grpo_trainer_backend} trainer_micro_batch="
          f"{getattr(backend, 'micro_batch_size', 0) or R}")

    # --- streamed per-trajectory completion (option C tail filling) ------
    # The offline-mode kernel retires request r when, after committing the
    # iteration's tokens and advancing step[r], either
    #   step[r] + 1 >= max_seq_length, or
    #   tokens[r, step[r]] == eos and step[r] >= prompt_len
    # (persistent_kernel.cuh, MODE_OFFLINE prepare_next_batch step 1). Both
    # signals live in device global memory the host can watch while the
    # kernel runs, which gives per-trajectory completion in offline mode
    # without the online pinned completion ring.
    engine_eos = -1 if getattr(args, "ignore_eos", False) else int(eos_token_id)
    poll_stream = None
    if stream_replay:
        poll_stream = torch.cuda.Stream(device=engine_dev)
        _step_h = torch.empty(R, dtype=step.dtype, pin_memory=True)
        _step_h2 = torch.empty(R, dtype=step.dtype, pin_memory=True)
        _tok_h = torch.empty(1, dtype=tokens.dtype, pin_memory=True)
        _row_tok_h = torch.empty(max_seq, dtype=tokens.dtype, pin_memory=True)
        _row_prob_h = torch.empty(
            prob_buffer.shape[1], dtype=prob_buffer.dtype, pin_memory=True)

    def _poll_read(dst, src):
        # DtoH read of live engine state on a dedicated side stream. It must
        # be copy-engine-only: a read enqueued on the launch stream would
        # serialize behind the running megakernel, and any SM kernel
        # (gather/elementwise) would never be scheduled because every SM is
        # pinned by a persistent worker block. Contiguous slice -> pinned
        # host tensor is a pure cudaMemcpyAsync.
        with torch.cuda.stream(poll_stream):
            dst.copy_(src, non_blocking=True)
        poll_stream.synchronize()

    def _streamed_sample(r, plen, end):
        # Same extraction as the post-rollout read in rollout(); float32
        # bits survive the DtoH copy unchanged, so lp_old is bitwise the
        # value the post-sync read produces (verified per step below).
        _poll_read(_row_tok_h[:end], tokens[r, :end])
        if end > 1:
            _poll_read(_row_prob_h[:end - 1], prob_buffer[r, :end - 1])
        ids = _row_tok_h[:end].tolist()
        lp = []
        pos = []
        for t in range(plen, end):
            p = float(_row_prob_h[t - 1])
            if p > 0.0:
                lp.append(math.log(p))
                pos.append(t)
        return {"ids": ids, "plen": plen, "pos": pos, "lp_old": lp}

    # Wave-size floor: completed trajectories accumulate until at least
    # this many are ready (or the kernel exits), bounding the number of
    # separate replay graphs per step to ~4. Chunked prefill staggers
    # decode starts, so rows retire one-by-one ~4 engine iterations apart;
    # per-trajectory (batch-1) forwards fragment the later backward, which
    # measured 3-4x slower than the batched graph and dwarfs the overlap.
    stream_wave_min = max(1, R // 4)

    def _drain_completions(plen, done_ev, on_wave):
        # Poll step[]/tokens[] until every request has retired, emitting
        # completion WAVES of at least stream_wave_min trajectories (the
        # final flush may be smaller). Detection is torn-read safe: a row
        # is accepted only if step[r] is unchanged across two reads (or
        # the kernel already exited); a stale tokens[r, step[r]] read can
        # only MISS an eos (the buffer is zeroed at rollout start and
        # committed generated tokens are never eos before the final one),
        # delaying detection by one poll.
        pending = set(range(R))
        acc = []
        while pending:
            kernel_done = done_ev.query()
            _poll_read(_step_h, step)
            s1 = _step_h.tolist()
            cand = []
            for r in sorted(pending):
                sr = int(s1[r])
                if sr + 1 >= max_seq:
                    cand.append((r, sr))
                elif sr >= plen:
                    _poll_read(_tok_h, tokens[r, sr:sr + 1])
                    if int(_tok_h[0]) == engine_eos:
                        cand.append((r, sr))
            if cand and not kernel_done:
                _poll_read(_step_h2, step)
                s2 = _step_h2.tolist()
                cand = [(r, sr) for r, sr in cand if int(s2[r]) == sr]
            for r, sr in cand:
                pending.discard(r)
                acc.append((r, _streamed_sample(r, plen, sr + 1)))
            if acc and (len(acc) >= stream_wave_min or not pending):
                on_wave(sorted(acc))
                acc = []
            if pending:
                if kernel_done:
                    raise RuntimeError(
                        "megakernel exited but requests "
                        f"{sorted(pending)} do not satisfy the offline "
                        "retirement condition; completion polling is "
                        "inconsistent with the kernel")
                time.sleep(0.001)

    def rollout(prompt_ids, on_wave=None):
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
        if on_wave is not None:
            done_ev = torch.cuda.Event()
            done_ev.record()
            _drain_completions(plen, done_ev, on_wave)
        torch.cuda.synchronize()
        if on_wave is not None and disagg:
            # fold any straggling streamed trainer work into the rollout
            # wall time (it overlapped the decode tail on the other GPU)
            torch.cuda.synchronize(train_dev)
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
            # A trajectory that fills the sequence buffer (no eos before
            # the cap) cannot be fully teacher-forced: prefill needs one
            # free decode slot. Truncate the resubmitted prompt by one
            # token; the uncovered final position falls back to the frozen
            # lp_old below (its ratio contribution is exactly 1, so no bias
            # enters the surrogate).
            lengths = [min(len(s["ids"]), max_seq - 1) for s in samples]
            # Uniform prompt length across the group: pad shorter rows by
            # repeating their final token. Padding sits at positions >= the
            # row's own trajectory end, so causal attention cannot let it
            # change the captured P(token_t | prefix) for t < lengths[r].
            # This keeps the rescore prefill the same uniform-shape
            # workload as the rollout prefill; a ragged 16-way prompt batch
            # (mixed chunked-prefill/decode schedule) was observed to
            # deadlock the demo-mode megakernel.
            lmax = max(lengths)
            for r, s in enumerate(samples):
                ids = s["ids"][: lengths[r]]
                ids = ids + [ids[-1]] * (lmax - len(ids))
                tokens[r, :lmax] = torch.tensor(ids, device=dev)
            prompt_lengths.fill_(lmax)
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
            # authoritative values feed bind_forward_values against the
            # trainer's differentiable replay -> live on the trainer device
            lps.append(torch.tensor(lp, dtype=torch.float32, device=train_dev))
        return lps

    clip_eps = 0.2
    measure_old_recompute = getattr(args, "grpo_measure_old_recompute", False)

    if start_step >= len(data):
        raise ValueError(
            f"dataset provides {len(data)} prompts but the checkpoint has "
            f"already consumed {start_step}; nothing to resume"
        )
    # The dataset reader is deterministic and sequential, so the saved
    # data_cursor (== completed outer steps) is the full dataset state.
    for it in range(start_step, min(args.grpo_steps, len(data))):
        prompt_ids, gold = data[it]

        # Per-step streaming state: as completion waves arrive during the
        # rollout, compute each trajectory's reward (CPU) and its trainer
        # replay-forward (trainer GPU, graph retained) while the engine GPU
        # still decodes the stragglers. Only the forward is streamed; the
        # loss/backward/optimizer wait for all rewards (group-normalized
        # advantages) after the rollout.
        streamed = {}
        stream_stat = {"t_cb": 0.0, "waves": 0}

        def _on_wave(items, gold=gold, streamed=streamed, stat=stream_stat):
            cb0 = time.perf_counter()
            stat["waves"] += 1
            for r, sample in items:
                text = tokenizer.decode(sample["ids"][sample["plen"]:],
                                        skip_special_tokens=True)
                streamed[r] = {
                    "sample": sample,
                    "reward": 1.0 if extract_answer(text) == gold else 0.0,
                    "lp": None,
                }
            fwd = [(r, sample) for r, sample in items if sample["lp_old"]]
            if fwd:
                lps = backend.selected_token_logprobs(
                    [sample for _, sample in fwd])
                for (r, _), lp in zip(fwd, lps):
                    streamed[r]["lp"] = lp
            stat["t_cb"] += time.perf_counter() - cb0

        sync_devices()
        t0 = time.perf_counter()
        samples = rollout(prompt_ids,
                          on_wave=_on_wave if stream_replay else None)
        t_rollout = time.perf_counter() - t0
        t_rw0 = time.perf_counter()
        if stream_replay:
            # Semantics guard: pi_old capture and the streamed trainer
            # forward must have seen EXACTLY the trajectories the post-sync
            # read reports -- device placement must not change what enters
            # the objective.
            if sorted(streamed) != list(range(R)):
                raise RuntimeError(
                    f"streamed completions cover {sorted(streamed)} but the "
                    f"rollout has {R} requests")
            for r in range(R):
                a, b = streamed[r]["sample"], samples[r]
                if (a["ids"] != b["ids"] or a["plen"] != b["plen"]
                        or a["pos"] != b["pos"] or a["lp_old"] != b["lp_old"]):
                    raise RuntimeError(
                        f"streamed trajectory {r} diverges from the "
                        "post-rollout read; mid-rollout completion polling "
                        "returned torn state")
            rewards = [streamed[r]["reward"] for r in range(R)]
        else:
            rewards = []
            for s in samples:
                text = tokenizer.decode(s["ids"][s["plen"]:],
                                        skip_special_tokens=True)
                rewards.append(1.0 if extract_answer(text) == gold else 0.0)
        t_reward = time.perf_counter() - t_rw0
        # Bitwise fingerprint of the whole rollout group (token ids, prompt
        # length, capture positions, and exact rollout logprobs via float
        # repr round-trip); lets an interrupted+resumed run be compared
        # against an uninterrupted one from the step logs alone.
        rollout_digest = hashlib.sha256(json.dumps(
            [{"ids": s["ids"], "plen": s["plen"], "pos": s["pos"],
              "lp_old": s["lp_old"]} for s in samples],
            sort_keys=True,
        ).encode()).hexdigest()
        rw = torch.tensor(rewards, device=train_dev)
        adv = (rw - rw.mean()) / (rw.std() + 1e-6)

        # This pass is a counterfactual baseline, not part of the zero-TIM
        # engine.  Keep it opt-in so E2E timing does not charge MPK for work
        # that the design eliminates.
        t_old_recompute = 0.0
        if measure_old_recompute:
            old_recompute_samples = [s for s in samples if s["lp_old"]]
            sync_devices()
            t1 = time.perf_counter()
            with torch.no_grad():
                backend.selected_token_logprobs(old_recompute_samples)
            sync_devices()
            t_old_recompute = time.perf_counter() - t1

        sync_devices()
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
            if stream_replay and ep == 1:
                # epoch-1 replay forwards were streamed during the rollout
                # tail (trainer weights unchanged since the rollout, so the
                # graphs are valid for this update)
                trainer_lps = []
                for i in active:
                    lp = streamed[i]["lp"]
                    if lp is None:
                        raise RuntimeError(
                            f"active trajectory {i} has no streamed trainer "
                            "forward")
                    trainer_lps.append(lp)
            else:
                trainer_lps = backend.selected_token_logprobs(active_samples)
            for active_idx, i in enumerate(active):
                s = samples[i]
                lp_old = torch.tensor(s["lp_old"], dtype=torch.float32,
                                      device=train_dev)
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
                sync_devices()
                # sync trainer -> rollout weights (MPK reads them at the
                # next epoch's rescore / the next rollout); when the trainer
                # is disaggregated this is a fenced cross-device P2P copy —
                # the engine kernel has exited, so no mixed-version rollout
                # is possible
                t_sync_start = time.perf_counter()
                sync_report = sync_weights()
                sync_devices()
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
        sync_devices()
        t_train = time.perf_counter() - t2
        t_step = time.perf_counter() - t0
        rec = {
            "step": it,
            "reward_mean": float(rw.mean()),
            "rewards": rewards,
            "rollout_sha256": rollout_digest,
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
            "t_reward_s": round(t_reward, 6),
            "trainer_device": str(train_dev),
            "engine_device": str(engine_dev),
            "disaggregated": disagg,
            "streamed_replay_fwd": stream_replay,
            "stream_waves": stream_stat["waves"] if stream_replay else 0,
            "t_stream_cb_s": round(stream_stat["t_cb"], 4)
            if stream_replay else 0.0,
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
        if save_every and (it + 1) % save_every == 0:
            # Saved AFTER this step's optimizer update and trainer->engine
            # weight sync: the checkpoint state is exactly "it+1 outer steps
            # completed".  Resume re-loads the trainer, restores RNG, and
            # re-arms the engine through the same weight-sync step, so the
            # deterministic engine's position-keyed sampling reproduces the
            # uninterrupted run's rollouts bitwise from step it+1 onward.
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_step_{it + 1}.pt")
            t_ck0 = time.perf_counter()
            save_checkpoint(ckpt_path, {
                "outer_step": it + 1,
                "data_cursor": it + 1,
                "trainer_backend": args.grpo_trainer_backend,
                "trainer": backend.state_dict(),
                "rng": capture_rng_state(),
                "config": config_echo(args),
            })
            print(f"[e19] checkpoint step {it + 1} -> {ckpt_path} "
                  f"({time.perf_counter() - t_ck0:.2f}s)")
        if it % 5 == 0:
            print(f"[e19] {rec}")
    log_f.close()
    if close_backend is not None:
        close_backend()
        atexit.unregister(close_backend)
    print(f"[e19] DONE arm={arm} log={out_path}")
