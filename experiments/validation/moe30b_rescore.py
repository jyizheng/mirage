#!/usr/bin/env python3
"""Gate (v): serving-level rescore == rollout for Qwen3-30B-A3B (MoE),
bitwise.  This is the RQ4 row refresh for the MoE online path.

Same shape as e35_serving_rescore.py, on the deterministic MoE engine:
  Request A: rollout (greedy) through the ONLINE engine with capture.
  Request B: A's full token sequence resubmitted as a prompt -- the
  capture task teacher-forces every prompt position during prefill.
Compare P(token t) float32 bit patterns over A's generated region.
"""
import os
import sys

os.environ.setdefault("MPK_DET_NUM_SPLITS", "4")
sys.path.insert(0, "/workspace/mirage-det/python")

import numpy as np
import torch

from mirage.engine.model_runner import ModelRunner, RunnerConfig
from mirage.engine.llm_engine import LLMEngine

cfg = RunnerConfig(model="Qwen/Qwen3-30B-A3B",
                   max_num_batched_requests=16,
                   max_num_batched_tokens=16,
                   pinned_ring_capacity=32,
                   deterministic=True,
                   capture_logprobs=True)
runner = ModelRunner(cfg)
engine = LLMEngine(runner)
rt = runner.runtime
pb = runner.prob_buffer

# -- A: rollout --
PROMPT = "Give me a short introduction to large language model."
res = engine.submit(PROMPT, timeout=1800.0)
prompt_ids = engine.tokenizer_manager.tokenize(PROMPT, True)
plen = len(prompt_ids)
full_ids = list(prompt_ids) + res["token_ids"]
end = len(full_ids)
buf_a = pb.cpu().numpy().copy()
rows = np.where(np.abs(buf_a).sum(1) > 0)[0]
print("[moe-rescore] rollout rows nonzero:", rows.tolist(),
      "gen:", len(res["token_ids"]), flush=True)
row_a = int(rows[0])
roll_bits = buf_a[row_a].view(np.int32)[plen - 1:end - 1].copy()

# -- B: rescore (full sequence as prompt) --
t = torch.tensor(full_ids, dtype=torch.int64)
rt.submit(7, t)
row_b, final_b = rt.wait_for_request(7, timeout=1800.0)
rt.release_request(7)
buf_b = pb.cpu().numpy().copy()
res_bits = buf_b[row_b].view(np.int32)[plen - 1:end - 1].copy()
print("[moe-rescore] rescore row:", row_b, "final_step:", final_b, flush=True)

n = len(roll_bits)
mism = int((roll_bits != res_bits).sum())
print(f"[moe-rescore] SERVING RESCORE: {n - mism}/{n} bitwise-identical, "
      f"{mism} mismatches", flush=True)
if mism:
    bad = np.where(roll_bits != res_bits)[0][:5]
    for i in bad:
        print(f"  slot {plen - 1 + i}: roll {roll_bits[i]:#010x} "
              f"vs rescore {res_bits[i]:#010x}", flush=True)
engine.close()
print("[moe-rescore] " + ("PASS" if mism == 0 else "FAIL"), flush=True)
sys.exit(0 if mism == 0 else 1)
