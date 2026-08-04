# E35: serving-level rescore consistency, bitwise.
# Request A: rollout (greedy) through the ONLINE engine with capture.
# Request B: A's full token sequence resubmitted as a prompt -- the
# capture task teacher-forces every prompt position during prefill.
# Compare P(token t) float32 bit patterns over A's generated region.
import sys
sys.path.insert(0, "/workspace/mirage-det/python")
import numpy as np
import torch
from mirage.engine.model_runner import ModelRunner, RunnerConfig
from mirage.engine.llm_engine import LLMEngine

cfg = RunnerConfig(model="Qwen/Qwen3-8B", max_num_batched_requests=4,
                   capture_logprobs=True)
runner = ModelRunner(cfg)
engine = LLMEngine(runner)
rt = runner.runtime
pb = runner.prob_buffer

# ── A: rollout ──
res = engine.submit("Give me a short introduction to large language model.")
ids_a = None
# engine.submit released the completion; re-derive ids from result
# use the ENGINE's exact tokenization (its template settings), not ours
prompt_ids = engine.tokenizer_manager.tokenize(
    "Give me a short introduction to large language model.", True)
plen = len(prompt_ids)
full_ids = list(prompt_ids) + res["token_ids"]
end = len(full_ids)
# rollout row: capture wrote it; find via nonzero rows
buf_a = pb.cpu().numpy().copy()
rows = np.where(np.abs(buf_a).sum(1) > 0)[0]
print("[e35] rollout rows nonzero:", rows.tolist(), "gen:", len(res["token_ids"]), flush=True)
row_a = int(rows[0])
roll_bits = buf_a[row_a].view(np.int32)[plen - 1:end - 1].copy()

# ── B: rescore (full sequence as prompt) ──
t = torch.tensor(full_ids, dtype=torch.int64)
rt.submit(7, t)
row_b, final_b = rt.wait_for_request(7, timeout=600.0)
rt.release_request(7)
buf_b = pb.cpu().numpy().copy()
res_bits = buf_b[row_b].view(np.int32)[plen - 1:end - 1].copy()
print("[e35] rescore row:", row_b, "final_step:", final_b, flush=True)

n = len(roll_bits)
mism = int((roll_bits != res_bits).sum())
print(f"[e35] SERVING RESCORE: {n - mism}/{n} bitwise-identical, {mism} mismatches", flush=True)
if mism:
    bad = np.where(roll_bits != res_bits)[0][:5]
    for i in bad:
        print(f"  slot {plen-1+i}: roll {roll_bits[i]:#010x} vs rescore {res_bits[i]:#010x}", flush=True)
engine.close()
print("[e35] DONE", flush=True)
