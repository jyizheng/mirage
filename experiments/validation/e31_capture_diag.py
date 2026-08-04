import sys
sys.path.insert(0, "/workspace/mirage-det/python")
import numpy as np
import torch
from mirage.engine.model_runner import ModelRunner, RunnerConfig
from mirage.engine.llm_engine import LLMEngine

cfg = RunnerConfig(model="Qwen/Qwen3-8B", max_num_batched_requests=4,
                   capture_logprobs=True)
runner = ModelRunner(cfg)
print("[e31] prob_buffer:", tuple(runner.prob_buffer.shape),
      hex(runner.prob_buffer.data_ptr()), flush=True)
print("[e31] in _model_tensors:",
      "prob_buffer" in runner.mpk.persistent_kernel._model_tensors, flush=True)

engine = LLMEngine(runner)
res = engine.submit("Give me a short introduction to large language model.")
n = len(res["token_ids"])
lp_finite = sum(1 for x in res.get("logprobs", []) if x is not None)
print("[e31] generated:", n, "finite logprobs:", lp_finite, flush=True)

# persistent kernel occupies the SMs: only copy-engine ops are safe here
buf = runner.prob_buffer.cpu().numpy()
print("[e31] buffer |sum|:", float(np.abs(buf).sum()),
      "nonzero rows:", np.where(np.abs(buf).sum(1) > 0)[0].tolist(), flush=True)
nz = np.nonzero(buf[0])[0]
print("[e31] row0 nonzero slots:", nz[:12].tolist(), "count:", len(nz), flush=True)
print("[e31] row0[:8]:", buf[0, :8].tolist(), flush=True)
print("[e31] row0 tail (diag) [-8:]:", buf[0, -8:].tolist(), flush=True)
print("[e31] diag: exec_count", buf[0,-1], "nt0", buf[0,-2], "sl0", buf[0,-3], "plen0", buf[0,-4], "step0", buf[0,-5], "writes", buf[0,-6], flush=True)
engine.close()
print("[e31] DONE", flush=True)
