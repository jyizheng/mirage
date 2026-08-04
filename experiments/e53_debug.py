# E53 debug probe: instrumented submit_group at mbt16 to localize the hang.
# Prints per-member (rid, row, step) over time so we can tell apart:
#   ring never drained / rows never assigned  -> GPU admission stuck
#   rows assigned, steps frozen at P-1        -> task-graph hang on batch
#   steps advance, completion never signaled  -> completion-ring bug
# Also times the prefix-KV copy with an explicit synchronize (in a thread,
# so a copy that can never execute shows up as SYNC-HANG instead of
# blocking the probe).
import sys
import threading
import time

import torch

from mirage.engine.model_runner import ModelRunner, RunnerConfig
from mirage.engine.llm_engine import LLMEngine

MBT = int(sys.argv[1]) if len(sys.argv) > 1 else 16
GROUP = 16
PROMPT = ("Natalia sold clips to 48 of her friends in April, and then she "
          "sold half as many clips in May. How many clips did Natalia sell "
          "altogether in April and May?\nThink briefly, then give the final "
          "numeric answer after '####'.")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


config = RunnerConfig(
    model="Qwen/Qwen3-1.7B",
    max_num_batched_requests=16,
    max_num_batched_tokens=MBT,
    capture_logprobs=True,
    deterministic=True,
    sampling_seed=42,
    ignore_eos=True,
    max_seq_length=512,
    max_num_pages=16,
    page_size=4096,
)
log(f"building engine mbt={MBT} ...")
engine = LLMEngine(ModelRunner(config))
rt = engine.runtime

log("sanity: single submit ...")
r = engine.submit(PROMPT, use_template=False, timeout=300.0)
log(f"single ok, {len(r['token_ids'])} tokens")

token_ids = engine.tokenizer_manager.tokenize(PROMPT, False)
P = len(token_ids)
log(f"prompt_len={P}")
t = torch.tensor(token_ids, dtype=torch.int64)

# ── member 0 ──
rid0 = engine._next_rid
engine._next_rid += 1
with engine._submit_lock:
    rt.submit(rid0, t)
row0 = -1
while row0 < 0:
    row0 = rt.find_row_for_rid(rid0)
    time.sleep(1e-3)
log(f"member0 rid={rid0} row={row0}")
while rt.get_current_step_at_row(row0) < P:
    time.sleep(1e-3)
log(f"member0 prefill done, step={rt.get_current_step_at_row(row0)}")

meta = engine.model_runner.meta_tensors
src = int(meta["paged_kv_indices_buffer"][0].cpu().item())
log(f"member0 page src={src}")

builder = engine.model_runner.mpk.model_builder
k_cache, v_cache = builder.k_cache, builder.v_cache
n_pages = k_cache.shape[1]
num_layers = k_cache.shape[0]
pfx = P - 1

t0 = time.perf_counter()
for layer in range(num_layers):
    k_src = k_cache[layer, src, :pfx]
    v_src = v_cache[layer, src, :pfx]
    for dst in range(n_pages):
        if dst == src:
            continue
        k_cache[layer, dst, :pfx].copy_(k_src)
        v_cache[layer, dst, :pfx].copy_(v_src)
t_issue = time.perf_counter() - t0
log(f"copies issued in {t_issue:.3f}s; synchronizing ...")

sync_done = threading.Event()


def _sync():
    torch.cuda.synchronize()
    sync_done.set()


threading.Thread(target=_sync, daemon=True).start()
if sync_done.wait(30.0):
    log(f"copy sync OK, total {time.perf_counter() - t0:.3f}s")
else:
    log("SYNC-HANG: prefix-KV copies cannot execute while the megakernel "
        "holds the SMs — copies are NOT pure copy-engine ops here")

# verify one destination actually holds the prefix
probe = k_cache[0, (src + 1) % n_pages, : min(8, pfx)].float().cpu()
ref = k_cache[0, src, : min(8, pfx)].float().cpu()
log(f"dst-vs-src prefix match: {torch.equal(probe, ref)}")

# ── members 1..G-1 ──
rids = [rid0]
for i in range(GROUP - 1):
    rid = engine._next_rid
    engine._next_rid += 1
    with engine._submit_lock:
        accepted = rt.submit(rid, t, initial_step=pfx)
    rids.append(rid)
    log(f"member{i + 1} rid={rid} ring_accepted={accepted}")

# ── observe ──
for tick in range(40):
    states = []
    for rid in rids:
        row = rt.find_row_for_rid(rid)
        step = rt.get_current_step_at_row(row) if row >= 0 else -1
        states.append((rid, row, step))
    done = sorted(rt._completions.keys())
    log(f"tick{tick}: waiting={rt.waiting_count} completions={done}")
    log("  " + " ".join(f"{r}:{row}@{s}" for r, row, s in states))
    if len(done) >= GROUP:
        log("ALL COMPLETE")
        break
    time.sleep(3)

log("probe finished")
