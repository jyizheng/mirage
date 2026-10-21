"""DeepSeek-V4.1 on MPK, offline mode: one request, decoded by MPK's own scheduler in a single launch.

The decode graph (models/deepseek_v41) keeps its own window / compressed caches, so the paged KV
pool only has to satisfy the scheduler: one page spanning the whole sequence.

  python demo/deepseek_v41/demo.py --ckpt /workspace/models/DeepSeek-V4.1-4layer \
      --prompt-ids /workspace/dsv41/golden_step/steps.pt --prompt-len 300 --gen 16
"""

import argparse
import json
import os
import time
from glob import glob

import torch
from safetensors.torch import load_file

import mirage
from mirage.mpk.models.deepseek_v41 import DeepSeekV41DecodeBuilder, engram_hash_tables
from mirage.mpk.persistent_kernel import PersistentKernel


def offline_meta(max_seq, prompt):
    i32 = dict(dtype=torch.int32, device="cuda")
    tokens = torch.zeros(1, max_seq, dtype=torch.int64, device="cuda")
    tokens[0, :len(prompt)] = torch.tensor(prompt, device="cuda")
    return {
        "step": torch.zeros(1, **i32),
        "tokens": tokens,
        "prompt_lengths": torch.tensor([len(prompt)], **i32),
        "input_tokens": torch.zeros(1, dtype=torch.int64, device="cuda"),
        "output_tokens": torch.zeros(1, dtype=torch.int64, device="cuda"),
        "num_new_tokens": torch.zeros(1, **i32),
        "qo_indptr_buffer": torch.zeros(2, **i32),
        "paged_kv_indptr_buffer_0": torch.zeros(2, **i32),
        "paged_kv_indices_buffer_0": torch.zeros(4, **i32),
        "paged_kv_last_page_len_buffer_0": torch.zeros(1, **i32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="", help="dir with tokenizer.json (default: --ckpt)")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--prompt-ids", default="", help="steps.pt from dsv41_ref_stepwise.py: use its prompt")
    ap.add_argument("--prompt-len", type=int, default=0)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--output-dir", default="/tmp/dsv41_offline")
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.ckpt, "config.json")))
    cfg = cfg.get("text_config", cfg)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer or a.ckpt)
    if a.prompt_ids:
        gold = torch.load(a.prompt_ids)
        prompt = gold["tokens"][:a.prompt_len or gold["prompt_len"]]
    else:
        prompt = tokenizer.encode(a.prompt)
    max_seq = a.max_seq_len

    sd = {}
    for shard in sorted(glob(os.path.join(a.ckpt, "*.safetensors"))):
        sd.update(load_file(shard, device="cuda"))
    meta = offline_meta(max_seq, prompt)
    nw, ns = mirage.get_configurations_from_gpu(0)
    pk = PersistentKernel(
        mode="offline", world_size=1, mpi_rank=0, num_workers=nw, num_local_schedulers=ns,
        num_remote_schedulers=0, max_seq_length=max_seq, max_num_batched_requests=1,
        max_num_batched_tokens=1, max_num_pages=4, page_size=max_seq, meta_tensors=meta,
        profiler_tensor=None, trace_name="", spec_decode_config=None, use_cutlass_kernel=False,
        eos_token_id=-1)
    b = DeepSeekV41DecodeBuilder(pk, cfg, max_seq_len=max_seq, tokens=meta["tokens"][0],
                                 step=meta["step"], next_token=meta["output_tokens"])
    b.build(sd, engram_hash_tables(cfg, tokenizer))
    del sd
    torch.cuda.empty_cache()
    t0 = time.time()
    pk.compile(output_dir=a.output_dir)
    print(f"compiled in {time.time() - t0:.0f}s", flush=True)

    torch.cuda.synchronize()
    t0 = time.time()
    pk()
    torch.cuda.synchronize()
    dt = time.time() - t0
    steps = int(meta["step"][0])
    gen = meta["tokens"][0, len(prompt):steps + 1].tolist()
    print(f"offline: {steps} steps ({len(prompt)} prompt + {len(gen)} generated) in {dt:.2f}s, "
          f"{dt / max(steps, 1) * 1e3:.2f} ms/step")
    print(f"generated ids[:16]: {gen[:16]}")
    print(f"text[:16]: {tokenizer.decode(gen[:16])!r}")
    pk.finalize()


if __name__ == "__main__":
    main()
