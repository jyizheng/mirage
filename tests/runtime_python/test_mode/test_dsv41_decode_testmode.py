"""DeepSeek-V4.1 decode graph (models/deepseek_v41) against DeepSeek's reference, step by step.

Needs the bf16 checkpoint (DSV41_CKPT, e.g. the 4-layer slice) and a token-by-token reference run
(DSV41_GOLDEN_STEP, from dsv41_ref_stepwise.py: steps.pt + tokenizer/). The graph runs in test_mode,
one launch per position: the prompt is teacher-forced, then the graph decodes from its own argmax.
Every step compares the sublayer outputs and logits with the reference and the argmax token.
"""

import json
import os
import sys
import time
from glob import glob

import torch
from safetensors.torch import load_file

import mirage
from mirage.mpk.models.deepseek_v41 import DeepSeekV41DecodeBuilder, engram_hash_tables
from mirage.mpk.persistent_kernel import PersistentKernel


def rel(a, b):
    a, b = a.float().reshape(-1).cpu(), b.float().reshape(-1).cpu()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def main():
    ckpt, golden = os.environ["DSV41_CKPT"], os.environ["DSV41_GOLDEN_STEP"]
    max_steps = int(os.environ.get("DSV41_MAX_STEPS", "0"))
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    cfg = cfg.get("text_config", cfg)
    gold = torch.load(os.path.join(golden, "steps.pt"))
    tokens_ref, prompt_len, steps = gold["tokens"], gold["prompt_len"], gold["steps"]
    if max_steps:
        steps = steps[:max_steps]
    max_seq = int(os.environ.get("DSV41_MAX_SEQ", "512"))

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(golden, "tokenizer"))
    t0 = time.time()
    tables = engram_hash_tables(cfg, tokenizer)
    sd = {}
    for shard in sorted(glob(os.path.join(ckpt, "*.safetensors"))):
        sd.update(load_file(shard, device="cuda"))
    print(f"loaded {len(sd)} tensors + engram tables in {time.time() - t0:.0f}s", flush=True)

    nw, ns = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(test_mode=True, num_workers=nw, num_local_schedulers=ns, mpi_rank=0, world_size=1)
    pk = PersistentKernel(**params)
    b = DeepSeekV41DecodeBuilder(pk, cfg, max_seq_len=max_seq)
    t0 = time.time()
    b.build(sd, tables)
    del sd
    torch.cuda.empty_cache()
    print(f"built graph in {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    pk.compile(output_dir=os.path.dirname(os.path.abspath(__file__)))
    print(f"compiled in {time.time() - t0:.0f}s", flush=True)

    L, hc = cfg["num_hidden_layers"], cfg["hc_mult"]
    pairs = [("layers.0.engram", "l0_engram_out")] if 0 in cfg["engram_layer_ids"] else []
    for i in range(L):
        pairs += [(f"layers.{i}.attn", f"l{i}_attn_out"), (f"layers.{i}.ffn", f"l{i}_ffn_out"),
                  (f"layers.{i}", f"l{i}_stream_out")]
    pairs += [("norm", "final_norm")]

    tok = b.bufs["tokens"]

    def launch(pos):
        b.bufs["step"].fill_(pos)
        pk.meta_tensors["step"].zero_()
        pk.init_request_func()
        pk()
        torch.cuda.synchronize()
        return int(b.bufs["next_token"].item())

    # 1) teacher-forced over every step (prompt and the reference's greedy continuation), so each
    #    step's outputs are comparable with the reference's.
    tok[:len(tokens_ref)] = torch.tensor(tokens_ref, device="cuda")
    worst, agree, mpk_argmax = {}, 0, []
    t0 = time.time()
    for pos, ref in enumerate(steps):
        nxt = launch(pos)
        agree += nxt == ref["argmax"]
        mpk_argmax.append(nxt)
        errs = {g: rel(b.bufs[m], ref[g]) for g, m in pairs}
        errs["logits"] = rel(b.bufs["logits"], ref["logits"])
        for k, v in errs.items():
            if v > worst.get(k, (-1, 0))[0]:
                worst[k] = (v, pos)
        if pos < 2 or pos % 50 == 0 or errs["logits"] > 0.1:
            top = ", ".join(f"{k}={v:.2e}" for k, v in errs.items())
            print(f"step {pos}: argmax mpk={nxt} ref={ref['argmax']} | {top}", flush=True)
    n = len(steps)
    print(f"teacher-forced: {n} steps in {time.time() - t0:.0f}s; argmax agreement {agree}/{n}")
    topk = cfg["index_topk"]
    if n > topk:
        late = sum(mpk_argmax[p] == steps[p]["argmax"] for p in range(topk, n))
        print(f"  steps >= {topk} (indexer selecting in the ratio-1 layers): {late}/{n - topk}")
    if os.environ.get("DSV41_DUMP"):
        json.dump(mpk_argmax, open(os.environ["DSV41_DUMP"], "w"))
    print("worst relative error per output (value, step):")
    for k, (v, p) in worst.items():
        print(f"  {k:18s} {v:.3e} @ {p}")

    # 2) free-running from the prompt: the graph decodes from its own argmax.
    tok.zero_()
    tok[:prompt_len] = torch.tensor(tokens_ref[:prompt_len], device="cuda")
    generated = []
    for pos in range(len(steps)):
        nxt = launch(pos)
        if pos + 1 >= prompt_len:
            tok[pos + 1] = nxt
            generated.append(nxt)
    ref_gen = tokens_ref[prompt_len:prompt_len + len(generated)]
    same_prefix = next((j for j, (x, y) in enumerate(zip(generated, ref_gen)) if x != y), len(generated))
    print(f"generated mpk: {generated}")
    print(f"generated ref: {ref_gen}")
    print(f"text mpk: {tokenizer.decode(generated)!r}")
    print(f"text ref: {tokenizer.decode(ref_gen)!r}")
    print(f"free-running: first {same_prefix} generated tokens equal the reference's")
    pk.finalize()
    min_agree = float(os.environ.get("DSV41_MIN_AGREE", "0.75"))
    ok = agree >= min_agree * n
    print("PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
