"""DeepSeek-V4.1 on MPK, online_pinned mode, through mirage.engine.ModelRunner (the path an
external serving harness uses). Submits the same prompt twice, back to back, and checks both
requests decode the same tokens: the second one runs on caches the first one left behind.

  python demo/deepseek_v41/online.py --ckpt /workspace/models/DeepSeek-V4.1-4layer \
      --prompt-ids /workspace/dsv41/golden_step/steps.pt --expect 127810,96933,111515
"""

import argparse
import threading
import time

import torch

from mirage.engine.model_runner import ModelRunner, RunnerConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt-ids", required=True, help="steps.pt from dsv41_ref_stepwise.py")
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--requests", type=int, default=2)
    ap.add_argument("--expect", default="", help="comma-separated expected first generated ids")
    ap.add_argument("--output-dir", default="/tmp/dsv41_online")
    a = ap.parse_args()

    gold = torch.load(a.prompt_ids)
    prompt = torch.tensor(gold["tokens"][:gold["prompt_len"]], dtype=torch.int64)
    runner = ModelRunner(RunnerConfig(
        model=a.ckpt, max_num_batched_requests=1, max_num_batched_tokens=1,
        max_seq_length=a.max_seq_len, max_num_pages=4, page_size=a.max_seq_len,
        output_dir=a.output_dir, use_cutlass_kernel=False))
    rt = runner.runtime
    rt.reset()
    rt.start()
    kernel = threading.Thread(target=runner, daemon=True)
    kernel.start()

    outs = []
    for rid in range(a.requests):
        t0 = time.time()
        rt.submit(rid, prompt)
        row, final_step = rt.wait_for_request(rid, 300.0, 1e-4)
        dt = time.time() - t0
        toks = rt.read_tokens_at_row(row, final_step)[len(prompt):].tolist()
        rt.release_request(rid)
        outs.append(toks)
        print(f"request {rid}: {len(toks)} tokens in {dt:.2f}s (final_step {final_step}); "
              f"first 16 {toks[:16]}; text {runner.tokenizer.decode(toks[:16])!r}", flush=True)
    rt.request_shutdown()
    kernel.join(timeout=60)
    ok = all(o == outs[0] for o in outs)
    if a.expect:
        want = [int(x) for x in a.expect.split(",")]
        ok &= outs[0][:len(want)] == want
    print("PASSED" if ok else "FAILED")


if __name__ == "__main__":
    main()
