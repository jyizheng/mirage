#!/bin/bash
set -u
LOG=/tmp/e24b.log
(cd /workspace/mirage && CUDA_VISIBLE_DEVICES=4 python demo/qwen3/demo_30B_A3B.py --use-mirage --max-num-batched-tokens 8 > /tmp/e24b_mpk_moe_mbt8.log 2>&1; echo "mpk-moe-mbt8 rc=$?" >> $LOG) &
MPID=$!
echo "=== sgl det no-cuda-graph ===" >> $LOG
CUDA_VISIBLE_DEVICES=3 /workspace/sgl-venv/bin/python -m sglang.launch_server \
  --model-path Qwen/Qwen3-30B-A3B --port 8322 --host 127.0.0.1 \
  --enable-deterministic-inference --attention-backend triton \
  --disable-cuda-graph > /tmp/e24b_server_det.log 2>&1 &
SPID=$!
ok=0
for i in $(seq 1 120); do
  sleep 10
  curl -s http://127.0.0.1:8322/health_generate > /dev/null 2>&1 && { ok=1; break; }
  kill -0 $SPID 2>/dev/null || { echo "det no-graph server died too" >> $LOG; tail -8 /tmp/e24b_server_det.log >> $LOG; break; }
done
if [ $ok = 1 ]; then
  /workspace/sgl-venv/bin/python - >> $LOG 2>&1 <<PYEOF
import json, time, urllib.request
prompt = ("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
          "<|im_start|>user\nGive me a short introduction to large language model.<|im_end|>\n"
          "<|im_start|>assistant\n")
def gen(n):
    body = json.dumps({"text": prompt, "sampling_params": {"temperature": 0, "max_new_tokens": n, "ignore_eos": True}}).encode()
    req = urllib.request.Request("http://127.0.0.1:8322/generate", data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        json.loads(r.read())
    return time.perf_counter()-t0
gen(16)
best = min(gen(256)/256*1000 for _ in range(3))
print(f"det-no-graph per-token ms best-of-3: {best:.3f}")
PYEOF
fi
kill $SPID 2>/dev/null; pkill -f sglang.launch_server; wait $MPID
echo "E24B DONE" >> $LOG; touch /tmp/e24b.done
