#!/bin/bash
set -u
VENV=/workspace/sgl-venv
LOG=/tmp/e24.log

# MPK MoE det (GPU4), background
(cd /workspace/mirage && CUDA_VISIBLE_DEVICES=4 python demo/qwen3/demo_30B_A3B.py --use-mirage > /tmp/e24_mpk_moe.log 2>&1; echo "mpk-moe rc=$?" >> $LOG) &
MPKPID=$!

run_mode () {
  local name=$1; shift
  echo "=== sgl $name ===" >> $LOG
  CUDA_VISIBLE_DEVICES=3 $VENV/bin/python -m sglang.launch_server \
    --model-path Qwen/Qwen3-30B-A3B --port 8322 --host 127.0.0.1 \
    "$@" > /tmp/e24_server_$name.log 2>&1 &
  local spid=$!
  for i in $(seq 1 120); do
    sleep 10
    curl -s http://127.0.0.1:8322/health_generate > /dev/null 2>&1 && break
    kill -0 $spid 2>/dev/null || { echo "server died ($name)" >> $LOG; tail -3 /tmp/e24_server_$name.log >> $LOG; return 1; }
  done
  $VENV/bin/python - >> $LOG 2>&1 <<PYEOF
import json, time, urllib.request
prompt = ("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
          "<|im_start|>user\nGive me a short introduction to large language model.<|im_end|>\n"
          "<|im_start|>assistant\n")
def gen(n):
    body = json.dumps({"text": prompt, "sampling_params": {"temperature": 0, "max_new_tokens": n, "ignore_eos": True}}).encode()
    req = urllib.request.Request("http://127.0.0.1:8322/generate", data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.loads(r.read())
    return time.perf_counter()-t0, out["meta_info"]["completion_tokens"]
gen(16)
best = min(gen(256)[0]/256*1000 for _ in range(3))
print(f"per-token ms best-of-3: {best:.3f}")
PYEOF
  kill $spid 2>/dev/null; sleep 8; pkill -f sglang.launch_server; sleep 8
}

run_mode normal
run_mode det --enable-deterministic-inference --attention-backend triton
wait $MPKPID
echo "E24 DONE" >> $LOG
touch /tmp/e24.done
