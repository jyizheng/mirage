#!/bin/bash
set -u
# MPK arm (GPU5): e19 GRPO with timing, demo-path capture
(cd /workspace/mirage-det && PYTHONPATH=/workspace/mirage-det/python CUDA_VISIBLE_DEVICES=5 \
  python demo/qwen3/demo.py --use-mirage --model Qwen/Qwen3-1.7B \
  --max-num-batched-requests 8 --deterministic --sampling-seed 42 \
  --capture-probs --grpo-steps 20 --grpo-arm mpk \
  --grpo-log /tmp/e32_mpk.jsonl > /tmp/e32_mpk.log 2>&1; echo "mpk rc=$?" >> /tmp/e32.log) &
MPID=$!
# SGLang arm (GPU4): server + recompute deployment
CUDA_VISIBLE_DEVICES=4 /workspace/sgl-venv/bin/python -m sglang.launch_server \
  --model-path Qwen/Qwen3-1.7B --port 8322 --host 127.0.0.1 \
  > /tmp/e32_sgl_server.log 2>&1 &
SPID=$!
for i in $(seq 1 90); do
  sleep 10
  curl -s http://127.0.0.1:8322/health_generate > /dev/null 2>&1 && break
  kill -0 $SPID 2>/dev/null || { echo "sgl died" >> /tmp/e32.log; break; }
done
/workspace/sgl-venv/bin/python /tmp/e32_sgl.py > /tmp/e32_sgl.log 2>&1
echo "sgl rc=$?" >> /tmp/e32.log
kill $SPID 2>/dev/null; sleep 5; pkill -f "sglang.launch_serve[r]"
wait $MPID
touch /tmp/e32.done
