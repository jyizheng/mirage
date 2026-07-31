#!/bin/bash
set -u
# MPK arm 8B (GPU5)
(cd /workspace/mirage-det && PYTHONPATH=/workspace/mirage-det/python CUDA_VISIBLE_DEVICES=5 \
  python demo/qwen3/demo.py --use-mirage --model Qwen/Qwen3-8B \
  --max-num-batched-requests 8 --deterministic --sampling-seed 42 \
  --capture-probs --grpo-steps 20 --grpo-arm mpk \
  --grpo-log /tmp/e38_mpk.jsonl > /tmp/e38_mpk.log 2>&1; echo "mpk8b rc=$?" >> /tmp/e38.log) &
MPID=$!
# SGLang arm 8B (GPU4)
CUDA_VISIBLE_DEVICES=4 /workspace/sgl-venv/bin/python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B --port 8323 --host 127.0.0.1 \
  > /tmp/e38_sgl_server.log 2>&1 &
SPID=$!
for i in $(seq 1 90); do
  sleep 10
  curl -s http://127.0.0.1:8323/health_generate > /dev/null 2>&1 && break
  kill -0 $SPID 2>/dev/null || { echo "sgl died" >> /tmp/e38.log; break; }
done
sed "s/PORT = 8322/PORT = 8323/; s/Qwen3-1.7B/Qwen3-8B/; s|/tmp/e32_sgl.jsonl|/tmp/e38_sgl.jsonl|" /tmp/e32_sgl.py > /tmp/e38_sgl.py
/workspace/sgl-venv/bin/python /tmp/e38_sgl.py > /tmp/e38_sgl.log 2>&1
echo "sgl8b rc=$?" >> /tmp/e38.log
kill $SPID 2>/dev/null; sleep 5; pkill -f "port 8323"
wait $MPID
touch /tmp/e38.done
