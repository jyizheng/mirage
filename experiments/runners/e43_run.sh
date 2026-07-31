#!/bin/bash
set -u
CUDA_VISIBLE_DEVICES=4 /workspace/sgl-venv/bin/python -m sglang.launch_server \
  --model-path Qwen/Qwen3-1.7B --port 8322 --host 127.0.0.1 \
  > /tmp/e43_server.log 2>&1 &
SPID=$!
for i in $(seq 1 90); do
  sleep 10
  curl -s http://127.0.0.1:8322/health_generate >/dev/null 2>&1 && break
  kill -0 $SPID 2>/dev/null || { echo died >> /tmp/e43.log; break; }
done
cd /workspace/miles && PYTHONPATH=/workspace/miles /workspace/sgl-venv/bin/python \
  /tmp/e43_miles_baseline.py /tmp/e41_ref.json 8322 >> /tmp/e43.log 2>&1
echo "rc=$?" >> /tmp/e43.log
kill $SPID 2>/dev/null; pkill -f sglang.launch_server
touch /tmp/e43.done
