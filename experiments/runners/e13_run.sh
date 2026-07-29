#!/bin/bash
# E13 orchestrator: three server configs, same client protocol, GPU 3.
set -u
export CUDA_VISIBLE_DEVICES=3
SGLV=/workspace/sgl-venv/bin/python

wait_http () { # port, pid
  for i in $(seq 1 90); do
    sleep 10
    curl -s "http://127.0.0.1:$1/$3" > /dev/null 2>&1 && return 0
    kill -0 $2 2>/dev/null || return 1
  done
  return 1
}

echo "=== MPK-Det server ==="
cd /workspace/mirage
MPK_DETERMINISTIC=1 python -m mirage.engine.launch_server --port 8331 \
  --max-num-batched-requests 8 > /tmp/e13_mpk.log 2>&1 &
PID=$!
if wait_http 8331 $PID v1/models || grep -q "Uvicorn running" /tmp/e13_mpk.log; then
  sleep 5; python3 /tmp/e13_throughput.py 8331 mpk
else
  echo "MPK SERVER FAILED"; tail -3 /tmp/e13_mpk.log
fi
kill $PID 2>/dev/null; pkill -f "mirage.engine.launch_server"; sleep 8

echo "=== SGLang normal ==="
$SGLV -m sglang.launch_server --model-path Qwen/Qwen3-8B --port 8332 \
  --host 127.0.0.1 > /tmp/e13_sgln.log 2>&1 &
PID=$!
wait_http 8332 $PID health_generate && $SGLV /tmp/e13_throughput.py 8332 sgl \
  || { echo "SGL NORMAL FAILED"; tail -3 /tmp/e13_sgln.log; }
kill $PID 2>/dev/null; pkill -f sglang.launch_server; sleep 8

echo "=== SGLang deterministic (triton) ==="
$SGLV -m sglang.launch_server --model-path Qwen/Qwen3-8B --port 8333 \
  --host 127.0.0.1 --enable-deterministic-inference --attention-backend triton \
  > /tmp/e13_sgld.log 2>&1 &
PID=$!
wait_http 8333 $PID health_generate && $SGLV /tmp/e13_throughput.py 8333 sgl \
  || { echo "SGL DET FAILED"; tail -3 /tmp/e13_sgld.log; }
kill $PID 2>/dev/null; pkill -f sglang.launch_server
echo "E13 COMPLETE"
