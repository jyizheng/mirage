#!/bin/bash
set -u
export CUDA_VISIBLE_DEVICES=3
VENV=/workspace/sgl-venv
LOG=/tmp/e22b.log
echo "=== default backend ===" >> $LOG
$VENV/bin/python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B --port 8322 --host 127.0.0.1 \
  --context-length 10240 > /tmp/e22b_server.log 2>&1 &
SPID=$!
for i in $(seq 1 90); do
  sleep 10
  curl -s http://127.0.0.1:8322/health_generate > /dev/null 2>&1 && break
  kill -0 $SPID 2>/dev/null || { echo died >> $LOG; exit 1; }
done
$VENV/bin/python /tmp/e22_len.py default >> $LOG 2>&1
kill $SPID 2>/dev/null; sleep 8; pkill -f sglang.launch_server
echo "E22B DONE" >> $LOG; touch /tmp/e22b.done
