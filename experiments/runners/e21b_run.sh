#!/bin/bash
set -u
export CUDA_VISIBLE_DEVICES=3
VENV=/workspace/sgl-venv
LOG=/tmp/e21b.log
run () {
  echo "=== $1 ===" >> $LOG; shift
  $VENV/bin/python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B --port 8322 --host 127.0.0.1 "$@" \
    > /tmp/e21b_server.log 2>&1 &
  SPID=$!
  for i in $(seq 1 90); do
    sleep 10
    curl -s http://127.0.0.1:8322/health_generate > /dev/null 2>&1 && break
    kill -0 $SPID 2>/dev/null || { echo "server died" >> $LOG; return 1; }
  done
  $VENV/bin/python /tmp/e21_throughput.py 8322 sgl >> $LOG 2>&1
  kill $SPID 2>/dev/null; sleep 8; pkill -f sglang.launch_server; sleep 8
}
run "sgl normal" --attention-backend triton
run "sgl det" --enable-deterministic-inference --attention-backend triton
echo "E21B DONE" >> $LOG
touch /tmp/e21b.done
