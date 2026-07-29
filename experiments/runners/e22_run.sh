#!/bin/bash
set -u
export CUDA_VISIBLE_DEVICES=3
VENV=/workspace/sgl-venv
LOG=/tmp/e22.log
while [ ! -f /tmp/e21b.done ]; do sleep 20; done
sleep 15
run () {
  echo "=== $1 ===" >> $LOG; local tag=$1; shift
  $VENV/bin/python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B --port 8322 --host 127.0.0.1 \
    --context-length 10240 "$@" > /tmp/e22_server.log 2>&1 &
  SPID=$!
  for i in $(seq 1 90); do
    sleep 10
    curl -s http://127.0.0.1:8322/health_generate > /dev/null 2>&1 && break
    kill -0 $SPID 2>/dev/null || { echo "server died" >> $LOG; tail -3 /tmp/e22_server.log >> $LOG; return 1; }
  done
  $VENV/bin/python /tmp/e22_len.py $tag >> $LOG 2>&1
  kill $SPID 2>/dev/null; sleep 8; pkill -f sglang.launch_server; sleep 8
}
run normal --attention-backend triton
run det --enable-deterministic-inference --attention-backend triton
echo "E22 DONE" >> $LOG
touch /tmp/e22.done
