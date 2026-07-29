#!/bin/bash
# E20 orchestrator: SGLang self-rescore delta_t, normal + deterministic.
set -u
export CUDA_VISIBLE_DEVICES=3
VENV=/workspace/sgl-venv
PORT=8322

run_mode () {
  local name=$1; shift
  echo "=== mode: $name ===" >> /tmp/e20.log
  $VENV/bin/python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B --port $PORT --host 127.0.0.1 \
    "$@" > /tmp/e20_server_$name.log 2>&1 &
  local spid=$!
  local ok=0
  for i in $(seq 1 90); do
    sleep 10
    if curl -s http://127.0.0.1:$PORT/health_generate > /dev/null 2>&1; then
      ok=1; echo "ready ~$((i*10))s" >> /tmp/e20.log; break
    fi
    kill -0 $spid 2>/dev/null || break
  done
  if [ $ok = 1 ]; then
    $VENV/bin/python /tmp/e20_dt.py "$name" >> /tmp/e20.log 2>&1
    if [ "$name" = det ]; then
      # cross-engine raw dump against the MPK greedy/sampled refs
      $VENV/bin/python /tmp/e20_cross.py /tmp/rescore_full2/ref.json cross_greedy; $VENV/bin/python /tmp/e20_cross.py /tmp/rescore_sampled3/ref.json cross_sampled >> /tmp/e20.log 2>&1 || true
    fi
  else
    echo "SERVER FAILED ($name)" >> /tmp/e20.log
    tail -5 /tmp/e20_server_$name.log >> /tmp/e20.log
  fi
  kill $spid 2>/dev/null; sleep 8
  pkill -f "sglang.launch_server" 2>/dev/null; sleep 8
}

run_mode normal --attention-backend triton
run_mode det --enable-deterministic-inference --attention-backend triton
echo "E20 ALL DONE" >> /tmp/e20.log
touch /tmp/e20.done
