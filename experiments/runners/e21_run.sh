#!/bin/bash
# E21: cross-engine delta_t dumps (det SGLang server), then throughput
# rerun with the flush_waiting fix: MPK-Det, SGLang normal, SGLang det.
set -u
export CUDA_VISIBLE_DEVICES=3
VENV=/workspace/sgl-venv
LOG=/tmp/e21.log

sgl_up () {
  $VENV/bin/python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B --port 8322 --host 127.0.0.1 \
    "$@" > /tmp/e21_server.log 2>&1 &
  SPID=$!
  for i in $(seq 1 90); do
    sleep 10
    curl -s http://127.0.0.1:8322/health_generate > /dev/null 2>&1 && return 0
    kill -0 $SPID 2>/dev/null || return 1
  done
  return 1
}
sgl_down () { kill $SPID 2>/dev/null; sleep 8; pkill -f sglang.launch_server; sleep 8; }

echo "=== cross-engine dt (det server) ===" >> $LOG
if sgl_up --enable-deterministic-inference --attention-backend triton; then
  $VENV/bin/python /tmp/e20_cross.py /tmp/rescore_full2/ref.json cross_greedy >> $LOG 2>&1
  $VENV/bin/python /tmp/e20_cross.py /tmp/rescore_sampled3/ref.json cross_sampled >> $LOG 2>&1
  echo "--- sgl det throughput ---" >> $LOG
  $VENV/bin/python /tmp/e21_throughput.py 8322 sgl >> $LOG 2>&1
else
  echo "det server failed" >> $LOG; tail -5 /tmp/e21_server.log >> $LOG
fi
sgl_down

echo "=== sgl normal throughput ===" >> $LOG
if sgl_up --attention-backend triton; then
  $VENV/bin/python /tmp/e21_throughput.py 8322 sgl >> $LOG 2>&1
fi
sgl_down

echo "=== MPK-Det server throughput (post-fix) ===" >> $LOG
cd /workspace/mirage
MPK_DETERMINISTIC=1 python -m mirage.engine.launch_server --port 8331 \
  --max-num-batched-requests 8 > /tmp/e21_mpk.log 2>&1 &
MPID=$!
ok=0
for i in $(seq 1 120); do
  sleep 10
  curl -s http://127.0.0.1:8331/v1/models > /dev/null 2>&1 && { ok=1; break; }
  grep -q "Uvicorn running" /tmp/e21_mpk.log && { ok=1; break; }
  kill -0 $MPID 2>/dev/null || break
done
if [ $ok = 1 ]; then
  sleep 5
  python3 /tmp/e21_throughput.py 8331 mpk >> $LOG 2>&1
else
  echo "MPK SERVER FAILED" >> $LOG; tail -5 /tmp/e21_mpk.log >> $LOG
fi
kill $MPID 2>/dev/null; pkill -f "mirage.engine.launch_server"; sleep 5
echo "E21 ALL DONE" >> $LOG
touch /tmp/e21.done
