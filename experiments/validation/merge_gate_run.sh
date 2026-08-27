#!/bin/bash
# Merge-head determinism gates + references_v2 regeneration (dense 1.7B).
# Unlike refv2_run.sh this has no pre/post phases: it runs entirely at the
# current HEAD and gates
#   1. seeded x2 bitwise across a full server restart
#   2. greedy x2 bitwise across a full server restart
#   3. all 16 greedy group members bitwise match the single greedy run
# and reports (informationally) whether the trajectories differ from the
# committed experiments/data/references_v2 JSONs — expected after a
# numerics-changing merge (e.g. upstream #755 lm_head pad -1e4).
# Artifacts + RESULTS.txt in $OUT; $OUT/gate.done or $OUT/gate.fail marks
# completion.
#
# Usage: merge_gate_run.sh <gpu>
set -u
GPU=${1:?gpu}
MODEL=Qwen/Qwen3-1.7B
PORT=8343
OUT=/workspace/mergegate
REPO=/workspace/mirage-det
VAL=$REPO/experiments/validation

mkdir -p "$OUT"
rm -f "$OUT/gate.done" "$OUT/gate.fail"
cd "$REPO" || exit 1
export PYTHONPATH=$REPO/python
export MPK_DET_NUM_SPLITS=4
COMMIT=$(git rev-parse --short HEAD)

log() { echo "[$(date '+%F %T')] $*" >> "$OUT/driver.log"; }
die() { log "FAIL: $*"; touch "$OUT/gate.fail"; exit 1; }

start_server() {  # logfile, extra args...
  local LOG=$1; shift
  CUDA_VISIBLE_DEVICES=$GPU python3 -m mirage.engine.launch_server \
    --model $MODEL --port $PORT --deterministic --capture-logprobs \
    --ignore-eos --max-seq-length 512 --max-num-batched-requests 16 \
    --max-num-batched-tokens 16 --max-num-pages 16 \
    --pinned-ring-capacity 32 --request-timeout 900 \
    "$@" > "$LOG" 2>&1 &
  SPID=$!
  local i
  for i in $(seq 1 240); do
    sleep 10
    grep -q "Uvicorn running" "$LOG" && { sleep 5; return 0; }
    kill -0 $SPID 2>/dev/null || return 1
  done
  return 1
}

stop_server() {
  kill -9 $SPID 2>/dev/null
  sleep 5
  pkill -9 -f "launch_serve[r] .*--port $PORT" 2>/dev/null
  sleep 3
}

client() {  # output-name, mode, group-size
  python3 "$VAL/refv2_client.py" --port $PORT --model $MODEL \
    --output "$OUT/$1.json" --mode "$2" --group-size "$3" \
    --commit "$COMMIT" >> "$OUT/driver.log" 2>&1
}

log "merge gate start on $COMMIT (gpu $GPU)"

start_server "$OUT/server_seeded1.log" --sampling-seed 42 \
  || die "seeded server 1"
client seeded_run1 seeded 0 || die "seeded run1"
stop_server

start_server "$OUT/server_seeded2.log" --sampling-seed 42 \
  || die "seeded server 2"
client seeded_run2 seeded 0 || die "seeded run2"
stop_server

start_server "$OUT/server_greedy1.log" || die "greedy server 1"
client greedy_run1 greedy 0 || die "greedy run1"
client group16 greedy 16 || die "group16 run"
stop_server

start_server "$OUT/server_greedy2.log" || die "greedy server 2"
client greedy_run2 greedy 0 || die "greedy run2"
stop_server

if python3 "$VAL/merge_gate_analyze.py" --dir "$OUT" \
     --refs "$REPO/experiments/data/references_v2" \
     >> "$OUT/driver.log" 2>&1; then
  log "ALL GATES PASS"
  touch "$OUT/gate.done"
else
  log "GATES FAILED (see RESULTS.txt)"
  touch "$OUT/gate.fail"
  exit 1
fi
