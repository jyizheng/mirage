#!/bin/bash
# references_v2 gate suite for the Gumbel-spike clamp (sampling.cuh).
# Runs on an idle dev pod against /workspace/mirage-det:
#   phase A (pre-fix parent): seeded + greedy baseline runs
#   phase B (fix commit):     seeded x2 (restart between), greedy,
#                             16-member greedy group
# then refv2_analyze.py gates everything. Artifacts + RESULTS.txt in $OUT;
# $OUT/gate.done or $OUT/gate.fail marks completion.
#
# Usage: refv2_run.sh <gpu> <pre-commit> <post-commit>
set -u
GPU=${1:?gpu}
PRE=${2:?pre-fix commit}
POST=${3:?post-fix commit}
MODEL=Qwen/Qwen3-1.7B
PORT=8341
OUT=/workspace/refv2
REPO=/workspace/mirage-det

mkdir -p "$OUT"
rm -f "$OUT/gate.done" "$OUT/gate.fail"
cd "$REPO" || exit 1
export PYTHONPATH=$REPO/python
export MPK_DET_NUM_SPLITS=4

log() { echo "[$(date '+%F %T')] $*" >> "$OUT/driver.log"; }
die() { log "FAIL: $*"; git checkout -q fix-deterministic-decode; touch "$OUT/gate.fail"; exit 1; }

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

client() {  # output-name, mode, group-size, commit, then run
  python3 "$OUT/refv2_client.py" --port $PORT --model $MODEL \
    --output "$OUT/$1.json" --mode "$2" --group-size "$3" --commit "$4" \
    >> "$OUT/driver.log" 2>&1
}

# Scripts live at $POST; stage them outside the tree so the $PRE checkout
# can use them too.
git fetch origin fix-deterministic-decode >> "$OUT/driver.log" 2>&1 \
  || die "git fetch"
git checkout -q "$POST" || die "checkout POST for staging"
cp experiments/validation/refv2_client.py \
   experiments/validation/refv2_analyze.py "$OUT/" || die "stage scripts"

# ── phase A: pre-fix baselines ──────────────────────────────────────────────
git checkout -q "$PRE" || die "checkout PRE"
log "phase A on $(git rev-parse --short HEAD)"

start_server "$OUT/server_pre_seeded.log" --sampling-seed 42 \
  || die "pre seeded server"
client pre_seeded seeded 0 "$PRE" || die "pre seeded run"
stop_server

start_server "$OUT/server_pre_greedy.log" || die "pre greedy server"
client pre_greedy greedy 0 "$PRE" || die "pre greedy run"
stop_server

# ── phase B: post-fix gates ─────────────────────────────────────────────────
git checkout -q "$POST" || die "checkout POST"
log "phase B on $(git rev-parse --short HEAD)"

start_server "$OUT/server_post_seeded1.log" --sampling-seed 42 \
  || die "post seeded server 1"
client post_seeded_run1 seeded 0 "$POST" || die "post seeded run1"
stop_server

# Full server restart between repeats: the determinism claim covers fresh
# processes, not just request replay.
start_server "$OUT/server_post_seeded2.log" --sampling-seed 42 \
  || die "post seeded server 2"
client post_seeded_run2 seeded 0 "$POST" || die "post seeded run2"
stop_server

start_server "$OUT/server_post_greedy.log" || die "post greedy server"
client post_greedy greedy 0 "$POST" || die "post greedy run"
client post_group16 greedy 16 "$POST" || die "post group16 run"
stop_server

# ── gates ───────────────────────────────────────────────────────────────────
git checkout -q fix-deterministic-decode
if python3 "$OUT/refv2_analyze.py" --dir "$OUT" >> "$OUT/driver.log" 2>&1; then
  log "ALL GATES PASS"
  touch "$OUT/gate.done"
else
  log "GATES FAILED (see RESULTS.txt)"
  touch "$OUT/gate.fail"
  exit 1
fi
