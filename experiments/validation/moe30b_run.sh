#!/bin/bash
# Qwen3-30B-A3B MoE online determinism gate suite (design phases b+c).
# Runs on an idle dev pod against /workspace/mirage-det at HEAD:
#   session 1  greedy 30B: greedy run1+run2, 16-member greedy group,
#              batch-invariance solo + concurrent   (gates i, iii, iv)
#   session 2  greedy 30B, fresh process: greedy run3 (gate i, restart leg)
#   session 3  seeded 30B (--sampling-seed 42): seeded run1 (gate ii)
#   session 4  seeded 30B, fresh process: seeded run2 (gate ii)
#   session 5  dense 1.7B greedy: run1+run2 (gate vi regression)
#   in-process rescore == rollout (gate v, RQ4 MoE row)
# then moe30b_analyze.py gates the HTTP runs. Artifacts + RESULTS.txt in
# $OUT; $OUT/gate.done or $OUT/gate.fail marks completion.
#
# Usage: moe30b_run.sh <gpu>
set -u
GPU=${1:?gpu}
MOE_MODEL=Qwen/Qwen3-30B-A3B
DENSE_MODEL=Qwen/Qwen3-1.7B
PORT=8342
OUT=/workspace/moe30b
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

start_server() {  # logfile, model, extra args...
  local LOG=$1 MODEL=$2; shift 2
  CUDA_VISIBLE_DEVICES=$GPU python3 -m mirage.engine.launch_server \
    --model "$MODEL" --port $PORT --deterministic --capture-logprobs \
    --ignore-eos --max-seq-length 512 --max-num-batched-requests 16 \
    --max-num-batched-tokens 16 --max-num-pages 16 \
    --pinned-ring-capacity 32 --request-timeout 1800 \
    "$@" > "$LOG" 2>&1 &
  SPID=$!
  local i
  for i in $(seq 1 360); do
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
  python3 "$VAL/moe30b_client.py" --port $PORT --model $MOE_MODEL \
    --output "$OUT/$1.json" --mode "$2" --group-size "$3" \
    --commit "$COMMIT" >> "$OUT/driver.log" 2>&1
}

# ── session 1: greedy 30B (gates i, iii, iv) ────────────────────────────────
log "suite start on $COMMIT (gpu $GPU)"
start_server "$OUT/server_greedy1.log" $MOE_MODEL || die "greedy server 1"
log "greedy server 1 up"
client moe_greedy_run1 greedy 0 || die "greedy run1"
client moe_greedy_run2 greedy 0 || die "greedy run2"
client moe_group16 greedy 16 || die "group16 run"
python3 "$VAL/moe30b_batch_client.py" --port $PORT --model $MOE_MODEL \
  --output "$OUT/moe_batch_solo.json" --phase solo --commit "$COMMIT" \
  >> "$OUT/driver.log" 2>&1 || die "batch solo"
python3 "$VAL/moe30b_batch_client.py" --port $PORT --model $MOE_MODEL \
  --output "$OUT/moe_batch_concurrent.json" --phase batch --commit "$COMMIT" \
  >> "$OUT/driver.log" 2>&1 || die "batch concurrent"
stop_server
log "session 1 done"

# ── session 2: greedy 30B, fresh process (gate i restart leg) ───────────────
start_server "$OUT/server_greedy2.log" $MOE_MODEL || die "greedy server 2"
client moe_greedy_run3 greedy 0 || die "greedy run3"
stop_server
log "session 2 done"

# ── sessions 3+4: seeded 30B across restart (gate ii) ───────────────────────
start_server "$OUT/server_seeded1.log" $MOE_MODEL --sampling-seed 42 \
  || die "seeded server 1"
client moe_seeded_run1 seeded 0 || die "seeded run1"
stop_server
start_server "$OUT/server_seeded2.log" $MOE_MODEL --sampling-seed 42 \
  || die "seeded server 2"
client moe_seeded_run2 seeded 0 || die "seeded run2"
stop_server
log "sessions 3+4 done"

# ── session 5: dense 1.7B greedy regression (gate vi) ───────────────────────
start_server "$OUT/server_dense.log" $DENSE_MODEL || die "dense server"
python3 "$VAL/moe30b_client.py" --port $PORT --model $DENSE_MODEL \
  --output "$OUT/dense_greedy_run1.json" --mode greedy --group-size 0 \
  --commit "$COMMIT" >> "$OUT/driver.log" 2>&1 || die "dense run1"
python3 "$VAL/moe30b_client.py" --port $PORT --model $DENSE_MODEL \
  --output "$OUT/dense_greedy_run2.json" --mode greedy --group-size 0 \
  --commit "$COMMIT" >> "$OUT/driver.log" 2>&1 || die "dense run2"
stop_server
log "session 5 done"

# ── gate v: rescore == rollout, in-process (RQ4 MoE row) ────────────────────
CUDA_VISIBLE_DEVICES=$GPU python3 "$VAL/moe30b_rescore.py" \
  > "$OUT/rescore.log" 2>&1
RESCORE_RC=$?
log "rescore rc=$RESCORE_RC"

# ── gates ───────────────────────────────────────────────────────────────────
python3 "$VAL/moe30b_analyze.py" --dir "$OUT" >> "$OUT/driver.log" 2>&1
HTTP_RC=$?
{ echo "gate v_rescore_eq_rollout: $([ $RESCORE_RC -eq 0 ] \
    && echo PASS || echo FAIL)";
  grep -a "SERVING RESCORE" "$OUT/rescore.log"; } >> "$OUT/RESULTS.txt"
if [ $HTTP_RC -eq 0 ] && [ $RESCORE_RC -eq 0 ]; then
  log "ALL GATES PASS"
  touch "$OUT/gate.done"
else
  log "GATES FAILED (see RESULTS.txt)"
  touch "$OUT/gate.fail"
  exit 1
fi
