#!/bin/bash
# E53 full validation v2 (post-fix): mbt8 A/B, mbt16 A/B, greedy bitwise
# gate. All phases redone — the pre-fix mbt8 number was correctness-
# tainted (stale-step bug copied the wrong page's KV). Hardened:
# --pinned-ring-capacity 32 (lockstep completions overflow an 8-slot
# ring), --request-timeout 900, kill -9 cleanup, per-phase .ok markers.
set -u
GPU=${1:?gpu}
MODEL=${2:-Qwen/Qwen3-1.7B}
PORT=8332
cd /workspace/mirage-det
export PYTHONPATH=/workspace/mirage-det/python

start_server() {  # args: logfile, extra args...
  local LOG=$1; shift
  CUDA_VISIBLE_DEVICES=$GPU python -m mirage.engine.launch_server \
    --model "$MODEL" --port $PORT --deterministic --capture-logprobs \
    --ignore-eos --max-seq-length 512 --max-num-batched-requests 16 \
    --max-num-pages 16 --pinned-ring-capacity 32 --request-timeout 900 \
    "$@" > "$LOG" 2>&1 &
  SPID=$!
  for i in $(seq 1 120); do
    sleep 10
    grep -q "Uvicorn running" "$LOG" && return 0
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

for MBT in 8 16; do
  TAG="mbt${MBT}"
  [ -f /tmp/e53_r2_$TAG.ok ] && continue
  if start_server /tmp/e53_server_$TAG.log \
      --sampling-seed 42 --max-num-batched-tokens $MBT; then
    sleep 5
    python experiments/e52_online_mpk.py --model "$MODEL" --port $PORT \
      --steps 5 --group-size 16 --output /tmp/e52_$TAG.jsonl \
      > /tmp/e52_$TAG.log 2>&1
    RC_A=$?
    python experiments/e53_online_mpk_group.py --model "$MODEL" --port $PORT \
      --steps 5 --group-size 16 --output /tmp/e53_$TAG.jsonl \
      --dump-first-step /tmp/e53_${TAG}_dump.json \
      > /tmp/e53_$TAG.log 2>&1
    RC_B=$?
    echo "r2 $TAG: e52 rc=$RC_A e53 rc=$RC_B" >> /tmp/e53_r2.log
    if [ $RC_A -eq 0 ] && [ $RC_B -eq 0 ]; then
      python3 - >> /tmp/e53_r2.log 2>&1 <<PY
import json
def avg(f):
    rs = [json.loads(l)["t_rollout_s"] for l in open(f)]
    return sum(rs) / len(rs)
a = avg("/tmp/e52_$TAG.jsonl"); b = avg("/tmp/e53_$TAG.jsonl")
print(f"$TAG rollout/step: individual {a:.3f}s  shared-prefix {b:.3f}s  "
      f"speedup {a/b:.2f}x")
PY
      touch /tmp/e53_r2_$TAG.ok
    fi
  else
    echo "r2 $TAG server failed to start" >> /tmp/e53_r2.log
  fi
  stop_server
done

if [ -f /tmp/e53_r2_mbt8.ok ] && [ -f /tmp/e53_r2_mbt16.ok ] \
    && [ ! -f /tmp/e53_r2_greedy.ok ]; then
  if start_server /tmp/e53_server_greedy.log --max-num-batched-tokens 8; then
    sleep 5
    python3 - >> /tmp/e53_r2.log 2>&1 <<'PY'
import json, urllib.request
def post(path, payload):
    req = urllib.request.Request(f"http://127.0.0.1:8332{path}",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())
prompt = "Give me a short introduction to large language model."
ref = post("/v1/completions", {"prompt": prompt})["choices"][0]
grp = post("/v1/group_completions",
           {"prompt": prompt, "group_size": 16})["choices"]
ok_tok = all(c["token_ids"] == ref["token_ids"] for c in grp)
ok_lp = all(c["logprobs"]["token_logprobs"] ==
            ref["logprobs"]["token_logprobs"] for c in grp)
print(f"GREEDY SHARED-PREFIX: tokens identical across 16 members+ref: "
      f"{ok_tok}; logprobs bitwise: {ok_lp}")
assert ok_tok and ok_lp
PY
    [ $? -eq 0 ] && touch /tmp/e53_r2_greedy.ok
  else
    echo "r2 greedy server failed to start" >> /tmp/e53_r2.log
  fi
  stop_server
fi

if [ -f /tmp/e53_r2_mbt8.ok ] && [ -f /tmp/e53_r2_mbt16.ok ] \
    && [ -f /tmp/e53_r2_greedy.ok ]; then
  touch /tmp/e53_r2.done
fi
