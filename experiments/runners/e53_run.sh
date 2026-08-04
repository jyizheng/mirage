#!/bin/bash
# E53: shared-prefix group rollout (+ optional mbt specialization) A/B.
# For MBT in 8 [16]: start the online server, run
#   (a) e52 (16 individual requests)   -> t_rollout baseline
#   (b) e53 (/v1/group_completions)    -> shared-prefix rollout
#   (c) greedy correctness: one unshared /v1/completions vs e53 members
#       must be bitwise-identical (token_ids and logprobs).
# Usage: e53_run.sh <gpu> [model] [mbt_list...]
set -u
GPU=${1:?gpu}
MODEL=${2:-Qwen/Qwen3-1.7B}
shift 2 2>/dev/null || shift $#
MBTS=(${@:-8 16})
PORT=8332
cd /workspace/mirage-det
export PYTHONPATH=/workspace/mirage-det/python

for MBT in "${MBTS[@]}"; do
  TAG="mbt${MBT}"
  CUDA_VISIBLE_DEVICES=$GPU python -m mirage.engine.launch_server \
    --model "$MODEL" --port $PORT --deterministic --capture-logprobs \
    --sampling-seed 42 --ignore-eos --max-seq-length 512 \
    --max-num-batched-requests 16 --max-num-pages 16 \
    --max-num-batched-tokens "$MBT" > /tmp/e53_server_$TAG.log 2>&1 &
  SPID=$!
  for i in $(seq 1 120); do
    sleep 10
    grep -q "Uvicorn running" /tmp/e53_server_$TAG.log && break
    kill -0 $SPID 2>/dev/null || { echo "$TAG server died" >> /tmp/e53.log; break; }
  done
  sleep 5
  python experiments/e52_online_mpk.py --model "$MODEL" --port $PORT \
    --steps 5 --group-size 16 --output /tmp/e52_$TAG.jsonl \
    > /tmp/e52_$TAG.log 2>&1
  echo "e52 $TAG rc=$?" >> /tmp/e53.log
  python experiments/e53_online_mpk_group.py --model "$MODEL" --port $PORT \
    --steps 5 --group-size 16 --output /tmp/e53_$TAG.jsonl \
    --dump-first-step /tmp/e53_${TAG}_dump.json \
    > /tmp/e53_$TAG.log 2>&1
  echo "e53 $TAG rc=$?" >> /tmp/e53.log
  kill $SPID 2>/dev/null; sleep 8; pkill -f "mirage.engine.launch_serve[r]"; sleep 5
  python3 - >> /tmp/e53.log 2>&1 <<PY
import json
def avg(f):
    rs=[json.loads(l)["t_rollout_s"] for l in open(f)]
    return sum(rs)/len(rs)
a=avg("/tmp/e52_$TAG.jsonl"); b=avg("/tmp/e53_$TAG.jsonl")
print(f"$TAG rollout/step: individual {a:.3f}s  shared-prefix {b:.3f}s  speedup {a/b:.2f}x")
PY
done

# ── greedy correctness: unshared reference vs shared members, bitwise ──
CUDA_VISIBLE_DEVICES=$GPU python -m mirage.engine.launch_server \
  --model "$MODEL" --port $PORT --deterministic --capture-logprobs \
  --ignore-eos --max-seq-length 512 \
  --max-num-batched-requests 16 --max-num-pages 16 \
  --max-num-batched-tokens 8 > /tmp/e53_server_greedy.log 2>&1 &
SPID=$!
for i in $(seq 1 120); do
  sleep 10
  grep -q "Uvicorn running" /tmp/e53_server_greedy.log && break
  kill -0 $SPID 2>/dev/null || { echo "greedy server died" >> /tmp/e53.log; break; }
done
sleep 5
python3 - >> /tmp/e53.log 2>&1 <<'PY'
import json, urllib.request
def post(path, payload):
    req = urllib.request.Request(f"http://127.0.0.1:8332{path}",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())
prompt = "Give me a short introduction to large language model."
ref = post("/v1/completions", {"prompt": prompt})["choices"][0]
grp = post("/v1/group_completions", {"prompt": prompt, "group_size": 16})["choices"]
ok_tok = all(c["token_ids"] == ref["token_ids"] for c in grp)
ok_lp = all(c["logprobs"]["token_logprobs"] ==
            ref["logprobs"]["token_logprobs"] for c in grp)
print(f"GREEDY SHARED-PREFIX: tokens identical across 16 members+ref: {ok_tok}; "
      f"logprobs bitwise: {ok_lp}")
PY
kill $SPID 2>/dev/null; sleep 5; pkill -f "mirage.engine.launch_serve[r]"
touch /tmp/e53.done
