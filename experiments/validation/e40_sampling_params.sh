#!/bin/bash
set -u
cd /workspace/mirage-det
pip install -e . --no-build-isolation > /tmp/e40_build.log 2>&1
echo "build rc=$?" >> /tmp/e40.log
run () { # tag extra-args
  local tag=$1; shift
  PYTHONPATH=/workspace/mirage-det/python CUDA_VISIBLE_DEVICES=7 \
    python demo/qwen3/demo.py --use-mirage --model Qwen/Qwen3-1.7B \
    --deterministic --sampling-seed 42 --capture-probs \
    --dump-tokens-file /tmp/e40_$tag.json "$@" > /tmp/e40_$tag.log 2>&1
  echo "$tag rc=$?" >> /tmp/e40.log
}
# A: default params must equal the plain-seed path bitwise (regression)
run defaultA
run defaultB
# B: top-k=20 twice (determinism) + top-p=0.9 + temperature=0.7
run topk20a --top_k 20
run topk20b --top_k 20
run topp09  --top_p 0.9
run temp07  --temperature 0.7
python3 - >> /tmp/e40.log 2>&1 <<PY
import json
def load(t): 
    d=json.load(open(f"/tmp/e40_{t}.json")); return d["token_ids"], d["prob_bits"]
dA=load("defaultA"); dB=load("defaultB")
print("DEFAULT bitwise-stable:", dA==dB)
kA=load("topk20a"); kB=load("topk20b")
print("TOPK20 rerun identical:", kA==kB)
# top-k should change the trajectory vs default (truncation active)
print("TOPK20 differs from default:", kA[0]!=dA[0])
pp=load("topp09"); tp=load("temp07")
print("TOPP09 gen len:", len(pp[0]), "TEMP07 gen len:", len(tp[0]))
PY
touch /tmp/e40.done
