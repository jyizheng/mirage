#!/bin/bash
set -u
cd /workspace/mirage-det
pip install -e . --no-build-isolation > /tmp/e42_build.log 2>&1
echo "build rc=$?" >> /tmp/e42.log
cd demo/qwen3
export PYTHONPATH=/workspace/mirage-det/python
CUDA_VISIBLE_DEVICES=7 python demo.py --use-mirage --model Qwen/Qwen3-1.7B \
  --deterministic --sampling-seed 42 --capture-probs --reference \
  --dump-tokens-file /tmp/e42_ref.json > /tmp/e42_run.log 2>&1
echo "ref rc=$?" >> /tmp/e42.log
python3 - >> /tmp/e42.log 2>&1 <<PY
import json
try:
    r=json.load(open("/tmp/e42_ref.json"))
except Exception as e:
    print("no dump:", e); raise SystemExit
plen=r["prompt_length"]; end=len(r["token_ids"])
pol=r["prob_bits"]; ref=r["ref_prob_bits"]
def cmp(lo,hi):
    a=pol[lo:hi]; b=ref[lo:hi]; n=sum(1 for x in a if x!=0)
    mm=sum(1 for x,y in zip(a,b) if x!=0 and x!=y); return n,mm
n1,m1=cmp(0,plen-1); n2,m2=cmp(plen-1,end-1)
print(f"REF==POLICY prompt teacher-forcing: {n1-m1}/{n1} bitwise ({m1} mm)")
print(f"REF==POLICY generated (decode cap): {n2-m2}/{n2} bitwise ({m2} mm)")
PY
touch /tmp/e42.done
