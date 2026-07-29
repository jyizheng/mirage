#!/bin/bash
set -u
cd /workspace/mirage
export CUDA_VISIBLE_DEVICES=4
# run 1: greedy rollout with capture
python demo/qwen3/demo_30B_A3B.py --use-mirage --capture-probs \
  --dump-tokens-file /tmp/moe_ref.json > /tmp/moe_ref.log 2>&1
echo "ref rc=$?" >> /tmp/moe_rescore.log
# run 2: rescore -- full token sequence as prompt, teacher-forcing capture
python3 - <<PY
import json
r = json.load(open("/tmp/moe_ref.json"))
json.dump(r["token_ids"], open("/tmp/moe_full_prompt.json","w"))
PY
python demo/qwen3/demo_30B_A3B.py --use-mirage --capture-probs \
  --prompt-ids-file /tmp/moe_full_prompt.json \
  --dump-tokens-file /tmp/moe_rescore.json > /tmp/moe_rescore_run.log 2>&1
echo "rescore rc=$?" >> /tmp/moe_rescore.log
# compare bitwise on the teacher-forced region
python3 - >> /tmp/moe_rescore.log 2>&1 <<PY
import json
ref = json.load(open("/tmp/moe_ref.json"))
res = json.load(open("/tmp/moe_rescore.json"))
p0 = ref["prompt_length"]
n = len(ref["token_ids"])
match = mism = skipped = 0
for t in range(p0, n):
    a = ref["prob_bits"][t-1]
    b = res["prob_bits"][t-1]
    if a == 0 and b == 0:
        skipped += 1; continue
    if a == b: match += 1
    else:
        mism += 1
        if mism <= 3:
            print(f"MISMATCH at {t}: ref={a} rescore={b}")
print(f"MOE RESCORE: {match} bitwise-identical, {mism} mismatches, {skipped} zero-skipped, region [{p0},{n})")
PY
touch /tmp/moe_rescore.done
