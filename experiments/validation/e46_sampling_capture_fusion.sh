#!/bin/bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

python -m pip install -e . --no-build-isolation \
  > /tmp/e46_build.log 2>&1

run_one() {
  local variant=$1
  local repetition=$2
  shift 2
  PYTHONPATH="$repo_root/python" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}" \
    python demo/qwen3/demo.py \
      --use-mirage \
      --model Qwen/Qwen3-1.7B \
      --max-num-batched-requests 8 \
      --max-num-batched-tokens 8 \
      --ignore-eos \
      --deterministic \
      --sampling-seed 42 \
      --capture-probs \
      --dump-tokens-file "/tmp/e46_${variant}_${repetition}.json" \
      "$@" \
      > "/tmp/e46_${variant}_${repetition}.log" 2>&1
}

for repetition in 1 2 3; do
  run_one standalone "$repetition" --no-fused-sampling-capture
  run_one fused "$repetition"
done

python - <<'PY'
import json
import statistics


def load(variant, repetition):
    with open(f"/tmp/e46_{variant}_{repetition}.json") as handle:
        return json.load(handle)


standalone = [load("standalone", repetition) for repetition in (1, 2, 3)]
fused = [load("fused", repetition) for repetition in (1, 2, 3)]

reference_tokens = standalone[0]["token_ids"]
reference_probs = standalone[0]["prob_bits"]
for record in standalone + fused:
    assert record["token_ids"] == reference_tokens, "token trace mismatch"
    assert record["prob_bits"] == reference_probs, "probability bits mismatch"

standalone_ms = statistics.median(
    record["latency_ms_per_token"] for record in standalone
)
fused_ms = statistics.median(
    record["latency_ms_per_token"] for record in fused
)
result = {
    "experiment": "E46 sampled probability-capture fusion",
    "model": "Qwen/Qwen3-1.7B",
    "group_size": 8,
    "generated_tokens_per_request": (
        len(reference_tokens) - standalone[0]["prompt_length"]
    ),
    "runs_per_variant": 3,
    "bitwise_identical": True,
    "standalone_median_ms_per_token": standalone_ms,
    "fused_median_ms_per_token": fused_ms,
    "speedup": standalone_ms / fused_ms,
}
with open("/tmp/e46_result.json", "w") as handle:
    json.dump(result, handle, indent=2)
print(json.dumps(result, indent=2))
PY
