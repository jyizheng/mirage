#!/bin/bash
set -u

# E44: trainer -> MPK rollout weight-sync benchmark.
# Produces /tmp/e44_synthetic.json and /tmp/e44_mpk.json.

cd /workspace/mirage-det || exit 1

PYTHONPATH=/workspace/mirage-det/python CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  python experiments/e44_weight_sync_bench.py \
    --target synthetic --device cuda --steps 20 --warmup 3 \
    --world-size 1 > /tmp/e44_synthetic.json 2> /tmp/e44_synthetic.log
echo "e44 synthetic rc=$?" >> /tmp/e44.log

PYTHONPATH=/workspace/mirage-det/python CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  python experiments/e44_weight_sync_bench.py \
    --target mpk --model "${MODEL:-Qwen/Qwen3-1.7B}" \
    --max-num-batched-requests 8 --max-num-batched-tokens 8 \
    --max-seq-length 512 --max-num-pages 16 --page-size 4096 \
    --steps 20 --warmup 3 > /tmp/e44_mpk.json 2> /tmp/e44_mpk.log
echo "e44 mpk rc=$?" >> /tmp/e44.log

touch /tmp/e44.done
