#!/bin/bash
# SGLang same-hardware baseline: normal vs deterministic mode.
# Measures single-request greedy decode ms/token on Qwen3-8B, GPU 3.
set -u
VENV=/workspace/sgl-venv
PORT=8322
export CUDA_VISIBLE_DEVICES=3

run_mode () {
  local name=$1; shift
  local extra_args=("$@")
  echo "=== mode: $name ==="
  $VENV/bin/python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B --port $PORT --host 127.0.0.1 \
    "${extra_args[@]}" > /tmp/sgl_server_$name.log 2>&1 &
  local spid=$!
  # wait for readiness
  for i in $(seq 1 120); do
    sleep 10
    if curl -s http://127.0.0.1:$PORT/health_generate > /dev/null 2>&1; then
      echo "server ready after ~$((i*10))s"
      break
    fi
    if ! kill -0 $spid 2>/dev/null; then
      echo "SERVER DIED (mode $name); tail:"; tail -5 /tmp/sgl_server_$name.log
      return 1
    fi
  done
  $VENV/bin/python - <<'EOF'
import json, time, urllib.request
PORT=8322
prompt = ("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
          "<|im_start|>user\nGive me a short introduction to large language model.<|im_end|>\n"
          "<|im_start|>assistant\n")
def gen(max_new):
    body = json.dumps({"text": prompt,
                       "sampling_params": {"temperature": 0, "max_new_tokens": max_new, "ignore_eos": True}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    dt = time.perf_counter() - t0
    ct = out["meta_info"]["completion_tokens"]
    return dt, ct
gen(16)  # warmup
times = []
for _ in range(3):
    dt, ct = gen(256)
    times.append(dt / ct * 1000)
print(f"per-token latency ms (3 runs of 256 tok): {[f'{t:.3f}' for t in times]}, best={min(times):.3f}")
EOF
  kill $spid 2>/dev/null; sleep 8; pkill -f "sglang.launch_server" 2>/dev/null; sleep 5
}

run_mode normal_triton --attention-backend triton
#run_mode deterministic --enable-deterministic-inference --attention-backend triton
echo "ALL DONE"
