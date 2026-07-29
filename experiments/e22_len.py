# E22: SGLang per-token decode latency vs context length (one mode).
# Counterpart of the MPK tax-vs-length sweep: does the op-invariance tax
# decay with L the way MPK's epilogue tax does?
# Prompt = filler repeated to ~L tokens; warm call caches the prefix, then
# 3 timed 128-token greedy generations; report best per-token ms.
import json
import sys
import time
import urllib.request

PORT = 8322
MODE = sys.argv[1]
FILLER = ("The quick brown fox jumps over the lazy dog while the curious "
          "cat watches from the windowsill of the old library. ")


def post(body, timeout=900):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/generate", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


for L in (1024, 2048, 4096, 8192):
    # ~22 tokens per filler sentence; build to roughly L-160 prompt tokens
    reps = max(1, (L - 160) // 22)
    prompt = FILLER * reps + "\nSummarize the text above in one sentence."
    body = {"text": prompt,
            "sampling_params": {"temperature": 0, "max_new_tokens": 128,
                                "ignore_eos": True}}
    out = post(body)  # warm (prefill + cache)
    plen = out["meta_info"]["prompt_tokens"]
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        out = post(body)
        dt = time.perf_counter() - t0
        ct = out["meta_info"]["completion_tokens"]
        best = min(best, dt / ct * 1000)
    print(f"{MODE} L~{L} (prompt={plen}): {best:.3f} ms/token", flush=True)
