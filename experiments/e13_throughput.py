# E13: throughput under concurrent load.
# Fixed prompt, 128 new tokens (ignore_eos), concurrency C in {1,4,8};
# N = C*4 requests per point; report aggregate output tokens/sec.
# Works against MPK server (/v1/completions) or SGLang (/generate).
import json
import sys
import threading
import time
import urllib.request

PORT = int(sys.argv[1])
KIND = sys.argv[2]  # mpk | sgl
PROMPT = "Give me a short introduction to large language model."
NEW = 128


def one_request():
    if KIND == "sgl":
        body = {"text": PROMPT,
                "sampling_params": {"temperature": 0, "max_new_tokens": NEW,
                                    "ignore_eos": True}}
        path = "/generate"
    else:
        body = {"prompt": PROMPT, "max_tokens": NEW}
        path = "/v1/completions"
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        json.loads(r.read())


one_request()  # warmup
for C in (1, 4, 8):
    N = C * 4
    sem = threading.Semaphore(C)
    errs = []

    def worker():
        with sem:
            try:
                one_request()
            except Exception as e:  # noqa
                errs.append(str(e))

    t0 = time.perf_counter()
    ts = [threading.Thread(target=worker) for _ in range(N)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    dt = time.perf_counter() - t0
    if errs:
        print(f"C={C}: ERRORS {errs[:2]}")
    else:
        print(f"C={C}: {N} reqs, {N*NEW} tokens in {dt:.2f}s -> "
              f"{N*NEW/dt:.1f} tok/s")
