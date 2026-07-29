# E21: throughput under concurrent load, actual-token accounting.
# Greedy decode, same prompt on both engines; completion token counts are
# read from the server response (usage.completion_tokens for MPK,
# meta_info.completion_tokens for SGLang), so engines that ignore
# max_tokens are still measured correctly.
import json
import sys
import threading
import time
import urllib.request

PORT = int(sys.argv[1])
KIND = sys.argv[2]  # mpk | sgl
PROMPT = "Give me a short introduction to large language model."
NEW = 260


def one_request():
    if KIND == "sgl":
        body = {"text": PROMPT,
                "sampling_params": {"temperature": 0, "max_new_tokens": NEW}}
        path = "/generate"
    else:
        body = {"prompt": PROMPT, "max_tokens": NEW}
        path = "/v1/completions"
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.loads(r.read())
    if KIND == "sgl":
        return out["meta_info"]["completion_tokens"]
    return out["usage"]["completion_tokens"]


one_request()  # warmup
for C in (1, 2, 4, 8):
    N = C * 4
    sem = threading.Semaphore(C)
    errs = []
    toks = []

    def worker():
        with sem:
            try:
                toks.append(one_request())
            except Exception as e:  # noqa
                errs.append(str(e))

    t0 = time.perf_counter()
    ts = [threading.Thread(target=worker) for _ in range(N)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    dt = time.perf_counter() - t0
    total = sum(toks)
    tag = f"C={C}: {len(toks)}/{N} reqs, {total} tokens in {dt:.2f}s -> " \
          f"{total/dt:.1f} tok/s"
    if errs:
        tag += f"  ERRORS({len(errs)}): {errs[:2]}"
    print(tag, flush=True)
