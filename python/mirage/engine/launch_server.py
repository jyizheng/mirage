"""Launch the Mirage LLM Engine as an OpenAI-compatible HTTP server.

Usage::

    python -m mirage.engine.launch_server \\
        --model Qwen/Qwen3-8B \\
        --max-num-batched-requests 4 \\
        --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse

from .model_runner import ModelRunner, RunnerConfig
from .llm_engine import LLMEngine


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    config: RunnerConfig = app.state.runner_config
    runner = ModelRunner(config)
    engine = LLMEngine(runner)
    app.state.engine = engine
    yield
    engine.close()


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="MPK LLM Engine", lifespan=lifespan)


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _parse_json(request: Request) -> dict:
    """Parse JSON body, returning 400 on empty or malformed input."""
    try:
        return await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or empty JSON body")


def _extract_prompt(messages: list[dict]) -> str:
    """Pull the last user message from an OpenAI chat messages list."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg["content"]
    return ""


async def _stream_bridge(
    engine: LLMEngine, prompt: str, timeout: float,
) -> AsyncGenerator[str, None]:
    """Bridge a synchronous streaming generator to async SSE chunks.

    Each request gets its own daemon thread so concurrent requests are never
    gated by the default ``ThreadPoolExecutor`` pool size.  Items produced by
    the thread are handed to the event loop via ``call_soon_threadsafe`` so
    that the asyncio queue is accessed only from the event-loop thread.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _put(text: str, is_final: bool, error: str | None) -> None:
        """Called on the event-loop thread; safe to touch the asyncio queue."""
        queue.put_nowait((text, is_final, error))

    def _run() -> None:
        try:
            gen = engine.submit(prompt, stream=True, timeout=timeout)
            for text, is_final in gen:
                loop.call_soon_threadsafe(_put, text, is_final, None)
        except BaseException as exc:
            loop.call_soon_threadsafe(_put, "", True, str(exc))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    while True:
        text, is_final, error = await queue.get()

        if error:
            yield f"data: {{\"error\": \"{error}\"}}\n\n"
            break

        chunk = json.dumps({
            "choices": [{"delta": {"content": text}, "index": 0}],
        })
        yield f"data: {chunk}\n\n"
        if is_final:
            break

    thread.join()
    yield "data: [DONE]\n\n"


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/metrics")
async def metrics(request: Request):
    """Prometheus text-format metrics (hand-formatted, no client library).

    All counters come from :class:`~.llm_engine.EngineStats`, which is
    updated only at request submit/completion boundaries — reading this
    endpoint never touches the GPU or the engine's polling loops.
    """
    engine = request.app.state.engine
    config = engine.model_runner.config
    mpk = engine.model_runner.mpk
    snap = engine.stats.snapshot()

    lines: list[str] = []

    def emit(name: str, mtype: str, help_text: str, value, labels: str = ""):
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {mtype}")
        lines.append(f"{name}{labels} {value}")

    emit("mirage_requests_submitted_total", "counter",
         "Requests submitted to the engine (group members count "
         "individually).", snap["requests_submitted"])
    emit("mirage_requests_completed_total", "counter",
         "Requests completed successfully.", snap["requests_completed"])
    emit("mirage_requests_failed_total", "counter",
         "Requests that timed out or errored.", snap["requests_failed"])
    emit("mirage_requests_in_flight", "gauge",
         "Requests submitted but not yet completed or failed.",
         snap["requests_in_flight"])
    emit("mirage_tokens_generated_total", "counter",
         "Generated (non-prompt) tokens across all completed requests.",
         snap["tokens_generated"])
    emit("mirage_last_request_duration_seconds", "gauge",
         "Wall-clock duration of the most recently completed request.",
         f"{snap['last_request_duration_s']:.6f}")
    emit("mirage_engine_uptime_seconds", "gauge",
         "Seconds since the engine was constructed.",
         f"{snap['uptime_s']:.3f}")

    info_labels = (
        '{model="%s",num_workers="%d",num_schedulers="%d",'
        'max_num_batched_requests="%d",max_num_batched_tokens="%d",'
        'max_num_pages="%d",page_size="%d",pinned_ring_capacity="%d",'
        'max_seq_length="%d",deterministic="%s",sampling_seed="%s"}'
        % (config.model, mpk.num_workers, mpk.num_schedulers,
           config.max_num_batched_requests, config.max_num_batched_tokens,
           config.max_num_pages, config.page_size,
           config.pinned_ring_capacity, config.max_seq_length,
           str(config.deterministic).lower(), config.sampling_seed))
    emit("mirage_engine_info", "gauge",
         "Engine build/config echo (always 1).", 1, info_labels)

    return PlainTextResponse(
        "\n".join(lines) + "\n",
        media_type="text/plain; version=0.0.4; charset=utf-8")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await _parse_json(request)
    prompt = _extract_prompt(body.get("messages", []))
    stream = body.get("stream", False)
    timeout = request.app.state.request_timeout

    if stream:
        return StreamingResponse(
            _stream_bridge(request.app.state.engine, prompt, timeout),
            media_type="text/event-stream",
        )
    else:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: request.app.state.engine.submit(
                prompt, timeout=timeout),
        )
        return {
            "id": "chatcmpl-0",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result["text"]},
                "finish_reason": "stop",
            }],
            "usage": {"completion_tokens": len(result["token_ids"])},
        }


@app.post("/v1/completions")
async def completions(request: Request):
    body = await _parse_json(request)
    prompt = body.get("prompt")
    prompt_token_ids = body.get("prompt_token_ids")
    stream = body.get("stream", False)
    timeout = request.app.state.request_timeout

    # `prompt_token_ids` (list of ints) bypasses tokenization entirely —
    # partial-rollout resume submits prompt tokens + the partial generation
    # here and the engine continues decoding.  Mutually exclusive with
    # `prompt`.
    if prompt_token_ids is not None:
        if prompt is not None:
            raise HTTPException(
                status_code=400,
                detail="`prompt` and `prompt_token_ids` are mutually "
                       "exclusive")
        if stream:
            raise HTTPException(
                status_code=400,
                detail="`prompt_token_ids` does not support stream=true")
        if (not isinstance(prompt_token_ids, list) or not prompt_token_ids
                or not all(isinstance(t, int) for t in prompt_token_ids)):
            raise HTTPException(
                status_code=400,
                detail="`prompt_token_ids` must be a non-empty list of ints")
    elif prompt is None:
        prompt = ""

    if stream:
        return StreamingResponse(
            _stream_bridge(request.app.state.engine, prompt, timeout),
            media_type="text/event-stream",
        )
    else:
        loop = asyncio.get_running_loop()
        use_template = body.get("use_template", True)
        max_new_tokens = body.get("max_new_tokens")
        result = await loop.run_in_executor(
            None, lambda: request.app.state.engine.submit(
                prompt,
                use_template=use_template,
                timeout=timeout,
                prompt_token_ids=prompt_token_ids,
                max_new_tokens=max_new_tokens),
        )
        choice = {
            "index": 0,
            "text": result["text"],
            "token_ids": result["token_ids"],
            "prompt_token_ids": result["prompt_token_ids"],
            "buffer_row": result["buffer_row"],
            "finish_reason": "stop",
        }
        if "logprobs" in result:
            choice["logprobs"] = {"token_logprobs": result["logprobs"]}
        return {
            "id": "cmpl-0",
            "object": "text_completion",
            "choices": [choice],
            "usage": {"completion_tokens": len(result["token_ids"])},
        }


@app.post("/v1/group_completions")
async def group_completions(request: Request):
    """GRPO group rollout: one prompt, ``group_size`` trajectories, with
    shared-prefix prefill (the prompt's KV is computed once and replicated;
    members admit via the runtime's prefix-cache path). Requires an
    otherwise-idle engine."""
    body = await _parse_json(request)
    prompt = body.get("prompt", "")
    group_size = int(body.get("group_size", 1))
    use_template = body.get("use_template", True)
    timeout = request.app.state.request_timeout

    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(
        None, lambda: request.app.state.engine.submit_group(
            prompt, group_size, use_template=use_template, timeout=timeout),
    )
    choices = []
    for i, result in enumerate(results):
        choice = {
            "index": i,
            "text": result["text"],
            "token_ids": result["token_ids"],
            "finish_reason": "stop",
        }
        if "logprobs" in result:
            choice["logprobs"] = {"token_logprobs": result["logprobs"]}
        choices.append(choice)
    return {
        "id": "cmpl-group-0",
        "object": "text_completion",
        "choices": choices,
        "usage": {"completion_tokens":
                  sum(len(r["token_ids"]) for r in results)},
    }


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Mirage LLM Engine Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", default=8000, type=int, help="Port to listen on")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="HuggingFace model name")
    parser.add_argument("--model-path", default=None, help="Path to local model")
    parser.add_argument("--max-num-batched-requests", default=4, type=int)
    parser.add_argument("--max-num-batched-tokens", default=8, type=int)
    parser.add_argument("--max-seq-length", default=512, type=int)
    parser.add_argument("--max-num-pages", default=16, type=int)
    parser.add_argument("--page-size", default=4096, type=int)
    parser.add_argument("--pinned-ring-capacity", default=8, type=int)
    parser.add_argument("--output-dir", default=None, help="Output directory for compiled artifacts")
    parser.add_argument("--deterministic", action="store_true",
                        help="Compile deterministic kernel variants (bitwise-reproducible, rescore-consistent rollouts)")
    parser.add_argument("--capture-logprobs", action="store_true",
                        help="Compile the probability-capture task into the graph and return per-token logprobs in completions")
    parser.add_argument("--sampling-seed", type=int, default=None,
                        help="Enable position-keyed stochastic sampling.")
    parser.add_argument("--frequency-penalty", type=float, default=0.0,
                        help="OpenAI-style frequency penalty over generated "
                             "tokens. Compile-time engine default baked into "
                             "the sampling kernel (requires --sampling-seed); "
                             "per-request overrides are not supported.")
    parser.add_argument("--presence-penalty", type=float, default=0.0,
                        help="OpenAI-style presence penalty over generated "
                             "tokens (compile-time; requires "
                             "--sampling-seed).")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="HF-style repetition penalty over generated "
                             "tokens (compile-time; requires "
                             "--sampling-seed).")
    parser.add_argument("--ignore-eos", action="store_true",
                        help="Generate every request to max sequence length.")
    parser.add_argument("--request-timeout", default=7200.0, type=float,
                        help="Per-request timeout in seconds (default: 7200)")
    parser.add_argument("--num-workers", default=-1, type=int,
                        help="Worker CTA count (-1 = auto-probe)")
    parser.add_argument("--num-schedulers", default=-1, type=int,
                        help="Scheduler warp count (-1 = auto-probe)")
    args = parser.parse_args()

    config = RunnerConfig(
        model=args.model,
        model_path=args.model_path,
        max_num_batched_requests=args.max_num_batched_requests,
        max_num_batched_tokens=args.max_num_batched_tokens,
        capture_logprobs=args.capture_logprobs,
        deterministic=args.deterministic,
        sampling_seed=args.sampling_seed,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        ignore_eos=args.ignore_eos,
        max_seq_length=args.max_seq_length,
        max_num_pages=args.max_num_pages,
        page_size=args.page_size,
        pinned_ring_capacity=args.pinned_ring_capacity,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        num_schedulers=args.num_schedulers,
    )
    app.state.runner_config = config
    app.state.request_timeout = args.request_timeout
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
