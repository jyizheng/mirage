"""LLMEngine — concurrent serving loop backed on persistent kernel + ring buffer.
"""

from __future__ import annotations

import queue
import threading
import time

import torch

import math

from .model_runner import ModelRunner
from .tokenizer_manager import TokenizerManager
from ..mpk.online_pinned_runtime import OnlinePinnedRuntime


class _StreamingMonitor:
    """Single background thread that monitors all active streaming sessions.

    Replaces the per-request polling thread (Thread B) with one shared daemon
    that drains completions, polls step progress, and enqueues tokens for
    every registered session.  This eliminates the thread explosion that
    causes GIL contention under concurrent load.
    """

    def __init__(self, runtime: OnlinePinnedRuntime, tokenizer_manager: TokenizerManager) -> None:
        self._runtime = runtime
        self._tokenizer_manager = tokenizer_manager
        self._sessions: dict[int, dict] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def register(self, rid: int, prompt_len: int, timeout: float) -> queue.Queue:
        """Register a streaming session and return its token queue."""
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._sessions[rid] = {
                'q': q,
                'prompt_len': prompt_len,
                'row': -1,
                'last_step': prompt_len - 1,
                'deadline': time.monotonic() + timeout,
            }
        return q

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                # Drain completions and flush waiting queue.
                self._runtime.drain_completions()

                with self._lock:
                    for rid, s in list(self._sessions.items()):
                        try:
                            # Phase 1 — discover buffer row.
                            if s['row'] == -1:
                                row = self._runtime.find_row_for_rid(rid)
                                if row >= 0:
                                    s['row'] = row
                                elif time.monotonic() > s['deadline']:
                                    s['q'].put(("__timeout__", True))
                                    del self._sessions[rid]
                                continue

                            row = s['row']

                            # Check completion via runtime bookkeeping.
                            with self._runtime._lock:
                                if rid in self._runtime._completions:
                                    _, final_step = self._runtime._completions[rid]
                                    self._yield_remaining(s, row, final_step)
                                    self._runtime.release_request(rid)
                                    del self._sessions[rid]
                                    continue

                            # Phase 2 — poll step progress, read only new tokens.
                            current_step = self._runtime.get_current_step_at_row(row)
                            if current_step > s['last_step']:
                                new_tokens = self._runtime.read_tokens_range(
                                    row, s['last_step'] + 1, current_step)
                                for tid in new_tokens.tolist():
                                    text = self._tokenizer_manager.decode_single(tid)
                                    s['last_step'] += 1
                                    s['q'].put((text, False))

                            # Check timeout.
                            if time.monotonic() > s['deadline']:
                                s['q'].put(("__timeout__", True))
                                del self._sessions[rid]

                        except Exception:
                            s['q'].put(("__error__", True))
                            del self._sessions[rid]

            except Exception:
                pass

            self._stop.wait(0.002)  # 2 ms polling interval

    def _yield_remaining(self, s: dict, row: int, final_step: int) -> None:
        """Yield all remaining tokens for a completed request."""
        if final_step > s['last_step']:
            new_tokens = self._runtime.read_tokens_range(
                row, s['last_step'] + 1, final_step)
            new_ids = new_tokens.tolist()
            for j, tid in enumerate(new_ids):
                text = self._tokenizer_manager.decode_single(tid)
                is_final = (j == len(new_ids) - 1)
                s['q'].put((text, is_final))
                s['last_step'] += 1
        else:
            s['q'].put(("", True))

    def shutdown(self) -> None:
        self._stop.set()


class LLMEngine:
    """Generation loop backed by the ``online_pinned`` persistent kernel.

    The kernel runs in a background thread so the engine can accept requests
    concurrently.  Each call to :meth:`submit` allocates a unique, never-
    repeating *request id* (rid), stages tokens in the pinned inbox, writes a
    ring-buffer entry, and then blocks until the GPU reports completion for
    that specific rid.  The GPU manages its own buffer-row pool and a
    waiting/running queue pair, so the CPU does not need to reason about
    slot availability.

    Args:
        model_runner: A fully constructed :class:`ModelRunner` whose MPK is
                      compiled in ``online_pinned`` mode.
    """

    def __init__(self, model_runner: ModelRunner) -> None:
        self.model_runner = model_runner
        self.runtime: OnlinePinnedRuntime = model_runner.runtime
        self.tokenizer_manager = TokenizerManager(model_runner.tokenizer)

        # Monotonically incrementing request id (never wraps).
        self._next_rid: int = 0

        # Serialises ring-buffer writes so two callers do not interleave.
        self._submit_lock = threading.RLock()

        # Shared streaming monitor — replaces per-request polling threads.
        self._monitor = _StreamingMonitor(self.runtime, self.tokenizer_manager)

        # Background kernel bookkeeping.
        self._kernel_launched: threading.Event = threading.Event()
        self._kernel_thread: threading.Thread | None = None

        # Launch MPK kernels
        self._ensure_kernel_running()

    # ── Public API ────────────────────────────────────────────────────────

    def submit(
        self,
        prompt: str,
        use_template: bool = True,
        timeout: float = 120.0,
        poll_interval: float = 1e-4,
        stream: bool = False,
    ):
        """Submit a single prompt for generation.

        Safe to call concurrently — each invocation gets a unique rid and
        serialises the ring-buffer write under an internal lock.

        Args:
            prompt:        String prompt.
            use_template:  Apply chat template before tokenizing.
            timeout:       Seconds to wait before raising :exc:`TimeoutError`.
            poll_interval: Seconds between completion-ring polls.
            stream:        If True, returns a generator yielding ``(text,
                           is_final)`` tuples. Otherwise returns a dict.

        Returns:
            When stream=False: ``{"text": str, "token_ids": list[int]}``
            When stream=True:  generator yielding ``(text, is_final)``
        """
        token_ids = self.tokenizer_manager.tokenize(prompt, use_template)
        prompt_len = len(token_ids)

        rid = self._next_rid
        self._next_rid += 1

        t = torch.tensor(token_ids, dtype=torch.int64)
        with self._submit_lock:
            self.runtime.submit(rid, t)

        if stream:
            return self._submit_stream(rid, prompt_len, timeout, poll_interval)
        else:
            buffer_row, final_step = self.runtime.wait_for_request(
                rid, timeout, poll_interval)
            full_tokens = self.runtime.read_tokens_at_row(buffer_row, final_step)
            output_ids = full_tokens[prompt_len:].tolist()
            result = {
                "text": self.tokenizer_manager.decode(output_ids),
                "token_ids": output_ids,
            }
            prob_buffer = getattr(self.model_runner, "prob_buffer", None)
            if prob_buffer is not None:
                # P(chosen token at position t) sits at buffer[row, t-1];
                # generated tokens occupy t in [prompt_len, final_step].
                probs = prob_buffer[
                    buffer_row, prompt_len - 1 : final_step].tolist()
                result["logprobs"] = [
                    math.log(p) if p > 0.0 else None for p in probs]
            self.runtime.release_request(rid)
            return result

    def submit_group(
        self,
        prompt: str,
        group_size: int,
        use_template: bool = True,
        timeout: float = 300.0,
        poll_interval: float = 1e-4,
    ):
        """GRPO-style group rollout with SHARED-PREFIX prefill.

        All ``group_size`` trajectories share one prompt, so the prompt's
        KV is computed ONCE: member 0 prefills normally; once its prefill
        completes we copy the prompt-region KV of its page to every other
        page in the pool (pure copy-engine D2D, safe alongside the
        persistent kernel), then admit members 1..G-1 with
        ``initial_step = P-1`` — the runtime's prefix-cache admission path
        skips the cached positions and live-prefills only the final prompt
        token, which produces the first generation logits. Because the
        kernels are deterministic, the copied KV is bitwise what each
        member would have computed itself, so trajectories are unchanged
        (greedy) or per-slot reproducible (seeded sampling).

        Requirements: engine idle (no other in-flight requests — asserted),
        one KV page per request (max_seq_length <= page_size),
        group_size <= max_num_pages, and pinned_ring_capacity >
        group_size: members admitted in the same scheduler step finish in
        lockstep, and the GPU pushes completions without checking ring
        fullness — a wave of >= capacity simultaneous completions is lost
        (observed: 8-at-once with capacity 8 lost 7 of 8; 7-at-once was
        fine).
        """
        rt = self.runtime
        assert not rt._completions and rt.waiting_count == 0, \
            "submit_group requires an idle engine"
        builder = self.model_runner.mpk.model_builder
        k_cache, v_cache = builder.k_cache, builder.v_cache
        meta = self.model_runner.meta_tensors
        page_size = k_cache.shape[2]
        n_pages = k_cache.shape[1]
        assert group_size <= n_pages, "group_size exceeds page pool"
        assert group_size < rt._cap, \
            "group_size must be < pinned_ring_capacity (lockstep " \
            "completions overflow the completion ring and are lost)"

        token_ids = self.tokenizer_manager.tokenize(prompt, use_template)
        prompt_len = len(token_ids)
        assert prompt_len >= 2 and prompt_len <= page_size
        t = torch.tensor(token_ids, dtype=torch.int64)

        # The GPU only starts writing a row's pinned step mirror once the
        # request is batched, so a reused row still shows its PREVIOUS
        # occupant's final step — which is >= prompt_len and would make the
        # prefill wait below pass instantly. The engine is idle (asserted),
        # so no row is being written: clear all mirrors first.
        rt._pinned_step.fill_(-1)

        # ── member 0: normal submission, prefills the shared prompt ──
        rid0 = self._next_rid
        self._next_rid += 1
        with self._submit_lock:
            rt.submit(rid0, t)
        deadline = time.monotonic() + timeout
        row0 = -1
        while row0 < 0:
            row0 = rt.find_row_for_rid(rid0)
            if time.monotonic() > deadline:
                raise TimeoutError("submit_group: member-0 admission")
            time.sleep(poll_interval)
        while rt.get_current_step_at_row(row0) < prompt_len:
            if time.monotonic() > deadline:
                raise TimeoutError("submit_group: member-0 prefill")
            time.sleep(poll_interval)

        # ── locate member 0's page (stable double-read; engine is idle
        # apart from member 0, so slot 0 is its batch slot) ──
        def read_page():
            return int(meta["paged_kv_indices_buffer"][0].cpu().item())
        src = read_page()
        while True:
            again = read_page()
            if again == src:
                break
            src = again

        # ── replicate the prompt-prefix KV to every other page.
        # Contiguous slice copies only: copy-engine transfers, safe while
        # the megakernel occupies the SMs. Prefix is [0, P-1); position
        # P-1 is live-prefilled by each member to produce its own first
        # logits row.
        pfx = prompt_len - 1
        num_layers = k_cache.shape[0]
        for layer in range(num_layers):
            k_src = k_cache[layer, src, :pfx]
            v_src = v_cache[layer, src, :pfx]
            for dst in range(n_pages):
                if dst == src:
                    continue
                # per-(layer, page) prefix slices are contiguous, so these
                # are pure memcpys on the copy engine
                k_cache[layer, dst, :pfx].copy_(k_src)
                v_cache[layer, dst, :pfx].copy_(v_src)
        # The copies are asynchronous — members must not admit before the
        # prefix KV has actually landed. Wait for the copies ONLY, via an
        # event on their stream: a full torch.cuda.synchronize() is
        # cudaDeviceSynchronize, which also waits on the resident
        # megakernel's stream and therefore never returns.
        copies_done = torch.cuda.Event()
        copies_done.record()
        copies_done.synchronize()

        # ── members 1..G-1: prefix-cached admission ──
        rids = [rid0]
        for _ in range(group_size - 1):
            rid = self._next_rid
            self._next_rid += 1
            with self._submit_lock:
                rt.submit(rid, t, initial_step=pfx)
            rids.append(rid)

        # ── collect ──
        results = []
        prob_buffer = getattr(self.model_runner, "prob_buffer", None)
        for rid in rids:
            buffer_row, final_step = rt.wait_for_request(
                rid, max(deadline - time.monotonic(), 1.0), poll_interval)
            full_tokens = rt.read_tokens_at_row(buffer_row, final_step)
            output_ids = full_tokens[prompt_len:].tolist()
            result = {
                "text": self.tokenizer_manager.decode(output_ids),
                "token_ids": output_ids,
            }
            if prob_buffer is not None:
                probs = prob_buffer[
                    buffer_row, prompt_len - 1 : final_step].tolist()
                result["logprobs"] = [
                    math.log(p) if p > 0.0 else None for p in probs]
            rt.release_request(rid)
            results.append(result)
        return results

    # ── Internal ──────────────────────────────────────────────────────────

    def _ensure_kernel_running(self) -> None:
        """Launch the persistent kernel once in a background daemon thread."""
        if self._kernel_launched.is_set():
            return
        with self._submit_lock:
            if self._kernel_launched.is_set():
                return
            self.runtime.reset()
            self._kernel_thread = threading.Thread(
                target=self.model_runner, daemon=True)
            self._kernel_thread.start()
            self._kernel_launched.set()

    def _submit_stream(
        self,
        rid: int,
        prompt_len: int,
        timeout: float,
        poll_interval: float,
    ):
        """Generator: yield ``(text, is_final)`` as tokens are decoded.

        Registers with the shared :class:`_StreamingMonitor` instead of
        spawning a dedicated polling thread, so the number of polling
        threads stays constant regardless of concurrent request count.
        """
        q = self._monitor.register(rid, prompt_len, timeout)

        def generator():
            while True:
                try:
                    text, is_final = q.get(timeout=0.05)
                except queue.Empty:
                    continue
                if text == "__timeout__":
                    raise TimeoutError(
                        f"stream timed out for rid={rid}")
                if text == "__error__":
                    raise RuntimeError(
                        f"stream error for rid={rid}")
                yield (text, is_final)
                if is_final:
                    break

        return generator()

    def close(self) -> None:
        """Signal the GPU kernel to shut down at the next idle cycle."""
        self._monitor.shutdown()
        self.runtime.shutdown()
