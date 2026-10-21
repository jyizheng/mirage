"""DeepSeek-V4.1 on MPK: one-token-per-step decode graph (M0).

Follows DeepSeek's inference/model.py op for op, built from the kernels in
tasks/blackwell/dsv4_sm100.cuh and mhc_sm100.cuh. Scope of this first version:

  * one request, one token per step; the prompt is fed token by token.
  * bf16 checkpoints (e.g. the 4-layer slice), single GPU.
  * the indexer is built when the compressed length can exceed index_topk; its
    first level (candidate blocks) keeps every block up to
    candidate_topk_blocks * candidate_block_size positions and is not built.

Runtime state the caller drives (all on the GPU):
  self.tokens      int64 [max_seq_len]  token ids, prompt then generated
  self.step        int32 [1]            position of the token decoded this step
  self.next_token  int64 [1]            argmax output of the step
"""

import math
import os

import numpy as np
import torch

from ...persistent_kernel import PersistentKernel


# ---------------------------------------------------------------- reference helpers
def precompute_freqs(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    """inference/model.py precompute_freqs_cis, returned as fp32 [seqlen, dim / 2, 2] = (cos, sin)."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(seqlen), freqs)
    cis = torch.polar(torch.ones_like(freqs), freqs)
    return torch.view_as_real(cis).contiguous()


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            return False
        f += 2
    return True


def engram_hash_tables(cfg: dict, tokenizer):
    """inference/engram.py EngramLayout + NgramHashState buffers, for each engram layer:
    (token_map [V] int64, multipliers [NGRAM] int64, primes [(NGRAM-1)*HEADS] int64,
    offsets [(NGRAM-1)*HEADS] int64, pad_id)."""
    from tokenizers import Regex, normalizers

    sentinel = ""
    normalizer = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(), normalizers.Replace(sentinel, " "),
    ])
    backend = tokenizer.backend_tokenizer
    key_to_new, lookup = {}, [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.setdefault(key, len(key_to_new))
        lookup[token_id] = new_id
    vocab = len(key_to_new)
    assert vocab == cfg["engram_compressed_vocab_size"], (vocab, cfg["engram_compressed_vocab_size"])
    token_map = torch.tensor(lookup, dtype=torch.int64)

    ngram, heads = cfg["engram_max_ngram_size"], cfg["engram_n_heads"]
    bound = max(1, (np.iinfo(np.int64).max // vocab) // 2)
    seen, tables = set(), {}
    for layer_id in cfg["engram_layer_ids"]:
        flat = []
        for _ in range(ngram - 1):
            current = cfg["engram_vocab_size"] - 1
            for _ in range(heads):
                current += 1
                while not _is_prime(current) or current in seen:
                    current += 1
                seen.add(current)
                flat.append(current)
        offsets = np.cumsum([0, *flat[:-1]])
        mult = np.random.default_rng(10007 * layer_id).integers(
            low=0, high=bound, size=(ngram,), dtype=np.int64) * 2 + 1
        tables[layer_id] = (
            token_map,
            torch.tensor(mult, dtype=torch.int64),
            torch.tensor(flat, dtype=torch.int64),
            torch.tensor(offsets, dtype=torch.int64),
            lookup[cfg["engram_pad_token_id"]],
        )
    return tables


def _divisor_at_most(n: int, cap: int) -> int:
    for d in range(min(n, cap), 0, -1):
        if n % d == 0:
            return d
    return 1


# ---------------------------------------------------------------- builder
class DeepSeekV41DecodeBuilder:
    BLOCK = (256, 1, 1)

    def __init__(self, mpk: PersistentKernel, cfg: dict, max_seq_len: int, device="cuda",
                 tokens: torch.Tensor = None, step: torch.Tensor = None,
                 next_token: torch.Tensor = None):
        """tokens / step / next_token default to tensors owned by the builder (for a caller that
        drives the steps itself); pass MPK's meta tensors (tokens[0], step[:1],
        output_tokens[:1]) to run under MPK's own scheduler."""
        self.mpk = mpk
        self.cfg = cfg
        self.device = device
        self.max_seq_len = max_seq_len
        self.dim = cfg["hidden_size"]
        self.hc = cfg["hc_mult"]
        self.mix = (2 + self.hc) * self.hc
        self.n_heads = cfg["num_attention_heads"]
        self.head_dim = cfg["head_dim"]
        self.rope_dim = cfg["qk_rope_head_dim"]
        self.window = cfg["sliding_window"]
        self.ratios = list(cfg["compress_ratios"])
        self.kv_sources = set(cfg["kv_source_layer_ids"])
        self.eps = cfg["rms_norm_eps"]
        self.n_experts = cfg["n_routed_experts"]
        self.topk = cfg["num_experts_per_tok"]
        self.inter = cfg["moe_intermediate_size"]
        self.o_groups = cfg["o_groups"]
        self.o_rank = cfg["o_lora_rank"]
        self.vocab = cfg["vocab_size"]
        self.index_sources = set(cfg["index_source_layer_ids"])
        self.index_topk = cfg["index_topk"]
        # The indexer's first level (candidate blocks) keeps every block while the compressed
        # length fits in candidate_topk_blocks * candidate_block_size; it is not built.
        n_cand = cfg.get("candidate_topk_blocks", 0) * cfg.get("candidate_block_size", 0)
        for i, r in enumerate(self.ratios):
            if r:
                assert i in self.index_sources or max_seq_len // r <= self.index_topk
                assert not n_cand or max_seq_len // r <= n_cand, (
                    "candidate-block selection is not implemented: max_seq_len too long")
        assert cfg.get("num_nextn_predict_layers", 0) == 0
        self.keep = []  # torch tensors the graph points at
        self.bufs = {}  # name -> torch tensor, for inspection by tests
        self._torch = {}  # id(DTensor) -> the torch tensor it was attached from
        self._dts = []

        def runtime(t, name, shape, dtype):
            if t is None:
                return self._t(torch.zeros(shape, dtype=dtype), name)
            assert t.dtype == dtype and t.is_contiguous() and t.numel() == math.prod(shape), (name, t.shape)
            self.keep.append(t)
            self.bufs[name] = t
            dt = self.mpk.attach_input(torch_tensor=t, name=name)
            self._dts.append(dt)
            self._torch[id(dt)] = t
            return dt

        self.tokens = runtime(tokens, "tokens", (max_seq_len,), torch.int64)
        self.step = runtime(step, "step", (1,), torch.int32)
        self.next_token = runtime(next_token, "next_token", (1,), torch.int64)

    # ------------------------------------------------------------ tensor helpers
    def _t(self, tensor: torch.Tensor, name: str, buf: bool = True):
        tensor = tensor.to(self.device).contiguous()
        self.keep.append(tensor)
        if buf:
            self.bufs[name] = tensor
        dt = self.mpk.attach_input(torch_tensor=tensor, name=name)
        self._dts.append(dt)
        self._torch[id(dt)] = tensor
        return dt

    def tt(self, dt) -> torch.Tensor:
        return self._torch[id(dt)]

    def _new(self, name, shape, dtype=torch.bfloat16, fill=0.0):
        return self._t(torch.full(shape, fill, dtype=dtype), name)

    def _w(self, sd, key, name=None, shape=None, dtype=None):
        t = sd[key]
        if dtype is not None:
            t = t.to(dtype)
        if shape is not None:
            t = t.reshape(shape)
        return self._t(t, name or key.replace(".", "_"), buf=False)

    def _k(self, kernel, inputs, outputs, grid, tmpl, scalars=()):
        self.mpk.dsv4_kernel_layer(kernel, inputs, outputs, grid_dim=grid, block_dim=self.BLOCK,
                                   template_args=tmpl, scalars=list(scalars))

    N = (-1, -1, -1)  # not partitioned

    # ------------------------------------------------------------ kernel wrappers
    def gemv(self, x, w, out, out_fp32=False, idx=None, groups=1, x_per_group=False, max_tasks=128):
        """out = x @ w^T. w: [N, K] or, with groups / idx, [E, N, K]. out: [1, N] or [R, C]
        (R * C = N, split by rows) or, with groups, [G, N]."""
        wt = self.tt(w)
        k = wt.shape[-1]
        n = wt.shape[-2]
        if groups == 1 and idx is None:
            n = self.tt(out).numel()
            assert wt.numel() == n * k, (tuple(wt.shape), tuple(self.tt(out).shape))
            rows = self.tt(out).shape[0]
            if rows > 1:
                nt = _divisor_at_most(rows, max_tasks)
                omap = (-1, 0, -1)
            else:
                nt = _divisor_at_most(n, max_tasks)
                omap = (-1, 1, -1)
            ins = [(x, self.N), (w, (-1, 0, -1)), (self._dummy_idx, self.N)]
            grid = (1, nt, 1)
            wgs = 0
        else:
            nt = _divisor_at_most(n, max(1, max_tasks // groups))
            xmap = (0, -1, -1) if x_per_group else self.N
            if idx is not None:
                ins = [(x, xmap), (w, (-1, 1, -1)), (idx, (0, -1, -1))]
                wgs = n * k
            else:
                ins = [(x, xmap), (w, (0, 1, -1)), (self._dummy_idx, self.N)]
                wgs = 0
            omap = (0, 1, -1)
            grid = (groups, nt, 1)
        self._k("gemv", ins, [(out, omap)], grid,
                [k, n // nt, 1, 0, k, wgs, 0, int(idx is not None), int(out_fp32)])

    def rmsnorm(self, x, w, out):
        d = self.tt(w).shape[0]
        self._k("rmsnorm", [(x, self.N), (w, self.N)], [(out, self.N)], (1, 1, 1),
                [1, d, d, d], [float(self.eps)])

    def rope_quant(self, x, freqs, out, heads, inverse=False, quant=0, qblock=32, pos_shift=0,
                   slot_div=1, slot_mod=1, max_tasks=8, d=None):
        d = d or self.head_dim
        nt = _divisor_at_most(heads, max_tasks) if slot_mod == 1 else 1
        pmap = (0, -1, -1) if nt > 1 else self.N
        self._k("rope_quant", [(x, pmap), (freqs, self.N), (self.step_dt, self.N)], [(out, pmap)],
                (nt, 1, 1),
                [heads // nt, d, self.rope_dim, int(inverse), quant, qblock, pos_shift, slot_div,
                 slot_mod, d, d])

    # ------------------------------------------------------------ build
    def build(self, sd: dict, engram_tables: dict):
        mpk, cfg, dim, hc, mix = self.mpk, self.cfg, self.dim, self.hc, self.mix
        self.step_dt = self.step
        self._dummy_idx = self._new("dummy_idx", (1,), torch.int32)
        self._dummy_f32 = self._new("dummy_f32", (1,), torch.float32)
        self._dummy_kv = self._new("dummy_kv", (1, self.head_dim))
        L = cfg["num_hidden_layers"]

        # rope tables: pure sliding-window layers use base rope_theta without YaRN
        rs = cfg.get("rope_scaling") or {}
        self.freqs = {}
        for r in sorted(set(self.ratios)):
            if r:
                f = precompute_freqs(self.rope_dim, self.max_seq_len, rs.get("original_max_position_embeddings", 0),
                                     cfg["compress_rope_theta"], rs.get("factor", 1), rs.get("beta_fast", 32),
                                     rs.get("beta_slow", 1))
            else:
                f = precompute_freqs(self.rope_dim, self.max_seq_len, 0, cfg["rope_theta"], rs.get("factor", 1),
                                     rs.get("beta_fast", 32), rs.get("beta_slow", 1))
            self.freqs[r] = self._t(f, f"freqs_r{r}", buf=False)

        # embed, expanded to hc copies: hc_post with post = 1, comb = 0, zero residual
        embed = self._new("embed_out", (1, dim))
        w_embed = self._w(sd, "embed.weight")
        nt = 8
        self._k("embed", [(self.tokens, self.N), (self.step, self.N), (w_embed, (-1, 1, -1))],
                [(embed, (-1, 1, -1))], (1, nt, 1), [dim // nt, dim])
        expand_mix = torch.zeros(1, mix, dtype=torch.float32)
        expand_mix[0, hc:2 * hc] = 1.0
        expand_mix = self._t(expand_mix, "expand_mix", buf=False)
        zeros_stream = self._new("zeros_stream", (1, hc, dim))
        x = self._new("stream_embed", (1, hc, dim))
        mpk.dsv4_hc_post_layer(x=embed, residual=zeros_stream, mix=expand_mix, output=x,
                               grid_dim=(1, 4, 1), block_dim=self.BLOCK)
        pre_mix = torch.zeros(1, mix, dtype=torch.float32)
        pre_mix[0, 0] = 1.0
        pre_mix = self._t(pre_mix, "identity_pre_mix", buf=False)

        self.comp_cache = {}
        self.index_k_cache = {}
        for i in range(L):
            if i in cfg["engram_layer_ids"]:
                x = self._build_engram(i, sd, x, engram_tables[i])
            x, pre_mix = self._build_block(i, sd, x, pre_mix)

        # head
        xh = self._new("final_pre", (1, dim))
        mpk.dsv4_hc_pre_layer(x=x, mix=pre_mix, output=xh, grid_dim=(1, 4, 1), block_dim=self.BLOCK)
        xn = self._new("final_norm", (1, dim))
        self.rmsnorm(xh, self._w(sd, "norm.weight"), xn)
        logits = self._new("logits", (1, self.vocab), torch.float32)
        self.gemv(xn, self._w(sd, "head.weight"), logits, out_fp32=True)
        self._k("argmax", [(logits, self.N)], [(self.next_token, self.N)], (1, 1, 1), [self.vocab])

    def _build_engram(self, i, sd, x, tables):
        cfg, dim, hc = self.cfg, self.dim, self.hc
        p = f"layers.{i}.engram."
        token_map, mult, primes, offsets, pad_id = tables
        ngram, heads, hd = cfg["engram_max_ngram_size"], cfg["engram_n_heads"], cfg["engram_head_dim"]
        ncols = (ngram - 1) * heads
        ids = self._new(f"l{i}_engram_ids", (ncols,), torch.int32)
        self._k("engram_hash",
                [(self.tokens, self.N), (self._t(token_map, f"l{i}_engram_token_map", False), self.N),
                 (self._t(mult, f"l{i}_engram_mult", False), self.N),
                 (self._t(primes, f"l{i}_engram_primes", False), self.N),
                 (self._t(offsets, f"l{i}_engram_offsets", False), self.N), (self.step, self.N)],
                [(ids, self.N)], (1, 1, 1), [ngram, heads], [int(pad_id)])
        emb = self._new(f"l{i}_engram_emb", (ncols, hd))
        table = self._w(sd, p + "embed.weight")
        scale = self._t(sd[p + "embed.scale"].view(torch.uint8), f"l{i}_engram_scale", buf=False)
        self._k("engram_lookup", [(ids, self.N), (table, self.N), (scale, self.N)], [(emb, self.N)],
                (1, 1, 1), [ncols, hd])
        kv = self._new(f"l{i}_engram_kv", (1, dim * (hc + 1)))
        self.gemv(emb, self._w(sd, p + "wkv.weight"), kv)
        w = (sd[p + "q_weight"].float() * sd[p + "k_weight"].float())
        out = self._new(f"l{i}_engram_out", (1, hc, dim))
        self._k("engram_gate", [(x, self.N), (kv, self.N), (self._t(w, f"l{i}_engram_w", False), self.N)],
                [(out, self.N)], (1, 1, 1), [hc, dim], [float(self.eps), float(dim ** -0.5)])
        return out

    def _hc_mixes(self, i, sd, x, which):
        p = f"layers.{i}.hc_{which}_"
        mix = self._new(f"l{i}_{which}_mix", (1, self.mix), torch.float32)
        self.mpk.dsv4_hc_mixes_layer(
            x=x, fn=self._w(sd, p + "fn"), scale=self._w(sd, p + "scale"), base=self._w(sd, p + "base"),
            output=mix, grid_dim=(1, 1, 1), block_dim=self.BLOCK,
            sinkhorn_iters=self.cfg["hc_sinkhorn_iters"], norm_eps=float(self.eps),
            hc_eps=float(self.cfg["hc_eps"]))
        return mix

    def _build_block(self, i, sd, x, pre_mix):
        mpk, dim, hc = self.mpk, self.dim, self.hc
        # attention sublayer: its own mixes feed the post and the next sublayer's pre
        attn_mix = self._hc_mixes(i, sd, x, "attn")
        xa = self._new(f"l{i}_attn_in", (1, dim))
        mpk.dsv4_hc_pre_layer(x=x, mix=pre_mix, output=xa, grid_dim=(1, 4, 1), block_dim=self.BLOCK)
        xn = self._new(f"l{i}_attn_norm", (1, dim))
        self.rmsnorm(xa, self._w(sd, f"layers.{i}.attn_norm.weight"), xn)
        attn_out = self._build_attention(i, sd, xn)
        xs = self._new(f"l{i}_stream_attn", (1, hc, dim))
        mpk.dsv4_hc_post_layer(x=attn_out, residual=x, mix=attn_mix, output=xs,
                               grid_dim=(1, 4, 1), block_dim=self.BLOCK)

        ffn_mix = self._hc_mixes(i, sd, xs, "ffn")
        xf = self._new(f"l{i}_ffn_in", (1, dim))
        mpk.dsv4_hc_pre_layer(x=xs, mix=attn_mix, output=xf, grid_dim=(1, 4, 1), block_dim=self.BLOCK)
        xfn = self._new(f"l{i}_ffn_norm", (1, dim))
        self.rmsnorm(xf, self._w(sd, f"layers.{i}.ffn_norm.weight"), xfn)
        ffn_out = self._build_moe(i, sd, xfn)
        out = self._new(f"l{i}_stream_out", (1, hc, dim))
        mpk.dsv4_hc_post_layer(x=ffn_out, residual=xs, mix=ffn_mix, output=out,
                               grid_dim=(1, 4, 1), block_dim=self.BLOCK)
        return out, ffn_mix

    def _build_attention(self, i, sd, xn):
        cfg, dim = self.cfg, self.dim
        p = f"layers.{i}.attn."
        H, D = self.n_heads, self.head_dim
        ratio = self.ratios[i]
        freqs = self.freqs[ratio]

        qa = self._new(f"l{i}_qa", (1, cfg["q_lora_rank"]))
        self.gemv(xn, self._w(sd, p + "wq_a.weight"), qa)
        qn = self._new(f"l{i}_qa_norm", (1, cfg["q_lora_rank"]))
        self.rmsnorm(qa, self._w(sd, p + "q_norm.weight"), qn)
        q = self._new(f"l{i}_q", (H, D))
        self.gemv(qn, self._w(sd, p + "wq_b.weight", shape=(H, D, cfg["q_lora_rank"])), q, max_tasks=64)
        q_r = self._new(f"l{i}_q_rope", (H, D))
        self.rope_quant(q, freqs, q_r, heads=H)

        # sliding window: kv_norm(wkv x), RoPE, fp8 fake quant, into ring slot step % window
        kv = self._new(f"l{i}_kv", (1, D))
        self.gemv(xn, self._w(sd, p + "wkv.weight"), kv)
        kvn = self._new(f"l{i}_kv_norm", (1, D))
        self.rmsnorm(kv, self._w(sd, p + "kv_norm.weight"), kvn)
        win = self._new(f"l{i}_window_cache", (self.window, D))
        self.rope_quant(kvn, freqs, win, heads=1, quant=1, qblock=32, slot_mod=self.window)

        if ratio and i in self.kv_sources:
            n_comp = self.max_seq_len // ratio
            cache = self._new(f"l{i}_compress_cache", (n_comp, D))
            cp = p + "compressor."
            if ratio == 1:
                ckv = self._new(f"l{i}_comp_kv", (1, D))
                self.gemv(xn, self._w(sd, cp + "wkv.weight"), ckv)
                pooled = ckv
            else:
                ckv = self._new(f"l{i}_comp_kv", (1, D), torch.float32)
                csc = self._new(f"l{i}_comp_score", (1, D), torch.float32)
                self.gemv(xn, self._w(sd, cp + "wkv.weight"), ckv, out_fp32=True)
                self.gemv(xn, self._w(sd, cp + "wgate.weight"), csc, out_fp32=True)
                kv_state = self._new(f"l{i}_comp_kv_state", (ratio, D), torch.float32)
                sc_state = self._new(f"l{i}_comp_score_state", (ratio, D), torch.float32, float("-inf"))
                pooled = self._new(f"l{i}_comp_pooled", (1, D))
                self._k("compress_pool", [(ckv, self.N), (csc, self.N), (self.step, self.N)],
                        [(kv_state, self.N), (sc_state, self.N), (pooled, self.N)], (1, 1, 1), [D, ratio])
            cn = self._new(f"l{i}_comp_norm", (1, D))
            self.rmsnorm(pooled, self._w(sd, cp + "norm.weight"), cn)
            if self._use_indexer(i):
                # index keys come from the RoPE-free latent, one per closed group
                ip = p + "indexer."
                ihd = cfg["index_head_dim"]
                ik = self._new(f"l{i}_index_k", (1, ihd))
                self.gemv(cn, self._w(sd, ip + "wk.weight"), ik)
                ikn = self._new(f"l{i}_index_k_norm", (1, ihd))
                self.rmsnorm(ik, self._w(sd, ip + "k_norm.weight"), ikn)
                kc = self._new(f"l{i}_index_k_cache", (n_comp, ihd))
                self.rope_quant(ikn, freqs, kc, heads=1, d=ihd, quant=3, qblock=32,
                                pos_shift=1 - ratio, slot_div=ratio, slot_mod=n_comp)
                self.index_k_cache[ratio] = kc
            # a latent stands for the first token of its group: RoPE at step + 1 - ratio
            self.rope_quant(cn, freqs, cache, heads=1, quant=2, qblock=16, pos_shift=1 - ratio,
                            slot_div=ratio, slot_mod=n_comp)
            self.comp_cache[ratio] = cache
        comp = self.comp_cache.get(ratio) if ratio else None
        n_comp = self.tt(comp).shape[0] if comp is not None else 0

        selected, sel = 0, self._dummy_idx
        if comp is not None and self._use_indexer(i):
            sel = self._build_indexer(i, sd, xn, qn, freqs, ratio, n_comp)
            selected = self.index_topk

        o = self._new(f"l{i}_attn_o", (H, D))
        heads_per_task = 4
        nt = H // heads_per_task
        self._k("attention",
                [(q_r, (0, -1, -1)), (win, self.N), (comp if comp is not None else self._dummy_kv, self.N),
                 (sel, self.N), (self._w(sd, p + "attn_sink"), (0, -1, -1)), (self.step, self.N)],
                [(o, (0, -1, -1))], (nt, 1, 1),
                [heads_per_task, D, self.window, ratio, n_comp, selected, D, D], [float(D ** -0.5)])
        # remove the query rotation, fp8 fake quant, then the grouped low-rank output projection
        g = self.o_groups
        o_r = self._new(f"l{i}_attn_o_rope", (g, H * D // g))
        self.rope_quant(o, freqs, o_r, heads=H, inverse=True, quant=1, qblock=32, max_tasks=g)
        oa = self._new(f"l{i}_attn_oa", (g, self.o_rank))
        self.gemv(o_r, self._w(sd, p + "wo_a.weight", shape=(g, self.o_rank, H * D // g)), oa,
                  groups=g, x_per_group=True)
        out = self._new(f"l{i}_attn_out", (1, dim))
        self.gemv(oa, self._w(sd, p + "wo_b.weight"), out)
        return out

    def _use_indexer(self, i):
        r = self.ratios[i]
        return bool(r) and i in self.index_sources and self.max_seq_len // r > self.index_topk

    def _build_indexer(self, i, sd, xn, qn, freqs, ratio, n_comp):
        """Indexer.forward without candidate blocks: fp4 queries against the index keys, rectified
        scores mixed by weights_proj, the index_topk best compressed positions."""
        cfg = self.cfg
        ip = f"layers.{i}.attn.indexer."
        nh, hd = cfg["index_n_heads"], cfg["index_head_dim"]
        iq = self._new(f"l{i}_index_q", (nh, hd))
        self.gemv(qn, self._w(sd, ip + "wq_b.weight", shape=(nh, hd, cfg["q_lora_rank"])), iq, max_tasks=32)
        iq_r = self._new(f"l{i}_index_q_rope", (nh, hd))
        self.rope_quant(iq, freqs, iq_r, heads=nh, d=hd, quant=3, qblock=32)
        iw = self._new(f"l{i}_index_w", (1, nh))
        self.gemv(xn, self._w(sd, ip + "weights_proj.weight"), iw)
        kc = self.index_k_cache[ratio]
        scores = self._new(f"l{i}_index_scores", (n_comp,), torch.float32)
        nt = _divisor_at_most(n_comp, 64)
        self._k("index_score", [(iq_r, self.N), (kc, (0, -1, -1)), (iw, self.N)], [(scores, (0, -1, -1))],
                (nt, 1, 1), [nh, hd, n_comp // nt], [float(hd ** -0.5 * nh ** -0.5)])
        sel = self._new(f"l{i}_index_topk", (self.index_topk,), torch.int32)
        self._k("index_topk", [(scores, self.N), (self.step, self.N)], [(sel, self.N)], (1, 1, 1),
                [n_comp, self.index_topk, ratio])
        return sel

    def _build_moe(self, i, sd, xfn):
        dim, E, K, I = self.dim, self.n_experts, self.topk, self.inter
        p = f"layers.{i}.ffn."
        logits = self._new(f"l{i}_gate_logits", (1, E), torch.float32)
        self.gemv(xfn, self._w(sd, p + "gate.weight"), logits, out_fp32=True)
        weights = self._new(f"l{i}_route_weights", (K,), torch.float32)
        idx = self._new(f"l{i}_route_idx", (K,), torch.int32)
        self._k("router", [(logits, self.N), (self._w(sd, p + "gate.bias"), self.N)],
                [(weights, self.N), (idx, self.N)], (1, 1, 1), [E, K],
                [float(self.cfg["routed_scaling_factor"])])

        def stacked(w):
            keys = [f"{p}experts.{e}.{w}.weight" for e in range(E)]
            t = torch.stack([sd[k] for k in keys])
            for k in keys:
                del sd[k]
            return self._t(t, f"l{i}_experts_{w}", buf=False)

        limit = float(self.cfg["swiglu_limit"])
        gate = self._new(f"l{i}_moe_gate", (K, I))
        up = self._new(f"l{i}_moe_up", (K, I))
        self.gemv(xfn, stacked("w1"), gate, idx=idx, groups=K)
        self.gemv(xfn, stacked("w3"), up, idx=idx, groups=K)
        act = self._new(f"l{i}_moe_act", (K, I))
        self._k("swiglu", [(gate, self.N), (up, self.N), (weights, self.N)], [(act, self.N)], (1, 1, 1),
                [K, I, 1], [limit])
        eo = self._new(f"l{i}_moe_expert_out", (K, dim))
        self.gemv(act, stacked("w2"), eo, idx=idx, groups=K, x_per_group=True)

        sp = p + "shared_experts."
        sg = self._new(f"l{i}_shared_gate", (1, I))
        su = self._new(f"l{i}_shared_up", (1, I))
        self.gemv(xfn, self._w(sd, sp + "w1.weight"), sg)
        self.gemv(xfn, self._w(sd, sp + "w3.weight"), su)
        sa = self._new(f"l{i}_shared_act", (1, I))
        self._k("swiglu", [(sg, self.N), (su, self.N), (self._dummy_f32, self.N)], [(sa, self.N)],
                (1, 1, 1), [1, I, 0], [limit])
        so = self._new(f"l{i}_shared_out", (1, dim))
        self.gemv(sa, self._w(sd, sp + "w2.weight"), so)

        out = self._new(f"l{i}_ffn_out", (1, dim))
        nt = 8
        self._k("moe_combine", [(eo, (-1, 1, -1)), (idx, self.N), (so, (-1, 1, -1))],
                [(out, (-1, 1, -1))], (1, nt, 1), [K, dim // nt, dim])
        return out


# ---------------------------------------------------------------- MPK model registry entry
from ...model_registry import register_model_builder  # noqa: E402
from ..graph_builder import GraphBuilder  # noqa: E402


def load_tokenizer(model_dir: str):
    """The tokenizer alone: transformers cannot parse V4.1's config.json, so load it from a
    directory holding only the tokenizer files (as DeepSeek's convert.py does)."""
    import shutil
    import tempfile

    from transformers import AutoTokenizer

    tok_dir = tempfile.mkdtemp(prefix="dsv41_tokenizer_")
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json"):
        if os.path.exists(os.path.join(model_dir, f)):
            shutil.copy(os.path.join(model_dir, f), tok_dir)
    return AutoTokenizer.from_pretrained(tok_dir)


@register_model_builder("deepseek_v41", "DeepseekV41ForCausalLM", "CharyZeng/DeepSeek-V4.1-4layer")
class DeepSeekV41Builder(GraphBuilder):
    """DeepSeekV41DecodeBuilder behind MPK's model registry (MPK.build / mirage.engine.ModelRunner).

    Uses row 0 of MPK's meta tensors, so the kernel must serve one request per batch and one
    token per step: max_num_batched_requests == max_num_batched_tokens == 1."""

    # The graph ends in argmax: greedy only. Per-request max_new_tokens / stop ids still apply,
    # they are checked by the scheduler, not the sampler.
    per_request_sampling = False

    def __init__(self, mpk: PersistentKernel, weights=None):
        super().__init__(mpk, weights)
        self.tokenizer = None
        self.eos_token_id = -1

    def build_from_model(self, model_name: str, model_path: str | None = None):
        import json
        from glob import glob

        from safetensors.torch import load_file

        mpk = self.mpk
        assert mpk.world_size == 1, "DeepSeek-V4.1 on MPK: single GPU only"
        assert mpk.max_num_batched_requests == 1 and mpk.max_num_batched_tokens == 1, (
            "DeepSeek-V4.1 on MPK decodes one request, one token per step")
        path = model_path or model_name
        if not os.path.isdir(path):
            from huggingface_hub import snapshot_download
            path = snapshot_download(path)
        cfg = json.load(open(os.path.join(path, "config.json")))
        cfg = cfg.get("text_config", cfg)
        self.tokenizer = load_tokenizer(path)
        eos = cfg.get("eos_token_id", self.tokenizer.eos_token_id)
        self.eos_token_id = int(eos[0] if isinstance(eos, list) else eos)
        sd = {}
        for shard in sorted(glob(os.path.join(path, "*.safetensors"))):
            sd.update(load_file(shard, device="cuda"))
        meta = mpk.meta_tensors
        self.decode_builder = DeepSeekV41DecodeBuilder(
            mpk, cfg, max_seq_len=mpk.max_seq_length, tokens=meta["tokens"][0],
            step=meta["step"][:1], next_token=meta["output_tokens"].view(-1)[:1])
        self.decode_builder.build(sd, engram_hash_tables(cfg, self.tokenizer))
        self.vocab_size = cfg["vocab_size"]
        del sd
        torch.cuda.empty_cache()

    def decode(self, ids: torch.Tensor):
        return self.tokenizer.decode(ids)
