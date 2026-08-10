"""Qwen3-MoE (e.g. Qwen/Qwen3-30B-A3B) builder for the online engine.

Reuses the dense Qwen3Builder wholesale (embedding, attention, sampling /
logprob tail, deterministic o_proj plumbing) and swaps only the per-layer
MLP block for the MoE expert block:

    post_attn_layernorm -> router gate linear -> top-k softmax routing
    -> expert W13 group GEMM (bf16) -> silu_mul -> expert W2 group GEMM
    -> weighted top-k combine + residual (moe_mul_sum_add)

Layer wiring and grid sizes mirror the validated offline demo
(demo/qwen3/demo_30B_A3B.py); tensor allocation follows the online
DeepSeek-V3 MoE precedent (models/deepseek_v3/builder.py:_build_moe_mlp).

Architecture facts (Qwen3-30B-A3B HF config): 128 experts, top-8,
moe_intermediate_size 768, norm_topk_prob=true (the routing kernel
hard-codes renormalize=true, matching), mlp_only_layers=[] (every layer
is MoE), and NO shared expert.

Determinism status (phase a): the dense ops keep their deterministic
paths; the router gate is built as a single-tile linear (no split-K
tma_reduce_add, design hazard row 1) and the expert GEMMs / combine are
fixed-order by construction. The remaining known gap is the
arrival-order active-expert compaction inside topk_softmax (design
hazard row 5) — value-neutral today, gated in phase (b).
"""

import torch

from ..utils import grid_for_rmsnorm_linear_layer
from ..qwen3.builder import Qwen3Builder
from ...model_registry import register_model_builder
from ....core import bfloat16, float32, int32


@register_model_builder("Qwen3-MoE", "Qwen/Qwen3-30B-A3B")
class Qwen3MoeBuilder(Qwen3Builder):

    @staticmethod
    def _hf_model_class():
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            Qwen3MoeForCausalLM,
        )
        return Qwen3MoeForCausalLM

    def build_from_dict(self, state_dict: dict, with_lm_head: bool):
        if getattr(self, "model", None) is None:
            # build_from_config passes a bare state dict; the MoE builder
            # needs the HF config for num_experts / top-k and currently
            # only supports the build_from_model path.
            raise NotImplementedError(
                "Qwen3MoeBuilder requires the build_from_model path "
                "(HF config needed for num_experts / num_experts_per_tok)")
        if self.world_size != 1:
            raise NotImplementedError(
                "Qwen3MoeBuilder: expert-weight sharding not implemented; "
                "run with TP=1 (30B-A3B bf16 fits on one GPU)")
        if self.mpk.mode == "online_notoken":
            raise NotImplementedError(
                "Qwen3MoeBuilder: fixed-tensor (online_notoken) MoE "
                "intermediates not wired yet")
        cfg = self.model.config
        self.num_experts = cfg.num_experts
        self.num_experts_per_tok = cfg.num_experts_per_tok
        self.moe_intermediate_size = cfg.moe_intermediate_size
        # The SM100 routing kernel hard-codes renormalize=true
        # (task_register.cc, register_moe_topk_softmax_sm100_task) and
        # requires a power-of-two expert count.
        assert getattr(cfg, "norm_topk_prob", False), (
            "routing kernel renormalizes top-k weights; config disagrees")
        assert self.num_experts & (self.num_experts - 1) == 0, (
            f"topk_softmax kernel needs power-of-2 experts, "
            f"got {self.num_experts}")
        # Dense-fallback layers (mlp_only_layers / decoder_sparse_step)
        # are not wired; Qwen3-30B-A3B uses MoE in every layer.
        assert not getattr(cfg, "mlp_only_layers", []), (
            "mlp_only_layers not supported")
        assert getattr(cfg, "decoder_sparse_step", 1) == 1, (
            "decoder_sparse_step != 1 not supported")
        self._fuse_expert_weights(state_dict)
        super().build_from_dict(state_dict, with_lm_head)

    def _fuse_expert_weights(self, state_dict: dict):
        """Stack per-expert HF weights into the fused [E, 2I, H] gate_up and
        [E, H, I] down tensors the moe_w13/w2 kernels expect (same layout as
        the offline demo: gate rows first, then up rows).

        Frees the per-expert originals layer by layer (drop the HF expert
        module + the state-dict refs) so peak VRAM overhead stays ~one fused
        layer (~1.2 GB for 30B-A3B) instead of a full second copy.
        attach_input does not hold tensor references, so the fused stacks
        are kept alive on the builder via self.shuffled_tensors.
        """
        E = self.num_experts
        inter = self.moe_intermediate_size
        hidden = self.hidden_size
        for i in range(self.num_layers):
            prefix = f"model.layers.{i}.mlp.experts"
            g0 = state_dict[f"{prefix}.0.gate_proj.weight"]
            assert g0.shape == (inter, hidden), g0.shape
            gate_up = torch.empty((E, 2 * inter, hidden),
                                  dtype=g0.dtype, device=g0.device)
            down = torch.empty((E, hidden, inter),
                               dtype=g0.dtype, device=g0.device)
            for e in range(E):
                gate_up[e, :inter].copy_(
                    state_dict.pop(f"{prefix}.{e}.gate_proj.weight"))
                gate_up[e, inter:].copy_(
                    state_dict.pop(f"{prefix}.{e}.up_proj.weight"))
                down[e].copy_(
                    state_dict.pop(f"{prefix}.{e}.down_proj.weight"))
            # Drop the HF expert module so the original storages free now.
            del self.model.model.layers[i].mlp.experts
            self.shuffled_tensors[f"layer_{i}_moe_gate_up_proj"] = gate_up
            self.shuffled_tensors[f"layer_{i}_moe_down_proj"] = down
            torch.cuda.empty_cache()

    def new_intermediate_tensors(self):
        super().new_intermediate_tensors()
        # MoE routing + expert intermediates, shared across layers exactly
        # like the dense intermediates above (offline-demo pattern; layers
        # execute sequentially so reuse is safe).
        mbt = self.max_num_batched_tokens
        E = self.num_experts
        topk = self.num_experts_per_tok
        self.moe_gate_out = self.mpk.new_tensor(
            dims=(mbt, E), dtype=bfloat16,
            name="moe_gate_out", io_category="cuda_tensor")
        self.moe_topk_weight = self.mpk.new_tensor(
            dims=(mbt, topk), dtype=float32,
            name="moe_topk_weight", io_category="cuda_tensor")
        self.moe_routing_indices = self.mpk.new_tensor(
            dims=(E, mbt), dtype=int32,
            name="moe_routing_indices", io_category="cuda_tensor")
        self.moe_mask = self.mpk.new_tensor(
            dims=(E + 1,), dtype=int32,
            name="moe_mask", io_category="cuda_tensor")
        self.moe_mid = self.mpk.new_tensor(
            dims=(mbt, topk, 2 * self.moe_intermediate_size),
            dtype=bfloat16, name="moe_mid", io_category="cuda_tensor")
        self.moe_silu_out = self.mpk.new_tensor(
            dims=(mbt, topk, self.moe_intermediate_size),
            dtype=bfloat16, name="moe_silu_out", io_category="cuda_tensor")
        self.moe_down_out = self.mpk.new_tensor(
            dims=(mbt, topk, self.hidden_size),
            dtype=bfloat16, name="moe_down_out", io_category="cuda_tensor")
        self.moe_weighted_sum_out = self.mpk.new_tensor(
            dims=(mbt, self.hidden_size), dtype=bfloat16,
            name="moe_weighted_sum_out",
            io_category="nvshmem_tensor" if self.world_size > 1
            else "cuda_tensor")

    def _build_mlp_block(self, i: int, state_dict: dict,
                         use_splitk: bool, deterministic: bool):
        """MoE expert block replacing the dense MLP.

        use_splitk is intentionally ignored here: the router gate output is
        tiny (N=num_experts), and the split-K combine (tma_reduce_add) is
        the arrival-order-nondeterministic hazard upstream of top-k
        selection (design hazard row 1) — the single-tile linear family is
        both sufficient and deterministic by construction.
        """
        prefix = f"model.layers.{i}."
        w_norm = self.mpk.attach_input(
            torch_tensor=state_dict[f"{prefix}post_attention_layernorm.weight"],
            name=f"layer_{i}_post_attn_layernorm",
        )
        w_moe_gate = self.mpk.attach_input(
            torch_tensor=state_dict[f"{prefix}mlp.gate.weight"],
            name=f"layer_{i}_moe_gate",
        )
        w_gate_up = self.mpk.attach_input(
            torch_tensor=self.shuffled_tensors[f"layer_{i}_moe_gate_up_proj"],
            name=f"layer_{i}_moe_gate_up_proj",
        )
        w_down = self.mpk.attach_input(
            torch_tensor=self.shuffled_tensors[f"layer_{i}_moe_down_proj"],
            name=f"layer_{i}_moe_down_proj",
        )
        self.mpk.rmsnorm_layer(
            input=self.x,
            weight=w_norm,
            output=self.rmsnorm_out,
            grid_dim=(self.mpk.max_num_batched_tokens, 1, 1),
            block_dim=(128, 1, 1),
        )
        # Router gate: keep >=8 bf16 outputs per task (16B TMA alignment;
        # same rule as the DeepSeek-V3 router).
        router_grid = min(grid_for_rmsnorm_linear_layer(self.num_experts),
                          self.num_experts // 8)
        self.mpk.linear_layer(
            input=self.rmsnorm_out,
            weight=w_moe_gate,
            output=self.moe_gate_out,
            grid_dim=(router_grid, 1, 1),
            block_dim=(128, 1, 1),
        )
        self.mpk.moe_topk_softmax_routing_layer(
            input=self.moe_gate_out,
            output=(self.moe_topk_weight, self.moe_routing_indices,
                    self.moe_mask),
            grid_dim=(1, 1, 1),
            block_dim=(256, 1, 1),  # 8 warps required by the topk kernel
        )
        # Expert GEMM grids follow the validated offline demo
        # (demo/qwen3/demo_30B_A3B.py): grid.x is the expert stride,
        # grid.y tiles the output width in 128-column tiles.
        self.mpk.moe_w13_linear_layer(
            input=self.rmsnorm_out,
            weight=w_gate_up,
            moe_routing_indices=self.moe_routing_indices,
            moe_mask=self.moe_mask,
            output=self.moe_mid,
            grid_dim=(10, 2 * self.moe_intermediate_size // 128, 1),
            block_dim=(128, 1, 1),
        )
        self.mpk.moe_silu_mul_layer(
            input=self.moe_mid,
            output=self.moe_silu_out,
            grid_dim=(self.mpk.max_num_batched_tokens,
                      self.num_experts_per_tok, 1),
            block_dim=(128, 1, 1),
        )
        self.mpk.moe_w2_linear_layer(
            input=self.moe_silu_out,
            weight=w_down,
            moe_routing_indices=self.moe_routing_indices,
            moe_mask=self.moe_mask,
            output=self.moe_down_out,
            grid_dim=(8, self.hidden_size // 128, 1),
            block_dim=(128, 1, 1),
        )
        # Weighted top-k combine + residual: fixed-order fp32 accumulation
        # inside the kernel (no arrival-order MoE combine in MPK).
        self.mpk.moe_mul_sum_add_layer(
            input=self.moe_down_out,
            weight=self.moe_topk_weight,
            residual=self.x,
            output=self.moe_weighted_sum_out,
            grid_dim=(self.mpk.max_num_batched_tokens,
                      self.hidden_size // 256, 1),
            block_dim=(128, 1, 1),
        )
        self.x = self.moe_weighted_sum_out
        if self.world_size > 1:
            self.mpk.allreduce_layer(
                input=self.moe_weighted_sum_out,
                buffer=self.allreduce_buf,
                output=self.mlp_final,
                grid_dim=(self.hidden_size // 64, 1, 1),
                block_dim=(128, 1, 1),
            )
            self.x = self.mlp_final
