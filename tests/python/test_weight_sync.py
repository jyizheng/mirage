import pytest

torch = pytest.importorskip("torch")

from mirage.mpk.weight_sync import (
    build_name_matching_sync_plan,
    build_qwen3_mpk_sync_plan,
)


def test_name_matching_sync_ties_lm_head_to_embeddings():
    src = {
        "model.embed_tokens.weight": torch.arange(12, dtype=torch.float32).view(3, 4),
        "model.norm.weight": torch.arange(4, dtype=torch.float32),
    }
    dst = {
        "model.embed_tokens.weight": torch.zeros(3, 4),
        "model.norm.weight": torch.zeros(4),
        "lm_head.weight": torch.zeros(5, 4),
    }

    plan = build_name_matching_sync_plan(src, dst)
    report = plan.sync(src, dst)

    assert report.tensors == 3
    assert torch.equal(dst["model.embed_tokens.weight"], src["model.embed_tokens.weight"])
    assert torch.equal(dst["model.norm.weight"], src["model.norm.weight"])
    assert torch.equal(dst["lm_head.weight"][:3], src["model.embed_tokens.weight"])
    assert torch.equal(dst["lm_head.weight"][3:], torch.zeros(2, 4))


def test_qwen3_plan_maps_online_attached_names():
    src = _qwen3_source(num_layers=1)
    dst = {
        "embed_tokens": torch.zeros_like(src["model.embed_tokens.weight"]),
        "model_norm_weight": torch.zeros_like(src["model.norm.weight"]),
        "lm_head": torch.zeros(6, 4),
        "layer_0_q_proj": torch.zeros_like(src["model.layers.0.self_attn.q_proj.weight"]),
        "layer_0_k_proj": torch.zeros_like(src["model.layers.0.self_attn.k_proj.weight"]),
        "layer_0_v_proj": torch.zeros_like(src["model.layers.0.self_attn.v_proj.weight"]),
        "layer_0_o_proj": torch.zeros_like(src["model.layers.0.self_attn.o_proj.weight"]),
        "layer_0_gate_proj": torch.zeros_like(src["model.layers.0.mlp.gate_proj.weight"]),
        "layer_0_up_proj": torch.zeros_like(src["model.layers.0.mlp.up_proj.weight"]),
        "layer_0_down_proj": torch.zeros_like(src["model.layers.0.mlp.down_proj.weight"]),
        "layer_0_input_layernorm": torch.zeros_like(src["model.layers.0.input_layernorm.weight"]),
        "layer_0_post_attn_layernorm": torch.zeros_like(
            src["model.layers.0.post_attention_layernorm.weight"]),
        "layer_0_q_norm": torch.zeros_like(src["model.layers.0.self_attn.q_norm.weight"]),
        "layer_0_k_norm": torch.zeros_like(src["model.layers.0.self_attn.k_norm.weight"]),
    }

    plan = build_qwen3_mpk_sync_plan(src, dst)
    report = plan.sync(src, dst)

    assert report.tensors == len(dst)
    assert torch.equal(dst["embed_tokens"], src["model.embed_tokens.weight"])
    assert torch.equal(dst["lm_head"][:5], src["lm_head.weight"])
    assert torch.equal(dst["lm_head"][5:], torch.zeros(1, 4))
    assert torch.equal(dst["layer_0_down_proj"], src["model.layers.0.mlp.down_proj.weight"])


def test_qwen3_plan_slices_tensor_parallel_targets():
    src = _qwen3_source(num_layers=1)
    dst = {
        "layer_0_q_proj": torch.zeros(2, 4),
        "layer_0_down_proj": torch.zeros(4, 3),
    }

    plan = build_qwen3_mpk_sync_plan(src, dst, rank=1, world_size=2)
    plan.sync(src, dst)

    assert torch.equal(
        dst["layer_0_q_proj"],
        src["model.layers.0.self_attn.q_proj.weight"][2:4],
    )
    assert torch.equal(
        dst["layer_0_down_proj"],
        src["model.layers.0.mlp.down_proj.weight"][:, 3:6],
    )


def _qwen3_source(num_layers: int):
    src = {
        "model.embed_tokens.weight": torch.arange(20, dtype=torch.float32).view(5, 4),
        "lm_head.weight": torch.arange(20, 40, dtype=torch.float32).view(5, 4),
        "model.norm.weight": torch.arange(4, dtype=torch.float32),
    }
    for i in range(num_layers):
        p = f"model.layers.{i}."
        src[p + "input_layernorm.weight"] = torch.arange(4, dtype=torch.float32) + 10
        src[p + "post_attention_layernorm.weight"] = torch.arange(4, dtype=torch.float32) + 20
        src[p + "self_attn.q_proj.weight"] = torch.arange(16, dtype=torch.float32).view(4, 4)
        src[p + "self_attn.k_proj.weight"] = torch.arange(8, dtype=torch.float32).view(2, 4)
        src[p + "self_attn.v_proj.weight"] = torch.arange(8, 16, dtype=torch.float32).view(2, 4)
        src[p + "self_attn.o_proj.weight"] = torch.arange(16, dtype=torch.float32).view(4, 4)
        src[p + "self_attn.q_norm.weight"] = torch.arange(2, dtype=torch.float32)
        src[p + "self_attn.k_norm.weight"] = torch.arange(2, dtype=torch.float32)
        src[p + "mlp.gate_proj.weight"] = torch.arange(24, dtype=torch.float32).view(6, 4)
        src[p + "mlp.up_proj.weight"] = torch.arange(24, 48, dtype=torch.float32).view(6, 4)
        src[p + "mlp.down_proj.weight"] = torch.arange(24, dtype=torch.float32).view(4, 6)
    return src
