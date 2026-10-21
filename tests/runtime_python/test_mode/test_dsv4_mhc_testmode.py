"""DeepSeek-V4.1 hyper-connection tasks (dsv4_hc_mixes / hc_pre / hc_post) in test_mode.

Reference: Block.hc_mixes / hc_pre / hc_post and hc_split_sinkhorn from DeepSeek's
inference/model.py and kernel.py, transcribed to plain torch below.

By default the inputs are random. With DSV41_CKPT (the 4-layer slice) and DSV41_GOLDEN (dumps
from dsv41_ref_dump.py) set, layer 0's real hc weights and the real prefill stream are used.
"""

import json
import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

HC = 4
MIX = (2 + HC) * HC


def ref_split_sinkhorn(mixes, scale, base, iters, eps):
    pre = torch.sigmoid(mixes[:, :HC] * scale[0] + base[:HC]) + eps
    post = 2 * torch.sigmoid(mixes[:, HC:2 * HC] * scale[1] + base[HC:2 * HC])
    comb = (mixes[:, 2 * HC:] * scale[2] + base[2 * HC:]).view(-1, HC, HC)
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return torch.cat([pre, post, comb.flatten(1)], dim=-1)


def ref_hc_mixes(x, fn, scale, base, iters, norm_eps, hc_eps):
    xf = x.flatten(1).float()
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + norm_eps)
    mixes = torch.nn.functional.linear(xf, fn) * rsqrt
    return ref_split_sinkhorn(mixes, scale, base, iters, hc_eps)


def ref_hc_pre(x, mix):
    return torch.sum(mix[:, :HC].unsqueeze(-1) * x.float(), dim=1).to(x.dtype)


def ref_hc_post(x, residual, mix):
    post = mix[:, HC:2 * HC]
    comb = mix[:, 2 * HC:].view(-1, HC, HC)
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=1)
    return y.type_as(x)


def load_inputs(tokens, dim, device):
    ckpt, golden = os.environ.get("DSV41_CKPT"), os.environ.get("DSV41_GOLDEN")
    if not (ckpt and golden):
        torch.manual_seed(0)
        stream = torch.randn(tokens, HC, dim, dtype=torch.bfloat16, device=device)
        fn = torch.randn(MIX, HC * dim, dtype=torch.float32, device=device) * 0.01
        scale = torch.rand(3, device=device) + 0.5
        base = torch.randn(MIX, device=device)
        sub_out = torch.randn(tokens, dim, dtype=torch.bfloat16, device=device)
        return stream, fn, scale, base, sub_out, 1e-6, "random"
    from safetensors import safe_open

    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    cfg = cfg.get("text_config", cfg)
    idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    names = {k: f"layers.0.hc_attn_{k}" for k in ("fn", "scale", "base")}
    w = {}
    for k, n in names.items():
        with safe_open(os.path.join(ckpt, idx[n]), "pt", device=device) as f:
            w[k] = f.get_tensor(n).float()
    rec = {r["name"]: r for r in torch.load(os.path.join(golden, "prefill.pt"))["records"]}
    stream = rec["layers.0"]["inputs"][0][0, :tokens].to(device)  # [tokens, hc, dim] bf16
    sub_out = rec["layers.0.attn"]["output"][0, :tokens].to(device)
    return stream, w["fn"], w["scale"], w["base"], sub_out, cfg["rms_norm_eps"], "layer 0 prefill"


def test_dsv4_mhc_testmode():
    device = "cuda"
    tokens, dim, iters, hc_eps = 8, 5120, 20, 1e-6
    stream, fn, scale, base, sub_out, norm_eps, source = load_inputs(tokens, dim, device)
    assert stream.shape == (tokens, HC, dim), stream.shape
    print(f"inputs: {source}, norm_eps={norm_eps}")

    mix = torch.zeros(tokens, MIX, dtype=torch.float32, device=device)
    pre_out = torch.zeros(tokens, dim, dtype=torch.bfloat16, device=device)
    post_out = torch.zeros(tokens, HC, dim, dtype=torch.bfloat16, device=device)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(test_mode=True, num_workers=num_workers, num_local_schedulers=num_schedulers,
                  mpi_rank=0, world_size=1)
    pk = PersistentKernel(**params)
    x_dt = pk.attach_input(stream, name="stream")
    fn_dt = pk.attach_input(fn.contiguous(), name="hc_fn")
    scale_dt = pk.attach_input(scale.contiguous(), name="hc_scale")
    base_dt = pk.attach_input(base.contiguous(), name="hc_base")
    sub_dt = pk.attach_input(sub_out, name="sub_out")
    mix_dt = pk.attach_input(mix, name="mix")
    pre_dt = pk.attach_input(pre_out, name="pre_out")
    post_dt = pk.attach_input(post_out, name="post_out")

    block = (256, 1, 1)
    pk.dsv4_hc_mixes_layer(x=x_dt, fn=fn_dt, scale=scale_dt, base=base_dt, output=mix_dt,
                           grid_dim=(tokens, 1, 1), block_dim=block, sinkhorn_iters=iters,
                           norm_eps=norm_eps, hc_eps=hc_eps)
    pk.dsv4_hc_pre_layer(x=x_dt, mix=mix_dt, output=pre_dt, grid_dim=(tokens, 4, 1), block_dim=block)
    pk.dsv4_hc_post_layer(x=sub_dt, residual=x_dt, mix=mix_dt, output=post_dt,
                          grid_dim=(tokens, 4, 1), block_dim=block)
    pk.compile(output_dir=os.path.dirname(os.path.abspath(__file__)))
    pk()
    torch.cuda.synchronize()

    ref_mix = ref_hc_mixes(stream, fn, scale, base, iters, norm_eps, hc_eps)
    # hc_pre / hc_post are checked on the kernel's own mix so their error is their own.
    ref_pre = ref_hc_pre(stream, mix)
    ref_post = ref_hc_post(sub_out, stream, mix)

    ok = True

    def check(name, got, want, atol, rtol):
        nonlocal ok
        diff = (got.float() - want.float()).abs()
        bad = diff > atol + rtol * want.float().abs()
        exact = (got == want).float().mean().item()
        print(f"{name:8s} max_abs={diff.max().item():.3e} bitwise_equal={exact:.4f} violations={int(bad.sum())}")
        ok &= not bool(bad.any())

    check("pre", mix[:, :HC], ref_mix[:, :HC], 1e-5, 1e-4)
    check("post", mix[:, HC:2 * HC], ref_mix[:, HC:2 * HC], 1e-5, 1e-4)
    check("comb", mix[:, 2 * HC:], ref_mix[:, 2 * HC:], 1e-5, 1e-4)
    # bf16 outputs: allow one ulp of rounding difference.
    check("hc_pre", pre_out, ref_pre, 1e-6, 2 ** -7)
    check("hc_post", post_out, ref_post, 1e-6, 2 ** -7)
    pk.finalize()
    if not ok:
        print("FAILED")
        sys.exit(1)
    print("PASSED: dsv4 mHC tasks match the reference")


if __name__ == "__main__":
    test_dsv4_mhc_testmode()
