"""DeepSeek-V4.1 quantized GEMV (dsv4 "qgemv") against torch.

The reference's linear() on fp8 / fp4 weights: the input is fp8-quantized per 32 with ue8m0 scales,
weights are fp8 with one scale per 32x32 block, or e2m1 (two per byte, low nibble first) with one
scale per 32 along K. Covers a plain fp8 projection, fp4 experts picked by idx, an fp4 down
projection with per-expert inputs, and (with a second GPU) weights read from a peer GPU's memory.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

FP4 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def act_quant(x):
    """act_quant(x, 32, ue8m0, inplace=True) on fp32 values of a bf16 row."""
    b = x.float().view(*x.shape[:-1], -1, 32)
    amax = b.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    return ((b / s).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s).view_as(x.float())


def deq_fp8(w, s):
    n, k = w.shape
    return (w.float().view(n // 32, 32, k // 32, 32) * s.float()[:, None, :, None]).view(n, k)


def deq_fp4(w, s):
    n, kh = w.shape
    u = w.view(torch.uint8).long()
    v = torch.stack([FP4.to(w.device)[u & 15], FP4.to(w.device)[u >> 4]], -1).view(n, kh * 2)
    return (v.view(n, -1, 32) * s.float()[..., None]).view(n, kh * 2)


def rand_scales(*shape, device):
    # ue8m0 bytes around 2^-6 .. 2^-3
    return torch.randint(121, 125, shape, dtype=torch.uint8, device=device).view(torch.float8_e8m0fnu)


def main():
    torch.manual_seed(0)
    dev = "cuda:0"
    peer = "cuda:1" if torch.cuda.device_count() > 1 and torch.cuda.can_device_access_peer(0, 1) else None
    if peer:
        # the persistent kernel reads the peer's memory directly
        torch.ones(1, device=peer).copy_(torch.ones(1, device=dev))
        torch.ones(1, device=dev).copy_(torch.ones(1, device=peer))

    # fp8 projection: [N, K] = [1280, 5120]
    N8, K8 = 1280, 5120
    x8 = torch.randn(1, K8, device=dev).bfloat16()
    w8 = (torch.randn(N8, K8, device=dev) * 2).to(torch.float8_e4m3fn)
    s8 = rand_scales(N8 // 32, K8 // 32, device=dev)
    o8 = torch.zeros(1, N8, dtype=torch.bfloat16, device=dev)

    # fp4 experts: E x [I, D/2], 6 picked by idx; then down projection E x [D, I/2]
    E, TOPK, D, I = 16, 6, 5120, 2304
    xe = torch.randn(1, D, device=dev).bfloat16()
    wedev = peer or dev
    w1 = torch.randint(0, 256, (E, I, D // 2), dtype=torch.uint8, device=wedev)
    s1 = rand_scales(E, I, D // 32, device=wedev)
    w2 = torch.randint(0, 256, (E, D, I // 2), dtype=torch.uint8, device=dev)
    s2 = rand_scales(E, D, I // 32, device=dev)
    idx = torch.tensor([3, 0, 15, 7, 7, 9], dtype=torch.int32, device=dev)
    h = torch.zeros(TOPK, I, dtype=torch.bfloat16, device=dev)
    xd = torch.randn(TOPK, I, device=dev).bfloat16()
    od = torch.zeros(TOPK, D, dtype=torch.bfloat16, device=dev)

    nw, ns = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(test_mode=True, num_workers=nw, num_local_schedulers=ns, mpi_rank=0, world_size=1)
    pk = PersistentKernel(**params)
    a = lambda t, n: pk.attach_input(t, name=n)
    N_ = (-1, -1, -1)
    u8 = lambda t: t.view(torch.uint8)
    dummy = a(torch.zeros(1, dtype=torch.int32, device=dev), "dummy")

    nt = 40
    pk.dsv4_kernel_layer("qgemv", [(a(x8, "x8"), N_), (a(u8(w8), "w8"), (-1, 0, -1)),
                                   (a(u8(s8), "s8"), (-1, 0, -1)), (dummy, N_)],
                         [(a(o8, "o8"), (-1, 1, -1))], grid_dim=(1, nt, 1), block_dim=(256, 1, 1),
                         template_args=[K8, N8 // nt, 1, 0, K8, 0, K8 // 32, 0, 0, 0, 0, 1])
    nt = 16
    pk.dsv4_kernel_layer("qgemv", [(a(xe, "xe"), N_), (a(w1, "w1"), (-1, 1, -1)), (a(u8(s1), "s1"), (-1, 1, -1)),
                                   (a(idx, "idx"), (0, -1, -1))],
                         [(a(h, "h"), (0, 1, -1))], grid_dim=(TOPK, nt, 1), block_dim=(256, 1, 1),
                         template_args=[D, I // nt, 1, 0, D // 2, I * D // 2, D // 32, I * D // 32, 0, 1, 0, 2])
    nt = 20
    pk.dsv4_kernel_layer("qgemv", [(a(xd, "xd"), (0, -1, -1)), (a(w2, "w2"), (-1, 1, -1)),
                                   (a(u8(s2), "s2"), (-1, 1, -1)), (a(idx, "idx2"), (0, -1, -1))],
                         [(a(od, "od"), (0, 1, -1))], grid_dim=(TOPK, nt, 1), block_dim=(256, 1, 1),
                         template_args=[I, D // nt, 1, 0, I // 2, D * I // 2, I // 32, D * I // 32, 0, 1, 0, 2])
    pk.compile(output_dir=os.path.dirname(os.path.abspath(__file__)))
    pk.meta_tensors["step"].zero_()
    pk.init_request_func()
    pk()
    torch.cuda.synchronize()

    def check(name, got, want):
        rel = ((got.float() - want).norm() / want.norm()).item()
        exact = (got == want.bfloat16()).float().mean().item()
        print(f"{name}: rel={rel:.2e} bf16-exact={exact:.4f}")
        return rel < 2e-3

    ok = check("fp8 [1280x5120]", o8, act_quant(x8) @ deq_fp8(w8, s8).T)
    w1c, s1c = w1.to(dev), s1.to(dev)
    want_h = torch.stack([deq_fp4(w1c[e], s1c[e]) @ act_quant(xe)[0] for e in idx.tolist()])
    ok &= check(f"fp4 experts by idx [6x2304x5120]{' (peer GPU)' if peer else ''}", h, want_h)
    want_d = torch.stack([deq_fp4(w2[e], s2[e]) @ act_quant(xd[g]) for g, e in enumerate(idx.tolist())])
    ok &= check("fp4 down per expert [6x5120x2304]", od, want_d)
    pk.finalize()
    print("PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
