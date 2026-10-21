"""DeepSeek-V4.1 indexer kernels (dsv4 "index_score" + "index_topk") against torch.

Scores follow Indexer.forward's bf16 einsum / relu / weighting / head sum; the top-k must pick the
same set as torch.topk on the kernel's own scores (ties aside), in ascending position order, and be
identical run to run.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

NH, HD, N, TOPK, RATIO = 32, 128, 2048, 512, 1


def ref_scores(q, k, w, scale):
    dot = (q.float() @ k.float().T).bfloat16()                       # [NH, N]
    wb = (w.float() * scale).bfloat16()
    prod = (dot.float().relu() * wb.float().view(-1, 1)).bfloat16()
    return prod.float().sum(0).bfloat16().float()


def main():
    torch.manual_seed(0)
    q = torch.randn(NH, HD, device="cuda").bfloat16()
    k = torch.randn(N, HD, device="cuda").bfloat16()
    w = torch.randn(1, NH, device="cuda").bfloat16()
    step = torch.zeros(1, dtype=torch.int32, device="cuda")
    scores = torch.zeros(N, dtype=torch.float32, device="cuda")
    sel = torch.zeros(TOPK, dtype=torch.int32, device="cuda")
    scale = HD ** -0.5 * NH ** -0.5

    nw, ns = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(test_mode=True, num_workers=nw, num_local_schedulers=ns, mpi_rank=0, world_size=1)
    pk = PersistentKernel(**params)
    a = lambda t, n: pk.attach_input(t, name=n)
    qd, kd, wd, std, scd, sld = a(q, "q"), a(k, "k"), a(w, "w"), a(step, "step"), a(scores, "scores"), a(sel, "sel")
    N_ = (-1, -1, -1)
    nt = 32
    pk.dsv4_kernel_layer("index_score", [(qd, N_), (kd, (0, -1, -1)), (wd, N_)], [(scd, (0, -1, -1))],
                         grid_dim=(nt, 1, 1), block_dim=(256, 1, 1), template_args=[NH, HD, N // nt],
                         scalars=[float(scale)])
    pk.dsv4_kernel_layer("index_topk", [(scd, N_), (std, N_)], [(sld, N_)], grid_dim=(1, 1, 1),
                         block_dim=(256, 1, 1), template_args=[N, TOPK, RATIO])
    pk.compile(output_dir=os.path.dirname(os.path.abspath(__file__)))

    def run(s):
        step.fill_(s)
        pk.meta_tensors["step"].zero_()
        pk.init_request_func()
        pk()
        torch.cuda.synchronize()
        return sel.clone()

    ok = True
    ref = ref_scores(q, k, w, scale)
    for s in (0, 100, 510, 511, 512, 1000, 2047):
        got = run(s)
        n = min((s + 1) // RATIO, N)
        rel = ((scores - ref).norm() / ref.norm()).item()
        exact = (scores == ref).float().mean().item()
        m = min(n, TOPK)
        picked = got[:m]
        valid = bool((got[m:] == -1).all()) and bool((picked[1:] > picked[:-1]).all() if m > 1 else True)
        if n <= TOPK:
            same = picked.tolist() == list(range(n))
        else:
            want = torch.topk(scores[:n], TOPK).indices
            cut = scores[:n][want].min()
            # every pick scores at least the cut, and everything above the cut is picked
            same = bool((scores[:n][picked.long()] >= cut).all()) and \
                set(torch.nonzero(scores[:n] > cut).view(-1).tolist()) <= set(picked.tolist())
        again = run(s)
        det = torch.equal(got, again)
        print(f"step {s:4d}: n={n:4d} scores rel={rel:.2e} bitwise={exact:.4f} topk ok={same} "
              f"ordered/padded={valid} deterministic={det}")
        ok &= rel < 1e-2 and same and valid and det
    pk.finalize()
    print("PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
