"""DeepSeek-V4.1 decode attention (dsv4 "attention" kernel) against torch, over a range of positions.

Keys are the sliding-window ring (slots <= step valid) followed by the compressed cache (rows
< (step + 1) / ratio valid), with a per-head sink in the softmax denominator, as sparse_attn in
DeepSeek's inference/kernel.py when every position is selected.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

H, D, WINDOW, RATIO, MAXC = 64, 512, 128, 1, 512
HEADS_PER_TASK = 4


def reference(q, win, comp, sink, step, scale):
    n_win = min(step + 1, WINDOW)
    n_comp = (step + 1) // RATIO
    kv = torch.cat([win[:n_win], comp[:n_comp]]).float()
    s = q.float() @ kv.T * scale                                   # [H, keys]
    m = s.amax(-1, keepdim=True)
    e = torch.exp(s - m)
    denom = e.sum(-1, keepdim=True) + torch.exp(sink.view(-1, 1) - m)
    return ((e.bfloat16().float() @ kv) / denom).bfloat16()


def main():
    torch.manual_seed(0)
    q = (torch.randn(H, D, device="cuda") * 2).bfloat16()
    win = torch.randn(WINDOW, D, device="cuda").bfloat16()
    comp = torch.randn(MAXC, D, device="cuda").bfloat16()
    sink = torch.randn(H, device="cuda")
    step = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.zeros(H, D, dtype=torch.bfloat16, device="cuda")
    scale = D ** -0.5

    nw, ns = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(test_mode=True, num_workers=nw, num_local_schedulers=ns, mpi_rank=0, world_size=1)
    pk = PersistentKernel(**params)
    a = lambda t, n: pk.attach_input(t, name=n)
    qd, wd, cd, sd, std, od = a(q, "q"), a(win, "win"), a(comp, "comp"), a(sink, "sink"), a(step, "step"), a(out, "out")
    N = (-1, -1, -1)
    pk.dsv4_kernel_layer("attention", [(qd, (0, -1, -1)), (wd, N), (cd, N), (sd, (0, -1, -1)), (std, N)],
                         [(od, (0, -1, -1))], grid_dim=(H // HEADS_PER_TASK, 1, 1), block_dim=(256, 1, 1),
                         template_args=[HEADS_PER_TASK, D, WINDOW, RATIO, MAXC, D, D], scalars=[float(scale)])
    pk.compile(output_dir=os.path.dirname(os.path.abspath(__file__)))
    ok = True
    for s in (0, 1, 31, 63, 64, 100, 127, 128, 159, 160, 200, 300, 511):
        step.fill_(s)
        out.zero_()
        pk.meta_tensors["step"].zero_()
        pk.init_request_func()
        pk()
        torch.cuda.synchronize()
        ref = reference(q, win, comp, sink, s, scale)
        r = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
        print(f"step {s:3d}: keys={min(s + 1, WINDOW) + (s + 1) // RATIO:4d} rel={r:.3e}")
        ok &= r < 5e-3
    pk.finalize()
    print("PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
