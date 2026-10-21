"""Can a test_mode kernel be launched repeatedly with host-updated inputs? (dsv4 step-by-step harness)"""
import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel


def main():
    torch.manual_seed(0)
    x = torch.randn(4, 4096, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(4096, dtype=torch.bfloat16, device="cuda")
    out = torch.zeros_like(x)
    nw, ns = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(test_mode=True, num_workers=nw, num_local_schedulers=ns, mpi_rank=0, world_size=1)
    pk = PersistentKernel(**params)
    xd, wd, od = pk.attach_input(x, name="x"), pk.attach_input(w, name="w"), pk.attach_input(out, name="out")
    pk.dsv4_kernel_layer("rmsnorm", [(xd, (0, -1, -1)), (wd, (-1, -1, -1))], [(od, (0, -1, -1))],
                         grid_dim=(4, 1, 1), block_dim=(256, 1, 1),
                         template_args=[1, 4096, 4096, 4096], scalars=[1e-20])
    pk.compile(output_dir=os.path.dirname(os.path.abspath(__file__)))
    ok = True
    for it in range(5):
        x.copy_(torch.randn_like(x))
        out.zero_()
        # test_mode retires its one request after each launch: re-admit it.
        pk.meta_tensors["step"].zero_()
        pk.init_request_func()
        pk()
        torch.cuda.synchronize()
        xf = x.float()
        ref = (w.float() * (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-20))).bfloat16()
        d = (out.float() - ref.float()).abs().max().item()
        eq = (out == ref).float().mean().item()
        print(f"launch {it}: max_abs={d:.3e} bitwise_equal={eq:.4f}")
        ok &= eq > 0.999
    pk.finalize()
    print("PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
