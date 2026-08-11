from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import shutil
from os import path

this_dir = os.path.dirname(os.path.abspath(__file__))

nvcc_path = shutil.which("nvcc")
if nvcc_path:
    cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
else:
    cuda_home = "/usr/local/cuda"

cuda_include_dir = os.path.join(cuda_home, "include")
cuda_library_dirs = [
    os.path.join(cuda_home, "lib"),
    os.path.join(cuda_home, "lib64"),
    os.path.join(cuda_home, "lib64", "stubs"),
]

macros=[("MIRAGE_BACKEND_USE_CUDA", None), ("MIRAGE_FINGERPRINT_USE_CUDA", None)]


def _sm100_family_arch():
    """Target the local GPU within the SM100 family (sm_100a cubins do not
    load on sm_103/B300 and vice versa)."""
    try:
        import torch
        major, minor = torch.cuda.get_device_capability()
        if major == 10:
            return f"-gencode=arch=compute_{major}{minor}a,code=sm_{major}{minor}a"
    except Exception:
        pass
    return "-gencode=arch=compute_100a,code=sm_100a"

setup(
    name='runtime_kernel_blackwell',
    ext_modules=[
        CUDAExtension(
            name='runtime_kernel_blackwell',
            sources=[
                os.path.join(this_dir, 'runtime_kernel_wrapper_sm100.cu'),
            ],
            depends=[
                os.path.join(this_dir, '../../../../include/mirage/persistent_kernel/tasks/blackwell/topk_softmax_sm100.cuh'),
                os.path.join(this_dir, '../../../../include/mirage/persistent_kernel/tasks/blackwell/moe_linear_sm100.cuh'),
                os.path.join(this_dir, '../../../../include/mirage/persistent_kernel/tasks/blackwell/mul_sum_add_sm100.cuh'),
                os.path.join(this_dir, '../../../../include/mirage/persistent_kernel/tasks/blackwell/utils.cuh'),
            ],
            define_macros=macros,
            include_dirs=[
                os.path.join(this_dir, '../../../../include/mirage/persistent_kernel/'),
                os.path.join(this_dir, '../../../../include/mirage/persistent_kernel/tasks/'),
                os.path.join(this_dir, '../../../../include'),
                os.path.join(this_dir, '../../../../deps/cutlass/include'),
                os.path.join(this_dir, '../../../../deps/cutlass/tools/util/include'),
            ],
            libraries=["cuda"],
            library_dirs=cuda_library_dirs,
            extra_compile_args={
                'cxx': ['-DMIRAGE_GRACE_BLACKWELL'],
                'nvcc': [
                    '-O3',
                    _sm100_family_arch(),
                    '-DMIRAGE_GRACE_BLACKWELL',
                    '-DMPK_ENABLE_TMA',
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
