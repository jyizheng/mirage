# Ferret gate harness build.
#
# Adapted from tests/runtime_python/blackwell/sm100_linear/setup.py:
#   - nvcc/CUDA_HOME taken from the environment (the original hardcoded
#     /usr/local/cuda-12.8, which does not exist on this pod; CUDA 13.0 lives
#     at /usr/local/cuda).
#   - arch is compute_103a/sm_103a (torch reports capability (10,3) on this
#     pod; nvidia-smi's device name is wrong -- trust torch).
#   - include dirs point at MIRAGE_ROOT (default: the read-only live clone).
#     Candidate builds set MIRAGE_ROOT to a tree containing the OPTIMIZED
#     kernel headers and GATE_EXT_NAME to a distinct module name.
#
# Env knobs:
#   GATE_EXT_NAME  extension/module name   (default: gate_ref_kernels)
#   MIRAGE_ROOT    mirage source tree      (default: /root/.cache/mirage-det)
#   GATE_GENCODE   override -gencode flag  (default: 103a)
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = os.path.dirname(os.path.abspath(__file__))

ext_name = os.environ.get("GATE_EXT_NAME", "gate_ref_kernels")
mirage_root = os.environ.get("MIRAGE_ROOT", "/root/.cache/mirage-det")
gencode = os.environ.get(
    "GATE_GENCODE", "-gencode=arch=compute_103a,code=sm_103a"
)

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")

# The pod's CUDA 13 toolkit does not ship cusparse/cusolver headers (needed
# transitively by torch's ATen headers). compat_include holds symlinks to
# ONLY those headers from the pip nvidia cu13 package -- do NOT add the whole
# pip include dir: its crt/host_runtime.h is from a newer CUDA minor and
# breaks nvcc 13.0 host stubs (__cudaLaunch macro arity clash).
_default_compat = os.path.join(this_dir, "..", "compat_include")
extra_includes = [
    p
    for p in [os.environ.get("GATE_EXTRA_INCLUDE"), _default_compat]
    if p and os.path.isdir(p)
]

macros = [
    ("MIRAGE_BACKEND_USE_CUDA", None),
    ("MIRAGE_FINGERPRINT_USE_CUDA", None),
]

setup(
    name=ext_name,
    ext_modules=[
        CUDAExtension(
            name=ext_name,
            sources=[
                os.path.join(this_dir, "gate_bind.cu"),
                os.path.join(this_dir, "gate_cases_m8.cu"),
                os.path.join(this_dir, "gate_cases_m16.cu"),
            ],
            define_macros=macros,
            include_dirs=[
                os.path.join(mirage_root, "include/mirage/persistent_kernel"),
                os.path.join(
                    mirage_root, "include/mirage/persistent_kernel/tasks"
                ),
                os.path.join(mirage_root, "include"),
                os.path.join(mirage_root, "deps/cutlass/include"),
                os.path.join(mirage_root, "deps/cutlass/tools/util/include"),
            ]
            + extra_includes,
            libraries=["cuda"],
            extra_compile_args={
                "cxx": ["-DMIRAGE_GRACE_BLACKWELL"],
                "nvcc": [
                    "-O3",
                    gencode,
                    "-DMIRAGE_GRACE_BLACKWELL",
                    "-DMPK_ENABLE_TMA",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
