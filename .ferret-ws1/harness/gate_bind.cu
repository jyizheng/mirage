// Ferret gate harness: torch extension bindings + shape dispatch.
// Deliberately does NOT include the heavy kernel headers: case launchers are
// linked from gate_cases_m8.cu / gate_cases_m16.cu.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// M=8 cases
void gate_linear_m8_n64_k2048(void const *, void const *, void *,
                              cudaStream_t);
void gate_linear_m8_n128_k2048(void const *, void const *, void *,
                               cudaStream_t);
void gate_linear_m8_n256_k2048(void const *, void const *, void *,
                               cudaStream_t);
void gate_splitk_partial_m8_ks256(void const *, void const *, void *,
                                  cudaStream_t);
void gate_splitk_partial_m8_ks768(void const *, void const *, void *,
                                  cudaStream_t);
void gate_splitk_reduce_m8(void const *, void const *, void *, cudaStream_t);
// M=16 cases
void gate_linear_m16_n64_k2048(void const *, void const *, void *,
                               cudaStream_t);
void gate_linear_m16_n128_k2048(void const *, void const *, void *,
                                cudaStream_t);
void gate_linear_m16_n256_k2048(void const *, void const *, void *,
                                cudaStream_t);
void gate_splitk_partial_m16_ks256(void const *, void const *, void *,
                                   cudaStream_t);
void gate_splitk_partial_m16_ks768(void const *, void const *, void *,
                                   cudaStream_t);
void gate_splitk_reduce_m16(void const *, void const *, void *, cudaStream_t);

namespace {

void check_tensor(torch::Tensor const &t, char const *name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.scalar_type() == torch::kBFloat16, name, " must be bf16");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.dim() == 2, name, " must be 2-D");
}

// out[M, N] = x[M, K] @ w[N, K]^T   (bf16 in/out, fp32 accum, NOBIAS)
void linear(torch::Tensor x, torch::Tensor w, torch::Tensor out) {
  check_tensor(x, "x");
  check_tensor(w, "w");
  check_tensor(out, "out");
  int64_t const B = x.size(0);
  int64_t const K = x.size(1);
  int64_t const N = w.size(0);
  TORCH_CHECK(w.size(1) == K, "w K mismatch");
  TORCH_CHECK(out.size(0) == B && out.size(1) == N, "out shape mismatch");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  void const *xp = x.data_ptr();
  void const *wp = w.data_ptr();
  void *op = out.data_ptr();
  if (B == 8 && N == 64 && K == 2048) {
    gate_linear_m8_n64_k2048(xp, wp, op, stream);
  } else if (B == 8 && N == 128 && K == 2048) {
    gate_linear_m8_n128_k2048(xp, wp, op, stream);
  } else if (B == 8 && N == 256 && K == 2048) {
    gate_linear_m8_n256_k2048(xp, wp, op, stream);
  } else if (B == 16 && N == 64 && K == 2048) {
    gate_linear_m16_n64_k2048(xp, wp, op, stream);
  } else if (B == 16 && N == 128 && K == 2048) {
    gate_linear_m16_n128_k2048(xp, wp, op, stream);
  } else if (B == 16 && N == 256 && K == 2048) {
    gate_linear_m16_n256_k2048(xp, wp, op, stream);
  } else {
    TORCH_CHECK(false, "unsupported linear case M=", B, " N=", N, " K=", K);
  }
  C10_CUDA_CHECK(cudaGetLastError());
}

// partials[(s*M):(s+1)*M, :] = x[:, s*Ks:(s+1)*Ks] @ w[:, s*Ks:(s+1)*Ks]^T
// for s in 0..7, each split a plain-TMA-store partial linear task.
void splitk_partial(torch::Tensor x, torch::Tensor w, torch::Tensor partials) {
  check_tensor(x, "x");
  check_tensor(w, "w");
  check_tensor(partials, "partials");
  int64_t const B = x.size(0);
  int64_t const K = x.size(1);
  int64_t const N = w.size(0);
  TORCH_CHECK(w.size(1) == K, "w K mismatch");
  TORCH_CHECK(N == 128, "splitk partial expects N=128, got ", N);
  TORCH_CHECK(partials.size(0) == 8 * B && partials.size(1) == N,
              "partials must be [8*M, 128]");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  void const *xp = x.data_ptr();
  void const *wp = w.data_ptr();
  void *pp = partials.data_ptr();
  if (B == 8 && K == 2048) {
    gate_splitk_partial_m8_ks256(xp, wp, pp, stream);
  } else if (B == 8 && K == 6144) {
    gate_splitk_partial_m8_ks768(xp, wp, pp, stream);
  } else if (B == 16 && K == 2048) {
    gate_splitk_partial_m16_ks256(xp, wp, pp, stream);
  } else if (B == 16 && K == 6144) {
    gate_splitk_partial_m16_ks768(xp, wp, pp, stream);
  } else {
    TORCH_CHECK(false, "unsupported splitk case M=", B, " K=", K);
  }
  C10_CUDA_CHECK(cudaGetLastError());
}

// out[r,c] = bf16(fp32(residual[r,c]) + sum_s fp32(partials[s*M+r, c]))
void splitk_reduce(torch::Tensor partials,
                   torch::Tensor residual,
                   torch::Tensor out) {
  check_tensor(partials, "partials");
  check_tensor(residual, "residual");
  check_tensor(out, "out");
  int64_t const B = out.size(0);
  int64_t const N = out.size(1);
  TORCH_CHECK(N == 128, "splitk_reduce expects N=128, got ", N);
  TORCH_CHECK(partials.size(0) == 8 * B && partials.size(1) == N,
              "partials must be [8*M, 128]");
  TORCH_CHECK(residual.size(0) == B && residual.size(1) == N,
              "residual must be [M, 128]");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  void const *pp = partials.data_ptr();
  void const *rp = residual.data_ptr();
  void *op = out.data_ptr();
  if (B == 8) {
    gate_splitk_reduce_m8(pp, rp, op, stream);
  } else if (B == 16) {
    gate_splitk_reduce_m16(pp, rp, op, stream);
  } else {
    TORCH_CHECK(false, "unsupported splitk_reduce case M=", B);
  }
  C10_CUDA_CHECK(cudaGetLastError());
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear", &linear,
        "linear_sm100_mpk task: out[M,N] = x[M,K] @ w[N,K]^T (bf16, fp32 acc)");
  m.def("splitk_partial", &splitk_partial,
        "8x deterministic split-K partial linear tasks -> partials[8*M,128]");
  m.def("splitk_reduce", &splitk_reduce,
        "fixed-order split-K combine with residual");
}
