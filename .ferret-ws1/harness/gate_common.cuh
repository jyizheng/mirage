// Ferret gate harness: common launchers for the frozen correctness+perf gate.
//
// This header wraps the UNMODIFIED in-tree kernels
//   kernel::linear_sm100_mpk_task_impl   (tasks/blackwell/linear_sm100_mpk.cuh)
//   kernel::splitk_reduce_task_impl      (tasks/blackwell/splitk_reduce_sm100.cuh)
// exactly the way the production task registration does
// (src/kernel/task_register.cc register_linear_sm100_task /
//  register_splitk_linear_sm100_task partial_out=true /
//  register_splitk_reduce_sm100_task), including:
//   - MMA_M=128, MMA_N=16, TILE_SIZE=64, OUTPUT_ATOM_SIZE=128
//   - 256-thread CTA, 224KB dynamic smem, cluster (1,1,1)
//   - TMA input/output smem boxes clamped to min(MMA_N, BATCH_SIZE)
//     (see tma.cuh fill_tma_desc_by_task, TASK_LINEAR_SM100 case)
//   - split-K partial variant: NOBIAS=true, SplitK=false (plain TMA store),
//     per-split REDUCTION_SIZE with full-tensor GMEM_STRIDE_ROW
//     (strided K-slice views), disjoint partials slices.
//
// The ONLY intended difference vs tests/runtime_python/blackwell/sm100_linear
// is host-side plumbing: TMA descriptors are built once per (ptr,ptr,ptr)
// tuple and cached (the original harness cudaMalloc'd descriptors on every
// call), and kernels launch on the caller-provided stream without a device
// sync, so CUDA-event timing measures kernel time. Reference and candidate
// are always built from this same file, so the comparison is like-for-like.
#pragma once

// cute/tensor.hpp must come FIRST: the kernel header's own include order
// (cooperative_copy.hpp before tensor.hpp) trips a copy_atom/prefetch
// include cycle under this cutlass+nvcc combo. Production TUs include the
// full task_header.cuh whose earlier includes establish cute/tensor.hpp
// first; we replicate that here, then include ONLY the two kernels under
// test (task_header.cuh drags in non-inline definitions that break
// multi-TU linking).
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include <cute/tensor.hpp>

#include "blackwell/linear_sm100_mpk.cuh"
#include "blackwell/splitk_reduce_sm100.cuh"
#include "hopper/tma_2d.cuh"
#include "runtime_header.h"
#include "tma.cuh"
#include <cuda_runtime.h>

#include <cstdio>
#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>

using bfloat16 = cute::bfloat16_t;

namespace gate {

struct DescTriple {
  CUtensorMap *input;
  CUtensorMap *weight;
  CUtensorMap *output;
};

inline void gate_cuda_check(cudaError_t e, char const *what) {
  if (e != cudaSuccess) {
    fprintf(stderr, "gate: %s failed: %s\n", what, cudaGetErrorString(e));
    throw std::runtime_error(std::string("gate: ") + what + " failed: " +
                             cudaGetErrorString(e));
  }
}

// ---------------------------------------------------------------------------
// Linear (and split-K partial) kernel wrapper: identical instantiation shape
// to the production codegen in register_linear_sm100_task /
// register_splitk_linear_sm100_task(partial_out=true).
// ---------------------------------------------------------------------------
template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int REDUCTION_STRIDE,
          class BiasTensor,
          int MMA_M,
          int MMA_N,
          bool NoBias,
          int NUM_AB_STAGE = 8,
          int NUM_ACC_STAGE = 2,
          int NUM_C_STAGE = 4>
__global__ __launch_bounds__(256, 1) void gate_linear_sm100_wrapper(
    void *tma_a_desc_ptr,
    void *tma_b_desc_ptr,
    BiasTensor mBias,
    void *tma_out_desc_ptr) {

  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;

  constexpr int TMA_CP_ASYNC_SIZE = 64;
  constexpr int TILE_SIZE = 64;
  constexpr int TMA_CP_ASYNC_REPEAT_COL =
      (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;
  constexpr int OUTPUT_ATOM_SIZE = 128;
  constexpr int OUTPUT_TMA_CP_SIZE = 128;
  constexpr int OUTPUT_ATOM_REPEAT_COL =
      (OUTPUT_ATOM_SIZE + OUTPUT_TMA_CP_SIZE - 1) / OUTPUT_TMA_CP_SIZE;

  using TMA_B =
      kernel::tma::tma_2d<bfloat16,
                          B,
                          M,
                          S,
                          BATCH_SIZE,                /*GMEM_ROW_*/
                          REDUCTION_SIZE,            /*GMEM_COL_*/
                          MMA_N,                     /*SMEM_ROW_*/
                          TMA_CP_ASYNC_SIZE,         /*SMEM_COL_*/
                          REDUCTION_STRIDE,          /*GMEM_STRIDE_ROW_*/
                          1,                         /*GMEM_STRIDE_COL_*/
                          1,                         /*SMEM_REPEAT_ROW_*/
                          TMA_CP_ASYNC_REPEAT_COL,   /*SMEM_REPEAT_COL_*/
                          MMA_N * TMA_CP_ASYNC_SIZE, /*SMEM_STRIDE_*/
                          true>;
  using TMA_A =
      kernel::tma::tma_2d<bfloat16,
                          B,
                          M,
                          S,
                          OUTPUT_SIZE,               /*GMEM_ROW_*/
                          REDUCTION_SIZE,            /*GMEM_COL_*/
                          MMA_M,                     /*SMEM_ROW_*/
                          TMA_CP_ASYNC_SIZE,         /*SMEM_COL_*/
                          REDUCTION_STRIDE,          /*GMEM_STRIDE_ROW_*/
                          1,                         /*GMEM_STRIDE_COL_*/
                          1,                         /*SMEM_REPEAT_ROW_*/
                          TMA_CP_ASYNC_REPEAT_COL,   /*SMEM_REPEAT_COL_*/
                          MMA_M * TMA_CP_ASYNC_SIZE, /*SMEM_STRIDE_*/
                          true>;
  using TMA_OUT =
      kernel::tma::tma_2d<bfloat16,
                          0,
                          M,
                          S,
                          BATCH_SIZE,             /*GMEM_ROW_*/
                          OUTPUT_SIZE,            /*GMEM_COL_*/
                          MMA_N,                  /*SMEM_ROW_*/
                          MMA_M,                  /*SMEM_COL_*/
                          OUTPUT_SIZE,            /*GMEM_STRIDE_ROW_*/
                          1,                      /*GMEM_STRIDE_COL_*/
                          1,                      /*SMEM_REPEAT_ROW_*/
                          OUTPUT_ATOM_REPEAT_COL, /*SMEM_REPEAT_COL_*/
                          MMA_N * MMA_M,          /*SMEM_STRIDE_*/
                          true>;

  TMA_A tma_a(static_cast<CUtensorMap *>(tma_a_desc_ptr));
  TMA_B tma_b(static_cast<CUtensorMap *>(tma_b_desc_ptr));
  TMA_OUT tma_out(static_cast<CUtensorMap *>(tma_out_desc_ptr));

  kernel::linear_sm100_mpk_task_impl<T,
                                     TMA_A,
                                     TMA_B,
                                     BiasTensor,
                                     TMA_OUT,
                                     MMA_M,
                                     MMA_N,
                                     BATCH_SIZE,
                                     OUTPUT_SIZE,
                                     REDUCTION_SIZE,
                                     NoBias,
                                     /*SplitK=*/false,
                                     NUM_AB_STAGE,
                                     NUM_ACC_STAGE,
                                     NUM_C_STAGE>(tma_a, tma_b, mBias, tma_out);
}

// Host-side descriptor construction, mirroring
// fill_tma_desc_by_task(TASK_LINEAR_SM100 / TASK_SPLITK_PARTIAL_LINEAR_SM100)
// in include/mirage/persistent_kernel/tma.cuh (incl. min(MMA_N, batch) box
// clamping on input and output).
template <int BATCH, int OUT, int RED, int RED_STRIDE>
inline DescTriple build_linear_descs(void const *input,
                                     void const *weight,
                                     void *output) {
  constexpr int MMA_M = 128;
  constexpr int MMA_N = 16;
  constexpr int TMA_CP_ASYNC_SIZE = 64;
  constexpr int TILE_SIZE = 64;
  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;
  constexpr uint32_t kClampedBN = (BATCH < MMA_N) ? BATCH : MMA_N;

  CUtensorMap host_i, host_w, host_o;

  uint64_t i_gmem_shape[2] = {static_cast<uint64_t>(BATCH),
                              static_cast<uint64_t>(RED)};
  uint64_t i_gmem_stride[2] = {1, static_cast<uint64_t>(RED_STRIDE)};
  uint32_t i_smem_shape[2] = {kClampedBN,
                              static_cast<uint32_t>(TMA_CP_ASYNC_SIZE)};
  size_t const smem_repeat_col =
      (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;
  mirage::runtime::fill_tma_desc<bfloat16, B, M, S, 2>(
      &host_i,
      const_cast<void *>(input),
      i_gmem_shape,
      i_gmem_stride,
      i_smem_shape,
      1,
      smem_repeat_col);

  uint64_t w_gmem_shape[2] = {static_cast<uint64_t>(OUT),
                              static_cast<uint64_t>(RED)};
  uint64_t w_gmem_stride[2] = {1, static_cast<uint64_t>(RED_STRIDE)};
  uint32_t w_smem_shape[2] = {static_cast<uint32_t>(MMA_M),
                              static_cast<uint32_t>(TMA_CP_ASYNC_SIZE)};
  mirage::runtime::fill_tma_desc<bfloat16, B, M, S, 2>(
      &host_w,
      const_cast<void *>(weight),
      w_gmem_shape,
      w_gmem_stride,
      w_smem_shape,
      1,
      smem_repeat_col);

  uint64_t o_gmem_shape[2] = {static_cast<uint64_t>(BATCH),
                              static_cast<uint64_t>(OUT)};
  uint64_t o_gmem_stride[2] = {1, static_cast<uint64_t>(OUT)};
  uint32_t o_smem_shape[2] = {kClampedBN, static_cast<uint32_t>(MMA_M)};
  mirage::runtime::fill_tma_desc<bfloat16, 0, M, S, 2>(
      &host_o, output, o_gmem_shape, o_gmem_stride, o_smem_shape, 1, 1);

  DescTriple d;
  gate_cuda_check(cudaMalloc(&d.input, sizeof(CUtensorMap)), "cudaMalloc");
  gate_cuda_check(cudaMalloc(&d.weight, sizeof(CUtensorMap)), "cudaMalloc");
  gate_cuda_check(cudaMalloc(&d.output, sizeof(CUtensorMap)), "cudaMalloc");
  gate_cuda_check(cudaMemcpy(d.input,
                             &host_i,
                             sizeof(CUtensorMap),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy");
  gate_cuda_check(cudaMemcpy(d.weight,
                             &host_w,
                             sizeof(CUtensorMap),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy");
  gate_cuda_check(cudaMemcpy(d.output,
                             &host_o,
                             sizeof(CUtensorMap),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy");
  return d;
}

// One kernel launch of the linear / split-K-partial task on `stream`.
// Descriptors are cached per (input, weight, output) pointer tuple.
template <int BATCH, int OUT, int RED, int RED_STRIDE>
void gate_launch_linear(void const *input,
                        void const *weight,
                        void *output,
                        cudaStream_t stream) {
  constexpr int MMA_M = 128;
  constexpr int MMA_N = 16;

  static std::mutex mu;
  static std::map<std::tuple<void const *, void const *, void *>, DescTriple>
      cache;
  DescTriple d;
  {
    std::lock_guard<std::mutex> g(mu);
    auto key = std::make_tuple(input, weight, output);
    auto it = cache.find(key);
    if (it == cache.end()) {
      d = build_linear_descs<BATCH, OUT, RED, RED_STRIDE>(
          input, weight, output);
      cache.emplace(key, d);
    } else {
      d = it->second;
    }
  }

  // Bias tensor: unused (NoBias=true), same shape convention as the harness.
  cute::Layout layout_Bias =
      cute::make_layout(cute::make_shape(BATCH, OUT),
                        cute::make_stride(OUT, cute::Int<1>{}));
  cute::Tensor mBias = cute::make_tensor(
      cute::make_gmem_ptr(static_cast<bfloat16 *>(nullptr)), layout_Bias);

  auto *kernel_ptr = &gate_linear_sm100_wrapper<bfloat16,
                                                BATCH,
                                                OUT,
                                                RED,
                                                RED_STRIDE,
                                                decltype(mBias),
                                                MMA_M,
                                                MMA_N,
                                                /*NoBias=*/true>;
  int constexpr smemBytes = 224 * 1024;
  static std::once_flag once;
  std::call_once(once, [&] {
    gate_cuda_check(
        cudaFuncSetAttribute(kernel_ptr,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smemBytes),
        "cudaFuncSetAttribute");
  });

  cutlass::ClusterLaunchParams params = {
      dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1), smemBytes, stream};
  cutlass::Status status =
      cutlass::launch_kernel_on_cluster(params,
                                        (void const *)kernel_ptr,
                                        (void *)d.weight,
                                        (void *)d.input,
                                        mBias,
                                        (void *)d.output);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("gate: cluster kernel launch failed");
  }
}

// Deterministic split-K partial pass: NUM_SPLITS sequential launches of the
// partial linear task. Split s reads act[:, s*KSPLIT:(s+1)*KSPLIT] and
// weight[:, s*KSPLIT:(s+1)*KSPLIT] as strided views of the full-K tensors
// (GMEM_STRIDE_ROW = KFULL, exactly like production task tensors) and TMA-
// stores into the disjoint partials slice partials[s*BATCH:(s+1)*BATCH, :].
template <int BATCH, int KSPLIT, int KFULL>
void gate_launch_splitk_partial(void const *act,
                                void const *weight,
                                void *partials,
                                cudaStream_t stream) {
  static_assert(KFULL % KSPLIT == 0, "KFULL must be divisible by KSPLIT");
  constexpr int OUT = 128;
  constexpr int NUM_SPLITS = KFULL / KSPLIT;
  static_assert(NUM_SPLITS == 8, "gate expects NUM_SPLITS == 8");
  bfloat16 const *a = static_cast<bfloat16 const *>(act);
  bfloat16 const *w = static_cast<bfloat16 const *>(weight);
  bfloat16 *p = static_cast<bfloat16 *>(partials);
  for (int s = 0; s < NUM_SPLITS; ++s) {
    gate_launch_linear<BATCH, OUT, KSPLIT, KFULL>(
        a + static_cast<size_t>(s) * KSPLIT,
        w + static_cast<size_t>(s) * KSPLIT,
        p + static_cast<size_t>(s) * BATCH * OUT,
        stream);
  }
}

// ---------------------------------------------------------------------------
// Split-K reduce wrapper: identical instantiation shape to
// register_splitk_reduce_sm100_task codegen; 256-thread worker CTA.
// ---------------------------------------------------------------------------
template <typename T, int NUM_SPLITS, int BATCH, int OUT>
__global__ __launch_bounds__(256, 1) void gate_splitk_reduce_wrapper(
    void const *partials, void const *residual, void *output) {
  kernel::splitk_reduce_task_impl<T,
                                  NUM_SPLITS,
                                  BATCH,
                                  OUT,
                                  /*PARTIAL_STRIDE=*/OUT,
                                  /*STRIDE=*/OUT,
                                  /*WITH_RESIDUAL=*/true>(
      partials, residual, output);
}

template <int BATCH>
void gate_launch_splitk_reduce(void const *partials,
                               void const *residual,
                               void *output,
                               cudaStream_t stream) {
  gate_splitk_reduce_wrapper<bfloat16, 8, BATCH, 128>
      <<<1, 256, 0, stream>>>(partials, residual, output);
}

} // namespace gate
