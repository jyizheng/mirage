/* Copyright 2026 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

namespace kernel {

// Deterministic combine for split-K linear partials:
//   output[b, n] = residual[b, n] + sum_{s=0..NUM_SPLITS-1} partials[s*B+b, n]
//
// The partials buffer is written by TASK_SPLITK_PARTIAL_LINEAR_SM100 tasks
// with plain TMA stores (one disjoint [BATCH_SIZE, n-tile] slice per K-split,
// no atomics), and this task accumulates the splits in a fixed order. Every
// output element therefore has a bit-reproducible reduction order that does
// not depend on task scheduling or on how many requests share the batch —
// unlike the tma_reduce_add path of TASK_SPLITK_LINEAR_SM100, whose partials
// combine in task-completion order.
// WITH_RESIDUAL mirrors the enable_residual convention of
// linear_with_residual: under tensor parallelism only rank 0 adds the
// residual, since the outputs are subsequently allreduced across ranks.
template <typename T,
          int NUM_SPLITS,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int PARTIAL_STRIDE,
          int STRIDE,
          bool WITH_RESIDUAL = true>
__device__ __forceinline__ void splitk_reduce_task_impl(
    void const *partials_ptr, void const *residual_ptr, void *output_ptr) {
  T const *__restrict__ partials = static_cast<T const *>(partials_ptr);
  T const *__restrict__ residual = static_cast<T const *>(residual_ptr);
  T *__restrict__ out = static_cast<T *>(output_ptr);

#pragma unroll 4
  for (int i = threadIdx.x; i < BATCH_SIZE * OUTPUT_SIZE; i += blockDim.x) {
    int row = i / OUTPUT_SIZE;
    int col = i % OUTPUT_SIZE;
    float acc = WITH_RESIDUAL ? float(residual[row * STRIDE + col]) : 0.0f;
#pragma unroll
    for (int s = 0; s < NUM_SPLITS; s++) {
      acc += float(partials[(s * BATCH_SIZE + row) * PARTIAL_STRIDE + col]);
    }
    out[row * STRIDE + col] = T(acc);
  }
}

} // namespace kernel
