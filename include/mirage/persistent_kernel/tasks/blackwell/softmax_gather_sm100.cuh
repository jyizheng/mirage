/* Copyright 2025 CMU
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
#include "tasks/common/common_header.cuh"
#include "tasks/common/online_softmax.cuh"

// Fused softmax + gather: given logits [BATCH_SIZE, VOCAB_SIZE] and
// token_ids [BATCH_SIZE], output prob[batch] =
// softmax(logits[batch])[token_id].
//
// Does NOT materialize the full probability distribution — computes:
//   max_val = max(logits[batch, :])
//   log_sum = log(sum(exp(logits[batch, :] - max_val)))
//   prob = exp(logits[batch, token_id] - max_val - log_sum)
//
// Uses warp-cooperative parallel reduction over the vocab dimension.
// Grid: (BATCH_SIZE, 1, 1), Block: (256, 1, 1).
// Each block handles one batch element.

namespace kernel {

template <typename T, int BATCH_SIZE, int VOCAB_SIZE>
__device__ __forceinline__ void
    softmax_gather_task_impl(void const *__restrict__ logits_ptr,
                             void const *__restrict__ token_ids_ptr,
                             void *__restrict__ output_probs_ptr) {
  T const *__restrict__ logits = static_cast<T const *>(logits_ptr);
  long long const *__restrict__ token_ids =
      static_cast<long long const *>(token_ids_ptr);
  float *__restrict__ output_probs = static_cast<float *>(output_probs_ptr);

  int const tid = threadIdx.x;
  __shared__ float smem_max[8];
  __shared__ float smem_sum[8];

  for (int batch_idx = 0; batch_idx < BATCH_SIZE; ++batch_idx) {
    T const *row = logits + batch_idx * VOCAB_SIZE;
    int const target_id = static_cast<int>(token_ids[batch_idx]);

    OnlineSoftmaxStats stats = {-1e30f, 0.0f};
    constexpr int VEC_SIZE = 4;
    constexpr int CHUNK_SIZE = 256 * VEC_SIZE;
    for (int chunk = 0; chunk < (VOCAB_SIZE + CHUNK_SIZE - 1) / CHUNK_SIZE;
         ++chunk) {
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        int const vocab_idx = (chunk * 256 + tid) * VEC_SIZE + j;
        if (vocab_idx < VOCAB_SIZE) {
          stats = online_softmax_add(
              stats, static_cast<float>(row[vocab_idx]));
        }
      }
    }
    stats = online_softmax_reduce_256(stats, smem_max, smem_sum);

    // Phase 3: Compute probability at target token
    if (tid == 0) {
      float logit_at_target = (target_id >= 0 && target_id < VOCAB_SIZE)
                                  ? static_cast<float>(row[target_id])
                                  : -1e30f;
      float prob = __expf(logit_at_target - stats.max) / stats.sum;
      output_probs[batch_idx] = prob;
    }
    __syncthreads();
  }
}

} // namespace kernel
