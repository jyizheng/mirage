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
#include "tasks/common/common_header.cuh"

namespace kernel {

// Per-token probability capture over ALL rows (teacher-forcing prefill
// rows AND the generating row of each request's window).
//
// For every valid logits row i of every request r, computes
//   P(tokens[r, pos+1] | prefix through pos),  pos = seq_len_r - n_r + i
// and stores it at buffer[r, step_r + i] — the same slot convention the
// decode-side capture (softmax_gather + prob_scatter) uses, so a rescore
// pass over a trajectory fills the same buffer positions a decode pass
// would have. Rows are captured only while the NEXT position is still
// inside the prompt region (teacher forcing): in a rescore run the whole
// trajectory is prompt, so every position is captured; in a rollout run
// this task never overlaps the decode capture's slots.
//
// The per-row softmax reduction below is copied VERBATIM from
// softmax_gather_sm100.cuh so the two capture paths are bitwise-identical
// for the same logits row.
//
// Grid: (1, 1, 1); one task walks all requests. Block: (256, 1, 1).
template <typename T,
          int NUM_REQUESTS,
          int VOCAB_SIZE,
          int MAX_SEQ,
          int PAGE_SIZE>
__device__ __forceinline__ void prefill_prob_capture_task_impl(
    void const *__restrict__ logits_ptr,
    int const *__restrict__ prompt_lengths_ptr,
    long long const *__restrict__ chosen_tokens_ptr,
    void *__restrict__ buffer_ptr,
    int const *__restrict__ qo_indptr_buffer_ptr,
    int const *__restrict__ paged_kv_indptr_buffer_ptr,
    int const *__restrict__ paged_kv_last_page_len_buffer_ptr,
    long long const *__restrict__ all_tokens_ptr,
    int const *__restrict__ step_ptr) {
  T const *__restrict__ logits = static_cast<T const *>(logits_ptr);
  float *__restrict__ buffer = static_cast<float *>(buffer_ptr);

  int const tid = threadIdx.x;
  int const num_threads = blockDim.x;

  for (int rid = 0; rid < NUM_REQUESTS; ++rid) {
    int const first_token_pos = qo_indptr_buffer_ptr[rid];
    int const last_token_pos = qo_indptr_buffer_ptr[rid + 1];
    int const num_tokens = last_token_pos - first_token_pos;
    if (num_tokens <= 0) {
      continue;
    }
    // seq_len derivation mirrors multitoken_paged_attention_sm100_task_impl
    int const first_page_pos = paged_kv_indptr_buffer_ptr[rid];
    int const last_page_pos = paged_kv_indptr_buffer_ptr[rid + 1];
    int const num_pages = last_page_pos - first_page_pos;
    int const seq_len =
        (num_pages - 1) * PAGE_SIZE + paged_kv_last_page_len_buffer_ptr[rid];
    int const prompt_len = prompt_lengths_ptr[rid];
    int const step_val = step_ptr[rid];

    for (int i = 0; i < num_tokens; ++i) {
      int const pos = seq_len - num_tokens + i;
      if (pos + 1 >= MAX_SEQ) {
        continue;
      }
      int const row_idx = first_token_pos + i;
      long long target;
      if (pos + 1 < prompt_len) {
        // teacher-forcing row: the next token is a given prompt token
        target = all_tokens_ptr[(long long)rid * MAX_SEQ + pos + 1];
      } else if (i == num_tokens - 1) {
        // generating row: capture P(token chosen this iteration). Reading
        // chosen_tokens through a graph input (not runtime_config) makes
        // the dependency on the sampling/argmax task explicit. Row->request
        // mapping goes through qo_indptr, so this stays correct when
        // requests finish at different times and the window re-packs --
        // unlike the row-indexed softmax_gather+prob_scatter pair this
        // replaces.
        target = chosen_tokens_ptr[row_idx];
      } else {
        continue;
      }
      int const target_id = static_cast<int>(target);
      T const *row = logits + (long long)row_idx * VOCAB_SIZE;

      // ---- softmax + gather, verbatim from softmax_gather_sm100.cuh ----
      float local_max = -1e30f;
      for (int v = tid; v < VOCAB_SIZE; v += num_threads) {
        float val = static_cast<float>(row[v]);
        local_max = fmaxf(local_max, val);
      }
      for (int mask = 16; mask > 0; mask >>= 1) {
        local_max =
            fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, mask));
      }
      __shared__ float smem_max[8]; // max 8 warps
      int warp_id = tid / 32;
      int lane_id = tid % 32;
      if (lane_id == 0) {
        smem_max[warp_id] = local_max;
      }
      __syncthreads();
      if (tid < 8) {
        float wmax = smem_max[tid];
        for (int mask = 4; mask > 0; mask >>= 1) {
          wmax = fmaxf(wmax, __shfl_xor_sync(0xff, wmax, mask));
        }
        smem_max[0] = wmax;
      }
      __syncthreads();
      float const global_max = smem_max[0];

      float local_sum = 0.0f;
      for (int v = tid; v < VOCAB_SIZE; v += num_threads) {
        local_sum += __expf(static_cast<float>(row[v]) - global_max);
      }
      for (int mask = 16; mask > 0; mask >>= 1) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, mask);
      }
      __shared__ float smem_sum[8];
      if (lane_id == 0) {
        smem_sum[warp_id] = local_sum;
      }
      __syncthreads();
      if (tid < 8) {
        float wsum = smem_sum[tid];
        for (int mask = 4; mask > 0; mask >>= 1) {
          wsum += __shfl_xor_sync(0xff, wsum, mask);
        }
        smem_sum[0] = wsum;
      }
      __syncthreads();
      float const global_sum = smem_sum[0];

      if (tid == 0) {
        float logit_at_target = (target_id >= 0 && target_id < VOCAB_SIZE)
                                    ? static_cast<float>(row[target_id])
                                    : -1e30f;
        float prob = __expf(logit_at_target - global_max) / global_sum;
        int const slot = step_val + i;
        if (slot >= 0 && slot < MAX_SEQ) {
          buffer[(long long)rid * MAX_SEQ + slot] = prob;
        }
      }
      __syncthreads();
    }
  }
}

} // namespace kernel
