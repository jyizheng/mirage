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
#include "tasks/common/online_softmax.cuh"

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
// All selected-token softmax paths use the same one-pass online reduction
// and fixed tree, so they remain bitwise-identical for the same logits row.
//
// Grid: (1, 1, 1); one task walks all requests. Block: (256, 1, 1).
template <typename T,
          int NUM_REQUESTS,
          int VOCAB_SIZE,
          int MAX_SEQ,
          int PAGE_SIZE,
          int NUM_RID_TASKS = 1>
__device__ __forceinline__ void prefill_prob_capture_task_impl(
    int const my_rid,
    void const *__restrict__ logits_ptr,
    int const *__restrict__ prompt_lengths_ptr,
    long long const *__restrict__ chosen_tokens_ptr,
    void *__restrict__ buffer_ptr,
    int const *__restrict__ request_ids_ptr,
    int const *__restrict__ qo_indptr_buffer_ptr,
    int const *__restrict__ paged_kv_indptr_buffer_ptr,
    int const *__restrict__ paged_kv_last_page_len_buffer_ptr,
    long long const *__restrict__ all_tokens_ptr,
    int const *__restrict__ step_ptr) {
  T const *__restrict__ logits = static_cast<T const *>(logits_ptr);
  float *__restrict__ buffer = static_cast<float *>(buffer_ptr);

  int const tid = threadIdx.x;
  __shared__ float smem_max[8];
  __shared__ float smem_sum[8];

  // Parallel over requests when the layer is launched with grid.x > 1:
  // task i owns rids {i, i+NUM_RID_TASKS, ...} (disjoint buffer rows, so
  // condition (a) holds; per-row reduction order unchanged, so (b) holds).
  for (int rid = my_rid; rid < NUM_REQUESTS; rid += NUM_RID_TASKS) {
    // rid indexes BATCH SLOTS (the qo/paged window arrays). Per-request
    // state (step, prompt_length, tokens, prob buffer) is indexed by
    // BUFFER ROW; the scheduler maintains the slot->row map in
    // request_ids. The two coincide in batch mode but not in online
    // serving, where the row pool is a stack.
    int const first_token_pos = qo_indptr_buffer_ptr[rid];
    int const last_token_pos = qo_indptr_buffer_ptr[rid + 1];
    int const num_tokens = last_token_pos - first_token_pos;
    if (num_tokens <= 0) {
      continue;
    }
    int const req_row =
        request_ids_ptr[rid] < 0 ? rid : request_ids_ptr[rid];
    // seq_len derivation mirrors multitoken_paged_attention_sm100_task_impl
    int const first_page_pos = paged_kv_indptr_buffer_ptr[rid];
    int const last_page_pos = paged_kv_indptr_buffer_ptr[rid + 1];
    int const num_pages = last_page_pos - first_page_pos;
    int const seq_len =
        (num_pages - 1) * PAGE_SIZE + paged_kv_last_page_len_buffer_ptr[rid];
    int const prompt_len = prompt_lengths_ptr[req_row];
    int const step_val = step_ptr[req_row];

    for (int i = 0; i < num_tokens; ++i) {
      int const pos = seq_len - num_tokens + i;
      if (pos + 1 >= MAX_SEQ) {
        continue;
      }
      int const row_idx = first_token_pos + i;
      long long target;
      if (pos + 1 < prompt_len) {
        // teacher-forcing row: the next token is a given prompt token
        target = all_tokens_ptr[(long long)req_row * MAX_SEQ + pos + 1];
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

      if (tid == 0) {
        float logit_at_target = (target_id >= 0 && target_id < VOCAB_SIZE)
                                    ? static_cast<float>(row[target_id])
                                    : -1e30f;
        float prob = __expf(logit_at_target - stats.max) / stats.sum;
        int const slot = step_val + i;
        if (slot >= 0 && slot < MAX_SEQ) {
          buffer[(long long)req_row * MAX_SEQ + slot] = prob;
        }
      }
      __syncthreads();
    }
  }
}

} // namespace kernel
