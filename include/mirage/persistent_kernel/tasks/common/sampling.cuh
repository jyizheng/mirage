/* Copyright (c) 2025 by CMU.
 * Copyright (c) 2025 by FlashInfer team.
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

/*
 * Sampling from logits using Gumbel-Max trick
 * Based on FlashInfer's sampling kernel (Apache License 2.0).
 */

#pragma once

#include "tasks/common/online_softmax.cuh"

#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda/std/limits>
#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>

namespace kernel {

using namespace cub;

// Helper function for ceiling division
template <typename T>
__host__ __device__ __forceinline__ T sampling_ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

// True -infinity for the sampling DType.
//
// cuda::std::numeric_limits is NOT specialized for mirage's
// type::bfloat16_t; the primary template's infinity() returns T() == +0.0,
// so guards spelled "-cuda::std::numeric_limits<DType>::infinity()"
// silently evaluated to **-0.0** for bf16. A -0.0 "guard" outranks every
// negative candidate, so vocab shards whose logits+noise are all negative
// stored (-0.0, out-of-range index) as their partial winner (observed as
// val_bits=0x8000 idx=2047 partials on all-negative shards every step).
template <typename T>
__device__ __forceinline__ T sampling_neg_inf() {
  static_assert(cuda::std::numeric_limits<T>::is_specialized,
                "add an explicit sampling_neg_inf specialization");
  return -cuda::std::numeric_limits<T>::infinity();
}
template <>
__device__ __forceinline__ type::bfloat16_t
    sampling_neg_inf<type::bfloat16_t>() {
  // bf16 -inf (0xFF80); constructed from float -inf via cvt.rn.bf16.f32.
  return type::bfloat16_t(-cuda::std::numeric_limits<float>::infinity());
}

/******************* vec_t - Simplified Vector Type *******************/

template <typename T, size_t vec_size>
struct sampling_vec_t {
  T data[vec_size];

  __device__ __forceinline__ T &operator[](size_t i) {
    return data[i];
  }
  __device__ __forceinline__ T const &operator[](size_t i) const {
    return data[i];
  }

  __device__ __forceinline__ void fill(T val) {
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      data[i] = val;
    }
  }

  __device__ __forceinline__ void cast_load(T const *ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      data[i] = ptr[i];
    }
  }
};

/******************* DataAndIndex Structure *******************/

// Map a bf16 value to a uint16 whose unsigned order matches the float
// order (larger float => larger key), so "value >= tau" becomes an
// integer key comparison usable in a deterministic bisection. Standard
// radix-float transform: flip all bits for negatives, set the sign bit
// for non-negatives. NaNs are not expected in logits.
__device__ __forceinline__ uint16_t sampling_bf16_orderkey(type::bfloat16_t v) {
  uint16_t const u = v.storage;
  return (u & 0x8000u) ? static_cast<uint16_t>(~u)
                       : static_cast<uint16_t>(u | 0x8000u);
}

template <typename DType, typename IdType>
struct SamplingDataAndIndex {
  DType data;
  IdType index;

  __device__ SamplingDataAndIndex
      operator+(SamplingDataAndIndex const &other) const {
    if (data > other.data) {
      return {data, index};
    } else {
      return {other.data, other.index};
    }
  }

  __device__ SamplingDataAndIndex &
      operator+=(SamplingDataAndIndex const &other) {
    if (data > other.data) {
      return *this;
    } else {
      data = other.data;
      index = other.index;
      return *this;
    }
  }
};

/******************* Gumbel Noise Generation *******************/

template <typename DType, uint32_t VEC_SIZE>
__device__ __forceinline__ sampling_vec_t<DType, VEC_SIZE>
    GenerateSamplingGumbelNoise(uint64_t philox_seed,
                                uint64_t philox_offset,
                                uint64_t subsequence) {
  curandStatePhilox4_32_10_t state;
  sampling_vec_t<float, VEC_SIZE> noise;
  constexpr float kEPSILON = 1e-20f;
  constexpr float kLOG2 = 0.6931471806f;

  auto uniform2gumbel = [](float x) {
    // curand_uniform yields values in (0, 1]: the top 128 u32 draws map to
    // exactly 1.0f (probability 2^-25 per draw). gumbel(1.0f) degenerates to
    // -kLOG2 * log2f(kEPSILON) = +46.05 -- a deterministic noise spike that
    // force-samples whatever token it lands on, ~3x the largest legitimate
    // Gumbel value (+16.3 at u = 1 - 2^-24). Clamp to the largest float
    // strictly below 1.0f (0x1.fffffep-1f = 1 - 2^-24), merging the 1.0f
    // bucket into the adjacent top bucket the generator already produces.
    // Pure per-draw fminf: no RNG state or draw-order change, so seeded
    // runs remain bitwise deterministic (but trajectories differ from
    // pre-fix builds; references regenerated in references_v2).
    x = fminf(x, 0x1.fffffep-1f);
    return -kLOG2 * log2f(-log2f(x + kEPSILON) + kEPSILON);
  };

#pragma unroll
  for (uint32_t i = 0; i + 4 <= VEC_SIZE; i += 4) {
    curand_init(philox_seed, subsequence + i, philox_offset, &state);
    float4 noise_vec = curand_uniform4(&state);
    noise[i] = uniform2gumbel(noise_vec.x);
    noise[i + 1] = uniform2gumbel(noise_vec.y);
    noise[i + 2] = uniform2gumbel(noise_vec.z);
    noise[i + 3] = uniform2gumbel(noise_vec.w);
  }

  if constexpr (VEC_SIZE % 4 != 0) {
    curand_init(
        philox_seed, subsequence + VEC_SIZE / 4 * 4, philox_offset, &state);
    float4 noise_vec = curand_uniform4(&state);
    if constexpr (VEC_SIZE % 4 == 1) {
      noise[VEC_SIZE - 1] = uniform2gumbel(noise_vec.x);
    } else if constexpr (VEC_SIZE % 4 == 2) {
      noise[VEC_SIZE - 2] = uniform2gumbel(noise_vec.x);
      noise[VEC_SIZE - 1] = uniform2gumbel(noise_vec.y);
    } else if constexpr (VEC_SIZE % 4 == 3) {
      noise[VEC_SIZE - 3] = uniform2gumbel(noise_vec.x);
      noise[VEC_SIZE - 2] = uniform2gumbel(noise_vec.y);
      noise[VEC_SIZE - 1] = uniform2gumbel(noise_vec.z);
    }
  }

  if constexpr (std::is_same_v<DType, float>) {
    return noise;
  } else {
    sampling_vec_t<DType, VEC_SIZE> ret;
#pragma unroll
    for (uint32_t i = 0; i < VEC_SIZE; ++i) {
      ret[i] = static_cast<DType>(noise[i]);
    }
    return ret;
  }
}

/******************* Sampling From Logits Kernel *******************/

constexpr BlockScanAlgorithm SAMPLING_SCAN_ALGO = BLOCK_SCAN_WARP_SCANS;
constexpr BlockReduceAlgorithm SAMPLING_REDUCE_ALGO =
    BLOCK_REDUCE_WARP_REDUCTIONS;

// First stage of parallel position-keyed Gumbel-Max. Each task owns one
// contiguous vocabulary shard (the graph dmap offsets logits/partials to that
// shard) and visits every active row. The existing fixed-layout argmax-reduce
// task combines the shard winners in increasing shard order.
template <uint32_t BLOCK_THREADS,
          uint32_t VEC_SIZE,
          typename DType,
          typename IdType,
          int BATCH_SIZE,
          int CHUNK_SIZE,
          int NUM_PARTIAL_TASKS>
__device__ __forceinline__ void sampling_partial_poskeyed_kernel(
    int const part_idx,
    DType const *logits,
    DType *partial_values,
    IdType *partial_indices,
    uint32_t vocab_size,
    uint64_t philox_seed,
    int const *request_ids_ptr,
    int const *qo_indptr_buffer_ptr,
    int const *paged_kv_indptr_buffer_ptr,
    int const *paged_kv_last_page_len_ptr,
    int num_requests,
    int page_size,
    int max_seq,
    int num_active_tokens,
    float inv_temperature = 1.0f,
    float frequency_penalty = 0.0f,
    float presence_penalty = 0.0f,
    float repetition_penalty = 1.0f,
    long long const *all_tokens_ptr = nullptr,
    int const *prompt_lengths_ptr = nullptr,
    int const *sampling_params_ptr = nullptr) {
  uint32_t const tx = threadIdx.x;
  using SharedMem = typename BlockReduce<SamplingDataAndIndex<DType, IdType>,
                                         BLOCK_THREADS,
                                         SAMPLING_REDUCE_ALGO>::TempStorage;
  extern __shared__ __align__(alignof(SharedMem)) uint8_t smem_sampling_logit[];
  auto &temp_storage = reinterpret_cast<SharedMem &>(smem_sampling_logit);
  // Per-row sampling-record staging: thread 0 reads the 9 meaningful lanes
  // once and broadcasts them through shared memory, so every thread of the
  // block sees ONE consistent view of the record — block uniformity of the
  // override branch by construction, and a single load per row instead of
  // one per thread.  (Per-thread reads of the record were observed to
  // produce rare single-token nondeterminism on GB300; see the per-request
  // sampling design notes.)
  __shared__ int sp_shared[9];

  for (int row = 0; row < num_active_tokens; ++row) {
    int rid = 0;
    while (rid < num_requests && row >= qo_indptr_buffer_ptr[rid + 1]) {
      ++rid;
    }
    if (rid >= num_requests || row < qo_indptr_buffer_ptr[rid]) {
      continue;
    }

    int const first_token_pos = qo_indptr_buffer_ptr[rid];
    int const last_token_pos = qo_indptr_buffer_ptr[rid + 1];
    int const num_tokens = last_token_pos - first_token_pos;
    int const num_pages =
        paged_kv_indptr_buffer_ptr[rid + 1] - paged_kv_indptr_buffer_ptr[rid];
    int const seq_len =
        (num_pages - 1) * page_size + paged_kv_last_page_len_ptr[rid];
    int const pos = seq_len - num_tokens + (row - first_token_pos);
    int const noise_rid =
        request_ids_ptr[rid] < 0 ? rid : request_ids_ptr[rid];
    uint64_t const philox_offset =
        static_cast<uint64_t>(noise_rid) * static_cast<uint64_t>(max_seq) +
        pos + 1;

    // Per-request runtime sampling params. The row's 12-lane int32 record
    // (lane layout: runtime_header.h SamplingParamLane, written verbatim by
    // the scheduler's admission drain) overrides the compiled constants when
    // flags bit0 is set; flags==0 keeps the compiled-constant fast path —
    // the arithmetic below is then exactly the pre-override expression, so
    // all-defaults rows stay bitwise identical. The branch is BLOCK-UNIFORM
    // (every thread is on the same row). The milli-int decodings match the
    // codegen literals (1000.0f / temp_milli, milli * 1e-3f) so a value
    // supplied per-request is bitwise identical to the same value compiled
    // in. sampling_params_ptr is nullptr in the legacy codegen forms, where
    // this whole block folds away.
    float row_inv_temperature = inv_temperature;
    float row_frequency_penalty = frequency_penalty;
    float row_presence_penalty = presence_penalty;
    float row_repetition_penalty = repetition_penalty;
    uint64_t row_philox_seed = philox_seed;
    if (sampling_params_ptr != nullptr) {
      if (threadIdx.x == 0) {
        int const *sp = sampling_params_ptr + noise_rid * 12;
#pragma unroll
        for (int j = 0; j < 9; ++j) {
          sp_shared[j] = sp[j];
        }
      }
      __syncthreads();
      int const sp_flags = sp_shared[0];
      if (sp_flags & 1) {
        if (sp_shared[1] > 0) { // temp_milli; 0 sentinel = engine default
          row_inv_temperature = 1000.0f / static_cast<float>(sp_shared[1]);
        }
        if (sp_flags & 2) { // per-request seed present (int64 split)
          row_philox_seed =
              static_cast<uint64_t>(static_cast<uint32_t>(sp_shared[4])) |
              (static_cast<uint64_t>(static_cast<uint32_t>(sp_shared[5])) << 32);
        }
        row_frequency_penalty = static_cast<float>(sp_shared[6]) * 1e-3f;
        row_presence_penalty = static_cast<float>(sp_shared[7]) * 1e-3f;
        row_repetition_penalty = static_cast<float>(sp_shared[8]) * 1e-3f;
      }
    }
    // Stateless history-scan penalties (OpenAI frequency/presence + HF
    // repetition). Uniform branch: at the no-op defaults (0/0/1) — including
    // the legacy 3-arg codegen form that relies on the default arguments —
    // the whole scan folds away and the emitted arithmetic below is exactly
    // the pre-penalty expression, keeping existing bitwise gates valid.
    bool const do_penalties = (row_frequency_penalty != 0.0f) ||
                              (row_presence_penalty != 0.0f) ||
                              (row_repetition_penalty != 1.0f);

    // Generated-token history for this request: tokens[] holds
    // prompt + generated at their absolute positions (keyed by the stable
    // buffer row, same as the noise), so the completion-so-far is
    // [prompt_len, pos]. Empty for prefill rows (pos + 1 <= prompt_len),
    // so prompt processing is untouched.
    int pen_hist_begin = 0;
    int pen_hist_end = 0;
    if (do_penalties) {
      pen_hist_begin = prompt_lengths_ptr[noise_rid];
      pen_hist_end = pos + 1;
    }

    SamplingDataAndIndex<DType, IdType> max_data = {
        sampling_neg_inf<DType>(), 0};
    uint32_t const n_chunks =
        sampling_ceil_div(static_cast<uint32_t>(CHUNK_SIZE),
                          BLOCK_THREADS * VEC_SIZE);
    uint64_t const row_off =
        static_cast<uint64_t>(row) * CHUNK_SIZE * NUM_PARTIAL_TASKS;
    uint32_t const global_part_off = part_idx * CHUNK_SIZE;

    for (uint32_t c = 0; c < n_chunks; ++c) {
      uint32_t const local_base = (c * BLOCK_THREADS + tx) * VEC_SIZE;
      sampling_vec_t<DType, VEC_SIZE> logits_vec;
      logits_vec.fill(sampling_neg_inf<DType>());
      if (local_base < CHUNK_SIZE && global_part_off + local_base < vocab_size) {
        logits_vec.cast_load(logits + row_off + local_base);
      }
      sampling_vec_t<DType, VEC_SIZE> gumbel_noise =
          GenerateSamplingGumbelNoise<DType, VEC_SIZE>(
              row_philox_seed, philox_offset,
              static_cast<uint64_t>(global_part_off + local_base));

      // Per-element occurrence counts of this thread's vocab ids in the
      // generated history. Fixed sequential scan order, per-thread register
      // counters, no atomics -> deterministic and bitwise-reproducible.
      int pen_count[VEC_SIZE];
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        pen_count[j] = 0;
      }
      if (do_penalties) {
        long long const hist_base =
            static_cast<long long>(noise_rid) * max_seq;
        for (int t = pen_hist_begin; t < pen_hist_end; ++t) {
          long long const tok = all_tokens_ptr[hist_base + t];
#pragma unroll
          for (uint32_t j = 0; j < VEC_SIZE; ++j) {
            if (tok == static_cast<long long>(global_part_off + local_base +
                                              j)) {
              ++pen_count[j];
            }
          }
        }
      }

      SamplingDataAndIndex<DType, IdType> candidates[VEC_SIZE];
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        uint32_t const local_idx = local_base + j;
        uint32_t const global_idx = global_part_off + local_idx;
        // Penalties act on the raw fp32 logit BEFORE temperature scaling
        // and the Gumbel draw. HF repetition first (logit>0 ? l/rp : l*rp),
        // then the OpenAI subtraction:
        //   logit -= presence*(count>0) + frequency*count.
        float logit_f = static_cast<float>(logits_vec[j]);
        if (do_penalties && pen_count[j] > 0) {
          if (row_repetition_penalty != 1.0f) {
            logit_f = logit_f > 0.0f ? logit_f / row_repetition_penalty
                                     : logit_f * row_repetition_penalty;
          }
          logit_f -= row_presence_penalty +
                     row_frequency_penalty * static_cast<float>(pen_count[j]);
        }
        candidates[j].data =
            local_idx < CHUNK_SIZE && global_idx < vocab_size
                ? static_cast<DType>(logit_f * row_inv_temperature) +
                      gumbel_noise[j]
                : sampling_neg_inf<DType>();
        candidates[j].index = local_idx;
      }
      max_data += BlockReduce<SamplingDataAndIndex<DType, IdType>,
                              BLOCK_THREADS,
                              SAMPLING_REDUCE_ALGO>(temp_storage)
                      .template Sum<VEC_SIZE>(candidates);
      // cub requires a barrier before temp_storage is REUSED. Without it,
      // thread 0's serial combine of warp_aggregates (this chunk) races
      // with the other warps' lane-0 stores for the NEXT chunk, splicing
      // (data, index) across chunks — the winning index could then be a
      // guarded out-of-range candidate ((256+tx)*4+3 >= CHUNK_SIZE), i.e.
      // a token in the NEXT shard's range. Root cause of the rare
      // one-token nondeterminism in seeded decode (~1-3/30 runs).
      __syncthreads();
    }

    if (tx == 0) {
      partial_values[row * NUM_PARTIAL_TASKS] = max_data.data;
      partial_indices[row * NUM_PARTIAL_TASKS] = max_data.index;
    }
    __syncthreads();
  }
}

template <uint32_t BLOCK_THREADS,
          uint32_t VEC_SIZE,
          typename DType,
          typename IdType>
__device__ __forceinline__ void
    sampling_from_logits_kernel(DType *logits,
                                IdType *output,
                                uint32_t d,
                                uint64_t philox_seed,
                                uint64_t philox_offset,
                                int batch_size) {
  uint32_t const tx = threadIdx.x;

  using SharedMem = typename BlockReduce<SamplingDataAndIndex<DType, IdType>,
                                         BLOCK_THREADS,
                                         SAMPLING_REDUCE_ALGO>::TempStorage;
  extern __shared__ __align__(alignof(SharedMem)) uint8_t smem_sampling_logit[];
  auto &temp_storage = reinterpret_cast<SharedMem &>(smem_sampling_logit);

  // Loop over all batches
  for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    sampling_vec_t<DType, VEC_SIZE> logits_vec;
    SamplingDataAndIndex<DType, IdType> max_data = {
        sampling_neg_inf<DType>(), 0};

    // Process logits in chunks with vectorized loads
    for (uint32_t i = 0; i < sampling_ceil_div(d, BLOCK_THREADS * VEC_SIZE);
         ++i) {
      logits_vec.fill(sampling_neg_inf<DType>());

      // Load logits vector if within bounds
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        logits_vec.cast_load(logits + batch_idx * d +
                             i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
      }

      // Generate Gumbel noise
      sampling_vec_t<DType, VEC_SIZE> gumbel_noise =
          GenerateSamplingGumbelNoise<DType, VEC_SIZE>(
              philox_seed,
              philox_offset,
              static_cast<uint64_t>(batch_idx * d +
                                    (i * BLOCK_THREADS + tx) * VEC_SIZE));

      // Add noise to logits and prepare for reduction
      SamplingDataAndIndex<DType, IdType> cur_data[VEC_SIZE];
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        cur_data[j].data = (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d
                               ? logits_vec[j] + gumbel_noise[j]
                               : sampling_neg_inf<DType>();
        cur_data[j].index = (i * BLOCK_THREADS + tx) * VEC_SIZE + j;
      }

      // Find maximum across block
      max_data += BlockReduce<SamplingDataAndIndex<DType, IdType>,
                              BLOCK_THREADS,
                              SAMPLING_REDUCE_ALGO>(temp_storage)
                      .template Sum<VEC_SIZE>(cur_data);
      // Barrier before temp_storage reuse (see
      // sampling_partial_poskeyed_kernel).
      __syncthreads();
    }

    // Write output for this batch
    if (tx == 0) {
      output[batch_idx] = max_data.index;
    }

    // Sync before next batch iteration to reuse shared memory
    __syncthreads();
  }
}

// Position-keyed variant: Gumbel noise is a pure function of
// (philox_seed, request, absolute sequence position) instead of
// (seed, step, window row). This makes sampling invariant to how the
// window is chunked (prefill tail row vs decode row 0) and to batch
// composition, and reproducible when a trajectory prefix is replayed.
// Row -> (request, position) mapping mirrors
// multitoken_paged_attention_sm100_task_impl / prefill_prob_capture.
template <uint32_t BLOCK_THREADS,
          uint32_t VEC_SIZE,
          typename DType,
          typename IdType,
          bool CAPTURE_PROBS = false>
__device__ __forceinline__ void
    sampling_from_logits_poskeyed_kernel(int const my_rid,
                                         int const num_rid_tasks,
                                         DType *logits,
                                         IdType *output,
                                         uint32_t d,
                                         uint64_t philox_seed,
                                         int const *request_ids_ptr,
                                         int const *qo_indptr_buffer_ptr,
                                         int const *paged_kv_indptr_buffer_ptr,
                                         int const *paged_kv_last_page_len_ptr,
                                         int num_requests,
                                         int page_size,
                                         int max_seq,
                                         float inv_temperature = 1.0f,
                                         int top_k = 0,
                                         int top_p_milli = 1000,
                                         float *prob_buffer_ptr = nullptr,
                                         int const *prompt_lengths_ptr = nullptr,
                                         long long const *all_tokens_ptr = nullptr,
                                         int const *step_ptr = nullptr) {
  uint32_t const tx = threadIdx.x;

  using SharedMem = typename BlockReduce<SamplingDataAndIndex<DType, IdType>,
                                         BLOCK_THREADS,
                                         SAMPLING_REDUCE_ALGO>::TempStorage;
  extern __shared__ __align__(alignof(SharedMem)) uint8_t smem_sampling_logit[];
  auto &temp_storage = reinterpret_cast<SharedMem &>(smem_sampling_logit);

  // Shared scratch for the deterministic threshold search: an integer
  // counter (top-k, order-independent exact) and a float accumulator
  // (top-p mass; single block + fixed thread layout => reproducible).
  __shared__ int smem_thr_count;
  __shared__ float smem_thr_mass;
  __shared__ float smem_capture_max[8];
  __shared__ float smem_capture_sum[8];
  bool const do_topk = top_k > 0;
  bool const do_topp = top_p_milli < 1000;

  // Parallel over requests when launched with grid.x > 1 (task_metadata
  // request_id): disjoint output rows, noise still f(seed, rid, pos).
  for (int rid = my_rid; rid < num_requests; rid += num_rid_tasks) {
    int const first_token_pos = qo_indptr_buffer_ptr[rid];
    int const last_token_pos = qo_indptr_buffer_ptr[rid + 1];
    int const num_tokens = last_token_pos - first_token_pos;
    if (num_tokens <= 0) {
      continue;
    }
    int const num_pages =
        paged_kv_indptr_buffer_ptr[rid + 1] - paged_kv_indptr_buffer_ptr[rid];
    int const seq_len =
        (num_pages - 1) * page_size + paged_kv_last_page_len_ptr[rid];

    for (int i = 0; i < num_tokens; ++i) {
      int const row = first_token_pos + i;
      int const pos = seq_len - num_tokens + i;
      // unique per (request, position); +1 keeps offset 0 unused.
      // Key by BUFFER ROW (request_ids[slot]), not batch slot: the slot
      // can change when the online window re-packs mid-request, while the
      // row is stable for the request's lifetime. Batch mode maps them
      // identically, so offline results are unchanged.
      int const noise_rid =
          request_ids_ptr[rid] < 0 ? rid : request_ids_ptr[rid];
      uint64_t const philox_offset =
          static_cast<uint64_t>(noise_rid) * static_cast<uint64_t>(max_seq) +
          pos + 1;

      sampling_vec_t<DType, VEC_SIZE> logits_vec;
      SamplingDataAndIndex<DType, IdType> max_data = {
          sampling_neg_inf<DType>(), 0};

      uint64_t const row_off = static_cast<uint64_t>(row) * d;
      uint32_t const n_chunks = sampling_ceil_div(d, BLOCK_THREADS * VEC_SIZE);
      OnlineSoftmaxStats capture_stats = {-1e30f, 0.0f};

      // ---- deterministic top-k / top-p threshold on the ORDERED KEY ----
      // bf16 bits -> uint16 monotone key (larger float => larger key), so
      // "keep value >= tau" is an integer key comparison. Temperature (a
      // positive scale) preserves order, so both thresholds are found by
      // bisecting the SAME 16-bit key space; the mass sum uses the
      // temperature-scaled logits. All reductions are integer counts or a
      // single-block fixed-layout float sum -> reproducible run to run.
      uint16_t keep_key = 0;  // 0 keeps everything
      if (do_topk || do_topp) {
        // top-p needs the normalizer once (full softmax mass, scaled).
        float total_mass = 1.0f;
        if (do_topp) {
          if (tx == 0) { smem_thr_mass = 0.0f; }
          __syncthreads();
          float lm = 0.0f;
          for (uint32_t c = 0; c < n_chunks; ++c) {
            uint32_t const base = (c * BLOCK_THREADS + tx) * VEC_SIZE;
            sampling_vec_t<DType, VEC_SIZE> lv;
            lv.fill(sampling_neg_inf<DType>());
            if (base < d) {
              lv.cast_load(logits + row_off + c * BLOCK_THREADS * VEC_SIZE +
                           tx * VEC_SIZE);
            }
#pragma unroll
            for (uint32_t j = 0; j < VEC_SIZE; ++j) {
              if (base + j < d) {
                lm += __expf(static_cast<float>(lv[j]) * inv_temperature);
              }
            }
          }
          atomicAdd_block(&smem_thr_mass, lm);
          __syncthreads();
          total_mass = smem_thr_mass;
        }
        float const p_target = static_cast<float>(top_p_milli) * 1e-3f;

        // int math to avoid uint16 overflow at hi-lo+1 == 0x10000.
        int lo = 0, hi = 0xFFFF;
        // Invariant: keys in [lo, 0xFFFF] satisfy every active constraint
        // (>= top_k elements AND >= p mass). Raise lo to the highest such
        // key. 16 steps pin the exact bf16 boundary.
        for (int it = 0; it < 16; ++it) {
          uint16_t const mid = static_cast<uint16_t>(lo + ((hi - lo + 1) >> 1));
          if (tx == 0) { smem_thr_count = 0; smem_thr_mass = 0.0f; }
          __syncthreads();
          int local_cnt = 0;
          float local_mass = 0.0f;
          for (uint32_t c = 0; c < n_chunks; ++c) {
            uint32_t const base = (c * BLOCK_THREADS + tx) * VEC_SIZE;
            sampling_vec_t<DType, VEC_SIZE> lv;
            lv.fill(sampling_neg_inf<DType>());
            if (base < d) {
              lv.cast_load(logits + row_off + c * BLOCK_THREADS * VEC_SIZE +
                           tx * VEC_SIZE);
            }
#pragma unroll
            for (uint32_t j = 0; j < VEC_SIZE; ++j) {
              if (base + j >= d) { continue; }
              if (sampling_bf16_orderkey(lv[j]) >= mid) {
                local_cnt += 1;
                if (do_topp) {
                  local_mass += __expf(static_cast<float>(lv[j]) *
                                       inv_temperature);
                }
              }
            }
          }
          atomicAdd_block(&smem_thr_count, local_cnt);
          if (do_topp) { atomicAdd_block(&smem_thr_mass, local_mass); }
          __syncthreads();
          bool ok = true;
          if (do_topk && smem_thr_count < top_k) { ok = false; }
          if (do_topp && smem_thr_mass < p_target * total_mass) { ok = false; }
          lo = ok ? mid : lo;
          hi = ok ? hi : (mid - 1);
          __syncthreads();
        }
        keep_key = static_cast<uint16_t>(lo);
      }

      for (uint32_t c = 0; c < n_chunks; ++c) {
        logits_vec.fill(sampling_neg_inf<DType>());
        if ((c * BLOCK_THREADS + tx) * VEC_SIZE < d) {
          logits_vec.cast_load(logits + row_off +
                               c * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
        }

        if constexpr (CAPTURE_PROBS) {
#pragma unroll
          for (uint32_t j = 0; j < VEC_SIZE; ++j) {
            uint32_t const idx = (c * BLOCK_THREADS + tx) * VEC_SIZE + j;
            if (idx < d) {
              capture_stats = online_softmax_add(
                  capture_stats, static_cast<float>(logits_vec[j]));
            }
          }
        }

        // noise subsequence is the vocab element index only; the
        // (request, position) identity lives in philox_offset
        sampling_vec_t<DType, VEC_SIZE> gumbel_noise =
            GenerateSamplingGumbelNoise<DType, VEC_SIZE>(
                philox_seed,
                philox_offset,
                static_cast<uint64_t>((c * BLOCK_THREADS + tx) * VEC_SIZE));

        SamplingDataAndIndex<DType, IdType> cur_data[VEC_SIZE];
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
          uint32_t const idx = (c * BLOCK_THREADS + tx) * VEC_SIZE + j;
          bool keep = idx < d;
          if (keep && (do_topk || do_topp)) {
            keep = sampling_bf16_orderkey(logits_vec[j]) >= keep_key;
          }
          cur_data[j].data =
              keep ? static_cast<DType>(static_cast<float>(logits_vec[j]) *
                                        inv_temperature) +
                         gumbel_noise[j]
                   : sampling_neg_inf<DType>();
          cur_data[j].index = idx;
        }

        max_data += BlockReduce<SamplingDataAndIndex<DType, IdType>,
                                BLOCK_THREADS,
                                SAMPLING_REDUCE_ALGO>(temp_storage)
                        .template Sum<VEC_SIZE>(cur_data);
        // Barrier before temp_storage reuse (see
        // sampling_partial_poskeyed_kernel).
        __syncthreads();
      }

      if (tx == 0) {
        output[row] = max_data.index;
      }

      if constexpr (CAPTURE_PROBS) {
        // The online (max, sum) state is accumulated while sampling already
        // has each logit in registers. The shared fixed reduction tree is
        // also used by standalone capture and softmax-gather.
        static_assert(BLOCK_THREADS == 256);
        capture_stats = online_softmax_reduce_256(
            capture_stats, smem_capture_max, smem_capture_sum);

        if (tx == 0) {
          int const req_row =
              request_ids_ptr[rid] < 0 ? rid : request_ids_ptr[rid];
          int const prompt_len = prompt_lengths_ptr[req_row];
          long long target = -1;
          if (pos + 1 < prompt_len) {
            target = all_tokens_ptr[
                static_cast<long long>(req_row) * max_seq + pos + 1];
          } else if (i == num_tokens - 1) {
            target = max_data.index;
          }
          int const target_id = static_cast<int>(target);
          int const slot = step_ptr[req_row] + i;
          if (target_id >= 0 && target_id < static_cast<int>(d) &&
              slot >= 0 && slot < max_seq) {
            float const target_logit =
                static_cast<float>(logits[row_off + target_id]);
            prob_buffer_ptr[
                static_cast<long long>(req_row) * max_seq + slot] =
                __expf(target_logit - capture_stats.max) /
                capture_stats.sum;
          }
        }
      }
      __syncthreads();
    }
  }
}

} // namespace kernel
