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

struct OnlineSoftmaxStats {
  float max;
  float sum;
};

__device__ __forceinline__ OnlineSoftmaxStats
    online_softmax_add(OnlineSoftmaxStats stats, float value) {
  if (stats.sum == 0.0f) {
    return {value, 1.0f};
  }
  if (value > stats.max) {
    stats.sum = stats.sum * __expf(stats.max - value) + 1.0f;
    stats.max = value;
  } else {
    stats.sum += __expf(value - stats.max);
  }
  return stats;
}

__device__ __forceinline__ OnlineSoftmaxStats
    online_softmax_combine(OnlineSoftmaxStats lhs, OnlineSoftmaxStats rhs) {
  if (lhs.sum == 0.0f) {
    return rhs;
  }
  if (rhs.sum == 0.0f) {
    return lhs;
  }
  float const max_value = fmaxf(lhs.max, rhs.max);
  float const sum = lhs.sum * __expf(lhs.max - max_value) +
                    rhs.sum * __expf(rhs.max - max_value);
  return {max_value, sum};
}

// Fixed 256-thread reduction tree shared by every selected-token softmax
// path. Callers must scan logits in the same (chunk, lane, vector-element)
// order to obtain bitwise-identical probabilities.
__device__ __forceinline__ OnlineSoftmaxStats online_softmax_reduce_256(
    OnlineSoftmaxStats stats,
    float *__restrict__ warp_max,
    float *__restrict__ warp_sum) {
  int const lane_id = threadIdx.x % 32;
  int const warp_id = threadIdx.x / 32;
  for (int offset = 16; offset > 0; offset >>= 1) {
    OnlineSoftmaxStats const rhs = {
        __shfl_down_sync(0xffffffff, stats.max, offset),
        __shfl_down_sync(0xffffffff, stats.sum, offset)};
    if (lane_id < offset) {
      stats = online_softmax_combine(stats, rhs);
    }
  }
  if (lane_id == 0) {
    warp_max[warp_id] = stats.max;
    warp_sum[warp_id] = stats.sum;
  }
  __syncthreads();

  if (threadIdx.x < 8) {
    stats = {warp_max[threadIdx.x], warp_sum[threadIdx.x]};
    for (int offset = 4; offset > 0; offset >>= 1) {
      OnlineSoftmaxStats const rhs = {
          __shfl_down_sync(0xff, stats.max, offset),
          __shfl_down_sync(0xff, stats.sum, offset)};
      if (lane_id < offset) {
        stats = online_softmax_combine(stats, rhs);
      }
    }
    if (threadIdx.x == 0) {
      warp_max[0] = stats.max;
      warp_sum[0] = stats.sum;
    }
  }
  __syncthreads();
  return {warp_max[0], warp_sum[0]};
}

} // namespace kernel
