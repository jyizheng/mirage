/* Copyright 2025 Mirage Team
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

// Manifold-constrained hyper-connections (DeepSeek-V4.1). The residual stream
// of each token is HC parallel copies of the hidden state, [HC, DIM]. Every
// sublayer derives MIX = (2 + HC) * HC coefficients from the stream:
//   pre[HC]       collapses the copies into the sublayer input  (hc_pre)
//   post[HC]      spreads the sublayer output over the copies   (hc_post)
//   comb[HC][HC]  doubly stochastic residual mixing (Sinkhorn)  (hc_post)
// stored per token as one fp32 vector [pre | post | comb].

// mix = sinkhorn_split(rsqrt(mean(x^2) + norm_eps) * (fn @ x)), one task per
// BATCH_SIZE tokens. x: bf16 [BATCH, HC * DIM] with row stride X_STRIDE; fn:
// fp32 [MIX, HC * DIM]; scale: fp32 [3]; base: fp32 [MIX]; mix: fp32
// [BATCH, MIX]. Matches hc_mixes + hc_split_sinkhorn in inference/model.py.
template <typename T,
          int BATCH_SIZE,
          int HC,
          int DIM,
          int X_STRIDE,
          int SINKHORN_ITERS>
__device__ __forceinline__ void
    hc_mixes_sm100_task_impl(void const *input_ptr,
                             void const *fn_ptr,
                             void const *scale_ptr,
                             void const *base_ptr,
                             void *mix_ptr,
                             float norm_eps,
                             float hc_eps) {
  constexpr int MIX = (2 + HC) * HC;
  constexpr int K = HC * DIM;
  constexpr int NUM_WARPS = NUM_THREADS / 32;
  // One partial per output coefficient plus the sum of squares.
  constexpr int NACC = MIX + 1;
  static_assert(K % (2 * NUM_THREADS) == 0, "HC * DIM must split over threads");

  T const *__restrict__ d_x = static_cast<T const *>(input_ptr);
  float const *__restrict__ d_fn = static_cast<float const *>(fn_ptr);
  float const *__restrict__ d_scale = static_cast<float const *>(scale_ptr);
  float const *__restrict__ d_base = static_cast<float const *>(base_ptr);
  float *__restrict__ d_mix = static_cast<float *>(mix_ptr);

  extern __shared__ char smem[];
  float(*s_part)[NACC] = reinterpret_cast<float(*)[NACC]>(smem);
  float *s_mix = reinterpret_cast<float *>(smem) + NUM_WARPS * NACC;

  int const lane = threadIdx.x % 32;
  int const warp = threadIdx.x / 32;

  for (int b = 0; b < BATCH_SIZE; ++b) {
    T const *x = d_x + b * X_STRIDE;
    float acc[NACC];
#pragma unroll
    for (int m = 0; m < NACC; ++m) {
      acc[m] = 0.0f;
    }
    // Two consecutive elements per thread so the bf16 load is 4 bytes and the
    // fp32 weight load is 8 bytes; consecutive threads stay contiguous.
    for (int k = 2 * threadIdx.x; k < K; k += 2 * NUM_THREADS) {
      float x0 = float(x[k]);
      float x1 = float(x[k + 1]);
      acc[MIX] += x0 * x0 + x1 * x1;
#pragma unroll
      for (int m = 0; m < MIX; ++m) {
        float2 w = *reinterpret_cast<float2 const *>(d_fn + m * K + k);
        acc[m] += w.x * x0 + w.y * x1;
      }
    }
#pragma unroll
    for (int m = 0; m < NACC; ++m) {
      float v = acc[m];
#pragma unroll
      for (int off = 16; off > 0; off /= 2) {
        v += __shfl_xor_sync(0xffffffff, v, off);
      }
      if (lane == 0) {
        s_part[warp][m] = v;
      }
    }
    __syncthreads();
    if (threadIdx.x < MIX) {
      float dot = 0.0f, sumsq = 0.0f;
#pragma unroll
      for (int w = 0; w < NUM_WARPS; ++w) {
        dot += s_part[w][threadIdx.x];
        sumsq += s_part[w][MIX];
      }
      float rs = rsqrtf(sumsq / float(K) + norm_eps);
      int const m = threadIdx.x;
      int const group = m < HC ? 0 : (m < 2 * HC ? 1 : 2);
      s_mix[m] = dot * rs * d_scale[group] + d_base[m];
    }
    __syncthreads();
    // The split and the HC x HC Sinkhorn are a few hundred flops: one thread.
    if (threadIdx.x == 0) {
      float *out = d_mix + b * MIX;
#pragma unroll
      for (int j = 0; j < HC; ++j) {
        out[j] = 1.0f / (1.0f + expf(-s_mix[j])) + hc_eps;
        out[HC + j] = 2.0f / (1.0f + expf(-s_mix[HC + j]));
      }
      float c[HC][HC];
#pragma unroll
      for (int j = 0; j < HC; ++j) {
        float mx = s_mix[2 * HC + j * HC];
#pragma unroll
        for (int k = 1; k < HC; ++k) {
          mx = fmaxf(mx, s_mix[2 * HC + j * HC + k]);
        }
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < HC; ++k) {
          c[j][k] = expf(s_mix[2 * HC + j * HC + k] - mx);
          sum += c[j][k];
        }
#pragma unroll
        for (int k = 0; k < HC; ++k) {
          c[j][k] = c[j][k] / sum + hc_eps;
        }
      }
      // Column normalize, then SINKHORN_ITERS - 1 rounds of row + column.
      for (int it = 0; it < SINKHORN_ITERS; ++it) {
        if (it > 0) {
#pragma unroll
          for (int j = 0; j < HC; ++j) {
            float sum = 0.0f;
#pragma unroll
            for (int k = 0; k < HC; ++k) {
              sum += c[j][k];
            }
#pragma unroll
            for (int k = 0; k < HC; ++k) {
              c[j][k] = c[j][k] / (sum + hc_eps);
            }
          }
        }
#pragma unroll
        for (int k = 0; k < HC; ++k) {
          float sum = 0.0f;
#pragma unroll
          for (int j = 0; j < HC; ++j) {
            sum += c[j][k];
          }
#pragma unroll
          for (int j = 0; j < HC; ++j) {
            c[j][k] = c[j][k] / (sum + hc_eps);
          }
        }
      }
#pragma unroll
      for (int j = 0; j < HC; ++j) {
#pragma unroll
        for (int k = 0; k < HC; ++k) {
          out[2 * HC + j * HC + k] = c[j][k];
        }
      }
    }
    __syncthreads();
  }
} // hc_mixes_sm100_task_impl

// y[d] = sum_j pre[j] * x[j][d] (hc_pre). x: bf16 [BATCH, HC, *] with row
// stride X_STRIDE and copy stride X_HC_STRIDE, this task's DIM_CHUNK columns;
// mix: fp32 [BATCH, MIX] (pre at offset 0); y: bf16 [BATCH, DIM_CHUNK] with
// row stride Y_STRIDE.
template <typename T,
          int BATCH_SIZE,
          int HC,
          int DIM_CHUNK,
          int X_STRIDE,
          int X_HC_STRIDE,
          int Y_STRIDE>
__device__ __forceinline__ void hc_pre_sm100_task_impl(void const *input_ptr,
                                                       void const *mix_ptr,
                                                       void *output_ptr) {
  constexpr int MIX = (2 + HC) * HC;
  T const *__restrict__ d_x = static_cast<T const *>(input_ptr);
  float const *__restrict__ d_mix = static_cast<float const *>(mix_ptr);
  T *__restrict__ d_y = static_cast<T *>(output_ptr);
  for (int b = 0; b < BATCH_SIZE; ++b) {
    float pre[HC];
#pragma unroll
    for (int j = 0; j < HC; ++j) {
      pre[j] = d_mix[b * MIX + j];
    }
    for (int d = threadIdx.x; d < DIM_CHUNK; d += NUM_THREADS) {
      float y = 0.0f;
#pragma unroll
      for (int j = 0; j < HC; ++j) {
        y += pre[j] * float(d_x[b * X_STRIDE + j * X_HC_STRIDE + d]);
      }
      d_y[b * Y_STRIDE + d] = T(y);
    }
  }
} // hc_pre_sm100_task_impl

// out[i][d] = post[i] * x[d] + sum_j comb[j][i] * residual[j][d] (hc_post).
// x: bf16 [BATCH, DIM_CHUNK] (row stride X_STRIDE); residual and out: bf16
// [BATCH, HC, *] with row stride R_STRIDE and copy stride R_HC_STRIDE; mix:
// fp32 [BATCH, MIX].
template <typename T,
          int BATCH_SIZE,
          int HC,
          int DIM_CHUNK,
          int X_STRIDE,
          int R_STRIDE,
          int R_HC_STRIDE>
__device__ __forceinline__ void hc_post_sm100_task_impl(void const *input_ptr,
                                                        void const *residual_ptr,
                                                        void const *mix_ptr,
                                                        void *output_ptr) {
  constexpr int MIX = (2 + HC) * HC;
  T const *__restrict__ d_x = static_cast<T const *>(input_ptr);
  T const *__restrict__ d_r = static_cast<T const *>(residual_ptr);
  float const *__restrict__ d_mix = static_cast<float const *>(mix_ptr);
  T *__restrict__ d_out = static_cast<T *>(output_ptr);
  for (int b = 0; b < BATCH_SIZE; ++b) {
    float post[HC], comb[HC][HC];
#pragma unroll
    for (int j = 0; j < HC; ++j) {
      post[j] = d_mix[b * MIX + HC + j];
#pragma unroll
      for (int i = 0; i < HC; ++i) {
        comb[j][i] = d_mix[b * MIX + 2 * HC + j * HC + i];
      }
    }
    for (int d = threadIdx.x; d < DIM_CHUNK; d += NUM_THREADS) {
      float x = float(d_x[b * X_STRIDE + d]);
      float r[HC];
#pragma unroll
      for (int j = 0; j < HC; ++j) {
        r[j] = float(d_r[b * R_STRIDE + j * R_HC_STRIDE + d]);
      }
#pragma unroll
      for (int i = 0; i < HC; ++i) {
        float mixed = 0.0f;
#pragma unroll
        for (int j = 0; j < HC; ++j) {
          mixed += comb[j][i] * r[j];
        }
        d_out[b * R_STRIDE + i * R_HC_STRIDE + d] = T(post[i] * x + mixed);
      }
    }
  }
} // hc_post_sm100_task_impl

} // namespace kernel
