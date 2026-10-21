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
#include <cuda_fp4.h>
#include <cuda_fp8.h>

// DeepSeek-V4.1 decode tasks (one token per step), SM100. Numerics follow
// DeepSeek's inference/model.py and kernel.py: bf16 storage between ops, fp32
// math inside them, and the reference's fake quantization (quantize, then
// dequantize back to bf16) at the same points.
//
// Tasks that depend on the position read it from `step` (int32, the position
// of the token being decoded).

namespace kernel {
namespace dsv4 {

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off /= 2) {
    v += __shfl_xor_sync(0xffffffff, v, off);
  }
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int off = 16; off > 0; off /= 2) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
  }
  return v;
}

// Block-wide sum over NUM_THREADS threads; `red` holds NUM_THREADS / 32
// floats. Every thread gets the result.
__device__ __forceinline__ float block_sum(float v, float *red) {
  constexpr int NUM_WARPS = NUM_THREADS / 32;
  v = warp_sum(v);
  __syncthreads();
  if (threadIdx.x % 32 == 0) {
    red[threadIdx.x / 32] = v;
  }
  __syncthreads();
  float total = 0.0f;
#pragma unroll
  for (int w = 0; w < NUM_WARPS; ++w) {
    total += red[w];
  }
  return total;
}

__device__ __forceinline__ float block_max(float v, float *red) {
  constexpr int NUM_WARPS = NUM_THREADS / 32;
  v = warp_max(v);
  __syncthreads();
  if (threadIdx.x % 32 == 0) {
    red[threadIdx.x / 32] = v;
  }
  __syncthreads();
  float m = red[0];
#pragma unroll
  for (int w = 1; w < NUM_WARPS; ++w) {
    m = fmaxf(m, red[w]);
  }
  return m;
}

// 2^ceil(log2(x)) by exponent bits, as kernel.py's fast_round_scale.
__device__ __forceinline__ float pow2_ceil(float x) {
  uint32_t bits = __float_as_uint(x);
  int e = int((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0 ? 1 : 0);
  return __uint_as_float(uint32_t(e + 127) << 23);
}

// Fake quantization of one value whose 32-lane (or 16-lane) group shares
// `amax`. QUANT_FP8: e4m3 with a power-of-two scale (act_quant, ue8m0).
// QUANT_FP4_E4M3: e2m1 with an e4m3 scale (fp4_act_quant, compressed KV).
enum QuantKind { QUANT_NONE = 0, QUANT_FP8 = 1, QUANT_FP4_E4M3 = 2 };

template <int QUANT>
__device__ __forceinline__ float fake_quant(float v, float amax) {
  if constexpr (QUANT == QUANT_FP8) {
    float s = pow2_ceil(fmaxf(amax, 1e-4f) * (1.0f / 448.0f));
    float q = fminf(fmaxf(v / s, -448.0f), 448.0f);
    return float(__nv_fp8_e4m3(q)) * s;
  } else if constexpr (QUANT == QUANT_FP4_E4M3) {
    float s = float(__nv_fp8_e4m3(fmaxf(amax, 6.0f * 0.001953125f) / 6.0f));
    float q = fminf(fmaxf(v / s, -6.0f), 6.0f);
    return float(__nv_fp4_e2m1(q)) * s;
  } else {
    return v;
  }
}

} // namespace dsv4

// ---------------------------------------------------------------------------
// GEMV: out[g][n] = sum_k x[g'][k] * W[e][n][k], fp32 accumulation.
// One task computes N_CHUNK rows of G_PER_TASK groups. The weight group e is
// the task's group, or idx[g] when HAS_IDX (MoE experts). X_GROUP_STRIDE = 0
// broadcasts one input row to every group. Output is bf16 (a bf16 Linear) or
// fp32 (the reference's fp32 Linear on upcast weights, e.g. the gate and head).
// ---------------------------------------------------------------------------
template <typename T,
          int K,
          int N_CHUNK,
          int G_PER_TASK,
          int X_GROUP_STRIDE,
          int W_ROW_STRIDE,
          long long W_GROUP_STRIDE,
          int OUT_GROUP_STRIDE,
          int HAS_IDX,
          int OUT_FP32>
__device__ __forceinline__ void dsv4_gemv_task_impl(void const *x_ptr,
                                                    void const *w_ptr,
                                                    void const *idx_ptr,
                                                    void *out_ptr) {
  constexpr int NUM_WARPS = NUM_THREADS / 32;
  constexpr int VEC = 8; // bf16 per 16-byte load
  static_assert(K % (32 * VEC) == 0, "K must be a multiple of 256");
  T const *__restrict__ d_x = static_cast<T const *>(x_ptr);
  T const *__restrict__ d_w = static_cast<T const *>(w_ptr);
  int const *__restrict__ d_idx = static_cast<int const *>(idx_ptr);
  extern __shared__ char smem[];
  float *s_x = reinterpret_cast<float *>(smem);
  int const lane = threadIdx.x % 32;
  int const warp = threadIdx.x / 32;

  for (int g = 0; g < G_PER_TASK; ++g) {
    __syncthreads();
    for (int k = threadIdx.x; k < K; k += NUM_THREADS) {
      s_x[k] = float(d_x[g * X_GROUP_STRIDE + k]);
    }
    __syncthreads();
    long long w_group = HAS_IDX ? (long long)d_idx[g] * W_GROUP_STRIDE
                                : (long long)g * W_GROUP_STRIDE;
    for (int n = warp; n < N_CHUNK; n += NUM_WARPS) {
      T const *w_row = d_w + w_group + (long long)n * W_ROW_STRIDE;
      float acc = 0.0f;
#pragma unroll 4
      for (int k = lane * VEC; k < K; k += 32 * VEC) {
        uint4 packed = *reinterpret_cast<uint4 const *>(w_row + k);
        T const *w8 = reinterpret_cast<T const *>(&packed);
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
          acc += float(w8[i]) * s_x[k + i];
        }
      }
      acc = dsv4::warp_sum(acc);
      if (lane == 0) {
        if constexpr (OUT_FP32) {
          static_cast<float *>(out_ptr)[g * OUT_GROUP_STRIDE + n] = acc;
        } else {
          static_cast<T *>(out_ptr)[g * OUT_GROUP_STRIDE + n] = T(acc);
        }
      }
    }
  }
} // dsv4_gemv_task_impl

// ---------------------------------------------------------------------------
// RMSNorm over D: y = bf16(w * x * rsqrt(mean(x^2) + eps)), ROWS rows.
// ---------------------------------------------------------------------------
template <typename T, int ROWS, int D, int X_STRIDE, int Y_STRIDE>
__device__ __forceinline__ void dsv4_rmsnorm_task_impl(void const *x_ptr,
                                                       void const *w_ptr,
                                                       void *y_ptr,
                                                       float eps) {
  T const *__restrict__ d_x = static_cast<T const *>(x_ptr);
  T const *__restrict__ d_w = static_cast<T const *>(w_ptr);
  T *__restrict__ d_y = static_cast<T *>(y_ptr);
  extern __shared__ char smem[];
  float *red = reinterpret_cast<float *>(smem);
  for (int r = 0; r < ROWS; ++r) {
    float ss = 0.0f;
    for (int d = threadIdx.x; d < D; d += NUM_THREADS) {
      float v = float(d_x[r * X_STRIDE + d]);
      ss += v * v;
    }
    ss = dsv4::block_sum(ss, red);
    float rs = rsqrtf(ss / float(D) + eps);
    for (int d = threadIdx.x; d < D; d += NUM_THREADS) {
      float v = float(d_x[r * X_STRIDE + d]) * rs;
      d_y[r * Y_STRIDE + d] = T(float(d_w[d]) * v);
    }
  }
} // dsv4_rmsnorm_task_impl

// ---------------------------------------------------------------------------
// RoPE on the last RD dims of each of H rows of width D (adjacent pairs as
// complex numbers, INVERSE conjugates), rounded to bf16, then optional fake
// quantization in QBLOCK-wide groups, written to a cache slot.
//   rope position = step + POS_SHIFT
//   slot          = ((step + 1 - SLOT_DIV) / SLOT_DIV) % SLOT_MOD
//   written only when (step + 1) % SLOT_DIV == 0 (a compression group closes)
// SLOT_DIV = SLOT_MOD = 1 writes row 0 every step (a plain output).
// freqs: fp32 [max_pos, RD / 2, 2] = (cos, sin) per pair.
// ---------------------------------------------------------------------------
template <typename T,
          int H,
          int D,
          int RD,
          int INVERSE,
          int QUANT,
          int QBLOCK,
          int POS_SHIFT,
          int SLOT_DIV,
          int SLOT_MOD,
          int X_STRIDE,
          int OUT_STRIDE>
__device__ __forceinline__ void dsv4_rope_quant_task_impl(void const *x_ptr,
                                                          void const *freqs_ptr,
                                                          void const *step_ptr,
                                                          void *out_ptr) {
  static_assert(D % 32 == 0 && RD % 2 == 0 && RD <= D, "");
  static_assert(QBLOCK == 16 || QBLOCK == 32, "");
  T const *__restrict__ d_x = static_cast<T const *>(x_ptr);
  float const *__restrict__ d_freqs = static_cast<float const *>(freqs_ptr);
  int const step = *static_cast<int const *>(step_ptr);
  if ((step + 1) % SLOT_DIV != 0) {
    return;
  }
  int const slot = ((step + 1 - SLOT_DIV) / SLOT_DIV) % SLOT_MOD;
  int const pos = step + POS_SHIFT;
  T *__restrict__ d_out = static_cast<T *>(out_ptr) + (long long)slot * H * OUT_STRIDE;
  int const lane = threadIdx.x % 32;
  int const warp = threadIdx.x / 32;
  constexpr int NUM_WARPS = NUM_THREADS / 32;
  constexpr int CHUNKS = D / 32; // 32 elements per warp iteration
  for (int c = warp; c < H * CHUNKS; c += NUM_WARPS) {
    int const h = c / CHUNKS;
    int const d = (c % CHUNKS) * 32 + lane;
    float v = float(d_x[h * X_STRIDE + d]);
    // RD and D - RD are even, so a pair never straddles the RoPE boundary
    // and both halves of a pair sit in adjacent lanes.
    float partner = __shfl_xor_sync(0xffffffff, v, 1);
    if (d >= D - RD) {
      int const p = (d - (D - RD)) / 2;
      float cs = d_freqs[(pos * (RD / 2) + p) * 2];
      float sn = d_freqs[(pos * (RD / 2) + p) * 2 + 1];
      if (INVERSE) {
        sn = -sn;
      }
      // (re, im) * (cs, sn): re' = re*cs - im*sn, im' = re*sn + im*cs.
      v = (d % 2 == 0) ? (v * cs - partner * sn) : (partner * sn + v * cs);
    }
    v = float(T(v));
    if constexpr (QUANT != dsv4::QUANT_NONE) {
      float amax = fabsf(v);
#pragma unroll
      for (int off = QBLOCK / 2; off > 0; off /= 2) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
      }
      v = dsv4::fake_quant<QUANT>(v, amax);
    }
    d_out[h * OUT_STRIDE + d] = T(v);
  }
} // dsv4_rope_quant_task_impl

// ---------------------------------------------------------------------------
// Compressor pooling for compress_ratio RATIO > 1 (Compressor.forward decode
// path): stash this token's kv / score (fp32 [D]) in slot step % RATIO; when
// the group closes, pooled = bf16(sum_j kv_j * softmax_j(score_j)).
// ---------------------------------------------------------------------------
template <typename T, int D, int RATIO>
__device__ __forceinline__ void dsv4_compress_pool_task_impl(
    void const *kv_ptr,
    void const *score_ptr,
    void const *step_ptr,
    void *kv_state_ptr,
    void *score_state_ptr,
    void *pooled_ptr) {
  float const *__restrict__ d_kv = static_cast<float const *>(kv_ptr);
  float const *__restrict__ d_score = static_cast<float const *>(score_ptr);
  float *__restrict__ kv_state = static_cast<float *>(kv_state_ptr);
  float *__restrict__ score_state = static_cast<float *>(score_state_ptr);
  T *__restrict__ d_pooled = static_cast<T *>(pooled_ptr);
  int const step = *static_cast<int const *>(step_ptr);
  int const slot = step % RATIO;
  bool const close = (step + 1) % RATIO == 0;
  for (int d = threadIdx.x; d < D; d += NUM_THREADS) {
    kv_state[slot * D + d] = d_kv[d];
    score_state[slot * D + d] = d_score[d];
    if (close) {
      float m = score_state[d];
#pragma unroll
      for (int j = 1; j < RATIO; ++j) {
        m = fmaxf(m, score_state[j * D + d]);
      }
      float e[RATIO], sum = 0.0f;
#pragma unroll
      for (int j = 0; j < RATIO; ++j) {
        e[j] = expf(score_state[j * D + d] - m);
        sum += e[j];
      }
      float acc = 0.0f;
#pragma unroll
      for (int j = 0; j < RATIO; ++j) {
        acc += kv_state[j * D + d] * (e[j] / sum);
      }
      d_pooled[d] = T(acc);
    }
  }
} // dsv4_compress_pool_task_impl

// ---------------------------------------------------------------------------
// Decode attention over [window ring | compressed cache] with a per-head sink
// (sparse_attn with every reachable position selected, which is what the
// indexer's top-k returns while the compressed length is <= index_topk).
//   window slot s is valid iff s <= step (the ring is still filling);
//   compressed row j is valid iff j < (step + 1) / RATIO  (RATIO 0: none).
// K = V = the cached row. P is rounded to bf16 before the PV product, as the
// reference kernel does.
// q, out: [HEADS, D] (this task's heads); sink: fp32 [HEADS].
// ---------------------------------------------------------------------------
template <typename T,
          int HEADS,
          int D,
          int WINDOW,
          int RATIO,
          int MAX_COMPRESSED,
          int Q_STRIDE,
          int OUT_STRIDE>
__device__ __forceinline__ void dsv4_attention_task_impl(void const *q_ptr,
                                                         void const *win_ptr,
                                                         void const *comp_ptr,
                                                         void const *sink_ptr,
                                                         void const *step_ptr,
                                                         void *out_ptr,
                                                         float scale) {
  constexpr int NUM_WARPS = NUM_THREADS / 32;
  constexpr int MAX_KEYS = WINDOW + MAX_COMPRESSED;
  constexpr int DPL = D / 32; // dims per lane in the score dot product
  static_assert(D % 64 == 0, "");
  T const *__restrict__ d_q = static_cast<T const *>(q_ptr);
  T const *__restrict__ d_win = static_cast<T const *>(win_ptr);
  T const *__restrict__ d_comp = static_cast<T const *>(comp_ptr);
  float const *__restrict__ d_sink = static_cast<float const *>(sink_ptr);
  T *__restrict__ d_out = static_cast<T *>(out_ptr);
  int const step = *static_cast<int const *>(step_ptr);
  int const n_win = step + 1 < WINDOW ? step + 1 : WINDOW;
  int n_comp = RATIO > 0 ? (step + 1) / (RATIO > 0 ? RATIO : 1) : 0;
  n_comp = n_comp < MAX_COMPRESSED ? n_comp : MAX_COMPRESSED;
  int const n_keys = n_win + n_comp;

  extern __shared__ char smem[];
  float *s_score = reinterpret_cast<float *>(smem); // [HEADS][MAX_KEYS]
  float *s_red = s_score + HEADS * MAX_KEYS;        // [NUM_WARPS]
  int const lane = threadIdx.x % 32;
  int const warp = threadIdx.x / 32;

  // Scores: one warp per key, all heads at once.
  float qf[HEADS][DPL];
#pragma unroll
  for (int h = 0; h < HEADS; ++h) {
#pragma unroll
    for (int i = 0; i < DPL; ++i) {
      qf[h][i] = float(d_q[h * Q_STRIDE + lane + 32 * i]);
    }
  }
  for (int j = warp; j < n_keys; j += NUM_WARPS) {
    T const *row = j < n_win ? d_win + (long long)j * D
                             : d_comp + (long long)(j - n_win) * D;
    float kf[DPL];
#pragma unroll
    for (int i = 0; i < DPL; ++i) {
      kf[i] = float(row[lane + 32 * i]);
    }
#pragma unroll
    for (int h = 0; h < HEADS; ++h) {
      float s = 0.0f;
#pragma unroll
      for (int i = 0; i < DPL; ++i) {
        s += qf[h][i] * kf[i];
      }
      s = dsv4::warp_sum(s);
      if (lane == 0) {
        s_score[h * MAX_KEYS + j] = s * scale;
      }
    }
  }
  __syncthreads();

  for (int h = 0; h < HEADS; ++h) {
    float *sc = s_score + h * MAX_KEYS;
    float m = -1e30f;
    for (int j = threadIdx.x; j < n_keys; j += NUM_THREADS) {
      m = fmaxf(m, sc[j]);
    }
    m = dsv4::block_max(m, s_red);
    float sum = 0.0f;
    __syncthreads();
    for (int j = threadIdx.x; j < n_keys; j += NUM_THREADS) {
      float e = expf(sc[j] - m);
      sum += e;
      sc[j] = float(T(e)); // P in bf16 for the PV product
    }
    sum = dsv4::block_sum(sum, s_red) + expf(d_sink[h] - m);
    __syncthreads();
    for (int d = threadIdx.x; d < D; d += NUM_THREADS) {
      float acc = 0.0f;
      for (int j = 0; j < n_keys; ++j) {
        T const *row = j < n_win ? d_win + (long long)j * D
                                 : d_comp + (long long)(j - n_win) * D;
        acc += sc[j] * float(row[d]);
      }
      d_out[h * OUT_STRIDE + d] = T(acc / sum);
    }
  }
} // dsv4_attention_task_impl

// ---------------------------------------------------------------------------
// MoE gate (Gate.forward, sqrtsoftplus): scores = sqrt(softplus(logits));
// pick top-K of scores + bias; weights = scores of the picks, normalized
// (sum + 1e-20) and scaled by route_scale. One token, E experts.
// ---------------------------------------------------------------------------
template <int E, int TOPK>
__device__ __forceinline__ void dsv4_router_task_impl(void const *logits_ptr,
                                                      void const *bias_ptr,
                                                      void *weights_ptr,
                                                      void *idx_ptr,
                                                      float route_scale) {
  float const *__restrict__ d_logits = static_cast<float const *>(logits_ptr);
  float const *__restrict__ d_bias = static_cast<float const *>(bias_ptr);
  extern __shared__ char smem[];
  float *s_score = reinterpret_cast<float *>(smem); // [E]
  float *s_sel = s_score + E;                       // [E] score + bias
  for (int e = threadIdx.x; e < E; e += NUM_THREADS) {
    float l = d_logits[e];
    // torch softplus: x above threshold 20 stays linear.
    float sp = l > 20.0f ? l : log1pf(expf(l));
    float s = sqrtf(sp);
    s_score[e] = s;
    s_sel[e] = s + d_bias[e];
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    int const lane = threadIdx.x;
    float wsum = 0.0f;
    float w[TOPK];
    int id[TOPK];
    for (int t = 0; t < TOPK; ++t) {
      float best = -INFINITY;
      int best_e = E;
      for (int e = lane; e < E; e += 32) {
        float v = s_sel[e];
        if (v > best || (v == best && e < best_e)) {
          best = v;
          best_e = e;
        }
      }
#pragma unroll
      for (int off = 16; off > 0; off /= 2) {
        float ov = __shfl_xor_sync(0xffffffff, best, off);
        int oe = __shfl_xor_sync(0xffffffff, best_e, off);
        if (ov > best || (ov == best && oe < best_e)) {
          best = ov;
          best_e = oe;
        }
      }
      id[t] = best_e;
      w[t] = s_score[best_e];
      wsum += w[t];
      __syncwarp();
      if (lane == 0) {
        s_sel[best_e] = -INFINITY;
      }
      __syncwarp();
    }
    if (lane == 0) {
      for (int t = 0; t < TOPK; ++t) {
        static_cast<float *>(weights_ptr)[t] = w[t] / (wsum + 1e-20f) * route_scale;
        static_cast<int *>(idx_ptr)[t] = id[t];
      }
    }
  }
} // dsv4_router_task_impl

// ---------------------------------------------------------------------------
// Clamped SwiGLU (Expert.forward): h = [gate | up] bf16 [G, 2I];
// out = bf16(w_g * silu(min(gate, L)) * clamp(up, -L, L)), w_g = 1 without
// weights (the shared expert).
// ---------------------------------------------------------------------------
template <typename T, int G, int I, int HAS_WEIGHTS>
__device__ __forceinline__ void dsv4_swiglu_task_impl(void const *h_ptr,
                                                      void const *weights_ptr,
                                                      void *out_ptr,
                                                      float limit) {
  T const *__restrict__ d_h = static_cast<T const *>(h_ptr);
  float const *__restrict__ d_w = static_cast<float const *>(weights_ptr);
  T *__restrict__ d_out = static_cast<T *>(out_ptr);
  for (int i = threadIdx.x; i < G * I; i += NUM_THREADS) {
    int const g = i / I, c = i % I;
    float gate = float(d_h[g * 2 * I + c]);
    float up = float(d_h[g * 2 * I + I + c]);
    up = fminf(fmaxf(up, -limit), limit);
    gate = fminf(gate, limit);
    float v = gate / (1.0f + expf(-gate)) * up;
    if (HAS_WEIGHTS) {
      v = d_w[g] * v;
    }
    d_out[i] = T(v);
  }
} // dsv4_swiglu_task_impl

// ---------------------------------------------------------------------------
// MoE combine: out = bf16(sum of the TOPK expert outputs in ascending expert
// id + shared), as MoE.forward accumulates them in fp32. DIM_CHUNK columns.
// ---------------------------------------------------------------------------
template <typename T, int TOPK, int DIM_CHUNK, int EXPERT_STRIDE>
__device__ __forceinline__ void dsv4_moe_combine_task_impl(void const *experts_ptr,
                                                           void const *idx_ptr,
                                                           void const *shared_ptr,
                                                           void *out_ptr) {
  T const *__restrict__ d_exp = static_cast<T const *>(experts_ptr);
  int const *__restrict__ d_idx = static_cast<int const *>(idx_ptr);
  T const *__restrict__ d_shared = static_cast<T const *>(shared_ptr);
  T *__restrict__ d_out = static_cast<T *>(out_ptr);
  int order[TOPK];
#pragma unroll
  for (int t = 0; t < TOPK; ++t) {
    order[t] = t;
  }
  // Insertion sort of the slots by expert id.
#pragma unroll
  for (int a = 1; a < TOPK; ++a) {
#pragma unroll
    for (int b = a; b > 0; --b) {
      if (d_idx[order[b]] < d_idx[order[b - 1]]) {
        int tmp = order[b];
        order[b] = order[b - 1];
        order[b - 1] = tmp;
      }
    }
  }
  for (int d = threadIdx.x; d < DIM_CHUNK; d += NUM_THREADS) {
    float acc = 0.0f;
#pragma unroll
    for (int t = 0; t < TOPK; ++t) {
      acc += float(d_exp[order[t] * EXPERT_STRIDE + d]);
    }
    acc += float(d_shared[d]);
    d_out[d] = T(acc);
  }
} // dsv4_moe_combine_task_impl

// ---------------------------------------------------------------------------
// Engram n-gram hash (NgramHashState.forward, one position, one layer):
// t_s = token_map[tokens[step - s]] for s < NGRAM (pad_id once the look-back
// leaves the sequence), rolling XOR of t_s * mult[s]; hash i (the (i+2)-gram)
// of head h = rolling_i % primes[i][h] + offsets[i][h]. out: int32
// [(NGRAM-1) * HEADS].
// ---------------------------------------------------------------------------
template <int NGRAM, int HEADS>
__device__ __forceinline__ void dsv4_engram_hash_task_impl(
    void const *tokens_ptr,
    void const *token_map_ptr,
    void const *mult_ptr,
    void const *primes_ptr,
    void const *offsets_ptr,
    void const *step_ptr,
    void *out_ptr,
    int pad_id) {
  long long const *__restrict__ tokens = static_cast<long long const *>(tokens_ptr);
  long long const *__restrict__ token_map = static_cast<long long const *>(token_map_ptr);
  long long const *__restrict__ mult = static_cast<long long const *>(mult_ptr);
  long long const *__restrict__ primes = static_cast<long long const *>(primes_ptr);
  long long const *__restrict__ offsets = static_cast<long long const *>(offsets_ptr);
  int const step = *static_cast<int const *>(step_ptr);
  int const c = threadIdx.x;
  if (c >= (NGRAM - 1) * HEADS) {
    return;
  }
  int const i = c / HEADS; // n-gram size i + 2
  long long rolling = 0;
  for (int s = 0; s <= i + 1; ++s) {
    long long t = step - s >= 0 ? token_map[tokens[step - s]] : (long long)pad_id;
    long long p = t * mult[s];
    rolling = s == 0 ? p : (rolling ^ p);
  }
  long long h = rolling % primes[c];
  static_cast<int *>(out_ptr)[c] = int(h + offsets[c]);
} // dsv4_engram_hash_task_impl

// Engram table lookup: rows of an fp8 e4m3 table [*, HEAD_DIM] with ue8m0
// per-32 scales [*, HEAD_DIM / 32], dequantized to bf16. out [NCOLS, HEAD_DIM].
template <typename T, int NCOLS, int HEAD_DIM>
__device__ __forceinline__ void dsv4_engram_lookup_task_impl(
    void const *ids_ptr,
    void const *table_ptr,
    void const *scale_ptr,
    void *out_ptr) {
  int const *__restrict__ ids = static_cast<int const *>(ids_ptr);
  __nv_fp8_e4m3 const *__restrict__ table = static_cast<__nv_fp8_e4m3 const *>(table_ptr);
  uint8_t const *__restrict__ scale = static_cast<uint8_t const *>(scale_ptr);
  T *__restrict__ d_out = static_cast<T *>(out_ptr);
  for (int i = threadIdx.x; i < NCOLS * HEAD_DIM; i += NUM_THREADS) {
    int const c = i / HEAD_DIM, d = i % HEAD_DIM;
    long long row = ids[c];
    float v = float(table[row * HEAD_DIM + d]);
    // e8m0: the byte is the biased exponent.
    float s = __uint_as_float(uint32_t(scale[row * (HEAD_DIM / 32) + d / 32]) << 23);
    d_out[i] = T(v * s);
  }
} // dsv4_engram_lookup_task_impl

// Engram gate (Engram.forward): kv = [key (HC x DIM) | value (DIM)] bf16;
// per copy j: dot = sum(h_j * w_j * key_j) * rsqrt(mean h_j^2 + eps) *
// rsqrt(mean key_j^2 + eps) / sqrt(DIM); gate = sigmoid(signed sqrt(dot));
// out_j = bf16(h_j + gate * value). One task per token.
template <typename T, int HC, int DIM>
__device__ __forceinline__ void dsv4_engram_gate_task_impl(void const *h_ptr,
                                                           void const *kv_ptr,
                                                           void const *w_ptr,
                                                           void *out_ptr,
                                                           float eps,
                                                           float inv_sqrt_dim) {
  T const *__restrict__ d_h = static_cast<T const *>(h_ptr);
  T const *__restrict__ d_kv = static_cast<T const *>(kv_ptr);
  float const *__restrict__ d_w = static_cast<float const *>(w_ptr);
  T *__restrict__ d_out = static_cast<T *>(out_ptr);
  extern __shared__ char smem[];
  float *red = reinterpret_cast<float *>(smem);
  for (int j = 0; j < HC; ++j) {
    float hh = 0.0f, kk = 0.0f, hk = 0.0f;
    for (int d = threadIdx.x; d < DIM; d += NUM_THREADS) {
      float h = float(d_h[j * DIM + d]);
      float k = float(d_kv[j * DIM + d]);
      hh += h * h;
      kk += k * k;
      hk += h * d_w[j * DIM + d] * k;
    }
    hh = dsv4::block_sum(hh, red);
    kk = dsv4::block_sum(kk, red);
    hk = dsv4::block_sum(hk, red);
    float rstd = rsqrtf(hh / float(DIM) + eps) * rsqrtf(kk / float(DIM) + eps);
    float dot = hk * rstd * inv_sqrt_dim;
    float sq = sqrtf(fmaxf(fabsf(dot), 1e-6f));
    float gate = 1.0f / (1.0f + expf(-copysignf(sq, dot)));
    for (int d = threadIdx.x; d < DIM; d += NUM_THREADS) {
      float h = float(d_h[j * DIM + d]);
      float v = float(d_kv[HC * DIM + d]);
      d_out[j * DIM + d] = T(h + gate * v);
    }
  }
} // dsv4_engram_gate_task_impl

// Argmax over fp32 logits [V] (lowest index on ties, as torch.argmax), written
// as int64 to out[0].
template <int V>
__device__ __forceinline__ void dsv4_argmax_task_impl(void const *logits_ptr,
                                                      void *out_ptr) {
  constexpr int NUM_WARPS = NUM_THREADS / 32;
  float const *__restrict__ d_l = static_cast<float const *>(logits_ptr);
  extern __shared__ char smem[];
  float *s_v = reinterpret_cast<float *>(smem);
  int *s_i = reinterpret_cast<int *>(s_v + NUM_WARPS);
  float best = -INFINITY;
  int best_i = V;
  for (int i = threadIdx.x; i < V; i += NUM_THREADS) {
    float v = d_l[i];
    if (v > best) {
      best = v;
      best_i = i;
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off /= 2) {
    float ov = __shfl_xor_sync(0xffffffff, best, off);
    int oi = __shfl_xor_sync(0xffffffff, best_i, off);
    if (ov > best || (ov == best && oi < best_i)) {
      best = ov;
      best_i = oi;
    }
  }
  if (threadIdx.x % 32 == 0) {
    s_v[threadIdx.x / 32] = best;
    s_i[threadIdx.x / 32] = best_i;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int w = 1; w < NUM_WARPS; ++w) {
      if (s_v[w] > best || (s_v[w] == best && s_i[w] < best_i)) {
        best = s_v[w];
        best_i = s_i[w];
      }
    }
    static_cast<long long *>(out_ptr)[0] = best_i;
  }
} // dsv4_argmax_task_impl

} // namespace kernel
