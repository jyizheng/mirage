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
#include "../common/common_header.cuh"
#include "barrier.cuh"
#include "tma_2d.cuh"
#include "tma_3d.cuh"
#include "tma_4d.cuh"
#include <cuda.h>
namespace kernel {
namespace tma {

__device__ static inline void async_proxy_fence() {
  asm volatile("fence.proxy.async.shared::cta;");
}

// Cross-proxy fence for GLOBAL memory: makes async-proxy writes (TMA
// stores / reduce-adds) that have already completed (store_async_wait<0>)
// visible to generic-proxy readers.  Inside the persistent megakernel a
// task's completion is published with a generic-proxy release
// (atom.add.release on the event counter) and there is no kernel-end
// implicit fence, so without this fence a consumer task acquiring the
// event on another SM could read torn output.  Issue it after the final
// store_async_wait<0>() of every task that TMA-stores its outputs.
__device__ static inline void async_proxy_fence_global() {
  asm volatile("fence.proxy.async.global;" ::: "memory");
}

__device__ static inline void store_commit_group() {
  asm volatile("cp.async.bulk.commit_group;");
}

template <int N = 0>
__device__ static inline void store_async_wait() {
  asm volatile("cp.async.bulk.wait_group %0;" : : "n"(N) : "memory");
}

__device__ __forceinline__ void prefetch_tma_descriptor(CUtensorMap const *p) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(p);
  asm volatile("prefetch.tensormap [%0];" ::"l"(gmem_int_desc) : "memory");
}

} // namespace tma
} // namespace kernel
