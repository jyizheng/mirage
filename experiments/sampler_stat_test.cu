// P1 sampler statistical validation: drives the EXACT device functions of
// tasks/common/sampling.cuh (position-keyed Gumbel-Max) over synthetic
// logits and checks (1) the empirical distribution matches softmax
// (chi-square) and (2) draws at different positions are independent
// (lag-1 transition matrix chi-square).
//
// Build: nvcc -std=c++17 -arch=sm_100a -DMPK_TARGET_CC=100 \
//   -DMIRAGE_BACKEND_USE_CUDA -I include -I include/mirage/persistent_kernel \
//   -I include/mirage/persistent_kernel/tasks/common \
//   sampler_stat_test.cu -o /tmp/sampler_stat_test
#include <cmath>
#include <cstdio>
#include <vector>

#include "sampling.cuh"

constexpr int VOCAB = 8;
constexpr int N = 1'000'000;   // positions
constexpr uint64_t SEED = 20260723;

__constant__ float c_logits[VOCAB];

__global__ void draw_kernel(int *out) {
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= N) {
    return;
  }
  // replicate the poskeyed scheme: offset = rid*L+pos+1 (rid=0), subseq=elem
#ifdef OLD_SCHEME
  // upstream scheme: constant offset, subsequence ignores position ->
  // every position reuses the same noise vector
  uint64_t philox_offset = 0;
  (void)pos;
#else
  uint64_t philox_offset = (uint64_t)pos + 1;
#endif
  float best = -1e30f;
  int arg = -1;
  for (int j = 0; j < VOCAB; j++) {
    auto g = kernel::GenerateSamplingGumbelNoise<float, 1>(
        SEED, philox_offset, (uint64_t)j);
    float v = c_logits[j] + g[0];
    if (v > best) {
      best = v;
      arg = j;
    }
  }
  out[pos] = arg;
}

int main() {
  float h_logits[VOCAB] = {2.0f, 1.5f, 1.0f, 0.5f, 0.0f, -0.5f, -1.0f, -2.0f};
  cudaMemcpyToSymbol(c_logits, h_logits, sizeof(h_logits));

  int *d_out;
  cudaMalloc(&d_out, N * sizeof(int));
  draw_kernel<<<(N + 255) / 256, 256>>>(d_out);
  std::vector<int> out(N);
  cudaMemcpy(out.data(), d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

  // expected softmax
  double Z = 0, p[VOCAB];
  for (int j = 0; j < VOCAB; j++) {
    Z += exp((double)h_logits[j]);
  }
  for (int j = 0; j < VOCAB; j++) {
    p[j] = exp((double)h_logits[j]) / Z;
  }

  // (1) marginal chi-square
  long cnt[VOCAB] = {0};
  for (int i = 0; i < N; i++) {
    cnt[out[i]]++;
  }
  double chi2 = 0;
  for (int j = 0; j < VOCAB; j++) {
    double e = p[j] * N;
    chi2 += (cnt[j] - e) * (cnt[j] - e) / e;
  }
  printf("marginal chi2=%.2f dof=%d (95%% crit=14.07)\n", chi2, VOCAB - 1);
  for (int j = 0; j < VOCAB; j++) {
    printf("  tok %d: obs %.5f exp %.5f\n", j, (double)cnt[j] / N, p[j]);
  }

  // (2) lag-1 independence: transition counts vs product of marginals
  static long t[VOCAB][VOCAB] = {{0}};
  for (int i = 0; i + 1 < N; i++) {
    t[out[i]][out[i + 1]]++;
  }
  double chi2t = 0;
  int dof = 0;
  for (int a = 0; a < VOCAB; a++) {
    for (int b = 0; b < VOCAB; b++) {
      double e = p[a] * p[b] * (N - 1);
      if (e < 5) {
        continue;
      }
      chi2t += (t[a][b] - e) * (t[a][b] - e) / e;
      dof++;
    }
  }
  dof -= (VOCAB - 1) * 2 + 1;
  double crit = dof + 1.645 * sqrt(2.0 * dof);
  printf("lag-1 independence chi2=%.2f dof=%d (95%% crit~=%.1f)\n",
         chi2t, dof, crit);
  bool ok = chi2 < 14.07 * 1.5 && chi2t < crit * 1.5;
  printf("%s\n", ok ? "SAMPLER STATS: PASS" : "SAMPLER STATS: FAIL");
  return ok ? 0 : 1;
}
