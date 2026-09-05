// Ferret gate harness: M=8 case instantiations.
#include "gate_common.cuh"

void gate_linear_m8_n64_k2048(void const *x, void const *w, void *o,
                              cudaStream_t s) {
  gate::gate_launch_linear<8, 64, 2048, 2048>(x, w, o, s);
}
void gate_linear_m8_n128_k2048(void const *x, void const *w, void *o,
                               cudaStream_t s) {
  gate::gate_launch_linear<8, 128, 2048, 2048>(x, w, o, s);
}
void gate_linear_m8_n256_k2048(void const *x, void const *w, void *o,
                               cudaStream_t s) {
  gate::gate_launch_linear<8, 256, 2048, 2048>(x, w, o, s);
}
void gate_splitk_partial_m8_ks256(void const *x, void const *w, void *p,
                                  cudaStream_t s) {
  gate::gate_launch_splitk_partial<8, 256, 2048>(x, w, p, s);
}
void gate_splitk_partial_m8_ks768(void const *x, void const *w, void *p,
                                  cudaStream_t s) {
  gate::gate_launch_splitk_partial<8, 768, 6144>(x, w, p, s);
}
void gate_splitk_reduce_m8(void const *p, void const *r, void *o,
                           cudaStream_t s) {
  gate::gate_launch_splitk_reduce<8>(p, r, o, s);
}
