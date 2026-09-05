// Ferret gate harness: M=16 case instantiations.
#include "gate_common.cuh"

void gate_linear_m16_n64_k2048(void const *x, void const *w, void *o,
                               cudaStream_t s) {
  gate::gate_launch_linear<16, 64, 2048, 2048>(x, w, o, s);
}
void gate_linear_m16_n128_k2048(void const *x, void const *w, void *o,
                                cudaStream_t s) {
  gate::gate_launch_linear<16, 128, 2048, 2048>(x, w, o, s);
}
void gate_linear_m16_n256_k2048(void const *x, void const *w, void *o,
                                cudaStream_t s) {
  gate::gate_launch_linear<16, 256, 2048, 2048>(x, w, o, s);
}
void gate_splitk_partial_m16_ks256(void const *x, void const *w, void *p,
                                   cudaStream_t s) {
  gate::gate_launch_splitk_partial<16, 256, 2048>(x, w, p, s);
}
void gate_splitk_partial_m16_ks768(void const *x, void const *w, void *p,
                                   cudaStream_t s) {
  gate::gate_launch_splitk_partial<16, 768, 6144>(x, w, p, s);
}
void gate_splitk_reduce_m16(void const *p, void const *r, void *o,
                            cudaStream_t s) {
  gate::gate_launch_splitk_reduce<16>(p, r, o, s);
}
