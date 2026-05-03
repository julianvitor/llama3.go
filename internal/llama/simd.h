// simd.h
#ifndef SIMD_H
#define SIMD_H

#include <stdint.h>

float dot_product(const float* a, const float* b, int n);
void matvec_q8_c(const void* w_data, const float* x_scales, const int8_t* x_qs, float* out, int start_row, int end_row, int cols);
void matvec_f32_c(const float* w_data, const float* x_data, float* out, int start_row, int end_row, int cols);

#endif