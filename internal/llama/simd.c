#include "simd.h"
#include <stdint.h>

/* --- Detecção de Arquitetura e Inclusão de Headers --- */
#if defined(__x86_64__) || defined(_M_X64)
    #pragma GCC optimize("O3")
    #pragma GCC target("avx2,fma,f16c")
    #include <immintrin.h>
    #define USE_AVX2
#elif defined(__ARM_NEON) || defined(__aarch64__)
    #include <arm_neon.h>
    #define USE_NEON
#endif

/* --- Estruturas e Auxiliares --- */
typedef struct {
    uint16_t d;       // escala float16
    int8_t qs[32];    // 32 pesos quantizados
} __attribute__((packed)) block_q8_0;

inline static float decode_f16(uint16_t h) {
    uint32_t w = (h & 0x7fff) << 13;
    uint32_t exp = h & 0x7c00;
    w += (h & 0x8000) << 16;
    if (exp == 0x7c00) w += 0x3fc00000;
    else if (exp != 0) w += 0x38000000;
    union { uint32_t u; float f; } u;
    u.u = w;
    return u.f;
}

/* --- 1. Matrix-Vector Multiplication (Quantizada Q8_0) --- */
void matvec_q8_c(const void* w_data, const float* x_scales, const int8_t* x_qs, float* out, int start_row, int end_row, int cols) {
    const block_q8_0* w_blocks = (const block_q8_0*)w_data;
    int nb = cols / 32;

    for (int r = start_row; r < end_row; r++) {
        const block_q8_0* row_w = w_blocks + (r * nb);
        float row_sum = 0.0f;

#if defined(USE_AVX2)
        __m256 v_sum = _mm256_setzero_ps();
        for (int b = 0; b < nb; b++) {
            float d = decode_f16(row_w[b].d) * x_scales[b];
            __m256 v_scale = _mm256_set1_ps(d);

            __m256i vw = _mm256_loadu_si256((const __m256i*)row_w[b].qs);
            __m256i vx = _mm256_loadu_si256((const __m256i*)(x_qs + (b * 32)));

            __m256i v_low_w = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vw));
            __m256i v_low_x = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vx));
            __m256i v_dot_low = _mm256_madd_epi16(v_low_w, v_low_x);

            __m256i v_high_w = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vw, 1));
            __m256i v_high_x = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vx, 1));
            __m256i v_dot_high = _mm256_madd_epi16(v_high_w, v_high_x);

            v_sum = _mm256_fmadd_ps(_mm256_cvtepi32_ps(v_dot_low), v_scale, v_sum);
            v_sum = _mm256_fmadd_ps(_mm256_cvtepi32_ps(v_dot_high), v_scale, v_sum);
        }
        // Redução horizontal
        __m128 v128 = _mm_add_ps(_mm256_extractf128_ps(v_sum, 1), _mm256_castps256_ps128(v_sum));
        v128 = _mm_add_ps(v128, _mm_movehl_ps(v128, v128));
        v128 = _mm_add_ps(v128, _mm_shuffle_ps(v128, v128, 1));
        row_sum = _mm_cvtss_f32(v128);

#elif defined(USE_NEON)
        // Simpler, safer NEON implementation: widen int8 to int16 using NEON
        // then store and compute dot in scalar to avoid subtle intrinsics bugs.
        for (int b = 0; b < nb; b++) {
            float d = decode_f16(row_w[b].d) * x_scales[b];
            const int8_t* wptr = row_w[b].qs;
            const int8_t* xptr = x_qs + (b * 32);

            int16_t wtmp[32];
            int16_t xtmp[32];

            // process two chunks of 16
            for (int off = 0; off < 32; off += 16) {
                int8x16_t vw = vld1q_s8(wptr + off);
                int8x16_t vx = vld1q_s8(xptr + off);

                int16x8_t w_lo = vmovl_s8(vget_low_s8(vw));
                int16x8_t w_hi = vmovl_s8(vget_high_s8(vw));
                int16x8_t x_lo = vmovl_s8(vget_low_s8(vx));
                int16x8_t x_hi = vmovl_s8(vget_high_s8(vx));

                vst1q_s16(&wtmp[off], w_lo);
                vst1q_s16(&wtmp[off + 8], w_hi);
                vst1q_s16(&xtmp[off], x_lo);
                vst1q_s16(&xtmp[off + 8], x_hi);
            }

            float acc = 0.0f;
            for (int i = 0; i < 32; i++) {
                acc += (float)wtmp[i] * (float)xtmp[i];
            }
            row_sum += d * acc;
        }
#else // Fallback Escalar para ARM/Genérico
        for (int b = 0; b < nb; b++) {
            float d = decode_f16(row_w[b].d) * x_scales[b];
            for (int i = 0; i < 32; i++) {
                row_sum += d * row_w[b].qs[i] * x_qs[b * 32 + i];
            }
        }
#endif
    out[r] = row_sum;
    }
}

/* --- 2. Matrix-Vector Multiplication (Float32) --- */
void matvec_f32_c(const float* w_data, const float* x_data, float* out, int start_row, int end_row, int cols) {
    for (int r = start_row; r < end_row; r++) {
        out[r] = dot_product(w_data + (r * cols), x_data, cols);
    }
}

/* --- 3. Dot Product (Float32) --- */
float dot_product(const float* a, const float* b, int n) {
    float sum = 0.0f;
    int i = 0;

#if defined(USE_AVX2)
    __m256 acc = _mm256_setzero_ps();
    for (; i <= n - 8; i += 8) {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(&a[i]), _mm256_loadu_ps(&b[i]), acc);
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, acc);
    for (int j = 0; j < 8; j++) sum += tmp[j];

#elif defined(USE_NEON)
    float32x4_t acc = vdupq_n_f32(0);
    for (; i <= n - 4; i += 4) {
        acc = vmlaq_f32(acc, vld1q_f32(&a[i]), vld1q_f32(&b[i]));
    }
    sum = vaddvq_f32(acc); // Redução NEON mais moderna
#endif

    // Resto (Tail processing)
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}