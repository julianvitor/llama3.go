#include "simd.h"
#include <stdint.h>

#if defined(__x86_64__) || defined(_M_X64)
    #pragma GCC optimize("O3")
    #pragma GCC target("avx2,fma,f16c")
    #include <immintrin.h>
#elif defined(__ARM_NEON) || defined(__aarch64__)
    #include <arm_neon.h>
#endif

// Struct de 34 bytes para os pesos (W)
typedef struct {
    uint16_t d;       // float16
    int8_t qs[32];    // 32 pesos
} __attribute__((packed)) block_q8_0;

// Conversão manual de float16 para float32
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

void matvec_q8_c(const void* w_data, const float* x_scales, const int8_t* x_qs, float* out, int start_row, int end_row, int cols) {
    const block_q8_0* w_blocks = (const block_q8_0*)w_data;
    int nb = cols / 32;

    for (int r = start_row; r < end_row; r++) {
        const block_q8_0* row_w = w_blocks + (r * nb);
        __m256 v_sum = _mm256_setzero_ps();

        for (int b = 0; b < nb; b++) {
            // Escala combinada: dw (f16 decodificado) * dx (f32 direto)
            float d = decode_f16(row_w[b].d) * x_scales[b];
            __m256 v_scale = _mm256_set1_ps(d);

            // Carrega os 32 pesos de W e os 32 de X
            __m256i vw = _mm256_loadu_si256((const __m256i*)row_w[b].qs);
            __m256i vx = _mm256_loadu_si256((const __m256i*)(x_qs + (b * 32)));

            // Dot product int8 -> int16 -> int32
            __m256i v_low_w = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vw));
            __m256i v_low_x = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vx));
            __m256i v_dot_low = _mm256_madd_epi16(v_low_w, v_low_x);

            __m256i v_high_w = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vw, 1));
            __m256i v_high_x = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vx, 1));
            __m256i v_dot_high = _mm256_madd_epi16(v_high_w, v_high_x);

            // Converte para float e acumula com FMA
            v_sum = _mm256_fmadd_ps(_mm256_cvtepi32_ps(v_dot_low), v_scale, v_sum);
            v_sum = _mm256_fmadd_ps(_mm256_cvtepi32_ps(v_dot_high), v_scale, v_sum);
        }

        // Redução horizontal eficiente
        __m128 v128 = _mm_add_ps(_mm256_extractf128_ps(v_sum, 1), _mm256_castps256_ps128(v_sum));
        v128 = _mm_add_ps(v128, _mm_movehl_ps(v128, v128));
        v128 = _mm_add_ps(v128, _mm_shuffle_ps(v128, v128, 1));
        out[r] = _mm_cvtss_f32(v128);
    }
}

void matvec_f32_c(const float* w_data, const float* x_data, float* out, int start_row, int end_row, int cols) {
    for (int r = start_row; r < end_row; r++) {
        const float* row_w = w_data + (r * cols);
        float sum = 0.0f;
        int i = 0;

#if defined(__AVX2__)
        __m256 v_sum = _mm256_setzero_ps();
        for (; i <= cols - 8; i += 8) {
            v_sum = _mm256_fmadd_ps(_mm256_loadu_ps(&row_w[i]), _mm256_loadu_ps(&x_data[i]), v_sum);
        }
        float tmp[8];
        _mm256_storeu_ps(tmp, v_sum);
        for(int j=0; j<8; j++) sum += tmp[j];
#endif
        for (; i < cols; i++) {
            sum += row_w[i] * x_data[i];
        }
        out[r] = sum;
    }
}