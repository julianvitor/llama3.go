#include <stdint.h>

#if defined(__AVX2__)
#include <immintrin.h>

float dot_product(const float* a, const float* b, int n) {
    __m256 acc = _mm256_setzero_ps();

    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }

    float tmp[8];
    _mm256_storeu_ps(tmp, acc);

    float sum = 0;
    for (int j = 0; j < 8; j++) sum += tmp[j];

    for (; i < n; i++) {
        sum += a[i] * b[i];
    }

    return sum;
}

#elif defined(__ARM_NEON)
#include <arm_neon.h>

float dot_product(const float* a, const float* b, int n) {
    float32x4_t acc = vdupq_n_f32(0);

    int i = 0;
    for (; i <= n - 4; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        acc = vmlaq_f32(acc, va, vb);
    }

    float tmp[4];
    vst1q_f32(tmp, acc);

    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    for (; i < n; i++) {
        sum += a[i] * b[i];
    }

    return sum;
}

#else

float dot_product(const float* a, const float* b, int n) {
    float sum = 0;

    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }

    return sum;
}

#endif