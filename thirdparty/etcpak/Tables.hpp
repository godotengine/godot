#ifndef __TABLES_HPP__
#define __TABLES_HPP__

#include <stdint.h>

#ifdef __AVX2__
#  include <immintrin.h>
#endif
#ifdef __SSE4_1__
#  include <smmintrin.h>
#endif
#ifdef __ARM_NEON
#  include <arm_neon.h>
#endif

extern const int32_t g_table[8][4];
extern const int64_t g_table256[8][4];

extern const uint32_t g_id[4][16];

extern const uint32_t g_avg2[16];

extern const uint32_t g_flags[64];

extern const int32_t g_alpha[16][8];
extern const int32_t g_alpha11Mul[16];
extern const int32_t g_alphaRange[16];

#ifdef __SSE4_1__
extern const __m128i g_table_SIMD[2];
extern const __m128i g_table128_SIMD[2];
extern const __m128i g_table256_SIMD[4];

extern const __m128i g_alpha_SIMD[16];
extern const __m128i g_alphaRange_SIMD;
#endif

#ifdef __AVX2__
extern const __m256i g_alpha_AVX[8];
extern const __m256i g_alphaRange_AVX;
#endif

#ifdef __ARM_NEON
extern const int16x8_t g_table128_NEON[2];
extern const int32x4_t g_table256_NEON[4];
extern const int16x8_t g_alpha_NEON[16];
extern const int16x8_t g_alphaRange_NEON;
#endif

#endif
