#ifndef __TABLES_HPP__
#define __TABLES_HPP__

#include <stdint.h>

#ifdef __SSE4_1__
#include <smmintrin.h>
#endif

extern const int32_t g_table[8][4];
extern const int64_t g_table256[8][4];

extern const uint32_t g_id[4][16];

extern const uint32_t g_avg2[16];

extern const uint32_t g_flags[64];

extern const int32_t g_alpha[16][8];
extern const int32_t g_alphaRange[16];

#ifdef __SSE4_1__
extern const uint8_t g_flags_AVX2[64];
extern const __m128i g_table_SIMD[2];
extern const __m128i g_table128_SIMD[2];
extern const __m128i g_table256_SIMD[4];

extern const __m128i g_alpha_SIMD[16];
extern const __m128i g_alphaRange_SIMD;
#endif

#endif
