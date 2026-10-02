/* adler32_ssse3_p.h -- adler32 ssse3 utility functions
 * Copyright (C) 2022 Adam Stylinski
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#ifndef ADLER32_SSSE3_P_H_
#define ADLER32_SSSE3_P_H_

#ifdef X86_SSSE3

#include <immintrin.h>
#include <stdint.h>

static inline uint32_t partial_hsum(__m128i x) {
    __m128i second_int = _mm_srli_si128(x, 8);
    __m128i sum = _mm_add_epi32(x, second_int);
    return _mm_cvtsi128_si32(sum);
}

static inline uint32_t hsum(__m128i x) {
    __m128i sum1 = _mm_unpackhi_epi64(x, x);
    __m128i sum2 = _mm_add_epi32(x, sum1);
    __m128i sum3 = _mm_shuffle_epi32(sum2, 0x01);
    __m128i sum4 = _mm_add_epi32(sum2, sum3);
    return _mm_cvtsi128_si32(sum4);
}
#endif

#endif
