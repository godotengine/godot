// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

/* Make precision match SSE, at the cost of some performance */
#if !defined(__aarch64__)
#  define SSE2NEON_PRECISE_DIV 1
#  define SSE2NEON_PRECISE_SQRT 1
#endif

#include "sse2neon.h"

__forceinline __m128 _mm_fmsub_ps(__m128 a, __m128 b, __m128 c) {
   __m128 neg_c = vreinterpretq_m128_f32(vnegq_f32(vreinterpretq_f32_m128(c)));
  return _mm_fmadd_ps(a, b, neg_c);
}

__forceinline __m128 _mm_fnmadd_ps(__m128 a, __m128 b, __m128 c) {
#if defined(__aarch64__)
    return vreinterpretq_m128_f32(vfmsq_f32(vreinterpretq_f32_m128(c),
                                            vreinterpretq_f32_m128(b),
                                            vreinterpretq_f32_m128(a)));
#else
    return _mm_sub_ps(c, _mm_mul_ps(a, b));
#endif
}

__forceinline __m128 _mm_fnmsub_ps(__m128 a, __m128 b, __m128 c) {
  return vreinterpretq_m128_f32(vnegq_f32(vreinterpretq_f32_m128(_mm_fmadd_ps(a,b,c))));
}


/* Dummy defines for floating point control */
#define _MM_MASK_MASK 0x1f80
#define _MM_MASK_DIV_ZERO 0x200
#define _MM_FLUSH_ZERO_ON 0x8000
#define _MM_MASK_DENORM 0x100
#define _MM_SET_EXCEPTION_MASK(x)
#define _MM_SET_FLUSH_ZERO_MODE(x)

__forceinline int _mm_getcsr()
{
  return 0;
}

__forceinline void _mm_mfence()
{
  __sync_synchronize();
}
