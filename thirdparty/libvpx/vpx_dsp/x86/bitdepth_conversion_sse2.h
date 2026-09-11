/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#ifndef VPX_VPX_DSP_X86_BITDEPTH_CONVERSION_SSE2_H_
#define VPX_VPX_DSP_X86_BITDEPTH_CONVERSION_SSE2_H_

#include <xmmintrin.h>

#include "./vpx_config.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/vpx_dsp_common.h"

// Load 8 16 bit values. If the source is 32 bits then pack down with
// saturation.
static INLINE __m128i load_tran_low(const tran_low_t *a) {
#if CONFIG_VP9_HIGHBITDEPTH
  const __m128i a_low = _mm_load_si128((const __m128i *)a);
  return _mm_packs_epi32(a_low, *(const __m128i *)(a + 4));
#else
  return _mm_load_si128((const __m128i *)a);
#endif
}

// Store 8 16 bit values. If the destination is 32 bits then sign extend the
// values by multiplying by 1.
static INLINE void store_tran_low(__m128i a, tran_low_t *b) {
#if CONFIG_VP9_HIGHBITDEPTH
  const __m128i one = _mm_set1_epi16(1);
  const __m128i a_hi = _mm_mulhi_epi16(a, one);
  const __m128i a_lo = _mm_mullo_epi16(a, one);
  const __m128i a_1 = _mm_unpacklo_epi16(a_lo, a_hi);
  const __m128i a_2 = _mm_unpackhi_epi16(a_lo, a_hi);
  _mm_store_si128((__m128i *)(b), a_1);
  _mm_store_si128((__m128i *)(b + 4), a_2);
#else
  _mm_store_si128((__m128i *)(b), a);
#endif
}

// Zero fill 8 positions in the output buffer.
static INLINE void store_zero_tran_low(tran_low_t *a) {
  const __m128i zero = _mm_setzero_si128();
#if CONFIG_VP9_HIGHBITDEPTH
  _mm_store_si128((__m128i *)(a), zero);
  _mm_store_si128((__m128i *)(a + 4), zero);
#else
  _mm_store_si128((__m128i *)(a), zero);
#endif
}
#endif  // VPX_VPX_DSP_X86_BITDEPTH_CONVERSION_SSE2_H_
