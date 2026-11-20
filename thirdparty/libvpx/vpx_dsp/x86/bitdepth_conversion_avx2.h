/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#ifndef VPX_VPX_DSP_X86_BITDEPTH_CONVERSION_AVX2_H_
#define VPX_VPX_DSP_X86_BITDEPTH_CONVERSION_AVX2_H_

#include <immintrin.h>

#include "./vpx_config.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/vpx_dsp_common.h"

// Load 16 16 bit values. If the source is 32 bits then pack down with
// saturation.
static INLINE __m256i load_tran_low(const tran_low_t *a) {
#if CONFIG_VP9_HIGHBITDEPTH
  const __m256i a_low = _mm256_loadu_si256((const __m256i *)a);
  const __m256i a_high = _mm256_loadu_si256((const __m256i *)(a + 8));
  return _mm256_packs_epi32(a_low, a_high);
#else
  return _mm256_loadu_si256((const __m256i *)a);
#endif
}

static INLINE void store_tran_low(__m256i a, tran_low_t *b) {
#if CONFIG_VP9_HIGHBITDEPTH
  const __m256i one = _mm256_set1_epi16(1);
  const __m256i a_hi = _mm256_mulhi_epi16(a, one);
  const __m256i a_lo = _mm256_mullo_epi16(a, one);
  const __m256i a_1 = _mm256_unpacklo_epi16(a_lo, a_hi);
  const __m256i a_2 = _mm256_unpackhi_epi16(a_lo, a_hi);
  _mm256_storeu_si256((__m256i *)b, a_1);
  _mm256_storeu_si256((__m256i *)(b + 8), a_2);
#else
  _mm256_storeu_si256((__m256i *)b, a);
#endif
}
#endif  // VPX_VPX_DSP_X86_BITDEPTH_CONVERSION_AVX2_H_
