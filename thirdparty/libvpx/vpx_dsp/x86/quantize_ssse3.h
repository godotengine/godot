/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_X86_QUANTIZE_SSSE3_H_
#define VPX_VPX_DSP_X86_QUANTIZE_SSSE3_H_

#include <emmintrin.h>

#include "./vpx_config.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/x86/quantize_sse2.h"

static INLINE void calculate_dqcoeff_and_store_32x32(const __m128i qcoeff,
                                                     const __m128i dequant,
                                                     const __m128i zero,
                                                     tran_low_t *dqcoeff) {
  // Un-sign to bias rounding like C.
  const __m128i coeff = _mm_abs_epi16(qcoeff);

  const __m128i sign_0 = _mm_unpacklo_epi16(zero, qcoeff);
  const __m128i sign_1 = _mm_unpackhi_epi16(zero, qcoeff);

  const __m128i low = _mm_mullo_epi16(coeff, dequant);
  const __m128i high = _mm_mulhi_epi16(coeff, dequant);
  __m128i dqcoeff32_0 = _mm_unpacklo_epi16(low, high);
  __m128i dqcoeff32_1 = _mm_unpackhi_epi16(low, high);

  // "Divide" by 2.
  dqcoeff32_0 = _mm_srli_epi32(dqcoeff32_0, 1);
  dqcoeff32_1 = _mm_srli_epi32(dqcoeff32_1, 1);

  dqcoeff32_0 = _mm_sign_epi32(dqcoeff32_0, sign_0);
  dqcoeff32_1 = _mm_sign_epi32(dqcoeff32_1, sign_1);

#if CONFIG_VP9_HIGHBITDEPTH
  _mm_store_si128((__m128i *)(dqcoeff), dqcoeff32_0);
  _mm_store_si128((__m128i *)(dqcoeff + 4), dqcoeff32_1);
#else
  _mm_store_si128((__m128i *)(dqcoeff),
                  _mm_packs_epi32(dqcoeff32_0, dqcoeff32_1));
#endif  // CONFIG_VP9_HIGHBITDEPTH
}

#endif  // VPX_VPX_DSP_X86_QUANTIZE_SSSE3_H_
