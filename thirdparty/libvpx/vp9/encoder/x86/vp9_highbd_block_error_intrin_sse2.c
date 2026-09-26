/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <emmintrin.h>
#include <stdio.h>

#include "./vp9_rtcd.h"
#include "vp9/common/vp9_common.h"

int64_t vp9_highbd_block_error_sse2(const tran_low_t *coeff,
                                    const tran_low_t *dqcoeff,
                                    intptr_t block_size, int64_t *ssz, int bd) {
  int i, j, test;
  uint32_t temp[4];
  __m128i max, min, cmp0, cmp1, cmp2, cmp3;
  int64_t error = 0, sqcoeff = 0;
  const int shift = 2 * (bd - 8);
  const int rounding = shift > 0 ? 1 << (shift - 1) : 0;

  for (i = 0; i < block_size; i += 8) {
    // Load the data into xmm registers
    __m128i mm_coeff = _mm_load_si128((const __m128i *)(coeff + i));
    __m128i mm_coeff2 = _mm_load_si128((const __m128i *)(coeff + i + 4));
    __m128i mm_dqcoeff = _mm_load_si128((const __m128i *)(dqcoeff + i));
    __m128i mm_dqcoeff2 = _mm_load_si128((const __m128i *)(dqcoeff + i + 4));
    // Check if any values require more than 15 bit
    max = _mm_set1_epi32(0x3fff);
    min = _mm_set1_epi32((int32_t)0xffffc000);
    cmp0 = _mm_xor_si128(_mm_cmpgt_epi32(mm_coeff, max),
                         _mm_cmplt_epi32(mm_coeff, min));
    cmp1 = _mm_xor_si128(_mm_cmpgt_epi32(mm_coeff2, max),
                         _mm_cmplt_epi32(mm_coeff2, min));
    cmp2 = _mm_xor_si128(_mm_cmpgt_epi32(mm_dqcoeff, max),
                         _mm_cmplt_epi32(mm_dqcoeff, min));
    cmp3 = _mm_xor_si128(_mm_cmpgt_epi32(mm_dqcoeff2, max),
                         _mm_cmplt_epi32(mm_dqcoeff2, min));
    test = _mm_movemask_epi8(
        _mm_or_si128(_mm_or_si128(cmp0, cmp1), _mm_or_si128(cmp2, cmp3)));

    if (!test) {
      __m128i mm_diff, error_sse2, sqcoeff_sse2;
      mm_coeff = _mm_packs_epi32(mm_coeff, mm_coeff2);
      mm_dqcoeff = _mm_packs_epi32(mm_dqcoeff, mm_dqcoeff2);
      mm_diff = _mm_sub_epi16(mm_coeff, mm_dqcoeff);
      error_sse2 = _mm_madd_epi16(mm_diff, mm_diff);
      sqcoeff_sse2 = _mm_madd_epi16(mm_coeff, mm_coeff);
      _mm_storeu_si128((__m128i *)temp, error_sse2);
      error = error + temp[0] + temp[1] + temp[2] + temp[3];
      _mm_storeu_si128((__m128i *)temp, sqcoeff_sse2);
      sqcoeff += temp[0] + temp[1] + temp[2] + temp[3];
    } else {
      for (j = 0; j < 8; j++) {
        const int64_t diff = coeff[i + j] - dqcoeff[i + j];
        error += diff * diff;
        sqcoeff += (int64_t)coeff[i + j] * (int64_t)coeff[i + j];
      }
    }
  }
  assert(error >= 0 && sqcoeff >= 0);
  error = (error + rounding) >> shift;
  sqcoeff = (sqcoeff + rounding) >> shift;

  *ssz = sqcoeff;
  return error;
}
