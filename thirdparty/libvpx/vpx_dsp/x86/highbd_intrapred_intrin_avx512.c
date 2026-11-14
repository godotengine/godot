/*
 *  Copyright (c) 2025 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <immintrin.h>

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"

DECLARE_ALIGNED(64, static const uint16_t,
                rshift_1w[32]) = { 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                   12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                   23, 24, 25, 26, 27, 28, 29, 30, 31, 31 };
DECLARE_ALIGNED(64, static const uint16_t,
                rshift_2w[32]) = { 2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                   24, 25, 26, 27, 28, 29, 30, 31, 31, 31 };

static INLINE __m512i avg3_epu16_avx512(const __m512i *x, const __m512i *y,
                                        const __m512i *z) {
  const __m512i one = _mm512_set1_epi16(1);
  const __m512i a = _mm512_avg_epu16(*x, *z);
  const __m512i b =
      _mm512_subs_epu16(a, _mm512_and_si512(_mm512_xor_si512(*x, *z), one));
  return _mm512_avg_epu16(b, *y);
}

void vpx_highbd_d63_predictor_32x32_avx512(uint16_t *dst, ptrdiff_t stride,
                                           const uint16_t *above,
                                           const uint16_t *left, int bd) {
  (void)left;
  (void)bd;

  const __m512i A = _mm512_loadu_si512((const __m512i *)above);

  // B = shift by 1 (2 bytes)
  const __m512i B = _mm512_permutexvar_epi16(*(const __m512i *)rshift_1w, A);

  // C = shift by 2 (4 bytes)
  const __m512i C = _mm512_permutexvar_epi16(*(const __m512i *)rshift_2w, A);

  __m512i avg2 = _mm512_avg_epu16(A, B);
  __m512i avg3 = avg3_epu16_avx512(&A, &B, &C);

  for (int i = 0; i < 30; i += 2) {
    _mm512_storeu_si512((__m512i *)dst, avg2);
    dst += stride;

    _mm512_storeu_si512((__m512i *)dst, avg3);
    dst += stride;

    avg2 = _mm512_permutexvar_epi16(*(const __m512i *)rshift_1w, avg2);
    avg3 = _mm512_permutexvar_epi16(*(const __m512i *)rshift_1w, avg3);
  }

  _mm512_storeu_si512((__m512i *)dst, avg2);
  dst += stride;

  _mm512_storeu_si512((__m512i *)dst, avg3);
}

static INLINE void d207_store_16x32_avx512(uint16_t **dst,
                                           const ptrdiff_t stride,
                                           const __m512i *abcd,
                                           const __m512i *efgh) {
  _mm512_storeu_si512((__m512i *)*dst, *abcd);
  *dst += stride;

  __m512i shift = _mm512_alignr_epi32(*efgh, *abcd, 1);
  _mm512_storeu_si512((__m512i *)*dst, shift);
  *dst += stride;

  shift = _mm512_alignr_epi32(*efgh, *abcd, 2);
  _mm512_storeu_si512((__m512i *)*dst, shift);
  *dst += stride;

  shift = _mm512_alignr_epi32(*efgh, *abcd, 3);
  _mm512_storeu_si512((__m512i *)*dst, shift);
  *dst += stride;

  shift = _mm512_alignr_epi32(*efgh, *abcd, 4);
  _mm512_storeu_si512((__m512i *)*dst, shift);
  *dst += stride;

  shift = _mm512_alignr_epi32(*efgh, *abcd, 5);
  _mm512_storeu_si512((__m512i *)*dst, shift);
  *dst += stride;

  shift = _mm512_alignr_epi32(*efgh, *abcd, 6);
  _mm512_storeu_si512((__m512i *)*dst, shift);
  *dst += stride;

  shift = _mm512_alignr_epi32(*efgh, *abcd, 7);
  _mm512_storeu_si512((__m512i *)*dst, shift);
  *dst += stride;

  shift = _mm512_alignr_epi32(*efgh, *abcd, 8);
  _mm512_storeu_si512((__m512i *)*dst, shift);
  *dst += stride;

  shift = _mm512_alignr_epi32(*efgh, *abcd, 9);
  _mm512_storeu_si512((__m512i *)*dst, shift);
  *dst += stride;

  shift = _mm512_alignr_epi32(*efgh, *abcd, 10);
  _mm512_storeu_si512((__m512i *)*dst, shift);
  *dst += stride;

  shift = _mm512_alignr_epi32(*efgh, *abcd, 11);
  _mm512_storeu_si512((__m512i *)*dst, shift);
  *dst += stride;

  shift = _mm512_alignr_epi32(*efgh, *abcd, 12);
  _mm512_storeu_si512((__m512i *)*dst, shift);
  *dst += stride;

  shift = _mm512_alignr_epi32(*efgh, *abcd, 13);
  _mm512_storeu_si512((__m512i *)*dst, shift);
  *dst += stride;

  shift = _mm512_alignr_epi32(*efgh, *abcd, 14);
  _mm512_storeu_si512((__m512i *)*dst, shift);
  *dst += stride;

  shift = _mm512_alignr_epi32(*efgh, *abcd, 15);
  _mm512_storeu_si512((__m512i *)*dst, shift);
  *dst += stride;
}

void vpx_highbd_d207_predictor_32x32_avx512(uint16_t *dst, ptrdiff_t stride,
                                            const uint16_t *above,
                                            const uint16_t *left, int bd) {
  (void)above;
  (void)bd;

  const __m512i A = _mm512_loadu_si512((const __m512i *)left);
  const __m512i AR = _mm512_set1_epi16(left[31]);

  // B = shift by 1 (2 bytes)
  __m512i B = _mm512_permutexvar_epi16(*(const __m512i *)rshift_1w, A);
  // C = shift by 2 (4 bytes)
  __m512i C = _mm512_alignr_epi32(AR, A, 1);

  __m512i avg2 = _mm512_avg_epu16(A, B);
  __m512i avg3 = avg3_epu16_avx512(&A, &B, &C);

  __m512i out_aceg = _mm512_unpacklo_epi16(avg2, avg3);
  __m512i out_bdfh = _mm512_unpackhi_epi16(avg2, avg3);

  __m512i out_abcd = _mm512_permutex2var_epi64(
      out_aceg, _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0), out_bdfh);
  __m512i out_efgh = _mm512_permutex2var_epi64(
      out_aceg, _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4), out_bdfh);

  d207_store_16x32_avx512(&dst, stride, &out_abcd, &out_efgh);
  d207_store_16x32_avx512(&dst, stride, &out_efgh, &AR);
}
