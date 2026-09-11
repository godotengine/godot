/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#include <immintrin.h>  // AVX512
#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"

static INLINE void sad64xhx4d_avx512(const uint8_t *src_ptr, int src_stride,
                                     const uint8_t *const ref_array[4],
                                     int ref_stride, int h,
                                     uint32_t sad_array[4]) {
  __m512i src_reg, ref0_reg, ref1_reg, ref2_reg, ref3_reg;
  __m512i sum_ref0, sum_ref1, sum_ref2, sum_ref3;
  __m512i sum_mlow, sum_mhigh;
  int i;
  const uint8_t *ref0, *ref1, *ref2, *ref3;

  ref0 = ref_array[0];
  ref1 = ref_array[1];
  ref2 = ref_array[2];
  ref3 = ref_array[3];
  sum_ref0 = _mm512_set1_epi16(0);
  sum_ref1 = _mm512_set1_epi16(0);
  sum_ref2 = _mm512_set1_epi16(0);
  sum_ref3 = _mm512_set1_epi16(0);
  for (i = 0; i < h; i++) {
    // load src and all ref[]
    src_reg = _mm512_loadu_si512((const __m512i *)src_ptr);
    ref0_reg = _mm512_loadu_si512((const __m512i *)ref0);
    ref1_reg = _mm512_loadu_si512((const __m512i *)ref1);
    ref2_reg = _mm512_loadu_si512((const __m512i *)ref2);
    ref3_reg = _mm512_loadu_si512((const __m512i *)ref3);
    // sum of the absolute differences between every ref[] to src
    ref0_reg = _mm512_sad_epu8(ref0_reg, src_reg);
    ref1_reg = _mm512_sad_epu8(ref1_reg, src_reg);
    ref2_reg = _mm512_sad_epu8(ref2_reg, src_reg);
    ref3_reg = _mm512_sad_epu8(ref3_reg, src_reg);
    // sum every ref[]
    sum_ref0 = _mm512_add_epi32(sum_ref0, ref0_reg);
    sum_ref1 = _mm512_add_epi32(sum_ref1, ref1_reg);
    sum_ref2 = _mm512_add_epi32(sum_ref2, ref2_reg);
    sum_ref3 = _mm512_add_epi32(sum_ref3, ref3_reg);

    src_ptr += src_stride;
    ref0 += ref_stride;
    ref1 += ref_stride;
    ref2 += ref_stride;
    ref3 += ref_stride;
  }
  {
    __m256i sum256;
    __m128i sum128;
    // in sum_ref[] the result is saved in the first 4 bytes
    // the other 4 bytes are zeroed.
    // sum_ref1 and sum_ref3 are shifted left by 4 bytes
    sum_ref1 = _mm512_bslli_epi128(sum_ref1, 4);
    sum_ref3 = _mm512_bslli_epi128(sum_ref3, 4);

    // merge sum_ref0 and sum_ref1 also sum_ref2 and sum_ref3
    sum_ref0 = _mm512_or_si512(sum_ref0, sum_ref1);
    sum_ref2 = _mm512_or_si512(sum_ref2, sum_ref3);

    // merge every 64 bit from each sum_ref[]
    sum_mlow = _mm512_unpacklo_epi64(sum_ref0, sum_ref2);
    sum_mhigh = _mm512_unpackhi_epi64(sum_ref0, sum_ref2);

    // add the low 64 bit to the high 64 bit
    sum_mlow = _mm512_add_epi32(sum_mlow, sum_mhigh);

    // add the low 128 bit to the high 128 bit
    sum256 = _mm256_add_epi32(_mm512_castsi512_si256(sum_mlow),
                              _mm512_extracti32x8_epi32(sum_mlow, 1));
    sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum256),
                           _mm256_extractf128_si256(sum256, 1));

    _mm_storeu_si128((__m128i *)(sad_array), sum128);
  }
}

void vpx_sad64x64x4d_avx512(const uint8_t *src, int src_stride,
                            const uint8_t *const ref_array[4], int ref_stride,
                            uint32_t sad_array[4]) {
  sad64xhx4d_avx512(src, src_stride, ref_array, ref_stride, 64, sad_array);
}

#define SADS64_H(h)                                                          \
  void vpx_sad_skip_64x##h##x4d_avx512(                                      \
      const uint8_t *src, int src_stride, const uint8_t *const ref_array[4], \
      int ref_stride, uint32_t sad_array[4]) {                               \
    sad64xhx4d_avx512(src, 2 * src_stride, ref_array, 2 * ref_stride,        \
                      ((h) >> 1), sad_array);                                \
    sad_array[0] <<= 1;                                                      \
    sad_array[1] <<= 1;                                                      \
    sad_array[2] <<= 1;                                                      \
    sad_array[3] <<= 1;                                                      \
  }

SADS64_H(64)
SADS64_H(32)
