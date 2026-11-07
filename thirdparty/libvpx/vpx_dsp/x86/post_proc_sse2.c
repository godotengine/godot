/*
 *  Copyright (c) 2018 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>
#include <emmintrin.h>

#include <stdio.h>

#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/x86/mem_sse2.h"

extern const int16_t vpx_rv[];

void vpx_mbpost_proc_down_sse2(unsigned char *dst, int pitch, int rows,
                               int cols, int flimit) {
  int col;
  const __m128i zero = _mm_setzero_si128();
  const __m128i f = _mm_set1_epi32(flimit);
  DECLARE_ALIGNED(16, int16_t, above_context[8 * 8]);

  // 8 columns are processed at a time.
  // If rows is less than 8 the bottom border extension fails.
  assert(cols % 8 == 0);
  assert(rows >= 8);

  for (col = 0; col < cols; col += 8) {
    int row, i;
    __m128i s = _mm_loadl_epi64((__m128i *)dst);
    __m128i sum, sumsq_0, sumsq_1;
    __m128i tmp_0, tmp_1;
    __m128i below_context = _mm_setzero_si128();

    s = _mm_unpacklo_epi8(s, zero);

    for (i = 0; i < 8; ++i) {
      _mm_store_si128((__m128i *)above_context + i, s);
    }

    // sum *= 9
    sum = _mm_slli_epi16(s, 3);
    sum = _mm_add_epi16(s, sum);

    // sum^2 * 9 == (sum * 9) * sum
    tmp_0 = _mm_mullo_epi16(sum, s);
    tmp_1 = _mm_mulhi_epi16(sum, s);

    sumsq_0 = _mm_unpacklo_epi16(tmp_0, tmp_1);
    sumsq_1 = _mm_unpackhi_epi16(tmp_0, tmp_1);

    // Prime sum/sumsq
    for (i = 1; i <= 6; ++i) {
      __m128i a = _mm_loadl_epi64((__m128i *)(dst + i * pitch));
      a = _mm_unpacklo_epi8(a, zero);
      sum = _mm_add_epi16(sum, a);
      a = _mm_mullo_epi16(a, a);
      sumsq_0 = _mm_add_epi32(sumsq_0, _mm_unpacklo_epi16(a, zero));
      sumsq_1 = _mm_add_epi32(sumsq_1, _mm_unpackhi_epi16(a, zero));
    }

    for (row = 0; row < rows + 8; row++) {
      const __m128i above =
          _mm_load_si128((__m128i *)above_context + (row & 7));
      __m128i this_row = _mm_loadl_epi64((__m128i *)(dst + row * pitch));
      __m128i above_sq, below_sq;
      __m128i mask_0, mask_1;
      __m128i multmp_0, multmp_1;
      __m128i rv;
      __m128i out;

      this_row = _mm_unpacklo_epi8(this_row, zero);

      if (row + 7 < rows) {
        // Instead of copying the end context we just stop loading when we get
        // to the last one.
        below_context = _mm_loadl_epi64((__m128i *)(dst + (row + 7) * pitch));
        below_context = _mm_unpacklo_epi8(below_context, zero);
      }

      sum = _mm_sub_epi16(sum, above);
      sum = _mm_add_epi16(sum, below_context);

      // context^2 fits in 16 bits. Don't need to mulhi and combine. Just zero
      // extend. Unfortunately we can't do below_sq - above_sq in 16 bits
      // because x86 does not have unpack with sign extension.
      above_sq = _mm_mullo_epi16(above, above);
      sumsq_0 = _mm_sub_epi32(sumsq_0, _mm_unpacklo_epi16(above_sq, zero));
      sumsq_1 = _mm_sub_epi32(sumsq_1, _mm_unpackhi_epi16(above_sq, zero));

      below_sq = _mm_mullo_epi16(below_context, below_context);
      sumsq_0 = _mm_add_epi32(sumsq_0, _mm_unpacklo_epi16(below_sq, zero));
      sumsq_1 = _mm_add_epi32(sumsq_1, _mm_unpackhi_epi16(below_sq, zero));

      // sumsq * 16 - sumsq == sumsq * 15
      mask_0 = _mm_slli_epi32(sumsq_0, 4);
      mask_0 = _mm_sub_epi32(mask_0, sumsq_0);
      mask_1 = _mm_slli_epi32(sumsq_1, 4);
      mask_1 = _mm_sub_epi32(mask_1, sumsq_1);

      multmp_0 = _mm_mullo_epi16(sum, sum);
      multmp_1 = _mm_mulhi_epi16(sum, sum);

      mask_0 = _mm_sub_epi32(mask_0, _mm_unpacklo_epi16(multmp_0, multmp_1));
      mask_1 = _mm_sub_epi32(mask_1, _mm_unpackhi_epi16(multmp_0, multmp_1));

      // mask - f gives a negative value when mask < f
      mask_0 = _mm_sub_epi32(mask_0, f);
      mask_1 = _mm_sub_epi32(mask_1, f);

      // Shift the sign bit down to create a mask
      mask_0 = _mm_srai_epi32(mask_0, 31);
      mask_1 = _mm_srai_epi32(mask_1, 31);

      mask_0 = _mm_packs_epi32(mask_0, mask_1);

      rv = _mm_loadu_si128((__m128i const *)(vpx_rv + (row & 127)));

      mask_1 = _mm_add_epi16(rv, sum);
      mask_1 = _mm_add_epi16(mask_1, this_row);
      mask_1 = _mm_srai_epi16(mask_1, 4);

      mask_1 = _mm_and_si128(mask_0, mask_1);
      mask_0 = _mm_andnot_si128(mask_0, this_row);
      out = _mm_or_si128(mask_1, mask_0);

      _mm_storel_epi64((__m128i *)(dst + row * pitch),
                       _mm_packus_epi16(out, zero));

      _mm_store_si128((__m128i *)above_context + ((row + 8) & 7), this_row);
    }

    dst += 8;
  }
}
