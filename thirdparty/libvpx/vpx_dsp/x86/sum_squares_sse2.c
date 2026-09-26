/*
 *  Copyright (c) 2016 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>
#include <emmintrin.h>

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/x86/mem_sse2.h"

uint64_t vpx_sum_squares_2d_i16_sse2(const int16_t *src, int stride, int size) {
  // Over 75% of all calls are with size == 4.
  if (size == 4) {
    __m128i s[2], sq[2], ss;

    s[0] = _mm_loadl_epi64((const __m128i *)(src + 0 * stride));
    s[0] = loadh_epi64(s[0], src + 1 * stride);
    s[1] = _mm_loadl_epi64((const __m128i *)(src + 2 * stride));
    s[1] = loadh_epi64(s[1], src + 3 * stride);
    sq[0] = _mm_madd_epi16(s[0], s[0]);
    sq[1] = _mm_madd_epi16(s[1], s[1]);
    sq[0] = _mm_add_epi32(sq[0], sq[1]);
    ss = _mm_add_epi32(sq[0], _mm_srli_si128(sq[0], 8));
    ss = _mm_add_epi32(ss, _mm_srli_epi64(ss, 32));

    return (uint64_t)_mm_cvtsi128_si32(ss);
  } else {
    // Generic case
    int r = size;
    const __m128i v_zext_mask_q = _mm_set_epi32(0, -1, 0, -1);
    __m128i v_acc_q = _mm_setzero_si128();

    assert(size % 8 == 0);

    do {
      int c = 0;
      __m128i v_acc_d = _mm_setzero_si128();

      do {
        const int16_t *const b = src + c;
        const __m128i v_val_0_w =
            _mm_load_si128((const __m128i *)(b + 0 * stride));
        const __m128i v_val_1_w =
            _mm_load_si128((const __m128i *)(b + 1 * stride));
        const __m128i v_val_2_w =
            _mm_load_si128((const __m128i *)(b + 2 * stride));
        const __m128i v_val_3_w =
            _mm_load_si128((const __m128i *)(b + 3 * stride));
        const __m128i v_val_4_w =
            _mm_load_si128((const __m128i *)(b + 4 * stride));
        const __m128i v_val_5_w =
            _mm_load_si128((const __m128i *)(b + 5 * stride));
        const __m128i v_val_6_w =
            _mm_load_si128((const __m128i *)(b + 6 * stride));
        const __m128i v_val_7_w =
            _mm_load_si128((const __m128i *)(b + 7 * stride));

        const __m128i v_sq_0_d = _mm_madd_epi16(v_val_0_w, v_val_0_w);
        const __m128i v_sq_1_d = _mm_madd_epi16(v_val_1_w, v_val_1_w);
        const __m128i v_sq_2_d = _mm_madd_epi16(v_val_2_w, v_val_2_w);
        const __m128i v_sq_3_d = _mm_madd_epi16(v_val_3_w, v_val_3_w);
        const __m128i v_sq_4_d = _mm_madd_epi16(v_val_4_w, v_val_4_w);
        const __m128i v_sq_5_d = _mm_madd_epi16(v_val_5_w, v_val_5_w);
        const __m128i v_sq_6_d = _mm_madd_epi16(v_val_6_w, v_val_6_w);
        const __m128i v_sq_7_d = _mm_madd_epi16(v_val_7_w, v_val_7_w);

        const __m128i v_sum_01_d = _mm_add_epi32(v_sq_0_d, v_sq_1_d);
        const __m128i v_sum_23_d = _mm_add_epi32(v_sq_2_d, v_sq_3_d);
        const __m128i v_sum_45_d = _mm_add_epi32(v_sq_4_d, v_sq_5_d);
        const __m128i v_sum_67_d = _mm_add_epi32(v_sq_6_d, v_sq_7_d);

        const __m128i v_sum_0123_d = _mm_add_epi32(v_sum_01_d, v_sum_23_d);
        const __m128i v_sum_4567_d = _mm_add_epi32(v_sum_45_d, v_sum_67_d);

        v_acc_d = _mm_add_epi32(v_acc_d, v_sum_0123_d);
        v_acc_d = _mm_add_epi32(v_acc_d, v_sum_4567_d);
        c += 8;
      } while (c < size);

      v_acc_q = _mm_add_epi64(v_acc_q, _mm_and_si128(v_acc_d, v_zext_mask_q));
      v_acc_q = _mm_add_epi64(v_acc_q, _mm_srli_epi64(v_acc_d, 32));

      src += 8 * stride;
      r -= 8;
    } while (r);

    v_acc_q = _mm_add_epi64(v_acc_q, _mm_srli_si128(v_acc_q, 8));

#if VPX_ARCH_X86_64
    return (uint64_t)_mm_cvtsi128_si64(v_acc_q);
#else
    {
      uint64_t tmp;
      _mm_storel_epi64((__m128i *)&tmp, v_acc_q);
      return tmp;
    }
#endif
  }
}
