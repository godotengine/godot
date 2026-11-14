/*
 *  Copyright (c) 2024 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>
#include <tmmintrin.h>  // SSSE3

#include "./vp9_rtcd.h"
#include "./vpx_dsp_rtcd.h"
#include "vp9/encoder/vp9_temporal_filter.h"

static INLINE void highbd_shuffle_12tap_filter_ssse3(const int16_t *filter,
                                                     __m128i *f) {
  const __m128i f_low = _mm_loadu_si128((const __m128i *)filter);
  const __m128i f_high = _mm_loadl_epi64((const __m128i *)(filter + 8));

  f[0] = _mm_shuffle_epi32(f_low, 0x00);
  f[1] = _mm_shuffle_epi32(f_low, 0x55);
  f[2] = _mm_shuffle_epi32(f_low, 0xaa);
  f[3] = _mm_shuffle_epi32(f_low, 0xff);
  f[4] = _mm_shuffle_epi32(f_high, 0x00);
  f[5] = _mm_shuffle_epi32(f_high, 0x55);
}

static INLINE void unpacklo_src_ssse3(__m128i *a, __m128i *s) {
  s[0] = _mm_unpacklo_epi16(a[0], a[1]);
  s[1] = _mm_unpacklo_epi16(a[2], a[3]);
  s[2] = _mm_unpacklo_epi16(a[4], a[5]);
  s[3] = _mm_unpacklo_epi16(a[6], a[7]);
  s[4] = _mm_unpacklo_epi16(a[8], a[9]);
}

static INLINE void unpackhi_src_ssse3(__m128i *a, __m128i *s) {
  s[0] = _mm_unpackhi_epi16(a[0], a[1]);
  s[1] = _mm_unpackhi_epi16(a[2], a[3]);
  s[2] = _mm_unpackhi_epi16(a[4], a[5]);
  s[3] = _mm_unpackhi_epi16(a[6], a[7]);
  s[4] = _mm_unpackhi_epi16(a[8], a[9]);
}

static INLINE __m128i highbd_convolve_12tap(const __m128i *s,
                                            const __m128i *f) {
  const __m128i rounding = _mm_set1_epi32(1 << (FILTER_BITS - 1));
  const __m128i res_0 = _mm_madd_epi16(s[0], f[0]);
  const __m128i res_1 = _mm_madd_epi16(s[1], f[1]);
  const __m128i res_2 = _mm_madd_epi16(s[2], f[2]);
  const __m128i res_3 = _mm_madd_epi16(s[3], f[3]);
  const __m128i res_4 = _mm_madd_epi16(s[4], f[4]);
  const __m128i res_5 = _mm_madd_epi16(s[5], f[5]);

  const __m128i res_6 = _mm_add_epi32(
      _mm_add_epi32(res_0, res_1),
      _mm_add_epi32(_mm_add_epi32(res_2, res_3), _mm_add_epi32(res_4, res_5)));
  const __m128i res =
      _mm_srai_epi32(_mm_add_epi32(res_6, rounding), FILTER_BITS);
  return res;
}

static INLINE void reuse_src_data_ssse3(const __m128i *src, __m128i *des) {
  des[0] = src[0];
  des[1] = src[1];
  des[2] = src[2];
  des[3] = src[3];
  des[4] = src[4];
}

void vpx_highbd_convolve12_horiz_ssse3(const uint16_t *src,
                                       ptrdiff_t src_stride, uint16_t *dst,
                                       ptrdiff_t dst_stride,
                                       const InterpKernel12 *filter, int x0_q4,
                                       int x_step_q4, int y0_q4, int y_step_q4,
                                       int w, int h, int bd) {
  assert(x_step_q4 == 16);
  (void)y0_q4;
  (void)x_step_q4;
  (void)y_step_q4;
  const uint16_t *src_ptr = src;
  src_ptr -= MAX_FILTER_TAP / 2 - 1;
  __m128i s[6], f[6];
  const __m128i max = _mm_set1_epi16((1 << bd) - 1);
  const __m128i min = _mm_setzero_si128();
  highbd_shuffle_12tap_filter_ssse3(filter[x0_q4], f);

  for (int j = 0; j < w; j += 8) {
    for (int i = 0; i < h; i++) {
      // s00 s01 s02 s03 s04 s05 s06 s07
      const __m128i r0 =
          _mm_loadu_si128((const __m128i *)&src_ptr[i * src_stride + j]);
      // s08 s09 s010 s011 s012 s013 s014 s015
      const __m128i r1 =
          _mm_loadu_si128((const __m128i *)&src_ptr[i * src_stride + j + 8]);
      // s016 s017 s018 s019 s020 s021 s022 s023
      const __m128i r2 =
          _mm_loadu_si128((const __m128i *)&src_ptr[i * src_stride + j + 16]);

      // even pixels
      s[0] = r0;
      s[1] = _mm_alignr_epi8(r1, r0, 4);
      s[2] = _mm_alignr_epi8(r1, r0, 8);
      s[3] = _mm_alignr_epi8(r1, r0, 12);
      s[4] = r1;
      s[5] = _mm_alignr_epi8(r2, r1, 4);

      // 00 02 04 06
      __m128i res_even = highbd_convolve_12tap(s, f);

      // odd pixels
      s[0] = _mm_alignr_epi8(r1, r0, 2);
      s[1] = _mm_alignr_epi8(r1, r0, 6);
      s[2] = _mm_alignr_epi8(r1, r0, 10);
      s[3] = _mm_alignr_epi8(r1, r0, 14);
      s[4] = _mm_alignr_epi8(r2, r1, 2);
      s[5] = _mm_alignr_epi8(r2, r1, 6);

      // 01 03 05 07
      __m128i res_odd = highbd_convolve_12tap(s, f);

      // 00 01 02 03
      const __m128i res_0 = _mm_unpacklo_epi32(res_even, res_odd);
      // 04 05 06 07
      const __m128i res_1 = _mm_unpackhi_epi32(res_even, res_odd);
      // 00 01 02 03 | 04 05 06 07
      const __m128i res_2 = _mm_packs_epi32(res_0, res_1);
      const __m128i res = _mm_max_epi16(_mm_min_epi16(res_2, max), min);
      _mm_storeu_si128((__m128i *)&dst[i * dst_stride + j], res);
    }
  }
}

void vpx_highbd_convolve12_vert_ssse3(const uint16_t *src, ptrdiff_t src_stride,
                                      uint16_t *dst, ptrdiff_t dst_stride,
                                      const InterpKernel12 *filter, int x0_q4,
                                      int x_step_q4, int y0_q4, int y_step_q4,
                                      int w, int h, int bd) {
  assert(y_step_q4 == 16);
  (void)x0_q4;
  (void)x_step_q4;
  (void)y_step_q4;
  const uint16_t *src_ptr = src;
  src_ptr -= src_stride * (MAX_FILTER_TAP / 2 - 1);
  __m128i s[12], r[12], a[11], f[6];
  const __m128i max = _mm_set1_epi16((1 << bd) - 1);
  const __m128i min = _mm_setzero_si128();
  highbd_shuffle_12tap_filter_ssse3(filter[y0_q4], f);

  for (int j = 0; j < w; j += 8) {
    a[0] = _mm_loadu_si128((const __m128i *)(src_ptr + 0 * src_stride + j));
    a[1] = _mm_loadu_si128((const __m128i *)(src_ptr + 1 * src_stride + j));
    a[2] = _mm_loadu_si128((const __m128i *)(src_ptr + 2 * src_stride + j));
    a[3] = _mm_loadu_si128((const __m128i *)(src_ptr + 3 * src_stride + j));
    a[4] = _mm_loadu_si128((const __m128i *)(src_ptr + 4 * src_stride + j));
    a[5] = _mm_loadu_si128((const __m128i *)(src_ptr + 5 * src_stride + j));
    a[6] = _mm_loadu_si128((const __m128i *)(src_ptr + 6 * src_stride + j));
    a[7] = _mm_loadu_si128((const __m128i *)(src_ptr + 7 * src_stride + j));
    a[8] = _mm_loadu_si128((const __m128i *)(src_ptr + 8 * src_stride + j));
    a[9] = _mm_loadu_si128((const __m128i *)(src_ptr + 9 * src_stride + j));
    a[10] = _mm_loadu_si128((const __m128i *)(src_ptr + 10 * src_stride + j));

    // even row
    unpacklo_src_ssse3(a, s);
    unpackhi_src_ssse3(a, s + 6);
    // odd row
    unpacklo_src_ssse3(a + 1, r);
    unpackhi_src_ssse3(a + 1, r + 6);

    for (int i = 0; i < h; i += 2) {
      const __m128i s0 = _mm_loadu_si128(
          (const __m128i *)(src_ptr + (i + 10) * src_stride + j));
      const __m128i s1 = _mm_loadu_si128(
          (const __m128i *)(src_ptr + (i + 11) * src_stride + j));
      const __m128i s2 = _mm_loadu_si128(
          (const __m128i *)(src_ptr + (i + 12) * src_stride + j));

      s[5] = _mm_unpacklo_epi16(s0, s1);
      r[5] = _mm_unpacklo_epi16(s1, s2);

      s[11] = _mm_unpackhi_epi16(s0, s1);
      r[11] = _mm_unpackhi_epi16(s1, s2);

      // 00 01 02 03
      const __m128i res_a = highbd_convolve_12tap(s, f);
      // 04 05 06 07
      const __m128i res_b = highbd_convolve_12tap(s + 6, f);
      // 10 11 12 13
      const __m128i res_c = highbd_convolve_12tap(r, f);
      // 14 15 16 17
      const __m128i res_d = highbd_convolve_12tap(r + 6, f);

      // 00 01 02 03 | 04 05 06 07
      const __m128i res_0 = _mm_packs_epi32(res_a, res_b);
      // 10 11 12 13 | 14 15 16 17
      const __m128i res_1 = _mm_packs_epi32(res_c, res_d);
      const __m128i res_r0 = _mm_max_epi16(_mm_min_epi16(res_0, max), min);
      const __m128i res_r1 = _mm_max_epi16(_mm_min_epi16(res_1, max), min);

      _mm_storeu_si128((__m128i *)&dst[i * dst_stride + j], res_r0);
      _mm_storeu_si128((__m128i *)&dst[(i + 1) * dst_stride + j], res_r1);

      reuse_src_data_ssse3(s + 1, s);
      reuse_src_data_ssse3(s + 7, s + 6);
      reuse_src_data_ssse3(r + 1, r);
      reuse_src_data_ssse3(r + 7, r + 6);
    }
  }
}

void vpx_highbd_convolve12_ssse3(const uint16_t *src, ptrdiff_t src_stride,
                                 uint16_t *dst, ptrdiff_t dst_stride,
                                 const InterpKernel12 *filter, int x0_q4,
                                 int x_step_q4, int y0_q4, int y_step_q4, int w,
                                 int h, int bd) {
  assert(x_step_q4 == 16 && y_step_q4 == 16);
  assert(h == 32 || h == 16 || h == 8);
  assert(w == 32 || w == 16 || w == 8);
  DECLARE_ALIGNED(32, uint16_t, temp[BW * (BH + MAX_FILTER_TAP - 1)]);
  const int temp_stride = BW;
  const int intermediate_height =
      (((h - 1) * y_step_q4 + y0_q4) >> SUBPEL_BITS) + MAX_FILTER_TAP;

  vpx_highbd_convolve12_horiz_ssse3(src - src_stride * (MAX_FILTER_TAP / 2 - 1),
                                    src_stride, temp, temp_stride, filter,
                                    x0_q4, x_step_q4, y0_q4, y_step_q4, w,
                                    intermediate_height, bd);
  vpx_highbd_convolve12_vert_ssse3(
      temp + temp_stride * (MAX_FILTER_TAP / 2 - 1), temp_stride, dst,
      dst_stride, filter, x0_q4, x_step_q4, y0_q4, y_step_q4, w, h, bd);
}
