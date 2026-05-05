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
#include <immintrin.h>

#include "./vp9_rtcd.h"
#include "./vpx_dsp_rtcd.h"
#include "vp9/encoder/vp9_temporal_filter.h"

static INLINE void highbd_shuffle_12tap_filter_avx2(const int16_t *filter,
                                                    __m256i *f) {
  const __m256i f_low =
      _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)filter));
  const __m256i f_high = _mm256_broadcastsi128_si256(
      _mm_loadl_epi64((const __m128i *)(filter + 8)));

  f[0] = _mm256_shuffle_epi32(f_low, 0x00);
  f[1] = _mm256_shuffle_epi32(f_low, 0x55);
  f[2] = _mm256_shuffle_epi32(f_low, 0xaa);
  f[3] = _mm256_shuffle_epi32(f_low, 0xff);
  f[4] = _mm256_shuffle_epi32(f_high, 0x00);
  f[5] = _mm256_shuffle_epi32(f_high, 0x55);
}

static INLINE __m256i highbd_convolve_12tap(const __m256i *s,
                                            const __m256i *f) {
  const __m256i res_0 = _mm256_madd_epi16(s[0], f[0]);
  const __m256i res_1 = _mm256_madd_epi16(s[1], f[1]);
  const __m256i res_2 = _mm256_madd_epi16(s[2], f[2]);
  const __m256i res_3 = _mm256_madd_epi16(s[3], f[3]);
  const __m256i res_4 = _mm256_madd_epi16(s[4], f[4]);
  const __m256i res_5 = _mm256_madd_epi16(s[5], f[5]);

  const __m256i res =
      _mm256_add_epi32(_mm256_add_epi32(res_0, res_1),
                       _mm256_add_epi32(_mm256_add_epi32(res_2, res_3),
                                        _mm256_add_epi32(res_4, res_5)));
  return res;
}

static INLINE void reuse_src_data_avx2(const __m256i *src, __m256i *des) {
  des[0] = src[0];
  des[1] = src[1];
  des[2] = src[2];
  des[3] = src[3];
  des[4] = src[4];
}

void vpx_highbd_convolve12_horiz_avx2(const uint16_t *src, ptrdiff_t src_stride,
                                      uint16_t *dst, ptrdiff_t dst_stride,
                                      const InterpKernel12 *filter, int x0_q4,
                                      int x_step_q4, int y0_q4, int y_step_q4,
                                      int w, int h, int bd) {
  assert(x_step_q4 == 16);
  (void)y0_q4;
  (void)x_step_q4;
  (void)y_step_q4;
  const uint16_t *src_ptr = src;
  src_ptr -= MAX_FILTER_TAP / 2 - 1;
  __m256i s[6], f[6];
  const __m256i rounding = _mm256_set1_epi32(1 << (FILTER_BITS - 1));
  const __m256i max = _mm256_set1_epi16((1 << bd) - 1);
  highbd_shuffle_12tap_filter_avx2(filter[x0_q4], f);

  for (int j = 0; j < w; j += 8) {
    for (int i = 0; i < h; i += 2) {
      // s00 s01 s02 s03 s04 s05 s06 s07 s08 s09 s010 s011 s012 s013 s014 s015
      const __m256i row0 =
          _mm256_loadu_si256((const __m256i *)&src_ptr[i * src_stride + j]);
      // s10 s11 s12 s13 s14 s15 s16 s17 s18 s19 s110 s111 s112 s113 s114
      // s115
      const __m256i row1 = _mm256_loadu_si256(
          (const __m256i *)&src_ptr[(i + 1) * src_stride + j]);
      // s016 s017 s018 s019 s020 s021 s022 s023
      const __m128i row0_16 =
          _mm_loadu_si128((const __m128i *)&src_ptr[i * src_stride + j + 16]);
      // s116 s117 s118 s119 s120 s121 s122 s123
      const __m128i row1_16 = _mm_loadu_si128(
          (const __m128i *)&src_ptr[(i + 1) * src_stride + j + 16]);

      // s00 s01 s02 s03 s04 s05 s06 s07 | s10 s11 s12 s13 s14 s15 s16 s17
      const __m256i r0 = _mm256_permute2x128_si256(row0, row1, 0x20);
      // s08 s09 s010 s011 s012 s013 s014 s015 | s18 s19 s110 s111 s112 s113
      // s114 s115
      const __m256i r1 = _mm256_permute2x128_si256(row0, row1, 0x31);
      // s016 s017 s018 s019 s020 s021 s022 s023 | s116 s117 s118 s119 s120 s121
      // s122 s123
      const __m256i r2 =
          _mm256_inserti128_si256(_mm256_castsi128_si256(row0_16), row1_16, 1);

      // even pixels
      s[0] = r0;
      s[1] = _mm256_alignr_epi8(r1, r0, 4);
      s[2] = _mm256_alignr_epi8(r1, r0, 8);
      s[3] = _mm256_alignr_epi8(r1, r0, 12);
      s[4] = r1;
      s[5] = _mm256_alignr_epi8(r2, r1, 4);

      // 00 02 04 06 | 10 12 14 16
      __m256i res_even = highbd_convolve_12tap(s, f);
      res_even =
          _mm256_srai_epi32(_mm256_add_epi32(res_even, rounding), FILTER_BITS);

      // odd pixels
      s[0] = _mm256_alignr_epi8(r1, r0, 2);
      s[1] = _mm256_alignr_epi8(r1, r0, 6);
      s[2] = _mm256_alignr_epi8(r1, r0, 10);
      s[3] = _mm256_alignr_epi8(r1, r0, 14);
      s[4] = _mm256_alignr_epi8(r2, r1, 2);
      s[5] = _mm256_alignr_epi8(r2, r1, 6);

      // 01 03 05 07 | 11 13 15 17
      __m256i res_odd = highbd_convolve_12tap(s, f);
      res_odd =
          _mm256_srai_epi32(_mm256_add_epi32(res_odd, rounding), FILTER_BITS);

      // 00 01 02 03 | 10 11 12 13
      const __m256i res_0 = _mm256_unpacklo_epi32(res_even, res_odd);
      // 04 05 06 07 | 14 15 16 17
      const __m256i res_1 = _mm256_unpackhi_epi32(res_even, res_odd);
      // 00 01 02 03 | 04 05 06 07 | 10 11 12 13 | 14 15 16 17
      const __m256i res_2 = _mm256_packus_epi32(res_0, res_1);
      const __m256i res = _mm256_min_epi16(res_2, max);
      _mm_storeu_si128((__m128i *)&dst[i * dst_stride + j],
                       _mm256_castsi256_si128(res));
      if (i + 1 < h) {
        _mm_storeu_si128((__m128i *)(&dst[(i + 1) * dst_stride + j]),
                         _mm256_extractf128_si256(res, 1));
      }
    }
  }
}

void vpx_highbd_convolve12_vert_avx2(const uint16_t *src, ptrdiff_t src_stride,
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
  __m256i s[12], f[6];
  const __m256i rounding = _mm256_set1_epi32(((1 << FILTER_BITS) >> 1));
  const __m256i max = _mm256_set1_epi16((1 << bd) - 1);
  highbd_shuffle_12tap_filter_avx2(filter[y0_q4], f);

  for (int j = 0; j < w; j += 8) {
    __m128i s0 =
        _mm_loadu_si128((const __m128i *)(src_ptr + 0 * src_stride + j));
    __m128i s1 =
        _mm_loadu_si128((const __m128i *)(src_ptr + 1 * src_stride + j));
    __m128i s2 =
        _mm_loadu_si128((const __m128i *)(src_ptr + 2 * src_stride + j));
    __m128i s3 =
        _mm_loadu_si128((const __m128i *)(src_ptr + 3 * src_stride + j));
    __m128i s4 =
        _mm_loadu_si128((const __m128i *)(src_ptr + 4 * src_stride + j));
    __m128i s5 =
        _mm_loadu_si128((const __m128i *)(src_ptr + 5 * src_stride + j));
    __m128i s6 =
        _mm_loadu_si128((const __m128i *)(src_ptr + 6 * src_stride + j));
    __m128i s7 =
        _mm_loadu_si128((const __m128i *)(src_ptr + 7 * src_stride + j));
    __m128i s8 =
        _mm_loadu_si128((const __m128i *)(src_ptr + 8 * src_stride + j));
    __m128i s9 =
        _mm_loadu_si128((const __m128i *)(src_ptr + 9 * src_stride + j));
    __m128i s10t =
        _mm_loadu_si128((const __m128i *)(src_ptr + 10 * src_stride + j));

    __m256i r01 = _mm256_inserti128_si256(_mm256_castsi128_si256(s0), s1, 1);
    __m256i r12 = _mm256_inserti128_si256(_mm256_castsi128_si256(s1), s2, 1);
    __m256i r23 = _mm256_inserti128_si256(_mm256_castsi128_si256(s2), s3, 1);
    __m256i r34 = _mm256_inserti128_si256(_mm256_castsi128_si256(s3), s4, 1);
    __m256i r45 = _mm256_inserti128_si256(_mm256_castsi128_si256(s4), s5, 1);
    __m256i r56 = _mm256_inserti128_si256(_mm256_castsi128_si256(s5), s6, 1);
    __m256i r67 = _mm256_inserti128_si256(_mm256_castsi128_si256(s6), s7, 1);
    __m256i r78 = _mm256_inserti128_si256(_mm256_castsi128_si256(s7), s8, 1);
    __m256i r89 = _mm256_inserti128_si256(_mm256_castsi128_si256(s8), s9, 1);
    __m256i r910 = _mm256_inserti128_si256(_mm256_castsi128_si256(s9), s10t, 1);

    s[0] = _mm256_unpacklo_epi16(r01, r12);
    s[1] = _mm256_unpacklo_epi16(r23, r34);
    s[2] = _mm256_unpacklo_epi16(r45, r56);
    s[3] = _mm256_unpacklo_epi16(r67, r78);
    s[4] = _mm256_unpacklo_epi16(r89, r910);

    s[6] = _mm256_unpackhi_epi16(r01, r12);
    s[7] = _mm256_unpackhi_epi16(r23, r34);
    s[8] = _mm256_unpackhi_epi16(r45, r56);
    s[9] = _mm256_unpackhi_epi16(r67, r78);
    s[10] = _mm256_unpackhi_epi16(r89, r910);
    for (int i = 0; i < h; i += 2) {
      const __m128i s10 = _mm_loadu_si128(
          (const __m128i *)(src_ptr + (i + 10) * src_stride + j));
      const __m128i s11 = _mm_loadu_si128(
          (const __m128i *)(src_ptr + (i + 11) * src_stride + j));
      const __m128i s12 = _mm_loadu_si128(
          (const __m128i *)(src_ptr + (i + 12) * src_stride + j));
      __m256i r1011 =
          _mm256_inserti128_si256(_mm256_castsi128_si256(s10), s11, 1);
      __m256i r1112 =
          _mm256_inserti128_si256(_mm256_castsi128_si256(s11), s12, 1);

      s[5] = _mm256_unpacklo_epi16(r1011, r1112);
      s[11] = _mm256_unpackhi_epi16(r1011, r1112);

      // 00 01 02 03 | 10 11 12 13
      const __m256i res_a = highbd_convolve_12tap(s, f);
      __m256i res_a_round =
          _mm256_srai_epi32(_mm256_add_epi32(res_a, rounding), FILTER_BITS);
      // 04 05 06 07 | 14 15 16 17
      const __m256i res_b = highbd_convolve_12tap(s + 6, f);
      __m256i res_b_round =
          _mm256_srai_epi32(_mm256_add_epi32(res_b, rounding), FILTER_BITS);

      // 00 01 02 03 | 04 05 06 07 | 10 11 12 13 | 14 15 16 17
      const __m256i res_0 = _mm256_packus_epi32(res_a_round, res_b_round);
      const __m256i res = _mm256_min_epi16(res_0, max);
      _mm_storeu_si128((__m128i *)&dst[i * dst_stride + j],
                       _mm256_castsi256_si128(res));

      _mm_storeu_si128((__m128i *)(&dst[(i + 1) * dst_stride + j]),
                       _mm256_extractf128_si256(res, 1));

      reuse_src_data_avx2(s + 1, s);
      reuse_src_data_avx2(s + 7, s + 6);
    }
  }
}

void vpx_highbd_convolve12_avx2(const uint16_t *src, ptrdiff_t src_stride,
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

  vpx_highbd_convolve12_horiz_avx2(src - src_stride * (MAX_FILTER_TAP / 2 - 1),
                                   src_stride, temp, temp_stride, filter, x0_q4,
                                   x_step_q4, y0_q4, y_step_q4, w,
                                   intermediate_height, bd);
  vpx_highbd_convolve12_vert_avx2(temp + temp_stride * (MAX_FILTER_TAP / 2 - 1),
                                  temp_stride, dst, dst_stride, filter, x0_q4,
                                  x_step_q4, y0_q4, y_step_q4, w, h, bd);
}
