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

DECLARE_ALIGNED(16, static const uint8_t,
                shuffle_src_mask1_ssse3[32]) = { 0, 1, 1, 2, 2, 3, 3, 4,
                                                 4, 5, 5, 6, 6, 7, 7, 8 };

DECLARE_ALIGNED(16, static const uint8_t,
                shuffle_src_mask2_ssse3[32]) = { 2, 3, 3, 4, 4, 5, 5, 6,
                                                 6, 7, 7, 8, 8, 9, 9, 10 };

DECLARE_ALIGNED(16, static const uint8_t,
                shuffle_src_mask3_ssse3[32]) = { 4, 5, 5, 6,  6,  7,  7,  8,
                                                 8, 9, 9, 10, 10, 11, 11, 12 };

DECLARE_ALIGNED(16, static const uint8_t, shuffle_src_mask4_ssse3[32]) = {
  6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14
};

static INLINE void sign_extend_16bit_to_32bit_ssse3(__m128i in, __m128i zero,
                                                    __m128i *out_lo,
                                                    __m128i *out_hi) {
  const __m128i sign_bits = _mm_cmpgt_epi16(zero, in);
  *out_lo = _mm_unpacklo_epi16(in, sign_bits);
  *out_hi = _mm_unpackhi_epi16(in, sign_bits);
}

static INLINE void shuffle_12tap_filter_ssse3(const int16_t *filter,
                                              __m128i *f) {
  const __m128i f_low = _mm_loadu_si128((const __m128i *)filter);
  const __m128i f_high = _mm_loadl_epi64((const __m128i *)(filter + 8));

  f[0] = _mm_shuffle_epi8(f_low, _mm_set1_epi16(0x0200u));
  f[1] = _mm_shuffle_epi8(f_low, _mm_set1_epi16(0x0604u));
  f[2] = _mm_shuffle_epi8(f_low, _mm_set1_epi16(0x0a08u));
  f[3] = _mm_shuffle_epi8(f_low, _mm_set1_epi16(0x0e0cu));
  f[4] = _mm_shuffle_epi8(f_high, _mm_set1_epi16(0x0200u));
  f[5] = _mm_shuffle_epi8(f_high, _mm_set1_epi16(0x0604u));
}

static INLINE void shuffle_src_data_ssse3(const __m128i *r1, const __m128i *r2,
                                          const __m128i *f, __m128i *s) {
  s[0] = _mm_shuffle_epi8(*r1, f[0]);
  s[1] = _mm_shuffle_epi8(*r1, f[1]);
  s[2] = _mm_shuffle_epi8(*r1, f[2]);
  s[3] = _mm_shuffle_epi8(*r1, f[3]);
  s[4] = _mm_shuffle_epi8(*r2, f[0]);
  s[5] = _mm_shuffle_epi8(*r2, f[1]);
}

static INLINE void reuse_src_data_ssse3(const __m128i *src, __m128i *des) {
  des[0] = src[0];
  des[1] = src[1];
  des[2] = src[2];
  des[3] = src[3];
  des[4] = src[4];
}

static INLINE __m128i convolve12_16_ssse3(const __m128i *const s,
                                          const __m128i *const f) {
  // multiply 2 adjacent elements with the filter and add the result
  const __m128i k_64 = _mm_set1_epi16(1 << (FILTER_BITS - 1));
  const __m128i x0 = _mm_maddubs_epi16(s[0], f[0]);
  const __m128i x1 = _mm_maddubs_epi16(s[1], f[1]);
  const __m128i x2 = _mm_maddubs_epi16(s[2], f[2]);
  const __m128i x3 = _mm_maddubs_epi16(s[3], f[3]);
  const __m128i x4 = _mm_maddubs_epi16(s[4], f[4]);
  const __m128i x5 = _mm_maddubs_epi16(s[5], f[5]);
  __m128i sum1, sum2, sum3, s0, s1, s2, s3, s4, s5;

  sum1 = _mm_add_epi16(x0, x2);
  sum2 = _mm_add_epi16(x3, x5);
  sum3 = _mm_add_epi16(x1, x4);
  sum3 = _mm_add_epi16(sum3, k_64);

  sign_extend_16bit_to_32bit_ssse3(sum1, _mm_setzero_si128(), &s0, &s1);
  sign_extend_16bit_to_32bit_ssse3(sum2, _mm_setzero_si128(), &s2, &s3);
  sign_extend_16bit_to_32bit_ssse3(sum3, _mm_setzero_si128(), &s4, &s5);
  sum1 = _mm_add_epi32(s0, s2);
  sum2 = _mm_add_epi32(s1, s3);
  sum1 = _mm_add_epi32(sum1, s4);
  sum2 = _mm_add_epi32(sum2, s5);

  // round and shift by 7 bit each 32 bit
  // 0 1 2 3
  sum1 = _mm_srai_epi32(sum1, FILTER_BITS);
  // 4 5 6 7
  sum2 = _mm_srai_epi32(sum2, FILTER_BITS);

  // 0 1 2 3 4 5 6 7
  __m128i const res = _mm_packs_epi32(sum1, sum2);
  return res;
}

void vpx_convolve12_horiz_ssse3(const uint8_t *src, ptrdiff_t src_stride,
                                uint8_t *dst, ptrdiff_t dst_stride,
                                const InterpKernel12 *filter, int x0_q4,
                                int x_step_q4, int y0_q4, int y_step_q4, int w,
                                int h) {
  assert(x_step_q4 == 16);
  assert(w == 32 || w == 16 || w == 8);
  (void)y0_q4;
  (void)x_step_q4;
  (void)y_step_q4;
  const uint8_t *src_ptr = src;
  src_ptr -= MAX_FILTER_TAP / 2 - 1;
  __m128i s[6], f[6], src_mask[4];

  shuffle_12tap_filter_ssse3(filter[x0_q4], f);
  src_mask[0] = _mm_load_si128((__m128i const *)shuffle_src_mask1_ssse3);
  src_mask[1] = _mm_load_si128((__m128i const *)shuffle_src_mask2_ssse3);
  src_mask[2] = _mm_load_si128((__m128i const *)shuffle_src_mask3_ssse3);
  src_mask[3] = _mm_load_si128((__m128i const *)shuffle_src_mask4_ssse3);
  if (w == 8) {
    for (int i = 0; i < h; i += 2) {
      // s00 s01 s02 s03 s04 s05 s06 s07 s08 s09 s010 s011 s012 s013 s014 s015
      const __m128i row0 =
          _mm_loadu_si128((const __m128i *)&src_ptr[i * src_stride]);
      // s08 s09 s010 s011 s012 s013 s014 s015 s016 s017 s018 s019 s020 s021
      // s022 s023
      const __m128i row0_8 =
          _mm_loadu_si128((const __m128i *)&src_ptr[i * src_stride + 8]);
      // s10 s11 s12 s13 s14 s15 s16 s17 s18 s19 s110 s111 s112 s113 s114 s115
      const __m128i row1 =
          _mm_loadu_si128((const __m128i *)&src_ptr[(i + 1) * src_stride]);
      const __m128i row1_8 =
          _mm_loadu_si128((const __m128i *)&src_ptr[(i + 1) * src_stride + 8]);

      shuffle_src_data_ssse3(&row0, &row0_8, src_mask, s);
      const __m128i res_0 = convolve12_16_ssse3(s, f);

      shuffle_src_data_ssse3(&row1, &row1_8, src_mask, s);
      const __m128i res_1 = convolve12_16_ssse3(s, f);

      const __m128i res = _mm_packus_epi16(res_0, res_1);
      _mm_storel_epi64((__m128i *)&dst[i * dst_stride], res);
      _mm_storel_epi64((__m128i *)&dst[(i + 1) * dst_stride],
                       _mm_srli_si128(res, 8));
    }
  } else {
    for (int j = 0; j < w; j += 16) {
      for (int i = 0; i < h; i++) {
        // s00 s01 s02 s03 s04 s05 s06 s07 s08 s09 s010 s011 s012 s013 s014 s015
        const __m128i r0 =
            _mm_loadu_si128((const __m128i *)&src_ptr[i * src_stride + j]);
        // s016 s017 s018 s019 s020 s021 s022 s023 s024 s025 s026 s027 s028 s029
        // s030 s031
        const __m128i r2 =
            _mm_loadu_si128((const __m128i *)&src_ptr[i * src_stride + j + 16]);

        // s08 s09 s010 s011 s012 s013 s014 s015 s016 s017 s018 s019 s020 s021
        // s022 s023
        const __m128i r1 = _mm_alignr_epi8(r2, r0, 8);

        shuffle_src_data_ssse3(&r0, &r1, src_mask, s);
        const __m128i res_0 = convolve12_16_ssse3(s, f);

        shuffle_src_data_ssse3(&r1, &r2, src_mask, s);
        const __m128i res_1 = convolve12_16_ssse3(s, f);

        const __m128i res = _mm_packus_epi16(res_0, res_1);
        _mm_storeu_si128((__m128i *)&dst[i * dst_stride + j], res);
      }
    }
  }
}

void vpx_convolve12_vert_ssse3(const uint8_t *src, ptrdiff_t src_stride,
                               uint8_t *dst, ptrdiff_t dst_stride,
                               const InterpKernel12 *filter, int x0_q4,
                               int x_step_q4, int y0_q4, int y_step_q4, int w,
                               int h) {
  assert(y_step_q4 == 16);
  assert(h == 32 || h == 16 || h == 8);
  assert(w == 32 || w == 16 || w == 8);
  (void)x0_q4;
  (void)x_step_q4;
  (void)y_step_q4;
  const uint8_t *src_ptr = src;
  src_ptr -= src_stride * (MAX_FILTER_TAP / 2 - 1);
  __m128i s[12], f[6];

  shuffle_12tap_filter_ssse3(filter[y0_q4], f);
  for (int j = 0; j < w; j += 8) {
    const __m128i s0 =
        _mm_loadl_epi64((const __m128i *)(src_ptr + 0 * src_stride + j));
    const __m128i s1 =
        _mm_loadl_epi64((const __m128i *)(src_ptr + 1 * src_stride + j));
    const __m128i s2 =
        _mm_loadl_epi64((const __m128i *)(src_ptr + 2 * src_stride + j));
    const __m128i s3 =
        _mm_loadl_epi64((const __m128i *)(src_ptr + 3 * src_stride + j));
    const __m128i s4 =
        _mm_loadl_epi64((const __m128i *)(src_ptr + 4 * src_stride + j));
    const __m128i s5 =
        _mm_loadl_epi64((const __m128i *)(src_ptr + 5 * src_stride + j));
    const __m128i s6 =
        _mm_loadl_epi64((const __m128i *)(src_ptr + 6 * src_stride + j));
    const __m128i s7 =
        _mm_loadl_epi64((const __m128i *)(src_ptr + 7 * src_stride + j));
    const __m128i s8 =
        _mm_loadl_epi64((const __m128i *)(src_ptr + 8 * src_stride + j));
    const __m128i s9 =
        _mm_loadl_epi64((const __m128i *)(src_ptr + 9 * src_stride + j));
    const __m128i s10t =
        _mm_loadl_epi64((const __m128i *)(src_ptr + 10 * src_stride + j));

    // 00 10 01 11 02 12 03 13 04 14 05 15 06 16 07 17
    s[0] = _mm_unpacklo_epi8(s0, s1);
    s[1] = _mm_unpacklo_epi8(s2, s3);
    s[2] = _mm_unpacklo_epi8(s4, s5);
    s[3] = _mm_unpacklo_epi8(s6, s7);
    s[4] = _mm_unpacklo_epi8(s8, s9);

    s[6] = _mm_unpacklo_epi8(s1, s2);
    s[7] = _mm_unpacklo_epi8(s3, s4);
    s[8] = _mm_unpacklo_epi8(s5, s6);
    s[9] = _mm_unpacklo_epi8(s7, s8);
    s[10] = _mm_unpacklo_epi8(s9, s10t);
    for (int i = 0; i < h; i += 2) {
      const __m128i s10 = _mm_loadl_epi64(
          (const __m128i *)(src_ptr + (i + 10) * src_stride + j));
      const __m128i s11 = _mm_loadl_epi64(
          (const __m128i *)(src_ptr + (i + 11) * src_stride + j));
      const __m128i s12 = _mm_loadl_epi64(
          (const __m128i *)(src_ptr + (i + 12) * src_stride + j));

      s[5] = _mm_unpacklo_epi8(s10, s11);
      s[11] = _mm_unpacklo_epi8(s11, s12);

      const __m128i res_0 = convolve12_16_ssse3(s, f);
      const __m128i res_1 = convolve12_16_ssse3(s + 6, f);

      __m128i res = _mm_packus_epi16(res_0, res_1);

      _mm_storel_epi64((__m128i *)&dst[i * dst_stride + j], res);
      _mm_storel_epi64((__m128i *)&dst[(i + 1) * dst_stride + j],
                       _mm_srli_si128(res, 8));

      reuse_src_data_ssse3(s + 1, s);
      reuse_src_data_ssse3(s + 7, s + 6);
    }
  }
}

void vpx_convolve12_ssse3(const uint8_t *src, ptrdiff_t src_stride,
                          uint8_t *dst, ptrdiff_t dst_stride,
                          const InterpKernel12 *filter, int x0_q4,
                          int x_step_q4, int y0_q4, int y_step_q4, int w,
                          int h) {
  assert(x_step_q4 == 16 && y_step_q4 == 16);
  assert(h == 32 || h == 16 || h == 8);
  assert(w == 32 || w == 16 || w == 8);
  DECLARE_ALIGNED(32, uint8_t, temp[BW * (BH + MAX_FILTER_TAP - 1)]);
  const int temp_stride = BW;
  const int intermediate_height =
      (((h - 1) * y_step_q4 + y0_q4) >> SUBPEL_BITS) + MAX_FILTER_TAP;
  vpx_convolve12_horiz_ssse3(src - src_stride * (MAX_FILTER_TAP / 2 - 1),
                             src_stride, temp, temp_stride, filter, x0_q4,
                             x_step_q4, y0_q4, y_step_q4, w,
                             intermediate_height);
  vpx_convolve12_vert_ssse3(temp + temp_stride * (MAX_FILTER_TAP / 2 - 1),
                            temp_stride, dst, dst_stride, filter, x0_q4,
                            x_step_q4, y0_q4, y_step_q4, w, h);
}
