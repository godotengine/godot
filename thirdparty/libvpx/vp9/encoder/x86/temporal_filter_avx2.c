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
#include "vp9/encoder/vp9_temporal_filter.h"

DECLARE_ALIGNED(32, static const uint8_t,
                shuffle_src_mask1_avx2[32]) = { 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
                                                6, 6, 7, 7, 8, 0, 1, 1, 2, 2, 3,
                                                3, 4, 4, 5, 5, 6, 6, 7, 7, 8 };

DECLARE_ALIGNED(32, static const uint8_t, shuffle_src_mask2_avx2[32]) = {
  2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10,
  2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10
};

DECLARE_ALIGNED(32, static const uint8_t, shuffle_src_mask3_avx2[32]) = {
  4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12,
  4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12
};

DECLARE_ALIGNED(32, static const uint8_t, shuffle_src_mask4_avx2[32]) = {
  6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14,
  6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14
};

static INLINE void shuffle_12tap_filter_avx2(const int16_t *filter,
                                             __m256i *f) {
  const __m256i f_low =
      _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)filter));
  const __m256i f_high = _mm256_broadcastsi128_si256(
      _mm_loadl_epi64((const __m128i *)(filter + 8)));

  f[0] = _mm256_shuffle_epi8(f_low, _mm256_set1_epi16(0x0200u));
  f[1] = _mm256_shuffle_epi8(f_low, _mm256_set1_epi16(0x0604u));
  f[2] = _mm256_shuffle_epi8(f_low, _mm256_set1_epi16(0x0a08u));
  f[3] = _mm256_shuffle_epi8(f_low, _mm256_set1_epi16(0x0e0cu));
  f[4] = _mm256_shuffle_epi8(f_high, _mm256_set1_epi16(0x0200u));
  f[5] = _mm256_shuffle_epi8(f_high, _mm256_set1_epi16(0x0604u));
}

static INLINE void shuffle_src_data_avx2(const __m256i *r1, const __m256i *r2,
                                         const __m256i *f, __m256i *s) {
  s[0] = _mm256_shuffle_epi8(*r1, f[0]);
  s[1] = _mm256_shuffle_epi8(*r1, f[1]);
  s[2] = _mm256_shuffle_epi8(*r1, f[2]);
  s[3] = _mm256_shuffle_epi8(*r1, f[3]);
  s[4] = _mm256_shuffle_epi8(*r2, f[0]);
  s[5] = _mm256_shuffle_epi8(*r2, f[1]);
}

static INLINE void reuse_src_data_avx2(const __m256i *src, __m256i *des) {
  des[0] = src[0];
  des[1] = src[1];
  des[2] = src[2];
  des[3] = src[3];
  des[4] = src[4];
}

static INLINE __m256i convolve12_16_avx2(const __m256i *s, const __m256i *f) {
  // multiply 2 adjacent elements with the filter and add the result
  const __m256i k_64 = _mm256_set1_epi16(1 << (FILTER_BITS - 1));
  const __m256i x0 = _mm256_maddubs_epi16(s[0], f[0]);
  const __m256i x1 = _mm256_maddubs_epi16(s[1], f[1]);
  const __m256i x2 = _mm256_maddubs_epi16(s[2], f[2]);
  const __m256i x3 = _mm256_maddubs_epi16(s[3], f[3]);
  const __m256i x4 = _mm256_maddubs_epi16(s[4], f[4]);
  const __m256i x5 = _mm256_maddubs_epi16(s[5], f[5]);
  __m256i sum1, sum2, sum3;

  sum1 = _mm256_add_epi16(x0, x2);
  sum2 = _mm256_add_epi16(x3, x5);
  sum3 = _mm256_add_epi16(x1, x4);
  sum3 = _mm256_add_epi16(sum3, k_64);

  const __m256i s0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(sum1));
  const __m256i s1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(sum1, 1));
  const __m256i s2 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(sum2));
  const __m256i s3 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(sum2, 1));
  const __m256i s4 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(sum3));
  const __m256i s5 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(sum3, 1));

  sum1 = _mm256_add_epi32(s0, s2);
  sum2 = _mm256_add_epi32(s1, s3);
  sum1 = _mm256_add_epi32(sum1, s4);
  sum2 = _mm256_add_epi32(sum2, s5);

  // round and shift by 7 bit each 32 bit
  // 0 1 2 3 4 5 6 7
  sum1 = _mm256_srai_epi32(sum1, FILTER_BITS);
  // 8 9 10 11 12 13 14 15
  sum2 = _mm256_srai_epi32(sum2, FILTER_BITS);

  // 0 1 2 3 8 9 10 11 4 5 6 7 12 13 14 15
  // 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
  __m256i const res =
      _mm256_permute4x64_epi64(_mm256_packus_epi32(sum1, sum2), 0xD8);
  return res;
}

void vpx_convolve12_horiz_avx2(const uint8_t *src, ptrdiff_t src_stride,
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
  __m256i s[6], f[6], src_mask[4];

  shuffle_12tap_filter_avx2(filter[x0_q4], f);
  src_mask[0] = _mm256_load_si256((__m256i const *)shuffle_src_mask1_avx2);
  src_mask[1] = _mm256_load_si256((__m256i const *)shuffle_src_mask2_avx2);
  src_mask[2] = _mm256_load_si256((__m256i const *)shuffle_src_mask3_avx2);
  src_mask[3] = _mm256_load_si256((__m256i const *)shuffle_src_mask4_avx2);
  if (w == 8) {
    for (int i = 0; i < h; i += 4) {
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
      // s20 s21 s22 s23 s24 s25 s26 s27 s28 s29 s210 s211 s212 s213 s214 s215
      const __m128i row2 =
          _mm_loadu_si128((const __m128i *)&src_ptr[(i + 2) * src_stride]);
      const __m128i row2_8 =
          _mm_loadu_si128((const __m128i *)&src_ptr[(i + 2) * src_stride + 8]);
      // s30 s31 s32 s33 s34 s35 s36 s37 s38 s39 s310 s311 s312 s313 s314 s115
      const __m128i row3 =
          _mm_loadu_si128((const __m128i *)&src_ptr[(i + 3) * src_stride]);
      const __m128i row3_8 =
          _mm_loadu_si128((const __m128i *)&src_ptr[(i + 3) * src_stride + 8]);
      // s00 s01 s02 s03 s04 s05 s06 s07 s08 s09 s010 s011 s012 s013 s014 s015 |
      // s10 s11 s12 s13 s14 s15 s16 s17 s18 s19 s110 s111 s112 s113 s114 s115
      const __m256i row01 =
          _mm256_inserti128_si256(_mm256_castsi128_si256(row0), row1, 1);
      // s20 s21 s22 s23 s24 s25 s26 s27 s28 s29 s210 s211 s212 s213 s214 s215 |
      // s30 s31 s32 s33 s34 s35 s36 s37 s38 s39 s310 s311 s312 s313 s314 s115
      const __m256i row23 =
          _mm256_inserti128_si256(_mm256_castsi128_si256(row2), row3, 1);
      // s08 s09 s010 s011 s012 s013 s014 s015 s016 s017 s018 s019 s020 s021
      // s022 s023 | s18 s19 s110 s111 s112 s113 s114 s115 s116 s117 s118 s119
      // s120 s121 s122 s123
      const __m256i row01_8 =
          _mm256_inserti128_si256(_mm256_castsi128_si256(row0_8), row1_8, 1);
      const __m256i row23_8 =
          _mm256_inserti128_si256(_mm256_castsi128_si256(row2_8), row3_8, 1);

      shuffle_src_data_avx2(&row01, &row01_8, src_mask, s);
      const __m256i res_0 = convolve12_16_avx2(s, f);

      shuffle_src_data_avx2(&row23, &row23_8, src_mask, s);
      const __m256i res_1 = convolve12_16_avx2(s, f);

      // 00 01 02 03 04 05 06 07 | 10 11 12 13 14 15 16 17 | 08 09 010 011 012
      // 013 014 015 | 18 19 110 111 112 113 114 115
      const __m256i res = _mm256_packus_epi16(res_0, res_1);
      const __m128i res_lo = _mm256_castsi256_si128(res);
      const __m128i res_hi = _mm256_extracti128_si256(res, 1);

      _mm_storel_epi64((__m128i *)&dst[i * dst_stride], res_lo);
      _mm_storel_epi64((__m128i *)&dst[(i + 1) * dst_stride], res_hi);
      _mm_storel_epi64((__m128i *)&dst[(i + 2) * dst_stride],
                       _mm_srli_si128(res_lo, 8));
      _mm_storel_epi64((__m128i *)&dst[(i + 3) * dst_stride],
                       _mm_srli_si128(res_hi, 8));
    }
  } else {
    for (int j = 0; j < w; j += 16) {
      for (int i = 0; i < h; i += 2) {
        // s00 s01 s02 s03 s04 s05 s06 s07 s08 s09 s010 s011 s012 s013 s014 s015
        const __m128i row0 =
            _mm_loadu_si128((const __m128i *)&src_ptr[i * src_stride + j]);
        // s016 s017 s018 s019 s020 s021 s022 s023 s024 s025 s026 s027 s028 s029
        // s030 s031
        const __m128i row0_16 =
            _mm_loadu_si128((const __m128i *)&src_ptr[i * src_stride + j + 16]);
        // s10 s11 s12 s13 s14 s15 s16 s17 s18 s19 s110 s111 s112 s113 s114
        // s115
        const __m128i row1 = _mm_loadu_si128(
            (const __m128i *)&src_ptr[(i + 1) * src_stride + j]);
        // s116 s117 s118 s119 s120 s121 s122 s123 s124 s125 s126 s127 s128
        // s129 s130 s131
        const __m128i row1_16 = _mm_loadu_si128(
            (const __m128i *)&src_ptr[(i + 1) * src_stride + j + 16]);

        // s00 s01 s02 s03 s04 s05 s06 s07 s08 s09 s010 s011 s012 s013 s014 s015
        // | s10 s11 s12 s13 s14 s15 s16 s17 s18 s19 s110 s111 s112 s113 s114
        // s115
        const __m256i r0 =
            _mm256_inserti128_si256(_mm256_castsi128_si256(row0), row1, 1);
        // s016 s017 s018 s019 s020 s021 s022 s023 s024 s025 s026 s027 s028 s029
        // s030 s031 | s116 s117 s118 s119 s120 s121 s122 s123 s124 s125 s126
        // s127 s128 s129 s130 s131
        const __m256i r2 = _mm256_inserti128_si256(
            _mm256_castsi128_si256(row0_16), row1_16, 1);

        // s08 s09 s010 s011 s012 s013 s014 s015 s016 s017 s018 s019 s020 s021
        // s022 s023 | s18 s19 s110 s111 s112 s113 s114 s115 s116 s117 s118 s119
        // s120 s121 s122 s123
        const __m256i r1 = _mm256_alignr_epi8(r2, r0, 8);

        shuffle_src_data_avx2(&r0, &r1, src_mask, s);
        const __m256i res_0 = convolve12_16_avx2(s, f);

        shuffle_src_data_avx2(&r1, &r2, src_mask, s);
        const __m256i res_1 = convolve12_16_avx2(s, f);

        const __m256i res = _mm256_packus_epi16(res_0, res_1);

        _mm_storeu_si128((__m128i *)&dst[i * dst_stride + j],
                         _mm256_castsi256_si128(res));
        if (i + 1 < h) {
          _mm_storeu_si128((__m128i *)&dst[(i + 1) * dst_stride + j],
                           _mm256_extracti128_si256(res, 1));
        }
      }
    }
  }
}

void vpx_convolve12_vert_avx2(const uint8_t *src, ptrdiff_t src_stride,
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
  __m256i s[12], f[6];

  shuffle_12tap_filter_avx2(filter[y0_q4], f);
  if (w == 8) {
    const __m128i s0 =
        _mm_loadl_epi64((const __m128i *)(src_ptr + 0 * src_stride));
    const __m128i s1 =
        _mm_loadl_epi64((const __m128i *)(src_ptr + 1 * src_stride));
    const __m128i s2 =
        _mm_loadl_epi64((const __m128i *)(src_ptr + 2 * src_stride));
    const __m128i s3 =
        _mm_loadl_epi64((const __m128i *)(src_ptr + 3 * src_stride));
    const __m128i s4 =
        _mm_loadl_epi64((const __m128i *)(src_ptr + 4 * src_stride));
    const __m128i s5 =
        _mm_loadl_epi64((const __m128i *)(src_ptr + 5 * src_stride));
    const __m128i s6 =
        _mm_loadl_epi64((const __m128i *)(src_ptr + 6 * src_stride));
    const __m128i s7 =
        _mm_loadl_epi64((const __m128i *)(src_ptr + 7 * src_stride));
    const __m128i s8 =
        _mm_loadl_epi64((const __m128i *)(src_ptr + 8 * src_stride));
    const __m128i s9 =
        _mm_loadl_epi64((const __m128i *)(src_ptr + 9 * src_stride));
    const __m128i s10t =
        _mm_loadl_epi64((const __m128i *)(src_ptr + 10 * src_stride));

    const __m256i r01 =
        _mm256_inserti128_si256(_mm256_castsi128_si256(s0), s1, 1);
    const __m256i r12 =
        _mm256_inserti128_si256(_mm256_castsi128_si256(s1), s2, 1);
    const __m256i r23 =
        _mm256_inserti128_si256(_mm256_castsi128_si256(s2), s3, 1);
    const __m256i r34 =
        _mm256_inserti128_si256(_mm256_castsi128_si256(s3), s4, 1);
    const __m256i r45 =
        _mm256_inserti128_si256(_mm256_castsi128_si256(s4), s5, 1);
    const __m256i r56 =
        _mm256_inserti128_si256(_mm256_castsi128_si256(s5), s6, 1);
    const __m256i r67 =
        _mm256_inserti128_si256(_mm256_castsi128_si256(s6), s7, 1);
    const __m256i r78 =
        _mm256_inserti128_si256(_mm256_castsi128_si256(s7), s8, 1);
    const __m256i r89 =
        _mm256_inserti128_si256(_mm256_castsi128_si256(s8), s9, 1);
    const __m256i r910 =
        _mm256_inserti128_si256(_mm256_castsi128_si256(s9), s10t, 1);

    s[0] = _mm256_unpacklo_epi8(r01, r12);
    s[1] = _mm256_unpacklo_epi8(r23, r34);
    s[2] = _mm256_unpacklo_epi8(r45, r56);
    s[3] = _mm256_unpacklo_epi8(r67, r78);
    s[4] = _mm256_unpacklo_epi8(r89, r910);
    for (int i = 0; i < h; i += 2) {
      const __m128i s10 =
          _mm_loadl_epi64((const __m128i *)(src_ptr + (i + 10) * src_stride));
      const __m128i s11 =
          _mm_loadl_epi64((const __m128i *)(src_ptr + (i + 11) * src_stride));
      const __m128i s12 =
          _mm_loadl_epi64((const __m128i *)(src_ptr + (i + 12) * src_stride));

      const __m256i r1011 =
          _mm256_inserti128_si256(_mm256_castsi128_si256(s10), s11, 1);
      const __m256i r1112 =
          _mm256_inserti128_si256(_mm256_castsi128_si256(s11), s12, 1);
      s[5] = _mm256_unpacklo_epi8(r1011, r1112);
      const __m256i res_0 = convolve12_16_avx2(s, f);

      __m256i res = _mm256_packus_epi16(res_0, res_0);

      _mm_storel_epi64((__m128i *)&dst[i * dst_stride],
                       _mm256_castsi256_si128(res));
      _mm_storel_epi64((__m128i *)&dst[(i + 1) * dst_stride],
                       _mm256_extracti128_si256(res, 1));

      reuse_src_data_avx2(s + 1, s);
    }
  } else {
    for (int j = 0; j < w; j += 16) {
      const __m128i s0 =
          _mm_loadu_si128((const __m128i *)(src_ptr + 0 * src_stride + j));
      const __m128i s1 =
          _mm_loadu_si128((const __m128i *)(src_ptr + 1 * src_stride + j));
      const __m128i s2 =
          _mm_loadu_si128((const __m128i *)(src_ptr + 2 * src_stride + j));
      const __m128i s3 =
          _mm_loadu_si128((const __m128i *)(src_ptr + 3 * src_stride + j));
      const __m128i s4 =
          _mm_loadu_si128((const __m128i *)(src_ptr + 4 * src_stride + j));
      const __m128i s5 =
          _mm_loadu_si128((const __m128i *)(src_ptr + 5 * src_stride + j));
      const __m128i s6 =
          _mm_loadu_si128((const __m128i *)(src_ptr + 6 * src_stride + j));
      const __m128i s7 =
          _mm_loadu_si128((const __m128i *)(src_ptr + 7 * src_stride + j));
      const __m128i s8 =
          _mm_loadu_si128((const __m128i *)(src_ptr + 8 * src_stride + j));
      const __m128i s9 =
          _mm_loadu_si128((const __m128i *)(src_ptr + 9 * src_stride + j));
      const __m128i s10t =
          _mm_loadu_si128((const __m128i *)(src_ptr + 10 * src_stride + j));

      const __m256i r01 =
          _mm256_inserti128_si256(_mm256_castsi128_si256(s0), s1, 1);
      const __m256i r12 =
          _mm256_inserti128_si256(_mm256_castsi128_si256(s1), s2, 1);
      const __m256i r23 =
          _mm256_inserti128_si256(_mm256_castsi128_si256(s2), s3, 1);
      const __m256i r34 =
          _mm256_inserti128_si256(_mm256_castsi128_si256(s3), s4, 1);
      const __m256i r45 =
          _mm256_inserti128_si256(_mm256_castsi128_si256(s4), s5, 1);
      const __m256i r56 =
          _mm256_inserti128_si256(_mm256_castsi128_si256(s5), s6, 1);
      const __m256i r67 =
          _mm256_inserti128_si256(_mm256_castsi128_si256(s6), s7, 1);
      const __m256i r78 =
          _mm256_inserti128_si256(_mm256_castsi128_si256(s7), s8, 1);
      const __m256i r89 =
          _mm256_inserti128_si256(_mm256_castsi128_si256(s8), s9, 1);
      const __m256i r910 =
          _mm256_inserti128_si256(_mm256_castsi128_si256(s9), s10t, 1);

      s[0] = _mm256_unpacklo_epi8(r01, r12);
      s[1] = _mm256_unpacklo_epi8(r23, r34);
      s[2] = _mm256_unpacklo_epi8(r45, r56);
      s[3] = _mm256_unpacklo_epi8(r67, r78);
      s[4] = _mm256_unpacklo_epi8(r89, r910);

      s[6] = _mm256_unpackhi_epi8(r01, r12);
      s[7] = _mm256_unpackhi_epi8(r23, r34);
      s[8] = _mm256_unpackhi_epi8(r45, r56);
      s[9] = _mm256_unpackhi_epi8(r67, r78);
      s[10] = _mm256_unpackhi_epi8(r89, r910);
      for (int i = 0; i < h; i += 2) {
        const __m128i s10 = _mm_loadu_si128(
            (const __m128i *)(src_ptr + (i + 10) * src_stride + j));
        const __m128i s11 = _mm_loadu_si128(
            (const __m128i *)(src_ptr + (i + 11) * src_stride + j));
        const __m128i s12 = _mm_loadu_si128(
            (const __m128i *)(src_ptr + (i + 12) * src_stride + j));

        const __m256i r1011 =
            _mm256_inserti128_si256(_mm256_castsi128_si256(s10), s11, 1);
        const __m256i r1112 =
            _mm256_inserti128_si256(_mm256_castsi128_si256(s11), s12, 1);

        s[5] = _mm256_unpacklo_epi8(r1011, r1112);
        s[11] = _mm256_unpackhi_epi8(r1011, r1112);

        const __m256i res_0 = convolve12_16_avx2(s, f);
        const __m256i res_1 = convolve12_16_avx2(s + 6, f);

        __m256i res = _mm256_packus_epi16(res_0, res_1);

        _mm_storeu_si128((__m128i *)&dst[i * dst_stride + j],
                         _mm256_castsi256_si128(res));
        _mm_storeu_si128((__m128i *)&dst[(i + 1) * dst_stride + j],
                         _mm256_extracti128_si256(res, 1));

        reuse_src_data_avx2(s + 1, s);
        reuse_src_data_avx2(s + 7, s + 6);
      }
    }
  }
}

void vpx_convolve12_avx2(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst,
                         ptrdiff_t dst_stride, const InterpKernel12 *filter,
                         int x0_q4, int x_step_q4, int y0_q4, int y_step_q4,
                         int w, int h) {
  assert(x_step_q4 == 16 && y_step_q4 == 16);
  assert(h == 32 || h == 16 || h == 8);
  assert(w == 32 || w == 16 || w == 8);
  DECLARE_ALIGNED(32, uint8_t, temp[BW * (BH + MAX_FILTER_TAP - 1)]);
  const int temp_stride = BW;
  const int intermediate_height =
      (((h - 1) * y_step_q4 + y0_q4) >> SUBPEL_BITS) + MAX_FILTER_TAP;
  vpx_convolve12_horiz_avx2(src - src_stride * (MAX_FILTER_TAP / 2 - 1),
                            src_stride, temp, temp_stride, filter, x0_q4,
                            x_step_q4, y0_q4, y_step_q4, w,
                            intermediate_height);
  vpx_convolve12_vert_avx2(temp + temp_stride * (MAX_FILTER_TAP / 2 - 1),
                           temp_stride, dst, dst_stride, filter, x0_q4,
                           x_step_q4, y0_q4, y_step_q4, w, h);
}
