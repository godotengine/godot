/*
 *  Copyright (c) 2023 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>
#include <smmintrin.h>

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"

#include "vpx_ports/mem.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/x86/mem_sse2.h"

static INLINE int64_t summary_all_sse4(const __m128i *sum_all) {
  int64_t sum;
  const __m128i sum0 = _mm_cvtepu32_epi64(*sum_all);
  const __m128i sum1 = _mm_cvtepu32_epi64(_mm_srli_si128(*sum_all, 8));
  const __m128i sum_2x64 = _mm_add_epi64(sum0, sum1);
  const __m128i sum_1x64 = _mm_add_epi64(sum_2x64, _mm_srli_si128(sum_2x64, 8));
  _mm_storel_epi64((__m128i *)&sum, sum_1x64);
  return sum;
}

#if CONFIG_VP9_HIGHBITDEPTH
static INLINE void summary_32_sse4(const __m128i *sum32, __m128i *sum64) {
  const __m128i sum0 = _mm_cvtepu32_epi64(*sum32);
  const __m128i sum1 = _mm_cvtepu32_epi64(_mm_srli_si128(*sum32, 8));
  *sum64 = _mm_add_epi64(sum0, *sum64);
  *sum64 = _mm_add_epi64(sum1, *sum64);
}
#endif

static INLINE void sse_w16_sse4_1(__m128i *sum, const uint8_t *a,
                                  const uint8_t *b) {
  const __m128i v_a0 = _mm_loadu_si128((const __m128i *)a);
  const __m128i v_b0 = _mm_loadu_si128((const __m128i *)b);
  const __m128i v_a00_w = _mm_cvtepu8_epi16(v_a0);
  const __m128i v_a01_w = _mm_cvtepu8_epi16(_mm_srli_si128(v_a0, 8));
  const __m128i v_b00_w = _mm_cvtepu8_epi16(v_b0);
  const __m128i v_b01_w = _mm_cvtepu8_epi16(_mm_srli_si128(v_b0, 8));
  const __m128i v_d00_w = _mm_sub_epi16(v_a00_w, v_b00_w);
  const __m128i v_d01_w = _mm_sub_epi16(v_a01_w, v_b01_w);
  *sum = _mm_add_epi32(*sum, _mm_madd_epi16(v_d00_w, v_d00_w));
  *sum = _mm_add_epi32(*sum, _mm_madd_epi16(v_d01_w, v_d01_w));
}

static INLINE void sse4x2_sse4_1(const uint8_t *a, int a_stride,
                                 const uint8_t *b, int b_stride, __m128i *sum) {
  const __m128i v_a0 = load_unaligned_u32(a);
  const __m128i v_a1 = load_unaligned_u32(a + a_stride);
  const __m128i v_b0 = load_unaligned_u32(b);
  const __m128i v_b1 = load_unaligned_u32(b + b_stride);
  const __m128i v_a_w = _mm_cvtepu8_epi16(_mm_unpacklo_epi32(v_a0, v_a1));
  const __m128i v_b_w = _mm_cvtepu8_epi16(_mm_unpacklo_epi32(v_b0, v_b1));
  const __m128i v_d_w = _mm_sub_epi16(v_a_w, v_b_w);
  *sum = _mm_add_epi32(*sum, _mm_madd_epi16(v_d_w, v_d_w));
}

static INLINE void sse8_sse4_1(const uint8_t *a, const uint8_t *b,
                               __m128i *sum) {
  const __m128i v_a0 = _mm_loadl_epi64((const __m128i *)a);
  const __m128i v_b0 = _mm_loadl_epi64((const __m128i *)b);
  const __m128i v_a_w = _mm_cvtepu8_epi16(v_a0);
  const __m128i v_b_w = _mm_cvtepu8_epi16(v_b0);
  const __m128i v_d_w = _mm_sub_epi16(v_a_w, v_b_w);
  *sum = _mm_add_epi32(*sum, _mm_madd_epi16(v_d_w, v_d_w));
}

int64_t vpx_sse_sse4_1(const uint8_t *a, int a_stride, const uint8_t *b,
                       int b_stride, int width, int height) {
  int y = 0;
  int64_t sse = 0;
  __m128i sum = _mm_setzero_si128();
  switch (width) {
    case 4:
      do {
        sse4x2_sse4_1(a, a_stride, b, b_stride, &sum);
        a += a_stride << 1;
        b += b_stride << 1;
        y += 2;
      } while (y < height);
      sse = summary_all_sse4(&sum);
      break;
    case 8:
      do {
        sse8_sse4_1(a, b, &sum);
        a += a_stride;
        b += b_stride;
        y += 1;
      } while (y < height);
      sse = summary_all_sse4(&sum);
      break;
    case 16:
      do {
        sse_w16_sse4_1(&sum, a, b);
        a += a_stride;
        b += b_stride;
        y += 1;
      } while (y < height);
      sse = summary_all_sse4(&sum);
      break;
    case 32:
      do {
        sse_w16_sse4_1(&sum, a, b);
        sse_w16_sse4_1(&sum, a + 16, b + 16);
        a += a_stride;
        b += b_stride;
        y += 1;
      } while (y < height);
      sse = summary_all_sse4(&sum);
      break;
    case 64:
      do {
        sse_w16_sse4_1(&sum, a, b);
        sse_w16_sse4_1(&sum, a + 16 * 1, b + 16 * 1);
        sse_w16_sse4_1(&sum, a + 16 * 2, b + 16 * 2);
        sse_w16_sse4_1(&sum, a + 16 * 3, b + 16 * 3);
        a += a_stride;
        b += b_stride;
        y += 1;
      } while (y < height);
      sse = summary_all_sse4(&sum);
      break;
    default:
      if (width & 0x07) {
        do {
          int i = 0;
          do {
            sse8_sse4_1(a + i, b + i, &sum);
            sse8_sse4_1(a + i + a_stride, b + i + b_stride, &sum);
            i += 8;
          } while (i + 4 < width);
          sse4x2_sse4_1(a + i, a_stride, b + i, b_stride, &sum);
          a += (a_stride << 1);
          b += (b_stride << 1);
          y += 2;
        } while (y < height);
      } else {
        do {
          int i = 0;
          do {
            sse8_sse4_1(a + i, b + i, &sum);
            i += 8;
          } while (i < width);
          a += a_stride;
          b += b_stride;
          y += 1;
        } while (y < height);
      }
      sse = summary_all_sse4(&sum);
      break;
  }

  return sse;
}

#if CONFIG_VP9_HIGHBITDEPTH
static INLINE void highbd_sse_w4x2_sse4_1(__m128i *sum, const uint16_t *a,
                                          int a_stride, const uint16_t *b,
                                          int b_stride) {
  const __m128i v_a0 = _mm_loadl_epi64((const __m128i *)a);
  const __m128i v_a1 = _mm_loadl_epi64((const __m128i *)(a + a_stride));
  const __m128i v_b0 = _mm_loadl_epi64((const __m128i *)b);
  const __m128i v_b1 = _mm_loadl_epi64((const __m128i *)(b + b_stride));
  const __m128i v_a_w = _mm_unpacklo_epi64(v_a0, v_a1);
  const __m128i v_b_w = _mm_unpacklo_epi64(v_b0, v_b1);
  const __m128i v_d_w = _mm_sub_epi16(v_a_w, v_b_w);
  *sum = _mm_add_epi32(*sum, _mm_madd_epi16(v_d_w, v_d_w));
}

static INLINE void highbd_sse_w8_sse4_1(__m128i *sum, const uint16_t *a,
                                        const uint16_t *b) {
  const __m128i v_a_w = _mm_loadu_si128((const __m128i *)a);
  const __m128i v_b_w = _mm_loadu_si128((const __m128i *)b);
  const __m128i v_d_w = _mm_sub_epi16(v_a_w, v_b_w);
  *sum = _mm_add_epi32(*sum, _mm_madd_epi16(v_d_w, v_d_w));
}

int64_t vpx_highbd_sse_sse4_1(const uint8_t *a8, int a_stride,
                              const uint8_t *b8, int b_stride, int width,
                              int height) {
  int32_t y = 0;
  int64_t sse = 0;
  uint16_t *a = CONVERT_TO_SHORTPTR(a8);
  uint16_t *b = CONVERT_TO_SHORTPTR(b8);
  __m128i sum = _mm_setzero_si128();
  switch (width) {
    case 4:
      do {
        highbd_sse_w4x2_sse4_1(&sum, a, a_stride, b, b_stride);
        a += a_stride << 1;
        b += b_stride << 1;
        y += 2;
      } while (y < height);
      sse = summary_all_sse4(&sum);
      break;
    case 8:
      do {
        highbd_sse_w8_sse4_1(&sum, a, b);
        a += a_stride;
        b += b_stride;
        y += 1;
      } while (y < height);
      sse = summary_all_sse4(&sum);
      break;
    case 16:
      do {
        int l = 0;
        __m128i sum32 = _mm_setzero_si128();
        do {
          highbd_sse_w8_sse4_1(&sum32, a, b);
          highbd_sse_w8_sse4_1(&sum32, a + 8, b + 8);
          a += a_stride;
          b += b_stride;
          l += 1;
        } while (l < 64 && l < (height - y));
        summary_32_sse4(&sum32, &sum);
        y += 64;
      } while (y < height);
      _mm_storel_epi64((__m128i *)&sse,
                       _mm_add_epi64(sum, _mm_srli_si128(sum, 8)));
      break;
    case 32:
      do {
        int l = 0;
        __m128i sum32 = _mm_setzero_si128();
        do {
          highbd_sse_w8_sse4_1(&sum32, a, b);
          highbd_sse_w8_sse4_1(&sum32, a + 8 * 1, b + 8 * 1);
          highbd_sse_w8_sse4_1(&sum32, a + 8 * 2, b + 8 * 2);
          highbd_sse_w8_sse4_1(&sum32, a + 8 * 3, b + 8 * 3);
          a += a_stride;
          b += b_stride;
          l += 1;
        } while (l < 32 && l < (height - y));
        summary_32_sse4(&sum32, &sum);
        y += 32;
      } while (y < height);
      _mm_storel_epi64((__m128i *)&sse,
                       _mm_add_epi64(sum, _mm_srli_si128(sum, 8)));
      break;
    case 64:
      do {
        int l = 0;
        __m128i sum32 = _mm_setzero_si128();
        do {
          highbd_sse_w8_sse4_1(&sum32, a, b);
          highbd_sse_w8_sse4_1(&sum32, a + 8 * 1, b + 8 * 1);
          highbd_sse_w8_sse4_1(&sum32, a + 8 * 2, b + 8 * 2);
          highbd_sse_w8_sse4_1(&sum32, a + 8 * 3, b + 8 * 3);
          highbd_sse_w8_sse4_1(&sum32, a + 8 * 4, b + 8 * 4);
          highbd_sse_w8_sse4_1(&sum32, a + 8 * 5, b + 8 * 5);
          highbd_sse_w8_sse4_1(&sum32, a + 8 * 6, b + 8 * 6);
          highbd_sse_w8_sse4_1(&sum32, a + 8 * 7, b + 8 * 7);
          a += a_stride;
          b += b_stride;
          l += 1;
        } while (l < 16 && l < (height - y));
        summary_32_sse4(&sum32, &sum);
        y += 16;
      } while (y < height);
      _mm_storel_epi64((__m128i *)&sse,
                       _mm_add_epi64(sum, _mm_srli_si128(sum, 8)));
      break;
    default:
      if (width & 0x7) {
        do {
          __m128i sum32 = _mm_setzero_si128();
          int i = 0;
          do {
            highbd_sse_w8_sse4_1(&sum32, a + i, b + i);
            highbd_sse_w8_sse4_1(&sum32, a + i + a_stride, b + i + b_stride);
            i += 8;
          } while (i + 4 < width);
          highbd_sse_w4x2_sse4_1(&sum32, a + i, a_stride, b + i, b_stride);
          a += (a_stride << 1);
          b += (b_stride << 1);
          y += 2;
          summary_32_sse4(&sum32, &sum);
        } while (y < height);
      } else {
        do {
          int l = 0;
          __m128i sum32 = _mm_setzero_si128();
          do {
            int i = 0;
            do {
              highbd_sse_w8_sse4_1(&sum32, a + i, b + i);
              i += 8;
            } while (i < width);
            a += a_stride;
            b += b_stride;
            l += 1;
          } while (l < 8 && l < (height - y));
          summary_32_sse4(&sum32, &sum);
          y += 8;
        } while (y < height);
      }
      _mm_storel_epi64((__m128i *)&sse,
                       _mm_add_epi64(sum, _mm_srli_si128(sum, 8)));
      break;
  }
  return sse;
}
#endif  // CONFIG_VP9_HIGHBITDEPTH
