/*
 *  Copyright (c) 2023 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <immintrin.h>
#include <smmintrin.h>
#include <stdint.h>

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"

#include "vpx_ports/mem.h"
#include "vpx_dsp/x86/mem_sse2.h"

static INLINE void sse_w32_avx2(__m256i *sum, const uint8_t *a,
                                const uint8_t *b) {
  const __m256i v_a0 = _mm256_loadu_si256((const __m256i *)a);
  const __m256i v_b0 = _mm256_loadu_si256((const __m256i *)b);
  const __m256i zero = _mm256_setzero_si256();
  const __m256i v_a00_w = _mm256_unpacklo_epi8(v_a0, zero);
  const __m256i v_a01_w = _mm256_unpackhi_epi8(v_a0, zero);
  const __m256i v_b00_w = _mm256_unpacklo_epi8(v_b0, zero);
  const __m256i v_b01_w = _mm256_unpackhi_epi8(v_b0, zero);
  const __m256i v_d00_w = _mm256_sub_epi16(v_a00_w, v_b00_w);
  const __m256i v_d01_w = _mm256_sub_epi16(v_a01_w, v_b01_w);
  *sum = _mm256_add_epi32(*sum, _mm256_madd_epi16(v_d00_w, v_d00_w));
  *sum = _mm256_add_epi32(*sum, _mm256_madd_epi16(v_d01_w, v_d01_w));
}

static INLINE int64_t summary_all_avx2(const __m256i *sum_all) {
  int64_t sum;
  __m256i zero = _mm256_setzero_si256();
  const __m256i sum0_4x64 = _mm256_unpacklo_epi32(*sum_all, zero);
  const __m256i sum1_4x64 = _mm256_unpackhi_epi32(*sum_all, zero);
  const __m256i sum_4x64 = _mm256_add_epi64(sum0_4x64, sum1_4x64);
  const __m128i sum_2x64 = _mm_add_epi64(_mm256_castsi256_si128(sum_4x64),
                                         _mm256_extracti128_si256(sum_4x64, 1));
  const __m128i sum_1x64 = _mm_add_epi64(sum_2x64, _mm_srli_si128(sum_2x64, 8));
  _mm_storel_epi64((__m128i *)&sum, sum_1x64);
  return sum;
}

#if CONFIG_VP9_HIGHBITDEPTH
static INLINE void summary_32_avx2(const __m256i *sum32, __m256i *sum) {
  const __m256i sum0_4x64 =
      _mm256_cvtepu32_epi64(_mm256_castsi256_si128(*sum32));
  const __m256i sum1_4x64 =
      _mm256_cvtepu32_epi64(_mm256_extracti128_si256(*sum32, 1));
  const __m256i sum_4x64 = _mm256_add_epi64(sum0_4x64, sum1_4x64);
  *sum = _mm256_add_epi64(*sum, sum_4x64);
}

static INLINE int64_t summary_4x64_avx2(const __m256i sum_4x64) {
  int64_t sum;
  const __m128i sum_2x64 = _mm_add_epi64(_mm256_castsi256_si128(sum_4x64),
                                         _mm256_extracti128_si256(sum_4x64, 1));
  const __m128i sum_1x64 = _mm_add_epi64(sum_2x64, _mm_srli_si128(sum_2x64, 8));

  _mm_storel_epi64((__m128i *)&sum, sum_1x64);
  return sum;
}
#endif

static INLINE void sse_w4x4_avx2(const uint8_t *a, int a_stride,
                                 const uint8_t *b, int b_stride, __m256i *sum) {
  const __m128i v_a0 = load_unaligned_u32(a);
  const __m128i v_a1 = load_unaligned_u32(a + a_stride);
  const __m128i v_a2 = load_unaligned_u32(a + a_stride * 2);
  const __m128i v_a3 = load_unaligned_u32(a + a_stride * 3);
  const __m128i v_b0 = load_unaligned_u32(b);
  const __m128i v_b1 = load_unaligned_u32(b + b_stride);
  const __m128i v_b2 = load_unaligned_u32(b + b_stride * 2);
  const __m128i v_b3 = load_unaligned_u32(b + b_stride * 3);
  const __m128i v_a0123 = _mm_unpacklo_epi64(_mm_unpacklo_epi32(v_a0, v_a1),
                                             _mm_unpacklo_epi32(v_a2, v_a3));
  const __m128i v_b0123 = _mm_unpacklo_epi64(_mm_unpacklo_epi32(v_b0, v_b1),
                                             _mm_unpacklo_epi32(v_b2, v_b3));
  const __m256i v_a_w = _mm256_cvtepu8_epi16(v_a0123);
  const __m256i v_b_w = _mm256_cvtepu8_epi16(v_b0123);
  const __m256i v_d_w = _mm256_sub_epi16(v_a_w, v_b_w);
  *sum = _mm256_add_epi32(*sum, _mm256_madd_epi16(v_d_w, v_d_w));
}

static INLINE void sse_w8x2_avx2(const uint8_t *a, int a_stride,
                                 const uint8_t *b, int b_stride, __m256i *sum) {
  const __m128i v_a0 = _mm_loadl_epi64((const __m128i *)a);
  const __m128i v_a1 = _mm_loadl_epi64((const __m128i *)(a + a_stride));
  const __m128i v_b0 = _mm_loadl_epi64((const __m128i *)b);
  const __m128i v_b1 = _mm_loadl_epi64((const __m128i *)(b + b_stride));
  const __m256i v_a_w = _mm256_cvtepu8_epi16(_mm_unpacklo_epi64(v_a0, v_a1));
  const __m256i v_b_w = _mm256_cvtepu8_epi16(_mm_unpacklo_epi64(v_b0, v_b1));
  const __m256i v_d_w = _mm256_sub_epi16(v_a_w, v_b_w);
  *sum = _mm256_add_epi32(*sum, _mm256_madd_epi16(v_d_w, v_d_w));
}

int64_t vpx_sse_avx2(const uint8_t *a, int a_stride, const uint8_t *b,
                     int b_stride, int width, int height) {
  int32_t y = 0;
  int64_t sse = 0;
  __m256i sum = _mm256_setzero_si256();
  __m256i zero = _mm256_setzero_si256();
  switch (width) {
    case 4:
      do {
        sse_w4x4_avx2(a, a_stride, b, b_stride, &sum);
        a += a_stride << 2;
        b += b_stride << 2;
        y += 4;
      } while (y < height);
      sse = summary_all_avx2(&sum);
      break;
    case 8:
      do {
        sse_w8x2_avx2(a, a_stride, b, b_stride, &sum);
        a += a_stride << 1;
        b += b_stride << 1;
        y += 2;
      } while (y < height);
      sse = summary_all_avx2(&sum);
      break;
    case 16:
      do {
        const __m128i v_a0 = _mm_loadu_si128((const __m128i *)a);
        const __m128i v_a1 = _mm_loadu_si128((const __m128i *)(a + a_stride));
        const __m128i v_b0 = _mm_loadu_si128((const __m128i *)b);
        const __m128i v_b1 = _mm_loadu_si128((const __m128i *)(b + b_stride));
        const __m256i v_a =
            _mm256_insertf128_si256(_mm256_castsi128_si256(v_a0), v_a1, 0x01);
        const __m256i v_b =
            _mm256_insertf128_si256(_mm256_castsi128_si256(v_b0), v_b1, 0x01);
        const __m256i v_al = _mm256_unpacklo_epi8(v_a, zero);
        const __m256i v_au = _mm256_unpackhi_epi8(v_a, zero);
        const __m256i v_bl = _mm256_unpacklo_epi8(v_b, zero);
        const __m256i v_bu = _mm256_unpackhi_epi8(v_b, zero);
        const __m256i v_asub = _mm256_sub_epi16(v_al, v_bl);
        const __m256i v_bsub = _mm256_sub_epi16(v_au, v_bu);
        const __m256i temp =
            _mm256_add_epi32(_mm256_madd_epi16(v_asub, v_asub),
                             _mm256_madd_epi16(v_bsub, v_bsub));
        sum = _mm256_add_epi32(sum, temp);
        a += a_stride << 1;
        b += b_stride << 1;
        y += 2;
      } while (y < height);
      sse = summary_all_avx2(&sum);
      break;
    case 32:
      do {
        sse_w32_avx2(&sum, a, b);
        a += a_stride;
        b += b_stride;
        y += 1;
      } while (y < height);
      sse = summary_all_avx2(&sum);
      break;
    case 64:
      do {
        sse_w32_avx2(&sum, a, b);
        sse_w32_avx2(&sum, a + 32, b + 32);
        a += a_stride;
        b += b_stride;
        y += 1;
      } while (y < height);
      sse = summary_all_avx2(&sum);
      break;
    default:
      if ((width & 0x07) == 0) {
        do {
          int i = 0;
          do {
            sse_w8x2_avx2(a + i, a_stride, b + i, b_stride, &sum);
            i += 8;
          } while (i < width);
          a += a_stride << 1;
          b += b_stride << 1;
          y += 2;
        } while (y < height);
      } else {
        do {
          int i = 0;
          do {
            const uint8_t *a2;
            const uint8_t *b2;
            sse_w8x2_avx2(a + i, a_stride, b + i, b_stride, &sum);
            a2 = a + i + (a_stride << 1);
            b2 = b + i + (b_stride << 1);
            sse_w8x2_avx2(a2, a_stride, b2, b_stride, &sum);
            i += 8;
          } while (i + 4 < width);
          sse_w4x4_avx2(a + i, a_stride, b + i, b_stride, &sum);
          a += a_stride << 2;
          b += b_stride << 2;
          y += 4;
        } while (y < height);
      }
      sse = summary_all_avx2(&sum);
      break;
  }

  return sse;
}

#if CONFIG_VP9_HIGHBITDEPTH
static INLINE void highbd_sse_w16_avx2(__m256i *sum, const uint16_t *a,
                                       const uint16_t *b) {
  const __m256i v_a_w = _mm256_loadu_si256((const __m256i *)a);
  const __m256i v_b_w = _mm256_loadu_si256((const __m256i *)b);
  const __m256i v_d_w = _mm256_sub_epi16(v_a_w, v_b_w);
  *sum = _mm256_add_epi32(*sum, _mm256_madd_epi16(v_d_w, v_d_w));
}

static INLINE void highbd_sse_w4x4_avx2(__m256i *sum, const uint16_t *a,
                                        int a_stride, const uint16_t *b,
                                        int b_stride) {
  const __m128i v_a0 = _mm_loadl_epi64((const __m128i *)a);
  const __m128i v_a1 = _mm_loadl_epi64((const __m128i *)(a + a_stride));
  const __m128i v_a2 = _mm_loadl_epi64((const __m128i *)(a + a_stride * 2));
  const __m128i v_a3 = _mm_loadl_epi64((const __m128i *)(a + a_stride * 3));
  const __m128i v_b0 = _mm_loadl_epi64((const __m128i *)b);
  const __m128i v_b1 = _mm_loadl_epi64((const __m128i *)(b + b_stride));
  const __m128i v_b2 = _mm_loadl_epi64((const __m128i *)(b + b_stride * 2));
  const __m128i v_b3 = _mm_loadl_epi64((const __m128i *)(b + b_stride * 3));
  const __m128i v_a_hi = _mm_unpacklo_epi64(v_a0, v_a1);
  const __m128i v_a_lo = _mm_unpacklo_epi64(v_a2, v_a3);
  const __m256i v_a_w =
      _mm256_insertf128_si256(_mm256_castsi128_si256(v_a_lo), v_a_hi, 1);
  const __m128i v_b_hi = _mm_unpacklo_epi64(v_b0, v_b1);
  const __m128i v_b_lo = _mm_unpacklo_epi64(v_b2, v_b3);
  const __m256i v_b_w =
      _mm256_insertf128_si256(_mm256_castsi128_si256(v_b_lo), v_b_hi, 1);
  const __m256i v_d_w = _mm256_sub_epi16(v_a_w, v_b_w);
  *sum = _mm256_add_epi32(*sum, _mm256_madd_epi16(v_d_w, v_d_w));
}

static INLINE void highbd_sse_w8x2_avx2(__m256i *sum, const uint16_t *a,
                                        int a_stride, const uint16_t *b,
                                        int b_stride) {
  const __m128i v_a_hi = _mm_loadu_si128((const __m128i *)(a + a_stride));
  const __m128i v_a_lo = _mm_loadu_si128((const __m128i *)a);
  const __m256i v_a_w =
      _mm256_insertf128_si256(_mm256_castsi128_si256(v_a_lo), v_a_hi, 1);
  const __m128i v_b_hi = _mm_loadu_si128((const __m128i *)(b + b_stride));
  const __m128i v_b_lo = _mm_loadu_si128((const __m128i *)b);
  const __m256i v_b_w =
      _mm256_insertf128_si256(_mm256_castsi128_si256(v_b_lo), v_b_hi, 1);
  const __m256i v_d_w = _mm256_sub_epi16(v_a_w, v_b_w);
  *sum = _mm256_add_epi32(*sum, _mm256_madd_epi16(v_d_w, v_d_w));
}

int64_t vpx_highbd_sse_avx2(const uint8_t *a8, int a_stride, const uint8_t *b8,
                            int b_stride, int width, int height) {
  int32_t y = 0;
  int64_t sse = 0;
  uint16_t *a = CONVERT_TO_SHORTPTR(a8);
  uint16_t *b = CONVERT_TO_SHORTPTR(b8);
  __m256i sum = _mm256_setzero_si256();
  switch (width) {
    case 4:
      do {
        highbd_sse_w4x4_avx2(&sum, a, a_stride, b, b_stride);
        a += a_stride << 2;
        b += b_stride << 2;
        y += 4;
      } while (y < height);
      sse = summary_all_avx2(&sum);
      break;
    case 8:
      do {
        highbd_sse_w8x2_avx2(&sum, a, a_stride, b, b_stride);
        a += a_stride << 1;
        b += b_stride << 1;
        y += 2;
      } while (y < height);
      sse = summary_all_avx2(&sum);
      break;
    case 16:
      do {
        highbd_sse_w16_avx2(&sum, a, b);
        a += a_stride;
        b += b_stride;
        y += 1;
      } while (y < height);
      sse = summary_all_avx2(&sum);
      break;
    case 32:
      do {
        int l = 0;
        __m256i sum32 = _mm256_setzero_si256();
        do {
          highbd_sse_w16_avx2(&sum32, a, b);
          highbd_sse_w16_avx2(&sum32, a + 16, b + 16);
          a += a_stride;
          b += b_stride;
          l += 1;
        } while (l < 64 && l < (height - y));
        summary_32_avx2(&sum32, &sum);
        y += 64;
      } while (y < height);
      sse = summary_4x64_avx2(sum);
      break;
    case 64:
      do {
        int l = 0;
        __m256i sum32 = _mm256_setzero_si256();
        do {
          highbd_sse_w16_avx2(&sum32, a, b);
          highbd_sse_w16_avx2(&sum32, a + 16 * 1, b + 16 * 1);
          highbd_sse_w16_avx2(&sum32, a + 16 * 2, b + 16 * 2);
          highbd_sse_w16_avx2(&sum32, a + 16 * 3, b + 16 * 3);
          a += a_stride;
          b += b_stride;
          l += 1;
        } while (l < 32 && l < (height - y));
        summary_32_avx2(&sum32, &sum);
        y += 32;
      } while (y < height);
      sse = summary_4x64_avx2(sum);
      break;
    default:
      if (width & 0x7) {
        do {
          int i = 0;
          __m256i sum32 = _mm256_setzero_si256();
          do {
            const uint16_t *a2;
            const uint16_t *b2;
            highbd_sse_w8x2_avx2(&sum32, a + i, a_stride, b + i, b_stride);
            a2 = a + i + (a_stride << 1);
            b2 = b + i + (b_stride << 1);
            highbd_sse_w8x2_avx2(&sum32, a2, a_stride, b2, b_stride);
            i += 8;
          } while (i + 4 < width);
          highbd_sse_w4x4_avx2(&sum32, a + i, a_stride, b + i, b_stride);
          summary_32_avx2(&sum32, &sum);
          a += a_stride << 2;
          b += b_stride << 2;
          y += 4;
        } while (y < height);
      } else {
        do {
          int l = 0;
          __m256i sum32 = _mm256_setzero_si256();
          do {
            int i = 0;
            do {
              highbd_sse_w8x2_avx2(&sum32, a + i, a_stride, b + i, b_stride);
              i += 8;
            } while (i < width);
            a += a_stride << 1;
            b += b_stride << 1;
            l += 2;
          } while (l < 8 && l < (height - y));
          summary_32_avx2(&sum32, &sum);
          y += 8;
        } while (y < height);
      }
      sse = summary_4x64_avx2(sum);
      break;
  }
  return sse;
}
#endif  // CONFIG_VP9_HIGHBITDEPTH
