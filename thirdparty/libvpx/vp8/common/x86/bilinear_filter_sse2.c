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
#include <xmmintrin.h>

#include "./vp8_rtcd.h"
#include "./vpx_config.h"
#include "vp8/common/filter.h"
#include "vpx_dsp/x86/mem_sse2.h"
#include "vpx_ports/mem.h"

static INLINE void horizontal_16x16(uint8_t *src, const int stride,
                                    uint16_t *dst, const int xoffset) {
  int h;
  const __m128i zero = _mm_setzero_si128();

  if (xoffset == 0) {
    for (h = 0; h < 17; ++h) {
      const __m128i a = _mm_loadu_si128((__m128i *)src);
      const __m128i a_lo = _mm_unpacklo_epi8(a, zero);
      const __m128i a_hi = _mm_unpackhi_epi8(a, zero);
      _mm_store_si128((__m128i *)dst, a_lo);
      _mm_store_si128((__m128i *)(dst + 8), a_hi);
      src += stride;
      dst += 16;
    }
    return;
  }

  {
    const __m128i round_factor = _mm_set1_epi16(1 << (VP8_FILTER_SHIFT - 1));
    const __m128i hfilter_0 = _mm_set1_epi16(vp8_bilinear_filters[xoffset][0]);
    const __m128i hfilter_1 = _mm_set1_epi16(vp8_bilinear_filters[xoffset][1]);

    for (h = 0; h < 17; ++h) {
      const __m128i a = _mm_loadu_si128((__m128i *)src);
      const __m128i a_lo = _mm_unpacklo_epi8(a, zero);
      const __m128i a_hi = _mm_unpackhi_epi8(a, zero);
      const __m128i a_lo_filtered = _mm_mullo_epi16(a_lo, hfilter_0);
      const __m128i a_hi_filtered = _mm_mullo_epi16(a_hi, hfilter_0);

      const __m128i b = _mm_loadu_si128((__m128i *)(src + 1));
      const __m128i b_lo = _mm_unpacklo_epi8(b, zero);
      const __m128i b_hi = _mm_unpackhi_epi8(b, zero);
      const __m128i b_lo_filtered = _mm_mullo_epi16(b_lo, hfilter_1);
      const __m128i b_hi_filtered = _mm_mullo_epi16(b_hi, hfilter_1);

      const __m128i sum_lo = _mm_add_epi16(a_lo_filtered, b_lo_filtered);
      const __m128i sum_hi = _mm_add_epi16(a_hi_filtered, b_hi_filtered);

      const __m128i compensated_lo = _mm_add_epi16(sum_lo, round_factor);
      const __m128i compensated_hi = _mm_add_epi16(sum_hi, round_factor);

      const __m128i shifted_lo =
          _mm_srai_epi16(compensated_lo, VP8_FILTER_SHIFT);
      const __m128i shifted_hi =
          _mm_srai_epi16(compensated_hi, VP8_FILTER_SHIFT);

      _mm_store_si128((__m128i *)dst, shifted_lo);
      _mm_store_si128((__m128i *)(dst + 8), shifted_hi);
      src += stride;
      dst += 16;
    }
  }
}

static INLINE void vertical_16x16(uint16_t *src, uint8_t *dst, const int stride,
                                  const int yoffset) {
  int h;

  if (yoffset == 0) {
    for (h = 0; h < 16; ++h) {
      const __m128i row_lo = _mm_load_si128((__m128i *)src);
      const __m128i row_hi = _mm_load_si128((__m128i *)(src + 8));
      const __m128i packed = _mm_packus_epi16(row_lo, row_hi);
      _mm_store_si128((__m128i *)dst, packed);
      src += 16;
      dst += stride;
    }
    return;
  }

  {
    const __m128i round_factor = _mm_set1_epi16(1 << (VP8_FILTER_SHIFT - 1));
    const __m128i vfilter_0 = _mm_set1_epi16(vp8_bilinear_filters[yoffset][0]);
    const __m128i vfilter_1 = _mm_set1_epi16(vp8_bilinear_filters[yoffset][1]);

    __m128i row_0_lo = _mm_load_si128((__m128i *)src);
    __m128i row_0_hi = _mm_load_si128((__m128i *)(src + 8));
    src += 16;
    for (h = 0; h < 16; ++h) {
      const __m128i row_0_lo_filtered = _mm_mullo_epi16(row_0_lo, vfilter_0);
      const __m128i row_0_hi_filtered = _mm_mullo_epi16(row_0_hi, vfilter_0);

      const __m128i row_1_lo = _mm_load_si128((__m128i *)src);
      const __m128i row_1_hi = _mm_load_si128((__m128i *)(src + 8));
      const __m128i row_1_lo_filtered = _mm_mullo_epi16(row_1_lo, vfilter_1);
      const __m128i row_1_hi_filtered = _mm_mullo_epi16(row_1_hi, vfilter_1);

      const __m128i sum_lo =
          _mm_add_epi16(row_0_lo_filtered, row_1_lo_filtered);
      const __m128i sum_hi =
          _mm_add_epi16(row_0_hi_filtered, row_1_hi_filtered);

      const __m128i compensated_lo = _mm_add_epi16(sum_lo, round_factor);
      const __m128i compensated_hi = _mm_add_epi16(sum_hi, round_factor);

      const __m128i shifted_lo =
          _mm_srai_epi16(compensated_lo, VP8_FILTER_SHIFT);
      const __m128i shifted_hi =
          _mm_srai_epi16(compensated_hi, VP8_FILTER_SHIFT);

      const __m128i packed = _mm_packus_epi16(shifted_lo, shifted_hi);
      _mm_store_si128((__m128i *)dst, packed);
      row_0_lo = row_1_lo;
      row_0_hi = row_1_hi;
      src += 16;
      dst += stride;
    }
  }
}

void vp8_bilinear_predict16x16_sse2(uint8_t *src_ptr, int src_pixels_per_line,
                                    int xoffset, int yoffset, uint8_t *dst_ptr,
                                    int dst_pitch) {
  DECLARE_ALIGNED(16, uint16_t, FData[16 * 17]);

  assert((xoffset | yoffset) != 0);

  horizontal_16x16(src_ptr, src_pixels_per_line, FData, xoffset);

  vertical_16x16(FData, dst_ptr, dst_pitch, yoffset);
}

static INLINE void horizontal_8xN(uint8_t *src, const int stride, uint16_t *dst,
                                  const int xoffset, const int height) {
  int h;
  const __m128i zero = _mm_setzero_si128();

  if (xoffset == 0) {
    for (h = 0; h < height; ++h) {
      const __m128i a = _mm_loadl_epi64((__m128i *)src);
      const __m128i a_u16 = _mm_unpacklo_epi8(a, zero);
      _mm_store_si128((__m128i *)dst, a_u16);
      src += stride;
      dst += 8;
    }
    return;
  }

  {
    const __m128i round_factor = _mm_set1_epi16(1 << (VP8_FILTER_SHIFT - 1));
    const __m128i hfilter_0 = _mm_set1_epi16(vp8_bilinear_filters[xoffset][0]);
    const __m128i hfilter_1 = _mm_set1_epi16(vp8_bilinear_filters[xoffset][1]);

    // Filter horizontally. Rather than load the whole array and transpose, load
    // 16 values (overreading) and shift to set up the second value. Do an
    // "extra" 9th line so the vertical pass has the necessary context.
    for (h = 0; h < height; ++h) {
      const __m128i a = _mm_loadu_si128((__m128i *)src);
      const __m128i b = _mm_srli_si128(a, 1);
      const __m128i a_u16 = _mm_unpacklo_epi8(a, zero);
      const __m128i b_u16 = _mm_unpacklo_epi8(b, zero);
      const __m128i a_filtered = _mm_mullo_epi16(a_u16, hfilter_0);
      const __m128i b_filtered = _mm_mullo_epi16(b_u16, hfilter_1);
      const __m128i sum = _mm_add_epi16(a_filtered, b_filtered);
      const __m128i compensated = _mm_add_epi16(sum, round_factor);
      const __m128i shifted = _mm_srai_epi16(compensated, VP8_FILTER_SHIFT);
      _mm_store_si128((__m128i *)dst, shifted);
      src += stride;
      dst += 8;
    }
  }
}

static INLINE void vertical_8xN(uint16_t *src, uint8_t *dst, const int stride,
                                const int yoffset, const int height) {
  int h;

  if (yoffset == 0) {
    for (h = 0; h < height; ++h) {
      const __m128i row = _mm_load_si128((__m128i *)src);
      const __m128i packed = _mm_packus_epi16(row, row);
      _mm_storel_epi64((__m128i *)dst, packed);
      src += 8;
      dst += stride;
    }
    return;
  }

  {
    const __m128i round_factor = _mm_set1_epi16(1 << (VP8_FILTER_SHIFT - 1));
    const __m128i vfilter_0 = _mm_set1_epi16(vp8_bilinear_filters[yoffset][0]);
    const __m128i vfilter_1 = _mm_set1_epi16(vp8_bilinear_filters[yoffset][1]);

    __m128i row_0 = _mm_load_si128((__m128i *)src);
    src += 8;
    for (h = 0; h < height; ++h) {
      const __m128i row_1 = _mm_load_si128((__m128i *)src);
      const __m128i row_0_filtered = _mm_mullo_epi16(row_0, vfilter_0);
      const __m128i row_1_filtered = _mm_mullo_epi16(row_1, vfilter_1);
      const __m128i sum = _mm_add_epi16(row_0_filtered, row_1_filtered);
      const __m128i compensated = _mm_add_epi16(sum, round_factor);
      const __m128i shifted = _mm_srai_epi16(compensated, VP8_FILTER_SHIFT);
      const __m128i packed = _mm_packus_epi16(shifted, shifted);
      _mm_storel_epi64((__m128i *)dst, packed);
      row_0 = row_1;
      src += 8;
      dst += stride;
    }
  }
}

void vp8_bilinear_predict8x8_sse2(uint8_t *src_ptr, int src_pixels_per_line,
                                  int xoffset, int yoffset, uint8_t *dst_ptr,
                                  int dst_pitch) {
  DECLARE_ALIGNED(16, uint16_t, FData[8 * 9]);

  assert((xoffset | yoffset) != 0);

  horizontal_8xN(src_ptr, src_pixels_per_line, FData, xoffset, 9);

  vertical_8xN(FData, dst_ptr, dst_pitch, yoffset, 8);
}

void vp8_bilinear_predict8x4_sse2(uint8_t *src_ptr, int src_pixels_per_line,
                                  int xoffset, int yoffset, uint8_t *dst_ptr,
                                  int dst_pitch) {
  DECLARE_ALIGNED(16, uint16_t, FData[8 * 5]);

  assert((xoffset | yoffset) != 0);

  horizontal_8xN(src_ptr, src_pixels_per_line, FData, xoffset, 5);

  vertical_8xN(FData, dst_ptr, dst_pitch, yoffset, 4);
}

static INLINE void horizontal_4x4(uint8_t *src, const int stride, uint16_t *dst,
                                  const int xoffset) {
  int h;
  const __m128i zero = _mm_setzero_si128();

  if (xoffset == 0) {
    for (h = 0; h < 5; ++h) {
      const __m128i a = load_unaligned_u32(src);
      const __m128i a_u16 = _mm_unpacklo_epi8(a, zero);
      _mm_storel_epi64((__m128i *)dst, a_u16);
      src += stride;
      dst += 4;
    }
    return;
  }

  {
    const __m128i round_factor = _mm_set1_epi16(1 << (VP8_FILTER_SHIFT - 1));
    const __m128i hfilter_0 = _mm_set1_epi16(vp8_bilinear_filters[xoffset][0]);
    const __m128i hfilter_1 = _mm_set1_epi16(vp8_bilinear_filters[xoffset][1]);

    for (h = 0; h < 5; ++h) {
      const __m128i a = load_unaligned_u32(src);
      const __m128i b = load_unaligned_u32(src + 1);
      const __m128i a_u16 = _mm_unpacklo_epi8(a, zero);
      const __m128i b_u16 = _mm_unpacklo_epi8(b, zero);
      const __m128i a_filtered = _mm_mullo_epi16(a_u16, hfilter_0);
      const __m128i b_filtered = _mm_mullo_epi16(b_u16, hfilter_1);
      const __m128i sum = _mm_add_epi16(a_filtered, b_filtered);
      const __m128i compensated = _mm_add_epi16(sum, round_factor);
      const __m128i shifted = _mm_srai_epi16(compensated, VP8_FILTER_SHIFT);
      _mm_storel_epi64((__m128i *)dst, shifted);
      src += stride;
      dst += 4;
    }
  }
}

static INLINE void vertical_4x4(uint16_t *src, uint8_t *dst, const int stride,
                                const int yoffset) {
  int h;

  if (yoffset == 0) {
    for (h = 0; h < 4; h += 2) {
      const __m128i row = _mm_load_si128((__m128i *)src);
      __m128i packed = _mm_packus_epi16(row, row);
      store_unaligned_u32(dst, packed);
      dst += stride;
      packed = _mm_srli_si128(packed, 4);
      store_unaligned_u32(dst, packed);
      dst += stride;
      src += 8;
    }
    return;
  }

  {
    const __m128i round_factor = _mm_set1_epi16(1 << (VP8_FILTER_SHIFT - 1));
    const __m128i vfilter_0 = _mm_set1_epi16(vp8_bilinear_filters[yoffset][0]);
    const __m128i vfilter_1 = _mm_set1_epi16(vp8_bilinear_filters[yoffset][1]);

    for (h = 0; h < 4; h += 2) {
      const __m128i row_0 = _mm_load_si128((__m128i *)src);
      const __m128i row_1 = _mm_loadu_si128((__m128i *)(src + 4));
      const __m128i row_0_filtered = _mm_mullo_epi16(row_0, vfilter_0);
      const __m128i row_1_filtered = _mm_mullo_epi16(row_1, vfilter_1);
      const __m128i sum = _mm_add_epi16(row_0_filtered, row_1_filtered);
      const __m128i compensated = _mm_add_epi16(sum, round_factor);
      const __m128i shifted = _mm_srai_epi16(compensated, VP8_FILTER_SHIFT);
      __m128i packed = _mm_packus_epi16(shifted, shifted);
      storeu_int32(dst, _mm_cvtsi128_si32(packed));
      packed = _mm_srli_si128(packed, 4);
      dst += stride;
      storeu_int32(dst, _mm_cvtsi128_si32(packed));
      dst += stride;
      src += 8;
    }
  }
}

void vp8_bilinear_predict4x4_sse2(uint8_t *src_ptr, int src_pixels_per_line,
                                  int xoffset, int yoffset, uint8_t *dst_ptr,
                                  int dst_pitch) {
  DECLARE_ALIGNED(16, uint16_t, FData[4 * 5]);

  assert((xoffset | yoffset) != 0);

  horizontal_4x4(src_ptr, src_pixels_per_line, FData, xoffset);

  vertical_4x4(FData, dst_ptr, dst_pitch, yoffset);
}
