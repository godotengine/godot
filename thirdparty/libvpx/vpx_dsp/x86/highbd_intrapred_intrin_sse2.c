/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <emmintrin.h>  // SSE2

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"

// -----------------------------------------------------------------------------

void vpx_highbd_h_predictor_4x4_sse2(uint16_t *dst, ptrdiff_t stride,
                                     const uint16_t *above,
                                     const uint16_t *left, int bd) {
  const __m128i left_u16 = _mm_loadl_epi64((const __m128i *)left);
  const __m128i row0 = _mm_shufflelo_epi16(left_u16, 0x0);
  const __m128i row1 = _mm_shufflelo_epi16(left_u16, 0x55);
  const __m128i row2 = _mm_shufflelo_epi16(left_u16, 0xaa);
  const __m128i row3 = _mm_shufflelo_epi16(left_u16, 0xff);
  (void)above;
  (void)bd;
  _mm_storel_epi64((__m128i *)dst, row0);
  dst += stride;
  _mm_storel_epi64((__m128i *)dst, row1);
  dst += stride;
  _mm_storel_epi64((__m128i *)dst, row2);
  dst += stride;
  _mm_storel_epi64((__m128i *)dst, row3);
}

void vpx_highbd_h_predictor_8x8_sse2(uint16_t *dst, ptrdiff_t stride,
                                     const uint16_t *above,
                                     const uint16_t *left, int bd) {
  const __m128i left_u16 = _mm_load_si128((const __m128i *)left);
  const __m128i row0 = _mm_shufflelo_epi16(left_u16, 0x0);
  const __m128i row1 = _mm_shufflelo_epi16(left_u16, 0x55);
  const __m128i row2 = _mm_shufflelo_epi16(left_u16, 0xaa);
  const __m128i row3 = _mm_shufflelo_epi16(left_u16, 0xff);
  const __m128i row4 = _mm_shufflehi_epi16(left_u16, 0x0);
  const __m128i row5 = _mm_shufflehi_epi16(left_u16, 0x55);
  const __m128i row6 = _mm_shufflehi_epi16(left_u16, 0xaa);
  const __m128i row7 = _mm_shufflehi_epi16(left_u16, 0xff);
  (void)above;
  (void)bd;
  _mm_store_si128((__m128i *)dst, _mm_unpacklo_epi64(row0, row0));
  dst += stride;
  _mm_store_si128((__m128i *)dst, _mm_unpacklo_epi64(row1, row1));
  dst += stride;
  _mm_store_si128((__m128i *)dst, _mm_unpacklo_epi64(row2, row2));
  dst += stride;
  _mm_store_si128((__m128i *)dst, _mm_unpacklo_epi64(row3, row3));
  dst += stride;
  _mm_store_si128((__m128i *)dst, _mm_unpackhi_epi64(row4, row4));
  dst += stride;
  _mm_store_si128((__m128i *)dst, _mm_unpackhi_epi64(row5, row5));
  dst += stride;
  _mm_store_si128((__m128i *)dst, _mm_unpackhi_epi64(row6, row6));
  dst += stride;
  _mm_store_si128((__m128i *)dst, _mm_unpackhi_epi64(row7, row7));
}

static INLINE void h_store_16_unpacklo(uint16_t **dst, const ptrdiff_t stride,
                                       const __m128i *row) {
  const __m128i val = _mm_unpacklo_epi64(*row, *row);
  _mm_store_si128((__m128i *)*dst, val);
  _mm_store_si128((__m128i *)(*dst + 8), val);
  *dst += stride;
}

static INLINE void h_store_16_unpackhi(uint16_t **dst, const ptrdiff_t stride,
                                       const __m128i *row) {
  const __m128i val = _mm_unpackhi_epi64(*row, *row);
  _mm_store_si128((__m128i *)(*dst), val);
  _mm_store_si128((__m128i *)(*dst + 8), val);
  *dst += stride;
}

void vpx_highbd_h_predictor_16x16_sse2(uint16_t *dst, ptrdiff_t stride,
                                       const uint16_t *above,
                                       const uint16_t *left, int bd) {
  int i;
  (void)above;
  (void)bd;

  for (i = 0; i < 2; i++, left += 8) {
    const __m128i left_u16 = _mm_load_si128((const __m128i *)left);
    const __m128i row0 = _mm_shufflelo_epi16(left_u16, 0x0);
    const __m128i row1 = _mm_shufflelo_epi16(left_u16, 0x55);
    const __m128i row2 = _mm_shufflelo_epi16(left_u16, 0xaa);
    const __m128i row3 = _mm_shufflelo_epi16(left_u16, 0xff);
    const __m128i row4 = _mm_shufflehi_epi16(left_u16, 0x0);
    const __m128i row5 = _mm_shufflehi_epi16(left_u16, 0x55);
    const __m128i row6 = _mm_shufflehi_epi16(left_u16, 0xaa);
    const __m128i row7 = _mm_shufflehi_epi16(left_u16, 0xff);
    h_store_16_unpacklo(&dst, stride, &row0);
    h_store_16_unpacklo(&dst, stride, &row1);
    h_store_16_unpacklo(&dst, stride, &row2);
    h_store_16_unpacklo(&dst, stride, &row3);
    h_store_16_unpackhi(&dst, stride, &row4);
    h_store_16_unpackhi(&dst, stride, &row5);
    h_store_16_unpackhi(&dst, stride, &row6);
    h_store_16_unpackhi(&dst, stride, &row7);
  }
}

static INLINE void h_store_32_unpacklo(uint16_t **dst, const ptrdiff_t stride,
                                       const __m128i *row) {
  const __m128i val = _mm_unpacklo_epi64(*row, *row);
  _mm_store_si128((__m128i *)(*dst), val);
  _mm_store_si128((__m128i *)(*dst + 8), val);
  _mm_store_si128((__m128i *)(*dst + 16), val);
  _mm_store_si128((__m128i *)(*dst + 24), val);
  *dst += stride;
}

static INLINE void h_store_32_unpackhi(uint16_t **dst, const ptrdiff_t stride,
                                       const __m128i *row) {
  const __m128i val = _mm_unpackhi_epi64(*row, *row);
  _mm_store_si128((__m128i *)(*dst), val);
  _mm_store_si128((__m128i *)(*dst + 8), val);
  _mm_store_si128((__m128i *)(*dst + 16), val);
  _mm_store_si128((__m128i *)(*dst + 24), val);
  *dst += stride;
}

void vpx_highbd_h_predictor_32x32_sse2(uint16_t *dst, ptrdiff_t stride,
                                       const uint16_t *above,
                                       const uint16_t *left, int bd) {
  int i;
  (void)above;
  (void)bd;

  for (i = 0; i < 4; i++, left += 8) {
    const __m128i left_u16 = _mm_load_si128((const __m128i *)left);
    const __m128i row0 = _mm_shufflelo_epi16(left_u16, 0x0);
    const __m128i row1 = _mm_shufflelo_epi16(left_u16, 0x55);
    const __m128i row2 = _mm_shufflelo_epi16(left_u16, 0xaa);
    const __m128i row3 = _mm_shufflelo_epi16(left_u16, 0xff);
    const __m128i row4 = _mm_shufflehi_epi16(left_u16, 0x0);
    const __m128i row5 = _mm_shufflehi_epi16(left_u16, 0x55);
    const __m128i row6 = _mm_shufflehi_epi16(left_u16, 0xaa);
    const __m128i row7 = _mm_shufflehi_epi16(left_u16, 0xff);
    h_store_32_unpacklo(&dst, stride, &row0);
    h_store_32_unpacklo(&dst, stride, &row1);
    h_store_32_unpacklo(&dst, stride, &row2);
    h_store_32_unpacklo(&dst, stride, &row3);
    h_store_32_unpackhi(&dst, stride, &row4);
    h_store_32_unpackhi(&dst, stride, &row5);
    h_store_32_unpackhi(&dst, stride, &row6);
    h_store_32_unpackhi(&dst, stride, &row7);
  }
}

//------------------------------------------------------------------------------
// DC 4x4

static INLINE __m128i dc_sum_4(const uint16_t *ref) {
  const __m128i _dcba = _mm_loadl_epi64((const __m128i *)ref);
  const __m128i _xxdc = _mm_shufflelo_epi16(_dcba, 0xe);
  const __m128i a = _mm_add_epi16(_dcba, _xxdc);
  return _mm_add_epi16(a, _mm_shufflelo_epi16(a, 0x1));
}

static INLINE void dc_store_4x4(uint16_t *dst, ptrdiff_t stride,
                                const __m128i *dc) {
  const __m128i dc_dup = _mm_shufflelo_epi16(*dc, 0x0);
  int i;
  for (i = 0; i < 4; ++i, dst += stride) {
    _mm_storel_epi64((__m128i *)dst, dc_dup);
  }
}

void vpx_highbd_dc_left_predictor_4x4_sse2(uint16_t *dst, ptrdiff_t stride,
                                           const uint16_t *above,
                                           const uint16_t *left, int bd) {
  const __m128i two = _mm_cvtsi32_si128(2);
  const __m128i sum = dc_sum_4(left);
  const __m128i dc = _mm_srli_epi16(_mm_add_epi16(sum, two), 2);
  (void)above;
  (void)bd;
  dc_store_4x4(dst, stride, &dc);
}

void vpx_highbd_dc_top_predictor_4x4_sse2(uint16_t *dst, ptrdiff_t stride,
                                          const uint16_t *above,
                                          const uint16_t *left, int bd) {
  const __m128i two = _mm_cvtsi32_si128(2);
  const __m128i sum = dc_sum_4(above);
  const __m128i dc = _mm_srli_epi16(_mm_add_epi16(sum, two), 2);
  (void)left;
  (void)bd;
  dc_store_4x4(dst, stride, &dc);
}

void vpx_highbd_dc_128_predictor_4x4_sse2(uint16_t *dst, ptrdiff_t stride,
                                          const uint16_t *above,
                                          const uint16_t *left, int bd) {
  const __m128i dc = _mm_cvtsi32_si128(1 << (bd - 1));
  const __m128i dc_dup = _mm_shufflelo_epi16(dc, 0x0);
  (void)above;
  (void)left;
  dc_store_4x4(dst, stride, &dc_dup);
}

//------------------------------------------------------------------------------
// DC 8x8

static INLINE __m128i dc_sum_8(const uint16_t *ref) {
  const __m128i ref_u16 = _mm_load_si128((const __m128i *)ref);
  const __m128i _dcba = _mm_add_epi16(ref_u16, _mm_srli_si128(ref_u16, 8));
  const __m128i _xxdc = _mm_shufflelo_epi16(_dcba, 0xe);
  const __m128i a = _mm_add_epi16(_dcba, _xxdc);

  return _mm_add_epi16(a, _mm_shufflelo_epi16(a, 0x1));
}

static INLINE void dc_store_8x8(uint16_t *dst, ptrdiff_t stride,
                                const __m128i *dc) {
  const __m128i dc_dup_lo = _mm_shufflelo_epi16(*dc, 0);
  const __m128i dc_dup = _mm_unpacklo_epi64(dc_dup_lo, dc_dup_lo);
  int i;
  for (i = 0; i < 8; ++i, dst += stride) {
    _mm_store_si128((__m128i *)dst, dc_dup);
  }
}

void vpx_highbd_dc_left_predictor_8x8_sse2(uint16_t *dst, ptrdiff_t stride,
                                           const uint16_t *above,
                                           const uint16_t *left, int bd) {
  const __m128i four = _mm_cvtsi32_si128(4);
  const __m128i sum = dc_sum_8(left);
  const __m128i dc = _mm_srli_epi16(_mm_add_epi16(sum, four), 3);
  (void)above;
  (void)bd;
  dc_store_8x8(dst, stride, &dc);
}

void vpx_highbd_dc_top_predictor_8x8_sse2(uint16_t *dst, ptrdiff_t stride,
                                          const uint16_t *above,
                                          const uint16_t *left, int bd) {
  const __m128i four = _mm_cvtsi32_si128(4);
  const __m128i sum = dc_sum_8(above);
  const __m128i dc = _mm_srli_epi16(_mm_add_epi16(sum, four), 3);
  (void)left;
  (void)bd;
  dc_store_8x8(dst, stride, &dc);
}

void vpx_highbd_dc_128_predictor_8x8_sse2(uint16_t *dst, ptrdiff_t stride,
                                          const uint16_t *above,
                                          const uint16_t *left, int bd) {
  const __m128i dc = _mm_cvtsi32_si128(1 << (bd - 1));
  const __m128i dc_dup = _mm_shufflelo_epi16(dc, 0x0);
  (void)above;
  (void)left;
  dc_store_8x8(dst, stride, &dc_dup);
}

//------------------------------------------------------------------------------
// DC 16x16

static INLINE __m128i dc_sum_16(const uint16_t *ref) {
  const __m128i sum_lo = dc_sum_8(ref);
  const __m128i sum_hi = dc_sum_8(ref + 8);
  return _mm_add_epi16(sum_lo, sum_hi);
}

static INLINE void dc_store_16x16(uint16_t *dst, ptrdiff_t stride,
                                  const __m128i *dc) {
  const __m128i dc_dup_lo = _mm_shufflelo_epi16(*dc, 0);
  const __m128i dc_dup = _mm_unpacklo_epi64(dc_dup_lo, dc_dup_lo);
  int i;
  for (i = 0; i < 16; ++i, dst += stride) {
    _mm_store_si128((__m128i *)dst, dc_dup);
    _mm_store_si128((__m128i *)(dst + 8), dc_dup);
  }
}

void vpx_highbd_dc_left_predictor_16x16_sse2(uint16_t *dst, ptrdiff_t stride,
                                             const uint16_t *above,
                                             const uint16_t *left, int bd) {
  const __m128i eight = _mm_cvtsi32_si128(8);
  const __m128i sum = dc_sum_16(left);
  const __m128i dc = _mm_srli_epi16(_mm_add_epi16(sum, eight), 4);
  (void)above;
  (void)bd;
  dc_store_16x16(dst, stride, &dc);
}

void vpx_highbd_dc_top_predictor_16x16_sse2(uint16_t *dst, ptrdiff_t stride,
                                            const uint16_t *above,
                                            const uint16_t *left, int bd) {
  const __m128i eight = _mm_cvtsi32_si128(8);
  const __m128i sum = dc_sum_16(above);
  const __m128i dc = _mm_srli_epi16(_mm_add_epi16(sum, eight), 4);
  (void)left;
  (void)bd;
  dc_store_16x16(dst, stride, &dc);
}

void vpx_highbd_dc_128_predictor_16x16_sse2(uint16_t *dst, ptrdiff_t stride,
                                            const uint16_t *above,
                                            const uint16_t *left, int bd) {
  const __m128i dc = _mm_cvtsi32_si128(1 << (bd - 1));
  const __m128i dc_dup = _mm_shufflelo_epi16(dc, 0x0);
  (void)above;
  (void)left;
  dc_store_16x16(dst, stride, &dc_dup);
}

//------------------------------------------------------------------------------
// DC 32x32

static INLINE __m128i dc_sum_32(const uint16_t *ref) {
  const __m128i zero = _mm_setzero_si128();
  const __m128i sum_a = dc_sum_16(ref);
  const __m128i sum_b = dc_sum_16(ref + 16);
  // 12 bit bd will outrange, so expand to 32 bit before adding final total
  return _mm_add_epi32(_mm_unpacklo_epi16(sum_a, zero),
                       _mm_unpacklo_epi16(sum_b, zero));
}

static INLINE void dc_store_32x32(uint16_t *dst, ptrdiff_t stride,
                                  const __m128i *dc) {
  const __m128i dc_dup_lo = _mm_shufflelo_epi16(*dc, 0);
  const __m128i dc_dup = _mm_unpacklo_epi64(dc_dup_lo, dc_dup_lo);
  int i;
  for (i = 0; i < 32; ++i, dst += stride) {
    _mm_store_si128((__m128i *)dst, dc_dup);
    _mm_store_si128((__m128i *)(dst + 8), dc_dup);
    _mm_store_si128((__m128i *)(dst + 16), dc_dup);
    _mm_store_si128((__m128i *)(dst + 24), dc_dup);
  }
}

void vpx_highbd_dc_left_predictor_32x32_sse2(uint16_t *dst, ptrdiff_t stride,
                                             const uint16_t *above,
                                             const uint16_t *left, int bd) {
  const __m128i sixteen = _mm_cvtsi32_si128(16);
  const __m128i sum = dc_sum_32(left);
  const __m128i dc = _mm_srli_epi32(_mm_add_epi32(sum, sixteen), 5);
  (void)above;
  (void)bd;
  dc_store_32x32(dst, stride, &dc);
}

void vpx_highbd_dc_top_predictor_32x32_sse2(uint16_t *dst, ptrdiff_t stride,
                                            const uint16_t *above,
                                            const uint16_t *left, int bd) {
  const __m128i sixteen = _mm_cvtsi32_si128(16);
  const __m128i sum = dc_sum_32(above);
  const __m128i dc = _mm_srli_epi32(_mm_add_epi32(sum, sixteen), 5);
  (void)left;
  (void)bd;
  dc_store_32x32(dst, stride, &dc);
}

void vpx_highbd_dc_128_predictor_32x32_sse2(uint16_t *dst, ptrdiff_t stride,
                                            const uint16_t *above,
                                            const uint16_t *left, int bd) {
  const __m128i dc = _mm_cvtsi32_si128(1 << (bd - 1));
  const __m128i dc_dup = _mm_shufflelo_epi16(dc, 0x0);
  (void)above;
  (void)left;
  dc_store_32x32(dst, stride, &dc_dup);
}

// -----------------------------------------------------------------------------
/*
; ------------------------------------------
; input: x, y, z, result
;
; trick from pascal
; (x+2y+z+2)>>2 can be calculated as:
; result = avg(x,z)
; result -= xor(x,z) & 1
; result = avg(result,y)
; ------------------------------------------
*/
static INLINE __m128i avg3_epu16(const __m128i *x, const __m128i *y,
                                 const __m128i *z) {
  const __m128i one = _mm_set1_epi16(1);
  const __m128i a = _mm_avg_epu16(*x, *z);
  const __m128i b =
      _mm_subs_epu16(a, _mm_and_si128(_mm_xor_si128(*x, *z), one));
  return _mm_avg_epu16(b, *y);
}

void vpx_highbd_d117_predictor_4x4_sse2(uint16_t *dst, ptrdiff_t stride,
                                        const uint16_t *above,
                                        const uint16_t *left, int bd) {
  const int I = left[0];
  const int J = left[1];
  const int K = left[2];
  const __m128i XXXXABCD = _mm_loadu_si128((const __m128i *)(above - 4));
  const __m128i KXXXABCD = _mm_insert_epi16(XXXXABCD, K, 0);
  const __m128i KJXXABCD = _mm_insert_epi16(KXXXABCD, J, 1);
  const __m128i KJIXABCD = _mm_insert_epi16(KJXXABCD, I, 2);
  const __m128i JIXABCD0 = _mm_srli_si128(KJIXABCD, 2);
  const __m128i IXABCD00 = _mm_srli_si128(KJIXABCD, 4);
  const __m128i avg2 = _mm_avg_epu16(KJIXABCD, JIXABCD0);
  const __m128i avg3 = avg3_epu16(&KJIXABCD, &JIXABCD0, &IXABCD00);
  const __m128i row0 = _mm_srli_si128(avg2, 6);
  const __m128i row1 = _mm_srli_si128(avg3, 4);
  const __m128i row2 = _mm_srli_si128(avg2, 4);
  const __m128i row3 = _mm_srli_si128(avg3, 2);
  (void)bd;
  _mm_storel_epi64((__m128i *)dst, row0);
  dst += stride;
  _mm_storel_epi64((__m128i *)dst, row1);
  dst += stride;
  _mm_storel_epi64((__m128i *)dst, row2);
  dst += stride;
  _mm_storel_epi64((__m128i *)dst, row3);

  dst -= stride;
  dst[0] = _mm_extract_epi16(avg3, 1);
  dst[stride] = _mm_extract_epi16(avg3, 0);
}

void vpx_highbd_d135_predictor_4x4_sse2(uint16_t *dst, ptrdiff_t stride,
                                        const uint16_t *above,
                                        const uint16_t *left, int bd) {
  const int I = left[0];
  const int J = left[1];
  const int K = left[2];
  const int L = left[3];
  const __m128i XXXXABCD = _mm_loadu_si128((const __m128i *)(above - 4));
  const __m128i KXXXABCD = _mm_insert_epi16(XXXXABCD, K, 0);
  const __m128i KJXXABCD = _mm_insert_epi16(KXXXABCD, J, 1);
  const __m128i KJIXABCD = _mm_insert_epi16(KJXXABCD, I, 2);
  const __m128i JIXABCD0 = _mm_srli_si128(KJIXABCD, 2);
  const __m128i LKJIXABC = _mm_insert_epi16(_mm_slli_si128(KJIXABCD, 2), L, 0);
  const __m128i avg3 = avg3_epu16(&JIXABCD0, &KJIXABCD, &LKJIXABC);
  const __m128i row0 = _mm_srli_si128(avg3, 6);
  const __m128i row1 = _mm_srli_si128(avg3, 4);
  const __m128i row2 = _mm_srli_si128(avg3, 2);
  const __m128i row3 = avg3;
  (void)bd;
  _mm_storel_epi64((__m128i *)dst, row0);
  dst += stride;
  _mm_storel_epi64((__m128i *)dst, row1);
  dst += stride;
  _mm_storel_epi64((__m128i *)dst, row2);
  dst += stride;
  _mm_storel_epi64((__m128i *)dst, row3);
}

void vpx_highbd_d153_predictor_4x4_sse2(uint16_t *dst, ptrdiff_t stride,
                                        const uint16_t *above,
                                        const uint16_t *left, int bd) {
  const int I = left[0];
  const int J = left[1];
  const int K = left[2];
  const int L = left[3];
  const __m128i XXXXXABC = _mm_castps_si128(
      _mm_loadh_pi(_mm_setzero_ps(), (const __m64 *)(above - 1)));
  const __m128i LXXXXABC = _mm_insert_epi16(XXXXXABC, L, 0);
  const __m128i LKXXXABC = _mm_insert_epi16(LXXXXABC, K, 1);
  const __m128i LKJXXABC = _mm_insert_epi16(LKXXXABC, J, 2);
  const __m128i LKJIXABC = _mm_insert_epi16(LKJXXABC, I, 3);
  const __m128i KJIXABC0 = _mm_srli_si128(LKJIXABC, 2);
  const __m128i JIXABC00 = _mm_srli_si128(LKJIXABC, 4);
  const __m128i avg3 = avg3_epu16(&LKJIXABC, &KJIXABC0, &JIXABC00);
  const __m128i avg2 = _mm_avg_epu16(LKJIXABC, KJIXABC0);
  const __m128i row3 = _mm_unpacklo_epi16(avg2, avg3);
  const __m128i row2 = _mm_srli_si128(row3, 4);
  const __m128i row1 = _mm_srli_si128(row3, 8);
  const __m128i row0 = _mm_srli_si128(avg3, 4);
  (void)bd;
  _mm_storel_epi64((__m128i *)dst, row0);
  dst[0] = _mm_extract_epi16(avg2, 3);
  dst += stride;
  _mm_storel_epi64((__m128i *)dst, row1);
  dst += stride;
  _mm_storel_epi64((__m128i *)dst, row2);
  dst += stride;
  _mm_storel_epi64((__m128i *)dst, row3);
}

void vpx_highbd_d207_predictor_4x4_sse2(uint16_t *dst, ptrdiff_t stride,
                                        const uint16_t *above,
                                        const uint16_t *left, int bd) {
  const __m128i IJKL0000 = _mm_load_si128((const __m128i *)left);
  const __m128i LLLL0000 = _mm_shufflelo_epi16(IJKL0000, 0xff);
  const __m128i IJKLLLLL = _mm_unpacklo_epi64(IJKL0000, LLLL0000);
  const __m128i JKLLLLL0 = _mm_srli_si128(IJKLLLLL, 2);
  const __m128i KLLLLL00 = _mm_srli_si128(IJKLLLLL, 4);
  const __m128i avg3 = avg3_epu16(&IJKLLLLL, &JKLLLLL0, &KLLLLL00);
  const __m128i avg2 = _mm_avg_epu16(IJKLLLLL, JKLLLLL0);
  const __m128i row0 = _mm_unpacklo_epi16(avg2, avg3);
  const __m128i row1 = _mm_srli_si128(row0, 4);
  const __m128i row2 = _mm_srli_si128(row0, 8);
  const __m128i row3 = LLLL0000;
  (void)above;
  (void)bd;
  _mm_storel_epi64((__m128i *)dst, row0);
  dst += stride;
  _mm_storel_epi64((__m128i *)dst, row1);
  dst += stride;
  _mm_storel_epi64((__m128i *)dst, row2);
  dst += stride;
  _mm_storel_epi64((__m128i *)dst, row3);
}

void vpx_highbd_d63_predictor_4x4_sse2(uint16_t *dst, ptrdiff_t stride,
                                       const uint16_t *above,
                                       const uint16_t *left, int bd) {
  const __m128i ABCDEFGH = _mm_loadu_si128((const __m128i *)above);
  const __m128i BCDEFGH0 = _mm_srli_si128(ABCDEFGH, 2);
  const __m128i CDEFGH00 = _mm_srli_si128(ABCDEFGH, 4);
  const __m128i avg3 = avg3_epu16(&ABCDEFGH, &BCDEFGH0, &CDEFGH00);
  const __m128i avg2 = _mm_avg_epu16(ABCDEFGH, BCDEFGH0);
  const __m128i row0 = avg2;
  const __m128i row1 = avg3;
  const __m128i row2 = _mm_srli_si128(avg2, 2);
  const __m128i row3 = _mm_srli_si128(avg3, 2);
  (void)left;
  (void)bd;
  _mm_storel_epi64((__m128i *)dst, row0);
  dst += stride;
  _mm_storel_epi64((__m128i *)dst, row1);
  dst += stride;
  _mm_storel_epi64((__m128i *)dst, row2);
  dst += stride;
  _mm_storel_epi64((__m128i *)dst, row3);
}
