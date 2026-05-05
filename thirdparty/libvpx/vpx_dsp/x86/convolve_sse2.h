/*
 *  Copyright (c) 2018 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_X86_CONVOLVE_SSE2_H_
#define VPX_VPX_DSP_X86_CONVOLVE_SSE2_H_

#include <emmintrin.h>  // SSE2

#include "./vpx_config.h"

// Interprets the input register as 16-bit words 7 6 5 4 3 2 1 0, then returns
// values at index 2 and 3 to return 3 2 3 2 3 2 3 2 as 16-bit words
static INLINE __m128i extract_quarter_2_epi16_sse2(const __m128i *const reg) {
  __m128i tmp = _mm_unpacklo_epi32(*reg, *reg);
  return _mm_unpackhi_epi64(tmp, tmp);
}

// Interprets the input register as 16-bit words 7 6 5 4 3 2 1 0, then returns
// values at index 2 and 3 to return 5 4 5 4 5 4 5 4 as 16-bit words.
static INLINE __m128i extract_quarter_3_epi16_sse2(const __m128i *const reg) {
  __m128i tmp = _mm_unpackhi_epi32(*reg, *reg);
  return _mm_unpacklo_epi64(tmp, tmp);
}

// Interprets src as 8-bit words, zero extends to form 16-bit words, then
// multiplies with ker and add the adjacent results to form 32-bit words.
// Finally adds the result from 1 and 2 together.
static INLINE __m128i mm_madd_add_epi8_sse2(const __m128i *const src_1,
                                            const __m128i *const src_2,
                                            const __m128i *const ker_1,
                                            const __m128i *const ker_2) {
  const __m128i src_1_half = _mm_unpacklo_epi8(*src_1, _mm_setzero_si128());
  const __m128i src_2_half = _mm_unpacklo_epi8(*src_2, _mm_setzero_si128());
  const __m128i madd_1 = _mm_madd_epi16(src_1_half, *ker_1);
  const __m128i madd_2 = _mm_madd_epi16(src_2_half, *ker_2);
  return _mm_add_epi32(madd_1, madd_2);
}

// Interprets src as 16-bit words, then multiplies with ker and add the
// adjacent results to form 32-bit words. Finally adds the result from 1 and 2
// together.
static INLINE __m128i mm_madd_add_epi16_sse2(const __m128i *const src_1,
                                             const __m128i *const src_2,
                                             const __m128i *const ker_1,
                                             const __m128i *const ker_2) {
  const __m128i madd_1 = _mm_madd_epi16(*src_1, *ker_1);
  const __m128i madd_2 = _mm_madd_epi16(*src_2, *ker_2);
  return _mm_add_epi32(madd_1, madd_2);
}

static INLINE __m128i mm_madd_packs_epi16_sse2(const __m128i *const src_0,
                                               const __m128i *const src_1,
                                               const __m128i *const ker) {
  const __m128i madd_1 = _mm_madd_epi16(*src_0, *ker);
  const __m128i madd_2 = _mm_madd_epi16(*src_1, *ker);
  return _mm_packs_epi32(madd_1, madd_2);
}

// Interleaves src_1 and src_2
static INLINE __m128i mm_zip_epi32_sse2(const __m128i *const src_1,
                                        const __m128i *const src_2) {
  const __m128i tmp_1 = _mm_unpacklo_epi32(*src_1, *src_2);
  const __m128i tmp_2 = _mm_unpackhi_epi32(*src_1, *src_2);
  return _mm_packs_epi32(tmp_1, tmp_2);
}

static INLINE __m128i mm_round_epi32_sse2(const __m128i *const src,
                                          const __m128i *const half_depth,
                                          const int depth) {
  const __m128i nearest_src = _mm_add_epi32(*src, *half_depth);
  return _mm_srai_epi32(nearest_src, depth);
}

static INLINE __m128i mm_round_epi16_sse2(const __m128i *const src,
                                          const __m128i *const half_depth,
                                          const int depth) {
  const __m128i nearest_src = _mm_adds_epi16(*src, *half_depth);
  return _mm_srai_epi16(nearest_src, depth);
}

#endif  // VPX_VPX_DSP_X86_CONVOLVE_SSE2_H_
