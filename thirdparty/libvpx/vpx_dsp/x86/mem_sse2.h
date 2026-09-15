/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_X86_MEM_SSE2_H_
#define VPX_VPX_DSP_X86_MEM_SSE2_H_

#include <emmintrin.h>  // SSE2
#include <string.h>

#include "./vpx_config.h"

static INLINE void storeu_int32(void *dst, int32_t v) {
  memcpy(dst, &v, sizeof(v));
}

static INLINE int32_t loadu_int32(const void *src) {
  int32_t v;
  memcpy(&v, src, sizeof(v));
  return v;
}

static INLINE __m128i load_unaligned_u32(const void *a) {
  int val;
  memcpy(&val, a, sizeof(val));
  return _mm_cvtsi32_si128(val);
}

static INLINE void store_unaligned_u32(void *const a, const __m128i v) {
  const int val = _mm_cvtsi128_si32(v);
  memcpy(a, &val, sizeof(val));
}

#define mm_storelu(dst, v) memcpy((dst), (const char *)&(v), 8)
#define mm_storehu(dst, v) memcpy((dst), (const char *)&(v) + 8, 8)

static INLINE __m128i loadh_epi64(const __m128i s, const void *const src) {
  return _mm_castps_si128(
      _mm_loadh_pi(_mm_castsi128_ps(s), (const __m64 *)src));
}

static INLINE void load_8bit_4x4(const uint8_t *const s, const ptrdiff_t stride,
                                 __m128i *const d) {
  d[0] = _mm_cvtsi32_si128(*(const int *)(s + 0 * stride));
  d[1] = _mm_cvtsi32_si128(*(const int *)(s + 1 * stride));
  d[2] = _mm_cvtsi32_si128(*(const int *)(s + 2 * stride));
  d[3] = _mm_cvtsi32_si128(*(const int *)(s + 3 * stride));
}

static INLINE void load_8bit_4x8(const uint8_t *const s, const ptrdiff_t stride,
                                 __m128i *const d) {
  load_8bit_4x4(s + 0 * stride, stride, &d[0]);
  load_8bit_4x4(s + 4 * stride, stride, &d[4]);
}

static INLINE void load_8bit_8x4(const uint8_t *const s, const ptrdiff_t stride,
                                 __m128i *const d) {
  d[0] = _mm_loadl_epi64((const __m128i *)(s + 0 * stride));
  d[1] = _mm_loadl_epi64((const __m128i *)(s + 1 * stride));
  d[2] = _mm_loadl_epi64((const __m128i *)(s + 2 * stride));
  d[3] = _mm_loadl_epi64((const __m128i *)(s + 3 * stride));
}

static INLINE void load_8bit_8x8(const uint8_t *const s, const ptrdiff_t stride,
                                 __m128i *const d) {
  load_8bit_8x4(s + 0 * stride, stride, &d[0]);
  load_8bit_8x4(s + 4 * stride, stride, &d[4]);
}

static INLINE void load_8bit_16x8(const uint8_t *const s,
                                  const ptrdiff_t stride, __m128i *const d) {
  d[0] = _mm_load_si128((const __m128i *)(s + 0 * stride));
  d[1] = _mm_load_si128((const __m128i *)(s + 1 * stride));
  d[2] = _mm_load_si128((const __m128i *)(s + 2 * stride));
  d[3] = _mm_load_si128((const __m128i *)(s + 3 * stride));
  d[4] = _mm_load_si128((const __m128i *)(s + 4 * stride));
  d[5] = _mm_load_si128((const __m128i *)(s + 5 * stride));
  d[6] = _mm_load_si128((const __m128i *)(s + 6 * stride));
  d[7] = _mm_load_si128((const __m128i *)(s + 7 * stride));
}

static INLINE void loadu_8bit_16x4(const uint8_t *const s,
                                   const ptrdiff_t stride, __m128i *const d) {
  d[0] = _mm_loadu_si128((const __m128i *)(s + 0 * stride));
  d[1] = _mm_loadu_si128((const __m128i *)(s + 1 * stride));
  d[2] = _mm_loadu_si128((const __m128i *)(s + 2 * stride));
  d[3] = _mm_loadu_si128((const __m128i *)(s + 3 * stride));
}

static INLINE void loadu_8bit_16x8(const uint8_t *const s,
                                   const ptrdiff_t stride, __m128i *const d) {
  loadu_8bit_16x4(s + 0 * stride, stride, &d[0]);
  loadu_8bit_16x4(s + 4 * stride, stride, &d[4]);
}

static INLINE void _mm_storeh_epi64(__m128i *const d, const __m128i s) {
  _mm_storeh_pi((__m64 *)d, _mm_castsi128_ps(s));
}

static INLINE void store_8bit_4x4(const __m128i *const s, uint8_t *const d,
                                  const ptrdiff_t stride) {
  *(int *)(d + 0 * stride) = _mm_cvtsi128_si32(s[0]);
  *(int *)(d + 1 * stride) = _mm_cvtsi128_si32(s[1]);
  *(int *)(d + 2 * stride) = _mm_cvtsi128_si32(s[2]);
  *(int *)(d + 3 * stride) = _mm_cvtsi128_si32(s[3]);
}

static INLINE void store_8bit_4x4_sse2(const __m128i s, uint8_t *const d,
                                       const ptrdiff_t stride) {
  __m128i ss[4];

  ss[0] = s;
  ss[1] = _mm_srli_si128(s, 4);
  ss[2] = _mm_srli_si128(s, 8);
  ss[3] = _mm_srli_si128(s, 12);
  store_8bit_4x4(ss, d, stride);
}

static INLINE void store_8bit_8x4_from_16x2(const __m128i *const s,
                                            uint8_t *const d,
                                            const ptrdiff_t stride) {
  _mm_storel_epi64((__m128i *)(d + 0 * stride), s[0]);
  _mm_storeh_epi64((__m128i *)(d + 1 * stride), s[0]);
  _mm_storel_epi64((__m128i *)(d + 2 * stride), s[1]);
  _mm_storeh_epi64((__m128i *)(d + 3 * stride), s[1]);
}

static INLINE void store_8bit_8x8(const __m128i *const s, uint8_t *const d,
                                  const ptrdiff_t stride) {
  _mm_storel_epi64((__m128i *)(d + 0 * stride), s[0]);
  _mm_storel_epi64((__m128i *)(d + 1 * stride), s[1]);
  _mm_storel_epi64((__m128i *)(d + 2 * stride), s[2]);
  _mm_storel_epi64((__m128i *)(d + 3 * stride), s[3]);
  _mm_storel_epi64((__m128i *)(d + 4 * stride), s[4]);
  _mm_storel_epi64((__m128i *)(d + 5 * stride), s[5]);
  _mm_storel_epi64((__m128i *)(d + 6 * stride), s[6]);
  _mm_storel_epi64((__m128i *)(d + 7 * stride), s[7]);
}

static INLINE void storeu_8bit_16x4(const __m128i *const s, uint8_t *const d,
                                    const ptrdiff_t stride) {
  _mm_storeu_si128((__m128i *)(d + 0 * stride), s[0]);
  _mm_storeu_si128((__m128i *)(d + 1 * stride), s[1]);
  _mm_storeu_si128((__m128i *)(d + 2 * stride), s[2]);
  _mm_storeu_si128((__m128i *)(d + 3 * stride), s[3]);
}

#endif  // VPX_VPX_DSP_X86_MEM_SSE2_H_
