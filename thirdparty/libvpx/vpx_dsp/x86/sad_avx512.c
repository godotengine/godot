/*
 *  Copyright (c) 2025 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#include <immintrin.h>
#include "./vpx_dsp_rtcd.h"
#include "vpx_ports/mem.h"

static INLINE unsigned int sad64xh_avx512(const uint8_t *src_ptr,
                                          int src_stride,
                                          const uint8_t *ref_ptr,
                                          int ref_stride, int h) {
  int i, res;
  __m512i sad_reg, ref_reg;
  __m512i sum_sad = _mm512_setzero_si512();
  for (i = 0; i < h; i++) {
    ref_reg = _mm512_loadu_si512((const __m512i *)ref_ptr);
    sad_reg =
        _mm512_sad_epu8(ref_reg, _mm512_loadu_si512((__m512 const *)src_ptr));
    sum_sad = _mm512_add_epi32(sum_sad, sad_reg);
    ref_ptr += ref_stride;
    src_ptr += src_stride;
  }
  res = _mm512_reduce_add_epi32(sum_sad);
  return res;
}

#define FSAD64_H(h)                                                           \
  unsigned int vpx_sad64x##h##_avx512(const uint8_t *src_ptr, int src_stride, \
                                      const uint8_t *ref_ptr,                 \
                                      int ref_stride) {                       \
    return sad64xh_avx512(src_ptr, src_stride, ref_ptr, ref_stride, h);       \
  }

#define FSADS64_H(h)                                                  \
  unsigned int vpx_sad_skip_64x##h##_avx512(                          \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, \
      int ref_stride) {                                               \
    return 2 * sad64xh_avx512(src_ptr, src_stride * 2, ref_ptr,       \
                              ref_stride * 2, h / 2);                 \
  }

#define FSAD64  \
  FSAD64_H(64)  \
  FSAD64_H(32)  \
  FSADS64_H(64) \
  FSADS64_H(32)

FSAD64

#undef FSAD64
#undef FSAD64_H
#undef FSADS64_H

#define FSADAVG64_H(h)                                                         \
  unsigned int vpx_sad64x##h##_avg_avx512(                                     \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,          \
      int ref_stride, const uint8_t *second_pred) {                            \
    int i;                                                                     \
    __m512i sad_reg, ref_reg;                                                  \
    __m512i sum_sad = _mm512_setzero_si512();                                  \
    for (i = 0; i < h; i++) {                                                  \
      ref_reg = _mm512_loadu_si512((const __m512i *)ref_ptr);                  \
      ref_reg = _mm512_avg_epu8(                                               \
          ref_reg, _mm512_loadu_si512((const __m512i *)second_pred));          \
      sad_reg = _mm512_sad_epu8(ref_reg,                                       \
                                _mm512_loadu_si512((const __m512i *)src_ptr)); \
      sum_sad = _mm512_add_epi32(sum_sad, sad_reg);                            \
      ref_ptr += ref_stride;                                                   \
      src_ptr += src_stride;                                                   \
      second_pred += 64;                                                       \
    }                                                                          \
    return (unsigned int)_mm512_reduce_add_epi32(sum_sad);                     \
  }

#define FSADAVG64 \
  FSADAVG64_H(64) \
  FSADAVG64_H(32)

FSADAVG64

#undef FSADAVG64
#undef FSADAVG64_H
