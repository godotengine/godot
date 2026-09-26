/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
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

static INLINE unsigned int sad64xh_avx2(const uint8_t *src_ptr, int src_stride,
                                        const uint8_t *ref_ptr, int ref_stride,
                                        int h) {
  int i, res;
  __m256i sad1_reg, sad2_reg, ref1_reg, ref2_reg;
  __m256i sum_sad = _mm256_setzero_si256();
  __m256i sum_sad_h;
  __m128i sum_sad128;
  for (i = 0; i < h; i++) {
    ref1_reg = _mm256_loadu_si256((__m256i const *)ref_ptr);
    ref2_reg = _mm256_loadu_si256((__m256i const *)(ref_ptr + 32));
    sad1_reg =
        _mm256_sad_epu8(ref1_reg, _mm256_loadu_si256((__m256i const *)src_ptr));
    sad2_reg = _mm256_sad_epu8(
        ref2_reg, _mm256_loadu_si256((__m256i const *)(src_ptr + 32)));
    sum_sad = _mm256_add_epi32(sum_sad, _mm256_add_epi32(sad1_reg, sad2_reg));
    ref_ptr += ref_stride;
    src_ptr += src_stride;
  }
  sum_sad_h = _mm256_srli_si256(sum_sad, 8);
  sum_sad = _mm256_add_epi32(sum_sad, sum_sad_h);
  sum_sad128 = _mm256_extracti128_si256(sum_sad, 1);
  sum_sad128 = _mm_add_epi32(_mm256_castsi256_si128(sum_sad), sum_sad128);
  res = _mm_cvtsi128_si32(sum_sad128);
  return res;
}

static INLINE unsigned int sad32xh_avx2(const uint8_t *src_ptr, int src_stride,
                                        const uint8_t *ref_ptr, int ref_stride,
                                        int h) {
  int i, res;
  __m256i sad1_reg, sad2_reg, ref1_reg, ref2_reg;
  __m256i sum_sad = _mm256_setzero_si256();
  __m256i sum_sad_h;
  __m128i sum_sad128;
  const int ref2_stride = ref_stride << 1;
  const int src2_stride = src_stride << 1;
  const int max = h >> 1;
  for (i = 0; i < max; i++) {
    ref1_reg = _mm256_loadu_si256((__m256i const *)ref_ptr);
    ref2_reg = _mm256_loadu_si256((__m256i const *)(ref_ptr + ref_stride));
    sad1_reg =
        _mm256_sad_epu8(ref1_reg, _mm256_loadu_si256((__m256i const *)src_ptr));
    sad2_reg = _mm256_sad_epu8(
        ref2_reg, _mm256_loadu_si256((__m256i const *)(src_ptr + src_stride)));
    sum_sad = _mm256_add_epi32(sum_sad, _mm256_add_epi32(sad1_reg, sad2_reg));
    ref_ptr += ref2_stride;
    src_ptr += src2_stride;
  }
  sum_sad_h = _mm256_srli_si256(sum_sad, 8);
  sum_sad = _mm256_add_epi32(sum_sad, sum_sad_h);
  sum_sad128 = _mm256_extracti128_si256(sum_sad, 1);
  sum_sad128 = _mm_add_epi32(_mm256_castsi256_si128(sum_sad), sum_sad128);
  res = _mm_cvtsi128_si32(sum_sad128);
  return res;
}

#define FSAD64_H(h)                                                           \
  unsigned int vpx_sad64x##h##_avx2(const uint8_t *src_ptr, int src_stride,   \
                                    const uint8_t *ref_ptr, int ref_stride) { \
    return sad64xh_avx2(src_ptr, src_stride, ref_ptr, ref_stride, h);         \
  }

#define FSADS64_H(h)                                                          \
  unsigned int vpx_sad_skip_64x##h##_avx2(                                    \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,         \
      int ref_stride) {                                                       \
    return 2 * sad64xh_avx2(src_ptr, src_stride * 2, ref_ptr, ref_stride * 2, \
                            h / 2);                                           \
  }

#define FSAD32_H(h)                                                           \
  unsigned int vpx_sad32x##h##_avx2(const uint8_t *src_ptr, int src_stride,   \
                                    const uint8_t *ref_ptr, int ref_stride) { \
    return sad32xh_avx2(src_ptr, src_stride, ref_ptr, ref_stride, h);         \
  }

#define FSADS32_H(h)                                                          \
  unsigned int vpx_sad_skip_32x##h##_avx2(                                    \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,         \
      int ref_stride) {                                                       \
    return 2 * sad32xh_avx2(src_ptr, src_stride * 2, ref_ptr, ref_stride * 2, \
                            h / 2);                                           \
  }

#define FSAD64  \
  FSAD64_H(64)  \
  FSAD64_H(32)  \
  FSADS64_H(64) \
  FSADS64_H(32)

#define FSAD32  \
  FSAD32_H(64)  \
  FSAD32_H(32)  \
  FSAD32_H(16)  \
  FSADS32_H(64) \
  FSADS32_H(32) \
  FSADS32_H(16)

FSAD64
FSAD32

#undef FSAD64
#undef FSAD32
#undef FSAD64_H
#undef FSAD32_H
#undef FSADS64_H
#undef FSADS32_H

#define FSADAVG64_H(h)                                                        \
  unsigned int vpx_sad64x##h##_avg_avx2(                                      \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,         \
      int ref_stride, const uint8_t *second_pred) {                           \
    int i;                                                                    \
    __m256i sad1_reg, sad2_reg, ref1_reg, ref2_reg;                           \
    __m256i sum_sad = _mm256_setzero_si256();                                 \
    __m256i sum_sad_h;                                                        \
    __m128i sum_sad128;                                                       \
    for (i = 0; i < h; i++) {                                                 \
      ref1_reg = _mm256_loadu_si256((__m256i const *)ref_ptr);                \
      ref2_reg = _mm256_loadu_si256((__m256i const *)(ref_ptr + 32));         \
      ref1_reg = _mm256_avg_epu8(                                             \
          ref1_reg, _mm256_loadu_si256((__m256i const *)second_pred));        \
      ref2_reg = _mm256_avg_epu8(                                             \
          ref2_reg, _mm256_loadu_si256((__m256i const *)(second_pred + 32))); \
      sad1_reg = _mm256_sad_epu8(                                             \
          ref1_reg, _mm256_loadu_si256((__m256i const *)src_ptr));            \
      sad2_reg = _mm256_sad_epu8(                                             \
          ref2_reg, _mm256_loadu_si256((__m256i const *)(src_ptr + 32)));     \
      sum_sad =                                                               \
          _mm256_add_epi32(sum_sad, _mm256_add_epi32(sad1_reg, sad2_reg));    \
      ref_ptr += ref_stride;                                                  \
      src_ptr += src_stride;                                                  \
      second_pred += 64;                                                      \
    }                                                                         \
    sum_sad_h = _mm256_srli_si256(sum_sad, 8);                                \
    sum_sad = _mm256_add_epi32(sum_sad, sum_sad_h);                           \
    sum_sad128 = _mm256_extracti128_si256(sum_sad, 1);                        \
    sum_sad128 = _mm_add_epi32(_mm256_castsi256_si128(sum_sad), sum_sad128);  \
    return (unsigned int)_mm_cvtsi128_si32(sum_sad128);                       \
  }

#define FSADAVG32_H(h)                                                        \
  unsigned int vpx_sad32x##h##_avg_avx2(                                      \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,         \
      int ref_stride, const uint8_t *second_pred) {                           \
    int i;                                                                    \
    __m256i sad1_reg, sad2_reg, ref1_reg, ref2_reg;                           \
    __m256i sum_sad = _mm256_setzero_si256();                                 \
    __m256i sum_sad_h;                                                        \
    __m128i sum_sad128;                                                       \
    int ref2_stride = ref_stride << 1;                                        \
    int src2_stride = src_stride << 1;                                        \
    int max = h >> 1;                                                         \
    for (i = 0; i < max; i++) {                                               \
      ref1_reg = _mm256_loadu_si256((__m256i const *)ref_ptr);                \
      ref2_reg = _mm256_loadu_si256((__m256i const *)(ref_ptr + ref_stride)); \
      ref1_reg = _mm256_avg_epu8(                                             \
          ref1_reg, _mm256_loadu_si256((__m256i const *)second_pred));        \
      ref2_reg = _mm256_avg_epu8(                                             \
          ref2_reg, _mm256_loadu_si256((__m256i const *)(second_pred + 32))); \
      sad1_reg = _mm256_sad_epu8(                                             \
          ref1_reg, _mm256_loadu_si256((__m256i const *)src_ptr));            \
      sad2_reg = _mm256_sad_epu8(                                             \
          ref2_reg,                                                           \
          _mm256_loadu_si256((__m256i const *)(src_ptr + src_stride)));       \
      sum_sad =                                                               \
          _mm256_add_epi32(sum_sad, _mm256_add_epi32(sad1_reg, sad2_reg));    \
      ref_ptr += ref2_stride;                                                 \
      src_ptr += src2_stride;                                                 \
      second_pred += 64;                                                      \
    }                                                                         \
    sum_sad_h = _mm256_srli_si256(sum_sad, 8);                                \
    sum_sad = _mm256_add_epi32(sum_sad, sum_sad_h);                           \
    sum_sad128 = _mm256_extracti128_si256(sum_sad, 1);                        \
    sum_sad128 = _mm_add_epi32(_mm256_castsi256_si128(sum_sad), sum_sad128);  \
    return (unsigned int)_mm_cvtsi128_si32(sum_sad128);                       \
  }

#define FSADAVG64 \
  FSADAVG64_H(64) \
  FSADAVG64_H(32)

#define FSADAVG32 \
  FSADAVG32_H(64) \
  FSADAVG32_H(32) \
  FSADAVG32_H(16)

FSADAVG64
FSADAVG32

#undef FSADAVG64
#undef FSADAVG32
#undef FSADAVG64_H
#undef FSADAVG32_H
