/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#include <immintrin.h>  // AVX2
#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"

// Note with sums[4] some versions of Visual Studio may fail due to parameter
// alignment, though the functions should be equivalent:
// error C2719: 'sums': formal parameter with requested alignment of 32 won't be
// aligned
static INLINE void calc_final_4(const __m256i *const sums /*[4]*/,
                                uint32_t sad_array[4]) {
  const __m256i t0 = _mm256_hadd_epi32(sums[0], sums[1]);
  const __m256i t1 = _mm256_hadd_epi32(sums[2], sums[3]);
  const __m256i t2 = _mm256_hadd_epi32(t0, t1);
  const __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(t2),
                                    _mm256_extractf128_si256(t2, 1));
  _mm_storeu_si128((__m128i *)sad_array, sum);
}

static INLINE void sad32xhx4d_avx2(const uint8_t *src_ptr, int src_stride,
                                   const uint8_t *const ref_array[4],
                                   int ref_stride, int h,
                                   uint32_t sad_array[4]) {
  int i;
  const uint8_t *refs[4];
  __m256i sums[4];

  refs[0] = ref_array[0];
  refs[1] = ref_array[1];
  refs[2] = ref_array[2];
  refs[3] = ref_array[3];
  sums[0] = _mm256_setzero_si256();
  sums[1] = _mm256_setzero_si256();
  sums[2] = _mm256_setzero_si256();
  sums[3] = _mm256_setzero_si256();

  for (i = 0; i < h; i++) {
    __m256i r[4];

    // load src and all ref[]
    const __m256i s = _mm256_load_si256((const __m256i *)src_ptr);
    r[0] = _mm256_loadu_si256((const __m256i *)refs[0]);
    r[1] = _mm256_loadu_si256((const __m256i *)refs[1]);
    r[2] = _mm256_loadu_si256((const __m256i *)refs[2]);
    r[3] = _mm256_loadu_si256((const __m256i *)refs[3]);

    // sum of the absolute differences between every ref[] to src
    r[0] = _mm256_sad_epu8(r[0], s);
    r[1] = _mm256_sad_epu8(r[1], s);
    r[2] = _mm256_sad_epu8(r[2], s);
    r[3] = _mm256_sad_epu8(r[3], s);

    // sum every ref[]
    sums[0] = _mm256_add_epi32(sums[0], r[0]);
    sums[1] = _mm256_add_epi32(sums[1], r[1]);
    sums[2] = _mm256_add_epi32(sums[2], r[2]);
    sums[3] = _mm256_add_epi32(sums[3], r[3]);

    src_ptr += src_stride;
    refs[0] += ref_stride;
    refs[1] += ref_stride;
    refs[2] += ref_stride;
    refs[3] += ref_stride;
  }

  calc_final_4(sums, sad_array);
}

static INLINE void sad64xhx4d_avx2(const uint8_t *src_ptr, int src_stride,
                                   const uint8_t *const ref_array[4],
                                   int ref_stride, int h,
                                   uint32_t sad_array[4]) {
  __m256i sums[4];
  int i;
  const uint8_t *refs[4];

  refs[0] = ref_array[0];
  refs[1] = ref_array[1];
  refs[2] = ref_array[2];
  refs[3] = ref_array[3];
  sums[0] = _mm256_setzero_si256();
  sums[1] = _mm256_setzero_si256();
  sums[2] = _mm256_setzero_si256();
  sums[3] = _mm256_setzero_si256();

  for (i = 0; i < h; i++) {
    __m256i r_lo[4], r_hi[4];
    // load 64 bytes from src and all ref[]
    const __m256i s_lo = _mm256_load_si256((const __m256i *)src_ptr);
    const __m256i s_hi = _mm256_load_si256((const __m256i *)(src_ptr + 32));
    r_lo[0] = _mm256_loadu_si256((const __m256i *)refs[0]);
    r_hi[0] = _mm256_loadu_si256((const __m256i *)(refs[0] + 32));
    r_lo[1] = _mm256_loadu_si256((const __m256i *)refs[1]);
    r_hi[1] = _mm256_loadu_si256((const __m256i *)(refs[1] + 32));
    r_lo[2] = _mm256_loadu_si256((const __m256i *)refs[2]);
    r_hi[2] = _mm256_loadu_si256((const __m256i *)(refs[2] + 32));
    r_lo[3] = _mm256_loadu_si256((const __m256i *)refs[3]);
    r_hi[3] = _mm256_loadu_si256((const __m256i *)(refs[3] + 32));

    // sum of the absolute differences between every ref[] to src
    r_lo[0] = _mm256_sad_epu8(r_lo[0], s_lo);
    r_lo[1] = _mm256_sad_epu8(r_lo[1], s_lo);
    r_lo[2] = _mm256_sad_epu8(r_lo[2], s_lo);
    r_lo[3] = _mm256_sad_epu8(r_lo[3], s_lo);
    r_hi[0] = _mm256_sad_epu8(r_hi[0], s_hi);
    r_hi[1] = _mm256_sad_epu8(r_hi[1], s_hi);
    r_hi[2] = _mm256_sad_epu8(r_hi[2], s_hi);
    r_hi[3] = _mm256_sad_epu8(r_hi[3], s_hi);

    // sum every ref[]
    sums[0] = _mm256_add_epi32(sums[0], r_lo[0]);
    sums[1] = _mm256_add_epi32(sums[1], r_lo[1]);
    sums[2] = _mm256_add_epi32(sums[2], r_lo[2]);
    sums[3] = _mm256_add_epi32(sums[3], r_lo[3]);
    sums[0] = _mm256_add_epi32(sums[0], r_hi[0]);
    sums[1] = _mm256_add_epi32(sums[1], r_hi[1]);
    sums[2] = _mm256_add_epi32(sums[2], r_hi[2]);
    sums[3] = _mm256_add_epi32(sums[3], r_hi[3]);

    src_ptr += src_stride;
    refs[0] += ref_stride;
    refs[1] += ref_stride;
    refs[2] += ref_stride;
    refs[3] += ref_stride;
  }

  calc_final_4(sums, sad_array);
}

#define SAD64_H(h)                                                         \
  void vpx_sad64x##h##x4d_avx2(const uint8_t *src, int src_stride,         \
                               const uint8_t *const ref_array[4],          \
                               int ref_stride, uint32_t sad_array[4]) {    \
    sad64xhx4d_avx2(src, src_stride, ref_array, ref_stride, h, sad_array); \
  }

#define SAD32_H(h)                                                         \
  void vpx_sad32x##h##x4d_avx2(const uint8_t *src, int src_stride,         \
                               const uint8_t *const ref_array[4],          \
                               int ref_stride, uint32_t sad_array[4]) {    \
    sad32xhx4d_avx2(src, src_stride, ref_array, ref_stride, h, sad_array); \
  }

SAD64_H(64)
SAD32_H(32)

#define SADS64_H(h)                                                           \
  void vpx_sad_skip_64x##h##x4d_avx2(const uint8_t *src, int src_stride,      \
                                     const uint8_t *const ref_array[4],       \
                                     int ref_stride, uint32_t sad_array[4]) { \
    sad64xhx4d_avx2(src, 2 * src_stride, ref_array, 2 * ref_stride,           \
                    ((h) >> 1), sad_array);                                   \
    sad_array[0] <<= 1;                                                       \
    sad_array[1] <<= 1;                                                       \
    sad_array[2] <<= 1;                                                       \
    sad_array[3] <<= 1;                                                       \
  }

#define SADS32_H(h)                                                           \
  void vpx_sad_skip_32x##h##x4d_avx2(const uint8_t *src, int src_stride,      \
                                     const uint8_t *const ref_array[4],       \
                                     int ref_stride, uint32_t sad_array[4]) { \
    sad32xhx4d_avx2(src, 2 * src_stride, ref_array, 2 * ref_stride,           \
                    ((h) >> 1), sad_array);                                   \
    sad_array[0] <<= 1;                                                       \
    sad_array[1] <<= 1;                                                       \
    sad_array[2] <<= 1;                                                       \
    sad_array[3] <<= 1;                                                       \
  }

SADS64_H(64)
SADS64_H(32)

SADS32_H(64)
SADS32_H(32)
SADS32_H(16)
