/*
 *  Copyright (c) 2022 The WebM project authors. All Rights Reserved.
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

static VPX_FORCE_INLINE void calc_final_4(const __m256i *const sums /*[4]*/,
                                          uint32_t sad_array[4]) {
  const __m256i t0 = _mm256_hadd_epi32(sums[0], sums[1]);
  const __m256i t1 = _mm256_hadd_epi32(sums[2], sums[3]);
  const __m256i t2 = _mm256_hadd_epi32(t0, t1);
  const __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(t2),
                                    _mm256_extractf128_si256(t2, 1));
  _mm_storeu_si128((__m128i *)sad_array, sum);
}

static VPX_FORCE_INLINE void highbd_sad64xHx4d(__m256i *sums_16 /*[4]*/,
                                               const uint16_t *src,
                                               int src_stride,
                                               uint16_t *refs[4],
                                               int ref_stride, int height) {
  int i;
  for (i = 0; i < height; ++i) {
    // load src and all ref[]
    const __m256i s0 = _mm256_load_si256((const __m256i *)src);
    const __m256i s1 = _mm256_load_si256((const __m256i *)(src + 16));
    const __m256i s2 = _mm256_load_si256((const __m256i *)(src + 32));
    const __m256i s3 = _mm256_load_si256((const __m256i *)(src + 48));
    int x;

    for (x = 0; x < 4; ++x) {
      __m256i r[4];
      r[0] = _mm256_loadu_si256((const __m256i *)refs[x]);
      r[1] = _mm256_loadu_si256((const __m256i *)(refs[x] + 16));
      r[2] = _mm256_loadu_si256((const __m256i *)(refs[x] + 32));
      r[3] = _mm256_loadu_si256((const __m256i *)(refs[x] + 48));

      // absolute differences between every ref[] to src
      r[0] = _mm256_abs_epi16(_mm256_sub_epi16(r[0], s0));
      r[1] = _mm256_abs_epi16(_mm256_sub_epi16(r[1], s1));
      r[2] = _mm256_abs_epi16(_mm256_sub_epi16(r[2], s2));
      r[3] = _mm256_abs_epi16(_mm256_sub_epi16(r[3], s3));

      // sum every abs diff
      sums_16[x] = _mm256_add_epi16(sums_16[x], _mm256_add_epi16(r[0], r[1]));
      sums_16[x] = _mm256_add_epi16(sums_16[x], _mm256_add_epi16(r[2], r[3]));
    }

    src += src_stride;
    refs[0] += ref_stride;
    refs[1] += ref_stride;
    refs[2] += ref_stride;
    refs[3] += ref_stride;
  }
}

static VPX_FORCE_INLINE void highbd_sad64xNx4d_avx2(
    const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4],
    int ref_stride, uint32_t sad_array[4], int n) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src_ptr);
  uint16_t *refs[4];
  __m256i sums_16[4];
  __m256i sums_32[4];
  int i;

  refs[0] = CONVERT_TO_SHORTPTR(ref_array[0]);
  refs[1] = CONVERT_TO_SHORTPTR(ref_array[1]);
  refs[2] = CONVERT_TO_SHORTPTR(ref_array[2]);
  refs[3] = CONVERT_TO_SHORTPTR(ref_array[3]);
  sums_32[0] = _mm256_setzero_si256();
  sums_32[1] = _mm256_setzero_si256();
  sums_32[2] = _mm256_setzero_si256();
  sums_32[3] = _mm256_setzero_si256();

  for (i = 0; i < (n / 2); ++i) {
    sums_16[0] = _mm256_setzero_si256();
    sums_16[1] = _mm256_setzero_si256();
    sums_16[2] = _mm256_setzero_si256();
    sums_16[3] = _mm256_setzero_si256();

    highbd_sad64xHx4d(sums_16, src, src_stride, refs, ref_stride, 2);

    /* sums_16 will outrange after 2 rows, so add current sums_16 to
     * sums_32*/
    sums_32[0] = _mm256_add_epi32(
        sums_32[0],
        _mm256_add_epi32(
            _mm256_cvtepu16_epi32(_mm256_castsi256_si128(sums_16[0])),
            _mm256_cvtepu16_epi32(_mm256_extractf128_si256(sums_16[0], 1))));
    sums_32[1] = _mm256_add_epi32(
        sums_32[1],
        _mm256_add_epi32(
            _mm256_cvtepu16_epi32(_mm256_castsi256_si128(sums_16[1])),
            _mm256_cvtepu16_epi32(_mm256_extractf128_si256(sums_16[1], 1))));
    sums_32[2] = _mm256_add_epi32(
        sums_32[2],
        _mm256_add_epi32(
            _mm256_cvtepu16_epi32(_mm256_castsi256_si128(sums_16[2])),
            _mm256_cvtepu16_epi32(_mm256_extractf128_si256(sums_16[2], 1))));
    sums_32[3] = _mm256_add_epi32(
        sums_32[3],
        _mm256_add_epi32(
            _mm256_cvtepu16_epi32(_mm256_castsi256_si128(sums_16[3])),
            _mm256_cvtepu16_epi32(_mm256_extractf128_si256(sums_16[3], 1))));

    src += src_stride << 1;
  }
  calc_final_4(sums_32, sad_array);
}

#define HIGHBD_SAD64XNX4D(n)                                                   \
  void vpx_highbd_sad64x##n##x4d_avx2(const uint8_t *src, int src_stride,      \
                                      const uint8_t *const ref_array[4],       \
                                      int ref_stride, uint32_t sad_array[4]) { \
    highbd_sad64xNx4d_avx2(src, src_stride, ref_array, ref_stride, sad_array,  \
                           n);                                                 \
  }

#define HIGHBD_SADSKIP64XNx4D(n)                                             \
  void vpx_highbd_sad_skip_64x##n##x4d_avx2(                                 \
      const uint8_t *src, int src_stride, const uint8_t *const ref_array[4], \
      int ref_stride, uint32_t sad_array[4]) {                               \
    highbd_sad64xNx4d_avx2(src, 2 * src_stride, ref_array, 2 * ref_stride,   \
                           sad_array, n / 2);                                \
    sad_array[0] <<= 1;                                                      \
    sad_array[1] <<= 1;                                                      \
    sad_array[2] <<= 1;                                                      \
    sad_array[3] <<= 1;                                                      \
  }

static VPX_FORCE_INLINE void highbd_sad32xHx4d(__m256i *sums_16 /*[4]*/,
                                               const uint16_t *src,
                                               int src_stride,
                                               uint16_t *refs[4],
                                               int ref_stride, int height) {
  int i;
  for (i = 0; i < height; i++) {
    __m256i r[8];

    // load src and all ref[]
    const __m256i s = _mm256_load_si256((const __m256i *)src);
    const __m256i s2 = _mm256_load_si256((const __m256i *)(src + 16));
    r[0] = _mm256_loadu_si256((const __m256i *)refs[0]);
    r[1] = _mm256_loadu_si256((const __m256i *)(refs[0] + 16));
    r[2] = _mm256_loadu_si256((const __m256i *)refs[1]);
    r[3] = _mm256_loadu_si256((const __m256i *)(refs[1] + 16));
    r[4] = _mm256_loadu_si256((const __m256i *)refs[2]);
    r[5] = _mm256_loadu_si256((const __m256i *)(refs[2] + 16));
    r[6] = _mm256_loadu_si256((const __m256i *)refs[3]);
    r[7] = _mm256_loadu_si256((const __m256i *)(refs[3] + 16));

    // absolute differences between every ref[] to src
    r[0] = _mm256_abs_epi16(_mm256_sub_epi16(r[0], s));
    r[1] = _mm256_abs_epi16(_mm256_sub_epi16(r[1], s2));
    r[2] = _mm256_abs_epi16(_mm256_sub_epi16(r[2], s));
    r[3] = _mm256_abs_epi16(_mm256_sub_epi16(r[3], s2));
    r[4] = _mm256_abs_epi16(_mm256_sub_epi16(r[4], s));
    r[5] = _mm256_abs_epi16(_mm256_sub_epi16(r[5], s2));
    r[6] = _mm256_abs_epi16(_mm256_sub_epi16(r[6], s));
    r[7] = _mm256_abs_epi16(_mm256_sub_epi16(r[7], s2));

    // sum every abs diff
    sums_16[0] = _mm256_add_epi16(sums_16[0], _mm256_add_epi16(r[0], r[1]));
    sums_16[1] = _mm256_add_epi16(sums_16[1], _mm256_add_epi16(r[2], r[3]));
    sums_16[2] = _mm256_add_epi16(sums_16[2], _mm256_add_epi16(r[4], r[5]));
    sums_16[3] = _mm256_add_epi16(sums_16[3], _mm256_add_epi16(r[6], r[7]));

    src += src_stride;
    refs[0] += ref_stride;
    refs[1] += ref_stride;
    refs[2] += ref_stride;
    refs[3] += ref_stride;
  }
}

static VPX_FORCE_INLINE void highbd_sad32xNx4d_avx2(
    const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4],
    int ref_stride, uint32_t sad_array[4], int n) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src_ptr);
  uint16_t *refs[4];
  __m256i sums_16[4];
  __m256i sums_32[4];
  int i;

  refs[0] = CONVERT_TO_SHORTPTR(ref_array[0]);
  refs[1] = CONVERT_TO_SHORTPTR(ref_array[1]);
  refs[2] = CONVERT_TO_SHORTPTR(ref_array[2]);
  refs[3] = CONVERT_TO_SHORTPTR(ref_array[3]);
  sums_32[0] = _mm256_setzero_si256();
  sums_32[1] = _mm256_setzero_si256();
  sums_32[2] = _mm256_setzero_si256();
  sums_32[3] = _mm256_setzero_si256();

  for (i = 0; i < (n / 8); ++i) {
    sums_16[0] = _mm256_setzero_si256();
    sums_16[1] = _mm256_setzero_si256();
    sums_16[2] = _mm256_setzero_si256();
    sums_16[3] = _mm256_setzero_si256();

    highbd_sad32xHx4d(sums_16, src, src_stride, refs, ref_stride, 8);

    /* sums_16 will outrange after 8 rows, so add current sums_16 to
     * sums_32*/
    sums_32[0] = _mm256_add_epi32(
        sums_32[0],
        _mm256_add_epi32(
            _mm256_cvtepu16_epi32(_mm256_castsi256_si128(sums_16[0])),
            _mm256_cvtepu16_epi32(_mm256_extractf128_si256(sums_16[0], 1))));
    sums_32[1] = _mm256_add_epi32(
        sums_32[1],
        _mm256_add_epi32(
            _mm256_cvtepu16_epi32(_mm256_castsi256_si128(sums_16[1])),
            _mm256_cvtepu16_epi32(_mm256_extractf128_si256(sums_16[1], 1))));
    sums_32[2] = _mm256_add_epi32(
        sums_32[2],
        _mm256_add_epi32(
            _mm256_cvtepu16_epi32(_mm256_castsi256_si128(sums_16[2])),
            _mm256_cvtepu16_epi32(_mm256_extractf128_si256(sums_16[2], 1))));
    sums_32[3] = _mm256_add_epi32(
        sums_32[3],
        _mm256_add_epi32(
            _mm256_cvtepu16_epi32(_mm256_castsi256_si128(sums_16[3])),
            _mm256_cvtepu16_epi32(_mm256_extractf128_si256(sums_16[3], 1))));

    src += src_stride << 3;
  }
  calc_final_4(sums_32, sad_array);
}

#define HIGHBD_SAD32XNX4D(n)                                                   \
  void vpx_highbd_sad32x##n##x4d_avx2(const uint8_t *src, int src_stride,      \
                                      const uint8_t *const ref_array[4],       \
                                      int ref_stride, uint32_t sad_array[4]) { \
    highbd_sad32xNx4d_avx2(src, src_stride, ref_array, ref_stride, sad_array,  \
                           n);                                                 \
  }

#define HIGHBD_SADSKIP32XNx4D(n)                                             \
  void vpx_highbd_sad_skip_32x##n##x4d_avx2(                                 \
      const uint8_t *src, int src_stride, const uint8_t *const ref_array[4], \
      int ref_stride, uint32_t sad_array[4]) {                               \
    highbd_sad32xNx4d_avx2(src, 2 * src_stride, ref_array, 2 * ref_stride,   \
                           sad_array, n / 2);                                \
    sad_array[0] <<= 1;                                                      \
    sad_array[1] <<= 1;                                                      \
    sad_array[2] <<= 1;                                                      \
    sad_array[3] <<= 1;                                                      \
  }

static VPX_FORCE_INLINE void highbd_sad16xHx4d(__m256i *sums_16 /*[4]*/,
                                               const uint16_t *src,
                                               int src_stride,
                                               uint16_t *refs[4],
                                               int ref_stride, int height) {
  int i;
  for (i = 0; i < height; i++) {
    __m256i r[4];

    // load src and all ref[]
    const __m256i s = _mm256_load_si256((const __m256i *)src);
    r[0] = _mm256_loadu_si256((const __m256i *)refs[0]);
    r[1] = _mm256_loadu_si256((const __m256i *)refs[1]);
    r[2] = _mm256_loadu_si256((const __m256i *)refs[2]);
    r[3] = _mm256_loadu_si256((const __m256i *)refs[3]);

    // absolute differences between every ref[] to src
    r[0] = _mm256_abs_epi16(_mm256_sub_epi16(r[0], s));
    r[1] = _mm256_abs_epi16(_mm256_sub_epi16(r[1], s));
    r[2] = _mm256_abs_epi16(_mm256_sub_epi16(r[2], s));
    r[3] = _mm256_abs_epi16(_mm256_sub_epi16(r[3], s));

    // sum every abs diff
    sums_16[0] = _mm256_add_epi16(sums_16[0], r[0]);
    sums_16[1] = _mm256_add_epi16(sums_16[1], r[1]);
    sums_16[2] = _mm256_add_epi16(sums_16[2], r[2]);
    sums_16[3] = _mm256_add_epi16(sums_16[3], r[3]);

    src += src_stride;
    refs[0] += ref_stride;
    refs[1] += ref_stride;
    refs[2] += ref_stride;
    refs[3] += ref_stride;
  }
}

static VPX_FORCE_INLINE void highbd_sad16xNx4d_avx2(
    const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4],
    int ref_stride, uint32_t sad_array[4], int n) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src_ptr);
  uint16_t *refs[4];
  __m256i sums_16[4];
  __m256i sums_32[4];
  const int height = VPXMIN(16, n);
  const int num_iters = n / height;
  int i;

  refs[0] = CONVERT_TO_SHORTPTR(ref_array[0]);
  refs[1] = CONVERT_TO_SHORTPTR(ref_array[1]);
  refs[2] = CONVERT_TO_SHORTPTR(ref_array[2]);
  refs[3] = CONVERT_TO_SHORTPTR(ref_array[3]);
  sums_32[0] = _mm256_setzero_si256();
  sums_32[1] = _mm256_setzero_si256();
  sums_32[2] = _mm256_setzero_si256();
  sums_32[3] = _mm256_setzero_si256();

  for (i = 0; i < num_iters; ++i) {
    sums_16[0] = _mm256_setzero_si256();
    sums_16[1] = _mm256_setzero_si256();
    sums_16[2] = _mm256_setzero_si256();
    sums_16[3] = _mm256_setzero_si256();

    highbd_sad16xHx4d(sums_16, src, src_stride, refs, ref_stride, height);

    // sums_16 will outrange after 16 rows, so add current sums_16 to sums_32
    sums_32[0] = _mm256_add_epi32(
        sums_32[0],
        _mm256_add_epi32(
            _mm256_cvtepu16_epi32(_mm256_castsi256_si128(sums_16[0])),
            _mm256_cvtepu16_epi32(_mm256_extractf128_si256(sums_16[0], 1))));
    sums_32[1] = _mm256_add_epi32(
        sums_32[1],
        _mm256_add_epi32(
            _mm256_cvtepu16_epi32(_mm256_castsi256_si128(sums_16[1])),
            _mm256_cvtepu16_epi32(_mm256_extractf128_si256(sums_16[1], 1))));
    sums_32[2] = _mm256_add_epi32(
        sums_32[2],
        _mm256_add_epi32(
            _mm256_cvtepu16_epi32(_mm256_castsi256_si128(sums_16[2])),
            _mm256_cvtepu16_epi32(_mm256_extractf128_si256(sums_16[2], 1))));
    sums_32[3] = _mm256_add_epi32(
        sums_32[3],
        _mm256_add_epi32(
            _mm256_cvtepu16_epi32(_mm256_castsi256_si128(sums_16[3])),
            _mm256_cvtepu16_epi32(_mm256_extractf128_si256(sums_16[3], 1))));

    src += src_stride << 4;
  }
  calc_final_4(sums_32, sad_array);
}

#define HIGHBD_SAD16XNX4D(n)                                                   \
  void vpx_highbd_sad16x##n##x4d_avx2(const uint8_t *src, int src_stride,      \
                                      const uint8_t *const ref_array[4],       \
                                      int ref_stride, uint32_t sad_array[4]) { \
    highbd_sad16xNx4d_avx2(src, src_stride, ref_array, ref_stride, sad_array,  \
                           n);                                                 \
  }

#define HIGHBD_SADSKIP16XNx4D(n)                                             \
  void vpx_highbd_sad_skip_16x##n##x4d_avx2(                                 \
      const uint8_t *src, int src_stride, const uint8_t *const ref_array[4], \
      int ref_stride, uint32_t sad_array[4]) {                               \
    highbd_sad16xNx4d_avx2(src, 2 * src_stride, ref_array, 2 * ref_stride,   \
                           sad_array, n / 2);                                \
    sad_array[0] <<= 1;                                                      \
    sad_array[1] <<= 1;                                                      \
    sad_array[2] <<= 1;                                                      \
    sad_array[3] <<= 1;                                                      \
  }

void vpx_highbd_sad16x16x4d_avx2(const uint8_t *src_ptr, int src_stride,
                                 const uint8_t *const ref_array[4],
                                 int ref_stride, uint32_t sad_array[4]) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src_ptr);
  uint16_t *refs[4];
  __m256i sums_16[4];

  refs[0] = CONVERT_TO_SHORTPTR(ref_array[0]);
  refs[1] = CONVERT_TO_SHORTPTR(ref_array[1]);
  refs[2] = CONVERT_TO_SHORTPTR(ref_array[2]);
  refs[3] = CONVERT_TO_SHORTPTR(ref_array[3]);
  sums_16[0] = _mm256_setzero_si256();
  sums_16[1] = _mm256_setzero_si256();
  sums_16[2] = _mm256_setzero_si256();
  sums_16[3] = _mm256_setzero_si256();

  highbd_sad16xHx4d(sums_16, src, src_stride, refs, ref_stride, 16);

  {
    __m256i sums_32[4];
    sums_32[0] = _mm256_add_epi32(
        _mm256_cvtepu16_epi32(_mm256_castsi256_si128(sums_16[0])),
        _mm256_cvtepu16_epi32(_mm256_extractf128_si256(sums_16[0], 1)));
    sums_32[1] = _mm256_add_epi32(
        _mm256_cvtepu16_epi32(_mm256_castsi256_si128(sums_16[1])),
        _mm256_cvtepu16_epi32(_mm256_extractf128_si256(sums_16[1], 1)));
    sums_32[2] = _mm256_add_epi32(
        _mm256_cvtepu16_epi32(_mm256_castsi256_si128(sums_16[2])),
        _mm256_cvtepu16_epi32(_mm256_extractf128_si256(sums_16[2], 1)));
    sums_32[3] = _mm256_add_epi32(
        _mm256_cvtepu16_epi32(_mm256_castsi256_si128(sums_16[3])),
        _mm256_cvtepu16_epi32(_mm256_extractf128_si256(sums_16[3], 1)));
    calc_final_4(sums_32, sad_array);
  }
}

void vpx_highbd_sad16x8x4d_avx2(const uint8_t *src_ptr, int src_stride,
                                const uint8_t *const ref_array[4],
                                int ref_stride, uint32_t sad_array[4]) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src_ptr);
  uint16_t *refs[4];
  __m256i sums_16[4];

  refs[0] = CONVERT_TO_SHORTPTR(ref_array[0]);
  refs[1] = CONVERT_TO_SHORTPTR(ref_array[1]);
  refs[2] = CONVERT_TO_SHORTPTR(ref_array[2]);
  refs[3] = CONVERT_TO_SHORTPTR(ref_array[3]);
  sums_16[0] = _mm256_setzero_si256();
  sums_16[1] = _mm256_setzero_si256();
  sums_16[2] = _mm256_setzero_si256();
  sums_16[3] = _mm256_setzero_si256();

  highbd_sad16xHx4d(sums_16, src, src_stride, refs, ref_stride, 8);

  {
    __m256i sums_32[4];
    sums_32[0] = _mm256_add_epi32(
        _mm256_cvtepu16_epi32(_mm256_castsi256_si128(sums_16[0])),
        _mm256_cvtepu16_epi32(_mm256_extractf128_si256(sums_16[0], 1)));
    sums_32[1] = _mm256_add_epi32(
        _mm256_cvtepu16_epi32(_mm256_castsi256_si128(sums_16[1])),
        _mm256_cvtepu16_epi32(_mm256_extractf128_si256(sums_16[1], 1)));
    sums_32[2] = _mm256_add_epi32(
        _mm256_cvtepu16_epi32(_mm256_castsi256_si128(sums_16[2])),
        _mm256_cvtepu16_epi32(_mm256_extractf128_si256(sums_16[2], 1)));
    sums_32[3] = _mm256_add_epi32(
        _mm256_cvtepu16_epi32(_mm256_castsi256_si128(sums_16[3])),
        _mm256_cvtepu16_epi32(_mm256_extractf128_si256(sums_16[3], 1)));
    calc_final_4(sums_32, sad_array);
  }
}

// clang-format off
HIGHBD_SAD64XNX4D(64)
HIGHBD_SADSKIP64XNx4D(64)

HIGHBD_SAD64XNX4D(32)
HIGHBD_SADSKIP64XNx4D(32)

HIGHBD_SAD32XNX4D(64)
HIGHBD_SADSKIP32XNx4D(64)

HIGHBD_SAD32XNX4D(32)
HIGHBD_SADSKIP32XNx4D(32)

HIGHBD_SAD32XNX4D(16)
HIGHBD_SADSKIP32XNx4D(16)

HIGHBD_SAD16XNX4D(32)
HIGHBD_SADSKIP16XNx4D(32)

HIGHBD_SADSKIP16XNx4D(16)

HIGHBD_SADSKIP16XNx4D(8)
    // clang-format on
