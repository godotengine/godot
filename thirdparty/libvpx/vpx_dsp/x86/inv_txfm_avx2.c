/*
 *  Copyright (c) 2023 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <immintrin.h>  // AVX2

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/txfm_common.h"

#define PAIR256_SET_EPI16(a, b)                                            \
  _mm256_set_epi16((int16_t)(b), (int16_t)(a), (int16_t)(b), (int16_t)(a), \
                   (int16_t)(b), (int16_t)(a), (int16_t)(b), (int16_t)(a), \
                   (int16_t)(b), (int16_t)(a), (int16_t)(b), (int16_t)(a), \
                   (int16_t)(b), (int16_t)(a), (int16_t)(b), (int16_t)(a))

static INLINE void idct_load16x16(const tran_low_t *input, __m256i *in,
                                  int stride) {
  int i;
  // Load 16x16 values
  for (i = 0; i < 16; i++) {
#if CONFIG_VP9_HIGHBITDEPTH
    const __m128i in0 = _mm_loadu_si128((const __m128i *)(input + i * stride));
    const __m128i in1 =
        _mm_loadu_si128((const __m128i *)((input + i * stride) + 4));
    const __m128i in2 =
        _mm_loadu_si128((const __m128i *)((input + i * stride) + 8));
    const __m128i in3 =
        _mm_loadu_si128((const __m128i *)((input + i * stride) + 12));
    const __m128i ls = _mm_packs_epi32(in0, in1);
    const __m128i rs = _mm_packs_epi32(in2, in3);
    in[i] = _mm256_inserti128_si256(_mm256_castsi128_si256(ls), rs, 1);
#else
    in[i] = _mm256_load_si256((const __m256i *)(input + i * stride));
#endif
  }
}

static INLINE __m256i dct_round_shift_avx2(__m256i in) {
  const __m256i t = _mm256_add_epi32(in, _mm256_set1_epi32(DCT_CONST_ROUNDING));
  return _mm256_srai_epi32(t, DCT_CONST_BITS);
}

static INLINE __m256i idct_madd_round_shift_avx2(__m256i *in, __m256i *cospi) {
  const __m256i t = _mm256_madd_epi16(*in, *cospi);
  return dct_round_shift_avx2(t);
}

// Calculate the dot product between in0/1 and x and wrap to short.
static INLINE __m256i idct_calc_wraplow_avx2(__m256i *in0, __m256i *in1,
                                             __m256i *x) {
  const __m256i t0 = idct_madd_round_shift_avx2(in0, x);
  const __m256i t1 = idct_madd_round_shift_avx2(in1, x);
  return _mm256_packs_epi32(t0, t1);
}

// Multiply elements by constants and add them together.
static INLINE void butterfly16(__m256i in0, __m256i in1, int c0, int c1,
                               __m256i *out0, __m256i *out1) {
  __m256i cst0 = PAIR256_SET_EPI16(c0, -c1);
  __m256i cst1 = PAIR256_SET_EPI16(c1, c0);
  __m256i lo = _mm256_unpacklo_epi16(in0, in1);
  __m256i hi = _mm256_unpackhi_epi16(in0, in1);
  *out0 = idct_calc_wraplow_avx2(&lo, &hi, &cst0);
  *out1 = idct_calc_wraplow_avx2(&lo, &hi, &cst1);
}

static INLINE void idct16_16col(__m256i *in, __m256i *out) {
  __m256i step1[16], step2[16];

  // stage 2
  butterfly16(in[1], in[15], cospi_30_64, cospi_2_64, &step2[8], &step2[15]);
  butterfly16(in[9], in[7], cospi_14_64, cospi_18_64, &step2[9], &step2[14]);
  butterfly16(in[5], in[11], cospi_22_64, cospi_10_64, &step2[10], &step2[13]);
  butterfly16(in[13], in[3], cospi_6_64, cospi_26_64, &step2[11], &step2[12]);

  // stage 3
  butterfly16(in[2], in[14], cospi_28_64, cospi_4_64, &step1[4], &step1[7]);
  butterfly16(in[10], in[6], cospi_12_64, cospi_20_64, &step1[5], &step1[6]);
  step1[8] = _mm256_add_epi16(step2[8], step2[9]);
  step1[9] = _mm256_sub_epi16(step2[8], step2[9]);
  step1[10] = _mm256_sub_epi16(step2[11], step2[10]);
  step1[11] = _mm256_add_epi16(step2[10], step2[11]);
  step1[12] = _mm256_add_epi16(step2[12], step2[13]);
  step1[13] = _mm256_sub_epi16(step2[12], step2[13]);
  step1[14] = _mm256_sub_epi16(step2[15], step2[14]);
  step1[15] = _mm256_add_epi16(step2[14], step2[15]);

  // stage 4
  butterfly16(in[0], in[8], cospi_16_64, cospi_16_64, &step2[1], &step2[0]);
  butterfly16(in[4], in[12], cospi_24_64, cospi_8_64, &step2[2], &step2[3]);
  butterfly16(step1[14], step1[9], cospi_24_64, cospi_8_64, &step2[9],
              &step2[14]);
  butterfly16(step1[10], step1[13], -cospi_8_64, -cospi_24_64, &step2[13],
              &step2[10]);
  step2[5] = _mm256_sub_epi16(step1[4], step1[5]);
  step1[4] = _mm256_add_epi16(step1[4], step1[5]);
  step2[6] = _mm256_sub_epi16(step1[7], step1[6]);
  step1[7] = _mm256_add_epi16(step1[6], step1[7]);
  step2[8] = step1[8];
  step2[11] = step1[11];
  step2[12] = step1[12];
  step2[15] = step1[15];

  // stage 5
  step1[0] = _mm256_add_epi16(step2[0], step2[3]);
  step1[1] = _mm256_add_epi16(step2[1], step2[2]);
  step1[2] = _mm256_sub_epi16(step2[1], step2[2]);
  step1[3] = _mm256_sub_epi16(step2[0], step2[3]);
  butterfly16(step2[6], step2[5], cospi_16_64, cospi_16_64, &step1[5],
              &step1[6]);
  step1[8] = _mm256_add_epi16(step2[8], step2[11]);
  step1[9] = _mm256_add_epi16(step2[9], step2[10]);
  step1[10] = _mm256_sub_epi16(step2[9], step2[10]);
  step1[11] = _mm256_sub_epi16(step2[8], step2[11]);
  step1[12] = _mm256_sub_epi16(step2[15], step2[12]);
  step1[13] = _mm256_sub_epi16(step2[14], step2[13]);
  step1[14] = _mm256_add_epi16(step2[14], step2[13]);
  step1[15] = _mm256_add_epi16(step2[15], step2[12]);

  // stage 6
  step2[0] = _mm256_add_epi16(step1[0], step1[7]);
  step2[1] = _mm256_add_epi16(step1[1], step1[6]);
  step2[2] = _mm256_add_epi16(step1[2], step1[5]);
  step2[3] = _mm256_add_epi16(step1[3], step1[4]);
  step2[4] = _mm256_sub_epi16(step1[3], step1[4]);
  step2[5] = _mm256_sub_epi16(step1[2], step1[5]);
  step2[6] = _mm256_sub_epi16(step1[1], step1[6]);
  step2[7] = _mm256_sub_epi16(step1[0], step1[7]);
  butterfly16(step1[13], step1[10], cospi_16_64, cospi_16_64, &step2[10],
              &step2[13]);
  butterfly16(step1[12], step1[11], cospi_16_64, cospi_16_64, &step2[11],
              &step2[12]);

  // stage 7
  out[0] = _mm256_add_epi16(step2[0], step1[15]);
  out[1] = _mm256_add_epi16(step2[1], step1[14]);
  out[2] = _mm256_add_epi16(step2[2], step2[13]);
  out[3] = _mm256_add_epi16(step2[3], step2[12]);
  out[4] = _mm256_add_epi16(step2[4], step2[11]);
  out[5] = _mm256_add_epi16(step2[5], step2[10]);
  out[6] = _mm256_add_epi16(step2[6], step1[9]);
  out[7] = _mm256_add_epi16(step2[7], step1[8]);
  out[8] = _mm256_sub_epi16(step2[7], step1[8]);
  out[9] = _mm256_sub_epi16(step2[6], step1[9]);
  out[10] = _mm256_sub_epi16(step2[5], step2[10]);
  out[11] = _mm256_sub_epi16(step2[4], step2[11]);
  out[12] = _mm256_sub_epi16(step2[3], step2[12]);
  out[13] = _mm256_sub_epi16(step2[2], step2[13]);
  out[14] = _mm256_sub_epi16(step2[1], step1[14]);
  out[15] = _mm256_sub_epi16(step2[0], step1[15]);
}

static INLINE void recon_and_store16(uint8_t *dest, __m256i in_x) {
  const __m256i zero = _mm256_setzero_si256();
  __m256i d0 = _mm256_castsi128_si256(_mm_loadu_si128((__m128i *)(dest)));
  d0 = _mm256_permute4x64_epi64(d0, 0xd8);
  d0 = _mm256_unpacklo_epi8(d0, zero);
  d0 = _mm256_add_epi16(in_x, d0);
  d0 = _mm256_packus_epi16(
      d0, _mm256_castsi128_si256(_mm256_extractf128_si256(d0, 1)));

  _mm_storeu_si128((__m128i *)dest, _mm256_castsi256_si128(d0));
}

static INLINE void write_buffer_16x1(uint8_t *dest, __m256i in) {
  const __m256i final_rounding = _mm256_set1_epi16(1 << 5);
  __m256i out;
  out = _mm256_adds_epi16(in, final_rounding);
  out = _mm256_srai_epi16(out, 6);
  recon_and_store16(dest, out);
}

static INLINE void store_buffer_16x32(__m256i *in, uint8_t *dst, int stride) {
  const __m256i final_rounding = _mm256_set1_epi16(1 << 5);
  int j = 0;
  while (j < 32) {
    in[j] = _mm256_adds_epi16(in[j], final_rounding);
    in[j + 1] = _mm256_adds_epi16(in[j + 1], final_rounding);

    in[j] = _mm256_srai_epi16(in[j], 6);
    in[j + 1] = _mm256_srai_epi16(in[j + 1], 6);

    recon_and_store16(dst, in[j]);
    dst += stride;
    recon_and_store16(dst, in[j + 1]);
    dst += stride;
    j += 2;
  }
}

static INLINE void transpose2_8x8_avx2(__m256i *in, __m256i *out) {
  int i;
  __m256i t[16], u[16];
  // (1st, 2nd) ==> (lo, hi)
  //   (0, 1)   ==>  (0, 1)
  //   (2, 3)   ==>  (2, 3)
  //   (4, 5)   ==>  (4, 5)
  //   (6, 7)   ==>  (6, 7)
  for (i = 0; i < 4; i++) {
    t[2 * i] = _mm256_unpacklo_epi16(in[2 * i], in[2 * i + 1]);
    t[2 * i + 1] = _mm256_unpackhi_epi16(in[2 * i], in[2 * i + 1]);
  }

  // (1st, 2nd) ==> (lo, hi)
  //   (0, 2)   ==>  (0, 2)
  //   (1, 3)   ==>  (1, 3)
  //   (4, 6)   ==>  (4, 6)
  //   (5, 7)   ==>  (5, 7)
  for (i = 0; i < 2; i++) {
    u[i] = _mm256_unpacklo_epi32(t[i], t[i + 2]);
    u[i + 2] = _mm256_unpackhi_epi32(t[i], t[i + 2]);

    u[i + 4] = _mm256_unpacklo_epi32(t[i + 4], t[i + 6]);
    u[i + 6] = _mm256_unpackhi_epi32(t[i + 4], t[i + 6]);
  }

  // (1st, 2nd) ==> (lo, hi)
  //   (0, 4)   ==>  (0, 1)
  //   (1, 5)   ==>  (4, 5)
  //   (2, 6)   ==>  (2, 3)
  //   (3, 7)   ==>  (6, 7)
  for (i = 0; i < 2; i++) {
    out[2 * i] = _mm256_unpacklo_epi64(u[2 * i], u[2 * i + 4]);
    out[2 * i + 1] = _mm256_unpackhi_epi64(u[2 * i], u[2 * i + 4]);

    out[2 * i + 4] = _mm256_unpacklo_epi64(u[2 * i + 1], u[2 * i + 5]);
    out[2 * i + 5] = _mm256_unpackhi_epi64(u[2 * i + 1], u[2 * i + 5]);
  }
}

static INLINE void transpose_16bit_16x16_avx2(__m256i *in, __m256i *out) {
  __m256i t[16];

#define LOADL(idx)                                                            \
  t[idx] = _mm256_castsi128_si256(_mm_load_si128((__m128i const *)&in[idx])); \
  t[idx] = _mm256_inserti128_si256(                                           \
      t[idx], _mm_load_si128((__m128i const *)&in[(idx) + 8]), 1);

#define LOADR(idx)                                                           \
  t[8 + (idx)] =                                                             \
      _mm256_castsi128_si256(_mm_load_si128((__m128i const *)&in[idx] + 1)); \
  t[8 + (idx)] = _mm256_inserti128_si256(                                    \
      t[8 + (idx)], _mm_load_si128((__m128i const *)&in[(idx) + 8] + 1), 1);

  // load left 8x16
  LOADL(0)
  LOADL(1)
  LOADL(2)
  LOADL(3)
  LOADL(4)
  LOADL(5)
  LOADL(6)
  LOADL(7)

  // load right 8x16
  LOADR(0)
  LOADR(1)
  LOADR(2)
  LOADR(3)
  LOADR(4)
  LOADR(5)
  LOADR(6)
  LOADR(7)

  // get the top 16x8 result
  transpose2_8x8_avx2(t, out);
  // get the bottom 16x8 result
  transpose2_8x8_avx2(&t[8], &out[8]);
}

void vpx_idct16x16_256_add_avx2(const tran_low_t *input, uint8_t *dest,
                                int stride) {
  int i;
  __m256i in[16];

  // Load 16x16 values
  idct_load16x16(input, in, 16);

  transpose_16bit_16x16_avx2(in, in);
  idct16_16col(in, in);

  transpose_16bit_16x16_avx2(in, in);
  idct16_16col(in, in);

  for (i = 0; i < 16; ++i) {
    write_buffer_16x1(dest + i * stride, in[i]);
  }
}

// Only do addition and subtraction butterfly, size = 16, 32
static INLINE void add_sub_butterfly_avx2(__m256i *in, __m256i *out, int size) {
  int i = 0;
  const int num = size >> 1;
  const int bound = size - 1;
  while (i < num) {
    out[i] = _mm256_add_epi16(in[i], in[bound - i]);
    out[bound - i] = _mm256_sub_epi16(in[i], in[bound - i]);
    i++;
  }
}

// For each 16x32 block __m256i in[32],
// Input with index, 0, 4, 8, 12, 16, 20, 24, 28
// output pixels: 0-7 in __m256i out[32]
static INLINE void idct32_1024_16x32_quarter_1(__m256i *in, __m256i *out) {
  __m256i step1[8], step2[8];

  // stage 3
  butterfly16(in[4], in[28], cospi_28_64, cospi_4_64, &step1[4], &step1[7]);
  butterfly16(in[20], in[12], cospi_12_64, cospi_20_64, &step1[5], &step1[6]);

  // stage 4
  butterfly16(in[0], in[16], cospi_16_64, cospi_16_64, &step2[1], &step2[0]);
  butterfly16(in[8], in[24], cospi_24_64, cospi_8_64, &step2[2], &step2[3]);
  step2[4] = _mm256_add_epi16(step1[4], step1[5]);
  step2[5] = _mm256_sub_epi16(step1[4], step1[5]);
  step2[6] = _mm256_sub_epi16(step1[7], step1[6]);
  step2[7] = _mm256_add_epi16(step1[7], step1[6]);

  // stage 5
  step1[0] = _mm256_add_epi16(step2[0], step2[3]);
  step1[1] = _mm256_add_epi16(step2[1], step2[2]);
  step1[2] = _mm256_sub_epi16(step2[1], step2[2]);
  step1[3] = _mm256_sub_epi16(step2[0], step2[3]);
  step1[4] = step2[4];
  butterfly16(step2[6], step2[5], cospi_16_64, cospi_16_64, &step1[5],
              &step1[6]);
  step1[7] = step2[7];

  // stage 6
  out[0] = _mm256_add_epi16(step1[0], step1[7]);
  out[1] = _mm256_add_epi16(step1[1], step1[6]);
  out[2] = _mm256_add_epi16(step1[2], step1[5]);
  out[3] = _mm256_add_epi16(step1[3], step1[4]);
  out[4] = _mm256_sub_epi16(step1[3], step1[4]);
  out[5] = _mm256_sub_epi16(step1[2], step1[5]);
  out[6] = _mm256_sub_epi16(step1[1], step1[6]);
  out[7] = _mm256_sub_epi16(step1[0], step1[7]);
}

static INLINE void idct32_16x32_quarter_2_stage_4_to_6(__m256i *step1,
                                                       __m256i *out) {
  __m256i step2[32];

  // stage 4
  step2[8] = step1[8];
  step2[15] = step1[15];
  butterfly16(step1[14], step1[9], cospi_24_64, cospi_8_64, &step2[9],
              &step2[14]);
  butterfly16(step1[13], step1[10], -cospi_8_64, cospi_24_64, &step2[10],
              &step2[13]);
  step2[11] = step1[11];
  step2[12] = step1[12];

  // stage 5
  step1[8] = _mm256_add_epi16(step2[8], step2[11]);
  step1[9] = _mm256_add_epi16(step2[9], step2[10]);
  step1[10] = _mm256_sub_epi16(step2[9], step2[10]);
  step1[11] = _mm256_sub_epi16(step2[8], step2[11]);
  step1[12] = _mm256_sub_epi16(step2[15], step2[12]);
  step1[13] = _mm256_sub_epi16(step2[14], step2[13]);
  step1[14] = _mm256_add_epi16(step2[14], step2[13]);
  step1[15] = _mm256_add_epi16(step2[15], step2[12]);

  // stage 6
  out[8] = step1[8];
  out[9] = step1[9];
  butterfly16(step1[13], step1[10], cospi_16_64, cospi_16_64, &out[10],
              &out[13]);
  butterfly16(step1[12], step1[11], cospi_16_64, cospi_16_64, &out[11],
              &out[12]);
  out[14] = step1[14];
  out[15] = step1[15];
}

// For each 16x32 block __m256i in[32],
// Input with index, 2, 6, 10, 14, 18, 22, 26, 30
// output pixels: 8-15 in __m256i out[32]
static INLINE void idct32_1024_16x32_quarter_2(__m256i *in, __m256i *out) {
  __m256i step1[16], step2[16];

  // stage 2
  butterfly16(in[2], in[30], cospi_30_64, cospi_2_64, &step2[8], &step2[15]);
  butterfly16(in[18], in[14], cospi_14_64, cospi_18_64, &step2[9], &step2[14]);
  butterfly16(in[10], in[22], cospi_22_64, cospi_10_64, &step2[10], &step2[13]);
  butterfly16(in[26], in[6], cospi_6_64, cospi_26_64, &step2[11], &step2[12]);

  // stage 3
  step1[8] = _mm256_add_epi16(step2[8], step2[9]);
  step1[9] = _mm256_sub_epi16(step2[8], step2[9]);
  step1[10] = _mm256_sub_epi16(step2[11], step2[10]);
  step1[11] = _mm256_add_epi16(step2[11], step2[10]);
  step1[12] = _mm256_add_epi16(step2[12], step2[13]);
  step1[13] = _mm256_sub_epi16(step2[12], step2[13]);
  step1[14] = _mm256_sub_epi16(step2[15], step2[14]);
  step1[15] = _mm256_add_epi16(step2[15], step2[14]);

  idct32_16x32_quarter_2_stage_4_to_6(step1, out);
}

static INLINE void idct32_16x32_quarter_3_4_stage_4_to_7(__m256i *step1,
                                                         __m256i *out) {
  __m256i step2[32];

  // stage 4
  step2[16] = _mm256_add_epi16(step1[16], step1[19]);
  step2[17] = _mm256_add_epi16(step1[17], step1[18]);
  step2[18] = _mm256_sub_epi16(step1[17], step1[18]);
  step2[19] = _mm256_sub_epi16(step1[16], step1[19]);
  step2[20] = _mm256_sub_epi16(step1[23], step1[20]);
  step2[21] = _mm256_sub_epi16(step1[22], step1[21]);
  step2[22] = _mm256_add_epi16(step1[22], step1[21]);
  step2[23] = _mm256_add_epi16(step1[23], step1[20]);

  step2[24] = _mm256_add_epi16(step1[24], step1[27]);
  step2[25] = _mm256_add_epi16(step1[25], step1[26]);
  step2[26] = _mm256_sub_epi16(step1[25], step1[26]);
  step2[27] = _mm256_sub_epi16(step1[24], step1[27]);
  step2[28] = _mm256_sub_epi16(step1[31], step1[28]);
  step2[29] = _mm256_sub_epi16(step1[30], step1[29]);
  step2[30] = _mm256_add_epi16(step1[29], step1[30]);
  step2[31] = _mm256_add_epi16(step1[28], step1[31]);

  // stage 5
  step1[16] = step2[16];
  step1[17] = step2[17];
  butterfly16(step2[29], step2[18], cospi_24_64, cospi_8_64, &step1[18],
              &step1[29]);
  butterfly16(step2[28], step2[19], cospi_24_64, cospi_8_64, &step1[19],
              &step1[28]);
  butterfly16(step2[27], step2[20], -cospi_8_64, cospi_24_64, &step1[20],
              &step1[27]);
  butterfly16(step2[26], step2[21], -cospi_8_64, cospi_24_64, &step1[21],
              &step1[26]);
  step1[22] = step2[22];
  step1[23] = step2[23];
  step1[24] = step2[24];
  step1[25] = step2[25];
  step1[30] = step2[30];
  step1[31] = step2[31];

  // stage 6
  out[16] = _mm256_add_epi16(step1[16], step1[23]);
  out[17] = _mm256_add_epi16(step1[17], step1[22]);
  out[18] = _mm256_add_epi16(step1[18], step1[21]);
  out[19] = _mm256_add_epi16(step1[19], step1[20]);
  step2[20] = _mm256_sub_epi16(step1[19], step1[20]);
  step2[21] = _mm256_sub_epi16(step1[18], step1[21]);
  step2[22] = _mm256_sub_epi16(step1[17], step1[22]);
  step2[23] = _mm256_sub_epi16(step1[16], step1[23]);

  step2[24] = _mm256_sub_epi16(step1[31], step1[24]);
  step2[25] = _mm256_sub_epi16(step1[30], step1[25]);
  step2[26] = _mm256_sub_epi16(step1[29], step1[26]);
  step2[27] = _mm256_sub_epi16(step1[28], step1[27]);
  out[28] = _mm256_add_epi16(step1[27], step1[28]);
  out[29] = _mm256_add_epi16(step1[26], step1[29]);
  out[30] = _mm256_add_epi16(step1[25], step1[30]);
  out[31] = _mm256_add_epi16(step1[24], step1[31]);

  // stage 7
  butterfly16(step2[27], step2[20], cospi_16_64, cospi_16_64, &out[20],
              &out[27]);
  butterfly16(step2[26], step2[21], cospi_16_64, cospi_16_64, &out[21],
              &out[26]);
  butterfly16(step2[25], step2[22], cospi_16_64, cospi_16_64, &out[22],
              &out[25]);
  butterfly16(step2[24], step2[23], cospi_16_64, cospi_16_64, &out[23],
              &out[24]);
}

static INLINE void idct32_1024_16x32_quarter_1_2(__m256i *in, __m256i *out) {
  __m256i temp[16];

  // For each 16x32 block __m256i in[32],
  // Input with index, 0, 4, 8, 12, 16, 20, 24, 28
  // output pixels: 0-7 in __m256i out[32]
  idct32_1024_16x32_quarter_1(in, temp);

  // Input with index, 2, 6, 10, 14, 18, 22, 26, 30
  // output pixels: 8-15 in __m256i out[32]
  idct32_1024_16x32_quarter_2(in, temp);

  // stage 7
  add_sub_butterfly_avx2(temp, out, 16);
}

// For each 16x32 block __m256i in[32],
// Input with odd index,
// 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31
// output pixels: 16-23, 24-31 in __m256i out[32]
static INLINE void idct32_1024_16x32_quarter_3_4(__m256i *in, __m256i *out) {
  __m256i step1[32], step2[32];

  // stage 1
  butterfly16(in[1], in[31], cospi_31_64, cospi_1_64, &step1[16], &step1[31]);
  butterfly16(in[17], in[15], cospi_15_64, cospi_17_64, &step1[17], &step1[30]);
  butterfly16(in[9], in[23], cospi_23_64, cospi_9_64, &step1[18], &step1[29]);
  butterfly16(in[25], in[7], cospi_7_64, cospi_25_64, &step1[19], &step1[28]);

  butterfly16(in[5], in[27], cospi_27_64, cospi_5_64, &step1[20], &step1[27]);
  butterfly16(in[21], in[11], cospi_11_64, cospi_21_64, &step1[21], &step1[26]);

  butterfly16(in[13], in[19], cospi_19_64, cospi_13_64, &step1[22], &step1[25]);
  butterfly16(in[29], in[3], cospi_3_64, cospi_29_64, &step1[23], &step1[24]);

  // stage 2
  step2[16] = _mm256_add_epi16(step1[16], step1[17]);
  step2[17] = _mm256_sub_epi16(step1[16], step1[17]);
  step2[18] = _mm256_sub_epi16(step1[19], step1[18]);
  step2[19] = _mm256_add_epi16(step1[19], step1[18]);
  step2[20] = _mm256_add_epi16(step1[20], step1[21]);
  step2[21] = _mm256_sub_epi16(step1[20], step1[21]);
  step2[22] = _mm256_sub_epi16(step1[23], step1[22]);
  step2[23] = _mm256_add_epi16(step1[23], step1[22]);

  step2[24] = _mm256_add_epi16(step1[24], step1[25]);
  step2[25] = _mm256_sub_epi16(step1[24], step1[25]);
  step2[26] = _mm256_sub_epi16(step1[27], step1[26]);
  step2[27] = _mm256_add_epi16(step1[27], step1[26]);
  step2[28] = _mm256_add_epi16(step1[28], step1[29]);
  step2[29] = _mm256_sub_epi16(step1[28], step1[29]);
  step2[30] = _mm256_sub_epi16(step1[31], step1[30]);
  step2[31] = _mm256_add_epi16(step1[31], step1[30]);

  // stage 3
  step1[16] = step2[16];
  step1[31] = step2[31];
  butterfly16(step2[30], step2[17], cospi_28_64, cospi_4_64, &step1[17],
              &step1[30]);
  butterfly16(step2[29], step2[18], -cospi_4_64, cospi_28_64, &step1[18],
              &step1[29]);
  step1[19] = step2[19];
  step1[20] = step2[20];
  butterfly16(step2[26], step2[21], cospi_12_64, cospi_20_64, &step1[21],
              &step1[26]);
  butterfly16(step2[25], step2[22], -cospi_20_64, cospi_12_64, &step1[22],
              &step1[25]);
  step1[23] = step2[23];
  step1[24] = step2[24];
  step1[27] = step2[27];
  step1[28] = step2[28];

  idct32_16x32_quarter_3_4_stage_4_to_7(step1, out);
}

static INLINE void idct32_1024_16x32(__m256i *in, __m256i *out) {
  __m256i temp[32];

  // For each 16x32 block __m256i in[32],
  // Input with index, 0, 4, 8, 12, 16, 20, 24, 28
  // output pixels: 0-7 in __m256i out[32]
  // AND
  // Input with index, 2, 6, 10, 14, 18, 22, 26, 30
  // output pixels: 8-15 in __m256i out[32]
  idct32_1024_16x32_quarter_1_2(in, temp);

  // For each 16x32 block __m256i in[32],
  // Input with odd index,
  // 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31
  // output pixels: 16-23, 24-31 in __m256i out[32]
  idct32_1024_16x32_quarter_3_4(in, temp);

  // final stage
  add_sub_butterfly_avx2(temp, out, 32);
}

void vpx_idct32x32_1024_add_avx2(const tran_low_t *input, uint8_t *dest,
                                 int stride) {
  __m256i l[32], r[32], out[32], *in;
  int i;

  in = l;

  for (i = 0; i < 2; i++) {
    idct_load16x16(input, in, 32);
    transpose_16bit_16x16_avx2(in, in);

    idct_load16x16(input + 16, in + 16, 32);
    transpose_16bit_16x16_avx2(in + 16, in + 16);
    idct32_1024_16x32(in, in);

    in = r;
    input += 32 << 4;
  }

  for (i = 0; i < 32; i += 16) {
    transpose_16bit_16x16_avx2(l + i, out);
    transpose_16bit_16x16_avx2(r + i, out + 16);
    idct32_1024_16x32(out, out);

    store_buffer_16x32(out, dest, stride);
    dest += 16;
  }
}

// Case when only upper-left 16x16 has non-zero coeff
void vpx_idct32x32_135_add_avx2(const tran_low_t *input, uint8_t *dest,
                                int stride) {
  __m256i in[32], io[32], out[32];
  int i;

  for (i = 16; i < 32; i++) {
    in[i] = _mm256_setzero_si256();
  }

  // rows
  idct_load16x16(input, in, 32);
  transpose_16bit_16x16_avx2(in, in);
  idct32_1024_16x32(in, io);

  // columns
  for (i = 0; i < 32; i += 16) {
    transpose_16bit_16x16_avx2(io + i, in);
    idct32_1024_16x32(in, out);

    store_buffer_16x32(out, dest, stride);
    dest += 16;
  }
}
