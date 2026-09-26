/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <tmmintrin.h>

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/x86/inv_txfm_sse2.h"
#include "vpx_dsp/x86/inv_txfm_ssse3.h"
#include "vpx_dsp/x86/transpose_sse2.h"
#include "vpx_dsp/x86/txfm_common_sse2.h"

static INLINE void partial_butterfly_ssse3(const __m128i in, const int c0,
                                           const int c1, __m128i *const out0,
                                           __m128i *const out1) {
  const __m128i cst0 = _mm_set1_epi16(2 * c0);
  const __m128i cst1 = _mm_set1_epi16(2 * c1);
  *out0 = _mm_mulhrs_epi16(in, cst0);
  *out1 = _mm_mulhrs_epi16(in, cst1);
}

static INLINE __m128i partial_butterfly_cospi16_ssse3(const __m128i in) {
  const __m128i coef_pair = _mm_set1_epi16(2 * cospi_16_64);
  return _mm_mulhrs_epi16(in, coef_pair);
}

void vpx_idct8x8_12_add_ssse3(const tran_low_t *input, uint8_t *dest,
                              int stride) {
  __m128i io[8];

  io[0] = load_input_data4(input + 0 * 8);
  io[1] = load_input_data4(input + 1 * 8);
  io[2] = load_input_data4(input + 2 * 8);
  io[3] = load_input_data4(input + 3 * 8);

  idct8x8_12_add_kernel_ssse3(io);
  write_buffer_8x8(io, dest, stride);
}

// Group the coefficient calculation into smaller functions to prevent stack
// spillover in 32x32 idct optimizations:
// quarter_1: 0-7
// quarter_2: 8-15
// quarter_3_4: 16-23, 24-31

// For each 8x32 block __m128i in[32],
// Input with index, 0, 4
// output pixels: 0-7 in __m128i out[32]
static INLINE void idct32_34_8x32_quarter_1(const __m128i *const in /*in[32]*/,
                                            __m128i *const out /*out[8]*/) {
  __m128i step1[8], step2[8];

  // stage 3
  partial_butterfly_ssse3(in[4], cospi_28_64, cospi_4_64, &step1[4], &step1[7]);

  // stage 4
  step2[0] = partial_butterfly_cospi16_ssse3(in[0]);
  step2[4] = step1[4];
  step2[5] = step1[4];
  step2[6] = step1[7];
  step2[7] = step1[7];

  // stage 5
  step1[0] = step2[0];
  step1[1] = step2[0];
  step1[2] = step2[0];
  step1[3] = step2[0];
  step1[4] = step2[4];
  butterfly(step2[6], step2[5], cospi_16_64, cospi_16_64, &step1[5], &step1[6]);
  step1[7] = step2[7];

  // stage 6
  out[0] = _mm_add_epi16(step1[0], step1[7]);
  out[1] = _mm_add_epi16(step1[1], step1[6]);
  out[2] = _mm_add_epi16(step1[2], step1[5]);
  out[3] = _mm_add_epi16(step1[3], step1[4]);
  out[4] = _mm_sub_epi16(step1[3], step1[4]);
  out[5] = _mm_sub_epi16(step1[2], step1[5]);
  out[6] = _mm_sub_epi16(step1[1], step1[6]);
  out[7] = _mm_sub_epi16(step1[0], step1[7]);
}

// For each 8x32 block __m128i in[32],
// Input with index, 2, 6
// output pixels: 8-15 in __m128i out[32]
static INLINE void idct32_34_8x32_quarter_2(const __m128i *const in /*in[32]*/,
                                            __m128i *const out /*out[16]*/) {
  __m128i step1[16], step2[16];

  // stage 2
  partial_butterfly_ssse3(in[2], cospi_30_64, cospi_2_64, &step2[8],
                          &step2[15]);
  partial_butterfly_ssse3(in[6], -cospi_26_64, cospi_6_64, &step2[11],
                          &step2[12]);

  // stage 3
  step1[8] = step2[8];
  step1[9] = step2[8];
  step1[14] = step2[15];
  step1[15] = step2[15];
  step1[10] = step2[11];
  step1[11] = step2[11];
  step1[12] = step2[12];
  step1[13] = step2[12];

  idct32_8x32_quarter_2_stage_4_to_6(step1, out);
}

static INLINE void idct32_34_8x32_quarter_1_2(
    const __m128i *const in /*in[32]*/, __m128i *const out /*out[32]*/) {
  __m128i temp[16];
  idct32_34_8x32_quarter_1(in, temp);
  idct32_34_8x32_quarter_2(in, temp);
  // stage 7
  add_sub_butterfly(temp, out, 16);
}

// For each 8x32 block __m128i in[32],
// Input with odd index, 1, 3, 5, 7
// output pixels: 16-23, 24-31 in __m128i out[32]
static INLINE void idct32_34_8x32_quarter_3_4(
    const __m128i *const in /*in[32]*/, __m128i *const out /*out[32]*/) {
  __m128i step1[32];

  // stage 1
  partial_butterfly_ssse3(in[1], cospi_31_64, cospi_1_64, &step1[16],
                          &step1[31]);
  partial_butterfly_ssse3(in[7], -cospi_25_64, cospi_7_64, &step1[19],
                          &step1[28]);
  partial_butterfly_ssse3(in[5], cospi_27_64, cospi_5_64, &step1[20],
                          &step1[27]);
  partial_butterfly_ssse3(in[3], -cospi_29_64, cospi_3_64, &step1[23],
                          &step1[24]);

  // stage 3
  butterfly(step1[31], step1[16], cospi_28_64, cospi_4_64, &step1[17],
            &step1[30]);
  butterfly(step1[28], step1[19], -cospi_4_64, cospi_28_64, &step1[18],
            &step1[29]);
  butterfly(step1[27], step1[20], cospi_12_64, cospi_20_64, &step1[21],
            &step1[26]);
  butterfly(step1[24], step1[23], -cospi_20_64, cospi_12_64, &step1[22],
            &step1[25]);

  idct32_8x32_quarter_3_4_stage_4_to_7(step1, out);
}

void idct32_34_8x32_ssse3(const __m128i *const in /*in[32]*/,
                          __m128i *const out /*out[32]*/) {
  __m128i temp[32];

  idct32_34_8x32_quarter_1_2(in, temp);
  idct32_34_8x32_quarter_3_4(in, temp);
  // final stage
  add_sub_butterfly(temp, out, 32);
}

// Only upper-left 8x8 has non-zero coeff
void vpx_idct32x32_34_add_ssse3(const tran_low_t *input, uint8_t *dest,
                                int stride) {
  __m128i io[32], col[32];
  int i;

  // Load input data. Only need to load the top left 8x8 block.
  load_transpose_16bit_8x8(input, 32, io);
  idct32_34_8x32_ssse3(io, col);

  for (i = 0; i < 32; i += 8) {
    int j;
    transpose_16bit_8x8(col + i, io);
    idct32_34_8x32_ssse3(io, io);

    for (j = 0; j < 32; ++j) {
      write_buffer_8x1(dest + j * stride, io[j]);
    }

    dest += 8;
  }
}

// For each 8x32 block __m128i in[32],
// Input with index, 0, 4, 8, 12
// output pixels: 0-7 in __m128i out[32]
static INLINE void idct32_135_8x32_quarter_1(const __m128i *const in /*in[32]*/,
                                             __m128i *const out /*out[8]*/) {
  __m128i step1[8], step2[8];

  // stage 3
  partial_butterfly_ssse3(in[4], cospi_28_64, cospi_4_64, &step1[4], &step1[7]);
  partial_butterfly_ssse3(in[12], -cospi_20_64, cospi_12_64, &step1[5],
                          &step1[6]);

  // stage 4
  step2[0] = partial_butterfly_cospi16_ssse3(in[0]);
  partial_butterfly_ssse3(in[8], cospi_24_64, cospi_8_64, &step2[2], &step2[3]);
  step2[4] = _mm_add_epi16(step1[4], step1[5]);
  step2[5] = _mm_sub_epi16(step1[4], step1[5]);
  step2[6] = _mm_sub_epi16(step1[7], step1[6]);
  step2[7] = _mm_add_epi16(step1[7], step1[6]);

  // stage 5
  step1[0] = _mm_add_epi16(step2[0], step2[3]);
  step1[1] = _mm_add_epi16(step2[0], step2[2]);
  step1[2] = _mm_sub_epi16(step2[0], step2[2]);
  step1[3] = _mm_sub_epi16(step2[0], step2[3]);
  step1[4] = step2[4];
  butterfly(step2[6], step2[5], cospi_16_64, cospi_16_64, &step1[5], &step1[6]);
  step1[7] = step2[7];

  // stage 6
  out[0] = _mm_add_epi16(step1[0], step1[7]);
  out[1] = _mm_add_epi16(step1[1], step1[6]);
  out[2] = _mm_add_epi16(step1[2], step1[5]);
  out[3] = _mm_add_epi16(step1[3], step1[4]);
  out[4] = _mm_sub_epi16(step1[3], step1[4]);
  out[5] = _mm_sub_epi16(step1[2], step1[5]);
  out[6] = _mm_sub_epi16(step1[1], step1[6]);
  out[7] = _mm_sub_epi16(step1[0], step1[7]);
}

// For each 8x32 block __m128i in[32],
// Input with index, 2, 6, 10, 14
// output pixels: 8-15 in __m128i out[32]
static INLINE void idct32_135_8x32_quarter_2(const __m128i *const in /*in[32]*/,
                                             __m128i *const out /*out[16]*/) {
  __m128i step1[16], step2[16];

  // stage 2
  partial_butterfly_ssse3(in[2], cospi_30_64, cospi_2_64, &step2[8],
                          &step2[15]);
  partial_butterfly_ssse3(in[14], -cospi_18_64, cospi_14_64, &step2[9],
                          &step2[14]);
  partial_butterfly_ssse3(in[10], cospi_22_64, cospi_10_64, &step2[10],
                          &step2[13]);
  partial_butterfly_ssse3(in[6], -cospi_26_64, cospi_6_64, &step2[11],
                          &step2[12]);

  // stage 3
  step1[8] = _mm_add_epi16(step2[8], step2[9]);
  step1[9] = _mm_sub_epi16(step2[8], step2[9]);
  step1[10] = _mm_sub_epi16(step2[11], step2[10]);
  step1[11] = _mm_add_epi16(step2[11], step2[10]);
  step1[12] = _mm_add_epi16(step2[12], step2[13]);
  step1[13] = _mm_sub_epi16(step2[12], step2[13]);
  step1[14] = _mm_sub_epi16(step2[15], step2[14]);
  step1[15] = _mm_add_epi16(step2[15], step2[14]);

  idct32_8x32_quarter_2_stage_4_to_6(step1, out);
}

static INLINE void idct32_135_8x32_quarter_1_2(
    const __m128i *const in /*in[32]*/, __m128i *const out /*out[32]*/) {
  __m128i temp[16];
  idct32_135_8x32_quarter_1(in, temp);
  idct32_135_8x32_quarter_2(in, temp);
  // stage 7
  add_sub_butterfly(temp, out, 16);
}

// For each 8x32 block __m128i in[32],
// Input with odd index,
// 1, 3, 5, 7, 9, 11, 13, 15
// output pixels: 16-23, 24-31 in __m128i out[32]
static INLINE void idct32_135_8x32_quarter_3_4(
    const __m128i *const in /*in[32]*/, __m128i *const out /*out[32]*/) {
  __m128i step1[32], step2[32];

  // stage 1
  partial_butterfly_ssse3(in[1], cospi_31_64, cospi_1_64, &step1[16],
                          &step1[31]);
  partial_butterfly_ssse3(in[15], -cospi_17_64, cospi_15_64, &step1[17],
                          &step1[30]);
  partial_butterfly_ssse3(in[9], cospi_23_64, cospi_9_64, &step1[18],
                          &step1[29]);
  partial_butterfly_ssse3(in[7], -cospi_25_64, cospi_7_64, &step1[19],
                          &step1[28]);

  partial_butterfly_ssse3(in[5], cospi_27_64, cospi_5_64, &step1[20],
                          &step1[27]);
  partial_butterfly_ssse3(in[11], -cospi_21_64, cospi_11_64, &step1[21],
                          &step1[26]);

  partial_butterfly_ssse3(in[13], cospi_19_64, cospi_13_64, &step1[22],
                          &step1[25]);
  partial_butterfly_ssse3(in[3], -cospi_29_64, cospi_3_64, &step1[23],
                          &step1[24]);

  // stage 2
  step2[16] = _mm_add_epi16(step1[16], step1[17]);
  step2[17] = _mm_sub_epi16(step1[16], step1[17]);
  step2[18] = _mm_sub_epi16(step1[19], step1[18]);
  step2[19] = _mm_add_epi16(step1[19], step1[18]);
  step2[20] = _mm_add_epi16(step1[20], step1[21]);
  step2[21] = _mm_sub_epi16(step1[20], step1[21]);
  step2[22] = _mm_sub_epi16(step1[23], step1[22]);
  step2[23] = _mm_add_epi16(step1[23], step1[22]);

  step2[24] = _mm_add_epi16(step1[24], step1[25]);
  step2[25] = _mm_sub_epi16(step1[24], step1[25]);
  step2[26] = _mm_sub_epi16(step1[27], step1[26]);
  step2[27] = _mm_add_epi16(step1[27], step1[26]);
  step2[28] = _mm_add_epi16(step1[28], step1[29]);
  step2[29] = _mm_sub_epi16(step1[28], step1[29]);
  step2[30] = _mm_sub_epi16(step1[31], step1[30]);
  step2[31] = _mm_add_epi16(step1[31], step1[30]);

  // stage 3
  step1[16] = step2[16];
  step1[31] = step2[31];
  butterfly(step2[30], step2[17], cospi_28_64, cospi_4_64, &step1[17],
            &step1[30]);
  butterfly(step2[29], step2[18], -cospi_4_64, cospi_28_64, &step1[18],
            &step1[29]);
  step1[19] = step2[19];
  step1[20] = step2[20];
  butterfly(step2[26], step2[21], cospi_12_64, cospi_20_64, &step1[21],
            &step1[26]);
  butterfly(step2[25], step2[22], -cospi_20_64, cospi_12_64, &step1[22],
            &step1[25]);
  step1[23] = step2[23];
  step1[24] = step2[24];
  step1[27] = step2[27];
  step1[28] = step2[28];

  idct32_8x32_quarter_3_4_stage_4_to_7(step1, out);
}

void idct32_135_8x32_ssse3(const __m128i *const in /*in[32]*/,
                           __m128i *const out /*out[32]*/) {
  __m128i temp[32];
  idct32_135_8x32_quarter_1_2(in, temp);
  idct32_135_8x32_quarter_3_4(in, temp);
  // final stage
  add_sub_butterfly(temp, out, 32);
}

void vpx_idct32x32_135_add_ssse3(const tran_low_t *input, uint8_t *dest,
                                 int stride) {
  __m128i col[2][32], io[32];
  int i;

  // rows
  for (i = 0; i < 2; i++) {
    load_transpose_16bit_8x8(&input[0], 32, &io[0]);
    load_transpose_16bit_8x8(&input[8], 32, &io[8]);
    idct32_135_8x32_ssse3(io, col[i]);
    input += 32 << 3;
  }

  // columns
  for (i = 0; i < 32; i += 8) {
    transpose_16bit_8x8(col[0] + i, io);
    transpose_16bit_8x8(col[1] + i, io + 8);
    idct32_135_8x32_ssse3(io, io);
    store_buffer_8x32(io, dest, stride);
    dest += 8;
  }
}
