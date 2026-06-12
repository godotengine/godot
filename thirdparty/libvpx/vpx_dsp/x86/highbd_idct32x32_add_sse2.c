/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/x86/highbd_inv_txfm_sse2.h"
#include "vpx_dsp/x86/inv_txfm_sse2.h"
#include "vpx_dsp/x86/transpose_sse2.h"
#include "vpx_dsp/x86/txfm_common_sse2.h"

static INLINE void highbd_idct32_4x32_quarter_2_stage_4_to_6(
    __m128i *const step1 /*step1[16]*/, __m128i *const out /*out[16]*/) {
  __m128i step2[32];

  // stage 4
  step2[8] = step1[8];
  step2[15] = step1[15];
  highbd_butterfly_sse2(step1[14], step1[9], cospi_24_64, cospi_8_64, &step2[9],
                        &step2[14]);
  highbd_butterfly_sse2(step1[10], step1[13], cospi_8_64, cospi_24_64,
                        &step2[13], &step2[10]);
  step2[11] = step1[11];
  step2[12] = step1[12];

  // stage 5
  step1[8] = _mm_add_epi32(step2[8], step2[11]);
  step1[9] = _mm_add_epi32(step2[9], step2[10]);
  step1[10] = _mm_sub_epi32(step2[9], step2[10]);
  step1[11] = _mm_sub_epi32(step2[8], step2[11]);
  step1[12] = _mm_sub_epi32(step2[15], step2[12]);
  step1[13] = _mm_sub_epi32(step2[14], step2[13]);
  step1[14] = _mm_add_epi32(step2[14], step2[13]);
  step1[15] = _mm_add_epi32(step2[15], step2[12]);

  // stage 6
  out[8] = step1[8];
  out[9] = step1[9];
  highbd_butterfly_sse2(step1[13], step1[10], cospi_16_64, cospi_16_64,
                        &out[10], &out[13]);
  highbd_butterfly_sse2(step1[12], step1[11], cospi_16_64, cospi_16_64,
                        &out[11], &out[12]);
  out[14] = step1[14];
  out[15] = step1[15];
}

static INLINE void highbd_idct32_4x32_quarter_3_4_stage_4_to_7(
    __m128i *const step1 /*step1[32]*/, __m128i *const out /*out[32]*/) {
  __m128i step2[32];

  // stage 4
  step2[16] = _mm_add_epi32(step1[16], step1[19]);
  step2[17] = _mm_add_epi32(step1[17], step1[18]);
  step2[18] = _mm_sub_epi32(step1[17], step1[18]);
  step2[19] = _mm_sub_epi32(step1[16], step1[19]);
  step2[20] = _mm_sub_epi32(step1[20], step1[23]);  // step2[20] = -step2[20]
  step2[21] = _mm_sub_epi32(step1[21], step1[22]);  // step2[21] = -step2[21]
  step2[22] = _mm_add_epi32(step1[21], step1[22]);
  step2[23] = _mm_add_epi32(step1[20], step1[23]);

  step2[24] = _mm_add_epi32(step1[27], step1[24]);
  step2[25] = _mm_add_epi32(step1[26], step1[25]);
  step2[26] = _mm_sub_epi32(step1[26], step1[25]);  // step2[26] = -step2[26]
  step2[27] = _mm_sub_epi32(step1[27], step1[24]);  // step2[27] = -step2[27]
  step2[28] = _mm_sub_epi32(step1[31], step1[28]);
  step2[29] = _mm_sub_epi32(step1[30], step1[29]);
  step2[30] = _mm_add_epi32(step1[29], step1[30]);
  step2[31] = _mm_add_epi32(step1[28], step1[31]);

  // stage 5
  step1[16] = step2[16];
  step1[17] = step2[17];
  highbd_butterfly_sse2(step2[29], step2[18], cospi_24_64, cospi_8_64,
                        &step1[18], &step1[29]);
  highbd_butterfly_sse2(step2[28], step2[19], cospi_24_64, cospi_8_64,
                        &step1[19], &step1[28]);
  highbd_butterfly_sse2(step2[20], step2[27], cospi_8_64, cospi_24_64,
                        &step1[27], &step1[20]);
  highbd_butterfly_sse2(step2[21], step2[26], cospi_8_64, cospi_24_64,
                        &step1[26], &step1[21]);
  step1[22] = step2[22];
  step1[23] = step2[23];
  step1[24] = step2[24];
  step1[25] = step2[25];
  step1[30] = step2[30];
  step1[31] = step2[31];

  // stage 6
  step2[16] = _mm_add_epi32(step1[16], step1[23]);
  step2[17] = _mm_add_epi32(step1[17], step1[22]);
  step2[18] = _mm_add_epi32(step1[18], step1[21]);
  step2[19] = _mm_add_epi32(step1[19], step1[20]);
  step2[20] = _mm_sub_epi32(step1[19], step1[20]);
  step2[21] = _mm_sub_epi32(step1[18], step1[21]);
  step2[22] = _mm_sub_epi32(step1[17], step1[22]);
  step2[23] = _mm_sub_epi32(step1[16], step1[23]);

  step2[24] = _mm_sub_epi32(step1[31], step1[24]);
  step2[25] = _mm_sub_epi32(step1[30], step1[25]);
  step2[26] = _mm_sub_epi32(step1[29], step1[26]);
  step2[27] = _mm_sub_epi32(step1[28], step1[27]);
  step2[28] = _mm_add_epi32(step1[27], step1[28]);
  step2[29] = _mm_add_epi32(step1[26], step1[29]);
  step2[30] = _mm_add_epi32(step1[25], step1[30]);
  step2[31] = _mm_add_epi32(step1[24], step1[31]);

  // stage 7
  out[16] = step2[16];
  out[17] = step2[17];
  out[18] = step2[18];
  out[19] = step2[19];
  highbd_butterfly_sse2(step2[27], step2[20], cospi_16_64, cospi_16_64,
                        &out[20], &out[27]);
  highbd_butterfly_sse2(step2[26], step2[21], cospi_16_64, cospi_16_64,
                        &out[21], &out[26]);
  highbd_butterfly_sse2(step2[25], step2[22], cospi_16_64, cospi_16_64,
                        &out[22], &out[25]);
  highbd_butterfly_sse2(step2[24], step2[23], cospi_16_64, cospi_16_64,
                        &out[23], &out[24]);
  out[28] = step2[28];
  out[29] = step2[29];
  out[30] = step2[30];
  out[31] = step2[31];
}

// Group the coefficient calculation into smaller functions to prevent stack
// spillover in 32x32 idct optimizations:
// quarter_1: 0-7
// quarter_2: 8-15
// quarter_3_4: 16-23, 24-31

// For each 4x32 block __m128i in[32],
// Input with index, 0, 4, 8, 12, 16, 20, 24, 28
// output pixels: 0-7 in __m128i out[32]
static INLINE void highbd_idct32_1024_4x32_quarter_1(
    const __m128i *const in /*in[32]*/, __m128i *const out /*out[8]*/) {
  __m128i step1[8], step2[8];

  // stage 3
  highbd_butterfly_sse2(in[4], in[28], cospi_28_64, cospi_4_64, &step1[4],
                        &step1[7]);
  highbd_butterfly_sse2(in[20], in[12], cospi_12_64, cospi_20_64, &step1[5],
                        &step1[6]);

  // stage 4
  highbd_butterfly_sse2(in[0], in[16], cospi_16_64, cospi_16_64, &step2[1],
                        &step2[0]);
  highbd_butterfly_sse2(in[8], in[24], cospi_24_64, cospi_8_64, &step2[2],
                        &step2[3]);
  step2[4] = _mm_add_epi32(step1[4], step1[5]);
  step2[5] = _mm_sub_epi32(step1[4], step1[5]);
  step2[6] = _mm_sub_epi32(step1[7], step1[6]);
  step2[7] = _mm_add_epi32(step1[7], step1[6]);

  // stage 5
  step1[0] = _mm_add_epi32(step2[0], step2[3]);
  step1[1] = _mm_add_epi32(step2[1], step2[2]);
  step1[2] = _mm_sub_epi32(step2[1], step2[2]);
  step1[3] = _mm_sub_epi32(step2[0], step2[3]);
  step1[4] = step2[4];
  highbd_butterfly_sse2(step2[6], step2[5], cospi_16_64, cospi_16_64, &step1[5],
                        &step1[6]);
  step1[7] = step2[7];

  // stage 6
  out[0] = _mm_add_epi32(step1[0], step1[7]);
  out[1] = _mm_add_epi32(step1[1], step1[6]);
  out[2] = _mm_add_epi32(step1[2], step1[5]);
  out[3] = _mm_add_epi32(step1[3], step1[4]);
  out[4] = _mm_sub_epi32(step1[3], step1[4]);
  out[5] = _mm_sub_epi32(step1[2], step1[5]);
  out[6] = _mm_sub_epi32(step1[1], step1[6]);
  out[7] = _mm_sub_epi32(step1[0], step1[7]);
}

// For each 4x32 block __m128i in[32],
// Input with index, 2, 6, 10, 14, 18, 22, 26, 30
// output pixels: 8-15 in __m128i out[32]
static INLINE void highbd_idct32_1024_4x32_quarter_2(
    const __m128i *in /*in[32]*/, __m128i *out /*out[16]*/) {
  __m128i step1[32], step2[32];

  // stage 2
  highbd_butterfly_sse2(in[2], in[30], cospi_30_64, cospi_2_64, &step2[8],
                        &step2[15]);
  highbd_butterfly_sse2(in[18], in[14], cospi_14_64, cospi_18_64, &step2[9],
                        &step2[14]);
  highbd_butterfly_sse2(in[10], in[22], cospi_22_64, cospi_10_64, &step2[10],
                        &step2[13]);
  highbd_butterfly_sse2(in[26], in[6], cospi_6_64, cospi_26_64, &step2[11],
                        &step2[12]);

  // stage 3
  step1[8] = _mm_add_epi32(step2[8], step2[9]);
  step1[9] = _mm_sub_epi32(step2[8], step2[9]);
  step1[14] = _mm_sub_epi32(step2[15], step2[14]);
  step1[15] = _mm_add_epi32(step2[15], step2[14]);
  step1[10] = _mm_sub_epi32(step2[10], step2[11]);  // step1[10] = -step1[10]
  step1[11] = _mm_add_epi32(step2[10], step2[11]);
  step1[12] = _mm_add_epi32(step2[13], step2[12]);
  step1[13] = _mm_sub_epi32(step2[13], step2[12]);  // step1[13] = -step1[13]

  highbd_idct32_4x32_quarter_2_stage_4_to_6(step1, out);
}

static INLINE void highbd_idct32_1024_4x32_quarter_1_2(
    const __m128i *const in /*in[32]*/, __m128i *const out /*out[32]*/) {
  __m128i temp[16];
  highbd_idct32_1024_4x32_quarter_1(in, temp);
  highbd_idct32_1024_4x32_quarter_2(in, temp);
  // stage 7
  highbd_add_sub_butterfly(temp, out, 16);
}

// For each 4x32 block __m128i in[32],
// Input with odd index,
// 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31
// output pixels: 16-23, 24-31 in __m128i out[32]
static INLINE void highbd_idct32_1024_4x32_quarter_3_4(
    const __m128i *const in /*in[32]*/, __m128i *const out /*out[32]*/) {
  __m128i step1[32], step2[32];

  // stage 1
  highbd_butterfly_sse2(in[1], in[31], cospi_31_64, cospi_1_64, &step1[16],
                        &step1[31]);
  highbd_butterfly_sse2(in[17], in[15], cospi_15_64, cospi_17_64, &step1[17],
                        &step1[30]);
  highbd_butterfly_sse2(in[9], in[23], cospi_23_64, cospi_9_64, &step1[18],
                        &step1[29]);
  highbd_butterfly_sse2(in[25], in[7], cospi_7_64, cospi_25_64, &step1[19],
                        &step1[28]);

  highbd_butterfly_sse2(in[5], in[27], cospi_27_64, cospi_5_64, &step1[20],
                        &step1[27]);
  highbd_butterfly_sse2(in[21], in[11], cospi_11_64, cospi_21_64, &step1[21],
                        &step1[26]);

  highbd_butterfly_sse2(in[13], in[19], cospi_19_64, cospi_13_64, &step1[22],
                        &step1[25]);
  highbd_butterfly_sse2(in[29], in[3], cospi_3_64, cospi_29_64, &step1[23],
                        &step1[24]);

  // stage 2
  step2[16] = _mm_add_epi32(step1[16], step1[17]);
  step2[17] = _mm_sub_epi32(step1[16], step1[17]);
  step2[18] = _mm_sub_epi32(step1[18], step1[19]);  // step2[18] = -step2[18]
  step2[19] = _mm_add_epi32(step1[18], step1[19]);
  step2[20] = _mm_add_epi32(step1[20], step1[21]);
  step2[21] = _mm_sub_epi32(step1[20], step1[21]);
  step2[22] = _mm_sub_epi32(step1[22], step1[23]);  // step2[22] = -step2[22]
  step2[23] = _mm_add_epi32(step1[22], step1[23]);

  step2[24] = _mm_add_epi32(step1[25], step1[24]);
  step2[25] = _mm_sub_epi32(step1[25], step1[24]);  // step2[25] = -step2[25]
  step2[26] = _mm_sub_epi32(step1[27], step1[26]);
  step2[27] = _mm_add_epi32(step1[27], step1[26]);
  step2[28] = _mm_add_epi32(step1[29], step1[28]);
  step2[29] = _mm_sub_epi32(step1[29], step1[28]);  // step2[29] = -step2[29]
  step2[30] = _mm_sub_epi32(step1[31], step1[30]);
  step2[31] = _mm_add_epi32(step1[31], step1[30]);

  // stage 3
  step1[16] = step2[16];
  step1[31] = step2[31];
  highbd_butterfly_sse2(step2[30], step2[17], cospi_28_64, cospi_4_64,
                        &step1[17], &step1[30]);
  highbd_butterfly_sse2(step2[18], step2[29], cospi_4_64, cospi_28_64,
                        &step1[29], &step1[18]);
  step1[19] = step2[19];
  step1[20] = step2[20];
  highbd_butterfly_sse2(step2[26], step2[21], cospi_12_64, cospi_20_64,
                        &step1[21], &step1[26]);
  highbd_butterfly_sse2(step2[22], step2[25], cospi_20_64, cospi_12_64,
                        &step1[25], &step1[22]);
  step1[23] = step2[23];
  step1[24] = step2[24];
  step1[27] = step2[27];
  step1[28] = step2[28];

  highbd_idct32_4x32_quarter_3_4_stage_4_to_7(step1, out);
}

static void highbd_idct32_1024_4x32(__m128i *const io /*io[32]*/) {
  __m128i temp[32];

  highbd_idct32_1024_4x32_quarter_1_2(io, temp);
  highbd_idct32_1024_4x32_quarter_3_4(io, temp);
  // final stage
  highbd_add_sub_butterfly(temp, io, 32);
}

void vpx_highbd_idct32x32_1024_add_sse2(const tran_low_t *input, uint16_t *dest,
                                        int stride, int bd) {
  int i, j;

  if (bd == 8) {
    __m128i col[4][32], io[32];

    // rows
    for (i = 0; i < 4; i++) {
      highbd_load_pack_transpose_32bit_8x8(&input[0], 32, &io[0]);
      highbd_load_pack_transpose_32bit_8x8(&input[8], 32, &io[8]);
      highbd_load_pack_transpose_32bit_8x8(&input[16], 32, &io[16]);
      highbd_load_pack_transpose_32bit_8x8(&input[24], 32, &io[24]);
      idct32_1024_8x32(io, col[i]);
      input += 32 << 3;
    }

    // columns
    for (i = 0; i < 32; i += 8) {
      // Transpose 32x8 block to 8x32 block
      transpose_16bit_8x8(col[0] + i, io);
      transpose_16bit_8x8(col[1] + i, io + 8);
      transpose_16bit_8x8(col[2] + i, io + 16);
      transpose_16bit_8x8(col[3] + i, io + 24);
      idct32_1024_8x32(io, io);
      for (j = 0; j < 32; ++j) {
        highbd_write_buffer_8(dest + j * stride, io[j], bd);
      }
      dest += 8;
    }
  } else {
    __m128i all[8][32], out[32], *in;

    for (i = 0; i < 8; i++) {
      in = all[i];
      highbd_load_transpose_32bit_8x4(&input[0], 32, &in[0]);
      highbd_load_transpose_32bit_8x4(&input[8], 32, &in[8]);
      highbd_load_transpose_32bit_8x4(&input[16], 32, &in[16]);
      highbd_load_transpose_32bit_8x4(&input[24], 32, &in[24]);
      highbd_idct32_1024_4x32(in);
      input += 4 * 32;
    }

    for (i = 0; i < 32; i += 4) {
      transpose_32bit_4x4(all[0] + i, out + 0);
      transpose_32bit_4x4(all[1] + i, out + 4);
      transpose_32bit_4x4(all[2] + i, out + 8);
      transpose_32bit_4x4(all[3] + i, out + 12);
      transpose_32bit_4x4(all[4] + i, out + 16);
      transpose_32bit_4x4(all[5] + i, out + 20);
      transpose_32bit_4x4(all[6] + i, out + 24);
      transpose_32bit_4x4(all[7] + i, out + 28);
      highbd_idct32_1024_4x32(out);

      for (j = 0; j < 32; ++j) {
        highbd_write_buffer_4(dest + j * stride, out[j], bd);
      }
      dest += 4;
    }
  }
}

// -----------------------------------------------------------------------------

// For each 4x32 block __m128i in[32],
// Input with index, 0, 4, 8, 12
// output pixels: 0-7 in __m128i out[32]
static INLINE void highbd_idct32_135_4x32_quarter_1(
    const __m128i *const in /*in[32]*/, __m128i *const out /*out[8]*/) {
  __m128i step1[8], step2[8];

  // stage 3
  highbd_partial_butterfly_sse2(in[4], cospi_28_64, cospi_4_64, &step1[4],
                                &step1[7]);
  highbd_partial_butterfly_neg_sse2(in[12], cospi_12_64, cospi_20_64, &step1[5],
                                    &step1[6]);

  // stage 4
  highbd_partial_butterfly_sse2(in[0], cospi_16_64, cospi_16_64, &step2[1],
                                &step2[0]);
  highbd_partial_butterfly_sse2(in[8], cospi_24_64, cospi_8_64, &step2[2],
                                &step2[3]);
  step2[4] = _mm_add_epi32(step1[4], step1[5]);
  step2[5] = _mm_sub_epi32(step1[4], step1[5]);
  step2[6] = _mm_sub_epi32(step1[7], step1[6]);
  step2[7] = _mm_add_epi32(step1[7], step1[6]);

  // stage 5
  step1[0] = _mm_add_epi32(step2[0], step2[3]);
  step1[1] = _mm_add_epi32(step2[1], step2[2]);
  step1[2] = _mm_sub_epi32(step2[1], step2[2]);
  step1[3] = _mm_sub_epi32(step2[0], step2[3]);
  step1[4] = step2[4];
  highbd_butterfly_sse2(step2[6], step2[5], cospi_16_64, cospi_16_64, &step1[5],
                        &step1[6]);
  step1[7] = step2[7];

  // stage 6
  out[0] = _mm_add_epi32(step1[0], step1[7]);
  out[1] = _mm_add_epi32(step1[1], step1[6]);
  out[2] = _mm_add_epi32(step1[2], step1[5]);
  out[3] = _mm_add_epi32(step1[3], step1[4]);
  out[4] = _mm_sub_epi32(step1[3], step1[4]);
  out[5] = _mm_sub_epi32(step1[2], step1[5]);
  out[6] = _mm_sub_epi32(step1[1], step1[6]);
  out[7] = _mm_sub_epi32(step1[0], step1[7]);
}

// For each 4x32 block __m128i in[32],
// Input with index, 2, 6, 10, 14
// output pixels: 8-15 in __m128i out[32]
static INLINE void highbd_idct32_135_4x32_quarter_2(
    const __m128i *in /*in[32]*/, __m128i *out /*out[16]*/) {
  __m128i step1[32], step2[32];

  // stage 2
  highbd_partial_butterfly_sse2(in[2], cospi_30_64, cospi_2_64, &step2[8],
                                &step2[15]);
  highbd_partial_butterfly_neg_sse2(in[14], cospi_14_64, cospi_18_64, &step2[9],
                                    &step2[14]);
  highbd_partial_butterfly_sse2(in[10], cospi_22_64, cospi_10_64, &step2[10],
                                &step2[13]);
  highbd_partial_butterfly_neg_sse2(in[6], cospi_6_64, cospi_26_64, &step2[11],
                                    &step2[12]);

  // stage 3
  step1[8] = _mm_add_epi32(step2[8], step2[9]);
  step1[9] = _mm_sub_epi32(step2[8], step2[9]);
  step1[14] = _mm_sub_epi32(step2[15], step2[14]);
  step1[15] = _mm_add_epi32(step2[15], step2[14]);
  step1[10] = _mm_sub_epi32(step2[10], step2[11]);  // step1[10] = -step1[10]
  step1[11] = _mm_add_epi32(step2[10], step2[11]);
  step1[12] = _mm_add_epi32(step2[13], step2[12]);
  step1[13] = _mm_sub_epi32(step2[13], step2[12]);  // step1[13] = -step1[13]

  highbd_idct32_4x32_quarter_2_stage_4_to_6(step1, out);
}

static INLINE void highbd_idct32_135_4x32_quarter_1_2(
    const __m128i *const in /*in[32]*/, __m128i *const out /*out[32]*/) {
  __m128i temp[16];
  highbd_idct32_135_4x32_quarter_1(in, temp);
  highbd_idct32_135_4x32_quarter_2(in, temp);
  // stage 7
  highbd_add_sub_butterfly(temp, out, 16);
}

// For each 4x32 block __m128i in[32],
// Input with odd index,
// 1, 3, 5, 7, 9, 11, 13, 15
// output pixels: 16-23, 24-31 in __m128i out[32]
static INLINE void highbd_idct32_135_4x32_quarter_3_4(
    const __m128i *const in /*in[32]*/, __m128i *const out /*out[32]*/) {
  __m128i step1[32], step2[32];

  // stage 1
  highbd_partial_butterfly_sse2(in[1], cospi_31_64, cospi_1_64, &step1[16],
                                &step1[31]);
  highbd_partial_butterfly_neg_sse2(in[15], cospi_15_64, cospi_17_64,
                                    &step1[17], &step1[30]);
  highbd_partial_butterfly_sse2(in[9], cospi_23_64, cospi_9_64, &step1[18],
                                &step1[29]);
  highbd_partial_butterfly_neg_sse2(in[7], cospi_7_64, cospi_25_64, &step1[19],
                                    &step1[28]);

  highbd_partial_butterfly_sse2(in[5], cospi_27_64, cospi_5_64, &step1[20],
                                &step1[27]);
  highbd_partial_butterfly_neg_sse2(in[11], cospi_11_64, cospi_21_64,
                                    &step1[21], &step1[26]);

  highbd_partial_butterfly_sse2(in[13], cospi_19_64, cospi_13_64, &step1[22],
                                &step1[25]);
  highbd_partial_butterfly_neg_sse2(in[3], cospi_3_64, cospi_29_64, &step1[23],
                                    &step1[24]);

  // stage 2
  step2[16] = _mm_add_epi32(step1[16], step1[17]);
  step2[17] = _mm_sub_epi32(step1[16], step1[17]);
  step2[18] = _mm_sub_epi32(step1[18], step1[19]);  // step2[18] = -step2[18]
  step2[19] = _mm_add_epi32(step1[18], step1[19]);
  step2[20] = _mm_add_epi32(step1[20], step1[21]);
  step2[21] = _mm_sub_epi32(step1[20], step1[21]);
  step2[22] = _mm_sub_epi32(step1[22], step1[23]);  // step2[22] = -step2[22]
  step2[23] = _mm_add_epi32(step1[22], step1[23]);

  step2[24] = _mm_add_epi32(step1[25], step1[24]);
  step2[25] = _mm_sub_epi32(step1[25], step1[24]);  // step2[25] = -step2[25]
  step2[26] = _mm_sub_epi32(step1[27], step1[26]);
  step2[27] = _mm_add_epi32(step1[27], step1[26]);
  step2[28] = _mm_add_epi32(step1[29], step1[28]);
  step2[29] = _mm_sub_epi32(step1[29], step1[28]);  // step2[29] = -step2[29]
  step2[30] = _mm_sub_epi32(step1[31], step1[30]);
  step2[31] = _mm_add_epi32(step1[31], step1[30]);

  // stage 3
  step1[16] = step2[16];
  step1[31] = step2[31];
  highbd_butterfly_sse2(step2[30], step2[17], cospi_28_64, cospi_4_64,
                        &step1[17], &step1[30]);
  highbd_butterfly_sse2(step2[18], step2[29], cospi_4_64, cospi_28_64,
                        &step1[29], &step1[18]);
  step1[19] = step2[19];
  step1[20] = step2[20];
  highbd_butterfly_sse2(step2[26], step2[21], cospi_12_64, cospi_20_64,
                        &step1[21], &step1[26]);
  highbd_butterfly_sse2(step2[22], step2[25], cospi_20_64, cospi_12_64,
                        &step1[25], &step1[22]);
  step1[23] = step2[23];
  step1[24] = step2[24];
  step1[27] = step2[27];
  step1[28] = step2[28];

  highbd_idct32_4x32_quarter_3_4_stage_4_to_7(step1, out);
}

static void highbd_idct32_135_4x32(__m128i *const io /*io[32]*/) {
  __m128i temp[32];

  highbd_idct32_135_4x32_quarter_1_2(io, temp);
  highbd_idct32_135_4x32_quarter_3_4(io, temp);
  // final stage
  highbd_add_sub_butterfly(temp, io, 32);
}

void vpx_highbd_idct32x32_135_add_sse2(const tran_low_t *input, uint16_t *dest,
                                       int stride, int bd) {
  int i, j;

  if (bd == 8) {
    __m128i col[2][32], in[32], out[32];

    for (i = 16; i < 32; i++) {
      in[i] = _mm_setzero_si128();
    }

    // rows
    for (i = 0; i < 2; i++) {
      highbd_load_pack_transpose_32bit_8x8(&input[0], 32, &in[0]);
      highbd_load_pack_transpose_32bit_8x8(&input[8], 32, &in[8]);
      idct32_1024_8x32(in, col[i]);
      input += 32 << 3;
    }

    // columns
    for (i = 0; i < 32; i += 8) {
      transpose_16bit_8x8(col[0] + i, in);
      transpose_16bit_8x8(col[1] + i, in + 8);
      idct32_1024_8x32(in, out);
      for (j = 0; j < 32; ++j) {
        highbd_write_buffer_8(dest + j * stride, out[j], bd);
      }
      dest += 8;
    }
  } else {
    __m128i all[8][32], out[32], *in;

    for (i = 0; i < 4; i++) {
      in = all[i];
      highbd_load_transpose_32bit_8x4(&input[0], 32, &in[0]);
      highbd_load_transpose_32bit_8x4(&input[8], 32, &in[8]);
      highbd_idct32_135_4x32(in);
      input += 4 * 32;
    }

    for (i = 0; i < 32; i += 4) {
      transpose_32bit_4x4(all[0] + i, out + 0);
      transpose_32bit_4x4(all[1] + i, out + 4);
      transpose_32bit_4x4(all[2] + i, out + 8);
      transpose_32bit_4x4(all[3] + i, out + 12);
      highbd_idct32_135_4x32(out);

      for (j = 0; j < 32; ++j) {
        highbd_write_buffer_4(dest + j * stride, out[j], bd);
      }
      dest += 4;
    }
  }
}

// -----------------------------------------------------------------------------

// For each 4x32 block __m128i in[32],
// Input with index, 0, 4
// output pixels: 0-7 in __m128i out[32]
static INLINE void highbd_idct32_34_4x32_quarter_1(
    const __m128i *const in /*in[32]*/, __m128i *const out /*out[8]*/) {
  __m128i step1[8], step2[8];

  // stage 3
  highbd_partial_butterfly_sse2(in[4], cospi_28_64, cospi_4_64, &step1[4],
                                &step1[7]);

  // stage 4
  highbd_partial_butterfly_sse2(in[0], cospi_16_64, cospi_16_64, &step2[1],
                                &step2[0]);
  step2[4] = step1[4];
  step2[5] = step1[4];
  step2[6] = step1[7];
  step2[7] = step1[7];

  // stage 5
  step1[0] = step2[0];
  step1[1] = step2[1];
  step1[2] = step2[1];
  step1[3] = step2[0];
  step1[4] = step2[4];
  highbd_butterfly_sse2(step2[6], step2[5], cospi_16_64, cospi_16_64, &step1[5],
                        &step1[6]);
  step1[7] = step2[7];

  // stage 6
  out[0] = _mm_add_epi32(step1[0], step1[7]);
  out[1] = _mm_add_epi32(step1[1], step1[6]);
  out[2] = _mm_add_epi32(step1[2], step1[5]);
  out[3] = _mm_add_epi32(step1[3], step1[4]);
  out[4] = _mm_sub_epi32(step1[3], step1[4]);
  out[5] = _mm_sub_epi32(step1[2], step1[5]);
  out[6] = _mm_sub_epi32(step1[1], step1[6]);
  out[7] = _mm_sub_epi32(step1[0], step1[7]);
}

// For each 4x32 block __m128i in[32],
// Input with index, 2, 6
// output pixels: 8-15 in __m128i out[32]
static INLINE void highbd_idct32_34_4x32_quarter_2(const __m128i *in /*in[32]*/,
                                                   __m128i *out /*out[16]*/) {
  __m128i step1[32], step2[32];

  // stage 2
  highbd_partial_butterfly_sse2(in[2], cospi_30_64, cospi_2_64, &step2[8],
                                &step2[15]);
  highbd_partial_butterfly_neg_sse2(in[6], cospi_6_64, cospi_26_64, &step2[11],
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

  step1[10] =
      _mm_sub_epi32(_mm_setzero_si128(), step1[10]);  // step1[10] = -step1[10]
  step1[13] =
      _mm_sub_epi32(_mm_setzero_si128(), step1[13]);  // step1[13] = -step1[13]
  highbd_idct32_4x32_quarter_2_stage_4_to_6(step1, out);
}

static INLINE void highbd_idct32_34_4x32_quarter_1_2(
    const __m128i *const in /*in[32]*/, __m128i *const out /*out[32]*/) {
  __m128i temp[16];
  highbd_idct32_34_4x32_quarter_1(in, temp);
  highbd_idct32_34_4x32_quarter_2(in, temp);
  // stage 7
  highbd_add_sub_butterfly(temp, out, 16);
}

// For each 4x32 block __m128i in[32],
// Input with odd index,
// 1, 3, 5, 7
// output pixels: 16-23, 24-31 in __m128i out[32]
static INLINE void highbd_idct32_34_4x32_quarter_3_4(
    const __m128i *const in /*in[32]*/, __m128i *const out /*out[32]*/) {
  __m128i step1[32], step2[32];

  // stage 1
  highbd_partial_butterfly_sse2(in[1], cospi_31_64, cospi_1_64, &step1[16],
                                &step1[31]);
  highbd_partial_butterfly_neg_sse2(in[7], cospi_7_64, cospi_25_64, &step1[19],
                                    &step1[28]);

  highbd_partial_butterfly_sse2(in[5], cospi_27_64, cospi_5_64, &step1[20],
                                &step1[27]);
  highbd_partial_butterfly_neg_sse2(in[3], cospi_3_64, cospi_29_64, &step1[23],
                                    &step1[24]);

  // stage 2
  step2[16] = step1[16];
  step2[17] = step1[16];
  step2[18] = step1[19];
  step2[19] = step1[19];
  step2[20] = step1[20];
  step2[21] = step1[20];
  step2[22] = step1[23];
  step2[23] = step1[23];

  step2[24] = step1[24];
  step2[25] = step1[24];
  step2[26] = step1[27];
  step2[27] = step1[27];
  step2[28] = step1[28];
  step2[29] = step1[28];
  step2[30] = step1[31];
  step2[31] = step1[31];

  // stage 3
  step2[18] =
      _mm_sub_epi32(_mm_setzero_si128(), step2[18]);  // step2[18] = -step2[18]
  step2[22] =
      _mm_sub_epi32(_mm_setzero_si128(), step2[22]);  // step2[22] = -step2[22]
  step2[25] =
      _mm_sub_epi32(_mm_setzero_si128(), step2[25]);  // step2[25] = -step2[25]
  step2[29] =
      _mm_sub_epi32(_mm_setzero_si128(), step2[29]);  // step2[29] = -step2[29]
  step1[16] = step2[16];
  step1[31] = step2[31];
  highbd_butterfly_sse2(step2[30], step2[17], cospi_28_64, cospi_4_64,
                        &step1[17], &step1[30]);
  highbd_butterfly_sse2(step2[18], step2[29], cospi_4_64, cospi_28_64,
                        &step1[29], &step1[18]);
  step1[19] = step2[19];
  step1[20] = step2[20];
  highbd_butterfly_sse2(step2[26], step2[21], cospi_12_64, cospi_20_64,
                        &step1[21], &step1[26]);
  highbd_butterfly_sse2(step2[22], step2[25], cospi_20_64, cospi_12_64,
                        &step1[25], &step1[22]);
  step1[23] = step2[23];
  step1[24] = step2[24];
  step1[27] = step2[27];
  step1[28] = step2[28];

  highbd_idct32_4x32_quarter_3_4_stage_4_to_7(step1, out);
}

static void highbd_idct32_34_4x32(__m128i *const io /*io[32]*/) {
  __m128i temp[32];

  highbd_idct32_34_4x32_quarter_1_2(io, temp);
  highbd_idct32_34_4x32_quarter_3_4(io, temp);
  // final stage
  highbd_add_sub_butterfly(temp, io, 32);
}

void vpx_highbd_idct32x32_34_add_sse2(const tran_low_t *input, uint16_t *dest,
                                      int stride, int bd) {
  int i, j;

  if (bd == 8) {
    __m128i col[32], in[32], out[32];

    // rows
    highbd_load_pack_transpose_32bit_8x8(&input[0], 32, &in[0]);
    idct32_34_8x32_sse2(in, col);

    // columns
    for (i = 0; i < 32; i += 8) {
      transpose_16bit_8x8(col + i, in);
      idct32_34_8x32_sse2(in, out);
      for (j = 0; j < 32; ++j) {
        highbd_write_buffer_8(dest + j * stride, out[j], bd);
      }
      dest += 8;
    }
  } else {
    __m128i all[8][32], out[32], *in;

    for (i = 0; i < 4; i++) {
      in = all[i];
      highbd_load_transpose_32bit_8x4(&input[0], 32, &in[0]);
      highbd_load_transpose_32bit_8x4(&input[8], 32, &in[8]);
      highbd_idct32_34_4x32(in);
      input += 4 * 32;
    }

    for (i = 0; i < 32; i += 4) {
      transpose_32bit_4x4(all[0] + i, out + 0);
      transpose_32bit_4x4(all[1] + i, out + 4);
      transpose_32bit_4x4(all[2] + i, out + 8);
      transpose_32bit_4x4(all[3] + i, out + 12);
      highbd_idct32_34_4x32(out);

      for (j = 0; j < 32; ++j) {
        highbd_write_buffer_4(dest + j * stride, out[j], bd);
      }
      dest += 4;
    }
  }
}

void vpx_highbd_idct32x32_1_add_sse2(const tran_low_t *input, uint16_t *dest,
                                     int stride, int bd) {
  highbd_idct_1_add_kernel(input, dest, stride, bd, 32);
}
