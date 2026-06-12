/*
 *  Copyright (c) 2016 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <arm_neon.h>

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/arm/idct_neon.h"
#include "vpx_dsp/arm/mem_neon.h"
#include "vpx_dsp/arm/transpose_neon.h"
#include "vpx_dsp/txfm_common.h"

// Only for the first pass of the  _34_ variant. Since it only uses values from
// the top left 8x8 it can safely assume all the remaining values are 0 and skip
// an awful lot of calculations. In fact, only the first 6 columns make the cut.
// None of the elements in the 7th or 8th column are used so it skips any calls
// to input[67] too.
// In C this does a single row of 32 for each call. Here it transposes the top
// left 8x8 to allow using SIMD.

// vp9/common/vp9_scan.c:vp9_default_iscan_32x32 arranges the first 34 non-zero
// coefficients as follows:
//    0  1  2  3  4  5  6  7
// 0  0  2  5 10 17 25
// 1  1  4  8 15 22 30
// 2  3  7 12 18 28
// 3  6 11 16 23 31
// 4  9 14 19 29
// 5 13 20 26
// 6 21 27 33
// 7 24 32
void vpx_idct32_6_neon(const tran_low_t *input, int16_t *output) {
  int16x8_t in[8], s1[32], s2[32], s3[32];

  in[0] = load_tran_low_to_s16q(input);
  input += 32;
  in[1] = load_tran_low_to_s16q(input);
  input += 32;
  in[2] = load_tran_low_to_s16q(input);
  input += 32;
  in[3] = load_tran_low_to_s16q(input);
  input += 32;
  in[4] = load_tran_low_to_s16q(input);
  input += 32;
  in[5] = load_tran_low_to_s16q(input);
  input += 32;
  in[6] = load_tran_low_to_s16q(input);
  input += 32;
  in[7] = load_tran_low_to_s16q(input);
  transpose_s16_8x8(&in[0], &in[1], &in[2], &in[3], &in[4], &in[5], &in[6],
                    &in[7]);

  // stage 1
  // input[1] * cospi_31_64 - input[31] * cospi_1_64 (but input[31] == 0)
  s1[16] = multiply_shift_and_narrow_s16(in[1], cospi_31_64);
  // input[1] * cospi_1_64 + input[31] * cospi_31_64 (but input[31] == 0)
  s1[31] = multiply_shift_and_narrow_s16(in[1], cospi_1_64);

  s1[20] = multiply_shift_and_narrow_s16(in[5], cospi_27_64);
  s1[27] = multiply_shift_and_narrow_s16(in[5], cospi_5_64);

  s1[23] = multiply_shift_and_narrow_s16(in[3], -cospi_29_64);
  s1[24] = multiply_shift_and_narrow_s16(in[3], cospi_3_64);

  // stage 2
  s2[8] = multiply_shift_and_narrow_s16(in[2], cospi_30_64);
  s2[15] = multiply_shift_and_narrow_s16(in[2], cospi_2_64);

  // stage 3
  s1[4] = multiply_shift_and_narrow_s16(in[4], cospi_28_64);
  s1[7] = multiply_shift_and_narrow_s16(in[4], cospi_4_64);

  s1[17] = multiply_accumulate_shift_and_narrow_s16(s1[16], -cospi_4_64, s1[31],
                                                    cospi_28_64);
  s1[30] = multiply_accumulate_shift_and_narrow_s16(s1[16], cospi_28_64, s1[31],
                                                    cospi_4_64);

  s1[21] = multiply_accumulate_shift_and_narrow_s16(s1[20], -cospi_20_64,
                                                    s1[27], cospi_12_64);
  s1[26] = multiply_accumulate_shift_and_narrow_s16(s1[20], cospi_12_64, s1[27],
                                                    cospi_20_64);

  s1[22] = multiply_accumulate_shift_and_narrow_s16(s1[23], -cospi_12_64,
                                                    s1[24], -cospi_20_64);
  s1[25] = multiply_accumulate_shift_and_narrow_s16(s1[23], -cospi_20_64,
                                                    s1[24], cospi_12_64);

  // stage 4
  s1[0] = multiply_shift_and_narrow_s16(in[0], cospi_16_64);

  s2[9] = multiply_accumulate_shift_and_narrow_s16(s2[8], -cospi_8_64, s2[15],
                                                   cospi_24_64);
  s2[14] = multiply_accumulate_shift_and_narrow_s16(s2[8], cospi_24_64, s2[15],
                                                    cospi_8_64);

  s2[20] = vsubq_s16(s1[23], s1[20]);
  s2[21] = vsubq_s16(s1[22], s1[21]);
  s2[22] = vaddq_s16(s1[21], s1[22]);
  s2[23] = vaddq_s16(s1[20], s1[23]);
  s2[24] = vaddq_s16(s1[24], s1[27]);
  s2[25] = vaddq_s16(s1[25], s1[26]);
  s2[26] = vsubq_s16(s1[25], s1[26]);
  s2[27] = vsubq_s16(s1[24], s1[27]);

  // stage 5
  s1[5] = sub_multiply_shift_and_narrow_s16(s1[7], s1[4], cospi_16_64);
  s1[6] = add_multiply_shift_and_narrow_s16(s1[4], s1[7], cospi_16_64);

  s1[18] = multiply_accumulate_shift_and_narrow_s16(s1[17], -cospi_8_64, s1[30],
                                                    cospi_24_64);
  s1[29] = multiply_accumulate_shift_and_narrow_s16(s1[17], cospi_24_64, s1[30],
                                                    cospi_8_64);

  s1[19] = multiply_accumulate_shift_and_narrow_s16(s1[16], -cospi_8_64, s1[31],
                                                    cospi_24_64);
  s1[28] = multiply_accumulate_shift_and_narrow_s16(s1[16], cospi_24_64, s1[31],
                                                    cospi_8_64);

  s1[20] = multiply_accumulate_shift_and_narrow_s16(s2[20], -cospi_24_64,
                                                    s2[27], -cospi_8_64);
  s1[27] = multiply_accumulate_shift_and_narrow_s16(s2[20], -cospi_8_64, s2[27],
                                                    cospi_24_64);

  s1[21] = multiply_accumulate_shift_and_narrow_s16(s2[21], -cospi_24_64,
                                                    s2[26], -cospi_8_64);
  s1[26] = multiply_accumulate_shift_and_narrow_s16(s2[21], -cospi_8_64, s2[26],
                                                    cospi_24_64);

  // stage 6
  s2[0] = vaddq_s16(s1[0], s1[7]);
  s2[1] = vaddq_s16(s1[0], s1[6]);
  s2[2] = vaddq_s16(s1[0], s1[5]);
  s2[3] = vaddq_s16(s1[0], s1[4]);
  s2[4] = vsubq_s16(s1[0], s1[4]);
  s2[5] = vsubq_s16(s1[0], s1[5]);
  s2[6] = vsubq_s16(s1[0], s1[6]);
  s2[7] = vsubq_s16(s1[0], s1[7]);

  s2[10] = sub_multiply_shift_and_narrow_s16(s2[14], s2[9], cospi_16_64);
  s2[13] = add_multiply_shift_and_narrow_s16(s2[9], s2[14], cospi_16_64);

  s2[11] = sub_multiply_shift_and_narrow_s16(s2[15], s2[8], cospi_16_64);
  s2[12] = add_multiply_shift_and_narrow_s16(s2[8], s2[15], cospi_16_64);

  s2[16] = vaddq_s16(s1[16], s2[23]);
  s2[17] = vaddq_s16(s1[17], s2[22]);
  s2[18] = vaddq_s16(s1[18], s1[21]);
  s2[19] = vaddq_s16(s1[19], s1[20]);
  s2[20] = vsubq_s16(s1[19], s1[20]);
  s2[21] = vsubq_s16(s1[18], s1[21]);
  s2[22] = vsubq_s16(s1[17], s2[22]);
  s2[23] = vsubq_s16(s1[16], s2[23]);

  s3[24] = vsubq_s16(s1[31], s2[24]);
  s3[25] = vsubq_s16(s1[30], s2[25]);
  s3[26] = vsubq_s16(s1[29], s1[26]);
  s3[27] = vsubq_s16(s1[28], s1[27]);
  s2[28] = vaddq_s16(s1[27], s1[28]);
  s2[29] = vaddq_s16(s1[26], s1[29]);
  s2[30] = vaddq_s16(s2[25], s1[30]);
  s2[31] = vaddq_s16(s2[24], s1[31]);

  // stage 7
  s1[0] = vaddq_s16(s2[0], s2[15]);
  s1[1] = vaddq_s16(s2[1], s2[14]);
  s1[2] = vaddq_s16(s2[2], s2[13]);
  s1[3] = vaddq_s16(s2[3], s2[12]);
  s1[4] = vaddq_s16(s2[4], s2[11]);
  s1[5] = vaddq_s16(s2[5], s2[10]);
  s1[6] = vaddq_s16(s2[6], s2[9]);
  s1[7] = vaddq_s16(s2[7], s2[8]);
  s1[8] = vsubq_s16(s2[7], s2[8]);
  s1[9] = vsubq_s16(s2[6], s2[9]);
  s1[10] = vsubq_s16(s2[5], s2[10]);
  s1[11] = vsubq_s16(s2[4], s2[11]);
  s1[12] = vsubq_s16(s2[3], s2[12]);
  s1[13] = vsubq_s16(s2[2], s2[13]);
  s1[14] = vsubq_s16(s2[1], s2[14]);
  s1[15] = vsubq_s16(s2[0], s2[15]);

  s1[20] = sub_multiply_shift_and_narrow_s16(s3[27], s2[20], cospi_16_64);
  s1[27] = add_multiply_shift_and_narrow_s16(s2[20], s3[27], cospi_16_64);

  s1[21] = sub_multiply_shift_and_narrow_s16(s3[26], s2[21], cospi_16_64);
  s1[26] = add_multiply_shift_and_narrow_s16(s2[21], s3[26], cospi_16_64);

  s1[22] = sub_multiply_shift_and_narrow_s16(s3[25], s2[22], cospi_16_64);
  s1[25] = add_multiply_shift_and_narrow_s16(s2[22], s3[25], cospi_16_64);

  s1[23] = sub_multiply_shift_and_narrow_s16(s3[24], s2[23], cospi_16_64);
  s1[24] = add_multiply_shift_and_narrow_s16(s2[23], s3[24], cospi_16_64);

  // final stage
  vst1q_s16(output, vaddq_s16(s1[0], s2[31]));
  output += 8;
  vst1q_s16(output, vaddq_s16(s1[1], s2[30]));
  output += 8;
  vst1q_s16(output, vaddq_s16(s1[2], s2[29]));
  output += 8;
  vst1q_s16(output, vaddq_s16(s1[3], s2[28]));
  output += 8;
  vst1q_s16(output, vaddq_s16(s1[4], s1[27]));
  output += 8;
  vst1q_s16(output, vaddq_s16(s1[5], s1[26]));
  output += 8;
  vst1q_s16(output, vaddq_s16(s1[6], s1[25]));
  output += 8;
  vst1q_s16(output, vaddq_s16(s1[7], s1[24]));
  output += 8;

  vst1q_s16(output, vaddq_s16(s1[8], s1[23]));
  output += 8;
  vst1q_s16(output, vaddq_s16(s1[9], s1[22]));
  output += 8;
  vst1q_s16(output, vaddq_s16(s1[10], s1[21]));
  output += 8;
  vst1q_s16(output, vaddq_s16(s1[11], s1[20]));
  output += 8;
  vst1q_s16(output, vaddq_s16(s1[12], s2[19]));
  output += 8;
  vst1q_s16(output, vaddq_s16(s1[13], s2[18]));
  output += 8;
  vst1q_s16(output, vaddq_s16(s1[14], s2[17]));
  output += 8;
  vst1q_s16(output, vaddq_s16(s1[15], s2[16]));
  output += 8;

  vst1q_s16(output, vsubq_s16(s1[15], s2[16]));
  output += 8;
  vst1q_s16(output, vsubq_s16(s1[14], s2[17]));
  output += 8;
  vst1q_s16(output, vsubq_s16(s1[13], s2[18]));
  output += 8;
  vst1q_s16(output, vsubq_s16(s1[12], s2[19]));
  output += 8;
  vst1q_s16(output, vsubq_s16(s1[11], s1[20]));
  output += 8;
  vst1q_s16(output, vsubq_s16(s1[10], s1[21]));
  output += 8;
  vst1q_s16(output, vsubq_s16(s1[9], s1[22]));
  output += 8;
  vst1q_s16(output, vsubq_s16(s1[8], s1[23]));
  output += 8;

  vst1q_s16(output, vsubq_s16(s1[7], s1[24]));
  output += 8;
  vst1q_s16(output, vsubq_s16(s1[6], s1[25]));
  output += 8;
  vst1q_s16(output, vsubq_s16(s1[5], s1[26]));
  output += 8;
  vst1q_s16(output, vsubq_s16(s1[4], s1[27]));
  output += 8;
  vst1q_s16(output, vsubq_s16(s1[3], s2[28]));
  output += 8;
  vst1q_s16(output, vsubq_s16(s1[2], s2[29]));
  output += 8;
  vst1q_s16(output, vsubq_s16(s1[1], s2[30]));
  output += 8;
  vst1q_s16(output, vsubq_s16(s1[0], s2[31]));
}

void vpx_idct32_8_neon(const int16_t *input, void *const output, int stride,
                       const int highbd_flag) {
  int16x8_t in[8], s1[32], s2[32], s3[32], out[32];

  load_and_transpose_s16_8x8(input, 8, &in[0], &in[1], &in[2], &in[3], &in[4],
                             &in[5], &in[6], &in[7]);

  // stage 1
  s1[16] = multiply_shift_and_narrow_s16(in[1], cospi_31_64);
  s1[31] = multiply_shift_and_narrow_s16(in[1], cospi_1_64);

  // Different for _8_
  s1[19] = multiply_shift_and_narrow_s16(in[7], -cospi_25_64);
  s1[28] = multiply_shift_and_narrow_s16(in[7], cospi_7_64);

  s1[20] = multiply_shift_and_narrow_s16(in[5], cospi_27_64);
  s1[27] = multiply_shift_and_narrow_s16(in[5], cospi_5_64);

  s1[23] = multiply_shift_and_narrow_s16(in[3], -cospi_29_64);
  s1[24] = multiply_shift_and_narrow_s16(in[3], cospi_3_64);

  // stage 2
  s2[8] = multiply_shift_and_narrow_s16(in[2], cospi_30_64);
  s2[15] = multiply_shift_and_narrow_s16(in[2], cospi_2_64);

  s2[11] = multiply_shift_and_narrow_s16(in[6], -cospi_26_64);
  s2[12] = multiply_shift_and_narrow_s16(in[6], cospi_6_64);

  // stage 3
  s1[4] = multiply_shift_and_narrow_s16(in[4], cospi_28_64);
  s1[7] = multiply_shift_and_narrow_s16(in[4], cospi_4_64);

  s1[17] = multiply_accumulate_shift_and_narrow_s16(s1[16], -cospi_4_64, s1[31],
                                                    cospi_28_64);
  s1[30] = multiply_accumulate_shift_and_narrow_s16(s1[16], cospi_28_64, s1[31],
                                                    cospi_4_64);

  // Different for _8_
  s1[18] = multiply_accumulate_shift_and_narrow_s16(s1[19], -cospi_28_64,
                                                    s1[28], -cospi_4_64);
  s1[29] = multiply_accumulate_shift_and_narrow_s16(s1[19], -cospi_4_64, s1[28],
                                                    cospi_28_64);

  s1[21] = multiply_accumulate_shift_and_narrow_s16(s1[20], -cospi_20_64,
                                                    s1[27], cospi_12_64);
  s1[26] = multiply_accumulate_shift_and_narrow_s16(s1[20], cospi_12_64, s1[27],
                                                    cospi_20_64);

  s1[22] = multiply_accumulate_shift_and_narrow_s16(s1[23], -cospi_12_64,
                                                    s1[24], -cospi_20_64);
  s1[25] = multiply_accumulate_shift_and_narrow_s16(s1[23], -cospi_20_64,
                                                    s1[24], cospi_12_64);

  // stage 4
  s1[0] = multiply_shift_and_narrow_s16(in[0], cospi_16_64);

  s2[9] = multiply_accumulate_shift_and_narrow_s16(s2[8], -cospi_8_64, s2[15],
                                                   cospi_24_64);
  s2[14] = multiply_accumulate_shift_and_narrow_s16(s2[8], cospi_24_64, s2[15],
                                                    cospi_8_64);

  s2[10] = multiply_accumulate_shift_and_narrow_s16(s2[11], -cospi_24_64,
                                                    s2[12], -cospi_8_64);
  s2[13] = multiply_accumulate_shift_and_narrow_s16(s2[11], -cospi_8_64, s2[12],
                                                    cospi_24_64);

  s2[16] = vaddq_s16(s1[16], s1[19]);

  s2[17] = vaddq_s16(s1[17], s1[18]);
  s2[18] = vsubq_s16(s1[17], s1[18]);

  s2[19] = vsubq_s16(s1[16], s1[19]);

  s2[20] = vsubq_s16(s1[23], s1[20]);
  s2[21] = vsubq_s16(s1[22], s1[21]);

  s2[22] = vaddq_s16(s1[21], s1[22]);
  s2[23] = vaddq_s16(s1[20], s1[23]);

  s2[24] = vaddq_s16(s1[24], s1[27]);
  s2[25] = vaddq_s16(s1[25], s1[26]);
  s2[26] = vsubq_s16(s1[25], s1[26]);
  s2[27] = vsubq_s16(s1[24], s1[27]);

  s2[28] = vsubq_s16(s1[31], s1[28]);
  s2[29] = vsubq_s16(s1[30], s1[29]);
  s2[30] = vaddq_s16(s1[29], s1[30]);
  s2[31] = vaddq_s16(s1[28], s1[31]);

  // stage 5
  s1[5] = sub_multiply_shift_and_narrow_s16(s1[7], s1[4], cospi_16_64);
  s1[6] = add_multiply_shift_and_narrow_s16(s1[4], s1[7], cospi_16_64);

  s1[8] = vaddq_s16(s2[8], s2[11]);
  s1[9] = vaddq_s16(s2[9], s2[10]);
  s1[10] = vsubq_s16(s2[9], s2[10]);
  s1[11] = vsubq_s16(s2[8], s2[11]);
  s1[12] = vsubq_s16(s2[15], s2[12]);
  s1[13] = vsubq_s16(s2[14], s2[13]);
  s1[14] = vaddq_s16(s2[13], s2[14]);
  s1[15] = vaddq_s16(s2[12], s2[15]);

  s1[18] = multiply_accumulate_shift_and_narrow_s16(s2[18], -cospi_8_64, s2[29],
                                                    cospi_24_64);
  s1[29] = multiply_accumulate_shift_and_narrow_s16(s2[18], cospi_24_64, s2[29],
                                                    cospi_8_64);

  s1[19] = multiply_accumulate_shift_and_narrow_s16(s2[19], -cospi_8_64, s2[28],
                                                    cospi_24_64);
  s1[28] = multiply_accumulate_shift_and_narrow_s16(s2[19], cospi_24_64, s2[28],
                                                    cospi_8_64);

  s1[20] = multiply_accumulate_shift_and_narrow_s16(s2[20], -cospi_24_64,
                                                    s2[27], -cospi_8_64);
  s1[27] = multiply_accumulate_shift_and_narrow_s16(s2[20], -cospi_8_64, s2[27],
                                                    cospi_24_64);

  s1[21] = multiply_accumulate_shift_and_narrow_s16(s2[21], -cospi_24_64,
                                                    s2[26], -cospi_8_64);
  s1[26] = multiply_accumulate_shift_and_narrow_s16(s2[21], -cospi_8_64, s2[26],
                                                    cospi_24_64);

  // stage 6
  s2[0] = vaddq_s16(s1[0], s1[7]);
  s2[1] = vaddq_s16(s1[0], s1[6]);
  s2[2] = vaddq_s16(s1[0], s1[5]);
  s2[3] = vaddq_s16(s1[0], s1[4]);
  s2[4] = vsubq_s16(s1[0], s1[4]);
  s2[5] = vsubq_s16(s1[0], s1[5]);
  s2[6] = vsubq_s16(s1[0], s1[6]);
  s2[7] = vsubq_s16(s1[0], s1[7]);

  s2[10] = sub_multiply_shift_and_narrow_s16(s1[13], s1[10], cospi_16_64);
  s2[13] = add_multiply_shift_and_narrow_s16(s1[10], s1[13], cospi_16_64);

  s2[11] = sub_multiply_shift_and_narrow_s16(s1[12], s1[11], cospi_16_64);
  s2[12] = add_multiply_shift_and_narrow_s16(s1[11], s1[12], cospi_16_64);

  s1[16] = vaddq_s16(s2[16], s2[23]);
  s1[17] = vaddq_s16(s2[17], s2[22]);
  s2[18] = vaddq_s16(s1[18], s1[21]);
  s2[19] = vaddq_s16(s1[19], s1[20]);
  s2[20] = vsubq_s16(s1[19], s1[20]);
  s2[21] = vsubq_s16(s1[18], s1[21]);
  s1[22] = vsubq_s16(s2[17], s2[22]);
  s1[23] = vsubq_s16(s2[16], s2[23]);

  s3[24] = vsubq_s16(s2[31], s2[24]);
  s3[25] = vsubq_s16(s2[30], s2[25]);
  s3[26] = vsubq_s16(s1[29], s1[26]);
  s3[27] = vsubq_s16(s1[28], s1[27]);
  s2[28] = vaddq_s16(s1[27], s1[28]);
  s2[29] = vaddq_s16(s1[26], s1[29]);
  s2[30] = vaddq_s16(s2[25], s2[30]);
  s2[31] = vaddq_s16(s2[24], s2[31]);

  // stage 7
  s1[0] = vaddq_s16(s2[0], s1[15]);
  s1[1] = vaddq_s16(s2[1], s1[14]);
  s1[2] = vaddq_s16(s2[2], s2[13]);
  s1[3] = vaddq_s16(s2[3], s2[12]);
  s1[4] = vaddq_s16(s2[4], s2[11]);
  s1[5] = vaddq_s16(s2[5], s2[10]);
  s1[6] = vaddq_s16(s2[6], s1[9]);
  s1[7] = vaddq_s16(s2[7], s1[8]);
  s1[8] = vsubq_s16(s2[7], s1[8]);
  s1[9] = vsubq_s16(s2[6], s1[9]);
  s1[10] = vsubq_s16(s2[5], s2[10]);
  s1[11] = vsubq_s16(s2[4], s2[11]);
  s1[12] = vsubq_s16(s2[3], s2[12]);
  s1[13] = vsubq_s16(s2[2], s2[13]);
  s1[14] = vsubq_s16(s2[1], s1[14]);
  s1[15] = vsubq_s16(s2[0], s1[15]);

  s1[20] = sub_multiply_shift_and_narrow_s16(s3[27], s2[20], cospi_16_64);
  s1[27] = add_multiply_shift_and_narrow_s16(s2[20], s3[27], cospi_16_64);

  s1[21] = sub_multiply_shift_and_narrow_s16(s3[26], s2[21], cospi_16_64);
  s1[26] = add_multiply_shift_and_narrow_s16(s2[21], s3[26], cospi_16_64);

  s2[22] = sub_multiply_shift_and_narrow_s16(s3[25], s1[22], cospi_16_64);
  s1[25] = add_multiply_shift_and_narrow_s16(s1[22], s3[25], cospi_16_64);

  s2[23] = sub_multiply_shift_and_narrow_s16(s3[24], s1[23], cospi_16_64);
  s1[24] = add_multiply_shift_and_narrow_s16(s1[23], s3[24], cospi_16_64);

  // final stage
  out[0] = final_add(s1[0], s2[31]);
  out[1] = final_add(s1[1], s2[30]);
  out[2] = final_add(s1[2], s2[29]);
  out[3] = final_add(s1[3], s2[28]);
  out[4] = final_add(s1[4], s1[27]);
  out[5] = final_add(s1[5], s1[26]);
  out[6] = final_add(s1[6], s1[25]);
  out[7] = final_add(s1[7], s1[24]);
  out[8] = final_add(s1[8], s2[23]);
  out[9] = final_add(s1[9], s2[22]);
  out[10] = final_add(s1[10], s1[21]);
  out[11] = final_add(s1[11], s1[20]);
  out[12] = final_add(s1[12], s2[19]);
  out[13] = final_add(s1[13], s2[18]);
  out[14] = final_add(s1[14], s1[17]);
  out[15] = final_add(s1[15], s1[16]);
  out[16] = final_sub(s1[15], s1[16]);
  out[17] = final_sub(s1[14], s1[17]);
  out[18] = final_sub(s1[13], s2[18]);
  out[19] = final_sub(s1[12], s2[19]);
  out[20] = final_sub(s1[11], s1[20]);
  out[21] = final_sub(s1[10], s1[21]);
  out[22] = final_sub(s1[9], s2[22]);
  out[23] = final_sub(s1[8], s2[23]);
  out[24] = final_sub(s1[7], s1[24]);
  out[25] = final_sub(s1[6], s1[25]);
  out[26] = final_sub(s1[5], s1[26]);
  out[27] = final_sub(s1[4], s1[27]);
  out[28] = final_sub(s1[3], s2[28]);
  out[29] = final_sub(s1[2], s2[29]);
  out[30] = final_sub(s1[1], s2[30]);
  out[31] = final_sub(s1[0], s2[31]);

  if (highbd_flag) {
    highbd_add_and_store_bd8(out, output, stride);
  } else {
    uint8_t *const outputT = (uint8_t *)output;
    add_and_store_u8_s16(out + 0, outputT, stride);
    add_and_store_u8_s16(out + 8, outputT + (8 * stride), stride);
    add_and_store_u8_s16(out + 16, outputT + (16 * stride), stride);
    add_and_store_u8_s16(out + 24, outputT + (24 * stride), stride);
  }
}

void vpx_idct32x32_34_add_neon(const tran_low_t *input, uint8_t *dest,
                               int stride) {
  int i;
  int16_t temp[32 * 8];
  int16_t *t = temp;

  vpx_idct32_6_neon(input, t);

  for (i = 0; i < 32; i += 8) {
    vpx_idct32_8_neon(t, dest, stride, 0);
    t += (8 * 8);
    dest += 8;
  }
}
