/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
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
#include "vpx_dsp/arm/highbd_idct_neon.h"
#include "vpx_dsp/arm/idct_neon.h"
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
static void vpx_highbd_idct32_6_neon(const tran_low_t *input, int32_t *output) {
  int32x4x2_t in[8], s1[32], s2[32], s3[32];

  in[0].val[0] = vld1q_s32(input);
  in[0].val[1] = vld1q_s32(input + 4);
  input += 32;
  in[1].val[0] = vld1q_s32(input);
  in[1].val[1] = vld1q_s32(input + 4);
  input += 32;
  in[2].val[0] = vld1q_s32(input);
  in[2].val[1] = vld1q_s32(input + 4);
  input += 32;
  in[3].val[0] = vld1q_s32(input);
  in[3].val[1] = vld1q_s32(input + 4);
  input += 32;
  in[4].val[0] = vld1q_s32(input);
  in[4].val[1] = vld1q_s32(input + 4);
  input += 32;
  in[5].val[0] = vld1q_s32(input);
  in[5].val[1] = vld1q_s32(input + 4);
  input += 32;
  in[6].val[0] = vld1q_s32(input);
  in[6].val[1] = vld1q_s32(input + 4);
  input += 32;
  in[7].val[0] = vld1q_s32(input);
  in[7].val[1] = vld1q_s32(input + 4);
  transpose_s32_8x8(&in[0], &in[1], &in[2], &in[3], &in[4], &in[5], &in[6],
                    &in[7]);

  // stage 1
  // input[1] * cospi_31_64 - input[31] * cospi_1_64 (but input[31] == 0)
  s1[16] = multiply_shift_and_narrow_s32_dual(in[1], cospi_31_64);
  // input[1] * cospi_1_64 + input[31] * cospi_31_64 (but input[31] == 0)
  s1[31] = multiply_shift_and_narrow_s32_dual(in[1], cospi_1_64);

  s1[20] = multiply_shift_and_narrow_s32_dual(in[5], cospi_27_64);
  s1[27] = multiply_shift_and_narrow_s32_dual(in[5], cospi_5_64);

  s1[23] = multiply_shift_and_narrow_s32_dual(in[3], -cospi_29_64);
  s1[24] = multiply_shift_and_narrow_s32_dual(in[3], cospi_3_64);

  // stage 2
  s2[8] = multiply_shift_and_narrow_s32_dual(in[2], cospi_30_64);
  s2[15] = multiply_shift_and_narrow_s32_dual(in[2], cospi_2_64);

  // stage 3
  s1[4] = multiply_shift_and_narrow_s32_dual(in[4], cospi_28_64);
  s1[7] = multiply_shift_and_narrow_s32_dual(in[4], cospi_4_64);

  s1[17] = multiply_accumulate_shift_and_narrow_s32_dual(s1[16], -cospi_4_64,
                                                         s1[31], cospi_28_64);
  s1[30] = multiply_accumulate_shift_and_narrow_s32_dual(s1[16], cospi_28_64,
                                                         s1[31], cospi_4_64);

  s1[21] = multiply_accumulate_shift_and_narrow_s32_dual(s1[20], -cospi_20_64,
                                                         s1[27], cospi_12_64);
  s1[26] = multiply_accumulate_shift_and_narrow_s32_dual(s1[20], cospi_12_64,
                                                         s1[27], cospi_20_64);

  s1[22] = multiply_accumulate_shift_and_narrow_s32_dual(s1[23], -cospi_12_64,
                                                         s1[24], -cospi_20_64);
  s1[25] = multiply_accumulate_shift_and_narrow_s32_dual(s1[23], -cospi_20_64,
                                                         s1[24], cospi_12_64);

  // stage 4
  s1[0] = multiply_shift_and_narrow_s32_dual(in[0], cospi_16_64);

  s2[9] = multiply_accumulate_shift_and_narrow_s32_dual(s2[8], -cospi_8_64,
                                                        s2[15], cospi_24_64);
  s2[14] = multiply_accumulate_shift_and_narrow_s32_dual(s2[8], cospi_24_64,
                                                         s2[15], cospi_8_64);

  s2[20] = highbd_idct_sub_dual(s1[23], s1[20]);
  s2[21] = highbd_idct_sub_dual(s1[22], s1[21]);
  s2[22] = highbd_idct_add_dual(s1[21], s1[22]);
  s2[23] = highbd_idct_add_dual(s1[20], s1[23]);
  s2[24] = highbd_idct_add_dual(s1[24], s1[27]);
  s2[25] = highbd_idct_add_dual(s1[25], s1[26]);
  s2[26] = highbd_idct_sub_dual(s1[25], s1[26]);
  s2[27] = highbd_idct_sub_dual(s1[24], s1[27]);

  // stage 5
  s1[5] = sub_multiply_shift_and_narrow_s32_dual(s1[7], s1[4], cospi_16_64);
  s1[6] = add_multiply_shift_and_narrow_s32_dual(s1[4], s1[7], cospi_16_64);

  s1[18] = multiply_accumulate_shift_and_narrow_s32_dual(s1[17], -cospi_8_64,
                                                         s1[30], cospi_24_64);
  s1[29] = multiply_accumulate_shift_and_narrow_s32_dual(s1[17], cospi_24_64,
                                                         s1[30], cospi_8_64);

  s1[19] = multiply_accumulate_shift_and_narrow_s32_dual(s1[16], -cospi_8_64,
                                                         s1[31], cospi_24_64);
  s1[28] = multiply_accumulate_shift_and_narrow_s32_dual(s1[16], cospi_24_64,
                                                         s1[31], cospi_8_64);

  s1[20] = multiply_accumulate_shift_and_narrow_s32_dual(s2[20], -cospi_24_64,
                                                         s2[27], -cospi_8_64);
  s1[27] = multiply_accumulate_shift_and_narrow_s32_dual(s2[20], -cospi_8_64,
                                                         s2[27], cospi_24_64);

  s1[21] = multiply_accumulate_shift_and_narrow_s32_dual(s2[21], -cospi_24_64,
                                                         s2[26], -cospi_8_64);
  s1[26] = multiply_accumulate_shift_and_narrow_s32_dual(s2[21], -cospi_8_64,
                                                         s2[26], cospi_24_64);

  // stage 6
  s2[0] = highbd_idct_add_dual(s1[0], s1[7]);
  s2[1] = highbd_idct_add_dual(s1[0], s1[6]);
  s2[2] = highbd_idct_add_dual(s1[0], s1[5]);
  s2[3] = highbd_idct_add_dual(s1[0], s1[4]);
  s2[4] = highbd_idct_sub_dual(s1[0], s1[4]);
  s2[5] = highbd_idct_sub_dual(s1[0], s1[5]);
  s2[6] = highbd_idct_sub_dual(s1[0], s1[6]);
  s2[7] = highbd_idct_sub_dual(s1[0], s1[7]);

  s2[10] = sub_multiply_shift_and_narrow_s32_dual(s2[14], s2[9], cospi_16_64);
  s2[13] = add_multiply_shift_and_narrow_s32_dual(s2[9], s2[14], cospi_16_64);

  s2[11] = sub_multiply_shift_and_narrow_s32_dual(s2[15], s2[8], cospi_16_64);
  s2[12] = add_multiply_shift_and_narrow_s32_dual(s2[8], s2[15], cospi_16_64);

  s2[16] = highbd_idct_add_dual(s1[16], s2[23]);
  s2[17] = highbd_idct_add_dual(s1[17], s2[22]);
  s2[18] = highbd_idct_add_dual(s1[18], s1[21]);
  s2[19] = highbd_idct_add_dual(s1[19], s1[20]);
  s2[20] = highbd_idct_sub_dual(s1[19], s1[20]);
  s2[21] = highbd_idct_sub_dual(s1[18], s1[21]);
  s2[22] = highbd_idct_sub_dual(s1[17], s2[22]);
  s2[23] = highbd_idct_sub_dual(s1[16], s2[23]);

  s3[24] = highbd_idct_sub_dual(s1[31], s2[24]);
  s3[25] = highbd_idct_sub_dual(s1[30], s2[25]);
  s3[26] = highbd_idct_sub_dual(s1[29], s1[26]);
  s3[27] = highbd_idct_sub_dual(s1[28], s1[27]);
  s2[28] = highbd_idct_add_dual(s1[27], s1[28]);
  s2[29] = highbd_idct_add_dual(s1[26], s1[29]);
  s2[30] = highbd_idct_add_dual(s2[25], s1[30]);
  s2[31] = highbd_idct_add_dual(s2[24], s1[31]);

  // stage 7
  s1[0] = highbd_idct_add_dual(s2[0], s2[15]);
  s1[1] = highbd_idct_add_dual(s2[1], s2[14]);
  s1[2] = highbd_idct_add_dual(s2[2], s2[13]);
  s1[3] = highbd_idct_add_dual(s2[3], s2[12]);
  s1[4] = highbd_idct_add_dual(s2[4], s2[11]);
  s1[5] = highbd_idct_add_dual(s2[5], s2[10]);
  s1[6] = highbd_idct_add_dual(s2[6], s2[9]);
  s1[7] = highbd_idct_add_dual(s2[7], s2[8]);
  s1[8] = highbd_idct_sub_dual(s2[7], s2[8]);
  s1[9] = highbd_idct_sub_dual(s2[6], s2[9]);
  s1[10] = highbd_idct_sub_dual(s2[5], s2[10]);
  s1[11] = highbd_idct_sub_dual(s2[4], s2[11]);
  s1[12] = highbd_idct_sub_dual(s2[3], s2[12]);
  s1[13] = highbd_idct_sub_dual(s2[2], s2[13]);
  s1[14] = highbd_idct_sub_dual(s2[1], s2[14]);
  s1[15] = highbd_idct_sub_dual(s2[0], s2[15]);

  s1[20] = sub_multiply_shift_and_narrow_s32_dual(s3[27], s2[20], cospi_16_64);
  s1[27] = add_multiply_shift_and_narrow_s32_dual(s2[20], s3[27], cospi_16_64);

  s1[21] = sub_multiply_shift_and_narrow_s32_dual(s3[26], s2[21], cospi_16_64);
  s1[26] = add_multiply_shift_and_narrow_s32_dual(s2[21], s3[26], cospi_16_64);

  s1[22] = sub_multiply_shift_and_narrow_s32_dual(s3[25], s2[22], cospi_16_64);
  s1[25] = add_multiply_shift_and_narrow_s32_dual(s2[22], s3[25], cospi_16_64);

  s1[23] = sub_multiply_shift_and_narrow_s32_dual(s3[24], s2[23], cospi_16_64);
  s1[24] = add_multiply_shift_and_narrow_s32_dual(s2[23], s3[24], cospi_16_64);

  // final stage
  s3[0] = highbd_idct_add_dual(s1[0], s2[31]);
  s3[1] = highbd_idct_add_dual(s1[1], s2[30]);
  s3[2] = highbd_idct_add_dual(s1[2], s2[29]);
  s3[3] = highbd_idct_add_dual(s1[3], s2[28]);
  s3[4] = highbd_idct_add_dual(s1[4], s1[27]);
  s3[5] = highbd_idct_add_dual(s1[5], s1[26]);
  s3[6] = highbd_idct_add_dual(s1[6], s1[25]);
  s3[7] = highbd_idct_add_dual(s1[7], s1[24]);
  s3[8] = highbd_idct_add_dual(s1[8], s1[23]);
  s3[9] = highbd_idct_add_dual(s1[9], s1[22]);
  s3[10] = highbd_idct_add_dual(s1[10], s1[21]);
  s3[11] = highbd_idct_add_dual(s1[11], s1[20]);
  s3[12] = highbd_idct_add_dual(s1[12], s2[19]);
  s3[13] = highbd_idct_add_dual(s1[13], s2[18]);
  s3[14] = highbd_idct_add_dual(s1[14], s2[17]);
  s3[15] = highbd_idct_add_dual(s1[15], s2[16]);
  s3[16] = highbd_idct_sub_dual(s1[15], s2[16]);
  s3[17] = highbd_idct_sub_dual(s1[14], s2[17]);
  s3[18] = highbd_idct_sub_dual(s1[13], s2[18]);
  s3[19] = highbd_idct_sub_dual(s1[12], s2[19]);
  s3[20] = highbd_idct_sub_dual(s1[11], s1[20]);
  s3[21] = highbd_idct_sub_dual(s1[10], s1[21]);
  s3[22] = highbd_idct_sub_dual(s1[9], s1[22]);
  s3[23] = highbd_idct_sub_dual(s1[8], s1[23]);
  s3[24] = highbd_idct_sub_dual(s1[7], s1[24]);
  s3[25] = highbd_idct_sub_dual(s1[6], s1[25]);
  s3[26] = highbd_idct_sub_dual(s1[5], s1[26]);
  s3[27] = highbd_idct_sub_dual(s1[4], s1[27]);
  s3[28] = highbd_idct_sub_dual(s1[3], s2[28]);
  s3[29] = highbd_idct_sub_dual(s1[2], s2[29]);
  s3[30] = highbd_idct_sub_dual(s1[1], s2[30]);
  s3[31] = highbd_idct_sub_dual(s1[0], s2[31]);

  vst1q_s32(output, s3[0].val[0]);
  output += 4;
  vst1q_s32(output, s3[0].val[1]);
  output += 4;
  vst1q_s32(output, s3[1].val[0]);
  output += 4;
  vst1q_s32(output, s3[1].val[1]);
  output += 4;
  vst1q_s32(output, s3[2].val[0]);
  output += 4;
  vst1q_s32(output, s3[2].val[1]);
  output += 4;
  vst1q_s32(output, s3[3].val[0]);
  output += 4;
  vst1q_s32(output, s3[3].val[1]);
  output += 4;
  vst1q_s32(output, s3[4].val[0]);
  output += 4;
  vst1q_s32(output, s3[4].val[1]);
  output += 4;
  vst1q_s32(output, s3[5].val[0]);
  output += 4;
  vst1q_s32(output, s3[5].val[1]);
  output += 4;
  vst1q_s32(output, s3[6].val[0]);
  output += 4;
  vst1q_s32(output, s3[6].val[1]);
  output += 4;
  vst1q_s32(output, s3[7].val[0]);
  output += 4;
  vst1q_s32(output, s3[7].val[1]);
  output += 4;

  vst1q_s32(output, s3[8].val[0]);
  output += 4;
  vst1q_s32(output, s3[8].val[1]);
  output += 4;
  vst1q_s32(output, s3[9].val[0]);
  output += 4;
  vst1q_s32(output, s3[9].val[1]);
  output += 4;
  vst1q_s32(output, s3[10].val[0]);
  output += 4;
  vst1q_s32(output, s3[10].val[1]);
  output += 4;
  vst1q_s32(output, s3[11].val[0]);
  output += 4;
  vst1q_s32(output, s3[11].val[1]);
  output += 4;
  vst1q_s32(output, s3[12].val[0]);
  output += 4;
  vst1q_s32(output, s3[12].val[1]);
  output += 4;
  vst1q_s32(output, s3[13].val[0]);
  output += 4;
  vst1q_s32(output, s3[13].val[1]);
  output += 4;
  vst1q_s32(output, s3[14].val[0]);
  output += 4;
  vst1q_s32(output, s3[14].val[1]);
  output += 4;
  vst1q_s32(output, s3[15].val[0]);
  output += 4;
  vst1q_s32(output, s3[15].val[1]);
  output += 4;

  vst1q_s32(output, s3[16].val[0]);
  output += 4;
  vst1q_s32(output, s3[16].val[1]);
  output += 4;
  vst1q_s32(output, s3[17].val[0]);
  output += 4;
  vst1q_s32(output, s3[17].val[1]);
  output += 4;
  vst1q_s32(output, s3[18].val[0]);
  output += 4;
  vst1q_s32(output, s3[18].val[1]);
  output += 4;
  vst1q_s32(output, s3[19].val[0]);
  output += 4;
  vst1q_s32(output, s3[19].val[1]);
  output += 4;
  vst1q_s32(output, s3[20].val[0]);
  output += 4;
  vst1q_s32(output, s3[20].val[1]);
  output += 4;
  vst1q_s32(output, s3[21].val[0]);
  output += 4;
  vst1q_s32(output, s3[21].val[1]);
  output += 4;
  vst1q_s32(output, s3[22].val[0]);
  output += 4;
  vst1q_s32(output, s3[22].val[1]);
  output += 4;
  vst1q_s32(output, s3[23].val[0]);
  output += 4;
  vst1q_s32(output, s3[23].val[1]);
  output += 4;

  vst1q_s32(output, s3[24].val[0]);
  output += 4;
  vst1q_s32(output, s3[24].val[1]);
  output += 4;
  vst1q_s32(output, s3[25].val[0]);
  output += 4;
  vst1q_s32(output, s3[25].val[1]);
  output += 4;
  vst1q_s32(output, s3[26].val[0]);
  output += 4;
  vst1q_s32(output, s3[26].val[1]);
  output += 4;
  vst1q_s32(output, s3[27].val[0]);
  output += 4;
  vst1q_s32(output, s3[27].val[1]);
  output += 4;
  vst1q_s32(output, s3[28].val[0]);
  output += 4;
  vst1q_s32(output, s3[28].val[1]);
  output += 4;
  vst1q_s32(output, s3[29].val[0]);
  output += 4;
  vst1q_s32(output, s3[29].val[1]);
  output += 4;
  vst1q_s32(output, s3[30].val[0]);
  output += 4;
  vst1q_s32(output, s3[30].val[1]);
  output += 4;
  vst1q_s32(output, s3[31].val[0]);
  output += 4;
  vst1q_s32(output, s3[31].val[1]);
}

static void vpx_highbd_idct32_8_neon(const int32_t *input, uint16_t *output,
                                     int stride, const int bd) {
  int32x4x2_t in[8], s1[32], s2[32], s3[32], out[32];

  load_and_transpose_s32_8x8(input, 8, &in[0], &in[1], &in[2], &in[3], &in[4],
                             &in[5], &in[6], &in[7]);

  // stage 1
  s1[16] = multiply_shift_and_narrow_s32_dual(in[1], cospi_31_64);
  s1[31] = multiply_shift_and_narrow_s32_dual(in[1], cospi_1_64);

  // Different for _8_
  s1[19] = multiply_shift_and_narrow_s32_dual(in[7], -cospi_25_64);
  s1[28] = multiply_shift_and_narrow_s32_dual(in[7], cospi_7_64);

  s1[20] = multiply_shift_and_narrow_s32_dual(in[5], cospi_27_64);
  s1[27] = multiply_shift_and_narrow_s32_dual(in[5], cospi_5_64);

  s1[23] = multiply_shift_and_narrow_s32_dual(in[3], -cospi_29_64);
  s1[24] = multiply_shift_and_narrow_s32_dual(in[3], cospi_3_64);

  // stage 2
  s2[8] = multiply_shift_and_narrow_s32_dual(in[2], cospi_30_64);
  s2[15] = multiply_shift_and_narrow_s32_dual(in[2], cospi_2_64);

  s2[11] = multiply_shift_and_narrow_s32_dual(in[6], -cospi_26_64);
  s2[12] = multiply_shift_and_narrow_s32_dual(in[6], cospi_6_64);

  // stage 3
  s1[4] = multiply_shift_and_narrow_s32_dual(in[4], cospi_28_64);
  s1[7] = multiply_shift_and_narrow_s32_dual(in[4], cospi_4_64);

  s1[17] = multiply_accumulate_shift_and_narrow_s32_dual(s1[16], -cospi_4_64,
                                                         s1[31], cospi_28_64);
  s1[30] = multiply_accumulate_shift_and_narrow_s32_dual(s1[16], cospi_28_64,
                                                         s1[31], cospi_4_64);

  // Different for _8_
  s1[18] = multiply_accumulate_shift_and_narrow_s32_dual(s1[19], -cospi_28_64,
                                                         s1[28], -cospi_4_64);
  s1[29] = multiply_accumulate_shift_and_narrow_s32_dual(s1[19], -cospi_4_64,
                                                         s1[28], cospi_28_64);

  s1[21] = multiply_accumulate_shift_and_narrow_s32_dual(s1[20], -cospi_20_64,
                                                         s1[27], cospi_12_64);
  s1[26] = multiply_accumulate_shift_and_narrow_s32_dual(s1[20], cospi_12_64,
                                                         s1[27], cospi_20_64);

  s1[22] = multiply_accumulate_shift_and_narrow_s32_dual(s1[23], -cospi_12_64,
                                                         s1[24], -cospi_20_64);
  s1[25] = multiply_accumulate_shift_and_narrow_s32_dual(s1[23], -cospi_20_64,
                                                         s1[24], cospi_12_64);

  // stage 4
  s1[0] = multiply_shift_and_narrow_s32_dual(in[0], cospi_16_64);

  s2[9] = multiply_accumulate_shift_and_narrow_s32_dual(s2[8], -cospi_8_64,
                                                        s2[15], cospi_24_64);
  s2[14] = multiply_accumulate_shift_and_narrow_s32_dual(s2[8], cospi_24_64,
                                                         s2[15], cospi_8_64);

  s2[10] = multiply_accumulate_shift_and_narrow_s32_dual(s2[11], -cospi_24_64,
                                                         s2[12], -cospi_8_64);
  s2[13] = multiply_accumulate_shift_and_narrow_s32_dual(s2[11], -cospi_8_64,
                                                         s2[12], cospi_24_64);

  s2[16] = highbd_idct_add_dual(s1[16], s1[19]);

  s2[17] = highbd_idct_add_dual(s1[17], s1[18]);
  s2[18] = highbd_idct_sub_dual(s1[17], s1[18]);

  s2[19] = highbd_idct_sub_dual(s1[16], s1[19]);

  s2[20] = highbd_idct_sub_dual(s1[23], s1[20]);
  s2[21] = highbd_idct_sub_dual(s1[22], s1[21]);

  s2[22] = highbd_idct_add_dual(s1[21], s1[22]);
  s2[23] = highbd_idct_add_dual(s1[20], s1[23]);

  s2[24] = highbd_idct_add_dual(s1[24], s1[27]);
  s2[25] = highbd_idct_add_dual(s1[25], s1[26]);
  s2[26] = highbd_idct_sub_dual(s1[25], s1[26]);
  s2[27] = highbd_idct_sub_dual(s1[24], s1[27]);

  s2[28] = highbd_idct_sub_dual(s1[31], s1[28]);
  s2[29] = highbd_idct_sub_dual(s1[30], s1[29]);
  s2[30] = highbd_idct_add_dual(s1[29], s1[30]);
  s2[31] = highbd_idct_add_dual(s1[28], s1[31]);

  // stage 5
  s1[5] = sub_multiply_shift_and_narrow_s32_dual(s1[7], s1[4], cospi_16_64);
  s1[6] = add_multiply_shift_and_narrow_s32_dual(s1[4], s1[7], cospi_16_64);

  s1[8] = highbd_idct_add_dual(s2[8], s2[11]);
  s1[9] = highbd_idct_add_dual(s2[9], s2[10]);
  s1[10] = highbd_idct_sub_dual(s2[9], s2[10]);
  s1[11] = highbd_idct_sub_dual(s2[8], s2[11]);
  s1[12] = highbd_idct_sub_dual(s2[15], s2[12]);
  s1[13] = highbd_idct_sub_dual(s2[14], s2[13]);
  s1[14] = highbd_idct_add_dual(s2[13], s2[14]);
  s1[15] = highbd_idct_add_dual(s2[12], s2[15]);

  s1[18] = multiply_accumulate_shift_and_narrow_s32_dual(s2[18], -cospi_8_64,
                                                         s2[29], cospi_24_64);
  s1[29] = multiply_accumulate_shift_and_narrow_s32_dual(s2[18], cospi_24_64,
                                                         s2[29], cospi_8_64);

  s1[19] = multiply_accumulate_shift_and_narrow_s32_dual(s2[19], -cospi_8_64,
                                                         s2[28], cospi_24_64);
  s1[28] = multiply_accumulate_shift_and_narrow_s32_dual(s2[19], cospi_24_64,
                                                         s2[28], cospi_8_64);

  s1[20] = multiply_accumulate_shift_and_narrow_s32_dual(s2[20], -cospi_24_64,
                                                         s2[27], -cospi_8_64);
  s1[27] = multiply_accumulate_shift_and_narrow_s32_dual(s2[20], -cospi_8_64,
                                                         s2[27], cospi_24_64);

  s1[21] = multiply_accumulate_shift_and_narrow_s32_dual(s2[21], -cospi_24_64,
                                                         s2[26], -cospi_8_64);
  s1[26] = multiply_accumulate_shift_and_narrow_s32_dual(s2[21], -cospi_8_64,
                                                         s2[26], cospi_24_64);

  // stage 6
  s2[0] = highbd_idct_add_dual(s1[0], s1[7]);
  s2[1] = highbd_idct_add_dual(s1[0], s1[6]);
  s2[2] = highbd_idct_add_dual(s1[0], s1[5]);
  s2[3] = highbd_idct_add_dual(s1[0], s1[4]);
  s2[4] = highbd_idct_sub_dual(s1[0], s1[4]);
  s2[5] = highbd_idct_sub_dual(s1[0], s1[5]);
  s2[6] = highbd_idct_sub_dual(s1[0], s1[6]);
  s2[7] = highbd_idct_sub_dual(s1[0], s1[7]);

  s2[10] = sub_multiply_shift_and_narrow_s32_dual(s1[13], s1[10], cospi_16_64);
  s2[13] = add_multiply_shift_and_narrow_s32_dual(s1[10], s1[13], cospi_16_64);

  s2[11] = sub_multiply_shift_and_narrow_s32_dual(s1[12], s1[11], cospi_16_64);
  s2[12] = add_multiply_shift_and_narrow_s32_dual(s1[11], s1[12], cospi_16_64);

  s1[16] = highbd_idct_add_dual(s2[16], s2[23]);
  s1[17] = highbd_idct_add_dual(s2[17], s2[22]);
  s2[18] = highbd_idct_add_dual(s1[18], s1[21]);
  s2[19] = highbd_idct_add_dual(s1[19], s1[20]);
  s2[20] = highbd_idct_sub_dual(s1[19], s1[20]);
  s2[21] = highbd_idct_sub_dual(s1[18], s1[21]);
  s1[22] = highbd_idct_sub_dual(s2[17], s2[22]);
  s1[23] = highbd_idct_sub_dual(s2[16], s2[23]);

  s3[24] = highbd_idct_sub_dual(s2[31], s2[24]);
  s3[25] = highbd_idct_sub_dual(s2[30], s2[25]);
  s3[26] = highbd_idct_sub_dual(s1[29], s1[26]);
  s3[27] = highbd_idct_sub_dual(s1[28], s1[27]);
  s2[28] = highbd_idct_add_dual(s1[27], s1[28]);
  s2[29] = highbd_idct_add_dual(s1[26], s1[29]);
  s2[30] = highbd_idct_add_dual(s2[25], s2[30]);
  s2[31] = highbd_idct_add_dual(s2[24], s2[31]);

  // stage 7
  s1[0] = highbd_idct_add_dual(s2[0], s1[15]);
  s1[1] = highbd_idct_add_dual(s2[1], s1[14]);
  s1[2] = highbd_idct_add_dual(s2[2], s2[13]);
  s1[3] = highbd_idct_add_dual(s2[3], s2[12]);
  s1[4] = highbd_idct_add_dual(s2[4], s2[11]);
  s1[5] = highbd_idct_add_dual(s2[5], s2[10]);
  s1[6] = highbd_idct_add_dual(s2[6], s1[9]);
  s1[7] = highbd_idct_add_dual(s2[7], s1[8]);
  s1[8] = highbd_idct_sub_dual(s2[7], s1[8]);
  s1[9] = highbd_idct_sub_dual(s2[6], s1[9]);
  s1[10] = highbd_idct_sub_dual(s2[5], s2[10]);
  s1[11] = highbd_idct_sub_dual(s2[4], s2[11]);
  s1[12] = highbd_idct_sub_dual(s2[3], s2[12]);
  s1[13] = highbd_idct_sub_dual(s2[2], s2[13]);
  s1[14] = highbd_idct_sub_dual(s2[1], s1[14]);
  s1[15] = highbd_idct_sub_dual(s2[0], s1[15]);

  s1[20] = sub_multiply_shift_and_narrow_s32_dual(s3[27], s2[20], cospi_16_64);
  s1[27] = add_multiply_shift_and_narrow_s32_dual(s2[20], s3[27], cospi_16_64);

  s1[21] = sub_multiply_shift_and_narrow_s32_dual(s3[26], s2[21], cospi_16_64);
  s1[26] = add_multiply_shift_and_narrow_s32_dual(s2[21], s3[26], cospi_16_64);

  s2[22] = sub_multiply_shift_and_narrow_s32_dual(s3[25], s1[22], cospi_16_64);
  s1[25] = add_multiply_shift_and_narrow_s32_dual(s1[22], s3[25], cospi_16_64);

  s2[23] = sub_multiply_shift_and_narrow_s32_dual(s3[24], s1[23], cospi_16_64);
  s1[24] = add_multiply_shift_and_narrow_s32_dual(s1[23], s3[24], cospi_16_64);

  // final stage
  out[0] = highbd_idct_add_dual(s1[0], s2[31]);
  out[1] = highbd_idct_add_dual(s1[1], s2[30]);
  out[2] = highbd_idct_add_dual(s1[2], s2[29]);
  out[3] = highbd_idct_add_dual(s1[3], s2[28]);
  out[4] = highbd_idct_add_dual(s1[4], s1[27]);
  out[5] = highbd_idct_add_dual(s1[5], s1[26]);
  out[6] = highbd_idct_add_dual(s1[6], s1[25]);
  out[7] = highbd_idct_add_dual(s1[7], s1[24]);
  out[8] = highbd_idct_add_dual(s1[8], s2[23]);
  out[9] = highbd_idct_add_dual(s1[9], s2[22]);
  out[10] = highbd_idct_add_dual(s1[10], s1[21]);
  out[11] = highbd_idct_add_dual(s1[11], s1[20]);
  out[12] = highbd_idct_add_dual(s1[12], s2[19]);
  out[13] = highbd_idct_add_dual(s1[13], s2[18]);
  out[14] = highbd_idct_add_dual(s1[14], s1[17]);
  out[15] = highbd_idct_add_dual(s1[15], s1[16]);
  out[16] = highbd_idct_sub_dual(s1[15], s1[16]);
  out[17] = highbd_idct_sub_dual(s1[14], s1[17]);
  out[18] = highbd_idct_sub_dual(s1[13], s2[18]);
  out[19] = highbd_idct_sub_dual(s1[12], s2[19]);
  out[20] = highbd_idct_sub_dual(s1[11], s1[20]);
  out[21] = highbd_idct_sub_dual(s1[10], s1[21]);
  out[22] = highbd_idct_sub_dual(s1[9], s2[22]);
  out[23] = highbd_idct_sub_dual(s1[8], s2[23]);
  out[24] = highbd_idct_sub_dual(s1[7], s1[24]);
  out[25] = highbd_idct_sub_dual(s1[6], s1[25]);
  out[26] = highbd_idct_sub_dual(s1[5], s1[26]);
  out[27] = highbd_idct_sub_dual(s1[4], s1[27]);
  out[28] = highbd_idct_sub_dual(s1[3], s2[28]);
  out[29] = highbd_idct_sub_dual(s1[2], s2[29]);
  out[30] = highbd_idct_sub_dual(s1[1], s2[30]);
  out[31] = highbd_idct_sub_dual(s1[0], s2[31]);

  highbd_idct16x16_add_store(out, output, stride, bd);
  highbd_idct16x16_add_store(out + 16, output + 16 * stride, stride, bd);
}

void vpx_highbd_idct32x32_34_add_neon(const tran_low_t *input, uint16_t *dest,
                                      int stride, int bd) {
  int i;

  if (bd == 8) {
    int16_t temp[32 * 8];
    int16_t *t = temp;

    vpx_idct32_6_neon(input, t);

    for (i = 0; i < 32; i += 8) {
      vpx_idct32_8_neon(t, dest, stride, 1);
      t += (8 * 8);
      dest += 8;
    }
  } else {
    int32_t temp[32 * 8];
    int32_t *t = temp;

    vpx_highbd_idct32_6_neon(input, t);

    for (i = 0; i < 32; i += 8) {
      vpx_highbd_idct32_8_neon(t, dest, stride, bd);
      t += (8 * 8);
      dest += 8;
    }
  }
}
