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

static INLINE void load_8x8_s32_dual(
    const tran_low_t *input, int32x4x2_t *const in0, int32x4x2_t *const in1,
    int32x4x2_t *const in2, int32x4x2_t *const in3, int32x4x2_t *const in4,
    int32x4x2_t *const in5, int32x4x2_t *const in6, int32x4x2_t *const in7) {
  in0->val[0] = vld1q_s32(input);
  in0->val[1] = vld1q_s32(input + 4);
  input += 32;
  in1->val[0] = vld1q_s32(input);
  in1->val[1] = vld1q_s32(input + 4);
  input += 32;
  in2->val[0] = vld1q_s32(input);
  in2->val[1] = vld1q_s32(input + 4);
  input += 32;
  in3->val[0] = vld1q_s32(input);
  in3->val[1] = vld1q_s32(input + 4);
  input += 32;
  in4->val[0] = vld1q_s32(input);
  in4->val[1] = vld1q_s32(input + 4);
  input += 32;
  in5->val[0] = vld1q_s32(input);
  in5->val[1] = vld1q_s32(input + 4);
  input += 32;
  in6->val[0] = vld1q_s32(input);
  in6->val[1] = vld1q_s32(input + 4);
  input += 32;
  in7->val[0] = vld1q_s32(input);
  in7->val[1] = vld1q_s32(input + 4);
}

static INLINE void load_4x8_s32_dual(const tran_low_t *input,
                                     int32x4_t *const in0, int32x4_t *const in1,
                                     int32x4_t *const in2, int32x4_t *const in3,
                                     int32x4_t *const in4, int32x4_t *const in5,
                                     int32x4_t *const in6,
                                     int32x4_t *const in7) {
  *in0 = vld1q_s32(input);
  input += 32;
  *in1 = vld1q_s32(input);
  input += 32;
  *in2 = vld1q_s32(input);
  input += 32;
  *in3 = vld1q_s32(input);
  input += 32;
  *in4 = vld1q_s32(input);
  input += 32;
  *in5 = vld1q_s32(input);
  input += 32;
  *in6 = vld1q_s32(input);
  input += 32;
  *in7 = vld1q_s32(input);
}

// Only for the first pass of the  _135_ variant. Since it only uses values from
// the top left 16x16 it can safely assume all the remaining values are 0 and
// skip an awful lot of calculations. In fact, only the first 12 columns make
// the cut. None of the elements in the 13th, 14th, 15th or 16th columns are
// used so it skips any calls to input[12|13|14|15] too.
// In C this does a single row of 32 for each call. Here it transposes the top
// left 12x8 to allow using SIMD.

// vp9/common/vp9_scan.c:vp9_default_iscan_32x32 arranges the first 135 non-zero
// coefficients as follows:
//      0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
//  0   0   2   5  10  17  25  38  47  62  83 101 121
//  1   1   4   8  15  22  30  45  58  74  92 112 133
//  2   3   7  12  18  28  36  52  64  82 102 118
//  3   6  11  16  23  31  43  60  73  90 109 126
//  4   9  14  19  29  37  50  65  78  98 116 134
//  5  13  20  26  35  44  54  72  85 105 123
//  6  21  27  33  42  53  63  80  94 113 132
//  7  24  32  39  48  57  71  88 104 120
//  8  34  40  46  56  68  81  96 111 130
//  9  41  49  55  67  77  91 107 124
// 10  51  59  66  76  89  99 119 131
// 11  61  69  75  87 100 114 129
// 12  70  79  86  97 108 122
// 13  84  93 103 110 125
// 14  98 106 115 127
// 15 117 128
static void vpx_highbd_idct32_12_neon(const tran_low_t *const input,
                                      int32_t *output) {
  int32x4x2_t in[12], s1[32], s2[32], s3[32], s4[32], s5[32], s6[32], s7[32],
      s8[32];

  load_8x8_s32_dual(input, &in[0], &in[1], &in[2], &in[3], &in[4], &in[5],
                    &in[6], &in[7]);
  transpose_s32_8x8(&in[0], &in[1], &in[2], &in[3], &in[4], &in[5], &in[6],
                    &in[7]);

  load_4x8_s32_dual(input + 8, &in[8].val[0], &in[8].val[1], &in[9].val[0],
                    &in[9].val[1], &in[10].val[0], &in[10].val[1],
                    &in[11].val[0], &in[11].val[1]);
  transpose_s32_4x8(&in[8].val[0], &in[8].val[1], &in[9].val[0], &in[9].val[1],
                    &in[10].val[0], &in[10].val[1], &in[11].val[0],
                    &in[11].val[1]);

  // stage 1
  s1[16] = multiply_shift_and_narrow_s32_dual(in[1], cospi_31_64);
  s1[31] = multiply_shift_and_narrow_s32_dual(in[1], cospi_1_64);

  s1[18] = multiply_shift_and_narrow_s32_dual(in[9], cospi_23_64);
  s1[29] = multiply_shift_and_narrow_s32_dual(in[9], cospi_9_64);

  s1[19] = multiply_shift_and_narrow_s32_dual(in[7], -cospi_25_64);
  s1[28] = multiply_shift_and_narrow_s32_dual(in[7], cospi_7_64);

  s1[20] = multiply_shift_and_narrow_s32_dual(in[5], cospi_27_64);
  s1[27] = multiply_shift_and_narrow_s32_dual(in[5], cospi_5_64);

  s1[21] = multiply_shift_and_narrow_s32_dual(in[11], -cospi_21_64);
  s1[26] = multiply_shift_and_narrow_s32_dual(in[11], cospi_11_64);

  s1[23] = multiply_shift_and_narrow_s32_dual(in[3], -cospi_29_64);
  s1[24] = multiply_shift_and_narrow_s32_dual(in[3], cospi_3_64);

  // stage 2
  s2[8] = multiply_shift_and_narrow_s32_dual(in[2], cospi_30_64);
  s2[15] = multiply_shift_and_narrow_s32_dual(in[2], cospi_2_64);

  s2[10] = multiply_shift_and_narrow_s32_dual(in[10], cospi_22_64);
  s2[13] = multiply_shift_and_narrow_s32_dual(in[10], cospi_10_64);

  s2[11] = multiply_shift_and_narrow_s32_dual(in[6], -cospi_26_64);
  s2[12] = multiply_shift_and_narrow_s32_dual(in[6], cospi_6_64);

  s2[18] = highbd_idct_sub_dual(s1[19], s1[18]);
  s2[19] = highbd_idct_add_dual(s1[18], s1[19]);
  s2[20] = highbd_idct_add_dual(s1[20], s1[21]);
  s2[21] = highbd_idct_sub_dual(s1[20], s1[21]);
  s2[26] = highbd_idct_sub_dual(s1[27], s1[26]);
  s2[27] = highbd_idct_add_dual(s1[26], s1[27]);
  s2[28] = highbd_idct_add_dual(s1[28], s1[29]);
  s2[29] = highbd_idct_sub_dual(s1[28], s1[29]);

  // stage 3
  s3[4] = multiply_shift_and_narrow_s32_dual(in[4], cospi_28_64);
  s3[7] = multiply_shift_and_narrow_s32_dual(in[4], cospi_4_64);

  s3[10] = highbd_idct_sub_dual(s2[11], s2[10]);
  s3[11] = highbd_idct_add_dual(s2[10], s2[11]);
  s3[12] = highbd_idct_add_dual(s2[12], s2[13]);
  s3[13] = highbd_idct_sub_dual(s2[12], s2[13]);

  s3[17] = multiply_accumulate_shift_and_narrow_s32_dual(s1[16], -cospi_4_64,
                                                         s1[31], cospi_28_64);
  s3[30] = multiply_accumulate_shift_and_narrow_s32_dual(s1[16], cospi_28_64,
                                                         s1[31], cospi_4_64);

  s3[18] = multiply_accumulate_shift_and_narrow_s32_dual(s2[18], -cospi_28_64,
                                                         s2[29], -cospi_4_64);
  s3[29] = multiply_accumulate_shift_and_narrow_s32_dual(s2[18], -cospi_4_64,
                                                         s2[29], cospi_28_64);

  s3[21] = multiply_accumulate_shift_and_narrow_s32_dual(s2[21], -cospi_20_64,
                                                         s2[26], cospi_12_64);
  s3[26] = multiply_accumulate_shift_and_narrow_s32_dual(s2[21], cospi_12_64,
                                                         s2[26], cospi_20_64);

  s3[22] = multiply_accumulate_shift_and_narrow_s32_dual(s1[23], -cospi_12_64,
                                                         s1[24], -cospi_20_64);
  s3[25] = multiply_accumulate_shift_and_narrow_s32_dual(s1[23], -cospi_20_64,
                                                         s1[24], cospi_12_64);

  // stage 4
  s4[0] = multiply_shift_and_narrow_s32_dual(in[0], cospi_16_64);
  s4[2] = multiply_shift_and_narrow_s32_dual(in[8], cospi_24_64);
  s4[3] = multiply_shift_and_narrow_s32_dual(in[8], cospi_8_64);

  s4[9] = multiply_accumulate_shift_and_narrow_s32_dual(s2[8], -cospi_8_64,
                                                        s2[15], cospi_24_64);
  s4[14] = multiply_accumulate_shift_and_narrow_s32_dual(s2[8], cospi_24_64,
                                                         s2[15], cospi_8_64);

  s4[10] = multiply_accumulate_shift_and_narrow_s32_dual(s3[10], -cospi_24_64,
                                                         s3[13], -cospi_8_64);
  s4[13] = multiply_accumulate_shift_and_narrow_s32_dual(s3[10], -cospi_8_64,
                                                         s3[13], cospi_24_64);

  s4[16] = highbd_idct_add_dual(s1[16], s2[19]);
  s4[17] = highbd_idct_add_dual(s3[17], s3[18]);
  s4[18] = highbd_idct_sub_dual(s3[17], s3[18]);
  s4[19] = highbd_idct_sub_dual(s1[16], s2[19]);
  s4[20] = highbd_idct_sub_dual(s1[23], s2[20]);
  s4[21] = highbd_idct_sub_dual(s3[22], s3[21]);
  s4[22] = highbd_idct_add_dual(s3[21], s3[22]);
  s4[23] = highbd_idct_add_dual(s2[20], s1[23]);
  s4[24] = highbd_idct_add_dual(s1[24], s2[27]);
  s4[25] = highbd_idct_add_dual(s3[25], s3[26]);
  s4[26] = highbd_idct_sub_dual(s3[25], s3[26]);
  s4[27] = highbd_idct_sub_dual(s1[24], s2[27]);
  s4[28] = highbd_idct_sub_dual(s1[31], s2[28]);
  s4[29] = highbd_idct_sub_dual(s3[30], s3[29]);
  s4[30] = highbd_idct_add_dual(s3[29], s3[30]);
  s4[31] = highbd_idct_add_dual(s2[28], s1[31]);

  // stage 5
  s5[0] = highbd_idct_add_dual(s4[0], s4[3]);
  s5[1] = highbd_idct_add_dual(s4[0], s4[2]);
  s5[2] = highbd_idct_sub_dual(s4[0], s4[2]);
  s5[3] = highbd_idct_sub_dual(s4[0], s4[3]);

  s5[5] = sub_multiply_shift_and_narrow_s32_dual(s3[7], s3[4], cospi_16_64);
  s5[6] = add_multiply_shift_and_narrow_s32_dual(s3[4], s3[7], cospi_16_64);

  s5[8] = highbd_idct_add_dual(s2[8], s3[11]);
  s5[9] = highbd_idct_add_dual(s4[9], s4[10]);
  s5[10] = highbd_idct_sub_dual(s4[9], s4[10]);
  s5[11] = highbd_idct_sub_dual(s2[8], s3[11]);
  s5[12] = highbd_idct_sub_dual(s2[15], s3[12]);
  s5[13] = highbd_idct_sub_dual(s4[14], s4[13]);
  s5[14] = highbd_idct_add_dual(s4[13], s4[14]);
  s5[15] = highbd_idct_add_dual(s2[15], s3[12]);

  s5[18] = multiply_accumulate_shift_and_narrow_s32_dual(s4[18], -cospi_8_64,
                                                         s4[29], cospi_24_64);
  s5[29] = multiply_accumulate_shift_and_narrow_s32_dual(s4[18], cospi_24_64,
                                                         s4[29], cospi_8_64);

  s5[19] = multiply_accumulate_shift_and_narrow_s32_dual(s4[19], -cospi_8_64,
                                                         s4[28], cospi_24_64);
  s5[28] = multiply_accumulate_shift_and_narrow_s32_dual(s4[19], cospi_24_64,
                                                         s4[28], cospi_8_64);

  s5[20] = multiply_accumulate_shift_and_narrow_s32_dual(s4[20], -cospi_24_64,
                                                         s4[27], -cospi_8_64);
  s5[27] = multiply_accumulate_shift_and_narrow_s32_dual(s4[20], -cospi_8_64,
                                                         s4[27], cospi_24_64);

  s5[21] = multiply_accumulate_shift_and_narrow_s32_dual(s4[21], -cospi_24_64,
                                                         s4[26], -cospi_8_64);
  s5[26] = multiply_accumulate_shift_and_narrow_s32_dual(s4[21], -cospi_8_64,
                                                         s4[26], cospi_24_64);

  // stage 6
  s6[0] = highbd_idct_add_dual(s5[0], s3[7]);
  s6[1] = highbd_idct_add_dual(s5[1], s5[6]);
  s6[2] = highbd_idct_add_dual(s5[2], s5[5]);
  s6[3] = highbd_idct_add_dual(s5[3], s3[4]);
  s6[4] = highbd_idct_sub_dual(s5[3], s3[4]);
  s6[5] = highbd_idct_sub_dual(s5[2], s5[5]);
  s6[6] = highbd_idct_sub_dual(s5[1], s5[6]);
  s6[7] = highbd_idct_sub_dual(s5[0], s3[7]);

  s6[10] = sub_multiply_shift_and_narrow_s32_dual(s5[13], s5[10], cospi_16_64);
  s6[13] = add_multiply_shift_and_narrow_s32_dual(s5[10], s5[13], cospi_16_64);

  s6[11] = sub_multiply_shift_and_narrow_s32_dual(s5[12], s5[11], cospi_16_64);
  s6[12] = add_multiply_shift_and_narrow_s32_dual(s5[11], s5[12], cospi_16_64);

  s6[16] = highbd_idct_add_dual(s4[16], s4[23]);
  s6[17] = highbd_idct_add_dual(s4[17], s4[22]);
  s6[18] = highbd_idct_add_dual(s5[18], s5[21]);
  s6[19] = highbd_idct_add_dual(s5[19], s5[20]);
  s6[20] = highbd_idct_sub_dual(s5[19], s5[20]);
  s6[21] = highbd_idct_sub_dual(s5[18], s5[21]);
  s6[22] = highbd_idct_sub_dual(s4[17], s4[22]);
  s6[23] = highbd_idct_sub_dual(s4[16], s4[23]);

  s6[24] = highbd_idct_sub_dual(s4[31], s4[24]);
  s6[25] = highbd_idct_sub_dual(s4[30], s4[25]);
  s6[26] = highbd_idct_sub_dual(s5[29], s5[26]);
  s6[27] = highbd_idct_sub_dual(s5[28], s5[27]);
  s6[28] = highbd_idct_add_dual(s5[27], s5[28]);
  s6[29] = highbd_idct_add_dual(s5[26], s5[29]);
  s6[30] = highbd_idct_add_dual(s4[25], s4[30]);
  s6[31] = highbd_idct_add_dual(s4[24], s4[31]);

  // stage 7
  s7[0] = highbd_idct_add_dual(s6[0], s5[15]);
  s7[1] = highbd_idct_add_dual(s6[1], s5[14]);
  s7[2] = highbd_idct_add_dual(s6[2], s6[13]);
  s7[3] = highbd_idct_add_dual(s6[3], s6[12]);
  s7[4] = highbd_idct_add_dual(s6[4], s6[11]);
  s7[5] = highbd_idct_add_dual(s6[5], s6[10]);
  s7[6] = highbd_idct_add_dual(s6[6], s5[9]);
  s7[7] = highbd_idct_add_dual(s6[7], s5[8]);
  s7[8] = highbd_idct_sub_dual(s6[7], s5[8]);
  s7[9] = highbd_idct_sub_dual(s6[6], s5[9]);
  s7[10] = highbd_idct_sub_dual(s6[5], s6[10]);
  s7[11] = highbd_idct_sub_dual(s6[4], s6[11]);
  s7[12] = highbd_idct_sub_dual(s6[3], s6[12]);
  s7[13] = highbd_idct_sub_dual(s6[2], s6[13]);
  s7[14] = highbd_idct_sub_dual(s6[1], s5[14]);
  s7[15] = highbd_idct_sub_dual(s6[0], s5[15]);

  s7[20] = sub_multiply_shift_and_narrow_s32_dual(s6[27], s6[20], cospi_16_64);
  s7[27] = add_multiply_shift_and_narrow_s32_dual(s6[20], s6[27], cospi_16_64);

  s7[21] = sub_multiply_shift_and_narrow_s32_dual(s6[26], s6[21], cospi_16_64);
  s7[26] = add_multiply_shift_and_narrow_s32_dual(s6[21], s6[26], cospi_16_64);

  s7[22] = sub_multiply_shift_and_narrow_s32_dual(s6[25], s6[22], cospi_16_64);
  s7[25] = add_multiply_shift_and_narrow_s32_dual(s6[22], s6[25], cospi_16_64);

  s7[23] = sub_multiply_shift_and_narrow_s32_dual(s6[24], s6[23], cospi_16_64);
  s7[24] = add_multiply_shift_and_narrow_s32_dual(s6[23], s6[24], cospi_16_64);

  // final stage
  s8[0] = highbd_idct_add_dual(s7[0], s6[31]);
  s8[1] = highbd_idct_add_dual(s7[1], s6[30]);
  s8[2] = highbd_idct_add_dual(s7[2], s6[29]);
  s8[3] = highbd_idct_add_dual(s7[3], s6[28]);
  s8[4] = highbd_idct_add_dual(s7[4], s7[27]);
  s8[5] = highbd_idct_add_dual(s7[5], s7[26]);
  s8[6] = highbd_idct_add_dual(s7[6], s7[25]);
  s8[7] = highbd_idct_add_dual(s7[7], s7[24]);
  s8[8] = highbd_idct_add_dual(s7[8], s7[23]);
  s8[9] = highbd_idct_add_dual(s7[9], s7[22]);
  s8[10] = highbd_idct_add_dual(s7[10], s7[21]);
  s8[11] = highbd_idct_add_dual(s7[11], s7[20]);
  s8[12] = highbd_idct_add_dual(s7[12], s6[19]);
  s8[13] = highbd_idct_add_dual(s7[13], s6[18]);
  s8[14] = highbd_idct_add_dual(s7[14], s6[17]);
  s8[15] = highbd_idct_add_dual(s7[15], s6[16]);
  s8[16] = highbd_idct_sub_dual(s7[15], s6[16]);
  s8[17] = highbd_idct_sub_dual(s7[14], s6[17]);
  s8[18] = highbd_idct_sub_dual(s7[13], s6[18]);
  s8[19] = highbd_idct_sub_dual(s7[12], s6[19]);
  s8[20] = highbd_idct_sub_dual(s7[11], s7[20]);
  s8[21] = highbd_idct_sub_dual(s7[10], s7[21]);
  s8[22] = highbd_idct_sub_dual(s7[9], s7[22]);
  s8[23] = highbd_idct_sub_dual(s7[8], s7[23]);
  s8[24] = highbd_idct_sub_dual(s7[7], s7[24]);
  s8[25] = highbd_idct_sub_dual(s7[6], s7[25]);
  s8[26] = highbd_idct_sub_dual(s7[5], s7[26]);
  s8[27] = highbd_idct_sub_dual(s7[4], s7[27]);
  s8[28] = highbd_idct_sub_dual(s7[3], s6[28]);
  s8[29] = highbd_idct_sub_dual(s7[2], s6[29]);
  s8[30] = highbd_idct_sub_dual(s7[1], s6[30]);
  s8[31] = highbd_idct_sub_dual(s7[0], s6[31]);

  vst1q_s32(output + 0, s8[0].val[0]);
  vst1q_s32(output + 4, s8[0].val[1]);
  output += 16;
  vst1q_s32(output + 0, s8[1].val[0]);
  vst1q_s32(output + 4, s8[1].val[1]);
  output += 16;
  vst1q_s32(output + 0, s8[2].val[0]);
  vst1q_s32(output + 4, s8[2].val[1]);
  output += 16;
  vst1q_s32(output + 0, s8[3].val[0]);
  vst1q_s32(output + 4, s8[3].val[1]);
  output += 16;
  vst1q_s32(output + 0, s8[4].val[0]);
  vst1q_s32(output + 4, s8[4].val[1]);
  output += 16;
  vst1q_s32(output + 0, s8[5].val[0]);
  vst1q_s32(output + 4, s8[5].val[1]);
  output += 16;
  vst1q_s32(output + 0, s8[6].val[0]);
  vst1q_s32(output + 4, s8[6].val[1]);
  output += 16;
  vst1q_s32(output + 0, s8[7].val[0]);
  vst1q_s32(output + 4, s8[7].val[1]);
  output += 16;

  vst1q_s32(output + 0, s8[8].val[0]);
  vst1q_s32(output + 4, s8[8].val[1]);
  output += 16;
  vst1q_s32(output + 0, s8[9].val[0]);
  vst1q_s32(output + 4, s8[9].val[1]);
  output += 16;
  vst1q_s32(output + 0, s8[10].val[0]);
  vst1q_s32(output + 4, s8[10].val[1]);
  output += 16;
  vst1q_s32(output + 0, s8[11].val[0]);
  vst1q_s32(output + 4, s8[11].val[1]);
  output += 16;
  vst1q_s32(output + 0, s8[12].val[0]);
  vst1q_s32(output + 4, s8[12].val[1]);
  output += 16;
  vst1q_s32(output + 0, s8[13].val[0]);
  vst1q_s32(output + 4, s8[13].val[1]);
  output += 16;
  vst1q_s32(output + 0, s8[14].val[0]);
  vst1q_s32(output + 4, s8[14].val[1]);
  output += 16;
  vst1q_s32(output + 0, s8[15].val[0]);
  vst1q_s32(output + 4, s8[15].val[1]);
  output += 16;

  vst1q_s32(output + 0, s8[16].val[0]);
  vst1q_s32(output + 4, s8[16].val[1]);
  output += 16;
  vst1q_s32(output + 0, s8[17].val[0]);
  vst1q_s32(output + 4, s8[17].val[1]);
  output += 16;
  vst1q_s32(output + 0, s8[18].val[0]);
  vst1q_s32(output + 4, s8[18].val[1]);
  output += 16;
  vst1q_s32(output + 0, s8[19].val[0]);
  vst1q_s32(output + 4, s8[19].val[1]);
  output += 16;
  vst1q_s32(output + 0, s8[20].val[0]);
  vst1q_s32(output + 4, s8[20].val[1]);
  output += 16;
  vst1q_s32(output + 0, s8[21].val[0]);
  vst1q_s32(output + 4, s8[21].val[1]);
  output += 16;
  vst1q_s32(output + 0, s8[22].val[0]);
  vst1q_s32(output + 4, s8[22].val[1]);
  output += 16;
  vst1q_s32(output + 0, s8[23].val[0]);
  vst1q_s32(output + 4, s8[23].val[1]);
  output += 16;

  vst1q_s32(output + 0, s8[24].val[0]);
  vst1q_s32(output + 4, s8[24].val[1]);
  output += 16;
  vst1q_s32(output + 0, s8[25].val[0]);
  vst1q_s32(output + 4, s8[25].val[1]);
  output += 16;
  vst1q_s32(output + 0, s8[26].val[0]);
  vst1q_s32(output + 4, s8[26].val[1]);
  output += 16;
  vst1q_s32(output + 0, s8[27].val[0]);
  vst1q_s32(output + 4, s8[27].val[1]);
  output += 16;
  vst1q_s32(output + 0, s8[28].val[0]);
  vst1q_s32(output + 4, s8[28].val[1]);
  output += 16;
  vst1q_s32(output + 0, s8[29].val[0]);
  vst1q_s32(output + 4, s8[29].val[1]);
  output += 16;
  vst1q_s32(output + 0, s8[30].val[0]);
  vst1q_s32(output + 4, s8[30].val[1]);
  output += 16;
  vst1q_s32(output + 0, s8[31].val[0]);
  vst1q_s32(output + 4, s8[31].val[1]);
}

static void vpx_highbd_idct32_16_neon(const int32_t *const input,
                                      uint16_t *const output, const int stride,
                                      const int bd) {
  int32x4x2_t in[16], s1[32], s2[32], s3[32], s4[32], s5[32], s6[32], s7[32],
      out[32];

  load_and_transpose_s32_8x8(input, 16, &in[0], &in[1], &in[2], &in[3], &in[4],
                             &in[5], &in[6], &in[7]);

  load_and_transpose_s32_8x8(input + 8, 16, &in[8], &in[9], &in[10], &in[11],
                             &in[12], &in[13], &in[14], &in[15]);

  // stage 1
  s1[16] = multiply_shift_and_narrow_s32_dual(in[1], cospi_31_64);
  s1[31] = multiply_shift_and_narrow_s32_dual(in[1], cospi_1_64);

  s1[17] = multiply_shift_and_narrow_s32_dual(in[15], -cospi_17_64);
  s1[30] = multiply_shift_and_narrow_s32_dual(in[15], cospi_15_64);

  s1[18] = multiply_shift_and_narrow_s32_dual(in[9], cospi_23_64);
  s1[29] = multiply_shift_and_narrow_s32_dual(in[9], cospi_9_64);

  s1[19] = multiply_shift_and_narrow_s32_dual(in[7], -cospi_25_64);
  s1[28] = multiply_shift_and_narrow_s32_dual(in[7], cospi_7_64);

  s1[20] = multiply_shift_and_narrow_s32_dual(in[5], cospi_27_64);
  s1[27] = multiply_shift_and_narrow_s32_dual(in[5], cospi_5_64);

  s1[21] = multiply_shift_and_narrow_s32_dual(in[11], -cospi_21_64);
  s1[26] = multiply_shift_and_narrow_s32_dual(in[11], cospi_11_64);

  s1[22] = multiply_shift_and_narrow_s32_dual(in[13], cospi_19_64);
  s1[25] = multiply_shift_and_narrow_s32_dual(in[13], cospi_13_64);

  s1[23] = multiply_shift_and_narrow_s32_dual(in[3], -cospi_29_64);
  s1[24] = multiply_shift_and_narrow_s32_dual(in[3], cospi_3_64);

  // stage 2
  s2[8] = multiply_shift_and_narrow_s32_dual(in[2], cospi_30_64);
  s2[15] = multiply_shift_and_narrow_s32_dual(in[2], cospi_2_64);

  s2[9] = multiply_shift_and_narrow_s32_dual(in[14], -cospi_18_64);
  s2[14] = multiply_shift_and_narrow_s32_dual(in[14], cospi_14_64);

  s2[10] = multiply_shift_and_narrow_s32_dual(in[10], cospi_22_64);
  s2[13] = multiply_shift_and_narrow_s32_dual(in[10], cospi_10_64);

  s2[11] = multiply_shift_and_narrow_s32_dual(in[6], -cospi_26_64);
  s2[12] = multiply_shift_and_narrow_s32_dual(in[6], cospi_6_64);

  s2[16] = highbd_idct_add_dual(s1[16], s1[17]);
  s2[17] = highbd_idct_sub_dual(s1[16], s1[17]);
  s2[18] = highbd_idct_sub_dual(s1[19], s1[18]);
  s2[19] = highbd_idct_add_dual(s1[18], s1[19]);
  s2[20] = highbd_idct_add_dual(s1[20], s1[21]);
  s2[21] = highbd_idct_sub_dual(s1[20], s1[21]);
  s2[22] = highbd_idct_sub_dual(s1[23], s1[22]);
  s2[23] = highbd_idct_add_dual(s1[22], s1[23]);
  s2[24] = highbd_idct_add_dual(s1[24], s1[25]);
  s2[25] = highbd_idct_sub_dual(s1[24], s1[25]);
  s2[26] = highbd_idct_sub_dual(s1[27], s1[26]);
  s2[27] = highbd_idct_add_dual(s1[26], s1[27]);
  s2[28] = highbd_idct_add_dual(s1[28], s1[29]);
  s2[29] = highbd_idct_sub_dual(s1[28], s1[29]);
  s2[30] = highbd_idct_sub_dual(s1[31], s1[30]);
  s2[31] = highbd_idct_add_dual(s1[30], s1[31]);

  // stage 3
  s3[4] = multiply_shift_and_narrow_s32_dual(in[4], cospi_28_64);
  s3[7] = multiply_shift_and_narrow_s32_dual(in[4], cospi_4_64);

  s3[5] = multiply_shift_and_narrow_s32_dual(in[12], -cospi_20_64);
  s3[6] = multiply_shift_and_narrow_s32_dual(in[12], cospi_12_64);

  s3[8] = highbd_idct_add_dual(s2[8], s2[9]);
  s3[9] = highbd_idct_sub_dual(s2[8], s2[9]);
  s3[10] = highbd_idct_sub_dual(s2[11], s2[10]);
  s3[11] = highbd_idct_add_dual(s2[10], s2[11]);
  s3[12] = highbd_idct_add_dual(s2[12], s2[13]);
  s3[13] = highbd_idct_sub_dual(s2[12], s2[13]);
  s3[14] = highbd_idct_sub_dual(s2[15], s2[14]);
  s3[15] = highbd_idct_add_dual(s2[14], s2[15]);

  s3[17] = multiply_accumulate_shift_and_narrow_s32_dual(s2[17], -cospi_4_64,
                                                         s2[30], cospi_28_64);
  s3[30] = multiply_accumulate_shift_and_narrow_s32_dual(s2[17], cospi_28_64,
                                                         s2[30], cospi_4_64);

  s3[18] = multiply_accumulate_shift_and_narrow_s32_dual(s2[18], -cospi_28_64,
                                                         s2[29], -cospi_4_64);
  s3[29] = multiply_accumulate_shift_and_narrow_s32_dual(s2[18], -cospi_4_64,
                                                         s2[29], cospi_28_64);

  s3[21] = multiply_accumulate_shift_and_narrow_s32_dual(s2[21], -cospi_20_64,
                                                         s2[26], cospi_12_64);
  s3[26] = multiply_accumulate_shift_and_narrow_s32_dual(s2[21], cospi_12_64,
                                                         s2[26], cospi_20_64);

  s3[22] = multiply_accumulate_shift_and_narrow_s32_dual(s2[22], -cospi_12_64,
                                                         s2[25], -cospi_20_64);
  s3[25] = multiply_accumulate_shift_and_narrow_s32_dual(s2[22], -cospi_20_64,
                                                         s2[25], cospi_12_64);

  // stage 4
  s4[0] = multiply_shift_and_narrow_s32_dual(in[0], cospi_16_64);
  s4[2] = multiply_shift_and_narrow_s32_dual(in[8], cospi_24_64);
  s4[3] = multiply_shift_and_narrow_s32_dual(in[8], cospi_8_64);

  s4[4] = highbd_idct_add_dual(s3[4], s3[5]);
  s4[5] = highbd_idct_sub_dual(s3[4], s3[5]);
  s4[6] = highbd_idct_sub_dual(s3[7], s3[6]);
  s4[7] = highbd_idct_add_dual(s3[6], s3[7]);

  s4[9] = multiply_accumulate_shift_and_narrow_s32_dual(s3[9], -cospi_8_64,
                                                        s3[14], cospi_24_64);
  s4[14] = multiply_accumulate_shift_and_narrow_s32_dual(s3[9], cospi_24_64,
                                                         s3[14], cospi_8_64);

  s4[10] = multiply_accumulate_shift_and_narrow_s32_dual(s3[10], -cospi_24_64,
                                                         s3[13], -cospi_8_64);
  s4[13] = multiply_accumulate_shift_and_narrow_s32_dual(s3[10], -cospi_8_64,
                                                         s3[13], cospi_24_64);

  s4[16] = highbd_idct_add_dual(s2[16], s2[19]);
  s4[17] = highbd_idct_add_dual(s3[17], s3[18]);
  s4[18] = highbd_idct_sub_dual(s3[17], s3[18]);
  s4[19] = highbd_idct_sub_dual(s2[16], s2[19]);
  s4[20] = highbd_idct_sub_dual(s2[23], s2[20]);
  s4[21] = highbd_idct_sub_dual(s3[22], s3[21]);
  s4[22] = highbd_idct_add_dual(s3[21], s3[22]);
  s4[23] = highbd_idct_add_dual(s2[20], s2[23]);
  s4[24] = highbd_idct_add_dual(s2[24], s2[27]);
  s4[25] = highbd_idct_add_dual(s3[25], s3[26]);
  s4[26] = highbd_idct_sub_dual(s3[25], s3[26]);
  s4[27] = highbd_idct_sub_dual(s2[24], s2[27]);
  s4[28] = highbd_idct_sub_dual(s2[31], s2[28]);
  s4[29] = highbd_idct_sub_dual(s3[30], s3[29]);
  s4[30] = highbd_idct_add_dual(s3[29], s3[30]);
  s4[31] = highbd_idct_add_dual(s2[28], s2[31]);

  // stage 5
  s5[0] = highbd_idct_add_dual(s4[0], s4[3]);
  s5[1] = highbd_idct_add_dual(s4[0], s4[2]);
  s5[2] = highbd_idct_sub_dual(s4[0], s4[2]);
  s5[3] = highbd_idct_sub_dual(s4[0], s4[3]);

  s5[5] = sub_multiply_shift_and_narrow_s32_dual(s4[6], s4[5], cospi_16_64);
  s5[6] = add_multiply_shift_and_narrow_s32_dual(s4[5], s4[6], cospi_16_64);

  s5[8] = highbd_idct_add_dual(s3[8], s3[11]);
  s5[9] = highbd_idct_add_dual(s4[9], s4[10]);
  s5[10] = highbd_idct_sub_dual(s4[9], s4[10]);
  s5[11] = highbd_idct_sub_dual(s3[8], s3[11]);
  s5[12] = highbd_idct_sub_dual(s3[15], s3[12]);
  s5[13] = highbd_idct_sub_dual(s4[14], s4[13]);
  s5[14] = highbd_idct_add_dual(s4[13], s4[14]);
  s5[15] = highbd_idct_add_dual(s3[15], s3[12]);

  s5[18] = multiply_accumulate_shift_and_narrow_s32_dual(s4[18], -cospi_8_64,
                                                         s4[29], cospi_24_64);
  s5[29] = multiply_accumulate_shift_and_narrow_s32_dual(s4[18], cospi_24_64,
                                                         s4[29], cospi_8_64);

  s5[19] = multiply_accumulate_shift_and_narrow_s32_dual(s4[19], -cospi_8_64,
                                                         s4[28], cospi_24_64);
  s5[28] = multiply_accumulate_shift_and_narrow_s32_dual(s4[19], cospi_24_64,
                                                         s4[28], cospi_8_64);

  s5[20] = multiply_accumulate_shift_and_narrow_s32_dual(s4[20], -cospi_24_64,
                                                         s4[27], -cospi_8_64);
  s5[27] = multiply_accumulate_shift_and_narrow_s32_dual(s4[20], -cospi_8_64,
                                                         s4[27], cospi_24_64);

  s5[21] = multiply_accumulate_shift_and_narrow_s32_dual(s4[21], -cospi_24_64,
                                                         s4[26], -cospi_8_64);
  s5[26] = multiply_accumulate_shift_and_narrow_s32_dual(s4[21], -cospi_8_64,
                                                         s4[26], cospi_24_64);

  // stage 6
  s6[0] = highbd_idct_add_dual(s5[0], s4[7]);
  s6[1] = highbd_idct_add_dual(s5[1], s5[6]);
  s6[2] = highbd_idct_add_dual(s5[2], s5[5]);
  s6[3] = highbd_idct_add_dual(s5[3], s4[4]);
  s6[4] = highbd_idct_sub_dual(s5[3], s4[4]);
  s6[5] = highbd_idct_sub_dual(s5[2], s5[5]);
  s6[6] = highbd_idct_sub_dual(s5[1], s5[6]);
  s6[7] = highbd_idct_sub_dual(s5[0], s4[7]);

  s6[10] = sub_multiply_shift_and_narrow_s32_dual(s5[13], s5[10], cospi_16_64);
  s6[13] = add_multiply_shift_and_narrow_s32_dual(s5[10], s5[13], cospi_16_64);

  s6[11] = sub_multiply_shift_and_narrow_s32_dual(s5[12], s5[11], cospi_16_64);
  s6[12] = add_multiply_shift_and_narrow_s32_dual(s5[11], s5[12], cospi_16_64);

  s6[16] = highbd_idct_add_dual(s4[16], s4[23]);
  s6[17] = highbd_idct_add_dual(s4[17], s4[22]);
  s6[18] = highbd_idct_add_dual(s5[18], s5[21]);
  s6[19] = highbd_idct_add_dual(s5[19], s5[20]);
  s6[20] = highbd_idct_sub_dual(s5[19], s5[20]);
  s6[21] = highbd_idct_sub_dual(s5[18], s5[21]);
  s6[22] = highbd_idct_sub_dual(s4[17], s4[22]);
  s6[23] = highbd_idct_sub_dual(s4[16], s4[23]);
  s6[24] = highbd_idct_sub_dual(s4[31], s4[24]);
  s6[25] = highbd_idct_sub_dual(s4[30], s4[25]);
  s6[26] = highbd_idct_sub_dual(s5[29], s5[26]);
  s6[27] = highbd_idct_sub_dual(s5[28], s5[27]);
  s6[28] = highbd_idct_add_dual(s5[27], s5[28]);
  s6[29] = highbd_idct_add_dual(s5[26], s5[29]);
  s6[30] = highbd_idct_add_dual(s4[25], s4[30]);
  s6[31] = highbd_idct_add_dual(s4[24], s4[31]);

  // stage 7
  s7[0] = highbd_idct_add_dual(s6[0], s5[15]);
  s7[1] = highbd_idct_add_dual(s6[1], s5[14]);
  s7[2] = highbd_idct_add_dual(s6[2], s6[13]);
  s7[3] = highbd_idct_add_dual(s6[3], s6[12]);
  s7[4] = highbd_idct_add_dual(s6[4], s6[11]);
  s7[5] = highbd_idct_add_dual(s6[5], s6[10]);
  s7[6] = highbd_idct_add_dual(s6[6], s5[9]);
  s7[7] = highbd_idct_add_dual(s6[7], s5[8]);
  s7[8] = highbd_idct_sub_dual(s6[7], s5[8]);
  s7[9] = highbd_idct_sub_dual(s6[6], s5[9]);
  s7[10] = highbd_idct_sub_dual(s6[5], s6[10]);
  s7[11] = highbd_idct_sub_dual(s6[4], s6[11]);
  s7[12] = highbd_idct_sub_dual(s6[3], s6[12]);
  s7[13] = highbd_idct_sub_dual(s6[2], s6[13]);
  s7[14] = highbd_idct_sub_dual(s6[1], s5[14]);
  s7[15] = highbd_idct_sub_dual(s6[0], s5[15]);

  s7[20] = sub_multiply_shift_and_narrow_s32_dual(s6[27], s6[20], cospi_16_64);
  s7[27] = add_multiply_shift_and_narrow_s32_dual(s6[20], s6[27], cospi_16_64);

  s7[21] = sub_multiply_shift_and_narrow_s32_dual(s6[26], s6[21], cospi_16_64);
  s7[26] = add_multiply_shift_and_narrow_s32_dual(s6[21], s6[26], cospi_16_64);

  s7[22] = sub_multiply_shift_and_narrow_s32_dual(s6[25], s6[22], cospi_16_64);
  s7[25] = add_multiply_shift_and_narrow_s32_dual(s6[22], s6[25], cospi_16_64);

  s7[23] = sub_multiply_shift_and_narrow_s32_dual(s6[24], s6[23], cospi_16_64);
  s7[24] = add_multiply_shift_and_narrow_s32_dual(s6[23], s6[24], cospi_16_64);

  // final stage
  out[0] = highbd_idct_add_dual(s7[0], s6[31]);
  out[1] = highbd_idct_add_dual(s7[1], s6[30]);
  out[2] = highbd_idct_add_dual(s7[2], s6[29]);
  out[3] = highbd_idct_add_dual(s7[3], s6[28]);
  out[4] = highbd_idct_add_dual(s7[4], s7[27]);
  out[5] = highbd_idct_add_dual(s7[5], s7[26]);
  out[6] = highbd_idct_add_dual(s7[6], s7[25]);
  out[7] = highbd_idct_add_dual(s7[7], s7[24]);
  out[8] = highbd_idct_add_dual(s7[8], s7[23]);
  out[9] = highbd_idct_add_dual(s7[9], s7[22]);
  out[10] = highbd_idct_add_dual(s7[10], s7[21]);
  out[11] = highbd_idct_add_dual(s7[11], s7[20]);
  out[12] = highbd_idct_add_dual(s7[12], s6[19]);
  out[13] = highbd_idct_add_dual(s7[13], s6[18]);
  out[14] = highbd_idct_add_dual(s7[14], s6[17]);
  out[15] = highbd_idct_add_dual(s7[15], s6[16]);
  out[16] = highbd_idct_sub_dual(s7[15], s6[16]);
  out[17] = highbd_idct_sub_dual(s7[14], s6[17]);
  out[18] = highbd_idct_sub_dual(s7[13], s6[18]);
  out[19] = highbd_idct_sub_dual(s7[12], s6[19]);
  out[20] = highbd_idct_sub_dual(s7[11], s7[20]);
  out[21] = highbd_idct_sub_dual(s7[10], s7[21]);
  out[22] = highbd_idct_sub_dual(s7[9], s7[22]);
  out[23] = highbd_idct_sub_dual(s7[8], s7[23]);
  out[24] = highbd_idct_sub_dual(s7[7], s7[24]);
  out[25] = highbd_idct_sub_dual(s7[6], s7[25]);
  out[26] = highbd_idct_sub_dual(s7[5], s7[26]);
  out[27] = highbd_idct_sub_dual(s7[4], s7[27]);
  out[28] = highbd_idct_sub_dual(s7[3], s6[28]);
  out[29] = highbd_idct_sub_dual(s7[2], s6[29]);
  out[30] = highbd_idct_sub_dual(s7[1], s6[30]);
  out[31] = highbd_idct_sub_dual(s7[0], s6[31]);

  highbd_idct16x16_add_store(out, output, stride, bd);
  highbd_idct16x16_add_store(out + 16, output + 16 * stride, stride, bd);
}

void vpx_highbd_idct32x32_135_add_neon(const tran_low_t *input, uint16_t *dest,
                                       int stride, int bd) {
  int i;

  if (bd == 8) {
    int16_t temp[32 * 16];
    int16_t *t = temp;
    vpx_idct32_12_neon(input, temp);
    vpx_idct32_12_neon(input + 32 * 8, temp + 8);

    for (i = 0; i < 32; i += 8) {
      vpx_idct32_16_neon(t, dest, stride, 1);
      t += (16 * 8);
      dest += 8;
    }
  } else {
    int32_t temp[32 * 16];
    int32_t *t = temp;
    vpx_highbd_idct32_12_neon(input, temp);
    vpx_highbd_idct32_12_neon(input + 32 * 8, temp + 8);

    for (i = 0; i < 32; i += 8) {
      vpx_highbd_idct32_16_neon(t, dest, stride, bd);
      t += (16 * 8);
      dest += 8;
    }
  }
}
