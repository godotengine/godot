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
#include "vpx_dsp/txfm_common.h"
#include "vpx_dsp/arm/mem_neon.h"
#include "vpx_dsp/arm/transpose_neon.h"
#include "vpx_dsp/arm/fdct_neon.h"
#include "vpx_dsp/arm/fdct32x32_neon.h"

// Most gcc 4.9 distributions outside of Android do not generate correct code
// for this function.
#if !defined(__clang__) && !defined(__ANDROID__) && defined(__GNUC__) && \
    __GNUC__ == 4 && __GNUC_MINOR__ <= 9

void vpx_fdct32x32_neon(const int16_t *input, tran_low_t *output, int stride) {
  vpx_fdct32x32_c(input, output, stride);
}

void vpx_fdct32x32_rd_neon(const int16_t *input, tran_low_t *output,
                           int stride) {
  vpx_fdct32x32_rd_c(input, output, stride);
}

#else

void vpx_fdct32x32_neon(const int16_t *input, tran_low_t *output, int stride) {
  int16x8_t temp0[32];
  int16x8_t temp1[32];
  int16x8_t temp2[32];
  int16x8_t temp3[32];
  int16x8_t temp4[32];
  int16x8_t temp5[32];

  // Process in 8x32 columns.
  load_cross(input, stride, temp0);
  scale_input(temp0, temp5);
  dct_body_first_pass(temp5, temp1);

  load_cross(input + 8, stride, temp0);
  scale_input(temp0, temp5);
  dct_body_first_pass(temp5, temp2);

  load_cross(input + 16, stride, temp0);
  scale_input(temp0, temp5);
  dct_body_first_pass(temp5, temp3);

  load_cross(input + 24, stride, temp0);
  scale_input(temp0, temp5);
  dct_body_first_pass(temp5, temp4);

  // Generate the top row by munging the first set of 8 from each one together.
  transpose_s16_8x8q(&temp1[0], &temp0[0]);
  transpose_s16_8x8q(&temp2[0], &temp0[8]);
  transpose_s16_8x8q(&temp3[0], &temp0[16]);
  transpose_s16_8x8q(&temp4[0], &temp0[24]);

  dct_body_second_pass(temp0, temp5);

  transpose_s16_8x8(&temp5[0], &temp5[1], &temp5[2], &temp5[3], &temp5[4],
                    &temp5[5], &temp5[6], &temp5[7]);
  transpose_s16_8x8(&temp5[8], &temp5[9], &temp5[10], &temp5[11], &temp5[12],
                    &temp5[13], &temp5[14], &temp5[15]);
  transpose_s16_8x8(&temp5[16], &temp5[17], &temp5[18], &temp5[19], &temp5[20],
                    &temp5[21], &temp5[22], &temp5[23]);
  transpose_s16_8x8(&temp5[24], &temp5[25], &temp5[26], &temp5[27], &temp5[28],
                    &temp5[29], &temp5[30], &temp5[31]);
  store(output, temp5);

  // Second row of 8x32.
  transpose_s16_8x8q(&temp1[8], &temp0[0]);
  transpose_s16_8x8q(&temp2[8], &temp0[8]);
  transpose_s16_8x8q(&temp3[8], &temp0[16]);
  transpose_s16_8x8q(&temp4[8], &temp0[24]);

  dct_body_second_pass(temp0, temp5);

  transpose_s16_8x8(&temp5[0], &temp5[1], &temp5[2], &temp5[3], &temp5[4],
                    &temp5[5], &temp5[6], &temp5[7]);
  transpose_s16_8x8(&temp5[8], &temp5[9], &temp5[10], &temp5[11], &temp5[12],
                    &temp5[13], &temp5[14], &temp5[15]);
  transpose_s16_8x8(&temp5[16], &temp5[17], &temp5[18], &temp5[19], &temp5[20],
                    &temp5[21], &temp5[22], &temp5[23]);
  transpose_s16_8x8(&temp5[24], &temp5[25], &temp5[26], &temp5[27], &temp5[28],
                    &temp5[29], &temp5[30], &temp5[31]);
  store(output + 8 * 32, temp5);

  // Third row of 8x32
  transpose_s16_8x8q(&temp1[16], &temp0[0]);
  transpose_s16_8x8q(&temp2[16], &temp0[8]);
  transpose_s16_8x8q(&temp3[16], &temp0[16]);
  transpose_s16_8x8q(&temp4[16], &temp0[24]);

  dct_body_second_pass(temp0, temp5);

  transpose_s16_8x8(&temp5[0], &temp5[1], &temp5[2], &temp5[3], &temp5[4],
                    &temp5[5], &temp5[6], &temp5[7]);
  transpose_s16_8x8(&temp5[8], &temp5[9], &temp5[10], &temp5[11], &temp5[12],
                    &temp5[13], &temp5[14], &temp5[15]);
  transpose_s16_8x8(&temp5[16], &temp5[17], &temp5[18], &temp5[19], &temp5[20],
                    &temp5[21], &temp5[22], &temp5[23]);
  transpose_s16_8x8(&temp5[24], &temp5[25], &temp5[26], &temp5[27], &temp5[28],
                    &temp5[29], &temp5[30], &temp5[31]);
  store(output + 16 * 32, temp5);

  // Final row of 8x32.
  transpose_s16_8x8q(&temp1[24], &temp0[0]);
  transpose_s16_8x8q(&temp2[24], &temp0[8]);
  transpose_s16_8x8q(&temp3[24], &temp0[16]);
  transpose_s16_8x8q(&temp4[24], &temp0[24]);

  dct_body_second_pass(temp0, temp5);

  transpose_s16_8x8(&temp5[0], &temp5[1], &temp5[2], &temp5[3], &temp5[4],
                    &temp5[5], &temp5[6], &temp5[7]);
  transpose_s16_8x8(&temp5[8], &temp5[9], &temp5[10], &temp5[11], &temp5[12],
                    &temp5[13], &temp5[14], &temp5[15]);
  transpose_s16_8x8(&temp5[16], &temp5[17], &temp5[18], &temp5[19], &temp5[20],
                    &temp5[21], &temp5[22], &temp5[23]);
  transpose_s16_8x8(&temp5[24], &temp5[25], &temp5[26], &temp5[27], &temp5[28],
                    &temp5[29], &temp5[30], &temp5[31]);
  store(output + 24 * 32, temp5);
}

void vpx_fdct32x32_rd_neon(const int16_t *input, tran_low_t *output,
                           int stride) {
  int16x8_t temp0[32];
  int16x8_t temp1[32];
  int16x8_t temp2[32];
  int16x8_t temp3[32];
  int16x8_t temp4[32];
  int16x8_t temp5[32];

  // Process in 8x32 columns.
  load_cross(input, stride, temp0);
  scale_input(temp0, temp5);
  dct_body_first_pass(temp5, temp1);

  load_cross(input + 8, stride, temp0);
  scale_input(temp0, temp5);
  dct_body_first_pass(temp5, temp2);

  load_cross(input + 16, stride, temp0);
  scale_input(temp0, temp5);
  dct_body_first_pass(temp5, temp3);

  load_cross(input + 24, stride, temp0);
  scale_input(temp0, temp5);
  dct_body_first_pass(temp5, temp4);

  // Generate the top row by munging the first set of 8 from each one together.
  transpose_s16_8x8q(&temp1[0], &temp0[0]);
  transpose_s16_8x8q(&temp2[0], &temp0[8]);
  transpose_s16_8x8q(&temp3[0], &temp0[16]);
  transpose_s16_8x8q(&temp4[0], &temp0[24]);

  dct_body_second_pass_rd(temp0, temp5);

  transpose_s16_8x8(&temp5[0], &temp5[1], &temp5[2], &temp5[3], &temp5[4],
                    &temp5[5], &temp5[6], &temp5[7]);
  transpose_s16_8x8(&temp5[8], &temp5[9], &temp5[10], &temp5[11], &temp5[12],
                    &temp5[13], &temp5[14], &temp5[15]);
  transpose_s16_8x8(&temp5[16], &temp5[17], &temp5[18], &temp5[19], &temp5[20],
                    &temp5[21], &temp5[22], &temp5[23]);
  transpose_s16_8x8(&temp5[24], &temp5[25], &temp5[26], &temp5[27], &temp5[28],
                    &temp5[29], &temp5[30], &temp5[31]);
  store(output, temp5);

  // Second row of 8x32.
  transpose_s16_8x8q(&temp1[8], &temp0[0]);
  transpose_s16_8x8q(&temp2[8], &temp0[8]);
  transpose_s16_8x8q(&temp3[8], &temp0[16]);
  transpose_s16_8x8q(&temp4[8], &temp0[24]);

  dct_body_second_pass_rd(temp0, temp5);

  transpose_s16_8x8(&temp5[0], &temp5[1], &temp5[2], &temp5[3], &temp5[4],
                    &temp5[5], &temp5[6], &temp5[7]);
  transpose_s16_8x8(&temp5[8], &temp5[9], &temp5[10], &temp5[11], &temp5[12],
                    &temp5[13], &temp5[14], &temp5[15]);
  transpose_s16_8x8(&temp5[16], &temp5[17], &temp5[18], &temp5[19], &temp5[20],
                    &temp5[21], &temp5[22], &temp5[23]);
  transpose_s16_8x8(&temp5[24], &temp5[25], &temp5[26], &temp5[27], &temp5[28],
                    &temp5[29], &temp5[30], &temp5[31]);
  store(output + 8 * 32, temp5);

  // Third row of 8x32
  transpose_s16_8x8q(&temp1[16], &temp0[0]);
  transpose_s16_8x8q(&temp2[16], &temp0[8]);
  transpose_s16_8x8q(&temp3[16], &temp0[16]);
  transpose_s16_8x8q(&temp4[16], &temp0[24]);

  dct_body_second_pass_rd(temp0, temp5);

  transpose_s16_8x8(&temp5[0], &temp5[1], &temp5[2], &temp5[3], &temp5[4],
                    &temp5[5], &temp5[6], &temp5[7]);
  transpose_s16_8x8(&temp5[8], &temp5[9], &temp5[10], &temp5[11], &temp5[12],
                    &temp5[13], &temp5[14], &temp5[15]);
  transpose_s16_8x8(&temp5[16], &temp5[17], &temp5[18], &temp5[19], &temp5[20],
                    &temp5[21], &temp5[22], &temp5[23]);
  transpose_s16_8x8(&temp5[24], &temp5[25], &temp5[26], &temp5[27], &temp5[28],
                    &temp5[29], &temp5[30], &temp5[31]);
  store(output + 16 * 32, temp5);

  // Final row of 8x32.
  transpose_s16_8x8q(&temp1[24], &temp0[0]);
  transpose_s16_8x8q(&temp2[24], &temp0[8]);
  transpose_s16_8x8q(&temp3[24], &temp0[16]);
  transpose_s16_8x8q(&temp4[24], &temp0[24]);

  dct_body_second_pass_rd(temp0, temp5);

  transpose_s16_8x8(&temp5[0], &temp5[1], &temp5[2], &temp5[3], &temp5[4],
                    &temp5[5], &temp5[6], &temp5[7]);
  transpose_s16_8x8(&temp5[8], &temp5[9], &temp5[10], &temp5[11], &temp5[12],
                    &temp5[13], &temp5[14], &temp5[15]);
  transpose_s16_8x8(&temp5[16], &temp5[17], &temp5[18], &temp5[19], &temp5[20],
                    &temp5[21], &temp5[22], &temp5[23]);
  transpose_s16_8x8(&temp5[24], &temp5[25], &temp5[26], &temp5[27], &temp5[28],
                    &temp5[29], &temp5[30], &temp5[31]);
  store(output + 24 * 32, temp5);
}

#if CONFIG_VP9_HIGHBITDEPTH

void vpx_highbd_fdct32x32_neon(const int16_t *input, tran_low_t *output,
                               int stride) {
  int16x8_t temp0[32];
  int32x4_t left1[32], left2[32], left3[32], left4[32], right1[32], right2[32],
      right3[32], right4[32];
  int32x4_t left5[32], right5[32], left6[32], right6[32], left7[32], right7[32],
      left8[32], right8[32];
  int32x4_t temp1[32], temp2[32];

  // Process in 8x32 columns.
  load_cross(input, stride, temp0);
  highbd_scale_input(temp0, left1, right1);
  highbd_dct8x32_body_first_pass(left1, right1);
  highbd_partial_sub_round_shift(left1, right1);

  load_cross(input + 8, stride, temp0);
  highbd_scale_input(temp0, left2, right2);
  highbd_dct8x32_body_first_pass(left2, right2);
  highbd_partial_sub_round_shift(left2, right2);

  load_cross(input + 16, stride, temp0);
  highbd_scale_input(temp0, left3, right3);
  highbd_dct8x32_body_first_pass(left3, right3);
  highbd_partial_sub_round_shift(left3, right3);

  load_cross(input + 24, stride, temp0);
  highbd_scale_input(temp0, left4, right4);
  highbd_dct8x32_body_first_pass(left4, right4);
  highbd_partial_sub_round_shift(left4, right4);

  // Generate the top row by munging the first set of 8 from each one together.
  transpose_s32_8x8_2(left1, right1, temp1, temp2);
  transpose_s32_8x8_2(left2, right2, temp1 + 8, temp2 + 8);
  transpose_s32_8x8_2(left3, right3, temp1 + 16, temp2 + 16);
  transpose_s32_8x8_2(left4, right4, temp1 + 24, temp2 + 24);

  highbd_cross_input(temp1, temp2, left5, right5);
  highbd_dct8x32_body_second_pass(left5, right5);
  highbd_partial_add_round_shift(left5, right5);

  // Second row of 8x32.
  transpose_s32_8x8_2(left1 + 8, right1 + 8, temp1, temp2);
  transpose_s32_8x8_2(left2 + 8, right2 + 8, temp1 + 8, temp2 + 8);
  transpose_s32_8x8_2(left3 + 8, right3 + 8, temp1 + 16, temp2 + 16);
  transpose_s32_8x8_2(left4 + 8, right4 + 8, temp1 + 24, temp2 + 24);

  highbd_cross_input(temp1, temp2, left6, right6);
  highbd_dct8x32_body_second_pass(left6, right6);
  highbd_partial_add_round_shift(left6, right6);

  // Third row of 8x32
  transpose_s32_8x8_2(left1 + 16, right1 + 16, temp1, temp2);
  transpose_s32_8x8_2(left2 + 16, right2 + 16, temp1 + 8, temp2 + 8);
  transpose_s32_8x8_2(left3 + 16, right3 + 16, temp1 + 16, temp2 + 16);
  transpose_s32_8x8_2(left4 + 16, right4 + 16, temp1 + 24, temp2 + 24);

  highbd_cross_input(temp1, temp2, left7, right7);
  highbd_dct8x32_body_second_pass(left7, right7);
  highbd_partial_add_round_shift(left7, right7);

  // Final row of 8x32.
  transpose_s32_8x8_2(left1 + 24, right1 + 24, temp1, temp2);
  transpose_s32_8x8_2(left2 + 24, right2 + 24, temp1 + 8, temp2 + 8);
  transpose_s32_8x8_2(left3 + 24, right3 + 24, temp1 + 16, temp2 + 16);
  transpose_s32_8x8_2(left4 + 24, right4 + 24, temp1 + 24, temp2 + 24);

  highbd_cross_input(temp1, temp2, left8, right8);
  highbd_dct8x32_body_second_pass(left8, right8);
  highbd_partial_add_round_shift(left8, right8);

  // Final transpose
  transpose_s32_8x8_2(left5, right5, left1, right1);
  transpose_s32_8x8_2(left5 + 8, right5 + 8, left2, right2);
  transpose_s32_8x8_2(left5 + 16, right5 + 16, left3, right3);
  transpose_s32_8x8_2(left5 + 24, right5 + 24, left4, right4);
  transpose_s32_8x8_2(left6, right6, left1 + 8, right1 + 8);
  transpose_s32_8x8_2(left6 + 8, right6 + 8, left2 + 8, right2 + 8);
  transpose_s32_8x8_2(left6 + 16, right6 + 16, left3 + 8, right3 + 8);
  transpose_s32_8x8_2(left6 + 24, right6 + 24, left4 + 8, right4 + 8);
  transpose_s32_8x8_2(left7, right7, left1 + 16, right1 + 16);
  transpose_s32_8x8_2(left7 + 8, right7 + 8, left2 + 16, right2 + 16);
  transpose_s32_8x8_2(left7 + 16, right7 + 16, left3 + 16, right3 + 16);
  transpose_s32_8x8_2(left7 + 24, right7 + 24, left4 + 16, right4 + 16);
  transpose_s32_8x8_2(left8, right8, left1 + 24, right1 + 24);
  transpose_s32_8x8_2(left8 + 8, right8 + 8, left2 + 24, right2 + 24);
  transpose_s32_8x8_2(left8 + 16, right8 + 16, left3 + 24, right3 + 24);
  transpose_s32_8x8_2(left8 + 24, right8 + 24, left4 + 24, right4 + 24);

  store32x32_s32(output, left1, right1, left2, right2, left3, right3, left4,
                 right4);
}

void vpx_highbd_fdct32x32_rd_neon(const int16_t *input, tran_low_t *output,
                                  int stride) {
  int16x8_t temp0[32];
  int32x4_t left1[32], left2[32], left3[32], left4[32], right1[32], right2[32],
      right3[32], right4[32];
  int32x4_t left5[32], right5[32], left6[32], right6[32], left7[32], right7[32],
      left8[32], right8[32];
  int32x4_t temp1[32], temp2[32];

  // Process in 8x32 columns.
  load_cross(input, stride, temp0);
  highbd_scale_input(temp0, left1, right1);
  highbd_dct8x32_body_first_pass(left1, right1);
  highbd_partial_sub_round_shift(left1, right1);

  load_cross(input + 8, stride, temp0);
  highbd_scale_input(temp0, left2, right2);
  highbd_dct8x32_body_first_pass(left2, right2);
  highbd_partial_sub_round_shift(left2, right2);

  load_cross(input + 16, stride, temp0);
  highbd_scale_input(temp0, left3, right3);
  highbd_dct8x32_body_first_pass(left3, right3);
  highbd_partial_sub_round_shift(left3, right3);

  load_cross(input + 24, stride, temp0);
  highbd_scale_input(temp0, left4, right4);
  highbd_dct8x32_body_first_pass(left4, right4);
  highbd_partial_sub_round_shift(left4, right4);

  // Generate the top row by munging the first set of 8 from each one together.
  transpose_s32_8x8_2(left1, right1, temp1, temp2);
  transpose_s32_8x8_2(left2, right2, temp1 + 8, temp2 + 8);
  transpose_s32_8x8_2(left3, right3, temp1 + 16, temp2 + 16);
  transpose_s32_8x8_2(left4, right4, temp1 + 24, temp2 + 24);

  highbd_cross_input(temp1, temp2, left5, right5);
  highbd_dct8x32_body_second_pass_rd(left5, right5);

  // Second row of 8x32.
  transpose_s32_8x8_2(left1 + 8, right1 + 8, temp1, temp2);
  transpose_s32_8x8_2(left2 + 8, right2 + 8, temp1 + 8, temp2 + 8);
  transpose_s32_8x8_2(left3 + 8, right3 + 8, temp1 + 16, temp2 + 16);
  transpose_s32_8x8_2(left4 + 8, right4 + 8, temp1 + 24, temp2 + 24);

  highbd_cross_input(temp1, temp2, left6, right6);
  highbd_dct8x32_body_second_pass_rd(left6, right6);

  // Third row of 8x32
  transpose_s32_8x8_2(left1 + 16, right1 + 16, temp1, temp2);
  transpose_s32_8x8_2(left2 + 16, right2 + 16, temp1 + 8, temp2 + 8);
  transpose_s32_8x8_2(left3 + 16, right3 + 16, temp1 + 16, temp2 + 16);
  transpose_s32_8x8_2(left4 + 16, right4 + 16, temp1 + 24, temp2 + 24);

  highbd_cross_input(temp1, temp2, left7, right7);
  highbd_dct8x32_body_second_pass_rd(left7, right7);

  // Final row of 8x32.
  transpose_s32_8x8_2(left1 + 24, right1 + 24, temp1, temp2);
  transpose_s32_8x8_2(left2 + 24, right2 + 24, temp1 + 8, temp2 + 8);
  transpose_s32_8x8_2(left3 + 24, right3 + 24, temp1 + 16, temp2 + 16);
  transpose_s32_8x8_2(left4 + 24, right4 + 24, temp1 + 24, temp2 + 24);

  highbd_cross_input(temp1, temp2, left8, right8);
  highbd_dct8x32_body_second_pass_rd(left8, right8);

  // Final transpose
  transpose_s32_8x8_2(left5, right5, left1, right1);
  transpose_s32_8x8_2(left5 + 8, right5 + 8, left2, right2);
  transpose_s32_8x8_2(left5 + 16, right5 + 16, left3, right3);
  transpose_s32_8x8_2(left5 + 24, right5 + 24, left4, right4);
  transpose_s32_8x8_2(left6, right6, left1 + 8, right1 + 8);
  transpose_s32_8x8_2(left6 + 8, right6 + 8, left2 + 8, right2 + 8);
  transpose_s32_8x8_2(left6 + 16, right6 + 16, left3 + 8, right3 + 8);
  transpose_s32_8x8_2(left6 + 24, right6 + 24, left4 + 8, right4 + 8);
  transpose_s32_8x8_2(left7, right7, left1 + 16, right1 + 16);
  transpose_s32_8x8_2(left7 + 8, right7 + 8, left2 + 16, right2 + 16);
  transpose_s32_8x8_2(left7 + 16, right7 + 16, left3 + 16, right3 + 16);
  transpose_s32_8x8_2(left7 + 24, right7 + 24, left4 + 16, right4 + 16);
  transpose_s32_8x8_2(left8, right8, left1 + 24, right1 + 24);
  transpose_s32_8x8_2(left8 + 8, right8 + 8, left2 + 24, right2 + 24);
  transpose_s32_8x8_2(left8 + 16, right8 + 16, left3 + 24, right3 + 24);
  transpose_s32_8x8_2(left8 + 24, right8 + 24, left4 + 24, right4 + 24);

  store32x32_s32(output, left1, right1, left2, right2, left3, right3, left4,
                 right4);
}

#endif  // CONFIG_VP9_HIGHBITDEPTH

#endif  // !defined(__clang__) && !defined(__ANDROID__) && defined(__GNUC__) &&
        // __GNUC__ == 4 && __GNUC_MINOR__ <= 9
