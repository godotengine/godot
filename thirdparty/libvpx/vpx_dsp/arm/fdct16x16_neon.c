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
#include "vpx_dsp/arm/fdct16x16_neon.h"

// Some builds of gcc 4.9.2 and .3 have trouble with some of the inline
// functions.
#if !defined(__clang__) && !defined(__ANDROID__) && defined(__GNUC__) && \
    __GNUC__ == 4 && __GNUC_MINOR__ == 9 && __GNUC_PATCHLEVEL__ < 4

void vpx_fdct16x16_neon(const int16_t *input, tran_low_t *output, int stride) {
  vpx_fdct16x16_c(input, output, stride);
}

#else

// Main body of fdct16x16.
static void vpx_fdct8x16_body(const int16x8_t *in /*[16]*/,
                              int16x8_t *out /*[16]*/) {
  int16x8_t s[8];
  int16x8_t x[4];
  int16x8_t step[8];

  // stage 1
  // From fwd_txfm.c: Work on the first eight values; fdct8(input,
  // even_results);"
  s[0] = vaddq_s16(in[0], in[7]);
  s[1] = vaddq_s16(in[1], in[6]);
  s[2] = vaddq_s16(in[2], in[5]);
  s[3] = vaddq_s16(in[3], in[4]);
  s[4] = vsubq_s16(in[3], in[4]);
  s[5] = vsubq_s16(in[2], in[5]);
  s[6] = vsubq_s16(in[1], in[6]);
  s[7] = vsubq_s16(in[0], in[7]);

  // fdct4(step, step);
  x[0] = vaddq_s16(s[0], s[3]);
  x[1] = vaddq_s16(s[1], s[2]);
  x[2] = vsubq_s16(s[1], s[2]);
  x[3] = vsubq_s16(s[0], s[3]);

  // out[0] = fdct_round_shift((x0 + x1) * cospi_16_64)
  // out[8] = fdct_round_shift((x0 - x1) * cospi_16_64)
  butterfly_one_coeff_s16_s32_fast_narrow(x[0], x[1], cospi_16_64, &out[0],
                                          &out[8]);
  // out[4]  = fdct_round_shift(x3 * cospi_8_64  + x2 * cospi_24_64);
  // out[12] = fdct_round_shift(x3 * cospi_24_64 - x2 * cospi_8_64);
  butterfly_two_coeff(x[3], x[2], cospi_8_64, cospi_24_64, &out[4], &out[12]);

  //  Stage 2
  // Re-using source s5/s6
  // s5 = fdct_round_shift((s6 - s5) * cospi_16_64)
  // s6 = fdct_round_shift((s6 + s5) * cospi_16_64)
  butterfly_one_coeff_s16_fast(s[6], s[5], cospi_16_64, &s[6], &s[5]);

  //  Stage 3
  x[0] = vaddq_s16(s[4], s[5]);
  x[1] = vsubq_s16(s[4], s[5]);
  x[2] = vsubq_s16(s[7], s[6]);
  x[3] = vaddq_s16(s[7], s[6]);

  // Stage 4
  // out[2]  = fdct_round_shift(x3 * cospi_4_64  + x0 * cospi_28_64)
  // out[14] = fdct_round_shift(x3 * cospi_28_64 - x0 * cospi_4_64)
  butterfly_two_coeff(x[3], x[0], cospi_4_64, cospi_28_64, &out[2], &out[14]);
  // out[6]  = fdct_round_shift(x2 * cospi_20_64 + x1 * cospi_12_64)
  // out[10] = fdct_round_shift(x2 * cospi_12_64 - x1 * cospi_20_64)
  butterfly_two_coeff(x[2], x[1], cospi_20_64, cospi_12_64, &out[10], &out[6]);

  // step 2
  // From fwd_txfm.c: Work on the next eight values; step1 -> odd_results"
  // That file distinguished between "in_high" and "step1" but the only
  // difference is that "in_high" is the first 8 values and "step 1" is the
  // second. Here, since they are all in one array, "step1" values are += 8.

  // step2[2] = fdct_round_shift((step1[5] - step1[2]) * cospi_16_64)
  // step2[3] = fdct_round_shift((step1[4] - step1[3]) * cospi_16_64)
  // step2[4] = fdct_round_shift((step1[4] + step1[3]) * cospi_16_64)
  // step2[5] = fdct_round_shift((step1[5] + step1[2]) * cospi_16_64)
  butterfly_one_coeff_s16_fast(in[13], in[10], cospi_16_64, &s[5], &s[2]);
  butterfly_one_coeff_s16_fast(in[12], in[11], cospi_16_64, &s[4], &s[3]);

  // step 3
  s[0] = vaddq_s16(in[8], s[3]);
  s[1] = vaddq_s16(in[9], s[2]);
  x[0] = vsubq_s16(in[9], s[2]);
  x[1] = vsubq_s16(in[8], s[3]);
  x[2] = vsubq_s16(in[15], s[4]);
  x[3] = vsubq_s16(in[14], s[5]);
  s[6] = vaddq_s16(in[14], s[5]);
  s[7] = vaddq_s16(in[15], s[4]);

  // step 4
  // step2[6] = fdct_round_shift(step3[6] * cospi_8_64  + step3[1] *
  // cospi_24_64) step2[1] = fdct_round_shift(step3[6] * cospi_24_64 - step3[1]
  // * cospi_8_64)
  butterfly_two_coeff(s[6], s[1], cospi_8_64, cospi_24_64, &s[6], &s[1]);

  // step2[2] = fdct_round_shift(step3[2] * cospi_24_64 + step3[5] * cospi_8_64)
  // step2[5] = fdct_round_shift(step3[2] * cospi_8_64  - step3[5] *
  // cospi_24_64)
  butterfly_two_coeff(x[0], x[3], cospi_24_64, cospi_8_64, &s[2], &s[5]);

  // step 5
  step[0] = vaddq_s16(s[0], s[1]);
  step[1] = vsubq_s16(s[0], s[1]);
  step[2] = vaddq_s16(x[1], s[2]);
  step[3] = vsubq_s16(x[1], s[2]);
  step[4] = vsubq_s16(x[2], s[5]);
  step[5] = vaddq_s16(x[2], s[5]);
  step[6] = vsubq_s16(s[7], s[6]);
  step[7] = vaddq_s16(s[7], s[6]);

  // step 6
  // out[9] = fdct_round_shift(step1[6] * cospi_18_64 + step1[1] * cospi_14_64)
  // out[7] = fdct_round_shift(step1[6] * cospi_14_64 - step1[1] * cospi_18_64)
  butterfly_two_coeff(step[6], step[1], cospi_18_64, cospi_14_64, &out[9],
                      &out[7]);
  // out[1]  = fdct_round_shift(step1[7] * cospi_2_64  + step1[0] * cospi_30_64)
  // out[15] = fdct_round_shift(step1[7] * cospi_30_64 - step1[0] * cospi_2_64)
  butterfly_two_coeff(step[7], step[0], cospi_2_64, cospi_30_64, &out[1],
                      &out[15]);

  // out[13] = fdct_round_shift(step1[4] * cospi_26_64 + step1[3] * cospi_6_64)
  // out[3]  = fdct_round_shift(step1[4] * cospi_6_64  - step1[3] * cospi_26_64)
  butterfly_two_coeff(step[4], step[3], cospi_26_64, cospi_6_64, &out[13],
                      &out[3]);

  // out[5]  = fdct_round_shift(step1[5] * cospi_10_64 + step1[2] * cospi_22_64)
  // out[11] = fdct_round_shift(step1[5] * cospi_22_64 - step1[2] * cospi_10_64)
  butterfly_two_coeff(step[5], step[2], cospi_10_64, cospi_22_64, &out[5],
                      &out[11]);
}

void vpx_fdct16x16_neon(const int16_t *input, tran_low_t *output, int stride) {
  int16x8_t temp0[16];
  int16x8_t temp1[16];
  int16x8_t temp2[16];
  int16x8_t temp3[16];

  // Left half.
  load_cross(input, stride, temp0);
  scale_input(temp0, temp1);
  vpx_fdct8x16_body(temp1, temp0);

  // Right half.
  load_cross(input + 8, stride, temp1);
  scale_input(temp1, temp2);
  vpx_fdct8x16_body(temp2, temp1);

  // Transpose top left and top right quarters into one contiguous location to
  // process to the top half.

  transpose_s16_8x8q(&temp0[0], &temp2[0]);
  transpose_s16_8x8q(&temp1[0], &temp2[8]);
  partial_round_shift(temp2);
  cross_input(temp2, temp3);
  vpx_fdct8x16_body(temp3, temp2);
  transpose_s16_8x8(&temp2[0], &temp2[1], &temp2[2], &temp2[3], &temp2[4],
                    &temp2[5], &temp2[6], &temp2[7]);
  transpose_s16_8x8(&temp2[8], &temp2[9], &temp2[10], &temp2[11], &temp2[12],
                    &temp2[13], &temp2[14], &temp2[15]);
  store(output, temp2);
  store(output + 8, temp2 + 8);
  output += 8 * 16;

  // Transpose bottom left and bottom right quarters into one contiguous
  // location to process to the bottom half.
  transpose_s16_8x8q(&temp0[8], &temp1[0]);

  transpose_s16_8x8(&temp1[8], &temp1[9], &temp1[10], &temp1[11], &temp1[12],
                    &temp1[13], &temp1[14], &temp1[15]);
  partial_round_shift(temp1);
  cross_input(temp1, temp0);
  vpx_fdct8x16_body(temp0, temp1);
  transpose_s16_8x8(&temp1[0], &temp1[1], &temp1[2], &temp1[3], &temp1[4],
                    &temp1[5], &temp1[6], &temp1[7]);
  transpose_s16_8x8(&temp1[8], &temp1[9], &temp1[10], &temp1[11], &temp1[12],
                    &temp1[13], &temp1[14], &temp1[15]);
  store(output, temp1);
  store(output + 8, temp1 + 8);
}

#if CONFIG_VP9_HIGHBITDEPTH

// Main body of fdct8x16 column
static void vpx_highbd_fdct8x16_body(int32x4_t *left /*[16]*/,
                                     int32x4_t *right /* [16] */) {
  int32x4_t sl[8];
  int32x4_t sr[8];
  int32x4_t xl[4];
  int32x4_t xr[4];
  int32x4_t inl[8];
  int32x4_t inr[8];
  int32x4_t stepl[8];
  int32x4_t stepr[8];

  // stage 1
  // From fwd_txfm.c: Work on the first eight values; fdct8(input,
  // even_results);"
  sl[0] = vaddq_s32(left[0], left[7]);
  sr[0] = vaddq_s32(right[0], right[7]);
  sl[1] = vaddq_s32(left[1], left[6]);
  sr[1] = vaddq_s32(right[1], right[6]);
  sl[2] = vaddq_s32(left[2], left[5]);
  sr[2] = vaddq_s32(right[2], right[5]);
  sl[3] = vaddq_s32(left[3], left[4]);
  sr[3] = vaddq_s32(right[3], right[4]);
  sl[4] = vsubq_s32(left[3], left[4]);
  sr[4] = vsubq_s32(right[3], right[4]);
  sl[5] = vsubq_s32(left[2], left[5]);
  sr[5] = vsubq_s32(right[2], right[5]);
  sl[6] = vsubq_s32(left[1], left[6]);
  sr[6] = vsubq_s32(right[1], right[6]);
  sl[7] = vsubq_s32(left[0], left[7]);
  sr[7] = vsubq_s32(right[0], right[7]);

  // Copy values 8-15 as we're storing in-place
  inl[0] = left[8];
  inr[0] = right[8];
  inl[1] = left[9];
  inr[1] = right[9];
  inl[2] = left[10];
  inr[2] = right[10];
  inl[3] = left[11];
  inr[3] = right[11];
  inl[4] = left[12];
  inr[4] = right[12];
  inl[5] = left[13];
  inr[5] = right[13];
  inl[6] = left[14];
  inr[6] = right[14];
  inl[7] = left[15];
  inr[7] = right[15];

  // fdct4(step, step);
  xl[0] = vaddq_s32(sl[0], sl[3]);
  xr[0] = vaddq_s32(sr[0], sr[3]);
  xl[1] = vaddq_s32(sl[1], sl[2]);
  xr[1] = vaddq_s32(sr[1], sr[2]);
  xl[2] = vsubq_s32(sl[1], sl[2]);
  xr[2] = vsubq_s32(sr[1], sr[2]);
  xl[3] = vsubq_s32(sl[0], sl[3]);
  xr[3] = vsubq_s32(sr[0], sr[3]);

  // out[0] = fdct_round_shift((x0 + x1) * cospi_16_64)
  // out[8] = fdct_round_shift((x0 - x1) * cospi_16_64)
  butterfly_one_coeff_s32_fast(xl[0], xr[0], xl[1], xr[1], cospi_16_64,
                               &left[0], &right[0], &left[8], &right[8]);

  // out[4]  = fdct_round_shift(x3 * cospi_8_64  + x2 * cospi_24_64);
  // out[12] = fdct_round_shift(x3 * cospi_24_64 - x2 * cospi_8_64);
  butterfly_two_coeff_s32_s64_narrow(xl[3], xr[3], xl[2], xr[2], cospi_8_64,
                                     cospi_24_64, &left[4], &right[4],
                                     &left[12], &right[12]);

  //  Stage 2
  // Re-using source s5/s6
  // s5 = fdct_round_shift((s6 - s5) * cospi_16_64)
  // s6 = fdct_round_shift((s6 + s5) * cospi_16_64)
  butterfly_one_coeff_s32_fast(sl[6], sr[6], sl[5], sr[5], cospi_16_64, &sl[6],
                               &sr[6], &sl[5], &sr[5]);

  //  Stage 3
  xl[0] = vaddq_s32(sl[4], sl[5]);
  xr[0] = vaddq_s32(sr[4], sr[5]);
  xl[1] = vsubq_s32(sl[4], sl[5]);
  xr[1] = vsubq_s32(sr[4], sr[5]);
  xl[2] = vsubq_s32(sl[7], sl[6]);
  xr[2] = vsubq_s32(sr[7], sr[6]);
  xl[3] = vaddq_s32(sl[7], sl[6]);
  xr[3] = vaddq_s32(sr[7], sr[6]);

  // Stage 4
  // out[2]  = fdct_round_shift(x3 * cospi_4_64  + x0 * cospi_28_64)
  // out[14] = fdct_round_shift(x3 * cospi_28_64 - x0 * cospi_4_64)
  butterfly_two_coeff_s32_s64_narrow(xl[3], xr[3], xl[0], xr[0], cospi_4_64,
                                     cospi_28_64, &left[2], &right[2],
                                     &left[14], &right[14]);
  // out[6]  = fdct_round_shift(x2 * cospi_20_64 + x1 * cospi_12_64)
  // out[10] = fdct_round_shift(x2 * cospi_12_64 - x1 * cospi_20_64)
  butterfly_two_coeff_s32_s64_narrow(xl[2], xr[2], xl[1], xr[1], cospi_20_64,
                                     cospi_12_64, &left[10], &right[10],
                                     &left[6], &right[6]);

  // step 2
  // From fwd_txfm.c: Work on the next eight values; step1 -> odd_results"
  // That file distinguished between "in_high" and "step1" but the only
  // difference is that "in_high" is the first 8 values and "step 1" is the
  // second. Here, since they are all in one array, "step1" values are += 8.

  // step2[2] = fdct_round_shift((step1[5] - step1[2]) * cospi_16_64)
  // step2[3] = fdct_round_shift((step1[4] - step1[3]) * cospi_16_64)
  // step2[4] = fdct_round_shift((step1[4] + step1[3]) * cospi_16_64)
  // step2[5] = fdct_round_shift((step1[5] + step1[2]) * cospi_16_64)
  butterfly_one_coeff_s32_fast(inl[5], inr[5], inl[2], inr[2], cospi_16_64,
                               &sl[5], &sr[5], &sl[2], &sr[2]);
  butterfly_one_coeff_s32_fast(inl[4], inr[4], inl[3], inr[3], cospi_16_64,
                               &sl[4], &sr[4], &sl[3], &sr[3]);

  // step 3
  sl[0] = vaddq_s32(inl[0], sl[3]);
  sr[0] = vaddq_s32(inr[0], sr[3]);
  sl[1] = vaddq_s32(inl[1], sl[2]);
  sr[1] = vaddq_s32(inr[1], sr[2]);
  xl[0] = vsubq_s32(inl[1], sl[2]);
  xr[0] = vsubq_s32(inr[1], sr[2]);
  xl[1] = vsubq_s32(inl[0], sl[3]);
  xr[1] = vsubq_s32(inr[0], sr[3]);
  xl[2] = vsubq_s32(inl[7], sl[4]);
  xr[2] = vsubq_s32(inr[7], sr[4]);
  xl[3] = vsubq_s32(inl[6], sl[5]);
  xr[3] = vsubq_s32(inr[6], sr[5]);
  sl[6] = vaddq_s32(inl[6], sl[5]);
  sr[6] = vaddq_s32(inr[6], sr[5]);
  sl[7] = vaddq_s32(inl[7], sl[4]);
  sr[7] = vaddq_s32(inr[7], sr[4]);

  // step 4
  // step2[6] = fdct_round_shift(step3[6] * cospi_8_64  + step3[1] *
  // cospi_24_64) step2[1] = fdct_round_shift(step3[6] * cospi_24_64 - step3[1]
  // * cospi_8_64)
  butterfly_two_coeff_s32_s64_narrow(sl[6], sr[6], sl[1], sr[1], cospi_8_64,
                                     cospi_24_64, &sl[6], &sr[6], &sl[1],
                                     &sr[1]);
  // step2[2] = fdct_round_shift(step3[2] * cospi_24_64 + step3[5] * cospi_8_64)
  // step2[5] = fdct_round_shift(step3[2] * cospi_8_64  - step3[5] *
  // cospi_24_64)
  butterfly_two_coeff_s32_s64_narrow(xl[0], xr[0], xl[3], xr[3], cospi_24_64,
                                     cospi_8_64, &sl[2], &sr[2], &sl[5],
                                     &sr[5]);

  // step 5
  stepl[0] = vaddq_s32(sl[0], sl[1]);
  stepr[0] = vaddq_s32(sr[0], sr[1]);
  stepl[1] = vsubq_s32(sl[0], sl[1]);
  stepr[1] = vsubq_s32(sr[0], sr[1]);
  stepl[2] = vaddq_s32(xl[1], sl[2]);
  stepr[2] = vaddq_s32(xr[1], sr[2]);
  stepl[3] = vsubq_s32(xl[1], sl[2]);
  stepr[3] = vsubq_s32(xr[1], sr[2]);
  stepl[4] = vsubq_s32(xl[2], sl[5]);
  stepr[4] = vsubq_s32(xr[2], sr[5]);
  stepl[5] = vaddq_s32(xl[2], sl[5]);
  stepr[5] = vaddq_s32(xr[2], sr[5]);
  stepl[6] = vsubq_s32(sl[7], sl[6]);
  stepr[6] = vsubq_s32(sr[7], sr[6]);
  stepl[7] = vaddq_s32(sl[7], sl[6]);
  stepr[7] = vaddq_s32(sr[7], sr[6]);

  // step 6
  // out[9] = fdct_round_shift(step1[6] * cospi_18_64 + step1[1] * cospi_14_64)
  // out[7] = fdct_round_shift(step1[6] * cospi_14_64 - step1[1] * cospi_18_64)
  butterfly_two_coeff_s32_s64_narrow(stepl[6], stepr[6], stepl[1], stepr[1],
                                     cospi_18_64, cospi_14_64, &left[9],
                                     &right[9], &left[7], &right[7]);
  // out[1]  = fdct_round_shift(step1[7] * cospi_2_64  + step1[0] * cospi_30_64)
  // out[15] = fdct_round_shift(step1[7] * cospi_30_64 - step1[0] * cospi_2_64)
  butterfly_two_coeff_s32_s64_narrow(stepl[7], stepr[7], stepl[0], stepr[0],
                                     cospi_2_64, cospi_30_64, &left[1],
                                     &right[1], &left[15], &right[15]);
  // out[13] = fdct_round_shift(step1[4] * cospi_26_64 + step1[3] * cospi_6_64)
  // out[3]  = fdct_round_shift(step1[4] * cospi_6_64  - step1[3] * cospi_26_64)
  butterfly_two_coeff_s32_s64_narrow(stepl[4], stepr[4], stepl[3], stepr[3],
                                     cospi_26_64, cospi_6_64, &left[13],
                                     &right[13], &left[3], &right[3]);
  // out[5]  = fdct_round_shift(step1[5] * cospi_10_64 + step1[2] * cospi_22_64)
  // out[11] = fdct_round_shift(step1[5] * cospi_22_64 - step1[2] * cospi_10_64)
  butterfly_two_coeff_s32_s64_narrow(stepl[5], stepr[5], stepl[2], stepr[2],
                                     cospi_10_64, cospi_22_64, &left[5],
                                     &right[5], &left[11], &right[11]);
}

void vpx_highbd_fdct16x16_neon(const int16_t *input, tran_low_t *output,
                               int stride) {
  int16x8_t temp0[16];
  int32x4_t left1[16], left2[16], left3[16], left4[16], right1[16], right2[16],
      right3[16], right4[16];

  // Left half.
  load_cross(input, stride, temp0);
  highbd_scale_input(temp0, left1, right1);
  vpx_highbd_fdct8x16_body(left1, right1);

  // right half.
  load_cross(input + 8, stride, temp0);
  highbd_scale_input(temp0, left2, right2);
  vpx_highbd_fdct8x16_body(left2, right2);

  // Transpose top left and top right quarters into one contiguous location to
  // process to the top half.

  transpose_s32_8x8_2(left1, right1, left3, right3);
  transpose_s32_8x8_2(left2, right2, left3 + 8, right3 + 8);
  transpose_s32_8x8_2(left1 + 8, right1 + 8, left4, right4);
  transpose_s32_8x8_2(left2 + 8, right2 + 8, left4 + 8, right4 + 8);

  highbd_partial_round_shift(left3, right3);
  highbd_cross_input(left3, right3, left1, right1);
  vpx_highbd_fdct8x16_body(left1, right1);

  // Transpose bottom left and bottom right quarters into one contiguous
  // location to process to the bottom half.

  highbd_partial_round_shift(left4, right4);
  highbd_cross_input(left4, right4, left2, right2);
  vpx_highbd_fdct8x16_body(left2, right2);

  transpose_s32_8x8_2(left1, right1, left3, right3);
  transpose_s32_8x8_2(left2, right2, left3 + 8, right3 + 8);
  transpose_s32_8x8_2(left1 + 8, right1 + 8, left4, right4);
  transpose_s32_8x8_2(left2 + 8, right2 + 8, left4 + 8, right4 + 8);
  store16_s32(output, left3);
  output += 4;
  store16_s32(output, right3);
  output += 4;

  store16_s32(output, left4);
  output += 4;
  store16_s32(output, right4);
}
#endif  // CONFIG_VP9_HIGHBITDEPTH

#endif  // !defined(__clang__) && !defined(__ANDROID__) && defined(__GNUC__) &&
        // __GNUC__ == 4 && __GNUC_MINOR__ == 9 && __GNUC_PATCHLEVEL__ < 4
