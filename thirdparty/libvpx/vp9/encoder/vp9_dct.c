/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>
#include <math.h>

#include "./vp9_rtcd.h"
#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"

#include "vp9/common/vp9_blockd.h"
#include "vp9/common/vp9_idct.h"
#include "vpx_dsp/fwd_txfm.h"
#include "vpx_ports/mem.h"

static void fdct4(const tran_low_t *input, tran_low_t *output) {
  tran_high_t step[4];
  tran_high_t temp1, temp2;

  step[0] = input[0] + input[3];
  step[1] = input[1] + input[2];
  step[2] = input[1] - input[2];
  step[3] = input[0] - input[3];

  temp1 = (step[0] + step[1]) * cospi_16_64;
  temp2 = (step[0] - step[1]) * cospi_16_64;
  output[0] = (tran_low_t)fdct_round_shift(temp1);
  output[2] = (tran_low_t)fdct_round_shift(temp2);
  temp1 = step[2] * cospi_24_64 + step[3] * cospi_8_64;
  temp2 = -step[2] * cospi_8_64 + step[3] * cospi_24_64;
  output[1] = (tran_low_t)fdct_round_shift(temp1);
  output[3] = (tran_low_t)fdct_round_shift(temp2);
}

static void fdct8(const tran_low_t *input, tran_low_t *output) {
  tran_high_t s0, s1, s2, s3, s4, s5, s6, s7;  // canbe16
  tran_high_t t0, t1, t2, t3;                  // needs32
  tran_high_t x0, x1, x2, x3;                  // canbe16

  // stage 1
  s0 = input[0] + input[7];
  s1 = input[1] + input[6];
  s2 = input[2] + input[5];
  s3 = input[3] + input[4];
  s4 = input[3] - input[4];
  s5 = input[2] - input[5];
  s6 = input[1] - input[6];
  s7 = input[0] - input[7];

  // fdct4(step, step);
  x0 = s0 + s3;
  x1 = s1 + s2;
  x2 = s1 - s2;
  x3 = s0 - s3;
  t0 = (x0 + x1) * cospi_16_64;
  t1 = (x0 - x1) * cospi_16_64;
  t2 = x2 * cospi_24_64 + x3 * cospi_8_64;
  t3 = -x2 * cospi_8_64 + x3 * cospi_24_64;
  output[0] = (tran_low_t)fdct_round_shift(t0);
  output[2] = (tran_low_t)fdct_round_shift(t2);
  output[4] = (tran_low_t)fdct_round_shift(t1);
  output[6] = (tran_low_t)fdct_round_shift(t3);

  // Stage 2
  t0 = (s6 - s5) * cospi_16_64;
  t1 = (s6 + s5) * cospi_16_64;
  t2 = (tran_low_t)fdct_round_shift(t0);
  t3 = (tran_low_t)fdct_round_shift(t1);

  // Stage 3
  x0 = s4 + t2;
  x1 = s4 - t2;
  x2 = s7 - t3;
  x3 = s7 + t3;

  // Stage 4
  t0 = x0 * cospi_28_64 + x3 * cospi_4_64;
  t1 = x1 * cospi_12_64 + x2 * cospi_20_64;
  t2 = x2 * cospi_12_64 + x1 * -cospi_20_64;
  t3 = x3 * cospi_28_64 + x0 * -cospi_4_64;
  output[1] = (tran_low_t)fdct_round_shift(t0);
  output[3] = (tran_low_t)fdct_round_shift(t2);
  output[5] = (tran_low_t)fdct_round_shift(t1);
  output[7] = (tran_low_t)fdct_round_shift(t3);
}

static void fdct16(const tran_low_t in[16], tran_low_t out[16]) {
  tran_high_t step1[8];      // canbe16
  tran_high_t step2[8];      // canbe16
  tran_high_t step3[8];      // canbe16
  tran_high_t input[8];      // canbe16
  tran_high_t temp1, temp2;  // needs32

  // step 1
  input[0] = in[0] + in[15];
  input[1] = in[1] + in[14];
  input[2] = in[2] + in[13];
  input[3] = in[3] + in[12];
  input[4] = in[4] + in[11];
  input[5] = in[5] + in[10];
  input[6] = in[6] + in[9];
  input[7] = in[7] + in[8];

  step1[0] = in[7] - in[8];
  step1[1] = in[6] - in[9];
  step1[2] = in[5] - in[10];
  step1[3] = in[4] - in[11];
  step1[4] = in[3] - in[12];
  step1[5] = in[2] - in[13];
  step1[6] = in[1] - in[14];
  step1[7] = in[0] - in[15];

  // fdct8(step, step);
  {
    tran_high_t s0, s1, s2, s3, s4, s5, s6, s7;  // canbe16
    tran_high_t t0, t1, t2, t3;                  // needs32
    tran_high_t x0, x1, x2, x3;                  // canbe16

    // stage 1
    s0 = input[0] + input[7];
    s1 = input[1] + input[6];
    s2 = input[2] + input[5];
    s3 = input[3] + input[4];
    s4 = input[3] - input[4];
    s5 = input[2] - input[5];
    s6 = input[1] - input[6];
    s7 = input[0] - input[7];

    // fdct4(step, step);
    x0 = s0 + s3;
    x1 = s1 + s2;
    x2 = s1 - s2;
    x3 = s0 - s3;
    t0 = (x0 + x1) * cospi_16_64;
    t1 = (x0 - x1) * cospi_16_64;
    t2 = x3 * cospi_8_64 + x2 * cospi_24_64;
    t3 = x3 * cospi_24_64 - x2 * cospi_8_64;
    out[0] = (tran_low_t)fdct_round_shift(t0);
    out[4] = (tran_low_t)fdct_round_shift(t2);
    out[8] = (tran_low_t)fdct_round_shift(t1);
    out[12] = (tran_low_t)fdct_round_shift(t3);

    // Stage 2
    t0 = (s6 - s5) * cospi_16_64;
    t1 = (s6 + s5) * cospi_16_64;
    t2 = fdct_round_shift(t0);
    t3 = fdct_round_shift(t1);

    // Stage 3
    x0 = s4 + t2;
    x1 = s4 - t2;
    x2 = s7 - t3;
    x3 = s7 + t3;

    // Stage 4
    t0 = x0 * cospi_28_64 + x3 * cospi_4_64;
    t1 = x1 * cospi_12_64 + x2 * cospi_20_64;
    t2 = x2 * cospi_12_64 + x1 * -cospi_20_64;
    t3 = x3 * cospi_28_64 + x0 * -cospi_4_64;
    out[2] = (tran_low_t)fdct_round_shift(t0);
    out[6] = (tran_low_t)fdct_round_shift(t2);
    out[10] = (tran_low_t)fdct_round_shift(t1);
    out[14] = (tran_low_t)fdct_round_shift(t3);
  }

  // step 2
  temp1 = (step1[5] - step1[2]) * cospi_16_64;
  temp2 = (step1[4] - step1[3]) * cospi_16_64;
  step2[2] = fdct_round_shift(temp1);
  step2[3] = fdct_round_shift(temp2);
  temp1 = (step1[4] + step1[3]) * cospi_16_64;
  temp2 = (step1[5] + step1[2]) * cospi_16_64;
  step2[4] = fdct_round_shift(temp1);
  step2[5] = fdct_round_shift(temp2);

  // step 3
  step3[0] = step1[0] + step2[3];
  step3[1] = step1[1] + step2[2];
  step3[2] = step1[1] - step2[2];
  step3[3] = step1[0] - step2[3];
  step3[4] = step1[7] - step2[4];
  step3[5] = step1[6] - step2[5];
  step3[6] = step1[6] + step2[5];
  step3[7] = step1[7] + step2[4];

  // step 4
  temp1 = step3[1] * -cospi_8_64 + step3[6] * cospi_24_64;
  temp2 = step3[2] * cospi_24_64 + step3[5] * cospi_8_64;
  step2[1] = fdct_round_shift(temp1);
  step2[2] = fdct_round_shift(temp2);
  temp1 = step3[2] * cospi_8_64 - step3[5] * cospi_24_64;
  temp2 = step3[1] * cospi_24_64 + step3[6] * cospi_8_64;
  step2[5] = fdct_round_shift(temp1);
  step2[6] = fdct_round_shift(temp2);

  // step 5
  step1[0] = step3[0] + step2[1];
  step1[1] = step3[0] - step2[1];
  step1[2] = step3[3] + step2[2];
  step1[3] = step3[3] - step2[2];
  step1[4] = step3[4] - step2[5];
  step1[5] = step3[4] + step2[5];
  step1[6] = step3[7] - step2[6];
  step1[7] = step3[7] + step2[6];

  // step 6
  temp1 = step1[0] * cospi_30_64 + step1[7] * cospi_2_64;
  temp2 = step1[1] * cospi_14_64 + step1[6] * cospi_18_64;
  out[1] = (tran_low_t)fdct_round_shift(temp1);
  out[9] = (tran_low_t)fdct_round_shift(temp2);

  temp1 = step1[2] * cospi_22_64 + step1[5] * cospi_10_64;
  temp2 = step1[3] * cospi_6_64 + step1[4] * cospi_26_64;
  out[5] = (tran_low_t)fdct_round_shift(temp1);
  out[13] = (tran_low_t)fdct_round_shift(temp2);

  temp1 = step1[3] * -cospi_26_64 + step1[4] * cospi_6_64;
  temp2 = step1[2] * -cospi_10_64 + step1[5] * cospi_22_64;
  out[3] = (tran_low_t)fdct_round_shift(temp1);
  out[11] = (tran_low_t)fdct_round_shift(temp2);

  temp1 = step1[1] * -cospi_18_64 + step1[6] * cospi_14_64;
  temp2 = step1[0] * -cospi_2_64 + step1[7] * cospi_30_64;
  out[7] = (tran_low_t)fdct_round_shift(temp1);
  out[15] = (tran_low_t)fdct_round_shift(temp2);
}

static void fadst4(const tran_low_t *input, tran_low_t *output) {
  tran_high_t x0, x1, x2, x3;
  tran_high_t s0, s1, s2, s3, s4, s5, s6, s7;

  x0 = input[0];
  x1 = input[1];
  x2 = input[2];
  x3 = input[3];

  if (!(x0 | x1 | x2 | x3)) {
    output[0] = output[1] = output[2] = output[3] = 0;
    return;
  }

  s0 = sinpi_1_9 * x0;
  s1 = sinpi_4_9 * x0;
  s2 = sinpi_2_9 * x1;
  s3 = sinpi_1_9 * x1;
  s4 = sinpi_3_9 * x2;
  s5 = sinpi_4_9 * x3;
  s6 = sinpi_2_9 * x3;
  s7 = x0 + x1 - x3;

  x0 = s0 + s2 + s5;
  x1 = sinpi_3_9 * s7;
  x2 = s1 - s3 + s6;
  x3 = s4;

  s0 = x0 + x3;
  s1 = x1;
  s2 = x2 - x3;
  s3 = x2 - x0 + x3;

  // 1-D transform scaling factor is sqrt(2).
  output[0] = (tran_low_t)fdct_round_shift(s0);
  output[1] = (tran_low_t)fdct_round_shift(s1);
  output[2] = (tran_low_t)fdct_round_shift(s2);
  output[3] = (tran_low_t)fdct_round_shift(s3);
}

static void fadst8(const tran_low_t *input, tran_low_t *output) {
  tran_high_t s0, s1, s2, s3, s4, s5, s6, s7;

  tran_high_t x0 = input[7];
  tran_high_t x1 = input[0];
  tran_high_t x2 = input[5];
  tran_high_t x3 = input[2];
  tran_high_t x4 = input[3];
  tran_high_t x5 = input[4];
  tran_high_t x6 = input[1];
  tran_high_t x7 = input[6];

  // stage 1
  s0 = cospi_2_64 * x0 + cospi_30_64 * x1;
  s1 = cospi_30_64 * x0 - cospi_2_64 * x1;
  s2 = cospi_10_64 * x2 + cospi_22_64 * x3;
  s3 = cospi_22_64 * x2 - cospi_10_64 * x3;
  s4 = cospi_18_64 * x4 + cospi_14_64 * x5;
  s5 = cospi_14_64 * x4 - cospi_18_64 * x5;
  s6 = cospi_26_64 * x6 + cospi_6_64 * x7;
  s7 = cospi_6_64 * x6 - cospi_26_64 * x7;

  x0 = fdct_round_shift(s0 + s4);
  x1 = fdct_round_shift(s1 + s5);
  x2 = fdct_round_shift(s2 + s6);
  x3 = fdct_round_shift(s3 + s7);
  x4 = fdct_round_shift(s0 - s4);
  x5 = fdct_round_shift(s1 - s5);
  x6 = fdct_round_shift(s2 - s6);
  x7 = fdct_round_shift(s3 - s7);

  // stage 2
  s0 = x0;
  s1 = x1;
  s2 = x2;
  s3 = x3;
  s4 = cospi_8_64 * x4 + cospi_24_64 * x5;
  s5 = cospi_24_64 * x4 - cospi_8_64 * x5;
  s6 = -cospi_24_64 * x6 + cospi_8_64 * x7;
  s7 = cospi_8_64 * x6 + cospi_24_64 * x7;

  x0 = s0 + s2;
  x1 = s1 + s3;
  x2 = s0 - s2;
  x3 = s1 - s3;
  x4 = fdct_round_shift(s4 + s6);
  x5 = fdct_round_shift(s5 + s7);
  x6 = fdct_round_shift(s4 - s6);
  x7 = fdct_round_shift(s5 - s7);

  // stage 3
  s2 = cospi_16_64 * (x2 + x3);
  s3 = cospi_16_64 * (x2 - x3);
  s6 = cospi_16_64 * (x6 + x7);
  s7 = cospi_16_64 * (x6 - x7);

  x2 = fdct_round_shift(s2);
  x3 = fdct_round_shift(s3);
  x6 = fdct_round_shift(s6);
  x7 = fdct_round_shift(s7);

  output[0] = (tran_low_t)x0;
  output[1] = (tran_low_t)-x4;
  output[2] = (tran_low_t)x6;
  output[3] = (tran_low_t)-x2;
  output[4] = (tran_low_t)x3;
  output[5] = (tran_low_t)-x7;
  output[6] = (tran_low_t)x5;
  output[7] = (tran_low_t)-x1;
}

static void fadst16(const tran_low_t *input, tran_low_t *output) {
  tran_high_t s0, s1, s2, s3, s4, s5, s6, s7, s8;
  tran_high_t s9, s10, s11, s12, s13, s14, s15;

  tran_high_t x0 = input[15];
  tran_high_t x1 = input[0];
  tran_high_t x2 = input[13];
  tran_high_t x3 = input[2];
  tran_high_t x4 = input[11];
  tran_high_t x5 = input[4];
  tran_high_t x6 = input[9];
  tran_high_t x7 = input[6];
  tran_high_t x8 = input[7];
  tran_high_t x9 = input[8];
  tran_high_t x10 = input[5];
  tran_high_t x11 = input[10];
  tran_high_t x12 = input[3];
  tran_high_t x13 = input[12];
  tran_high_t x14 = input[1];
  tran_high_t x15 = input[14];

  // stage 1
  s0 = x0 * cospi_1_64 + x1 * cospi_31_64;
  s1 = x0 * cospi_31_64 - x1 * cospi_1_64;
  s2 = x2 * cospi_5_64 + x3 * cospi_27_64;
  s3 = x2 * cospi_27_64 - x3 * cospi_5_64;
  s4 = x4 * cospi_9_64 + x5 * cospi_23_64;
  s5 = x4 * cospi_23_64 - x5 * cospi_9_64;
  s6 = x6 * cospi_13_64 + x7 * cospi_19_64;
  s7 = x6 * cospi_19_64 - x7 * cospi_13_64;
  s8 = x8 * cospi_17_64 + x9 * cospi_15_64;
  s9 = x8 * cospi_15_64 - x9 * cospi_17_64;
  s10 = x10 * cospi_21_64 + x11 * cospi_11_64;
  s11 = x10 * cospi_11_64 - x11 * cospi_21_64;
  s12 = x12 * cospi_25_64 + x13 * cospi_7_64;
  s13 = x12 * cospi_7_64 - x13 * cospi_25_64;
  s14 = x14 * cospi_29_64 + x15 * cospi_3_64;
  s15 = x14 * cospi_3_64 - x15 * cospi_29_64;

  x0 = fdct_round_shift(s0 + s8);
  x1 = fdct_round_shift(s1 + s9);
  x2 = fdct_round_shift(s2 + s10);
  x3 = fdct_round_shift(s3 + s11);
  x4 = fdct_round_shift(s4 + s12);
  x5 = fdct_round_shift(s5 + s13);
  x6 = fdct_round_shift(s6 + s14);
  x7 = fdct_round_shift(s7 + s15);
  x8 = fdct_round_shift(s0 - s8);
  x9 = fdct_round_shift(s1 - s9);
  x10 = fdct_round_shift(s2 - s10);
  x11 = fdct_round_shift(s3 - s11);
  x12 = fdct_round_shift(s4 - s12);
  x13 = fdct_round_shift(s5 - s13);
  x14 = fdct_round_shift(s6 - s14);
  x15 = fdct_round_shift(s7 - s15);

  // stage 2
  s0 = x0;
  s1 = x1;
  s2 = x2;
  s3 = x3;
  s4 = x4;
  s5 = x5;
  s6 = x6;
  s7 = x7;
  s8 = x8 * cospi_4_64 + x9 * cospi_28_64;
  s9 = x8 * cospi_28_64 - x9 * cospi_4_64;
  s10 = x10 * cospi_20_64 + x11 * cospi_12_64;
  s11 = x10 * cospi_12_64 - x11 * cospi_20_64;
  s12 = -x12 * cospi_28_64 + x13 * cospi_4_64;
  s13 = x12 * cospi_4_64 + x13 * cospi_28_64;
  s14 = -x14 * cospi_12_64 + x15 * cospi_20_64;
  s15 = x14 * cospi_20_64 + x15 * cospi_12_64;

  x0 = s0 + s4;
  x1 = s1 + s5;
  x2 = s2 + s6;
  x3 = s3 + s7;
  x4 = s0 - s4;
  x5 = s1 - s5;
  x6 = s2 - s6;
  x7 = s3 - s7;
  x8 = fdct_round_shift(s8 + s12);
  x9 = fdct_round_shift(s9 + s13);
  x10 = fdct_round_shift(s10 + s14);
  x11 = fdct_round_shift(s11 + s15);
  x12 = fdct_round_shift(s8 - s12);
  x13 = fdct_round_shift(s9 - s13);
  x14 = fdct_round_shift(s10 - s14);
  x15 = fdct_round_shift(s11 - s15);

  // stage 3
  s0 = x0;
  s1 = x1;
  s2 = x2;
  s3 = x3;
  s4 = x4 * cospi_8_64 + x5 * cospi_24_64;
  s5 = x4 * cospi_24_64 - x5 * cospi_8_64;
  s6 = -x6 * cospi_24_64 + x7 * cospi_8_64;
  s7 = x6 * cospi_8_64 + x7 * cospi_24_64;
  s8 = x8;
  s9 = x9;
  s10 = x10;
  s11 = x11;
  s12 = x12 * cospi_8_64 + x13 * cospi_24_64;
  s13 = x12 * cospi_24_64 - x13 * cospi_8_64;
  s14 = -x14 * cospi_24_64 + x15 * cospi_8_64;
  s15 = x14 * cospi_8_64 + x15 * cospi_24_64;

  x0 = s0 + s2;
  x1 = s1 + s3;
  x2 = s0 - s2;
  x3 = s1 - s3;
  x4 = fdct_round_shift(s4 + s6);
  x5 = fdct_round_shift(s5 + s7);
  x6 = fdct_round_shift(s4 - s6);
  x7 = fdct_round_shift(s5 - s7);
  x8 = s8 + s10;
  x9 = s9 + s11;
  x10 = s8 - s10;
  x11 = s9 - s11;
  x12 = fdct_round_shift(s12 + s14);
  x13 = fdct_round_shift(s13 + s15);
  x14 = fdct_round_shift(s12 - s14);
  x15 = fdct_round_shift(s13 - s15);

  // stage 4
  s2 = (-cospi_16_64) * (x2 + x3);
  s3 = cospi_16_64 * (x2 - x3);
  s6 = cospi_16_64 * (x6 + x7);
  s7 = cospi_16_64 * (-x6 + x7);
  s10 = cospi_16_64 * (x10 + x11);
  s11 = cospi_16_64 * (-x10 + x11);
  s14 = (-cospi_16_64) * (x14 + x15);
  s15 = cospi_16_64 * (x14 - x15);

  x2 = fdct_round_shift(s2);
  x3 = fdct_round_shift(s3);
  x6 = fdct_round_shift(s6);
  x7 = fdct_round_shift(s7);
  x10 = fdct_round_shift(s10);
  x11 = fdct_round_shift(s11);
  x14 = fdct_round_shift(s14);
  x15 = fdct_round_shift(s15);

  output[0] = (tran_low_t)x0;
  output[1] = (tran_low_t)-x8;
  output[2] = (tran_low_t)x12;
  output[3] = (tran_low_t)-x4;
  output[4] = (tran_low_t)x6;
  output[5] = (tran_low_t)x14;
  output[6] = (tran_low_t)x10;
  output[7] = (tran_low_t)x2;
  output[8] = (tran_low_t)x3;
  output[9] = (tran_low_t)x11;
  output[10] = (tran_low_t)x15;
  output[11] = (tran_low_t)x7;
  output[12] = (tran_low_t)x5;
  output[13] = (tran_low_t)-x13;
  output[14] = (tran_low_t)x9;
  output[15] = (tran_low_t)-x1;
}

static const transform_2d FHT_4[] = {
  { fdct4, fdct4 },   // DCT_DCT  = 0
  { fadst4, fdct4 },  // ADST_DCT = 1
  { fdct4, fadst4 },  // DCT_ADST = 2
  { fadst4, fadst4 }  // ADST_ADST = 3
};

static const transform_2d FHT_8[] = {
  { fdct8, fdct8 },   // DCT_DCT  = 0
  { fadst8, fdct8 },  // ADST_DCT = 1
  { fdct8, fadst8 },  // DCT_ADST = 2
  { fadst8, fadst8 }  // ADST_ADST = 3
};

static const transform_2d FHT_16[] = {
  { fdct16, fdct16 },   // DCT_DCT  = 0
  { fadst16, fdct16 },  // ADST_DCT = 1
  { fdct16, fadst16 },  // DCT_ADST = 2
  { fadst16, fadst16 }  // ADST_ADST = 3
};

void vp9_fht4x4_c(const int16_t *input, tran_low_t *output, int stride,
                  int tx_type) {
  if (tx_type == DCT_DCT) {
    vpx_fdct4x4_c(input, output, stride);
  } else {
    tran_low_t out[4 * 4];
    int i, j;
    tran_low_t temp_in[4], temp_out[4];
    const transform_2d ht = FHT_4[tx_type];

    // Columns
    for (i = 0; i < 4; ++i) {
      for (j = 0; j < 4; ++j) temp_in[j] = input[j * stride + i] * 16;
      if (i == 0 && temp_in[0]) temp_in[0] += 1;
      ht.cols(temp_in, temp_out);
      for (j = 0; j < 4; ++j) out[j * 4 + i] = temp_out[j];
    }

    // Rows
    for (i = 0; i < 4; ++i) {
      for (j = 0; j < 4; ++j) temp_in[j] = out[j + i * 4];
      ht.rows(temp_in, temp_out);
      for (j = 0; j < 4; ++j) output[j + i * 4] = (temp_out[j] + 1) >> 2;
    }
  }
}

void vp9_fht8x8_c(const int16_t *input, tran_low_t *output, int stride,
                  int tx_type) {
  if (tx_type == DCT_DCT) {
    vpx_fdct8x8_c(input, output, stride);
  } else {
    tran_low_t out[64];
    int i, j;
    tran_low_t temp_in[8], temp_out[8];
    const transform_2d ht = FHT_8[tx_type];

    // Columns
    for (i = 0; i < 8; ++i) {
      for (j = 0; j < 8; ++j) temp_in[j] = input[j * stride + i] * 4;
      ht.cols(temp_in, temp_out);
      for (j = 0; j < 8; ++j) out[j * 8 + i] = temp_out[j];
    }

    // Rows
    for (i = 0; i < 8; ++i) {
      for (j = 0; j < 8; ++j) temp_in[j] = out[j + i * 8];
      ht.rows(temp_in, temp_out);
      for (j = 0; j < 8; ++j)
        output[j + i * 8] = (temp_out[j] + (temp_out[j] < 0)) >> 1;
    }
  }
}

/* 4-point reversible, orthonormal Walsh-Hadamard in 3.5 adds, 0.5 shifts per
   pixel. */
void vp9_fwht4x4_c(const int16_t *input, tran_low_t *output, int stride) {
  int i;
  tran_high_t a1, b1, c1, d1, e1;
  const int16_t *ip_pass0 = input;
  const tran_low_t *ip = NULL;
  tran_low_t *op = output;

  for (i = 0; i < 4; i++) {
    a1 = ip_pass0[0 * stride];
    b1 = ip_pass0[1 * stride];
    c1 = ip_pass0[2 * stride];
    d1 = ip_pass0[3 * stride];

    a1 += b1;
    d1 = d1 - c1;
    e1 = (a1 - d1) >> 1;
    b1 = e1 - b1;
    c1 = e1 - c1;
    a1 -= c1;
    d1 += b1;
    op[0] = (tran_low_t)a1;
    op[4] = (tran_low_t)c1;
    op[8] = (tran_low_t)d1;
    op[12] = (tran_low_t)b1;

    ip_pass0++;
    op++;
  }
  ip = output;
  op = output;

  for (i = 0; i < 4; i++) {
    a1 = ip[0];
    b1 = ip[1];
    c1 = ip[2];
    d1 = ip[3];

    a1 += b1;
    d1 -= c1;
    e1 = (a1 - d1) >> 1;
    b1 = e1 - b1;
    c1 = e1 - c1;
    a1 -= c1;
    d1 += b1;
    op[0] = (tran_low_t)(a1 * UNIT_QUANT_FACTOR);
    op[1] = (tran_low_t)(c1 * UNIT_QUANT_FACTOR);
    op[2] = (tran_low_t)(d1 * UNIT_QUANT_FACTOR);
    op[3] = (tran_low_t)(b1 * UNIT_QUANT_FACTOR);

    ip += 4;
    op += 4;
  }
}

void vp9_fht16x16_c(const int16_t *input, tran_low_t *output, int stride,
                    int tx_type) {
  if (tx_type == DCT_DCT) {
    vpx_fdct16x16_c(input, output, stride);
  } else {
    tran_low_t out[256];
    int i, j;
    tran_low_t temp_in[16], temp_out[16];
    const transform_2d ht = FHT_16[tx_type];

    // Columns
    for (i = 0; i < 16; ++i) {
      for (j = 0; j < 16; ++j) temp_in[j] = input[j * stride + i] * 4;
      ht.cols(temp_in, temp_out);
      for (j = 0; j < 16; ++j)
        out[j * 16 + i] = (temp_out[j] + 1 + (temp_out[j] < 0)) >> 2;
    }

    // Rows
    for (i = 0; i < 16; ++i) {
      for (j = 0; j < 16; ++j) temp_in[j] = out[j + i * 16];
      ht.rows(temp_in, temp_out);
      for (j = 0; j < 16; ++j) output[j + i * 16] = temp_out[j];
    }
  }
}

#if CONFIG_VP9_HIGHBITDEPTH
void vp9_highbd_fht4x4_c(const int16_t *input, tran_low_t *output, int stride,
                         int tx_type) {
  vp9_fht4x4_c(input, output, stride, tx_type);
}

void vp9_highbd_fht8x8_c(const int16_t *input, tran_low_t *output, int stride,
                         int tx_type) {
  vp9_fht8x8_c(input, output, stride, tx_type);
}

void vp9_highbd_fwht4x4_c(const int16_t *input, tran_low_t *output,
                          int stride) {
  vp9_fwht4x4_c(input, output, stride);
}

void vp9_highbd_fht16x16_c(const int16_t *input, tran_low_t *output, int stride,
                           int tx_type) {
  vp9_fht16x16_c(input, output, stride, tx_type);
}
#endif  // CONFIG_VP9_HIGHBITDEPTH
