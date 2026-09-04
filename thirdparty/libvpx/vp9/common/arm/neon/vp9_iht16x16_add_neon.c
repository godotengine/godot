/*
 *  Copyright (c) 2018 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <arm_neon.h>
#include <assert.h>

#include "./vp9_rtcd.h"
#include "./vpx_config.h"
#include "vp9/common/vp9_common.h"
#include "vp9/common/arm/neon/vp9_iht_neon.h"
#include "vpx_dsp/arm/idct_neon.h"
#include "vpx_dsp/arm/mem_neon.h"
#include "vpx_dsp/arm/transpose_neon.h"

void vpx_iadst16x16_256_add_half1d(const void *const input, int16_t *output,
                                   void *const dest, const int stride,
                                   const int highbd_flag) {
  int16x8_t in[16], out[16];
  const int16x4_t c_1_31_5_27 =
      create_s16x4_neon(cospi_1_64, cospi_31_64, cospi_5_64, cospi_27_64);
  const int16x4_t c_9_23_13_19 =
      create_s16x4_neon(cospi_9_64, cospi_23_64, cospi_13_64, cospi_19_64);
  const int16x4_t c_17_15_21_11 =
      create_s16x4_neon(cospi_17_64, cospi_15_64, cospi_21_64, cospi_11_64);
  const int16x4_t c_25_7_29_3 =
      create_s16x4_neon(cospi_25_64, cospi_7_64, cospi_29_64, cospi_3_64);
  const int16x4_t c_4_28_20_12 =
      create_s16x4_neon(cospi_4_64, cospi_28_64, cospi_20_64, cospi_12_64);
  const int16x4_t c_16_n16_8_24 =
      create_s16x4_neon(cospi_16_64, -cospi_16_64, cospi_8_64, cospi_24_64);
  int16x8_t x[16], t[12];
  int32x4_t s0[2], s1[2], s2[2], s3[2], s4[2], s5[2], s6[2], s7[2];
  int32x4_t s8[2], s9[2], s10[2], s11[2], s12[2], s13[2], s14[2], s15[2];

  // Load input (16x8)
  if (output) {
    const tran_low_t *inputT = (const tran_low_t *)input;
    in[0] = load_tran_low_to_s16q(inputT);
    inputT += 8;
    in[8] = load_tran_low_to_s16q(inputT);
    inputT += 8;
    in[1] = load_tran_low_to_s16q(inputT);
    inputT += 8;
    in[9] = load_tran_low_to_s16q(inputT);
    inputT += 8;
    in[2] = load_tran_low_to_s16q(inputT);
    inputT += 8;
    in[10] = load_tran_low_to_s16q(inputT);
    inputT += 8;
    in[3] = load_tran_low_to_s16q(inputT);
    inputT += 8;
    in[11] = load_tran_low_to_s16q(inputT);
    inputT += 8;
    in[4] = load_tran_low_to_s16q(inputT);
    inputT += 8;
    in[12] = load_tran_low_to_s16q(inputT);
    inputT += 8;
    in[5] = load_tran_low_to_s16q(inputT);
    inputT += 8;
    in[13] = load_tran_low_to_s16q(inputT);
    inputT += 8;
    in[6] = load_tran_low_to_s16q(inputT);
    inputT += 8;
    in[14] = load_tran_low_to_s16q(inputT);
    inputT += 8;
    in[7] = load_tran_low_to_s16q(inputT);
    inputT += 8;
    in[15] = load_tran_low_to_s16q(inputT);
  } else {
    const int16_t *inputT = (const int16_t *)input;
    in[0] = vld1q_s16(inputT);
    inputT += 8;
    in[8] = vld1q_s16(inputT);
    inputT += 8;
    in[1] = vld1q_s16(inputT);
    inputT += 8;
    in[9] = vld1q_s16(inputT);
    inputT += 8;
    in[2] = vld1q_s16(inputT);
    inputT += 8;
    in[10] = vld1q_s16(inputT);
    inputT += 8;
    in[3] = vld1q_s16(inputT);
    inputT += 8;
    in[11] = vld1q_s16(inputT);
    inputT += 8;
    in[4] = vld1q_s16(inputT);
    inputT += 8;
    in[12] = vld1q_s16(inputT);
    inputT += 8;
    in[5] = vld1q_s16(inputT);
    inputT += 8;
    in[13] = vld1q_s16(inputT);
    inputT += 8;
    in[6] = vld1q_s16(inputT);
    inputT += 8;
    in[14] = vld1q_s16(inputT);
    inputT += 8;
    in[7] = vld1q_s16(inputT);
    inputT += 8;
    in[15] = vld1q_s16(inputT);
  }

  // Transpose
  transpose_s16_8x8(&in[0], &in[1], &in[2], &in[3], &in[4], &in[5], &in[6],
                    &in[7]);
  transpose_s16_8x8(&in[8], &in[9], &in[10], &in[11], &in[12], &in[13], &in[14],
                    &in[15]);

  x[0] = in[15];
  x[1] = in[0];
  x[2] = in[13];
  x[3] = in[2];
  x[4] = in[11];
  x[5] = in[4];
  x[6] = in[9];
  x[7] = in[6];
  x[8] = in[7];
  x[9] = in[8];
  x[10] = in[5];
  x[11] = in[10];
  x[12] = in[3];
  x[13] = in[12];
  x[14] = in[1];
  x[15] = in[14];

  // stage 1
  iadst_butterfly_lane_0_1_neon(x[0], x[1], c_1_31_5_27, s0, s1);
  iadst_butterfly_lane_2_3_neon(x[2], x[3], c_1_31_5_27, s2, s3);
  iadst_butterfly_lane_0_1_neon(x[4], x[5], c_9_23_13_19, s4, s5);
  iadst_butterfly_lane_2_3_neon(x[6], x[7], c_9_23_13_19, s6, s7);
  iadst_butterfly_lane_0_1_neon(x[8], x[9], c_17_15_21_11, s8, s9);
  iadst_butterfly_lane_2_3_neon(x[10], x[11], c_17_15_21_11, s10, s11);
  iadst_butterfly_lane_0_1_neon(x[12], x[13], c_25_7_29_3, s12, s13);
  iadst_butterfly_lane_2_3_neon(x[14], x[15], c_25_7_29_3, s14, s15);

  x[0] = add_dct_const_round_shift_low_8(s0, s8);
  x[1] = add_dct_const_round_shift_low_8(s1, s9);
  x[2] = add_dct_const_round_shift_low_8(s2, s10);
  x[3] = add_dct_const_round_shift_low_8(s3, s11);
  x[4] = add_dct_const_round_shift_low_8(s4, s12);
  x[5] = add_dct_const_round_shift_low_8(s5, s13);
  x[6] = add_dct_const_round_shift_low_8(s6, s14);
  x[7] = add_dct_const_round_shift_low_8(s7, s15);
  x[8] = sub_dct_const_round_shift_low_8(s0, s8);
  x[9] = sub_dct_const_round_shift_low_8(s1, s9);
  x[10] = sub_dct_const_round_shift_low_8(s2, s10);
  x[11] = sub_dct_const_round_shift_low_8(s3, s11);
  x[12] = sub_dct_const_round_shift_low_8(s4, s12);
  x[13] = sub_dct_const_round_shift_low_8(s5, s13);
  x[14] = sub_dct_const_round_shift_low_8(s6, s14);
  x[15] = sub_dct_const_round_shift_low_8(s7, s15);

  // stage 2
  t[0] = x[0];
  t[1] = x[1];
  t[2] = x[2];
  t[3] = x[3];
  t[4] = x[4];
  t[5] = x[5];
  t[6] = x[6];
  t[7] = x[7];
  iadst_butterfly_lane_0_1_neon(x[8], x[9], c_4_28_20_12, s8, s9);
  iadst_butterfly_lane_2_3_neon(x[10], x[11], c_4_28_20_12, s10, s11);
  iadst_butterfly_lane_1_0_neon(x[13], x[12], c_4_28_20_12, s13, s12);
  iadst_butterfly_lane_3_2_neon(x[15], x[14], c_4_28_20_12, s15, s14);

  x[0] = vaddq_s16(t[0], t[4]);
  x[1] = vaddq_s16(t[1], t[5]);
  x[2] = vaddq_s16(t[2], t[6]);
  x[3] = vaddq_s16(t[3], t[7]);
  x[4] = vsubq_s16(t[0], t[4]);
  x[5] = vsubq_s16(t[1], t[5]);
  x[6] = vsubq_s16(t[2], t[6]);
  x[7] = vsubq_s16(t[3], t[7]);
  x[8] = add_dct_const_round_shift_low_8(s8, s12);
  x[9] = add_dct_const_round_shift_low_8(s9, s13);
  x[10] = add_dct_const_round_shift_low_8(s10, s14);
  x[11] = add_dct_const_round_shift_low_8(s11, s15);
  x[12] = sub_dct_const_round_shift_low_8(s8, s12);
  x[13] = sub_dct_const_round_shift_low_8(s9, s13);
  x[14] = sub_dct_const_round_shift_low_8(s10, s14);
  x[15] = sub_dct_const_round_shift_low_8(s11, s15);

  // stage 3
  t[0] = x[0];
  t[1] = x[1];
  t[2] = x[2];
  t[3] = x[3];
  iadst_butterfly_lane_2_3_neon(x[4], x[5], c_16_n16_8_24, s4, s5);
  iadst_butterfly_lane_3_2_neon(x[7], x[6], c_16_n16_8_24, s7, s6);
  t[8] = x[8];
  t[9] = x[9];
  t[10] = x[10];
  t[11] = x[11];
  iadst_butterfly_lane_2_3_neon(x[12], x[13], c_16_n16_8_24, s12, s13);
  iadst_butterfly_lane_3_2_neon(x[15], x[14], c_16_n16_8_24, s15, s14);

  x[0] = vaddq_s16(t[0], t[2]);
  x[1] = vaddq_s16(t[1], t[3]);
  x[2] = vsubq_s16(t[0], t[2]);
  x[3] = vsubq_s16(t[1], t[3]);
  x[4] = add_dct_const_round_shift_low_8(s4, s6);
  x[5] = add_dct_const_round_shift_low_8(s5, s7);
  x[6] = sub_dct_const_round_shift_low_8(s4, s6);
  x[7] = sub_dct_const_round_shift_low_8(s5, s7);
  x[8] = vaddq_s16(t[8], t[10]);
  x[9] = vaddq_s16(t[9], t[11]);
  x[10] = vsubq_s16(t[8], t[10]);
  x[11] = vsubq_s16(t[9], t[11]);
  x[12] = add_dct_const_round_shift_low_8(s12, s14);
  x[13] = add_dct_const_round_shift_low_8(s13, s15);
  x[14] = sub_dct_const_round_shift_low_8(s12, s14);
  x[15] = sub_dct_const_round_shift_low_8(s13, s15);

  // stage 4
  iadst_half_butterfly_neg_neon(&x[3], &x[2], c_16_n16_8_24);
  iadst_half_butterfly_pos_neon(&x[7], &x[6], c_16_n16_8_24);
  iadst_half_butterfly_pos_neon(&x[11], &x[10], c_16_n16_8_24);
  iadst_half_butterfly_neg_neon(&x[15], &x[14], c_16_n16_8_24);

  out[0] = x[0];
  out[1] = vnegq_s16(x[8]);
  out[2] = x[12];
  out[3] = vnegq_s16(x[4]);
  out[4] = x[6];
  out[5] = x[14];
  out[6] = x[10];
  out[7] = x[2];
  out[8] = x[3];
  out[9] = x[11];
  out[10] = x[15];
  out[11] = x[7];
  out[12] = x[5];
  out[13] = vnegq_s16(x[13]);
  out[14] = x[9];
  out[15] = vnegq_s16(x[1]);

  if (output) {
    idct16x16_store_pass1(out, output);
  } else {
    if (highbd_flag) {
      idct16x16_add_store_bd8(out, dest, stride);
    } else {
      idct16x16_add_store(out, dest, stride);
    }
  }
}

void vp9_iht16x16_256_add_neon(const tran_low_t *input, uint8_t *dest,
                               int stride, int tx_type) {
  static const iht_2d IHT_16[] = {
    { vpx_idct16x16_256_add_half1d,
      vpx_idct16x16_256_add_half1d },  // DCT_DCT  = 0
    { vpx_iadst16x16_256_add_half1d,
      vpx_idct16x16_256_add_half1d },  // ADST_DCT = 1
    { vpx_idct16x16_256_add_half1d,
      vpx_iadst16x16_256_add_half1d },  // DCT_ADST = 2
    { vpx_iadst16x16_256_add_half1d,
      vpx_iadst16x16_256_add_half1d }  // ADST_ADST = 3
  };
  const iht_2d ht = IHT_16[tx_type];
  int16_t row_output[16 * 16];

  // pass 1
  ht.rows(input, row_output, dest, stride, 0);               // upper 8 rows
  ht.rows(input + 8 * 16, row_output + 8, dest, stride, 0);  // lower 8 rows

  // pass 2
  ht.cols(row_output, NULL, dest, stride, 0);               // left 8 columns
  ht.cols(row_output + 16 * 8, NULL, dest + 8, stride, 0);  // right 8 columns
}
