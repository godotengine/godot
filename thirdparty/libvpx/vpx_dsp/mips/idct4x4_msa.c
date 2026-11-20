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
#include "vpx_dsp/mips/inv_txfm_msa.h"

void vpx_iwht4x4_16_add_msa(const int16_t *input, uint8_t *dst,
                            int32_t dst_stride) {
  v8i16 in0, in1, in2, in3;
  v4i32 in0_r, in1_r, in2_r, in3_r, in4_r;

  /* load vector elements of 4x4 block */
  LD4x4_SH(input, in0, in2, in3, in1);
  TRANSPOSE4x4_SH_SH(in0, in2, in3, in1, in0, in2, in3, in1);
  UNPCK_R_SH_SW(in0, in0_r);
  UNPCK_R_SH_SW(in2, in2_r);
  UNPCK_R_SH_SW(in3, in3_r);
  UNPCK_R_SH_SW(in1, in1_r);
  SRA_4V(in0_r, in1_r, in2_r, in3_r, UNIT_QUANT_SHIFT);

  in0_r += in2_r;
  in3_r -= in1_r;
  in4_r = (in0_r - in3_r) >> 1;
  in1_r = in4_r - in1_r;
  in2_r = in4_r - in2_r;
  in0_r -= in1_r;
  in3_r += in2_r;

  TRANSPOSE4x4_SW_SW(in0_r, in1_r, in2_r, in3_r, in0_r, in1_r, in2_r, in3_r);

  in0_r += in1_r;
  in2_r -= in3_r;
  in4_r = (in0_r - in2_r) >> 1;
  in3_r = in4_r - in3_r;
  in1_r = in4_r - in1_r;
  in0_r -= in3_r;
  in2_r += in1_r;

  PCKEV_H4_SH(in0_r, in0_r, in1_r, in1_r, in2_r, in2_r, in3_r, in3_r, in0, in1,
              in2, in3);
  ADDBLK_ST4x4_UB(in0, in3, in1, in2, dst, dst_stride);
}

void vpx_iwht4x4_1_add_msa(const int16_t *input, uint8_t *dst,
                           int32_t dst_stride) {
  int16_t a1, e1;
  v8i16 in1, in0 = { 0 };

  a1 = input[0] >> UNIT_QUANT_SHIFT;
  e1 = a1 >> 1;
  a1 -= e1;

  in0 = __msa_insert_h(in0, 0, a1);
  in0 = __msa_insert_h(in0, 1, e1);
  in0 = __msa_insert_h(in0, 2, e1);
  in0 = __msa_insert_h(in0, 3, e1);

  in1 = in0 >> 1;
  in0 -= in1;

  ADDBLK_ST4x4_UB(in0, in1, in1, in1, dst, dst_stride);
}

void vpx_idct4x4_16_add_msa(const int16_t *input, uint8_t *dst,
                            int32_t dst_stride) {
  v8i16 in0, in1, in2, in3;

  /* load vector elements of 4x4 block */
  LD4x4_SH(input, in0, in1, in2, in3);
  /* rows */
  TRANSPOSE4x4_SH_SH(in0, in1, in2, in3, in0, in1, in2, in3);
  VP9_IDCT4x4(in0, in1, in2, in3, in0, in1, in2, in3);
  /* columns */
  TRANSPOSE4x4_SH_SH(in0, in1, in2, in3, in0, in1, in2, in3);
  VP9_IDCT4x4(in0, in1, in2, in3, in0, in1, in2, in3);
  /* rounding (add 2^3, divide by 2^4) */
  SRARI_H4_SH(in0, in1, in2, in3, 4);
  ADDBLK_ST4x4_UB(in0, in1, in2, in3, dst, dst_stride);
}

void vpx_idct4x4_1_add_msa(const int16_t *input, uint8_t *dst,
                           int32_t dst_stride) {
  int16_t out;
  v8i16 vec;

  out = ROUND_POWER_OF_TWO((input[0] * cospi_16_64), DCT_CONST_BITS);
  out = ROUND_POWER_OF_TWO((out * cospi_16_64), DCT_CONST_BITS);
  out = ROUND_POWER_OF_TWO(out, 4);
  vec = __msa_fill_h(out);

  ADDBLK_ST4x4_UB(vec, vec, vec, vec, dst, dst_stride);
}
