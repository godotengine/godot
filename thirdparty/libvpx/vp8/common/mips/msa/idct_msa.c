/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vp8_rtcd.h"
#include "vp8/common/blockd.h"
#include "vp8/common/mips/msa/vp8_macros_msa.h"

static const int32_t cospi8sqrt2minus1 = 20091;
static const int32_t sinpi8sqrt2 = 35468;

#define TRANSPOSE_TWO_4x4_H(in0, in1, in2, in3, out0, out1, out2, out3) \
  {                                                                     \
    v8i16 s4_m, s5_m, s6_m, s7_m;                                       \
                                                                        \
    TRANSPOSE8X4_SH_SH(in0, in1, in2, in3, s4_m, s5_m, s6_m, s7_m);     \
    ILVR_D2_SH(s6_m, s4_m, s7_m, s5_m, out0, out2);                     \
    out1 = (v8i16)__msa_ilvl_d((v2i64)s6_m, (v2i64)s4_m);               \
    out3 = (v8i16)__msa_ilvl_d((v2i64)s7_m, (v2i64)s5_m);               \
  }

#define EXPAND_TO_H_MULTIPLY_SINPI8SQRT2_PCK_TO_W(in)    \
  ({                                                     \
    v8i16 out_m;                                         \
    v8i16 zero_m = { 0 };                                \
    v4i32 tmp1_m, tmp2_m;                                \
    v4i32 sinpi8_sqrt2_m = __msa_fill_w(sinpi8sqrt2);    \
                                                         \
    ILVRL_H2_SW(in, zero_m, tmp1_m, tmp2_m);             \
    tmp1_m >>= 16;                                       \
    tmp2_m >>= 16;                                       \
    tmp1_m = (tmp1_m * sinpi8_sqrt2_m) >> 16;            \
    tmp2_m = (tmp2_m * sinpi8_sqrt2_m) >> 16;            \
    out_m = __msa_pckev_h((v8i16)tmp2_m, (v8i16)tmp1_m); \
                                                         \
    out_m;                                               \
  })

#define VP8_IDCT_1D_H(in0, in1, in2, in3, out0, out1, out2, out3) \
  {                                                               \
    v8i16 a1_m, b1_m, c1_m, d1_m;                                 \
    v8i16 c_tmp1_m, c_tmp2_m, d_tmp1_m, d_tmp2_m;                 \
    v8i16 const_cospi8sqrt2minus1_m;                              \
                                                                  \
    const_cospi8sqrt2minus1_m = __msa_fill_h(cospi8sqrt2minus1);  \
    a1_m = in0 + in2;                                             \
    b1_m = in0 - in2;                                             \
    c_tmp1_m = EXPAND_TO_H_MULTIPLY_SINPI8SQRT2_PCK_TO_W(in1);    \
    c_tmp2_m = __msa_mul_q_h(in3, const_cospi8sqrt2minus1_m);     \
    c_tmp2_m = c_tmp2_m >> 1;                                     \
    c_tmp2_m = in3 + c_tmp2_m;                                    \
    c1_m = c_tmp1_m - c_tmp2_m;                                   \
    d_tmp1_m = __msa_mul_q_h(in1, const_cospi8sqrt2minus1_m);     \
    d_tmp1_m = d_tmp1_m >> 1;                                     \
    d_tmp1_m = in1 + d_tmp1_m;                                    \
    d_tmp2_m = EXPAND_TO_H_MULTIPLY_SINPI8SQRT2_PCK_TO_W(in3);    \
    d1_m = d_tmp1_m + d_tmp2_m;                                   \
    BUTTERFLY_4(a1_m, b1_m, c1_m, d1_m, out0, out1, out2, out3);  \
  }

#define VP8_IDCT_1D_W(in0, in1, in2, in3, out0, out1, out2, out3) \
  {                                                               \
    v4i32 a1_m, b1_m, c1_m, d1_m;                                 \
    v4i32 c_tmp1_m, c_tmp2_m, d_tmp1_m, d_tmp2_m;                 \
    v4i32 const_cospi8sqrt2minus1_m, sinpi8_sqrt2_m;              \
                                                                  \
    const_cospi8sqrt2minus1_m = __msa_fill_w(cospi8sqrt2minus1);  \
    sinpi8_sqrt2_m = __msa_fill_w(sinpi8sqrt2);                   \
    a1_m = in0 + in2;                                             \
    b1_m = in0 - in2;                                             \
    c_tmp1_m = (in1 * sinpi8_sqrt2_m) >> 16;                      \
    c_tmp2_m = in3 + ((in3 * const_cospi8sqrt2minus1_m) >> 16);   \
    c1_m = c_tmp1_m - c_tmp2_m;                                   \
    d_tmp1_m = in1 + ((in1 * const_cospi8sqrt2minus1_m) >> 16);   \
    d_tmp2_m = (in3 * sinpi8_sqrt2_m) >> 16;                      \
    d1_m = d_tmp1_m + d_tmp2_m;                                   \
    BUTTERFLY_4(a1_m, b1_m, c1_m, d1_m, out0, out1, out2, out3);  \
  }

static void idct4x4_addblk_msa(int16_t *input, uint8_t *pred,
                               int32_t pred_stride, uint8_t *dest,
                               int32_t dest_stride) {
  v8i16 input0, input1;
  v4i32 in0, in1, in2, in3, hz0, hz1, hz2, hz3, vt0, vt1, vt2, vt3;
  v4i32 res0, res1, res2, res3;
  v16i8 zero = { 0 };
  v16i8 pred0, pred1, pred2, pred3;

  LD_SH2(input, 8, input0, input1);
  UNPCK_SH_SW(input0, in0, in1);
  UNPCK_SH_SW(input1, in2, in3);
  VP8_IDCT_1D_W(in0, in1, in2, in3, hz0, hz1, hz2, hz3);
  TRANSPOSE4x4_SW_SW(hz0, hz1, hz2, hz3, hz0, hz1, hz2, hz3);
  VP8_IDCT_1D_W(hz0, hz1, hz2, hz3, vt0, vt1, vt2, vt3);
  SRARI_W4_SW(vt0, vt1, vt2, vt3, 3);
  TRANSPOSE4x4_SW_SW(vt0, vt1, vt2, vt3, vt0, vt1, vt2, vt3);
  LD_SB4(pred, pred_stride, pred0, pred1, pred2, pred3);
  ILVR_B4_SW(zero, pred0, zero, pred1, zero, pred2, zero, pred3, res0, res1,
             res2, res3);
  ILVR_H4_SW(zero, res0, zero, res1, zero, res2, zero, res3, res0, res1, res2,
             res3);
  ADD4(res0, vt0, res1, vt1, res2, vt2, res3, vt3, res0, res1, res2, res3);
  res0 = CLIP_SW_0_255(res0);
  res1 = CLIP_SW_0_255(res1);
  res2 = CLIP_SW_0_255(res2);
  res3 = CLIP_SW_0_255(res3);
  PCKEV_B2_SW(res0, res1, res2, res3, vt0, vt1);
  res0 = (v4i32)__msa_pckev_b((v16i8)vt0, (v16i8)vt1);
  ST4x4_UB(res0, res0, 3, 2, 1, 0, dest, dest_stride);
}

static void idct4x4_addconst_msa(int16_t in_dc, uint8_t *pred,
                                 int32_t pred_stride, uint8_t *dest,
                                 int32_t dest_stride) {
  v8i16 vec, res0, res1, res2, res3, dst0, dst1;
  v16i8 zero = { 0 };
  v16i8 pred0, pred1, pred2, pred3;

  vec = __msa_fill_h(in_dc);
  vec = __msa_srari_h(vec, 3);
  LD_SB4(pred, pred_stride, pred0, pred1, pred2, pred3);
  ILVR_B4_SH(zero, pred0, zero, pred1, zero, pred2, zero, pred3, res0, res1,
             res2, res3);
  ADD4(res0, vec, res1, vec, res2, vec, res3, vec, res0, res1, res2, res3);
  CLIP_SH4_0_255(res0, res1, res2, res3);
  PCKEV_B2_SH(res1, res0, res3, res2, dst0, dst1);
  dst0 = (v8i16)__msa_pckev_w((v4i32)dst1, (v4i32)dst0);
  ST4x4_UB(dst0, dst0, 0, 1, 2, 3, dest, dest_stride);
}

void vp8_short_inv_walsh4x4_msa(int16_t *input, int16_t *mb_dqcoeff) {
  v8i16 input0, input1, tmp0, tmp1, tmp2, tmp3, out0, out1;
  const v8i16 mask0 = { 0, 1, 2, 3, 8, 9, 10, 11 };
  const v8i16 mask1 = { 4, 5, 6, 7, 12, 13, 14, 15 };
  const v8i16 mask2 = { 0, 4, 8, 12, 1, 5, 9, 13 };
  const v8i16 mask3 = { 3, 7, 11, 15, 2, 6, 10, 14 };

  LD_SH2(input, 8, input0, input1);
  input1 = (v8i16)__msa_sldi_b((v16i8)input1, (v16i8)input1, 8);
  tmp0 = input0 + input1;
  tmp1 = input0 - input1;
  VSHF_H2_SH(tmp0, tmp1, tmp0, tmp1, mask0, mask1, tmp2, tmp3);
  out0 = tmp2 + tmp3;
  out1 = tmp2 - tmp3;
  VSHF_H2_SH(out0, out1, out0, out1, mask2, mask3, input0, input1);
  tmp0 = input0 + input1;
  tmp1 = input0 - input1;
  VSHF_H2_SH(tmp0, tmp1, tmp0, tmp1, mask0, mask1, tmp2, tmp3);
  tmp0 = tmp2 + tmp3;
  tmp1 = tmp2 - tmp3;
  ADD2(tmp0, 3, tmp1, 3, out0, out1);
  out0 >>= 3;
  out1 >>= 3;
  mb_dqcoeff[0] = __msa_copy_s_h(out0, 0);
  mb_dqcoeff[16] = __msa_copy_s_h(out0, 4);
  mb_dqcoeff[32] = __msa_copy_s_h(out1, 0);
  mb_dqcoeff[48] = __msa_copy_s_h(out1, 4);
  mb_dqcoeff[64] = __msa_copy_s_h(out0, 1);
  mb_dqcoeff[80] = __msa_copy_s_h(out0, 5);
  mb_dqcoeff[96] = __msa_copy_s_h(out1, 1);
  mb_dqcoeff[112] = __msa_copy_s_h(out1, 5);
  mb_dqcoeff[128] = __msa_copy_s_h(out0, 2);
  mb_dqcoeff[144] = __msa_copy_s_h(out0, 6);
  mb_dqcoeff[160] = __msa_copy_s_h(out1, 2);
  mb_dqcoeff[176] = __msa_copy_s_h(out1, 6);
  mb_dqcoeff[192] = __msa_copy_s_h(out0, 3);
  mb_dqcoeff[208] = __msa_copy_s_h(out0, 7);
  mb_dqcoeff[224] = __msa_copy_s_h(out1, 3);
  mb_dqcoeff[240] = __msa_copy_s_h(out1, 7);
}

static void dequant_idct4x4_addblk_msa(int16_t *input, int16_t *dequant_input,
                                       uint8_t *dest, int32_t dest_stride) {
  v8i16 input0, input1, dequant_in0, dequant_in1, mul0, mul1;
  v8i16 in0, in1, in2, in3, hz0_h, hz1_h, hz2_h, hz3_h;
  v16u8 dest0, dest1, dest2, dest3;
  v4i32 hz0_w, hz1_w, hz2_w, hz3_w, vt0, vt1, vt2, vt3, res0, res1, res2, res3;
  v2i64 zero = { 0 };

  LD_SH2(input, 8, input0, input1);
  LD_SH2(dequant_input, 8, dequant_in0, dequant_in1);
  MUL2(input0, dequant_in0, input1, dequant_in1, mul0, mul1);
  PCKEV_D2_SH(zero, mul0, zero, mul1, in0, in2);
  PCKOD_D2_SH(zero, mul0, zero, mul1, in1, in3);
  VP8_IDCT_1D_H(in0, in1, in2, in3, hz0_h, hz1_h, hz2_h, hz3_h);
  PCKEV_D2_SH(hz1_h, hz0_h, hz3_h, hz2_h, mul0, mul1);
  UNPCK_SH_SW(mul0, hz0_w, hz1_w);
  UNPCK_SH_SW(mul1, hz2_w, hz3_w);
  TRANSPOSE4x4_SW_SW(hz0_w, hz1_w, hz2_w, hz3_w, hz0_w, hz1_w, hz2_w, hz3_w);
  VP8_IDCT_1D_W(hz0_w, hz1_w, hz2_w, hz3_w, vt0, vt1, vt2, vt3);
  SRARI_W4_SW(vt0, vt1, vt2, vt3, 3);
  TRANSPOSE4x4_SW_SW(vt0, vt1, vt2, vt3, vt0, vt1, vt2, vt3);
  LD_UB4(dest, dest_stride, dest0, dest1, dest2, dest3);
  ILVR_B4_SW(zero, dest0, zero, dest1, zero, dest2, zero, dest3, res0, res1,
             res2, res3);
  ILVR_H4_SW(zero, res0, zero, res1, zero, res2, zero, res3, res0, res1, res2,
             res3);
  ADD4(res0, vt0, res1, vt1, res2, vt2, res3, vt3, res0, res1, res2, res3);
  res0 = CLIP_SW_0_255(res0);
  res1 = CLIP_SW_0_255(res1);
  res2 = CLIP_SW_0_255(res2);
  res3 = CLIP_SW_0_255(res3);
  PCKEV_B2_SW(res0, res1, res2, res3, vt0, vt1);
  res0 = (v4i32)__msa_pckev_b((v16i8)vt0, (v16i8)vt1);
  ST4x4_UB(res0, res0, 3, 2, 1, 0, dest, dest_stride);
}

static void dequant_idct4x4_addblk_2x_msa(int16_t *input,
                                          int16_t *dequant_input, uint8_t *dest,
                                          int32_t dest_stride) {
  v16u8 dest0, dest1, dest2, dest3;
  v8i16 in0, in1, in2, in3, mul0, mul1, mul2, mul3, dequant_in0, dequant_in1;
  v8i16 hz0, hz1, hz2, hz3, vt0, vt1, vt2, vt3, res0, res1, res2, res3;
  v4i32 hz0l, hz1l, hz2l, hz3l, hz0r, hz1r, hz2r, hz3r;
  v4i32 vt0l, vt1l, vt2l, vt3l, vt0r, vt1r, vt2r, vt3r;
  v16i8 zero = { 0 };

  LD_SH4(input, 8, in0, in1, in2, in3);
  LD_SH2(dequant_input, 8, dequant_in0, dequant_in1);
  MUL4(in0, dequant_in0, in1, dequant_in1, in2, dequant_in0, in3, dequant_in1,
       mul0, mul1, mul2, mul3);
  PCKEV_D2_SH(mul2, mul0, mul3, mul1, in0, in2);
  PCKOD_D2_SH(mul2, mul0, mul3, mul1, in1, in3);
  VP8_IDCT_1D_H(in0, in1, in2, in3, hz0, hz1, hz2, hz3);
  TRANSPOSE_TWO_4x4_H(hz0, hz1, hz2, hz3, hz0, hz1, hz2, hz3);
  UNPCK_SH_SW(hz0, hz0r, hz0l);
  UNPCK_SH_SW(hz1, hz1r, hz1l);
  UNPCK_SH_SW(hz2, hz2r, hz2l);
  UNPCK_SH_SW(hz3, hz3r, hz3l);
  VP8_IDCT_1D_W(hz0l, hz1l, hz2l, hz3l, vt0l, vt1l, vt2l, vt3l);
  SRARI_W4_SW(vt0l, vt1l, vt2l, vt3l, 3);
  VP8_IDCT_1D_W(hz0r, hz1r, hz2r, hz3r, vt0r, vt1r, vt2r, vt3r);
  SRARI_W4_SW(vt0r, vt1r, vt2r, vt3r, 3);
  PCKEV_H4_SH(vt0l, vt0r, vt1l, vt1r, vt2l, vt2r, vt3l, vt3r, vt0, vt1, vt2,
              vt3);
  TRANSPOSE_TWO_4x4_H(vt0, vt1, vt2, vt3, vt0, vt1, vt2, vt3);
  LD_UB4(dest, dest_stride, dest0, dest1, dest2, dest3);
  ILVR_B4_SH(zero, dest0, zero, dest1, zero, dest2, zero, dest3, res0, res1,
             res2, res3);
  ADD4(res0, vt0, res1, vt1, res2, vt2, res3, vt3, res0, res1, res2, res3);
  CLIP_SH4_0_255(res0, res1, res2, res3);
  PCKEV_B2_SW(res1, res0, res3, res2, vt0l, vt1l);
  ST8x4_UB(vt0l, vt1l, dest, dest_stride);

  __asm__ __volatile__(
      "sw   $zero,    0(%[input])  \n\t"
      "sw   $zero,    4(%[input])  \n\t"
      "sw   $zero,    8(%[input])  \n\t"
      "sw   $zero,   12(%[input])  \n\t"
      "sw   $zero,   16(%[input])  \n\t"
      "sw   $zero,   20(%[input])  \n\t"
      "sw   $zero,   24(%[input])  \n\t"
      "sw   $zero,   28(%[input])  \n\t"
      "sw   $zero,   32(%[input])  \n\t"
      "sw   $zero,   36(%[input])  \n\t"
      "sw   $zero,   40(%[input])  \n\t"
      "sw   $zero,   44(%[input])  \n\t"
      "sw   $zero,   48(%[input])  \n\t"
      "sw   $zero,   52(%[input])  \n\t"
      "sw   $zero,   56(%[input])  \n\t"
      "sw   $zero,   60(%[input])  \n\t" ::

          [input] "r"(input));
}

static void dequant_idct_addconst_2x_msa(int16_t *input, int16_t *dequant_input,
                                         uint8_t *dest, int32_t dest_stride) {
  v8i16 input_dc0, input_dc1, vec, res0, res1, res2, res3;
  v16u8 dest0, dest1, dest2, dest3;
  v16i8 zero = { 0 };

  input_dc0 = __msa_fill_h(input[0] * dequant_input[0]);
  input_dc1 = __msa_fill_h(input[16] * dequant_input[0]);
  SRARI_H2_SH(input_dc0, input_dc1, 3);
  vec = (v8i16)__msa_pckev_d((v2i64)input_dc1, (v2i64)input_dc0);
  input[0] = 0;
  input[16] = 0;
  LD_UB4(dest, dest_stride, dest0, dest1, dest2, dest3);
  ILVR_B4_SH(zero, dest0, zero, dest1, zero, dest2, zero, dest3, res0, res1,
             res2, res3);
  ADD4(res0, vec, res1, vec, res2, vec, res3, vec, res0, res1, res2, res3);
  CLIP_SH4_0_255(res0, res1, res2, res3);
  PCKEV_B2_SH(res1, res0, res3, res2, res0, res1);
  ST8x4_UB(res0, res1, dest, dest_stride);
}

void vp8_short_idct4x4llm_msa(int16_t *input, uint8_t *pred_ptr,
                              int32_t pred_stride, uint8_t *dst_ptr,
                              int32_t dst_stride) {
  idct4x4_addblk_msa(input, pred_ptr, pred_stride, dst_ptr, dst_stride);
}

void vp8_dc_only_idct_add_msa(int16_t input_dc, uint8_t *pred_ptr,
                              int32_t pred_stride, uint8_t *dst_ptr,
                              int32_t dst_stride) {
  idct4x4_addconst_msa(input_dc, pred_ptr, pred_stride, dst_ptr, dst_stride);
}

void vp8_dequantize_b_msa(BLOCKD *d, int16_t *DQC) {
  v8i16 dqc0, dqc1, q0, q1, dq0, dq1;

  LD_SH2(DQC, 8, dqc0, dqc1);
  LD_SH2(d->qcoeff, 8, q0, q1);
  MUL2(dqc0, q0, dqc1, q1, dq0, dq1);
  ST_SH2(dq0, dq1, d->dqcoeff, 8);
}

void vp8_dequant_idct_add_msa(int16_t *input, int16_t *dq, uint8_t *dest,
                              int32_t stride) {
  dequant_idct4x4_addblk_msa(input, dq, dest, stride);

  __asm__ __volatile__(
      "sw     $zero,    0(%[input])     \n\t"
      "sw     $zero,    4(%[input])     \n\t"
      "sw     $zero,    8(%[input])     \n\t"
      "sw     $zero,   12(%[input])     \n\t"
      "sw     $zero,   16(%[input])     \n\t"
      "sw     $zero,   20(%[input])     \n\t"
      "sw     $zero,   24(%[input])     \n\t"
      "sw     $zero,   28(%[input])     \n\t"

      :
      : [input] "r"(input));
}

void vp8_dequant_idct_add_y_block_msa(int16_t *q, int16_t *dq, uint8_t *dst,
                                      int32_t stride, char *eobs) {
  int16_t *eobs_h = (int16_t *)eobs;
  uint8_t i;

  for (i = 4; i--;) {
    if (eobs_h[0]) {
      if (eobs_h[0] & 0xfefe) {
        dequant_idct4x4_addblk_2x_msa(q, dq, dst, stride);
      } else {
        dequant_idct_addconst_2x_msa(q, dq, dst, stride);
      }
    }

    q += 32;

    if (eobs_h[1]) {
      if (eobs_h[1] & 0xfefe) {
        dequant_idct4x4_addblk_2x_msa(q, dq, dst + 8, stride);
      } else {
        dequant_idct_addconst_2x_msa(q, dq, dst + 8, stride);
      }
    }

    q += 32;
    dst += (4 * stride);
    eobs_h += 2;
  }
}

void vp8_dequant_idct_add_uv_block_msa(int16_t *q, int16_t *dq, uint8_t *dst_u,
                                       uint8_t *dst_v, int32_t stride,
                                       char *eobs) {
  int16_t *eobs_h = (int16_t *)eobs;

  if (eobs_h[0]) {
    if (eobs_h[0] & 0xfefe) {
      dequant_idct4x4_addblk_2x_msa(q, dq, dst_u, stride);
    } else {
      dequant_idct_addconst_2x_msa(q, dq, dst_u, stride);
    }
  }

  q += 32;
  dst_u += (stride * 4);

  if (eobs_h[1]) {
    if (eobs_h[1] & 0xfefe) {
      dequant_idct4x4_addblk_2x_msa(q, dq, dst_u, stride);
    } else {
      dequant_idct_addconst_2x_msa(q, dq, dst_u, stride);
    }
  }

  q += 32;

  if (eobs_h[2]) {
    if (eobs_h[2] & 0xfefe) {
      dequant_idct4x4_addblk_2x_msa(q, dq, dst_v, stride);
    } else {
      dequant_idct_addconst_2x_msa(q, dq, dst_v, stride);
    }
  }

  q += 32;
  dst_v += (stride * 4);

  if (eobs_h[3]) {
    if (eobs_h[3] & 0xfefe) {
      dequant_idct4x4_addblk_2x_msa(q, dq, dst_v, stride);
    } else {
      dequant_idct_addconst_2x_msa(q, dq, dst_v, stride);
    }
  }
}
