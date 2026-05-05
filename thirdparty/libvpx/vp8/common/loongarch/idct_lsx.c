/*
 *  Copyright (c) 2022 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vp8_rtcd.h"
#include "vp8/common/blockd.h"
#include "vpx_util/loongson_intrinsics.h"

static const int32_t cospi8sqrt2minus1 = 20091;
static const int32_t sinpi8sqrt2 = 35468;

#define TRANSPOSE8X4_SH_SH(in0, in1, in2, in3, out0, out1, out2, out3)    \
  do {                                                                    \
    __m128i tmp0_m, tmp1_m, tmp2_m, tmp3_m;                               \
                                                                          \
    DUP2_ARG2(__lsx_vilvl_h, in1, in0, in3, in2, tmp0_m, tmp1_m);         \
    DUP2_ARG2(__lsx_vilvh_h, in1, in0, in3, in2, tmp2_m, tmp3_m);         \
    DUP2_ARG2(__lsx_vilvl_w, tmp1_m, tmp0_m, tmp3_m, tmp2_m, out0, out2); \
    DUP2_ARG2(__lsx_vilvh_w, tmp1_m, tmp0_m, tmp3_m, tmp2_m, out1, out3); \
  } while (0)

#define TRANSPOSE_TWO_4x4_H(in0, in1, in2, in3, out0, out1, out2, out3) \
  do {                                                                  \
    __m128i s4_m, s5_m, s6_m, s7_m;                                     \
                                                                        \
    TRANSPOSE8X4_SH_SH(in0, in1, in2, in3, s4_m, s5_m, s6_m, s7_m);     \
    DUP2_ARG2(__lsx_vilvl_d, s6_m, s4_m, s7_m, s5_m, out0, out2);       \
    out1 = __lsx_vilvh_d(s6_m, s4_m);                                   \
    out3 = __lsx_vilvh_d(s7_m, s5_m);                                   \
  } while (0)

#define EXPAND_TO_H_MULTIPLY_SINPI8SQRT2_PCK_TO_W(in0, in1)   \
  do {                                                        \
    __m128i zero_m = __lsx_vldi(0);                           \
    __m128i tmp1_m, tmp2_m;                                   \
    __m128i sinpi8_sqrt2_m = __lsx_vreplgr2vr_w(sinpi8sqrt2); \
                                                              \
    tmp1_m = __lsx_vilvl_h(in0, zero_m);                      \
    tmp2_m = __lsx_vilvh_h(in0, zero_m);                      \
    tmp1_m = __lsx_vsrai_w(tmp1_m, 16);                       \
    tmp2_m = __lsx_vsrai_w(tmp2_m, 16);                       \
    tmp1_m = __lsx_vmul_w(tmp1_m, sinpi8_sqrt2_m);            \
    tmp1_m = __lsx_vsrai_w(tmp1_m, 16);                       \
    tmp2_m = __lsx_vmul_w(tmp2_m, sinpi8_sqrt2_m);            \
    tmp2_m = __lsx_vsrai_w(tmp2_m, 16);                       \
    in1 = __lsx_vpickev_h(tmp2_m, tmp1_m);                    \
  } while (0)

#define VP8_IDCT_1D_H(in0, in1, in2, in3, out0, out1, out2, out3)      \
  do {                                                                 \
    __m128i a1_m, b1_m, c1_m, d1_m;                                    \
    __m128i c_tmp1_m, c_tmp2_m;                                        \
    __m128i d_tmp1_m, d_tmp2_m;                                        \
    __m128i const_cospi8sqrt2minus1_m;                                 \
                                                                       \
    const_cospi8sqrt2minus1_m = __lsx_vreplgr2vr_h(cospi8sqrt2minus1); \
    a1_m = __lsx_vadd_h(in0, in2);                                     \
    b1_m = __lsx_vsub_h(in0, in2);                                     \
    EXPAND_TO_H_MULTIPLY_SINPI8SQRT2_PCK_TO_W(in1, c_tmp1_m);          \
                                                                       \
    c_tmp2_m = __lsx_vmuh_h(in3, const_cospi8sqrt2minus1_m);           \
    c_tmp2_m = __lsx_vslli_h(c_tmp2_m, 1);                             \
    c_tmp2_m = __lsx_vsrai_h(c_tmp2_m, 1);                             \
    c_tmp2_m = __lsx_vadd_h(in3, c_tmp2_m);                            \
    c1_m = __lsx_vsub_h(c_tmp1_m, c_tmp2_m);                           \
                                                                       \
    d_tmp1_m = __lsx_vmuh_h(in1, const_cospi8sqrt2minus1_m);           \
    d_tmp1_m = __lsx_vslli_h(d_tmp1_m, 1);                             \
    d_tmp1_m = __lsx_vsrai_h(d_tmp1_m, 1);                             \
    d_tmp1_m = __lsx_vadd_h(in1, d_tmp1_m);                            \
    EXPAND_TO_H_MULTIPLY_SINPI8SQRT2_PCK_TO_W(in3, d_tmp2_m);          \
    d1_m = __lsx_vadd_h(d_tmp1_m, d_tmp2_m);                           \
    LSX_BUTTERFLY_4_H(a1_m, b1_m, c1_m, d1_m, out0, out1, out2, out3); \
  } while (0)

#define VP8_IDCT_1D_W(in0, in1, in2, in3, out0, out1, out2, out3)      \
  do {                                                                 \
    __m128i a1_m, b1_m, c1_m, d1_m;                                    \
    __m128i c_tmp1_m, c_tmp2_m, d_tmp1_m, d_tmp2_m;                    \
    __m128i const_cospi8sqrt2minus1_m, sinpi8_sqrt2_m;                 \
                                                                       \
    const_cospi8sqrt2minus1_m = __lsx_vreplgr2vr_w(cospi8sqrt2minus1); \
    sinpi8_sqrt2_m = __lsx_vreplgr2vr_w(sinpi8sqrt2);                  \
    a1_m = __lsx_vadd_w(in0, in2);                                     \
    b1_m = __lsx_vsub_w(in0, in2);                                     \
    c_tmp1_m = __lsx_vmul_w(in1, sinpi8_sqrt2_m);                      \
    c_tmp1_m = __lsx_vsrai_w(c_tmp1_m, 16);                            \
    c_tmp2_m = __lsx_vmul_w(in3, const_cospi8sqrt2minus1_m);           \
    c_tmp2_m = __lsx_vsrai_w(c_tmp2_m, 16);                            \
    c_tmp2_m = __lsx_vadd_w(in3, c_tmp2_m);                            \
    c1_m = __lsx_vsub_w(c_tmp1_m, c_tmp2_m);                           \
    d_tmp1_m = __lsx_vmul_w(in1, const_cospi8sqrt2minus1_m);           \
    d_tmp1_m = __lsx_vsrai_w(d_tmp1_m, 16);                            \
    d_tmp1_m = __lsx_vadd_w(in1, d_tmp1_m);                            \
    d_tmp2_m = __lsx_vmul_w(in3, sinpi8_sqrt2_m);                      \
    d_tmp2_m = __lsx_vsrai_w(d_tmp2_m, 16);                            \
    d1_m = __lsx_vadd_w(d_tmp1_m, d_tmp2_m);                           \
    LSX_BUTTERFLY_4_W(a1_m, b1_m, c1_m, d1_m, out0, out1, out2, out3); \
  } while (0)

#define UNPCK_SH_SW(in, out0, out1)  \
  do {                               \
    out0 = __lsx_vsllwil_w_h(in, 0); \
    out1 = __lsx_vexth_w_h(in);      \
  } while (0)

static void idct4x4_addconst_lsx(int16_t in_dc, uint8_t *pred,
                                 int32_t pred_stride, uint8_t *dest,
                                 int32_t dest_stride) {
  __m128i vec, res0, res1, res2, res3, dst0, dst1;
  __m128i pred0, pred1, pred2, pred3;
  __m128i zero = __lsx_vldi(0);

  int32_t pred_stride2 = pred_stride << 1;
  int32_t pred_stride3 = pred_stride2 + pred_stride;

  vec = __lsx_vreplgr2vr_h(in_dc);
  vec = __lsx_vsrari_h(vec, 3);
  pred0 = __lsx_vld(pred, 0);
  DUP2_ARG2(__lsx_vldx, pred, pred_stride, pred, pred_stride2, pred1, pred2);
  pred3 = __lsx_vldx(pred, pred_stride3);
  DUP4_ARG2(__lsx_vilvl_b, zero, pred0, zero, pred1, zero, pred2, zero, pred3,
            res0, res1, res2, res3);
  DUP4_ARG2(__lsx_vadd_h, res0, vec, res1, vec, res2, vec, res3, vec, res0,
            res1, res2, res3);
  res0 = __lsx_vclip255_h(res0);
  res1 = __lsx_vclip255_h(res1);
  res2 = __lsx_vclip255_h(res2);
  res3 = __lsx_vclip255_h(res3);

  DUP2_ARG2(__lsx_vpickev_b, res1, res0, res3, res2, dst0, dst1);
  dst0 = __lsx_vpickev_w(dst1, dst0);
  __lsx_vstelm_w(dst0, dest, 0, 0);
  dest += dest_stride;
  __lsx_vstelm_w(dst0, dest, 0, 1);
  dest += dest_stride;
  __lsx_vstelm_w(dst0, dest, 0, 2);
  dest += dest_stride;
  __lsx_vstelm_w(dst0, dest, 0, 3);
}

void vp8_dc_only_idct_add_lsx(int16_t input_dc, uint8_t *pred_ptr,
                              int32_t pred_stride, uint8_t *dst_ptr,
                              int32_t dst_stride) {
  idct4x4_addconst_lsx(input_dc, pred_ptr, pred_stride, dst_ptr, dst_stride);
}

static void dequant_idct4x4_addblk_2x_lsx(int16_t *input,
                                          int16_t *dequant_input, uint8_t *dest,
                                          int32_t dest_stride) {
  __m128i dest0, dest1, dest2, dest3;
  __m128i in0, in1, in2, in3, mul0, mul1, mul2, mul3, dequant_in0, dequant_in1;
  __m128i hz0, hz1, hz2, hz3, vt0, vt1, vt2, vt3, res0, res1, res2, res3;
  __m128i hz0l, hz1l, hz2l, hz3l, hz0r, hz1r, hz2r, hz3r;
  __m128i vt0l, vt1l, vt2l, vt3l, vt0r, vt1r, vt2r, vt3r;
  __m128i zero = __lsx_vldi(0);

  int32_t dest_stride2 = dest_stride << 1;
  int32_t dest_stride3 = dest_stride2 + dest_stride;

  DUP4_ARG2(__lsx_vld, input, 0, input, 16, input, 32, input, 48, in0, in1, in2,
            in3);
  DUP2_ARG2(__lsx_vld, dequant_input, 0, dequant_input, 16, dequant_in0,
            dequant_in1);

  DUP4_ARG2(__lsx_vmul_h, in0, dequant_in0, in1, dequant_in1, in2, dequant_in0,
            in3, dequant_in1, mul0, mul1, mul2, mul3);
  DUP2_ARG2(__lsx_vpickev_d, mul2, mul0, mul3, mul1, in0, in2);
  DUP2_ARG2(__lsx_vpickod_d, mul2, mul0, mul3, mul1, in1, in3);

  VP8_IDCT_1D_H(in0, in1, in2, in3, hz0, hz1, hz2, hz3);
  TRANSPOSE_TWO_4x4_H(hz0, hz1, hz2, hz3, hz0, hz1, hz2, hz3);
  UNPCK_SH_SW(hz0, hz0r, hz0l);
  UNPCK_SH_SW(hz1, hz1r, hz1l);
  UNPCK_SH_SW(hz2, hz2r, hz2l);
  UNPCK_SH_SW(hz3, hz3r, hz3l);
  VP8_IDCT_1D_W(hz0l, hz1l, hz2l, hz3l, vt0l, vt1l, vt2l, vt3l);
  DUP4_ARG2(__lsx_vsrari_w, vt0l, 3, vt1l, 3, vt2l, 3, vt3l, 3, vt0l, vt1l,
            vt2l, vt3l);
  VP8_IDCT_1D_W(hz0r, hz1r, hz2r, hz3r, vt0r, vt1r, vt2r, vt3r);
  DUP4_ARG2(__lsx_vsrari_w, vt0r, 3, vt1r, 3, vt2r, 3, vt3r, 3, vt0r, vt1r,
            vt2r, vt3r);
  DUP4_ARG2(__lsx_vpickev_h, vt0l, vt0r, vt1l, vt1r, vt2l, vt2r, vt3l, vt3r,
            vt0, vt1, vt2, vt3);
  TRANSPOSE_TWO_4x4_H(vt0, vt1, vt2, vt3, vt0, vt1, vt2, vt3);
  dest0 = __lsx_vld(dest, 0);
  DUP2_ARG2(__lsx_vldx, dest, dest_stride, dest, dest_stride2, dest1, dest2);
  dest3 = __lsx_vldx(dest, dest_stride3);
  DUP4_ARG2(__lsx_vilvl_b, zero, dest0, zero, dest1, zero, dest2, zero, dest3,
            res0, res1, res2, res3);
  DUP4_ARG2(__lsx_vadd_h, res0, vt0, res1, vt1, res2, vt2, res3, vt3, res0,
            res1, res2, res3);

  res0 = __lsx_vclip255_h(res0);
  res1 = __lsx_vclip255_h(res1);
  res2 = __lsx_vclip255_h(res2);
  res3 = __lsx_vclip255_h(res3);
  DUP2_ARG2(__lsx_vpickev_b, res1, res0, res3, res2, vt0l, vt1l);

  __lsx_vstelm_d(vt0l, dest, 0, 0);
  __lsx_vstelm_d(vt0l, dest + dest_stride, 0, 1);
  __lsx_vstelm_d(vt1l, dest + dest_stride2, 0, 0);
  __lsx_vstelm_d(vt1l, dest + dest_stride3, 0, 1);

  __lsx_vst(zero, input, 0);
  __lsx_vst(zero, input, 16);
  __lsx_vst(zero, input, 32);
  __lsx_vst(zero, input, 48);
}

static void dequant_idct_addconst_2x_lsx(int16_t *input, int16_t *dequant_input,
                                         uint8_t *dest, int32_t dest_stride) {
  __m128i input_dc0, input_dc1, vec, res0, res1, res2, res3;
  __m128i dest0, dest1, dest2, dest3;
  __m128i zero = __lsx_vldi(0);
  int32_t dest_stride2 = dest_stride << 1;
  int32_t dest_stride3 = dest_stride2 + dest_stride;

  input_dc0 = __lsx_vreplgr2vr_h(input[0] * dequant_input[0]);
  input_dc1 = __lsx_vreplgr2vr_h(input[16] * dequant_input[0]);
  DUP2_ARG2(__lsx_vsrari_h, input_dc0, 3, input_dc1, 3, input_dc0, input_dc1);
  vec = __lsx_vpickev_d(input_dc1, input_dc0);
  input[0] = 0;
  input[16] = 0;
  dest0 = __lsx_vld(dest, 0);
  DUP2_ARG2(__lsx_vldx, dest, dest_stride, dest, dest_stride2, dest1, dest2);
  dest3 = __lsx_vldx(dest, dest_stride3);
  DUP4_ARG2(__lsx_vilvl_b, zero, dest0, zero, dest1, zero, dest2, zero, dest3,
            res0, res1, res2, res3);
  DUP4_ARG2(__lsx_vadd_h, res0, vec, res1, vec, res2, vec, res3, vec, res0,
            res1, res2, res3);
  res0 = __lsx_vclip255_h(res0);
  res1 = __lsx_vclip255_h(res1);
  res2 = __lsx_vclip255_h(res2);
  res3 = __lsx_vclip255_h(res3);

  DUP2_ARG2(__lsx_vpickev_b, res1, res0, res3, res2, res0, res1);
  __lsx_vstelm_d(res0, dest, 0, 0);
  __lsx_vstelm_d(res0, dest + dest_stride, 0, 1);
  __lsx_vstelm_d(res1, dest + dest_stride2, 0, 0);
  __lsx_vstelm_d(res1, dest + dest_stride3, 0, 1);
}

void vp8_dequant_idct_add_y_block_lsx(int16_t *q, int16_t *dq, uint8_t *dst,
                                      int32_t stride, char *eobs) {
  int16_t *eobs_h = (int16_t *)eobs;
  uint8_t i;

  for (i = 4; i--;) {
    if (eobs_h[0]) {
      if (eobs_h[0] & 0xfefe) {
        dequant_idct4x4_addblk_2x_lsx(q, dq, dst, stride);
      } else {
        dequant_idct_addconst_2x_lsx(q, dq, dst, stride);
      }
    }

    q += 32;

    if (eobs_h[1]) {
      if (eobs_h[1] & 0xfefe) {
        dequant_idct4x4_addblk_2x_lsx(q, dq, dst + 8, stride);
      } else {
        dequant_idct_addconst_2x_lsx(q, dq, dst + 8, stride);
      }
    }

    q += 32;
    dst += (4 * stride);
    eobs_h += 2;
  }
}

void vp8_dequant_idct_add_uv_block_lsx(int16_t *q, int16_t *dq, uint8_t *dst_u,
                                       uint8_t *dst_v, int32_t stride,
                                       char *eobs) {
  int16_t *eobs_h = (int16_t *)eobs;
  if (eobs_h[0]) {
    if (eobs_h[0] & 0xfefe) {
      dequant_idct4x4_addblk_2x_lsx(q, dq, dst_u, stride);
    } else {
      dequant_idct_addconst_2x_lsx(q, dq, dst_u, stride);
    }
  }

  q += 32;
  dst_u += (stride * 4);

  if (eobs_h[1]) {
    if (eobs_h[1] & 0xfefe) {
      dequant_idct4x4_addblk_2x_lsx(q, dq, dst_u, stride);
    } else {
      dequant_idct_addconst_2x_lsx(q, dq, dst_u, stride);
    }
  }

  q += 32;

  if (eobs_h[2]) {
    if (eobs_h[2] & 0xfefe) {
      dequant_idct4x4_addblk_2x_lsx(q, dq, dst_v, stride);
    } else {
      dequant_idct_addconst_2x_lsx(q, dq, dst_v, stride);
    }
  }
  q += 32;
  dst_v += (stride * 4);

  if (eobs_h[3]) {
    if (eobs_h[3] & 0xfefe) {
      dequant_idct4x4_addblk_2x_lsx(q, dq, dst_v, stride);
    } else {
      dequant_idct_addconst_2x_lsx(q, dq, dst_v, stride);
    }
  }
}
