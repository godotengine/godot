// Copyright 2016 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// MSA version of dsp functions
//
// Author(s):  Prashant Patil   (prashant.patil@imgtec.com)


#include "./dsp.h"

#if defined(WEBP_USE_MSA)

#include "./msa_macro.h"

//------------------------------------------------------------------------------
// Transforms

#define IDCT_1D_W(in0, in1, in2, in3, out0, out1, out2, out3) {  \
  v4i32 a1_m, b1_m, c1_m, d1_m;                                  \
  v4i32 c_tmp1_m, c_tmp2_m, d_tmp1_m, d_tmp2_m;                  \
  const v4i32 cospi8sqrt2minus1 = __msa_fill_w(20091);           \
  const v4i32 sinpi8sqrt2 = __msa_fill_w(35468);                 \
                                                                 \
  a1_m = in0 + in2;                                              \
  b1_m = in0 - in2;                                              \
  c_tmp1_m = (in1 * sinpi8sqrt2) >> 16;                          \
  c_tmp2_m = in3 + ((in3 * cospi8sqrt2minus1) >> 16);            \
  c1_m = c_tmp1_m - c_tmp2_m;                                    \
  d_tmp1_m = in1 + ((in1 * cospi8sqrt2minus1) >> 16);            \
  d_tmp2_m = (in3 * sinpi8sqrt2) >> 16;                          \
  d1_m = d_tmp1_m + d_tmp2_m;                                    \
  BUTTERFLY_4(a1_m, b1_m, c1_m, d1_m, out0, out1, out2, out3);   \
}
#define MULT1(a) ((((a) * 20091) >> 16) + (a))
#define MULT2(a) (((a) * 35468) >> 16)

static void TransformOne(const int16_t* in, uint8_t* dst) {
  v8i16 input0, input1;
  v4i32 in0, in1, in2, in3, hz0, hz1, hz2, hz3, vt0, vt1, vt2, vt3;
  v4i32 res0, res1, res2, res3;
  const v16i8 zero = { 0 };
  v16i8 dest0, dest1, dest2, dest3;

  LD_SH2(in, 8, input0, input1);
  UNPCK_SH_SW(input0, in0, in1);
  UNPCK_SH_SW(input1, in2, in3);
  IDCT_1D_W(in0, in1, in2, in3, hz0, hz1, hz2, hz3);
  TRANSPOSE4x4_SW_SW(hz0, hz1, hz2, hz3, hz0, hz1, hz2, hz3);
  IDCT_1D_W(hz0, hz1, hz2, hz3, vt0, vt1, vt2, vt3);
  SRARI_W4_SW(vt0, vt1, vt2, vt3, 3);
  TRANSPOSE4x4_SW_SW(vt0, vt1, vt2, vt3, vt0, vt1, vt2, vt3);
  LD_SB4(dst, BPS, dest0, dest1, dest2, dest3);
  ILVR_B4_SW(zero, dest0, zero, dest1, zero, dest2, zero, dest3,
             res0, res1, res2, res3);
  ILVR_H4_SW(zero, res0, zero, res1, zero, res2, zero, res3,
             res0, res1, res2, res3);
  ADD4(res0, vt0, res1, vt1, res2, vt2, res3, vt3, res0, res1, res2, res3);
  CLIP_SW4_0_255(res0, res1, res2, res3);
  PCKEV_B2_SW(res0, res1, res2, res3, vt0, vt1);
  res0 = (v4i32)__msa_pckev_b((v16i8)vt0, (v16i8)vt1);
  ST4x4_UB(res0, res0, 3, 2, 1, 0, dst, BPS);
}

static void TransformTwo(const int16_t* in, uint8_t* dst, int do_two) {
  TransformOne(in, dst);
  if (do_two) {
    TransformOne(in + 16, dst + 4);
  }
}

static void TransformWHT(const int16_t* in, int16_t* out) {
  v8i16 input0, input1;
  const v8i16 mask0 = { 0, 1, 2, 3, 8, 9, 10, 11 };
  const v8i16 mask1 = { 4, 5, 6, 7, 12, 13, 14, 15 };
  const v8i16 mask2 = { 0, 4, 8, 12, 1, 5, 9, 13 };
  const v8i16 mask3 = { 3, 7, 11, 15, 2, 6, 10, 14 };
  v8i16 tmp0, tmp1, tmp2, tmp3;
  v8i16 out0, out1;

  LD_SH2(in, 8, input0, input1);
  input1 = SLDI_SH(input1, input1, 8);
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
  ADDVI_H2_SH(tmp0, 3, tmp1, 3, out0, out1);
  SRAI_H2_SH(out0, out1, 3);
  out[0] = __msa_copy_s_h(out0, 0);
  out[16] = __msa_copy_s_h(out0, 4);
  out[32] = __msa_copy_s_h(out1, 0);
  out[48] = __msa_copy_s_h(out1, 4);
  out[64] = __msa_copy_s_h(out0, 1);
  out[80] = __msa_copy_s_h(out0, 5);
  out[96] = __msa_copy_s_h(out1, 1);
  out[112] = __msa_copy_s_h(out1, 5);
  out[128] = __msa_copy_s_h(out0, 2);
  out[144] = __msa_copy_s_h(out0, 6);
  out[160] = __msa_copy_s_h(out1, 2);
  out[176] = __msa_copy_s_h(out1, 6);
  out[192] = __msa_copy_s_h(out0, 3);
  out[208] = __msa_copy_s_h(out0, 7);
  out[224] = __msa_copy_s_h(out1, 3);
  out[240] = __msa_copy_s_h(out1, 7);
}

static void TransformDC(const int16_t* in, uint8_t* dst) {
  const int DC = (in[0] + 4) >> 3;
  const v8i16 tmp0 = __msa_fill_h(DC);
  ADDBLK_ST4x4_UB(tmp0, tmp0, tmp0, tmp0, dst, BPS);
}

static void TransformAC3(const int16_t* in, uint8_t* dst) {
  const int a = in[0] + 4;
  const int c4 = MULT2(in[4]);
  const int d4 = MULT1(in[4]);
  const int in2 = MULT2(in[1]);
  const int in3 = MULT1(in[1]);
  v4i32 tmp0 = { 0 };
  v4i32 out0 = __msa_fill_w(a + d4);
  v4i32 out1 = __msa_fill_w(a + c4);
  v4i32 out2 = __msa_fill_w(a - c4);
  v4i32 out3 = __msa_fill_w(a - d4);
  v4i32 res0, res1, res2, res3;
  const v4i32 zero = { 0 };
  v16u8 dest0, dest1, dest2, dest3;

  INSERT_W4_SW(in3, in2, -in2, -in3, tmp0);
  ADD4(out0, tmp0, out1, tmp0, out2, tmp0, out3, tmp0,
       out0, out1, out2, out3);
  SRAI_W4_SW(out0, out1, out2, out3, 3);
  LD_UB4(dst, BPS, dest0, dest1, dest2, dest3);
  ILVR_B4_SW(zero, dest0, zero, dest1, zero, dest2, zero, dest3,
             res0, res1, res2, res3);
  ILVR_H4_SW(zero, res0, zero, res1, zero, res2, zero, res3,
             res0, res1, res2, res3);
  ADD4(res0, out0, res1, out1, res2, out2, res3, out3, res0, res1, res2, res3);
  CLIP_SW4_0_255(res0, res1, res2, res3);
  PCKEV_B2_SW(res0, res1, res2, res3, out0, out1);
  res0 = (v4i32)__msa_pckev_b((v16i8)out0, (v16i8)out1);
  ST4x4_UB(res0, res0, 3, 2, 1, 0, dst, BPS);
}

//------------------------------------------------------------------------------
// Entry point

extern void VP8DspInitMSA(void);

WEBP_TSAN_IGNORE_FUNCTION void VP8DspInitMSA(void) {
  VP8TransformWHT = TransformWHT;
  VP8Transform = TransformTwo;
  VP8TransformDC = TransformDC;
  VP8TransformAC3 = TransformAC3;
}

#else  // !WEBP_USE_MSA

WEBP_DSP_INIT_STUB(VP8DspInitMSA)

#endif  // WEBP_USE_MSA
