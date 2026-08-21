/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_MIPS_INV_TXFM_MSA_H_
#define VPX_VPX_DSP_MIPS_INV_TXFM_MSA_H_

#include "vpx_dsp/mips/macros_msa.h"
#include "vpx_dsp/mips/txfm_macros_msa.h"
#include "vpx_dsp/txfm_common.h"

#define VP9_ADST8(in0, in1, in2, in3, in4, in5, in6, in7, out0, out1, out2,  \
                  out3, out4, out5, out6, out7)                              \
  {                                                                          \
    v8i16 cnst0_m, cnst1_m, cnst2_m, cnst3_m, cnst4_m;                       \
    v8i16 vec0_m, vec1_m, vec2_m, vec3_m, s0_m, s1_m;                        \
    v8i16 coeff0_m = { cospi_2_64,  cospi_6_64,  cospi_10_64, cospi_14_64,   \
                       cospi_18_64, cospi_22_64, cospi_26_64, cospi_30_64 }; \
    v8i16 coeff1_m = { cospi_8_64,  -cospi_8_64,  cospi_16_64, -cospi_16_64, \
                       cospi_24_64, -cospi_24_64, 0,           0 };          \
                                                                             \
    SPLATI_H2_SH(coeff0_m, 0, 7, cnst0_m, cnst1_m);                          \
    cnst2_m = -cnst0_m;                                                      \
    ILVEV_H2_SH(cnst0_m, cnst1_m, cnst1_m, cnst2_m, cnst0_m, cnst1_m);       \
    SPLATI_H2_SH(coeff0_m, 4, 3, cnst2_m, cnst3_m);                          \
    cnst4_m = -cnst2_m;                                                      \
    ILVEV_H2_SH(cnst2_m, cnst3_m, cnst3_m, cnst4_m, cnst2_m, cnst3_m);       \
                                                                             \
    ILVRL_H2_SH(in0, in7, vec1_m, vec0_m);                                   \
    ILVRL_H2_SH(in4, in3, vec3_m, vec2_m);                                   \
    DOT_ADD_SUB_SRARI_PCK(vec0_m, vec1_m, vec2_m, vec3_m, cnst0_m, cnst1_m,  \
                          cnst2_m, cnst3_m, in7, in0, in4, in3);             \
                                                                             \
    SPLATI_H2_SH(coeff0_m, 2, 5, cnst0_m, cnst1_m);                          \
    cnst2_m = -cnst0_m;                                                      \
    ILVEV_H2_SH(cnst0_m, cnst1_m, cnst1_m, cnst2_m, cnst0_m, cnst1_m);       \
    SPLATI_H2_SH(coeff0_m, 6, 1, cnst2_m, cnst3_m);                          \
    cnst4_m = -cnst2_m;                                                      \
    ILVEV_H2_SH(cnst2_m, cnst3_m, cnst3_m, cnst4_m, cnst2_m, cnst3_m);       \
                                                                             \
    ILVRL_H2_SH(in2, in5, vec1_m, vec0_m);                                   \
    ILVRL_H2_SH(in6, in1, vec3_m, vec2_m);                                   \
                                                                             \
    DOT_ADD_SUB_SRARI_PCK(vec0_m, vec1_m, vec2_m, vec3_m, cnst0_m, cnst1_m,  \
                          cnst2_m, cnst3_m, in5, in2, in6, in1);             \
    BUTTERFLY_4(in7, in0, in2, in5, s1_m, s0_m, in2, in5);                   \
    out7 = -s0_m;                                                            \
    out0 = s1_m;                                                             \
                                                                             \
    SPLATI_H4_SH(coeff1_m, 0, 4, 1, 5, cnst0_m, cnst1_m, cnst2_m, cnst3_m);  \
                                                                             \
    ILVEV_H2_SH(cnst3_m, cnst0_m, cnst1_m, cnst2_m, cnst3_m, cnst2_m);       \
    cnst0_m = __msa_ilvev_h(cnst1_m, cnst0_m);                               \
    cnst1_m = cnst0_m;                                                       \
                                                                             \
    ILVRL_H2_SH(in4, in3, vec1_m, vec0_m);                                   \
    ILVRL_H2_SH(in6, in1, vec3_m, vec2_m);                                   \
    DOT_ADD_SUB_SRARI_PCK(vec0_m, vec1_m, vec2_m, vec3_m, cnst0_m, cnst2_m,  \
                          cnst3_m, cnst1_m, out1, out6, s0_m, s1_m);         \
                                                                             \
    SPLATI_H2_SH(coeff1_m, 2, 3, cnst0_m, cnst1_m);                          \
    cnst1_m = __msa_ilvev_h(cnst1_m, cnst0_m);                               \
                                                                             \
    ILVRL_H2_SH(in2, in5, vec1_m, vec0_m);                                   \
    ILVRL_H2_SH(s0_m, s1_m, vec3_m, vec2_m);                                 \
    out3 = DOT_SHIFT_RIGHT_PCK_H(vec0_m, vec1_m, cnst0_m);                   \
    out4 = DOT_SHIFT_RIGHT_PCK_H(vec0_m, vec1_m, cnst1_m);                   \
    out2 = DOT_SHIFT_RIGHT_PCK_H(vec2_m, vec3_m, cnst0_m);                   \
    out5 = DOT_SHIFT_RIGHT_PCK_H(vec2_m, vec3_m, cnst1_m);                   \
                                                                             \
    out1 = -out1;                                                            \
    out3 = -out3;                                                            \
    out5 = -out5;                                                            \
  }

#define VP9_SET_COSPI_PAIR(c0_h, c1_h)  \
  ({                                    \
    v8i16 out0_m, r0_m, r1_m;           \
                                        \
    r0_m = __msa_fill_h(c0_h);          \
    r1_m = __msa_fill_h(c1_h);          \
    out0_m = __msa_ilvev_h(r1_m, r0_m); \
                                        \
    out0_m;                             \
  })

#define VP9_ADDBLK_ST8x4_UB(dst, dst_stride, in0, in1, in2, in3)               \
  {                                                                            \
    uint8_t *dst_m = (uint8_t *)(dst);                                         \
    v16u8 dst0_m, dst1_m, dst2_m, dst3_m;                                      \
    v16i8 tmp0_m, tmp1_m;                                                      \
    v16i8 zero_m = { 0 };                                                      \
    v8i16 res0_m, res1_m, res2_m, res3_m;                                      \
                                                                               \
    LD_UB4(dst_m, dst_stride, dst0_m, dst1_m, dst2_m, dst3_m);                 \
    ILVR_B4_SH(zero_m, dst0_m, zero_m, dst1_m, zero_m, dst2_m, zero_m, dst3_m, \
               res0_m, res1_m, res2_m, res3_m);                                \
    ADD4(res0_m, in0, res1_m, in1, res2_m, in2, res3_m, in3, res0_m, res1_m,   \
         res2_m, res3_m);                                                      \
    CLIP_SH4_0_255(res0_m, res1_m, res2_m, res3_m);                            \
    PCKEV_B2_SB(res1_m, res0_m, res3_m, res2_m, tmp0_m, tmp1_m);               \
    ST8x4_UB(tmp0_m, tmp1_m, dst_m, dst_stride);                               \
  }

#define VP9_IDCT4x4(in0, in1, in2, in3, out0, out1, out2, out3)             \
  {                                                                         \
    v8i16 c0_m, c1_m, c2_m, c3_m;                                           \
    v8i16 step0_m, step1_m;                                                 \
    v4i32 tmp0_m, tmp1_m, tmp2_m, tmp3_m;                                   \
                                                                            \
    c0_m = VP9_SET_COSPI_PAIR(cospi_16_64, cospi_16_64);                    \
    c1_m = VP9_SET_COSPI_PAIR(cospi_16_64, -cospi_16_64);                   \
    step0_m = __msa_ilvr_h(in2, in0);                                       \
    DOTP_SH2_SW(step0_m, step0_m, c0_m, c1_m, tmp0_m, tmp1_m);              \
                                                                            \
    c2_m = VP9_SET_COSPI_PAIR(cospi_24_64, -cospi_8_64);                    \
    c3_m = VP9_SET_COSPI_PAIR(cospi_8_64, cospi_24_64);                     \
    step1_m = __msa_ilvr_h(in3, in1);                                       \
    DOTP_SH2_SW(step1_m, step1_m, c2_m, c3_m, tmp2_m, tmp3_m);              \
    SRARI_W4_SW(tmp0_m, tmp1_m, tmp2_m, tmp3_m, DCT_CONST_BITS);            \
                                                                            \
    PCKEV_H2_SW(tmp1_m, tmp0_m, tmp3_m, tmp2_m, tmp0_m, tmp2_m);            \
    SLDI_B2_0_SW(tmp0_m, tmp2_m, tmp1_m, tmp3_m, 8);                        \
    BUTTERFLY_4((v8i16)tmp0_m, (v8i16)tmp1_m, (v8i16)tmp2_m, (v8i16)tmp3_m, \
                out0, out1, out2, out3);                                    \
  }

#define VP9_IADST4x4(in0, in1, in2, in3, out0, out1, out2, out3)       \
  {                                                                    \
    v8i16 res0_m, res1_m, c0_m, c1_m;                                  \
    v8i16 k1_m, k2_m, k3_m, k4_m;                                      \
    v8i16 zero_m = { 0 };                                              \
    v4i32 tmp0_m, tmp1_m, tmp2_m, tmp3_m;                              \
    v4i32 int0_m, int1_m, int2_m, int3_m;                              \
    v8i16 mask_m = { sinpi_1_9,  sinpi_2_9,  sinpi_3_9,  sinpi_4_9,    \
                     -sinpi_1_9, -sinpi_2_9, -sinpi_3_9, -sinpi_4_9 }; \
                                                                       \
    SPLATI_H4_SH(mask_m, 3, 0, 1, 2, c0_m, c1_m, k1_m, k2_m);          \
    ILVEV_H2_SH(c0_m, c1_m, k1_m, k2_m, c0_m, c1_m);                   \
    ILVR_H2_SH(in0, in2, in1, in3, res0_m, res1_m);                    \
    DOTP_SH2_SW(res0_m, res1_m, c0_m, c1_m, tmp2_m, tmp1_m);           \
    int0_m = tmp2_m + tmp1_m;                                          \
                                                                       \
    SPLATI_H2_SH(mask_m, 4, 7, k4_m, k3_m);                            \
    ILVEV_H2_SH(k4_m, k1_m, k3_m, k2_m, c0_m, c1_m);                   \
    DOTP_SH2_SW(res0_m, res1_m, c0_m, c1_m, tmp0_m, tmp1_m);           \
    int1_m = tmp0_m + tmp1_m;                                          \
                                                                       \
    c0_m = __msa_splati_h(mask_m, 6);                                  \
    ILVL_H2_SH(k2_m, c0_m, zero_m, k2_m, c0_m, c1_m);                  \
    ILVR_H2_SH(in0, in2, in1, in3, res0_m, res1_m);                    \
    DOTP_SH2_SW(res0_m, res1_m, c0_m, c1_m, tmp0_m, tmp1_m);           \
    int2_m = tmp0_m + tmp1_m;                                          \
                                                                       \
    c0_m = __msa_splati_h(mask_m, 6);                                  \
    c0_m = __msa_ilvev_h(c0_m, k1_m);                                  \
                                                                       \
    res0_m = __msa_ilvr_h((in1), (in3));                               \
    tmp0_m = __msa_dotp_s_w(res0_m, c0_m);                             \
    int3_m = tmp2_m + tmp0_m;                                          \
                                                                       \
    res0_m = __msa_ilvr_h((in2), (in3));                               \
    c1_m = __msa_ilvev_h(k4_m, k3_m);                                  \
                                                                       \
    tmp2_m = __msa_dotp_s_w(res0_m, c1_m);                             \
    res1_m = __msa_ilvr_h((in0), (in2));                               \
    c1_m = __msa_ilvev_h(k1_m, zero_m);                                \
                                                                       \
    tmp3_m = __msa_dotp_s_w(res1_m, c1_m);                             \
    int3_m += tmp2_m;                                                  \
    int3_m += tmp3_m;                                                  \
                                                                       \
    SRARI_W4_SW(int0_m, int1_m, int2_m, int3_m, DCT_CONST_BITS);       \
    PCKEV_H2_SH(int0_m, int0_m, int1_m, int1_m, out0, out1);           \
    PCKEV_H2_SH(int2_m, int2_m, int3_m, int3_m, out2, out3);           \
  }

#define VP9_SET_CONST_PAIR(mask_h, idx1_h, idx2_h)    \
  ({                                                  \
    v8i16 c0_m, c1_m;                                 \
                                                      \
    SPLATI_H2_SH(mask_h, idx1_h, idx2_h, c0_m, c1_m); \
    c0_m = __msa_ilvev_h(c1_m, c0_m);                 \
                                                      \
    c0_m;                                             \
  })

/* multiply and add macro */
#define VP9_MADD(inp0, inp1, inp2, inp3, cst0, cst1, cst2, cst3, out0, out1,  \
                 out2, out3)                                                  \
  {                                                                           \
    v8i16 madd_s0_m, madd_s1_m, madd_s2_m, madd_s3_m;                         \
    v4i32 tmp0_madd, tmp1_madd, tmp2_madd, tmp3_madd;                         \
                                                                              \
    ILVRL_H2_SH(inp1, inp0, madd_s1_m, madd_s0_m);                            \
    ILVRL_H2_SH(inp3, inp2, madd_s3_m, madd_s2_m);                            \
    DOTP_SH4_SW(madd_s1_m, madd_s0_m, madd_s1_m, madd_s0_m, cst0, cst0, cst1, \
                cst1, tmp0_madd, tmp1_madd, tmp2_madd, tmp3_madd);            \
    SRARI_W4_SW(tmp0_madd, tmp1_madd, tmp2_madd, tmp3_madd, DCT_CONST_BITS);  \
    PCKEV_H2_SH(tmp1_madd, tmp0_madd, tmp3_madd, tmp2_madd, out0, out1);      \
    DOTP_SH4_SW(madd_s3_m, madd_s2_m, madd_s3_m, madd_s2_m, cst2, cst2, cst3, \
                cst3, tmp0_madd, tmp1_madd, tmp2_madd, tmp3_madd);            \
    SRARI_W4_SW(tmp0_madd, tmp1_madd, tmp2_madd, tmp3_madd, DCT_CONST_BITS);  \
    PCKEV_H2_SH(tmp1_madd, tmp0_madd, tmp3_madd, tmp2_madd, out2, out3);      \
  }

/* idct 8x8 macro */
#define VP9_IDCT8x8_1D(in0, in1, in2, in3, in4, in5, in6, in7, out0, out1,    \
                       out2, out3, out4, out5, out6, out7)                    \
  {                                                                           \
    v8i16 tp0_m, tp1_m, tp2_m, tp3_m, tp4_m, tp5_m, tp6_m, tp7_m;             \
    v8i16 k0_m, k1_m, k2_m, k3_m, res0_m, res1_m, res2_m, res3_m;             \
    v4i32 tmp0_m, tmp1_m, tmp2_m, tmp3_m;                                     \
    v8i16 mask_m = { cospi_28_64, cospi_4_64,  cospi_20_64,  cospi_12_64,     \
                     cospi_16_64, -cospi_4_64, -cospi_20_64, -cospi_16_64 };  \
                                                                              \
    k0_m = VP9_SET_CONST_PAIR(mask_m, 0, 5);                                  \
    k1_m = VP9_SET_CONST_PAIR(mask_m, 1, 0);                                  \
    k2_m = VP9_SET_CONST_PAIR(mask_m, 6, 3);                                  \
    k3_m = VP9_SET_CONST_PAIR(mask_m, 3, 2);                                  \
    VP9_MADD(in1, in7, in3, in5, k0_m, k1_m, k2_m, k3_m, in1, in7, in3, in5); \
    SUB2(in1, in3, in7, in5, res0_m, res1_m);                                 \
    k0_m = VP9_SET_CONST_PAIR(mask_m, 4, 7);                                  \
    k1_m = __msa_splati_h(mask_m, 4);                                         \
                                                                              \
    ILVRL_H2_SH(res0_m, res1_m, res2_m, res3_m);                              \
    DOTP_SH4_SW(res2_m, res3_m, res2_m, res3_m, k0_m, k0_m, k1_m, k1_m,       \
                tmp0_m, tmp1_m, tmp2_m, tmp3_m);                              \
    SRARI_W4_SW(tmp0_m, tmp1_m, tmp2_m, tmp3_m, DCT_CONST_BITS);              \
    tp4_m = in1 + in3;                                                        \
    PCKEV_H2_SH(tmp1_m, tmp0_m, tmp3_m, tmp2_m, tp5_m, tp6_m);                \
    tp7_m = in7 + in5;                                                        \
    k2_m = VP9_SET_COSPI_PAIR(cospi_24_64, -cospi_8_64);                      \
    k3_m = VP9_SET_COSPI_PAIR(cospi_8_64, cospi_24_64);                       \
    VP9_MADD(in0, in4, in2, in6, k1_m, k0_m, k2_m, k3_m, in0, in4, in2, in6); \
    BUTTERFLY_4(in0, in4, in2, in6, tp0_m, tp1_m, tp2_m, tp3_m);              \
    BUTTERFLY_8(tp0_m, tp1_m, tp2_m, tp3_m, tp4_m, tp5_m, tp6_m, tp7_m, out0, \
                out1, out2, out3, out4, out5, out6, out7);                    \
  }

#define VP9_IADST8x8_1D(in0, in1, in2, in3, in4, in5, in6, in7, out0, out1,   \
                        out2, out3, out4, out5, out6, out7)                   \
  {                                                                           \
    v4i32 r0_m, r1_m, r2_m, r3_m, r4_m, r5_m, r6_m, r7_m;                     \
    v4i32 m0_m, m1_m, m2_m, m3_m, t0_m, t1_m;                                 \
    v8i16 res0_m, res1_m, res2_m, res3_m, k0_m, k1_m, in_s0, in_s1;           \
    v8i16 mask1_m = { cospi_2_64,  cospi_30_64,  -cospi_2_64, cospi_10_64,    \
                      cospi_22_64, -cospi_10_64, cospi_18_64, cospi_14_64 };  \
    v8i16 mask2_m = { cospi_14_64,  -cospi_18_64, cospi_26_64, cospi_6_64,    \
                      -cospi_26_64, cospi_8_64,   cospi_24_64, -cospi_8_64 }; \
    v8i16 mask3_m = {                                                         \
      -cospi_24_64, cospi_8_64, cospi_16_64, -cospi_16_64, 0, 0, 0, 0         \
    };                                                                        \
                                                                              \
    k0_m = VP9_SET_CONST_PAIR(mask1_m, 0, 1);                                 \
    k1_m = VP9_SET_CONST_PAIR(mask1_m, 1, 2);                                 \
    ILVRL_H2_SH(in1, in0, in_s1, in_s0);                                      \
    DOTP_SH4_SW(in_s1, in_s0, in_s1, in_s0, k0_m, k0_m, k1_m, k1_m, r0_m,     \
                r1_m, r2_m, r3_m);                                            \
    k0_m = VP9_SET_CONST_PAIR(mask1_m, 6, 7);                                 \
    k1_m = VP9_SET_CONST_PAIR(mask2_m, 0, 1);                                 \
    ILVRL_H2_SH(in5, in4, in_s1, in_s0);                                      \
    DOTP_SH4_SW(in_s1, in_s0, in_s1, in_s0, k0_m, k0_m, k1_m, k1_m, r4_m,     \
                r5_m, r6_m, r7_m);                                            \
    ADD4(r0_m, r4_m, r1_m, r5_m, r2_m, r6_m, r3_m, r7_m, m0_m, m1_m, m2_m,    \
         m3_m);                                                               \
    SRARI_W4_SW(m0_m, m1_m, m2_m, m3_m, DCT_CONST_BITS);                      \
    PCKEV_H2_SH(m1_m, m0_m, m3_m, m2_m, res0_m, res1_m);                      \
    SUB4(r0_m, r4_m, r1_m, r5_m, r2_m, r6_m, r3_m, r7_m, m0_m, m1_m, m2_m,    \
         m3_m);                                                               \
    SRARI_W4_SW(m0_m, m1_m, m2_m, m3_m, DCT_CONST_BITS);                      \
    PCKEV_H2_SW(m1_m, m0_m, m3_m, m2_m, t0_m, t1_m);                          \
    k0_m = VP9_SET_CONST_PAIR(mask1_m, 3, 4);                                 \
    k1_m = VP9_SET_CONST_PAIR(mask1_m, 4, 5);                                 \
    ILVRL_H2_SH(in3, in2, in_s1, in_s0);                                      \
    DOTP_SH4_SW(in_s1, in_s0, in_s1, in_s0, k0_m, k0_m, k1_m, k1_m, r0_m,     \
                r1_m, r2_m, r3_m);                                            \
    k0_m = VP9_SET_CONST_PAIR(mask2_m, 2, 3);                                 \
    k1_m = VP9_SET_CONST_PAIR(mask2_m, 3, 4);                                 \
    ILVRL_H2_SH(in7, in6, in_s1, in_s0);                                      \
    DOTP_SH4_SW(in_s1, in_s0, in_s1, in_s0, k0_m, k0_m, k1_m, k1_m, r4_m,     \
                r5_m, r6_m, r7_m);                                            \
    ADD4(r0_m, r4_m, r1_m, r5_m, r2_m, r6_m, r3_m, r7_m, m0_m, m1_m, m2_m,    \
         m3_m);                                                               \
    SRARI_W4_SW(m0_m, m1_m, m2_m, m3_m, DCT_CONST_BITS);                      \
    PCKEV_H2_SH(m1_m, m0_m, m3_m, m2_m, res2_m, res3_m);                      \
    SUB4(r0_m, r4_m, r1_m, r5_m, r2_m, r6_m, r3_m, r7_m, m0_m, m1_m, m2_m,    \
         m3_m);                                                               \
    SRARI_W4_SW(m0_m, m1_m, m2_m, m3_m, DCT_CONST_BITS);                      \
    PCKEV_H2_SW(m1_m, m0_m, m3_m, m2_m, r2_m, r3_m);                          \
    ILVRL_H2_SW(r3_m, r2_m, m2_m, m3_m);                                      \
    BUTTERFLY_4(res0_m, res1_m, res3_m, res2_m, out0, in7, in4, in3);         \
    k0_m = VP9_SET_CONST_PAIR(mask2_m, 5, 6);                                 \
    k1_m = VP9_SET_CONST_PAIR(mask2_m, 6, 7);                                 \
    ILVRL_H2_SH(t1_m, t0_m, in_s1, in_s0);                                    \
    DOTP_SH4_SW(in_s1, in_s0, in_s1, in_s0, k0_m, k0_m, k1_m, k1_m, r0_m,     \
                r1_m, r2_m, r3_m);                                            \
    k1_m = VP9_SET_CONST_PAIR(mask3_m, 0, 1);                                 \
    DOTP_SH4_SW(m2_m, m3_m, m2_m, m3_m, k0_m, k0_m, k1_m, k1_m, r4_m, r5_m,   \
                r6_m, r7_m);                                                  \
    ADD4(r0_m, r6_m, r1_m, r7_m, r2_m, r4_m, r3_m, r5_m, m0_m, m1_m, m2_m,    \
         m3_m);                                                               \
    SRARI_W4_SW(m0_m, m1_m, m2_m, m3_m, DCT_CONST_BITS);                      \
    PCKEV_H2_SH(m1_m, m0_m, m3_m, m2_m, in1, out6);                           \
    SUB4(r0_m, r6_m, r1_m, r7_m, r2_m, r4_m, r3_m, r5_m, m0_m, m1_m, m2_m,    \
         m3_m);                                                               \
    SRARI_W4_SW(m0_m, m1_m, m2_m, m3_m, DCT_CONST_BITS);                      \
    PCKEV_H2_SH(m1_m, m0_m, m3_m, m2_m, in2, in5);                            \
    k0_m = VP9_SET_CONST_PAIR(mask3_m, 2, 2);                                 \
    k1_m = VP9_SET_CONST_PAIR(mask3_m, 2, 3);                                 \
    ILVRL_H2_SH(in4, in3, in_s1, in_s0);                                      \
    DOTP_SH4_SW(in_s1, in_s0, in_s1, in_s0, k0_m, k0_m, k1_m, k1_m, m0_m,     \
                m1_m, m2_m, m3_m);                                            \
    SRARI_W4_SW(m0_m, m1_m, m2_m, m3_m, DCT_CONST_BITS);                      \
    PCKEV_H2_SH(m1_m, m0_m, m3_m, m2_m, in3, out4);                           \
    ILVRL_H2_SW(in5, in2, m2_m, m3_m);                                        \
    DOTP_SH4_SW(m2_m, m3_m, m2_m, m3_m, k0_m, k0_m, k1_m, k1_m, m0_m, m1_m,   \
                m2_m, m3_m);                                                  \
    SRARI_W4_SW(m0_m, m1_m, m2_m, m3_m, DCT_CONST_BITS);                      \
    PCKEV_H2_SH(m1_m, m0_m, m3_m, m2_m, out2, in5);                           \
                                                                              \
    out1 = -in1;                                                              \
    out3 = -in3;                                                              \
    out5 = -in5;                                                              \
    out7 = -in7;                                                              \
  }

#define VP9_IADST8x16_1D(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11,     \
                         r12, r13, r14, r15, out0, out1, out2, out3, out4,     \
                         out5, out6, out7, out8, out9, out10, out11, out12,    \
                         out13, out14, out15)                                  \
  {                                                                            \
    v8i16 g0_m, g1_m, g2_m, g3_m, g4_m, g5_m, g6_m, g7_m;                      \
    v8i16 g8_m, g9_m, g10_m, g11_m, g12_m, g13_m, g14_m, g15_m;                \
    v8i16 h0_m, h1_m, h2_m, h3_m, h4_m, h5_m, h6_m, h7_m;                      \
    v8i16 h8_m, h9_m, h10_m, h11_m;                                            \
    v8i16 k0_m, k1_m, k2_m, k3_m;                                              \
                                                                               \
    /* stage 1 */                                                              \
    k0_m = VP9_SET_COSPI_PAIR(cospi_1_64, cospi_31_64);                        \
    k1_m = VP9_SET_COSPI_PAIR(cospi_31_64, -cospi_1_64);                       \
    k2_m = VP9_SET_COSPI_PAIR(cospi_17_64, cospi_15_64);                       \
    k3_m = VP9_SET_COSPI_PAIR(cospi_15_64, -cospi_17_64);                      \
    MADD_BF(r15, r0, r7, r8, k0_m, k1_m, k2_m, k3_m, g0_m, g1_m, g2_m, g3_m);  \
    k0_m = VP9_SET_COSPI_PAIR(cospi_5_64, cospi_27_64);                        \
    k1_m = VP9_SET_COSPI_PAIR(cospi_27_64, -cospi_5_64);                       \
    k2_m = VP9_SET_COSPI_PAIR(cospi_21_64, cospi_11_64);                       \
    k3_m = VP9_SET_COSPI_PAIR(cospi_11_64, -cospi_21_64);                      \
    MADD_BF(r13, r2, r5, r10, k0_m, k1_m, k2_m, k3_m, g4_m, g5_m, g6_m, g7_m); \
    k0_m = VP9_SET_COSPI_PAIR(cospi_9_64, cospi_23_64);                        \
    k1_m = VP9_SET_COSPI_PAIR(cospi_23_64, -cospi_9_64);                       \
    k2_m = VP9_SET_COSPI_PAIR(cospi_25_64, cospi_7_64);                        \
    k3_m = VP9_SET_COSPI_PAIR(cospi_7_64, -cospi_25_64);                       \
    MADD_BF(r11, r4, r3, r12, k0_m, k1_m, k2_m, k3_m, g8_m, g9_m, g10_m,       \
            g11_m);                                                            \
    k0_m = VP9_SET_COSPI_PAIR(cospi_13_64, cospi_19_64);                       \
    k1_m = VP9_SET_COSPI_PAIR(cospi_19_64, -cospi_13_64);                      \
    k2_m = VP9_SET_COSPI_PAIR(cospi_29_64, cospi_3_64);                        \
    k3_m = VP9_SET_COSPI_PAIR(cospi_3_64, -cospi_29_64);                       \
    MADD_BF(r9, r6, r1, r14, k0_m, k1_m, k2_m, k3_m, g12_m, g13_m, g14_m,      \
            g15_m);                                                            \
                                                                               \
    /* stage 2 */                                                              \
    k0_m = VP9_SET_COSPI_PAIR(cospi_4_64, cospi_28_64);                        \
    k1_m = VP9_SET_COSPI_PAIR(cospi_28_64, -cospi_4_64);                       \
    k2_m = VP9_SET_COSPI_PAIR(-cospi_28_64, cospi_4_64);                       \
    MADD_BF(g1_m, g3_m, g9_m, g11_m, k0_m, k1_m, k2_m, k0_m, h0_m, h1_m, h2_m, \
            h3_m);                                                             \
    k0_m = VP9_SET_COSPI_PAIR(cospi_12_64, cospi_20_64);                       \
    k1_m = VP9_SET_COSPI_PAIR(-cospi_20_64, cospi_12_64);                      \
    k2_m = VP9_SET_COSPI_PAIR(cospi_20_64, -cospi_12_64);                      \
    MADD_BF(g7_m, g5_m, g15_m, g13_m, k0_m, k1_m, k2_m, k0_m, h4_m, h5_m,      \
            h6_m, h7_m);                                                       \
    BUTTERFLY_4(h0_m, h2_m, h6_m, h4_m, out8, out9, out11, out10);             \
    BUTTERFLY_8(g0_m, g2_m, g4_m, g6_m, g14_m, g12_m, g10_m, g8_m, h8_m, h9_m, \
                h10_m, h11_m, h6_m, h4_m, h2_m, h0_m);                         \
                                                                               \
    /* stage 3 */                                                              \
    BUTTERFLY_4(h8_m, h9_m, h11_m, h10_m, out0, out1, h11_m, h10_m);           \
    k0_m = VP9_SET_COSPI_PAIR(cospi_8_64, cospi_24_64);                        \
    k1_m = VP9_SET_COSPI_PAIR(cospi_24_64, -cospi_8_64);                       \
    k2_m = VP9_SET_COSPI_PAIR(-cospi_24_64, cospi_8_64);                       \
    MADD_BF(h0_m, h2_m, h4_m, h6_m, k0_m, k1_m, k2_m, k0_m, out4, out6, out5,  \
            out7);                                                             \
    MADD_BF(h1_m, h3_m, h5_m, h7_m, k0_m, k1_m, k2_m, k0_m, out12, out14,      \
            out13, out15);                                                     \
                                                                               \
    /* stage 4 */                                                              \
    k0_m = VP9_SET_COSPI_PAIR(cospi_16_64, cospi_16_64);                       \
    k1_m = VP9_SET_COSPI_PAIR(-cospi_16_64, -cospi_16_64);                     \
    k2_m = VP9_SET_COSPI_PAIR(cospi_16_64, -cospi_16_64);                      \
    k3_m = VP9_SET_COSPI_PAIR(-cospi_16_64, cospi_16_64);                      \
    MADD_SHORT(h10_m, h11_m, k1_m, k2_m, out2, out3);                          \
    MADD_SHORT(out6, out7, k0_m, k3_m, out6, out7);                            \
    MADD_SHORT(out10, out11, k0_m, k3_m, out10, out11);                        \
    MADD_SHORT(out14, out15, k1_m, k2_m, out14, out15);                        \
  }

void vpx_idct16_1d_columns_addblk_msa(int16_t *input, uint8_t *dst,
                                      int32_t dst_stride);
void vpx_idct16_1d_rows_msa(const int16_t *input, int16_t *output);
void vpx_iadst16_1d_columns_addblk_msa(int16_t *input, uint8_t *dst,
                                       int32_t dst_stride);
void vpx_iadst16_1d_rows_msa(const int16_t *input, int16_t *output);
#endif  // VPX_VPX_DSP_MIPS_INV_TXFM_MSA_H_
