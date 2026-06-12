/*
 *  Copyright (c) 2022 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_LOONGARCH_FWD_TXFM_LSX_H_
#define VPX_VPX_DSP_LOONGARCH_FWD_TXFM_LSX_H_

#include "vpx_dsp/loongarch/txfm_macros_lsx.h"
#include "vpx_dsp/txfm_common.h"

#define VP9_FDCT4(in0, in1, in2, in3, out0, out1, out2, out3)                 \
  do {                                                                        \
    __m128i cnst0_m, cnst1_m, cnst2_m, cnst3_m;                               \
    __m128i vec0_m, vec1_m, vec2_m, vec3_m;                                   \
    __m128i vec4_m, vec5_m, vec6_m, vec7_m;                                   \
    __m128i coeff_m = { 0x187e3b21d2bf2d41, 0x000000000000c4df };             \
                                                                              \
    LSX_BUTTERFLY_4_H(in0, in1, in2, in3, vec0_m, vec1_m, vec2_m, vec3_m);    \
    DUP2_ARG2(__lsx_vilvl_h, vec1_m, vec0_m, vec3_m, vec2_m, vec0_m, vec2_m); \
    DUP2_ARG2(__lsx_vreplvei_h, coeff_m, 0, coeff_m, 1, cnst0_m, cnst1_m);    \
    cnst1_m = __lsx_vpackev_h(cnst1_m, cnst0_m);                              \
    vec5_m = __lsx_vdp2_w_h(vec0_m, cnst1_m);                                 \
    DUP2_ARG2(__lsx_vreplvei_h, coeff_m, 4, coeff_m, 3, cnst2_m, cnst3_m);    \
    cnst2_m = __lsx_vpackev_h(cnst3_m, cnst2_m);                              \
    vec7_m = __lsx_vdp2_w_h(vec2_m, cnst2_m);                                 \
                                                                              \
    vec4_m = __lsx_vdp2_w_h(vec0_m, cnst0_m);                                 \
    cnst2_m = __lsx_vreplvei_h(coeff_m, 2);                                   \
    cnst2_m = __lsx_vpackev_h(cnst2_m, cnst3_m);                              \
    vec6_m = __lsx_vdp2_w_h(vec2_m, cnst2_m);                                 \
                                                                              \
    DUP4_ARG3(__lsx_vssrarni_h_w, vec4_m, vec4_m, DCT_CONST_BITS, vec5_m,     \
              vec5_m, DCT_CONST_BITS, vec6_m, vec6_m, DCT_CONST_BITS, vec7_m, \
              vec7_m, DCT_CONST_BITS, out0, out2, out1, out3);                \
  } while (0)

#define VP9_FDCT8(in0, in1, in2, in3, in4, in5, in6, in7, out0, out1, out2, \
                  out3, out4, out5, out6, out7)                             \
  do {                                                                      \
    __m128i s0_m, s1_m, s2_m, s3_m, s4_m, s5_m, s6_m;                       \
    __m128i s7_m, x0_m, x1_m, x2_m, x3_m;                                   \
    __m128i coeff_m = { 0x187e3b21d2bf2d41, 0x238e35370c7c3ec5 };           \
                                                                            \
    /* FDCT stage1 */                                                       \
    LSX_BUTTERFLY_8_H(in0, in1, in2, in3, in4, in5, in6, in7, s0_m, s1_m,   \
                      s2_m, s3_m, s4_m, s5_m, s6_m, s7_m);                  \
    LSX_BUTTERFLY_4_H(s0_m, s1_m, s2_m, s3_m, x0_m, x1_m, x2_m, x3_m);      \
    DUP2_ARG2(__lsx_vilvh_h, x1_m, x0_m, x3_m, x2_m, s0_m, s2_m);           \
    DUP2_ARG2(__lsx_vilvl_h, x1_m, x0_m, x3_m, x2_m, s1_m, s3_m);           \
    DUP2_ARG2(__lsx_vreplvei_h, coeff_m, 0, coeff_m, 1, x0_m, x1_m);        \
    x1_m = __lsx_vpackev_h(x1_m, x0_m);                                     \
    DOT_SHIFT_RIGHT_PCK_H(s0_m, s1_m, x1_m, out4);                          \
                                                                            \
    DUP2_ARG2(__lsx_vreplvei_h, coeff_m, 2, coeff_m, 3, x2_m, x3_m);        \
    x2_m = __lsx_vneg_h(x2_m);                                              \
    x2_m = __lsx_vpackev_h(x3_m, x2_m);                                     \
    DOT_SHIFT_RIGHT_PCK_H(s2_m, s3_m, x2_m, out6);                          \
                                                                            \
    DOT_SHIFT_RIGHT_PCK_H(s0_m, s1_m, x0_m, out0);                          \
    x2_m = __lsx_vreplvei_h(coeff_m, 2);                                    \
    x2_m = __lsx_vpackev_h(x2_m, x3_m);                                     \
    DOT_SHIFT_RIGHT_PCK_H(s2_m, s3_m, x2_m, out2);                          \
                                                                            \
    /* stage2 */                                                            \
    s1_m = __lsx_vilvl_h(s5_m, s6_m);                                       \
    s0_m = __lsx_vilvh_h(s5_m, s6_m);                                       \
                                                                            \
    DOT_SHIFT_RIGHT_PCK_H(s0_m, s1_m, x0_m, s6_m);                          \
    DOT_SHIFT_RIGHT_PCK_H(s0_m, s1_m, x1_m, s5_m);                          \
                                                                            \
    /* stage3 */                                                            \
    LSX_BUTTERFLY_4_H(s4_m, s7_m, s6_m, s5_m, x0_m, x3_m, x2_m, x1_m);      \
                                                                            \
    /* stage4 */                                                            \
    DUP2_ARG2(__lsx_vilvh_h, x3_m, x0_m, x2_m, x1_m, s4_m, s6_m);           \
    DUP2_ARG2(__lsx_vilvl_h, x3_m, x0_m, x2_m, x1_m, s5_m, s7_m);           \
                                                                            \
    DUP2_ARG2(__lsx_vreplvei_h, coeff_m, 4, coeff_m, 5, x0_m, x1_m);        \
    x1_m = __lsx_vpackev_h(x0_m, x1_m);                                     \
    DOT_SHIFT_RIGHT_PCK_H(s4_m, s5_m, x1_m, out1);                          \
                                                                            \
    DUP2_ARG2(__lsx_vreplvei_h, coeff_m, 6, coeff_m, 7, x2_m, x3_m);        \
    x2_m = __lsx_vpackev_h(x3_m, x2_m);                                     \
    DOT_SHIFT_RIGHT_PCK_H(s6_m, s7_m, x2_m, out5);                          \
                                                                            \
    x1_m = __lsx_vreplvei_h(coeff_m, 5);                                    \
    x0_m = __lsx_vneg_h(x0_m);                                              \
    x0_m = __lsx_vpackev_h(x1_m, x0_m);                                     \
    DOT_SHIFT_RIGHT_PCK_H(s4_m, s5_m, x0_m, out7);                          \
    x2_m = __lsx_vreplvei_h(coeff_m, 6);                                    \
    x3_m = __lsx_vneg_h(x3_m);                                              \
    x2_m = __lsx_vpackev_h(x2_m, x3_m);                                     \
    DOT_SHIFT_RIGHT_PCK_H(s6_m, s7_m, x2_m, out3);                          \
  } while (0)

#define SRLI_AVE_S_4V_H(in0, in1, in2, in3, in4, in5, in6, in7)             \
  do {                                                                      \
    __m128i vec0_m, vec1_m, vec2_m, vec3_m, vec4_m, vec5_m, vec6_m, vec7_m; \
                                                                            \
    DUP4_ARG2(__lsx_vsrli_h, in0, 15, in1, 15, in2, 15, in3, 15, vec0_m,    \
              vec1_m, vec2_m, vec3_m);                                      \
    DUP4_ARG2(__lsx_vsrli_h, in4, 15, in5, 15, in6, 15, in7, 15, vec4_m,    \
              vec5_m, vec6_m, vec7_m);                                      \
    DUP4_ARG2(__lsx_vavg_h, vec0_m, in0, vec1_m, in1, vec2_m, in2, vec3_m,  \
              in3, in0, in1, in2, in3);                                     \
    DUP4_ARG2(__lsx_vavg_h, vec4_m, in4, vec5_m, in5, vec6_m, in6, vec7_m,  \
              in7, in4, in5, in6, in7);                                     \
  } while (0)

#define FDCT32_POSTPROC_2V_POS_H(vec0, vec1) \
  do {                                       \
    __m128i tp0_m, tp1_m;                    \
    __m128i one = __lsx_vreplgr2vr_h(1);     \
                                             \
    tp0_m = __lsx_vslei_h(vec0, 0);          \
    tp1_m = __lsx_vslei_h(vec1, 0);          \
    tp0_m = __lsx_vxori_b(tp0_m, 255);       \
    tp1_m = __lsx_vxori_b(tp1_m, 255);       \
    vec0 = __lsx_vadd_h(vec0, one);          \
    vec1 = __lsx_vadd_h(vec1, one);          \
    tp0_m = __lsx_vand_v(one, tp0_m);        \
    tp1_m = __lsx_vand_v(one, tp1_m);        \
    vec0 = __lsx_vadd_h(vec0, tp0_m);        \
    vec1 = __lsx_vadd_h(vec1, tp1_m);        \
    vec0 = __lsx_vsrai_h(vec0, 2);           \
    vec1 = __lsx_vsrai_h(vec1, 2);           \
  } while (0)

#define FDCT_POSTPROC_2V_NEG_H(vec0, vec1) \
  do {                                     \
    __m128i tp0_m, tp1_m;                  \
    __m128i one_m = __lsx_vldi(0x401);     \
                                           \
    tp0_m = __lsx_vslti_h(vec0, 0);        \
    tp1_m = __lsx_vslti_h(vec1, 0);        \
    vec0 = __lsx_vadd_h(vec0, one_m);      \
    vec1 = __lsx_vadd_h(vec1, one_m);      \
    tp0_m = __lsx_vand_v(one_m, tp0_m);    \
    tp1_m = __lsx_vand_v(one_m, tp1_m);    \
    vec0 = __lsx_vadd_h(vec0, tp0_m);      \
    vec1 = __lsx_vadd_h(vec1, tp1_m);      \
    vec0 = __lsx_vsrai_h(vec0, 2);         \
    vec1 = __lsx_vsrai_h(vec1, 2);         \
  } while (0)

#define FDCT32_POSTPROC_NEG_W(vec)         \
  do {                                     \
    __m128i temp_m;                        \
    __m128i one_m = __lsx_vreplgr2vr_w(1); \
                                           \
    temp_m = __lsx_vslti_w(vec, 0);        \
    vec = __lsx_vadd_w(vec, one_m);        \
    temp_m = __lsx_vand_v(one_m, temp_m);  \
    vec = __lsx_vadd_w(vec, temp_m);       \
    vec = __lsx_vsrai_w(vec, 2);           \
  } while (0)

#define DOTP_CONST_PAIR_W(reg0_left, reg1_left, reg0_right, reg1_right,       \
                          const0, const1, out0, out1, out2, out3)             \
  do {                                                                        \
    __m128i s0_m, s1_m, s2_m, s3_m, s4_m, s5_m, s6_m, s7_m;                   \
    __m128i tp0_m, tp1_m, tp2_m, tp3_m, _tmp0, _tmp1;                         \
    __m128i k0_m = __lsx_vreplgr2vr_w((int32_t)const0);                       \
                                                                              \
    s0_m = __lsx_vreplgr2vr_w((int32_t)const1);                               \
    k0_m = __lsx_vpackev_w(s0_m, k0_m);                                       \
                                                                              \
    DUP2_ARG1(__lsx_vneg_w, reg1_left, reg1_right, _tmp0, _tmp1);             \
    s1_m = __lsx_vilvl_w(_tmp0, reg0_left);                                   \
    s0_m = __lsx_vilvh_w(_tmp0, reg0_left);                                   \
    s3_m = __lsx_vilvl_w(reg0_left, reg1_left);                               \
    s2_m = __lsx_vilvh_w(reg0_left, reg1_left);                               \
    s5_m = __lsx_vilvl_w(_tmp1, reg0_right);                                  \
    s4_m = __lsx_vilvh_w(_tmp1, reg0_right);                                  \
    s7_m = __lsx_vilvl_w(reg0_right, reg1_right);                             \
    s6_m = __lsx_vilvh_w(reg0_right, reg1_right);                             \
    DUP2_ARG2(__lsx_vdp2_d_w, s0_m, k0_m, s1_m, k0_m, tp0_m, tp1_m);          \
    DUP2_ARG2(__lsx_vdp2_d_w, s4_m, k0_m, s5_m, k0_m, tp2_m, tp3_m);          \
    DUP2_ARG3(__lsx_vssrarni_w_d, tp0_m, tp1_m, DCT_CONST_BITS, tp2_m, tp3_m, \
              DCT_CONST_BITS, out0, out1);                                    \
    DUP2_ARG2(__lsx_vdp2_d_w, s2_m, k0_m, s3_m, k0_m, tp0_m, tp1_m);          \
    DUP2_ARG2(__lsx_vdp2_d_w, s6_m, k0_m, s7_m, k0_m, tp2_m, tp3_m);          \
    DUP2_ARG3(__lsx_vssrarni_w_d, tp0_m, tp1_m, DCT_CONST_BITS, tp2_m, tp3_m, \
              DCT_CONST_BITS, out2, out3);                                    \
  } while (0)

#define VP9_ADDBLK_ST8x4_UB(dst, _stride, _stride2, _stride3, in0, in1, in2,   \
                            in3)                                               \
  do {                                                                         \
    __m128i dst0_m, dst1_m, dst2_m, dst3_m;                                    \
    __m128i tmp0_m, tmp1_m;                                                    \
    __m128i res0_m, res1_m, res2_m, res3_m;                                    \
                                                                               \
    dst0_m = __lsx_vld(dst, 0);                                                \
    DUP2_ARG2(__lsx_vldx, dst, _stride, dst, _stride2, dst1_m, dst2_m);        \
    dst3_m = __lsx_vldx(dst, _stride3);                                        \
    DUP4_ARG2(__lsx_vsllwil_hu_bu, dst0_m, 0, dst1_m, 0, dst2_m, 0, dst3_m, 0, \
              res0_m, res1_m, res2_m, res3_m);                                 \
    DUP4_ARG2(__lsx_vadd_h, res0_m, in0, res1_m, in1, res2_m, in2, res3_m,     \
              in3, res0_m, res1_m, res2_m, res3_m);                            \
    DUP2_ARG3(__lsx_vssrarni_bu_h, res1_m, res0_m, 0, res3_m, res2_m, 0,       \
              tmp0_m, tmp1_m);                                                 \
    __lsx_vstelm_d(tmp0_m, dst, 0, 0);                                         \
    __lsx_vstelm_d(tmp0_m, dst + _stride, 0, 1);                               \
    __lsx_vstelm_d(tmp1_m, dst + _stride2, 0, 0);                              \
    __lsx_vstelm_d(tmp1_m, dst + _stride3, 0, 1);                              \
  } while (0)

#define FDCT8x16_EVEN(in0, in1, in2, in3, in4, in5, in6, in7, out0, out1, \
                      out2, out3, out4, out5, out6, out7)                 \
  do {                                                                    \
    __m128i s0_m, s1_m, s2_m, s3_m, s4_m, s5_m, s6_m, s7_m;               \
    __m128i x0_m, x1_m, x2_m, x3_m;                                       \
    __m128i coeff_m = { 0x187e3b21d2bf2d41, 0x238e35370c7c3ec5 };         \
                                                                          \
    /* FDCT stage1 */                                                     \
    LSX_BUTTERFLY_8_H(in0, in1, in2, in3, in4, in5, in6, in7, s0_m, s1_m, \
                      s2_m, s3_m, s4_m, s5_m, s6_m, s7_m);                \
    LSX_BUTTERFLY_4_H(s0_m, s1_m, s2_m, s3_m, x0_m, x1_m, x2_m, x3_m);    \
    DUP2_ARG2(__lsx_vilvh_h, x1_m, x0_m, x3_m, x2_m, s0_m, s2_m);         \
    DUP2_ARG2(__lsx_vilvl_h, x1_m, x0_m, x3_m, x2_m, s1_m, s3_m);         \
    DUP2_ARG2(__lsx_vreplvei_h, coeff_m, 0, coeff_m, 1, x0_m, x1_m);      \
    x1_m = __lsx_vpackev_h(x1_m, x0_m);                                   \
    DOT_SHIFT_RIGHT_PCK_H(s0_m, s1_m, x1_m, out4);                        \
                                                                          \
    DUP2_ARG2(__lsx_vreplvei_h, coeff_m, 2, coeff_m, 3, x2_m, x3_m);      \
    x2_m = __lsx_vneg_h(x2_m);                                            \
    x2_m = __lsx_vpackev_h(x3_m, x2_m);                                   \
    DOT_SHIFT_RIGHT_PCK_H(s2_m, s3_m, x2_m, out6);                        \
                                                                          \
    DOT_SHIFT_RIGHT_PCK_H(s0_m, s1_m, x0_m, out0);                        \
    x2_m = __lsx_vreplvei_h(coeff_m, 2);                                  \
    x2_m = __lsx_vpackev_h(x2_m, x3_m);                                   \
    DOT_SHIFT_RIGHT_PCK_H(s2_m, s3_m, x2_m, out2);                        \
                                                                          \
    /* stage2 */                                                          \
    s1_m = __lsx_vilvl_h(s5_m, s6_m);                                     \
    s0_m = __lsx_vilvh_h(s5_m, s6_m);                                     \
                                                                          \
    DOT_SHIFT_RIGHT_PCK_H(s0_m, s1_m, x0_m, s6_m);                        \
    DOT_SHIFT_RIGHT_PCK_H(s0_m, s1_m, x1_m, s5_m);                        \
                                                                          \
    /* stage3 */                                                          \
    LSX_BUTTERFLY_4_H(s4_m, s7_m, s6_m, s5_m, x0_m, x3_m, x2_m, x1_m);    \
                                                                          \
    /* stage4 */                                                          \
    DUP2_ARG2(__lsx_vilvh_h, x3_m, x0_m, x2_m, x1_m, s4_m, s6_m);         \
    DUP2_ARG2(__lsx_vilvl_h, x3_m, x0_m, x2_m, x1_m, s5_m, s7_m);         \
                                                                          \
    DUP2_ARG2(__lsx_vreplvei_h, coeff_m, 4, coeff_m, 5, x0_m, x1_m);      \
    x1_m = __lsx_vpackev_h(x0_m, x1_m);                                   \
    DOT_SHIFT_RIGHT_PCK_H(s4_m, s5_m, x1_m, out1);                        \
                                                                          \
    DUP2_ARG2(__lsx_vreplvei_h, coeff_m, 6, coeff_m, 7, x2_m, x3_m);      \
    x2_m = __lsx_vpackev_h(x3_m, x2_m);                                   \
    DOT_SHIFT_RIGHT_PCK_H(s6_m, s7_m, x2_m, out5);                        \
                                                                          \
    x1_m = __lsx_vreplvei_h(coeff_m, 5);                                  \
    x0_m = __lsx_vneg_h(x0_m);                                            \
    x0_m = __lsx_vpackev_h(x1_m, x0_m);                                   \
    DOT_SHIFT_RIGHT_PCK_H(s4_m, s5_m, x0_m, out7);                        \
                                                                          \
    x2_m = __lsx_vreplvei_h(coeff_m, 6);                                  \
    x3_m = __lsx_vneg_h(x3_m);                                            \
    x2_m = __lsx_vpackev_h(x2_m, x3_m);                                   \
    DOT_SHIFT_RIGHT_PCK_H(s6_m, s7_m, x2_m, out3);                        \
  } while (0)

#define FDCT8x16_ODD(input0, input1, input2, input3, input4, input5, input6,  \
                     input7, out1, out3, out5, out7, out9, out11, out13,      \
                     out15)                                                   \
  do {                                                                        \
    __m128i stp21_m, stp22_m, stp23_m, stp24_m, stp25_m, stp26_m;             \
    __m128i stp30_m, stp31_m, stp32_m, stp33_m, stp34_m, stp35_m;             \
    __m128i stp36_m, stp37_m, vec0_m, vec1_m;                                 \
    __m128i vec2_m, vec3_m, vec4_m, vec5_m, vec6_m;                           \
    __m128i cnst0_m, cnst1_m, cnst4_m, cnst5_m;                               \
    __m128i coeff_m = { 0x187e3b21d2bf2d41, 0x238e3537e782c4df };             \
    __m128i coeff1_m = { 0x289a317906463fb1, 0x12943d3f1e2b3871 };            \
    __m128i coeff2_m = { 0xed6cd766c78fc04f, 0x0 };                           \
                                                                              \
    /* stp 1 */                                                               \
    DUP2_ARG2(__lsx_vilvh_h, input2, input5, input3, input4, vec2_m, vec4_m); \
    DUP2_ARG2(__lsx_vilvl_h, input2, input5, input3, input4, vec3_m, vec5_m); \
                                                                              \
    cnst4_m = __lsx_vreplvei_h(coeff_m, 0);                                   \
    DOT_SHIFT_RIGHT_PCK_H(vec2_m, vec3_m, cnst4_m, stp25_m);                  \
                                                                              \
    cnst5_m = __lsx_vreplvei_h(coeff_m, 1);                                   \
    cnst5_m = __lsx_vpackev_h(cnst5_m, cnst4_m);                              \
    DOT_SHIFT_RIGHT_PCK_H(vec2_m, vec3_m, cnst5_m, stp22_m);                  \
    DOT_SHIFT_RIGHT_PCK_H(vec4_m, vec5_m, cnst4_m, stp24_m);                  \
    DOT_SHIFT_RIGHT_PCK_H(vec4_m, vec5_m, cnst5_m, stp23_m);                  \
                                                                              \
    /* stp2 */                                                                \
    LSX_BUTTERFLY_4_H(input0, input1, stp22_m, stp23_m, stp30_m, stp31_m,     \
                      stp32_m, stp33_m);                                      \
    LSX_BUTTERFLY_4_H(input7, input6, stp25_m, stp24_m, stp37_m, stp36_m,     \
                      stp35_m, stp34_m);                                      \
                                                                              \
    DUP2_ARG2(__lsx_vilvh_h, stp36_m, stp31_m, stp35_m, stp32_m, vec2_m,      \
              vec4_m);                                                        \
    DUP2_ARG2(__lsx_vilvl_h, stp36_m, stp31_m, stp35_m, stp32_m, vec3_m,      \
              vec5_m);                                                        \
                                                                              \
    DUP2_ARG2(__lsx_vreplvei_h, coeff_m, 2, coeff_m, 3, cnst0_m, cnst1_m);    \
    cnst0_m = __lsx_vpackev_h(cnst0_m, cnst1_m);                              \
    DOT_SHIFT_RIGHT_PCK_H(vec2_m, vec3_m, cnst0_m, stp26_m);                  \
                                                                              \
    cnst0_m = __lsx_vreplvei_h(coeff_m, 4);                                   \
    cnst1_m = __lsx_vpackev_h(cnst1_m, cnst0_m);                              \
    DOT_SHIFT_RIGHT_PCK_H(vec2_m, vec3_m, cnst1_m, stp21_m);                  \
                                                                              \
    DUP2_ARG2(__lsx_vreplvei_h, coeff_m, 5, coeff_m, 2, cnst0_m, cnst1_m);    \
    cnst1_m = __lsx_vpackev_h(cnst0_m, cnst1_m);                              \
    DOT_SHIFT_RIGHT_PCK_H(vec4_m, vec5_m, cnst1_m, stp25_m);                  \
                                                                              \
    cnst0_m = __lsx_vreplvei_h(coeff_m, 3);                                   \
    cnst1_m = __lsx_vpackev_h(cnst1_m, cnst0_m);                              \
    DOT_SHIFT_RIGHT_PCK_H(vec4_m, vec5_m, cnst1_m, stp22_m);                  \
                                                                              \
    /* stp4 */                                                                \
    LSX_BUTTERFLY_4_H(stp30_m, stp37_m, stp26_m, stp21_m, vec6_m, vec2_m,     \
                      vec4_m, vec5_m);                                        \
    LSX_BUTTERFLY_4_H(stp33_m, stp34_m, stp25_m, stp22_m, stp21_m, stp23_m,   \
                      stp24_m, stp31_m);                                      \
                                                                              \
    vec1_m = __lsx_vilvl_h(vec2_m, vec6_m);                                   \
    vec0_m = __lsx_vilvh_h(vec2_m, vec6_m);                                   \
    DUP2_ARG2(__lsx_vreplvei_h, coeff1_m, 0, coeff1_m, 1, cnst0_m, cnst1_m);  \
    cnst0_m = __lsx_vpackev_h(cnst0_m, cnst1_m);                              \
                                                                              \
    DOT_SHIFT_RIGHT_PCK_H(vec0_m, vec1_m, cnst0_m, out1);                     \
                                                                              \
    cnst0_m = __lsx_vreplvei_h(coeff2_m, 0);                                  \
    cnst0_m = __lsx_vpackev_h(cnst1_m, cnst0_m);                              \
    DOT_SHIFT_RIGHT_PCK_H(vec0_m, vec1_m, cnst0_m, out15);                    \
                                                                              \
    vec1_m = __lsx_vilvl_h(vec4_m, vec5_m);                                   \
    vec0_m = __lsx_vilvh_h(vec4_m, vec5_m);                                   \
    DUP2_ARG2(__lsx_vreplvei_h, coeff1_m, 2, coeff1_m, 3, cnst0_m, cnst1_m);  \
    cnst1_m = __lsx_vpackev_h(cnst1_m, cnst0_m);                              \
                                                                              \
    DOT_SHIFT_RIGHT_PCK_H(vec0_m, vec1_m, cnst1_m, out9);                     \
                                                                              \
    cnst1_m = __lsx_vreplvei_h(coeff2_m, 2);                                  \
    cnst0_m = __lsx_vpackev_h(cnst0_m, cnst1_m);                              \
    DOT_SHIFT_RIGHT_PCK_H(vec0_m, vec1_m, cnst0_m, out7);                     \
                                                                              \
    vec1_m = __lsx_vilvl_h(stp23_m, stp21_m);                                 \
    vec0_m = __lsx_vilvh_h(stp23_m, stp21_m);                                 \
    DUP2_ARG2(__lsx_vreplvei_h, coeff1_m, 4, coeff1_m, 5, cnst0_m, cnst1_m);  \
    cnst0_m = __lsx_vpackev_h(cnst0_m, cnst1_m);                              \
    DOT_SHIFT_RIGHT_PCK_H(vec0_m, vec1_m, cnst0_m, out5);                     \
                                                                              \
    cnst0_m = __lsx_vreplvei_h(coeff2_m, 1);                                  \
    cnst0_m = __lsx_vpackev_h(cnst1_m, cnst0_m);                              \
    DOT_SHIFT_RIGHT_PCK_H(vec0_m, vec1_m, cnst0_m, out11);                    \
                                                                              \
    vec1_m = __lsx_vilvl_h(stp24_m, stp31_m);                                 \
    vec0_m = __lsx_vilvh_h(stp24_m, stp31_m);                                 \
    DUP2_ARG2(__lsx_vreplvei_h, coeff1_m, 6, coeff1_m, 7, cnst0_m, cnst1_m);  \
    cnst1_m = __lsx_vpackev_h(cnst1_m, cnst0_m);                              \
                                                                              \
    DOT_SHIFT_RIGHT_PCK_H(vec0_m, vec1_m, cnst1_m, out13);                    \
                                                                              \
    cnst1_m = __lsx_vreplvei_h(coeff2_m, 3);                                  \
    cnst0_m = __lsx_vpackev_h(cnst0_m, cnst1_m);                              \
    DOT_SHIFT_RIGHT_PCK_H(vec0_m, vec1_m, cnst0_m, out3);                     \
  } while (0)

void fdct8x16_1d_column(const int16_t *input, int16_t *tmp_ptr,
                        int32_t src_stride);
void fdct16x8_1d_row(int16_t *input, int16_t *output);
#endif  // VPX_VPX_DSP_LOONGARCH_FWD_TXFM_LSX_H_
