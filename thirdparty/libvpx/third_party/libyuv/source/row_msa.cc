/*
 *  Copyright 2016 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <string.h>

#include "libyuv/row.h"

// This module is for GCC MSA
#if !defined(LIBYUV_DISABLE_MSA) && defined(__mips_msa)
#include "libyuv/macros_msa.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

#define ALPHA_VAL (-1)

// Fill YUV -> RGB conversion constants into vectors
#define YUVTORGB_SETUP(yuvconst, ub, vr, ug, vg, bb, bg, br, yg) \
  {                                                              \
    ub = __msa_fill_w(yuvconst->kUVToB[0]);                      \
    vr = __msa_fill_w(yuvconst->kUVToR[1]);                      \
    ug = __msa_fill_w(yuvconst->kUVToG[0]);                      \
    vg = __msa_fill_w(yuvconst->kUVToG[1]);                      \
    bb = __msa_fill_w(yuvconst->kUVBiasB[0]);                    \
    bg = __msa_fill_w(yuvconst->kUVBiasG[0]);                    \
    br = __msa_fill_w(yuvconst->kUVBiasR[0]);                    \
    yg = __msa_fill_w(yuvconst->kYToRgb[0]);                     \
  }

// Load YUV 422 pixel data
#define READYUV422(psrc_y, psrc_u, psrc_v, out_y, out_u, out_v)    \
  {                                                                \
    uint64_t y_m;                                                  \
    uint32_t u_m, v_m;                                             \
    v4i32 zero_m = {0};                                            \
    y_m = LD(psrc_y);                                              \
    u_m = LW(psrc_u);                                              \
    v_m = LW(psrc_v);                                              \
    out_y = (v16u8)__msa_insert_d((v2i64)zero_m, 0, (int64_t)y_m); \
    out_u = (v16u8)__msa_insert_w(zero_m, 0, (int32_t)u_m);        \
    out_v = (v16u8)__msa_insert_w(zero_m, 0, (int32_t)v_m);        \
  }

// Clip input vector elements between 0 to 255
#define CLIP_0TO255(in0, in1, in2, in3, in4, in5) \
  {                                               \
    v4i32 max_m = __msa_ldi_w(0xFF);              \
                                                  \
    in0 = __msa_maxi_s_w(in0, 0);                 \
    in1 = __msa_maxi_s_w(in1, 0);                 \
    in2 = __msa_maxi_s_w(in2, 0);                 \
    in3 = __msa_maxi_s_w(in3, 0);                 \
    in4 = __msa_maxi_s_w(in4, 0);                 \
    in5 = __msa_maxi_s_w(in5, 0);                 \
    in0 = __msa_min_s_w(max_m, in0);              \
    in1 = __msa_min_s_w(max_m, in1);              \
    in2 = __msa_min_s_w(max_m, in2);              \
    in3 = __msa_min_s_w(max_m, in3);              \
    in4 = __msa_min_s_w(max_m, in4);              \
    in5 = __msa_min_s_w(max_m, in5);              \
  }

// Convert 8 pixels of YUV 420 to RGB.
#define YUVTORGB(in_y, in_uv, ubvr, ugvg, bb, bg, br, yg, out_b, out_g, out_r) \
  {                                                                            \
    v8i16 vec0_m, vec1_m;                                                      \
    v4i32 reg0_m, reg1_m, reg2_m, reg3_m, reg4_m;                              \
    v4i32 reg5_m, reg6_m, reg7_m;                                              \
    v16i8 zero_m = {0};                                                        \
                                                                               \
    vec0_m = (v8i16)__msa_ilvr_b((v16i8)in_y, (v16i8)in_y);                    \
    vec1_m = (v8i16)__msa_ilvr_b((v16i8)zero_m, (v16i8)in_uv);                 \
    reg0_m = (v4i32)__msa_ilvr_h((v8i16)zero_m, (v8i16)vec0_m);                \
    reg1_m = (v4i32)__msa_ilvl_h((v8i16)zero_m, (v8i16)vec0_m);                \
    reg2_m = (v4i32)__msa_ilvr_h((v8i16)zero_m, (v8i16)vec1_m);                \
    reg3_m = (v4i32)__msa_ilvl_h((v8i16)zero_m, (v8i16)vec1_m);                \
    reg0_m *= yg;                                                              \
    reg1_m *= yg;                                                              \
    reg2_m *= ubvr;                                                            \
    reg3_m *= ubvr;                                                            \
    reg0_m = __msa_srai_w(reg0_m, 16);                                         \
    reg1_m = __msa_srai_w(reg1_m, 16);                                         \
    reg4_m = __msa_dotp_s_w((v8i16)vec1_m, (v8i16)ugvg);                       \
    reg5_m = __msa_ilvev_w(reg2_m, reg2_m);                                    \
    reg6_m = __msa_ilvev_w(reg3_m, reg3_m);                                    \
    reg7_m = __msa_ilvr_w(reg4_m, reg4_m);                                     \
    reg2_m = __msa_ilvod_w(reg2_m, reg2_m);                                    \
    reg3_m = __msa_ilvod_w(reg3_m, reg3_m);                                    \
    reg4_m = __msa_ilvl_w(reg4_m, reg4_m);                                     \
    reg5_m = reg0_m - reg5_m;                                                  \
    reg6_m = reg1_m - reg6_m;                                                  \
    reg2_m = reg0_m - reg2_m;                                                  \
    reg3_m = reg1_m - reg3_m;                                                  \
    reg7_m = reg0_m - reg7_m;                                                  \
    reg4_m = reg1_m - reg4_m;                                                  \
    reg5_m += bb;                                                              \
    reg6_m += bb;                                                              \
    reg7_m += bg;                                                              \
    reg4_m += bg;                                                              \
    reg2_m += br;                                                              \
    reg3_m += br;                                                              \
    reg5_m = __msa_srai_w(reg5_m, 6);                                          \
    reg6_m = __msa_srai_w(reg6_m, 6);                                          \
    reg7_m = __msa_srai_w(reg7_m, 6);                                          \
    reg4_m = __msa_srai_w(reg4_m, 6);                                          \
    reg2_m = __msa_srai_w(reg2_m, 6);                                          \
    reg3_m = __msa_srai_w(reg3_m, 6);                                          \
    CLIP_0TO255(reg5_m, reg6_m, reg7_m, reg4_m, reg2_m, reg3_m);               \
    out_b = __msa_pckev_h((v8i16)reg6_m, (v8i16)reg5_m);                       \
    out_g = __msa_pckev_h((v8i16)reg4_m, (v8i16)reg7_m);                       \
    out_r = __msa_pckev_h((v8i16)reg3_m, (v8i16)reg2_m);                       \
  }

// Pack and Store 8 ARGB values.
#define STOREARGB(in0, in1, in2, in3, pdst_argb)           \
  {                                                        \
    v8i16 vec0_m, vec1_m;                                  \
    v16u8 dst0_m, dst1_m;                                  \
    vec0_m = (v8i16)__msa_ilvev_b((v16i8)in1, (v16i8)in0); \
    vec1_m = (v8i16)__msa_ilvev_b((v16i8)in3, (v16i8)in2); \
    dst0_m = (v16u8)__msa_ilvr_h(vec1_m, vec0_m);          \
    dst1_m = (v16u8)__msa_ilvl_h(vec1_m, vec0_m);          \
    ST_UB2(dst0_m, dst1_m, pdst_argb, 16);                 \
  }

// Takes ARGB input and calculates Y.
#define ARGBTOY(argb0, argb1, argb2, argb3, const0, const1, const2, shift, \
                y_out)                                                     \
  {                                                                        \
    v16u8 vec0_m, vec1_m, vec2_m, vec3_m;                                  \
    v8u16 reg0_m, reg1_m;                                                  \
                                                                           \
    vec0_m = (v16u8)__msa_pckev_h((v8i16)argb1, (v8i16)argb0);             \
    vec1_m = (v16u8)__msa_pckev_h((v8i16)argb3, (v8i16)argb2);             \
    vec2_m = (v16u8)__msa_pckod_h((v8i16)argb1, (v8i16)argb0);             \
    vec3_m = (v16u8)__msa_pckod_h((v8i16)argb3, (v8i16)argb2);             \
    reg0_m = __msa_dotp_u_h(vec0_m, const0);                               \
    reg1_m = __msa_dotp_u_h(vec1_m, const0);                               \
    reg0_m = __msa_dpadd_u_h(reg0_m, vec2_m, const1);                      \
    reg1_m = __msa_dpadd_u_h(reg1_m, vec3_m, const1);                      \
    reg0_m += const2;                                                      \
    reg1_m += const2;                                                      \
    reg0_m = (v8u16)__msa_srai_h((v8i16)reg0_m, shift);                    \
    reg1_m = (v8u16)__msa_srai_h((v8i16)reg1_m, shift);                    \
    y_out = (v16u8)__msa_pckev_b((v16i8)reg1_m, (v16i8)reg0_m);            \
  }

// Loads current and next row of ARGB input and averages it to calculate U and V
#define READ_ARGB(s_ptr, t_ptr, argb0, argb1, argb2, argb3)               \
  {                                                                       \
    v16u8 src0_m, src1_m, src2_m, src3_m, src4_m, src5_m, src6_m, src7_m; \
    v16u8 vec0_m, vec1_m, vec2_m, vec3_m, vec4_m, vec5_m, vec6_m, vec7_m; \
    v16u8 vec8_m, vec9_m;                                                 \
    v8u16 reg0_m, reg1_m, reg2_m, reg3_m, reg4_m, reg5_m, reg6_m, reg7_m; \
    v8u16 reg8_m, reg9_m;                                                 \
                                                                          \
    src0_m = (v16u8)__msa_ld_b((v16i8*)s, 0);                             \
    src1_m = (v16u8)__msa_ld_b((v16i8*)s, 16);                            \
    src2_m = (v16u8)__msa_ld_b((v16i8*)s, 32);                            \
    src3_m = (v16u8)__msa_ld_b((v16i8*)s, 48);                            \
    src4_m = (v16u8)__msa_ld_b((v16i8*)t, 0);                             \
    src5_m = (v16u8)__msa_ld_b((v16i8*)t, 16);                            \
    src6_m = (v16u8)__msa_ld_b((v16i8*)t, 32);                            \
    src7_m = (v16u8)__msa_ld_b((v16i8*)t, 48);                            \
    vec0_m = (v16u8)__msa_ilvr_b((v16i8)src0_m, (v16i8)src4_m);           \
    vec1_m = (v16u8)__msa_ilvr_b((v16i8)src1_m, (v16i8)src5_m);           \
    vec2_m = (v16u8)__msa_ilvr_b((v16i8)src2_m, (v16i8)src6_m);           \
    vec3_m = (v16u8)__msa_ilvr_b((v16i8)src3_m, (v16i8)src7_m);           \
    vec4_m = (v16u8)__msa_ilvl_b((v16i8)src0_m, (v16i8)src4_m);           \
    vec5_m = (v16u8)__msa_ilvl_b((v16i8)src1_m, (v16i8)src5_m);           \
    vec6_m = (v16u8)__msa_ilvl_b((v16i8)src2_m, (v16i8)src6_m);           \
    vec7_m = (v16u8)__msa_ilvl_b((v16i8)src3_m, (v16i8)src7_m);           \
    reg0_m = __msa_hadd_u_h(vec0_m, vec0_m);                              \
    reg1_m = __msa_hadd_u_h(vec1_m, vec1_m);                              \
    reg2_m = __msa_hadd_u_h(vec2_m, vec2_m);                              \
    reg3_m = __msa_hadd_u_h(vec3_m, vec3_m);                              \
    reg4_m = __msa_hadd_u_h(vec4_m, vec4_m);                              \
    reg5_m = __msa_hadd_u_h(vec5_m, vec5_m);                              \
    reg6_m = __msa_hadd_u_h(vec6_m, vec6_m);                              \
    reg7_m = __msa_hadd_u_h(vec7_m, vec7_m);                              \
    reg8_m = (v8u16)__msa_pckev_d((v2i64)reg4_m, (v2i64)reg0_m);          \
    reg9_m = (v8u16)__msa_pckev_d((v2i64)reg5_m, (v2i64)reg1_m);          \
    reg8_m += (v8u16)__msa_pckod_d((v2i64)reg4_m, (v2i64)reg0_m);         \
    reg9_m += (v8u16)__msa_pckod_d((v2i64)reg5_m, (v2i64)reg1_m);         \
    reg0_m = (v8u16)__msa_pckev_d((v2i64)reg6_m, (v2i64)reg2_m);          \
    reg1_m = (v8u16)__msa_pckev_d((v2i64)reg7_m, (v2i64)reg3_m);          \
    reg0_m += (v8u16)__msa_pckod_d((v2i64)reg6_m, (v2i64)reg2_m);         \
    reg1_m += (v8u16)__msa_pckod_d((v2i64)reg7_m, (v2i64)reg3_m);         \
    reg8_m = (v8u16)__msa_srai_h((v8i16)reg8_m, 2);                       \
    reg9_m = (v8u16)__msa_srai_h((v8i16)reg9_m, 2);                       \
    reg0_m = (v8u16)__msa_srai_h((v8i16)reg0_m, 2);                       \
    reg1_m = (v8u16)__msa_srai_h((v8i16)reg1_m, 2);                       \
    argb0 = (v16u8)__msa_pckev_b((v16i8)reg9_m, (v16i8)reg8_m);           \
    argb1 = (v16u8)__msa_pckev_b((v16i8)reg1_m, (v16i8)reg0_m);           \
    src0_m = (v16u8)__msa_ld_b((v16i8*)s, 64);                            \
    src1_m = (v16u8)__msa_ld_b((v16i8*)s, 80);                            \
    src2_m = (v16u8)__msa_ld_b((v16i8*)s, 96);                            \
    src3_m = (v16u8)__msa_ld_b((v16i8*)s, 112);                           \
    src4_m = (v16u8)__msa_ld_b((v16i8*)t, 64);                            \
    src5_m = (v16u8)__msa_ld_b((v16i8*)t, 80);                            \
    src6_m = (v16u8)__msa_ld_b((v16i8*)t, 96);                            \
    src7_m = (v16u8)__msa_ld_b((v16i8*)t, 112);                           \
    vec2_m = (v16u8)__msa_ilvr_b((v16i8)src0_m, (v16i8)src4_m);           \
    vec3_m = (v16u8)__msa_ilvr_b((v16i8)src1_m, (v16i8)src5_m);           \
    vec4_m = (v16u8)__msa_ilvr_b((v16i8)src2_m, (v16i8)src6_m);           \
    vec5_m = (v16u8)__msa_ilvr_b((v16i8)src3_m, (v16i8)src7_m);           \
    vec6_m = (v16u8)__msa_ilvl_b((v16i8)src0_m, (v16i8)src4_m);           \
    vec7_m = (v16u8)__msa_ilvl_b((v16i8)src1_m, (v16i8)src5_m);           \
    vec8_m = (v16u8)__msa_ilvl_b((v16i8)src2_m, (v16i8)src6_m);           \
    vec9_m = (v16u8)__msa_ilvl_b((v16i8)src3_m, (v16i8)src7_m);           \
    reg0_m = __msa_hadd_u_h(vec2_m, vec2_m);                              \
    reg1_m = __msa_hadd_u_h(vec3_m, vec3_m);                              \
    reg2_m = __msa_hadd_u_h(vec4_m, vec4_m);                              \
    reg3_m = __msa_hadd_u_h(vec5_m, vec5_m);                              \
    reg4_m = __msa_hadd_u_h(vec6_m, vec6_m);                              \
    reg5_m = __msa_hadd_u_h(vec7_m, vec7_m);                              \
    reg6_m = __msa_hadd_u_h(vec8_m, vec8_m);                              \
    reg7_m = __msa_hadd_u_h(vec9_m, vec9_m);                              \
    reg8_m = (v8u16)__msa_pckev_d((v2i64)reg4_m, (v2i64)reg0_m);          \
    reg9_m = (v8u16)__msa_pckev_d((v2i64)reg5_m, (v2i64)reg1_m);          \
    reg8_m += (v8u16)__msa_pckod_d((v2i64)reg4_m, (v2i64)reg0_m);         \
    reg9_m += (v8u16)__msa_pckod_d((v2i64)reg5_m, (v2i64)reg1_m);         \
    reg0_m = (v8u16)__msa_pckev_d((v2i64)reg6_m, (v2i64)reg2_m);          \
    reg1_m = (v8u16)__msa_pckev_d((v2i64)reg7_m, (v2i64)reg3_m);          \
    reg0_m += (v8u16)__msa_pckod_d((v2i64)reg6_m, (v2i64)reg2_m);         \
    reg1_m += (v8u16)__msa_pckod_d((v2i64)reg7_m, (v2i64)reg3_m);         \
    reg8_m = (v8u16)__msa_srai_h((v8i16)reg8_m, 2);                       \
    reg9_m = (v8u16)__msa_srai_h((v8i16)reg9_m, 2);                       \
    reg0_m = (v8u16)__msa_srai_h((v8i16)reg0_m, 2);                       \
    reg1_m = (v8u16)__msa_srai_h((v8i16)reg1_m, 2);                       \
    argb2 = (v16u8)__msa_pckev_b((v16i8)reg9_m, (v16i8)reg8_m);           \
    argb3 = (v16u8)__msa_pckev_b((v16i8)reg1_m, (v16i8)reg0_m);           \
  }

// Takes ARGB input and calculates U and V.
#define ARGBTOUV(argb0, argb1, argb2, argb3, const0, const1, const2, const3, \
                 shf0, shf1, shf2, shf3, v_out, u_out)                       \
  {                                                                          \
    v16u8 vec0_m, vec1_m, vec2_m, vec3_m, vec4_m, vec5_m, vec6_m, vec7_m;    \
    v8u16 reg0_m, reg1_m, reg2_m, reg3_m;                                    \
                                                                             \
    vec0_m = (v16u8)__msa_vshf_b(shf0, (v16i8)argb1, (v16i8)argb0);          \
    vec1_m = (v16u8)__msa_vshf_b(shf0, (v16i8)argb3, (v16i8)argb2);          \
    vec2_m = (v16u8)__msa_vshf_b(shf1, (v16i8)argb1, (v16i8)argb0);          \
    vec3_m = (v16u8)__msa_vshf_b(shf1, (v16i8)argb3, (v16i8)argb2);          \
    vec4_m = (v16u8)__msa_vshf_b(shf2, (v16i8)argb1, (v16i8)argb0);          \
    vec5_m = (v16u8)__msa_vshf_b(shf2, (v16i8)argb3, (v16i8)argb2);          \
    vec6_m = (v16u8)__msa_vshf_b(shf3, (v16i8)argb1, (v16i8)argb0);          \
    vec7_m = (v16u8)__msa_vshf_b(shf3, (v16i8)argb3, (v16i8)argb2);          \
    reg0_m = __msa_dotp_u_h(vec0_m, const1);                                 \
    reg1_m = __msa_dotp_u_h(vec1_m, const1);                                 \
    reg2_m = __msa_dotp_u_h(vec4_m, const1);                                 \
    reg3_m = __msa_dotp_u_h(vec5_m, const1);                                 \
    reg0_m += const3;                                                        \
    reg1_m += const3;                                                        \
    reg2_m += const3;                                                        \
    reg3_m += const3;                                                        \
    reg0_m -= __msa_dotp_u_h(vec2_m, const0);                                \
    reg1_m -= __msa_dotp_u_h(vec3_m, const0);                                \
    reg2_m -= __msa_dotp_u_h(vec6_m, const2);                                \
    reg3_m -= __msa_dotp_u_h(vec7_m, const2);                                \
    v_out = (v16u8)__msa_pckod_b((v16i8)reg1_m, (v16i8)reg0_m);              \
    u_out = (v16u8)__msa_pckod_b((v16i8)reg3_m, (v16i8)reg2_m);              \
  }

// Load I444 pixel data
#define READI444(psrc_y, psrc_u, psrc_v, out_y, out_u, out_v) \
  {                                                           \
    uint64_t y_m, u_m, v_m;                                   \
    v2i64 zero_m = {0};                                       \
    y_m = LD(psrc_y);                                         \
    u_m = LD(psrc_u);                                         \
    v_m = LD(psrc_v);                                         \
    out_y = (v16u8)__msa_insert_d(zero_m, 0, (int64_t)y_m);   \
    out_u = (v16u8)__msa_insert_d(zero_m, 0, (int64_t)u_m);   \
    out_v = (v16u8)__msa_insert_d(zero_m, 0, (int64_t)v_m);   \
  }

void MirrorRow_MSA(const uint8_t* src, uint8_t* dst, int width) {
  int x;
  v16u8 src0, src1, src2, src3;
  v16u8 dst0, dst1, dst2, dst3;
  v16i8 shuffler = {15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  src += width - 64;

  for (x = 0; x < width; x += 64) {
    LD_UB4(src, 16, src3, src2, src1, src0);
    VSHF_B2_UB(src3, src3, src2, src2, shuffler, shuffler, dst3, dst2);
    VSHF_B2_UB(src1, src1, src0, src0, shuffler, shuffler, dst1, dst0);
    ST_UB4(dst0, dst1, dst2, dst3, dst, 16);
    dst += 64;
    src -= 64;
  }
}

void ARGBMirrorRow_MSA(const uint8_t* src, uint8_t* dst, int width) {
  int x;
  v16u8 src0, src1, src2, src3;
  v16u8 dst0, dst1, dst2, dst3;
  v16i8 shuffler = {12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3};
  src += width * 4 - 64;

  for (x = 0; x < width; x += 16) {
    LD_UB4(src, 16, src3, src2, src1, src0);
    VSHF_B2_UB(src3, src3, src2, src2, shuffler, shuffler, dst3, dst2);
    VSHF_B2_UB(src1, src1, src0, src0, shuffler, shuffler, dst1, dst0);
    ST_UB4(dst0, dst1, dst2, dst3, dst, 16);
    dst += 64;
    src -= 64;
  }
}

void I422ToYUY2Row_MSA(const uint8_t* src_y,
                       const uint8_t* src_u,
                       const uint8_t* src_v,
                       uint8_t* dst_yuy2,
                       int width) {
  int x;
  v16u8 src_u0, src_v0, src_y0, src_y1, vec_uv0, vec_uv1;
  v16u8 dst_yuy2_0, dst_yuy2_1, dst_yuy2_2, dst_yuy2_3;

  for (x = 0; x < width; x += 32) {
    src_u0 = LD_UB(src_u);
    src_v0 = LD_UB(src_v);
    LD_UB2(src_y, 16, src_y0, src_y1);
    ILVRL_B2_UB(src_v0, src_u0, vec_uv0, vec_uv1);
    ILVRL_B2_UB(vec_uv0, src_y0, dst_yuy2_0, dst_yuy2_1);
    ILVRL_B2_UB(vec_uv1, src_y1, dst_yuy2_2, dst_yuy2_3);
    ST_UB4(dst_yuy2_0, dst_yuy2_1, dst_yuy2_2, dst_yuy2_3, dst_yuy2, 16);
    src_u += 16;
    src_v += 16;
    src_y += 32;
    dst_yuy2 += 64;
  }
}

void I422ToUYVYRow_MSA(const uint8_t* src_y,
                       const uint8_t* src_u,
                       const uint8_t* src_v,
                       uint8_t* dst_uyvy,
                       int width) {
  int x;
  v16u8 src_u0, src_v0, src_y0, src_y1, vec_uv0, vec_uv1;
  v16u8 dst_uyvy0, dst_uyvy1, dst_uyvy2, dst_uyvy3;

  for (x = 0; x < width; x += 32) {
    src_u0 = LD_UB(src_u);
    src_v0 = LD_UB(src_v);
    LD_UB2(src_y, 16, src_y0, src_y1);
    ILVRL_B2_UB(src_v0, src_u0, vec_uv0, vec_uv1);
    ILVRL_B2_UB(src_y0, vec_uv0, dst_uyvy0, dst_uyvy1);
    ILVRL_B2_UB(src_y1, vec_uv1, dst_uyvy2, dst_uyvy3);
    ST_UB4(dst_uyvy0, dst_uyvy1, dst_uyvy2, dst_uyvy3, dst_uyvy, 16);
    src_u += 16;
    src_v += 16;
    src_y += 32;
    dst_uyvy += 64;
  }
}

void I422ToARGBRow_MSA(const uint8_t* src_y,
                       const uint8_t* src_u,
                       const uint8_t* src_v,
                       uint8_t* dst_argb,
                       const struct YuvConstants* yuvconstants,
                       int width) {
  int x;
  v16u8 src0, src1, src2;
  v8i16 vec0, vec1, vec2;
  v4i32 vec_ub, vec_vr, vec_ug, vec_vg, vec_bb, vec_bg, vec_br, vec_yg;
  v4i32 vec_ubvr, vec_ugvg;
  v16u8 alpha = (v16u8)__msa_ldi_b(ALPHA_VAL);

  YUVTORGB_SETUP(yuvconstants, vec_ub, vec_vr, vec_ug, vec_vg, vec_bb, vec_bg,
                 vec_br, vec_yg);
  vec_ubvr = __msa_ilvr_w(vec_vr, vec_ub);
  vec_ugvg = (v4i32)__msa_ilvev_h((v8i16)vec_vg, (v8i16)vec_ug);

  for (x = 0; x < width; x += 8) {
    READYUV422(src_y, src_u, src_v, src0, src1, src2);
    src1 = (v16u8)__msa_ilvr_b((v16i8)src2, (v16i8)src1);
    YUVTORGB(src0, src1, vec_ubvr, vec_ugvg, vec_bb, vec_bg, vec_br, vec_yg,
             vec0, vec1, vec2);
    STOREARGB(vec0, vec1, vec2, alpha, dst_argb);
    src_y += 8;
    src_u += 4;
    src_v += 4;
    dst_argb += 32;
  }
}

void I422ToRGBARow_MSA(const uint8_t* src_y,
                       const uint8_t* src_u,
                       const uint8_t* src_v,
                       uint8_t* dst_argb,
                       const struct YuvConstants* yuvconstants,
                       int width) {
  int x;
  v16u8 src0, src1, src2;
  v8i16 vec0, vec1, vec2;
  v4i32 vec_ub, vec_vr, vec_ug, vec_vg, vec_bb, vec_bg, vec_br, vec_yg;
  v4i32 vec_ubvr, vec_ugvg;
  v16u8 alpha = (v16u8)__msa_ldi_b(ALPHA_VAL);

  YUVTORGB_SETUP(yuvconstants, vec_ub, vec_vr, vec_ug, vec_vg, vec_bb, vec_bg,
                 vec_br, vec_yg);
  vec_ubvr = __msa_ilvr_w(vec_vr, vec_ub);
  vec_ugvg = (v4i32)__msa_ilvev_h((v8i16)vec_vg, (v8i16)vec_ug);

  for (x = 0; x < width; x += 8) {
    READYUV422(src_y, src_u, src_v, src0, src1, src2);
    src1 = (v16u8)__msa_ilvr_b((v16i8)src2, (v16i8)src1);
    YUVTORGB(src0, src1, vec_ubvr, vec_ugvg, vec_bb, vec_bg, vec_br, vec_yg,
             vec0, vec1, vec2);
    STOREARGB(alpha, vec0, vec1, vec2, dst_argb);
    src_y += 8;
    src_u += 4;
    src_v += 4;
    dst_argb += 32;
  }
}

void I422AlphaToARGBRow_MSA(const uint8_t* src_y,
                            const uint8_t* src_u,
                            const uint8_t* src_v,
                            const uint8_t* src_a,
                            uint8_t* dst_argb,
                            const struct YuvConstants* yuvconstants,
                            int width) {
  int x;
  int64_t data_a;
  v16u8 src0, src1, src2, src3;
  v8i16 vec0, vec1, vec2;
  v4i32 vec_ub, vec_vr, vec_ug, vec_vg, vec_bb, vec_bg, vec_br, vec_yg;
  v4i32 vec_ubvr, vec_ugvg;
  v4i32 zero = {0};

  YUVTORGB_SETUP(yuvconstants, vec_ub, vec_vr, vec_ug, vec_vg, vec_bb, vec_bg,
                 vec_br, vec_yg);
  vec_ubvr = __msa_ilvr_w(vec_vr, vec_ub);
  vec_ugvg = (v4i32)__msa_ilvev_h((v8i16)vec_vg, (v8i16)vec_ug);

  for (x = 0; x < width; x += 8) {
    data_a = LD(src_a);
    READYUV422(src_y, src_u, src_v, src0, src1, src2);
    src1 = (v16u8)__msa_ilvr_b((v16i8)src2, (v16i8)src1);
    src3 = (v16u8)__msa_insert_d((v2i64)zero, 0, data_a);
    YUVTORGB(src0, src1, vec_ubvr, vec_ugvg, vec_bb, vec_bg, vec_br, vec_yg,
             vec0, vec1, vec2);
    src3 = (v16u8)__msa_ilvr_b((v16i8)src3, (v16i8)src3);
    STOREARGB(vec0, vec1, vec2, src3, dst_argb);
    src_y += 8;
    src_u += 4;
    src_v += 4;
    src_a += 8;
    dst_argb += 32;
  }
}

void I422ToRGB24Row_MSA(const uint8_t* src_y,
                        const uint8_t* src_u,
                        const uint8_t* src_v,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int32_t width) {
  int x;
  int64_t data_u, data_v;
  v16u8 src0, src1, src2, src3, src4, dst0, dst1, dst2;
  v8i16 vec0, vec1, vec2, vec3, vec4, vec5;
  v4i32 vec_ub, vec_vr, vec_ug, vec_vg, vec_bb, vec_bg, vec_br, vec_yg;
  v4i32 vec_ubvr, vec_ugvg;
  v16u8 reg0, reg1, reg2, reg3;
  v2i64 zero = {0};
  v16i8 shuffler0 = {0, 1, 16, 2, 3, 17, 4, 5, 18, 6, 7, 19, 8, 9, 20, 10};
  v16i8 shuffler1 = {0, 21, 1, 2, 22, 3, 4, 23, 5, 6, 24, 7, 8, 25, 9, 10};
  v16i8 shuffler2 = {26, 6,  7,  27, 8,  9,  28, 10,
                     11, 29, 12, 13, 30, 14, 15, 31};

  YUVTORGB_SETUP(yuvconstants, vec_ub, vec_vr, vec_ug, vec_vg, vec_bb, vec_bg,
                 vec_br, vec_yg);
  vec_ubvr = __msa_ilvr_w(vec_vr, vec_ub);
  vec_ugvg = (v4i32)__msa_ilvev_h((v8i16)vec_vg, (v8i16)vec_ug);

  for (x = 0; x < width; x += 16) {
    src0 = (v16u8)__msa_ld_b((v16u8*)src_y, 0);
    data_u = LD(src_u);
    data_v = LD(src_v);
    src1 = (v16u8)__msa_insert_d(zero, 0, data_u);
    src2 = (v16u8)__msa_insert_d(zero, 0, data_v);
    src1 = (v16u8)__msa_ilvr_b((v16i8)src2, (v16i8)src1);
    src3 = (v16u8)__msa_sldi_b((v16i8)src0, (v16i8)src0, 8);
    src4 = (v16u8)__msa_sldi_b((v16i8)src1, (v16i8)src1, 8);
    YUVTORGB(src0, src1, vec_ubvr, vec_ugvg, vec_bb, vec_bg, vec_br, vec_yg,
             vec0, vec1, vec2);
    YUVTORGB(src3, src4, vec_ubvr, vec_ugvg, vec_bb, vec_bg, vec_br, vec_yg,
             vec3, vec4, vec5);
    reg0 = (v16u8)__msa_ilvev_b((v16i8)vec1, (v16i8)vec0);
    reg2 = (v16u8)__msa_ilvev_b((v16i8)vec4, (v16i8)vec3);
    reg3 = (v16u8)__msa_pckev_b((v16i8)vec5, (v16i8)vec2);
    reg1 = (v16u8)__msa_sldi_b((v16i8)reg2, (v16i8)reg0, 11);
    dst0 = (v16u8)__msa_vshf_b(shuffler0, (v16i8)reg3, (v16i8)reg0);
    dst1 = (v16u8)__msa_vshf_b(shuffler1, (v16i8)reg3, (v16i8)reg1);
    dst2 = (v16u8)__msa_vshf_b(shuffler2, (v16i8)reg3, (v16i8)reg2);
    ST_UB2(dst0, dst1, dst_argb, 16);
    ST_UB(dst2, (dst_argb + 32));
    src_y += 16;
    src_u += 8;
    src_v += 8;
    dst_argb += 48;
  }
}

// TODO(fbarchard): Consider AND instead of shift to isolate 5 upper bits of R.
void I422ToRGB565Row_MSA(const uint8_t* src_y,
                         const uint8_t* src_u,
                         const uint8_t* src_v,
                         uint8_t* dst_rgb565,
                         const struct YuvConstants* yuvconstants,
                         int width) {
  int x;
  v16u8 src0, src1, src2, dst0;
  v8i16 vec0, vec1, vec2;
  v4i32 vec_ub, vec_vr, vec_ug, vec_vg, vec_bb, vec_bg, vec_br, vec_yg;
  v4i32 vec_ubvr, vec_ugvg;

  YUVTORGB_SETUP(yuvconstants, vec_ub, vec_vr, vec_ug, vec_vg, vec_bb, vec_bg,
                 vec_br, vec_yg);
  vec_ubvr = __msa_ilvr_w(vec_vr, vec_ub);
  vec_ugvg = (v4i32)__msa_ilvev_h((v8i16)vec_vg, (v8i16)vec_ug);

  for (x = 0; x < width; x += 8) {
    READYUV422(src_y, src_u, src_v, src0, src1, src2);
    src1 = (v16u8)__msa_ilvr_b((v16i8)src2, (v16i8)src1);
    YUVTORGB(src0, src1, vec_ubvr, vec_ugvg, vec_bb, vec_bg, vec_br, vec_yg,
             vec0, vec2, vec1);
    vec0 = __msa_srai_h(vec0, 3);
    vec1 = __msa_srai_h(vec1, 3);
    vec2 = __msa_srai_h(vec2, 2);
    vec1 = __msa_slli_h(vec1, 11);
    vec2 = __msa_slli_h(vec2, 5);
    vec0 |= vec1;
    dst0 = (v16u8)(vec2 | vec0);
    ST_UB(dst0, dst_rgb565);
    src_y += 8;
    src_u += 4;
    src_v += 4;
    dst_rgb565 += 16;
  }
}

// TODO(fbarchard): Consider AND instead of shift to isolate 4 upper bits of G.
void I422ToARGB4444Row_MSA(const uint8_t* src_y,
                           const uint8_t* src_u,
                           const uint8_t* src_v,
                           uint8_t* dst_argb4444,
                           const struct YuvConstants* yuvconstants,
                           int width) {
  int x;
  v16u8 src0, src1, src2, dst0;
  v8i16 vec0, vec1, vec2;
  v8u16 reg0, reg1, reg2;
  v4i32 vec_ub, vec_vr, vec_ug, vec_vg, vec_bb, vec_bg, vec_br, vec_yg;
  v4i32 vec_ubvr, vec_ugvg;
  v8u16 const_0xF000 = (v8u16)__msa_fill_h(0xF000);

  YUVTORGB_SETUP(yuvconstants, vec_ub, vec_vr, vec_ug, vec_vg, vec_bb, vec_bg,
                 vec_br, vec_yg);
  vec_ubvr = __msa_ilvr_w(vec_vr, vec_ub);
  vec_ugvg = (v4i32)__msa_ilvev_h((v8i16)vec_vg, (v8i16)vec_ug);

  for (x = 0; x < width; x += 8) {
    READYUV422(src_y, src_u, src_v, src0, src1, src2);
    src1 = (v16u8)__msa_ilvr_b((v16i8)src2, (v16i8)src1);
    YUVTORGB(src0, src1, vec_ubvr, vec_ugvg, vec_bb, vec_bg, vec_br, vec_yg,
             vec0, vec1, vec2);
    reg0 = (v8u16)__msa_srai_h(vec0, 4);
    reg1 = (v8u16)__msa_srai_h(vec1, 4);
    reg2 = (v8u16)__msa_srai_h(vec2, 4);
    reg1 = (v8u16)__msa_slli_h((v8i16)reg1, 4);
    reg2 = (v8u16)__msa_slli_h((v8i16)reg2, 8);
    reg1 |= const_0xF000;
    reg0 |= reg2;
    dst0 = (v16u8)(reg1 | reg0);
    ST_UB(dst0, dst_argb4444);
    src_y += 8;
    src_u += 4;
    src_v += 4;
    dst_argb4444 += 16;
  }
}

void I422ToARGB1555Row_MSA(const uint8_t* src_y,
                           const uint8_t* src_u,
                           const uint8_t* src_v,
                           uint8_t* dst_argb1555,
                           const struct YuvConstants* yuvconstants,
                           int width) {
  int x;
  v16u8 src0, src1, src2, dst0;
  v8i16 vec0, vec1, vec2;
  v8u16 reg0, reg1, reg2;
  v4i32 vec_ub, vec_vr, vec_ug, vec_vg, vec_bb, vec_bg, vec_br, vec_yg;
  v4i32 vec_ubvr, vec_ugvg;
  v8u16 const_0x8000 = (v8u16)__msa_fill_h(0x8000);

  YUVTORGB_SETUP(yuvconstants, vec_ub, vec_vr, vec_ug, vec_vg, vec_bb, vec_bg,
                 vec_br, vec_yg);
  vec_ubvr = __msa_ilvr_w(vec_vr, vec_ub);
  vec_ugvg = (v4i32)__msa_ilvev_h((v8i16)vec_vg, (v8i16)vec_ug);

  for (x = 0; x < width; x += 8) {
    READYUV422(src_y, src_u, src_v, src0, src1, src2);
    src1 = (v16u8)__msa_ilvr_b((v16i8)src2, (v16i8)src1);
    YUVTORGB(src0, src1, vec_ubvr, vec_ugvg, vec_bb, vec_bg, vec_br, vec_yg,
             vec0, vec1, vec2);
    reg0 = (v8u16)__msa_srai_h(vec0, 3);
    reg1 = (v8u16)__msa_srai_h(vec1, 3);
    reg2 = (v8u16)__msa_srai_h(vec2, 3);
    reg1 = (v8u16)__msa_slli_h((v8i16)reg1, 5);
    reg2 = (v8u16)__msa_slli_h((v8i16)reg2, 10);
    reg1 |= const_0x8000;
    reg0 |= reg2;
    dst0 = (v16u8)(reg1 | reg0);
    ST_UB(dst0, dst_argb1555);
    src_y += 8;
    src_u += 4;
    src_v += 4;
    dst_argb1555 += 16;
  }
}

void YUY2ToYRow_MSA(const uint8_t* src_yuy2, uint8_t* dst_y, int width) {
  int x;
  v16u8 src0, src1, src2, src3, dst0, dst1;

  for (x = 0; x < width; x += 32) {
    LD_UB4(src_yuy2, 16, src0, src1, src2, src3);
    dst0 = (v16u8)__msa_pckev_b((v16i8)src1, (v16i8)src0);
    dst1 = (v16u8)__msa_pckev_b((v16i8)src3, (v16i8)src2);
    ST_UB2(dst0, dst1, dst_y, 16);
    src_yuy2 += 64;
    dst_y += 32;
  }
}

void YUY2ToUVRow_MSA(const uint8_t* src_yuy2,
                     int src_stride_yuy2,
                     uint8_t* dst_u,
                     uint8_t* dst_v,
                     int width) {
  const uint8_t* src_yuy2_next = src_yuy2 + src_stride_yuy2;
  int x;
  v16u8 src0, src1, src2, src3, src4, src5, src6, src7;
  v16u8 vec0, vec1, dst0, dst1;

  for (x = 0; x < width; x += 32) {
    LD_UB4(src_yuy2, 16, src0, src1, src2, src3);
    LD_UB4(src_yuy2_next, 16, src4, src5, src6, src7);
    src0 = (v16u8)__msa_pckod_b((v16i8)src1, (v16i8)src0);
    src1 = (v16u8)__msa_pckod_b((v16i8)src3, (v16i8)src2);
    src2 = (v16u8)__msa_pckod_b((v16i8)src5, (v16i8)src4);
    src3 = (v16u8)__msa_pckod_b((v16i8)src7, (v16i8)src6);
    vec0 = __msa_aver_u_b(src0, src2);
    vec1 = __msa_aver_u_b(src1, src3);
    dst0 = (v16u8)__msa_pckev_b((v16i8)vec1, (v16i8)vec0);
    dst1 = (v16u8)__msa_pckod_b((v16i8)vec1, (v16i8)vec0);
    ST_UB(dst0, dst_u);
    ST_UB(dst1, dst_v);
    src_yuy2 += 64;
    src_yuy2_next += 64;
    dst_u += 16;
    dst_v += 16;
  }
}

void YUY2ToUV422Row_MSA(const uint8_t* src_yuy2,
                        uint8_t* dst_u,
                        uint8_t* dst_v,
                        int width) {
  int x;
  v16u8 src0, src1, src2, src3, dst0, dst1;

  for (x = 0; x < width; x += 32) {
    LD_UB4(src_yuy2, 16, src0, src1, src2, src3);
    src0 = (v16u8)__msa_pckod_b((v16i8)src1, (v16i8)src0);
    src1 = (v16u8)__msa_pckod_b((v16i8)src3, (v16i8)src2);
    dst0 = (v16u8)__msa_pckev_b((v16i8)src1, (v16i8)src0);
    dst1 = (v16u8)__msa_pckod_b((v16i8)src1, (v16i8)src0);
    ST_UB(dst0, dst_u);
    ST_UB(dst1, dst_v);
    src_yuy2 += 64;
    dst_u += 16;
    dst_v += 16;
  }
}

void UYVYToYRow_MSA(const uint8_t* src_uyvy, uint8_t* dst_y, int width) {
  int x;
  v16u8 src0, src1, src2, src3, dst0, dst1;

  for (x = 0; x < width; x += 32) {
    LD_UB4(src_uyvy, 16, src0, src1, src2, src3);
    dst0 = (v16u8)__msa_pckod_b((v16i8)src1, (v16i8)src0);
    dst1 = (v16u8)__msa_pckod_b((v16i8)src3, (v16i8)src2);
    ST_UB2(dst0, dst1, dst_y, 16);
    src_uyvy += 64;
    dst_y += 32;
  }
}

void UYVYToUVRow_MSA(const uint8_t* src_uyvy,
                     int src_stride_uyvy,
                     uint8_t* dst_u,
                     uint8_t* dst_v,
                     int width) {
  const uint8_t* src_uyvy_next = src_uyvy + src_stride_uyvy;
  int x;
  v16u8 src0, src1, src2, src3, src4, src5, src6, src7;
  v16u8 vec0, vec1, dst0, dst1;

  for (x = 0; x < width; x += 32) {
    LD_UB4(src_uyvy, 16, src0, src1, src2, src3);
    LD_UB4(src_uyvy_next, 16, src4, src5, src6, src7);
    src0 = (v16u8)__msa_pckev_b((v16i8)src1, (v16i8)src0);
    src1 = (v16u8)__msa_pckev_b((v16i8)src3, (v16i8)src2);
    src2 = (v16u8)__msa_pckev_b((v16i8)src5, (v16i8)src4);
    src3 = (v16u8)__msa_pckev_b((v16i8)src7, (v16i8)src6);
    vec0 = __msa_aver_u_b(src0, src2);
    vec1 = __msa_aver_u_b(src1, src3);
    dst0 = (v16u8)__msa_pckev_b((v16i8)vec1, (v16i8)vec0);
    dst1 = (v16u8)__msa_pckod_b((v16i8)vec1, (v16i8)vec0);
    ST_UB(dst0, dst_u);
    ST_UB(dst1, dst_v);
    src_uyvy += 64;
    src_uyvy_next += 64;
    dst_u += 16;
    dst_v += 16;
  }
}

void UYVYToUV422Row_MSA(const uint8_t* src_uyvy,
                        uint8_t* dst_u,
                        uint8_t* dst_v,
                        int width) {
  int x;
  v16u8 src0, src1, src2, src3, dst0, dst1;

  for (x = 0; x < width; x += 32) {
    LD_UB4(src_uyvy, 16, src0, src1, src2, src3);
    src0 = (v16u8)__msa_pckev_b((v16i8)src1, (v16i8)src0);
    src1 = (v16u8)__msa_pckev_b((v16i8)src3, (v16i8)src2);
    dst0 = (v16u8)__msa_pckev_b((v16i8)src1, (v16i8)src0);
    dst1 = (v16u8)__msa_pckod_b((v16i8)src1, (v16i8)src0);
    ST_UB(dst0, dst_u);
    ST_UB(dst1, dst_v);
    src_uyvy += 64;
    dst_u += 16;
    dst_v += 16;
  }
}

void ARGBToYRow_MSA(const uint8_t* src_argb0, uint8_t* dst_y, int width) {
  int x;
  v16u8 src0, src1, src2, src3, vec0, vec1, vec2, vec3, dst0;
  v8u16 reg0, reg1, reg2, reg3, reg4, reg5;
  v16i8 zero = {0};
  v8u16 const_0x19 = (v8u16)__msa_ldi_h(0x19);
  v8u16 const_0x81 = (v8u16)__msa_ldi_h(0x81);
  v8u16 const_0x42 = (v8u16)__msa_ldi_h(0x42);
  v8u16 const_0x1080 = (v8u16)__msa_fill_h(0x1080);

  for (x = 0; x < width; x += 16) {
    src0 = (v16u8)__msa_ld_b((v16u8*)src_argb0, 0);
    src1 = (v16u8)__msa_ld_b((v16u8*)src_argb0, 16);
    src2 = (v16u8)__msa_ld_b((v16u8*)src_argb0, 32);
    src3 = (v16u8)__msa_ld_b((v16u8*)src_argb0, 48);
    vec0 = (v16u8)__msa_pckev_b((v16i8)src1, (v16i8)src0);
    vec1 = (v16u8)__msa_pckev_b((v16i8)src3, (v16i8)src2);
    vec2 = (v16u8)__msa_pckod_b((v16i8)src1, (v16i8)src0);
    vec3 = (v16u8)__msa_pckod_b((v16i8)src3, (v16i8)src2);
    reg0 = (v8u16)__msa_ilvev_b(zero, (v16i8)vec0);
    reg1 = (v8u16)__msa_ilvev_b(zero, (v16i8)vec1);
    reg2 = (v8u16)__msa_ilvev_b(zero, (v16i8)vec2);
    reg3 = (v8u16)__msa_ilvev_b(zero, (v16i8)vec3);
    reg4 = (v8u16)__msa_ilvod_b(zero, (v16i8)vec0);
    reg5 = (v8u16)__msa_ilvod_b(zero, (v16i8)vec1);
    reg0 *= const_0x19;
    reg1 *= const_0x19;
    reg2 *= const_0x81;
    reg3 *= const_0x81;
    reg4 *= const_0x42;
    reg5 *= const_0x42;
    reg0 += reg2;
    reg1 += reg3;
    reg0 += reg4;
    reg1 += reg5;
    reg0 += const_0x1080;
    reg1 += const_0x1080;
    reg0 = (v8u16)__msa_srai_h((v8i16)reg0, 8);
    reg1 = (v8u16)__msa_srai_h((v8i16)reg1, 8);
    dst0 = (v16u8)__msa_pckev_b((v16i8)reg1, (v16i8)reg0);
    ST_UB(dst0, dst_y);
    src_argb0 += 64;
    dst_y += 16;
  }
}

void ARGBToUVRow_MSA(const uint8_t* src_argb0,
                     int src_stride_argb,
                     uint8_t* dst_u,
                     uint8_t* dst_v,
                     int width) {
  int x;
  const uint8_t* src_argb0_next = src_argb0 + src_stride_argb;
  v16u8 src0, src1, src2, src3, src4, src5, src6, src7;
  v16u8 vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, vec9;
  v8u16 reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8, reg9;
  v16u8 dst0, dst1;
  v8u16 const_0x70 = (v8u16)__msa_ldi_h(0x70);
  v8u16 const_0x4A = (v8u16)__msa_ldi_h(0x4A);
  v8u16 const_0x26 = (v8u16)__msa_ldi_h(0x26);
  v8u16 const_0x5E = (v8u16)__msa_ldi_h(0x5E);
  v8u16 const_0x12 = (v8u16)__msa_ldi_h(0x12);
  v8u16 const_0x8080 = (v8u16)__msa_fill_h(0x8080);

  for (x = 0; x < width; x += 32) {
    src0 = (v16u8)__msa_ld_b((v16u8*)src_argb0, 0);
    src1 = (v16u8)__msa_ld_b((v16u8*)src_argb0, 16);
    src2 = (v16u8)__msa_ld_b((v16u8*)src_argb0, 32);
    src3 = (v16u8)__msa_ld_b((v16u8*)src_argb0, 48);
    src4 = (v16u8)__msa_ld_b((v16u8*)src_argb0, 64);
    src5 = (v16u8)__msa_ld_b((v16u8*)src_argb0, 80);
    src6 = (v16u8)__msa_ld_b((v16u8*)src_argb0, 96);
    src7 = (v16u8)__msa_ld_b((v16u8*)src_argb0, 112);
    vec0 = (v16u8)__msa_pckev_b((v16i8)src1, (v16i8)src0);
    vec1 = (v16u8)__msa_pckev_b((v16i8)src3, (v16i8)src2);
    vec2 = (v16u8)__msa_pckev_b((v16i8)src5, (v16i8)src4);
    vec3 = (v16u8)__msa_pckev_b((v16i8)src7, (v16i8)src6);
    vec4 = (v16u8)__msa_pckod_b((v16i8)src1, (v16i8)src0);
    vec5 = (v16u8)__msa_pckod_b((v16i8)src3, (v16i8)src2);
    vec6 = (v16u8)__msa_pckod_b((v16i8)src5, (v16i8)src4);
    vec7 = (v16u8)__msa_pckod_b((v16i8)src7, (v16i8)src6);
    vec8 = (v16u8)__msa_pckev_b((v16i8)vec1, (v16i8)vec0);
    vec9 = (v16u8)__msa_pckev_b((v16i8)vec3, (v16i8)vec2);
    vec4 = (v16u8)__msa_pckev_b((v16i8)vec5, (v16i8)vec4);
    vec5 = (v16u8)__msa_pckev_b((v16i8)vec7, (v16i8)vec6);
    vec0 = (v16u8)__msa_pckod_b((v16i8)vec1, (v16i8)vec0);
    vec1 = (v16u8)__msa_pckod_b((v16i8)vec3, (v16i8)vec2);
    reg0 = __msa_hadd_u_h(vec8, vec8);
    reg1 = __msa_hadd_u_h(vec9, vec9);
    reg2 = __msa_hadd_u_h(vec4, vec4);
    reg3 = __msa_hadd_u_h(vec5, vec5);
    reg4 = __msa_hadd_u_h(vec0, vec0);
    reg5 = __msa_hadd_u_h(vec1, vec1);
    src0 = (v16u8)__msa_ld_b((v16u8*)src_argb0_next, 0);
    src1 = (v16u8)__msa_ld_b((v16u8*)src_argb0_next, 16);
    src2 = (v16u8)__msa_ld_b((v16u8*)src_argb0_next, 32);
    src3 = (v16u8)__msa_ld_b((v16u8*)src_argb0_next, 48);
    src4 = (v16u8)__msa_ld_b((v16u8*)src_argb0_next, 64);
    src5 = (v16u8)__msa_ld_b((v16u8*)src_argb0_next, 80);
    src6 = (v16u8)__msa_ld_b((v16u8*)src_argb0_next, 96);
    src7 = (v16u8)__msa_ld_b((v16u8*)src_argb0_next, 112);
    vec0 = (v16u8)__msa_pckev_b((v16i8)src1, (v16i8)src0);
    vec1 = (v16u8)__msa_pckev_b((v16i8)src3, (v16i8)src2);
    vec2 = (v16u8)__msa_pckev_b((v16i8)src5, (v16i8)src4);
    vec3 = (v16u8)__msa_pckev_b((v16i8)src7, (v16i8)src6);
    vec4 = (v16u8)__msa_pckod_b((v16i8)src1, (v16i8)src0);
    vec5 = (v16u8)__msa_pckod_b((v16i8)src3, (v16i8)src2);
    vec6 = (v16u8)__msa_pckod_b((v16i8)src5, (v16i8)src4);
    vec7 = (v16u8)__msa_pckod_b((v16i8)src7, (v16i8)src6);
    vec8 = (v16u8)__msa_pckev_b((v16i8)vec1, (v16i8)vec0);
    vec9 = (v16u8)__msa_pckev_b((v16i8)vec3, (v16i8)vec2);
    vec4 = (v16u8)__msa_pckev_b((v16i8)vec5, (v16i8)vec4);
    vec5 = (v16u8)__msa_pckev_b((v16i8)vec7, (v16i8)vec6);
    vec0 = (v16u8)__msa_pckod_b((v16i8)vec1, (v16i8)vec0);
    vec1 = (v16u8)__msa_pckod_b((v16i8)vec3, (v16i8)vec2);
    reg0 += __msa_hadd_u_h(vec8, vec8);
    reg1 += __msa_hadd_u_h(vec9, vec9);
    reg2 += __msa_hadd_u_h(vec4, vec4);
    reg3 += __msa_hadd_u_h(vec5, vec5);
    reg4 += __msa_hadd_u_h(vec0, vec0);
    reg5 += __msa_hadd_u_h(vec1, vec1);
    reg0 = (v8u16)__msa_srai_h((v8i16)reg0, 2);
    reg1 = (v8u16)__msa_srai_h((v8i16)reg1, 2);
    reg2 = (v8u16)__msa_srai_h((v8i16)reg2, 2);
    reg3 = (v8u16)__msa_srai_h((v8i16)reg3, 2);
    reg4 = (v8u16)__msa_srai_h((v8i16)reg4, 2);
    reg5 = (v8u16)__msa_srai_h((v8i16)reg5, 2);
    reg6 = reg0 * const_0x70;
    reg7 = reg1 * const_0x70;
    reg8 = reg2 * const_0x4A;
    reg9 = reg3 * const_0x4A;
    reg6 += const_0x8080;
    reg7 += const_0x8080;
    reg8 += reg4 * const_0x26;
    reg9 += reg5 * const_0x26;
    reg0 *= const_0x12;
    reg1 *= const_0x12;
    reg2 *= const_0x5E;
    reg3 *= const_0x5E;
    reg4 *= const_0x70;
    reg5 *= const_0x70;
    reg2 += reg0;
    reg3 += reg1;
    reg4 += const_0x8080;
    reg5 += const_0x8080;
    reg6 -= reg8;
    reg7 -= reg9;
    reg4 -= reg2;
    reg5 -= reg3;
    reg6 = (v8u16)__msa_srai_h((v8i16)reg6, 8);
    reg7 = (v8u16)__msa_srai_h((v8i16)reg7, 8);
    reg4 = (v8u16)__msa_srai_h((v8i16)reg4, 8);
    reg5 = (v8u16)__msa_srai_h((v8i16)reg5, 8);
    dst0 = (v16u8)__msa_pckev_b((v16i8)reg7, (v16i8)reg6);
    dst1 = (v16u8)__msa_pckev_b((v16i8)reg5, (v16i8)reg4);
    ST_UB(dst0, dst_u);
    ST_UB(dst1, dst_v);
    src_argb0 += 128;
    src_argb0_next += 128;
    dst_u += 16;
    dst_v += 16;
  }
}

void ARGBToRGB24Row_MSA(const uint8_t* src_argb, uint8_t* dst_rgb, int width) {
  int x;
  v16u8 src0, src1, src2, src3, dst0, dst1, dst2;
  v16i8 shuffler0 = {0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20};
  v16i8 shuffler1 = {5,  6,  8,  9,  10, 12, 13, 14,
                     16, 17, 18, 20, 21, 22, 24, 25};
  v16i8 shuffler2 = {10, 12, 13, 14, 16, 17, 18, 20,
                     21, 22, 24, 25, 26, 28, 29, 30};

  for (x = 0; x < width; x += 16) {
    src0 = (v16u8)__msa_ld_b((const v16i8*)src_argb, 0);
    src1 = (v16u8)__msa_ld_b((const v16i8*)src_argb, 16);
    src2 = (v16u8)__msa_ld_b((const v16i8*)src_argb, 32);
    src3 = (v16u8)__msa_ld_b((const v16i8*)src_argb, 48);
    dst0 = (v16u8)__msa_vshf_b(shuffler0, (v16i8)src1, (v16i8)src0);
    dst1 = (v16u8)__msa_vshf_b(shuffler1, (v16i8)src2, (v16i8)src1);
    dst2 = (v16u8)__msa_vshf_b(shuffler2, (v16i8)src3, (v16i8)src2);
    ST_UB2(dst0, dst1, dst_rgb, 16);
    ST_UB(dst2, (dst_rgb + 32));
    src_argb += 64;
    dst_rgb += 48;
  }
}

void ARGBToRAWRow_MSA(const uint8_t* src_argb, uint8_t* dst_rgb, int width) {
  int x;
  v16u8 src0, src1, src2, src3, dst0, dst1, dst2;
  v16i8 shuffler0 = {2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, 18, 17, 16, 22};
  v16i8 shuffler1 = {5,  4,  10, 9,  8,  14, 13, 12,
                     18, 17, 16, 22, 21, 20, 26, 25};
  v16i8 shuffler2 = {8,  14, 13, 12, 18, 17, 16, 22,
                     21, 20, 26, 25, 24, 30, 29, 28};

  for (x = 0; x < width; x += 16) {
    src0 = (v16u8)__msa_ld_b((const v16i8*)src_argb, 0);
    src1 = (v16u8)__msa_ld_b((const v16i8*)src_argb, 16);
    src2 = (v16u8)__msa_ld_b((const v16i8*)src_argb, 32);
    src3 = (v16u8)__msa_ld_b((const v16i8*)src_argb, 48);
    dst0 = (v16u8)__msa_vshf_b(shuffler0, (v16i8)src1, (v16i8)src0);
    dst1 = (v16u8)__msa_vshf_b(shuffler1, (v16i8)src2, (v16i8)src1);
    dst2 = (v16u8)__msa_vshf_b(shuffler2, (v16i8)src3, (v16i8)src2);
    ST_UB2(dst0, dst1, dst_rgb, 16);
    ST_UB(dst2, (dst_rgb + 32));
    src_argb += 64;
    dst_rgb += 48;
  }
}

void ARGBToRGB565Row_MSA(const uint8_t* src_argb, uint8_t* dst_rgb, int width) {
  int x;
  v16u8 src0, src1, dst0;
  v16u8 vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7;
  v16i8 zero = {0};

  for (x = 0; x < width; x += 8) {
    src0 = (v16u8)__msa_ld_b((const v16i8*)src_argb, 0);
    src1 = (v16u8)__msa_ld_b((const v16i8*)src_argb, 16);
    vec0 = (v16u8)__msa_srai_b((v16i8)src0, 3);
    vec1 = (v16u8)__msa_slli_b((v16i8)src0, 3);
    vec2 = (v16u8)__msa_srai_b((v16i8)src0, 5);
    vec4 = (v16u8)__msa_srai_b((v16i8)src1, 3);
    vec5 = (v16u8)__msa_slli_b((v16i8)src1, 3);
    vec6 = (v16u8)__msa_srai_b((v16i8)src1, 5);
    vec1 = (v16u8)__msa_sldi_b(zero, (v16i8)vec1, 1);
    vec2 = (v16u8)__msa_sldi_b(zero, (v16i8)vec2, 1);
    vec5 = (v16u8)__msa_sldi_b(zero, (v16i8)vec5, 1);
    vec6 = (v16u8)__msa_sldi_b(zero, (v16i8)vec6, 1);
    vec3 = (v16u8)__msa_sldi_b(zero, (v16i8)src0, 2);
    vec7 = (v16u8)__msa_sldi_b(zero, (v16i8)src1, 2);
    vec0 = __msa_binsli_b(vec0, vec1, 2);
    vec1 = __msa_binsli_b(vec2, vec3, 4);
    vec4 = __msa_binsli_b(vec4, vec5, 2);
    vec5 = __msa_binsli_b(vec6, vec7, 4);
    vec0 = (v16u8)__msa_ilvev_b((v16i8)vec1, (v16i8)vec0);
    vec4 = (v16u8)__msa_ilvev_b((v16i8)vec5, (v16i8)vec4);
    dst0 = (v16u8)__msa_pckev_h((v8i16)vec4, (v8i16)vec0);
    ST_UB(dst0, dst_rgb);
    src_argb += 32;
    dst_rgb += 16;
  }
}

void ARGBToARGB1555Row_MSA(const uint8_t* src_argb,
                           uint8_t* dst_rgb,
                           int width) {
  int x;
  v16u8 src0, src1, dst0;
  v16u8 vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, vec9;
  v16i8 zero = {0};

  for (x = 0; x < width; x += 8) {
    src0 = (v16u8)__msa_ld_b((const v16i8*)src_argb, 0);
    src1 = (v16u8)__msa_ld_b((const v16i8*)src_argb, 16);
    vec0 = (v16u8)__msa_srai_b((v16i8)src0, 3);
    vec1 = (v16u8)__msa_slli_b((v16i8)src0, 2);
    vec2 = (v16u8)__msa_srai_b((v16i8)vec0, 3);
    vec1 = (v16u8)__msa_sldi_b(zero, (v16i8)vec1, 1);
    vec2 = (v16u8)__msa_sldi_b(zero, (v16i8)vec2, 1);
    vec3 = (v16u8)__msa_srai_b((v16i8)src0, 1);
    vec5 = (v16u8)__msa_srai_b((v16i8)src1, 3);
    vec6 = (v16u8)__msa_slli_b((v16i8)src1, 2);
    vec7 = (v16u8)__msa_srai_b((v16i8)vec5, 3);
    vec6 = (v16u8)__msa_sldi_b(zero, (v16i8)vec6, 1);
    vec7 = (v16u8)__msa_sldi_b(zero, (v16i8)vec7, 1);
    vec8 = (v16u8)__msa_srai_b((v16i8)src1, 1);
    vec3 = (v16u8)__msa_sldi_b(zero, (v16i8)vec3, 2);
    vec8 = (v16u8)__msa_sldi_b(zero, (v16i8)vec8, 2);
    vec4 = (v16u8)__msa_sldi_b(zero, (v16i8)src0, 3);
    vec9 = (v16u8)__msa_sldi_b(zero, (v16i8)src1, 3);
    vec0 = __msa_binsli_b(vec0, vec1, 2);
    vec5 = __msa_binsli_b(vec5, vec6, 2);
    vec1 = __msa_binsli_b(vec2, vec3, 5);
    vec6 = __msa_binsli_b(vec7, vec8, 5);
    vec1 = __msa_binsli_b(vec1, vec4, 0);
    vec6 = __msa_binsli_b(vec6, vec9, 0);
    vec0 = (v16u8)__msa_ilvev_b((v16i8)vec1, (v16i8)vec0);
    vec1 = (v16u8)__msa_ilvev_b((v16i8)vec6, (v16i8)vec5);
    dst0 = (v16u8)__msa_pckev_h((v8i16)vec1, (v8i16)vec0);
    ST_UB(dst0, dst_rgb);
    src_argb += 32;
    dst_rgb += 16;
  }
}

void ARGBToARGB4444Row_MSA(const uint8_t* src_argb,
                           uint8_t* dst_rgb,
                           int width) {
  int x;
  v16u8 src0, src1;
  v16u8 vec0, vec1;
  v16u8 dst0;
  v16i8 zero = {0};

  for (x = 0; x < width; x += 8) {
    src0 = (v16u8)__msa_ld_b((const v16i8*)src_argb, 0);
    src1 = (v16u8)__msa_ld_b((const v16i8*)src_argb, 16);
    vec0 = (v16u8)__msa_srai_b((v16i8)src0, 4);
    vec1 = (v16u8)__msa_srai_b((v16i8)src1, 4);
    src0 = (v16u8)__msa_sldi_b(zero, (v16i8)src0, 1);
    src1 = (v16u8)__msa_sldi_b(zero, (v16i8)src1, 1);
    vec0 = __msa_binsli_b(vec0, src0, 3);
    vec1 = __msa_binsli_b(vec1, src1, 3);
    dst0 = (v16u8)__msa_pckev_b((v16i8)vec1, (v16i8)vec0);
    ST_UB(dst0, dst_rgb);
    src_argb += 32;
    dst_rgb += 16;
  }
}

void ARGBToUV444Row_MSA(const uint8_t* src_argb,
                        uint8_t* dst_u,
                        uint8_t* dst_v,
                        int32_t width) {
  int32_t x;
  v16u8 src0, src1, src2, src3, reg0, reg1, reg2, reg3, dst0, dst1;
  v8u16 vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7;
  v8u16 vec8, vec9, vec10, vec11;
  v8u16 const_112 = (v8u16)__msa_ldi_h(112);
  v8u16 const_74 = (v8u16)__msa_ldi_h(74);
  v8u16 const_38 = (v8u16)__msa_ldi_h(38);
  v8u16 const_94 = (v8u16)__msa_ldi_h(94);
  v8u16 const_18 = (v8u16)__msa_ldi_h(18);
  v8u16 const_32896 = (v8u16)__msa_fill_h(32896);
  v16i8 zero = {0};

  for (x = width; x > 0; x -= 16) {
    src0 = (v16u8)__msa_ld_b((const v16i8*)src_argb, 0);
    src1 = (v16u8)__msa_ld_b((const v16i8*)src_argb, 16);
    src2 = (v16u8)__msa_ld_b((const v16i8*)src_argb, 32);
    src3 = (v16u8)__msa_ld_b((const v16i8*)src_argb, 48);
    reg0 = (v16u8)__msa_pckev_b((v16i8)src1, (v16i8)src0);
    reg1 = (v16u8)__msa_pckev_b((v16i8)src3, (v16i8)src2);
    reg2 = (v16u8)__msa_pckod_b((v16i8)src1, (v16i8)src0);
    reg3 = (v16u8)__msa_pckod_b((v16i8)src3, (v16i8)src2);
    src0 = (v16u8)__msa_pckev_b((v16i8)reg1, (v16i8)reg0);
    src1 = (v16u8)__msa_pckev_b((v16i8)reg3, (v16i8)reg2);
    src2 = (v16u8)__msa_pckod_b((v16i8)reg1, (v16i8)reg0);
    vec0 = (v8u16)__msa_ilvr_b(zero, (v16i8)src0);
    vec1 = (v8u16)__msa_ilvl_b(zero, (v16i8)src0);
    vec2 = (v8u16)__msa_ilvr_b(zero, (v16i8)src1);
    vec3 = (v8u16)__msa_ilvl_b(zero, (v16i8)src1);
    vec4 = (v8u16)__msa_ilvr_b(zero, (v16i8)src2);
    vec5 = (v8u16)__msa_ilvl_b(zero, (v16i8)src2);
    vec10 = vec0 * const_18;
    vec11 = vec1 * const_18;
    vec8 = vec2 * const_94;
    vec9 = vec3 * const_94;
    vec6 = vec4 * const_112;
    vec7 = vec5 * const_112;
    vec0 *= const_112;
    vec1 *= const_112;
    vec2 *= const_74;
    vec3 *= const_74;
    vec4 *= const_38;
    vec5 *= const_38;
    vec8 += vec10;
    vec9 += vec11;
    vec6 += const_32896;
    vec7 += const_32896;
    vec0 += const_32896;
    vec1 += const_32896;
    vec2 += vec4;
    vec3 += vec5;
    vec0 -= vec2;
    vec1 -= vec3;
    vec6 -= vec8;
    vec7 -= vec9;
    vec0 = (v8u16)__msa_srai_h((v8i16)vec0, 8);
    vec1 = (v8u16)__msa_srai_h((v8i16)vec1, 8);
    vec6 = (v8u16)__msa_srai_h((v8i16)vec6, 8);
    vec7 = (v8u16)__msa_srai_h((v8i16)vec7, 8);
    dst0 = (v16u8)__msa_pckev_b((v16i8)vec1, (v16i8)vec0);
    dst1 = (v16u8)__msa_pckev_b((v16i8)vec7, (v16i8)vec6);
    ST_UB(dst0, dst_u);
    ST_UB(dst1, dst_v);
    src_argb += 64;
    dst_u += 16;
    dst_v += 16;
  }
}

void ARGBMultiplyRow_MSA(const uint8_t* src_argb0,
                         const uint8_t* src_argb1,
                         uint8_t* dst_argb,
                         int width) {
  int x;
  v16u8 src0, src1, dst0;
  v8u16 vec0, vec1, vec2, vec3;
  v4u32 reg0, reg1, reg2, reg3;
  v8i16 zero = {0};

  for (x = 0; x < width; x += 4) {
    src0 = (v16u8)__msa_ld_b((const v16i8*)src_argb0, 0);
    src1 = (v16u8)__msa_ld_b((const v16i8*)src_argb1, 0);
    vec0 = (v8u16)__msa_ilvr_b((v16i8)src0, (v16i8)src0);
    vec1 = (v8u16)__msa_ilvl_b((v16i8)src0, (v16i8)src0);
    vec2 = (v8u16)__msa_ilvr_b((v16i8)zero, (v16i8)src1);
    vec3 = (v8u16)__msa_ilvl_b((v16i8)zero, (v16i8)src1);
    reg0 = (v4u32)__msa_ilvr_h(zero, (v8i16)vec0);
    reg1 = (v4u32)__msa_ilvl_h(zero, (v8i16)vec0);
    reg2 = (v4u32)__msa_ilvr_h(zero, (v8i16)vec1);
    reg3 = (v4u32)__msa_ilvl_h(zero, (v8i16)vec1);
    reg0 *= (v4u32)__msa_ilvr_h(zero, (v8i16)vec2);
    reg1 *= (v4u32)__msa_ilvl_h(zero, (v8i16)vec2);
    reg2 *= (v4u32)__msa_ilvr_h(zero, (v8i16)vec3);
    reg3 *= (v4u32)__msa_ilvl_h(zero, (v8i16)vec3);
    reg0 = (v4u32)__msa_srai_w((v4i32)reg0, 16);
    reg1 = (v4u32)__msa_srai_w((v4i32)reg1, 16);
    reg2 = (v4u32)__msa_srai_w((v4i32)reg2, 16);
    reg3 = (v4u32)__msa_srai_w((v4i32)reg3, 16);
    vec0 = (v8u16)__msa_pckev_h((v8i16)reg1, (v8i16)reg0);
    vec1 = (v8u16)__msa_pckev_h((v8i16)reg3, (v8i16)reg2);
    dst0 = (v16u8)__msa_pckev_b((v16i8)vec1, (v16i8)vec0);
    ST_UB(dst0, dst_argb);
    src_argb0 += 16;
    src_argb1 += 16;
    dst_argb += 16;
  }
}

void ARGBAddRow_MSA(const uint8_t* src_argb0,
                    const uint8_t* src_argb1,
                    uint8_t* dst_argb,
                    int width) {
  int x;
  v16u8 src0, src1, src2, src3, dst0, dst1;

  for (x = 0; x < width; x += 8) {
    src0 = (v16u8)__msa_ld_b((const v16i8*)src_argb0, 0);
    src1 = (v16u8)__msa_ld_b((const v16i8*)src_argb0, 16);
    src2 = (v16u8)__msa_ld_b((const v16i8*)src_argb1, 0);
    src3 = (v16u8)__msa_ld_b((const v16i8*)src_argb1, 16);
    dst0 = __msa_adds_u_b(src0, src2);
    dst1 = __msa_adds_u_b(src1, src3);
    ST_UB2(dst0, dst1, dst_argb, 16);
    src_argb0 += 32;
    src_argb1 += 32;
    dst_argb += 32;
  }
}

void ARGBSubtractRow_MSA(const uint8_t* src_argb0,
                         const uint8_t* src_argb1,
                         uint8_t* dst_argb,
                         int width) {
  int x;
  v16u8 src0, src1, src2, src3, dst0, dst1;

  for (x = 0; x < width; x += 8) {
    src0 = (v16u8)__msa_ld_b((const v16i8*)src_argb0, 0);
    src1 = (v16u8)__msa_ld_b((const v16i8*)src_argb0, 16);
    src2 = (v16u8)__msa_ld_b((const v16i8*)src_argb1, 0);
    src3 = (v16u8)__msa_ld_b((const v16i8*)src_argb1, 16);
    dst0 = __msa_subs_u_b(src0, src2);
    dst1 = __msa_subs_u_b(src1, src3);
    ST_UB2(dst0, dst1, dst_argb, 16);
    src_argb0 += 32;
    src_argb1 += 32;
    dst_argb += 32;
  }
}

void ARGBAttenuateRow_MSA(const uint8_t* src_argb,
                          uint8_t* dst_argb,
                          int width) {
  int x;
  v16u8 src0, src1, dst0, dst1;
  v8u16 vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, vec9;
  v4u32 reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7;
  v8i16 zero = {0};
  v16u8 mask = {0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255};

  for (x = 0; x < width; x += 8) {
    src0 = (v16u8)__msa_ld_b((const v16i8*)src_argb, 0);
    src1 = (v16u8)__msa_ld_b((const v16i8*)src_argb, 16);
    vec0 = (v8u16)__msa_ilvr_b((v16i8)src0, (v16i8)src0);
    vec1 = (v8u16)__msa_ilvl_b((v16i8)src0, (v16i8)src0);
    vec2 = (v8u16)__msa_ilvr_b((v16i8)src1, (v16i8)src1);
    vec3 = (v8u16)__msa_ilvl_b((v16i8)src1, (v16i8)src1);
    vec4 = (v8u16)__msa_fill_h(vec0[3]);
    vec5 = (v8u16)__msa_fill_h(vec0[7]);
    vec6 = (v8u16)__msa_fill_h(vec1[3]);
    vec7 = (v8u16)__msa_fill_h(vec1[7]);
    vec4 = (v8u16)__msa_pckev_d((v2i64)vec5, (v2i64)vec4);
    vec5 = (v8u16)__msa_pckev_d((v2i64)vec7, (v2i64)vec6);
    vec6 = (v8u16)__msa_fill_h(vec2[3]);
    vec7 = (v8u16)__msa_fill_h(vec2[7]);
    vec8 = (v8u16)__msa_fill_h(vec3[3]);
    vec9 = (v8u16)__msa_fill_h(vec3[7]);
    vec6 = (v8u16)__msa_pckev_d((v2i64)vec7, (v2i64)vec6);
    vec7 = (v8u16)__msa_pckev_d((v2i64)vec9, (v2i64)vec8);
    reg0 = (v4u32)__msa_ilvr_h(zero, (v8i16)vec4);
    reg1 = (v4u32)__msa_ilvl_h(zero, (v8i16)vec4);
    reg2 = (v4u32)__msa_ilvr_h(zero, (v8i16)vec5);
    reg3 = (v4u32)__msa_ilvl_h(zero, (v8i16)vec5);
    reg4 = (v4u32)__msa_ilvr_h(zero, (v8i16)vec6);
    reg5 = (v4u32)__msa_ilvl_h(zero, (v8i16)vec6);
    reg6 = (v4u32)__msa_ilvr_h(zero, (v8i16)vec7);
    reg7 = (v4u32)__msa_ilvl_h(zero, (v8i16)vec7);
    reg0 *= (v4u32)__msa_ilvr_h(zero, (v8i16)vec0);
    reg1 *= (v4u32)__msa_ilvl_h(zero, (v8i16)vec0);
    reg2 *= (v4u32)__msa_ilvr_h(zero, (v8i16)vec1);
    reg3 *= (v4u32)__msa_ilvl_h(zero, (v8i16)vec1);
    reg4 *= (v4u32)__msa_ilvr_h(zero, (v8i16)vec2);
    reg5 *= (v4u32)__msa_ilvl_h(zero, (v8i16)vec2);
    reg6 *= (v4u32)__msa_ilvr_h(zero, (v8i16)vec3);
    reg7 *= (v4u32)__msa_ilvl_h(zero, (v8i16)vec3);
    reg0 = (v4u32)__msa_srai_w((v4i32)reg0, 24);
    reg1 = (v4u32)__msa_srai_w((v4i32)reg1, 24);
    reg2 = (v4u32)__msa_srai_w((v4i32)reg2, 24);
    reg3 = (v4u32)__msa_srai_w((v4i32)reg3, 24);
    reg4 = (v4u32)__msa_srai_w((v4i32)reg4, 24);
    reg5 = (v4u32)__msa_srai_w((v4i32)reg5, 24);
    reg6 = (v4u32)__msa_srai_w((v4i32)reg6, 24);
    reg7 = (v4u32)__msa_srai_w((v4i32)reg7, 24);
    vec0 = (v8u16)__msa_pckev_h((v8i16)reg1, (v8i16)reg0);
    vec1 = (v8u16)__msa_pckev_h((v8i16)reg3, (v8i16)reg2);
    vec2 = (v8u16)__msa_pckev_h((v8i16)reg5, (v8i16)reg4);
    vec3 = (v8u16)__msa_pckev_h((v8i16)reg7, (v8i16)reg6);
    dst0 = (v16u8)__msa_pckev_b((v16i8)vec1, (v16i8)vec0);
    dst1 = (v16u8)__msa_pckev_b((v16i8)vec3, (v16i8)vec2);
    dst0 = __msa_bmnz_v(dst0, src0, mask);
    dst1 = __msa_bmnz_v(dst1, src1, mask);
    ST_UB2(dst0, dst1, dst_argb, 16);
    src_argb += 32;
    dst_argb += 32;
  }
}

void ARGBToRGB565DitherRow_MSA(const uint8_t* src_argb,
                               uint8_t* dst_rgb,
                               uint32_t dither4,
                               int width) {
  int x;
  v16u8 src0, src1, dst0, vec0, vec1;
  v8i16 vec_d0;
  v8i16 reg0, reg1, reg2;
  v16i8 zero = {0};
  v8i16 max = __msa_ldi_h(0xFF);

  vec_d0 = (v8i16)__msa_fill_w(dither4);
  vec_d0 = (v8i16)__msa_ilvr_b(zero, (v16i8)vec_d0);

  for (x = 0; x < width; x += 8) {
    src0 = (v16u8)__msa_ld_b((const v16i8*)src_argb, 0);
    src1 = (v16u8)__msa_ld_b((const v16i8*)src_argb, 16);
    vec0 = (v16u8)__msa_pckev_b((v16i8)src1, (v16i8)src0);
    vec1 = (v16u8)__msa_pckod_b((v16i8)src1, (v16i8)src0);
    reg0 = (v8i16)__msa_ilvev_b(zero, (v16i8)vec0);
    reg1 = (v8i16)__msa_ilvev_b(zero, (v16i8)vec1);
    reg2 = (v8i16)__msa_ilvod_b(zero, (v16i8)vec0);
    reg0 += vec_d0;
    reg1 += vec_d0;
    reg2 += vec_d0;
    reg0 = __msa_maxi_s_h((v8i16)reg0, 0);
    reg1 = __msa_maxi_s_h((v8i16)reg1, 0);
    reg2 = __msa_maxi_s_h((v8i16)reg2, 0);
    reg0 = __msa_min_s_h((v8i16)max, (v8i16)reg0);
    reg1 = __msa_min_s_h((v8i16)max, (v8i16)reg1);
    reg2 = __msa_min_s_h((v8i16)max, (v8i16)reg2);
    reg0 = __msa_srai_h(reg0, 3);
    reg2 = __msa_srai_h(reg2, 3);
    reg1 = __msa_srai_h(reg1, 2);
    reg2 = __msa_slli_h(reg2, 11);
    reg1 = __msa_slli_h(reg1, 5);
    reg0 |= reg1;
    dst0 = (v16u8)(reg0 | reg2);
    ST_UB(dst0, dst_rgb);
    src_argb += 32;
    dst_rgb += 16;
  }
}

void ARGBShuffleRow_MSA(const uint8_t* src_argb,
                        uint8_t* dst_argb,
                        const uint8_t* shuffler,
                        int width) {
  int x;
  v16u8 src0, src1, dst0, dst1;
  v16i8 vec0;
  v16i8 shuffler_vec = {0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12};
  int32_t val = LW((int32_t*)shuffler);

  vec0 = (v16i8)__msa_fill_w(val);
  shuffler_vec += vec0;

  for (x = 0; x < width; x += 8) {
    src0 = (v16u8)__msa_ld_b((const v16u8*)src_argb, 0);
    src1 = (v16u8)__msa_ld_b((const v16u8*)src_argb, 16);
    dst0 = (v16u8)__msa_vshf_b(shuffler_vec, (v16i8)src0, (v16i8)src0);
    dst1 = (v16u8)__msa_vshf_b(shuffler_vec, (v16i8)src1, (v16i8)src1);
    ST_UB2(dst0, dst1, dst_argb, 16);
    src_argb += 32;
    dst_argb += 32;
  }
}

void ARGBShadeRow_MSA(const uint8_t* src_argb,
                      uint8_t* dst_argb,
                      int width,
                      uint32_t value) {
  int x;
  v16u8 src0, dst0;
  v8u16 vec0, vec1;
  v4u32 reg0, reg1, reg2, reg3, rgba_scale;
  v8i16 zero = {0};

  rgba_scale[0] = value;
  rgba_scale = (v4u32)__msa_ilvr_b((v16i8)rgba_scale, (v16i8)rgba_scale);
  rgba_scale = (v4u32)__msa_ilvr_h(zero, (v8i16)rgba_scale);

  for (x = 0; x < width; x += 4) {
    src0 = (v16u8)__msa_ld_b((const v16u8*)src_argb, 0);
    vec0 = (v8u16)__msa_ilvr_b((v16i8)src0, (v16i8)src0);
    vec1 = (v8u16)__msa_ilvl_b((v16i8)src0, (v16i8)src0);
    reg0 = (v4u32)__msa_ilvr_h(zero, (v8i16)vec0);
    reg1 = (v4u32)__msa_ilvl_h(zero, (v8i16)vec0);
    reg2 = (v4u32)__msa_ilvr_h(zero, (v8i16)vec1);
    reg3 = (v4u32)__msa_ilvl_h(zero, (v8i16)vec1);
    reg0 *= rgba_scale;
    reg1 *= rgba_scale;
    reg2 *= rgba_scale;
    reg3 *= rgba_scale;
    reg0 = (v4u32)__msa_srai_w((v4i32)reg0, 24);
    reg1 = (v4u32)__msa_srai_w((v4i32)reg1, 24);
    reg2 = (v4u32)__msa_srai_w((v4i32)reg2, 24);
    reg3 = (v4u32)__msa_srai_w((v4i32)reg3, 24);
    vec0 = (v8u16)__msa_pckev_h((v8i16)reg1, (v8i16)reg0);
    vec1 = (v8u16)__msa_pckev_h((v8i16)reg3, (v8i16)reg2);
    dst0 = (v16u8)__msa_pckev_b((v16i8)vec1, (v16i8)vec0);
    ST_UB(dst0, dst_argb);
    src_argb += 16;
    dst_argb += 16;
  }
}

void ARGBGrayRow_MSA(const uint8_t* src_argb, uint8_t* dst_argb, int width) {
  int x;
  v16u8 src0, src1, vec0, vec1, dst0, dst1;
  v8u16 reg0;
  v16u8 const_0x26 = (v16u8)__msa_ldi_h(0x26);
  v16u8 const_0x4B0F = (v16u8)__msa_fill_h(0x4B0F);

  for (x = 0; x < width; x += 8) {
    src0 = (v16u8)__msa_ld_b((const v16u8*)src_argb, 0);
    src1 = (v16u8)__msa_ld_b((const v16u8*)src_argb, 16);
    vec0 = (v16u8)__msa_pckev_h((v8i16)src1, (v8i16)src0);
    vec1 = (v16u8)__msa_pckod_h((v8i16)src1, (v8i16)src0);
    reg0 = __msa_dotp_u_h(vec0, const_0x4B0F);
    reg0 = __msa_dpadd_u_h(reg0, vec1, const_0x26);
    reg0 = (v8u16)__msa_srari_h((v8i16)reg0, 7);
    vec0 = (v16u8)__msa_ilvev_b((v16i8)reg0, (v16i8)reg0);
    vec1 = (v16u8)__msa_ilvod_b((v16i8)vec1, (v16i8)vec0);
    dst0 = (v16u8)__msa_ilvr_b((v16i8)vec1, (v16i8)vec0);
    dst1 = (v16u8)__msa_ilvl_b((v16i8)vec1, (v16i8)vec0);
    ST_UB2(dst0, dst1, dst_argb, 16);
    src_argb += 32;
    dst_argb += 32;
  }
}

void ARGBSepiaRow_MSA(uint8_t* dst_argb, int width) {
  int x;
  v16u8 src0, src1, dst0, dst1, vec0, vec1, vec2, vec3, vec4, vec5;
  v8u16 reg0, reg1, reg2;
  v16u8 const_0x4411 = (v16u8)__msa_fill_h(0x4411);
  v16u8 const_0x23 = (v16u8)__msa_ldi_h(0x23);
  v16u8 const_0x5816 = (v16u8)__msa_fill_h(0x5816);
  v16u8 const_0x2D = (v16u8)__msa_ldi_h(0x2D);
  v16u8 const_0x6218 = (v16u8)__msa_fill_h(0x6218);
  v16u8 const_0x32 = (v16u8)__msa_ldi_h(0x32);
  v8u16 const_0xFF = (v8u16)__msa_ldi_h(0xFF);

  for (x = 0; x < width; x += 8) {
    src0 = (v16u8)__msa_ld_b((v16u8*)dst_argb, 0);
    src1 = (v16u8)__msa_ld_b((v16u8*)dst_argb, 16);
    vec0 = (v16u8)__msa_pckev_h((v8i16)src1, (v8i16)src0);
    vec1 = (v16u8)__msa_pckod_h((v8i16)src1, (v8i16)src0);
    vec3 = (v16u8)__msa_pckod_b((v16i8)vec1, (v16i8)vec1);
    reg0 = (v8u16)__msa_dotp_u_h(vec0, const_0x4411);
    reg1 = (v8u16)__msa_dotp_u_h(vec0, const_0x5816);
    reg2 = (v8u16)__msa_dotp_u_h(vec0, const_0x6218);
    reg0 = (v8u16)__msa_dpadd_u_h(reg0, vec1, const_0x23);
    reg1 = (v8u16)__msa_dpadd_u_h(reg1, vec1, const_0x2D);
    reg2 = (v8u16)__msa_dpadd_u_h(reg2, vec1, const_0x32);
    reg0 = (v8u16)__msa_srai_h((v8i16)reg0, 7);
    reg1 = (v8u16)__msa_srai_h((v8i16)reg1, 7);
    reg2 = (v8u16)__msa_srai_h((v8i16)reg2, 7);
    reg1 = (v8u16)__msa_min_u_h((v8u16)reg1, const_0xFF);
    reg2 = (v8u16)__msa_min_u_h((v8u16)reg2, const_0xFF);
    vec0 = (v16u8)__msa_pckev_b((v16i8)reg0, (v16i8)reg0);
    vec1 = (v16u8)__msa_pckev_b((v16i8)reg1, (v16i8)reg1);
    vec2 = (v16u8)__msa_pckev_b((v16i8)reg2, (v16i8)reg2);
    vec4 = (v16u8)__msa_ilvr_b((v16i8)vec2, (v16i8)vec0);
    vec5 = (v16u8)__msa_ilvr_b((v16i8)vec3, (v16i8)vec1);
    dst0 = (v16u8)__msa_ilvr_b((v16i8)vec5, (v16i8)vec4);
    dst1 = (v16u8)__msa_ilvl_b((v16i8)vec5, (v16i8)vec4);
    ST_UB2(dst0, dst1, dst_argb, 16);
    dst_argb += 32;
  }
}

void ARGB4444ToARGBRow_MSA(const uint8_t* src_argb4444,
                           uint8_t* dst_argb,
                           int width) {
  int x;
  v16u8 src0, src1;
  v8u16 vec0, vec1, vec2, vec3;
  v16u8 dst0, dst1, dst2, dst3;

  for (x = 0; x < width; x += 16) {
    src0 = (v16u8)__msa_ld_b((const v16u8*)src_argb4444, 0);
    src1 = (v16u8)__msa_ld_b((const v16u8*)src_argb4444, 16);
    vec0 = (v8u16)__msa_andi_b(src0, 0x0F);
    vec1 = (v8u16)__msa_andi_b(src1, 0x0F);
    vec2 = (v8u16)__msa_andi_b(src0, 0xF0);
    vec3 = (v8u16)__msa_andi_b(src1, 0xF0);
    vec0 |= (v8u16)__msa_slli_b((v16i8)vec0, 4);
    vec1 |= (v8u16)__msa_slli_b((v16i8)vec1, 4);
    vec2 |= (v8u16)__msa_srli_b((v16i8)vec2, 4);
    vec3 |= (v8u16)__msa_srli_b((v16i8)vec3, 4);
    dst0 = (v16u8)__msa_ilvr_b((v16i8)vec2, (v16i8)vec0);
    dst1 = (v16u8)__msa_ilvl_b((v16i8)vec2, (v16i8)vec0);
    dst2 = (v16u8)__msa_ilvr_b((v16i8)vec3, (v16i8)vec1);
    dst3 = (v16u8)__msa_ilvl_b((v16i8)vec3, (v16i8)vec1);
    ST_UB4(dst0, dst1, dst2, dst3, dst_argb, 16);
    src_argb4444 += 32;
    dst_argb += 64;
  }
}

void ARGB1555ToARGBRow_MSA(const uint8_t* src_argb1555,
                           uint8_t* dst_argb,
                           int width) {
  int x;
  v8u16 src0, src1;
  v8u16 vec0, vec1, vec2, vec3, vec4, vec5;
  v16u8 reg0, reg1, reg2, reg3, reg4, reg5, reg6;
  v16u8 dst0, dst1, dst2, dst3;
  v8u16 const_0x1F = (v8u16)__msa_ldi_h(0x1F);

  for (x = 0; x < width; x += 16) {
    src0 = (v8u16)__msa_ld_h((const v8u16*)src_argb1555, 0);
    src1 = (v8u16)__msa_ld_h((const v8u16*)src_argb1555, 16);
    vec0 = src0 & const_0x1F;
    vec1 = src1 & const_0x1F;
    src0 = (v8u16)__msa_srli_h((v8i16)src0, 5);
    src1 = (v8u16)__msa_srli_h((v8i16)src1, 5);
    vec2 = src0 & const_0x1F;
    vec3 = src1 & const_0x1F;
    src0 = (v8u16)__msa_srli_h((v8i16)src0, 5);
    src1 = (v8u16)__msa_srli_h((v8i16)src1, 5);
    vec4 = src0 & const_0x1F;
    vec5 = src1 & const_0x1F;
    src0 = (v8u16)__msa_srli_h((v8i16)src0, 5);
    src1 = (v8u16)__msa_srli_h((v8i16)src1, 5);
    reg0 = (v16u8)__msa_pckev_b((v16i8)vec1, (v16i8)vec0);
    reg1 = (v16u8)__msa_pckev_b((v16i8)vec3, (v16i8)vec2);
    reg2 = (v16u8)__msa_pckev_b((v16i8)vec5, (v16i8)vec4);
    reg3 = (v16u8)__msa_pckev_b((v16i8)src1, (v16i8)src0);
    reg4 = (v16u8)__msa_slli_b((v16i8)reg0, 3);
    reg5 = (v16u8)__msa_slli_b((v16i8)reg1, 3);
    reg6 = (v16u8)__msa_slli_b((v16i8)reg2, 3);
    reg4 |= (v16u8)__msa_srai_b((v16i8)reg0, 2);
    reg5 |= (v16u8)__msa_srai_b((v16i8)reg1, 2);
    reg6 |= (v16u8)__msa_srai_b((v16i8)reg2, 2);
    reg3 = -reg3;
    reg0 = (v16u8)__msa_ilvr_b((v16i8)reg6, (v16i8)reg4);
    reg1 = (v16u8)__msa_ilvl_b((v16i8)reg6, (v16i8)reg4);
    reg2 = (v16u8)__msa_ilvr_b((v16i8)reg3, (v16i8)reg5);
    reg3 = (v16u8)__msa_ilvl_b((v16i8)reg3, (v16i8)reg5);
    dst0 = (v16u8)__msa_ilvr_b((v16i8)reg2, (v16i8)reg0);
    dst1 = (v16u8)__msa_ilvl_b((v16i8)reg2, (v16i8)reg0);
    dst2 = (v16u8)__msa_ilvr_b((v16i8)reg3, (v16i8)reg1);
    dst3 = (v16u8)__msa_ilvl_b((v16i8)reg3, (v16i8)reg1);
    ST_UB4(dst0, dst1, dst2, dst3, dst_argb, 16);
    src_argb1555 += 32;
    dst_argb += 64;
  }
}

void RGB565ToARGBRow_MSA(const uint8_t* src_rgb565,
                         uint8_t* dst_argb,
                         int width) {
  int x;
  v8u16 src0, src1, vec0, vec1, vec2, vec3, vec4, vec5;
  v8u16 reg0, reg1, reg2, reg3, reg4, reg5;
  v16u8 res0, res1, res2, res3, dst0, dst1, dst2, dst3;
  v16u8 alpha = (v16u8)__msa_ldi_b(ALPHA_VAL);
  v8u16 const_0x1F = (v8u16)__msa_ldi_h(0x1F);
  v8u16 const_0x7E0 = (v8u16)__msa_fill_h(0x7E0);
  v8u16 const_0xF800 = (v8u16)__msa_fill_h(0xF800);

  for (x = 0; x < width; x += 16) {
    src0 = (v8u16)__msa_ld_h((const v8u16*)src_rgb565, 0);
    src1 = (v8u16)__msa_ld_h((const v8u16*)src_rgb565, 16);
    vec0 = src0 & const_0x1F;
    vec1 = src0 & const_0x7E0;
    vec2 = src0 & const_0xF800;
    vec3 = src1 & const_0x1F;
    vec4 = src1 & const_0x7E0;
    vec5 = src1 & const_0xF800;
    reg0 = (v8u16)__msa_slli_h((v8i16)vec0, 3);
    reg1 = (v8u16)__msa_srli_h((v8i16)vec1, 3);
    reg2 = (v8u16)__msa_srli_h((v8i16)vec2, 8);
    reg3 = (v8u16)__msa_slli_h((v8i16)vec3, 3);
    reg4 = (v8u16)__msa_srli_h((v8i16)vec4, 3);
    reg5 = (v8u16)__msa_srli_h((v8i16)vec5, 8);
    reg0 |= (v8u16)__msa_srli_h((v8i16)vec0, 2);
    reg1 |= (v8u16)__msa_srli_h((v8i16)vec1, 9);
    reg2 |= (v8u16)__msa_srli_h((v8i16)vec2, 13);
    reg3 |= (v8u16)__msa_srli_h((v8i16)vec3, 2);
    reg4 |= (v8u16)__msa_srli_h((v8i16)vec4, 9);
    reg5 |= (v8u16)__msa_srli_h((v8i16)vec5, 13);
    res0 = (v16u8)__msa_ilvev_b((v16i8)reg2, (v16i8)reg0);
    res1 = (v16u8)__msa_ilvev_b((v16i8)alpha, (v16i8)reg1);
    res2 = (v16u8)__msa_ilvev_b((v16i8)reg5, (v16i8)reg3);
    res3 = (v16u8)__msa_ilvev_b((v16i8)alpha, (v16i8)reg4);
    dst0 = (v16u8)__msa_ilvr_b((v16i8)res1, (v16i8)res0);
    dst1 = (v16u8)__msa_ilvl_b((v16i8)res1, (v16i8)res0);
    dst2 = (v16u8)__msa_ilvr_b((v16i8)res3, (v16i8)res2);
    dst3 = (v16u8)__msa_ilvl_b((v16i8)res3, (v16i8)res2);
    ST_UB4(dst0, dst1, dst2, dst3, dst_argb, 16);
    src_rgb565 += 32;
    dst_argb += 64;
  }
}

void RGB24ToARGBRow_MSA(const uint8_t* src_rgb24,
                        uint8_t* dst_argb,
                        int width) {
  int x;
  v16u8 src0, src1, src2;
  v16u8 vec0, vec1, vec2;
  v16u8 dst0, dst1, dst2, dst3;
  v16u8 alpha = (v16u8)__msa_ldi_b(ALPHA_VAL);
  v16i8 shuffler = {0, 1, 2, 16, 3, 4, 5, 17, 6, 7, 8, 18, 9, 10, 11, 19};

  for (x = 0; x < width; x += 16) {
    src0 = (v16u8)__msa_ld_b((const v16i8*)src_rgb24, 0);
    src1 = (v16u8)__msa_ld_b((const v16i8*)src_rgb24, 16);
    src2 = (v16u8)__msa_ld_b((const v16i8*)src_rgb24, 32);
    vec0 = (v16u8)__msa_sldi_b((v16i8)src1, (v16i8)src0, 12);
    vec1 = (v16u8)__msa_sldi_b((v16i8)src2, (v16i8)src1, 8);
    vec2 = (v16u8)__msa_sldi_b((v16i8)src2, (v16i8)src2, 4);
    dst0 = (v16u8)__msa_vshf_b(shuffler, (v16i8)alpha, (v16i8)src0);
    dst1 = (v16u8)__msa_vshf_b(shuffler, (v16i8)alpha, (v16i8)vec0);
    dst2 = (v16u8)__msa_vshf_b(shuffler, (v16i8)alpha, (v16i8)vec1);
    dst3 = (v16u8)__msa_vshf_b(shuffler, (v16i8)alpha, (v16i8)vec2);
    ST_UB4(dst0, dst1, dst2, dst3, dst_argb, 16);
    src_rgb24 += 48;
    dst_argb += 64;
  }
}

void RAWToARGBRow_MSA(const uint8_t* src_raw, uint8_t* dst_argb, int width) {
  int x;
  v16u8 src0, src1, src2;
  v16u8 vec0, vec1, vec2;
  v16u8 dst0, dst1, dst2, dst3;
  v16u8 alpha = (v16u8)__msa_ldi_b(ALPHA_VAL);
  v16i8 mask = {2, 1, 0, 16, 5, 4, 3, 17, 8, 7, 6, 18, 11, 10, 9, 19};

  for (x = 0; x < width; x += 16) {
    src0 = (v16u8)__msa_ld_b((const v16i8*)src_raw, 0);
    src1 = (v16u8)__msa_ld_b((const v16i8*)src_raw, 16);
    src2 = (v16u8)__msa_ld_b((const v16i8*)src_raw, 32);
    vec0 = (v16u8)__msa_sldi_b((v16i8)src1, (v16i8)src0, 12);
    vec1 = (v16u8)__msa_sldi_b((v16i8)src2, (v16i8)src1, 8);
    vec2 = (v16u8)__msa_sldi_b((v16i8)src2, (v16i8)src2, 4);
    dst0 = (v16u8)__msa_vshf_b(mask, (v16i8)alpha, (v16i8)src0);
    dst1 = (v16u8)__msa_vshf_b(mask, (v16i8)alpha, (v16i8)vec0);
    dst2 = (v16u8)__msa_vshf_b(mask, (v16i8)alpha, (v16i8)vec1);
    dst3 = (v16u8)__msa_vshf_b(mask, (v16i8)alpha, (v16i8)vec2);
    ST_UB4(dst0, dst1, dst2, dst3, dst_argb, 16);
    src_raw += 48;
    dst_argb += 64;
  }
}

void ARGB1555ToYRow_MSA(const uint8_t* src_argb1555,
                        uint8_t* dst_y,
                        int width) {
  int x;
  v8u16 src0, src1, vec0, vec1, vec2, vec3, vec4, vec5;
  v8u16 reg0, reg1, reg2, reg3, reg4, reg5;
  v16u8 dst0;
  v8u16 const_0x19 = (v8u16)__msa_ldi_h(0x19);
  v8u16 const_0x81 = (v8u16)__msa_ldi_h(0x81);
  v8u16 const_0x42 = (v8u16)__msa_ldi_h(0x42);
  v8u16 const_0x1F = (v8u16)__msa_ldi_h(0x1F);
  v8u16 const_0x1080 = (v8u16)__msa_fill_h(0x1080);

  for (x = 0; x < width; x += 16) {
    src0 = (v8u16)__msa_ld_b((const v8i16*)src_argb1555, 0);
    src1 = (v8u16)__msa_ld_b((const v8i16*)src_argb1555, 16);
    vec0 = src0 & const_0x1F;
    vec1 = src1 & const_0x1F;
    src0 = (v8u16)__msa_srai_h((v8i16)src0, 5);
    src1 = (v8u16)__msa_srai_h((v8i16)src1, 5);
    vec2 = src0 & const_0x1F;
    vec3 = src1 & const_0x1F;
    src0 = (v8u16)__msa_srai_h((v8i16)src0, 5);
    src1 = (v8u16)__msa_srai_h((v8i16)src1, 5);
    vec4 = src0 & const_0x1F;
    vec5 = src1 & const_0x1F;
    reg0 = (v8u16)__msa_slli_h((v8i16)vec0, 3);
    reg1 = (v8u16)__msa_slli_h((v8i16)vec1, 3);
    reg0 |= (v8u16)__msa_srai_h((v8i16)vec0, 2);
    reg1 |= (v8u16)__msa_srai_h((v8i16)vec1, 2);
    reg2 = (v8u16)__msa_slli_h((v8i16)vec2, 3);
    reg3 = (v8u16)__msa_slli_h((v8i16)vec3, 3);
    reg2 |= (v8u16)__msa_srai_h((v8i16)vec2, 2);
    reg3 |= (v8u16)__msa_srai_h((v8i16)vec3, 2);
    reg4 = (v8u16)__msa_slli_h((v8i16)vec4, 3);
    reg5 = (v8u16)__msa_slli_h((v8i16)vec5, 3);
    reg4 |= (v8u16)__msa_srai_h((v8i16)vec4, 2);
    reg5 |= (v8u16)__msa_srai_h((v8i16)vec5, 2);
    reg0 *= const_0x19;
    reg1 *= const_0x19;
    reg2 *= const_0x81;
    reg3 *= const_0x81;
    reg4 *= const_0x42;
    reg5 *= const_0x42;
    reg0 += reg2;
    reg1 += reg3;
    reg0 += reg4;
    reg1 += reg5;
    reg0 += const_0x1080;
    reg1 += const_0x1080;
    reg0 = (v8u16)__msa_srai_h((v8i16)reg0, 8);
    reg1 = (v8u16)__msa_srai_h((v8i16)reg1, 8);
    dst0 = (v16u8)__msa_pckev_b((v16i8)reg1, (v16i8)reg0);
    ST_UB(dst0, dst_y);
    src_argb1555 += 32;
    dst_y += 16;
  }
}

void RGB565ToYRow_MSA(const uint8_t* src_rgb565, uint8_t* dst_y, int width) {
  int x;
  v8u16 src0, src1, vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7;
  v8u16 reg0, reg1, reg2, reg3, reg4, reg5;
  v4u32 res0, res1, res2, res3;
  v16u8 dst0;
  v4u32 const_0x810019 = (v4u32)__msa_fill_w(0x810019);
  v4u32 const_0x010042 = (v4u32)__msa_fill_w(0x010042);
  v8i16 const_0x1080 = __msa_fill_h(0x1080);
  v8u16 const_0x1F = (v8u16)__msa_ldi_h(0x1F);
  v8u16 const_0x7E0 = (v8u16)__msa_fill_h(0x7E0);
  v8u16 const_0xF800 = (v8u16)__msa_fill_h(0xF800);

  for (x = 0; x < width; x += 16) {
    src0 = (v8u16)__msa_ld_b((const v8i16*)src_rgb565, 0);
    src1 = (v8u16)__msa_ld_b((const v8i16*)src_rgb565, 16);
    vec0 = src0 & const_0x1F;
    vec1 = src0 & const_0x7E0;
    vec2 = src0 & const_0xF800;
    vec3 = src1 & const_0x1F;
    vec4 = src1 & const_0x7E0;
    vec5 = src1 & const_0xF800;
    reg0 = (v8u16)__msa_slli_h((v8i16)vec0, 3);
    reg1 = (v8u16)__msa_srli_h((v8i16)vec1, 3);
    reg2 = (v8u16)__msa_srli_h((v8i16)vec2, 8);
    reg3 = (v8u16)__msa_slli_h((v8i16)vec3, 3);
    reg4 = (v8u16)__msa_srli_h((v8i16)vec4, 3);
    reg5 = (v8u16)__msa_srli_h((v8i16)vec5, 8);
    reg0 |= (v8u16)__msa_srli_h((v8i16)vec0, 2);
    reg1 |= (v8u16)__msa_srli_h((v8i16)vec1, 9);
    reg2 |= (v8u16)__msa_srli_h((v8i16)vec2, 13);
    reg3 |= (v8u16)__msa_srli_h((v8i16)vec3, 2);
    reg4 |= (v8u16)__msa_srli_h((v8i16)vec4, 9);
    reg5 |= (v8u16)__msa_srli_h((v8i16)vec5, 13);
    vec0 = (v8u16)__msa_ilvr_h((v8i16)reg1, (v8i16)reg0);
    vec1 = (v8u16)__msa_ilvl_h((v8i16)reg1, (v8i16)reg0);
    vec2 = (v8u16)__msa_ilvr_h((v8i16)reg4, (v8i16)reg3);
    vec3 = (v8u16)__msa_ilvl_h((v8i16)reg4, (v8i16)reg3);
    vec4 = (v8u16)__msa_ilvr_h(const_0x1080, (v8i16)reg2);
    vec5 = (v8u16)__msa_ilvl_h(const_0x1080, (v8i16)reg2);
    vec6 = (v8u16)__msa_ilvr_h(const_0x1080, (v8i16)reg5);
    vec7 = (v8u16)__msa_ilvl_h(const_0x1080, (v8i16)reg5);
    res0 = __msa_dotp_u_w(vec0, (v8u16)const_0x810019);
    res1 = __msa_dotp_u_w(vec1, (v8u16)const_0x810019);
    res2 = __msa_dotp_u_w(vec2, (v8u16)const_0x810019);
    res3 = __msa_dotp_u_w(vec3, (v8u16)const_0x810019);
    res0 = __msa_dpadd_u_w(res0, vec4, (v8u16)const_0x010042);
    res1 = __msa_dpadd_u_w(res1, vec5, (v8u16)const_0x010042);
    res2 = __msa_dpadd_u_w(res2, vec6, (v8u16)const_0x010042);
    res3 = __msa_dpadd_u_w(res3, vec7, (v8u16)const_0x010042);
    res0 = (v4u32)__msa_srai_w((v4i32)res0, 8);
    res1 = (v4u32)__msa_srai_w((v4i32)res1, 8);
    res2 = (v4u32)__msa_srai_w((v4i32)res2, 8);
    res3 = (v4u32)__msa_srai_w((v4i32)res3, 8);
    vec0 = (v8u16)__msa_pckev_h((v8i16)res1, (v8i16)res0);
    vec1 = (v8u16)__msa_pckev_h((v8i16)res3, (v8i16)res2);
    dst0 = (v16u8)__msa_pckev_b((v16i8)vec1, (v16i8)vec0);
    ST_UB(dst0, dst_y);
    src_rgb565 += 32;
    dst_y += 16;
  }
}

void RGB24ToYRow_MSA(const uint8_t* src_argb0, uint8_t* dst_y, int width) {
  int x;
  v16u8 src0, src1, src2, reg0, reg1, reg2, reg3, dst0;
  v8u16 vec0, vec1, vec2, vec3;
  v8u16 const_0x8119 = (v8u16)__msa_fill_h(0x8119);
  v8u16 const_0x42 = (v8u16)__msa_fill_h(0x42);
  v8u16 const_0x1080 = (v8u16)__msa_fill_h(0x1080);
  v16i8 mask0 = {0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10, 11, 12};
  v16i8 mask1 = {12, 13, 14, 15, 15, 16, 17, 18,
                 18, 19, 20, 21, 21, 22, 23, 24};
  v16i8 mask2 = {8, 9, 10, 11, 11, 12, 13, 14, 14, 15, 16, 17, 17, 18, 19, 20};
  v16i8 mask3 = {4, 5, 6, 7, 7, 8, 9, 10, 10, 11, 12, 13, 13, 14, 15, 16};
  v16i8 zero = {0};

  for (x = 0; x < width; x += 16) {
    src0 = (v16u8)__msa_ld_b((const v16i8*)src_argb0, 0);
    src1 = (v16u8)__msa_ld_b((const v16i8*)src_argb0, 16);
    src2 = (v16u8)__msa_ld_b((const v16i8*)src_argb0, 32);
    reg0 = (v16u8)__msa_vshf_b(mask0, zero, (v16i8)src0);
    reg1 = (v16u8)__msa_vshf_b(mask1, (v16i8)src1, (v16i8)src0);
    reg2 = (v16u8)__msa_vshf_b(mask2, (v16i8)src2, (v16i8)src1);
    reg3 = (v16u8)__msa_vshf_b(mask3, zero, (v16i8)src2);
    vec0 = (v8u16)__msa_pckev_h((v8i16)reg1, (v8i16)reg0);
    vec1 = (v8u16)__msa_pckev_h((v8i16)reg3, (v8i16)reg2);
    vec2 = (v8u16)__msa_pckod_h((v8i16)reg1, (v8i16)reg0);
    vec3 = (v8u16)__msa_pckod_h((v8i16)reg3, (v8i16)reg2);
    vec0 = __msa_dotp_u_h((v16u8)vec0, (v16u8)const_0x8119);
    vec1 = __msa_dotp_u_h((v16u8)vec1, (v16u8)const_0x8119);
    vec0 = __msa_dpadd_u_h(vec0, (v16u8)vec2, (v16u8)const_0x42);
    vec1 = __msa_dpadd_u_h(vec1, (v16u8)vec3, (v16u8)const_0x42);
    vec0 += const_0x1080;
    vec1 += const_0x1080;
    vec0 = (v8u16)__msa_srai_h((v8i16)vec0, 8);
    vec1 = (v8u16)__msa_srai_h((v8i16)vec1, 8);
    dst0 = (v16u8)__msa_pckev_b((v16i8)vec1, (v16i8)vec0);
    ST_UB(dst0, dst_y);
    src_argb0 += 48;
    dst_y += 16;
  }
}

void RAWToYRow_MSA(const uint8_t* src_argb0, uint8_t* dst_y, int width) {
  int x;
  v16u8 src0, src1, src2, reg0, reg1, reg2, reg3, dst0;
  v8u16 vec0, vec1, vec2, vec3;
  v8u16 const_0x8142 = (v8u16)__msa_fill_h(0x8142);
  v8u16 const_0x19 = (v8u16)__msa_fill_h(0x19);
  v8u16 const_0x1080 = (v8u16)__msa_fill_h(0x1080);
  v16i8 mask0 = {0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10, 11, 12};
  v16i8 mask1 = {12, 13, 14, 15, 15, 16, 17, 18,
                 18, 19, 20, 21, 21, 22, 23, 24};
  v16i8 mask2 = {8, 9, 10, 11, 11, 12, 13, 14, 14, 15, 16, 17, 17, 18, 19, 20};
  v16i8 mask3 = {4, 5, 6, 7, 7, 8, 9, 10, 10, 11, 12, 13, 13, 14, 15, 16};
  v16i8 zero = {0};

  for (x = 0; x < width; x += 16) {
    src0 = (v16u8)__msa_ld_b((const v16i8*)src_argb0, 0);
    src1 = (v16u8)__msa_ld_b((const v16i8*)src_argb0, 16);
    src2 = (v16u8)__msa_ld_b((const v16i8*)src_argb0, 32);
    reg0 = (v16u8)__msa_vshf_b(mask0, zero, (v16i8)src0);
    reg1 = (v16u8)__msa_vshf_b(mask1, (v16i8)src1, (v16i8)src0);
    reg2 = (v16u8)__msa_vshf_b(mask2, (v16i8)src2, (v16i8)src1);
    reg3 = (v16u8)__msa_vshf_b(mask3, zero, (v16i8)src2);
    vec0 = (v8u16)__msa_pckev_h((v8i16)reg1, (v8i16)reg0);
    vec1 = (v8u16)__msa_pckev_h((v8i16)reg3, (v8i16)reg2);
    vec2 = (v8u16)__msa_pckod_h((v8i16)reg1, (v8i16)reg0);
    vec3 = (v8u16)__msa_pckod_h((v8i16)reg3, (v8i16)reg2);
    vec0 = __msa_dotp_u_h((v16u8)vec0, (v16u8)const_0x8142);
    vec1 = __msa_dotp_u_h((v16u8)vec1, (v16u8)const_0x8142);
    vec0 = __msa_dpadd_u_h(vec0, (v16u8)vec2, (v16u8)const_0x19);
    vec1 = __msa_dpadd_u_h(vec1, (v16u8)vec3, (v16u8)const_0x19);
    vec0 += const_0x1080;
    vec1 += const_0x1080;
    vec0 = (v8u16)__msa_srai_h((v8i16)vec0, 8);
    vec1 = (v8u16)__msa_srai_h((v8i16)vec1, 8);
    dst0 = (v16u8)__msa_pckev_b((v16i8)vec1, (v16i8)vec0);
    ST_UB(dst0, dst_y);
    src_argb0 += 48;
    dst_y += 16;
  }
}

void ARGB1555ToUVRow_MSA(const uint8_t* src_argb1555,
                         int src_stride_argb1555,
                         uint8_t* dst_u,
                         uint8_t* dst_v,
                         int width) {
  int x;
  const uint16_t* s = (const uint16_t*)src_argb1555;
  const uint16_t* t = (const uint16_t*)(src_argb1555 + src_stride_argb1555);
  int64_t res0, res1;
  v8u16 src0, src1, src2, src3, reg0, reg1, reg2, reg3;
  v8u16 vec0, vec1, vec2, vec3, vec4, vec5, vec6;
  v16u8 dst0;
  v8u16 const_0x70 = (v8u16)__msa_ldi_h(0x70);
  v8u16 const_0x4A = (v8u16)__msa_ldi_h(0x4A);
  v8u16 const_0x26 = (v8u16)__msa_ldi_h(0x26);
  v8u16 const_0x5E = (v8u16)__msa_ldi_h(0x5E);
  v8u16 const_0x12 = (v8u16)__msa_ldi_h(0x12);
  v8u16 const_0x8080 = (v8u16)__msa_fill_h(0x8080);
  v8u16 const_0x1F = (v8u16)__msa_ldi_h(0x1F);

  for (x = 0; x < width; x += 16) {
    src0 = (v8u16)__msa_ld_b((v8i16*)s, 0);
    src1 = (v8u16)__msa_ld_b((v8i16*)s, 16);
    src2 = (v8u16)__msa_ld_b((v8i16*)t, 0);
    src3 = (v8u16)__msa_ld_b((v8i16*)t, 16);
    vec0 = src0 & const_0x1F;
    vec1 = src1 & const_0x1F;
    vec0 += src2 & const_0x1F;
    vec1 += src3 & const_0x1F;
    vec0 = (v8u16)__msa_pckev_b((v16i8)vec1, (v16i8)vec0);
    src0 = (v8u16)__msa_srai_h((v8i16)src0, 5);
    src1 = (v8u16)__msa_srai_h((v8i16)src1, 5);
    src2 = (v8u16)__msa_srai_h((v8i16)src2, 5);
    src3 = (v8u16)__msa_srai_h((v8i16)src3, 5);
    vec2 = src0 & const_0x1F;
    vec3 = src1 & const_0x1F;
    vec2 += src2 & const_0x1F;
    vec3 += src3 & const_0x1F;
    vec2 = (v8u16)__msa_pckev_b((v16i8)vec3, (v16i8)vec2);
    src0 = (v8u16)__msa_srai_h((v8i16)src0, 5);
    src1 = (v8u16)__msa_srai_h((v8i16)src1, 5);
    src2 = (v8u16)__msa_srai_h((v8i16)src2, 5);
    src3 = (v8u16)__msa_srai_h((v8i16)src3, 5);
    vec4 = src0 & const_0x1F;
    vec5 = src1 & const_0x1F;
    vec4 += src2 & const_0x1F;
    vec5 += src3 & const_0x1F;
    vec4 = (v8u16)__msa_pckev_b((v16i8)vec5, (v16i8)vec4);
    vec0 = __msa_hadd_u_h((v16u8)vec0, (v16u8)vec0);
    vec2 = __msa_hadd_u_h((v16u8)vec2, (v16u8)vec2);
    vec4 = __msa_hadd_u_h((v16u8)vec4, (v16u8)vec4);
    vec6 = (v8u16)__msa_slli_h((v8i16)vec0, 1);
    vec6 |= (v8u16)__msa_srai_h((v8i16)vec0, 6);
    vec0 = (v8u16)__msa_slli_h((v8i16)vec2, 1);
    vec0 |= (v8u16)__msa_srai_h((v8i16)vec2, 6);
    vec2 = (v8u16)__msa_slli_h((v8i16)vec4, 1);
    vec2 |= (v8u16)__msa_srai_h((v8i16)vec4, 6);
    reg0 = vec6 * const_0x70;
    reg1 = vec0 * const_0x4A;
    reg2 = vec2 * const_0x70;
    reg3 = vec0 * const_0x5E;
    reg0 += const_0x8080;
    reg1 += vec2 * const_0x26;
    reg2 += const_0x8080;
    reg3 += vec6 * const_0x12;
    reg0 -= reg1;
    reg2 -= reg3;
    reg0 = (v8u16)__msa_srai_h((v8i16)reg0, 8);
    reg2 = (v8u16)__msa_srai_h((v8i16)reg2, 8);
    dst0 = (v16u8)__msa_pckev_b((v16i8)reg2, (v16i8)reg0);
    res0 = __msa_copy_u_d((v2i64)dst0, 0);
    res1 = __msa_copy_u_d((v2i64)dst0, 1);
    SD(res0, dst_u);
    SD(res1, dst_v);
    s += 16;
    t += 16;
    dst_u += 8;
    dst_v += 8;
  }
}

void RGB565ToUVRow_MSA(const uint8_t* src_rgb565,
                       int src_stride_rgb565,
                       uint8_t* dst_u,
                       uint8_t* dst_v,
                       int width) {
  int x;
  const uint16_t* s = (const uint16_t*)src_rgb565;
  const uint16_t* t = (const uint16_t*)(src_rgb565 + src_stride_rgb565);
  int64_t res0, res1;
  v8u16 src0, src1, src2, src3, reg0, reg1, reg2, reg3;
  v8u16 vec0, vec1, vec2, vec3, vec4, vec5;
  v16u8 dst0;
  v8u16 const_0x70 = (v8u16)__msa_ldi_h(0x70);
  v8u16 const_0x4A = (v8u16)__msa_ldi_h(0x4A);
  v8u16 const_0x26 = (v8u16)__msa_ldi_h(0x26);
  v8u16 const_0x5E = (v8u16)__msa_ldi_h(0x5E);
  v8u16 const_0x12 = (v8u16)__msa_ldi_h(0x12);
  v8u16 const_32896 = (v8u16)__msa_fill_h(0x8080);
  v8u16 const_0x1F = (v8u16)__msa_ldi_h(0x1F);
  v8u16 const_0x3F = (v8u16)__msa_fill_h(0x3F);

  for (x = 0; x < width; x += 16) {
    src0 = (v8u16)__msa_ld_b((v8i16*)s, 0);
    src1 = (v8u16)__msa_ld_b((v8i16*)s, 16);
    src2 = (v8u16)__msa_ld_b((v8i16*)t, 0);
    src3 = (v8u16)__msa_ld_b((v8i16*)t, 16);
    vec0 = src0 & const_0x1F;
    vec1 = src1 & const_0x1F;
    vec0 += src2 & const_0x1F;
    vec1 += src3 & const_0x1F;
    vec0 = (v8u16)__msa_pckev_b((v16i8)vec1, (v16i8)vec0);
    src0 = (v8u16)__msa_srai_h((v8i16)src0, 5);
    src1 = (v8u16)__msa_srai_h((v8i16)src1, 5);
    src2 = (v8u16)__msa_srai_h((v8i16)src2, 5);
    src3 = (v8u16)__msa_srai_h((v8i16)src3, 5);
    vec2 = src0 & const_0x3F;
    vec3 = src1 & const_0x3F;
    vec2 += src2 & const_0x3F;
    vec3 += src3 & const_0x3F;
    vec1 = (v8u16)__msa_pckev_b((v16i8)vec3, (v16i8)vec2);
    src0 = (v8u16)__msa_srai_h((v8i16)src0, 6);
    src1 = (v8u16)__msa_srai_h((v8i16)src1, 6);
    src2 = (v8u16)__msa_srai_h((v8i16)src2, 6);
    src3 = (v8u16)__msa_srai_h((v8i16)src3, 6);
    vec4 = src0 & const_0x1F;
    vec5 = src1 & const_0x1F;
    vec4 += src2 & const_0x1F;
    vec5 += src3 & const_0x1F;
    vec2 = (v8u16)__msa_pckev_b((v16i8)vec5, (v16i8)vec4);
    vec0 = __msa_hadd_u_h((v16u8)vec0, (v16u8)vec0);
    vec1 = __msa_hadd_u_h((v16u8)vec1, (v16u8)vec1);
    vec2 = __msa_hadd_u_h((v16u8)vec2, (v16u8)vec2);
    vec3 = (v8u16)__msa_slli_h((v8i16)vec0, 1);
    vec3 |= (v8u16)__msa_srai_h((v8i16)vec0, 6);
    vec4 = (v8u16)__msa_slli_h((v8i16)vec2, 1);
    vec4 |= (v8u16)__msa_srai_h((v8i16)vec2, 6);
    reg0 = vec3 * const_0x70;
    reg1 = vec1 * const_0x4A;
    reg2 = vec4 * const_0x70;
    reg3 = vec1 * const_0x5E;
    reg0 += const_32896;
    reg1 += vec4 * const_0x26;
    reg2 += const_32896;
    reg3 += vec3 * const_0x12;
    reg0 -= reg1;
    reg2 -= reg3;
    reg0 = (v8u16)__msa_srai_h((v8i16)reg0, 8);
    reg2 = (v8u16)__msa_srai_h((v8i16)reg2, 8);
    dst0 = (v16u8)__msa_pckev_b((v16i8)reg2, (v16i8)reg0);
    res0 = __msa_copy_u_d((v2i64)dst0, 0);
    res1 = __msa_copy_u_d((v2i64)dst0, 1);
    SD(res0, dst_u);
    SD(res1, dst_v);
    s += 16;
    t += 16;
    dst_u += 8;
    dst_v += 8;
  }
}

void RGB24ToUVRow_MSA(const uint8_t* src_rgb0,
                      int src_stride_rgb,
                      uint8_t* dst_u,
                      uint8_t* dst_v,
                      int width) {
  int x;
  const uint8_t* s = src_rgb0;
  const uint8_t* t = src_rgb0 + src_stride_rgb;
  int64_t res0, res1;
  v16u8 src0, src1, src2, src3, src4, src5, src6, src7;
  v16u8 inp0, inp1, inp2, inp3, inp4, inp5;
  v8u16 vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7;
  v8i16 reg0, reg1, reg2, reg3;
  v16u8 dst0;
  v8u16 const_0x70 = (v8u16)__msa_fill_h(0x70);
  v8u16 const_0x4A = (v8u16)__msa_fill_h(0x4A);
  v8u16 const_0x26 = (v8u16)__msa_fill_h(0x26);
  v8u16 const_0x5E = (v8u16)__msa_fill_h(0x5E);
  v8u16 const_0x12 = (v8u16)__msa_fill_h(0x12);
  v8u16 const_0x8080 = (v8u16)__msa_fill_h(0x8080);
  v16i8 mask = {0, 1, 2, 16, 3, 4, 5, 17, 6, 7, 8, 18, 9, 10, 11, 19};
  v16i8 zero = {0};

  for (x = 0; x < width; x += 16) {
    inp0 = (v16u8)__msa_ld_b((const v16i8*)s, 0);
    inp1 = (v16u8)__msa_ld_b((const v16i8*)s, 16);
    inp2 = (v16u8)__msa_ld_b((const v16i8*)s, 32);
    inp3 = (v16u8)__msa_ld_b((const v16i8*)t, 0);
    inp4 = (v16u8)__msa_ld_b((const v16i8*)t, 16);
    inp5 = (v16u8)__msa_ld_b((const v16i8*)t, 32);
    src1 = (v16u8)__msa_sldi_b((v16i8)inp1, (v16i8)inp0, 12);
    src5 = (v16u8)__msa_sldi_b((v16i8)inp4, (v16i8)inp3, 12);
    src2 = (v16u8)__msa_sldi_b((v16i8)inp2, (v16i8)inp1, 8);
    src6 = (v16u8)__msa_sldi_b((v16i8)inp5, (v16i8)inp4, 8);
    src3 = (v16u8)__msa_sldi_b((v16i8)inp2, (v16i8)inp2, 4);
    src7 = (v16u8)__msa_sldi_b((v16i8)inp5, (v16i8)inp5, 4);
    src0 = (v16u8)__msa_vshf_b(mask, (v16i8)zero, (v16i8)inp0);
    src1 = (v16u8)__msa_vshf_b(mask, (v16i8)zero, (v16i8)src1);
    src2 = (v16u8)__msa_vshf_b(mask, (v16i8)zero, (v16i8)src2);
    src3 = (v16u8)__msa_vshf_b(mask, (v16i8)zero, (v16i8)src3);
    src4 = (v16u8)__msa_vshf_b(mask, (v16i8)zero, (v16i8)inp3);
    src5 = (v16u8)__msa_vshf_b(mask, (v16i8)zero, (v16i8)src5);
    src6 = (v16u8)__msa_vshf_b(mask, (v16i8)zero, (v16i8)src6);
    src7 = (v16u8)__msa_vshf_b(mask, (v16i8)zero, (v16i8)src7);
    vec0 = (v8u16)__msa_ilvr_b((v16i8)src4, (v16i8)src0);
    vec1 = (v8u16)__msa_ilvl_b((v16i8)src4, (v16i8)src0);
    vec2 = (v8u16)__msa_ilvr_b((v16i8)src5, (v16i8)src1);
    vec3 = (v8u16)__msa_ilvl_b((v16i8)src5, (v16i8)src1);
    vec4 = (v8u16)__msa_ilvr_b((v16i8)src6, (v16i8)src2);
    vec5 = (v8u16)__msa_ilvl_b((v16i8)src6, (v16i8)src2);
    vec6 = (v8u16)__msa_ilvr_b((v16i8)src7, (v16i8)src3);
    vec7 = (v8u16)__msa_ilvl_b((v16i8)src7, (v16i8)src3);
    vec0 = (v8u16)__msa_hadd_u_h((v16u8)vec0, (v16u8)vec0);
    vec1 = (v8u16)__msa_hadd_u_h((v16u8)vec1, (v16u8)vec1);
    vec2 = (v8u16)__msa_hadd_u_h((v16u8)vec2, (v16u8)vec2);
    vec3 = (v8u16)__msa_hadd_u_h((v16u8)vec3, (v16u8)vec3);
    vec4 = (v8u16)__msa_hadd_u_h((v16u8)vec4, (v16u8)vec4);
    vec5 = (v8u16)__msa_hadd_u_h((v16u8)vec5, (v16u8)vec5);
    vec6 = (v8u16)__msa_hadd_u_h((v16u8)vec6, (v16u8)vec6);
    vec7 = (v8u16)__msa_hadd_u_h((v16u8)vec7, (v16u8)vec7);
    reg0 = (v8i16)__msa_pckev_d((v2i64)vec1, (v2i64)vec0);
    reg1 = (v8i16)__msa_pckev_d((v2i64)vec3, (v2i64)vec2);
    reg2 = (v8i16)__msa_pckev_d((v2i64)vec5, (v2i64)vec4);
    reg3 = (v8i16)__msa_pckev_d((v2i64)vec7, (v2i64)vec6);
    reg0 += (v8i16)__msa_pckod_d((v2i64)vec1, (v2i64)vec0);
    reg1 += (v8i16)__msa_pckod_d((v2i64)vec3, (v2i64)vec2);
    reg2 += (v8i16)__msa_pckod_d((v2i64)vec5, (v2i64)vec4);
    reg3 += (v8i16)__msa_pckod_d((v2i64)vec7, (v2i64)vec6);
    reg0 = __msa_srai_h((v8i16)reg0, 2);
    reg1 = __msa_srai_h((v8i16)reg1, 2);
    reg2 = __msa_srai_h((v8i16)reg2, 2);
    reg3 = __msa_srai_h((v8i16)reg3, 2);
    vec4 = (v8u16)__msa_pckev_h(reg1, reg0);
    vec5 = (v8u16)__msa_pckev_h(reg3, reg2);
    vec6 = (v8u16)__msa_pckod_h(reg1, reg0);
    vec7 = (v8u16)__msa_pckod_h(reg3, reg2);
    vec0 = (v8u16)__msa_pckev_h((v8i16)vec5, (v8i16)vec4);
    vec1 = (v8u16)__msa_pckev_h((v8i16)vec7, (v8i16)vec6);
    vec2 = (v8u16)__msa_pckod_h((v8i16)vec5, (v8i16)vec4);
    vec3 = vec0 * const_0x70;
    vec4 = vec1 * const_0x4A;
    vec5 = vec2 * const_0x26;
    vec2 *= const_0x70;
    vec1 *= const_0x5E;
    vec0 *= const_0x12;
    reg0 = __msa_subv_h((v8i16)vec3, (v8i16)vec4);
    reg1 = __msa_subv_h((v8i16)const_0x8080, (v8i16)vec5);
    reg2 = __msa_subv_h((v8i16)vec2, (v8i16)vec1);
    reg3 = __msa_subv_h((v8i16)const_0x8080, (v8i16)vec0);
    reg0 += reg1;
    reg2 += reg3;
    reg0 = __msa_srai_h(reg0, 8);
    reg2 = __msa_srai_h(reg2, 8);
    dst0 = (v16u8)__msa_pckev_b((v16i8)reg2, (v16i8)reg0);
    res0 = __msa_copy_u_d((v2i64)dst0, 0);
    res1 = __msa_copy_u_d((v2i64)dst0, 1);
    SD(res0, dst_u);
    SD(res1, dst_v);
    t += 48;
    s += 48;
    dst_u += 8;
    dst_v += 8;
  }
}

void RAWToUVRow_MSA(const uint8_t* src_rgb0,
                    int src_stride_rgb,
                    uint8_t* dst_u,
                    uint8_t* dst_v,
                    int width) {
  int x;
  const uint8_t* s = src_rgb0;
  const uint8_t* t = src_rgb0 + src_stride_rgb;
  int64_t res0, res1;
  v16u8 inp0, inp1, inp2, inp3, inp4, inp5;
  v16u8 src0, src1, src2, src3, src4, src5, src6, src7;
  v8u16 vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7;
  v8i16 reg0, reg1, reg2, reg3;
  v16u8 dst0;
  v8u16 const_0x70 = (v8u16)__msa_fill_h(0x70);
  v8u16 const_0x4A = (v8u16)__msa_fill_h(0x4A);
  v8u16 const_0x26 = (v8u16)__msa_fill_h(0x26);
  v8u16 const_0x5E = (v8u16)__msa_fill_h(0x5E);
  v8u16 const_0x12 = (v8u16)__msa_fill_h(0x12);
  v8u16 const_0x8080 = (v8u16)__msa_fill_h(0x8080);
  v16i8 mask = {0, 1, 2, 16, 3, 4, 5, 17, 6, 7, 8, 18, 9, 10, 11, 19};
  v16i8 zero = {0};

  for (x = 0; x < width; x += 16) {
    inp0 = (v16u8)__msa_ld_b((const v16i8*)s, 0);
    inp1 = (v16u8)__msa_ld_b((const v16i8*)s, 16);
    inp2 = (v16u8)__msa_ld_b((const v16i8*)s, 32);
    inp3 = (v16u8)__msa_ld_b((const v16i8*)t, 0);
    inp4 = (v16u8)__msa_ld_b((const v16i8*)t, 16);
    inp5 = (v16u8)__msa_ld_b((const v16i8*)t, 32);
    src1 = (v16u8)__msa_sldi_b((v16i8)inp1, (v16i8)inp0, 12);
    src5 = (v16u8)__msa_sldi_b((v16i8)inp4, (v16i8)inp3, 12);
    src2 = (v16u8)__msa_sldi_b((v16i8)inp2, (v16i8)inp1, 8);
    src6 = (v16u8)__msa_sldi_b((v16i8)inp5, (v16i8)inp4, 8);
    src3 = (v16u8)__msa_sldi_b((v16i8)inp2, (v16i8)inp2, 4);
    src7 = (v16u8)__msa_sldi_b((v16i8)inp5, (v16i8)inp5, 4);
    src0 = (v16u8)__msa_vshf_b(mask, (v16i8)zero, (v16i8)inp0);
    src1 = (v16u8)__msa_vshf_b(mask, (v16i8)zero, (v16i8)src1);
    src2 = (v16u8)__msa_vshf_b(mask, (v16i8)zero, (v16i8)src2);
    src3 = (v16u8)__msa_vshf_b(mask, (v16i8)zero, (v16i8)src3);
    src4 = (v16u8)__msa_vshf_b(mask, (v16i8)zero, (v16i8)inp3);
    src5 = (v16u8)__msa_vshf_b(mask, (v16i8)zero, (v16i8)src5);
    src6 = (v16u8)__msa_vshf_b(mask, (v16i8)zero, (v16i8)src6);
    src7 = (v16u8)__msa_vshf_b(mask, (v16i8)zero, (v16i8)src7);
    vec0 = (v8u16)__msa_ilvr_b((v16i8)src4, (v16i8)src0);
    vec1 = (v8u16)__msa_ilvl_b((v16i8)src4, (v16i8)src0);
    vec2 = (v8u16)__msa_ilvr_b((v16i8)src5, (v16i8)src1);
    vec3 = (v8u16)__msa_ilvl_b((v16i8)src5, (v16i8)src1);
    vec4 = (v8u16)__msa_ilvr_b((v16i8)src6, (v16i8)src2);
    vec5 = (v8u16)__msa_ilvl_b((v16i8)src6, (v16i8)src2);
    vec6 = (v8u16)__msa_ilvr_b((v16i8)src7, (v16i8)src3);
    vec7 = (v8u16)__msa_ilvl_b((v16i8)src7, (v16i8)src3);
    vec0 = (v8u16)__msa_hadd_u_h((v16u8)vec0, (v16u8)vec0);
    vec1 = (v8u16)__msa_hadd_u_h((v16u8)vec1, (v16u8)vec1);
    vec2 = (v8u16)__msa_hadd_u_h((v16u8)vec2, (v16u8)vec2);
    vec3 = (v8u16)__msa_hadd_u_h((v16u8)vec3, (v16u8)vec3);
    vec4 = (v8u16)__msa_hadd_u_h((v16u8)vec4, (v16u8)vec4);
    vec5 = (v8u16)__msa_hadd_u_h((v16u8)vec5, (v16u8)vec5);
    vec6 = (v8u16)__msa_hadd_u_h((v16u8)vec6, (v16u8)vec6);
    vec7 = (v8u16)__msa_hadd_u_h((v16u8)vec7, (v16u8)vec7);
    reg0 = (v8i16)__msa_pckev_d((v2i64)vec1, (v2i64)vec0);
    reg1 = (v8i16)__msa_pckev_d((v2i64)vec3, (v2i64)vec2);
    reg2 = (v8i16)__msa_pckev_d((v2i64)vec5, (v2i64)vec4);
    reg3 = (v8i16)__msa_pckev_d((v2i64)vec7, (v2i64)vec6);
    reg0 += (v8i16)__msa_pckod_d((v2i64)vec1, (v2i64)vec0);
    reg1 += (v8i16)__msa_pckod_d((v2i64)vec3, (v2i64)vec2);
    reg2 += (v8i16)__msa_pckod_d((v2i64)vec5, (v2i64)vec4);
    reg3 += (v8i16)__msa_pckod_d((v2i64)vec7, (v2i64)vec6);
    reg0 = __msa_srai_h(reg0, 2);
    reg1 = __msa_srai_h(reg1, 2);
    reg2 = __msa_srai_h(reg2, 2);
    reg3 = __msa_srai_h(reg3, 2);
    vec4 = (v8u16)__msa_pckev_h((v8i16)reg1, (v8i16)reg0);
    vec5 = (v8u16)__msa_pckev_h((v8i16)reg3, (v8i16)reg2);
    vec6 = (v8u16)__msa_pckod_h((v8i16)reg1, (v8i16)reg0);
    vec7 = (v8u16)__msa_pckod_h((v8i16)reg3, (v8i16)reg2);
    vec0 = (v8u16)__msa_pckod_h((v8i16)vec5, (v8i16)vec4);
    vec1 = (v8u16)__msa_pckev_h((v8i16)vec7, (v8i16)vec6);
    vec2 = (v8u16)__msa_pckev_h((v8i16)vec5, (v8i16)vec4);
    vec3 = vec0 * const_0x70;
    vec4 = vec1 * const_0x4A;
    vec5 = vec2 * const_0x26;
    vec2 *= const_0x70;
    vec1 *= const_0x5E;
    vec0 *= const_0x12;
    reg0 = __msa_subv_h((v8i16)vec3, (v8i16)vec4);
    reg1 = __msa_subv_h((v8i16)const_0x8080, (v8i16)vec5);
    reg2 = __msa_subv_h((v8i16)vec2, (v8i16)vec1);
    reg3 = __msa_subv_h((v8i16)const_0x8080, (v8i16)vec0);
    reg0 += reg1;
    reg2 += reg3;
    reg0 = __msa_srai_h(reg0, 8);
    reg2 = __msa_srai_h(reg2, 8);
    dst0 = (v16u8)__msa_pckev_b((v16i8)reg2, (v16i8)reg0);
    res0 = __msa_copy_u_d((v2i64)dst0, 0);
    res1 = __msa_copy_u_d((v2i64)dst0, 1);
    SD(res0, dst_u);
    SD(res1, dst_v);
    t += 48;
    s += 48;
    dst_u += 8;
    dst_v += 8;
  }
}

void NV12ToARGBRow_MSA(const uint8_t* src_y,
                       const uint8_t* src_uv,
                       uint8_t* dst_argb,
                       const struct YuvConstants* yuvconstants,
                       int width) {
  int x;
  uint64_t val0, val1;
  v16u8 src0, src1, res0, res1, dst0, dst1;
  v8i16 vec0, vec1, vec2;
  v4i32 vec_ub, vec_vr, vec_ug, vec_vg, vec_bb, vec_bg, vec_br, vec_yg;
  v4i32 vec_ubvr, vec_ugvg;
  v16u8 zero = {0};
  v16u8 alpha = (v16u8)__msa_ldi_b(ALPHA_VAL);

  YUVTORGB_SETUP(yuvconstants, vec_ub, vec_vr, vec_ug, vec_vg, vec_bb, vec_bg,
                 vec_br, vec_yg);
  vec_ubvr = __msa_ilvr_w(vec_vr, vec_ub);
  vec_ugvg = (v4i32)__msa_ilvev_h((v8i16)vec_vg, (v8i16)vec_ug);

  for (x = 0; x < width; x += 8) {
    val0 = LD(src_y);
    val1 = LD(src_uv);
    src0 = (v16u8)__msa_insert_d((v2i64)zero, 0, val0);
    src1 = (v16u8)__msa_insert_d((v2i64)zero, 0, val1);
    YUVTORGB(src0, src1, vec_ubvr, vec_ugvg, vec_bb, vec_bg, vec_br, vec_yg,
             vec0, vec1, vec2);
    res0 = (v16u8)__msa_ilvev_b((v16i8)vec2, (v16i8)vec0);
    res1 = (v16u8)__msa_ilvev_b((v16i8)alpha, (v16i8)vec1);
    dst0 = (v16u8)__msa_ilvr_b((v16i8)res1, (v16i8)res0);
    dst1 = (v16u8)__msa_ilvl_b((v16i8)res1, (v16i8)res0);
    ST_UB2(dst0, dst1, dst_argb, 16);
    src_y += 8;
    src_uv += 8;
    dst_argb += 32;
  }
}

void NV12ToRGB565Row_MSA(const uint8_t* src_y,
                         const uint8_t* src_uv,
                         uint8_t* dst_rgb565,
                         const struct YuvConstants* yuvconstants,
                         int width) {
  int x;
  uint64_t val0, val1;
  v16u8 src0, src1, dst0;
  v8i16 vec0, vec1, vec2;
  v4i32 vec_ub, vec_vr, vec_ug, vec_vg, vec_bb, vec_bg, vec_br, vec_yg;
  v4i32 vec_ubvr, vec_ugvg;
  v16u8 zero = {0};

  YUVTORGB_SETUP(yuvconstants, vec_ub, vec_vr, vec_ug, vec_vg, vec_bb, vec_bg,
                 vec_br, vec_yg);
  vec_ubvr = __msa_ilvr_w(vec_vr, vec_ub);
  vec_ugvg = (v4i32)__msa_ilvev_h((v8i16)vec_vg, (v8i16)vec_ug);

  for (x = 0; x < width; x += 8) {
    val0 = LD(src_y);
    val1 = LD(src_uv);
    src0 = (v16u8)__msa_insert_d((v2i64)zero, 0, val0);
    src1 = (v16u8)__msa_insert_d((v2i64)zero, 0, val1);
    YUVTORGB(src0, src1, vec_ubvr, vec_ugvg, vec_bb, vec_bg, vec_br, vec_yg,
             vec0, vec1, vec2);
    vec0 = vec0 >> 3;
    vec1 = (vec1 >> 2) << 5;
    vec2 = (vec2 >> 3) << 11;
    dst0 = (v16u8)(vec0 | vec1 | vec2);
    ST_UB(dst0, dst_rgb565);
    src_y += 8;
    src_uv += 8;
    dst_rgb565 += 16;
  }
}

void NV21ToARGBRow_MSA(const uint8_t* src_y,
                       const uint8_t* src_vu,
                       uint8_t* dst_argb,
                       const struct YuvConstants* yuvconstants,
                       int width) {
  int x;
  uint64_t val0, val1;
  v16u8 src0, src1, res0, res1, dst0, dst1;
  v8i16 vec0, vec1, vec2;
  v4i32 vec_ub, vec_vr, vec_ug, vec_vg, vec_bb, vec_bg, vec_br, vec_yg;
  v4i32 vec_ubvr, vec_ugvg;
  v16u8 alpha = (v16u8)__msa_ldi_b(ALPHA_VAL);
  v16u8 zero = {0};
  v16i8 shuffler = {1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14};

  YUVTORGB_SETUP(yuvconstants, vec_ub, vec_vr, vec_ug, vec_vg, vec_bb, vec_bg,
                 vec_br, vec_yg);
  vec_ubvr = __msa_ilvr_w(vec_vr, vec_ub);
  vec_ugvg = (v4i32)__msa_ilvev_h((v8i16)vec_vg, (v8i16)vec_ug);

  for (x = 0; x < width; x += 8) {
    val0 = LD(src_y);
    val1 = LD(src_vu);
    src0 = (v16u8)__msa_insert_d((v2i64)zero, 0, val0);
    src1 = (v16u8)__msa_insert_d((v2i64)zero, 0, val1);
    src1 = (v16u8)__msa_vshf_b(shuffler, (v16i8)src1, (v16i8)src1);
    YUVTORGB(src0, src1, vec_ubvr, vec_ugvg, vec_bb, vec_bg, vec_br, vec_yg,
             vec0, vec1, vec2);
    res0 = (v16u8)__msa_ilvev_b((v16i8)vec2, (v16i8)vec0);
    res1 = (v16u8)__msa_ilvev_b((v16i8)alpha, (v16i8)vec1);
    dst0 = (v16u8)__msa_ilvr_b((v16i8)res1, (v16i8)res0);
    dst1 = (v16u8)__msa_ilvl_b((v16i8)res1, (v16i8)res0);
    ST_UB2(dst0, dst1, dst_argb, 16);
    src_y += 8;
    src_vu += 8;
    dst_argb += 32;
  }
}

void SobelRow_MSA(const uint8_t* src_sobelx,
                  const uint8_t* src_sobely,
                  uint8_t* dst_argb,
                  int width) {
  int x;
  v16u8 src0, src1, vec0, dst0, dst1, dst2, dst3;
  v16i8 mask0 = {0, 0, 0, 16, 1, 1, 1, 16, 2, 2, 2, 16, 3, 3, 3, 16};
  v16i8 const_0x4 = __msa_ldi_b(0x4);
  v16i8 mask1 = mask0 + const_0x4;
  v16i8 mask2 = mask1 + const_0x4;
  v16i8 mask3 = mask2 + const_0x4;
  v16u8 alpha = (v16u8)__msa_ldi_b(ALPHA_VAL);

  for (x = 0; x < width; x += 16) {
    src0 = (v16u8)__msa_ld_b((const v16i8*)src_sobelx, 0);
    src1 = (v16u8)__msa_ld_b((const v16i8*)src_sobely, 0);
    vec0 = __msa_adds_u_b(src0, src1);
    dst0 = (v16u8)__msa_vshf_b(mask0, (v16i8)alpha, (v16i8)vec0);
    dst1 = (v16u8)__msa_vshf_b(mask1, (v16i8)alpha, (v16i8)vec0);
    dst2 = (v16u8)__msa_vshf_b(mask2, (v16i8)alpha, (v16i8)vec0);
    dst3 = (v16u8)__msa_vshf_b(mask3, (v16i8)alpha, (v16i8)vec0);
    ST_UB4(dst0, dst1, dst2, dst3, dst_argb, 16);
    src_sobelx += 16;
    src_sobely += 16;
    dst_argb += 64;
  }
}

void SobelToPlaneRow_MSA(const uint8_t* src_sobelx,
                         const uint8_t* src_sobely,
                         uint8_t* dst_y,
                         int width) {
  int x;
  v16u8 src0, src1, src2, src3, dst0, dst1;

  for (x = 0; x < width; x += 32) {
    src0 = (v16u8)__msa_ld_b((const v16i8*)src_sobelx, 0);
    src1 = (v16u8)__msa_ld_b((const v16i8*)src_sobelx, 16);
    src2 = (v16u8)__msa_ld_b((const v16i8*)src_sobely, 0);
    src3 = (v16u8)__msa_ld_b((const v16i8*)src_sobely, 16);
    dst0 = __msa_adds_u_b(src0, src2);
    dst1 = __msa_adds_u_b(src1, src3);
    ST_UB2(dst0, dst1, dst_y, 16);
    src_sobelx += 32;
    src_sobely += 32;
    dst_y += 32;
  }
}

void SobelXYRow_MSA(const uint8_t* src_sobelx,
                    const uint8_t* src_sobely,
                    uint8_t* dst_argb,
                    int width) {
  int x;
  v16u8 src0, src1, vec0, vec1, vec2;
  v16u8 reg0, reg1, dst0, dst1, dst2, dst3;
  v16u8 alpha = (v16u8)__msa_ldi_b(ALPHA_VAL);

  for (x = 0; x < width; x += 16) {
    src0 = (v16u8)__msa_ld_b((const v16i8*)src_sobelx, 0);
    src1 = (v16u8)__msa_ld_b((const v16i8*)src_sobely, 0);
    vec0 = __msa_adds_u_b(src0, src1);
    vec1 = (v16u8)__msa_ilvr_b((v16i8)src0, (v16i8)src1);
    vec2 = (v16u8)__msa_ilvl_b((v16i8)src0, (v16i8)src1);
    reg0 = (v16u8)__msa_ilvr_b((v16i8)alpha, (v16i8)vec0);
    reg1 = (v16u8)__msa_ilvl_b((v16i8)alpha, (v16i8)vec0);
    dst0 = (v16u8)__msa_ilvr_b((v16i8)reg0, (v16i8)vec1);
    dst1 = (v16u8)__msa_ilvl_b((v16i8)reg0, (v16i8)vec1);
    dst2 = (v16u8)__msa_ilvr_b((v16i8)reg1, (v16i8)vec2);
    dst3 = (v16u8)__msa_ilvl_b((v16i8)reg1, (v16i8)vec2);
    ST_UB4(dst0, dst1, dst2, dst3, dst_argb, 16);
    src_sobelx += 16;
    src_sobely += 16;
    dst_argb += 64;
  }
}

void ARGBToYJRow_MSA(const uint8_t* src_argb0, uint8_t* dst_y, int width) {
  int x;
  v16u8 src0, src1, src2, src3, dst0;
  v16u8 const_0x4B0F = (v16u8)__msa_fill_h(0x4B0F);
  v16u8 const_0x26 = (v16u8)__msa_fill_h(0x26);
  v8u16 const_0x40 = (v8u16)__msa_fill_h(0x40);

  for (x = 0; x < width; x += 16) {
    src0 = (v16u8)__msa_ld_b((const v16i8*)src_argb0, 0);
    src1 = (v16u8)__msa_ld_b((const v16i8*)src_argb0, 16);
    src2 = (v16u8)__msa_ld_b((const v16i8*)src_argb0, 32);
    src3 = (v16u8)__msa_ld_b((const v16i8*)src_argb0, 48);
    ARGBTOY(src0, src1, src2, src3, const_0x4B0F, const_0x26, const_0x40, 7,
            dst0);
    ST_UB(dst0, dst_y);
    src_argb0 += 64;
    dst_y += 16;
  }
}

void BGRAToYRow_MSA(const uint8_t* src_argb0, uint8_t* dst_y, int width) {
  int x;
  v16u8 src0, src1, src2, src3, dst0;
  v16u8 const_0x4200 = (v16u8)__msa_fill_h(0x4200);
  v16u8 const_0x1981 = (v16u8)__msa_fill_h(0x1981);
  v8u16 const_0x1080 = (v8u16)__msa_fill_h(0x1080);

  for (x = 0; x < width; x += 16) {
    src0 = (v16u8)__msa_ld_b((const v16i8*)src_argb0, 0);
    src1 = (v16u8)__msa_ld_b((const v16i8*)src_argb0, 16);
    src2 = (v16u8)__msa_ld_b((const v16i8*)src_argb0, 32);
    src3 = (v16u8)__msa_ld_b((const v16i8*)src_argb0, 48);
    ARGBTOY(src0, src1, src2, src3, const_0x4200, const_0x1981, const_0x1080, 8,
            dst0);
    ST_UB(dst0, dst_y);
    src_argb0 += 64;
    dst_y += 16;
  }
}

void ABGRToYRow_MSA(const uint8_t* src_argb0, uint8_t* dst_y, int width) {
  int x;
  v16u8 src0, src1, src2, src3, dst0;
  v16u8 const_0x8142 = (v16u8)__msa_fill_h(0x8142);
  v16u8 const_0x19 = (v16u8)__msa_fill_h(0x19);
  v8u16 const_0x1080 = (v8u16)__msa_fill_h(0x1080);

  for (x = 0; x < width; x += 16) {
    src0 = (v16u8)__msa_ld_b((const v16i8*)src_argb0, 0);
    src1 = (v16u8)__msa_ld_b((const v16i8*)src_argb0, 16);
    src2 = (v16u8)__msa_ld_b((const v16i8*)src_argb0, 32);
    src3 = (v16u8)__msa_ld_b((const v16i8*)src_argb0, 48);
    ARGBTOY(src0, src1, src2, src3, const_0x8142, const_0x19, const_0x1080, 8,
            dst0);
    ST_UB(dst0, dst_y);
    src_argb0 += 64;
    dst_y += 16;
  }
}

void RGBAToYRow_MSA(const uint8_t* src_argb0, uint8_t* dst_y, int width) {
  int x;
  v16u8 src0, src1, src2, src3, dst0;
  v16u8 const_0x1900 = (v16u8)__msa_fill_h(0x1900);
  v16u8 const_0x4281 = (v16u8)__msa_fill_h(0x4281);
  v8u16 const_0x1080 = (v8u16)__msa_fill_h(0x1080);

  for (x = 0; x < width; x += 16) {
    src0 = (v16u8)__msa_ld_b((const v16i8*)src_argb0, 0);
    src1 = (v16u8)__msa_ld_b((const v16i8*)src_argb0, 16);
    src2 = (v16u8)__msa_ld_b((const v16i8*)src_argb0, 32);
    src3 = (v16u8)__msa_ld_b((const v16i8*)src_argb0, 48);
    ARGBTOY(src0, src1, src2, src3, const_0x1900, const_0x4281, const_0x1080, 8,
            dst0);
    ST_UB(dst0, dst_y);
    src_argb0 += 64;
    dst_y += 16;
  }
}

void ARGBToUVJRow_MSA(const uint8_t* src_rgb0,
                      int src_stride_rgb,
                      uint8_t* dst_u,
                      uint8_t* dst_v,
                      int width) {
  int x;
  const uint8_t* s = src_rgb0;
  const uint8_t* t = src_rgb0 + src_stride_rgb;
  v16u8 src0, src1, src2, src3, src4, src5, src6, src7;
  v16u8 vec0, vec1, vec2, vec3;
  v16u8 dst0, dst1;
  v16i8 shuffler0 = {0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29};
  v16i8 shuffler1 = {2,  3,  6,  7,  10, 11, 14, 15,
                     18, 19, 22, 23, 26, 27, 30, 31};
  v16i8 shuffler2 = {0, 3, 4, 7, 8, 11, 12, 15, 16, 19, 20, 23, 24, 27, 28, 31};
  v16i8 shuffler3 = {1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22, 25, 26, 29, 30};
  v16u8 const_0x7F = (v16u8)__msa_fill_h(0x7F);
  v16u8 const_0x6B14 = (v16u8)__msa_fill_h(0x6B14);
  v16u8 const_0x2B54 = (v16u8)__msa_fill_h(0x2B54);
  v8u16 const_0x8080 = (v8u16)__msa_fill_h(0x8080);

  for (x = 0; x < width; x += 32) {
    src0 = (v16u8)__msa_ld_b((const v16i8*)s, 0);
    src1 = (v16u8)__msa_ld_b((const v16i8*)s, 16);
    src2 = (v16u8)__msa_ld_b((const v16i8*)s, 32);
    src3 = (v16u8)__msa_ld_b((const v16i8*)s, 48);
    src4 = (v16u8)__msa_ld_b((const v16i8*)t, 0);
    src5 = (v16u8)__msa_ld_b((const v16i8*)t, 16);
    src6 = (v16u8)__msa_ld_b((const v16i8*)t, 32);
    src7 = (v16u8)__msa_ld_b((const v16i8*)t, 48);
    src0 = __msa_aver_u_b(src0, src4);
    src1 = __msa_aver_u_b(src1, src5);
    src2 = __msa_aver_u_b(src2, src6);
    src3 = __msa_aver_u_b(src3, src7);
    src4 = (v16u8)__msa_pckev_w((v4i32)src1, (v4i32)src0);
    src5 = (v16u8)__msa_pckev_w((v4i32)src3, (v4i32)src2);
    src6 = (v16u8)__msa_pckod_w((v4i32)src1, (v4i32)src0);
    src7 = (v16u8)__msa_pckod_w((v4i32)src3, (v4i32)src2);
    vec0 = __msa_aver_u_b(src4, src6);
    vec1 = __msa_aver_u_b(src5, src7);
    src0 = (v16u8)__msa_ld_b((v16i8*)s, 64);
    src1 = (v16u8)__msa_ld_b((v16i8*)s, 80);
    src2 = (v16u8)__msa_ld_b((v16i8*)s, 96);
    src3 = (v16u8)__msa_ld_b((v16i8*)s, 112);
    src4 = (v16u8)__msa_ld_b((v16i8*)t, 64);
    src5 = (v16u8)__msa_ld_b((v16i8*)t, 80);
    src6 = (v16u8)__msa_ld_b((v16i8*)t, 96);
    src7 = (v16u8)__msa_ld_b((v16i8*)t, 112);
    src0 = __msa_aver_u_b(src0, src4);
    src1 = __msa_aver_u_b(src1, src5);
    src2 = __msa_aver_u_b(src2, src6);
    src3 = __msa_aver_u_b(src3, src7);
    src4 = (v16u8)__msa_pckev_w((v4i32)src1, (v4i32)src0);
    src5 = (v16u8)__msa_pckev_w((v4i32)src3, (v4i32)src2);
    src6 = (v16u8)__msa_pckod_w((v4i32)src1, (v4i32)src0);
    src7 = (v16u8)__msa_pckod_w((v4i32)src3, (v4i32)src2);
    vec2 = __msa_aver_u_b(src4, src6);
    vec3 = __msa_aver_u_b(src5, src7);
    ARGBTOUV(vec0, vec1, vec2, vec3, const_0x6B14, const_0x7F, const_0x2B54,
             const_0x8080, shuffler1, shuffler0, shuffler2, shuffler3, dst0,
             dst1);
    ST_UB(dst0, dst_v);
    ST_UB(dst1, dst_u);
    s += 128;
    t += 128;
    dst_v += 16;
    dst_u += 16;
  }
}

void BGRAToUVRow_MSA(const uint8_t* src_rgb0,
                     int src_stride_rgb,
                     uint8_t* dst_u,
                     uint8_t* dst_v,
                     int width) {
  int x;
  const uint8_t* s = src_rgb0;
  const uint8_t* t = src_rgb0 + src_stride_rgb;
  v16u8 dst0, dst1, vec0, vec1, vec2, vec3;
  v16i8 shuffler0 = {0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29};
  v16i8 shuffler1 = {2,  3,  6,  7,  10, 11, 14, 15,
                     18, 19, 22, 23, 26, 27, 30, 31};
  v16i8 shuffler2 = {0, 3, 4, 7, 8, 11, 12, 15, 16, 19, 20, 23, 24, 27, 28, 31};
  v16i8 shuffler3 = {2, 1, 6, 5, 10, 9, 14, 13, 18, 17, 22, 21, 26, 25, 30, 29};
  v16u8 const_0x125E = (v16u8)__msa_fill_h(0x125E);
  v16u8 const_0x7000 = (v16u8)__msa_fill_h(0x7000);
  v16u8 const_0x264A = (v16u8)__msa_fill_h(0x264A);
  v8u16 const_0x8080 = (v8u16)__msa_fill_h(0x8080);

  for (x = 0; x < width; x += 32) {
    READ_ARGB(s, t, vec0, vec1, vec2, vec3);
    ARGBTOUV(vec0, vec1, vec2, vec3, const_0x125E, const_0x7000, const_0x264A,
             const_0x8080, shuffler0, shuffler1, shuffler2, shuffler3, dst0,
             dst1);
    ST_UB(dst0, dst_v);
    ST_UB(dst1, dst_u);
    s += 128;
    t += 128;
    dst_v += 16;
    dst_u += 16;
  }
}

void ABGRToUVRow_MSA(const uint8_t* src_rgb0,
                     int src_stride_rgb,
                     uint8_t* dst_u,
                     uint8_t* dst_v,
                     int width) {
  int x;
  const uint8_t* s = src_rgb0;
  const uint8_t* t = src_rgb0 + src_stride_rgb;
  v16u8 src0, src1, src2, src3;
  v16u8 dst0, dst1;
  v16i8 shuffler0 = {0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29};
  v16i8 shuffler1 = {2,  3,  6,  7,  10, 11, 14, 15,
                     18, 19, 22, 23, 26, 27, 30, 31};
  v16i8 shuffler2 = {0, 3, 4, 7, 8, 11, 12, 15, 16, 19, 20, 23, 24, 27, 28, 31};
  v16i8 shuffler3 = {1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22, 25, 26, 29, 30};
  v16u8 const_0x4A26 = (v16u8)__msa_fill_h(0x4A26);
  v16u8 const_0x0070 = (v16u8)__msa_fill_h(0x0070);
  v16u8 const_0x125E = (v16u8)__msa_fill_h(0x125E);
  v8u16 const_0x8080 = (v8u16)__msa_fill_h(0x8080);

  for (x = 0; x < width; x += 32) {
    READ_ARGB(s, t, src0, src1, src2, src3);
    ARGBTOUV(src0, src1, src2, src3, const_0x4A26, const_0x0070, const_0x125E,
             const_0x8080, shuffler1, shuffler0, shuffler2, shuffler3, dst0,
             dst1);
    ST_UB(dst0, dst_u);
    ST_UB(dst1, dst_v);
    s += 128;
    t += 128;
    dst_u += 16;
    dst_v += 16;
  }
}

void RGBAToUVRow_MSA(const uint8_t* src_rgb0,
                     int src_stride_rgb,
                     uint8_t* dst_u,
                     uint8_t* dst_v,
                     int width) {
  int x;
  const uint8_t* s = src_rgb0;
  const uint8_t* t = src_rgb0 + src_stride_rgb;
  v16u8 dst0, dst1, vec0, vec1, vec2, vec3;
  v16i8 shuffler0 = {0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29};
  v16i8 shuffler1 = {2,  3,  6,  7,  10, 11, 14, 15,
                     18, 19, 22, 23, 26, 27, 30, 31};
  v16i8 shuffler2 = {0, 3, 4, 7, 8, 11, 12, 15, 16, 19, 20, 23, 24, 27, 28, 31};
  v16i8 shuffler3 = {2, 1, 6, 5, 10, 9, 14, 13, 18, 17, 22, 21, 26, 25, 30, 29};
  v16u8 const_0x125E = (v16u8)__msa_fill_h(0x264A);
  v16u8 const_0x7000 = (v16u8)__msa_fill_h(0x7000);
  v16u8 const_0x264A = (v16u8)__msa_fill_h(0x125E);
  v8u16 const_0x8080 = (v8u16)__msa_fill_h(0x8080);

  for (x = 0; x < width; x += 32) {
    READ_ARGB(s, t, vec0, vec1, vec2, vec3);
    ARGBTOUV(vec0, vec1, vec2, vec3, const_0x125E, const_0x7000, const_0x264A,
             const_0x8080, shuffler0, shuffler1, shuffler2, shuffler3, dst0,
             dst1);
    ST_UB(dst0, dst_u);
    ST_UB(dst1, dst_v);
    s += 128;
    t += 128;
    dst_u += 16;
    dst_v += 16;
  }
}

void I444ToARGBRow_MSA(const uint8_t* src_y,
                       const uint8_t* src_u,
                       const uint8_t* src_v,
                       uint8_t* dst_argb,
                       const struct YuvConstants* yuvconstants,
                       int width) {
  int x;
  v16u8 src0, src1, src2, dst0, dst1;
  v8u16 vec0, vec1, vec2;
  v4i32 reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8, reg9;
  v4i32 vec_ub, vec_vr, vec_ug, vec_vg, vec_bb, vec_bg, vec_br, vec_yg;
  v16u8 alpha = (v16u8)__msa_ldi_b(ALPHA_VAL);
  v8i16 zero = {0};

  YUVTORGB_SETUP(yuvconstants, vec_ub, vec_vr, vec_ug, vec_vg, vec_bb, vec_bg,
                 vec_br, vec_yg);

  for (x = 0; x < width; x += 8) {
    READI444(src_y, src_u, src_v, src0, src1, src2);
    vec0 = (v8u16)__msa_ilvr_b((v16i8)src0, (v16i8)src0);
    reg0 = (v4i32)__msa_ilvr_h((v8i16)zero, (v8i16)vec0);
    reg1 = (v4i32)__msa_ilvl_h((v8i16)zero, (v8i16)vec0);
    reg0 *= vec_yg;
    reg1 *= vec_yg;
    reg0 = __msa_srai_w(reg0, 16);
    reg1 = __msa_srai_w(reg1, 16);
    reg4 = reg0 + vec_br;
    reg5 = reg1 + vec_br;
    reg2 = reg0 + vec_bg;
    reg3 = reg1 + vec_bg;
    reg0 += vec_bb;
    reg1 += vec_bb;
    vec0 = (v8u16)__msa_ilvr_b((v16i8)zero, (v16i8)src1);
    vec1 = (v8u16)__msa_ilvr_b((v16i8)zero, (v16i8)src2);
    reg6 = (v4i32)__msa_ilvr_h((v8i16)zero, (v8i16)vec0);
    reg7 = (v4i32)__msa_ilvl_h((v8i16)zero, (v8i16)vec0);
    reg8 = (v4i32)__msa_ilvr_h((v8i16)zero, (v8i16)vec1);
    reg9 = (v4i32)__msa_ilvl_h((v8i16)zero, (v8i16)vec1);
    reg0 -= reg6 * vec_ub;
    reg1 -= reg7 * vec_ub;
    reg2 -= reg6 * vec_ug;
    reg3 -= reg7 * vec_ug;
    reg4 -= reg8 * vec_vr;
    reg5 -= reg9 * vec_vr;
    reg2 -= reg8 * vec_vg;
    reg3 -= reg9 * vec_vg;
    reg0 = __msa_srai_w(reg0, 6);
    reg1 = __msa_srai_w(reg1, 6);
    reg2 = __msa_srai_w(reg2, 6);
    reg3 = __msa_srai_w(reg3, 6);
    reg4 = __msa_srai_w(reg4, 6);
    reg5 = __msa_srai_w(reg5, 6);
    CLIP_0TO255(reg0, reg1, reg2, reg3, reg4, reg5);
    vec0 = (v8u16)__msa_pckev_h((v8i16)reg1, (v8i16)reg0);
    vec1 = (v8u16)__msa_pckev_h((v8i16)reg3, (v8i16)reg2);
    vec2 = (v8u16)__msa_pckev_h((v8i16)reg5, (v8i16)reg4);
    vec0 = (v8u16)__msa_ilvev_b((v16i8)vec1, (v16i8)vec0);
    vec1 = (v8u16)__msa_ilvev_b((v16i8)alpha, (v16i8)vec2);
    dst0 = (v16u8)__msa_ilvr_h((v8i16)vec1, (v8i16)vec0);
    dst1 = (v16u8)__msa_ilvl_h((v8i16)vec1, (v8i16)vec0);
    ST_UB2(dst0, dst1, dst_argb, 16);
    src_y += 8;
    src_u += 8;
    src_v += 8;
    dst_argb += 32;
  }
}

void I400ToARGBRow_MSA(const uint8_t* src_y, uint8_t* dst_argb, int width) {
  int x;
  v16u8 src0, res0, res1, res2, res3, res4, dst0, dst1, dst2, dst3;
  v8i16 vec0, vec1;
  v4i32 reg0, reg1, reg2, reg3;
  v4i32 vec_yg = __msa_fill_w(0x4A35);
  v8i16 vec_ygb = __msa_fill_h(0xFB78);
  v16u8 alpha = (v16u8)__msa_ldi_b(ALPHA_VAL);
  v8i16 max = __msa_ldi_h(0xFF);
  v8i16 zero = {0};

  for (x = 0; x < width; x += 16) {
    src0 = (v16u8)__msa_ld_b((const v16i8*)src_y, 0);
    vec0 = (v8i16)__msa_ilvr_b((v16i8)src0, (v16i8)src0);
    vec1 = (v8i16)__msa_ilvl_b((v16i8)src0, (v16i8)src0);
    reg0 = (v4i32)__msa_ilvr_h(zero, vec0);
    reg1 = (v4i32)__msa_ilvl_h(zero, vec0);
    reg2 = (v4i32)__msa_ilvr_h(zero, vec1);
    reg3 = (v4i32)__msa_ilvl_h(zero, vec1);
    reg0 *= vec_yg;
    reg1 *= vec_yg;
    reg2 *= vec_yg;
    reg3 *= vec_yg;
    reg0 = __msa_srai_w(reg0, 16);
    reg1 = __msa_srai_w(reg1, 16);
    reg2 = __msa_srai_w(reg2, 16);
    reg3 = __msa_srai_w(reg3, 16);
    vec0 = (v8i16)__msa_pckev_h((v8i16)reg1, (v8i16)reg0);
    vec1 = (v8i16)__msa_pckev_h((v8i16)reg3, (v8i16)reg2);
    vec0 += vec_ygb;
    vec1 += vec_ygb;
    vec0 = __msa_srai_h(vec0, 6);
    vec1 = __msa_srai_h(vec1, 6);
    vec0 = __msa_maxi_s_h(vec0, 0);
    vec1 = __msa_maxi_s_h(vec1, 0);
    vec0 = __msa_min_s_h(max, vec0);
    vec1 = __msa_min_s_h(max, vec1);
    res0 = (v16u8)__msa_pckev_b((v16i8)vec1, (v16i8)vec0);
    res1 = (v16u8)__msa_ilvr_b((v16i8)res0, (v16i8)res0);
    res2 = (v16u8)__msa_ilvl_b((v16i8)res0, (v16i8)res0);
    res3 = (v16u8)__msa_ilvr_b((v16i8)alpha, (v16i8)res0);
    res4 = (v16u8)__msa_ilvl_b((v16i8)alpha, (v16i8)res0);
    dst0 = (v16u8)__msa_ilvr_b((v16i8)res3, (v16i8)res1);
    dst1 = (v16u8)__msa_ilvl_b((v16i8)res3, (v16i8)res1);
    dst2 = (v16u8)__msa_ilvr_b((v16i8)res4, (v16i8)res2);
    dst3 = (v16u8)__msa_ilvl_b((v16i8)res4, (v16i8)res2);
    ST_UB4(dst0, dst1, dst2, dst3, dst_argb, 16);
    src_y += 16;
    dst_argb += 64;
  }
}

void J400ToARGBRow_MSA(const uint8_t* src_y, uint8_t* dst_argb, int width) {
  int x;
  v16u8 src0, vec0, vec1, vec2, vec3, dst0, dst1, dst2, dst3;
  v16u8 alpha = (v16u8)__msa_ldi_b(ALPHA_VAL);

  for (x = 0; x < width; x += 16) {
    src0 = (v16u8)__msa_ld_b((const v16i8*)src_y, 0);
    vec0 = (v16u8)__msa_ilvr_b((v16i8)src0, (v16i8)src0);
    vec1 = (v16u8)__msa_ilvl_b((v16i8)src0, (v16i8)src0);
    vec2 = (v16u8)__msa_ilvr_b((v16i8)alpha, (v16i8)src0);
    vec3 = (v16u8)__msa_ilvl_b((v16i8)alpha, (v16i8)src0);
    dst0 = (v16u8)__msa_ilvr_b((v16i8)vec2, (v16i8)vec0);
    dst1 = (v16u8)__msa_ilvl_b((v16i8)vec2, (v16i8)vec0);
    dst2 = (v16u8)__msa_ilvr_b((v16i8)vec3, (v16i8)vec1);
    dst3 = (v16u8)__msa_ilvl_b((v16i8)vec3, (v16i8)vec1);
    ST_UB4(dst0, dst1, dst2, dst3, dst_argb, 16);
    src_y += 16;
    dst_argb += 64;
  }
}

void YUY2ToARGBRow_MSA(const uint8_t* src_yuy2,
                       uint8_t* dst_argb,
                       const struct YuvConstants* yuvconstants,
                       int width) {
  int x;
  v16u8 src0, src1, src2;
  v8i16 vec0, vec1, vec2;
  v4i32 vec_ub, vec_vr, vec_ug, vec_vg, vec_bb, vec_bg, vec_br, vec_yg;
  v4i32 vec_ubvr, vec_ugvg;
  v16u8 alpha = (v16u8)__msa_ldi_b(ALPHA_VAL);

  YUVTORGB_SETUP(yuvconstants, vec_ub, vec_vr, vec_ug, vec_vg, vec_bb, vec_bg,
                 vec_br, vec_yg);
  vec_ubvr = __msa_ilvr_w(vec_vr, vec_ub);
  vec_ugvg = (v4i32)__msa_ilvev_h((v8i16)vec_vg, (v8i16)vec_ug);

  for (x = 0; x < width; x += 8) {
    src0 = (v16u8)__msa_ld_b((const v16i8*)src_yuy2, 0);
    src1 = (v16u8)__msa_pckev_b((v16i8)src0, (v16i8)src0);
    src2 = (v16u8)__msa_pckod_b((v16i8)src0, (v16i8)src0);
    YUVTORGB(src1, src2, vec_ubvr, vec_ugvg, vec_bb, vec_bg, vec_br, vec_yg,
             vec0, vec1, vec2);
    STOREARGB(vec0, vec1, vec2, alpha, dst_argb);
    src_yuy2 += 16;
    dst_argb += 32;
  }
}

void UYVYToARGBRow_MSA(const uint8_t* src_uyvy,
                       uint8_t* dst_argb,
                       const struct YuvConstants* yuvconstants,
                       int width) {
  int x;
  v16u8 src0, src1, src2;
  v8i16 vec0, vec1, vec2;
  v4i32 vec_ub, vec_vr, vec_ug, vec_vg, vec_bb, vec_bg, vec_br, vec_yg;
  v4i32 vec_ubvr, vec_ugvg;
  v16u8 alpha = (v16u8)__msa_ldi_b(ALPHA_VAL);

  YUVTORGB_SETUP(yuvconstants, vec_ub, vec_vr, vec_ug, vec_vg, vec_bb, vec_bg,
                 vec_br, vec_yg);
  vec_ubvr = __msa_ilvr_w(vec_vr, vec_ub);
  vec_ugvg = (v4i32)__msa_ilvev_h((v8i16)vec_vg, (v8i16)vec_ug);

  for (x = 0; x < width; x += 8) {
    src0 = (v16u8)__msa_ld_b((const v16i8*)src_uyvy, 0);
    src1 = (v16u8)__msa_pckod_b((v16i8)src0, (v16i8)src0);
    src2 = (v16u8)__msa_pckev_b((v16i8)src0, (v16i8)src0);
    YUVTORGB(src1, src2, vec_ubvr, vec_ugvg, vec_bb, vec_bg, vec_br, vec_yg,
             vec0, vec1, vec2);
    STOREARGB(vec0, vec1, vec2, alpha, dst_argb);
    src_uyvy += 16;
    dst_argb += 32;
  }
}

void InterpolateRow_MSA(uint8_t* dst_ptr,
                        const uint8_t* src_ptr,
                        ptrdiff_t src_stride,
                        int width,
                        int32_t source_y_fraction) {
  int32_t y1_fraction = source_y_fraction;
  int32_t y0_fraction = 256 - y1_fraction;
  uint16_t y_fractions;
  const uint8_t* s = src_ptr;
  const uint8_t* t = src_ptr + src_stride;
  int x;
  v16u8 src0, src1, src2, src3, dst0, dst1;
  v8u16 vec0, vec1, vec2, vec3, y_frac;

  if (0 == y1_fraction) {
    memcpy(dst_ptr, src_ptr, width);
    return;
  }

  if (128 == y1_fraction) {
    for (x = 0; x < width; x += 32) {
      src0 = (v16u8)__msa_ld_b((const v16i8*)s, 0);
      src1 = (v16u8)__msa_ld_b((const v16i8*)s, 16);
      src2 = (v16u8)__msa_ld_b((const v16i8*)t, 0);
      src3 = (v16u8)__msa_ld_b((const v16i8*)t, 16);
      dst0 = __msa_aver_u_b(src0, src2);
      dst1 = __msa_aver_u_b(src1, src3);
      ST_UB2(dst0, dst1, dst_ptr, 16);
      s += 32;
      t += 32;
      dst_ptr += 32;
    }
    return;
  }

  y_fractions = (uint16_t)(y0_fraction + (y1_fraction << 8));
  y_frac = (v8u16)__msa_fill_h(y_fractions);

  for (x = 0; x < width; x += 32) {
    src0 = (v16u8)__msa_ld_b((const v16i8*)s, 0);
    src1 = (v16u8)__msa_ld_b((const v16i8*)s, 16);
    src2 = (v16u8)__msa_ld_b((const v16i8*)t, 0);
    src3 = (v16u8)__msa_ld_b((const v16i8*)t, 16);
    vec0 = (v8u16)__msa_ilvr_b((v16i8)src2, (v16i8)src0);
    vec1 = (v8u16)__msa_ilvl_b((v16i8)src2, (v16i8)src0);
    vec2 = (v8u16)__msa_ilvr_b((v16i8)src3, (v16i8)src1);
    vec3 = (v8u16)__msa_ilvl_b((v16i8)src3, (v16i8)src1);
    vec0 = (v8u16)__msa_dotp_u_h((v16u8)vec0, (v16u8)y_frac);
    vec1 = (v8u16)__msa_dotp_u_h((v16u8)vec1, (v16u8)y_frac);
    vec2 = (v8u16)__msa_dotp_u_h((v16u8)vec2, (v16u8)y_frac);
    vec3 = (v8u16)__msa_dotp_u_h((v16u8)vec3, (v16u8)y_frac);
    vec0 = (v8u16)__msa_srari_h((v8i16)vec0, 8);
    vec1 = (v8u16)__msa_srari_h((v8i16)vec1, 8);
    vec2 = (v8u16)__msa_srari_h((v8i16)vec2, 8);
    vec3 = (v8u16)__msa_srari_h((v8i16)vec3, 8);
    dst0 = (v16u8)__msa_pckev_b((v16i8)vec1, (v16i8)vec0);
    dst1 = (v16u8)__msa_pckev_b((v16i8)vec3, (v16i8)vec2);
    ST_UB2(dst0, dst1, dst_ptr, 16);
    s += 32;
    t += 32;
    dst_ptr += 32;
  }
}

void ARGBSetRow_MSA(uint8_t* dst_argb, uint32_t v32, int width) {
  int x;
  v4i32 dst0 = __builtin_msa_fill_w(v32);

  for (x = 0; x < width; x += 4) {
    ST_UB(dst0, dst_argb);
    dst_argb += 16;
  }
}

void RAWToRGB24Row_MSA(const uint8_t* src_raw, uint8_t* dst_rgb24, int width) {
  int x;
  v16u8 src0, src1, src2, src3, src4, dst0, dst1, dst2;
  v16i8 shuffler0 = {2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9, 14, 13, 12, 17};
  v16i8 shuffler1 = {8,  7,  12, 11, 10, 15, 14, 13,
                     18, 17, 16, 21, 20, 19, 24, 23};
  v16i8 shuffler2 = {14, 19, 18, 17, 22, 21, 20, 25,
                     24, 23, 28, 27, 26, 31, 30, 29};

  for (x = 0; x < width; x += 16) {
    src0 = (v16u8)__msa_ld_b((const v16i8*)src_raw, 0);
    src1 = (v16u8)__msa_ld_b((const v16i8*)src_raw, 16);
    src2 = (v16u8)__msa_ld_b((const v16i8*)src_raw, 32);
    src3 = (v16u8)__msa_sldi_b((v16i8)src1, (v16i8)src0, 8);
    src4 = (v16u8)__msa_sldi_b((v16i8)src2, (v16i8)src1, 8);
    dst0 = (v16u8)__msa_vshf_b(shuffler0, (v16i8)src1, (v16i8)src0);
    dst1 = (v16u8)__msa_vshf_b(shuffler1, (v16i8)src4, (v16i8)src3);
    dst2 = (v16u8)__msa_vshf_b(shuffler2, (v16i8)src2, (v16i8)src1);
    ST_UB2(dst0, dst1, dst_rgb24, 16);
    ST_UB(dst2, (dst_rgb24 + 32));
    src_raw += 48;
    dst_rgb24 += 48;
  }
}

void MergeUVRow_MSA(const uint8_t* src_u,
                    const uint8_t* src_v,
                    uint8_t* dst_uv,
                    int width) {
  int x;
  v16u8 src0, src1, dst0, dst1;

  for (x = 0; x < width; x += 16) {
    src0 = (v16u8)__msa_ld_b((const v16i8*)src_u, 0);
    src1 = (v16u8)__msa_ld_b((const v16i8*)src_v, 0);
    dst0 = (v16u8)__msa_ilvr_b((v16i8)src1, (v16i8)src0);
    dst1 = (v16u8)__msa_ilvl_b((v16i8)src1, (v16i8)src0);
    ST_UB2(dst0, dst1, dst_uv, 16);
    src_u += 16;
    src_v += 16;
    dst_uv += 32;
  }
}

void ARGBExtractAlphaRow_MSA(const uint8_t* src_argb,
                             uint8_t* dst_a,
                             int width) {
  int i;
  v16u8 src0, src1, src2, src3, vec0, vec1, dst0;

  for (i = 0; i < width; i += 16) {
    src0 = (v16u8)__msa_ld_b((const v16i8*)src_argb, 0);
    src1 = (v16u8)__msa_ld_b((const v16i8*)src_argb, 16);
    src2 = (v16u8)__msa_ld_b((const v16i8*)src_argb, 32);
    src3 = (v16u8)__msa_ld_b((const v16i8*)src_argb, 48);
    vec0 = (v16u8)__msa_pckod_b((v16i8)src1, (v16i8)src0);
    vec1 = (v16u8)__msa_pckod_b((v16i8)src3, (v16i8)src2);
    dst0 = (v16u8)__msa_pckod_b((v16i8)vec1, (v16i8)vec0);
    ST_UB(dst0, dst_a);
    src_argb += 64;
    dst_a += 16;
  }
}

void ARGBBlendRow_MSA(const uint8_t* src_argb0,
                      const uint8_t* src_argb1,
                      uint8_t* dst_argb,
                      int width) {
  int x;
  v16u8 src0, src1, src2, src3, dst0, dst1;
  v8u16 vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7;
  v8u16 vec8, vec9, vec10, vec11, vec12, vec13;
  v8u16 const_256 = (v8u16)__msa_ldi_h(256);
  v16u8 const_255 = (v16u8)__msa_ldi_b(255);
  v16u8 mask = {0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255};
  v16i8 zero = {0};

  for (x = 0; x < width; x += 8) {
    src0 = (v16u8)__msa_ld_b((const v16i8*)src_argb0, 0);
    src1 = (v16u8)__msa_ld_b((const v16i8*)src_argb0, 16);
    src2 = (v16u8)__msa_ld_b((const v16i8*)src_argb1, 0);
    src3 = (v16u8)__msa_ld_b((const v16i8*)src_argb1, 16);
    vec0 = (v8u16)__msa_ilvr_b(zero, (v16i8)src0);
    vec1 = (v8u16)__msa_ilvl_b(zero, (v16i8)src0);
    vec2 = (v8u16)__msa_ilvr_b(zero, (v16i8)src1);
    vec3 = (v8u16)__msa_ilvl_b(zero, (v16i8)src1);
    vec4 = (v8u16)__msa_ilvr_b(zero, (v16i8)src2);
    vec5 = (v8u16)__msa_ilvl_b(zero, (v16i8)src2);
    vec6 = (v8u16)__msa_ilvr_b(zero, (v16i8)src3);
    vec7 = (v8u16)__msa_ilvl_b(zero, (v16i8)src3);
    vec8 = (v8u16)__msa_fill_h(vec0[3]);
    vec9 = (v8u16)__msa_fill_h(vec0[7]);
    vec10 = (v8u16)__msa_fill_h(vec1[3]);
    vec11 = (v8u16)__msa_fill_h(vec1[7]);
    vec8 = (v8u16)__msa_pckev_d((v2i64)vec9, (v2i64)vec8);
    vec9 = (v8u16)__msa_pckev_d((v2i64)vec11, (v2i64)vec10);
    vec10 = (v8u16)__msa_fill_h(vec2[3]);
    vec11 = (v8u16)__msa_fill_h(vec2[7]);
    vec12 = (v8u16)__msa_fill_h(vec3[3]);
    vec13 = (v8u16)__msa_fill_h(vec3[7]);
    vec10 = (v8u16)__msa_pckev_d((v2i64)vec11, (v2i64)vec10);
    vec11 = (v8u16)__msa_pckev_d((v2i64)vec13, (v2i64)vec12);
    vec8 = const_256 - vec8;
    vec9 = const_256 - vec9;
    vec10 = const_256 - vec10;
    vec11 = const_256 - vec11;
    vec8 *= vec4;
    vec9 *= vec5;
    vec10 *= vec6;
    vec11 *= vec7;
    vec8 = (v8u16)__msa_srai_h((v8i16)vec8, 8);
    vec9 = (v8u16)__msa_srai_h((v8i16)vec9, 8);
    vec10 = (v8u16)__msa_srai_h((v8i16)vec10, 8);
    vec11 = (v8u16)__msa_srai_h((v8i16)vec11, 8);
    vec0 += vec8;
    vec1 += vec9;
    vec2 += vec10;
    vec3 += vec11;
    dst0 = (v16u8)__msa_pckev_b((v16i8)vec1, (v16i8)vec0);
    dst1 = (v16u8)__msa_pckev_b((v16i8)vec3, (v16i8)vec2);
    dst0 = __msa_bmnz_v(dst0, const_255, mask);
    dst1 = __msa_bmnz_v(dst1, const_255, mask);
    ST_UB2(dst0, dst1, dst_argb, 16);
    src_argb0 += 32;
    src_argb1 += 32;
    dst_argb += 32;
  }
}

void ARGBQuantizeRow_MSA(uint8_t* dst_argb,
                         int scale,
                         int interval_size,
                         int interval_offset,
                         int width) {
  int x;
  v16u8 src0, src1, src2, src3, dst0, dst1, dst2, dst3;
  v8i16 vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7;
  v4i32 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
  v4i32 tmp8, tmp9, tmp10, tmp11, tmp12, tmp13, tmp14, tmp15;
  v4i32 vec_scale = __msa_fill_w(scale);
  v16u8 vec_int_sz = (v16u8)__msa_fill_b(interval_size);
  v16u8 vec_int_ofst = (v16u8)__msa_fill_b(interval_offset);
  v16i8 mask = {0, 1, 2, 19, 4, 5, 6, 23, 8, 9, 10, 27, 12, 13, 14, 31};
  v16i8 zero = {0};

  for (x = 0; x < width; x += 8) {
    src0 = (v16u8)__msa_ld_b((v16i8*)dst_argb, 0);
    src1 = (v16u8)__msa_ld_b((v16i8*)dst_argb, 16);
    src2 = (v16u8)__msa_ld_b((v16i8*)dst_argb, 32);
    src3 = (v16u8)__msa_ld_b((v16i8*)dst_argb, 48);
    vec0 = (v8i16)__msa_ilvr_b(zero, (v16i8)src0);
    vec1 = (v8i16)__msa_ilvl_b(zero, (v16i8)src0);
    vec2 = (v8i16)__msa_ilvr_b(zero, (v16i8)src1);
    vec3 = (v8i16)__msa_ilvl_b(zero, (v16i8)src1);
    vec4 = (v8i16)__msa_ilvr_b(zero, (v16i8)src2);
    vec5 = (v8i16)__msa_ilvl_b(zero, (v16i8)src2);
    vec6 = (v8i16)__msa_ilvr_b(zero, (v16i8)src3);
    vec7 = (v8i16)__msa_ilvl_b(zero, (v16i8)src3);
    tmp0 = (v4i32)__msa_ilvr_h((v8i16)zero, (v8i16)vec0);
    tmp1 = (v4i32)__msa_ilvl_h((v8i16)zero, (v8i16)vec0);
    tmp2 = (v4i32)__msa_ilvr_h((v8i16)zero, (v8i16)vec1);
    tmp3 = (v4i32)__msa_ilvl_h((v8i16)zero, (v8i16)vec1);
    tmp4 = (v4i32)__msa_ilvr_h((v8i16)zero, (v8i16)vec2);
    tmp5 = (v4i32)__msa_ilvl_h((v8i16)zero, (v8i16)vec2);
    tmp6 = (v4i32)__msa_ilvr_h((v8i16)zero, (v8i16)vec3);
    tmp7 = (v4i32)__msa_ilvl_h((v8i16)zero, (v8i16)vec3);
    tmp8 = (v4i32)__msa_ilvr_h((v8i16)zero, (v8i16)vec4);
    tmp9 = (v4i32)__msa_ilvl_h((v8i16)zero, (v8i16)vec4);
    tmp10 = (v4i32)__msa_ilvr_h((v8i16)zero, (v8i16)vec5);
    tmp11 = (v4i32)__msa_ilvl_h((v8i16)zero, (v8i16)vec5);
    tmp12 = (v4i32)__msa_ilvr_h((v8i16)zero, (v8i16)vec6);
    tmp13 = (v4i32)__msa_ilvl_h((v8i16)zero, (v8i16)vec6);
    tmp14 = (v4i32)__msa_ilvr_h((v8i16)zero, (v8i16)vec7);
    tmp15 = (v4i32)__msa_ilvl_h((v8i16)zero, (v8i16)vec7);
    tmp0 *= vec_scale;
    tmp1 *= vec_scale;
    tmp2 *= vec_scale;
    tmp3 *= vec_scale;
    tmp4 *= vec_scale;
    tmp5 *= vec_scale;
    tmp6 *= vec_scale;
    tmp7 *= vec_scale;
    tmp8 *= vec_scale;
    tmp9 *= vec_scale;
    tmp10 *= vec_scale;
    tmp11 *= vec_scale;
    tmp12 *= vec_scale;
    tmp13 *= vec_scale;
    tmp14 *= vec_scale;
    tmp15 *= vec_scale;
    tmp0 >>= 16;
    tmp1 >>= 16;
    tmp2 >>= 16;
    tmp3 >>= 16;
    tmp4 >>= 16;
    tmp5 >>= 16;
    tmp6 >>= 16;
    tmp7 >>= 16;
    tmp8 >>= 16;
    tmp9 >>= 16;
    tmp10 >>= 16;
    tmp11 >>= 16;
    tmp12 >>= 16;
    tmp13 >>= 16;
    tmp14 >>= 16;
    tmp15 >>= 16;
    vec0 = (v8i16)__msa_pckev_h((v8i16)tmp1, (v8i16)tmp0);
    vec1 = (v8i16)__msa_pckev_h((v8i16)tmp3, (v8i16)tmp2);
    vec2 = (v8i16)__msa_pckev_h((v8i16)tmp5, (v8i16)tmp4);
    vec3 = (v8i16)__msa_pckev_h((v8i16)tmp7, (v8i16)tmp6);
    vec4 = (v8i16)__msa_pckev_h((v8i16)tmp9, (v8i16)tmp8);
    vec5 = (v8i16)__msa_pckev_h((v8i16)tmp11, (v8i16)tmp10);
    vec6 = (v8i16)__msa_pckev_h((v8i16)tmp13, (v8i16)tmp12);
    vec7 = (v8i16)__msa_pckev_h((v8i16)tmp15, (v8i16)tmp14);
    dst0 = (v16u8)__msa_pckev_b((v16i8)vec1, (v16i8)vec0);
    dst1 = (v16u8)__msa_pckev_b((v16i8)vec3, (v16i8)vec2);
    dst2 = (v16u8)__msa_pckev_b((v16i8)vec5, (v16i8)vec4);
    dst3 = (v16u8)__msa_pckev_b((v16i8)vec7, (v16i8)vec6);
    dst0 *= vec_int_sz;
    dst1 *= vec_int_sz;
    dst2 *= vec_int_sz;
    dst3 *= vec_int_sz;
    dst0 += vec_int_ofst;
    dst1 += vec_int_ofst;
    dst2 += vec_int_ofst;
    dst3 += vec_int_ofst;
    dst0 = (v16u8)__msa_vshf_b(mask, (v16i8)src0, (v16i8)dst0);
    dst1 = (v16u8)__msa_vshf_b(mask, (v16i8)src1, (v16i8)dst1);
    dst2 = (v16u8)__msa_vshf_b(mask, (v16i8)src2, (v16i8)dst2);
    dst3 = (v16u8)__msa_vshf_b(mask, (v16i8)src3, (v16i8)dst3);
    ST_UB4(dst0, dst1, dst2, dst3, dst_argb, 16);
    dst_argb += 64;
  }
}

void ARGBColorMatrixRow_MSA(const uint8_t* src_argb,
                            uint8_t* dst_argb,
                            const int8_t* matrix_argb,
                            int width) {
  int32_t x;
  v16i8 src0;
  v16u8 src1, src2, dst0, dst1;
  v8i16 vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, vec9;
  v8i16 vec10, vec11, vec12, vec13, vec14, vec15, vec16, vec17;
  v4i32 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
  v4i32 tmp8, tmp9, tmp10, tmp11, tmp12, tmp13, tmp14, tmp15;
  v16i8 zero = {0};
  v8i16 max = __msa_ldi_h(255);

  src0 = __msa_ld_b((v16i8*)matrix_argb, 0);
  vec0 = (v8i16)__msa_ilvr_b(zero, src0);
  vec1 = (v8i16)__msa_ilvl_b(zero, src0);

  for (x = 0; x < width; x += 8) {
    src1 = (v16u8)__msa_ld_b((const v16i8*)src_argb, 0);
    src2 = (v16u8)__msa_ld_b((const v16i8*)src_argb, 16);
    vec2 = (v8i16)__msa_ilvr_b(zero, (v16i8)src1);
    vec3 = (v8i16)__msa_ilvl_b(zero, (v16i8)src1);
    vec4 = (v8i16)__msa_ilvr_b(zero, (v16i8)src2);
    vec5 = (v8i16)__msa_ilvl_b(zero, (v16i8)src2);
    vec6 = (v8i16)__msa_pckod_d((v2i64)vec2, (v2i64)vec2);
    vec7 = (v8i16)__msa_pckod_d((v2i64)vec3, (v2i64)vec3);
    vec8 = (v8i16)__msa_pckod_d((v2i64)vec4, (v2i64)vec4);
    vec9 = (v8i16)__msa_pckod_d((v2i64)vec5, (v2i64)vec5);
    vec2 = (v8i16)__msa_pckev_d((v2i64)vec2, (v2i64)vec2);
    vec3 = (v8i16)__msa_pckev_d((v2i64)vec3, (v2i64)vec3);
    vec4 = (v8i16)__msa_pckev_d((v2i64)vec4, (v2i64)vec4);
    vec5 = (v8i16)__msa_pckev_d((v2i64)vec5, (v2i64)vec5);
    vec10 = vec2 * vec0;
    vec11 = vec2 * vec1;
    vec12 = vec6 * vec0;
    vec13 = vec6 * vec1;
    tmp0 = __msa_hadd_s_w(vec10, vec10);
    tmp1 = __msa_hadd_s_w(vec11, vec11);
    tmp2 = __msa_hadd_s_w(vec12, vec12);
    tmp3 = __msa_hadd_s_w(vec13, vec13);
    vec14 = vec3 * vec0;
    vec15 = vec3 * vec1;
    vec16 = vec7 * vec0;
    vec17 = vec7 * vec1;
    tmp4 = __msa_hadd_s_w(vec14, vec14);
    tmp5 = __msa_hadd_s_w(vec15, vec15);
    tmp6 = __msa_hadd_s_w(vec16, vec16);
    tmp7 = __msa_hadd_s_w(vec17, vec17);
    vec10 = __msa_pckev_h((v8i16)tmp1, (v8i16)tmp0);
    vec11 = __msa_pckev_h((v8i16)tmp3, (v8i16)tmp2);
    vec12 = __msa_pckev_h((v8i16)tmp5, (v8i16)tmp4);
    vec13 = __msa_pckev_h((v8i16)tmp7, (v8i16)tmp6);
    tmp0 = __msa_hadd_s_w(vec10, vec10);
    tmp1 = __msa_hadd_s_w(vec11, vec11);
    tmp2 = __msa_hadd_s_w(vec12, vec12);
    tmp3 = __msa_hadd_s_w(vec13, vec13);
    tmp0 = __msa_srai_w(tmp0, 6);
    tmp1 = __msa_srai_w(tmp1, 6);
    tmp2 = __msa_srai_w(tmp2, 6);
    tmp3 = __msa_srai_w(tmp3, 6);
    vec2 = vec4 * vec0;
    vec6 = vec4 * vec1;
    vec3 = vec8 * vec0;
    vec7 = vec8 * vec1;
    tmp8 = __msa_hadd_s_w(vec2, vec2);
    tmp9 = __msa_hadd_s_w(vec6, vec6);
    tmp10 = __msa_hadd_s_w(vec3, vec3);
    tmp11 = __msa_hadd_s_w(vec7, vec7);
    vec4 = vec5 * vec0;
    vec8 = vec5 * vec1;
    vec5 = vec9 * vec0;
    vec9 = vec9 * vec1;
    tmp12 = __msa_hadd_s_w(vec4, vec4);
    tmp13 = __msa_hadd_s_w(vec8, vec8);
    tmp14 = __msa_hadd_s_w(vec5, vec5);
    tmp15 = __msa_hadd_s_w(vec9, vec9);
    vec14 = __msa_pckev_h((v8i16)tmp9, (v8i16)tmp8);
    vec15 = __msa_pckev_h((v8i16)tmp11, (v8i16)tmp10);
    vec16 = __msa_pckev_h((v8i16)tmp13, (v8i16)tmp12);
    vec17 = __msa_pckev_h((v8i16)tmp15, (v8i16)tmp14);
    tmp4 = __msa_hadd_s_w(vec14, vec14);
    tmp5 = __msa_hadd_s_w(vec15, vec15);
    tmp6 = __msa_hadd_s_w(vec16, vec16);
    tmp7 = __msa_hadd_s_w(vec17, vec17);
    tmp4 = __msa_srai_w(tmp4, 6);
    tmp5 = __msa_srai_w(tmp5, 6);
    tmp6 = __msa_srai_w(tmp6, 6);
    tmp7 = __msa_srai_w(tmp7, 6);
    vec10 = __msa_pckev_h((v8i16)tmp1, (v8i16)tmp0);
    vec11 = __msa_pckev_h((v8i16)tmp3, (v8i16)tmp2);
    vec12 = __msa_pckev_h((v8i16)tmp5, (v8i16)tmp4);
    vec13 = __msa_pckev_h((v8i16)tmp7, (v8i16)tmp6);
    vec10 = __msa_maxi_s_h(vec10, 0);
    vec11 = __msa_maxi_s_h(vec11, 0);
    vec12 = __msa_maxi_s_h(vec12, 0);
    vec13 = __msa_maxi_s_h(vec13, 0);
    vec10 = __msa_min_s_h(vec10, max);
    vec11 = __msa_min_s_h(vec11, max);
    vec12 = __msa_min_s_h(vec12, max);
    vec13 = __msa_min_s_h(vec13, max);
    dst0 = (v16u8)__msa_pckev_b((v16i8)vec11, (v16i8)vec10);
    dst1 = (v16u8)__msa_pckev_b((v16i8)vec13, (v16i8)vec12);
    ST_UB2(dst0, dst1, dst_argb, 16);
    src_argb += 32;
    dst_argb += 32;
  }
}

void SplitUVRow_MSA(const uint8_t* src_uv,
                    uint8_t* dst_u,
                    uint8_t* dst_v,
                    int width) {
  int x;
  v16u8 src0, src1, src2, src3, dst0, dst1, dst2, dst3;

  for (x = 0; x < width; x += 32) {
    src0 = (v16u8)__msa_ld_b((const v16i8*)src_uv, 0);
    src1 = (v16u8)__msa_ld_b((const v16i8*)src_uv, 16);
    src2 = (v16u8)__msa_ld_b((const v16i8*)src_uv, 32);
    src3 = (v16u8)__msa_ld_b((const v16i8*)src_uv, 48);
    dst0 = (v16u8)__msa_pckev_b((v16i8)src1, (v16i8)src0);
    dst1 = (v16u8)__msa_pckev_b((v16i8)src3, (v16i8)src2);
    dst2 = (v16u8)__msa_pckod_b((v16i8)src1, (v16i8)src0);
    dst3 = (v16u8)__msa_pckod_b((v16i8)src3, (v16i8)src2);
    ST_UB2(dst0, dst1, dst_u, 16);
    ST_UB2(dst2, dst3, dst_v, 16);
    src_uv += 64;
    dst_u += 32;
    dst_v += 32;
  }
}

void SetRow_MSA(uint8_t* dst, uint8_t v8, int width) {
  int x;
  v16u8 dst0 = (v16u8)__msa_fill_b(v8);

  for (x = 0; x < width; x += 16) {
    ST_UB(dst0, dst);
    dst += 16;
  }
}

void MirrorUVRow_MSA(const uint8_t* src_uv,
                     uint8_t* dst_u,
                     uint8_t* dst_v,
                     int width) {
  int x;
  v16u8 src0, src1, src2, src3;
  v16u8 dst0, dst1, dst2, dst3;
  v16i8 mask0 = {30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0};
  v16i8 mask1 = {31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1};

  src_uv += (2 * width);

  for (x = 0; x < width; x += 32) {
    src_uv -= 64;
    src2 = (v16u8)__msa_ld_b((const v16i8*)src_uv, 0);
    src3 = (v16u8)__msa_ld_b((const v16i8*)src_uv, 16);
    src0 = (v16u8)__msa_ld_b((const v16i8*)src_uv, 32);
    src1 = (v16u8)__msa_ld_b((const v16i8*)src_uv, 48);
    dst0 = (v16u8)__msa_vshf_b(mask1, (v16i8)src1, (v16i8)src0);
    dst1 = (v16u8)__msa_vshf_b(mask1, (v16i8)src3, (v16i8)src2);
    dst2 = (v16u8)__msa_vshf_b(mask0, (v16i8)src1, (v16i8)src0);
    dst3 = (v16u8)__msa_vshf_b(mask0, (v16i8)src3, (v16i8)src2);
    ST_UB2(dst0, dst1, dst_v, 16);
    ST_UB2(dst2, dst3, dst_u, 16);
    dst_u += 32;
    dst_v += 32;
  }
}

void SobelXRow_MSA(const uint8_t* src_y0,
                   const uint8_t* src_y1,
                   const uint8_t* src_y2,
                   uint8_t* dst_sobelx,
                   int32_t width) {
  int x;
  v16u8 src0, src1, src2, src3, src4, src5, dst0;
  v8i16 vec0, vec1, vec2, vec3, vec4, vec5;
  v16i8 mask0 = {0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9};
  v16i8 tmp = __msa_ldi_b(8);
  v16i8 mask1 = mask0 + tmp;
  v8i16 zero = {0};
  v8i16 max = __msa_ldi_h(255);

  for (x = 0; x < width; x += 16) {
    src0 = (v16u8)__msa_ld_b((const v16i8*)src_y0, 0);
    src1 = (v16u8)__msa_ld_b((const v16i8*)src_y0, 16);
    src2 = (v16u8)__msa_ld_b((const v16i8*)src_y1, 0);
    src3 = (v16u8)__msa_ld_b((const v16i8*)src_y1, 16);
    src4 = (v16u8)__msa_ld_b((const v16i8*)src_y2, 0);
    src5 = (v16u8)__msa_ld_b((const v16i8*)src_y2, 16);
    vec0 = (v8i16)__msa_vshf_b(mask0, (v16i8)src1, (v16i8)src0);
    vec1 = (v8i16)__msa_vshf_b(mask1, (v16i8)src1, (v16i8)src0);
    vec2 = (v8i16)__msa_vshf_b(mask0, (v16i8)src3, (v16i8)src2);
    vec3 = (v8i16)__msa_vshf_b(mask1, (v16i8)src3, (v16i8)src2);
    vec4 = (v8i16)__msa_vshf_b(mask0, (v16i8)src5, (v16i8)src4);
    vec5 = (v8i16)__msa_vshf_b(mask1, (v16i8)src5, (v16i8)src4);
    vec0 = (v8i16)__msa_hsub_u_h((v16u8)vec0, (v16u8)vec0);
    vec1 = (v8i16)__msa_hsub_u_h((v16u8)vec1, (v16u8)vec1);
    vec2 = (v8i16)__msa_hsub_u_h((v16u8)vec2, (v16u8)vec2);
    vec3 = (v8i16)__msa_hsub_u_h((v16u8)vec3, (v16u8)vec3);
    vec4 = (v8i16)__msa_hsub_u_h((v16u8)vec4, (v16u8)vec4);
    vec5 = (v8i16)__msa_hsub_u_h((v16u8)vec5, (v16u8)vec5);
    vec0 += vec2;
    vec1 += vec3;
    vec4 += vec2;
    vec5 += vec3;
    vec0 += vec4;
    vec1 += vec5;
    vec0 = __msa_add_a_h(zero, vec0);
    vec1 = __msa_add_a_h(zero, vec1);
    vec0 = __msa_maxi_s_h(vec0, 0);
    vec1 = __msa_maxi_s_h(vec1, 0);
    vec0 = __msa_min_s_h(max, vec0);
    vec1 = __msa_min_s_h(max, vec1);
    dst0 = (v16u8)__msa_pckev_b((v16i8)vec1, (v16i8)vec0);
    ST_UB(dst0, dst_sobelx);
    src_y0 += 16;
    src_y1 += 16;
    src_y2 += 16;
    dst_sobelx += 16;
  }
}

void SobelYRow_MSA(const uint8_t* src_y0,
                   const uint8_t* src_y1,
                   uint8_t* dst_sobely,
                   int32_t width) {
  int x;
  v16u8 src0, src1, dst0;
  v8i16 vec0, vec1, vec2, vec3, vec4, vec5, vec6;
  v8i16 zero = {0};
  v8i16 max = __msa_ldi_h(255);

  for (x = 0; x < width; x += 16) {
    src0 = (v16u8)__msa_ld_b((const v16i8*)src_y0, 0);
    src1 = (v16u8)__msa_ld_b((const v16i8*)src_y1, 0);
    vec0 = (v8i16)__msa_ilvr_b((v16i8)zero, (v16i8)src0);
    vec1 = (v8i16)__msa_ilvl_b((v16i8)zero, (v16i8)src0);
    vec2 = (v8i16)__msa_ilvr_b((v16i8)zero, (v16i8)src1);
    vec3 = (v8i16)__msa_ilvl_b((v16i8)zero, (v16i8)src1);
    vec0 -= vec2;
    vec1 -= vec3;
    vec6[0] = src_y0[16] - src_y1[16];
    vec6[1] = src_y0[17] - src_y1[17];
    vec2 = (v8i16)__msa_sldi_b((v16i8)vec1, (v16i8)vec0, 2);
    vec3 = (v8i16)__msa_sldi_b((v16i8)vec6, (v16i8)vec1, 2);
    vec4 = (v8i16)__msa_sldi_b((v16i8)vec1, (v16i8)vec0, 4);
    vec5 = (v8i16)__msa_sldi_b((v16i8)vec6, (v16i8)vec1, 4);
    vec0 += vec2;
    vec1 += vec3;
    vec4 += vec2;
    vec5 += vec3;
    vec0 += vec4;
    vec1 += vec5;
    vec0 = __msa_add_a_h(zero, vec0);
    vec1 = __msa_add_a_h(zero, vec1);
    vec0 = __msa_maxi_s_h(vec0, 0);
    vec1 = __msa_maxi_s_h(vec1, 0);
    vec0 = __msa_min_s_h(max, vec0);
    vec1 = __msa_min_s_h(max, vec1);
    dst0 = (v16u8)__msa_pckev_b((v16i8)vec1, (v16i8)vec0);
    ST_UB(dst0, dst_sobely);
    src_y0 += 16;
    src_y1 += 16;
    dst_sobely += 16;
  }
}

void HalfFloatRow_MSA(const uint16_t* src,
                      uint16_t* dst,
                      float scale,
                      int width) {
  int i;
  v8u16 src0, src1, src2, src3, dst0, dst1, dst2, dst3;
  v4u32 vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7;
  v4f32 fvec0, fvec1, fvec2, fvec3, fvec4, fvec5, fvec6, fvec7;
  v4f32 mult_vec;
  v8i16 zero = {0};
  mult_vec[0] = 1.9259299444e-34f * scale;
  mult_vec = (v4f32)__msa_splati_w((v4i32)mult_vec, 0);

  for (i = 0; i < width; i += 32) {
    src0 = (v8u16)__msa_ld_h((v8i16*)src, 0);
    src1 = (v8u16)__msa_ld_h((v8i16*)src, 16);
    src2 = (v8u16)__msa_ld_h((v8i16*)src, 32);
    src3 = (v8u16)__msa_ld_h((v8i16*)src, 48);
    vec0 = (v4u32)__msa_ilvr_h(zero, (v8i16)src0);
    vec1 = (v4u32)__msa_ilvl_h(zero, (v8i16)src0);
    vec2 = (v4u32)__msa_ilvr_h(zero, (v8i16)src1);
    vec3 = (v4u32)__msa_ilvl_h(zero, (v8i16)src1);
    vec4 = (v4u32)__msa_ilvr_h(zero, (v8i16)src2);
    vec5 = (v4u32)__msa_ilvl_h(zero, (v8i16)src2);
    vec6 = (v4u32)__msa_ilvr_h(zero, (v8i16)src3);
    vec7 = (v4u32)__msa_ilvl_h(zero, (v8i16)src3);
    fvec0 = __msa_ffint_u_w(vec0);
    fvec1 = __msa_ffint_u_w(vec1);
    fvec2 = __msa_ffint_u_w(vec2);
    fvec3 = __msa_ffint_u_w(vec3);
    fvec4 = __msa_ffint_u_w(vec4);
    fvec5 = __msa_ffint_u_w(vec5);
    fvec6 = __msa_ffint_u_w(vec6);
    fvec7 = __msa_ffint_u_w(vec7);
    fvec0 *= mult_vec;
    fvec1 *= mult_vec;
    fvec2 *= mult_vec;
    fvec3 *= mult_vec;
    fvec4 *= mult_vec;
    fvec5 *= mult_vec;
    fvec6 *= mult_vec;
    fvec7 *= mult_vec;
    vec0 = ((v4u32)fvec0) >> 13;
    vec1 = ((v4u32)fvec1) >> 13;
    vec2 = ((v4u32)fvec2) >> 13;
    vec3 = ((v4u32)fvec3) >> 13;
    vec4 = ((v4u32)fvec4) >> 13;
    vec5 = ((v4u32)fvec5) >> 13;
    vec6 = ((v4u32)fvec6) >> 13;
    vec7 = ((v4u32)fvec7) >> 13;
    dst0 = (v8u16)__msa_pckev_h((v8i16)vec1, (v8i16)vec0);
    dst1 = (v8u16)__msa_pckev_h((v8i16)vec3, (v8i16)vec2);
    dst2 = (v8u16)__msa_pckev_h((v8i16)vec5, (v8i16)vec4);
    dst3 = (v8u16)__msa_pckev_h((v8i16)vec7, (v8i16)vec6);
    ST_UH2(dst0, dst1, dst, 8);
    ST_UH2(dst2, dst3, dst + 16, 8);
    src += 32;
    dst += 32;
  }
}

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

#endif  // !defined(LIBYUV_DISABLE_MSA) && defined(__mips_msa)
