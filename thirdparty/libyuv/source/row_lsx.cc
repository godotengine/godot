/*
 *  Copyright 2022 The LibYuv Project Authors. All rights reserved.
 *
 *  Copyright (c) 2022 Loongson Technology Corporation Limited
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "libyuv/row.h"

#if !defined(LIBYUV_DISABLE_LSX) && defined(__loongarch_sx)
#include "libyuv/loongson_intrinsics.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// Fill YUV -> RGB conversion constants into vectors
#define YUVTORGB_SETUP(yuvconst, vr, ub, vg, ug, yg, yb) \
  {                                                      \
    ub = __lsx_vreplgr2vr_h(yuvconst->kUVToB[0]);        \
    vr = __lsx_vreplgr2vr_h(yuvconst->kUVToR[1]);        \
    ug = __lsx_vreplgr2vr_h(yuvconst->kUVToG[0]);        \
    vg = __lsx_vreplgr2vr_h(yuvconst->kUVToG[1]);        \
    yg = __lsx_vreplgr2vr_h(yuvconst->kYToRgb[0]);       \
    yb = __lsx_vreplgr2vr_w(yuvconst->kYBiasToRgb[0]);   \
  }

// Load 32 YUV422 pixel data
#define READYUV422_D(psrc_y, psrc_u, psrc_v, out_y, uv_l, uv_h) \
  {                                                             \
    __m128i temp0, temp1;                                       \
                                                                \
    DUP2_ARG2(__lsx_vld, psrc_y, 0, psrc_u, 0, out_y, temp0);   \
    temp1 = __lsx_vld(psrc_v, 0);                               \
    temp0 = __lsx_vsub_b(temp0, const_80);                      \
    temp1 = __lsx_vsub_b(temp1, const_80);                      \
    temp0 = __lsx_vsllwil_h_b(temp0, 0);                        \
    temp1 = __lsx_vsllwil_h_b(temp1, 0);                        \
    uv_l = __lsx_vilvl_h(temp0, temp1);                         \
    uv_h = __lsx_vilvh_h(temp0, temp1);                         \
  }

// Load 16 YUV422 pixel data
#define READYUV422(psrc_y, psrc_u, psrc_v, out_y, uv) \
  {                                                   \
    __m128i temp0, temp1;                             \
                                                      \
    out_y = __lsx_vld(psrc_y, 0);                     \
    temp0 = __lsx_vldrepl_d(psrc_u, 0);               \
    temp1 = __lsx_vldrepl_d(psrc_v, 0);               \
    uv = __lsx_vilvl_b(temp0, temp1);                 \
    uv = __lsx_vsub_b(uv, const_80);                  \
    uv = __lsx_vsllwil_h_b(uv, 0);                    \
  }

// Convert 16 pixels of YUV420 to RGB.
#define YUVTORGB_D(in_y, in_uvl, in_uvh, ubvr, ugvg, yg, yb, b_l, b_h, g_l, \
                   g_h, r_l, r_h)                                           \
  {                                                                         \
    __m128i u_l, u_h, v_l, v_h;                                             \
    __m128i yl_ev, yl_od, yh_ev, yh_od;                                     \
    __m128i temp0, temp1, temp2, temp3;                                     \
                                                                            \
    temp0 = __lsx_vilvl_b(in_y, in_y);                                      \
    temp1 = __lsx_vilvh_b(in_y, in_y);                                      \
    yl_ev = __lsx_vmulwev_w_hu_h(temp0, yg);                                \
    yl_od = __lsx_vmulwod_w_hu_h(temp0, yg);                                \
    yh_ev = __lsx_vmulwev_w_hu_h(temp1, yg);                                \
    yh_od = __lsx_vmulwod_w_hu_h(temp1, yg);                                \
    DUP4_ARG2(__lsx_vsrai_w, yl_ev, 16, yl_od, 16, yh_ev, 16, yh_od, 16,    \
              yl_ev, yl_od, yh_ev, yh_od);                                  \
    yl_ev = __lsx_vadd_w(yl_ev, yb);                                        \
    yl_od = __lsx_vadd_w(yl_od, yb);                                        \
    yh_ev = __lsx_vadd_w(yh_ev, yb);                                        \
    yh_od = __lsx_vadd_w(yh_od, yb);                                        \
    v_l = __lsx_vmulwev_w_h(in_uvl, ubvr);                                  \
    u_l = __lsx_vmulwod_w_h(in_uvl, ubvr);                                  \
    v_h = __lsx_vmulwev_w_h(in_uvh, ubvr);                                  \
    u_h = __lsx_vmulwod_w_h(in_uvh, ubvr);                                  \
    temp0 = __lsx_vadd_w(yl_ev, u_l);                                       \
    temp1 = __lsx_vadd_w(yl_od, u_l);                                       \
    temp2 = __lsx_vadd_w(yh_ev, u_h);                                       \
    temp3 = __lsx_vadd_w(yh_od, u_h);                                       \
    DUP4_ARG2(__lsx_vsrai_w, temp0, 6, temp1, 6, temp2, 6, temp3, 6, temp0, \
              temp1, temp2, temp3);                                         \
    DUP4_ARG1(__lsx_vclip255_w, temp0, temp1, temp2, temp3, temp0, temp1,   \
              temp2, temp3);                                                \
    b_l = __lsx_vpackev_h(temp1, temp0);                                    \
    b_h = __lsx_vpackev_h(temp3, temp2);                                    \
    temp0 = __lsx_vadd_w(yl_ev, v_l);                                       \
    temp1 = __lsx_vadd_w(yl_od, v_l);                                       \
    temp2 = __lsx_vadd_w(yh_ev, v_h);                                       \
    temp3 = __lsx_vadd_w(yh_od, v_h);                                       \
    DUP4_ARG2(__lsx_vsrai_w, temp0, 6, temp1, 6, temp2, 6, temp3, 6, temp0, \
              temp1, temp2, temp3);                                         \
    DUP4_ARG1(__lsx_vclip255_w, temp0, temp1, temp2, temp3, temp0, temp1,   \
              temp2, temp3);                                                \
    r_l = __lsx_vpackev_h(temp1, temp0);                                    \
    r_h = __lsx_vpackev_h(temp3, temp2);                                    \
    DUP2_ARG2(__lsx_vdp2_w_h, in_uvl, ugvg, in_uvh, ugvg, u_l, u_h);        \
    temp0 = __lsx_vsub_w(yl_ev, u_l);                                       \
    temp1 = __lsx_vsub_w(yl_od, u_l);                                       \
    temp2 = __lsx_vsub_w(yh_ev, u_h);                                       \
    temp3 = __lsx_vsub_w(yh_od, u_h);                                       \
    DUP4_ARG2(__lsx_vsrai_w, temp0, 6, temp1, 6, temp2, 6, temp3, 6, temp0, \
              temp1, temp2, temp3);                                         \
    DUP4_ARG1(__lsx_vclip255_w, temp0, temp1, temp2, temp3, temp0, temp1,   \
              temp2, temp3);                                                \
    g_l = __lsx_vpackev_h(temp1, temp0);                                    \
    g_h = __lsx_vpackev_h(temp3, temp2);                                    \
  }

// Convert 8 pixels of YUV420 to RGB.
#define YUVTORGB(in_y, in_vu, vrub, vgug, yg, yb, out_b, out_g, out_r) \
  {                                                                    \
    __m128i y_ev, y_od, u_l, v_l;                                      \
    __m128i tmp0, tmp1, tmp2, tmp3;                                    \
                                                                       \
    tmp0 = __lsx_vilvl_b(in_y, in_y);                                  \
    y_ev = __lsx_vmulwev_w_hu_h(tmp0, yg);                             \
    y_od = __lsx_vmulwod_w_hu_h(tmp0, yg);                             \
    y_ev = __lsx_vsrai_w(y_ev, 16);                                    \
    y_od = __lsx_vsrai_w(y_od, 16);                                    \
    y_ev = __lsx_vadd_w(y_ev, yb);                                     \
    y_od = __lsx_vadd_w(y_od, yb);                                     \
    in_vu = __lsx_vilvl_b(zero, in_vu);                                \
    in_vu = __lsx_vsub_h(in_vu, const_80);                             \
    u_l = __lsx_vmulwev_w_h(in_vu, vrub);                              \
    v_l = __lsx_vmulwod_w_h(in_vu, vrub);                              \
    tmp0 = __lsx_vadd_w(y_ev, u_l);                                    \
    tmp1 = __lsx_vadd_w(y_od, u_l);                                    \
    tmp2 = __lsx_vadd_w(y_ev, v_l);                                    \
    tmp3 = __lsx_vadd_w(y_od, v_l);                                    \
    tmp0 = __lsx_vsrai_w(tmp0, 6);                                     \
    tmp1 = __lsx_vsrai_w(tmp1, 6);                                     \
    tmp2 = __lsx_vsrai_w(tmp2, 6);                                     \
    tmp3 = __lsx_vsrai_w(tmp3, 6);                                     \
    tmp0 = __lsx_vclip255_w(tmp0);                                     \
    tmp1 = __lsx_vclip255_w(tmp1);                                     \
    tmp2 = __lsx_vclip255_w(tmp2);                                     \
    tmp3 = __lsx_vclip255_w(tmp3);                                     \
    out_b = __lsx_vpackev_h(tmp1, tmp0);                               \
    out_r = __lsx_vpackev_h(tmp3, tmp2);                               \
    tmp0 = __lsx_vdp2_w_h(in_vu, vgug);                                \
    tmp1 = __lsx_vsub_w(y_ev, tmp0);                                   \
    tmp2 = __lsx_vsub_w(y_od, tmp0);                                   \
    tmp1 = __lsx_vsrai_w(tmp1, 6);                                     \
    tmp2 = __lsx_vsrai_w(tmp2, 6);                                     \
    tmp1 = __lsx_vclip255_w(tmp1);                                     \
    tmp2 = __lsx_vclip255_w(tmp2);                                     \
    out_g = __lsx_vpackev_h(tmp2, tmp1);                               \
  }

// Convert I444 pixels of YUV420 to RGB.
#define I444TORGB(in_yy, in_u, in_v, ub, vr, ugvg, yg, yb, out_b, out_g, \
                  out_r)                                                 \
  {                                                                      \
    __m128i y_ev, y_od, u_ev, v_ev, u_od, v_od;                          \
    __m128i tmp0, tmp1, tmp2, tmp3;                                      \
                                                                         \
    y_ev = __lsx_vmulwev_w_hu_h(in_yy, yg);                              \
    y_od = __lsx_vmulwod_w_hu_h(in_yy, yg);                              \
    y_ev = __lsx_vsrai_w(y_ev, 16);                                      \
    y_od = __lsx_vsrai_w(y_od, 16);                                      \
    y_ev = __lsx_vadd_w(y_ev, yb);                                       \
    y_od = __lsx_vadd_w(y_od, yb);                                       \
    in_u = __lsx_vsub_h(in_u, const_80);                                 \
    in_v = __lsx_vsub_h(in_v, const_80);                                 \
    u_ev = __lsx_vmulwev_w_h(in_u, ub);                                  \
    u_od = __lsx_vmulwod_w_h(in_u, ub);                                  \
    v_ev = __lsx_vmulwev_w_h(in_v, vr);                                  \
    v_od = __lsx_vmulwod_w_h(in_v, vr);                                  \
    tmp0 = __lsx_vadd_w(y_ev, u_ev);                                     \
    tmp1 = __lsx_vadd_w(y_od, u_od);                                     \
    tmp2 = __lsx_vadd_w(y_ev, v_ev);                                     \
    tmp3 = __lsx_vadd_w(y_od, v_od);                                     \
    tmp0 = __lsx_vsrai_w(tmp0, 6);                                       \
    tmp1 = __lsx_vsrai_w(tmp1, 6);                                       \
    tmp2 = __lsx_vsrai_w(tmp2, 6);                                       \
    tmp3 = __lsx_vsrai_w(tmp3, 6);                                       \
    tmp0 = __lsx_vclip255_w(tmp0);                                       \
    tmp1 = __lsx_vclip255_w(tmp1);                                       \
    tmp2 = __lsx_vclip255_w(tmp2);                                       \
    tmp3 = __lsx_vclip255_w(tmp3);                                       \
    out_b = __lsx_vpackev_h(tmp1, tmp0);                                 \
    out_r = __lsx_vpackev_h(tmp3, tmp2);                                 \
    u_ev = __lsx_vpackev_h(in_u, in_v);                                  \
    u_od = __lsx_vpackod_h(in_u, in_v);                                  \
    v_ev = __lsx_vdp2_w_h(u_ev, ugvg);                                   \
    v_od = __lsx_vdp2_w_h(u_od, ugvg);                                   \
    tmp0 = __lsx_vsub_w(y_ev, v_ev);                                     \
    tmp1 = __lsx_vsub_w(y_od, v_od);                                     \
    tmp0 = __lsx_vsrai_w(tmp0, 6);                                       \
    tmp1 = __lsx_vsrai_w(tmp1, 6);                                       \
    tmp0 = __lsx_vclip255_w(tmp0);                                       \
    tmp1 = __lsx_vclip255_w(tmp1);                                       \
    out_g = __lsx_vpackev_h(tmp1, tmp0);                                 \
  }

// Pack and Store 16 ARGB values.
#define STOREARGB_D(a_l, a_h, r_l, r_h, g_l, g_h, b_l, b_h, pdst_argb) \
  {                                                                    \
    __m128i temp0, temp1, temp2, temp3;                                \
    temp0 = __lsx_vpackev_b(g_l, b_l);                                 \
    temp1 = __lsx_vpackev_b(a_l, r_l);                                 \
    temp2 = __lsx_vpackev_b(g_h, b_h);                                 \
    temp3 = __lsx_vpackev_b(a_h, r_h);                                 \
    r_l = __lsx_vilvl_h(temp1, temp0);                                 \
    r_h = __lsx_vilvh_h(temp1, temp0);                                 \
    g_l = __lsx_vilvl_h(temp3, temp2);                                 \
    g_h = __lsx_vilvh_h(temp3, temp2);                                 \
    __lsx_vst(r_l, pdst_argb, 0);                                      \
    __lsx_vst(r_h, pdst_argb, 16);                                     \
    __lsx_vst(g_l, pdst_argb, 32);                                     \
    __lsx_vst(g_h, pdst_argb, 48);                                     \
    pdst_argb += 64;                                                   \
  }

// Pack and Store 8 ARGB values.
#define STOREARGB(in_a, in_r, in_g, in_b, pdst_argb) \
  {                                                  \
    __m128i temp0, temp1;                            \
    __m128i dst0, dst1;                              \
                                                     \
    temp0 = __lsx_vpackev_b(in_g, in_b);             \
    temp1 = __lsx_vpackev_b(in_a, in_r);             \
    dst0 = __lsx_vilvl_h(temp1, temp0);              \
    dst1 = __lsx_vilvh_h(temp1, temp0);              \
    __lsx_vst(dst0, pdst_argb, 0);                   \
    __lsx_vst(dst1, pdst_argb, 16);                  \
    pdst_argb += 32;                                 \
  }

#define RGBTOUV(_tmpb, _tmpg, _tmpr, _nexb, _nexg, _nexr, _dst0) \
  {                                                              \
    __m128i _tmp0, _tmp1, _tmp2, _tmp3;                          \
    __m128i _reg0, _reg1;                                        \
    _tmp0 = __lsx_vaddwev_h_bu(_tmpb, _nexb);                    \
    _tmp1 = __lsx_vaddwod_h_bu(_tmpb, _nexb);                    \
    _tmp2 = __lsx_vaddwev_h_bu(_tmpg, _nexg);                    \
    _tmp3 = __lsx_vaddwod_h_bu(_tmpg, _nexg);                    \
    _reg0 = __lsx_vaddwev_h_bu(_tmpr, _nexr);                    \
    _reg1 = __lsx_vaddwod_h_bu(_tmpr, _nexr);                    \
    _tmpb = __lsx_vavgr_hu(_tmp0, _tmp1);                        \
    _tmpg = __lsx_vavgr_hu(_tmp2, _tmp3);                        \
    _tmpr = __lsx_vavgr_hu(_reg0, _reg1);                        \
    _reg0 = __lsx_vmadd_h(const_8080, const_112, _tmpb);         \
    _reg1 = __lsx_vmadd_h(const_8080, const_112, _tmpr);         \
    _reg0 = __lsx_vmsub_h(_reg0, const_74, _tmpg);               \
    _reg1 = __lsx_vmsub_h(_reg1, const_94, _tmpg);               \
    _reg0 = __lsx_vmsub_h(_reg0, const_38, _tmpr);               \
    _reg1 = __lsx_vmsub_h(_reg1, const_18, _tmpb);               \
    _dst0 = __lsx_vpickod_b(_reg1, _reg0);                       \
  }

void MirrorRow_LSX(const uint8_t* src, uint8_t* dst, int width) {
  int x;
  int len = width / 32;
  __m128i src0, src1;
  __m128i shuffler = {0x08090A0B0C0D0E0F, 0x0001020304050607};
  src += width - 32;
  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lsx_vld, src, 0, src, 16, src0, src1);
    DUP2_ARG3(__lsx_vshuf_b, src0, src0, shuffler, src1, src1, shuffler, src0,
              src1);
    __lsx_vst(src1, dst, 0);
    __lsx_vst(src0, dst, 16);
    dst += 32;
    src -= 32;
  }
}

void MirrorUVRow_LSX(const uint8_t* src_uv, uint8_t* dst_uv, int width) {
  int x;
  int len = width / 8;
  __m128i src, dst;
  __m128i shuffler = {0x0004000500060007, 0x0000000100020003};

  src_uv += (width - 8) << 1;
  for (x = 0; x < len; x++) {
    src = __lsx_vld(src_uv, 0);
    dst = __lsx_vshuf_h(shuffler, src, src);
    __lsx_vst(dst, dst_uv, 0);
    src_uv -= 16;
    dst_uv += 16;
  }
}

void ARGBMirrorRow_LSX(const uint8_t* src, uint8_t* dst, int width) {
  int x;
  int len = width / 8;
  __m128i src0, src1;
  __m128i shuffler = {0x0B0A09080F0E0D0C, 0x0302010007060504};

  src += (width * 4) - 32;
  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lsx_vld, src, 0, src, 16, src0, src1);
    DUP2_ARG3(__lsx_vshuf_b, src0, src0, shuffler, src1, src1, shuffler, src0,
              src1);
    __lsx_vst(src1, dst, 0);
    __lsx_vst(src0, dst, 16);
    dst += 32;
    src -= 32;
  }
}

void I422ToYUY2Row_LSX(const uint8_t* src_y,
                       const uint8_t* src_u,
                       const uint8_t* src_v,
                       uint8_t* dst_yuy2,
                       int width) {
  int x;
  int len = width / 16;
  __m128i src_u0, src_v0, src_y0, vec_uv0;
  __m128i vec_yuy2_0, vec_yuy2_1;

  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lsx_vld, src_u, 0, src_v, 0, src_u0, src_v0);
    src_y0 = __lsx_vld(src_y, 0);
    vec_uv0 = __lsx_vilvl_b(src_v0, src_u0);
    vec_yuy2_0 = __lsx_vilvl_b(vec_uv0, src_y0);
    vec_yuy2_1 = __lsx_vilvh_b(vec_uv0, src_y0);
    __lsx_vst(vec_yuy2_0, dst_yuy2, 0);
    __lsx_vst(vec_yuy2_1, dst_yuy2, 16);
    src_u += 8;
    src_v += 8;
    src_y += 16;
    dst_yuy2 += 32;
  }
}

void I422ToUYVYRow_LSX(const uint8_t* src_y,
                       const uint8_t* src_u,
                       const uint8_t* src_v,
                       uint8_t* dst_uyvy,
                       int width) {
  int x;
  int len = width / 16;
  __m128i src_u0, src_v0, src_y0, vec_uv0;
  __m128i vec_uyvy0, vec_uyvy1;

  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lsx_vld, src_u, 0, src_v, 0, src_u0, src_v0);
    src_y0 = __lsx_vld(src_y, 0);
    vec_uv0 = __lsx_vilvl_b(src_v0, src_u0);
    vec_uyvy0 = __lsx_vilvl_b(src_y0, vec_uv0);
    vec_uyvy1 = __lsx_vilvh_b(src_y0, vec_uv0);
    __lsx_vst(vec_uyvy0, dst_uyvy, 0);
    __lsx_vst(vec_uyvy1, dst_uyvy, 16);
    src_u += 8;
    src_v += 8;
    src_y += 16;
    dst_uyvy += 32;
  }
}

void I422ToARGBRow_LSX(const uint8_t* src_y,
                       const uint8_t* src_u,
                       const uint8_t* src_v,
                       uint8_t* dst_argb,
                       const struct YuvConstants* yuvconstants,
                       int width) {
  int x;
  int len = width / 16;
  __m128i vec_yb, vec_yg, vec_ub, vec_ug, vec_vr, vec_vg;
  __m128i vec_ubvr, vec_ugvg;
  __m128i alpha = __lsx_vldi(0xFF);
  __m128i const_80 = __lsx_vldi(0x80);

  YUVTORGB_SETUP(yuvconstants, vec_vr, vec_ub, vec_vg, vec_ug, vec_yg, vec_yb);
  vec_ubvr = __lsx_vilvl_h(vec_ub, vec_vr);
  vec_ugvg = __lsx_vilvl_h(vec_ug, vec_vg);

  for (x = 0; x < len; x++) {
    __m128i y, uv_l, uv_h, b_l, b_h, g_l, g_h, r_l, r_h;

    READYUV422_D(src_y, src_u, src_v, y, uv_l, uv_h);
    YUVTORGB_D(y, uv_l, uv_h, vec_ubvr, vec_ugvg, vec_yg, vec_yb, b_l, b_h, g_l,
               g_h, r_l, r_h);
    STOREARGB_D(alpha, alpha, r_l, r_h, g_l, g_h, b_l, b_h, dst_argb);
    src_y += 16;
    src_u += 8;
    src_v += 8;
  }
}

void I422ToRGBARow_LSX(const uint8_t* src_y,
                       const uint8_t* src_u,
                       const uint8_t* src_v,
                       uint8_t* dst_argb,
                       const struct YuvConstants* yuvconstants,
                       int width) {
  int x;
  int len = width / 16;
  __m128i vec_yb, vec_yg, vec_ub, vec_vr, vec_ug, vec_vg;
  __m128i vec_ubvr, vec_ugvg;
  __m128i alpha = __lsx_vldi(0xFF);
  __m128i const_80 = __lsx_vldi(0x80);

  YUVTORGB_SETUP(yuvconstants, vec_vr, vec_ub, vec_vg, vec_ug, vec_yg, vec_yb);
  vec_ubvr = __lsx_vilvl_h(vec_ub, vec_vr);
  vec_ugvg = __lsx_vilvl_h(vec_ug, vec_vg);

  for (x = 0; x < len; x++) {
    __m128i y, uv_l, uv_h, b_l, b_h, g_l, g_h, r_l, r_h;

    READYUV422_D(src_y, src_u, src_v, y, uv_l, uv_h);
    YUVTORGB_D(y, uv_l, uv_h, vec_ubvr, vec_ugvg, vec_yg, vec_yb, b_l, b_h, g_l,
               g_h, r_l, r_h);
    STOREARGB_D(r_l, r_h, g_l, g_h, b_l, b_h, alpha, alpha, dst_argb);
    src_y += 16;
    src_u += 8;
    src_v += 8;
  }
}

void I422AlphaToARGBRow_LSX(const uint8_t* src_y,
                            const uint8_t* src_u,
                            const uint8_t* src_v,
                            const uint8_t* src_a,
                            uint8_t* dst_argb,
                            const struct YuvConstants* yuvconstants,
                            int width) {
  int x;
  int len = width / 16;
  int res = width & 15;
  __m128i vec_yb, vec_yg, vec_ub, vec_vr, vec_ug, vec_vg;
  __m128i vec_ubvr, vec_ugvg;
  __m128i zero = __lsx_vldi(0);
  __m128i const_80 = __lsx_vldi(0x80);

  YUVTORGB_SETUP(yuvconstants, vec_vr, vec_ub, vec_vg, vec_ug, vec_yg, vec_yb);
  vec_ubvr = __lsx_vilvl_h(vec_ub, vec_vr);
  vec_ugvg = __lsx_vilvl_h(vec_ug, vec_vg);

  for (x = 0; x < len; x++) {
    __m128i y, uv_l, uv_h, b_l, b_h, g_l, g_h, r_l, r_h, a_l, a_h;

    y = __lsx_vld(src_a, 0);
    a_l = __lsx_vilvl_b(zero, y);
    a_h = __lsx_vilvh_b(zero, y);
    READYUV422_D(src_y, src_u, src_v, y, uv_l, uv_h);
    YUVTORGB_D(y, uv_l, uv_h, vec_ubvr, vec_ugvg, vec_yg, vec_yb, b_l, b_h, g_l,
               g_h, r_l, r_h);
    STOREARGB_D(a_l, a_h, r_l, r_h, g_l, g_h, b_l, b_h, dst_argb);
    src_y += 16;
    src_u += 8;
    src_v += 8;
    src_a += 16;
  }
  if (res) {
    __m128i y, uv, r, g, b, a;
    a = __lsx_vld(src_a, 0);
    a = __lsx_vsllwil_hu_bu(a, 0);
    READYUV422(src_y, src_u, src_v, y, uv);
    YUVTORGB(y, uv, vec_ubvr, vec_ugvg, vec_yg, vec_yb, b, g, r);
    STOREARGB(a, r, g, b, dst_argb);
  }
}

void I422ToRGB24Row_LSX(const uint8_t* src_y,
                        const uint8_t* src_u,
                        const uint8_t* src_v,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int32_t width) {
  int x;
  int len = width / 16;
  __m128i vec_yb, vec_yg, vec_ub, vec_vr, vec_ug, vec_vg;
  __m128i vec_ubvr, vec_ugvg;
  __m128i const_80 = __lsx_vldi(0x80);
  __m128i shuffler0 = {0x0504120302100100, 0x0A18090816070614};
  __m128i shuffler1 = {0x1E0F0E1C0D0C1A0B, 0x1E0F0E1C0D0C1A0B};

  YUVTORGB_SETUP(yuvconstants, vec_vr, vec_ub, vec_vg, vec_ug, vec_yg, vec_yb);
  vec_ubvr = __lsx_vilvl_h(vec_ub, vec_vr);
  vec_ugvg = __lsx_vilvl_h(vec_ug, vec_vg);

  for (x = 0; x < len; x++) {
    __m128i y, uv_l, uv_h, b_l, b_h, g_l, g_h, r_l, r_h;
    __m128i temp0, temp1, temp2, temp3;

    READYUV422_D(src_y, src_u, src_v, y, uv_l, uv_h);
    YUVTORGB_D(y, uv_l, uv_h, vec_ubvr, vec_ugvg, vec_yg, vec_yb, b_l, b_h, g_l,
               g_h, r_l, r_h);
    temp0 = __lsx_vpackev_b(g_l, b_l);
    temp1 = __lsx_vpackev_b(g_h, b_h);
    DUP4_ARG3(__lsx_vshuf_b, r_l, temp0, shuffler1, r_h, temp1, shuffler1, r_l,
              temp0, shuffler0, r_h, temp1, shuffler0, temp2, temp3, temp0,
              temp1);

    b_l = __lsx_vilvl_d(temp1, temp2);
    b_h = __lsx_vilvh_d(temp3, temp1);
    __lsx_vst(temp0, dst_argb, 0);
    __lsx_vst(b_l, dst_argb, 16);
    __lsx_vst(b_h, dst_argb, 32);
    dst_argb += 48;
    src_y += 16;
    src_u += 8;
    src_v += 8;
  }
}

// TODO(fbarchard): Consider AND instead of shift to isolate 5 upper bits of R.
void I422ToRGB565Row_LSX(const uint8_t* src_y,
                         const uint8_t* src_u,
                         const uint8_t* src_v,
                         uint8_t* dst_rgb565,
                         const struct YuvConstants* yuvconstants,
                         int width) {
  int x;
  int len = width / 16;
  __m128i vec_yb, vec_yg, vec_ub, vec_vr, vec_ug, vec_vg;
  __m128i vec_ubvr, vec_ugvg;
  __m128i const_80 = __lsx_vldi(0x80);

  YUVTORGB_SETUP(yuvconstants, vec_vr, vec_ub, vec_vg, vec_ug, vec_yg, vec_yb);
  vec_ubvr = __lsx_vilvl_h(vec_ub, vec_vr);
  vec_ugvg = __lsx_vilvl_h(vec_ug, vec_vg);

  for (x = 0; x < len; x++) {
    __m128i y, uv_l, uv_h, b_l, b_h, g_l, g_h, r_l, r_h;

    READYUV422_D(src_y, src_u, src_v, y, uv_l, uv_h);
    YUVTORGB_D(y, uv_l, uv_h, vec_ubvr, vec_ugvg, vec_yg, vec_yb, b_l, b_h, g_l,
               g_h, r_l, r_h);
    b_l = __lsx_vsrli_h(b_l, 3);
    b_h = __lsx_vsrli_h(b_h, 3);
    g_l = __lsx_vsrli_h(g_l, 2);
    g_h = __lsx_vsrli_h(g_h, 2);
    r_l = __lsx_vsrli_h(r_l, 3);
    r_h = __lsx_vsrli_h(r_h, 3);
    r_l = __lsx_vslli_h(r_l, 11);
    r_h = __lsx_vslli_h(r_h, 11);
    g_l = __lsx_vslli_h(g_l, 5);
    g_h = __lsx_vslli_h(g_h, 5);
    r_l = __lsx_vor_v(r_l, g_l);
    r_l = __lsx_vor_v(r_l, b_l);
    r_h = __lsx_vor_v(r_h, g_h);
    r_h = __lsx_vor_v(r_h, b_h);
    __lsx_vst(r_l, dst_rgb565, 0);
    __lsx_vst(r_h, dst_rgb565, 16);
    dst_rgb565 += 32;
    src_y += 16;
    src_u += 8;
    src_v += 8;
  }
}

// TODO(fbarchard): Consider AND instead of shift to isolate 4 upper bits of G.
void I422ToARGB4444Row_LSX(const uint8_t* src_y,
                           const uint8_t* src_u,
                           const uint8_t* src_v,
                           uint8_t* dst_argb4444,
                           const struct YuvConstants* yuvconstants,
                           int width) {
  int x;
  int len = width / 16;
  __m128i vec_yb, vec_yg, vec_ub, vec_vr, vec_ug, vec_vg;
  __m128i vec_ubvr, vec_ugvg;
  __m128i const_80 = __lsx_vldi(0x80);
  __m128i alpha = (__m128i)v2u64{0xF000F000F000F000, 0xF000F000F000F000};
  __m128i mask = {0x00F000F000F000F0, 0x00F000F000F000F0};

  YUVTORGB_SETUP(yuvconstants, vec_vr, vec_ub, vec_vg, vec_ug, vec_yg, vec_yb);
  vec_ubvr = __lsx_vilvl_h(vec_ub, vec_vr);
  vec_ugvg = __lsx_vilvl_h(vec_ug, vec_vg);

  for (x = 0; x < len; x++) {
    __m128i y, uv_l, uv_h, b_l, b_h, g_l, g_h, r_l, r_h;

    READYUV422_D(src_y, src_u, src_v, y, uv_l, uv_h);
    YUVTORGB_D(y, uv_l, uv_h, vec_ubvr, vec_ugvg, vec_yg, vec_yb, b_l, b_h, g_l,
               g_h, r_l, r_h);
    b_l = __lsx_vsrli_h(b_l, 4);
    b_h = __lsx_vsrli_h(b_h, 4);
    r_l = __lsx_vsrli_h(r_l, 4);
    r_h = __lsx_vsrli_h(r_h, 4);
    g_l = __lsx_vand_v(g_l, mask);
    g_h = __lsx_vand_v(g_h, mask);
    r_l = __lsx_vslli_h(r_l, 8);
    r_h = __lsx_vslli_h(r_h, 8);
    r_l = __lsx_vor_v(r_l, alpha);
    r_h = __lsx_vor_v(r_h, alpha);
    r_l = __lsx_vor_v(r_l, g_l);
    r_h = __lsx_vor_v(r_h, g_h);
    r_l = __lsx_vor_v(r_l, b_l);
    r_h = __lsx_vor_v(r_h, b_h);
    __lsx_vst(r_l, dst_argb4444, 0);
    __lsx_vst(r_h, dst_argb4444, 16);
    dst_argb4444 += 32;
    src_y += 16;
    src_u += 8;
    src_v += 8;
  }
}

void I422ToARGB1555Row_LSX(const uint8_t* src_y,
                           const uint8_t* src_u,
                           const uint8_t* src_v,
                           uint8_t* dst_argb1555,
                           const struct YuvConstants* yuvconstants,
                           int width) {
  int x;
  int len = width / 16;
  __m128i vec_yb, vec_yg, vec_ub, vec_vr, vec_ug, vec_vg;
  __m128i vec_ubvr, vec_ugvg;
  __m128i const_80 = __lsx_vldi(0x80);
  __m128i alpha = (__m128i)v2u64{0x8000800080008000, 0x8000800080008000};

  YUVTORGB_SETUP(yuvconstants, vec_vr, vec_ub, vec_vg, vec_ug, vec_yg, vec_yb);
  vec_ubvr = __lsx_vilvl_h(vec_ub, vec_vr);
  vec_ugvg = __lsx_vilvl_h(vec_ug, vec_vg);

  for (x = 0; x < len; x++) {
    __m128i y, uv_l, uv_h, b_l, b_h, g_l, g_h, r_l, r_h;

    READYUV422_D(src_y, src_u, src_v, y, uv_l, uv_h);
    YUVTORGB_D(y, uv_l, uv_h, vec_ubvr, vec_ugvg, vec_yg, vec_yb, b_l, b_h, g_l,
               g_h, r_l, r_h);
    b_l = __lsx_vsrli_h(b_l, 3);
    b_h = __lsx_vsrli_h(b_h, 3);
    g_l = __lsx_vsrli_h(g_l, 3);

    g_h = __lsx_vsrli_h(g_h, 3);
    g_l = __lsx_vslli_h(g_l, 5);
    g_h = __lsx_vslli_h(g_h, 5);
    r_l = __lsx_vsrli_h(r_l, 3);
    r_h = __lsx_vsrli_h(r_h, 3);
    r_l = __lsx_vslli_h(r_l, 10);
    r_h = __lsx_vslli_h(r_h, 10);
    r_l = __lsx_vor_v(r_l, alpha);
    r_h = __lsx_vor_v(r_h, alpha);
    r_l = __lsx_vor_v(r_l, g_l);
    r_h = __lsx_vor_v(r_h, g_h);
    r_l = __lsx_vor_v(r_l, b_l);
    r_h = __lsx_vor_v(r_h, b_h);
    __lsx_vst(r_l, dst_argb1555, 0);
    __lsx_vst(r_h, dst_argb1555, 16);
    dst_argb1555 += 32;
    src_y += 16;
    src_u += 8;
    src_v += 8;
  }
}

void YUY2ToYRow_LSX(const uint8_t* src_yuy2, uint8_t* dst_y, int width) {
  int x;
  int len = width / 16;
  __m128i src0, src1, dst0;

  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lsx_vld, src_yuy2, 0, src_yuy2, 16, src0, src1);
    dst0 = __lsx_vpickev_b(src1, src0);
    __lsx_vst(dst0, dst_y, 0);
    src_yuy2 += 32;
    dst_y += 16;
  }
}

void YUY2ToUVRow_LSX(const uint8_t* src_yuy2,
                     int src_stride_yuy2,
                     uint8_t* dst_u,
                     uint8_t* dst_v,
                     int width) {
  const uint8_t* src_yuy2_next = src_yuy2 + src_stride_yuy2;
  int x;
  int len = width / 16;
  __m128i src0, src1, src2, src3;
  __m128i tmp0, dst0, dst1;

  for (x = 0; x < len; x++) {
    DUP4_ARG2(__lsx_vld, src_yuy2, 0, src_yuy2, 16, src_yuy2_next, 0,
              src_yuy2_next, 16, src0, src1, src2, src3);
    src0 = __lsx_vpickod_b(src1, src0);
    src1 = __lsx_vpickod_b(src3, src2);
    tmp0 = __lsx_vavgr_bu(src1, src0);
    dst0 = __lsx_vpickev_b(tmp0, tmp0);
    dst1 = __lsx_vpickod_b(tmp0, tmp0);
    __lsx_vstelm_d(dst0, dst_u, 0, 0);
    __lsx_vstelm_d(dst1, dst_v, 0, 0);
    src_yuy2 += 32;
    src_yuy2_next += 32;
    dst_u += 8;
    dst_v += 8;
  }
}

void YUY2ToUV422Row_LSX(const uint8_t* src_yuy2,
                        uint8_t* dst_u,
                        uint8_t* dst_v,
                        int width) {
  int x;
  int len = width / 16;
  __m128i src0, src1, tmp0, dst0, dst1;

  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lsx_vld, src_yuy2, 0, src_yuy2, 16, src0, src1);
    tmp0 = __lsx_vpickod_b(src1, src0);
    dst0 = __lsx_vpickev_b(tmp0, tmp0);
    dst1 = __lsx_vpickod_b(tmp0, tmp0);
    __lsx_vstelm_d(dst0, dst_u, 0, 0);
    __lsx_vstelm_d(dst1, dst_v, 0, 0);
    src_yuy2 += 32;
    dst_u += 8;
    dst_v += 8;
  }
}

void UYVYToYRow_LSX(const uint8_t* src_uyvy, uint8_t* dst_y, int width) {
  int x;
  int len = width / 16;
  __m128i src0, src1, dst0;

  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lsx_vld, src_uyvy, 0, src_uyvy, 16, src0, src1);
    dst0 = __lsx_vpickod_b(src1, src0);
    __lsx_vst(dst0, dst_y, 0);
    src_uyvy += 32;
    dst_y += 16;
  }
}

void UYVYToUVRow_LSX(const uint8_t* src_uyvy,
                     int src_stride_uyvy,
                     uint8_t* dst_u,
                     uint8_t* dst_v,
                     int width) {
  const uint8_t* src_uyvy_next = src_uyvy + src_stride_uyvy;
  int x;
  int len = width / 16;
  __m128i src0, src1, src2, src3, tmp0, dst0, dst1;

  for (x = 0; x < len; x++) {
    DUP4_ARG2(__lsx_vld, src_uyvy, 0, src_uyvy, 16, src_uyvy_next, 0,
              src_uyvy_next, 16, src0, src1, src2, src3);
    src0 = __lsx_vpickev_b(src1, src0);
    src1 = __lsx_vpickev_b(src3, src2);
    tmp0 = __lsx_vavgr_bu(src1, src0);
    dst0 = __lsx_vpickev_b(tmp0, tmp0);
    dst1 = __lsx_vpickod_b(tmp0, tmp0);
    __lsx_vstelm_d(dst0, dst_u, 0, 0);
    __lsx_vstelm_d(dst1, dst_v, 0, 0);
    src_uyvy += 32;
    src_uyvy_next += 32;
    dst_u += 8;
    dst_v += 8;
  }
}

void UYVYToUV422Row_LSX(const uint8_t* src_uyvy,
                        uint8_t* dst_u,
                        uint8_t* dst_v,
                        int width) {
  int x;
  int len = width / 16;
  __m128i src0, src1, tmp0, dst0, dst1;

  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lsx_vld, src_uyvy, 0, src_uyvy, 16, src0, src1);
    tmp0 = __lsx_vpickev_b(src1, src0);
    dst0 = __lsx_vpickev_b(tmp0, tmp0);
    dst1 = __lsx_vpickod_b(tmp0, tmp0);
    __lsx_vstelm_d(dst0, dst_u, 0, 0);
    __lsx_vstelm_d(dst1, dst_v, 0, 0);
    src_uyvy += 32;
    dst_u += 8;
    dst_v += 8;
  }
}

void ARGBToUVRow_LSX(const uint8_t* src_argb0,
                     int src_stride_argb,
                     uint8_t* dst_u,
                     uint8_t* dst_v,
                     int width) {
  int x;
  int len = width / 16;
  const uint8_t* src_argb1 = src_argb0 + src_stride_argb;

  __m128i src0, src1, src2, src3, src4, src5, src6, src7;
  __m128i vec0, vec1, vec2, vec3;
  __m128i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, dst0, dst1;
  __m128i const_0x70 = {0x0038003800380038, 0x0038003800380038};
  __m128i const_0x4A = {0x0025002500250025, 0x0025002500250025};
  __m128i const_0x26 = {0x0013001300130013, 0x0013001300130013};
  __m128i const_0x5E = {0x002f002f002f002f, 0x002f002f002f002f};
  __m128i const_0x12 = {0x0009000900090009, 0x0009000900090009};
  __m128i const_0x8080 = (__m128i)v2u64{0x8080808080808080, 0x8080808080808080};
  for (x = 0; x < len; x++) {
    DUP4_ARG2(__lsx_vld, src_argb0, 0, src_argb0, 16, src_argb0, 32, src_argb0,
              48, src0, src1, src2, src3);
    DUP4_ARG2(__lsx_vld, src_argb1, 0, src_argb1, 16, src_argb1, 32, src_argb1,
              48, src4, src5, src6, src7);
    vec0 = __lsx_vaddwev_h_bu(src0, src4);
    vec1 = __lsx_vaddwev_h_bu(src1, src5);
    vec2 = __lsx_vaddwev_h_bu(src2, src6);
    vec3 = __lsx_vaddwev_h_bu(src3, src7);
    tmp0 = __lsx_vpickev_h(vec1, vec0);
    tmp1 = __lsx_vpickev_h(vec3, vec2);
    tmp2 = __lsx_vpickod_h(vec1, vec0);
    tmp3 = __lsx_vpickod_h(vec3, vec2);
    vec0 = __lsx_vaddwod_h_bu(src0, src4);
    vec1 = __lsx_vaddwod_h_bu(src1, src5);
    vec2 = __lsx_vaddwod_h_bu(src2, src6);
    vec3 = __lsx_vaddwod_h_bu(src3, src7);
    tmp4 = __lsx_vpickev_h(vec1, vec0);
    tmp5 = __lsx_vpickev_h(vec3, vec2);
    vec0 = __lsx_vpickev_h(tmp1, tmp0);
    vec1 = __lsx_vpickod_h(tmp1, tmp0);
    src0 = __lsx_vavgr_h(vec0, vec1);
    vec0 = __lsx_vpickev_h(tmp3, tmp2);
    vec1 = __lsx_vpickod_h(tmp3, tmp2);
    src1 = __lsx_vavgr_h(vec0, vec1);
    vec0 = __lsx_vpickev_h(tmp5, tmp4);
    vec1 = __lsx_vpickod_h(tmp5, tmp4);
    src2 = __lsx_vavgr_h(vec0, vec1);
    dst0 = __lsx_vmadd_h(const_0x8080, src0, const_0x70);
    dst0 = __lsx_vmsub_h(dst0, src2, const_0x4A);
    dst0 = __lsx_vmsub_h(dst0, src1, const_0x26);
    dst1 = __lsx_vmadd_h(const_0x8080, src1, const_0x70);
    dst1 = __lsx_vmsub_h(dst1, src2, const_0x5E);
    dst1 = __lsx_vmsub_h(dst1, src0, const_0x12);
    dst0 = __lsx_vsrai_h(dst0, 8);
    dst1 = __lsx_vsrai_h(dst1, 8);
    dst0 = __lsx_vpickev_b(dst1, dst0);
    __lsx_vstelm_d(dst0, dst_u, 0, 0);
    __lsx_vstelm_d(dst0, dst_v, 0, 1);
    src_argb0 += 64;
    src_argb1 += 64;
    dst_u += 8;
    dst_v += 8;
  }
}

void ARGBToRGB24Row_LSX(const uint8_t* src_argb, uint8_t* dst_rgb, int width) {
  int x;
  int len = (width / 16) - 1;
  __m128i src0, src1, src2, src3;
  __m128i tmp0, tmp1, tmp2, tmp3;
  __m128i shuf = {0x0908060504020100, 0x000000000E0D0C0A};
  for (x = 0; x < len; x++) {
    DUP4_ARG2(__lsx_vld, src_argb, 0, src_argb, 16, src_argb, 32, src_argb, 48,
              src0, src1, src2, src3);
    tmp0 = __lsx_vshuf_b(src0, src0, shuf);
    tmp1 = __lsx_vshuf_b(src1, src1, shuf);
    tmp2 = __lsx_vshuf_b(src2, src2, shuf);
    tmp3 = __lsx_vshuf_b(src3, src3, shuf);
    __lsx_vst(tmp0, dst_rgb, 0);
    __lsx_vst(tmp1, dst_rgb, 12);
    __lsx_vst(tmp2, dst_rgb, 24);
    __lsx_vst(tmp3, dst_rgb, 36);
    dst_rgb += 48;
    src_argb += 64;
  }
  DUP4_ARG2(__lsx_vld, src_argb, 0, src_argb, 16, src_argb, 32, src_argb, 48,
            src0, src1, src2, src3);
  tmp0 = __lsx_vshuf_b(src0, src0, shuf);
  tmp1 = __lsx_vshuf_b(src1, src1, shuf);
  tmp2 = __lsx_vshuf_b(src2, src2, shuf);
  tmp3 = __lsx_vshuf_b(src3, src3, shuf);
  __lsx_vst(tmp0, dst_rgb, 0);
  __lsx_vst(tmp1, dst_rgb, 12);
  __lsx_vst(tmp2, dst_rgb, 24);
  dst_rgb += 36;
  __lsx_vst(tmp3, dst_rgb, 0);
}

void ARGBToRAWRow_LSX(const uint8_t* src_argb, uint8_t* dst_rgb, int width) {
  int x;
  int len = (width / 16) - 1;
  __m128i src0, src1, src2, src3;
  __m128i tmp0, tmp1, tmp2, tmp3;
  __m128i shuf = {0x090A040506000102, 0x000000000C0D0E08};
  for (x = 0; x < len; x++) {
    DUP4_ARG2(__lsx_vld, src_argb, 0, src_argb, 16, src_argb, 32, src_argb, 48,
              src0, src1, src2, src3);
    tmp0 = __lsx_vshuf_b(src0, src0, shuf);
    tmp1 = __lsx_vshuf_b(src1, src1, shuf);
    tmp2 = __lsx_vshuf_b(src2, src2, shuf);
    tmp3 = __lsx_vshuf_b(src3, src3, shuf);
    __lsx_vst(tmp0, dst_rgb, 0);
    __lsx_vst(tmp1, dst_rgb, 12);
    __lsx_vst(tmp2, dst_rgb, 24);
    __lsx_vst(tmp3, dst_rgb, 36);
    dst_rgb += 48;
    src_argb += 64;
  }
  DUP4_ARG2(__lsx_vld, src_argb, 0, src_argb, 16, src_argb, 32, src_argb, 48,
            src0, src1, src2, src3);
  tmp0 = __lsx_vshuf_b(src0, src0, shuf);
  tmp1 = __lsx_vshuf_b(src1, src1, shuf);
  tmp2 = __lsx_vshuf_b(src2, src2, shuf);
  tmp3 = __lsx_vshuf_b(src3, src3, shuf);
  __lsx_vst(tmp0, dst_rgb, 0);
  __lsx_vst(tmp1, dst_rgb, 12);
  __lsx_vst(tmp2, dst_rgb, 24);
  dst_rgb += 36;
  __lsx_vst(tmp3, dst_rgb, 0);
}

void ARGBToRGB565Row_LSX(const uint8_t* src_argb, uint8_t* dst_rgb, int width) {
  int x;
  int len = width / 8;
  __m128i zero = __lsx_vldi(0);
  __m128i src0, src1, tmp0, tmp1, dst0;
  __m128i shift = {0x0300030003000300, 0x0300030003000300};

  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lsx_vld, src_argb, 0, src_argb, 16, src0, src1);
    tmp0 = __lsx_vpickev_b(src1, src0);
    tmp1 = __lsx_vpickod_b(src1, src0);
    tmp0 = __lsx_vsrli_b(tmp0, 3);
    tmp1 = __lsx_vpackev_b(zero, tmp1);
    tmp1 = __lsx_vsrli_h(tmp1, 2);
    tmp0 = __lsx_vsll_b(tmp0, shift);
    tmp1 = __lsx_vslli_h(tmp1, 5);
    dst0 = __lsx_vor_v(tmp0, tmp1);
    __lsx_vst(dst0, dst_rgb, 0);
    dst_rgb += 16;
    src_argb += 32;
  }
}

void ARGBToARGB1555Row_LSX(const uint8_t* src_argb,
                           uint8_t* dst_rgb,
                           int width) {
  int x;
  int len = width / 8;
  __m128i zero = __lsx_vldi(0);
  __m128i src0, src1, tmp0, tmp1, tmp2, tmp3, dst0;
  __m128i shift1 = {0x0703070307030703, 0x0703070307030703};
  __m128i shift2 = {0x0200020002000200, 0x0200020002000200};

  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lsx_vld, src_argb, 0, src_argb, 16, src0, src1);
    tmp0 = __lsx_vpickev_b(src1, src0);
    tmp1 = __lsx_vpickod_b(src1, src0);
    tmp0 = __lsx_vsrli_b(tmp0, 3);
    tmp1 = __lsx_vsrl_b(tmp1, shift1);
    tmp0 = __lsx_vsll_b(tmp0, shift2);
    tmp2 = __lsx_vpackev_b(zero, tmp1);
    tmp3 = __lsx_vpackod_b(zero, tmp1);
    tmp2 = __lsx_vslli_h(tmp2, 5);
    tmp3 = __lsx_vslli_h(tmp3, 15);
    dst0 = __lsx_vor_v(tmp0, tmp2);
    dst0 = __lsx_vor_v(dst0, tmp3);
    __lsx_vst(dst0, dst_rgb, 0);
    dst_rgb += 16;
    src_argb += 32;
  }
}

void ARGBToARGB4444Row_LSX(const uint8_t* src_argb,
                           uint8_t* dst_rgb,
                           int width) {
  int x;
  int len = width / 8;
  __m128i src0, src1, tmp0, tmp1, dst0;

  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lsx_vld, src_argb, 0, src_argb, 16, src0, src1);
    tmp0 = __lsx_vpickev_b(src1, src0);
    tmp1 = __lsx_vpickod_b(src1, src0);
    tmp1 = __lsx_vandi_b(tmp1, 0xF0);
    tmp0 = __lsx_vsrli_b(tmp0, 4);
    dst0 = __lsx_vor_v(tmp1, tmp0);
    __lsx_vst(dst0, dst_rgb, 0);
    dst_rgb += 16;
    src_argb += 32;
  }
}

void ARGBToUV444Row_LSX(const uint8_t* src_argb,
                        uint8_t* dst_u,
                        uint8_t* dst_v,
                        int32_t width) {
  int x;
  int len = width / 16;
  __m128i src0, src1, src2, src3;
  __m128i tmp0, tmp1, tmp2, tmp3;
  __m128i reg0, reg1, reg2, reg3, dst0, dst1;
  __m128i const_112 = __lsx_vldi(112);
  __m128i const_74 = __lsx_vldi(74);
  __m128i const_38 = __lsx_vldi(38);
  __m128i const_94 = __lsx_vldi(94);
  __m128i const_18 = __lsx_vldi(18);
  __m128i const_0x8080 = (__m128i)v2u64{0x8080808080808080, 0x8080808080808080};
  for (x = 0; x < len; x++) {
    DUP4_ARG2(__lsx_vld, src_argb, 0, src_argb, 16, src_argb, 32, src_argb, 48,
              src0, src1, src2, src3);
    tmp0 = __lsx_vpickev_h(src1, src0);
    tmp1 = __lsx_vpickod_h(src1, src0);
    tmp2 = __lsx_vpickev_h(src3, src2);
    tmp3 = __lsx_vpickod_h(src3, src2);
    reg0 = __lsx_vmaddwev_h_bu(const_0x8080, tmp0, const_112);
    reg1 = __lsx_vmaddwev_h_bu(const_0x8080, tmp2, const_112);
    reg2 = __lsx_vmulwod_h_bu(tmp0, const_74);
    reg3 = __lsx_vmulwod_h_bu(tmp2, const_74);
    reg2 = __lsx_vmaddwev_h_bu(reg2, tmp1, const_38);
    reg3 = __lsx_vmaddwev_h_bu(reg3, tmp3, const_38);
    reg0 = __lsx_vsub_h(reg0, reg2);
    reg1 = __lsx_vsub_h(reg1, reg3);
    reg0 = __lsx_vsrai_h(reg0, 8);
    reg1 = __lsx_vsrai_h(reg1, 8);
    dst0 = __lsx_vpickev_b(reg1, reg0);

    reg0 = __lsx_vmaddwev_h_bu(const_0x8080, tmp1, const_112);
    reg1 = __lsx_vmaddwev_h_bu(const_0x8080, tmp3, const_112);
    reg2 = __lsx_vmulwev_h_bu(tmp0, const_18);
    reg3 = __lsx_vmulwev_h_bu(tmp2, const_18);
    reg2 = __lsx_vmaddwod_h_bu(reg2, tmp0, const_94);
    reg3 = __lsx_vmaddwod_h_bu(reg3, tmp2, const_94);
    reg0 = __lsx_vsub_h(reg0, reg2);
    reg1 = __lsx_vsub_h(reg1, reg3);
    reg0 = __lsx_vsrai_h(reg0, 8);
    reg1 = __lsx_vsrai_h(reg1, 8);
    dst1 = __lsx_vpickev_b(reg1, reg0);

    __lsx_vst(dst0, dst_u, 0);
    __lsx_vst(dst1, dst_v, 0);
    dst_u += 16;
    dst_v += 16;
    src_argb += 64;
  }
}

void ARGBMultiplyRow_LSX(const uint8_t* src_argb0,
                         const uint8_t* src_argb1,
                         uint8_t* dst_argb,
                         int width) {
  int x;
  int len = width / 4;
  __m128i zero = __lsx_vldi(0);
  __m128i src0, src1, dst0, dst1;
  __m128i tmp0, tmp1, tmp2, tmp3;

  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lsx_vld, src_argb0, 0, src_argb1, 0, src0, src1);
    tmp0 = __lsx_vilvl_b(src0, src0);
    tmp1 = __lsx_vilvh_b(src0, src0);
    tmp2 = __lsx_vilvl_b(zero, src1);
    tmp3 = __lsx_vilvh_b(zero, src1);
    dst0 = __lsx_vmuh_hu(tmp0, tmp2);
    dst1 = __lsx_vmuh_hu(tmp1, tmp3);
    dst0 = __lsx_vpickev_b(dst1, dst0);
    __lsx_vst(dst0, dst_argb, 0);
    src_argb0 += 16;
    src_argb1 += 16;
    dst_argb += 16;
  }
}

void ARGBAddRow_LSX(const uint8_t* src_argb0,
                    const uint8_t* src_argb1,
                    uint8_t* dst_argb,
                    int width) {
  int x;
  int len = width / 4;
  __m128i src0, src1, dst0;

  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lsx_vld, src_argb0, 0, src_argb1, 0, src0, src1);
    dst0 = __lsx_vsadd_bu(src0, src1);
    __lsx_vst(dst0, dst_argb, 0);
    src_argb0 += 16;
    src_argb1 += 16;
    dst_argb += 16;
  }
}

void ARGBSubtractRow_LSX(const uint8_t* src_argb0,
                         const uint8_t* src_argb1,
                         uint8_t* dst_argb,
                         int width) {
  int x;
  int len = width / 4;
  __m128i src0, src1, dst0;

  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lsx_vld, src_argb0, 0, src_argb1, 0, src0, src1);
    dst0 = __lsx_vssub_bu(src0, src1);
    __lsx_vst(dst0, dst_argb, 0);
    src_argb0 += 16;
    src_argb1 += 16;
    dst_argb += 16;
  }
}

void ARGBAttenuateRow_LSX(const uint8_t* src_argb,
                          uint8_t* dst_argb,
                          int width) {
  int x;
  int len = width / 8;
  __m128i src0, src1, tmp0, tmp1;
  __m128i reg0, reg1, reg2, reg3, reg4, reg5;
  __m128i b, g, r, a, dst0, dst1;
  __m128i control = {0x0005000100040000, 0x0007000300060002};
  __m128i zero = __lsx_vldi(0);
  __m128i const_add = __lsx_vldi(0x8ff);

  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lsx_vld, src_argb, 0, src_argb, 16, src0, src1);
    tmp0 = __lsx_vpickev_b(src1, src0);
    tmp1 = __lsx_vpickod_b(src1, src0);
    b = __lsx_vpackev_b(zero, tmp0);
    r = __lsx_vpackod_b(zero, tmp0);
    g = __lsx_vpackev_b(zero, tmp1);
    a = __lsx_vpackod_b(zero, tmp1);
    reg0 = __lsx_vmaddwev_w_hu(const_add, b, a);
    reg1 = __lsx_vmaddwod_w_hu(const_add, b, a);
    reg2 = __lsx_vmaddwev_w_hu(const_add, r, a);
    reg3 = __lsx_vmaddwod_w_hu(const_add, r, a);
    reg4 = __lsx_vmaddwev_w_hu(const_add, g, a);
    reg5 = __lsx_vmaddwod_w_hu(const_add, g, a);
    reg0 = __lsx_vssrani_h_w(reg1, reg0, 8);
    reg2 = __lsx_vssrani_h_w(reg3, reg2, 8);
    reg4 = __lsx_vssrani_h_w(reg5, reg4, 8);
    reg0 = __lsx_vshuf_h(control, reg0, reg0);
    reg2 = __lsx_vshuf_h(control, reg2, reg2);
    reg4 = __lsx_vshuf_h(control, reg4, reg4);
    tmp0 = __lsx_vpackev_b(reg4, reg0);
    tmp1 = __lsx_vpackev_b(a, reg2);
    dst0 = __lsx_vilvl_h(tmp1, tmp0);
    dst1 = __lsx_vilvh_h(tmp1, tmp0);
    __lsx_vst(dst0, dst_argb, 0);
    __lsx_vst(dst1, dst_argb, 16);
    dst_argb += 32;
    src_argb += 32;
  }
}

void ARGBToRGB565DitherRow_LSX(const uint8_t* src_argb,
                               uint8_t* dst_rgb,
                               uint32_t dither4,
                               int width) {
  int x;
  int len = width / 8;
  __m128i src0, src1, tmp0, tmp1, dst0;
  __m128i b, g, r;
  __m128i zero = __lsx_vldi(0);
  __m128i vec_dither = __lsx_vldrepl_w(&dither4, 0);

  vec_dither = __lsx_vilvl_b(zero, vec_dither);
  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lsx_vld, src_argb, 0, src_argb, 16, src0, src1);
    tmp0 = __lsx_vpickev_b(src1, src0);
    tmp1 = __lsx_vpickod_b(src1, src0);
    b = __lsx_vpackev_b(zero, tmp0);
    r = __lsx_vpackod_b(zero, tmp0);
    g = __lsx_vpackev_b(zero, tmp1);
    b = __lsx_vadd_h(b, vec_dither);
    g = __lsx_vadd_h(g, vec_dither);
    r = __lsx_vadd_h(r, vec_dither);
    DUP2_ARG1(__lsx_vclip255_h, b, g, b, g);
    r = __lsx_vclip255_h(r);
    b = __lsx_vsrai_h(b, 3);
    g = __lsx_vsrai_h(g, 2);
    r = __lsx_vsrai_h(r, 3);
    g = __lsx_vslli_h(g, 5);
    r = __lsx_vslli_h(r, 11);
    dst0 = __lsx_vor_v(b, g);
    dst0 = __lsx_vor_v(dst0, r);
    __lsx_vst(dst0, dst_rgb, 0);
    src_argb += 32;
    dst_rgb += 16;
  }
}

void ARGBShuffleRow_LSX(const uint8_t* src_argb,
                        uint8_t* dst_argb,
                        const uint8_t* shuffler,
                        int width) {
  int x;
  int len = width / 8;
  __m128i src0, src1, dst0, dst1;
  __m128i shuf = {0x0404040400000000, 0x0C0C0C0C08080808};
  __m128i temp = __lsx_vldrepl_w(shuffler, 0);

  shuf = __lsx_vadd_b(shuf, temp);
  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lsx_vld, src_argb, 0, src_argb, 16, src0, src1);
    dst0 = __lsx_vshuf_b(src0, src0, shuf);
    dst1 = __lsx_vshuf_b(src1, src1, shuf);
    __lsx_vst(dst0, dst_argb, 0);
    __lsx_vst(dst1, dst_argb, 16);
    src_argb += 32;
    dst_argb += 32;
  }
}

void ARGBShadeRow_LSX(const uint8_t* src_argb,
                      uint8_t* dst_argb,
                      int width,
                      uint32_t value) {
  int x;
  int len = width / 4;
  __m128i src0, dst0, tmp0, tmp1;
  __m128i vec_value = __lsx_vreplgr2vr_w(value);

  vec_value = __lsx_vilvl_b(vec_value, vec_value);
  for (x = 0; x < len; x++) {
    src0 = __lsx_vld(src_argb, 0);
    tmp0 = __lsx_vilvl_b(src0, src0);
    tmp1 = __lsx_vilvh_b(src0, src0);
    tmp0 = __lsx_vmuh_hu(tmp0, vec_value);
    tmp1 = __lsx_vmuh_hu(tmp1, vec_value);
    dst0 = __lsx_vpickod_b(tmp1, tmp0);
    __lsx_vst(dst0, dst_argb, 0);
    src_argb += 16;
    dst_argb += 16;
  }
}

void ARGBGrayRow_LSX(const uint8_t* src_argb, uint8_t* dst_argb, int width) {
  int x;
  int len = width / 8;
  __m128i src0, src1, tmp0, tmp1;
  __m128i reg0, reg1, reg2, dst0, dst1;
  __m128i const_128 = __lsx_vldi(0x480);
  __m128i const_150 = __lsx_vldi(0x96);
  __m128i const_br = {0x4D1D4D1D4D1D4D1D, 0x4D1D4D1D4D1D4D1D};

  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lsx_vld, src_argb, 0, src_argb, 16, src0, src1);
    tmp0 = __lsx_vpickev_b(src1, src0);
    tmp1 = __lsx_vpickod_b(src1, src0);
    reg0 = __lsx_vdp2_h_bu(tmp0, const_br);
    reg1 = __lsx_vmaddwev_h_bu(const_128, tmp1, const_150);
    reg2 = __lsx_vadd_h(reg0, reg1);
    tmp0 = __lsx_vpackod_b(reg2, reg2);
    tmp1 = __lsx_vpackod_b(tmp1, reg2);
    dst0 = __lsx_vilvl_h(tmp1, tmp0);
    dst1 = __lsx_vilvh_h(tmp1, tmp0);
    __lsx_vst(dst0, dst_argb, 0);
    __lsx_vst(dst1, dst_argb, 16);
    src_argb += 32;
    dst_argb += 32;
  }
}

void ARGBSepiaRow_LSX(uint8_t* dst_argb, int width) {
  int x;
  int len = width / 8;
  __m128i src0, src1, tmp0, tmp1;
  __m128i reg0, reg1, spb, spg, spr;
  __m128i dst0, dst1;
  __m128i spb_g = __lsx_vldi(68);
  __m128i spg_g = __lsx_vldi(88);
  __m128i spr_g = __lsx_vldi(98);
  __m128i spb_br = {0x2311231123112311, 0x2311231123112311};
  __m128i spg_br = {0x2D162D162D162D16, 0x2D162D162D162D16};
  __m128i spr_br = {0x3218321832183218, 0x3218321832183218};
  __m128i shuff = {0x1706150413021100, 0x1F0E1D0C1B0A1908};

  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lsx_vld, dst_argb, 0, dst_argb, 16, src0, src1);
    tmp0 = __lsx_vpickev_b(src1, src0);
    tmp1 = __lsx_vpickod_b(src1, src0);
    DUP2_ARG2(__lsx_vdp2_h_bu, tmp0, spb_br, tmp0, spg_br, spb, spg);
    spr = __lsx_vdp2_h_bu(tmp0, spr_br);
    spb = __lsx_vmaddwev_h_bu(spb, tmp1, spb_g);
    spg = __lsx_vmaddwev_h_bu(spg, tmp1, spg_g);
    spr = __lsx_vmaddwev_h_bu(spr, tmp1, spr_g);
    spb = __lsx_vsrli_h(spb, 7);
    spg = __lsx_vsrli_h(spg, 7);
    spr = __lsx_vsrli_h(spr, 7);
    spg = __lsx_vsat_hu(spg, 7);
    spr = __lsx_vsat_hu(spr, 7);
    reg0 = __lsx_vpackev_b(spg, spb);
    reg1 = __lsx_vshuf_b(tmp1, spr, shuff);
    dst0 = __lsx_vilvl_h(reg1, reg0);
    dst1 = __lsx_vilvh_h(reg1, reg0);
    __lsx_vst(dst0, dst_argb, 0);
    __lsx_vst(dst1, dst_argb, 16);
    dst_argb += 32;
  }
}

void ARGB4444ToARGBRow_LSX(const uint8_t* src_argb4444,
                           uint8_t* dst_argb,
                           int width) {
  int x;
  int len = width / 16;
  __m128i src0, src1;
  __m128i tmp0, tmp1, tmp2, tmp3;
  __m128i reg0, reg1, reg2, reg3;
  __m128i dst0, dst1, dst2, dst3;

  for (x = 0; x < len; x++) {
    src0 = __lsx_vld(src_argb4444, 0);
    src1 = __lsx_vld(src_argb4444, 16);
    tmp0 = __lsx_vandi_b(src0, 0x0F);
    tmp1 = __lsx_vandi_b(src0, 0xF0);
    tmp2 = __lsx_vandi_b(src1, 0x0F);
    tmp3 = __lsx_vandi_b(src1, 0xF0);
    reg0 = __lsx_vslli_b(tmp0, 4);
    reg2 = __lsx_vslli_b(tmp2, 4);
    reg1 = __lsx_vsrli_b(tmp1, 4);
    reg3 = __lsx_vsrli_b(tmp3, 4);
    DUP4_ARG2(__lsx_vor_v, tmp0, reg0, tmp1, reg1, tmp2, reg2, tmp3, reg3, tmp0,
              tmp1, tmp2, tmp3);
    dst0 = __lsx_vilvl_b(tmp1, tmp0);
    dst2 = __lsx_vilvl_b(tmp3, tmp2);
    dst1 = __lsx_vilvh_b(tmp1, tmp0);
    dst3 = __lsx_vilvh_b(tmp3, tmp2);
    __lsx_vst(dst0, dst_argb, 0);
    __lsx_vst(dst1, dst_argb, 16);
    __lsx_vst(dst2, dst_argb, 32);
    __lsx_vst(dst3, dst_argb, 48);
    dst_argb += 64;
    src_argb4444 += 32;
  }
}

void ARGB1555ToARGBRow_LSX(const uint8_t* src_argb1555,
                           uint8_t* dst_argb,
                           int width) {
  int x;
  int len = width / 16;
  __m128i src0, src1;
  __m128i tmp0, tmp1, tmpb, tmpg, tmpr, tmpa;
  __m128i reg0, reg1, reg2;
  __m128i dst0, dst1, dst2, dst3;

  for (x = 0; x < len; x++) {
    src0 = __lsx_vld(src_argb1555, 0);
    src1 = __lsx_vld(src_argb1555, 16);
    tmp0 = __lsx_vpickev_b(src1, src0);
    tmp1 = __lsx_vpickod_b(src1, src0);
    tmpb = __lsx_vandi_b(tmp0, 0x1F);
    tmpg = __lsx_vsrli_b(tmp0, 5);
    reg0 = __lsx_vandi_b(tmp1, 0x03);
    reg0 = __lsx_vslli_b(reg0, 3);
    tmpg = __lsx_vor_v(tmpg, reg0);
    reg1 = __lsx_vandi_b(tmp1, 0x7C);
    tmpr = __lsx_vsrli_b(reg1, 2);
    tmpa = __lsx_vsrli_b(tmp1, 7);
    tmpa = __lsx_vneg_b(tmpa);
    reg0 = __lsx_vslli_b(tmpb, 3);
    reg1 = __lsx_vslli_b(tmpg, 3);
    reg2 = __lsx_vslli_b(tmpr, 3);
    tmpb = __lsx_vsrli_b(tmpb, 2);
    tmpg = __lsx_vsrli_b(tmpg, 2);
    tmpr = __lsx_vsrli_b(tmpr, 2);
    tmpb = __lsx_vor_v(reg0, tmpb);
    tmpg = __lsx_vor_v(reg1, tmpg);
    tmpr = __lsx_vor_v(reg2, tmpr);
    DUP2_ARG2(__lsx_vilvl_b, tmpg, tmpb, tmpa, tmpr, reg0, reg1);
    dst0 = __lsx_vilvl_h(reg1, reg0);
    dst1 = __lsx_vilvh_h(reg1, reg0);
    DUP2_ARG2(__lsx_vilvh_b, tmpg, tmpb, tmpa, tmpr, reg0, reg1);
    dst2 = __lsx_vilvl_h(reg1, reg0);
    dst3 = __lsx_vilvh_h(reg1, reg0);
    __lsx_vst(dst0, dst_argb, 0);
    __lsx_vst(dst1, dst_argb, 16);
    __lsx_vst(dst2, dst_argb, 32);
    __lsx_vst(dst3, dst_argb, 48);
    dst_argb += 64;
    src_argb1555 += 32;
  }
}

void RGB565ToARGBRow_LSX(const uint8_t* src_rgb565,
                         uint8_t* dst_argb,
                         int width) {
  int x;
  int len = width / 16;
  __m128i src0, src1;
  __m128i tmp0, tmp1, tmpb, tmpg, tmpr;
  __m128i reg0, reg1, dst0, dst1, dst2, dst3;
  __m128i alpha = __lsx_vldi(0xFF);

  for (x = 0; x < len; x++) {
    src0 = __lsx_vld(src_rgb565, 0);
    src1 = __lsx_vld(src_rgb565, 16);
    tmp0 = __lsx_vpickev_b(src1, src0);
    tmp1 = __lsx_vpickod_b(src1, src0);
    tmpb = __lsx_vandi_b(tmp0, 0x1F);
    tmpr = __lsx_vandi_b(tmp1, 0xF8);
    reg1 = __lsx_vandi_b(tmp1, 0x07);
    reg0 = __lsx_vsrli_b(tmp0, 5);
    reg1 = __lsx_vslli_b(reg1, 3);
    tmpg = __lsx_vor_v(reg1, reg0);
    reg0 = __lsx_vslli_b(tmpb, 3);
    reg1 = __lsx_vsrli_b(tmpb, 2);
    tmpb = __lsx_vor_v(reg1, reg0);
    reg0 = __lsx_vslli_b(tmpg, 2);
    reg1 = __lsx_vsrli_b(tmpg, 4);
    tmpg = __lsx_vor_v(reg1, reg0);
    reg0 = __lsx_vsrli_b(tmpr, 5);
    tmpr = __lsx_vor_v(tmpr, reg0);
    DUP2_ARG2(__lsx_vilvl_b, tmpg, tmpb, alpha, tmpr, reg0, reg1);
    dst0 = __lsx_vilvl_h(reg1, reg0);
    dst1 = __lsx_vilvh_h(reg1, reg0);
    DUP2_ARG2(__lsx_vilvh_b, tmpg, tmpb, alpha, tmpr, reg0, reg1);
    dst2 = __lsx_vilvl_h(reg1, reg0);
    dst3 = __lsx_vilvh_h(reg1, reg0);
    __lsx_vst(dst0, dst_argb, 0);
    __lsx_vst(dst1, dst_argb, 16);
    __lsx_vst(dst2, dst_argb, 32);
    __lsx_vst(dst3, dst_argb, 48);
    dst_argb += 64;
    src_rgb565 += 32;
  }
}

void RGB24ToARGBRow_LSX(const uint8_t* src_rgb24,
                        uint8_t* dst_argb,
                        int width) {
  int x;
  int len = width / 16;
  __m128i src0, src1, src2;
  __m128i tmp0, tmp1, tmp2;
  __m128i dst0, dst1, dst2, dst3;
  __m128i alpha = __lsx_vldi(0xFF);
  __m128i shuf0 = {0x131211100F0E0D0C, 0x1B1A191817161514};
  __m128i shuf1 = {0x1F1E1D1C1B1A1918, 0x0706050403020100};
  __m128i shuf2 = {0x0B0A090807060504, 0x131211100F0E0D0C};
  __m128i shuf3 = {0x1005040310020100, 0x100B0A0910080706};

  for (x = 0; x < len; x++) {
    src0 = __lsx_vld(src_rgb24, 0);
    src1 = __lsx_vld(src_rgb24, 16);
    src2 = __lsx_vld(src_rgb24, 32);
    DUP2_ARG3(__lsx_vshuf_b, src1, src0, shuf0, src1, src2, shuf1, tmp0, tmp1);
    tmp2 = __lsx_vshuf_b(src1, src2, shuf2);
    DUP4_ARG3(__lsx_vshuf_b, alpha, src0, shuf3, alpha, tmp0, shuf3, alpha,
              tmp1, shuf3, alpha, tmp2, shuf3, dst0, dst1, dst2, dst3);
    __lsx_vst(dst0, dst_argb, 0);
    __lsx_vst(dst1, dst_argb, 16);
    __lsx_vst(dst2, dst_argb, 32);
    __lsx_vst(dst3, dst_argb, 48);
    dst_argb += 64;
    src_rgb24 += 48;
  }
}

void RAWToARGBRow_LSX(const uint8_t* src_raw, uint8_t* dst_argb, int width) {
  int x;
  int len = width / 16;
  __m128i src0, src1, src2;
  __m128i tmp0, tmp1, tmp2;
  __m128i dst0, dst1, dst2, dst3;
  __m128i alpha = __lsx_vldi(0xFF);
  __m128i shuf0 = {0x131211100F0E0D0C, 0x1B1A191817161514};
  __m128i shuf1 = {0x1F1E1D1C1B1A1918, 0x0706050403020100};
  __m128i shuf2 = {0x0B0A090807060504, 0x131211100F0E0D0C};
  __m128i shuf3 = {0x1003040510000102, 0x10090A0B10060708};

  for (x = 0; x < len; x++) {
    src0 = __lsx_vld(src_raw, 0);
    src1 = __lsx_vld(src_raw, 16);
    src2 = __lsx_vld(src_raw, 32);
    DUP2_ARG3(__lsx_vshuf_b, src1, src0, shuf0, src1, src2, shuf1, tmp0, tmp1);
    tmp2 = __lsx_vshuf_b(src1, src2, shuf2);
    DUP4_ARG3(__lsx_vshuf_b, alpha, src0, shuf3, alpha, tmp0, shuf3, alpha,
              tmp1, shuf3, alpha, tmp2, shuf3, dst0, dst1, dst2, dst3);
    __lsx_vst(dst0, dst_argb, 0);
    __lsx_vst(dst1, dst_argb, 16);
    __lsx_vst(dst2, dst_argb, 32);
    __lsx_vst(dst3, dst_argb, 48);
    dst_argb += 64;
    src_raw += 48;
  }
}

void ARGB1555ToYRow_LSX(const uint8_t* src_argb1555,
                        uint8_t* dst_y,
                        int width) {
  int x;
  int len = width / 16;
  __m128i src0, src1;
  __m128i tmp0, tmp1, tmpb, tmpg, tmpr;
  __m128i reg0, reg1, reg2, dst0;
  __m128i const_66 = __lsx_vldi(66);
  __m128i const_129 = __lsx_vldi(129);
  __m128i const_25 = __lsx_vldi(25);
  __m128i const_1080 = {0x1080108010801080, 0x1080108010801080};

  for (x = 0; x < len; x++) {
    src0 = __lsx_vld(src_argb1555, 0);
    src1 = __lsx_vld(src_argb1555, 16);
    tmp0 = __lsx_vpickev_b(src1, src0);
    tmp1 = __lsx_vpickod_b(src1, src0);
    tmpb = __lsx_vandi_b(tmp0, 0x1F);
    tmpg = __lsx_vsrli_b(tmp0, 5);
    reg0 = __lsx_vandi_b(tmp1, 0x03);
    reg0 = __lsx_vslli_b(reg0, 3);
    tmpg = __lsx_vor_v(tmpg, reg0);
    reg1 = __lsx_vandi_b(tmp1, 0x7C);
    tmpr = __lsx_vsrli_b(reg1, 2);
    reg0 = __lsx_vslli_b(tmpb, 3);
    reg1 = __lsx_vslli_b(tmpg, 3);
    reg2 = __lsx_vslli_b(tmpr, 3);
    tmpb = __lsx_vsrli_b(tmpb, 2);
    tmpg = __lsx_vsrli_b(tmpg, 2);
    tmpr = __lsx_vsrli_b(tmpr, 2);
    tmpb = __lsx_vor_v(reg0, tmpb);
    tmpg = __lsx_vor_v(reg1, tmpg);
    tmpr = __lsx_vor_v(reg2, tmpr);
    reg0 = __lsx_vmaddwev_h_bu(const_1080, tmpb, const_25);
    reg1 = __lsx_vmaddwod_h_bu(const_1080, tmpb, const_25);
    reg0 = __lsx_vmaddwev_h_bu(reg0, tmpg, const_129);
    reg1 = __lsx_vmaddwod_h_bu(reg1, tmpg, const_129);
    reg0 = __lsx_vmaddwev_h_bu(reg0, tmpr, const_66);
    reg1 = __lsx_vmaddwod_h_bu(reg1, tmpr, const_66);
    dst0 = __lsx_vpackod_b(reg1, reg0);
    __lsx_vst(dst0, dst_y, 0);
    dst_y += 16;
    src_argb1555 += 32;
  }
}

void ARGB1555ToUVRow_LSX(const uint8_t* src_argb1555,
                         int src_stride_argb1555,
                         uint8_t* dst_u,
                         uint8_t* dst_v,
                         int width) {
  int x;
  int len = width / 16;
  const uint8_t* next_argb1555 = src_argb1555 + src_stride_argb1555;
  __m128i src0, src1, src2, src3;
  __m128i tmp0, tmp1, tmp2, tmp3;
  __m128i tmpb, tmpg, tmpr, nexb, nexg, nexr;
  __m128i reg0, reg1, reg2, reg3, dst0;
  __m128i const_112 = __lsx_vldi(0x438);
  __m128i const_74 = __lsx_vldi(0x425);
  __m128i const_38 = __lsx_vldi(0x413);
  __m128i const_94 = __lsx_vldi(0x42F);
  __m128i const_18 = __lsx_vldi(0x409);
  __m128i const_8080 = (__m128i)v2u64{0x8080808080808080, 0x8080808080808080};

  for (x = 0; x < len; x++) {
    DUP4_ARG2(__lsx_vld, src_argb1555, 0, src_argb1555, 16, next_argb1555, 0,
              next_argb1555, 16, src0, src1, src2, src3);
    DUP2_ARG2(__lsx_vpickev_b, src1, src0, src3, src2, tmp0, tmp2);
    DUP2_ARG2(__lsx_vpickod_b, src1, src0, src3, src2, tmp1, tmp3);
    tmpb = __lsx_vandi_b(tmp0, 0x1F);
    nexb = __lsx_vandi_b(tmp2, 0x1F);
    tmpg = __lsx_vsrli_b(tmp0, 5);
    nexg = __lsx_vsrli_b(tmp2, 5);
    reg0 = __lsx_vandi_b(tmp1, 0x03);
    reg2 = __lsx_vandi_b(tmp3, 0x03);
    reg0 = __lsx_vslli_b(reg0, 3);
    reg2 = __lsx_vslli_b(reg2, 3);
    tmpg = __lsx_vor_v(tmpg, reg0);
    nexg = __lsx_vor_v(nexg, reg2);
    reg1 = __lsx_vandi_b(tmp1, 0x7C);
    reg3 = __lsx_vandi_b(tmp3, 0x7C);
    tmpr = __lsx_vsrli_b(reg1, 2);
    nexr = __lsx_vsrli_b(reg3, 2);
    reg0 = __lsx_vslli_b(tmpb, 3);
    reg1 = __lsx_vslli_b(tmpg, 3);
    reg2 = __lsx_vslli_b(tmpr, 3);
    tmpb = __lsx_vsrli_b(tmpb, 2);
    tmpg = __lsx_vsrli_b(tmpg, 2);
    tmpr = __lsx_vsrli_b(tmpr, 2);
    tmpb = __lsx_vor_v(reg0, tmpb);
    tmpg = __lsx_vor_v(reg1, tmpg);
    tmpr = __lsx_vor_v(reg2, tmpr);
    reg0 = __lsx_vslli_b(nexb, 3);
    reg1 = __lsx_vslli_b(nexg, 3);
    reg2 = __lsx_vslli_b(nexr, 3);
    nexb = __lsx_vsrli_b(nexb, 2);
    nexg = __lsx_vsrli_b(nexg, 2);
    nexr = __lsx_vsrli_b(nexr, 2);
    nexb = __lsx_vor_v(reg0, nexb);
    nexg = __lsx_vor_v(reg1, nexg);
    nexr = __lsx_vor_v(reg2, nexr);
    RGBTOUV(tmpb, tmpg, tmpr, nexb, nexg, nexr, dst0);
    __lsx_vstelm_d(dst0, dst_u, 0, 0);
    __lsx_vstelm_d(dst0, dst_v, 0, 1);
    dst_u += 8;
    dst_v += 8;
    src_argb1555 += 32;
    next_argb1555 += 32;
  }
}

void RGB565ToYRow_LSX(const uint8_t* src_rgb565, uint8_t* dst_y, int width) {
  int x;
  int len = width / 16;
  __m128i src0, src1;
  __m128i tmp0, tmp1, tmpb, tmpg, tmpr;
  __m128i reg0, reg1, dst0;
  __m128i const_66 = __lsx_vldi(66);
  __m128i const_129 = __lsx_vldi(129);
  __m128i const_25 = __lsx_vldi(25);
  __m128i const_1080 = {0x1080108010801080, 0x1080108010801080};

  for (x = 0; x < len; x++) {
    src0 = __lsx_vld(src_rgb565, 0);
    src1 = __lsx_vld(src_rgb565, 16);
    tmp0 = __lsx_vpickev_b(src1, src0);
    tmp1 = __lsx_vpickod_b(src1, src0);
    tmpb = __lsx_vandi_b(tmp0, 0x1F);
    tmpr = __lsx_vandi_b(tmp1, 0xF8);
    reg1 = __lsx_vandi_b(tmp1, 0x07);
    reg0 = __lsx_vsrli_b(tmp0, 5);
    reg1 = __lsx_vslli_b(reg1, 3);
    tmpg = __lsx_vor_v(reg1, reg0);
    reg0 = __lsx_vslli_b(tmpb, 3);
    reg1 = __lsx_vsrli_b(tmpb, 2);
    tmpb = __lsx_vor_v(reg1, reg0);
    reg0 = __lsx_vslli_b(tmpg, 2);
    reg1 = __lsx_vsrli_b(tmpg, 4);
    tmpg = __lsx_vor_v(reg1, reg0);
    reg0 = __lsx_vsrli_b(tmpr, 5);
    tmpr = __lsx_vor_v(tmpr, reg0);
    reg0 = __lsx_vmaddwev_h_bu(const_1080, tmpb, const_25);
    reg1 = __lsx_vmaddwod_h_bu(const_1080, tmpb, const_25);
    reg0 = __lsx_vmaddwev_h_bu(reg0, tmpg, const_129);
    reg1 = __lsx_vmaddwod_h_bu(reg1, tmpg, const_129);
    reg0 = __lsx_vmaddwev_h_bu(reg0, tmpr, const_66);
    reg1 = __lsx_vmaddwod_h_bu(reg1, tmpr, const_66);
    dst0 = __lsx_vpackod_b(reg1, reg0);
    __lsx_vst(dst0, dst_y, 0);
    dst_y += 16;
    src_rgb565 += 32;
  }
}

void RGB565ToUVRow_LSX(const uint8_t* src_rgb565,
                       int src_stride_rgb565,
                       uint8_t* dst_u,
                       uint8_t* dst_v,
                       int width) {
  int x;
  int len = width / 16;
  const uint8_t* next_rgb565 = src_rgb565 + src_stride_rgb565;
  __m128i src0, src1, src2, src3;
  __m128i tmp0, tmp1, tmp2, tmp3;
  __m128i tmpb, tmpg, tmpr, nexb, nexg, nexr;
  __m128i reg0, reg1, reg2, reg3, dst0;
  __m128i const_112 = __lsx_vldi(0x438);
  __m128i const_74 = __lsx_vldi(0x425);
  __m128i const_38 = __lsx_vldi(0x413);
  __m128i const_94 = __lsx_vldi(0x42F);
  __m128i const_18 = __lsx_vldi(0x409);
  __m128i const_8080 = (__m128i)v2u64{0x8080808080808080, 0x8080808080808080};

  for (x = 0; x < len; x++) {
    DUP4_ARG2(__lsx_vld, src_rgb565, 0, src_rgb565, 16, next_rgb565, 0,
              next_rgb565, 16, src0, src1, src2, src3);
    DUP2_ARG2(__lsx_vpickev_b, src1, src0, src3, src2, tmp0, tmp2);
    DUP2_ARG2(__lsx_vpickod_b, src1, src0, src3, src2, tmp1, tmp3);
    tmpb = __lsx_vandi_b(tmp0, 0x1F);
    tmpr = __lsx_vandi_b(tmp1, 0xF8);
    nexb = __lsx_vandi_b(tmp2, 0x1F);
    nexr = __lsx_vandi_b(tmp3, 0xF8);
    reg1 = __lsx_vandi_b(tmp1, 0x07);
    reg3 = __lsx_vandi_b(tmp3, 0x07);
    reg0 = __lsx_vsrli_b(tmp0, 5);
    reg1 = __lsx_vslli_b(reg1, 3);
    reg2 = __lsx_vsrli_b(tmp2, 5);
    reg3 = __lsx_vslli_b(reg3, 3);
    tmpg = __lsx_vor_v(reg1, reg0);
    nexg = __lsx_vor_v(reg2, reg3);
    reg0 = __lsx_vslli_b(tmpb, 3);
    reg1 = __lsx_vsrli_b(tmpb, 2);
    reg2 = __lsx_vslli_b(nexb, 3);
    reg3 = __lsx_vsrli_b(nexb, 2);
    tmpb = __lsx_vor_v(reg1, reg0);
    nexb = __lsx_vor_v(reg2, reg3);
    reg0 = __lsx_vslli_b(tmpg, 2);
    reg1 = __lsx_vsrli_b(tmpg, 4);
    reg2 = __lsx_vslli_b(nexg, 2);
    reg3 = __lsx_vsrli_b(nexg, 4);
    tmpg = __lsx_vor_v(reg1, reg0);
    nexg = __lsx_vor_v(reg2, reg3);
    reg0 = __lsx_vsrli_b(tmpr, 5);
    reg2 = __lsx_vsrli_b(nexr, 5);
    tmpr = __lsx_vor_v(tmpr, reg0);
    nexr = __lsx_vor_v(nexr, reg2);
    RGBTOUV(tmpb, tmpg, tmpr, nexb, nexg, nexr, dst0);
    __lsx_vstelm_d(dst0, dst_u, 0, 0);
    __lsx_vstelm_d(dst0, dst_v, 0, 1);
    dst_u += 8;
    dst_v += 8;
    src_rgb565 += 32;
    next_rgb565 += 32;
  }
}

void RGB24ToUVRow_LSX(const uint8_t* src_rgb24,
                      int src_stride_rgb24,
                      uint8_t* dst_u,
                      uint8_t* dst_v,
                      int width) {
  int x;
  const uint8_t* next_rgb24 = src_rgb24 + src_stride_rgb24;
  int len = width / 16;
  __m128i src0, src1, src2;
  __m128i nex0, nex1, nex2, dst0;
  __m128i tmpb, tmpg, tmpr, nexb, nexg, nexr;
  __m128i const_112 = __lsx_vldi(0x438);
  __m128i const_74 = __lsx_vldi(0x425);
  __m128i const_38 = __lsx_vldi(0x413);
  __m128i const_94 = __lsx_vldi(0x42F);
  __m128i const_18 = __lsx_vldi(0x409);
  __m128i const_8080 = (__m128i)v2u64{0x8080808080808080, 0x8080808080808080};
  __m128i shuff0_b = {0x15120F0C09060300, 0x00000000001E1B18};
  __m128i shuff1_b = {0x0706050403020100, 0x1D1A1714110A0908};
  __m128i shuff0_g = {0x1613100D0A070401, 0x00000000001F1C19};
  __m128i shuff1_g = {0x0706050403020100, 0x1E1B1815120A0908};
  __m128i shuff0_r = {0x1714110E0B080502, 0x0000000000001D1A};
  __m128i shuff1_r = {0x0706050403020100, 0x1F1C191613100908};

  for (x = 0; x < len; x++) {
    src0 = __lsx_vld(src_rgb24, 0);
    src1 = __lsx_vld(src_rgb24, 16);
    src2 = __lsx_vld(src_rgb24, 32);
    nex0 = __lsx_vld(next_rgb24, 0);
    nex1 = __lsx_vld(next_rgb24, 16);
    nex2 = __lsx_vld(next_rgb24, 32);
    DUP2_ARG3(__lsx_vshuf_b, src1, src0, shuff0_b, nex1, nex0, shuff0_b, tmpb,
              nexb);
    DUP2_ARG3(__lsx_vshuf_b, src1, src0, shuff0_g, nex1, nex0, shuff0_g, tmpg,
              nexg);
    DUP2_ARG3(__lsx_vshuf_b, src1, src0, shuff0_r, nex1, nex0, shuff0_r, tmpr,
              nexr);
    DUP2_ARG3(__lsx_vshuf_b, src2, tmpb, shuff1_b, nex2, nexb, shuff1_b, tmpb,
              nexb);
    DUP2_ARG3(__lsx_vshuf_b, src2, tmpg, shuff1_g, nex2, nexg, shuff1_g, tmpg,
              nexg);
    DUP2_ARG3(__lsx_vshuf_b, src2, tmpr, shuff1_r, nex2, nexr, shuff1_r, tmpr,
              nexr);
    RGBTOUV(tmpb, tmpg, tmpr, nexb, nexg, nexr, dst0);
    __lsx_vstelm_d(dst0, dst_u, 0, 0);
    __lsx_vstelm_d(dst0, dst_v, 0, 1);
    dst_u += 8;
    dst_v += 8;
    src_rgb24 += 48;
    next_rgb24 += 48;
  }
}

void RAWToUVRow_LSX(const uint8_t* src_raw,
                    int src_stride_raw,
                    uint8_t* dst_u,
                    uint8_t* dst_v,
                    int width) {
  int x;
  const uint8_t* next_raw = src_raw + src_stride_raw;
  int len = width / 16;
  __m128i src0, src1, src2;
  __m128i nex0, nex1, nex2, dst0;
  __m128i tmpb, tmpg, tmpr, nexb, nexg, nexr;
  __m128i const_112 = __lsx_vldi(0x438);
  __m128i const_74 = __lsx_vldi(0x425);
  __m128i const_38 = __lsx_vldi(0x413);
  __m128i const_94 = __lsx_vldi(0x42F);
  __m128i const_18 = __lsx_vldi(0x409);
  __m128i const_8080 = (__m128i)v2u64{0x8080808080808080, 0x8080808080808080};
  __m128i shuff0_r = {0x15120F0C09060300, 0x00000000001E1B18};
  __m128i shuff1_r = {0x0706050403020100, 0x1D1A1714110A0908};
  __m128i shuff0_g = {0x1613100D0A070401, 0x00000000001F1C19};
  __m128i shuff1_g = {0x0706050403020100, 0x1E1B1815120A0908};
  __m128i shuff0_b = {0x1714110E0B080502, 0x0000000000001D1A};
  __m128i shuff1_b = {0x0706050403020100, 0x1F1C191613100908};

  for (x = 0; x < len; x++) {
    src0 = __lsx_vld(src_raw, 0);
    src1 = __lsx_vld(src_raw, 16);
    src2 = __lsx_vld(src_raw, 32);
    nex0 = __lsx_vld(next_raw, 0);
    nex1 = __lsx_vld(next_raw, 16);
    nex2 = __lsx_vld(next_raw, 32);
    DUP2_ARG3(__lsx_vshuf_b, src1, src0, shuff0_b, nex1, nex0, shuff0_b, tmpb,
              nexb);
    DUP2_ARG3(__lsx_vshuf_b, src1, src0, shuff0_g, nex1, nex0, shuff0_g, tmpg,
              nexg);
    DUP2_ARG3(__lsx_vshuf_b, src1, src0, shuff0_r, nex1, nex0, shuff0_r, tmpr,
              nexr);
    DUP2_ARG3(__lsx_vshuf_b, src2, tmpb, shuff1_b, nex2, nexb, shuff1_b, tmpb,
              nexb);
    DUP2_ARG3(__lsx_vshuf_b, src2, tmpg, shuff1_g, nex2, nexg, shuff1_g, tmpg,
              nexg);
    DUP2_ARG3(__lsx_vshuf_b, src2, tmpr, shuff1_r, nex2, nexr, shuff1_r, tmpr,
              nexr);
    RGBTOUV(tmpb, tmpg, tmpr, nexb, nexg, nexr, dst0);
    __lsx_vstelm_d(dst0, dst_u, 0, 0);
    __lsx_vstelm_d(dst0, dst_v, 0, 1);
    dst_u += 8;
    dst_v += 8;
    src_raw += 48;
    next_raw += 48;
  }
}

void NV12ToARGBRow_LSX(const uint8_t* src_y,
                       const uint8_t* src_uv,
                       uint8_t* dst_argb,
                       const struct YuvConstants* yuvconstants,
                       int width) {
  int x;
  int len = width / 8;
  __m128i vec_y, vec_vu;
  __m128i vec_vr, vec_ub, vec_vg, vec_ug, vec_yg, vec_yb;
  __m128i vec_vrub, vec_vgug;
  __m128i out_b, out_g, out_r;
  __m128i const_80 = __lsx_vldi(0x480);
  __m128i alpha = __lsx_vldi(0xFF);
  __m128i zero = __lsx_vldi(0);

  YUVTORGB_SETUP(yuvconstants, vec_vr, vec_ub, vec_vg, vec_ug, vec_yg, vec_yb);
  vec_vrub = __lsx_vilvl_h(vec_vr, vec_ub);
  vec_vgug = __lsx_vilvl_h(vec_vg, vec_ug);

  for (x = 0; x < len; x++) {
    vec_y = __lsx_vld(src_y, 0);
    vec_vu = __lsx_vld(src_uv, 0);
    YUVTORGB(vec_y, vec_vu, vec_vrub, vec_vgug, vec_yg, vec_yb, out_b, out_g,
             out_r);
    STOREARGB(alpha, out_r, out_g, out_b, dst_argb);
    src_y += 8;
    src_uv += 8;
  }
}

void NV12ToRGB565Row_LSX(const uint8_t* src_y,
                         const uint8_t* src_uv,
                         uint8_t* dst_rgb565,
                         const struct YuvConstants* yuvconstants,
                         int width) {
  int x;
  int len = width / 8;
  __m128i vec_y, vec_vu;
  __m128i vec_vr, vec_ub, vec_vg, vec_ug, vec_yg, vec_yb;
  __m128i vec_vrub, vec_vgug;
  __m128i out_b, out_g, out_r;
  __m128i const_80 = __lsx_vldi(0x480);
  __m128i zero = __lsx_vldi(0);

  YUVTORGB_SETUP(yuvconstants, vec_vr, vec_ub, vec_vg, vec_ug, vec_yg, vec_yb);
  vec_vrub = __lsx_vilvl_h(vec_vr, vec_ub);
  vec_vgug = __lsx_vilvl_h(vec_vg, vec_ug);

  for (x = 0; x < len; x++) {
    vec_y = __lsx_vld(src_y, 0);
    vec_vu = __lsx_vld(src_uv, 0);
    YUVTORGB(vec_y, vec_vu, vec_vrub, vec_vgug, vec_yg, vec_yb, out_b, out_g,
             out_r);
    out_b = __lsx_vsrli_h(out_b, 3);
    out_g = __lsx_vsrli_h(out_g, 2);
    out_r = __lsx_vsrli_h(out_r, 3);
    out_g = __lsx_vslli_h(out_g, 5);
    out_r = __lsx_vslli_h(out_r, 11);
    out_r = __lsx_vor_v(out_r, out_g);
    out_r = __lsx_vor_v(out_r, out_b);
    __lsx_vst(out_r, dst_rgb565, 0);
    src_y += 8;
    src_uv += 8;
    dst_rgb565 += 16;
  }
}

void NV21ToARGBRow_LSX(const uint8_t* src_y,
                       const uint8_t* src_vu,
                       uint8_t* dst_argb,
                       const struct YuvConstants* yuvconstants,
                       int width) {
  int x;
  int len = width / 8;
  __m128i vec_y, vec_uv;
  __m128i vec_vr, vec_ub, vec_vg, vec_ug, vec_yg, vec_yb;
  __m128i vec_ubvr, vec_ugvg;
  __m128i out_b, out_g, out_r;
  __m128i const_80 = __lsx_vldi(0x480);
  __m128i alpha = __lsx_vldi(0xFF);
  __m128i zero = __lsx_vldi(0);

  YUVTORGB_SETUP(yuvconstants, vec_vr, vec_ub, vec_vg, vec_ug, vec_yg, vec_yb);
  vec_ubvr = __lsx_vilvl_h(vec_ub, vec_vr);
  vec_ugvg = __lsx_vilvl_h(vec_ug, vec_vg);

  for (x = 0; x < len; x++) {
    vec_y = __lsx_vld(src_y, 0);
    vec_uv = __lsx_vld(src_vu, 0);
    YUVTORGB(vec_y, vec_uv, vec_ubvr, vec_ugvg, vec_yg, vec_yb, out_r, out_g,
             out_b);
    STOREARGB(alpha, out_r, out_g, out_b, dst_argb);
    src_y += 8;
    src_vu += 8;
  }
}

void SobelRow_LSX(const uint8_t* src_sobelx,
                  const uint8_t* src_sobely,
                  uint8_t* dst_argb,
                  int width) {
  int x;
  int len = width / 16;
  __m128i src0, src1, tmp0;
  __m128i out0, out1, out2, out3;
  __m128i alpha = __lsx_vldi(0xFF);
  __m128i shuff0 = {0x1001010110000000, 0x1003030310020202};
  __m128i shuff1 = __lsx_vaddi_bu(shuff0, 0x04);
  __m128i shuff2 = __lsx_vaddi_bu(shuff1, 0x04);
  __m128i shuff3 = __lsx_vaddi_bu(shuff2, 0x04);

  for (x = 0; x < len; x++) {
    src0 = __lsx_vld(src_sobelx, 0);
    src1 = __lsx_vld(src_sobely, 0);
    tmp0 = __lsx_vsadd_bu(src0, src1);
    DUP4_ARG3(__lsx_vshuf_b, alpha, tmp0, shuff0, alpha, tmp0, shuff1, alpha,
              tmp0, shuff2, alpha, tmp0, shuff3, out0, out1, out2, out3);
    __lsx_vst(out0, dst_argb, 0);
    __lsx_vst(out1, dst_argb, 16);
    __lsx_vst(out2, dst_argb, 32);
    __lsx_vst(out3, dst_argb, 48);
    src_sobelx += 16;
    src_sobely += 16;
    dst_argb += 64;
  }
}

void SobelToPlaneRow_LSX(const uint8_t* src_sobelx,
                         const uint8_t* src_sobely,
                         uint8_t* dst_y,
                         int width) {
  int x;
  int len = width / 32;
  __m128i src0, src1, src2, src3, dst0, dst1;

  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lsx_vld, src_sobelx, 0, src_sobelx, 16, src0, src1);
    DUP2_ARG2(__lsx_vld, src_sobely, 0, src_sobely, 16, src2, src3);
    dst0 = __lsx_vsadd_bu(src0, src2);
    dst1 = __lsx_vsadd_bu(src1, src3);
    __lsx_vst(dst0, dst_y, 0);
    __lsx_vst(dst1, dst_y, 16);
    src_sobelx += 32;
    src_sobely += 32;
    dst_y += 32;
  }
}

void SobelXYRow_LSX(const uint8_t* src_sobelx,
                    const uint8_t* src_sobely,
                    uint8_t* dst_argb,
                    int width) {
  int x;
  int len = width / 16;
  __m128i src_r, src_b, src_g;
  __m128i tmp0, tmp1, tmp2, tmp3;
  __m128i dst0, dst1, dst2, dst3;
  __m128i alpha = __lsx_vldi(0xFF);

  for (x = 0; x < len; x++) {
    src_r = __lsx_vld(src_sobelx, 0);
    src_b = __lsx_vld(src_sobely, 0);
    src_g = __lsx_vsadd_bu(src_r, src_b);
    tmp0 = __lsx_vilvl_b(src_g, src_b);
    tmp1 = __lsx_vilvh_b(src_g, src_b);
    tmp2 = __lsx_vilvl_b(alpha, src_r);
    tmp3 = __lsx_vilvh_b(alpha, src_r);
    dst0 = __lsx_vilvl_h(tmp2, tmp0);
    dst1 = __lsx_vilvh_h(tmp2, tmp0);
    dst2 = __lsx_vilvl_h(tmp3, tmp1);
    dst3 = __lsx_vilvh_h(tmp3, tmp1);
    __lsx_vst(dst0, dst_argb, 0);
    __lsx_vst(dst1, dst_argb, 16);
    __lsx_vst(dst2, dst_argb, 32);
    __lsx_vst(dst3, dst_argb, 48);
    src_sobelx += 16;
    src_sobely += 16;
    dst_argb += 64;
  }
}

void BGRAToUVRow_LSX(const uint8_t* src_bgra,
                     int src_stride_bgra,
                     uint8_t* dst_u,
                     uint8_t* dst_v,
                     int width) {
  int x;
  const uint8_t* next_bgra = src_bgra + src_stride_bgra;
  int len = width / 16;
  __m128i src0, src1, src2, src3;
  __m128i nex0, nex1, nex2, nex3;
  __m128i tmp0, tmp1, tmp2, tmp3, dst0;
  __m128i tmpb, tmpg, tmpr, nexb, nexg, nexr;
  __m128i const_112 = __lsx_vldi(0x438);
  __m128i const_74 = __lsx_vldi(0x425);
  __m128i const_38 = __lsx_vldi(0x413);
  __m128i const_94 = __lsx_vldi(0x42F);
  __m128i const_18 = __lsx_vldi(0x409);
  __m128i const_8080 = (__m128i)v2u64{0x8080808080808080, 0x8080808080808080};

  for (x = 0; x < len; x++) {
    DUP4_ARG2(__lsx_vld, src_bgra, 0, src_bgra, 16, src_bgra, 32, src_bgra, 48,
              src0, src1, src2, src3);
    DUP4_ARG2(__lsx_vld, next_bgra, 0, next_bgra, 16, next_bgra, 32, next_bgra,
              48, nex0, nex1, nex2, nex3);
    tmp0 = __lsx_vpickod_b(src1, src0);
    tmp1 = __lsx_vpickev_b(src1, src0);
    tmp2 = __lsx_vpickod_b(src3, src2);
    tmp3 = __lsx_vpickev_b(src3, src2);
    tmpb = __lsx_vpickod_b(tmp2, tmp0);
    tmpr = __lsx_vpickev_b(tmp2, tmp0);
    tmpg = __lsx_vpickod_b(tmp3, tmp1);
    tmp0 = __lsx_vpickod_b(nex1, nex0);
    tmp1 = __lsx_vpickev_b(nex1, nex0);
    tmp2 = __lsx_vpickod_b(nex3, nex2);
    tmp3 = __lsx_vpickev_b(nex3, nex2);
    nexb = __lsx_vpickod_b(tmp2, tmp0);
    nexr = __lsx_vpickev_b(tmp2, tmp0);
    nexg = __lsx_vpickod_b(tmp3, tmp1);
    RGBTOUV(tmpb, tmpg, tmpr, nexb, nexg, nexr, dst0);
    __lsx_vstelm_d(dst0, dst_u, 0, 0);
    __lsx_vstelm_d(dst0, dst_v, 0, 1);
    dst_u += 8;
    dst_v += 8;
    src_bgra += 64;
    next_bgra += 64;
  }
}

void ABGRToUVRow_LSX(const uint8_t* src_abgr,
                     int src_stride_abgr,
                     uint8_t* dst_u,
                     uint8_t* dst_v,
                     int width) {
  int x;
  const uint8_t* next_abgr = src_abgr + src_stride_abgr;
  int len = width / 16;
  __m128i src0, src1, src2, src3;
  __m128i nex0, nex1, nex2, nex3;
  __m128i tmp0, tmp1, tmp2, tmp3, dst0;
  __m128i tmpb, tmpg, tmpr, nexb, nexg, nexr;
  __m128i const_112 = __lsx_vldi(0x438);
  __m128i const_74 = __lsx_vldi(0x425);
  __m128i const_38 = __lsx_vldi(0x413);
  __m128i const_94 = __lsx_vldi(0x42F);
  __m128i const_18 = __lsx_vldi(0x409);
  __m128i const_8080 = (__m128i)v2u64{0x8080808080808080, 0x8080808080808080};

  for (x = 0; x < len; x++) {
    DUP4_ARG2(__lsx_vld, src_abgr, 0, src_abgr, 16, src_abgr, 32, src_abgr, 48,
              src0, src1, src2, src3);
    DUP4_ARG2(__lsx_vld, next_abgr, 0, next_abgr, 16, next_abgr, 32, next_abgr,
              48, nex0, nex1, nex2, nex3);
    tmp0 = __lsx_vpickev_b(src1, src0);
    tmp1 = __lsx_vpickod_b(src1, src0);
    tmp2 = __lsx_vpickev_b(src3, src2);
    tmp3 = __lsx_vpickod_b(src3, src2);
    tmpb = __lsx_vpickod_b(tmp2, tmp0);
    tmpr = __lsx_vpickev_b(tmp2, tmp0);
    tmpg = __lsx_vpickev_b(tmp3, tmp1);
    tmp0 = __lsx_vpickev_b(nex1, nex0);
    tmp1 = __lsx_vpickod_b(nex1, nex0);
    tmp2 = __lsx_vpickev_b(nex3, nex2);
    tmp3 = __lsx_vpickod_b(nex3, nex2);
    nexb = __lsx_vpickod_b(tmp2, tmp0);
    nexr = __lsx_vpickev_b(tmp2, tmp0);
    nexg = __lsx_vpickev_b(tmp3, tmp1);
    RGBTOUV(tmpb, tmpg, tmpr, nexb, nexg, nexr, dst0);
    __lsx_vstelm_d(dst0, dst_u, 0, 0);
    __lsx_vstelm_d(dst0, dst_v, 0, 1);
    dst_u += 8;
    dst_v += 8;
    src_abgr += 64;
    next_abgr += 64;
  }
}

void RGBAToUVRow_LSX(const uint8_t* src_rgba,
                     int src_stride_rgba,
                     uint8_t* dst_u,
                     uint8_t* dst_v,
                     int width) {
  int x;
  const uint8_t* next_rgba = src_rgba + src_stride_rgba;
  int len = width / 16;
  __m128i src0, src1, src2, src3;
  __m128i nex0, nex1, nex2, nex3;
  __m128i tmp0, tmp1, tmp2, tmp3, dst0;
  __m128i tmpb, tmpg, tmpr, nexb, nexg, nexr;
  __m128i const_112 = __lsx_vldi(0x438);
  __m128i const_74 = __lsx_vldi(0x425);
  __m128i const_38 = __lsx_vldi(0x413);
  __m128i const_94 = __lsx_vldi(0x42F);
  __m128i const_18 = __lsx_vldi(0x409);
  __m128i const_8080 = (__m128i)v2u64{0x8080808080808080, 0x8080808080808080};

  for (x = 0; x < len; x++) {
    DUP4_ARG2(__lsx_vld, src_rgba, 0, src_rgba, 16, src_rgba, 32, src_rgba, 48,
              src0, src1, src2, src3);
    DUP4_ARG2(__lsx_vld, next_rgba, 0, next_rgba, 16, next_rgba, 32, next_rgba,
              48, nex0, nex1, nex2, nex3);
    tmp0 = __lsx_vpickod_b(src1, src0);
    tmp1 = __lsx_vpickev_b(src1, src0);
    tmp2 = __lsx_vpickod_b(src3, src2);
    tmp3 = __lsx_vpickev_b(src3, src2);
    tmpr = __lsx_vpickod_b(tmp2, tmp0);
    tmpb = __lsx_vpickev_b(tmp2, tmp0);
    tmpg = __lsx_vpickod_b(tmp3, tmp1);
    tmp0 = __lsx_vpickod_b(nex1, nex0);
    tmp1 = __lsx_vpickev_b(nex1, nex0);
    tmp2 = __lsx_vpickod_b(nex3, nex2);
    tmp3 = __lsx_vpickev_b(nex3, nex2);
    nexr = __lsx_vpickod_b(tmp2, tmp0);
    nexb = __lsx_vpickev_b(tmp2, tmp0);
    nexg = __lsx_vpickod_b(tmp3, tmp1);
    RGBTOUV(tmpb, tmpg, tmpr, nexb, nexg, nexr, dst0);
    __lsx_vstelm_d(dst0, dst_u, 0, 0);
    __lsx_vstelm_d(dst0, dst_v, 0, 1);
    dst_u += 8;
    dst_v += 8;
    src_rgba += 64;
    next_rgba += 64;
  }
}

void ARGBToUVJRow_LSX(const uint8_t* src_argb,
                      int src_stride_argb,
                      uint8_t* dst_u,
                      uint8_t* dst_v,
                      int width) {
  int x;
  const uint8_t* next_argb = src_argb + src_stride_argb;
  int len = width / 16;
  __m128i src0, src1, src2, src3;
  __m128i nex0, nex1, nex2, nex3;
  __m128i tmp0, tmp1, tmp2, tmp3;
  __m128i reg0, reg1, dst0;
  __m128i tmpb, tmpg, tmpr, nexb, nexg, nexr;
  __m128i const_63 = __lsx_vldi(0x43F);
  __m128i const_42 = __lsx_vldi(0x42A);
  __m128i const_21 = __lsx_vldi(0x415);
  __m128i const_53 = __lsx_vldi(0x435);
  __m128i const_10 = __lsx_vldi(0x40A);
  __m128i const_8080 = (__m128i)v2u64{0x8080808080808080, 0x8080808080808080};

  for (x = 0; x < len; x++) {
    DUP4_ARG2(__lsx_vld, src_argb, 0, src_argb, 16, src_argb, 32, src_argb, 48,
              src0, src1, src2, src3);
    DUP4_ARG2(__lsx_vld, next_argb, 0, next_argb, 16, next_argb, 32, next_argb,
              48, nex0, nex1, nex2, nex3);
    tmp0 = __lsx_vpickev_b(src1, src0);
    tmp1 = __lsx_vpickod_b(src1, src0);
    tmp2 = __lsx_vpickev_b(src3, src2);
    tmp3 = __lsx_vpickod_b(src3, src2);
    tmpr = __lsx_vpickod_b(tmp2, tmp0);
    tmpb = __lsx_vpickev_b(tmp2, tmp0);
    tmpg = __lsx_vpickev_b(tmp3, tmp1);
    tmp0 = __lsx_vpickev_b(nex1, nex0);
    tmp1 = __lsx_vpickod_b(nex1, nex0);
    tmp2 = __lsx_vpickev_b(nex3, nex2);
    tmp3 = __lsx_vpickod_b(nex3, nex2);
    nexr = __lsx_vpickod_b(tmp2, tmp0);
    nexb = __lsx_vpickev_b(tmp2, tmp0);
    nexg = __lsx_vpickev_b(tmp3, tmp1);
    tmp0 = __lsx_vaddwev_h_bu(tmpb, nexb);
    tmp1 = __lsx_vaddwod_h_bu(tmpb, nexb);
    tmp2 = __lsx_vaddwev_h_bu(tmpg, nexg);
    tmp3 = __lsx_vaddwod_h_bu(tmpg, nexg);
    reg0 = __lsx_vaddwev_h_bu(tmpr, nexr);
    reg1 = __lsx_vaddwod_h_bu(tmpr, nexr);
    tmpb = __lsx_vavgr_hu(tmp0, tmp1);
    tmpg = __lsx_vavgr_hu(tmp2, tmp3);
    tmpr = __lsx_vavgr_hu(reg0, reg1);
    reg0 = __lsx_vmadd_h(const_8080, const_63, tmpb);
    reg1 = __lsx_vmadd_h(const_8080, const_63, tmpr);
    reg0 = __lsx_vmsub_h(reg0, const_42, tmpg);
    reg1 = __lsx_vmsub_h(reg1, const_53, tmpg);
    reg0 = __lsx_vmsub_h(reg0, const_21, tmpr);
    reg1 = __lsx_vmsub_h(reg1, const_10, tmpb);
    dst0 = __lsx_vpickod_b(reg1, reg0);
    __lsx_vstelm_d(dst0, dst_u, 0, 0);
    __lsx_vstelm_d(dst0, dst_v, 0, 1);
    dst_u += 8;
    dst_v += 8;
    src_argb += 64;
    next_argb += 64;
  }
}

void I444ToARGBRow_LSX(const uint8_t* src_y,
                       const uint8_t* src_u,
                       const uint8_t* src_v,
                       uint8_t* dst_argb,
                       const struct YuvConstants* yuvconstants,
                       int width) {
  int x;
  int len = width / 16;
  __m128i vec_y, vec_u, vec_v, out_b, out_g, out_r;
  __m128i vec_yl, vec_yh, vec_ul, vec_vl, vec_uh, vec_vh;
  __m128i vec_vr, vec_ub, vec_vg, vec_ug, vec_yg, vec_yb, vec_ugvg;
  __m128i const_80 = __lsx_vldi(0x480);
  __m128i alpha = __lsx_vldi(0xFF);
  __m128i zero = __lsx_vldi(0);

  YUVTORGB_SETUP(yuvconstants, vec_vr, vec_ub, vec_vg, vec_ug, vec_yg, vec_yb);
  vec_ugvg = __lsx_vilvl_h(vec_ug, vec_vg);

  for (x = 0; x < len; x++) {
    vec_y = __lsx_vld(src_y, 0);
    vec_u = __lsx_vld(src_u, 0);
    vec_v = __lsx_vld(src_v, 0);
    vec_yl = __lsx_vilvl_b(vec_y, vec_y);
    vec_ul = __lsx_vilvl_b(zero, vec_u);
    vec_vl = __lsx_vilvl_b(zero, vec_v);
    I444TORGB(vec_yl, vec_ul, vec_vl, vec_ub, vec_vr, vec_ugvg, vec_yg, vec_yb,
              out_b, out_g, out_r);
    STOREARGB(alpha, out_r, out_g, out_b, dst_argb);
    vec_yh = __lsx_vilvh_b(vec_y, vec_y);
    vec_uh = __lsx_vilvh_b(zero, vec_u);
    vec_vh = __lsx_vilvh_b(zero, vec_v);
    I444TORGB(vec_yh, vec_uh, vec_vh, vec_ub, vec_vr, vec_ugvg, vec_yg, vec_yb,
              out_b, out_g, out_r);
    STOREARGB(alpha, out_r, out_g, out_b, dst_argb);
    src_y += 16;
    src_u += 16;
    src_v += 16;
  }
}

void I400ToARGBRow_LSX(const uint8_t* src_y,
                       uint8_t* dst_argb,
                       const struct YuvConstants* yuvconstants,
                       int width) {
  int x;
  int len = width / 16;
  __m128i vec_y, vec_yl, vec_yh, out0;
  __m128i y_ev, y_od, dst0, dst1, dst2, dst3;
  __m128i temp0, temp1;
  __m128i alpha = __lsx_vldi(0xFF);
  __m128i vec_yg = __lsx_vreplgr2vr_h(yuvconstants->kYToRgb[0]);
  __m128i vec_yb = __lsx_vreplgr2vr_w(yuvconstants->kYBiasToRgb[0]);

  for (x = 0; x < len; x++) {
    vec_y = __lsx_vld(src_y, 0);
    vec_yl = __lsx_vilvl_b(vec_y, vec_y);
    y_ev = __lsx_vmulwev_w_hu_h(vec_yl, vec_yg);
    y_od = __lsx_vmulwod_w_hu_h(vec_yl, vec_yg);
    y_ev = __lsx_vsrai_w(y_ev, 16);
    y_od = __lsx_vsrai_w(y_od, 16);
    y_ev = __lsx_vadd_w(y_ev, vec_yb);
    y_od = __lsx_vadd_w(y_od, vec_yb);
    y_ev = __lsx_vsrai_w(y_ev, 6);
    y_od = __lsx_vsrai_w(y_od, 6);
    y_ev = __lsx_vclip255_w(y_ev);
    y_od = __lsx_vclip255_w(y_od);
    out0 = __lsx_vpackev_h(y_od, y_ev);
    temp0 = __lsx_vpackev_b(out0, out0);
    temp1 = __lsx_vpackev_b(alpha, out0);
    dst0 = __lsx_vilvl_h(temp1, temp0);
    dst1 = __lsx_vilvh_h(temp1, temp0);
    vec_yh = __lsx_vilvh_b(vec_y, vec_y);
    y_ev = __lsx_vmulwev_w_hu_h(vec_yh, vec_yg);
    y_od = __lsx_vmulwod_w_hu_h(vec_yh, vec_yg);
    y_ev = __lsx_vsrai_w(y_ev, 16);
    y_od = __lsx_vsrai_w(y_od, 16);
    y_ev = __lsx_vadd_w(y_ev, vec_yb);
    y_od = __lsx_vadd_w(y_od, vec_yb);
    y_ev = __lsx_vsrai_w(y_ev, 6);
    y_od = __lsx_vsrai_w(y_od, 6);
    y_ev = __lsx_vclip255_w(y_ev);
    y_od = __lsx_vclip255_w(y_od);
    out0 = __lsx_vpackev_h(y_od, y_ev);
    temp0 = __lsx_vpackev_b(out0, out0);
    temp1 = __lsx_vpackev_b(alpha, out0);
    dst2 = __lsx_vilvl_h(temp1, temp0);
    dst3 = __lsx_vilvh_h(temp1, temp0);
    __lsx_vst(dst0, dst_argb, 0);
    __lsx_vst(dst1, dst_argb, 16);
    __lsx_vst(dst2, dst_argb, 32);
    __lsx_vst(dst3, dst_argb, 48);
    dst_argb += 64;
    src_y += 16;
  }
}

void J400ToARGBRow_LSX(const uint8_t* src_y, uint8_t* dst_argb, int width) {
  int x;
  int len = width / 16;
  __m128i vec_y, dst0, dst1, dst2, dst3;
  __m128i tmp0, tmp1, tmp2, tmp3;
  __m128i alpha = __lsx_vldi(0xFF);

  for (x = 0; x < len; x++) {
    vec_y = __lsx_vld(src_y, 0);
    tmp0 = __lsx_vilvl_b(vec_y, vec_y);
    tmp1 = __lsx_vilvh_b(vec_y, vec_y);
    tmp2 = __lsx_vilvl_b(alpha, vec_y);
    tmp3 = __lsx_vilvh_b(alpha, vec_y);
    dst0 = __lsx_vilvl_h(tmp2, tmp0);
    dst1 = __lsx_vilvh_h(tmp2, tmp0);
    dst2 = __lsx_vilvl_h(tmp3, tmp1);
    dst3 = __lsx_vilvh_h(tmp3, tmp1);
    __lsx_vst(dst0, dst_argb, 0);
    __lsx_vst(dst1, dst_argb, 16);
    __lsx_vst(dst2, dst_argb, 32);
    __lsx_vst(dst3, dst_argb, 48);
    dst_argb += 64;
    src_y += 16;
  }
}

void YUY2ToARGBRow_LSX(const uint8_t* src_yuy2,
                       uint8_t* dst_argb,
                       const struct YuvConstants* yuvconstants,
                       int width) {
  int x;
  int len = width / 8;
  __m128i src0, vec_y, vec_vu;
  __m128i vec_vr, vec_ub, vec_vg, vec_ug, vec_yg, vec_yb;
  __m128i vec_vrub, vec_vgug;
  __m128i out_b, out_g, out_r;
  __m128i const_80 = __lsx_vldi(0x480);
  __m128i zero = __lsx_vldi(0);
  __m128i alpha = __lsx_vldi(0xFF);

  YUVTORGB_SETUP(yuvconstants, vec_vr, vec_ub, vec_vg, vec_ug, vec_yg, vec_yb);
  vec_vrub = __lsx_vilvl_h(vec_vr, vec_ub);
  vec_vgug = __lsx_vilvl_h(vec_vg, vec_ug);

  for (x = 0; x < len; x++) {
    src0 = __lsx_vld(src_yuy2, 0);
    vec_y = __lsx_vpickev_b(src0, src0);
    vec_vu = __lsx_vpickod_b(src0, src0);
    YUVTORGB(vec_y, vec_vu, vec_vrub, vec_vgug, vec_yg, vec_yb, out_b, out_g,
             out_r);
    STOREARGB(alpha, out_r, out_g, out_b, dst_argb);
    src_yuy2 += 16;
  }
}

void UYVYToARGBRow_LSX(const uint8_t* src_uyvy,
                       uint8_t* dst_argb,
                       const struct YuvConstants* yuvconstants,
                       int width) {
  int x;
  int len = width / 8;
  __m128i src0, vec_y, vec_vu;
  __m128i vec_vr, vec_ub, vec_vg, vec_ug, vec_yg, vec_yb;
  __m128i vec_vrub, vec_vgug;
  __m128i out_b, out_g, out_r;
  __m128i const_80 = __lsx_vldi(0x480);
  __m128i zero = __lsx_vldi(0);
  __m128i alpha = __lsx_vldi(0xFF);

  YUVTORGB_SETUP(yuvconstants, vec_vr, vec_ub, vec_vg, vec_ug, vec_yg, vec_yb);
  vec_vrub = __lsx_vilvl_h(vec_vr, vec_ub);
  vec_vgug = __lsx_vilvl_h(vec_vg, vec_ug);

  for (x = 0; x < len; x++) {
    src0 = __lsx_vld(src_uyvy, 0);
    vec_y = __lsx_vpickod_b(src0, src0);
    vec_vu = __lsx_vpickev_b(src0, src0);
    YUVTORGB(vec_y, vec_vu, vec_vrub, vec_vgug, vec_yg, vec_yb, out_b, out_g,
             out_r);
    STOREARGB(alpha, out_r, out_g, out_b, dst_argb);
    src_uyvy += 16;
  }
}

void InterpolateRow_LSX(uint8_t* dst_ptr,
                        const uint8_t* src_ptr,
                        ptrdiff_t src_stride,
                        int width,
                        int32_t source_y_fraction) {
  int x;
  int y1_fraction = source_y_fraction;
  int y0_fraction = 256 - y1_fraction;
  const uint8_t* nex_ptr = src_ptr + src_stride;
  uint16_t y_fractions;
  int len = width / 32;
  __m128i src0, src1, nex0, nex1;
  __m128i dst0, dst1, y_frac;
  __m128i tmp0, tmp1, tmp2, tmp3;
  __m128i const_128 = __lsx_vldi(0x480);

  if (y1_fraction == 0) {
    for (x = 0; x < len; x++) {
      DUP2_ARG2(__lsx_vld, src_ptr, 0, src_ptr, 16, src0, src1);
      __lsx_vst(src0, dst_ptr, 0);
      __lsx_vst(src1, dst_ptr, 16);
      src_ptr += 32;
      dst_ptr += 32;
    }
    return;
  }

  if (y1_fraction == 128) {
    for (x = 0; x < len; x++) {
      DUP2_ARG2(__lsx_vld, src_ptr, 0, src_ptr, 16, src0, src1);
      DUP2_ARG2(__lsx_vld, nex_ptr, 0, nex_ptr, 16, nex0, nex1);
      dst0 = __lsx_vavgr_bu(src0, nex0);
      dst1 = __lsx_vavgr_bu(src1, nex1);
      __lsx_vst(dst0, dst_ptr, 0);
      __lsx_vst(dst1, dst_ptr, 16);
      src_ptr += 32;
      nex_ptr += 32;
      dst_ptr += 32;
    }
    return;
  }

  y_fractions = (uint16_t)(y0_fraction + (y1_fraction << 8));
  y_frac = __lsx_vreplgr2vr_h(y_fractions);

  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lsx_vld, src_ptr, 0, src_ptr, 16, src0, src1);
    DUP2_ARG2(__lsx_vld, nex_ptr, 0, nex_ptr, 16, nex0, nex1);
    tmp0 = __lsx_vilvl_b(nex0, src0);
    tmp1 = __lsx_vilvh_b(nex0, src0);
    tmp2 = __lsx_vilvl_b(nex1, src1);
    tmp3 = __lsx_vilvh_b(nex1, src1);
    tmp0 = __lsx_vdp2add_h_bu(const_128, tmp0, y_frac);
    tmp1 = __lsx_vdp2add_h_bu(const_128, tmp1, y_frac);
    tmp2 = __lsx_vdp2add_h_bu(const_128, tmp2, y_frac);
    tmp3 = __lsx_vdp2add_h_bu(const_128, tmp3, y_frac);
    dst0 = __lsx_vsrlni_b_h(tmp1, tmp0, 8);
    dst1 = __lsx_vsrlni_b_h(tmp3, tmp2, 8);
    __lsx_vst(dst0, dst_ptr, 0);
    __lsx_vst(dst1, dst_ptr, 16);
    src_ptr += 32;
    nex_ptr += 32;
    dst_ptr += 32;
  }
}

void ARGBSetRow_LSX(uint8_t* dst_argb, uint32_t v32, int width) {
  int x;
  int len = width / 4;
  __m128i dst0 = __lsx_vreplgr2vr_w(v32);

  for (x = 0; x < len; x++) {
    __lsx_vst(dst0, dst_argb, 0);
    dst_argb += 16;
  }
}

void RAWToRGB24Row_LSX(const uint8_t* src_raw, uint8_t* dst_rgb24, int width) {
  int x;
  int len = width / 16;
  __m128i src0, src1, src2;
  __m128i dst0, dst1, dst2;
  __m128i shuf0 = {0x0708030405000102, 0x110C0D0E090A0B06};
  __m128i shuf1 = {0x1516171213140F10, 0x1F1E1B1C1D18191A};
  __m128i shuf2 = {0x090405060102031E, 0x0D0E0F0A0B0C0708};

  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lsx_vld, src_raw, 0, src_raw, 16, src0, src1);
    src2 = __lsx_vld(src_raw, 32);
    DUP2_ARG3(__lsx_vshuf_b, src1, src0, shuf0, src1, src0, shuf1, dst0, dst1);
    dst2 = __lsx_vshuf_b(src1, src2, shuf2);
    dst1 = __lsx_vinsgr2vr_b(dst1, src_raw[32], 0x0E);
    __lsx_vst(dst0, dst_rgb24, 0);
    __lsx_vst(dst1, dst_rgb24, 16);
    __lsx_vst(dst2, dst_rgb24, 32);
    dst_rgb24 += 48;
    src_raw += 48;
  }
}

void MergeUVRow_LSX(const uint8_t* src_u,
                    const uint8_t* src_v,
                    uint8_t* dst_uv,
                    int width) {
  int x;
  int len = width / 16;
  __m128i src0, src1, dst0, dst1;

  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lsx_vld, src_u, 0, src_v, 0, src0, src1);
    dst0 = __lsx_vilvl_b(src1, src0);
    dst1 = __lsx_vilvh_b(src1, src0);
    __lsx_vst(dst0, dst_uv, 0);
    __lsx_vst(dst1, dst_uv, 16);
    src_u += 16;
    src_v += 16;
    dst_uv += 32;
  }
}

void ARGBExtractAlphaRow_LSX(const uint8_t* src_argb,
                             uint8_t* dst_a,
                             int width) {
  int x;
  int len = width / 16;
  __m128i src0, src1, src2, src3, tmp0, tmp1, dst0;

  for (x = 0; x < len; x++) {
    DUP4_ARG2(__lsx_vld, src_argb, 0, src_argb, 16, src_argb, 32, src_argb, 48,
              src0, src1, src2, src3);
    tmp0 = __lsx_vpickod_b(src1, src0);
    tmp1 = __lsx_vpickod_b(src3, src2);
    dst0 = __lsx_vpickod_b(tmp1, tmp0);
    __lsx_vst(dst0, dst_a, 0);
    src_argb += 64;
    dst_a += 16;
  }
}

void ARGBBlendRow_LSX(const uint8_t* src_argb,
                      const uint8_t* src_argb1,
                      uint8_t* dst_argb,
                      int width) {
  int x;
  int len = width / 8;
  __m128i src0, src1, src2, src3;
  __m128i tmp0, tmp1, dst0, dst1;
  __m128i reg0, reg1, reg2, reg3;
  __m128i a0, a1, a2, a3;
  __m128i const_256 = __lsx_vldi(0x500);
  __m128i zero = __lsx_vldi(0);
  __m128i alpha = __lsx_vldi(0xFF);
  __m128i control = (__m128i)v2u64{0xFF000000FF000000, 0xFF000000FF000000};

  for (x = 0; x < len; x++) {
    DUP4_ARG2(__lsx_vld, src_argb, 0, src_argb, 16, src_argb1, 0, src_argb1, 16,
              src0, src1, src2, src3);
    tmp0 = __lsx_vshuf4i_b(src0, 0xFF);
    tmp1 = __lsx_vshuf4i_b(src1, 0xFF);
    a0 = __lsx_vilvl_b(zero, tmp0);
    a1 = __lsx_vilvh_b(zero, tmp0);
    a2 = __lsx_vilvl_b(zero, tmp1);
    a3 = __lsx_vilvh_b(zero, tmp1);
    reg0 = __lsx_vilvl_b(zero, src2);
    reg1 = __lsx_vilvh_b(zero, src2);
    reg2 = __lsx_vilvl_b(zero, src3);
    reg3 = __lsx_vilvh_b(zero, src3);
    DUP4_ARG2(__lsx_vsub_h, const_256, a0, const_256, a1, const_256, a2,
              const_256, a3, a0, a1, a2, a3);
    DUP4_ARG2(__lsx_vmul_h, a0, reg0, a1, reg1, a2, reg2, a3, reg3, reg0, reg1,
              reg2, reg3);
    DUP2_ARG3(__lsx_vsrani_b_h, reg1, reg0, 8, reg3, reg2, 8, dst0, dst1);
    dst0 = __lsx_vsadd_bu(dst0, src0);
    dst1 = __lsx_vsadd_bu(dst1, src1);
    dst0 = __lsx_vbitsel_v(dst0, alpha, control);
    dst1 = __lsx_vbitsel_v(dst1, alpha, control);
    __lsx_vst(dst0, dst_argb, 0);
    __lsx_vst(dst1, dst_argb, 16);
    src_argb += 32;
    src_argb1 += 32;
    dst_argb += 32;
  }
}

void ARGBQuantizeRow_LSX(uint8_t* dst_argb,
                         int scale,
                         int interval_size,
                         int interval_offset,
                         int width) {
  int x;
  int len = width / 16;
  __m128i src0, src1, src2, src3, dst0, dst1, dst2, dst3;
  __m128i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
  __m128i reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7;
  __m128i vec_size = __lsx_vreplgr2vr_b(interval_size);
  __m128i vec_offset = __lsx_vreplgr2vr_b(interval_offset);
  __m128i vec_scale = __lsx_vreplgr2vr_w(scale);
  __m128i zero = __lsx_vldi(0);
  __m128i control = (__m128i)v2u64{0xFF000000FF000000, 0xFF000000FF000000};

  for (x = 0; x < len; x++) {
    DUP4_ARG2(__lsx_vld, dst_argb, 0, dst_argb, 16, dst_argb, 32, dst_argb, 48,
              src0, src1, src2, src3);
    reg0 = __lsx_vilvl_b(zero, src0);
    reg1 = __lsx_vilvh_b(zero, src0);
    reg2 = __lsx_vilvl_b(zero, src1);
    reg3 = __lsx_vilvh_b(zero, src1);
    reg4 = __lsx_vilvl_b(zero, src2);
    reg5 = __lsx_vilvh_b(zero, src2);
    reg6 = __lsx_vilvl_b(zero, src3);
    reg7 = __lsx_vilvh_b(zero, src3);
    tmp0 = __lsx_vilvl_h(zero, reg0);
    tmp1 = __lsx_vilvh_h(zero, reg0);
    tmp2 = __lsx_vilvl_h(zero, reg1);
    tmp3 = __lsx_vilvh_h(zero, reg1);
    tmp4 = __lsx_vilvl_h(zero, reg2);
    tmp5 = __lsx_vilvh_h(zero, reg2);
    tmp6 = __lsx_vilvl_h(zero, reg3);
    tmp7 = __lsx_vilvh_h(zero, reg3);
    DUP4_ARG2(__lsx_vmul_w, tmp0, vec_scale, tmp1, vec_scale, tmp2, vec_scale,
              tmp3, vec_scale, tmp0, tmp1, tmp2, tmp3);
    DUP4_ARG2(__lsx_vmul_w, tmp4, vec_scale, tmp5, vec_scale, tmp6, vec_scale,
              tmp7, vec_scale, tmp4, tmp5, tmp6, tmp7);
    DUP4_ARG3(__lsx_vsrani_h_w, tmp1, tmp0, 16, tmp3, tmp2, 16, tmp5, tmp4, 16,
              tmp7, tmp6, 16, reg0, reg1, reg2, reg3);
    dst0 = __lsx_vpickev_b(reg1, reg0);
    dst1 = __lsx_vpickev_b(reg3, reg2);
    tmp0 = __lsx_vilvl_h(zero, reg4);
    tmp1 = __lsx_vilvh_h(zero, reg4);
    tmp2 = __lsx_vilvl_h(zero, reg5);
    tmp3 = __lsx_vilvh_h(zero, reg5);
    tmp4 = __lsx_vilvl_h(zero, reg6);
    tmp5 = __lsx_vilvh_h(zero, reg6);
    tmp6 = __lsx_vilvl_h(zero, reg7);
    tmp7 = __lsx_vilvh_h(zero, reg7);
    DUP4_ARG2(__lsx_vmul_w, tmp0, vec_scale, tmp1, vec_scale, tmp2, vec_scale,
              tmp3, vec_scale, tmp0, tmp1, tmp2, tmp3);
    DUP4_ARG2(__lsx_vmul_w, tmp4, vec_scale, tmp5, vec_scale, tmp6, vec_scale,
              tmp7, vec_scale, tmp4, tmp5, tmp6, tmp7);
    DUP4_ARG3(__lsx_vsrani_h_w, tmp1, tmp0, 16, tmp3, tmp2, 16, tmp5, tmp4, 16,
              tmp7, tmp6, 16, reg0, reg1, reg2, reg3);
    dst2 = __lsx_vpickev_b(reg1, reg0);
    dst3 = __lsx_vpickev_b(reg3, reg2);
    DUP4_ARG2(__lsx_vmul_b, dst0, vec_size, dst1, vec_size, dst2, vec_size,
              dst3, vec_size, dst0, dst1, dst2, dst3);
    DUP4_ARG2(__lsx_vadd_b, dst0, vec_offset, dst1, vec_offset, dst2,
              vec_offset, dst3, vec_offset, dst0, dst1, dst2, dst3);
    DUP4_ARG3(__lsx_vbitsel_v, dst0, src0, control, dst1, src1, control, dst2,
              src2, control, dst3, src3, control, dst0, dst1, dst2, dst3);
    __lsx_vst(dst0, dst_argb, 0);
    __lsx_vst(dst1, dst_argb, 16);
    __lsx_vst(dst2, dst_argb, 32);
    __lsx_vst(dst3, dst_argb, 48);
    dst_argb += 64;
  }
}

void ARGBColorMatrixRow_LSX(const uint8_t* src_argb,
                            uint8_t* dst_argb,
                            const int8_t* matrix_argb,
                            int width) {
  int x;
  int len = width / 8;
  __m128i src0, src1, tmp0, tmp1, dst0, dst1;
  __m128i tmp_b, tmp_g, tmp_r, tmp_a;
  __m128i reg_b, reg_g, reg_r, reg_a;
  __m128i matrix_b = __lsx_vldrepl_w(matrix_argb, 0);
  __m128i matrix_g = __lsx_vldrepl_w(matrix_argb, 4);
  __m128i matrix_r = __lsx_vldrepl_w(matrix_argb, 8);
  __m128i matrix_a = __lsx_vldrepl_w(matrix_argb, 12);

  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lsx_vld, src_argb, 0, src_argb, 16, src0, src1);
    DUP4_ARG2(__lsx_vdp2_h_bu_b, src0, matrix_b, src0, matrix_g, src0, matrix_r,
              src0, matrix_a, tmp_b, tmp_g, tmp_r, tmp_a);
    DUP4_ARG2(__lsx_vdp2_h_bu_b, src1, matrix_b, src1, matrix_g, src1, matrix_r,
              src1, matrix_a, reg_b, reg_g, reg_r, reg_a);
    DUP4_ARG2(__lsx_vhaddw_w_h, tmp_b, tmp_b, tmp_g, tmp_g, tmp_r, tmp_r, tmp_a,
              tmp_a, tmp_b, tmp_g, tmp_r, tmp_a);
    DUP4_ARG2(__lsx_vhaddw_w_h, reg_b, reg_b, reg_g, reg_g, reg_r, reg_r, reg_a,
              reg_a, reg_b, reg_g, reg_r, reg_a);
    DUP4_ARG2(__lsx_vsrai_w, tmp_b, 6, tmp_g, 6, tmp_r, 6, tmp_a, 6, tmp_b,
              tmp_g, tmp_r, tmp_a);
    DUP4_ARG2(__lsx_vsrai_w, reg_b, 6, reg_g, 6, reg_r, 6, reg_a, 6, reg_b,
              reg_g, reg_r, reg_a);
    DUP4_ARG1(__lsx_vclip255_w, tmp_b, tmp_g, tmp_r, tmp_a, tmp_b, tmp_g, tmp_r,
              tmp_a)
    DUP4_ARG1(__lsx_vclip255_w, reg_b, reg_g, reg_r, reg_a, reg_b, reg_g, reg_r,
              reg_a)
    DUP4_ARG2(__lsx_vpickev_h, reg_b, tmp_b, reg_g, tmp_g, reg_r, tmp_r, reg_a,
              tmp_a, tmp_b, tmp_g, tmp_r, tmp_a);
    tmp0 = __lsx_vpackev_b(tmp_g, tmp_b);
    tmp1 = __lsx_vpackev_b(tmp_a, tmp_r);
    dst0 = __lsx_vilvl_h(tmp1, tmp0);
    dst1 = __lsx_vilvh_h(tmp1, tmp0);
    __lsx_vst(dst0, dst_argb, 0);
    __lsx_vst(dst1, dst_argb, 16);
    src_argb += 32;
    dst_argb += 32;
  }
}

void SplitUVRow_LSX(const uint8_t* src_uv,
                    uint8_t* dst_u,
                    uint8_t* dst_v,
                    int width) {
  int x;
  int len = width / 32;
  __m128i src0, src1, src2, src3;
  __m128i dst0, dst1, dst2, dst3;

  for (x = 0; x < len; x++) {
    DUP4_ARG2(__lsx_vld, src_uv, 0, src_uv, 16, src_uv, 32, src_uv, 48, src0,
              src1, src2, src3);
    DUP2_ARG2(__lsx_vpickev_b, src1, src0, src3, src2, dst0, dst1);
    DUP2_ARG2(__lsx_vpickod_b, src1, src0, src3, src2, dst2, dst3);
    __lsx_vst(dst0, dst_u, 0);
    __lsx_vst(dst1, dst_u, 16);
    __lsx_vst(dst2, dst_v, 0);
    __lsx_vst(dst3, dst_v, 16);
    src_uv += 64;
    dst_u += 32;
    dst_v += 32;
  }
}

void SetRow_LSX(uint8_t* dst, uint8_t v8, int width) {
  int x;
  int len = width / 16;
  __m128i dst0 = __lsx_vreplgr2vr_b(v8);

  for (x = 0; x < len; x++) {
    __lsx_vst(dst0, dst, 0);
    dst += 16;
  }
}

void MirrorSplitUVRow_LSX(const uint8_t* src_uv,
                          uint8_t* dst_u,
                          uint8_t* dst_v,
                          int width) {
  int x;
  int len = width / 32;
  __m128i src0, src1, src2, src3;
  __m128i dst0, dst1, dst2, dst3;
  __m128i shuff0 = {0x10121416181A1C1E, 0x00020406080A0C0E};
  __m128i shuff1 = {0x11131517191B1D1F, 0x01030507090B0D0F};

  src_uv += (width << 1);
  for (x = 0; x < len; x++) {
    src_uv -= 64;
    DUP4_ARG2(__lsx_vld, src_uv, 0, src_uv, 16, src_uv, 32, src_uv, 48, src2,
              src3, src0, src1);
    DUP4_ARG3(__lsx_vshuf_b, src1, src0, shuff1, src3, src2, shuff1, src1, src0,
              shuff0, src3, src2, shuff0, dst0, dst1, dst2, dst3);
    __lsx_vst(dst0, dst_v, 0);
    __lsx_vst(dst1, dst_v, 16);
    __lsx_vst(dst2, dst_u, 0);
    __lsx_vst(dst3, dst_u, 16);
    dst_u += 32;
    dst_v += 32;
  }
}

void HalfFloatRow_LSX(const uint16_t* src,
                      uint16_t* dst,
                      float scale,
                      int width) {
  int x;
  int len = width / 32;
  float mult = 1.9259299444e-34f * scale;
  __m128i src0, src1, src2, src3, dst0, dst1, dst2, dst3;
  __m128i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
  __m128 reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7;
  __m128 vec_mult = (__m128)__lsx_vldrepl_w(&mult, 0);
  __m128i zero = __lsx_vldi(0);

  for (x = 0; x < len; x++) {
    DUP4_ARG2(__lsx_vld, src, 0, src, 16, src, 32, src, 48, src0, src1, src2,
              src3);
    DUP4_ARG2(__lsx_vilvl_h, zero, src0, zero, src1, zero, src2, zero, src3,
              tmp0, tmp2, tmp4, tmp6);
    DUP4_ARG2(__lsx_vilvh_h, zero, src0, zero, src1, zero, src2, zero, src3,
              tmp1, tmp3, tmp5, tmp7);
    DUP4_ARG1(__lsx_vffint_s_wu, tmp0, tmp2, tmp4, tmp6, reg0, reg2, reg4,
              reg6);
    DUP4_ARG1(__lsx_vffint_s_wu, tmp1, tmp3, tmp5, tmp7, reg1, reg3, reg5,
              reg7);
    DUP4_ARG2(__lsx_vfmul_s, reg0, vec_mult, reg1, vec_mult, reg2, vec_mult,
              reg3, vec_mult, reg0, reg1, reg2, reg3);
    DUP4_ARG2(__lsx_vfmul_s, reg4, vec_mult, reg5, vec_mult, reg6, vec_mult,
              reg7, vec_mult, reg4, reg5, reg6, reg7);
    DUP4_ARG2(__lsx_vsrli_w, (v4u32)reg0, 13, (v4u32)reg1, 13, (v4u32)reg2, 13,
              (v4u32)reg3, 13, tmp0, tmp1, tmp2, tmp3);
    DUP4_ARG2(__lsx_vsrli_w, (v4u32)reg4, 13, (v4u32)reg5, 13, (v4u32)reg6, 13,
              (v4u32)reg7, 13, tmp4, tmp5, tmp6, tmp7);
    DUP4_ARG2(__lsx_vpickev_h, tmp1, tmp0, tmp3, tmp2, tmp5, tmp4, tmp7, tmp6,
              dst0, dst1, dst2, dst3);
    __lsx_vst(dst0, dst, 0);
    __lsx_vst(dst1, dst, 16);
    __lsx_vst(dst2, dst, 32);
    __lsx_vst(dst3, dst, 48);
    src += 32;
    dst += 32;
  }
}

#ifndef RgbConstants
struct RgbConstants {
  uint8_t kRGBToY[4];
  uint16_t kAddY;
  uint16_t pad;
};
#define RgbConstants RgbConstants

// RGB to JPeg coefficients
// B * 0.1140 coefficient = 29
// G * 0.5870 coefficient = 150
// R * 0.2990 coefficient = 77
// Add 0.5 = 0x80
static const struct RgbConstants kRgb24JPEGConstants = {{29, 150, 77, 0},
                                                        128,
                                                        0};

static const struct RgbConstants kRawJPEGConstants = {{77, 150, 29, 0}, 128, 0};

// RGB to BT.601 coefficients
// B * 0.1016 coefficient = 25
// G * 0.5078 coefficient = 129
// R * 0.2578 coefficient = 66
// Add 16.5 = 0x1080

static const struct RgbConstants kRgb24I601Constants = {{25, 129, 66, 0},
                                                        0x1080,
                                                        0};

static const struct RgbConstants kRawI601Constants = {{66, 129, 25, 0},
                                                      0x1080,
                                                      0};
#endif  // RgbConstants

// ARGB expects first 3 values to contain RGB and 4th value is ignored.
static void ARGBToYMatrixRow_LSX(const uint8_t* src_argb,
                                 uint8_t* dst_y,
                                 int width,
                                 const struct RgbConstants* rgbconstants) {
  asm volatile(
      "vldrepl.b      $vr0,  %3,    0             \n\t"  // load rgbconstants
      "vldrepl.b      $vr1,  %3,    1             \n\t"  // load rgbconstants
      "vldrepl.b      $vr2,  %3,    2             \n\t"  // load rgbconstants
      "vldrepl.h      $vr3,  %3,    4             \n\t"  // load rgbconstants
      "1:                                         \n\t"
      "vld            $vr4,  %0,    0             \n\t"
      "vld            $vr5,  %0,    16            \n\t"
      "vld            $vr6,  %0,    32            \n\t"
      "vld            $vr7,  %0,    48            \n\t"  // load 16 pixels of
                                                         // ARGB
      "vor.v          $vr12, $vr3,  $vr3          \n\t"
      "vor.v          $vr13, $vr3,  $vr3          \n\t"
      "addi.d         %2,    %2,    -16           \n\t"  // 16 processed per
                                                         // loop.
      "vpickev.b      $vr8,  $vr5,  $vr4          \n\t"  // BR
      "vpickev.b      $vr10, $vr7,  $vr6          \n\t"
      "vpickod.b      $vr9,  $vr5,  $vr4          \n\t"  // GA
      "vpickod.b      $vr11, $vr7,  $vr6          \n\t"
      "vmaddwev.h.bu  $vr12, $vr8,  $vr0          \n\t"  // B
      "vmaddwev.h.bu  $vr13, $vr10, $vr0          \n\t"
      "vmaddwev.h.bu  $vr12, $vr9,  $vr1          \n\t"  // G
      "vmaddwev.h.bu  $vr13, $vr11, $vr1          \n\t"
      "vmaddwod.h.bu  $vr12, $vr8,  $vr2          \n\t"  // R
      "vmaddwod.h.bu  $vr13, $vr10, $vr2          \n\t"
      "addi.d         %0,    %0,    64            \n\t"
      "vpickod.b      $vr10, $vr13, $vr12         \n\t"
      "vst            $vr10, %1,    0             \n\t"
      "addi.d         %1,    %1,    16            \n\t"
      "bnez           %2,    1b                   \n\t"
      : "+&r"(src_argb),  // %0
        "+&r"(dst_y),     // %1
        "+&r"(width)      // %2
      : "r"(rgbconstants)
      : "memory");
}

void ARGBToYRow_LSX(const uint8_t* src_argb, uint8_t* dst_y, int width) {
  ARGBToYMatrixRow_LSX(src_argb, dst_y, width, &kRgb24I601Constants);
}

void ARGBToYJRow_LSX(const uint8_t* src_argb, uint8_t* dst_yj, int width) {
  ARGBToYMatrixRow_LSX(src_argb, dst_yj, width, &kRgb24JPEGConstants);
}

void ABGRToYRow_LSX(const uint8_t* src_abgr, uint8_t* dst_y, int width) {
  ARGBToYMatrixRow_LSX(src_abgr, dst_y, width, &kRawI601Constants);
}

void ABGRToYJRow_LSX(const uint8_t* src_abgr, uint8_t* dst_yj, int width) {
  ARGBToYMatrixRow_LSX(src_abgr, dst_yj, width, &kRawJPEGConstants);
}

// RGBA expects first value to be A and ignored, then 3 values to contain RGB.
// Same code as ARGB, except the LD4
static void RGBAToYMatrixRow_LSX(const uint8_t* src_rgba,
                                 uint8_t* dst_y,
                                 int width,
                                 const struct RgbConstants* rgbconstants) {
  asm volatile(
      "vldrepl.b      $vr0,  %3,    0             \n\t"  // load rgbconstants
      "vldrepl.b      $vr1,  %3,    1             \n\t"  // load rgbconstants
      "vldrepl.b      $vr2,  %3,    2             \n\t"  // load rgbconstants
      "vldrepl.h      $vr3,  %3,    4             \n\t"  // load rgbconstants
      "1:                                         \n\t"
      "vld            $vr4,  %0,    0             \n\t"
      "vld            $vr5,  %0,    16            \n\t"
      "vld            $vr6,  %0,    32            \n\t"
      "vld            $vr7,  %0,    48            \n\t"  // load 16 pixels of
                                                         // RGBA
      "vor.v          $vr12, $vr3,  $vr3          \n\t"
      "vor.v          $vr13, $vr3,  $vr3          \n\t"
      "addi.d         %2,    %2,    -16           \n\t"  // 16 processed per
                                                         // loop.
      "vpickev.b      $vr8,  $vr5,  $vr4          \n\t"  // AG
      "vpickev.b      $vr10, $vr7,  $vr6          \n\t"
      "vpickod.b      $vr9,  $vr5,  $vr4          \n\t"  // BR
      "vpickod.b      $vr11, $vr7,  $vr6          \n\t"
      "vmaddwev.h.bu  $vr12, $vr9,  $vr0          \n\t"  // B
      "vmaddwev.h.bu  $vr13, $vr11, $vr0          \n\t"
      "vmaddwod.h.bu  $vr12, $vr8,  $vr1          \n\t"  // G
      "vmaddwod.h.bu  $vr13, $vr10, $vr1          \n\t"
      "vmaddwod.h.bu  $vr12, $vr9,  $vr2          \n\t"  // R
      "vmaddwod.h.bu  $vr13, $vr11, $vr2          \n\t"
      "addi.d         %0,    %0,    64            \n\t"
      "vpickod.b      $vr10, $vr13, $vr12         \n\t"
      "vst            $vr10, %1,    0             \n\t"
      "addi.d         %1,    %1,    16            \n\t"
      "bnez           %2,    1b                   \n\t"
      : "+&r"(src_rgba),  // %0
        "+&r"(dst_y),     // %1
        "+&r"(width)      // %2
      : "r"(rgbconstants)
      : "memory");
}

void RGBAToYRow_LSX(const uint8_t* src_rgba, uint8_t* dst_y, int width) {
  RGBAToYMatrixRow_LSX(src_rgba, dst_y, width, &kRgb24I601Constants);
}

void RGBAToYJRow_LSX(const uint8_t* src_rgba, uint8_t* dst_yj, int width) {
  RGBAToYMatrixRow_LSX(src_rgba, dst_yj, width, &kRgb24JPEGConstants);
}

void BGRAToYRow_LSX(const uint8_t* src_bgra, uint8_t* dst_y, int width) {
  RGBAToYMatrixRow_LSX(src_bgra, dst_y, width, &kRawI601Constants);
}

static void RGBToYMatrixRow_LSX(const uint8_t* src_rgba,
                                uint8_t* dst_y,
                                int width,
                                const struct RgbConstants* rgbconstants) {
  int8_t shuff[64] = {0,  2,  3,  5,  6,  8,  9,  11, 12, 14, 15, 17, 18,
                      20, 21, 23, 24, 26, 27, 29, 30, 0,  1,  3,  4,  6,
                      7,  9,  10, 12, 13, 15, 1,  0,  4,  0,  7,  0,  10,
                      0,  13, 0,  16, 0,  19, 0,  22, 0,  25, 0,  28, 0,
                      31, 0,  2,  0,  5,  0,  8,  0,  11, 0,  14, 0};
  asm volatile(
      "vldrepl.b      $vr0,  %3,    0             \n\t"  // load rgbconstants
      "vldrepl.b      $vr1,  %3,    1             \n\t"  // load rgbconstants
      "vldrepl.b      $vr2,  %3,    2             \n\t"  // load rgbconstants
      "vldrepl.h      $vr3,  %3,    4             \n\t"  // load rgbconstants
      "vld            $vr4,  %4,    0             \n\t"  // load shuff
      "vld            $vr5,  %4,    16            \n\t"
      "vld            $vr6,  %4,    32            \n\t"
      "vld            $vr7,  %4,    48            \n\t"
      "1:                                         \n\t"
      "vld            $vr8,  %0,    0             \n\t"
      "vld            $vr9,  %0,    16            \n\t"
      "vld            $vr10, %0,    32            \n\t"  // load 16 pixels of
                                                         // RGB
      "vor.v          $vr12, $vr3,  $vr3          \n\t"
      "vor.v          $vr13, $vr3,  $vr3          \n\t"
      "addi.d         %2,    %2,    -16           \n\t"  // 16 processed per
                                                         // loop.
      "vshuf.b        $vr14, $vr9,  $vr8,  $vr4   \n\t"
      "vshuf.b        $vr15, $vr9,  $vr10, $vr5   \n\t"
      "vshuf.b        $vr16, $vr9,  $vr8,  $vr6   \n\t"
      "vshuf.b        $vr17, $vr9,  $vr10, $vr7   \n\t"
      "vmaddwev.h.bu  $vr12, $vr16, $vr1          \n\t"  // G
      "vmaddwev.h.bu  $vr13, $vr17, $vr1          \n\t"
      "vmaddwev.h.bu  $vr12, $vr14, $vr0          \n\t"  // B
      "vmaddwev.h.bu  $vr13, $vr15, $vr0          \n\t"
      "vmaddwod.h.bu  $vr12, $vr14, $vr2          \n\t"  // R
      "vmaddwod.h.bu  $vr13, $vr15, $vr2          \n\t"
      "addi.d         %0,    %0,    48            \n\t"
      "vpickod.b      $vr10, $vr13, $vr12         \n\t"
      "vst            $vr10, %1,    0             \n\t"
      "addi.d         %1,    %1,    16            \n\t"
      "bnez           %2,    1b                   \n\t"
      : "+&r"(src_rgba),    // %0
        "+&r"(dst_y),       // %1
        "+&r"(width)        // %2
      : "r"(rgbconstants),  // %3
        "r"(shuff)          // %4
      : "memory");
}

void RGB24ToYJRow_LSX(const uint8_t* src_rgb24, uint8_t* dst_yj, int width) {
  RGBToYMatrixRow_LSX(src_rgb24, dst_yj, width, &kRgb24JPEGConstants);
}

void RAWToYJRow_LSX(const uint8_t* src_raw, uint8_t* dst_yj, int width) {
  RGBToYMatrixRow_LSX(src_raw, dst_yj, width, &kRawJPEGConstants);
}

void RGB24ToYRow_LSX(const uint8_t* src_rgb24, uint8_t* dst_y, int width) {
  RGBToYMatrixRow_LSX(src_rgb24, dst_y, width, &kRgb24I601Constants);
}

void RAWToYRow_LSX(const uint8_t* src_raw, uint8_t* dst_y, int width) {
  RGBToYMatrixRow_LSX(src_raw, dst_y, width, &kRawI601Constants);
}

// undef for unified sources build
#undef YUVTORGB_SETUP
#undef READYUV422_D
#undef READYUV422
#undef YUVTORGB_D
#undef YUVTORGB
#undef I444TORGB
#undef STOREARGB_D
#undef STOREARGB
#undef RGBTOUV

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

#endif  // !defined(LIBYUV_DISABLE_LSX) && defined(__loongarch_sx)
