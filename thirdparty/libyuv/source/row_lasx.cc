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

#if !defined(LIBYUV_DISABLE_LASX) && defined(__loongarch_asx)
#include "libyuv/loongson_intrinsics.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

#define ALPHA_VAL (-1)

// Fill YUV -> RGB conversion constants into vectors
#define YUVTORGB_SETUP(yuvconst, ub, vr, ug, vg, yg, yb) \
  {                                                      \
    ub = __lasx_xvreplgr2vr_h(yuvconst->kUVToB[0]);      \
    vr = __lasx_xvreplgr2vr_h(yuvconst->kUVToR[1]);      \
    ug = __lasx_xvreplgr2vr_h(yuvconst->kUVToG[0]);      \
    vg = __lasx_xvreplgr2vr_h(yuvconst->kUVToG[1]);      \
    yg = __lasx_xvreplgr2vr_h(yuvconst->kYToRgb[0]);     \
    yb = __lasx_xvreplgr2vr_w(yuvconst->kYBiasToRgb[0]); \
  }

// Load 32 YUV422 pixel data
#define READYUV422_D(psrc_y, psrc_u, psrc_v, out_y, uv_l, uv_h) \
  {                                                             \
    __m256i temp0, temp1;                                       \
                                                                \
    DUP2_ARG2(__lasx_xvld, psrc_y, 0, psrc_u, 0, out_y, temp0); \
    temp1 = __lasx_xvld(psrc_v, 0);                             \
    temp0 = __lasx_xvsub_b(temp0, const_0x80);                  \
    temp1 = __lasx_xvsub_b(temp1, const_0x80);                  \
    temp0 = __lasx_vext2xv_h_b(temp0);                          \
    temp1 = __lasx_vext2xv_h_b(temp1);                          \
    uv_l = __lasx_xvilvl_h(temp0, temp1);                       \
    uv_h = __lasx_xvilvh_h(temp0, temp1);                       \
  }

// Load 16 YUV422 pixel data
#define READYUV422(psrc_y, psrc_u, psrc_v, out_y, uv) \
  {                                                   \
    __m256i temp0, temp1;                             \
                                                      \
    out_y = __lasx_xvld(psrc_y, 0);                   \
    temp0 = __lasx_xvldrepl_d(psrc_u, 0);             \
    temp1 = __lasx_xvldrepl_d(psrc_v, 0);             \
    uv = __lasx_xvilvl_b(temp0, temp1);               \
    uv = __lasx_xvsub_b(uv, const_0x80);              \
    uv = __lasx_vext2xv_h_b(uv);                      \
  }

// Convert 16 pixels of YUV420 to RGB.
#define YUVTORGB_D(in_y, in_uvl, in_uvh, ubvr, ugvg, yg, yb, b_l, b_h, g_l,   \
                   g_h, r_l, r_h)                                             \
  {                                                                           \
    __m256i u_l, u_h, v_l, v_h;                                               \
    __m256i yl_ev, yl_od, yh_ev, yh_od;                                       \
    __m256i temp0, temp1, temp2, temp3;                                       \
                                                                              \
    temp0 = __lasx_xvilvl_b(in_y, in_y);                                      \
    temp1 = __lasx_xvilvh_b(in_y, in_y);                                      \
    yl_ev = __lasx_xvmulwev_w_hu_h(temp0, yg);                                \
    yl_od = __lasx_xvmulwod_w_hu_h(temp0, yg);                                \
    yh_ev = __lasx_xvmulwev_w_hu_h(temp1, yg);                                \
    yh_od = __lasx_xvmulwod_w_hu_h(temp1, yg);                                \
    DUP4_ARG2(__lasx_xvsrai_w, yl_ev, 16, yl_od, 16, yh_ev, 16, yh_od, 16,    \
              yl_ev, yl_od, yh_ev, yh_od);                                    \
    yl_ev = __lasx_xvadd_w(yl_ev, yb);                                        \
    yl_od = __lasx_xvadd_w(yl_od, yb);                                        \
    yh_ev = __lasx_xvadd_w(yh_ev, yb);                                        \
    yh_od = __lasx_xvadd_w(yh_od, yb);                                        \
    v_l = __lasx_xvmulwev_w_h(in_uvl, ubvr);                                  \
    u_l = __lasx_xvmulwod_w_h(in_uvl, ubvr);                                  \
    v_h = __lasx_xvmulwev_w_h(in_uvh, ubvr);                                  \
    u_h = __lasx_xvmulwod_w_h(in_uvh, ubvr);                                  \
    temp0 = __lasx_xvadd_w(yl_ev, u_l);                                       \
    temp1 = __lasx_xvadd_w(yl_od, u_l);                                       \
    temp2 = __lasx_xvadd_w(yh_ev, u_h);                                       \
    temp3 = __lasx_xvadd_w(yh_od, u_h);                                       \
    DUP4_ARG2(__lasx_xvsrai_w, temp0, 6, temp1, 6, temp2, 6, temp3, 6, temp0, \
              temp1, temp2, temp3);                                           \
    DUP4_ARG1(__lasx_xvclip255_w, temp0, temp1, temp2, temp3, temp0, temp1,   \
              temp2, temp3);                                                  \
    b_l = __lasx_xvpackev_h(temp1, temp0);                                    \
    b_h = __lasx_xvpackev_h(temp3, temp2);                                    \
    temp0 = __lasx_xvadd_w(yl_ev, v_l);                                       \
    temp1 = __lasx_xvadd_w(yl_od, v_l);                                       \
    temp2 = __lasx_xvadd_w(yh_ev, v_h);                                       \
    temp3 = __lasx_xvadd_w(yh_od, v_h);                                       \
    DUP4_ARG2(__lasx_xvsrai_w, temp0, 6, temp1, 6, temp2, 6, temp3, 6, temp0, \
              temp1, temp2, temp3);                                           \
    DUP4_ARG1(__lasx_xvclip255_w, temp0, temp1, temp2, temp3, temp0, temp1,   \
              temp2, temp3);                                                  \
    r_l = __lasx_xvpackev_h(temp1, temp0);                                    \
    r_h = __lasx_xvpackev_h(temp3, temp2);                                    \
    DUP2_ARG2(__lasx_xvdp2_w_h, in_uvl, ugvg, in_uvh, ugvg, u_l, u_h);        \
    temp0 = __lasx_xvsub_w(yl_ev, u_l);                                       \
    temp1 = __lasx_xvsub_w(yl_od, u_l);                                       \
    temp2 = __lasx_xvsub_w(yh_ev, u_h);                                       \
    temp3 = __lasx_xvsub_w(yh_od, u_h);                                       \
    DUP4_ARG2(__lasx_xvsrai_w, temp0, 6, temp1, 6, temp2, 6, temp3, 6, temp0, \
              temp1, temp2, temp3);                                           \
    DUP4_ARG1(__lasx_xvclip255_w, temp0, temp1, temp2, temp3, temp0, temp1,   \
              temp2, temp3);                                                  \
    g_l = __lasx_xvpackev_h(temp1, temp0);                                    \
    g_h = __lasx_xvpackev_h(temp3, temp2);                                    \
  }

// Convert 8 pixels of YUV420 to RGB.
#define YUVTORGB(in_y, in_uv, ubvr, ugvg, yg, yb, out_b, out_g, out_r) \
  {                                                                    \
    __m256i u_l, v_l, yl_ev, yl_od;                                    \
    __m256i temp0, temp1;                                              \
                                                                       \
    in_y = __lasx_xvpermi_d(in_y, 0xD8);                               \
    temp0 = __lasx_xvilvl_b(in_y, in_y);                               \
    yl_ev = __lasx_xvmulwev_w_hu_h(temp0, yg);                         \
    yl_od = __lasx_xvmulwod_w_hu_h(temp0, yg);                         \
    DUP2_ARG2(__lasx_xvsrai_w, yl_ev, 16, yl_od, 16, yl_ev, yl_od);    \
    yl_ev = __lasx_xvadd_w(yl_ev, yb);                                 \
    yl_od = __lasx_xvadd_w(yl_od, yb);                                 \
    v_l = __lasx_xvmulwev_w_h(in_uv, ubvr);                            \
    u_l = __lasx_xvmulwod_w_h(in_uv, ubvr);                            \
    temp0 = __lasx_xvadd_w(yl_ev, u_l);                                \
    temp1 = __lasx_xvadd_w(yl_od, u_l);                                \
    DUP2_ARG2(__lasx_xvsrai_w, temp0, 6, temp1, 6, temp0, temp1);      \
    DUP2_ARG1(__lasx_xvclip255_w, temp0, temp1, temp0, temp1);         \
    out_b = __lasx_xvpackev_h(temp1, temp0);                           \
    temp0 = __lasx_xvadd_w(yl_ev, v_l);                                \
    temp1 = __lasx_xvadd_w(yl_od, v_l);                                \
    DUP2_ARG2(__lasx_xvsrai_w, temp0, 6, temp1, 6, temp0, temp1);      \
    DUP2_ARG1(__lasx_xvclip255_w, temp0, temp1, temp0, temp1);         \
    out_r = __lasx_xvpackev_h(temp1, temp0);                           \
    u_l = __lasx_xvdp2_w_h(in_uv, ugvg);                               \
    temp0 = __lasx_xvsub_w(yl_ev, u_l);                                \
    temp1 = __lasx_xvsub_w(yl_od, u_l);                                \
    DUP2_ARG2(__lasx_xvsrai_w, temp0, 6, temp1, 6, temp0, temp1);      \
    DUP2_ARG1(__lasx_xvclip255_w, temp0, temp1, temp0, temp1);         \
    out_g = __lasx_xvpackev_h(temp1, temp0);                           \
  }

// Pack and Store 16 ARGB values.
#define STOREARGB_D(a_l, a_h, r_l, r_h, g_l, g_h, b_l, b_h, pdst_argb) \
  {                                                                    \
    __m256i temp0, temp1, temp2, temp3;                                \
                                                                       \
    temp0 = __lasx_xvpackev_b(g_l, b_l);                               \
    temp1 = __lasx_xvpackev_b(a_l, r_l);                               \
    temp2 = __lasx_xvpackev_b(g_h, b_h);                               \
    temp3 = __lasx_xvpackev_b(a_h, r_h);                               \
    r_l = __lasx_xvilvl_h(temp1, temp0);                               \
    r_h = __lasx_xvilvh_h(temp1, temp0);                               \
    g_l = __lasx_xvilvl_h(temp3, temp2);                               \
    g_h = __lasx_xvilvh_h(temp3, temp2);                               \
    temp0 = __lasx_xvpermi_q(r_h, r_l, 0x20);                          \
    temp1 = __lasx_xvpermi_q(g_h, g_l, 0x20);                          \
    temp2 = __lasx_xvpermi_q(r_h, r_l, 0x31);                          \
    temp3 = __lasx_xvpermi_q(g_h, g_l, 0x31);                          \
    __lasx_xvst(temp0, pdst_argb, 0);                                  \
    __lasx_xvst(temp1, pdst_argb, 32);                                 \
    __lasx_xvst(temp2, pdst_argb, 64);                                 \
    __lasx_xvst(temp3, pdst_argb, 96);                                 \
    pdst_argb += 128;                                                  \
  }

// Pack and Store 8 ARGB values.
#define STOREARGB(in_a, in_r, in_g, in_b, pdst_argb) \
  {                                                  \
    __m256i temp0, temp1, temp2, temp3;              \
                                                     \
    temp0 = __lasx_xvpackev_b(in_g, in_b);           \
    temp1 = __lasx_xvpackev_b(in_a, in_r);           \
    temp2 = __lasx_xvilvl_h(temp1, temp0);           \
    temp3 = __lasx_xvilvh_h(temp1, temp0);           \
    temp0 = __lasx_xvpermi_q(temp3, temp2, 0x20);    \
    temp1 = __lasx_xvpermi_q(temp3, temp2, 0x31);    \
    __lasx_xvst(temp0, pdst_argb, 0);                \
    __lasx_xvst(temp1, pdst_argb, 32);               \
    pdst_argb += 64;                                 \
  }

#define RGBTOUV(_tmpb, _tmpg, _tmpr, _nexb, _nexg, _nexr, _reg0, _reg1) \
  {                                                                     \
    __m256i _tmp0, _tmp1, _tmp2, _tmp3;                                 \
    _tmp0 = __lasx_xvaddwev_h_bu(_tmpb, _nexb);                         \
    _tmp1 = __lasx_xvaddwod_h_bu(_tmpb, _nexb);                         \
    _tmp2 = __lasx_xvaddwev_h_bu(_tmpg, _nexg);                         \
    _tmp3 = __lasx_xvaddwod_h_bu(_tmpg, _nexg);                         \
    _reg0 = __lasx_xvaddwev_h_bu(_tmpr, _nexr);                         \
    _reg1 = __lasx_xvaddwod_h_bu(_tmpr, _nexr);                         \
    _tmpb = __lasx_xvavgr_hu(_tmp0, _tmp1);                             \
    _tmpg = __lasx_xvavgr_hu(_tmp2, _tmp3);                             \
    _tmpr = __lasx_xvavgr_hu(_reg0, _reg1);                             \
    _reg0 = __lasx_xvmadd_h(const_8080, const_112, _tmpb);              \
    _reg1 = __lasx_xvmadd_h(const_8080, const_112, _tmpr);              \
    _reg0 = __lasx_xvmsub_h(_reg0, const_74, _tmpg);                    \
    _reg1 = __lasx_xvmsub_h(_reg1, const_94, _tmpg);                    \
    _reg0 = __lasx_xvmsub_h(_reg0, const_38, _tmpr);                    \
    _reg1 = __lasx_xvmsub_h(_reg1, const_18, _tmpb);                    \
  }

void MirrorRow_LASX(const uint8_t* src, uint8_t* dst, int width) {
  int x;
  int len = width / 64;
  __m256i src0, src1;
  __m256i shuffler = {0x08090A0B0C0D0E0F, 0x0001020304050607,
                      0x08090A0B0C0D0E0F, 0x0001020304050607};
  src += width - 64;
  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lasx_xvld, src, 0, src, 32, src0, src1);
    DUP2_ARG3(__lasx_xvshuf_b, src0, src0, shuffler, src1, src1, shuffler, src0,
              src1);
    src0 = __lasx_xvpermi_q(src0, src0, 0x01);
    src1 = __lasx_xvpermi_q(src1, src1, 0x01);
    __lasx_xvst(src1, dst, 0);
    __lasx_xvst(src0, dst, 32);
    dst += 64;
    src -= 64;
  }
}

void MirrorUVRow_LASX(const uint8_t* src_uv, uint8_t* dst_uv, int width) {
  int x;
  int len = width / 16;
  __m256i src, dst;
  __m256i shuffler = {0x0004000500060007, 0x0000000100020003,
                      0x0004000500060007, 0x0000000100020003};

  src_uv += (width - 16) << 1;
  for (x = 0; x < len; x++) {
    src = __lasx_xvld(src_uv, 0);
    dst = __lasx_xvshuf_h(shuffler, src, src);
    dst = __lasx_xvpermi_q(dst, dst, 0x01);
    __lasx_xvst(dst, dst_uv, 0);
    src_uv -= 32;
    dst_uv += 32;
  }
}

void ARGBMirrorRow_LASX(const uint8_t* src, uint8_t* dst, int width) {
  int x;
  int len = width / 16;
  __m256i src0, src1;
  __m256i dst0, dst1;
  __m256i shuffler = {0x0B0A09080F0E0D0C, 0x0302010007060504,
                      0x0B0A09080F0E0D0C, 0x0302010007060504};
  src += (width * 4) - 64;
  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lasx_xvld, src, 0, src, 32, src0, src1);
    DUP2_ARG3(__lasx_xvshuf_b, src0, src0, shuffler, src1, src1, shuffler, src0,
              src1);
    dst1 = __lasx_xvpermi_q(src0, src0, 0x01);
    dst0 = __lasx_xvpermi_q(src1, src1, 0x01);
    __lasx_xvst(dst0, dst, 0);
    __lasx_xvst(dst1, dst, 32);
    dst += 64;
    src -= 64;
  }
}

void I422ToYUY2Row_LASX(const uint8_t* src_y,
                        const uint8_t* src_u,
                        const uint8_t* src_v,
                        uint8_t* dst_yuy2,
                        int width) {
  int x;
  int len = width / 32;
  __m256i src_u0, src_v0, src_y0, vec_uv0;
  __m256i vec_yuy2_0, vec_yuy2_1;
  __m256i dst_yuy2_0, dst_yuy2_1;

  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lasx_xvld, src_u, 0, src_v, 0, src_u0, src_v0);
    src_y0 = __lasx_xvld(src_y, 0);
    src_u0 = __lasx_xvpermi_d(src_u0, 0xD8);
    src_v0 = __lasx_xvpermi_d(src_v0, 0xD8);
    vec_uv0 = __lasx_xvilvl_b(src_v0, src_u0);
    vec_yuy2_0 = __lasx_xvilvl_b(vec_uv0, src_y0);
    vec_yuy2_1 = __lasx_xvilvh_b(vec_uv0, src_y0);
    dst_yuy2_0 = __lasx_xvpermi_q(vec_yuy2_1, vec_yuy2_0, 0x20);
    dst_yuy2_1 = __lasx_xvpermi_q(vec_yuy2_1, vec_yuy2_0, 0x31);
    __lasx_xvst(dst_yuy2_0, dst_yuy2, 0);
    __lasx_xvst(dst_yuy2_1, dst_yuy2, 32);
    src_u += 16;
    src_v += 16;
    src_y += 32;
    dst_yuy2 += 64;
  }
}

void I422ToUYVYRow_LASX(const uint8_t* src_y,
                        const uint8_t* src_u,
                        const uint8_t* src_v,
                        uint8_t* dst_uyvy,
                        int width) {
  int x;
  int len = width / 32;
  __m256i src_u0, src_v0, src_y0, vec_uv0;
  __m256i vec_uyvy0, vec_uyvy1;
  __m256i dst_uyvy0, dst_uyvy1;

  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lasx_xvld, src_u, 0, src_v, 0, src_u0, src_v0);
    src_y0 = __lasx_xvld(src_y, 0);
    src_u0 = __lasx_xvpermi_d(src_u0, 0xD8);
    src_v0 = __lasx_xvpermi_d(src_v0, 0xD8);
    vec_uv0 = __lasx_xvilvl_b(src_v0, src_u0);
    vec_uyvy0 = __lasx_xvilvl_b(src_y0, vec_uv0);
    vec_uyvy1 = __lasx_xvilvh_b(src_y0, vec_uv0);
    dst_uyvy0 = __lasx_xvpermi_q(vec_uyvy1, vec_uyvy0, 0x20);
    dst_uyvy1 = __lasx_xvpermi_q(vec_uyvy1, vec_uyvy0, 0x31);
    __lasx_xvst(dst_uyvy0, dst_uyvy, 0);
    __lasx_xvst(dst_uyvy1, dst_uyvy, 32);
    src_u += 16;
    src_v += 16;
    src_y += 32;
    dst_uyvy += 64;
  }
}

void I422ToARGBRow_LASX(const uint8_t* src_y,
                        const uint8_t* src_u,
                        const uint8_t* src_v,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  int x;
  int len = width / 32;
  __m256i vec_yb, vec_yg, vec_ub, vec_ug, vec_vr, vec_vg;
  __m256i vec_ubvr, vec_ugvg;
  __m256i alpha = __lasx_xvldi(0xFF);
  __m256i const_0x80 = __lasx_xvldi(0x80);

  YUVTORGB_SETUP(yuvconstants, vec_ub, vec_vr, vec_ug, vec_vg, vec_yg, vec_yb);
  vec_ubvr = __lasx_xvilvl_h(vec_ub, vec_vr);
  vec_ugvg = __lasx_xvilvl_h(vec_ug, vec_vg);

  for (x = 0; x < len; x++) {
    __m256i y, uv_l, uv_h, b_l, b_h, g_l, g_h, r_l, r_h;

    READYUV422_D(src_y, src_u, src_v, y, uv_l, uv_h);
    YUVTORGB_D(y, uv_l, uv_h, vec_ubvr, vec_ugvg, vec_yg, vec_yb, b_l, b_h, g_l,
               g_h, r_l, r_h);
    STOREARGB_D(alpha, alpha, r_l, r_h, g_l, g_h, b_l, b_h, dst_argb);
    src_y += 32;
    src_u += 16;
    src_v += 16;
  }
}

void I422ToRGBARow_LASX(const uint8_t* src_y,
                        const uint8_t* src_u,
                        const uint8_t* src_v,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  int x;
  int len = width / 32;
  __m256i vec_yb, vec_yg, vec_ub, vec_vr, vec_ug, vec_vg;
  __m256i vec_ubvr, vec_ugvg;
  __m256i alpha = __lasx_xvldi(0xFF);
  __m256i const_0x80 = __lasx_xvldi(0x80);

  YUVTORGB_SETUP(yuvconstants, vec_ub, vec_vr, vec_ug, vec_vg, vec_yg, vec_yb);
  vec_ubvr = __lasx_xvilvl_h(vec_ub, vec_vr);
  vec_ugvg = __lasx_xvilvl_h(vec_ug, vec_vg);

  for (x = 0; x < len; x++) {
    __m256i y, uv_l, uv_h, b_l, b_h, g_l, g_h, r_l, r_h;

    READYUV422_D(src_y, src_u, src_v, y, uv_l, uv_h);
    YUVTORGB_D(y, uv_l, uv_h, vec_ubvr, vec_ugvg, vec_yg, vec_yb, b_l, b_h, g_l,
               g_h, r_l, r_h);
    STOREARGB_D(r_l, r_h, g_l, g_h, b_l, b_h, alpha, alpha, dst_argb);
    src_y += 32;
    src_u += 16;
    src_v += 16;
  }
}

void I422AlphaToARGBRow_LASX(const uint8_t* src_y,
                             const uint8_t* src_u,
                             const uint8_t* src_v,
                             const uint8_t* src_a,
                             uint8_t* dst_argb,
                             const struct YuvConstants* yuvconstants,
                             int width) {
  int x;
  int len = width / 32;
  int res = width & 31;
  __m256i vec_yb, vec_yg, vec_ub, vec_vr, vec_ug, vec_vg;
  __m256i vec_ubvr, vec_ugvg;
  __m256i zero = __lasx_xvldi(0);
  __m256i const_0x80 = __lasx_xvldi(0x80);

  YUVTORGB_SETUP(yuvconstants, vec_ub, vec_vr, vec_ug, vec_vg, vec_yg, vec_yb);
  vec_ubvr = __lasx_xvilvl_h(vec_ub, vec_vr);
  vec_ugvg = __lasx_xvilvl_h(vec_ug, vec_vg);

  for (x = 0; x < len; x++) {
    __m256i y, uv_l, uv_h, b_l, b_h, g_l, g_h, r_l, r_h, a_l, a_h;

    y = __lasx_xvld(src_a, 0);
    a_l = __lasx_xvilvl_b(zero, y);
    a_h = __lasx_xvilvh_b(zero, y);
    READYUV422_D(src_y, src_u, src_v, y, uv_l, uv_h);
    YUVTORGB_D(y, uv_l, uv_h, vec_ubvr, vec_ugvg, vec_yg, vec_yb, b_l, b_h, g_l,
               g_h, r_l, r_h);
    STOREARGB_D(a_l, a_h, r_l, r_h, g_l, g_h, b_l, b_h, dst_argb);
    src_y += 32;
    src_u += 16;
    src_v += 16;
    src_a += 32;
  }
  if (res) {
    __m256i y, uv, r, g, b, a;
    a = __lasx_xvld(src_a, 0);
    a = __lasx_vext2xv_hu_bu(a);
    READYUV422(src_y, src_u, src_v, y, uv);
    YUVTORGB(y, uv, vec_ubvr, vec_ugvg, vec_yg, vec_yb, b, g, r);
    STOREARGB(a, r, g, b, dst_argb);
  }
}

void I422ToRGB24Row_LASX(const uint8_t* src_y,
                         const uint8_t* src_u,
                         const uint8_t* src_v,
                         uint8_t* dst_argb,
                         const struct YuvConstants* yuvconstants,
                         int32_t width) {
  int x;
  int len = width / 32;
  __m256i vec_yb, vec_yg, vec_ub, vec_vr, vec_ug, vec_vg;
  __m256i vec_ubvr, vec_ugvg;
  __m256i const_0x80 = __lasx_xvldi(0x80);
  __m256i shuffler0 = {0x0504120302100100, 0x0A18090816070614,
                       0x0504120302100100, 0x0A18090816070614};
  __m256i shuffler1 = {0x1E0F0E1C0D0C1A0B, 0x1E0F0E1C0D0C1A0B,
                       0x1E0F0E1C0D0C1A0B, 0x1E0F0E1C0D0C1A0B};

  YUVTORGB_SETUP(yuvconstants, vec_ub, vec_vr, vec_ug, vec_vg, vec_yg, vec_yb);
  vec_ubvr = __lasx_xvilvl_h(vec_ub, vec_vr);
  vec_ugvg = __lasx_xvilvl_h(vec_ug, vec_vg);

  for (x = 0; x < len; x++) {
    __m256i y, uv_l, uv_h, b_l, b_h, g_l, g_h, r_l, r_h;
    __m256i temp0, temp1, temp2, temp3;

    READYUV422_D(src_y, src_u, src_v, y, uv_l, uv_h);
    YUVTORGB_D(y, uv_l, uv_h, vec_ubvr, vec_ugvg, vec_yg, vec_yb, b_l, b_h, g_l,
               g_h, r_l, r_h);
    temp0 = __lasx_xvpackev_b(g_l, b_l);
    temp1 = __lasx_xvpackev_b(g_h, b_h);
    DUP4_ARG3(__lasx_xvshuf_b, r_l, temp0, shuffler1, r_h, temp1, shuffler1,
              r_l, temp0, shuffler0, r_h, temp1, shuffler0, temp2, temp3, temp0,
              temp1);

    b_l = __lasx_xvilvl_d(temp1, temp2);
    b_h = __lasx_xvilvh_d(temp3, temp1);
    temp1 = __lasx_xvpermi_q(b_l, temp0, 0x20);
    temp2 = __lasx_xvpermi_q(temp0, b_h, 0x30);
    temp3 = __lasx_xvpermi_q(b_h, b_l, 0x31);
    __lasx_xvst(temp1, dst_argb, 0);
    __lasx_xvst(temp2, dst_argb, 32);
    __lasx_xvst(temp3, dst_argb, 64);
    dst_argb += 96;
    src_y += 32;
    src_u += 16;
    src_v += 16;
  }
}

// TODO(fbarchard): Consider AND instead of shift to isolate 5 upper bits of R.
void I422ToRGB565Row_LASX(const uint8_t* src_y,
                          const uint8_t* src_u,
                          const uint8_t* src_v,
                          uint8_t* dst_rgb565,
                          const struct YuvConstants* yuvconstants,
                          int width) {
  int x;
  int len = width / 32;
  __m256i vec_yb, vec_yg, vec_ub, vec_vr, vec_ug, vec_vg;
  __m256i vec_ubvr, vec_ugvg;
  __m256i const_0x80 = __lasx_xvldi(0x80);

  YUVTORGB_SETUP(yuvconstants, vec_ub, vec_vr, vec_ug, vec_vg, vec_yg, vec_yb);
  vec_ubvr = __lasx_xvilvl_h(vec_ub, vec_vr);
  vec_ugvg = __lasx_xvilvl_h(vec_ug, vec_vg);

  for (x = 0; x < len; x++) {
    __m256i y, uv_l, uv_h, b_l, b_h, g_l, g_h, r_l, r_h;
    __m256i dst_l, dst_h;

    READYUV422_D(src_y, src_u, src_v, y, uv_l, uv_h);
    YUVTORGB_D(y, uv_l, uv_h, vec_ubvr, vec_ugvg, vec_yg, vec_yb, b_l, b_h, g_l,
               g_h, r_l, r_h);
    b_l = __lasx_xvsrli_h(b_l, 3);
    b_h = __lasx_xvsrli_h(b_h, 3);
    g_l = __lasx_xvsrli_h(g_l, 2);
    g_h = __lasx_xvsrli_h(g_h, 2);
    r_l = __lasx_xvsrli_h(r_l, 3);
    r_h = __lasx_xvsrli_h(r_h, 3);
    r_l = __lasx_xvslli_h(r_l, 11);
    r_h = __lasx_xvslli_h(r_h, 11);
    g_l = __lasx_xvslli_h(g_l, 5);
    g_h = __lasx_xvslli_h(g_h, 5);
    r_l = __lasx_xvor_v(r_l, g_l);
    r_l = __lasx_xvor_v(r_l, b_l);
    r_h = __lasx_xvor_v(r_h, g_h);
    r_h = __lasx_xvor_v(r_h, b_h);
    dst_l = __lasx_xvpermi_q(r_h, r_l, 0x20);
    dst_h = __lasx_xvpermi_q(r_h, r_l, 0x31);
    __lasx_xvst(dst_l, dst_rgb565, 0);
    __lasx_xvst(dst_h, dst_rgb565, 32);
    dst_rgb565 += 64;
    src_y += 32;
    src_u += 16;
    src_v += 16;
  }
}

// TODO(fbarchard): Consider AND instead of shift to isolate 4 upper bits of G.
void I422ToARGB4444Row_LASX(const uint8_t* src_y,
                            const uint8_t* src_u,
                            const uint8_t* src_v,
                            uint8_t* dst_argb4444,
                            const struct YuvConstants* yuvconstants,
                            int width) {
  int x;
  int len = width / 32;
  __m256i vec_yb, vec_yg, vec_ub, vec_vr, vec_ug, vec_vg;
  __m256i vec_ubvr, vec_ugvg;
  __m256i const_0x80 = __lasx_xvldi(0x80);
  __m256i alpha = (__m256i)v4u64{0xF000F000F000F000, 0xF000F000F000F000,
                                 0xF000F000F000F000, 0xF000F000F000F000};
  __m256i mask = {0x00F000F000F000F0, 0x00F000F000F000F0, 0x00F000F000F000F0,
                  0x00F000F000F000F0};

  YUVTORGB_SETUP(yuvconstants, vec_ub, vec_vr, vec_ug, vec_vg, vec_yg, vec_yb);
  vec_ubvr = __lasx_xvilvl_h(vec_ub, vec_vr);
  vec_ugvg = __lasx_xvilvl_h(vec_ug, vec_vg);

  for (x = 0; x < len; x++) {
    __m256i y, uv_l, uv_h, b_l, b_h, g_l, g_h, r_l, r_h;
    __m256i dst_l, dst_h;

    READYUV422_D(src_y, src_u, src_v, y, uv_l, uv_h);
    YUVTORGB_D(y, uv_l, uv_h, vec_ubvr, vec_ugvg, vec_yg, vec_yb, b_l, b_h, g_l,
               g_h, r_l, r_h);
    b_l = __lasx_xvsrli_h(b_l, 4);
    b_h = __lasx_xvsrli_h(b_h, 4);
    r_l = __lasx_xvsrli_h(r_l, 4);
    r_h = __lasx_xvsrli_h(r_h, 4);
    g_l = __lasx_xvand_v(g_l, mask);
    g_h = __lasx_xvand_v(g_h, mask);
    r_l = __lasx_xvslli_h(r_l, 8);
    r_h = __lasx_xvslli_h(r_h, 8);
    r_l = __lasx_xvor_v(r_l, alpha);
    r_h = __lasx_xvor_v(r_h, alpha);
    r_l = __lasx_xvor_v(r_l, g_l);
    r_h = __lasx_xvor_v(r_h, g_h);
    r_l = __lasx_xvor_v(r_l, b_l);
    r_h = __lasx_xvor_v(r_h, b_h);
    dst_l = __lasx_xvpermi_q(r_h, r_l, 0x20);
    dst_h = __lasx_xvpermi_q(r_h, r_l, 0x31);
    __lasx_xvst(dst_l, dst_argb4444, 0);
    __lasx_xvst(dst_h, dst_argb4444, 32);
    dst_argb4444 += 64;
    src_y += 32;
    src_u += 16;
    src_v += 16;
  }
}

void I422ToARGB1555Row_LASX(const uint8_t* src_y,
                            const uint8_t* src_u,
                            const uint8_t* src_v,
                            uint8_t* dst_argb1555,
                            const struct YuvConstants* yuvconstants,
                            int width) {
  int x;
  int len = width / 32;
  __m256i vec_yb, vec_yg, vec_ub, vec_vr, vec_ug, vec_vg;
  __m256i vec_ubvr, vec_ugvg;
  __m256i const_0x80 = __lasx_xvldi(0x80);
  __m256i alpha = (__m256i)v4u64{0x8000800080008000, 0x8000800080008000,
                                 0x8000800080008000, 0x8000800080008000};

  YUVTORGB_SETUP(yuvconstants, vec_ub, vec_vr, vec_ug, vec_vg, vec_yg, vec_yb);
  vec_ubvr = __lasx_xvilvl_h(vec_ub, vec_vr);
  vec_ugvg = __lasx_xvilvl_h(vec_ug, vec_vg);

  for (x = 0; x < len; x++) {
    __m256i y, uv_l, uv_h, b_l, b_h, g_l, g_h, r_l, r_h;
    __m256i dst_l, dst_h;

    READYUV422_D(src_y, src_u, src_v, y, uv_l, uv_h);
    YUVTORGB_D(y, uv_l, uv_h, vec_ubvr, vec_ugvg, vec_yg, vec_yb, b_l, b_h, g_l,
               g_h, r_l, r_h);
    b_l = __lasx_xvsrli_h(b_l, 3);
    b_h = __lasx_xvsrli_h(b_h, 3);
    g_l = __lasx_xvsrli_h(g_l, 3);
    g_h = __lasx_xvsrli_h(g_h, 3);
    g_l = __lasx_xvslli_h(g_l, 5);
    g_h = __lasx_xvslli_h(g_h, 5);
    r_l = __lasx_xvsrli_h(r_l, 3);
    r_h = __lasx_xvsrli_h(r_h, 3);
    r_l = __lasx_xvslli_h(r_l, 10);
    r_h = __lasx_xvslli_h(r_h, 10);
    r_l = __lasx_xvor_v(r_l, alpha);
    r_h = __lasx_xvor_v(r_h, alpha);
    r_l = __lasx_xvor_v(r_l, g_l);
    r_h = __lasx_xvor_v(r_h, g_h);
    r_l = __lasx_xvor_v(r_l, b_l);
    r_h = __lasx_xvor_v(r_h, b_h);
    dst_l = __lasx_xvpermi_q(r_h, r_l, 0x20);
    dst_h = __lasx_xvpermi_q(r_h, r_l, 0x31);
    __lasx_xvst(dst_l, dst_argb1555, 0);
    __lasx_xvst(dst_h, dst_argb1555, 32);
    dst_argb1555 += 64;
    src_y += 32;
    src_u += 16;
    src_v += 16;
  }
}

void YUY2ToYRow_LASX(const uint8_t* src_yuy2, uint8_t* dst_y, int width) {
  int x;
  int len = width / 32;
  __m256i src0, src1, dst0;

  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lasx_xvld, src_yuy2, 0, src_yuy2, 32, src0, src1);
    dst0 = __lasx_xvpickev_b(src1, src0);
    dst0 = __lasx_xvpermi_d(dst0, 0xD8);
    __lasx_xvst(dst0, dst_y, 0);
    src_yuy2 += 64;
    dst_y += 32;
  }
}

void YUY2ToUVRow_LASX(const uint8_t* src_yuy2,
                      int src_stride_yuy2,
                      uint8_t* dst_u,
                      uint8_t* dst_v,
                      int width) {
  const uint8_t* src_yuy2_next = src_yuy2 + src_stride_yuy2;
  int x;
  int len = width / 32;
  __m256i src0, src1, src2, src3;
  __m256i tmp0, dst0, dst1;

  for (x = 0; x < len; x++) {
    DUP4_ARG2(__lasx_xvld, src_yuy2, 0, src_yuy2, 32, src_yuy2_next, 0,
              src_yuy2_next, 32, src0, src1, src2, src3);
    src0 = __lasx_xvpickod_b(src1, src0);
    src1 = __lasx_xvpickod_b(src3, src2);
    tmp0 = __lasx_xvavgr_bu(src1, src0);
    tmp0 = __lasx_xvpermi_d(tmp0, 0xD8);
    dst0 = __lasx_xvpickev_b(tmp0, tmp0);
    dst1 = __lasx_xvpickod_b(tmp0, tmp0);
    __lasx_xvstelm_d(dst0, dst_u, 0, 0);
    __lasx_xvstelm_d(dst0, dst_u, 8, 2);
    __lasx_xvstelm_d(dst1, dst_v, 0, 0);
    __lasx_xvstelm_d(dst1, dst_v, 8, 2);
    src_yuy2 += 64;
    src_yuy2_next += 64;
    dst_u += 16;
    dst_v += 16;
  }
}

void YUY2ToUV422Row_LASX(const uint8_t* src_yuy2,
                         uint8_t* dst_u,
                         uint8_t* dst_v,
                         int width) {
  int x;
  int len = width / 32;
  __m256i src0, src1, tmp0, dst0, dst1;

  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lasx_xvld, src_yuy2, 0, src_yuy2, 32, src0, src1);
    tmp0 = __lasx_xvpickod_b(src1, src0);
    tmp0 = __lasx_xvpermi_d(tmp0, 0xD8);
    dst0 = __lasx_xvpickev_b(tmp0, tmp0);
    dst1 = __lasx_xvpickod_b(tmp0, tmp0);
    __lasx_xvstelm_d(dst0, dst_u, 0, 0);
    __lasx_xvstelm_d(dst0, dst_u, 8, 2);
    __lasx_xvstelm_d(dst1, dst_v, 0, 0);
    __lasx_xvstelm_d(dst1, dst_v, 8, 2);
    src_yuy2 += 64;
    dst_u += 16;
    dst_v += 16;
  }
}

void UYVYToYRow_LASX(const uint8_t* src_uyvy, uint8_t* dst_y, int width) {
  int x;
  int len = width / 32;
  __m256i src0, src1, dst0;

  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lasx_xvld, src_uyvy, 0, src_uyvy, 32, src0, src1);
    dst0 = __lasx_xvpickod_b(src1, src0);
    dst0 = __lasx_xvpermi_d(dst0, 0xD8);
    __lasx_xvst(dst0, dst_y, 0);
    src_uyvy += 64;
    dst_y += 32;
  }
}

void UYVYToUVRow_LASX(const uint8_t* src_uyvy,
                      int src_stride_uyvy,
                      uint8_t* dst_u,
                      uint8_t* dst_v,
                      int width) {
  const uint8_t* src_uyvy_next = src_uyvy + src_stride_uyvy;
  int x;
  int len = width / 32;
  __m256i src0, src1, src2, src3, tmp0, dst0, dst1;

  for (x = 0; x < len; x++) {
    DUP4_ARG2(__lasx_xvld, src_uyvy, 0, src_uyvy, 32, src_uyvy_next, 0,
              src_uyvy_next, 32, src0, src1, src2, src3);
    src0 = __lasx_xvpickev_b(src1, src0);
    src1 = __lasx_xvpickev_b(src3, src2);
    tmp0 = __lasx_xvavgr_bu(src1, src0);
    tmp0 = __lasx_xvpermi_d(tmp0, 0xD8);
    dst0 = __lasx_xvpickev_b(tmp0, tmp0);
    dst1 = __lasx_xvpickod_b(tmp0, tmp0);
    __lasx_xvstelm_d(dst0, dst_u, 0, 0);
    __lasx_xvstelm_d(dst0, dst_u, 8, 2);
    __lasx_xvstelm_d(dst1, dst_v, 0, 0);
    __lasx_xvstelm_d(dst1, dst_v, 8, 2);
    src_uyvy += 64;
    src_uyvy_next += 64;
    dst_u += 16;
    dst_v += 16;
  }
}

void UYVYToUV422Row_LASX(const uint8_t* src_uyvy,
                         uint8_t* dst_u,
                         uint8_t* dst_v,
                         int width) {
  int x;
  int len = width / 32;
  __m256i src0, src1, tmp0, dst0, dst1;

  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lasx_xvld, src_uyvy, 0, src_uyvy, 32, src0, src1);
    tmp0 = __lasx_xvpickev_b(src1, src0);
    tmp0 = __lasx_xvpermi_d(tmp0, 0xD8);
    dst0 = __lasx_xvpickev_b(tmp0, tmp0);
    dst1 = __lasx_xvpickod_b(tmp0, tmp0);
    __lasx_xvstelm_d(dst0, dst_u, 0, 0);
    __lasx_xvstelm_d(dst0, dst_u, 8, 2);
    __lasx_xvstelm_d(dst1, dst_v, 0, 0);
    __lasx_xvstelm_d(dst1, dst_v, 8, 2);
    src_uyvy += 64;
    dst_u += 16;
    dst_v += 16;
  }
}

void ARGBToUVRow_LASX(const uint8_t* src_argb0,
                      int src_stride_argb,
                      uint8_t* dst_u,
                      uint8_t* dst_v,
                      int width) {
  int x;
  int len = width / 32;
  const uint8_t* src_argb1 = src_argb0 + src_stride_argb;

  __m256i src0, src1, src2, src3, src4, src5, src6, src7;
  __m256i vec0, vec1, vec2, vec3;
  __m256i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, dst0, dst1;
  __m256i const_0x70 = {0x0038003800380038, 0x0038003800380038,
                        0x0038003800380038, 0x0038003800380038};
  __m256i const_0x4A = {0x0025002500250025, 0x0025002500250025,
                        0x0025002500250025, 0x0025002500250025};
  __m256i const_0x26 = {0x0013001300130013, 0x0013001300130013,
                        0x0013001300130013, 0x0013001300130013};
  __m256i const_0x5E = {0x002f002f002f002f, 0x002f002f002f002f,
                        0x002f002f002f002f, 0x002f002f002f002f};
  __m256i const_0x12 = {0x0009000900090009, 0x0009000900090009,
                        0x0009000900090009, 0x0009000900090009};
  __m256i control = {0x0000000400000000, 0x0000000500000001, 0x0000000600000002,
                     0x0000000700000003};
  __m256i const_0x8080 = (__m256i)v4u64{0x8080808080808080, 0x8080808080808080,
                                        0x8080808080808080, 0x8080808080808080};

  for (x = 0; x < len; x++) {
    DUP4_ARG2(__lasx_xvld, src_argb0, 0, src_argb0, 32, src_argb0, 64,
              src_argb0, 96, src0, src1, src2, src3);
    DUP4_ARG2(__lasx_xvld, src_argb1, 0, src_argb1, 32, src_argb1, 64,
              src_argb1, 96, src4, src5, src6, src7);
    vec0 = __lasx_xvaddwev_h_bu(src0, src4);
    vec1 = __lasx_xvaddwev_h_bu(src1, src5);
    vec2 = __lasx_xvaddwev_h_bu(src2, src6);
    vec3 = __lasx_xvaddwev_h_bu(src3, src7);
    tmp0 = __lasx_xvpickev_h(vec1, vec0);
    tmp1 = __lasx_xvpickev_h(vec3, vec2);
    tmp2 = __lasx_xvpickod_h(vec1, vec0);
    tmp3 = __lasx_xvpickod_h(vec3, vec2);
    vec0 = __lasx_xvaddwod_h_bu(src0, src4);
    vec1 = __lasx_xvaddwod_h_bu(src1, src5);
    vec2 = __lasx_xvaddwod_h_bu(src2, src6);
    vec3 = __lasx_xvaddwod_h_bu(src3, src7);
    tmp4 = __lasx_xvpickev_h(vec1, vec0);
    tmp5 = __lasx_xvpickev_h(vec3, vec2);
    vec0 = __lasx_xvpickev_h(tmp1, tmp0);
    vec1 = __lasx_xvpickod_h(tmp1, tmp0);
    src0 = __lasx_xvavgr_h(vec0, vec1);
    vec0 = __lasx_xvpickev_h(tmp3, tmp2);
    vec1 = __lasx_xvpickod_h(tmp3, tmp2);
    src1 = __lasx_xvavgr_h(vec0, vec1);
    vec0 = __lasx_xvpickev_h(tmp5, tmp4);
    vec1 = __lasx_xvpickod_h(tmp5, tmp4);
    src2 = __lasx_xvavgr_h(vec0, vec1);
    dst0 = __lasx_xvmadd_h(const_0x8080, src0, const_0x70);
    dst0 = __lasx_xvmsub_h(dst0, src2, const_0x4A);
    dst0 = __lasx_xvmsub_h(dst0, src1, const_0x26);
    dst1 = __lasx_xvmadd_h(const_0x8080, src1, const_0x70);
    dst1 = __lasx_xvmsub_h(dst1, src2, const_0x5E);
    dst1 = __lasx_xvmsub_h(dst1, src0, const_0x12);
    dst0 = __lasx_xvperm_w(dst0, control);
    dst1 = __lasx_xvperm_w(dst1, control);
    dst0 = __lasx_xvssrani_b_h(dst0, dst0, 8);
    dst1 = __lasx_xvssrani_b_h(dst1, dst1, 8);
    __lasx_xvstelm_d(dst0, dst_u, 0, 0);
    __lasx_xvstelm_d(dst0, dst_u, 8, 2);
    __lasx_xvstelm_d(dst1, dst_v, 0, 0);
    __lasx_xvstelm_d(dst1, dst_v, 8, 2);
    src_argb0 += 128;
    src_argb1 += 128;
    dst_u += 16;
    dst_v += 16;
  }
}

void ARGBToRGB24Row_LASX(const uint8_t* src_argb, uint8_t* dst_rgb, int width) {
  int x;
  int len = (width / 32) - 1;
  __m256i src0, src1, src2, src3;
  __m256i tmp0, tmp1, tmp2, tmp3;
  __m256i shuf = {0x0908060504020100, 0x000000000E0D0C0A, 0x0908060504020100,
                  0x000000000E0D0C0A};
  __m256i control = {0x0000000100000000, 0x0000000400000002, 0x0000000600000005,
                     0x0000000700000003};
  for (x = 0; x < len; x++) {
    DUP4_ARG2(__lasx_xvld, src_argb, 0, src_argb, 32, src_argb, 64, src_argb,
              96, src0, src1, src2, src3);
    tmp0 = __lasx_xvshuf_b(src0, src0, shuf);
    tmp1 = __lasx_xvshuf_b(src1, src1, shuf);
    tmp2 = __lasx_xvshuf_b(src2, src2, shuf);
    tmp3 = __lasx_xvshuf_b(src3, src3, shuf);
    tmp0 = __lasx_xvperm_w(tmp0, control);
    tmp1 = __lasx_xvperm_w(tmp1, control);
    tmp2 = __lasx_xvperm_w(tmp2, control);
    tmp3 = __lasx_xvperm_w(tmp3, control);
    __lasx_xvst(tmp0, dst_rgb, 0);
    __lasx_xvst(tmp1, dst_rgb, 24);
    __lasx_xvst(tmp2, dst_rgb, 48);
    __lasx_xvst(tmp3, dst_rgb, 72);
    dst_rgb += 96;
    src_argb += 128;
  }
  DUP4_ARG2(__lasx_xvld, src_argb, 0, src_argb, 32, src_argb, 64, src_argb, 96,
            src0, src1, src2, src3);
  tmp0 = __lasx_xvshuf_b(src0, src0, shuf);
  tmp1 = __lasx_xvshuf_b(src1, src1, shuf);
  tmp2 = __lasx_xvshuf_b(src2, src2, shuf);
  tmp3 = __lasx_xvshuf_b(src3, src3, shuf);
  tmp0 = __lasx_xvperm_w(tmp0, control);
  tmp1 = __lasx_xvperm_w(tmp1, control);
  tmp2 = __lasx_xvperm_w(tmp2, control);
  tmp3 = __lasx_xvperm_w(tmp3, control);
  __lasx_xvst(tmp0, dst_rgb, 0);
  __lasx_xvst(tmp1, dst_rgb, 24);
  __lasx_xvst(tmp2, dst_rgb, 48);
  dst_rgb += 72;
  __lasx_xvstelm_d(tmp3, dst_rgb, 0, 0);
  __lasx_xvstelm_d(tmp3, dst_rgb, 8, 1);
  __lasx_xvstelm_d(tmp3, dst_rgb, 16, 2);
}

void ARGBToRAWRow_LASX(const uint8_t* src_argb, uint8_t* dst_rgb, int width) {
  int x;
  int len = (width / 32) - 1;
  __m256i src0, src1, src2, src3;
  __m256i tmp0, tmp1, tmp2, tmp3;
  __m256i shuf = {0x090A040506000102, 0x000000000C0D0E08, 0x090A040506000102,
                  0x000000000C0D0E08};
  __m256i control = {0x0000000100000000, 0x0000000400000002, 0x0000000600000005,
                     0x0000000700000003};
  for (x = 0; x < len; x++) {
    DUP4_ARG2(__lasx_xvld, src_argb, 0, src_argb, 32, src_argb, 64, src_argb,
              96, src0, src1, src2, src3);
    tmp0 = __lasx_xvshuf_b(src0, src0, shuf);
    tmp1 = __lasx_xvshuf_b(src1, src1, shuf);
    tmp2 = __lasx_xvshuf_b(src2, src2, shuf);
    tmp3 = __lasx_xvshuf_b(src3, src3, shuf);
    tmp0 = __lasx_xvperm_w(tmp0, control);
    tmp1 = __lasx_xvperm_w(tmp1, control);
    tmp2 = __lasx_xvperm_w(tmp2, control);
    tmp3 = __lasx_xvperm_w(tmp3, control);
    __lasx_xvst(tmp0, dst_rgb, 0);
    __lasx_xvst(tmp1, dst_rgb, 24);
    __lasx_xvst(tmp2, dst_rgb, 48);
    __lasx_xvst(tmp3, dst_rgb, 72);
    dst_rgb += 96;
    src_argb += 128;
  }
  DUP4_ARG2(__lasx_xvld, src_argb, 0, src_argb, 32, src_argb, 64, src_argb, 96,
            src0, src1, src2, src3);
  tmp0 = __lasx_xvshuf_b(src0, src0, shuf);
  tmp1 = __lasx_xvshuf_b(src1, src1, shuf);
  tmp2 = __lasx_xvshuf_b(src2, src2, shuf);
  tmp3 = __lasx_xvshuf_b(src3, src3, shuf);
  tmp0 = __lasx_xvperm_w(tmp0, control);
  tmp1 = __lasx_xvperm_w(tmp1, control);
  tmp2 = __lasx_xvperm_w(tmp2, control);
  tmp3 = __lasx_xvperm_w(tmp3, control);
  __lasx_xvst(tmp0, dst_rgb, 0);
  __lasx_xvst(tmp1, dst_rgb, 24);
  __lasx_xvst(tmp2, dst_rgb, 48);
  dst_rgb += 72;
  __lasx_xvstelm_d(tmp3, dst_rgb, 0, 0);
  __lasx_xvstelm_d(tmp3, dst_rgb, 8, 1);
  __lasx_xvstelm_d(tmp3, dst_rgb, 16, 2);
}

void ARGBToRGB565Row_LASX(const uint8_t* src_argb,
                          uint8_t* dst_rgb,
                          int width) {
  int x;
  int len = width / 16;
  __m256i zero = __lasx_xvldi(0);
  __m256i src0, src1, tmp0, tmp1, dst0;
  __m256i shift = {0x0300030003000300, 0x0300030003000300, 0x0300030003000300,
                   0x0300030003000300};

  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lasx_xvld, src_argb, 0, src_argb, 32, src0, src1);
    tmp0 = __lasx_xvpickev_b(src1, src0);
    tmp1 = __lasx_xvpickod_b(src1, src0);
    tmp0 = __lasx_xvsrli_b(tmp0, 3);
    tmp1 = __lasx_xvpackev_b(zero, tmp1);
    tmp1 = __lasx_xvsrli_h(tmp1, 2);
    tmp0 = __lasx_xvsll_b(tmp0, shift);
    tmp1 = __lasx_xvslli_h(tmp1, 5);
    dst0 = __lasx_xvor_v(tmp0, tmp1);
    dst0 = __lasx_xvpermi_d(dst0, 0xD8);
    __lasx_xvst(dst0, dst_rgb, 0);
    dst_rgb += 32;
    src_argb += 64;
  }
}

void ARGBToARGB1555Row_LASX(const uint8_t* src_argb,
                            uint8_t* dst_rgb,
                            int width) {
  int x;
  int len = width / 16;
  __m256i zero = __lasx_xvldi(0);
  __m256i src0, src1, tmp0, tmp1, tmp2, tmp3, dst0;
  __m256i shift1 = {0x0703070307030703, 0x0703070307030703, 0x0703070307030703,
                    0x0703070307030703};
  __m256i shift2 = {0x0200020002000200, 0x0200020002000200, 0x0200020002000200,
                    0x0200020002000200};

  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lasx_xvld, src_argb, 0, src_argb, 32, src0, src1);
    tmp0 = __lasx_xvpickev_b(src1, src0);
    tmp1 = __lasx_xvpickod_b(src1, src0);
    tmp0 = __lasx_xvsrli_b(tmp0, 3);
    tmp1 = __lasx_xvsrl_b(tmp1, shift1);
    tmp0 = __lasx_xvsll_b(tmp0, shift2);
    tmp2 = __lasx_xvpackev_b(zero, tmp1);
    tmp3 = __lasx_xvpackod_b(zero, tmp1);
    tmp2 = __lasx_xvslli_h(tmp2, 5);
    tmp3 = __lasx_xvslli_h(tmp3, 15);
    dst0 = __lasx_xvor_v(tmp0, tmp2);
    dst0 = __lasx_xvor_v(dst0, tmp3);
    dst0 = __lasx_xvpermi_d(dst0, 0xD8);
    __lasx_xvst(dst0, dst_rgb, 0);
    dst_rgb += 32;
    src_argb += 64;
  }
}

void ARGBToARGB4444Row_LASX(const uint8_t* src_argb,
                            uint8_t* dst_rgb,
                            int width) {
  int x;
  int len = width / 16;
  __m256i src0, src1, tmp0, tmp1, dst0;

  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lasx_xvld, src_argb, 0, src_argb, 32, src0, src1);
    tmp0 = __lasx_xvpickev_b(src1, src0);
    tmp1 = __lasx_xvpickod_b(src1, src0);
    tmp1 = __lasx_xvandi_b(tmp1, 0xF0);
    tmp0 = __lasx_xvsrli_b(tmp0, 4);
    dst0 = __lasx_xvor_v(tmp1, tmp0);
    dst0 = __lasx_xvpermi_d(dst0, 0xD8);
    __lasx_xvst(dst0, dst_rgb, 0);
    dst_rgb += 32;
    src_argb += 64;
  }
}

void ARGBToUV444Row_LASX(const uint8_t* src_argb,
                         uint8_t* dst_u,
                         uint8_t* dst_v,
                         int32_t width) {
  int x;
  int len = width / 32;
  __m256i src0, src1, src2, src3;
  __m256i tmp0, tmp1, tmp2, tmp3;
  __m256i reg0, reg1, reg2, reg3, dst0, dst1;
  __m256i const_112 = __lasx_xvldi(112);
  __m256i const_74 = __lasx_xvldi(74);
  __m256i const_38 = __lasx_xvldi(38);
  __m256i const_94 = __lasx_xvldi(94);
  __m256i const_18 = __lasx_xvldi(18);
  __m256i const_0x8080 = (__m256i)v4u64{0x8080808080808080, 0x8080808080808080,
                                        0x8080808080808080, 0x8080808080808080};
  __m256i control = {0x0000000400000000, 0x0000000500000001, 0x0000000600000002,
                     0x0000000700000003};
  for (x = 0; x < len; x++) {
    DUP4_ARG2(__lasx_xvld, src_argb, 0, src_argb, 32, src_argb, 64, src_argb,
              96, src0, src1, src2, src3);
    tmp0 = __lasx_xvpickev_h(src1, src0);
    tmp1 = __lasx_xvpickod_h(src1, src0);
    tmp2 = __lasx_xvpickev_h(src3, src2);
    tmp3 = __lasx_xvpickod_h(src3, src2);
    reg0 = __lasx_xvmaddwev_h_bu(const_0x8080, tmp0, const_112);
    reg1 = __lasx_xvmaddwev_h_bu(const_0x8080, tmp2, const_112);
    reg2 = __lasx_xvmulwod_h_bu(tmp0, const_74);
    reg3 = __lasx_xvmulwod_h_bu(tmp2, const_74);
    reg2 = __lasx_xvmaddwev_h_bu(reg2, tmp1, const_38);
    reg3 = __lasx_xvmaddwev_h_bu(reg3, tmp3, const_38);
    reg0 = __lasx_xvsub_h(reg0, reg2);
    reg1 = __lasx_xvsub_h(reg1, reg3);
    dst0 = __lasx_xvssrani_b_h(reg1, reg0, 8);
    dst0 = __lasx_xvperm_w(dst0, control);
    reg0 = __lasx_xvmaddwev_h_bu(const_0x8080, tmp1, const_112);
    reg1 = __lasx_xvmaddwev_h_bu(const_0x8080, tmp3, const_112);
    reg2 = __lasx_xvmulwev_h_bu(tmp0, const_18);
    reg3 = __lasx_xvmulwev_h_bu(tmp2, const_18);
    reg2 = __lasx_xvmaddwod_h_bu(reg2, tmp0, const_94);
    reg3 = __lasx_xvmaddwod_h_bu(reg3, tmp2, const_94);
    reg0 = __lasx_xvsub_h(reg0, reg2);
    reg1 = __lasx_xvsub_h(reg1, reg3);
    dst1 = __lasx_xvssrani_b_h(reg1, reg0, 8);
    dst1 = __lasx_xvperm_w(dst1, control);
    __lasx_xvst(dst0, dst_u, 0);
    __lasx_xvst(dst1, dst_v, 0);
    dst_u += 32;
    dst_v += 32;
    src_argb += 128;
  }
}

void ARGBMultiplyRow_LASX(const uint8_t* src_argb0,
                          const uint8_t* src_argb1,
                          uint8_t* dst_argb,
                          int width) {
  int x;
  int len = width / 8;
  __m256i zero = __lasx_xvldi(0);
  __m256i src0, src1, dst0, dst1;
  __m256i tmp0, tmp1, tmp2, tmp3;

  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lasx_xvld, src_argb0, 0, src_argb1, 0, src0, src1);
    tmp0 = __lasx_xvilvl_b(src0, src0);
    tmp1 = __lasx_xvilvh_b(src0, src0);
    tmp2 = __lasx_xvilvl_b(zero, src1);
    tmp3 = __lasx_xvilvh_b(zero, src1);
    dst0 = __lasx_xvmuh_hu(tmp0, tmp2);
    dst1 = __lasx_xvmuh_hu(tmp1, tmp3);
    dst0 = __lasx_xvpickev_b(dst1, dst0);
    __lasx_xvst(dst0, dst_argb, 0);
    src_argb0 += 32;
    src_argb1 += 32;
    dst_argb += 32;
  }
}

void ARGBAddRow_LASX(const uint8_t* src_argb0,
                     const uint8_t* src_argb1,
                     uint8_t* dst_argb,
                     int width) {
  int x;
  int len = width / 8;
  __m256i src0, src1, dst0;

  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lasx_xvld, src_argb0, 0, src_argb1, 0, src0, src1);
    dst0 = __lasx_xvsadd_bu(src0, src1);
    __lasx_xvst(dst0, dst_argb, 0);
    src_argb0 += 32;
    src_argb1 += 32;
    dst_argb += 32;
  }
}

void ARGBSubtractRow_LASX(const uint8_t* src_argb0,
                          const uint8_t* src_argb1,
                          uint8_t* dst_argb,
                          int width) {
  int x;
  int len = width / 8;
  __m256i src0, src1, dst0;

  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lasx_xvld, src_argb0, 0, src_argb1, 0, src0, src1);
    dst0 = __lasx_xvssub_bu(src0, src1);
    __lasx_xvst(dst0, dst_argb, 0);
    src_argb0 += 32;
    src_argb1 += 32;
    dst_argb += 32;
  }
}

void ARGBAttenuateRow_LASX(const uint8_t* src_argb,
                           uint8_t* dst_argb,
                           int width) {
  int x;
  int len = width / 16;
  __m256i src0, src1, tmp0, tmp1;
  __m256i reg0, reg1, reg2, reg3, reg4, reg5;
  __m256i b, g, r, a, dst0, dst1;
  __m256i control = {0x0005000100040000, 0x0007000300060002, 0x0005000100040000,
                     0x0007000300060002};
  __m256i zero = __lasx_xvldi(0);
  __m256i const_add = __lasx_xvldi(0x8ff);

  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lasx_xvld, src_argb, 0, src_argb, 32, src0, src1);
    tmp0 = __lasx_xvpickev_b(src1, src0);
    tmp1 = __lasx_xvpickod_b(src1, src0);
    b = __lasx_xvpackev_b(zero, tmp0);
    r = __lasx_xvpackod_b(zero, tmp0);
    g = __lasx_xvpackev_b(zero, tmp1);
    a = __lasx_xvpackod_b(zero, tmp1);
    reg0 = __lasx_xvmaddwev_w_hu(const_add, b, a);
    reg1 = __lasx_xvmaddwod_w_hu(const_add, b, a);
    reg2 = __lasx_xvmaddwev_w_hu(const_add, r, a);
    reg3 = __lasx_xvmaddwod_w_hu(const_add, r, a);
    reg4 = __lasx_xvmaddwev_w_hu(const_add, g, a);
    reg5 = __lasx_xvmaddwod_w_hu(const_add, g, a);
    reg0 = __lasx_xvssrani_h_w(reg1, reg0, 8);
    reg2 = __lasx_xvssrani_h_w(reg3, reg2, 8);
    reg4 = __lasx_xvssrani_h_w(reg5, reg4, 8);
    reg0 = __lasx_xvshuf_h(control, reg0, reg0);
    reg2 = __lasx_xvshuf_h(control, reg2, reg2);
    reg4 = __lasx_xvshuf_h(control, reg4, reg4);
    tmp0 = __lasx_xvpackev_b(reg4, reg0);
    tmp1 = __lasx_xvpackev_b(a, reg2);
    dst0 = __lasx_xvilvl_h(tmp1, tmp0);
    dst1 = __lasx_xvilvh_h(tmp1, tmp0);
    __lasx_xvst(dst0, dst_argb, 0);
    __lasx_xvst(dst1, dst_argb, 32);
    dst_argb += 64;
    src_argb += 64;
  }
}

void ARGBToRGB565DitherRow_LASX(const uint8_t* src_argb,
                                uint8_t* dst_rgb,
                                uint32_t dither4,
                                int width) {
  int x;
  int len = width / 16;
  __m256i src0, src1, tmp0, tmp1, dst0;
  __m256i b, g, r;
  __m256i zero = __lasx_xvldi(0);
  __m256i vec_dither = __lasx_xvldrepl_w(&dither4, 0);

  vec_dither = __lasx_xvilvl_b(zero, vec_dither);
  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lasx_xvld, src_argb, 0, src_argb, 32, src0, src1);
    tmp0 = __lasx_xvpickev_b(src1, src0);
    tmp1 = __lasx_xvpickod_b(src1, src0);
    b = __lasx_xvpackev_b(zero, tmp0);
    r = __lasx_xvpackod_b(zero, tmp0);
    g = __lasx_xvpackev_b(zero, tmp1);
    b = __lasx_xvadd_h(b, vec_dither);
    g = __lasx_xvadd_h(g, vec_dither);
    r = __lasx_xvadd_h(r, vec_dither);
    DUP2_ARG1(__lasx_xvclip255_h, b, g, b, g);
    r = __lasx_xvclip255_h(r);
    b = __lasx_xvsrai_h(b, 3);
    g = __lasx_xvsrai_h(g, 2);
    r = __lasx_xvsrai_h(r, 3);
    g = __lasx_xvslli_h(g, 5);
    r = __lasx_xvslli_h(r, 11);
    dst0 = __lasx_xvor_v(b, g);
    dst0 = __lasx_xvor_v(dst0, r);
    dst0 = __lasx_xvpermi_d(dst0, 0xD8);
    __lasx_xvst(dst0, dst_rgb, 0);
    src_argb += 64;
    dst_rgb += 32;
  }
}

void ARGBShuffleRow_LASX(const uint8_t* src_argb,
                         uint8_t* dst_argb,
                         const uint8_t* shuffler,
                         int width) {
  int x;
  int len = width / 16;
  __m256i src0, src1, dst0, dst1;
  __m256i shuf = {0x0404040400000000, 0x0C0C0C0C08080808, 0x0404040400000000,
                  0x0C0C0C0C08080808};
  __m256i temp = __lasx_xvldrepl_w(shuffler, 0);

  shuf = __lasx_xvadd_b(shuf, temp);
  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lasx_xvld, src_argb, 0, src_argb, 32, src0, src1);
    dst0 = __lasx_xvshuf_b(src0, src0, shuf);
    dst1 = __lasx_xvshuf_b(src1, src1, shuf);
    __lasx_xvst(dst0, dst_argb, 0);
    __lasx_xvst(dst1, dst_argb, 32);
    src_argb += 64;
    dst_argb += 64;
  }
}

void ARGBShadeRow_LASX(const uint8_t* src_argb,
                       uint8_t* dst_argb,
                       int width,
                       uint32_t value) {
  int x;
  int len = width / 8;
  __m256i src0, dst0, tmp0, tmp1;
  __m256i vec_value = __lasx_xvreplgr2vr_w(value);

  vec_value = __lasx_xvilvl_b(vec_value, vec_value);
  for (x = 0; x < len; x++) {
    src0 = __lasx_xvld(src_argb, 0);
    tmp0 = __lasx_xvilvl_b(src0, src0);
    tmp1 = __lasx_xvilvh_b(src0, src0);
    tmp0 = __lasx_xvmuh_hu(tmp0, vec_value);
    tmp1 = __lasx_xvmuh_hu(tmp1, vec_value);
    dst0 = __lasx_xvpickod_b(tmp1, tmp0);
    __lasx_xvst(dst0, dst_argb, 0);
    src_argb += 32;
    dst_argb += 32;
  }
}

void ARGBGrayRow_LASX(const uint8_t* src_argb, uint8_t* dst_argb, int width) {
  int x;
  int len = width / 16;
  __m256i src0, src1, tmp0, tmp1;
  __m256i reg0, reg1, reg2, dst0, dst1;
  __m256i const_128 = __lasx_xvldi(0x480);
  __m256i const_150 = __lasx_xvldi(0x96);
  __m256i const_br = {0x4D1D4D1D4D1D4D1D, 0x4D1D4D1D4D1D4D1D,
                      0x4D1D4D1D4D1D4D1D, 0x4D1D4D1D4D1D4D1D};

  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lasx_xvld, src_argb, 0, src_argb, 32, src0, src1);
    tmp0 = __lasx_xvpickev_b(src1, src0);
    tmp1 = __lasx_xvpickod_b(src1, src0);
    reg0 = __lasx_xvdp2_h_bu(tmp0, const_br);
    reg1 = __lasx_xvmaddwev_h_bu(const_128, tmp1, const_150);
    reg2 = __lasx_xvadd_h(reg0, reg1);
    tmp0 = __lasx_xvpackod_b(reg2, reg2);
    tmp1 = __lasx_xvpackod_b(tmp1, reg2);
    dst0 = __lasx_xvilvl_h(tmp1, tmp0);
    dst1 = __lasx_xvilvh_h(tmp1, tmp0);
    __lasx_xvst(dst0, dst_argb, 0);
    __lasx_xvst(dst1, dst_argb, 32);
    src_argb += 64;
    dst_argb += 64;
  }
}

void ARGBSepiaRow_LASX(uint8_t* dst_argb, int width) {
  int x;
  int len = width / 16;
  __m256i src0, src1, tmp0, tmp1;
  __m256i reg0, reg1, spb, spg, spr;
  __m256i dst0, dst1;
  __m256i spb_g = __lasx_xvldi(68);
  __m256i spg_g = __lasx_xvldi(88);
  __m256i spr_g = __lasx_xvldi(98);
  __m256i spb_br = {0x2311231123112311, 0x2311231123112311, 0x2311231123112311,
                    0x2311231123112311};
  __m256i spg_br = {0x2D162D162D162D16, 0x2D162D162D162D16, 0x2D162D162D162D16,
                    0x2D162D162D162D16};
  __m256i spr_br = {0x3218321832183218, 0x3218321832183218, 0x3218321832183218,
                    0x3218321832183218};
  __m256i shuff = {0x1706150413021100, 0x1F0E1D0C1B0A1908, 0x1706150413021100,
                   0x1F0E1D0C1B0A1908};

  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lasx_xvld, dst_argb, 0, dst_argb, 32, src0, src1);
    tmp0 = __lasx_xvpickev_b(src1, src0);
    tmp1 = __lasx_xvpickod_b(src1, src0);
    DUP2_ARG2(__lasx_xvdp2_h_bu, tmp0, spb_br, tmp0, spg_br, spb, spg);
    spr = __lasx_xvdp2_h_bu(tmp0, spr_br);
    spb = __lasx_xvmaddwev_h_bu(spb, tmp1, spb_g);
    spg = __lasx_xvmaddwev_h_bu(spg, tmp1, spg_g);
    spr = __lasx_xvmaddwev_h_bu(spr, tmp1, spr_g);
    spb = __lasx_xvsrli_h(spb, 7);
    spg = __lasx_xvsrli_h(spg, 7);
    spr = __lasx_xvsrli_h(spr, 7);
    spg = __lasx_xvsat_hu(spg, 7);
    spr = __lasx_xvsat_hu(spr, 7);
    reg0 = __lasx_xvpackev_b(spg, spb);
    reg1 = __lasx_xvshuf_b(tmp1, spr, shuff);
    dst0 = __lasx_xvilvl_h(reg1, reg0);
    dst1 = __lasx_xvilvh_h(reg1, reg0);
    __lasx_xvst(dst0, dst_argb, 0);
    __lasx_xvst(dst1, dst_argb, 32);
    dst_argb += 64;
  }
}

void ARGB4444ToARGBRow_LASX(const uint8_t* src_argb4444,
                            uint8_t* dst_argb,
                            int width) {
  int x;
  int len = width / 32;
  __m256i src0, src1;
  __m256i tmp0, tmp1, tmp2, tmp3;
  __m256i reg0, reg1, reg2, reg3;
  __m256i dst0, dst1, dst2, dst3;

  for (x = 0; x < len; x++) {
    src0 = __lasx_xvld(src_argb4444, 0);
    src1 = __lasx_xvld(src_argb4444, 32);
    DUP4_ARG2(__lasx_xvandi_b, src0, 0x0F, src0, 0xF0, src1, 0x0F, src1, 0xF0,
              tmp0, tmp1, tmp2, tmp3);
    DUP2_ARG2(__lasx_xvslli_b, tmp0, 4, tmp2, 4, reg0, reg2);
    DUP2_ARG2(__lasx_xvsrli_b, tmp1, 4, tmp3, 4, reg1, reg3);
    DUP4_ARG2(__lasx_xvor_v, tmp0, reg0, tmp1, reg1, tmp2, reg2, tmp3, reg3,
              tmp0, tmp1, tmp2, tmp3);
    DUP2_ARG2(__lasx_xvilvl_b, tmp1, tmp0, tmp3, tmp2, reg0, reg2);
    DUP2_ARG2(__lasx_xvilvh_b, tmp1, tmp0, tmp3, tmp2, reg1, reg3);
    DUP4_ARG3(__lasx_xvpermi_q, reg1, reg0, 0x20, reg1, reg0, 0x31, reg3, reg2,
              0x20, reg3, reg2, 0x31, dst0, dst1, dst2, dst3);
    __lasx_xvst(dst0, dst_argb, 0);
    __lasx_xvst(dst1, dst_argb, 32);
    __lasx_xvst(dst2, dst_argb, 64);
    __lasx_xvst(dst3, dst_argb, 96);
    src_argb4444 += 64;
    dst_argb += 128;
  }
}

void ARGB1555ToARGBRow_LASX(const uint8_t* src_argb1555,
                            uint8_t* dst_argb,
                            int width) {
  int x;
  int len = width / 32;
  __m256i src0, src1;
  __m256i tmp0, tmp1, tmpb, tmpg, tmpr, tmpa;
  __m256i reg0, reg1, reg2, reg3;
  __m256i dst0, dst1, dst2, dst3;

  for (x = 0; x < len; x++) {
    src0 = __lasx_xvld(src_argb1555, 0);
    src1 = __lasx_xvld(src_argb1555, 32);
    tmp0 = __lasx_xvpickev_b(src1, src0);
    tmp1 = __lasx_xvpickod_b(src1, src0);
    tmpb = __lasx_xvandi_b(tmp0, 0x1F);
    tmpg = __lasx_xvsrli_b(tmp0, 5);
    reg0 = __lasx_xvandi_b(tmp1, 0x03);
    reg0 = __lasx_xvslli_b(reg0, 3);
    tmpg = __lasx_xvor_v(tmpg, reg0);
    reg1 = __lasx_xvandi_b(tmp1, 0x7C);
    tmpr = __lasx_xvsrli_b(reg1, 2);
    tmpa = __lasx_xvsrli_b(tmp1, 7);
    tmpa = __lasx_xvneg_b(tmpa);
    reg0 = __lasx_xvslli_b(tmpb, 3);
    reg1 = __lasx_xvslli_b(tmpg, 3);
    reg2 = __lasx_xvslli_b(tmpr, 3);
    tmpb = __lasx_xvsrli_b(tmpb, 2);
    tmpg = __lasx_xvsrli_b(tmpg, 2);
    tmpr = __lasx_xvsrli_b(tmpr, 2);
    tmpb = __lasx_xvor_v(reg0, tmpb);
    tmpg = __lasx_xvor_v(reg1, tmpg);
    tmpr = __lasx_xvor_v(reg2, tmpr);
    DUP2_ARG2(__lasx_xvilvl_b, tmpg, tmpb, tmpa, tmpr, reg0, reg1);
    DUP2_ARG2(__lasx_xvilvh_b, tmpg, tmpb, tmpa, tmpr, reg2, reg3);
    dst0 = __lasx_xvilvl_h(reg1, reg0);
    dst1 = __lasx_xvilvh_h(reg1, reg0);
    dst2 = __lasx_xvilvl_h(reg3, reg2);
    dst3 = __lasx_xvilvh_h(reg3, reg2);
    DUP4_ARG3(__lasx_xvpermi_q, dst1, dst0, 0x20, dst1, dst0, 0x31, dst3, dst2,
              0x20, dst3, dst2, 0x31, reg0, reg1, reg2, reg3);
    __lasx_xvst(reg0, dst_argb, 0);
    __lasx_xvst(reg1, dst_argb, 32);
    __lasx_xvst(reg2, dst_argb, 64);
    __lasx_xvst(reg3, dst_argb, 96);
    src_argb1555 += 64;
    dst_argb += 128;
  }
}

void RGB565ToARGBRow_LASX(const uint8_t* src_rgb565,
                          uint8_t* dst_argb,
                          int width) {
  int x;
  int len = width / 32;
  __m256i src0, src1;
  __m256i tmp0, tmp1, tmpb, tmpg, tmpr;
  __m256i reg0, reg1, reg2, reg3, dst0, dst1, dst2, dst3;
  __m256i alpha = __lasx_xvldi(0xFF);

  for (x = 0; x < len; x++) {
    src0 = __lasx_xvld(src_rgb565, 0);
    src1 = __lasx_xvld(src_rgb565, 32);
    tmp0 = __lasx_xvpickev_b(src1, src0);
    tmp1 = __lasx_xvpickod_b(src1, src0);
    tmpb = __lasx_xvandi_b(tmp0, 0x1F);
    tmpr = __lasx_xvandi_b(tmp1, 0xF8);
    reg1 = __lasx_xvandi_b(tmp1, 0x07);
    reg0 = __lasx_xvsrli_b(tmp0, 5);
    reg1 = __lasx_xvslli_b(reg1, 3);
    tmpg = __lasx_xvor_v(reg1, reg0);
    reg0 = __lasx_xvslli_b(tmpb, 3);
    reg1 = __lasx_xvsrli_b(tmpb, 2);
    tmpb = __lasx_xvor_v(reg1, reg0);
    reg0 = __lasx_xvslli_b(tmpg, 2);
    reg1 = __lasx_xvsrli_b(tmpg, 4);
    tmpg = __lasx_xvor_v(reg1, reg0);
    reg0 = __lasx_xvsrli_b(tmpr, 5);
    tmpr = __lasx_xvor_v(tmpr, reg0);
    DUP2_ARG2(__lasx_xvilvl_b, tmpg, tmpb, alpha, tmpr, reg0, reg1);
    dst0 = __lasx_xvilvl_h(reg1, reg0);
    dst1 = __lasx_xvilvh_h(reg1, reg0);
    DUP2_ARG2(__lasx_xvilvh_b, tmpg, tmpb, alpha, tmpr, reg0, reg1);
    dst2 = __lasx_xvilvl_h(reg1, reg0);
    dst3 = __lasx_xvilvh_h(reg1, reg0);
    DUP4_ARG3(__lasx_xvpermi_q, dst1, dst0, 0x20, dst1, dst0, 0x31, dst3, dst2,
              0x20, dst3, dst2, 0x31, reg0, reg1, reg2, reg3);
    __lasx_xvst(reg0, dst_argb, 0);
    __lasx_xvst(reg1, dst_argb, 32);
    __lasx_xvst(reg2, dst_argb, 64);
    __lasx_xvst(reg3, dst_argb, 96);
    src_rgb565 += 64;
    dst_argb += 128;
  }
}

void RGB24ToARGBRow_LASX(const uint8_t* src_rgb24,
                         uint8_t* dst_argb,
                         int width) {
  int x;
  int len = width / 32;
  __m256i src0, src1, src2;
  __m256i tmp0, tmp1, tmp2;
  __m256i dst0, dst1, dst2, dst3;
  __m256i reg0, reg1, reg2, reg3;
  __m256i alpha = __lasx_xvldi(0xFF);
  __m256i shuf0 = {0x131211100F0E0D0C, 0x1B1A191817161514, 0x131211100F0E0D0C,
                   0x1B1A191817161514};
  __m256i shuf1 = {0x1F1E1D1C1B1A1918, 0x0706050403020100, 0x1F1E1D1C1B1A1918,
                   0x0706050403020100};
  __m256i shuf2 = {0x0B0A090807060504, 0x131211100F0E0D0C, 0x0B0A090807060504,
                   0x131211100F0E0D0C};
  __m256i shuf3 = {0x1005040310020100, 0x100B0A0910080706, 0x1005040310020100,
                   0x100B0A0910080706};

  for (x = 0; x < len; x++) {
    reg0 = __lasx_xvld(src_rgb24, 0);
    reg1 = __lasx_xvld(src_rgb24, 32);
    reg2 = __lasx_xvld(src_rgb24, 64);
    src0 = __lasx_xvpermi_q(reg1, reg0, 0x30);
    src1 = __lasx_xvpermi_q(reg2, reg0, 0x21);
    src2 = __lasx_xvpermi_q(reg2, reg1, 0x30);
    DUP2_ARG3(__lasx_xvshuf_b, src1, src0, shuf0, src1, src2, shuf1, tmp0,
              tmp1);
    tmp2 = __lasx_xvshuf_b(src1, src2, shuf2);
    DUP4_ARG3(__lasx_xvshuf_b, alpha, src0, shuf3, alpha, tmp0, shuf3, alpha,
              tmp1, shuf3, alpha, tmp2, shuf3, reg0, reg1, reg2, reg3);
    DUP4_ARG3(__lasx_xvpermi_q, reg1, reg0, 0x20, reg3, reg2, 0x20, reg1, reg0,
              0x31, reg3, reg2, 0x31, dst0, dst1, dst2, dst3);
    __lasx_xvst(dst0, dst_argb, 0);
    __lasx_xvst(dst1, dst_argb, 32);
    __lasx_xvst(dst2, dst_argb, 64);
    __lasx_xvst(dst3, dst_argb, 96);
    src_rgb24 += 96;
    dst_argb += 128;
  }
}

void RAWToARGBRow_LASX(const uint8_t* src_raw, uint8_t* dst_argb, int width) {
  int x;
  int len = width / 32;
  __m256i src0, src1, src2;
  __m256i tmp0, tmp1, tmp2, reg0, reg1, reg2, reg3;
  __m256i dst0, dst1, dst2, dst3;
  __m256i alpha = __lasx_xvldi(0xFF);
  __m256i shuf0 = {0x131211100F0E0D0C, 0x1B1A191817161514, 0x131211100F0E0D0C,
                   0x1B1A191817161514};
  __m256i shuf1 = {0x1F1E1D1C1B1A1918, 0x0706050403020100, 0x1F1E1D1C1B1A1918,
                   0x0706050403020100};
  __m256i shuf2 = {0x0B0A090807060504, 0x131211100F0E0D0C, 0x0B0A090807060504,
                   0x131211100F0E0D0C};
  __m256i shuf3 = {0x1003040510000102, 0x10090A0B10060708, 0x1003040510000102,
                   0x10090A0B10060708};

  for (x = 0; x < len; x++) {
    reg0 = __lasx_xvld(src_raw, 0);
    reg1 = __lasx_xvld(src_raw, 32);
    reg2 = __lasx_xvld(src_raw, 64);
    src0 = __lasx_xvpermi_q(reg1, reg0, 0x30);
    src1 = __lasx_xvpermi_q(reg2, reg0, 0x21);
    src2 = __lasx_xvpermi_q(reg2, reg1, 0x30);
    DUP2_ARG3(__lasx_xvshuf_b, src1, src0, shuf0, src1, src2, shuf1, tmp0,
              tmp1);
    tmp2 = __lasx_xvshuf_b(src1, src2, shuf2);
    DUP4_ARG3(__lasx_xvshuf_b, alpha, src0, shuf3, alpha, tmp0, shuf3, alpha,
              tmp1, shuf3, alpha, tmp2, shuf3, reg0, reg1, reg2, reg3);
    DUP4_ARG3(__lasx_xvpermi_q, reg1, reg0, 0x20, reg3, reg2, 0x20, reg1, reg0,
              0x31, reg3, reg2, 0x31, dst0, dst1, dst2, dst3);
    __lasx_xvst(dst0, dst_argb, 0);
    __lasx_xvst(dst1, dst_argb, 32);
    __lasx_xvst(dst2, dst_argb, 64);
    __lasx_xvst(dst3, dst_argb, 96);
    src_raw += 96;
    dst_argb += 128;
  }
}

void ARGB1555ToYRow_LASX(const uint8_t* src_argb1555,
                         uint8_t* dst_y,
                         int width) {
  int x;
  int len = width / 32;
  __m256i src0, src1;
  __m256i tmp0, tmp1, tmpb, tmpg, tmpr;
  __m256i reg0, reg1, reg2, dst0;
  __m256i const_66 = __lasx_xvldi(66);
  __m256i const_129 = __lasx_xvldi(129);
  __m256i const_25 = __lasx_xvldi(25);
  __m256i const_1080 = {0x1080108010801080, 0x1080108010801080,
                        0x1080108010801080, 0x1080108010801080};

  for (x = 0; x < len; x++) {
    src0 = __lasx_xvld(src_argb1555, 0);
    src1 = __lasx_xvld(src_argb1555, 32);
    tmp0 = __lasx_xvpickev_b(src1, src0);
    tmp1 = __lasx_xvpickod_b(src1, src0);
    tmpb = __lasx_xvandi_b(tmp0, 0x1F);
    tmpg = __lasx_xvsrli_b(tmp0, 5);
    reg0 = __lasx_xvandi_b(tmp1, 0x03);
    reg0 = __lasx_xvslli_b(reg0, 3);
    tmpg = __lasx_xvor_v(tmpg, reg0);
    reg1 = __lasx_xvandi_b(tmp1, 0x7C);
    tmpr = __lasx_xvsrli_b(reg1, 2);
    reg0 = __lasx_xvslli_b(tmpb, 3);
    reg1 = __lasx_xvslli_b(tmpg, 3);
    reg2 = __lasx_xvslli_b(tmpr, 3);
    tmpb = __lasx_xvsrli_b(tmpb, 2);
    tmpg = __lasx_xvsrli_b(tmpg, 2);
    tmpr = __lasx_xvsrli_b(tmpr, 2);
    tmpb = __lasx_xvor_v(reg0, tmpb);
    tmpg = __lasx_xvor_v(reg1, tmpg);
    tmpr = __lasx_xvor_v(reg2, tmpr);
    reg0 = __lasx_xvmaddwev_h_bu(const_1080, tmpb, const_25);
    reg1 = __lasx_xvmaddwod_h_bu(const_1080, tmpb, const_25);
    reg0 = __lasx_xvmaddwev_h_bu(reg0, tmpg, const_129);
    reg1 = __lasx_xvmaddwod_h_bu(reg1, tmpg, const_129);
    reg0 = __lasx_xvmaddwev_h_bu(reg0, tmpr, const_66);
    reg1 = __lasx_xvmaddwod_h_bu(reg1, tmpr, const_66);
    dst0 = __lasx_xvpackod_b(reg1, reg0);
    dst0 = __lasx_xvpermi_d(dst0, 0xD8);
    __lasx_xvst(dst0, dst_y, 0);
    src_argb1555 += 64;
    dst_y += 32;
  }
}

void ARGB1555ToUVRow_LASX(const uint8_t* src_argb1555,
                          int src_stride_argb1555,
                          uint8_t* dst_u,
                          uint8_t* dst_v,
                          int width) {
  int x;
  int len = width / 32;
  const uint8_t* next_argb1555 = src_argb1555 + src_stride_argb1555;
  __m256i src0, src1, src2, src3;
  __m256i tmp0, tmp1, tmp2, tmp3;
  __m256i tmpb, tmpg, tmpr, nexb, nexg, nexr;
  __m256i reg0, reg1, reg2, reg3, dst0;
  __m256i const_112 = __lasx_xvldi(0x438);
  __m256i const_74 = __lasx_xvldi(0x425);
  __m256i const_38 = __lasx_xvldi(0x413);
  __m256i const_94 = __lasx_xvldi(0x42F);
  __m256i const_18 = __lasx_xvldi(0x409);
  __m256i const_8080 = (__m256i)v4u64{0x8080808080808080, 0x8080808080808080,
                                      0x8080808080808080, 0x8080808080808080};

  for (x = 0; x < len; x++) {
    DUP4_ARG2(__lasx_xvld, src_argb1555, 0, src_argb1555, 32, next_argb1555, 0,
              next_argb1555, 32, src0, src1, src2, src3);
    DUP2_ARG2(__lasx_xvpickev_b, src1, src0, src3, src2, tmp0, tmp2);
    DUP2_ARG2(__lasx_xvpickod_b, src1, src0, src3, src2, tmp1, tmp3);
    tmpb = __lasx_xvandi_b(tmp0, 0x1F);
    nexb = __lasx_xvandi_b(tmp2, 0x1F);
    tmpg = __lasx_xvsrli_b(tmp0, 5);
    nexg = __lasx_xvsrli_b(tmp2, 5);
    reg0 = __lasx_xvandi_b(tmp1, 0x03);
    reg2 = __lasx_xvandi_b(tmp3, 0x03);
    reg0 = __lasx_xvslli_b(reg0, 3);
    reg2 = __lasx_xvslli_b(reg2, 3);
    tmpg = __lasx_xvor_v(tmpg, reg0);
    nexg = __lasx_xvor_v(nexg, reg2);
    reg1 = __lasx_xvandi_b(tmp1, 0x7C);
    reg3 = __lasx_xvandi_b(tmp3, 0x7C);
    tmpr = __lasx_xvsrli_b(reg1, 2);
    nexr = __lasx_xvsrli_b(reg3, 2);
    reg0 = __lasx_xvslli_b(tmpb, 3);
    reg1 = __lasx_xvslli_b(tmpg, 3);
    reg2 = __lasx_xvslli_b(tmpr, 3);
    tmpb = __lasx_xvsrli_b(tmpb, 2);
    tmpg = __lasx_xvsrli_b(tmpg, 2);
    tmpr = __lasx_xvsrli_b(tmpr, 2);
    tmpb = __lasx_xvor_v(reg0, tmpb);
    tmpg = __lasx_xvor_v(reg1, tmpg);
    tmpr = __lasx_xvor_v(reg2, tmpr);
    reg0 = __lasx_xvslli_b(nexb, 3);
    reg1 = __lasx_xvslli_b(nexg, 3);
    reg2 = __lasx_xvslli_b(nexr, 3);
    nexb = __lasx_xvsrli_b(nexb, 2);
    nexg = __lasx_xvsrli_b(nexg, 2);
    nexr = __lasx_xvsrli_b(nexr, 2);
    nexb = __lasx_xvor_v(reg0, nexb);
    nexg = __lasx_xvor_v(reg1, nexg);
    nexr = __lasx_xvor_v(reg2, nexr);
    RGBTOUV(tmpb, tmpg, tmpr, nexb, nexg, nexr, reg0, reg1);
    reg0 = __lasx_xvpermi_d(reg0, 0xD8);
    reg1 = __lasx_xvpermi_d(reg1, 0xD8);
    dst0 = __lasx_xvpickod_b(reg1, reg0);
    __lasx_xvstelm_d(dst0, dst_u, 0, 0);
    __lasx_xvstelm_d(dst0, dst_v, 0, 1);
    __lasx_xvstelm_d(dst0, dst_u, 8, 2);
    __lasx_xvstelm_d(dst0, dst_v, 8, 3);
    src_argb1555 += 64;
    next_argb1555 += 64;
    dst_u += 16;
    dst_v += 16;
  }
}

void RGB565ToYRow_LASX(const uint8_t* src_rgb565, uint8_t* dst_y, int width) {
  int x;
  int len = width / 32;
  __m256i src0, src1;
  __m256i tmp0, tmp1, tmpb, tmpg, tmpr;
  __m256i reg0, reg1, dst0;
  __m256i const_66 = __lasx_xvldi(66);
  __m256i const_129 = __lasx_xvldi(129);
  __m256i const_25 = __lasx_xvldi(25);
  __m256i const_1080 = {0x1080108010801080, 0x1080108010801080,
                        0x1080108010801080, 0x1080108010801080};

  for (x = 0; x < len; x++) {
    src0 = __lasx_xvld(src_rgb565, 0);
    src1 = __lasx_xvld(src_rgb565, 32);
    tmp0 = __lasx_xvpickev_b(src1, src0);
    tmp1 = __lasx_xvpickod_b(src1, src0);
    tmpb = __lasx_xvandi_b(tmp0, 0x1F);
    tmpr = __lasx_xvandi_b(tmp1, 0xF8);
    reg1 = __lasx_xvandi_b(tmp1, 0x07);
    reg0 = __lasx_xvsrli_b(tmp0, 5);
    reg1 = __lasx_xvslli_b(reg1, 3);
    tmpg = __lasx_xvor_v(reg1, reg0);
    reg0 = __lasx_xvslli_b(tmpb, 3);
    reg1 = __lasx_xvsrli_b(tmpb, 2);
    tmpb = __lasx_xvor_v(reg1, reg0);
    reg0 = __lasx_xvslli_b(tmpg, 2);
    reg1 = __lasx_xvsrli_b(tmpg, 4);
    tmpg = __lasx_xvor_v(reg1, reg0);
    reg0 = __lasx_xvsrli_b(tmpr, 5);
    tmpr = __lasx_xvor_v(tmpr, reg0);
    reg0 = __lasx_xvmaddwev_h_bu(const_1080, tmpb, const_25);
    reg1 = __lasx_xvmaddwod_h_bu(const_1080, tmpb, const_25);
    reg0 = __lasx_xvmaddwev_h_bu(reg0, tmpg, const_129);
    reg1 = __lasx_xvmaddwod_h_bu(reg1, tmpg, const_129);
    reg0 = __lasx_xvmaddwev_h_bu(reg0, tmpr, const_66);
    reg1 = __lasx_xvmaddwod_h_bu(reg1, tmpr, const_66);
    dst0 = __lasx_xvpackod_b(reg1, reg0);
    dst0 = __lasx_xvpermi_d(dst0, 0xD8);
    __lasx_xvst(dst0, dst_y, 0);
    dst_y += 32;
    src_rgb565 += 64;
  }
}

void RGB565ToUVRow_LASX(const uint8_t* src_rgb565,
                        int src_stride_rgb565,
                        uint8_t* dst_u,
                        uint8_t* dst_v,
                        int width) {
  int x;
  int len = width / 32;
  const uint8_t* next_rgb565 = src_rgb565 + src_stride_rgb565;
  __m256i src0, src1, src2, src3;
  __m256i tmp0, tmp1, tmp2, tmp3;
  __m256i tmpb, tmpg, tmpr, nexb, nexg, nexr;
  __m256i reg0, reg1, reg2, reg3, dst0;
  __m256i const_112 = __lasx_xvldi(0x438);
  __m256i const_74 = __lasx_xvldi(0x425);
  __m256i const_38 = __lasx_xvldi(0x413);
  __m256i const_94 = __lasx_xvldi(0x42F);
  __m256i const_18 = __lasx_xvldi(0x409);
  __m256i const_8080 = (__m256i)v4u64{0x8080808080808080, 0x8080808080808080,
                                      0x8080808080808080, 0x8080808080808080};

  for (x = 0; x < len; x++) {
    DUP4_ARG2(__lasx_xvld, src_rgb565, 0, src_rgb565, 32, next_rgb565, 0,
              next_rgb565, 32, src0, src1, src2, src3);
    DUP2_ARG2(__lasx_xvpickev_b, src1, src0, src3, src2, tmp0, tmp2);
    DUP2_ARG2(__lasx_xvpickod_b, src1, src0, src3, src2, tmp1, tmp3);
    tmpb = __lasx_xvandi_b(tmp0, 0x1F);
    tmpr = __lasx_xvandi_b(tmp1, 0xF8);
    nexb = __lasx_xvandi_b(tmp2, 0x1F);
    nexr = __lasx_xvandi_b(tmp3, 0xF8);
    reg1 = __lasx_xvandi_b(tmp1, 0x07);
    reg3 = __lasx_xvandi_b(tmp3, 0x07);
    reg0 = __lasx_xvsrli_b(tmp0, 5);
    reg1 = __lasx_xvslli_b(reg1, 3);
    reg2 = __lasx_xvsrli_b(tmp2, 5);
    reg3 = __lasx_xvslli_b(reg3, 3);
    tmpg = __lasx_xvor_v(reg1, reg0);
    nexg = __lasx_xvor_v(reg2, reg3);
    reg0 = __lasx_xvslli_b(tmpb, 3);
    reg1 = __lasx_xvsrli_b(tmpb, 2);
    reg2 = __lasx_xvslli_b(nexb, 3);
    reg3 = __lasx_xvsrli_b(nexb, 2);
    tmpb = __lasx_xvor_v(reg1, reg0);
    nexb = __lasx_xvor_v(reg2, reg3);
    reg0 = __lasx_xvslli_b(tmpg, 2);
    reg1 = __lasx_xvsrli_b(tmpg, 4);
    reg2 = __lasx_xvslli_b(nexg, 2);
    reg3 = __lasx_xvsrli_b(nexg, 4);
    tmpg = __lasx_xvor_v(reg1, reg0);
    nexg = __lasx_xvor_v(reg2, reg3);
    reg0 = __lasx_xvsrli_b(tmpr, 5);
    reg2 = __lasx_xvsrli_b(nexr, 5);
    tmpr = __lasx_xvor_v(tmpr, reg0);
    nexr = __lasx_xvor_v(nexr, reg2);
    RGBTOUV(tmpb, tmpg, tmpr, nexb, nexg, nexr, reg0, reg1);
    reg0 = __lasx_xvpermi_d(reg0, 0xD8);
    reg1 = __lasx_xvpermi_d(reg1, 0xD8);
    dst0 = __lasx_xvpickod_b(reg1, reg0);
    __lasx_xvstelm_d(dst0, dst_u, 0, 0);
    __lasx_xvstelm_d(dst0, dst_v, 0, 1);
    __lasx_xvstelm_d(dst0, dst_u, 8, 2);
    __lasx_xvstelm_d(dst0, dst_v, 8, 3);
    dst_u += 16;
    dst_v += 16;
    src_rgb565 += 64;
    next_rgb565 += 64;
  }
}

void RGB24ToUVRow_LASX(const uint8_t* src_rgb24,
                       int src_stride_rgb24,
                       uint8_t* dst_u,
                       uint8_t* dst_v,
                       int width) {
  int x;
  const uint8_t* next_rgb24 = src_rgb24 + src_stride_rgb24;
  int len = width / 32;
  __m256i src0, src1, src2, reg0, reg1, reg2;
  __m256i nex0, nex1, nex2, dst0, tmp0, tmp1, tmp2;
  __m256i tmpb, tmpg, tmpr, nexb, nexg, nexr;
  __m256i const_112 = __lasx_xvldi(0x438);
  __m256i const_74 = __lasx_xvldi(0x425);
  __m256i const_38 = __lasx_xvldi(0x413);
  __m256i const_94 = __lasx_xvldi(0x42F);
  __m256i const_18 = __lasx_xvldi(0x409);
  __m256i const_8080 = (__m256i)v4u64{0x8080808080808080, 0x8080808080808080,
                                      0x8080808080808080, 0x8080808080808080};
  __m256i shuff0_b = {0x15120F0C09060300, 0x00000000001E1B18,
                      0x15120F0C09060300, 0x00000000001E1B18};
  __m256i shuff1_b = {0x0706050403020100, 0x1D1A1714110A0908,
                      0x0706050403020100, 0x1D1A1714110A0908};
  __m256i shuff0_g = {0x1613100D0A070401, 0x00000000001F1C19,
                      0x1613100D0A070401, 0x00000000001F1C19};
  __m256i shuff1_g = {0x0706050403020100, 0x1E1B1815120A0908,
                      0x0706050403020100, 0x1E1B1815120A0908};
  __m256i shuff0_r = {0x1714110E0B080502, 0x0000000000001D1A,
                      0x1714110E0B080502, 0x0000000000001D1A};
  __m256i shuff1_r = {0x0706050403020100, 0x1F1C191613100908,
                      0x0706050403020100, 0x1F1C191613100908};

  for (x = 0; x < len; x++) {
    DUP4_ARG2(__lasx_xvld, src_rgb24, 0, src_rgb24, 32, src_rgb24, 64,
              next_rgb24, 0, reg0, reg1, reg2, tmp0);
    DUP2_ARG2(__lasx_xvld, next_rgb24, 32, next_rgb24, 64, tmp1, tmp2);
    DUP4_ARG3(__lasx_xvpermi_q, reg1, reg0, 0x30, reg2, reg0, 0x21, reg2, reg1,
              0x30, tmp1, tmp0, 0x30, src0, src1, src2, nex0);
    DUP2_ARG3(__lasx_xvpermi_q, tmp2, tmp0, 0x21, tmp2, tmp1, 0x30, nex1, nex2);
    DUP2_ARG3(__lasx_xvshuf_b, src1, src0, shuff0_b, nex1, nex0, shuff0_b, tmpb,
              nexb);
    DUP2_ARG3(__lasx_xvshuf_b, src1, src0, shuff0_g, nex1, nex0, shuff0_g, tmpg,
              nexg);
    DUP2_ARG3(__lasx_xvshuf_b, src1, src0, shuff0_r, nex1, nex0, shuff0_r, tmpr,
              nexr);
    DUP2_ARG3(__lasx_xvshuf_b, src2, tmpb, shuff1_b, nex2, nexb, shuff1_b, tmpb,
              nexb);
    DUP2_ARG3(__lasx_xvshuf_b, src2, tmpg, shuff1_g, nex2, nexg, shuff1_g, tmpg,
              nexg);
    DUP2_ARG3(__lasx_xvshuf_b, src2, tmpr, shuff1_r, nex2, nexr, shuff1_r, tmpr,
              nexr);
    RGBTOUV(tmpb, tmpg, tmpr, nexb, nexg, nexr, reg0, reg1);
    dst0 = __lasx_xvpickod_b(reg1, reg0);
    __lasx_xvstelm_d(dst0, dst_u, 0, 0);
    __lasx_xvstelm_d(dst0, dst_v, 0, 1);
    __lasx_xvstelm_d(dst0, dst_u, 8, 2);
    __lasx_xvstelm_d(dst0, dst_v, 8, 3);
    src_rgb24 += 96;
    next_rgb24 += 96;
    dst_u += 16;
    dst_v += 16;
  }
}

void RAWToUVRow_LASX(const uint8_t* src_raw,
                     int src_stride_raw,
                     uint8_t* dst_u,
                     uint8_t* dst_v,
                     int width) {
  int x;
  const uint8_t* next_raw = src_raw + src_stride_raw;
  int len = width / 32;
  __m256i src0, src1, src2, reg0, reg1, reg2;
  __m256i nex0, nex1, nex2, dst0, tmp0, tmp1, tmp2;
  __m256i tmpb, tmpg, tmpr, nexb, nexg, nexr;
  __m256i const_112 = __lasx_xvldi(0x438);
  __m256i const_74 = __lasx_xvldi(0x425);
  __m256i const_38 = __lasx_xvldi(0x413);
  __m256i const_94 = __lasx_xvldi(0x42F);
  __m256i const_18 = __lasx_xvldi(0x409);
  __m256i const_8080 = (__m256i)v4u64{0x8080808080808080, 0x8080808080808080,
                                      0x8080808080808080, 0x8080808080808080};
  __m256i shuff0_r = {0x15120F0C09060300, 0x00000000001E1B18,
                      0x15120F0C09060300, 0x00000000001E1B18};
  __m256i shuff1_r = {0x0706050403020100, 0x1D1A1714110A0908,
                      0x0706050403020100, 0x1D1A1714110A0908};
  __m256i shuff0_g = {0x1613100D0A070401, 0x00000000001F1C19,
                      0x1613100D0A070401, 0x00000000001F1C19};
  __m256i shuff1_g = {0x0706050403020100, 0x1E1B1815120A0908,
                      0x0706050403020100, 0x1E1B1815120A0908};
  __m256i shuff0_b = {0x1714110E0B080502, 0x0000000000001D1A,
                      0x1714110E0B080502, 0x0000000000001D1A};
  __m256i shuff1_b = {0x0706050403020100, 0x1F1C191613100908,
                      0x0706050403020100, 0x1F1C191613100908};

  for (x = 0; x < len; x++) {
    DUP4_ARG2(__lasx_xvld, src_raw, 0, src_raw, 32, src_raw, 64, next_raw, 0,
              reg0, reg1, reg2, tmp0);
    DUP2_ARG2(__lasx_xvld, next_raw, 32, next_raw, 64, tmp1, tmp2);
    DUP4_ARG3(__lasx_xvpermi_q, reg1, reg0, 0x30, reg2, reg0, 0x21, reg2, reg1,
              0x30, tmp1, tmp0, 0x30, src0, src1, src2, nex0);
    DUP2_ARG3(__lasx_xvpermi_q, tmp2, tmp0, 0x21, tmp2, tmp1, 0x30, nex1, nex2);
    DUP2_ARG3(__lasx_xvshuf_b, src1, src0, shuff0_b, nex1, nex0, shuff0_b, tmpb,
              nexb);
    DUP2_ARG3(__lasx_xvshuf_b, src1, src0, shuff0_g, nex1, nex0, shuff0_g, tmpg,
              nexg);
    DUP2_ARG3(__lasx_xvshuf_b, src1, src0, shuff0_r, nex1, nex0, shuff0_r, tmpr,
              nexr);
    DUP2_ARG3(__lasx_xvshuf_b, src2, tmpb, shuff1_b, nex2, nexb, shuff1_b, tmpb,
              nexb);
    DUP2_ARG3(__lasx_xvshuf_b, src2, tmpg, shuff1_g, nex2, nexg, shuff1_g, tmpg,
              nexg);
    DUP2_ARG3(__lasx_xvshuf_b, src2, tmpr, shuff1_r, nex2, nexr, shuff1_r, tmpr,
              nexr);
    RGBTOUV(tmpb, tmpg, tmpr, nexb, nexg, nexr, reg0, reg1);
    dst0 = __lasx_xvpickod_b(reg1, reg0);
    __lasx_xvstelm_d(dst0, dst_u, 0, 0);
    __lasx_xvstelm_d(dst0, dst_v, 0, 1);
    __lasx_xvstelm_d(dst0, dst_u, 8, 2);
    __lasx_xvstelm_d(dst0, dst_v, 8, 3);
    src_raw += 96;
    next_raw += 96;
    dst_u += 16;
    dst_v += 16;
  }
}

void NV12ToARGBRow_LASX(const uint8_t* src_y,
                        const uint8_t* src_uv,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  int x;
  int len = width / 16;
  __m256i vec_yg, vec_yb, vec_ub, vec_vr, vec_ug, vec_vg;
  __m256i vec_vrub, vec_vgug, vec_y, vec_vu;
  __m256i out_b, out_g, out_r;
  __m256i const_0x80 = __lasx_xvldi(0x80);
  __m256i alpha = __lasx_xvldi(0xFF);

  YUVTORGB_SETUP(yuvconstants, vec_ub, vec_vr, vec_ug, vec_vg, vec_yg, vec_yb);
  vec_vrub = __lasx_xvilvl_h(vec_vr, vec_ub);
  vec_vgug = __lasx_xvilvl_h(vec_vg, vec_ug);

  for (x = 0; x < len; x++) {
    vec_y = __lasx_xvld(src_y, 0);
    vec_vu = __lasx_xvld(src_uv, 0);
    vec_vu = __lasx_xvsub_b(vec_vu, const_0x80);
    vec_vu = __lasx_vext2xv_h_b(vec_vu);
    YUVTORGB(vec_y, vec_vu, vec_vrub, vec_vgug, vec_yg, vec_yb, out_r, out_g,
             out_b);
    STOREARGB(alpha, out_r, out_g, out_b, dst_argb);
    src_y += 16;
    src_uv += 16;
  }
}

void NV12ToRGB565Row_LASX(const uint8_t* src_y,
                          const uint8_t* src_uv,
                          uint8_t* dst_rgb565,
                          const struct YuvConstants* yuvconstants,
                          int width) {
  int x;
  int len = width / 16;
  __m256i vec_yg, vec_yb, vec_ub, vec_vr, vec_ug, vec_vg;
  __m256i vec_vrub, vec_vgug, vec_y, vec_vu;
  __m256i out_b, out_g, out_r;
  __m256i const_0x80 = __lasx_xvldi(0x80);

  YUVTORGB_SETUP(yuvconstants, vec_ub, vec_vr, vec_ug, vec_vg, vec_yg, vec_yb);
  vec_vrub = __lasx_xvilvl_h(vec_vr, vec_ub);
  vec_vgug = __lasx_xvilvl_h(vec_vg, vec_ug);

  for (x = 0; x < len; x++) {
    vec_y = __lasx_xvld(src_y, 0);
    vec_vu = __lasx_xvld(src_uv, 0);
    vec_vu = __lasx_xvsub_b(vec_vu, const_0x80);
    vec_vu = __lasx_vext2xv_h_b(vec_vu);
    YUVTORGB(vec_y, vec_vu, vec_vrub, vec_vgug, vec_yg, vec_yb, out_r, out_g,
             out_b);
    out_b = __lasx_xvsrli_h(out_b, 3);
    out_g = __lasx_xvsrli_h(out_g, 2);
    out_r = __lasx_xvsrli_h(out_r, 3);
    out_g = __lasx_xvslli_h(out_g, 5);
    out_r = __lasx_xvslli_h(out_r, 11);
    out_r = __lasx_xvor_v(out_r, out_g);
    out_r = __lasx_xvor_v(out_r, out_b);
    __lasx_xvst(out_r, dst_rgb565, 0);
    src_y += 16;
    src_uv += 16;
    dst_rgb565 += 32;
  }
}

void NV21ToARGBRow_LASX(const uint8_t* src_y,
                        const uint8_t* src_uv,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  int x;
  int len = width / 16;
  __m256i vec_yg, vec_yb, vec_ub, vec_vr, vec_ug, vec_vg;
  __m256i vec_ubvr, vec_ugvg, vec_y, vec_uv;
  __m256i out_b, out_g, out_r;
  __m256i const_0x80 = __lasx_xvldi(0x80);
  __m256i alpha = __lasx_xvldi(0xFF);

  YUVTORGB_SETUP(yuvconstants, vec_ub, vec_vr, vec_ug, vec_vg, vec_yg, vec_yb);
  vec_ubvr = __lasx_xvilvl_h(vec_ub, vec_vr);
  vec_ugvg = __lasx_xvilvl_h(vec_ug, vec_vg);

  for (x = 0; x < len; x++) {
    vec_y = __lasx_xvld(src_y, 0);
    vec_uv = __lasx_xvld(src_uv, 0);
    vec_uv = __lasx_xvsub_b(vec_uv, const_0x80);
    vec_uv = __lasx_vext2xv_h_b(vec_uv);
    YUVTORGB(vec_y, vec_uv, vec_ubvr, vec_ugvg, vec_yg, vec_yb, out_b, out_g,
             out_r);
    STOREARGB(alpha, out_r, out_g, out_b, dst_argb);
    src_y += 16;
    src_uv += 16;
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
static void ARGBToYMatrixRow_LASX(const uint8_t* src_argb,
                                  uint8_t* dst_y,
                                  int width,
                                  const struct RgbConstants* rgbconstants) {
  int32_t shuff[8] = {0, 4, 1, 5, 2, 6, 3, 7};
  asm volatile(
      "xvldrepl.b      $xr0,  %3,    0             \n\t"  // load rgbconstants
      "xvldrepl.b      $xr1,  %3,    1             \n\t"  // load rgbconstants
      "xvldrepl.b      $xr2,  %3,    2             \n\t"  // load rgbconstants
      "xvldrepl.h      $xr3,  %3,    4             \n\t"  // load rgbconstants
      "xvld            $xr20, %4,    0             \n\t"  // load shuff
      "1:                                          \n\t"
      "xvld            $xr4,  %0,    0             \n\t"
      "xvld            $xr5,  %0,    32            \n\t"
      "xvld            $xr6,  %0,    64            \n\t"
      "xvld            $xr7,  %0,    96            \n\t"  // load 32 pixels of
                                                          // ARGB
      "xvor.v          $xr12, $xr3,  $xr3          \n\t"
      "xvor.v          $xr13, $xr3,  $xr3          \n\t"
      "addi.d          %2,    %2,    -32           \n\t"  // 32 processed per
                                                          // loop.
      "xvpickev.b      $xr8,  $xr5,  $xr4          \n\t"  // BR
      "xvpickev.b      $xr10, $xr7,  $xr6          \n\t"
      "xvpickod.b      $xr9,  $xr5,  $xr4          \n\t"  // GA
      "xvpickod.b      $xr11, $xr7,  $xr6          \n\t"
      "xvmaddwev.h.bu  $xr12, $xr8,  $xr0          \n\t"  // B
      "xvmaddwev.h.bu  $xr13, $xr10, $xr0          \n\t"
      "xvmaddwev.h.bu  $xr12, $xr9,  $xr1          \n\t"  // G
      "xvmaddwev.h.bu  $xr13, $xr11, $xr1          \n\t"
      "xvmaddwod.h.bu  $xr12, $xr8,  $xr2          \n\t"  // R
      "xvmaddwod.h.bu  $xr13, $xr10, $xr2          \n\t"
      "addi.d          %0,    %0,    128           \n\t"
      "xvpickod.b      $xr10, $xr13, $xr12         \n\t"
      "xvperm.w        $xr11, $xr10, $xr20         \n\t"
      "xvst            $xr11, %1,    0             \n\t"
      "addi.d          %1,    %1,    32            \n\t"
      "bnez            %2,    1b                   \n\t"
      : "+&r"(src_argb),  // %0
        "+&r"(dst_y),     // %1
        "+&r"(width)      // %2
      : "r"(rgbconstants), "r"(shuff)
      : "memory");
}

void ARGBToYRow_LASX(const uint8_t* src_argb, uint8_t* dst_y, int width) {
  ARGBToYMatrixRow_LASX(src_argb, dst_y, width, &kRgb24I601Constants);
}

void ARGBToYJRow_LASX(const uint8_t* src_argb, uint8_t* dst_yj, int width) {
  ARGBToYMatrixRow_LASX(src_argb, dst_yj, width, &kRgb24JPEGConstants);
}

void ABGRToYRow_LASX(const uint8_t* src_abgr, uint8_t* dst_y, int width) {
  ARGBToYMatrixRow_LASX(src_abgr, dst_y, width, &kRawI601Constants);
}

void ABGRToYJRow_LASX(const uint8_t* src_abgr, uint8_t* dst_yj, int width) {
  ARGBToYMatrixRow_LASX(src_abgr, dst_yj, width, &kRawJPEGConstants);
}

// RGBA expects first value to be A and ignored, then 3 values to contain RGB.
// Same code as ARGB, except the LD4
static void RGBAToYMatrixRow_LASX(const uint8_t* src_rgba,
                                  uint8_t* dst_y,
                                  int width,
                                  const struct RgbConstants* rgbconstants) {
  int32_t shuff[8] = {0, 4, 1, 5, 2, 6, 3, 7};
  asm volatile(
      "xvldrepl.b      $xr0,  %3,    0             \n\t"  // load rgbconstants
      "xvldrepl.b      $xr1,  %3,    1             \n\t"  // load rgbconstants
      "xvldrepl.b      $xr2,  %3,    2             \n\t"  // load rgbconstants
      "xvldrepl.h      $xr3,  %3,    4             \n\t"  // load rgbconstants
      "xvld            $xr20, %4,    0             \n\t"  // load shuff
      "1:                                          \n\t"
      "xvld            $xr4,  %0,    0             \n\t"
      "xvld            $xr5,  %0,    32            \n\t"
      "xvld            $xr6,  %0,    64            \n\t"
      "xvld            $xr7,  %0,    96            \n\t"  // load 32 pixels of
                                                          // RGBA
      "xvor.v          $xr12, $xr3,  $xr3          \n\t"
      "xvor.v          $xr13, $xr3,  $xr3          \n\t"
      "addi.d          %2,    %2,    -32           \n\t"  // 32 processed per
                                                          // loop.
      "xvpickev.b      $xr8,  $xr5,  $xr4          \n\t"  // AG
      "xvpickev.b      $xr10, $xr7,  $xr6          \n\t"
      "xvpickod.b      $xr9,  $xr5,  $xr4          \n\t"  // BR
      "xvpickod.b      $xr11, $xr7,  $xr6          \n\t"
      "xvmaddwev.h.bu  $xr12, $xr9,  $xr0          \n\t"  // B
      "xvmaddwev.h.bu  $xr13, $xr11, $xr0          \n\t"
      "xvmaddwod.h.bu  $xr12, $xr8,  $xr1          \n\t"  // G
      "xvmaddwod.h.bu  $xr13, $xr10, $xr1          \n\t"
      "xvmaddwod.h.bu  $xr12, $xr9,  $xr2          \n\t"  // R
      "xvmaddwod.h.bu  $xr13, $xr11, $xr2          \n\t"
      "addi.d          %0,    %0,    128           \n\t"
      "xvpickod.b      $xr10, $xr13, $xr12         \n\t"
      "xvperm.w        $xr11, $xr10, $xr20         \n\t"
      "xvst            $xr11, %1,    0             \n\t"
      "addi.d          %1,    %1,    32            \n\t"
      "bnez            %2,    1b                   \n\t"
      : "+&r"(src_rgba),  // %0
        "+&r"(dst_y),     // %1
        "+&r"(width)      // %2
      : "r"(rgbconstants), "r"(shuff)
      : "memory");
}

void RGBAToYRow_LASX(const uint8_t* src_rgba, uint8_t* dst_y, int width) {
  RGBAToYMatrixRow_LASX(src_rgba, dst_y, width, &kRgb24I601Constants);
}

void RGBAToYJRow_LASX(const uint8_t* src_rgba, uint8_t* dst_yj, int width) {
  RGBAToYMatrixRow_LASX(src_rgba, dst_yj, width, &kRgb24JPEGConstants);
}

void BGRAToYRow_LASX(const uint8_t* src_bgra, uint8_t* dst_y, int width) {
  RGBAToYMatrixRow_LASX(src_bgra, dst_y, width, &kRawI601Constants);
}

static void RGBToYMatrixRow_LASX(const uint8_t* src_rgba,
                                 uint8_t* dst_y,
                                 int width,
                                 const struct RgbConstants* rgbconstants) {
  int8_t shuff[128] = {
      0,  2,  3,  5,  6,  8, 9,  11, 12, 14, 15, 17, 18, 20, 21, 23,
      0,  2,  3,  5,  6,  8, 9,  11, 12, 14, 15, 17, 18, 20, 21, 23,
      24, 26, 27, 29, 30, 0, 1,  3,  4,  6,  7,  9,  10, 12, 13, 15,
      24, 26, 27, 29, 30, 0, 1,  3,  4,  6,  7,  9,  10, 12, 13, 15,
      1,  0,  4,  0,  7,  0, 10, 0,  13, 0,  16, 0,  19, 0,  22, 0,
      1,  0,  4,  0,  7,  0, 10, 0,  13, 0,  16, 0,  19, 0,  22, 0,
      25, 0,  28, 0,  31, 0, 2,  0,  5,  0,  8,  0,  11, 0,  14, 0,
      25, 0,  28, 0,  31, 0, 2,  0,  5,  0,  8,  0,  11, 0,  14, 0};
  asm volatile(
      "xvldrepl.b      $xr0,  %3,    0             \n\t"  // load rgbconstants
      "xvldrepl.b      $xr1,  %3,    1             \n\t"  // load rgbconstants
      "xvldrepl.b      $xr2,  %3,    2             \n\t"  // load rgbconstants
      "xvldrepl.h      $xr3,  %3,    4             \n\t"  // load rgbconstants
      "xvld            $xr4,  %4,    0             \n\t"  // load shuff
      "xvld            $xr5,  %4,    32            \n\t"
      "xvld            $xr6,  %4,    64            \n\t"
      "xvld            $xr7,  %4,    96            \n\t"
      "1:                                          \n\t"
      "xvld            $xr8,  %0,    0             \n\t"
      "xvld            $xr9,  %0,    32            \n\t"
      "xvld            $xr10, %0,    64            \n\t"  // load 32 pixels of
                                                          // RGB
      "xvor.v          $xr12, $xr3,  $xr3          \n\t"
      "xvor.v          $xr13, $xr3,  $xr3          \n\t"
      "xvor.v          $xr11, $xr9,  $xr9          \n\t"
      "addi.d          %2,    %2,    -32           \n\t"  // 32 processed per
                                                          // loop.
      "xvpermi.q       $xr9,  $xr8,  0x30          \n\t"  // src0
      "xvpermi.q       $xr8,  $xr10, 0x03          \n\t"  // src1
      "xvpermi.q       $xr10, $xr11, 0x30          \n\t"  // src2
      "xvshuf.b        $xr14, $xr8,  $xr9,  $xr4   \n\t"
      "xvshuf.b        $xr15, $xr8,  $xr10, $xr5   \n\t"
      "xvshuf.b        $xr16, $xr8,  $xr9,  $xr6   \n\t"
      "xvshuf.b        $xr17, $xr8,  $xr10, $xr7   \n\t"
      "xvmaddwev.h.bu  $xr12, $xr16, $xr1          \n\t"  // G
      "xvmaddwev.h.bu  $xr13, $xr17, $xr1          \n\t"
      "xvmaddwev.h.bu  $xr12, $xr14, $xr0          \n\t"  // B
      "xvmaddwev.h.bu  $xr13, $xr15, $xr0          \n\t"
      "xvmaddwod.h.bu  $xr12, $xr14, $xr2          \n\t"  // R
      "xvmaddwod.h.bu  $xr13, $xr15, $xr2          \n\t"
      "addi.d          %0,    %0,    96            \n\t"
      "xvpickod.b      $xr10, $xr13, $xr12         \n\t"
      "xvst            $xr10, %1,    0             \n\t"
      "addi.d          %1,    %1,    32            \n\t"
      "bnez            %2,    1b                   \n\t"
      : "+&r"(src_rgba),    // %0
        "+&r"(dst_y),       // %1
        "+&r"(width)        // %2
      : "r"(rgbconstants),  // %3
        "r"(shuff)          // %4
      : "memory");
}

void RGB24ToYJRow_LASX(const uint8_t* src_rgb24, uint8_t* dst_yj, int width) {
  RGBToYMatrixRow_LASX(src_rgb24, dst_yj, width, &kRgb24JPEGConstants);
}

void RAWToYJRow_LASX(const uint8_t* src_raw, uint8_t* dst_yj, int width) {
  RGBToYMatrixRow_LASX(src_raw, dst_yj, width, &kRawJPEGConstants);
}

void RGB24ToYRow_LASX(const uint8_t* src_rgb24, uint8_t* dst_y, int width) {
  RGBToYMatrixRow_LASX(src_rgb24, dst_y, width, &kRgb24I601Constants);
}

void RAWToYRow_LASX(const uint8_t* src_raw, uint8_t* dst_y, int width) {
  RGBToYMatrixRow_LASX(src_raw, dst_y, width, &kRawI601Constants);
}

void ARGBToUVJRow_LASX(const uint8_t* src_argb,
                       int src_stride_argb,
                       uint8_t* dst_u,
                       uint8_t* dst_v,
                       int width) {
  int x;
  const uint8_t* next_argb = src_argb + src_stride_argb;
  int len = width / 32;
  __m256i src0, src1, src2, src3;
  __m256i nex0, nex1, nex2, nex3;
  __m256i tmp0, tmp1, tmp2, tmp3;
  __m256i reg0, reg1, dst0;
  __m256i tmpb, tmpg, tmpr, nexb, nexg, nexr;
  __m256i const_63 = __lasx_xvldi(0x43F);
  __m256i const_42 = __lasx_xvldi(0x42A);
  __m256i const_21 = __lasx_xvldi(0x415);
  __m256i const_53 = __lasx_xvldi(0x435);
  __m256i const_10 = __lasx_xvldi(0x40A);
  __m256i const_8080 = (__m256i)v4u64{0x8080808080808080, 0x8080808080808080,
                                      0x8080808080808080, 0x8080808080808080};
  __m256i shuff = {0x1614060412100200, 0x1E1C0E0C1A180A08, 0x1715070513110301,
                   0x1F1D0F0D1B190B09};

  for (x = 0; x < len; x++) {
    DUP4_ARG2(__lasx_xvld, src_argb, 0, src_argb, 32, src_argb, 64, src_argb,
              96, src0, src1, src2, src3);
    DUP4_ARG2(__lasx_xvld, next_argb, 0, next_argb, 32, next_argb, 64,
              next_argb, 96, nex0, nex1, nex2, nex3);
    tmp0 = __lasx_xvpickev_b(src1, src0);
    tmp1 = __lasx_xvpickod_b(src1, src0);
    tmp2 = __lasx_xvpickev_b(src3, src2);
    tmp3 = __lasx_xvpickod_b(src3, src2);
    tmpr = __lasx_xvpickod_b(tmp2, tmp0);
    tmpb = __lasx_xvpickev_b(tmp2, tmp0);
    tmpg = __lasx_xvpickev_b(tmp3, tmp1);
    tmp0 = __lasx_xvpickev_b(nex1, nex0);
    tmp1 = __lasx_xvpickod_b(nex1, nex0);
    tmp2 = __lasx_xvpickev_b(nex3, nex2);
    tmp3 = __lasx_xvpickod_b(nex3, nex2);
    nexr = __lasx_xvpickod_b(tmp2, tmp0);
    nexb = __lasx_xvpickev_b(tmp2, tmp0);
    nexg = __lasx_xvpickev_b(tmp3, tmp1);
    tmp0 = __lasx_xvaddwev_h_bu(tmpb, nexb);
    tmp1 = __lasx_xvaddwod_h_bu(tmpb, nexb);
    tmp2 = __lasx_xvaddwev_h_bu(tmpg, nexg);
    tmp3 = __lasx_xvaddwod_h_bu(tmpg, nexg);
    reg0 = __lasx_xvaddwev_h_bu(tmpr, nexr);
    reg1 = __lasx_xvaddwod_h_bu(tmpr, nexr);
    tmpb = __lasx_xvavgr_hu(tmp0, tmp1);
    tmpg = __lasx_xvavgr_hu(tmp2, tmp3);
    tmpr = __lasx_xvavgr_hu(reg0, reg1);
    reg0 = __lasx_xvmadd_h(const_8080, const_63, tmpb);
    reg1 = __lasx_xvmadd_h(const_8080, const_63, tmpr);
    reg0 = __lasx_xvmsub_h(reg0, const_42, tmpg);
    reg1 = __lasx_xvmsub_h(reg1, const_53, tmpg);
    reg0 = __lasx_xvmsub_h(reg0, const_21, tmpr);
    reg1 = __lasx_xvmsub_h(reg1, const_10, tmpb);
    dst0 = __lasx_xvpackod_b(reg1, reg0);
    tmp0 = __lasx_xvpermi_d(dst0, 0x44);
    tmp1 = __lasx_xvpermi_d(dst0, 0xEE);
    dst0 = __lasx_xvshuf_b(tmp1, tmp0, shuff);
    __lasx_xvstelm_d(dst0, dst_u, 0, 0);
    __lasx_xvstelm_d(dst0, dst_v, 0, 2);
    __lasx_xvstelm_d(dst0, dst_u, 8, 1);
    __lasx_xvstelm_d(dst0, dst_v, 8, 3);
    dst_u += 16;
    dst_v += 16;
    src_argb += 128;
    next_argb += 128;
  }
}

// undef for unified sources build
#undef ALPHA_VAL
#undef YUVTORGB_SETUP
#undef READYUV422_D
#undef READYUV422
#undef YUVTORGB_D
#undef YUVTORGB
#undef STOREARGB_D
#undef STOREARGB
#undef RGBTOUV

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

#endif  // !defined(LIBYUV_DISABLE_LASX) && defined(__loongarch_asx)
