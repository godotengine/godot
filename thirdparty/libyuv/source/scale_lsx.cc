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

#include <assert.h>

#include "libyuv/scale_row.h"

#if !defined(LIBYUV_DISABLE_LSX) && defined(__loongarch_sx)
#include "libyuv/loongson_intrinsics.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

#define LOAD_DATA(_src, _in, _out)                                       \
  {                                                                      \
    int _tmp1, _tmp2, _tmp3, _tmp4;                                      \
    DUP4_ARG2(__lsx_vpickve2gr_w, _in, 0, _in, 1, _in, 2, _in, 3, _tmp1, \
              _tmp2, _tmp3, _tmp4);                                      \
    _out = __lsx_vinsgr2vr_w(_out, _src[_tmp1], 0);                      \
    _out = __lsx_vinsgr2vr_w(_out, _src[_tmp2], 1);                      \
    _out = __lsx_vinsgr2vr_w(_out, _src[_tmp3], 2);                      \
    _out = __lsx_vinsgr2vr_w(_out, _src[_tmp4], 3);                      \
  }

void ScaleARGBRowDown2_LSX(const uint8_t* src_argb,
                           ptrdiff_t src_stride,
                           uint8_t* dst_argb,
                           int dst_width) {
  int x;
  int len = dst_width / 4;
  (void)src_stride;
  __m128i src0, src1, dst0;

  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lsx_vld, src_argb, 0, src_argb, 16, src0, src1);
    dst0 = __lsx_vpickod_w(src1, src0);
    __lsx_vst(dst0, dst_argb, 0);
    src_argb += 32;
    dst_argb += 16;
  }
}

void ScaleARGBRowDown2Linear_LSX(const uint8_t* src_argb,
                                 ptrdiff_t src_stride,
                                 uint8_t* dst_argb,
                                 int dst_width) {
  int x;
  int len = dst_width / 4;
  (void)src_stride;
  __m128i src0, src1, tmp0, tmp1, dst0;

  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lsx_vld, src_argb, 0, src_argb, 16, src0, src1);
    tmp0 = __lsx_vpickev_w(src1, src0);
    tmp1 = __lsx_vpickod_w(src1, src0);
    dst0 = __lsx_vavgr_bu(tmp1, tmp0);
    __lsx_vst(dst0, dst_argb, 0);
    src_argb += 32;
    dst_argb += 16;
  }
}

void ScaleARGBRowDown2Box_LSX(const uint8_t* src_argb,
                              ptrdiff_t src_stride,
                              uint8_t* dst_argb,
                              int dst_width) {
  int x;
  int len = dst_width / 4;
  const uint8_t* s = src_argb;
  const uint8_t* t = src_argb + src_stride;
  __m128i src0, src1, src2, src3, tmp0, tmp1, tmp2, tmp3, dst0;
  __m128i reg0, reg1, reg2, reg3;
  __m128i shuff = {0x0703060205010400, 0x0F0B0E0A0D090C08};

  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lsx_vld, s, 0, s, 16, src0, src1);
    DUP2_ARG2(__lsx_vld, t, 0, t, 16, src2, src3);
    DUP4_ARG3(__lsx_vshuf_b, src0, src0, shuff, src1, src1, shuff, src2, src2,
              shuff, src3, src3, shuff, tmp0, tmp1, tmp2, tmp3);
    DUP4_ARG2(__lsx_vhaddw_hu_bu, tmp0, tmp0, tmp1, tmp1, tmp2, tmp2, tmp3,
              tmp3, reg0, reg1, reg2, reg3);
    DUP2_ARG2(__lsx_vsadd_hu, reg0, reg2, reg1, reg3, reg0, reg1);
    dst0 = __lsx_vsrarni_b_h(reg1, reg0, 2);
    __lsx_vst(dst0, dst_argb, 0);
    s += 32;
    t += 32;
    dst_argb += 16;
  }
}

void ScaleARGBRowDownEven_LSX(const uint8_t* src_argb,
                              ptrdiff_t src_stride,
                              int32_t src_stepx,
                              uint8_t* dst_argb,
                              int dst_width) {
  int x;
  int len = dst_width / 4;
  int32_t stepx = src_stepx << 2;
  (void)src_stride;
  __m128i dst0, dst1, dst2, dst3;

  for (x = 0; x < len; x++) {
    dst0 = __lsx_vldrepl_w(src_argb, 0);
    src_argb += stepx;
    dst1 = __lsx_vldrepl_w(src_argb, 0);
    src_argb += stepx;
    dst2 = __lsx_vldrepl_w(src_argb, 0);
    src_argb += stepx;
    dst3 = __lsx_vldrepl_w(src_argb, 0);
    src_argb += stepx;
    __lsx_vstelm_w(dst0, dst_argb, 0, 0);
    __lsx_vstelm_w(dst1, dst_argb, 4, 0);
    __lsx_vstelm_w(dst2, dst_argb, 8, 0);
    __lsx_vstelm_w(dst3, dst_argb, 12, 0);
    dst_argb += 16;
  }
}

void ScaleARGBRowDownEvenBox_LSX(const uint8_t* src_argb,
                                 ptrdiff_t src_stride,
                                 int src_stepx,
                                 uint8_t* dst_argb,
                                 int dst_width) {
  int x;
  int len = dst_width / 4;
  int32_t stepx = src_stepx * 4;
  const uint8_t* next_argb = src_argb + src_stride;
  __m128i src0, src1, src2, src3;
  __m128i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
  __m128i reg0, reg1, dst0;

  for (x = 0; x < len; x++) {
    tmp0 = __lsx_vldrepl_d(src_argb, 0);
    src_argb += stepx;
    tmp1 = __lsx_vldrepl_d(src_argb, 0);
    src_argb += stepx;
    tmp2 = __lsx_vldrepl_d(src_argb, 0);
    src_argb += stepx;
    tmp3 = __lsx_vldrepl_d(src_argb, 0);
    src_argb += stepx;
    tmp4 = __lsx_vldrepl_d(next_argb, 0);
    next_argb += stepx;
    tmp5 = __lsx_vldrepl_d(next_argb, 0);
    next_argb += stepx;
    tmp6 = __lsx_vldrepl_d(next_argb, 0);
    next_argb += stepx;
    tmp7 = __lsx_vldrepl_d(next_argb, 0);
    next_argb += stepx;
    DUP4_ARG2(__lsx_vilvl_d, tmp1, tmp0, tmp3, tmp2, tmp5, tmp4, tmp7, tmp6,
              src0, src1, src2, src3);
    DUP2_ARG2(__lsx_vaddwev_h_bu, src0, src2, src1, src3, tmp0, tmp2);
    DUP2_ARG2(__lsx_vaddwod_h_bu, src0, src2, src1, src3, tmp1, tmp3);
    DUP2_ARG2(__lsx_vpackev_w, tmp1, tmp0, tmp3, tmp2, reg0, reg1);
    DUP2_ARG2(__lsx_vpackod_w, tmp1, tmp0, tmp3, tmp2, tmp4, tmp5);
    DUP2_ARG2(__lsx_vadd_h, reg0, tmp4, reg1, tmp5, reg0, reg1);
    dst0 = __lsx_vsrarni_b_h(reg1, reg0, 2);
    dst0 = __lsx_vshuf4i_b(dst0, 0xD8);
    __lsx_vst(dst0, dst_argb, 0);
    dst_argb += 16;
  }
}

void ScaleRowDown2_LSX(const uint8_t* src_ptr,
                       ptrdiff_t src_stride,
                       uint8_t* dst,
                       int dst_width) {
  int x;
  int len = dst_width / 32;
  __m128i src0, src1, src2, src3, dst0, dst1;
  (void)src_stride;

  for (x = 0; x < len; x++) {
    DUP4_ARG2(__lsx_vld, src_ptr, 0, src_ptr, 16, src_ptr, 32, src_ptr, 48,
              src0, src1, src2, src3);
    DUP2_ARG2(__lsx_vpickod_b, src1, src0, src3, src2, dst0, dst1);
    __lsx_vst(dst0, dst, 0);
    __lsx_vst(dst1, dst, 16);
    src_ptr += 64;
    dst += 32;
  }
}

void ScaleRowDown2Linear_LSX(const uint8_t* src_ptr,
                             ptrdiff_t src_stride,
                             uint8_t* dst,
                             int dst_width) {
  int x;
  int len = dst_width / 32;
  __m128i src0, src1, src2, src3;
  __m128i tmp0, tmp1, tmp2, tmp3, dst0, dst1;
  (void)src_stride;

  for (x = 0; x < len; x++) {
    DUP4_ARG2(__lsx_vld, src_ptr, 0, src_ptr, 16, src_ptr, 32, src_ptr, 48,
              src0, src1, src2, src3);
    DUP2_ARG2(__lsx_vpickev_b, src1, src0, src3, src2, tmp0, tmp2);
    DUP2_ARG2(__lsx_vpickod_b, src1, src0, src3, src2, tmp1, tmp3);
    DUP2_ARG2(__lsx_vavgr_bu, tmp0, tmp1, tmp2, tmp3, dst0, dst1);
    __lsx_vst(dst0, dst, 0);
    __lsx_vst(dst1, dst, 16);
    src_ptr += 64;
    dst += 32;
  }
}

void ScaleRowDown2Box_LSX(const uint8_t* src_ptr,
                          ptrdiff_t src_stride,
                          uint8_t* dst,
                          int dst_width) {
  int x;
  int len = dst_width / 32;
  const uint8_t* src_nex = src_ptr + src_stride;
  __m128i src0, src1, src2, src3, src4, src5, src6, src7;
  __m128i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
  __m128i dst0, dst1;

  for (x = 0; x < len; x++) {
    DUP4_ARG2(__lsx_vld, src_ptr, 0, src_ptr, 16, src_ptr, 32, src_ptr, 48,
              src0, src1, src2, src3);
    DUP4_ARG2(__lsx_vld, src_nex, 0, src_nex, 16, src_nex, 32, src_nex, 48,
              src4, src5, src6, src7);
    DUP4_ARG2(__lsx_vaddwev_h_bu, src0, src4, src1, src5, src2, src6, src3,
              src7, tmp0, tmp2, tmp4, tmp6);
    DUP4_ARG2(__lsx_vaddwod_h_bu, src0, src4, src1, src5, src2, src6, src3,
              src7, tmp1, tmp3, tmp5, tmp7);
    DUP4_ARG2(__lsx_vadd_h, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7,
              tmp0, tmp1, tmp2, tmp3);
    DUP2_ARG3(__lsx_vsrarni_b_h, tmp1, tmp0, 2, tmp3, tmp2, 2, dst0, dst1);
    __lsx_vst(dst0, dst, 0);
    __lsx_vst(dst1, dst, 16);
    src_ptr += 64;
    src_nex += 64;
    dst += 32;
  }
}

void ScaleRowDown4_LSX(const uint8_t* src_ptr,
                       ptrdiff_t src_stride,
                       uint8_t* dst,
                       int dst_width) {
  int x;
  int len = dst_width / 16;
  __m128i src0, src1, src2, src3, tmp0, tmp1, dst0;
  (void)src_stride;

  for (x = 0; x < len; x++) {
    DUP4_ARG2(__lsx_vld, src_ptr, 0, src_ptr, 16, src_ptr, 32, src_ptr, 48,
              src0, src1, src2, src3);
    DUP2_ARG2(__lsx_vpickev_b, src1, src0, src3, src2, tmp0, tmp1);
    dst0 = __lsx_vpickod_b(tmp1, tmp0);
    __lsx_vst(dst0, dst, 0);
    src_ptr += 64;
    dst += 16;
  }
}

void ScaleRowDown4Box_LSX(const uint8_t* src_ptr,
                          ptrdiff_t src_stride,
                          uint8_t* dst,
                          int dst_width) {
  int x;
  int len = dst_width / 16;
  const uint8_t* ptr1 = src_ptr + src_stride;
  const uint8_t* ptr2 = ptr1 + src_stride;
  const uint8_t* ptr3 = ptr2 + src_stride;
  __m128i src0, src1, src2, src3, src4, src5, src6, src7;
  __m128i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
  __m128i reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, dst0;

  for (x = 0; x < len; x++) {
    DUP4_ARG2(__lsx_vld, src_ptr, 0, src_ptr, 16, src_ptr, 32, src_ptr, 48,
              src0, src1, src2, src3);
    DUP4_ARG2(__lsx_vld, ptr1, 0, ptr1, 16, ptr1, 32, ptr1, 48, src4, src5,
              src6, src7);
    DUP4_ARG2(__lsx_vaddwev_h_bu, src0, src4, src1, src5, src2, src6, src3,
              src7, tmp0, tmp2, tmp4, tmp6);
    DUP4_ARG2(__lsx_vaddwod_h_bu, src0, src4, src1, src5, src2, src6, src3,
              src7, tmp1, tmp3, tmp5, tmp7);
    DUP4_ARG2(__lsx_vadd_h, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7,
              reg0, reg1, reg2, reg3);
    DUP4_ARG2(__lsx_vld, ptr2, 0, ptr2, 16, ptr2, 32, ptr2, 48, src0, src1,
              src2, src3);
    DUP4_ARG2(__lsx_vld, ptr3, 0, ptr3, 16, ptr3, 32, ptr3, 48, src4, src5,
              src6, src7);
    DUP4_ARG2(__lsx_vaddwev_h_bu, src0, src4, src1, src5, src2, src6, src3,
              src7, tmp0, tmp2, tmp4, tmp6);
    DUP4_ARG2(__lsx_vaddwod_h_bu, src0, src4, src1, src5, src2, src6, src3,
              src7, tmp1, tmp3, tmp5, tmp7);
    DUP4_ARG2(__lsx_vadd_h, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7,
              reg4, reg5, reg6, reg7);
    DUP4_ARG2(__lsx_vadd_h, reg0, reg4, reg1, reg5, reg2, reg6, reg3, reg7,
              reg0, reg1, reg2, reg3);
    DUP4_ARG2(__lsx_vhaddw_wu_hu, reg0, reg0, reg1, reg1, reg2, reg2, reg3,
              reg3, reg0, reg1, reg2, reg3);
    DUP2_ARG3(__lsx_vsrarni_h_w, reg1, reg0, 4, reg3, reg2, 4, tmp0, tmp1);
    dst0 = __lsx_vpickev_b(tmp1, tmp0);
    __lsx_vst(dst0, dst, 0);
    src_ptr += 64;
    ptr1 += 64;
    ptr2 += 64;
    ptr3 += 64;
    dst += 16;
  }
}

void ScaleRowDown38_LSX(const uint8_t* src_ptr,
                        ptrdiff_t src_stride,
                        uint8_t* dst,
                        int dst_width) {
  int x, len;
  __m128i src0, src1, tmp0;
  __m128i shuff = {0x13100E0B08060300, 0x000000001E1B1816};

  assert(dst_width % 3 == 0);
  len = dst_width / 12;
  (void)src_stride;

  for (x = 0; x < len; x++) {
    DUP2_ARG2(__lsx_vld, src_ptr, 0, src_ptr, 16, src0, src1);
    tmp0 = __lsx_vshuf_b(src1, src0, shuff);
    __lsx_vstelm_d(tmp0, dst, 0, 0);
    __lsx_vstelm_w(tmp0, dst, 8, 2);
    src_ptr += 32;
    dst += 12;
  }
}

void ScaleRowDown38_2_Box_LSX(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_ptr,
                              int dst_width) {
  int x, len;
  const uint8_t* src_nex = src_ptr + src_stride;
  __m128i src0, src1, src2, src3, dst0;
  __m128i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
  __m128i reg0, reg1, reg2, reg3;
  __m128i shuff = {0x0A08160604120200, 0x000000001E0E0C1A};
  __m128i const_0x2AAA = __lsx_vreplgr2vr_h(0x2AAA);
  __m128i const_0x4000 = __lsx_vreplgr2vr_w(0x4000);

  assert((dst_width % 3 == 0) && (dst_width > 0));
  len = dst_width / 12;

  for (x = 0; x < len; x++) {
    DUP4_ARG2(__lsx_vld, src_ptr, 0, src_ptr, 16, src_nex, 0, src_nex, 16, src0,
              src1, src2, src3);
    DUP2_ARG2(__lsx_vaddwev_h_bu, src0, src2, src1, src3, tmp0, tmp2);
    DUP2_ARG2(__lsx_vaddwod_h_bu, src0, src2, src1, src3, tmp1, tmp3);
    DUP2_ARG2(__lsx_vpickev_h, tmp2, tmp0, tmp3, tmp1, reg0, reg1);
    DUP2_ARG2(__lsx_vpackod_h, tmp1, tmp0, tmp3, tmp2, reg2, reg3);
    tmp4 = __lsx_vpickev_w(reg3, reg2);
    tmp5 = __lsx_vadd_h(reg0, reg1);
    tmp6 = __lsx_vadd_h(tmp5, tmp4);
    tmp7 = __lsx_vmuh_h(tmp6, const_0x2AAA);
    tmp0 = __lsx_vpickod_w(reg3, reg2);
    tmp1 = __lsx_vhaddw_wu_hu(tmp0, tmp0);
    tmp2 = __lsx_vmul_w(tmp1, const_0x4000);
    dst0 = __lsx_vshuf_b(tmp2, tmp7, shuff);
    __lsx_vstelm_d(dst0, dst_ptr, 0, 0);
    __lsx_vstelm_w(dst0, dst_ptr, 8, 2);
    src_ptr += 32;
    src_nex += 32;
    dst_ptr += 12;
  }
}

void ScaleRowDown38_3_Box_LSX(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_ptr,
                              int dst_width) {
  int x, len;
  const uint8_t* ptr1 = src_ptr + src_stride;
  const uint8_t* ptr2 = ptr1 + src_stride;
  __m128i src0, src1, src2, src3, src4, src5;
  __m128i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
  __m128i reg0, reg1, reg2, reg3, dst0;
  __m128i zero = __lsx_vldi(0);
  __m128i shuff = {0x0A08160604120200, 0x000000001E0E0C1A};
  __m128i const_0x1C71 = __lsx_vreplgr2vr_h(0x1C71);
  __m128i const_0x2AAA = __lsx_vreplgr2vr_w(0x2AAA);

  assert((dst_width % 3 == 0) && (dst_width > 0));
  len = dst_width / 12;

  for (x = 0; x < len; x++) {
    DUP4_ARG2(__lsx_vld, src_ptr, 0, src_ptr, 16, ptr1, 0, ptr1, 16, src0, src1,
              src2, src3);
    DUP2_ARG2(__lsx_vld, ptr2, 0, ptr2, 16, src4, src5);
    DUP2_ARG2(__lsx_vaddwev_h_bu, src0, src2, src1, src3, tmp0, tmp2);
    DUP2_ARG2(__lsx_vaddwod_h_bu, src0, src2, src1, src3, tmp1, tmp3);
    DUP2_ARG2(__lsx_vpackev_b, zero, src4, zero, src5, tmp4, tmp6);
    DUP2_ARG2(__lsx_vpackod_b, zero, src4, zero, src5, tmp5, tmp7);
    DUP4_ARG2(__lsx_vadd_h, tmp0, tmp4, tmp1, tmp5, tmp2, tmp6, tmp3, tmp7,
              tmp0, tmp1, tmp2, tmp3);
    DUP2_ARG2(__lsx_vpickev_h, tmp2, tmp0, tmp3, tmp1, reg0, reg1);
    DUP2_ARG2(__lsx_vpackod_h, tmp1, tmp0, tmp3, tmp2, reg2, reg3);
    tmp4 = __lsx_vpickev_w(reg3, reg2);
    tmp5 = __lsx_vadd_h(reg0, reg1);
    tmp6 = __lsx_vadd_h(tmp5, tmp4);
    tmp7 = __lsx_vmuh_h(tmp6, const_0x1C71);
    tmp0 = __lsx_vpickod_w(reg3, reg2);
    tmp1 = __lsx_vhaddw_wu_hu(tmp0, tmp0);
    tmp2 = __lsx_vmul_w(tmp1, const_0x2AAA);
    dst0 = __lsx_vshuf_b(tmp2, tmp7, shuff);
    __lsx_vstelm_d(dst0, dst_ptr, 0, 0);
    __lsx_vstelm_w(dst0, dst_ptr, 8, 2);
    src_ptr += 32;
    ptr1 += 32;
    ptr2 += 32;
    dst_ptr += 12;
  }
}

void ScaleAddRow_LSX(const uint8_t* src_ptr, uint16_t* dst_ptr, int src_width) {
  int x;
  int len = src_width / 16;
  __m128i src0, tmp0, tmp1, dst0, dst1;
  __m128i zero = __lsx_vldi(0);

  assert(src_width > 0);

  for (x = 0; x < len; x++) {
    src0 = __lsx_vld(src_ptr, 0);
    DUP2_ARG2(__lsx_vld, dst_ptr, 0, dst_ptr, 16, dst0, dst1);
    tmp0 = __lsx_vilvl_b(zero, src0);
    tmp1 = __lsx_vilvh_b(zero, src0);
    DUP2_ARG2(__lsx_vadd_h, dst0, tmp0, dst1, tmp1, dst0, dst1);
    __lsx_vst(dst0, dst_ptr, 0);
    __lsx_vst(dst1, dst_ptr, 16);
    src_ptr += 16;
    dst_ptr += 16;
  }
}

void ScaleFilterCols_LSX(uint8_t* dst_ptr,
                         const uint8_t* src_ptr,
                         int dst_width,
                         int x,
                         int dx) {
  int j;
  int len = dst_width / 16;
  __m128i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
  __m128i reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7;
  __m128i vec0, vec1, dst0;
  __m128i vec_x = __lsx_vreplgr2vr_w(x);
  __m128i vec_dx = __lsx_vreplgr2vr_w(dx);
  __m128i const1 = __lsx_vreplgr2vr_w(0xFFFF);
  __m128i const2 = __lsx_vreplgr2vr_w(0x40);
  __m128i const_tmp = {0x0000000100000000, 0x0000000300000002};

  vec0 = __lsx_vmul_w(vec_dx, const_tmp);
  vec1 = __lsx_vslli_w(vec_dx, 2);
  vec_x = __lsx_vadd_w(vec_x, vec0);

  for (j = 0; j < len; j++) {
    tmp0 = __lsx_vsrai_w(vec_x, 16);
    tmp4 = __lsx_vand_v(vec_x, const1);
    vec_x = __lsx_vadd_w(vec_x, vec1);
    tmp1 = __lsx_vsrai_w(vec_x, 16);
    tmp5 = __lsx_vand_v(vec_x, const1);
    vec_x = __lsx_vadd_w(vec_x, vec1);
    tmp2 = __lsx_vsrai_w(vec_x, 16);
    tmp6 = __lsx_vand_v(vec_x, const1);
    vec_x = __lsx_vadd_w(vec_x, vec1);
    tmp3 = __lsx_vsrai_w(vec_x, 16);
    tmp7 = __lsx_vand_v(vec_x, const1);
    vec_x = __lsx_vadd_w(vec_x, vec1);
    DUP4_ARG2(__lsx_vsrai_w, tmp4, 9, tmp5, 9, tmp6, 9, tmp7, 9, tmp4, tmp5,
              tmp6, tmp7);
    LOAD_DATA(src_ptr, tmp0, reg0);
    LOAD_DATA(src_ptr, tmp1, reg1);
    LOAD_DATA(src_ptr, tmp2, reg2);
    LOAD_DATA(src_ptr, tmp3, reg3);
    DUP4_ARG2(__lsx_vaddi_wu, tmp0, 1, tmp1, 1, tmp2, 1, tmp3, 1, tmp0, tmp1,
              tmp2, tmp3);
    LOAD_DATA(src_ptr, tmp0, reg4);
    LOAD_DATA(src_ptr, tmp1, reg5);
    LOAD_DATA(src_ptr, tmp2, reg6);
    LOAD_DATA(src_ptr, tmp3, reg7);
    DUP4_ARG2(__lsx_vsub_w, reg4, reg0, reg5, reg1, reg6, reg2, reg7, reg3,
              reg4, reg5, reg6, reg7);
    DUP4_ARG2(__lsx_vmul_w, reg4, tmp4, reg5, tmp5, reg6, tmp6, reg7, tmp7,
              reg4, reg5, reg6, reg7);
    DUP4_ARG2(__lsx_vadd_w, reg4, const2, reg5, const2, reg6, const2, reg7,
              const2, reg4, reg5, reg6, reg7);
    DUP4_ARG2(__lsx_vsrai_w, reg4, 7, reg5, 7, reg6, 7, reg7, 7, reg4, reg5,
              reg6, reg7);
    DUP4_ARG2(__lsx_vadd_w, reg0, reg4, reg1, reg5, reg2, reg6, reg3, reg7,
              reg0, reg1, reg2, reg3);
    DUP2_ARG2(__lsx_vpickev_h, reg1, reg0, reg3, reg2, tmp0, tmp1);
    dst0 = __lsx_vpickev_b(tmp1, tmp0);
    __lsx_vst(dst0, dst_ptr, 0);
    dst_ptr += 16;
  }
}

void ScaleARGBCols_LSX(uint8_t* dst_argb,
                       const uint8_t* src_argb,
                       int dst_width,
                       int x,
                       int dx) {
  const uint32_t* src = (const uint32_t*)src_argb;
  uint32_t* dst = (uint32_t*)dst_argb;
  int j;
  int len = dst_width / 4;
  __m128i tmp0, tmp1, tmp2, dst0;
  __m128i vec_x = __lsx_vreplgr2vr_w(x);
  __m128i vec_dx = __lsx_vreplgr2vr_w(dx);
  __m128i const_tmp = {0x0000000100000000, 0x0000000300000002};

  tmp0 = __lsx_vmul_w(vec_dx, const_tmp);
  tmp1 = __lsx_vslli_w(vec_dx, 2);
  vec_x = __lsx_vadd_w(vec_x, tmp0);

  for (j = 0; j < len; j++) {
    tmp2 = __lsx_vsrai_w(vec_x, 16);
    vec_x = __lsx_vadd_w(vec_x, tmp1);
    LOAD_DATA(src, tmp2, dst0);
    __lsx_vst(dst0, dst, 0);
    dst += 4;
  }
}

void ScaleARGBFilterCols_LSX(uint8_t* dst_argb,
                             const uint8_t* src_argb,
                             int dst_width,
                             int x,
                             int dx) {
  const uint32_t* src = (const uint32_t*)src_argb;
  int j;
  int len = dst_width / 8;
  __m128i src0, src1, src2, src3;
  __m128i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
  __m128i reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7;
  __m128i vec0, vec1, dst0, dst1;
  __m128i vec_x = __lsx_vreplgr2vr_w(x);
  __m128i vec_dx = __lsx_vreplgr2vr_w(dx);
  __m128i const_tmp = {0x0000000100000000, 0x0000000300000002};
  __m128i const_7f = __lsx_vldi(0x7F);

  vec0 = __lsx_vmul_w(vec_dx, const_tmp);
  vec1 = __lsx_vslli_w(vec_dx, 2);
  vec_x = __lsx_vadd_w(vec_x, vec0);

  for (j = 0; j < len; j++) {
    tmp0 = __lsx_vsrai_w(vec_x, 16);
    reg0 = __lsx_vsrai_w(vec_x, 9);
    vec_x = __lsx_vadd_w(vec_x, vec1);
    tmp1 = __lsx_vsrai_w(vec_x, 16);
    reg1 = __lsx_vsrai_w(vec_x, 9);
    vec_x = __lsx_vadd_w(vec_x, vec1);
    DUP2_ARG2(__lsx_vand_v, reg0, const_7f, reg1, const_7f, reg0, reg1);
    DUP2_ARG2(__lsx_vshuf4i_b, reg0, 0, reg1, 0, reg0, reg1);
    DUP2_ARG2(__lsx_vxor_v, reg0, const_7f, reg1, const_7f, reg2, reg3);
    DUP2_ARG2(__lsx_vilvl_b, reg0, reg2, reg1, reg3, reg4, reg6);
    DUP2_ARG2(__lsx_vilvh_b, reg0, reg2, reg1, reg3, reg5, reg7);
    LOAD_DATA(src, tmp0, src0);
    LOAD_DATA(src, tmp1, src1);
    DUP2_ARG2(__lsx_vaddi_wu, tmp0, 1, tmp1, 1, tmp0, tmp1);
    LOAD_DATA(src, tmp0, src2);
    LOAD_DATA(src, tmp1, src3);
    DUP2_ARG2(__lsx_vilvl_b, src2, src0, src3, src1, tmp4, tmp6);
    DUP2_ARG2(__lsx_vilvh_b, src2, src0, src3, src1, tmp5, tmp7);
    DUP4_ARG2(__lsx_vdp2_h_bu, tmp4, reg4, tmp5, reg5, tmp6, reg6, tmp7, reg7,
              tmp0, tmp1, tmp2, tmp3);
    DUP2_ARG3(__lsx_vsrani_b_h, tmp1, tmp0, 7, tmp3, tmp2, 7, dst0, dst1);
    __lsx_vst(dst0, dst_argb, 0);
    __lsx_vst(dst1, dst_argb, 16);
    dst_argb += 32;
  }
}

void ScaleRowDown34_LSX(const uint8_t* src_ptr,
                        ptrdiff_t src_stride,
                        uint8_t* dst,
                        int dst_width) {
  int x;
  (void)src_stride;
  __m128i src0, src1, src2, src3;
  __m128i dst0, dst1, dst2;
  __m128i shuff0 = {0x0908070504030100, 0x141311100F0D0C0B};
  __m128i shuff1 = {0x0F0D0C0B09080705, 0x1918171514131110};
  __m128i shuff2 = {0x141311100F0D0C0B, 0x1F1D1C1B19181715};

  assert((dst_width % 3 == 0) && (dst_width > 0));

  for (x = 0; x < dst_width; x += 48) {
    DUP4_ARG2(__lsx_vld, src_ptr, 0, src_ptr, 16, src_ptr, 32, src_ptr, 48,
              src0, src1, src2, src3);
    DUP2_ARG3(__lsx_vshuf_b, src1, src0, shuff0, src2, src1, shuff1, dst0,
              dst1);
    dst2 = __lsx_vshuf_b(src3, src2, shuff2);
    __lsx_vst(dst0, dst, 0);
    __lsx_vst(dst1, dst, 16);
    __lsx_vst(dst2, dst, 32);
    src_ptr += 64;
    dst += 48;
  }
}

void ScaleRowDown34_0_Box_LSX(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* d,
                              int dst_width) {
  const uint8_t* src_nex = src_ptr + src_stride;
  int x;
  __m128i src0, src1, src2, src3, src4, src5, src6, src7;
  __m128i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9;
  __m128i tmp10, tmp11, dst0, dst1, dst2;
  __m128i const0 = {0x0103030101010103, 0x0101010303010101};
  __m128i const1 = {0x0301010101030301, 0x0103030101010103};
  __m128i const2 = {0x0101010303010101, 0x0301010101030301};
  __m128i shuff0 = {0x0504030202010100, 0x0A09090807060605};
  __m128i shuff1 = {0x0F0E0E0D0D0C0B0A, 0x1514131212111110};
  __m128i shuff2 = {0x0A09090807060605, 0x0F0E0E0D0D0C0B0A};
  __m128i shift0 = {0x0002000200010002, 0x0001000200020001};
  __m128i shift1 = {0x0002000100020002, 0x0002000200010002};
  __m128i shift2 = {0x0001000200020001, 0x0002000100020002};

  assert((dst_width % 3 == 0) && (dst_width > 0));

  for (x = 0; x < dst_width; x += 48) {
    DUP4_ARG2(__lsx_vld, src_ptr, 0, src_ptr, 16, src_ptr, 32, src_ptr, 48,
              src0, src1, src2, src3);
    DUP4_ARG2(__lsx_vld, src_nex, 0, src_nex, 16, src_nex, 32, src_nex, 48,
              src4, src5, src6, src7);
    DUP4_ARG3(__lsx_vshuf_b, src0, src0, shuff0, src1, src0, shuff1, src1, src1,
              shuff2, src2, src2, shuff0, tmp0, tmp1, tmp2, tmp3);
    DUP4_ARG3(__lsx_vshuf_b, src3, src2, shuff1, src3, src3, shuff2, src4, src4,
              shuff0, src5, src4, shuff1, tmp4, tmp5, tmp6, tmp7);
    DUP4_ARG3(__lsx_vshuf_b, src5, src5, shuff2, src6, src6, shuff0, src7, src6,
              shuff1, src7, src7, shuff2, tmp8, tmp9, tmp10, tmp11);
    DUP4_ARG2(__lsx_vdp2_h_bu, tmp0, const0, tmp1, const1, tmp2, const2, tmp3,
              const0, src0, src1, src2, src3);
    DUP4_ARG2(__lsx_vdp2_h_bu, tmp4, const1, tmp5, const2, tmp6, const0, tmp7,
              const1, src4, src5, src6, src7);
    DUP4_ARG2(__lsx_vdp2_h_bu, tmp8, const2, tmp9, const0, tmp10, const1, tmp11,
              const2, tmp0, tmp1, tmp2, tmp3);
    DUP4_ARG2(__lsx_vsrar_h, src0, shift0, src1, shift1, src2, shift2, src3,
              shift0, src0, src1, src2, src3);
    DUP4_ARG2(__lsx_vsrar_h, src4, shift1, src5, shift2, src6, shift0, src7,
              shift1, src4, src5, src6, src7);
    DUP4_ARG2(__lsx_vsrar_h, tmp0, shift2, tmp1, shift0, tmp2, shift1, tmp3,
              shift2, tmp0, tmp1, tmp2, tmp3);
    DUP4_ARG2(__lsx_vslli_h, src0, 1, src1, 1, src2, 1, src3, 1, tmp5, tmp6,
              tmp7, tmp8);
    DUP2_ARG2(__lsx_vslli_h, src4, 1, src5, 1, tmp9, tmp10);
    DUP4_ARG2(__lsx_vadd_h, src0, tmp5, src1, tmp6, src2, tmp7, src3, tmp8,
              src0, src1, src2, src3);
    DUP2_ARG2(__lsx_vadd_h, src4, tmp9, src5, tmp10, src4, src5);
    DUP4_ARG2(__lsx_vadd_h, src0, src6, src1, src7, src2, tmp0, src3, tmp1,
              src0, src1, src2, src3);
    DUP2_ARG2(__lsx_vadd_h, src4, tmp2, src5, tmp3, src4, src5);
    DUP2_ARG3(__lsx_vsrarni_b_h, src1, src0, 2, src3, src2, 2, dst0, dst1);
    dst2 = __lsx_vsrarni_b_h(src5, src4, 2);
    __lsx_vst(dst0, d, 0);
    __lsx_vst(dst1, d, 16);
    __lsx_vst(dst2, d, 32);
    src_ptr += 64;
    src_nex += 64;
    d += 48;
  }
}

void ScaleRowDown34_1_Box_LSX(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* d,
                              int dst_width) {
  const uint8_t* src_nex = src_ptr + src_stride;
  int x;
  __m128i src0, src1, src2, src3, src4, src5, src6, src7;
  __m128i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9;
  __m128i tmp10, tmp11, dst0, dst1, dst2;
  __m128i const0 = {0x0103030101010103, 0x0101010303010101};
  __m128i const1 = {0x0301010101030301, 0x0103030101010103};
  __m128i const2 = {0x0101010303010101, 0x0301010101030301};
  __m128i shuff0 = {0x0504030202010100, 0x0A09090807060605};
  __m128i shuff1 = {0x0F0E0E0D0D0C0B0A, 0x1514131212111110};
  __m128i shuff2 = {0x0A09090807060605, 0x0F0E0E0D0D0C0B0A};
  __m128i shift0 = {0x0002000200010002, 0x0001000200020001};
  __m128i shift1 = {0x0002000100020002, 0x0002000200010002};
  __m128i shift2 = {0x0001000200020001, 0x0002000100020002};

  assert((dst_width % 3 == 0) && (dst_width > 0));

  for (x = 0; x < dst_width; x += 48) {
    DUP4_ARG2(__lsx_vld, src_ptr, 0, src_ptr, 16, src_ptr, 32, src_ptr, 48,
              src0, src1, src2, src3);
    DUP4_ARG2(__lsx_vld, src_nex, 0, src_nex, 16, src_nex, 32, src_nex, 48,
              src4, src5, src6, src7);
    DUP4_ARG3(__lsx_vshuf_b, src0, src0, shuff0, src1, src0, shuff1, src1, src1,
              shuff2, src2, src2, shuff0, tmp0, tmp1, tmp2, tmp3);
    DUP4_ARG3(__lsx_vshuf_b, src3, src2, shuff1, src3, src3, shuff2, src4, src4,
              shuff0, src5, src4, shuff1, tmp4, tmp5, tmp6, tmp7);
    DUP4_ARG3(__lsx_vshuf_b, src5, src5, shuff2, src6, src6, shuff0, src7, src6,
              shuff1, src7, src7, shuff2, tmp8, tmp9, tmp10, tmp11);
    DUP4_ARG2(__lsx_vdp2_h_bu, tmp0, const0, tmp1, const1, tmp2, const2, tmp3,
              const0, src0, src1, src2, src3);
    DUP4_ARG2(__lsx_vdp2_h_bu, tmp4, const1, tmp5, const2, tmp6, const0, tmp7,
              const1, src4, src5, src6, src7);
    DUP4_ARG2(__lsx_vdp2_h_bu, tmp8, const2, tmp9, const0, tmp10, const1, tmp11,
              const2, tmp0, tmp1, tmp2, tmp3);
    DUP4_ARG2(__lsx_vsrar_h, src0, shift0, src1, shift1, src2, shift2, src3,
              shift0, src0, src1, src2, src3);
    DUP4_ARG2(__lsx_vsrar_h, src4, shift1, src5, shift2, src6, shift0, src7,
              shift1, src4, src5, src6, src7);
    DUP4_ARG2(__lsx_vsrar_h, tmp0, shift2, tmp1, shift0, tmp2, shift1, tmp3,
              shift2, tmp0, tmp1, tmp2, tmp3);
    DUP4_ARG2(__lsx_vadd_h, src0, src6, src1, src7, src2, tmp0, src3, tmp1,
              src0, src1, src2, src3);
    DUP2_ARG2(__lsx_vadd_h, src4, tmp2, src5, tmp3, src4, src5);
    DUP2_ARG3(__lsx_vsrarni_b_h, src1, src0, 1, src3, src2, 1, dst0, dst1);
    dst2 = __lsx_vsrarni_b_h(src5, src4, 1);
    __lsx_vst(dst0, d, 0);
    __lsx_vst(dst1, d, 16);
    __lsx_vst(dst2, d, 32);
    src_ptr += 64;
    src_nex += 64;
    d += 48;
  }
}

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

#endif  // !defined(LIBYUV_DISABLE_LSX) && defined(__loongarch_sx)
