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

#include "libyuv/rotate_row.h"

#if !defined(LIBYUV_DISABLE_LSX) && defined(__loongarch_sx)
#include "libyuv/loongson_intrinsics.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

#define ILVLH_B(in0, in1, in2, in3, out0, out1, out2, out3)   \
  {                                                           \
    DUP2_ARG2(__lsx_vilvl_b, in1, in0, in3, in2, out0, out2); \
    DUP2_ARG2(__lsx_vilvh_b, in1, in0, in3, in2, out1, out3); \
  }

#define ILVLH_H(in0, in1, in2, in3, out0, out1, out2, out3)   \
  {                                                           \
    DUP2_ARG2(__lsx_vilvl_h, in1, in0, in3, in2, out0, out2); \
    DUP2_ARG2(__lsx_vilvh_h, in1, in0, in3, in2, out1, out3); \
  }

#define ILVLH_W(in0, in1, in2, in3, out0, out1, out2, out3)   \
  {                                                           \
    DUP2_ARG2(__lsx_vilvl_w, in1, in0, in3, in2, out0, out2); \
    DUP2_ARG2(__lsx_vilvh_w, in1, in0, in3, in2, out1, out3); \
  }

#define ILVLH_D(in0, in1, in2, in3, out0, out1, out2, out3)   \
  {                                                           \
    DUP2_ARG2(__lsx_vilvl_d, in1, in0, in3, in2, out0, out2); \
    DUP2_ARG2(__lsx_vilvh_d, in1, in0, in3, in2, out1, out3); \
  }

#define LSX_ST_4(_dst0, _dst1, _dst2, _dst3, _dst, _stride, _stride2, \
                 _stride3, _stride4)                                  \
  {                                                                   \
    __lsx_vst(_dst0, _dst, 0);                                        \
    __lsx_vstx(_dst1, _dst, _stride);                                 \
    __lsx_vstx(_dst2, _dst, _stride2);                                \
    __lsx_vstx(_dst3, _dst, _stride3);                                \
    _dst += _stride4;                                                 \
  }

#define LSX_ST_2(_dst0, _dst1, _dst, _stride, _stride2) \
  {                                                     \
    __lsx_vst(_dst0, _dst, 0);                          \
    __lsx_vstx(_dst1, _dst, _stride);                   \
    _dst += _stride2;                                   \
  }

void TransposeUVWx16_C(const uint8_t* src,
                       int src_stride,
                       uint8_t* dst_a,
                       int dst_stride_a,
                       uint8_t* dst_b,
                       int dst_stride_b,
                       int width) {
  TransposeUVWx8_C(src, src_stride, dst_a, dst_stride_a, dst_b, dst_stride_b,
                   width);
  TransposeUVWx8_C((src + 8 * src_stride), src_stride, (dst_a + 8),
                   dst_stride_a, (dst_b + 8), dst_stride_b, width);
}

void TransposeWx16_LSX(const uint8_t* src,
                       int src_stride,
                       uint8_t* dst,
                       int dst_stride,
                       int width) {
  int x;
  int len = width / 16;
  uint8_t* s;
  int src_stride2 = src_stride << 1;
  int src_stride3 = src_stride + src_stride2;
  int src_stride4 = src_stride2 << 1;
  int dst_stride2 = dst_stride << 1;
  int dst_stride3 = dst_stride + dst_stride2;
  int dst_stride4 = dst_stride2 << 1;
  __m128i src0, src1, src2, src3, dst0, dst1, dst2, dst3;
  __m128i tmp0, tmp1, tmp2, tmp3;
  __m128i reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7;
  __m128i res0, res1, res2, res3, res4, res5, res6, res7, res8, res9;

  for (x = 0; x < len; x++) {
    s = (uint8_t*)src;
    src0 = __lsx_vld(s, 0);
    src1 = __lsx_vldx(s, src_stride);
    src2 = __lsx_vldx(s, src_stride2);
    src3 = __lsx_vldx(s, src_stride3);
    s += src_stride4;
    ILVLH_B(src0, src1, src2, src3, tmp0, tmp1, tmp2, tmp3);
    ILVLH_H(tmp0, tmp2, tmp1, tmp3, reg0, reg1, reg2, reg3);
    src0 = __lsx_vld(s, 0);
    src1 = __lsx_vldx(s, src_stride);
    src2 = __lsx_vldx(s, src_stride2);
    src3 = __lsx_vldx(s, src_stride3);
    s += src_stride4;
    ILVLH_B(src0, src1, src2, src3, tmp0, tmp1, tmp2, tmp3);
    ILVLH_H(tmp0, tmp2, tmp1, tmp3, reg4, reg5, reg6, reg7);
    ILVLH_W(reg0, reg4, reg1, reg5, res0, res1, res2, res3);
    ILVLH_W(reg2, reg6, reg3, reg7, res4, res5, res6, res7);
    src0 = __lsx_vld(s, 0);
    src1 = __lsx_vldx(s, src_stride);
    src2 = __lsx_vldx(s, src_stride2);
    src3 = __lsx_vldx(s, src_stride3);
    s += src_stride4;
    ILVLH_B(src0, src1, src2, src3, tmp0, tmp1, tmp2, tmp3);
    ILVLH_H(tmp0, tmp2, tmp1, tmp3, reg0, reg1, reg2, reg3);
    src0 = __lsx_vld(s, 0);
    src1 = __lsx_vldx(s, src_stride);
    src2 = __lsx_vldx(s, src_stride2);
    src3 = __lsx_vldx(s, src_stride3);
    s += src_stride4;
    ILVLH_B(src0, src1, src2, src3, tmp0, tmp1, tmp2, tmp3);
    ILVLH_H(tmp0, tmp2, tmp1, tmp3, reg4, reg5, reg6, reg7);
    res8 = __lsx_vilvl_w(reg4, reg0);
    res9 = __lsx_vilvh_w(reg4, reg0);
    ILVLH_D(res0, res8, res1, res9, dst0, dst1, dst2, dst3);
    LSX_ST_4(dst0, dst1, dst2, dst3, dst, dst_stride, dst_stride2, dst_stride3,
             dst_stride4);
    res8 = __lsx_vilvl_w(reg5, reg1);
    res9 = __lsx_vilvh_w(reg5, reg1);
    ILVLH_D(res2, res8, res3, res9, dst0, dst1, dst2, dst3);
    LSX_ST_4(dst0, dst1, dst2, dst3, dst, dst_stride, dst_stride2, dst_stride3,
             dst_stride4);
    res8 = __lsx_vilvl_w(reg6, reg2);
    res9 = __lsx_vilvh_w(reg6, reg2);
    ILVLH_D(res4, res8, res5, res9, dst0, dst1, dst2, dst3);
    LSX_ST_4(dst0, dst1, dst2, dst3, dst, dst_stride, dst_stride2, dst_stride3,
             dst_stride4);
    res8 = __lsx_vilvl_w(reg7, reg3);
    res9 = __lsx_vilvh_w(reg7, reg3);
    ILVLH_D(res6, res8, res7, res9, dst0, dst1, dst2, dst3);
    LSX_ST_4(dst0, dst1, dst2, dst3, dst, dst_stride, dst_stride2, dst_stride3,
             dst_stride4);
    src += 16;
  }
}

void TransposeUVWx16_LSX(const uint8_t* src,
                         int src_stride,
                         uint8_t* dst_a,
                         int dst_stride_a,
                         uint8_t* dst_b,
                         int dst_stride_b,
                         int width) {
  int x;
  int len = width / 8;
  uint8_t* s;
  int src_stride2 = src_stride << 1;
  int src_stride3 = src_stride + src_stride2;
  int src_stride4 = src_stride2 << 1;
  int dst_stride_a2 = dst_stride_a << 1;
  int dst_stride_b2 = dst_stride_b << 1;
  __m128i src0, src1, src2, src3, dst0, dst1, dst2, dst3;
  __m128i tmp0, tmp1, tmp2, tmp3;
  __m128i reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7;
  __m128i res0, res1, res2, res3, res4, res5, res6, res7, res8, res9;

  for (x = 0; x < len; x++) {
    s = (uint8_t*)src;
    src0 = __lsx_vld(s, 0);
    src1 = __lsx_vldx(s, src_stride);
    src2 = __lsx_vldx(s, src_stride2);
    src3 = __lsx_vldx(s, src_stride3);
    s += src_stride4;
    ILVLH_B(src0, src1, src2, src3, tmp0, tmp1, tmp2, tmp3);
    ILVLH_H(tmp0, tmp2, tmp1, tmp3, reg0, reg1, reg2, reg3);
    src0 = __lsx_vld(s, 0);
    src1 = __lsx_vldx(s, src_stride);
    src2 = __lsx_vldx(s, src_stride2);
    src3 = __lsx_vldx(s, src_stride3);
    s += src_stride4;
    ILVLH_B(src0, src1, src2, src3, tmp0, tmp1, tmp2, tmp3);
    ILVLH_H(tmp0, tmp2, tmp1, tmp3, reg4, reg5, reg6, reg7);
    ILVLH_W(reg0, reg4, reg1, reg5, res0, res1, res2, res3);
    ILVLH_W(reg2, reg6, reg3, reg7, res4, res5, res6, res7);
    src0 = __lsx_vld(s, 0);
    src1 = __lsx_vldx(s, src_stride);
    src2 = __lsx_vldx(s, src_stride2);
    src3 = __lsx_vldx(s, src_stride3);
    s += src_stride4;
    ILVLH_B(src0, src1, src2, src3, tmp0, tmp1, tmp2, tmp3);
    ILVLH_H(tmp0, tmp2, tmp1, tmp3, reg0, reg1, reg2, reg3);
    src0 = __lsx_vld(s, 0);
    src1 = __lsx_vldx(s, src_stride);
    src2 = __lsx_vldx(s, src_stride2);
    src3 = __lsx_vldx(s, src_stride3);
    s += src_stride4;
    ILVLH_B(src0, src1, src2, src3, tmp0, tmp1, tmp2, tmp3);
    ILVLH_H(tmp0, tmp2, tmp1, tmp3, reg4, reg5, reg6, reg7);
    res8 = __lsx_vilvl_w(reg4, reg0);
    res9 = __lsx_vilvh_w(reg4, reg0);
    ILVLH_D(res0, res8, res1, res9, dst0, dst1, dst2, dst3);
    LSX_ST_2(dst0, dst2, dst_a, dst_stride_a, dst_stride_a2);
    LSX_ST_2(dst1, dst3, dst_b, dst_stride_b, dst_stride_b2);
    res8 = __lsx_vilvl_w(reg5, reg1);
    res9 = __lsx_vilvh_w(reg5, reg1);
    ILVLH_D(res2, res8, res3, res9, dst0, dst1, dst2, dst3);
    LSX_ST_2(dst0, dst2, dst_a, dst_stride_a, dst_stride_a2);
    LSX_ST_2(dst1, dst3, dst_b, dst_stride_b, dst_stride_b2);
    res8 = __lsx_vilvl_w(reg6, reg2);
    res9 = __lsx_vilvh_w(reg6, reg2);
    ILVLH_D(res4, res8, res5, res9, dst0, dst1, dst2, dst3);
    LSX_ST_2(dst0, dst2, dst_a, dst_stride_a, dst_stride_a2);
    LSX_ST_2(dst1, dst3, dst_b, dst_stride_b, dst_stride_b2);
    res8 = __lsx_vilvl_w(reg7, reg3);
    res9 = __lsx_vilvh_w(reg7, reg3);
    ILVLH_D(res6, res8, res7, res9, dst0, dst1, dst2, dst3);
    LSX_ST_2(dst0, dst2, dst_a, dst_stride_a, dst_stride_a2);
    LSX_ST_2(dst1, dst3, dst_b, dst_stride_b, dst_stride_b2);
    src += 16;
  }
}

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

#endif  // !defined(LIBYUV_DISABLE_LSX) && defined(__loongarch_sx)
