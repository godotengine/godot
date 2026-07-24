/*
 *  Copyright 2016 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "libyuv/rotate_row.h"

// This module is for GCC MSA
#if !defined(LIBYUV_DISABLE_MSA) && defined(__mips_msa)
#include "libyuv/macros_msa.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

#define ILVRL_B(in0, in1, in2, in3, out0, out1, out2, out3) \
  {                                                         \
    out0 = (v16u8)__msa_ilvr_b((v16i8)in1, (v16i8)in0);     \
    out1 = (v16u8)__msa_ilvl_b((v16i8)in1, (v16i8)in0);     \
    out2 = (v16u8)__msa_ilvr_b((v16i8)in3, (v16i8)in2);     \
    out3 = (v16u8)__msa_ilvl_b((v16i8)in3, (v16i8)in2);     \
  }

#define ILVRL_H(in0, in1, in2, in3, out0, out1, out2, out3) \
  {                                                         \
    out0 = (v16u8)__msa_ilvr_h((v8i16)in1, (v8i16)in0);     \
    out1 = (v16u8)__msa_ilvl_h((v8i16)in1, (v8i16)in0);     \
    out2 = (v16u8)__msa_ilvr_h((v8i16)in3, (v8i16)in2);     \
    out3 = (v16u8)__msa_ilvl_h((v8i16)in3, (v8i16)in2);     \
  }

#define ILVRL_W(in0, in1, in2, in3, out0, out1, out2, out3) \
  {                                                         \
    out0 = (v16u8)__msa_ilvr_w((v4i32)in1, (v4i32)in0);     \
    out1 = (v16u8)__msa_ilvl_w((v4i32)in1, (v4i32)in0);     \
    out2 = (v16u8)__msa_ilvr_w((v4i32)in3, (v4i32)in2);     \
    out3 = (v16u8)__msa_ilvl_w((v4i32)in3, (v4i32)in2);     \
  }

#define ILVRL_D(in0, in1, in2, in3, out0, out1, out2, out3) \
  {                                                         \
    out0 = (v16u8)__msa_ilvr_d((v2i64)in1, (v2i64)in0);     \
    out1 = (v16u8)__msa_ilvl_d((v2i64)in1, (v2i64)in0);     \
    out2 = (v16u8)__msa_ilvr_d((v2i64)in3, (v2i64)in2);     \
    out3 = (v16u8)__msa_ilvl_d((v2i64)in3, (v2i64)in2);     \
  }

void TransposeWx16_C(const uint8_t* src,
                     int src_stride,
                     uint8_t* dst,
                     int dst_stride,
                     int width) {
  TransposeWx8_C(src, src_stride, dst, dst_stride, width);
  TransposeWx8_C((src + 8 * src_stride), src_stride, (dst + 8), dst_stride,
                 width);
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

void TransposeWx16_MSA(const uint8_t* src,
                       int src_stride,
                       uint8_t* dst,
                       int dst_stride,
                       int width) {
  int x;
  const uint8_t* s;
  v16u8 src0, src1, src2, src3, dst0, dst1, dst2, dst3, vec0, vec1, vec2, vec3;
  v16u8 reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7;
  v16u8 res0, res1, res2, res3, res4, res5, res6, res7, res8, res9;

  for (x = 0; x < width; x += 16) {
    s = src;
    src0 = (v16u8)__msa_ld_b((v16i8*)s, 0);
    s += src_stride;
    src1 = (v16u8)__msa_ld_b((v16i8*)s, 0);
    s += src_stride;
    src2 = (v16u8)__msa_ld_b((v16i8*)s, 0);
    s += src_stride;
    src3 = (v16u8)__msa_ld_b((v16i8*)s, 0);
    s += src_stride;
    ILVRL_B(src0, src1, src2, src3, vec0, vec1, vec2, vec3);
    ILVRL_H(vec0, vec2, vec1, vec3, reg0, reg1, reg2, reg3);
    src0 = (v16u8)__msa_ld_b((v16i8*)s, 0);
    s += src_stride;
    src1 = (v16u8)__msa_ld_b((v16i8*)s, 0);
    s += src_stride;
    src2 = (v16u8)__msa_ld_b((v16i8*)s, 0);
    s += src_stride;
    src3 = (v16u8)__msa_ld_b((v16i8*)s, 0);
    s += src_stride;
    ILVRL_B(src0, src1, src2, src3, vec0, vec1, vec2, vec3);
    ILVRL_H(vec0, vec2, vec1, vec3, reg4, reg5, reg6, reg7);
    ILVRL_W(reg0, reg4, reg1, reg5, res0, res1, res2, res3);
    ILVRL_W(reg2, reg6, reg3, reg7, res4, res5, res6, res7);
    src0 = (v16u8)__msa_ld_b((v16i8*)s, 0);
    s += src_stride;
    src1 = (v16u8)__msa_ld_b((v16i8*)s, 0);
    s += src_stride;
    src2 = (v16u8)__msa_ld_b((v16i8*)s, 0);
    s += src_stride;
    src3 = (v16u8)__msa_ld_b((v16i8*)s, 0);
    s += src_stride;
    ILVRL_B(src0, src1, src2, src3, vec0, vec1, vec2, vec3);
    ILVRL_H(vec0, vec2, vec1, vec3, reg0, reg1, reg2, reg3);
    src0 = (v16u8)__msa_ld_b((v16i8*)s, 0);
    s += src_stride;
    src1 = (v16u8)__msa_ld_b((v16i8*)s, 0);
    s += src_stride;
    src2 = (v16u8)__msa_ld_b((v16i8*)s, 0);
    s += src_stride;
    src3 = (v16u8)__msa_ld_b((v16i8*)s, 0);
    s += src_stride;
    ILVRL_B(src0, src1, src2, src3, vec0, vec1, vec2, vec3);
    ILVRL_H(vec0, vec2, vec1, vec3, reg4, reg5, reg6, reg7);
    res8 = (v16u8)__msa_ilvr_w((v4i32)reg4, (v4i32)reg0);
    res9 = (v16u8)__msa_ilvl_w((v4i32)reg4, (v4i32)reg0);
    ILVRL_D(res0, res8, res1, res9, dst0, dst1, dst2, dst3);
    ST_UB4(dst0, dst1, dst2, dst3, dst, dst_stride);
    dst += dst_stride * 4;
    res8 = (v16u8)__msa_ilvr_w((v4i32)reg5, (v4i32)reg1);
    res9 = (v16u8)__msa_ilvl_w((v4i32)reg5, (v4i32)reg1);
    ILVRL_D(res2, res8, res3, res9, dst0, dst1, dst2, dst3);
    ST_UB4(dst0, dst1, dst2, dst3, dst, dst_stride);
    dst += dst_stride * 4;
    res8 = (v16u8)__msa_ilvr_w((v4i32)reg6, (v4i32)reg2);
    res9 = (v16u8)__msa_ilvl_w((v4i32)reg6, (v4i32)reg2);
    ILVRL_D(res4, res8, res5, res9, dst0, dst1, dst2, dst3);
    ST_UB4(dst0, dst1, dst2, dst3, dst, dst_stride);
    dst += dst_stride * 4;
    res8 = (v16u8)__msa_ilvr_w((v4i32)reg7, (v4i32)reg3);
    res9 = (v16u8)__msa_ilvl_w((v4i32)reg7, (v4i32)reg3);
    ILVRL_D(res6, res8, res7, res9, dst0, dst1, dst2, dst3);
    ST_UB4(dst0, dst1, dst2, dst3, dst, dst_stride);
    src += 16;
    dst += dst_stride * 4;
  }
}

void TransposeUVWx16_MSA(const uint8_t* src,
                         int src_stride,
                         uint8_t* dst_a,
                         int dst_stride_a,
                         uint8_t* dst_b,
                         int dst_stride_b,
                         int width) {
  int x;
  const uint8_t* s;
  v16u8 src0, src1, src2, src3, dst0, dst1, dst2, dst3, vec0, vec1, vec2, vec3;
  v16u8 reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7;
  v16u8 res0, res1, res2, res3, res4, res5, res6, res7, res8, res9;

  for (x = 0; x < width; x += 8) {
    s = src;
    src0 = (v16u8)__msa_ld_b((v16i8*)s, 0);
    s += src_stride;
    src1 = (v16u8)__msa_ld_b((v16i8*)s, 0);
    s += src_stride;
    src2 = (v16u8)__msa_ld_b((v16i8*)s, 0);
    s += src_stride;
    src3 = (v16u8)__msa_ld_b((v16i8*)s, 0);
    s += src_stride;
    ILVRL_B(src0, src1, src2, src3, vec0, vec1, vec2, vec3);
    ILVRL_H(vec0, vec2, vec1, vec3, reg0, reg1, reg2, reg3);
    src0 = (v16u8)__msa_ld_b((v16i8*)s, 0);
    s += src_stride;
    src1 = (v16u8)__msa_ld_b((v16i8*)s, 0);
    s += src_stride;
    src2 = (v16u8)__msa_ld_b((v16i8*)s, 0);
    s += src_stride;
    src3 = (v16u8)__msa_ld_b((v16i8*)s, 0);
    s += src_stride;
    ILVRL_B(src0, src1, src2, src3, vec0, vec1, vec2, vec3);
    ILVRL_H(vec0, vec2, vec1, vec3, reg4, reg5, reg6, reg7);
    ILVRL_W(reg0, reg4, reg1, reg5, res0, res1, res2, res3);
    ILVRL_W(reg2, reg6, reg3, reg7, res4, res5, res6, res7);
    src0 = (v16u8)__msa_ld_b((v16i8*)s, 0);
    s += src_stride;
    src1 = (v16u8)__msa_ld_b((v16i8*)s, 0);
    s += src_stride;
    src2 = (v16u8)__msa_ld_b((v16i8*)s, 0);
    s += src_stride;
    src3 = (v16u8)__msa_ld_b((v16i8*)s, 0);
    s += src_stride;
    ILVRL_B(src0, src1, src2, src3, vec0, vec1, vec2, vec3);
    ILVRL_H(vec0, vec2, vec1, vec3, reg0, reg1, reg2, reg3);
    src0 = (v16u8)__msa_ld_b((v16i8*)s, 0);
    s += src_stride;
    src1 = (v16u8)__msa_ld_b((v16i8*)s, 0);
    s += src_stride;
    src2 = (v16u8)__msa_ld_b((v16i8*)s, 0);
    s += src_stride;
    src3 = (v16u8)__msa_ld_b((v16i8*)s, 0);
    s += src_stride;
    ILVRL_B(src0, src1, src2, src3, vec0, vec1, vec2, vec3);
    ILVRL_H(vec0, vec2, vec1, vec3, reg4, reg5, reg6, reg7);
    res8 = (v16u8)__msa_ilvr_w((v4i32)reg4, (v4i32)reg0);
    res9 = (v16u8)__msa_ilvl_w((v4i32)reg4, (v4i32)reg0);
    ILVRL_D(res0, res8, res1, res9, dst0, dst1, dst2, dst3);
    ST_UB2(dst0, dst2, dst_a, dst_stride_a);
    ST_UB2(dst1, dst3, dst_b, dst_stride_b);
    dst_a += dst_stride_a * 2;
    dst_b += dst_stride_b * 2;
    res8 = (v16u8)__msa_ilvr_w((v4i32)reg5, (v4i32)reg1);
    res9 = (v16u8)__msa_ilvl_w((v4i32)reg5, (v4i32)reg1);
    ILVRL_D(res2, res8, res3, res9, dst0, dst1, dst2, dst3);
    ST_UB2(dst0, dst2, dst_a, dst_stride_a);
    ST_UB2(dst1, dst3, dst_b, dst_stride_b);
    dst_a += dst_stride_a * 2;
    dst_b += dst_stride_b * 2;
    res8 = (v16u8)__msa_ilvr_w((v4i32)reg6, (v4i32)reg2);
    res9 = (v16u8)__msa_ilvl_w((v4i32)reg6, (v4i32)reg2);
    ILVRL_D(res4, res8, res5, res9, dst0, dst1, dst2, dst3);
    ST_UB2(dst0, dst2, dst_a, dst_stride_a);
    ST_UB2(dst1, dst3, dst_b, dst_stride_b);
    dst_a += dst_stride_a * 2;
    dst_b += dst_stride_b * 2;
    res8 = (v16u8)__msa_ilvr_w((v4i32)reg7, (v4i32)reg3);
    res9 = (v16u8)__msa_ilvl_w((v4i32)reg7, (v4i32)reg3);
    ILVRL_D(res6, res8, res7, res9, dst0, dst1, dst2, dst3);
    ST_UB2(dst0, dst2, dst_a, dst_stride_a);
    ST_UB2(dst1, dst3, dst_b, dst_stride_b);
    src += 16;
    dst_a += dst_stride_a * 2;
    dst_b += dst_stride_b * 2;
  }
}

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

#endif  // !defined(LIBYUV_DISABLE_MSA) && defined(__mips_msa)
