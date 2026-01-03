/*
 *  Copyright 2017 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "libyuv/basic_types.h"

#include "libyuv/compare_row.h"
#include "libyuv/row.h"

// This module is for GCC MSA
#if !defined(LIBYUV_DISABLE_MSA) && defined(__mips_msa)
#include "libyuv/macros_msa.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

uint32_t HammingDistance_MSA(const uint8_t* src_a,
                             const uint8_t* src_b,
                             int count) {
  uint32_t diff = 0u;
  int i;
  v16u8 src0, src1, src2, src3;
  v2i64 vec0 = {0}, vec1 = {0};

  for (i = 0; i < count; i += 32) {
    src0 = (v16u8)__msa_ld_b((v16i8*)src_a, 0);
    src1 = (v16u8)__msa_ld_b((v16i8*)src_a, 16);
    src2 = (v16u8)__msa_ld_b((v16i8*)src_b, 0);
    src3 = (v16u8)__msa_ld_b((v16i8*)src_b, 16);
    src0 ^= src2;
    src1 ^= src3;
    vec0 += __msa_pcnt_d((v2i64)src0);
    vec1 += __msa_pcnt_d((v2i64)src1);
    src_a += 32;
    src_b += 32;
  }

  vec0 += vec1;
  diff = (uint32_t)__msa_copy_u_w((v4i32)vec0, 0);
  diff += (uint32_t)__msa_copy_u_w((v4i32)vec0, 2);
  return diff;
}

uint32_t SumSquareError_MSA(const uint8_t* src_a,
                            const uint8_t* src_b,
                            int count) {
  uint32_t sse = 0u;
  int i;
  v16u8 src0, src1, src2, src3;
  v8i16 vec0, vec1, vec2, vec3;
  v4i32 reg0 = {0}, reg1 = {0}, reg2 = {0}, reg3 = {0};
  v2i64 tmp0;

  for (i = 0; i < count; i += 32) {
    src0 = (v16u8)__msa_ld_b((v16i8*)src_a, 0);
    src1 = (v16u8)__msa_ld_b((v16i8*)src_a, 16);
    src2 = (v16u8)__msa_ld_b((v16i8*)src_b, 0);
    src3 = (v16u8)__msa_ld_b((v16i8*)src_b, 16);
    vec0 = (v8i16)__msa_ilvr_b((v16i8)src2, (v16i8)src0);
    vec1 = (v8i16)__msa_ilvl_b((v16i8)src2, (v16i8)src0);
    vec2 = (v8i16)__msa_ilvr_b((v16i8)src3, (v16i8)src1);
    vec3 = (v8i16)__msa_ilvl_b((v16i8)src3, (v16i8)src1);
    vec0 = __msa_hsub_u_h((v16u8)vec0, (v16u8)vec0);
    vec1 = __msa_hsub_u_h((v16u8)vec1, (v16u8)vec1);
    vec2 = __msa_hsub_u_h((v16u8)vec2, (v16u8)vec2);
    vec3 = __msa_hsub_u_h((v16u8)vec3, (v16u8)vec3);
    reg0 = __msa_dpadd_s_w(reg0, vec0, vec0);
    reg1 = __msa_dpadd_s_w(reg1, vec1, vec1);
    reg2 = __msa_dpadd_s_w(reg2, vec2, vec2);
    reg3 = __msa_dpadd_s_w(reg3, vec3, vec3);
    src_a += 32;
    src_b += 32;
  }

  reg0 += reg1;
  reg2 += reg3;
  reg0 += reg2;
  tmp0 = __msa_hadd_s_d(reg0, reg0);
  sse = (uint32_t)__msa_copy_u_w((v4i32)tmp0, 0);
  sse += (uint32_t)__msa_copy_u_w((v4i32)tmp0, 2);
  return sse;
}

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

#endif  // !defined(LIBYUV_DISABLE_MSA) && defined(__mips_msa)
