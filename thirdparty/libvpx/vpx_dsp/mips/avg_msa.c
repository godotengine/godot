/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#include <stdlib.h>

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/mips/macros_msa.h"

uint32_t vpx_avg_8x8_msa(const uint8_t *src, int32_t src_stride) {
  uint32_t sum_out;
  v16u8 src0, src1, src2, src3, src4, src5, src6, src7;
  v8u16 sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7;
  v4u32 sum = { 0 };

  LD_UB8(src, src_stride, src0, src1, src2, src3, src4, src5, src6, src7);
  HADD_UB4_UH(src0, src1, src2, src3, sum0, sum1, sum2, sum3);
  HADD_UB4_UH(src4, src5, src6, src7, sum4, sum5, sum6, sum7);
  ADD4(sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum0, sum2, sum4, sum6);
  ADD2(sum0, sum2, sum4, sum6, sum0, sum4);
  sum0 += sum4;

  sum = __msa_hadd_u_w(sum0, sum0);
  sum0 = (v8u16)__msa_pckev_h((v8i16)sum, (v8i16)sum);
  sum = __msa_hadd_u_w(sum0, sum0);
  sum = (v4u32)__msa_srari_w((v4i32)sum, 6);
  sum_out = __msa_copy_u_w((v4i32)sum, 0);

  return sum_out;
}

uint32_t vpx_avg_4x4_msa(const uint8_t *src, int32_t src_stride) {
  uint32_t sum_out;
  uint32_t src0, src1, src2, src3;
  v16u8 vec = { 0 };
  v8u16 sum0;
  v4u32 sum1;
  v2u64 sum2;

  LW4(src, src_stride, src0, src1, src2, src3);
  INSERT_W4_UB(src0, src1, src2, src3, vec);

  sum0 = __msa_hadd_u_h(vec, vec);
  sum1 = __msa_hadd_u_w(sum0, sum0);
  sum0 = (v8u16)__msa_pckev_h((v8i16)sum1, (v8i16)sum1);
  sum1 = __msa_hadd_u_w(sum0, sum0);
  sum2 = __msa_hadd_u_d(sum1, sum1);
  sum1 = (v4u32)__msa_srari_w((v4i32)sum2, 4);
  sum_out = __msa_copy_u_w((v4i32)sum1, 0);

  return sum_out;
}

#if !CONFIG_VP9_HIGHBITDEPTH
void vpx_hadamard_8x8_msa(const int16_t *src, ptrdiff_t src_stride,
                          int16_t *dst) {
  v8i16 src0, src1, src2, src3, src4, src5, src6, src7;
  v8i16 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

  LD_SH8(src, src_stride, src0, src1, src2, src3, src4, src5, src6, src7);
  BUTTERFLY_8(src0, src2, src4, src6, src7, src5, src3, src1, tmp0, tmp2, tmp4,
              tmp6, tmp7, tmp5, tmp3, tmp1);
  BUTTERFLY_8(tmp0, tmp1, tmp4, tmp5, tmp7, tmp6, tmp3, tmp2, src0, src1, src4,
              src5, src7, src6, src3, src2);
  BUTTERFLY_8(src0, src1, src2, src3, src7, src6, src5, src4, tmp0, tmp7, tmp3,
              tmp4, tmp5, tmp1, tmp6, tmp2);
  TRANSPOSE8x8_SH_SH(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, src0, src1,
                     src2, src3, src4, src5, src6, src7);
  BUTTERFLY_8(src0, src2, src4, src6, src7, src5, src3, src1, tmp0, tmp2, tmp4,
              tmp6, tmp7, tmp5, tmp3, tmp1);
  BUTTERFLY_8(tmp0, tmp1, tmp4, tmp5, tmp7, tmp6, tmp3, tmp2, src0, src1, src4,
              src5, src7, src6, src3, src2);
  BUTTERFLY_8(src0, src1, src2, src3, src7, src6, src5, src4, tmp0, tmp7, tmp3,
              tmp4, tmp5, tmp1, tmp6, tmp2);
  TRANSPOSE8x8_SH_SH(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, src0, src1,
                     src2, src3, src4, src5, src6, src7);
  ST_SH8(src0, src1, src2, src3, src4, src5, src6, src7, dst, 8);
}

void vpx_hadamard_16x16_msa(const int16_t *src, ptrdiff_t src_stride,
                            int16_t *dst) {
  v8i16 src0, src1, src2, src3, src4, src5, src6, src7, src8, src9, src10;
  v8i16 src11, src12, src13, src14, src15, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5;
  v8i16 tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12, tmp13, tmp14, tmp15;
  v8i16 res0, res1, res2, res3, res4, res5, res6, res7;

  LD_SH2(src, 8, src0, src8);
  src += src_stride;
  LD_SH2(src, 8, src1, src9);
  src += src_stride;
  LD_SH2(src, 8, src2, src10);
  src += src_stride;
  LD_SH2(src, 8, src3, src11);
  src += src_stride;
  LD_SH2(src, 8, src4, src12);
  src += src_stride;
  LD_SH2(src, 8, src5, src13);
  src += src_stride;
  LD_SH2(src, 8, src6, src14);
  src += src_stride;
  LD_SH2(src, 8, src7, src15);
  src += src_stride;

  BUTTERFLY_8(src0, src2, src4, src6, src7, src5, src3, src1, tmp0, tmp2, tmp4,
              tmp6, tmp7, tmp5, tmp3, tmp1);
  BUTTERFLY_8(src8, src10, src12, src14, src15, src13, src11, src9, tmp8, tmp10,
              tmp12, tmp14, tmp15, tmp13, tmp11, tmp9);

  BUTTERFLY_8(tmp0, tmp1, tmp4, tmp5, tmp7, tmp6, tmp3, tmp2, src0, src1, src4,
              src5, src7, src6, src3, src2);
  BUTTERFLY_8(src0, src1, src2, src3, src7, src6, src5, src4, tmp0, tmp7, tmp3,
              tmp4, tmp5, tmp1, tmp6, tmp2);
  TRANSPOSE8x8_SH_SH(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, src0, src1,
                     src2, src3, src4, src5, src6, src7);
  BUTTERFLY_8(src0, src2, src4, src6, src7, src5, src3, src1, tmp0, tmp2, tmp4,
              tmp6, tmp7, tmp5, tmp3, tmp1);
  BUTTERFLY_8(tmp0, tmp1, tmp4, tmp5, tmp7, tmp6, tmp3, tmp2, src0, src1, src4,
              src5, src7, src6, src3, src2);
  BUTTERFLY_8(src0, src1, src2, src3, src7, src6, src5, src4, tmp0, tmp7, tmp3,
              tmp4, tmp5, tmp1, tmp6, tmp2);
  TRANSPOSE8x8_SH_SH(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, src0, src1,
                     src2, src11, src4, src5, src6, src7);
  ST_SH8(src0, src1, src2, src11, src4, src5, src6, src7, dst, 8);

  BUTTERFLY_8(tmp8, tmp9, tmp12, tmp13, tmp15, tmp14, tmp11, tmp10, src8, src9,
              src12, src13, src15, src14, src11, src10);
  BUTTERFLY_8(src8, src9, src10, src11, src15, src14, src13, src12, tmp8, tmp15,
              tmp11, tmp12, tmp13, tmp9, tmp14, tmp10);
  TRANSPOSE8x8_SH_SH(tmp8, tmp9, tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, src8,
                     src9, src10, src11, src12, src13, src14, src15);
  BUTTERFLY_8(src8, src10, src12, src14, src15, src13, src11, src9, tmp8, tmp10,
              tmp12, tmp14, tmp15, tmp13, tmp11, tmp9);
  BUTTERFLY_8(tmp8, tmp9, tmp12, tmp13, tmp15, tmp14, tmp11, tmp10, src8, src9,
              src12, src13, src15, src14, src11, src10);
  BUTTERFLY_8(src8, src9, src10, src11, src15, src14, src13, src12, tmp8, tmp15,
              tmp11, tmp12, tmp13, tmp9, tmp14, tmp10);
  TRANSPOSE8x8_SH_SH(tmp8, tmp9, tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, res0,
                     res1, res2, res3, res4, res5, res6, res7);

  LD_SH2(src, 8, src0, src8);
  src += src_stride;
  LD_SH2(src, 8, src1, src9);
  src += src_stride;
  LD_SH2(src, 8, src2, src10);
  src += src_stride;
  LD_SH2(src, 8, src3, src11);
  src += src_stride;

  ST_SH8(res0, res1, res2, res3, res4, res5, res6, res7, dst + 64, 8);

  LD_SH2(src, 8, src4, src12);
  src += src_stride;
  LD_SH2(src, 8, src5, src13);
  src += src_stride;
  LD_SH2(src, 8, src6, src14);
  src += src_stride;
  LD_SH2(src, 8, src7, src15);
  src += src_stride;

  BUTTERFLY_8(src0, src2, src4, src6, src7, src5, src3, src1, tmp0, tmp2, tmp4,
              tmp6, tmp7, tmp5, tmp3, tmp1);
  BUTTERFLY_8(src8, src10, src12, src14, src15, src13, src11, src9, tmp8, tmp10,
              tmp12, tmp14, tmp15, tmp13, tmp11, tmp9);

  BUTTERFLY_8(tmp0, tmp1, tmp4, tmp5, tmp7, tmp6, tmp3, tmp2, src0, src1, src4,
              src5, src7, src6, src3, src2);
  BUTTERFLY_8(src0, src1, src2, src3, src7, src6, src5, src4, tmp0, tmp7, tmp3,
              tmp4, tmp5, tmp1, tmp6, tmp2);
  TRANSPOSE8x8_SH_SH(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, src0, src1,
                     src2, src3, src4, src5, src6, src7);
  BUTTERFLY_8(src0, src2, src4, src6, src7, src5, src3, src1, tmp0, tmp2, tmp4,
              tmp6, tmp7, tmp5, tmp3, tmp1);
  BUTTERFLY_8(tmp0, tmp1, tmp4, tmp5, tmp7, tmp6, tmp3, tmp2, src0, src1, src4,
              src5, src7, src6, src3, src2);
  BUTTERFLY_8(src0, src1, src2, src3, src7, src6, src5, src4, tmp0, tmp7, tmp3,
              tmp4, tmp5, tmp1, tmp6, tmp2);
  TRANSPOSE8x8_SH_SH(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, src0, src1,
                     src2, src3, src4, src5, src6, src7);
  ST_SH8(src0, src1, src2, src3, src4, src5, src6, src7, dst + 2 * 64, 8);

  BUTTERFLY_8(tmp8, tmp9, tmp12, tmp13, tmp15, tmp14, tmp11, tmp10, src8, src9,
              src12, src13, src15, src14, src11, src10);
  BUTTERFLY_8(src8, src9, src10, src11, src15, src14, src13, src12, tmp8, tmp15,
              tmp11, tmp12, tmp13, tmp9, tmp14, tmp10);
  TRANSPOSE8x8_SH_SH(tmp8, tmp9, tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, src8,
                     src9, src10, src11, src12, src13, src14, src15);
  BUTTERFLY_8(src8, src10, src12, src14, src15, src13, src11, src9, tmp8, tmp10,
              tmp12, tmp14, tmp15, tmp13, tmp11, tmp9);
  BUTTERFLY_8(tmp8, tmp9, tmp12, tmp13, tmp15, tmp14, tmp11, tmp10, src8, src9,
              src12, src13, src15, src14, src11, src10);
  BUTTERFLY_8(src8, src9, src10, src11, src15, src14, src13, src12, tmp8, tmp15,
              tmp11, tmp12, tmp13, tmp9, tmp14, tmp10);
  TRANSPOSE8x8_SH_SH(tmp8, tmp9, tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, res0,
                     res1, res2, res3, res4, res5, res6, res7);
  ST_SH8(res0, res1, res2, res3, res4, res5, res6, res7, dst + 3 * 64, 8);

  LD_SH4(dst, 64, src0, src1, src2, src3);
  LD_SH4(dst + 8, 64, src4, src5, src6, src7);

  BUTTERFLY_8(src0, src2, src4, src6, src7, src5, src3, src1, tmp0, tmp2, tmp4,
              tmp6, tmp7, tmp5, tmp3, tmp1);
  SRA_4V(tmp0, tmp1, tmp2, tmp3, 1);
  SRA_4V(tmp4, tmp5, tmp6, tmp7, 1);
  BUTTERFLY_8(tmp0, tmp1, tmp4, tmp5, tmp7, tmp6, tmp3, tmp2, src0, src1, src4,
              src5, src7, src6, src3, src2);

  ST_SH4(src0, src1, src2, src3, dst, 64);
  ST_SH4(src4, src5, src6, src7, dst + 8, 64);
  dst += 16;

  LD_SH4(dst, 64, src0, src1, src2, src3);
  LD_SH4(dst + 8, 64, src4, src5, src6, src7);

  BUTTERFLY_8(src0, src2, src4, src6, src7, src5, src3, src1, tmp0, tmp2, tmp4,
              tmp6, tmp7, tmp5, tmp3, tmp1);
  SRA_4V(tmp0, tmp1, tmp2, tmp3, 1);
  SRA_4V(tmp4, tmp5, tmp6, tmp7, 1);
  BUTTERFLY_8(tmp0, tmp1, tmp4, tmp5, tmp7, tmp6, tmp3, tmp2, src0, src1, src4,
              src5, src7, src6, src3, src2);

  ST_SH4(src0, src1, src2, src3, dst, 64);
  ST_SH4(src4, src5, src6, src7, dst + 8, 64);
  dst += 16;

  LD_SH4(dst, 64, src0, src1, src2, src3);
  LD_SH4(dst + 8, 64, src4, src5, src6, src7);

  BUTTERFLY_8(src0, src2, src4, src6, src7, src5, src3, src1, tmp0, tmp2, tmp4,
              tmp6, tmp7, tmp5, tmp3, tmp1);
  SRA_4V(tmp0, tmp1, tmp2, tmp3, 1);
  SRA_4V(tmp4, tmp5, tmp6, tmp7, 1);
  BUTTERFLY_8(tmp0, tmp1, tmp4, tmp5, tmp7, tmp6, tmp3, tmp2, src0, src1, src4,
              src5, src7, src6, src3, src2);

  ST_SH4(src0, src1, src2, src3, dst, 64);
  ST_SH4(src4, src5, src6, src7, dst + 8, 64);
  dst += 16;

  LD_SH4(dst, 64, src0, src1, src2, src3);
  LD_SH4(dst + 8, 64, src4, src5, src6, src7);

  BUTTERFLY_8(src0, src2, src4, src6, src7, src5, src3, src1, tmp0, tmp2, tmp4,
              tmp6, tmp7, tmp5, tmp3, tmp1);
  SRA_4V(tmp0, tmp1, tmp2, tmp3, 1);
  SRA_4V(tmp4, tmp5, tmp6, tmp7, 1);
  BUTTERFLY_8(tmp0, tmp1, tmp4, tmp5, tmp7, tmp6, tmp3, tmp2, src0, src1, src4,
              src5, src7, src6, src3, src2);

  ST_SH4(src0, src1, src2, src3, dst, 64);
  ST_SH4(src4, src5, src6, src7, dst + 8, 64);
}

int vpx_satd_msa(const int16_t *data, int length) {
  int i, satd;
  v8i16 src0, src1, src2, src3, src4, src5, src6, src7;
  v8i16 src8, src9, src10, src11, src12, src13, src14, src15;
  v8i16 zero = { 0 };
  v8u16 tmp0_h, tmp1_h, tmp2_h, tmp3_h, tmp4_h, tmp5_h, tmp6_h, tmp7_h;
  v4u32 tmp0_w = { 0 };

  if (16 == length) {
    LD_SH2(data, 8, src0, src1);
    tmp0_h = (v8u16)__msa_asub_s_h(src0, zero);
    tmp1_h = (v8u16)__msa_asub_s_h(src1, zero);
    tmp0_w = __msa_hadd_u_w(tmp0_h, tmp0_h);
    tmp0_w += __msa_hadd_u_w(tmp1_h, tmp1_h);
    satd = HADD_UW_U32(tmp0_w);
  } else if (64 == length) {
    LD_SH8(data, 8, src0, src1, src2, src3, src4, src5, src6, src7);

    tmp0_h = (v8u16)__msa_asub_s_h(src0, zero);
    tmp1_h = (v8u16)__msa_asub_s_h(src1, zero);
    tmp2_h = (v8u16)__msa_asub_s_h(src2, zero);
    tmp3_h = (v8u16)__msa_asub_s_h(src3, zero);
    tmp4_h = (v8u16)__msa_asub_s_h(src4, zero);
    tmp5_h = (v8u16)__msa_asub_s_h(src5, zero);
    tmp6_h = (v8u16)__msa_asub_s_h(src6, zero);
    tmp7_h = (v8u16)__msa_asub_s_h(src7, zero);

    tmp0_w = __msa_hadd_u_w(tmp0_h, tmp0_h);
    tmp0_w += __msa_hadd_u_w(tmp1_h, tmp1_h);
    tmp0_w += __msa_hadd_u_w(tmp2_h, tmp2_h);
    tmp0_w += __msa_hadd_u_w(tmp3_h, tmp3_h);
    tmp0_w += __msa_hadd_u_w(tmp4_h, tmp4_h);
    tmp0_w += __msa_hadd_u_w(tmp5_h, tmp5_h);
    tmp0_w += __msa_hadd_u_w(tmp6_h, tmp6_h);
    tmp0_w += __msa_hadd_u_w(tmp7_h, tmp7_h);

    satd = HADD_UW_U32(tmp0_w);
  } else if (256 == length) {
    for (i = 0; i < 2; ++i) {
      LD_SH8(data, 8, src0, src1, src2, src3, src4, src5, src6, src7);
      data += 8 * 8;
      LD_SH8(data, 8, src8, src9, src10, src11, src12, src13, src14, src15);
      data += 8 * 8;

      tmp0_h = (v8u16)__msa_asub_s_h(src0, zero);
      tmp1_h = (v8u16)__msa_asub_s_h(src1, zero);
      tmp2_h = (v8u16)__msa_asub_s_h(src2, zero);
      tmp3_h = (v8u16)__msa_asub_s_h(src3, zero);
      tmp4_h = (v8u16)__msa_asub_s_h(src4, zero);
      tmp5_h = (v8u16)__msa_asub_s_h(src5, zero);
      tmp6_h = (v8u16)__msa_asub_s_h(src6, zero);
      tmp7_h = (v8u16)__msa_asub_s_h(src7, zero);

      tmp0_w += __msa_hadd_u_w(tmp0_h, tmp0_h);
      tmp0_w += __msa_hadd_u_w(tmp1_h, tmp1_h);
      tmp0_w += __msa_hadd_u_w(tmp2_h, tmp2_h);
      tmp0_w += __msa_hadd_u_w(tmp3_h, tmp3_h);
      tmp0_w += __msa_hadd_u_w(tmp4_h, tmp4_h);
      tmp0_w += __msa_hadd_u_w(tmp5_h, tmp5_h);
      tmp0_w += __msa_hadd_u_w(tmp6_h, tmp6_h);
      tmp0_w += __msa_hadd_u_w(tmp7_h, tmp7_h);

      tmp0_h = (v8u16)__msa_asub_s_h(src8, zero);
      tmp1_h = (v8u16)__msa_asub_s_h(src9, zero);
      tmp2_h = (v8u16)__msa_asub_s_h(src10, zero);
      tmp3_h = (v8u16)__msa_asub_s_h(src11, zero);
      tmp4_h = (v8u16)__msa_asub_s_h(src12, zero);
      tmp5_h = (v8u16)__msa_asub_s_h(src13, zero);
      tmp6_h = (v8u16)__msa_asub_s_h(src14, zero);
      tmp7_h = (v8u16)__msa_asub_s_h(src15, zero);

      tmp0_w += __msa_hadd_u_w(tmp0_h, tmp0_h);
      tmp0_w += __msa_hadd_u_w(tmp1_h, tmp1_h);
      tmp0_w += __msa_hadd_u_w(tmp2_h, tmp2_h);
      tmp0_w += __msa_hadd_u_w(tmp3_h, tmp3_h);
      tmp0_w += __msa_hadd_u_w(tmp4_h, tmp4_h);
      tmp0_w += __msa_hadd_u_w(tmp5_h, tmp5_h);
      tmp0_w += __msa_hadd_u_w(tmp6_h, tmp6_h);
      tmp0_w += __msa_hadd_u_w(tmp7_h, tmp7_h);
    }

    satd = HADD_UW_U32(tmp0_w);
  } else if (1024 == length) {
    for (i = 0; i < 8; ++i) {
      LD_SH8(data, 8, src0, src1, src2, src3, src4, src5, src6, src7);
      data += 8 * 8;
      LD_SH8(data, 8, src8, src9, src10, src11, src12, src13, src14, src15);
      data += 8 * 8;

      tmp0_h = (v8u16)__msa_asub_s_h(src0, zero);
      tmp1_h = (v8u16)__msa_asub_s_h(src1, zero);
      tmp2_h = (v8u16)__msa_asub_s_h(src2, zero);
      tmp3_h = (v8u16)__msa_asub_s_h(src3, zero);
      tmp4_h = (v8u16)__msa_asub_s_h(src4, zero);
      tmp5_h = (v8u16)__msa_asub_s_h(src5, zero);
      tmp6_h = (v8u16)__msa_asub_s_h(src6, zero);
      tmp7_h = (v8u16)__msa_asub_s_h(src7, zero);

      tmp0_w += __msa_hadd_u_w(tmp0_h, tmp0_h);
      tmp0_w += __msa_hadd_u_w(tmp1_h, tmp1_h);
      tmp0_w += __msa_hadd_u_w(tmp2_h, tmp2_h);
      tmp0_w += __msa_hadd_u_w(tmp3_h, tmp3_h);
      tmp0_w += __msa_hadd_u_w(tmp4_h, tmp4_h);
      tmp0_w += __msa_hadd_u_w(tmp5_h, tmp5_h);
      tmp0_w += __msa_hadd_u_w(tmp6_h, tmp6_h);
      tmp0_w += __msa_hadd_u_w(tmp7_h, tmp7_h);

      tmp0_h = (v8u16)__msa_asub_s_h(src8, zero);
      tmp1_h = (v8u16)__msa_asub_s_h(src9, zero);
      tmp2_h = (v8u16)__msa_asub_s_h(src10, zero);
      tmp3_h = (v8u16)__msa_asub_s_h(src11, zero);
      tmp4_h = (v8u16)__msa_asub_s_h(src12, zero);
      tmp5_h = (v8u16)__msa_asub_s_h(src13, zero);
      tmp6_h = (v8u16)__msa_asub_s_h(src14, zero);
      tmp7_h = (v8u16)__msa_asub_s_h(src15, zero);

      tmp0_w += __msa_hadd_u_w(tmp0_h, tmp0_h);
      tmp0_w += __msa_hadd_u_w(tmp1_h, tmp1_h);
      tmp0_w += __msa_hadd_u_w(tmp2_h, tmp2_h);
      tmp0_w += __msa_hadd_u_w(tmp3_h, tmp3_h);
      tmp0_w += __msa_hadd_u_w(tmp4_h, tmp4_h);
      tmp0_w += __msa_hadd_u_w(tmp5_h, tmp5_h);
      tmp0_w += __msa_hadd_u_w(tmp6_h, tmp6_h);
      tmp0_w += __msa_hadd_u_w(tmp7_h, tmp7_h);
    }

    satd = HADD_UW_U32(tmp0_w);
  } else {
    satd = 0;

    for (i = 0; i < length; ++i) {
      satd += abs(data[i]);
    }
  }

  return satd;
}
#endif  // !CONFIG_VP9_HIGHBITDEPTH

void vpx_int_pro_row_msa(int16_t hbuf[16], const uint8_t *ref,
                         const int ref_stride, const int height) {
  int i;
  v16u8 ref0, ref1, ref2, ref3, ref4, ref5, ref6, ref7;
  v8i16 hbuf_r = { 0 };
  v8i16 hbuf_l = { 0 };
  v8i16 ref0_r, ref0_l, ref1_r, ref1_l, ref2_r, ref2_l, ref3_r, ref3_l;
  v8i16 ref4_r, ref4_l, ref5_r, ref5_l, ref6_r, ref6_l, ref7_r, ref7_l;

  if (16 == height) {
    for (i = 2; i--;) {
      LD_UB8(ref, ref_stride, ref0, ref1, ref2, ref3, ref4, ref5, ref6, ref7);
      ref += 8 * ref_stride;
      UNPCK_UB_SH(ref0, ref0_r, ref0_l);
      UNPCK_UB_SH(ref1, ref1_r, ref1_l);
      UNPCK_UB_SH(ref2, ref2_r, ref2_l);
      UNPCK_UB_SH(ref3, ref3_r, ref3_l);
      UNPCK_UB_SH(ref4, ref4_r, ref4_l);
      UNPCK_UB_SH(ref5, ref5_r, ref5_l);
      UNPCK_UB_SH(ref6, ref6_r, ref6_l);
      UNPCK_UB_SH(ref7, ref7_r, ref7_l);
      ADD4(hbuf_r, ref0_r, hbuf_l, ref0_l, hbuf_r, ref1_r, hbuf_l, ref1_l,
           hbuf_r, hbuf_l, hbuf_r, hbuf_l);
      ADD4(hbuf_r, ref2_r, hbuf_l, ref2_l, hbuf_r, ref3_r, hbuf_l, ref3_l,
           hbuf_r, hbuf_l, hbuf_r, hbuf_l);
      ADD4(hbuf_r, ref4_r, hbuf_l, ref4_l, hbuf_r, ref5_r, hbuf_l, ref5_l,
           hbuf_r, hbuf_l, hbuf_r, hbuf_l);
      ADD4(hbuf_r, ref6_r, hbuf_l, ref6_l, hbuf_r, ref7_r, hbuf_l, ref7_l,
           hbuf_r, hbuf_l, hbuf_r, hbuf_l);
    }

    SRA_2V(hbuf_r, hbuf_l, 3);
    ST_SH2(hbuf_r, hbuf_l, hbuf, 8);
  } else if (32 == height) {
    for (i = 2; i--;) {
      LD_UB8(ref, ref_stride, ref0, ref1, ref2, ref3, ref4, ref5, ref6, ref7);
      ref += 8 * ref_stride;
      UNPCK_UB_SH(ref0, ref0_r, ref0_l);
      UNPCK_UB_SH(ref1, ref1_r, ref1_l);
      UNPCK_UB_SH(ref2, ref2_r, ref2_l);
      UNPCK_UB_SH(ref3, ref3_r, ref3_l);
      UNPCK_UB_SH(ref4, ref4_r, ref4_l);
      UNPCK_UB_SH(ref5, ref5_r, ref5_l);
      UNPCK_UB_SH(ref6, ref6_r, ref6_l);
      UNPCK_UB_SH(ref7, ref7_r, ref7_l);
      ADD4(hbuf_r, ref0_r, hbuf_l, ref0_l, hbuf_r, ref1_r, hbuf_l, ref1_l,
           hbuf_r, hbuf_l, hbuf_r, hbuf_l);
      ADD4(hbuf_r, ref2_r, hbuf_l, ref2_l, hbuf_r, ref3_r, hbuf_l, ref3_l,
           hbuf_r, hbuf_l, hbuf_r, hbuf_l);
      ADD4(hbuf_r, ref4_r, hbuf_l, ref4_l, hbuf_r, ref5_r, hbuf_l, ref5_l,
           hbuf_r, hbuf_l, hbuf_r, hbuf_l);
      ADD4(hbuf_r, ref6_r, hbuf_l, ref6_l, hbuf_r, ref7_r, hbuf_l, ref7_l,
           hbuf_r, hbuf_l, hbuf_r, hbuf_l);
      LD_UB8(ref, ref_stride, ref0, ref1, ref2, ref3, ref4, ref5, ref6, ref7);
      ref += 8 * ref_stride;
      UNPCK_UB_SH(ref0, ref0_r, ref0_l);
      UNPCK_UB_SH(ref1, ref1_r, ref1_l);
      UNPCK_UB_SH(ref2, ref2_r, ref2_l);
      UNPCK_UB_SH(ref3, ref3_r, ref3_l);
      UNPCK_UB_SH(ref4, ref4_r, ref4_l);
      UNPCK_UB_SH(ref5, ref5_r, ref5_l);
      UNPCK_UB_SH(ref6, ref6_r, ref6_l);
      UNPCK_UB_SH(ref7, ref7_r, ref7_l);
      ADD4(hbuf_r, ref0_r, hbuf_l, ref0_l, hbuf_r, ref1_r, hbuf_l, ref1_l,
           hbuf_r, hbuf_l, hbuf_r, hbuf_l);
      ADD4(hbuf_r, ref2_r, hbuf_l, ref2_l, hbuf_r, ref3_r, hbuf_l, ref3_l,
           hbuf_r, hbuf_l, hbuf_r, hbuf_l);
      ADD4(hbuf_r, ref4_r, hbuf_l, ref4_l, hbuf_r, ref5_r, hbuf_l, ref5_l,
           hbuf_r, hbuf_l, hbuf_r, hbuf_l);
      ADD4(hbuf_r, ref6_r, hbuf_l, ref6_l, hbuf_r, ref7_r, hbuf_l, ref7_l,
           hbuf_r, hbuf_l, hbuf_r, hbuf_l);
    }

    SRA_2V(hbuf_r, hbuf_l, 4);
    ST_SH2(hbuf_r, hbuf_l, hbuf, 8);
  } else if (64 == height) {
    for (i = 4; i--;) {
      LD_UB8(ref, ref_stride, ref0, ref1, ref2, ref3, ref4, ref5, ref6, ref7);
      ref += 8 * ref_stride;
      UNPCK_UB_SH(ref0, ref0_r, ref0_l);
      UNPCK_UB_SH(ref1, ref1_r, ref1_l);
      UNPCK_UB_SH(ref2, ref2_r, ref2_l);
      UNPCK_UB_SH(ref3, ref3_r, ref3_l);
      UNPCK_UB_SH(ref4, ref4_r, ref4_l);
      UNPCK_UB_SH(ref5, ref5_r, ref5_l);
      UNPCK_UB_SH(ref6, ref6_r, ref6_l);
      UNPCK_UB_SH(ref7, ref7_r, ref7_l);
      ADD4(hbuf_r, ref0_r, hbuf_l, ref0_l, hbuf_r, ref1_r, hbuf_l, ref1_l,
           hbuf_r, hbuf_l, hbuf_r, hbuf_l);
      ADD4(hbuf_r, ref2_r, hbuf_l, ref2_l, hbuf_r, ref3_r, hbuf_l, ref3_l,
           hbuf_r, hbuf_l, hbuf_r, hbuf_l);
      ADD4(hbuf_r, ref4_r, hbuf_l, ref4_l, hbuf_r, ref5_r, hbuf_l, ref5_l,
           hbuf_r, hbuf_l, hbuf_r, hbuf_l);
      ADD4(hbuf_r, ref6_r, hbuf_l, ref6_l, hbuf_r, ref7_r, hbuf_l, ref7_l,
           hbuf_r, hbuf_l, hbuf_r, hbuf_l);
      LD_UB8(ref, ref_stride, ref0, ref1, ref2, ref3, ref4, ref5, ref6, ref7);
      ref += 8 * ref_stride;
      UNPCK_UB_SH(ref0, ref0_r, ref0_l);
      UNPCK_UB_SH(ref1, ref1_r, ref1_l);
      UNPCK_UB_SH(ref2, ref2_r, ref2_l);
      UNPCK_UB_SH(ref3, ref3_r, ref3_l);
      UNPCK_UB_SH(ref4, ref4_r, ref4_l);
      UNPCK_UB_SH(ref5, ref5_r, ref5_l);
      UNPCK_UB_SH(ref6, ref6_r, ref6_l);
      UNPCK_UB_SH(ref7, ref7_r, ref7_l);
      ADD4(hbuf_r, ref0_r, hbuf_l, ref0_l, hbuf_r, ref1_r, hbuf_l, ref1_l,
           hbuf_r, hbuf_l, hbuf_r, hbuf_l);
      ADD4(hbuf_r, ref2_r, hbuf_l, ref2_l, hbuf_r, ref3_r, hbuf_l, ref3_l,
           hbuf_r, hbuf_l, hbuf_r, hbuf_l);
      ADD4(hbuf_r, ref4_r, hbuf_l, ref4_l, hbuf_r, ref5_r, hbuf_l, ref5_l,
           hbuf_r, hbuf_l, hbuf_r, hbuf_l);
      ADD4(hbuf_r, ref6_r, hbuf_l, ref6_l, hbuf_r, ref7_r, hbuf_l, ref7_l,
           hbuf_r, hbuf_l, hbuf_r, hbuf_l);
    }

    SRA_2V(hbuf_r, hbuf_l, 5);
    ST_SH2(hbuf_r, hbuf_l, hbuf, 8);
  } else {
    const int norm_factor = height >> 1;
    int cnt;

    for (cnt = 0; cnt < 16; cnt++) {
      hbuf[cnt] = 0;
    }

    for (i = 0; i < height; ++i) {
      for (cnt = 0; cnt < 16; cnt++) {
        hbuf[cnt] += ref[cnt];
      }

      ref += ref_stride;
    }

    for (cnt = 0; cnt < 16; cnt++) {
      hbuf[cnt] /= norm_factor;
    }
  }
}

int16_t vpx_int_pro_col_msa(const uint8_t *ref, const int width) {
  int16_t sum;
  v16u8 ref0, ref1, ref2, ref3;
  v8u16 ref0_h;

  if (16 == width) {
    ref0 = LD_UB(ref);
    ref0_h = __msa_hadd_u_h(ref0, ref0);
    sum = HADD_UH_U32(ref0_h);
  } else if (32 == width) {
    LD_UB2(ref, 16, ref0, ref1);
    ref0_h = __msa_hadd_u_h(ref0, ref0);
    ref0_h += __msa_hadd_u_h(ref1, ref1);
    sum = HADD_UH_U32(ref0_h);
  } else if (64 == width) {
    LD_UB4(ref, 16, ref0, ref1, ref2, ref3);
    ref0_h = __msa_hadd_u_h(ref0, ref0);
    ref0_h += __msa_hadd_u_h(ref1, ref1);
    ref0_h += __msa_hadd_u_h(ref2, ref2);
    ref0_h += __msa_hadd_u_h(ref3, ref3);
    sum = HADD_UH_U32(ref0_h);
  } else {
    int idx;

    sum = 0;
    for (idx = 0; idx < width; ++idx) {
      sum += ref[idx];
    }
  }

  return sum;
}

int vpx_vector_var_msa(const int16_t *ref, const int16_t *src, const int bwl) {
  int sse, mean, var;
  v8i16 src0, src1, src2, src3, src4, src5, src6, src7, ref0, ref1, ref2;
  v8i16 ref3, ref4, ref5, ref6, ref7, src_l0_m, src_l1_m, src_l2_m, src_l3_m;
  v8i16 src_l4_m, src_l5_m, src_l6_m, src_l7_m;
  v4i32 res_l0_m, res_l1_m, res_l2_m, res_l3_m, res_l4_m, res_l5_m, res_l6_m;
  v4i32 res_l7_m, mean_v;
  v2i64 sse_v;

  if (2 == bwl) {
    LD_SH2(src, 8, src0, src1);
    LD_SH2(ref, 8, ref0, ref1);

    ILVRL_H2_SH(src0, ref0, src_l0_m, src_l1_m);
    ILVRL_H2_SH(src1, ref1, src_l2_m, src_l3_m);
    HSUB_UH2_SW(src_l0_m, src_l1_m, res_l0_m, res_l1_m);
    HSUB_UH2_SW(src_l2_m, src_l3_m, res_l2_m, res_l3_m);
    sse_v = __msa_dotp_s_d(res_l0_m, res_l0_m);
    sse_v = __msa_dpadd_s_d(sse_v, res_l1_m, res_l1_m);
    DPADD_SD2_SD(res_l2_m, res_l3_m, sse_v, sse_v);
    mean_v = res_l0_m + res_l1_m;
    mean_v += res_l2_m + res_l3_m;

    sse_v += __msa_splati_d(sse_v, 1);
    sse = __msa_copy_s_w((v4i32)sse_v, 0);

    mean = HADD_SW_S32(mean_v);
  } else if (3 == bwl) {
    LD_SH4(src, 8, src0, src1, src2, src3);
    LD_SH4(ref, 8, ref0, ref1, ref2, ref3);

    ILVRL_H2_SH(src0, ref0, src_l0_m, src_l1_m);
    ILVRL_H2_SH(src1, ref1, src_l2_m, src_l3_m);
    ILVRL_H2_SH(src2, ref2, src_l4_m, src_l5_m);
    ILVRL_H2_SH(src3, ref3, src_l6_m, src_l7_m);
    HSUB_UH2_SW(src_l0_m, src_l1_m, res_l0_m, res_l1_m);
    HSUB_UH2_SW(src_l2_m, src_l3_m, res_l2_m, res_l3_m);
    HSUB_UH2_SW(src_l4_m, src_l5_m, res_l4_m, res_l5_m);
    HSUB_UH2_SW(src_l6_m, src_l7_m, res_l6_m, res_l7_m);
    sse_v = __msa_dotp_s_d(res_l0_m, res_l0_m);
    sse_v = __msa_dpadd_s_d(sse_v, res_l1_m, res_l1_m);
    DPADD_SD2_SD(res_l2_m, res_l3_m, sse_v, sse_v);
    DPADD_SD2_SD(res_l4_m, res_l5_m, sse_v, sse_v);
    DPADD_SD2_SD(res_l6_m, res_l7_m, sse_v, sse_v);
    mean_v = res_l0_m + res_l1_m;
    mean_v += res_l2_m + res_l3_m;
    mean_v += res_l4_m + res_l5_m;
    mean_v += res_l6_m + res_l7_m;

    sse_v += __msa_splati_d(sse_v, 1);
    sse = __msa_copy_s_w((v4i32)sse_v, 0);

    mean = HADD_SW_S32(mean_v);
  } else if (4 == bwl) {
    LD_SH8(src, 8, src0, src1, src2, src3, src4, src5, src6, src7);
    LD_SH8(ref, 8, ref0, ref1, ref2, ref3, ref4, ref5, ref6, ref7);

    ILVRL_H2_SH(src0, ref0, src_l0_m, src_l1_m);
    ILVRL_H2_SH(src1, ref1, src_l2_m, src_l3_m);
    ILVRL_H2_SH(src2, ref2, src_l4_m, src_l5_m);
    ILVRL_H2_SH(src3, ref3, src_l6_m, src_l7_m);
    HSUB_UH2_SW(src_l0_m, src_l1_m, res_l0_m, res_l1_m);
    HSUB_UH2_SW(src_l2_m, src_l3_m, res_l2_m, res_l3_m);
    HSUB_UH2_SW(src_l4_m, src_l5_m, res_l4_m, res_l5_m);
    HSUB_UH2_SW(src_l6_m, src_l7_m, res_l6_m, res_l7_m);
    sse_v = __msa_dotp_s_d(res_l0_m, res_l0_m);
    sse_v = __msa_dpadd_s_d(sse_v, res_l1_m, res_l1_m);
    DPADD_SD2_SD(res_l2_m, res_l3_m, sse_v, sse_v);
    DPADD_SD2_SD(res_l4_m, res_l5_m, sse_v, sse_v);
    DPADD_SD2_SD(res_l6_m, res_l7_m, sse_v, sse_v);
    mean_v = res_l0_m + res_l1_m;
    mean_v += res_l2_m + res_l3_m;
    mean_v += res_l4_m + res_l5_m;
    mean_v += res_l6_m + res_l7_m;

    ILVRL_H2_SH(src4, ref4, src_l0_m, src_l1_m);
    ILVRL_H2_SH(src5, ref5, src_l2_m, src_l3_m);
    ILVRL_H2_SH(src6, ref6, src_l4_m, src_l5_m);
    ILVRL_H2_SH(src7, ref7, src_l6_m, src_l7_m);
    HSUB_UH2_SW(src_l0_m, src_l1_m, res_l0_m, res_l1_m);
    HSUB_UH2_SW(src_l2_m, src_l3_m, res_l2_m, res_l3_m);
    HSUB_UH2_SW(src_l4_m, src_l5_m, res_l4_m, res_l5_m);
    HSUB_UH2_SW(src_l6_m, src_l7_m, res_l6_m, res_l7_m);
    DPADD_SD2_SD(res_l0_m, res_l1_m, sse_v, sse_v);
    DPADD_SD2_SD(res_l2_m, res_l3_m, sse_v, sse_v);
    DPADD_SD2_SD(res_l4_m, res_l5_m, sse_v, sse_v);
    DPADD_SD2_SD(res_l6_m, res_l7_m, sse_v, sse_v);
    mean_v += res_l0_m + res_l1_m;
    mean_v += res_l2_m + res_l3_m;
    mean_v += res_l4_m + res_l5_m;
    mean_v += res_l6_m + res_l7_m;

    sse_v += __msa_splati_d(sse_v, 1);
    sse = __msa_copy_s_w((v4i32)sse_v, 0);

    mean = HADD_SW_S32(mean_v);
  } else {
    int i;
    const int width = 4 << bwl;

    sse = 0;
    mean = 0;

    for (i = 0; i < width; ++i) {
      const int diff = ref[i] - src[i];

      mean += diff;
      sse += diff * diff;
    }
  }

  var = sse - ((mean * mean) >> (bwl + 2));

  return var;
}

void vpx_minmax_8x8_msa(const uint8_t *s, int p, const uint8_t *d, int dp,
                        int *min, int *max) {
  v16u8 s0, s1, s2, s3, s4, s5, s6, s7, d0, d1, d2, d3, d4, d5, d6, d7;
  v16u8 diff0, diff1, diff2, diff3, min0, min1, max0, max1;

  LD_UB8(s, p, s0, s1, s2, s3, s4, s5, s6, s7);
  LD_UB8(d, dp, d0, d1, d2, d3, d4, d5, d6, d7);
  PCKEV_D4_UB(s1, s0, s3, s2, s5, s4, s7, s6, s0, s1, s2, s3);
  PCKEV_D4_UB(d1, d0, d3, d2, d5, d4, d7, d6, d0, d1, d2, d3);

  diff0 = __msa_asub_u_b(s0, d0);
  diff1 = __msa_asub_u_b(s1, d1);
  diff2 = __msa_asub_u_b(s2, d2);
  diff3 = __msa_asub_u_b(s3, d3);

  min0 = __msa_min_u_b(diff0, diff1);
  min1 = __msa_min_u_b(diff2, diff3);
  min0 = __msa_min_u_b(min0, min1);

  max0 = __msa_max_u_b(diff0, diff1);
  max1 = __msa_max_u_b(diff2, diff3);
  max0 = __msa_max_u_b(max0, max1);

  min1 = (v16u8)__msa_sldi_b((v16i8)min1, (v16i8)min0, 8);
  min0 = __msa_min_u_b(min0, min1);
  max1 = (v16u8)__msa_sldi_b((v16i8)max1, (v16i8)max0, 8);
  max0 = __msa_max_u_b(max0, max1);

  min1 = (v16u8)__msa_sldi_b((v16i8)min1, (v16i8)min0, 4);
  min0 = __msa_min_u_b(min0, min1);
  max1 = (v16u8)__msa_sldi_b((v16i8)max1, (v16i8)max0, 4);
  max0 = __msa_max_u_b(max0, max1);

  min1 = (v16u8)__msa_sldi_b((v16i8)min1, (v16i8)min0, 2);
  min0 = __msa_min_u_b(min0, min1);
  max1 = (v16u8)__msa_sldi_b((v16i8)max1, (v16i8)max0, 2);
  max0 = __msa_max_u_b(max0, max1);

  min1 = (v16u8)__msa_sldi_b((v16i8)min1, (v16i8)min0, 1);
  min0 = __msa_min_u_b(min0, min1);
  max1 = (v16u8)__msa_sldi_b((v16i8)max1, (v16i8)max0, 1);
  max0 = __msa_max_u_b(max0, max1);

  *min = min0[0];
  *max = max0[0];
}
