/*
 *  Copyright (c) 2022 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/loongarch/variance_lsx.h"

#define VARIANCE_WxH(sse, diff, shift) \
  (sse) - (((uint32_t)(diff) * (diff)) >> (shift))

#define VARIANCE_LARGE_WxH(sse, diff, shift) \
  (sse) - (((int64_t)(diff) * (diff)) >> (shift))

static uint32_t sse_diff_8width_lsx(const uint8_t *src_ptr, int32_t src_stride,
                                    const uint8_t *ref_ptr, int32_t ref_stride,
                                    int32_t height, int32_t *diff) {
  int32_t res, ht_cnt = (height >> 2);
  __m128i src0, src1, src2, src3, ref0, ref1, ref2, ref3, vec;
  __m128i avg = __lsx_vldi(0);
  __m128i var = avg;
  int32_t src_stride2 = src_stride << 1;
  int32_t src_stride3 = src_stride2 + src_stride;
  int32_t src_stride4 = src_stride2 << 1;
  int32_t ref_stride2 = ref_stride << 1;
  int32_t ref_stride3 = ref_stride2 + ref_stride;
  int32_t ref_stride4 = ref_stride2 << 1;

  for (; ht_cnt--;) {
    DUP4_ARG2(__lsx_vld, src_ptr, 0, src_ptr + src_stride, 0,
              src_ptr + src_stride2, 0, src_ptr + src_stride3, 0, src0, src1,
              src2, src3);
    src_ptr += src_stride4;
    DUP4_ARG2(__lsx_vld, ref_ptr, 0, ref_ptr + ref_stride, 0,
              ref_ptr + ref_stride2, 0, ref_ptr + ref_stride3, 0, ref0, ref1,
              ref2, ref3);
    ref_ptr += ref_stride4;

    DUP4_ARG2(__lsx_vpickev_d, src1, src0, src3, src2, ref1, ref0, ref3, ref2,
              src0, src1, ref0, ref1);
    CALC_MSE_AVG_B(src0, ref0, var, avg);
    CALC_MSE_AVG_B(src1, ref1, var, avg);
  }

  vec = __lsx_vhaddw_w_h(avg, avg);
  HADD_SW_S32(vec, *diff);
  HADD_SW_S32(var, res);
  return res;
}

static uint32_t sse_diff_16width_lsx(const uint8_t *src_ptr, int32_t src_stride,
                                     const uint8_t *ref_ptr, int32_t ref_stride,
                                     int32_t height, int32_t *diff) {
  int32_t res, ht_cnt = (height >> 2);
  __m128i src, ref, vec;
  __m128i avg = __lsx_vldi(0);
  __m128i var = avg;

  for (; ht_cnt--;) {
    src = __lsx_vld(src_ptr, 0);
    src_ptr += src_stride;
    ref = __lsx_vld(ref_ptr, 0);
    ref_ptr += ref_stride;
    CALC_MSE_AVG_B(src, ref, var, avg);

    src = __lsx_vld(src_ptr, 0);
    src_ptr += src_stride;
    ref = __lsx_vld(ref_ptr, 0);
    ref_ptr += ref_stride;
    CALC_MSE_AVG_B(src, ref, var, avg);
    src = __lsx_vld(src_ptr, 0);
    src_ptr += src_stride;
    ref = __lsx_vld(ref_ptr, 0);
    ref_ptr += ref_stride;
    CALC_MSE_AVG_B(src, ref, var, avg);

    src = __lsx_vld(src_ptr, 0);
    src_ptr += src_stride;
    ref = __lsx_vld(ref_ptr, 0);
    ref_ptr += ref_stride;
    CALC_MSE_AVG_B(src, ref, var, avg);
  }
  vec = __lsx_vhaddw_w_h(avg, avg);
  HADD_SW_S32(vec, *diff);
  HADD_SW_S32(var, res);
  return res;
}

static uint32_t sse_diff_32width_lsx(const uint8_t *src_ptr, int32_t src_stride,
                                     const uint8_t *ref_ptr, int32_t ref_stride,
                                     int32_t height, int32_t *diff) {
  int32_t res, ht_cnt = (height >> 2);
  __m128i avg = __lsx_vldi(0);
  __m128i src0, src1, ref0, ref1;
  __m128i vec;
  __m128i var = avg;

  for (; ht_cnt--;) {
    DUP2_ARG2(__lsx_vld, src_ptr, 0, src_ptr, 16, src0, src1);
    src_ptr += src_stride;
    DUP2_ARG2(__lsx_vld, ref_ptr, 0, ref_ptr, 16, ref0, ref1);
    ref_ptr += ref_stride;
    CALC_MSE_AVG_B(src0, ref0, var, avg);
    CALC_MSE_AVG_B(src1, ref1, var, avg);

    DUP2_ARG2(__lsx_vld, src_ptr, 0, src_ptr, 16, src0, src1);
    src_ptr += src_stride;
    DUP2_ARG2(__lsx_vld, ref_ptr, 0, ref_ptr, 16, ref0, ref1);
    ref_ptr += ref_stride;
    CALC_MSE_AVG_B(src0, ref0, var, avg);
    CALC_MSE_AVG_B(src1, ref1, var, avg);

    DUP2_ARG2(__lsx_vld, src_ptr, 0, src_ptr, 16, src0, src1);
    src_ptr += src_stride;
    DUP2_ARG2(__lsx_vld, ref_ptr, 0, ref_ptr, 16, ref0, ref1);
    ref_ptr += ref_stride;
    CALC_MSE_AVG_B(src0, ref0, var, avg);
    CALC_MSE_AVG_B(src1, ref1, var, avg);

    DUP2_ARG2(__lsx_vld, src_ptr, 0, src_ptr, 16, src0, src1);
    src_ptr += src_stride;
    DUP2_ARG2(__lsx_vld, ref_ptr, 0, ref_ptr, 16, ref0, ref1);
    ref_ptr += ref_stride;
    CALC_MSE_AVG_B(src0, ref0, var, avg);
    CALC_MSE_AVG_B(src1, ref1, var, avg);
  }

  vec = __lsx_vhaddw_w_h(avg, avg);
  HADD_SW_S32(vec, *diff);
  HADD_SW_S32(var, res);
  return res;
}

static uint32_t sse_diff_64x64_lsx(const uint8_t *src_ptr, int32_t src_stride,
                                   const uint8_t *ref_ptr, int32_t ref_stride,
                                   int32_t *diff) {
  int32_t res, ht_cnt = 32;
  __m128i avg0 = __lsx_vldi(0);
  __m128i src0, src1, src2, src3;
  __m128i ref0, ref1, ref2, ref3;
  __m128i vec0, vec1;
  __m128i avg1 = avg0;
  __m128i avg2 = avg0;
  __m128i avg3 = avg0;
  __m128i var = avg0;

  for (; ht_cnt--;) {
    DUP4_ARG2(__lsx_vld, src_ptr, 0, src_ptr, 16, src_ptr, 32, src_ptr, 48,
              src0, src1, src2, src3);
    src_ptr += src_stride;
    DUP4_ARG2(__lsx_vld, ref_ptr, 0, ref_ptr, 16, ref_ptr, 32, ref_ptr, 48,
              ref0, ref1, ref2, ref3);
    ref_ptr += ref_stride;

    CALC_MSE_AVG_B(src0, ref0, var, avg0);
    CALC_MSE_AVG_B(src1, ref1, var, avg1);
    CALC_MSE_AVG_B(src2, ref2, var, avg2);
    CALC_MSE_AVG_B(src3, ref3, var, avg3);
    DUP4_ARG2(__lsx_vld, src_ptr, 0, src_ptr, 16, src_ptr, 32, src_ptr, 48,
              src0, src1, src2, src3);
    src_ptr += src_stride;
    DUP4_ARG2(__lsx_vld, ref_ptr, 0, ref_ptr, 16, ref_ptr, 32, ref_ptr, 48,
              ref0, ref1, ref2, ref3);
    ref_ptr += ref_stride;
    CALC_MSE_AVG_B(src0, ref0, var, avg0);
    CALC_MSE_AVG_B(src1, ref1, var, avg1);
    CALC_MSE_AVG_B(src2, ref2, var, avg2);
    CALC_MSE_AVG_B(src3, ref3, var, avg3);
  }
  vec0 = __lsx_vhaddw_w_h(avg0, avg0);
  vec1 = __lsx_vhaddw_w_h(avg1, avg1);
  vec0 = __lsx_vadd_w(vec0, vec1);
  vec1 = __lsx_vhaddw_w_h(avg2, avg2);
  vec0 = __lsx_vadd_w(vec0, vec1);
  vec1 = __lsx_vhaddw_w_h(avg3, avg3);
  vec0 = __lsx_vadd_w(vec0, vec1);
  HADD_SW_S32(vec0, *diff);
  HADD_SW_S32(var, res);
  return res;
}

#define VARIANCE_8Wx8H(sse, diff) VARIANCE_WxH(sse, diff, 6)
#define VARIANCE_16Wx16H(sse, diff) VARIANCE_WxH(sse, diff, 8)

#define VARIANCE_32Wx32H(sse, diff) VARIANCE_LARGE_WxH(sse, diff, 10)
#define VARIANCE_64Wx64H(sse, diff) VARIANCE_LARGE_WxH(sse, diff, 12)

#define VPX_VARIANCE_WDXHT_LSX(wd, ht)                                         \
  uint32_t vpx_variance##wd##x##ht##_lsx(                                      \
      const uint8_t *src, int32_t src_stride, const uint8_t *ref,              \
      int32_t ref_stride, uint32_t *sse) {                                     \
    int32_t diff;                                                              \
                                                                               \
    *sse =                                                                     \
        sse_diff_##wd##width_lsx(src, src_stride, ref, ref_stride, ht, &diff); \
                                                                               \
    return VARIANCE_##wd##Wx##ht##H(*sse, diff);                               \
  }

static uint32_t sse_16width_lsx(const uint8_t *src_ptr, int32_t src_stride,
                                const uint8_t *ref_ptr, int32_t ref_stride,
                                int32_t height) {
  int32_t res, ht_cnt = (height >> 2);
  __m128i src, ref;
  __m128i var = __lsx_vldi(0);

  for (; ht_cnt--;) {
    DUP2_ARG2(__lsx_vld, src_ptr, 0, ref_ptr, 0, src, ref);
    src_ptr += src_stride;
    ref_ptr += ref_stride;
    CALC_MSE_B(src, ref, var);

    DUP2_ARG2(__lsx_vld, src_ptr, 0, ref_ptr, 0, src, ref);
    src_ptr += src_stride;
    ref_ptr += ref_stride;
    CALC_MSE_B(src, ref, var);

    DUP2_ARG2(__lsx_vld, src_ptr, 0, ref_ptr, 0, src, ref);
    src_ptr += src_stride;
    ref_ptr += ref_stride;
    CALC_MSE_B(src, ref, var);

    DUP2_ARG2(__lsx_vld, src_ptr, 0, ref_ptr, 0, src, ref);
    src_ptr += src_stride;
    ref_ptr += ref_stride;
    CALC_MSE_B(src, ref, var);
  }
  HADD_SW_S32(var, res);
  return res;
}

VPX_VARIANCE_WDXHT_LSX(8, 8)
VPX_VARIANCE_WDXHT_LSX(16, 16)
VPX_VARIANCE_WDXHT_LSX(32, 32)

uint32_t vpx_variance64x64_lsx(const uint8_t *src, int32_t src_stride,
                               const uint8_t *ref, int32_t ref_stride,
                               uint32_t *sse) {
  int32_t diff;

  *sse = sse_diff_64x64_lsx(src, src_stride, ref, ref_stride, &diff);

  return VARIANCE_64Wx64H(*sse, diff);
}

uint32_t vpx_mse16x16_lsx(const uint8_t *src, int32_t src_stride,
                          const uint8_t *ref, int32_t ref_stride,
                          uint32_t *sse) {
  *sse = sse_16width_lsx(src, src_stride, ref, ref_stride, 16);

  return *sse;
}

void vpx_get16x16var_lsx(const uint8_t *src, int32_t src_stride,
                         const uint8_t *ref, int32_t ref_stride, uint32_t *sse,
                         int32_t *sum) {
  *sse = sse_diff_16width_lsx(src, src_stride, ref, ref_stride, 16, sum);
}
