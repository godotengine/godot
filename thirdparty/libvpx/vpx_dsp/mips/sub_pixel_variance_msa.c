/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vpx_dsp_rtcd.h"
#include "vpx_ports/mem.h"
#include "vpx_dsp/mips/macros_msa.h"
#include "vpx_dsp/variance.h"

static const uint8_t bilinear_filters_msa[8][2] = {
  { 128, 0 }, { 112, 16 }, { 96, 32 }, { 80, 48 },
  { 64, 64 }, { 48, 80 },  { 32, 96 }, { 16, 112 },
};

#define CALC_MSE_AVG_B(src, ref, var, sub)                          \
  {                                                                 \
    v16u8 src_l0_m, src_l1_m;                                       \
    v8i16 res_l0_m, res_l1_m;                                       \
                                                                    \
    ILVRL_B2_UB(src, ref, src_l0_m, src_l1_m);                      \
    HSUB_UB2_SH(src_l0_m, src_l1_m, res_l0_m, res_l1_m);            \
    DPADD_SH2_SW(res_l0_m, res_l1_m, res_l0_m, res_l1_m, var, var); \
                                                                    \
    (sub) += res_l0_m + res_l1_m;                                   \
  }

#define VARIANCE_WxH(sse, diff, shift) \
  (sse) - (((uint32_t)(diff) * (diff)) >> (shift))

#define VARIANCE_LARGE_WxH(sse, diff, shift) \
  (sse) - (((int64_t)(diff) * (diff)) >> (shift))

static uint32_t avg_sse_diff_4width_msa(const uint8_t *src_ptr,
                                        int32_t src_stride,
                                        const uint8_t *ref_ptr,
                                        int32_t ref_stride,
                                        const uint8_t *sec_pred, int32_t height,
                                        int32_t *diff) {
  int32_t ht_cnt;
  uint32_t src0, src1, src2, src3;
  uint32_t ref0, ref1, ref2, ref3;
  v16u8 pred, src = { 0 };
  v16u8 ref = { 0 };
  v8i16 avg = { 0 };
  v4i32 vec, var = { 0 };

  for (ht_cnt = (height >> 2); ht_cnt--;) {
    pred = LD_UB(sec_pred);
    sec_pred += 16;
    LW4(src_ptr, src_stride, src0, src1, src2, src3);
    src_ptr += (4 * src_stride);
    LW4(ref_ptr, ref_stride, ref0, ref1, ref2, ref3);
    ref_ptr += (4 * ref_stride);

    INSERT_W4_UB(src0, src1, src2, src3, src);
    INSERT_W4_UB(ref0, ref1, ref2, ref3, ref);

    src = __msa_aver_u_b(src, pred);
    CALC_MSE_AVG_B(src, ref, var, avg);
  }

  vec = __msa_hadd_s_w(avg, avg);
  *diff = HADD_SW_S32(vec);

  return HADD_SW_S32(var);
}

static uint32_t avg_sse_diff_8width_msa(const uint8_t *src_ptr,
                                        int32_t src_stride,
                                        const uint8_t *ref_ptr,
                                        int32_t ref_stride,
                                        const uint8_t *sec_pred, int32_t height,
                                        int32_t *diff) {
  int32_t ht_cnt;
  v16u8 src0, src1, src2, src3;
  v16u8 ref0, ref1, ref2, ref3;
  v16u8 pred0, pred1;
  v8i16 avg = { 0 };
  v4i32 vec, var = { 0 };

  for (ht_cnt = (height >> 2); ht_cnt--;) {
    LD_UB2(sec_pred, 16, pred0, pred1);
    sec_pred += 32;
    LD_UB4(src_ptr, src_stride, src0, src1, src2, src3);
    src_ptr += (4 * src_stride);
    LD_UB4(ref_ptr, ref_stride, ref0, ref1, ref2, ref3);
    ref_ptr += (4 * ref_stride);

    PCKEV_D4_UB(src1, src0, src3, src2, ref1, ref0, ref3, ref2, src0, src1,
                ref0, ref1);
    AVER_UB2_UB(src0, pred0, src1, pred1, src0, src1);
    CALC_MSE_AVG_B(src0, ref0, var, avg);
    CALC_MSE_AVG_B(src1, ref1, var, avg);
  }

  vec = __msa_hadd_s_w(avg, avg);
  *diff = HADD_SW_S32(vec);

  return HADD_SW_S32(var);
}

static uint32_t avg_sse_diff_16width_msa(const uint8_t *src_ptr,
                                         int32_t src_stride,
                                         const uint8_t *ref_ptr,
                                         int32_t ref_stride,
                                         const uint8_t *sec_pred,
                                         int32_t height, int32_t *diff) {
  int32_t ht_cnt;
  v16u8 src, ref, pred;
  v8i16 avg = { 0 };
  v4i32 vec, var = { 0 };

  for (ht_cnt = (height >> 2); ht_cnt--;) {
    pred = LD_UB(sec_pred);
    sec_pred += 16;
    src = LD_UB(src_ptr);
    src_ptr += src_stride;
    ref = LD_UB(ref_ptr);
    ref_ptr += ref_stride;
    src = __msa_aver_u_b(src, pred);
    CALC_MSE_AVG_B(src, ref, var, avg);

    pred = LD_UB(sec_pred);
    sec_pred += 16;
    src = LD_UB(src_ptr);
    src_ptr += src_stride;
    ref = LD_UB(ref_ptr);
    ref_ptr += ref_stride;
    src = __msa_aver_u_b(src, pred);
    CALC_MSE_AVG_B(src, ref, var, avg);

    pred = LD_UB(sec_pred);
    sec_pred += 16;
    src = LD_UB(src_ptr);
    src_ptr += src_stride;
    ref = LD_UB(ref_ptr);
    ref_ptr += ref_stride;
    src = __msa_aver_u_b(src, pred);
    CALC_MSE_AVG_B(src, ref, var, avg);

    pred = LD_UB(sec_pred);
    sec_pred += 16;
    src = LD_UB(src_ptr);
    src_ptr += src_stride;
    ref = LD_UB(ref_ptr);
    ref_ptr += ref_stride;
    src = __msa_aver_u_b(src, pred);
    CALC_MSE_AVG_B(src, ref, var, avg);
  }

  vec = __msa_hadd_s_w(avg, avg);
  *diff = HADD_SW_S32(vec);

  return HADD_SW_S32(var);
}

static uint32_t avg_sse_diff_32width_msa(const uint8_t *src_ptr,
                                         int32_t src_stride,
                                         const uint8_t *ref_ptr,
                                         int32_t ref_stride,
                                         const uint8_t *sec_pred,
                                         int32_t height, int32_t *diff) {
  int32_t ht_cnt;
  v16u8 src0, src1, ref0, ref1, pred0, pred1;
  v8i16 avg = { 0 };
  v4i32 vec, var = { 0 };

  for (ht_cnt = (height >> 2); ht_cnt--;) {
    LD_UB2(sec_pred, 16, pred0, pred1);
    sec_pred += 32;
    LD_UB2(src_ptr, 16, src0, src1);
    src_ptr += src_stride;
    LD_UB2(ref_ptr, 16, ref0, ref1);
    ref_ptr += ref_stride;
    AVER_UB2_UB(src0, pred0, src1, pred1, src0, src1);
    CALC_MSE_AVG_B(src0, ref0, var, avg);
    CALC_MSE_AVG_B(src1, ref1, var, avg);

    LD_UB2(sec_pred, 16, pred0, pred1);
    sec_pred += 32;
    LD_UB2(src_ptr, 16, src0, src1);
    src_ptr += src_stride;
    LD_UB2(ref_ptr, 16, ref0, ref1);
    ref_ptr += ref_stride;
    AVER_UB2_UB(src0, pred0, src1, pred1, src0, src1);
    CALC_MSE_AVG_B(src0, ref0, var, avg);
    CALC_MSE_AVG_B(src1, ref1, var, avg);

    LD_UB2(sec_pred, 16, pred0, pred1);
    sec_pred += 32;
    LD_UB2(src_ptr, 16, src0, src1);
    src_ptr += src_stride;
    LD_UB2(ref_ptr, 16, ref0, ref1);
    ref_ptr += ref_stride;
    AVER_UB2_UB(src0, pred0, src1, pred1, src0, src1);
    CALC_MSE_AVG_B(src0, ref0, var, avg);
    CALC_MSE_AVG_B(src1, ref1, var, avg);

    LD_UB2(sec_pred, 16, pred0, pred1);
    sec_pred += 32;
    LD_UB2(src_ptr, 16, src0, src1);
    src_ptr += src_stride;
    LD_UB2(ref_ptr, 16, ref0, ref1);
    ref_ptr += ref_stride;
    AVER_UB2_UB(src0, pred0, src1, pred1, src0, src1);
    CALC_MSE_AVG_B(src0, ref0, var, avg);
    CALC_MSE_AVG_B(src1, ref1, var, avg);
  }

  vec = __msa_hadd_s_w(avg, avg);
  *diff = HADD_SW_S32(vec);

  return HADD_SW_S32(var);
}

static uint32_t avg_sse_diff_32x64_msa(const uint8_t *src_ptr,
                                       int32_t src_stride,
                                       const uint8_t *ref_ptr,
                                       int32_t ref_stride,
                                       const uint8_t *sec_pred, int32_t *diff) {
  int32_t ht_cnt;
  v16u8 src0, src1, ref0, ref1, pred0, pred1;
  v8i16 avg0 = { 0 };
  v8i16 avg1 = { 0 };
  v4i32 vec, var = { 0 };

  for (ht_cnt = 16; ht_cnt--;) {
    LD_UB2(sec_pred, 16, pred0, pred1);
    sec_pred += 32;
    LD_UB2(src_ptr, 16, src0, src1);
    src_ptr += src_stride;
    LD_UB2(ref_ptr, 16, ref0, ref1);
    ref_ptr += ref_stride;
    AVER_UB2_UB(src0, pred0, src1, pred1, src0, src1);
    CALC_MSE_AVG_B(src0, ref0, var, avg0);
    CALC_MSE_AVG_B(src1, ref1, var, avg1);

    LD_UB2(sec_pred, 16, pred0, pred1);
    sec_pred += 32;
    LD_UB2(src_ptr, 16, src0, src1);
    src_ptr += src_stride;
    LD_UB2(ref_ptr, 16, ref0, ref1);
    ref_ptr += ref_stride;
    AVER_UB2_UB(src0, pred0, src1, pred1, src0, src1);
    CALC_MSE_AVG_B(src0, ref0, var, avg0);
    CALC_MSE_AVG_B(src1, ref1, var, avg1);

    LD_UB2(sec_pred, 16, pred0, pred1);
    sec_pred += 32;
    LD_UB2(src_ptr, 16, src0, src1);
    src_ptr += src_stride;
    LD_UB2(ref_ptr, 16, ref0, ref1);
    ref_ptr += ref_stride;
    AVER_UB2_UB(src0, pred0, src1, pred1, src0, src1);
    CALC_MSE_AVG_B(src0, ref0, var, avg0);
    CALC_MSE_AVG_B(src1, ref1, var, avg1);

    LD_UB2(sec_pred, 16, pred0, pred1);
    sec_pred += 32;
    LD_UB2(src_ptr, 16, src0, src1);
    src_ptr += src_stride;
    LD_UB2(ref_ptr, 16, ref0, ref1);
    ref_ptr += ref_stride;
    AVER_UB2_UB(src0, pred0, src1, pred1, src0, src1);
    CALC_MSE_AVG_B(src0, ref0, var, avg0);
    CALC_MSE_AVG_B(src1, ref1, var, avg1);
  }

  vec = __msa_hadd_s_w(avg0, avg0);
  vec += __msa_hadd_s_w(avg1, avg1);
  *diff = HADD_SW_S32(vec);

  return HADD_SW_S32(var);
}

static uint32_t avg_sse_diff_64x32_msa(const uint8_t *src_ptr,
                                       int32_t src_stride,
                                       const uint8_t *ref_ptr,
                                       int32_t ref_stride,
                                       const uint8_t *sec_pred, int32_t *diff) {
  int32_t ht_cnt;
  v16u8 src0, src1, src2, src3;
  v16u8 ref0, ref1, ref2, ref3;
  v16u8 pred0, pred1, pred2, pred3;
  v8i16 avg0 = { 0 };
  v8i16 avg1 = { 0 };
  v4i32 vec, var = { 0 };

  for (ht_cnt = 16; ht_cnt--;) {
    LD_UB4(sec_pred, 16, pred0, pred1, pred2, pred3);
    sec_pred += 64;
    LD_UB4(src_ptr, 16, src0, src1, src2, src3);
    src_ptr += src_stride;
    LD_UB4(ref_ptr, 16, ref0, ref1, ref2, ref3);
    ref_ptr += ref_stride;
    AVER_UB4_UB(src0, pred0, src1, pred1, src2, pred2, src3, pred3, src0, src1,
                src2, src3);
    CALC_MSE_AVG_B(src0, ref0, var, avg0);
    CALC_MSE_AVG_B(src2, ref2, var, avg0);
    CALC_MSE_AVG_B(src1, ref1, var, avg1);
    CALC_MSE_AVG_B(src3, ref3, var, avg1);

    LD_UB4(sec_pred, 16, pred0, pred1, pred2, pred3);
    sec_pred += 64;
    LD_UB4(src_ptr, 16, src0, src1, src2, src3);
    src_ptr += src_stride;
    LD_UB4(ref_ptr, 16, ref0, ref1, ref2, ref3);
    ref_ptr += ref_stride;
    AVER_UB4_UB(src0, pred0, src1, pred1, src2, pred2, src3, pred3, src0, src1,
                src2, src3);
    CALC_MSE_AVG_B(src0, ref0, var, avg0);
    CALC_MSE_AVG_B(src2, ref2, var, avg0);
    CALC_MSE_AVG_B(src1, ref1, var, avg1);
    CALC_MSE_AVG_B(src3, ref3, var, avg1);
  }

  vec = __msa_hadd_s_w(avg0, avg0);
  vec += __msa_hadd_s_w(avg1, avg1);

  *diff = HADD_SW_S32(vec);

  return HADD_SW_S32(var);
}

static uint32_t avg_sse_diff_64x64_msa(const uint8_t *src_ptr,
                                       int32_t src_stride,
                                       const uint8_t *ref_ptr,
                                       int32_t ref_stride,
                                       const uint8_t *sec_pred, int32_t *diff) {
  int32_t ht_cnt;
  v16u8 src0, src1, src2, src3;
  v16u8 ref0, ref1, ref2, ref3;
  v16u8 pred0, pred1, pred2, pred3;
  v8i16 avg0 = { 0 };
  v8i16 avg1 = { 0 };
  v8i16 avg2 = { 0 };
  v8i16 avg3 = { 0 };
  v4i32 vec, var = { 0 };

  for (ht_cnt = 32; ht_cnt--;) {
    LD_UB4(sec_pred, 16, pred0, pred1, pred2, pred3);
    sec_pred += 64;
    LD_UB4(src_ptr, 16, src0, src1, src2, src3);
    src_ptr += src_stride;
    LD_UB4(ref_ptr, 16, ref0, ref1, ref2, ref3);
    ref_ptr += ref_stride;
    AVER_UB4_UB(src0, pred0, src1, pred1, src2, pred2, src3, pred3, src0, src1,
                src2, src3);
    CALC_MSE_AVG_B(src0, ref0, var, avg0);
    CALC_MSE_AVG_B(src1, ref1, var, avg1);
    CALC_MSE_AVG_B(src2, ref2, var, avg2);
    CALC_MSE_AVG_B(src3, ref3, var, avg3);

    LD_UB4(sec_pred, 16, pred0, pred1, pred2, pred3);
    sec_pred += 64;
    LD_UB4(src_ptr, 16, src0, src1, src2, src3);
    src_ptr += src_stride;
    LD_UB4(ref_ptr, 16, ref0, ref1, ref2, ref3);
    ref_ptr += ref_stride;
    AVER_UB4_UB(src0, pred0, src1, pred1, src2, pred2, src3, pred3, src0, src1,
                src2, src3);
    CALC_MSE_AVG_B(src0, ref0, var, avg0);
    CALC_MSE_AVG_B(src1, ref1, var, avg1);
    CALC_MSE_AVG_B(src2, ref2, var, avg2);
    CALC_MSE_AVG_B(src3, ref3, var, avg3);
  }

  vec = __msa_hadd_s_w(avg0, avg0);
  vec += __msa_hadd_s_w(avg1, avg1);
  vec += __msa_hadd_s_w(avg2, avg2);
  vec += __msa_hadd_s_w(avg3, avg3);
  *diff = HADD_SW_S32(vec);

  return HADD_SW_S32(var);
}

static uint32_t sub_pixel_sse_diff_4width_h_msa(
    const uint8_t *src, int32_t src_stride, const uint8_t *dst,
    int32_t dst_stride, const uint8_t *filter, int32_t height, int32_t *diff) {
  int16_t filtval;
  uint32_t loop_cnt;
  uint32_t ref0, ref1, ref2, ref3;
  v16u8 filt0, ref = { 0 };
  v16i8 src0, src1, src2, src3;
  v16i8 mask = { 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8 };
  v8u16 vec0, vec1, vec2, vec3;
  v8i16 avg = { 0 };
  v4i32 vec, var = { 0 };

  filtval = LH(filter);
  filt0 = (v16u8)__msa_fill_h(filtval);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_SB4(src, src_stride, src0, src1, src2, src3);
    src += (4 * src_stride);
    LW4(dst, dst_stride, ref0, ref1, ref2, ref3);
    dst += (4 * dst_stride);
    INSERT_W4_UB(ref0, ref1, ref2, ref3, ref);
    VSHF_B2_UH(src0, src0, src1, src1, mask, mask, vec0, vec1);
    VSHF_B2_UH(src2, src2, src3, src3, mask, mask, vec2, vec3);
    DOTP_UB4_UH(vec0, vec1, vec2, vec3, filt0, filt0, filt0, filt0, vec0, vec1,
                vec2, vec3);
    SRARI_H4_UH(vec0, vec1, vec2, vec3, FILTER_BITS);
    PCKEV_B4_SB(vec0, vec0, vec1, vec1, vec2, vec2, vec3, vec3, src0, src1,
                src2, src3);
    ILVEV_W2_SB(src0, src1, src2, src3, src0, src2);
    src0 = (v16i8)__msa_ilvev_d((v2i64)src2, (v2i64)src0);
    CALC_MSE_AVG_B(src0, ref, var, avg);
  }

  vec = __msa_hadd_s_w(avg, avg);
  *diff = HADD_SW_S32(vec);

  return HADD_SW_S32(var);
}

static uint32_t sub_pixel_sse_diff_8width_h_msa(
    const uint8_t *src, int32_t src_stride, const uint8_t *dst,
    int32_t dst_stride, const uint8_t *filter, int32_t height, int32_t *diff) {
  int16_t filtval;
  uint32_t loop_cnt;
  v16u8 filt0, out, ref0, ref1, ref2, ref3;
  v16i8 src0, src1, src2, src3;
  v16i8 mask = { 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8 };
  v8u16 vec0, vec1, vec2, vec3;
  v8i16 avg = { 0 };
  v4i32 vec, var = { 0 };

  filtval = LH(filter);
  filt0 = (v16u8)__msa_fill_h(filtval);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_SB4(src, src_stride, src0, src1, src2, src3);
    src += (4 * src_stride);
    LD_UB4(dst, dst_stride, ref0, ref1, ref2, ref3);
    dst += (4 * dst_stride);

    PCKEV_D2_UB(ref1, ref0, ref3, ref2, ref0, ref1);
    VSHF_B2_UH(src0, src0, src1, src1, mask, mask, vec0, vec1);
    VSHF_B2_UH(src2, src2, src3, src3, mask, mask, vec2, vec3);
    DOTP_UB4_UH(vec0, vec1, vec2, vec3, filt0, filt0, filt0, filt0, vec0, vec1,
                vec2, vec3);
    SRARI_H4_UH(vec0, vec1, vec2, vec3, FILTER_BITS);
    PCKEV_B4_SB(vec0, vec0, vec1, vec1, vec2, vec2, vec3, vec3, src0, src1,
                src2, src3);
    out = (v16u8)__msa_ilvev_d((v2i64)src1, (v2i64)src0);
    CALC_MSE_AVG_B(out, ref0, var, avg);
    out = (v16u8)__msa_ilvev_d((v2i64)src3, (v2i64)src2);
    CALC_MSE_AVG_B(out, ref1, var, avg);
  }

  vec = __msa_hadd_s_w(avg, avg);
  *diff = HADD_SW_S32(vec);

  return HADD_SW_S32(var);
}

static uint32_t sub_pixel_sse_diff_16width_h_msa(
    const uint8_t *src, int32_t src_stride, const uint8_t *dst,
    int32_t dst_stride, const uint8_t *filter, int32_t height, int32_t *diff) {
  int16_t filtval;
  uint32_t loop_cnt;
  v16i8 src0, src1, src2, src3, src4, src5, src6, src7;
  v16i8 mask = { 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8 };
  v16u8 dst0, dst1, dst2, dst3, filt0;
  v8u16 vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7;
  v8u16 out0, out1, out2, out3, out4, out5, out6, out7;
  v8i16 avg = { 0 };
  v4i32 vec, var = { 0 };

  filtval = LH(filter);
  filt0 = (v16u8)__msa_fill_h(filtval);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_SB4(src, src_stride, src0, src2, src4, src6);
    LD_SB4(src + 8, src_stride, src1, src3, src5, src7);
    src += (4 * src_stride);
    LD_UB4(dst, dst_stride, dst0, dst1, dst2, dst3);
    dst += (4 * dst_stride);

    VSHF_B2_UH(src0, src0, src1, src1, mask, mask, vec0, vec1);
    VSHF_B2_UH(src2, src2, src3, src3, mask, mask, vec2, vec3);
    VSHF_B2_UH(src4, src4, src5, src5, mask, mask, vec4, vec5);
    VSHF_B2_UH(src6, src6, src7, src7, mask, mask, vec6, vec7);
    DOTP_UB4_UH(vec0, vec1, vec2, vec3, filt0, filt0, filt0, filt0, out0, out1,
                out2, out3);
    DOTP_UB4_UH(vec4, vec5, vec6, vec7, filt0, filt0, filt0, filt0, out4, out5,
                out6, out7);
    SRARI_H4_UH(out0, out1, out2, out3, FILTER_BITS);
    SRARI_H4_UH(out4, out5, out6, out7, FILTER_BITS);
    PCKEV_B4_SB(out1, out0, out3, out2, out5, out4, out7, out6, src0, src1,
                src2, src3);
    CALC_MSE_AVG_B(src0, dst0, var, avg);
    CALC_MSE_AVG_B(src1, dst1, var, avg);
    CALC_MSE_AVG_B(src2, dst2, var, avg);
    CALC_MSE_AVG_B(src3, dst3, var, avg);
  }

  vec = __msa_hadd_s_w(avg, avg);
  *diff = HADD_SW_S32(vec);

  return HADD_SW_S32(var);
}

static uint32_t sub_pixel_sse_diff_32width_h_msa(
    const uint8_t *src, int32_t src_stride, const uint8_t *dst,
    int32_t dst_stride, const uint8_t *filter, int32_t height, int32_t *diff) {
  uint32_t loop_cnt, sse = 0;
  int32_t diff0[2];

  for (loop_cnt = 0; loop_cnt < 2; ++loop_cnt) {
    sse += sub_pixel_sse_diff_16width_h_msa(src, src_stride, dst, dst_stride,
                                            filter, height, &diff0[loop_cnt]);
    src += 16;
    dst += 16;
  }

  *diff = diff0[0] + diff0[1];

  return sse;
}

static uint32_t sub_pixel_sse_diff_64width_h_msa(
    const uint8_t *src, int32_t src_stride, const uint8_t *dst,
    int32_t dst_stride, const uint8_t *filter, int32_t height, int32_t *diff) {
  uint32_t loop_cnt, sse = 0;
  int32_t diff0[4];

  for (loop_cnt = 0; loop_cnt < 4; ++loop_cnt) {
    sse += sub_pixel_sse_diff_16width_h_msa(src, src_stride, dst, dst_stride,
                                            filter, height, &diff0[loop_cnt]);
    src += 16;
    dst += 16;
  }

  *diff = diff0[0] + diff0[1] + diff0[2] + diff0[3];

  return sse;
}

static uint32_t sub_pixel_sse_diff_4width_v_msa(
    const uint8_t *src, int32_t src_stride, const uint8_t *dst,
    int32_t dst_stride, const uint8_t *filter, int32_t height, int32_t *diff) {
  int16_t filtval;
  uint32_t loop_cnt;
  uint32_t ref0, ref1, ref2, ref3;
  v16u8 src0, src1, src2, src3, src4, out;
  v16u8 src10_r, src32_r, src21_r, src43_r;
  v16u8 ref = { 0 };
  v16u8 src2110, src4332;
  v16u8 filt0;
  v8i16 avg = { 0 };
  v4i32 vec, var = { 0 };
  v8u16 tmp0, tmp1;

  filtval = LH(filter);
  filt0 = (v16u8)__msa_fill_h(filtval);

  src0 = LD_UB(src);
  src += src_stride;

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_UB4(src, src_stride, src1, src2, src3, src4);
    src += (4 * src_stride);
    LW4(dst, dst_stride, ref0, ref1, ref2, ref3);
    dst += (4 * dst_stride);

    INSERT_W4_UB(ref0, ref1, ref2, ref3, ref);
    ILVR_B4_UB(src1, src0, src2, src1, src3, src2, src4, src3, src10_r, src21_r,
               src32_r, src43_r);
    ILVR_D2_UB(src21_r, src10_r, src43_r, src32_r, src2110, src4332);
    DOTP_UB2_UH(src2110, src4332, filt0, filt0, tmp0, tmp1);
    SRARI_H2_UH(tmp0, tmp1, FILTER_BITS);
    out = (v16u8)__msa_pckev_b((v16i8)tmp1, (v16i8)tmp0);
    CALC_MSE_AVG_B(out, ref, var, avg);
    src0 = src4;
  }

  vec = __msa_hadd_s_w(avg, avg);
  *diff = HADD_SW_S32(vec);

  return HADD_SW_S32(var);
}

static uint32_t sub_pixel_sse_diff_8width_v_msa(
    const uint8_t *src, int32_t src_stride, const uint8_t *dst,
    int32_t dst_stride, const uint8_t *filter, int32_t height, int32_t *diff) {
  int16_t filtval;
  uint32_t loop_cnt;
  v16u8 src0, src1, src2, src3, src4;
  v16u8 ref0, ref1, ref2, ref3;
  v8u16 vec0, vec1, vec2, vec3;
  v8u16 tmp0, tmp1, tmp2, tmp3;
  v16u8 filt0;
  v8i16 avg = { 0 };
  v4i32 vec, var = { 0 };

  filtval = LH(filter);
  filt0 = (v16u8)__msa_fill_h(filtval);

  src0 = LD_UB(src);
  src += src_stride;

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_UB4(src, src_stride, src1, src2, src3, src4);
    src += (4 * src_stride);
    LD_UB4(dst, dst_stride, ref0, ref1, ref2, ref3);
    dst += (4 * dst_stride);

    PCKEV_D2_UB(ref1, ref0, ref3, ref2, ref0, ref1);
    ILVR_B4_UH(src1, src0, src2, src1, src3, src2, src4, src3, vec0, vec1, vec2,
               vec3);
    DOTP_UB4_UH(vec0, vec1, vec2, vec3, filt0, filt0, filt0, filt0, tmp0, tmp1,
                tmp2, tmp3);
    SRARI_H4_UH(tmp0, tmp1, tmp2, tmp3, FILTER_BITS);
    PCKEV_B2_UB(tmp1, tmp0, tmp3, tmp2, src0, src1);
    CALC_MSE_AVG_B(src0, ref0, var, avg);
    CALC_MSE_AVG_B(src1, ref1, var, avg);
    src0 = src4;
  }

  vec = __msa_hadd_s_w(avg, avg);
  *diff = HADD_SW_S32(vec);

  return HADD_SW_S32(var);
}

static uint32_t sub_pixel_sse_diff_16width_v_msa(
    const uint8_t *src, int32_t src_stride, const uint8_t *dst,
    int32_t dst_stride, const uint8_t *filter, int32_t height, int32_t *diff) {
  int16_t filtval;
  uint32_t loop_cnt;
  v16u8 ref0, ref1, ref2, ref3;
  v16u8 src0, src1, src2, src3, src4;
  v16u8 out0, out1, out2, out3;
  v16u8 vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7;
  v8u16 tmp0, tmp1, tmp2, tmp3;
  v16u8 filt0;
  v8i16 avg = { 0 };
  v4i32 vec, var = { 0 };

  filtval = LH(filter);
  filt0 = (v16u8)__msa_fill_h(filtval);

  src0 = LD_UB(src);
  src += src_stride;

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_UB4(src, src_stride, src1, src2, src3, src4);
    src += (4 * src_stride);
    LD_UB4(dst, dst_stride, ref0, ref1, ref2, ref3);
    dst += (4 * dst_stride);

    ILVR_B2_UB(src1, src0, src2, src1, vec0, vec2);
    ILVL_B2_UB(src1, src0, src2, src1, vec1, vec3);
    DOTP_UB2_UH(vec0, vec1, filt0, filt0, tmp0, tmp1);
    SRARI_H2_UH(tmp0, tmp1, FILTER_BITS);
    out0 = (v16u8)__msa_pckev_b((v16i8)tmp1, (v16i8)tmp0);

    ILVR_B2_UB(src3, src2, src4, src3, vec4, vec6);
    ILVL_B2_UB(src3, src2, src4, src3, vec5, vec7);
    DOTP_UB2_UH(vec2, vec3, filt0, filt0, tmp2, tmp3);
    SRARI_H2_UH(tmp2, tmp3, FILTER_BITS);
    out1 = (v16u8)__msa_pckev_b((v16i8)tmp3, (v16i8)tmp2);

    DOTP_UB2_UH(vec4, vec5, filt0, filt0, tmp0, tmp1);
    SRARI_H2_UH(tmp0, tmp1, FILTER_BITS);
    out2 = (v16u8)__msa_pckev_b((v16i8)tmp1, (v16i8)tmp0);
    DOTP_UB2_UH(vec6, vec7, filt0, filt0, tmp2, tmp3);
    SRARI_H2_UH(tmp2, tmp3, FILTER_BITS);
    out3 = (v16u8)__msa_pckev_b((v16i8)tmp3, (v16i8)tmp2);

    src0 = src4;

    CALC_MSE_AVG_B(out0, ref0, var, avg);
    CALC_MSE_AVG_B(out1, ref1, var, avg);
    CALC_MSE_AVG_B(out2, ref2, var, avg);
    CALC_MSE_AVG_B(out3, ref3, var, avg);
  }

  vec = __msa_hadd_s_w(avg, avg);
  *diff = HADD_SW_S32(vec);

  return HADD_SW_S32(var);
}

static uint32_t sub_pixel_sse_diff_32width_v_msa(
    const uint8_t *src, int32_t src_stride, const uint8_t *dst,
    int32_t dst_stride, const uint8_t *filter, int32_t height, int32_t *diff) {
  uint32_t loop_cnt, sse = 0;
  int32_t diff0[2];

  for (loop_cnt = 0; loop_cnt < 2; ++loop_cnt) {
    sse += sub_pixel_sse_diff_16width_v_msa(src, src_stride, dst, dst_stride,
                                            filter, height, &diff0[loop_cnt]);
    src += 16;
    dst += 16;
  }

  *diff = diff0[0] + diff0[1];

  return sse;
}

static uint32_t sub_pixel_sse_diff_64width_v_msa(
    const uint8_t *src, int32_t src_stride, const uint8_t *dst,
    int32_t dst_stride, const uint8_t *filter, int32_t height, int32_t *diff) {
  uint32_t loop_cnt, sse = 0;
  int32_t diff0[4];

  for (loop_cnt = 0; loop_cnt < 4; ++loop_cnt) {
    sse += sub_pixel_sse_diff_16width_v_msa(src, src_stride, dst, dst_stride,
                                            filter, height, &diff0[loop_cnt]);
    src += 16;
    dst += 16;
  }

  *diff = diff0[0] + diff0[1] + diff0[2] + diff0[3];

  return sse;
}

static uint32_t sub_pixel_sse_diff_4width_hv_msa(
    const uint8_t *src, int32_t src_stride, const uint8_t *dst,
    int32_t dst_stride, const uint8_t *filter_horiz, const uint8_t *filter_vert,
    int32_t height, int32_t *diff) {
  int16_t filtval;
  uint32_t loop_cnt;
  uint32_t ref0, ref1, ref2, ref3;
  v16u8 src0, src1, src2, src3, src4;
  v16u8 out, ref = { 0 };
  v16u8 filt_vt, filt_hz, vec0, vec1;
  v16u8 mask = { 0, 1, 1, 2, 2, 3, 3, 4, 16, 17, 17, 18, 18, 19, 19, 20 };
  v8u16 hz_out0, hz_out1, hz_out2, hz_out3, hz_out4;
  v8u16 tmp0, tmp1;
  v8i16 avg = { 0 };
  v4i32 vec, var = { 0 };

  filtval = LH(filter_horiz);
  filt_hz = (v16u8)__msa_fill_h(filtval);
  filtval = LH(filter_vert);
  filt_vt = (v16u8)__msa_fill_h(filtval);

  src0 = LD_UB(src);
  src += src_stride;

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_UB4(src, src_stride, src1, src2, src3, src4);
    src += (4 * src_stride);
    LW4(dst, dst_stride, ref0, ref1, ref2, ref3);
    dst += (4 * dst_stride);
    INSERT_W4_UB(ref0, ref1, ref2, ref3, ref);
    hz_out0 = HORIZ_2TAP_FILT_UH(src0, src1, mask, filt_hz, FILTER_BITS);
    hz_out2 = HORIZ_2TAP_FILT_UH(src2, src3, mask, filt_hz, FILTER_BITS);
    hz_out4 = HORIZ_2TAP_FILT_UH(src4, src4, mask, filt_hz, FILTER_BITS);
    hz_out1 = (v8u16)__msa_sldi_b((v16i8)hz_out2, (v16i8)hz_out0, 8);
    hz_out3 = (v8u16)__msa_pckod_d((v2i64)hz_out4, (v2i64)hz_out2);
    ILVEV_B2_UB(hz_out0, hz_out1, hz_out2, hz_out3, vec0, vec1);
    DOTP_UB2_UH(vec0, vec1, filt_vt, filt_vt, tmp0, tmp1);
    SRARI_H2_UH(tmp0, tmp1, FILTER_BITS);
    out = (v16u8)__msa_pckev_b((v16i8)tmp1, (v16i8)tmp0);
    CALC_MSE_AVG_B(out, ref, var, avg);
    src0 = src4;
  }

  vec = __msa_hadd_s_w(avg, avg);
  *diff = HADD_SW_S32(vec);

  return HADD_SW_S32(var);
}

static uint32_t sub_pixel_sse_diff_8width_hv_msa(
    const uint8_t *src, int32_t src_stride, const uint8_t *dst,
    int32_t dst_stride, const uint8_t *filter_horiz, const uint8_t *filter_vert,
    int32_t height, int32_t *diff) {
  int16_t filtval;
  uint32_t loop_cnt;
  v16u8 ref0, ref1, ref2, ref3;
  v16u8 src0, src1, src2, src3, src4;
  v16u8 out0, out1;
  v16u8 mask = { 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8 };
  v8u16 hz_out0, hz_out1;
  v8u16 tmp0, tmp1, tmp2, tmp3;
  v16u8 filt_vt, filt_hz, vec0;
  v8i16 avg = { 0 };
  v4i32 vec, var = { 0 };

  filtval = LH(filter_horiz);
  filt_hz = (v16u8)__msa_fill_h(filtval);
  filtval = LH(filter_vert);
  filt_vt = (v16u8)__msa_fill_h(filtval);

  src0 = LD_UB(src);
  src += src_stride;
  hz_out0 = HORIZ_2TAP_FILT_UH(src0, src0, mask, filt_hz, FILTER_BITS);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_UB4(src, src_stride, src1, src2, src3, src4);
    src += (4 * src_stride);
    LD_UB4(dst, dst_stride, ref0, ref1, ref2, ref3);
    dst += (4 * dst_stride);

    PCKEV_D2_UB(ref1, ref0, ref3, ref2, ref0, ref1);
    hz_out1 = HORIZ_2TAP_FILT_UH(src1, src1, mask, filt_hz, FILTER_BITS);
    vec0 = (v16u8)__msa_ilvev_b((v16i8)hz_out1, (v16i8)hz_out0);
    tmp0 = __msa_dotp_u_h(vec0, filt_vt);
    hz_out0 = HORIZ_2TAP_FILT_UH(src2, src2, mask, filt_hz, FILTER_BITS);
    vec0 = (v16u8)__msa_ilvev_b((v16i8)hz_out0, (v16i8)hz_out1);
    tmp1 = __msa_dotp_u_h(vec0, filt_vt);
    SRARI_H2_UH(tmp0, tmp1, FILTER_BITS);
    hz_out1 = HORIZ_2TAP_FILT_UH(src3, src3, mask, filt_hz, FILTER_BITS);
    vec0 = (v16u8)__msa_ilvev_b((v16i8)hz_out1, (v16i8)hz_out0);
    tmp2 = __msa_dotp_u_h(vec0, filt_vt);
    hz_out0 = HORIZ_2TAP_FILT_UH(src4, src4, mask, filt_hz, FILTER_BITS);
    vec0 = (v16u8)__msa_ilvev_b((v16i8)hz_out0, (v16i8)hz_out1);
    tmp3 = __msa_dotp_u_h(vec0, filt_vt);
    SRARI_H2_UH(tmp2, tmp3, FILTER_BITS);
    PCKEV_B2_UB(tmp1, tmp0, tmp3, tmp2, out0, out1);
    CALC_MSE_AVG_B(out0, ref0, var, avg);
    CALC_MSE_AVG_B(out1, ref1, var, avg);
  }

  vec = __msa_hadd_s_w(avg, avg);
  *diff = HADD_SW_S32(vec);

  return HADD_SW_S32(var);
}

static uint32_t sub_pixel_sse_diff_16width_hv_msa(
    const uint8_t *src, int32_t src_stride, const uint8_t *dst,
    int32_t dst_stride, const uint8_t *filter_horiz, const uint8_t *filter_vert,
    int32_t height, int32_t *diff) {
  int16_t filtval;
  uint32_t loop_cnt;
  v16u8 src0, src1, src2, src3, src4, src5, src6, src7;
  v16u8 ref0, ref1, ref2, ref3;
  v16u8 filt_hz, filt_vt, vec0, vec1;
  v16u8 mask = { 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8 };
  v8u16 hz_out0, hz_out1, hz_out2, hz_out3;
  v8u16 tmp0, tmp1;
  v8i16 avg = { 0 };
  v4i32 vec, var = { 0 };

  filtval = LH(filter_horiz);
  filt_hz = (v16u8)__msa_fill_h(filtval);
  filtval = LH(filter_vert);
  filt_vt = (v16u8)__msa_fill_h(filtval);

  LD_UB2(src, 8, src0, src1);
  src += src_stride;

  hz_out0 = HORIZ_2TAP_FILT_UH(src0, src0, mask, filt_hz, FILTER_BITS);
  hz_out2 = HORIZ_2TAP_FILT_UH(src1, src1, mask, filt_hz, FILTER_BITS);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_UB4(src, src_stride, src0, src2, src4, src6);
    LD_UB4(src + 8, src_stride, src1, src3, src5, src7);
    src += (4 * src_stride);
    LD_UB4(dst, dst_stride, ref0, ref1, ref2, ref3);
    dst += (4 * dst_stride);

    hz_out1 = HORIZ_2TAP_FILT_UH(src0, src0, mask, filt_hz, FILTER_BITS);
    hz_out3 = HORIZ_2TAP_FILT_UH(src1, src1, mask, filt_hz, FILTER_BITS);
    ILVEV_B2_UB(hz_out0, hz_out1, hz_out2, hz_out3, vec0, vec1);
    DOTP_UB2_UH(vec0, vec1, filt_vt, filt_vt, tmp0, tmp1);
    SRARI_H2_UH(tmp0, tmp1, FILTER_BITS);
    src0 = (v16u8)__msa_pckev_b((v16i8)tmp1, (v16i8)tmp0);

    hz_out0 = HORIZ_2TAP_FILT_UH(src2, src2, mask, filt_hz, FILTER_BITS);
    hz_out2 = HORIZ_2TAP_FILT_UH(src3, src3, mask, filt_hz, FILTER_BITS);
    ILVEV_B2_UB(hz_out1, hz_out0, hz_out3, hz_out2, vec0, vec1);
    DOTP_UB2_UH(vec0, vec1, filt_vt, filt_vt, tmp0, tmp1);
    SRARI_H2_UH(tmp0, tmp1, FILTER_BITS);
    src1 = (v16u8)__msa_pckev_b((v16i8)tmp1, (v16i8)tmp0);

    hz_out1 = HORIZ_2TAP_FILT_UH(src4, src4, mask, filt_hz, FILTER_BITS);
    hz_out3 = HORIZ_2TAP_FILT_UH(src5, src5, mask, filt_hz, FILTER_BITS);
    ILVEV_B2_UB(hz_out0, hz_out1, hz_out2, hz_out3, vec0, vec1);
    DOTP_UB2_UH(vec0, vec1, filt_vt, filt_vt, tmp0, tmp1);
    SRARI_H2_UH(tmp0, tmp1, FILTER_BITS);
    src2 = (v16u8)__msa_pckev_b((v16i8)tmp1, (v16i8)tmp0);

    hz_out0 = HORIZ_2TAP_FILT_UH(src6, src6, mask, filt_hz, FILTER_BITS);
    hz_out2 = HORIZ_2TAP_FILT_UH(src7, src7, mask, filt_hz, FILTER_BITS);
    ILVEV_B2_UB(hz_out1, hz_out0, hz_out3, hz_out2, vec0, vec1);
    DOTP_UB2_UH(vec0, vec1, filt_vt, filt_vt, tmp0, tmp1);
    SRARI_H2_UH(tmp0, tmp1, FILTER_BITS);
    src3 = (v16u8)__msa_pckev_b((v16i8)tmp1, (v16i8)tmp0);

    CALC_MSE_AVG_B(src0, ref0, var, avg);
    CALC_MSE_AVG_B(src1, ref1, var, avg);
    CALC_MSE_AVG_B(src2, ref2, var, avg);
    CALC_MSE_AVG_B(src3, ref3, var, avg);
  }

  vec = __msa_hadd_s_w(avg, avg);
  *diff = HADD_SW_S32(vec);

  return HADD_SW_S32(var);
}

static uint32_t sub_pixel_sse_diff_32width_hv_msa(
    const uint8_t *src, int32_t src_stride, const uint8_t *dst,
    int32_t dst_stride, const uint8_t *filter_horiz, const uint8_t *filter_vert,
    int32_t height, int32_t *diff) {
  uint32_t loop_cnt, sse = 0;
  int32_t diff0[2];

  for (loop_cnt = 0; loop_cnt < 2; ++loop_cnt) {
    sse += sub_pixel_sse_diff_16width_hv_msa(src, src_stride, dst, dst_stride,
                                             filter_horiz, filter_vert, height,
                                             &diff0[loop_cnt]);
    src += 16;
    dst += 16;
  }

  *diff = diff0[0] + diff0[1];

  return sse;
}

static uint32_t sub_pixel_sse_diff_64width_hv_msa(
    const uint8_t *src, int32_t src_stride, const uint8_t *dst,
    int32_t dst_stride, const uint8_t *filter_horiz, const uint8_t *filter_vert,
    int32_t height, int32_t *diff) {
  uint32_t loop_cnt, sse = 0;
  int32_t diff0[4];

  for (loop_cnt = 0; loop_cnt < 4; ++loop_cnt) {
    sse += sub_pixel_sse_diff_16width_hv_msa(src, src_stride, dst, dst_stride,
                                             filter_horiz, filter_vert, height,
                                             &diff0[loop_cnt]);
    src += 16;
    dst += 16;
  }

  *diff = diff0[0] + diff0[1] + diff0[2] + diff0[3];

  return sse;
}

static uint32_t sub_pixel_avg_sse_diff_4width_h_msa(
    const uint8_t *src, int32_t src_stride, const uint8_t *dst,
    int32_t dst_stride, const uint8_t *sec_pred, const uint8_t *filter,
    int32_t height, int32_t *diff) {
  int16_t filtval;
  uint32_t loop_cnt;
  uint32_t ref0, ref1, ref2, ref3;
  v16u8 out, pred, filt0, ref = { 0 };
  v16i8 src0, src1, src2, src3;
  v16i8 mask = { 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8 };
  v8u16 vec0, vec1, vec2, vec3;
  v8i16 avg = { 0 };
  v4i32 vec, var = { 0 };

  filtval = LH(filter);
  filt0 = (v16u8)__msa_fill_h(filtval);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_SB4(src, src_stride, src0, src1, src2, src3);
    src += (4 * src_stride);
    pred = LD_UB(sec_pred);
    sec_pred += 16;
    LW4(dst, dst_stride, ref0, ref1, ref2, ref3);
    dst += (4 * dst_stride);

    INSERT_W4_UB(ref0, ref1, ref2, ref3, ref);
    VSHF_B2_UH(src0, src0, src1, src1, mask, mask, vec0, vec1);
    VSHF_B2_UH(src2, src2, src3, src3, mask, mask, vec2, vec3);
    DOTP_UB4_UH(vec0, vec1, vec2, vec3, filt0, filt0, filt0, filt0, vec0, vec1,
                vec2, vec3);
    SRARI_H4_UH(vec0, vec1, vec2, vec3, FILTER_BITS);
    PCKEV_B4_SB(vec0, vec0, vec1, vec1, vec2, vec2, vec3, vec3, src0, src1,
                src2, src3);
    ILVEV_W2_SB(src0, src1, src2, src3, src0, src2);
    out = (v16u8)__msa_ilvev_d((v2i64)src2, (v2i64)src0);
    out = __msa_aver_u_b(out, pred);
    CALC_MSE_AVG_B(out, ref, var, avg);
  }

  vec = __msa_hadd_s_w(avg, avg);
  *diff = HADD_SW_S32(vec);

  return HADD_SW_S32(var);
}

static uint32_t sub_pixel_avg_sse_diff_8width_h_msa(
    const uint8_t *src, int32_t src_stride, const uint8_t *dst,
    int32_t dst_stride, const uint8_t *sec_pred, const uint8_t *filter,
    int32_t height, int32_t *diff) {
  int16_t filtval;
  uint32_t loop_cnt;
  v16u8 out, pred, filt0;
  v16u8 ref0, ref1, ref2, ref3;
  v16i8 src0, src1, src2, src3;
  v16i8 mask = { 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8 };
  v8u16 vec0, vec1, vec2, vec3;
  v8i16 avg = { 0 };
  v4i32 vec, var = { 0 };

  filtval = LH(filter);
  filt0 = (v16u8)__msa_fill_h(filtval);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_SB4(src, src_stride, src0, src1, src2, src3);
    src += (4 * src_stride);
    LD_UB4(dst, dst_stride, ref0, ref1, ref2, ref3);
    dst += (4 * dst_stride);

    PCKEV_D2_UB(ref1, ref0, ref3, ref2, ref0, ref1);
    VSHF_B2_UH(src0, src0, src1, src1, mask, mask, vec0, vec1);
    VSHF_B2_UH(src2, src2, src3, src3, mask, mask, vec2, vec3);
    DOTP_UB4_UH(vec0, vec1, vec2, vec3, filt0, filt0, filt0, filt0, vec0, vec1,
                vec2, vec3);
    SRARI_H4_UH(vec0, vec1, vec2, vec3, FILTER_BITS);
    PCKEV_B4_SB(vec0, vec0, vec1, vec1, vec2, vec2, vec3, vec3, src0, src1,
                src2, src3);
    out = (v16u8)__msa_ilvev_d((v2i64)src1, (v2i64)src0);

    pred = LD_UB(sec_pred);
    sec_pred += 16;
    out = __msa_aver_u_b(out, pred);
    CALC_MSE_AVG_B(out, ref0, var, avg);
    out = (v16u8)__msa_ilvev_d((v2i64)src3, (v2i64)src2);
    pred = LD_UB(sec_pred);
    sec_pred += 16;
    out = __msa_aver_u_b(out, pred);
    CALC_MSE_AVG_B(out, ref1, var, avg);
  }

  vec = __msa_hadd_s_w(avg, avg);
  *diff = HADD_SW_S32(vec);

  return HADD_SW_S32(var);
}

static uint32_t subpel_avg_ssediff_16w_h_msa(
    const uint8_t *src, int32_t src_stride, const uint8_t *dst,
    int32_t dst_stride, const uint8_t *sec_pred, const uint8_t *filter,
    int32_t height, int32_t *diff, int32_t width) {
  int16_t filtval;
  uint32_t loop_cnt;
  v16i8 src0, src1, src2, src3, src4, src5, src6, src7;
  v16i8 mask = { 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8 };
  v16u8 dst0, dst1, dst2, dst3;
  v16u8 tmp0, tmp1, tmp2, tmp3;
  v16u8 pred0, pred1, pred2, pred3, filt0;
  v8u16 vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7;
  v8u16 out0, out1, out2, out3, out4, out5, out6, out7;
  v8i16 avg = { 0 };
  v4i32 vec, var = { 0 };

  filtval = LH(filter);
  filt0 = (v16u8)__msa_fill_h(filtval);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_SB4(src, src_stride, src0, src2, src4, src6);
    LD_SB4(src + 8, src_stride, src1, src3, src5, src7);
    src += (4 * src_stride);
    LD_UB4(dst, dst_stride, dst0, dst1, dst2, dst3);
    dst += (4 * dst_stride);
    LD_UB4(sec_pred, width, pred0, pred1, pred2, pred3);
    sec_pred += (4 * width);

    VSHF_B2_UH(src0, src0, src1, src1, mask, mask, vec0, vec1);
    VSHF_B2_UH(src2, src2, src3, src3, mask, mask, vec2, vec3);
    VSHF_B2_UH(src4, src4, src5, src5, mask, mask, vec4, vec5);
    VSHF_B2_UH(src6, src6, src7, src7, mask, mask, vec6, vec7);
    DOTP_UB4_UH(vec0, vec1, vec2, vec3, filt0, filt0, filt0, filt0, out0, out1,
                out2, out3);
    DOTP_UB4_UH(vec4, vec5, vec6, vec7, filt0, filt0, filt0, filt0, out4, out5,
                out6, out7);
    SRARI_H4_UH(out0, out1, out2, out3, FILTER_BITS);
    SRARI_H4_UH(out4, out5, out6, out7, FILTER_BITS);
    PCKEV_B4_UB(out1, out0, out3, out2, out5, out4, out7, out6, tmp0, tmp1,
                tmp2, tmp3);
    AVER_UB4_UB(tmp0, pred0, tmp1, pred1, tmp2, pred2, tmp3, pred3, tmp0, tmp1,
                tmp2, tmp3);

    CALC_MSE_AVG_B(tmp0, dst0, var, avg);
    CALC_MSE_AVG_B(tmp1, dst1, var, avg);
    CALC_MSE_AVG_B(tmp2, dst2, var, avg);
    CALC_MSE_AVG_B(tmp3, dst3, var, avg);
  }

  vec = __msa_hadd_s_w(avg, avg);
  *diff = HADD_SW_S32(vec);

  return HADD_SW_S32(var);
}

static uint32_t sub_pixel_avg_sse_diff_16width_h_msa(
    const uint8_t *src, int32_t src_stride, const uint8_t *dst,
    int32_t dst_stride, const uint8_t *sec_pred, const uint8_t *filter,
    int32_t height, int32_t *diff) {
  return subpel_avg_ssediff_16w_h_msa(src, src_stride, dst, dst_stride,
                                      sec_pred, filter, height, diff, 16);
}

static uint32_t sub_pixel_avg_sse_diff_32width_h_msa(
    const uint8_t *src, int32_t src_stride, const uint8_t *dst,
    int32_t dst_stride, const uint8_t *sec_pred, const uint8_t *filter,
    int32_t height, int32_t *diff) {
  uint32_t loop_cnt, sse = 0;
  int32_t diff0[2];

  for (loop_cnt = 0; loop_cnt < 2; ++loop_cnt) {
    sse +=
        subpel_avg_ssediff_16w_h_msa(src, src_stride, dst, dst_stride, sec_pred,
                                     filter, height, &diff0[loop_cnt], 32);
    src += 16;
    dst += 16;
    sec_pred += 16;
  }

  *diff = diff0[0] + diff0[1];

  return sse;
}

static uint32_t sub_pixel_avg_sse_diff_64width_h_msa(
    const uint8_t *src, int32_t src_stride, const uint8_t *dst,
    int32_t dst_stride, const uint8_t *sec_pred, const uint8_t *filter,
    int32_t height, int32_t *diff) {
  uint32_t loop_cnt, sse = 0;
  int32_t diff0[4];

  for (loop_cnt = 0; loop_cnt < 4; ++loop_cnt) {
    sse +=
        subpel_avg_ssediff_16w_h_msa(src, src_stride, dst, dst_stride, sec_pred,
                                     filter, height, &diff0[loop_cnt], 64);
    src += 16;
    dst += 16;
    sec_pred += 16;
  }

  *diff = diff0[0] + diff0[1] + diff0[2] + diff0[3];

  return sse;
}

static uint32_t sub_pixel_avg_sse_diff_4width_v_msa(
    const uint8_t *src, int32_t src_stride, const uint8_t *dst,
    int32_t dst_stride, const uint8_t *sec_pred, const uint8_t *filter,
    int32_t height, int32_t *diff) {
  int16_t filtval;
  uint32_t loop_cnt;
  uint32_t ref0, ref1, ref2, ref3;
  v16u8 src0, src1, src2, src3, src4;
  v16u8 src10_r, src32_r, src21_r, src43_r;
  v16u8 out, pred, ref = { 0 };
  v16u8 src2110, src4332, filt0;
  v8i16 avg = { 0 };
  v4i32 vec, var = { 0 };
  v8u16 tmp0, tmp1;

  filtval = LH(filter);
  filt0 = (v16u8)__msa_fill_h(filtval);

  src0 = LD_UB(src);
  src += src_stride;

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_UB4(src, src_stride, src1, src2, src3, src4);
    src += (4 * src_stride);
    pred = LD_UB(sec_pred);
    sec_pred += 16;
    LW4(dst, dst_stride, ref0, ref1, ref2, ref3);
    dst += (4 * dst_stride);

    INSERT_W4_UB(ref0, ref1, ref2, ref3, ref);
    ILVR_B4_UB(src1, src0, src2, src1, src3, src2, src4, src3, src10_r, src21_r,
               src32_r, src43_r);
    ILVR_D2_UB(src21_r, src10_r, src43_r, src32_r, src2110, src4332);
    DOTP_UB2_UH(src2110, src4332, filt0, filt0, tmp0, tmp1);
    SRARI_H2_UH(tmp0, tmp1, FILTER_BITS);

    out = (v16u8)__msa_pckev_b((v16i8)tmp1, (v16i8)tmp0);
    out = __msa_aver_u_b(out, pred);
    CALC_MSE_AVG_B(out, ref, var, avg);
    src0 = src4;
  }

  vec = __msa_hadd_s_w(avg, avg);
  *diff = HADD_SW_S32(vec);

  return HADD_SW_S32(var);
}

static uint32_t sub_pixel_avg_sse_diff_8width_v_msa(
    const uint8_t *src, int32_t src_stride, const uint8_t *dst,
    int32_t dst_stride, const uint8_t *sec_pred, const uint8_t *filter,
    int32_t height, int32_t *diff) {
  int16_t filtval;
  uint32_t loop_cnt;
  v16u8 src0, src1, src2, src3, src4;
  v16u8 ref0, ref1, ref2, ref3;
  v16u8 pred0, pred1, filt0;
  v8u16 vec0, vec1, vec2, vec3;
  v8u16 tmp0, tmp1, tmp2, tmp3;
  v8i16 avg = { 0 };
  v4i32 vec, var = { 0 };

  filtval = LH(filter);
  filt0 = (v16u8)__msa_fill_h(filtval);

  src0 = LD_UB(src);
  src += src_stride;

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_UB4(src, src_stride, src1, src2, src3, src4);
    src += (4 * src_stride);
    LD_UB2(sec_pred, 16, pred0, pred1);
    sec_pred += 32;
    LD_UB4(dst, dst_stride, ref0, ref1, ref2, ref3);
    dst += (4 * dst_stride);
    PCKEV_D2_UB(ref1, ref0, ref3, ref2, ref0, ref1);
    ILVR_B4_UH(src1, src0, src2, src1, src3, src2, src4, src3, vec0, vec1, vec2,
               vec3);
    DOTP_UB4_UH(vec0, vec1, vec2, vec3, filt0, filt0, filt0, filt0, tmp0, tmp1,
                tmp2, tmp3);
    SRARI_H4_UH(tmp0, tmp1, tmp2, tmp3, FILTER_BITS);
    PCKEV_B2_UB(tmp1, tmp0, tmp3, tmp2, src0, src1);
    AVER_UB2_UB(src0, pred0, src1, pred1, src0, src1);
    CALC_MSE_AVG_B(src0, ref0, var, avg);
    CALC_MSE_AVG_B(src1, ref1, var, avg);

    src0 = src4;
  }

  vec = __msa_hadd_s_w(avg, avg);
  *diff = HADD_SW_S32(vec);

  return HADD_SW_S32(var);
}

static uint32_t subpel_avg_ssediff_16w_v_msa(
    const uint8_t *src, int32_t src_stride, const uint8_t *dst,
    int32_t dst_stride, const uint8_t *sec_pred, const uint8_t *filter,
    int32_t height, int32_t *diff, int32_t width) {
  int16_t filtval;
  uint32_t loop_cnt;
  v16u8 ref0, ref1, ref2, ref3;
  v16u8 pred0, pred1, pred2, pred3;
  v16u8 src0, src1, src2, src3, src4;
  v16u8 out0, out1, out2, out3, filt0;
  v8u16 vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7;
  v8u16 tmp0, tmp1, tmp2, tmp3;
  v8i16 avg = { 0 };
  v4i32 vec, var = { 0 };

  filtval = LH(filter);
  filt0 = (v16u8)__msa_fill_h(filtval);

  src0 = LD_UB(src);
  src += src_stride;

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_UB4(src, src_stride, src1, src2, src3, src4);
    src += (4 * src_stride);
    LD_UB4(sec_pred, width, pred0, pred1, pred2, pred3);
    sec_pred += (4 * width);

    ILVR_B2_UH(src1, src0, src2, src1, vec0, vec2);
    ILVL_B2_UH(src1, src0, src2, src1, vec1, vec3);
    DOTP_UB2_UH(vec0, vec1, filt0, filt0, tmp0, tmp1);
    SRARI_H2_UH(tmp0, tmp1, FILTER_BITS);
    out0 = (v16u8)__msa_pckev_b((v16i8)tmp1, (v16i8)tmp0);

    ILVR_B2_UH(src3, src2, src4, src3, vec4, vec6);
    ILVL_B2_UH(src3, src2, src4, src3, vec5, vec7);
    DOTP_UB2_UH(vec2, vec3, filt0, filt0, tmp2, tmp3);
    SRARI_H2_UH(tmp2, tmp3, FILTER_BITS);
    out1 = (v16u8)__msa_pckev_b((v16i8)tmp3, (v16i8)tmp2);

    DOTP_UB2_UH(vec4, vec5, filt0, filt0, tmp0, tmp1);
    SRARI_H2_UH(tmp0, tmp1, FILTER_BITS);
    out2 = (v16u8)__msa_pckev_b((v16i8)tmp1, (v16i8)tmp0);

    DOTP_UB2_UH(vec6, vec7, filt0, filt0, tmp2, tmp3);
    SRARI_H2_UH(tmp2, tmp3, FILTER_BITS);
    out3 = (v16u8)__msa_pckev_b((v16i8)tmp3, (v16i8)tmp2);

    src0 = src4;
    LD_UB4(dst, dst_stride, ref0, ref1, ref2, ref3);
    dst += (4 * dst_stride);

    AVER_UB4_UB(out0, pred0, out1, pred1, out2, pred2, out3, pred3, out0, out1,
                out2, out3);

    CALC_MSE_AVG_B(out0, ref0, var, avg);
    CALC_MSE_AVG_B(out1, ref1, var, avg);
    CALC_MSE_AVG_B(out2, ref2, var, avg);
    CALC_MSE_AVG_B(out3, ref3, var, avg);
  }

  vec = __msa_hadd_s_w(avg, avg);
  *diff = HADD_SW_S32(vec);

  return HADD_SW_S32(var);
}

static uint32_t sub_pixel_avg_sse_diff_16width_v_msa(
    const uint8_t *src, int32_t src_stride, const uint8_t *dst,
    int32_t dst_stride, const uint8_t *sec_pred, const uint8_t *filter,
    int32_t height, int32_t *diff) {
  return subpel_avg_ssediff_16w_v_msa(src, src_stride, dst, dst_stride,
                                      sec_pred, filter, height, diff, 16);
}

static uint32_t sub_pixel_avg_sse_diff_32width_v_msa(
    const uint8_t *src, int32_t src_stride, const uint8_t *dst,
    int32_t dst_stride, const uint8_t *sec_pred, const uint8_t *filter,
    int32_t height, int32_t *diff) {
  uint32_t loop_cnt, sse = 0;
  int32_t diff0[2];

  for (loop_cnt = 0; loop_cnt < 2; ++loop_cnt) {
    sse +=
        subpel_avg_ssediff_16w_v_msa(src, src_stride, dst, dst_stride, sec_pred,
                                     filter, height, &diff0[loop_cnt], 32);
    src += 16;
    dst += 16;
    sec_pred += 16;
  }

  *diff = diff0[0] + diff0[1];

  return sse;
}

static uint32_t sub_pixel_avg_sse_diff_64width_v_msa(
    const uint8_t *src, int32_t src_stride, const uint8_t *dst,
    int32_t dst_stride, const uint8_t *sec_pred, const uint8_t *filter,
    int32_t height, int32_t *diff) {
  uint32_t loop_cnt, sse = 0;
  int32_t diff0[4];

  for (loop_cnt = 0; loop_cnt < 4; ++loop_cnt) {
    sse +=
        subpel_avg_ssediff_16w_v_msa(src, src_stride, dst, dst_stride, sec_pred,
                                     filter, height, &diff0[loop_cnt], 64);
    src += 16;
    dst += 16;
    sec_pred += 16;
  }

  *diff = diff0[0] + diff0[1] + diff0[2] + diff0[3];

  return sse;
}

static uint32_t sub_pixel_avg_sse_diff_4width_hv_msa(
    const uint8_t *src, int32_t src_stride, const uint8_t *dst,
    int32_t dst_stride, const uint8_t *sec_pred, const uint8_t *filter_horiz,
    const uint8_t *filter_vert, int32_t height, int32_t *diff) {
  int16_t filtval;
  uint32_t loop_cnt;
  uint32_t ref0, ref1, ref2, ref3;
  v16u8 src0, src1, src2, src3, src4;
  v16u8 mask = { 0, 1, 1, 2, 2, 3, 3, 4, 16, 17, 17, 18, 18, 19, 19, 20 };
  v16u8 filt_hz, filt_vt, vec0, vec1;
  v16u8 out, pred, ref = { 0 };
  v8u16 hz_out0, hz_out1, hz_out2, hz_out3, hz_out4, tmp0, tmp1;
  v8i16 avg = { 0 };
  v4i32 vec, var = { 0 };

  filtval = LH(filter_horiz);
  filt_hz = (v16u8)__msa_fill_h(filtval);
  filtval = LH(filter_vert);
  filt_vt = (v16u8)__msa_fill_h(filtval);

  src0 = LD_UB(src);
  src += src_stride;

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_UB4(src, src_stride, src1, src2, src3, src4);
    src += (4 * src_stride);
    pred = LD_UB(sec_pred);
    sec_pred += 16;
    LW4(dst, dst_stride, ref0, ref1, ref2, ref3);
    dst += (4 * dst_stride);
    INSERT_W4_UB(ref0, ref1, ref2, ref3, ref);
    hz_out0 = HORIZ_2TAP_FILT_UH(src0, src1, mask, filt_hz, FILTER_BITS);
    hz_out2 = HORIZ_2TAP_FILT_UH(src2, src3, mask, filt_hz, FILTER_BITS);
    hz_out4 = HORIZ_2TAP_FILT_UH(src4, src4, mask, filt_hz, FILTER_BITS);
    hz_out1 = (v8u16)__msa_sldi_b((v16i8)hz_out2, (v16i8)hz_out0, 8);
    hz_out3 = (v8u16)__msa_pckod_d((v2i64)hz_out4, (v2i64)hz_out2);
    ILVEV_B2_UB(hz_out0, hz_out1, hz_out2, hz_out3, vec0, vec1);
    DOTP_UB2_UH(vec0, vec1, filt_vt, filt_vt, tmp0, tmp1);
    SRARI_H2_UH(tmp0, tmp1, FILTER_BITS);
    out = (v16u8)__msa_pckev_b((v16i8)tmp1, (v16i8)tmp0);
    out = __msa_aver_u_b(out, pred);
    CALC_MSE_AVG_B(out, ref, var, avg);
    src0 = src4;
  }

  vec = __msa_hadd_s_w(avg, avg);
  *diff = HADD_SW_S32(vec);

  return HADD_SW_S32(var);
}

static uint32_t sub_pixel_avg_sse_diff_8width_hv_msa(
    const uint8_t *src, int32_t src_stride, const uint8_t *dst,
    int32_t dst_stride, const uint8_t *sec_pred, const uint8_t *filter_horiz,
    const uint8_t *filter_vert, int32_t height, int32_t *diff) {
  int16_t filtval;
  uint32_t loop_cnt;
  v16u8 ref0, ref1, ref2, ref3;
  v16u8 src0, src1, src2, src3, src4;
  v16u8 pred0, pred1, out0, out1;
  v16u8 filt_hz, filt_vt, vec0;
  v16u8 mask = { 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8 };
  v8u16 hz_out0, hz_out1, tmp0, tmp1, tmp2, tmp3;
  v8i16 avg = { 0 };
  v4i32 vec, var = { 0 };

  filtval = LH(filter_horiz);
  filt_hz = (v16u8)__msa_fill_h(filtval);
  filtval = LH(filter_vert);
  filt_vt = (v16u8)__msa_fill_h(filtval);

  src0 = LD_UB(src);
  src += src_stride;
  hz_out0 = HORIZ_2TAP_FILT_UH(src0, src0, mask, filt_hz, FILTER_BITS);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_UB4(src, src_stride, src1, src2, src3, src4);
    src += (4 * src_stride);
    LD_UB2(sec_pred, 16, pred0, pred1);
    sec_pred += 32;
    LD_UB4(dst, dst_stride, ref0, ref1, ref2, ref3);
    dst += (4 * dst_stride);

    PCKEV_D2_UB(ref1, ref0, ref3, ref2, ref0, ref1);
    hz_out1 = HORIZ_2TAP_FILT_UH(src1, src1, mask, filt_hz, FILTER_BITS);

    vec0 = (v16u8)__msa_ilvev_b((v16i8)hz_out1, (v16i8)hz_out0);
    tmp0 = __msa_dotp_u_h(vec0, filt_vt);
    hz_out0 = HORIZ_2TAP_FILT_UH(src2, src2, mask, filt_hz, FILTER_BITS);

    vec0 = (v16u8)__msa_ilvev_b((v16i8)hz_out0, (v16i8)hz_out1);
    tmp1 = __msa_dotp_u_h(vec0, filt_vt);
    SRARI_H2_UH(tmp0, tmp1, FILTER_BITS);
    hz_out1 = HORIZ_2TAP_FILT_UH(src3, src3, mask, filt_hz, FILTER_BITS);

    vec0 = (v16u8)__msa_ilvev_b((v16i8)hz_out1, (v16i8)hz_out0);
    tmp2 = __msa_dotp_u_h(vec0, filt_vt);
    hz_out0 = HORIZ_2TAP_FILT_UH(src4, src4, mask, filt_hz, FILTER_BITS);

    vec0 = (v16u8)__msa_ilvev_b((v16i8)hz_out0, (v16i8)hz_out1);
    tmp3 = __msa_dotp_u_h(vec0, filt_vt);

    SRARI_H2_UH(tmp2, tmp3, FILTER_BITS);
    PCKEV_B2_UB(tmp1, tmp0, tmp3, tmp2, out0, out1);
    AVER_UB2_UB(out0, pred0, out1, pred1, out0, out1);

    CALC_MSE_AVG_B(out0, ref0, var, avg);
    CALC_MSE_AVG_B(out1, ref1, var, avg);
  }

  vec = __msa_hadd_s_w(avg, avg);
  *diff = HADD_SW_S32(vec);

  return HADD_SW_S32(var);
}

static uint32_t subpel_avg_ssediff_16w_hv_msa(
    const uint8_t *src, int32_t src_stride, const uint8_t *dst,
    int32_t dst_stride, const uint8_t *sec_pred, const uint8_t *filter_horiz,
    const uint8_t *filter_vert, int32_t height, int32_t *diff, int32_t width) {
  int16_t filtval;
  uint32_t loop_cnt;
  v16u8 src0, src1, src2, src3, src4, src5, src6, src7;
  v16u8 ref0, ref1, ref2, ref3;
  v16u8 pred0, pred1, pred2, pred3;
  v16u8 out0, out1, out2, out3;
  v16u8 filt_hz, filt_vt, vec0, vec1;
  v16u8 mask = { 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8 };
  v8u16 hz_out0, hz_out1, hz_out2, hz_out3, tmp0, tmp1;
  v8i16 avg = { 0 };
  v4i32 vec, var = { 0 };

  filtval = LH(filter_horiz);
  filt_hz = (v16u8)__msa_fill_h(filtval);
  filtval = LH(filter_vert);
  filt_vt = (v16u8)__msa_fill_h(filtval);

  LD_UB2(src, 8, src0, src1);
  src += src_stride;

  hz_out0 = HORIZ_2TAP_FILT_UH(src0, src0, mask, filt_hz, FILTER_BITS);
  hz_out2 = HORIZ_2TAP_FILT_UH(src1, src1, mask, filt_hz, FILTER_BITS);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_UB4(src, src_stride, src0, src2, src4, src6);
    LD_UB4(src + 8, src_stride, src1, src3, src5, src7);
    src += (4 * src_stride);
    LD_UB4(sec_pred, width, pred0, pred1, pred2, pred3);
    sec_pred += (4 * width);

    hz_out1 = HORIZ_2TAP_FILT_UH(src0, src0, mask, filt_hz, FILTER_BITS);
    hz_out3 = HORIZ_2TAP_FILT_UH(src1, src1, mask, filt_hz, FILTER_BITS);
    ILVEV_B2_UB(hz_out0, hz_out1, hz_out2, hz_out3, vec0, vec1);
    DOTP_UB2_UH(vec0, vec1, filt_vt, filt_vt, tmp0, tmp1);
    SRARI_H2_UH(tmp0, tmp1, FILTER_BITS);
    out0 = (v16u8)__msa_pckev_b((v16i8)tmp1, (v16i8)tmp0);

    hz_out0 = HORIZ_2TAP_FILT_UH(src2, src2, mask, filt_hz, FILTER_BITS);
    hz_out2 = HORIZ_2TAP_FILT_UH(src3, src3, mask, filt_hz, FILTER_BITS);
    ILVEV_B2_UB(hz_out1, hz_out0, hz_out3, hz_out2, vec0, vec1);
    DOTP_UB2_UH(vec0, vec1, filt_vt, filt_vt, tmp0, tmp1);
    SRARI_H2_UH(tmp0, tmp1, FILTER_BITS);
    out1 = (v16u8)__msa_pckev_b((v16i8)tmp1, (v16i8)tmp0);

    hz_out1 = HORIZ_2TAP_FILT_UH(src4, src4, mask, filt_hz, FILTER_BITS);
    hz_out3 = HORIZ_2TAP_FILT_UH(src5, src5, mask, filt_hz, FILTER_BITS);
    ILVEV_B2_UB(hz_out0, hz_out1, hz_out2, hz_out3, vec0, vec1);
    DOTP_UB2_UH(vec0, vec1, filt_vt, filt_vt, tmp0, tmp1);
    SRARI_H2_UH(tmp0, tmp1, FILTER_BITS);
    out2 = (v16u8)__msa_pckev_b((v16i8)tmp1, (v16i8)tmp0);

    hz_out0 = HORIZ_2TAP_FILT_UH(src6, src6, mask, filt_hz, FILTER_BITS);
    hz_out2 = HORIZ_2TAP_FILT_UH(src7, src7, mask, filt_hz, FILTER_BITS);
    ILVEV_B2_UB(hz_out1, hz_out0, hz_out3, hz_out2, vec0, vec1);
    DOTP_UB2_UH(vec0, vec1, filt_vt, filt_vt, tmp0, tmp1);
    SRARI_H2_UH(tmp0, tmp1, FILTER_BITS);
    out3 = (v16u8)__msa_pckev_b((v16i8)tmp1, (v16i8)tmp0);

    LD_UB4(dst, dst_stride, ref0, ref1, ref2, ref3);
    dst += (4 * dst_stride);

    AVER_UB4_UB(out0, pred0, out1, pred1, out2, pred2, out3, pred3, out0, out1,
                out2, out3);

    CALC_MSE_AVG_B(out0, ref0, var, avg);
    CALC_MSE_AVG_B(out1, ref1, var, avg);
    CALC_MSE_AVG_B(out2, ref2, var, avg);
    CALC_MSE_AVG_B(out3, ref3, var, avg);
  }

  vec = __msa_hadd_s_w(avg, avg);
  *diff = HADD_SW_S32(vec);

  return HADD_SW_S32(var);
}

static uint32_t sub_pixel_avg_sse_diff_16width_hv_msa(
    const uint8_t *src, int32_t src_stride, const uint8_t *dst,
    int32_t dst_stride, const uint8_t *sec_pred, const uint8_t *filter_horiz,
    const uint8_t *filter_vert, int32_t height, int32_t *diff) {
  return subpel_avg_ssediff_16w_hv_msa(src, src_stride, dst, dst_stride,
                                       sec_pred, filter_horiz, filter_vert,
                                       height, diff, 16);
}

static uint32_t sub_pixel_avg_sse_diff_32width_hv_msa(
    const uint8_t *src, int32_t src_stride, const uint8_t *dst,
    int32_t dst_stride, const uint8_t *sec_pred, const uint8_t *filter_horiz,
    const uint8_t *filter_vert, int32_t height, int32_t *diff) {
  uint32_t loop_cnt, sse = 0;
  int32_t diff0[2];

  for (loop_cnt = 0; loop_cnt < 2; ++loop_cnt) {
    sse += subpel_avg_ssediff_16w_hv_msa(src, src_stride, dst, dst_stride,
                                         sec_pred, filter_horiz, filter_vert,
                                         height, &diff0[loop_cnt], 32);
    src += 16;
    dst += 16;
    sec_pred += 16;
  }

  *diff = diff0[0] + diff0[1];

  return sse;
}

static uint32_t sub_pixel_avg_sse_diff_64width_hv_msa(
    const uint8_t *src, int32_t src_stride, const uint8_t *dst,
    int32_t dst_stride, const uint8_t *sec_pred, const uint8_t *filter_horiz,
    const uint8_t *filter_vert, int32_t height, int32_t *diff) {
  uint32_t loop_cnt, sse = 0;
  int32_t diff0[4];

  for (loop_cnt = 0; loop_cnt < 4; ++loop_cnt) {
    sse += subpel_avg_ssediff_16w_hv_msa(src, src_stride, dst, dst_stride,
                                         sec_pred, filter_horiz, filter_vert,
                                         height, &diff0[loop_cnt], 64);
    src += 16;
    dst += 16;
    sec_pred += 16;
  }

  *diff = diff0[0] + diff0[1] + diff0[2] + diff0[3];

  return sse;
}

#define VARIANCE_4Wx4H(sse, diff) VARIANCE_WxH(sse, diff, 4);
#define VARIANCE_4Wx8H(sse, diff) VARIANCE_WxH(sse, diff, 5);
#define VARIANCE_8Wx4H(sse, diff) VARIANCE_WxH(sse, diff, 5);
#define VARIANCE_8Wx8H(sse, diff) VARIANCE_WxH(sse, diff, 6);
#define VARIANCE_8Wx16H(sse, diff) VARIANCE_WxH(sse, diff, 7);
#define VARIANCE_16Wx8H(sse, diff) VARIANCE_WxH(sse, diff, 7);
#define VARIANCE_16Wx16H(sse, diff) VARIANCE_WxH(sse, diff, 8);

#define VARIANCE_16Wx32H(sse, diff) VARIANCE_LARGE_WxH(sse, diff, 9);
#define VARIANCE_32Wx16H(sse, diff) VARIANCE_LARGE_WxH(sse, diff, 9);
#define VARIANCE_32Wx32H(sse, diff) VARIANCE_LARGE_WxH(sse, diff, 10);
#define VARIANCE_32Wx64H(sse, diff) VARIANCE_LARGE_WxH(sse, diff, 11);
#define VARIANCE_64Wx32H(sse, diff) VARIANCE_LARGE_WxH(sse, diff, 11);
#define VARIANCE_64Wx64H(sse, diff) VARIANCE_LARGE_WxH(sse, diff, 12);

#define VPX_SUB_PIXEL_VARIANCE_WDXHT_MSA(wd, ht)                              \
  uint32_t vpx_sub_pixel_variance##wd##x##ht##_msa(                           \
      const uint8_t *src, int32_t src_stride, int32_t x_offset,               \
      int32_t y_offset, const uint8_t *ref, int32_t ref_stride,               \
      uint32_t *sse) {                                                        \
    int32_t diff;                                                             \
    uint32_t var;                                                             \
    const uint8_t *h_filter = bilinear_filters_msa[x_offset];                 \
    const uint8_t *v_filter = bilinear_filters_msa[y_offset];                 \
                                                                              \
    if (y_offset) {                                                           \
      if (x_offset) {                                                         \
        *sse = sub_pixel_sse_diff_##wd##width_hv_msa(                         \
            src, src_stride, ref, ref_stride, h_filter, v_filter, ht, &diff); \
      } else {                                                                \
        *sse = sub_pixel_sse_diff_##wd##width_v_msa(                          \
            src, src_stride, ref, ref_stride, v_filter, ht, &diff);           \
      }                                                                       \
                                                                              \
      var = VARIANCE_##wd##Wx##ht##H(*sse, diff);                             \
    } else {                                                                  \
      if (x_offset) {                                                         \
        *sse = sub_pixel_sse_diff_##wd##width_h_msa(                          \
            src, src_stride, ref, ref_stride, h_filter, ht, &diff);           \
                                                                              \
        var = VARIANCE_##wd##Wx##ht##H(*sse, diff);                           \
      } else {                                                                \
        var = vpx_variance##wd##x##ht##_msa(src, src_stride, ref, ref_stride, \
                                            sse);                             \
      }                                                                       \
    }                                                                         \
                                                                              \
    return var;                                                               \
  }

VPX_SUB_PIXEL_VARIANCE_WDXHT_MSA(4, 4);
VPX_SUB_PIXEL_VARIANCE_WDXHT_MSA(4, 8);

VPX_SUB_PIXEL_VARIANCE_WDXHT_MSA(8, 4);
VPX_SUB_PIXEL_VARIANCE_WDXHT_MSA(8, 8);
VPX_SUB_PIXEL_VARIANCE_WDXHT_MSA(8, 16);

VPX_SUB_PIXEL_VARIANCE_WDXHT_MSA(16, 8);
VPX_SUB_PIXEL_VARIANCE_WDXHT_MSA(16, 16);
VPX_SUB_PIXEL_VARIANCE_WDXHT_MSA(16, 32);

VPX_SUB_PIXEL_VARIANCE_WDXHT_MSA(32, 16);
VPX_SUB_PIXEL_VARIANCE_WDXHT_MSA(32, 32);
VPX_SUB_PIXEL_VARIANCE_WDXHT_MSA(32, 64);

VPX_SUB_PIXEL_VARIANCE_WDXHT_MSA(64, 32);
VPX_SUB_PIXEL_VARIANCE_WDXHT_MSA(64, 64);

#define VPX_SUB_PIXEL_AVG_VARIANCE_WDXHT_MSA(wd, ht)                          \
  uint32_t vpx_sub_pixel_avg_variance##wd##x##ht##_msa(                       \
      const uint8_t *src_ptr, int32_t src_stride, int32_t x_offset,           \
      int32_t y_offset, const uint8_t *ref_ptr, int32_t ref_stride,           \
      uint32_t *sse, const uint8_t *sec_pred) {                               \
    int32_t diff;                                                             \
    const uint8_t *h_filter = bilinear_filters_msa[x_offset];                 \
    const uint8_t *v_filter = bilinear_filters_msa[y_offset];                 \
                                                                              \
    if (y_offset) {                                                           \
      if (x_offset) {                                                         \
        *sse = sub_pixel_avg_sse_diff_##wd##width_hv_msa(                     \
            src_ptr, src_stride, ref_ptr, ref_stride, sec_pred, h_filter,     \
            v_filter, ht, &diff);                                             \
      } else {                                                                \
        *sse = sub_pixel_avg_sse_diff_##wd##width_v_msa(                      \
            src_ptr, src_stride, ref_ptr, ref_stride, sec_pred, v_filter, ht, \
            &diff);                                                           \
      }                                                                       \
    } else {                                                                  \
      if (x_offset) {                                                         \
        *sse = sub_pixel_avg_sse_diff_##wd##width_h_msa(                      \
            src_ptr, src_stride, ref_ptr, ref_stride, sec_pred, h_filter, ht, \
            &diff);                                                           \
      } else {                                                                \
        *sse = avg_sse_diff_##wd##width_msa(src_ptr, src_stride, ref_ptr,     \
                                            ref_stride, sec_pred, ht, &diff); \
      }                                                                       \
    }                                                                         \
                                                                              \
    return VARIANCE_##wd##Wx##ht##H(*sse, diff);                              \
  }

VPX_SUB_PIXEL_AVG_VARIANCE_WDXHT_MSA(4, 4);
VPX_SUB_PIXEL_AVG_VARIANCE_WDXHT_MSA(4, 8);

VPX_SUB_PIXEL_AVG_VARIANCE_WDXHT_MSA(8, 4);
VPX_SUB_PIXEL_AVG_VARIANCE_WDXHT_MSA(8, 8);
VPX_SUB_PIXEL_AVG_VARIANCE_WDXHT_MSA(8, 16);

VPX_SUB_PIXEL_AVG_VARIANCE_WDXHT_MSA(16, 8);
VPX_SUB_PIXEL_AVG_VARIANCE_WDXHT_MSA(16, 16);
VPX_SUB_PIXEL_AVG_VARIANCE_WDXHT_MSA(16, 32);

VPX_SUB_PIXEL_AVG_VARIANCE_WDXHT_MSA(32, 16);
VPX_SUB_PIXEL_AVG_VARIANCE_WDXHT_MSA(32, 32);

uint32_t vpx_sub_pixel_avg_variance32x64_msa(const uint8_t *src_ptr,
                                             int32_t src_stride,
                                             int32_t x_offset, int32_t y_offset,
                                             const uint8_t *ref_ptr,
                                             int32_t ref_stride, uint32_t *sse,
                                             const uint8_t *sec_pred) {
  int32_t diff;
  const uint8_t *h_filter = bilinear_filters_msa[x_offset];
  const uint8_t *v_filter = bilinear_filters_msa[y_offset];

  if (y_offset) {
    if (x_offset) {
      *sse = sub_pixel_avg_sse_diff_32width_hv_msa(
          src_ptr, src_stride, ref_ptr, ref_stride, sec_pred, h_filter,
          v_filter, 64, &diff);
    } else {
      *sse = sub_pixel_avg_sse_diff_32width_v_msa(src_ptr, src_stride, ref_ptr,
                                                  ref_stride, sec_pred,
                                                  v_filter, 64, &diff);
    }
  } else {
    if (x_offset) {
      *sse = sub_pixel_avg_sse_diff_32width_h_msa(src_ptr, src_stride, ref_ptr,
                                                  ref_stride, sec_pred,
                                                  h_filter, 64, &diff);
    } else {
      *sse = avg_sse_diff_32x64_msa(src_ptr, src_stride, ref_ptr, ref_stride,
                                    sec_pred, &diff);
    }
  }

  return VARIANCE_32Wx64H(*sse, diff);
}

#define VPX_SUB_PIXEL_AVG_VARIANCE64XHEIGHT_MSA(ht)                           \
  uint32_t vpx_sub_pixel_avg_variance64x##ht##_msa(                           \
      const uint8_t *src_ptr, int32_t src_stride, int32_t x_offset,           \
      int32_t y_offset, const uint8_t *ref_ptr, int32_t ref_stride,           \
      uint32_t *sse, const uint8_t *sec_pred) {                               \
    int32_t diff;                                                             \
    const uint8_t *h_filter = bilinear_filters_msa[x_offset];                 \
    const uint8_t *v_filter = bilinear_filters_msa[y_offset];                 \
                                                                              \
    if (y_offset) {                                                           \
      if (x_offset) {                                                         \
        *sse = sub_pixel_avg_sse_diff_64width_hv_msa(                         \
            src_ptr, src_stride, ref_ptr, ref_stride, sec_pred, h_filter,     \
            v_filter, ht, &diff);                                             \
      } else {                                                                \
        *sse = sub_pixel_avg_sse_diff_64width_v_msa(                          \
            src_ptr, src_stride, ref_ptr, ref_stride, sec_pred, v_filter, ht, \
            &diff);                                                           \
      }                                                                       \
    } else {                                                                  \
      if (x_offset) {                                                         \
        *sse = sub_pixel_avg_sse_diff_64width_h_msa(                          \
            src_ptr, src_stride, ref_ptr, ref_stride, sec_pred, h_filter, ht, \
            &diff);                                                           \
      } else {                                                                \
        *sse = avg_sse_diff_64x##ht##_msa(src_ptr, src_stride, ref_ptr,       \
                                          ref_stride, sec_pred, &diff);       \
      }                                                                       \
    }                                                                         \
                                                                              \
    return VARIANCE_64Wx##ht##H(*sse, diff);                                  \
  }

VPX_SUB_PIXEL_AVG_VARIANCE64XHEIGHT_MSA(32);
VPX_SUB_PIXEL_AVG_VARIANCE64XHEIGHT_MSA(64);
