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
#include "vpx_dsp/mips/macros_msa.h"

#define CALC_MSE_B(src, ref, var)                                   \
  {                                                                 \
    v16u8 src_l0_m, src_l1_m;                                       \
    v8i16 res_l0_m, res_l1_m;                                       \
                                                                    \
    ILVRL_B2_UB(src, ref, src_l0_m, src_l1_m);                      \
    HSUB_UB2_SH(src_l0_m, src_l1_m, res_l0_m, res_l1_m);            \
    DPADD_SH2_SW(res_l0_m, res_l1_m, res_l0_m, res_l1_m, var, var); \
  }

#define CALC_MSE_AVG_B(src, ref, var, sub)                          \
  {                                                                 \
    v16u8 src_l0_m, src_l1_m;                                       \
    v8i16 res_l0_m, res_l1_m;                                       \
                                                                    \
    ILVRL_B2_UB(src, ref, src_l0_m, src_l1_m);                      \
    HSUB_UB2_SH(src_l0_m, src_l1_m, res_l0_m, res_l1_m);            \
    DPADD_SH2_SW(res_l0_m, res_l1_m, res_l0_m, res_l1_m, var, var); \
                                                                    \
    sub += res_l0_m + res_l1_m;                                     \
  }

#define VARIANCE_WxH(sse, diff, shift) \
  (sse) - (((uint32_t)(diff) * (diff)) >> (shift))

#define VARIANCE_LARGE_WxH(sse, diff, shift) \
  (sse) - (((int64_t)(diff) * (diff)) >> (shift))

static uint32_t sse_diff_4width_msa(const uint8_t *src_ptr, int32_t src_stride,
                                    const uint8_t *ref_ptr, int32_t ref_stride,
                                    int32_t height, int32_t *diff) {
  uint32_t src0, src1, src2, src3;
  uint32_t ref0, ref1, ref2, ref3;
  int32_t ht_cnt;
  v16u8 src = { 0 };
  v16u8 ref = { 0 };
  v8i16 avg = { 0 };
  v4i32 vec, var = { 0 };

  for (ht_cnt = (height >> 2); ht_cnt--;) {
    LW4(src_ptr, src_stride, src0, src1, src2, src3);
    src_ptr += (4 * src_stride);
    LW4(ref_ptr, ref_stride, ref0, ref1, ref2, ref3);
    ref_ptr += (4 * ref_stride);

    INSERT_W4_UB(src0, src1, src2, src3, src);
    INSERT_W4_UB(ref0, ref1, ref2, ref3, ref);
    CALC_MSE_AVG_B(src, ref, var, avg);
  }

  vec = __msa_hadd_s_w(avg, avg);
  *diff = HADD_SW_S32(vec);

  return HADD_SW_S32(var);
}

static uint32_t sse_diff_8width_msa(const uint8_t *src_ptr, int32_t src_stride,
                                    const uint8_t *ref_ptr, int32_t ref_stride,
                                    int32_t height, int32_t *diff) {
  int32_t ht_cnt;
  v16u8 src0, src1, src2, src3;
  v16u8 ref0, ref1, ref2, ref3;
  v8i16 avg = { 0 };
  v4i32 vec, var = { 0 };

  for (ht_cnt = (height >> 2); ht_cnt--;) {
    LD_UB4(src_ptr, src_stride, src0, src1, src2, src3);
    src_ptr += (4 * src_stride);
    LD_UB4(ref_ptr, ref_stride, ref0, ref1, ref2, ref3);
    ref_ptr += (4 * ref_stride);

    PCKEV_D4_UB(src1, src0, src3, src2, ref1, ref0, ref3, ref2, src0, src1,
                ref0, ref1);
    CALC_MSE_AVG_B(src0, ref0, var, avg);
    CALC_MSE_AVG_B(src1, ref1, var, avg);
  }

  vec = __msa_hadd_s_w(avg, avg);
  *diff = HADD_SW_S32(vec);

  return HADD_SW_S32(var);
}

static uint32_t sse_diff_16width_msa(const uint8_t *src_ptr, int32_t src_stride,
                                     const uint8_t *ref_ptr, int32_t ref_stride,
                                     int32_t height, int32_t *diff) {
  int32_t ht_cnt;
  v16u8 src, ref;
  v8i16 avg = { 0 };
  v4i32 vec, var = { 0 };

  for (ht_cnt = (height >> 2); ht_cnt--;) {
    src = LD_UB(src_ptr);
    src_ptr += src_stride;
    ref = LD_UB(ref_ptr);
    ref_ptr += ref_stride;
    CALC_MSE_AVG_B(src, ref, var, avg);

    src = LD_UB(src_ptr);
    src_ptr += src_stride;
    ref = LD_UB(ref_ptr);
    ref_ptr += ref_stride;
    CALC_MSE_AVG_B(src, ref, var, avg);

    src = LD_UB(src_ptr);
    src_ptr += src_stride;
    ref = LD_UB(ref_ptr);
    ref_ptr += ref_stride;
    CALC_MSE_AVG_B(src, ref, var, avg);

    src = LD_UB(src_ptr);
    src_ptr += src_stride;
    ref = LD_UB(ref_ptr);
    ref_ptr += ref_stride;
    CALC_MSE_AVG_B(src, ref, var, avg);
  }

  vec = __msa_hadd_s_w(avg, avg);
  *diff = HADD_SW_S32(vec);

  return HADD_SW_S32(var);
}

static uint32_t sse_diff_32width_msa(const uint8_t *src_ptr, int32_t src_stride,
                                     const uint8_t *ref_ptr, int32_t ref_stride,
                                     int32_t height, int32_t *diff) {
  int32_t ht_cnt;
  v16u8 src0, src1, ref0, ref1;
  v8i16 avg = { 0 };
  v4i32 vec, var = { 0 };

  for (ht_cnt = (height >> 2); ht_cnt--;) {
    LD_UB2(src_ptr, 16, src0, src1);
    src_ptr += src_stride;
    LD_UB2(ref_ptr, 16, ref0, ref1);
    ref_ptr += ref_stride;
    CALC_MSE_AVG_B(src0, ref0, var, avg);
    CALC_MSE_AVG_B(src1, ref1, var, avg);

    LD_UB2(src_ptr, 16, src0, src1);
    src_ptr += src_stride;
    LD_UB2(ref_ptr, 16, ref0, ref1);
    ref_ptr += ref_stride;
    CALC_MSE_AVG_B(src0, ref0, var, avg);
    CALC_MSE_AVG_B(src1, ref1, var, avg);

    LD_UB2(src_ptr, 16, src0, src1);
    src_ptr += src_stride;
    LD_UB2(ref_ptr, 16, ref0, ref1);
    ref_ptr += ref_stride;
    CALC_MSE_AVG_B(src0, ref0, var, avg);
    CALC_MSE_AVG_B(src1, ref1, var, avg);

    LD_UB2(src_ptr, 16, src0, src1);
    src_ptr += src_stride;
    LD_UB2(ref_ptr, 16, ref0, ref1);
    ref_ptr += ref_stride;
    CALC_MSE_AVG_B(src0, ref0, var, avg);
    CALC_MSE_AVG_B(src1, ref1, var, avg);
  }

  vec = __msa_hadd_s_w(avg, avg);
  *diff = HADD_SW_S32(vec);

  return HADD_SW_S32(var);
}

static uint32_t sse_diff_32x64_msa(const uint8_t *src_ptr, int32_t src_stride,
                                   const uint8_t *ref_ptr, int32_t ref_stride,
                                   int32_t *diff) {
  int32_t ht_cnt;
  v16u8 src0, src1, ref0, ref1;
  v8i16 avg0 = { 0 };
  v8i16 avg1 = { 0 };
  v4i32 vec, var = { 0 };

  for (ht_cnt = 16; ht_cnt--;) {
    LD_UB2(src_ptr, 16, src0, src1);
    src_ptr += src_stride;
    LD_UB2(ref_ptr, 16, ref0, ref1);
    ref_ptr += ref_stride;
    CALC_MSE_AVG_B(src0, ref0, var, avg0);
    CALC_MSE_AVG_B(src1, ref1, var, avg1);

    LD_UB2(src_ptr, 16, src0, src1);
    src_ptr += src_stride;
    LD_UB2(ref_ptr, 16, ref0, ref1);
    ref_ptr += ref_stride;
    CALC_MSE_AVG_B(src0, ref0, var, avg0);
    CALC_MSE_AVG_B(src1, ref1, var, avg1);

    LD_UB2(src_ptr, 16, src0, src1);
    src_ptr += src_stride;
    LD_UB2(ref_ptr, 16, ref0, ref1);
    ref_ptr += ref_stride;
    CALC_MSE_AVG_B(src0, ref0, var, avg0);
    CALC_MSE_AVG_B(src1, ref1, var, avg1);

    LD_UB2(src_ptr, 16, src0, src1);
    src_ptr += src_stride;
    LD_UB2(ref_ptr, 16, ref0, ref1);
    ref_ptr += ref_stride;
    CALC_MSE_AVG_B(src0, ref0, var, avg0);
    CALC_MSE_AVG_B(src1, ref1, var, avg1);
  }

  vec = __msa_hadd_s_w(avg0, avg0);
  vec += __msa_hadd_s_w(avg1, avg1);
  *diff = HADD_SW_S32(vec);

  return HADD_SW_S32(var);
}

static uint32_t sse_diff_64x32_msa(const uint8_t *src_ptr, int32_t src_stride,
                                   const uint8_t *ref_ptr, int32_t ref_stride,
                                   int32_t *diff) {
  int32_t ht_cnt;
  v16u8 src0, src1, src2, src3;
  v16u8 ref0, ref1, ref2, ref3;
  v8i16 avg0 = { 0 };
  v8i16 avg1 = { 0 };
  v4i32 vec, var = { 0 };

  for (ht_cnt = 16; ht_cnt--;) {
    LD_UB4(src_ptr, 16, src0, src1, src2, src3);
    src_ptr += src_stride;
    LD_UB4(ref_ptr, 16, ref0, ref1, ref2, ref3);
    ref_ptr += ref_stride;
    CALC_MSE_AVG_B(src0, ref0, var, avg0);
    CALC_MSE_AVG_B(src2, ref2, var, avg0);
    CALC_MSE_AVG_B(src1, ref1, var, avg1);
    CALC_MSE_AVG_B(src3, ref3, var, avg1);

    LD_UB4(src_ptr, 16, src0, src1, src2, src3);
    src_ptr += src_stride;
    LD_UB4(ref_ptr, 16, ref0, ref1, ref2, ref3);
    ref_ptr += ref_stride;
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

static uint32_t sse_diff_64x64_msa(const uint8_t *src_ptr, int32_t src_stride,
                                   const uint8_t *ref_ptr, int32_t ref_stride,
                                   int32_t *diff) {
  int32_t ht_cnt;
  v16u8 src0, src1, src2, src3;
  v16u8 ref0, ref1, ref2, ref3;
  v8i16 avg0 = { 0 };
  v8i16 avg1 = { 0 };
  v8i16 avg2 = { 0 };
  v8i16 avg3 = { 0 };
  v4i32 vec, var = { 0 };

  for (ht_cnt = 32; ht_cnt--;) {
    LD_UB4(src_ptr, 16, src0, src1, src2, src3);
    src_ptr += src_stride;
    LD_UB4(ref_ptr, 16, ref0, ref1, ref2, ref3);
    ref_ptr += ref_stride;

    CALC_MSE_AVG_B(src0, ref0, var, avg0);
    CALC_MSE_AVG_B(src1, ref1, var, avg1);
    CALC_MSE_AVG_B(src2, ref2, var, avg2);
    CALC_MSE_AVG_B(src3, ref3, var, avg3);
    LD_UB4(src_ptr, 16, src0, src1, src2, src3);
    src_ptr += src_stride;
    LD_UB4(ref_ptr, 16, ref0, ref1, ref2, ref3);
    ref_ptr += ref_stride;
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

static uint32_t get_mb_ss_msa(const int16_t *src) {
  uint32_t sum, cnt;
  v8i16 src0, src1, src2, src3;
  v4i32 src0_l, src1_l, src2_l, src3_l;
  v4i32 src0_r, src1_r, src2_r, src3_r;
  v2i64 sq_src_l = { 0 };
  v2i64 sq_src_r = { 0 };

  for (cnt = 8; cnt--;) {
    LD_SH4(src, 8, src0, src1, src2, src3);
    src += 4 * 8;

    UNPCK_SH_SW(src0, src0_l, src0_r);
    UNPCK_SH_SW(src1, src1_l, src1_r);
    UNPCK_SH_SW(src2, src2_l, src2_r);
    UNPCK_SH_SW(src3, src3_l, src3_r);

    DPADD_SD2_SD(src0_l, src0_r, sq_src_l, sq_src_r);
    DPADD_SD2_SD(src1_l, src1_r, sq_src_l, sq_src_r);
    DPADD_SD2_SD(src2_l, src2_r, sq_src_l, sq_src_r);
    DPADD_SD2_SD(src3_l, src3_r, sq_src_l, sq_src_r);
  }

  sq_src_l += __msa_splati_d(sq_src_l, 1);
  sq_src_r += __msa_splati_d(sq_src_r, 1);

  sum = __msa_copy_s_d(sq_src_l, 0);
  sum += __msa_copy_s_d(sq_src_r, 0);

  return sum;
}

static uint32_t sse_4width_msa(const uint8_t *src_ptr, int32_t src_stride,
                               const uint8_t *ref_ptr, int32_t ref_stride,
                               int32_t height) {
  int32_t ht_cnt;
  uint32_t src0, src1, src2, src3;
  uint32_t ref0, ref1, ref2, ref3;
  v16u8 src = { 0 };
  v16u8 ref = { 0 };
  v4i32 var = { 0 };

  for (ht_cnt = (height >> 2); ht_cnt--;) {
    LW4(src_ptr, src_stride, src0, src1, src2, src3);
    src_ptr += (4 * src_stride);
    LW4(ref_ptr, ref_stride, ref0, ref1, ref2, ref3);
    ref_ptr += (4 * ref_stride);

    INSERT_W4_UB(src0, src1, src2, src3, src);
    INSERT_W4_UB(ref0, ref1, ref2, ref3, ref);
    CALC_MSE_B(src, ref, var);
  }

  return HADD_SW_S32(var);
}

static uint32_t sse_8width_msa(const uint8_t *src_ptr, int32_t src_stride,
                               const uint8_t *ref_ptr, int32_t ref_stride,
                               int32_t height) {
  int32_t ht_cnt;
  v16u8 src0, src1, src2, src3;
  v16u8 ref0, ref1, ref2, ref3;
  v4i32 var = { 0 };

  for (ht_cnt = (height >> 2); ht_cnt--;) {
    LD_UB4(src_ptr, src_stride, src0, src1, src2, src3);
    src_ptr += (4 * src_stride);
    LD_UB4(ref_ptr, ref_stride, ref0, ref1, ref2, ref3);
    ref_ptr += (4 * ref_stride);

    PCKEV_D4_UB(src1, src0, src3, src2, ref1, ref0, ref3, ref2, src0, src1,
                ref0, ref1);
    CALC_MSE_B(src0, ref0, var);
    CALC_MSE_B(src1, ref1, var);
  }

  return HADD_SW_S32(var);
}

static uint32_t sse_16width_msa(const uint8_t *src_ptr, int32_t src_stride,
                                const uint8_t *ref_ptr, int32_t ref_stride,
                                int32_t height) {
  int32_t ht_cnt;
  v16u8 src, ref;
  v4i32 var = { 0 };

  for (ht_cnt = (height >> 2); ht_cnt--;) {
    src = LD_UB(src_ptr);
    src_ptr += src_stride;
    ref = LD_UB(ref_ptr);
    ref_ptr += ref_stride;
    CALC_MSE_B(src, ref, var);

    src = LD_UB(src_ptr);
    src_ptr += src_stride;
    ref = LD_UB(ref_ptr);
    ref_ptr += ref_stride;
    CALC_MSE_B(src, ref, var);

    src = LD_UB(src_ptr);
    src_ptr += src_stride;
    ref = LD_UB(ref_ptr);
    ref_ptr += ref_stride;
    CALC_MSE_B(src, ref, var);

    src = LD_UB(src_ptr);
    src_ptr += src_stride;
    ref = LD_UB(ref_ptr);
    ref_ptr += ref_stride;
    CALC_MSE_B(src, ref, var);
  }

  return HADD_SW_S32(var);
}

static uint32_t sse_32width_msa(const uint8_t *src_ptr, int32_t src_stride,
                                const uint8_t *ref_ptr, int32_t ref_stride,
                                int32_t height) {
  int32_t ht_cnt;
  v16u8 src0, src1, ref0, ref1;
  v4i32 var = { 0 };

  for (ht_cnt = (height >> 2); ht_cnt--;) {
    LD_UB2(src_ptr, 16, src0, src1);
    src_ptr += src_stride;
    LD_UB2(ref_ptr, 16, ref0, ref1);
    ref_ptr += ref_stride;
    CALC_MSE_B(src0, ref0, var);
    CALC_MSE_B(src1, ref1, var);

    LD_UB2(src_ptr, 16, src0, src1);
    src_ptr += src_stride;
    LD_UB2(ref_ptr, 16, ref0, ref1);
    ref_ptr += ref_stride;
    CALC_MSE_B(src0, ref0, var);
    CALC_MSE_B(src1, ref1, var);

    LD_UB2(src_ptr, 16, src0, src1);
    src_ptr += src_stride;
    LD_UB2(ref_ptr, 16, ref0, ref1);
    ref_ptr += ref_stride;
    CALC_MSE_B(src0, ref0, var);
    CALC_MSE_B(src1, ref1, var);

    LD_UB2(src_ptr, 16, src0, src1);
    src_ptr += src_stride;
    LD_UB2(ref_ptr, 16, ref0, ref1);
    ref_ptr += ref_stride;
    CALC_MSE_B(src0, ref0, var);
    CALC_MSE_B(src1, ref1, var);
  }

  return HADD_SW_S32(var);
}

static uint32_t sse_64width_msa(const uint8_t *src_ptr, int32_t src_stride,
                                const uint8_t *ref_ptr, int32_t ref_stride,
                                int32_t height) {
  int32_t ht_cnt;
  v16u8 src0, src1, src2, src3;
  v16u8 ref0, ref1, ref2, ref3;
  v4i32 var = { 0 };

  for (ht_cnt = height >> 1; ht_cnt--;) {
    LD_UB4(src_ptr, 16, src0, src1, src2, src3);
    src_ptr += src_stride;
    LD_UB4(ref_ptr, 16, ref0, ref1, ref2, ref3);
    ref_ptr += ref_stride;
    CALC_MSE_B(src0, ref0, var);
    CALC_MSE_B(src2, ref2, var);
    CALC_MSE_B(src1, ref1, var);
    CALC_MSE_B(src3, ref3, var);

    LD_UB4(src_ptr, 16, src0, src1, src2, src3);
    src_ptr += src_stride;
    LD_UB4(ref_ptr, 16, ref0, ref1, ref2, ref3);
    ref_ptr += ref_stride;
    CALC_MSE_B(src0, ref0, var);
    CALC_MSE_B(src2, ref2, var);
    CALC_MSE_B(src1, ref1, var);
    CALC_MSE_B(src3, ref3, var);
  }

  return HADD_SW_S32(var);
}

uint32_t vpx_get4x4sse_cs_msa(const uint8_t *src_ptr, int32_t src_stride,
                              const uint8_t *ref_ptr, int32_t ref_stride) {
  uint32_t src0, src1, src2, src3;
  uint32_t ref0, ref1, ref2, ref3;
  v16i8 src = { 0 };
  v16i8 ref = { 0 };
  v4i32 err0 = { 0 };

  LW4(src_ptr, src_stride, src0, src1, src2, src3);
  LW4(ref_ptr, ref_stride, ref0, ref1, ref2, ref3);
  INSERT_W4_SB(src0, src1, src2, src3, src);
  INSERT_W4_SB(ref0, ref1, ref2, ref3, ref);
  CALC_MSE_B(src, ref, err0);

  return HADD_SW_S32(err0);
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

#define VPX_VARIANCE_WDXHT_MSA(wd, ht)                                         \
  uint32_t vpx_variance##wd##x##ht##_msa(                                      \
      const uint8_t *src, int32_t src_stride, const uint8_t *ref,              \
      int32_t ref_stride, uint32_t *sse) {                                     \
    int32_t diff;                                                              \
                                                                               \
    *sse =                                                                     \
        sse_diff_##wd##width_msa(src, src_stride, ref, ref_stride, ht, &diff); \
                                                                               \
    return VARIANCE_##wd##Wx##ht##H(*sse, diff);                               \
  }

VPX_VARIANCE_WDXHT_MSA(4, 4);
VPX_VARIANCE_WDXHT_MSA(4, 8);

VPX_VARIANCE_WDXHT_MSA(8, 4)
VPX_VARIANCE_WDXHT_MSA(8, 8)
VPX_VARIANCE_WDXHT_MSA(8, 16)

VPX_VARIANCE_WDXHT_MSA(16, 8)
VPX_VARIANCE_WDXHT_MSA(16, 16)
VPX_VARIANCE_WDXHT_MSA(16, 32)

VPX_VARIANCE_WDXHT_MSA(32, 16)
VPX_VARIANCE_WDXHT_MSA(32, 32)

uint32_t vpx_variance32x64_msa(const uint8_t *src, int32_t src_stride,
                               const uint8_t *ref, int32_t ref_stride,
                               uint32_t *sse) {
  int32_t diff;

  *sse = sse_diff_32x64_msa(src, src_stride, ref, ref_stride, &diff);

  return VARIANCE_32Wx64H(*sse, diff);
}

uint32_t vpx_variance64x32_msa(const uint8_t *src, int32_t src_stride,
                               const uint8_t *ref, int32_t ref_stride,
                               uint32_t *sse) {
  int32_t diff;

  *sse = sse_diff_64x32_msa(src, src_stride, ref, ref_stride, &diff);

  return VARIANCE_64Wx32H(*sse, diff);
}

uint32_t vpx_variance64x64_msa(const uint8_t *src, int32_t src_stride,
                               const uint8_t *ref, int32_t ref_stride,
                               uint32_t *sse) {
  int32_t diff;

  *sse = sse_diff_64x64_msa(src, src_stride, ref, ref_stride, &diff);

  return VARIANCE_64Wx64H(*sse, diff);
}

uint32_t vpx_mse8x8_msa(const uint8_t *src, int32_t src_stride,
                        const uint8_t *ref, int32_t ref_stride, uint32_t *sse) {
  *sse = sse_8width_msa(src, src_stride, ref, ref_stride, 8);

  return *sse;
}

uint32_t vpx_mse8x16_msa(const uint8_t *src, int32_t src_stride,
                         const uint8_t *ref, int32_t ref_stride,
                         uint32_t *sse) {
  *sse = sse_8width_msa(src, src_stride, ref, ref_stride, 16);

  return *sse;
}

uint32_t vpx_mse16x8_msa(const uint8_t *src, int32_t src_stride,
                         const uint8_t *ref, int32_t ref_stride,
                         uint32_t *sse) {
  *sse = sse_16width_msa(src, src_stride, ref, ref_stride, 8);

  return *sse;
}

uint32_t vpx_mse16x16_msa(const uint8_t *src, int32_t src_stride,
                          const uint8_t *ref, int32_t ref_stride,
                          uint32_t *sse) {
  *sse = sse_16width_msa(src, src_stride, ref, ref_stride, 16);

  return *sse;
}

void vpx_get8x8var_msa(const uint8_t *src, int32_t src_stride,
                       const uint8_t *ref, int32_t ref_stride, uint32_t *sse,
                       int32_t *sum) {
  *sse = sse_diff_8width_msa(src, src_stride, ref, ref_stride, 8, sum);
}

void vpx_get16x16var_msa(const uint8_t *src, int32_t src_stride,
                         const uint8_t *ref, int32_t ref_stride, uint32_t *sse,
                         int32_t *sum) {
  *sse = sse_diff_16width_msa(src, src_stride, ref, ref_stride, 16, sum);
}

uint32_t vpx_get_mb_ss_msa(const int16_t *src) { return get_mb_ss_msa(src); }
