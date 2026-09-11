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
#include "vpx_util/loongson_intrinsics.h"

static void avg_width4_lsx(const uint8_t *src, int32_t src_stride, uint8_t *dst,
                           int32_t dst_stride, int32_t height) {
  int32_t cnt;
  __m128i src0, src1;
  __m128i dst0, dst1;

  int32_t src_stride2 = src_stride << 1;

  if ((height % 2) == 0) {
    for (cnt = (height / 2); cnt--;) {
      src0 = __lsx_vld(src, 0);
      src1 = __lsx_vldx(src, src_stride);
      src += src_stride2;

      dst0 = __lsx_vld(dst, 0);
      dst1 = __lsx_vldx(dst, dst_stride);
      DUP2_ARG2(__lsx_vavgr_bu, src0, dst0, src1, dst1, dst0, dst1);

      __lsx_vstelm_w(dst0, dst, 0, 0);
      dst += dst_stride;
      __lsx_vstelm_w(dst1, dst, 0, 0);
      dst += dst_stride;
    }
  }
}

static void avg_width8_lsx(const uint8_t *src, int32_t src_stride, uint8_t *dst,
                           int32_t dst_stride, int32_t height) {
  int32_t cnt = (height / 4);
  __m128i src0, src1, src2, src3;
  __m128i dst0, dst1, dst2, dst3;

  int32_t src_stride2 = src_stride << 1;
  int32_t src_stride3 = src_stride2 + src_stride;
  int32_t src_stride4 = src_stride2 << 1;

  int32_t dst_stride2 = dst_stride << 1;
  int32_t dst_stride3 = dst_stride2 + dst_stride;

  for (; cnt--;) {
    src0 = __lsx_vld(src, 0);
    DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride2, src1, src2);
    src3 = __lsx_vldx(src, src_stride3);
    src += src_stride4;

    dst0 = __lsx_vld(dst, 0);
    DUP2_ARG2(__lsx_vldx, dst, dst_stride, dst, dst_stride2, dst1, dst2);
    dst3 = __lsx_vldx(dst, dst_stride3);

    DUP4_ARG2(__lsx_vavgr_bu, src0, dst0, src1, dst1, src2, dst2, src3, dst3,
              dst0, dst1, dst2, dst3);

    __lsx_vstelm_d(dst0, dst, 0, 0);
    dst += dst_stride;
    __lsx_vstelm_d(dst1, dst, 0, 0);
    dst += dst_stride;
    __lsx_vstelm_d(dst2, dst, 0, 0);
    dst += dst_stride;
    __lsx_vstelm_d(dst3, dst, 0, 0);
    dst += dst_stride;
  }
}

static void avg_width16_lsx(const uint8_t *src, int32_t src_stride,
                            uint8_t *dst, int32_t dst_stride, int32_t height) {
  int32_t cnt = (height / 8);
  __m128i src0, src1, src2, src3, src4, src5, src6, src7;
  __m128i dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7;

  int32_t src_stride2 = src_stride << 1;
  int32_t src_stride3 = src_stride2 + src_stride;
  int32_t src_stride4 = src_stride2 << 1;

  int32_t dst_stride2 = dst_stride << 1;
  int32_t dst_stride3 = dst_stride2 + dst_stride;
  int32_t dst_stride4 = dst_stride2 << 1;

  for (; cnt--;) {
    src0 = __lsx_vld(src, 0);
    DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride2, src1, src2);
    src3 = __lsx_vldx(src, src_stride3);
    src += src_stride4;
    src4 = __lsx_vld(src, 0);
    DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride2, src5, src6);
    src7 = __lsx_vldx(src, src_stride3);
    src += src_stride4;

    dst0 = __lsx_vld(dst, 0);
    DUP2_ARG2(__lsx_vldx, dst, dst_stride, dst, dst_stride2, dst1, dst2);
    dst3 = __lsx_vldx(dst, dst_stride3);
    dst += dst_stride4;
    dst4 = __lsx_vld(dst, 0);
    DUP2_ARG2(__lsx_vldx, dst, dst_stride, dst, dst_stride2, dst5, dst6);
    dst7 = __lsx_vldx(dst, dst_stride3);
    dst -= dst_stride4;

    DUP4_ARG2(__lsx_vavgr_bu, src0, dst0, src1, dst1, src2, dst2, src3, dst3,
              dst0, dst1, dst2, dst3);
    DUP4_ARG2(__lsx_vavgr_bu, src4, dst4, src5, dst5, src6, dst6, src7, dst7,
              dst4, dst5, dst6, dst7);

    __lsx_vst(dst0, dst, 0);
    __lsx_vstx(dst1, dst, dst_stride);
    __lsx_vstx(dst2, dst, dst_stride2);
    __lsx_vstx(dst3, dst, dst_stride3);
    dst += dst_stride4;
    __lsx_vst(dst4, dst, 0);
    __lsx_vstx(dst5, dst, dst_stride);
    __lsx_vstx(dst6, dst, dst_stride2);
    __lsx_vstx(dst7, dst, dst_stride3);
    dst += dst_stride4;
  }
}

static void avg_width32_lsx(const uint8_t *src, int32_t src_stride,
                            uint8_t *dst, int32_t dst_stride, int32_t height) {
  int32_t cnt = (height / 8);
  __m128i src0, src1, src2, src3, src4, src5, src6, src7;
  __m128i src8, src9, src10, src11, src12, src13, src14, src15;
  __m128i dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7;
  __m128i dst8, dst9, dst10, dst11, dst12, dst13, dst14, dst15;

  int32_t src_stride2 = src_stride << 1;
  int32_t src_stride3 = src_stride2 + src_stride;
  int32_t src_stride4 = src_stride2 << 1;

  int32_t dst_stride2 = dst_stride << 1;
  int32_t dst_stride3 = dst_stride2 + dst_stride;
  int32_t dst_stride4 = dst_stride2 << 1;

  for (; cnt--;) {
    uint8_t *dst_tmp = dst;
    uint8_t *dst_tmp1 = dst_tmp + 16;
    uint8_t *src_tmp = src + 16;

    src0 = __lsx_vld(src, 0);
    DUP2_ARG2(__lsx_vld, src, 0, src_tmp, 0, src0, src1);
    DUP4_ARG2(__lsx_vldx, src, src_stride, src_tmp, src_stride, src,
              src_stride2, src_tmp, src_stride2, src2, src3, src4, src5);
    DUP2_ARG2(__lsx_vldx, src, src_stride3, src_tmp, src_stride3, src6, src7);
    src += src_stride4;

    DUP2_ARG2(__lsx_vld, dst_tmp, 0, dst_tmp1, 0, dst0, dst1);
    DUP4_ARG2(__lsx_vldx, dst_tmp, dst_stride, dst_tmp1, dst_stride, dst_tmp,
              dst_stride2, dst_tmp1, dst_stride2, dst2, dst3, dst4, dst5);
    DUP2_ARG2(__lsx_vldx, dst_tmp, dst_stride3, dst_tmp1, dst_stride3, dst6,
              dst7);
    dst_tmp += dst_stride4;
    dst_tmp1 += dst_stride4;

    src_tmp = src + 16;
    DUP2_ARG2(__lsx_vld, src, 0, src_tmp, 0, src8, src9);
    DUP4_ARG2(__lsx_vldx, src, src_stride, src_tmp, src_stride, src,
              src_stride2, src_tmp, src_stride2, src10, src11, src12, src13);
    DUP2_ARG2(__lsx_vldx, src, src_stride3, src_tmp, src_stride3, src14, src15);
    src += src_stride4;

    DUP2_ARG2(__lsx_vld, dst_tmp, 0, dst_tmp1, 0, dst8, dst9);
    DUP4_ARG2(__lsx_vldx, dst_tmp, dst_stride, dst_tmp1, dst_stride, dst_tmp,
              dst_stride2, dst_tmp1, dst_stride2, dst10, dst11, dst12, dst13);
    DUP2_ARG2(__lsx_vldx, dst_tmp, dst_stride3, dst_tmp1, dst_stride3, dst14,
              dst15);
    DUP4_ARG2(__lsx_vavgr_bu, src0, dst0, src1, dst1, src2, dst2, src3, dst3,
              dst0, dst1, dst2, dst3);
    DUP4_ARG2(__lsx_vavgr_bu, src4, dst4, src5, dst5, src6, dst6, src7, dst7,
              dst4, dst5, dst6, dst7);
    DUP4_ARG2(__lsx_vavgr_bu, src8, dst8, src9, dst9, src10, dst10, src11,
              dst11, dst8, dst9, dst10, dst11);
    DUP4_ARG2(__lsx_vavgr_bu, src12, dst12, src13, dst13, src14, dst14, src15,
              dst15, dst12, dst13, dst14, dst15);

    dst_tmp = dst + 16;
    __lsx_vst(dst0, dst, 0);
    __lsx_vstx(dst2, dst, dst_stride);
    __lsx_vstx(dst4, dst, dst_stride2);
    __lsx_vstx(dst6, dst, dst_stride3);
    __lsx_vst(dst1, dst_tmp, 0);
    __lsx_vstx(dst3, dst_tmp, dst_stride);
    __lsx_vstx(dst5, dst_tmp, dst_stride2);
    __lsx_vstx(dst7, dst_tmp, dst_stride3);
    dst += dst_stride4;

    __lsx_vst(dst8, dst, 0);
    __lsx_vstx(dst10, dst, dst_stride);
    __lsx_vstx(dst12, dst, dst_stride2);
    __lsx_vstx(dst14, dst, dst_stride3);
    __lsx_vst(dst9, dst_tmp1, 0);
    __lsx_vstx(dst11, dst_tmp1, dst_stride);
    __lsx_vstx(dst13, dst_tmp1, dst_stride2);
    __lsx_vstx(dst15, dst_tmp1, dst_stride3);
    dst += dst_stride4;
  }
}

static void avg_width64_lsx(const uint8_t *src, int32_t src_stride,
                            uint8_t *dst, int32_t dst_stride, int32_t height) {
  int32_t cnt = (height / 4);
  uint8_t *dst_tmp = dst;

  __m128i src0, src1, src2, src3, src4, src5, src6, src7;
  __m128i src8, src9, src10, src11, src12, src13, src14, src15;
  __m128i dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7;
  __m128i dst8, dst9, dst10, dst11, dst12, dst13, dst14, dst15;

  for (; cnt--;) {
    DUP4_ARG2(__lsx_vld, src, 0, src, 16, src, 32, src, 48, src0, src1, src2,
              src3);
    src += src_stride;
    DUP4_ARG2(__lsx_vld, src, 0, src, 16, src, 32, src, 48, src4, src5, src6,
              src7);
    src += src_stride;
    DUP4_ARG2(__lsx_vld, src, 0, src, 16, src, 32, src, 48, src8, src9, src10,
              src11);
    src += src_stride;
    DUP4_ARG2(__lsx_vld, src, 0, src, 16, src, 32, src, 48, src12, src13, src14,
              src15);
    src += src_stride;

    DUP4_ARG2(__lsx_vld, dst_tmp, 0, dst_tmp, 16, dst_tmp, 32, dst_tmp, 48,
              dst0, dst1, dst2, dst3);
    dst_tmp += dst_stride;
    DUP4_ARG2(__lsx_vld, dst_tmp, 0, dst_tmp, 16, dst_tmp, 32, dst_tmp, 48,
              dst4, dst5, dst6, dst7);
    dst_tmp += dst_stride;
    DUP4_ARG2(__lsx_vld, dst_tmp, 0, dst_tmp, 16, dst_tmp, 32, dst_tmp, 48,
              dst8, dst9, dst10, dst11);
    dst_tmp += dst_stride;
    DUP4_ARG2(__lsx_vld, dst_tmp, 0, dst_tmp, 16, dst_tmp, 32, dst_tmp, 48,
              dst12, dst13, dst14, dst15);
    dst_tmp += dst_stride;

    DUP4_ARG2(__lsx_vavgr_bu, src0, dst0, src1, dst1, src2, dst2, src3, dst3,
              dst0, dst1, dst2, dst3);
    DUP4_ARG2(__lsx_vavgr_bu, src4, dst4, src5, dst5, src6, dst6, src7, dst7,
              dst4, dst5, dst6, dst7);
    DUP4_ARG2(__lsx_vavgr_bu, src8, dst8, src9, dst9, src10, dst10, src11,
              dst11, dst8, dst9, dst10, dst11);
    DUP4_ARG2(__lsx_vavgr_bu, src12, dst12, src13, dst13, src14, dst14, src15,
              dst15, dst12, dst13, dst14, dst15);

    __lsx_vst(dst0, dst, 0);
    __lsx_vst(dst1, dst, 16);
    __lsx_vst(dst2, dst, 32);
    __lsx_vst(dst3, dst, 48);
    dst += dst_stride;
    __lsx_vst(dst4, dst, 0);
    __lsx_vst(dst5, dst, 16);
    __lsx_vst(dst6, dst, 32);
    __lsx_vst(dst7, dst, 48);
    dst += dst_stride;
    __lsx_vst(dst8, dst, 0);
    __lsx_vst(dst9, dst, 16);
    __lsx_vst(dst10, dst, 32);
    __lsx_vst(dst11, dst, 48);
    dst += dst_stride;
    __lsx_vst(dst12, dst, 0);
    __lsx_vst(dst13, dst, 16);
    __lsx_vst(dst14, dst, 32);
    __lsx_vst(dst15, dst, 48);
    dst += dst_stride;
  }
}

void vpx_convolve_avg_lsx(const uint8_t *src, ptrdiff_t src_stride,
                          uint8_t *dst, ptrdiff_t dst_stride,
                          const InterpKernel *filter, int x0_q4,
                          int32_t x_step_q4, int y0_q4, int32_t y_step_q4,
                          int32_t w, int32_t h) {
  (void)filter;
  (void)x0_q4;
  (void)x_step_q4;
  (void)y0_q4;
  (void)y_step_q4;
  switch (w) {
    case 4: {
      avg_width4_lsx(src, src_stride, dst, dst_stride, h);
      break;
    }

    case 8: {
      avg_width8_lsx(src, src_stride, dst, dst_stride, h);
      break;
    }
    case 16: {
      avg_width16_lsx(src, src_stride, dst, dst_stride, h);
      break;
    }
    case 32: {
      avg_width32_lsx(src, src_stride, dst, dst_stride, h);
      break;
    }
    case 64: {
      avg_width64_lsx(src, src_stride, dst, dst_stride, h);
      break;
    }
    default: {
      int32_t lp, cnt;
      for (cnt = h; cnt--;) {
        for (lp = 0; lp < w; ++lp) {
          dst[lp] = (((dst[lp] + src[lp]) + 1) >> 1);
        }
        src += src_stride;
        dst += dst_stride;
      }
      break;
    }
  }
}
