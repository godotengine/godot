/*
 *  Copyright (c) 2022 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <string.h>
#include "./vpx_dsp_rtcd.h"
#include "vpx_util/loongson_intrinsics.h"

static void copy_width8_lsx(const uint8_t *src, int32_t src_stride,
                            uint8_t *dst, int32_t dst_stride, int32_t height) {
  int32_t cnt;
  __m128i src0, src1, src2, src3, src4, src5, src6, src7;
  int32_t src_stride2 = src_stride << 1;
  int32_t src_stride3 = src_stride2 + src_stride;
  int32_t src_stride4 = src_stride2 << 1;

  if ((height % 12) == 0) {
    for (cnt = (height / 12); cnt--;) {
      src0 = __lsx_vld(src, 0);
      DUP4_ARG2(__lsx_vldx, src, src_stride, src, src_stride2, src, src_stride3,
                src, src_stride4, src1, src2, src3, src4);
      src += src_stride4;
      DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride2, src5, src6);
      src += src_stride2;
      src7 = __lsx_vldx(src, src_stride);
      src += src_stride2;

      __lsx_vstelm_d(src0, dst, 0, 0);
      dst += dst_stride;
      __lsx_vstelm_d(src1, dst, 0, 0);
      dst += dst_stride;
      __lsx_vstelm_d(src2, dst, 0, 0);
      dst += dst_stride;
      __lsx_vstelm_d(src3, dst, 0, 0);
      dst += dst_stride;

      __lsx_vstelm_d(src4, dst, 0, 0);
      dst += dst_stride;
      __lsx_vstelm_d(src5, dst, 0, 0);
      dst += dst_stride;
      __lsx_vstelm_d(src6, dst, 0, 0);
      dst += dst_stride;
      __lsx_vstelm_d(src7, dst, 0, 0);
      dst += dst_stride;

      src0 = __lsx_vld(src, 0);
      DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride2, src1, src2);
      src3 = __lsx_vldx(src, src_stride3);
      src += src_stride4;

      __lsx_vstelm_d(src0, dst, 0, 0);
      dst += dst_stride;
      __lsx_vstelm_d(src1, dst, 0, 0);
      dst += dst_stride;
      __lsx_vstelm_d(src2, dst, 0, 0);
      dst += dst_stride;
      __lsx_vstelm_d(src3, dst, 0, 0);
      dst += dst_stride;
    }
  } else if ((height % 8) == 0) {
    for (cnt = height >> 3; cnt--;) {
      src0 = __lsx_vld(src, 0);
      DUP4_ARG2(__lsx_vldx, src, src_stride, src, src_stride2, src, src_stride3,
                src, src_stride4, src1, src2, src3, src4);
      src += src_stride4;
      DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride2, src5, src6);
      src += src_stride2;
      src7 = __lsx_vldx(src, src_stride);
      src += src_stride2;

      __lsx_vstelm_d(src0, dst, 0, 0);
      dst += dst_stride;
      __lsx_vstelm_d(src1, dst, 0, 0);
      dst += dst_stride;
      __lsx_vstelm_d(src2, dst, 0, 0);
      dst += dst_stride;
      __lsx_vstelm_d(src3, dst, 0, 0);
      dst += dst_stride;

      __lsx_vstelm_d(src4, dst, 0, 0);
      dst += dst_stride;
      __lsx_vstelm_d(src5, dst, 0, 0);
      dst += dst_stride;
      __lsx_vstelm_d(src6, dst, 0, 0);
      dst += dst_stride;
      __lsx_vstelm_d(src7, dst, 0, 0);
      dst += dst_stride;
    }
  } else if ((height % 4) == 0) {
    for (cnt = (height / 4); cnt--;) {
      src0 = __lsx_vld(src, 0);
      DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride2, src1, src2);
      src3 = __lsx_vldx(src, src_stride3);
      src += src_stride4;

      __lsx_vstelm_d(src0, dst, 0, 0);
      dst += dst_stride;
      __lsx_vstelm_d(src1, dst, 0, 0);
      dst += dst_stride;
      __lsx_vstelm_d(src2, dst, 0, 0);
      dst += dst_stride;
      __lsx_vstelm_d(src3, dst, 0, 0);
      dst += dst_stride;
    }
  } else if ((height % 2) == 0) {
    for (cnt = (height / 2); cnt--;) {
      src0 = __lsx_vld(src, 0);
      src1 = __lsx_vldx(src, src_stride);
      src += src_stride2;

      __lsx_vstelm_d(src0, dst, 0, 0);
      dst += dst_stride;
      __lsx_vstelm_d(src1, dst, 0, 0);
      dst += dst_stride;
    }
  }
}

static void copy_16multx8mult_lsx(const uint8_t *src, int32_t src_stride,
                                  uint8_t *dst, int32_t dst_stride,
                                  int32_t height, int32_t width) {
  int32_t cnt, loop_cnt;
  uint8_t *src_tmp;
  uint8_t *dst_tmp;
  __m128i src0, src1, src2, src3, src4, src5, src6, src7;
  int32_t src_stride2 = src_stride << 1;
  int32_t src_stride3 = src_stride2 + src_stride;
  int32_t src_stride4 = src_stride2 << 1;

  for (cnt = (width >> 4); cnt--;) {
    src_tmp = (uint8_t *)src;
    dst_tmp = dst;

    for (loop_cnt = (height >> 3); loop_cnt--;) {
      src0 = __lsx_vld(src_tmp, 0);
      DUP4_ARG2(__lsx_vldx, src_tmp, src_stride, src_tmp, src_stride2, src_tmp,
                src_stride3, src_tmp, src_stride4, src1, src2, src3, src4);
      src_tmp += src_stride4;
      DUP2_ARG2(__lsx_vldx, src_tmp, src_stride, src_tmp, src_stride2, src5,
                src6);
      src_tmp += src_stride2;
      src7 = __lsx_vldx(src_tmp, src_stride);
      src_tmp += src_stride2;

      __lsx_vst(src0, dst_tmp, 0);
      dst_tmp += dst_stride;
      __lsx_vst(src1, dst_tmp, 0);
      dst_tmp += dst_stride;
      __lsx_vst(src2, dst_tmp, 0);
      dst_tmp += dst_stride;
      __lsx_vst(src3, dst_tmp, 0);
      dst_tmp += dst_stride;
      __lsx_vst(src4, dst_tmp, 0);
      dst_tmp += dst_stride;
      __lsx_vst(src5, dst_tmp, 0);
      dst_tmp += dst_stride;
      __lsx_vst(src6, dst_tmp, 0);
      dst_tmp += dst_stride;
      __lsx_vst(src7, dst_tmp, 0);
      dst_tmp += dst_stride;
    }
    src += 16;
    dst += 16;
  }
}

static void copy_width16_lsx(const uint8_t *src, int32_t src_stride,
                             uint8_t *dst, int32_t dst_stride, int32_t height) {
  int32_t cnt;
  __m128i src0, src1, src2, src3, src4, src5, src6, src7;
  int32_t src_stride2 = src_stride << 1;
  int32_t src_stride3 = src_stride2 + src_stride;
  int32_t src_stride4 = src_stride2 << 1;

  if ((height % 12) == 0) {
    for (cnt = (height / 12); cnt--;) {
      src0 = __lsx_vld(src, 0);
      DUP4_ARG2(__lsx_vldx, src, src_stride, src, src_stride2, src, src_stride3,
                src, src_stride4, src1, src2, src3, src4);
      src += src_stride4;
      DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride2, src5, src6);
      src += src_stride2;
      src7 = __lsx_vldx(src, src_stride);
      src += src_stride2;

      __lsx_vst(src0, dst, 0);
      dst += dst_stride;
      __lsx_vst(src1, dst, 0);
      dst += dst_stride;
      __lsx_vst(src2, dst, 0);
      dst += dst_stride;
      __lsx_vst(src3, dst, 0);
      dst += dst_stride;
      __lsx_vst(src4, dst, 0);
      dst += dst_stride;
      __lsx_vst(src5, dst, 0);
      dst += dst_stride;
      __lsx_vst(src6, dst, 0);
      dst += dst_stride;
      __lsx_vst(src7, dst, 0);
      dst += dst_stride;

      src0 = __lsx_vld(src, 0);
      DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride2, src1, src2);
      src3 = __lsx_vldx(src, src_stride3);
      src += src_stride4;

      __lsx_vst(src0, dst, 0);
      dst += dst_stride;
      __lsx_vst(src1, dst, 0);
      dst += dst_stride;
      __lsx_vst(src2, dst, 0);
      dst += dst_stride;
      __lsx_vst(src3, dst, 0);
      dst += dst_stride;
    }
  } else if ((height % 8) == 0) {
    copy_16multx8mult_lsx(src, src_stride, dst, dst_stride, height, 16);
  } else if ((height % 4) == 0) {
    for (cnt = (height >> 2); cnt--;) {
      src0 = __lsx_vld(src, 0);
      DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride2, src1, src2);
      src3 = __lsx_vldx(src, src_stride3);
      src += src_stride4;

      __lsx_vst(src0, dst, 0);
      dst += dst_stride;
      __lsx_vst(src1, dst, 0);
      dst += dst_stride;
      __lsx_vst(src2, dst, 0);
      dst += dst_stride;
      __lsx_vst(src3, dst, 0);
      dst += dst_stride;
    }
  }
}

static void copy_width32_lsx(const uint8_t *src, int32_t src_stride,
                             uint8_t *dst, int32_t dst_stride, int32_t height) {
  int32_t cnt;
  uint8_t *src_tmp;
  uint8_t *dst_tmp;
  __m128i src0, src1, src2, src3, src4, src5, src6, src7;
  int32_t src_stride2 = src_stride << 1;
  int32_t src_stride3 = src_stride2 + src_stride;
  int32_t src_stride4 = src_stride2 << 1;

  if ((height % 12) == 0) {
    for (cnt = (height / 12); cnt--;) {
      src0 = __lsx_vld(src, 0);
      DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride2, src1, src2);
      src3 = __lsx_vldx(src, src_stride3);

      src_tmp = (uint8_t *)src + 16;
      src4 = __lsx_vld(src_tmp, 0);
      DUP2_ARG2(__lsx_vldx, src_tmp, src_stride, src_tmp, src_stride2, src5,
                src6);
      src7 = __lsx_vldx(src_tmp, src_stride3);
      src += src_stride4;

      __lsx_vst(src0, dst, 0);
      dst += dst_stride;
      __lsx_vst(src1, dst, 0);
      dst += dst_stride;
      __lsx_vst(src2, dst, 0);
      dst += dst_stride;
      __lsx_vst(src3, dst, 0);
      dst += dst_stride;

      dst_tmp = dst + 16;
      __lsx_vst(src4, dst_tmp, 0);
      dst_tmp += dst_stride;
      __lsx_vst(src5, dst_tmp, 0);
      dst_tmp += dst_stride;
      __lsx_vst(src6, dst_tmp, 0);
      dst_tmp += dst_stride;
      __lsx_vst(src7, dst_tmp, 0);
      dst_tmp += dst_stride;

      src0 = __lsx_vld(src, 0);
      DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride2, src1, src2);
      src3 = __lsx_vldx(src, src_stride3);

      src_tmp = (uint8_t *)src + 16;
      src4 = __lsx_vld(src_tmp, 0);
      DUP2_ARG2(__lsx_vldx, src_tmp, src_stride, src_tmp, src_stride2, src5,
                src6);
      src7 = __lsx_vldx(src_tmp, src_stride3);
      src += src_stride4;

      __lsx_vst(src0, dst, 0);
      dst += dst_stride;
      __lsx_vst(src1, dst, 0);
      dst += dst_stride;
      __lsx_vst(src2, dst, 0);
      dst += dst_stride;
      __lsx_vst(src3, dst, 0);
      dst += dst_stride;

      dst_tmp = dst + 16;
      __lsx_vst(src4, dst_tmp, 0);
      dst_tmp += dst_stride;
      __lsx_vst(src5, dst_tmp, 0);
      dst_tmp += dst_stride;
      __lsx_vst(src6, dst_tmp, 0);
      dst_tmp += dst_stride;
      __lsx_vst(src7, dst_tmp, 0);
      dst_tmp += dst_stride;

      src0 = __lsx_vld(src, 0);
      DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride2, src1, src2);
      src3 = __lsx_vldx(src, src_stride3);

      src_tmp = (uint8_t *)src + 16;
      src4 = __lsx_vld(src_tmp, 0);
      DUP2_ARG2(__lsx_vldx, src_tmp, src_stride, src_tmp, src_stride2, src5,
                src6);
      src7 = __lsx_vldx(src_tmp, src_stride3);
      src += src_stride4;

      __lsx_vst(src0, dst, 0);
      dst += dst_stride;
      __lsx_vst(src1, dst, 0);
      dst += dst_stride;
      __lsx_vst(src2, dst, 0);
      dst += dst_stride;
      __lsx_vst(src3, dst, 0);
      dst += dst_stride;

      dst_tmp = dst + 16;
      __lsx_vst(src4, dst_tmp, 0);
      dst_tmp += dst_stride;
      __lsx_vst(src5, dst_tmp, 0);
      dst_tmp += dst_stride;
      __lsx_vst(src6, dst_tmp, 0);
      dst_tmp += dst_stride;
      __lsx_vst(src7, dst_tmp, 0);
      dst_tmp += dst_stride;
    }
  } else if ((height % 8) == 0) {
    copy_16multx8mult_lsx(src, src_stride, dst, dst_stride, height, 32);
  } else if ((height % 4) == 0) {
    for (cnt = (height >> 2); cnt--;) {
      src0 = __lsx_vld(src, 0);
      DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride2, src1, src2);
      src3 = __lsx_vldx(src, src_stride3);

      src_tmp = (uint8_t *)src + 16;
      src4 = __lsx_vld(src_tmp, 0);
      DUP2_ARG2(__lsx_vldx, src_tmp, src_stride, src_tmp, src_stride2, src5,
                src6);
      src7 = __lsx_vldx(src_tmp, src_stride3);
      src += src_stride4;

      __lsx_vst(src0, dst, 0);
      dst += dst_stride;
      __lsx_vst(src1, dst, 0);
      dst += dst_stride;
      __lsx_vst(src2, dst, 0);
      dst += dst_stride;
      __lsx_vst(src3, dst, 0);
      dst += dst_stride;

      dst_tmp = dst + 16;
      __lsx_vst(src4, dst_tmp, 0);
      dst_tmp += dst_stride;
      __lsx_vst(src5, dst_tmp, 0);
      dst_tmp += dst_stride;
      __lsx_vst(src6, dst_tmp, 0);
      dst_tmp += dst_stride;
      __lsx_vst(src7, dst_tmp, 0);
      dst_tmp += dst_stride;
    }
  }
}

static void copy_width64_lsx(const uint8_t *src, int32_t src_stride,
                             uint8_t *dst, int32_t dst_stride, int32_t height) {
  copy_16multx8mult_lsx(src, src_stride, dst, dst_stride, height, 64);
}

void vpx_convolve_copy_lsx(const uint8_t *src, ptrdiff_t src_stride,
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
      uint32_t cnt;
      __m128i tmp;
      for (cnt = h; cnt--;) {
        tmp = __lsx_vldrepl_w(src, 0);
        __lsx_vstelm_w(tmp, dst, 0, 0);
        src += src_stride;
        dst += dst_stride;
      }
      break;
    }
    case 8: {
      copy_width8_lsx(src, src_stride, dst, dst_stride, h);
      break;
    }
    case 16: {
      copy_width16_lsx(src, src_stride, dst, dst_stride, h);
      break;
    }
    case 32: {
      copy_width32_lsx(src, src_stride, dst, dst_stride, h);
      break;
    }
    case 64: {
      copy_width64_lsx(src, src_stride, dst, dst_stride, h);
      break;
    }
    default: {
      uint32_t cnt;
      for (cnt = h; cnt--;) {
        memcpy(dst, src, w);
        src += src_stride;
        dst += dst_stride;
      }
      break;
    }
  }
}
