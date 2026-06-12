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

static void avg_width4_msa(const uint8_t *src, int32_t src_stride, uint8_t *dst,
                           int32_t dst_stride, int32_t height) {
  int32_t cnt;
  uint32_t out0, out1, out2, out3;
  v16u8 src0, src1, src2, src3;
  v16u8 dst0, dst1, dst2, dst3;

  if (0 == (height % 4)) {
    for (cnt = (height / 4); cnt--;) {
      LD_UB4(src, src_stride, src0, src1, src2, src3);
      src += (4 * src_stride);

      LD_UB4(dst, dst_stride, dst0, dst1, dst2, dst3);

      AVER_UB4_UB(src0, dst0, src1, dst1, src2, dst2, src3, dst3, dst0, dst1,
                  dst2, dst3);

      out0 = __msa_copy_u_w((v4i32)dst0, 0);
      out1 = __msa_copy_u_w((v4i32)dst1, 0);
      out2 = __msa_copy_u_w((v4i32)dst2, 0);
      out3 = __msa_copy_u_w((v4i32)dst3, 0);
      SW4(out0, out1, out2, out3, dst, dst_stride);
      dst += (4 * dst_stride);
    }
  } else if (0 == (height % 2)) {
    for (cnt = (height / 2); cnt--;) {
      LD_UB2(src, src_stride, src0, src1);
      src += (2 * src_stride);

      LD_UB2(dst, dst_stride, dst0, dst1);

      AVER_UB2_UB(src0, dst0, src1, dst1, dst0, dst1);

      out0 = __msa_copy_u_w((v4i32)dst0, 0);
      out1 = __msa_copy_u_w((v4i32)dst1, 0);
      SW(out0, dst);
      dst += dst_stride;
      SW(out1, dst);
      dst += dst_stride;
    }
  }
}

static void avg_width8_msa(const uint8_t *src, int32_t src_stride, uint8_t *dst,
                           int32_t dst_stride, int32_t height) {
  int32_t cnt;
  uint64_t out0, out1, out2, out3;
  v16u8 src0, src1, src2, src3;
  v16u8 dst0, dst1, dst2, dst3;

  for (cnt = (height / 4); cnt--;) {
    LD_UB4(src, src_stride, src0, src1, src2, src3);
    src += (4 * src_stride);
    LD_UB4(dst, dst_stride, dst0, dst1, dst2, dst3);

    AVER_UB4_UB(src0, dst0, src1, dst1, src2, dst2, src3, dst3, dst0, dst1,
                dst2, dst3);

    out0 = __msa_copy_u_d((v2i64)dst0, 0);
    out1 = __msa_copy_u_d((v2i64)dst1, 0);
    out2 = __msa_copy_u_d((v2i64)dst2, 0);
    out3 = __msa_copy_u_d((v2i64)dst3, 0);
    SD4(out0, out1, out2, out3, dst, dst_stride);
    dst += (4 * dst_stride);
  }
}

static void avg_width16_msa(const uint8_t *src, int32_t src_stride,
                            uint8_t *dst, int32_t dst_stride, int32_t height) {
  int32_t cnt;
  v16u8 src0, src1, src2, src3, src4, src5, src6, src7;
  v16u8 dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7;

  for (cnt = (height / 8); cnt--;) {
    LD_UB8(src, src_stride, src0, src1, src2, src3, src4, src5, src6, src7);
    src += (8 * src_stride);
    LD_UB8(dst, dst_stride, dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7);

    AVER_UB4_UB(src0, dst0, src1, dst1, src2, dst2, src3, dst3, dst0, dst1,
                dst2, dst3);
    AVER_UB4_UB(src4, dst4, src5, dst5, src6, dst6, src7, dst7, dst4, dst5,
                dst6, dst7);
    ST_UB8(dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7, dst, dst_stride);
    dst += (8 * dst_stride);
  }
}

static void avg_width32_msa(const uint8_t *src, int32_t src_stride,
                            uint8_t *dst, int32_t dst_stride, int32_t height) {
  int32_t cnt;
  uint8_t *dst_dup = dst;
  v16u8 src0, src1, src2, src3, src4, src5, src6, src7;
  v16u8 src8, src9, src10, src11, src12, src13, src14, src15;
  v16u8 dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7;
  v16u8 dst8, dst9, dst10, dst11, dst12, dst13, dst14, dst15;

  for (cnt = (height / 8); cnt--;) {
    LD_UB4(src, src_stride, src0, src2, src4, src6);
    LD_UB4(src + 16, src_stride, src1, src3, src5, src7);
    src += (4 * src_stride);
    LD_UB4(dst_dup, dst_stride, dst0, dst2, dst4, dst6);
    LD_UB4(dst_dup + 16, dst_stride, dst1, dst3, dst5, dst7);
    dst_dup += (4 * dst_stride);
    LD_UB4(src, src_stride, src8, src10, src12, src14);
    LD_UB4(src + 16, src_stride, src9, src11, src13, src15);
    src += (4 * src_stride);
    LD_UB4(dst_dup, dst_stride, dst8, dst10, dst12, dst14);
    LD_UB4(dst_dup + 16, dst_stride, dst9, dst11, dst13, dst15);
    dst_dup += (4 * dst_stride);

    AVER_UB4_UB(src0, dst0, src1, dst1, src2, dst2, src3, dst3, dst0, dst1,
                dst2, dst3);
    AVER_UB4_UB(src4, dst4, src5, dst5, src6, dst6, src7, dst7, dst4, dst5,
                dst6, dst7);
    AVER_UB4_UB(src8, dst8, src9, dst9, src10, dst10, src11, dst11, dst8, dst9,
                dst10, dst11);
    AVER_UB4_UB(src12, dst12, src13, dst13, src14, dst14, src15, dst15, dst12,
                dst13, dst14, dst15);

    ST_UB4(dst0, dst2, dst4, dst6, dst, dst_stride);
    ST_UB4(dst1, dst3, dst5, dst7, dst + 16, dst_stride);
    dst += (4 * dst_stride);
    ST_UB4(dst8, dst10, dst12, dst14, dst, dst_stride);
    ST_UB4(dst9, dst11, dst13, dst15, dst + 16, dst_stride);
    dst += (4 * dst_stride);
  }
}

static void avg_width64_msa(const uint8_t *src, int32_t src_stride,
                            uint8_t *dst, int32_t dst_stride, int32_t height) {
  int32_t cnt;
  uint8_t *dst_dup = dst;
  v16u8 src0, src1, src2, src3, src4, src5, src6, src7;
  v16u8 src8, src9, src10, src11, src12, src13, src14, src15;
  v16u8 dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7;
  v16u8 dst8, dst9, dst10, dst11, dst12, dst13, dst14, dst15;

  for (cnt = (height / 4); cnt--;) {
    LD_UB4(src, 16, src0, src1, src2, src3);
    src += src_stride;
    LD_UB4(src, 16, src4, src5, src6, src7);
    src += src_stride;
    LD_UB4(src, 16, src8, src9, src10, src11);
    src += src_stride;
    LD_UB4(src, 16, src12, src13, src14, src15);
    src += src_stride;

    LD_UB4(dst_dup, 16, dst0, dst1, dst2, dst3);
    dst_dup += dst_stride;
    LD_UB4(dst_dup, 16, dst4, dst5, dst6, dst7);
    dst_dup += dst_stride;
    LD_UB4(dst_dup, 16, dst8, dst9, dst10, dst11);
    dst_dup += dst_stride;
    LD_UB4(dst_dup, 16, dst12, dst13, dst14, dst15);
    dst_dup += dst_stride;

    AVER_UB4_UB(src0, dst0, src1, dst1, src2, dst2, src3, dst3, dst0, dst1,
                dst2, dst3);
    AVER_UB4_UB(src4, dst4, src5, dst5, src6, dst6, src7, dst7, dst4, dst5,
                dst6, dst7);
    AVER_UB4_UB(src8, dst8, src9, dst9, src10, dst10, src11, dst11, dst8, dst9,
                dst10, dst11);
    AVER_UB4_UB(src12, dst12, src13, dst13, src14, dst14, src15, dst15, dst12,
                dst13, dst14, dst15);

    ST_UB4(dst0, dst1, dst2, dst3, dst, 16);
    dst += dst_stride;
    ST_UB4(dst4, dst5, dst6, dst7, dst, 16);
    dst += dst_stride;
    ST_UB4(dst8, dst9, dst10, dst11, dst, 16);
    dst += dst_stride;
    ST_UB4(dst12, dst13, dst14, dst15, dst, 16);
    dst += dst_stride;
  }
}

void vpx_convolve_avg_msa(const uint8_t *src, ptrdiff_t src_stride,
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
      avg_width4_msa(src, src_stride, dst, dst_stride, h);
      break;
    }
    case 8: {
      avg_width8_msa(src, src_stride, dst, dst_stride, h);
      break;
    }
    case 16: {
      avg_width16_msa(src, src_stride, dst, dst_stride, h);
      break;
    }
    case 32: {
      avg_width32_msa(src, src_stride, dst, dst_stride, h);
      break;
    }
    case 64: {
      avg_width64_msa(src, src_stride, dst, dst_stride, h);
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
