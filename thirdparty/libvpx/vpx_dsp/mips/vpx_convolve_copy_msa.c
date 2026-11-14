/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <string.h>
#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/mips/macros_msa.h"

static void copy_width8_msa(const uint8_t *src, int32_t src_stride,
                            uint8_t *dst, int32_t dst_stride, int32_t height) {
  int32_t cnt;
  uint64_t out0, out1, out2, out3, out4, out5, out6, out7;
  v16u8 src0, src1, src2, src3, src4, src5, src6, src7;

  if (0 == height % 12) {
    for (cnt = (height / 12); cnt--;) {
      LD_UB8(src, src_stride, src0, src1, src2, src3, src4, src5, src6, src7);
      src += (8 * src_stride);

      out0 = __msa_copy_u_d((v2i64)src0, 0);
      out1 = __msa_copy_u_d((v2i64)src1, 0);
      out2 = __msa_copy_u_d((v2i64)src2, 0);
      out3 = __msa_copy_u_d((v2i64)src3, 0);
      out4 = __msa_copy_u_d((v2i64)src4, 0);
      out5 = __msa_copy_u_d((v2i64)src5, 0);
      out6 = __msa_copy_u_d((v2i64)src6, 0);
      out7 = __msa_copy_u_d((v2i64)src7, 0);

      SD4(out0, out1, out2, out3, dst, dst_stride);
      dst += (4 * dst_stride);
      SD4(out4, out5, out6, out7, dst, dst_stride);
      dst += (4 * dst_stride);

      LD_UB4(src, src_stride, src0, src1, src2, src3);
      src += (4 * src_stride);

      out0 = __msa_copy_u_d((v2i64)src0, 0);
      out1 = __msa_copy_u_d((v2i64)src1, 0);
      out2 = __msa_copy_u_d((v2i64)src2, 0);
      out3 = __msa_copy_u_d((v2i64)src3, 0);
      SD4(out0, out1, out2, out3, dst, dst_stride);
      dst += (4 * dst_stride);
    }
  } else if (0 == height % 8) {
    for (cnt = height >> 3; cnt--;) {
      LD_UB8(src, src_stride, src0, src1, src2, src3, src4, src5, src6, src7);
      src += (8 * src_stride);

      out0 = __msa_copy_u_d((v2i64)src0, 0);
      out1 = __msa_copy_u_d((v2i64)src1, 0);
      out2 = __msa_copy_u_d((v2i64)src2, 0);
      out3 = __msa_copy_u_d((v2i64)src3, 0);
      out4 = __msa_copy_u_d((v2i64)src4, 0);
      out5 = __msa_copy_u_d((v2i64)src5, 0);
      out6 = __msa_copy_u_d((v2i64)src6, 0);
      out7 = __msa_copy_u_d((v2i64)src7, 0);

      SD4(out0, out1, out2, out3, dst, dst_stride);
      dst += (4 * dst_stride);
      SD4(out4, out5, out6, out7, dst, dst_stride);
      dst += (4 * dst_stride);
    }
  } else if (0 == height % 4) {
    for (cnt = (height / 4); cnt--;) {
      LD_UB4(src, src_stride, src0, src1, src2, src3);
      src += (4 * src_stride);
      out0 = __msa_copy_u_d((v2i64)src0, 0);
      out1 = __msa_copy_u_d((v2i64)src1, 0);
      out2 = __msa_copy_u_d((v2i64)src2, 0);
      out3 = __msa_copy_u_d((v2i64)src3, 0);

      SD4(out0, out1, out2, out3, dst, dst_stride);
      dst += (4 * dst_stride);
    }
  } else if (0 == height % 2) {
    for (cnt = (height / 2); cnt--;) {
      LD_UB2(src, src_stride, src0, src1);
      src += (2 * src_stride);
      out0 = __msa_copy_u_d((v2i64)src0, 0);
      out1 = __msa_copy_u_d((v2i64)src1, 0);

      SD(out0, dst);
      dst += dst_stride;
      SD(out1, dst);
      dst += dst_stride;
    }
  }
}

static void copy_16multx8mult_msa(const uint8_t *src, int32_t src_stride,
                                  uint8_t *dst, int32_t dst_stride,
                                  int32_t height, int32_t width) {
  int32_t cnt, loop_cnt;
  const uint8_t *src_tmp;
  uint8_t *dst_tmp;
  v16u8 src0, src1, src2, src3, src4, src5, src6, src7;

  for (cnt = (width >> 4); cnt--;) {
    src_tmp = src;
    dst_tmp = dst;

    for (loop_cnt = (height >> 3); loop_cnt--;) {
      LD_UB8(src_tmp, src_stride, src0, src1, src2, src3, src4, src5, src6,
             src7);
      src_tmp += (8 * src_stride);

      ST_UB8(src0, src1, src2, src3, src4, src5, src6, src7, dst_tmp,
             dst_stride);
      dst_tmp += (8 * dst_stride);
    }

    src += 16;
    dst += 16;
  }
}

static void copy_width16_msa(const uint8_t *src, int32_t src_stride,
                             uint8_t *dst, int32_t dst_stride, int32_t height) {
  int32_t cnt;
  v16u8 src0, src1, src2, src3, src4, src5, src6, src7;

  if (0 == height % 12) {
    for (cnt = (height / 12); cnt--;) {
      LD_UB8(src, src_stride, src0, src1, src2, src3, src4, src5, src6, src7);
      src += (8 * src_stride);
      ST_UB8(src0, src1, src2, src3, src4, src5, src6, src7, dst, dst_stride);
      dst += (8 * dst_stride);

      LD_UB4(src, src_stride, src0, src1, src2, src3);
      src += (4 * src_stride);
      ST_UB4(src0, src1, src2, src3, dst, dst_stride);
      dst += (4 * dst_stride);
    }
  } else if (0 == height % 8) {
    copy_16multx8mult_msa(src, src_stride, dst, dst_stride, height, 16);
  } else if (0 == height % 4) {
    for (cnt = (height >> 2); cnt--;) {
      LD_UB4(src, src_stride, src0, src1, src2, src3);
      src += (4 * src_stride);

      ST_UB4(src0, src1, src2, src3, dst, dst_stride);
      dst += (4 * dst_stride);
    }
  }
}

static void copy_width32_msa(const uint8_t *src, int32_t src_stride,
                             uint8_t *dst, int32_t dst_stride, int32_t height) {
  int32_t cnt;
  v16u8 src0, src1, src2, src3, src4, src5, src6, src7;

  if (0 == height % 12) {
    for (cnt = (height / 12); cnt--;) {
      LD_UB4(src, src_stride, src0, src1, src2, src3);
      LD_UB4(src + 16, src_stride, src4, src5, src6, src7);
      src += (4 * src_stride);
      ST_UB4(src0, src1, src2, src3, dst, dst_stride);
      ST_UB4(src4, src5, src6, src7, dst + 16, dst_stride);
      dst += (4 * dst_stride);

      LD_UB4(src, src_stride, src0, src1, src2, src3);
      LD_UB4(src + 16, src_stride, src4, src5, src6, src7);
      src += (4 * src_stride);
      ST_UB4(src0, src1, src2, src3, dst, dst_stride);
      ST_UB4(src4, src5, src6, src7, dst + 16, dst_stride);
      dst += (4 * dst_stride);

      LD_UB4(src, src_stride, src0, src1, src2, src3);
      LD_UB4(src + 16, src_stride, src4, src5, src6, src7);
      src += (4 * src_stride);
      ST_UB4(src0, src1, src2, src3, dst, dst_stride);
      ST_UB4(src4, src5, src6, src7, dst + 16, dst_stride);
      dst += (4 * dst_stride);
    }
  } else if (0 == height % 8) {
    copy_16multx8mult_msa(src, src_stride, dst, dst_stride, height, 32);
  } else if (0 == height % 4) {
    for (cnt = (height >> 2); cnt--;) {
      LD_UB4(src, src_stride, src0, src1, src2, src3);
      LD_UB4(src + 16, src_stride, src4, src5, src6, src7);
      src += (4 * src_stride);
      ST_UB4(src0, src1, src2, src3, dst, dst_stride);
      ST_UB4(src4, src5, src6, src7, dst + 16, dst_stride);
      dst += (4 * dst_stride);
    }
  }
}

static void copy_width64_msa(const uint8_t *src, int32_t src_stride,
                             uint8_t *dst, int32_t dst_stride, int32_t height) {
  copy_16multx8mult_msa(src, src_stride, dst, dst_stride, height, 64);
}

void vpx_convolve_copy_msa(const uint8_t *src, ptrdiff_t src_stride,
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
      uint32_t cnt, tmp;
      /* 1 word storage */
      for (cnt = h; cnt--;) {
        tmp = LW(src);
        SW(tmp, dst);
        src += src_stride;
        dst += dst_stride;
      }
      break;
    }
    case 8: {
      copy_width8_msa(src, src_stride, dst, dst_stride, h);
      break;
    }
    case 16: {
      copy_width16_msa(src, src_stride, dst, dst_stride, h);
      break;
    }
    case 32: {
      copy_width32_msa(src, src_stride, dst, dst_stride, h);
      break;
    }
    case 64: {
      copy_width64_msa(src, src_stride, dst, dst_stride, h);
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
