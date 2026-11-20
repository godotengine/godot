/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <arm_neon.h>
#include <string.h>

#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"

void vpx_convolve_copy_neon(const uint8_t *src, ptrdiff_t src_stride,
                            uint8_t *dst, ptrdiff_t dst_stride,
                            const InterpKernel *filter, int x0_q4,
                            int x_step_q4, int y0_q4, int y_step_q4, int w,
                            int h) {
  (void)filter;
  (void)x0_q4;
  (void)x_step_q4;
  (void)y0_q4;
  (void)y_step_q4;

  if (w < 8) {  // copy4
    do {
      memcpy(dst, src, 4);
      src += src_stride;
      dst += dst_stride;
      memcpy(dst, src, 4);
      src += src_stride;
      dst += dst_stride;
      h -= 2;
    } while (h != 0);
  } else if (w == 8) {  // copy8
    uint8x8_t s0, s1;
    do {
      s0 = vld1_u8(src);
      src += src_stride;
      s1 = vld1_u8(src);
      src += src_stride;

      vst1_u8(dst, s0);
      dst += dst_stride;
      vst1_u8(dst, s1);
      dst += dst_stride;
      h -= 2;
    } while (h != 0);
  } else if (w < 32) {  // copy16
    uint8x16_t s0, s1;
    do {
      s0 = vld1q_u8(src);
      src += src_stride;
      s1 = vld1q_u8(src);
      src += src_stride;

      vst1q_u8(dst, s0);
      dst += dst_stride;
      vst1q_u8(dst, s1);
      dst += dst_stride;
      h -= 2;
    } while (h != 0);
  } else if (w == 32) {  // copy32
    uint8x16_t s0, s1, s2, s3;
    do {
      s0 = vld1q_u8(src);
      s1 = vld1q_u8(src + 16);
      src += src_stride;
      s2 = vld1q_u8(src);
      s3 = vld1q_u8(src + 16);
      src += src_stride;

      vst1q_u8(dst, s0);
      vst1q_u8(dst + 16, s1);
      dst += dst_stride;
      vst1q_u8(dst, s2);
      vst1q_u8(dst + 16, s3);
      dst += dst_stride;
      h -= 2;
    } while (h != 0);
  } else {  // copy64
    uint8x16_t s0, s1, s2, s3;
    do {
      s0 = vld1q_u8(src);
      s1 = vld1q_u8(src + 16);
      s2 = vld1q_u8(src + 32);
      s3 = vld1q_u8(src + 48);
      src += src_stride;

      vst1q_u8(dst, s0);
      vst1q_u8(dst + 16, s1);
      vst1q_u8(dst + 32, s2);
      vst1q_u8(dst + 48, s3);
      dst += dst_stride;
    } while (--h);
  }
}
