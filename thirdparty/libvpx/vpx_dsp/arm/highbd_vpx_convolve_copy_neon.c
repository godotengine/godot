/*
 *  Copyright (c) 2016 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <arm_neon.h>

#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"

void vpx_highbd_convolve_copy_neon(const uint16_t *src, ptrdiff_t src_stride,
                                   uint16_t *dst, ptrdiff_t dst_stride,
                                   const InterpKernel *filter, int x0_q4,
                                   int x_step_q4, int y0_q4, int y_step_q4,
                                   int w, int h, int bd) {
  (void)filter;
  (void)x0_q4;
  (void)x_step_q4;
  (void)y0_q4;
  (void)y_step_q4;
  (void)bd;

  if (w < 8) {  // copy4
    uint16x4_t s0, s1;
    do {
      s0 = vld1_u16(src);
      src += src_stride;
      s1 = vld1_u16(src);
      src += src_stride;

      vst1_u16(dst, s0);
      dst += dst_stride;
      vst1_u16(dst, s1);
      dst += dst_stride;
      h -= 2;
    } while (h != 0);
  } else if (w == 8) {  // copy8
    uint16x8_t s0, s1;
    do {
      s0 = vld1q_u16(src);
      src += src_stride;
      s1 = vld1q_u16(src);
      src += src_stride;

      vst1q_u16(dst, s0);
      dst += dst_stride;
      vst1q_u16(dst, s1);
      dst += dst_stride;
      h -= 2;
    } while (h != 0);
  } else if (w < 32) {  // copy16
    uint16x8_t s0, s1, s2, s3;
    do {
      s0 = vld1q_u16(src);
      s1 = vld1q_u16(src + 8);
      src += src_stride;
      s2 = vld1q_u16(src);
      s3 = vld1q_u16(src + 8);
      src += src_stride;

      vst1q_u16(dst, s0);
      vst1q_u16(dst + 8, s1);
      dst += dst_stride;
      vst1q_u16(dst, s2);
      vst1q_u16(dst + 8, s3);
      dst += dst_stride;
      h -= 2;
    } while (h != 0);
  } else if (w == 32) {  // copy32
    uint16x8_t s0, s1, s2, s3;
    do {
      s0 = vld1q_u16(src);
      s1 = vld1q_u16(src + 8);
      s2 = vld1q_u16(src + 16);
      s3 = vld1q_u16(src + 24);
      src += src_stride;

      vst1q_u16(dst, s0);
      vst1q_u16(dst + 8, s1);
      vst1q_u16(dst + 16, s2);
      vst1q_u16(dst + 24, s3);
      dst += dst_stride;
    } while (--h != 0);
  } else {  // copy64
    uint16x8_t s0, s1, s2, s3, s4, s5, s6, s7;
    do {
      s0 = vld1q_u16(src);
      s1 = vld1q_u16(src + 8);
      s2 = vld1q_u16(src + 16);
      s3 = vld1q_u16(src + 24);
      s4 = vld1q_u16(src + 32);
      s5 = vld1q_u16(src + 40);
      s6 = vld1q_u16(src + 48);
      s7 = vld1q_u16(src + 56);
      src += src_stride;

      vst1q_u16(dst, s0);
      vst1q_u16(dst + 8, s1);
      vst1q_u16(dst + 16, s2);
      vst1q_u16(dst + 24, s3);
      vst1q_u16(dst + 32, s4);
      vst1q_u16(dst + 40, s5);
      vst1q_u16(dst + 48, s6);
      vst1q_u16(dst + 56, s7);
      dst += dst_stride;
    } while (--h != 0);
  }
}
