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

void vpx_highbd_convolve_avg_neon(const uint16_t *src, ptrdiff_t src_stride,
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

  if (w < 8) {  // avg4
    uint16x4_t s0, s1, d0, d1;
    uint16x8_t s01, d01;
    do {
      s0 = vld1_u16(src);
      d0 = vld1_u16(dst);
      src += src_stride;
      s1 = vld1_u16(src);
      d1 = vld1_u16(dst + dst_stride);
      src += src_stride;
      s01 = vcombine_u16(s0, s1);
      d01 = vcombine_u16(d0, d1);
      d01 = vrhaddq_u16(s01, d01);
      vst1_u16(dst, vget_low_u16(d01));
      dst += dst_stride;
      vst1_u16(dst, vget_high_u16(d01));
      dst += dst_stride;
      h -= 2;
    } while (h > 0);
  } else if (w == 8) {  // avg8
    uint16x8_t s0, s1, d0, d1;
    do {
      s0 = vld1q_u16(src);
      d0 = vld1q_u16(dst);
      src += src_stride;
      s1 = vld1q_u16(src);
      d1 = vld1q_u16(dst + dst_stride);
      src += src_stride;

      d0 = vrhaddq_u16(s0, d0);
      d1 = vrhaddq_u16(s1, d1);

      vst1q_u16(dst, d0);
      dst += dst_stride;
      vst1q_u16(dst, d1);
      dst += dst_stride;
      h -= 2;
    } while (h > 0);
  } else if (w < 32) {  // avg16
    uint16x8_t s0l, s0h, s1l, s1h, d0l, d0h, d1l, d1h;
    do {
      s0l = vld1q_u16(src);
      s0h = vld1q_u16(src + 8);
      d0l = vld1q_u16(dst);
      d0h = vld1q_u16(dst + 8);
      src += src_stride;
      s1l = vld1q_u16(src);
      s1h = vld1q_u16(src + 8);
      d1l = vld1q_u16(dst + dst_stride);
      d1h = vld1q_u16(dst + dst_stride + 8);
      src += src_stride;

      d0l = vrhaddq_u16(s0l, d0l);
      d0h = vrhaddq_u16(s0h, d0h);
      d1l = vrhaddq_u16(s1l, d1l);
      d1h = vrhaddq_u16(s1h, d1h);

      vst1q_u16(dst, d0l);
      vst1q_u16(dst + 8, d0h);
      dst += dst_stride;
      vst1q_u16(dst, d1l);
      vst1q_u16(dst + 8, d1h);
      dst += dst_stride;
      h -= 2;
    } while (h > 0);
  } else if (w == 32) {  // avg32
    uint16x8_t s0, s1, s2, s3, d0, d1, d2, d3;
    do {
      s0 = vld1q_u16(src);
      s1 = vld1q_u16(src + 8);
      s2 = vld1q_u16(src + 16);
      s3 = vld1q_u16(src + 24);
      d0 = vld1q_u16(dst);
      d1 = vld1q_u16(dst + 8);
      d2 = vld1q_u16(dst + 16);
      d3 = vld1q_u16(dst + 24);
      src += src_stride;

      d0 = vrhaddq_u16(s0, d0);
      d1 = vrhaddq_u16(s1, d1);
      d2 = vrhaddq_u16(s2, d2);
      d3 = vrhaddq_u16(s3, d3);

      vst1q_u16(dst, d0);
      vst1q_u16(dst + 8, d1);
      vst1q_u16(dst + 16, d2);
      vst1q_u16(dst + 24, d3);
      dst += dst_stride;

      s0 = vld1q_u16(src);
      s1 = vld1q_u16(src + 8);
      s2 = vld1q_u16(src + 16);
      s3 = vld1q_u16(src + 24);
      d0 = vld1q_u16(dst);
      d1 = vld1q_u16(dst + 8);
      d2 = vld1q_u16(dst + 16);
      d3 = vld1q_u16(dst + 24);
      src += src_stride;

      d0 = vrhaddq_u16(s0, d0);
      d1 = vrhaddq_u16(s1, d1);
      d2 = vrhaddq_u16(s2, d2);
      d3 = vrhaddq_u16(s3, d3);

      vst1q_u16(dst, d0);
      vst1q_u16(dst + 8, d1);
      vst1q_u16(dst + 16, d2);
      vst1q_u16(dst + 24, d3);
      dst += dst_stride;
      h -= 2;
    } while (h > 0);
  } else {  // avg64
    uint16x8_t s0, s1, s2, s3, d0, d1, d2, d3;
    do {
      s0 = vld1q_u16(src);
      s1 = vld1q_u16(src + 8);
      s2 = vld1q_u16(src + 16);
      s3 = vld1q_u16(src + 24);
      d0 = vld1q_u16(dst);
      d1 = vld1q_u16(dst + 8);
      d2 = vld1q_u16(dst + 16);
      d3 = vld1q_u16(dst + 24);

      d0 = vrhaddq_u16(s0, d0);
      d1 = vrhaddq_u16(s1, d1);
      d2 = vrhaddq_u16(s2, d2);
      d3 = vrhaddq_u16(s3, d3);

      vst1q_u16(dst, d0);
      vst1q_u16(dst + 8, d1);
      vst1q_u16(dst + 16, d2);
      vst1q_u16(dst + 24, d3);

      s0 = vld1q_u16(src + 32);
      s1 = vld1q_u16(src + 40);
      s2 = vld1q_u16(src + 48);
      s3 = vld1q_u16(src + 56);
      d0 = vld1q_u16(dst + 32);
      d1 = vld1q_u16(dst + 40);
      d2 = vld1q_u16(dst + 48);
      d3 = vld1q_u16(dst + 56);

      d0 = vrhaddq_u16(s0, d0);
      d1 = vrhaddq_u16(s1, d1);
      d2 = vrhaddq_u16(s2, d2);
      d3 = vrhaddq_u16(s3, d3);

      vst1q_u16(dst + 32, d0);
      vst1q_u16(dst + 40, d1);
      vst1q_u16(dst + 48, d2);
      vst1q_u16(dst + 56, d3);
      src += src_stride;
      dst += dst_stride;
    } while (--h);
  }
}
