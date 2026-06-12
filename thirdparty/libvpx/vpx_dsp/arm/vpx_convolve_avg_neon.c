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

#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"

void vpx_convolve_avg_neon(const uint8_t *src, ptrdiff_t src_stride,
                           uint8_t *dst, ptrdiff_t dst_stride,
                           const InterpKernel *filter, int x0_q4, int x_step_q4,
                           int y0_q4, int y_step_q4, int w, int h) {
  (void)filter;
  (void)x0_q4;
  (void)x_step_q4;
  (void)y0_q4;
  (void)y_step_q4;

  if (w < 8) {  // avg4
    uint8x8_t s0, s1;
    uint8x8_t dd0 = vdup_n_u8(0);
    uint32x2x2_t s01;
    do {
      s0 = vld1_u8(src);
      src += src_stride;
      s1 = vld1_u8(src);
      src += src_stride;
      s01 = vzip_u32(vreinterpret_u32_u8(s0), vreinterpret_u32_u8(s1));
      dd0 = vreinterpret_u8_u32(
          vld1_lane_u32((const uint32_t *)dst, vreinterpret_u32_u8(dd0), 0));
      dd0 = vreinterpret_u8_u32(vld1_lane_u32(
          (const uint32_t *)(dst + dst_stride), vreinterpret_u32_u8(dd0), 1));
      dd0 = vrhadd_u8(vreinterpret_u8_u32(s01.val[0]), dd0);
      vst1_lane_u32((uint32_t *)dst, vreinterpret_u32_u8(dd0), 0);
      dst += dst_stride;
      vst1_lane_u32((uint32_t *)dst, vreinterpret_u32_u8(dd0), 1);
      dst += dst_stride;
      h -= 2;
    } while (h != 0);
  } else if (w == 8) {  // avg8
    uint8x8_t s0, s1, d0, d1;
    uint8x16_t s01, d01;
    do {
      s0 = vld1_u8(src);
      src += src_stride;
      s1 = vld1_u8(src);
      src += src_stride;
      d0 = vld1_u8(dst);
      d1 = vld1_u8(dst + dst_stride);

      s01 = vcombine_u8(s0, s1);
      d01 = vcombine_u8(d0, d1);
      d01 = vrhaddq_u8(s01, d01);

      vst1_u8(dst, vget_low_u8(d01));
      dst += dst_stride;
      vst1_u8(dst, vget_high_u8(d01));
      dst += dst_stride;
      h -= 2;
    } while (h != 0);
  } else if (w < 32) {  // avg16
    uint8x16_t s0, s1, d0, d1;
    do {
      s0 = vld1q_u8(src);
      src += src_stride;
      s1 = vld1q_u8(src);
      src += src_stride;
      d0 = vld1q_u8(dst);
      d1 = vld1q_u8(dst + dst_stride);

      d0 = vrhaddq_u8(s0, d0);
      d1 = vrhaddq_u8(s1, d1);

      vst1q_u8(dst, d0);
      dst += dst_stride;
      vst1q_u8(dst, d1);
      dst += dst_stride;
      h -= 2;
    } while (h != 0);
  } else if (w == 32) {  // avg32
    uint8x16_t s0, s1, s2, s3, d0, d1, d2, d3;
    do {
      s0 = vld1q_u8(src);
      s1 = vld1q_u8(src + 16);
      src += src_stride;
      s2 = vld1q_u8(src);
      s3 = vld1q_u8(src + 16);
      src += src_stride;
      d0 = vld1q_u8(dst);
      d1 = vld1q_u8(dst + 16);
      d2 = vld1q_u8(dst + dst_stride);
      d3 = vld1q_u8(dst + dst_stride + 16);

      d0 = vrhaddq_u8(s0, d0);
      d1 = vrhaddq_u8(s1, d1);
      d2 = vrhaddq_u8(s2, d2);
      d3 = vrhaddq_u8(s3, d3);

      vst1q_u8(dst, d0);
      vst1q_u8(dst + 16, d1);
      dst += dst_stride;
      vst1q_u8(dst, d2);
      vst1q_u8(dst + 16, d3);
      dst += dst_stride;
      h -= 2;
    } while (h != 0);
  } else {  // avg64
    uint8x16_t s0, s1, s2, s3, d0, d1, d2, d3;
    do {
      s0 = vld1q_u8(src);
      s1 = vld1q_u8(src + 16);
      s2 = vld1q_u8(src + 32);
      s3 = vld1q_u8(src + 48);
      src += src_stride;
      d0 = vld1q_u8(dst);
      d1 = vld1q_u8(dst + 16);
      d2 = vld1q_u8(dst + 32);
      d3 = vld1q_u8(dst + 48);

      d0 = vrhaddq_u8(s0, d0);
      d1 = vrhaddq_u8(s1, d1);
      d2 = vrhaddq_u8(s2, d2);
      d3 = vrhaddq_u8(s3, d3);

      vst1q_u8(dst, d0);
      vst1q_u8(dst + 16, d1);
      vst1q_u8(dst + 32, d2);
      vst1q_u8(dst + 48, d3);
      dst += dst_stride;
    } while (--h);
  }
}
