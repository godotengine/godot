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

void vpx_convolve_copy_neon(
    const uint8_t *src,    // r0
    ptrdiff_t src_stride,  // r1
    uint8_t *dst,          // r2
    ptrdiff_t dst_stride,  // r3
    const int16_t *filter_x,
    int filter_x_stride,
    const int16_t *filter_y,
    int filter_y_stride,
    int w,
    int h) {
  uint8x8_t d0u8, d2u8;
  uint8x16_t q0u8, q1u8, q2u8, q3u8;
  (void)filter_x;  (void)filter_x_stride;
  (void)filter_y;  (void)filter_y_stride;

  if (w > 32) {  // copy64
    for (; h > 0; h--) {
      q0u8 = vld1q_u8(src);
      q1u8 = vld1q_u8(src + 16);
      q2u8 = vld1q_u8(src + 32);
      q3u8 = vld1q_u8(src + 48);
      src += src_stride;

      vst1q_u8(dst, q0u8);
      vst1q_u8(dst + 16, q1u8);
      vst1q_u8(dst + 32, q2u8);
      vst1q_u8(dst + 48, q3u8);
      dst += dst_stride;
    }
  } else if (w == 32) {  // copy32
    for (; h > 0; h -= 2) {
      q0u8 = vld1q_u8(src);
      q1u8 = vld1q_u8(src + 16);
      src += src_stride;
      q2u8 = vld1q_u8(src);
      q3u8 = vld1q_u8(src + 16);
      src += src_stride;

      vst1q_u8(dst, q0u8);
      vst1q_u8(dst + 16, q1u8);
      dst += dst_stride;
      vst1q_u8(dst, q2u8);
      vst1q_u8(dst + 16, q3u8);
      dst += dst_stride;
    }
  } else if (w > 8) {  // copy16
    for (; h > 0; h -= 2) {
      q0u8 = vld1q_u8(src);
      src += src_stride;
      q1u8 = vld1q_u8(src);
      src += src_stride;

      vst1q_u8(dst, q0u8);
      dst += dst_stride;
      vst1q_u8(dst, q1u8);
      dst += dst_stride;
    }
  } else if (w == 8) {  // copy8
    for (; h > 0; h -= 2) {
      d0u8 = vld1_u8(src);
      src += src_stride;
      d2u8 = vld1_u8(src);
      src += src_stride;

      vst1_u8(dst, d0u8);
      dst += dst_stride;
      vst1_u8(dst, d2u8);
      dst += dst_stride;
    }
  } else {  // copy4
    for (; h > 0; h--) {
      *(uint32_t *)dst = *(const uint32_t *)src;
      src += src_stride;
      dst += dst_stride;
    }
  }
  return;
}
