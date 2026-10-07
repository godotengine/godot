/*
 *  Copyright (c) 2026 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_ARM_SUBPEL_VARIANCE_NEON_H_
#define VPX_VPX_DSP_ARM_SUBPEL_VARIANCE_NEON_H_

#include "vpx_config.h"
#include "vpx_dsp/arm/mem_neon.h"

// Process a block exactly 4 wide and a multiple of 2 high.
static INLINE void var_filter_block2d_bil_w4(const uint8_t *src_ptr,
                                             uint8_t *dst_ptr, int src_stride,
                                             int pixel_step, int dst_height,
                                             int filter_offset) {
  const uint8x8_t f0 = vdup_n_u8(8 - filter_offset);
  const uint8x8_t f1 = vdup_n_u8(filter_offset);

  int i = dst_height;
  do {
    uint8x8_t s0 = load_unaligned_u8(src_ptr, src_stride);
    uint8x8_t s1 = load_unaligned_u8(src_ptr + pixel_step, src_stride);
    uint16x8_t blend = vmlal_u8(vmull_u8(s0, f0), s1, f1);
    uint8x8_t blend_u8 = vrshrn_n_u16(blend, 3);
    vst1_u8(dst_ptr, blend_u8);

    src_ptr += 2 * src_stride;
    dst_ptr += 2 * 4;
    i -= 2;
  } while (i != 0);
}

// Process a block exactly 8 wide and any height.
static INLINE void var_filter_block2d_bil_w8(const uint8_t *src_ptr,
                                             uint8_t *dst_ptr, int src_stride,
                                             int pixel_step, int dst_height,
                                             int filter_offset) {
  const uint8x8_t f0 = vdup_n_u8(8 - filter_offset);
  const uint8x8_t f1 = vdup_n_u8(filter_offset);

  int i = dst_height;
  do {
    uint8x8_t s0 = vld1_u8(src_ptr);
    uint8x8_t s1 = vld1_u8(src_ptr + pixel_step);
    uint16x8_t blend = vmlal_u8(vmull_u8(s0, f0), s1, f1);
    uint8x8_t blend_u8 = vrshrn_n_u16(blend, 3);
    vst1_u8(dst_ptr, blend_u8);

    src_ptr += src_stride;
    dst_ptr += 8;
  } while (--i != 0);
}

// Process a block which is a multiple of 16 wide and any height.
static INLINE void var_filter_block2d_bil_large(const uint8_t *src_ptr,
                                                uint8_t *dst_ptr,
                                                int src_stride, int pixel_step,
                                                int dst_width, int dst_height,
                                                int filter_offset) {
  const uint8x8_t f0 = vdup_n_u8(8 - filter_offset);
  const uint8x8_t f1 = vdup_n_u8(filter_offset);

  int i = dst_height;
  do {
    int j = 0;
    do {
      uint8x16_t s0 = vld1q_u8(src_ptr + j);
      uint8x16_t s1 = vld1q_u8(src_ptr + j + pixel_step);
      uint16x8_t blend_l =
          vmlal_u8(vmull_u8(vget_low_u8(s0), f0), vget_low_u8(s1), f1);
      uint16x8_t blend_h =
          vmlal_u8(vmull_u8(vget_high_u8(s0), f0), vget_high_u8(s1), f1);
      uint8x8_t out_lo = vrshrn_n_u16(blend_l, 3);
      uint8x8_t out_hi = vrshrn_n_u16(blend_h, 3);
      vst1q_u8(dst_ptr + j, vcombine_u8(out_lo, out_hi));

      j += 16;
    } while (j < dst_width);

    src_ptr += src_stride;
    dst_ptr += dst_width;
  } while (--i != 0);
}

static INLINE void var_filter_block2d_bil_w16(const uint8_t *src_ptr,
                                              uint8_t *dst_ptr, int src_stride,
                                              int pixel_step, int dst_height,
                                              int filter_offset) {
  var_filter_block2d_bil_large(src_ptr, dst_ptr, src_stride, pixel_step, 16,
                               dst_height, filter_offset);
}
static INLINE void var_filter_block2d_bil_w32(const uint8_t *src_ptr,
                                              uint8_t *dst_ptr, int src_stride,
                                              int pixel_step, int dst_height,
                                              int filter_offset) {
  var_filter_block2d_bil_large(src_ptr, dst_ptr, src_stride, pixel_step, 32,
                               dst_height, filter_offset);
}
static INLINE void var_filter_block2d_bil_w64(const uint8_t *src_ptr,
                                              uint8_t *dst_ptr, int src_stride,
                                              int pixel_step, int dst_height,
                                              int filter_offset) {
  var_filter_block2d_bil_large(src_ptr, dst_ptr, src_stride, pixel_step, 64,
                               dst_height, filter_offset);
}

static INLINE void var_filter_block2d_avg(const uint8_t *src_ptr,
                                          uint8_t *dst_ptr, int src_stride,
                                          int pixel_step, int dst_width,
                                          int dst_height) {
  int i = dst_height;

  // We only specialize on the filter values for large block sizes (>= 16x16.)
  assert(dst_width >= 16 && dst_width % 16 == 0);

  do {
    int j = 0;
    do {
      uint8x16_t s0 = vld1q_u8(src_ptr + j);
      uint8x16_t s1 = vld1q_u8(src_ptr + j + pixel_step);
      uint8x16_t avg = vrhaddq_u8(s0, s1);
      vst1q_u8(dst_ptr + j, avg);

      j += 16;
    } while (j < dst_width);

    src_ptr += src_stride;
    dst_ptr += dst_width;
  } while (--i != 0);
}

#endif  // VPX_VPX_DSP_ARM_SUBPEL_VARIANCE_NEON_H_
