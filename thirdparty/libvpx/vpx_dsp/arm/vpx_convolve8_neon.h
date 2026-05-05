/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_ARM_VPX_CONVOLVE8_NEON_H_
#define VPX_VPX_DSP_ARM_VPX_CONVOLVE8_NEON_H_

#include <arm_neon.h>

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/vpx_filter.h"

static INLINE int16x4_t convolve8_4(const int16x4_t s0, const int16x4_t s1,
                                    const int16x4_t s2, const int16x4_t s3,
                                    const int16x4_t s4, const int16x4_t s5,
                                    const int16x4_t s6, const int16x4_t s7,
                                    const int16x8_t filters) {
  const int16x4_t filters_lo = vget_low_s16(filters);
  const int16x4_t filters_hi = vget_high_s16(filters);
  int16x4_t sum;

  sum = vmul_lane_s16(s0, filters_lo, 0);
  sum = vmla_lane_s16(sum, s1, filters_lo, 1);
  sum = vmla_lane_s16(sum, s2, filters_lo, 2);
  sum = vmla_lane_s16(sum, s5, filters_hi, 1);
  sum = vmla_lane_s16(sum, s6, filters_hi, 2);
  sum = vmla_lane_s16(sum, s7, filters_hi, 3);
  sum = vqadd_s16(sum, vmul_lane_s16(s3, filters_lo, 3));
  sum = vqadd_s16(sum, vmul_lane_s16(s4, filters_hi, 0));
  return sum;
}

static INLINE uint8x8_t convolve8_8(const int16x8_t s0, const int16x8_t s1,
                                    const int16x8_t s2, const int16x8_t s3,
                                    const int16x8_t s4, const int16x8_t s5,
                                    const int16x8_t s6, const int16x8_t s7,
                                    const int16x8_t filters) {
  const int16x4_t filters_lo = vget_low_s16(filters);
  const int16x4_t filters_hi = vget_high_s16(filters);
  int16x8_t sum;

  sum = vmulq_lane_s16(s0, filters_lo, 0);
  sum = vmlaq_lane_s16(sum, s1, filters_lo, 1);
  sum = vmlaq_lane_s16(sum, s2, filters_lo, 2);
  sum = vmlaq_lane_s16(sum, s5, filters_hi, 1);
  sum = vmlaq_lane_s16(sum, s6, filters_hi, 2);
  sum = vmlaq_lane_s16(sum, s7, filters_hi, 3);
  sum = vqaddq_s16(sum, vmulq_lane_s16(s3, filters_lo, 3));
  sum = vqaddq_s16(sum, vmulq_lane_s16(s4, filters_hi, 0));
  return vqrshrun_n_s16(sum, FILTER_BITS);
}

static INLINE uint8x8_t scale_filter_8(const uint8x8_t *const s,
                                       const int16x8_t filters) {
  int16x8_t ss[8];

  ss[0] = vreinterpretq_s16_u16(vmovl_u8(s[0]));
  ss[1] = vreinterpretq_s16_u16(vmovl_u8(s[1]));
  ss[2] = vreinterpretq_s16_u16(vmovl_u8(s[2]));
  ss[3] = vreinterpretq_s16_u16(vmovl_u8(s[3]));
  ss[4] = vreinterpretq_s16_u16(vmovl_u8(s[4]));
  ss[5] = vreinterpretq_s16_u16(vmovl_u8(s[5]));
  ss[6] = vreinterpretq_s16_u16(vmovl_u8(s[6]));
  ss[7] = vreinterpretq_s16_u16(vmovl_u8(s[7]));

  return convolve8_8(ss[0], ss[1], ss[2], ss[3], ss[4], ss[5], ss[6], ss[7],
                     filters);
}

// 2-tap (bilinear) filter values are always positive, but 4-tap filter values
// are negative on the outer edges (taps 0 and 3), with taps 1 and 2 having much
// greater positive values to compensate. To use instructions that operate on
// 8-bit types we also need the types to be unsigned. Subtracting the products
// of taps 0 and 3 from the products of taps 1 and 2 always works given that
// 2-tap filters are 0-padded.
static INLINE uint8x8_t convolve4_8(const uint8x8_t s0, const uint8x8_t s1,
                                    const uint8x8_t s2, const uint8x8_t s3,
                                    const uint8x8_t filter_taps[4]) {
  uint16x8_t sum = vmull_u8(s1, filter_taps[1]);
  sum = vmlal_u8(sum, s2, filter_taps[2]);
  sum = vmlsl_u8(sum, s0, filter_taps[0]);
  sum = vmlsl_u8(sum, s3, filter_taps[3]);
  // We halved the filter values so -1 from right shift.
  return vqrshrun_n_s16(vreinterpretq_s16_u16(sum), FILTER_BITS - 1);
}

static INLINE void convolve_4tap_vert_neon(const uint8_t *src,
                                           ptrdiff_t src_stride, uint8_t *dst,
                                           ptrdiff_t dst_stride, int w, int h,
                                           const int16x8_t filter) {
  // 4-tap and bilinear filter values are even, so halve them to reduce
  // intermediate precision requirements.
  const uint8x8_t y_filter =
      vshrn_n_u16(vreinterpretq_u16_s16(vabsq_s16(filter)), 1);

  // Neon does not have lane-referencing multiply or multiply-accumulate
  // instructions that operate on vectors of 8-bit elements. This means we have
  // to duplicate filter taps into a whole vector and use standard multiply /
  // multiply-accumulate instructions.
  const uint8x8_t filter_taps[4] = { vdup_lane_u8(y_filter, 2),
                                     vdup_lane_u8(y_filter, 3),
                                     vdup_lane_u8(y_filter, 4),
                                     vdup_lane_u8(y_filter, 5) };

  if (w == 4) {
    uint8x8_t s01 = load_unaligned_u8(src + 0 * src_stride, src_stride);
    uint8x8_t s12 = load_unaligned_u8(src + 1 * src_stride, src_stride);

    src += 2 * src_stride;

    do {
      uint8x8_t s23 = load_unaligned_u8(src + 0 * src_stride, src_stride);
      uint8x8_t s34 = load_unaligned_u8(src + 1 * src_stride, src_stride);
      uint8x8_t s45 = load_unaligned_u8(src + 2 * src_stride, src_stride);
      uint8x8_t s56 = load_unaligned_u8(src + 3 * src_stride, src_stride);

      uint8x8_t d01 = convolve4_8(s01, s12, s23, s34, filter_taps);
      uint8x8_t d23 = convolve4_8(s23, s34, s45, s56, filter_taps);

      store_unaligned_u8(dst + 0 * dst_stride, dst_stride, d01);
      store_unaligned_u8(dst + 2 * dst_stride, dst_stride, d23);

      s01 = s45;
      s12 = s56;
      src += 4 * src_stride;
      dst += 4 * dst_stride;
      h -= 4;
    } while (h != 0);
  } else {
    do {
      const uint8_t *s = src;
      uint8_t *d = dst;
      int height = h;

      uint8x8_t s0, s1, s2;
      load_u8_8x3(s, src_stride, &s0, &s1, &s2);

      s += 3 * src_stride;

      do {
        uint8x8_t s3, s4, s5, s6;
        load_u8_8x4(s, src_stride, &s3, &s4, &s5, &s6);

        uint8x8_t d0 = convolve4_8(s0, s1, s2, s3, filter_taps);
        uint8x8_t d1 = convolve4_8(s1, s2, s3, s4, filter_taps);
        uint8x8_t d2 = convolve4_8(s2, s3, s4, s5, filter_taps);
        uint8x8_t d3 = convolve4_8(s3, s4, s5, s6, filter_taps);

        store_u8_8x4(d, dst_stride, d0, d1, d2, d3);

        s0 = s4;
        s1 = s5;
        s2 = s6;
        s += 4 * src_stride;
        d += 4 * dst_stride;
        height -= 4;
      } while (height != 0);
      src += 8;
      dst += 8;
      w -= 8;
    } while (w != 0);
  }
}

#endif  // VPX_VPX_DSP_ARM_VPX_CONVOLVE8_NEON_H_
