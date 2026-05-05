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
#include <assert.h>

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/arm/highbd_convolve8_neon.h"
#include "vpx_dsp/arm/mem_neon.h"
#include "vpx_dsp/arm/transpose_neon.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_dsp/vpx_filter.h"
#include "vpx_ports/mem.h"

static INLINE uint16x4_t
highbd_convolve8_4(const int16x4_t s0, const int16x4_t s1, const int16x4_t s2,
                   const int16x4_t s3, const int16x4_t s4, const int16x4_t s5,
                   const int16x4_t s6, const int16x4_t s7,
                   const int16x8_t filters, const uint16x4_t max) {
  const int16x4_t filters_lo = vget_low_s16(filters);
  const int16x4_t filters_hi = vget_high_s16(filters);

  int32x4_t sum = vmull_lane_s16(s0, filters_lo, 0);
  sum = vmlal_lane_s16(sum, s1, filters_lo, 1);
  sum = vmlal_lane_s16(sum, s2, filters_lo, 2);
  sum = vmlal_lane_s16(sum, s3, filters_lo, 3);
  sum = vmlal_lane_s16(sum, s4, filters_hi, 0);
  sum = vmlal_lane_s16(sum, s5, filters_hi, 1);
  sum = vmlal_lane_s16(sum, s6, filters_hi, 2);
  sum = vmlal_lane_s16(sum, s7, filters_hi, 3);

  uint16x4_t res = vqrshrun_n_s32(sum, FILTER_BITS);
  return vmin_u16(res, max);
}

static INLINE uint16x8_t
highbd_convolve8_8(const int16x8_t s0, const int16x8_t s1, const int16x8_t s2,
                   const int16x8_t s3, const int16x8_t s4, const int16x8_t s5,
                   const int16x8_t s6, const int16x8_t s7,
                   const int16x8_t filters, const uint16x8_t max) {
  const int16x4_t filters_lo = vget_low_s16(filters);
  const int16x4_t filters_hi = vget_high_s16(filters);

  int32x4_t sum0 = vmull_lane_s16(vget_low_s16(s0), filters_lo, 0);
  sum0 = vmlal_lane_s16(sum0, vget_low_s16(s1), filters_lo, 1);
  sum0 = vmlal_lane_s16(sum0, vget_low_s16(s2), filters_lo, 2);
  sum0 = vmlal_lane_s16(sum0, vget_low_s16(s3), filters_lo, 3);
  sum0 = vmlal_lane_s16(sum0, vget_low_s16(s4), filters_hi, 0);
  sum0 = vmlal_lane_s16(sum0, vget_low_s16(s5), filters_hi, 1);
  sum0 = vmlal_lane_s16(sum0, vget_low_s16(s6), filters_hi, 2);
  sum0 = vmlal_lane_s16(sum0, vget_low_s16(s7), filters_hi, 3);

  int32x4_t sum1 = vmull_lane_s16(vget_high_s16(s0), filters_lo, 0);
  sum1 = vmlal_lane_s16(sum1, vget_high_s16(s1), filters_lo, 1);
  sum1 = vmlal_lane_s16(sum1, vget_high_s16(s2), filters_lo, 2);
  sum1 = vmlal_lane_s16(sum1, vget_high_s16(s3), filters_lo, 3);
  sum1 = vmlal_lane_s16(sum1, vget_high_s16(s4), filters_hi, 0);
  sum1 = vmlal_lane_s16(sum1, vget_high_s16(s5), filters_hi, 1);
  sum1 = vmlal_lane_s16(sum1, vget_high_s16(s6), filters_hi, 2);
  sum1 = vmlal_lane_s16(sum1, vget_high_s16(s7), filters_hi, 3);

  uint16x8_t res = vcombine_u16(vqrshrun_n_s32(sum0, FILTER_BITS),
                                vqrshrun_n_s32(sum1, FILTER_BITS));
  return vminq_u16(res, max);
}

static INLINE void highbd_convolve_4tap_horiz_neon(
    const uint16_t *src, ptrdiff_t src_stride, uint16_t *dst,
    ptrdiff_t dst_stride, int w, int h, const int16x4_t filter, int bd) {
  if (w == 4) {
    const uint16x4_t max = vdup_n_u16((1 << bd) - 1);
    const int16_t *s = (const int16_t *)src;
    uint16_t *d = dst;

    do {
      int16x4_t s0[4], s1[4], s2[4], s3[4];
      load_s16_4x4(s + 0 * src_stride, 1, &s0[0], &s0[1], &s0[2], &s0[3]);
      load_s16_4x4(s + 1 * src_stride, 1, &s1[0], &s1[1], &s1[2], &s1[3]);
      load_s16_4x4(s + 2 * src_stride, 1, &s2[0], &s2[1], &s2[2], &s2[3]);
      load_s16_4x4(s + 3 * src_stride, 1, &s3[0], &s3[1], &s3[2], &s3[3]);

      uint16x4_t d0 =
          highbd_convolve4_4_neon(s0[0], s0[1], s0[2], s0[3], filter, max);
      uint16x4_t d1 =
          highbd_convolve4_4_neon(s1[0], s1[1], s1[2], s1[3], filter, max);
      uint16x4_t d2 =
          highbd_convolve4_4_neon(s2[0], s2[1], s2[2], s2[3], filter, max);
      uint16x4_t d3 =
          highbd_convolve4_4_neon(s3[0], s3[1], s3[2], s3[3], filter, max);

      store_u16_4x4(d, dst_stride, d0, d1, d2, d3);

      s += 4 * src_stride;
      d += 4 * dst_stride;
      h -= 4;
    } while (h != 0);
  } else {
    const uint16x8_t max = vdupq_n_u16((1 << bd) - 1);

    do {
      const int16_t *s = (const int16_t *)src;
      uint16_t *d = dst;
      int width = w;

      do {
        int16x8_t s0[4], s1[4], s2[4], s3[4];
        load_s16_8x4(s + 0 * src_stride, 1, &s0[0], &s0[1], &s0[2], &s0[3]);
        load_s16_8x4(s + 1 * src_stride, 1, &s1[0], &s1[1], &s1[2], &s1[3]);
        load_s16_8x4(s + 2 * src_stride, 1, &s2[0], &s2[1], &s2[2], &s2[3]);
        load_s16_8x4(s + 3 * src_stride, 1, &s3[0], &s3[1], &s3[2], &s3[3]);

        uint16x8_t d0 =
            highbd_convolve4_8_neon(s0[0], s0[1], s0[2], s0[3], filter, max);
        uint16x8_t d1 =
            highbd_convolve4_8_neon(s1[0], s1[1], s1[2], s1[3], filter, max);
        uint16x8_t d2 =
            highbd_convolve4_8_neon(s2[0], s2[1], s2[2], s2[3], filter, max);
        uint16x8_t d3 =
            highbd_convolve4_8_neon(s3[0], s3[1], s3[2], s3[3], filter, max);

        store_u16_8x4(d, dst_stride, d0, d1, d2, d3);

        s += 8;
        d += 8;
        width -= 8;
      } while (width != 0);
      src += 4 * src_stride;
      dst += 4 * dst_stride;
      h -= 4;
    } while (h != 0);
  }
}

static INLINE void highbd_convolve_8tap_horiz_neon(
    const uint16_t *src, ptrdiff_t src_stride, uint16_t *dst,
    ptrdiff_t dst_stride, int w, int h, const int16x8_t filter, int bd) {
  if (w == 4) {
    const uint16x4_t max = vdup_n_u16((1 << bd) - 1);
    const int16_t *s = (const int16_t *)src;
    uint16_t *d = dst;

    do {
      int16x4_t s0[8], s1[8], s2[8], s3[8];
      load_s16_4x8(s + 0 * src_stride, 1, &s0[0], &s0[1], &s0[2], &s0[3],
                   &s0[4], &s0[5], &s0[6], &s0[7]);
      load_s16_4x8(s + 1 * src_stride, 1, &s1[0], &s1[1], &s1[2], &s1[3],
                   &s1[4], &s1[5], &s1[6], &s1[7]);
      load_s16_4x8(s + 2 * src_stride, 1, &s2[0], &s2[1], &s2[2], &s2[3],
                   &s2[4], &s2[5], &s2[6], &s2[7]);
      load_s16_4x8(s + 3 * src_stride, 1, &s3[0], &s3[1], &s3[2], &s3[3],
                   &s3[4], &s3[5], &s3[6], &s3[7]);

      uint16x4_t d0 = highbd_convolve8_4(s0[0], s0[1], s0[2], s0[3], s0[4],
                                         s0[5], s0[6], s0[7], filter, max);
      uint16x4_t d1 = highbd_convolve8_4(s1[0], s1[1], s1[2], s1[3], s1[4],
                                         s1[5], s1[6], s1[7], filter, max);
      uint16x4_t d2 = highbd_convolve8_4(s2[0], s2[1], s2[2], s2[3], s2[4],
                                         s2[5], s2[6], s2[7], filter, max);
      uint16x4_t d3 = highbd_convolve8_4(s3[0], s3[1], s3[2], s3[3], s3[4],
                                         s3[5], s3[6], s3[7], filter, max);

      store_u16_4x4(d, dst_stride, d0, d1, d2, d3);

      s += 4 * src_stride;
      d += 4 * dst_stride;
      h -= 4;
    } while (h != 0);
  } else {
    const uint16x8_t max = vdupq_n_u16((1 << bd) - 1);

    do {
      const int16_t *s = (const int16_t *)src;
      uint16_t *d = dst;
      int width = w;

      do {
        int16x8_t s0[8], s1[8], s2[8], s3[8];
        load_s16_8x8(s + 0 * src_stride, 1, &s0[0], &s0[1], &s0[2], &s0[3],
                     &s0[4], &s0[5], &s0[6], &s0[7]);
        load_s16_8x8(s + 1 * src_stride, 1, &s1[0], &s1[1], &s1[2], &s1[3],
                     &s1[4], &s1[5], &s1[6], &s1[7]);
        load_s16_8x8(s + 2 * src_stride, 1, &s2[0], &s2[1], &s2[2], &s2[3],
                     &s2[4], &s2[5], &s2[6], &s2[7]);
        load_s16_8x8(s + 3 * src_stride, 1, &s3[0], &s3[1], &s3[2], &s3[3],
                     &s3[4], &s3[5], &s3[6], &s3[7]);

        uint16x8_t d0 = highbd_convolve8_8(s0[0], s0[1], s0[2], s0[3], s0[4],
                                           s0[5], s0[6], s0[7], filter, max);
        uint16x8_t d1 = highbd_convolve8_8(s1[0], s1[1], s1[2], s1[3], s1[4],
                                           s1[5], s1[6], s1[7], filter, max);
        uint16x8_t d2 = highbd_convolve8_8(s2[0], s2[1], s2[2], s2[3], s2[4],
                                           s2[5], s2[6], s2[7], filter, max);
        uint16x8_t d3 = highbd_convolve8_8(s3[0], s3[1], s3[2], s3[3], s3[4],
                                           s3[5], s3[6], s3[7], filter, max);

        store_u16_8x4(d, dst_stride, d0, d1, d2, d3);

        s += 8;
        d += 8;
        width -= 8;
      } while (width != 0);
      src += 4 * src_stride;
      dst += 4 * dst_stride;
      h -= 4;
    } while (h != 0);
  }
}

void vpx_highbd_convolve8_horiz_neon(const uint16_t *src, ptrdiff_t src_stride,
                                     uint16_t *dst, ptrdiff_t dst_stride,
                                     const InterpKernel *filter, int x0_q4,
                                     int x_step_q4, int y0_q4, int y_step_q4,
                                     int w, int h, int bd) {
  if (x_step_q4 != 16) {
    vpx_highbd_convolve8_horiz_c(src, src_stride, dst, dst_stride, filter,
                                 x0_q4, x_step_q4, y0_q4, y_step_q4, w, h, bd);
    return;
  }

  assert((intptr_t)dst % 4 == 0);
  assert(dst_stride % 4 == 0);
  assert(x_step_q4 == 16);

  (void)x_step_q4;
  (void)y0_q4;
  (void)y_step_q4;

  if (vpx_get_filter_taps(filter[x0_q4]) <= 4) {
    const int16x4_t x_filter_4tap = vld1_s16(filter[x0_q4] + 2);
    highbd_convolve_4tap_horiz_neon(src - 1, src_stride, dst, dst_stride, w, h,
                                    x_filter_4tap, bd);
  } else {
    const int16x8_t x_filter_8tap = vld1q_s16(filter[x0_q4]);
    highbd_convolve_8tap_horiz_neon(src - 3, src_stride, dst, dst_stride, w, h,
                                    x_filter_8tap, bd);
  }
}

void vpx_highbd_convolve8_avg_horiz_neon(const uint16_t *src,
                                         ptrdiff_t src_stride, uint16_t *dst,
                                         ptrdiff_t dst_stride,
                                         const InterpKernel *filter, int x0_q4,
                                         int x_step_q4, int y0_q4,
                                         int y_step_q4, int w, int h, int bd) {
  if (x_step_q4 != 16) {
    vpx_highbd_convolve8_avg_horiz_c(src, src_stride, dst, dst_stride, filter,
                                     x0_q4, x_step_q4, y0_q4, y_step_q4, w, h,
                                     bd);
    return;
  }

  assert((intptr_t)dst % 4 == 0);
  assert(dst_stride % 4 == 0);

  const int16x8_t filters = vld1q_s16(filter[x0_q4]);

  src -= 3;

  if (w == 4) {
    const uint16x4_t max = vdup_n_u16((1 << bd) - 1);
    const int16_t *s = (const int16_t *)src;
    uint16_t *d = dst;

    do {
      int16x4_t s0[8], s1[8], s2[8], s3[8];
      load_s16_4x8(s + 0 * src_stride, 1, &s0[0], &s0[1], &s0[2], &s0[3],
                   &s0[4], &s0[5], &s0[6], &s0[7]);
      load_s16_4x8(s + 1 * src_stride, 1, &s1[0], &s1[1], &s1[2], &s1[3],
                   &s1[4], &s1[5], &s1[6], &s1[7]);
      load_s16_4x8(s + 2 * src_stride, 1, &s2[0], &s2[1], &s2[2], &s2[3],
                   &s2[4], &s2[5], &s2[6], &s2[7]);
      load_s16_4x8(s + 3 * src_stride, 1, &s3[0], &s3[1], &s3[2], &s3[3],
                   &s3[4], &s3[5], &s3[6], &s3[7]);

      uint16x4_t d0 = highbd_convolve8_4(s0[0], s0[1], s0[2], s0[3], s0[4],
                                         s0[5], s0[6], s0[7], filters, max);
      uint16x4_t d1 = highbd_convolve8_4(s1[0], s1[1], s1[2], s1[3], s1[4],
                                         s1[5], s1[6], s1[7], filters, max);
      uint16x4_t d2 = highbd_convolve8_4(s2[0], s2[1], s2[2], s2[3], s2[4],
                                         s2[5], s2[6], s2[7], filters, max);
      uint16x4_t d3 = highbd_convolve8_4(s3[0], s3[1], s3[2], s3[3], s3[4],
                                         s3[5], s3[6], s3[7], filters, max);

      d0 = vrhadd_u16(d0, vld1_u16(d + 0 * dst_stride));
      d1 = vrhadd_u16(d1, vld1_u16(d + 1 * dst_stride));
      d2 = vrhadd_u16(d2, vld1_u16(d + 2 * dst_stride));
      d3 = vrhadd_u16(d3, vld1_u16(d + 3 * dst_stride));

      store_u16_4x4(d, dst_stride, d0, d1, d2, d3);

      s += 4 * src_stride;
      d += 4 * dst_stride;
      h -= 4;
    } while (h != 0);
  } else {
    const uint16x8_t max = vdupq_n_u16((1 << bd) - 1);

    do {
      const int16_t *s = (const int16_t *)src;
      uint16_t *d = dst;
      int width = w;

      do {
        int16x8_t s0[8], s1[8], s2[8], s3[8];
        load_s16_8x8(s + 0 * src_stride, 1, &s0[0], &s0[1], &s0[2], &s0[3],
                     &s0[4], &s0[5], &s0[6], &s0[7]);
        load_s16_8x8(s + 1 * src_stride, 1, &s1[0], &s1[1], &s1[2], &s1[3],
                     &s1[4], &s1[5], &s1[6], &s1[7]);
        load_s16_8x8(s + 2 * src_stride, 1, &s2[0], &s2[1], &s2[2], &s2[3],
                     &s2[4], &s2[5], &s2[6], &s2[7]);
        load_s16_8x8(s + 3 * src_stride, 1, &s3[0], &s3[1], &s3[2], &s3[3],
                     &s3[4], &s3[5], &s3[6], &s3[7]);

        uint16x8_t d0 = highbd_convolve8_8(s0[0], s0[1], s0[2], s0[3], s0[4],
                                           s0[5], s0[6], s0[7], filters, max);
        uint16x8_t d1 = highbd_convolve8_8(s1[0], s1[1], s1[2], s1[3], s1[4],
                                           s1[5], s1[6], s1[7], filters, max);
        uint16x8_t d2 = highbd_convolve8_8(s2[0], s2[1], s2[2], s2[3], s2[4],
                                           s2[5], s2[6], s2[7], filters, max);
        uint16x8_t d3 = highbd_convolve8_8(s3[0], s3[1], s3[2], s3[3], s3[4],
                                           s3[5], s3[6], s3[7], filters, max);

        d0 = vrhaddq_u16(d0, vld1q_u16(d + 0 * dst_stride));
        d1 = vrhaddq_u16(d1, vld1q_u16(d + 1 * dst_stride));
        d2 = vrhaddq_u16(d2, vld1q_u16(d + 2 * dst_stride));
        d3 = vrhaddq_u16(d3, vld1q_u16(d + 3 * dst_stride));

        store_u16_8x4(d, dst_stride, d0, d1, d2, d3);

        s += 8;
        d += 8;
        width -= 8;
      } while (width != 0);
      src += 4 * src_stride;
      dst += 4 * dst_stride;
      h -= 4;
    } while (h != 0);
  }
}

static INLINE void highbd_convolve_4tap_vert_neon(
    const uint16_t *src, ptrdiff_t src_stride, uint16_t *dst,
    ptrdiff_t dst_stride, int w, int h, const int16x4_t filter, int bd) {
  if (w == 4) {
    const uint16x4_t max = vdup_n_u16((1 << bd) - 1);
    const int16_t *s = (const int16_t *)src;
    uint16_t *d = dst;

    int16x4_t s0, s1, s2;
    load_s16_4x3(s, src_stride, &s0, &s1, &s2);

    s += 3 * src_stride;

    do {
      int16x4_t s3, s4, s5, s6;
      load_s16_4x4(s, src_stride, &s3, &s4, &s5, &s6);

      uint16x4_t d0 = highbd_convolve4_4_neon(s0, s1, s2, s3, filter, max);
      uint16x4_t d1 = highbd_convolve4_4_neon(s1, s2, s3, s4, filter, max);
      uint16x4_t d2 = highbd_convolve4_4_neon(s2, s3, s4, s5, filter, max);
      uint16x4_t d3 = highbd_convolve4_4_neon(s3, s4, s5, s6, filter, max);

      store_u16_4x4(d, dst_stride, d0, d1, d2, d3);

      s0 = s4;
      s1 = s5;
      s2 = s6;
      s += 4 * src_stride;
      d += 4 * dst_stride;
      h -= 4;
    } while (h != 0);
  } else {
    const uint16x8_t max = vdupq_n_u16((1 << bd) - 1);

    do {
      const int16_t *s = (const int16_t *)src;
      uint16_t *d = dst;
      int height = h;

      int16x8_t s0, s1, s2;
      load_s16_8x3(s, src_stride, &s0, &s1, &s2);

      s += 3 * src_stride;

      do {
        int16x8_t s3, s4, s5, s6;
        load_s16_8x4(s, src_stride, &s3, &s4, &s5, &s6);

        uint16x8_t d0 = highbd_convolve4_8_neon(s0, s1, s2, s3, filter, max);
        uint16x8_t d1 = highbd_convolve4_8_neon(s1, s2, s3, s4, filter, max);
        uint16x8_t d2 = highbd_convolve4_8_neon(s2, s3, s4, s5, filter, max);
        uint16x8_t d3 = highbd_convolve4_8_neon(s3, s4, s5, s6, filter, max);

        store_u16_8x4(d, dst_stride, d0, d1, d2, d3);

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

static INLINE void highbd_convolve_8tap_vert_neon(
    const uint16_t *src, ptrdiff_t src_stride, uint16_t *dst,
    ptrdiff_t dst_stride, int w, int h, const int16x8_t filter, int bd) {
  if (w == 4) {
    const uint16x4_t max = vdup_n_u16((1 << bd) - 1);
    const int16_t *s = (const int16_t *)src;
    uint16_t *d = dst;

    int16x4_t s0, s1, s2, s3, s4, s5, s6;
    load_s16_4x7(s, src_stride, &s0, &s1, &s2, &s3, &s4, &s5, &s6);

    s += 7 * src_stride;

    do {
      int16x4_t s7, s8, s9, s10;
      load_s16_4x4(s, src_stride, &s7, &s8, &s9, &s10);

      uint16x4_t d0 =
          highbd_convolve8_4(s0, s1, s2, s3, s4, s5, s6, s7, filter, max);
      uint16x4_t d1 =
          highbd_convolve8_4(s1, s2, s3, s4, s5, s6, s7, s8, filter, max);
      uint16x4_t d2 =
          highbd_convolve8_4(s2, s3, s4, s5, s6, s7, s8, s9, filter, max);
      uint16x4_t d3 =
          highbd_convolve8_4(s3, s4, s5, s6, s7, s8, s9, s10, filter, max);

      store_u16_4x4(d, dst_stride, d0, d1, d2, d3);

      s0 = s4;
      s1 = s5;
      s2 = s6;
      s3 = s7;
      s4 = s8;
      s5 = s9;
      s6 = s10;
      s += 4 * src_stride;
      d += 4 * dst_stride;
      h -= 4;
    } while (h != 0);
  } else {
    const uint16x8_t max = vdupq_n_u16((1 << bd) - 1);

    do {
      const int16_t *s = (const int16_t *)src;
      uint16_t *d = dst;
      int height = h;

      int16x8_t s0, s1, s2, s3, s4, s5, s6;
      load_s16_8x7(s, src_stride, &s0, &s1, &s2, &s3, &s4, &s5, &s6);

      s += 7 * src_stride;

      do {
        int16x8_t s7, s8, s9, s10;
        load_s16_8x4(s, src_stride, &s7, &s8, &s9, &s10);

        uint16x8_t d0 =
            highbd_convolve8_8(s0, s1, s2, s3, s4, s5, s6, s7, filter, max);
        uint16x8_t d1 =
            highbd_convolve8_8(s1, s2, s3, s4, s5, s6, s7, s8, filter, max);
        uint16x8_t d2 =
            highbd_convolve8_8(s2, s3, s4, s5, s6, s7, s8, s9, filter, max);
        uint16x8_t d3 =
            highbd_convolve8_8(s3, s4, s5, s6, s7, s8, s9, s10, filter, max);

        store_u16_8x4(d, dst_stride, d0, d1, d2, d3);

        s0 = s4;
        s1 = s5;
        s2 = s6;
        s3 = s7;
        s4 = s8;
        s5 = s9;
        s6 = s10;
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

void vpx_highbd_convolve8_vert_neon(const uint16_t *src, ptrdiff_t src_stride,
                                    uint16_t *dst, ptrdiff_t dst_stride,
                                    const InterpKernel *filter, int x0_q4,
                                    int x_step_q4, int y0_q4, int y_step_q4,
                                    int w, int h, int bd) {
  if (y_step_q4 != 16) {
    vpx_highbd_convolve8_vert_c(src, src_stride, dst, dst_stride, filter, x0_q4,
                                x_step_q4, y0_q4, y_step_q4, w, h, bd);
    return;
  }

  assert((intptr_t)dst % 4 == 0);
  assert(dst_stride % 4 == 0);
  assert(y_step_q4 == 16);

  (void)x_step_q4;
  (void)y0_q4;
  (void)y_step_q4;

  if (vpx_get_filter_taps(filter[y0_q4]) <= 4) {
    const int16x4_t y_filter_4tap = vld1_s16(filter[y0_q4] + 2);
    highbd_convolve_4tap_vert_neon(src - src_stride, src_stride, dst,
                                   dst_stride, w, h, y_filter_4tap, bd);
  } else {
    const int16x8_t y_filter_8tap = vld1q_s16(filter[y0_q4]);
    highbd_convolve_8tap_vert_neon(src - 3 * src_stride, src_stride, dst,
                                   dst_stride, w, h, y_filter_8tap, bd);
  }
}

void vpx_highbd_convolve8_avg_vert_neon(const uint16_t *src,
                                        ptrdiff_t src_stride, uint16_t *dst,
                                        ptrdiff_t dst_stride,
                                        const InterpKernel *filter, int x0_q4,
                                        int x_step_q4, int y0_q4, int y_step_q4,
                                        int w, int h, int bd) {
  if (y_step_q4 != 16) {
    vpx_highbd_convolve8_avg_vert_c(src, src_stride, dst, dst_stride, filter,
                                    x0_q4, x_step_q4, y0_q4, y_step_q4, w, h,
                                    bd);
    return;
  }

  assert((intptr_t)dst % 4 == 0);
  assert(dst_stride % 4 == 0);

  const int16x8_t filters = vld1q_s16(filter[y0_q4]);

  src -= 3 * src_stride;

  if (w == 4) {
    const uint16x4_t max = vdup_n_u16((1 << bd) - 1);
    const int16_t *s = (const int16_t *)src;
    uint16_t *d = dst;

    int16x4_t s0, s1, s2, s3, s4, s5, s6;
    load_s16_4x7(s, src_stride, &s0, &s1, &s2, &s3, &s4, &s5, &s6);

    s += 7 * src_stride;

    do {
      int16x4_t s7, s8, s9, s10;
      load_s16_4x4(s, src_stride, &s7, &s8, &s9, &s10);

      uint16x4_t d0 =
          highbd_convolve8_4(s0, s1, s2, s3, s4, s5, s6, s7, filters, max);
      uint16x4_t d1 =
          highbd_convolve8_4(s1, s2, s3, s4, s5, s6, s7, s8, filters, max);
      uint16x4_t d2 =
          highbd_convolve8_4(s2, s3, s4, s5, s6, s7, s8, s9, filters, max);
      uint16x4_t d3 =
          highbd_convolve8_4(s3, s4, s5, s6, s7, s8, s9, s10, filters, max);

      d0 = vrhadd_u16(d0, vld1_u16(d + 0 * dst_stride));
      d1 = vrhadd_u16(d1, vld1_u16(d + 1 * dst_stride));
      d2 = vrhadd_u16(d2, vld1_u16(d + 2 * dst_stride));
      d3 = vrhadd_u16(d3, vld1_u16(d + 3 * dst_stride));

      store_u16_4x4(d, dst_stride, d0, d1, d2, d3);

      s0 = s4;
      s1 = s5;
      s2 = s6;
      s3 = s7;
      s4 = s8;
      s5 = s9;
      s6 = s10;
      s += 4 * src_stride;
      d += 4 * dst_stride;
      h -= 4;
    } while (h != 0);
  } else {
    const uint16x8_t max = vdupq_n_u16((1 << bd) - 1);

    do {
      const int16_t *s = (const int16_t *)src;
      uint16_t *d = dst;
      int height = h;

      int16x8_t s0, s1, s2, s3, s4, s5, s6;
      load_s16_8x7(s, src_stride, &s0, &s1, &s2, &s3, &s4, &s5, &s6);

      s += 7 * src_stride;

      do {
        int16x8_t s7, s8, s9, s10;
        load_s16_8x4(s, src_stride, &s7, &s8, &s9, &s10);

        uint16x8_t d0 =
            highbd_convolve8_8(s0, s1, s2, s3, s4, s5, s6, s7, filters, max);
        uint16x8_t d1 =
            highbd_convolve8_8(s1, s2, s3, s4, s5, s6, s7, s8, filters, max);
        uint16x8_t d2 =
            highbd_convolve8_8(s2, s3, s4, s5, s6, s7, s8, s9, filters, max);
        uint16x8_t d3 =
            highbd_convolve8_8(s3, s4, s5, s6, s7, s8, s9, s10, filters, max);

        d0 = vrhaddq_u16(d0, vld1q_u16(d + 0 * dst_stride));
        d1 = vrhaddq_u16(d1, vld1q_u16(d + 1 * dst_stride));
        d2 = vrhaddq_u16(d2, vld1q_u16(d + 2 * dst_stride));
        d3 = vrhaddq_u16(d3, vld1q_u16(d + 3 * dst_stride));

        store_u16_8x4(d, dst_stride, d0, d1, d2, d3);

        s0 = s4;
        s1 = s5;
        s2 = s6;
        s3 = s7;
        s4 = s8;
        s5 = s9;
        s6 = s10;
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

static INLINE void highbd_convolve_2d_4tap_neon(
    const uint16_t *src, ptrdiff_t src_stride, uint16_t *dst,
    ptrdiff_t dst_stride, int w, int h, const int16x4_t x_filter,
    const int16x4_t y_filter, int bd) {
  if (w == 4) {
    const uint16x4_t max = vdup_n_u16((1 << bd) - 1);
    const int16_t *s = (const int16_t *)src;
    uint16_t *d = dst;

    int16x4_t h_s0[4], h_s1[4], h_s2[4];
    load_s16_4x4(s + 0 * src_stride, 1, &h_s0[0], &h_s0[1], &h_s0[2], &h_s0[3]);
    load_s16_4x4(s + 1 * src_stride, 1, &h_s1[0], &h_s1[1], &h_s1[2], &h_s1[3]);
    load_s16_4x4(s + 2 * src_stride, 1, &h_s2[0], &h_s2[1], &h_s2[2], &h_s2[3]);

    int16x4_t v_s0 = vreinterpret_s16_u16(highbd_convolve4_4_neon(
        h_s0[0], h_s0[1], h_s0[2], h_s0[3], x_filter, max));
    int16x4_t v_s1 = vreinterpret_s16_u16(highbd_convolve4_4_neon(
        h_s1[0], h_s1[1], h_s1[2], h_s1[3], x_filter, max));
    int16x4_t v_s2 = vreinterpret_s16_u16(highbd_convolve4_4_neon(
        h_s2[0], h_s2[1], h_s2[2], h_s2[3], x_filter, max));

    s += 3 * src_stride;

    do {
      int16x4_t h_s3[4], h_s4[4], h_s5[4], h_s6[4];
      load_s16_4x4(s + 0 * src_stride, 1, &h_s3[0], &h_s3[1], &h_s3[2],
                   &h_s3[3]);
      load_s16_4x4(s + 1 * src_stride, 1, &h_s4[0], &h_s4[1], &h_s4[2],
                   &h_s4[3]);
      load_s16_4x4(s + 2 * src_stride, 1, &h_s5[0], &h_s5[1], &h_s5[2],
                   &h_s5[3]);
      load_s16_4x4(s + 3 * src_stride, 1, &h_s6[0], &h_s6[1], &h_s6[2],
                   &h_s6[3]);

      int16x4_t v_s3 = vreinterpret_s16_u16(highbd_convolve4_4_neon(
          h_s3[0], h_s3[1], h_s3[2], h_s3[3], x_filter, max));
      int16x4_t v_s4 = vreinterpret_s16_u16(highbd_convolve4_4_neon(
          h_s4[0], h_s4[1], h_s4[2], h_s4[3], x_filter, max));
      int16x4_t v_s5 = vreinterpret_s16_u16(highbd_convolve4_4_neon(
          h_s5[0], h_s5[1], h_s5[2], h_s5[3], x_filter, max));
      int16x4_t v_s6 = vreinterpret_s16_u16(highbd_convolve4_4_neon(
          h_s6[0], h_s6[1], h_s6[2], h_s6[3], x_filter, max));

      uint16x4_t d0 =
          highbd_convolve4_4_neon(v_s0, v_s1, v_s2, v_s3, y_filter, max);
      uint16x4_t d1 =
          highbd_convolve4_4_neon(v_s1, v_s2, v_s3, v_s4, y_filter, max);
      uint16x4_t d2 =
          highbd_convolve4_4_neon(v_s2, v_s3, v_s4, v_s5, y_filter, max);
      uint16x4_t d3 =
          highbd_convolve4_4_neon(v_s3, v_s4, v_s5, v_s6, y_filter, max);

      store_u16_4x4(d, dst_stride, d0, d1, d2, d3);

      v_s0 = v_s4;
      v_s1 = v_s5;
      v_s2 = v_s6;
      s += 4 * src_stride;
      d += 4 * dst_stride;
      h -= 4;
    } while (h != 0);

    return;
  }

  const uint16x8_t max = vdupq_n_u16((1 << bd) - 1);

  do {
    const int16_t *s = (const int16_t *)src;
    uint16_t *d = dst;
    int height = h;

    int16x8_t h_s0[4], h_s1[4], h_s2[4];
    load_s16_8x4(s + 0 * src_stride, 1, &h_s0[0], &h_s0[1], &h_s0[2], &h_s0[3]);
    load_s16_8x4(s + 1 * src_stride, 1, &h_s1[0], &h_s1[1], &h_s1[2], &h_s1[3]);
    load_s16_8x4(s + 2 * src_stride, 1, &h_s2[0], &h_s2[1], &h_s2[2], &h_s2[3]);

    int16x8_t v_s0 = vreinterpretq_s16_u16(highbd_convolve4_8_neon(
        h_s0[0], h_s0[1], h_s0[2], h_s0[3], x_filter, max));
    int16x8_t v_s1 = vreinterpretq_s16_u16(highbd_convolve4_8_neon(
        h_s1[0], h_s1[1], h_s1[2], h_s1[3], x_filter, max));
    int16x8_t v_s2 = vreinterpretq_s16_u16(highbd_convolve4_8_neon(
        h_s2[0], h_s2[1], h_s2[2], h_s2[3], x_filter, max));

    s += 3 * src_stride;

    do {
      int16x8_t h_s3[4], h_s4[4], h_s5[4], h_s6[4];
      load_s16_8x4(s + 0 * src_stride, 1, &h_s3[0], &h_s3[1], &h_s3[2],
                   &h_s3[3]);
      load_s16_8x4(s + 1 * src_stride, 1, &h_s4[0], &h_s4[1], &h_s4[2],
                   &h_s4[3]);
      load_s16_8x4(s + 2 * src_stride, 1, &h_s5[0], &h_s5[1], &h_s5[2],
                   &h_s5[3]);
      load_s16_8x4(s + 3 * src_stride, 1, &h_s6[0], &h_s6[1], &h_s6[2],
                   &h_s6[3]);

      int16x8_t v_s3 = vreinterpretq_s16_u16(highbd_convolve4_8_neon(
          h_s3[0], h_s3[1], h_s3[2], h_s3[3], x_filter, max));
      int16x8_t v_s4 = vreinterpretq_s16_u16(highbd_convolve4_8_neon(
          h_s4[0], h_s4[1], h_s4[2], h_s4[3], x_filter, max));
      int16x8_t v_s5 = vreinterpretq_s16_u16(highbd_convolve4_8_neon(
          h_s5[0], h_s5[1], h_s5[2], h_s5[3], x_filter, max));
      int16x8_t v_s6 = vreinterpretq_s16_u16(highbd_convolve4_8_neon(
          h_s6[0], h_s6[1], h_s6[2], h_s6[3], x_filter, max));

      uint16x8_t d0 =
          highbd_convolve4_8_neon(v_s0, v_s1, v_s2, v_s3, y_filter, max);
      uint16x8_t d1 =
          highbd_convolve4_8_neon(v_s1, v_s2, v_s3, v_s4, y_filter, max);
      uint16x8_t d2 =
          highbd_convolve4_8_neon(v_s2, v_s3, v_s4, v_s5, y_filter, max);
      uint16x8_t d3 =
          highbd_convolve4_8_neon(v_s3, v_s4, v_s5, v_s6, y_filter, max);

      store_u16_8x4(d, dst_stride, d0, d1, d2, d3);

      v_s0 = v_s4;
      v_s1 = v_s5;
      v_s2 = v_s6;
      s += 4 * src_stride;
      d += 4 * dst_stride;
      height -= 4;
    } while (height != 0);
    src += 8;
    dst += 8;
    w -= 8;
  } while (w != 0);
}

static INLINE void highbd_convolve_2d_8tap_neon(
    const uint16_t *src, ptrdiff_t src_stride, uint16_t *dst,
    ptrdiff_t dst_stride, int w, int h, const int16x8_t x_filter,
    const int16x8_t y_filter, int bd) {
  if (w == 4) {
    const uint16x4_t max = vdup_n_u16((1 << bd) - 1);
    const int16_t *s = (const int16_t *)src;
    uint16_t *d = dst;

    int16x4_t h_s0[8], h_s1[8], h_s2[8], h_s3[8], h_s4[8], h_s5[8], h_s6[8];
    load_s16_4x8(s + 0 * src_stride, 1, &h_s0[0], &h_s0[1], &h_s0[2], &h_s0[3],
                 &h_s0[4], &h_s0[5], &h_s0[6], &h_s0[7]);
    load_s16_4x8(s + 1 * src_stride, 1, &h_s1[0], &h_s1[1], &h_s1[2], &h_s1[3],
                 &h_s1[4], &h_s1[5], &h_s1[6], &h_s1[7]);
    load_s16_4x8(s + 2 * src_stride, 1, &h_s2[0], &h_s2[1], &h_s2[2], &h_s2[3],
                 &h_s2[4], &h_s2[5], &h_s2[6], &h_s2[7]);
    load_s16_4x8(s + 3 * src_stride, 1, &h_s3[0], &h_s3[1], &h_s3[2], &h_s3[3],
                 &h_s3[4], &h_s3[5], &h_s3[6], &h_s3[7]);
    load_s16_4x8(s + 4 * src_stride, 1, &h_s4[0], &h_s4[1], &h_s4[2], &h_s4[3],
                 &h_s4[4], &h_s4[5], &h_s4[6], &h_s4[7]);
    load_s16_4x8(s + 5 * src_stride, 1, &h_s5[0], &h_s5[1], &h_s5[2], &h_s5[3],
                 &h_s5[4], &h_s5[5], &h_s5[6], &h_s5[7]);
    load_s16_4x8(s + 6 * src_stride, 1, &h_s6[0], &h_s6[1], &h_s6[2], &h_s6[3],
                 &h_s6[4], &h_s6[5], &h_s6[6], &h_s6[7]);

    int16x4_t v_s0 = vreinterpret_s16_u16(
        highbd_convolve8_4(h_s0[0], h_s0[1], h_s0[2], h_s0[3], h_s0[4], h_s0[5],
                           h_s0[6], h_s0[7], x_filter, max));
    int16x4_t v_s1 = vreinterpret_s16_u16(
        highbd_convolve8_4(h_s1[0], h_s1[1], h_s1[2], h_s1[3], h_s1[4], h_s1[5],
                           h_s1[6], h_s1[7], x_filter, max));
    int16x4_t v_s2 = vreinterpret_s16_u16(
        highbd_convolve8_4(h_s2[0], h_s2[1], h_s2[2], h_s2[3], h_s2[4], h_s2[5],
                           h_s2[6], h_s2[7], x_filter, max));
    int16x4_t v_s3 = vreinterpret_s16_u16(
        highbd_convolve8_4(h_s3[0], h_s3[1], h_s3[2], h_s3[3], h_s3[4], h_s3[5],
                           h_s3[6], h_s3[7], x_filter, max));
    int16x4_t v_s4 = vreinterpret_s16_u16(
        highbd_convolve8_4(h_s4[0], h_s4[1], h_s4[2], h_s4[3], h_s4[4], h_s4[5],
                           h_s4[6], h_s4[7], x_filter, max));
    int16x4_t v_s5 = vreinterpret_s16_u16(
        highbd_convolve8_4(h_s5[0], h_s5[1], h_s5[2], h_s5[3], h_s5[4], h_s5[5],
                           h_s5[6], h_s5[7], x_filter, max));
    int16x4_t v_s6 = vreinterpret_s16_u16(
        highbd_convolve8_4(h_s6[0], h_s6[1], h_s6[2], h_s6[3], h_s6[4], h_s6[5],
                           h_s6[6], h_s6[7], x_filter, max));

    s += 7 * src_stride;

    do {
      int16x4_t h_s7[8], h_s8[8], h_s9[8], h_s10[8];
      load_s16_4x8(s + 0 * src_stride, 1, &h_s7[0], &h_s7[1], &h_s7[2],
                   &h_s7[3], &h_s7[4], &h_s7[5], &h_s7[6], &h_s7[7]);
      load_s16_4x8(s + 1 * src_stride, 1, &h_s8[0], &h_s8[1], &h_s8[2],
                   &h_s8[3], &h_s8[4], &h_s8[5], &h_s8[6], &h_s8[7]);
      load_s16_4x8(s + 2 * src_stride, 1, &h_s9[0], &h_s9[1], &h_s9[2],
                   &h_s9[3], &h_s9[4], &h_s9[5], &h_s9[6], &h_s9[7]);
      load_s16_4x8(s + 3 * src_stride, 1, &h_s10[0], &h_s10[1], &h_s10[2],
                   &h_s10[3], &h_s10[4], &h_s10[5], &h_s10[6], &h_s10[7]);

      int16x4_t v_s7 = vreinterpret_s16_u16(
          highbd_convolve8_4(h_s7[0], h_s7[1], h_s7[2], h_s7[3], h_s7[4],
                             h_s7[5], h_s7[6], h_s7[7], x_filter, max));
      int16x4_t v_s8 = vreinterpret_s16_u16(
          highbd_convolve8_4(h_s8[0], h_s8[1], h_s8[2], h_s8[3], h_s8[4],
                             h_s8[5], h_s8[6], h_s8[7], x_filter, max));
      int16x4_t v_s9 = vreinterpret_s16_u16(
          highbd_convolve8_4(h_s9[0], h_s9[1], h_s9[2], h_s9[3], h_s9[4],
                             h_s9[5], h_s9[6], h_s9[7], x_filter, max));
      int16x4_t v_s10 = vreinterpret_s16_u16(
          highbd_convolve8_4(h_s10[0], h_s10[1], h_s10[2], h_s10[3], h_s10[4],
                             h_s10[5], h_s10[6], h_s10[7], x_filter, max));

      uint16x4_t d0 = highbd_convolve8_4(v_s0, v_s1, v_s2, v_s3, v_s4, v_s5,
                                         v_s6, v_s7, y_filter, max);
      uint16x4_t d1 = highbd_convolve8_4(v_s1, v_s2, v_s3, v_s4, v_s5, v_s6,
                                         v_s7, v_s8, y_filter, max);
      uint16x4_t d2 = highbd_convolve8_4(v_s2, v_s3, v_s4, v_s5, v_s6, v_s7,
                                         v_s8, v_s9, y_filter, max);
      uint16x4_t d3 = highbd_convolve8_4(v_s3, v_s4, v_s5, v_s6, v_s7, v_s8,
                                         v_s9, v_s10, y_filter, max);

      store_u16_4x4(d, dst_stride, d0, d1, d2, d3);

      v_s0 = v_s4;
      v_s1 = v_s5;
      v_s2 = v_s6;
      v_s3 = v_s7;
      v_s4 = v_s8;
      v_s5 = v_s9;
      v_s6 = v_s10;
      s += 4 * src_stride;
      d += 4 * dst_stride;
      h -= 4;
    } while (h != 0);

    return;
  }

  const uint16x8_t max = vdupq_n_u16((1 << bd) - 1);

  do {
    const int16_t *s = (const int16_t *)src;
    uint16_t *d = dst;
    int height = h;

    int16x8_t h_s0[8], h_s1[8], h_s2[8], h_s3[8], h_s4[8], h_s5[8], h_s6[8];
    load_s16_8x8(s + 0 * src_stride, 1, &h_s0[0], &h_s0[1], &h_s0[2], &h_s0[3],
                 &h_s0[4], &h_s0[5], &h_s0[6], &h_s0[7]);
    load_s16_8x8(s + 1 * src_stride, 1, &h_s1[0], &h_s1[1], &h_s1[2], &h_s1[3],
                 &h_s1[4], &h_s1[5], &h_s1[6], &h_s1[7]);
    load_s16_8x8(s + 2 * src_stride, 1, &h_s2[0], &h_s2[1], &h_s2[2], &h_s2[3],
                 &h_s2[4], &h_s2[5], &h_s2[6], &h_s2[7]);
    load_s16_8x8(s + 3 * src_stride, 1, &h_s3[0], &h_s3[1], &h_s3[2], &h_s3[3],
                 &h_s3[4], &h_s3[5], &h_s3[6], &h_s3[7]);
    load_s16_8x8(s + 4 * src_stride, 1, &h_s4[0], &h_s4[1], &h_s4[2], &h_s4[3],
                 &h_s4[4], &h_s4[5], &h_s4[6], &h_s4[7]);
    load_s16_8x8(s + 5 * src_stride, 1, &h_s5[0], &h_s5[1], &h_s5[2], &h_s5[3],
                 &h_s5[4], &h_s5[5], &h_s5[6], &h_s5[7]);
    load_s16_8x8(s + 6 * src_stride, 1, &h_s6[0], &h_s6[1], &h_s6[2], &h_s6[3],
                 &h_s6[4], &h_s6[5], &h_s6[6], &h_s6[7]);

    int16x8_t v_s0 = vreinterpretq_s16_u16(
        highbd_convolve8_8(h_s0[0], h_s0[1], h_s0[2], h_s0[3], h_s0[4], h_s0[5],
                           h_s0[6], h_s0[7], x_filter, max));
    int16x8_t v_s1 = vreinterpretq_s16_u16(
        highbd_convolve8_8(h_s1[0], h_s1[1], h_s1[2], h_s1[3], h_s1[4], h_s1[5],
                           h_s1[6], h_s1[7], x_filter, max));
    int16x8_t v_s2 = vreinterpretq_s16_u16(
        highbd_convolve8_8(h_s2[0], h_s2[1], h_s2[2], h_s2[3], h_s2[4], h_s2[5],
                           h_s2[6], h_s2[7], x_filter, max));
    int16x8_t v_s3 = vreinterpretq_s16_u16(
        highbd_convolve8_8(h_s3[0], h_s3[1], h_s3[2], h_s3[3], h_s3[4], h_s3[5],
                           h_s3[6], h_s3[7], x_filter, max));
    int16x8_t v_s4 = vreinterpretq_s16_u16(
        highbd_convolve8_8(h_s4[0], h_s4[1], h_s4[2], h_s4[3], h_s4[4], h_s4[5],
                           h_s4[6], h_s4[7], x_filter, max));
    int16x8_t v_s5 = vreinterpretq_s16_u16(
        highbd_convolve8_8(h_s5[0], h_s5[1], h_s5[2], h_s5[3], h_s5[4], h_s5[5],
                           h_s5[6], h_s5[7], x_filter, max));
    int16x8_t v_s6 = vreinterpretq_s16_u16(
        highbd_convolve8_8(h_s6[0], h_s6[1], h_s6[2], h_s6[3], h_s6[4], h_s6[5],
                           h_s6[6], h_s6[7], x_filter, max));

    s += 7 * src_stride;

    do {
      int16x8_t h_s7[8], h_s8[8], h_s9[8], h_s10[8];
      load_s16_8x8(s + 0 * src_stride, 1, &h_s7[0], &h_s7[1], &h_s7[2],
                   &h_s7[3], &h_s7[4], &h_s7[5], &h_s7[6], &h_s7[7]);
      load_s16_8x8(s + 1 * src_stride, 1, &h_s8[0], &h_s8[1], &h_s8[2],
                   &h_s8[3], &h_s8[4], &h_s8[5], &h_s8[6], &h_s8[7]);
      load_s16_8x8(s + 2 * src_stride, 1, &h_s9[0], &h_s9[1], &h_s9[2],
                   &h_s9[3], &h_s9[4], &h_s9[5], &h_s9[6], &h_s9[7]);
      load_s16_8x8(s + 3 * src_stride, 1, &h_s10[0], &h_s10[1], &h_s10[2],
                   &h_s10[3], &h_s10[4], &h_s10[5], &h_s10[6], &h_s10[7]);

      int16x8_t v_s7 = vreinterpretq_s16_u16(
          highbd_convolve8_8(h_s7[0], h_s7[1], h_s7[2], h_s7[3], h_s7[4],
                             h_s7[5], h_s7[6], h_s7[7], x_filter, max));
      int16x8_t v_s8 = vreinterpretq_s16_u16(
          highbd_convolve8_8(h_s8[0], h_s8[1], h_s8[2], h_s8[3], h_s8[4],
                             h_s8[5], h_s8[6], h_s8[7], x_filter, max));
      int16x8_t v_s9 = vreinterpretq_s16_u16(
          highbd_convolve8_8(h_s9[0], h_s9[1], h_s9[2], h_s9[3], h_s9[4],
                             h_s9[5], h_s9[6], h_s9[7], x_filter, max));
      int16x8_t v_s10 = vreinterpretq_s16_u16(
          highbd_convolve8_8(h_s10[0], h_s10[1], h_s10[2], h_s10[3], h_s10[4],
                             h_s10[5], h_s10[6], h_s10[7], x_filter, max));

      uint16x8_t d0 = highbd_convolve8_8(v_s0, v_s1, v_s2, v_s3, v_s4, v_s5,
                                         v_s6, v_s7, y_filter, max);
      uint16x8_t d1 = highbd_convolve8_8(v_s1, v_s2, v_s3, v_s4, v_s5, v_s6,
                                         v_s7, v_s8, y_filter, max);
      uint16x8_t d2 = highbd_convolve8_8(v_s2, v_s3, v_s4, v_s5, v_s6, v_s7,
                                         v_s8, v_s9, y_filter, max);
      uint16x8_t d3 = highbd_convolve8_8(v_s3, v_s4, v_s5, v_s6, v_s7, v_s8,
                                         v_s9, v_s10, y_filter, max);

      store_u16_8x4(d, dst_stride, d0, d1, d2, d3);

      v_s0 = v_s4;
      v_s1 = v_s5;
      v_s2 = v_s6;
      v_s3 = v_s7;
      v_s4 = v_s8;
      v_s5 = v_s9;
      v_s6 = v_s10;
      s += 4 * src_stride;
      d += 4 * dst_stride;
      height -= 4;
    } while (height != 0);
    src += 8;
    dst += 8;
    w -= 8;
  } while (w != 0);
}

void vpx_highbd_convolve8_neon(const uint16_t *src, ptrdiff_t src_stride,
                               uint16_t *dst, ptrdiff_t dst_stride,
                               const InterpKernel *filter, int x0_q4,
                               int x_step_q4, int y0_q4, int y_step_q4, int w,
                               int h, int bd) {
  if (x_step_q4 != 16 || y_step_q4 != 16) {
    vpx_highbd_convolve8_c(src, src_stride, dst, dst_stride, filter, x0_q4,
                           x_step_q4, y0_q4, y_step_q4, w, h, bd);
    return;
  }

  const int x_filter_taps = vpx_get_filter_taps(filter[x0_q4]) <= 4 ? 4 : 8;
  const int y_filter_taps = vpx_get_filter_taps(filter[y0_q4]) <= 4 ? 4 : 8;
  // Account for needing filter_taps / 2 - 1 lines prior and filter_taps / 2
  // lines post both horizontally and vertically.
  const ptrdiff_t horiz_offset = x_filter_taps / 2 - 1;
  const ptrdiff_t vert_offset = (y_filter_taps / 2 - 1) * src_stride;

  if (x_filter_taps == 4 && y_filter_taps == 4) {
    const int16x4_t x_filter = vld1_s16(filter[x0_q4] + 2);
    const int16x4_t y_filter = vld1_s16(filter[y0_q4] + 2);

    highbd_convolve_2d_4tap_neon(src - horiz_offset - vert_offset, src_stride,
                                 dst, dst_stride, w, h, x_filter, y_filter, bd);
    return;
  }

  const int16x8_t x_filter = vld1q_s16(filter[x0_q4]);
  const int16x8_t y_filter = vld1q_s16(filter[y0_q4]);

  highbd_convolve_2d_8tap_neon(src - horiz_offset - vert_offset, src_stride,
                               dst, dst_stride, w, h, x_filter, y_filter, bd);
}

void vpx_highbd_convolve8_avg_neon(const uint16_t *src, ptrdiff_t src_stride,
                                   uint16_t *dst, ptrdiff_t dst_stride,
                                   const InterpKernel *filter, int x0_q4,
                                   int x_step_q4, int y0_q4, int y_step_q4,
                                   int w, int h, int bd) {
  if (x_step_q4 != 16 || y_step_q4 != 16) {
    vpx_highbd_convolve8_avg_c(src, src_stride, dst, dst_stride, filter, x0_q4,
                               x_step_q4, y0_q4, y_step_q4, w, h, bd);
    return;
  }

  // Averaging convolution always uses an 8-tap filter.
  const ptrdiff_t horiz_offset = SUBPEL_TAPS / 2 - 1;
  const ptrdiff_t vert_offset = (SUBPEL_TAPS / 2 - 1) * src_stride;
  // Account for needing SUBPEL_TAPS / 2 - 1 lines prior and SUBPEL_TAPS / 2
  // lines post both horizontally and vertically.
  src = src - horiz_offset - vert_offset;

  const int16x8_t x_filter = vld1q_s16(filter[x0_q4]);
  const int16x8_t y_filter = vld1q_s16(filter[y0_q4]);

  if (w == 4) {
    const uint16x4_t max = vdup_n_u16((1 << bd) - 1);
    const int16_t *s = (const int16_t *)src;
    uint16_t *d = dst;

    int16x4_t h_s0[8], h_s1[8], h_s2[8], h_s3[8], h_s4[8], h_s5[8], h_s6[8];
    load_s16_4x8(s + 0 * src_stride, 1, &h_s0[0], &h_s0[1], &h_s0[2], &h_s0[3],
                 &h_s0[4], &h_s0[5], &h_s0[6], &h_s0[7]);
    load_s16_4x8(s + 1 * src_stride, 1, &h_s1[0], &h_s1[1], &h_s1[2], &h_s1[3],
                 &h_s1[4], &h_s1[5], &h_s1[6], &h_s1[7]);
    load_s16_4x8(s + 2 * src_stride, 1, &h_s2[0], &h_s2[1], &h_s2[2], &h_s2[3],
                 &h_s2[4], &h_s2[5], &h_s2[6], &h_s2[7]);
    load_s16_4x8(s + 3 * src_stride, 1, &h_s3[0], &h_s3[1], &h_s3[2], &h_s3[3],
                 &h_s3[4], &h_s3[5], &h_s3[6], &h_s3[7]);
    load_s16_4x8(s + 4 * src_stride, 1, &h_s4[0], &h_s4[1], &h_s4[2], &h_s4[3],
                 &h_s4[4], &h_s4[5], &h_s4[6], &h_s4[7]);
    load_s16_4x8(s + 5 * src_stride, 1, &h_s5[0], &h_s5[1], &h_s5[2], &h_s5[3],
                 &h_s5[4], &h_s5[5], &h_s5[6], &h_s5[7]);
    load_s16_4x8(s + 6 * src_stride, 1, &h_s6[0], &h_s6[1], &h_s6[2], &h_s6[3],
                 &h_s6[4], &h_s6[5], &h_s6[6], &h_s6[7]);

    int16x4_t v_s0 = vreinterpret_s16_u16(
        highbd_convolve8_4(h_s0[0], h_s0[1], h_s0[2], h_s0[3], h_s0[4], h_s0[5],
                           h_s0[6], h_s0[7], x_filter, max));
    int16x4_t v_s1 = vreinterpret_s16_u16(
        highbd_convolve8_4(h_s1[0], h_s1[1], h_s1[2], h_s1[3], h_s1[4], h_s1[5],
                           h_s1[6], h_s1[7], x_filter, max));
    int16x4_t v_s2 = vreinterpret_s16_u16(
        highbd_convolve8_4(h_s2[0], h_s2[1], h_s2[2], h_s2[3], h_s2[4], h_s2[5],
                           h_s2[6], h_s2[7], x_filter, max));
    int16x4_t v_s3 = vreinterpret_s16_u16(
        highbd_convolve8_4(h_s3[0], h_s3[1], h_s3[2], h_s3[3], h_s3[4], h_s3[5],
                           h_s3[6], h_s3[7], x_filter, max));
    int16x4_t v_s4 = vreinterpret_s16_u16(
        highbd_convolve8_4(h_s4[0], h_s4[1], h_s4[2], h_s4[3], h_s4[4], h_s4[5],
                           h_s4[6], h_s4[7], x_filter, max));
    int16x4_t v_s5 = vreinterpret_s16_u16(
        highbd_convolve8_4(h_s5[0], h_s5[1], h_s5[2], h_s5[3], h_s5[4], h_s5[5],
                           h_s5[6], h_s5[7], x_filter, max));
    int16x4_t v_s6 = vreinterpret_s16_u16(
        highbd_convolve8_4(h_s6[0], h_s6[1], h_s6[2], h_s6[3], h_s6[4], h_s6[5],
                           h_s6[6], h_s6[7], x_filter, max));

    s += 7 * src_stride;

    do {
      int16x4_t h_s7[8], h_s8[8], h_s9[8], h_s10[8];
      load_s16_4x8(s + 0 * src_stride, 1, &h_s7[0], &h_s7[1], &h_s7[2],
                   &h_s7[3], &h_s7[4], &h_s7[5], &h_s7[6], &h_s7[7]);
      load_s16_4x8(s + 1 * src_stride, 1, &h_s8[0], &h_s8[1], &h_s8[2],
                   &h_s8[3], &h_s8[4], &h_s8[5], &h_s8[6], &h_s8[7]);
      load_s16_4x8(s + 2 * src_stride, 1, &h_s9[0], &h_s9[1], &h_s9[2],
                   &h_s9[3], &h_s9[4], &h_s9[5], &h_s9[6], &h_s9[7]);
      load_s16_4x8(s + 3 * src_stride, 1, &h_s10[0], &h_s10[1], &h_s10[2],
                   &h_s10[3], &h_s10[4], &h_s10[5], &h_s10[6], &h_s10[7]);

      int16x4_t v_s7 = vreinterpret_s16_u16(
          highbd_convolve8_4(h_s7[0], h_s7[1], h_s7[2], h_s7[3], h_s7[4],
                             h_s7[5], h_s7[6], h_s7[7], x_filter, max));
      int16x4_t v_s8 = vreinterpret_s16_u16(
          highbd_convolve8_4(h_s8[0], h_s8[1], h_s8[2], h_s8[3], h_s8[4],
                             h_s8[5], h_s8[6], h_s8[7], x_filter, max));
      int16x4_t v_s9 = vreinterpret_s16_u16(
          highbd_convolve8_4(h_s9[0], h_s9[1], h_s9[2], h_s9[3], h_s9[4],
                             h_s9[5], h_s9[6], h_s9[7], x_filter, max));
      int16x4_t v_s10 = vreinterpret_s16_u16(
          highbd_convolve8_4(h_s10[0], h_s10[1], h_s10[2], h_s10[3], h_s10[4],
                             h_s10[5], h_s10[6], h_s10[7], x_filter, max));

      uint16x4_t d0 = highbd_convolve8_4(v_s0, v_s1, v_s2, v_s3, v_s4, v_s5,
                                         v_s6, v_s7, y_filter, max);
      uint16x4_t d1 = highbd_convolve8_4(v_s1, v_s2, v_s3, v_s4, v_s5, v_s6,
                                         v_s7, v_s8, y_filter, max);
      uint16x4_t d2 = highbd_convolve8_4(v_s2, v_s3, v_s4, v_s5, v_s6, v_s7,
                                         v_s8, v_s9, y_filter, max);
      uint16x4_t d3 = highbd_convolve8_4(v_s3, v_s4, v_s5, v_s6, v_s7, v_s8,
                                         v_s9, v_s10, y_filter, max);

      d0 = vrhadd_u16(d0, vld1_u16(d + 0 * dst_stride));
      d1 = vrhadd_u16(d1, vld1_u16(d + 1 * dst_stride));
      d2 = vrhadd_u16(d2, vld1_u16(d + 2 * dst_stride));
      d3 = vrhadd_u16(d3, vld1_u16(d + 3 * dst_stride));

      store_u16_4x4(d, dst_stride, d0, d1, d2, d3);

      v_s0 = v_s4;
      v_s1 = v_s5;
      v_s2 = v_s6;
      v_s3 = v_s7;
      v_s4 = v_s8;
      v_s5 = v_s9;
      v_s6 = v_s10;
      s += 4 * src_stride;
      d += 4 * dst_stride;
      h -= 4;
    } while (h != 0);

    return;
  }

  const uint16x8_t max = vdupq_n_u16((1 << bd) - 1);

  do {
    const int16_t *s = (const int16_t *)src;
    uint16_t *d = dst;
    int height = h;

    int16x8_t h_s0[8], h_s1[8], h_s2[8], h_s3[8], h_s4[8], h_s5[8], h_s6[8];
    load_s16_8x8(s + 0 * src_stride, 1, &h_s0[0], &h_s0[1], &h_s0[2], &h_s0[3],
                 &h_s0[4], &h_s0[5], &h_s0[6], &h_s0[7]);
    load_s16_8x8(s + 1 * src_stride, 1, &h_s1[0], &h_s1[1], &h_s1[2], &h_s1[3],
                 &h_s1[4], &h_s1[5], &h_s1[6], &h_s1[7]);
    load_s16_8x8(s + 2 * src_stride, 1, &h_s2[0], &h_s2[1], &h_s2[2], &h_s2[3],
                 &h_s2[4], &h_s2[5], &h_s2[6], &h_s2[7]);
    load_s16_8x8(s + 3 * src_stride, 1, &h_s3[0], &h_s3[1], &h_s3[2], &h_s3[3],
                 &h_s3[4], &h_s3[5], &h_s3[6], &h_s3[7]);
    load_s16_8x8(s + 4 * src_stride, 1, &h_s4[0], &h_s4[1], &h_s4[2], &h_s4[3],
                 &h_s4[4], &h_s4[5], &h_s4[6], &h_s4[7]);
    load_s16_8x8(s + 5 * src_stride, 1, &h_s5[0], &h_s5[1], &h_s5[2], &h_s5[3],
                 &h_s5[4], &h_s5[5], &h_s5[6], &h_s5[7]);
    load_s16_8x8(s + 6 * src_stride, 1, &h_s6[0], &h_s6[1], &h_s6[2], &h_s6[3],
                 &h_s6[4], &h_s6[5], &h_s6[6], &h_s6[7]);

    int16x8_t v_s0 = vreinterpretq_s16_u16(
        highbd_convolve8_8(h_s0[0], h_s0[1], h_s0[2], h_s0[3], h_s0[4], h_s0[5],
                           h_s0[6], h_s0[7], x_filter, max));
    int16x8_t v_s1 = vreinterpretq_s16_u16(
        highbd_convolve8_8(h_s1[0], h_s1[1], h_s1[2], h_s1[3], h_s1[4], h_s1[5],
                           h_s1[6], h_s1[7], x_filter, max));
    int16x8_t v_s2 = vreinterpretq_s16_u16(
        highbd_convolve8_8(h_s2[0], h_s2[1], h_s2[2], h_s2[3], h_s2[4], h_s2[5],
                           h_s2[6], h_s2[7], x_filter, max));
    int16x8_t v_s3 = vreinterpretq_s16_u16(
        highbd_convolve8_8(h_s3[0], h_s3[1], h_s3[2], h_s3[3], h_s3[4], h_s3[5],
                           h_s3[6], h_s3[7], x_filter, max));
    int16x8_t v_s4 = vreinterpretq_s16_u16(
        highbd_convolve8_8(h_s4[0], h_s4[1], h_s4[2], h_s4[3], h_s4[4], h_s4[5],
                           h_s4[6], h_s4[7], x_filter, max));
    int16x8_t v_s5 = vreinterpretq_s16_u16(
        highbd_convolve8_8(h_s5[0], h_s5[1], h_s5[2], h_s5[3], h_s5[4], h_s5[5],
                           h_s5[6], h_s5[7], x_filter, max));
    int16x8_t v_s6 = vreinterpretq_s16_u16(
        highbd_convolve8_8(h_s6[0], h_s6[1], h_s6[2], h_s6[3], h_s6[4], h_s6[5],
                           h_s6[6], h_s6[7], x_filter, max));

    s += 7 * src_stride;

    do {
      int16x8_t h_s7[8], h_s8[8], h_s9[8], h_s10[8];
      load_s16_8x8(s + 0 * src_stride, 1, &h_s7[0], &h_s7[1], &h_s7[2],
                   &h_s7[3], &h_s7[4], &h_s7[5], &h_s7[6], &h_s7[7]);
      load_s16_8x8(s + 1 * src_stride, 1, &h_s8[0], &h_s8[1], &h_s8[2],
                   &h_s8[3], &h_s8[4], &h_s8[5], &h_s8[6], &h_s8[7]);
      load_s16_8x8(s + 2 * src_stride, 1, &h_s9[0], &h_s9[1], &h_s9[2],
                   &h_s9[3], &h_s9[4], &h_s9[5], &h_s9[6], &h_s9[7]);
      load_s16_8x8(s + 3 * src_stride, 1, &h_s10[0], &h_s10[1], &h_s10[2],
                   &h_s10[3], &h_s10[4], &h_s10[5], &h_s10[6], &h_s10[7]);

      int16x8_t v_s7 = vreinterpretq_s16_u16(
          highbd_convolve8_8(h_s7[0], h_s7[1], h_s7[2], h_s7[3], h_s7[4],
                             h_s7[5], h_s7[6], h_s7[7], x_filter, max));
      int16x8_t v_s8 = vreinterpretq_s16_u16(
          highbd_convolve8_8(h_s8[0], h_s8[1], h_s8[2], h_s8[3], h_s8[4],
                             h_s8[5], h_s8[6], h_s8[7], x_filter, max));
      int16x8_t v_s9 = vreinterpretq_s16_u16(
          highbd_convolve8_8(h_s9[0], h_s9[1], h_s9[2], h_s9[3], h_s9[4],
                             h_s9[5], h_s9[6], h_s9[7], x_filter, max));
      int16x8_t v_s10 = vreinterpretq_s16_u16(
          highbd_convolve8_8(h_s10[0], h_s10[1], h_s10[2], h_s10[3], h_s10[4],
                             h_s10[5], h_s10[6], h_s10[7], x_filter, max));

      uint16x8_t d0 = highbd_convolve8_8(v_s0, v_s1, v_s2, v_s3, v_s4, v_s5,
                                         v_s6, v_s7, y_filter, max);
      uint16x8_t d1 = highbd_convolve8_8(v_s1, v_s2, v_s3, v_s4, v_s5, v_s6,
                                         v_s7, v_s8, y_filter, max);
      uint16x8_t d2 = highbd_convolve8_8(v_s2, v_s3, v_s4, v_s5, v_s6, v_s7,
                                         v_s8, v_s9, y_filter, max);
      uint16x8_t d3 = highbd_convolve8_8(v_s3, v_s4, v_s5, v_s6, v_s7, v_s8,
                                         v_s9, v_s10, y_filter, max);

      d0 = vrhaddq_u16(d0, vld1q_u16(d + 0 * dst_stride));
      d1 = vrhaddq_u16(d1, vld1q_u16(d + 1 * dst_stride));
      d2 = vrhaddq_u16(d2, vld1q_u16(d + 2 * dst_stride));
      d3 = vrhaddq_u16(d3, vld1q_u16(d + 3 * dst_stride));

      store_u16_8x4(d, dst_stride, d0, d1, d2, d3);

      v_s0 = v_s4;
      v_s1 = v_s5;
      v_s2 = v_s6;
      v_s3 = v_s7;
      v_s4 = v_s8;
      v_s5 = v_s9;
      v_s6 = v_s10;
      s += 4 * src_stride;
      d += 4 * dst_stride;
      height -= 4;
    } while (height != 0);
    src += 8;
    dst += 8;
    w -= 8;
  } while (w != 0);
}
