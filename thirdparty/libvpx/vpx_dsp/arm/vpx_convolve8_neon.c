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
#include <assert.h>

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/arm/mem_neon.h"
#include "vpx_dsp/arm/transpose_neon.h"
#include "vpx_dsp/arm/vpx_convolve8_neon.h"
#include "vpx_dsp/vpx_filter.h"
#include "vpx_ports/mem.h"

static INLINE void convolve_4tap_horiz_neon(const uint8_t *src,
                                            ptrdiff_t src_stride, uint8_t *dst,
                                            ptrdiff_t dst_stride, int w, int h,
                                            const int16x8_t filter) {
  // 4-tap and bilinear filter values are even, so halve them to reduce
  // intermediate precision requirements.
  const uint8x8_t x_filter =
      vshrn_n_u16(vreinterpretq_u16_s16(vabsq_s16(filter)), 1);

  // Neon does not have lane-referencing multiply or multiply-accumulate
  // instructions that operate on vectors of 8-bit elements. This means we have
  // to duplicate filter taps into a whole vector and use standard multiply /
  // multiply-accumulate instructions.
  const uint8x8_t filter_taps[4] = { vdup_lane_u8(x_filter, 2),
                                     vdup_lane_u8(x_filter, 3),
                                     vdup_lane_u8(x_filter, 4),
                                     vdup_lane_u8(x_filter, 5) };

  if (w == 4) {
    do {
      uint8x8_t s01[4];

      s01[0] = load_unaligned_u8(src + 0, src_stride);
      s01[1] = load_unaligned_u8(src + 1, src_stride);
      s01[2] = load_unaligned_u8(src + 2, src_stride);
      s01[3] = load_unaligned_u8(src + 3, src_stride);

      uint8x8_t d01 = convolve4_8(s01[0], s01[1], s01[2], s01[3], filter_taps);

      store_unaligned_u8(dst, dst_stride, d01);

      src += 2 * src_stride;
      dst += 2 * dst_stride;
      h -= 2;
    } while (h > 0);
  } else {
    do {
      const uint8_t *s = src;
      uint8_t *d = dst;
      int width = w;

      do {
        uint8x8_t s0[4], s1[4];

        s0[0] = vld1_u8(s + 0);
        s0[1] = vld1_u8(s + 1);
        s0[2] = vld1_u8(s + 2);
        s0[3] = vld1_u8(s + 3);

        s1[0] = vld1_u8(s + src_stride + 0);
        s1[1] = vld1_u8(s + src_stride + 1);
        s1[2] = vld1_u8(s + src_stride + 2);
        s1[3] = vld1_u8(s + src_stride + 3);

        uint8x8_t d0 = convolve4_8(s0[0], s0[1], s0[2], s0[3], filter_taps);
        uint8x8_t d1 = convolve4_8(s1[0], s1[1], s1[2], s1[3], filter_taps);

        vst1_u8(d, d0);
        vst1_u8(d + dst_stride, d1);
        s += 8;
        d += 8;
        width -= 8;
      } while (width != 0);
      src += 2 * src_stride;
      dst += 2 * dst_stride;
      h -= 2;
    } while (h > 0);
  }
}

static INLINE void convolve_8tap_horiz_neon(const uint8_t *src,
                                            ptrdiff_t src_stride, uint8_t *dst,
                                            ptrdiff_t dst_stride, int w, int h,
                                            const int16x8_t filter) {
  if (h == 4) {
    uint8x8_t t0, t1, t2, t3;
    load_u8_8x4(src, src_stride, &t0, &t1, &t2, &t3);

    transpose_u8_8x4(&t0, &t1, &t2, &t3);
    int16x4_t s0 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t0)));
    int16x4_t s1 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t1)));
    int16x4_t s2 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t2)));
    int16x4_t s3 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t3)));
    int16x4_t s4 = vget_high_s16(vreinterpretq_s16_u16(vmovl_u8(t0)));
    int16x4_t s5 = vget_high_s16(vreinterpretq_s16_u16(vmovl_u8(t1)));
    int16x4_t s6 = vget_high_s16(vreinterpretq_s16_u16(vmovl_u8(t2)));

    src += 7;

    do {
      uint8x8_t t7, t8, t9, t10;
      load_u8_8x4(src, src_stride, &t7, &t8, &t9, &t10);

      transpose_u8_8x4(&t7, &t8, &t9, &t10);
      int16x4_t s7 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t7)));
      int16x4_t s8 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t8)));
      int16x4_t s9 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t9)));
      int16x4_t s10 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t10)));

      int16x4_t d0 = convolve8_4(s0, s1, s2, s3, s4, s5, s6, s7, filter);
      int16x4_t d1 = convolve8_4(s1, s2, s3, s4, s5, s6, s7, s8, filter);
      int16x4_t d2 = convolve8_4(s2, s3, s4, s5, s6, s7, s8, s9, filter);
      int16x4_t d3 = convolve8_4(s3, s4, s5, s6, s7, s8, s9, s10, filter);
      uint8x8_t d01 = vqrshrun_n_s16(vcombine_s16(d0, d1), FILTER_BITS);
      uint8x8_t d23 = vqrshrun_n_s16(vcombine_s16(d2, d3), FILTER_BITS);

      transpose_u8_4x4(&d01, &d23);

      store_u8(dst + 0 * dst_stride, 2 * dst_stride, d01);
      store_u8(dst + 1 * dst_stride, 2 * dst_stride, d23);

      s0 = s4;
      s1 = s5;
      s2 = s6;
      s3 = s7;
      s4 = s8;
      s5 = s9;
      s6 = s10;
      src += 4;
      dst += 4;
      w -= 4;
    } while (w != 0);
  } else {
    if (w == 4) {
      do {
        uint8x8_t t0, t1, t2, t3, t4, t5, t6, t7;
        load_u8_8x8(src, src_stride, &t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7);

        transpose_u8_8x8(&t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7);
        int16x8_t s0 = vreinterpretq_s16_u16(vmovl_u8(t0));
        int16x8_t s1 = vreinterpretq_s16_u16(vmovl_u8(t1));
        int16x8_t s2 = vreinterpretq_s16_u16(vmovl_u8(t2));
        int16x8_t s3 = vreinterpretq_s16_u16(vmovl_u8(t3));
        int16x8_t s4 = vreinterpretq_s16_u16(vmovl_u8(t4));
        int16x8_t s5 = vreinterpretq_s16_u16(vmovl_u8(t5));
        int16x8_t s6 = vreinterpretq_s16_u16(vmovl_u8(t6));

        load_u8_8x8(src + 7, src_stride, &t0, &t1, &t2, &t3, &t4, &t5, &t6,
                    &t7);

        transpose_u8_4x8(&t0, &t1, &t2, &t3, t4, t5, t6, t7);
        int16x8_t s7 = vreinterpretq_s16_u16(vmovl_u8(t0));
        int16x8_t s8 = vreinterpretq_s16_u16(vmovl_u8(t1));
        int16x8_t s9 = vreinterpretq_s16_u16(vmovl_u8(t2));
        int16x8_t s10 = vreinterpretq_s16_u16(vmovl_u8(t3));

        uint8x8_t d04 = convolve8_8(s0, s1, s2, s3, s4, s5, s6, s7, filter);
        uint8x8_t d15 = convolve8_8(s1, s2, s3, s4, s5, s6, s7, s8, filter);
        uint8x8_t d26 = convolve8_8(s2, s3, s4, s5, s6, s7, s8, s9, filter);
        uint8x8_t d37 = convolve8_8(s3, s4, s5, s6, s7, s8, s9, s10, filter);

        transpose_u8_8x4(&d04, &d15, &d26, &d37);

        store_u8(dst + 0 * dst_stride, 4 * dst_stride, d04);
        store_u8(dst + 1 * dst_stride, 4 * dst_stride, d15);
        store_u8(dst + 2 * dst_stride, 4 * dst_stride, d26);
        store_u8(dst + 3 * dst_stride, 4 * dst_stride, d37);

        src += 8 * src_stride;
        dst += 8 * dst_stride;
        h -= 8;
      } while (h > 0);
    } else {
      do {
        uint8x8_t t0, t1, t2, t3, t4, t5, t6, t7;
        load_u8_8x8(src, src_stride, &t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7);

        transpose_u8_8x8(&t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7);
        int16x8_t s0 = vreinterpretq_s16_u16(vmovl_u8(t0));
        int16x8_t s1 = vreinterpretq_s16_u16(vmovl_u8(t1));
        int16x8_t s2 = vreinterpretq_s16_u16(vmovl_u8(t2));
        int16x8_t s3 = vreinterpretq_s16_u16(vmovl_u8(t3));
        int16x8_t s4 = vreinterpretq_s16_u16(vmovl_u8(t4));
        int16x8_t s5 = vreinterpretq_s16_u16(vmovl_u8(t5));
        int16x8_t s6 = vreinterpretq_s16_u16(vmovl_u8(t6));

        const uint8_t *s = src + 7;
        uint8_t *d = dst;
        int width = w;

        do {
          uint8x8_t t8, t9, t10, t11, t12, t13, t14, t15;
          load_u8_8x8(s, src_stride, &t8, &t9, &t10, &t11, &t12, &t13, &t14,
                      &t15);

          transpose_u8_8x8(&t8, &t9, &t10, &t11, &t12, &t13, &t14, &t15);
          int16x8_t s7 = vreinterpretq_s16_u16(vmovl_u8(t8));
          int16x8_t s8 = vreinterpretq_s16_u16(vmovl_u8(t9));
          int16x8_t s9 = vreinterpretq_s16_u16(vmovl_u8(t10));
          int16x8_t s10 = vreinterpretq_s16_u16(vmovl_u8(t11));
          int16x8_t s11 = vreinterpretq_s16_u16(vmovl_u8(t12));
          int16x8_t s12 = vreinterpretq_s16_u16(vmovl_u8(t13));
          int16x8_t s13 = vreinterpretq_s16_u16(vmovl_u8(t14));
          int16x8_t s14 = vreinterpretq_s16_u16(vmovl_u8(t15));

          uint8x8_t d0 = convolve8_8(s0, s1, s2, s3, s4, s5, s6, s7, filter);
          uint8x8_t d1 = convolve8_8(s1, s2, s3, s4, s5, s6, s7, s8, filter);
          uint8x8_t d2 = convolve8_8(s2, s3, s4, s5, s6, s7, s8, s9, filter);
          uint8x8_t d3 = convolve8_8(s3, s4, s5, s6, s7, s8, s9, s10, filter);
          uint8x8_t d4 = convolve8_8(s4, s5, s6, s7, s8, s9, s10, s11, filter);
          uint8x8_t d5 = convolve8_8(s5, s6, s7, s8, s9, s10, s11, s12, filter);
          uint8x8_t d6 =
              convolve8_8(s6, s7, s8, s9, s10, s11, s12, s13, filter);
          uint8x8_t d7 =
              convolve8_8(s7, s8, s9, s10, s11, s12, s13, s14, filter);

          transpose_u8_8x8(&d0, &d1, &d2, &d3, &d4, &d5, &d6, &d7);

          store_u8_8x8(d, dst_stride, d0, d1, d2, d3, d4, d5, d6, d7);

          s0 = s8;
          s1 = s9;
          s2 = s10;
          s3 = s11;
          s4 = s12;
          s5 = s13;
          s6 = s14;
          s += 8;
          d += 8;
          width -= 8;
        } while (width != 0);
        src += 8 * src_stride;
        dst += 8 * dst_stride;
        h -= 8;
      } while (h > 0);
    }
  }
}

void vpx_convolve8_horiz_neon(const uint8_t *src, ptrdiff_t src_stride,
                              uint8_t *dst, ptrdiff_t dst_stride,
                              const InterpKernel *filter, int x0_q4,
                              int x_step_q4, int y0_q4, int y_step_q4, int w,
                              int h) {
  assert((intptr_t)dst % 4 == 0);
  assert(dst_stride % 4 == 0);
  assert(x_step_q4 == 16);

  (void)x_step_q4;
  (void)y0_q4;
  (void)y_step_q4;

  const int16x8_t x_filter = vld1q_s16(filter[x0_q4]);

  if (vpx_get_filter_taps(filter[x0_q4]) <= 4) {
    convolve_4tap_horiz_neon(src - 1, src_stride, dst, dst_stride, w, h,
                             x_filter);
  } else {
    convolve_8tap_horiz_neon(src - 3, src_stride, dst, dst_stride, w, h,
                             x_filter);
  }
}

void vpx_convolve8_avg_horiz_neon(const uint8_t *src, ptrdiff_t src_stride,
                                  uint8_t *dst, ptrdiff_t dst_stride,
                                  const InterpKernel *filter, int x0_q4,
                                  int x_step_q4, int y0_q4, int y_step_q4,
                                  int w, int h) {
  const int16x8_t filters = vld1q_s16(filter[x0_q4]);

  assert((intptr_t)dst % 4 == 0);
  assert(dst_stride % 4 == 0);
  assert(x_step_q4 == 16);

  (void)x_step_q4;
  (void)y0_q4;
  (void)y_step_q4;

  src -= 3;

  if (h == 4) {
    uint8x8_t t0, t1, t2, t3;
    load_u8_8x4(src, src_stride, &t0, &t1, &t2, &t3);

    transpose_u8_8x4(&t0, &t1, &t2, &t3);
    int16x4_t s0 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t0)));
    int16x4_t s1 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t1)));
    int16x4_t s2 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t2)));
    int16x4_t s3 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t3)));
    int16x4_t s4 = vget_high_s16(vreinterpretq_s16_u16(vmovl_u8(t0)));
    int16x4_t s5 = vget_high_s16(vreinterpretq_s16_u16(vmovl_u8(t1)));
    int16x4_t s6 = vget_high_s16(vreinterpretq_s16_u16(vmovl_u8(t2)));

    src += 7;

    do {
      uint8x8_t t7, t8, t9, t10;
      load_u8_8x4(src, src_stride, &t7, &t8, &t9, &t10);

      transpose_u8_8x4(&t7, &t8, &t9, &t10);
      int16x4_t s7 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t7)));
      int16x4_t s8 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t8)));
      int16x4_t s9 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t9)));
      int16x4_t s10 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t10)));

      int16x4_t d0 = convolve8_4(s0, s1, s2, s3, s4, s5, s6, s7, filters);
      int16x4_t d1 = convolve8_4(s1, s2, s3, s4, s5, s6, s7, s8, filters);
      int16x4_t d2 = convolve8_4(s2, s3, s4, s5, s6, s7, s8, s9, filters);
      int16x4_t d3 = convolve8_4(s3, s4, s5, s6, s7, s8, s9, s10, filters);
      uint8x8_t d01 = vqrshrun_n_s16(vcombine_s16(d0, d1), FILTER_BITS);
      uint8x8_t d23 = vqrshrun_n_s16(vcombine_s16(d2, d3), FILTER_BITS);

      transpose_u8_4x4(&d01, &d23);

      uint8x8_t dd01 = load_u8(dst + 0 * dst_stride, 2 * dst_stride);
      uint8x8_t dd23 = load_u8(dst + 1 * dst_stride, 2 * dst_stride);

      d01 = vrhadd_u8(d01, dd01);
      d23 = vrhadd_u8(d23, dd23);

      store_u8(dst + 0 * dst_stride, 2 * dst_stride, d01);
      store_u8(dst + 1 * dst_stride, 2 * dst_stride, d23);

      s0 = s4;
      s1 = s5;
      s2 = s6;
      s3 = s7;
      s4 = s8;
      s5 = s9;
      s6 = s10;
      src += 4;
      dst += 4;
      w -= 4;
    } while (w != 0);
  } else {
    if (w == 4) {
      do {
        uint8x8_t t0, t1, t2, t3, t4, t5, t6, t7;
        load_u8_8x8(src, src_stride, &t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7);

        transpose_u8_8x8(&t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7);
        int16x8_t s0 = vreinterpretq_s16_u16(vmovl_u8(t0));
        int16x8_t s1 = vreinterpretq_s16_u16(vmovl_u8(t1));
        int16x8_t s2 = vreinterpretq_s16_u16(vmovl_u8(t2));
        int16x8_t s3 = vreinterpretq_s16_u16(vmovl_u8(t3));
        int16x8_t s4 = vreinterpretq_s16_u16(vmovl_u8(t4));
        int16x8_t s5 = vreinterpretq_s16_u16(vmovl_u8(t5));
        int16x8_t s6 = vreinterpretq_s16_u16(vmovl_u8(t6));

        load_u8_8x8(src + 7, src_stride, &t0, &t1, &t2, &t3, &t4, &t5, &t6,
                    &t7);

        transpose_u8_4x8(&t0, &t1, &t2, &t3, t4, t5, t6, t7);
        int16x8_t s7 = vreinterpretq_s16_u16(vmovl_u8(t0));
        int16x8_t s8 = vreinterpretq_s16_u16(vmovl_u8(t1));
        int16x8_t s9 = vreinterpretq_s16_u16(vmovl_u8(t2));
        int16x8_t s10 = vreinterpretq_s16_u16(vmovl_u8(t3));

        uint8x8_t d04 = convolve8_8(s0, s1, s2, s3, s4, s5, s6, s7, filters);
        uint8x8_t d15 = convolve8_8(s1, s2, s3, s4, s5, s6, s7, s8, filters);
        uint8x8_t d26 = convolve8_8(s2, s3, s4, s5, s6, s7, s8, s9, filters);
        uint8x8_t d37 = convolve8_8(s3, s4, s5, s6, s7, s8, s9, s10, filters);

        transpose_u8_8x4(&d04, &d15, &d26, &d37);

        uint8x8_t dd04 = load_u8(dst + 0 * dst_stride, 4 * dst_stride);
        uint8x8_t dd15 = load_u8(dst + 1 * dst_stride, 4 * dst_stride);
        uint8x8_t dd26 = load_u8(dst + 2 * dst_stride, 4 * dst_stride);
        uint8x8_t dd37 = load_u8(dst + 3 * dst_stride, 4 * dst_stride);

        d04 = vrhadd_u8(d04, dd04);
        d15 = vrhadd_u8(d15, dd15);
        d26 = vrhadd_u8(d26, dd26);
        d37 = vrhadd_u8(d37, dd37);

        store_u8(dst + 0 * dst_stride, 4 * dst_stride, d04);
        store_u8(dst + 1 * dst_stride, 4 * dst_stride, d15);
        store_u8(dst + 2 * dst_stride, 4 * dst_stride, d26);
        store_u8(dst + 3 * dst_stride, 4 * dst_stride, d37);

        src += 8 * src_stride;
        dst += 8 * dst_stride;
        h -= 8;
      } while (h != 0);
    } else {
      do {
        uint8x8_t t0, t1, t2, t3, t4, t5, t6, t7;
        load_u8_8x8(src, src_stride, &t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7);

        transpose_u8_8x8(&t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7);
        int16x8_t s0 = vreinterpretq_s16_u16(vmovl_u8(t0));
        int16x8_t s1 = vreinterpretq_s16_u16(vmovl_u8(t1));
        int16x8_t s2 = vreinterpretq_s16_u16(vmovl_u8(t2));
        int16x8_t s3 = vreinterpretq_s16_u16(vmovl_u8(t3));
        int16x8_t s4 = vreinterpretq_s16_u16(vmovl_u8(t4));
        int16x8_t s5 = vreinterpretq_s16_u16(vmovl_u8(t5));
        int16x8_t s6 = vreinterpretq_s16_u16(vmovl_u8(t6));

        const uint8_t *s = src + 7;
        uint8_t *d = dst;
        int width = w;

        do {
          uint8x8_t t8, t9, t10, t11, t12, t13, t14, t15;
          load_u8_8x8(s, src_stride, &t8, &t9, &t10, &t11, &t12, &t13, &t14,
                      &t15);

          transpose_u8_8x8(&t8, &t9, &t10, &t11, &t12, &t13, &t14, &t15);
          int16x8_t s7 = vreinterpretq_s16_u16(vmovl_u8(t8));
          int16x8_t s8 = vreinterpretq_s16_u16(vmovl_u8(t9));
          int16x8_t s9 = vreinterpretq_s16_u16(vmovl_u8(t10));
          int16x8_t s10 = vreinterpretq_s16_u16(vmovl_u8(t11));
          int16x8_t s11 = vreinterpretq_s16_u16(vmovl_u8(t12));
          int16x8_t s12 = vreinterpretq_s16_u16(vmovl_u8(t13));
          int16x8_t s13 = vreinterpretq_s16_u16(vmovl_u8(t14));
          int16x8_t s14 = vreinterpretq_s16_u16(vmovl_u8(t15));

          uint8x8_t d0 = convolve8_8(s0, s1, s2, s3, s4, s5, s6, s7, filters);
          uint8x8_t d1 = convolve8_8(s1, s2, s3, s4, s5, s6, s7, s8, filters);
          uint8x8_t d2 = convolve8_8(s2, s3, s4, s5, s6, s7, s8, s9, filters);
          uint8x8_t d3 = convolve8_8(s3, s4, s5, s6, s7, s8, s9, s10, filters);
          uint8x8_t d4 = convolve8_8(s4, s5, s6, s7, s8, s9, s10, s11, filters);
          uint8x8_t d5 =
              convolve8_8(s5, s6, s7, s8, s9, s10, s11, s12, filters);
          uint8x8_t d6 =
              convolve8_8(s6, s7, s8, s9, s10, s11, s12, s13, filters);
          uint8x8_t d7 =
              convolve8_8(s7, s8, s9, s10, s11, s12, s13, s14, filters);

          transpose_u8_8x8(&d0, &d1, &d2, &d3, &d4, &d5, &d6, &d7);

          d0 = vrhadd_u8(d0, vld1_u8(d + 0 * dst_stride));
          d1 = vrhadd_u8(d1, vld1_u8(d + 1 * dst_stride));
          d2 = vrhadd_u8(d2, vld1_u8(d + 2 * dst_stride));
          d3 = vrhadd_u8(d3, vld1_u8(d + 3 * dst_stride));
          d4 = vrhadd_u8(d4, vld1_u8(d + 4 * dst_stride));
          d5 = vrhadd_u8(d5, vld1_u8(d + 5 * dst_stride));
          d6 = vrhadd_u8(d6, vld1_u8(d + 6 * dst_stride));
          d7 = vrhadd_u8(d7, vld1_u8(d + 7 * dst_stride));

          store_u8_8x8(d, dst_stride, d0, d1, d2, d3, d4, d5, d6, d7);

          s0 = s8;
          s1 = s9;
          s2 = s10;
          s3 = s11;
          s4 = s12;
          s5 = s13;
          s6 = s14;
          s += 8;
          d += 8;
          width -= 8;
        } while (width != 0);
        src += 8 * src_stride;
        dst += 8 * dst_stride;
        h -= 8;
      } while (h != 0);
    }
  }
}

static INLINE void convolve_8tap_vert_neon(const uint8_t *src,
                                           ptrdiff_t src_stride, uint8_t *dst,
                                           ptrdiff_t dst_stride, int w, int h,
                                           const int16x8_t filter) {
  if (w == 4) {
    uint8x8_t t0, t1, t2, t3, t4, t5, t6;
    load_u8_8x7(src, src_stride, &t0, &t1, &t2, &t3, &t4, &t5, &t6);
    int16x4_t s0 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(t0)));
    int16x4_t s1 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(t1)));
    int16x4_t s2 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(t2)));
    int16x4_t s3 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(t3)));
    int16x4_t s4 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(t4)));
    int16x4_t s5 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(t5)));
    int16x4_t s6 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(t6)));

    src += 7 * src_stride;

    do {
      uint8x8_t t7, t8, t9, t10;
      load_u8_8x4(src, src_stride, &t7, &t8, &t9, &t10);
      int16x4_t s7 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(t7)));
      int16x4_t s8 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(t8)));
      int16x4_t s9 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(t9)));
      int16x4_t s10 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(t10)));

      int16x4_t d0 = convolve8_4(s0, s1, s2, s3, s4, s5, s6, s7, filter);
      int16x4_t d1 = convolve8_4(s1, s2, s3, s4, s5, s6, s7, s8, filter);
      int16x4_t d2 = convolve8_4(s2, s3, s4, s5, s6, s7, s8, s9, filter);
      int16x4_t d3 = convolve8_4(s3, s4, s5, s6, s7, s8, s9, s10, filter);
      uint8x8_t d01 = vqrshrun_n_s16(vcombine_s16(d0, d1), FILTER_BITS);
      uint8x8_t d23 = vqrshrun_n_s16(vcombine_s16(d2, d3), FILTER_BITS);

      store_u8(dst + 0 * dst_stride, dst_stride, d01);
      store_u8(dst + 2 * dst_stride, dst_stride, d23);

      s0 = s4;
      s1 = s5;
      s2 = s6;
      s3 = s7;
      s4 = s8;
      s5 = s9;
      s6 = s10;
      src += 4 * src_stride;
      dst += 4 * dst_stride;
      h -= 4;
    } while (h != 0);
  } else {
    do {
      uint8x8_t t0, t1, t2, t3, t4, t5, t6;
      load_u8_8x7(src, src_stride, &t0, &t1, &t2, &t3, &t4, &t5, &t6);
      int16x8_t s0 = vreinterpretq_s16_u16(vmovl_u8(t0));
      int16x8_t s1 = vreinterpretq_s16_u16(vmovl_u8(t1));
      int16x8_t s2 = vreinterpretq_s16_u16(vmovl_u8(t2));
      int16x8_t s3 = vreinterpretq_s16_u16(vmovl_u8(t3));
      int16x8_t s4 = vreinterpretq_s16_u16(vmovl_u8(t4));
      int16x8_t s5 = vreinterpretq_s16_u16(vmovl_u8(t5));
      int16x8_t s6 = vreinterpretq_s16_u16(vmovl_u8(t6));

      const uint8_t *s = src + 7 * src_stride;
      uint8_t *d = dst;
      int height = h;

      do {
        uint8x8_t t7, t8, t9, t10;
        load_u8_8x4(s, src_stride, &t7, &t8, &t9, &t10);
        int16x8_t s7 = vreinterpretq_s16_u16(vmovl_u8(t7));
        int16x8_t s8 = vreinterpretq_s16_u16(vmovl_u8(t8));
        int16x8_t s9 = vreinterpretq_s16_u16(vmovl_u8(t9));
        int16x8_t s10 = vreinterpretq_s16_u16(vmovl_u8(t10));

        uint8x8_t d0 = convolve8_8(s0, s1, s2, s3, s4, s5, s6, s7, filter);
        uint8x8_t d1 = convolve8_8(s1, s2, s3, s4, s5, s6, s7, s8, filter);
        uint8x8_t d2 = convolve8_8(s2, s3, s4, s5, s6, s7, s8, s9, filter);
        uint8x8_t d3 = convolve8_8(s3, s4, s5, s6, s7, s8, s9, s10, filter);

        store_u8_8x4(d, dst_stride, d0, d1, d2, d3);

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

void vpx_convolve8_vert_neon(const uint8_t *src, ptrdiff_t src_stride,
                             uint8_t *dst, ptrdiff_t dst_stride,
                             const InterpKernel *filter, int x0_q4,
                             int x_step_q4, int y0_q4, int y_step_q4, int w,
                             int h) {
  assert((intptr_t)dst % 4 == 0);
  assert(dst_stride % 4 == 0);
  assert(y_step_q4 == 16);

  (void)x0_q4;
  (void)x_step_q4;
  (void)y_step_q4;

  const int16x8_t y_filter = vld1q_s16(filter[y0_q4]);

  if (vpx_get_filter_taps(filter[y0_q4]) <= 4) {
    convolve_4tap_vert_neon(src - src_stride, src_stride, dst, dst_stride, w, h,
                            y_filter);
  } else {
    convolve_8tap_vert_neon(src - 3 * src_stride, src_stride, dst, dst_stride,
                            w, h, y_filter);
  }
}

void vpx_convolve8_avg_vert_neon(const uint8_t *src, ptrdiff_t src_stride,
                                 uint8_t *dst, ptrdiff_t dst_stride,
                                 const InterpKernel *filter, int x0_q4,
                                 int x_step_q4, int y0_q4, int y_step_q4, int w,
                                 int h) {
  const int16x8_t filters = vld1q_s16(filter[y0_q4]);

  assert((intptr_t)dst % 4 == 0);
  assert(dst_stride % 4 == 0);
  assert(y_step_q4 == 16);

  (void)x0_q4;
  (void)x_step_q4;
  (void)y_step_q4;

  src -= 3 * src_stride;

  if (w == 4) {
    uint8x8_t t0, t1, t2, t3, t4, t5, t6;
    load_u8_8x7(src, src_stride, &t0, &t1, &t2, &t3, &t4, &t5, &t6);
    int16x4_t s0 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(t0)));
    int16x4_t s1 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(t1)));
    int16x4_t s2 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(t2)));
    int16x4_t s3 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(t3)));
    int16x4_t s4 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(t4)));
    int16x4_t s5 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(t5)));
    int16x4_t s6 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(t6)));

    src += 7 * src_stride;

    do {
      uint8x8_t t7, t8, t9, t10;
      load_u8_8x4(src, src_stride, &t7, &t8, &t9, &t10);
      int16x4_t s7 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(t7)));
      int16x4_t s8 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(t8)));
      int16x4_t s9 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(t9)));
      int16x4_t s10 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(t10)));

      int16x4_t d0 = convolve8_4(s0, s1, s2, s3, s4, s5, s6, s7, filters);
      int16x4_t d1 = convolve8_4(s1, s2, s3, s4, s5, s6, s7, s8, filters);
      int16x4_t d2 = convolve8_4(s2, s3, s4, s5, s6, s7, s8, s9, filters);
      int16x4_t d3 = convolve8_4(s3, s4, s5, s6, s7, s8, s9, s10, filters);
      uint8x8_t d01 = vqrshrun_n_s16(vcombine_s16(d0, d1), FILTER_BITS);
      uint8x8_t d23 = vqrshrun_n_s16(vcombine_s16(d2, d3), FILTER_BITS);

      uint8x8_t dd01 = load_u8(dst + 0 * dst_stride, dst_stride);
      uint8x8_t dd23 = load_u8(dst + 2 * dst_stride, dst_stride);

      d01 = vrhadd_u8(d01, dd01);
      d23 = vrhadd_u8(d23, dd23);

      store_u8(dst + 0 * dst_stride, dst_stride, d01);
      store_u8(dst + 2 * dst_stride, dst_stride, d23);

      s0 = s4;
      s1 = s5;
      s2 = s6;
      s3 = s7;
      s4 = s8;
      s5 = s9;
      s6 = s10;
      src += 4 * src_stride;
      dst += 4 * dst_stride;
      h -= 4;
    } while (h != 0);
  } else {
    do {
      uint8x8_t t0, t1, t2, t3, t4, t5, t6;
      load_u8_8x7(src, src_stride, &t0, &t1, &t2, &t3, &t4, &t5, &t6);
      int16x8_t s0 = vreinterpretq_s16_u16(vmovl_u8(t0));
      int16x8_t s1 = vreinterpretq_s16_u16(vmovl_u8(t1));
      int16x8_t s2 = vreinterpretq_s16_u16(vmovl_u8(t2));
      int16x8_t s3 = vreinterpretq_s16_u16(vmovl_u8(t3));
      int16x8_t s4 = vreinterpretq_s16_u16(vmovl_u8(t4));
      int16x8_t s5 = vreinterpretq_s16_u16(vmovl_u8(t5));
      int16x8_t s6 = vreinterpretq_s16_u16(vmovl_u8(t6));

      const uint8_t *s = src + 7 * src_stride;
      uint8_t *d = dst;
      int height = h;

      do {
        uint8x8_t t7, t8, t9, t10;
        load_u8_8x4(s, src_stride, &t7, &t8, &t9, &t10);
        int16x8_t s7 = vreinterpretq_s16_u16(vmovl_u8(t7));
        int16x8_t s8 = vreinterpretq_s16_u16(vmovl_u8(t8));
        int16x8_t s9 = vreinterpretq_s16_u16(vmovl_u8(t9));
        int16x8_t s10 = vreinterpretq_s16_u16(vmovl_u8(t10));

        uint8x8_t d0 = convolve8_8(s0, s1, s2, s3, s4, s5, s6, s7, filters);
        uint8x8_t d1 = convolve8_8(s1, s2, s3, s4, s5, s6, s7, s8, filters);
        uint8x8_t d2 = convolve8_8(s2, s3, s4, s5, s6, s7, s8, s9, filters);
        uint8x8_t d3 = convolve8_8(s3, s4, s5, s6, s7, s8, s9, s10, filters);

        d0 = vrhadd_u8(d0, vld1_u8(d + 0 * dst_stride));
        d1 = vrhadd_u8(d1, vld1_u8(d + 1 * dst_stride));
        d2 = vrhadd_u8(d2, vld1_u8(d + 2 * dst_stride));
        d3 = vrhadd_u8(d3, vld1_u8(d + 3 * dst_stride));

        store_u8_8x4(d, dst_stride, d0, d1, d2, d3);

        s0 = s4;
        s1 = s5;
        s2 = s6;
        s3 = s7;
        s4 = s8;
        s5 = s9;
        s6 = s10;
        height -= 4;
        s += 4 * src_stride;
        d += 4 * dst_stride;
      } while (height != 0);
      src += 8;
      dst += 8;
      w -= 8;
    } while (w != 0);
  }
}
