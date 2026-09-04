/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <arm_neon.h>
#include <assert.h>
#include <string.h>

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/arm/mem_neon.h"
#include "vpx_dsp/arm/transpose_neon.h"
#include "vpx_dsp/arm/vpx_convolve8_neon.h"
#include "vpx_dsp/vpx_filter.h"
#include "vpx_ports/mem.h"

static INLINE void scaledconvolve_horiz_neon(
    const uint8_t *src, const ptrdiff_t src_stride, uint8_t *dst,
    const ptrdiff_t dst_stride, const InterpKernel *const x_filter,
    const int x0_q4, const int x_step_q4, int w, int h) {
  DECLARE_ALIGNED(16, uint8_t, temp[8 * 8]);

  src -= SUBPEL_TAPS / 2 - 1;

  if (w == 4) {
    do {
      int x_q4 = x0_q4;

      // Process a 4x4 tile.
      for (int r = 0; r < 4; ++r) {
        const uint8_t *s = &src[x_q4 >> SUBPEL_BITS];

        if (x_q4 & SUBPEL_MASK) {
          const int16x8_t filter = vld1q_s16(x_filter[x_q4 & SUBPEL_MASK]);

          uint8x8_t t0, t1, t2, t3;
          load_u8_8x4(s, src_stride, &t0, &t1, &t2, &t3);
          transpose_u8_8x4(&t0, &t1, &t2, &t3);

          int16x4_t s0 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t0)));
          int16x4_t s1 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t1)));
          int16x4_t s2 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t2)));
          int16x4_t s3 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t3)));
          int16x4_t s4 = vget_high_s16(vreinterpretq_s16_u16(vmovl_u8(t0)));
          int16x4_t s5 = vget_high_s16(vreinterpretq_s16_u16(vmovl_u8(t1)));
          int16x4_t s6 = vget_high_s16(vreinterpretq_s16_u16(vmovl_u8(t2)));
          int16x4_t s7 = vget_high_s16(vreinterpretq_s16_u16(vmovl_u8(t3)));

          int16x4_t dd0 = convolve8_4(s0, s1, s2, s3, s4, s5, s6, s7, filter);
          uint8x8_t d0 =
              vqrshrun_n_s16(vcombine_s16(dd0, vdup_n_s16(0)), FILTER_BITS);

          store_u8_4x1(&temp[4 * r], d0);
        } else {
          // Memcpy for non-subpel locations.
          s += SUBPEL_TAPS / 2 - 1;

          for (int c = 0; c < 4; ++c) {
            temp[r * 4 + c] = s[c * src_stride];
          }
        }
        x_q4 += x_step_q4;
      }

      // Transpose the 4x4 result tile and store.
      uint8x8_t d01 = vld1_u8(temp + 0);
      uint8x8_t d23 = vld1_u8(temp + 8);

      transpose_u8_4x4(&d01, &d23);

      store_u8_4x1(dst + 0 * dst_stride, d01);
      store_u8_4x1(dst + 1 * dst_stride, d23);
      store_u8_4x1_high(dst + 2 * dst_stride, d01);
      store_u8_4x1_high(dst + 3 * dst_stride, d23);

      src += 4 * src_stride;
      dst += 4 * dst_stride;
      h -= 4;
    } while (h > 0);
    return;
  }

  do {
    int x_q4 = x0_q4;
    uint8_t *d = dst;
    int width = w;

    do {
      // Process an 8x8 tile.
      for (int r = 0; r < 8; ++r) {
        const uint8_t *s = &src[x_q4 >> SUBPEL_BITS];

        if (x_q4 & SUBPEL_MASK) {
          const int16x8_t filter = vld1q_s16(x_filter[x_q4 & SUBPEL_MASK]);

          uint8x8_t t0, t1, t2, t3, t4, t5, t6, t7;
          load_u8_8x8(s, src_stride, &t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7);

          transpose_u8_8x8(&t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7);
          int16x8_t s0 = vreinterpretq_s16_u16(vmovl_u8(t0));
          int16x8_t s1 = vreinterpretq_s16_u16(vmovl_u8(t1));
          int16x8_t s2 = vreinterpretq_s16_u16(vmovl_u8(t2));
          int16x8_t s3 = vreinterpretq_s16_u16(vmovl_u8(t3));
          int16x8_t s4 = vreinterpretq_s16_u16(vmovl_u8(t4));
          int16x8_t s5 = vreinterpretq_s16_u16(vmovl_u8(t5));
          int16x8_t s6 = vreinterpretq_s16_u16(vmovl_u8(t6));
          int16x8_t s7 = vreinterpretq_s16_u16(vmovl_u8(t7));

          uint8x8_t d0 = convolve8_8(s0, s1, s2, s3, s4, s5, s6, s7, filter);

          vst1_u8(&temp[r * 8], d0);
        } else {
          // Memcpy for non-subpel locations.
          s += SUBPEL_TAPS / 2 - 1;

          for (int c = 0; c < 8; ++c) {
            temp[r * 8 + c] = s[c * src_stride];
          }
        }
        x_q4 += x_step_q4;
      }

      // Transpose the 8x8 result tile and store.
      uint8x8_t d0, d1, d2, d3, d4, d5, d6, d7;
      load_u8_8x8(temp, 8, &d0, &d1, &d2, &d3, &d4, &d5, &d6, &d7);

      transpose_u8_8x8(&d0, &d1, &d2, &d3, &d4, &d5, &d6, &d7);

      store_u8_8x8(d, dst_stride, d0, d1, d2, d3, d4, d5, d6, d7);

      d += 8;
      width -= 8;
    } while (width != 0);

    src += 8 * src_stride;
    dst += 8 * dst_stride;
    h -= 8;
  } while (h > 0);
}

static INLINE void scaledconvolve_vert_neon(
    const uint8_t *src, const ptrdiff_t src_stride, uint8_t *dst,
    const ptrdiff_t dst_stride, const InterpKernel *const y_filter,
    const int y0_q4, const int y_step_q4, int w, int h) {
  int y_q4 = y0_q4;

  if (w == 4) {
    do {
      const uint8_t *s = &src[(y_q4 >> SUBPEL_BITS) * src_stride];

      if (y_q4 & SUBPEL_MASK) {
        const int16x8_t filter = vld1q_s16(y_filter[y_q4 & SUBPEL_MASK]);

        uint8x8_t t0, t1, t2, t3, t4, t5, t6, t7;
        load_u8_8x8(s, src_stride, &t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7);
        int16x4_t s0 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t0)));
        int16x4_t s1 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t1)));
        int16x4_t s2 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t2)));
        int16x4_t s3 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t3)));
        int16x4_t s4 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t4)));
        int16x4_t s5 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t5)));
        int16x4_t s6 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t6)));
        int16x4_t s7 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t7)));

        int16x4_t dd0 = convolve8_4(s0, s1, s2, s3, s4, s5, s6, s7, filter);
        uint8x8_t d0 =
            vqrshrun_n_s16(vcombine_s16(dd0, vdup_n_s16(0)), FILTER_BITS);

        store_u8_4x1(dst, d0);
      } else {
        // Memcpy for non-subpel locations.
        memcpy(dst, &s[(SUBPEL_TAPS / 2 - 1) * src_stride], 4);
      }

      y_q4 += y_step_q4;
      dst += dst_stride;
    } while (--h != 0);
    return;
  }

  if (w == 8) {
    do {
      const uint8_t *s = &src[(y_q4 >> SUBPEL_BITS) * src_stride];

      if (y_q4 & SUBPEL_MASK) {
        const int16x8_t filter = vld1q_s16(y_filter[y_q4 & SUBPEL_MASK]);

        uint8x8_t t0, t1, t2, t3, t4, t5, t6, t7;
        load_u8_8x8(s, src_stride, &t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7);
        int16x8_t s0 = vreinterpretq_s16_u16(vmovl_u8(t0));
        int16x8_t s1 = vreinterpretq_s16_u16(vmovl_u8(t1));
        int16x8_t s2 = vreinterpretq_s16_u16(vmovl_u8(t2));
        int16x8_t s3 = vreinterpretq_s16_u16(vmovl_u8(t3));
        int16x8_t s4 = vreinterpretq_s16_u16(vmovl_u8(t4));
        int16x8_t s5 = vreinterpretq_s16_u16(vmovl_u8(t5));
        int16x8_t s6 = vreinterpretq_s16_u16(vmovl_u8(t6));
        int16x8_t s7 = vreinterpretq_s16_u16(vmovl_u8(t7));

        uint8x8_t d0 = convolve8_8(s0, s1, s2, s3, s4, s5, s6, s7, filter);

        vst1_u8(dst, d0);
      } else {
        // Memcpy for non-subpel locations.
        memcpy(dst, &s[(SUBPEL_TAPS / 2 - 1) * src_stride], 8);
      }

      y_q4 += y_step_q4;
      dst += dst_stride;
    } while (--h != 0);
    return;
  }

  do {
    const uint8_t *s = &src[(y_q4 >> SUBPEL_BITS) * src_stride];
    uint8_t *d = dst;
    int width = w;

    if (y_q4 & SUBPEL_MASK) {
      do {
        const int16x8_t filter = vld1q_s16(y_filter[y_q4 & SUBPEL_MASK]);

        uint8x16_t t0, t1, t2, t3, t4, t5, t6, t7;
        load_u8_16x8(s, src_stride, &t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7);

        int16x8_t s0[2], s1[2], s2[2], s3[2], s4[2], s5[2], s6[2], s7[2];
        s0[0] = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(t0)));
        s1[0] = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(t1)));
        s2[0] = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(t2)));
        s3[0] = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(t3)));
        s4[0] = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(t4)));
        s5[0] = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(t5)));
        s6[0] = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(t6)));
        s7[0] = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(t7)));

        s0[1] = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(t0)));
        s1[1] = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(t1)));
        s2[1] = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(t2)));
        s3[1] = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(t3)));
        s4[1] = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(t4)));
        s5[1] = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(t5)));
        s6[1] = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(t6)));
        s7[1] = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(t7)));

        uint8x8_t d0 = convolve8_8(s0[0], s1[0], s2[0], s3[0], s4[0], s5[0],
                                   s6[0], s7[0], filter);
        uint8x8_t d1 = convolve8_8(s0[1], s1[1], s2[1], s3[1], s4[1], s5[1],
                                   s6[1], s7[1], filter);

        vst1q_u8(d, vcombine_u8(d0, d1));

        s += 16;
        d += 16;
        width -= 16;
      } while (width != 0);
    } else {
      // Memcpy for non-subpel locations.
      s += (SUBPEL_TAPS / 2 - 1) * src_stride;

      do {
        uint8x16_t s0 = vld1q_u8(s);
        vst1q_u8(d, s0);
        s += 16;
        d += 16;
        width -= 16;
      } while (width != 0);
    }

    y_q4 += y_step_q4;
    dst += dst_stride;
  } while (--h != 0);
}

void vpx_scaled_2d_neon(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst,
                        ptrdiff_t dst_stride, const InterpKernel *filter,
                        int x0_q4, int x_step_q4, int y0_q4, int y_step_q4,
                        int w, int h) {
  // Fixed size intermediate buffer, im_block, places limits on parameters.
  // 2d filtering proceeds in 2 steps:
  //   (1) Interpolate horizontally into an intermediate buffer, temp.
  //   (2) Interpolate temp vertically to derive the sub-pixel result.
  // Deriving the maximum number of rows in the im_block buffer (135):
  // --Smallest scaling factor is x1/2 ==> y_step_q4 = 32 (Normative).
  // --Largest block size is 64x64 pixels.
  // --64 rows in the downscaled frame span a distance of (64 - 1) * 32 in the
  //   original frame (in 1/16th pixel units).
  // --Must round-up because block may be located at sub-pixel position.
  // --Require an additional SUBPEL_TAPS rows for the 8-tap filter tails.
  // --((64 - 1) * 32 + 15) >> 4 + 8 = 135.
  // --Require an additional 8 rows for the horiz_w8 transpose tail.
  // When calling in frame scaling function, the smallest scaling factor is x1/4
  // ==> y_step_q4 = 64. Since w and h are at most 16, the temp buffer is still
  // big enough.
  DECLARE_ALIGNED(16, uint8_t, im_block[(135 + 8) * 64]);
  const int im_height =
      (((h - 1) * y_step_q4 + y0_q4) >> SUBPEL_BITS) + SUBPEL_TAPS;
  const ptrdiff_t im_stride = 64;

  assert(w <= 64);
  assert(h <= 64);
  assert(y_step_q4 <= 32 || (y_step_q4 <= 64 && h <= 32));
  assert(x_step_q4 <= 64);

  scaledconvolve_horiz_neon(src - src_stride * (SUBPEL_TAPS / 2 - 1),
                            src_stride, im_block, im_stride, filter, x0_q4,
                            x_step_q4, w, im_height);

  scaledconvolve_vert_neon(im_block, im_stride, dst, dst_stride, filter, y0_q4,
                           y_step_q4, w, h);
}
