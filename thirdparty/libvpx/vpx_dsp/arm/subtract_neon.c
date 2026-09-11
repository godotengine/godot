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

void vpx_subtract_block_neon(int rows, int cols, int16_t *diff,
                             ptrdiff_t diff_stride, const uint8_t *src,
                             ptrdiff_t src_stride, const uint8_t *pred,
                             ptrdiff_t pred_stride) {
  int r = rows, c;

  if (cols > 16) {
    do {
      for (c = 0; c < cols; c += 32) {
        const uint8x16_t s0 = vld1q_u8(&src[c + 0]);
        const uint8x16_t s1 = vld1q_u8(&src[c + 16]);
        const uint8x16_t p0 = vld1q_u8(&pred[c + 0]);
        const uint8x16_t p1 = vld1q_u8(&pred[c + 16]);
        const uint16x8_t d0 = vsubl_u8(vget_low_u8(s0), vget_low_u8(p0));
        const uint16x8_t d1 = vsubl_u8(vget_high_u8(s0), vget_high_u8(p0));
        const uint16x8_t d2 = vsubl_u8(vget_low_u8(s1), vget_low_u8(p1));
        const uint16x8_t d3 = vsubl_u8(vget_high_u8(s1), vget_high_u8(p1));
        vst1q_s16(&diff[c + 0], vreinterpretq_s16_u16(d0));
        vst1q_s16(&diff[c + 8], vreinterpretq_s16_u16(d1));
        vst1q_s16(&diff[c + 16], vreinterpretq_s16_u16(d2));
        vst1q_s16(&diff[c + 24], vreinterpretq_s16_u16(d3));
      }
      diff += diff_stride;
      pred += pred_stride;
      src += src_stride;
    } while (--r);
  } else if (cols > 8) {
    do {
      const uint8x16_t s = vld1q_u8(&src[0]);
      const uint8x16_t p = vld1q_u8(&pred[0]);
      const uint16x8_t d0 = vsubl_u8(vget_low_u8(s), vget_low_u8(p));
      const uint16x8_t d1 = vsubl_u8(vget_high_u8(s), vget_high_u8(p));
      vst1q_s16(&diff[0], vreinterpretq_s16_u16(d0));
      vst1q_s16(&diff[8], vreinterpretq_s16_u16(d1));
      diff += diff_stride;
      pred += pred_stride;
      src += src_stride;
    } while (--r);
  } else if (cols > 4) {
    do {
      const uint8x8_t s = vld1_u8(&src[0]);
      const uint8x8_t p = vld1_u8(&pred[0]);
      const uint16x8_t v_diff = vsubl_u8(s, p);
      vst1q_s16(&diff[0], vreinterpretq_s16_u16(v_diff));
      diff += diff_stride;
      pred += pred_stride;
      src += src_stride;
    } while (--r);
  } else {
    assert(cols == 4);
    do {
      const uint8x8_t s = load_unaligned_u8(src, (int)src_stride);
      const uint8x8_t p = load_unaligned_u8(pred, (int)pred_stride);
      const uint16x8_t d = vsubl_u8(s, p);
      vst1_s16(diff + 0 * diff_stride, vreinterpret_s16_u16(vget_low_u16(d)));
      vst1_s16(diff + 1 * diff_stride, vreinterpret_s16_u16(vget_high_u16(d)));
      diff += 2 * diff_stride;
      pred += 2 * pred_stride;
      src += 2 * src_stride;
      r -= 2;
    } while (r);
  }
}

#if CONFIG_VP9_HIGHBITDEPTH
void vpx_highbd_subtract_block_neon(int rows, int cols, int16_t *diff_ptr,
                                    ptrdiff_t diff_stride,
                                    const uint8_t *src8_ptr,
                                    ptrdiff_t src_stride,
                                    const uint8_t *pred8_ptr,
                                    ptrdiff_t pred_stride, int bd) {
  int r = rows, c;
  uint16_t *src = CONVERT_TO_SHORTPTR(src8_ptr);
  uint16_t *pred = CONVERT_TO_SHORTPTR(pred8_ptr);
  (void)bd;

  if (cols >= 16) {
    do {
      for (c = 0; c < cols; c += 16) {
        const uint16x8_t s0 = vld1q_u16(&src[c + 0]);
        const uint16x8_t s1 = vld1q_u16(&src[c + 8]);
        const uint16x8_t p0 = vld1q_u16(&pred[c + 0]);
        const uint16x8_t p1 = vld1q_u16(&pred[c + 8]);
        const uint16x8_t d0 = vsubq_u16(s0, p0);
        const uint16x8_t d1 = vsubq_u16(s1, p1);
        vst1q_s16(&diff_ptr[c + 0], vreinterpretq_s16_u16(d0));
        vst1q_s16(&diff_ptr[c + 8], vreinterpretq_s16_u16(d1));
      }
      diff_ptr += diff_stride;
      pred += pred_stride;
      src += src_stride;
    } while (--r);
  } else if (cols >= 8) {
    do {
      for (c = 0; c < cols; c += 8) {
        const uint16x8_t s = vld1q_u16(&src[c]);
        const uint16x8_t p = vld1q_u16(&pred[c]);
        const uint16x8_t d0 = vsubq_u16(s, p);
        vst1q_s16(&diff_ptr[c], vreinterpretq_s16_u16(d0));
      }
      diff_ptr += diff_stride;
      pred += pred_stride;
      src += src_stride;
    } while (--r);
  } else if (cols >= 4) {
    do {
      for (c = 0; c < cols; c += 4) {
        const uint16x4_t s = vld1_u16(&src[c]);
        const uint16x4_t p = vld1_u16(&pred[c]);
        const uint16x4_t v_diff = vsub_u16(s, p);
        vst1_s16(&diff_ptr[c], vreinterpret_s16_u16(v_diff));
      }
      diff_ptr += diff_stride;
      pred += pred_stride;
      src += src_stride;
    } while (--r);
  }
}
#endif  // CONFIG_VP9_HIGHBITDEPTH
