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

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/arm/mem_neon.h"

void vpx_comp_avg_pred_neon(uint8_t *comp, const uint8_t *pred, int width,
                            int height, const uint8_t *ref, int ref_stride) {
  if (width > 8) {
    int x, y = height;
    do {
      for (x = 0; x < width; x += 16) {
        const uint8x16_t p = vld1q_u8(pred + x);
        const uint8x16_t r = vld1q_u8(ref + x);
        const uint8x16_t avg = vrhaddq_u8(p, r);
        vst1q_u8(comp + x, avg);
      }
      comp += width;
      pred += width;
      ref += ref_stride;
    } while (--y);
  } else if (width == 8) {
    int i = width * height;
    do {
      const uint8x16_t p = vld1q_u8(pred);
      uint8x16_t r;
      const uint8x8_t r_0 = vld1_u8(ref);
      const uint8x8_t r_1 = vld1_u8(ref + ref_stride);
      r = vcombine_u8(r_0, r_1);
      ref += 2 * ref_stride;
      r = vrhaddq_u8(r, p);
      vst1q_u8(comp, r);

      pred += 16;
      comp += 16;
      i -= 16;
    } while (i);
  } else {
    int i = width * height;
    assert(width == 4);
    do {
      const uint8x16_t p = vld1q_u8(pred);
      uint8x16_t r;

      r = load_unaligned_u8q(ref, ref_stride);
      ref += 4 * ref_stride;
      r = vrhaddq_u8(r, p);
      vst1q_u8(comp, r);

      pred += 16;
      comp += 16;
      i -= 16;
    } while (i);
  }
}
