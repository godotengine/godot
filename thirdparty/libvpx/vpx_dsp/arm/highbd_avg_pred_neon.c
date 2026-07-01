/*
 *  Copyright (c) 2023 The WebM project authors. All Rights Reserved.
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
#include "./vpx_config.h"

void vpx_highbd_comp_avg_pred_neon(uint16_t *comp_pred, const uint16_t *pred,
                                   int width, int height, const uint16_t *ref,
                                   int ref_stride) {
  int i = height;
  if (width > 8) {
    do {
      int j = 0;
      do {
        const uint16x8_t p = vld1q_u16(pred + j);
        const uint16x8_t r = vld1q_u16(ref + j);

        uint16x8_t avg = vrhaddq_u16(p, r);
        vst1q_u16(comp_pred + j, avg);

        j += 8;
      } while (j < width);

      comp_pred += width;
      pred += width;
      ref += ref_stride;
    } while (--i != 0);
  } else if (width == 8) {
    do {
      const uint16x8_t p = vld1q_u16(pred);
      const uint16x8_t r = vld1q_u16(ref);

      uint16x8_t avg = vrhaddq_u16(p, r);
      vst1q_u16(comp_pred, avg);

      comp_pred += width;
      pred += width;
      ref += ref_stride;
    } while (--i != 0);
  } else {
    assert(width == 4);
    do {
      const uint16x4_t p = vld1_u16(pred);
      const uint16x4_t r = vld1_u16(ref);

      uint16x4_t avg = vrhadd_u16(p, r);
      vst1_u16(comp_pred, avg);

      comp_pred += width;
      pred += width;
      ref += ref_stride;
    } while (--i != 0);
  }
}
