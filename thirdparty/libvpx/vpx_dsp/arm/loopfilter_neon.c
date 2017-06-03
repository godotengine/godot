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
#include "./vpx_config.h"
#include "vpx/vpx_integer.h"

void vpx_lpf_vertical_4_dual_neon(uint8_t *s, int p,
                                  const uint8_t *blimit0,
                                  const uint8_t *limit0,
                                  const uint8_t *thresh0,
                                  const uint8_t *blimit1,
                                  const uint8_t *limit1,
                                  const uint8_t *thresh1) {
  vpx_lpf_vertical_4_neon(s, p, blimit0, limit0, thresh0);
  vpx_lpf_vertical_4_neon(s + 8 * p, p, blimit1, limit1, thresh1);
}

#if HAVE_NEON_ASM
void vpx_lpf_horizontal_8_dual_neon(uint8_t *s, int p /* pitch */,
                                    const uint8_t *blimit0,
                                    const uint8_t *limit0,
                                    const uint8_t *thresh0,
                                    const uint8_t *blimit1,
                                    const uint8_t *limit1,
                                    const uint8_t *thresh1) {
  vpx_lpf_horizontal_8_neon(s, p, blimit0, limit0, thresh0);
  vpx_lpf_horizontal_8_neon(s + 8, p, blimit1, limit1, thresh1);
}

void vpx_lpf_vertical_8_dual_neon(uint8_t *s, int p,
                                  const uint8_t *blimit0,
                                  const uint8_t *limit0,
                                  const uint8_t *thresh0,
                                  const uint8_t *blimit1,
                                  const uint8_t *limit1,
                                  const uint8_t *thresh1) {
  vpx_lpf_vertical_8_neon(s, p, blimit0, limit0, thresh0);
  vpx_lpf_vertical_8_neon(s + 8 * p, p, blimit1, limit1, thresh1);
}

void vpx_lpf_vertical_16_dual_neon(uint8_t *s, int p,
                                   const uint8_t *blimit,
                                   const uint8_t *limit,
                                   const uint8_t *thresh) {
  vpx_lpf_vertical_16_neon(s, p, blimit, limit, thresh);
  vpx_lpf_vertical_16_neon(s + 8 * p, p, blimit, limit, thresh);
}
#endif  // HAVE_NEON_ASM
