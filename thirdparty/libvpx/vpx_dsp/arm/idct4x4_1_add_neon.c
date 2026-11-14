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

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/arm/mem_neon.h"
#include "vpx_dsp/inv_txfm.h"

static INLINE void idct4x4_1_add_kernel(uint8_t **dest, const int stride,
                                        const int16x8_t res,
                                        uint32x2_t *const d) {
  uint16x8_t a;
  uint8x8_t b;
  *d = vld1_lane_u32((const uint32_t *)*dest, *d, 0);
  *d = vld1_lane_u32((const uint32_t *)(*dest + stride), *d, 1);
  a = vaddw_u8(vreinterpretq_u16_s16(res), vreinterpret_u8_u32(*d));
  b = vqmovun_s16(vreinterpretq_s16_u16(a));
  vst1_lane_u32((uint32_t *)*dest, vreinterpret_u32_u8(b), 0);
  *dest += stride;
  vst1_lane_u32((uint32_t *)*dest, vreinterpret_u32_u8(b), 1);
  *dest += stride;
}

void vpx_idct4x4_1_add_neon(const tran_low_t *input, uint8_t *dest,
                            int stride) {
  const int16_t out0 =
      WRAPLOW(dct_const_round_shift((int16_t)input[0] * cospi_16_64));
  const int16_t out1 = WRAPLOW(dct_const_round_shift(out0 * cospi_16_64));
  const int16_t a1 = ROUND_POWER_OF_TWO(out1, 4);
  const int16x8_t dc = vdupq_n_s16(a1);
  uint32x2_t d = vdup_n_u32(0);

  assert(!((intptr_t)dest % sizeof(uint32_t)));
  assert(!(stride % sizeof(uint32_t)));

  idct4x4_1_add_kernel(&dest, stride, dc, &d);
  idct4x4_1_add_kernel(&dest, stride, dc, &d);
}
