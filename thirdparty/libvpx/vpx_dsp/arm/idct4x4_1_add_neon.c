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

#include "vpx_dsp/inv_txfm.h"
#include "vpx_ports/mem.h"

void vpx_idct4x4_1_add_neon(
        int16_t *input,
        uint8_t *dest,
        int dest_stride) {
    uint8x8_t d6u8;
    uint32x2_t d2u32 = vdup_n_u32(0);
    uint16x8_t q8u16;
    int16x8_t q0s16;
    uint8_t *d1, *d2;
    int16_t i, a1, cospi_16_64 = 11585;
    int16_t out = dct_const_round_shift(input[0] * cospi_16_64);
    out = dct_const_round_shift(out * cospi_16_64);
    a1 = ROUND_POWER_OF_TWO(out, 4);

    q0s16 = vdupq_n_s16(a1);

    // dc_only_idct_add
    d1 = d2 = dest;
    for (i = 0; i < 2; i++) {
        d2u32 = vld1_lane_u32((const uint32_t *)d1, d2u32, 0);
        d1 += dest_stride;
        d2u32 = vld1_lane_u32((const uint32_t *)d1, d2u32, 1);
        d1 += dest_stride;

        q8u16 = vaddw_u8(vreinterpretq_u16_s16(q0s16),
                         vreinterpret_u8_u32(d2u32));
        d6u8 = vqmovun_s16(vreinterpretq_s16_u16(q8u16));

        vst1_lane_u32((uint32_t *)d2, vreinterpret_u32_u8(d6u8), 0);
        d2 += dest_stride;
        vst1_lane_u32((uint32_t *)d2, vreinterpret_u32_u8(d6u8), 1);
        d2 += dest_stride;
    }
    return;
}
