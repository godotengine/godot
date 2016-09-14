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

void vpx_idct16x16_1_add_neon(
        int16_t *input,
        uint8_t *dest,
        int dest_stride) {
    uint8x8_t d2u8, d3u8, d30u8, d31u8;
    uint64x1_t d2u64, d3u64, d4u64, d5u64;
    uint16x8_t q0u16, q9u16, q10u16, q11u16, q12u16;
    int16x8_t q0s16;
    uint8_t *d1, *d2;
    int16_t i, j, a1, cospi_16_64 = 11585;
    int16_t out = dct_const_round_shift(input[0] * cospi_16_64);
    out = dct_const_round_shift(out * cospi_16_64);
    a1 = ROUND_POWER_OF_TWO(out, 6);

    q0s16 = vdupq_n_s16(a1);
    q0u16 = vreinterpretq_u16_s16(q0s16);

    for (d1 = d2 = dest, i = 0; i < 4; i++) {
        for (j = 0; j < 2; j++) {
            d2u64 = vld1_u64((const uint64_t *)d1);
            d3u64 = vld1_u64((const uint64_t *)(d1 + 8));
            d1 += dest_stride;
            d4u64 = vld1_u64((const uint64_t *)d1);
            d5u64 = vld1_u64((const uint64_t *)(d1 + 8));
            d1 += dest_stride;

            q9u16 = vaddw_u8(q0u16, vreinterpret_u8_u64(d2u64));
            q10u16 = vaddw_u8(q0u16, vreinterpret_u8_u64(d3u64));
            q11u16 = vaddw_u8(q0u16, vreinterpret_u8_u64(d4u64));
            q12u16 = vaddw_u8(q0u16, vreinterpret_u8_u64(d5u64));

            d2u8 = vqmovun_s16(vreinterpretq_s16_u16(q9u16));
            d3u8 = vqmovun_s16(vreinterpretq_s16_u16(q10u16));
            d30u8 = vqmovun_s16(vreinterpretq_s16_u16(q11u16));
            d31u8 = vqmovun_s16(vreinterpretq_s16_u16(q12u16));

            vst1_u64((uint64_t *)d2, vreinterpret_u64_u8(d2u8));
            vst1_u64((uint64_t *)(d2 + 8), vreinterpret_u64_u8(d3u8));
            d2 += dest_stride;
            vst1_u64((uint64_t *)d2, vreinterpret_u64_u8(d30u8));
            vst1_u64((uint64_t *)(d2 + 8), vreinterpret_u64_u8(d31u8));
            d2 += dest_stride;
        }
    }
    return;
}
