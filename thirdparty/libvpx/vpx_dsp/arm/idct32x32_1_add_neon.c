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

#include "./vpx_config.h"

#include "vpx_dsp/inv_txfm.h"
#include "vpx_ports/mem.h"

static INLINE void LD_16x8(
        uint8_t *d,
        int d_stride,
        uint8x16_t *q8u8,
        uint8x16_t *q9u8,
        uint8x16_t *q10u8,
        uint8x16_t *q11u8,
        uint8x16_t *q12u8,
        uint8x16_t *q13u8,
        uint8x16_t *q14u8,
        uint8x16_t *q15u8) {
    *q8u8 = vld1q_u8(d);
    d += d_stride;
    *q9u8 = vld1q_u8(d);
    d += d_stride;
    *q10u8 = vld1q_u8(d);
    d += d_stride;
    *q11u8 = vld1q_u8(d);
    d += d_stride;
    *q12u8 = vld1q_u8(d);
    d += d_stride;
    *q13u8 = vld1q_u8(d);
    d += d_stride;
    *q14u8 = vld1q_u8(d);
    d += d_stride;
    *q15u8 = vld1q_u8(d);
    return;
}

static INLINE void ADD_DIFF_16x8(
        uint8x16_t qdiffu8,
        uint8x16_t *q8u8,
        uint8x16_t *q9u8,
        uint8x16_t *q10u8,
        uint8x16_t *q11u8,
        uint8x16_t *q12u8,
        uint8x16_t *q13u8,
        uint8x16_t *q14u8,
        uint8x16_t *q15u8) {
    *q8u8 = vqaddq_u8(*q8u8, qdiffu8);
    *q9u8 = vqaddq_u8(*q9u8, qdiffu8);
    *q10u8 = vqaddq_u8(*q10u8, qdiffu8);
    *q11u8 = vqaddq_u8(*q11u8, qdiffu8);
    *q12u8 = vqaddq_u8(*q12u8, qdiffu8);
    *q13u8 = vqaddq_u8(*q13u8, qdiffu8);
    *q14u8 = vqaddq_u8(*q14u8, qdiffu8);
    *q15u8 = vqaddq_u8(*q15u8, qdiffu8);
    return;
}

static INLINE void SUB_DIFF_16x8(
        uint8x16_t qdiffu8,
        uint8x16_t *q8u8,
        uint8x16_t *q9u8,
        uint8x16_t *q10u8,
        uint8x16_t *q11u8,
        uint8x16_t *q12u8,
        uint8x16_t *q13u8,
        uint8x16_t *q14u8,
        uint8x16_t *q15u8) {
    *q8u8 = vqsubq_u8(*q8u8, qdiffu8);
    *q9u8 = vqsubq_u8(*q9u8, qdiffu8);
    *q10u8 = vqsubq_u8(*q10u8, qdiffu8);
    *q11u8 = vqsubq_u8(*q11u8, qdiffu8);
    *q12u8 = vqsubq_u8(*q12u8, qdiffu8);
    *q13u8 = vqsubq_u8(*q13u8, qdiffu8);
    *q14u8 = vqsubq_u8(*q14u8, qdiffu8);
    *q15u8 = vqsubq_u8(*q15u8, qdiffu8);
    return;
}

static INLINE void ST_16x8(
        uint8_t *d,
        int d_stride,
        uint8x16_t *q8u8,
        uint8x16_t *q9u8,
        uint8x16_t *q10u8,
        uint8x16_t *q11u8,
        uint8x16_t *q12u8,
        uint8x16_t *q13u8,
        uint8x16_t *q14u8,
        uint8x16_t *q15u8) {
    vst1q_u8(d, *q8u8);
    d += d_stride;
    vst1q_u8(d, *q9u8);
    d += d_stride;
    vst1q_u8(d, *q10u8);
    d += d_stride;
    vst1q_u8(d, *q11u8);
    d += d_stride;
    vst1q_u8(d, *q12u8);
    d += d_stride;
    vst1q_u8(d, *q13u8);
    d += d_stride;
    vst1q_u8(d, *q14u8);
    d += d_stride;
    vst1q_u8(d, *q15u8);
    return;
}

void vpx_idct32x32_1_add_neon(
        int16_t *input,
        uint8_t *dest,
        int dest_stride) {
    uint8x16_t q0u8, q8u8, q9u8, q10u8, q11u8, q12u8, q13u8, q14u8, q15u8;
    int i, j, dest_stride8;
    uint8_t *d;
    int16_t a1, cospi_16_64 = 11585;
    int16_t out = dct_const_round_shift(input[0] * cospi_16_64);

    out = dct_const_round_shift(out * cospi_16_64);
    a1 = ROUND_POWER_OF_TWO(out, 6);

    dest_stride8 = dest_stride * 8;
    if (a1 >= 0) {  // diff_positive_32_32
        a1 = a1 < 0 ? 0 : a1 > 255 ? 255 : a1;
        q0u8 = vdupq_n_u8(a1);
        for (i = 0; i < 2; i++, dest += 16) {  // diff_positive_32_32_loop
            d = dest;
            for (j = 0; j < 4; j++) {
                LD_16x8(d, dest_stride, &q8u8, &q9u8, &q10u8, &q11u8,
                                        &q12u8, &q13u8, &q14u8, &q15u8);
                ADD_DIFF_16x8(q0u8, &q8u8, &q9u8, &q10u8, &q11u8,
                                    &q12u8, &q13u8, &q14u8, &q15u8);
                ST_16x8(d, dest_stride, &q8u8, &q9u8, &q10u8, &q11u8,
                                        &q12u8, &q13u8, &q14u8, &q15u8);
                d += dest_stride8;
            }
        }
    } else {  // diff_negative_32_32
        a1 = -a1;
        a1 = a1 < 0 ? 0 : a1 > 255 ? 255 : a1;
        q0u8 = vdupq_n_u8(a1);
        for (i = 0; i < 2; i++, dest += 16) {  // diff_negative_32_32_loop
            d = dest;
            for (j = 0; j < 4; j++) {
                LD_16x8(d, dest_stride, &q8u8, &q9u8, &q10u8, &q11u8,
                                        &q12u8, &q13u8, &q14u8, &q15u8);
                SUB_DIFF_16x8(q0u8, &q8u8, &q9u8, &q10u8, &q11u8,
                                    &q12u8, &q13u8, &q14u8, &q15u8);
                ST_16x8(d, dest_stride, &q8u8, &q9u8, &q10u8, &q11u8,
                                        &q12u8, &q13u8, &q14u8, &q15u8);
                d += dest_stride8;
            }
        }
    }
    return;
}
