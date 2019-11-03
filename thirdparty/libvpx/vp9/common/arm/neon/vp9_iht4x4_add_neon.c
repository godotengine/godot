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

#include "./vp9_rtcd.h"
#include "./vpx_config.h"
#include "vp9/common/vp9_common.h"

static int16_t sinpi_1_9 = 0x14a3;
static int16_t sinpi_2_9 = 0x26c9;
static int16_t sinpi_3_9 = 0x3441;
static int16_t sinpi_4_9 = 0x3b6c;
static int16_t cospi_8_64 = 0x3b21;
static int16_t cospi_16_64 = 0x2d41;
static int16_t cospi_24_64 = 0x187e;

static INLINE void TRANSPOSE4X4(
        int16x8_t *q8s16,
        int16x8_t *q9s16) {
    int32x4_t q8s32, q9s32;
    int16x4x2_t d0x2s16, d1x2s16;
    int32x4x2_t q0x2s32;

    d0x2s16 = vtrn_s16(vget_low_s16(*q8s16), vget_high_s16(*q8s16));
    d1x2s16 = vtrn_s16(vget_low_s16(*q9s16), vget_high_s16(*q9s16));

    q8s32 = vreinterpretq_s32_s16(vcombine_s16(d0x2s16.val[0], d0x2s16.val[1]));
    q9s32 = vreinterpretq_s32_s16(vcombine_s16(d1x2s16.val[0], d1x2s16.val[1]));
    q0x2s32 = vtrnq_s32(q8s32, q9s32);

    *q8s16 = vreinterpretq_s16_s32(q0x2s32.val[0]);
    *q9s16 = vreinterpretq_s16_s32(q0x2s32.val[1]);
    return;
}

static INLINE void GENERATE_COSINE_CONSTANTS(
        int16x4_t *d0s16,
        int16x4_t *d1s16,
        int16x4_t *d2s16) {
    *d0s16 = vdup_n_s16(cospi_8_64);
    *d1s16 = vdup_n_s16(cospi_16_64);
    *d2s16 = vdup_n_s16(cospi_24_64);
    return;
}

static INLINE void GENERATE_SINE_CONSTANTS(
        int16x4_t *d3s16,
        int16x4_t *d4s16,
        int16x4_t *d5s16,
        int16x8_t *q3s16) {
    *d3s16 = vdup_n_s16(sinpi_1_9);
    *d4s16 = vdup_n_s16(sinpi_2_9);
    *q3s16 = vdupq_n_s16(sinpi_3_9);
    *d5s16 = vdup_n_s16(sinpi_4_9);
    return;
}

static INLINE void IDCT4x4_1D(
        int16x4_t *d0s16,
        int16x4_t *d1s16,
        int16x4_t *d2s16,
        int16x8_t *q8s16,
        int16x8_t *q9s16) {
    int16x4_t d16s16, d17s16, d18s16, d19s16, d23s16, d24s16;
    int16x4_t d26s16, d27s16, d28s16, d29s16;
    int32x4_t q10s32, q13s32, q14s32, q15s32;
    int16x8_t q13s16, q14s16;

    d16s16 = vget_low_s16(*q8s16);
    d17s16 = vget_high_s16(*q8s16);
    d18s16 = vget_low_s16(*q9s16);
    d19s16 = vget_high_s16(*q9s16);

    d23s16 = vadd_s16(d16s16, d18s16);
    d24s16 = vsub_s16(d16s16, d18s16);

    q15s32 = vmull_s16(d17s16, *d2s16);
    q10s32 = vmull_s16(d17s16, *d0s16);
    q13s32 = vmull_s16(d23s16, *d1s16);
    q14s32 = vmull_s16(d24s16, *d1s16);
    q15s32 = vmlsl_s16(q15s32, d19s16, *d0s16);
    q10s32 = vmlal_s16(q10s32, d19s16, *d2s16);

    d26s16 = vqrshrn_n_s32(q13s32, 14);
    d27s16 = vqrshrn_n_s32(q14s32, 14);
    d29s16 = vqrshrn_n_s32(q15s32, 14);
    d28s16 = vqrshrn_n_s32(q10s32, 14);

    q13s16 = vcombine_s16(d26s16, d27s16);
    q14s16 = vcombine_s16(d28s16, d29s16);
    *q8s16 = vaddq_s16(q13s16, q14s16);
    *q9s16 = vsubq_s16(q13s16, q14s16);
    *q9s16 = vcombine_s16(vget_high_s16(*q9s16),
                          vget_low_s16(*q9s16));  // vswp
    return;
}

static INLINE void IADST4x4_1D(
        int16x4_t *d3s16,
        int16x4_t *d4s16,
        int16x4_t *d5s16,
        int16x8_t *q3s16,
        int16x8_t *q8s16,
        int16x8_t *q9s16) {
    int16x4_t d6s16, d16s16, d17s16, d18s16, d19s16;
    int32x4_t q8s32, q9s32, q10s32, q11s32, q12s32, q13s32, q14s32, q15s32;

    d6s16 = vget_low_s16(*q3s16);

    d16s16 = vget_low_s16(*q8s16);
    d17s16 = vget_high_s16(*q8s16);
    d18s16 = vget_low_s16(*q9s16);
    d19s16 = vget_high_s16(*q9s16);

    q10s32 = vmull_s16(*d3s16, d16s16);
    q11s32 = vmull_s16(*d4s16, d16s16);
    q12s32 = vmull_s16(d6s16, d17s16);
    q13s32 = vmull_s16(*d5s16, d18s16);
    q14s32 = vmull_s16(*d3s16, d18s16);
    q15s32 = vmovl_s16(d16s16);
    q15s32 = vaddw_s16(q15s32, d19s16);
    q8s32  = vmull_s16(*d4s16, d19s16);
    q15s32 = vsubw_s16(q15s32, d18s16);
    q9s32  = vmull_s16(*d5s16, d19s16);

    q10s32 = vaddq_s32(q10s32, q13s32);
    q10s32 = vaddq_s32(q10s32, q8s32);
    q11s32 = vsubq_s32(q11s32, q14s32);
    q8s32  = vdupq_n_s32(sinpi_3_9);
    q11s32 = vsubq_s32(q11s32, q9s32);
    q15s32 = vmulq_s32(q15s32, q8s32);

    q13s32 = vaddq_s32(q10s32, q12s32);
    q10s32 = vaddq_s32(q10s32, q11s32);
    q14s32 = vaddq_s32(q11s32, q12s32);
    q10s32 = vsubq_s32(q10s32, q12s32);

    d16s16 = vqrshrn_n_s32(q13s32, 14);
    d17s16 = vqrshrn_n_s32(q14s32, 14);
    d18s16 = vqrshrn_n_s32(q15s32, 14);
    d19s16 = vqrshrn_n_s32(q10s32, 14);

    *q8s16 = vcombine_s16(d16s16, d17s16);
    *q9s16 = vcombine_s16(d18s16, d19s16);
    return;
}

void vp9_iht4x4_16_add_neon(const tran_low_t *input, uint8_t *dest,
                            int dest_stride, int tx_type) {
    uint8x8_t d26u8, d27u8;
    int16x4_t d0s16, d1s16, d2s16, d3s16, d4s16, d5s16;
    uint32x2_t d26u32, d27u32;
    int16x8_t q3s16, q8s16, q9s16;
    uint16x8_t q8u16, q9u16;

    d26u32 = d27u32 = vdup_n_u32(0);

    q8s16 = vld1q_s16(input);
    q9s16 = vld1q_s16(input + 8);

    TRANSPOSE4X4(&q8s16, &q9s16);

    switch (tx_type) {
      case 0:  // idct_idct is not supported. Fall back to C
        vp9_iht4x4_16_add_c(input, dest, dest_stride, tx_type);
        return;
        break;
      case 1:  // iadst_idct
        // generate constants
        GENERATE_COSINE_CONSTANTS(&d0s16, &d1s16, &d2s16);
        GENERATE_SINE_CONSTANTS(&d3s16, &d4s16, &d5s16, &q3s16);

        // first transform rows
        IDCT4x4_1D(&d0s16, &d1s16, &d2s16, &q8s16, &q9s16);

        // transpose the matrix
        TRANSPOSE4X4(&q8s16, &q9s16);

        // then transform columns
        IADST4x4_1D(&d3s16, &d4s16, &d5s16, &q3s16, &q8s16, &q9s16);
        break;
      case 2:  // idct_iadst
        // generate constantsyy
        GENERATE_COSINE_CONSTANTS(&d0s16, &d1s16, &d2s16);
        GENERATE_SINE_CONSTANTS(&d3s16, &d4s16, &d5s16, &q3s16);

        // first transform rows
        IADST4x4_1D(&d3s16, &d4s16, &d5s16, &q3s16, &q8s16, &q9s16);

        // transpose the matrix
        TRANSPOSE4X4(&q8s16, &q9s16);

        // then transform columns
        IDCT4x4_1D(&d0s16, &d1s16, &d2s16, &q8s16, &q9s16);
        break;
      case 3:  // iadst_iadst
        // generate constants
        GENERATE_SINE_CONSTANTS(&d3s16, &d4s16, &d5s16, &q3s16);

        // first transform rows
        IADST4x4_1D(&d3s16, &d4s16, &d5s16, &q3s16, &q8s16, &q9s16);

        // transpose the matrix
        TRANSPOSE4X4(&q8s16, &q9s16);

        // then transform columns
        IADST4x4_1D(&d3s16, &d4s16, &d5s16, &q3s16, &q8s16, &q9s16);
        break;
      default:  // iadst_idct
        assert(0);
        break;
    }

    q8s16 = vrshrq_n_s16(q8s16, 4);
    q9s16 = vrshrq_n_s16(q9s16, 4);

    d26u32 = vld1_lane_u32((const uint32_t *)dest, d26u32, 0);
    dest += dest_stride;
    d26u32 = vld1_lane_u32((const uint32_t *)dest, d26u32, 1);
    dest += dest_stride;
    d27u32 = vld1_lane_u32((const uint32_t *)dest, d27u32, 0);
    dest += dest_stride;
    d27u32 = vld1_lane_u32((const uint32_t *)dest, d27u32, 1);

    q8u16 = vaddw_u8(vreinterpretq_u16_s16(q8s16), vreinterpret_u8_u32(d26u32));
    q9u16 = vaddw_u8(vreinterpretq_u16_s16(q9s16), vreinterpret_u8_u32(d27u32));

    d26u8 = vqmovun_s16(vreinterpretq_s16_u16(q8u16));
    d27u8 = vqmovun_s16(vreinterpretq_s16_u16(q9u16));

    vst1_lane_u32((uint32_t *)dest, vreinterpret_u32_u8(d27u8), 1);
    dest -= dest_stride;
    vst1_lane_u32((uint32_t *)dest, vreinterpret_u32_u8(d27u8), 0);
    dest -= dest_stride;
    vst1_lane_u32((uint32_t *)dest, vreinterpret_u32_u8(d26u8), 1);
    dest -= dest_stride;
    vst1_lane_u32((uint32_t *)dest, vreinterpret_u32_u8(d26u8), 0);
    return;
}
