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
#include "vpx_dsp/txfm_common.h"

static INLINE void TRANSPOSE8X8(
        int16x8_t *q8s16,
        int16x8_t *q9s16,
        int16x8_t *q10s16,
        int16x8_t *q11s16,
        int16x8_t *q12s16,
        int16x8_t *q13s16,
        int16x8_t *q14s16,
        int16x8_t *q15s16) {
    int16x4_t d16s16, d17s16, d18s16, d19s16, d20s16, d21s16, d22s16, d23s16;
    int16x4_t d24s16, d25s16, d26s16, d27s16, d28s16, d29s16, d30s16, d31s16;
    int32x4x2_t q0x2s32, q1x2s32, q2x2s32, q3x2s32;
    int16x8x2_t q0x2s16, q1x2s16, q2x2s16, q3x2s16;

    d16s16 = vget_low_s16(*q8s16);
    d17s16 = vget_high_s16(*q8s16);
    d18s16 = vget_low_s16(*q9s16);
    d19s16 = vget_high_s16(*q9s16);
    d20s16 = vget_low_s16(*q10s16);
    d21s16 = vget_high_s16(*q10s16);
    d22s16 = vget_low_s16(*q11s16);
    d23s16 = vget_high_s16(*q11s16);
    d24s16 = vget_low_s16(*q12s16);
    d25s16 = vget_high_s16(*q12s16);
    d26s16 = vget_low_s16(*q13s16);
    d27s16 = vget_high_s16(*q13s16);
    d28s16 = vget_low_s16(*q14s16);
    d29s16 = vget_high_s16(*q14s16);
    d30s16 = vget_low_s16(*q15s16);
    d31s16 = vget_high_s16(*q15s16);

    *q8s16  = vcombine_s16(d16s16, d24s16);  // vswp d17, d24
    *q9s16  = vcombine_s16(d18s16, d26s16);  // vswp d19, d26
    *q10s16 = vcombine_s16(d20s16, d28s16);  // vswp d21, d28
    *q11s16 = vcombine_s16(d22s16, d30s16);  // vswp d23, d30
    *q12s16 = vcombine_s16(d17s16, d25s16);
    *q13s16 = vcombine_s16(d19s16, d27s16);
    *q14s16 = vcombine_s16(d21s16, d29s16);
    *q15s16 = vcombine_s16(d23s16, d31s16);

    q0x2s32 = vtrnq_s32(vreinterpretq_s32_s16(*q8s16),
                        vreinterpretq_s32_s16(*q10s16));
    q1x2s32 = vtrnq_s32(vreinterpretq_s32_s16(*q9s16),
                        vreinterpretq_s32_s16(*q11s16));
    q2x2s32 = vtrnq_s32(vreinterpretq_s32_s16(*q12s16),
                        vreinterpretq_s32_s16(*q14s16));
    q3x2s32 = vtrnq_s32(vreinterpretq_s32_s16(*q13s16),
                        vreinterpretq_s32_s16(*q15s16));

    q0x2s16 = vtrnq_s16(vreinterpretq_s16_s32(q0x2s32.val[0]),   // q8
                        vreinterpretq_s16_s32(q1x2s32.val[0]));  // q9
    q1x2s16 = vtrnq_s16(vreinterpretq_s16_s32(q0x2s32.val[1]),   // q10
                        vreinterpretq_s16_s32(q1x2s32.val[1]));  // q11
    q2x2s16 = vtrnq_s16(vreinterpretq_s16_s32(q2x2s32.val[0]),   // q12
                        vreinterpretq_s16_s32(q3x2s32.val[0]));  // q13
    q3x2s16 = vtrnq_s16(vreinterpretq_s16_s32(q2x2s32.val[1]),   // q14
                        vreinterpretq_s16_s32(q3x2s32.val[1]));  // q15

    *q8s16  = q0x2s16.val[0];
    *q9s16  = q0x2s16.val[1];
    *q10s16 = q1x2s16.val[0];
    *q11s16 = q1x2s16.val[1];
    *q12s16 = q2x2s16.val[0];
    *q13s16 = q2x2s16.val[1];
    *q14s16 = q3x2s16.val[0];
    *q15s16 = q3x2s16.val[1];
    return;
}

static INLINE void IDCT8x8_1D(
        int16x8_t *q8s16,
        int16x8_t *q9s16,
        int16x8_t *q10s16,
        int16x8_t *q11s16,
        int16x8_t *q12s16,
        int16x8_t *q13s16,
        int16x8_t *q14s16,
        int16x8_t *q15s16) {
    int16x4_t d0s16, d1s16, d2s16, d3s16;
    int16x4_t d8s16, d9s16, d10s16, d11s16, d12s16, d13s16, d14s16, d15s16;
    int16x4_t d16s16, d17s16, d18s16, d19s16, d20s16, d21s16, d22s16, d23s16;
    int16x4_t d24s16, d25s16, d26s16, d27s16, d28s16, d29s16, d30s16, d31s16;
    int16x8_t q0s16, q1s16, q2s16, q3s16, q4s16, q5s16, q6s16, q7s16;
    int32x4_t q2s32, q3s32, q5s32, q6s32, q8s32, q9s32;
    int32x4_t q10s32, q11s32, q12s32, q13s32, q15s32;

    d0s16 = vdup_n_s16(cospi_28_64);
    d1s16 = vdup_n_s16(cospi_4_64);
    d2s16 = vdup_n_s16(cospi_12_64);
    d3s16 = vdup_n_s16(cospi_20_64);

    d16s16 = vget_low_s16(*q8s16);
    d17s16 = vget_high_s16(*q8s16);
    d18s16 = vget_low_s16(*q9s16);
    d19s16 = vget_high_s16(*q9s16);
    d20s16 = vget_low_s16(*q10s16);
    d21s16 = vget_high_s16(*q10s16);
    d22s16 = vget_low_s16(*q11s16);
    d23s16 = vget_high_s16(*q11s16);
    d24s16 = vget_low_s16(*q12s16);
    d25s16 = vget_high_s16(*q12s16);
    d26s16 = vget_low_s16(*q13s16);
    d27s16 = vget_high_s16(*q13s16);
    d28s16 = vget_low_s16(*q14s16);
    d29s16 = vget_high_s16(*q14s16);
    d30s16 = vget_low_s16(*q15s16);
    d31s16 = vget_high_s16(*q15s16);

    q2s32 = vmull_s16(d18s16, d0s16);
    q3s32 = vmull_s16(d19s16, d0s16);
    q5s32 = vmull_s16(d26s16, d2s16);
    q6s32 = vmull_s16(d27s16, d2s16);

    q2s32 = vmlsl_s16(q2s32, d30s16, d1s16);
    q3s32 = vmlsl_s16(q3s32, d31s16, d1s16);
    q5s32 = vmlsl_s16(q5s32, d22s16, d3s16);
    q6s32 = vmlsl_s16(q6s32, d23s16, d3s16);

    d8s16 = vqrshrn_n_s32(q2s32, 14);
    d9s16 = vqrshrn_n_s32(q3s32, 14);
    d10s16 = vqrshrn_n_s32(q5s32, 14);
    d11s16 = vqrshrn_n_s32(q6s32, 14);
    q4s16 = vcombine_s16(d8s16, d9s16);
    q5s16 = vcombine_s16(d10s16, d11s16);

    q2s32 = vmull_s16(d18s16, d1s16);
    q3s32 = vmull_s16(d19s16, d1s16);
    q9s32 = vmull_s16(d26s16, d3s16);
    q13s32 = vmull_s16(d27s16, d3s16);

    q2s32 = vmlal_s16(q2s32, d30s16, d0s16);
    q3s32 = vmlal_s16(q3s32, d31s16, d0s16);
    q9s32 = vmlal_s16(q9s32, d22s16, d2s16);
    q13s32 = vmlal_s16(q13s32, d23s16, d2s16);

    d14s16 = vqrshrn_n_s32(q2s32, 14);
    d15s16 = vqrshrn_n_s32(q3s32, 14);
    d12s16 = vqrshrn_n_s32(q9s32, 14);
    d13s16 = vqrshrn_n_s32(q13s32, 14);
    q6s16 = vcombine_s16(d12s16, d13s16);
    q7s16 = vcombine_s16(d14s16, d15s16);

    d0s16 = vdup_n_s16(cospi_16_64);

    q2s32 = vmull_s16(d16s16, d0s16);
    q3s32 = vmull_s16(d17s16, d0s16);
    q13s32 = vmull_s16(d16s16, d0s16);
    q15s32 = vmull_s16(d17s16, d0s16);

    q2s32 = vmlal_s16(q2s32, d24s16, d0s16);
    q3s32 = vmlal_s16(q3s32, d25s16, d0s16);
    q13s32 = vmlsl_s16(q13s32, d24s16, d0s16);
    q15s32 = vmlsl_s16(q15s32, d25s16, d0s16);

    d0s16 = vdup_n_s16(cospi_24_64);
    d1s16 = vdup_n_s16(cospi_8_64);

    d18s16 = vqrshrn_n_s32(q2s32, 14);
    d19s16 = vqrshrn_n_s32(q3s32, 14);
    d22s16 = vqrshrn_n_s32(q13s32, 14);
    d23s16 = vqrshrn_n_s32(q15s32, 14);
    *q9s16 = vcombine_s16(d18s16, d19s16);
    *q11s16 = vcombine_s16(d22s16, d23s16);

    q2s32 = vmull_s16(d20s16, d0s16);
    q3s32 = vmull_s16(d21s16, d0s16);
    q8s32 = vmull_s16(d20s16, d1s16);
    q12s32 = vmull_s16(d21s16, d1s16);

    q2s32 = vmlsl_s16(q2s32, d28s16, d1s16);
    q3s32 = vmlsl_s16(q3s32, d29s16, d1s16);
    q8s32 = vmlal_s16(q8s32, d28s16, d0s16);
    q12s32 = vmlal_s16(q12s32, d29s16, d0s16);

    d26s16 = vqrshrn_n_s32(q2s32, 14);
    d27s16 = vqrshrn_n_s32(q3s32, 14);
    d30s16 = vqrshrn_n_s32(q8s32, 14);
    d31s16 = vqrshrn_n_s32(q12s32, 14);
    *q13s16 = vcombine_s16(d26s16, d27s16);
    *q15s16 = vcombine_s16(d30s16, d31s16);

    q0s16 = vaddq_s16(*q9s16, *q15s16);
    q1s16 = vaddq_s16(*q11s16, *q13s16);
    q2s16 = vsubq_s16(*q11s16, *q13s16);
    q3s16 = vsubq_s16(*q9s16, *q15s16);

    *q13s16 = vsubq_s16(q4s16, q5s16);
    q4s16 = vaddq_s16(q4s16, q5s16);
    *q14s16 = vsubq_s16(q7s16, q6s16);
    q7s16 = vaddq_s16(q7s16, q6s16);
    d26s16 = vget_low_s16(*q13s16);
    d27s16 = vget_high_s16(*q13s16);
    d28s16 = vget_low_s16(*q14s16);
    d29s16 = vget_high_s16(*q14s16);

    d16s16 = vdup_n_s16(cospi_16_64);

    q9s32 = vmull_s16(d28s16, d16s16);
    q10s32 = vmull_s16(d29s16, d16s16);
    q11s32 = vmull_s16(d28s16, d16s16);
    q12s32 = vmull_s16(d29s16, d16s16);

    q9s32 = vmlsl_s16(q9s32,  d26s16, d16s16);
    q10s32 = vmlsl_s16(q10s32, d27s16, d16s16);
    q11s32 = vmlal_s16(q11s32, d26s16, d16s16);
    q12s32 = vmlal_s16(q12s32, d27s16, d16s16);

    d10s16 = vqrshrn_n_s32(q9s32, 14);
    d11s16 = vqrshrn_n_s32(q10s32, 14);
    d12s16 = vqrshrn_n_s32(q11s32, 14);
    d13s16 = vqrshrn_n_s32(q12s32, 14);
    q5s16 = vcombine_s16(d10s16, d11s16);
    q6s16 = vcombine_s16(d12s16, d13s16);

    *q8s16 = vaddq_s16(q0s16, q7s16);
    *q9s16 = vaddq_s16(q1s16, q6s16);
    *q10s16 = vaddq_s16(q2s16, q5s16);
    *q11s16 = vaddq_s16(q3s16, q4s16);
    *q12s16 = vsubq_s16(q3s16, q4s16);
    *q13s16 = vsubq_s16(q2s16, q5s16);
    *q14s16 = vsubq_s16(q1s16, q6s16);
    *q15s16 = vsubq_s16(q0s16, q7s16);
    return;
}

void vpx_idct8x8_64_add_neon(
        int16_t *input,
        uint8_t *dest,
        int dest_stride) {
    uint8_t *d1, *d2;
    uint8x8_t d0u8, d1u8, d2u8, d3u8;
    uint64x1_t d0u64, d1u64, d2u64, d3u64;
    int16x8_t q8s16, q9s16, q10s16, q11s16, q12s16, q13s16, q14s16, q15s16;
    uint16x8_t q8u16, q9u16, q10u16, q11u16;

    q8s16 = vld1q_s16(input);
    q9s16 = vld1q_s16(input + 8);
    q10s16 = vld1q_s16(input + 16);
    q11s16 = vld1q_s16(input + 24);
    q12s16 = vld1q_s16(input + 32);
    q13s16 = vld1q_s16(input + 40);
    q14s16 = vld1q_s16(input + 48);
    q15s16 = vld1q_s16(input + 56);

    TRANSPOSE8X8(&q8s16, &q9s16, &q10s16, &q11s16,
                 &q12s16, &q13s16, &q14s16, &q15s16);

    IDCT8x8_1D(&q8s16, &q9s16, &q10s16, &q11s16,
               &q12s16, &q13s16, &q14s16, &q15s16);

    TRANSPOSE8X8(&q8s16, &q9s16, &q10s16, &q11s16,
                 &q12s16, &q13s16, &q14s16, &q15s16);

    IDCT8x8_1D(&q8s16, &q9s16, &q10s16, &q11s16,
               &q12s16, &q13s16, &q14s16, &q15s16);

    q8s16 = vrshrq_n_s16(q8s16, 5);
    q9s16 = vrshrq_n_s16(q9s16, 5);
    q10s16 = vrshrq_n_s16(q10s16, 5);
    q11s16 = vrshrq_n_s16(q11s16, 5);
    q12s16 = vrshrq_n_s16(q12s16, 5);
    q13s16 = vrshrq_n_s16(q13s16, 5);
    q14s16 = vrshrq_n_s16(q14s16, 5);
    q15s16 = vrshrq_n_s16(q15s16, 5);

    d1 = d2 = dest;

    d0u64 = vld1_u64((uint64_t *)d1);
    d1 += dest_stride;
    d1u64 = vld1_u64((uint64_t *)d1);
    d1 += dest_stride;
    d2u64 = vld1_u64((uint64_t *)d1);
    d1 += dest_stride;
    d3u64 = vld1_u64((uint64_t *)d1);
    d1 += dest_stride;

    q8u16 = vaddw_u8(vreinterpretq_u16_s16(q8s16),
                     vreinterpret_u8_u64(d0u64));
    q9u16 = vaddw_u8(vreinterpretq_u16_s16(q9s16),
                     vreinterpret_u8_u64(d1u64));
    q10u16 = vaddw_u8(vreinterpretq_u16_s16(q10s16),
                      vreinterpret_u8_u64(d2u64));
    q11u16 = vaddw_u8(vreinterpretq_u16_s16(q11s16),
                      vreinterpret_u8_u64(d3u64));

    d0u8 = vqmovun_s16(vreinterpretq_s16_u16(q8u16));
    d1u8 = vqmovun_s16(vreinterpretq_s16_u16(q9u16));
    d2u8 = vqmovun_s16(vreinterpretq_s16_u16(q10u16));
    d3u8 = vqmovun_s16(vreinterpretq_s16_u16(q11u16));

    vst1_u64((uint64_t *)d2, vreinterpret_u64_u8(d0u8));
    d2 += dest_stride;
    vst1_u64((uint64_t *)d2, vreinterpret_u64_u8(d1u8));
    d2 += dest_stride;
    vst1_u64((uint64_t *)d2, vreinterpret_u64_u8(d2u8));
    d2 += dest_stride;
    vst1_u64((uint64_t *)d2, vreinterpret_u64_u8(d3u8));
    d2 += dest_stride;

    q8s16 = q12s16;
    q9s16 = q13s16;
    q10s16 = q14s16;
    q11s16 = q15s16;

    d0u64 = vld1_u64((uint64_t *)d1);
    d1 += dest_stride;
    d1u64 = vld1_u64((uint64_t *)d1);
    d1 += dest_stride;
    d2u64 = vld1_u64((uint64_t *)d1);
    d1 += dest_stride;
    d3u64 = vld1_u64((uint64_t *)d1);
    d1 += dest_stride;

    q8u16 = vaddw_u8(vreinterpretq_u16_s16(q8s16),
                     vreinterpret_u8_u64(d0u64));
    q9u16 = vaddw_u8(vreinterpretq_u16_s16(q9s16),
                     vreinterpret_u8_u64(d1u64));
    q10u16 = vaddw_u8(vreinterpretq_u16_s16(q10s16),
                      vreinterpret_u8_u64(d2u64));
    q11u16 = vaddw_u8(vreinterpretq_u16_s16(q11s16),
                      vreinterpret_u8_u64(d3u64));

    d0u8 = vqmovun_s16(vreinterpretq_s16_u16(q8u16));
    d1u8 = vqmovun_s16(vreinterpretq_s16_u16(q9u16));
    d2u8 = vqmovun_s16(vreinterpretq_s16_u16(q10u16));
    d3u8 = vqmovun_s16(vreinterpretq_s16_u16(q11u16));

    vst1_u64((uint64_t *)d2, vreinterpret_u64_u8(d0u8));
    d2 += dest_stride;
    vst1_u64((uint64_t *)d2, vreinterpret_u64_u8(d1u8));
    d2 += dest_stride;
    vst1_u64((uint64_t *)d2, vreinterpret_u64_u8(d2u8));
    d2 += dest_stride;
    vst1_u64((uint64_t *)d2, vreinterpret_u64_u8(d3u8));
    d2 += dest_stride;
    return;
}

void vpx_idct8x8_12_add_neon(
        int16_t *input,
        uint8_t *dest,
        int dest_stride) {
    uint8_t *d1, *d2;
    uint8x8_t d0u8, d1u8, d2u8, d3u8;
    int16x4_t d10s16, d11s16, d12s16, d13s16, d16s16;
    int16x4_t d26s16, d27s16, d28s16, d29s16;
    uint64x1_t d0u64, d1u64, d2u64, d3u64;
    int16x8_t q0s16, q1s16, q2s16, q3s16, q4s16, q5s16, q6s16, q7s16;
    int16x8_t q8s16, q9s16, q10s16, q11s16, q12s16, q13s16, q14s16, q15s16;
    uint16x8_t q8u16, q9u16, q10u16, q11u16;
    int32x4_t q9s32, q10s32, q11s32, q12s32;

    q8s16 = vld1q_s16(input);
    q9s16 = vld1q_s16(input + 8);
    q10s16 = vld1q_s16(input + 16);
    q11s16 = vld1q_s16(input + 24);
    q12s16 = vld1q_s16(input + 32);
    q13s16 = vld1q_s16(input + 40);
    q14s16 = vld1q_s16(input + 48);
    q15s16 = vld1q_s16(input + 56);

    TRANSPOSE8X8(&q8s16, &q9s16, &q10s16, &q11s16,
                 &q12s16, &q13s16, &q14s16, &q15s16);

    // First transform rows
    // stage 1
    q0s16 = vdupq_n_s16(cospi_28_64 * 2);
    q1s16 = vdupq_n_s16(cospi_4_64 * 2);

    q4s16 = vqrdmulhq_s16(q9s16, q0s16);

    q0s16 = vdupq_n_s16(-cospi_20_64 * 2);

    q7s16 = vqrdmulhq_s16(q9s16, q1s16);

    q1s16 = vdupq_n_s16(cospi_12_64 * 2);

    q5s16 = vqrdmulhq_s16(q11s16, q0s16);

    q0s16 = vdupq_n_s16(cospi_16_64 * 2);

    q6s16 = vqrdmulhq_s16(q11s16, q1s16);

    // stage 2 & stage 3 - even half
    q1s16 = vdupq_n_s16(cospi_24_64 * 2);

    q9s16 = vqrdmulhq_s16(q8s16, q0s16);

    q0s16 = vdupq_n_s16(cospi_8_64 * 2);

    q13s16 = vqrdmulhq_s16(q10s16, q1s16);

    q15s16 = vqrdmulhq_s16(q10s16, q0s16);

    // stage 3 -odd half
    q0s16 = vaddq_s16(q9s16, q15s16);
    q1s16 = vaddq_s16(q9s16, q13s16);
    q2s16 = vsubq_s16(q9s16, q13s16);
    q3s16 = vsubq_s16(q9s16, q15s16);

    // stage 2 - odd half
    q13s16 = vsubq_s16(q4s16, q5s16);
    q4s16 = vaddq_s16(q4s16, q5s16);
    q14s16 = vsubq_s16(q7s16, q6s16);
    q7s16 = vaddq_s16(q7s16, q6s16);
    d26s16 = vget_low_s16(q13s16);
    d27s16 = vget_high_s16(q13s16);
    d28s16 = vget_low_s16(q14s16);
    d29s16 = vget_high_s16(q14s16);

    d16s16 = vdup_n_s16(cospi_16_64);
    q9s32 = vmull_s16(d28s16, d16s16);
    q10s32 = vmull_s16(d29s16, d16s16);
    q11s32 = vmull_s16(d28s16, d16s16);
    q12s32 = vmull_s16(d29s16, d16s16);

    q9s32 = vmlsl_s16(q9s32,  d26s16, d16s16);
    q10s32 = vmlsl_s16(q10s32, d27s16, d16s16);
    q11s32 = vmlal_s16(q11s32, d26s16, d16s16);
    q12s32 = vmlal_s16(q12s32, d27s16, d16s16);

    d10s16 = vqrshrn_n_s32(q9s32, 14);
    d11s16 = vqrshrn_n_s32(q10s32, 14);
    d12s16 = vqrshrn_n_s32(q11s32, 14);
    d13s16 = vqrshrn_n_s32(q12s32, 14);
    q5s16 = vcombine_s16(d10s16, d11s16);
    q6s16 = vcombine_s16(d12s16, d13s16);

    // stage 4
    q8s16 = vaddq_s16(q0s16, q7s16);
    q9s16 = vaddq_s16(q1s16, q6s16);
    q10s16 = vaddq_s16(q2s16, q5s16);
    q11s16 = vaddq_s16(q3s16, q4s16);
    q12s16 = vsubq_s16(q3s16, q4s16);
    q13s16 = vsubq_s16(q2s16, q5s16);
    q14s16 = vsubq_s16(q1s16, q6s16);
    q15s16 = vsubq_s16(q0s16, q7s16);

    TRANSPOSE8X8(&q8s16, &q9s16, &q10s16, &q11s16,
                 &q12s16, &q13s16, &q14s16, &q15s16);

    IDCT8x8_1D(&q8s16, &q9s16, &q10s16, &q11s16,
               &q12s16, &q13s16, &q14s16, &q15s16);

    q8s16 = vrshrq_n_s16(q8s16, 5);
    q9s16 = vrshrq_n_s16(q9s16, 5);
    q10s16 = vrshrq_n_s16(q10s16, 5);
    q11s16 = vrshrq_n_s16(q11s16, 5);
    q12s16 = vrshrq_n_s16(q12s16, 5);
    q13s16 = vrshrq_n_s16(q13s16, 5);
    q14s16 = vrshrq_n_s16(q14s16, 5);
    q15s16 = vrshrq_n_s16(q15s16, 5);

    d1 = d2 = dest;

    d0u64 = vld1_u64((uint64_t *)d1);
    d1 += dest_stride;
    d1u64 = vld1_u64((uint64_t *)d1);
    d1 += dest_stride;
    d2u64 = vld1_u64((uint64_t *)d1);
    d1 += dest_stride;
    d3u64 = vld1_u64((uint64_t *)d1);
    d1 += dest_stride;

    q8u16 = vaddw_u8(vreinterpretq_u16_s16(q8s16),
                     vreinterpret_u8_u64(d0u64));
    q9u16 = vaddw_u8(vreinterpretq_u16_s16(q9s16),
                     vreinterpret_u8_u64(d1u64));
    q10u16 = vaddw_u8(vreinterpretq_u16_s16(q10s16),
                      vreinterpret_u8_u64(d2u64));
    q11u16 = vaddw_u8(vreinterpretq_u16_s16(q11s16),
                      vreinterpret_u8_u64(d3u64));

    d0u8 = vqmovun_s16(vreinterpretq_s16_u16(q8u16));
    d1u8 = vqmovun_s16(vreinterpretq_s16_u16(q9u16));
    d2u8 = vqmovun_s16(vreinterpretq_s16_u16(q10u16));
    d3u8 = vqmovun_s16(vreinterpretq_s16_u16(q11u16));

    vst1_u64((uint64_t *)d2, vreinterpret_u64_u8(d0u8));
    d2 += dest_stride;
    vst1_u64((uint64_t *)d2, vreinterpret_u64_u8(d1u8));
    d2 += dest_stride;
    vst1_u64((uint64_t *)d2, vreinterpret_u64_u8(d2u8));
    d2 += dest_stride;
    vst1_u64((uint64_t *)d2, vreinterpret_u64_u8(d3u8));
    d2 += dest_stride;

    q8s16 = q12s16;
    q9s16 = q13s16;
    q10s16 = q14s16;
    q11s16 = q15s16;

    d0u64 = vld1_u64((uint64_t *)d1);
    d1 += dest_stride;
    d1u64 = vld1_u64((uint64_t *)d1);
    d1 += dest_stride;
    d2u64 = vld1_u64((uint64_t *)d1);
    d1 += dest_stride;
    d3u64 = vld1_u64((uint64_t *)d1);
    d1 += dest_stride;

    q8u16 = vaddw_u8(vreinterpretq_u16_s16(q8s16),
                     vreinterpret_u8_u64(d0u64));
    q9u16 = vaddw_u8(vreinterpretq_u16_s16(q9s16),
                     vreinterpret_u8_u64(d1u64));
    q10u16 = vaddw_u8(vreinterpretq_u16_s16(q10s16),
                      vreinterpret_u8_u64(d2u64));
    q11u16 = vaddw_u8(vreinterpretq_u16_s16(q11s16),
                      vreinterpret_u8_u64(d3u64));

    d0u8 = vqmovun_s16(vreinterpretq_s16_u16(q8u16));
    d1u8 = vqmovun_s16(vreinterpretq_s16_u16(q9u16));
    d2u8 = vqmovun_s16(vreinterpretq_s16_u16(q10u16));
    d3u8 = vqmovun_s16(vreinterpretq_s16_u16(q11u16));

    vst1_u64((uint64_t *)d2, vreinterpret_u64_u8(d0u8));
    d2 += dest_stride;
    vst1_u64((uint64_t *)d2, vreinterpret_u64_u8(d1u8));
    d2 += dest_stride;
    vst1_u64((uint64_t *)d2, vreinterpret_u64_u8(d2u8));
    d2 += dest_stride;
    vst1_u64((uint64_t *)d2, vreinterpret_u64_u8(d3u8));
    d2 += dest_stride;
    return;
}
