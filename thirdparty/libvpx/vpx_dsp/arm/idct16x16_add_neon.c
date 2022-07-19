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

void vpx_idct16x16_256_add_neon_pass1(
        int16_t *in,
        int16_t *out,
        int output_stride) {
    int16x4_t d0s16, d1s16, d2s16, d3s16;
    int16x4_t d8s16, d9s16, d10s16, d11s16, d12s16, d13s16, d14s16, d15s16;
    int16x4_t d16s16, d17s16, d18s16, d19s16, d20s16, d21s16, d22s16, d23s16;
    int16x4_t d24s16, d25s16, d26s16, d27s16, d28s16, d29s16, d30s16, d31s16;
    uint64x1_t d16u64, d17u64, d18u64, d19u64, d20u64, d21u64, d22u64, d23u64;
    uint64x1_t d24u64, d25u64, d26u64, d27u64, d28u64, d29u64, d30u64, d31u64;
    int16x8_t q0s16, q1s16, q2s16, q3s16, q4s16, q5s16, q6s16, q7s16;
    int16x8_t q8s16, q9s16, q10s16, q11s16, q12s16, q13s16, q14s16, q15s16;
    int32x4_t q0s32, q1s32, q2s32, q3s32, q5s32, q6s32, q9s32;
    int32x4_t q10s32, q11s32, q12s32, q13s32, q15s32;
    int16x8x2_t q0x2s16;

    q0x2s16 = vld2q_s16(in);
    q8s16  = q0x2s16.val[0];
    in += 16;
    q0x2s16 = vld2q_s16(in);
    q9s16  = q0x2s16.val[0];
    in += 16;
    q0x2s16 = vld2q_s16(in);
    q10s16 = q0x2s16.val[0];
    in += 16;
    q0x2s16 = vld2q_s16(in);
    q11s16 = q0x2s16.val[0];
    in += 16;
    q0x2s16 = vld2q_s16(in);
    q12s16 = q0x2s16.val[0];
    in += 16;
    q0x2s16 = vld2q_s16(in);
    q13s16 = q0x2s16.val[0];
    in += 16;
    q0x2s16 = vld2q_s16(in);
    q14s16 = q0x2s16.val[0];
    in += 16;
    q0x2s16 = vld2q_s16(in);
    q15s16 = q0x2s16.val[0];

    TRANSPOSE8X8(&q8s16, &q9s16, &q10s16, &q11s16,
                 &q12s16, &q13s16, &q14s16, &q15s16);

    d16s16 = vget_low_s16(q8s16);
    d17s16 = vget_high_s16(q8s16);
    d18s16 = vget_low_s16(q9s16);
    d19s16 = vget_high_s16(q9s16);
    d20s16 = vget_low_s16(q10s16);
    d21s16 = vget_high_s16(q10s16);
    d22s16 = vget_low_s16(q11s16);
    d23s16 = vget_high_s16(q11s16);
    d24s16 = vget_low_s16(q12s16);
    d25s16 = vget_high_s16(q12s16);
    d26s16 = vget_low_s16(q13s16);
    d27s16 = vget_high_s16(q13s16);
    d28s16 = vget_low_s16(q14s16);
    d29s16 = vget_high_s16(q14s16);
    d30s16 = vget_low_s16(q15s16);
    d31s16 = vget_high_s16(q15s16);

    // stage 3
    d0s16 = vdup_n_s16(cospi_28_64);
    d1s16 = vdup_n_s16(cospi_4_64);

    q2s32 = vmull_s16(d18s16, d0s16);
    q3s32 = vmull_s16(d19s16, d0s16);
    q5s32 = vmull_s16(d18s16, d1s16);
    q6s32 = vmull_s16(d19s16, d1s16);

    q2s32 = vmlsl_s16(q2s32, d30s16, d1s16);
    q3s32 = vmlsl_s16(q3s32, d31s16, d1s16);
    q5s32 = vmlal_s16(q5s32, d30s16, d0s16);
    q6s32 = vmlal_s16(q6s32, d31s16, d0s16);

    d2s16 = vdup_n_s16(cospi_12_64);
    d3s16 = vdup_n_s16(cospi_20_64);

    d8s16 = vqrshrn_n_s32(q2s32, 14);
    d9s16 = vqrshrn_n_s32(q3s32, 14);
    d14s16 = vqrshrn_n_s32(q5s32, 14);
    d15s16 = vqrshrn_n_s32(q6s32, 14);
    q4s16 = vcombine_s16(d8s16, d9s16);
    q7s16 = vcombine_s16(d14s16, d15s16);

    q2s32 = vmull_s16(d26s16, d2s16);
    q3s32 = vmull_s16(d27s16, d2s16);
    q9s32 = vmull_s16(d26s16, d3s16);
    q15s32 = vmull_s16(d27s16, d3s16);

    q2s32 = vmlsl_s16(q2s32, d22s16, d3s16);
    q3s32 = vmlsl_s16(q3s32, d23s16, d3s16);
    q9s32 = vmlal_s16(q9s32, d22s16, d2s16);
    q15s32 = vmlal_s16(q15s32, d23s16, d2s16);

    d10s16 = vqrshrn_n_s32(q2s32, 14);
    d11s16 = vqrshrn_n_s32(q3s32, 14);
    d12s16 = vqrshrn_n_s32(q9s32, 14);
    d13s16 = vqrshrn_n_s32(q15s32, 14);
    q5s16 = vcombine_s16(d10s16, d11s16);
    q6s16 = vcombine_s16(d12s16, d13s16);

    // stage 4
    d30s16 = vdup_n_s16(cospi_16_64);

    q2s32 = vmull_s16(d16s16, d30s16);
    q11s32 = vmull_s16(d17s16, d30s16);
    q0s32 = vmull_s16(d24s16, d30s16);
    q1s32 = vmull_s16(d25s16, d30s16);

    d30s16 = vdup_n_s16(cospi_24_64);
    d31s16 = vdup_n_s16(cospi_8_64);

    q3s32 = vaddq_s32(q2s32, q0s32);
    q12s32 = vaddq_s32(q11s32, q1s32);
    q13s32 = vsubq_s32(q2s32, q0s32);
    q1s32 = vsubq_s32(q11s32, q1s32);

    d16s16 = vqrshrn_n_s32(q3s32, 14);
    d17s16 = vqrshrn_n_s32(q12s32, 14);
    d18s16 = vqrshrn_n_s32(q13s32, 14);
    d19s16 = vqrshrn_n_s32(q1s32, 14);
    q8s16 = vcombine_s16(d16s16, d17s16);
    q9s16 = vcombine_s16(d18s16, d19s16);

    q0s32 = vmull_s16(d20s16, d31s16);
    q1s32 = vmull_s16(d21s16, d31s16);
    q12s32 = vmull_s16(d20s16, d30s16);
    q13s32 = vmull_s16(d21s16, d30s16);

    q0s32 = vmlal_s16(q0s32, d28s16, d30s16);
    q1s32 = vmlal_s16(q1s32, d29s16, d30s16);
    q12s32 = vmlsl_s16(q12s32, d28s16, d31s16);
    q13s32 = vmlsl_s16(q13s32, d29s16, d31s16);

    d22s16 = vqrshrn_n_s32(q0s32, 14);
    d23s16 = vqrshrn_n_s32(q1s32, 14);
    d20s16 = vqrshrn_n_s32(q12s32, 14);
    d21s16 = vqrshrn_n_s32(q13s32, 14);
    q10s16 = vcombine_s16(d20s16, d21s16);
    q11s16 = vcombine_s16(d22s16, d23s16);

    q13s16 = vsubq_s16(q4s16, q5s16);
    q4s16 = vaddq_s16(q4s16, q5s16);
    q14s16 = vsubq_s16(q7s16, q6s16);
    q15s16 = vaddq_s16(q6s16, q7s16);
    d26s16 = vget_low_s16(q13s16);
    d27s16 = vget_high_s16(q13s16);
    d28s16 = vget_low_s16(q14s16);
    d29s16 = vget_high_s16(q14s16);

    // stage 5
    q0s16 = vaddq_s16(q8s16, q11s16);
    q1s16 = vaddq_s16(q9s16, q10s16);
    q2s16 = vsubq_s16(q9s16, q10s16);
    q3s16 = vsubq_s16(q8s16, q11s16);

    d16s16 = vdup_n_s16(cospi_16_64);

    q11s32 = vmull_s16(d26s16, d16s16);
    q12s32 = vmull_s16(d27s16, d16s16);
    q9s32 = vmull_s16(d28s16, d16s16);
    q10s32 = vmull_s16(d29s16, d16s16);

    q6s32 = vsubq_s32(q9s32, q11s32);
    q13s32 = vsubq_s32(q10s32, q12s32);
    q9s32 = vaddq_s32(q9s32, q11s32);
    q10s32 = vaddq_s32(q10s32, q12s32);

    d10s16 = vqrshrn_n_s32(q6s32, 14);
    d11s16 = vqrshrn_n_s32(q13s32, 14);
    d12s16 = vqrshrn_n_s32(q9s32, 14);
    d13s16 = vqrshrn_n_s32(q10s32, 14);
    q5s16 = vcombine_s16(d10s16, d11s16);
    q6s16 = vcombine_s16(d12s16, d13s16);

    // stage 6
    q8s16 = vaddq_s16(q0s16, q15s16);
    q9s16 = vaddq_s16(q1s16, q6s16);
    q10s16 = vaddq_s16(q2s16, q5s16);
    q11s16 = vaddq_s16(q3s16, q4s16);
    q12s16 = vsubq_s16(q3s16, q4s16);
    q13s16 = vsubq_s16(q2s16, q5s16);
    q14s16 = vsubq_s16(q1s16, q6s16);
    q15s16 = vsubq_s16(q0s16, q15s16);

    d16u64 = vreinterpret_u64_s16(vget_low_s16(q8s16));
    d17u64 = vreinterpret_u64_s16(vget_high_s16(q8s16));
    d18u64 = vreinterpret_u64_s16(vget_low_s16(q9s16));
    d19u64 = vreinterpret_u64_s16(vget_high_s16(q9s16));
    d20u64 = vreinterpret_u64_s16(vget_low_s16(q10s16));
    d21u64 = vreinterpret_u64_s16(vget_high_s16(q10s16));
    d22u64 = vreinterpret_u64_s16(vget_low_s16(q11s16));
    d23u64 = vreinterpret_u64_s16(vget_high_s16(q11s16));
    d24u64 = vreinterpret_u64_s16(vget_low_s16(q12s16));
    d25u64 = vreinterpret_u64_s16(vget_high_s16(q12s16));
    d26u64 = vreinterpret_u64_s16(vget_low_s16(q13s16));
    d27u64 = vreinterpret_u64_s16(vget_high_s16(q13s16));
    d28u64 = vreinterpret_u64_s16(vget_low_s16(q14s16));
    d29u64 = vreinterpret_u64_s16(vget_high_s16(q14s16));
    d30u64 = vreinterpret_u64_s16(vget_low_s16(q15s16));
    d31u64 = vreinterpret_u64_s16(vget_high_s16(q15s16));

    // store the data
    output_stride >>= 1;  // output_stride / 2, out is int16_t
    vst1_u64((uint64_t *)out, d16u64);
    out += output_stride;
    vst1_u64((uint64_t *)out, d17u64);
    out += output_stride;
    vst1_u64((uint64_t *)out, d18u64);
    out += output_stride;
    vst1_u64((uint64_t *)out, d19u64);
    out += output_stride;
    vst1_u64((uint64_t *)out, d20u64);
    out += output_stride;
    vst1_u64((uint64_t *)out, d21u64);
    out += output_stride;
    vst1_u64((uint64_t *)out, d22u64);
    out += output_stride;
    vst1_u64((uint64_t *)out, d23u64);
    out += output_stride;
    vst1_u64((uint64_t *)out, d24u64);
    out += output_stride;
    vst1_u64((uint64_t *)out, d25u64);
    out += output_stride;
    vst1_u64((uint64_t *)out, d26u64);
    out += output_stride;
    vst1_u64((uint64_t *)out, d27u64);
    out += output_stride;
    vst1_u64((uint64_t *)out, d28u64);
    out += output_stride;
    vst1_u64((uint64_t *)out, d29u64);
    out += output_stride;
    vst1_u64((uint64_t *)out, d30u64);
    out += output_stride;
    vst1_u64((uint64_t *)out, d31u64);
    return;
}

void vpx_idct16x16_256_add_neon_pass2(
        int16_t *src,
        int16_t *out,
        int16_t *pass1Output,
        int16_t skip_adding,
        uint8_t *dest,
        int dest_stride) {
    uint8_t *d;
    uint8x8_t d12u8, d13u8;
    int16x4_t d0s16, d1s16, d2s16, d3s16, d4s16, d5s16, d6s16, d7s16;
    int16x4_t d8s16, d9s16, d10s16, d11s16, d12s16, d13s16, d14s16, d15s16;
    int16x4_t d16s16, d17s16, d18s16, d19s16, d20s16, d21s16, d22s16, d23s16;
    int16x4_t d24s16, d25s16, d26s16, d27s16, d28s16, d29s16, d30s16, d31s16;
    uint64x1_t d24u64, d25u64, d26u64, d27u64;
    int64x1_t d12s64, d13s64;
    uint16x8_t q2u16, q3u16, q4u16, q5u16, q8u16;
    uint16x8_t q9u16, q12u16, q13u16, q14u16, q15u16;
    int16x8_t q0s16, q1s16, q2s16, q3s16, q4s16, q5s16, q6s16, q7s16;
    int16x8_t q8s16, q9s16, q10s16, q11s16, q12s16, q13s16, q14s16, q15s16;
    int32x4_t q0s32, q1s32, q2s32, q3s32, q4s32, q5s32, q6s32, q8s32, q9s32;
    int32x4_t q10s32, q11s32, q12s32, q13s32;
    int16x8x2_t q0x2s16;

    q0x2s16 = vld2q_s16(src);
    q8s16  = q0x2s16.val[0];
    src += 16;
    q0x2s16 = vld2q_s16(src);
    q9s16  = q0x2s16.val[0];
    src += 16;
    q0x2s16 = vld2q_s16(src);
    q10s16 = q0x2s16.val[0];
    src += 16;
    q0x2s16 = vld2q_s16(src);
    q11s16 = q0x2s16.val[0];
    src += 16;
    q0x2s16 = vld2q_s16(src);
    q12s16 = q0x2s16.val[0];
    src += 16;
    q0x2s16 = vld2q_s16(src);
    q13s16 = q0x2s16.val[0];
    src += 16;
    q0x2s16 = vld2q_s16(src);
    q14s16 = q0x2s16.val[0];
    src += 16;
    q0x2s16 = vld2q_s16(src);
    q15s16 = q0x2s16.val[0];

    TRANSPOSE8X8(&q8s16, &q9s16, &q10s16, &q11s16,
                 &q12s16, &q13s16, &q14s16, &q15s16);

    d16s16 = vget_low_s16(q8s16);
    d17s16 = vget_high_s16(q8s16);
    d18s16 = vget_low_s16(q9s16);
    d19s16 = vget_high_s16(q9s16);
    d20s16 = vget_low_s16(q10s16);
    d21s16 = vget_high_s16(q10s16);
    d22s16 = vget_low_s16(q11s16);
    d23s16 = vget_high_s16(q11s16);
    d24s16 = vget_low_s16(q12s16);
    d25s16 = vget_high_s16(q12s16);
    d26s16 = vget_low_s16(q13s16);
    d27s16 = vget_high_s16(q13s16);
    d28s16 = vget_low_s16(q14s16);
    d29s16 = vget_high_s16(q14s16);
    d30s16 = vget_low_s16(q15s16);
    d31s16 = vget_high_s16(q15s16);

    // stage 3
    d12s16 = vdup_n_s16(cospi_30_64);
    d13s16 = vdup_n_s16(cospi_2_64);

    q2s32 = vmull_s16(d16s16, d12s16);
    q3s32 = vmull_s16(d17s16, d12s16);
    q1s32 = vmull_s16(d16s16, d13s16);
    q4s32 = vmull_s16(d17s16, d13s16);

    q2s32 = vmlsl_s16(q2s32, d30s16, d13s16);
    q3s32 = vmlsl_s16(q3s32, d31s16, d13s16);
    q1s32 = vmlal_s16(q1s32, d30s16, d12s16);
    q4s32 = vmlal_s16(q4s32, d31s16, d12s16);

    d0s16 = vqrshrn_n_s32(q2s32, 14);
    d1s16 = vqrshrn_n_s32(q3s32, 14);
    d14s16 = vqrshrn_n_s32(q1s32, 14);
    d15s16 = vqrshrn_n_s32(q4s32, 14);
    q0s16 = vcombine_s16(d0s16, d1s16);
    q7s16 = vcombine_s16(d14s16, d15s16);

    d30s16 = vdup_n_s16(cospi_14_64);
    d31s16 = vdup_n_s16(cospi_18_64);

    q2s32 = vmull_s16(d24s16, d30s16);
    q3s32 = vmull_s16(d25s16, d30s16);
    q4s32 = vmull_s16(d24s16, d31s16);
    q5s32 = vmull_s16(d25s16, d31s16);

    q2s32 = vmlsl_s16(q2s32, d22s16, d31s16);
    q3s32 = vmlsl_s16(q3s32, d23s16, d31s16);
    q4s32 = vmlal_s16(q4s32, d22s16, d30s16);
    q5s32 = vmlal_s16(q5s32, d23s16, d30s16);

    d2s16 = vqrshrn_n_s32(q2s32, 14);
    d3s16 = vqrshrn_n_s32(q3s32, 14);
    d12s16 = vqrshrn_n_s32(q4s32, 14);
    d13s16 = vqrshrn_n_s32(q5s32, 14);
    q1s16 = vcombine_s16(d2s16, d3s16);
    q6s16 = vcombine_s16(d12s16, d13s16);

    d30s16 = vdup_n_s16(cospi_22_64);
    d31s16 = vdup_n_s16(cospi_10_64);

    q11s32 = vmull_s16(d20s16, d30s16);
    q12s32 = vmull_s16(d21s16, d30s16);
    q4s32 = vmull_s16(d20s16, d31s16);
    q5s32 = vmull_s16(d21s16, d31s16);

    q11s32 = vmlsl_s16(q11s32, d26s16, d31s16);
    q12s32 = vmlsl_s16(q12s32, d27s16, d31s16);
    q4s32 = vmlal_s16(q4s32, d26s16, d30s16);
    q5s32 = vmlal_s16(q5s32, d27s16, d30s16);

    d4s16 = vqrshrn_n_s32(q11s32, 14);
    d5s16 = vqrshrn_n_s32(q12s32, 14);
    d11s16 = vqrshrn_n_s32(q5s32, 14);
    d10s16 = vqrshrn_n_s32(q4s32, 14);
    q2s16 = vcombine_s16(d4s16, d5s16);
    q5s16 = vcombine_s16(d10s16, d11s16);

    d30s16 = vdup_n_s16(cospi_6_64);
    d31s16 = vdup_n_s16(cospi_26_64);

    q10s32 = vmull_s16(d28s16, d30s16);
    q11s32 = vmull_s16(d29s16, d30s16);
    q12s32 = vmull_s16(d28s16, d31s16);
    q13s32 = vmull_s16(d29s16, d31s16);

    q10s32 = vmlsl_s16(q10s32, d18s16, d31s16);
    q11s32 = vmlsl_s16(q11s32, d19s16, d31s16);
    q12s32 = vmlal_s16(q12s32, d18s16, d30s16);
    q13s32 = vmlal_s16(q13s32, d19s16, d30s16);

    d6s16 = vqrshrn_n_s32(q10s32, 14);
    d7s16 = vqrshrn_n_s32(q11s32, 14);
    d8s16 = vqrshrn_n_s32(q12s32, 14);
    d9s16 = vqrshrn_n_s32(q13s32, 14);
    q3s16 = vcombine_s16(d6s16, d7s16);
    q4s16 = vcombine_s16(d8s16, d9s16);

    // stage 3
    q9s16  = vsubq_s16(q0s16, q1s16);
    q0s16  = vaddq_s16(q0s16, q1s16);
    q10s16 = vsubq_s16(q3s16, q2s16);
    q11s16 = vaddq_s16(q2s16, q3s16);
    q12s16 = vaddq_s16(q4s16, q5s16);
    q13s16 = vsubq_s16(q4s16, q5s16);
    q14s16 = vsubq_s16(q7s16, q6s16);
    q7s16  = vaddq_s16(q6s16, q7s16);

    // stage 4
    d18s16 = vget_low_s16(q9s16);
    d19s16 = vget_high_s16(q9s16);
    d20s16 = vget_low_s16(q10s16);
    d21s16 = vget_high_s16(q10s16);
    d26s16 = vget_low_s16(q13s16);
    d27s16 = vget_high_s16(q13s16);
    d28s16 = vget_low_s16(q14s16);
    d29s16 = vget_high_s16(q14s16);

    d30s16 = vdup_n_s16(cospi_8_64);
    d31s16 = vdup_n_s16(cospi_24_64);

    q2s32 = vmull_s16(d18s16, d31s16);
    q3s32 = vmull_s16(d19s16, d31s16);
    q4s32 = vmull_s16(d28s16, d31s16);
    q5s32 = vmull_s16(d29s16, d31s16);

    q2s32 = vmlal_s16(q2s32, d28s16, d30s16);
    q3s32 = vmlal_s16(q3s32, d29s16, d30s16);
    q4s32 = vmlsl_s16(q4s32, d18s16, d30s16);
    q5s32 = vmlsl_s16(q5s32, d19s16, d30s16);

    d12s16 = vqrshrn_n_s32(q2s32, 14);
    d13s16 = vqrshrn_n_s32(q3s32, 14);
    d2s16 = vqrshrn_n_s32(q4s32, 14);
    d3s16 = vqrshrn_n_s32(q5s32, 14);
    q1s16 = vcombine_s16(d2s16, d3s16);
    q6s16 = vcombine_s16(d12s16, d13s16);

    q3s16 = q11s16;
    q4s16 = q12s16;

    d30s16 = vdup_n_s16(-cospi_8_64);
    q11s32 = vmull_s16(d26s16, d30s16);
    q12s32 = vmull_s16(d27s16, d30s16);
    q8s32 = vmull_s16(d20s16, d30s16);
    q9s32 = vmull_s16(d21s16, d30s16);

    q11s32 = vmlsl_s16(q11s32, d20s16, d31s16);
    q12s32 = vmlsl_s16(q12s32, d21s16, d31s16);
    q8s32 = vmlal_s16(q8s32, d26s16, d31s16);
    q9s32 = vmlal_s16(q9s32, d27s16, d31s16);

    d4s16 = vqrshrn_n_s32(q11s32, 14);
    d5s16 = vqrshrn_n_s32(q12s32, 14);
    d10s16 = vqrshrn_n_s32(q8s32, 14);
    d11s16 = vqrshrn_n_s32(q9s32, 14);
    q2s16 = vcombine_s16(d4s16, d5s16);
    q5s16 = vcombine_s16(d10s16, d11s16);

    // stage 5
    q8s16  = vaddq_s16(q0s16, q3s16);
    q9s16  = vaddq_s16(q1s16, q2s16);
    q10s16 = vsubq_s16(q1s16, q2s16);
    q11s16 = vsubq_s16(q0s16, q3s16);
    q12s16 = vsubq_s16(q7s16, q4s16);
    q13s16 = vsubq_s16(q6s16, q5s16);
    q14s16 = vaddq_s16(q6s16, q5s16);
    q15s16 = vaddq_s16(q7s16, q4s16);

    // stage 6
    d20s16 = vget_low_s16(q10s16);
    d21s16 = vget_high_s16(q10s16);
    d22s16 = vget_low_s16(q11s16);
    d23s16 = vget_high_s16(q11s16);
    d24s16 = vget_low_s16(q12s16);
    d25s16 = vget_high_s16(q12s16);
    d26s16 = vget_low_s16(q13s16);
    d27s16 = vget_high_s16(q13s16);

    d14s16 = vdup_n_s16(cospi_16_64);

    q3s32 = vmull_s16(d26s16, d14s16);
    q4s32 = vmull_s16(d27s16, d14s16);
    q0s32 = vmull_s16(d20s16, d14s16);
    q1s32 = vmull_s16(d21s16, d14s16);

    q5s32 = vsubq_s32(q3s32, q0s32);
    q6s32 = vsubq_s32(q4s32, q1s32);
    q10s32 = vaddq_s32(q3s32, q0s32);
    q4s32 = vaddq_s32(q4s32, q1s32);

    d4s16 = vqrshrn_n_s32(q5s32, 14);
    d5s16 = vqrshrn_n_s32(q6s32, 14);
    d10s16 = vqrshrn_n_s32(q10s32, 14);
    d11s16 = vqrshrn_n_s32(q4s32, 14);
    q2s16 = vcombine_s16(d4s16, d5s16);
    q5s16 = vcombine_s16(d10s16, d11s16);

    q0s32 = vmull_s16(d22s16, d14s16);
    q1s32 = vmull_s16(d23s16, d14s16);
    q13s32 = vmull_s16(d24s16, d14s16);
    q6s32 = vmull_s16(d25s16, d14s16);

    q10s32 = vsubq_s32(q13s32, q0s32);
    q4s32 = vsubq_s32(q6s32, q1s32);
    q13s32 = vaddq_s32(q13s32, q0s32);
    q6s32 = vaddq_s32(q6s32, q1s32);

    d6s16 = vqrshrn_n_s32(q10s32, 14);
    d7s16 = vqrshrn_n_s32(q4s32, 14);
    d8s16 = vqrshrn_n_s32(q13s32, 14);
    d9s16 = vqrshrn_n_s32(q6s32, 14);
    q3s16 = vcombine_s16(d6s16, d7s16);
    q4s16 = vcombine_s16(d8s16, d9s16);

    // stage 7
    if (skip_adding != 0) {
        d = dest;
        // load the data in pass1
        q0s16 = vld1q_s16(pass1Output);
        pass1Output += 8;
        q1s16 = vld1q_s16(pass1Output);
        pass1Output += 8;
        d12s64 = vld1_s64((int64_t *)dest);
        dest += dest_stride;
        d13s64 = vld1_s64((int64_t *)dest);
        dest += dest_stride;

        q12s16 = vaddq_s16(q0s16, q15s16);
        q13s16 = vaddq_s16(q1s16, q14s16);
        q12s16 = vrshrq_n_s16(q12s16, 6);
        q13s16 = vrshrq_n_s16(q13s16, 6);
        q12u16 = vaddw_u8(vreinterpretq_u16_s16(q12s16),
                          vreinterpret_u8_s64(d12s64));
        q13u16 = vaddw_u8(vreinterpretq_u16_s16(q13s16),
                          vreinterpret_u8_s64(d13s64));
        d12u8 = vqmovun_s16(vreinterpretq_s16_u16(q12u16));
        d13u8 = vqmovun_s16(vreinterpretq_s16_u16(q13u16));
        vst1_u64((uint64_t *)d, vreinterpret_u64_u8(d12u8));
        d += dest_stride;
        vst1_u64((uint64_t *)d, vreinterpret_u64_u8(d13u8));
        d += dest_stride;
        q14s16 = vsubq_s16(q1s16, q14s16);
        q15s16 = vsubq_s16(q0s16, q15s16);

        q10s16 = vld1q_s16(pass1Output);
        pass1Output += 8;
        q11s16 = vld1q_s16(pass1Output);
        pass1Output += 8;
        d12s64 = vld1_s64((int64_t *)dest);
        dest += dest_stride;
        d13s64 = vld1_s64((int64_t *)dest);
        dest += dest_stride;
        q12s16 = vaddq_s16(q10s16, q5s16);
        q13s16 = vaddq_s16(q11s16, q4s16);
        q12s16 = vrshrq_n_s16(q12s16, 6);
        q13s16 = vrshrq_n_s16(q13s16, 6);
        q12u16 = vaddw_u8(vreinterpretq_u16_s16(q12s16),
                          vreinterpret_u8_s64(d12s64));
        q13u16 = vaddw_u8(vreinterpretq_u16_s16(q13s16),
                          vreinterpret_u8_s64(d13s64));
        d12u8 = vqmovun_s16(vreinterpretq_s16_u16(q12u16));
        d13u8 = vqmovun_s16(vreinterpretq_s16_u16(q13u16));
        vst1_u64((uint64_t *)d, vreinterpret_u64_u8(d12u8));
        d += dest_stride;
        vst1_u64((uint64_t *)d, vreinterpret_u64_u8(d13u8));
        d += dest_stride;
        q4s16 = vsubq_s16(q11s16, q4s16);
        q5s16 = vsubq_s16(q10s16, q5s16);

        q0s16 = vld1q_s16(pass1Output);
        pass1Output += 8;
        q1s16 = vld1q_s16(pass1Output);
        pass1Output += 8;
        d12s64 = vld1_s64((int64_t *)dest);
        dest += dest_stride;
        d13s64 = vld1_s64((int64_t *)dest);
        dest += dest_stride;
        q12s16 = vaddq_s16(q0s16, q3s16);
        q13s16 = vaddq_s16(q1s16, q2s16);
        q12s16 = vrshrq_n_s16(q12s16, 6);
        q13s16 = vrshrq_n_s16(q13s16, 6);
        q12u16 = vaddw_u8(vreinterpretq_u16_s16(q12s16),
                          vreinterpret_u8_s64(d12s64));
        q13u16 = vaddw_u8(vreinterpretq_u16_s16(q13s16),
                          vreinterpret_u8_s64(d13s64));
        d12u8 = vqmovun_s16(vreinterpretq_s16_u16(q12u16));
        d13u8 = vqmovun_s16(vreinterpretq_s16_u16(q13u16));
        vst1_u64((uint64_t *)d, vreinterpret_u64_u8(d12u8));
        d += dest_stride;
        vst1_u64((uint64_t *)d, vreinterpret_u64_u8(d13u8));
        d += dest_stride;
        q2s16 = vsubq_s16(q1s16, q2s16);
        q3s16 = vsubq_s16(q0s16, q3s16);

        q10s16 = vld1q_s16(pass1Output);
        pass1Output += 8;
        q11s16 = vld1q_s16(pass1Output);
        d12s64 = vld1_s64((int64_t *)dest);
        dest += dest_stride;
        d13s64 = vld1_s64((int64_t *)dest);
        dest += dest_stride;
        q12s16 = vaddq_s16(q10s16, q9s16);
        q13s16 = vaddq_s16(q11s16, q8s16);
        q12s16 = vrshrq_n_s16(q12s16, 6);
        q13s16 = vrshrq_n_s16(q13s16, 6);
        q12u16 = vaddw_u8(vreinterpretq_u16_s16(q12s16),
                          vreinterpret_u8_s64(d12s64));
        q13u16 = vaddw_u8(vreinterpretq_u16_s16(q13s16),
                          vreinterpret_u8_s64(d13s64));
        d12u8 = vqmovun_s16(vreinterpretq_s16_u16(q12u16));
        d13u8 = vqmovun_s16(vreinterpretq_s16_u16(q13u16));
        vst1_u64((uint64_t *)d, vreinterpret_u64_u8(d12u8));
        d += dest_stride;
        vst1_u64((uint64_t *)d, vreinterpret_u64_u8(d13u8));
        d += dest_stride;
        q8s16 = vsubq_s16(q11s16, q8s16);
        q9s16 = vsubq_s16(q10s16, q9s16);

        // store the data  out 8,9,10,11,12,13,14,15
        d12s64 = vld1_s64((int64_t *)dest);
        dest += dest_stride;
        q8s16 = vrshrq_n_s16(q8s16, 6);
        q8u16 = vaddw_u8(vreinterpretq_u16_s16(q8s16),
                         vreinterpret_u8_s64(d12s64));
        d12u8 = vqmovun_s16(vreinterpretq_s16_u16(q8u16));
        vst1_u64((uint64_t *)d, vreinterpret_u64_u8(d12u8));
        d += dest_stride;

        d12s64 = vld1_s64((int64_t *)dest);
        dest += dest_stride;
        q9s16 = vrshrq_n_s16(q9s16, 6);
        q9u16 = vaddw_u8(vreinterpretq_u16_s16(q9s16),
                          vreinterpret_u8_s64(d12s64));
        d12u8 = vqmovun_s16(vreinterpretq_s16_u16(q9u16));
        vst1_u64((uint64_t *)d, vreinterpret_u64_u8(d12u8));
        d += dest_stride;

        d12s64 = vld1_s64((int64_t *)dest);
        dest += dest_stride;
        q2s16 = vrshrq_n_s16(q2s16, 6);
        q2u16 = vaddw_u8(vreinterpretq_u16_s16(q2s16),
                          vreinterpret_u8_s64(d12s64));
        d12u8 = vqmovun_s16(vreinterpretq_s16_u16(q2u16));
        vst1_u64((uint64_t *)d, vreinterpret_u64_u8(d12u8));
        d += dest_stride;

        d12s64 = vld1_s64((int64_t *)dest);
        dest += dest_stride;
        q3s16 = vrshrq_n_s16(q3s16, 6);
        q3u16 = vaddw_u8(vreinterpretq_u16_s16(q3s16),
                         vreinterpret_u8_s64(d12s64));
        d12u8 = vqmovun_s16(vreinterpretq_s16_u16(q3u16));
        vst1_u64((uint64_t *)d, vreinterpret_u64_u8(d12u8));
        d += dest_stride;

        d12s64 = vld1_s64((int64_t *)dest);
        dest += dest_stride;
        q4s16 = vrshrq_n_s16(q4s16, 6);
        q4u16 = vaddw_u8(vreinterpretq_u16_s16(q4s16),
                         vreinterpret_u8_s64(d12s64));
        d12u8 = vqmovun_s16(vreinterpretq_s16_u16(q4u16));
        vst1_u64((uint64_t *)d, vreinterpret_u64_u8(d12u8));
        d += dest_stride;

        d12s64 = vld1_s64((int64_t *)dest);
        dest += dest_stride;
        q5s16 = vrshrq_n_s16(q5s16, 6);
        q5u16 = vaddw_u8(vreinterpretq_u16_s16(q5s16),
                         vreinterpret_u8_s64(d12s64));
        d12u8 = vqmovun_s16(vreinterpretq_s16_u16(q5u16));
        vst1_u64((uint64_t *)d, vreinterpret_u64_u8(d12u8));
        d += dest_stride;

        d12s64 = vld1_s64((int64_t *)dest);
        dest += dest_stride;
        q14s16 = vrshrq_n_s16(q14s16, 6);
        q14u16 = vaddw_u8(vreinterpretq_u16_s16(q14s16),
                          vreinterpret_u8_s64(d12s64));
        d12u8 = vqmovun_s16(vreinterpretq_s16_u16(q14u16));
        vst1_u64((uint64_t *)d, vreinterpret_u64_u8(d12u8));
        d += dest_stride;

        d12s64 = vld1_s64((int64_t *)dest);
        q15s16 = vrshrq_n_s16(q15s16, 6);
        q15u16 = vaddw_u8(vreinterpretq_u16_s16(q15s16),
                          vreinterpret_u8_s64(d12s64));
        d12u8 = vqmovun_s16(vreinterpretq_s16_u16(q15u16));
        vst1_u64((uint64_t *)d, vreinterpret_u64_u8(d12u8));
    } else {  // skip_adding_dest
        q0s16 = vld1q_s16(pass1Output);
        pass1Output += 8;
        q1s16 = vld1q_s16(pass1Output);
        pass1Output += 8;
        q12s16 = vaddq_s16(q0s16, q15s16);
        q13s16 = vaddq_s16(q1s16, q14s16);
        d24u64 = vreinterpret_u64_s16(vget_low_s16(q12s16));
        d25u64 = vreinterpret_u64_s16(vget_high_s16(q12s16));
        d26u64 = vreinterpret_u64_s16(vget_low_s16(q13s16));
        d27u64 = vreinterpret_u64_s16(vget_high_s16(q13s16));
        vst1_u64((uint64_t *)out, d24u64);
        out += 4;
        vst1_u64((uint64_t *)out, d25u64);
        out += 12;
        vst1_u64((uint64_t *)out, d26u64);
        out += 4;
        vst1_u64((uint64_t *)out, d27u64);
        out += 12;
        q14s16 = vsubq_s16(q1s16, q14s16);
        q15s16 = vsubq_s16(q0s16, q15s16);

        q10s16 = vld1q_s16(pass1Output);
        pass1Output += 8;
        q11s16 = vld1q_s16(pass1Output);
        pass1Output += 8;
        q12s16 = vaddq_s16(q10s16, q5s16);
        q13s16 = vaddq_s16(q11s16, q4s16);
        d24u64 = vreinterpret_u64_s16(vget_low_s16(q12s16));
        d25u64 = vreinterpret_u64_s16(vget_high_s16(q12s16));
        d26u64 = vreinterpret_u64_s16(vget_low_s16(q13s16));
        d27u64 = vreinterpret_u64_s16(vget_high_s16(q13s16));
        vst1_u64((uint64_t *)out, d24u64);
        out += 4;
        vst1_u64((uint64_t *)out, d25u64);
        out += 12;
        vst1_u64((uint64_t *)out, d26u64);
        out += 4;
        vst1_u64((uint64_t *)out, d27u64);
        out += 12;
        q4s16 = vsubq_s16(q11s16, q4s16);
        q5s16 = vsubq_s16(q10s16, q5s16);

        q0s16 = vld1q_s16(pass1Output);
        pass1Output += 8;
        q1s16 = vld1q_s16(pass1Output);
        pass1Output += 8;
        q12s16 = vaddq_s16(q0s16, q3s16);
        q13s16 = vaddq_s16(q1s16, q2s16);
        d24u64 = vreinterpret_u64_s16(vget_low_s16(q12s16));
        d25u64 = vreinterpret_u64_s16(vget_high_s16(q12s16));
        d26u64 = vreinterpret_u64_s16(vget_low_s16(q13s16));
        d27u64 = vreinterpret_u64_s16(vget_high_s16(q13s16));
        vst1_u64((uint64_t *)out, d24u64);
        out += 4;
        vst1_u64((uint64_t *)out, d25u64);
        out += 12;
        vst1_u64((uint64_t *)out, d26u64);
        out += 4;
        vst1_u64((uint64_t *)out, d27u64);
        out += 12;
        q2s16 = vsubq_s16(q1s16, q2s16);
        q3s16 = vsubq_s16(q0s16, q3s16);

        q10s16 = vld1q_s16(pass1Output);
        pass1Output += 8;
        q11s16 = vld1q_s16(pass1Output);
        pass1Output += 8;
        q12s16 = vaddq_s16(q10s16, q9s16);
        q13s16 = vaddq_s16(q11s16, q8s16);
        d24u64 = vreinterpret_u64_s16(vget_low_s16(q12s16));
        d25u64 = vreinterpret_u64_s16(vget_high_s16(q12s16));
        d26u64 = vreinterpret_u64_s16(vget_low_s16(q13s16));
        d27u64 = vreinterpret_u64_s16(vget_high_s16(q13s16));
        vst1_u64((uint64_t *)out, d24u64);
        out += 4;
        vst1_u64((uint64_t *)out, d25u64);
        out += 12;
        vst1_u64((uint64_t *)out, d26u64);
        out += 4;
        vst1_u64((uint64_t *)out, d27u64);
        out += 12;
        q8s16 = vsubq_s16(q11s16, q8s16);
        q9s16 = vsubq_s16(q10s16, q9s16);

        vst1_u64((uint64_t *)out, vreinterpret_u64_s16(vget_low_s16(q8s16)));
        out += 4;
        vst1_u64((uint64_t *)out, vreinterpret_u64_s16(vget_high_s16(q8s16)));
        out += 12;
        vst1_u64((uint64_t *)out, vreinterpret_u64_s16(vget_low_s16(q9s16)));
        out += 4;
        vst1_u64((uint64_t *)out, vreinterpret_u64_s16(vget_high_s16(q9s16)));
        out += 12;
        vst1_u64((uint64_t *)out, vreinterpret_u64_s16(vget_low_s16(q2s16)));
        out += 4;
        vst1_u64((uint64_t *)out, vreinterpret_u64_s16(vget_high_s16(q2s16)));
        out += 12;
        vst1_u64((uint64_t *)out, vreinterpret_u64_s16(vget_low_s16(q3s16)));
        out += 4;
        vst1_u64((uint64_t *)out, vreinterpret_u64_s16(vget_high_s16(q3s16)));
        out += 12;
        vst1_u64((uint64_t *)out, vreinterpret_u64_s16(vget_low_s16(q4s16)));
        out += 4;
        vst1_u64((uint64_t *)out, vreinterpret_u64_s16(vget_high_s16(q4s16)));
        out += 12;
        vst1_u64((uint64_t *)out, vreinterpret_u64_s16(vget_low_s16(q5s16)));
        out += 4;
        vst1_u64((uint64_t *)out, vreinterpret_u64_s16(vget_high_s16(q5s16)));
        out += 12;
        vst1_u64((uint64_t *)out, vreinterpret_u64_s16(vget_low_s16(q14s16)));
        out += 4;
        vst1_u64((uint64_t *)out, vreinterpret_u64_s16(vget_high_s16(q14s16)));
        out += 12;
        vst1_u64((uint64_t *)out, vreinterpret_u64_s16(vget_low_s16(q15s16)));
        out += 4;
        vst1_u64((uint64_t *)out, vreinterpret_u64_s16(vget_high_s16(q15s16)));
    }
    return;
}

void vpx_idct16x16_10_add_neon_pass1(
        int16_t *in,
        int16_t *out,
        int output_stride) {
    int16x4_t d4s16;
    int16x4_t d8s16, d9s16, d10s16, d11s16, d12s16, d13s16, d14s16, d15s16;
    uint64x1_t d4u64, d5u64, d18u64, d19u64, d20u64, d21u64, d22u64, d23u64;
    uint64x1_t d24u64, d25u64, d26u64, d27u64, d28u64, d29u64, d30u64, d31u64;
    int16x8_t q0s16, q1s16, q2s16, q4s16, q5s16, q6s16, q7s16;
    int16x8_t q8s16, q9s16, q10s16, q11s16, q12s16, q13s16, q14s16, q15s16;
    int32x4_t q6s32, q9s32;
    int32x4_t q10s32, q11s32, q12s32, q15s32;
    int16x8x2_t q0x2s16;

    q0x2s16 = vld2q_s16(in);
    q8s16 = q0x2s16.val[0];
    in += 16;
    q0x2s16 = vld2q_s16(in);
    q9s16 = q0x2s16.val[0];
    in += 16;
    q0x2s16 = vld2q_s16(in);
    q10s16 = q0x2s16.val[0];
    in += 16;
    q0x2s16 = vld2q_s16(in);
    q11s16 = q0x2s16.val[0];
    in += 16;
    q0x2s16 = vld2q_s16(in);
    q12s16 = q0x2s16.val[0];
    in += 16;
    q0x2s16 = vld2q_s16(in);
    q13s16 = q0x2s16.val[0];
    in += 16;
    q0x2s16 = vld2q_s16(in);
    q14s16 = q0x2s16.val[0];
    in += 16;
    q0x2s16 = vld2q_s16(in);
    q15s16 = q0x2s16.val[0];

    TRANSPOSE8X8(&q8s16, &q9s16, &q10s16, &q11s16,
                 &q12s16, &q13s16, &q14s16, &q15s16);

    // stage 3
    q0s16 = vdupq_n_s16(cospi_28_64 * 2);
    q1s16 = vdupq_n_s16(cospi_4_64 * 2);

    q4s16 = vqrdmulhq_s16(q9s16, q0s16);
    q7s16 = vqrdmulhq_s16(q9s16, q1s16);

    // stage 4
    q1s16 = vdupq_n_s16(cospi_16_64 * 2);
    d4s16 = vdup_n_s16(cospi_16_64);

    q8s16 = vqrdmulhq_s16(q8s16, q1s16);

    d8s16 = vget_low_s16(q4s16);
    d9s16 = vget_high_s16(q4s16);
    d14s16 = vget_low_s16(q7s16);
    d15s16 = vget_high_s16(q7s16);
    q9s32  = vmull_s16(d14s16, d4s16);
    q10s32 = vmull_s16(d15s16, d4s16);
    q12s32 = vmull_s16(d9s16, d4s16);
    q11s32 = vmull_s16(d8s16, d4s16);

    q15s32 = vsubq_s32(q10s32, q12s32);
    q6s32 = vsubq_s32(q9s32, q11s32);
    q9s32 = vaddq_s32(q9s32, q11s32);
    q10s32 = vaddq_s32(q10s32, q12s32);

    d11s16 = vqrshrn_n_s32(q15s32, 14);
    d10s16 = vqrshrn_n_s32(q6s32, 14);
    d12s16 = vqrshrn_n_s32(q9s32, 14);
    d13s16 = vqrshrn_n_s32(q10s32, 14);
    q5s16 = vcombine_s16(d10s16, d11s16);
    q6s16 = vcombine_s16(d12s16, d13s16);

    // stage 6
    q2s16 = vaddq_s16(q8s16, q7s16);
    q9s16 = vaddq_s16(q8s16, q6s16);
    q10s16 = vaddq_s16(q8s16, q5s16);
    q11s16 = vaddq_s16(q8s16, q4s16);
    q12s16 = vsubq_s16(q8s16, q4s16);
    q13s16 = vsubq_s16(q8s16, q5s16);
    q14s16 = vsubq_s16(q8s16, q6s16);
    q15s16 = vsubq_s16(q8s16, q7s16);

    d4u64 = vreinterpret_u64_s16(vget_low_s16(q2s16));
    d5u64 = vreinterpret_u64_s16(vget_high_s16(q2s16));
    d18u64 = vreinterpret_u64_s16(vget_low_s16(q9s16));
    d19u64 = vreinterpret_u64_s16(vget_high_s16(q9s16));
    d20u64 = vreinterpret_u64_s16(vget_low_s16(q10s16));
    d21u64 = vreinterpret_u64_s16(vget_high_s16(q10s16));
    d22u64 = vreinterpret_u64_s16(vget_low_s16(q11s16));
    d23u64 = vreinterpret_u64_s16(vget_high_s16(q11s16));
    d24u64 = vreinterpret_u64_s16(vget_low_s16(q12s16));
    d25u64 = vreinterpret_u64_s16(vget_high_s16(q12s16));
    d26u64 = vreinterpret_u64_s16(vget_low_s16(q13s16));
    d27u64 = vreinterpret_u64_s16(vget_high_s16(q13s16));
    d28u64 = vreinterpret_u64_s16(vget_low_s16(q14s16));
    d29u64 = vreinterpret_u64_s16(vget_high_s16(q14s16));
    d30u64 = vreinterpret_u64_s16(vget_low_s16(q15s16));
    d31u64 = vreinterpret_u64_s16(vget_high_s16(q15s16));

    // store the data
    output_stride >>= 1;  // output_stride / 2, out is int16_t
    vst1_u64((uint64_t *)out, d4u64);
    out += output_stride;
    vst1_u64((uint64_t *)out, d5u64);
    out += output_stride;
    vst1_u64((uint64_t *)out, d18u64);
    out += output_stride;
    vst1_u64((uint64_t *)out, d19u64);
    out += output_stride;
    vst1_u64((uint64_t *)out, d20u64);
    out += output_stride;
    vst1_u64((uint64_t *)out, d21u64);
    out += output_stride;
    vst1_u64((uint64_t *)out, d22u64);
    out += output_stride;
    vst1_u64((uint64_t *)out, d23u64);
    out += output_stride;
    vst1_u64((uint64_t *)out, d24u64);
    out += output_stride;
    vst1_u64((uint64_t *)out, d25u64);
    out += output_stride;
    vst1_u64((uint64_t *)out, d26u64);
    out += output_stride;
    vst1_u64((uint64_t *)out, d27u64);
    out += output_stride;
    vst1_u64((uint64_t *)out, d28u64);
    out += output_stride;
    vst1_u64((uint64_t *)out, d29u64);
    out += output_stride;
    vst1_u64((uint64_t *)out, d30u64);
    out += output_stride;
    vst1_u64((uint64_t *)out, d31u64);
    return;
}

void vpx_idct16x16_10_add_neon_pass2(
        int16_t *src,
        int16_t *out,
        int16_t *pass1Output,
        int16_t skip_adding,
        uint8_t *dest,
        int dest_stride) {
    int16x4_t d0s16, d1s16, d2s16, d3s16, d4s16, d5s16, d6s16, d7s16;
    int16x4_t d8s16, d9s16, d10s16, d11s16, d12s16, d13s16, d14s16, d15s16;
    int16x4_t d20s16, d21s16, d22s16, d23s16;
    int16x4_t d24s16, d25s16, d26s16, d27s16, d30s16, d31s16;
    uint64x1_t d4u64, d5u64, d6u64, d7u64, d8u64, d9u64, d10u64, d11u64;
    uint64x1_t d16u64, d17u64, d18u64, d19u64;
    uint64x1_t d24u64, d25u64, d26u64, d27u64, d28u64, d29u64, d30u64, d31u64;
    int16x8_t q0s16, q1s16, q2s16, q3s16, q4s16, q5s16, q6s16, q7s16;
    int16x8_t q8s16, q9s16, q10s16, q11s16, q12s16, q13s16, q14s16, q15s16;
    int32x4_t q0s32, q1s32, q2s32, q3s32, q4s32, q5s32, q6s32, q8s32, q9s32;
    int32x4_t q10s32, q11s32, q12s32, q13s32;
    int16x8x2_t q0x2s16;
    (void)skip_adding;
    (void)dest;
    (void)dest_stride;

    q0x2s16 = vld2q_s16(src);
    q8s16 = q0x2s16.val[0];
    src += 16;
    q0x2s16 = vld2q_s16(src);
    q9s16 = q0x2s16.val[0];
    src += 16;
    q0x2s16 = vld2q_s16(src);
    q10s16 = q0x2s16.val[0];
    src += 16;
    q0x2s16 = vld2q_s16(src);
    q11s16 = q0x2s16.val[0];
    src += 16;
    q0x2s16 = vld2q_s16(src);
    q12s16 = q0x2s16.val[0];
    src += 16;
    q0x2s16 = vld2q_s16(src);
    q13s16 = q0x2s16.val[0];
    src += 16;
    q0x2s16 = vld2q_s16(src);
    q14s16 = q0x2s16.val[0];
    src += 16;
    q0x2s16 = vld2q_s16(src);
    q15s16 = q0x2s16.val[0];

    TRANSPOSE8X8(&q8s16, &q9s16, &q10s16, &q11s16,
                 &q12s16, &q13s16, &q14s16, &q15s16);

    // stage 3
    q6s16 = vdupq_n_s16(cospi_30_64 * 2);
    q0s16 = vqrdmulhq_s16(q8s16, q6s16);
    q6s16 = vdupq_n_s16(cospi_2_64 * 2);
    q7s16 = vqrdmulhq_s16(q8s16, q6s16);

    q15s16 = vdupq_n_s16(-cospi_26_64 * 2);
    q14s16 = vdupq_n_s16(cospi_6_64 * 2);
    q3s16 = vqrdmulhq_s16(q9s16, q15s16);
    q4s16 = vqrdmulhq_s16(q9s16, q14s16);

    // stage 4
    d0s16 = vget_low_s16(q0s16);
    d1s16 = vget_high_s16(q0s16);
    d6s16 = vget_low_s16(q3s16);
    d7s16 = vget_high_s16(q3s16);
    d8s16 = vget_low_s16(q4s16);
    d9s16 = vget_high_s16(q4s16);
    d14s16 = vget_low_s16(q7s16);
    d15s16 = vget_high_s16(q7s16);

    d30s16 = vdup_n_s16(cospi_8_64);
    d31s16 = vdup_n_s16(cospi_24_64);

    q12s32 = vmull_s16(d14s16, d31s16);
    q5s32 = vmull_s16(d15s16, d31s16);
    q2s32 = vmull_s16(d0s16, d31s16);
    q11s32 = vmull_s16(d1s16, d31s16);

    q12s32 = vmlsl_s16(q12s32, d0s16, d30s16);
    q5s32 = vmlsl_s16(q5s32, d1s16, d30s16);
    q2s32 = vmlal_s16(q2s32, d14s16, d30s16);
    q11s32 = vmlal_s16(q11s32, d15s16, d30s16);

    d2s16 = vqrshrn_n_s32(q12s32, 14);
    d3s16 = vqrshrn_n_s32(q5s32, 14);
    d12s16 = vqrshrn_n_s32(q2s32, 14);
    d13s16 = vqrshrn_n_s32(q11s32, 14);
    q1s16 = vcombine_s16(d2s16, d3s16);
    q6s16 = vcombine_s16(d12s16, d13s16);

    d30s16 = vdup_n_s16(-cospi_8_64);
    q10s32 = vmull_s16(d8s16, d30s16);
    q13s32 = vmull_s16(d9s16, d30s16);
    q8s32 = vmull_s16(d6s16, d30s16);
    q9s32 = vmull_s16(d7s16, d30s16);

    q10s32 = vmlsl_s16(q10s32, d6s16, d31s16);
    q13s32 = vmlsl_s16(q13s32, d7s16, d31s16);
    q8s32 = vmlal_s16(q8s32, d8s16, d31s16);
    q9s32 = vmlal_s16(q9s32, d9s16, d31s16);

    d4s16 = vqrshrn_n_s32(q10s32, 14);
    d5s16 = vqrshrn_n_s32(q13s32, 14);
    d10s16 = vqrshrn_n_s32(q8s32, 14);
    d11s16 = vqrshrn_n_s32(q9s32, 14);
    q2s16 = vcombine_s16(d4s16, d5s16);
    q5s16 = vcombine_s16(d10s16, d11s16);

    // stage 5
    q8s16  = vaddq_s16(q0s16, q3s16);
    q9s16  = vaddq_s16(q1s16, q2s16);
    q10s16 = vsubq_s16(q1s16, q2s16);
    q11s16 = vsubq_s16(q0s16, q3s16);
    q12s16 = vsubq_s16(q7s16, q4s16);
    q13s16 = vsubq_s16(q6s16, q5s16);
    q14s16 = vaddq_s16(q6s16, q5s16);
    q15s16 = vaddq_s16(q7s16, q4s16);

    // stage 6
    d20s16 = vget_low_s16(q10s16);
    d21s16 = vget_high_s16(q10s16);
    d22s16 = vget_low_s16(q11s16);
    d23s16 = vget_high_s16(q11s16);
    d24s16 = vget_low_s16(q12s16);
    d25s16 = vget_high_s16(q12s16);
    d26s16 = vget_low_s16(q13s16);
    d27s16 = vget_high_s16(q13s16);

    d14s16 = vdup_n_s16(cospi_16_64);
    q3s32 = vmull_s16(d26s16, d14s16);
    q4s32 = vmull_s16(d27s16, d14s16);
    q0s32 = vmull_s16(d20s16, d14s16);
    q1s32 = vmull_s16(d21s16, d14s16);

    q5s32 = vsubq_s32(q3s32, q0s32);
    q6s32 = vsubq_s32(q4s32, q1s32);
    q0s32 = vaddq_s32(q3s32, q0s32);
    q4s32 = vaddq_s32(q4s32, q1s32);

    d4s16 = vqrshrn_n_s32(q5s32, 14);
    d5s16 = vqrshrn_n_s32(q6s32, 14);
    d10s16 = vqrshrn_n_s32(q0s32, 14);
    d11s16 = vqrshrn_n_s32(q4s32, 14);
    q2s16 = vcombine_s16(d4s16, d5s16);
    q5s16 = vcombine_s16(d10s16, d11s16);

    q0s32 = vmull_s16(d22s16, d14s16);
    q1s32 = vmull_s16(d23s16, d14s16);
    q13s32 = vmull_s16(d24s16, d14s16);
    q6s32 = vmull_s16(d25s16, d14s16);

    q10s32 = vsubq_s32(q13s32, q0s32);
    q4s32 = vsubq_s32(q6s32, q1s32);
    q13s32 = vaddq_s32(q13s32, q0s32);
    q6s32 = vaddq_s32(q6s32, q1s32);

    d6s16 = vqrshrn_n_s32(q10s32, 14);
    d7s16 = vqrshrn_n_s32(q4s32, 14);
    d8s16 = vqrshrn_n_s32(q13s32, 14);
    d9s16 = vqrshrn_n_s32(q6s32, 14);
    q3s16 = vcombine_s16(d6s16, d7s16);
    q4s16 = vcombine_s16(d8s16, d9s16);

    // stage 7
    q0s16 = vld1q_s16(pass1Output);
    pass1Output += 8;
    q1s16 = vld1q_s16(pass1Output);
    pass1Output += 8;
    q12s16 = vaddq_s16(q0s16, q15s16);
    q13s16 = vaddq_s16(q1s16, q14s16);
    d24u64 = vreinterpret_u64_s16(vget_low_s16(q12s16));
    d25u64 = vreinterpret_u64_s16(vget_high_s16(q12s16));
    d26u64 = vreinterpret_u64_s16(vget_low_s16(q13s16));
    d27u64 = vreinterpret_u64_s16(vget_high_s16(q13s16));
    vst1_u64((uint64_t *)out, d24u64);
    out += 4;
    vst1_u64((uint64_t *)out, d25u64);
    out += 12;
    vst1_u64((uint64_t *)out, d26u64);
    out += 4;
    vst1_u64((uint64_t *)out, d27u64);
    out += 12;
    q14s16 = vsubq_s16(q1s16, q14s16);
    q15s16 = vsubq_s16(q0s16, q15s16);

    q10s16 = vld1q_s16(pass1Output);
    pass1Output += 8;
    q11s16 = vld1q_s16(pass1Output);
    pass1Output += 8;
    q12s16 = vaddq_s16(q10s16, q5s16);
    q13s16 = vaddq_s16(q11s16, q4s16);
    d24u64 = vreinterpret_u64_s16(vget_low_s16(q12s16));
    d25u64 = vreinterpret_u64_s16(vget_high_s16(q12s16));
    d26u64 = vreinterpret_u64_s16(vget_low_s16(q13s16));
    d27u64 = vreinterpret_u64_s16(vget_high_s16(q13s16));
    vst1_u64((uint64_t *)out, d24u64);
    out += 4;
    vst1_u64((uint64_t *)out, d25u64);
    out += 12;
    vst1_u64((uint64_t *)out, d26u64);
    out += 4;
    vst1_u64((uint64_t *)out, d27u64);
    out += 12;
    q4s16 = vsubq_s16(q11s16, q4s16);
    q5s16 = vsubq_s16(q10s16, q5s16);

    q0s16 = vld1q_s16(pass1Output);
    pass1Output += 8;
    q1s16 = vld1q_s16(pass1Output);
    pass1Output += 8;
    q12s16 = vaddq_s16(q0s16, q3s16);
    q13s16 = vaddq_s16(q1s16, q2s16);
    d24u64 = vreinterpret_u64_s16(vget_low_s16(q12s16));
    d25u64 = vreinterpret_u64_s16(vget_high_s16(q12s16));
    d26u64 = vreinterpret_u64_s16(vget_low_s16(q13s16));
    d27u64 = vreinterpret_u64_s16(vget_high_s16(q13s16));
    vst1_u64((uint64_t *)out, d24u64);
    out += 4;
    vst1_u64((uint64_t *)out, d25u64);
    out += 12;
    vst1_u64((uint64_t *)out, d26u64);
    out += 4;
    vst1_u64((uint64_t *)out, d27u64);
    out += 12;
    q2s16 = vsubq_s16(q1s16, q2s16);
    q3s16 = vsubq_s16(q0s16, q3s16);

    q10s16 = vld1q_s16(pass1Output);
    pass1Output += 8;
    q11s16 = vld1q_s16(pass1Output);
    q12s16 = vaddq_s16(q10s16, q9s16);
    q13s16 = vaddq_s16(q11s16, q8s16);
    d24u64 = vreinterpret_u64_s16(vget_low_s16(q12s16));
    d25u64 = vreinterpret_u64_s16(vget_high_s16(q12s16));
    d26u64 = vreinterpret_u64_s16(vget_low_s16(q13s16));
    d27u64 = vreinterpret_u64_s16(vget_high_s16(q13s16));
    vst1_u64((uint64_t *)out, d24u64);
    out += 4;
    vst1_u64((uint64_t *)out, d25u64);
    out += 12;
    vst1_u64((uint64_t *)out, d26u64);
    out += 4;
    vst1_u64((uint64_t *)out, d27u64);
    out += 12;
    q8s16 = vsubq_s16(q11s16, q8s16);
    q9s16 = vsubq_s16(q10s16, q9s16);

    d4u64  = vreinterpret_u64_s16(vget_low_s16(q2s16));
    d5u64  = vreinterpret_u64_s16(vget_high_s16(q2s16));
    d6u64  = vreinterpret_u64_s16(vget_low_s16(q3s16));
    d7u64  = vreinterpret_u64_s16(vget_high_s16(q3s16));
    d8u64  = vreinterpret_u64_s16(vget_low_s16(q4s16));
    d9u64  = vreinterpret_u64_s16(vget_high_s16(q4s16));
    d10u64 = vreinterpret_u64_s16(vget_low_s16(q5s16));
    d11u64 = vreinterpret_u64_s16(vget_high_s16(q5s16));
    d16u64 = vreinterpret_u64_s16(vget_low_s16(q8s16));
    d17u64 = vreinterpret_u64_s16(vget_high_s16(q8s16));
    d18u64 = vreinterpret_u64_s16(vget_low_s16(q9s16));
    d19u64 = vreinterpret_u64_s16(vget_high_s16(q9s16));
    d28u64 = vreinterpret_u64_s16(vget_low_s16(q14s16));
    d29u64 = vreinterpret_u64_s16(vget_high_s16(q14s16));
    d30u64 = vreinterpret_u64_s16(vget_low_s16(q15s16));
    d31u64 = vreinterpret_u64_s16(vget_high_s16(q15s16));

    vst1_u64((uint64_t *)out, d16u64);
    out += 4;
    vst1_u64((uint64_t *)out, d17u64);
    out += 12;
    vst1_u64((uint64_t *)out, d18u64);
    out += 4;
    vst1_u64((uint64_t *)out, d19u64);
    out += 12;
    vst1_u64((uint64_t *)out, d4u64);
    out += 4;
    vst1_u64((uint64_t *)out, d5u64);
    out += 12;
    vst1_u64((uint64_t *)out, d6u64);
    out += 4;
    vst1_u64((uint64_t *)out, d7u64);
    out += 12;
    vst1_u64((uint64_t *)out, d8u64);
    out += 4;
    vst1_u64((uint64_t *)out, d9u64);
    out += 12;
    vst1_u64((uint64_t *)out, d10u64);
    out += 4;
    vst1_u64((uint64_t *)out, d11u64);
    out += 12;
    vst1_u64((uint64_t *)out, d28u64);
    out += 4;
    vst1_u64((uint64_t *)out, d29u64);
    out += 12;
    vst1_u64((uint64_t *)out, d30u64);
    out += 4;
    vst1_u64((uint64_t *)out, d31u64);
    return;
}
