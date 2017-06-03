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

void vpx_idct4x4_16_add_neon(
        int16_t *input,
        uint8_t *dest,
        int dest_stride) {
    uint8x8_t d26u8, d27u8;
    uint32x2_t d26u32, d27u32;
    uint16x8_t q8u16, q9u16;
    int16x4_t d16s16, d17s16, d18s16, d19s16, d20s16, d21s16;
    int16x4_t d22s16, d23s16, d24s16, d26s16, d27s16, d28s16, d29s16;
    int16x8_t q8s16, q9s16, q13s16, q14s16;
    int32x4_t q1s32, q13s32, q14s32, q15s32;
    int16x4x2_t d0x2s16, d1x2s16;
    int32x4x2_t q0x2s32;
    uint8_t *d;
    int16_t cospi_8_64 = 15137;
    int16_t cospi_16_64 = 11585;
    int16_t cospi_24_64 = 6270;

    d26u32 = d27u32 = vdup_n_u32(0);

    q8s16 = vld1q_s16(input);
    q9s16 = vld1q_s16(input + 8);

    d16s16 = vget_low_s16(q8s16);
    d17s16 = vget_high_s16(q8s16);
    d18s16 = vget_low_s16(q9s16);
    d19s16 = vget_high_s16(q9s16);

    d0x2s16 = vtrn_s16(d16s16, d17s16);
    d1x2s16 = vtrn_s16(d18s16, d19s16);
    q8s16 = vcombine_s16(d0x2s16.val[0], d0x2s16.val[1]);
    q9s16 = vcombine_s16(d1x2s16.val[0], d1x2s16.val[1]);

    d20s16 = vdup_n_s16(cospi_8_64);
    d21s16 = vdup_n_s16(cospi_16_64);

    q0x2s32 = vtrnq_s32(vreinterpretq_s32_s16(q8s16),
                        vreinterpretq_s32_s16(q9s16));
    d16s16 = vget_low_s16(vreinterpretq_s16_s32(q0x2s32.val[0]));
    d17s16 = vget_high_s16(vreinterpretq_s16_s32(q0x2s32.val[0]));
    d18s16 = vget_low_s16(vreinterpretq_s16_s32(q0x2s32.val[1]));
    d19s16 = vget_high_s16(vreinterpretq_s16_s32(q0x2s32.val[1]));

    d22s16 = vdup_n_s16(cospi_24_64);

    // stage 1
    d23s16 = vadd_s16(d16s16, d18s16);
    d24s16 = vsub_s16(d16s16, d18s16);

    q15s32 = vmull_s16(d17s16, d22s16);
    q1s32  = vmull_s16(d17s16, d20s16);
    q13s32 = vmull_s16(d23s16, d21s16);
    q14s32 = vmull_s16(d24s16, d21s16);

    q15s32 = vmlsl_s16(q15s32, d19s16, d20s16);
    q1s32  = vmlal_s16(q1s32,  d19s16, d22s16);

    d26s16 = vqrshrn_n_s32(q13s32, 14);
    d27s16 = vqrshrn_n_s32(q14s32, 14);
    d29s16 = vqrshrn_n_s32(q15s32, 14);
    d28s16 = vqrshrn_n_s32(q1s32,  14);
    q13s16 = vcombine_s16(d26s16, d27s16);
    q14s16 = vcombine_s16(d28s16, d29s16);

    // stage 2
    q8s16 = vaddq_s16(q13s16, q14s16);
    q9s16 = vsubq_s16(q13s16, q14s16);

    d16s16 = vget_low_s16(q8s16);
    d17s16 = vget_high_s16(q8s16);
    d18s16 = vget_high_s16(q9s16);  // vswp d18 d19
    d19s16 = vget_low_s16(q9s16);

    d0x2s16 = vtrn_s16(d16s16, d17s16);
    d1x2s16 = vtrn_s16(d18s16, d19s16);
    q8s16 = vcombine_s16(d0x2s16.val[0], d0x2s16.val[1]);
    q9s16 = vcombine_s16(d1x2s16.val[0], d1x2s16.val[1]);

    q0x2s32 = vtrnq_s32(vreinterpretq_s32_s16(q8s16),
                        vreinterpretq_s32_s16(q9s16));
    d16s16 = vget_low_s16(vreinterpretq_s16_s32(q0x2s32.val[0]));
    d17s16 = vget_high_s16(vreinterpretq_s16_s32(q0x2s32.val[0]));
    d18s16 = vget_low_s16(vreinterpretq_s16_s32(q0x2s32.val[1]));
    d19s16 = vget_high_s16(vreinterpretq_s16_s32(q0x2s32.val[1]));

    // do the transform on columns
    // stage 1
    d23s16 = vadd_s16(d16s16, d18s16);
    d24s16 = vsub_s16(d16s16, d18s16);

    q15s32 = vmull_s16(d17s16, d22s16);
    q1s32  = vmull_s16(d17s16, d20s16);
    q13s32 = vmull_s16(d23s16, d21s16);
    q14s32 = vmull_s16(d24s16, d21s16);

    q15s32 = vmlsl_s16(q15s32, d19s16, d20s16);
    q1s32  = vmlal_s16(q1s32,  d19s16, d22s16);

    d26s16 = vqrshrn_n_s32(q13s32, 14);
    d27s16 = vqrshrn_n_s32(q14s32, 14);
    d29s16 = vqrshrn_n_s32(q15s32, 14);
    d28s16 = vqrshrn_n_s32(q1s32,  14);
    q13s16 = vcombine_s16(d26s16, d27s16);
    q14s16 = vcombine_s16(d28s16, d29s16);

    // stage 2
    q8s16 = vaddq_s16(q13s16, q14s16);
    q9s16 = vsubq_s16(q13s16, q14s16);

    q8s16 = vrshrq_n_s16(q8s16, 4);
    q9s16 = vrshrq_n_s16(q9s16, 4);

    d = dest;
    d26u32 = vld1_lane_u32((const uint32_t *)d, d26u32, 0);
    d += dest_stride;
    d26u32 = vld1_lane_u32((const uint32_t *)d, d26u32, 1);
    d += dest_stride;
    d27u32 = vld1_lane_u32((const uint32_t *)d, d27u32, 1);
    d += dest_stride;
    d27u32 = vld1_lane_u32((const uint32_t *)d, d27u32, 0);

    q8u16 = vaddw_u8(vreinterpretq_u16_s16(q8s16),
                     vreinterpret_u8_u32(d26u32));
    q9u16 = vaddw_u8(vreinterpretq_u16_s16(q9s16),
                     vreinterpret_u8_u32(d27u32));

    d26u8 = vqmovun_s16(vreinterpretq_s16_u16(q8u16));
    d27u8 = vqmovun_s16(vreinterpretq_s16_u16(q9u16));

    d = dest;
    vst1_lane_u32((uint32_t *)d, vreinterpret_u32_u8(d26u8), 0);
    d += dest_stride;
    vst1_lane_u32((uint32_t *)d, vreinterpret_u32_u8(d26u8), 1);
    d += dest_stride;
    vst1_lane_u32((uint32_t *)d, vreinterpret_u32_u8(d27u8), 1);
    d += dest_stride;
    vst1_lane_u32((uint32_t *)d, vreinterpret_u32_u8(d27u8), 0);
    return;
}
