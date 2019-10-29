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

static INLINE void mbloop_filter_neon(
        uint8x8_t dblimit,   // mblimit
        uint8x8_t dlimit,    // limit
        uint8x8_t dthresh,   // thresh
        uint8x8_t d3u8,      // p2
        uint8x8_t d4u8,      // p2
        uint8x8_t d5u8,      // p1
        uint8x8_t d6u8,      // p0
        uint8x8_t d7u8,      // q0
        uint8x8_t d16u8,     // q1
        uint8x8_t d17u8,     // q2
        uint8x8_t d18u8,     // q3
        uint8x8_t *d0ru8,    // p1
        uint8x8_t *d1ru8,    // p1
        uint8x8_t *d2ru8,    // p0
        uint8x8_t *d3ru8,    // q0
        uint8x8_t *d4ru8,    // q1
        uint8x8_t *d5ru8) {  // q1
    uint32_t flat;
    uint8x8_t d0u8, d1u8, d2u8, d19u8, d20u8, d21u8, d22u8, d23u8, d24u8;
    uint8x8_t d25u8, d26u8, d27u8, d28u8, d29u8, d30u8, d31u8;
    int16x8_t q15s16;
    uint16x8_t q10u16, q14u16;
    int8x8_t d21s8, d24s8, d25s8, d26s8, d28s8, d29s8, d30s8;

    d19u8 = vabd_u8(d3u8, d4u8);
    d20u8 = vabd_u8(d4u8, d5u8);
    d21u8 = vabd_u8(d5u8, d6u8);
    d22u8 = vabd_u8(d16u8, d7u8);
    d23u8 = vabd_u8(d17u8, d16u8);
    d24u8 = vabd_u8(d18u8, d17u8);

    d19u8 = vmax_u8(d19u8, d20u8);
    d20u8 = vmax_u8(d21u8, d22u8);

    d25u8 = vabd_u8(d6u8, d4u8);

    d23u8 = vmax_u8(d23u8, d24u8);

    d26u8 = vabd_u8(d7u8, d17u8);

    d19u8 = vmax_u8(d19u8, d20u8);

    d24u8 = vabd_u8(d6u8, d7u8);
    d27u8 = vabd_u8(d3u8, d6u8);
    d28u8 = vabd_u8(d18u8, d7u8);

    d19u8 = vmax_u8(d19u8, d23u8);

    d23u8 = vabd_u8(d5u8, d16u8);
    d24u8 = vqadd_u8(d24u8, d24u8);


    d19u8 = vcge_u8(dlimit, d19u8);


    d25u8 = vmax_u8(d25u8, d26u8);
    d26u8 = vmax_u8(d27u8, d28u8);

    d23u8 = vshr_n_u8(d23u8, 1);

    d25u8 = vmax_u8(d25u8, d26u8);

    d24u8 = vqadd_u8(d24u8, d23u8);

    d20u8 = vmax_u8(d20u8, d25u8);

    d23u8 = vdup_n_u8(1);
    d24u8 = vcge_u8(dblimit, d24u8);

    d21u8 = vcgt_u8(d21u8, dthresh);

    d20u8 = vcge_u8(d23u8, d20u8);

    d19u8 = vand_u8(d19u8, d24u8);

    d23u8 = vcgt_u8(d22u8, dthresh);

    d20u8 = vand_u8(d20u8, d19u8);

    d22u8 = vdup_n_u8(0x80);

    d23u8 = vorr_u8(d21u8, d23u8);

    q10u16 = vcombine_u16(vreinterpret_u16_u8(d20u8),
                          vreinterpret_u16_u8(d21u8));

    d30u8 = vshrn_n_u16(q10u16, 4);
    flat = vget_lane_u32(vreinterpret_u32_u8(d30u8), 0);

    if (flat == 0xffffffff) {  // Check for all 1's, power_branch_only
        d27u8 = vdup_n_u8(3);
        d21u8 = vdup_n_u8(2);
        q14u16 = vaddl_u8(d6u8, d7u8);
        q14u16 = vmlal_u8(q14u16, d3u8, d27u8);
        q14u16 = vmlal_u8(q14u16, d4u8, d21u8);
        q14u16 = vaddw_u8(q14u16, d5u8);
        *d0ru8 = vqrshrn_n_u16(q14u16, 3);

        q14u16 = vsubw_u8(q14u16, d3u8);
        q14u16 = vsubw_u8(q14u16, d4u8);
        q14u16 = vaddw_u8(q14u16, d5u8);
        q14u16 = vaddw_u8(q14u16, d16u8);
        *d1ru8 = vqrshrn_n_u16(q14u16, 3);

        q14u16 = vsubw_u8(q14u16, d3u8);
        q14u16 = vsubw_u8(q14u16, d5u8);
        q14u16 = vaddw_u8(q14u16, d6u8);
        q14u16 = vaddw_u8(q14u16, d17u8);
        *d2ru8 = vqrshrn_n_u16(q14u16, 3);

        q14u16 = vsubw_u8(q14u16, d3u8);
        q14u16 = vsubw_u8(q14u16, d6u8);
        q14u16 = vaddw_u8(q14u16, d7u8);
        q14u16 = vaddw_u8(q14u16, d18u8);
        *d3ru8 = vqrshrn_n_u16(q14u16, 3);

        q14u16 = vsubw_u8(q14u16, d4u8);
        q14u16 = vsubw_u8(q14u16, d7u8);
        q14u16 = vaddw_u8(q14u16, d16u8);
        q14u16 = vaddw_u8(q14u16, d18u8);
        *d4ru8 = vqrshrn_n_u16(q14u16, 3);

        q14u16 = vsubw_u8(q14u16, d5u8);
        q14u16 = vsubw_u8(q14u16, d16u8);
        q14u16 = vaddw_u8(q14u16, d17u8);
        q14u16 = vaddw_u8(q14u16, d18u8);
        *d5ru8 = vqrshrn_n_u16(q14u16, 3);
    } else {
        d21u8 = veor_u8(d7u8,  d22u8);
        d24u8 = veor_u8(d6u8,  d22u8);
        d25u8 = veor_u8(d5u8,  d22u8);
        d26u8 = veor_u8(d16u8, d22u8);

        d27u8 = vdup_n_u8(3);

        d28s8 = vsub_s8(vreinterpret_s8_u8(d21u8), vreinterpret_s8_u8(d24u8));
        d29s8 = vqsub_s8(vreinterpret_s8_u8(d25u8), vreinterpret_s8_u8(d26u8));

        q15s16 = vmull_s8(d28s8, vreinterpret_s8_u8(d27u8));

        d29s8 = vand_s8(d29s8, vreinterpret_s8_u8(d23u8));

        q15s16 = vaddw_s8(q15s16, d29s8);

        d29u8 = vdup_n_u8(4);

        d28s8 = vqmovn_s16(q15s16);

        d28s8 = vand_s8(d28s8, vreinterpret_s8_u8(d19u8));

        d30s8 = vqadd_s8(d28s8, vreinterpret_s8_u8(d27u8));
        d29s8 = vqadd_s8(d28s8, vreinterpret_s8_u8(d29u8));
        d30s8 = vshr_n_s8(d30s8, 3);
        d29s8 = vshr_n_s8(d29s8, 3);

        d24s8 = vqadd_s8(vreinterpret_s8_u8(d24u8), d30s8);
        d21s8 = vqsub_s8(vreinterpret_s8_u8(d21u8), d29s8);

        d29s8 = vrshr_n_s8(d29s8, 1);
        d29s8 = vbic_s8(d29s8, vreinterpret_s8_u8(d23u8));

        d25s8 = vqadd_s8(vreinterpret_s8_u8(d25u8), d29s8);
        d26s8 = vqsub_s8(vreinterpret_s8_u8(d26u8), d29s8);

        if (flat == 0) {  // filter_branch_only
            *d0ru8 = d4u8;
            *d1ru8 = veor_u8(vreinterpret_u8_s8(d25s8), d22u8);
            *d2ru8 = veor_u8(vreinterpret_u8_s8(d24s8), d22u8);
            *d3ru8 = veor_u8(vreinterpret_u8_s8(d21s8), d22u8);
            *d4ru8 = veor_u8(vreinterpret_u8_s8(d26s8), d22u8);
            *d5ru8 = d17u8;
            return;
        }

        d21u8 = veor_u8(vreinterpret_u8_s8(d21s8), d22u8);
        d24u8 = veor_u8(vreinterpret_u8_s8(d24s8), d22u8);
        d25u8 = veor_u8(vreinterpret_u8_s8(d25s8), d22u8);
        d26u8 = veor_u8(vreinterpret_u8_s8(d26s8), d22u8);

        d23u8 = vdup_n_u8(2);
        q14u16 = vaddl_u8(d6u8, d7u8);
        q14u16 = vmlal_u8(q14u16, d3u8, d27u8);
        q14u16 = vmlal_u8(q14u16, d4u8, d23u8);

        d0u8 = vbsl_u8(d20u8, dblimit, d4u8);

        q14u16 = vaddw_u8(q14u16, d5u8);

        d1u8 = vbsl_u8(d20u8, dlimit, d25u8);

        d30u8 = vqrshrn_n_u16(q14u16, 3);

        q14u16 = vsubw_u8(q14u16, d3u8);
        q14u16 = vsubw_u8(q14u16, d4u8);
        q14u16 = vaddw_u8(q14u16, d5u8);
        q14u16 = vaddw_u8(q14u16, d16u8);

        d2u8 = vbsl_u8(d20u8, dthresh, d24u8);

        d31u8 = vqrshrn_n_u16(q14u16, 3);

        q14u16 = vsubw_u8(q14u16, d3u8);
        q14u16 = vsubw_u8(q14u16, d5u8);
        q14u16 = vaddw_u8(q14u16, d6u8);
        q14u16 = vaddw_u8(q14u16, d17u8);

        *d0ru8 = vbsl_u8(d20u8, d30u8, d0u8);

        d23u8 = vqrshrn_n_u16(q14u16, 3);

        q14u16 = vsubw_u8(q14u16, d3u8);
        q14u16 = vsubw_u8(q14u16, d6u8);
        q14u16 = vaddw_u8(q14u16, d7u8);

        *d1ru8 = vbsl_u8(d20u8, d31u8, d1u8);

        q14u16 = vaddw_u8(q14u16, d18u8);

        *d2ru8 = vbsl_u8(d20u8, d23u8, d2u8);

        d22u8 = vqrshrn_n_u16(q14u16, 3);

        q14u16 = vsubw_u8(q14u16, d4u8);
        q14u16 = vsubw_u8(q14u16, d7u8);
        q14u16 = vaddw_u8(q14u16, d16u8);

        d3u8 = vbsl_u8(d20u8, d3u8, d21u8);

        q14u16 = vaddw_u8(q14u16, d18u8);

        d4u8 = vbsl_u8(d20u8, d4u8, d26u8);

        d6u8 = vqrshrn_n_u16(q14u16, 3);

        q14u16 = vsubw_u8(q14u16, d5u8);
        q14u16 = vsubw_u8(q14u16, d16u8);
        q14u16 = vaddw_u8(q14u16, d17u8);
        q14u16 = vaddw_u8(q14u16, d18u8);

        d5u8 = vbsl_u8(d20u8, d5u8, d17u8);

        d7u8 = vqrshrn_n_u16(q14u16, 3);

        *d3ru8 = vbsl_u8(d20u8, d22u8, d3u8);
        *d4ru8 = vbsl_u8(d20u8, d6u8, d4u8);
        *d5ru8 = vbsl_u8(d20u8, d7u8, d5u8);
    }
    return;
}

void vpx_lpf_horizontal_8_neon(
        uint8_t *src,
        int pitch,
        const uint8_t *blimit,
        const uint8_t *limit,
        const uint8_t *thresh) {
    int i;
    uint8_t *s, *psrc;
    uint8x8_t dblimit, dlimit, dthresh;
    uint8x8_t d0u8, d1u8, d2u8, d3u8, d4u8, d5u8, d6u8, d7u8;
    uint8x8_t d16u8, d17u8, d18u8;

    dblimit = vld1_u8(blimit);
    dlimit = vld1_u8(limit);
    dthresh = vld1_u8(thresh);

    psrc = src - (pitch << 2);
    for (i = 0; i < 1; i++) {
        s = psrc + i * 8;

        d3u8  = vld1_u8(s);
        s += pitch;
        d4u8  = vld1_u8(s);
        s += pitch;
        d5u8  = vld1_u8(s);
        s += pitch;
        d6u8  = vld1_u8(s);
        s += pitch;
        d7u8  = vld1_u8(s);
        s += pitch;
        d16u8 = vld1_u8(s);
        s += pitch;
        d17u8 = vld1_u8(s);
        s += pitch;
        d18u8 = vld1_u8(s);

        mbloop_filter_neon(dblimit, dlimit, dthresh,
                           d3u8, d4u8, d5u8, d6u8, d7u8, d16u8, d17u8, d18u8,
                           &d0u8, &d1u8, &d2u8, &d3u8, &d4u8, &d5u8);

        s -= (pitch * 6);
        vst1_u8(s, d0u8);
        s += pitch;
        vst1_u8(s, d1u8);
        s += pitch;
        vst1_u8(s, d2u8);
        s += pitch;
        vst1_u8(s, d3u8);
        s += pitch;
        vst1_u8(s, d4u8);
        s += pitch;
        vst1_u8(s, d5u8);
    }
    return;
}

void vpx_lpf_vertical_8_neon(
        uint8_t *src,
        int pitch,
        const uint8_t *blimit,
        const uint8_t *limit,
        const uint8_t *thresh) {
    int i;
    uint8_t *s;
    uint8x8_t dblimit, dlimit, dthresh;
    uint8x8_t d0u8, d1u8, d2u8, d3u8, d4u8, d5u8, d6u8, d7u8;
    uint8x8_t d16u8, d17u8, d18u8;
    uint32x2x2_t d2tmp0, d2tmp1, d2tmp2, d2tmp3;
    uint16x4x2_t d2tmp4, d2tmp5, d2tmp6, d2tmp7;
    uint8x8x2_t d2tmp8, d2tmp9, d2tmp10, d2tmp11;
    uint8x8x4_t d4Result;
    uint8x8x2_t d2Result;

    dblimit = vld1_u8(blimit);
    dlimit = vld1_u8(limit);
    dthresh = vld1_u8(thresh);

    for (i = 0; i < 1; i++) {
        s = src + (i * (pitch << 3)) - 4;

        d3u8 = vld1_u8(s);
        s += pitch;
        d4u8 = vld1_u8(s);
        s += pitch;
        d5u8 = vld1_u8(s);
        s += pitch;
        d6u8 = vld1_u8(s);
        s += pitch;
        d7u8 = vld1_u8(s);
        s += pitch;
        d16u8 = vld1_u8(s);
        s += pitch;
        d17u8 = vld1_u8(s);
        s += pitch;
        d18u8 = vld1_u8(s);

        d2tmp0 = vtrn_u32(vreinterpret_u32_u8(d3u8),
                          vreinterpret_u32_u8(d7u8));
        d2tmp1 = vtrn_u32(vreinterpret_u32_u8(d4u8),
                          vreinterpret_u32_u8(d16u8));
        d2tmp2 = vtrn_u32(vreinterpret_u32_u8(d5u8),
                          vreinterpret_u32_u8(d17u8));
        d2tmp3 = vtrn_u32(vreinterpret_u32_u8(d6u8),
                          vreinterpret_u32_u8(d18u8));

        d2tmp4 = vtrn_u16(vreinterpret_u16_u32(d2tmp0.val[0]),
                          vreinterpret_u16_u32(d2tmp2.val[0]));
        d2tmp5 = vtrn_u16(vreinterpret_u16_u32(d2tmp1.val[0]),
                          vreinterpret_u16_u32(d2tmp3.val[0]));
        d2tmp6 = vtrn_u16(vreinterpret_u16_u32(d2tmp0.val[1]),
                          vreinterpret_u16_u32(d2tmp2.val[1]));
        d2tmp7 = vtrn_u16(vreinterpret_u16_u32(d2tmp1.val[1]),
                          vreinterpret_u16_u32(d2tmp3.val[1]));

        d2tmp8 = vtrn_u8(vreinterpret_u8_u16(d2tmp4.val[0]),
                         vreinterpret_u8_u16(d2tmp5.val[0]));
        d2tmp9 = vtrn_u8(vreinterpret_u8_u16(d2tmp4.val[1]),
                         vreinterpret_u8_u16(d2tmp5.val[1]));
        d2tmp10 = vtrn_u8(vreinterpret_u8_u16(d2tmp6.val[0]),
                          vreinterpret_u8_u16(d2tmp7.val[0]));
        d2tmp11 = vtrn_u8(vreinterpret_u8_u16(d2tmp6.val[1]),
                          vreinterpret_u8_u16(d2tmp7.val[1]));

        d3u8 = d2tmp8.val[0];
        d4u8 = d2tmp8.val[1];
        d5u8 = d2tmp9.val[0];
        d6u8 = d2tmp9.val[1];
        d7u8 = d2tmp10.val[0];
        d16u8 = d2tmp10.val[1];
        d17u8 = d2tmp11.val[0];
        d18u8 = d2tmp11.val[1];

        mbloop_filter_neon(dblimit, dlimit, dthresh,
                           d3u8, d4u8, d5u8, d6u8, d7u8, d16u8, d17u8, d18u8,
                           &d0u8, &d1u8, &d2u8, &d3u8, &d4u8, &d5u8);

        d4Result.val[0] = d0u8;
        d4Result.val[1] = d1u8;
        d4Result.val[2] = d2u8;
        d4Result.val[3] = d3u8;

        d2Result.val[0] = d4u8;
        d2Result.val[1] = d5u8;

        s = src - 3;
        vst4_lane_u8(s, d4Result, 0);
        s += pitch;
        vst4_lane_u8(s, d4Result, 1);
        s += pitch;
        vst4_lane_u8(s, d4Result, 2);
        s += pitch;
        vst4_lane_u8(s, d4Result, 3);
        s += pitch;
        vst4_lane_u8(s, d4Result, 4);
        s += pitch;
        vst4_lane_u8(s, d4Result, 5);
        s += pitch;
        vst4_lane_u8(s, d4Result, 6);
        s += pitch;
        vst4_lane_u8(s, d4Result, 7);

        s = src + 1;
        vst2_lane_u8(s, d2Result, 0);
        s += pitch;
        vst2_lane_u8(s, d2Result, 1);
        s += pitch;
        vst2_lane_u8(s, d2Result, 2);
        s += pitch;
        vst2_lane_u8(s, d2Result, 3);
        s += pitch;
        vst2_lane_u8(s, d2Result, 4);
        s += pitch;
        vst2_lane_u8(s, d2Result, 5);
        s += pitch;
        vst2_lane_u8(s, d2Result, 6);
        s += pitch;
        vst2_lane_u8(s, d2Result, 7);
    }
    return;
}
