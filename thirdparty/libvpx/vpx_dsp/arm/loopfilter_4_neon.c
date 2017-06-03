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

static INLINE void loop_filter_neon(
        uint8x8_t dblimit,    // flimit
        uint8x8_t dlimit,     // limit
        uint8x8_t dthresh,    // thresh
        uint8x8_t d3u8,       // p3
        uint8x8_t d4u8,       // p2
        uint8x8_t d5u8,       // p1
        uint8x8_t d6u8,       // p0
        uint8x8_t d7u8,       // q0
        uint8x8_t d16u8,      // q1
        uint8x8_t d17u8,      // q2
        uint8x8_t d18u8,      // q3
        uint8x8_t *d4ru8,     // p1
        uint8x8_t *d5ru8,     // p0
        uint8x8_t *d6ru8,     // q0
        uint8x8_t *d7ru8) {   // q1
    uint8x8_t d19u8, d20u8, d21u8, d22u8, d23u8, d27u8, d28u8;
    int16x8_t q12s16;
    int8x8_t d19s8, d20s8, d21s8, d26s8, d27s8, d28s8;

    d19u8 = vabd_u8(d3u8, d4u8);
    d20u8 = vabd_u8(d4u8, d5u8);
    d21u8 = vabd_u8(d5u8, d6u8);
    d22u8 = vabd_u8(d16u8, d7u8);
    d3u8  = vabd_u8(d17u8, d16u8);
    d4u8  = vabd_u8(d18u8, d17u8);

    d19u8 = vmax_u8(d19u8, d20u8);
    d20u8 = vmax_u8(d21u8, d22u8);
    d3u8  = vmax_u8(d3u8,  d4u8);
    d23u8 = vmax_u8(d19u8, d20u8);

    d17u8 = vabd_u8(d6u8, d7u8);

    d21u8 = vcgt_u8(d21u8, dthresh);
    d22u8 = vcgt_u8(d22u8, dthresh);
    d23u8 = vmax_u8(d23u8, d3u8);

    d28u8 = vabd_u8(d5u8, d16u8);
    d17u8 = vqadd_u8(d17u8, d17u8);

    d23u8 = vcge_u8(dlimit, d23u8);

    d18u8 = vdup_n_u8(0x80);
    d5u8  = veor_u8(d5u8,  d18u8);
    d6u8  = veor_u8(d6u8,  d18u8);
    d7u8  = veor_u8(d7u8,  d18u8);
    d16u8 = veor_u8(d16u8, d18u8);

    d28u8 = vshr_n_u8(d28u8, 1);
    d17u8 = vqadd_u8(d17u8, d28u8);

    d19u8 = vdup_n_u8(3);

    d28s8 = vsub_s8(vreinterpret_s8_u8(d7u8),
                    vreinterpret_s8_u8(d6u8));

    d17u8 = vcge_u8(dblimit, d17u8);

    d27s8 = vqsub_s8(vreinterpret_s8_u8(d5u8),
                     vreinterpret_s8_u8(d16u8));

    d22u8 = vorr_u8(d21u8, d22u8);

    q12s16 = vmull_s8(d28s8, vreinterpret_s8_u8(d19u8));

    d27u8 = vand_u8(vreinterpret_u8_s8(d27s8), d22u8);
    d23u8 = vand_u8(d23u8, d17u8);

    q12s16 = vaddw_s8(q12s16, vreinterpret_s8_u8(d27u8));

    d17u8 = vdup_n_u8(4);

    d27s8 = vqmovn_s16(q12s16);
    d27u8 = vand_u8(vreinterpret_u8_s8(d27s8), d23u8);
    d27s8 = vreinterpret_s8_u8(d27u8);

    d28s8 = vqadd_s8(d27s8, vreinterpret_s8_u8(d19u8));
    d27s8 = vqadd_s8(d27s8, vreinterpret_s8_u8(d17u8));
    d28s8 = vshr_n_s8(d28s8, 3);
    d27s8 = vshr_n_s8(d27s8, 3);

    d19s8 = vqadd_s8(vreinterpret_s8_u8(d6u8), d28s8);
    d26s8 = vqsub_s8(vreinterpret_s8_u8(d7u8), d27s8);

    d27s8 = vrshr_n_s8(d27s8, 1);
    d27s8 = vbic_s8(d27s8, vreinterpret_s8_u8(d22u8));

    d21s8 = vqadd_s8(vreinterpret_s8_u8(d5u8), d27s8);
    d20s8 = vqsub_s8(vreinterpret_s8_u8(d16u8), d27s8);

    *d4ru8 = veor_u8(vreinterpret_u8_s8(d21s8), d18u8);
    *d5ru8 = veor_u8(vreinterpret_u8_s8(d19s8), d18u8);
    *d6ru8 = veor_u8(vreinterpret_u8_s8(d26s8), d18u8);
    *d7ru8 = veor_u8(vreinterpret_u8_s8(d20s8), d18u8);
    return;
}

void vpx_lpf_horizontal_4_neon(
        uint8_t *src,
        int pitch,
        const uint8_t *blimit,
        const uint8_t *limit,
        const uint8_t *thresh) {
    int i;
    uint8_t *s, *psrc;
    uint8x8_t dblimit, dlimit, dthresh;
    uint8x8_t d3u8, d4u8, d5u8, d6u8, d7u8, d16u8, d17u8, d18u8;

    dblimit = vld1_u8(blimit);
    dlimit = vld1_u8(limit);
    dthresh = vld1_u8(thresh);

    psrc = src - (pitch << 2);
    for (i = 0; i < 1; i++) {
        s = psrc + i * 8;

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

        loop_filter_neon(dblimit, dlimit, dthresh,
                         d3u8, d4u8, d5u8, d6u8, d7u8, d16u8, d17u8, d18u8,
                         &d4u8, &d5u8, &d6u8, &d7u8);

        s -= (pitch * 5);
        vst1_u8(s, d4u8);
        s += pitch;
        vst1_u8(s, d5u8);
        s += pitch;
        vst1_u8(s, d6u8);
        s += pitch;
        vst1_u8(s, d7u8);
    }
    return;
}

void vpx_lpf_vertical_4_neon(
        uint8_t *src,
        int pitch,
        const uint8_t *blimit,
        const uint8_t *limit,
        const uint8_t *thresh) {
    int i, pitch8;
    uint8_t *s;
    uint8x8_t dblimit, dlimit, dthresh;
    uint8x8_t d3u8, d4u8, d5u8, d6u8, d7u8, d16u8, d17u8, d18u8;
    uint32x2x2_t d2tmp0, d2tmp1, d2tmp2, d2tmp3;
    uint16x4x2_t d2tmp4, d2tmp5, d2tmp6, d2tmp7;
    uint8x8x2_t d2tmp8, d2tmp9, d2tmp10, d2tmp11;
    uint8x8x4_t d4Result;

    dblimit = vld1_u8(blimit);
    dlimit = vld1_u8(limit);
    dthresh = vld1_u8(thresh);

    pitch8 = pitch * 8;
    for (i = 0; i < 1; i++, src += pitch8) {
        s = src - (i + 1) * 4;

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

        loop_filter_neon(dblimit, dlimit, dthresh,
                         d3u8, d4u8, d5u8, d6u8, d7u8, d16u8, d17u8, d18u8,
                         &d4u8, &d5u8, &d6u8, &d7u8);

        d4Result.val[0] = d4u8;
        d4Result.val[1] = d5u8;
        d4Result.val[2] = d6u8;
        d4Result.val[3] = d7u8;

        src -= 2;
        vst4_lane_u8(src, d4Result, 0);
        src += pitch;
        vst4_lane_u8(src, d4Result, 1);
        src += pitch;
        vst4_lane_u8(src, d4Result, 2);
        src += pitch;
        vst4_lane_u8(src, d4Result, 3);
        src += pitch;
        vst4_lane_u8(src, d4Result, 4);
        src += pitch;
        vst4_lane_u8(src, d4Result, 5);
        src += pitch;
        vst4_lane_u8(src, d4Result, 6);
        src += pitch;
        vst4_lane_u8(src, d4Result, 7);
    }
    return;
}
