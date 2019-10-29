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
#include "./vpx_config.h"
#include "vpx/vpx_integer.h"

static INLINE void loop_filter_neon_16(
        uint8x16_t qblimit,  // blimit
        uint8x16_t qlimit,   // limit
        uint8x16_t qthresh,  // thresh
        uint8x16_t q3,       // p3
        uint8x16_t q4,       // p2
        uint8x16_t q5,       // p1
        uint8x16_t q6,       // p0
        uint8x16_t q7,       // q0
        uint8x16_t q8,       // q1
        uint8x16_t q9,       // q2
        uint8x16_t q10,      // q3
        uint8x16_t *q5r,     // p1
        uint8x16_t *q6r,     // p0
        uint8x16_t *q7r,     // q0
        uint8x16_t *q8r) {   // q1
    uint8x16_t q1u8, q2u8, q11u8, q12u8, q13u8, q14u8, q15u8;
    int16x8_t q2s16, q11s16;
    uint16x8_t q4u16;
    int8x16_t q0s8, q1s8, q2s8, q11s8, q12s8, q13s8;
    int8x8_t d2s8, d3s8;

    q11u8 = vabdq_u8(q3, q4);
    q12u8 = vabdq_u8(q4, q5);
    q13u8 = vabdq_u8(q5, q6);
    q14u8 = vabdq_u8(q8, q7);
    q3 = vabdq_u8(q9, q8);
    q4 = vabdq_u8(q10, q9);

    q11u8 = vmaxq_u8(q11u8, q12u8);
    q12u8 = vmaxq_u8(q13u8, q14u8);
    q3 = vmaxq_u8(q3, q4);
    q15u8 = vmaxq_u8(q11u8, q12u8);

    q9 = vabdq_u8(q6, q7);

    // vp8_hevmask
    q13u8 = vcgtq_u8(q13u8, qthresh);
    q14u8 = vcgtq_u8(q14u8, qthresh);
    q15u8 = vmaxq_u8(q15u8, q3);

    q2u8 = vabdq_u8(q5, q8);
    q9 = vqaddq_u8(q9, q9);

    q15u8 = vcgeq_u8(qlimit, q15u8);

    // vp8_filter() function
    // convert to signed
    q10 = vdupq_n_u8(0x80);
    q8 = veorq_u8(q8, q10);
    q7 = veorq_u8(q7, q10);
    q6 = veorq_u8(q6, q10);
    q5 = veorq_u8(q5, q10);

    q2u8 = vshrq_n_u8(q2u8, 1);
    q9 = vqaddq_u8(q9, q2u8);

    q2s16 = vsubl_s8(vget_low_s8(vreinterpretq_s8_u8(q7)),
                     vget_low_s8(vreinterpretq_s8_u8(q6)));
    q11s16 = vsubl_s8(vget_high_s8(vreinterpretq_s8_u8(q7)),
                      vget_high_s8(vreinterpretq_s8_u8(q6)));

    q9 = vcgeq_u8(qblimit, q9);

    q1s8 = vqsubq_s8(vreinterpretq_s8_u8(q5),
                    vreinterpretq_s8_u8(q8));

    q14u8 = vorrq_u8(q13u8, q14u8);

    q4u16 = vdupq_n_u16(3);
    q2s16 = vmulq_s16(q2s16, vreinterpretq_s16_u16(q4u16));
    q11s16 = vmulq_s16(q11s16, vreinterpretq_s16_u16(q4u16));

    q1u8 = vandq_u8(vreinterpretq_u8_s8(q1s8), q14u8);
    q15u8 = vandq_u8(q15u8, q9);

    q1s8 = vreinterpretq_s8_u8(q1u8);
    q2s16 = vaddw_s8(q2s16, vget_low_s8(q1s8));
    q11s16 = vaddw_s8(q11s16, vget_high_s8(q1s8));

    q4 = vdupq_n_u8(3);
    q9 = vdupq_n_u8(4);
    // vp8_filter = clamp(vp8_filter + 3 * ( qs0 - ps0))
    d2s8 = vqmovn_s16(q2s16);
    d3s8 = vqmovn_s16(q11s16);
    q1s8 = vcombine_s8(d2s8, d3s8);
    q1u8 = vandq_u8(vreinterpretq_u8_s8(q1s8), q15u8);
    q1s8 = vreinterpretq_s8_u8(q1u8);

    q2s8 = vqaddq_s8(q1s8, vreinterpretq_s8_u8(q4));
    q1s8 = vqaddq_s8(q1s8, vreinterpretq_s8_u8(q9));
    q2s8 = vshrq_n_s8(q2s8, 3);
    q1s8 = vshrq_n_s8(q1s8, 3);

    q11s8 = vqaddq_s8(vreinterpretq_s8_u8(q6), q2s8);
    q0s8 = vqsubq_s8(vreinterpretq_s8_u8(q7), q1s8);

    q1s8 = vrshrq_n_s8(q1s8, 1);
    q1s8 = vbicq_s8(q1s8, vreinterpretq_s8_u8(q14u8));

    q13s8 = vqaddq_s8(vreinterpretq_s8_u8(q5), q1s8);
    q12s8 = vqsubq_s8(vreinterpretq_s8_u8(q8), q1s8);

    *q8r = veorq_u8(vreinterpretq_u8_s8(q12s8), q10);
    *q7r = veorq_u8(vreinterpretq_u8_s8(q0s8),  q10);
    *q6r = veorq_u8(vreinterpretq_u8_s8(q11s8), q10);
    *q5r = veorq_u8(vreinterpretq_u8_s8(q13s8), q10);
    return;
}

void vpx_lpf_horizontal_4_dual_neon(uint8_t *s, int p /* pitch */,
                                    const uint8_t *blimit0,
                                    const uint8_t *limit0,
                                    const uint8_t *thresh0,
                                    const uint8_t *blimit1,
                                    const uint8_t *limit1,
                                    const uint8_t *thresh1) {
    uint8x8_t dblimit0, dlimit0, dthresh0, dblimit1, dlimit1, dthresh1;
    uint8x16_t qblimit, qlimit, qthresh;
    uint8x16_t q3u8, q4u8, q5u8, q6u8, q7u8, q8u8, q9u8, q10u8;

    dblimit0 = vld1_u8(blimit0);
    dlimit0 = vld1_u8(limit0);
    dthresh0 = vld1_u8(thresh0);
    dblimit1 = vld1_u8(blimit1);
    dlimit1 = vld1_u8(limit1);
    dthresh1 = vld1_u8(thresh1);
    qblimit = vcombine_u8(dblimit0, dblimit1);
    qlimit = vcombine_u8(dlimit0, dlimit1);
    qthresh = vcombine_u8(dthresh0, dthresh1);

    s -= (p << 2);

    q3u8 = vld1q_u8(s);
    s += p;
    q4u8 = vld1q_u8(s);
    s += p;
    q5u8 = vld1q_u8(s);
    s += p;
    q6u8 = vld1q_u8(s);
    s += p;
    q7u8 = vld1q_u8(s);
    s += p;
    q8u8 = vld1q_u8(s);
    s += p;
    q9u8 = vld1q_u8(s);
    s += p;
    q10u8 = vld1q_u8(s);

    loop_filter_neon_16(qblimit, qlimit, qthresh,
                        q3u8, q4u8, q5u8, q6u8, q7u8, q8u8, q9u8, q10u8,
                        &q5u8, &q6u8, &q7u8, &q8u8);

    s -= (p * 5);
    vst1q_u8(s, q5u8);
    s += p;
    vst1q_u8(s, q6u8);
    s += p;
    vst1q_u8(s, q7u8);
    s += p;
    vst1q_u8(s, q8u8);
    return;
}
