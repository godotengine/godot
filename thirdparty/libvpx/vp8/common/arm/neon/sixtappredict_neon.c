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
#include "vpx_ports/mem.h"

static const int8_t vp8_sub_pel_filters[8][8] = {
    {0,  0,  128,   0,   0, 0, 0, 0},  /* note that 1/8 pel positionyys are */
    {0, -6,  123,  12,  -1, 0, 0, 0},  /*    just as per alpha -0.5 bicubic */
    {2, -11, 108,  36,  -8, 1, 0, 0},  /* New 1/4 pel 6 tap filter */
    {0, -9,   93,  50,  -6, 0, 0, 0},
    {3, -16,  77,  77, -16, 3, 0, 0},  /* New 1/2 pel 6 tap filter */
    {0, -6,   50,  93,  -9, 0, 0, 0},
    {1, -8,   36, 108, -11, 2, 0, 0},  /* New 1/4 pel 6 tap filter */
    {0, -1,   12, 123,  -6, 0, 0, 0},
};

void vp8_sixtap_predict8x4_neon(
        unsigned char *src_ptr,
        int src_pixels_per_line,
        int xoffset,
        int yoffset,
        unsigned char *dst_ptr,
        int dst_pitch) {
    unsigned char *src;
    uint8x8_t d0u8, d1u8, d2u8, d3u8, d4u8, d5u8, d6u8, d7u8, d8u8, d9u8;
    uint8x8_t d22u8, d23u8, d24u8, d25u8, d26u8;
    uint8x8_t d27u8, d28u8, d29u8, d30u8, d31u8;
    int8x8_t dtmps8, d0s8, d1s8, d2s8, d3s8, d4s8, d5s8;
    uint16x8_t q3u16, q4u16, q5u16, q6u16, q7u16;
    uint16x8_t q8u16, q9u16, q10u16, q11u16, q12u16;
    int16x8_t q3s16, q4s16, q5s16, q6s16, q7s16;
    int16x8_t q8s16, q9s16, q10s16, q11s16, q12s16;
    uint8x16_t q3u8, q4u8, q5u8, q6u8, q7u8;

    if (xoffset == 0) {  // secondpass_filter8x4_only
        // load second_pass filter
        dtmps8 = vld1_s8(vp8_sub_pel_filters[yoffset]);
        d0s8 = vdup_lane_s8(dtmps8, 0);
        d1s8 = vdup_lane_s8(dtmps8, 1);
        d2s8 = vdup_lane_s8(dtmps8, 2);
        d3s8 = vdup_lane_s8(dtmps8, 3);
        d4s8 = vdup_lane_s8(dtmps8, 4);
        d5s8 = vdup_lane_s8(dtmps8, 5);
        d0u8 = vreinterpret_u8_s8(vabs_s8(d0s8));
        d1u8 = vreinterpret_u8_s8(vabs_s8(d1s8));
        d2u8 = vreinterpret_u8_s8(vabs_s8(d2s8));
        d3u8 = vreinterpret_u8_s8(vabs_s8(d3s8));
        d4u8 = vreinterpret_u8_s8(vabs_s8(d4s8));
        d5u8 = vreinterpret_u8_s8(vabs_s8(d5s8));

        // load src data
        src = src_ptr - src_pixels_per_line * 2;
        d22u8 = vld1_u8(src);
        src += src_pixels_per_line;
        d23u8 = vld1_u8(src);
        src += src_pixels_per_line;
        d24u8 = vld1_u8(src);
        src += src_pixels_per_line;
        d25u8 = vld1_u8(src);
        src += src_pixels_per_line;
        d26u8 = vld1_u8(src);
        src += src_pixels_per_line;
        d27u8 = vld1_u8(src);
        src += src_pixels_per_line;
        d28u8 = vld1_u8(src);
        src += src_pixels_per_line;
        d29u8 = vld1_u8(src);
        src += src_pixels_per_line;
        d30u8 = vld1_u8(src);

        q3u16 = vmull_u8(d22u8, d0u8);
        q4u16 = vmull_u8(d23u8, d0u8);
        q5u16 = vmull_u8(d24u8, d0u8);
        q6u16 = vmull_u8(d25u8, d0u8);

        q3u16 = vmlsl_u8(q3u16, d23u8, d1u8);
        q4u16 = vmlsl_u8(q4u16, d24u8, d1u8);
        q5u16 = vmlsl_u8(q5u16, d25u8, d1u8);
        q6u16 = vmlsl_u8(q6u16, d26u8, d1u8);

        q3u16 = vmlsl_u8(q3u16, d26u8, d4u8);
        q4u16 = vmlsl_u8(q4u16, d27u8, d4u8);
        q5u16 = vmlsl_u8(q5u16, d28u8, d4u8);
        q6u16 = vmlsl_u8(q6u16, d29u8, d4u8);

        q3u16 = vmlal_u8(q3u16, d24u8, d2u8);
        q4u16 = vmlal_u8(q4u16, d25u8, d2u8);
        q5u16 = vmlal_u8(q5u16, d26u8, d2u8);
        q6u16 = vmlal_u8(q6u16, d27u8, d2u8);

        q3u16 = vmlal_u8(q3u16, d27u8, d5u8);
        q4u16 = vmlal_u8(q4u16, d28u8, d5u8);
        q5u16 = vmlal_u8(q5u16, d29u8, d5u8);
        q6u16 = vmlal_u8(q6u16, d30u8, d5u8);

        q7u16 = vmull_u8(d25u8, d3u8);
        q8u16 = vmull_u8(d26u8, d3u8);
        q9u16 = vmull_u8(d27u8, d3u8);
        q10u16 = vmull_u8(d28u8, d3u8);

        q3s16 = vreinterpretq_s16_u16(q3u16);
        q4s16 = vreinterpretq_s16_u16(q4u16);
        q5s16 = vreinterpretq_s16_u16(q5u16);
        q6s16 = vreinterpretq_s16_u16(q6u16);
        q7s16 = vreinterpretq_s16_u16(q7u16);
        q8s16 = vreinterpretq_s16_u16(q8u16);
        q9s16 = vreinterpretq_s16_u16(q9u16);
        q10s16 = vreinterpretq_s16_u16(q10u16);

        q7s16 = vqaddq_s16(q7s16, q3s16);
        q8s16 = vqaddq_s16(q8s16, q4s16);
        q9s16 = vqaddq_s16(q9s16, q5s16);
        q10s16 = vqaddq_s16(q10s16, q6s16);

        d6u8 = vqrshrun_n_s16(q7s16, 7);
        d7u8 = vqrshrun_n_s16(q8s16, 7);
        d8u8 = vqrshrun_n_s16(q9s16, 7);
        d9u8 = vqrshrun_n_s16(q10s16, 7);

        vst1_u8(dst_ptr, d6u8);
        dst_ptr += dst_pitch;
        vst1_u8(dst_ptr, d7u8);
        dst_ptr += dst_pitch;
        vst1_u8(dst_ptr, d8u8);
        dst_ptr += dst_pitch;
        vst1_u8(dst_ptr, d9u8);
        return;
    }

    // load first_pass filter
    dtmps8 = vld1_s8(vp8_sub_pel_filters[xoffset]);
    d0s8 = vdup_lane_s8(dtmps8, 0);
    d1s8 = vdup_lane_s8(dtmps8, 1);
    d2s8 = vdup_lane_s8(dtmps8, 2);
    d3s8 = vdup_lane_s8(dtmps8, 3);
    d4s8 = vdup_lane_s8(dtmps8, 4);
    d5s8 = vdup_lane_s8(dtmps8, 5);
    d0u8 = vreinterpret_u8_s8(vabs_s8(d0s8));
    d1u8 = vreinterpret_u8_s8(vabs_s8(d1s8));
    d2u8 = vreinterpret_u8_s8(vabs_s8(d2s8));
    d3u8 = vreinterpret_u8_s8(vabs_s8(d3s8));
    d4u8 = vreinterpret_u8_s8(vabs_s8(d4s8));
    d5u8 = vreinterpret_u8_s8(vabs_s8(d5s8));

    // First pass: output_height lines x output_width columns (9x4)
    if (yoffset == 0)  // firstpass_filter4x4_only
        src = src_ptr - 2;
    else
        src = src_ptr - 2 - (src_pixels_per_line * 2);
    q3u8 = vld1q_u8(src);
    src += src_pixels_per_line;
    q4u8 = vld1q_u8(src);
    src += src_pixels_per_line;
    q5u8 = vld1q_u8(src);
    src += src_pixels_per_line;
    q6u8 = vld1q_u8(src);

    q7u16  = vmull_u8(vget_low_u8(q3u8), d0u8);
    q8u16  = vmull_u8(vget_low_u8(q4u8), d0u8);
    q9u16  = vmull_u8(vget_low_u8(q5u8), d0u8);
    q10u16 = vmull_u8(vget_low_u8(q6u8), d0u8);

    d28u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 1);
    d29u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 1);
    d30u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 1);
    d31u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 1);

    q7u16  = vmlsl_u8(q7u16, d28u8, d1u8);
    q8u16  = vmlsl_u8(q8u16, d29u8, d1u8);
    q9u16  = vmlsl_u8(q9u16, d30u8, d1u8);
    q10u16 = vmlsl_u8(q10u16, d31u8, d1u8);

    d28u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 4);
    d29u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 4);
    d30u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 4);
    d31u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 4);

    q7u16  = vmlsl_u8(q7u16, d28u8, d4u8);
    q8u16  = vmlsl_u8(q8u16, d29u8, d4u8);
    q9u16  = vmlsl_u8(q9u16, d30u8, d4u8);
    q10u16 = vmlsl_u8(q10u16, d31u8, d4u8);

    d28u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 2);
    d29u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 2);
    d30u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 2);
    d31u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 2);

    q7u16  = vmlal_u8(q7u16, d28u8, d2u8);
    q8u16  = vmlal_u8(q8u16, d29u8, d2u8);
    q9u16  = vmlal_u8(q9u16, d30u8, d2u8);
    q10u16 = vmlal_u8(q10u16, d31u8, d2u8);

    d28u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 5);
    d29u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 5);
    d30u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 5);
    d31u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 5);

    q7u16 = vmlal_u8(q7u16, d28u8, d5u8);
    q8u16 = vmlal_u8(q8u16, d29u8, d5u8);
    q9u16 = vmlal_u8(q9u16, d30u8, d5u8);
    q10u16 = vmlal_u8(q10u16, d31u8, d5u8);

    d28u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 3);
    d29u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 3);
    d30u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 3);
    d31u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 3);

    q3u16 = vmull_u8(d28u8, d3u8);
    q4u16 = vmull_u8(d29u8, d3u8);
    q5u16 = vmull_u8(d30u8, d3u8);
    q6u16 = vmull_u8(d31u8, d3u8);

    q3s16 = vreinterpretq_s16_u16(q3u16);
    q4s16 = vreinterpretq_s16_u16(q4u16);
    q5s16 = vreinterpretq_s16_u16(q5u16);
    q6s16 = vreinterpretq_s16_u16(q6u16);
    q7s16 = vreinterpretq_s16_u16(q7u16);
    q8s16 = vreinterpretq_s16_u16(q8u16);
    q9s16 = vreinterpretq_s16_u16(q9u16);
    q10s16 = vreinterpretq_s16_u16(q10u16);

    q7s16 = vqaddq_s16(q7s16, q3s16);
    q8s16 = vqaddq_s16(q8s16, q4s16);
    q9s16 = vqaddq_s16(q9s16, q5s16);
    q10s16 = vqaddq_s16(q10s16, q6s16);

    d22u8 = vqrshrun_n_s16(q7s16, 7);
    d23u8 = vqrshrun_n_s16(q8s16, 7);
    d24u8 = vqrshrun_n_s16(q9s16, 7);
    d25u8 = vqrshrun_n_s16(q10s16, 7);

    if (yoffset == 0) {  // firstpass_filter8x4_only
        vst1_u8(dst_ptr, d22u8);
        dst_ptr += dst_pitch;
        vst1_u8(dst_ptr, d23u8);
        dst_ptr += dst_pitch;
        vst1_u8(dst_ptr, d24u8);
        dst_ptr += dst_pitch;
        vst1_u8(dst_ptr, d25u8);
        return;
    }

    // First Pass on rest 5-line data
    src += src_pixels_per_line;
    q3u8 = vld1q_u8(src);
    src += src_pixels_per_line;
    q4u8 = vld1q_u8(src);
    src += src_pixels_per_line;
    q5u8 = vld1q_u8(src);
    src += src_pixels_per_line;
    q6u8 = vld1q_u8(src);
    src += src_pixels_per_line;
    q7u8 = vld1q_u8(src);

    q8u16 = vmull_u8(vget_low_u8(q3u8), d0u8);
    q9u16 = vmull_u8(vget_low_u8(q4u8), d0u8);
    q10u16 = vmull_u8(vget_low_u8(q5u8), d0u8);
    q11u16 = vmull_u8(vget_low_u8(q6u8), d0u8);
    q12u16 = vmull_u8(vget_low_u8(q7u8), d0u8);

    d27u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 1);
    d28u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 1);
    d29u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 1);
    d30u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 1);
    d31u8 = vext_u8(vget_low_u8(q7u8), vget_high_u8(q7u8), 1);

    q8u16  = vmlsl_u8(q8u16, d27u8, d1u8);
    q9u16  = vmlsl_u8(q9u16, d28u8, d1u8);
    q10u16 = vmlsl_u8(q10u16, d29u8, d1u8);
    q11u16 = vmlsl_u8(q11u16, d30u8, d1u8);
    q12u16 = vmlsl_u8(q12u16, d31u8, d1u8);

    d27u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 4);
    d28u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 4);
    d29u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 4);
    d30u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 4);
    d31u8 = vext_u8(vget_low_u8(q7u8), vget_high_u8(q7u8), 4);

    q8u16  = vmlsl_u8(q8u16, d27u8, d4u8);
    q9u16  = vmlsl_u8(q9u16, d28u8, d4u8);
    q10u16 = vmlsl_u8(q10u16, d29u8, d4u8);
    q11u16 = vmlsl_u8(q11u16, d30u8, d4u8);
    q12u16 = vmlsl_u8(q12u16, d31u8, d4u8);

    d27u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 2);
    d28u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 2);
    d29u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 2);
    d30u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 2);
    d31u8 = vext_u8(vget_low_u8(q7u8), vget_high_u8(q7u8), 2);

    q8u16  = vmlal_u8(q8u16, d27u8, d2u8);
    q9u16  = vmlal_u8(q9u16, d28u8, d2u8);
    q10u16 = vmlal_u8(q10u16, d29u8, d2u8);
    q11u16 = vmlal_u8(q11u16, d30u8, d2u8);
    q12u16 = vmlal_u8(q12u16, d31u8, d2u8);

    d27u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 5);
    d28u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 5);
    d29u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 5);
    d30u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 5);
    d31u8 = vext_u8(vget_low_u8(q7u8), vget_high_u8(q7u8), 5);

    q8u16  = vmlal_u8(q8u16, d27u8, d5u8);
    q9u16  = vmlal_u8(q9u16, d28u8, d5u8);
    q10u16 = vmlal_u8(q10u16, d29u8, d5u8);
    q11u16 = vmlal_u8(q11u16, d30u8, d5u8);
    q12u16 = vmlal_u8(q12u16, d31u8, d5u8);

    d27u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 3);
    d28u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 3);
    d29u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 3);
    d30u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 3);
    d31u8 = vext_u8(vget_low_u8(q7u8), vget_high_u8(q7u8), 3);

    q3u16 = vmull_u8(d27u8, d3u8);
    q4u16 = vmull_u8(d28u8, d3u8);
    q5u16 = vmull_u8(d29u8, d3u8);
    q6u16 = vmull_u8(d30u8, d3u8);
    q7u16 = vmull_u8(d31u8, d3u8);

    q3s16 = vreinterpretq_s16_u16(q3u16);
    q4s16 = vreinterpretq_s16_u16(q4u16);
    q5s16 = vreinterpretq_s16_u16(q5u16);
    q6s16 = vreinterpretq_s16_u16(q6u16);
    q7s16 = vreinterpretq_s16_u16(q7u16);
    q8s16 = vreinterpretq_s16_u16(q8u16);
    q9s16 = vreinterpretq_s16_u16(q9u16);
    q10s16 = vreinterpretq_s16_u16(q10u16);
    q11s16 = vreinterpretq_s16_u16(q11u16);
    q12s16 = vreinterpretq_s16_u16(q12u16);

    q8s16 = vqaddq_s16(q8s16, q3s16);
    q9s16 = vqaddq_s16(q9s16, q4s16);
    q10s16 = vqaddq_s16(q10s16, q5s16);
    q11s16 = vqaddq_s16(q11s16, q6s16);
    q12s16 = vqaddq_s16(q12s16, q7s16);

    d26u8 = vqrshrun_n_s16(q8s16, 7);
    d27u8 = vqrshrun_n_s16(q9s16, 7);
    d28u8 = vqrshrun_n_s16(q10s16, 7);
    d29u8 = vqrshrun_n_s16(q11s16, 7);
    d30u8 = vqrshrun_n_s16(q12s16, 7);

    // Second pass: 8x4
    dtmps8 = vld1_s8(vp8_sub_pel_filters[yoffset]);
    d0s8 = vdup_lane_s8(dtmps8, 0);
    d1s8 = vdup_lane_s8(dtmps8, 1);
    d2s8 = vdup_lane_s8(dtmps8, 2);
    d3s8 = vdup_lane_s8(dtmps8, 3);
    d4s8 = vdup_lane_s8(dtmps8, 4);
    d5s8 = vdup_lane_s8(dtmps8, 5);
    d0u8 = vreinterpret_u8_s8(vabs_s8(d0s8));
    d1u8 = vreinterpret_u8_s8(vabs_s8(d1s8));
    d2u8 = vreinterpret_u8_s8(vabs_s8(d2s8));
    d3u8 = vreinterpret_u8_s8(vabs_s8(d3s8));
    d4u8 = vreinterpret_u8_s8(vabs_s8(d4s8));
    d5u8 = vreinterpret_u8_s8(vabs_s8(d5s8));

    q3u16 = vmull_u8(d22u8, d0u8);
    q4u16 = vmull_u8(d23u8, d0u8);
    q5u16 = vmull_u8(d24u8, d0u8);
    q6u16 = vmull_u8(d25u8, d0u8);

    q3u16 = vmlsl_u8(q3u16, d23u8, d1u8);
    q4u16 = vmlsl_u8(q4u16, d24u8, d1u8);
    q5u16 = vmlsl_u8(q5u16, d25u8, d1u8);
    q6u16 = vmlsl_u8(q6u16, d26u8, d1u8);

    q3u16 = vmlsl_u8(q3u16, d26u8, d4u8);
    q4u16 = vmlsl_u8(q4u16, d27u8, d4u8);
    q5u16 = vmlsl_u8(q5u16, d28u8, d4u8);
    q6u16 = vmlsl_u8(q6u16, d29u8, d4u8);

    q3u16 = vmlal_u8(q3u16, d24u8, d2u8);
    q4u16 = vmlal_u8(q4u16, d25u8, d2u8);
    q5u16 = vmlal_u8(q5u16, d26u8, d2u8);
    q6u16 = vmlal_u8(q6u16, d27u8, d2u8);

    q3u16 = vmlal_u8(q3u16, d27u8, d5u8);
    q4u16 = vmlal_u8(q4u16, d28u8, d5u8);
    q5u16 = vmlal_u8(q5u16, d29u8, d5u8);
    q6u16 = vmlal_u8(q6u16, d30u8, d5u8);

    q7u16 = vmull_u8(d25u8, d3u8);
    q8u16 = vmull_u8(d26u8, d3u8);
    q9u16 = vmull_u8(d27u8, d3u8);
    q10u16 = vmull_u8(d28u8, d3u8);

    q3s16 = vreinterpretq_s16_u16(q3u16);
    q4s16 = vreinterpretq_s16_u16(q4u16);
    q5s16 = vreinterpretq_s16_u16(q5u16);
    q6s16 = vreinterpretq_s16_u16(q6u16);
    q7s16 = vreinterpretq_s16_u16(q7u16);
    q8s16 = vreinterpretq_s16_u16(q8u16);
    q9s16 = vreinterpretq_s16_u16(q9u16);
    q10s16 = vreinterpretq_s16_u16(q10u16);

    q7s16 = vqaddq_s16(q7s16, q3s16);
    q8s16 = vqaddq_s16(q8s16, q4s16);
    q9s16 = vqaddq_s16(q9s16, q5s16);
    q10s16 = vqaddq_s16(q10s16, q6s16);

    d6u8 = vqrshrun_n_s16(q7s16, 7);
    d7u8 = vqrshrun_n_s16(q8s16, 7);
    d8u8 = vqrshrun_n_s16(q9s16, 7);
    d9u8 = vqrshrun_n_s16(q10s16, 7);

    vst1_u8(dst_ptr, d6u8);
    dst_ptr += dst_pitch;
    vst1_u8(dst_ptr, d7u8);
    dst_ptr += dst_pitch;
    vst1_u8(dst_ptr, d8u8);
    dst_ptr += dst_pitch;
    vst1_u8(dst_ptr, d9u8);
    return;
}

void vp8_sixtap_predict8x8_neon(
        unsigned char *src_ptr,
        int src_pixels_per_line,
        int xoffset,
        int yoffset,
        unsigned char *dst_ptr,
        int dst_pitch) {
    unsigned char *src, *tmpp;
    unsigned char tmp[64];
    int i;
    uint8x8_t d0u8, d1u8, d2u8, d3u8, d4u8, d5u8, d6u8, d7u8, d8u8, d9u8;
    uint8x8_t d18u8, d19u8, d20u8, d21u8, d22u8, d23u8, d24u8, d25u8;
    uint8x8_t d26u8, d27u8, d28u8, d29u8, d30u8, d31u8;
    int8x8_t dtmps8, d0s8, d1s8, d2s8, d3s8, d4s8, d5s8;
    uint16x8_t q3u16, q4u16, q5u16, q6u16, q7u16;
    uint16x8_t q8u16, q9u16, q10u16, q11u16, q12u16;
    int16x8_t q3s16, q4s16, q5s16, q6s16, q7s16;
    int16x8_t q8s16, q9s16, q10s16, q11s16, q12s16;
    uint8x16_t q3u8, q4u8, q5u8, q6u8, q7u8, q9u8, q10u8, q11u8, q12u8;

    if (xoffset == 0) {  // secondpass_filter8x8_only
        // load second_pass filter
        dtmps8 = vld1_s8(vp8_sub_pel_filters[yoffset]);
        d0s8 = vdup_lane_s8(dtmps8, 0);
        d1s8 = vdup_lane_s8(dtmps8, 1);
        d2s8 = vdup_lane_s8(dtmps8, 2);
        d3s8 = vdup_lane_s8(dtmps8, 3);
        d4s8 = vdup_lane_s8(dtmps8, 4);
        d5s8 = vdup_lane_s8(dtmps8, 5);
        d0u8 = vreinterpret_u8_s8(vabs_s8(d0s8));
        d1u8 = vreinterpret_u8_s8(vabs_s8(d1s8));
        d2u8 = vreinterpret_u8_s8(vabs_s8(d2s8));
        d3u8 = vreinterpret_u8_s8(vabs_s8(d3s8));
        d4u8 = vreinterpret_u8_s8(vabs_s8(d4s8));
        d5u8 = vreinterpret_u8_s8(vabs_s8(d5s8));

        // load src data
        src = src_ptr - src_pixels_per_line * 2;
        d18u8 = vld1_u8(src);
        src += src_pixels_per_line;
        d19u8 = vld1_u8(src);
        src += src_pixels_per_line;
        d20u8 = vld1_u8(src);
        src += src_pixels_per_line;
        d21u8 = vld1_u8(src);
        src += src_pixels_per_line;
        d22u8 = vld1_u8(src);
        src += src_pixels_per_line;
        d23u8 = vld1_u8(src);
        src += src_pixels_per_line;
        d24u8 = vld1_u8(src);
        src += src_pixels_per_line;
        d25u8 = vld1_u8(src);
        src += src_pixels_per_line;
        d26u8 = vld1_u8(src);
        src += src_pixels_per_line;
        d27u8 = vld1_u8(src);
        src += src_pixels_per_line;
        d28u8 = vld1_u8(src);
        src += src_pixels_per_line;
        d29u8 = vld1_u8(src);
        src += src_pixels_per_line;
        d30u8 = vld1_u8(src);

        for (i = 2; i > 0; i--) {
            q3u16 = vmull_u8(d18u8, d0u8);
            q4u16 = vmull_u8(d19u8, d0u8);
            q5u16 = vmull_u8(d20u8, d0u8);
            q6u16 = vmull_u8(d21u8, d0u8);

            q3u16 = vmlsl_u8(q3u16, d19u8, d1u8);
            q4u16 = vmlsl_u8(q4u16, d20u8, d1u8);
            q5u16 = vmlsl_u8(q5u16, d21u8, d1u8);
            q6u16 = vmlsl_u8(q6u16, d22u8, d1u8);

            q3u16 = vmlsl_u8(q3u16, d22u8, d4u8);
            q4u16 = vmlsl_u8(q4u16, d23u8, d4u8);
            q5u16 = vmlsl_u8(q5u16, d24u8, d4u8);
            q6u16 = vmlsl_u8(q6u16, d25u8, d4u8);

            q3u16 = vmlal_u8(q3u16, d20u8, d2u8);
            q4u16 = vmlal_u8(q4u16, d21u8, d2u8);
            q5u16 = vmlal_u8(q5u16, d22u8, d2u8);
            q6u16 = vmlal_u8(q6u16, d23u8, d2u8);

            q3u16 = vmlal_u8(q3u16, d23u8, d5u8);
            q4u16 = vmlal_u8(q4u16, d24u8, d5u8);
            q5u16 = vmlal_u8(q5u16, d25u8, d5u8);
            q6u16 = vmlal_u8(q6u16, d26u8, d5u8);

            q7u16 = vmull_u8(d21u8, d3u8);
            q8u16 = vmull_u8(d22u8, d3u8);
            q9u16 = vmull_u8(d23u8, d3u8);
            q10u16 = vmull_u8(d24u8, d3u8);

            q3s16 = vreinterpretq_s16_u16(q3u16);
            q4s16 = vreinterpretq_s16_u16(q4u16);
            q5s16 = vreinterpretq_s16_u16(q5u16);
            q6s16 = vreinterpretq_s16_u16(q6u16);
            q7s16 = vreinterpretq_s16_u16(q7u16);
            q8s16 = vreinterpretq_s16_u16(q8u16);
            q9s16 = vreinterpretq_s16_u16(q9u16);
            q10s16 = vreinterpretq_s16_u16(q10u16);

            q7s16 = vqaddq_s16(q7s16, q3s16);
            q8s16 = vqaddq_s16(q8s16, q4s16);
            q9s16 = vqaddq_s16(q9s16, q5s16);
            q10s16 = vqaddq_s16(q10s16, q6s16);

            d6u8 = vqrshrun_n_s16(q7s16, 7);
            d7u8 = vqrshrun_n_s16(q8s16, 7);
            d8u8 = vqrshrun_n_s16(q9s16, 7);
            d9u8 = vqrshrun_n_s16(q10s16, 7);

            d18u8 = d22u8;
            d19u8 = d23u8;
            d20u8 = d24u8;
            d21u8 = d25u8;
            d22u8 = d26u8;
            d23u8 = d27u8;
            d24u8 = d28u8;
            d25u8 = d29u8;
            d26u8 = d30u8;

            vst1_u8(dst_ptr, d6u8);
            dst_ptr += dst_pitch;
            vst1_u8(dst_ptr, d7u8);
            dst_ptr += dst_pitch;
            vst1_u8(dst_ptr, d8u8);
            dst_ptr += dst_pitch;
            vst1_u8(dst_ptr, d9u8);
            dst_ptr += dst_pitch;
        }
        return;
    }

    // load first_pass filter
    dtmps8 = vld1_s8(vp8_sub_pel_filters[xoffset]);
    d0s8 = vdup_lane_s8(dtmps8, 0);
    d1s8 = vdup_lane_s8(dtmps8, 1);
    d2s8 = vdup_lane_s8(dtmps8, 2);
    d3s8 = vdup_lane_s8(dtmps8, 3);
    d4s8 = vdup_lane_s8(dtmps8, 4);
    d5s8 = vdup_lane_s8(dtmps8, 5);
    d0u8 = vreinterpret_u8_s8(vabs_s8(d0s8));
    d1u8 = vreinterpret_u8_s8(vabs_s8(d1s8));
    d2u8 = vreinterpret_u8_s8(vabs_s8(d2s8));
    d3u8 = vreinterpret_u8_s8(vabs_s8(d3s8));
    d4u8 = vreinterpret_u8_s8(vabs_s8(d4s8));
    d5u8 = vreinterpret_u8_s8(vabs_s8(d5s8));

    // First pass: output_height lines x output_width columns (9x4)
    if (yoffset == 0)  // firstpass_filter4x4_only
        src = src_ptr - 2;
    else
        src = src_ptr - 2 - (src_pixels_per_line * 2);

    tmpp = tmp;
    for (i = 2; i > 0; i--) {
        q3u8 = vld1q_u8(src);
        src += src_pixels_per_line;
        q4u8 = vld1q_u8(src);
        src += src_pixels_per_line;
        q5u8 = vld1q_u8(src);
        src += src_pixels_per_line;
        q6u8 = vld1q_u8(src);
        src += src_pixels_per_line;

        __builtin_prefetch(src);
        __builtin_prefetch(src + src_pixels_per_line);
        __builtin_prefetch(src + src_pixels_per_line * 2);

        q7u16 = vmull_u8(vget_low_u8(q3u8), d0u8);
        q8u16 = vmull_u8(vget_low_u8(q4u8), d0u8);
        q9u16 = vmull_u8(vget_low_u8(q5u8), d0u8);
        q10u16 = vmull_u8(vget_low_u8(q6u8), d0u8);

        d28u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 1);
        d29u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 1);
        d30u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 1);
        d31u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 1);

        q7u16 = vmlsl_u8(q7u16, d28u8, d1u8);
        q8u16 = vmlsl_u8(q8u16, d29u8, d1u8);
        q9u16 = vmlsl_u8(q9u16, d30u8, d1u8);
        q10u16 = vmlsl_u8(q10u16, d31u8, d1u8);

        d28u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 4);
        d29u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 4);
        d30u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 4);
        d31u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 4);

        q7u16 = vmlsl_u8(q7u16, d28u8, d4u8);
        q8u16 = vmlsl_u8(q8u16, d29u8, d4u8);
        q9u16 = vmlsl_u8(q9u16, d30u8, d4u8);
        q10u16 = vmlsl_u8(q10u16, d31u8, d4u8);

        d28u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 2);
        d29u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 2);
        d30u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 2);
        d31u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 2);

        q7u16 = vmlal_u8(q7u16, d28u8, d2u8);
        q8u16 = vmlal_u8(q8u16, d29u8, d2u8);
        q9u16 = vmlal_u8(q9u16, d30u8, d2u8);
        q10u16 = vmlal_u8(q10u16, d31u8, d2u8);

        d28u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 5);
        d29u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 5);
        d30u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 5);
        d31u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 5);

        q7u16 = vmlal_u8(q7u16, d28u8, d5u8);
        q8u16 = vmlal_u8(q8u16, d29u8, d5u8);
        q9u16 = vmlal_u8(q9u16, d30u8, d5u8);
        q10u16 = vmlal_u8(q10u16, d31u8, d5u8);

        d28u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 3);
        d29u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 3);
        d30u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 3);
        d31u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 3);

        q3u16 = vmull_u8(d28u8, d3u8);
        q4u16 = vmull_u8(d29u8, d3u8);
        q5u16 = vmull_u8(d30u8, d3u8);
        q6u16 = vmull_u8(d31u8, d3u8);

        q3s16 = vreinterpretq_s16_u16(q3u16);
        q4s16 = vreinterpretq_s16_u16(q4u16);
        q5s16 = vreinterpretq_s16_u16(q5u16);
        q6s16 = vreinterpretq_s16_u16(q6u16);
        q7s16 = vreinterpretq_s16_u16(q7u16);
        q8s16 = vreinterpretq_s16_u16(q8u16);
        q9s16 = vreinterpretq_s16_u16(q9u16);
        q10s16 = vreinterpretq_s16_u16(q10u16);

        q7s16 = vqaddq_s16(q7s16, q3s16);
        q8s16 = vqaddq_s16(q8s16, q4s16);
        q9s16 = vqaddq_s16(q9s16, q5s16);
        q10s16 = vqaddq_s16(q10s16, q6s16);

        d22u8 = vqrshrun_n_s16(q7s16, 7);
        d23u8 = vqrshrun_n_s16(q8s16, 7);
        d24u8 = vqrshrun_n_s16(q9s16, 7);
        d25u8 = vqrshrun_n_s16(q10s16, 7);

        if (yoffset == 0) {  // firstpass_filter8x4_only
            vst1_u8(dst_ptr, d22u8);
            dst_ptr += dst_pitch;
            vst1_u8(dst_ptr, d23u8);
            dst_ptr += dst_pitch;
            vst1_u8(dst_ptr, d24u8);
            dst_ptr += dst_pitch;
            vst1_u8(dst_ptr, d25u8);
            dst_ptr += dst_pitch;
        } else {
            vst1_u8(tmpp, d22u8);
            tmpp += 8;
            vst1_u8(tmpp, d23u8);
            tmpp += 8;
            vst1_u8(tmpp, d24u8);
            tmpp += 8;
            vst1_u8(tmpp, d25u8);
            tmpp += 8;
        }
    }
    if (yoffset == 0)
        return;

    // First Pass on rest 5-line data
    q3u8 = vld1q_u8(src);
    src += src_pixels_per_line;
    q4u8 = vld1q_u8(src);
    src += src_pixels_per_line;
    q5u8 = vld1q_u8(src);
    src += src_pixels_per_line;
    q6u8 = vld1q_u8(src);
    src += src_pixels_per_line;
    q7u8 = vld1q_u8(src);

    q8u16 = vmull_u8(vget_low_u8(q3u8), d0u8);
    q9u16 = vmull_u8(vget_low_u8(q4u8), d0u8);
    q10u16 = vmull_u8(vget_low_u8(q5u8), d0u8);
    q11u16 = vmull_u8(vget_low_u8(q6u8), d0u8);
    q12u16 = vmull_u8(vget_low_u8(q7u8), d0u8);

    d27u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 1);
    d28u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 1);
    d29u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 1);
    d30u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 1);
    d31u8 = vext_u8(vget_low_u8(q7u8), vget_high_u8(q7u8), 1);

    q8u16 = vmlsl_u8(q8u16, d27u8, d1u8);
    q9u16 = vmlsl_u8(q9u16, d28u8, d1u8);
    q10u16 = vmlsl_u8(q10u16, d29u8, d1u8);
    q11u16 = vmlsl_u8(q11u16, d30u8, d1u8);
    q12u16 = vmlsl_u8(q12u16, d31u8, d1u8);

    d27u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 4);
    d28u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 4);
    d29u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 4);
    d30u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 4);
    d31u8 = vext_u8(vget_low_u8(q7u8), vget_high_u8(q7u8), 4);

    q8u16 = vmlsl_u8(q8u16, d27u8, d4u8);
    q9u16 = vmlsl_u8(q9u16, d28u8, d4u8);
    q10u16 = vmlsl_u8(q10u16, d29u8, d4u8);
    q11u16 = vmlsl_u8(q11u16, d30u8, d4u8);
    q12u16 = vmlsl_u8(q12u16, d31u8, d4u8);

    d27u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 2);
    d28u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 2);
    d29u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 2);
    d30u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 2);
    d31u8 = vext_u8(vget_low_u8(q7u8), vget_high_u8(q7u8), 2);

    q8u16 = vmlal_u8(q8u16, d27u8, d2u8);
    q9u16 = vmlal_u8(q9u16, d28u8, d2u8);
    q10u16 = vmlal_u8(q10u16, d29u8, d2u8);
    q11u16 = vmlal_u8(q11u16, d30u8, d2u8);
    q12u16 = vmlal_u8(q12u16, d31u8, d2u8);

    d27u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 5);
    d28u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 5);
    d29u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 5);
    d30u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 5);
    d31u8 = vext_u8(vget_low_u8(q7u8), vget_high_u8(q7u8), 5);

    q8u16 = vmlal_u8(q8u16, d27u8, d5u8);
    q9u16 = vmlal_u8(q9u16, d28u8, d5u8);
    q10u16 = vmlal_u8(q10u16, d29u8, d5u8);
    q11u16 = vmlal_u8(q11u16, d30u8, d5u8);
    q12u16 = vmlal_u8(q12u16, d31u8, d5u8);

    d27u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 3);
    d28u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 3);
    d29u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 3);
    d30u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 3);
    d31u8 = vext_u8(vget_low_u8(q7u8), vget_high_u8(q7u8), 3);

    q3u16 = vmull_u8(d27u8, d3u8);
    q4u16 = vmull_u8(d28u8, d3u8);
    q5u16 = vmull_u8(d29u8, d3u8);
    q6u16 = vmull_u8(d30u8, d3u8);
    q7u16 = vmull_u8(d31u8, d3u8);

    q3s16 = vreinterpretq_s16_u16(q3u16);
    q4s16 = vreinterpretq_s16_u16(q4u16);
    q5s16 = vreinterpretq_s16_u16(q5u16);
    q6s16 = vreinterpretq_s16_u16(q6u16);
    q7s16 = vreinterpretq_s16_u16(q7u16);
    q8s16 = vreinterpretq_s16_u16(q8u16);
    q9s16 = vreinterpretq_s16_u16(q9u16);
    q10s16 = vreinterpretq_s16_u16(q10u16);
    q11s16 = vreinterpretq_s16_u16(q11u16);
    q12s16 = vreinterpretq_s16_u16(q12u16);

    q8s16 = vqaddq_s16(q8s16, q3s16);
    q9s16 = vqaddq_s16(q9s16, q4s16);
    q10s16 = vqaddq_s16(q10s16, q5s16);
    q11s16 = vqaddq_s16(q11s16, q6s16);
    q12s16 = vqaddq_s16(q12s16, q7s16);

    d26u8 = vqrshrun_n_s16(q8s16, 7);
    d27u8 = vqrshrun_n_s16(q9s16, 7);
    d28u8 = vqrshrun_n_s16(q10s16, 7);
    d29u8 = vqrshrun_n_s16(q11s16, 7);
    d30u8 = vqrshrun_n_s16(q12s16, 7);

    // Second pass: 8x8
    dtmps8 = vld1_s8(vp8_sub_pel_filters[yoffset]);
    d0s8 = vdup_lane_s8(dtmps8, 0);
    d1s8 = vdup_lane_s8(dtmps8, 1);
    d2s8 = vdup_lane_s8(dtmps8, 2);
    d3s8 = vdup_lane_s8(dtmps8, 3);
    d4s8 = vdup_lane_s8(dtmps8, 4);
    d5s8 = vdup_lane_s8(dtmps8, 5);
    d0u8 = vreinterpret_u8_s8(vabs_s8(d0s8));
    d1u8 = vreinterpret_u8_s8(vabs_s8(d1s8));
    d2u8 = vreinterpret_u8_s8(vabs_s8(d2s8));
    d3u8 = vreinterpret_u8_s8(vabs_s8(d3s8));
    d4u8 = vreinterpret_u8_s8(vabs_s8(d4s8));
    d5u8 = vreinterpret_u8_s8(vabs_s8(d5s8));

    tmpp = tmp;
    q9u8 = vld1q_u8(tmpp);
    tmpp += 16;
    q10u8 = vld1q_u8(tmpp);
    tmpp += 16;
    q11u8 = vld1q_u8(tmpp);
    tmpp += 16;
    q12u8 = vld1q_u8(tmpp);

    d18u8 = vget_low_u8(q9u8);
    d19u8 = vget_high_u8(q9u8);
    d20u8 = vget_low_u8(q10u8);
    d21u8 = vget_high_u8(q10u8);
    d22u8 = vget_low_u8(q11u8);
    d23u8 = vget_high_u8(q11u8);
    d24u8 = vget_low_u8(q12u8);
    d25u8 = vget_high_u8(q12u8);

    for (i = 2; i > 0; i--) {
        q3u16 = vmull_u8(d18u8, d0u8);
        q4u16 = vmull_u8(d19u8, d0u8);
        q5u16 = vmull_u8(d20u8, d0u8);
        q6u16 = vmull_u8(d21u8, d0u8);

        q3u16 = vmlsl_u8(q3u16, d19u8, d1u8);
        q4u16 = vmlsl_u8(q4u16, d20u8, d1u8);
        q5u16 = vmlsl_u8(q5u16, d21u8, d1u8);
        q6u16 = vmlsl_u8(q6u16, d22u8, d1u8);

        q3u16 = vmlsl_u8(q3u16, d22u8, d4u8);
        q4u16 = vmlsl_u8(q4u16, d23u8, d4u8);
        q5u16 = vmlsl_u8(q5u16, d24u8, d4u8);
        q6u16 = vmlsl_u8(q6u16, d25u8, d4u8);

        q3u16 = vmlal_u8(q3u16, d20u8, d2u8);
        q4u16 = vmlal_u8(q4u16, d21u8, d2u8);
        q5u16 = vmlal_u8(q5u16, d22u8, d2u8);
        q6u16 = vmlal_u8(q6u16, d23u8, d2u8);

        q3u16 = vmlal_u8(q3u16, d23u8, d5u8);
        q4u16 = vmlal_u8(q4u16, d24u8, d5u8);
        q5u16 = vmlal_u8(q5u16, d25u8, d5u8);
        q6u16 = vmlal_u8(q6u16, d26u8, d5u8);

        q7u16 = vmull_u8(d21u8, d3u8);
        q8u16 = vmull_u8(d22u8, d3u8);
        q9u16 = vmull_u8(d23u8, d3u8);
        q10u16 = vmull_u8(d24u8, d3u8);

        q3s16 = vreinterpretq_s16_u16(q3u16);
        q4s16 = vreinterpretq_s16_u16(q4u16);
        q5s16 = vreinterpretq_s16_u16(q5u16);
        q6s16 = vreinterpretq_s16_u16(q6u16);
        q7s16 = vreinterpretq_s16_u16(q7u16);
        q8s16 = vreinterpretq_s16_u16(q8u16);
        q9s16 = vreinterpretq_s16_u16(q9u16);
        q10s16 = vreinterpretq_s16_u16(q10u16);

        q7s16 = vqaddq_s16(q7s16, q3s16);
        q8s16 = vqaddq_s16(q8s16, q4s16);
        q9s16 = vqaddq_s16(q9s16, q5s16);
        q10s16 = vqaddq_s16(q10s16, q6s16);

        d6u8 = vqrshrun_n_s16(q7s16, 7);
        d7u8 = vqrshrun_n_s16(q8s16, 7);
        d8u8 = vqrshrun_n_s16(q9s16, 7);
        d9u8 = vqrshrun_n_s16(q10s16, 7);

        d18u8 = d22u8;
        d19u8 = d23u8;
        d20u8 = d24u8;
        d21u8 = d25u8;
        d22u8 = d26u8;
        d23u8 = d27u8;
        d24u8 = d28u8;
        d25u8 = d29u8;
        d26u8 = d30u8;

        vst1_u8(dst_ptr, d6u8);
        dst_ptr += dst_pitch;
        vst1_u8(dst_ptr, d7u8);
        dst_ptr += dst_pitch;
        vst1_u8(dst_ptr, d8u8);
        dst_ptr += dst_pitch;
        vst1_u8(dst_ptr, d9u8);
        dst_ptr += dst_pitch;
    }
    return;
}

void vp8_sixtap_predict16x16_neon(
        unsigned char *src_ptr,
        int src_pixels_per_line,
        int xoffset,
        int yoffset,
        unsigned char *dst_ptr,
        int dst_pitch) {
    unsigned char *src, *src_tmp, *dst, *tmpp;
    unsigned char tmp[336];
    int i, j;
    uint8x8_t d0u8, d1u8, d2u8, d3u8, d4u8, d5u8, d6u8, d7u8, d8u8, d9u8;
    uint8x8_t d10u8, d11u8, d12u8, d13u8, d14u8, d15u8, d18u8, d19u8;
    uint8x8_t d20u8, d21u8, d22u8, d23u8, d24u8, d25u8, d26u8, d27u8;
    uint8x8_t d28u8, d29u8, d30u8, d31u8;
    int8x8_t dtmps8, d0s8, d1s8, d2s8, d3s8, d4s8, d5s8;
    uint8x16_t q3u8, q4u8;
    uint16x8_t q3u16, q4u16, q5u16, q6u16, q7u16, q8u16, q9u16, q10u16;
    uint16x8_t q11u16, q12u16, q13u16, q15u16;
    int16x8_t q3s16, q4s16, q5s16, q6s16, q7s16, q8s16, q9s16, q10s16;
    int16x8_t q11s16, q12s16, q13s16, q15s16;

    if (xoffset == 0) {  // secondpass_filter8x8_only
        // load second_pass filter
        dtmps8 = vld1_s8(vp8_sub_pel_filters[yoffset]);
        d0s8 = vdup_lane_s8(dtmps8, 0);
        d1s8 = vdup_lane_s8(dtmps8, 1);
        d2s8 = vdup_lane_s8(dtmps8, 2);
        d3s8 = vdup_lane_s8(dtmps8, 3);
        d4s8 = vdup_lane_s8(dtmps8, 4);
        d5s8 = vdup_lane_s8(dtmps8, 5);
        d0u8 = vreinterpret_u8_s8(vabs_s8(d0s8));
        d1u8 = vreinterpret_u8_s8(vabs_s8(d1s8));
        d2u8 = vreinterpret_u8_s8(vabs_s8(d2s8));
        d3u8 = vreinterpret_u8_s8(vabs_s8(d3s8));
        d4u8 = vreinterpret_u8_s8(vabs_s8(d4s8));
        d5u8 = vreinterpret_u8_s8(vabs_s8(d5s8));

        // load src data
        src_tmp = src_ptr - src_pixels_per_line * 2;
        for (i = 0; i < 2; i++) {
            src = src_tmp + i * 8;
            dst = dst_ptr + i * 8;
            d18u8 = vld1_u8(src);
            src += src_pixels_per_line;
            d19u8 = vld1_u8(src);
            src += src_pixels_per_line;
            d20u8 = vld1_u8(src);
            src += src_pixels_per_line;
            d21u8 = vld1_u8(src);
            src += src_pixels_per_line;
            d22u8 = vld1_u8(src);
            src += src_pixels_per_line;
            for (j = 0; j < 4; j++) {
                d23u8 = vld1_u8(src);
                src += src_pixels_per_line;
                d24u8 = vld1_u8(src);
                src += src_pixels_per_line;
                d25u8 = vld1_u8(src);
                src += src_pixels_per_line;
                d26u8 = vld1_u8(src);
                src += src_pixels_per_line;

                q3u16 = vmull_u8(d18u8, d0u8);
                q4u16 = vmull_u8(d19u8, d0u8);
                q5u16 = vmull_u8(d20u8, d0u8);
                q6u16 = vmull_u8(d21u8, d0u8);

                q3u16 = vmlsl_u8(q3u16, d19u8, d1u8);
                q4u16 = vmlsl_u8(q4u16, d20u8, d1u8);
                q5u16 = vmlsl_u8(q5u16, d21u8, d1u8);
                q6u16 = vmlsl_u8(q6u16, d22u8, d1u8);

                q3u16 = vmlsl_u8(q3u16, d22u8, d4u8);
                q4u16 = vmlsl_u8(q4u16, d23u8, d4u8);
                q5u16 = vmlsl_u8(q5u16, d24u8, d4u8);
                q6u16 = vmlsl_u8(q6u16, d25u8, d4u8);

                q3u16 = vmlal_u8(q3u16, d20u8, d2u8);
                q4u16 = vmlal_u8(q4u16, d21u8, d2u8);
                q5u16 = vmlal_u8(q5u16, d22u8, d2u8);
                q6u16 = vmlal_u8(q6u16, d23u8, d2u8);

                q3u16 = vmlal_u8(q3u16, d23u8, d5u8);
                q4u16 = vmlal_u8(q4u16, d24u8, d5u8);
                q5u16 = vmlal_u8(q5u16, d25u8, d5u8);
                q6u16 = vmlal_u8(q6u16, d26u8, d5u8);

                q7u16 = vmull_u8(d21u8, d3u8);
                q8u16 = vmull_u8(d22u8, d3u8);
                q9u16 = vmull_u8(d23u8, d3u8);
                q10u16 = vmull_u8(d24u8, d3u8);

                q3s16 = vreinterpretq_s16_u16(q3u16);
                q4s16 = vreinterpretq_s16_u16(q4u16);
                q5s16 = vreinterpretq_s16_u16(q5u16);
                q6s16 = vreinterpretq_s16_u16(q6u16);
                q7s16 = vreinterpretq_s16_u16(q7u16);
                q8s16 = vreinterpretq_s16_u16(q8u16);
                q9s16 = vreinterpretq_s16_u16(q9u16);
                q10s16 = vreinterpretq_s16_u16(q10u16);

                q7s16 = vqaddq_s16(q7s16, q3s16);
                q8s16 = vqaddq_s16(q8s16, q4s16);
                q9s16 = vqaddq_s16(q9s16, q5s16);
                q10s16 = vqaddq_s16(q10s16, q6s16);

                d6u8 = vqrshrun_n_s16(q7s16, 7);
                d7u8 = vqrshrun_n_s16(q8s16, 7);
                d8u8 = vqrshrun_n_s16(q9s16, 7);
                d9u8 = vqrshrun_n_s16(q10s16, 7);

                d18u8 = d22u8;
                d19u8 = d23u8;
                d20u8 = d24u8;
                d21u8 = d25u8;
                d22u8 = d26u8;

                vst1_u8(dst, d6u8);
                dst += dst_pitch;
                vst1_u8(dst, d7u8);
                dst += dst_pitch;
                vst1_u8(dst, d8u8);
                dst += dst_pitch;
                vst1_u8(dst, d9u8);
                dst += dst_pitch;
            }
        }
        return;
    }

    // load first_pass filter
    dtmps8 = vld1_s8(vp8_sub_pel_filters[xoffset]);
    d0s8 = vdup_lane_s8(dtmps8, 0);
    d1s8 = vdup_lane_s8(dtmps8, 1);
    d2s8 = vdup_lane_s8(dtmps8, 2);
    d3s8 = vdup_lane_s8(dtmps8, 3);
    d4s8 = vdup_lane_s8(dtmps8, 4);
    d5s8 = vdup_lane_s8(dtmps8, 5);
    d0u8 = vreinterpret_u8_s8(vabs_s8(d0s8));
    d1u8 = vreinterpret_u8_s8(vabs_s8(d1s8));
    d2u8 = vreinterpret_u8_s8(vabs_s8(d2s8));
    d3u8 = vreinterpret_u8_s8(vabs_s8(d3s8));
    d4u8 = vreinterpret_u8_s8(vabs_s8(d4s8));
    d5u8 = vreinterpret_u8_s8(vabs_s8(d5s8));

    // First pass: output_height lines x output_width columns (9x4)
    if (yoffset == 0) {  // firstpass_filter4x4_only
        src = src_ptr - 2;
        dst = dst_ptr;
        for (i = 0; i < 8; i++) {
            d6u8 = vld1_u8(src);
            d7u8 = vld1_u8(src + 8);
            d8u8 = vld1_u8(src + 16);
            src += src_pixels_per_line;
            d9u8 = vld1_u8(src);
            d10u8 = vld1_u8(src + 8);
            d11u8 = vld1_u8(src + 16);
            src += src_pixels_per_line;

            __builtin_prefetch(src);
            __builtin_prefetch(src + src_pixels_per_line);

            q6u16 = vmull_u8(d6u8, d0u8);
            q7u16 = vmull_u8(d7u8, d0u8);
            q8u16 = vmull_u8(d9u8, d0u8);
            q9u16 = vmull_u8(d10u8, d0u8);

            d20u8 = vext_u8(d6u8, d7u8, 1);
            d21u8 = vext_u8(d9u8, d10u8, 1);
            d22u8 = vext_u8(d7u8, d8u8, 1);
            d23u8 = vext_u8(d10u8, d11u8, 1);
            d24u8 = vext_u8(d6u8, d7u8, 4);
            d25u8 = vext_u8(d9u8, d10u8, 4);
            d26u8 = vext_u8(d7u8, d8u8, 4);
            d27u8 = vext_u8(d10u8, d11u8, 4);
            d28u8 = vext_u8(d6u8, d7u8, 5);
            d29u8 = vext_u8(d9u8, d10u8, 5);

            q6u16 = vmlsl_u8(q6u16, d20u8, d1u8);
            q8u16 = vmlsl_u8(q8u16, d21u8, d1u8);
            q7u16 = vmlsl_u8(q7u16, d22u8, d1u8);
            q9u16 = vmlsl_u8(q9u16, d23u8, d1u8);
            q6u16 = vmlsl_u8(q6u16, d24u8, d4u8);
            q8u16 = vmlsl_u8(q8u16, d25u8, d4u8);
            q7u16 = vmlsl_u8(q7u16, d26u8, d4u8);
            q9u16 = vmlsl_u8(q9u16, d27u8, d4u8);
            q6u16 = vmlal_u8(q6u16, d28u8, d5u8);
            q8u16 = vmlal_u8(q8u16, d29u8, d5u8);

            d20u8 = vext_u8(d7u8, d8u8, 5);
            d21u8 = vext_u8(d10u8, d11u8, 5);
            d22u8 = vext_u8(d6u8, d7u8, 2);
            d23u8 = vext_u8(d9u8, d10u8, 2);
            d24u8 = vext_u8(d7u8, d8u8, 2);
            d25u8 = vext_u8(d10u8, d11u8, 2);
            d26u8 = vext_u8(d6u8, d7u8, 3);
            d27u8 = vext_u8(d9u8, d10u8, 3);
            d28u8 = vext_u8(d7u8, d8u8, 3);
            d29u8 = vext_u8(d10u8, d11u8, 3);

            q7u16 = vmlal_u8(q7u16, d20u8, d5u8);
            q9u16 = vmlal_u8(q9u16, d21u8, d5u8);
            q6u16 = vmlal_u8(q6u16, d22u8, d2u8);
            q8u16 = vmlal_u8(q8u16, d23u8, d2u8);
            q7u16 = vmlal_u8(q7u16, d24u8, d2u8);
            q9u16 = vmlal_u8(q9u16, d25u8, d2u8);

            q10u16 = vmull_u8(d26u8, d3u8);
            q11u16 = vmull_u8(d27u8, d3u8);
            q12u16 = vmull_u8(d28u8, d3u8);
            q15u16 = vmull_u8(d29u8, d3u8);

            q6s16 = vreinterpretq_s16_u16(q6u16);
            q7s16 = vreinterpretq_s16_u16(q7u16);
            q8s16 = vreinterpretq_s16_u16(q8u16);
            q9s16 = vreinterpretq_s16_u16(q9u16);
            q10s16 = vreinterpretq_s16_u16(q10u16);
            q11s16 = vreinterpretq_s16_u16(q11u16);
            q12s16 = vreinterpretq_s16_u16(q12u16);
            q15s16 = vreinterpretq_s16_u16(q15u16);

            q6s16 = vqaddq_s16(q6s16, q10s16);
            q8s16 = vqaddq_s16(q8s16, q11s16);
            q7s16 = vqaddq_s16(q7s16, q12s16);
            q9s16 = vqaddq_s16(q9s16, q15s16);

            d6u8 = vqrshrun_n_s16(q6s16, 7);
            d7u8 = vqrshrun_n_s16(q7s16, 7);
            d8u8 = vqrshrun_n_s16(q8s16, 7);
            d9u8 = vqrshrun_n_s16(q9s16, 7);

            q3u8 = vcombine_u8(d6u8, d7u8);
            q4u8 = vcombine_u8(d8u8, d9u8);
            vst1q_u8(dst, q3u8);
            dst += dst_pitch;
            vst1q_u8(dst, q4u8);
            dst += dst_pitch;
        }
        return;
    }

    src = src_ptr - 2 - src_pixels_per_line * 2;
    tmpp = tmp;
    for (i = 0; i < 7; i++) {
        d6u8 = vld1_u8(src);
        d7u8 = vld1_u8(src + 8);
        d8u8 = vld1_u8(src + 16);
        src += src_pixels_per_line;
        d9u8 = vld1_u8(src);
        d10u8 = vld1_u8(src + 8);
        d11u8 = vld1_u8(src + 16);
        src += src_pixels_per_line;
        d12u8 = vld1_u8(src);
        d13u8 = vld1_u8(src + 8);
        d14u8 = vld1_u8(src + 16);
        src += src_pixels_per_line;

        __builtin_prefetch(src);
        __builtin_prefetch(src + src_pixels_per_line);
        __builtin_prefetch(src + src_pixels_per_line * 2);

        q8u16 = vmull_u8(d6u8, d0u8);
        q9u16 = vmull_u8(d7u8, d0u8);
        q10u16 = vmull_u8(d9u8, d0u8);
        q11u16 = vmull_u8(d10u8, d0u8);
        q12u16 = vmull_u8(d12u8, d0u8);
        q13u16 = vmull_u8(d13u8, d0u8);

        d28u8 = vext_u8(d6u8, d7u8, 1);
        d29u8 = vext_u8(d9u8, d10u8, 1);
        d30u8 = vext_u8(d12u8, d13u8, 1);
        q8u16 = vmlsl_u8(q8u16, d28u8, d1u8);
        q10u16 = vmlsl_u8(q10u16, d29u8, d1u8);
        q12u16 = vmlsl_u8(q12u16, d30u8, d1u8);
        d28u8 = vext_u8(d7u8, d8u8, 1);
        d29u8 = vext_u8(d10u8, d11u8, 1);
        d30u8 = vext_u8(d13u8, d14u8, 1);
        q9u16  = vmlsl_u8(q9u16, d28u8, d1u8);
        q11u16 = vmlsl_u8(q11u16, d29u8, d1u8);
        q13u16 = vmlsl_u8(q13u16, d30u8, d1u8);

        d28u8 = vext_u8(d6u8, d7u8, 4);
        d29u8 = vext_u8(d9u8, d10u8, 4);
        d30u8 = vext_u8(d12u8, d13u8, 4);
        q8u16 = vmlsl_u8(q8u16, d28u8, d4u8);
        q10u16 = vmlsl_u8(q10u16, d29u8, d4u8);
        q12u16 = vmlsl_u8(q12u16, d30u8, d4u8);
        d28u8 = vext_u8(d7u8, d8u8, 4);
        d29u8 = vext_u8(d10u8, d11u8, 4);
        d30u8 = vext_u8(d13u8, d14u8, 4);
        q9u16 = vmlsl_u8(q9u16, d28u8, d4u8);
        q11u16 = vmlsl_u8(q11u16, d29u8, d4u8);
        q13u16 = vmlsl_u8(q13u16, d30u8, d4u8);

        d28u8 = vext_u8(d6u8, d7u8, 5);
        d29u8 = vext_u8(d9u8, d10u8, 5);
        d30u8 = vext_u8(d12u8, d13u8, 5);
        q8u16 = vmlal_u8(q8u16, d28u8, d5u8);
        q10u16 = vmlal_u8(q10u16, d29u8, d5u8);
        q12u16 = vmlal_u8(q12u16, d30u8, d5u8);
        d28u8 = vext_u8(d7u8, d8u8, 5);
        d29u8 = vext_u8(d10u8, d11u8, 5);
        d30u8 = vext_u8(d13u8, d14u8, 5);
        q9u16 = vmlal_u8(q9u16, d28u8, d5u8);
        q11u16 = vmlal_u8(q11u16, d29u8, d5u8);
        q13u16 = vmlal_u8(q13u16, d30u8, d5u8);

        d28u8 = vext_u8(d6u8, d7u8, 2);
        d29u8 = vext_u8(d9u8, d10u8, 2);
        d30u8 = vext_u8(d12u8, d13u8, 2);
        q8u16 = vmlal_u8(q8u16, d28u8, d2u8);
        q10u16 = vmlal_u8(q10u16, d29u8, d2u8);
        q12u16 = vmlal_u8(q12u16, d30u8, d2u8);
        d28u8 = vext_u8(d7u8, d8u8, 2);
        d29u8 = vext_u8(d10u8, d11u8, 2);
        d30u8 = vext_u8(d13u8, d14u8, 2);
        q9u16 = vmlal_u8(q9u16, d28u8, d2u8);
        q11u16 = vmlal_u8(q11u16, d29u8, d2u8);
        q13u16 = vmlal_u8(q13u16, d30u8, d2u8);

        d28u8 = vext_u8(d6u8, d7u8, 3);
        d29u8 = vext_u8(d9u8, d10u8, 3);
        d30u8 = vext_u8(d12u8, d13u8, 3);
        d15u8 = vext_u8(d7u8, d8u8, 3);
        d31u8 = vext_u8(d10u8, d11u8, 3);
        d6u8  = vext_u8(d13u8, d14u8, 3);
        q4u16 = vmull_u8(d28u8, d3u8);
        q5u16 = vmull_u8(d29u8, d3u8);
        q6u16 = vmull_u8(d30u8, d3u8);
        q4s16 = vreinterpretq_s16_u16(q4u16);
        q5s16 = vreinterpretq_s16_u16(q5u16);
        q6s16 = vreinterpretq_s16_u16(q6u16);
        q8s16 = vreinterpretq_s16_u16(q8u16);
        q10s16 = vreinterpretq_s16_u16(q10u16);
        q12s16 = vreinterpretq_s16_u16(q12u16);
        q8s16 = vqaddq_s16(q8s16, q4s16);
        q10s16 = vqaddq_s16(q10s16, q5s16);
        q12s16 = vqaddq_s16(q12s16, q6s16);

        q6u16 = vmull_u8(d15u8, d3u8);
        q7u16 = vmull_u8(d31u8, d3u8);
        q3u16 = vmull_u8(d6u8, d3u8);
        q3s16 = vreinterpretq_s16_u16(q3u16);
        q6s16 = vreinterpretq_s16_u16(q6u16);
        q7s16 = vreinterpretq_s16_u16(q7u16);
        q9s16 = vreinterpretq_s16_u16(q9u16);
        q11s16 = vreinterpretq_s16_u16(q11u16);
        q13s16 = vreinterpretq_s16_u16(q13u16);
        q9s16 = vqaddq_s16(q9s16, q6s16);
        q11s16 = vqaddq_s16(q11s16, q7s16);
        q13s16 = vqaddq_s16(q13s16, q3s16);

        d6u8 = vqrshrun_n_s16(q8s16, 7);
        d7u8 = vqrshrun_n_s16(q9s16, 7);
        d8u8 = vqrshrun_n_s16(q10s16, 7);
        d9u8 = vqrshrun_n_s16(q11s16, 7);
        d10u8 = vqrshrun_n_s16(q12s16, 7);
        d11u8 = vqrshrun_n_s16(q13s16, 7);

        vst1_u8(tmpp, d6u8);
        tmpp += 8;
        vst1_u8(tmpp, d7u8);
        tmpp += 8;
        vst1_u8(tmpp, d8u8);
        tmpp += 8;
        vst1_u8(tmpp, d9u8);
        tmpp += 8;
        vst1_u8(tmpp, d10u8);
        tmpp += 8;
        vst1_u8(tmpp, d11u8);
        tmpp += 8;
    }

    // Second pass: 16x16
    dtmps8 = vld1_s8(vp8_sub_pel_filters[yoffset]);
    d0s8 = vdup_lane_s8(dtmps8, 0);
    d1s8 = vdup_lane_s8(dtmps8, 1);
    d2s8 = vdup_lane_s8(dtmps8, 2);
    d3s8 = vdup_lane_s8(dtmps8, 3);
    d4s8 = vdup_lane_s8(dtmps8, 4);
    d5s8 = vdup_lane_s8(dtmps8, 5);
    d0u8 = vreinterpret_u8_s8(vabs_s8(d0s8));
    d1u8 = vreinterpret_u8_s8(vabs_s8(d1s8));
    d2u8 = vreinterpret_u8_s8(vabs_s8(d2s8));
    d3u8 = vreinterpret_u8_s8(vabs_s8(d3s8));
    d4u8 = vreinterpret_u8_s8(vabs_s8(d4s8));
    d5u8 = vreinterpret_u8_s8(vabs_s8(d5s8));

    for (i = 0; i < 2; i++) {
        dst = dst_ptr + 8 * i;
        tmpp = tmp + 8 * i;
        d18u8 = vld1_u8(tmpp);
        tmpp += 16;
        d19u8 = vld1_u8(tmpp);
        tmpp += 16;
        d20u8 = vld1_u8(tmpp);
        tmpp += 16;
        d21u8 = vld1_u8(tmpp);
        tmpp += 16;
        d22u8 = vld1_u8(tmpp);
        tmpp += 16;
        for (j = 0; j < 4; j++) {
            d23u8 = vld1_u8(tmpp);
            tmpp += 16;
            d24u8 = vld1_u8(tmpp);
            tmpp += 16;
            d25u8 = vld1_u8(tmpp);
            tmpp += 16;
            d26u8 = vld1_u8(tmpp);
            tmpp += 16;

            q3u16 = vmull_u8(d18u8, d0u8);
            q4u16 = vmull_u8(d19u8, d0u8);
            q5u16 = vmull_u8(d20u8, d0u8);
            q6u16 = vmull_u8(d21u8, d0u8);

            q3u16 = vmlsl_u8(q3u16, d19u8, d1u8);
            q4u16 = vmlsl_u8(q4u16, d20u8, d1u8);
            q5u16 = vmlsl_u8(q5u16, d21u8, d1u8);
            q6u16 = vmlsl_u8(q6u16, d22u8, d1u8);

            q3u16 = vmlsl_u8(q3u16, d22u8, d4u8);
            q4u16 = vmlsl_u8(q4u16, d23u8, d4u8);
            q5u16 = vmlsl_u8(q5u16, d24u8, d4u8);
            q6u16 = vmlsl_u8(q6u16, d25u8, d4u8);

            q3u16 = vmlal_u8(q3u16, d20u8, d2u8);
            q4u16 = vmlal_u8(q4u16, d21u8, d2u8);
            q5u16 = vmlal_u8(q5u16, d22u8, d2u8);
            q6u16 = vmlal_u8(q6u16, d23u8, d2u8);

            q3u16 = vmlal_u8(q3u16, d23u8, d5u8);
            q4u16 = vmlal_u8(q4u16, d24u8, d5u8);
            q5u16 = vmlal_u8(q5u16, d25u8, d5u8);
            q6u16 = vmlal_u8(q6u16, d26u8, d5u8);

            q7u16 = vmull_u8(d21u8, d3u8);
            q8u16 = vmull_u8(d22u8, d3u8);
            q9u16 = vmull_u8(d23u8, d3u8);
            q10u16 = vmull_u8(d24u8, d3u8);

            q3s16 = vreinterpretq_s16_u16(q3u16);
            q4s16 = vreinterpretq_s16_u16(q4u16);
            q5s16 = vreinterpretq_s16_u16(q5u16);
            q6s16 = vreinterpretq_s16_u16(q6u16);
            q7s16 = vreinterpretq_s16_u16(q7u16);
            q8s16 = vreinterpretq_s16_u16(q8u16);
            q9s16 = vreinterpretq_s16_u16(q9u16);
            q10s16 = vreinterpretq_s16_u16(q10u16);

            q7s16 = vqaddq_s16(q7s16, q3s16);
            q8s16 = vqaddq_s16(q8s16, q4s16);
            q9s16 = vqaddq_s16(q9s16, q5s16);
            q10s16 = vqaddq_s16(q10s16, q6s16);

            d6u8 = vqrshrun_n_s16(q7s16, 7);
            d7u8 = vqrshrun_n_s16(q8s16, 7);
            d8u8 = vqrshrun_n_s16(q9s16, 7);
            d9u8 = vqrshrun_n_s16(q10s16, 7);

            d18u8 = d22u8;
            d19u8 = d23u8;
            d20u8 = d24u8;
            d21u8 = d25u8;
            d22u8 = d26u8;

            vst1_u8(dst, d6u8);
            dst += dst_pitch;
            vst1_u8(dst, d7u8);
            dst += dst_pitch;
            vst1_u8(dst, d8u8);
            dst += dst_pitch;
            vst1_u8(dst, d9u8);
            dst += dst_pitch;
        }
    }
    return;
}
