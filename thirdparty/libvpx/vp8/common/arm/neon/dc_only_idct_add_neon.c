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

void vp8_dc_only_idct_add_neon(
        int16_t input_dc,
        unsigned char *pred_ptr,
        int pred_stride,
        unsigned char *dst_ptr,
        int dst_stride) {
    int i;
    uint16_t a1 = ((input_dc + 4) >> 3);
    uint32x2_t d2u32 = vdup_n_u32(0);
    uint8x8_t d2u8;
    uint16x8_t q1u16;
    uint16x8_t qAdd;

    qAdd = vdupq_n_u16(a1);

    for (i = 0; i < 2; i++) {
        d2u32 = vld1_lane_u32((const uint32_t *)pred_ptr, d2u32, 0);
        pred_ptr += pred_stride;
        d2u32 = vld1_lane_u32((const uint32_t *)pred_ptr, d2u32, 1);
        pred_ptr += pred_stride;

        q1u16 = vaddw_u8(qAdd, vreinterpret_u8_u32(d2u32));
        d2u8 = vqmovun_s16(vreinterpretq_s16_u16(q1u16));

        vst1_lane_u32((uint32_t *)dst_ptr, vreinterpret_u32_u8(d2u8), 0);
        dst_ptr += dst_stride;
        vst1_lane_u32((uint32_t *)dst_ptr, vreinterpret_u32_u8(d2u8), 1);
        dst_ptr += dst_stride;
    }
}
