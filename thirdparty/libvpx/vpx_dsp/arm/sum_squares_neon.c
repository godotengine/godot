/*
 *  Copyright (c) 2018 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <arm_neon.h>
#include <assert.h>

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/arm/sum_neon.h"

uint64_t vpx_sum_squares_2d_i16_neon(const int16_t *src, int stride, int size) {
  if (size == 4) {
    int16x4_t s[4];
    int32x4_t sum_s32;

    s[0] = vld1_s16(src + 0 * stride);
    s[1] = vld1_s16(src + 1 * stride);
    s[2] = vld1_s16(src + 2 * stride);
    s[3] = vld1_s16(src + 3 * stride);

    sum_s32 = vmull_s16(s[0], s[0]);
    sum_s32 = vmlal_s16(sum_s32, s[1], s[1]);
    sum_s32 = vmlal_s16(sum_s32, s[2], s[2]);
    sum_s32 = vmlal_s16(sum_s32, s[3], s[3]);

    return horizontal_long_add_uint32x4(vreinterpretq_u32_s32(sum_s32));
  } else {
    uint64x2_t sum_u64 = vdupq_n_u64(0);
    int rows = size;

    do {
      const int16_t *src_ptr = src;
      int32x4_t sum_s32[2] = { vdupq_n_s32(0), vdupq_n_s32(0) };
      int cols = size;

      do {
        int16x8_t s[8];

        s[0] = vld1q_s16(src_ptr + 0 * stride);
        s[1] = vld1q_s16(src_ptr + 1 * stride);
        s[2] = vld1q_s16(src_ptr + 2 * stride);
        s[3] = vld1q_s16(src_ptr + 3 * stride);
        s[4] = vld1q_s16(src_ptr + 4 * stride);
        s[5] = vld1q_s16(src_ptr + 5 * stride);
        s[6] = vld1q_s16(src_ptr + 6 * stride);
        s[7] = vld1q_s16(src_ptr + 7 * stride);

        sum_s32[0] =
            vmlal_s16(sum_s32[0], vget_low_s16(s[0]), vget_low_s16(s[0]));
        sum_s32[0] =
            vmlal_s16(sum_s32[0], vget_low_s16(s[1]), vget_low_s16(s[1]));
        sum_s32[0] =
            vmlal_s16(sum_s32[0], vget_low_s16(s[2]), vget_low_s16(s[2]));
        sum_s32[0] =
            vmlal_s16(sum_s32[0], vget_low_s16(s[3]), vget_low_s16(s[3]));
        sum_s32[0] =
            vmlal_s16(sum_s32[0], vget_low_s16(s[4]), vget_low_s16(s[4]));
        sum_s32[0] =
            vmlal_s16(sum_s32[0], vget_low_s16(s[5]), vget_low_s16(s[5]));
        sum_s32[0] =
            vmlal_s16(sum_s32[0], vget_low_s16(s[6]), vget_low_s16(s[6]));
        sum_s32[0] =
            vmlal_s16(sum_s32[0], vget_low_s16(s[7]), vget_low_s16(s[7]));

        sum_s32[1] =
            vmlal_s16(sum_s32[1], vget_high_s16(s[0]), vget_high_s16(s[0]));
        sum_s32[1] =
            vmlal_s16(sum_s32[1], vget_high_s16(s[1]), vget_high_s16(s[1]));
        sum_s32[1] =
            vmlal_s16(sum_s32[1], vget_high_s16(s[2]), vget_high_s16(s[2]));
        sum_s32[1] =
            vmlal_s16(sum_s32[1], vget_high_s16(s[3]), vget_high_s16(s[3]));
        sum_s32[1] =
            vmlal_s16(sum_s32[1], vget_high_s16(s[4]), vget_high_s16(s[4]));
        sum_s32[1] =
            vmlal_s16(sum_s32[1], vget_high_s16(s[5]), vget_high_s16(s[5]));
        sum_s32[1] =
            vmlal_s16(sum_s32[1], vget_high_s16(s[6]), vget_high_s16(s[6]));
        sum_s32[1] =
            vmlal_s16(sum_s32[1], vget_high_s16(s[7]), vget_high_s16(s[7]));

        src_ptr += 8;
        cols -= 8;
      } while (cols);

      sum_u64 = vpadalq_u32(sum_u64, vreinterpretq_u32_s32(sum_s32[0]));
      sum_u64 = vpadalq_u32(sum_u64, vreinterpretq_u32_s32(sum_s32[1]));
      src += 8 * stride;
      rows -= 8;
    } while (rows);

    return horizontal_add_uint64x2(sum_u64);
  }
}
