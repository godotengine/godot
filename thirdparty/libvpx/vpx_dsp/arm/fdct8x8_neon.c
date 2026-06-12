/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <arm_neon.h>

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/txfm_common.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_dsp/arm/idct_neon.h"
#include "vpx_dsp/arm/fdct_neon.h"
#include "vpx_dsp/arm/mem_neon.h"
#include "vpx_dsp/arm/fdct8x8_neon.h"

void vpx_fdct8x8_neon(const int16_t *input, tran_low_t *final_output,
                      int stride) {
  // stage 1
  int16x8_t in[8];
  in[0] = vshlq_n_s16(vld1q_s16(&input[0 * stride]), 2);
  in[1] = vshlq_n_s16(vld1q_s16(&input[1 * stride]), 2);
  in[2] = vshlq_n_s16(vld1q_s16(&input[2 * stride]), 2);
  in[3] = vshlq_n_s16(vld1q_s16(&input[3 * stride]), 2);
  in[4] = vshlq_n_s16(vld1q_s16(&input[4 * stride]), 2);
  in[5] = vshlq_n_s16(vld1q_s16(&input[5 * stride]), 2);
  in[6] = vshlq_n_s16(vld1q_s16(&input[6 * stride]), 2);
  in[7] = vshlq_n_s16(vld1q_s16(&input[7 * stride]), 2);

  vpx_fdct8x8_pass1_neon(in);
  vpx_fdct8x8_pass2_neon(in);
  {
    // from vpx_dct_sse2.c
    // Post-condition (division by two)
    //    division of two 16 bits signed numbers using shifts
    //    n / 2 = (n - (n >> 15)) >> 1
    const int16x8_t sign_in0 = vshrq_n_s16(in[0], 15);
    const int16x8_t sign_in1 = vshrq_n_s16(in[1], 15);
    const int16x8_t sign_in2 = vshrq_n_s16(in[2], 15);
    const int16x8_t sign_in3 = vshrq_n_s16(in[3], 15);
    const int16x8_t sign_in4 = vshrq_n_s16(in[4], 15);
    const int16x8_t sign_in5 = vshrq_n_s16(in[5], 15);
    const int16x8_t sign_in6 = vshrq_n_s16(in[6], 15);
    const int16x8_t sign_in7 = vshrq_n_s16(in[7], 15);
    in[0] = vhsubq_s16(in[0], sign_in0);
    in[1] = vhsubq_s16(in[1], sign_in1);
    in[2] = vhsubq_s16(in[2], sign_in2);
    in[3] = vhsubq_s16(in[3], sign_in3);
    in[4] = vhsubq_s16(in[4], sign_in4);
    in[5] = vhsubq_s16(in[5], sign_in5);
    in[6] = vhsubq_s16(in[6], sign_in6);
    in[7] = vhsubq_s16(in[7], sign_in7);
    // store results
    store_s16q_to_tran_low(final_output + 0 * 8, in[0]);
    store_s16q_to_tran_low(final_output + 1 * 8, in[1]);
    store_s16q_to_tran_low(final_output + 2 * 8, in[2]);
    store_s16q_to_tran_low(final_output + 3 * 8, in[3]);
    store_s16q_to_tran_low(final_output + 4 * 8, in[4]);
    store_s16q_to_tran_low(final_output + 5 * 8, in[5]);
    store_s16q_to_tran_low(final_output + 6 * 8, in[6]);
    store_s16q_to_tran_low(final_output + 7 * 8, in[7]);
  }
}

#if CONFIG_VP9_HIGHBITDEPTH

void vpx_highbd_fdct8x8_neon(const int16_t *input, tran_low_t *final_output,
                             int stride) {
  // input[M * stride] * 16
  int32x4_t left[8], right[8];
  int16x8_t in[8];
  in[0] = vld1q_s16(input + 0 * stride);
  in[1] = vld1q_s16(input + 1 * stride);
  in[2] = vld1q_s16(input + 2 * stride);
  in[3] = vld1q_s16(input + 3 * stride);
  in[4] = vld1q_s16(input + 4 * stride);
  in[5] = vld1q_s16(input + 5 * stride);
  in[6] = vld1q_s16(input + 6 * stride);
  in[7] = vld1q_s16(input + 7 * stride);

  left[0] = vshll_n_s16(vget_low_s16(in[0]), 2);
  left[1] = vshll_n_s16(vget_low_s16(in[1]), 2);
  left[2] = vshll_n_s16(vget_low_s16(in[2]), 2);
  left[3] = vshll_n_s16(vget_low_s16(in[3]), 2);
  left[4] = vshll_n_s16(vget_low_s16(in[4]), 2);
  left[5] = vshll_n_s16(vget_low_s16(in[5]), 2);
  left[6] = vshll_n_s16(vget_low_s16(in[6]), 2);
  left[7] = vshll_n_s16(vget_low_s16(in[7]), 2);
  right[0] = vshll_n_s16(vget_high_s16(in[0]), 2);
  right[1] = vshll_n_s16(vget_high_s16(in[1]), 2);
  right[2] = vshll_n_s16(vget_high_s16(in[2]), 2);
  right[3] = vshll_n_s16(vget_high_s16(in[3]), 2);
  right[4] = vshll_n_s16(vget_high_s16(in[4]), 2);
  right[5] = vshll_n_s16(vget_high_s16(in[5]), 2);
  right[6] = vshll_n_s16(vget_high_s16(in[6]), 2);
  right[7] = vshll_n_s16(vget_high_s16(in[7]), 2);

  vpx_highbd_fdct8x8_pass1_neon(left, right);
  vpx_highbd_fdct8x8_pass2_neon(left, right);
  {
    left[0] = add_round_shift_half_s32(left[0]);
    left[1] = add_round_shift_half_s32(left[1]);
    left[2] = add_round_shift_half_s32(left[2]);
    left[3] = add_round_shift_half_s32(left[3]);
    left[4] = add_round_shift_half_s32(left[4]);
    left[5] = add_round_shift_half_s32(left[5]);
    left[6] = add_round_shift_half_s32(left[6]);
    left[7] = add_round_shift_half_s32(left[7]);
    right[0] = add_round_shift_half_s32(right[0]);
    right[1] = add_round_shift_half_s32(right[1]);
    right[2] = add_round_shift_half_s32(right[2]);
    right[3] = add_round_shift_half_s32(right[3]);
    right[4] = add_round_shift_half_s32(right[4]);
    right[5] = add_round_shift_half_s32(right[5]);
    right[6] = add_round_shift_half_s32(right[6]);
    right[7] = add_round_shift_half_s32(right[7]);

    // store results
    vst1q_s32(final_output, left[0]);
    vst1q_s32(final_output + 4, right[0]);
    vst1q_s32(final_output + 8, left[1]);
    vst1q_s32(final_output + 12, right[1]);
    vst1q_s32(final_output + 16, left[2]);
    vst1q_s32(final_output + 20, right[2]);
    vst1q_s32(final_output + 24, left[3]);
    vst1q_s32(final_output + 28, right[3]);
    vst1q_s32(final_output + 32, left[4]);
    vst1q_s32(final_output + 36, right[4]);
    vst1q_s32(final_output + 40, left[5]);
    vst1q_s32(final_output + 44, right[5]);
    vst1q_s32(final_output + 48, left[6]);
    vst1q_s32(final_output + 52, right[6]);
    vst1q_s32(final_output + 56, left[7]);
    vst1q_s32(final_output + 60, right[7]);
  }
}

#endif  // CONFIG_VP9_HIGHBITDEPTH
