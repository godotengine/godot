/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
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
#include "vpx_dsp/arm/mem_neon.h"
#include "vpx_dsp/arm/sum_neon.h"

void vpx_fdct4x4_1_neon(const int16_t *input, tran_low_t *output, int stride) {
  int16x4_t a0, a1, a2, a3;
  int16x8_t b0, b1;
  int16x8_t c;

  a0 = vld1_s16(input);
  input += stride;
  a1 = vld1_s16(input);
  input += stride;
  a2 = vld1_s16(input);
  input += stride;
  a3 = vld1_s16(input);

  b0 = vcombine_s16(a0, a1);
  b1 = vcombine_s16(a2, a3);

  c = vaddq_s16(b0, b1);

  output[0] = (tran_low_t)(horizontal_add_int16x8(c) << 1);
  output[1] = 0;
}

// Visual Studio 2022 (cl.exe) < 17.7 targeting AArch64 with optimizations
// enabled will fail with an internal compiler error. See:
// https://developercommunity.visualstudio.com/t/Compiler-crash-C1001-when-building-a-for/10346110
#if defined(_MSC_VER) && _MSC_VER < 1937 && defined(_M_ARM64) && \
    !defined(__clang__)
#define AOM_WORK_AROUND_MSVC_BUG_10346110
#endif

#ifdef AOM_WORK_AROUND_MSVC_BUG_10346110
#pragma optimize("", off)
#endif
void vpx_fdct8x8_1_neon(const int16_t *input, tran_low_t *output, int stride) {
  int r;
  int16x8_t sum = vld1q_s16(&input[0]);

  for (r = 1; r < 8; ++r) {
    const int16x8_t input_00 = vld1q_s16(&input[r * stride]);
    sum = vaddq_s16(sum, input_00);
  }

  output[0] = (tran_low_t)horizontal_add_int16x8(sum);
  output[1] = 0;
}
#ifdef AOM_WORK_AROUND_MSVC_BUG_10346110
#pragma optimize("", on)
#endif
#undef AOM_WORK_AROUND_MSVC_BUG_10346110

void vpx_fdct16x16_1_neon(const int16_t *input, tran_low_t *output,
                          int stride) {
  int r;
  int16x8_t left = vld1q_s16(input);
  int16x8_t right = vld1q_s16(input + 8);
  int32_t sum;
  input += stride;

  for (r = 1; r < 16; ++r) {
    const int16x8_t a = vld1q_s16(input);
    const int16x8_t b = vld1q_s16(input + 8);
    input += stride;
    left = vaddq_s16(left, a);
    right = vaddq_s16(right, b);
  }

  sum = horizontal_add_int16x8(left) + horizontal_add_int16x8(right);

  output[0] = (tran_low_t)(sum >> 1);
  output[1] = 0;
}

void vpx_fdct32x32_1_neon(const int16_t *input, tran_low_t *output,
                          int stride) {
  int r;
  int16x8_t a0 = vld1q_s16(input);
  int16x8_t a1 = vld1q_s16(input + 8);
  int16x8_t a2 = vld1q_s16(input + 16);
  int16x8_t a3 = vld1q_s16(input + 24);
  int32_t sum;
  input += stride;

  for (r = 1; r < 32; ++r) {
    const int16x8_t b0 = vld1q_s16(input);
    const int16x8_t b1 = vld1q_s16(input + 8);
    const int16x8_t b2 = vld1q_s16(input + 16);
    const int16x8_t b3 = vld1q_s16(input + 24);
    input += stride;
    a0 = vaddq_s16(a0, b0);
    a1 = vaddq_s16(a1, b1);
    a2 = vaddq_s16(a2, b2);
    a3 = vaddq_s16(a3, b3);
  }

  sum = horizontal_add_int16x8(a0);
  sum += horizontal_add_int16x8(a1);
  sum += horizontal_add_int16x8(a2);
  sum += horizontal_add_int16x8(a3);
  output[0] = (tran_low_t)(sum >> 3);
  output[1] = 0;
}

#if CONFIG_VP9_HIGHBITDEPTH

void vpx_highbd_fdct16x16_1_neon(const int16_t *input, tran_low_t *output,
                                 int stride) {
  int32x4_t partial_sum[4] = { vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
                               vdupq_n_s32(0) };
  int32_t sum;

  int r = 0;
  do {
    const int16x8_t a = vld1q_s16(input);
    const int16x8_t b = vld1q_s16(input + 8);
    input += stride;
    partial_sum[0] = vaddw_s16(partial_sum[0], vget_low_s16(a));
    partial_sum[1] = vaddw_s16(partial_sum[1], vget_high_s16(a));
    partial_sum[2] = vaddw_s16(partial_sum[2], vget_low_s16(b));
    partial_sum[3] = vaddw_s16(partial_sum[3], vget_high_s16(b));
    r++;
  } while (r < 16);

  partial_sum[0] = vaddq_s32(partial_sum[0], partial_sum[1]);
  partial_sum[2] = vaddq_s32(partial_sum[2], partial_sum[3]);
  partial_sum[0] = vaddq_s32(partial_sum[0], partial_sum[2]);
  sum = horizontal_add_int32x4(partial_sum[0]);

  output[0] = (tran_low_t)(sum >> 1);
  output[1] = 0;
}

void vpx_highbd_fdct32x32_1_neon(const int16_t *input, tran_low_t *output,
                                 int stride) {
  int32x4_t partial_sum[4] = { vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
                               vdupq_n_s32(0) };

  int32_t sum;

  int r = 0;
  do {
    const int16x8_t a0 = vld1q_s16(input);
    const int16x8_t a1 = vld1q_s16(input + 8);
    const int16x8_t a2 = vld1q_s16(input + 16);
    const int16x8_t a3 = vld1q_s16(input + 24);
    input += stride;
    partial_sum[0] = vaddw_s16(partial_sum[0], vget_low_s16(a0));
    partial_sum[0] = vaddw_s16(partial_sum[0], vget_high_s16(a0));
    partial_sum[1] = vaddw_s16(partial_sum[1], vget_low_s16(a1));
    partial_sum[1] = vaddw_s16(partial_sum[1], vget_high_s16(a1));
    partial_sum[2] = vaddw_s16(partial_sum[2], vget_low_s16(a2));
    partial_sum[2] = vaddw_s16(partial_sum[2], vget_high_s16(a2));
    partial_sum[3] = vaddw_s16(partial_sum[3], vget_low_s16(a3));
    partial_sum[3] = vaddw_s16(partial_sum[3], vget_high_s16(a3));
    r++;
  } while (r < 32);

  partial_sum[0] = vaddq_s32(partial_sum[0], partial_sum[1]);
  partial_sum[2] = vaddq_s32(partial_sum[2], partial_sum[3]);
  partial_sum[0] = vaddq_s32(partial_sum[0], partial_sum[2]);
  sum = horizontal_add_int32x4(partial_sum[0]);

  output[0] = (tran_low_t)(sum >> 3);
  output[1] = 0;
}

#endif  // CONFIG_VP9_HIGHBITDEPTH
