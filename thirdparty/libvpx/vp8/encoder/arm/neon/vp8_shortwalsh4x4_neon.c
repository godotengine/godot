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

#include "./vp8_rtcd.h"
#include "vpx_ports/arm.h"

#ifdef VPX_INCOMPATIBLE_GCC
#include "./vp8_rtcd.h"
void vp8_short_walsh4x4_neon(int16_t *input, int16_t *output, int pitch) {
  vp8_short_walsh4x4_c(input, output, pitch);
}
#else
void vp8_short_walsh4x4_neon(int16_t *input, int16_t *output, int pitch) {
  uint16x4_t d16u16;
  int16x8_t q0s16, q1s16;
  int16x4_t dEmptys16, d0s16, d1s16, d2s16, d3s16, d4s16, d5s16, d6s16, d7s16;
  int32x4_t qEmptys32, q0s32, q1s32, q2s32, q3s32, q8s32;
  int32x4_t q9s32, q10s32, q11s32, q15s32;
  uint32x4_t q8u32, q9u32, q10u32, q11u32;
  int16x4x2_t v2tmp0, v2tmp1;
  int32x2x2_t v2tmp2, v2tmp3;

  dEmptys16 = vdup_n_s16(0);
  qEmptys32 = vdupq_n_s32(0);
  q15s32 = vdupq_n_s32(3);

  d0s16 = vld1_s16(input);
  input += pitch / 2;
  d1s16 = vld1_s16(input);
  input += pitch / 2;
  d2s16 = vld1_s16(input);
  input += pitch / 2;
  d3s16 = vld1_s16(input);

  v2tmp2 = vtrn_s32(vreinterpret_s32_s16(d0s16), vreinterpret_s32_s16(d2s16));
  v2tmp3 = vtrn_s32(vreinterpret_s32_s16(d1s16), vreinterpret_s32_s16(d3s16));
  v2tmp0 = vtrn_s16(vreinterpret_s16_s32(v2tmp2.val[0]),   // d0
                    vreinterpret_s16_s32(v2tmp3.val[0]));  // d1
  v2tmp1 = vtrn_s16(vreinterpret_s16_s32(v2tmp2.val[1]),   // d2
                    vreinterpret_s16_s32(v2tmp3.val[1]));  // d3

  d4s16 = vadd_s16(v2tmp0.val[0], v2tmp1.val[0]);
  d5s16 = vadd_s16(v2tmp0.val[1], v2tmp1.val[1]);
  d6s16 = vsub_s16(v2tmp0.val[1], v2tmp1.val[1]);
  d7s16 = vsub_s16(v2tmp0.val[0], v2tmp1.val[0]);

  d4s16 = vshl_n_s16(d4s16, 2);
  d5s16 = vshl_n_s16(d5s16, 2);
  d6s16 = vshl_n_s16(d6s16, 2);
  d7s16 = vshl_n_s16(d7s16, 2);

  d16u16 = vceq_s16(d4s16, dEmptys16);
  d16u16 = vmvn_u16(d16u16);

  d0s16 = vadd_s16(d4s16, d5s16);
  d3s16 = vsub_s16(d4s16, d5s16);
  d1s16 = vadd_s16(d7s16, d6s16);
  d2s16 = vsub_s16(d7s16, d6s16);

  d0s16 = vsub_s16(d0s16, vreinterpret_s16_u16(d16u16));

  // Second for-loop
  v2tmp2 = vtrn_s32(vreinterpret_s32_s16(d1s16), vreinterpret_s32_s16(d3s16));
  v2tmp3 = vtrn_s32(vreinterpret_s32_s16(d0s16), vreinterpret_s32_s16(d2s16));
  v2tmp0 = vtrn_s16(vreinterpret_s16_s32(v2tmp3.val[1]),   // d2
                    vreinterpret_s16_s32(v2tmp2.val[1]));  // d3
  v2tmp1 = vtrn_s16(vreinterpret_s16_s32(v2tmp3.val[0]),   // d0
                    vreinterpret_s16_s32(v2tmp2.val[0]));  // d1

  q8s32 = vaddl_s16(v2tmp1.val[0], v2tmp0.val[0]);
  q9s32 = vaddl_s16(v2tmp1.val[1], v2tmp0.val[1]);
  q10s32 = vsubl_s16(v2tmp1.val[1], v2tmp0.val[1]);
  q11s32 = vsubl_s16(v2tmp1.val[0], v2tmp0.val[0]);

  q0s32 = vaddq_s32(q8s32, q9s32);
  q1s32 = vaddq_s32(q11s32, q10s32);
  q2s32 = vsubq_s32(q11s32, q10s32);
  q3s32 = vsubq_s32(q8s32, q9s32);

  q8u32 = vcltq_s32(q0s32, qEmptys32);
  q9u32 = vcltq_s32(q1s32, qEmptys32);
  q10u32 = vcltq_s32(q2s32, qEmptys32);
  q11u32 = vcltq_s32(q3s32, qEmptys32);

  q8s32 = vreinterpretq_s32_u32(q8u32);
  q9s32 = vreinterpretq_s32_u32(q9u32);
  q10s32 = vreinterpretq_s32_u32(q10u32);
  q11s32 = vreinterpretq_s32_u32(q11u32);

  q0s32 = vsubq_s32(q0s32, q8s32);
  q1s32 = vsubq_s32(q1s32, q9s32);
  q2s32 = vsubq_s32(q2s32, q10s32);
  q3s32 = vsubq_s32(q3s32, q11s32);

  q8s32 = vaddq_s32(q0s32, q15s32);
  q9s32 = vaddq_s32(q1s32, q15s32);
  q10s32 = vaddq_s32(q2s32, q15s32);
  q11s32 = vaddq_s32(q3s32, q15s32);

  d0s16 = vshrn_n_s32(q8s32, 3);
  d1s16 = vshrn_n_s32(q9s32, 3);
  d2s16 = vshrn_n_s32(q10s32, 3);
  d3s16 = vshrn_n_s32(q11s32, 3);

  q0s16 = vcombine_s16(d0s16, d1s16);
  q1s16 = vcombine_s16(d2s16, d3s16);

  vst1q_s16(output, q0s16);
  vst1q_s16(output + 8, q1s16);
  return;
}
#endif  // VPX_INCOMPATIBLE_GCC
