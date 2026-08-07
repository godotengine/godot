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

#include "./vp9_rtcd.h"
#include "./vpx_config.h"
#include "vp9/common/vp9_common.h"
#include "vp9/common/arm/neon/vp9_iht_neon.h"
#include "vpx_dsp/arm/highbd_idct_neon.h"
#include "vpx_dsp/arm/idct_neon.h"
#include "vpx_dsp/arm/mem_neon.h"
#include "vpx_dsp/txfm_common.h"

static INLINE void highbd_iadst4(int32x4_t *const io) {
  const int32_t sinpis[4] = { sinpi_1_9, sinpi_2_9, sinpi_3_9, sinpi_4_9 };
  const int32x4_t sinpi = vld1q_s32(sinpis);
  int64x2x2_t s[7], t[4];
  int32x4_t s7;

  s[0].val[0] = vmull_lane_s32(vget_low_s32(io[0]), vget_low_s32(sinpi), 0);
  s[0].val[1] = vmull_lane_s32(vget_high_s32(io[0]), vget_low_s32(sinpi), 0);
  s[1].val[0] = vmull_lane_s32(vget_low_s32(io[0]), vget_low_s32(sinpi), 1);
  s[1].val[1] = vmull_lane_s32(vget_high_s32(io[0]), vget_low_s32(sinpi), 1);
  s[2].val[0] = vmull_lane_s32(vget_low_s32(io[1]), vget_high_s32(sinpi), 0);
  s[2].val[1] = vmull_lane_s32(vget_high_s32(io[1]), vget_high_s32(sinpi), 0);
  s[3].val[0] = vmull_lane_s32(vget_low_s32(io[2]), vget_high_s32(sinpi), 1);
  s[3].val[1] = vmull_lane_s32(vget_high_s32(io[2]), vget_high_s32(sinpi), 1);
  s[4].val[0] = vmull_lane_s32(vget_low_s32(io[2]), vget_low_s32(sinpi), 0);
  s[4].val[1] = vmull_lane_s32(vget_high_s32(io[2]), vget_low_s32(sinpi), 0);
  s[5].val[0] = vmull_lane_s32(vget_low_s32(io[3]), vget_low_s32(sinpi), 1);
  s[5].val[1] = vmull_lane_s32(vget_high_s32(io[3]), vget_low_s32(sinpi), 1);
  s[6].val[0] = vmull_lane_s32(vget_low_s32(io[3]), vget_high_s32(sinpi), 1);
  s[6].val[1] = vmull_lane_s32(vget_high_s32(io[3]), vget_high_s32(sinpi), 1);
  s7 = vsubq_s32(io[0], io[2]);
  s7 = vaddq_s32(s7, io[3]);

  s[0].val[0] = vaddq_s64(s[0].val[0], s[3].val[0]);
  s[0].val[1] = vaddq_s64(s[0].val[1], s[3].val[1]);
  s[0].val[0] = vaddq_s64(s[0].val[0], s[5].val[0]);
  s[0].val[1] = vaddq_s64(s[0].val[1], s[5].val[1]);
  s[1].val[0] = vsubq_s64(s[1].val[0], s[4].val[0]);
  s[1].val[1] = vsubq_s64(s[1].val[1], s[4].val[1]);
  s[1].val[0] = vsubq_s64(s[1].val[0], s[6].val[0]);
  s[1].val[1] = vsubq_s64(s[1].val[1], s[6].val[1]);
  s[3] = s[2];
  s[2].val[0] = vmull_lane_s32(vget_low_s32(s7), vget_high_s32(sinpi), 0);
  s[2].val[1] = vmull_lane_s32(vget_high_s32(s7), vget_high_s32(sinpi), 0);

  t[0].val[0] = vaddq_s64(s[0].val[0], s[3].val[0]);
  t[0].val[1] = vaddq_s64(s[0].val[1], s[3].val[1]);
  t[1].val[0] = vaddq_s64(s[1].val[0], s[3].val[0]);
  t[1].val[1] = vaddq_s64(s[1].val[1], s[3].val[1]);
  t[2] = s[2];
  t[3].val[0] = vaddq_s64(s[0].val[0], s[1].val[0]);
  t[3].val[1] = vaddq_s64(s[0].val[1], s[1].val[1]);
  t[3].val[0] = vsubq_s64(t[3].val[0], s[3].val[0]);
  t[3].val[1] = vsubq_s64(t[3].val[1], s[3].val[1]);
  io[0] = vcombine_s32(vrshrn_n_s64(t[0].val[0], DCT_CONST_BITS),
                       vrshrn_n_s64(t[0].val[1], DCT_CONST_BITS));
  io[1] = vcombine_s32(vrshrn_n_s64(t[1].val[0], DCT_CONST_BITS),
                       vrshrn_n_s64(t[1].val[1], DCT_CONST_BITS));
  io[2] = vcombine_s32(vrshrn_n_s64(t[2].val[0], DCT_CONST_BITS),
                       vrshrn_n_s64(t[2].val[1], DCT_CONST_BITS));
  io[3] = vcombine_s32(vrshrn_n_s64(t[3].val[0], DCT_CONST_BITS),
                       vrshrn_n_s64(t[3].val[1], DCT_CONST_BITS));
}

void vp9_highbd_iht4x4_16_add_neon(const tran_low_t *input, uint16_t *dest,
                                   int stride, int tx_type, int bd) {
  const int16x8_t max = vdupq_n_s16((1 << bd) - 1);
  int16x8_t a[2];
  int32x4_t c[4];

  c[0] = vld1q_s32(input);
  c[1] = vld1q_s32(input + 4);
  c[2] = vld1q_s32(input + 8);
  c[3] = vld1q_s32(input + 12);

  if (bd == 8) {
    a[0] = vcombine_s16(vmovn_s32(c[0]), vmovn_s32(c[1]));
    a[1] = vcombine_s16(vmovn_s32(c[2]), vmovn_s32(c[3]));
    transpose_s16_4x4q(&a[0], &a[1]);

    switch (tx_type) {
      case DCT_DCT:
        idct4x4_16_kernel_bd8(a);
        a[1] = vcombine_s16(vget_high_s16(a[1]), vget_low_s16(a[1]));
        transpose_s16_4x4q(&a[0], &a[1]);
        idct4x4_16_kernel_bd8(a);
        a[1] = vcombine_s16(vget_high_s16(a[1]), vget_low_s16(a[1]));
        break;

      case ADST_DCT:
        idct4x4_16_kernel_bd8(a);
        a[1] = vcombine_s16(vget_high_s16(a[1]), vget_low_s16(a[1]));
        transpose_s16_4x4q(&a[0], &a[1]);
        iadst4(a);
        break;

      case DCT_ADST:
        iadst4(a);
        transpose_s16_4x4q(&a[0], &a[1]);
        idct4x4_16_kernel_bd8(a);
        a[1] = vcombine_s16(vget_high_s16(a[1]), vget_low_s16(a[1]));
        break;

      default:
        assert(tx_type == ADST_ADST);
        iadst4(a);
        transpose_s16_4x4q(&a[0], &a[1]);
        iadst4(a);
        break;
    }
    a[0] = vrshrq_n_s16(a[0], 4);
    a[1] = vrshrq_n_s16(a[1], 4);
  } else {
    switch (tx_type) {
      case DCT_DCT: {
        const int32x4_t cospis = vld1q_s32(kCospi32);

        if (bd == 10) {
          idct4x4_16_kernel_bd10(cospis, c);
          idct4x4_16_kernel_bd10(cospis, c);
        } else {
          idct4x4_16_kernel_bd12(cospis, c);
          idct4x4_16_kernel_bd12(cospis, c);
        }
        break;
      }

      case ADST_DCT: {
        const int32x4_t cospis = vld1q_s32(kCospi32);

        if (bd == 10) {
          idct4x4_16_kernel_bd10(cospis, c);
        } else {
          idct4x4_16_kernel_bd12(cospis, c);
        }
        transpose_s32_4x4(&c[0], &c[1], &c[2], &c[3]);
        highbd_iadst4(c);
        break;
      }

      case DCT_ADST: {
        const int32x4_t cospis = vld1q_s32(kCospi32);

        transpose_s32_4x4(&c[0], &c[1], &c[2], &c[3]);
        highbd_iadst4(c);
        if (bd == 10) {
          idct4x4_16_kernel_bd10(cospis, c);
        } else {
          idct4x4_16_kernel_bd12(cospis, c);
        }
        break;
      }

      default: {
        assert(tx_type == ADST_ADST);
        transpose_s32_4x4(&c[0], &c[1], &c[2], &c[3]);
        highbd_iadst4(c);
        transpose_s32_4x4(&c[0], &c[1], &c[2], &c[3]);
        highbd_iadst4(c);
        break;
      }
    }
    a[0] = vcombine_s16(vqrshrn_n_s32(c[0], 4), vqrshrn_n_s32(c[1], 4));
    a[1] = vcombine_s16(vqrshrn_n_s32(c[2], 4), vqrshrn_n_s32(c[3], 4));
  }

  highbd_idct4x4_1_add_kernel1(&dest, stride, a[0], max);
  highbd_idct4x4_1_add_kernel1(&dest, stride, a[1], max);
}
