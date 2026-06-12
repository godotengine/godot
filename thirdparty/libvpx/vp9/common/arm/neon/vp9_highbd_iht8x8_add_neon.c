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

#include "./vpx_dsp_rtcd.h"
#include "vp9/common/vp9_enums.h"
#include "vp9/common/arm/neon/vp9_iht_neon.h"
#include "vpx_dsp/arm/highbd_idct_neon.h"
#include "vpx_dsp/arm/idct_neon.h"
#include "vpx_dsp/arm/transpose_neon.h"
#include "vpx_dsp/inv_txfm.h"

static INLINE void highbd_iadst_half_butterfly_neon(int32x4_t *const x,
                                                    const int32x2_t c) {
  const int32x4_t sum = vaddq_s32(x[0], x[1]);
  const int32x4_t sub = vsubq_s32(x[0], x[1]);
  const int64x2_t t0_lo = vmull_lane_s32(vget_low_s32(sum), c, 0);
  const int64x2_t t1_lo = vmull_lane_s32(vget_low_s32(sub), c, 0);
  const int64x2_t t0_hi = vmull_lane_s32(vget_high_s32(sum), c, 0);
  const int64x2_t t1_hi = vmull_lane_s32(vget_high_s32(sub), c, 0);
  const int32x2_t out0_lo = vrshrn_n_s64(t0_lo, DCT_CONST_BITS);
  const int32x2_t out1_lo = vrshrn_n_s64(t1_lo, DCT_CONST_BITS);
  const int32x2_t out0_hi = vrshrn_n_s64(t0_hi, DCT_CONST_BITS);
  const int32x2_t out1_hi = vrshrn_n_s64(t1_hi, DCT_CONST_BITS);

  x[0] = vcombine_s32(out0_lo, out0_hi);
  x[1] = vcombine_s32(out1_lo, out1_hi);
}

static INLINE void highbd_iadst_butterfly_lane_0_1_neon(const int32x4_t in0,
                                                        const int32x4_t in1,
                                                        const int32x2_t c,
                                                        int64x2_t *const s0,
                                                        int64x2_t *const s1) {
  const int64x2_t t0_lo = vmull_lane_s32(vget_low_s32(in0), c, 0);
  const int64x2_t t1_lo = vmull_lane_s32(vget_low_s32(in0), c, 1);
  const int64x2_t t0_hi = vmull_lane_s32(vget_high_s32(in0), c, 0);
  const int64x2_t t1_hi = vmull_lane_s32(vget_high_s32(in0), c, 1);

  s0[0] = vmlal_lane_s32(t0_lo, vget_low_s32(in1), c, 1);
  s1[0] = vmlsl_lane_s32(t1_lo, vget_low_s32(in1), c, 0);
  s0[1] = vmlal_lane_s32(t0_hi, vget_high_s32(in1), c, 1);
  s1[1] = vmlsl_lane_s32(t1_hi, vget_high_s32(in1), c, 0);
}

static INLINE void highbd_iadst_butterfly_lane_1_0_neon(const int32x4_t in0,
                                                        const int32x4_t in1,
                                                        const int32x2_t c,
                                                        int64x2_t *const s0,
                                                        int64x2_t *const s1) {
  const int64x2_t t0_lo = vmull_lane_s32(vget_low_s32(in0), c, 1);
  const int64x2_t t1_lo = vmull_lane_s32(vget_low_s32(in0), c, 0);
  const int64x2_t t0_hi = vmull_lane_s32(vget_high_s32(in0), c, 1);
  const int64x2_t t1_hi = vmull_lane_s32(vget_high_s32(in0), c, 0);

  s0[0] = vmlal_lane_s32(t0_lo, vget_low_s32(in1), c, 0);
  s1[0] = vmlsl_lane_s32(t1_lo, vget_low_s32(in1), c, 1);
  s0[1] = vmlal_lane_s32(t0_hi, vget_high_s32(in1), c, 0);
  s1[1] = vmlsl_lane_s32(t1_hi, vget_high_s32(in1), c, 1);
}

static INLINE int32x4_t highbd_add_dct_const_round_shift_low_8(
    const int64x2_t *const in0, const int64x2_t *const in1) {
  const int64x2_t sum_lo = vaddq_s64(in0[0], in1[0]);
  const int64x2_t sum_hi = vaddq_s64(in0[1], in1[1]);
  const int32x2_t out_lo = vrshrn_n_s64(sum_lo, DCT_CONST_BITS);
  const int32x2_t out_hi = vrshrn_n_s64(sum_hi, DCT_CONST_BITS);
  return vcombine_s32(out_lo, out_hi);
}

static INLINE int32x4_t highbd_sub_dct_const_round_shift_low_8(
    const int64x2_t *const in0, const int64x2_t *const in1) {
  const int64x2_t sub_lo = vsubq_s64(in0[0], in1[0]);
  const int64x2_t sub_hi = vsubq_s64(in0[1], in1[1]);
  const int32x2_t out_lo = vrshrn_n_s64(sub_lo, DCT_CONST_BITS);
  const int32x2_t out_hi = vrshrn_n_s64(sub_hi, DCT_CONST_BITS);
  return vcombine_s32(out_lo, out_hi);
}

static INLINE void highbd_iadst8(int32x4_t *const io0, int32x4_t *const io1,
                                 int32x4_t *const io2, int32x4_t *const io3,
                                 int32x4_t *const io4, int32x4_t *const io5,
                                 int32x4_t *const io6, int32x4_t *const io7) {
  const int32x4_t c0 =
      create_s32x4_neon(cospi_2_64, cospi_30_64, cospi_10_64, cospi_22_64);
  const int32x4_t c1 =
      create_s32x4_neon(cospi_18_64, cospi_14_64, cospi_26_64, cospi_6_64);
  const int32x4_t c2 =
      create_s32x4_neon(cospi_16_64, 0, cospi_8_64, cospi_24_64);
  int32x4_t x[8], t[4];
  int64x2_t s[8][2];

  x[0] = *io7;
  x[1] = *io0;
  x[2] = *io5;
  x[3] = *io2;
  x[4] = *io3;
  x[5] = *io4;
  x[6] = *io1;
  x[7] = *io6;

  // stage 1
  highbd_iadst_butterfly_lane_0_1_neon(x[0], x[1], vget_low_s32(c0), s[0],
                                       s[1]);
  highbd_iadst_butterfly_lane_0_1_neon(x[2], x[3], vget_high_s32(c0), s[2],
                                       s[3]);
  highbd_iadst_butterfly_lane_0_1_neon(x[4], x[5], vget_low_s32(c1), s[4],
                                       s[5]);
  highbd_iadst_butterfly_lane_0_1_neon(x[6], x[7], vget_high_s32(c1), s[6],
                                       s[7]);

  x[0] = highbd_add_dct_const_round_shift_low_8(s[0], s[4]);
  x[1] = highbd_add_dct_const_round_shift_low_8(s[1], s[5]);
  x[2] = highbd_add_dct_const_round_shift_low_8(s[2], s[6]);
  x[3] = highbd_add_dct_const_round_shift_low_8(s[3], s[7]);
  x[4] = highbd_sub_dct_const_round_shift_low_8(s[0], s[4]);
  x[5] = highbd_sub_dct_const_round_shift_low_8(s[1], s[5]);
  x[6] = highbd_sub_dct_const_round_shift_low_8(s[2], s[6]);
  x[7] = highbd_sub_dct_const_round_shift_low_8(s[3], s[7]);

  // stage 2
  t[0] = x[0];
  t[1] = x[1];
  t[2] = x[2];
  t[3] = x[3];
  highbd_iadst_butterfly_lane_0_1_neon(x[4], x[5], vget_high_s32(c2), s[4],
                                       s[5]);
  highbd_iadst_butterfly_lane_1_0_neon(x[7], x[6], vget_high_s32(c2), s[7],
                                       s[6]);

  x[0] = vaddq_s32(t[0], t[2]);
  x[1] = vaddq_s32(t[1], t[3]);
  x[2] = vsubq_s32(t[0], t[2]);
  x[3] = vsubq_s32(t[1], t[3]);
  x[4] = highbd_add_dct_const_round_shift_low_8(s[4], s[6]);
  x[5] = highbd_add_dct_const_round_shift_low_8(s[5], s[7]);
  x[6] = highbd_sub_dct_const_round_shift_low_8(s[4], s[6]);
  x[7] = highbd_sub_dct_const_round_shift_low_8(s[5], s[7]);

  // stage 3
  highbd_iadst_half_butterfly_neon(x + 2, vget_low_s32(c2));
  highbd_iadst_half_butterfly_neon(x + 6, vget_low_s32(c2));

  *io0 = x[0];
  *io1 = vnegq_s32(x[4]);
  *io2 = x[6];
  *io3 = vnegq_s32(x[2]);
  *io4 = x[3];
  *io5 = vnegq_s32(x[7]);
  *io6 = x[5];
  *io7 = vnegq_s32(x[1]);
}

void vp9_highbd_iht8x8_64_add_neon(const tran_low_t *input, uint16_t *dest,
                                   int stride, int tx_type, int bd) {
  int32x4_t a[16];
  int16x8_t c[8];

  a[0] = vld1q_s32(input);
  a[1] = vld1q_s32(input + 4);
  a[2] = vld1q_s32(input + 8);
  a[3] = vld1q_s32(input + 12);
  a[4] = vld1q_s32(input + 16);
  a[5] = vld1q_s32(input + 20);
  a[6] = vld1q_s32(input + 24);
  a[7] = vld1q_s32(input + 28);
  a[8] = vld1q_s32(input + 32);
  a[9] = vld1q_s32(input + 36);
  a[10] = vld1q_s32(input + 40);
  a[11] = vld1q_s32(input + 44);
  a[12] = vld1q_s32(input + 48);
  a[13] = vld1q_s32(input + 52);
  a[14] = vld1q_s32(input + 56);
  a[15] = vld1q_s32(input + 60);

  if (bd == 8) {
    c[0] = vcombine_s16(vmovn_s32(a[0]), vmovn_s32(a[1]));
    c[1] = vcombine_s16(vmovn_s32(a[2]), vmovn_s32(a[3]));
    c[2] = vcombine_s16(vmovn_s32(a[4]), vmovn_s32(a[5]));
    c[3] = vcombine_s16(vmovn_s32(a[6]), vmovn_s32(a[7]));
    c[4] = vcombine_s16(vmovn_s32(a[8]), vmovn_s32(a[9]));
    c[5] = vcombine_s16(vmovn_s32(a[10]), vmovn_s32(a[11]));
    c[6] = vcombine_s16(vmovn_s32(a[12]), vmovn_s32(a[13]));
    c[7] = vcombine_s16(vmovn_s32(a[14]), vmovn_s32(a[15]));

    switch (tx_type) {
      case DCT_DCT: {
        const int16x8_t cospis = vld1q_s16(kCospi);
        const int16x4_t cospis0 = vget_low_s16(cospis);   // cospi 0, 8, 16, 24
        const int16x4_t cospis1 = vget_high_s16(cospis);  // cospi 4, 12, 20, 28

        idct8x8_64_1d_bd8(cospis0, cospis1, c);
        idct8x8_64_1d_bd8(cospis0, cospis1, c);
        break;
      }

      case ADST_DCT: {
        const int16x8_t cospis = vld1q_s16(kCospi);
        const int16x4_t cospis0 = vget_low_s16(cospis);   // cospi 0, 8, 16, 24
        const int16x4_t cospis1 = vget_high_s16(cospis);  // cospi 4, 12, 20, 28

        idct8x8_64_1d_bd8(cospis0, cospis1, c);
        transpose_s16_8x8(&c[0], &c[1], &c[2], &c[3], &c[4], &c[5], &c[6],
                          &c[7]);
        iadst8(c);
        break;
      }

      case DCT_ADST: {
        const int16x8_t cospis = vld1q_s16(kCospi);
        const int16x4_t cospis0 = vget_low_s16(cospis);   // cospi 0, 8, 16, 24
        const int16x4_t cospis1 = vget_high_s16(cospis);  // cospi 4, 12, 20, 28

        transpose_s16_8x8(&c[0], &c[1], &c[2], &c[3], &c[4], &c[5], &c[6],
                          &c[7]);
        iadst8(c);
        idct8x8_64_1d_bd8(cospis0, cospis1, c);
        break;
      }

      default: {
        transpose_s16_8x8(&c[0], &c[1], &c[2], &c[3], &c[4], &c[5], &c[6],
                          &c[7]);
        iadst8(c);
        transpose_s16_8x8(&c[0], &c[1], &c[2], &c[3], &c[4], &c[5], &c[6],
                          &c[7]);
        iadst8(c);
        break;
      }
    }

    c[0] = vrshrq_n_s16(c[0], 5);
    c[1] = vrshrq_n_s16(c[1], 5);
    c[2] = vrshrq_n_s16(c[2], 5);
    c[3] = vrshrq_n_s16(c[3], 5);
    c[4] = vrshrq_n_s16(c[4], 5);
    c[5] = vrshrq_n_s16(c[5], 5);
    c[6] = vrshrq_n_s16(c[6], 5);
    c[7] = vrshrq_n_s16(c[7], 5);
  } else {
    switch (tx_type) {
      case DCT_DCT: {
        const int32x4_t cospis0 = vld1q_s32(kCospi32);  // cospi 0, 8, 16, 24
        const int32x4_t cospis1 =
            vld1q_s32(kCospi32 + 4);  // cospi 4, 12, 20, 28

        if (bd == 10) {
          idct8x8_64_half1d_bd10(cospis0, cospis1, &a[0], &a[1], &a[2], &a[3],
                                 &a[4], &a[5], &a[6], &a[7]);
          idct8x8_64_half1d_bd10(cospis0, cospis1, &a[8], &a[9], &a[10], &a[11],
                                 &a[12], &a[13], &a[14], &a[15]);
          idct8x8_64_half1d_bd10(cospis0, cospis1, &a[0], &a[8], &a[1], &a[9],
                                 &a[2], &a[10], &a[3], &a[11]);
          idct8x8_64_half1d_bd10(cospis0, cospis1, &a[4], &a[12], &a[5], &a[13],
                                 &a[6], &a[14], &a[7], &a[15]);
        } else {
          idct8x8_64_half1d_bd12(cospis0, cospis1, &a[0], &a[1], &a[2], &a[3],
                                 &a[4], &a[5], &a[6], &a[7]);
          idct8x8_64_half1d_bd12(cospis0, cospis1, &a[8], &a[9], &a[10], &a[11],
                                 &a[12], &a[13], &a[14], &a[15]);
          idct8x8_64_half1d_bd12(cospis0, cospis1, &a[0], &a[8], &a[1], &a[9],
                                 &a[2], &a[10], &a[3], &a[11]);
          idct8x8_64_half1d_bd12(cospis0, cospis1, &a[4], &a[12], &a[5], &a[13],
                                 &a[6], &a[14], &a[7], &a[15]);
        }
        break;
      }

      case ADST_DCT: {
        const int32x4_t cospis0 = vld1q_s32(kCospi32);  // cospi 0, 8, 16, 24
        const int32x4_t cospis1 =
            vld1q_s32(kCospi32 + 4);  // cospi 4, 12, 20, 28

        idct8x8_64_half1d_bd12(cospis0, cospis1, &a[0], &a[1], &a[2], &a[3],
                               &a[4], &a[5], &a[6], &a[7]);
        idct8x8_64_half1d_bd12(cospis0, cospis1, &a[8], &a[9], &a[10], &a[11],
                               &a[12], &a[13], &a[14], &a[15]);
        transpose_s32_8x4(&a[0], &a[8], &a[1], &a[9], &a[2], &a[10], &a[3],
                          &a[11]);
        highbd_iadst8(&a[0], &a[8], &a[1], &a[9], &a[2], &a[10], &a[3], &a[11]);
        transpose_s32_8x4(&a[4], &a[12], &a[5], &a[13], &a[6], &a[14], &a[7],
                          &a[15]);
        highbd_iadst8(&a[4], &a[12], &a[5], &a[13], &a[6], &a[14], &a[7],
                      &a[15]);
        break;
      }

      case DCT_ADST: {
        const int32x4_t cospis0 = vld1q_s32(kCospi32);  // cospi 0, 8, 16, 24
        const int32x4_t cospis1 =
            vld1q_s32(kCospi32 + 4);  // cospi 4, 12, 20, 28

        transpose_s32_8x4(&a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6],
                          &a[7]);
        highbd_iadst8(&a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6], &a[7]);
        transpose_s32_8x4(&a[8], &a[9], &a[10], &a[11], &a[12], &a[13], &a[14],
                          &a[15]);
        highbd_iadst8(&a[8], &a[9], &a[10], &a[11], &a[12], &a[13], &a[14],
                      &a[15]);
        idct8x8_64_half1d_bd12(cospis0, cospis1, &a[0], &a[8], &a[1], &a[9],
                               &a[2], &a[10], &a[3], &a[11]);
        idct8x8_64_half1d_bd12(cospis0, cospis1, &a[4], &a[12], &a[5], &a[13],
                               &a[6], &a[14], &a[7], &a[15]);
        break;
      }

      default: {
        assert(tx_type == ADST_ADST);
        transpose_s32_8x4(&a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6],
                          &a[7]);
        highbd_iadst8(&a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6], &a[7]);
        transpose_s32_8x4(&a[8], &a[9], &a[10], &a[11], &a[12], &a[13], &a[14],
                          &a[15]);
        highbd_iadst8(&a[8], &a[9], &a[10], &a[11], &a[12], &a[13], &a[14],
                      &a[15]);
        transpose_s32_8x4(&a[0], &a[8], &a[1], &a[9], &a[2], &a[10], &a[3],
                          &a[11]);
        highbd_iadst8(&a[0], &a[8], &a[1], &a[9], &a[2], &a[10], &a[3], &a[11]);
        transpose_s32_8x4(&a[4], &a[12], &a[5], &a[13], &a[6], &a[14], &a[7],
                          &a[15]);
        highbd_iadst8(&a[4], &a[12], &a[5], &a[13], &a[6], &a[14], &a[7],
                      &a[15]);
        break;
      }
    }

    c[0] = vcombine_s16(vrshrn_n_s32(a[0], 5), vrshrn_n_s32(a[4], 5));
    c[1] = vcombine_s16(vrshrn_n_s32(a[8], 5), vrshrn_n_s32(a[12], 5));
    c[2] = vcombine_s16(vrshrn_n_s32(a[1], 5), vrshrn_n_s32(a[5], 5));
    c[3] = vcombine_s16(vrshrn_n_s32(a[9], 5), vrshrn_n_s32(a[13], 5));
    c[4] = vcombine_s16(vrshrn_n_s32(a[2], 5), vrshrn_n_s32(a[6], 5));
    c[5] = vcombine_s16(vrshrn_n_s32(a[10], 5), vrshrn_n_s32(a[14], 5));
    c[6] = vcombine_s16(vrshrn_n_s32(a[3], 5), vrshrn_n_s32(a[7], 5));
    c[7] = vcombine_s16(vrshrn_n_s32(a[11], 5), vrshrn_n_s32(a[15], 5));
  }
  highbd_add8x8(c, dest, stride, bd);
}
