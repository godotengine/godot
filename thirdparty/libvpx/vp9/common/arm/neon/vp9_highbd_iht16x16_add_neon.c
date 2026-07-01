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

// Use macros to make sure argument lane is passed in as a constant integer.

#define vmull_lane_s32_dual(in, c, lane, out)                          \
  do {                                                                 \
    out[0].val[0] = vmull_lane_s32(vget_low_s32(in.val[0]), c, lane);  \
    out[0].val[1] = vmull_lane_s32(vget_low_s32(in.val[1]), c, lane);  \
    out[1].val[0] = vmull_lane_s32(vget_high_s32(in.val[0]), c, lane); \
    out[1].val[1] = vmull_lane_s32(vget_high_s32(in.val[1]), c, lane); \
  } while (0)

#define vmlal_lane_s32_dual(in, c, lane, out)                             \
  do {                                                                    \
    out[0].val[0] =                                                       \
        vmlal_lane_s32(out[0].val[0], vget_low_s32(in.val[0]), c, lane);  \
    out[0].val[1] =                                                       \
        vmlal_lane_s32(out[0].val[1], vget_low_s32(in.val[1]), c, lane);  \
    out[1].val[0] =                                                       \
        vmlal_lane_s32(out[1].val[0], vget_high_s32(in.val[0]), c, lane); \
    out[1].val[1] =                                                       \
        vmlal_lane_s32(out[1].val[1], vget_high_s32(in.val[1]), c, lane); \
  } while (0)

#define vmlsl_lane_s32_dual(in, c, lane, out)                             \
  do {                                                                    \
    out[0].val[0] =                                                       \
        vmlsl_lane_s32(out[0].val[0], vget_low_s32(in.val[0]), c, lane);  \
    out[0].val[1] =                                                       \
        vmlsl_lane_s32(out[0].val[1], vget_low_s32(in.val[1]), c, lane);  \
    out[1].val[0] =                                                       \
        vmlsl_lane_s32(out[1].val[0], vget_high_s32(in.val[0]), c, lane); \
    out[1].val[1] =                                                       \
        vmlsl_lane_s32(out[1].val[1], vget_high_s32(in.val[1]), c, lane); \
  } while (0)

static INLINE int32x4x2_t
highbd_dct_const_round_shift_low_8(const int64x2x2_t *const in) {
  int32x4x2_t out;
  out.val[0] = vcombine_s32(vrshrn_n_s64(in[0].val[0], DCT_CONST_BITS),
                            vrshrn_n_s64(in[1].val[0], DCT_CONST_BITS));
  out.val[1] = vcombine_s32(vrshrn_n_s64(in[0].val[1], DCT_CONST_BITS),
                            vrshrn_n_s64(in[1].val[1], DCT_CONST_BITS));
  return out;
}

#define highbd_iadst_half_butterfly(in, c, lane, out) \
  do {                                                \
    int64x2x2_t _t[2];                                \
    vmull_lane_s32_dual(in, c, lane, _t);             \
    out = highbd_dct_const_round_shift_low_8(_t);     \
  } while (0)

#define highbd_iadst_butterfly(in0, in1, c, lane0, lane1, s0, s1) \
  do {                                                            \
    vmull_lane_s32_dual(in0, c, lane0, s0);                       \
    vmull_lane_s32_dual(in0, c, lane1, s1);                       \
    vmlal_lane_s32_dual(in1, c, lane1, s0);                       \
    vmlsl_lane_s32_dual(in1, c, lane0, s1);                       \
  } while (0)

static INLINE int32x4x2_t vaddq_s32_dual(const int32x4x2_t in0,
                                         const int32x4x2_t in1) {
  int32x4x2_t out;
  out.val[0] = vaddq_s32(in0.val[0], in1.val[0]);
  out.val[1] = vaddq_s32(in0.val[1], in1.val[1]);
  return out;
}

static INLINE int64x2x2_t vaddq_s64_dual(const int64x2x2_t in0,
                                         const int64x2x2_t in1) {
  int64x2x2_t out;
  out.val[0] = vaddq_s64(in0.val[0], in1.val[0]);
  out.val[1] = vaddq_s64(in0.val[1], in1.val[1]);
  return out;
}

static INLINE int32x4x2_t vsubq_s32_dual(const int32x4x2_t in0,
                                         const int32x4x2_t in1) {
  int32x4x2_t out;
  out.val[0] = vsubq_s32(in0.val[0], in1.val[0]);
  out.val[1] = vsubq_s32(in0.val[1], in1.val[1]);
  return out;
}

static INLINE int64x2x2_t vsubq_s64_dual(const int64x2x2_t in0,
                                         const int64x2x2_t in1) {
  int64x2x2_t out;
  out.val[0] = vsubq_s64(in0.val[0], in1.val[0]);
  out.val[1] = vsubq_s64(in0.val[1], in1.val[1]);
  return out;
}

static INLINE int32x4x2_t vcombine_s32_dual(const int32x2x2_t in0,
                                            const int32x2x2_t in1) {
  int32x4x2_t out;
  out.val[0] = vcombine_s32(in0.val[0], in1.val[0]);
  out.val[1] = vcombine_s32(in0.val[1], in1.val[1]);
  return out;
}

static INLINE int32x4x2_t highbd_add_dct_const_round_shift_low_8(
    const int64x2x2_t *const in0, const int64x2x2_t *const in1) {
  const int64x2x2_t sum_lo = vaddq_s64_dual(in0[0], in1[0]);
  const int64x2x2_t sum_hi = vaddq_s64_dual(in0[1], in1[1]);
  int32x2x2_t out_lo, out_hi;

  out_lo.val[0] = vrshrn_n_s64(sum_lo.val[0], DCT_CONST_BITS);
  out_lo.val[1] = vrshrn_n_s64(sum_lo.val[1], DCT_CONST_BITS);
  out_hi.val[0] = vrshrn_n_s64(sum_hi.val[0], DCT_CONST_BITS);
  out_hi.val[1] = vrshrn_n_s64(sum_hi.val[1], DCT_CONST_BITS);
  return vcombine_s32_dual(out_lo, out_hi);
}

static INLINE int32x4x2_t highbd_sub_dct_const_round_shift_low_8(
    const int64x2x2_t *const in0, const int64x2x2_t *const in1) {
  const int64x2x2_t sub_lo = vsubq_s64_dual(in0[0], in1[0]);
  const int64x2x2_t sub_hi = vsubq_s64_dual(in0[1], in1[1]);
  int32x2x2_t out_lo, out_hi;

  out_lo.val[0] = vrshrn_n_s64(sub_lo.val[0], DCT_CONST_BITS);
  out_lo.val[1] = vrshrn_n_s64(sub_lo.val[1], DCT_CONST_BITS);
  out_hi.val[0] = vrshrn_n_s64(sub_hi.val[0], DCT_CONST_BITS);
  out_hi.val[1] = vrshrn_n_s64(sub_hi.val[1], DCT_CONST_BITS);
  return vcombine_s32_dual(out_lo, out_hi);
}

static INLINE int32x4x2_t vnegq_s32_dual(const int32x4x2_t in) {
  int32x4x2_t out;
  out.val[0] = vnegq_s32(in.val[0]);
  out.val[1] = vnegq_s32(in.val[1]);
  return out;
}

static void highbd_iadst16_neon(const int32_t *input, int32_t *output,
                                uint16_t *dest, const int stride,
                                const int bd) {
  const int32x4_t c_1_31_5_27 =
      create_s32x4_neon(cospi_1_64, cospi_31_64, cospi_5_64, cospi_27_64);
  const int32x4_t c_9_23_13_19 =
      create_s32x4_neon(cospi_9_64, cospi_23_64, cospi_13_64, cospi_19_64);
  const int32x4_t c_17_15_21_11 =
      create_s32x4_neon(cospi_17_64, cospi_15_64, cospi_21_64, cospi_11_64);
  const int32x4_t c_25_7_29_3 =
      create_s32x4_neon(cospi_25_64, cospi_7_64, cospi_29_64, cospi_3_64);
  const int32x4_t c_4_28_20_12 =
      create_s32x4_neon(cospi_4_64, cospi_28_64, cospi_20_64, cospi_12_64);
  const int32x4_t c_16_n16_8_24 =
      create_s32x4_neon(cospi_16_64, -cospi_16_64, cospi_8_64, cospi_24_64);
  int32x4x2_t in[16], out[16];
  int32x4x2_t x[16], t[12];
  int64x2x2_t s0[2], s1[2], s2[2], s3[2], s4[2], s5[2], s6[2], s7[2];
  int64x2x2_t s8[2], s9[2], s10[2], s11[2], s12[2], s13[2], s14[2], s15[2];

  // Load input (16x8)
  in[0].val[0] = vld1q_s32(input);
  in[0].val[1] = vld1q_s32(input + 4);
  input += 8;
  in[8].val[0] = vld1q_s32(input);
  in[8].val[1] = vld1q_s32(input + 4);
  input += 8;
  in[1].val[0] = vld1q_s32(input);
  in[1].val[1] = vld1q_s32(input + 4);
  input += 8;
  in[9].val[0] = vld1q_s32(input);
  in[9].val[1] = vld1q_s32(input + 4);
  input += 8;
  in[2].val[0] = vld1q_s32(input);
  in[2].val[1] = vld1q_s32(input + 4);
  input += 8;
  in[10].val[0] = vld1q_s32(input);
  in[10].val[1] = vld1q_s32(input + 4);
  input += 8;
  in[3].val[0] = vld1q_s32(input);
  in[3].val[1] = vld1q_s32(input + 4);
  input += 8;
  in[11].val[0] = vld1q_s32(input);
  in[11].val[1] = vld1q_s32(input + 4);
  input += 8;
  in[4].val[0] = vld1q_s32(input);
  in[4].val[1] = vld1q_s32(input + 4);
  input += 8;
  in[12].val[0] = vld1q_s32(input);
  in[12].val[1] = vld1q_s32(input + 4);
  input += 8;
  in[5].val[0] = vld1q_s32(input);
  in[5].val[1] = vld1q_s32(input + 4);
  input += 8;
  in[13].val[0] = vld1q_s32(input);
  in[13].val[1] = vld1q_s32(input + 4);
  input += 8;
  in[6].val[0] = vld1q_s32(input);
  in[6].val[1] = vld1q_s32(input + 4);
  input += 8;
  in[14].val[0] = vld1q_s32(input);
  in[14].val[1] = vld1q_s32(input + 4);
  input += 8;
  in[7].val[0] = vld1q_s32(input);
  in[7].val[1] = vld1q_s32(input + 4);
  input += 8;
  in[15].val[0] = vld1q_s32(input);
  in[15].val[1] = vld1q_s32(input + 4);

  // Transpose
  transpose_s32_8x8(&in[0], &in[1], &in[2], &in[3], &in[4], &in[5], &in[6],
                    &in[7]);
  transpose_s32_8x8(&in[8], &in[9], &in[10], &in[11], &in[12], &in[13], &in[14],
                    &in[15]);

  x[0] = in[15];
  x[1] = in[0];
  x[2] = in[13];
  x[3] = in[2];
  x[4] = in[11];
  x[5] = in[4];
  x[6] = in[9];
  x[7] = in[6];
  x[8] = in[7];
  x[9] = in[8];
  x[10] = in[5];
  x[11] = in[10];
  x[12] = in[3];
  x[13] = in[12];
  x[14] = in[1];
  x[15] = in[14];

  // stage 1
  highbd_iadst_butterfly(x[0], x[1], vget_low_s32(c_1_31_5_27), 0, 1, s0, s1);
  highbd_iadst_butterfly(x[2], x[3], vget_high_s32(c_1_31_5_27), 0, 1, s2, s3);
  highbd_iadst_butterfly(x[4], x[5], vget_low_s32(c_9_23_13_19), 0, 1, s4, s5);
  highbd_iadst_butterfly(x[6], x[7], vget_high_s32(c_9_23_13_19), 0, 1, s6, s7);
  highbd_iadst_butterfly(x[8], x[9], vget_low_s32(c_17_15_21_11), 0, 1, s8, s9);
  highbd_iadst_butterfly(x[10], x[11], vget_high_s32(c_17_15_21_11), 0, 1, s10,
                         s11);
  highbd_iadst_butterfly(x[12], x[13], vget_low_s32(c_25_7_29_3), 0, 1, s12,
                         s13);
  highbd_iadst_butterfly(x[14], x[15], vget_high_s32(c_25_7_29_3), 0, 1, s14,
                         s15);

  x[0] = highbd_add_dct_const_round_shift_low_8(s0, s8);
  x[1] = highbd_add_dct_const_round_shift_low_8(s1, s9);
  x[2] = highbd_add_dct_const_round_shift_low_8(s2, s10);
  x[3] = highbd_add_dct_const_round_shift_low_8(s3, s11);
  x[4] = highbd_add_dct_const_round_shift_low_8(s4, s12);
  x[5] = highbd_add_dct_const_round_shift_low_8(s5, s13);
  x[6] = highbd_add_dct_const_round_shift_low_8(s6, s14);
  x[7] = highbd_add_dct_const_round_shift_low_8(s7, s15);
  x[8] = highbd_sub_dct_const_round_shift_low_8(s0, s8);
  x[9] = highbd_sub_dct_const_round_shift_low_8(s1, s9);
  x[10] = highbd_sub_dct_const_round_shift_low_8(s2, s10);
  x[11] = highbd_sub_dct_const_round_shift_low_8(s3, s11);
  x[12] = highbd_sub_dct_const_round_shift_low_8(s4, s12);
  x[13] = highbd_sub_dct_const_round_shift_low_8(s5, s13);
  x[14] = highbd_sub_dct_const_round_shift_low_8(s6, s14);
  x[15] = highbd_sub_dct_const_round_shift_low_8(s7, s15);

  // stage 2
  t[0] = x[0];
  t[1] = x[1];
  t[2] = x[2];
  t[3] = x[3];
  t[4] = x[4];
  t[5] = x[5];
  t[6] = x[6];
  t[7] = x[7];
  highbd_iadst_butterfly(x[8], x[9], vget_low_s32(c_4_28_20_12), 0, 1, s8, s9);
  highbd_iadst_butterfly(x[10], x[11], vget_high_s32(c_4_28_20_12), 0, 1, s10,
                         s11);
  highbd_iadst_butterfly(x[13], x[12], vget_low_s32(c_4_28_20_12), 1, 0, s13,
                         s12);
  highbd_iadst_butterfly(x[15], x[14], vget_high_s32(c_4_28_20_12), 1, 0, s15,
                         s14);

  x[0] = vaddq_s32_dual(t[0], t[4]);
  x[1] = vaddq_s32_dual(t[1], t[5]);
  x[2] = vaddq_s32_dual(t[2], t[6]);
  x[3] = vaddq_s32_dual(t[3], t[7]);
  x[4] = vsubq_s32_dual(t[0], t[4]);
  x[5] = vsubq_s32_dual(t[1], t[5]);
  x[6] = vsubq_s32_dual(t[2], t[6]);
  x[7] = vsubq_s32_dual(t[3], t[7]);
  x[8] = highbd_add_dct_const_round_shift_low_8(s8, s12);
  x[9] = highbd_add_dct_const_round_shift_low_8(s9, s13);
  x[10] = highbd_add_dct_const_round_shift_low_8(s10, s14);
  x[11] = highbd_add_dct_const_round_shift_low_8(s11, s15);
  x[12] = highbd_sub_dct_const_round_shift_low_8(s8, s12);
  x[13] = highbd_sub_dct_const_round_shift_low_8(s9, s13);
  x[14] = highbd_sub_dct_const_round_shift_low_8(s10, s14);
  x[15] = highbd_sub_dct_const_round_shift_low_8(s11, s15);

  // stage 3
  t[0] = x[0];
  t[1] = x[1];
  t[2] = x[2];
  t[3] = x[3];
  highbd_iadst_butterfly(x[4], x[5], vget_high_s32(c_16_n16_8_24), 0, 1, s4,
                         s5);
  highbd_iadst_butterfly(x[7], x[6], vget_high_s32(c_16_n16_8_24), 1, 0, s7,
                         s6);
  t[8] = x[8];
  t[9] = x[9];
  t[10] = x[10];
  t[11] = x[11];
  highbd_iadst_butterfly(x[12], x[13], vget_high_s32(c_16_n16_8_24), 0, 1, s12,
                         s13);
  highbd_iadst_butterfly(x[15], x[14], vget_high_s32(c_16_n16_8_24), 1, 0, s15,
                         s14);

  x[0] = vaddq_s32_dual(t[0], t[2]);
  x[1] = vaddq_s32_dual(t[1], t[3]);
  x[2] = vsubq_s32_dual(t[0], t[2]);
  x[3] = vsubq_s32_dual(t[1], t[3]);
  x[4] = highbd_add_dct_const_round_shift_low_8(s4, s6);
  x[5] = highbd_add_dct_const_round_shift_low_8(s5, s7);
  x[6] = highbd_sub_dct_const_round_shift_low_8(s4, s6);
  x[7] = highbd_sub_dct_const_round_shift_low_8(s5, s7);
  x[8] = vaddq_s32_dual(t[8], t[10]);
  x[9] = vaddq_s32_dual(t[9], t[11]);
  x[10] = vsubq_s32_dual(t[8], t[10]);
  x[11] = vsubq_s32_dual(t[9], t[11]);
  x[12] = highbd_add_dct_const_round_shift_low_8(s12, s14);
  x[13] = highbd_add_dct_const_round_shift_low_8(s13, s15);
  x[14] = highbd_sub_dct_const_round_shift_low_8(s12, s14);
  x[15] = highbd_sub_dct_const_round_shift_low_8(s13, s15);

  // stage 4
  {
    const int32x4x2_t sum = vaddq_s32_dual(x[2], x[3]);
    const int32x4x2_t sub = vsubq_s32_dual(x[2], x[3]);
    highbd_iadst_half_butterfly(sum, vget_low_s32(c_16_n16_8_24), 1, x[2]);
    highbd_iadst_half_butterfly(sub, vget_low_s32(c_16_n16_8_24), 0, x[3]);
  }
  {
    const int32x4x2_t sum = vaddq_s32_dual(x[7], x[6]);
    const int32x4x2_t sub = vsubq_s32_dual(x[7], x[6]);
    highbd_iadst_half_butterfly(sum, vget_low_s32(c_16_n16_8_24), 0, x[6]);
    highbd_iadst_half_butterfly(sub, vget_low_s32(c_16_n16_8_24), 0, x[7]);
  }
  {
    const int32x4x2_t sum = vaddq_s32_dual(x[11], x[10]);
    const int32x4x2_t sub = vsubq_s32_dual(x[11], x[10]);
    highbd_iadst_half_butterfly(sum, vget_low_s32(c_16_n16_8_24), 0, x[10]);
    highbd_iadst_half_butterfly(sub, vget_low_s32(c_16_n16_8_24), 0, x[11]);
  }
  {
    const int32x4x2_t sum = vaddq_s32_dual(x[14], x[15]);
    const int32x4x2_t sub = vsubq_s32_dual(x[14], x[15]);
    highbd_iadst_half_butterfly(sum, vget_low_s32(c_16_n16_8_24), 1, x[14]);
    highbd_iadst_half_butterfly(sub, vget_low_s32(c_16_n16_8_24), 0, x[15]);
  }

  out[0] = x[0];
  out[1] = vnegq_s32_dual(x[8]);
  out[2] = x[12];
  out[3] = vnegq_s32_dual(x[4]);
  out[4] = x[6];
  out[5] = x[14];
  out[6] = x[10];
  out[7] = x[2];
  out[8] = x[3];
  out[9] = x[11];
  out[10] = x[15];
  out[11] = x[7];
  out[12] = x[5];
  out[13] = vnegq_s32_dual(x[13]);
  out[14] = x[9];
  out[15] = vnegq_s32_dual(x[1]);

  if (output) {
    highbd_idct16x16_store_pass1(out, output);
  } else {
    highbd_idct16x16_add_store(out, dest, stride, bd);
  }
}

typedef void (*highbd_iht_1d)(const int32_t *input, int32_t *output,
                              uint16_t *dest, const int stride, const int bd);

typedef struct {
  highbd_iht_1d cols, rows;  // vertical and horizontal
} highbd_iht_2d;

void vp9_highbd_iht16x16_256_add_neon(const tran_low_t *input, uint16_t *dest,
                                      int stride, int tx_type, int bd) {
  if (bd == 8) {
    static const iht_2d IHT_16[] = {
      { vpx_idct16x16_256_add_half1d,
        vpx_idct16x16_256_add_half1d },  // DCT_DCT  = 0
      { vpx_iadst16x16_256_add_half1d,
        vpx_idct16x16_256_add_half1d },  // ADST_DCT = 1
      { vpx_idct16x16_256_add_half1d,
        vpx_iadst16x16_256_add_half1d },  // DCT_ADST = 2
      { vpx_iadst16x16_256_add_half1d,
        vpx_iadst16x16_256_add_half1d }  // ADST_ADST = 3
    };
    const iht_2d ht = IHT_16[tx_type];
    int16_t row_output[16 * 16];

    // pass 1
    ht.rows(input, row_output, dest, stride, 1);               // upper 8 rows
    ht.rows(input + 8 * 16, row_output + 8, dest, stride, 1);  // lower 8 rows

    // pass 2
    ht.cols(row_output, NULL, dest, stride, 1);               // left 8 columns
    ht.cols(row_output + 16 * 8, NULL, dest + 8, stride, 1);  // right 8 columns
  } else {
    static const highbd_iht_2d IHT_16[] = {
      { vpx_highbd_idct16x16_256_add_half1d,
        vpx_highbd_idct16x16_256_add_half1d },  // DCT_DCT  = 0
      { highbd_iadst16_neon,
        vpx_highbd_idct16x16_256_add_half1d },  // ADST_DCT = 1
      { vpx_highbd_idct16x16_256_add_half1d,
        highbd_iadst16_neon },                      // DCT_ADST = 2
      { highbd_iadst16_neon, highbd_iadst16_neon }  // ADST_ADST = 3
    };
    const highbd_iht_2d ht = IHT_16[tx_type];
    int32_t row_output[16 * 16];

    // pass 1
    ht.rows(input, row_output, dest, stride, bd);               // upper 8 rows
    ht.rows(input + 8 * 16, row_output + 8, dest, stride, bd);  // lower 8 rows

    // pass 2
    ht.cols(row_output, NULL, dest, stride, bd);  // left 8 columns
    ht.cols(row_output + 8 * 16, NULL, dest + 8, stride,
            bd);  // right 8 columns
  }
}
