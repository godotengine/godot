/*
 *  Copyright (c) 2016 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_ARM_IDCT_NEON_H_
#define VPX_VPX_DSP_ARM_IDCT_NEON_H_

#include <arm_neon.h>

#include "./vpx_config.h"
#include "vpx_dsp/arm/transpose_neon.h"
#include "vpx_dsp/txfm_common.h"
#include "vpx_dsp/vpx_dsp_common.h"

static const int16_t kCospi[16] = {
  16384 /*  cospi_0_64  */, 15137 /*  cospi_8_64  */,
  11585 /*  cospi_16_64 */, 6270 /*  cospi_24_64 */,
  16069 /*  cospi_4_64  */, 13623 /*  cospi_12_64 */,
  -9102 /* -cospi_20_64 */, 3196 /*  cospi_28_64 */,
  16305 /*  cospi_2_64  */, 1606 /*  cospi_30_64 */,
  14449 /*  cospi_10_64 */, 7723 /*  cospi_22_64 */,
  15679 /*  cospi_6_64  */, -4756 /* -cospi_26_64 */,
  12665 /*  cospi_14_64 */, -10394 /* -cospi_18_64 */
};

static const int32_t kCospi32[16] = {
  16384 /*  cospi_0_64  */, 15137 /*  cospi_8_64  */,
  11585 /*  cospi_16_64 */, 6270 /*  cospi_24_64 */,
  16069 /*  cospi_4_64  */, 13623 /*  cospi_12_64 */,
  -9102 /* -cospi_20_64 */, 3196 /*  cospi_28_64 */,
  16305 /*  cospi_2_64  */, 1606 /*  cospi_30_64 */,
  14449 /*  cospi_10_64 */, 7723 /*  cospi_22_64 */,
  15679 /*  cospi_6_64  */, -4756 /* -cospi_26_64 */,
  12665 /*  cospi_14_64 */, -10394 /* -cospi_18_64 */
};

//------------------------------------------------------------------------------
// Use saturating add/sub to avoid overflow in 2nd pass in high bit-depth
static INLINE int16x8_t final_add(const int16x8_t a, const int16x8_t b) {
#if CONFIG_VP9_HIGHBITDEPTH
  return vqaddq_s16(a, b);
#else
  return vaddq_s16(a, b);
#endif
}

static INLINE int16x8_t final_sub(const int16x8_t a, const int16x8_t b) {
#if CONFIG_VP9_HIGHBITDEPTH
  return vqsubq_s16(a, b);
#else
  return vsubq_s16(a, b);
#endif
}

//------------------------------------------------------------------------------

static INLINE int32x4x2_t highbd_idct_add_dual(const int32x4x2_t s0,
                                               const int32x4x2_t s1) {
  int32x4x2_t t;
  t.val[0] = vaddq_s32(s0.val[0], s1.val[0]);
  t.val[1] = vaddq_s32(s0.val[1], s1.val[1]);
  return t;
}

static INLINE int32x4x2_t highbd_idct_sub_dual(const int32x4x2_t s0,
                                               const int32x4x2_t s1) {
  int32x4x2_t t;
  t.val[0] = vsubq_s32(s0.val[0], s1.val[0]);
  t.val[1] = vsubq_s32(s0.val[1], s1.val[1]);
  return t;
}

//------------------------------------------------------------------------------

static INLINE int16x8_t dct_const_round_shift_low_8(const int32x4_t *const in) {
  return vcombine_s16(vrshrn_n_s32(in[0], DCT_CONST_BITS),
                      vrshrn_n_s32(in[1], DCT_CONST_BITS));
}

static INLINE void dct_const_round_shift_low_8_dual(const int32x4_t *const t32,
                                                    int16x8_t *const d0,
                                                    int16x8_t *const d1) {
  *d0 = dct_const_round_shift_low_8(t32 + 0);
  *d1 = dct_const_round_shift_low_8(t32 + 2);
}

static INLINE int32x4x2_t
dct_const_round_shift_high_4x2(const int64x2_t *const in) {
  int32x4x2_t out;
  out.val[0] = vcombine_s32(vrshrn_n_s64(in[0], DCT_CONST_BITS),
                            vrshrn_n_s64(in[1], DCT_CONST_BITS));
  out.val[1] = vcombine_s32(vrshrn_n_s64(in[2], DCT_CONST_BITS),
                            vrshrn_n_s64(in[3], DCT_CONST_BITS));
  return out;
}

// Multiply a by a_const. Saturate, shift and narrow by DCT_CONST_BITS.
static INLINE int16x8_t multiply_shift_and_narrow_s16(const int16x8_t a,
                                                      const int16_t a_const) {
  // Shift by DCT_CONST_BITS + rounding will be within 16 bits for well formed
  // streams. See WRAPLOW and dct_const_round_shift for details.
  // This instruction doubles the result and returns the high half, essentially
  // resulting in a right shift by 15. By multiplying the constant first that
  // becomes a right shift by DCT_CONST_BITS.
  // The largest possible value used here is
  // vpx_dsp/txfm_common.h:cospi_1_64 = 16364 (* 2 = 32728) a which falls *just*
  // within the range of int16_t (+32767 / -32768) even when negated.
  return vqrdmulhq_n_s16(a, a_const * 2);
}

// Add a and b, then multiply by ab_const. Shift and narrow by DCT_CONST_BITS.
static INLINE int16x8_t add_multiply_shift_and_narrow_s16(
    const int16x8_t a, const int16x8_t b, const int16_t ab_const) {
  // In both add_ and it's pair, sub_, the input for well-formed streams will be
  // well within 16 bits (input to the idct is the difference between two frames
  // and will be within -255 to 255, or 9 bits)
  // However, for inputs over about 25,000 (valid for int16_t, but not for idct
  // input) this function can not use vaddq_s16.
  // In order to match existing behavior and intentionally out of range tests,
  // expand the addition up to 32 bits to prevent truncation.
  int32x4_t t[2];
  t[0] = vaddl_s16(vget_low_s16(a), vget_low_s16(b));
  t[1] = vaddl_s16(vget_high_s16(a), vget_high_s16(b));
  t[0] = vmulq_n_s32(t[0], ab_const);
  t[1] = vmulq_n_s32(t[1], ab_const);
  return dct_const_round_shift_low_8(t);
}

// Subtract b from a, then multiply by ab_const. Shift and narrow by
// DCT_CONST_BITS.
static INLINE int16x8_t sub_multiply_shift_and_narrow_s16(
    const int16x8_t a, const int16x8_t b, const int16_t ab_const) {
  int32x4_t t[2];
  t[0] = vsubl_s16(vget_low_s16(a), vget_low_s16(b));
  t[1] = vsubl_s16(vget_high_s16(a), vget_high_s16(b));
  t[0] = vmulq_n_s32(t[0], ab_const);
  t[1] = vmulq_n_s32(t[1], ab_const);
  return dct_const_round_shift_low_8(t);
}

// Multiply a by a_const and b by b_const, then accumulate. Shift and narrow by
// DCT_CONST_BITS.
static INLINE int16x8_t multiply_accumulate_shift_and_narrow_s16(
    const int16x8_t a, const int16_t a_const, const int16x8_t b,
    const int16_t b_const) {
  int32x4_t t[2];
  t[0] = vmull_n_s16(vget_low_s16(a), a_const);
  t[1] = vmull_n_s16(vget_high_s16(a), a_const);
  t[0] = vmlal_n_s16(t[0], vget_low_s16(b), b_const);
  t[1] = vmlal_n_s16(t[1], vget_high_s16(b), b_const);
  return dct_const_round_shift_low_8(t);
}

//------------------------------------------------------------------------------

// Note: The following 4 functions could use 32-bit operations for bit-depth 10.
//       However, although it's 20% faster with gcc, it's 20% slower with clang.
//       Use 64-bit operations for now.

// Multiply a by a_const. Saturate, shift and narrow by DCT_CONST_BITS.
static INLINE int32x4x2_t
multiply_shift_and_narrow_s32_dual(const int32x4x2_t a, const int32_t a_const) {
  int64x2_t b[4];

  b[0] = vmull_n_s32(vget_low_s32(a.val[0]), a_const);
  b[1] = vmull_n_s32(vget_high_s32(a.val[0]), a_const);
  b[2] = vmull_n_s32(vget_low_s32(a.val[1]), a_const);
  b[3] = vmull_n_s32(vget_high_s32(a.val[1]), a_const);
  return dct_const_round_shift_high_4x2(b);
}

// Add a and b, then multiply by ab_const. Shift and narrow by DCT_CONST_BITS.
static INLINE int32x4x2_t add_multiply_shift_and_narrow_s32_dual(
    const int32x4x2_t a, const int32x4x2_t b, const int32_t ab_const) {
  int32x4_t t[2];
  int64x2_t c[4];

  t[0] = vaddq_s32(a.val[0], b.val[0]);
  t[1] = vaddq_s32(a.val[1], b.val[1]);
  c[0] = vmull_n_s32(vget_low_s32(t[0]), ab_const);
  c[1] = vmull_n_s32(vget_high_s32(t[0]), ab_const);
  c[2] = vmull_n_s32(vget_low_s32(t[1]), ab_const);
  c[3] = vmull_n_s32(vget_high_s32(t[1]), ab_const);
  return dct_const_round_shift_high_4x2(c);
}

// Subtract b from a, then multiply by ab_const. Shift and narrow by
// DCT_CONST_BITS.
static INLINE int32x4x2_t sub_multiply_shift_and_narrow_s32_dual(
    const int32x4x2_t a, const int32x4x2_t b, const int32_t ab_const) {
  int32x4_t t[2];
  int64x2_t c[4];

  t[0] = vsubq_s32(a.val[0], b.val[0]);
  t[1] = vsubq_s32(a.val[1], b.val[1]);
  c[0] = vmull_n_s32(vget_low_s32(t[0]), ab_const);
  c[1] = vmull_n_s32(vget_high_s32(t[0]), ab_const);
  c[2] = vmull_n_s32(vget_low_s32(t[1]), ab_const);
  c[3] = vmull_n_s32(vget_high_s32(t[1]), ab_const);
  return dct_const_round_shift_high_4x2(c);
}

// Multiply a by a_const and b by b_const, then accumulate. Shift and narrow by
// DCT_CONST_BITS.
static INLINE int32x4x2_t multiply_accumulate_shift_and_narrow_s32_dual(
    const int32x4x2_t a, const int32_t a_const, const int32x4x2_t b,
    const int32_t b_const) {
  int64x2_t c[4];
  c[0] = vmull_n_s32(vget_low_s32(a.val[0]), a_const);
  c[1] = vmull_n_s32(vget_high_s32(a.val[0]), a_const);
  c[2] = vmull_n_s32(vget_low_s32(a.val[1]), a_const);
  c[3] = vmull_n_s32(vget_high_s32(a.val[1]), a_const);
  c[0] = vmlal_n_s32(c[0], vget_low_s32(b.val[0]), b_const);
  c[1] = vmlal_n_s32(c[1], vget_high_s32(b.val[0]), b_const);
  c[2] = vmlal_n_s32(c[2], vget_low_s32(b.val[1]), b_const);
  c[3] = vmlal_n_s32(c[3], vget_high_s32(b.val[1]), b_const);
  return dct_const_round_shift_high_4x2(c);
}

// Shift the output down by 6 and add it to the destination buffer.
static INLINE void add_and_store_u8_s16(const int16x8_t *const a, uint8_t *d,
                                        const int stride) {
  uint8x8_t b[8];
  int16x8_t c[8];

  b[0] = vld1_u8(d);
  d += stride;
  b[1] = vld1_u8(d);
  d += stride;
  b[2] = vld1_u8(d);
  d += stride;
  b[3] = vld1_u8(d);
  d += stride;
  b[4] = vld1_u8(d);
  d += stride;
  b[5] = vld1_u8(d);
  d += stride;
  b[6] = vld1_u8(d);
  d += stride;
  b[7] = vld1_u8(d);
  d -= (7 * stride);

  // c = b + (a >> 6)
  c[0] = vrsraq_n_s16(vreinterpretq_s16_u16(vmovl_u8(b[0])), a[0], 6);
  c[1] = vrsraq_n_s16(vreinterpretq_s16_u16(vmovl_u8(b[1])), a[1], 6);
  c[2] = vrsraq_n_s16(vreinterpretq_s16_u16(vmovl_u8(b[2])), a[2], 6);
  c[3] = vrsraq_n_s16(vreinterpretq_s16_u16(vmovl_u8(b[3])), a[3], 6);
  c[4] = vrsraq_n_s16(vreinterpretq_s16_u16(vmovl_u8(b[4])), a[4], 6);
  c[5] = vrsraq_n_s16(vreinterpretq_s16_u16(vmovl_u8(b[5])), a[5], 6);
  c[6] = vrsraq_n_s16(vreinterpretq_s16_u16(vmovl_u8(b[6])), a[6], 6);
  c[7] = vrsraq_n_s16(vreinterpretq_s16_u16(vmovl_u8(b[7])), a[7], 6);

  b[0] = vqmovun_s16(c[0]);
  b[1] = vqmovun_s16(c[1]);
  b[2] = vqmovun_s16(c[2]);
  b[3] = vqmovun_s16(c[3]);
  b[4] = vqmovun_s16(c[4]);
  b[5] = vqmovun_s16(c[5]);
  b[6] = vqmovun_s16(c[6]);
  b[7] = vqmovun_s16(c[7]);

  vst1_u8(d, b[0]);
  d += stride;
  vst1_u8(d, b[1]);
  d += stride;
  vst1_u8(d, b[2]);
  d += stride;
  vst1_u8(d, b[3]);
  d += stride;
  vst1_u8(d, b[4]);
  d += stride;
  vst1_u8(d, b[5]);
  d += stride;
  vst1_u8(d, b[6]);
  d += stride;
  vst1_u8(d, b[7]);
}

static INLINE uint8x16_t create_dcq(const int16_t dc) {
  // Clip both sides and gcc may compile to assembly 'usat'.
  const int16_t t = (dc < 0) ? 0 : ((dc > 255) ? 255 : dc);
  return vdupq_n_u8((uint8_t)t);
}

static INLINE void idct4x4_16_kernel_bd8(int16x8_t *const a) {
  const int16x4_t cospis = vld1_s16(kCospi);
  int16x4_t b[4];
  int32x4_t c[4];
  int16x8_t d[2];

  b[0] = vget_low_s16(a[0]);
  b[1] = vget_high_s16(a[0]);
  b[2] = vget_low_s16(a[1]);
  b[3] = vget_high_s16(a[1]);
  c[0] = vmull_lane_s16(b[0], cospis, 2);
  c[2] = vmull_lane_s16(b[1], cospis, 2);
  c[1] = vsubq_s32(c[0], c[2]);
  c[0] = vaddq_s32(c[0], c[2]);
  c[3] = vmull_lane_s16(b[2], cospis, 3);
  c[2] = vmull_lane_s16(b[2], cospis, 1);
  c[3] = vmlsl_lane_s16(c[3], b[3], cospis, 1);
  c[2] = vmlal_lane_s16(c[2], b[3], cospis, 3);
  dct_const_round_shift_low_8_dual(c, &d[0], &d[1]);
  a[0] = vaddq_s16(d[0], d[1]);
  a[1] = vsubq_s16(d[0], d[1]);
}

static INLINE void transpose_idct4x4_16_bd8(int16x8_t *const a) {
  transpose_s16_4x4q(&a[0], &a[1]);
  idct4x4_16_kernel_bd8(a);
}

static INLINE void idct8x8_12_pass1_bd8(const int16x4_t cospis0,
                                        const int16x4_t cospisd0,
                                        const int16x4_t cospisd1,
                                        int16x4_t *const io) {
  int16x4_t step1[8], step2[8];
  int32x4_t t32[2];

  transpose_s16_4x4d(&io[0], &io[1], &io[2], &io[3]);

  // stage 1
  step1[4] = vqrdmulh_lane_s16(io[1], cospisd1, 3);
  step1[5] = vqrdmulh_lane_s16(io[3], cospisd1, 2);
  step1[6] = vqrdmulh_lane_s16(io[3], cospisd1, 1);
  step1[7] = vqrdmulh_lane_s16(io[1], cospisd1, 0);

  // stage 2
  step2[1] = vqrdmulh_lane_s16(io[0], cospisd0, 2);
  step2[2] = vqrdmulh_lane_s16(io[2], cospisd0, 3);
  step2[3] = vqrdmulh_lane_s16(io[2], cospisd0, 1);

  step2[4] = vadd_s16(step1[4], step1[5]);
  step2[5] = vsub_s16(step1[4], step1[5]);
  step2[6] = vsub_s16(step1[7], step1[6]);
  step2[7] = vadd_s16(step1[7], step1[6]);

  // stage 3
  step1[0] = vadd_s16(step2[1], step2[3]);
  step1[1] = vadd_s16(step2[1], step2[2]);
  step1[2] = vsub_s16(step2[1], step2[2]);
  step1[3] = vsub_s16(step2[1], step2[3]);

  t32[1] = vmull_lane_s16(step2[6], cospis0, 2);
  t32[0] = vmlsl_lane_s16(t32[1], step2[5], cospis0, 2);
  t32[1] = vmlal_lane_s16(t32[1], step2[5], cospis0, 2);
  step1[5] = vrshrn_n_s32(t32[0], DCT_CONST_BITS);
  step1[6] = vrshrn_n_s32(t32[1], DCT_CONST_BITS);

  // stage 4
  io[0] = vadd_s16(step1[0], step2[7]);
  io[1] = vadd_s16(step1[1], step1[6]);
  io[2] = vadd_s16(step1[2], step1[5]);
  io[3] = vadd_s16(step1[3], step2[4]);
  io[4] = vsub_s16(step1[3], step2[4]);
  io[5] = vsub_s16(step1[2], step1[5]);
  io[6] = vsub_s16(step1[1], step1[6]);
  io[7] = vsub_s16(step1[0], step2[7]);
}

static INLINE void idct8x8_12_pass2_bd8(const int16x4_t cospis0,
                                        const int16x4_t cospisd0,
                                        const int16x4_t cospisd1,
                                        const int16x4_t *const input,
                                        int16x8_t *const output) {
  int16x8_t in[4];
  int16x8_t step1[8], step2[8];
  int32x4_t t32[8];

  transpose_s16_4x8(input[0], input[1], input[2], input[3], input[4], input[5],
                    input[6], input[7], &in[0], &in[1], &in[2], &in[3]);

  // stage 1
  step1[4] = vqrdmulhq_lane_s16(in[1], cospisd1, 3);
  step1[5] = vqrdmulhq_lane_s16(in[3], cospisd1, 2);
  step1[6] = vqrdmulhq_lane_s16(in[3], cospisd1, 1);
  step1[7] = vqrdmulhq_lane_s16(in[1], cospisd1, 0);

  // stage 2
  step2[1] = vqrdmulhq_lane_s16(in[0], cospisd0, 2);
  step2[2] = vqrdmulhq_lane_s16(in[2], cospisd0, 3);
  step2[3] = vqrdmulhq_lane_s16(in[2], cospisd0, 1);

  step2[4] = vaddq_s16(step1[4], step1[5]);
  step2[5] = vsubq_s16(step1[4], step1[5]);
  step2[6] = vsubq_s16(step1[7], step1[6]);
  step2[7] = vaddq_s16(step1[7], step1[6]);

  // stage 3
  step1[0] = vaddq_s16(step2[1], step2[3]);
  step1[1] = vaddq_s16(step2[1], step2[2]);
  step1[2] = vsubq_s16(step2[1], step2[2]);
  step1[3] = vsubq_s16(step2[1], step2[3]);

  t32[2] = vmull_lane_s16(vget_low_s16(step2[6]), cospis0, 2);
  t32[3] = vmull_lane_s16(vget_high_s16(step2[6]), cospis0, 2);
  t32[0] = vmlsl_lane_s16(t32[2], vget_low_s16(step2[5]), cospis0, 2);
  t32[1] = vmlsl_lane_s16(t32[3], vget_high_s16(step2[5]), cospis0, 2);
  t32[2] = vmlal_lane_s16(t32[2], vget_low_s16(step2[5]), cospis0, 2);
  t32[3] = vmlal_lane_s16(t32[3], vget_high_s16(step2[5]), cospis0, 2);
  dct_const_round_shift_low_8_dual(t32, &step1[5], &step1[6]);

  // stage 4
  output[0] = vaddq_s16(step1[0], step2[7]);
  output[1] = vaddq_s16(step1[1], step1[6]);
  output[2] = vaddq_s16(step1[2], step1[5]);
  output[3] = vaddq_s16(step1[3], step2[4]);
  output[4] = vsubq_s16(step1[3], step2[4]);
  output[5] = vsubq_s16(step1[2], step1[5]);
  output[6] = vsubq_s16(step1[1], step1[6]);
  output[7] = vsubq_s16(step1[0], step2[7]);
}

static INLINE void idct8x8_64_1d_bd8_kernel(const int16x4_t cospis0,
                                            const int16x4_t cospis1,
                                            int16x8_t *const io) {
  int16x4_t input1l, input1h, input3l, input3h, input5l, input5h, input7l,
      input7h;
  int16x4_t step1l[4], step1h[4];
  int16x8_t step1[8], step2[8];
  int32x4_t t32[8];

  // stage 1
  input1l = vget_low_s16(io[1]);
  input1h = vget_high_s16(io[1]);
  input3l = vget_low_s16(io[3]);
  input3h = vget_high_s16(io[3]);
  input5l = vget_low_s16(io[5]);
  input5h = vget_high_s16(io[5]);
  input7l = vget_low_s16(io[7]);
  input7h = vget_high_s16(io[7]);
  step1l[0] = vget_low_s16(io[0]);
  step1h[0] = vget_high_s16(io[0]);
  step1l[1] = vget_low_s16(io[2]);
  step1h[1] = vget_high_s16(io[2]);
  step1l[2] = vget_low_s16(io[4]);
  step1h[2] = vget_high_s16(io[4]);
  step1l[3] = vget_low_s16(io[6]);
  step1h[3] = vget_high_s16(io[6]);

  t32[0] = vmull_lane_s16(input1l, cospis1, 3);
  t32[1] = vmull_lane_s16(input1h, cospis1, 3);
  t32[2] = vmull_lane_s16(input3l, cospis1, 2);
  t32[3] = vmull_lane_s16(input3h, cospis1, 2);
  t32[4] = vmull_lane_s16(input3l, cospis1, 1);
  t32[5] = vmull_lane_s16(input3h, cospis1, 1);
  t32[6] = vmull_lane_s16(input1l, cospis1, 0);
  t32[7] = vmull_lane_s16(input1h, cospis1, 0);
  t32[0] = vmlsl_lane_s16(t32[0], input7l, cospis1, 0);
  t32[1] = vmlsl_lane_s16(t32[1], input7h, cospis1, 0);
  t32[2] = vmlal_lane_s16(t32[2], input5l, cospis1, 1);
  t32[3] = vmlal_lane_s16(t32[3], input5h, cospis1, 1);
  t32[4] = vmlsl_lane_s16(t32[4], input5l, cospis1, 2);
  t32[5] = vmlsl_lane_s16(t32[5], input5h, cospis1, 2);
  t32[6] = vmlal_lane_s16(t32[6], input7l, cospis1, 3);
  t32[7] = vmlal_lane_s16(t32[7], input7h, cospis1, 3);
  dct_const_round_shift_low_8_dual(&t32[0], &step1[4], &step1[5]);
  dct_const_round_shift_low_8_dual(&t32[4], &step1[6], &step1[7]);

  // stage 2
  t32[2] = vmull_lane_s16(step1l[0], cospis0, 2);
  t32[3] = vmull_lane_s16(step1h[0], cospis0, 2);
  t32[4] = vmull_lane_s16(step1l[1], cospis0, 3);
  t32[5] = vmull_lane_s16(step1h[1], cospis0, 3);
  t32[6] = vmull_lane_s16(step1l[1], cospis0, 1);
  t32[7] = vmull_lane_s16(step1h[1], cospis0, 1);
  t32[0] = vmlal_lane_s16(t32[2], step1l[2], cospis0, 2);
  t32[1] = vmlal_lane_s16(t32[3], step1h[2], cospis0, 2);
  t32[2] = vmlsl_lane_s16(t32[2], step1l[2], cospis0, 2);
  t32[3] = vmlsl_lane_s16(t32[3], step1h[2], cospis0, 2);
  t32[4] = vmlsl_lane_s16(t32[4], step1l[3], cospis0, 1);
  t32[5] = vmlsl_lane_s16(t32[5], step1h[3], cospis0, 1);
  t32[6] = vmlal_lane_s16(t32[6], step1l[3], cospis0, 3);
  t32[7] = vmlal_lane_s16(t32[7], step1h[3], cospis0, 3);
  dct_const_round_shift_low_8_dual(&t32[0], &step2[0], &step2[1]);
  dct_const_round_shift_low_8_dual(&t32[4], &step2[2], &step2[3]);

  step2[4] = vaddq_s16(step1[4], step1[5]);
  step2[5] = vsubq_s16(step1[4], step1[5]);
  step2[6] = vsubq_s16(step1[7], step1[6]);
  step2[7] = vaddq_s16(step1[7], step1[6]);

  // stage 3
  step1[0] = vaddq_s16(step2[0], step2[3]);
  step1[1] = vaddq_s16(step2[1], step2[2]);
  step1[2] = vsubq_s16(step2[1], step2[2]);
  step1[3] = vsubq_s16(step2[0], step2[3]);

  t32[2] = vmull_lane_s16(vget_low_s16(step2[6]), cospis0, 2);
  t32[3] = vmull_lane_s16(vget_high_s16(step2[6]), cospis0, 2);
  t32[0] = vmlsl_lane_s16(t32[2], vget_low_s16(step2[5]), cospis0, 2);
  t32[1] = vmlsl_lane_s16(t32[3], vget_high_s16(step2[5]), cospis0, 2);
  t32[2] = vmlal_lane_s16(t32[2], vget_low_s16(step2[5]), cospis0, 2);
  t32[3] = vmlal_lane_s16(t32[3], vget_high_s16(step2[5]), cospis0, 2);
  dct_const_round_shift_low_8_dual(t32, &step1[5], &step1[6]);

  // stage 4
  io[0] = vaddq_s16(step1[0], step2[7]);
  io[1] = vaddq_s16(step1[1], step1[6]);
  io[2] = vaddq_s16(step1[2], step1[5]);
  io[3] = vaddq_s16(step1[3], step2[4]);
  io[4] = vsubq_s16(step1[3], step2[4]);
  io[5] = vsubq_s16(step1[2], step1[5]);
  io[6] = vsubq_s16(step1[1], step1[6]);
  io[7] = vsubq_s16(step1[0], step2[7]);
}

static INLINE void idct8x8_64_1d_bd8(const int16x4_t cospis0,
                                     const int16x4_t cospis1,
                                     int16x8_t *const io) {
  transpose_s16_8x8(&io[0], &io[1], &io[2], &io[3], &io[4], &io[5], &io[6],
                    &io[7]);
  idct8x8_64_1d_bd8_kernel(cospis0, cospis1, io);
}

static INLINE void idct_cospi_8_24_q_kernel(const int16x8_t s0,
                                            const int16x8_t s1,
                                            const int16x4_t cospi_0_8_16_24,
                                            int32x4_t *const t32) {
  t32[0] = vmull_lane_s16(vget_low_s16(s0), cospi_0_8_16_24, 3);
  t32[1] = vmull_lane_s16(vget_high_s16(s0), cospi_0_8_16_24, 3);
  t32[2] = vmull_lane_s16(vget_low_s16(s1), cospi_0_8_16_24, 3);
  t32[3] = vmull_lane_s16(vget_high_s16(s1), cospi_0_8_16_24, 3);
  t32[0] = vmlsl_lane_s16(t32[0], vget_low_s16(s1), cospi_0_8_16_24, 1);
  t32[1] = vmlsl_lane_s16(t32[1], vget_high_s16(s1), cospi_0_8_16_24, 1);
  t32[2] = vmlal_lane_s16(t32[2], vget_low_s16(s0), cospi_0_8_16_24, 1);
  t32[3] = vmlal_lane_s16(t32[3], vget_high_s16(s0), cospi_0_8_16_24, 1);
}

static INLINE void idct_cospi_8_24_q(const int16x8_t s0, const int16x8_t s1,
                                     const int16x4_t cospi_0_8_16_24,
                                     int16x8_t *const d0, int16x8_t *const d1) {
  int32x4_t t32[4];

  idct_cospi_8_24_q_kernel(s0, s1, cospi_0_8_16_24, t32);
  dct_const_round_shift_low_8_dual(t32, d0, d1);
}

static INLINE void idct_cospi_8_24_neg_q(const int16x8_t s0, const int16x8_t s1,
                                         const int16x4_t cospi_0_8_16_24,
                                         int16x8_t *const d0,
                                         int16x8_t *const d1) {
  int32x4_t t32[4];

  idct_cospi_8_24_q_kernel(s0, s1, cospi_0_8_16_24, t32);
  t32[2] = vnegq_s32(t32[2]);
  t32[3] = vnegq_s32(t32[3]);
  dct_const_round_shift_low_8_dual(t32, d0, d1);
}

static INLINE void idct_cospi_16_16_q(const int16x8_t s0, const int16x8_t s1,
                                      const int16x4_t cospi_0_8_16_24,
                                      int16x8_t *const d0,
                                      int16x8_t *const d1) {
  int32x4_t t32[6];

  t32[4] = vmull_lane_s16(vget_low_s16(s1), cospi_0_8_16_24, 2);
  t32[5] = vmull_lane_s16(vget_high_s16(s1), cospi_0_8_16_24, 2);
  t32[0] = vmlsl_lane_s16(t32[4], vget_low_s16(s0), cospi_0_8_16_24, 2);
  t32[1] = vmlsl_lane_s16(t32[5], vget_high_s16(s0), cospi_0_8_16_24, 2);
  t32[2] = vmlal_lane_s16(t32[4], vget_low_s16(s0), cospi_0_8_16_24, 2);
  t32[3] = vmlal_lane_s16(t32[5], vget_high_s16(s0), cospi_0_8_16_24, 2);
  dct_const_round_shift_low_8_dual(t32, d0, d1);
}

static INLINE void idct_cospi_2_30(const int16x8_t s0, const int16x8_t s1,
                                   const int16x4_t cospi_2_30_10_22,
                                   int16x8_t *const d0, int16x8_t *const d1) {
  int32x4_t t32[4];

  t32[0] = vmull_lane_s16(vget_low_s16(s0), cospi_2_30_10_22, 1);
  t32[1] = vmull_lane_s16(vget_high_s16(s0), cospi_2_30_10_22, 1);
  t32[2] = vmull_lane_s16(vget_low_s16(s1), cospi_2_30_10_22, 1);
  t32[3] = vmull_lane_s16(vget_high_s16(s1), cospi_2_30_10_22, 1);
  t32[0] = vmlsl_lane_s16(t32[0], vget_low_s16(s1), cospi_2_30_10_22, 0);
  t32[1] = vmlsl_lane_s16(t32[1], vget_high_s16(s1), cospi_2_30_10_22, 0);
  t32[2] = vmlal_lane_s16(t32[2], vget_low_s16(s0), cospi_2_30_10_22, 0);
  t32[3] = vmlal_lane_s16(t32[3], vget_high_s16(s0), cospi_2_30_10_22, 0);
  dct_const_round_shift_low_8_dual(t32, d0, d1);
}

static INLINE void idct_cospi_4_28(const int16x8_t s0, const int16x8_t s1,
                                   const int16x4_t cospi_4_12_20N_28,
                                   int16x8_t *const d0, int16x8_t *const d1) {
  int32x4_t t32[4];

  t32[0] = vmull_lane_s16(vget_low_s16(s0), cospi_4_12_20N_28, 3);
  t32[1] = vmull_lane_s16(vget_high_s16(s0), cospi_4_12_20N_28, 3);
  t32[2] = vmull_lane_s16(vget_low_s16(s1), cospi_4_12_20N_28, 3);
  t32[3] = vmull_lane_s16(vget_high_s16(s1), cospi_4_12_20N_28, 3);
  t32[0] = vmlsl_lane_s16(t32[0], vget_low_s16(s1), cospi_4_12_20N_28, 0);
  t32[1] = vmlsl_lane_s16(t32[1], vget_high_s16(s1), cospi_4_12_20N_28, 0);
  t32[2] = vmlal_lane_s16(t32[2], vget_low_s16(s0), cospi_4_12_20N_28, 0);
  t32[3] = vmlal_lane_s16(t32[3], vget_high_s16(s0), cospi_4_12_20N_28, 0);
  dct_const_round_shift_low_8_dual(t32, d0, d1);
}

static INLINE void idct_cospi_6_26(const int16x8_t s0, const int16x8_t s1,
                                   const int16x4_t cospi_6_26N_14_18N,
                                   int16x8_t *const d0, int16x8_t *const d1) {
  int32x4_t t32[4];

  t32[0] = vmull_lane_s16(vget_low_s16(s0), cospi_6_26N_14_18N, 0);
  t32[1] = vmull_lane_s16(vget_high_s16(s0), cospi_6_26N_14_18N, 0);
  t32[2] = vmull_lane_s16(vget_low_s16(s1), cospi_6_26N_14_18N, 0);
  t32[3] = vmull_lane_s16(vget_high_s16(s1), cospi_6_26N_14_18N, 0);
  t32[0] = vmlal_lane_s16(t32[0], vget_low_s16(s1), cospi_6_26N_14_18N, 1);
  t32[1] = vmlal_lane_s16(t32[1], vget_high_s16(s1), cospi_6_26N_14_18N, 1);
  t32[2] = vmlsl_lane_s16(t32[2], vget_low_s16(s0), cospi_6_26N_14_18N, 1);
  t32[3] = vmlsl_lane_s16(t32[3], vget_high_s16(s0), cospi_6_26N_14_18N, 1);
  dct_const_round_shift_low_8_dual(t32, d0, d1);
}

static INLINE void idct_cospi_10_22(const int16x8_t s0, const int16x8_t s1,
                                    const int16x4_t cospi_2_30_10_22,
                                    int16x8_t *const d0, int16x8_t *const d1) {
  int32x4_t t32[4];

  t32[0] = vmull_lane_s16(vget_low_s16(s0), cospi_2_30_10_22, 3);
  t32[1] = vmull_lane_s16(vget_high_s16(s0), cospi_2_30_10_22, 3);
  t32[2] = vmull_lane_s16(vget_low_s16(s1), cospi_2_30_10_22, 3);
  t32[3] = vmull_lane_s16(vget_high_s16(s1), cospi_2_30_10_22, 3);
  t32[0] = vmlsl_lane_s16(t32[0], vget_low_s16(s1), cospi_2_30_10_22, 2);
  t32[1] = vmlsl_lane_s16(t32[1], vget_high_s16(s1), cospi_2_30_10_22, 2);
  t32[2] = vmlal_lane_s16(t32[2], vget_low_s16(s0), cospi_2_30_10_22, 2);
  t32[3] = vmlal_lane_s16(t32[3], vget_high_s16(s0), cospi_2_30_10_22, 2);
  dct_const_round_shift_low_8_dual(t32, d0, d1);
}

static INLINE void idct_cospi_12_20(const int16x8_t s0, const int16x8_t s1,
                                    const int16x4_t cospi_4_12_20N_28,
                                    int16x8_t *const d0, int16x8_t *const d1) {
  int32x4_t t32[4];

  t32[0] = vmull_lane_s16(vget_low_s16(s0), cospi_4_12_20N_28, 1);
  t32[1] = vmull_lane_s16(vget_high_s16(s0), cospi_4_12_20N_28, 1);
  t32[2] = vmull_lane_s16(vget_low_s16(s1), cospi_4_12_20N_28, 1);
  t32[3] = vmull_lane_s16(vget_high_s16(s1), cospi_4_12_20N_28, 1);
  t32[0] = vmlal_lane_s16(t32[0], vget_low_s16(s1), cospi_4_12_20N_28, 2);
  t32[1] = vmlal_lane_s16(t32[1], vget_high_s16(s1), cospi_4_12_20N_28, 2);
  t32[2] = vmlsl_lane_s16(t32[2], vget_low_s16(s0), cospi_4_12_20N_28, 2);
  t32[3] = vmlsl_lane_s16(t32[3], vget_high_s16(s0), cospi_4_12_20N_28, 2);
  dct_const_round_shift_low_8_dual(t32, d0, d1);
}

static INLINE void idct_cospi_14_18(const int16x8_t s0, const int16x8_t s1,
                                    const int16x4_t cospi_6_26N_14_18N,
                                    int16x8_t *const d0, int16x8_t *const d1) {
  int32x4_t t32[4];

  t32[0] = vmull_lane_s16(vget_low_s16(s0), cospi_6_26N_14_18N, 2);
  t32[1] = vmull_lane_s16(vget_high_s16(s0), cospi_6_26N_14_18N, 2);
  t32[2] = vmull_lane_s16(vget_low_s16(s1), cospi_6_26N_14_18N, 2);
  t32[3] = vmull_lane_s16(vget_high_s16(s1), cospi_6_26N_14_18N, 2);
  t32[0] = vmlal_lane_s16(t32[0], vget_low_s16(s1), cospi_6_26N_14_18N, 3);
  t32[1] = vmlal_lane_s16(t32[1], vget_high_s16(s1), cospi_6_26N_14_18N, 3);
  t32[2] = vmlsl_lane_s16(t32[2], vget_low_s16(s0), cospi_6_26N_14_18N, 3);
  t32[3] = vmlsl_lane_s16(t32[3], vget_high_s16(s0), cospi_6_26N_14_18N, 3);
  dct_const_round_shift_low_8_dual(t32, d0, d1);
}

static INLINE void idct16x16_add_stage7(const int16x8_t *const step2,
                                        int16x8_t *const out) {
#if CONFIG_VP9_HIGHBITDEPTH
  // Use saturating add/sub to avoid overflow in 2nd pass
  out[0] = vqaddq_s16(step2[0], step2[15]);
  out[1] = vqaddq_s16(step2[1], step2[14]);
  out[2] = vqaddq_s16(step2[2], step2[13]);
  out[3] = vqaddq_s16(step2[3], step2[12]);
  out[4] = vqaddq_s16(step2[4], step2[11]);
  out[5] = vqaddq_s16(step2[5], step2[10]);
  out[6] = vqaddq_s16(step2[6], step2[9]);
  out[7] = vqaddq_s16(step2[7], step2[8]);
  out[8] = vqsubq_s16(step2[7], step2[8]);
  out[9] = vqsubq_s16(step2[6], step2[9]);
  out[10] = vqsubq_s16(step2[5], step2[10]);
  out[11] = vqsubq_s16(step2[4], step2[11]);
  out[12] = vqsubq_s16(step2[3], step2[12]);
  out[13] = vqsubq_s16(step2[2], step2[13]);
  out[14] = vqsubq_s16(step2[1], step2[14]);
  out[15] = vqsubq_s16(step2[0], step2[15]);
#else
  out[0] = vaddq_s16(step2[0], step2[15]);
  out[1] = vaddq_s16(step2[1], step2[14]);
  out[2] = vaddq_s16(step2[2], step2[13]);
  out[3] = vaddq_s16(step2[3], step2[12]);
  out[4] = vaddq_s16(step2[4], step2[11]);
  out[5] = vaddq_s16(step2[5], step2[10]);
  out[6] = vaddq_s16(step2[6], step2[9]);
  out[7] = vaddq_s16(step2[7], step2[8]);
  out[8] = vsubq_s16(step2[7], step2[8]);
  out[9] = vsubq_s16(step2[6], step2[9]);
  out[10] = vsubq_s16(step2[5], step2[10]);
  out[11] = vsubq_s16(step2[4], step2[11]);
  out[12] = vsubq_s16(step2[3], step2[12]);
  out[13] = vsubq_s16(step2[2], step2[13]);
  out[14] = vsubq_s16(step2[1], step2[14]);
  out[15] = vsubq_s16(step2[0], step2[15]);
#endif
}

static INLINE void idct16x16_store_pass1(const int16x8_t *const out,
                                         int16_t *output) {
  // Save the result into output
  vst1q_s16(output, out[0]);
  output += 16;
  vst1q_s16(output, out[1]);
  output += 16;
  vst1q_s16(output, out[2]);
  output += 16;
  vst1q_s16(output, out[3]);
  output += 16;
  vst1q_s16(output, out[4]);
  output += 16;
  vst1q_s16(output, out[5]);
  output += 16;
  vst1q_s16(output, out[6]);
  output += 16;
  vst1q_s16(output, out[7]);
  output += 16;
  vst1q_s16(output, out[8]);
  output += 16;
  vst1q_s16(output, out[9]);
  output += 16;
  vst1q_s16(output, out[10]);
  output += 16;
  vst1q_s16(output, out[11]);
  output += 16;
  vst1q_s16(output, out[12]);
  output += 16;
  vst1q_s16(output, out[13]);
  output += 16;
  vst1q_s16(output, out[14]);
  output += 16;
  vst1q_s16(output, out[15]);
}

static INLINE void idct8x8_add8x1(const int16x8_t a, uint8_t **const dest,
                                  const int stride) {
  const uint8x8_t s = vld1_u8(*dest);
  const int16x8_t res = vrshrq_n_s16(a, 5);
  const uint16x8_t q = vaddw_u8(vreinterpretq_u16_s16(res), s);
  const uint8x8_t d = vqmovun_s16(vreinterpretq_s16_u16(q));
  vst1_u8(*dest, d);
  *dest += stride;
}

static INLINE void idct8x8_add8x8_neon(int16x8_t *const out, uint8_t *dest,
                                       const int stride) {
  idct8x8_add8x1(out[0], &dest, stride);
  idct8x8_add8x1(out[1], &dest, stride);
  idct8x8_add8x1(out[2], &dest, stride);
  idct8x8_add8x1(out[3], &dest, stride);
  idct8x8_add8x1(out[4], &dest, stride);
  idct8x8_add8x1(out[5], &dest, stride);
  idct8x8_add8x1(out[6], &dest, stride);
  idct8x8_add8x1(out[7], &dest, stride);
}

static INLINE void idct16x16_add8x1(const int16x8_t a, uint8_t **const dest,
                                    const int stride) {
  const uint8x8_t s = vld1_u8(*dest);
  const int16x8_t res = vrshrq_n_s16(a, 6);
  const uint16x8_t q = vaddw_u8(vreinterpretq_u16_s16(res), s);
  const uint8x8_t d = vqmovun_s16(vreinterpretq_s16_u16(q));
  vst1_u8(*dest, d);
  *dest += stride;
}

static INLINE void idct16x16_add_store(const int16x8_t *const out,
                                       uint8_t *dest, const int stride) {
  // Add the result to dest
  idct16x16_add8x1(out[0], &dest, stride);
  idct16x16_add8x1(out[1], &dest, stride);
  idct16x16_add8x1(out[2], &dest, stride);
  idct16x16_add8x1(out[3], &dest, stride);
  idct16x16_add8x1(out[4], &dest, stride);
  idct16x16_add8x1(out[5], &dest, stride);
  idct16x16_add8x1(out[6], &dest, stride);
  idct16x16_add8x1(out[7], &dest, stride);
  idct16x16_add8x1(out[8], &dest, stride);
  idct16x16_add8x1(out[9], &dest, stride);
  idct16x16_add8x1(out[10], &dest, stride);
  idct16x16_add8x1(out[11], &dest, stride);
  idct16x16_add8x1(out[12], &dest, stride);
  idct16x16_add8x1(out[13], &dest, stride);
  idct16x16_add8x1(out[14], &dest, stride);
  idct16x16_add8x1(out[15], &dest, stride);
}

static INLINE void highbd_idct16x16_add8x1(const int16x8_t a,
                                           const int16x8_t max,
                                           uint16_t **const dest,
                                           const int stride) {
  const uint16x8_t s = vld1q_u16(*dest);
  const int16x8_t res0 = vqaddq_s16(a, vreinterpretq_s16_u16(s));
  const int16x8_t res1 = vminq_s16(res0, max);
  const uint16x8_t d = vqshluq_n_s16(res1, 0);
  vst1q_u16(*dest, d);
  *dest += stride;
}

static INLINE void idct16x16_add_store_bd8(int16x8_t *const out, uint16_t *dest,
                                           const int stride) {
  // Add the result to dest
  const int16x8_t max = vdupq_n_s16((1 << 8) - 1);
  out[0] = vrshrq_n_s16(out[0], 6);
  out[1] = vrshrq_n_s16(out[1], 6);
  out[2] = vrshrq_n_s16(out[2], 6);
  out[3] = vrshrq_n_s16(out[3], 6);
  out[4] = vrshrq_n_s16(out[4], 6);
  out[5] = vrshrq_n_s16(out[5], 6);
  out[6] = vrshrq_n_s16(out[6], 6);
  out[7] = vrshrq_n_s16(out[7], 6);
  out[8] = vrshrq_n_s16(out[8], 6);
  out[9] = vrshrq_n_s16(out[9], 6);
  out[10] = vrshrq_n_s16(out[10], 6);
  out[11] = vrshrq_n_s16(out[11], 6);
  out[12] = vrshrq_n_s16(out[12], 6);
  out[13] = vrshrq_n_s16(out[13], 6);
  out[14] = vrshrq_n_s16(out[14], 6);
  out[15] = vrshrq_n_s16(out[15], 6);
  highbd_idct16x16_add8x1(out[0], max, &dest, stride);
  highbd_idct16x16_add8x1(out[1], max, &dest, stride);
  highbd_idct16x16_add8x1(out[2], max, &dest, stride);
  highbd_idct16x16_add8x1(out[3], max, &dest, stride);
  highbd_idct16x16_add8x1(out[4], max, &dest, stride);
  highbd_idct16x16_add8x1(out[5], max, &dest, stride);
  highbd_idct16x16_add8x1(out[6], max, &dest, stride);
  highbd_idct16x16_add8x1(out[7], max, &dest, stride);
  highbd_idct16x16_add8x1(out[8], max, &dest, stride);
  highbd_idct16x16_add8x1(out[9], max, &dest, stride);
  highbd_idct16x16_add8x1(out[10], max, &dest, stride);
  highbd_idct16x16_add8x1(out[11], max, &dest, stride);
  highbd_idct16x16_add8x1(out[12], max, &dest, stride);
  highbd_idct16x16_add8x1(out[13], max, &dest, stride);
  highbd_idct16x16_add8x1(out[14], max, &dest, stride);
  highbd_idct16x16_add8x1(out[15], max, &dest, stride);
}

static INLINE void highbd_idct16x16_add8x1_bd8(const int16x8_t a,
                                               uint16_t **const dest,
                                               const int stride) {
  const uint16x8_t s = vld1q_u16(*dest);
  const int16x8_t res = vrsraq_n_s16(vreinterpretq_s16_u16(s), a, 6);
  const uint16x8_t d = vmovl_u8(vqmovun_s16(res));
  vst1q_u16(*dest, d);
  *dest += stride;
}

static INLINE void highbd_add_and_store_bd8(const int16x8_t *const a,
                                            uint16_t *out, const int stride) {
  highbd_idct16x16_add8x1_bd8(a[0], &out, stride);
  highbd_idct16x16_add8x1_bd8(a[1], &out, stride);
  highbd_idct16x16_add8x1_bd8(a[2], &out, stride);
  highbd_idct16x16_add8x1_bd8(a[3], &out, stride);
  highbd_idct16x16_add8x1_bd8(a[4], &out, stride);
  highbd_idct16x16_add8x1_bd8(a[5], &out, stride);
  highbd_idct16x16_add8x1_bd8(a[6], &out, stride);
  highbd_idct16x16_add8x1_bd8(a[7], &out, stride);
  highbd_idct16x16_add8x1_bd8(a[8], &out, stride);
  highbd_idct16x16_add8x1_bd8(a[9], &out, stride);
  highbd_idct16x16_add8x1_bd8(a[10], &out, stride);
  highbd_idct16x16_add8x1_bd8(a[11], &out, stride);
  highbd_idct16x16_add8x1_bd8(a[12], &out, stride);
  highbd_idct16x16_add8x1_bd8(a[13], &out, stride);
  highbd_idct16x16_add8x1_bd8(a[14], &out, stride);
  highbd_idct16x16_add8x1_bd8(a[15], &out, stride);
  highbd_idct16x16_add8x1_bd8(a[16], &out, stride);
  highbd_idct16x16_add8x1_bd8(a[17], &out, stride);
  highbd_idct16x16_add8x1_bd8(a[18], &out, stride);
  highbd_idct16x16_add8x1_bd8(a[19], &out, stride);
  highbd_idct16x16_add8x1_bd8(a[20], &out, stride);
  highbd_idct16x16_add8x1_bd8(a[21], &out, stride);
  highbd_idct16x16_add8x1_bd8(a[22], &out, stride);
  highbd_idct16x16_add8x1_bd8(a[23], &out, stride);
  highbd_idct16x16_add8x1_bd8(a[24], &out, stride);
  highbd_idct16x16_add8x1_bd8(a[25], &out, stride);
  highbd_idct16x16_add8x1_bd8(a[26], &out, stride);
  highbd_idct16x16_add8x1_bd8(a[27], &out, stride);
  highbd_idct16x16_add8x1_bd8(a[28], &out, stride);
  highbd_idct16x16_add8x1_bd8(a[29], &out, stride);
  highbd_idct16x16_add8x1_bd8(a[30], &out, stride);
  highbd_idct16x16_add8x1_bd8(a[31], &out, stride);
}

void vpx_idct16x16_256_add_half1d(const void *const input, int16_t *output,
                                  void *const dest, const int stride,
                                  const int highbd_flag);

void vpx_idct16x16_38_add_half1d(const void *const input, int16_t *const output,
                                 void *const dest, const int stride,
                                 const int highbd_flag);

void vpx_idct16x16_10_add_half1d_pass1(const tran_low_t *input,
                                       int16_t *output);

void vpx_idct16x16_10_add_half1d_pass2(const int16_t *input,
                                       int16_t *const output, void *const dest,
                                       const int stride, const int highbd_flag);

void vpx_idct32_32_neon(const tran_low_t *input, uint8_t *dest,
                        const int stride, const int highbd_flag);

void vpx_idct32_12_neon(const tran_low_t *const input, int16_t *output);
void vpx_idct32_16_neon(const int16_t *const input, void *const output,
                        const int stride, const int highbd_flag);

void vpx_idct32_6_neon(const tran_low_t *input, int16_t *output);
void vpx_idct32_8_neon(const int16_t *input, void *const output, int stride,
                       const int highbd_flag);

#endif  // VPX_VPX_DSP_ARM_IDCT_NEON_H_
