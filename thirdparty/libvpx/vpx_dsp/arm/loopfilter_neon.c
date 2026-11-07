/*
 *  Copyright (c) 2016 The WebM project authors. All Rights Reserved.
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
#include "vpx_dsp/arm/transpose_neon.h"

// For all the static inline functions, the functions ending with '_8' process
// 8 samples in a bunch, and the functions ending with '_16' process 16 samples
// in a bunch.

#define FUN_LOAD_THRESH(w, r)                                             \
  static INLINE void load_thresh_##w(                                     \
      const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh, \
      uint8x##w##_t *blimit_vec, uint8x##w##_t *limit_vec,                \
      uint8x##w##_t *thresh_vec) {                                        \
    *blimit_vec = vld1##r##dup_u8(blimit);                                \
    *limit_vec = vld1##r##dup_u8(limit);                                  \
    *thresh_vec = vld1##r##dup_u8(thresh);                                \
  }

FUN_LOAD_THRESH(8, _)    // load_thresh_8
FUN_LOAD_THRESH(16, q_)  // load_thresh_16
#undef FUN_LOAD_THRESH

static INLINE void load_thresh_8_dual(
    const uint8_t *blimit0, const uint8_t *limit0, const uint8_t *thresh0,
    const uint8_t *blimit1, const uint8_t *limit1, const uint8_t *thresh1,
    uint8x16_t *blimit_vec, uint8x16_t *limit_vec, uint8x16_t *thresh_vec) {
  *blimit_vec = vcombine_u8(vld1_dup_u8(blimit0), vld1_dup_u8(blimit1));
  *limit_vec = vcombine_u8(vld1_dup_u8(limit0), vld1_dup_u8(limit1));
  *thresh_vec = vcombine_u8(vld1_dup_u8(thresh0), vld1_dup_u8(thresh1));
}

// Here flat is 64-bit long, with each 8-bit (or 4-bit) chunk being a mask of a
// pixel. When used to control filter branches, we only detect whether it is all
// 0s or all 1s. We pairwise add flat to a 32-bit long number flat_status.
// flat equals 0 if and only if flat_status equals 0.
// flat equals -1 (all 1s) if and only if flat_status equals -2. (This is true
// because each mask occupies more than 1 bit.)
static INLINE uint32_t calc_flat_status_8(uint8x8_t flat) {
  return vget_lane_u32(
      vreinterpret_u32_u64(vpaddl_u32(vreinterpret_u32_u8(flat))), 0);
}

// Here flat is 128-bit long, with each 8-bit chunk being a mask of a pixel.
// When used to control filter branches, we only detect whether it is all 0s or
// all 1s. We narrowing shift right each 16-bit chunk by 4 arithmetically, so
// we get a 64-bit long number, with each 4-bit chunk being a mask of a pixel.
// Then we pairwise add flat to a 32-bit long number flat_status.
// flat equals 0 if and only if flat_status equals 0.
// flat equals -1 (all 1s) if and only if flat_status equals -2. (This is true
// because each mask occupies more than 1 bit.)
static INLINE uint32_t calc_flat_status_16(uint8x16_t flat) {
  const uint8x8_t flat_4bit =
      vreinterpret_u8_s8(vshrn_n_s16(vreinterpretq_s16_u8(flat), 4));
  return calc_flat_status_8(flat_4bit);
}

#define FUN_FILTER_HEV_MASK4(w, r)                                            \
  static INLINE uint8x##w##_t filter_hev_mask4_##w(                           \
      const uint8x##w##_t limit, const uint8x##w##_t blimit,                  \
      const uint8x##w##_t thresh, const uint8x##w##_t p3,                     \
      const uint8x##w##_t p2, const uint8x##w##_t p1, const uint8x##w##_t p0, \
      const uint8x##w##_t q0, const uint8x##w##_t q1, const uint8x##w##_t q2, \
      const uint8x##w##_t q3, uint8x##w##_t *hev, uint8x##w##_t *mask) {      \
    uint8x##w##_t max, t0, t1;                                                \
                                                                              \
    max = vabd##r##u8(p1, p0);                                                \
    max = vmax##r##u8(max, vabd##r##u8(q1, q0));                              \
    *hev = vcgt##r##u8(max, thresh);                                          \
    *mask = vmax##r##u8(max, vabd##r##u8(p3, p2));                            \
    *mask = vmax##r##u8(*mask, vabd##r##u8(p2, p1));                          \
    *mask = vmax##r##u8(*mask, vabd##r##u8(q2, q1));                          \
    *mask = vmax##r##u8(*mask, vabd##r##u8(q3, q2));                          \
    t0 = vabd##r##u8(p0, q0);                                                 \
    t1 = vabd##r##u8(p1, q1);                                                 \
    t0 = vqadd##r##u8(t0, t0);                                                \
    t1 = vshr##r##n_u8(t1, 1);                                                \
    t0 = vqadd##r##u8(t0, t1);                                                \
    *mask = vcle##r##u8(*mask, limit);                                        \
    t0 = vcle##r##u8(t0, blimit);                                             \
    *mask = vand##r##u8(*mask, t0);                                           \
                                                                              \
    return max;                                                               \
  }

FUN_FILTER_HEV_MASK4(8, _)    // filter_hev_mask4_8
FUN_FILTER_HEV_MASK4(16, q_)  // filter_hev_mask4_16
#undef FUN_FILTER_HEV_MASK4

#define FUN_FILTER_FLAT_HEV_MASK(w, r)                                        \
  static INLINE uint8x##w##_t filter_flat_hev_mask_##w(                       \
      const uint8x##w##_t limit, const uint8x##w##_t blimit,                  \
      const uint8x##w##_t thresh, const uint8x##w##_t p3,                     \
      const uint8x##w##_t p2, const uint8x##w##_t p1, const uint8x##w##_t p0, \
      const uint8x##w##_t q0, const uint8x##w##_t q1, const uint8x##w##_t q2, \
      const uint8x##w##_t q3, uint8x##w##_t *flat, uint32_t *flat_status,     \
      uint8x##w##_t *hev) {                                                   \
    uint8x##w##_t max, mask;                                                  \
                                                                              \
    max = filter_hev_mask4_##w(limit, blimit, thresh, p3, p2, p1, p0, q0, q1, \
                               q2, q3, hev, &mask);                           \
    *flat = vmax##r##u8(max, vabd##r##u8(p2, p0));                            \
    *flat = vmax##r##u8(*flat, vabd##r##u8(q2, q0));                          \
    *flat = vmax##r##u8(*flat, vabd##r##u8(p3, p0));                          \
    *flat = vmax##r##u8(*flat, vabd##r##u8(q3, q0));                          \
    *flat = vcle##r##u8(*flat, vdup##r##n_u8(1)); /* flat_mask4() */          \
    *flat = vand##r##u8(*flat, mask);                                         \
    *flat_status = calc_flat_status_##w(*flat);                               \
                                                                              \
    return mask;                                                              \
  }

FUN_FILTER_FLAT_HEV_MASK(8, _)    // filter_flat_hev_mask_8
FUN_FILTER_FLAT_HEV_MASK(16, q_)  // filter_flat_hev_mask_16
#undef FUN_FILTER_FLAT_HEV_MASK

#define FUN_FLAT_MASK5(w, r)                                                  \
  static INLINE uint8x##w##_t flat_mask5_##w(                                 \
      const uint8x##w##_t p4, const uint8x##w##_t p3, const uint8x##w##_t p2, \
      const uint8x##w##_t p1, const uint8x##w##_t p0, const uint8x##w##_t q0, \
      const uint8x##w##_t q1, const uint8x##w##_t q2, const uint8x##w##_t q3, \
      const uint8x##w##_t q4, const uint8x##w##_t flat,                       \
      uint32_t *flat2_status) {                                               \
    uint8x##w##_t flat2 = vabd##r##u8(p4, p0);                                \
    flat2 = vmax##r##u8(flat2, vabd##r##u8(p3, p0));                          \
    flat2 = vmax##r##u8(flat2, vabd##r##u8(p2, p0));                          \
    flat2 = vmax##r##u8(flat2, vabd##r##u8(p1, p0));                          \
    flat2 = vmax##r##u8(flat2, vabd##r##u8(q1, q0));                          \
    flat2 = vmax##r##u8(flat2, vabd##r##u8(q2, q0));                          \
    flat2 = vmax##r##u8(flat2, vabd##r##u8(q3, q0));                          \
    flat2 = vmax##r##u8(flat2, vabd##r##u8(q4, q0));                          \
    flat2 = vcle##r##u8(flat2, vdup##r##n_u8(1));                             \
    flat2 = vand##r##u8(flat2, flat);                                         \
    *flat2_status = calc_flat_status_##w(flat2);                              \
                                                                              \
    return flat2;                                                             \
  }

FUN_FLAT_MASK5(8, _)    // flat_mask5_8
FUN_FLAT_MASK5(16, q_)  // flat_mask5_16
#undef FUN_FLAT_MASK5

#define FUN_FLIP_SIGN(w, r)                                         \
  static INLINE int8x##w##_t flip_sign_##w(const uint8x##w##_t v) { \
    const uint8x##w##_t sign_bit = vdup##r##n_u8(0x80);             \
    return vreinterpret##r##s8_u8(veor##r##u8(v, sign_bit));        \
  }

FUN_FLIP_SIGN(8, _)    // flip_sign_8
FUN_FLIP_SIGN(16, q_)  // flip_sign_16
#undef FUN_FLIP_SIGN

#define FUN_FLIP_SIGN_BACK(w, r)                                         \
  static INLINE uint8x##w##_t flip_sign_back_##w(const int8x##w##_t v) { \
    const int8x##w##_t sign_bit = vdup##r##n_s8((int8_t)0x80);           \
    return vreinterpret##r##u8_s8(veor##r##s8(v, sign_bit));             \
  }

FUN_FLIP_SIGN_BACK(8, _)    // flip_sign_back_8
FUN_FLIP_SIGN_BACK(16, q_)  // flip_sign_back_16
#undef FUN_FLIP_SIGN_BACK

static INLINE void filter_update_8(const uint8x8_t sub0, const uint8x8_t sub1,
                                   const uint8x8_t add0, const uint8x8_t add1,
                                   uint16x8_t *sum) {
  *sum = vsubw_u8(*sum, sub0);
  *sum = vsubw_u8(*sum, sub1);
  *sum = vaddw_u8(*sum, add0);
  *sum = vaddw_u8(*sum, add1);
}

static INLINE void filter_update_16(const uint8x16_t sub0,
                                    const uint8x16_t sub1,
                                    const uint8x16_t add0,
                                    const uint8x16_t add1, uint16x8_t *sum0,
                                    uint16x8_t *sum1) {
  *sum0 = vsubw_u8(*sum0, vget_low_u8(sub0));
  *sum1 = vsubw_u8(*sum1, vget_high_u8(sub0));
  *sum0 = vsubw_u8(*sum0, vget_low_u8(sub1));
  *sum1 = vsubw_u8(*sum1, vget_high_u8(sub1));
  *sum0 = vaddw_u8(*sum0, vget_low_u8(add0));
  *sum1 = vaddw_u8(*sum1, vget_high_u8(add0));
  *sum0 = vaddw_u8(*sum0, vget_low_u8(add1));
  *sum1 = vaddw_u8(*sum1, vget_high_u8(add1));
}

static INLINE uint8x8_t calc_7_tap_filter_8_kernel(const uint8x8_t sub0,
                                                   const uint8x8_t sub1,
                                                   const uint8x8_t add0,
                                                   const uint8x8_t add1,
                                                   uint16x8_t *sum) {
  filter_update_8(sub0, sub1, add0, add1, sum);
  return vrshrn_n_u16(*sum, 3);
}

static INLINE uint8x16_t calc_7_tap_filter_16_kernel(
    const uint8x16_t sub0, const uint8x16_t sub1, const uint8x16_t add0,
    const uint8x16_t add1, uint16x8_t *sum0, uint16x8_t *sum1) {
  filter_update_16(sub0, sub1, add0, add1, sum0, sum1);
  return vcombine_u8(vrshrn_n_u16(*sum0, 3), vrshrn_n_u16(*sum1, 3));
}

static INLINE uint8x8_t apply_15_tap_filter_8_kernel(
    const uint8x8_t flat, const uint8x8_t sub0, const uint8x8_t sub1,
    const uint8x8_t add0, const uint8x8_t add1, const uint8x8_t in,
    uint16x8_t *sum) {
  filter_update_8(sub0, sub1, add0, add1, sum);
  return vbsl_u8(flat, vrshrn_n_u16(*sum, 4), in);
}

static INLINE uint8x16_t apply_15_tap_filter_16_kernel(
    const uint8x16_t flat, const uint8x16_t sub0, const uint8x16_t sub1,
    const uint8x16_t add0, const uint8x16_t add1, const uint8x16_t in,
    uint16x8_t *sum0, uint16x8_t *sum1) {
  uint8x16_t t;
  filter_update_16(sub0, sub1, add0, add1, sum0, sum1);
  t = vcombine_u8(vrshrn_n_u16(*sum0, 4), vrshrn_n_u16(*sum1, 4));
  return vbslq_u8(flat, t, in);
}

// 7-tap filter [1, 1, 1, 2, 1, 1, 1]
static INLINE void calc_7_tap_filter_8(const uint8x8_t p3, const uint8x8_t p2,
                                       const uint8x8_t p1, const uint8x8_t p0,
                                       const uint8x8_t q0, const uint8x8_t q1,
                                       const uint8x8_t q2, const uint8x8_t q3,
                                       uint8x8_t *op2, uint8x8_t *op1,
                                       uint8x8_t *op0, uint8x8_t *oq0,
                                       uint8x8_t *oq1, uint8x8_t *oq2) {
  uint16x8_t sum;
  sum = vaddl_u8(p3, p3);   // 2*p3
  sum = vaddw_u8(sum, p3);  // 3*p3
  sum = vaddw_u8(sum, p2);  // 3*p3+p2
  sum = vaddw_u8(sum, p2);  // 3*p3+2*p2
  sum = vaddw_u8(sum, p1);  // 3*p3+2*p2+p1
  sum = vaddw_u8(sum, p0);  // 3*p3+2*p2+p1+p0
  sum = vaddw_u8(sum, q0);  // 3*p3+2*p2+p1+p0+q0
  *op2 = vrshrn_n_u16(sum, 3);
  *op1 = calc_7_tap_filter_8_kernel(p3, p2, p1, q1, &sum);
  *op0 = calc_7_tap_filter_8_kernel(p3, p1, p0, q2, &sum);
  *oq0 = calc_7_tap_filter_8_kernel(p3, p0, q0, q3, &sum);
  *oq1 = calc_7_tap_filter_8_kernel(p2, q0, q1, q3, &sum);
  *oq2 = calc_7_tap_filter_8_kernel(p1, q1, q2, q3, &sum);
}

static INLINE void calc_7_tap_filter_16(
    const uint8x16_t p3, const uint8x16_t p2, const uint8x16_t p1,
    const uint8x16_t p0, const uint8x16_t q0, const uint8x16_t q1,
    const uint8x16_t q2, const uint8x16_t q3, uint8x16_t *op2, uint8x16_t *op1,
    uint8x16_t *op0, uint8x16_t *oq0, uint8x16_t *oq1, uint8x16_t *oq2) {
  uint16x8_t sum0, sum1;
  sum0 = vaddl_u8(vget_low_u8(p3), vget_low_u8(p3));    // 2*p3
  sum1 = vaddl_u8(vget_high_u8(p3), vget_high_u8(p3));  // 2*p3
  sum0 = vaddw_u8(sum0, vget_low_u8(p3));               // 3*p3
  sum1 = vaddw_u8(sum1, vget_high_u8(p3));              // 3*p3
  sum0 = vaddw_u8(sum0, vget_low_u8(p2));               // 3*p3+p2
  sum1 = vaddw_u8(sum1, vget_high_u8(p2));              // 3*p3+p2
  sum0 = vaddw_u8(sum0, vget_low_u8(p2));               // 3*p3+2*p2
  sum1 = vaddw_u8(sum1, vget_high_u8(p2));              // 3*p3+2*p2
  sum0 = vaddw_u8(sum0, vget_low_u8(p1));               // 3*p3+2*p2+p1
  sum1 = vaddw_u8(sum1, vget_high_u8(p1));              // 3*p3+2*p2+p1
  sum0 = vaddw_u8(sum0, vget_low_u8(p0));               // 3*p3+2*p2+p1+p0
  sum1 = vaddw_u8(sum1, vget_high_u8(p0));              // 3*p3+2*p2+p1+p0
  sum0 = vaddw_u8(sum0, vget_low_u8(q0));               // 3*p3+2*p2+p1+p0+q0
  sum1 = vaddw_u8(sum1, vget_high_u8(q0));              // 3*p3+2*p2+p1+p0+q0
  *op2 = vcombine_u8(vrshrn_n_u16(sum0, 3), vrshrn_n_u16(sum1, 3));
  *op1 = calc_7_tap_filter_16_kernel(p3, p2, p1, q1, &sum0, &sum1);
  *op0 = calc_7_tap_filter_16_kernel(p3, p1, p0, q2, &sum0, &sum1);
  *oq0 = calc_7_tap_filter_16_kernel(p3, p0, q0, q3, &sum0, &sum1);
  *oq1 = calc_7_tap_filter_16_kernel(p2, q0, q1, q3, &sum0, &sum1);
  *oq2 = calc_7_tap_filter_16_kernel(p1, q1, q2, q3, &sum0, &sum1);
}

#define FUN_APPLY_7_TAP_FILTER(w, r)                                          \
  static INLINE void apply_7_tap_filter_##w(                                  \
      const uint8x##w##_t flat, const uint8x##w##_t p3,                       \
      const uint8x##w##_t p2, const uint8x##w##_t p1, const uint8x##w##_t p0, \
      const uint8x##w##_t q0, const uint8x##w##_t q1, const uint8x##w##_t q2, \
      const uint8x##w##_t q3, uint8x##w##_t *op2, uint8x##w##_t *op1,         \
      uint8x##w##_t *op0, uint8x##w##_t *oq0, uint8x##w##_t *oq1,             \
      uint8x##w##_t *oq2) {                                                   \
    uint8x##w##_t tp1, tp0, tq0, tq1;                                         \
    calc_7_tap_filter_##w(p3, p2, p1, p0, q0, q1, q2, q3, op2, &tp1, &tp0,    \
                          &tq0, &tq1, oq2);                                   \
    *op2 = vbsl##r##u8(flat, *op2, p2);                                       \
    *op1 = vbsl##r##u8(flat, tp1, *op1);                                      \
    *op0 = vbsl##r##u8(flat, tp0, *op0);                                      \
    *oq0 = vbsl##r##u8(flat, tq0, *oq0);                                      \
    *oq1 = vbsl##r##u8(flat, tq1, *oq1);                                      \
    *oq2 = vbsl##r##u8(flat, *oq2, q2);                                       \
  }

FUN_APPLY_7_TAP_FILTER(8, _)    // apply_7_tap_filter_8
FUN_APPLY_7_TAP_FILTER(16, q_)  // apply_7_tap_filter_16
#undef FUN_APPLY_7_TAP_FILTER

// 15-tap filter [1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1]
static INLINE void apply_15_tap_filter_8(
    const uint8x8_t flat2, const uint8x8_t p7, const uint8x8_t p6,
    const uint8x8_t p5, const uint8x8_t p4, const uint8x8_t p3,
    const uint8x8_t p2, const uint8x8_t p1, const uint8x8_t p0,
    const uint8x8_t q0, const uint8x8_t q1, const uint8x8_t q2,
    const uint8x8_t q3, const uint8x8_t q4, const uint8x8_t q5,
    const uint8x8_t q6, const uint8x8_t q7, uint8x8_t *op6, uint8x8_t *op5,
    uint8x8_t *op4, uint8x8_t *op3, uint8x8_t *op2, uint8x8_t *op1,
    uint8x8_t *op0, uint8x8_t *oq0, uint8x8_t *oq1, uint8x8_t *oq2,
    uint8x8_t *oq3, uint8x8_t *oq4, uint8x8_t *oq5, uint8x8_t *oq6) {
  uint16x8_t sum;
  sum = vshll_n_u8(p7, 3);  // 8*p7
  sum = vsubw_u8(sum, p7);  // 7*p7
  sum = vaddw_u8(sum, p6);  // 7*p7+p6
  sum = vaddw_u8(sum, p6);  // 7*p7+2*p6
  sum = vaddw_u8(sum, p5);  // 7*p7+2*p6+p5
  sum = vaddw_u8(sum, p4);  // 7*p7+2*p6+p5+p4
  sum = vaddw_u8(sum, p3);  // 7*p7+2*p6+p5+p4+p3
  sum = vaddw_u8(sum, p2);  // 7*p7+2*p6+p5+p4+p3+p2
  sum = vaddw_u8(sum, p1);  // 7*p7+2*p6+p5+p4+p3+p2+p1
  sum = vaddw_u8(sum, p0);  // 7*p7+2*p6+p5+p4+p3+p2+p1+p0
  sum = vaddw_u8(sum, q0);  // 7*p7+2*p6+p5+p4+p3+p2+p1+p0+q0
  *op6 = vbsl_u8(flat2, vrshrn_n_u16(sum, 4), p6);
  *op5 = apply_15_tap_filter_8_kernel(flat2, p7, p6, p5, q1, p5, &sum);
  *op4 = apply_15_tap_filter_8_kernel(flat2, p7, p5, p4, q2, p4, &sum);
  *op3 = apply_15_tap_filter_8_kernel(flat2, p7, p4, p3, q3, p3, &sum);
  *op2 = apply_15_tap_filter_8_kernel(flat2, p7, p3, p2, q4, *op2, &sum);
  *op1 = apply_15_tap_filter_8_kernel(flat2, p7, p2, p1, q5, *op1, &sum);
  *op0 = apply_15_tap_filter_8_kernel(flat2, p7, p1, p0, q6, *op0, &sum);
  *oq0 = apply_15_tap_filter_8_kernel(flat2, p7, p0, q0, q7, *oq0, &sum);
  *oq1 = apply_15_tap_filter_8_kernel(flat2, p6, q0, q1, q7, *oq1, &sum);
  *oq2 = apply_15_tap_filter_8_kernel(flat2, p5, q1, q2, q7, *oq2, &sum);
  *oq3 = apply_15_tap_filter_8_kernel(flat2, p4, q2, q3, q7, q3, &sum);
  *oq4 = apply_15_tap_filter_8_kernel(flat2, p3, q3, q4, q7, q4, &sum);
  *oq5 = apply_15_tap_filter_8_kernel(flat2, p2, q4, q5, q7, q5, &sum);
  *oq6 = apply_15_tap_filter_8_kernel(flat2, p1, q5, q6, q7, q6, &sum);
}

static INLINE void apply_15_tap_filter_16(
    const uint8x16_t flat2, const uint8x16_t p7, const uint8x16_t p6,
    const uint8x16_t p5, const uint8x16_t p4, const uint8x16_t p3,
    const uint8x16_t p2, const uint8x16_t p1, const uint8x16_t p0,
    const uint8x16_t q0, const uint8x16_t q1, const uint8x16_t q2,
    const uint8x16_t q3, const uint8x16_t q4, const uint8x16_t q5,
    const uint8x16_t q6, const uint8x16_t q7, uint8x16_t *op6, uint8x16_t *op5,
    uint8x16_t *op4, uint8x16_t *op3, uint8x16_t *op2, uint8x16_t *op1,
    uint8x16_t *op0, uint8x16_t *oq0, uint8x16_t *oq1, uint8x16_t *oq2,
    uint8x16_t *oq3, uint8x16_t *oq4, uint8x16_t *oq5, uint8x16_t *oq6) {
  uint16x8_t sum0, sum1;
  uint8x16_t t;
  sum0 = vshll_n_u8(vget_low_u8(p7), 3);    // 8*p7
  sum1 = vshll_n_u8(vget_high_u8(p7), 3);   // 8*p7
  sum0 = vsubw_u8(sum0, vget_low_u8(p7));   // 7*p7
  sum1 = vsubw_u8(sum1, vget_high_u8(p7));  // 7*p7
  sum0 = vaddw_u8(sum0, vget_low_u8(p6));   // 7*p7+p6
  sum1 = vaddw_u8(sum1, vget_high_u8(p6));  // 7*p7+p6
  sum0 = vaddw_u8(sum0, vget_low_u8(p6));   // 7*p7+2*p6
  sum1 = vaddw_u8(sum1, vget_high_u8(p6));  // 7*p7+2*p6
  sum0 = vaddw_u8(sum0, vget_low_u8(p5));   // 7*p7+2*p6+p5
  sum1 = vaddw_u8(sum1, vget_high_u8(p5));  // 7*p7+2*p6+p5
  sum0 = vaddw_u8(sum0, vget_low_u8(p4));   // 7*p7+2*p6+p5+p4
  sum1 = vaddw_u8(sum1, vget_high_u8(p4));  // 7*p7+2*p6+p5+p4
  sum0 = vaddw_u8(sum0, vget_low_u8(p3));   // 7*p7+2*p6+p5+p4+p3
  sum1 = vaddw_u8(sum1, vget_high_u8(p3));  // 7*p7+2*p6+p5+p4+p3
  sum0 = vaddw_u8(sum0, vget_low_u8(p2));   // 7*p7+2*p6+p5+p4+p3+p2
  sum1 = vaddw_u8(sum1, vget_high_u8(p2));  // 7*p7+2*p6+p5+p4+p3+p2
  sum0 = vaddw_u8(sum0, vget_low_u8(p1));   // 7*p7+2*p6+p5+p4+p3+p2+p1
  sum1 = vaddw_u8(sum1, vget_high_u8(p1));  // 7*p7+2*p6+p5+p4+p3+p2+p1
  sum0 = vaddw_u8(sum0, vget_low_u8(p0));   // 7*p7+2*p6+p5+p4+p3+p2+p1+p0
  sum1 = vaddw_u8(sum1, vget_high_u8(p0));  // 7*p7+2*p6+p5+p4+p3+p2+p1+p0
  sum0 = vaddw_u8(sum0, vget_low_u8(q0));   // 7*p7+2*p6+p5+p4+p3+p2+p1+p0+q0
  sum1 = vaddw_u8(sum1, vget_high_u8(q0));  // 7*p7+2*p6+p5+p4+p3+p2+p1+p0+q0
  t = vcombine_u8(vrshrn_n_u16(sum0, 4), vrshrn_n_u16(sum1, 4));
  *op6 = vbslq_u8(flat2, t, p6);
  *op5 = apply_15_tap_filter_16_kernel(flat2, p7, p6, p5, q1, p5, &sum0, &sum1);
  *op4 = apply_15_tap_filter_16_kernel(flat2, p7, p5, p4, q2, p4, &sum0, &sum1);
  *op3 = apply_15_tap_filter_16_kernel(flat2, p7, p4, p3, q3, p3, &sum0, &sum1);
  *op2 =
      apply_15_tap_filter_16_kernel(flat2, p7, p3, p2, q4, *op2, &sum0, &sum1);
  *op1 =
      apply_15_tap_filter_16_kernel(flat2, p7, p2, p1, q5, *op1, &sum0, &sum1);
  *op0 =
      apply_15_tap_filter_16_kernel(flat2, p7, p1, p0, q6, *op0, &sum0, &sum1);
  *oq0 =
      apply_15_tap_filter_16_kernel(flat2, p7, p0, q0, q7, *oq0, &sum0, &sum1);
  *oq1 =
      apply_15_tap_filter_16_kernel(flat2, p6, q0, q1, q7, *oq1, &sum0, &sum1);
  *oq2 =
      apply_15_tap_filter_16_kernel(flat2, p5, q1, q2, q7, *oq2, &sum0, &sum1);
  *oq3 = apply_15_tap_filter_16_kernel(flat2, p4, q2, q3, q7, q3, &sum0, &sum1);
  *oq4 = apply_15_tap_filter_16_kernel(flat2, p3, q3, q4, q7, q4, &sum0, &sum1);
  *oq5 = apply_15_tap_filter_16_kernel(flat2, p2, q4, q5, q7, q5, &sum0, &sum1);
  *oq6 = apply_15_tap_filter_16_kernel(flat2, p1, q5, q6, q7, q6, &sum0, &sum1);
}

#define FUN_FILTER4(w, r)                                                     \
  static INLINE void filter4_##w(                                             \
      const uint8x##w##_t mask, const uint8x##w##_t hev,                      \
      const uint8x##w##_t p1, const uint8x##w##_t p0, const uint8x##w##_t q0, \
      const uint8x##w##_t q1, uint8x##w##_t *op1, uint8x##w##_t *op0,         \
      uint8x##w##_t *oq0, uint8x##w##_t *oq1) {                               \
    int8x##w##_t filter, filter1, filter2, t;                                 \
    int8x##w##_t ps1 = flip_sign_##w(p1);                                     \
    int8x##w##_t ps0 = flip_sign_##w(p0);                                     \
    int8x##w##_t qs0 = flip_sign_##w(q0);                                     \
    int8x##w##_t qs1 = flip_sign_##w(q1);                                     \
                                                                              \
    /* add outer taps if we have high edge variance */                        \
    filter = vqsub##r##s8(ps1, qs1);                                          \
    filter = vand##r##s8(filter, vreinterpret##r##s8_u8(hev));                \
    t = vqsub##r##s8(qs0, ps0);                                               \
                                                                              \
    /* inner taps */                                                          \
    filter = vqadd##r##s8(filter, t);                                         \
    filter = vqadd##r##s8(filter, t);                                         \
    filter = vqadd##r##s8(filter, t);                                         \
    filter = vand##r##s8(filter, vreinterpret##r##s8_u8(mask));               \
                                                                              \
    /* save bottom 3 bits so that we round one side +4 and the other +3 */    \
    /* if it equals 4 we'll set it to adjust by -1 to account for the fact */ \
    /* we'd round it by 3 the other way */                                    \
    filter1 = vshr##r##n_s8(vqadd##r##s8(filter, vdup##r##n_s8(4)), 3);       \
    filter2 = vshr##r##n_s8(vqadd##r##s8(filter, vdup##r##n_s8(3)), 3);       \
                                                                              \
    qs0 = vqsub##r##s8(qs0, filter1);                                         \
    ps0 = vqadd##r##s8(ps0, filter2);                                         \
    *oq0 = flip_sign_back_##w(qs0);                                           \
    *op0 = flip_sign_back_##w(ps0);                                           \
                                                                              \
    /* outer tap adjustments */                                               \
    filter = vrshr##r##n_s8(filter1, 1);                                      \
    filter = vbic##r##s8(filter, vreinterpret##r##s8_u8(hev));                \
                                                                              \
    qs1 = vqsub##r##s8(qs1, filter);                                          \
    ps1 = vqadd##r##s8(ps1, filter);                                          \
    *oq1 = flip_sign_back_##w(qs1);                                           \
    *op1 = flip_sign_back_##w(ps1);                                           \
  }

FUN_FILTER4(8, _)    // filter4_8
FUN_FILTER4(16, q_)  // filter4_16
#undef FUN_FILTER4

#define FUN_FILTER8(w)                                                         \
  static INLINE void filter8_##w(                                              \
      const uint8x##w##_t mask, const uint8x##w##_t flat,                      \
      const uint32_t flat_status, const uint8x##w##_t hev,                     \
      const uint8x##w##_t p3, const uint8x##w##_t p2, const uint8x##w##_t p1,  \
      const uint8x##w##_t p0, const uint8x##w##_t q0, const uint8x##w##_t q1,  \
      const uint8x##w##_t q2, const uint8x##w##_t q3, uint8x##w##_t *op2,      \
      uint8x##w##_t *op1, uint8x##w##_t *op0, uint8x##w##_t *oq0,              \
      uint8x##w##_t *oq1, uint8x##w##_t *oq2) {                                \
    if (flat_status != (uint32_t)-2) {                                         \
      filter4_##w(mask, hev, p1, p0, q0, q1, op1, op0, oq0, oq1);              \
      *op2 = p2;                                                               \
      *oq2 = q2;                                                               \
      if (flat_status) {                                                       \
        apply_7_tap_filter_##w(flat, p3, p2, p1, p0, q0, q1, q2, q3, op2, op1, \
                               op0, oq0, oq1, oq2);                            \
      }                                                                        \
    } else {                                                                   \
      calc_7_tap_filter_##w(p3, p2, p1, p0, q0, q1, q2, q3, op2, op1, op0,     \
                            oq0, oq1, oq2);                                    \
    }                                                                          \
  }

FUN_FILTER8(8)   // filter8_8
FUN_FILTER8(16)  // filter8_16
#undef FUN_FILTER8

#define FUN_FILTER16(w)                                                        \
  static INLINE void filter16_##w(                                             \
      const uint8x##w##_t mask, const uint8x##w##_t flat,                      \
      const uint32_t flat_status, const uint8x##w##_t flat2,                   \
      const uint32_t flat2_status, const uint8x##w##_t hev,                    \
      const uint8x##w##_t p7, const uint8x##w##_t p6, const uint8x##w##_t p5,  \
      const uint8x##w##_t p4, const uint8x##w##_t p3, const uint8x##w##_t p2,  \
      const uint8x##w##_t p1, const uint8x##w##_t p0, const uint8x##w##_t q0,  \
      const uint8x##w##_t q1, const uint8x##w##_t q2, const uint8x##w##_t q3,  \
      const uint8x##w##_t q4, const uint8x##w##_t q5, const uint8x##w##_t q6,  \
      const uint8x##w##_t q7, uint8x##w##_t *op6, uint8x##w##_t *op5,          \
      uint8x##w##_t *op4, uint8x##w##_t *op3, uint8x##w##_t *op2,              \
      uint8x##w##_t *op1, uint8x##w##_t *op0, uint8x##w##_t *oq0,              \
      uint8x##w##_t *oq1, uint8x##w##_t *oq2, uint8x##w##_t *oq3,              \
      uint8x##w##_t *oq4, uint8x##w##_t *oq5, uint8x##w##_t *oq6) {            \
    if (flat_status != (uint32_t)-2) {                                         \
      filter4_##w(mask, hev, p1, p0, q0, q1, op1, op0, oq0, oq1);              \
    }                                                                          \
                                                                               \
    if (flat_status) {                                                         \
      *op2 = p2;                                                               \
      *oq2 = q2;                                                               \
      if (flat2_status != (uint32_t)-2) {                                      \
        apply_7_tap_filter_##w(flat, p3, p2, p1, p0, q0, q1, q2, q3, op2, op1, \
                               op0, oq0, oq1, oq2);                            \
      }                                                                        \
      if (flat2_status) {                                                      \
        apply_15_tap_filter_##w(flat2, p7, p6, p5, p4, p3, p2, p1, p0, q0, q1, \
                                q2, q3, q4, q5, q6, q7, op6, op5, op4, op3,    \
                                op2, op1, op0, oq0, oq1, oq2, oq3, oq4, oq5,   \
                                oq6);                                          \
      }                                                                        \
    }                                                                          \
  }

FUN_FILTER16(8)   // filter16_8
FUN_FILTER16(16)  // filter16_16
#undef FUN_FILTER16

#define FUN_LOAD8(w, r)                                                    \
  static INLINE void load_##w##x8(                                         \
      const uint8_t *s, const int p, uint8x##w##_t *p3, uint8x##w##_t *p2, \
      uint8x##w##_t *p1, uint8x##w##_t *p0, uint8x##w##_t *q0,             \
      uint8x##w##_t *q1, uint8x##w##_t *q2, uint8x##w##_t *q3) {           \
    *p3 = vld1##r##u8(s);                                                  \
    s += p;                                                                \
    *p2 = vld1##r##u8(s);                                                  \
    s += p;                                                                \
    *p1 = vld1##r##u8(s);                                                  \
    s += p;                                                                \
    *p0 = vld1##r##u8(s);                                                  \
    s += p;                                                                \
    *q0 = vld1##r##u8(s);                                                  \
    s += p;                                                                \
    *q1 = vld1##r##u8(s);                                                  \
    s += p;                                                                \
    *q2 = vld1##r##u8(s);                                                  \
    s += p;                                                                \
    *q3 = vld1##r##u8(s);                                                  \
  }

FUN_LOAD8(8, _)    // load_8x8
FUN_LOAD8(16, q_)  // load_16x8
#undef FUN_LOAD8

#define FUN_LOAD16(w, r)                                                   \
  static INLINE void load_##w##x16(                                        \
      const uint8_t *s, const int p, uint8x##w##_t *s0, uint8x##w##_t *s1, \
      uint8x##w##_t *s2, uint8x##w##_t *s3, uint8x##w##_t *s4,             \
      uint8x##w##_t *s5, uint8x##w##_t *s6, uint8x##w##_t *s7,             \
      uint8x##w##_t *s8, uint8x##w##_t *s9, uint8x##w##_t *s10,            \
      uint8x##w##_t *s11, uint8x##w##_t *s12, uint8x##w##_t *s13,          \
      uint8x##w##_t *s14, uint8x##w##_t *s15) {                            \
    *s0 = vld1##r##u8(s);                                                  \
    s += p;                                                                \
    *s1 = vld1##r##u8(s);                                                  \
    s += p;                                                                \
    *s2 = vld1##r##u8(s);                                                  \
    s += p;                                                                \
    *s3 = vld1##r##u8(s);                                                  \
    s += p;                                                                \
    *s4 = vld1##r##u8(s);                                                  \
    s += p;                                                                \
    *s5 = vld1##r##u8(s);                                                  \
    s += p;                                                                \
    *s6 = vld1##r##u8(s);                                                  \
    s += p;                                                                \
    *s7 = vld1##r##u8(s);                                                  \
    s += p;                                                                \
    *s8 = vld1##r##u8(s);                                                  \
    s += p;                                                                \
    *s9 = vld1##r##u8(s);                                                  \
    s += p;                                                                \
    *s10 = vld1##r##u8(s);                                                 \
    s += p;                                                                \
    *s11 = vld1##r##u8(s);                                                 \
    s += p;                                                                \
    *s12 = vld1##r##u8(s);                                                 \
    s += p;                                                                \
    *s13 = vld1##r##u8(s);                                                 \
    s += p;                                                                \
    *s14 = vld1##r##u8(s);                                                 \
    s += p;                                                                \
    *s15 = vld1##r##u8(s);                                                 \
  }

FUN_LOAD16(8, _)    // load_8x16
FUN_LOAD16(16, q_)  // load_16x16
#undef FUN_LOAD16

#define FUN_STORE4(w, r)                                                       \
  static INLINE void store_##w##x4(                                            \
      uint8_t *s, const int p, const uint8x##w##_t s0, const uint8x##w##_t s1, \
      const uint8x##w##_t s2, const uint8x##w##_t s3) {                        \
    vst1##r##u8(s, s0);                                                        \
    s += p;                                                                    \
    vst1##r##u8(s, s1);                                                        \
    s += p;                                                                    \
    vst1##r##u8(s, s2);                                                        \
    s += p;                                                                    \
    vst1##r##u8(s, s3);                                                        \
  }

FUN_STORE4(8, _)    // store_8x4
FUN_STORE4(16, q_)  // store_16x4
#undef FUN_STORE4

#define FUN_STORE6(w, r)                                                       \
  static INLINE void store_##w##x6(                                            \
      uint8_t *s, const int p, const uint8x##w##_t s0, const uint8x##w##_t s1, \
      const uint8x##w##_t s2, const uint8x##w##_t s3, const uint8x##w##_t s4,  \
      const uint8x##w##_t s5) {                                                \
    vst1##r##u8(s, s0);                                                        \
    s += p;                                                                    \
    vst1##r##u8(s, s1);                                                        \
    s += p;                                                                    \
    vst1##r##u8(s, s2);                                                        \
    s += p;                                                                    \
    vst1##r##u8(s, s3);                                                        \
    s += p;                                                                    \
    vst1##r##u8(s, s4);                                                        \
    s += p;                                                                    \
    vst1##r##u8(s, s5);                                                        \
  }

FUN_STORE6(8, _)    // store_8x6
FUN_STORE6(16, q_)  // store_16x6
#undef FUN_STORE6

static INLINE void store_4x8(uint8_t *s, const int p, const uint8x8_t p1,
                             const uint8x8_t p0, const uint8x8_t q0,
                             const uint8x8_t q1) {
  uint8x8x4_t o;

  o.val[0] = p1;
  o.val[1] = p0;
  o.val[2] = q0;
  o.val[3] = q1;
  vst4_lane_u8(s, o, 0);
  s += p;
  vst4_lane_u8(s, o, 1);
  s += p;
  vst4_lane_u8(s, o, 2);
  s += p;
  vst4_lane_u8(s, o, 3);
  s += p;
  vst4_lane_u8(s, o, 4);
  s += p;
  vst4_lane_u8(s, o, 5);
  s += p;
  vst4_lane_u8(s, o, 6);
  s += p;
  vst4_lane_u8(s, o, 7);
}

static INLINE void store_6x8(uint8_t *s, const int p, const uint8x8_t s0,
                             const uint8x8_t s1, const uint8x8_t s2,
                             const uint8x8_t s3, const uint8x8_t s4,
                             const uint8x8_t s5) {
  uint8x8x3_t o0, o1;

  o0.val[0] = s0;
  o0.val[1] = s1;
  o0.val[2] = s2;
  o1.val[0] = s3;
  o1.val[1] = s4;
  o1.val[2] = s5;
  vst3_lane_u8(s - 3, o0, 0);
  vst3_lane_u8(s + 0, o1, 0);
  s += p;
  vst3_lane_u8(s - 3, o0, 1);
  vst3_lane_u8(s + 0, o1, 1);
  s += p;
  vst3_lane_u8(s - 3, o0, 2);
  vst3_lane_u8(s + 0, o1, 2);
  s += p;
  vst3_lane_u8(s - 3, o0, 3);
  vst3_lane_u8(s + 0, o1, 3);
  s += p;
  vst3_lane_u8(s - 3, o0, 4);
  vst3_lane_u8(s + 0, o1, 4);
  s += p;
  vst3_lane_u8(s - 3, o0, 5);
  vst3_lane_u8(s + 0, o1, 5);
  s += p;
  vst3_lane_u8(s - 3, o0, 6);
  vst3_lane_u8(s + 0, o1, 6);
  s += p;
  vst3_lane_u8(s - 3, o0, 7);
  vst3_lane_u8(s + 0, o1, 7);
}

#define FUN_STORE8(w, r)                                                       \
  static INLINE void store_##w##x8(                                            \
      uint8_t *s, const int p, const uint8x##w##_t s0, const uint8x##w##_t s1, \
      const uint8x##w##_t s2, const uint8x##w##_t s3, const uint8x##w##_t s4,  \
      const uint8x##w##_t s5, const uint8x##w##_t s6,                          \
      const uint8x##w##_t s7) {                                                \
    vst1##r##u8(s, s0);                                                        \
    s += p;                                                                    \
    vst1##r##u8(s, s1);                                                        \
    s += p;                                                                    \
    vst1##r##u8(s, s2);                                                        \
    s += p;                                                                    \
    vst1##r##u8(s, s3);                                                        \
    s += p;                                                                    \
    vst1##r##u8(s, s4);                                                        \
    s += p;                                                                    \
    vst1##r##u8(s, s5);                                                        \
    s += p;                                                                    \
    vst1##r##u8(s, s6);                                                        \
    s += p;                                                                    \
    vst1##r##u8(s, s7);                                                        \
  }

FUN_STORE8(8, _)    // store_8x8
FUN_STORE8(16, q_)  // store_16x8
#undef FUN_STORE8

#define FUN_STORE14(w, r)                                                      \
  static INLINE void store_##w##x14(                                           \
      uint8_t *s, const int p, const uint8x##w##_t p6, const uint8x##w##_t p5, \
      const uint8x##w##_t p4, const uint8x##w##_t p3, const uint8x##w##_t p2,  \
      const uint8x##w##_t p1, const uint8x##w##_t p0, const uint8x##w##_t q0,  \
      const uint8x##w##_t q1, const uint8x##w##_t q2, const uint8x##w##_t q3,  \
      const uint8x##w##_t q4, const uint8x##w##_t q5, const uint8x##w##_t q6,  \
      const uint32_t flat_status, const uint32_t flat2_status) {               \
    if (flat_status) {                                                         \
      if (flat2_status) {                                                      \
        vst1##r##u8(s - 7 * p, p6);                                            \
        vst1##r##u8(s - 6 * p, p5);                                            \
        vst1##r##u8(s - 5 * p, p4);                                            \
        vst1##r##u8(s - 4 * p, p3);                                            \
        vst1##r##u8(s + 3 * p, q3);                                            \
        vst1##r##u8(s + 4 * p, q4);                                            \
        vst1##r##u8(s + 5 * p, q5);                                            \
        vst1##r##u8(s + 6 * p, q6);                                            \
      }                                                                        \
      vst1##r##u8(s - 3 * p, p2);                                              \
      vst1##r##u8(s + 2 * p, q2);                                              \
    }                                                                          \
    vst1##r##u8(s - 2 * p, p1);                                                \
    vst1##r##u8(s - 1 * p, p0);                                                \
    vst1##r##u8(s + 0 * p, q0);                                                \
    vst1##r##u8(s + 1 * p, q1);                                                \
  }

FUN_STORE14(8, _)    // store_8x14
FUN_STORE14(16, q_)  // store_16x14
#undef FUN_STORE14

static INLINE void store_16x16(uint8_t *s, const int p, const uint8x16_t s0,
                               const uint8x16_t s1, const uint8x16_t s2,
                               const uint8x16_t s3, const uint8x16_t s4,
                               const uint8x16_t s5, const uint8x16_t s6,
                               const uint8x16_t s7, const uint8x16_t s8,
                               const uint8x16_t s9, const uint8x16_t s10,
                               const uint8x16_t s11, const uint8x16_t s12,
                               const uint8x16_t s13, const uint8x16_t s14,
                               const uint8x16_t s15) {
  vst1q_u8(s, s0);
  s += p;
  vst1q_u8(s, s1);
  s += p;
  vst1q_u8(s, s2);
  s += p;
  vst1q_u8(s, s3);
  s += p;
  vst1q_u8(s, s4);
  s += p;
  vst1q_u8(s, s5);
  s += p;
  vst1q_u8(s, s6);
  s += p;
  vst1q_u8(s, s7);
  s += p;
  vst1q_u8(s, s8);
  s += p;
  vst1q_u8(s, s9);
  s += p;
  vst1q_u8(s, s10);
  s += p;
  vst1q_u8(s, s11);
  s += p;
  vst1q_u8(s, s12);
  s += p;
  vst1q_u8(s, s13);
  s += p;
  vst1q_u8(s, s14);
  s += p;
  vst1q_u8(s, s15);
}

#define FUN_HOR_4_KERNEL(name, w)                                           \
  static INLINE void lpf_horizontal_4##name##kernel(                        \
      uint8_t *s, const int p, const uint8x##w##_t blimit,                  \
      const uint8x##w##_t limit, const uint8x##w##_t thresh) {              \
    uint8x##w##_t p3, p2, p1, p0, q0, q1, q2, q3, mask, hev;                \
                                                                            \
    load_##w##x8(s - 4 * p, p, &p3, &p2, &p1, &p0, &q0, &q1, &q2, &q3);     \
    filter_hev_mask4_##w(limit, blimit, thresh, p3, p2, p1, p0, q0, q1, q2, \
                         q3, &hev, &mask);                                  \
    filter4_##w(mask, hev, p1, p0, q0, q1, &p1, &p0, &q0, &q1);             \
    store_##w##x4(s - 2 * p, p, p1, p0, q0, q1);                            \
  }

FUN_HOR_4_KERNEL(_, 8)        // lpf_horizontal_4_kernel
FUN_HOR_4_KERNEL(_dual_, 16)  // lpf_horizontal_4_dual_kernel
#undef FUN_HOR_4_KERNEL

void vpx_lpf_horizontal_4_neon(uint8_t *s, int p, const uint8_t *blimit,
                               const uint8_t *limit, const uint8_t *thresh) {
  uint8x8_t blimit_vec, limit_vec, thresh_vec;
  load_thresh_8(blimit, limit, thresh, &blimit_vec, &limit_vec, &thresh_vec);
  lpf_horizontal_4_kernel(s, p, blimit_vec, limit_vec, thresh_vec);
}

void vpx_lpf_horizontal_4_dual_neon(uint8_t *s, int p, const uint8_t *blimit0,
                                    const uint8_t *limit0,
                                    const uint8_t *thresh0,
                                    const uint8_t *blimit1,
                                    const uint8_t *limit1,
                                    const uint8_t *thresh1) {
  uint8x16_t blimit_vec, limit_vec, thresh_vec;
  load_thresh_8_dual(blimit0, limit0, thresh0, blimit1, limit1, thresh1,
                     &blimit_vec, &limit_vec, &thresh_vec);
  lpf_horizontal_4_dual_kernel(s, p, blimit_vec, limit_vec, thresh_vec);
}

void vpx_lpf_vertical_4_neon(uint8_t *s, int p, const uint8_t *blimit,
                             const uint8_t *limit, const uint8_t *thresh) {
  uint8x8_t blimit_vec, limit_vec, thresh_vec, p3, p2, p1, p0, q0, q1, q2, q3,
      mask, hev;
  load_thresh_8(blimit, limit, thresh, &blimit_vec, &limit_vec, &thresh_vec);
  load_8x8(s - 4, p, &p3, &p2, &p1, &p0, &q0, &q1, &q2, &q3);
  transpose_u8_8x8(&p3, &p2, &p1, &p0, &q0, &q1, &q2, &q3);
  filter_hev_mask4_8(limit_vec, blimit_vec, thresh_vec, p3, p2, p1, p0, q0, q1,
                     q2, q3, &hev, &mask);
  filter4_8(mask, hev, p1, p0, q0, q1, &p1, &p0, &q0, &q1);
  store_4x8(s - 2, p, p1, p0, q0, q1);
}

void vpx_lpf_vertical_4_dual_neon(uint8_t *s, int p, const uint8_t *blimit0,
                                  const uint8_t *limit0, const uint8_t *thresh0,
                                  const uint8_t *blimit1, const uint8_t *limit1,
                                  const uint8_t *thresh1) {
  uint8x16_t blimit_vec, limit_vec, thresh_vec, p3, p2, p1, p0, q0, q1, q2, q3,
      mask, hev;
  uint8x8_t s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14,
      s15;

  load_thresh_8_dual(blimit0, limit0, thresh0, blimit1, limit1, thresh1,
                     &blimit_vec, &limit_vec, &thresh_vec);
  load_8x16(s - 4, p, &s0, &s1, &s2, &s3, &s4, &s5, &s6, &s7, &s8, &s9, &s10,
            &s11, &s12, &s13, &s14, &s15);
  transpose_u8_8x16(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13,
                    s14, s15, &p3, &p2, &p1, &p0, &q0, &q1, &q2, &q3);
  filter_hev_mask4_16(limit_vec, blimit_vec, thresh_vec, p3, p2, p1, p0, q0, q1,
                      q2, q3, &hev, &mask);
  filter4_16(mask, hev, p1, p0, q0, q1, &p1, &p0, &q0, &q1);
  s -= 2;
  store_4x8(s, p, vget_low_u8(p1), vget_low_u8(p0), vget_low_u8(q0),
            vget_low_u8(q1));
  store_4x8(s + 8 * p, p, vget_high_u8(p1), vget_high_u8(p0), vget_high_u8(q0),
            vget_high_u8(q1));
}

void vpx_lpf_horizontal_8_neon(uint8_t *s, int p, const uint8_t *blimit,
                               const uint8_t *limit, const uint8_t *thresh) {
  uint8x8_t blimit_vec, limit_vec, thresh_vec, p3, p2, p1, p0, q0, q1, q2, q3,
      op2, op1, op0, oq0, oq1, oq2, mask, flat, hev;
  uint32_t flat_status;

  load_thresh_8(blimit, limit, thresh, &blimit_vec, &limit_vec, &thresh_vec);
  load_8x8(s - 4 * p, p, &p3, &p2, &p1, &p0, &q0, &q1, &q2, &q3);
  mask = filter_flat_hev_mask_8(limit_vec, blimit_vec, thresh_vec, p3, p2, p1,
                                p0, q0, q1, q2, q3, &flat, &flat_status, &hev);
  filter8_8(mask, flat, flat_status, hev, p3, p2, p1, p0, q0, q1, q2, q3, &op2,
            &op1, &op0, &oq0, &oq1, &oq2);
  store_8x6(s - 3 * p, p, op2, op1, op0, oq0, oq1, oq2);
}

void vpx_lpf_horizontal_8_dual_neon(uint8_t *s, int p, const uint8_t *blimit0,
                                    const uint8_t *limit0,
                                    const uint8_t *thresh0,
                                    const uint8_t *blimit1,
                                    const uint8_t *limit1,
                                    const uint8_t *thresh1) {
  uint8x16_t blimit_vec, limit_vec, thresh_vec, p3, p2, p1, p0, q0, q1, q2, q3,
      op2, op1, op0, oq0, oq1, oq2, mask, flat, hev;
  uint32_t flat_status;

  load_thresh_8_dual(blimit0, limit0, thresh0, blimit1, limit1, thresh1,
                     &blimit_vec, &limit_vec, &thresh_vec);
  load_16x8(s - 4 * p, p, &p3, &p2, &p1, &p0, &q0, &q1, &q2, &q3);
  mask = filter_flat_hev_mask_16(limit_vec, blimit_vec, thresh_vec, p3, p2, p1,
                                 p0, q0, q1, q2, q3, &flat, &flat_status, &hev);
  filter8_16(mask, flat, flat_status, hev, p3, p2, p1, p0, q0, q1, q2, q3, &op2,
             &op1, &op0, &oq0, &oq1, &oq2);
  store_16x6(s - 3 * p, p, op2, op1, op0, oq0, oq1, oq2);
}

void vpx_lpf_vertical_8_neon(uint8_t *s, int p, const uint8_t *blimit,
                             const uint8_t *limit, const uint8_t *thresh) {
  uint8x8_t blimit_vec, limit_vec, thresh_vec, p3, p2, p1, p0, q0, q1, q2, q3,
      op2, op1, op0, oq0, oq1, oq2, mask, flat, hev;
  uint32_t flat_status;

  load_thresh_8(blimit, limit, thresh, &blimit_vec, &limit_vec, &thresh_vec);
  load_8x8(s - 4, p, &p3, &p2, &p1, &p0, &q0, &q1, &q2, &q3);
  transpose_u8_8x8(&p3, &p2, &p1, &p0, &q0, &q1, &q2, &q3);
  mask = filter_flat_hev_mask_8(limit_vec, blimit_vec, thresh_vec, p3, p2, p1,
                                p0, q0, q1, q2, q3, &flat, &flat_status, &hev);
  filter8_8(mask, flat, flat_status, hev, p3, p2, p1, p0, q0, q1, q2, q3, &op2,
            &op1, &op0, &oq0, &oq1, &oq2);
  // Note: transpose + store_8x8() is faster than store_6x8().
  transpose_u8_8x8(&p3, &op2, &op1, &op0, &oq0, &oq1, &oq2, &q3);
  store_8x8(s - 4, p, p3, op2, op1, op0, oq0, oq1, oq2, q3);
}

void vpx_lpf_vertical_8_dual_neon(uint8_t *s, int p, const uint8_t *blimit0,
                                  const uint8_t *limit0, const uint8_t *thresh0,
                                  const uint8_t *blimit1, const uint8_t *limit1,
                                  const uint8_t *thresh1) {
  uint8x16_t blimit_vec, limit_vec, thresh_vec, p3, p2, p1, p0, q0, q1, q2, q3,
      op2, op1, op0, oq0, oq1, oq2, mask, flat, hev;
  uint8x8_t s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14,
      s15;
  uint32_t flat_status;

  load_thresh_8_dual(blimit0, limit0, thresh0, blimit1, limit1, thresh1,
                     &blimit_vec, &limit_vec, &thresh_vec);
  load_8x16(s - 4, p, &s0, &s1, &s2, &s3, &s4, &s5, &s6, &s7, &s8, &s9, &s10,
            &s11, &s12, &s13, &s14, &s15);
  transpose_u8_8x16(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13,
                    s14, s15, &p3, &p2, &p1, &p0, &q0, &q1, &q2, &q3);
  mask = filter_flat_hev_mask_16(limit_vec, blimit_vec, thresh_vec, p3, p2, p1,
                                 p0, q0, q1, q2, q3, &flat, &flat_status, &hev);
  filter8_16(mask, flat, flat_status, hev, p3, p2, p1, p0, q0, q1, q2, q3, &op2,
             &op1, &op0, &oq0, &oq1, &oq2);
  // Note: store_6x8() twice is faster than transpose + store_8x16().
  store_6x8(s, p, vget_low_u8(op2), vget_low_u8(op1), vget_low_u8(op0),
            vget_low_u8(oq0), vget_low_u8(oq1), vget_low_u8(oq2));
  store_6x8(s + 8 * p, p, vget_high_u8(op2), vget_high_u8(op1),
            vget_high_u8(op0), vget_high_u8(oq0), vget_high_u8(oq1),
            vget_high_u8(oq2));
}

#define FUN_LPF_16_KERNEL(name, w)                                             \
  static INLINE void lpf_16##name##kernel(                                     \
      const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh,      \
      const uint8x##w##_t p7, const uint8x##w##_t p6, const uint8x##w##_t p5,  \
      const uint8x##w##_t p4, const uint8x##w##_t p3, const uint8x##w##_t p2,  \
      const uint8x##w##_t p1, const uint8x##w##_t p0, const uint8x##w##_t q0,  \
      const uint8x##w##_t q1, const uint8x##w##_t q2, const uint8x##w##_t q3,  \
      const uint8x##w##_t q4, const uint8x##w##_t q5, const uint8x##w##_t q6,  \
      const uint8x##w##_t q7, uint8x##w##_t *op6, uint8x##w##_t *op5,          \
      uint8x##w##_t *op4, uint8x##w##_t *op3, uint8x##w##_t *op2,              \
      uint8x##w##_t *op1, uint8x##w##_t *op0, uint8x##w##_t *oq0,              \
      uint8x##w##_t *oq1, uint8x##w##_t *oq2, uint8x##w##_t *oq3,              \
      uint8x##w##_t *oq4, uint8x##w##_t *oq5, uint8x##w##_t *oq6,              \
      uint32_t *flat_status, uint32_t *flat2_status) {                         \
    uint8x##w##_t blimit_vec, limit_vec, thresh_vec, mask, flat, flat2, hev;   \
                                                                               \
    load_thresh_##w(blimit, limit, thresh, &blimit_vec, &limit_vec,            \
                    &thresh_vec);                                              \
    mask = filter_flat_hev_mask_##w(limit_vec, blimit_vec, thresh_vec, p3, p2, \
                                    p1, p0, q0, q1, q2, q3, &flat,             \
                                    flat_status, &hev);                        \
    flat2 = flat_mask5_##w(p7, p6, p5, p4, p0, q0, q4, q5, q6, q7, flat,       \
                           flat2_status);                                      \
    filter16_##w(mask, flat, *flat_status, flat2, *flat2_status, hev, p7, p6,  \
                 p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6, q7, op6,  \
                 op5, op4, op3, op2, op1, op0, oq0, oq1, oq2, oq3, oq4, oq5,   \
                 oq6);                                                         \
  }

FUN_LPF_16_KERNEL(_, 8)        // lpf_16_kernel
FUN_LPF_16_KERNEL(_dual_, 16)  // lpf_16_dual_kernel
#undef FUN_LPF_16_KERNEL

// Quiet warnings of the form: 'vpx_dsp/arm/loopfilter_neon.c|981 col 42|
// warning: 'oq1' may be used uninitialized in this function
// [-Wmaybe-uninitialized]', for oq1-op1. Without reworking the code or adding
// an additional branch this warning cannot be silenced otherwise. The
// loopfilter is only called when needed for a block so these output pixels
// will be set.
#if defined(__GNUC__) && __GNUC__ >= 4 && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

void vpx_lpf_horizontal_16_neon(uint8_t *s, int p, const uint8_t *blimit,
                                const uint8_t *limit, const uint8_t *thresh) {
  uint8x8_t p7, p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6, q7, op6,
      op5, op4, op3, op2, op1, op0, oq0, oq1, oq2, oq3, oq4, oq5, oq6;
  uint32_t flat_status, flat2_status;

  load_8x16(s - 8 * p, p, &p7, &p6, &p5, &p4, &p3, &p2, &p1, &p0, &q0, &q1, &q2,
            &q3, &q4, &q5, &q6, &q7);
  lpf_16_kernel(blimit, limit, thresh, p7, p6, p5, p4, p3, p2, p1, p0, q0, q1,
                q2, q3, q4, q5, q6, q7, &op6, &op5, &op4, &op3, &op2, &op1,
                &op0, &oq0, &oq1, &oq2, &oq3, &oq4, &oq5, &oq6, &flat_status,
                &flat2_status);
  store_8x14(s, p, op6, op5, op4, op3, op2, op1, op0, oq0, oq1, oq2, oq3, oq4,
             oq5, oq6, flat_status, flat2_status);
}

void vpx_lpf_horizontal_16_dual_neon(uint8_t *s, int p, const uint8_t *blimit,
                                     const uint8_t *limit,
                                     const uint8_t *thresh) {
  uint8x16_t p7, p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6, q7,
      op6, op5, op4, op3, op2, op1, op0, oq0, oq1, oq2, oq3, oq4, oq5, oq6;
  uint32_t flat_status, flat2_status;

  load_16x8(s - 4 * p, p, &p3, &p2, &p1, &p0, &q0, &q1, &q2, &q3);
  p7 = vld1q_u8(s - 8 * p);
  p6 = vld1q_u8(s - 7 * p);
  p5 = vld1q_u8(s - 6 * p);
  p4 = vld1q_u8(s - 5 * p);
  q4 = vld1q_u8(s + 4 * p);
  q5 = vld1q_u8(s + 5 * p);
  q6 = vld1q_u8(s + 6 * p);
  q7 = vld1q_u8(s + 7 * p);
  lpf_16_dual_kernel(blimit, limit, thresh, p7, p6, p5, p4, p3, p2, p1, p0, q0,
                     q1, q2, q3, q4, q5, q6, q7, &op6, &op5, &op4, &op3, &op2,
                     &op1, &op0, &oq0, &oq1, &oq2, &oq3, &oq4, &oq5, &oq6,
                     &flat_status, &flat2_status);
  store_16x14(s, p, op6, op5, op4, op3, op2, op1, op0, oq0, oq1, oq2, oq3, oq4,
              oq5, oq6, flat_status, flat2_status);
}

void vpx_lpf_vertical_16_neon(uint8_t *s, int p, const uint8_t *blimit,
                              const uint8_t *limit, const uint8_t *thresh) {
  uint8x8_t p7, p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6, q7, op6,
      op5, op4, op3, op2, op1, op0, oq0, oq1, oq2, oq3, oq4, oq5, oq6;
  uint8x16_t s0, s1, s2, s3, s4, s5, s6, s7;
  uint32_t flat_status, flat2_status;

  s -= 8;
  load_16x8(s, p, &s0, &s1, &s2, &s3, &s4, &s5, &s6, &s7);
  transpose_u8_16x8(s0, s1, s2, s3, s4, s5, s6, s7, &p7, &p6, &p5, &p4, &p3,
                    &p2, &p1, &p0, &q0, &q1, &q2, &q3, &q4, &q5, &q6, &q7);
  lpf_16_kernel(blimit, limit, thresh, p7, p6, p5, p4, p3, p2, p1, p0, q0, q1,
                q2, q3, q4, q5, q6, q7, &op6, &op5, &op4, &op3, &op2, &op1,
                &op0, &oq0, &oq1, &oq2, &oq3, &oq4, &oq5, &oq6, &flat_status,
                &flat2_status);
  if (flat_status) {
    if (flat2_status) {
      transpose_u8_8x16(p7, op6, op5, op4, op3, op2, op1, op0, oq0, oq1, oq2,
                        oq3, oq4, oq5, oq6, q7, &s0, &s1, &s2, &s3, &s4, &s5,
                        &s6, &s7);
      store_16x8(s, p, s0, s1, s2, s3, s4, s5, s6, s7);
    } else {
      // Note: transpose + store_8x8() is faster than store_6x8().
      transpose_u8_8x8(&p3, &op2, &op1, &op0, &oq0, &oq1, &oq2, &q3);
      store_8x8(s + 4, p, p3, op2, op1, op0, oq0, oq1, oq2, q3);
    }
  } else {
    store_4x8(s + 6, p, op1, op0, oq0, oq1);
  }
}

void vpx_lpf_vertical_16_dual_neon(uint8_t *s, int p, const uint8_t *blimit,
                                   const uint8_t *limit,
                                   const uint8_t *thresh) {
  uint8x16_t p7, p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6, q7,
      op6, op5, op4, op3, op2, op1, op0, oq0, oq1, oq2, oq3, oq4, oq5, oq6;
  uint8x16_t s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14,
      s15;
  uint32_t flat_status, flat2_status;

  s -= 8;
  load_16x16(s, p, &s0, &s1, &s2, &s3, &s4, &s5, &s6, &s7, &s8, &s9, &s10, &s11,
             &s12, &s13, &s14, &s15);
  transpose_u8_16x16(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13,
                     s14, s15, &p7, &p6, &p5, &p4, &p3, &p2, &p1, &p0, &q0, &q1,
                     &q2, &q3, &q4, &q5, &q6, &q7);
  lpf_16_dual_kernel(blimit, limit, thresh, p7, p6, p5, p4, p3, p2, p1, p0, q0,
                     q1, q2, q3, q4, q5, q6, q7, &op6, &op5, &op4, &op3, &op2,
                     &op1, &op0, &oq0, &oq1, &oq2, &oq3, &oq4, &oq5, &oq6,
                     &flat_status, &flat2_status);
  if (flat_status) {
    if (flat2_status) {
      transpose_u8_16x16(p7, op6, op5, op4, op3, op2, op1, op0, oq0, oq1, oq2,
                         oq3, oq4, oq5, oq6, q7, &s0, &s1, &s2, &s3, &s4, &s5,
                         &s6, &s7, &s8, &s9, &s10, &s11, &s12, &s13, &s14,
                         &s15);
      store_16x16(s, p, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12,
                  s13, s14, s15);
    } else {
      // Note: store_6x8() twice is faster than transpose + store_8x16().
      s += 8;
      store_6x8(s, p, vget_low_u8(op2), vget_low_u8(op1), vget_low_u8(op0),
                vget_low_u8(oq0), vget_low_u8(oq1), vget_low_u8(oq2));
      store_6x8(s + 8 * p, p, vget_high_u8(op2), vget_high_u8(op1),
                vget_high_u8(op0), vget_high_u8(oq0), vget_high_u8(oq1),
                vget_high_u8(oq2));
    }
  } else {
    s += 6;
    store_4x8(s, p, vget_low_u8(op1), vget_low_u8(op0), vget_low_u8(oq0),
              vget_low_u8(oq1));
    store_4x8(s + 8 * p, p, vget_high_u8(op1), vget_high_u8(op0),
              vget_high_u8(oq0), vget_high_u8(oq1));
  }
}

#if defined(__GNUC__) && __GNUC__ >= 4 && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
