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

static INLINE void load_thresh(const uint8_t *blimit, const uint8_t *limit,
                               const uint8_t *thresh, uint16x8_t *blimit_vec,
                               uint16x8_t *limit_vec, uint16x8_t *thresh_vec,
                               const int bd) {
  const int16x8_t shift = vdupq_n_s16(bd - 8);
  *blimit_vec = vmovl_u8(vld1_dup_u8(blimit));
  *limit_vec = vmovl_u8(vld1_dup_u8(limit));
  *thresh_vec = vmovl_u8(vld1_dup_u8(thresh));
  *blimit_vec = vshlq_u16(*blimit_vec, shift);
  *limit_vec = vshlq_u16(*limit_vec, shift);
  *thresh_vec = vshlq_u16(*thresh_vec, shift);
}

// Here flat is 128-bit long, with each 16-bit chunk being a mask of
// a pixel. When used to control filter branches, we only detect whether it is
// all 0s or all 1s. We pairwise add flat to a 32-bit long number flat_status.
// flat equals 0 if and only if flat_status equals 0.
// flat equals -1 (all 1s) if and only if flat_status equals -4. (This is true
// because each mask occupies more than 1 bit.)
static INLINE uint32_t calc_flat_status(const uint16x8_t flat) {
  const uint64x1_t t0 = vadd_u64(vreinterpret_u64_u16(vget_low_u16(flat)),
                                 vreinterpret_u64_u16(vget_high_u16(flat)));
  const uint64x1_t t1 = vpaddl_u32(vreinterpret_u32_u64(t0));
  return vget_lane_u32(vreinterpret_u32_u64(t1), 0);
}

static INLINE uint16x8_t
filter_hev_mask4(const uint16x8_t limit, const uint16x8_t blimit,
                 const uint16x8_t thresh, const uint16x8_t p3,
                 const uint16x8_t p2, const uint16x8_t p1, const uint16x8_t p0,
                 const uint16x8_t q0, const uint16x8_t q1, const uint16x8_t q2,
                 const uint16x8_t q3, uint16x8_t *hev, uint16x8_t *mask) {
  uint16x8_t max, t0, t1;

  max = vabdq_u16(p1, p0);
  max = vmaxq_u16(max, vabdq_u16(q1, q0));
  *hev = vcgtq_u16(max, thresh);
  *mask = vmaxq_u16(max, vabdq_u16(p3, p2));
  *mask = vmaxq_u16(*mask, vabdq_u16(p2, p1));
  *mask = vmaxq_u16(*mask, vabdq_u16(q2, q1));
  *mask = vmaxq_u16(*mask, vabdq_u16(q3, q2));
  t0 = vabdq_u16(p0, q0);
  t1 = vabdq_u16(p1, q1);
  t0 = vaddq_u16(t0, t0);
  t1 = vshrq_n_u16(t1, 1);
  t0 = vaddq_u16(t0, t1);
  *mask = vcleq_u16(*mask, limit);
  t0 = vcleq_u16(t0, blimit);
  *mask = vandq_u16(*mask, t0);

  return max;
}

static INLINE uint16x8_t filter_flat_hev_mask(
    const uint16x8_t limit, const uint16x8_t blimit, const uint16x8_t thresh,
    const uint16x8_t p3, const uint16x8_t p2, const uint16x8_t p1,
    const uint16x8_t p0, const uint16x8_t q0, const uint16x8_t q1,
    const uint16x8_t q2, const uint16x8_t q3, uint16x8_t *flat,
    uint32_t *flat_status, uint16x8_t *hev, const int bd) {
  uint16x8_t mask;
  const uint16x8_t max = filter_hev_mask4(limit, blimit, thresh, p3, p2, p1, p0,
                                          q0, q1, q2, q3, hev, &mask);
  *flat = vmaxq_u16(max, vabdq_u16(p2, p0));
  *flat = vmaxq_u16(*flat, vabdq_u16(q2, q0));
  *flat = vmaxq_u16(*flat, vabdq_u16(p3, p0));
  *flat = vmaxq_u16(*flat, vabdq_u16(q3, q0));
  *flat = vcleq_u16(*flat, vdupq_n_u16(1 << (bd - 8))); /* flat_mask4() */
  *flat = vandq_u16(*flat, mask);
  *flat_status = calc_flat_status(*flat);

  return mask;
}

static INLINE uint16x8_t flat_mask5(const uint16x8_t p4, const uint16x8_t p3,
                                    const uint16x8_t p2, const uint16x8_t p1,
                                    const uint16x8_t p0, const uint16x8_t q0,
                                    const uint16x8_t q1, const uint16x8_t q2,
                                    const uint16x8_t q3, const uint16x8_t q4,
                                    const uint16x8_t flat,
                                    uint32_t *flat2_status, const int bd) {
  uint16x8_t flat2 = vabdq_u16(p4, p0);
  flat2 = vmaxq_u16(flat2, vabdq_u16(p3, p0));
  flat2 = vmaxq_u16(flat2, vabdq_u16(p2, p0));
  flat2 = vmaxq_u16(flat2, vabdq_u16(p1, p0));
  flat2 = vmaxq_u16(flat2, vabdq_u16(q1, q0));
  flat2 = vmaxq_u16(flat2, vabdq_u16(q2, q0));
  flat2 = vmaxq_u16(flat2, vabdq_u16(q3, q0));
  flat2 = vmaxq_u16(flat2, vabdq_u16(q4, q0));
  flat2 = vcleq_u16(flat2, vdupq_n_u16(1 << (bd - 8)));
  flat2 = vandq_u16(flat2, flat);
  *flat2_status = calc_flat_status(flat2);

  return flat2;
}

static INLINE int16x8_t flip_sign(const uint16x8_t v, const int bd) {
  const uint16x8_t offset = vdupq_n_u16(0x80 << (bd - 8));
  return vreinterpretq_s16_u16(vsubq_u16(v, offset));
}

static INLINE uint16x8_t flip_sign_back(const int16x8_t v, const int bd) {
  const int16x8_t offset = vdupq_n_s16(0x80 << (bd - 8));
  return vreinterpretq_u16_s16(vaddq_s16(v, offset));
}

static INLINE void filter_update(const uint16x8_t sub0, const uint16x8_t sub1,
                                 const uint16x8_t add0, const uint16x8_t add1,
                                 uint16x8_t *sum) {
  *sum = vsubq_u16(*sum, sub0);
  *sum = vsubq_u16(*sum, sub1);
  *sum = vaddq_u16(*sum, add0);
  *sum = vaddq_u16(*sum, add1);
}

static INLINE uint16x8_t calc_7_tap_filter_kernel(const uint16x8_t sub0,
                                                  const uint16x8_t sub1,
                                                  const uint16x8_t add0,
                                                  const uint16x8_t add1,
                                                  uint16x8_t *sum) {
  filter_update(sub0, sub1, add0, add1, sum);
  return vrshrq_n_u16(*sum, 3);
}

static INLINE uint16x8_t apply_15_tap_filter_kernel(
    const uint16x8_t flat, const uint16x8_t sub0, const uint16x8_t sub1,
    const uint16x8_t add0, const uint16x8_t add1, const uint16x8_t in,
    uint16x8_t *sum) {
  filter_update(sub0, sub1, add0, add1, sum);
  return vbslq_u16(flat, vrshrq_n_u16(*sum, 4), in);
}

// 7-tap filter [1, 1, 1, 2, 1, 1, 1]
static INLINE void calc_7_tap_filter(const uint16x8_t p3, const uint16x8_t p2,
                                     const uint16x8_t p1, const uint16x8_t p0,
                                     const uint16x8_t q0, const uint16x8_t q1,
                                     const uint16x8_t q2, const uint16x8_t q3,
                                     uint16x8_t *op2, uint16x8_t *op1,
                                     uint16x8_t *op0, uint16x8_t *oq0,
                                     uint16x8_t *oq1, uint16x8_t *oq2) {
  uint16x8_t sum;
  sum = vaddq_u16(p3, p3);   // 2*p3
  sum = vaddq_u16(sum, p3);  // 3*p3
  sum = vaddq_u16(sum, p2);  // 3*p3+p2
  sum = vaddq_u16(sum, p2);  // 3*p3+2*p2
  sum = vaddq_u16(sum, p1);  // 3*p3+2*p2+p1
  sum = vaddq_u16(sum, p0);  // 3*p3+2*p2+p1+p0
  sum = vaddq_u16(sum, q0);  // 3*p3+2*p2+p1+p0+q0
  *op2 = vrshrq_n_u16(sum, 3);
  *op1 = calc_7_tap_filter_kernel(p3, p2, p1, q1, &sum);
  *op0 = calc_7_tap_filter_kernel(p3, p1, p0, q2, &sum);
  *oq0 = calc_7_tap_filter_kernel(p3, p0, q0, q3, &sum);
  *oq1 = calc_7_tap_filter_kernel(p2, q0, q1, q3, &sum);
  *oq2 = calc_7_tap_filter_kernel(p1, q1, q2, q3, &sum);
}

static INLINE void apply_7_tap_filter(const uint16x8_t flat,
                                      const uint16x8_t p3, const uint16x8_t p2,
                                      const uint16x8_t p1, const uint16x8_t p0,
                                      const uint16x8_t q0, const uint16x8_t q1,
                                      const uint16x8_t q2, const uint16x8_t q3,
                                      uint16x8_t *op2, uint16x8_t *op1,
                                      uint16x8_t *op0, uint16x8_t *oq0,
                                      uint16x8_t *oq1, uint16x8_t *oq2) {
  uint16x8_t tp1, tp0, tq0, tq1;
  calc_7_tap_filter(p3, p2, p1, p0, q0, q1, q2, q3, op2, &tp1, &tp0, &tq0, &tq1,
                    oq2);
  *op2 = vbslq_u16(flat, *op2, p2);
  *op1 = vbslq_u16(flat, tp1, *op1);
  *op0 = vbslq_u16(flat, tp0, *op0);
  *oq0 = vbslq_u16(flat, tq0, *oq0);
  *oq1 = vbslq_u16(flat, tq1, *oq1);
  *oq2 = vbslq_u16(flat, *oq2, q2);
}

// 15-tap filter [1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1]
static INLINE void apply_15_tap_filter(
    const uint16x8_t flat2, const uint16x8_t p7, const uint16x8_t p6,
    const uint16x8_t p5, const uint16x8_t p4, const uint16x8_t p3,
    const uint16x8_t p2, const uint16x8_t p1, const uint16x8_t p0,
    const uint16x8_t q0, const uint16x8_t q1, const uint16x8_t q2,
    const uint16x8_t q3, const uint16x8_t q4, const uint16x8_t q5,
    const uint16x8_t q6, const uint16x8_t q7, uint16x8_t *op6, uint16x8_t *op5,
    uint16x8_t *op4, uint16x8_t *op3, uint16x8_t *op2, uint16x8_t *op1,
    uint16x8_t *op0, uint16x8_t *oq0, uint16x8_t *oq1, uint16x8_t *oq2,
    uint16x8_t *oq3, uint16x8_t *oq4, uint16x8_t *oq5, uint16x8_t *oq6) {
  uint16x8_t sum;
  sum = vshlq_n_u16(p7, 3);  // 8*p7
  sum = vsubq_u16(sum, p7);  // 7*p7
  sum = vaddq_u16(sum, p6);  // 7*p7+p6
  sum = vaddq_u16(sum, p6);  // 7*p7+2*p6
  sum = vaddq_u16(sum, p5);  // 7*p7+2*p6+p5
  sum = vaddq_u16(sum, p4);  // 7*p7+2*p6+p5+p4
  sum = vaddq_u16(sum, p3);  // 7*p7+2*p6+p5+p4+p3
  sum = vaddq_u16(sum, p2);  // 7*p7+2*p6+p5+p4+p3+p2
  sum = vaddq_u16(sum, p1);  // 7*p7+2*p6+p5+p4+p3+p2+p1
  sum = vaddq_u16(sum, p0);  // 7*p7+2*p6+p5+p4+p3+p2+p1+p0
  sum = vaddq_u16(sum, q0);  // 7*p7+2*p6+p5+p4+p3+p2+p1+p0+q0
  *op6 = vbslq_u16(flat2, vrshrq_n_u16(sum, 4), p6);
  *op5 = apply_15_tap_filter_kernel(flat2, p7, p6, p5, q1, p5, &sum);
  *op4 = apply_15_tap_filter_kernel(flat2, p7, p5, p4, q2, p4, &sum);
  *op3 = apply_15_tap_filter_kernel(flat2, p7, p4, p3, q3, p3, &sum);
  *op2 = apply_15_tap_filter_kernel(flat2, p7, p3, p2, q4, *op2, &sum);
  *op1 = apply_15_tap_filter_kernel(flat2, p7, p2, p1, q5, *op1, &sum);
  *op0 = apply_15_tap_filter_kernel(flat2, p7, p1, p0, q6, *op0, &sum);
  *oq0 = apply_15_tap_filter_kernel(flat2, p7, p0, q0, q7, *oq0, &sum);
  *oq1 = apply_15_tap_filter_kernel(flat2, p6, q0, q1, q7, *oq1, &sum);
  *oq2 = apply_15_tap_filter_kernel(flat2, p5, q1, q2, q7, *oq2, &sum);
  *oq3 = apply_15_tap_filter_kernel(flat2, p4, q2, q3, q7, q3, &sum);
  *oq4 = apply_15_tap_filter_kernel(flat2, p3, q3, q4, q7, q4, &sum);
  *oq5 = apply_15_tap_filter_kernel(flat2, p2, q4, q5, q7, q5, &sum);
  *oq6 = apply_15_tap_filter_kernel(flat2, p1, q5, q6, q7, q6, &sum);
}

static INLINE void filter4(const uint16x8_t mask, const uint16x8_t hev,
                           const uint16x8_t p1, const uint16x8_t p0,
                           const uint16x8_t q0, const uint16x8_t q1,
                           uint16x8_t *op1, uint16x8_t *op0, uint16x8_t *oq0,
                           uint16x8_t *oq1, const int bd) {
  const int16x8_t max = vdupq_n_s16((1 << (bd - 1)) - 1);
  const int16x8_t min = vdupq_n_s16((int16_t)(((uint32_t)-1) << (bd - 1)));
  int16x8_t filter, filter1, filter2, t;
  int16x8_t ps1 = flip_sign(p1, bd);
  int16x8_t ps0 = flip_sign(p0, bd);
  int16x8_t qs0 = flip_sign(q0, bd);
  int16x8_t qs1 = flip_sign(q1, bd);

  /* add outer taps if we have high edge variance */
  filter = vsubq_s16(ps1, qs1);
  filter = vmaxq_s16(filter, min);
  filter = vminq_s16(filter, max);
  filter = vandq_s16(filter, vreinterpretq_s16_u16(hev));
  t = vsubq_s16(qs0, ps0);

  /* inner taps */
  filter = vaddq_s16(filter, t);
  filter = vaddq_s16(filter, t);
  filter = vaddq_s16(filter, t);
  filter = vmaxq_s16(filter, min);
  filter = vminq_s16(filter, max);
  filter = vandq_s16(filter, vreinterpretq_s16_u16(mask));

  /* save bottom 3 bits so that we round one side +4 and the other +3 */
  /* if it equals 4 we'll set it to adjust by -1 to account for the fact */
  /* we'd round it by 3 the other way */
  t = vaddq_s16(filter, vdupq_n_s16(4));
  t = vminq_s16(t, max);
  filter1 = vshrq_n_s16(t, 3);
  t = vaddq_s16(filter, vdupq_n_s16(3));
  t = vminq_s16(t, max);
  filter2 = vshrq_n_s16(t, 3);

  qs0 = vsubq_s16(qs0, filter1);
  qs0 = vmaxq_s16(qs0, min);
  qs0 = vminq_s16(qs0, max);
  ps0 = vaddq_s16(ps0, filter2);
  ps0 = vmaxq_s16(ps0, min);
  ps0 = vminq_s16(ps0, max);
  *oq0 = flip_sign_back(qs0, bd);
  *op0 = flip_sign_back(ps0, bd);

  /* outer tap adjustments */
  filter = vrshrq_n_s16(filter1, 1);
  filter = vbicq_s16(filter, vreinterpretq_s16_u16(hev));

  qs1 = vsubq_s16(qs1, filter);
  qs1 = vmaxq_s16(qs1, min);
  qs1 = vminq_s16(qs1, max);
  ps1 = vaddq_s16(ps1, filter);
  ps1 = vmaxq_s16(ps1, min);
  ps1 = vminq_s16(ps1, max);
  *oq1 = flip_sign_back(qs1, bd);
  *op1 = flip_sign_back(ps1, bd);
}

static INLINE void filter8(const uint16x8_t mask, const uint16x8_t flat,
                           const uint32_t flat_status, const uint16x8_t hev,
                           const uint16x8_t p3, const uint16x8_t p2,
                           const uint16x8_t p1, const uint16x8_t p0,
                           const uint16x8_t q0, const uint16x8_t q1,
                           const uint16x8_t q2, const uint16x8_t q3,
                           uint16x8_t *op2, uint16x8_t *op1, uint16x8_t *op0,
                           uint16x8_t *oq0, uint16x8_t *oq1, uint16x8_t *oq2,
                           const int bd) {
  if (flat_status != (uint32_t)-4) {
    filter4(mask, hev, p1, p0, q0, q1, op1, op0, oq0, oq1, bd);
    *op2 = p2;
    *oq2 = q2;
    if (flat_status) {
      apply_7_tap_filter(flat, p3, p2, p1, p0, q0, q1, q2, q3, op2, op1, op0,
                         oq0, oq1, oq2);
    }
  } else {
    calc_7_tap_filter(p3, p2, p1, p0, q0, q1, q2, q3, op2, op1, op0, oq0, oq1,
                      oq2);
  }
}

static INLINE void filter16(
    const uint16x8_t mask, const uint16x8_t flat, const uint32_t flat_status,
    const uint16x8_t flat2, const uint32_t flat2_status, const uint16x8_t hev,
    const uint16x8_t p7, const uint16x8_t p6, const uint16x8_t p5,
    const uint16x8_t p4, const uint16x8_t p3, const uint16x8_t p2,
    const uint16x8_t p1, const uint16x8_t p0, const uint16x8_t q0,
    const uint16x8_t q1, const uint16x8_t q2, const uint16x8_t q3,
    const uint16x8_t q4, const uint16x8_t q5, const uint16x8_t q6,
    const uint16x8_t q7, uint16x8_t *op6, uint16x8_t *op5, uint16x8_t *op4,
    uint16x8_t *op3, uint16x8_t *op2, uint16x8_t *op1, uint16x8_t *op0,
    uint16x8_t *oq0, uint16x8_t *oq1, uint16x8_t *oq2, uint16x8_t *oq3,
    uint16x8_t *oq4, uint16x8_t *oq5, uint16x8_t *oq6, const int bd) {
  if (flat_status != (uint32_t)-4) {
    filter4(mask, hev, p1, p0, q0, q1, op1, op0, oq0, oq1, bd);
  }

  if (flat_status) {
    *op2 = p2;
    *oq2 = q2;
    if (flat2_status != (uint32_t)-4) {
      apply_7_tap_filter(flat, p3, p2, p1, p0, q0, q1, q2, q3, op2, op1, op0,
                         oq0, oq1, oq2);
    }
    if (flat2_status) {
      apply_15_tap_filter(flat2, p7, p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3,
                          q4, q5, q6, q7, op6, op5, op4, op3, op2, op1, op0,
                          oq0, oq1, oq2, oq3, oq4, oq5, oq6);
    }
  }
}

static INLINE void load_8x8(const uint16_t *s, const int p, uint16x8_t *p3,
                            uint16x8_t *p2, uint16x8_t *p1, uint16x8_t *p0,
                            uint16x8_t *q0, uint16x8_t *q1, uint16x8_t *q2,
                            uint16x8_t *q3) {
  *p3 = vld1q_u16(s);
  s += p;
  *p2 = vld1q_u16(s);
  s += p;
  *p1 = vld1q_u16(s);
  s += p;
  *p0 = vld1q_u16(s);
  s += p;
  *q0 = vld1q_u16(s);
  s += p;
  *q1 = vld1q_u16(s);
  s += p;
  *q2 = vld1q_u16(s);
  s += p;
  *q3 = vld1q_u16(s);
}

static INLINE void load_8x16(const uint16_t *s, const int p, uint16x8_t *s0,
                             uint16x8_t *s1, uint16x8_t *s2, uint16x8_t *s3,
                             uint16x8_t *s4, uint16x8_t *s5, uint16x8_t *s6,
                             uint16x8_t *s7, uint16x8_t *s8, uint16x8_t *s9,
                             uint16x8_t *s10, uint16x8_t *s11, uint16x8_t *s12,
                             uint16x8_t *s13, uint16x8_t *s14,
                             uint16x8_t *s15) {
  *s0 = vld1q_u16(s);
  s += p;
  *s1 = vld1q_u16(s);
  s += p;
  *s2 = vld1q_u16(s);
  s += p;
  *s3 = vld1q_u16(s);
  s += p;
  *s4 = vld1q_u16(s);
  s += p;
  *s5 = vld1q_u16(s);
  s += p;
  *s6 = vld1q_u16(s);
  s += p;
  *s7 = vld1q_u16(s);
  s += p;
  *s8 = vld1q_u16(s);
  s += p;
  *s9 = vld1q_u16(s);
  s += p;
  *s10 = vld1q_u16(s);
  s += p;
  *s11 = vld1q_u16(s);
  s += p;
  *s12 = vld1q_u16(s);
  s += p;
  *s13 = vld1q_u16(s);
  s += p;
  *s14 = vld1q_u16(s);
  s += p;
  *s15 = vld1q_u16(s);
}

static INLINE void store_8x4(uint16_t *s, const int p, const uint16x8_t s0,
                             const uint16x8_t s1, const uint16x8_t s2,
                             const uint16x8_t s3) {
  vst1q_u16(s, s0);
  s += p;
  vst1q_u16(s, s1);
  s += p;
  vst1q_u16(s, s2);
  s += p;
  vst1q_u16(s, s3);
}

static INLINE void store_8x6(uint16_t *s, const int p, const uint16x8_t s0,
                             const uint16x8_t s1, const uint16x8_t s2,
                             const uint16x8_t s3, const uint16x8_t s4,
                             const uint16x8_t s5) {
  vst1q_u16(s, s0);
  s += p;
  vst1q_u16(s, s1);
  s += p;
  vst1q_u16(s, s2);
  s += p;
  vst1q_u16(s, s3);
  s += p;
  vst1q_u16(s, s4);
  s += p;
  vst1q_u16(s, s5);
}

static INLINE void store_4x8(uint16_t *s, const int p, const uint16x8_t p1,
                             const uint16x8_t p0, const uint16x8_t q0,
                             const uint16x8_t q1) {
  uint16x8x4_t o;

  o.val[0] = p1;
  o.val[1] = p0;
  o.val[2] = q0;
  o.val[3] = q1;
  vst4q_lane_u16(s, o, 0);
  s += p;
  vst4q_lane_u16(s, o, 1);
  s += p;
  vst4q_lane_u16(s, o, 2);
  s += p;
  vst4q_lane_u16(s, o, 3);
  s += p;
  vst4q_lane_u16(s, o, 4);
  s += p;
  vst4q_lane_u16(s, o, 5);
  s += p;
  vst4q_lane_u16(s, o, 6);
  s += p;
  vst4q_lane_u16(s, o, 7);
}

static INLINE void store_6x8(uint16_t *s, const int p, const uint16x8_t s0,
                             const uint16x8_t s1, const uint16x8_t s2,
                             const uint16x8_t s3, const uint16x8_t s4,
                             const uint16x8_t s5) {
  uint16x8x3_t o0, o1;

  o0.val[0] = s0;
  o0.val[1] = s1;
  o0.val[2] = s2;
  o1.val[0] = s3;
  o1.val[1] = s4;
  o1.val[2] = s5;
  vst3q_lane_u16(s - 3, o0, 0);
  vst3q_lane_u16(s + 0, o1, 0);
  s += p;
  vst3q_lane_u16(s - 3, o0, 1);
  vst3q_lane_u16(s + 0, o1, 1);
  s += p;
  vst3q_lane_u16(s - 3, o0, 2);
  vst3q_lane_u16(s + 0, o1, 2);
  s += p;
  vst3q_lane_u16(s - 3, o0, 3);
  vst3q_lane_u16(s + 0, o1, 3);
  s += p;
  vst3q_lane_u16(s - 3, o0, 4);
  vst3q_lane_u16(s + 0, o1, 4);
  s += p;
  vst3q_lane_u16(s - 3, o0, 5);
  vst3q_lane_u16(s + 0, o1, 5);
  s += p;
  vst3q_lane_u16(s - 3, o0, 6);
  vst3q_lane_u16(s + 0, o1, 6);
  s += p;
  vst3q_lane_u16(s - 3, o0, 7);
  vst3q_lane_u16(s + 0, o1, 7);
}

static INLINE void store_7x8(uint16_t *s, const int p, const uint16x8_t s0,
                             const uint16x8_t s1, const uint16x8_t s2,
                             const uint16x8_t s3, const uint16x8_t s4,
                             const uint16x8_t s5, const uint16x8_t s6) {
  uint16x8x4_t o0;
  uint16x8x3_t o1;

  o0.val[0] = s0;
  o0.val[1] = s1;
  o0.val[2] = s2;
  o0.val[3] = s3;
  o1.val[0] = s4;
  o1.val[1] = s5;
  o1.val[2] = s6;
  vst4q_lane_u16(s - 4, o0, 0);
  vst3q_lane_u16(s + 0, o1, 0);
  s += p;
  vst4q_lane_u16(s - 4, o0, 1);
  vst3q_lane_u16(s + 0, o1, 1);
  s += p;
  vst4q_lane_u16(s - 4, o0, 2);
  vst3q_lane_u16(s + 0, o1, 2);
  s += p;
  vst4q_lane_u16(s - 4, o0, 3);
  vst3q_lane_u16(s + 0, o1, 3);
  s += p;
  vst4q_lane_u16(s - 4, o0, 4);
  vst3q_lane_u16(s + 0, o1, 4);
  s += p;
  vst4q_lane_u16(s - 4, o0, 5);
  vst3q_lane_u16(s + 0, o1, 5);
  s += p;
  vst4q_lane_u16(s - 4, o0, 6);
  vst3q_lane_u16(s + 0, o1, 6);
  s += p;
  vst4q_lane_u16(s - 4, o0, 7);
  vst3q_lane_u16(s + 0, o1, 7);
}

static INLINE void store_8x14(uint16_t *s, const int p, const uint16x8_t p6,
                              const uint16x8_t p5, const uint16x8_t p4,
                              const uint16x8_t p3, const uint16x8_t p2,
                              const uint16x8_t p1, const uint16x8_t p0,
                              const uint16x8_t q0, const uint16x8_t q1,
                              const uint16x8_t q2, const uint16x8_t q3,
                              const uint16x8_t q4, const uint16x8_t q5,
                              const uint16x8_t q6, const uint32_t flat_status,
                              const uint32_t flat2_status) {
  if (flat_status) {
    if (flat2_status) {
      vst1q_u16(s - 7 * p, p6);
      vst1q_u16(s - 6 * p, p5);
      vst1q_u16(s - 5 * p, p4);
      vst1q_u16(s - 4 * p, p3);
      vst1q_u16(s + 3 * p, q3);
      vst1q_u16(s + 4 * p, q4);
      vst1q_u16(s + 5 * p, q5);
      vst1q_u16(s + 6 * p, q6);
    }
    vst1q_u16(s - 3 * p, p2);
    vst1q_u16(s + 2 * p, q2);
  }
  vst1q_u16(s - 2 * p, p1);
  vst1q_u16(s - 1 * p, p0);
  vst1q_u16(s + 0 * p, q0);
  vst1q_u16(s + 1 * p, q1);
}

void vpx_highbd_lpf_horizontal_4_neon(uint16_t *s, int p, const uint8_t *blimit,
                                      const uint8_t *limit,
                                      const uint8_t *thresh, int bd) {
  uint16x8_t blimit_vec, limit_vec, thresh_vec, p3, p2, p1, p0, q0, q1, q2, q3,
      mask, hev;

  load_thresh(blimit, limit, thresh, &blimit_vec, &limit_vec, &thresh_vec, bd);
  load_8x8(s - 4 * p, p, &p3, &p2, &p1, &p0, &q0, &q1, &q2, &q3);
  filter_hev_mask4(limit_vec, blimit_vec, thresh_vec, p3, p2, p1, p0, q0, q1,
                   q2, q3, &hev, &mask);
  filter4(mask, hev, p1, p0, q0, q1, &p1, &p0, &q0, &q1, bd);
  store_8x4(s - 2 * p, p, p1, p0, q0, q1);
}

void vpx_highbd_lpf_horizontal_4_dual_neon(
    uint16_t *s, int p, const uint8_t *blimit0, const uint8_t *limit0,
    const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1,
    const uint8_t *thresh1, int bd) {
  vpx_highbd_lpf_horizontal_4_neon(s, p, blimit0, limit0, thresh0, bd);
  vpx_highbd_lpf_horizontal_4_neon(s + 8, p, blimit1, limit1, thresh1, bd);
}

void vpx_highbd_lpf_vertical_4_neon(uint16_t *s, int p, const uint8_t *blimit,
                                    const uint8_t *limit, const uint8_t *thresh,
                                    int bd) {
  uint16x8_t blimit_vec, limit_vec, thresh_vec, p3, p2, p1, p0, q0, q1, q2, q3,
      mask, hev;

  load_8x8(s - 4, p, &p3, &p2, &p1, &p0, &q0, &q1, &q2, &q3);
  transpose_s16_8x8((int16x8_t *)&p3, (int16x8_t *)&p2, (int16x8_t *)&p1,
                    (int16x8_t *)&p0, (int16x8_t *)&q0, (int16x8_t *)&q1,
                    (int16x8_t *)&q2, (int16x8_t *)&q3);
  load_thresh(blimit, limit, thresh, &blimit_vec, &limit_vec, &thresh_vec, bd);
  filter_hev_mask4(limit_vec, blimit_vec, thresh_vec, p3, p2, p1, p0, q0, q1,
                   q2, q3, &hev, &mask);
  filter4(mask, hev, p1, p0, q0, q1, &p1, &p0, &q0, &q1, bd);
  store_4x8(s - 2, p, p1, p0, q0, q1);
}

void vpx_highbd_lpf_vertical_4_dual_neon(
    uint16_t *s, int p, const uint8_t *blimit0, const uint8_t *limit0,
    const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1,
    const uint8_t *thresh1, int bd) {
  vpx_highbd_lpf_vertical_4_neon(s, p, blimit0, limit0, thresh0, bd);
  vpx_highbd_lpf_vertical_4_neon(s + 8 * p, p, blimit1, limit1, thresh1, bd);
}

void vpx_highbd_lpf_horizontal_8_neon(uint16_t *s, int p, const uint8_t *blimit,
                                      const uint8_t *limit,
                                      const uint8_t *thresh, int bd) {
  uint16x8_t blimit_vec, limit_vec, thresh_vec, p3, p2, p1, p0, q0, q1, q2, q3,
      op2, op1, op0, oq0, oq1, oq2, mask, flat, hev;
  uint32_t flat_status;

  load_thresh(blimit, limit, thresh, &blimit_vec, &limit_vec, &thresh_vec, bd);
  load_8x8(s - 4 * p, p, &p3, &p2, &p1, &p0, &q0, &q1, &q2, &q3);
  mask = filter_flat_hev_mask(limit_vec, blimit_vec, thresh_vec, p3, p2, p1, p0,
                              q0, q1, q2, q3, &flat, &flat_status, &hev, bd);
  filter8(mask, flat, flat_status, hev, p3, p2, p1, p0, q0, q1, q2, q3, &op2,
          &op1, &op0, &oq0, &oq1, &oq2, bd);
  store_8x6(s - 3 * p, p, op2, op1, op0, oq0, oq1, oq2);
}

void vpx_highbd_lpf_horizontal_8_dual_neon(
    uint16_t *s, int p, const uint8_t *blimit0, const uint8_t *limit0,
    const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1,
    const uint8_t *thresh1, int bd) {
  vpx_highbd_lpf_horizontal_8_neon(s, p, blimit0, limit0, thresh0, bd);
  vpx_highbd_lpf_horizontal_8_neon(s + 8, p, blimit1, limit1, thresh1, bd);
}

void vpx_highbd_lpf_vertical_8_neon(uint16_t *s, int p, const uint8_t *blimit,
                                    const uint8_t *limit, const uint8_t *thresh,
                                    int bd) {
  uint16x8_t blimit_vec, limit_vec, thresh_vec, p3, p2, p1, p0, q0, q1, q2, q3,
      op2, op1, op0, oq0, oq1, oq2, mask, flat, hev;
  uint32_t flat_status;

  load_8x8(s - 4, p, &p3, &p2, &p1, &p0, &q0, &q1, &q2, &q3);
  transpose_s16_8x8((int16x8_t *)&p3, (int16x8_t *)&p2, (int16x8_t *)&p1,
                    (int16x8_t *)&p0, (int16x8_t *)&q0, (int16x8_t *)&q1,
                    (int16x8_t *)&q2, (int16x8_t *)&q3);
  load_thresh(blimit, limit, thresh, &blimit_vec, &limit_vec, &thresh_vec, bd);
  mask = filter_flat_hev_mask(limit_vec, blimit_vec, thresh_vec, p3, p2, p1, p0,
                              q0, q1, q2, q3, &flat, &flat_status, &hev, bd);
  filter8(mask, flat, flat_status, hev, p3, p2, p1, p0, q0, q1, q2, q3, &op2,
          &op1, &op0, &oq0, &oq1, &oq2, bd);
  // Note: store_6x8() is faster than transpose + store_8x8().
  store_6x8(s, p, op2, op1, op0, oq0, oq1, oq2);
}

void vpx_highbd_lpf_vertical_8_dual_neon(
    uint16_t *s, int p, const uint8_t *blimit0, const uint8_t *limit0,
    const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1,
    const uint8_t *thresh1, int bd) {
  vpx_highbd_lpf_vertical_8_neon(s, p, blimit0, limit0, thresh0, bd);
  vpx_highbd_lpf_vertical_8_neon(s + 8 * p, p, blimit1, limit1, thresh1, bd);
}

// Quiet warnings of the form: 'vpx_dsp/arm/highbd_loopfilter_neon.c|675 col 67|
// warning: 'oq1' may be used uninitialized in this function
// [-Wmaybe-uninitialized]', for oq1-op1. Without reworking the code or adding
// an additional branch this warning cannot be silenced otherwise. The
// loopfilter is only called when needed for a block so these output pixels
// will be set.
#if defined(__GNUC__) && __GNUC__ >= 4 && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

static void lpf_horizontal_16_kernel(uint16_t *s, int p,
                                     const uint16x8_t blimit_vec,
                                     const uint16x8_t limit_vec,
                                     const uint16x8_t thresh_vec,
                                     const int bd) {
  uint16x8_t mask, flat, flat2, hev, p7, p6, p5, p4, p3, p2, p1, p0, q0, q1, q2,
      q3, q4, q5, q6, q7, op6, op5, op4, op3, op2, op1, op0, oq0, oq1, oq2, oq3,
      oq4, oq5, oq6;
  uint32_t flat_status, flat2_status;

  load_8x16(s - 8 * p, p, &p7, &p6, &p5, &p4, &p3, &p2, &p1, &p0, &q0, &q1, &q2,
            &q3, &q4, &q5, &q6, &q7);
  mask = filter_flat_hev_mask(limit_vec, blimit_vec, thresh_vec, p3, p2, p1, p0,
                              q0, q1, q2, q3, &flat, &flat_status, &hev, bd);
  flat2 = flat_mask5(p7, p6, p5, p4, p0, q0, q4, q5, q6, q7, flat,
                     &flat2_status, bd);
  filter16(mask, flat, flat_status, flat2, flat2_status, hev, p7, p6, p5, p4,
           p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6, q7, &op6, &op5, &op4,
           &op3, &op2, &op1, &op0, &oq0, &oq1, &oq2, &oq3, &oq4, &oq5, &oq6,
           bd);
  store_8x14(s, p, op6, op5, op4, op3, op2, op1, op0, oq0, oq1, oq2, oq3, oq4,
             oq5, oq6, flat_status, flat2_status);
}

static void lpf_vertical_16_kernel(uint16_t *s, int p,
                                   const uint16x8_t blimit_vec,
                                   const uint16x8_t limit_vec,
                                   const uint16x8_t thresh_vec, const int bd) {
  uint16x8_t mask, flat, flat2, hev, p7, p6, p5, p4, p3, p2, p1, p0, q0, q1, q2,
      q3, q4, q5, q6, q7, op6, op5, op4, op3, op2, op1, op0, oq0, oq1, oq2, oq3,
      oq4, oq5, oq6;
  uint32_t flat_status, flat2_status;

  load_8x8(s - 8, p, &p7, &p6, &p5, &p4, &p3, &p2, &p1, &p0);
  transpose_s16_8x8((int16x8_t *)&p7, (int16x8_t *)&p6, (int16x8_t *)&p5,
                    (int16x8_t *)&p4, (int16x8_t *)&p3, (int16x8_t *)&p2,
                    (int16x8_t *)&p1, (int16x8_t *)&p0);
  load_8x8(s, p, &q0, &q1, &q2, &q3, &q4, &q5, &q6, &q7);
  transpose_s16_8x8((int16x8_t *)&q0, (int16x8_t *)&q1, (int16x8_t *)&q2,
                    (int16x8_t *)&q3, (int16x8_t *)&q4, (int16x8_t *)&q5,
                    (int16x8_t *)&q6, (int16x8_t *)&q7);
  mask = filter_flat_hev_mask(limit_vec, blimit_vec, thresh_vec, p3, p2, p1, p0,
                              q0, q1, q2, q3, &flat, &flat_status, &hev, bd);
  flat2 = flat_mask5(p7, p6, p5, p4, p0, q0, q4, q5, q6, q7, flat,
                     &flat2_status, bd);
  filter16(mask, flat, flat_status, flat2, flat2_status, hev, p7, p6, p5, p4,
           p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6, q7, &op6, &op5, &op4,
           &op3, &op2, &op1, &op0, &oq0, &oq1, &oq2, &oq3, &oq4, &oq5, &oq6,
           bd);
  if (flat_status) {
    if (flat2_status) {
      store_7x8(s - 3, p, op6, op5, op4, op3, op2, op1, op0);
      store_7x8(s + 4, p, oq0, oq1, oq2, oq3, oq4, oq5, oq6);
    } else {
      // Note: store_6x8() is faster than transpose + store_8x8().
      store_6x8(s, p, op2, op1, op0, oq0, oq1, oq2);
    }
  } else {
    store_4x8(s - 2, p, op1, op0, oq0, oq1);
  }
}

#if defined(__GNUC__) && __GNUC__ >= 4 && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

void vpx_highbd_lpf_horizontal_16_neon(uint16_t *s, int p,
                                       const uint8_t *blimit,
                                       const uint8_t *limit,
                                       const uint8_t *thresh, int bd) {
  uint16x8_t blimit_vec, limit_vec, thresh_vec;
  load_thresh(blimit, limit, thresh, &blimit_vec, &limit_vec, &thresh_vec, bd);
  lpf_horizontal_16_kernel(s, p, blimit_vec, limit_vec, thresh_vec, bd);
}

void vpx_highbd_lpf_horizontal_16_dual_neon(uint16_t *s, int p,
                                            const uint8_t *blimit,
                                            const uint8_t *limit,
                                            const uint8_t *thresh, int bd) {
  uint16x8_t blimit_vec, limit_vec, thresh_vec;
  load_thresh(blimit, limit, thresh, &blimit_vec, &limit_vec, &thresh_vec, bd);
  lpf_horizontal_16_kernel(s, p, blimit_vec, limit_vec, thresh_vec, bd);
  lpf_horizontal_16_kernel(s + 8, p, blimit_vec, limit_vec, thresh_vec, bd);
}

void vpx_highbd_lpf_vertical_16_neon(uint16_t *s, int p, const uint8_t *blimit,
                                     const uint8_t *limit,
                                     const uint8_t *thresh, int bd) {
  uint16x8_t blimit_vec, limit_vec, thresh_vec;
  load_thresh(blimit, limit, thresh, &blimit_vec, &limit_vec, &thresh_vec, bd);
  lpf_vertical_16_kernel(s, p, blimit_vec, limit_vec, thresh_vec, bd);
}

void vpx_highbd_lpf_vertical_16_dual_neon(uint16_t *s, int p,
                                          const uint8_t *blimit,
                                          const uint8_t *limit,
                                          const uint8_t *thresh, int bd) {
  uint16x8_t blimit_vec, limit_vec, thresh_vec;
  load_thresh(blimit, limit, thresh, &blimit_vec, &limit_vec, &thresh_vec, bd);
  lpf_vertical_16_kernel(s, p, blimit_vec, limit_vec, thresh_vec, bd);
  lpf_vertical_16_kernel(s + 8 * p, p, blimit_vec, limit_vec, thresh_vec, bd);
}
