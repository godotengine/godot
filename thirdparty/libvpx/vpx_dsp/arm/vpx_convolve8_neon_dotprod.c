/*
 *  Copyright (c) 2021 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <arm_neon.h>
#include <assert.h>

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/arm/mem_neon.h"
#include "vpx_dsp/arm/transpose_neon.h"
#include "vpx_dsp/arm/vpx_convolve8_neon.h"
#include "vpx_dsp/vpx_filter.h"
#include "vpx_ports/mem.h"

// Filter values always sum to 128.
#define FILTER_SUM 128

DECLARE_ALIGNED(16, static const uint8_t, dot_prod_permute_tbl[48]) = {
  0, 1, 2,  3,  1, 2,  3,  4,  2,  3,  4,  5,  3,  4,  5,  6,
  4, 5, 6,  7,  5, 6,  7,  8,  6,  7,  8,  9,  7,  8,  9,  10,
  8, 9, 10, 11, 9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14
};

DECLARE_ALIGNED(16, static const uint8_t, dot_prod_merge_block_tbl[48]) = {
  // Shift left and insert new last column in transposed 4x4 block.
  1, 2, 3, 16, 5, 6, 7, 20, 9, 10, 11, 24, 13, 14, 15, 28,
  // Shift left and insert two new columns in transposed 4x4 block.
  2, 3, 16, 17, 6, 7, 20, 21, 10, 11, 24, 25, 14, 15, 28, 29,
  // Shift left and insert three new columns in transposed 4x4 block.
  3, 16, 17, 18, 7, 20, 21, 22, 11, 24, 25, 26, 15, 28, 29, 30
};

static INLINE int16x4_t convolve4_4_h(const uint8x16_t samples,
                                      const int8x8_t filters,
                                      const uint8x16_t permute_tbl) {
  // Transform sample range to [-128, 127] for 8-bit signed dot product.
  int8x16_t samples_128 =
      vreinterpretq_s8_u8(vsubq_u8(samples, vdupq_n_u8(128)));

  // Permute samples ready for dot product.
  // { 0,  1,  2,  3,  1,  2,  3,  4,  2,  3,  4,  5,  3,  4,  5,  6 }
  int8x16_t perm_samples = vqtbl1q_s8(samples_128, permute_tbl);

  // Accumulate into 128 * FILTER_SUM to account for range transform. (Divide
  // by 2 since we halved the filter values.)
  int32x4_t acc = vdupq_n_s32(128 * FILTER_SUM / 2);
  int32x4_t sum = vdotq_lane_s32(acc, perm_samples, filters, 0);

  // Further narrowing and packing is performed by the caller.
  return vmovn_s32(sum);
}

static INLINE uint8x8_t convolve4_8_h(const uint8x16_t samples,
                                      const int8x8_t filters,
                                      const uint8x16x2_t permute_tbl) {
  // Transform sample range to [-128, 127] for 8-bit signed dot product.
  int8x16_t samples_128 =
      vreinterpretq_s8_u8(vsubq_u8(samples, vdupq_n_u8(128)));

  // Permute samples ready for dot product.
  // { 0,  1,  2,  3,  1,  2,  3,  4,  2,  3,  4,  5,  3,  4,  5,  6 }
  // { 4,  5,  6,  7,  5,  6,  7,  8,  6,  7,  8,  9,  7,  8,  9, 10 }
  int8x16_t perm_samples[2] = { vqtbl1q_s8(samples_128, permute_tbl.val[0]),
                                vqtbl1q_s8(samples_128, permute_tbl.val[1]) };

  // Accumulate into 128 * FILTER_SUM to account for range transform. (Divide
  // by 2 since we halved the filter values.)
  int32x4_t acc = vdupq_n_s32(128 * FILTER_SUM / 2);
  // First 4 output values.
  int32x4_t sum0 = vdotq_lane_s32(acc, perm_samples[0], filters, 0);
  // Second 4 output values.
  int32x4_t sum1 = vdotq_lane_s32(acc, perm_samples[1], filters, 0);

  // Narrow and re-pack.
  int16x8_t sum = vcombine_s16(vmovn_s32(sum0), vmovn_s32(sum1));
  // We halved the filter values so -1 from right shift.
  return vqrshrun_n_s16(sum, FILTER_BITS - 1);
}

static INLINE int16x4_t convolve8_4_h(const uint8x16_t samples,
                                      const int8x8_t filters,
                                      const uint8x16x2_t permute_tbl) {
  // Transform sample range to [-128, 127] for 8-bit signed dot product.
  int8x16_t samples_128 =
      vreinterpretq_s8_u8(vsubq_u8(samples, vdupq_n_u8(128)));

  // Permute samples ready for dot product.
  // { 0,  1,  2,  3,  1,  2,  3,  4,  2,  3,  4,  5,  3,  4,  5,  6 }
  // { 4,  5,  6,  7,  5,  6,  7,  8,  6,  7,  8,  9,  7,  8,  9, 10 }
  int8x16_t perm_samples[2] = { vqtbl1q_s8(samples_128, permute_tbl.val[0]),
                                vqtbl1q_s8(samples_128, permute_tbl.val[1]) };

  // Accumulate into 128 * FILTER_SUM to account for range transform.
  int32x4_t acc = vdupq_n_s32(128 * FILTER_SUM);
  int32x4_t sum = vdotq_lane_s32(acc, perm_samples[0], filters, 0);
  sum = vdotq_lane_s32(sum, perm_samples[1], filters, 1);

  // Further narrowing and packing is performed by the caller.
  return vshrn_n_s32(sum, 1);
}

static INLINE uint8x8_t convolve8_8_h(const uint8x16_t samples,
                                      const int8x8_t filters,
                                      const uint8x16x3_t permute_tbl) {
  // Transform sample range to [-128, 127] for 8-bit signed dot product.
  int8x16_t samples_128 =
      vreinterpretq_s8_u8(vsubq_u8(samples, vdupq_n_u8(128)));

  // Permute samples ready for dot product.
  // { 0,  1,  2,  3,  1,  2,  3,  4,  2,  3,  4,  5,  3,  4,  5,  6 }
  // { 4,  5,  6,  7,  5,  6,  7,  8,  6,  7,  8,  9,  7,  8,  9, 10 }
  // { 8,  9, 10, 11,  9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14 }
  int8x16_t perm_samples[3] = { vqtbl1q_s8(samples_128, permute_tbl.val[0]),
                                vqtbl1q_s8(samples_128, permute_tbl.val[1]),
                                vqtbl1q_s8(samples_128, permute_tbl.val[2]) };

  // Accumulate into 128 * FILTER_SUM to account for range transform.
  int32x4_t acc = vdupq_n_s32(128 * FILTER_SUM);
  // First 4 output values.
  int32x4_t sum0 = vdotq_lane_s32(acc, perm_samples[0], filters, 0);
  sum0 = vdotq_lane_s32(sum0, perm_samples[1], filters, 1);
  // Second 4 output values.
  int32x4_t sum1 = vdotq_lane_s32(acc, perm_samples[1], filters, 0);
  sum1 = vdotq_lane_s32(sum1, perm_samples[2], filters, 1);

  // Narrow and re-pack.
  int16x8_t sum = vcombine_s16(vshrn_n_s32(sum0, 1), vshrn_n_s32(sum1, 1));
  return vqrshrun_n_s16(sum, FILTER_BITS - 1);
}

static INLINE void convolve_4tap_horiz_neon_dotprod(
    const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst,
    ptrdiff_t dst_stride, int w, int h, const int8x8_t filter) {
  if (w == 4) {
    const uint8x16_t permute_tbl = vld1q_u8(dot_prod_permute_tbl);

    do {
      uint8x16_t s0, s1, s2, s3;
      load_u8_16x4(src, src_stride, &s0, &s1, &s2, &s3);

      int16x4_t t0 = convolve4_4_h(s0, filter, permute_tbl);
      int16x4_t t1 = convolve4_4_h(s1, filter, permute_tbl);
      int16x4_t t2 = convolve4_4_h(s2, filter, permute_tbl);
      int16x4_t t3 = convolve4_4_h(s3, filter, permute_tbl);
      // We halved the filter values so -1 from right shift.
      uint8x8_t d01 = vqrshrun_n_s16(vcombine_s16(t0, t1), FILTER_BITS - 1);
      uint8x8_t d23 = vqrshrun_n_s16(vcombine_s16(t2, t3), FILTER_BITS - 1);

      store_u8(dst + 0 * dst_stride, dst_stride, d01);
      store_u8(dst + 2 * dst_stride, dst_stride, d23);

      src += 4 * src_stride;
      dst += 4 * dst_stride;
      h -= 4;
    } while (h != 0);
  } else {
    const uint8x16x2_t permute_tbl = vld1q_u8_x2(dot_prod_permute_tbl);

    do {
      const uint8_t *s = src;
      uint8_t *d = dst;
      int width = w;

      do {
        uint8x16_t s0, s1, s2, s3;
        load_u8_16x4(s, src_stride, &s0, &s1, &s2, &s3);

        uint8x8_t d0 = convolve4_8_h(s0, filter, permute_tbl);
        uint8x8_t d1 = convolve4_8_h(s1, filter, permute_tbl);
        uint8x8_t d2 = convolve4_8_h(s2, filter, permute_tbl);
        uint8x8_t d3 = convolve4_8_h(s3, filter, permute_tbl);

        store_u8_8x4(d, dst_stride, d0, d1, d2, d3);

        s += 8;
        d += 8;
        width -= 8;
      } while (width != 0);
      src += 4 * src_stride;
      dst += 4 * dst_stride;
      h -= 4;
    } while (h != 0);
  }
}

static INLINE void convolve_8tap_horiz_neon_dotprod(
    const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst,
    ptrdiff_t dst_stride, int w, int h, const int8x8_t filter) {
  if (w == 4) {
    const uint8x16x2_t permute_tbl = vld1q_u8_x2(dot_prod_permute_tbl);

    do {
      uint8x16_t s0, s1, s2, s3;
      load_u8_16x4(src, src_stride, &s0, &s1, &s2, &s3);

      int16x4_t t0 = convolve8_4_h(s0, filter, permute_tbl);
      int16x4_t t1 = convolve8_4_h(s1, filter, permute_tbl);
      int16x4_t t2 = convolve8_4_h(s2, filter, permute_tbl);
      int16x4_t t3 = convolve8_4_h(s3, filter, permute_tbl);
      uint8x8_t d01 = vqrshrun_n_s16(vcombine_s16(t0, t1), FILTER_BITS - 1);
      uint8x8_t d23 = vqrshrun_n_s16(vcombine_s16(t2, t3), FILTER_BITS - 1);

      store_u8(dst + 0 * dst_stride, dst_stride, d01);
      store_u8(dst + 2 * dst_stride, dst_stride, d23);

      src += 4 * src_stride;
      dst += 4 * dst_stride;
      h -= 4;
    } while (h != 0);
  } else {
    const uint8x16x3_t permute_tbl = vld1q_u8_x3(dot_prod_permute_tbl);

    do {
      const uint8_t *s = src;
      uint8_t *d = dst;
      int width = w;

      do {
        uint8x16_t s0, s1, s2, s3;
        load_u8_16x4(s, src_stride, &s0, &s1, &s2, &s3);

        uint8x8_t d0 = convolve8_8_h(s0, filter, permute_tbl);
        uint8x8_t d1 = convolve8_8_h(s1, filter, permute_tbl);
        uint8x8_t d2 = convolve8_8_h(s2, filter, permute_tbl);
        uint8x8_t d3 = convolve8_8_h(s3, filter, permute_tbl);

        store_u8_8x4(d, dst_stride, d0, d1, d2, d3);

        s += 8;
        d += 8;
        width -= 8;
      } while (width != 0);
      src += 4 * src_stride;
      dst += 4 * dst_stride;
      h -= 4;
    } while (h != 0);
  }
}

void vpx_convolve8_horiz_neon_dotprod(const uint8_t *src, ptrdiff_t src_stride,
                                      uint8_t *dst, ptrdiff_t dst_stride,
                                      const InterpKernel *filter, int x0_q4,
                                      int x_step_q4, int y0_q4, int y_step_q4,
                                      int w, int h) {
  assert((intptr_t)dst % 4 == 0);
  assert(dst_stride % 4 == 0);
  assert(x_step_q4 == 16);

  (void)x_step_q4;
  (void)y0_q4;
  (void)y_step_q4;

  if (vpx_get_filter_taps(filter[x0_q4]) <= 4) {
    // Load 4-tap filter into first 4 elements of the vector.
    // All 4-tap and bilinear filter values are even, so halve them to reduce
    // intermediate precision requirements.
    const int16x4_t x_filter = vld1_s16(filter[x0_q4] + 2);
    const int8x8_t x_filter_4tap =
        vshrn_n_s16(vcombine_s16(x_filter, vdup_n_s16(0)), 1);

    convolve_4tap_horiz_neon_dotprod(src - 1, src_stride, dst, dst_stride, w, h,
                                     x_filter_4tap);

  } else {
    const int8x8_t x_filter_8tap = vmovn_s16(vld1q_s16(filter[x0_q4]));

    convolve_8tap_horiz_neon_dotprod(src - 3, src_stride, dst, dst_stride, w, h,
                                     x_filter_8tap);
  }
}

void vpx_convolve8_avg_horiz_neon_dotprod(const uint8_t *src,
                                          ptrdiff_t src_stride, uint8_t *dst,
                                          ptrdiff_t dst_stride,
                                          const InterpKernel *filter, int x0_q4,
                                          int x_step_q4, int y0_q4,
                                          int y_step_q4, int w, int h) {
  const int8x8_t filters = vmovn_s16(vld1q_s16(filter[x0_q4]));

  assert((intptr_t)dst % 4 == 0);
  assert(dst_stride % 4 == 0);
  assert(x_step_q4 == 16);

  (void)x_step_q4;
  (void)y0_q4;
  (void)y_step_q4;

  src -= 3;

  if (w == 4) {
    const uint8x16x2_t permute_tbl = vld1q_u8_x2(dot_prod_permute_tbl);

    do {
      uint8x16_t s0, s1, s2, s3;
      load_u8_16x4(src, src_stride, &s0, &s1, &s2, &s3);

      int16x4_t t0 = convolve8_4_h(s0, filters, permute_tbl);
      int16x4_t t1 = convolve8_4_h(s1, filters, permute_tbl);
      int16x4_t t2 = convolve8_4_h(s2, filters, permute_tbl);
      int16x4_t t3 = convolve8_4_h(s3, filters, permute_tbl);
      uint8x8_t d01 = vqrshrun_n_s16(vcombine_s16(t0, t1), FILTER_BITS - 1);
      uint8x8_t d23 = vqrshrun_n_s16(vcombine_s16(t2, t3), FILTER_BITS - 1);

      uint8x8_t dd01 = load_u8(dst + 0 * dst_stride, dst_stride);
      uint8x8_t dd23 = load_u8(dst + 2 * dst_stride, dst_stride);

      d01 = vrhadd_u8(d01, dd01);
      d23 = vrhadd_u8(d23, dd23);

      store_u8(dst + 0 * dst_stride, dst_stride, d01);
      store_u8(dst + 2 * dst_stride, dst_stride, d23);

      src += 4 * src_stride;
      dst += 4 * dst_stride;
      h -= 4;
    } while (h != 0);
  } else {
    const uint8x16x3_t permute_tbl = vld1q_u8_x3(dot_prod_permute_tbl);

    do {
      const uint8_t *s = src;
      uint8_t *d = dst;
      int width = w;

      do {
        uint8x16_t s0, s1, s2, s3;
        load_u8_16x4(s, src_stride, &s0, &s1, &s2, &s3);

        uint8x8_t d0 = convolve8_8_h(s0, filters, permute_tbl);
        uint8x8_t d1 = convolve8_8_h(s1, filters, permute_tbl);
        uint8x8_t d2 = convolve8_8_h(s2, filters, permute_tbl);
        uint8x8_t d3 = convolve8_8_h(s3, filters, permute_tbl);

        uint8x8_t dd0, dd1, dd2, dd3;
        load_u8_8x4(d, dst_stride, &dd0, &dd1, &dd2, &dd3);

        d0 = vrhadd_u8(d0, dd0);
        d1 = vrhadd_u8(d1, dd1);
        d2 = vrhadd_u8(d2, dd2);
        d3 = vrhadd_u8(d3, dd3);

        store_u8_8x4(d, dst_stride, d0, d1, d2, d3);

        s += 8;
        d += 8;
        width -= 8;
      } while (width != 0);
      src += 4 * src_stride;
      dst += 4 * dst_stride;
      h -= 4;
    } while (h != 0);
  }
}

static INLINE int16x4_t convolve8_4_v(const int8x16_t samples_lo,
                                      const int8x16_t samples_hi,
                                      const int8x8_t filters) {
  // The sample range transform and permutation are performed by the caller.

  // Accumulate into 128 * FILTER_SUM to account for range transform.
  int32x4_t acc = vdupq_n_s32(128 * FILTER_SUM);
  int32x4_t sum = vdotq_lane_s32(acc, samples_lo, filters, 0);
  sum = vdotq_lane_s32(sum, samples_hi, filters, 1);

  // Further narrowing and packing is performed by the caller.
  return vshrn_n_s32(sum, 1);
}

static INLINE uint8x8_t convolve8_8_v(const int8x16_t samples0_lo,
                                      const int8x16_t samples0_hi,
                                      const int8x16_t samples1_lo,
                                      const int8x16_t samples1_hi,
                                      const int8x8_t filters) {
  // The sample range transform and permutation are performed by the caller.

  // Accumulate into 128 * FILTER_SUM to account for range transform.
  int32x4_t acc = vdupq_n_s32(128 * FILTER_SUM);
  // First 4 output values.
  int32x4_t sum0 = vdotq_lane_s32(acc, samples0_lo, filters, 0);
  sum0 = vdotq_lane_s32(sum0, samples0_hi, filters, 1);
  // Second 4 output values.
  int32x4_t sum1 = vdotq_lane_s32(acc, samples1_lo, filters, 0);
  sum1 = vdotq_lane_s32(sum1, samples1_hi, filters, 1);

  // Narrow and re-pack.
  int16x8_t sum = vcombine_s16(vshrn_n_s32(sum0, 1), vshrn_n_s32(sum1, 1));
  return vqrshrun_n_s16(sum, FILTER_BITS - 1);
}

static INLINE void convolve_8tap_vert_neon_dotprod(
    const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst,
    ptrdiff_t dst_stride, int w, int h, const int8x8_t filter) {
  const uint8x16x3_t merge_block_tbl = vld1q_u8_x3(dot_prod_merge_block_tbl);

  if (w == 4) {
    uint8x8_t t0, t1, t2, t3, t4, t5, t6;
    load_u8_8x7(src, src_stride, &t0, &t1, &t2, &t3, &t4, &t5, &t6);
    src += 7 * src_stride;

    // Transform sample range to [-128, 127] for 8-bit signed dot product.
    int8x8_t s0 = vreinterpret_s8_u8(vsub_u8(t0, vdup_n_u8(128)));
    int8x8_t s1 = vreinterpret_s8_u8(vsub_u8(t1, vdup_n_u8(128)));
    int8x8_t s2 = vreinterpret_s8_u8(vsub_u8(t2, vdup_n_u8(128)));
    int8x8_t s3 = vreinterpret_s8_u8(vsub_u8(t3, vdup_n_u8(128)));
    int8x8_t s4 = vreinterpret_s8_u8(vsub_u8(t4, vdup_n_u8(128)));
    int8x8_t s5 = vreinterpret_s8_u8(vsub_u8(t5, vdup_n_u8(128)));
    int8x8_t s6 = vreinterpret_s8_u8(vsub_u8(t6, vdup_n_u8(128)));

    // This operation combines a conventional transpose and the sample permute
    // (see horizontal case) required before computing the dot product.
    int8x16_t s0123, s1234, s2345, s3456;
    transpose_concat_s8_4x4(s0, s1, s2, s3, &s0123);
    transpose_concat_s8_4x4(s1, s2, s3, s4, &s1234);
    transpose_concat_s8_4x4(s2, s3, s4, s5, &s2345);
    transpose_concat_s8_4x4(s3, s4, s5, s6, &s3456);

    do {
      uint8x8_t t7, t8, t9, t10;
      load_u8_8x4(src, src_stride, &t7, &t8, &t9, &t10);

      int8x8_t s7 = vreinterpret_s8_u8(vsub_u8(t7, vdup_n_u8(128)));
      int8x8_t s8 = vreinterpret_s8_u8(vsub_u8(t8, vdup_n_u8(128)));
      int8x8_t s9 = vreinterpret_s8_u8(vsub_u8(t9, vdup_n_u8(128)));
      int8x8_t s10 = vreinterpret_s8_u8(vsub_u8(t10, vdup_n_u8(128)));

      int8x16_t s78910;
      transpose_concat_s8_4x4(s7, s8, s9, s10, &s78910);

      // Merge new data into block from previous iteration.
      int8x16x2_t samples_LUT = { { s3456, s78910 } };
      int8x16_t s4567 = vqtbl2q_s8(samples_LUT, merge_block_tbl.val[0]);
      int8x16_t s5678 = vqtbl2q_s8(samples_LUT, merge_block_tbl.val[1]);
      int8x16_t s6789 = vqtbl2q_s8(samples_LUT, merge_block_tbl.val[2]);

      int16x4_t d0 = convolve8_4_v(s0123, s4567, filter);
      int16x4_t d1 = convolve8_4_v(s1234, s5678, filter);
      int16x4_t d2 = convolve8_4_v(s2345, s6789, filter);
      int16x4_t d3 = convolve8_4_v(s3456, s78910, filter);
      uint8x8_t d01 = vqrshrun_n_s16(vcombine_s16(d0, d1), FILTER_BITS - 1);
      uint8x8_t d23 = vqrshrun_n_s16(vcombine_s16(d2, d3), FILTER_BITS - 1);

      store_u8(dst + 0 * dst_stride, dst_stride, d01);
      store_u8(dst + 2 * dst_stride, dst_stride, d23);

      /* Prepare block for next iteration - re-using as much as possible. */
      /* Shuffle everything up four rows. */
      s0123 = s4567;
      s1234 = s5678;
      s2345 = s6789;
      s3456 = s78910;

      src += 4 * src_stride;
      dst += 4 * dst_stride;
      h -= 4;
    } while (h != 0);
  } else {
    do {
      const uint8_t *s = src;
      uint8_t *d = dst;
      int height = h;

      uint8x8_t t0, t1, t2, t3, t4, t5, t6;
      load_u8_8x7(s, src_stride, &t0, &t1, &t2, &t3, &t4, &t5, &t6);
      s += 7 * src_stride;

      // Transform sample range to [-128, 127] for 8-bit signed dot product.
      int8x8_t s0 = vreinterpret_s8_u8(vsub_u8(t0, vdup_n_u8(128)));
      int8x8_t s1 = vreinterpret_s8_u8(vsub_u8(t1, vdup_n_u8(128)));
      int8x8_t s2 = vreinterpret_s8_u8(vsub_u8(t2, vdup_n_u8(128)));
      int8x8_t s3 = vreinterpret_s8_u8(vsub_u8(t3, vdup_n_u8(128)));
      int8x8_t s4 = vreinterpret_s8_u8(vsub_u8(t4, vdup_n_u8(128)));
      int8x8_t s5 = vreinterpret_s8_u8(vsub_u8(t5, vdup_n_u8(128)));
      int8x8_t s6 = vreinterpret_s8_u8(vsub_u8(t6, vdup_n_u8(128)));

      // This operation combines a conventional transpose and the sample permute
      // (see horizontal case) required before computing the dot product.
      int8x16_t s0123_lo, s0123_hi, s1234_lo, s1234_hi, s2345_lo, s2345_hi,
          s3456_lo, s3456_hi;
      transpose_concat_s8_8x4(s0, s1, s2, s3, &s0123_lo, &s0123_hi);
      transpose_concat_s8_8x4(s1, s2, s3, s4, &s1234_lo, &s1234_hi);
      transpose_concat_s8_8x4(s2, s3, s4, s5, &s2345_lo, &s2345_hi);
      transpose_concat_s8_8x4(s3, s4, s5, s6, &s3456_lo, &s3456_hi);

      do {
        uint8x8_t t7, t8, t9, t10;
        load_u8_8x4(s, src_stride, &t7, &t8, &t9, &t10);

        int8x8_t s7 = vreinterpret_s8_u8(vsub_u8(t7, vdup_n_u8(128)));
        int8x8_t s8 = vreinterpret_s8_u8(vsub_u8(t8, vdup_n_u8(128)));
        int8x8_t s9 = vreinterpret_s8_u8(vsub_u8(t9, vdup_n_u8(128)));
        int8x8_t s10 = vreinterpret_s8_u8(vsub_u8(t10, vdup_n_u8(128)));

        int8x16_t s78910_lo, s78910_hi;
        transpose_concat_s8_8x4(s7, s8, s9, s10, &s78910_lo, &s78910_hi);

        // Merge new data into block from previous iteration.
        int8x16x2_t samples_LUT = { { s3456_lo, s78910_lo } };
        int8x16_t s4567_lo = vqtbl2q_s8(samples_LUT, merge_block_tbl.val[0]);
        int8x16_t s5678_lo = vqtbl2q_s8(samples_LUT, merge_block_tbl.val[1]);
        int8x16_t s6789_lo = vqtbl2q_s8(samples_LUT, merge_block_tbl.val[2]);

        samples_LUT.val[0] = s3456_hi;
        samples_LUT.val[1] = s78910_hi;
        int8x16_t s4567_hi = vqtbl2q_s8(samples_LUT, merge_block_tbl.val[0]);
        int8x16_t s5678_hi = vqtbl2q_s8(samples_LUT, merge_block_tbl.val[1]);
        int8x16_t s6789_hi = vqtbl2q_s8(samples_LUT, merge_block_tbl.val[2]);

        uint8x8_t d0 =
            convolve8_8_v(s0123_lo, s4567_lo, s0123_hi, s4567_hi, filter);
        uint8x8_t d1 =
            convolve8_8_v(s1234_lo, s5678_lo, s1234_hi, s5678_hi, filter);
        uint8x8_t d2 =
            convolve8_8_v(s2345_lo, s6789_lo, s2345_hi, s6789_hi, filter);
        uint8x8_t d3 =
            convolve8_8_v(s3456_lo, s78910_lo, s3456_hi, s78910_hi, filter);

        store_u8_8x4(d, dst_stride, d0, d1, d2, d3);

        // Prepare block for next iteration - re-using as much as possible.
        // Shuffle everything up four rows.
        s0123_lo = s4567_lo;
        s0123_hi = s4567_hi;
        s1234_lo = s5678_lo;
        s1234_hi = s5678_hi;
        s2345_lo = s6789_lo;
        s2345_hi = s6789_hi;
        s3456_lo = s78910_lo;
        s3456_hi = s78910_hi;

        s += 4 * src_stride;
        d += 4 * dst_stride;
        height -= 4;
      } while (height != 0);
      src += 8;
      dst += 8;
      w -= 8;
    } while (w != 0);
  }
}

void vpx_convolve8_vert_neon_dotprod(const uint8_t *src, ptrdiff_t src_stride,
                                     uint8_t *dst, ptrdiff_t dst_stride,
                                     const InterpKernel *filter, int x0_q4,
                                     int x_step_q4, int y0_q4, int y_step_q4,
                                     int w, int h) {
  assert((intptr_t)dst % 4 == 0);
  assert(dst_stride % 4 == 0);
  assert(y_step_q4 == 16);

  (void)x0_q4;
  (void)x_step_q4;
  (void)y_step_q4;

  if (vpx_get_filter_taps(filter[y0_q4]) <= 4) {
    const int16x8_t y_filter = vld1q_s16(filter[y0_q4]);

    convolve_4tap_vert_neon(src - src_stride, src_stride, dst, dst_stride, w, h,
                            y_filter);
  } else {
    const int8x8_t y_filter = vmovn_s16(vld1q_s16(filter[y0_q4]));

    convolve_8tap_vert_neon_dotprod(src - 3 * src_stride, src_stride, dst,
                                    dst_stride, w, h, y_filter);
  }
}

void vpx_convolve8_avg_vert_neon_dotprod(const uint8_t *src,
                                         ptrdiff_t src_stride, uint8_t *dst,
                                         ptrdiff_t dst_stride,
                                         const InterpKernel *filter, int x0_q4,
                                         int x_step_q4, int y0_q4,
                                         int y_step_q4, int w, int h) {
  const int8x8_t filters = vmovn_s16(vld1q_s16(filter[y0_q4]));
  const uint8x16x3_t merge_block_tbl = vld1q_u8_x3(dot_prod_merge_block_tbl);

  assert((intptr_t)dst % 4 == 0);
  assert(dst_stride % 4 == 0);
  assert(y_step_q4 == 16);

  (void)x0_q4;
  (void)x_step_q4;
  (void)y_step_q4;

  src -= 3 * src_stride;

  if (w == 4) {
    uint8x8_t t0, t1, t2, t3, t4, t5, t6;
    load_u8_8x7(src, src_stride, &t0, &t1, &t2, &t3, &t4, &t5, &t6);
    src += 7 * src_stride;

    // Transform sample range to [-128, 127] for 8-bit signed dot product.
    int8x8_t s0 = vreinterpret_s8_u8(vsub_u8(t0, vdup_n_u8(128)));
    int8x8_t s1 = vreinterpret_s8_u8(vsub_u8(t1, vdup_n_u8(128)));
    int8x8_t s2 = vreinterpret_s8_u8(vsub_u8(t2, vdup_n_u8(128)));
    int8x8_t s3 = vreinterpret_s8_u8(vsub_u8(t3, vdup_n_u8(128)));
    int8x8_t s4 = vreinterpret_s8_u8(vsub_u8(t4, vdup_n_u8(128)));
    int8x8_t s5 = vreinterpret_s8_u8(vsub_u8(t5, vdup_n_u8(128)));
    int8x8_t s6 = vreinterpret_s8_u8(vsub_u8(t6, vdup_n_u8(128)));

    // This operation combines a conventional transpose and the sample permute
    // (see horizontal case) required before computing the dot product.
    int8x16_t s0123, s1234, s2345, s3456;
    transpose_concat_s8_4x4(s0, s1, s2, s3, &s0123);
    transpose_concat_s8_4x4(s1, s2, s3, s4, &s1234);
    transpose_concat_s8_4x4(s2, s3, s4, s5, &s2345);
    transpose_concat_s8_4x4(s3, s4, s5, s6, &s3456);

    do {
      uint8x8_t t7, t8, t9, t10;
      load_u8_8x4(src, src_stride, &t7, &t8, &t9, &t10);

      int8x8_t s7 = vreinterpret_s8_u8(vsub_u8(t7, vdup_n_u8(128)));
      int8x8_t s8 = vreinterpret_s8_u8(vsub_u8(t8, vdup_n_u8(128)));
      int8x8_t s9 = vreinterpret_s8_u8(vsub_u8(t9, vdup_n_u8(128)));
      int8x8_t s10 = vreinterpret_s8_u8(vsub_u8(t10, vdup_n_u8(128)));

      int8x16_t s78910;
      transpose_concat_s8_4x4(s7, s8, s9, s10, &s78910);

      // Merge new data into block from previous iteration.
      int8x16x2_t samples_LUT = { { s3456, s78910 } };
      int8x16_t s4567 = vqtbl2q_s8(samples_LUT, merge_block_tbl.val[0]);
      int8x16_t s5678 = vqtbl2q_s8(samples_LUT, merge_block_tbl.val[1]);
      int8x16_t s6789 = vqtbl2q_s8(samples_LUT, merge_block_tbl.val[2]);

      int16x4_t d0 = convolve8_4_v(s0123, s4567, filters);
      int16x4_t d1 = convolve8_4_v(s1234, s5678, filters);
      int16x4_t d2 = convolve8_4_v(s2345, s6789, filters);
      int16x4_t d3 = convolve8_4_v(s3456, s78910, filters);
      uint8x8_t d01 = vqrshrun_n_s16(vcombine_s16(d0, d1), FILTER_BITS - 1);
      uint8x8_t d23 = vqrshrun_n_s16(vcombine_s16(d2, d3), FILTER_BITS - 1);

      uint8x8_t dd01 = load_u8(dst + 0 * dst_stride, dst_stride);
      uint8x8_t dd23 = load_u8(dst + 2 * dst_stride, dst_stride);

      d01 = vrhadd_u8(d01, dd01);
      d23 = vrhadd_u8(d23, dd23);

      store_u8(dst + 0 * dst_stride, dst_stride, d01);
      store_u8(dst + 2 * dst_stride, dst_stride, d23);

      // Prepare block for next iteration - re-using as much as possible.
      // Shuffle everything up four rows.
      s0123 = s4567;
      s1234 = s5678;
      s2345 = s6789;
      s3456 = s78910;

      src += 4 * src_stride;
      dst += 4 * dst_stride;
      h -= 4;
    } while (h != 0);
  } else {
    do {
      const uint8_t *s = src;
      uint8_t *d = dst;
      int height = h;

      uint8x8_t t0, t1, t2, t3, t4, t5, t6;
      load_u8_8x7(s, src_stride, &t0, &t1, &t2, &t3, &t4, &t5, &t6);
      s += 7 * src_stride;

      // Transform sample range to [-128, 127] for 8-bit signed dot product.
      int8x8_t s0 = vreinterpret_s8_u8(vsub_u8(t0, vdup_n_u8(128)));
      int8x8_t s1 = vreinterpret_s8_u8(vsub_u8(t1, vdup_n_u8(128)));
      int8x8_t s2 = vreinterpret_s8_u8(vsub_u8(t2, vdup_n_u8(128)));
      int8x8_t s3 = vreinterpret_s8_u8(vsub_u8(t3, vdup_n_u8(128)));
      int8x8_t s4 = vreinterpret_s8_u8(vsub_u8(t4, vdup_n_u8(128)));
      int8x8_t s5 = vreinterpret_s8_u8(vsub_u8(t5, vdup_n_u8(128)));
      int8x8_t s6 = vreinterpret_s8_u8(vsub_u8(t6, vdup_n_u8(128)));

      // This operation combines a conventional transpose and the sample permute
      // (see horizontal case) required before computing the dot product.
      int8x16_t s0123_lo, s0123_hi, s1234_lo, s1234_hi, s2345_lo, s2345_hi,
          s3456_lo, s3456_hi;
      transpose_concat_s8_8x4(s0, s1, s2, s3, &s0123_lo, &s0123_hi);
      transpose_concat_s8_8x4(s1, s2, s3, s4, &s1234_lo, &s1234_hi);
      transpose_concat_s8_8x4(s2, s3, s4, s5, &s2345_lo, &s2345_hi);
      transpose_concat_s8_8x4(s3, s4, s5, s6, &s3456_lo, &s3456_hi);

      do {
        uint8x8_t t7, t8, t9, t10;
        load_u8_8x4(s, src_stride, &t7, &t8, &t9, &t10);

        int8x8_t s7 = vreinterpret_s8_u8(vsub_u8(t7, vdup_n_u8(128)));
        int8x8_t s8 = vreinterpret_s8_u8(vsub_u8(t8, vdup_n_u8(128)));
        int8x8_t s9 = vreinterpret_s8_u8(vsub_u8(t9, vdup_n_u8(128)));
        int8x8_t s10 = vreinterpret_s8_u8(vsub_u8(t10, vdup_n_u8(128)));

        int8x16_t s78910_lo, s78910_hi;
        transpose_concat_s8_8x4(s7, s8, s9, s10, &s78910_lo, &s78910_hi);

        // Merge new data into block from previous iteration.
        int8x16x2_t samples_LUT = { { s3456_lo, s78910_lo } };
        int8x16_t s4567_lo = vqtbl2q_s8(samples_LUT, merge_block_tbl.val[0]);
        int8x16_t s5678_lo = vqtbl2q_s8(samples_LUT, merge_block_tbl.val[1]);
        int8x16_t s6789_lo = vqtbl2q_s8(samples_LUT, merge_block_tbl.val[2]);

        samples_LUT.val[0] = s3456_hi;
        samples_LUT.val[1] = s78910_hi;
        int8x16_t s4567_hi = vqtbl2q_s8(samples_LUT, merge_block_tbl.val[0]);
        int8x16_t s5678_hi = vqtbl2q_s8(samples_LUT, merge_block_tbl.val[1]);
        int8x16_t s6789_hi = vqtbl2q_s8(samples_LUT, merge_block_tbl.val[2]);

        uint8x8_t d0 =
            convolve8_8_v(s0123_lo, s4567_lo, s0123_hi, s4567_hi, filters);
        uint8x8_t d1 =
            convolve8_8_v(s1234_lo, s5678_lo, s1234_hi, s5678_hi, filters);
        uint8x8_t d2 =
            convolve8_8_v(s2345_lo, s6789_lo, s2345_hi, s6789_hi, filters);
        uint8x8_t d3 =
            convolve8_8_v(s3456_lo, s78910_lo, s3456_hi, s78910_hi, filters);

        uint8x8_t dd0, dd1, dd2, dd3;
        load_u8_8x4(d, dst_stride, &dd0, &dd1, &dd2, &dd3);

        d0 = vrhadd_u8(d0, dd0);
        d1 = vrhadd_u8(d1, dd1);
        d2 = vrhadd_u8(d2, dd2);
        d3 = vrhadd_u8(d3, dd3);

        store_u8_8x4(d, dst_stride, d0, d1, d2, d3);

        // Prepare block for next iteration - re-using as much as possible.
        // Shuffle everything up four rows.
        s0123_lo = s4567_lo;
        s0123_hi = s4567_hi;
        s1234_lo = s5678_lo;
        s1234_hi = s5678_hi;
        s2345_lo = s6789_lo;
        s2345_hi = s6789_hi;
        s3456_lo = s78910_lo;
        s3456_hi = s78910_hi;

        s += 4 * src_stride;
        d += 4 * dst_stride;
        height -= 4;
      } while (height != 0);
      src += 8;
      dst += 8;
      w -= 8;
    } while (w != 0);
  }
}

static INLINE void convolve_4tap_2d_neon_dotprod(const uint8_t *src,
                                                 ptrdiff_t src_stride,
                                                 uint8_t *dst,
                                                 ptrdiff_t dst_stride, int w,
                                                 int h, const int8x8_t x_filter,
                                                 const uint8x8_t y_filter) {
  // Neon does not have lane-referencing multiply or multiply-accumulate
  // instructions that operate on vectors of 8-bit elements. This means we have
  // to duplicate filter taps into a whole vector and use standard multiply /
  // multiply-accumulate instructions.
  const uint8x8_t y_filter_taps[4] = { vdup_lane_u8(y_filter, 2),
                                       vdup_lane_u8(y_filter, 3),
                                       vdup_lane_u8(y_filter, 4),
                                       vdup_lane_u8(y_filter, 5) };

  if (w == 4) {
    const uint8x16_t permute_tbl = vld1q_u8(dot_prod_permute_tbl);

    uint8x16_t h_s0, h_s1, h_s2;
    load_u8_16x3(src, src_stride, &h_s0, &h_s1, &h_s2);

    int16x4_t t0 = convolve4_4_h(h_s0, x_filter, permute_tbl);
    int16x4_t t1 = convolve4_4_h(h_s1, x_filter, permute_tbl);
    int16x4_t t2 = convolve4_4_h(h_s2, x_filter, permute_tbl);
    // We halved the filter values so -1 from right shift.
    uint8x8_t v_s01 = vqrshrun_n_s16(vcombine_s16(t0, t1), FILTER_BITS - 1);
    uint8x8_t v_s12 = vqrshrun_n_s16(vcombine_s16(t1, t2), FILTER_BITS - 1);

    src += 3 * src_stride;

    do {
      uint8x16_t h_s3, h_s4, h_s5, h_s6;
      load_u8_16x4(src, src_stride, &h_s3, &h_s4, &h_s5, &h_s6);

      int16x4_t t3 = convolve4_4_h(h_s3, x_filter, permute_tbl);
      int16x4_t t4 = convolve4_4_h(h_s4, x_filter, permute_tbl);
      int16x4_t t5 = convolve4_4_h(h_s5, x_filter, permute_tbl);
      int16x4_t t6 = convolve4_4_h(h_s6, x_filter, permute_tbl);
      // We halved the filter values so -1 from right shift.
      uint8x8_t v_s34 = vqrshrun_n_s16(vcombine_s16(t3, t4), FILTER_BITS - 1);
      uint8x8_t v_s56 = vqrshrun_n_s16(vcombine_s16(t5, t6), FILTER_BITS - 1);
      uint8x8_t v_s23 = vext_u8(v_s12, v_s34, 4);
      uint8x8_t v_s45 = vext_u8(v_s34, v_s56, 4);

      uint8x8_t d01 = convolve4_8(v_s01, v_s12, v_s23, v_s34, y_filter_taps);
      uint8x8_t d23 = convolve4_8(v_s23, v_s34, v_s45, v_s56, y_filter_taps);

      store_unaligned_u8(dst + 0 * dst_stride, dst_stride, d01);
      store_unaligned_u8(dst + 2 * dst_stride, dst_stride, d23);

      v_s01 = v_s45;
      v_s12 = v_s56;
      src += 4 * src_stride;
      dst += 4 * dst_stride;
      h -= 4;
    } while (h != 0);
  } else {
    const uint8x16x2_t permute_tbl = vld1q_u8_x2(dot_prod_permute_tbl);

    do {
      const uint8_t *s = src;
      uint8_t *d = dst;
      int height = h;

      uint8x16_t h_s0, h_s1, h_s2;
      load_u8_16x3(s, src_stride, &h_s0, &h_s1, &h_s2);

      uint8x8_t v_s0 = convolve4_8_h(h_s0, x_filter, permute_tbl);
      uint8x8_t v_s1 = convolve4_8_h(h_s1, x_filter, permute_tbl);
      uint8x8_t v_s2 = convolve4_8_h(h_s2, x_filter, permute_tbl);

      s += 3 * src_stride;

      do {
        uint8x16_t h_s3, h_s4, h_s5, h_s6;
        load_u8_16x4(s, src_stride, &h_s3, &h_s4, &h_s5, &h_s6);

        uint8x8_t v_s3 = convolve4_8_h(h_s3, x_filter, permute_tbl);
        uint8x8_t v_s4 = convolve4_8_h(h_s4, x_filter, permute_tbl);
        uint8x8_t v_s5 = convolve4_8_h(h_s5, x_filter, permute_tbl);
        uint8x8_t v_s6 = convolve4_8_h(h_s6, x_filter, permute_tbl);

        uint8x8_t d0 = convolve4_8(v_s0, v_s1, v_s2, v_s3, y_filter_taps);
        uint8x8_t d1 = convolve4_8(v_s1, v_s2, v_s3, v_s4, y_filter_taps);
        uint8x8_t d2 = convolve4_8(v_s2, v_s3, v_s4, v_s5, y_filter_taps);
        uint8x8_t d3 = convolve4_8(v_s3, v_s4, v_s5, v_s6, y_filter_taps);

        store_u8_8x4(d, dst_stride, d0, d1, d2, d3);

        v_s0 = v_s4;
        v_s1 = v_s5;
        v_s2 = v_s6;
        s += 4 * src_stride;
        d += 4 * dst_stride;
        height -= 4;
      } while (height != 0);
      src += 8;
      dst += 8;
      w -= 8;
    } while (w != 0);
  }
}

static INLINE void convolve_8tap_2d_horiz_neon_dotprod(
    const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst,
    ptrdiff_t dst_stride, int w, int h, const int8x8_t filter) {
  if (w == 4) {
    const uint8x16x2_t permute_tbl = vld1q_u8_x2(dot_prod_permute_tbl);

    do {
      uint8x16_t s0, s1, s2, s3;
      load_u8_16x4(src, src_stride, &s0, &s1, &s2, &s3);

      int16x4_t d0 = convolve8_4_h(s0, filter, permute_tbl);
      int16x4_t d1 = convolve8_4_h(s1, filter, permute_tbl);
      int16x4_t d2 = convolve8_4_h(s2, filter, permute_tbl);
      int16x4_t d3 = convolve8_4_h(s3, filter, permute_tbl);
      uint8x8_t d01 = vqrshrun_n_s16(vcombine_s16(d0, d1), FILTER_BITS - 1);
      uint8x8_t d23 = vqrshrun_n_s16(vcombine_s16(d2, d3), FILTER_BITS - 1);

      store_u8(dst + 0 * dst_stride, dst_stride, d01);
      store_u8(dst + 2 * dst_stride, dst_stride, d23);

      src += 4 * src_stride;
      dst += 4 * dst_stride;
      h -= 4;
    } while (h > 3);

    // Process final three rows (h % 4 == 3). See vpx_convolve_neon_i8mm()
    // below for further details on possible values of block height.
    uint8x16_t s0, s1, s2;
    load_u8_16x3(src, src_stride, &s0, &s1, &s2);

    int16x4_t d0 = convolve8_4_h(s0, filter, permute_tbl);
    int16x4_t d1 = convolve8_4_h(s1, filter, permute_tbl);
    int16x4_t d2 = convolve8_4_h(s2, filter, permute_tbl);
    uint8x8_t d01 = vqrshrun_n_s16(vcombine_s16(d0, d1), FILTER_BITS - 1);
    uint8x8_t d23 =
        vqrshrun_n_s16(vcombine_s16(d2, vdup_n_s16(0)), FILTER_BITS - 1);

    store_u8(dst + 0 * dst_stride, dst_stride, d01);
    store_u8_4x1(dst + 2 * dst_stride, d23);
  } else {
    const uint8x16x3_t permute_tbl = vld1q_u8_x3(dot_prod_permute_tbl);

    do {
      const uint8_t *s = src;
      uint8_t *d = dst;
      int width = w;

      do {
        uint8x16_t s0, s1, s2, s3;
        load_u8_16x4(s, src_stride, &s0, &s1, &s2, &s3);

        uint8x8_t d0 = convolve8_8_h(s0, filter, permute_tbl);
        uint8x8_t d1 = convolve8_8_h(s1, filter, permute_tbl);
        uint8x8_t d2 = convolve8_8_h(s2, filter, permute_tbl);
        uint8x8_t d3 = convolve8_8_h(s3, filter, permute_tbl);

        store_u8_8x4(d, dst_stride, d0, d1, d2, d3);

        s += 8;
        d += 8;
        width -= 8;
      } while (width > 0);
      src += 4 * src_stride;
      dst += 4 * dst_stride;
      h -= 4;
    } while (h > 3);

    // Process final three rows (h % 4 == 3). See vpx_convolve_neon_i8mm()
    // below for further details on possible values of block height.
    const uint8_t *s = src;
    uint8_t *d = dst;
    int width = w;

    do {
      uint8x16_t s0, s1, s2;
      load_u8_16x3(s, src_stride, &s0, &s1, &s2);

      uint8x8_t d0 = convolve8_8_h(s0, filter, permute_tbl);
      uint8x8_t d1 = convolve8_8_h(s1, filter, permute_tbl);
      uint8x8_t d2 = convolve8_8_h(s2, filter, permute_tbl);

      store_u8_8x3(d, dst_stride, d0, d1, d2);

      s += 8;
      d += 8;
      width -= 8;
    } while (width > 0);
  }
}

void vpx_convolve8_neon_dotprod(const uint8_t *src, ptrdiff_t src_stride,
                                uint8_t *dst, ptrdiff_t dst_stride,
                                const InterpKernel *filter, int x0_q4,
                                int x_step_q4, int y0_q4, int y_step_q4, int w,
                                int h) {
  assert(x_step_q4 == 16);
  assert(y_step_q4 == 16);

  (void)x_step_q4;
  (void)y_step_q4;

  const int x_filter_taps = vpx_get_filter_taps(filter[x0_q4]) <= 4 ? 4 : 8;
  const int y_filter_taps = vpx_get_filter_taps(filter[y0_q4]) <= 4 ? 4 : 8;
  // Account for needing filter_taps / 2 - 1 lines prior and filter_taps / 2
  // lines post both horizontally and vertically.
  const ptrdiff_t horiz_offset = x_filter_taps / 2 - 1;
  const ptrdiff_t vert_offset = (y_filter_taps / 2 - 1) * src_stride;

  if (x_filter_taps == 4 && y_filter_taps == 4) {
    const int16x4_t x_filter = vld1_s16(filter[x0_q4] + 2);
    const int16x8_t y_filter = vld1q_s16(filter[y0_q4]);

    // 4-tap and bilinear filter values are even, so halve them to reduce
    // intermediate precision requirements.
    const int8x8_t x_filter_4tap =
        vshrn_n_s16(vcombine_s16(x_filter, vdup_n_s16(0)), 1);
    const uint8x8_t y_filter_4tap =
        vshrn_n_u16(vreinterpretq_u16_s16(vabsq_s16(y_filter)), 1);

    convolve_4tap_2d_neon_dotprod(src - horiz_offset - vert_offset, src_stride,
                                  dst, dst_stride, w, h, x_filter_4tap,
                                  y_filter_4tap);
    return;
  }

  // Given our constraints: w <= 64, h <= 64, taps <= 8 we can reduce the
  // maximum buffer size to 64 * (64 + 7).
  DECLARE_ALIGNED(32, uint8_t, im_block[64 * 71]);
  const int im_stride = 64;
  const int im_height = h + SUBPEL_TAPS - 1;

  const int8x8_t x_filter_8tap = vmovn_s16(vld1q_s16(filter[x0_q4]));
  const int8x8_t y_filter_8tap = vmovn_s16(vld1q_s16(filter[y0_q4]));

  convolve_8tap_2d_horiz_neon_dotprod(src - horiz_offset - vert_offset,
                                      src_stride, im_block, im_stride, w,
                                      im_height, x_filter_8tap);

  convolve_8tap_vert_neon_dotprod(im_block, im_stride, dst, dst_stride, w, h,
                                  y_filter_8tap);
}

void vpx_convolve8_avg_neon_dotprod(const uint8_t *src, ptrdiff_t src_stride,
                                    uint8_t *dst, ptrdiff_t dst_stride,
                                    const InterpKernel *filter, int x0_q4,
                                    int x_step_q4, int y0_q4, int y_step_q4,
                                    int w, int h) {
  DECLARE_ALIGNED(32, uint8_t, im_block[64 * 71]);
  const int im_stride = 64;

  // Averaging convolution always uses an 8-tap filter.
  // Account for the vertical phase needing 3 lines prior and 4 lines post.
  const int im_height = h + SUBPEL_TAPS - 1;
  const ptrdiff_t offset = SUBPEL_TAPS / 2 - 1;

  assert(y_step_q4 == 16);
  assert(x_step_q4 == 16);

  const int8x8_t x_filter_8tap = vmovn_s16(vld1q_s16(filter[x0_q4]));

  convolve_8tap_2d_horiz_neon_dotprod(src - offset - offset * src_stride,
                                      src_stride, im_block, im_stride, w,
                                      im_height, x_filter_8tap);

  vpx_convolve8_avg_vert_neon_dotprod(im_block + offset * im_stride, im_stride,
                                      dst, dst_stride, filter, x0_q4, x_step_q4,
                                      y0_q4, y_step_q4, w, h);
}
