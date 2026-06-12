/*
 *  Copyright (c) 2025 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>
#include <arm_neon.h>

#include "./vp9_rtcd.h"
#include "./vpx_config.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/arm/mem_neon.h"
#include "vpx_dsp/arm/transpose_neon.h"
#include "vp9/encoder/vp9_temporal_filter.h"

DECLARE_ALIGNED(16, static const uint8_t, kMatMulPermuteTbl[32]) = {
  // clang-format off
  0,  1,  2,  3,  4,  5,  6,  7,  2,  3,  4,  5,  6,  7,  8,  9,
  4,  5,  6,  7,  8,  9, 10, 11,  6,  7,  8,  9, 10, 11, 12, 13
  // clang-format on
};

DECLARE_ALIGNED(16, static const uint8_t, kDotProdMergeBlockTbl[48]) = {
  // clang-format off
  // Shift left and insert new last column in transposed 4x4 block.
  1,  2,  3, 16,  5,  6,  7, 20,  9, 10, 11, 24, 13, 14, 15, 28,
  // Shift left and insert two new columns in transposed 4x4 block.
  2,  3, 16, 17,  6,  7, 20, 21, 10, 11, 24, 25, 14, 15, 28, 29,
  // Shift left and insert three new columns in transposed 4x4 block.
  3, 16, 17, 18,  7, 20, 21, 22, 11, 24, 25, 26, 15, 28, 29, 30
  // clang-format on
};

static INLINE uint8x8_t convolve12_8_h(uint8x16_t samples[2],
                                       const int8x16_t filter[2],
                                       const uint8x16x2_t perm_tbl) {
  // Permute samples ready for matrix multiply.
  // {  0,  1,  2,  3,  4,  5,  6,  7,  2,  3,  4,  5,  6,  7,  8,  9 }
  // {  4,  5,  6,  7,  8,  9, 10, 11,  6,  7,  8,  9, 10, 11, 12, 13 }
  // {  6,  7,  8,  9, 10, 11, 12, 13,  8,  9, 10, 11, 12, 13, 14, 15 }
  // { 10, 11, 12, 13, 14, 15, 16, 17, 12, 13, 14, 15, 16, 17, 18, 19 }
  uint8x16_t perm_samples[4] = { vqtbl1q_u8(samples[0], perm_tbl.val[0]),
                                 vqtbl1q_u8(samples[0], perm_tbl.val[1]),
                                 vqtbl1q_u8(samples[1], perm_tbl.val[0]),
                                 vqtbl1q_u8(samples[1], perm_tbl.val[1]) };

  // These instructions multiply a 2x8 matrix (samples) by an 8x2 matrix
  // (filter), destructively accumulating into the destination register.
  int32x4_t sum0123 = vusmmlaq_s32(vdupq_n_s32(0), perm_samples[0], filter[0]);
  int32x4_t sum4567 = vusmmlaq_s32(vdupq_n_s32(0), perm_samples[1], filter[0]);
  sum0123 = vusmmlaq_s32(sum0123, perm_samples[2], filter[1]);
  sum4567 = vusmmlaq_s32(sum4567, perm_samples[3], filter[1]);

  // Narrow and re-pack.
  int16x8_t sum_s16 = vcombine_s16(vqrshrn_n_s32(sum0123, FILTER_BITS),
                                   vqrshrn_n_s32(sum4567, FILTER_BITS));
  return vqmovun_s16(sum_s16);
}

void vpx_convolve12_horiz_neon_i8mm(const uint8_t *src, ptrdiff_t src_stride,
                                    uint8_t *dst, ptrdiff_t dst_stride,
                                    const InterpKernel12 *filter, int x0_q4,
                                    int x_step_q4, int y0_q4, int y_step_q4,
                                    int w, int h) {
  // Scaling not supported by Neon implementation.
  if (x_step_q4 != 16) {
    vpx_convolve12_horiz_c(src, src_stride, dst, dst_stride, filter, x0_q4,
                           x_step_q4, y0_q4, y_step_q4, w, h);
    return;
  }

  assert(w == 32 || w == 16 || w == 8);
  assert(h == 32 || h == 16 || h == 8);

  // Split 12-tap filter into two 6-tap filters, masking the top two elements.
  // { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0 }
  const int8x8_t mask = vcreate_s8(0x0000ffffffffffff);
  const int8x8_t filter_0 = vand_s8(vmovn_s16(vld1q_s16(filter[x0_q4])), mask);
  const int8x8_t filter_1 =
      vext_s8(vmovn_s16(vld1q_s16(filter[x0_q4] + 4)), vdup_n_s8(0), 2);

  // Stagger each 6-tap filter to enable use of matrix multiply instructions.
  // { f0, f1, f2, f3, f4, f5,  0,  0,  0, f0, f1, f2, f3, f4, f5,  0 }
  const int8x16_t x_filter[2] = {
    vcombine_s8(filter_0, vext_s8(filter_0, filter_0, 7)),
    vcombine_s8(filter_1, vext_s8(filter_1, filter_1, 7))
  };

  const uint8x16x2_t permute_tbl = vld1q_u8_x2(kMatMulPermuteTbl);

  src -= MAX_FILTER_TAP / 2 - 1;

  do {
    const uint8_t *s = src;
    uint8_t *d = dst;
    int width = w;

    do {
      uint8x16_t s0[2], s1[2], s2[2], s3[2];
      load_u8_16x4(s, src_stride, &s0[0], &s1[0], &s2[0], &s3[0]);
      load_u8_16x4(s + 6, src_stride, &s0[1], &s1[1], &s2[1], &s3[1]);

      uint8x8_t d0 = convolve12_8_h(s0, x_filter, permute_tbl);
      uint8x8_t d1 = convolve12_8_h(s1, x_filter, permute_tbl);
      uint8x8_t d2 = convolve12_8_h(s2, x_filter, permute_tbl);
      uint8x8_t d3 = convolve12_8_h(s3, x_filter, permute_tbl);

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

static INLINE uint8x8_t convolve12_8_v(
    const uint8x16_t s0_lo, const uint8x16_t s0_hi, const uint8x16_t s1_lo,
    const uint8x16_t s1_hi, const uint8x16_t s2_lo, const uint8x16_t s2_hi,
    const int8x8_t filters_0_7, const int8x8_t filters_4_11) {
  // The sample range transform and permutation are performed by the caller.
  int32x4_t sum0123 = vusdotq_lane_s32(vdupq_n_s32(0), s0_lo, filters_0_7, 0);
  sum0123 = vusdotq_lane_s32(sum0123, s1_lo, filters_0_7, 1);
  sum0123 = vusdotq_lane_s32(sum0123, s2_lo, filters_4_11, 1);

  int32x4_t sum4567 = vusdotq_lane_s32(vdupq_n_s32(0), s0_hi, filters_0_7, 0);
  sum4567 = vusdotq_lane_s32(sum4567, s1_hi, filters_0_7, 1);
  sum4567 = vusdotq_lane_s32(sum4567, s2_hi, filters_4_11, 1);

  // Narrow and re-pack.
  int16x8_t sum = vcombine_s16(vqmovn_s32(sum0123), vqmovn_s32(sum4567));
  return vqrshrun_n_s16(sum, FILTER_BITS);
}

void vpx_convolve12_vert_neon_i8mm(const uint8_t *src, ptrdiff_t src_stride,
                                   uint8_t *dst, ptrdiff_t dst_stride,
                                   const InterpKernel12 *filter, int x0_q4,
                                   int x_step_q4, int y0_q4, int y_step_q4,
                                   int w, int h) {
  // Scaling not supported by Neon implementation.
  if (y_step_q4 != 16) {
    vpx_convolve12_vert_c(src, src_stride, dst, dst_stride, filter, x0_q4,
                          x_step_q4, y0_q4, y_step_q4, w, h);
    return;
  }

  assert(w == 32 || w == 16 || w == 8);
  assert(h == 32 || h == 16 || h == 8);

  const int8x8_t filter_0_7 = vmovn_s16(vld1q_s16(filter[y0_q4]));
  const int8x8_t filter_4_11 = vmovn_s16(vld1q_s16(filter[y0_q4] + 4));

  const uint8x16x3_t merge_block_tbl = vld1q_u8_x3(kDotProdMergeBlockTbl);

  src -= src_stride * (MAX_FILTER_TAP / 2 - 1);

  do {
    int height = h;
    const uint8_t *s = src;
    uint8_t *d = dst;

    uint8x8_t s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, sA;
    load_u8_8x11(s, src_stride, &s0, &s1, &s2, &s3, &s4, &s5, &s6, &s7, &s8,
                 &s9, &sA);
    s += 11 * src_stride;

    // This operation combines a conventional transpose and the sample permute
    // (see horizontal case) required before computing the dot product.
    uint8x16_t s0123_lo, s0123_hi, s1234_lo, s1234_hi, s2345_lo, s2345_hi,
        s3456_lo, s3456_hi, s4567_lo, s4567_hi, s5678_lo, s5678_hi, s6789_lo,
        s6789_hi, s789A_lo, s789A_hi;
    transpose_concat_u8_8x4(s0, s1, s2, s3, &s0123_lo, &s0123_hi);
    transpose_concat_u8_8x4(s1, s2, s3, s4, &s1234_lo, &s1234_hi);
    transpose_concat_u8_8x4(s2, s3, s4, s5, &s2345_lo, &s2345_hi);
    transpose_concat_u8_8x4(s3, s4, s5, s6, &s3456_lo, &s3456_hi);
    transpose_concat_u8_8x4(s4, s5, s6, s7, &s4567_lo, &s4567_hi);
    transpose_concat_u8_8x4(s5, s6, s7, s8, &s5678_lo, &s5678_hi);
    transpose_concat_u8_8x4(s6, s7, s8, s9, &s6789_lo, &s6789_hi);
    transpose_concat_u8_8x4(s7, s8, s9, sA, &s789A_lo, &s789A_hi);

    do {
      uint8x8_t sB, sC, sD, sE;
      load_u8_8x4(s, src_stride, &sB, &sC, &sD, &sE);

      uint8x16_t s89AB_lo, s89AB_hi, s9ABC_lo, s9ABC_hi, sABCD_lo, sABCD_hi,
          sBCDE_lo, sBCDE_hi;
      transpose_concat_u8_8x4(sB, sC, sD, sE, &sBCDE_lo, &sBCDE_hi);

      // Merge new data into block from previous iteration.
      uint8x16x2_t samples_LUT_lo = { { s789A_lo, sBCDE_lo } };
      s89AB_lo = vqtbl2q_u8(samples_LUT_lo, merge_block_tbl.val[0]);
      s9ABC_lo = vqtbl2q_u8(samples_LUT_lo, merge_block_tbl.val[1]);
      sABCD_lo = vqtbl2q_u8(samples_LUT_lo, merge_block_tbl.val[2]);

      uint8x16x2_t samples_LUT_hi = { { s789A_hi, sBCDE_hi } };
      s89AB_hi = vqtbl2q_u8(samples_LUT_hi, merge_block_tbl.val[0]);
      s9ABC_hi = vqtbl2q_u8(samples_LUT_hi, merge_block_tbl.val[1]);
      sABCD_hi = vqtbl2q_u8(samples_LUT_hi, merge_block_tbl.val[2]);

      uint8x8_t d0 =
          convolve12_8_v(s0123_lo, s0123_hi, s4567_lo, s4567_hi, s89AB_lo,
                         s89AB_hi, filter_0_7, filter_4_11);
      uint8x8_t d1 =
          convolve12_8_v(s1234_lo, s1234_hi, s5678_lo, s5678_hi, s9ABC_lo,
                         s9ABC_hi, filter_0_7, filter_4_11);
      uint8x8_t d2 =
          convolve12_8_v(s2345_lo, s2345_hi, s6789_lo, s6789_hi, sABCD_lo,
                         sABCD_hi, filter_0_7, filter_4_11);
      uint8x8_t d3 =
          convolve12_8_v(s3456_lo, s3456_hi, s789A_lo, s789A_hi, sBCDE_lo,
                         sBCDE_hi, filter_0_7, filter_4_11);

      store_u8_8x4(d, dst_stride, d0, d1, d2, d3);

      // Prepare block for next iteration - re-using as much as possible.
      // Shuffle everything up four rows.
      s0123_lo = s4567_lo;
      s0123_hi = s4567_hi;
      s1234_lo = s5678_lo;
      s1234_hi = s5678_hi;
      s2345_lo = s6789_lo;
      s2345_hi = s6789_hi;
      s3456_lo = s789A_lo;
      s3456_hi = s789A_hi;
      s4567_lo = s89AB_lo;
      s4567_hi = s89AB_hi;
      s5678_lo = s9ABC_lo;
      s5678_hi = s9ABC_hi;
      s6789_lo = sABCD_lo;
      s6789_hi = sABCD_hi;
      s789A_lo = sBCDE_lo;
      s789A_hi = sBCDE_hi;

      s += 4 * src_stride;
      d += 4 * dst_stride;
      height -= 4;
    } while (height != 0);
    src += 8;
    dst += 8;
    w -= 8;
  } while (w != 0);
}

static INLINE void vpx_convolve12_2d_horiz_neon_i8mm(
    const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst,
    ptrdiff_t dst_stride, const InterpKernel12 *filter, int x0_q4, int w,
    int h) {
  assert(w == 32 || w == 16 || w == 8);
  assert(h % 4 == 3);

  // Split 12-tap filter into two 6-tap filters, masking the top two elements.
  // { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0 }
  const int8x8_t mask = vcreate_s8(0x0000ffffffffffff);
  const int8x8_t filter_0 = vand_s8(vmovn_s16(vld1q_s16(filter[x0_q4])), mask);
  const int8x8_t filter_1 =
      vext_s8(vmovn_s16(vld1q_s16(filter[x0_q4] + 4)), vdup_n_s8(0), 2);

  // Stagger each 6-tap filter to enable use of matrix multiply instructions.
  // { f0, f1, f2, f3, f4, f5,  0,  0,  0, f0, f1, f2, f3, f4, f5,  0 }
  const int8x16_t x_filter[2] = {
    vcombine_s8(filter_0, vext_s8(filter_0, filter_0, 7)),
    vcombine_s8(filter_1, vext_s8(filter_1, filter_1, 7))
  };

  const uint8x16x2_t permute_tbl = vld1q_u8_x2(kMatMulPermuteTbl);

  src -= MAX_FILTER_TAP / 2 - 1;

  do {
    const uint8_t *s = src;
    uint8_t *d = dst;
    int width = w;

    do {
      uint8x16_t s0[2], s1[2], s2[2], s3[2];
      load_u8_16x4(s, src_stride, &s0[0], &s1[0], &s2[0], &s3[0]);
      load_u8_16x4(s + 6, src_stride, &s0[1], &s1[1], &s2[1], &s3[1]);

      uint8x8_t d0 = convolve12_8_h(s0, x_filter, permute_tbl);
      uint8x8_t d1 = convolve12_8_h(s1, x_filter, permute_tbl);
      uint8x8_t d2 = convolve12_8_h(s2, x_filter, permute_tbl);
      uint8x8_t d3 = convolve12_8_h(s3, x_filter, permute_tbl);

      store_u8_8x4(d, dst_stride, d0, d1, d2, d3);

      s += 8;
      d += 8;
      width -= 8;
    } while (width != 0);
    src += 4 * src_stride;
    dst += 4 * dst_stride;
    h -= 4;
  } while (h != 3);

  do {
    uint8x16_t s0[2], s1[2], s2[2];
    load_u8_16x3(src, src_stride, &s0[0], &s1[0], &s2[0]);
    load_u8_16x3(src + 6, src_stride, &s0[1], &s1[1], &s2[1]);

    uint8x8_t d0 = convolve12_8_h(s0, x_filter, permute_tbl);
    uint8x8_t d1 = convolve12_8_h(s1, x_filter, permute_tbl);
    uint8x8_t d2 = convolve12_8_h(s2, x_filter, permute_tbl);

    store_u8_8x3(dst, dst_stride, d0, d1, d2);

    src += 8;
    dst += 8;
    w -= 8;
  } while (w != 0);
}

void vpx_convolve12_neon_i8mm(const uint8_t *src, ptrdiff_t src_stride,
                              uint8_t *dst, ptrdiff_t dst_stride,
                              const InterpKernel12 *filter, int x0_q4,
                              int x_step_q4, int y0_q4, int y_step_q4, int w,
                              int h) {
  // Scaling not supported by Neon implementation.
  if (x_step_q4 != 16 || y_step_q4 != 16) {
    vpx_convolve12_c(src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4,
                     y0_q4, y_step_q4, w, h);
    return;
  }

  assert(w == 32 || w == 16 || w == 8);
  assert(h == 32 || h == 16 || h == 8);

  DECLARE_ALIGNED(32, uint8_t, im_block[BW * (BH + MAX_FILTER_TAP)]);

  const int im_stride = BW;
  // Account for the vertical pass needing MAX_FILTER_TAP / 2 - 1 lines prior
  // and MAX_FILTER_TAP / 2 lines post.
  const int im_height = h + MAX_FILTER_TAP - 1;
  const ptrdiff_t border_offset = MAX_FILTER_TAP / 2 - 1;

  // Filter starting border_offset rows up.
  vpx_convolve12_2d_horiz_neon_i8mm(src - src_stride * border_offset,
                                    src_stride, im_block, im_stride, filter,
                                    x0_q4, w, im_height);

  vpx_convolve12_vert_neon_i8mm(im_block + im_stride * border_offset, im_stride,
                                dst, dst_stride, filter, x0_q4, x_step_q4,
                                y0_q4, y_step_q4, w, h);
}
