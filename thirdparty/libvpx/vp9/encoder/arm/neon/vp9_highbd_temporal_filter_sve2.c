/*
 *  Copyright (c) 2025 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <arm_neon.h>
#include <arm_neon_sve_bridge.h>
#include <arm_sve.h>
#include <assert.h>

#include "./vp9_rtcd.h"
#include "./vpx_config.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/arm/mem_neon.h"
#include "vpx_dsp/arm/transpose_neon.h"
#include "vp9/encoder/vp9_temporal_filter.h"
#include "vpx_dsp/arm/vpx_neon_sve_bridge.h"
#include "vpx_dsp/arm/vpx_neon_sve2_bridge.h"

DECLARE_ALIGNED(16, static const uint16_t, kDotProdPermuteTbl[32]) = {
  // clang-format off
  0,  1,  2,  3,  1,  2,  3,  4,
  2,  3,  4,  5,  3,  4,  5,  6,
  4,  5,  6,  7,  5,  6,  7,  0,
  6,  7,  0,  1,  7,  0,  1,  2,
  // clang-format on
};

static INLINE uint16x8_t highbd_convolve12_8_h(
    const int16x8_t s0, const int16x8_t s1, const int16x8_t s2,
    const int16x8_t filter_0_7, const int16x8_t filter_4_11,
    const uint16x8x4_t perm_tbl, const uint16x8_t max) {
  int16x8_t perm_samples[8];

  perm_samples[0] = vpx_tbl_s16(s0, perm_tbl.val[0]);
  perm_samples[1] = vpx_tbl_s16(s0, perm_tbl.val[1]);
  perm_samples[2] = vpx_tbl2_s16(s0, s1, perm_tbl.val[2]);
  perm_samples[3] = vpx_tbl2_s16(s0, s1, perm_tbl.val[3]);
  perm_samples[4] = vpx_tbl_s16(s1, perm_tbl.val[0]);
  perm_samples[5] = vpx_tbl_s16(s1, perm_tbl.val[1]);
  perm_samples[6] = vpx_tbl2_s16(s1, s2, perm_tbl.val[2]);
  perm_samples[7] = vpx_tbl2_s16(s1, s2, perm_tbl.val[3]);

  int64x2_t sum01 =
      vpx_dotq_lane_s16(vdupq_n_s64(0), perm_samples[0], filter_0_7, 0);
  sum01 = vpx_dotq_lane_s16(sum01, perm_samples[2], filter_0_7, 1);
  sum01 = vpx_dotq_lane_s16(sum01, perm_samples[4], filter_4_11, 1);

  int64x2_t sum23 =
      vpx_dotq_lane_s16(vdupq_n_s64(0), perm_samples[1], filter_0_7, 0);
  sum23 = vpx_dotq_lane_s16(sum23, perm_samples[3], filter_0_7, 1);
  sum23 = vpx_dotq_lane_s16(sum23, perm_samples[5], filter_4_11, 1);

  int64x2_t sum45 =
      vpx_dotq_lane_s16(vdupq_n_s64(0), perm_samples[2], filter_0_7, 0);
  sum45 = vpx_dotq_lane_s16(sum45, perm_samples[4], filter_0_7, 1);
  sum45 = vpx_dotq_lane_s16(sum45, perm_samples[6], filter_4_11, 1);

  int64x2_t sum67 =
      vpx_dotq_lane_s16(vdupq_n_s64(0), perm_samples[3], filter_0_7, 0);
  sum67 = vpx_dotq_lane_s16(sum67, perm_samples[5], filter_0_7, 1);
  sum67 = vpx_dotq_lane_s16(sum67, perm_samples[7], filter_4_11, 1);

  int32x4_t sum0123 = vcombine_s32(vmovn_s64(sum01), vmovn_s64(sum23));
  int32x4_t sum4567 = vcombine_s32(vmovn_s64(sum45), vmovn_s64(sum67));

  uint16x8_t res = vcombine_u16(vqrshrun_n_s32(sum0123, FILTER_BITS),
                                vqrshrun_n_s32(sum4567, FILTER_BITS));
  return vminq_u16(res, max);
}

void vpx_highbd_convolve12_horiz_sve2(const uint16_t *src, ptrdiff_t src_stride,
                                      uint16_t *dst, ptrdiff_t dst_stride,
                                      const InterpKernel12 *filter, int x0_q4,
                                      int x_step_q4, int y0_q4, int y_step_q4,
                                      int w, int h, int bd) {
  // Scaling not supported by SVE2 implementation.
  if (x_step_q4 != 16) {
    vpx_highbd_convolve12_horiz_c(src, src_stride, dst, dst_stride, filter,
                                  x0_q4, x_step_q4, y0_q4, y_step_q4, w, h, bd);
    return;
  }

  assert(w == 32 || w == 16 || w == 8);
  assert(h % 4 == 0);

  const int16x8_t filter_0_7 = vld1q_s16(filter[x0_q4]);
  const int16x8_t filter_4_11 = vld1q_s16(filter[x0_q4] + 4);
  const uint16x8_t max = vdupq_n_u16((1 << bd) - 1);
  uint16x8x4_t permute_tbl = vld1q_u16_x4(kDotProdPermuteTbl);

  // Scale indices by size of the true vector length to avoid reading from an
  // 'undefined' portion of a vector on a system with SVE vectors > 128-bit.
  permute_tbl.val[2] = vsetq_lane_u16(svcnth(), permute_tbl.val[2], 7);
  permute_tbl.val[3] = vsetq_lane_u16(svcnth(), permute_tbl.val[3], 5);
  uint16x8_t permute_tbl_3_offsets =
      vreinterpretq_u16_u64(vdupq_n_u64(svcnth() * 0x0001000100000000ULL));
  permute_tbl.val[3] =
      vaddq_u16(permute_tbl.val[3], permute_tbl_3_offsets);  // 2, 3, 6, 7

  src -= MAX_FILTER_TAP / 2 - 1;

  do {
    const int16_t *s = (const int16_t *)src;
    uint16_t *d = dst;
    int width = w;

    do {
      int16x8_t s0[3], s1[3];

      load_s16_8x3(s + 0 * src_stride, 8, &s0[0], &s0[1], &s0[2]);
      load_s16_8x3(s + 1 * src_stride, 8, &s1[0], &s1[1], &s1[2]);

      uint16x8_t d0 = highbd_convolve12_8_h(s0[0], s0[1], s0[2], filter_0_7,
                                            filter_4_11, permute_tbl, max);
      uint16x8_t d1 = highbd_convolve12_8_h(s1[0], s1[1], s1[2], filter_0_7,
                                            filter_4_11, permute_tbl, max);

      vst1q_u16(d + 0 * dst_stride, d0);
      vst1q_u16(d + 1 * dst_stride, d1);

      s += 8;
      d += 8;
      width -= 8;
    } while (width != 0);
    src += 2 * src_stride;
    dst += 2 * dst_stride;
    h -= 2;
  } while (h != 0);
}

static INLINE uint16x4_t highbd_convolve12_4_v(const int16x8_t s0[2],
                                               const int16x8_t s1[2],
                                               const int16x8_t s2[2],
                                               const int16x8_t filter_0_7,
                                               const int16x8_t filter_4_11,
                                               const uint16x4_t max) {
  int64x2_t sum01 = vpx_dotq_lane_s16(vdupq_n_s64(0), s0[0], filter_0_7, 0);
  sum01 = vpx_dotq_lane_s16(sum01, s1[0], filter_0_7, 1);
  sum01 = vpx_dotq_lane_s16(sum01, s2[0], filter_4_11, 1);

  int64x2_t sum23 = vpx_dotq_lane_s16(vdupq_n_s64(0), s0[1], filter_0_7, 0);
  sum23 = vpx_dotq_lane_s16(sum23, s1[1], filter_0_7, 1);
  sum23 = vpx_dotq_lane_s16(sum23, s2[1], filter_4_11, 1);

  int32x4_t sum0123 = vcombine_s32(vmovn_s64(sum01), vmovn_s64(sum23));

  uint16x4_t res = vqrshrun_n_s32(sum0123, FILTER_BITS);

  return vmin_u16(res, max);
}

void vpx_highbd_convolve12_vert_sve2(const uint16_t *src, ptrdiff_t src_stride,
                                     uint16_t *dst, ptrdiff_t dst_stride,
                                     const InterpKernel12 *filter, int x0_q4,
                                     int x_step_q4, int y0_q4, int y_step_q4,
                                     int w, int h, int bd) {
  // Scaling not supported by SVE2 implementation.
  if (y_step_q4 != 16) {
    vpx_highbd_convolve12_vert_c(src, src_stride, dst, dst_stride, filter,
                                 x0_q4, x_step_q4, y0_q4, y_step_q4, w, h, bd);
    return;
  }
  assert(w == 32 || w == 16 || w == 8);
  assert(h % 4 == 0);

  const int16x8_t filter_0_7 = vld1q_s16(filter[y0_q4]);
  const int16x8_t filter_4_11 = vld1q_s16(filter[y0_q4] + 4);

  const uint16x4_t max = vdup_n_u16((1 << bd) - 1);

  src -= src_stride * (MAX_FILTER_TAP / 2 - 1);

  do {
    const int16_t *s = (const int16_t *)src;
    uint16_t *d = dst;
    int height = h;

    int16x4_t s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, sA;
    load_s16_4x11(s, src_stride, &s0, &s1, &s2, &s3, &s4, &s5, &s6, &s7, &s8,
                  &s9, &sA);
    s += 11 * src_stride;

    int16x8_t s0123[2], s1234[2], s2345[2], s3456[2], s4567[2], s5678[2],
        s6789[2], s789A[2];
    transpose_concat_s16_4x4(s0, s1, s2, s3, &s0123[0], &s0123[1]);
    transpose_concat_s16_4x4(s1, s2, s3, s4, &s1234[0], &s1234[1]);
    transpose_concat_s16_4x4(s2, s3, s4, s5, &s2345[0], &s2345[1]);
    transpose_concat_s16_4x4(s3, s4, s5, s6, &s3456[0], &s3456[1]);
    transpose_concat_s16_4x4(s4, s5, s6, s7, &s4567[0], &s4567[1]);
    transpose_concat_s16_4x4(s5, s6, s7, s8, &s5678[0], &s5678[1]);
    transpose_concat_s16_4x4(s6, s7, s8, s9, &s6789[0], &s6789[1]);
    transpose_concat_s16_4x4(s7, s8, s9, sA, &s789A[0], &s789A[1]);

    do {
      int16x4_t sB, sC, sD, sE;
      load_s16_4x4(s, src_stride, &sB, &sC, &sD, &sE);

      int16x8_t s89AB[2], s9ABC[2], sABCD[2], sBCDE[2];
      transpose_concat_s16_4x4(s8, s9, sA, sB, &s89AB[0], &s89AB[1]);
      transpose_concat_s16_4x4(s9, sA, sB, sC, &s9ABC[0], &s9ABC[1]);
      transpose_concat_s16_4x4(sA, sB, sC, sD, &sABCD[0], &sABCD[1]);
      transpose_concat_s16_4x4(sB, sC, sD, sE, &sBCDE[0], &sBCDE[1]);

      uint16x4_t d0 = highbd_convolve12_4_v(s0123, s4567, s89AB, filter_0_7,
                                            filter_4_11, max);
      uint16x4_t d1 = highbd_convolve12_4_v(s1234, s5678, s9ABC, filter_0_7,
                                            filter_4_11, max);
      uint16x4_t d2 = highbd_convolve12_4_v(s2345, s6789, sABCD, filter_0_7,
                                            filter_4_11, max);
      uint16x4_t d3 = highbd_convolve12_4_v(s3456, s789A, sBCDE, filter_0_7,
                                            filter_4_11, max);

      store_u16_4x4(d, dst_stride, d0, d1, d2, d3);

      // Prepare block for next iteration - reusing as much as possible.
      // Shuffle everything up four rows.
      s0123[0] = s4567[0];
      s0123[1] = s4567[1];
      s1234[0] = s5678[0];
      s1234[1] = s5678[1];
      s2345[0] = s6789[0];
      s2345[1] = s6789[1];
      s3456[0] = s789A[0];
      s3456[1] = s789A[1];
      s4567[0] = s89AB[0];
      s4567[1] = s89AB[1];
      s5678[0] = s9ABC[0];
      s5678[1] = s9ABC[1];
      s6789[0] = sABCD[0];
      s6789[1] = sABCD[1];
      s789A[0] = sBCDE[0];
      s789A[1] = sBCDE[1];

      s8 = sC;
      s9 = sD;
      sA = sE;

      s += 4 * src_stride;
      d += 4 * dst_stride;
      height -= 4;
    } while (height != 0);
    src += 4;
    dst += 4;
    w -= 4;
  } while (w != 0);
}

void vpx_highbd_convolve12_sve2(const uint16_t *src, ptrdiff_t src_stride,
                                uint16_t *dst, ptrdiff_t dst_stride,
                                const InterpKernel12 *filter, int x0_q4,
                                int x_step_q4, int y0_q4, int y_step_q4, int w,
                                int h, int bd) {
  // Scaling not supported by SVE2 implementation.
  if (x_step_q4 != 16 || y_step_q4 != 16) {
    vpx_highbd_convolve12_c(src, src_stride, dst, dst_stride, filter, x0_q4,
                            x_step_q4, y0_q4, y_step_q4, w, h, bd);
    return;
  }

  assert(w == 32 || w == 16 || w == 8);
  assert(h == 32 || h == 16 || h == 8);

  DECLARE_ALIGNED(32, uint16_t, im_block[BW * (BH + MAX_FILTER_TAP)]);

  const int im_stride = BW;
  // Account for the vertical pass needing MAX_FILTER_TAP / 2 - 1 lines prior
  // and MAX_FILTER_TAP / 2 lines post. (+1 to make total divisible by 4.)
  const int im_height = h + MAX_FILTER_TAP;
  const ptrdiff_t border_offset = MAX_FILTER_TAP / 2 - 1;

  // Filter starting border_offset rows up.
  vpx_highbd_convolve12_horiz_sve2(
      src - src_stride * border_offset, src_stride, im_block, im_stride, filter,
      x0_q4, x_step_q4, y0_q4, y_step_q4, w, im_height, bd);

  vpx_highbd_convolve12_vert_sve2(im_block + im_stride * border_offset,
                                  im_stride, dst, dst_stride, filter, x0_q4,
                                  x_step_q4, y0_q4, y_step_q4, w, h, bd);
}
