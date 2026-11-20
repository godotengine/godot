/*
 *  Copyright (c) 2024 The WebM project authors. All Rights Reserved.
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
#include "vpx_dsp/arm/highbd_convolve8_neon.h"
#include "vpx_dsp/arm/highbd_convolve8_sve.h"
#include "vpx_dsp/arm/mem_neon.h"
#include "vpx_dsp/arm/transpose_neon.h"
#include "vpx_dsp/arm/vpx_neon_sve_bridge.h"
#include "vpx_dsp/arm/vpx_neon_sve2_bridge.h"

// clang-format off
DECLARE_ALIGNED(16, static const uint16_t, kDotProdMergeBlockTbl[24]) = {
  // Shift left and insert new last column in transposed 4x4 block.
  1, 2, 3, 0, 5, 6, 7, 4,
  // Shift left and insert two new columns in transposed 4x4 block.
  2, 3, 0, 1, 6, 7, 4, 5,
  // Shift left and insert three new columns in transposed 4x4 block.
  3, 0, 1, 2, 7, 4, 5, 6,
};
// clang-format on

DECLARE_ALIGNED(16, static const uint16_t, kTblConv4_8[8]) = { 0, 2, 4, 6,
                                                               1, 3, 5, 7 };

static INLINE uint16x4_t highbd_convolve8_4_v(int16x8_t s_lo[2],
                                              int16x8_t s_hi[2],
                                              int16x8_t filter,
                                              uint16x4_t max) {
  int64x2_t sum01 = vpx_dotq_lane_s16(vdupq_n_s64(0), s_lo[0], filter, 0);
  sum01 = vpx_dotq_lane_s16(sum01, s_hi[0], filter, 1);

  int64x2_t sum23 = vpx_dotq_lane_s16(vdupq_n_s64(0), s_lo[1], filter, 0);
  sum23 = vpx_dotq_lane_s16(sum23, s_hi[1], filter, 1);

  int32x4_t sum0123 = vcombine_s32(vmovn_s64(sum01), vmovn_s64(sum23));

  uint16x4_t res = vqrshrun_n_s32(sum0123, FILTER_BITS);
  return vmin_u16(res, max);
}

static INLINE uint16x8_t highbd_convolve8_8_v(const int16x8_t s_lo[4],
                                              const int16x8_t s_hi[4],
                                              const int16x8_t filter,
                                              const uint16x8_t max) {
  int64x2_t sum01 = vpx_dotq_lane_s16(vdupq_n_s64(0), s_lo[0], filter, 0);
  sum01 = vpx_dotq_lane_s16(sum01, s_hi[0], filter, 1);

  int64x2_t sum23 = vpx_dotq_lane_s16(vdupq_n_s64(0), s_lo[1], filter, 0);
  sum23 = vpx_dotq_lane_s16(sum23, s_hi[1], filter, 1);

  int64x2_t sum45 = vpx_dotq_lane_s16(vdupq_n_s64(0), s_lo[2], filter, 0);
  sum45 = vpx_dotq_lane_s16(sum45, s_hi[2], filter, 1);

  int64x2_t sum67 = vpx_dotq_lane_s16(vdupq_n_s64(0), s_lo[3], filter, 0);
  sum67 = vpx_dotq_lane_s16(sum67, s_hi[3], filter, 1);

  int32x4_t sum0123 = vcombine_s32(vmovn_s64(sum01), vmovn_s64(sum23));
  int32x4_t sum4567 = vcombine_s32(vmovn_s64(sum45), vmovn_s64(sum67));

  uint16x8_t res = vcombine_u16(vqrshrun_n_s32(sum0123, FILTER_BITS),
                                vqrshrun_n_s32(sum4567, FILTER_BITS));
  return vminq_u16(res, max);
}

static INLINE void highbd_convolve8_8tap_vert_sve2(
    const uint16_t *src, ptrdiff_t src_stride, uint16_t *dst,
    ptrdiff_t dst_stride, int w, int h, const int16x8_t filter, int bd) {
  assert(w >= 4 && h >= 4);

  do {
    const uint16x4_t max = vdup_n_u16((1 << bd) - 1);
    const int16_t *s = (const int16_t *)src;
    uint16_t *d = dst;
    int height = h;

    int16x4_t s0, s1, s2, s3, s4, s5, s6;
    load_s16_4x7(s, src_stride, &s0, &s1, &s2, &s3, &s4, &s5, &s6);
    s += 7 * src_stride;

    int16x8_t s0123[2], s1234[2], s2345[2], s3456[2];
    transpose_concat_s16_4x4(s0, s1, s2, s3, &s0123[0], &s0123[1]);
    transpose_concat_s16_4x4(s1, s2, s3, s4, &s1234[0], &s1234[1]);
    transpose_concat_s16_4x4(s2, s3, s4, s5, &s2345[0], &s2345[1]);
    transpose_concat_s16_4x4(s3, s4, s5, s6, &s3456[0], &s3456[1]);

    do {
      int16x4_t s7, s8, s9, sA;

      load_s16_4x4(s, src_stride, &s7, &s8, &s9, &sA);

      int16x8_t s4567[2], s5678[2], s6789[2], s789A[2];
      transpose_concat_s16_4x4(s4, s5, s6, s7, &s4567[0], &s4567[1]);
      transpose_concat_s16_4x4(s5, s6, s7, s8, &s5678[0], &s5678[1]);
      transpose_concat_s16_4x4(s6, s7, s8, s9, &s6789[0], &s6789[1]);
      transpose_concat_s16_4x4(s7, s8, s9, sA, &s789A[0], &s789A[1]);

      uint16x4_t d0 = highbd_convolve8_4_v(s0123, s4567, filter, max);
      uint16x4_t d1 = highbd_convolve8_4_v(s1234, s5678, filter, max);
      uint16x4_t d2 = highbd_convolve8_4_v(s2345, s6789, filter, max);
      uint16x4_t d3 = highbd_convolve8_4_v(s3456, s789A, filter, max);

      store_u16_4x4(d, dst_stride, d0, d1, d2, d3);

      s0123[0] = s4567[0];
      s0123[1] = s4567[1];
      s1234[0] = s5678[0];
      s1234[1] = s5678[1];
      s2345[0] = s6789[0];
      s2345[1] = s6789[1];
      s3456[0] = s789A[0];
      s3456[1] = s789A[1];

      s4 = s8;
      s5 = s9;
      s6 = sA;

      s += 4 * src_stride;
      d += 4 * dst_stride;
      height -= 4;
    } while (height != 0);

    src += 4;
    dst += 4;
    w -= 4;
  } while (w != 0);
}

void vpx_highbd_convolve8_vert_sve2(const uint16_t *src, ptrdiff_t src_stride,
                                    uint16_t *dst, ptrdiff_t dst_stride,
                                    const InterpKernel *filter, int x0_q4,
                                    int x_step_q4, int y0_q4, int y_step_q4,
                                    int w, int h, int bd) {
  if (y_step_q4 != 16) {
    vpx_highbd_convolve8_vert_c(src, src_stride, dst, dst_stride, filter, x0_q4,
                                x_step_q4, y0_q4, y_step_q4, w, h, bd);
    return;
  }

  assert((intptr_t)dst % 4 == 0);
  assert(dst_stride % 4 == 0);
  assert(y_step_q4 == 16);

  (void)x_step_q4;
  (void)y0_q4;
  (void)y_step_q4;

  if (vpx_get_filter_taps(filter[y0_q4]) <= 4) {
    vpx_highbd_convolve8_vert_neon(src, src_stride, dst, dst_stride, filter,
                                   x0_q4, x_step_q4, y0_q4, y_step_q4, w, h,
                                   bd);
  } else {
    const int16x8_t y_filter_8tap = vld1q_s16(filter[y0_q4]);
    highbd_convolve8_8tap_vert_sve2(src - 3 * src_stride, src_stride, dst,
                                    dst_stride, w, h, y_filter_8tap, bd);
  }
}

void vpx_highbd_convolve8_avg_vert_sve2(const uint16_t *src,
                                        ptrdiff_t src_stride, uint16_t *dst,
                                        ptrdiff_t dst_stride,
                                        const InterpKernel *filter, int x0_q4,
                                        int x_step_q4, int y0_q4, int y_step_q4,
                                        int w, int h, int bd) {
  if (y_step_q4 != 16) {
    vpx_highbd_convolve8_avg_vert_c(src, src_stride, dst, dst_stride, filter,
                                    x0_q4, x_step_q4, y0_q4, y_step_q4, w, h,
                                    bd);
    return;
  }

  assert((intptr_t)dst % 4 == 0);
  assert(dst_stride % 4 == 0);

  const int16x8_t filters = vld1q_s16(filter[y0_q4]);

  src -= 3 * src_stride;

  uint16x8x3_t merge_tbl_idx = vld1q_u16_x3(kDotProdMergeBlockTbl);

  // Correct indices by the size of vector length.
  merge_tbl_idx.val[0] = vaddq_u16(
      merge_tbl_idx.val[0],
      vreinterpretq_u16_u64(vdupq_n_u64(svcnth() * 0x0001000000000000ULL)));
  merge_tbl_idx.val[1] = vaddq_u16(
      merge_tbl_idx.val[1],
      vreinterpretq_u16_u64(vdupq_n_u64(svcnth() * 0x0001000100000000ULL)));
  merge_tbl_idx.val[2] = vaddq_u16(
      merge_tbl_idx.val[2],
      vreinterpretq_u16_u64(vdupq_n_u64(svcnth() * 0x0001000100010000ULL)));

  if (w == 4) {
    const uint16x4_t max = vdup_n_u16((1 << bd) - 1);
    const int16_t *s = (const int16_t *)src;
    uint16_t *d = dst;

    int16x4_t s0, s1, s2, s3, s4, s5, s6;
    load_s16_4x7(s, src_stride, &s0, &s1, &s2, &s3, &s4, &s5, &s6);
    s += 7 * src_stride;

    int16x8_t s0123[2], s1234[2], s2345[2], s3456[2];
    transpose_concat_s16_4x4(s0, s1, s2, s3, &s0123[0], &s0123[1]);
    transpose_concat_s16_4x4(s1, s2, s3, s4, &s1234[0], &s1234[1]);
    transpose_concat_s16_4x4(s2, s3, s4, s5, &s2345[0], &s2345[1]);
    transpose_concat_s16_4x4(s3, s4, s5, s6, &s3456[0], &s3456[1]);

    do {
      int16x4_t s7, s8, s9, sA;

      load_s16_4x4(s, src_stride, &s7, &s8, &s9, &sA);

      int16x8_t s4567[2], s5678[2], s6789[2], s789A[2];
      transpose_concat_s16_4x4(s7, s8, s9, sA, &s789A[0], &s789A[1]);

      vpx_tbl2x2_s16(s3456, s789A, s4567, merge_tbl_idx.val[0]);
      vpx_tbl2x2_s16(s3456, s789A, s5678, merge_tbl_idx.val[1]);
      vpx_tbl2x2_s16(s3456, s789A, s6789, merge_tbl_idx.val[2]);

      uint16x4_t d0 = highbd_convolve8_4_v(s0123, s4567, filters, max);
      uint16x4_t d1 = highbd_convolve8_4_v(s1234, s5678, filters, max);
      uint16x4_t d2 = highbd_convolve8_4_v(s2345, s6789, filters, max);
      uint16x4_t d3 = highbd_convolve8_4_v(s3456, s789A, filters, max);

      d0 = vrhadd_u16(d0, vld1_u16(d + 0 * dst_stride));
      d1 = vrhadd_u16(d1, vld1_u16(d + 1 * dst_stride));
      d2 = vrhadd_u16(d2, vld1_u16(d + 2 * dst_stride));
      d3 = vrhadd_u16(d3, vld1_u16(d + 3 * dst_stride));

      store_u16_4x4(d, dst_stride, d0, d1, d2, d3);

      s0123[0] = s4567[0];
      s0123[1] = s4567[1];
      s1234[0] = s5678[0];
      s1234[1] = s5678[1];
      s2345[0] = s6789[0];
      s2345[1] = s6789[1];
      s3456[0] = s789A[0];
      s3456[1] = s789A[1];

      s += 4 * src_stride;
      d += 4 * dst_stride;
      h -= 4;
    } while (h != 0);
  } else {
    const uint16x8_t max = vdupq_n_u16((1 << bd) - 1);

    do {
      const int16_t *s = (const int16_t *)src;
      uint16_t *d = dst;
      int height = h;

      int16x8_t s0, s1, s2, s3, s4, s5, s6;
      load_s16_8x7(s, src_stride, &s0, &s1, &s2, &s3, &s4, &s5, &s6);
      s += 7 * src_stride;

      int16x8_t s0123[4], s1234[4], s2345[4], s3456[4];
      transpose_concat_s16_8x4(s0, s1, s2, s3, &s0123[0], &s0123[1], &s0123[2],
                               &s0123[3]);
      transpose_concat_s16_8x4(s1, s2, s3, s4, &s1234[0], &s1234[1], &s1234[2],
                               &s1234[3]);
      transpose_concat_s16_8x4(s2, s3, s4, s5, &s2345[0], &s2345[1], &s2345[2],
                               &s2345[3]);
      transpose_concat_s16_8x4(s3, s4, s5, s6, &s3456[0], &s3456[1], &s3456[2],
                               &s3456[3]);

      do {
        int16x8_t s7, s8, s9, sA;
        load_s16_8x4(s, src_stride, &s7, &s8, &s9, &sA);

        int16x8_t s4567[4], s5678[5], s6789[4], s789A[4];
        transpose_concat_s16_8x4(s7, s8, s9, sA, &s789A[0], &s789A[1],
                                 &s789A[2], &s789A[3]);

        vpx_tbl2x4_s16(s3456, s789A, s4567, merge_tbl_idx.val[0]);
        vpx_tbl2x4_s16(s3456, s789A, s5678, merge_tbl_idx.val[1]);
        vpx_tbl2x4_s16(s3456, s789A, s6789, merge_tbl_idx.val[2]);

        uint16x8_t d0 = highbd_convolve8_8_v(s0123, s4567, filters, max);
        uint16x8_t d1 = highbd_convolve8_8_v(s1234, s5678, filters, max);
        uint16x8_t d2 = highbd_convolve8_8_v(s2345, s6789, filters, max);
        uint16x8_t d3 = highbd_convolve8_8_v(s3456, s789A, filters, max);

        d0 = vrhaddq_u16(d0, vld1q_u16(d + 0 * dst_stride));
        d1 = vrhaddq_u16(d1, vld1q_u16(d + 1 * dst_stride));
        d2 = vrhaddq_u16(d2, vld1q_u16(d + 2 * dst_stride));
        d3 = vrhaddq_u16(d3, vld1q_u16(d + 3 * dst_stride));

        store_u16_8x4(d, dst_stride, d0, d1, d2, d3);

        s0123[0] = s4567[0];
        s0123[1] = s4567[1];
        s0123[2] = s4567[2];
        s0123[3] = s4567[3];
        s1234[0] = s5678[0];
        s1234[1] = s5678[1];
        s1234[2] = s5678[2];
        s1234[3] = s5678[3];
        s2345[0] = s6789[0];
        s2345[1] = s6789[1];
        s2345[2] = s6789[2];
        s2345[3] = s6789[3];
        s3456[0] = s789A[0];
        s3456[1] = s789A[1];
        s3456[2] = s789A[2];
        s3456[3] = s789A[3];

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

static INLINE void highbd_convolve_2d_4tap_sve2(
    const uint16_t *src, ptrdiff_t src_stride, uint16_t *dst,
    ptrdiff_t dst_stride, int w, int h, const int16x4_t x_filters,
    const int16x4_t y_filters, int bd) {
  const int16x8_t x_filter = vcombine_s16(x_filters, vdup_n_s16(0));

  if (w == 4) {
    const uint16x4_t max = vdup_n_u16((1 << bd) - 1);
    const int16_t *s = (const int16_t *)src;
    uint16_t *d = dst;

    int16x4_t h_s0[4], h_s1[4], h_s2[4];
    load_s16_4x4(s + 0 * src_stride, 1, &h_s0[0], &h_s0[1], &h_s0[2], &h_s0[3]);
    load_s16_4x4(s + 1 * src_stride, 1, &h_s1[0], &h_s1[1], &h_s1[2], &h_s1[3]);
    load_s16_4x4(s + 2 * src_stride, 1, &h_s2[0], &h_s2[1], &h_s2[2], &h_s2[3]);

    int16x4_t v_s0 =
        vreinterpret_s16_u16(highbd_convolve4_4_sve(h_s0, x_filter, max));
    int16x4_t v_s1 =
        vreinterpret_s16_u16(highbd_convolve4_4_sve(h_s1, x_filter, max));
    int16x4_t v_s2 =
        vreinterpret_s16_u16(highbd_convolve4_4_sve(h_s2, x_filter, max));

    s += 3 * src_stride;

    do {
      int16x4_t h_s3[4], h_s4[4], h_s5[4], h_s6[4];
      load_s16_4x4(s + 0 * src_stride, 1, &h_s3[0], &h_s3[1], &h_s3[2],
                   &h_s3[3]);
      load_s16_4x4(s + 1 * src_stride, 1, &h_s4[0], &h_s4[1], &h_s4[2],
                   &h_s4[3]);
      load_s16_4x4(s + 2 * src_stride, 1, &h_s5[0], &h_s5[1], &h_s5[2],
                   &h_s5[3]);
      load_s16_4x4(s + 3 * src_stride, 1, &h_s6[0], &h_s6[1], &h_s6[2],
                   &h_s6[3]);

      int16x4_t v_s3 =
          vreinterpret_s16_u16(highbd_convolve4_4_sve(h_s3, x_filter, max));
      int16x4_t v_s4 =
          vreinterpret_s16_u16(highbd_convolve4_4_sve(h_s4, x_filter, max));
      int16x4_t v_s5 =
          vreinterpret_s16_u16(highbd_convolve4_4_sve(h_s5, x_filter, max));
      int16x4_t v_s6 =
          vreinterpret_s16_u16(highbd_convolve4_4_sve(h_s6, x_filter, max));

      uint16x4_t d0 =
          highbd_convolve4_4_neon(v_s0, v_s1, v_s2, v_s3, y_filters, max);
      uint16x4_t d1 =
          highbd_convolve4_4_neon(v_s1, v_s2, v_s3, v_s4, y_filters, max);
      uint16x4_t d2 =
          highbd_convolve4_4_neon(v_s2, v_s3, v_s4, v_s5, y_filters, max);
      uint16x4_t d3 =
          highbd_convolve4_4_neon(v_s3, v_s4, v_s5, v_s6, y_filters, max);

      store_u16_4x4(d, dst_stride, d0, d1, d2, d3);

      v_s0 = v_s4;
      v_s1 = v_s5;
      v_s2 = v_s6;
      s += 4 * src_stride;
      d += 4 * dst_stride;
      h -= 4;
    } while (h != 0);

  } else {
    const uint16x8_t max = vdupq_n_u16((1 << bd) - 1);
    const uint16x8_t idx = vld1q_u16(kTblConv4_8);

    do {
      const int16_t *s = (const int16_t *)src;
      uint16_t *d = dst;
      int height = h;

      int16x8_t h_s0[4], h_s1[4], h_s2[4];
      load_s16_8x4(s + 0 * src_stride, 1, &h_s0[0], &h_s0[1], &h_s0[2],
                   &h_s0[3]);
      load_s16_8x4(s + 1 * src_stride, 1, &h_s1[0], &h_s1[1], &h_s1[2],
                   &h_s1[3]);
      load_s16_8x4(s + 2 * src_stride, 1, &h_s2[0], &h_s2[1], &h_s2[2],
                   &h_s2[3]);

      int16x8_t v_s0 = vreinterpretq_s16_u16(
          highbd_convolve4_8_sve(h_s0, x_filter, max, idx));
      int16x8_t v_s1 = vreinterpretq_s16_u16(
          highbd_convolve4_8_sve(h_s1, x_filter, max, idx));
      int16x8_t v_s2 = vreinterpretq_s16_u16(
          highbd_convolve4_8_sve(h_s2, x_filter, max, idx));

      s += 3 * src_stride;

      do {
        int16x8_t h_s3[4], h_s4[4], h_s5[4], h_s6[4];
        load_s16_8x4(s + 0 * src_stride, 1, &h_s3[0], &h_s3[1], &h_s3[2],
                     &h_s3[3]);
        load_s16_8x4(s + 1 * src_stride, 1, &h_s4[0], &h_s4[1], &h_s4[2],
                     &h_s4[3]);
        load_s16_8x4(s + 2 * src_stride, 1, &h_s5[0], &h_s5[1], &h_s5[2],
                     &h_s5[3]);
        load_s16_8x4(s + 3 * src_stride, 1, &h_s6[0], &h_s6[1], &h_s6[2],
                     &h_s6[3]);

        int16x8_t v_s3 = vreinterpretq_s16_u16(
            highbd_convolve4_8_sve(h_s3, x_filter, max, idx));
        int16x8_t v_s4 = vreinterpretq_s16_u16(
            highbd_convolve4_8_sve(h_s4, x_filter, max, idx));
        int16x8_t v_s5 = vreinterpretq_s16_u16(
            highbd_convolve4_8_sve(h_s5, x_filter, max, idx));
        int16x8_t v_s6 = vreinterpretq_s16_u16(
            highbd_convolve4_8_sve(h_s6, x_filter, max, idx));

        uint16x8_t d0 =
            highbd_convolve4_8_neon(v_s0, v_s1, v_s2, v_s3, y_filters, max);
        uint16x8_t d1 =
            highbd_convolve4_8_neon(v_s1, v_s2, v_s3, v_s4, y_filters, max);
        uint16x8_t d2 =
            highbd_convolve4_8_neon(v_s2, v_s3, v_s4, v_s5, y_filters, max);
        uint16x8_t d3 =
            highbd_convolve4_8_neon(v_s3, v_s4, v_s5, v_s6, y_filters, max);

        store_u16_8x4(d, dst_stride, d0, d1, d2, d3);

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

static INLINE void highbd_convolve8_2d_horiz_sve2(
    const uint16_t *src, ptrdiff_t src_stride, uint16_t *dst,
    ptrdiff_t dst_stride, const InterpKernel *filter, int x0_q4, int x_step_q4,
    int y0_q4, int y_step_q4, int w, int h, int bd) {
  assert((intptr_t)dst % 4 == 0);
  assert(dst_stride % 4 == 0);
  assert(x_step_q4 == 16);
  assert(h % 4 == 3 && h >= 7);

  (void)x_step_q4;
  (void)y0_q4;
  (void)y_step_q4;

  const int16x8_t filters = vld1q_s16(filter[x0_q4]);

  src -= 3;

  if (w == 4) {
    const uint16x4_t max = vdup_n_u16((1 << bd) - 1);
    const int16_t *s = (const int16_t *)src;
    uint16_t *d = dst;

    do {
      int16x8_t s0[4], s1[4], s2[4], s3[4];
      load_s16_8x4(s + 0 * src_stride, 1, &s0[0], &s0[1], &s0[2], &s0[3]);
      load_s16_8x4(s + 1 * src_stride, 1, &s1[0], &s1[1], &s1[2], &s1[3]);
      load_s16_8x4(s + 2 * src_stride, 1, &s2[0], &s2[1], &s2[2], &s2[3]);
      load_s16_8x4(s + 3 * src_stride, 1, &s3[0], &s3[1], &s3[2], &s3[3]);

      uint16x4_t d0 = highbd_convolve8_4(s0, filters, max);
      uint16x4_t d1 = highbd_convolve8_4(s1, filters, max);
      uint16x4_t d2 = highbd_convolve8_4(s2, filters, max);
      uint16x4_t d3 = highbd_convolve8_4(s3, filters, max);

      store_u16_4x4(d, dst_stride, d0, d1, d2, d3);

      s += 4 * src_stride;
      d += 4 * dst_stride;
      h -= 4;
    } while (h != 3);

    // Process final three rows (h % 4 == 3).
    int16x8_t s0[4], s1[4], s2[4];
    load_s16_8x4(s + 0 * src_stride, 1, &s0[0], &s0[1], &s0[2], &s0[3]);
    load_s16_8x4(s + 1 * src_stride, 1, &s1[0], &s1[1], &s1[2], &s1[3]);
    load_s16_8x4(s + 2 * src_stride, 1, &s2[0], &s2[1], &s2[2], &s2[3]);

    uint16x4_t d0 = highbd_convolve8_4(s0, filters, max);
    uint16x4_t d1 = highbd_convolve8_4(s1, filters, max);
    uint16x4_t d2 = highbd_convolve8_4(s2, filters, max);

    store_u16_4x3(d, dst_stride, d0, d1, d2);
  } else {
    const uint16x8_t max = vdupq_n_u16((1 << bd) - 1);

    do {
      const int16_t *s = (const int16_t *)src;
      uint16_t *d = dst;
      int width = w;

      do {
        int16x8_t s0[8], s1[8], s2[8], s3[8];
        load_s16_8x8(s + 0 * src_stride, 1, &s0[0], &s0[1], &s0[2], &s0[3],
                     &s0[4], &s0[5], &s0[6], &s0[7]);
        load_s16_8x8(s + 1 * src_stride, 1, &s1[0], &s1[1], &s1[2], &s1[3],
                     &s1[4], &s1[5], &s1[6], &s1[7]);
        load_s16_8x8(s + 2 * src_stride, 1, &s2[0], &s2[1], &s2[2], &s2[3],
                     &s2[4], &s2[5], &s2[6], &s2[7]);
        load_s16_8x8(s + 3 * src_stride, 1, &s3[0], &s3[1], &s3[2], &s3[3],
                     &s3[4], &s3[5], &s3[6], &s3[7]);

        uint16x8_t d0 = highbd_convolve8_8(s0, filters, max);
        uint16x8_t d1 = highbd_convolve8_8(s1, filters, max);
        uint16x8_t d2 = highbd_convolve8_8(s2, filters, max);
        uint16x8_t d3 = highbd_convolve8_8(s3, filters, max);

        store_u16_8x4(d, dst_stride, d0, d1, d2, d3);

        s += 8;
        d += 8;
        width -= 8;
      } while (width != 0);
      src += 4 * src_stride;
      dst += 4 * dst_stride;
      h -= 4;
    } while (h != 3);

    // Process final three rows (h % 4 == 3).
    const int16_t *s = (const int16_t *)src;
    uint16_t *d = dst;
    int width = w;

    do {
      int16x8_t s0[8], s1[8], s2[8];
      load_s16_8x8(s + 0 * src_stride, 1, &s0[0], &s0[1], &s0[2], &s0[3],
                   &s0[4], &s0[5], &s0[6], &s0[7]);
      load_s16_8x8(s + 1 * src_stride, 1, &s1[0], &s1[1], &s1[2], &s1[3],
                   &s1[4], &s1[5], &s1[6], &s1[7]);
      load_s16_8x8(s + 2 * src_stride, 1, &s2[0], &s2[1], &s2[2], &s2[3],
                   &s2[4], &s2[5], &s2[6], &s2[7]);

      uint16x8_t d0 = highbd_convolve8_8(s0, filters, max);
      uint16x8_t d1 = highbd_convolve8_8(s1, filters, max);
      uint16x8_t d2 = highbd_convolve8_8(s2, filters, max);

      store_u16_8x3(d, dst_stride, d0, d1, d2);

      s += 8;
      d += 8;
      width -= 8;
    } while (width != 0);
  }
}

void vpx_highbd_convolve8_sve2(const uint16_t *src, ptrdiff_t src_stride,
                               uint16_t *dst, ptrdiff_t dst_stride,
                               const InterpKernel *filter, int x0_q4,
                               int x_step_q4, int y0_q4, int y_step_q4, int w,
                               int h, int bd) {
  if (x_step_q4 != 16 || y_step_q4 != 16) {
    vpx_highbd_convolve8_c(src, src_stride, dst, dst_stride, filter, x0_q4,
                           x_step_q4, y0_q4, y_step_q4, w, h, bd);
    return;
  }

  assert(y_step_q4 == 16);
  assert(x_step_q4 == 16);

  const int horiz_filter_taps = vpx_get_filter_taps(filter[x0_q4]) <= 4 ? 4 : 8;
  const int vert_filter_taps = vpx_get_filter_taps(filter[y0_q4]) <= 4 ? 4 : 8;

  if (horiz_filter_taps == 4 || vert_filter_taps == 4) {
    const ptrdiff_t horiz_offset = horiz_filter_taps / 2 - 1;
    const ptrdiff_t vert_offset = (vert_filter_taps / 2 - 1) * src_stride;
    const int16x4_t x_filter = vld1_s16(filter[x0_q4] + 2);
    const int16x4_t y_filter = vld1_s16(filter[y0_q4] + 2);

    highbd_convolve_2d_4tap_sve2(src - horiz_offset - vert_offset, src_stride,
                                 dst, dst_stride, w, h, x_filter, y_filter, bd);
    return;
  }

  // Given our constraints: w <= 64, h <= 64, taps <= 8 we can reduce the
  // maximum buffer size to 64 * (64 + 7).
  DECLARE_ALIGNED(32, uint16_t, im_block[64 * 71]);
  const int im_stride = 64;

  // Account for the vertical phase needing SUBPEL_TAPS / 2 - 1 lines prior
  // and SUBPEL_TAPS / 2 lines post.
  const int im_height = h + SUBPEL_TAPS - 1;
  const ptrdiff_t border_offset = SUBPEL_TAPS / 2 - 1;

  highbd_convolve8_2d_horiz_sve2(src - src_stride * border_offset, src_stride,
                                 im_block, im_stride, filter, x0_q4, x_step_q4,
                                 y0_q4, y_step_q4, w, im_height, bd);

  // Step into the temporary buffer border_offset rows to get actual frame data.
  vpx_highbd_convolve8_vert_sve2(im_block + im_stride * border_offset,
                                 im_stride, dst, dst_stride, filter, x0_q4,
                                 x_step_q4, y0_q4, y_step_q4, w, h, bd);
}

void vpx_highbd_convolve8_avg_sve2(const uint16_t *src, ptrdiff_t src_stride,
                                   uint16_t *dst, ptrdiff_t dst_stride,
                                   const InterpKernel *filter, int x0_q4,
                                   int x_step_q4, int y0_q4, int y_step_q4,
                                   int w, int h, int bd) {
  if (x_step_q4 != 16 || y_step_q4 != 16) {
    vpx_highbd_convolve8_avg_c(src, src_stride, dst, dst_stride, filter, x0_q4,
                               x_step_q4, y0_q4, y_step_q4, w, h, bd);
    return;
  }

  assert(y_step_q4 == 16);
  assert(x_step_q4 == 16);

  // Given our constraints: w <= 64, h <= 64, taps <= 8 we can reduce the
  // maximum buffer size to 64 * (64 + 7).
  DECLARE_ALIGNED(32, uint16_t, im_block[64 * 71]);
  const int im_stride = 64;

  // Account for the vertical phase needing SUBPEL_TAPS / 2 - 1 lines prior
  // and SUBPEL_TAPS / 2 lines post.
  const int im_height = h + SUBPEL_TAPS - 1;
  const ptrdiff_t border_offset = SUBPEL_TAPS / 2 - 1;

  highbd_convolve8_2d_horiz_sve2(src - src_stride * border_offset, src_stride,
                                 im_block, im_stride, filter, x0_q4, x_step_q4,
                                 y0_q4, y_step_q4, w, im_height, bd);

  // Step into the temporary buffer border_offset rows to get actual frame data.
  vpx_highbd_convolve8_avg_vert_sve2(im_block + im_stride * border_offset,
                                     im_stride, dst, dst_stride, filter, x0_q4,
                                     x_step_q4, y0_q4, y_step_q4, w, h, bd);
}
