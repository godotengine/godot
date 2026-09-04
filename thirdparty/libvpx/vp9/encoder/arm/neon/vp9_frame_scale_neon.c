/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <arm_neon.h>

#include "./vp9_rtcd.h"
#include "./vpx_dsp_rtcd.h"
#include "./vpx_scale_rtcd.h"
#include "vp9/common/vp9_blockd.h"
#include "vpx_dsp/arm/mem_neon.h"
#include "vpx_dsp/arm/transpose_neon.h"
#include "vpx_dsp/arm/vpx_convolve8_neon.h"
#include "vpx_dsp/vpx_filter.h"
#include "vpx_scale/yv12config.h"

// Note: The scaling functions could write extra rows and columns in dst, which
// exceed the right and bottom boundaries of the destination frame. We rely on
// the following frame extension function to fix these rows and columns.

static INLINE void scale_plane_2_to_1_phase_0(const uint8_t *src,
                                              const int src_stride,
                                              uint8_t *dst,
                                              const int dst_stride, const int w,
                                              const int h) {
  const int max_width = (w + 15) & ~15;
  int y = h;

  assert(w && h);

  do {
    int x = max_width;
    do {
      const uint8x16x2_t s = vld2q_u8(src);
      vst1q_u8(dst, s.val[0]);
      src += 32;
      dst += 16;
      x -= 16;
    } while (x);
    src += 2 * (src_stride - max_width);
    dst += dst_stride - max_width;
  } while (--y);
}

static INLINE void scale_plane_4_to_1_phase_0(const uint8_t *src,
                                              const int src_stride,
                                              uint8_t *dst,
                                              const int dst_stride, const int w,
                                              const int h) {
  const int max_width = (w + 15) & ~15;
  int y = h;

  assert(w && h);

  do {
    int x = max_width;
    do {
      const uint8x16x4_t s = vld4q_u8(src);
      vst1q_u8(dst, s.val[0]);
      src += 64;
      dst += 16;
      x -= 16;
    } while (x);
    src += 4 * (src_stride - max_width);
    dst += dst_stride - max_width;
  } while (--y);
}

static INLINE void scale_plane_bilinear_kernel(
    const uint8x16_t in0, const uint8x16_t in1, const uint8x16_t in2,
    const uint8x16_t in3, const uint8x8_t coef0, const uint8x8_t coef1,
    uint8_t *const dst) {
  const uint16x8_t h0 = vmull_u8(vget_low_u8(in0), coef0);
  const uint16x8_t h1 = vmull_u8(vget_high_u8(in0), coef0);
  const uint16x8_t h2 = vmull_u8(vget_low_u8(in2), coef0);
  const uint16x8_t h3 = vmull_u8(vget_high_u8(in2), coef0);
  const uint16x8_t h4 = vmlal_u8(h0, vget_low_u8(in1), coef1);
  const uint16x8_t h5 = vmlal_u8(h1, vget_high_u8(in1), coef1);
  const uint16x8_t h6 = vmlal_u8(h2, vget_low_u8(in3), coef1);
  const uint16x8_t h7 = vmlal_u8(h3, vget_high_u8(in3), coef1);

  const uint8x8_t hor0 = vrshrn_n_u16(h4, 7);  // temp: 00 01 02 03 04 05 06 07
  const uint8x8_t hor1 = vrshrn_n_u16(h5, 7);  // temp: 08 09 0A 0B 0C 0D 0E 0F
  const uint8x8_t hor2 = vrshrn_n_u16(h6, 7);  // temp: 10 11 12 13 14 15 16 17
  const uint8x8_t hor3 = vrshrn_n_u16(h7, 7);  // temp: 18 19 1A 1B 1C 1D 1E 1F
  const uint16x8_t v0 = vmull_u8(hor0, coef0);
  const uint16x8_t v1 = vmull_u8(hor1, coef0);
  const uint16x8_t v2 = vmlal_u8(v0, hor2, coef1);
  const uint16x8_t v3 = vmlal_u8(v1, hor3, coef1);
  // dst: 0 1 2 3 4 5 6 7  8 9 A B C D E F
  const uint8x16_t d = vcombine_u8(vrshrn_n_u16(v2, 7), vrshrn_n_u16(v3, 7));
  vst1q_u8(dst, d);
}

static INLINE void scale_plane_2_to_1_bilinear(
    const uint8_t *const src, const int src_stride, uint8_t *dst,
    const int dst_stride, const int w, const int h, const int16_t c0,
    const int16_t c1) {
  const int max_width = (w + 15) & ~15;
  const uint8_t *src0 = src;
  const uint8_t *src1 = src + src_stride;
  const uint8x8_t coef0 = vdup_n_u8(c0);
  const uint8x8_t coef1 = vdup_n_u8(c1);
  int y = h;

  assert(w && h);

  do {
    int x = max_width;
    do {
      // 000 002 004 006 008 00A 00C 00E  010 012 014 016 018 01A 01C 01E
      // 001 003 005 007 009 00B 00D 00F  011 013 015 017 019 01B 01D 01F
      const uint8x16x2_t s0 = vld2q_u8(src0);
      // 100 102 104 106 108 10A 10C 10E  110 112 114 116 118 11A 11C 11E
      // 101 103 105 107 109 10B 10D 10F  111 113 115 117 119 11B 11D 11F
      const uint8x16x2_t s1 = vld2q_u8(src1);
      scale_plane_bilinear_kernel(s0.val[0], s0.val[1], s1.val[0], s1.val[1],
                                  coef0, coef1, dst);
      src0 += 32;
      src1 += 32;
      dst += 16;
      x -= 16;
    } while (x);
    src0 += 2 * (src_stride - max_width);
    src1 += 2 * (src_stride - max_width);
    dst += dst_stride - max_width;
  } while (--y);
}

static INLINE void scale_plane_4_to_1_bilinear(
    const uint8_t *const src, const int src_stride, uint8_t *dst,
    const int dst_stride, const int w, const int h, const int16_t c0,
    const int16_t c1) {
  const int max_width = (w + 15) & ~15;
  const uint8_t *src0 = src;
  const uint8_t *src1 = src + src_stride;
  const uint8x8_t coef0 = vdup_n_u8(c0);
  const uint8x8_t coef1 = vdup_n_u8(c1);
  int y = h;

  assert(w && h);

  do {
    int x = max_width;
    do {
      // (*) -- useless
      // 000 004 008 00C 010 014 018 01C  020 024 028 02C 030 034 038 03C
      // 001 005 009 00D 011 015 019 01D  021 025 029 02D 031 035 039 03D
      // 002 006 00A 00E 012 016 01A 01E  022 026 02A 02E 032 036 03A 03E (*)
      // 003 007 00B 00F 013 017 01B 01F  023 027 02B 02F 033 037 03B 03F (*)
      const uint8x16x4_t s0 = vld4q_u8(src0);
      // 100 104 108 10C 110 114 118 11C  120 124 128 12C 130 134 138 13C
      // 101 105 109 10D 111 115 119 11D  121 125 129 12D 131 135 139 13D
      // 102 106 10A 10E 112 116 11A 11E  122 126 12A 12E 132 136 13A 13E (*)
      // 103 107 10B 10F 113 117 11B 11F  123 127 12B 12F 133 137 13B 13F (*)
      const uint8x16x4_t s1 = vld4q_u8(src1);
      scale_plane_bilinear_kernel(s0.val[0], s0.val[1], s1.val[0], s1.val[1],
                                  coef0, coef1, dst);
      src0 += 64;
      src1 += 64;
      dst += 16;
      x -= 16;
    } while (x);
    src0 += 4 * (src_stride - max_width);
    src1 += 4 * (src_stride - max_width);
    dst += dst_stride - max_width;
  } while (--y);
}

static INLINE uint8x8_t scale_filter_bilinear(const uint8x8_t *const s,
                                              const uint8x8_t *const coef) {
  const uint16x8_t h0 = vmull_u8(s[0], coef[0]);
  const uint16x8_t h1 = vmlal_u8(h0, s[1], coef[1]);

  return vrshrn_n_u16(h1, 7);
}

static void scale_plane_2_to_1_general(const uint8_t *src, const int src_stride,
                                       uint8_t *dst, const int dst_stride,
                                       const int w, const int h,
                                       const int16_t *const coef,
                                       uint8_t *const temp_buffer) {
  const int width_hor = (w + 3) & ~3;
  const int width_ver = (w + 7) & ~7;
  const int height_hor = (2 * h + SUBPEL_TAPS - 2 + 7) & ~7;
  const int height_ver = (h + 3) & ~3;
  const int16x8_t filters = vld1q_s16(coef);
  int x, y = height_hor;
  uint8_t *t = temp_buffer;
  uint8x8_t s[14], d[4];

  assert(w && h);

  src -= (SUBPEL_TAPS / 2 - 1) * src_stride + SUBPEL_TAPS / 2 + 1;

  // horizontal 4x8
  // Note: processing 4x8 is about 20% faster than processing row by row using
  // vld4_u8().
  do {
    load_u8_8x8(src + 2, src_stride, &s[0], &s[1], &s[2], &s[3], &s[4], &s[5],
                &s[6], &s[7]);
    transpose_u8_8x8(&s[0], &s[1], &s[2], &s[3], &s[4], &s[5], &s[6], &s[7]);
    x = width_hor;

    do {
      src += 8;
      load_u8_8x8(src, src_stride, &s[6], &s[7], &s[8], &s[9], &s[10], &s[11],
                  &s[12], &s[13]);
      transpose_u8_8x8(&s[6], &s[7], &s[8], &s[9], &s[10], &s[11], &s[12],
                       &s[13]);

      d[0] = scale_filter_8(&s[0], filters);  // 00 10 20 30 40 50 60 70
      d[1] = scale_filter_8(&s[2], filters);  // 01 11 21 31 41 51 61 71
      d[2] = scale_filter_8(&s[4], filters);  // 02 12 22 32 42 52 62 72
      d[3] = scale_filter_8(&s[6], filters);  // 03 13 23 33 43 53 63 73
      // 00 01 02 03 40 41 42 43
      // 10 11 12 13 50 51 52 53
      // 20 21 22 23 60 61 62 63
      // 30 31 32 33 70 71 72 73
      transpose_u8_8x4(&d[0], &d[1], &d[2], &d[3]);
      vst1_lane_u32((uint32_t *)(t + 0 * width_hor), vreinterpret_u32_u8(d[0]),
                    0);
      vst1_lane_u32((uint32_t *)(t + 1 * width_hor), vreinterpret_u32_u8(d[1]),
                    0);
      vst1_lane_u32((uint32_t *)(t + 2 * width_hor), vreinterpret_u32_u8(d[2]),
                    0);
      vst1_lane_u32((uint32_t *)(t + 3 * width_hor), vreinterpret_u32_u8(d[3]),
                    0);
      vst1_lane_u32((uint32_t *)(t + 4 * width_hor), vreinterpret_u32_u8(d[0]),
                    1);
      vst1_lane_u32((uint32_t *)(t + 5 * width_hor), vreinterpret_u32_u8(d[1]),
                    1);
      vst1_lane_u32((uint32_t *)(t + 6 * width_hor), vreinterpret_u32_u8(d[2]),
                    1);
      vst1_lane_u32((uint32_t *)(t + 7 * width_hor), vreinterpret_u32_u8(d[3]),
                    1);

      s[0] = s[8];
      s[1] = s[9];
      s[2] = s[10];
      s[3] = s[11];
      s[4] = s[12];
      s[5] = s[13];

      t += 4;
      x -= 4;
    } while (x);
    src += 8 * src_stride - 2 * width_hor;
    t += 7 * width_hor;
    y -= 8;
  } while (y);

  // vertical 8x4
  x = width_ver;
  t = temp_buffer;
  do {
    load_u8_8x8(t, width_hor, &s[0], &s[1], &s[2], &s[3], &s[4], &s[5], &s[6],
                &s[7]);
    t += 6 * width_hor;
    y = height_ver;

    do {
      load_u8_8x8(t, width_hor, &s[6], &s[7], &s[8], &s[9], &s[10], &s[11],
                  &s[12], &s[13]);
      t += 8 * width_hor;

      d[0] = scale_filter_8(&s[0], filters);  // 00 01 02 03 04 05 06 07
      d[1] = scale_filter_8(&s[2], filters);  // 10 11 12 13 14 15 16 17
      d[2] = scale_filter_8(&s[4], filters);  // 20 21 22 23 24 25 26 27
      d[3] = scale_filter_8(&s[6], filters);  // 30 31 32 33 34 35 36 37
      vst1_u8(dst + 0 * dst_stride, d[0]);
      vst1_u8(dst + 1 * dst_stride, d[1]);
      vst1_u8(dst + 2 * dst_stride, d[2]);
      vst1_u8(dst + 3 * dst_stride, d[3]);

      s[0] = s[8];
      s[1] = s[9];
      s[2] = s[10];
      s[3] = s[11];
      s[4] = s[12];
      s[5] = s[13];

      dst += 4 * dst_stride;
      y -= 4;
    } while (y);
    t -= width_hor * (2 * height_ver + 6);
    t += 8;
    dst -= height_ver * dst_stride;
    dst += 8;
    x -= 8;
  } while (x);
}

static void scale_plane_4_to_1_general(const uint8_t *src, const int src_stride,
                                       uint8_t *dst, const int dst_stride,
                                       const int w, const int h,
                                       const int16_t *const coef,
                                       uint8_t *const temp_buffer) {
  const int width_hor = (w + 1) & ~1;
  const int width_ver = (w + 7) & ~7;
  const int height_hor = (4 * h + SUBPEL_TAPS - 2 + 7) & ~7;
  const int height_ver = (h + 1) & ~1;
  const int16x8_t filters = vld1q_s16(coef);
  int x, y = height_hor;
  uint8_t *t = temp_buffer;
  uint8x8_t s[12], d[2];

  assert(w && h);

  src -= (SUBPEL_TAPS / 2 - 1) * src_stride + SUBPEL_TAPS / 2 + 3;

  // horizontal 2x8
  // Note: processing 2x8 is about 20% faster than processing row by row using
  // vld4_u8().
  do {
    load_u8_8x8(src + 4, src_stride, &s[0], &s[1], &s[2], &s[3], &s[4], &s[5],
                &s[6], &s[7]);
    transpose_u8_4x8(&s[0], &s[1], &s[2], &s[3], s[4], s[5], s[6], s[7]);
    x = width_hor;

    do {
      uint8x8x2_t dd;
      src += 8;
      load_u8_8x8(src, src_stride, &s[4], &s[5], &s[6], &s[7], &s[8], &s[9],
                  &s[10], &s[11]);
      transpose_u8_8x8(&s[4], &s[5], &s[6], &s[7], &s[8], &s[9], &s[10],
                       &s[11]);

      d[0] = scale_filter_8(&s[0], filters);  // 00 10 20 30 40 50 60 70
      d[1] = scale_filter_8(&s[4], filters);  // 01 11 21 31 41 51 61 71
      // dd.val[0]: 00 01 20 21 40 41 60 61
      // dd.val[1]: 10 11 30 31 50 51 70 71
      dd = vtrn_u8(d[0], d[1]);
      vst1_lane_u16((uint16_t *)(t + 0 * width_hor),
                    vreinterpret_u16_u8(dd.val[0]), 0);
      vst1_lane_u16((uint16_t *)(t + 1 * width_hor),
                    vreinterpret_u16_u8(dd.val[1]), 0);
      vst1_lane_u16((uint16_t *)(t + 2 * width_hor),
                    vreinterpret_u16_u8(dd.val[0]), 1);
      vst1_lane_u16((uint16_t *)(t + 3 * width_hor),
                    vreinterpret_u16_u8(dd.val[1]), 1);
      vst1_lane_u16((uint16_t *)(t + 4 * width_hor),
                    vreinterpret_u16_u8(dd.val[0]), 2);
      vst1_lane_u16((uint16_t *)(t + 5 * width_hor),
                    vreinterpret_u16_u8(dd.val[1]), 2);
      vst1_lane_u16((uint16_t *)(t + 6 * width_hor),
                    vreinterpret_u16_u8(dd.val[0]), 3);
      vst1_lane_u16((uint16_t *)(t + 7 * width_hor),
                    vreinterpret_u16_u8(dd.val[1]), 3);

      s[0] = s[8];
      s[1] = s[9];
      s[2] = s[10];
      s[3] = s[11];

      t += 2;
      x -= 2;
    } while (x);
    src += 8 * src_stride - 4 * width_hor;
    t += 7 * width_hor;
    y -= 8;
  } while (y);

  // vertical 8x2
  x = width_ver;
  t = temp_buffer;
  do {
    load_u8_8x4(t, width_hor, &s[0], &s[1], &s[2], &s[3]);
    t += 4 * width_hor;
    y = height_ver;

    do {
      load_u8_8x8(t, width_hor, &s[4], &s[5], &s[6], &s[7], &s[8], &s[9],
                  &s[10], &s[11]);
      t += 8 * width_hor;

      d[0] = scale_filter_8(&s[0], filters);  // 00 01 02 03 04 05 06 07
      d[1] = scale_filter_8(&s[4], filters);  // 10 11 12 13 14 15 16 17
      vst1_u8(dst + 0 * dst_stride, d[0]);
      vst1_u8(dst + 1 * dst_stride, d[1]);

      s[0] = s[8];
      s[1] = s[9];
      s[2] = s[10];
      s[3] = s[11];

      dst += 2 * dst_stride;
      y -= 2;
    } while (y);
    t -= width_hor * (4 * height_ver + 4);
    t += 8;
    dst -= height_ver * dst_stride;
    dst += 8;
    x -= 8;
  } while (x);
}

// Notes for 4 to 3 scaling:
//
// 1. 6 rows are calculated in each horizontal inner loop, so width_hor must be
// multiple of 6, and no less than w.
//
// 2. 8 rows are calculated in each vertical inner loop, so width_ver must be
// multiple of 8, and no less than w.
//
// 3. 8 columns are calculated in each horizontal inner loop for further
// vertical scaling, so height_hor must be multiple of 8, and no less than
// 4 * h / 3.
//
// 4. 6 columns are calculated in each vertical inner loop, so height_ver must
// be multiple of 6, and no less than h.
//
// 5. The physical location of the last row of the 4 to 3 scaled frame is
// decided by phase_scaler, and are always less than 1 pixel below the last row
// of the original image.

static void scale_plane_4_to_3_bilinear(const uint8_t *src,
                                        const int src_stride, uint8_t *dst,
                                        const int dst_stride, const int w,
                                        const int h, const int phase_scaler,
                                        uint8_t *const temp_buffer) {
  static const int step_q4 = 16 * 4 / 3;
  const int width_hor = (w + 5) - ((w + 5) % 6);
  const int stride_hor = width_hor + 2;  // store 2 extra pixels
  const int width_ver = (w + 7) & ~7;
  // We only need 1 extra row below because there are only 2 bilinear
  // coefficients.
  const int height_hor = (4 * h / 3 + 1 + 7) & ~7;
  const int height_ver = (h + 5) - ((h + 5) % 6);
  int x, y = height_hor;
  uint8_t *t = temp_buffer;
  uint8x8_t s[9], d[8], c[6];

  assert(w && h);

  c[0] = vdup_n_u8((uint8_t)vp9_filter_kernels[BILINEAR][phase_scaler][3]);
  c[1] = vdup_n_u8((uint8_t)vp9_filter_kernels[BILINEAR][phase_scaler][4]);
  c[2] = vdup_n_u8(
      (uint8_t)vp9_filter_kernels[BILINEAR][(phase_scaler + 1 * step_q4) &
                                            SUBPEL_MASK][3]);
  c[3] = vdup_n_u8(
      (uint8_t)vp9_filter_kernels[BILINEAR][(phase_scaler + 1 * step_q4) &
                                            SUBPEL_MASK][4]);
  c[4] = vdup_n_u8(
      (uint8_t)vp9_filter_kernels[BILINEAR][(phase_scaler + 2 * step_q4) &
                                            SUBPEL_MASK][3]);
  c[5] = vdup_n_u8(
      (uint8_t)vp9_filter_kernels[BILINEAR][(phase_scaler + 2 * step_q4) &
                                            SUBPEL_MASK][4]);

  d[6] = vdup_n_u8(0);
  d[7] = vdup_n_u8(0);

  // horizontal 6x8
  do {
    load_u8_8x8(src, src_stride, &s[0], &s[1], &s[2], &s[3], &s[4], &s[5],
                &s[6], &s[7]);
    src += 1;
    transpose_u8_8x8(&s[0], &s[1], &s[2], &s[3], &s[4], &s[5], &s[6], &s[7]);
    x = width_hor;

    do {
      load_u8_8x8(src, src_stride, &s[1], &s[2], &s[3], &s[4], &s[5], &s[6],
                  &s[7], &s[8]);
      src += 8;
      transpose_u8_8x8(&s[1], &s[2], &s[3], &s[4], &s[5], &s[6], &s[7], &s[8]);

      // 00 10 20 30 40 50 60 70
      // 01 11 21 31 41 51 61 71
      // 02 12 22 32 42 52 62 72
      // 03 13 23 33 43 53 63 73
      // 04 14 24 34 44 54 64 74
      // 05 15 25 35 45 55 65 75
      d[0] = scale_filter_bilinear(&s[0], &c[0]);
      d[1] =
          scale_filter_bilinear(&s[(phase_scaler + 1 * step_q4) >> 4], &c[2]);
      d[2] =
          scale_filter_bilinear(&s[(phase_scaler + 2 * step_q4) >> 4], &c[4]);
      d[3] = scale_filter_bilinear(&s[4], &c[0]);
      d[4] = scale_filter_bilinear(&s[4 + ((phase_scaler + 1 * step_q4) >> 4)],
                                   &c[2]);
      d[5] = scale_filter_bilinear(&s[4 + ((phase_scaler + 2 * step_q4) >> 4)],
                                   &c[4]);

      // 00 01 02 03 04 05 xx xx
      // 10 11 12 13 14 15 xx xx
      // 20 21 22 23 24 25 xx xx
      // 30 31 32 33 34 35 xx xx
      // 40 41 42 43 44 45 xx xx
      // 50 51 52 53 54 55 xx xx
      // 60 61 62 63 64 65 xx xx
      // 70 71 72 73 74 75 xx xx
      transpose_u8_8x8(&d[0], &d[1], &d[2], &d[3], &d[4], &d[5], &d[6], &d[7]);
      // store 2 extra pixels
      vst1_u8(t + 0 * stride_hor, d[0]);
      vst1_u8(t + 1 * stride_hor, d[1]);
      vst1_u8(t + 2 * stride_hor, d[2]);
      vst1_u8(t + 3 * stride_hor, d[3]);
      vst1_u8(t + 4 * stride_hor, d[4]);
      vst1_u8(t + 5 * stride_hor, d[5]);
      vst1_u8(t + 6 * stride_hor, d[6]);
      vst1_u8(t + 7 * stride_hor, d[7]);

      s[0] = s[8];

      t += 6;
      x -= 6;
    } while (x);
    src += 8 * src_stride - 4 * width_hor / 3 - 1;
    t += 7 * stride_hor + 2;
    y -= 8;
  } while (y);

  // vertical 8x6
  x = width_ver;
  t = temp_buffer;
  do {
    load_u8_8x8(t, stride_hor, &s[0], &s[1], &s[2], &s[3], &s[4], &s[5], &s[6],
                &s[7]);
    t += stride_hor;
    y = height_ver;

    do {
      load_u8_8x8(t, stride_hor, &s[1], &s[2], &s[3], &s[4], &s[5], &s[6],
                  &s[7], &s[8]);
      t += 8 * stride_hor;

      d[0] = scale_filter_bilinear(&s[0], &c[0]);
      d[1] =
          scale_filter_bilinear(&s[(phase_scaler + 1 * step_q4) >> 4], &c[2]);
      d[2] =
          scale_filter_bilinear(&s[(phase_scaler + 2 * step_q4) >> 4], &c[4]);
      d[3] = scale_filter_bilinear(&s[4], &c[0]);
      d[4] = scale_filter_bilinear(&s[4 + ((phase_scaler + 1 * step_q4) >> 4)],
                                   &c[2]);
      d[5] = scale_filter_bilinear(&s[4 + ((phase_scaler + 2 * step_q4) >> 4)],
                                   &c[4]);
      vst1_u8(dst + 0 * dst_stride, d[0]);
      vst1_u8(dst + 1 * dst_stride, d[1]);
      vst1_u8(dst + 2 * dst_stride, d[2]);
      vst1_u8(dst + 3 * dst_stride, d[3]);
      vst1_u8(dst + 4 * dst_stride, d[4]);
      vst1_u8(dst + 5 * dst_stride, d[5]);

      s[0] = s[8];

      dst += 6 * dst_stride;
      y -= 6;
    } while (y);
    t -= stride_hor * (4 * height_ver / 3 + 1);
    t += 8;
    dst -= height_ver * dst_stride;
    dst += 8;
    x -= 8;
  } while (x);
}

static void scale_plane_4_to_3_general(const uint8_t *src, const int src_stride,
                                       uint8_t *dst, const int dst_stride,
                                       const int w, const int h,
                                       const InterpKernel *const coef,
                                       const int phase_scaler,
                                       uint8_t *const temp_buffer) {
  static const int step_q4 = 16 * 4 / 3;
  const int width_hor = (w + 5) - ((w + 5) % 6);
  const int stride_hor = width_hor + 2;  // store 2 extra pixels
  const int width_ver = (w + 7) & ~7;
  // We need (SUBPEL_TAPS - 1) extra rows: (SUBPEL_TAPS / 2 - 1) extra rows
  // above and (SUBPEL_TAPS / 2) extra rows below.
  const int height_hor = (4 * h / 3 + SUBPEL_TAPS - 1 + 7) & ~7;
  const int height_ver = (h + 5) - ((h + 5) % 6);
  const int16x8_t filters0 =
      vld1q_s16(coef[(phase_scaler + 0 * step_q4) & SUBPEL_MASK]);
  const int16x8_t filters1 =
      vld1q_s16(coef[(phase_scaler + 1 * step_q4) & SUBPEL_MASK]);
  const int16x8_t filters2 =
      vld1q_s16(coef[(phase_scaler + 2 * step_q4) & SUBPEL_MASK]);
  int x, y = height_hor;
  uint8_t *t = temp_buffer;
  uint8x8_t s[15], d[8];

  assert(w && h);

  src -= (SUBPEL_TAPS / 2 - 1) * src_stride + SUBPEL_TAPS / 2;
  d[6] = vdup_n_u8(0);
  d[7] = vdup_n_u8(0);

  // horizontal 6x8
  do {
    load_u8_8x8(src + 1, src_stride, &s[0], &s[1], &s[2], &s[3], &s[4], &s[5],
                &s[6], &s[7]);
    transpose_u8_8x8(&s[0], &s[1], &s[2], &s[3], &s[4], &s[5], &s[6], &s[7]);
    x = width_hor;

    do {
      src += 8;
      load_u8_8x8(src, src_stride, &s[7], &s[8], &s[9], &s[10], &s[11], &s[12],
                  &s[13], &s[14]);
      transpose_u8_8x8(&s[7], &s[8], &s[9], &s[10], &s[11], &s[12], &s[13],
                       &s[14]);

      // 00 10 20 30 40 50 60 70
      // 01 11 21 31 41 51 61 71
      // 02 12 22 32 42 52 62 72
      // 03 13 23 33 43 53 63 73
      // 04 14 24 34 44 54 64 74
      // 05 15 25 35 45 55 65 75
      d[0] = scale_filter_8(&s[0], filters0);
      d[1] = scale_filter_8(&s[(phase_scaler + 1 * step_q4) >> 4], filters1);
      d[2] = scale_filter_8(&s[(phase_scaler + 2 * step_q4) >> 4], filters2);
      d[3] = scale_filter_8(&s[4], filters0);
      d[4] =
          scale_filter_8(&s[4 + ((phase_scaler + 1 * step_q4) >> 4)], filters1);
      d[5] =
          scale_filter_8(&s[4 + ((phase_scaler + 2 * step_q4) >> 4)], filters2);

      // 00 01 02 03 04 05 xx xx
      // 10 11 12 13 14 15 xx xx
      // 20 21 22 23 24 25 xx xx
      // 30 31 32 33 34 35 xx xx
      // 40 41 42 43 44 45 xx xx
      // 50 51 52 53 54 55 xx xx
      // 60 61 62 63 64 65 xx xx
      // 70 71 72 73 74 75 xx xx
      transpose_u8_8x8(&d[0], &d[1], &d[2], &d[3], &d[4], &d[5], &d[6], &d[7]);
      // store 2 extra pixels
      vst1_u8(t + 0 * stride_hor, d[0]);
      vst1_u8(t + 1 * stride_hor, d[1]);
      vst1_u8(t + 2 * stride_hor, d[2]);
      vst1_u8(t + 3 * stride_hor, d[3]);
      vst1_u8(t + 4 * stride_hor, d[4]);
      vst1_u8(t + 5 * stride_hor, d[5]);
      vst1_u8(t + 6 * stride_hor, d[6]);
      vst1_u8(t + 7 * stride_hor, d[7]);

      s[0] = s[8];
      s[1] = s[9];
      s[2] = s[10];
      s[3] = s[11];
      s[4] = s[12];
      s[5] = s[13];
      s[6] = s[14];

      t += 6;
      x -= 6;
    } while (x);
    src += 8 * src_stride - 4 * width_hor / 3;
    t += 7 * stride_hor + 2;
    y -= 8;
  } while (y);

  // vertical 8x6
  x = width_ver;
  t = temp_buffer;
  do {
    load_u8_8x8(t, stride_hor, &s[0], &s[1], &s[2], &s[3], &s[4], &s[5], &s[6],
                &s[7]);
    t += 7 * stride_hor;
    y = height_ver;

    do {
      load_u8_8x8(t, stride_hor, &s[7], &s[8], &s[9], &s[10], &s[11], &s[12],
                  &s[13], &s[14]);
      t += 8 * stride_hor;

      d[0] = scale_filter_8(&s[0], filters0);
      d[1] = scale_filter_8(&s[(phase_scaler + 1 * step_q4) >> 4], filters1);
      d[2] = scale_filter_8(&s[(phase_scaler + 2 * step_q4) >> 4], filters2);
      d[3] = scale_filter_8(&s[4], filters0);
      d[4] =
          scale_filter_8(&s[4 + ((phase_scaler + 1 * step_q4) >> 4)], filters1);
      d[5] =
          scale_filter_8(&s[4 + ((phase_scaler + 2 * step_q4) >> 4)], filters2);
      vst1_u8(dst + 0 * dst_stride, d[0]);
      vst1_u8(dst + 1 * dst_stride, d[1]);
      vst1_u8(dst + 2 * dst_stride, d[2]);
      vst1_u8(dst + 3 * dst_stride, d[3]);
      vst1_u8(dst + 4 * dst_stride, d[4]);
      vst1_u8(dst + 5 * dst_stride, d[5]);

      s[0] = s[8];
      s[1] = s[9];
      s[2] = s[10];
      s[3] = s[11];
      s[4] = s[12];
      s[5] = s[13];
      s[6] = s[14];

      dst += 6 * dst_stride;
      y -= 6;
    } while (y);
    t -= stride_hor * (4 * height_ver / 3 + 7);
    t += 8;
    dst -= height_ver * dst_stride;
    dst += 8;
    x -= 8;
  } while (x);
}

void vp9_scale_and_extend_frame_neon(const YV12_BUFFER_CONFIG *src,
                                     YV12_BUFFER_CONFIG *dst,
                                     INTERP_FILTER filter_type,
                                     int phase_scaler) {
  const int src_w = src->y_crop_width;
  const int src_h = src->y_crop_height;
  const int dst_w = dst->y_crop_width;
  const int dst_h = dst->y_crop_height;
  const int dst_uv_w = dst->uv_crop_width;
  const int dst_uv_h = dst->uv_crop_height;
  int scaled = 0;

  // phase_scaler is usually 0 or 8.
  assert(phase_scaler >= 0 && phase_scaler < 16);

  if (2 * dst_w == src_w && 2 * dst_h == src_h) {
    // 2 to 1
    scaled = 1;
    if (phase_scaler == 0) {
      scale_plane_2_to_1_phase_0(src->y_buffer, src->y_stride, dst->y_buffer,
                                 dst->y_stride, dst_w, dst_h);
      scale_plane_2_to_1_phase_0(src->u_buffer, src->uv_stride, dst->u_buffer,
                                 dst->uv_stride, dst_uv_w, dst_uv_h);
      scale_plane_2_to_1_phase_0(src->v_buffer, src->uv_stride, dst->v_buffer,
                                 dst->uv_stride, dst_uv_w, dst_uv_h);
    } else if (filter_type == BILINEAR) {
      const int16_t c0 = vp9_filter_kernels[BILINEAR][phase_scaler][3];
      const int16_t c1 = vp9_filter_kernels[BILINEAR][phase_scaler][4];
      scale_plane_2_to_1_bilinear(src->y_buffer, src->y_stride, dst->y_buffer,
                                  dst->y_stride, dst_w, dst_h, c0, c1);
      scale_plane_2_to_1_bilinear(src->u_buffer, src->uv_stride, dst->u_buffer,
                                  dst->uv_stride, dst_uv_w, dst_uv_h, c0, c1);
      scale_plane_2_to_1_bilinear(src->v_buffer, src->uv_stride, dst->v_buffer,
                                  dst->uv_stride, dst_uv_w, dst_uv_h, c0, c1);
    } else {
      const int buffer_stride = (dst_w + 3) & ~3;
      const int buffer_height = (2 * dst_h + SUBPEL_TAPS - 2 + 7) & ~7;
      uint8_t *const temp_buffer =
          (uint8_t *)malloc(buffer_stride * buffer_height);
      if (temp_buffer) {
        scale_plane_2_to_1_general(
            src->y_buffer, src->y_stride, dst->y_buffer, dst->y_stride, dst_w,
            dst_h, vp9_filter_kernels[filter_type][phase_scaler], temp_buffer);
        scale_plane_2_to_1_general(
            src->u_buffer, src->uv_stride, dst->u_buffer, dst->uv_stride,
            dst_uv_w, dst_uv_h, vp9_filter_kernels[filter_type][phase_scaler],
            temp_buffer);
        scale_plane_2_to_1_general(
            src->v_buffer, src->uv_stride, dst->v_buffer, dst->uv_stride,
            dst_uv_w, dst_uv_h, vp9_filter_kernels[filter_type][phase_scaler],
            temp_buffer);
        free(temp_buffer);
      } else {
        scaled = 0;
      }
    }
  } else if (4 * dst_w == src_w && 4 * dst_h == src_h) {
    // 4 to 1
    scaled = 1;
    if (phase_scaler == 0) {
      scale_plane_4_to_1_phase_0(src->y_buffer, src->y_stride, dst->y_buffer,
                                 dst->y_stride, dst_w, dst_h);
      scale_plane_4_to_1_phase_0(src->u_buffer, src->uv_stride, dst->u_buffer,
                                 dst->uv_stride, dst_uv_w, dst_uv_h);
      scale_plane_4_to_1_phase_0(src->v_buffer, src->uv_stride, dst->v_buffer,
                                 dst->uv_stride, dst_uv_w, dst_uv_h);
    } else if (filter_type == BILINEAR) {
      const int16_t c0 = vp9_filter_kernels[BILINEAR][phase_scaler][3];
      const int16_t c1 = vp9_filter_kernels[BILINEAR][phase_scaler][4];
      scale_plane_4_to_1_bilinear(src->y_buffer, src->y_stride, dst->y_buffer,
                                  dst->y_stride, dst_w, dst_h, c0, c1);
      scale_plane_4_to_1_bilinear(src->u_buffer, src->uv_stride, dst->u_buffer,
                                  dst->uv_stride, dst_uv_w, dst_uv_h, c0, c1);
      scale_plane_4_to_1_bilinear(src->v_buffer, src->uv_stride, dst->v_buffer,
                                  dst->uv_stride, dst_uv_w, dst_uv_h, c0, c1);
    } else {
      const int buffer_stride = (dst_w + 1) & ~1;
      const int buffer_height = (4 * dst_h + SUBPEL_TAPS - 2 + 7) & ~7;
      uint8_t *const temp_buffer =
          (uint8_t *)malloc(buffer_stride * buffer_height);
      if (temp_buffer) {
        scale_plane_4_to_1_general(
            src->y_buffer, src->y_stride, dst->y_buffer, dst->y_stride, dst_w,
            dst_h, vp9_filter_kernels[filter_type][phase_scaler], temp_buffer);
        scale_plane_4_to_1_general(
            src->u_buffer, src->uv_stride, dst->u_buffer, dst->uv_stride,
            dst_uv_w, dst_uv_h, vp9_filter_kernels[filter_type][phase_scaler],
            temp_buffer);
        scale_plane_4_to_1_general(
            src->v_buffer, src->uv_stride, dst->v_buffer, dst->uv_stride,
            dst_uv_w, dst_uv_h, vp9_filter_kernels[filter_type][phase_scaler],
            temp_buffer);
        free(temp_buffer);
      } else {
        scaled = 0;
      }
    }
  } else if (4 * dst_w == 3 * src_w && 4 * dst_h == 3 * src_h) {
    // 4 to 3
    const int buffer_stride = (dst_w + 5) - ((dst_w + 5) % 6) + 2;
    const int buffer_height = (4 * dst_h / 3 + SUBPEL_TAPS - 1 + 7) & ~7;
    uint8_t *const temp_buffer =
        (uint8_t *)malloc(buffer_stride * buffer_height);
    if (temp_buffer) {
      scaled = 1;
      if (filter_type == BILINEAR) {
        scale_plane_4_to_3_bilinear(src->y_buffer, src->y_stride, dst->y_buffer,
                                    dst->y_stride, dst_w, dst_h, phase_scaler,
                                    temp_buffer);
        scale_plane_4_to_3_bilinear(src->u_buffer, src->uv_stride,
                                    dst->u_buffer, dst->uv_stride, dst_uv_w,
                                    dst_uv_h, phase_scaler, temp_buffer);
        scale_plane_4_to_3_bilinear(src->v_buffer, src->uv_stride,
                                    dst->v_buffer, dst->uv_stride, dst_uv_w,
                                    dst_uv_h, phase_scaler, temp_buffer);
      } else {
        scale_plane_4_to_3_general(
            src->y_buffer, src->y_stride, dst->y_buffer, dst->y_stride, dst_w,
            dst_h, vp9_filter_kernels[filter_type], phase_scaler, temp_buffer);
        scale_plane_4_to_3_general(src->u_buffer, src->uv_stride, dst->u_buffer,
                                   dst->uv_stride, dst_uv_w, dst_uv_h,
                                   vp9_filter_kernels[filter_type],
                                   phase_scaler, temp_buffer);
        scale_plane_4_to_3_general(src->v_buffer, src->uv_stride, dst->v_buffer,
                                   dst->uv_stride, dst_uv_w, dst_uv_h,
                                   vp9_filter_kernels[filter_type],
                                   phase_scaler, temp_buffer);
      }
      free(temp_buffer);
    }
  }

  if (scaled) {
    vpx_extend_frame_borders(dst);
  } else {
    // Call c version for all other scaling ratios.
    vp9_scale_and_extend_frame_c(src, dst, filter_type, phase_scaler);
  }
}
