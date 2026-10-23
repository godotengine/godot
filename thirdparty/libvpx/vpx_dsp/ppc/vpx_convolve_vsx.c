/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#include <assert.h>
#include <string.h>

#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/ppc/types_vsx.h"
#include "vpx_dsp/vpx_filter.h"

// TODO(lu_zero): unroll
static VPX_FORCE_INLINE void copy_w16(const uint8_t *src, ptrdiff_t src_stride,
                                      uint8_t *dst, ptrdiff_t dst_stride,
                                      int32_t h) {
  int i;

  for (i = h; i--;) {
    vec_vsx_st(vec_vsx_ld(0, src), 0, dst);
    src += src_stride;
    dst += dst_stride;
  }
}

static VPX_FORCE_INLINE void copy_w32(const uint8_t *src, ptrdiff_t src_stride,
                                      uint8_t *dst, ptrdiff_t dst_stride,
                                      int32_t h) {
  int i;

  for (i = h; i--;) {
    vec_vsx_st(vec_vsx_ld(0, src), 0, dst);
    vec_vsx_st(vec_vsx_ld(16, src), 16, dst);
    src += src_stride;
    dst += dst_stride;
  }
}

static VPX_FORCE_INLINE void copy_w64(const uint8_t *src, ptrdiff_t src_stride,
                                      uint8_t *dst, ptrdiff_t dst_stride,
                                      int32_t h) {
  int i;

  for (i = h; i--;) {
    vec_vsx_st(vec_vsx_ld(0, src), 0, dst);
    vec_vsx_st(vec_vsx_ld(16, src), 16, dst);
    vec_vsx_st(vec_vsx_ld(32, src), 32, dst);
    vec_vsx_st(vec_vsx_ld(48, src), 48, dst);
    src += src_stride;
    dst += dst_stride;
  }
}

void vpx_convolve_copy_vsx(const uint8_t *src, ptrdiff_t src_stride,
                           uint8_t *dst, ptrdiff_t dst_stride,
                           const InterpKernel *filter, int x0_q4, int x_step_q4,
                           int y0_q4, int32_t y_step_q4, int32_t w, int32_t h) {
  (void)filter;
  (void)x0_q4;
  (void)x_step_q4;
  (void)y0_q4;
  (void)y_step_q4;

  switch (w) {
    case 16: {
      copy_w16(src, src_stride, dst, dst_stride, h);
      break;
    }
    case 32: {
      copy_w32(src, src_stride, dst, dst_stride, h);
      break;
    }
    case 64: {
      copy_w64(src, src_stride, dst, dst_stride, h);
      break;
    }
    default: {
      int i;
      for (i = h; i--;) {
        memcpy(dst, src, w);
        src += src_stride;
        dst += dst_stride;
      }
      break;
    }
  }
}

static VPX_FORCE_INLINE void avg_w16(const uint8_t *src, ptrdiff_t src_stride,
                                     uint8_t *dst, ptrdiff_t dst_stride,
                                     int32_t h) {
  int i;

  for (i = h; i--;) {
    const uint8x16_t v = vec_avg(vec_vsx_ld(0, src), vec_vsx_ld(0, dst));
    vec_vsx_st(v, 0, dst);
    src += src_stride;
    dst += dst_stride;
  }
}

static VPX_FORCE_INLINE void avg_w32(const uint8_t *src, ptrdiff_t src_stride,
                                     uint8_t *dst, ptrdiff_t dst_stride,
                                     int32_t h) {
  int i;

  for (i = h; i--;) {
    const uint8x16_t v0 = vec_avg(vec_vsx_ld(0, src), vec_vsx_ld(0, dst));
    const uint8x16_t v1 = vec_avg(vec_vsx_ld(16, src), vec_vsx_ld(16, dst));
    vec_vsx_st(v0, 0, dst);
    vec_vsx_st(v1, 16, dst);
    src += src_stride;
    dst += dst_stride;
  }
}

static VPX_FORCE_INLINE void avg_w64(const uint8_t *src, ptrdiff_t src_stride,
                                     uint8_t *dst, ptrdiff_t dst_stride,
                                     int32_t h) {
  int i;

  for (i = h; i--;) {
    const uint8x16_t v0 = vec_avg(vec_vsx_ld(0, src), vec_vsx_ld(0, dst));
    const uint8x16_t v1 = vec_avg(vec_vsx_ld(16, src), vec_vsx_ld(16, dst));
    const uint8x16_t v2 = vec_avg(vec_vsx_ld(32, src), vec_vsx_ld(32, dst));
    const uint8x16_t v3 = vec_avg(vec_vsx_ld(48, src), vec_vsx_ld(48, dst));
    vec_vsx_st(v0, 0, dst);
    vec_vsx_st(v1, 16, dst);
    vec_vsx_st(v2, 32, dst);
    vec_vsx_st(v3, 48, dst);
    src += src_stride;
    dst += dst_stride;
  }
}

void vpx_convolve_avg_vsx(const uint8_t *src, ptrdiff_t src_stride,
                          uint8_t *dst, ptrdiff_t dst_stride,
                          const InterpKernel *filter, int x0_q4, int x_step_q4,
                          int y0_q4, int32_t y_step_q4, int32_t w, int32_t h) {
  switch (w) {
    case 16: {
      avg_w16(src, src_stride, dst, dst_stride, h);
      break;
    }
    case 32: {
      avg_w32(src, src_stride, dst, dst_stride, h);
      break;
    }
    case 64: {
      avg_w64(src, src_stride, dst, dst_stride, h);
      break;
    }
    default: {
      vpx_convolve_avg_c(src, src_stride, dst, dst_stride, filter, x0_q4,
                         x_step_q4, y0_q4, y_step_q4, w, h);
      break;
    }
  }
}

static VPX_FORCE_INLINE void convolve_line(uint8_t *dst, const int16x8_t s,
                                           const int16x8_t f) {
  const int32x4_t sum = vec_msum(s, f, vec_splat_s32(0));
  const int32x4_t bias =
      vec_sl(vec_splat_s32(1), vec_splat_u32(FILTER_BITS - 1));
  const int32x4_t avg = vec_sr(vec_sums(sum, bias), vec_splat_u32(FILTER_BITS));
  const uint8x16_t v = vec_splat(
      vec_packsu(vec_pack(avg, vec_splat_s32(0)), vec_splat_s16(0)), 3);
  vec_ste(v, 0, dst);
}

static VPX_FORCE_INLINE void convolve_line_h(uint8_t *dst,
                                             const uint8_t *const src_x,
                                             const int16_t *const x_filter) {
  const int16x8_t s = unpack_to_s16_h(vec_vsx_ld(0, src_x));
  const int16x8_t f = vec_vsx_ld(0, x_filter);

  convolve_line(dst, s, f);
}

// TODO(lu_zero): Implement 8x8 and bigger block special cases
static VPX_FORCE_INLINE void convolve_horiz(const uint8_t *src,
                                            ptrdiff_t src_stride, uint8_t *dst,
                                            ptrdiff_t dst_stride,
                                            const InterpKernel *x_filters,
                                            int x0_q4, int x_step_q4, int w,
                                            int h) {
  int x, y;
  src -= SUBPEL_TAPS / 2 - 1;

  for (y = 0; y < h; ++y) {
    int x_q4 = x0_q4;
    for (x = 0; x < w; ++x) {
      convolve_line_h(dst + x, &src[x_q4 >> SUBPEL_BITS],
                      x_filters[x_q4 & SUBPEL_MASK]);
      x_q4 += x_step_q4;
    }
    src += src_stride;
    dst += dst_stride;
  }
}

static VPX_FORCE_INLINE void convolve_avg_horiz(
    const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst,
    ptrdiff_t dst_stride, const InterpKernel *x_filters, int x0_q4,
    int x_step_q4, int w, int h) {
  int x, y;
  src -= SUBPEL_TAPS / 2 - 1;

  for (y = 0; y < h; ++y) {
    int x_q4 = x0_q4;
    for (x = 0; x < w; ++x) {
      uint8_t v;
      convolve_line_h(&v, &src[x_q4 >> SUBPEL_BITS],
                      x_filters[x_q4 & SUBPEL_MASK]);
      dst[x] = ROUND_POWER_OF_TWO(dst[x] + v, 1);
      x_q4 += x_step_q4;
    }
    src += src_stride;
    dst += dst_stride;
  }
}

static uint8x16_t transpose_line_u8_8x8(uint8x16_t a, uint8x16_t b,
                                        uint8x16_t c, uint8x16_t d,
                                        uint8x16_t e, uint8x16_t f,
                                        uint8x16_t g, uint8x16_t h) {
  uint16x8_t ab = (uint16x8_t)vec_mergeh(a, b);
  uint16x8_t cd = (uint16x8_t)vec_mergeh(c, d);
  uint16x8_t ef = (uint16x8_t)vec_mergeh(e, f);
  uint16x8_t gh = (uint16x8_t)vec_mergeh(g, h);

  uint32x4_t abcd = (uint32x4_t)vec_mergeh(ab, cd);
  uint32x4_t efgh = (uint32x4_t)vec_mergeh(ef, gh);

  return (uint8x16_t)vec_mergeh(abcd, efgh);
}

static VPX_FORCE_INLINE void convolve_line_v(uint8_t *dst,
                                             const uint8_t *const src_y,
                                             ptrdiff_t src_stride,
                                             const int16_t *const y_filter) {
  uint8x16_t s0 = vec_vsx_ld(0, src_y + 0 * src_stride);
  uint8x16_t s1 = vec_vsx_ld(0, src_y + 1 * src_stride);
  uint8x16_t s2 = vec_vsx_ld(0, src_y + 2 * src_stride);
  uint8x16_t s3 = vec_vsx_ld(0, src_y + 3 * src_stride);
  uint8x16_t s4 = vec_vsx_ld(0, src_y + 4 * src_stride);
  uint8x16_t s5 = vec_vsx_ld(0, src_y + 5 * src_stride);
  uint8x16_t s6 = vec_vsx_ld(0, src_y + 6 * src_stride);
  uint8x16_t s7 = vec_vsx_ld(0, src_y + 7 * src_stride);
  const int16x8_t f = vec_vsx_ld(0, y_filter);
  uint8_t buf[16];
  const uint8x16_t s = transpose_line_u8_8x8(s0, s1, s2, s3, s4, s5, s6, s7);

  vec_vsx_st(s, 0, buf);

  convolve_line(dst, unpack_to_s16_h(s), f);
}

static VPX_FORCE_INLINE void convolve_vert(const uint8_t *src,
                                           ptrdiff_t src_stride, uint8_t *dst,
                                           ptrdiff_t dst_stride,
                                           const InterpKernel *y_filters,
                                           int y0_q4, int y_step_q4, int w,
                                           int h) {
  int x, y;
  src -= src_stride * (SUBPEL_TAPS / 2 - 1);

  for (x = 0; x < w; ++x) {
    int y_q4 = y0_q4;
    for (y = 0; y < h; ++y) {
      convolve_line_v(dst + y * dst_stride,
                      &src[(y_q4 >> SUBPEL_BITS) * src_stride], src_stride,
                      y_filters[y_q4 & SUBPEL_MASK]);
      y_q4 += y_step_q4;
    }
    ++src;
    ++dst;
  }
}

static VPX_FORCE_INLINE void convolve_avg_vert(
    const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst,
    ptrdiff_t dst_stride, const InterpKernel *y_filters, int y0_q4,
    int y_step_q4, int w, int h) {
  int x, y;
  src -= src_stride * (SUBPEL_TAPS / 2 - 1);

  for (x = 0; x < w; ++x) {
    int y_q4 = y0_q4;
    for (y = 0; y < h; ++y) {
      uint8_t v;
      convolve_line_v(&v, &src[(y_q4 >> SUBPEL_BITS) * src_stride], src_stride,
                      y_filters[y_q4 & SUBPEL_MASK]);
      dst[y * dst_stride] = ROUND_POWER_OF_TWO(dst[y * dst_stride] + v, 1);
      y_q4 += y_step_q4;
    }
    ++src;
    ++dst;
  }
}

static VPX_FORCE_INLINE void convolve(const uint8_t *src, ptrdiff_t src_stride,
                                      uint8_t *dst, ptrdiff_t dst_stride,
                                      const InterpKernel *const filter,
                                      int x0_q4, int x_step_q4, int y0_q4,
                                      int y_step_q4, int w, int h) {
  // Note: Fixed size intermediate buffer, temp, places limits on parameters.
  // 2d filtering proceeds in 2 steps:
  //   (1) Interpolate horizontally into an intermediate buffer, temp.
  //   (2) Interpolate temp vertically to derive the sub-pixel result.
  // Deriving the maximum number of rows in the temp buffer (135):
  // --Smallest scaling factor is x1/2 ==> y_step_q4 = 32 (Normative).
  // --Largest block size is 64x64 pixels.
  // --64 rows in the downscaled frame span a distance of (64 - 1) * 32 in the
  //   original frame (in 1/16th pixel units).
  // --Must round-up because block may be located at sub-pixel position.
  // --Require an additional SUBPEL_TAPS rows for the 8-tap filter tails.
  // --((64 - 1) * 32 + 15) >> 4 + 8 = 135.
  DECLARE_ALIGNED(16, uint8_t, temp[64 * 135]);
  const int intermediate_height =
      (((h - 1) * y_step_q4 + y0_q4) >> SUBPEL_BITS) + SUBPEL_TAPS;

  assert(w <= 64);
  assert(h <= 64);
  assert(y_step_q4 <= 32);
  assert(x_step_q4 <= 32);

  convolve_horiz(src - src_stride * (SUBPEL_TAPS / 2 - 1), src_stride, temp, 64,
                 filter, x0_q4, x_step_q4, w, intermediate_height);
  convolve_vert(temp + 64 * (SUBPEL_TAPS / 2 - 1), 64, dst, dst_stride, filter,
                y0_q4, y_step_q4, w, h);
}

void vpx_convolve8_horiz_vsx(const uint8_t *src, ptrdiff_t src_stride,
                             uint8_t *dst, ptrdiff_t dst_stride,
                             const InterpKernel *filter, int x0_q4,
                             int x_step_q4, int y0_q4, int y_step_q4, int w,
                             int h) {
  (void)y0_q4;
  (void)y_step_q4;

  convolve_horiz(src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4, w,
                 h);
}

void vpx_convolve8_avg_horiz_vsx(const uint8_t *src, ptrdiff_t src_stride,
                                 uint8_t *dst, ptrdiff_t dst_stride,
                                 const InterpKernel *filter, int x0_q4,
                                 int x_step_q4, int y0_q4, int y_step_q4, int w,
                                 int h) {
  (void)y0_q4;
  (void)y_step_q4;

  convolve_avg_horiz(src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4,
                     w, h);
}

void vpx_convolve8_vert_vsx(const uint8_t *src, ptrdiff_t src_stride,
                            uint8_t *dst, ptrdiff_t dst_stride,
                            const InterpKernel *filter, int x0_q4,
                            int x_step_q4, int y0_q4, int y_step_q4, int w,
                            int h) {
  (void)x0_q4;
  (void)x_step_q4;

  convolve_vert(src, src_stride, dst, dst_stride, filter, y0_q4, y_step_q4, w,
                h);
}

void vpx_convolve8_avg_vert_vsx(const uint8_t *src, ptrdiff_t src_stride,
                                uint8_t *dst, ptrdiff_t dst_stride,
                                const InterpKernel *filter, int x0_q4,
                                int x_step_q4, int y0_q4, int y_step_q4, int w,
                                int h) {
  (void)x0_q4;
  (void)x_step_q4;

  convolve_avg_vert(src, src_stride, dst, dst_stride, filter, y0_q4, y_step_q4,
                    w, h);
}

void vpx_convolve8_vsx(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst,
                       ptrdiff_t dst_stride, const InterpKernel *filter,
                       int x0_q4, int x_step_q4, int y0_q4, int y_step_q4,
                       int w, int h) {
  convolve(src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4, y0_q4,
           y_step_q4, w, h);
}

void vpx_convolve8_avg_vsx(const uint8_t *src, ptrdiff_t src_stride,
                           uint8_t *dst, ptrdiff_t dst_stride,
                           const InterpKernel *filter, int x0_q4, int x_step_q4,
                           int y0_q4, int y_step_q4, int w, int h) {
  // Fixed size intermediate buffer places limits on parameters.
  DECLARE_ALIGNED(16, uint8_t, temp[64 * 64]);
  assert(w <= 64);
  assert(h <= 64);

  vpx_convolve8_vsx(src, src_stride, temp, 64, filter, x0_q4, x_step_q4, y0_q4,
                    y_step_q4, w, h);
  vpx_convolve_avg_vsx(temp, 64, dst, dst_stride, NULL, 0, 0, 0, 0, w, h);
}
