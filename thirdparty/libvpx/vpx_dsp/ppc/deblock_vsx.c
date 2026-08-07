/*
 *  Copyright (c) 2018 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/ppc/types_vsx.h"

extern const int16_t vpx_rv[];

static const uint8x16_t load_merge = { 0x00, 0x02, 0x04, 0x06, 0x08, 0x0A,
                                       0x0C, 0x0E, 0x18, 0x19, 0x1A, 0x1B,
                                       0x1C, 0x1D, 0x1E, 0x1F };

static const uint8x16_t st8_perm = { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05,
                                     0x06, 0x07, 0x18, 0x19, 0x1A, 0x1B,
                                     0x1C, 0x1D, 0x1E, 0x1F };

static INLINE uint8x16_t apply_filter(uint8x16_t ctx[4], uint8x16_t v,
                                      uint8x16_t filter) {
  const uint8x16_t k1 = vec_avg(ctx[0], ctx[1]);
  const uint8x16_t k2 = vec_avg(ctx[3], ctx[2]);
  const uint8x16_t k3 = vec_avg(k1, k2);
  const uint8x16_t f_a = vec_max(vec_absd(v, ctx[0]), vec_absd(v, ctx[1]));
  const uint8x16_t f_b = vec_max(vec_absd(v, ctx[2]), vec_absd(v, ctx[3]));
  const bool8x16_t mask = vec_cmplt(vec_max(f_a, f_b), filter);
  return vec_sel(v, vec_avg(k3, v), mask);
}

static INLINE void vert_ctx(uint8x16_t ctx[4], int col, uint8_t *src,
                            int stride) {
  ctx[0] = vec_vsx_ld(col - 2 * stride, src);
  ctx[1] = vec_vsx_ld(col - stride, src);
  ctx[2] = vec_vsx_ld(col + stride, src);
  ctx[3] = vec_vsx_ld(col + 2 * stride, src);
}

static INLINE void horz_ctx(uint8x16_t ctx[4], uint8x16_t left_ctx,
                            uint8x16_t v, uint8x16_t right_ctx) {
  static const uint8x16_t l2_perm = { 0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13,
                                      0x14, 0x15, 0x16, 0x17, 0x18, 0x19,
                                      0x1A, 0x1B, 0x1C, 0x1D };

  static const uint8x16_t l1_perm = { 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14,
                                      0x15, 0x16, 0x17, 0x18, 0x19, 0x1A,
                                      0x1B, 0x1C, 0x1D, 0x1E };

  static const uint8x16_t r1_perm = { 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
                                      0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C,
                                      0x0D, 0x0E, 0x0F, 0x10 };

  static const uint8x16_t r2_perm = { 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                                      0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D,
                                      0x0E, 0x0F, 0x10, 0x11 };
  ctx[0] = vec_perm(left_ctx, v, l2_perm);
  ctx[1] = vec_perm(left_ctx, v, l1_perm);
  ctx[2] = vec_perm(v, right_ctx, r1_perm);
  ctx[3] = vec_perm(v, right_ctx, r2_perm);
}
void vpx_post_proc_down_and_across_mb_row_vsx(unsigned char *src_ptr,
                                              unsigned char *dst_ptr,
                                              int src_pixels_per_line,
                                              int dst_pixels_per_line, int cols,
                                              unsigned char *f, int size) {
  int row, col;
  uint8x16_t ctx[4], out, v, left_ctx;

  for (row = 0; row < size; row++) {
    for (col = 0; col < cols - 8; col += 16) {
      const uint8x16_t filter = vec_vsx_ld(col, f);
      v = vec_vsx_ld(col, src_ptr);
      vert_ctx(ctx, col, src_ptr, src_pixels_per_line);
      vec_vsx_st(apply_filter(ctx, v, filter), col, dst_ptr);
    }

    if (col != cols) {
      const uint8x16_t filter = vec_vsx_ld(col, f);
      v = vec_vsx_ld(col, src_ptr);
      vert_ctx(ctx, col, src_ptr, src_pixels_per_line);
      out = apply_filter(ctx, v, filter);
      vec_vsx_st(vec_perm(out, v, st8_perm), col, dst_ptr);
    }

    /* now post_proc_across */
    left_ctx = vec_splats(dst_ptr[0]);
    v = vec_vsx_ld(0, dst_ptr);
    for (col = 0; col < cols - 8; col += 16) {
      const uint8x16_t filter = vec_vsx_ld(col, f);
      const uint8x16_t right_ctx = (col + 16 == cols)
                                       ? vec_splats(dst_ptr[cols - 1])
                                       : vec_vsx_ld(col, dst_ptr + 16);
      horz_ctx(ctx, left_ctx, v, right_ctx);
      vec_vsx_st(apply_filter(ctx, v, filter), col, dst_ptr);
      left_ctx = v;
      v = right_ctx;
    }

    if (col != cols) {
      const uint8x16_t filter = vec_vsx_ld(col, f);
      const uint8x16_t right_ctx = vec_splats(dst_ptr[cols - 1]);
      horz_ctx(ctx, left_ctx, v, right_ctx);
      out = apply_filter(ctx, v, filter);
      vec_vsx_st(vec_perm(out, v, st8_perm), col, dst_ptr);
    }

    src_ptr += src_pixels_per_line;
    dst_ptr += dst_pixels_per_line;
  }
}

// C: s[c + 7]
static INLINE int16x8_t next7l_s16(uint8x16_t c) {
  static const uint8x16_t next7_perm = {
    0x07, 0x10, 0x08, 0x11, 0x09, 0x12, 0x0A, 0x13,
    0x0B, 0x14, 0x0C, 0x15, 0x0D, 0x16, 0x0E, 0x17,
  };
  return (int16x8_t)vec_perm(c, vec_zeros_u8, next7_perm);
}

// Slide across window and add.
static INLINE int16x8_t slide_sum_s16(int16x8_t x) {
  // x = A B C D E F G H
  //
  // 0 A B C D E F G
  const int16x8_t sum1 = vec_add(x, vec_slo(x, vec_splats((int8_t)(2 << 3))));
  // 0 0 A B C D E F
  const int16x8_t sum2 = vec_add(vec_slo(x, vec_splats((int8_t)(4 << 3))),
                                 // 0 0 0 A B C D E
                                 vec_slo(x, vec_splats((int8_t)(6 << 3))));
  // 0 0 0 0 A B C D
  const int16x8_t sum3 = vec_add(vec_slo(x, vec_splats((int8_t)(8 << 3))),
                                 // 0 0 0 0 0 A B C
                                 vec_slo(x, vec_splats((int8_t)(10 << 3))));
  // 0 0 0 0 0 0 A B
  const int16x8_t sum4 = vec_add(vec_slo(x, vec_splats((int8_t)(12 << 3))),
                                 // 0 0 0 0 0 0 0 A
                                 vec_slo(x, vec_splats((int8_t)(14 << 3))));
  return vec_add(vec_add(sum1, sum2), vec_add(sum3, sum4));
}

// Slide across window and add.
static INLINE int32x4_t slide_sumsq_s32(int32x4_t xsq_even, int32x4_t xsq_odd) {
  //   0 A C E
  // + 0 B D F
  int32x4_t sumsq_1 = vec_add(vec_slo(xsq_even, vec_splats((int8_t)(4 << 3))),
                              vec_slo(xsq_odd, vec_splats((int8_t)(4 << 3))));
  //   0 0 A C
  // + 0 0 B D
  int32x4_t sumsq_2 = vec_add(vec_slo(xsq_even, vec_splats((int8_t)(8 << 3))),
                              vec_slo(xsq_odd, vec_splats((int8_t)(8 << 3))));
  //   0 0 0 A
  // + 0 0 0 B
  int32x4_t sumsq_3 = vec_add(vec_slo(xsq_even, vec_splats((int8_t)(12 << 3))),
                              vec_slo(xsq_odd, vec_splats((int8_t)(12 << 3))));
  sumsq_1 = vec_add(sumsq_1, xsq_even);
  sumsq_2 = vec_add(sumsq_2, sumsq_3);
  return vec_add(sumsq_1, sumsq_2);
}

// C: (b + sum + val) >> 4
static INLINE int16x8_t filter_s16(int16x8_t b, int16x8_t sum, int16x8_t val) {
  return vec_sra(vec_add(vec_add(b, sum), val), vec_splats((uint16_t)4));
}

// C: sumsq * 15 - sum * sum
static INLINE bool16x8_t mask_s16(int32x4_t sumsq_even, int32x4_t sumsq_odd,
                                  int16x8_t sum, int32x4_t lim) {
  static const uint8x16_t mask_merge = { 0x00, 0x01, 0x10, 0x11, 0x04, 0x05,
                                         0x14, 0x15, 0x08, 0x09, 0x18, 0x19,
                                         0x0C, 0x0D, 0x1C, 0x1D };
  const int32x4_t sumsq_odd_scaled =
      vec_mul(sumsq_odd, vec_splats((int32_t)15));
  const int32x4_t sumsq_even_scaled =
      vec_mul(sumsq_even, vec_splats((int32_t)15));
  const int32x4_t thres_odd = vec_sub(sumsq_odd_scaled, vec_mulo(sum, sum));
  const int32x4_t thres_even = vec_sub(sumsq_even_scaled, vec_mule(sum, sum));

  const bool32x4_t mask_odd = vec_cmplt(thres_odd, lim);
  const bool32x4_t mask_even = vec_cmplt(thres_even, lim);
  return vec_perm((bool16x8_t)mask_even, (bool16x8_t)mask_odd, mask_merge);
}

void vpx_mbpost_proc_across_ip_vsx(unsigned char *src, int pitch, int rows,
                                   int cols, int flimit) {
  int row, col;
  const int32x4_t lim = vec_splats(flimit);

  // 8 columns are processed at a time.
  assert(cols % 8 == 0);

  for (row = 0; row < rows; row++) {
    // The sum is signed and requires at most 13 bits.
    // (8 bits + sign) * 15 (4 bits)
    int16x8_t sum;
    // The sum of squares requires at most 20 bits.
    // (16 bits + sign) * 15 (4 bits)
    int32x4_t sumsq_even, sumsq_odd;

    // Fill left context with first col.
    int16x8_t left_ctx = vec_splats((int16_t)src[0]);
    int16_t s = src[0] * 9;
    int32_t ssq = src[0] * src[0] * 9 + 16;

    // Fill the next 6 columns of the sliding window with cols 2 to 7.
    for (col = 1; col <= 6; ++col) {
      s += src[col];
      ssq += src[col] * src[col];
    }
    // Set this sum to every element in the window.
    sum = vec_splats(s);
    sumsq_even = vec_splats(ssq);
    sumsq_odd = vec_splats(ssq);

    for (col = 0; col < cols; col += 8) {
      bool16x8_t mask;
      int16x8_t filtered, masked;
      uint8x16_t out;

      const uint8x16_t val = vec_vsx_ld(0, src + col);
      const int16x8_t val_high = unpack_to_s16_h(val);

      // C: s[c + 7]
      const int16x8_t right_ctx = (col + 8 == cols)
                                      ? vec_splats((int16_t)src[col + 7])
                                      : next7l_s16(val);

      // C: x = s[c + 7] - s[c - 8];
      const int16x8_t x = vec_sub(right_ctx, left_ctx);
      const int32x4_t xsq_even =
          vec_sub(vec_mule(right_ctx, right_ctx), vec_mule(left_ctx, left_ctx));
      const int32x4_t xsq_odd =
          vec_sub(vec_mulo(right_ctx, right_ctx), vec_mulo(left_ctx, left_ctx));

      const int32x4_t sumsq_tmp = slide_sumsq_s32(xsq_even, xsq_odd);
      // A C E G
      // 0 B D F
      // 0 A C E
      // 0 0 B D
      // 0 0 A C
      // 0 0 0 B
      // 0 0 0 A
      sumsq_even = vec_add(sumsq_even, sumsq_tmp);
      // B D F G
      // A C E G
      // 0 B D F
      // 0 A C E
      // 0 0 B D
      // 0 0 A C
      // 0 0 0 B
      // 0 0 0 A
      sumsq_odd = vec_add(sumsq_odd, vec_add(sumsq_tmp, xsq_odd));

      sum = vec_add(sum, slide_sum_s16(x));

      // C: (8 + sum + s[c]) >> 4
      filtered = filter_s16(vec_splats((int16_t)8), sum, val_high);
      // C: sumsq * 15 - sum * sum
      mask = mask_s16(sumsq_even, sumsq_odd, sum, lim);
      masked = vec_sel(val_high, filtered, mask);

      out = vec_perm((uint8x16_t)masked, vec_vsx_ld(0, src + col), load_merge);
      vec_vsx_st(out, 0, src + col);

      // Update window sum and square sum
      sum = vec_splat(sum, 7);
      sumsq_even = vec_splat(sumsq_odd, 3);
      sumsq_odd = vec_splat(sumsq_odd, 3);

      // C: s[c - 8] (for next iteration)
      left_ctx = val_high;
    }
    src += pitch;
  }
}

void vpx_mbpost_proc_down_vsx(uint8_t *dst, int pitch, int rows, int cols,
                              int flimit) {
  int col, row, i;
  int16x8_t window[16];
  const int32x4_t lim = vec_splats(flimit);

  // 8 columns are processed at a time.
  assert(cols % 8 == 0);
  // If rows is less than 8 the bottom border extension fails.
  assert(rows >= 8);

  for (col = 0; col < cols; col += 8) {
    // The sum is signed and requires at most 13 bits.
    // (8 bits + sign) * 15 (4 bits)
    int16x8_t r1, sum;
    // The sum of squares requires at most 20 bits.
    // (16 bits + sign) * 15 (4 bits)
    int32x4_t sumsq_even, sumsq_odd;

    r1 = unpack_to_s16_h(vec_vsx_ld(0, dst));
    // Fill sliding window with first row.
    for (i = 0; i <= 8; i++) {
      window[i] = r1;
    }
    // First 9 rows of the sliding window are the same.
    // sum = r1 * 9
    sum = vec_mladd(r1, vec_splats((int16_t)9), vec_zeros_s16);

    // sumsq = r1 * r1 * 9
    sumsq_even = vec_mule(sum, r1);
    sumsq_odd = vec_mulo(sum, r1);

    // Fill the next 6 rows of the sliding window with rows 2 to 7.
    for (i = 1; i <= 6; ++i) {
      const int16x8_t next_row = unpack_to_s16_h(vec_vsx_ld(i * pitch, dst));
      window[i + 8] = next_row;
      sum = vec_add(sum, next_row);
      sumsq_odd = vec_add(sumsq_odd, vec_mulo(next_row, next_row));
      sumsq_even = vec_add(sumsq_even, vec_mule(next_row, next_row));
    }

    for (row = 0; row < rows; row++) {
      int32x4_t d15_even, d15_odd, d0_even, d0_odd;
      bool16x8_t mask;
      int16x8_t filtered, masked;
      uint8x16_t out;

      const int16x8_t rv = vec_vsx_ld(0, vpx_rv + (row & 127));

      // Move the sliding window
      if (row + 7 < rows) {
        window[15] = unpack_to_s16_h(vec_vsx_ld((row + 7) * pitch, dst));
      } else {
        window[15] = window[14];
      }

      // C: sum += s[7 * pitch] - s[-8 * pitch];
      sum = vec_add(sum, vec_sub(window[15], window[0]));

      // C: sumsq += s[7 * pitch] * s[7 * pitch] - s[-8 * pitch] * s[-8 *
      // pitch];
      // Optimization Note: Caching a squared-window for odd and even is
      // slower than just repeating the multiplies.
      d15_odd = vec_mulo(window[15], window[15]);
      d15_even = vec_mule(window[15], window[15]);
      d0_odd = vec_mulo(window[0], window[0]);
      d0_even = vec_mule(window[0], window[0]);
      sumsq_odd = vec_add(sumsq_odd, vec_sub(d15_odd, d0_odd));
      sumsq_even = vec_add(sumsq_even, vec_sub(d15_even, d0_even));

      // C: (vpx_rv[(r & 127) + (c & 7)] + sum + s[0]) >> 4
      filtered = filter_s16(rv, sum, window[8]);

      // C: sumsq * 15 - sum * sum
      mask = mask_s16(sumsq_even, sumsq_odd, sum, lim);
      masked = vec_sel(window[8], filtered, mask);

      // TODO(ltrudeau) If cols % 16 == 0, we could just process 16 per
      // iteration
      out = vec_perm((uint8x16_t)masked, vec_vsx_ld(0, dst + row * pitch),
                     load_merge);
      vec_vsx_st(out, 0, dst + row * pitch);

      // Optimization Note: Turns out that the following loop is faster than
      // using pointers to manage the sliding window.
      for (i = 1; i < 16; i++) {
        window[i - 1] = window[i];
      }
    }
    dst += 8;
  }
}
