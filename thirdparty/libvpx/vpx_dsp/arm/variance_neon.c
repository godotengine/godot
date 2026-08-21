/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <arm_neon.h>
#include <assert.h>

#include "./vpx_dsp_rtcd.h"
#include "./vpx_config.h"

#include "vpx/vpx_integer.h"
#include "vpx_dsp/arm/mem_neon.h"
#include "vpx_dsp/arm/sum_neon.h"
#include "vpx_ports/mem.h"

// Process a block of width 4 two rows at a time.
static INLINE void variance_4xh_neon(const uint8_t *src_ptr, int src_stride,
                                     const uint8_t *ref_ptr, int ref_stride,
                                     int h, uint32_t *sse, int *sum) {
  int16x8_t sum_s16 = vdupq_n_s16(0);
  int32x4_t sse_s32 = vdupq_n_s32(0);
  int i = h;

  // Number of rows we can process before 'sum_s16' overflows:
  // 32767 / 255 ~= 128, but we use an 8-wide accumulator; so 256 4-wide rows.
  assert(h <= 256);

  do {
    const uint8x8_t s = load_unaligned_u8(src_ptr, src_stride);
    const uint8x8_t r = load_unaligned_u8(ref_ptr, ref_stride);
    const int16x8_t diff = vreinterpretq_s16_u16(vsubl_u8(s, r));

    sum_s16 = vaddq_s16(sum_s16, diff);

    sse_s32 = vmlal_s16(sse_s32, vget_low_s16(diff), vget_low_s16(diff));
    sse_s32 = vmlal_s16(sse_s32, vget_high_s16(diff), vget_high_s16(diff));

    src_ptr += 2 * src_stride;
    ref_ptr += 2 * ref_stride;
    i -= 2;
  } while (i != 0);

  *sum = horizontal_add_int16x8(sum_s16);
  *sse = (uint32_t)horizontal_add_int32x4(sse_s32);
}

// Process a block of width 8 one row at a time.
static INLINE void variance_8xh_neon(const uint8_t *src_ptr, int src_stride,
                                     const uint8_t *ref_ptr, int ref_stride,
                                     int h, uint32_t *sse, int *sum) {
  int16x8_t sum_s16 = vdupq_n_s16(0);
  int32x4_t sse_s32[2] = { vdupq_n_s32(0), vdupq_n_s32(0) };
  int i = h;

  // Number of rows we can process before 'sum_s16' overflows:
  // 32767 / 255 ~= 128
  assert(h <= 128);

  do {
    const uint8x8_t s = vld1_u8(src_ptr);
    const uint8x8_t r = vld1_u8(ref_ptr);
    const int16x8_t diff = vreinterpretq_s16_u16(vsubl_u8(s, r));

    sum_s16 = vaddq_s16(sum_s16, diff);

    sse_s32[0] = vmlal_s16(sse_s32[0], vget_low_s16(diff), vget_low_s16(diff));
    sse_s32[1] =
        vmlal_s16(sse_s32[1], vget_high_s16(diff), vget_high_s16(diff));

    src_ptr += src_stride;
    ref_ptr += ref_stride;
  } while (--i != 0);

  *sum = horizontal_add_int16x8(sum_s16);
  *sse = (uint32_t)horizontal_add_int32x4(vaddq_s32(sse_s32[0], sse_s32[1]));
}

// Process a block of width 16 one row at a time.
static INLINE void variance_16xh_neon(const uint8_t *src_ptr, int src_stride,
                                      const uint8_t *ref_ptr, int ref_stride,
                                      int h, uint32_t *sse, int *sum) {
  int16x8_t sum_s16[2] = { vdupq_n_s16(0), vdupq_n_s16(0) };
  int32x4_t sse_s32[2] = { vdupq_n_s32(0), vdupq_n_s32(0) };
  int i = h;

  // Number of rows we can process before 'sum_s16' accumulators overflow:
  // 32767 / 255 ~= 128, so 128 16-wide rows.
  assert(h <= 128);

  do {
    const uint8x16_t s = vld1q_u8(src_ptr);
    const uint8x16_t r = vld1q_u8(ref_ptr);

    const int16x8_t diff_l =
        vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(s), vget_low_u8(r)));
    const int16x8_t diff_h =
        vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(s), vget_high_u8(r)));

    sum_s16[0] = vaddq_s16(sum_s16[0], diff_l);
    sum_s16[1] = vaddq_s16(sum_s16[1], diff_h);

    sse_s32[0] =
        vmlal_s16(sse_s32[0], vget_low_s16(diff_l), vget_low_s16(diff_l));
    sse_s32[1] =
        vmlal_s16(sse_s32[1], vget_high_s16(diff_l), vget_high_s16(diff_l));
    sse_s32[0] =
        vmlal_s16(sse_s32[0], vget_low_s16(diff_h), vget_low_s16(diff_h));
    sse_s32[1] =
        vmlal_s16(sse_s32[1], vget_high_s16(diff_h), vget_high_s16(diff_h));

    src_ptr += src_stride;
    ref_ptr += ref_stride;
  } while (--i != 0);

  *sum = horizontal_add_int16x8(vaddq_s16(sum_s16[0], sum_s16[1]));
  *sse = (uint32_t)horizontal_add_int32x4(vaddq_s32(sse_s32[0], sse_s32[1]));
}

// Process a block of any size where the width is divisible by 16.
static INLINE void variance_large_neon(const uint8_t *src_ptr, int src_stride,
                                       const uint8_t *ref_ptr, int ref_stride,
                                       int w, int h, int h_limit,
                                       unsigned int *sse, int *sum) {
  int32x4_t sum_s32 = vdupq_n_s32(0);
  int32x4_t sse_s32[2] = { vdupq_n_s32(0), vdupq_n_s32(0) };

  // 'h_limit' is the number of 'w'-width rows we can process before our 16-bit
  // accumulator overflows. After hitting this limit we accumulate into 32-bit
  // elements.
  int h_tmp = h > h_limit ? h_limit : h;

  int i = 0;
  do {
    int16x8_t sum_s16[2] = { vdupq_n_s16(0), vdupq_n_s16(0) };
    do {
      int j = 0;
      do {
        const uint8x16_t s = vld1q_u8(src_ptr + j);
        const uint8x16_t r = vld1q_u8(ref_ptr + j);

        const int16x8_t diff_l =
            vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(s), vget_low_u8(r)));
        const int16x8_t diff_h =
            vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(s), vget_high_u8(r)));

        sum_s16[0] = vaddq_s16(sum_s16[0], diff_l);
        sum_s16[1] = vaddq_s16(sum_s16[1], diff_h);

        sse_s32[0] =
            vmlal_s16(sse_s32[0], vget_low_s16(diff_l), vget_low_s16(diff_l));
        sse_s32[1] =
            vmlal_s16(sse_s32[1], vget_high_s16(diff_l), vget_high_s16(diff_l));
        sse_s32[0] =
            vmlal_s16(sse_s32[0], vget_low_s16(diff_h), vget_low_s16(diff_h));
        sse_s32[1] =
            vmlal_s16(sse_s32[1], vget_high_s16(diff_h), vget_high_s16(diff_h));

        j += 16;
      } while (j < w);

      src_ptr += src_stride;
      ref_ptr += ref_stride;
      i++;
    } while (i < h_tmp);

    sum_s32 = vpadalq_s16(sum_s32, sum_s16[0]);
    sum_s32 = vpadalq_s16(sum_s32, sum_s16[1]);

    h_tmp += h_limit;
  } while (i < h);

  *sum = horizontal_add_int32x4(sum_s32);
  *sse = (uint32_t)horizontal_add_int32x4(vaddq_s32(sse_s32[0], sse_s32[1]));
}

static INLINE void variance_32xh_neon(const uint8_t *src, int src_stride,
                                      const uint8_t *ref, int ref_stride, int h,
                                      uint32_t *sse, int *sum) {
  variance_large_neon(src, src_stride, ref, ref_stride, 32, h, 64, sse, sum);
}

static INLINE void variance_64xh_neon(const uint8_t *src, int src_stride,
                                      const uint8_t *ref, int ref_stride, int h,
                                      uint32_t *sse, int *sum) {
  variance_large_neon(src, src_stride, ref, ref_stride, 64, h, 32, sse, sum);
}

void vpx_get8x8var_neon(const uint8_t *src_ptr, int src_stride,
                        const uint8_t *ref_ptr, int ref_stride,
                        unsigned int *sse, int *sum) {
  variance_8xh_neon(src_ptr, src_stride, ref_ptr, ref_stride, 8, sse, sum);
}

void vpx_get16x16var_neon(const uint8_t *src_ptr, int src_stride,
                          const uint8_t *ref_ptr, int ref_stride,
                          unsigned int *sse, int *sum) {
  variance_16xh_neon(src_ptr, src_stride, ref_ptr, ref_stride, 16, sse, sum);
}

#define VARIANCE_WXH_NEON(w, h, shift)                                        \
  unsigned int vpx_variance##w##x##h##_neon(                                  \
      const uint8_t *src, int src_stride, const uint8_t *ref, int ref_stride, \
      unsigned int *sse) {                                                    \
    int sum;                                                                  \
    variance_##w##xh_neon(src, src_stride, ref, ref_stride, h, sse, &sum);    \
    return *sse - (uint32_t)(((int64_t)sum * sum) >> shift);                  \
  }

VARIANCE_WXH_NEON(4, 4, 4)
VARIANCE_WXH_NEON(4, 8, 5)

VARIANCE_WXH_NEON(8, 4, 5)
VARIANCE_WXH_NEON(8, 8, 6)
VARIANCE_WXH_NEON(8, 16, 7)

VARIANCE_WXH_NEON(16, 8, 7)
VARIANCE_WXH_NEON(16, 16, 8)
VARIANCE_WXH_NEON(16, 32, 9)

VARIANCE_WXH_NEON(32, 16, 9)
VARIANCE_WXH_NEON(32, 32, 10)
VARIANCE_WXH_NEON(32, 64, 11)

VARIANCE_WXH_NEON(64, 32, 11)
VARIANCE_WXH_NEON(64, 64, 12)

#undef VARIANCE_WXH_NEON

static INLINE unsigned int vpx_mse8xh_neon(const unsigned char *src_ptr,
                                           int src_stride,
                                           const unsigned char *ref_ptr,
                                           int ref_stride, int h) {
  uint32x4_t sse_u32[2] = { vdupq_n_u32(0), vdupq_n_u32(0) };

  int i = h / 2;
  do {
    uint8x8_t s0, s1, r0, r1, diff0, diff1;
    uint16x8_t sse0, sse1;

    s0 = vld1_u8(src_ptr);
    src_ptr += src_stride;
    s1 = vld1_u8(src_ptr);
    src_ptr += src_stride;
    r0 = vld1_u8(ref_ptr);
    ref_ptr += ref_stride;
    r1 = vld1_u8(ref_ptr);
    ref_ptr += ref_stride;

    diff0 = vabd_u8(s0, r0);
    diff1 = vabd_u8(s1, r1);

    sse0 = vmull_u8(diff0, diff0);
    sse_u32[0] = vpadalq_u16(sse_u32[0], sse0);
    sse1 = vmull_u8(diff1, diff1);
    sse_u32[1] = vpadalq_u16(sse_u32[1], sse1);
  } while (--i != 0);

  return horizontal_add_uint32x4(vaddq_u32(sse_u32[0], sse_u32[1]));
}

static INLINE unsigned int vpx_mse16xh_neon(const unsigned char *src_ptr,
                                            int src_stride,
                                            const unsigned char *ref_ptr,
                                            int ref_stride, int h) {
  uint32x4_t sse_u32[2] = { vdupq_n_u32(0), vdupq_n_u32(0) };

  int i = h;
  do {
    uint8x16_t s, r, diff;
    uint16x8_t sse0, sse1;

    s = vld1q_u8(src_ptr);
    src_ptr += src_stride;
    r = vld1q_u8(ref_ptr);
    ref_ptr += ref_stride;

    diff = vabdq_u8(s, r);

    sse0 = vmull_u8(vget_low_u8(diff), vget_low_u8(diff));
    sse_u32[0] = vpadalq_u16(sse_u32[0], sse0);
    sse1 = vmull_u8(vget_high_u8(diff), vget_high_u8(diff));
    sse_u32[1] = vpadalq_u16(sse_u32[1], sse1);
  } while (--i != 0);

  return horizontal_add_uint32x4(vaddq_u32(sse_u32[0], sse_u32[1]));
}

unsigned int vpx_get4x4sse_cs_neon(const unsigned char *src_ptr, int src_stride,
                                   const unsigned char *ref_ptr,
                                   int ref_stride) {
  uint8x8_t s[2], r[2];
  uint16x8_t abs_diff[2];
  uint32x4_t sse;

  s[0] = load_u8(src_ptr, src_stride);
  r[0] = load_u8(ref_ptr, ref_stride);
  src_ptr += 2 * src_stride;
  ref_ptr += 2 * ref_stride;
  s[1] = load_u8(src_ptr, src_stride);
  r[1] = load_u8(ref_ptr, ref_stride);

  abs_diff[0] = vabdl_u8(s[0], r[0]);
  abs_diff[1] = vabdl_u8(s[1], r[1]);

  sse = vmull_u16(vget_low_u16(abs_diff[0]), vget_low_u16(abs_diff[0]));
  sse = vmlal_u16(sse, vget_high_u16(abs_diff[0]), vget_high_u16(abs_diff[0]));
  sse = vmlal_u16(sse, vget_low_u16(abs_diff[1]), vget_low_u16(abs_diff[1]));
  sse = vmlal_u16(sse, vget_high_u16(abs_diff[1]), vget_high_u16(abs_diff[1]));

  return horizontal_add_uint32x4(sse);
}

#define VPX_MSE_WXH_NEON(w, h)                                               \
  unsigned int vpx_mse##w##x##h##_neon(                                      \
      const unsigned char *src_ptr, int src_stride,                          \
      const unsigned char *ref_ptr, int ref_stride, unsigned int *sse) {     \
    *sse = vpx_mse##w##xh_neon(src_ptr, src_stride, ref_ptr, ref_stride, h); \
    return *sse;                                                             \
  }

VPX_MSE_WXH_NEON(8, 8)
VPX_MSE_WXH_NEON(8, 16)
VPX_MSE_WXH_NEON(16, 8)
VPX_MSE_WXH_NEON(16, 16)

#undef VPX_MSE_WXH_NEON
