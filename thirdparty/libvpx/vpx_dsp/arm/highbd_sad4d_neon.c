/*
 *  Copyright (c) 2023 The WebM project authors. All Rights Reserved.
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
#include "vpx_dsp/arm/sum_neon.h"

static INLINE void highbd_sad4xhx4d_neon(const uint8_t *src_ptr, int src_stride,
                                         const uint8_t *const ref_ptr[4],
                                         int ref_stride, uint32_t res[4],
                                         int h) {
  const uint16_t *src16_ptr = CONVERT_TO_SHORTPTR(src_ptr);
  const uint16_t *ref16_ptr0 = CONVERT_TO_SHORTPTR(ref_ptr[0]);
  const uint16_t *ref16_ptr1 = CONVERT_TO_SHORTPTR(ref_ptr[1]);
  const uint16_t *ref16_ptr2 = CONVERT_TO_SHORTPTR(ref_ptr[2]);
  const uint16_t *ref16_ptr3 = CONVERT_TO_SHORTPTR(ref_ptr[3]);

  uint32x4_t sum[4] = { vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0),
                        vdupq_n_u32(0) };

  int i = 0;
  do {
    uint16x4_t s = vld1_u16(src16_ptr + i * src_stride);
    uint16x4_t r0 = vld1_u16(ref16_ptr0 + i * ref_stride);
    uint16x4_t r1 = vld1_u16(ref16_ptr1 + i * ref_stride);
    uint16x4_t r2 = vld1_u16(ref16_ptr2 + i * ref_stride);
    uint16x4_t r3 = vld1_u16(ref16_ptr3 + i * ref_stride);

    sum[0] = vabal_u16(sum[0], s, r0);
    sum[1] = vabal_u16(sum[1], s, r1);
    sum[2] = vabal_u16(sum[2], s, r2);
    sum[3] = vabal_u16(sum[3], s, r3);

  } while (++i < h);

  vst1q_u32(res, horizontal_add_4d_uint32x4(sum));
}

static INLINE void highbd_sad8xhx4d_neon(const uint8_t *src_ptr, int src_stride,
                                         const uint8_t *const ref_ptr[4],
                                         int ref_stride, uint32_t res[4],
                                         int h) {
  const uint16_t *src16_ptr = CONVERT_TO_SHORTPTR(src_ptr);
  const uint16_t *ref16_ptr0 = CONVERT_TO_SHORTPTR(ref_ptr[0]);
  const uint16_t *ref16_ptr1 = CONVERT_TO_SHORTPTR(ref_ptr[1]);
  const uint16_t *ref16_ptr2 = CONVERT_TO_SHORTPTR(ref_ptr[2]);
  const uint16_t *ref16_ptr3 = CONVERT_TO_SHORTPTR(ref_ptr[3]);

  uint16x8_t sum[4] = { vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0),
                        vdupq_n_u16(0) };
  assert(h <= 16);

  int i = 0;
  do {
    uint16x8_t s = vld1q_u16(src16_ptr + i * src_stride);

    sum[0] = vabaq_u16(sum[0], s, vld1q_u16(ref16_ptr0 + i * ref_stride));
    sum[1] = vabaq_u16(sum[1], s, vld1q_u16(ref16_ptr1 + i * ref_stride));
    sum[2] = vabaq_u16(sum[2], s, vld1q_u16(ref16_ptr2 + i * ref_stride));
    sum[3] = vabaq_u16(sum[3], s, vld1q_u16(ref16_ptr3 + i * ref_stride));

  } while (++i < h);

  uint32x4_t sum_u32[4];
  sum_u32[0] = vpaddlq_u16(sum[0]);
  sum_u32[1] = vpaddlq_u16(sum[1]);
  sum_u32[2] = vpaddlq_u16(sum[2]);
  sum_u32[3] = vpaddlq_u16(sum[3]);
  vst1q_u32(res, horizontal_add_4d_uint32x4(sum_u32));
}

static INLINE void highbd_sadwxhx4d_neon(const uint8_t *src_ptr, int src_stride,
                                         const uint8_t *const ref_ptr[4],
                                         int ref_stride, uint32_t res[4], int w,
                                         int h, const int h_overflow) {
  const uint16_t *src16_ptr = CONVERT_TO_SHORTPTR(src_ptr);
  const uint16_t *ref16_ptr0 = CONVERT_TO_SHORTPTR(ref_ptr[0]);
  const uint16_t *ref16_ptr1 = CONVERT_TO_SHORTPTR(ref_ptr[1]);
  const uint16_t *ref16_ptr2 = CONVERT_TO_SHORTPTR(ref_ptr[2]);
  const uint16_t *ref16_ptr3 = CONVERT_TO_SHORTPTR(ref_ptr[3]);

  uint32x4_t sum_u32[4] = { vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0),
                            vdupq_n_u32(0) };

  const int h_limit = h < h_overflow ? h : h_overflow;
  assert(h % h_limit == 0);

  do {
    uint16x8_t sum_u16[4] = { vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0),
                              vdupq_n_u16(0) };
    int i = h_limit;
    do {
      int j = 0;
      do {
        uint16x8_t s0 = vld1q_u16(src16_ptr + j);
        sum_u16[0] = vabaq_u16(sum_u16[0], s0, vld1q_u16(ref16_ptr0 + j));
        sum_u16[1] = vabaq_u16(sum_u16[1], s0, vld1q_u16(ref16_ptr1 + j));
        sum_u16[2] = vabaq_u16(sum_u16[2], s0, vld1q_u16(ref16_ptr2 + j));
        sum_u16[3] = vabaq_u16(sum_u16[3], s0, vld1q_u16(ref16_ptr3 + j));

        uint16x8_t s1 = vld1q_u16(src16_ptr + j + 8);
        sum_u16[0] = vabaq_u16(sum_u16[0], s1, vld1q_u16(ref16_ptr0 + j + 8));
        sum_u16[1] = vabaq_u16(sum_u16[1], s1, vld1q_u16(ref16_ptr1 + j + 8));
        sum_u16[2] = vabaq_u16(sum_u16[2], s1, vld1q_u16(ref16_ptr2 + j + 8));
        sum_u16[3] = vabaq_u16(sum_u16[3], s1, vld1q_u16(ref16_ptr3 + j + 8));

        j += 16;
      } while (j < w);

      src16_ptr += src_stride;
      ref16_ptr0 += ref_stride;
      ref16_ptr1 += ref_stride;
      ref16_ptr2 += ref_stride;
      ref16_ptr3 += ref_stride;
    } while (--i != 0);

    sum_u32[0] = vpadalq_u16(sum_u32[0], sum_u16[0]);
    sum_u32[1] = vpadalq_u16(sum_u32[1], sum_u16[1]);
    sum_u32[2] = vpadalq_u16(sum_u32[2], sum_u16[2]);
    sum_u32[3] = vpadalq_u16(sum_u32[3], sum_u16[3]);

    h -= h_limit;
  } while (h != 0);

  vst1q_u32(res, horizontal_add_4d_uint32x4(sum_u32));
}

static INLINE void highbd_sad16xhx4d_neon(const uint8_t *src_ptr,
                                          int src_stride,
                                          const uint8_t *const ref_ptr[4],
                                          int ref_stride, uint32_t res[4],
                                          int h) {
  // 'h_overflow' is the number of 16-wide rows we can process before 16-bit
  // accumulators overflow. After hitting this limit accumulate into 32-bit
  // elements. 65535 / 4095 ~= 16, so 8 16-wide rows.
  const int h_overflow = 8;
  highbd_sadwxhx4d_neon(src_ptr, src_stride, ref_ptr, ref_stride, res, 16, h,
                        h_overflow);
}

static INLINE void highbd_sad32xhx4d_neon(const uint8_t *src_ptr,
                                          int src_stride,
                                          const uint8_t *const ref_ptr[4],
                                          int ref_stride, uint32_t res[4],
                                          int h) {
  // 'h_overflow' is the number of 32-wide rows we can process before 16-bit
  // accumulators overflow. After hitting this limit accumulate into 32-bit
  // elements. 65535 / 4095 ~= 16, so 4 32-wide rows.
  const int h_overflow = 4;
  highbd_sadwxhx4d_neon(src_ptr, src_stride, ref_ptr, ref_stride, res, 32, h,
                        h_overflow);
}

static INLINE void highbd_sad64xhx4d_neon(const uint8_t *src_ptr,
                                          int src_stride,
                                          const uint8_t *const ref_ptr[4],
                                          int ref_stride, uint32_t res[4],
                                          int h) {
  // 'h_overflow' is the number of 64-wide rows we can process before 16-bit
  // accumulators overflow. After hitting this limit accumulate into 32-bit
  // elements. 65535 / 4095 ~= 16, so 2 64-wide rows.
  const int h_overflow = 2;
  highbd_sadwxhx4d_neon(src_ptr, src_stride, ref_ptr, ref_stride, res, 64, h,
                        h_overflow);
}

#define HBD_SAD_WXH_4D_NEON(w, h)                                            \
  void vpx_highbd_sad##w##x##h##x4d_neon(                                    \
      const uint8_t *src, int src_stride, const uint8_t *const ref_array[4], \
      int ref_stride, uint32_t sad_array[4]) {                               \
    highbd_sad##w##xhx4d_neon(src, src_stride, ref_array, ref_stride,        \
                              sad_array, (h));                               \
  }

HBD_SAD_WXH_4D_NEON(4, 4)
HBD_SAD_WXH_4D_NEON(4, 8)

HBD_SAD_WXH_4D_NEON(8, 4)
HBD_SAD_WXH_4D_NEON(8, 8)
HBD_SAD_WXH_4D_NEON(8, 16)

HBD_SAD_WXH_4D_NEON(16, 8)
HBD_SAD_WXH_4D_NEON(16, 16)
HBD_SAD_WXH_4D_NEON(16, 32)

HBD_SAD_WXH_4D_NEON(32, 16)
HBD_SAD_WXH_4D_NEON(32, 32)
HBD_SAD_WXH_4D_NEON(32, 64)

HBD_SAD_WXH_4D_NEON(64, 32)
HBD_SAD_WXH_4D_NEON(64, 64)

#undef HBD_SAD_WXH_4D_NEON

#define HBD_SAD_SKIP_WXH_4D_NEON(w, h)                                        \
  void vpx_highbd_sad_skip_##w##x##h##x4d_neon(                               \
      const uint8_t *src, int src_stride, const uint8_t *const ref_array[4],  \
      int ref_stride, uint32_t sad_array[4]) {                                \
    highbd_sad##w##xhx4d_neon(src, 2 * src_stride, ref_array, 2 * ref_stride, \
                              sad_array, ((h) >> 1));                         \
    sad_array[0] <<= 1;                                                       \
    sad_array[1] <<= 1;                                                       \
    sad_array[2] <<= 1;                                                       \
    sad_array[3] <<= 1;                                                       \
  }

HBD_SAD_SKIP_WXH_4D_NEON(4, 4)
HBD_SAD_SKIP_WXH_4D_NEON(4, 8)

HBD_SAD_SKIP_WXH_4D_NEON(8, 4)
HBD_SAD_SKIP_WXH_4D_NEON(8, 8)
HBD_SAD_SKIP_WXH_4D_NEON(8, 16)

HBD_SAD_SKIP_WXH_4D_NEON(16, 8)
HBD_SAD_SKIP_WXH_4D_NEON(16, 16)
HBD_SAD_SKIP_WXH_4D_NEON(16, 32)

HBD_SAD_SKIP_WXH_4D_NEON(32, 16)
HBD_SAD_SKIP_WXH_4D_NEON(32, 32)
HBD_SAD_SKIP_WXH_4D_NEON(32, 64)

HBD_SAD_SKIP_WXH_4D_NEON(64, 32)
HBD_SAD_SKIP_WXH_4D_NEON(64, 64)

#undef HBD_SAD_SKIP_WXH_4D_NEON