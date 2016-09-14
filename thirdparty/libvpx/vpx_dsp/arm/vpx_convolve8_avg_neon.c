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

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"
#include "vpx_ports/mem.h"

static INLINE int32x4_t MULTIPLY_BY_Q0(
    int16x4_t dsrc0,
    int16x4_t dsrc1,
    int16x4_t dsrc2,
    int16x4_t dsrc3,
    int16x4_t dsrc4,
    int16x4_t dsrc5,
    int16x4_t dsrc6,
    int16x4_t dsrc7,
    int16x8_t q0s16) {
  int32x4_t qdst;
  int16x4_t d0s16, d1s16;

  d0s16 = vget_low_s16(q0s16);
  d1s16 = vget_high_s16(q0s16);

  qdst = vmull_lane_s16(dsrc0, d0s16, 0);
  qdst = vmlal_lane_s16(qdst, dsrc1, d0s16, 1);
  qdst = vmlal_lane_s16(qdst, dsrc2, d0s16, 2);
  qdst = vmlal_lane_s16(qdst, dsrc3, d0s16, 3);
  qdst = vmlal_lane_s16(qdst, dsrc4, d1s16, 0);
  qdst = vmlal_lane_s16(qdst, dsrc5, d1s16, 1);
  qdst = vmlal_lane_s16(qdst, dsrc6, d1s16, 2);
  qdst = vmlal_lane_s16(qdst, dsrc7, d1s16, 3);
  return qdst;
}

void vpx_convolve8_avg_horiz_neon(
    const uint8_t *src,
    ptrdiff_t src_stride,
    uint8_t *dst,
    ptrdiff_t dst_stride,
    const int16_t *filter_x,
    int x_step_q4,
    const int16_t *filter_y,  // unused
    int y_step_q4,            // unused
    int w,
    int h) {
  int width;
  const uint8_t *s;
  uint8_t *d;
  uint8x8_t d2u8, d3u8, d24u8, d25u8, d26u8, d27u8, d28u8, d29u8;
  uint32x2_t d2u32, d3u32, d6u32, d7u32, d28u32, d29u32, d30u32, d31u32;
  uint8x16_t q1u8, q3u8, q12u8, q13u8, q14u8, q15u8;
  int16x4_t d16s16, d17s16, d18s16, d19s16, d20s16, d22s16, d23s16;
  int16x4_t d24s16, d25s16, d26s16, d27s16;
  uint16x4_t d2u16, d3u16, d4u16, d5u16, d16u16, d17u16, d18u16, d19u16;
  int16x8_t q0s16;
  uint16x8_t q1u16, q2u16, q8u16, q9u16, q10u16, q11u16, q12u16, q13u16;
  int32x4_t q1s32, q2s32, q14s32, q15s32;
  uint16x8x2_t q0x2u16;
  uint8x8x2_t d0x2u8, d1x2u8;
  uint32x2x2_t d0x2u32;
  uint16x4x2_t d0x2u16, d1x2u16;
  uint32x4x2_t q0x2u32;

  assert(x_step_q4 == 16);

  q0s16 = vld1q_s16(filter_x);

  src -= 3;  // adjust for taps
  for (; h > 0; h -= 4) {  // loop_horiz_v
    s = src;
    d24u8 = vld1_u8(s);
    s += src_stride;
    d25u8 = vld1_u8(s);
    s += src_stride;
    d26u8 = vld1_u8(s);
    s += src_stride;
    d27u8 = vld1_u8(s);

    q12u8 = vcombine_u8(d24u8, d25u8);
    q13u8 = vcombine_u8(d26u8, d27u8);

    q0x2u16 = vtrnq_u16(vreinterpretq_u16_u8(q12u8),
                        vreinterpretq_u16_u8(q13u8));
    d24u8 = vreinterpret_u8_u16(vget_low_u16(q0x2u16.val[0]));
    d25u8 = vreinterpret_u8_u16(vget_high_u16(q0x2u16.val[0]));
    d26u8 = vreinterpret_u8_u16(vget_low_u16(q0x2u16.val[1]));
    d27u8 = vreinterpret_u8_u16(vget_high_u16(q0x2u16.val[1]));
    d0x2u8 = vtrn_u8(d24u8, d25u8);
    d1x2u8 = vtrn_u8(d26u8, d27u8);

    __builtin_prefetch(src + src_stride * 4);
    __builtin_prefetch(src + src_stride * 5);

    q8u16 = vmovl_u8(d0x2u8.val[0]);
    q9u16 = vmovl_u8(d0x2u8.val[1]);
    q10u16 = vmovl_u8(d1x2u8.val[0]);
    q11u16 = vmovl_u8(d1x2u8.val[1]);

    src += 7;
    d16u16 = vget_low_u16(q8u16);
    d17u16 = vget_high_u16(q8u16);
    d18u16 = vget_low_u16(q9u16);
    d19u16 = vget_high_u16(q9u16);
    q8u16 = vcombine_u16(d16u16, d18u16);  // vswp 17 18
    q9u16 = vcombine_u16(d17u16, d19u16);

    d20s16 = vreinterpret_s16_u16(vget_low_u16(q10u16));
    d23s16 = vreinterpret_s16_u16(vget_high_u16(q10u16));  // vmov 23 21
    for (width = w;
         width > 0;
         width -= 4, src += 4, dst += 4) {  // loop_horiz
      s = src;
      d28u32 = vld1_dup_u32((const uint32_t *)s);
      s += src_stride;
      d29u32 = vld1_dup_u32((const uint32_t *)s);
      s += src_stride;
      d31u32 = vld1_dup_u32((const uint32_t *)s);
      s += src_stride;
      d30u32 = vld1_dup_u32((const uint32_t *)s);

      __builtin_prefetch(src + 64);

      d0x2u16 = vtrn_u16(vreinterpret_u16_u32(d28u32),
                         vreinterpret_u16_u32(d31u32));
      d1x2u16 = vtrn_u16(vreinterpret_u16_u32(d29u32),
                         vreinterpret_u16_u32(d30u32));
      d0x2u8 = vtrn_u8(vreinterpret_u8_u16(d0x2u16.val[0]),   // d28
                       vreinterpret_u8_u16(d1x2u16.val[0]));  // d29
      d1x2u8 = vtrn_u8(vreinterpret_u8_u16(d0x2u16.val[1]),   // d31
                       vreinterpret_u8_u16(d1x2u16.val[1]));  // d30

      __builtin_prefetch(src + 64 + src_stride);

      q14u8 = vcombine_u8(d0x2u8.val[0], d0x2u8.val[1]);
      q15u8 = vcombine_u8(d1x2u8.val[1], d1x2u8.val[0]);
      q0x2u32 = vtrnq_u32(vreinterpretq_u32_u8(q14u8),
                          vreinterpretq_u32_u8(q15u8));

      d28u8 = vreinterpret_u8_u32(vget_low_u32(q0x2u32.val[0]));
      d29u8 = vreinterpret_u8_u32(vget_high_u32(q0x2u32.val[0]));
      q12u16 = vmovl_u8(d28u8);
      q13u16 = vmovl_u8(d29u8);

      __builtin_prefetch(src + 64 + src_stride * 2);

      d = dst;
      d6u32 = vld1_lane_u32((const uint32_t *)d, d6u32, 0);
      d += dst_stride;
      d7u32 = vld1_lane_u32((const uint32_t *)d, d7u32, 0);
      d += dst_stride;
      d6u32 = vld1_lane_u32((const uint32_t *)d, d6u32, 1);
      d += dst_stride;
      d7u32 = vld1_lane_u32((const uint32_t *)d, d7u32, 1);

      d16s16 = vreinterpret_s16_u16(vget_low_u16(q8u16));
      d17s16 = vreinterpret_s16_u16(vget_high_u16(q8u16));
      d18s16 = vreinterpret_s16_u16(vget_low_u16(q9u16));
      d19s16 = vreinterpret_s16_u16(vget_high_u16(q9u16));
      d22s16 = vreinterpret_s16_u16(vget_low_u16(q11u16));
      d24s16 = vreinterpret_s16_u16(vget_low_u16(q12u16));
      d25s16 = vreinterpret_s16_u16(vget_high_u16(q12u16));
      d26s16 = vreinterpret_s16_u16(vget_low_u16(q13u16));
      d27s16 = vreinterpret_s16_u16(vget_high_u16(q13u16));

      q1s32  = MULTIPLY_BY_Q0(d16s16, d17s16, d20s16, d22s16,
                              d18s16, d19s16, d23s16, d24s16, q0s16);
      q2s32  = MULTIPLY_BY_Q0(d17s16, d20s16, d22s16, d18s16,
                              d19s16, d23s16, d24s16, d26s16, q0s16);
      q14s32 = MULTIPLY_BY_Q0(d20s16, d22s16, d18s16, d19s16,
                              d23s16, d24s16, d26s16, d27s16, q0s16);
      q15s32 = MULTIPLY_BY_Q0(d22s16, d18s16, d19s16, d23s16,
                              d24s16, d26s16, d27s16, d25s16, q0s16);

      __builtin_prefetch(src + 64 + src_stride * 3);

      d2u16 = vqrshrun_n_s32(q1s32, 7);
      d3u16 = vqrshrun_n_s32(q2s32, 7);
      d4u16 = vqrshrun_n_s32(q14s32, 7);
      d5u16 = vqrshrun_n_s32(q15s32, 7);

      q1u16 = vcombine_u16(d2u16, d3u16);
      q2u16 = vcombine_u16(d4u16, d5u16);

      d2u8 = vqmovn_u16(q1u16);
      d3u8 = vqmovn_u16(q2u16);

      d0x2u16 = vtrn_u16(vreinterpret_u16_u8(d2u8),
                         vreinterpret_u16_u8(d3u8));
      d0x2u32 = vtrn_u32(vreinterpret_u32_u16(d0x2u16.val[0]),
                         vreinterpret_u32_u16(d0x2u16.val[1]));
      d0x2u8 = vtrn_u8(vreinterpret_u8_u32(d0x2u32.val[0]),
                       vreinterpret_u8_u32(d0x2u32.val[1]));

      q1u8 = vcombine_u8(d0x2u8.val[0], d0x2u8.val[1]);
      q3u8 = vreinterpretq_u8_u32(vcombine_u32(d6u32, d7u32));

      q1u8 = vrhaddq_u8(q1u8, q3u8);

      d2u32 = vreinterpret_u32_u8(vget_low_u8(q1u8));
      d3u32 = vreinterpret_u32_u8(vget_high_u8(q1u8));

      d = dst;
      vst1_lane_u32((uint32_t *)d, d2u32, 0);
      d += dst_stride;
      vst1_lane_u32((uint32_t *)d, d3u32, 0);
      d += dst_stride;
      vst1_lane_u32((uint32_t *)d, d2u32, 1);
      d += dst_stride;
      vst1_lane_u32((uint32_t *)d, d3u32, 1);

      q8u16 = q9u16;
      d20s16 = d23s16;
      q11u16 = q12u16;
      q9u16 = q13u16;
      d23s16 = vreinterpret_s16_u16(vget_high_u16(q11u16));
    }
    src += src_stride * 4 - w - 7;
    dst += dst_stride * 4 - w;
  }
  return;
}

void vpx_convolve8_avg_vert_neon(
    const uint8_t *src,
    ptrdiff_t src_stride,
    uint8_t *dst,
    ptrdiff_t dst_stride,
    const int16_t *filter_x,  // unused
    int x_step_q4,            // unused
    const int16_t *filter_y,
    int y_step_q4,
    int w,
    int h) {
  int height;
  const uint8_t *s;
  uint8_t *d;
  uint8x8_t d2u8, d3u8;
  uint32x2_t d2u32, d3u32, d6u32, d7u32;
  uint32x2_t d16u32, d18u32, d20u32, d22u32, d24u32, d26u32;
  uint8x16_t q1u8, q3u8;
  int16x4_t d16s16, d17s16, d18s16, d19s16, d20s16, d21s16, d22s16;
  int16x4_t d24s16, d25s16, d26s16, d27s16;
  uint16x4_t d2u16, d3u16, d4u16, d5u16;
  int16x8_t q0s16;
  uint16x8_t q1u16, q2u16, q8u16, q9u16, q10u16, q11u16, q12u16, q13u16;
  int32x4_t q1s32, q2s32, q14s32, q15s32;

  assert(y_step_q4 == 16);

  src -= src_stride * 3;
  q0s16 = vld1q_s16(filter_y);
  for (; w > 0; w -= 4, src += 4, dst += 4) {  // loop_vert_h
    s = src;
    d16u32 = vld1_lane_u32((const uint32_t *)s, d16u32, 0);
    s += src_stride;
    d16u32 = vld1_lane_u32((const uint32_t *)s, d16u32, 1);
    s += src_stride;
    d18u32 = vld1_lane_u32((const uint32_t *)s, d18u32, 0);
    s += src_stride;
    d18u32 = vld1_lane_u32((const uint32_t *)s, d18u32, 1);
    s += src_stride;
    d20u32 = vld1_lane_u32((const uint32_t *)s, d20u32, 0);
    s += src_stride;
    d20u32 = vld1_lane_u32((const uint32_t *)s, d20u32, 1);
    s += src_stride;
    d22u32 = vld1_lane_u32((const uint32_t *)s, d22u32, 0);
    s += src_stride;

    q8u16  = vmovl_u8(vreinterpret_u8_u32(d16u32));
    q9u16  = vmovl_u8(vreinterpret_u8_u32(d18u32));
    q10u16 = vmovl_u8(vreinterpret_u8_u32(d20u32));
    q11u16 = vmovl_u8(vreinterpret_u8_u32(d22u32));

    d18s16 = vreinterpret_s16_u16(vget_low_u16(q9u16));
    d19s16 = vreinterpret_s16_u16(vget_high_u16(q9u16));
    d22s16 = vreinterpret_s16_u16(vget_low_u16(q11u16));
    d = dst;
    for (height = h; height > 0; height -= 4) {  // loop_vert
      d24u32 = vld1_lane_u32((const uint32_t *)s, d24u32, 0);
      s += src_stride;
      d26u32 = vld1_lane_u32((const uint32_t *)s, d26u32, 0);
      s += src_stride;
      d26u32 = vld1_lane_u32((const uint32_t *)s, d26u32, 1);
      s += src_stride;
      d24u32 = vld1_lane_u32((const uint32_t *)s, d24u32, 1);
      s += src_stride;

      q12u16 = vmovl_u8(vreinterpret_u8_u32(d24u32));
      q13u16 = vmovl_u8(vreinterpret_u8_u32(d26u32));

      d6u32 = vld1_lane_u32((const uint32_t *)d, d6u32, 0);
      d += dst_stride;
      d6u32 = vld1_lane_u32((const uint32_t *)d, d6u32, 1);
      d += dst_stride;
      d7u32 = vld1_lane_u32((const uint32_t *)d, d7u32, 0);
      d += dst_stride;
      d7u32 = vld1_lane_u32((const uint32_t *)d, d7u32, 1);
      d -= dst_stride * 3;

      d16s16 = vreinterpret_s16_u16(vget_low_u16(q8u16));
      d17s16 = vreinterpret_s16_u16(vget_high_u16(q8u16));
      d20s16 = vreinterpret_s16_u16(vget_low_u16(q10u16));
      d21s16 = vreinterpret_s16_u16(vget_high_u16(q10u16));
      d24s16 = vreinterpret_s16_u16(vget_low_u16(q12u16));
      d25s16 = vreinterpret_s16_u16(vget_high_u16(q12u16));
      d26s16 = vreinterpret_s16_u16(vget_low_u16(q13u16));
      d27s16 = vreinterpret_s16_u16(vget_high_u16(q13u16));

      __builtin_prefetch(s);
      __builtin_prefetch(s + src_stride);
      q1s32  = MULTIPLY_BY_Q0(d16s16, d17s16, d18s16, d19s16,
                              d20s16, d21s16, d22s16, d24s16, q0s16);
      __builtin_prefetch(s + src_stride * 2);
      __builtin_prefetch(s + src_stride * 3);
      q2s32  = MULTIPLY_BY_Q0(d17s16, d18s16, d19s16, d20s16,
                              d21s16, d22s16, d24s16, d26s16, q0s16);
      __builtin_prefetch(d);
      __builtin_prefetch(d + dst_stride);
      q14s32 = MULTIPLY_BY_Q0(d18s16, d19s16, d20s16, d21s16,
                              d22s16, d24s16, d26s16, d27s16, q0s16);
      __builtin_prefetch(d + dst_stride * 2);
      __builtin_prefetch(d + dst_stride * 3);
      q15s32 = MULTIPLY_BY_Q0(d19s16, d20s16, d21s16, d22s16,
                              d24s16, d26s16, d27s16, d25s16, q0s16);

      d2u16 = vqrshrun_n_s32(q1s32, 7);
      d3u16 = vqrshrun_n_s32(q2s32, 7);
      d4u16 = vqrshrun_n_s32(q14s32, 7);
      d5u16 = vqrshrun_n_s32(q15s32, 7);

      q1u16 = vcombine_u16(d2u16, d3u16);
      q2u16 = vcombine_u16(d4u16, d5u16);

      d2u8 = vqmovn_u16(q1u16);
      d3u8 = vqmovn_u16(q2u16);

      q1u8 = vcombine_u8(d2u8, d3u8);
      q3u8 = vreinterpretq_u8_u32(vcombine_u32(d6u32, d7u32));

      q1u8 = vrhaddq_u8(q1u8, q3u8);

      d2u32 = vreinterpret_u32_u8(vget_low_u8(q1u8));
      d3u32 = vreinterpret_u32_u8(vget_high_u8(q1u8));

      vst1_lane_u32((uint32_t *)d, d2u32, 0);
      d += dst_stride;
      vst1_lane_u32((uint32_t *)d, d2u32, 1);
      d += dst_stride;
      vst1_lane_u32((uint32_t *)d, d3u32, 0);
      d += dst_stride;
      vst1_lane_u32((uint32_t *)d, d3u32, 1);
      d += dst_stride;

      q8u16 = q10u16;
      d18s16 = d22s16;
      d19s16 = d24s16;
      q10u16 = q13u16;
      d22s16 = d25s16;
    }
  }
  return;
}
