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
#include <string.h>
#include "./vpx_config.h"
#include "./vp8_rtcd.h"
#include "vpx_dsp/arm/mem_neon.h"
#include "vpx_ports/mem.h"

static const int8_t vp8_sub_pel_filters[8][8] = {
  { 0, 0, -128, 0, 0, 0, 0, 0 },    /* note that 1/8 pel positions are */
  { 0, -6, 123, 12, -1, 0, 0, 0 },  /*    just as per alpha -0.5 bicubic */
  { 2, -11, 108, 36, -8, 1, 0, 0 }, /* New 1/4 pel 6 tap filter */
  { 0, -9, 93, 50, -6, 0, 0, 0 },
  { 3, -16, 77, 77, -16, 3, 0, 0 }, /* New 1/2 pel 6 tap filter */
  { 0, -6, 50, 93, -9, 0, 0, 0 },
  { 1, -8, 36, 108, -11, 2, 0, 0 }, /* New 1/4 pel 6 tap filter */
  { 0, -1, 12, 123, -6, 0, 0, 0 },
};

// This table is derived from vp8/common/filter.c:vp8_sub_pel_filters.
// Apply abs() to all the values. Elements 0, 2, 3, and 5 are always positive.
// Elements 1 and 4 are either 0 or negative. The code accounts for this with
// multiply/accumulates which either add or subtract as needed. The other
// functions will be updated to use this table later.
// It is also expanded to 8 elements to allow loading into 64 bit neon
// registers.
static const uint8_t abs_filters[8][8] = {
  { 0, 0, 128, 0, 0, 0, 0, 0 },   { 0, 6, 123, 12, 1, 0, 0, 0 },
  { 2, 11, 108, 36, 8, 1, 0, 0 }, { 0, 9, 93, 50, 6, 0, 0, 0 },
  { 3, 16, 77, 77, 16, 3, 0, 0 }, { 0, 6, 50, 93, 9, 0, 0, 0 },
  { 1, 8, 36, 108, 11, 2, 0, 0 }, { 0, 1, 12, 123, 6, 0, 0, 0 },
};

static INLINE uint8x8_t load_and_shift(const unsigned char *a) {
  return vreinterpret_u8_u64(vshl_n_u64(vreinterpret_u64_u8(vld1_u8(a)), 32));
}

static INLINE void filter_add_accumulate(const uint8x16_t a, const uint8x16_t b,
                                         const uint8x8_t filter, uint16x8_t *c,
                                         uint16x8_t *d) {
  const uint32x2x2_t a_shuf = vzip_u32(vreinterpret_u32_u8(vget_low_u8(a)),
                                       vreinterpret_u32_u8(vget_high_u8(a)));
  const uint32x2x2_t b_shuf = vzip_u32(vreinterpret_u32_u8(vget_low_u8(b)),
                                       vreinterpret_u32_u8(vget_high_u8(b)));
  *c = vmlal_u8(*c, vreinterpret_u8_u32(a_shuf.val[0]), filter);
  *d = vmlal_u8(*d, vreinterpret_u8_u32(b_shuf.val[0]), filter);
}

static INLINE void filter_sub_accumulate(const uint8x16_t a, const uint8x16_t b,
                                         const uint8x8_t filter, uint16x8_t *c,
                                         uint16x8_t *d) {
  const uint32x2x2_t a_shuf = vzip_u32(vreinterpret_u32_u8(vget_low_u8(a)),
                                       vreinterpret_u32_u8(vget_high_u8(a)));
  const uint32x2x2_t b_shuf = vzip_u32(vreinterpret_u32_u8(vget_low_u8(b)),
                                       vreinterpret_u32_u8(vget_high_u8(b)));
  *c = vmlsl_u8(*c, vreinterpret_u8_u32(a_shuf.val[0]), filter);
  *d = vmlsl_u8(*d, vreinterpret_u8_u32(b_shuf.val[0]), filter);
}

static INLINE void yonly4x4(const unsigned char *src, int src_stride,
                            int filter_offset, unsigned char *dst,
                            int dst_stride) {
  uint8x8_t a0, a1, a2, a3, a4, a5, a6, a7, a8;
  uint8x8_t b0, b1, b2, b3, b4, b5, b6, b7, b8;
  uint16x8_t c0, c1, c2, c3;
  int16x8_t d0, d1;
  uint8x8_t e0, e1;

  const uint8x8_t filter = vld1_u8(abs_filters[filter_offset]);
  const uint8x8_t filter0 = vdup_lane_u8(filter, 0);
  const uint8x8_t filter1 = vdup_lane_u8(filter, 1);
  const uint8x8_t filter2 = vdup_lane_u8(filter, 2);
  const uint8x8_t filter3 = vdup_lane_u8(filter, 3);
  const uint8x8_t filter4 = vdup_lane_u8(filter, 4);
  const uint8x8_t filter5 = vdup_lane_u8(filter, 5);

  src -= src_stride * 2;
  // Shift the even rows to allow using 'vext' to combine the vectors. armv8
  // has vcopy_lane which would be interesting. This started as just a
  // horrible workaround for clang adding alignment hints to 32bit loads:
  // https://llvm.org/bugs/show_bug.cgi?id=24421
  // But it turns out it almost identical to casting the loads.
  a0 = load_and_shift(src);
  src += src_stride;
  a1 = vld1_u8(src);
  src += src_stride;
  a2 = load_and_shift(src);
  src += src_stride;
  a3 = vld1_u8(src);
  src += src_stride;
  a4 = load_and_shift(src);
  src += src_stride;
  a5 = vld1_u8(src);
  src += src_stride;
  a6 = load_and_shift(src);
  src += src_stride;
  a7 = vld1_u8(src);
  src += src_stride;
  a8 = vld1_u8(src);

  // Combine the rows so we can operate on 8 at a time.
  b0 = vext_u8(a0, a1, 4);
  b2 = vext_u8(a2, a3, 4);
  b4 = vext_u8(a4, a5, 4);
  b6 = vext_u8(a6, a7, 4);
  b8 = a8;

  // To keep with the 8-at-a-time theme, combine *alternate* rows. This
  // allows combining the odd rows with the even.
  b1 = vext_u8(b0, b2, 4);
  b3 = vext_u8(b2, b4, 4);
  b5 = vext_u8(b4, b6, 4);
  b7 = vext_u8(b6, b8, 4);

  // Multiply and expand to 16 bits.
  c0 = vmull_u8(b0, filter0);
  c1 = vmull_u8(b2, filter0);
  c2 = vmull_u8(b5, filter5);
  c3 = vmull_u8(b7, filter5);

  // Multiply, subtract and accumulate for filters 1 and 4 (the negative
  // ones).
  c0 = vmlsl_u8(c0, b4, filter4);
  c1 = vmlsl_u8(c1, b6, filter4);
  c2 = vmlsl_u8(c2, b1, filter1);
  c3 = vmlsl_u8(c3, b3, filter1);

  // Add more positive ones. vmlal should really return a signed type.
  // It's doing signed math internally, as evidenced by the fact we can do
  // subtractions followed by more additions. Ideally we could use
  // vqmlal/sl but that instruction doesn't exist. Might be able to
  // shoehorn vqdmlal/vqdmlsl in here but it would take some effort.
  c0 = vmlal_u8(c0, b2, filter2);
  c1 = vmlal_u8(c1, b4, filter2);
  c2 = vmlal_u8(c2, b3, filter3);
  c3 = vmlal_u8(c3, b5, filter3);

  // Use signed saturation math because vmlsl may have left some negative
  // numbers in there.
  d0 = vqaddq_s16(vreinterpretq_s16_u16(c2), vreinterpretq_s16_u16(c0));
  d1 = vqaddq_s16(vreinterpretq_s16_u16(c3), vreinterpretq_s16_u16(c1));

  // Use signed again because numbers like -200 need to be saturated to 0.
  e0 = vqrshrun_n_s16(d0, 7);
  e1 = vqrshrun_n_s16(d1, 7);

  store_unaligned_u8q(dst, dst_stride, vcombine_u8(e0, e1));
}

void vp8_sixtap_predict4x4_neon(unsigned char *src_ptr, int src_pixels_per_line,
                                int xoffset, int yoffset,
                                unsigned char *dst_ptr, int dst_pitch) {
  uint8x16_t s0, s1, s2, s3, s4;
  uint64x2_t s01, s23;
  // Variables to hold src[] elements for the given filter[]
  uint8x8_t s0_f5, s1_f5, s2_f5, s3_f5, s4_f5;
  uint8x8_t s4_f1, s4_f2, s4_f3, s4_f4;
  uint8x16_t s01_f0, s23_f0;
  uint64x2_t s01_f3, s23_f3;
  uint32x2x2_t s01_f3_q, s23_f3_q, s01_f5_q, s23_f5_q;
  // Accumulator variables.
  uint16x8_t d0123, d4567, d89;
  uint16x8_t d0123_a, d4567_a, d89_a;
  int16x8_t e0123, e4567, e89;
  // Second pass intermediates.
  uint8x8_t b0, b1, b2, b3, b4, b5, b6, b7, b8;
  uint16x8_t c0, c1, c2, c3;
  int16x8_t d0, d1;
  uint8x8_t e0, e1;
  uint8x8_t filter, filter0, filter1, filter2, filter3, filter4, filter5;

  if (xoffset == 0) {  // Second pass only.
    yonly4x4(src_ptr, src_pixels_per_line, yoffset, dst_ptr, dst_pitch);
    return;
  }

  if (yoffset == 0) {  // First pass only.
    src_ptr -= 2;
  } else {  // Add context for the second pass. 2 extra lines on top.
    src_ptr -= 2 + (src_pixels_per_line * 2);
  }

  filter = vld1_u8(abs_filters[xoffset]);
  filter0 = vdup_lane_u8(filter, 0);
  filter1 = vdup_lane_u8(filter, 1);
  filter2 = vdup_lane_u8(filter, 2);
  filter3 = vdup_lane_u8(filter, 3);
  filter4 = vdup_lane_u8(filter, 4);
  filter5 = vdup_lane_u8(filter, 5);

  // 2 bytes of context, 4 bytes of src values, 3 bytes of context, 7 bytes of
  // garbage. So much effort for that last single bit.
  // The low values of each pair are for filter0.
  s0 = vld1q_u8(src_ptr);
  src_ptr += src_pixels_per_line;
  s1 = vld1q_u8(src_ptr);
  src_ptr += src_pixels_per_line;
  s2 = vld1q_u8(src_ptr);
  src_ptr += src_pixels_per_line;
  s3 = vld1q_u8(src_ptr);
  src_ptr += src_pixels_per_line;

  // Shift to extract values for filter[5]
  // If src[] is 0, this puts:
  // 3 4 5 6 7 8 9 10 in s0_f5
  // Can't use vshr.u64 because it crosses the double word boundary.
  s0_f5 = vext_u8(vget_low_u8(s0), vget_high_u8(s0), 5);
  s1_f5 = vext_u8(vget_low_u8(s1), vget_high_u8(s1), 5);
  s2_f5 = vext_u8(vget_low_u8(s2), vget_high_u8(s2), 5);
  s3_f5 = vext_u8(vget_low_u8(s3), vget_high_u8(s3), 5);

  s01_f0 = vcombine_u8(vget_low_u8(s0), vget_low_u8(s1));
  s23_f0 = vcombine_u8(vget_low_u8(s2), vget_low_u8(s3));

  s01_f5_q = vzip_u32(vreinterpret_u32_u8(s0_f5), vreinterpret_u32_u8(s1_f5));
  s23_f5_q = vzip_u32(vreinterpret_u32_u8(s2_f5), vreinterpret_u32_u8(s3_f5));
  d0123 = vmull_u8(vreinterpret_u8_u32(s01_f5_q.val[0]), filter5);
  d4567 = vmull_u8(vreinterpret_u8_u32(s23_f5_q.val[0]), filter5);

  // Keep original src data as 64 bits to simplify shifting and extracting.
  s01 = vreinterpretq_u64_u8(s01_f0);
  s23 = vreinterpretq_u64_u8(s23_f0);

  // 3 4 5 6 * filter0
  filter_add_accumulate(s01_f0, s23_f0, filter0, &d0123, &d4567);

  // Shift over one to use -1, 0, 1, 2 for filter1
  // -1 0 1 2 * filter1
  filter_sub_accumulate(vreinterpretq_u8_u64(vshrq_n_u64(s01, 8)),
                        vreinterpretq_u8_u64(vshrq_n_u64(s23, 8)), filter1,
                        &d0123, &d4567);

  // 2 3 4 5 * filter4
  filter_sub_accumulate(vreinterpretq_u8_u64(vshrq_n_u64(s01, 32)),
                        vreinterpretq_u8_u64(vshrq_n_u64(s23, 32)), filter4,
                        &d0123, &d4567);

  // 0 1 2 3 * filter2
  filter_add_accumulate(vreinterpretq_u8_u64(vshrq_n_u64(s01, 16)),
                        vreinterpretq_u8_u64(vshrq_n_u64(s23, 16)), filter2,
                        &d0123, &d4567);

  // 1 2 3 4 * filter3
  s01_f3 = vshrq_n_u64(s01, 24);
  s23_f3 = vshrq_n_u64(s23, 24);
  s01_f3_q = vzip_u32(vreinterpret_u32_u64(vget_low_u64(s01_f3)),
                      vreinterpret_u32_u64(vget_high_u64(s01_f3)));
  s23_f3_q = vzip_u32(vreinterpret_u32_u64(vget_low_u64(s23_f3)),
                      vreinterpret_u32_u64(vget_high_u64(s23_f3)));
  // Accumulate into different registers so it can use saturated addition.
  d0123_a = vmull_u8(vreinterpret_u8_u32(s01_f3_q.val[0]), filter3);
  d4567_a = vmull_u8(vreinterpret_u8_u32(s23_f3_q.val[0]), filter3);

  e0123 =
      vqaddq_s16(vreinterpretq_s16_u16(d0123), vreinterpretq_s16_u16(d0123_a));
  e4567 =
      vqaddq_s16(vreinterpretq_s16_u16(d4567), vreinterpretq_s16_u16(d4567_a));

  // Shift and narrow.
  b0 = vqrshrun_n_s16(e0123, 7);
  b2 = vqrshrun_n_s16(e4567, 7);

  if (yoffset == 0) {  // firstpass_filter4x4_only
    store_unaligned_u8q(dst_ptr, dst_pitch, vcombine_u8(b0, b2));
    return;
  }

  // Load additional context when doing both filters.
  s0 = vld1q_u8(src_ptr);
  src_ptr += src_pixels_per_line;
  s1 = vld1q_u8(src_ptr);
  src_ptr += src_pixels_per_line;
  s2 = vld1q_u8(src_ptr);
  src_ptr += src_pixels_per_line;
  s3 = vld1q_u8(src_ptr);
  src_ptr += src_pixels_per_line;
  s4 = vld1q_u8(src_ptr);

  s0_f5 = vext_u8(vget_low_u8(s0), vget_high_u8(s0), 5);
  s1_f5 = vext_u8(vget_low_u8(s1), vget_high_u8(s1), 5);
  s2_f5 = vext_u8(vget_low_u8(s2), vget_high_u8(s2), 5);
  s3_f5 = vext_u8(vget_low_u8(s3), vget_high_u8(s3), 5);
  s4_f5 = vext_u8(vget_low_u8(s4), vget_high_u8(s4), 5);

  // 3 4 5 6 * filter0
  s01_f0 = vcombine_u8(vget_low_u8(s0), vget_low_u8(s1));
  s23_f0 = vcombine_u8(vget_low_u8(s2), vget_low_u8(s3));

  s01_f5_q = vzip_u32(vreinterpret_u32_u8(s0_f5), vreinterpret_u32_u8(s1_f5));
  s23_f5_q = vzip_u32(vreinterpret_u32_u8(s2_f5), vreinterpret_u32_u8(s3_f5));
  // But this time instead of 16 pixels to filter, there are 20. So an extra
  // run with a doubleword register.
  d0123 = vmull_u8(vreinterpret_u8_u32(s01_f5_q.val[0]), filter5);
  d4567 = vmull_u8(vreinterpret_u8_u32(s23_f5_q.val[0]), filter5);
  d89 = vmull_u8(s4_f5, filter5);

  // Save a copy as u64 for shifting.
  s01 = vreinterpretq_u64_u8(s01_f0);
  s23 = vreinterpretq_u64_u8(s23_f0);

  filter_add_accumulate(s01_f0, s23_f0, filter0, &d0123, &d4567);
  d89 = vmlal_u8(d89, vget_low_u8(s4), filter0);

  filter_sub_accumulate(vreinterpretq_u8_u64(vshrq_n_u64(s01, 8)),
                        vreinterpretq_u8_u64(vshrq_n_u64(s23, 8)), filter1,
                        &d0123, &d4567);
  s4_f1 = vext_u8(vget_low_u8(s4), vget_high_u8(s4), 1);
  d89 = vmlsl_u8(d89, s4_f1, filter1);

  filter_sub_accumulate(vreinterpretq_u8_u64(vshrq_n_u64(s01, 32)),
                        vreinterpretq_u8_u64(vshrq_n_u64(s23, 32)), filter4,
                        &d0123, &d4567);
  s4_f4 = vext_u8(vget_low_u8(s4), vget_high_u8(s4), 4);
  d89 = vmlsl_u8(d89, s4_f4, filter4);

  filter_add_accumulate(vreinterpretq_u8_u64(vshrq_n_u64(s01, 16)),
                        vreinterpretq_u8_u64(vshrq_n_u64(s23, 16)), filter2,
                        &d0123, &d4567);
  s4_f2 = vext_u8(vget_low_u8(s4), vget_high_u8(s4), 2);
  d89 = vmlal_u8(d89, s4_f2, filter2);

  s01_f3 = vshrq_n_u64(s01, 24);
  s23_f3 = vshrq_n_u64(s23, 24);
  s01_f3_q = vzip_u32(vreinterpret_u32_u64(vget_low_u64(s01_f3)),
                      vreinterpret_u32_u64(vget_high_u64(s01_f3)));
  s23_f3_q = vzip_u32(vreinterpret_u32_u64(vget_low_u64(s23_f3)),
                      vreinterpret_u32_u64(vget_high_u64(s23_f3)));
  s4_f3 = vext_u8(vget_low_u8(s4), vget_high_u8(s4), 3);
  d0123_a = vmull_u8(vreinterpret_u8_u32(s01_f3_q.val[0]), filter3);
  d4567_a = vmull_u8(vreinterpret_u8_u32(s23_f3_q.val[0]), filter3);
  d89_a = vmull_u8(s4_f3, filter3);

  e0123 =
      vqaddq_s16(vreinterpretq_s16_u16(d0123), vreinterpretq_s16_u16(d0123_a));
  e4567 =
      vqaddq_s16(vreinterpretq_s16_u16(d4567), vreinterpretq_s16_u16(d4567_a));
  e89 = vqaddq_s16(vreinterpretq_s16_u16(d89), vreinterpretq_s16_u16(d89_a));

  b4 = vqrshrun_n_s16(e0123, 7);
  b6 = vqrshrun_n_s16(e4567, 7);
  b8 = vqrshrun_n_s16(e89, 7);

  // Second pass: 4x4
  filter = vld1_u8(abs_filters[yoffset]);
  filter0 = vdup_lane_u8(filter, 0);
  filter1 = vdup_lane_u8(filter, 1);
  filter2 = vdup_lane_u8(filter, 2);
  filter3 = vdup_lane_u8(filter, 3);
  filter4 = vdup_lane_u8(filter, 4);
  filter5 = vdup_lane_u8(filter, 5);

  b1 = vext_u8(b0, b2, 4);
  b3 = vext_u8(b2, b4, 4);
  b5 = vext_u8(b4, b6, 4);
  b7 = vext_u8(b6, b8, 4);

  c0 = vmull_u8(b0, filter0);
  c1 = vmull_u8(b2, filter0);
  c2 = vmull_u8(b5, filter5);
  c3 = vmull_u8(b7, filter5);

  c0 = vmlsl_u8(c0, b4, filter4);
  c1 = vmlsl_u8(c1, b6, filter4);
  c2 = vmlsl_u8(c2, b1, filter1);
  c3 = vmlsl_u8(c3, b3, filter1);

  c0 = vmlal_u8(c0, b2, filter2);
  c1 = vmlal_u8(c1, b4, filter2);
  c2 = vmlal_u8(c2, b3, filter3);
  c3 = vmlal_u8(c3, b5, filter3);

  d0 = vqaddq_s16(vreinterpretq_s16_u16(c2), vreinterpretq_s16_u16(c0));
  d1 = vqaddq_s16(vreinterpretq_s16_u16(c3), vreinterpretq_s16_u16(c1));

  e0 = vqrshrun_n_s16(d0, 7);
  e1 = vqrshrun_n_s16(d1, 7);

  store_unaligned_u8q(dst_ptr, dst_pitch, vcombine_u8(e0, e1));
}

void vp8_sixtap_predict8x4_neon(unsigned char *src_ptr, int src_pixels_per_line,
                                int xoffset, int yoffset,
                                unsigned char *dst_ptr, int dst_pitch) {
  unsigned char *src;
  uint8x8_t d0u8, d1u8, d2u8, d3u8, d4u8, d5u8, d6u8, d7u8, d8u8, d9u8;
  uint8x8_t d22u8, d23u8, d24u8, d25u8, d26u8;
  uint8x8_t d27u8, d28u8, d29u8, d30u8, d31u8;
  int8x8_t dtmps8, d0s8, d1s8, d2s8, d3s8, d4s8, d5s8;
  uint16x8_t q3u16, q4u16, q5u16, q6u16, q7u16;
  uint16x8_t q8u16, q9u16, q10u16, q11u16, q12u16;
  int16x8_t q3s16, q4s16, q5s16, q6s16, q7s16;
  int16x8_t q8s16, q9s16, q10s16, q11s16, q12s16;
  uint8x16_t q3u8, q4u8, q5u8, q6u8, q7u8;

  if (xoffset == 0) {  // secondpass_filter8x4_only
    // load second_pass filter
    dtmps8 = vld1_s8(vp8_sub_pel_filters[yoffset]);
    d0s8 = vdup_lane_s8(dtmps8, 0);
    d1s8 = vdup_lane_s8(dtmps8, 1);
    d2s8 = vdup_lane_s8(dtmps8, 2);
    d3s8 = vdup_lane_s8(dtmps8, 3);
    d4s8 = vdup_lane_s8(dtmps8, 4);
    d5s8 = vdup_lane_s8(dtmps8, 5);
    d0u8 = vreinterpret_u8_s8(vabs_s8(d0s8));
    d1u8 = vreinterpret_u8_s8(vabs_s8(d1s8));
    d2u8 = vreinterpret_u8_s8(vabs_s8(d2s8));
    d3u8 = vreinterpret_u8_s8(vabs_s8(d3s8));
    d4u8 = vreinterpret_u8_s8(vabs_s8(d4s8));
    d5u8 = vreinterpret_u8_s8(vabs_s8(d5s8));

    // load src data
    src = src_ptr - src_pixels_per_line * 2;
    d22u8 = vld1_u8(src);
    src += src_pixels_per_line;
    d23u8 = vld1_u8(src);
    src += src_pixels_per_line;
    d24u8 = vld1_u8(src);
    src += src_pixels_per_line;
    d25u8 = vld1_u8(src);
    src += src_pixels_per_line;
    d26u8 = vld1_u8(src);
    src += src_pixels_per_line;
    d27u8 = vld1_u8(src);
    src += src_pixels_per_line;
    d28u8 = vld1_u8(src);
    src += src_pixels_per_line;
    d29u8 = vld1_u8(src);
    src += src_pixels_per_line;
    d30u8 = vld1_u8(src);

    q3u16 = vmull_u8(d22u8, d0u8);
    q4u16 = vmull_u8(d23u8, d0u8);
    q5u16 = vmull_u8(d24u8, d0u8);
    q6u16 = vmull_u8(d25u8, d0u8);

    q3u16 = vmlsl_u8(q3u16, d23u8, d1u8);
    q4u16 = vmlsl_u8(q4u16, d24u8, d1u8);
    q5u16 = vmlsl_u8(q5u16, d25u8, d1u8);
    q6u16 = vmlsl_u8(q6u16, d26u8, d1u8);

    q3u16 = vmlsl_u8(q3u16, d26u8, d4u8);
    q4u16 = vmlsl_u8(q4u16, d27u8, d4u8);
    q5u16 = vmlsl_u8(q5u16, d28u8, d4u8);
    q6u16 = vmlsl_u8(q6u16, d29u8, d4u8);

    q3u16 = vmlal_u8(q3u16, d24u8, d2u8);
    q4u16 = vmlal_u8(q4u16, d25u8, d2u8);
    q5u16 = vmlal_u8(q5u16, d26u8, d2u8);
    q6u16 = vmlal_u8(q6u16, d27u8, d2u8);

    q3u16 = vmlal_u8(q3u16, d27u8, d5u8);
    q4u16 = vmlal_u8(q4u16, d28u8, d5u8);
    q5u16 = vmlal_u8(q5u16, d29u8, d5u8);
    q6u16 = vmlal_u8(q6u16, d30u8, d5u8);

    q7u16 = vmull_u8(d25u8, d3u8);
    q8u16 = vmull_u8(d26u8, d3u8);
    q9u16 = vmull_u8(d27u8, d3u8);
    q10u16 = vmull_u8(d28u8, d3u8);

    q3s16 = vreinterpretq_s16_u16(q3u16);
    q4s16 = vreinterpretq_s16_u16(q4u16);
    q5s16 = vreinterpretq_s16_u16(q5u16);
    q6s16 = vreinterpretq_s16_u16(q6u16);
    q7s16 = vreinterpretq_s16_u16(q7u16);
    q8s16 = vreinterpretq_s16_u16(q8u16);
    q9s16 = vreinterpretq_s16_u16(q9u16);
    q10s16 = vreinterpretq_s16_u16(q10u16);

    q7s16 = vqaddq_s16(q7s16, q3s16);
    q8s16 = vqaddq_s16(q8s16, q4s16);
    q9s16 = vqaddq_s16(q9s16, q5s16);
    q10s16 = vqaddq_s16(q10s16, q6s16);

    d6u8 = vqrshrun_n_s16(q7s16, 7);
    d7u8 = vqrshrun_n_s16(q8s16, 7);
    d8u8 = vqrshrun_n_s16(q9s16, 7);
    d9u8 = vqrshrun_n_s16(q10s16, 7);

    vst1_u8(dst_ptr, d6u8);
    dst_ptr += dst_pitch;
    vst1_u8(dst_ptr, d7u8);
    dst_ptr += dst_pitch;
    vst1_u8(dst_ptr, d8u8);
    dst_ptr += dst_pitch;
    vst1_u8(dst_ptr, d9u8);
    return;
  }

  // load first_pass filter
  dtmps8 = vld1_s8(vp8_sub_pel_filters[xoffset]);
  d0s8 = vdup_lane_s8(dtmps8, 0);
  d1s8 = vdup_lane_s8(dtmps8, 1);
  d2s8 = vdup_lane_s8(dtmps8, 2);
  d3s8 = vdup_lane_s8(dtmps8, 3);
  d4s8 = vdup_lane_s8(dtmps8, 4);
  d5s8 = vdup_lane_s8(dtmps8, 5);
  d0u8 = vreinterpret_u8_s8(vabs_s8(d0s8));
  d1u8 = vreinterpret_u8_s8(vabs_s8(d1s8));
  d2u8 = vreinterpret_u8_s8(vabs_s8(d2s8));
  d3u8 = vreinterpret_u8_s8(vabs_s8(d3s8));
  d4u8 = vreinterpret_u8_s8(vabs_s8(d4s8));
  d5u8 = vreinterpret_u8_s8(vabs_s8(d5s8));

  // First pass: output_height lines x output_width columns (9x4)
  if (yoffset == 0)  // firstpass_filter4x4_only
    src = src_ptr - 2;
  else
    src = src_ptr - 2 - (src_pixels_per_line * 2);
  q3u8 = vld1q_u8(src);
  src += src_pixels_per_line;
  q4u8 = vld1q_u8(src);
  src += src_pixels_per_line;
  q5u8 = vld1q_u8(src);
  src += src_pixels_per_line;
  q6u8 = vld1q_u8(src);

  q7u16 = vmull_u8(vget_low_u8(q3u8), d0u8);
  q8u16 = vmull_u8(vget_low_u8(q4u8), d0u8);
  q9u16 = vmull_u8(vget_low_u8(q5u8), d0u8);
  q10u16 = vmull_u8(vget_low_u8(q6u8), d0u8);

  d28u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 1);
  d29u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 1);
  d30u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 1);
  d31u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 1);

  q7u16 = vmlsl_u8(q7u16, d28u8, d1u8);
  q8u16 = vmlsl_u8(q8u16, d29u8, d1u8);
  q9u16 = vmlsl_u8(q9u16, d30u8, d1u8);
  q10u16 = vmlsl_u8(q10u16, d31u8, d1u8);

  d28u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 4);
  d29u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 4);
  d30u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 4);
  d31u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 4);

  q7u16 = vmlsl_u8(q7u16, d28u8, d4u8);
  q8u16 = vmlsl_u8(q8u16, d29u8, d4u8);
  q9u16 = vmlsl_u8(q9u16, d30u8, d4u8);
  q10u16 = vmlsl_u8(q10u16, d31u8, d4u8);

  d28u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 2);
  d29u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 2);
  d30u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 2);
  d31u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 2);

  q7u16 = vmlal_u8(q7u16, d28u8, d2u8);
  q8u16 = vmlal_u8(q8u16, d29u8, d2u8);
  q9u16 = vmlal_u8(q9u16, d30u8, d2u8);
  q10u16 = vmlal_u8(q10u16, d31u8, d2u8);

  d28u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 5);
  d29u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 5);
  d30u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 5);
  d31u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 5);

  q7u16 = vmlal_u8(q7u16, d28u8, d5u8);
  q8u16 = vmlal_u8(q8u16, d29u8, d5u8);
  q9u16 = vmlal_u8(q9u16, d30u8, d5u8);
  q10u16 = vmlal_u8(q10u16, d31u8, d5u8);

  d28u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 3);
  d29u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 3);
  d30u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 3);
  d31u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 3);

  q3u16 = vmull_u8(d28u8, d3u8);
  q4u16 = vmull_u8(d29u8, d3u8);
  q5u16 = vmull_u8(d30u8, d3u8);
  q6u16 = vmull_u8(d31u8, d3u8);

  q3s16 = vreinterpretq_s16_u16(q3u16);
  q4s16 = vreinterpretq_s16_u16(q4u16);
  q5s16 = vreinterpretq_s16_u16(q5u16);
  q6s16 = vreinterpretq_s16_u16(q6u16);
  q7s16 = vreinterpretq_s16_u16(q7u16);
  q8s16 = vreinterpretq_s16_u16(q8u16);
  q9s16 = vreinterpretq_s16_u16(q9u16);
  q10s16 = vreinterpretq_s16_u16(q10u16);

  q7s16 = vqaddq_s16(q7s16, q3s16);
  q8s16 = vqaddq_s16(q8s16, q4s16);
  q9s16 = vqaddq_s16(q9s16, q5s16);
  q10s16 = vqaddq_s16(q10s16, q6s16);

  d22u8 = vqrshrun_n_s16(q7s16, 7);
  d23u8 = vqrshrun_n_s16(q8s16, 7);
  d24u8 = vqrshrun_n_s16(q9s16, 7);
  d25u8 = vqrshrun_n_s16(q10s16, 7);

  if (yoffset == 0) {  // firstpass_filter8x4_only
    vst1_u8(dst_ptr, d22u8);
    dst_ptr += dst_pitch;
    vst1_u8(dst_ptr, d23u8);
    dst_ptr += dst_pitch;
    vst1_u8(dst_ptr, d24u8);
    dst_ptr += dst_pitch;
    vst1_u8(dst_ptr, d25u8);
    return;
  }

  // First Pass on rest 5-line data
  src += src_pixels_per_line;
  q3u8 = vld1q_u8(src);
  src += src_pixels_per_line;
  q4u8 = vld1q_u8(src);
  src += src_pixels_per_line;
  q5u8 = vld1q_u8(src);
  src += src_pixels_per_line;
  q6u8 = vld1q_u8(src);
  src += src_pixels_per_line;
  q7u8 = vld1q_u8(src);

  q8u16 = vmull_u8(vget_low_u8(q3u8), d0u8);
  q9u16 = vmull_u8(vget_low_u8(q4u8), d0u8);
  q10u16 = vmull_u8(vget_low_u8(q5u8), d0u8);
  q11u16 = vmull_u8(vget_low_u8(q6u8), d0u8);
  q12u16 = vmull_u8(vget_low_u8(q7u8), d0u8);

  d27u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 1);
  d28u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 1);
  d29u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 1);
  d30u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 1);
  d31u8 = vext_u8(vget_low_u8(q7u8), vget_high_u8(q7u8), 1);

  q8u16 = vmlsl_u8(q8u16, d27u8, d1u8);
  q9u16 = vmlsl_u8(q9u16, d28u8, d1u8);
  q10u16 = vmlsl_u8(q10u16, d29u8, d1u8);
  q11u16 = vmlsl_u8(q11u16, d30u8, d1u8);
  q12u16 = vmlsl_u8(q12u16, d31u8, d1u8);

  d27u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 4);
  d28u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 4);
  d29u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 4);
  d30u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 4);
  d31u8 = vext_u8(vget_low_u8(q7u8), vget_high_u8(q7u8), 4);

  q8u16 = vmlsl_u8(q8u16, d27u8, d4u8);
  q9u16 = vmlsl_u8(q9u16, d28u8, d4u8);
  q10u16 = vmlsl_u8(q10u16, d29u8, d4u8);
  q11u16 = vmlsl_u8(q11u16, d30u8, d4u8);
  q12u16 = vmlsl_u8(q12u16, d31u8, d4u8);

  d27u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 2);
  d28u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 2);
  d29u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 2);
  d30u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 2);
  d31u8 = vext_u8(vget_low_u8(q7u8), vget_high_u8(q7u8), 2);

  q8u16 = vmlal_u8(q8u16, d27u8, d2u8);
  q9u16 = vmlal_u8(q9u16, d28u8, d2u8);
  q10u16 = vmlal_u8(q10u16, d29u8, d2u8);
  q11u16 = vmlal_u8(q11u16, d30u8, d2u8);
  q12u16 = vmlal_u8(q12u16, d31u8, d2u8);

  d27u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 5);
  d28u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 5);
  d29u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 5);
  d30u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 5);
  d31u8 = vext_u8(vget_low_u8(q7u8), vget_high_u8(q7u8), 5);

  q8u16 = vmlal_u8(q8u16, d27u8, d5u8);
  q9u16 = vmlal_u8(q9u16, d28u8, d5u8);
  q10u16 = vmlal_u8(q10u16, d29u8, d5u8);
  q11u16 = vmlal_u8(q11u16, d30u8, d5u8);
  q12u16 = vmlal_u8(q12u16, d31u8, d5u8);

  d27u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 3);
  d28u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 3);
  d29u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 3);
  d30u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 3);
  d31u8 = vext_u8(vget_low_u8(q7u8), vget_high_u8(q7u8), 3);

  q3u16 = vmull_u8(d27u8, d3u8);
  q4u16 = vmull_u8(d28u8, d3u8);
  q5u16 = vmull_u8(d29u8, d3u8);
  q6u16 = vmull_u8(d30u8, d3u8);
  q7u16 = vmull_u8(d31u8, d3u8);

  q3s16 = vreinterpretq_s16_u16(q3u16);
  q4s16 = vreinterpretq_s16_u16(q4u16);
  q5s16 = vreinterpretq_s16_u16(q5u16);
  q6s16 = vreinterpretq_s16_u16(q6u16);
  q7s16 = vreinterpretq_s16_u16(q7u16);
  q8s16 = vreinterpretq_s16_u16(q8u16);
  q9s16 = vreinterpretq_s16_u16(q9u16);
  q10s16 = vreinterpretq_s16_u16(q10u16);
  q11s16 = vreinterpretq_s16_u16(q11u16);
  q12s16 = vreinterpretq_s16_u16(q12u16);

  q8s16 = vqaddq_s16(q8s16, q3s16);
  q9s16 = vqaddq_s16(q9s16, q4s16);
  q10s16 = vqaddq_s16(q10s16, q5s16);
  q11s16 = vqaddq_s16(q11s16, q6s16);
  q12s16 = vqaddq_s16(q12s16, q7s16);

  d26u8 = vqrshrun_n_s16(q8s16, 7);
  d27u8 = vqrshrun_n_s16(q9s16, 7);
  d28u8 = vqrshrun_n_s16(q10s16, 7);
  d29u8 = vqrshrun_n_s16(q11s16, 7);
  d30u8 = vqrshrun_n_s16(q12s16, 7);

  // Second pass: 8x4
  dtmps8 = vld1_s8(vp8_sub_pel_filters[yoffset]);
  d0s8 = vdup_lane_s8(dtmps8, 0);
  d1s8 = vdup_lane_s8(dtmps8, 1);
  d2s8 = vdup_lane_s8(dtmps8, 2);
  d3s8 = vdup_lane_s8(dtmps8, 3);
  d4s8 = vdup_lane_s8(dtmps8, 4);
  d5s8 = vdup_lane_s8(dtmps8, 5);
  d0u8 = vreinterpret_u8_s8(vabs_s8(d0s8));
  d1u8 = vreinterpret_u8_s8(vabs_s8(d1s8));
  d2u8 = vreinterpret_u8_s8(vabs_s8(d2s8));
  d3u8 = vreinterpret_u8_s8(vabs_s8(d3s8));
  d4u8 = vreinterpret_u8_s8(vabs_s8(d4s8));
  d5u8 = vreinterpret_u8_s8(vabs_s8(d5s8));

  q3u16 = vmull_u8(d22u8, d0u8);
  q4u16 = vmull_u8(d23u8, d0u8);
  q5u16 = vmull_u8(d24u8, d0u8);
  q6u16 = vmull_u8(d25u8, d0u8);

  q3u16 = vmlsl_u8(q3u16, d23u8, d1u8);
  q4u16 = vmlsl_u8(q4u16, d24u8, d1u8);
  q5u16 = vmlsl_u8(q5u16, d25u8, d1u8);
  q6u16 = vmlsl_u8(q6u16, d26u8, d1u8);

  q3u16 = vmlsl_u8(q3u16, d26u8, d4u8);
  q4u16 = vmlsl_u8(q4u16, d27u8, d4u8);
  q5u16 = vmlsl_u8(q5u16, d28u8, d4u8);
  q6u16 = vmlsl_u8(q6u16, d29u8, d4u8);

  q3u16 = vmlal_u8(q3u16, d24u8, d2u8);
  q4u16 = vmlal_u8(q4u16, d25u8, d2u8);
  q5u16 = vmlal_u8(q5u16, d26u8, d2u8);
  q6u16 = vmlal_u8(q6u16, d27u8, d2u8);

  q3u16 = vmlal_u8(q3u16, d27u8, d5u8);
  q4u16 = vmlal_u8(q4u16, d28u8, d5u8);
  q5u16 = vmlal_u8(q5u16, d29u8, d5u8);
  q6u16 = vmlal_u8(q6u16, d30u8, d5u8);

  q7u16 = vmull_u8(d25u8, d3u8);
  q8u16 = vmull_u8(d26u8, d3u8);
  q9u16 = vmull_u8(d27u8, d3u8);
  q10u16 = vmull_u8(d28u8, d3u8);

  q3s16 = vreinterpretq_s16_u16(q3u16);
  q4s16 = vreinterpretq_s16_u16(q4u16);
  q5s16 = vreinterpretq_s16_u16(q5u16);
  q6s16 = vreinterpretq_s16_u16(q6u16);
  q7s16 = vreinterpretq_s16_u16(q7u16);
  q8s16 = vreinterpretq_s16_u16(q8u16);
  q9s16 = vreinterpretq_s16_u16(q9u16);
  q10s16 = vreinterpretq_s16_u16(q10u16);

  q7s16 = vqaddq_s16(q7s16, q3s16);
  q8s16 = vqaddq_s16(q8s16, q4s16);
  q9s16 = vqaddq_s16(q9s16, q5s16);
  q10s16 = vqaddq_s16(q10s16, q6s16);

  d6u8 = vqrshrun_n_s16(q7s16, 7);
  d7u8 = vqrshrun_n_s16(q8s16, 7);
  d8u8 = vqrshrun_n_s16(q9s16, 7);
  d9u8 = vqrshrun_n_s16(q10s16, 7);

  vst1_u8(dst_ptr, d6u8);
  dst_ptr += dst_pitch;
  vst1_u8(dst_ptr, d7u8);
  dst_ptr += dst_pitch;
  vst1_u8(dst_ptr, d8u8);
  dst_ptr += dst_pitch;
  vst1_u8(dst_ptr, d9u8);
}

void vp8_sixtap_predict8x8_neon(unsigned char *src_ptr, int src_pixels_per_line,
                                int xoffset, int yoffset,
                                unsigned char *dst_ptr, int dst_pitch) {
  unsigned char *src, *tmpp;
  unsigned char tmp[64];
  int i;
  uint8x8_t d0u8, d1u8, d2u8, d3u8, d4u8, d5u8, d6u8, d7u8, d8u8, d9u8;
  uint8x8_t d18u8, d19u8, d20u8, d21u8, d22u8, d23u8, d24u8, d25u8;
  uint8x8_t d26u8, d27u8, d28u8, d29u8, d30u8, d31u8;
  int8x8_t dtmps8, d0s8, d1s8, d2s8, d3s8, d4s8, d5s8;
  uint16x8_t q3u16, q4u16, q5u16, q6u16, q7u16;
  uint16x8_t q8u16, q9u16, q10u16, q11u16, q12u16;
  int16x8_t q3s16, q4s16, q5s16, q6s16, q7s16;
  int16x8_t q8s16, q9s16, q10s16, q11s16, q12s16;
  uint8x16_t q3u8, q4u8, q5u8, q6u8, q7u8, q9u8, q10u8, q11u8, q12u8;

  if (xoffset == 0) {  // secondpass_filter8x8_only
    // load second_pass filter
    dtmps8 = vld1_s8(vp8_sub_pel_filters[yoffset]);
    d0s8 = vdup_lane_s8(dtmps8, 0);
    d1s8 = vdup_lane_s8(dtmps8, 1);
    d2s8 = vdup_lane_s8(dtmps8, 2);
    d3s8 = vdup_lane_s8(dtmps8, 3);
    d4s8 = vdup_lane_s8(dtmps8, 4);
    d5s8 = vdup_lane_s8(dtmps8, 5);
    d0u8 = vreinterpret_u8_s8(vabs_s8(d0s8));
    d1u8 = vreinterpret_u8_s8(vabs_s8(d1s8));
    d2u8 = vreinterpret_u8_s8(vabs_s8(d2s8));
    d3u8 = vreinterpret_u8_s8(vabs_s8(d3s8));
    d4u8 = vreinterpret_u8_s8(vabs_s8(d4s8));
    d5u8 = vreinterpret_u8_s8(vabs_s8(d5s8));

    // load src data
    src = src_ptr - src_pixels_per_line * 2;
    d18u8 = vld1_u8(src);
    src += src_pixels_per_line;
    d19u8 = vld1_u8(src);
    src += src_pixels_per_line;
    d20u8 = vld1_u8(src);
    src += src_pixels_per_line;
    d21u8 = vld1_u8(src);
    src += src_pixels_per_line;
    d22u8 = vld1_u8(src);
    src += src_pixels_per_line;
    d23u8 = vld1_u8(src);
    src += src_pixels_per_line;
    d24u8 = vld1_u8(src);
    src += src_pixels_per_line;
    d25u8 = vld1_u8(src);
    src += src_pixels_per_line;
    d26u8 = vld1_u8(src);
    src += src_pixels_per_line;
    d27u8 = vld1_u8(src);
    src += src_pixels_per_line;
    d28u8 = vld1_u8(src);
    src += src_pixels_per_line;
    d29u8 = vld1_u8(src);
    src += src_pixels_per_line;
    d30u8 = vld1_u8(src);

    for (i = 2; i > 0; i--) {
      q3u16 = vmull_u8(d18u8, d0u8);
      q4u16 = vmull_u8(d19u8, d0u8);
      q5u16 = vmull_u8(d20u8, d0u8);
      q6u16 = vmull_u8(d21u8, d0u8);

      q3u16 = vmlsl_u8(q3u16, d19u8, d1u8);
      q4u16 = vmlsl_u8(q4u16, d20u8, d1u8);
      q5u16 = vmlsl_u8(q5u16, d21u8, d1u8);
      q6u16 = vmlsl_u8(q6u16, d22u8, d1u8);

      q3u16 = vmlsl_u8(q3u16, d22u8, d4u8);
      q4u16 = vmlsl_u8(q4u16, d23u8, d4u8);
      q5u16 = vmlsl_u8(q5u16, d24u8, d4u8);
      q6u16 = vmlsl_u8(q6u16, d25u8, d4u8);

      q3u16 = vmlal_u8(q3u16, d20u8, d2u8);
      q4u16 = vmlal_u8(q4u16, d21u8, d2u8);
      q5u16 = vmlal_u8(q5u16, d22u8, d2u8);
      q6u16 = vmlal_u8(q6u16, d23u8, d2u8);

      q3u16 = vmlal_u8(q3u16, d23u8, d5u8);
      q4u16 = vmlal_u8(q4u16, d24u8, d5u8);
      q5u16 = vmlal_u8(q5u16, d25u8, d5u8);
      q6u16 = vmlal_u8(q6u16, d26u8, d5u8);

      q7u16 = vmull_u8(d21u8, d3u8);
      q8u16 = vmull_u8(d22u8, d3u8);
      q9u16 = vmull_u8(d23u8, d3u8);
      q10u16 = vmull_u8(d24u8, d3u8);

      q3s16 = vreinterpretq_s16_u16(q3u16);
      q4s16 = vreinterpretq_s16_u16(q4u16);
      q5s16 = vreinterpretq_s16_u16(q5u16);
      q6s16 = vreinterpretq_s16_u16(q6u16);
      q7s16 = vreinterpretq_s16_u16(q7u16);
      q8s16 = vreinterpretq_s16_u16(q8u16);
      q9s16 = vreinterpretq_s16_u16(q9u16);
      q10s16 = vreinterpretq_s16_u16(q10u16);

      q7s16 = vqaddq_s16(q7s16, q3s16);
      q8s16 = vqaddq_s16(q8s16, q4s16);
      q9s16 = vqaddq_s16(q9s16, q5s16);
      q10s16 = vqaddq_s16(q10s16, q6s16);

      d6u8 = vqrshrun_n_s16(q7s16, 7);
      d7u8 = vqrshrun_n_s16(q8s16, 7);
      d8u8 = vqrshrun_n_s16(q9s16, 7);
      d9u8 = vqrshrun_n_s16(q10s16, 7);

      d18u8 = d22u8;
      d19u8 = d23u8;
      d20u8 = d24u8;
      d21u8 = d25u8;
      d22u8 = d26u8;
      d23u8 = d27u8;
      d24u8 = d28u8;
      d25u8 = d29u8;
      d26u8 = d30u8;

      vst1_u8(dst_ptr, d6u8);
      dst_ptr += dst_pitch;
      vst1_u8(dst_ptr, d7u8);
      dst_ptr += dst_pitch;
      vst1_u8(dst_ptr, d8u8);
      dst_ptr += dst_pitch;
      vst1_u8(dst_ptr, d9u8);
      dst_ptr += dst_pitch;
    }
    return;
  }

  // load first_pass filter
  dtmps8 = vld1_s8(vp8_sub_pel_filters[xoffset]);
  d0s8 = vdup_lane_s8(dtmps8, 0);
  d1s8 = vdup_lane_s8(dtmps8, 1);
  d2s8 = vdup_lane_s8(dtmps8, 2);
  d3s8 = vdup_lane_s8(dtmps8, 3);
  d4s8 = vdup_lane_s8(dtmps8, 4);
  d5s8 = vdup_lane_s8(dtmps8, 5);
  d0u8 = vreinterpret_u8_s8(vabs_s8(d0s8));
  d1u8 = vreinterpret_u8_s8(vabs_s8(d1s8));
  d2u8 = vreinterpret_u8_s8(vabs_s8(d2s8));
  d3u8 = vreinterpret_u8_s8(vabs_s8(d3s8));
  d4u8 = vreinterpret_u8_s8(vabs_s8(d4s8));
  d5u8 = vreinterpret_u8_s8(vabs_s8(d5s8));

  // First pass: output_height lines x output_width columns (9x4)
  if (yoffset == 0)  // firstpass_filter4x4_only
    src = src_ptr - 2;
  else
    src = src_ptr - 2 - (src_pixels_per_line * 2);

  tmpp = tmp;
  for (i = 2; i > 0; i--) {
    q3u8 = vld1q_u8(src);
    src += src_pixels_per_line;
    q4u8 = vld1q_u8(src);
    src += src_pixels_per_line;
    q5u8 = vld1q_u8(src);
    src += src_pixels_per_line;
    q6u8 = vld1q_u8(src);
    src += src_pixels_per_line;

    __builtin_prefetch(src);
    __builtin_prefetch(src + src_pixels_per_line);
    __builtin_prefetch(src + src_pixels_per_line * 2);

    q7u16 = vmull_u8(vget_low_u8(q3u8), d0u8);
    q8u16 = vmull_u8(vget_low_u8(q4u8), d0u8);
    q9u16 = vmull_u8(vget_low_u8(q5u8), d0u8);
    q10u16 = vmull_u8(vget_low_u8(q6u8), d0u8);

    d28u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 1);
    d29u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 1);
    d30u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 1);
    d31u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 1);

    q7u16 = vmlsl_u8(q7u16, d28u8, d1u8);
    q8u16 = vmlsl_u8(q8u16, d29u8, d1u8);
    q9u16 = vmlsl_u8(q9u16, d30u8, d1u8);
    q10u16 = vmlsl_u8(q10u16, d31u8, d1u8);

    d28u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 4);
    d29u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 4);
    d30u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 4);
    d31u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 4);

    q7u16 = vmlsl_u8(q7u16, d28u8, d4u8);
    q8u16 = vmlsl_u8(q8u16, d29u8, d4u8);
    q9u16 = vmlsl_u8(q9u16, d30u8, d4u8);
    q10u16 = vmlsl_u8(q10u16, d31u8, d4u8);

    d28u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 2);
    d29u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 2);
    d30u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 2);
    d31u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 2);

    q7u16 = vmlal_u8(q7u16, d28u8, d2u8);
    q8u16 = vmlal_u8(q8u16, d29u8, d2u8);
    q9u16 = vmlal_u8(q9u16, d30u8, d2u8);
    q10u16 = vmlal_u8(q10u16, d31u8, d2u8);

    d28u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 5);
    d29u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 5);
    d30u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 5);
    d31u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 5);

    q7u16 = vmlal_u8(q7u16, d28u8, d5u8);
    q8u16 = vmlal_u8(q8u16, d29u8, d5u8);
    q9u16 = vmlal_u8(q9u16, d30u8, d5u8);
    q10u16 = vmlal_u8(q10u16, d31u8, d5u8);

    d28u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 3);
    d29u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 3);
    d30u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 3);
    d31u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 3);

    q3u16 = vmull_u8(d28u8, d3u8);
    q4u16 = vmull_u8(d29u8, d3u8);
    q5u16 = vmull_u8(d30u8, d3u8);
    q6u16 = vmull_u8(d31u8, d3u8);

    q3s16 = vreinterpretq_s16_u16(q3u16);
    q4s16 = vreinterpretq_s16_u16(q4u16);
    q5s16 = vreinterpretq_s16_u16(q5u16);
    q6s16 = vreinterpretq_s16_u16(q6u16);
    q7s16 = vreinterpretq_s16_u16(q7u16);
    q8s16 = vreinterpretq_s16_u16(q8u16);
    q9s16 = vreinterpretq_s16_u16(q9u16);
    q10s16 = vreinterpretq_s16_u16(q10u16);

    q7s16 = vqaddq_s16(q7s16, q3s16);
    q8s16 = vqaddq_s16(q8s16, q4s16);
    q9s16 = vqaddq_s16(q9s16, q5s16);
    q10s16 = vqaddq_s16(q10s16, q6s16);

    d22u8 = vqrshrun_n_s16(q7s16, 7);
    d23u8 = vqrshrun_n_s16(q8s16, 7);
    d24u8 = vqrshrun_n_s16(q9s16, 7);
    d25u8 = vqrshrun_n_s16(q10s16, 7);

    if (yoffset == 0) {  // firstpass_filter8x4_only
      vst1_u8(dst_ptr, d22u8);
      dst_ptr += dst_pitch;
      vst1_u8(dst_ptr, d23u8);
      dst_ptr += dst_pitch;
      vst1_u8(dst_ptr, d24u8);
      dst_ptr += dst_pitch;
      vst1_u8(dst_ptr, d25u8);
      dst_ptr += dst_pitch;
    } else {
      vst1_u8(tmpp, d22u8);
      tmpp += 8;
      vst1_u8(tmpp, d23u8);
      tmpp += 8;
      vst1_u8(tmpp, d24u8);
      tmpp += 8;
      vst1_u8(tmpp, d25u8);
      tmpp += 8;
    }
  }
  if (yoffset == 0) return;

  // First Pass on rest 5-line data
  q3u8 = vld1q_u8(src);
  src += src_pixels_per_line;
  q4u8 = vld1q_u8(src);
  src += src_pixels_per_line;
  q5u8 = vld1q_u8(src);
  src += src_pixels_per_line;
  q6u8 = vld1q_u8(src);
  src += src_pixels_per_line;
  q7u8 = vld1q_u8(src);

  q8u16 = vmull_u8(vget_low_u8(q3u8), d0u8);
  q9u16 = vmull_u8(vget_low_u8(q4u8), d0u8);
  q10u16 = vmull_u8(vget_low_u8(q5u8), d0u8);
  q11u16 = vmull_u8(vget_low_u8(q6u8), d0u8);
  q12u16 = vmull_u8(vget_low_u8(q7u8), d0u8);

  d27u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 1);
  d28u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 1);
  d29u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 1);
  d30u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 1);
  d31u8 = vext_u8(vget_low_u8(q7u8), vget_high_u8(q7u8), 1);

  q8u16 = vmlsl_u8(q8u16, d27u8, d1u8);
  q9u16 = vmlsl_u8(q9u16, d28u8, d1u8);
  q10u16 = vmlsl_u8(q10u16, d29u8, d1u8);
  q11u16 = vmlsl_u8(q11u16, d30u8, d1u8);
  q12u16 = vmlsl_u8(q12u16, d31u8, d1u8);

  d27u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 4);
  d28u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 4);
  d29u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 4);
  d30u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 4);
  d31u8 = vext_u8(vget_low_u8(q7u8), vget_high_u8(q7u8), 4);

  q8u16 = vmlsl_u8(q8u16, d27u8, d4u8);
  q9u16 = vmlsl_u8(q9u16, d28u8, d4u8);
  q10u16 = vmlsl_u8(q10u16, d29u8, d4u8);
  q11u16 = vmlsl_u8(q11u16, d30u8, d4u8);
  q12u16 = vmlsl_u8(q12u16, d31u8, d4u8);

  d27u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 2);
  d28u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 2);
  d29u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 2);
  d30u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 2);
  d31u8 = vext_u8(vget_low_u8(q7u8), vget_high_u8(q7u8), 2);

  q8u16 = vmlal_u8(q8u16, d27u8, d2u8);
  q9u16 = vmlal_u8(q9u16, d28u8, d2u8);
  q10u16 = vmlal_u8(q10u16, d29u8, d2u8);
  q11u16 = vmlal_u8(q11u16, d30u8, d2u8);
  q12u16 = vmlal_u8(q12u16, d31u8, d2u8);

  d27u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 5);
  d28u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 5);
  d29u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 5);
  d30u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 5);
  d31u8 = vext_u8(vget_low_u8(q7u8), vget_high_u8(q7u8), 5);

  q8u16 = vmlal_u8(q8u16, d27u8, d5u8);
  q9u16 = vmlal_u8(q9u16, d28u8, d5u8);
  q10u16 = vmlal_u8(q10u16, d29u8, d5u8);
  q11u16 = vmlal_u8(q11u16, d30u8, d5u8);
  q12u16 = vmlal_u8(q12u16, d31u8, d5u8);

  d27u8 = vext_u8(vget_low_u8(q3u8), vget_high_u8(q3u8), 3);
  d28u8 = vext_u8(vget_low_u8(q4u8), vget_high_u8(q4u8), 3);
  d29u8 = vext_u8(vget_low_u8(q5u8), vget_high_u8(q5u8), 3);
  d30u8 = vext_u8(vget_low_u8(q6u8), vget_high_u8(q6u8), 3);
  d31u8 = vext_u8(vget_low_u8(q7u8), vget_high_u8(q7u8), 3);

  q3u16 = vmull_u8(d27u8, d3u8);
  q4u16 = vmull_u8(d28u8, d3u8);
  q5u16 = vmull_u8(d29u8, d3u8);
  q6u16 = vmull_u8(d30u8, d3u8);
  q7u16 = vmull_u8(d31u8, d3u8);

  q3s16 = vreinterpretq_s16_u16(q3u16);
  q4s16 = vreinterpretq_s16_u16(q4u16);
  q5s16 = vreinterpretq_s16_u16(q5u16);
  q6s16 = vreinterpretq_s16_u16(q6u16);
  q7s16 = vreinterpretq_s16_u16(q7u16);
  q8s16 = vreinterpretq_s16_u16(q8u16);
  q9s16 = vreinterpretq_s16_u16(q9u16);
  q10s16 = vreinterpretq_s16_u16(q10u16);
  q11s16 = vreinterpretq_s16_u16(q11u16);
  q12s16 = vreinterpretq_s16_u16(q12u16);

  q8s16 = vqaddq_s16(q8s16, q3s16);
  q9s16 = vqaddq_s16(q9s16, q4s16);
  q10s16 = vqaddq_s16(q10s16, q5s16);
  q11s16 = vqaddq_s16(q11s16, q6s16);
  q12s16 = vqaddq_s16(q12s16, q7s16);

  d26u8 = vqrshrun_n_s16(q8s16, 7);
  d27u8 = vqrshrun_n_s16(q9s16, 7);
  d28u8 = vqrshrun_n_s16(q10s16, 7);
  d29u8 = vqrshrun_n_s16(q11s16, 7);
  d30u8 = vqrshrun_n_s16(q12s16, 7);

  // Second pass: 8x8
  dtmps8 = vld1_s8(vp8_sub_pel_filters[yoffset]);
  d0s8 = vdup_lane_s8(dtmps8, 0);
  d1s8 = vdup_lane_s8(dtmps8, 1);
  d2s8 = vdup_lane_s8(dtmps8, 2);
  d3s8 = vdup_lane_s8(dtmps8, 3);
  d4s8 = vdup_lane_s8(dtmps8, 4);
  d5s8 = vdup_lane_s8(dtmps8, 5);
  d0u8 = vreinterpret_u8_s8(vabs_s8(d0s8));
  d1u8 = vreinterpret_u8_s8(vabs_s8(d1s8));
  d2u8 = vreinterpret_u8_s8(vabs_s8(d2s8));
  d3u8 = vreinterpret_u8_s8(vabs_s8(d3s8));
  d4u8 = vreinterpret_u8_s8(vabs_s8(d4s8));
  d5u8 = vreinterpret_u8_s8(vabs_s8(d5s8));

  tmpp = tmp;
  q9u8 = vld1q_u8(tmpp);
  tmpp += 16;
  q10u8 = vld1q_u8(tmpp);
  tmpp += 16;
  q11u8 = vld1q_u8(tmpp);
  tmpp += 16;
  q12u8 = vld1q_u8(tmpp);

  d18u8 = vget_low_u8(q9u8);
  d19u8 = vget_high_u8(q9u8);
  d20u8 = vget_low_u8(q10u8);
  d21u8 = vget_high_u8(q10u8);
  d22u8 = vget_low_u8(q11u8);
  d23u8 = vget_high_u8(q11u8);
  d24u8 = vget_low_u8(q12u8);
  d25u8 = vget_high_u8(q12u8);

  for (i = 2; i > 0; i--) {
    q3u16 = vmull_u8(d18u8, d0u8);
    q4u16 = vmull_u8(d19u8, d0u8);
    q5u16 = vmull_u8(d20u8, d0u8);
    q6u16 = vmull_u8(d21u8, d0u8);

    q3u16 = vmlsl_u8(q3u16, d19u8, d1u8);
    q4u16 = vmlsl_u8(q4u16, d20u8, d1u8);
    q5u16 = vmlsl_u8(q5u16, d21u8, d1u8);
    q6u16 = vmlsl_u8(q6u16, d22u8, d1u8);

    q3u16 = vmlsl_u8(q3u16, d22u8, d4u8);
    q4u16 = vmlsl_u8(q4u16, d23u8, d4u8);
    q5u16 = vmlsl_u8(q5u16, d24u8, d4u8);
    q6u16 = vmlsl_u8(q6u16, d25u8, d4u8);

    q3u16 = vmlal_u8(q3u16, d20u8, d2u8);
    q4u16 = vmlal_u8(q4u16, d21u8, d2u8);
    q5u16 = vmlal_u8(q5u16, d22u8, d2u8);
    q6u16 = vmlal_u8(q6u16, d23u8, d2u8);

    q3u16 = vmlal_u8(q3u16, d23u8, d5u8);
    q4u16 = vmlal_u8(q4u16, d24u8, d5u8);
    q5u16 = vmlal_u8(q5u16, d25u8, d5u8);
    q6u16 = vmlal_u8(q6u16, d26u8, d5u8);

    q7u16 = vmull_u8(d21u8, d3u8);
    q8u16 = vmull_u8(d22u8, d3u8);
    q9u16 = vmull_u8(d23u8, d3u8);
    q10u16 = vmull_u8(d24u8, d3u8);

    q3s16 = vreinterpretq_s16_u16(q3u16);
    q4s16 = vreinterpretq_s16_u16(q4u16);
    q5s16 = vreinterpretq_s16_u16(q5u16);
    q6s16 = vreinterpretq_s16_u16(q6u16);
    q7s16 = vreinterpretq_s16_u16(q7u16);
    q8s16 = vreinterpretq_s16_u16(q8u16);
    q9s16 = vreinterpretq_s16_u16(q9u16);
    q10s16 = vreinterpretq_s16_u16(q10u16);

    q7s16 = vqaddq_s16(q7s16, q3s16);
    q8s16 = vqaddq_s16(q8s16, q4s16);
    q9s16 = vqaddq_s16(q9s16, q5s16);
    q10s16 = vqaddq_s16(q10s16, q6s16);

    d6u8 = vqrshrun_n_s16(q7s16, 7);
    d7u8 = vqrshrun_n_s16(q8s16, 7);
    d8u8 = vqrshrun_n_s16(q9s16, 7);
    d9u8 = vqrshrun_n_s16(q10s16, 7);

    d18u8 = d22u8;
    d19u8 = d23u8;
    d20u8 = d24u8;
    d21u8 = d25u8;
    d22u8 = d26u8;
    d23u8 = d27u8;
    d24u8 = d28u8;
    d25u8 = d29u8;
    d26u8 = d30u8;

    vst1_u8(dst_ptr, d6u8);
    dst_ptr += dst_pitch;
    vst1_u8(dst_ptr, d7u8);
    dst_ptr += dst_pitch;
    vst1_u8(dst_ptr, d8u8);
    dst_ptr += dst_pitch;
    vst1_u8(dst_ptr, d9u8);
    dst_ptr += dst_pitch;
  }
}

void vp8_sixtap_predict16x16_neon(unsigned char *src_ptr,
                                  int src_pixels_per_line, int xoffset,
                                  int yoffset, unsigned char *dst_ptr,
                                  int dst_pitch) {
  unsigned char *src, *src_tmp, *dst, *tmpp;
  unsigned char tmp[336];
  int i, j;
  uint8x8_t d0u8, d1u8, d2u8, d3u8, d4u8, d5u8, d6u8, d7u8, d8u8, d9u8;
  uint8x8_t d10u8, d11u8, d12u8, d13u8, d14u8, d15u8, d18u8, d19u8;
  uint8x8_t d20u8, d21u8, d22u8, d23u8, d24u8, d25u8, d26u8, d27u8;
  uint8x8_t d28u8, d29u8, d30u8, d31u8;
  int8x8_t dtmps8, d0s8, d1s8, d2s8, d3s8, d4s8, d5s8;
  uint8x16_t q3u8, q4u8;
  uint16x8_t q3u16, q4u16, q5u16, q6u16, q7u16, q8u16, q9u16, q10u16;
  uint16x8_t q11u16, q12u16, q13u16, q15u16;
  int16x8_t q3s16, q4s16, q5s16, q6s16, q7s16, q8s16, q9s16, q10s16;
  int16x8_t q11s16, q12s16, q13s16, q15s16;

  if (xoffset == 0) {  // secondpass_filter8x8_only
    // load second_pass filter
    dtmps8 = vld1_s8(vp8_sub_pel_filters[yoffset]);
    d0s8 = vdup_lane_s8(dtmps8, 0);
    d1s8 = vdup_lane_s8(dtmps8, 1);
    d2s8 = vdup_lane_s8(dtmps8, 2);
    d3s8 = vdup_lane_s8(dtmps8, 3);
    d4s8 = vdup_lane_s8(dtmps8, 4);
    d5s8 = vdup_lane_s8(dtmps8, 5);
    d0u8 = vreinterpret_u8_s8(vabs_s8(d0s8));
    d1u8 = vreinterpret_u8_s8(vabs_s8(d1s8));
    d2u8 = vreinterpret_u8_s8(vabs_s8(d2s8));
    d3u8 = vreinterpret_u8_s8(vabs_s8(d3s8));
    d4u8 = vreinterpret_u8_s8(vabs_s8(d4s8));
    d5u8 = vreinterpret_u8_s8(vabs_s8(d5s8));

    // load src data
    src_tmp = src_ptr - src_pixels_per_line * 2;
    for (i = 0; i < 2; ++i) {
      src = src_tmp + i * 8;
      dst = dst_ptr + i * 8;
      d18u8 = vld1_u8(src);
      src += src_pixels_per_line;
      d19u8 = vld1_u8(src);
      src += src_pixels_per_line;
      d20u8 = vld1_u8(src);
      src += src_pixels_per_line;
      d21u8 = vld1_u8(src);
      src += src_pixels_per_line;
      d22u8 = vld1_u8(src);
      src += src_pixels_per_line;
      for (j = 0; j < 4; ++j) {
        d23u8 = vld1_u8(src);
        src += src_pixels_per_line;
        d24u8 = vld1_u8(src);
        src += src_pixels_per_line;
        d25u8 = vld1_u8(src);
        src += src_pixels_per_line;
        d26u8 = vld1_u8(src);
        src += src_pixels_per_line;

        q3u16 = vmull_u8(d18u8, d0u8);
        q4u16 = vmull_u8(d19u8, d0u8);
        q5u16 = vmull_u8(d20u8, d0u8);
        q6u16 = vmull_u8(d21u8, d0u8);

        q3u16 = vmlsl_u8(q3u16, d19u8, d1u8);
        q4u16 = vmlsl_u8(q4u16, d20u8, d1u8);
        q5u16 = vmlsl_u8(q5u16, d21u8, d1u8);
        q6u16 = vmlsl_u8(q6u16, d22u8, d1u8);

        q3u16 = vmlsl_u8(q3u16, d22u8, d4u8);
        q4u16 = vmlsl_u8(q4u16, d23u8, d4u8);
        q5u16 = vmlsl_u8(q5u16, d24u8, d4u8);
        q6u16 = vmlsl_u8(q6u16, d25u8, d4u8);

        q3u16 = vmlal_u8(q3u16, d20u8, d2u8);
        q4u16 = vmlal_u8(q4u16, d21u8, d2u8);
        q5u16 = vmlal_u8(q5u16, d22u8, d2u8);
        q6u16 = vmlal_u8(q6u16, d23u8, d2u8);

        q3u16 = vmlal_u8(q3u16, d23u8, d5u8);
        q4u16 = vmlal_u8(q4u16, d24u8, d5u8);
        q5u16 = vmlal_u8(q5u16, d25u8, d5u8);
        q6u16 = vmlal_u8(q6u16, d26u8, d5u8);

        q7u16 = vmull_u8(d21u8, d3u8);
        q8u16 = vmull_u8(d22u8, d3u8);
        q9u16 = vmull_u8(d23u8, d3u8);
        q10u16 = vmull_u8(d24u8, d3u8);

        q3s16 = vreinterpretq_s16_u16(q3u16);
        q4s16 = vreinterpretq_s16_u16(q4u16);
        q5s16 = vreinterpretq_s16_u16(q5u16);
        q6s16 = vreinterpretq_s16_u16(q6u16);
        q7s16 = vreinterpretq_s16_u16(q7u16);
        q8s16 = vreinterpretq_s16_u16(q8u16);
        q9s16 = vreinterpretq_s16_u16(q9u16);
        q10s16 = vreinterpretq_s16_u16(q10u16);

        q7s16 = vqaddq_s16(q7s16, q3s16);
        q8s16 = vqaddq_s16(q8s16, q4s16);
        q9s16 = vqaddq_s16(q9s16, q5s16);
        q10s16 = vqaddq_s16(q10s16, q6s16);

        d6u8 = vqrshrun_n_s16(q7s16, 7);
        d7u8 = vqrshrun_n_s16(q8s16, 7);
        d8u8 = vqrshrun_n_s16(q9s16, 7);
        d9u8 = vqrshrun_n_s16(q10s16, 7);

        d18u8 = d22u8;
        d19u8 = d23u8;
        d20u8 = d24u8;
        d21u8 = d25u8;
        d22u8 = d26u8;

        vst1_u8(dst, d6u8);
        dst += dst_pitch;
        vst1_u8(dst, d7u8);
        dst += dst_pitch;
        vst1_u8(dst, d8u8);
        dst += dst_pitch;
        vst1_u8(dst, d9u8);
        dst += dst_pitch;
      }
    }
    return;
  }

  // load first_pass filter
  dtmps8 = vld1_s8(vp8_sub_pel_filters[xoffset]);
  d0s8 = vdup_lane_s8(dtmps8, 0);
  d1s8 = vdup_lane_s8(dtmps8, 1);
  d2s8 = vdup_lane_s8(dtmps8, 2);
  d3s8 = vdup_lane_s8(dtmps8, 3);
  d4s8 = vdup_lane_s8(dtmps8, 4);
  d5s8 = vdup_lane_s8(dtmps8, 5);
  d0u8 = vreinterpret_u8_s8(vabs_s8(d0s8));
  d1u8 = vreinterpret_u8_s8(vabs_s8(d1s8));
  d2u8 = vreinterpret_u8_s8(vabs_s8(d2s8));
  d3u8 = vreinterpret_u8_s8(vabs_s8(d3s8));
  d4u8 = vreinterpret_u8_s8(vabs_s8(d4s8));
  d5u8 = vreinterpret_u8_s8(vabs_s8(d5s8));

  // First pass: output_height lines x output_width columns (9x4)
  if (yoffset == 0) {  // firstpass_filter4x4_only
    src = src_ptr - 2;
    dst = dst_ptr;
    for (i = 0; i < 8; ++i) {
      d6u8 = vld1_u8(src);
      d7u8 = vld1_u8(src + 8);
      d8u8 = vld1_u8(src + 16);
      src += src_pixels_per_line;
      d9u8 = vld1_u8(src);
      d10u8 = vld1_u8(src + 8);
      d11u8 = vld1_u8(src + 16);
      src += src_pixels_per_line;

      __builtin_prefetch(src);
      __builtin_prefetch(src + src_pixels_per_line);

      q6u16 = vmull_u8(d6u8, d0u8);
      q7u16 = vmull_u8(d7u8, d0u8);
      q8u16 = vmull_u8(d9u8, d0u8);
      q9u16 = vmull_u8(d10u8, d0u8);

      d20u8 = vext_u8(d6u8, d7u8, 1);
      d21u8 = vext_u8(d9u8, d10u8, 1);
      d22u8 = vext_u8(d7u8, d8u8, 1);
      d23u8 = vext_u8(d10u8, d11u8, 1);
      d24u8 = vext_u8(d6u8, d7u8, 4);
      d25u8 = vext_u8(d9u8, d10u8, 4);
      d26u8 = vext_u8(d7u8, d8u8, 4);
      d27u8 = vext_u8(d10u8, d11u8, 4);
      d28u8 = vext_u8(d6u8, d7u8, 5);
      d29u8 = vext_u8(d9u8, d10u8, 5);

      q6u16 = vmlsl_u8(q6u16, d20u8, d1u8);
      q8u16 = vmlsl_u8(q8u16, d21u8, d1u8);
      q7u16 = vmlsl_u8(q7u16, d22u8, d1u8);
      q9u16 = vmlsl_u8(q9u16, d23u8, d1u8);
      q6u16 = vmlsl_u8(q6u16, d24u8, d4u8);
      q8u16 = vmlsl_u8(q8u16, d25u8, d4u8);
      q7u16 = vmlsl_u8(q7u16, d26u8, d4u8);
      q9u16 = vmlsl_u8(q9u16, d27u8, d4u8);
      q6u16 = vmlal_u8(q6u16, d28u8, d5u8);
      q8u16 = vmlal_u8(q8u16, d29u8, d5u8);

      d20u8 = vext_u8(d7u8, d8u8, 5);
      d21u8 = vext_u8(d10u8, d11u8, 5);
      d22u8 = vext_u8(d6u8, d7u8, 2);
      d23u8 = vext_u8(d9u8, d10u8, 2);
      d24u8 = vext_u8(d7u8, d8u8, 2);
      d25u8 = vext_u8(d10u8, d11u8, 2);
      d26u8 = vext_u8(d6u8, d7u8, 3);
      d27u8 = vext_u8(d9u8, d10u8, 3);
      d28u8 = vext_u8(d7u8, d8u8, 3);
      d29u8 = vext_u8(d10u8, d11u8, 3);

      q7u16 = vmlal_u8(q7u16, d20u8, d5u8);
      q9u16 = vmlal_u8(q9u16, d21u8, d5u8);
      q6u16 = vmlal_u8(q6u16, d22u8, d2u8);
      q8u16 = vmlal_u8(q8u16, d23u8, d2u8);
      q7u16 = vmlal_u8(q7u16, d24u8, d2u8);
      q9u16 = vmlal_u8(q9u16, d25u8, d2u8);

      q10u16 = vmull_u8(d26u8, d3u8);
      q11u16 = vmull_u8(d27u8, d3u8);
      q12u16 = vmull_u8(d28u8, d3u8);
      q15u16 = vmull_u8(d29u8, d3u8);

      q6s16 = vreinterpretq_s16_u16(q6u16);
      q7s16 = vreinterpretq_s16_u16(q7u16);
      q8s16 = vreinterpretq_s16_u16(q8u16);
      q9s16 = vreinterpretq_s16_u16(q9u16);
      q10s16 = vreinterpretq_s16_u16(q10u16);
      q11s16 = vreinterpretq_s16_u16(q11u16);
      q12s16 = vreinterpretq_s16_u16(q12u16);
      q15s16 = vreinterpretq_s16_u16(q15u16);

      q6s16 = vqaddq_s16(q6s16, q10s16);
      q8s16 = vqaddq_s16(q8s16, q11s16);
      q7s16 = vqaddq_s16(q7s16, q12s16);
      q9s16 = vqaddq_s16(q9s16, q15s16);

      d6u8 = vqrshrun_n_s16(q6s16, 7);
      d7u8 = vqrshrun_n_s16(q7s16, 7);
      d8u8 = vqrshrun_n_s16(q8s16, 7);
      d9u8 = vqrshrun_n_s16(q9s16, 7);

      q3u8 = vcombine_u8(d6u8, d7u8);
      q4u8 = vcombine_u8(d8u8, d9u8);
      vst1q_u8(dst, q3u8);
      dst += dst_pitch;
      vst1q_u8(dst, q4u8);
      dst += dst_pitch;
    }
    return;
  }

  src = src_ptr - 2 - src_pixels_per_line * 2;
  tmpp = tmp;
  for (i = 0; i < 7; ++i) {
    d6u8 = vld1_u8(src);
    d7u8 = vld1_u8(src + 8);
    d8u8 = vld1_u8(src + 16);
    src += src_pixels_per_line;
    d9u8 = vld1_u8(src);
    d10u8 = vld1_u8(src + 8);
    d11u8 = vld1_u8(src + 16);
    src += src_pixels_per_line;
    d12u8 = vld1_u8(src);
    d13u8 = vld1_u8(src + 8);
    // Only 5 pixels are needed, avoid a potential out of bounds read.
    d14u8 = vld1_u8(src + 13);
    d14u8 = vext_u8(d14u8, d14u8, 3);
    src += src_pixels_per_line;

    __builtin_prefetch(src);
    __builtin_prefetch(src + src_pixels_per_line);
    __builtin_prefetch(src + src_pixels_per_line * 2);

    q8u16 = vmull_u8(d6u8, d0u8);
    q9u16 = vmull_u8(d7u8, d0u8);
    q10u16 = vmull_u8(d9u8, d0u8);
    q11u16 = vmull_u8(d10u8, d0u8);
    q12u16 = vmull_u8(d12u8, d0u8);
    q13u16 = vmull_u8(d13u8, d0u8);

    d28u8 = vext_u8(d6u8, d7u8, 1);
    d29u8 = vext_u8(d9u8, d10u8, 1);
    d30u8 = vext_u8(d12u8, d13u8, 1);
    q8u16 = vmlsl_u8(q8u16, d28u8, d1u8);
    q10u16 = vmlsl_u8(q10u16, d29u8, d1u8);
    q12u16 = vmlsl_u8(q12u16, d30u8, d1u8);
    d28u8 = vext_u8(d7u8, d8u8, 1);
    d29u8 = vext_u8(d10u8, d11u8, 1);
    d30u8 = vext_u8(d13u8, d14u8, 1);
    q9u16 = vmlsl_u8(q9u16, d28u8, d1u8);
    q11u16 = vmlsl_u8(q11u16, d29u8, d1u8);
    q13u16 = vmlsl_u8(q13u16, d30u8, d1u8);

    d28u8 = vext_u8(d6u8, d7u8, 4);
    d29u8 = vext_u8(d9u8, d10u8, 4);
    d30u8 = vext_u8(d12u8, d13u8, 4);
    q8u16 = vmlsl_u8(q8u16, d28u8, d4u8);
    q10u16 = vmlsl_u8(q10u16, d29u8, d4u8);
    q12u16 = vmlsl_u8(q12u16, d30u8, d4u8);
    d28u8 = vext_u8(d7u8, d8u8, 4);
    d29u8 = vext_u8(d10u8, d11u8, 4);
    d30u8 = vext_u8(d13u8, d14u8, 4);
    q9u16 = vmlsl_u8(q9u16, d28u8, d4u8);
    q11u16 = vmlsl_u8(q11u16, d29u8, d4u8);
    q13u16 = vmlsl_u8(q13u16, d30u8, d4u8);

    d28u8 = vext_u8(d6u8, d7u8, 5);
    d29u8 = vext_u8(d9u8, d10u8, 5);
    d30u8 = vext_u8(d12u8, d13u8, 5);
    q8u16 = vmlal_u8(q8u16, d28u8, d5u8);
    q10u16 = vmlal_u8(q10u16, d29u8, d5u8);
    q12u16 = vmlal_u8(q12u16, d30u8, d5u8);
    d28u8 = vext_u8(d7u8, d8u8, 5);
    d29u8 = vext_u8(d10u8, d11u8, 5);
    d30u8 = vext_u8(d13u8, d14u8, 5);
    q9u16 = vmlal_u8(q9u16, d28u8, d5u8);
    q11u16 = vmlal_u8(q11u16, d29u8, d5u8);
    q13u16 = vmlal_u8(q13u16, d30u8, d5u8);

    d28u8 = vext_u8(d6u8, d7u8, 2);
    d29u8 = vext_u8(d9u8, d10u8, 2);
    d30u8 = vext_u8(d12u8, d13u8, 2);
    q8u16 = vmlal_u8(q8u16, d28u8, d2u8);
    q10u16 = vmlal_u8(q10u16, d29u8, d2u8);
    q12u16 = vmlal_u8(q12u16, d30u8, d2u8);
    d28u8 = vext_u8(d7u8, d8u8, 2);
    d29u8 = vext_u8(d10u8, d11u8, 2);
    d30u8 = vext_u8(d13u8, d14u8, 2);
    q9u16 = vmlal_u8(q9u16, d28u8, d2u8);
    q11u16 = vmlal_u8(q11u16, d29u8, d2u8);
    q13u16 = vmlal_u8(q13u16, d30u8, d2u8);

    d28u8 = vext_u8(d6u8, d7u8, 3);
    d29u8 = vext_u8(d9u8, d10u8, 3);
    d30u8 = vext_u8(d12u8, d13u8, 3);
    d15u8 = vext_u8(d7u8, d8u8, 3);
    d31u8 = vext_u8(d10u8, d11u8, 3);
    d6u8 = vext_u8(d13u8, d14u8, 3);
    q4u16 = vmull_u8(d28u8, d3u8);
    q5u16 = vmull_u8(d29u8, d3u8);
    q6u16 = vmull_u8(d30u8, d3u8);
    q4s16 = vreinterpretq_s16_u16(q4u16);
    q5s16 = vreinterpretq_s16_u16(q5u16);
    q6s16 = vreinterpretq_s16_u16(q6u16);
    q8s16 = vreinterpretq_s16_u16(q8u16);
    q10s16 = vreinterpretq_s16_u16(q10u16);
    q12s16 = vreinterpretq_s16_u16(q12u16);
    q8s16 = vqaddq_s16(q8s16, q4s16);
    q10s16 = vqaddq_s16(q10s16, q5s16);
    q12s16 = vqaddq_s16(q12s16, q6s16);

    q6u16 = vmull_u8(d15u8, d3u8);
    q7u16 = vmull_u8(d31u8, d3u8);
    q3u16 = vmull_u8(d6u8, d3u8);
    q3s16 = vreinterpretq_s16_u16(q3u16);
    q6s16 = vreinterpretq_s16_u16(q6u16);
    q7s16 = vreinterpretq_s16_u16(q7u16);
    q9s16 = vreinterpretq_s16_u16(q9u16);
    q11s16 = vreinterpretq_s16_u16(q11u16);
    q13s16 = vreinterpretq_s16_u16(q13u16);
    q9s16 = vqaddq_s16(q9s16, q6s16);
    q11s16 = vqaddq_s16(q11s16, q7s16);
    q13s16 = vqaddq_s16(q13s16, q3s16);

    d6u8 = vqrshrun_n_s16(q8s16, 7);
    d7u8 = vqrshrun_n_s16(q9s16, 7);
    d8u8 = vqrshrun_n_s16(q10s16, 7);
    d9u8 = vqrshrun_n_s16(q11s16, 7);
    d10u8 = vqrshrun_n_s16(q12s16, 7);
    d11u8 = vqrshrun_n_s16(q13s16, 7);

    vst1_u8(tmpp, d6u8);
    tmpp += 8;
    vst1_u8(tmpp, d7u8);
    tmpp += 8;
    vst1_u8(tmpp, d8u8);
    tmpp += 8;
    vst1_u8(tmpp, d9u8);
    tmpp += 8;
    vst1_u8(tmpp, d10u8);
    tmpp += 8;
    vst1_u8(tmpp, d11u8);
    tmpp += 8;
  }

  // Second pass: 16x16
  dtmps8 = vld1_s8(vp8_sub_pel_filters[yoffset]);
  d0s8 = vdup_lane_s8(dtmps8, 0);
  d1s8 = vdup_lane_s8(dtmps8, 1);
  d2s8 = vdup_lane_s8(dtmps8, 2);
  d3s8 = vdup_lane_s8(dtmps8, 3);
  d4s8 = vdup_lane_s8(dtmps8, 4);
  d5s8 = vdup_lane_s8(dtmps8, 5);
  d0u8 = vreinterpret_u8_s8(vabs_s8(d0s8));
  d1u8 = vreinterpret_u8_s8(vabs_s8(d1s8));
  d2u8 = vreinterpret_u8_s8(vabs_s8(d2s8));
  d3u8 = vreinterpret_u8_s8(vabs_s8(d3s8));
  d4u8 = vreinterpret_u8_s8(vabs_s8(d4s8));
  d5u8 = vreinterpret_u8_s8(vabs_s8(d5s8));

  for (i = 0; i < 2; ++i) {
    dst = dst_ptr + 8 * i;
    tmpp = tmp + 8 * i;
    d18u8 = vld1_u8(tmpp);
    tmpp += 16;
    d19u8 = vld1_u8(tmpp);
    tmpp += 16;
    d20u8 = vld1_u8(tmpp);
    tmpp += 16;
    d21u8 = vld1_u8(tmpp);
    tmpp += 16;
    d22u8 = vld1_u8(tmpp);
    tmpp += 16;
    for (j = 0; j < 4; ++j) {
      d23u8 = vld1_u8(tmpp);
      tmpp += 16;
      d24u8 = vld1_u8(tmpp);
      tmpp += 16;
      d25u8 = vld1_u8(tmpp);
      tmpp += 16;
      d26u8 = vld1_u8(tmpp);
      tmpp += 16;

      q3u16 = vmull_u8(d18u8, d0u8);
      q4u16 = vmull_u8(d19u8, d0u8);
      q5u16 = vmull_u8(d20u8, d0u8);
      q6u16 = vmull_u8(d21u8, d0u8);

      q3u16 = vmlsl_u8(q3u16, d19u8, d1u8);
      q4u16 = vmlsl_u8(q4u16, d20u8, d1u8);
      q5u16 = vmlsl_u8(q5u16, d21u8, d1u8);
      q6u16 = vmlsl_u8(q6u16, d22u8, d1u8);

      q3u16 = vmlsl_u8(q3u16, d22u8, d4u8);
      q4u16 = vmlsl_u8(q4u16, d23u8, d4u8);
      q5u16 = vmlsl_u8(q5u16, d24u8, d4u8);
      q6u16 = vmlsl_u8(q6u16, d25u8, d4u8);

      q3u16 = vmlal_u8(q3u16, d20u8, d2u8);
      q4u16 = vmlal_u8(q4u16, d21u8, d2u8);
      q5u16 = vmlal_u8(q5u16, d22u8, d2u8);
      q6u16 = vmlal_u8(q6u16, d23u8, d2u8);

      q3u16 = vmlal_u8(q3u16, d23u8, d5u8);
      q4u16 = vmlal_u8(q4u16, d24u8, d5u8);
      q5u16 = vmlal_u8(q5u16, d25u8, d5u8);
      q6u16 = vmlal_u8(q6u16, d26u8, d5u8);

      q7u16 = vmull_u8(d21u8, d3u8);
      q8u16 = vmull_u8(d22u8, d3u8);
      q9u16 = vmull_u8(d23u8, d3u8);
      q10u16 = vmull_u8(d24u8, d3u8);

      q3s16 = vreinterpretq_s16_u16(q3u16);
      q4s16 = vreinterpretq_s16_u16(q4u16);
      q5s16 = vreinterpretq_s16_u16(q5u16);
      q6s16 = vreinterpretq_s16_u16(q6u16);
      q7s16 = vreinterpretq_s16_u16(q7u16);
      q8s16 = vreinterpretq_s16_u16(q8u16);
      q9s16 = vreinterpretq_s16_u16(q9u16);
      q10s16 = vreinterpretq_s16_u16(q10u16);

      q7s16 = vqaddq_s16(q7s16, q3s16);
      q8s16 = vqaddq_s16(q8s16, q4s16);
      q9s16 = vqaddq_s16(q9s16, q5s16);
      q10s16 = vqaddq_s16(q10s16, q6s16);

      d6u8 = vqrshrun_n_s16(q7s16, 7);
      d7u8 = vqrshrun_n_s16(q8s16, 7);
      d8u8 = vqrshrun_n_s16(q9s16, 7);
      d9u8 = vqrshrun_n_s16(q10s16, 7);

      d18u8 = d22u8;
      d19u8 = d23u8;
      d20u8 = d24u8;
      d21u8 = d25u8;
      d22u8 = d26u8;

      vst1_u8(dst, d6u8);
      dst += dst_pitch;
      vst1_u8(dst, d7u8);
      dst += dst_pitch;
      vst1_u8(dst, d8u8);
      dst += dst_pitch;
      vst1_u8(dst, d9u8);
      dst += dst_pitch;
    }
  }
}
