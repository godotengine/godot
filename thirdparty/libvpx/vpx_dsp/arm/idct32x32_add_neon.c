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

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/arm/idct_neon.h"
#include "vpx_dsp/arm/mem_neon.h"
#include "vpx_dsp/arm/transpose_neon.h"
#include "vpx_dsp/txfm_common.h"

static INLINE void load_from_transformed(const int16_t *const trans_buf,
                                         const int first, const int second,
                                         int16x8_t *const q0,
                                         int16x8_t *const q1) {
  *q0 = vld1q_s16(trans_buf + first * 8);
  *q1 = vld1q_s16(trans_buf + second * 8);
}

static INLINE void load_from_output(const int16_t *const out, const int first,
                                    const int second, int16x8_t *const q0,
                                    int16x8_t *const q1) {
  *q0 = vld1q_s16(out + first * 32);
  *q1 = vld1q_s16(out + second * 32);
}

static INLINE void store_in_output(int16_t *const out, const int first,
                                   const int second, const int16x8_t q0,
                                   const int16x8_t q1) {
  vst1q_s16(out + first * 32, q0);
  vst1q_s16(out + second * 32, q1);
}

static INLINE void store_combine_results(uint8_t *p1, uint8_t *p2,
                                         const int stride, int16x8_t q0,
                                         int16x8_t q1, int16x8_t q2,
                                         int16x8_t q3) {
  uint8x8_t d[4];

  d[0] = vld1_u8(p1);
  p1 += stride;
  d[1] = vld1_u8(p1);
  d[3] = vld1_u8(p2);
  p2 -= stride;
  d[2] = vld1_u8(p2);

  q0 = vrshrq_n_s16(q0, 6);
  q1 = vrshrq_n_s16(q1, 6);
  q2 = vrshrq_n_s16(q2, 6);
  q3 = vrshrq_n_s16(q3, 6);

  q0 = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(q0), d[0]));
  q1 = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(q1), d[1]));
  q2 = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(q2), d[2]));
  q3 = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(q3), d[3]));

  d[0] = vqmovun_s16(q0);
  d[1] = vqmovun_s16(q1);
  d[2] = vqmovun_s16(q2);
  d[3] = vqmovun_s16(q3);

  vst1_u8(p1, d[1]);
  p1 -= stride;
  vst1_u8(p1, d[0]);
  vst1_u8(p2, d[2]);
  p2 += stride;
  vst1_u8(p2, d[3]);
}

static INLINE void highbd_store_combine_results_bd8(uint16_t *p1, uint16_t *p2,
                                                    const int stride,
                                                    int16x8_t q0, int16x8_t q1,
                                                    int16x8_t q2,
                                                    int16x8_t q3) {
  uint16x8_t d[4];

  d[0] = vld1q_u16(p1);
  p1 += stride;
  d[1] = vld1q_u16(p1);
  d[3] = vld1q_u16(p2);
  p2 -= stride;
  d[2] = vld1q_u16(p2);

  q0 = vrshrq_n_s16(q0, 6);
  q1 = vrshrq_n_s16(q1, 6);
  q2 = vrshrq_n_s16(q2, 6);
  q3 = vrshrq_n_s16(q3, 6);

  q0 = vaddq_s16(q0, vreinterpretq_s16_u16(d[0]));
  q1 = vaddq_s16(q1, vreinterpretq_s16_u16(d[1]));
  q2 = vaddq_s16(q2, vreinterpretq_s16_u16(d[2]));
  q3 = vaddq_s16(q3, vreinterpretq_s16_u16(d[3]));

  d[0] = vmovl_u8(vqmovun_s16(q0));
  d[1] = vmovl_u8(vqmovun_s16(q1));
  d[2] = vmovl_u8(vqmovun_s16(q2));
  d[3] = vmovl_u8(vqmovun_s16(q3));

  vst1q_u16(p1, d[1]);
  p1 -= stride;
  vst1q_u16(p1, d[0]);
  vst1q_u16(p2, d[2]);
  p2 += stride;
  vst1q_u16(p2, d[3]);
}

static INLINE void do_butterfly(const int16x8_t qIn0, const int16x8_t qIn1,
                                const int16_t first_const,
                                const int16_t second_const,
                                int16x8_t *const qOut0,
                                int16x8_t *const qOut1) {
  int32x4_t q[4];
  int16x4_t d[6];

  d[0] = vget_low_s16(qIn0);
  d[1] = vget_high_s16(qIn0);
  d[2] = vget_low_s16(qIn1);
  d[3] = vget_high_s16(qIn1);

  // Note: using v{mul, mla, mls}l_n_s16 here slows down 35% with gcc 4.9.
  d[4] = vdup_n_s16(first_const);
  d[5] = vdup_n_s16(second_const);

  q[0] = vmull_s16(d[0], d[4]);
  q[1] = vmull_s16(d[1], d[4]);
  q[0] = vmlsl_s16(q[0], d[2], d[5]);
  q[1] = vmlsl_s16(q[1], d[3], d[5]);

  q[2] = vmull_s16(d[0], d[5]);
  q[3] = vmull_s16(d[1], d[5]);
  q[2] = vmlal_s16(q[2], d[2], d[4]);
  q[3] = vmlal_s16(q[3], d[3], d[4]);

  *qOut0 = vcombine_s16(vrshrn_n_s32(q[0], DCT_CONST_BITS),
                        vrshrn_n_s32(q[1], DCT_CONST_BITS));
  *qOut1 = vcombine_s16(vrshrn_n_s32(q[2], DCT_CONST_BITS),
                        vrshrn_n_s32(q[3], DCT_CONST_BITS));
}

static INLINE void load_s16x8q(const int16_t *in, int16x8_t *const s0,
                               int16x8_t *const s1, int16x8_t *const s2,
                               int16x8_t *const s3, int16x8_t *const s4,
                               int16x8_t *const s5, int16x8_t *const s6,
                               int16x8_t *const s7) {
  *s0 = vld1q_s16(in);
  in += 32;
  *s1 = vld1q_s16(in);
  in += 32;
  *s2 = vld1q_s16(in);
  in += 32;
  *s3 = vld1q_s16(in);
  in += 32;
  *s4 = vld1q_s16(in);
  in += 32;
  *s5 = vld1q_s16(in);
  in += 32;
  *s6 = vld1q_s16(in);
  in += 32;
  *s7 = vld1q_s16(in);
}

static INLINE void transpose_and_store_s16_8x8(int16x8_t a0, int16x8_t a1,
                                               int16x8_t a2, int16x8_t a3,
                                               int16x8_t a4, int16x8_t a5,
                                               int16x8_t a6, int16x8_t a7,
                                               int16_t **out) {
  transpose_s16_8x8(&a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7);

  vst1q_s16(*out, a0);
  *out += 8;
  vst1q_s16(*out, a1);
  *out += 8;
  vst1q_s16(*out, a2);
  *out += 8;
  vst1q_s16(*out, a3);
  *out += 8;
  vst1q_s16(*out, a4);
  *out += 8;
  vst1q_s16(*out, a5);
  *out += 8;
  vst1q_s16(*out, a6);
  *out += 8;
  vst1q_s16(*out, a7);
  *out += 8;
}

static INLINE void idct32_transpose_pair(const int16_t *input, int16_t *t_buf) {
  int i;
  int16x8_t s0, s1, s2, s3, s4, s5, s6, s7;

  for (i = 0; i < 4; i++, input += 8) {
    load_s16x8q(input, &s0, &s1, &s2, &s3, &s4, &s5, &s6, &s7);
    transpose_and_store_s16_8x8(s0, s1, s2, s3, s4, s5, s6, s7, &t_buf);
  }
}

#if CONFIG_VP9_HIGHBITDEPTH
static INLINE void load_s16x8q_tran_low(
    const tran_low_t *in, int16x8_t *const s0, int16x8_t *const s1,
    int16x8_t *const s2, int16x8_t *const s3, int16x8_t *const s4,
    int16x8_t *const s5, int16x8_t *const s6, int16x8_t *const s7) {
  *s0 = load_tran_low_to_s16q(in);
  in += 32;
  *s1 = load_tran_low_to_s16q(in);
  in += 32;
  *s2 = load_tran_low_to_s16q(in);
  in += 32;
  *s3 = load_tran_low_to_s16q(in);
  in += 32;
  *s4 = load_tran_low_to_s16q(in);
  in += 32;
  *s5 = load_tran_low_to_s16q(in);
  in += 32;
  *s6 = load_tran_low_to_s16q(in);
  in += 32;
  *s7 = load_tran_low_to_s16q(in);
}

static INLINE void idct32_transpose_pair_tran_low(const tran_low_t *input,
                                                  int16_t *t_buf) {
  int i;
  int16x8_t s0, s1, s2, s3, s4, s5, s6, s7;

  for (i = 0; i < 4; i++, input += 8) {
    load_s16x8q_tran_low(input, &s0, &s1, &s2, &s3, &s4, &s5, &s6, &s7);
    transpose_and_store_s16_8x8(s0, s1, s2, s3, s4, s5, s6, s7, &t_buf);
  }
}
#else  // !CONFIG_VP9_HIGHBITDEPTH
#define idct32_transpose_pair_tran_low idct32_transpose_pair
#endif  // CONFIG_VP9_HIGHBITDEPTH

static INLINE void idct32_bands_end_1st_pass(int16_t *const out,
                                             int16x8_t *const q) {
  store_in_output(out, 16, 17, q[6], q[7]);
  store_in_output(out, 14, 15, q[8], q[9]);

  load_from_output(out, 30, 31, &q[0], &q[1]);
  q[4] = vaddq_s16(q[2], q[1]);
  q[5] = vaddq_s16(q[3], q[0]);
  q[6] = vsubq_s16(q[3], q[0]);
  q[7] = vsubq_s16(q[2], q[1]);
  store_in_output(out, 30, 31, q[6], q[7]);
  store_in_output(out, 0, 1, q[4], q[5]);

  load_from_output(out, 12, 13, &q[0], &q[1]);
  q[2] = vaddq_s16(q[10], q[1]);
  q[3] = vaddq_s16(q[11], q[0]);
  q[4] = vsubq_s16(q[11], q[0]);
  q[5] = vsubq_s16(q[10], q[1]);

  load_from_output(out, 18, 19, &q[0], &q[1]);
  q[8] = vaddq_s16(q[4], q[1]);
  q[9] = vaddq_s16(q[5], q[0]);
  q[6] = vsubq_s16(q[5], q[0]);
  q[7] = vsubq_s16(q[4], q[1]);
  store_in_output(out, 18, 19, q[6], q[7]);
  store_in_output(out, 12, 13, q[8], q[9]);

  load_from_output(out, 28, 29, &q[0], &q[1]);
  q[4] = vaddq_s16(q[2], q[1]);
  q[5] = vaddq_s16(q[3], q[0]);
  q[6] = vsubq_s16(q[3], q[0]);
  q[7] = vsubq_s16(q[2], q[1]);
  store_in_output(out, 28, 29, q[6], q[7]);
  store_in_output(out, 2, 3, q[4], q[5]);

  load_from_output(out, 10, 11, &q[0], &q[1]);
  q[2] = vaddq_s16(q[12], q[1]);
  q[3] = vaddq_s16(q[13], q[0]);
  q[4] = vsubq_s16(q[13], q[0]);
  q[5] = vsubq_s16(q[12], q[1]);

  load_from_output(out, 20, 21, &q[0], &q[1]);
  q[8] = vaddq_s16(q[4], q[1]);
  q[9] = vaddq_s16(q[5], q[0]);
  q[6] = vsubq_s16(q[5], q[0]);
  q[7] = vsubq_s16(q[4], q[1]);
  store_in_output(out, 20, 21, q[6], q[7]);
  store_in_output(out, 10, 11, q[8], q[9]);

  load_from_output(out, 26, 27, &q[0], &q[1]);
  q[4] = vaddq_s16(q[2], q[1]);
  q[5] = vaddq_s16(q[3], q[0]);
  q[6] = vsubq_s16(q[3], q[0]);
  q[7] = vsubq_s16(q[2], q[1]);
  store_in_output(out, 26, 27, q[6], q[7]);
  store_in_output(out, 4, 5, q[4], q[5]);

  load_from_output(out, 8, 9, &q[0], &q[1]);
  q[2] = vaddq_s16(q[14], q[1]);
  q[3] = vaddq_s16(q[15], q[0]);
  q[4] = vsubq_s16(q[15], q[0]);
  q[5] = vsubq_s16(q[14], q[1]);

  load_from_output(out, 22, 23, &q[0], &q[1]);
  q[8] = vaddq_s16(q[4], q[1]);
  q[9] = vaddq_s16(q[5], q[0]);
  q[6] = vsubq_s16(q[5], q[0]);
  q[7] = vsubq_s16(q[4], q[1]);
  store_in_output(out, 22, 23, q[6], q[7]);
  store_in_output(out, 8, 9, q[8], q[9]);

  load_from_output(out, 24, 25, &q[0], &q[1]);
  q[4] = vaddq_s16(q[2], q[1]);
  q[5] = vaddq_s16(q[3], q[0]);
  q[6] = vsubq_s16(q[3], q[0]);
  q[7] = vsubq_s16(q[2], q[1]);
  store_in_output(out, 24, 25, q[6], q[7]);
  store_in_output(out, 6, 7, q[4], q[5]);
}

static INLINE void idct32_bands_end_2nd_pass(const int16_t *const out,
                                             uint8_t *const dest,
                                             const int stride,
                                             int16x8_t *const q) {
  uint8_t *dest0 = dest + 0 * stride;
  uint8_t *dest1 = dest + 31 * stride;
  uint8_t *dest2 = dest + 16 * stride;
  uint8_t *dest3 = dest + 15 * stride;
  const int str2 = stride << 1;

  store_combine_results(dest2, dest3, stride, q[6], q[7], q[8], q[9]);
  dest2 += str2;
  dest3 -= str2;

  load_from_output(out, 30, 31, &q[0], &q[1]);
  q[4] = final_add(q[2], q[1]);
  q[5] = final_add(q[3], q[0]);
  q[6] = final_sub(q[3], q[0]);
  q[7] = final_sub(q[2], q[1]);
  store_combine_results(dest0, dest1, stride, q[4], q[5], q[6], q[7]);
  dest0 += str2;
  dest1 -= str2;

  load_from_output(out, 12, 13, &q[0], &q[1]);
  q[2] = vaddq_s16(q[10], q[1]);
  q[3] = vaddq_s16(q[11], q[0]);
  q[4] = vsubq_s16(q[11], q[0]);
  q[5] = vsubq_s16(q[10], q[1]);

  load_from_output(out, 18, 19, &q[0], &q[1]);
  q[8] = final_add(q[4], q[1]);
  q[9] = final_add(q[5], q[0]);
  q[6] = final_sub(q[5], q[0]);
  q[7] = final_sub(q[4], q[1]);
  store_combine_results(dest2, dest3, stride, q[6], q[7], q[8], q[9]);
  dest2 += str2;
  dest3 -= str2;

  load_from_output(out, 28, 29, &q[0], &q[1]);
  q[4] = final_add(q[2], q[1]);
  q[5] = final_add(q[3], q[0]);
  q[6] = final_sub(q[3], q[0]);
  q[7] = final_sub(q[2], q[1]);
  store_combine_results(dest0, dest1, stride, q[4], q[5], q[6], q[7]);
  dest0 += str2;
  dest1 -= str2;

  load_from_output(out, 10, 11, &q[0], &q[1]);
  q[2] = vaddq_s16(q[12], q[1]);
  q[3] = vaddq_s16(q[13], q[0]);
  q[4] = vsubq_s16(q[13], q[0]);
  q[5] = vsubq_s16(q[12], q[1]);

  load_from_output(out, 20, 21, &q[0], &q[1]);
  q[8] = final_add(q[4], q[1]);
  q[9] = final_add(q[5], q[0]);
  q[6] = final_sub(q[5], q[0]);
  q[7] = final_sub(q[4], q[1]);
  store_combine_results(dest2, dest3, stride, q[6], q[7], q[8], q[9]);
  dest2 += str2;
  dest3 -= str2;

  load_from_output(out, 26, 27, &q[0], &q[1]);
  q[4] = final_add(q[2], q[1]);
  q[5] = final_add(q[3], q[0]);
  q[6] = final_sub(q[3], q[0]);
  q[7] = final_sub(q[2], q[1]);
  store_combine_results(dest0, dest1, stride, q[4], q[5], q[6], q[7]);
  dest0 += str2;
  dest1 -= str2;

  load_from_output(out, 8, 9, &q[0], &q[1]);
  q[2] = vaddq_s16(q[14], q[1]);
  q[3] = vaddq_s16(q[15], q[0]);
  q[4] = vsubq_s16(q[15], q[0]);
  q[5] = vsubq_s16(q[14], q[1]);

  load_from_output(out, 22, 23, &q[0], &q[1]);
  q[8] = final_add(q[4], q[1]);
  q[9] = final_add(q[5], q[0]);
  q[6] = final_sub(q[5], q[0]);
  q[7] = final_sub(q[4], q[1]);
  store_combine_results(dest2, dest3, stride, q[6], q[7], q[8], q[9]);

  load_from_output(out, 24, 25, &q[0], &q[1]);
  q[4] = final_add(q[2], q[1]);
  q[5] = final_add(q[3], q[0]);
  q[6] = final_sub(q[3], q[0]);
  q[7] = final_sub(q[2], q[1]);
  store_combine_results(dest0, dest1, stride, q[4], q[5], q[6], q[7]);
}

static INLINE void highbd_idct32_bands_end_2nd_pass_bd8(
    const int16_t *const out, uint16_t *const dest, const int stride,
    int16x8_t *const q) {
  uint16_t *dest0 = dest + 0 * stride;
  uint16_t *dest1 = dest + 31 * stride;
  uint16_t *dest2 = dest + 16 * stride;
  uint16_t *dest3 = dest + 15 * stride;
  const int str2 = stride << 1;

  highbd_store_combine_results_bd8(dest2, dest3, stride, q[6], q[7], q[8],
                                   q[9]);
  dest2 += str2;
  dest3 -= str2;

  load_from_output(out, 30, 31, &q[0], &q[1]);
  q[4] = final_add(q[2], q[1]);
  q[5] = final_add(q[3], q[0]);
  q[6] = final_sub(q[3], q[0]);
  q[7] = final_sub(q[2], q[1]);
  highbd_store_combine_results_bd8(dest0, dest1, stride, q[4], q[5], q[6],
                                   q[7]);
  dest0 += str2;
  dest1 -= str2;

  load_from_output(out, 12, 13, &q[0], &q[1]);
  q[2] = vaddq_s16(q[10], q[1]);
  q[3] = vaddq_s16(q[11], q[0]);
  q[4] = vsubq_s16(q[11], q[0]);
  q[5] = vsubq_s16(q[10], q[1]);

  load_from_output(out, 18, 19, &q[0], &q[1]);
  q[8] = final_add(q[4], q[1]);
  q[9] = final_add(q[5], q[0]);
  q[6] = final_sub(q[5], q[0]);
  q[7] = final_sub(q[4], q[1]);
  highbd_store_combine_results_bd8(dest2, dest3, stride, q[6], q[7], q[8],
                                   q[9]);
  dest2 += str2;
  dest3 -= str2;

  load_from_output(out, 28, 29, &q[0], &q[1]);
  q[4] = final_add(q[2], q[1]);
  q[5] = final_add(q[3], q[0]);
  q[6] = final_sub(q[3], q[0]);
  q[7] = final_sub(q[2], q[1]);
  highbd_store_combine_results_bd8(dest0, dest1, stride, q[4], q[5], q[6],
                                   q[7]);
  dest0 += str2;
  dest1 -= str2;

  load_from_output(out, 10, 11, &q[0], &q[1]);
  q[2] = vaddq_s16(q[12], q[1]);
  q[3] = vaddq_s16(q[13], q[0]);
  q[4] = vsubq_s16(q[13], q[0]);
  q[5] = vsubq_s16(q[12], q[1]);

  load_from_output(out, 20, 21, &q[0], &q[1]);
  q[8] = final_add(q[4], q[1]);
  q[9] = final_add(q[5], q[0]);
  q[6] = final_sub(q[5], q[0]);
  q[7] = final_sub(q[4], q[1]);
  highbd_store_combine_results_bd8(dest2, dest3, stride, q[6], q[7], q[8],
                                   q[9]);
  dest2 += str2;
  dest3 -= str2;

  load_from_output(out, 26, 27, &q[0], &q[1]);
  q[4] = final_add(q[2], q[1]);
  q[5] = final_add(q[3], q[0]);
  q[6] = final_sub(q[3], q[0]);
  q[7] = final_sub(q[2], q[1]);
  highbd_store_combine_results_bd8(dest0, dest1, stride, q[4], q[5], q[6],
                                   q[7]);
  dest0 += str2;
  dest1 -= str2;

  load_from_output(out, 8, 9, &q[0], &q[1]);
  q[2] = vaddq_s16(q[14], q[1]);
  q[3] = vaddq_s16(q[15], q[0]);
  q[4] = vsubq_s16(q[15], q[0]);
  q[5] = vsubq_s16(q[14], q[1]);

  load_from_output(out, 22, 23, &q[0], &q[1]);
  q[8] = final_add(q[4], q[1]);
  q[9] = final_add(q[5], q[0]);
  q[6] = final_sub(q[5], q[0]);
  q[7] = final_sub(q[4], q[1]);
  highbd_store_combine_results_bd8(dest2, dest3, stride, q[6], q[7], q[8],
                                   q[9]);

  load_from_output(out, 24, 25, &q[0], &q[1]);
  q[4] = final_add(q[2], q[1]);
  q[5] = final_add(q[3], q[0]);
  q[6] = final_sub(q[3], q[0]);
  q[7] = final_sub(q[2], q[1]);
  highbd_store_combine_results_bd8(dest0, dest1, stride, q[4], q[5], q[6],
                                   q[7]);
}

void vpx_idct32_32_neon(const tran_low_t *input, uint8_t *dest,
                        const int stride, const int highbd_flag) {
  int i, idct32_pass_loop;
  int16_t trans_buf[32 * 8];
  int16_t pass1[32 * 32];
  int16_t pass2[32 * 32];
  const int16_t *input_pass2 = pass1;  // input of pass2 is the result of pass1
  int16_t *out;
  int16x8_t q[16];
  uint16_t *dst = CAST_TO_SHORTPTR(dest);

  for (idct32_pass_loop = 0, out = pass1; idct32_pass_loop < 2;
       idct32_pass_loop++, out = pass2) {
    for (i = 0; i < 4; i++, out += 8) {  // idct32_bands_loop
      if (idct32_pass_loop == 0) {
        idct32_transpose_pair_tran_low(input, trans_buf);
        input += 32 * 8;
      } else {
        idct32_transpose_pair(input_pass2, trans_buf);
        input_pass2 += 32 * 8;
      }

      // -----------------------------------------
      // BLOCK A: 16-19,28-31
      // -----------------------------------------
      // generate 16,17,30,31
      // part of stage 1
      load_from_transformed(trans_buf, 1, 31, &q[14], &q[13]);
      do_butterfly(q[14], q[13], cospi_31_64, cospi_1_64, &q[0], &q[2]);
      load_from_transformed(trans_buf, 17, 15, &q[14], &q[13]);
      do_butterfly(q[14], q[13], cospi_15_64, cospi_17_64, &q[1], &q[3]);
      // part of stage 2
      q[4] = vaddq_s16(q[0], q[1]);
      q[13] = vsubq_s16(q[0], q[1]);
      q[6] = vaddq_s16(q[2], q[3]);
      q[14] = vsubq_s16(q[2], q[3]);
      // part of stage 3
      do_butterfly(q[14], q[13], cospi_28_64, cospi_4_64, &q[5], &q[7]);

      // generate 18,19,28,29
      // part of stage 1
      load_from_transformed(trans_buf, 9, 23, &q[14], &q[13]);
      do_butterfly(q[14], q[13], cospi_23_64, cospi_9_64, &q[0], &q[2]);
      load_from_transformed(trans_buf, 25, 7, &q[14], &q[13]);
      do_butterfly(q[14], q[13], cospi_7_64, cospi_25_64, &q[1], &q[3]);
      // part of stage 2
      q[13] = vsubq_s16(q[3], q[2]);
      q[3] = vaddq_s16(q[3], q[2]);
      q[14] = vsubq_s16(q[1], q[0]);
      q[2] = vaddq_s16(q[1], q[0]);
      // part of stage 3
      do_butterfly(q[14], q[13], -cospi_4_64, -cospi_28_64, &q[1], &q[0]);
      // part of stage 4
      q[8] = vaddq_s16(q[4], q[2]);
      q[9] = vaddq_s16(q[5], q[0]);
      q[10] = vaddq_s16(q[7], q[1]);
      q[15] = vaddq_s16(q[6], q[3]);
      q[13] = vsubq_s16(q[5], q[0]);
      q[14] = vsubq_s16(q[7], q[1]);
      store_in_output(out, 16, 31, q[8], q[15]);
      store_in_output(out, 17, 30, q[9], q[10]);
      // part of stage 5
      do_butterfly(q[14], q[13], cospi_24_64, cospi_8_64, &q[0], &q[1]);
      store_in_output(out, 29, 18, q[1], q[0]);
      // part of stage 4
      q[13] = vsubq_s16(q[4], q[2]);
      q[14] = vsubq_s16(q[6], q[3]);
      // part of stage 5
      do_butterfly(q[14], q[13], cospi_24_64, cospi_8_64, &q[4], &q[6]);
      store_in_output(out, 19, 28, q[4], q[6]);

      // -----------------------------------------
      // BLOCK B: 20-23,24-27
      // -----------------------------------------
      // generate 20,21,26,27
      // part of stage 1
      load_from_transformed(trans_buf, 5, 27, &q[14], &q[13]);
      do_butterfly(q[14], q[13], cospi_27_64, cospi_5_64, &q[0], &q[2]);
      load_from_transformed(trans_buf, 21, 11, &q[14], &q[13]);
      do_butterfly(q[14], q[13], cospi_11_64, cospi_21_64, &q[1], &q[3]);
      // part of stage 2
      q[13] = vsubq_s16(q[0], q[1]);
      q[0] = vaddq_s16(q[0], q[1]);
      q[14] = vsubq_s16(q[2], q[3]);
      q[2] = vaddq_s16(q[2], q[3]);
      // part of stage 3
      do_butterfly(q[14], q[13], cospi_12_64, cospi_20_64, &q[1], &q[3]);

      // generate 22,23,24,25
      // part of stage 1
      load_from_transformed(trans_buf, 13, 19, &q[14], &q[13]);
      do_butterfly(q[14], q[13], cospi_19_64, cospi_13_64, &q[5], &q[7]);
      load_from_transformed(trans_buf, 29, 3, &q[14], &q[13]);
      do_butterfly(q[14], q[13], cospi_3_64, cospi_29_64, &q[4], &q[6]);
      // part of stage 2
      q[14] = vsubq_s16(q[4], q[5]);
      q[5] = vaddq_s16(q[4], q[5]);
      q[13] = vsubq_s16(q[6], q[7]);
      q[6] = vaddq_s16(q[6], q[7]);
      // part of stage 3
      do_butterfly(q[14], q[13], -cospi_20_64, -cospi_12_64, &q[4], &q[7]);
      // part of stage 4
      q[10] = vaddq_s16(q[7], q[1]);
      q[11] = vaddq_s16(q[5], q[0]);
      q[12] = vaddq_s16(q[6], q[2]);
      q[15] = vaddq_s16(q[4], q[3]);
      // part of stage 6
      load_from_output(out, 16, 17, &q[14], &q[13]);
      q[8] = vaddq_s16(q[14], q[11]);
      q[9] = vaddq_s16(q[13], q[10]);
      q[13] = vsubq_s16(q[13], q[10]);
      q[11] = vsubq_s16(q[14], q[11]);
      store_in_output(out, 17, 16, q[9], q[8]);
      load_from_output(out, 30, 31, &q[14], &q[9]);
      q[8] = vsubq_s16(q[9], q[12]);
      q[10] = vaddq_s16(q[14], q[15]);
      q[14] = vsubq_s16(q[14], q[15]);
      q[12] = vaddq_s16(q[9], q[12]);
      store_in_output(out, 30, 31, q[10], q[12]);
      // part of stage 7
      do_butterfly(q[14], q[13], cospi_16_64, cospi_16_64, &q[13], &q[14]);
      store_in_output(out, 25, 22, q[14], q[13]);
      do_butterfly(q[8], q[11], cospi_16_64, cospi_16_64, &q[13], &q[14]);
      store_in_output(out, 24, 23, q[14], q[13]);
      // part of stage 4
      q[14] = vsubq_s16(q[5], q[0]);
      q[13] = vsubq_s16(q[6], q[2]);
      do_butterfly(q[14], q[13], -cospi_8_64, -cospi_24_64, &q[5], &q[6]);
      q[14] = vsubq_s16(q[7], q[1]);
      q[13] = vsubq_s16(q[4], q[3]);
      do_butterfly(q[14], q[13], -cospi_8_64, -cospi_24_64, &q[0], &q[1]);
      // part of stage 6
      load_from_output(out, 18, 19, &q[14], &q[13]);
      q[8] = vaddq_s16(q[14], q[1]);
      q[9] = vaddq_s16(q[13], q[6]);
      q[13] = vsubq_s16(q[13], q[6]);
      q[1] = vsubq_s16(q[14], q[1]);
      store_in_output(out, 18, 19, q[8], q[9]);
      load_from_output(out, 28, 29, &q[8], &q[9]);
      q[14] = vsubq_s16(q[8], q[5]);
      q[10] = vaddq_s16(q[8], q[5]);
      q[11] = vaddq_s16(q[9], q[0]);
      q[0] = vsubq_s16(q[9], q[0]);
      store_in_output(out, 28, 29, q[10], q[11]);
      // part of stage 7
      do_butterfly(q[14], q[13], cospi_16_64, cospi_16_64, &q[13], &q[14]);
      store_in_output(out, 20, 27, q[13], q[14]);
      do_butterfly(q[0], q[1], cospi_16_64, cospi_16_64, &q[1], &q[0]);
      store_in_output(out, 21, 26, q[1], q[0]);

      // -----------------------------------------
      // BLOCK C: 8-10,11-15
      // -----------------------------------------
      // generate 8,9,14,15
      // part of stage 2
      load_from_transformed(trans_buf, 2, 30, &q[14], &q[13]);
      do_butterfly(q[14], q[13], cospi_30_64, cospi_2_64, &q[0], &q[2]);
      load_from_transformed(trans_buf, 18, 14, &q[14], &q[13]);
      do_butterfly(q[14], q[13], cospi_14_64, cospi_18_64, &q[1], &q[3]);
      // part of stage 3
      q[13] = vsubq_s16(q[0], q[1]);
      q[0] = vaddq_s16(q[0], q[1]);
      q[14] = vsubq_s16(q[2], q[3]);
      q[2] = vaddq_s16(q[2], q[3]);
      // part of stage 4
      do_butterfly(q[14], q[13], cospi_24_64, cospi_8_64, &q[1], &q[3]);

      // generate 10,11,12,13
      // part of stage 2
      load_from_transformed(trans_buf, 10, 22, &q[14], &q[13]);
      do_butterfly(q[14], q[13], cospi_22_64, cospi_10_64, &q[5], &q[7]);
      load_from_transformed(trans_buf, 26, 6, &q[14], &q[13]);
      do_butterfly(q[14], q[13], cospi_6_64, cospi_26_64, &q[4], &q[6]);
      // part of stage 3
      q[14] = vsubq_s16(q[4], q[5]);
      q[5] = vaddq_s16(q[4], q[5]);
      q[13] = vsubq_s16(q[6], q[7]);
      q[6] = vaddq_s16(q[6], q[7]);
      // part of stage 4
      do_butterfly(q[14], q[13], -cospi_8_64, -cospi_24_64, &q[4], &q[7]);
      // part of stage 5
      q[8] = vaddq_s16(q[0], q[5]);
      q[9] = vaddq_s16(q[1], q[7]);
      q[13] = vsubq_s16(q[1], q[7]);
      q[14] = vsubq_s16(q[3], q[4]);
      q[10] = vaddq_s16(q[3], q[4]);
      q[15] = vaddq_s16(q[2], q[6]);
      store_in_output(out, 8, 15, q[8], q[15]);
      store_in_output(out, 9, 14, q[9], q[10]);
      // part of stage 6
      do_butterfly(q[14], q[13], cospi_16_64, cospi_16_64, &q[1], &q[3]);
      store_in_output(out, 13, 10, q[3], q[1]);
      q[13] = vsubq_s16(q[0], q[5]);
      q[14] = vsubq_s16(q[2], q[6]);
      do_butterfly(q[14], q[13], cospi_16_64, cospi_16_64, &q[1], &q[3]);
      store_in_output(out, 11, 12, q[1], q[3]);

      // -----------------------------------------
      // BLOCK D: 0-3,4-7
      // -----------------------------------------
      // generate 4,5,6,7
      // part of stage 3
      load_from_transformed(trans_buf, 4, 28, &q[14], &q[13]);
      do_butterfly(q[14], q[13], cospi_28_64, cospi_4_64, &q[0], &q[2]);
      load_from_transformed(trans_buf, 20, 12, &q[14], &q[13]);
      do_butterfly(q[14], q[13], cospi_12_64, cospi_20_64, &q[1], &q[3]);
      // part of stage 4
      q[13] = vsubq_s16(q[0], q[1]);
      q[0] = vaddq_s16(q[0], q[1]);
      q[14] = vsubq_s16(q[2], q[3]);
      q[2] = vaddq_s16(q[2], q[3]);
      // part of stage 5
      do_butterfly(q[14], q[13], cospi_16_64, cospi_16_64, &q[1], &q[3]);

      // generate 0,1,2,3
      // part of stage 4
      load_from_transformed(trans_buf, 0, 16, &q[14], &q[13]);
      do_butterfly(q[14], q[13], cospi_16_64, cospi_16_64, &q[5], &q[7]);
      load_from_transformed(trans_buf, 8, 24, &q[14], &q[13]);
      do_butterfly(q[14], q[13], cospi_24_64, cospi_8_64, &q[14], &q[6]);
      // part of stage 5
      q[4] = vaddq_s16(q[7], q[6]);
      q[7] = vsubq_s16(q[7], q[6]);
      q[6] = vsubq_s16(q[5], q[14]);
      q[5] = vaddq_s16(q[5], q[14]);
      // part of stage 6
      q[8] = vaddq_s16(q[4], q[2]);
      q[9] = vaddq_s16(q[5], q[3]);
      q[10] = vaddq_s16(q[6], q[1]);
      q[11] = vaddq_s16(q[7], q[0]);
      q[12] = vsubq_s16(q[7], q[0]);
      q[13] = vsubq_s16(q[6], q[1]);
      q[14] = vsubq_s16(q[5], q[3]);
      q[15] = vsubq_s16(q[4], q[2]);
      // part of stage 7
      load_from_output(out, 14, 15, &q[0], &q[1]);
      q[2] = vaddq_s16(q[8], q[1]);
      q[3] = vaddq_s16(q[9], q[0]);
      q[4] = vsubq_s16(q[9], q[0]);
      q[5] = vsubq_s16(q[8], q[1]);
      load_from_output(out, 16, 17, &q[0], &q[1]);
      q[8] = final_add(q[4], q[1]);
      q[9] = final_add(q[5], q[0]);
      q[6] = final_sub(q[5], q[0]);
      q[7] = final_sub(q[4], q[1]);

      if (idct32_pass_loop == 0) {
        idct32_bands_end_1st_pass(out, q);
      } else {
        if (highbd_flag) {
          highbd_idct32_bands_end_2nd_pass_bd8(out, dst, stride, q);
          dst += 8;
        } else {
          idct32_bands_end_2nd_pass(out, dest, stride, q);
          dest += 8;
        }
      }
    }
  }
}

void vpx_idct32x32_1024_add_neon(const tran_low_t *input, uint8_t *dest,
                                 int stride) {
  vpx_idct32_32_neon(input, dest, stride, 0);
}
