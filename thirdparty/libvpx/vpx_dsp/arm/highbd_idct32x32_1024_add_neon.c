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

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/arm/idct_neon.h"
#include "vpx_dsp/arm/transpose_neon.h"
#include "vpx_dsp/txfm_common.h"

static INLINE void load_from_transformed(const int32_t *const trans_buf,
                                         const int first, const int second,
                                         int32x4x2_t *const q0,
                                         int32x4x2_t *const q1) {
  q0->val[0] = vld1q_s32(trans_buf + first * 8);
  q0->val[1] = vld1q_s32(trans_buf + first * 8 + 4);
  q1->val[0] = vld1q_s32(trans_buf + second * 8);
  q1->val[1] = vld1q_s32(trans_buf + second * 8 + 4);
}

static INLINE void load_from_output(const int32_t *const out, const int first,
                                    const int second, int32x4x2_t *const q0,
                                    int32x4x2_t *const q1) {
  q0->val[0] = vld1q_s32(out + first * 32);
  q0->val[1] = vld1q_s32(out + first * 32 + 4);
  q1->val[0] = vld1q_s32(out + second * 32);
  q1->val[1] = vld1q_s32(out + second * 32 + 4);
}

static INLINE void store_in_output(int32_t *const out, const int first,
                                   const int second, const int32x4x2_t q0,
                                   const int32x4x2_t q1) {
  vst1q_s32(out + first * 32, q0.val[0]);
  vst1q_s32(out + first * 32 + 4, q0.val[1]);
  vst1q_s32(out + second * 32, q1.val[0]);
  vst1q_s32(out + second * 32 + 4, q1.val[1]);
}

static INLINE void highbd_store_combine_results(
    uint16_t *p1, uint16_t *p2, const int stride, const int32x4x2_t q0,
    const int32x4x2_t q1, const int32x4x2_t q2, const int32x4x2_t q3,
    const int16x8_t max) {
  int16x8_t o[4];
  uint16x8_t d[4];

  d[0] = vld1q_u16(p1);
  p1 += stride;
  d[1] = vld1q_u16(p1);
  d[3] = vld1q_u16(p2);
  p2 -= stride;
  d[2] = vld1q_u16(p2);

  o[0] = vcombine_s16(vrshrn_n_s32(q0.val[0], 6), vrshrn_n_s32(q0.val[1], 6));
  o[1] = vcombine_s16(vrshrn_n_s32(q1.val[0], 6), vrshrn_n_s32(q1.val[1], 6));
  o[2] = vcombine_s16(vrshrn_n_s32(q2.val[0], 6), vrshrn_n_s32(q2.val[1], 6));
  o[3] = vcombine_s16(vrshrn_n_s32(q3.val[0], 6), vrshrn_n_s32(q3.val[1], 6));

  o[0] = vqaddq_s16(o[0], vreinterpretq_s16_u16(d[0]));
  o[1] = vqaddq_s16(o[1], vreinterpretq_s16_u16(d[1]));
  o[2] = vqaddq_s16(o[2], vreinterpretq_s16_u16(d[2]));
  o[3] = vqaddq_s16(o[3], vreinterpretq_s16_u16(d[3]));
  o[0] = vminq_s16(o[0], max);
  o[1] = vminq_s16(o[1], max);
  o[2] = vminq_s16(o[2], max);
  o[3] = vminq_s16(o[3], max);
  d[0] = vqshluq_n_s16(o[0], 0);
  d[1] = vqshluq_n_s16(o[1], 0);
  d[2] = vqshluq_n_s16(o[2], 0);
  d[3] = vqshluq_n_s16(o[3], 0);

  vst1q_u16(p1, d[1]);
  p1 -= stride;
  vst1q_u16(p1, d[0]);
  vst1q_u16(p2, d[2]);
  p2 += stride;
  vst1q_u16(p2, d[3]);
}

static INLINE void do_butterfly(const int32x4x2_t qIn0, const int32x4x2_t qIn1,
                                const int32_t first_const,
                                const int32_t second_const,
                                int32x4x2_t *const qOut0,
                                int32x4x2_t *const qOut1) {
  int64x2x2_t q[4];
  int32x2_t d[6];

  // Note: using v{mul, mla, mls}l_n_s32 here slows down 35% with gcc 4.9.
  d[4] = vdup_n_s32(first_const);
  d[5] = vdup_n_s32(second_const);

  q[0].val[0] = vmull_s32(vget_low_s32(qIn0.val[0]), d[4]);
  q[0].val[1] = vmull_s32(vget_high_s32(qIn0.val[0]), d[4]);
  q[1].val[0] = vmull_s32(vget_low_s32(qIn0.val[1]), d[4]);
  q[1].val[1] = vmull_s32(vget_high_s32(qIn0.val[1]), d[4]);
  q[0].val[0] = vmlsl_s32(q[0].val[0], vget_low_s32(qIn1.val[0]), d[5]);
  q[0].val[1] = vmlsl_s32(q[0].val[1], vget_high_s32(qIn1.val[0]), d[5]);
  q[1].val[0] = vmlsl_s32(q[1].val[0], vget_low_s32(qIn1.val[1]), d[5]);
  q[1].val[1] = vmlsl_s32(q[1].val[1], vget_high_s32(qIn1.val[1]), d[5]);

  q[2].val[0] = vmull_s32(vget_low_s32(qIn0.val[0]), d[5]);
  q[2].val[1] = vmull_s32(vget_high_s32(qIn0.val[0]), d[5]);
  q[3].val[0] = vmull_s32(vget_low_s32(qIn0.val[1]), d[5]);
  q[3].val[1] = vmull_s32(vget_high_s32(qIn0.val[1]), d[5]);
  q[2].val[0] = vmlal_s32(q[2].val[0], vget_low_s32(qIn1.val[0]), d[4]);
  q[2].val[1] = vmlal_s32(q[2].val[1], vget_high_s32(qIn1.val[0]), d[4]);
  q[3].val[0] = vmlal_s32(q[3].val[0], vget_low_s32(qIn1.val[1]), d[4]);
  q[3].val[1] = vmlal_s32(q[3].val[1], vget_high_s32(qIn1.val[1]), d[4]);

  qOut0->val[0] = vcombine_s32(vrshrn_n_s64(q[0].val[0], DCT_CONST_BITS),
                               vrshrn_n_s64(q[0].val[1], DCT_CONST_BITS));
  qOut0->val[1] = vcombine_s32(vrshrn_n_s64(q[1].val[0], DCT_CONST_BITS),
                               vrshrn_n_s64(q[1].val[1], DCT_CONST_BITS));
  qOut1->val[0] = vcombine_s32(vrshrn_n_s64(q[2].val[0], DCT_CONST_BITS),
                               vrshrn_n_s64(q[2].val[1], DCT_CONST_BITS));
  qOut1->val[1] = vcombine_s32(vrshrn_n_s64(q[3].val[0], DCT_CONST_BITS),
                               vrshrn_n_s64(q[3].val[1], DCT_CONST_BITS));
}

static INLINE void load_s32x4q_dual(const int32_t *in, int32x4x2_t *const s) {
  s[0].val[0] = vld1q_s32(in);
  s[0].val[1] = vld1q_s32(in + 4);
  in += 32;
  s[1].val[0] = vld1q_s32(in);
  s[1].val[1] = vld1q_s32(in + 4);
  in += 32;
  s[2].val[0] = vld1q_s32(in);
  s[2].val[1] = vld1q_s32(in + 4);
  in += 32;
  s[3].val[0] = vld1q_s32(in);
  s[3].val[1] = vld1q_s32(in + 4);
  in += 32;
  s[4].val[0] = vld1q_s32(in);
  s[4].val[1] = vld1q_s32(in + 4);
  in += 32;
  s[5].val[0] = vld1q_s32(in);
  s[5].val[1] = vld1q_s32(in + 4);
  in += 32;
  s[6].val[0] = vld1q_s32(in);
  s[6].val[1] = vld1q_s32(in + 4);
  in += 32;
  s[7].val[0] = vld1q_s32(in);
  s[7].val[1] = vld1q_s32(in + 4);
}

static INLINE void transpose_and_store_s32_8x8(int32x4x2_t *const a,
                                               int32_t **out) {
  transpose_s32_8x8(&a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6], &a[7]);

  vst1q_s32(*out, a[0].val[0]);
  *out += 4;
  vst1q_s32(*out, a[0].val[1]);
  *out += 4;
  vst1q_s32(*out, a[1].val[0]);
  *out += 4;
  vst1q_s32(*out, a[1].val[1]);
  *out += 4;
  vst1q_s32(*out, a[2].val[0]);
  *out += 4;
  vst1q_s32(*out, a[2].val[1]);
  *out += 4;
  vst1q_s32(*out, a[3].val[0]);
  *out += 4;
  vst1q_s32(*out, a[3].val[1]);
  *out += 4;
  vst1q_s32(*out, a[4].val[0]);
  *out += 4;
  vst1q_s32(*out, a[4].val[1]);
  *out += 4;
  vst1q_s32(*out, a[5].val[0]);
  *out += 4;
  vst1q_s32(*out, a[5].val[1]);
  *out += 4;
  vst1q_s32(*out, a[6].val[0]);
  *out += 4;
  vst1q_s32(*out, a[6].val[1]);
  *out += 4;
  vst1q_s32(*out, a[7].val[0]);
  *out += 4;
  vst1q_s32(*out, a[7].val[1]);
  *out += 4;
}

static INLINE void idct32_transpose_pair(const int32_t *input, int32_t *t_buf) {
  int i;
  int32x4x2_t s[8];

  for (i = 0; i < 4; i++, input += 8) {
    load_s32x4q_dual(input, s);
    transpose_and_store_s32_8x8(s, &t_buf);
  }
}

static INLINE void idct32_bands_end_1st_pass(int32_t *const out,
                                             int32x4x2_t *const q) {
  store_in_output(out, 16, 17, q[6], q[7]);
  store_in_output(out, 14, 15, q[8], q[9]);

  load_from_output(out, 30, 31, &q[0], &q[1]);
  q[4] = highbd_idct_add_dual(q[2], q[1]);
  q[5] = highbd_idct_add_dual(q[3], q[0]);
  q[6] = highbd_idct_sub_dual(q[3], q[0]);
  q[7] = highbd_idct_sub_dual(q[2], q[1]);
  store_in_output(out, 30, 31, q[6], q[7]);
  store_in_output(out, 0, 1, q[4], q[5]);

  load_from_output(out, 12, 13, &q[0], &q[1]);
  q[2] = highbd_idct_add_dual(q[10], q[1]);
  q[3] = highbd_idct_add_dual(q[11], q[0]);
  q[4] = highbd_idct_sub_dual(q[11], q[0]);
  q[5] = highbd_idct_sub_dual(q[10], q[1]);

  load_from_output(out, 18, 19, &q[0], &q[1]);
  q[8] = highbd_idct_add_dual(q[4], q[1]);
  q[9] = highbd_idct_add_dual(q[5], q[0]);
  q[6] = highbd_idct_sub_dual(q[5], q[0]);
  q[7] = highbd_idct_sub_dual(q[4], q[1]);
  store_in_output(out, 18, 19, q[6], q[7]);
  store_in_output(out, 12, 13, q[8], q[9]);

  load_from_output(out, 28, 29, &q[0], &q[1]);
  q[4] = highbd_idct_add_dual(q[2], q[1]);
  q[5] = highbd_idct_add_dual(q[3], q[0]);
  q[6] = highbd_idct_sub_dual(q[3], q[0]);
  q[7] = highbd_idct_sub_dual(q[2], q[1]);
  store_in_output(out, 28, 29, q[6], q[7]);
  store_in_output(out, 2, 3, q[4], q[5]);

  load_from_output(out, 10, 11, &q[0], &q[1]);
  q[2] = highbd_idct_add_dual(q[12], q[1]);
  q[3] = highbd_idct_add_dual(q[13], q[0]);
  q[4] = highbd_idct_sub_dual(q[13], q[0]);
  q[5] = highbd_idct_sub_dual(q[12], q[1]);

  load_from_output(out, 20, 21, &q[0], &q[1]);
  q[8] = highbd_idct_add_dual(q[4], q[1]);
  q[9] = highbd_idct_add_dual(q[5], q[0]);
  q[6] = highbd_idct_sub_dual(q[5], q[0]);
  q[7] = highbd_idct_sub_dual(q[4], q[1]);
  store_in_output(out, 20, 21, q[6], q[7]);
  store_in_output(out, 10, 11, q[8], q[9]);

  load_from_output(out, 26, 27, &q[0], &q[1]);
  q[4] = highbd_idct_add_dual(q[2], q[1]);
  q[5] = highbd_idct_add_dual(q[3], q[0]);
  q[6] = highbd_idct_sub_dual(q[3], q[0]);
  q[7] = highbd_idct_sub_dual(q[2], q[1]);
  store_in_output(out, 26, 27, q[6], q[7]);
  store_in_output(out, 4, 5, q[4], q[5]);

  load_from_output(out, 8, 9, &q[0], &q[1]);
  q[2] = highbd_idct_add_dual(q[14], q[1]);
  q[3] = highbd_idct_add_dual(q[15], q[0]);
  q[4] = highbd_idct_sub_dual(q[15], q[0]);
  q[5] = highbd_idct_sub_dual(q[14], q[1]);

  load_from_output(out, 22, 23, &q[0], &q[1]);
  q[8] = highbd_idct_add_dual(q[4], q[1]);
  q[9] = highbd_idct_add_dual(q[5], q[0]);
  q[6] = highbd_idct_sub_dual(q[5], q[0]);
  q[7] = highbd_idct_sub_dual(q[4], q[1]);
  store_in_output(out, 22, 23, q[6], q[7]);
  store_in_output(out, 8, 9, q[8], q[9]);

  load_from_output(out, 24, 25, &q[0], &q[1]);
  q[4] = highbd_idct_add_dual(q[2], q[1]);
  q[5] = highbd_idct_add_dual(q[3], q[0]);
  q[6] = highbd_idct_sub_dual(q[3], q[0]);
  q[7] = highbd_idct_sub_dual(q[2], q[1]);
  store_in_output(out, 24, 25, q[6], q[7]);
  store_in_output(out, 6, 7, q[4], q[5]);
}

static INLINE void idct32_bands_end_2nd_pass(const int32_t *const out,
                                             uint16_t *const dest,
                                             const int stride,
                                             const int16x8_t max,
                                             int32x4x2_t *const q) {
  uint16_t *dest0 = dest + 0 * stride;
  uint16_t *dest1 = dest + 31 * stride;
  uint16_t *dest2 = dest + 16 * stride;
  uint16_t *dest3 = dest + 15 * stride;
  const int str2 = stride << 1;

  highbd_store_combine_results(dest2, dest3, stride, q[6], q[7], q[8], q[9],
                               max);
  dest2 += str2;
  dest3 -= str2;

  load_from_output(out, 30, 31, &q[0], &q[1]);
  q[4] = highbd_idct_add_dual(q[2], q[1]);
  q[5] = highbd_idct_add_dual(q[3], q[0]);
  q[6] = highbd_idct_sub_dual(q[3], q[0]);
  q[7] = highbd_idct_sub_dual(q[2], q[1]);
  highbd_store_combine_results(dest0, dest1, stride, q[4], q[5], q[6], q[7],
                               max);
  dest0 += str2;
  dest1 -= str2;

  load_from_output(out, 12, 13, &q[0], &q[1]);
  q[2] = highbd_idct_add_dual(q[10], q[1]);
  q[3] = highbd_idct_add_dual(q[11], q[0]);
  q[4] = highbd_idct_sub_dual(q[11], q[0]);
  q[5] = highbd_idct_sub_dual(q[10], q[1]);

  load_from_output(out, 18, 19, &q[0], &q[1]);
  q[8] = highbd_idct_add_dual(q[4], q[1]);
  q[9] = highbd_idct_add_dual(q[5], q[0]);
  q[6] = highbd_idct_sub_dual(q[5], q[0]);
  q[7] = highbd_idct_sub_dual(q[4], q[1]);
  highbd_store_combine_results(dest2, dest3, stride, q[6], q[7], q[8], q[9],
                               max);
  dest2 += str2;
  dest3 -= str2;

  load_from_output(out, 28, 29, &q[0], &q[1]);
  q[4] = highbd_idct_add_dual(q[2], q[1]);
  q[5] = highbd_idct_add_dual(q[3], q[0]);
  q[6] = highbd_idct_sub_dual(q[3], q[0]);
  q[7] = highbd_idct_sub_dual(q[2], q[1]);
  highbd_store_combine_results(dest0, dest1, stride, q[4], q[5], q[6], q[7],
                               max);
  dest0 += str2;
  dest1 -= str2;

  load_from_output(out, 10, 11, &q[0], &q[1]);
  q[2] = highbd_idct_add_dual(q[12], q[1]);
  q[3] = highbd_idct_add_dual(q[13], q[0]);
  q[4] = highbd_idct_sub_dual(q[13], q[0]);
  q[5] = highbd_idct_sub_dual(q[12], q[1]);

  load_from_output(out, 20, 21, &q[0], &q[1]);
  q[8] = highbd_idct_add_dual(q[4], q[1]);
  q[9] = highbd_idct_add_dual(q[5], q[0]);
  q[6] = highbd_idct_sub_dual(q[5], q[0]);
  q[7] = highbd_idct_sub_dual(q[4], q[1]);
  highbd_store_combine_results(dest2, dest3, stride, q[6], q[7], q[8], q[9],
                               max);
  dest2 += str2;
  dest3 -= str2;

  load_from_output(out, 26, 27, &q[0], &q[1]);
  q[4] = highbd_idct_add_dual(q[2], q[1]);
  q[5] = highbd_idct_add_dual(q[3], q[0]);
  q[6] = highbd_idct_sub_dual(q[3], q[0]);
  q[7] = highbd_idct_sub_dual(q[2], q[1]);
  highbd_store_combine_results(dest0, dest1, stride, q[4], q[5], q[6], q[7],
                               max);
  dest0 += str2;
  dest1 -= str2;

  load_from_output(out, 8, 9, &q[0], &q[1]);
  q[2] = highbd_idct_add_dual(q[14], q[1]);
  q[3] = highbd_idct_add_dual(q[15], q[0]);
  q[4] = highbd_idct_sub_dual(q[15], q[0]);
  q[5] = highbd_idct_sub_dual(q[14], q[1]);

  load_from_output(out, 22, 23, &q[0], &q[1]);
  q[8] = highbd_idct_add_dual(q[4], q[1]);
  q[9] = highbd_idct_add_dual(q[5], q[0]);
  q[6] = highbd_idct_sub_dual(q[5], q[0]);
  q[7] = highbd_idct_sub_dual(q[4], q[1]);
  highbd_store_combine_results(dest2, dest3, stride, q[6], q[7], q[8], q[9],
                               max);

  load_from_output(out, 24, 25, &q[0], &q[1]);
  q[4] = highbd_idct_add_dual(q[2], q[1]);
  q[5] = highbd_idct_add_dual(q[3], q[0]);
  q[6] = highbd_idct_sub_dual(q[3], q[0]);
  q[7] = highbd_idct_sub_dual(q[2], q[1]);
  highbd_store_combine_results(dest0, dest1, stride, q[4], q[5], q[6], q[7],
                               max);
}

static INLINE void vpx_highbd_idct32_32_neon(const tran_low_t *input,
                                             uint16_t *dst, const int stride,
                                             const int bd) {
  int i, idct32_pass_loop;
  int32_t trans_buf[32 * 8];
  int32_t pass1[32 * 32];
  int32_t pass2[32 * 32];
  int32_t *out;
  int32x4x2_t q[16];

  for (idct32_pass_loop = 0, out = pass1; idct32_pass_loop < 2;
       idct32_pass_loop++, input = pass1, out = pass2) {
    for (i = 0; i < 4; i++, out += 8) {  // idct32_bands_loop
      idct32_transpose_pair(input, trans_buf);
      input += 32 * 8;

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
      q[4] = highbd_idct_add_dual(q[0], q[1]);
      q[13] = highbd_idct_sub_dual(q[0], q[1]);
      q[6] = highbd_idct_add_dual(q[2], q[3]);
      q[14] = highbd_idct_sub_dual(q[2], q[3]);
      // part of stage 3
      do_butterfly(q[14], q[13], cospi_28_64, cospi_4_64, &q[5], &q[7]);

      // generate 18,19,28,29
      // part of stage 1
      load_from_transformed(trans_buf, 9, 23, &q[14], &q[13]);
      do_butterfly(q[14], q[13], cospi_23_64, cospi_9_64, &q[0], &q[2]);
      load_from_transformed(trans_buf, 25, 7, &q[14], &q[13]);
      do_butterfly(q[14], q[13], cospi_7_64, cospi_25_64, &q[1], &q[3]);
      // part of stage 2
      q[13] = highbd_idct_sub_dual(q[3], q[2]);
      q[3] = highbd_idct_add_dual(q[3], q[2]);
      q[14] = highbd_idct_sub_dual(q[1], q[0]);
      q[2] = highbd_idct_add_dual(q[1], q[0]);
      // part of stage 3
      do_butterfly(q[14], q[13], -cospi_4_64, -cospi_28_64, &q[1], &q[0]);
      // part of stage 4
      q[8] = highbd_idct_add_dual(q[4], q[2]);
      q[9] = highbd_idct_add_dual(q[5], q[0]);
      q[10] = highbd_idct_add_dual(q[7], q[1]);
      q[15] = highbd_idct_add_dual(q[6], q[3]);
      q[13] = highbd_idct_sub_dual(q[5], q[0]);
      q[14] = highbd_idct_sub_dual(q[7], q[1]);
      store_in_output(out, 16, 31, q[8], q[15]);
      store_in_output(out, 17, 30, q[9], q[10]);
      // part of stage 5
      do_butterfly(q[14], q[13], cospi_24_64, cospi_8_64, &q[0], &q[1]);
      store_in_output(out, 29, 18, q[1], q[0]);
      // part of stage 4
      q[13] = highbd_idct_sub_dual(q[4], q[2]);
      q[14] = highbd_idct_sub_dual(q[6], q[3]);
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
      q[13] = highbd_idct_sub_dual(q[0], q[1]);
      q[0] = highbd_idct_add_dual(q[0], q[1]);
      q[14] = highbd_idct_sub_dual(q[2], q[3]);
      q[2] = highbd_idct_add_dual(q[2], q[3]);
      // part of stage 3
      do_butterfly(q[14], q[13], cospi_12_64, cospi_20_64, &q[1], &q[3]);

      // generate 22,23,24,25
      // part of stage 1
      load_from_transformed(trans_buf, 13, 19, &q[14], &q[13]);
      do_butterfly(q[14], q[13], cospi_19_64, cospi_13_64, &q[5], &q[7]);
      load_from_transformed(trans_buf, 29, 3, &q[14], &q[13]);
      do_butterfly(q[14], q[13], cospi_3_64, cospi_29_64, &q[4], &q[6]);
      // part of stage 2
      q[14] = highbd_idct_sub_dual(q[4], q[5]);
      q[5] = highbd_idct_add_dual(q[4], q[5]);
      q[13] = highbd_idct_sub_dual(q[6], q[7]);
      q[6] = highbd_idct_add_dual(q[6], q[7]);
      // part of stage 3
      do_butterfly(q[14], q[13], -cospi_20_64, -cospi_12_64, &q[4], &q[7]);
      // part of stage 4
      q[10] = highbd_idct_add_dual(q[7], q[1]);
      q[11] = highbd_idct_add_dual(q[5], q[0]);
      q[12] = highbd_idct_add_dual(q[6], q[2]);
      q[15] = highbd_idct_add_dual(q[4], q[3]);
      // part of stage 6
      load_from_output(out, 16, 17, &q[14], &q[13]);
      q[8] = highbd_idct_add_dual(q[14], q[11]);
      q[9] = highbd_idct_add_dual(q[13], q[10]);
      q[13] = highbd_idct_sub_dual(q[13], q[10]);
      q[11] = highbd_idct_sub_dual(q[14], q[11]);
      store_in_output(out, 17, 16, q[9], q[8]);
      load_from_output(out, 30, 31, &q[14], &q[9]);
      q[8] = highbd_idct_sub_dual(q[9], q[12]);
      q[10] = highbd_idct_add_dual(q[14], q[15]);
      q[14] = highbd_idct_sub_dual(q[14], q[15]);
      q[12] = highbd_idct_add_dual(q[9], q[12]);
      store_in_output(out, 30, 31, q[10], q[12]);
      // part of stage 7
      do_butterfly(q[14], q[13], cospi_16_64, cospi_16_64, &q[13], &q[14]);
      store_in_output(out, 25, 22, q[14], q[13]);
      do_butterfly(q[8], q[11], cospi_16_64, cospi_16_64, &q[13], &q[14]);
      store_in_output(out, 24, 23, q[14], q[13]);
      // part of stage 4
      q[14] = highbd_idct_sub_dual(q[5], q[0]);
      q[13] = highbd_idct_sub_dual(q[6], q[2]);
      do_butterfly(q[14], q[13], -cospi_8_64, -cospi_24_64, &q[5], &q[6]);
      q[14] = highbd_idct_sub_dual(q[7], q[1]);
      q[13] = highbd_idct_sub_dual(q[4], q[3]);
      do_butterfly(q[14], q[13], -cospi_8_64, -cospi_24_64, &q[0], &q[1]);
      // part of stage 6
      load_from_output(out, 18, 19, &q[14], &q[13]);
      q[8] = highbd_idct_add_dual(q[14], q[1]);
      q[9] = highbd_idct_add_dual(q[13], q[6]);
      q[13] = highbd_idct_sub_dual(q[13], q[6]);
      q[1] = highbd_idct_sub_dual(q[14], q[1]);
      store_in_output(out, 18, 19, q[8], q[9]);
      load_from_output(out, 28, 29, &q[8], &q[9]);
      q[14] = highbd_idct_sub_dual(q[8], q[5]);
      q[10] = highbd_idct_add_dual(q[8], q[5]);
      q[11] = highbd_idct_add_dual(q[9], q[0]);
      q[0] = highbd_idct_sub_dual(q[9], q[0]);
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
      q[13] = highbd_idct_sub_dual(q[0], q[1]);
      q[0] = highbd_idct_add_dual(q[0], q[1]);
      q[14] = highbd_idct_sub_dual(q[2], q[3]);
      q[2] = highbd_idct_add_dual(q[2], q[3]);
      // part of stage 4
      do_butterfly(q[14], q[13], cospi_24_64, cospi_8_64, &q[1], &q[3]);

      // generate 10,11,12,13
      // part of stage 2
      load_from_transformed(trans_buf, 10, 22, &q[14], &q[13]);
      do_butterfly(q[14], q[13], cospi_22_64, cospi_10_64, &q[5], &q[7]);
      load_from_transformed(trans_buf, 26, 6, &q[14], &q[13]);
      do_butterfly(q[14], q[13], cospi_6_64, cospi_26_64, &q[4], &q[6]);
      // part of stage 3
      q[14] = highbd_idct_sub_dual(q[4], q[5]);
      q[5] = highbd_idct_add_dual(q[4], q[5]);
      q[13] = highbd_idct_sub_dual(q[6], q[7]);
      q[6] = highbd_idct_add_dual(q[6], q[7]);
      // part of stage 4
      do_butterfly(q[14], q[13], -cospi_8_64, -cospi_24_64, &q[4], &q[7]);
      // part of stage 5
      q[8] = highbd_idct_add_dual(q[0], q[5]);
      q[9] = highbd_idct_add_dual(q[1], q[7]);
      q[13] = highbd_idct_sub_dual(q[1], q[7]);
      q[14] = highbd_idct_sub_dual(q[3], q[4]);
      q[10] = highbd_idct_add_dual(q[3], q[4]);
      q[15] = highbd_idct_add_dual(q[2], q[6]);
      store_in_output(out, 8, 15, q[8], q[15]);
      store_in_output(out, 9, 14, q[9], q[10]);
      // part of stage 6
      do_butterfly(q[14], q[13], cospi_16_64, cospi_16_64, &q[1], &q[3]);
      store_in_output(out, 13, 10, q[3], q[1]);
      q[13] = highbd_idct_sub_dual(q[0], q[5]);
      q[14] = highbd_idct_sub_dual(q[2], q[6]);
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
      q[13] = highbd_idct_sub_dual(q[0], q[1]);
      q[0] = highbd_idct_add_dual(q[0], q[1]);
      q[14] = highbd_idct_sub_dual(q[2], q[3]);
      q[2] = highbd_idct_add_dual(q[2], q[3]);
      // part of stage 5
      do_butterfly(q[14], q[13], cospi_16_64, cospi_16_64, &q[1], &q[3]);

      // generate 0,1,2,3
      // part of stage 4
      load_from_transformed(trans_buf, 0, 16, &q[14], &q[13]);
      do_butterfly(q[14], q[13], cospi_16_64, cospi_16_64, &q[5], &q[7]);
      load_from_transformed(trans_buf, 8, 24, &q[14], &q[13]);
      do_butterfly(q[14], q[13], cospi_24_64, cospi_8_64, &q[14], &q[6]);
      // part of stage 5
      q[4] = highbd_idct_add_dual(q[7], q[6]);
      q[7] = highbd_idct_sub_dual(q[7], q[6]);
      q[6] = highbd_idct_sub_dual(q[5], q[14]);
      q[5] = highbd_idct_add_dual(q[5], q[14]);
      // part of stage 6
      q[8] = highbd_idct_add_dual(q[4], q[2]);
      q[9] = highbd_idct_add_dual(q[5], q[3]);
      q[10] = highbd_idct_add_dual(q[6], q[1]);
      q[11] = highbd_idct_add_dual(q[7], q[0]);
      q[12] = highbd_idct_sub_dual(q[7], q[0]);
      q[13] = highbd_idct_sub_dual(q[6], q[1]);
      q[14] = highbd_idct_sub_dual(q[5], q[3]);
      q[15] = highbd_idct_sub_dual(q[4], q[2]);
      // part of stage 7
      load_from_output(out, 14, 15, &q[0], &q[1]);
      q[2] = highbd_idct_add_dual(q[8], q[1]);
      q[3] = highbd_idct_add_dual(q[9], q[0]);
      q[4] = highbd_idct_sub_dual(q[9], q[0]);
      q[5] = highbd_idct_sub_dual(q[8], q[1]);
      load_from_output(out, 16, 17, &q[0], &q[1]);
      q[8] = highbd_idct_add_dual(q[4], q[1]);
      q[9] = highbd_idct_add_dual(q[5], q[0]);
      q[6] = highbd_idct_sub_dual(q[5], q[0]);
      q[7] = highbd_idct_sub_dual(q[4], q[1]);

      if (idct32_pass_loop == 0) {
        idct32_bands_end_1st_pass(out, q);
      } else {
        const int16x8_t max = vdupq_n_s16((1 << bd) - 1);
        idct32_bands_end_2nd_pass(out, dst, stride, max, q);
        dst += 8;
      }
    }
  }
}

void vpx_highbd_idct32x32_1024_add_neon(const tran_low_t *input, uint16_t *dest,
                                        int stride, int bd) {
  if (bd == 8) {
    vpx_idct32_32_neon(input, CAST_TO_BYTEPTR(dest), stride, 1);
  } else {
    vpx_highbd_idct32_32_neon(input, dest, stride, bd);
  }
}
