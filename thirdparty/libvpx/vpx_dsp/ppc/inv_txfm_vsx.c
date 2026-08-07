/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "vpx_dsp/ppc/bitdepth_conversion_vsx.h"
#include "vpx_dsp/ppc/types_vsx.h"
#include "vpx_dsp/ppc/inv_txfm_vsx.h"

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/inv_txfm.h"

static const int16x8_t cospi1_v = { 16364, 16364, 16364, 16364,
                                    16364, 16364, 16364, 16364 };
static const int16x8_t cospi1m_v = { -16364, -16364, -16364, -16364,
                                     -16364, -16364, -16364, -16364 };
static const int16x8_t cospi2_v = { 16305, 16305, 16305, 16305,
                                    16305, 16305, 16305, 16305 };
static const int16x8_t cospi2m_v = { -16305, -16305, -16305, -16305,
                                     -16305, -16305, -16305, -16305 };
static const int16x8_t cospi3_v = { 16207, 16207, 16207, 16207,
                                    16207, 16207, 16207, 16207 };
static const int16x8_t cospi4_v = { 16069, 16069, 16069, 16069,
                                    16069, 16069, 16069, 16069 };
static const int16x8_t cospi4m_v = { -16069, -16069, -16069, -16069,
                                     -16069, -16069, -16069, -16069 };
static const int16x8_t cospi5_v = { 15893, 15893, 15893, 15893,
                                    15893, 15893, 15893, 15893 };
static const int16x8_t cospi5m_v = { -15893, -15893, -15893, -15893,
                                     -15893, -15893, -15893, -15893 };
static const int16x8_t cospi6_v = { 15679, 15679, 15679, 15679,
                                    15679, 15679, 15679, 15679 };
static const int16x8_t cospi7_v = { 15426, 15426, 15426, 15426,
                                    15426, 15426, 15426, 15426 };
static const int16x8_t cospi8_v = { 15137, 15137, 15137, 15137,
                                    15137, 15137, 15137, 15137 };
static const int16x8_t cospi8m_v = { -15137, -15137, -15137, -15137,
                                     -15137, -15137, -15137, -15137 };
static const int16x8_t cospi9_v = { 14811, 14811, 14811, 14811,
                                    14811, 14811, 14811, 14811 };
static const int16x8_t cospi9m_v = { -14811, -14811, -14811, -14811,
                                     -14811, -14811, -14811, -14811 };
static const int16x8_t cospi10_v = { 14449, 14449, 14449, 14449,
                                     14449, 14449, 14449, 14449 };
static const int16x8_t cospi10m_v = { -14449, -14449, -14449, -14449,
                                      -14449, -14449, -14449, -14449 };
static const int16x8_t cospi11_v = { 14053, 14053, 14053, 14053,
                                     14053, 14053, 14053, 14053 };
static const int16x8_t cospi12_v = { 13623, 13623, 13623, 13623,
                                     13623, 13623, 13623, 13623 };
static const int16x8_t cospi12m_v = { -13623, -13623, -13623, -13623,
                                      -13623, -13623, -13623, -13623 };
static const int16x8_t cospi13_v = { 13160, 13160, 13160, 13160,
                                     13160, 13160, 13160, 13160 };
static const int16x8_t cospi13m_v = { -13160, -13160, -13160, -13160,
                                      -13160, -13160, -13160, -13160 };
static const int16x8_t cospi14_v = { 12665, 12665, 12665, 12665,
                                     12665, 12665, 12665, 12665 };
static const int16x8_t cospi15_v = { 12140, 12140, 12140, 12140,
                                     12140, 12140, 12140, 12140 };
static const int16x8_t cospi16_v = { 11585, 11585, 11585, 11585,
                                     11585, 11585, 11585, 11585 };
static const int16x8_t cospi16m_v = { -11585, -11585, -11585, -11585,
                                      -11585, -11585, -11585, -11585 };
static const int16x8_t cospi17_v = { 11003, 11003, 11003, 11003,
                                     11003, 11003, 11003, 11003 };
static const int16x8_t cospi17m_v = { -11003, -11003, -11003, -11003,
                                      -11003, -11003, -11003, -11003 };
static const int16x8_t cospi18_v = { 10394, 10394, 10394, 10394,
                                     10394, 10394, 10394, 10394 };
static const int16x8_t cospi18m_v = { -10394, -10394, -10394, -10394,
                                      -10394, -10394, -10394, -10394 };
static const int16x8_t cospi19_v = { 9760, 9760, 9760, 9760,
                                     9760, 9760, 9760, 9760 };
static const int16x8_t cospi20_v = { 9102, 9102, 9102, 9102,
                                     9102, 9102, 9102, 9102 };
static const int16x8_t cospi20m_v = { -9102, -9102, -9102, -9102,
                                      -9102, -9102, -9102, -9102 };
static const int16x8_t cospi21_v = { 8423, 8423, 8423, 8423,
                                     8423, 8423, 8423, 8423 };
static const int16x8_t cospi21m_v = { -8423, -8423, -8423, -8423,
                                      -8423, -8423, -8423, -8423 };
static const int16x8_t cospi22_v = { 7723, 7723, 7723, 7723,
                                     7723, 7723, 7723, 7723 };
static const int16x8_t cospi23_v = { 7005, 7005, 7005, 7005,
                                     7005, 7005, 7005, 7005 };
static const int16x8_t cospi24_v = { 6270, 6270, 6270, 6270,
                                     6270, 6270, 6270, 6270 };
static const int16x8_t cospi24m_v = { -6270, -6270, -6270, -6270,
                                      -6270, -6270, -6270, -6270 };
static const int16x8_t cospi25_v = { 5520, 5520, 5520, 5520,
                                     5520, 5520, 5520, 5520 };
static const int16x8_t cospi25m_v = { -5520, -5520, -5520, -5520,
                                      -5520, -5520, -5520, -5520 };
static const int16x8_t cospi26_v = { 4756, 4756, 4756, 4756,
                                     4756, 4756, 4756, 4756 };
static const int16x8_t cospi26m_v = { -4756, -4756, -4756, -4756,
                                      -4756, -4756, -4756, -4756 };
static const int16x8_t cospi27_v = { 3981, 3981, 3981, 3981,
                                     3981, 3981, 3981, 3981 };
static const int16x8_t cospi28_v = { 3196, 3196, 3196, 3196,
                                     3196, 3196, 3196, 3196 };
static const int16x8_t cospi28m_v = { -3196, -3196, -3196, -3196,
                                      -3196, -3196, -3196, -3196 };
static const int16x8_t cospi29_v = { 2404, 2404, 2404, 2404,
                                     2404, 2404, 2404, 2404 };
static const int16x8_t cospi29m_v = { -2404, -2404, -2404, -2404,
                                      -2404, -2404, -2404, -2404 };
static const int16x8_t cospi30_v = { 1606, 1606, 1606, 1606,
                                     1606, 1606, 1606, 1606 };
static const int16x8_t cospi31_v = { 804, 804, 804, 804, 804, 804, 804, 804 };

static const int16x8_t sinpi_1_9_v = { 5283, 5283, 5283, 5283,
                                       5283, 5283, 5283, 5283 };
static const int16x8_t sinpi_2_9_v = { 9929, 9929, 9929, 9929,
                                       9929, 9929, 9929, 9929 };
static const int16x8_t sinpi_3_9_v = { 13377, 13377, 13377, 13377,
                                       13377, 13377, 13377, 13377 };
static const int16x8_t sinpi_4_9_v = { 15212, 15212, 15212, 15212,
                                       15212, 15212, 15212, 15212 };

static uint8x16_t tr8_mask0 = {
  0x0,  0x1,  0x2,  0x3,  0x4,  0x5,  0x6,  0x7,
  0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17
};

static uint8x16_t tr8_mask1 = {
  0x8,  0x9,  0xA,  0xB,  0xC,  0xD,  0xE,  0xF,
  0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F
};

#define ROUND_SHIFT_INIT                                               \
  const int32x4_t shift = vec_sl(vec_splat_s32(1), vec_splat_u32(13)); \
  const uint32x4_t shift14 = vec_splat_u32(14);

#define DCT_CONST_ROUND_SHIFT(vec) vec = vec_sra(vec_add(vec, shift), shift14);

#define PIXEL_ADD_INIT               \
  int16x8_t add8 = vec_splat_s16(8); \
  uint16x8_t shift4 = vec_splat_u16(4);

#define PIXEL_ADD4(out, in) out = vec_sra(vec_add(in, add8), shift4);

#define IDCT4(in0, in1, out0, out1)                                           \
  t0 = vec_add(in0, in1);                                                     \
  t1 = vec_sub(in0, in1);                                                     \
  tmp16_0 = vec_mergeh(t0, t1);                                               \
  temp1 = vec_sra(vec_add(vec_mule(tmp16_0, cospi16_v), shift), shift14);     \
  temp2 = vec_sra(vec_add(vec_mulo(tmp16_0, cospi16_v), shift), shift14);     \
                                                                              \
  tmp16_0 = vec_mergel(in0, in1);                                             \
  temp3 = vec_sub(vec_mule(tmp16_0, cospi24_v), vec_mulo(tmp16_0, cospi8_v)); \
  DCT_CONST_ROUND_SHIFT(temp3);                                               \
  temp4 = vec_add(vec_mule(tmp16_0, cospi8_v), vec_mulo(tmp16_0, cospi24_v)); \
  DCT_CONST_ROUND_SHIFT(temp4);                                               \
                                                                              \
  step0 = vec_packs(temp1, temp2);                                            \
  step1 = vec_packs(temp4, temp3);                                            \
  out0 = vec_add(step0, step1);                                               \
  out1 = vec_sub(step0, step1);                                               \
  out1 = vec_perm(out1, out1, mask0);

#define PACK_STORE(v0, v1)                                \
  tmp16_0 = vec_add(vec_perm(d_u0, d_u1, tr8_mask0), v0); \
  tmp16_1 = vec_add(vec_perm(d_u2, d_u3, tr8_mask0), v1); \
  output_v = vec_packsu(tmp16_0, tmp16_1);                \
                                                          \
  vec_vsx_st(output_v, 0, tmp_dest);                      \
  for (i = 0; i < 4; i++)                                 \
    for (j = 0; j < 4; j++) dest[j * stride + i] = tmp_dest[j * 4 + i];

void vpx_round_store4x4_vsx(int16x8_t *in, int16x8_t *out, uint8_t *dest,
                            int stride) {
  int i, j;
  uint8x16_t dest0 = vec_vsx_ld(0, dest);
  uint8x16_t dest1 = vec_vsx_ld(stride, dest);
  uint8x16_t dest2 = vec_vsx_ld(2 * stride, dest);
  uint8x16_t dest3 = vec_vsx_ld(3 * stride, dest);
  uint8x16_t zerov = vec_splat_u8(0);
  int16x8_t d_u0 = (int16x8_t)vec_mergeh(dest0, zerov);
  int16x8_t d_u1 = (int16x8_t)vec_mergeh(dest1, zerov);
  int16x8_t d_u2 = (int16x8_t)vec_mergeh(dest2, zerov);
  int16x8_t d_u3 = (int16x8_t)vec_mergeh(dest3, zerov);
  int16x8_t tmp16_0, tmp16_1;
  uint8x16_t output_v;
  uint8_t tmp_dest[16];
  PIXEL_ADD_INIT;

  PIXEL_ADD4(out[0], in[0]);
  PIXEL_ADD4(out[1], in[1]);

  PACK_STORE(out[0], out[1]);
}

void vpx_idct4_vsx(int16x8_t *in, int16x8_t *out) {
  int32x4_t temp1, temp2, temp3, temp4;
  int16x8_t step0, step1, tmp16_0;
  uint8x16_t mask0 = { 0x8, 0x9, 0xA, 0xB, 0xC, 0xD, 0xE, 0xF,
                       0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7 };
  int16x8_t t0 = vec_mergeh(in[0], in[1]);
  int16x8_t t1 = vec_mergel(in[0], in[1]);
  ROUND_SHIFT_INIT

  in[0] = vec_mergeh(t0, t1);
  in[1] = vec_mergel(t0, t1);

  IDCT4(in[0], in[1], out[0], out[1]);
}

void vpx_idct4x4_16_add_vsx(const tran_low_t *input, uint8_t *dest,
                            int stride) {
  int16x8_t in[2], out[2];

  in[0] = load_tran_low(0, input);
  in[1] = load_tran_low(8 * sizeof(*input), input);
  // Rows
  vpx_idct4_vsx(in, out);

  // Columns
  vpx_idct4_vsx(out, in);

  vpx_round_store4x4_vsx(in, out, dest, stride);
}

#define TRANSPOSE8x8(in0, in1, in2, in3, in4, in5, in6, in7, out0, out1, out2, \
                     out3, out4, out5, out6, out7)                             \
  out0 = vec_mergeh(in0, in1);                                                 \
  out1 = vec_mergel(in0, in1);                                                 \
  out2 = vec_mergeh(in2, in3);                                                 \
  out3 = vec_mergel(in2, in3);                                                 \
  out4 = vec_mergeh(in4, in5);                                                 \
  out5 = vec_mergel(in4, in5);                                                 \
  out6 = vec_mergeh(in6, in7);                                                 \
  out7 = vec_mergel(in6, in7);                                                 \
  in0 = (int16x8_t)vec_mergeh((int32x4_t)out0, (int32x4_t)out2);               \
  in1 = (int16x8_t)vec_mergel((int32x4_t)out0, (int32x4_t)out2);               \
  in2 = (int16x8_t)vec_mergeh((int32x4_t)out1, (int32x4_t)out3);               \
  in3 = (int16x8_t)vec_mergel((int32x4_t)out1, (int32x4_t)out3);               \
  in4 = (int16x8_t)vec_mergeh((int32x4_t)out4, (int32x4_t)out6);               \
  in5 = (int16x8_t)vec_mergel((int32x4_t)out4, (int32x4_t)out6);               \
  in6 = (int16x8_t)vec_mergeh((int32x4_t)out5, (int32x4_t)out7);               \
  in7 = (int16x8_t)vec_mergel((int32x4_t)out5, (int32x4_t)out7);               \
  out0 = vec_perm(in0, in4, tr8_mask0);                                        \
  out1 = vec_perm(in0, in4, tr8_mask1);                                        \
  out2 = vec_perm(in1, in5, tr8_mask0);                                        \
  out3 = vec_perm(in1, in5, tr8_mask1);                                        \
  out4 = vec_perm(in2, in6, tr8_mask0);                                        \
  out5 = vec_perm(in2, in6, tr8_mask1);                                        \
  out6 = vec_perm(in3, in7, tr8_mask0);                                        \
  out7 = vec_perm(in3, in7, tr8_mask1);

/* for the: temp1 = step[x] * cospi_q - step[y] * cospi_z
 *          temp2 = step[x] * cospi_z + step[y] * cospi_q */
#define STEP8_0(inpt0, inpt1, outpt0, outpt1, cospi0, cospi1)             \
  tmp16_0 = vec_mergeh(inpt0, inpt1);                                     \
  tmp16_1 = vec_mergel(inpt0, inpt1);                                     \
  temp10 = vec_sub(vec_mule(tmp16_0, cospi0), vec_mulo(tmp16_0, cospi1)); \
  temp11 = vec_sub(vec_mule(tmp16_1, cospi0), vec_mulo(tmp16_1, cospi1)); \
  DCT_CONST_ROUND_SHIFT(temp10);                                          \
  DCT_CONST_ROUND_SHIFT(temp11);                                          \
  outpt0 = vec_packs(temp10, temp11);                                     \
  temp10 = vec_add(vec_mule(tmp16_0, cospi1), vec_mulo(tmp16_0, cospi0)); \
  temp11 = vec_add(vec_mule(tmp16_1, cospi1), vec_mulo(tmp16_1, cospi0)); \
  DCT_CONST_ROUND_SHIFT(temp10);                                          \
  DCT_CONST_ROUND_SHIFT(temp11);                                          \
  outpt1 = vec_packs(temp10, temp11);

#define STEP8_1(inpt0, inpt1, outpt0, outpt1, cospi) \
  tmp16_2 = vec_sub(inpt0, inpt1);                   \
  tmp16_3 = vec_add(inpt0, inpt1);                   \
  tmp16_0 = vec_mergeh(tmp16_2, tmp16_3);            \
  tmp16_1 = vec_mergel(tmp16_2, tmp16_3);            \
  temp10 = vec_mule(tmp16_0, cospi);                 \
  temp11 = vec_mule(tmp16_1, cospi);                 \
  DCT_CONST_ROUND_SHIFT(temp10);                     \
  DCT_CONST_ROUND_SHIFT(temp11);                     \
  outpt0 = vec_packs(temp10, temp11);                \
  temp10 = vec_mulo(tmp16_0, cospi);                 \
  temp11 = vec_mulo(tmp16_1, cospi);                 \
  DCT_CONST_ROUND_SHIFT(temp10);                     \
  DCT_CONST_ROUND_SHIFT(temp11);                     \
  outpt1 = vec_packs(temp10, temp11);

#define IDCT8(in0, in1, in2, in3, in4, in5, in6, in7)    \
  /* stage 1 */                                          \
  step0 = in0;                                           \
  step2 = in4;                                           \
  step1 = in2;                                           \
  step3 = in6;                                           \
                                                         \
  STEP8_0(in1, in7, step4, step7, cospi28_v, cospi4_v);  \
  STEP8_0(in5, in3, step5, step6, cospi12_v, cospi20_v); \
                                                         \
  /* stage 2 */                                          \
  STEP8_1(step0, step2, in1, in0, cospi16_v);            \
  STEP8_0(step1, step3, in2, in3, cospi24_v, cospi8_v);  \
  in4 = vec_add(step4, step5);                           \
  in5 = vec_sub(step4, step5);                           \
  in6 = vec_sub(step7, step6);                           \
  in7 = vec_add(step6, step7);                           \
                                                         \
  /* stage 3 */                                          \
  step0 = vec_add(in0, in3);                             \
  step1 = vec_add(in1, in2);                             \
  step2 = vec_sub(in1, in2);                             \
  step3 = vec_sub(in0, in3);                             \
  step4 = in4;                                           \
  STEP8_1(in6, in5, step5, step6, cospi16_v);            \
  step7 = in7;                                           \
                                                         \
  /* stage 4 */                                          \
  in0 = vec_add(step0, step7);                           \
  in1 = vec_add(step1, step6);                           \
  in2 = vec_add(step2, step5);                           \
  in3 = vec_add(step3, step4);                           \
  in4 = vec_sub(step3, step4);                           \
  in5 = vec_sub(step2, step5);                           \
  in6 = vec_sub(step1, step6);                           \
  in7 = vec_sub(step0, step7);

#define PIXEL_ADD(in, out, add, shiftx) \
  out = vec_add(vec_sra(vec_add(in, add), shiftx), out);

void vpx_idct8_vsx(int16x8_t *in, int16x8_t *out) {
  int16x8_t step0, step1, step2, step3, step4, step5, step6, step7;
  int16x8_t tmp16_0, tmp16_1, tmp16_2, tmp16_3;
  int32x4_t temp10, temp11;
  ROUND_SHIFT_INIT;

  TRANSPOSE8x8(in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7], out[0],
               out[1], out[2], out[3], out[4], out[5], out[6], out[7]);

  IDCT8(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]);
}

void vpx_round_store8x8_vsx(int16x8_t *in, uint8_t *dest, int stride) {
  uint8x16_t zerov = vec_splat_u8(0);
  uint8x16_t dest0 = vec_vsx_ld(0, dest);
  uint8x16_t dest1 = vec_vsx_ld(stride, dest);
  uint8x16_t dest2 = vec_vsx_ld(2 * stride, dest);
  uint8x16_t dest3 = vec_vsx_ld(3 * stride, dest);
  uint8x16_t dest4 = vec_vsx_ld(4 * stride, dest);
  uint8x16_t dest5 = vec_vsx_ld(5 * stride, dest);
  uint8x16_t dest6 = vec_vsx_ld(6 * stride, dest);
  uint8x16_t dest7 = vec_vsx_ld(7 * stride, dest);
  int16x8_t d_u0 = (int16x8_t)vec_mergeh(dest0, zerov);
  int16x8_t d_u1 = (int16x8_t)vec_mergeh(dest1, zerov);
  int16x8_t d_u2 = (int16x8_t)vec_mergeh(dest2, zerov);
  int16x8_t d_u3 = (int16x8_t)vec_mergeh(dest3, zerov);
  int16x8_t d_u4 = (int16x8_t)vec_mergeh(dest4, zerov);
  int16x8_t d_u5 = (int16x8_t)vec_mergeh(dest5, zerov);
  int16x8_t d_u6 = (int16x8_t)vec_mergeh(dest6, zerov);
  int16x8_t d_u7 = (int16x8_t)vec_mergeh(dest7, zerov);
  int16x8_t add = vec_sl(vec_splat_s16(8), vec_splat_u16(1));
  uint16x8_t shift5 = vec_splat_u16(5);
  uint8x16_t output0, output1, output2, output3;

  PIXEL_ADD(in[0], d_u0, add, shift5);
  PIXEL_ADD(in[1], d_u1, add, shift5);
  PIXEL_ADD(in[2], d_u2, add, shift5);
  PIXEL_ADD(in[3], d_u3, add, shift5);
  PIXEL_ADD(in[4], d_u4, add, shift5);
  PIXEL_ADD(in[5], d_u5, add, shift5);
  PIXEL_ADD(in[6], d_u6, add, shift5);
  PIXEL_ADD(in[7], d_u7, add, shift5);
  output0 = vec_packsu(d_u0, d_u1);
  output1 = vec_packsu(d_u2, d_u3);
  output2 = vec_packsu(d_u4, d_u5);
  output3 = vec_packsu(d_u6, d_u7);

  vec_vsx_st(xxpermdi(output0, dest0, 1), 0, dest);
  vec_vsx_st(xxpermdi(output0, dest1, 3), stride, dest);
  vec_vsx_st(xxpermdi(output1, dest2, 1), 2 * stride, dest);
  vec_vsx_st(xxpermdi(output1, dest3, 3), 3 * stride, dest);
  vec_vsx_st(xxpermdi(output2, dest4, 1), 4 * stride, dest);
  vec_vsx_st(xxpermdi(output2, dest5, 3), 5 * stride, dest);
  vec_vsx_st(xxpermdi(output3, dest6, 1), 6 * stride, dest);
  vec_vsx_st(xxpermdi(output3, dest7, 3), 7 * stride, dest);
}

void vpx_idct8x8_64_add_vsx(const tran_low_t *input, uint8_t *dest,
                            int stride) {
  int16x8_t src[8], tmp[8];

  src[0] = load_tran_low(0, input);
  src[1] = load_tran_low(8 * sizeof(*input), input);
  src[2] = load_tran_low(16 * sizeof(*input), input);
  src[3] = load_tran_low(24 * sizeof(*input), input);
  src[4] = load_tran_low(32 * sizeof(*input), input);
  src[5] = load_tran_low(40 * sizeof(*input), input);
  src[6] = load_tran_low(48 * sizeof(*input), input);
  src[7] = load_tran_low(56 * sizeof(*input), input);

  vpx_idct8_vsx(src, tmp);
  vpx_idct8_vsx(tmp, src);

  vpx_round_store8x8_vsx(src, dest, stride);
}

#define STEP16_1(inpt0, inpt1, outpt0, outpt1, cospi) \
  tmp16_0 = vec_mergeh(inpt0, inpt1);                 \
  tmp16_1 = vec_mergel(inpt0, inpt1);                 \
  temp10 = vec_mule(tmp16_0, cospi);                  \
  temp11 = vec_mule(tmp16_1, cospi);                  \
  temp20 = vec_mulo(tmp16_0, cospi);                  \
  temp21 = vec_mulo(tmp16_1, cospi);                  \
  temp30 = vec_sub(temp10, temp20);                   \
  temp10 = vec_add(temp10, temp20);                   \
  temp20 = vec_sub(temp11, temp21);                   \
  temp21 = vec_add(temp11, temp21);                   \
  DCT_CONST_ROUND_SHIFT(temp30);                      \
  DCT_CONST_ROUND_SHIFT(temp20);                      \
  outpt0 = vec_packs(temp30, temp20);                 \
  DCT_CONST_ROUND_SHIFT(temp10);                      \
  DCT_CONST_ROUND_SHIFT(temp21);                      \
  outpt1 = vec_packs(temp10, temp21);

#define IDCT16(in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, inA, inB,     \
               inC, inD, inE, inF, out0, out1, out2, out3, out4, out5, out6,   \
               out7, out8, out9, outA, outB, outC, outD, outE, outF)           \
  /* stage 1 */                                                                \
  /* out0 = in0; */                                                            \
  out1 = in8;                                                                  \
  out2 = in4;                                                                  \
  out3 = inC;                                                                  \
  out4 = in2;                                                                  \
  out5 = inA;                                                                  \
  out6 = in6;                                                                  \
  out7 = inE;                                                                  \
  out8 = in1;                                                                  \
  out9 = in9;                                                                  \
  outA = in5;                                                                  \
  outB = inD;                                                                  \
  outC = in3;                                                                  \
  outD = inB;                                                                  \
  outE = in7;                                                                  \
  outF = inF;                                                                  \
                                                                               \
  /* stage 2 */                                                                \
  /* in0 = out0; */                                                            \
  in1 = out1;                                                                  \
  in2 = out2;                                                                  \
  in3 = out3;                                                                  \
  in4 = out4;                                                                  \
  in5 = out5;                                                                  \
  in6 = out6;                                                                  \
  in7 = out7;                                                                  \
                                                                               \
  STEP8_0(out8, outF, in8, inF, cospi30_v, cospi2_v);                          \
  STEP8_0(out9, outE, in9, inE, cospi14_v, cospi18_v);                         \
  STEP8_0(outA, outD, inA, inD, cospi22_v, cospi10_v);                         \
  STEP8_0(outB, outC, inB, inC, cospi6_v, cospi26_v);                          \
                                                                               \
  /* stage 3 */                                                                \
  out0 = in0;                                                                  \
  out1 = in1;                                                                  \
  out2 = in2;                                                                  \
  out3 = in3;                                                                  \
                                                                               \
  STEP8_0(in4, in7, out4, out7, cospi28_v, cospi4_v);                          \
  STEP8_0(in5, in6, out5, out6, cospi12_v, cospi20_v);                         \
                                                                               \
  out8 = vec_add(in8, in9);                                                    \
  out9 = vec_sub(in8, in9);                                                    \
  outA = vec_sub(inB, inA);                                                    \
  outB = vec_add(inA, inB);                                                    \
  outC = vec_add(inC, inD);                                                    \
  outD = vec_sub(inC, inD);                                                    \
  outE = vec_sub(inF, inE);                                                    \
  outF = vec_add(inE, inF);                                                    \
                                                                               \
  /* stage 4 */                                                                \
  STEP16_1(out0, out1, in1, in0, cospi16_v);                                   \
  STEP8_0(out2, out3, in2, in3, cospi24_v, cospi8_v);                          \
  in4 = vec_add(out4, out5);                                                   \
  in5 = vec_sub(out4, out5);                                                   \
  in6 = vec_sub(out7, out6);                                                   \
  in7 = vec_add(out6, out7);                                                   \
                                                                               \
  in8 = out8;                                                                  \
  inF = outF;                                                                  \
  tmp16_0 = vec_mergeh(out9, outE);                                            \
  tmp16_1 = vec_mergel(out9, outE);                                            \
  temp10 = vec_sub(vec_mulo(tmp16_0, cospi24_v), vec_mule(tmp16_0, cospi8_v)); \
  temp11 = vec_sub(vec_mulo(tmp16_1, cospi24_v), vec_mule(tmp16_1, cospi8_v)); \
  DCT_CONST_ROUND_SHIFT(temp10);                                               \
  DCT_CONST_ROUND_SHIFT(temp11);                                               \
  in9 = vec_packs(temp10, temp11);                                             \
  temp10 = vec_add(vec_mule(tmp16_0, cospi24_v), vec_mulo(tmp16_0, cospi8_v)); \
  temp11 = vec_add(vec_mule(tmp16_1, cospi24_v), vec_mulo(tmp16_1, cospi8_v)); \
  DCT_CONST_ROUND_SHIFT(temp10);                                               \
  DCT_CONST_ROUND_SHIFT(temp11);                                               \
  inE = vec_packs(temp10, temp11);                                             \
                                                                               \
  tmp16_0 = vec_mergeh(outA, outD);                                            \
  tmp16_1 = vec_mergel(outA, outD);                                            \
  temp10 =                                                                     \
      vec_sub(vec_mule(tmp16_0, cospi24m_v), vec_mulo(tmp16_0, cospi8_v));     \
  temp11 =                                                                     \
      vec_sub(vec_mule(tmp16_1, cospi24m_v), vec_mulo(tmp16_1, cospi8_v));     \
  DCT_CONST_ROUND_SHIFT(temp10);                                               \
  DCT_CONST_ROUND_SHIFT(temp11);                                               \
  inA = vec_packs(temp10, temp11);                                             \
  temp10 = vec_sub(vec_mulo(tmp16_0, cospi24_v), vec_mule(tmp16_0, cospi8_v)); \
  temp11 = vec_sub(vec_mulo(tmp16_1, cospi24_v), vec_mule(tmp16_1, cospi8_v)); \
  DCT_CONST_ROUND_SHIFT(temp10);                                               \
  DCT_CONST_ROUND_SHIFT(temp11);                                               \
  inD = vec_packs(temp10, temp11);                                             \
                                                                               \
  inB = outB;                                                                  \
  inC = outC;                                                                  \
                                                                               \
  /* stage 5 */                                                                \
  out0 = vec_add(in0, in3);                                                    \
  out1 = vec_add(in1, in2);                                                    \
  out2 = vec_sub(in1, in2);                                                    \
  out3 = vec_sub(in0, in3);                                                    \
  out4 = in4;                                                                  \
  STEP16_1(in6, in5, out5, out6, cospi16_v);                                   \
  out7 = in7;                                                                  \
                                                                               \
  out8 = vec_add(in8, inB);                                                    \
  out9 = vec_add(in9, inA);                                                    \
  outA = vec_sub(in9, inA);                                                    \
  outB = vec_sub(in8, inB);                                                    \
  outC = vec_sub(inF, inC);                                                    \
  outD = vec_sub(inE, inD);                                                    \
  outE = vec_add(inD, inE);                                                    \
  outF = vec_add(inC, inF);                                                    \
                                                                               \
  /* stage 6 */                                                                \
  in0 = vec_add(out0, out7);                                                   \
  in1 = vec_add(out1, out6);                                                   \
  in2 = vec_add(out2, out5);                                                   \
  in3 = vec_add(out3, out4);                                                   \
  in4 = vec_sub(out3, out4);                                                   \
  in5 = vec_sub(out2, out5);                                                   \
  in6 = vec_sub(out1, out6);                                                   \
  in7 = vec_sub(out0, out7);                                                   \
  in8 = out8;                                                                  \
  in9 = out9;                                                                  \
  STEP16_1(outD, outA, inA, inD, cospi16_v);                                   \
  STEP16_1(outC, outB, inB, inC, cospi16_v);                                   \
  inE = outE;                                                                  \
  inF = outF;                                                                  \
                                                                               \
  /* stage 7 */                                                                \
  out0 = vec_add(in0, inF);                                                    \
  out1 = vec_add(in1, inE);                                                    \
  out2 = vec_add(in2, inD);                                                    \
  out3 = vec_add(in3, inC);                                                    \
  out4 = vec_add(in4, inB);                                                    \
  out5 = vec_add(in5, inA);                                                    \
  out6 = vec_add(in6, in9);                                                    \
  out7 = vec_add(in7, in8);                                                    \
  out8 = vec_sub(in7, in8);                                                    \
  out9 = vec_sub(in6, in9);                                                    \
  outA = vec_sub(in5, inA);                                                    \
  outB = vec_sub(in4, inB);                                                    \
  outC = vec_sub(in3, inC);                                                    \
  outD = vec_sub(in2, inD);                                                    \
  outE = vec_sub(in1, inE);                                                    \
  outF = vec_sub(in0, inF);

#define PIXEL_ADD_STORE16(in0, in1, dst, offset) \
  d_uh = (int16x8_t)vec_mergeh(dst, zerov);      \
  d_ul = (int16x8_t)vec_mergel(dst, zerov);      \
  PIXEL_ADD(in0, d_uh, add, shift6);             \
  PIXEL_ADD(in1, d_ul, add, shift6);             \
  vec_vsx_st(vec_packsu(d_uh, d_ul), offset, dest);

static void half_idct16x8_vsx(int16x8_t *src) {
  int16x8_t tmp0[8], tmp1[8];
  int32x4_t temp10, temp11, temp20, temp21, temp30;
  int16x8_t tmp16_0, tmp16_1;
  ROUND_SHIFT_INIT;

  TRANSPOSE8x8(src[0], src[2], src[4], src[6], src[8], src[10], src[12],
               src[14], tmp0[0], tmp0[1], tmp0[2], tmp0[3], tmp0[4], tmp0[5],
               tmp0[6], tmp0[7]);
  TRANSPOSE8x8(src[1], src[3], src[5], src[7], src[9], src[11], src[13],
               src[15], tmp1[0], tmp1[1], tmp1[2], tmp1[3], tmp1[4], tmp1[5],
               tmp1[6], tmp1[7]);
  IDCT16(tmp0[0], tmp0[1], tmp0[2], tmp0[3], tmp0[4], tmp0[5], tmp0[6], tmp0[7],
         tmp1[0], tmp1[1], tmp1[2], tmp1[3], tmp1[4], tmp1[5], tmp1[6], tmp1[7],
         src[0], src[2], src[4], src[6], src[8], src[10], src[12], src[14],
         src[1], src[3], src[5], src[7], src[9], src[11], src[13], src[15]);
}

void vpx_idct16_vsx(int16x8_t *src0, int16x8_t *src1) {
  int16x8_t tmp0[8], tmp1[8], tmp2[8], tmp3[8];
  int32x4_t temp10, temp11, temp20, temp21, temp30;
  int16x8_t tmp16_0, tmp16_1;
  ROUND_SHIFT_INIT;

  TRANSPOSE8x8(src0[0], src0[2], src0[4], src0[6], src0[8], src0[10], src0[12],
               src0[14], tmp0[0], tmp0[1], tmp0[2], tmp0[3], tmp0[4], tmp0[5],
               tmp0[6], tmp0[7]);
  TRANSPOSE8x8(src0[1], src0[3], src0[5], src0[7], src0[9], src0[11], src0[13],
               src0[15], tmp1[0], tmp1[1], tmp1[2], tmp1[3], tmp1[4], tmp1[5],
               tmp1[6], tmp1[7]);
  TRANSPOSE8x8(src1[0], src1[2], src1[4], src1[6], src1[8], src1[10], src1[12],
               src1[14], tmp2[0], tmp2[1], tmp2[2], tmp2[3], tmp2[4], tmp2[5],
               tmp2[6], tmp2[7]);
  TRANSPOSE8x8(src1[1], src1[3], src1[5], src1[7], src1[9], src1[11], src1[13],
               src1[15], tmp3[0], tmp3[1], tmp3[2], tmp3[3], tmp3[4], tmp3[5],
               tmp3[6], tmp3[7]);

  IDCT16(tmp0[0], tmp0[1], tmp0[2], tmp0[3], tmp0[4], tmp0[5], tmp0[6], tmp0[7],
         tmp1[0], tmp1[1], tmp1[2], tmp1[3], tmp1[4], tmp1[5], tmp1[6], tmp1[7],
         src0[0], src0[2], src0[4], src0[6], src0[8], src0[10], src0[12],
         src0[14], src1[0], src1[2], src1[4], src1[6], src1[8], src1[10],
         src1[12], src1[14]);

  IDCT16(tmp2[0], tmp2[1], tmp2[2], tmp2[3], tmp2[4], tmp2[5], tmp2[6], tmp2[7],
         tmp3[0], tmp3[1], tmp3[2], tmp3[3], tmp3[4], tmp3[5], tmp3[6], tmp3[7],
         src0[1], src0[3], src0[5], src0[7], src0[9], src0[11], src0[13],
         src0[15], src1[1], src1[3], src1[5], src1[7], src1[9], src1[11],
         src1[13], src1[15]);
}

void vpx_round_store16x16_vsx(int16x8_t *src0, int16x8_t *src1, uint8_t *dest,
                              int stride) {
  uint8x16_t destv[16];
  int16x8_t d_uh, d_ul;
  uint8x16_t zerov = vec_splat_u8(0);
  uint16x8_t shift6 = vec_splat_u16(6);
  int16x8_t add = vec_sl(vec_splat_s16(8), vec_splat_u16(2));

  // load dest
  LOAD_INPUT16(vec_vsx_ld, dest, 0, stride, destv);

  PIXEL_ADD_STORE16(src0[0], src0[1], destv[0], 0);
  PIXEL_ADD_STORE16(src0[2], src0[3], destv[1], stride);
  PIXEL_ADD_STORE16(src0[4], src0[5], destv[2], 2 * stride);
  PIXEL_ADD_STORE16(src0[6], src0[7], destv[3], 3 * stride);
  PIXEL_ADD_STORE16(src0[8], src0[9], destv[4], 4 * stride);
  PIXEL_ADD_STORE16(src0[10], src0[11], destv[5], 5 * stride);
  PIXEL_ADD_STORE16(src0[12], src0[13], destv[6], 6 * stride);
  PIXEL_ADD_STORE16(src0[14], src0[15], destv[7], 7 * stride);

  PIXEL_ADD_STORE16(src1[0], src1[1], destv[8], 8 * stride);
  PIXEL_ADD_STORE16(src1[2], src1[3], destv[9], 9 * stride);
  PIXEL_ADD_STORE16(src1[4], src1[5], destv[10], 10 * stride);
  PIXEL_ADD_STORE16(src1[6], src1[7], destv[11], 11 * stride);
  PIXEL_ADD_STORE16(src1[8], src1[9], destv[12], 12 * stride);
  PIXEL_ADD_STORE16(src1[10], src1[11], destv[13], 13 * stride);
  PIXEL_ADD_STORE16(src1[12], src1[13], destv[14], 14 * stride);
  PIXEL_ADD_STORE16(src1[14], src1[15], destv[15], 15 * stride);
}
void vpx_idct16x16_256_add_vsx(const tran_low_t *input, uint8_t *dest,
                               int stride) {
  int16x8_t src0[16], src1[16];
  int16x8_t tmp0[8], tmp1[8], tmp2[8], tmp3[8];
  int32x4_t temp10, temp11, temp20, temp21, temp30;
  int16x8_t tmp16_0, tmp16_1;
  ROUND_SHIFT_INIT;

  LOAD_INPUT16(load_tran_low, input, 0, 8 * sizeof(*input), src0);
  LOAD_INPUT16(load_tran_low, input, 8 * 8 * 2 * sizeof(*input),
               8 * sizeof(*input), src1);

  // transform rows
  // transform the upper half of 16x16 matrix
  half_idct16x8_vsx(src0);
  TRANSPOSE8x8(src0[0], src0[2], src0[4], src0[6], src0[8], src0[10], src0[12],
               src0[14], tmp0[0], tmp0[1], tmp0[2], tmp0[3], tmp0[4], tmp0[5],
               tmp0[6], tmp0[7]);
  TRANSPOSE8x8(src0[1], src0[3], src0[5], src0[7], src0[9], src0[11], src0[13],
               src0[15], tmp1[0], tmp1[1], tmp1[2], tmp1[3], tmp1[4], tmp1[5],
               tmp1[6], tmp1[7]);

  // transform the lower half of 16x16 matrix
  half_idct16x8_vsx(src1);
  TRANSPOSE8x8(src1[0], src1[2], src1[4], src1[6], src1[8], src1[10], src1[12],
               src1[14], tmp2[0], tmp2[1], tmp2[2], tmp2[3], tmp2[4], tmp2[5],
               tmp2[6], tmp2[7]);
  TRANSPOSE8x8(src1[1], src1[3], src1[5], src1[7], src1[9], src1[11], src1[13],
               src1[15], tmp3[0], tmp3[1], tmp3[2], tmp3[3], tmp3[4], tmp3[5],
               tmp3[6], tmp3[7]);

  // transform columns
  // left half first
  IDCT16(tmp0[0], tmp0[1], tmp0[2], tmp0[3], tmp0[4], tmp0[5], tmp0[6], tmp0[7],
         tmp2[0], tmp2[1], tmp2[2], tmp2[3], tmp2[4], tmp2[5], tmp2[6], tmp2[7],
         src0[0], src0[2], src0[4], src0[6], src0[8], src0[10], src0[12],
         src0[14], src1[0], src1[2], src1[4], src1[6], src1[8], src1[10],
         src1[12], src1[14]);
  // right half
  IDCT16(tmp1[0], tmp1[1], tmp1[2], tmp1[3], tmp1[4], tmp1[5], tmp1[6], tmp1[7],
         tmp3[0], tmp3[1], tmp3[2], tmp3[3], tmp3[4], tmp3[5], tmp3[6], tmp3[7],
         src0[1], src0[3], src0[5], src0[7], src0[9], src0[11], src0[13],
         src0[15], src1[1], src1[3], src1[5], src1[7], src1[9], src1[11],
         src1[13], src1[15]);

  vpx_round_store16x16_vsx(src0, src1, dest, stride);
}

#define LOAD_8x32(load, in00, in01, in02, in03, in10, in11, in12, in13, in20, \
                  in21, in22, in23, in30, in31, in32, in33, in40, in41, in42, \
                  in43, in50, in51, in52, in53, in60, in61, in62, in63, in70, \
                  in71, in72, in73, offset)                                   \
  /* load the first row from the 8x32 block*/                                 \
  in00 = load(offset, input);                                                 \
  in01 = load(offset + 16, input);                                            \
  in02 = load(offset + 2 * 16, input);                                        \
  in03 = load(offset + 3 * 16, input);                                        \
                                                                              \
  in10 = load(offset + 4 * 16, input);                                        \
  in11 = load(offset + 5 * 16, input);                                        \
  in12 = load(offset + 6 * 16, input);                                        \
  in13 = load(offset + 7 * 16, input);                                        \
                                                                              \
  in20 = load(offset + 8 * 16, input);                                        \
  in21 = load(offset + 9 * 16, input);                                        \
  in22 = load(offset + 10 * 16, input);                                       \
  in23 = load(offset + 11 * 16, input);                                       \
                                                                              \
  in30 = load(offset + 12 * 16, input);                                       \
  in31 = load(offset + 13 * 16, input);                                       \
  in32 = load(offset + 14 * 16, input);                                       \
  in33 = load(offset + 15 * 16, input);                                       \
                                                                              \
  in40 = load(offset + 16 * 16, input);                                       \
  in41 = load(offset + 17 * 16, input);                                       \
  in42 = load(offset + 18 * 16, input);                                       \
  in43 = load(offset + 19 * 16, input);                                       \
                                                                              \
  in50 = load(offset + 20 * 16, input);                                       \
  in51 = load(offset + 21 * 16, input);                                       \
  in52 = load(offset + 22 * 16, input);                                       \
  in53 = load(offset + 23 * 16, input);                                       \
                                                                              \
  in60 = load(offset + 24 * 16, input);                                       \
  in61 = load(offset + 25 * 16, input);                                       \
  in62 = load(offset + 26 * 16, input);                                       \
  in63 = load(offset + 27 * 16, input);                                       \
                                                                              \
  /* load the last row from the 8x32 block*/                                  \
  in70 = load(offset + 28 * 16, input);                                       \
  in71 = load(offset + 29 * 16, input);                                       \
  in72 = load(offset + 30 * 16, input);                                       \
  in73 = load(offset + 31 * 16, input);

/* for the: temp1 = -step[x] * cospi_q + step[y] * cospi_z
 *          temp2 = step[x] * cospi_z + step[y] * cospi_q */
#define STEP32(inpt0, inpt1, outpt0, outpt1, cospi0, cospi1)              \
  tmp16_0 = vec_mergeh(inpt0, inpt1);                                     \
  tmp16_1 = vec_mergel(inpt0, inpt1);                                     \
  temp10 = vec_sub(vec_mulo(tmp16_0, cospi1), vec_mule(tmp16_0, cospi0)); \
  temp11 = vec_sub(vec_mulo(tmp16_1, cospi1), vec_mule(tmp16_1, cospi0)); \
  DCT_CONST_ROUND_SHIFT(temp10);                                          \
  DCT_CONST_ROUND_SHIFT(temp11);                                          \
  outpt0 = vec_packs(temp10, temp11);                                     \
  temp10 = vec_add(vec_mule(tmp16_0, cospi1), vec_mulo(tmp16_0, cospi0)); \
  temp11 = vec_add(vec_mule(tmp16_1, cospi1), vec_mulo(tmp16_1, cospi0)); \
  DCT_CONST_ROUND_SHIFT(temp10);                                          \
  DCT_CONST_ROUND_SHIFT(temp11);                                          \
  outpt1 = vec_packs(temp10, temp11);

/* for the: temp1 = -step[x] * cospi_q - step[y] * cospi_z
 *          temp2 = -step[x] * cospi_z + step[y] * cospi_q */
#define STEP32_1(inpt0, inpt1, outpt0, outpt1, cospi0, cospi1, cospi1m)    \
  tmp16_0 = vec_mergeh(inpt0, inpt1);                                      \
  tmp16_1 = vec_mergel(inpt0, inpt1);                                      \
  temp10 = vec_sub(vec_mulo(tmp16_0, cospi1m), vec_mule(tmp16_0, cospi0)); \
  temp11 = vec_sub(vec_mulo(tmp16_1, cospi1m), vec_mule(tmp16_1, cospi0)); \
  DCT_CONST_ROUND_SHIFT(temp10);                                           \
  DCT_CONST_ROUND_SHIFT(temp11);                                           \
  outpt0 = vec_packs(temp10, temp11);                                      \
  temp10 = vec_sub(vec_mulo(tmp16_0, cospi0), vec_mule(tmp16_0, cospi1));  \
  temp11 = vec_sub(vec_mulo(tmp16_1, cospi0), vec_mule(tmp16_1, cospi1));  \
  DCT_CONST_ROUND_SHIFT(temp10);                                           \
  DCT_CONST_ROUND_SHIFT(temp11);                                           \
  outpt1 = vec_packs(temp10, temp11);

#define IDCT32(in0, in1, in2, in3, out)                                \
                                                                       \
  /* stage 1 */                                                        \
  /* out[0][0] = in[0][0]; */                                          \
  out[0][1] = in2[0];                                                  \
  out[0][2] = in1[0];                                                  \
  out[0][3] = in3[0];                                                  \
  out[0][4] = in0[4];                                                  \
  out[0][5] = in2[4];                                                  \
  out[0][6] = in1[4];                                                  \
  out[0][7] = in3[4];                                                  \
  out[1][0] = in0[2];                                                  \
  out[1][1] = in2[2];                                                  \
  out[1][2] = in1[2];                                                  \
  out[1][3] = in3[2];                                                  \
  out[1][4] = in0[6];                                                  \
  out[1][5] = in2[6];                                                  \
  out[1][6] = in1[6];                                                  \
  out[1][7] = in3[6];                                                  \
                                                                       \
  STEP8_0(in0[1], in3[7], out[2][0], out[3][7], cospi31_v, cospi1_v);  \
  STEP8_0(in2[1], in1[7], out[2][1], out[3][6], cospi15_v, cospi17_v); \
  STEP8_0(in1[1], in2[7], out[2][2], out[3][5], cospi23_v, cospi9_v);  \
  STEP8_0(in3[1], in0[7], out[2][3], out[3][4], cospi7_v, cospi25_v);  \
  STEP8_0(in0[5], in3[3], out[2][4], out[3][3], cospi27_v, cospi5_v);  \
  STEP8_0(in2[5], in1[3], out[2][5], out[3][2], cospi11_v, cospi21_v); \
  STEP8_0(in1[5], in2[3], out[2][6], out[3][1], cospi19_v, cospi13_v); \
  STEP8_0(in3[5], in0[3], out[2][7], out[3][0], cospi3_v, cospi29_v);  \
                                                                       \
  /* stage 2 */                                                        \
  /* in0[0] = out[0][0]; */                                            \
  in0[1] = out[0][1];                                                  \
  in0[2] = out[0][2];                                                  \
  in0[3] = out[0][3];                                                  \
  in0[4] = out[0][4];                                                  \
  in0[5] = out[0][5];                                                  \
  in0[6] = out[0][6];                                                  \
  in0[7] = out[0][7];                                                  \
                                                                       \
  STEP8_0(out[1][0], out[1][7], in1[0], in1[7], cospi30_v, cospi2_v);  \
  STEP8_0(out[1][1], out[1][6], in1[1], in1[6], cospi14_v, cospi18_v); \
  STEP8_0(out[1][2], out[1][5], in1[2], in1[5], cospi22_v, cospi10_v); \
  STEP8_0(out[1][3], out[1][4], in1[3], in1[4], cospi6_v, cospi26_v);  \
                                                                       \
  in2[0] = vec_add(out[2][0], out[2][1]);                              \
  in2[1] = vec_sub(out[2][0], out[2][1]);                              \
  in2[2] = vec_sub(out[2][3], out[2][2]);                              \
  in2[3] = vec_add(out[2][3], out[2][2]);                              \
  in2[4] = vec_add(out[2][4], out[2][5]);                              \
  in2[5] = vec_sub(out[2][4], out[2][5]);                              \
  in2[6] = vec_sub(out[2][7], out[2][6]);                              \
  in2[7] = vec_add(out[2][7], out[2][6]);                              \
  in3[0] = vec_add(out[3][0], out[3][1]);                              \
  in3[1] = vec_sub(out[3][0], out[3][1]);                              \
  in3[2] = vec_sub(out[3][3], out[3][2]);                              \
  in3[3] = vec_add(out[3][3], out[3][2]);                              \
  in3[4] = vec_add(out[3][4], out[3][5]);                              \
  in3[5] = vec_sub(out[3][4], out[3][5]);                              \
  in3[6] = vec_sub(out[3][7], out[3][6]);                              \
  in3[7] = vec_add(out[3][6], out[3][7]);                              \
                                                                       \
  /* stage 3 */                                                        \
  out[0][0] = in0[0];                                                  \
  out[0][1] = in0[1];                                                  \
  out[0][2] = in0[2];                                                  \
  out[0][3] = in0[3];                                                  \
                                                                       \
  STEP8_0(in0[4], in0[7], out[0][4], out[0][7], cospi28_v, cospi4_v);  \
  STEP8_0(in0[5], in0[6], out[0][5], out[0][6], cospi12_v, cospi20_v); \
                                                                       \
  out[1][0] = vec_add(in1[0], in1[1]);                                 \
  out[1][1] = vec_sub(in1[0], in1[1]);                                 \
  out[1][2] = vec_sub(in1[3], in1[2]);                                 \
  out[1][3] = vec_add(in1[2], in1[3]);                                 \
  out[1][4] = vec_add(in1[4], in1[5]);                                 \
  out[1][5] = vec_sub(in1[4], in1[5]);                                 \
  out[1][6] = vec_sub(in1[7], in1[6]);                                 \
  out[1][7] = vec_add(in1[6], in1[7]);                                 \
                                                                       \
  out[2][0] = in2[0];                                                  \
  out[3][7] = in3[7];                                                  \
  STEP32(in2[1], in3[6], out[2][1], out[3][6], cospi4_v, cospi28_v);   \
  STEP32_1(in2[2], in3[5], out[2][2], out[3][5], cospi28_v, cospi4_v,  \
           cospi4m_v);                                                 \
  out[2][3] = in2[3];                                                  \
  out[2][4] = in2[4];                                                  \
  STEP32(in2[5], in3[2], out[2][5], out[3][2], cospi20_v, cospi12_v);  \
  STEP32_1(in2[6], in3[1], out[2][6], out[3][1], cospi12_v, cospi20_v, \
           cospi20m_v);                                                \
  out[2][7] = in2[7];                                                  \
  out[3][0] = in3[0];                                                  \
  out[3][3] = in3[3];                                                  \
  out[3][4] = in3[4];                                                  \
                                                                       \
  /* stage 4 */                                                        \
  STEP16_1(out[0][0], out[0][1], in0[1], in0[0], cospi16_v);           \
  STEP8_0(out[0][2], out[0][3], in0[2], in0[3], cospi24_v, cospi8_v);  \
  in0[4] = vec_add(out[0][4], out[0][5]);                              \
  in0[5] = vec_sub(out[0][4], out[0][5]);                              \
  in0[6] = vec_sub(out[0][7], out[0][6]);                              \
  in0[7] = vec_add(out[0][7], out[0][6]);                              \
                                                                       \
  in1[0] = out[1][0];                                                  \
  in1[7] = out[1][7];                                                  \
  STEP32(out[1][1], out[1][6], in1[1], in1[6], cospi8_v, cospi24_v);   \
  STEP32_1(out[1][2], out[1][5], in1[2], in1[5], cospi24_v, cospi8_v,  \
           cospi8m_v);                                                 \
  in1[3] = out[1][3];                                                  \
  in1[4] = out[1][4];                                                  \
                                                                       \
  in2[0] = vec_add(out[2][0], out[2][3]);                              \
  in2[1] = vec_add(out[2][1], out[2][2]);                              \
  in2[2] = vec_sub(out[2][1], out[2][2]);                              \
  in2[3] = vec_sub(out[2][0], out[2][3]);                              \
  in2[4] = vec_sub(out[2][7], out[2][4]);                              \
  in2[5] = vec_sub(out[2][6], out[2][5]);                              \
  in2[6] = vec_add(out[2][5], out[2][6]);                              \
  in2[7] = vec_add(out[2][4], out[2][7]);                              \
                                                                       \
  in3[0] = vec_add(out[3][0], out[3][3]);                              \
  in3[1] = vec_add(out[3][1], out[3][2]);                              \
  in3[2] = vec_sub(out[3][1], out[3][2]);                              \
  in3[3] = vec_sub(out[3][0], out[3][3]);                              \
  in3[4] = vec_sub(out[3][7], out[3][4]);                              \
  in3[5] = vec_sub(out[3][6], out[3][5]);                              \
  in3[6] = vec_add(out[3][5], out[3][6]);                              \
  in3[7] = vec_add(out[3][4], out[3][7]);                              \
                                                                       \
  /* stage 5 */                                                        \
  out[0][0] = vec_add(in0[0], in0[3]);                                 \
  out[0][1] = vec_add(in0[1], in0[2]);                                 \
  out[0][2] = vec_sub(in0[1], in0[2]);                                 \
  out[0][3] = vec_sub(in0[0], in0[3]);                                 \
  out[0][4] = in0[4];                                                  \
  STEP16_1(in0[6], in0[5], out[0][5], out[0][6], cospi16_v);           \
  out[0][7] = in0[7];                                                  \
                                                                       \
  out[1][0] = vec_add(in1[0], in1[3]);                                 \
  out[1][1] = vec_add(in1[1], in1[2]);                                 \
  out[1][2] = vec_sub(in1[1], in1[2]);                                 \
  out[1][3] = vec_sub(in1[0], in1[3]);                                 \
  out[1][4] = vec_sub(in1[7], in1[4]);                                 \
  out[1][5] = vec_sub(in1[6], in1[5]);                                 \
  out[1][6] = vec_add(in1[5], in1[6]);                                 \
  out[1][7] = vec_add(in1[4], in1[7]);                                 \
                                                                       \
  out[2][0] = in2[0];                                                  \
  out[2][1] = in2[1];                                                  \
  STEP32(in2[2], in3[5], out[2][2], out[3][5], cospi8_v, cospi24_v);   \
  STEP32(in2[3], in3[4], out[2][3], out[3][4], cospi8_v, cospi24_v);   \
  STEP32_1(in2[4], in3[3], out[2][4], out[3][3], cospi24_v, cospi8_v,  \
           cospi8m_v);                                                 \
  STEP32_1(in2[5], in3[2], out[2][5], out[3][2], cospi24_v, cospi8_v,  \
           cospi8m_v);                                                 \
  out[2][6] = in2[6];                                                  \
  out[2][7] = in2[7];                                                  \
  out[3][0] = in3[0];                                                  \
  out[3][1] = in3[1];                                                  \
  out[3][6] = in3[6];                                                  \
  out[3][7] = in3[7];                                                  \
                                                                       \
  /* stage 6 */                                                        \
  in0[0] = vec_add(out[0][0], out[0][7]);                              \
  in0[1] = vec_add(out[0][1], out[0][6]);                              \
  in0[2] = vec_add(out[0][2], out[0][5]);                              \
  in0[3] = vec_add(out[0][3], out[0][4]);                              \
  in0[4] = vec_sub(out[0][3], out[0][4]);                              \
  in0[5] = vec_sub(out[0][2], out[0][5]);                              \
  in0[6] = vec_sub(out[0][1], out[0][6]);                              \
  in0[7] = vec_sub(out[0][0], out[0][7]);                              \
  in1[0] = out[1][0];                                                  \
  in1[1] = out[1][1];                                                  \
  STEP16_1(out[1][5], out[1][2], in1[2], in1[5], cospi16_v);           \
  STEP16_1(out[1][4], out[1][3], in1[3], in1[4], cospi16_v);           \
  in1[6] = out[1][6];                                                  \
  in1[7] = out[1][7];                                                  \
                                                                       \
  in2[0] = vec_add(out[2][0], out[2][7]);                              \
  in2[1] = vec_add(out[2][1], out[2][6]);                              \
  in2[2] = vec_add(out[2][2], out[2][5]);                              \
  in2[3] = vec_add(out[2][3], out[2][4]);                              \
  in2[4] = vec_sub(out[2][3], out[2][4]);                              \
  in2[5] = vec_sub(out[2][2], out[2][5]);                              \
  in2[6] = vec_sub(out[2][1], out[2][6]);                              \
  in2[7] = vec_sub(out[2][0], out[2][7]);                              \
                                                                       \
  in3[0] = vec_sub(out[3][7], out[3][0]);                              \
  in3[1] = vec_sub(out[3][6], out[3][1]);                              \
  in3[2] = vec_sub(out[3][5], out[3][2]);                              \
  in3[3] = vec_sub(out[3][4], out[3][3]);                              \
  in3[4] = vec_add(out[3][4], out[3][3]);                              \
  in3[5] = vec_add(out[3][5], out[3][2]);                              \
  in3[6] = vec_add(out[3][6], out[3][1]);                              \
  in3[7] = vec_add(out[3][7], out[3][0]);                              \
                                                                       \
  /* stage 7 */                                                        \
  out[0][0] = vec_add(in0[0], in1[7]);                                 \
  out[0][1] = vec_add(in0[1], in1[6]);                                 \
  out[0][2] = vec_add(in0[2], in1[5]);                                 \
  out[0][3] = vec_add(in0[3], in1[4]);                                 \
  out[0][4] = vec_add(in0[4], in1[3]);                                 \
  out[0][5] = vec_add(in0[5], in1[2]);                                 \
  out[0][6] = vec_add(in0[6], in1[1]);                                 \
  out[0][7] = vec_add(in0[7], in1[0]);                                 \
  out[1][0] = vec_sub(in0[7], in1[0]);                                 \
  out[1][1] = vec_sub(in0[6], in1[1]);                                 \
  out[1][2] = vec_sub(in0[5], in1[2]);                                 \
  out[1][3] = vec_sub(in0[4], in1[3]);                                 \
  out[1][4] = vec_sub(in0[3], in1[4]);                                 \
  out[1][5] = vec_sub(in0[2], in1[5]);                                 \
  out[1][6] = vec_sub(in0[1], in1[6]);                                 \
  out[1][7] = vec_sub(in0[0], in1[7]);                                 \
                                                                       \
  out[2][0] = in2[0];                                                  \
  out[2][1] = in2[1];                                                  \
  out[2][2] = in2[2];                                                  \
  out[2][3] = in2[3];                                                  \
  STEP16_1(in3[3], in2[4], out[2][4], out[3][3], cospi16_v);           \
  STEP16_1(in3[2], in2[5], out[2][5], out[3][2], cospi16_v);           \
  STEP16_1(in3[1], in2[6], out[2][6], out[3][1], cospi16_v);           \
  STEP16_1(in3[0], in2[7], out[2][7], out[3][0], cospi16_v);           \
  out[3][4] = in3[4];                                                  \
  out[3][5] = in3[5];                                                  \
  out[3][6] = in3[6];                                                  \
  out[3][7] = in3[7];                                                  \
                                                                       \
  /* final */                                                          \
  in0[0] = vec_add(out[0][0], out[3][7]);                              \
  in0[1] = vec_add(out[0][1], out[3][6]);                              \
  in0[2] = vec_add(out[0][2], out[3][5]);                              \
  in0[3] = vec_add(out[0][3], out[3][4]);                              \
  in0[4] = vec_add(out[0][4], out[3][3]);                              \
  in0[5] = vec_add(out[0][5], out[3][2]);                              \
  in0[6] = vec_add(out[0][6], out[3][1]);                              \
  in0[7] = vec_add(out[0][7], out[3][0]);                              \
  in1[0] = vec_add(out[1][0], out[2][7]);                              \
  in1[1] = vec_add(out[1][1], out[2][6]);                              \
  in1[2] = vec_add(out[1][2], out[2][5]);                              \
  in1[3] = vec_add(out[1][3], out[2][4]);                              \
  in1[4] = vec_add(out[1][4], out[2][3]);                              \
  in1[5] = vec_add(out[1][5], out[2][2]);                              \
  in1[6] = vec_add(out[1][6], out[2][1]);                              \
  in1[7] = vec_add(out[1][7], out[2][0]);                              \
  in2[0] = vec_sub(out[1][7], out[2][0]);                              \
  in2[1] = vec_sub(out[1][6], out[2][1]);                              \
  in2[2] = vec_sub(out[1][5], out[2][2]);                              \
  in2[3] = vec_sub(out[1][4], out[2][3]);                              \
  in2[4] = vec_sub(out[1][3], out[2][4]);                              \
  in2[5] = vec_sub(out[1][2], out[2][5]);                              \
  in2[6] = vec_sub(out[1][1], out[2][6]);                              \
  in2[7] = vec_sub(out[1][0], out[2][7]);                              \
  in3[0] = vec_sub(out[0][7], out[3][0]);                              \
  in3[1] = vec_sub(out[0][6], out[3][1]);                              \
  in3[2] = vec_sub(out[0][5], out[3][2]);                              \
  in3[3] = vec_sub(out[0][4], out[3][3]);                              \
  in3[4] = vec_sub(out[0][3], out[3][4]);                              \
  in3[5] = vec_sub(out[0][2], out[3][5]);                              \
  in3[6] = vec_sub(out[0][1], out[3][6]);                              \
  in3[7] = vec_sub(out[0][0], out[3][7]);

// NOT A FULL TRANSPOSE! Transposes just each 8x8 block in each row,
// does not transpose rows
#define TRANSPOSE_8x32(in, out)                                                \
  /* transpose 4 of 8x8 blocks */                                              \
  TRANSPOSE8x8(in[0][0], in[0][1], in[0][2], in[0][3], in[0][4], in[0][5],     \
               in[0][6], in[0][7], out[0][0], out[0][1], out[0][2], out[0][3], \
               out[0][4], out[0][5], out[0][6], out[0][7]);                    \
  TRANSPOSE8x8(in[1][0], in[1][1], in[1][2], in[1][3], in[1][4], in[1][5],     \
               in[1][6], in[1][7], out[1][0], out[1][1], out[1][2], out[1][3], \
               out[1][4], out[1][5], out[1][6], out[1][7]);                    \
  TRANSPOSE8x8(in[2][0], in[2][1], in[2][2], in[2][3], in[2][4], in[2][5],     \
               in[2][6], in[2][7], out[2][0], out[2][1], out[2][2], out[2][3], \
               out[2][4], out[2][5], out[2][6], out[2][7]);                    \
  TRANSPOSE8x8(in[3][0], in[3][1], in[3][2], in[3][3], in[3][4], in[3][5],     \
               in[3][6], in[3][7], out[3][0], out[3][1], out[3][2], out[3][3], \
               out[3][4], out[3][5], out[3][6], out[3][7]);

#define PIXEL_ADD_STORE32(in0, in1, in2, in3, step)          \
  dst = vec_vsx_ld((step) * stride, dest);                   \
  d_uh = (int16x8_t)vec_mergeh(dst, zerov);                  \
  d_ul = (int16x8_t)vec_mergel(dst, zerov);                  \
  PIXEL_ADD(in0, d_uh, add, shift6);                         \
  PIXEL_ADD(in1, d_ul, add, shift6);                         \
  vec_vsx_st(vec_packsu(d_uh, d_ul), (step) * stride, dest); \
  dst = vec_vsx_ld((step) * stride + 16, dest);              \
  d_uh = (int16x8_t)vec_mergeh(dst, zerov);                  \
  d_ul = (int16x8_t)vec_mergel(dst, zerov);                  \
  PIXEL_ADD(in2, d_uh, add, shift6);                         \
  PIXEL_ADD(in3, d_ul, add, shift6);                         \
  vec_vsx_st(vec_packsu(d_uh, d_ul), (step) * stride + 16, dest);

#define ADD_STORE_BLOCK(in, offset)                                        \
  PIXEL_ADD_STORE32(in[0][0], in[1][0], in[2][0], in[3][0], (offset) + 0); \
  PIXEL_ADD_STORE32(in[0][1], in[1][1], in[2][1], in[3][1], (offset) + 1); \
  PIXEL_ADD_STORE32(in[0][2], in[1][2], in[2][2], in[3][2], (offset) + 2); \
  PIXEL_ADD_STORE32(in[0][3], in[1][3], in[2][3], in[3][3], (offset) + 3); \
  PIXEL_ADD_STORE32(in[0][4], in[1][4], in[2][4], in[3][4], (offset) + 4); \
  PIXEL_ADD_STORE32(in[0][5], in[1][5], in[2][5], in[3][5], (offset) + 5); \
  PIXEL_ADD_STORE32(in[0][6], in[1][6], in[2][6], in[3][6], (offset) + 6); \
  PIXEL_ADD_STORE32(in[0][7], in[1][7], in[2][7], in[3][7], (offset) + 7);

void vpx_idct32x32_1024_add_vsx(const tran_low_t *input, uint8_t *dest,
                                int stride) {
  int16x8_t src0[4][8], src1[4][8], src2[4][8], src3[4][8], tmp[4][8];
  int16x8_t tmp16_0, tmp16_1;
  int32x4_t temp10, temp11, temp20, temp21, temp30;
  uint8x16_t dst;
  int16x8_t d_uh, d_ul;
  int16x8_t add = vec_sl(vec_splat_s16(8), vec_splat_u16(2));
  uint16x8_t shift6 = vec_splat_u16(6);
  uint8x16_t zerov = vec_splat_u8(0);

  ROUND_SHIFT_INIT;

  LOAD_8x32(load_tran_low, src0[0][0], src0[1][0], src0[2][0], src0[3][0],
            src0[0][1], src0[1][1], src0[2][1], src0[3][1], src0[0][2],
            src0[1][2], src0[2][2], src0[3][2], src0[0][3], src0[1][3],
            src0[2][3], src0[3][3], src0[0][4], src0[1][4], src0[2][4],
            src0[3][4], src0[0][5], src0[1][5], src0[2][5], src0[3][5],
            src0[0][6], src0[1][6], src0[2][6], src0[3][6], src0[0][7],
            src0[1][7], src0[2][7], src0[3][7], 0);
  // Rows
  // transpose the first row of 8x8 blocks
  TRANSPOSE_8x32(src0, tmp);
  // transform the 32x8 column
  IDCT32(tmp[0], tmp[1], tmp[2], tmp[3], src0);
  TRANSPOSE_8x32(tmp, src0);

  LOAD_8x32(load_tran_low, src1[0][0], src1[1][0], src1[2][0], src1[3][0],
            src1[0][1], src1[1][1], src1[2][1], src1[3][1], src1[0][2],
            src1[1][2], src1[2][2], src1[3][2], src1[0][3], src1[1][3],
            src1[2][3], src1[3][3], src1[0][4], src1[1][4], src1[2][4],
            src1[3][4], src1[0][5], src1[1][5], src1[2][5], src1[3][5],
            src1[0][6], src1[1][6], src1[2][6], src1[3][6], src1[0][7],
            src1[1][7], src1[2][7], src1[3][7], 512);
  TRANSPOSE_8x32(src1, tmp);
  IDCT32(tmp[0], tmp[1], tmp[2], tmp[3], src1);
  TRANSPOSE_8x32(tmp, src1);

  LOAD_8x32(load_tran_low, src2[0][0], src2[1][0], src2[2][0], src2[3][0],
            src2[0][1], src2[1][1], src2[2][1], src2[3][1], src2[0][2],
            src2[1][2], src2[2][2], src2[3][2], src2[0][3], src2[1][3],
            src2[2][3], src2[3][3], src2[0][4], src2[1][4], src2[2][4],
            src2[3][4], src2[0][5], src2[1][5], src2[2][5], src2[3][5],
            src2[0][6], src2[1][6], src2[2][6], src2[3][6], src2[0][7],
            src2[1][7], src2[2][7], src2[3][7], 1024);
  TRANSPOSE_8x32(src2, tmp);
  IDCT32(tmp[0], tmp[1], tmp[2], tmp[3], src2);
  TRANSPOSE_8x32(tmp, src2);

  LOAD_8x32(load_tran_low, src3[0][0], src3[1][0], src3[2][0], src3[3][0],
            src3[0][1], src3[1][1], src3[2][1], src3[3][1], src3[0][2],
            src3[1][2], src3[2][2], src3[3][2], src3[0][3], src3[1][3],
            src3[2][3], src3[3][3], src3[0][4], src3[1][4], src3[2][4],
            src3[3][4], src3[0][5], src3[1][5], src3[2][5], src3[3][5],
            src3[0][6], src3[1][6], src3[2][6], src3[3][6], src3[0][7],
            src3[1][7], src3[2][7], src3[3][7], 1536);
  TRANSPOSE_8x32(src3, tmp);
  IDCT32(tmp[0], tmp[1], tmp[2], tmp[3], src3);
  TRANSPOSE_8x32(tmp, src3);

  // Columns
  IDCT32(src0[0], src1[0], src2[0], src3[0], tmp);
  IDCT32(src0[1], src1[1], src2[1], src3[1], tmp);
  IDCT32(src0[2], src1[2], src2[2], src3[2], tmp);
  IDCT32(src0[3], src1[3], src2[3], src3[3], tmp);

  ADD_STORE_BLOCK(src0, 0);
  ADD_STORE_BLOCK(src1, 8);
  ADD_STORE_BLOCK(src2, 16);
  ADD_STORE_BLOCK(src3, 24);
}

#define TRANSFORM_COLS           \
  v32_a = vec_add(v32_a, v32_c); \
  v32_d = vec_sub(v32_d, v32_b); \
  v32_e = vec_sub(v32_a, v32_d); \
  v32_e = vec_sra(v32_e, one);   \
  v32_b = vec_sub(v32_e, v32_b); \
  v32_c = vec_sub(v32_e, v32_c); \
  v32_a = vec_sub(v32_a, v32_b); \
  v32_d = vec_add(v32_d, v32_c); \
  v_a = vec_packs(v32_a, v32_b); \
  v_c = vec_packs(v32_c, v32_d);

#define TRANSPOSE_WHT             \
  tmp_a = vec_mergeh(v_a, v_c);   \
  tmp_c = vec_mergel(v_a, v_c);   \
  v_a = vec_mergeh(tmp_a, tmp_c); \
  v_c = vec_mergel(tmp_a, tmp_c);

void vpx_iwht4x4_16_add_vsx(const tran_low_t *input, uint8_t *dest,
                            int stride) {
  int16x8_t v_a = load_tran_low(0, input);
  int16x8_t v_c = load_tran_low(8 * sizeof(*input), input);
  int16x8_t tmp_a, tmp_c;
  uint16x8_t two = vec_splat_u16(2);
  uint32x4_t one = vec_splat_u32(1);
  int16x8_t tmp16_0, tmp16_1;
  int32x4_t v32_a, v32_c, v32_d, v32_b, v32_e;
  uint8x16_t dest0 = vec_vsx_ld(0, dest);
  uint8x16_t dest1 = vec_vsx_ld(stride, dest);
  uint8x16_t dest2 = vec_vsx_ld(2 * stride, dest);
  uint8x16_t dest3 = vec_vsx_ld(3 * stride, dest);
  int16x8_t d_u0 = (int16x8_t)unpack_to_u16_h(dest0);
  int16x8_t d_u1 = (int16x8_t)unpack_to_u16_h(dest1);
  int16x8_t d_u2 = (int16x8_t)unpack_to_u16_h(dest2);
  int16x8_t d_u3 = (int16x8_t)unpack_to_u16_h(dest3);
  uint8x16_t output_v;
  uint8_t tmp_dest[16];
  int i, j;

  v_a = vec_sra(v_a, two);
  v_c = vec_sra(v_c, two);

  TRANSPOSE_WHT;

  v32_a = vec_unpackh(v_a);
  v32_c = vec_unpackl(v_a);

  v32_d = vec_unpackh(v_c);
  v32_b = vec_unpackl(v_c);

  TRANSFORM_COLS;

  TRANSPOSE_WHT;

  v32_a = vec_unpackh(v_a);
  v32_c = vec_unpackl(v_a);
  v32_d = vec_unpackh(v_c);
  v32_b = vec_unpackl(v_c);

  TRANSFORM_COLS;

  PACK_STORE(v_a, v_c);
}

void vp9_iadst4_vsx(int16x8_t *in, int16x8_t *out) {
  int16x8_t sinpi_1_3_v, sinpi_4_2_v, sinpi_2_3_v, sinpi_1_4_v, sinpi_12_n3_v;
  int32x4_t v_v[5], u_v[4];
  int32x4_t zerov = vec_splat_s32(0);
  int16x8_t tmp0, tmp1;
  int16x8_t zero16v = vec_splat_s16(0);
  uint32x4_t shift16 = vec_sl(vec_splat_u32(8), vec_splat_u32(1));
  ROUND_SHIFT_INIT;

  sinpi_1_3_v = vec_mergel(sinpi_1_9_v, sinpi_3_9_v);
  sinpi_4_2_v = vec_mergel(sinpi_4_9_v, sinpi_2_9_v);
  sinpi_2_3_v = vec_mergel(sinpi_2_9_v, sinpi_3_9_v);
  sinpi_1_4_v = vec_mergel(sinpi_1_9_v, sinpi_4_9_v);
  sinpi_12_n3_v = vec_mergel(vec_add(sinpi_1_9_v, sinpi_2_9_v),
                             vec_sub(zero16v, sinpi_3_9_v));

  tmp0 = (int16x8_t)vec_mergeh((int32x4_t)in[0], (int32x4_t)in[1]);
  tmp1 = (int16x8_t)vec_mergel((int32x4_t)in[0], (int32x4_t)in[1]);
  in[0] = (int16x8_t)vec_mergeh((int32x4_t)tmp0, (int32x4_t)tmp1);
  in[1] = (int16x8_t)vec_mergel((int32x4_t)tmp0, (int32x4_t)tmp1);

  v_v[0] = vec_msum(in[0], sinpi_1_3_v, zerov);
  v_v[1] = vec_msum(in[1], sinpi_4_2_v, zerov);
  v_v[2] = vec_msum(in[0], sinpi_2_3_v, zerov);
  v_v[3] = vec_msum(in[1], sinpi_1_4_v, zerov);
  v_v[4] = vec_msum(in[0], sinpi_12_n3_v, zerov);

  in[0] = vec_sub(in[0], in[1]);
  in[1] = (int16x8_t)vec_sra((int32x4_t)in[1], shift16);
  in[0] = vec_add(in[0], in[1]);
  in[0] = (int16x8_t)vec_sl((int32x4_t)in[0], shift16);

  u_v[0] = vec_add(v_v[0], v_v[1]);
  u_v[1] = vec_sub(v_v[2], v_v[3]);
  u_v[2] = vec_msum(in[0], sinpi_1_3_v, zerov);
  u_v[3] = vec_sub(v_v[1], v_v[3]);
  u_v[3] = vec_add(u_v[3], v_v[4]);

  DCT_CONST_ROUND_SHIFT(u_v[0]);
  DCT_CONST_ROUND_SHIFT(u_v[1]);
  DCT_CONST_ROUND_SHIFT(u_v[2]);
  DCT_CONST_ROUND_SHIFT(u_v[3]);

  out[0] = vec_packs(u_v[0], u_v[1]);
  out[1] = vec_packs(u_v[2], u_v[3]);
}

#define MSUM_ROUND_SHIFT(a, b, cospi) \
  b = vec_msums(a, cospi, zerov);     \
  DCT_CONST_ROUND_SHIFT(b);

#define IADST_WRAPLOW(in0, in1, tmp0, tmp1, out, cospi) \
  MSUM_ROUND_SHIFT(in0, tmp0, cospi);                   \
  MSUM_ROUND_SHIFT(in1, tmp1, cospi);                   \
  out = vec_packs(tmp0, tmp1);

void vp9_iadst8_vsx(int16x8_t *in, int16x8_t *out) {
  int32x4_t tmp0[16], tmp1[16];

  int32x4_t zerov = vec_splat_s32(0);
  int16x8_t zero16v = vec_splat_s16(0);
  int16x8_t cospi_p02_p30_v = vec_mergel(cospi2_v, cospi30_v);
  int16x8_t cospi_p30_m02_v = vec_mergel(cospi30_v, cospi2m_v);
  int16x8_t cospi_p10_p22_v = vec_mergel(cospi10_v, cospi22_v);
  int16x8_t cospi_p22_m10_v = vec_mergel(cospi22_v, cospi10m_v);
  int16x8_t cospi_p18_p14_v = vec_mergel(cospi18_v, cospi14_v);
  int16x8_t cospi_p14_m18_v = vec_mergel(cospi14_v, cospi18m_v);
  int16x8_t cospi_p26_p06_v = vec_mergel(cospi26_v, cospi6_v);
  int16x8_t cospi_p06_m26_v = vec_mergel(cospi6_v, cospi26m_v);
  int16x8_t cospi_p08_p24_v = vec_mergel(cospi8_v, cospi24_v);
  int16x8_t cospi_p24_m08_v = vec_mergel(cospi24_v, cospi8m_v);
  int16x8_t cospi_m24_p08_v = vec_mergel(cospi24m_v, cospi8_v);
  int16x8_t cospi_p16_m16_v = vec_mergel(cospi16_v, cospi16m_v);
  ROUND_SHIFT_INIT;

  TRANSPOSE8x8(in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7], out[0],
               out[1], out[2], out[3], out[4], out[5], out[6], out[7]);

  // stage 1
  // interleave and multiply/add into 32-bit integer
  in[0] = vec_mergeh(out[7], out[0]);
  in[1] = vec_mergel(out[7], out[0]);
  in[2] = vec_mergeh(out[5], out[2]);
  in[3] = vec_mergel(out[5], out[2]);
  in[4] = vec_mergeh(out[3], out[4]);
  in[5] = vec_mergel(out[3], out[4]);
  in[6] = vec_mergeh(out[1], out[6]);
  in[7] = vec_mergel(out[1], out[6]);

  tmp1[0] = vec_msum(in[0], cospi_p02_p30_v, zerov);
  tmp1[1] = vec_msum(in[1], cospi_p02_p30_v, zerov);
  tmp1[2] = vec_msum(in[0], cospi_p30_m02_v, zerov);
  tmp1[3] = vec_msum(in[1], cospi_p30_m02_v, zerov);
  tmp1[4] = vec_msum(in[2], cospi_p10_p22_v, zerov);
  tmp1[5] = vec_msum(in[3], cospi_p10_p22_v, zerov);
  tmp1[6] = vec_msum(in[2], cospi_p22_m10_v, zerov);
  tmp1[7] = vec_msum(in[3], cospi_p22_m10_v, zerov);
  tmp1[8] = vec_msum(in[4], cospi_p18_p14_v, zerov);
  tmp1[9] = vec_msum(in[5], cospi_p18_p14_v, zerov);
  tmp1[10] = vec_msum(in[4], cospi_p14_m18_v, zerov);
  tmp1[11] = vec_msum(in[5], cospi_p14_m18_v, zerov);
  tmp1[12] = vec_msum(in[6], cospi_p26_p06_v, zerov);
  tmp1[13] = vec_msum(in[7], cospi_p26_p06_v, zerov);
  tmp1[14] = vec_msum(in[6], cospi_p06_m26_v, zerov);
  tmp1[15] = vec_msum(in[7], cospi_p06_m26_v, zerov);

  tmp0[0] = vec_add(tmp1[0], tmp1[8]);
  tmp0[1] = vec_add(tmp1[1], tmp1[9]);
  tmp0[2] = vec_add(tmp1[2], tmp1[10]);
  tmp0[3] = vec_add(tmp1[3], tmp1[11]);
  tmp0[4] = vec_add(tmp1[4], tmp1[12]);
  tmp0[5] = vec_add(tmp1[5], tmp1[13]);
  tmp0[6] = vec_add(tmp1[6], tmp1[14]);
  tmp0[7] = vec_add(tmp1[7], tmp1[15]);
  tmp0[8] = vec_sub(tmp1[0], tmp1[8]);
  tmp0[9] = vec_sub(tmp1[1], tmp1[9]);
  tmp0[10] = vec_sub(tmp1[2], tmp1[10]);
  tmp0[11] = vec_sub(tmp1[3], tmp1[11]);
  tmp0[12] = vec_sub(tmp1[4], tmp1[12]);
  tmp0[13] = vec_sub(tmp1[5], tmp1[13]);
  tmp0[14] = vec_sub(tmp1[6], tmp1[14]);
  tmp0[15] = vec_sub(tmp1[7], tmp1[15]);

  // shift and rounding
  DCT_CONST_ROUND_SHIFT(tmp0[0]);
  DCT_CONST_ROUND_SHIFT(tmp0[1]);
  DCT_CONST_ROUND_SHIFT(tmp0[2]);
  DCT_CONST_ROUND_SHIFT(tmp0[3]);
  DCT_CONST_ROUND_SHIFT(tmp0[4]);
  DCT_CONST_ROUND_SHIFT(tmp0[5]);
  DCT_CONST_ROUND_SHIFT(tmp0[6]);
  DCT_CONST_ROUND_SHIFT(tmp0[7]);
  DCT_CONST_ROUND_SHIFT(tmp0[8]);
  DCT_CONST_ROUND_SHIFT(tmp0[9]);
  DCT_CONST_ROUND_SHIFT(tmp0[10]);
  DCT_CONST_ROUND_SHIFT(tmp0[11]);
  DCT_CONST_ROUND_SHIFT(tmp0[12]);
  DCT_CONST_ROUND_SHIFT(tmp0[13]);
  DCT_CONST_ROUND_SHIFT(tmp0[14]);
  DCT_CONST_ROUND_SHIFT(tmp0[15]);

  // back to 16-bit
  out[0] = vec_packs(tmp0[0], tmp0[1]);
  out[1] = vec_packs(tmp0[2], tmp0[3]);
  out[2] = vec_packs(tmp0[4], tmp0[5]);
  out[3] = vec_packs(tmp0[6], tmp0[7]);
  out[4] = vec_packs(tmp0[8], tmp0[9]);
  out[5] = vec_packs(tmp0[10], tmp0[11]);
  out[6] = vec_packs(tmp0[12], tmp0[13]);
  out[7] = vec_packs(tmp0[14], tmp0[15]);

  // stage 2
  in[0] = vec_add(out[0], out[2]);
  in[1] = vec_add(out[1], out[3]);
  in[2] = vec_sub(out[0], out[2]);
  in[3] = vec_sub(out[1], out[3]);
  in[4] = vec_mergeh(out[4], out[5]);
  in[5] = vec_mergel(out[4], out[5]);
  in[6] = vec_mergeh(out[6], out[7]);
  in[7] = vec_mergel(out[6], out[7]);

  tmp1[0] = vec_msum(in[4], cospi_p08_p24_v, zerov);
  tmp1[1] = vec_msum(in[5], cospi_p08_p24_v, zerov);
  tmp1[2] = vec_msum(in[4], cospi_p24_m08_v, zerov);
  tmp1[3] = vec_msum(in[5], cospi_p24_m08_v, zerov);
  tmp1[4] = vec_msum(in[6], cospi_m24_p08_v, zerov);
  tmp1[5] = vec_msum(in[7], cospi_m24_p08_v, zerov);
  tmp1[6] = vec_msum(in[6], cospi_p08_p24_v, zerov);
  tmp1[7] = vec_msum(in[7], cospi_p08_p24_v, zerov);

  tmp0[0] = vec_add(tmp1[0], tmp1[4]);
  tmp0[1] = vec_add(tmp1[1], tmp1[5]);
  tmp0[2] = vec_add(tmp1[2], tmp1[6]);
  tmp0[3] = vec_add(tmp1[3], tmp1[7]);
  tmp0[4] = vec_sub(tmp1[0], tmp1[4]);
  tmp0[5] = vec_sub(tmp1[1], tmp1[5]);
  tmp0[6] = vec_sub(tmp1[2], tmp1[6]);
  tmp0[7] = vec_sub(tmp1[3], tmp1[7]);

  DCT_CONST_ROUND_SHIFT(tmp0[0]);
  DCT_CONST_ROUND_SHIFT(tmp0[1]);
  DCT_CONST_ROUND_SHIFT(tmp0[2]);
  DCT_CONST_ROUND_SHIFT(tmp0[3]);
  DCT_CONST_ROUND_SHIFT(tmp0[4]);
  DCT_CONST_ROUND_SHIFT(tmp0[5]);
  DCT_CONST_ROUND_SHIFT(tmp0[6]);
  DCT_CONST_ROUND_SHIFT(tmp0[7]);

  in[4] = vec_packs(tmp0[0], tmp0[1]);
  in[5] = vec_packs(tmp0[2], tmp0[3]);
  in[6] = vec_packs(tmp0[4], tmp0[5]);
  in[7] = vec_packs(tmp0[6], tmp0[7]);

  // stage 3
  out[0] = vec_mergeh(in[2], in[3]);
  out[1] = vec_mergel(in[2], in[3]);
  out[2] = vec_mergeh(in[6], in[7]);
  out[3] = vec_mergel(in[6], in[7]);

  IADST_WRAPLOW(out[0], out[1], tmp0[0], tmp0[1], in[2], cospi16_v);
  IADST_WRAPLOW(out[0], out[1], tmp0[0], tmp0[1], in[3], cospi_p16_m16_v);
  IADST_WRAPLOW(out[2], out[3], tmp0[0], tmp0[1], in[6], cospi16_v);
  IADST_WRAPLOW(out[2], out[3], tmp0[0], tmp0[1], in[7], cospi_p16_m16_v);

  out[0] = in[0];
  out[2] = in[6];
  out[4] = in[3];
  out[6] = in[5];

  out[1] = vec_sub(zero16v, in[4]);
  out[3] = vec_sub(zero16v, in[2]);
  out[5] = vec_sub(zero16v, in[7]);
  out[7] = vec_sub(zero16v, in[1]);
}

static void iadst16x8_vsx(int16x8_t *in, int16x8_t *out) {
  int32x4_t tmp0[32], tmp1[32];
  int16x8_t tmp16_0[8];
  int16x8_t cospi_p01_p31 = vec_mergel(cospi1_v, cospi31_v);
  int16x8_t cospi_p31_m01 = vec_mergel(cospi31_v, cospi1m_v);
  int16x8_t cospi_p05_p27 = vec_mergel(cospi5_v, cospi27_v);
  int16x8_t cospi_p27_m05 = vec_mergel(cospi27_v, cospi5m_v);
  int16x8_t cospi_p09_p23 = vec_mergel(cospi9_v, cospi23_v);
  int16x8_t cospi_p23_m09 = vec_mergel(cospi23_v, cospi9m_v);
  int16x8_t cospi_p13_p19 = vec_mergel(cospi13_v, cospi19_v);
  int16x8_t cospi_p19_m13 = vec_mergel(cospi19_v, cospi13m_v);
  int16x8_t cospi_p17_p15 = vec_mergel(cospi17_v, cospi15_v);
  int16x8_t cospi_p15_m17 = vec_mergel(cospi15_v, cospi17m_v);
  int16x8_t cospi_p21_p11 = vec_mergel(cospi21_v, cospi11_v);
  int16x8_t cospi_p11_m21 = vec_mergel(cospi11_v, cospi21m_v);
  int16x8_t cospi_p25_p07 = vec_mergel(cospi25_v, cospi7_v);
  int16x8_t cospi_p07_m25 = vec_mergel(cospi7_v, cospi25m_v);
  int16x8_t cospi_p29_p03 = vec_mergel(cospi29_v, cospi3_v);
  int16x8_t cospi_p03_m29 = vec_mergel(cospi3_v, cospi29m_v);
  int16x8_t cospi_p04_p28 = vec_mergel(cospi4_v, cospi28_v);
  int16x8_t cospi_p28_m04 = vec_mergel(cospi28_v, cospi4m_v);
  int16x8_t cospi_p20_p12 = vec_mergel(cospi20_v, cospi12_v);
  int16x8_t cospi_p12_m20 = vec_mergel(cospi12_v, cospi20m_v);
  int16x8_t cospi_m28_p04 = vec_mergel(cospi28m_v, cospi4_v);
  int16x8_t cospi_m12_p20 = vec_mergel(cospi12m_v, cospi20_v);
  int16x8_t cospi_p08_p24 = vec_mergel(cospi8_v, cospi24_v);
  int16x8_t cospi_p24_m08 = vec_mergel(cospi24_v, cospi8m_v);
  int16x8_t cospi_m24_p08 = vec_mergel(cospi24m_v, cospi8_v);
  int32x4_t zerov = vec_splat_s32(0);
  ROUND_SHIFT_INIT;

  tmp16_0[0] = vec_mergeh(in[15], in[0]);
  tmp16_0[1] = vec_mergel(in[15], in[0]);
  tmp16_0[2] = vec_mergeh(in[13], in[2]);
  tmp16_0[3] = vec_mergel(in[13], in[2]);
  tmp16_0[4] = vec_mergeh(in[11], in[4]);
  tmp16_0[5] = vec_mergel(in[11], in[4]);
  tmp16_0[6] = vec_mergeh(in[9], in[6]);
  tmp16_0[7] = vec_mergel(in[9], in[6]);
  tmp16_0[8] = vec_mergeh(in[7], in[8]);
  tmp16_0[9] = vec_mergel(in[7], in[8]);
  tmp16_0[10] = vec_mergeh(in[5], in[10]);
  tmp16_0[11] = vec_mergel(in[5], in[10]);
  tmp16_0[12] = vec_mergeh(in[3], in[12]);
  tmp16_0[13] = vec_mergel(in[3], in[12]);
  tmp16_0[14] = vec_mergeh(in[1], in[14]);
  tmp16_0[15] = vec_mergel(in[1], in[14]);

  tmp0[0] = vec_msum(tmp16_0[0], cospi_p01_p31, zerov);
  tmp0[1] = vec_msum(tmp16_0[1], cospi_p01_p31, zerov);
  tmp0[2] = vec_msum(tmp16_0[0], cospi_p31_m01, zerov);
  tmp0[3] = vec_msum(tmp16_0[1], cospi_p31_m01, zerov);
  tmp0[4] = vec_msum(tmp16_0[2], cospi_p05_p27, zerov);
  tmp0[5] = vec_msum(tmp16_0[3], cospi_p05_p27, zerov);
  tmp0[6] = vec_msum(tmp16_0[2], cospi_p27_m05, zerov);
  tmp0[7] = vec_msum(tmp16_0[3], cospi_p27_m05, zerov);
  tmp0[8] = vec_msum(tmp16_0[4], cospi_p09_p23, zerov);
  tmp0[9] = vec_msum(tmp16_0[5], cospi_p09_p23, zerov);
  tmp0[10] = vec_msum(tmp16_0[4], cospi_p23_m09, zerov);
  tmp0[11] = vec_msum(tmp16_0[5], cospi_p23_m09, zerov);
  tmp0[12] = vec_msum(tmp16_0[6], cospi_p13_p19, zerov);
  tmp0[13] = vec_msum(tmp16_0[7], cospi_p13_p19, zerov);
  tmp0[14] = vec_msum(tmp16_0[6], cospi_p19_m13, zerov);
  tmp0[15] = vec_msum(tmp16_0[7], cospi_p19_m13, zerov);
  tmp0[16] = vec_msum(tmp16_0[8], cospi_p17_p15, zerov);
  tmp0[17] = vec_msum(tmp16_0[9], cospi_p17_p15, zerov);
  tmp0[18] = vec_msum(tmp16_0[8], cospi_p15_m17, zerov);
  tmp0[19] = vec_msum(tmp16_0[9], cospi_p15_m17, zerov);
  tmp0[20] = vec_msum(tmp16_0[10], cospi_p21_p11, zerov);
  tmp0[21] = vec_msum(tmp16_0[11], cospi_p21_p11, zerov);
  tmp0[22] = vec_msum(tmp16_0[10], cospi_p11_m21, zerov);
  tmp0[23] = vec_msum(tmp16_0[11], cospi_p11_m21, zerov);
  tmp0[24] = vec_msum(tmp16_0[12], cospi_p25_p07, zerov);
  tmp0[25] = vec_msum(tmp16_0[13], cospi_p25_p07, zerov);
  tmp0[26] = vec_msum(tmp16_0[12], cospi_p07_m25, zerov);
  tmp0[27] = vec_msum(tmp16_0[13], cospi_p07_m25, zerov);
  tmp0[28] = vec_msum(tmp16_0[14], cospi_p29_p03, zerov);
  tmp0[29] = vec_msum(tmp16_0[15], cospi_p29_p03, zerov);
  tmp0[30] = vec_msum(tmp16_0[14], cospi_p03_m29, zerov);
  tmp0[31] = vec_msum(tmp16_0[15], cospi_p03_m29, zerov);

  tmp1[0] = vec_add(tmp0[0], tmp0[16]);
  tmp1[1] = vec_add(tmp0[1], tmp0[17]);
  tmp1[2] = vec_add(tmp0[2], tmp0[18]);
  tmp1[3] = vec_add(tmp0[3], tmp0[19]);
  tmp1[4] = vec_add(tmp0[4], tmp0[20]);
  tmp1[5] = vec_add(tmp0[5], tmp0[21]);
  tmp1[6] = vec_add(tmp0[6], tmp0[22]);
  tmp1[7] = vec_add(tmp0[7], tmp0[23]);
  tmp1[8] = vec_add(tmp0[8], tmp0[24]);
  tmp1[9] = vec_add(tmp0[9], tmp0[25]);
  tmp1[10] = vec_add(tmp0[10], tmp0[26]);
  tmp1[11] = vec_add(tmp0[11], tmp0[27]);
  tmp1[12] = vec_add(tmp0[12], tmp0[28]);
  tmp1[13] = vec_add(tmp0[13], tmp0[29]);
  tmp1[14] = vec_add(tmp0[14], tmp0[30]);
  tmp1[15] = vec_add(tmp0[15], tmp0[31]);
  tmp1[16] = vec_sub(tmp0[0], tmp0[16]);
  tmp1[17] = vec_sub(tmp0[1], tmp0[17]);
  tmp1[18] = vec_sub(tmp0[2], tmp0[18]);
  tmp1[19] = vec_sub(tmp0[3], tmp0[19]);
  tmp1[20] = vec_sub(tmp0[4], tmp0[20]);
  tmp1[21] = vec_sub(tmp0[5], tmp0[21]);
  tmp1[22] = vec_sub(tmp0[6], tmp0[22]);
  tmp1[23] = vec_sub(tmp0[7], tmp0[23]);
  tmp1[24] = vec_sub(tmp0[8], tmp0[24]);
  tmp1[25] = vec_sub(tmp0[9], tmp0[25]);
  tmp1[26] = vec_sub(tmp0[10], tmp0[26]);
  tmp1[27] = vec_sub(tmp0[11], tmp0[27]);
  tmp1[28] = vec_sub(tmp0[12], tmp0[28]);
  tmp1[29] = vec_sub(tmp0[13], tmp0[29]);
  tmp1[30] = vec_sub(tmp0[14], tmp0[30]);
  tmp1[31] = vec_sub(tmp0[15], tmp0[31]);

  DCT_CONST_ROUND_SHIFT(tmp1[0]);
  DCT_CONST_ROUND_SHIFT(tmp1[1]);
  DCT_CONST_ROUND_SHIFT(tmp1[2]);
  DCT_CONST_ROUND_SHIFT(tmp1[3]);
  DCT_CONST_ROUND_SHIFT(tmp1[4]);
  DCT_CONST_ROUND_SHIFT(tmp1[5]);
  DCT_CONST_ROUND_SHIFT(tmp1[6]);
  DCT_CONST_ROUND_SHIFT(tmp1[7]);
  DCT_CONST_ROUND_SHIFT(tmp1[8]);
  DCT_CONST_ROUND_SHIFT(tmp1[9]);
  DCT_CONST_ROUND_SHIFT(tmp1[10]);
  DCT_CONST_ROUND_SHIFT(tmp1[11]);
  DCT_CONST_ROUND_SHIFT(tmp1[12]);
  DCT_CONST_ROUND_SHIFT(tmp1[13]);
  DCT_CONST_ROUND_SHIFT(tmp1[14]);
  DCT_CONST_ROUND_SHIFT(tmp1[15]);
  DCT_CONST_ROUND_SHIFT(tmp1[16]);
  DCT_CONST_ROUND_SHIFT(tmp1[17]);
  DCT_CONST_ROUND_SHIFT(tmp1[18]);
  DCT_CONST_ROUND_SHIFT(tmp1[19]);
  DCT_CONST_ROUND_SHIFT(tmp1[20]);
  DCT_CONST_ROUND_SHIFT(tmp1[21]);
  DCT_CONST_ROUND_SHIFT(tmp1[22]);
  DCT_CONST_ROUND_SHIFT(tmp1[23]);
  DCT_CONST_ROUND_SHIFT(tmp1[24]);
  DCT_CONST_ROUND_SHIFT(tmp1[25]);
  DCT_CONST_ROUND_SHIFT(tmp1[26]);
  DCT_CONST_ROUND_SHIFT(tmp1[27]);
  DCT_CONST_ROUND_SHIFT(tmp1[28]);
  DCT_CONST_ROUND_SHIFT(tmp1[29]);
  DCT_CONST_ROUND_SHIFT(tmp1[30]);
  DCT_CONST_ROUND_SHIFT(tmp1[31]);

  in[0] = vec_packs(tmp1[0], tmp1[1]);
  in[1] = vec_packs(tmp1[2], tmp1[3]);
  in[2] = vec_packs(tmp1[4], tmp1[5]);
  in[3] = vec_packs(tmp1[6], tmp1[7]);
  in[4] = vec_packs(tmp1[8], tmp1[9]);
  in[5] = vec_packs(tmp1[10], tmp1[11]);
  in[6] = vec_packs(tmp1[12], tmp1[13]);
  in[7] = vec_packs(tmp1[14], tmp1[15]);
  in[8] = vec_packs(tmp1[16], tmp1[17]);
  in[9] = vec_packs(tmp1[18], tmp1[19]);
  in[10] = vec_packs(tmp1[20], tmp1[21]);
  in[11] = vec_packs(tmp1[22], tmp1[23]);
  in[12] = vec_packs(tmp1[24], tmp1[25]);
  in[13] = vec_packs(tmp1[26], tmp1[27]);
  in[14] = vec_packs(tmp1[28], tmp1[29]);
  in[15] = vec_packs(tmp1[30], tmp1[31]);

  // stage 2
  tmp16_0[0] = vec_mergeh(in[8], in[9]);
  tmp16_0[1] = vec_mergel(in[8], in[9]);
  tmp16_0[2] = vec_mergeh(in[10], in[11]);
  tmp16_0[3] = vec_mergel(in[10], in[11]);
  tmp16_0[4] = vec_mergeh(in[12], in[13]);
  tmp16_0[5] = vec_mergel(in[12], in[13]);
  tmp16_0[6] = vec_mergeh(in[14], in[15]);
  tmp16_0[7] = vec_mergel(in[14], in[15]);

  tmp0[0] = vec_msum(tmp16_0[0], cospi_p04_p28, zerov);
  tmp0[1] = vec_msum(tmp16_0[1], cospi_p04_p28, zerov);
  tmp0[2] = vec_msum(tmp16_0[0], cospi_p28_m04, zerov);
  tmp0[3] = vec_msum(tmp16_0[1], cospi_p28_m04, zerov);
  tmp0[4] = vec_msum(tmp16_0[2], cospi_p20_p12, zerov);
  tmp0[5] = vec_msum(tmp16_0[3], cospi_p20_p12, zerov);
  tmp0[6] = vec_msum(tmp16_0[2], cospi_p12_m20, zerov);
  tmp0[7] = vec_msum(tmp16_0[3], cospi_p12_m20, zerov);
  tmp0[8] = vec_msum(tmp16_0[4], cospi_m28_p04, zerov);
  tmp0[9] = vec_msum(tmp16_0[5], cospi_m28_p04, zerov);
  tmp0[10] = vec_msum(tmp16_0[4], cospi_p04_p28, zerov);
  tmp0[11] = vec_msum(tmp16_0[5], cospi_p04_p28, zerov);
  tmp0[12] = vec_msum(tmp16_0[6], cospi_m12_p20, zerov);
  tmp0[13] = vec_msum(tmp16_0[7], cospi_m12_p20, zerov);
  tmp0[14] = vec_msum(tmp16_0[6], cospi_p20_p12, zerov);
  tmp0[15] = vec_msum(tmp16_0[7], cospi_p20_p12, zerov);

  tmp1[0] = vec_add(tmp0[0], tmp0[8]);
  tmp1[1] = vec_add(tmp0[1], tmp0[9]);
  tmp1[2] = vec_add(tmp0[2], tmp0[10]);
  tmp1[3] = vec_add(tmp0[3], tmp0[11]);
  tmp1[4] = vec_add(tmp0[4], tmp0[12]);
  tmp1[5] = vec_add(tmp0[5], tmp0[13]);
  tmp1[6] = vec_add(tmp0[6], tmp0[14]);
  tmp1[7] = vec_add(tmp0[7], tmp0[15]);
  tmp1[8] = vec_sub(tmp0[0], tmp0[8]);
  tmp1[9] = vec_sub(tmp0[1], tmp0[9]);
  tmp1[10] = vec_sub(tmp0[2], tmp0[10]);
  tmp1[11] = vec_sub(tmp0[3], tmp0[11]);
  tmp1[12] = vec_sub(tmp0[4], tmp0[12]);
  tmp1[13] = vec_sub(tmp0[5], tmp0[13]);
  tmp1[14] = vec_sub(tmp0[6], tmp0[14]);
  tmp1[15] = vec_sub(tmp0[7], tmp0[15]);

  DCT_CONST_ROUND_SHIFT(tmp1[0]);
  DCT_CONST_ROUND_SHIFT(tmp1[1]);
  DCT_CONST_ROUND_SHIFT(tmp1[2]);
  DCT_CONST_ROUND_SHIFT(tmp1[3]);
  DCT_CONST_ROUND_SHIFT(tmp1[4]);
  DCT_CONST_ROUND_SHIFT(tmp1[5]);
  DCT_CONST_ROUND_SHIFT(tmp1[6]);
  DCT_CONST_ROUND_SHIFT(tmp1[7]);
  DCT_CONST_ROUND_SHIFT(tmp1[8]);
  DCT_CONST_ROUND_SHIFT(tmp1[9]);
  DCT_CONST_ROUND_SHIFT(tmp1[10]);
  DCT_CONST_ROUND_SHIFT(tmp1[11]);
  DCT_CONST_ROUND_SHIFT(tmp1[12]);
  DCT_CONST_ROUND_SHIFT(tmp1[13]);
  DCT_CONST_ROUND_SHIFT(tmp1[14]);
  DCT_CONST_ROUND_SHIFT(tmp1[15]);

  tmp16_0[0] = vec_add(in[0], in[4]);
  tmp16_0[1] = vec_add(in[1], in[5]);
  tmp16_0[2] = vec_add(in[2], in[6]);
  tmp16_0[3] = vec_add(in[3], in[7]);
  tmp16_0[4] = vec_sub(in[0], in[4]);
  tmp16_0[5] = vec_sub(in[1], in[5]);
  tmp16_0[6] = vec_sub(in[2], in[6]);
  tmp16_0[7] = vec_sub(in[3], in[7]);
  tmp16_0[8] = vec_packs(tmp1[0], tmp1[1]);
  tmp16_0[9] = vec_packs(tmp1[2], tmp1[3]);
  tmp16_0[10] = vec_packs(tmp1[4], tmp1[5]);
  tmp16_0[11] = vec_packs(tmp1[6], tmp1[7]);
  tmp16_0[12] = vec_packs(tmp1[8], tmp1[9]);
  tmp16_0[13] = vec_packs(tmp1[10], tmp1[11]);
  tmp16_0[14] = vec_packs(tmp1[12], tmp1[13]);
  tmp16_0[15] = vec_packs(tmp1[14], tmp1[15]);

  // stage 3
  in[0] = vec_mergeh(tmp16_0[4], tmp16_0[5]);
  in[1] = vec_mergel(tmp16_0[4], tmp16_0[5]);
  in[2] = vec_mergeh(tmp16_0[6], tmp16_0[7]);
  in[3] = vec_mergel(tmp16_0[6], tmp16_0[7]);
  in[4] = vec_mergeh(tmp16_0[12], tmp16_0[13]);
  in[5] = vec_mergel(tmp16_0[12], tmp16_0[13]);
  in[6] = vec_mergeh(tmp16_0[14], tmp16_0[15]);
  in[7] = vec_mergel(tmp16_0[14], tmp16_0[15]);

  tmp0[0] = vec_msum(in[0], cospi_p08_p24, zerov);
  tmp0[1] = vec_msum(in[1], cospi_p08_p24, zerov);
  tmp0[2] = vec_msum(in[0], cospi_p24_m08, zerov);
  tmp0[3] = vec_msum(in[1], cospi_p24_m08, zerov);
  tmp0[4] = vec_msum(in[2], cospi_m24_p08, zerov);
  tmp0[5] = vec_msum(in[3], cospi_m24_p08, zerov);
  tmp0[6] = vec_msum(in[2], cospi_p08_p24, zerov);
  tmp0[7] = vec_msum(in[3], cospi_p08_p24, zerov);
  tmp0[8] = vec_msum(in[4], cospi_p08_p24, zerov);
  tmp0[9] = vec_msum(in[5], cospi_p08_p24, zerov);
  tmp0[10] = vec_msum(in[4], cospi_p24_m08, zerov);
  tmp0[11] = vec_msum(in[5], cospi_p24_m08, zerov);
  tmp0[12] = vec_msum(in[6], cospi_m24_p08, zerov);
  tmp0[13] = vec_msum(in[7], cospi_m24_p08, zerov);
  tmp0[14] = vec_msum(in[6], cospi_p08_p24, zerov);
  tmp0[15] = vec_msum(in[7], cospi_p08_p24, zerov);

  tmp1[0] = vec_add(tmp0[0], tmp0[4]);
  tmp1[1] = vec_add(tmp0[1], tmp0[5]);
  tmp1[2] = vec_add(tmp0[2], tmp0[6]);
  tmp1[3] = vec_add(tmp0[3], tmp0[7]);
  tmp1[4] = vec_sub(tmp0[0], tmp0[4]);
  tmp1[5] = vec_sub(tmp0[1], tmp0[5]);
  tmp1[6] = vec_sub(tmp0[2], tmp0[6]);
  tmp1[7] = vec_sub(tmp0[3], tmp0[7]);
  tmp1[8] = vec_add(tmp0[8], tmp0[12]);
  tmp1[9] = vec_add(tmp0[9], tmp0[13]);
  tmp1[10] = vec_add(tmp0[10], tmp0[14]);
  tmp1[11] = vec_add(tmp0[11], tmp0[15]);
  tmp1[12] = vec_sub(tmp0[8], tmp0[12]);
  tmp1[13] = vec_sub(tmp0[9], tmp0[13]);
  tmp1[14] = vec_sub(tmp0[10], tmp0[14]);
  tmp1[15] = vec_sub(tmp0[11], tmp0[15]);

  DCT_CONST_ROUND_SHIFT(tmp1[0]);
  DCT_CONST_ROUND_SHIFT(tmp1[1]);
  DCT_CONST_ROUND_SHIFT(tmp1[2]);
  DCT_CONST_ROUND_SHIFT(tmp1[3]);
  DCT_CONST_ROUND_SHIFT(tmp1[4]);
  DCT_CONST_ROUND_SHIFT(tmp1[5]);
  DCT_CONST_ROUND_SHIFT(tmp1[6]);
  DCT_CONST_ROUND_SHIFT(tmp1[7]);
  DCT_CONST_ROUND_SHIFT(tmp1[8]);
  DCT_CONST_ROUND_SHIFT(tmp1[9]);
  DCT_CONST_ROUND_SHIFT(tmp1[10]);
  DCT_CONST_ROUND_SHIFT(tmp1[11]);
  DCT_CONST_ROUND_SHIFT(tmp1[12]);
  DCT_CONST_ROUND_SHIFT(tmp1[13]);
  DCT_CONST_ROUND_SHIFT(tmp1[14]);
  DCT_CONST_ROUND_SHIFT(tmp1[15]);

  in[0] = vec_add(tmp16_0[0], tmp16_0[2]);
  in[1] = vec_add(tmp16_0[1], tmp16_0[3]);
  in[2] = vec_sub(tmp16_0[0], tmp16_0[2]);
  in[3] = vec_sub(tmp16_0[1], tmp16_0[3]);
  in[4] = vec_packs(tmp1[0], tmp1[1]);
  in[5] = vec_packs(tmp1[2], tmp1[3]);
  in[6] = vec_packs(tmp1[4], tmp1[5]);
  in[7] = vec_packs(tmp1[6], tmp1[7]);
  in[8] = vec_add(tmp16_0[8], tmp16_0[10]);
  in[9] = vec_add(tmp16_0[9], tmp16_0[11]);
  in[10] = vec_sub(tmp16_0[8], tmp16_0[10]);
  in[11] = vec_sub(tmp16_0[9], tmp16_0[11]);
  in[12] = vec_packs(tmp1[8], tmp1[9]);
  in[13] = vec_packs(tmp1[10], tmp1[11]);
  in[14] = vec_packs(tmp1[12], tmp1[13]);
  in[15] = vec_packs(tmp1[14], tmp1[15]);

  // stage 4
  out[0] = vec_mergeh(in[2], in[3]);
  out[1] = vec_mergel(in[2], in[3]);
  out[2] = vec_mergeh(in[6], in[7]);
  out[3] = vec_mergel(in[6], in[7]);
  out[4] = vec_mergeh(in[10], in[11]);
  out[5] = vec_mergel(in[10], in[11]);
  out[6] = vec_mergeh(in[14], in[15]);
  out[7] = vec_mergel(in[14], in[15]);
}

void vpx_iadst16_vsx(int16x8_t *src0, int16x8_t *src1) {
  int16x8_t tmp0[16], tmp1[16], tmp2[8];
  int32x4_t tmp3, tmp4;
  int16x8_t zero16v = vec_splat_s16(0);
  int32x4_t zerov = vec_splat_s32(0);
  int16x8_t cospi_p16_m16 = vec_mergel(cospi16_v, cospi16m_v);
  int16x8_t cospi_m16_p16 = vec_mergel(cospi16m_v, cospi16_v);
  ROUND_SHIFT_INIT;

  TRANSPOSE8x8(src0[0], src0[2], src0[4], src0[6], src0[8], src0[10], src0[12],
               src0[14], tmp0[0], tmp0[1], tmp0[2], tmp0[3], tmp0[4], tmp0[5],
               tmp0[6], tmp0[7]);
  TRANSPOSE8x8(src1[0], src1[2], src1[4], src1[6], src1[8], src1[10], src1[12],
               src1[14], tmp1[0], tmp1[1], tmp1[2], tmp1[3], tmp1[4], tmp1[5],
               tmp1[6], tmp1[7]);
  TRANSPOSE8x8(src0[1], src0[3], src0[5], src0[7], src0[9], src0[11], src0[13],
               src0[15], tmp0[8], tmp0[9], tmp0[10], tmp0[11], tmp0[12],
               tmp0[13], tmp0[14], tmp0[15]);
  TRANSPOSE8x8(src1[1], src1[3], src1[5], src1[7], src1[9], src1[11], src1[13],
               src1[15], tmp1[8], tmp1[9], tmp1[10], tmp1[11], tmp1[12],
               tmp1[13], tmp1[14], tmp1[15]);

  iadst16x8_vsx(tmp0, tmp2);
  IADST_WRAPLOW(tmp2[0], tmp2[1], tmp3, tmp4, src0[14], cospi16m_v);
  IADST_WRAPLOW(tmp2[0], tmp2[1], tmp3, tmp4, src1[0], cospi_p16_m16);
  IADST_WRAPLOW(tmp2[2], tmp2[3], tmp3, tmp4, src0[8], cospi16_v);
  IADST_WRAPLOW(tmp2[2], tmp2[3], tmp3, tmp4, src1[6], cospi_m16_p16);
  IADST_WRAPLOW(tmp2[4], tmp2[5], tmp3, tmp4, src0[12], cospi16_v);
  IADST_WRAPLOW(tmp2[4], tmp2[5], tmp3, tmp4, src1[2], cospi_m16_p16);
  IADST_WRAPLOW(tmp2[6], tmp2[7], tmp3, tmp4, src0[10], cospi16m_v);
  IADST_WRAPLOW(tmp2[6], tmp2[7], tmp3, tmp4, src1[4], cospi_p16_m16);

  src0[0] = tmp0[0];
  src0[2] = vec_sub(zero16v, tmp0[8]);
  src0[4] = tmp0[12];
  src0[6] = vec_sub(zero16v, tmp0[4]);
  src1[8] = tmp0[5];
  src1[10] = vec_sub(zero16v, tmp0[13]);
  src1[12] = tmp0[9];
  src1[14] = vec_sub(zero16v, tmp0[1]);

  iadst16x8_vsx(tmp1, tmp2);
  IADST_WRAPLOW(tmp2[0], tmp2[1], tmp3, tmp4, src0[15], cospi16m_v);
  IADST_WRAPLOW(tmp2[0], tmp2[1], tmp3, tmp4, src1[1], cospi_p16_m16);
  IADST_WRAPLOW(tmp2[2], tmp2[3], tmp3, tmp4, src0[9], cospi16_v);
  IADST_WRAPLOW(tmp2[2], tmp2[3], tmp3, tmp4, src1[7], cospi_m16_p16);
  IADST_WRAPLOW(tmp2[4], tmp2[5], tmp3, tmp4, src0[13], cospi16_v);
  IADST_WRAPLOW(tmp2[4], tmp2[5], tmp3, tmp4, src1[3], cospi_m16_p16);
  IADST_WRAPLOW(tmp2[6], tmp2[7], tmp3, tmp4, src0[11], cospi16m_v);
  IADST_WRAPLOW(tmp2[6], tmp2[7], tmp3, tmp4, src1[5], cospi_p16_m16);

  src0[1] = tmp1[0];
  src0[3] = vec_sub(zero16v, tmp1[8]);
  src0[5] = tmp1[12];
  src0[7] = vec_sub(zero16v, tmp1[4]);
  src1[9] = tmp1[5];
  src1[11] = vec_sub(zero16v, tmp1[13]);
  src1[13] = tmp1[9];
  src1[15] = vec_sub(zero16v, tmp1[1]);
}
