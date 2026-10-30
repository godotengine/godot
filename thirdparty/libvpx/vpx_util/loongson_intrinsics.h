/*
 * Copyright (c) 2022 The WebM project authors. All Rights Reserved.
 *
 * Use of this source code is governed by a BSD-style license
 * that can be found in the LICENSE file in the root of the source
 * tree. An additional intellectual property rights grant can be found
 * in the file PATENTS.  All contributing project authors may
 * be found in the AUTHORS file in the root of the source tree.
 *
 */

#ifndef VPX_VPX_UTIL_LOONGSON_INTRINSICS_H_
#define VPX_VPX_UTIL_LOONGSON_INTRINSICS_H_

/*
 * Copyright (c) 2021 Loongson Technology Corporation Limited
 * All rights reserved.
 *
 * Use of this source code is governed by a BSD-style license
 * that can be found in the LICENSE file in the root of the source
 * tree. An additional intellectual property rights grant can be found
 * in the file PATENTS.  All contributing project authors may
 * be found in the AUTHORS file in the root of the source tree.
 *
 * Contributed by Shiyou Yin <yinshiyou-hf@loongson.cn>
 *                Xiwei Gu   <guxiwei-hf@loongson.cn>
 *                Lu Wang    <wanglu@loongson.cn>
 *
 * This file is a header file for loongarch builtin extension.
 *
 */

#ifndef LOONGSON_INTRINSICS_H
#define LOONGSON_INTRINSICS_H

/**
 * MAJOR version: Macro usage changes.
 * MINOR version: Add new functions, or bug fixes.
 * MICRO version: Comment changes or implementation changes.
 */
#define LSOM_VERSION_MAJOR 1
#define LSOM_VERSION_MINOR 2
#define LSOM_VERSION_MICRO 1

#define DUP2_ARG1(_INS, _IN0, _IN1, _OUT0, _OUT1) \
  {                                               \
    _OUT0 = _INS(_IN0);                           \
    _OUT1 = _INS(_IN1);                           \
  }

#define DUP2_ARG2(_INS, _IN0, _IN1, _IN2, _IN3, _OUT0, _OUT1) \
  {                                                           \
    _OUT0 = _INS(_IN0, _IN1);                                 \
    _OUT1 = _INS(_IN2, _IN3);                                 \
  }

#define DUP2_ARG3(_INS, _IN0, _IN1, _IN2, _IN3, _IN4, _IN5, _OUT0, _OUT1) \
  {                                                                       \
    _OUT0 = _INS(_IN0, _IN1, _IN2);                                       \
    _OUT1 = _INS(_IN3, _IN4, _IN5);                                       \
  }

#define DUP4_ARG1(_INS, _IN0, _IN1, _IN2, _IN3, _OUT0, _OUT1, _OUT2, _OUT3) \
  {                                                                         \
    DUP2_ARG1(_INS, _IN0, _IN1, _OUT0, _OUT1);                              \
    DUP2_ARG1(_INS, _IN2, _IN3, _OUT2, _OUT3);                              \
  }

#define DUP4_ARG2(_INS, _IN0, _IN1, _IN2, _IN3, _IN4, _IN5, _IN6, _IN7, _OUT0, \
                  _OUT1, _OUT2, _OUT3)                                         \
  {                                                                            \
    DUP2_ARG2(_INS, _IN0, _IN1, _IN2, _IN3, _OUT0, _OUT1);                     \
    DUP2_ARG2(_INS, _IN4, _IN5, _IN6, _IN7, _OUT2, _OUT3);                     \
  }

#define DUP4_ARG3(_INS, _IN0, _IN1, _IN2, _IN3, _IN4, _IN5, _IN6, _IN7, _IN8, \
                  _IN9, _IN10, _IN11, _OUT0, _OUT1, _OUT2, _OUT3)             \
  {                                                                           \
    DUP2_ARG3(_INS, _IN0, _IN1, _IN2, _IN3, _IN4, _IN5, _OUT0, _OUT1);        \
    DUP2_ARG3(_INS, _IN6, _IN7, _IN8, _IN9, _IN10, _IN11, _OUT2, _OUT3);      \
  }

#ifdef __loongarch_sx
#include <lsxintrin.h>
/*
 * =============================================================================
 * Description : Dot product & addition of byte vector elements
 * Arguments   : Inputs  - in_c, in_h, in_l
 *               Outputs - out
 *               Return Type - halfword
 * Details     : Signed byte elements from in_h are multiplied by
 *               signed byte elements from in_l, and then added adjacent to
 *               each other to get a result twice the size of input. Then
 *               the results are added to signed half-word elements from in_c.
 * Example     : out = __lsx_vdp2add_h_b(in_c, in_h, in_l)
 *        in_c : 1,2,3,4, 1,2,3,4
 *        in_h : 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
 *        in_l : 8,7,6,5, 4,3,2,1, 8,7,6,5, 4,3,2,1
 *         out : 23,40,41,26, 23,40,41,26
 * =============================================================================
 */
static inline __m128i __lsx_vdp2add_h_b(__m128i in_c, __m128i in_h,
                                        __m128i in_l) {
  __m128i out;

  out = __lsx_vmaddwev_h_b(in_c, in_h, in_l);
  out = __lsx_vmaddwod_h_b(out, in_h, in_l);
  return out;
}

/*
 * =============================================================================
 * Description : Dot product & addition of byte vector elements
 * Arguments   : Inputs  - in_c, in_h, in_l
 *               Outputs - out
 *               Return Type - halfword
 * Details     : Unsigned byte elements from in_h are multiplied by
 *               unsigned byte elements from in_l, and then added adjacent to
 *               each other to get a result twice the size of input.
 *               The results are added to signed half-word elements from in_c.
 * Example     : out = __lsx_vdp2add_h_bu(in_c, in_h, in_l)
 *        in_c : 1,2,3,4, 1,2,3,4
 *        in_h : 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
 *        in_l : 8,7,6,5, 4,3,2,1, 8,7,6,5, 4,3,2,1
 *         out : 23,40,41,26, 23,40,41,26
 * =============================================================================
 */
static inline __m128i __lsx_vdp2add_h_bu(__m128i in_c, __m128i in_h,
                                         __m128i in_l) {
  __m128i out;

  out = __lsx_vmaddwev_h_bu(in_c, in_h, in_l);
  out = __lsx_vmaddwod_h_bu(out, in_h, in_l);
  return out;
}

/*
 * =============================================================================
 * Description : Dot product & addition of byte vector elements
 * Arguments   : Inputs  - in_c, in_h, in_l
 *               Outputs - out
 *               Return Type - halfword
 * Details     : Unsigned byte elements from in_h are multiplied by
 *               signed byte elements from in_l, and then added adjacent to
 *               each other to get a result twice the size of input.
 *               The results are added to signed half-word elements from in_c.
 * Example     : out = __lsx_vdp2add_h_bu_b(in_c, in_h, in_l)
 *        in_c : 1,1,1,1, 1,1,1,1
 *        in_h : 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
 *        in_l : -1,-2,-3,-4, -5,-6,-7,-8, 1,2,3,4, 5,6,7,8
 *         out : -4,-24,-60,-112, 6,26,62,114
 * =============================================================================
 */
static inline __m128i __lsx_vdp2add_h_bu_b(__m128i in_c, __m128i in_h,
                                           __m128i in_l) {
  __m128i out;

  out = __lsx_vmaddwev_h_bu_b(in_c, in_h, in_l);
  out = __lsx_vmaddwod_h_bu_b(out, in_h, in_l);
  return out;
}

/*
 * =============================================================================
 * Description : Dot product & addition of half-word vector elements
 * Arguments   : Inputs  - in_c, in_h, in_l
 *               Outputs - out
 *               Return Type - __m128i
 * Details     : Signed half-word elements from in_h are multiplied by
 *               signed half-word elements from in_l, and then added adjacent to
 *               each other to get a result twice the size of input.
 *               Then the results are added to signed word elements from in_c.
 * Example     : out = __lsx_vdp2add_h_b(in_c, in_h, in_l)
 *        in_c : 1,2,3,4
 *        in_h : 1,2,3,4, 5,6,7,8
 *        in_l : 8,7,6,5, 4,3,2,1
 *         out : 23,40,41,26
 * =============================================================================
 */
static inline __m128i __lsx_vdp2add_w_h(__m128i in_c, __m128i in_h,
                                        __m128i in_l) {
  __m128i out;

  out = __lsx_vmaddwev_w_h(in_c, in_h, in_l);
  out = __lsx_vmaddwod_w_h(out, in_h, in_l);
  return out;
}

/*
 * =============================================================================
 * Description : Dot product of byte vector elements
 * Arguments   : Inputs  - in_h, in_l
 *               Outputs - out
 *               Return Type - halfword
 * Details     : Signed byte elements from in_h are multiplied by
 *               signed byte elements from in_l, and then added adjacent to
 *               each other to get a result twice the size of input.
 * Example     : out = __lsx_vdp2_h_b(in_h, in_l)
 *        in_h : 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
 *        in_l : 8,7,6,5, 4,3,2,1, 8,7,6,5, 4,3,2,1
 *         out : 22,38,38,22, 22,38,38,22
 * =============================================================================
 */
static inline __m128i __lsx_vdp2_h_b(__m128i in_h, __m128i in_l) {
  __m128i out;

  out = __lsx_vmulwev_h_b(in_h, in_l);
  out = __lsx_vmaddwod_h_b(out, in_h, in_l);
  return out;
}

/*
 * =============================================================================
 * Description : Dot product of byte vector elements
 * Arguments   : Inputs  - in_h, in_l
 *               Outputs - out
 *               Return Type - halfword
 * Details     : Unsigned byte elements from in_h are multiplied by
 *               unsigned byte elements from in_l, and then added adjacent to
 *               each other to get a result twice the size of input.
 * Example     : out = __lsx_vdp2_h_bu(in_h, in_l)
 *        in_h : 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
 *        in_l : 8,7,6,5, 4,3,2,1, 8,7,6,5, 4,3,2,1
 *         out : 22,38,38,22, 22,38,38,22
 * =============================================================================
 */
static inline __m128i __lsx_vdp2_h_bu(__m128i in_h, __m128i in_l) {
  __m128i out;

  out = __lsx_vmulwev_h_bu(in_h, in_l);
  out = __lsx_vmaddwod_h_bu(out, in_h, in_l);
  return out;
}

/*
 * =============================================================================
 * Description : Dot product of byte vector elements
 * Arguments   : Inputs  - in_h, in_l
 *               Outputs - out
 *               Return Type - halfword
 * Details     : Unsigned byte elements from in_h are multiplied by
 *               signed byte elements from in_l, and then added adjacent to
 *               each other to get a result twice the size of input.
 * Example     : out = __lsx_vdp2_h_bu_b(in_h, in_l)
 *        in_h : 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
 *        in_l : 8,7,6,5, 4,3,2,1, 8,7,6,5, 4,3,2,-1
 *         out : 22,38,38,22, 22,38,38,6
 * =============================================================================
 */
static inline __m128i __lsx_vdp2_h_bu_b(__m128i in_h, __m128i in_l) {
  __m128i out;

  out = __lsx_vmulwev_h_bu_b(in_h, in_l);
  out = __lsx_vmaddwod_h_bu_b(out, in_h, in_l);
  return out;
}

/*
 * =============================================================================
 * Description : Dot product of byte vector elements
 * Arguments   : Inputs  - in_h, in_l
 *               Outputs - out
 *               Return Type - halfword
 * Details     : Signed byte elements from in_h are multiplied by
 *               signed byte elements from in_l, and then added adjacent to
 *               each other to get a result twice the size of input.
 * Example     : out = __lsx_vdp2_w_h(in_h, in_l)
 *        in_h : 1,2,3,4, 5,6,7,8
 *        in_l : 8,7,6,5, 4,3,2,1
 *         out : 22,38,38,22
 * =============================================================================
 */
static inline __m128i __lsx_vdp2_w_h(__m128i in_h, __m128i in_l) {
  __m128i out;

  out = __lsx_vmulwev_w_h(in_h, in_l);
  out = __lsx_vmaddwod_w_h(out, in_h, in_l);
  return out;
}

/*
 * =============================================================================
 * Description : Dot product of byte vector elements
 * Arguments   : Inputs  - in_h, in_l
 *               Outputs - out
 *               Return Type - double
 * Details     : Signed byte elements from in_h are multiplied by
 *               signed byte elements from in_l, and then added adjacent to
 *               each other to get a result twice the size of input.
 * Example     : out = __lsx_vdp2_d_w(in_h, in_l)
 *        in_h : 1,2,3,4
 *        in_l : 8,7,6,5
 *         out : 22,38
 * =============================================================================
 */
static inline __m128i __lsx_vdp2_d_w(__m128i in_h, __m128i in_l) {
  __m128i out;

  out = __lsx_vmulwev_d_w(in_h, in_l);
  out = __lsx_vmaddwod_d_w(out, in_h, in_l);
  return out;
}

/*
 * =============================================================================
 * Description : Clip all halfword elements of input vector between min & max
 *               out = ((_in) < (min)) ? (min) : (((_in) > (max)) ? (max) :
 *               (_in))
 * Arguments   : Inputs  - _in  (input vector)
 *                       - min  (min threshold)
 *                       - max  (max threshold)
 *               Outputs - out  (output vector with clipped elements)
 *               Return Type - signed halfword
 * Example     : out = __lsx_vclip_h(_in)
 *         _in : -8,2,280,249, -8,255,280,249
 *         min : 1,1,1,1, 1,1,1,1
 *         max : 9,9,9,9, 9,9,9,9
 *         out : 1,2,9,9, 1,9,9,9
 * =============================================================================
 */
static inline __m128i __lsx_vclip_h(__m128i _in, __m128i min, __m128i max) {
  __m128i out;

  out = __lsx_vmax_h(min, _in);
  out = __lsx_vmin_h(max, out);
  return out;
}

/*
 * =============================================================================
 * Description : Set each element of vector between 0 and 255
 * Arguments   : Inputs  - _in
 *               Outputs - out
 *               Return Type - halfword
 * Details     : Signed byte elements from _in are clamped between 0 and 255.
 * Example     : out = __lsx_vclip255_h(_in)
 *         _in : -8,255,280,249, -8,255,280,249
 *         out : 0,255,255,249, 0,255,255,249
 * =============================================================================
 */
static inline __m128i __lsx_vclip255_h(__m128i _in) {
  __m128i out;

  out = __lsx_vmaxi_h(_in, 0);
  out = __lsx_vsat_hu(out, 7);
  return out;
}

/*
 * =============================================================================
 * Description : Set each element of vector between 0 and 255
 * Arguments   : Inputs  - _in
 *               Outputs - out
 *               Return Type - word
 * Details     : Signed byte elements from _in are clamped between 0 and 255.
 * Example     : out = __lsx_vclip255_w(_in)
 *         _in : -8,255,280,249
 *         out : 0,255,255,249
 * =============================================================================
 */
static inline __m128i __lsx_vclip255_w(__m128i _in) {
  __m128i out;

  out = __lsx_vmaxi_w(_in, 0);
  out = __lsx_vsat_wu(out, 7);
  return out;
}

/*
 * =============================================================================
 * Description : Swap two variables
 * Arguments   : Inputs  - _in0, _in1
 *               Outputs - _in0, _in1 (in-place)
 * Details     : Swapping of two input variables using xor
 * Example     : LSX_SWAP(_in0, _in1)
 *        _in0 : 1,2,3,4
 *        _in1 : 5,6,7,8
 *   _in0(out) : 5,6,7,8
 *   _in1(out) : 1,2,3,4
 * =============================================================================
 */
#define LSX_SWAP(_in0, _in1)         \
  {                                  \
    _in0 = __lsx_vxor_v(_in0, _in1); \
    _in1 = __lsx_vxor_v(_in0, _in1); \
    _in0 = __lsx_vxor_v(_in0, _in1); \
  }

/*
 * =============================================================================
 * Description : Transpose 4x4 block with word elements in vectors
 * Arguments   : Inputs  - in0, in1, in2, in3
 *               Outputs - out0, out1, out2, out3
 * Details     :
 * Example     :
 *               1, 2, 3, 4            1, 5, 9,13
 *               5, 6, 7, 8    to      2, 6,10,14
 *               9,10,11,12  =====>    3, 7,11,15
 *              13,14,15,16            4, 8,12,16
 * =============================================================================
 */
#define LSX_TRANSPOSE4x4_W(_in0, _in1, _in2, _in3, _out0, _out1, _out2, _out3) \
  {                                                                            \
    __m128i _t0, _t1, _t2, _t3;                                                \
                                                                               \
    _t0 = __lsx_vilvl_w(_in1, _in0);                                           \
    _t1 = __lsx_vilvh_w(_in1, _in0);                                           \
    _t2 = __lsx_vilvl_w(_in3, _in2);                                           \
    _t3 = __lsx_vilvh_w(_in3, _in2);                                           \
    _out0 = __lsx_vilvl_d(_t2, _t0);                                           \
    _out1 = __lsx_vilvh_d(_t2, _t0);                                           \
    _out2 = __lsx_vilvl_d(_t3, _t1);                                           \
    _out3 = __lsx_vilvh_d(_t3, _t1);                                           \
  }

/*
 * =============================================================================
 * Description : Transpose 8x8 block with byte elements in vectors
 * Arguments   : Inputs  - _in0, _in1, _in2, _in3, _in4, _in5, _in6, _in7
 *               Outputs - _out0, _out1, _out2, _out3, _out4, _out5, _out6,
 *               _out7
 * Details     : The rows of the matrix become columns, and the columns
 *               become rows.
 * Example     : LSX_TRANSPOSE8x8_B
 *        _in0 : 00,01,02,03,04,05,06,07, 00,00,00,00,00,00,00,00
 *        _in1 : 10,11,12,13,14,15,16,17, 00,00,00,00,00,00,00,00
 *        _in2 : 20,21,22,23,24,25,26,27, 00,00,00,00,00,00,00,00
 *        _in3 : 30,31,32,33,34,35,36,37, 00,00,00,00,00,00,00,00
 *        _in4 : 40,41,42,43,44,45,46,47, 00,00,00,00,00,00,00,00
 *        _in5 : 50,51,52,53,54,55,56,57, 00,00,00,00,00,00,00,00
 *        _in6 : 60,61,62,63,64,65,66,67, 00,00,00,00,00,00,00,00
 *        _in7 : 70,71,72,73,74,75,76,77, 00,00,00,00,00,00,00,00
 *
 *      _ out0 : 00,10,20,30,40,50,60,70, 00,00,00,00,00,00,00,00
 *      _ out1 : 01,11,21,31,41,51,61,71, 00,00,00,00,00,00,00,00
 *      _ out2 : 02,12,22,32,42,52,62,72, 00,00,00,00,00,00,00,00
 *      _ out3 : 03,13,23,33,43,53,63,73, 00,00,00,00,00,00,00,00
 *      _ out4 : 04,14,24,34,44,54,64,74, 00,00,00,00,00,00,00,00
 *      _ out5 : 05,15,25,35,45,55,65,75, 00,00,00,00,00,00,00,00
 *      _ out6 : 06,16,26,36,46,56,66,76, 00,00,00,00,00,00,00,00
 *      _ out7 : 07,17,27,37,47,57,67,77, 00,00,00,00,00,00,00,00
 * =============================================================================
 */
#define LSX_TRANSPOSE8x8_B(_in0, _in1, _in2, _in3, _in4, _in5, _in6, _in7,  \
                           _out0, _out1, _out2, _out3, _out4, _out5, _out6, \
                           _out7)                                           \
  {                                                                         \
    __m128i zero = { 0 };                                                   \
    __m128i shuf8 = { 0x0F0E0D0C0B0A0908, 0x1716151413121110 };             \
    __m128i _t0, _t1, _t2, _t3, _t4, _t5, _t6, _t7;                         \
                                                                            \
    _t0 = __lsx_vilvl_b(_in2, _in0);                                        \
    _t1 = __lsx_vilvl_b(_in3, _in1);                                        \
    _t2 = __lsx_vilvl_b(_in6, _in4);                                        \
    _t3 = __lsx_vilvl_b(_in7, _in5);                                        \
    _t4 = __lsx_vilvl_b(_t1, _t0);                                          \
    _t5 = __lsx_vilvh_b(_t1, _t0);                                          \
    _t6 = __lsx_vilvl_b(_t3, _t2);                                          \
    _t7 = __lsx_vilvh_b(_t3, _t2);                                          \
    _out0 = __lsx_vilvl_w(_t6, _t4);                                        \
    _out2 = __lsx_vilvh_w(_t6, _t4);                                        \
    _out4 = __lsx_vilvl_w(_t7, _t5);                                        \
    _out6 = __lsx_vilvh_w(_t7, _t5);                                        \
    _out1 = __lsx_vshuf_b(zero, _out0, shuf8);                              \
    _out3 = __lsx_vshuf_b(zero, _out2, shuf8);                              \
    _out5 = __lsx_vshuf_b(zero, _out4, shuf8);                              \
    _out7 = __lsx_vshuf_b(zero, _out6, shuf8);                              \
  }

/*
 * =============================================================================
 * Description : Transpose 8x8 block with half-word elements in vectors
 * Arguments   : Inputs  - in0, in1, in2, in3, in4, in5, in6, in7
 *               Outputs - out0, out1, out2, out3, out4, out5, out6, out7
 * Details     :
 * Example     :
 *              00,01,02,03,04,05,06,07           00,10,20,30,40,50,60,70
 *              10,11,12,13,14,15,16,17           01,11,21,31,41,51,61,71
 *              20,21,22,23,24,25,26,27           02,12,22,32,42,52,62,72
 *              30,31,32,33,34,35,36,37    to     03,13,23,33,43,53,63,73
 *              40,41,42,43,44,45,46,47  ======>  04,14,24,34,44,54,64,74
 *              50,51,52,53,54,55,56,57           05,15,25,35,45,55,65,75
 *              60,61,62,63,64,65,66,67           06,16,26,36,46,56,66,76
 *              70,71,72,73,74,75,76,77           07,17,27,37,47,57,67,77
 * =============================================================================
 */
#define LSX_TRANSPOSE8x8_H(_in0, _in1, _in2, _in3, _in4, _in5, _in6, _in7,  \
                           _out0, _out1, _out2, _out3, _out4, _out5, _out6, \
                           _out7)                                           \
  {                                                                         \
    __m128i _s0, _s1, _t0, _t1, _t2, _t3, _t4, _t5, _t6, _t7;               \
                                                                            \
    _s0 = __lsx_vilvl_h(_in6, _in4);                                        \
    _s1 = __lsx_vilvl_h(_in7, _in5);                                        \
    _t0 = __lsx_vilvl_h(_s1, _s0);                                          \
    _t1 = __lsx_vilvh_h(_s1, _s0);                                          \
    _s0 = __lsx_vilvh_h(_in6, _in4);                                        \
    _s1 = __lsx_vilvh_h(_in7, _in5);                                        \
    _t2 = __lsx_vilvl_h(_s1, _s0);                                          \
    _t3 = __lsx_vilvh_h(_s1, _s0);                                          \
    _s0 = __lsx_vilvl_h(_in2, _in0);                                        \
    _s1 = __lsx_vilvl_h(_in3, _in1);                                        \
    _t4 = __lsx_vilvl_h(_s1, _s0);                                          \
    _t5 = __lsx_vilvh_h(_s1, _s0);                                          \
    _s0 = __lsx_vilvh_h(_in2, _in0);                                        \
    _s1 = __lsx_vilvh_h(_in3, _in1);                                        \
    _t6 = __lsx_vilvl_h(_s1, _s0);                                          \
    _t7 = __lsx_vilvh_h(_s1, _s0);                                          \
                                                                            \
    _out0 = __lsx_vpickev_d(_t0, _t4);                                      \
    _out2 = __lsx_vpickev_d(_t1, _t5);                                      \
    _out4 = __lsx_vpickev_d(_t2, _t6);                                      \
    _out6 = __lsx_vpickev_d(_t3, _t7);                                      \
    _out1 = __lsx_vpickod_d(_t0, _t4);                                      \
    _out3 = __lsx_vpickod_d(_t1, _t5);                                      \
    _out5 = __lsx_vpickod_d(_t2, _t6);                                      \
    _out7 = __lsx_vpickod_d(_t3, _t7);                                      \
  }

/*
 * =============================================================================
 * Description : Transpose input 8x4 byte block into 4x8
 * Arguments   : Inputs  - _in0, _in1, _in2, _in3      (input 8x4 byte block)
 *               Outputs - _out0, _out1, _out2, _out3  (output 4x8 byte block)
 *               Return Type - as per RTYPE
 * Details     : The rows of the matrix become columns, and the columns become
 *               rows.
 * Example     : LSX_TRANSPOSE8x4_B
 *        _in0 : 00,01,02,03,00,00,00,00, 00,00,00,00,00,00,00,00
 *        _in1 : 10,11,12,13,00,00,00,00, 00,00,00,00,00,00,00,00
 *        _in2 : 20,21,22,23,00,00,00,00, 00,00,00,00,00,00,00,00
 *        _in3 : 30,31,32,33,00,00,00,00, 00,00,00,00,00,00,00,00
 *        _in4 : 40,41,42,43,00,00,00,00, 00,00,00,00,00,00,00,00
 *        _in5 : 50,51,52,53,00,00,00,00, 00,00,00,00,00,00,00,00
 *        _in6 : 60,61,62,63,00,00,00,00, 00,00,00,00,00,00,00,00
 *        _in7 : 70,71,72,73,00,00,00,00, 00,00,00,00,00,00,00,00
 *
 *       _out0 : 00,10,20,30,40,50,60,70, 00,00,00,00,00,00,00,00
 *       _out1 : 01,11,21,31,41,51,61,71, 00,00,00,00,00,00,00,00
 *       _out2 : 02,12,22,32,42,52,62,72, 00,00,00,00,00,00,00,00
 *       _out3 : 03,13,23,33,43,53,63,73, 00,00,00,00,00,00,00,00
 * =============================================================================
 */
#define LSX_TRANSPOSE8x4_B(_in0, _in1, _in2, _in3, _in4, _in5, _in6, _in7, \
                           _out0, _out1, _out2, _out3)                     \
  {                                                                        \
    __m128i _tmp0_m, _tmp1_m, _tmp2_m, _tmp3_m;                            \
                                                                           \
    _tmp0_m = __lsx_vpackev_w(_in4, _in0);                                 \
    _tmp1_m = __lsx_vpackev_w(_in5, _in1);                                 \
    _tmp2_m = __lsx_vilvl_b(_tmp1_m, _tmp0_m);                             \
    _tmp0_m = __lsx_vpackev_w(_in6, _in2);                                 \
    _tmp1_m = __lsx_vpackev_w(_in7, _in3);                                 \
                                                                           \
    _tmp3_m = __lsx_vilvl_b(_tmp1_m, _tmp0_m);                             \
    _tmp0_m = __lsx_vilvl_h(_tmp3_m, _tmp2_m);                             \
    _tmp1_m = __lsx_vilvh_h(_tmp3_m, _tmp2_m);                             \
                                                                           \
    _out0 = __lsx_vilvl_w(_tmp1_m, _tmp0_m);                               \
    _out2 = __lsx_vilvh_w(_tmp1_m, _tmp0_m);                               \
    _out1 = __lsx_vilvh_d(_out2, _out0);                                   \
    _out3 = __lsx_vilvh_d(_out0, _out2);                                   \
  }

/*
 * =============================================================================
 * Description : Transpose 16x8 block with byte elements in vectors
 * Arguments   : Inputs  - in0, in1, in2, in3, in4, in5, in6, in7, in8
 *                         in9, in10, in11, in12, in13, in14, in15
 *               Outputs - out0, out1, out2, out3, out4, out5, out6, out7
 * Details     :
 * Example     :
 *              000,001,002,003,004,005,006,007
 *              008,009,010,011,012,013,014,015
 *              016,017,018,019,020,021,022,023
 *              024,025,026,027,028,029,030,031
 *              032,033,034,035,036,037,038,039
 *              040,041,042,043,044,045,046,047        000,008,...,112,120
 *              048,049,050,051,052,053,054,055        001,009,...,113,121
 *              056,057,058,059,060,061,062,063   to   002,010,...,114,122
 *              064,068,066,067,068,069,070,071 =====> 003,011,...,115,123
 *              072,073,074,075,076,077,078,079        004,012,...,116,124
 *              080,081,082,083,084,085,086,087        005,013,...,117,125
 *              088,089,090,091,092,093,094,095        006,014,...,118,126
 *              096,097,098,099,100,101,102,103        007,015,...,119,127
 *              104,105,106,107,108,109,110,111
 *              112,113,114,115,116,117,118,119
 *              120,121,122,123,124,125,126,127
 * =============================================================================
 */
#define LSX_TRANSPOSE16x8_B(_in0, _in1, _in2, _in3, _in4, _in5, _in6, _in7,  \
                            _in8, _in9, _in10, _in11, _in12, _in13, _in14,   \
                            _in15, _out0, _out1, _out2, _out3, _out4, _out5, \
                            _out6, _out7)                                    \
  {                                                                          \
    __m128i _tmp0, _tmp1, _tmp2, _tmp3, _tmp4, _tmp5, _tmp6, _tmp7;          \
    __m128i _t0, _t1, _t2, _t3, _t4, _t5, _t6, _t7;                          \
    DUP4_ARG2(__lsx_vilvl_b, _in2, _in0, _in3, _in1, _in6, _in4, _in7, _in5, \
              _tmp0, _tmp1, _tmp2, _tmp3);                                   \
    DUP4_ARG2(__lsx_vilvl_b, _in10, _in8, _in11, _in9, _in14, _in12, _in15,  \
              _in13, _tmp4, _tmp5, _tmp6, _tmp7);                            \
    DUP2_ARG2(__lsx_vilvl_b, _tmp1, _tmp0, _tmp3, _tmp2, _t0, _t2);          \
    DUP2_ARG2(__lsx_vilvh_b, _tmp1, _tmp0, _tmp3, _tmp2, _t1, _t3);          \
    DUP2_ARG2(__lsx_vilvl_b, _tmp5, _tmp4, _tmp7, _tmp6, _t4, _t6);          \
    DUP2_ARG2(__lsx_vilvh_b, _tmp5, _tmp4, _tmp7, _tmp6, _t5, _t7);          \
    DUP2_ARG2(__lsx_vilvl_w, _t2, _t0, _t3, _t1, _tmp0, _tmp4);              \
    DUP2_ARG2(__lsx_vilvh_w, _t2, _t0, _t3, _t1, _tmp2, _tmp6);              \
    DUP2_ARG2(__lsx_vilvl_w, _t6, _t4, _t7, _t5, _tmp1, _tmp5);              \
    DUP2_ARG2(__lsx_vilvh_w, _t6, _t4, _t7, _t5, _tmp3, _tmp7);              \
    DUP2_ARG2(__lsx_vilvl_d, _tmp1, _tmp0, _tmp3, _tmp2, _out0, _out2);      \
    DUP2_ARG2(__lsx_vilvh_d, _tmp1, _tmp0, _tmp3, _tmp2, _out1, _out3);      \
    DUP2_ARG2(__lsx_vilvl_d, _tmp5, _tmp4, _tmp7, _tmp6, _out4, _out6);      \
    DUP2_ARG2(__lsx_vilvh_d, _tmp5, _tmp4, _tmp7, _tmp6, _out5, _out7);      \
  }

/*
 * =============================================================================
 * Description : Butterfly of 4 input vectors
 * Arguments   : Inputs  - in0, in1, in2, in3
 *               Outputs - out0, out1, out2, out3
 * Details     : Butterfly operation
 * Example     :
 *               out0 = in0 + in3;
 *               out1 = in1 + in2;
 *               out2 = in1 - in2;
 *               out3 = in0 - in3;
 * =============================================================================
 */
#define LSX_BUTTERFLY_4_B(_in0, _in1, _in2, _in3, _out0, _out1, _out2, _out3) \
  {                                                                           \
    _out0 = __lsx_vadd_b(_in0, _in3);                                         \
    _out1 = __lsx_vadd_b(_in1, _in2);                                         \
    _out2 = __lsx_vsub_b(_in1, _in2);                                         \
    _out3 = __lsx_vsub_b(_in0, _in3);                                         \
  }
#define LSX_BUTTERFLY_4_H(_in0, _in1, _in2, _in3, _out0, _out1, _out2, _out3) \
  {                                                                           \
    _out0 = __lsx_vadd_h(_in0, _in3);                                         \
    _out1 = __lsx_vadd_h(_in1, _in2);                                         \
    _out2 = __lsx_vsub_h(_in1, _in2);                                         \
    _out3 = __lsx_vsub_h(_in0, _in3);                                         \
  }
#define LSX_BUTTERFLY_4_W(_in0, _in1, _in2, _in3, _out0, _out1, _out2, _out3) \
  {                                                                           \
    _out0 = __lsx_vadd_w(_in0, _in3);                                         \
    _out1 = __lsx_vadd_w(_in1, _in2);                                         \
    _out2 = __lsx_vsub_w(_in1, _in2);                                         \
    _out3 = __lsx_vsub_w(_in0, _in3);                                         \
  }
#define LSX_BUTTERFLY_4_D(_in0, _in1, _in2, _in3, _out0, _out1, _out2, _out3) \
  {                                                                           \
    _out0 = __lsx_vadd_d(_in0, _in3);                                         \
    _out1 = __lsx_vadd_d(_in1, _in2);                                         \
    _out2 = __lsx_vsub_d(_in1, _in2);                                         \
    _out3 = __lsx_vsub_d(_in0, _in3);                                         \
  }

/*
 * =============================================================================
 * Description : Butterfly of 8 input vectors
 * Arguments   : Inputs  - _in0, _in1, _in2, _in3, ~
 *               Outputs - _out0, _out1, _out2, _out3, ~
 * Details     : Butterfly operation
 * Example     :
 *              _out0 = _in0 + _in7;
 *              _out1 = _in1 + _in6;
 *              _out2 = _in2 + _in5;
 *              _out3 = _in3 + _in4;
 *              _out4 = _in3 - _in4;
 *              _out5 = _in2 - _in5;
 *              _out6 = _in1 - _in6;
 *              _out7 = _in0 - _in7;
 * =============================================================================
 */
#define LSX_BUTTERFLY_8_B(_in0, _in1, _in2, _in3, _in4, _in5, _in6, _in7,  \
                          _out0, _out1, _out2, _out3, _out4, _out5, _out6, \
                          _out7)                                           \
  {                                                                        \
    _out0 = __lsx_vadd_b(_in0, _in7);                                      \
    _out1 = __lsx_vadd_b(_in1, _in6);                                      \
    _out2 = __lsx_vadd_b(_in2, _in5);                                      \
    _out3 = __lsx_vadd_b(_in3, _in4);                                      \
    _out4 = __lsx_vsub_b(_in3, _in4);                                      \
    _out5 = __lsx_vsub_b(_in2, _in5);                                      \
    _out6 = __lsx_vsub_b(_in1, _in6);                                      \
    _out7 = __lsx_vsub_b(_in0, _in7);                                      \
  }

#define LSX_BUTTERFLY_8_H(_in0, _in1, _in2, _in3, _in4, _in5, _in6, _in7,  \
                          _out0, _out1, _out2, _out3, _out4, _out5, _out6, \
                          _out7)                                           \
  {                                                                        \
    _out0 = __lsx_vadd_h(_in0, _in7);                                      \
    _out1 = __lsx_vadd_h(_in1, _in6);                                      \
    _out2 = __lsx_vadd_h(_in2, _in5);                                      \
    _out3 = __lsx_vadd_h(_in3, _in4);                                      \
    _out4 = __lsx_vsub_h(_in3, _in4);                                      \
    _out5 = __lsx_vsub_h(_in2, _in5);                                      \
    _out6 = __lsx_vsub_h(_in1, _in6);                                      \
    _out7 = __lsx_vsub_h(_in0, _in7);                                      \
  }

#define LSX_BUTTERFLY_8_W(_in0, _in1, _in2, _in3, _in4, _in5, _in6, _in7,  \
                          _out0, _out1, _out2, _out3, _out4, _out5, _out6, \
                          _out7)                                           \
  {                                                                        \
    _out0 = __lsx_vadd_w(_in0, _in7);                                      \
    _out1 = __lsx_vadd_w(_in1, _in6);                                      \
    _out2 = __lsx_vadd_w(_in2, _in5);                                      \
    _out3 = __lsx_vadd_w(_in3, _in4);                                      \
    _out4 = __lsx_vsub_w(_in3, _in4);                                      \
    _out5 = __lsx_vsub_w(_in2, _in5);                                      \
    _out6 = __lsx_vsub_w(_in1, _in6);                                      \
    _out7 = __lsx_vsub_w(_in0, _in7);                                      \
  }

#define LSX_BUTTERFLY_8_D(_in0, _in1, _in2, _in3, _in4, _in5, _in6, _in7,  \
                          _out0, _out1, _out2, _out3, _out4, _out5, _out6, \
                          _out7)                                           \
  {                                                                        \
    _out0 = __lsx_vadd_d(_in0, _in7);                                      \
    _out1 = __lsx_vadd_d(_in1, _in6);                                      \
    _out2 = __lsx_vadd_d(_in2, _in5);                                      \
    _out3 = __lsx_vadd_d(_in3, _in4);                                      \
    _out4 = __lsx_vsub_d(_in3, _in4);                                      \
    _out5 = __lsx_vsub_d(_in2, _in5);                                      \
    _out6 = __lsx_vsub_d(_in1, _in6);                                      \
    _out7 = __lsx_vsub_d(_in0, _in7);                                      \
  }

/*
 * =============================================================================
 * Description : Butterfly of 16 input vectors
 * Arguments   : Inputs  - _in0, _in1, _in2, _in3, ~
 *               Outputs - _out0, _out1, _out2, _out3, ~
 * Details     : Butterfly operation
 * Example     :
 *              _out0 = _in0 + _in15;
 *              _out1 = _in1 + _in14;
 *              _out2 = _in2 + _in13;
 *              _out3 = _in3 + _in12;
 *              _out4 = _in4 + _in11;
 *              _out5 = _in5 + _in10;
 *              _out6 = _in6 + _in9;
 *              _out7 = _in7 + _in8;
 *              _out8 = _in7 - _in8;
 *              _out9 = _in6 - _in9;
 *              _out10 = _in5 - _in10;
 *              _out11 = _in4 - _in11;
 *              _out12 = _in3 - _in12;
 *              _out13 = _in2 - _in13;
 *              _out14 = _in1 - _in14;
 *              _out15 = _in0 - _in15;
 * =============================================================================
 */

#define LSX_BUTTERFLY_16_B(_in0, _in1, _in2, _in3, _in4, _in5, _in6, _in7,     \
                           _in8, _in9, _in10, _in11, _in12, _in13, _in14,      \
                           _in15, _out0, _out1, _out2, _out3, _out4, _out5,    \
                           _out6, _out7, _out8, _out9, _out10, _out11, _out12, \
                           _out13, _out14, _out15)                             \
  {                                                                            \
    _out0 = __lsx_vadd_b(_in0, _in15);                                         \
    _out1 = __lsx_vadd_b(_in1, _in14);                                         \
    _out2 = __lsx_vadd_b(_in2, _in13);                                         \
    _out3 = __lsx_vadd_b(_in3, _in12);                                         \
    _out4 = __lsx_vadd_b(_in4, _in11);                                         \
    _out5 = __lsx_vadd_b(_in5, _in10);                                         \
    _out6 = __lsx_vadd_b(_in6, _in9);                                          \
    _out7 = __lsx_vadd_b(_in7, _in8);                                          \
                                                                               \
    _out8 = __lsx_vsub_b(_in7, _in8);                                          \
    _out9 = __lsx_vsub_b(_in6, _in9);                                          \
    _out10 = __lsx_vsub_b(_in5, _in10);                                        \
    _out11 = __lsx_vsub_b(_in4, _in11);                                        \
    _out12 = __lsx_vsub_b(_in3, _in12);                                        \
    _out13 = __lsx_vsub_b(_in2, _in13);                                        \
    _out14 = __lsx_vsub_b(_in1, _in14);                                        \
    _out15 = __lsx_vsub_b(_in0, _in15);                                        \
  }

#define LSX_BUTTERFLY_16_H(_in0, _in1, _in2, _in3, _in4, _in5, _in6, _in7,     \
                           _in8, _in9, _in10, _in11, _in12, _in13, _in14,      \
                           _in15, _out0, _out1, _out2, _out3, _out4, _out5,    \
                           _out6, _out7, _out8, _out9, _out10, _out11, _out12, \
                           _out13, _out14, _out15)                             \
  {                                                                            \
    _out0 = __lsx_vadd_h(_in0, _in15);                                         \
    _out1 = __lsx_vadd_h(_in1, _in14);                                         \
    _out2 = __lsx_vadd_h(_in2, _in13);                                         \
    _out3 = __lsx_vadd_h(_in3, _in12);                                         \
    _out4 = __lsx_vadd_h(_in4, _in11);                                         \
    _out5 = __lsx_vadd_h(_in5, _in10);                                         \
    _out6 = __lsx_vadd_h(_in6, _in9);                                          \
    _out7 = __lsx_vadd_h(_in7, _in8);                                          \
                                                                               \
    _out8 = __lsx_vsub_h(_in7, _in8);                                          \
    _out9 = __lsx_vsub_h(_in6, _in9);                                          \
    _out10 = __lsx_vsub_h(_in5, _in10);                                        \
    _out11 = __lsx_vsub_h(_in4, _in11);                                        \
    _out12 = __lsx_vsub_h(_in3, _in12);                                        \
    _out13 = __lsx_vsub_h(_in2, _in13);                                        \
    _out14 = __lsx_vsub_h(_in1, _in14);                                        \
    _out15 = __lsx_vsub_h(_in0, _in15);                                        \
  }

#define LSX_BUTTERFLY_16_W(_in0, _in1, _in2, _in3, _in4, _in5, _in6, _in7,     \
                           _in8, _in9, _in10, _in11, _in12, _in13, _in14,      \
                           _in15, _out0, _out1, _out2, _out3, _out4, _out5,    \
                           _out6, _out7, _out8, _out9, _out10, _out11, _out12, \
                           _out13, _out14, _out15)                             \
  {                                                                            \
    _out0 = __lsx_vadd_w(_in0, _in15);                                         \
    _out1 = __lsx_vadd_w(_in1, _in14);                                         \
    _out2 = __lsx_vadd_w(_in2, _in13);                                         \
    _out3 = __lsx_vadd_w(_in3, _in12);                                         \
    _out4 = __lsx_vadd_w(_in4, _in11);                                         \
    _out5 = __lsx_vadd_w(_in5, _in10);                                         \
    _out6 = __lsx_vadd_w(_in6, _in9);                                          \
    _out7 = __lsx_vadd_w(_in7, _in8);                                          \
                                                                               \
    _out8 = __lsx_vsub_w(_in7, _in8);                                          \
    _out9 = __lsx_vsub_w(_in6, _in9);                                          \
    _out10 = __lsx_vsub_w(_in5, _in10);                                        \
    _out11 = __lsx_vsub_w(_in4, _in11);                                        \
    _out12 = __lsx_vsub_w(_in3, _in12);                                        \
    _out13 = __lsx_vsub_w(_in2, _in13);                                        \
    _out14 = __lsx_vsub_w(_in1, _in14);                                        \
    _out15 = __lsx_vsub_w(_in0, _in15);                                        \
  }

#define LSX_BUTTERFLY_16_D(_in0, _in1, _in2, _in3, _in4, _in5, _in6, _in7,     \
                           _in8, _in9, _in10, _in11, _in12, _in13, _in14,      \
                           _in15, _out0, _out1, _out2, _out3, _out4, _out5,    \
                           _out6, _out7, _out8, _out9, _out10, _out11, _out12, \
                           _out13, _out14, _out15)                             \
  {                                                                            \
    _out0 = __lsx_vadd_d(_in0, _in15);                                         \
    _out1 = __lsx_vadd_d(_in1, _in14);                                         \
    _out2 = __lsx_vadd_d(_in2, _in13);                                         \
    _out3 = __lsx_vadd_d(_in3, _in12);                                         \
    _out4 = __lsx_vadd_d(_in4, _in11);                                         \
    _out5 = __lsx_vadd_d(_in5, _in10);                                         \
    _out6 = __lsx_vadd_d(_in6, _in9);                                          \
    _out7 = __lsx_vadd_d(_in7, _in8);                                          \
                                                                               \
    _out8 = __lsx_vsub_d(_in7, _in8);                                          \
    _out9 = __lsx_vsub_d(_in6, _in9);                                          \
    _out10 = __lsx_vsub_d(_in5, _in10);                                        \
    _out11 = __lsx_vsub_d(_in4, _in11);                                        \
    _out12 = __lsx_vsub_d(_in3, _in12);                                        \
    _out13 = __lsx_vsub_d(_in2, _in13);                                        \
    _out14 = __lsx_vsub_d(_in1, _in14);                                        \
    _out15 = __lsx_vsub_d(_in0, _in15);                                        \
  }

#endif  // LSX

#ifdef __loongarch_asx
#include <lasxintrin.h>
/*
 * =============================================================================
 * Description : Dot product of byte vector elements
 * Arguments   : Inputs - in_h, in_l
 *               Output - out
 *               Return Type - signed halfword
 * Details     : Unsigned byte elements from in_h are multiplied with
 *               unsigned byte elements from in_l producing a result
 *               twice the size of input i.e. signed halfword.
 *               Then these multiplied results of adjacent odd-even elements
 *               are added to the out vector
 * Example     : See out = __lasx_xvdp2_w_h(in_h, in_l)
 * =============================================================================
 */
static inline __m256i __lasx_xvdp2_h_bu(__m256i in_h, __m256i in_l) {
  __m256i out;

  out = __lasx_xvmulwev_h_bu(in_h, in_l);
  out = __lasx_xvmaddwod_h_bu(out, in_h, in_l);
  return out;
}

/*
 * =============================================================================
 * Description : Dot product of byte vector elements
 * Arguments   : Inputs - in_h, in_l
 *               Output - out
 *               Return Type - signed halfword
 * Details     : Signed byte elements from in_h are multiplied with
 *               signed byte elements from in_l producing a result
 *               twice the size of input i.e. signed halfword.
 *               Then these multiplication results of adjacent odd-even elements
 *               are added to the out vector
 * Example     : See out = __lasx_xvdp2_w_h(in_h, in_l)
 * =============================================================================
 */
static inline __m256i __lasx_xvdp2_h_b(__m256i in_h, __m256i in_l) {
  __m256i out;

  out = __lasx_xvmulwev_h_b(in_h, in_l);
  out = __lasx_xvmaddwod_h_b(out, in_h, in_l);
  return out;
}

/*
 * =============================================================================
 * Description : Dot product of halfword vector elements
 * Arguments   : Inputs - in_h, in_l
 *               Output - out
 *               Return Type - signed word
 * Details     : Signed halfword elements from in_h are multiplied with
 *               signed halfword elements from in_l producing a result
 *               twice the size of input i.e. signed word.
 *               Then these multiplied results of adjacent odd-even elements
 *               are added to the out vector.
 * Example     : out = __lasx_xvdp2_w_h(in_h, in_l)
 *        in_h : 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
 *        in_l : 8,7,6,5, 4,3,2,1, 8,7,6,5, 4,3,2,1
 *         out : 22,38,38,22, 22,38,38,22
 * =============================================================================
 */
static inline __m256i __lasx_xvdp2_w_h(__m256i in_h, __m256i in_l) {
  __m256i out;

  out = __lasx_xvmulwev_w_h(in_h, in_l);
  out = __lasx_xvmaddwod_w_h(out, in_h, in_l);
  return out;
}

/*
 * =============================================================================
 * Description : Dot product of word vector elements
 * Arguments   : Inputs - in_h, in_l
 *               Output - out
 *               Return Type - signed double
 * Details     : Signed word elements from in_h are multiplied with
 *               signed word elements from in_l producing a result
 *               twice the size of input i.e. signed double-word.
 *               Then these multiplied results of adjacent odd-even elements
 *               are added to the out vector.
 * Example     : See out = __lasx_xvdp2_w_h(in_h, in_l)
 * =============================================================================
 */
static inline __m256i __lasx_xvdp2_d_w(__m256i in_h, __m256i in_l) {
  __m256i out;

  out = __lasx_xvmulwev_d_w(in_h, in_l);
  out = __lasx_xvmaddwod_d_w(out, in_h, in_l);
  return out;
}

/*
 * =============================================================================
 * Description : Dot product of halfword vector elements
 * Arguments   : Inputs - in_h, in_l
 *               Output - out
 *               Return Type - signed word
 * Details     : Unsigned halfword elements from in_h are multiplied with
 *               signed halfword elements from in_l producing a result
 *               twice the size of input i.e. unsigned word.
 *               Multiplication result of adjacent odd-even elements
 *               are added to the out vector
 * Example     : See out = __lasx_xvdp2_w_h(in_h, in_l)
 * =============================================================================
 */
static inline __m256i __lasx_xvdp2_w_hu_h(__m256i in_h, __m256i in_l) {
  __m256i out;

  out = __lasx_xvmulwev_w_hu_h(in_h, in_l);
  out = __lasx_xvmaddwod_w_hu_h(out, in_h, in_l);
  return out;
}

/*
 * =============================================================================
 * Description : Dot product & addition of byte vector elements
 * Arguments   : Inputs - in_h, in_l
 *               Output - out
 *               Return Type - halfword
 * Details     : Signed byte elements from in_h are multiplied with
 *               signed byte elements from in_l producing a result
 *               twice the size of input i.e. signed halfword.
 *               Then these multiplied results of adjacent odd-even elements
 *               are added to the in_c vector.
 * Example     : See out = __lasx_xvdp2add_w_h(in_c, in_h, in_l)
 * =============================================================================
 */
static inline __m256i __lasx_xvdp2add_h_b(__m256i in_c, __m256i in_h,
                                          __m256i in_l) {
  __m256i out;

  out = __lasx_xvmaddwev_h_b(in_c, in_h, in_l);
  out = __lasx_xvmaddwod_h_b(out, in_h, in_l);
  return out;
}

/*
 * =============================================================================
 * Description : Dot product & addition of byte vector elements
 * Arguments   : Inputs - in_h, in_l
 *               Output - out
 *               Return Type - halfword
 * Details     : Unsigned byte elements from in_h are multiplied with
 *               unsigned byte elements from in_l producing a result
 *               twice the size of input i.e. signed halfword.
 *               Then these multiplied results of adjacent odd-even elements
 *               are added to the in_c vector.
 * Example     : See out = __lasx_xvdp2add_w_h(in_c, in_h, in_l)
 * =============================================================================
 */
static inline __m256i __lasx_xvdp2add_h_bu(__m256i in_c, __m256i in_h,
                                           __m256i in_l) {
  __m256i out;

  out = __lasx_xvmaddwev_h_bu(in_c, in_h, in_l);
  out = __lasx_xvmaddwod_h_bu(out, in_h, in_l);
  return out;
}

/*
 * =============================================================================
 * Description : Dot product & addition of byte vector elements
 * Arguments   : Inputs - in_h, in_l
 *               Output - out
 *               Return Type - halfword
 * Details     : Unsigned byte elements from in_h are multiplied with
 *               signed byte elements from in_l producing a result
 *               twice the size of input i.e. signed halfword.
 *               Then these multiplied results of adjacent odd-even elements
 *               are added to the in_c vector.
 * Example     : See out = __lasx_xvdp2add_w_h(in_c, in_h, in_l)
 * =============================================================================
 */
static inline __m256i __lasx_xvdp2add_h_bu_b(__m256i in_c, __m256i in_h,
                                             __m256i in_l) {
  __m256i out;

  out = __lasx_xvmaddwev_h_bu_b(in_c, in_h, in_l);
  out = __lasx_xvmaddwod_h_bu_b(out, in_h, in_l);
  return out;
}

/*
 * =============================================================================
 * Description : Dot product of halfword vector elements
 * Arguments   : Inputs - in_c, in_h, in_l
 *               Output - out
 *               Return Type - per RTYPE
 * Details     : Signed halfword elements from in_h are multiplied with
 *               signed halfword elements from in_l producing a result
 *               twice the size of input i.e. signed word.
 *               Multiplication result of adjacent odd-even elements
 *               are added to the in_c vector.
 * Example     : out = __lasx_xvdp2add_w_h(in_c, in_h, in_l)
 *        in_c : 1,2,3,4, 1,2,3,4
 *        in_h : 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8,
 *        in_l : 8,7,6,5, 4,3,2,1, 8,7,6,5, 4,3,2,1,
 *         out : 23,40,41,26, 23,40,41,26
 * =============================================================================
 */
static inline __m256i __lasx_xvdp2add_w_h(__m256i in_c, __m256i in_h,
                                          __m256i in_l) {
  __m256i out;

  out = __lasx_xvmaddwev_w_h(in_c, in_h, in_l);
  out = __lasx_xvmaddwod_w_h(out, in_h, in_l);
  return out;
}

/*
 * =============================================================================
 * Description : Dot product of halfword vector elements
 * Arguments   : Inputs - in_c, in_h, in_l
 *               Output - out
 *               Return Type - signed word
 * Details     : Unsigned halfword elements from in_h are multiplied with
 *               unsigned halfword elements from in_l producing a result
 *               twice the size of input i.e. signed word.
 *               Multiplication result of adjacent odd-even elements
 *               are added to the in_c vector.
 * Example     : See out = __lasx_xvdp2add_w_h(in_c, in_h, in_l)
 * =============================================================================
 */
static inline __m256i __lasx_xvdp2add_w_hu(__m256i in_c, __m256i in_h,
                                           __m256i in_l) {
  __m256i out;

  out = __lasx_xvmaddwev_w_hu(in_c, in_h, in_l);
  out = __lasx_xvmaddwod_w_hu(out, in_h, in_l);
  return out;
}

/*
 * =============================================================================
 * Description : Dot product of halfword vector elements
 * Arguments   : Inputs - in_c, in_h, in_l
 *               Output - out
 *               Return Type - signed word
 * Details     : Unsigned halfword elements from in_h are multiplied with
 *               signed halfword elements from in_l producing a result
 *               twice the size of input i.e. signed word.
 *               Multiplication result of adjacent odd-even elements
 *               are added to the in_c vector
 * Example     : See out = __lasx_xvdp2add_w_h(in_c, in_h, in_l)
 * =============================================================================
 */
static inline __m256i __lasx_xvdp2add_w_hu_h(__m256i in_c, __m256i in_h,
                                             __m256i in_l) {
  __m256i out;

  out = __lasx_xvmaddwev_w_hu_h(in_c, in_h, in_l);
  out = __lasx_xvmaddwod_w_hu_h(out, in_h, in_l);
  return out;
}

/*
 * =============================================================================
 * Description : Vector Unsigned Dot Product and Subtract
 * Arguments   : Inputs - in_c, in_h, in_l
 *               Output - out
 *               Return Type - signed halfword
 * Details     : Unsigned byte elements from in_h are multiplied with
 *               unsigned byte elements from in_l producing a result
 *               twice the size of input i.e. signed halfword.
 *               Multiplication result of adjacent odd-even elements
 *               are added together and subtracted from double width elements
 *               in_c vector.
 * Example     : See out = __lasx_xvdp2sub_w_h(in_c, in_h, in_l)
 * =============================================================================
 */
static inline __m256i __lasx_xvdp2sub_h_bu(__m256i in_c, __m256i in_h,
                                           __m256i in_l) {
  __m256i out;

  out = __lasx_xvmulwev_h_bu(in_h, in_l);
  out = __lasx_xvmaddwod_h_bu(out, in_h, in_l);
  out = __lasx_xvsub_h(in_c, out);
  return out;
}

/*
 * =============================================================================
 * Description : Vector Signed Dot Product and Subtract
 * Arguments   : Inputs - in_c, in_h, in_l
 *               Output - out
 *               Return Type - signed word
 * Details     : Signed halfword elements from in_h are multiplied with
 *               Signed halfword elements from in_l producing a result
 *               twice the size of input i.e. signed word.
 *               Multiplication result of adjacent odd-even elements
 *               are added together and subtracted from double width elements
 *               in_c vector.
 * Example     : out = __lasx_xvdp2sub_w_h(in_c, in_h, in_l)
 *        in_c : 0,0,0,0, 0,0,0,0
 *        in_h : 3,1,3,0, 0,0,0,1, 0,0,1,1, 0,0,0,1
 *        in_l : 2,1,1,0, 1,0,0,0, 0,0,1,0, 1,0,0,1
 *         out : -7,-3,0,0, 0,-1,0,-1
 * =============================================================================
 */
static inline __m256i __lasx_xvdp2sub_w_h(__m256i in_c, __m256i in_h,
                                          __m256i in_l) {
  __m256i out;

  out = __lasx_xvmulwev_w_h(in_h, in_l);
  out = __lasx_xvmaddwod_w_h(out, in_h, in_l);
  out = __lasx_xvsub_w(in_c, out);
  return out;
}

/*
 * =============================================================================
 * Description : Dot product of halfword vector elements
 * Arguments   : Inputs - in_h, in_l
 *               Output - out
 *               Return Type - signed word
 * Details     : Signed halfword elements from in_h are multiplied with
 *               signed halfword elements from in_l producing a result
 *               four times the size of input i.e. signed doubleword.
 *               Then these multiplication results of four adjacent elements
 *               are added together and stored to the out vector.
 * Example     : out = __lasx_xvdp4_d_h(in_h, in_l)
 *        in_h :  3,1,3,0, 0,0,0,1, 0,0,1,-1, 0,0,0,1
 *        in_l : -2,1,1,0, 1,0,0,0, 0,0,1, 0, 1,0,0,1
 *         out : -2,0,1,1
 * =============================================================================
 */
static inline __m256i __lasx_xvdp4_d_h(__m256i in_h, __m256i in_l) {
  __m256i out;

  out = __lasx_xvmulwev_w_h(in_h, in_l);
  out = __lasx_xvmaddwod_w_h(out, in_h, in_l);
  out = __lasx_xvhaddw_d_w(out, out);
  return out;
}

/*
 * =============================================================================
 * Description : The high half of the vector elements are expanded and
 *               added after being doubled.
 * Arguments   : Inputs - in_h, in_l
 *               Output - out
 * Details     : The in_h vector and the in_l vector are added after the
 *               higher half of the two-fold sign extension (signed byte
 *               to signed halfword) and stored to the out vector.
 * Example     : See out = __lasx_xvaddwh_w_h(in_h, in_l)
 * =============================================================================
 */
static inline __m256i __lasx_xvaddwh_h_b(__m256i in_h, __m256i in_l) {
  __m256i out;

  out = __lasx_xvilvh_b(in_h, in_l);
  out = __lasx_xvhaddw_h_b(out, out);
  return out;
}

/*
 * =============================================================================
 * Description : The high half of the vector elements are expanded and
 *               added after being doubled.
 * Arguments   : Inputs - in_h, in_l
 *               Output - out
 * Details     : The in_h vector and the in_l vector are added after the
 *               higher half of the two-fold sign extension (signed halfword
 *               to signed word) and stored to the out vector.
 * Example     : out = __lasx_xvaddwh_w_h(in_h, in_l)
 *        in_h : 3, 0,3,0, 0,0,0,-1, 0,0,1,-1, 0,0,0,1
 *        in_l : 2,-1,1,2, 1,0,0, 0, 1,0,1, 0, 1,0,0,1
 *         out : 1,0,0,-1, 1,0,0, 2
 * =============================================================================
 */
static inline __m256i __lasx_xvaddwh_w_h(__m256i in_h, __m256i in_l) {
  __m256i out;

  out = __lasx_xvilvh_h(in_h, in_l);
  out = __lasx_xvhaddw_w_h(out, out);
  return out;
}

/*
 * =============================================================================
 * Description : The low half of the vector elements are expanded and
 *               added after being doubled.
 * Arguments   : Inputs - in_h, in_l
 *               Output - out
 * Details     : The in_h vector and the in_l vector are added after the
 *               lower half of the two-fold sign extension (signed byte
 *               to signed halfword) and stored to the out vector.
 * Example     : See out = __lasx_xvaddwl_w_h(in_h, in_l)
 * =============================================================================
 */
static inline __m256i __lasx_xvaddwl_h_b(__m256i in_h, __m256i in_l) {
  __m256i out;

  out = __lasx_xvilvl_b(in_h, in_l);
  out = __lasx_xvhaddw_h_b(out, out);
  return out;
}

/*
 * =============================================================================
 * Description : The low half of the vector elements are expanded and
 *               added after being doubled.
 * Arguments   : Inputs - in_h, in_l
 *               Output - out
 * Details     : The in_h vector and the in_l vector are added after the
 *               lower half of the two-fold sign extension (signed halfword
 *               to signed word) and stored to the out vector.
 * Example     : out = __lasx_xvaddwl_w_h(in_h, in_l)
 *        in_h : 3, 0,3,0, 0,0,0,-1, 0,0,1,-1, 0,0,0,1
 *        in_l : 2,-1,1,2, 1,0,0, 0, 1,0,1, 0, 1,0,0,1
 *         out : 5,-1,4,2, 1,0,2,-1
 * =============================================================================
 */
static inline __m256i __lasx_xvaddwl_w_h(__m256i in_h, __m256i in_l) {
  __m256i out;

  out = __lasx_xvilvl_h(in_h, in_l);
  out = __lasx_xvhaddw_w_h(out, out);
  return out;
}

/*
 * =============================================================================
 * Description : The low half of the vector elements are expanded and
 *               added after being doubled.
 * Arguments   : Inputs - in_h, in_l
 *               Output - out
 * Details     : The out vector and the out vector are added after the
 *               lower half of the two-fold zero extension (unsigned byte
 *               to unsigned halfword) and stored to the out vector.
 * Example     : See out = __lasx_xvaddwl_w_h(in_h, in_l)
 * =============================================================================
 */
static inline __m256i __lasx_xvaddwl_h_bu(__m256i in_h, __m256i in_l) {
  __m256i out;

  out = __lasx_xvilvl_b(in_h, in_l);
  out = __lasx_xvhaddw_hu_bu(out, out);
  return out;
}

/*
 * =============================================================================
 * Description : The low half of the vector elements are expanded and
 *               added after being doubled.
 * Arguments   : Inputs - in_h, in_l
 *               Output - out
 * Details     : The in_l vector after double zero extension (unsigned byte to
 *               signed halfword)，added to the in_h vector.
 * Example     : See out = __lasx_xvaddw_w_w_h(in_h, in_l)
 * =============================================================================
 */
static inline __m256i __lasx_xvaddw_h_h_bu(__m256i in_h, __m256i in_l) {
  __m256i out;

  out = __lasx_xvsllwil_hu_bu(in_l, 0);
  out = __lasx_xvadd_h(in_h, out);
  return out;
}

/*
 * =============================================================================
 * Description : The low half of the vector elements are expanded and
 *               added after being doubled.
 * Arguments   : Inputs - in_h, in_l
 *               Output - out
 * Details     : The in_l vector after double sign extension (signed halfword to
 *               signed word), added to the in_h vector.
 * Example     : out = __lasx_xvaddw_w_w_h(in_h, in_l)
 *        in_h : 0, 1,0,0, -1,0,0,1,
 *        in_l : 2,-1,1,2,  1,0,0,0, 0,0,1,0, 1,0,0,1,
 *         out : 2, 0,1,2, -1,0,1,1,
 * =============================================================================
 */
static inline __m256i __lasx_xvaddw_w_w_h(__m256i in_h, __m256i in_l) {
  __m256i out;

  out = __lasx_xvsllwil_w_h(in_l, 0);
  out = __lasx_xvadd_w(in_h, out);
  return out;
}

/*
 * =============================================================================
 * Description : Multiplication and addition calculation after expansion
 *               of the lower half of the vector.
 * Arguments   : Inputs - in_c, in_h, in_l
 *               Output - out
 * Details     : The in_h vector and the in_l vector are multiplied after
 *               the lower half of the two-fold sign extension (signed halfword
 *               to signed word), and the result is added to the vector in_c,
 *               then stored to the out vector.
 * Example     : out = __lasx_xvmaddwl_w_h(in_c, in_h, in_l)
 *        in_c : 1,2,3,4, 5,6,7,8
 *        in_h : 1,2,3,4, 1,2,3,4, 5,6,7,8, 5,6,7,8
 *        in_l : 200, 300, 400, 500,  2000, 3000, 4000, 5000,
 *              -200,-300,-400,-500, -2000,-3000,-4000,-5000
 *         out : 201, 602,1203,2004, -995, -1794,-2793,-3992
 * =============================================================================
 */
static inline __m256i __lasx_xvmaddwl_w_h(__m256i in_c, __m256i in_h,
                                          __m256i in_l) {
  __m256i tmp0, tmp1, out;

  tmp0 = __lasx_xvsllwil_w_h(in_h, 0);
  tmp1 = __lasx_xvsllwil_w_h(in_l, 0);
  tmp0 = __lasx_xvmul_w(tmp0, tmp1);
  out = __lasx_xvadd_w(tmp0, in_c);
  return out;
}

/*
 * =============================================================================
 * Description : Multiplication and addition calculation after expansion
 *               of the higher half of the vector.
 * Arguments   : Inputs - in_c, in_h, in_l
 *               Output - out
 * Details     : The in_h vector and the in_l vector are multiplied after
 *               the higher half of the two-fold sign extension (signed
 *               halfword to signed word), and the result is added to
 *               the vector in_c, then stored to the out vector.
 * Example     : See out = __lasx_xvmaddwl_w_h(in_c, in_h, in_l)
 * =============================================================================
 */
static inline __m256i __lasx_xvmaddwh_w_h(__m256i in_c, __m256i in_h,
                                          __m256i in_l) {
  __m256i tmp0, tmp1, out;

  tmp0 = __lasx_xvilvh_h(in_h, in_h);
  tmp1 = __lasx_xvilvh_h(in_l, in_l);
  tmp0 = __lasx_xvmulwev_w_h(tmp0, tmp1);
  out = __lasx_xvadd_w(tmp0, in_c);
  return out;
}

/*
 * =============================================================================
 * Description : Multiplication calculation after expansion of the lower
 *               half of the vector.
 * Arguments   : Inputs - in_h, in_l
 *               Output - out
 * Details     : The in_h vector and the in_l vector are multiplied after
 *               the lower half of the two-fold sign extension (signed
 *               halfword to signed word), then stored to the out vector.
 * Example     : out = __lasx_xvmulwl_w_h(in_h, in_l)
 *        in_h : 3,-1,3,0, 0,0,0,-1, 0,0,1,-1, 0,0,0,1
 *        in_l : 2,-1,1,2, 1,0,0, 0, 0,0,1, 0, 1,0,0,1
 *         out : 6,1,3,0, 0,0,1,0
 * =============================================================================
 */
static inline __m256i __lasx_xvmulwl_w_h(__m256i in_h, __m256i in_l) {
  __m256i tmp0, tmp1, out;

  tmp0 = __lasx_xvsllwil_w_h(in_h, 0);
  tmp1 = __lasx_xvsllwil_w_h(in_l, 0);
  out = __lasx_xvmul_w(tmp0, tmp1);
  return out;
}

/*
 * =============================================================================
 * Description : Multiplication calculation after expansion of the lower
 *               half of the vector.
 * Arguments   : Inputs - in_h, in_l
 *               Output - out
 * Details     : The in_h vector and the in_l vector are multiplied after
 *               the lower half of the two-fold sign extension (signed
 *               halfword to signed word), then stored to the out vector.
 * Example     : out = __lasx_xvmulwh_w_h(in_h, in_l)
 *        in_h : 3,-1,3,0, 0,0,0,-1, 0,0,1,-1, 0,0,0,1
 *        in_l : 2,-1,1,2, 1,0,0, 0, 0,0,1, 0, 1,0,0,1
 *         out : 0,0,0,0, 0,0,0,1
 * =============================================================================
 */
static inline __m256i __lasx_xvmulwh_w_h(__m256i in_h, __m256i in_l) {
  __m256i tmp0, tmp1, out;

  tmp0 = __lasx_xvilvh_h(in_h, in_h);
  tmp1 = __lasx_xvilvh_h(in_l, in_l);
  out = __lasx_xvmulwev_w_h(tmp0, tmp1);
  return out;
}

/*
 * =============================================================================
 * Description : The low half of the vector elements are added to the high half
 *               after being doubled, then saturated.
 * Arguments   : Inputs - in_h, in_l
 *               Output - out
 * Details     : The in_h vector adds the in_l vector after the lower half of
 *               the two-fold zero extension (unsigned byte to unsigned
 *               halfword) and then saturated. The results are stored to the out
 *               vector.
 * Example     : out = __lasx_xvsaddw_hu_hu_bu(in_h, in_l)
 *        in_h : 2,65532,1,2, 1,0,0,0, 0,0,1,0, 1,0,0,1
 *        in_l : 3,6,3,0, 0,0,0,1, 0,0,1,1, 0,0,0,1, 3,18,3,0, 0,0,0,1, 0,0,1,1,
 *               0,0,0,1
 *        out  : 5,65535,4,2, 1,0,0,1, 3,18,4,0, 1,0,0,2,
 * =============================================================================
 */
static inline __m256i __lasx_xvsaddw_hu_hu_bu(__m256i in_h, __m256i in_l) {
  __m256i tmp1, out;
  __m256i zero = { 0 };

  tmp1 = __lasx_xvilvl_b(zero, in_l);
  out = __lasx_xvsadd_hu(in_h, tmp1);
  return out;
}

/*
 * =============================================================================
 * Description : Clip all halfword elements of input vector between min & max
 *               out = ((in) < (min)) ? (min) : (((in) > (max)) ? (max) : (in))
 * Arguments   : Inputs  - in    (input vector)
 *                       - min   (min threshold)
 *                       - max   (max threshold)
 *               Outputs - in    (output vector with clipped elements)
 *               Return Type - signed halfword
 * Example     : out = __lasx_xvclip_h(in, min, max)
 *          in : -8,2,280,249, -8,255,280,249, 4,4,4,4, 5,5,5,5
 *         min : 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1
 *         max : 9,9,9,9, 9,9,9,9, 9,9,9,9, 9,9,9,9
 *         out : 1,2,9,9, 1,9,9,9, 4,4,4,4, 5,5,5,5
 * =============================================================================
 */
static inline __m256i __lasx_xvclip_h(__m256i in, __m256i min, __m256i max) {
  __m256i out;

  out = __lasx_xvmax_h(min, in);
  out = __lasx_xvmin_h(max, out);
  return out;
}

/*
 * =============================================================================
 * Description : Clip all signed halfword elements of input vector
 *               between 0 & 255
 * Arguments   : Inputs  - in   (input vector)
 *               Outputs - out  (output vector with clipped elements)
 *               Return Type - signed halfword
 * Example     : See out = __lasx_xvclip255_w(in)
 * =============================================================================
 */
static inline __m256i __lasx_xvclip255_h(__m256i in) {
  __m256i out;

  out = __lasx_xvmaxi_h(in, 0);
  out = __lasx_xvsat_hu(out, 7);
  return out;
}

/*
 * =============================================================================
 * Description : Clip all signed word elements of input vector
 *               between 0 & 255
 * Arguments   : Inputs - in   (input vector)
 *               Output - out  (output vector with clipped elements)
 *               Return Type - signed word
 * Example     : out = __lasx_xvclip255_w(in)
 *          in : -8,255,280,249, -8,255,280,249
 *         out :  0,255,255,249,  0,255,255,249
 * =============================================================================
 */
static inline __m256i __lasx_xvclip255_w(__m256i in) {
  __m256i out;

  out = __lasx_xvmaxi_w(in, 0);
  out = __lasx_xvsat_wu(out, 7);
  return out;
}

/*
 * =============================================================================
 * Description : Indexed halfword element values are replicated to all
 *               elements in output vector. If 'idx < 8' use xvsplati_l_*,
 *               if 'idx >= 8' use xvsplati_h_*.
 * Arguments   : Inputs - in, idx
 *               Output - out
 * Details     : Idx element value from in vector is replicated to all
 *               elements in out vector.
 *               Valid index range for halfword operation is 0-7
 * Example     : out = __lasx_xvsplati_l_h(in, idx)
 *          in : 20,10,11,12, 13,14,15,16, 0,0,2,0, 0,0,0,0
 *         idx : 0x02
 *         out : 11,11,11,11, 11,11,11,11, 11,11,11,11, 11,11,11,11
 * =============================================================================
 */
static inline __m256i __lasx_xvsplati_l_h(__m256i in, int idx) {
  __m256i out;

  out = __lasx_xvpermi_q(in, in, 0x02);
  out = __lasx_xvreplve_h(out, idx);
  return out;
}

/*
 * =============================================================================
 * Description : Indexed halfword element values are replicated to all
 *               elements in output vector. If 'idx < 8' use xvsplati_l_*,
 *               if 'idx >= 8' use xvsplati_h_*.
 * Arguments   : Inputs - in, idx
 *               Output - out
 * Details     : Idx element value from in vector is replicated to all
 *               elements in out vector.
 *               Valid index range for halfword operation is 0-7
 * Example     : out = __lasx_xvsplati_h_h(in, idx)
 *          in : 20,10,11,12, 13,14,15,16, 0,2,0,0, 0,0,0,0
 *         idx : 0x09
 *         out : 2,2,2,2, 2,2,2,2, 2,2,2,2, 2,2,2,2
 * =============================================================================
 */
static inline __m256i __lasx_xvsplati_h_h(__m256i in, int idx) {
  __m256i out;

  out = __lasx_xvpermi_q(in, in, 0x13);
  out = __lasx_xvreplve_h(out, idx);
  return out;
}

/*
 * =============================================================================
 * Description : Transpose 4x4 block with double-word elements in vectors
 * Arguments   : Inputs  - _in0, _in1, _in2, _in3
 *               Outputs - _out0, _out1, _out2, _out3
 * Example     : LASX_TRANSPOSE4x4_D
 *        _in0 : 1,2,3,4
 *        _in1 : 1,2,3,4
 *        _in2 : 1,2,3,4
 *        _in3 : 1,2,3,4
 *
 *       _out0 : 1,1,1,1
 *       _out1 : 2,2,2,2
 *       _out2 : 3,3,3,3
 *       _out3 : 4,4,4,4
 * =============================================================================
 */
#define LASX_TRANSPOSE4x4_D(_in0, _in1, _in2, _in3, _out0, _out1, _out2, \
                            _out3)                                       \
  {                                                                      \
    __m256i _tmp0, _tmp1, _tmp2, _tmp3;                                  \
    _tmp0 = __lasx_xvilvl_d(_in1, _in0);                                 \
    _tmp1 = __lasx_xvilvh_d(_in1, _in0);                                 \
    _tmp2 = __lasx_xvilvl_d(_in3, _in2);                                 \
    _tmp3 = __lasx_xvilvh_d(_in3, _in2);                                 \
    _out0 = __lasx_xvpermi_q(_tmp2, _tmp0, 0x20);                        \
    _out2 = __lasx_xvpermi_q(_tmp2, _tmp0, 0x31);                        \
    _out1 = __lasx_xvpermi_q(_tmp3, _tmp1, 0x20);                        \
    _out3 = __lasx_xvpermi_q(_tmp3, _tmp1, 0x31);                        \
  }

/*
 * =============================================================================
 * Description : Transpose 8x8 block with word elements in vectors
 * Arguments   : Inputs  - _in0, _in1, _in2, _in3, _in4, _in5, _in6, _in7
 *               Outputs - _out0, _out1, _out2, _out3, _out4, _out5, _out6,
 *               _out7
 * Example     : LASX_TRANSPOSE8x8_W
 *        _in0 : 1,2,3,4,5,6,7,8
 *        _in1 : 2,2,3,4,5,6,7,8
 *        _in2 : 3,2,3,4,5,6,7,8
 *        _in3 : 4,2,3,4,5,6,7,8
 *        _in4 : 5,2,3,4,5,6,7,8
 *        _in5 : 6,2,3,4,5,6,7,8
 *        _in6 : 7,2,3,4,5,6,7,8
 *        _in7 : 8,2,3,4,5,6,7,8
 *
 *       _out0 : 1,2,3,4,5,6,7,8
 *       _out1 : 2,2,2,2,2,2,2,2
 *       _out2 : 3,3,3,3,3,3,3,3
 *       _out3 : 4,4,4,4,4,4,4,4
 *       _out4 : 5,5,5,5,5,5,5,5
 *       _out5 : 6,6,6,6,6,6,6,6
 *       _out6 : 7,7,7,7,7,7,7,7
 *       _out7 : 8,8,8,8,8,8,8,8
 * =============================================================================
 */
#define LASX_TRANSPOSE8x8_W(_in0, _in1, _in2, _in3, _in4, _in5, _in6, _in7,  \
                            _out0, _out1, _out2, _out3, _out4, _out5, _out6, \
                            _out7)                                           \
  {                                                                          \
    __m256i _s0_m, _s1_m;                                                    \
    __m256i _tmp0_m, _tmp1_m, _tmp2_m, _tmp3_m;                              \
    __m256i _tmp4_m, _tmp5_m, _tmp6_m, _tmp7_m;                              \
                                                                             \
    _s0_m = __lasx_xvilvl_w(_in2, _in0);                                     \
    _s1_m = __lasx_xvilvl_w(_in3, _in1);                                     \
    _tmp0_m = __lasx_xvilvl_w(_s1_m, _s0_m);                                 \
    _tmp1_m = __lasx_xvilvh_w(_s1_m, _s0_m);                                 \
    _s0_m = __lasx_xvilvh_w(_in2, _in0);                                     \
    _s1_m = __lasx_xvilvh_w(_in3, _in1);                                     \
    _tmp2_m = __lasx_xvilvl_w(_s1_m, _s0_m);                                 \
    _tmp3_m = __lasx_xvilvh_w(_s1_m, _s0_m);                                 \
    _s0_m = __lasx_xvilvl_w(_in6, _in4);                                     \
    _s1_m = __lasx_xvilvl_w(_in7, _in5);                                     \
    _tmp4_m = __lasx_xvilvl_w(_s1_m, _s0_m);                                 \
    _tmp5_m = __lasx_xvilvh_w(_s1_m, _s0_m);                                 \
    _s0_m = __lasx_xvilvh_w(_in6, _in4);                                     \
    _s1_m = __lasx_xvilvh_w(_in7, _in5);                                     \
    _tmp6_m = __lasx_xvilvl_w(_s1_m, _s0_m);                                 \
    _tmp7_m = __lasx_xvilvh_w(_s1_m, _s0_m);                                 \
    _out0 = __lasx_xvpermi_q(_tmp4_m, _tmp0_m, 0x20);                        \
    _out1 = __lasx_xvpermi_q(_tmp5_m, _tmp1_m, 0x20);                        \
    _out2 = __lasx_xvpermi_q(_tmp6_m, _tmp2_m, 0x20);                        \
    _out3 = __lasx_xvpermi_q(_tmp7_m, _tmp3_m, 0x20);                        \
    _out4 = __lasx_xvpermi_q(_tmp4_m, _tmp0_m, 0x31);                        \
    _out5 = __lasx_xvpermi_q(_tmp5_m, _tmp1_m, 0x31);                        \
    _out6 = __lasx_xvpermi_q(_tmp6_m, _tmp2_m, 0x31);                        \
    _out7 = __lasx_xvpermi_q(_tmp7_m, _tmp3_m, 0x31);                        \
  }

/*
 * =============================================================================
 * Description : Transpose input 16x8 byte block
 * Arguments   : Inputs  - _in0, _in1, _in2, _in3, _in4, _in5, _in6, _in7,
 *                         _in8, _in9, _in10, _in11, _in12, _in13, _in14, _in15
 *                         (input 16x8 byte block)
 *               Outputs - _out0, _out1, _out2, _out3, _out4, _out5, _out6,
 *                         _out7 (output 8x16 byte block)
 * Details     : The rows of the matrix become columns, and the columns become
 *               rows.
 * Example     : See LASX_TRANSPOSE16x8_H
 * =============================================================================
 */
#define LASX_TRANSPOSE16x8_B(_in0, _in1, _in2, _in3, _in4, _in5, _in6, _in7,  \
                             _in8, _in9, _in10, _in11, _in12, _in13, _in14,   \
                             _in15, _out0, _out1, _out2, _out3, _out4, _out5, \
                             _out6, _out7)                                    \
  {                                                                           \
    __m256i _tmp0_m, _tmp1_m, _tmp2_m, _tmp3_m;                               \
    __m256i _tmp4_m, _tmp5_m, _tmp6_m, _tmp7_m;                               \
                                                                              \
    _tmp0_m = __lasx_xvilvl_b(_in2, _in0);                                    \
    _tmp1_m = __lasx_xvilvl_b(_in3, _in1);                                    \
    _tmp2_m = __lasx_xvilvl_b(_in6, _in4);                                    \
    _tmp3_m = __lasx_xvilvl_b(_in7, _in5);                                    \
    _tmp4_m = __lasx_xvilvl_b(_in10, _in8);                                   \
    _tmp5_m = __lasx_xvilvl_b(_in11, _in9);                                   \
    _tmp6_m = __lasx_xvilvl_b(_in14, _in12);                                  \
    _tmp7_m = __lasx_xvilvl_b(_in15, _in13);                                  \
    _out0 = __lasx_xvilvl_b(_tmp1_m, _tmp0_m);                                \
    _out1 = __lasx_xvilvh_b(_tmp1_m, _tmp0_m);                                \
    _out2 = __lasx_xvilvl_b(_tmp3_m, _tmp2_m);                                \
    _out3 = __lasx_xvilvh_b(_tmp3_m, _tmp2_m);                                \
    _out4 = __lasx_xvilvl_b(_tmp5_m, _tmp4_m);                                \
    _out5 = __lasx_xvilvh_b(_tmp5_m, _tmp4_m);                                \
    _out6 = __lasx_xvilvl_b(_tmp7_m, _tmp6_m);                                \
    _out7 = __lasx_xvilvh_b(_tmp7_m, _tmp6_m);                                \
    _tmp0_m = __lasx_xvilvl_w(_out2, _out0);                                  \
    _tmp2_m = __lasx_xvilvh_w(_out2, _out0);                                  \
    _tmp4_m = __lasx_xvilvl_w(_out3, _out1);                                  \
    _tmp6_m = __lasx_xvilvh_w(_out3, _out1);                                  \
    _tmp1_m = __lasx_xvilvl_w(_out6, _out4);                                  \
    _tmp3_m = __lasx_xvilvh_w(_out6, _out4);                                  \
    _tmp5_m = __lasx_xvilvl_w(_out7, _out5);                                  \
    _tmp7_m = __lasx_xvilvh_w(_out7, _out5);                                  \
    _out0 = __lasx_xvilvl_d(_tmp1_m, _tmp0_m);                                \
    _out1 = __lasx_xvilvh_d(_tmp1_m, _tmp0_m);                                \
    _out2 = __lasx_xvilvl_d(_tmp3_m, _tmp2_m);                                \
    _out3 = __lasx_xvilvh_d(_tmp3_m, _tmp2_m);                                \
    _out4 = __lasx_xvilvl_d(_tmp5_m, _tmp4_m);                                \
    _out5 = __lasx_xvilvh_d(_tmp5_m, _tmp4_m);                                \
    _out6 = __lasx_xvilvl_d(_tmp7_m, _tmp6_m);                                \
    _out7 = __lasx_xvilvh_d(_tmp7_m, _tmp6_m);                                \
  }

/*
 * =============================================================================
 * Description : Transpose input 16x8 byte block
 * Arguments   : Inputs  - _in0, _in1, _in2, _in3, _in4, _in5, _in6, _in7,
 *                         _in8, _in9, _in10, _in11, _in12, _in13, _in14, _in15
 *                         (input 16x8 byte block)
 *               Outputs - _out0, _out1, _out2, _out3, _out4, _out5, _out6,
 *                         _out7 (output 8x16 byte block)
 * Details     : The rows of the matrix become columns, and the columns become
 *               rows.
 * Example     : LASX_TRANSPOSE16x8_H
 *        _in0 : 1,2,3,4,5,6,7,8,0,0,0,0,0,0,0,0
 *        _in1 : 2,2,3,4,5,6,7,8,0,0,0,0,0,0,0,0
 *        _in2 : 3,2,3,4,5,6,7,8,0,0,0,0,0,0,0,0
 *        _in3 : 4,2,3,4,5,6,7,8,0,0,0,0,0,0,0,0
 *        _in4 : 5,2,3,4,5,6,7,8,0,0,0,0,0,0,0,0
 *        _in5 : 6,2,3,4,5,6,7,8,0,0,0,0,0,0,0,0
 *        _in6 : 7,2,3,4,5,6,7,8,0,0,0,0,0,0,0,0
 *        _in7 : 8,2,3,4,5,6,7,8,0,0,0,0,0,0,0,0
 *        _in8 : 9,2,3,4,5,6,7,8,0,0,0,0,0,0,0,0
 *        _in9 : 1,2,3,4,5,6,7,8,0,0,0,0,0,0,0,0
 *       _in10 : 0,2,3,4,5,6,7,8,0,0,0,0,0,0,0,0
 *       _in11 : 2,2,3,4,5,6,7,8,0,0,0,0,0,0,0,0
 *       _in12 : 3,2,3,4,5,6,7,8,0,0,0,0,0,0,0,0
 *       _in13 : 7,2,3,4,5,6,7,8,0,0,0,0,0,0,0,0
 *       _in14 : 5,2,3,4,5,6,7,8,0,0,0,0,0,0,0,0
 *       _in15 : 6,2,3,4,5,6,7,8,0,0,0,0,0,0,0,0
 *
 *       _out0 : 1,2,3,4,5,6,7,8,9,1,0,2,3,7,5,6
 *       _out1 : 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2
 *       _out2 : 3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3
 *       _out3 : 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4
 *       _out4 : 5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5
 *       _out5 : 6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6
 *       _out6 : 7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7
 *       _out7 : 8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8
 * =============================================================================
 */
#define LASX_TRANSPOSE16x8_H(_in0, _in1, _in2, _in3, _in4, _in5, _in6, _in7,  \
                             _in8, _in9, _in10, _in11, _in12, _in13, _in14,   \
                             _in15, _out0, _out1, _out2, _out3, _out4, _out5, \
                             _out6, _out7)                                    \
  {                                                                           \
    __m256i _tmp0_m, _tmp1_m, _tmp2_m, _tmp3_m;                               \
    __m256i _tmp4_m, _tmp5_m, _tmp6_m, _tmp7_m;                               \
    __m256i _t0, _t1, _t2, _t3, _t4, _t5, _t6, _t7;                           \
                                                                              \
    _tmp0_m = __lasx_xvilvl_h(_in2, _in0);                                    \
    _tmp1_m = __lasx_xvilvl_h(_in3, _in1);                                    \
    _tmp2_m = __lasx_xvilvl_h(_in6, _in4);                                    \
    _tmp3_m = __lasx_xvilvl_h(_in7, _in5);                                    \
    _tmp4_m = __lasx_xvilvl_h(_in10, _in8);                                   \
    _tmp5_m = __lasx_xvilvl_h(_in11, _in9);                                   \
    _tmp6_m = __lasx_xvilvl_h(_in14, _in12);                                  \
    _tmp7_m = __lasx_xvilvl_h(_in15, _in13);                                  \
    _t0 = __lasx_xvilvl_h(_tmp1_m, _tmp0_m);                                  \
    _t1 = __lasx_xvilvh_h(_tmp1_m, _tmp0_m);                                  \
    _t2 = __lasx_xvilvl_h(_tmp3_m, _tmp2_m);                                  \
    _t3 = __lasx_xvilvh_h(_tmp3_m, _tmp2_m);                                  \
    _t4 = __lasx_xvilvl_h(_tmp5_m, _tmp4_m);                                  \
    _t5 = __lasx_xvilvh_h(_tmp5_m, _tmp4_m);                                  \
    _t6 = __lasx_xvilvl_h(_tmp7_m, _tmp6_m);                                  \
    _t7 = __lasx_xvilvh_h(_tmp7_m, _tmp6_m);                                  \
    _tmp0_m = __lasx_xvilvl_d(_t2, _t0);                                      \
    _tmp2_m = __lasx_xvilvh_d(_t2, _t0);                                      \
    _tmp4_m = __lasx_xvilvl_d(_t3, _t1);                                      \
    _tmp6_m = __lasx_xvilvh_d(_t3, _t1);                                      \
    _tmp1_m = __lasx_xvilvl_d(_t6, _t4);                                      \
    _tmp3_m = __lasx_xvilvh_d(_t6, _t4);                                      \
    _tmp5_m = __lasx_xvilvl_d(_t7, _t5);                                      \
    _tmp7_m = __lasx_xvilvh_d(_t7, _t5);                                      \
    _out0 = __lasx_xvpermi_q(_tmp1_m, _tmp0_m, 0x20);                         \
    _out1 = __lasx_xvpermi_q(_tmp3_m, _tmp2_m, 0x20);                         \
    _out2 = __lasx_xvpermi_q(_tmp5_m, _tmp4_m, 0x20);                         \
    _out3 = __lasx_xvpermi_q(_tmp7_m, _tmp6_m, 0x20);                         \
                                                                              \
    _tmp0_m = __lasx_xvilvh_h(_in2, _in0);                                    \
    _tmp1_m = __lasx_xvilvh_h(_in3, _in1);                                    \
    _tmp2_m = __lasx_xvilvh_h(_in6, _in4);                                    \
    _tmp3_m = __lasx_xvilvh_h(_in7, _in5);                                    \
    _tmp4_m = __lasx_xvilvh_h(_in10, _in8);                                   \
    _tmp5_m = __lasx_xvilvh_h(_in11, _in9);                                   \
    _tmp6_m = __lasx_xvilvh_h(_in14, _in12);                                  \
    _tmp7_m = __lasx_xvilvh_h(_in15, _in13);                                  \
    _t0 = __lasx_xvilvl_h(_tmp1_m, _tmp0_m);                                  \
    _t1 = __lasx_xvilvh_h(_tmp1_m, _tmp0_m);                                  \
    _t2 = __lasx_xvilvl_h(_tmp3_m, _tmp2_m);                                  \
    _t3 = __lasx_xvilvh_h(_tmp3_m, _tmp2_m);                                  \
    _t4 = __lasx_xvilvl_h(_tmp5_m, _tmp4_m);                                  \
    _t5 = __lasx_xvilvh_h(_tmp5_m, _tmp4_m);                                  \
    _t6 = __lasx_xvilvl_h(_tmp7_m, _tmp6_m);                                  \
    _t7 = __lasx_xvilvh_h(_tmp7_m, _tmp6_m);                                  \
    _tmp0_m = __lasx_xvilvl_d(_t2, _t0);                                      \
    _tmp2_m = __lasx_xvilvh_d(_t2, _t0);                                      \
    _tmp4_m = __lasx_xvilvl_d(_t3, _t1);                                      \
    _tmp6_m = __lasx_xvilvh_d(_t3, _t1);                                      \
    _tmp1_m = __lasx_xvilvl_d(_t6, _t4);                                      \
    _tmp3_m = __lasx_xvilvh_d(_t6, _t4);                                      \
    _tmp5_m = __lasx_xvilvl_d(_t7, _t5);                                      \
    _tmp7_m = __lasx_xvilvh_d(_t7, _t5);                                      \
    _out4 = __lasx_xvpermi_q(_tmp1_m, _tmp0_m, 0x20);                         \
    _out5 = __lasx_xvpermi_q(_tmp3_m, _tmp2_m, 0x20);                         \
    _out6 = __lasx_xvpermi_q(_tmp5_m, _tmp4_m, 0x20);                         \
    _out7 = __lasx_xvpermi_q(_tmp7_m, _tmp6_m, 0x20);                         \
  }

/*
 * =============================================================================
 * Description : Transpose 4x4 block with halfword elements in vectors
 * Arguments   : Inputs  - _in0, _in1, _in2, _in3
 *               Outputs - _out0, _out1, _out2, _out3
 *               Return Type - signed halfword
 * Details     : The rows of the matrix become columns, and the columns become
 *               rows.
 * Example     : See LASX_TRANSPOSE8x8_H
 * =============================================================================
 */
#define LASX_TRANSPOSE4x4_H(_in0, _in1, _in2, _in3, _out0, _out1, _out2, \
                            _out3)                                       \
  {                                                                      \
    __m256i _s0_m, _s1_m;                                                \
                                                                         \
    _s0_m = __lasx_xvilvl_h(_in1, _in0);                                 \
    _s1_m = __lasx_xvilvl_h(_in3, _in2);                                 \
    _out0 = __lasx_xvilvl_w(_s1_m, _s0_m);                               \
    _out2 = __lasx_xvilvh_w(_s1_m, _s0_m);                               \
    _out1 = __lasx_xvilvh_d(_out0, _out0);                               \
    _out3 = __lasx_xvilvh_d(_out2, _out2);                               \
  }

/*
 * =============================================================================
 * Description : Transpose input 8x8 byte block
 * Arguments   : Inputs  - _in0, _in1, _in2, _in3, _in4, _in5, _in6, _in7
 *                         (input 8x8 byte block)
 *               Outputs - _out0, _out1, _out2, _out3, _out4, _out5, _out6,
 *                         _out7 (output 8x8 byte block)
 * Example     : See LASX_TRANSPOSE8x8_H
 * =============================================================================
 */
#define LASX_TRANSPOSE8x8_B(_in0, _in1, _in2, _in3, _in4, _in5, _in6, _in7,  \
                            _out0, _out1, _out2, _out3, _out4, _out5, _out6, \
                            _out7)                                           \
  {                                                                          \
    __m256i _tmp0_m, _tmp1_m, _tmp2_m, _tmp3_m;                              \
    __m256i _tmp4_m, _tmp5_m, _tmp6_m, _tmp7_m;                              \
    _tmp0_m = __lasx_xvilvl_b(_in2, _in0);                                   \
    _tmp1_m = __lasx_xvilvl_b(_in3, _in1);                                   \
    _tmp2_m = __lasx_xvilvl_b(_in6, _in4);                                   \
    _tmp3_m = __lasx_xvilvl_b(_in7, _in5);                                   \
    _tmp4_m = __lasx_xvilvl_b(_tmp1_m, _tmp0_m);                             \
    _tmp5_m = __lasx_xvilvh_b(_tmp1_m, _tmp0_m);                             \
    _tmp6_m = __lasx_xvilvl_b(_tmp3_m, _tmp2_m);                             \
    _tmp7_m = __lasx_xvilvh_b(_tmp3_m, _tmp2_m);                             \
    _out0 = __lasx_xvilvl_w(_tmp6_m, _tmp4_m);                               \
    _out2 = __lasx_xvilvh_w(_tmp6_m, _tmp4_m);                               \
    _out4 = __lasx_xvilvl_w(_tmp7_m, _tmp5_m);                               \
    _out6 = __lasx_xvilvh_w(_tmp7_m, _tmp5_m);                               \
    _out1 = __lasx_xvbsrl_v(_out0, 8);                                       \
    _out3 = __lasx_xvbsrl_v(_out2, 8);                                       \
    _out5 = __lasx_xvbsrl_v(_out4, 8);                                       \
    _out7 = __lasx_xvbsrl_v(_out6, 8);                                       \
  }

/*
 * =============================================================================
 * Description : Transpose 8x8 block with halfword elements in vectors.
 * Arguments   : Inputs  - _in0, _in1, ~
 *               Outputs - _out0, _out1, ~
 * Details     : The rows of the matrix become columns, and the columns become
 *               rows.
 * Example     : LASX_TRANSPOSE8x8_H
 *        _in0 : 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
 *        _in1 : 8,2,3,4, 5,6,7,8, 8,2,3,4, 5,6,7,8
 *        _in2 : 8,2,3,4, 5,6,7,8, 8,2,3,4, 5,6,7,8
 *        _in3 : 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
 *        _in4 : 9,2,3,4, 5,6,7,8, 9,2,3,4, 5,6,7,8
 *        _in5 : 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
 *        _in6 : 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
 *        _in7 : 9,2,3,4, 5,6,7,8, 9,2,3,4, 5,6,7,8
 *
 *       _out0 : 1,8,8,1, 9,1,1,9, 1,8,8,1, 9,1,1,9
 *       _out1 : 2,2,2,2, 2,2,2,2, 2,2,2,2, 2,2,2,2
 *       _out2 : 3,3,3,3, 3,3,3,3, 3,3,3,3, 3,3,3,3
 *       _out3 : 4,4,4,4, 4,4,4,4, 4,4,4,4, 4,4,4,4
 *       _out4 : 5,5,5,5, 5,5,5,5, 5,5,5,5, 5,5,5,5
 *       _out5 : 6,6,6,6, 6,6,6,6, 6,6,6,6, 6,6,6,6
 *       _out6 : 7,7,7,7, 7,7,7,7, 7,7,7,7, 7,7,7,7
 *       _out7 : 8,8,8,8, 8,8,8,8, 8,8,8,8, 8,8,8,8
 * =============================================================================
 */
#define LASX_TRANSPOSE8x8_H(_in0, _in1, _in2, _in3, _in4, _in5, _in6, _in7,  \
                            _out0, _out1, _out2, _out3, _out4, _out5, _out6, \
                            _out7)                                           \
  {                                                                          \
    __m256i _s0_m, _s1_m;                                                    \
    __m256i _tmp0_m, _tmp1_m, _tmp2_m, _tmp3_m;                              \
    __m256i _tmp4_m, _tmp5_m, _tmp6_m, _tmp7_m;                              \
                                                                             \
    _s0_m = __lasx_xvilvl_h(_in6, _in4);                                     \
    _s1_m = __lasx_xvilvl_h(_in7, _in5);                                     \
    _tmp0_m = __lasx_xvilvl_h(_s1_m, _s0_m);                                 \
    _tmp1_m = __lasx_xvilvh_h(_s1_m, _s0_m);                                 \
    _s0_m = __lasx_xvilvh_h(_in6, _in4);                                     \
    _s1_m = __lasx_xvilvh_h(_in7, _in5);                                     \
    _tmp2_m = __lasx_xvilvl_h(_s1_m, _s0_m);                                 \
    _tmp3_m = __lasx_xvilvh_h(_s1_m, _s0_m);                                 \
                                                                             \
    _s0_m = __lasx_xvilvl_h(_in2, _in0);                                     \
    _s1_m = __lasx_xvilvl_h(_in3, _in1);                                     \
    _tmp4_m = __lasx_xvilvl_h(_s1_m, _s0_m);                                 \
    _tmp5_m = __lasx_xvilvh_h(_s1_m, _s0_m);                                 \
    _s0_m = __lasx_xvilvh_h(_in2, _in0);                                     \
    _s1_m = __lasx_xvilvh_h(_in3, _in1);                                     \
    _tmp6_m = __lasx_xvilvl_h(_s1_m, _s0_m);                                 \
    _tmp7_m = __lasx_xvilvh_h(_s1_m, _s0_m);                                 \
                                                                             \
    _out0 = __lasx_xvpickev_d(_tmp0_m, _tmp4_m);                             \
    _out2 = __lasx_xvpickev_d(_tmp1_m, _tmp5_m);                             \
    _out4 = __lasx_xvpickev_d(_tmp2_m, _tmp6_m);                             \
    _out6 = __lasx_xvpickev_d(_tmp3_m, _tmp7_m);                             \
    _out1 = __lasx_xvpickod_d(_tmp0_m, _tmp4_m);                             \
    _out3 = __lasx_xvpickod_d(_tmp1_m, _tmp5_m);                             \
    _out5 = __lasx_xvpickod_d(_tmp2_m, _tmp6_m);                             \
    _out7 = __lasx_xvpickod_d(_tmp3_m, _tmp7_m);                             \
  }

/*
 * =============================================================================
 * Description : Butterfly of 4 input vectors
 * Arguments   : Inputs  - _in0, _in1, _in2, _in3
 *               Outputs - _out0, _out1, _out2, _out3
 * Details     : Butterfly operation
 * Example     : LASX_BUTTERFLY_4
 *               _out0 = _in0 + _in3;
 *               _out1 = _in1 + _in2;
 *               _out2 = _in1 - _in2;
 *               _out3 = _in0 - _in3;
 * =============================================================================
 */
#define LASX_BUTTERFLY_4_B(_in0, _in1, _in2, _in3, _out0, _out1, _out2, _out3) \
  {                                                                            \
    _out0 = __lasx_xvadd_b(_in0, _in3);                                        \
    _out1 = __lasx_xvadd_b(_in1, _in2);                                        \
    _out2 = __lasx_xvsub_b(_in1, _in2);                                        \
    _out3 = __lasx_xvsub_b(_in0, _in3);                                        \
  }
#define LASX_BUTTERFLY_4_H(_in0, _in1, _in2, _in3, _out0, _out1, _out2, _out3) \
  {                                                                            \
    _out0 = __lasx_xvadd_h(_in0, _in3);                                        \
    _out1 = __lasx_xvadd_h(_in1, _in2);                                        \
    _out2 = __lasx_xvsub_h(_in1, _in2);                                        \
    _out3 = __lasx_xvsub_h(_in0, _in3);                                        \
  }
#define LASX_BUTTERFLY_4_W(_in0, _in1, _in2, _in3, _out0, _out1, _out2, _out3) \
  {                                                                            \
    _out0 = __lasx_xvadd_w(_in0, _in3);                                        \
    _out1 = __lasx_xvadd_w(_in1, _in2);                                        \
    _out2 = __lasx_xvsub_w(_in1, _in2);                                        \
    _out3 = __lasx_xvsub_w(_in0, _in3);                                        \
  }
#define LASX_BUTTERFLY_4_D(_in0, _in1, _in2, _in3, _out0, _out1, _out2, _out3) \
  {                                                                            \
    _out0 = __lasx_xvadd_d(_in0, _in3);                                        \
    _out1 = __lasx_xvadd_d(_in1, _in2);                                        \
    _out2 = __lasx_xvsub_d(_in1, _in2);                                        \
    _out3 = __lasx_xvsub_d(_in0, _in3);                                        \
  }

/*
 * =============================================================================
 * Description : Butterfly of 8 input vectors
 * Arguments   : Inputs  - _in0, _in1, _in2, _in3, ~
 *               Outputs - _out0, _out1, _out2, _out3, ~
 * Details     : Butterfly operation
 * Example     : LASX_BUTTERFLY_8
 *               _out0 = _in0 + _in7;
 *               _out1 = _in1 + _in6;
 *               _out2 = _in2 + _in5;
 *               _out3 = _in3 + _in4;
 *               _out4 = _in3 - _in4;
 *               _out5 = _in2 - _in5;
 *               _out6 = _in1 - _in6;
 *               _out7 = _in0 - _in7;
 * =============================================================================
 */
#define LASX_BUTTERFLY_8_B(_in0, _in1, _in2, _in3, _in4, _in5, _in6, _in7,  \
                           _out0, _out1, _out2, _out3, _out4, _out5, _out6, \
                           _out7)                                           \
  {                                                                         \
    _out0 = __lasx_xvadd_b(_in0, _in7);                                     \
    _out1 = __lasx_xvadd_b(_in1, _in6);                                     \
    _out2 = __lasx_xvadd_b(_in2, _in5);                                     \
    _out3 = __lasx_xvadd_b(_in3, _in4);                                     \
    _out4 = __lasx_xvsub_b(_in3, _in4);                                     \
    _out5 = __lasx_xvsub_b(_in2, _in5);                                     \
    _out6 = __lasx_xvsub_b(_in1, _in6);                                     \
    _out7 = __lasx_xvsub_b(_in0, _in7);                                     \
  }

#define LASX_BUTTERFLY_8_H(_in0, _in1, _in2, _in3, _in4, _in5, _in6, _in7,  \
                           _out0, _out1, _out2, _out3, _out4, _out5, _out6, \
                           _out7)                                           \
  {                                                                         \
    _out0 = __lasx_xvadd_h(_in0, _in7);                                     \
    _out1 = __lasx_xvadd_h(_in1, _in6);                                     \
    _out2 = __lasx_xvadd_h(_in2, _in5);                                     \
    _out3 = __lasx_xvadd_h(_in3, _in4);                                     \
    _out4 = __lasx_xvsub_h(_in3, _in4);                                     \
    _out5 = __lasx_xvsub_h(_in2, _in5);                                     \
    _out6 = __lasx_xvsub_h(_in1, _in6);                                     \
    _out7 = __lasx_xvsub_h(_in0, _in7);                                     \
  }

#define LASX_BUTTERFLY_8_W(_in0, _in1, _in2, _in3, _in4, _in5, _in6, _in7,  \
                           _out0, _out1, _out2, _out3, _out4, _out5, _out6, \
                           _out7)                                           \
  {                                                                         \
    _out0 = __lasx_xvadd_w(_in0, _in7);                                     \
    _out1 = __lasx_xvadd_w(_in1, _in6);                                     \
    _out2 = __lasx_xvadd_w(_in2, _in5);                                     \
    _out3 = __lasx_xvadd_w(_in3, _in4);                                     \
    _out4 = __lasx_xvsub_w(_in3, _in4);                                     \
    _out5 = __lasx_xvsub_w(_in2, _in5);                                     \
    _out6 = __lasx_xvsub_w(_in1, _in6);                                     \
    _out7 = __lasx_xvsub_w(_in0, _in7);                                     \
  }

#define LASX_BUTTERFLY_8_D(_in0, _in1, _in2, _in3, _in4, _in5, _in6, _in7,  \
                           _out0, _out1, _out2, _out3, _out4, _out5, _out6, \
                           _out7)                                           \
  {                                                                         \
    _out0 = __lasx_xvadd_d(_in0, _in7);                                     \
    _out1 = __lasx_xvadd_d(_in1, _in6);                                     \
    _out2 = __lasx_xvadd_d(_in2, _in5);                                     \
    _out3 = __lasx_xvadd_d(_in3, _in4);                                     \
    _out4 = __lasx_xvsub_d(_in3, _in4);                                     \
    _out5 = __lasx_xvsub_d(_in2, _in5);                                     \
    _out6 = __lasx_xvsub_d(_in1, _in6);                                     \
    _out7 = __lasx_xvsub_d(_in0, _in7);                                     \
  }

#endif  // LASX

/*
 * =============================================================================
 * Description : Print out elements in vector.
 * Arguments   : Inputs  - RTYPE, _element_num, _in0, _enter
 *               Outputs -
 * Details     : Print out '_element_num' elements in 'RTYPE' vector '_in0', if
 *               '_enter' is TRUE, prefix "\nVP:" will be added first.
 * Example     : VECT_PRINT(v4i32,4,in0,1); // in0: 1,2,3,4
 *               VP:1,2,3,4,
 * =============================================================================
 */
#define VECT_PRINT(RTYPE, element_num, in0, enter)                 \
  {                                                                \
    RTYPE _tmp0 = (RTYPE)in0;                                      \
    int _i = 0;                                                    \
    if (enter) printf("\nVP:");                                    \
    for (_i = 0; _i < element_num; _i++) printf("%d,", _tmp0[_i]); \
  }

#endif /* LOONGSON_INTRINSICS_H */
#endif /* VPX_VPX_UTIL_LOONGSON_INTRINSICS_H_ */
