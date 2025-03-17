/*
 * AltiVec optimizations for libjpeg-turbo
 *
 * Copyright (C) 2014, D. R. Commander.  All Rights Reserved.
 *
 * This software is provided 'as-is', without any express or implied
 * warranty.  In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

/* FAST INTEGER FORWARD DCT
 *
 * This is similar to the SSE2 implementation, except that we left-shift the
 * constants by 1 less bit (the -1 in CONST_SHIFT.)  This is because
 * vec_madds(arg1, arg2, arg3) generates the 16-bit saturated sum of:
 *   the elements in arg3 + the most significant 17 bits of
 *     (the elements in arg1 * the elements in arg2).
 */

#include "jsimd_altivec.h"


#define F_0_382  98   /* FIX(0.382683433) */
#define F_0_541  139  /* FIX(0.541196100) */
#define F_0_707  181  /* FIX(0.707106781) */
#define F_1_306  334  /* FIX(1.306562965) */

#define CONST_BITS  8
#define PRE_MULTIPLY_SCALE_BITS  2
#define CONST_SHIFT  (16 - PRE_MULTIPLY_SCALE_BITS - CONST_BITS - 1)


#define DO_FDCT() { \
  /* Even part */ \
  \
  tmp10 = vec_add(tmp0, tmp3); \
  tmp13 = vec_sub(tmp0, tmp3); \
  tmp11 = vec_add(tmp1, tmp2); \
  tmp12 = vec_sub(tmp1, tmp2); \
  \
  out0  = vec_add(tmp10, tmp11); \
  out4  = vec_sub(tmp10, tmp11); \
  \
  z1 = vec_add(tmp12, tmp13); \
  z1 = vec_sl(z1, pre_multiply_scale_bits); \
  z1 = vec_madds(z1, pw_0707, pw_zero); \
  \
  out2 = vec_add(tmp13, z1); \
  out6 = vec_sub(tmp13, z1); \
  \
  /* Odd part */ \
  \
  tmp10 = vec_add(tmp4, tmp5); \
  tmp11 = vec_add(tmp5, tmp6); \
  tmp12 = vec_add(tmp6, tmp7); \
  \
  tmp10 = vec_sl(tmp10, pre_multiply_scale_bits); \
  tmp12 = vec_sl(tmp12, pre_multiply_scale_bits); \
  z5 = vec_sub(tmp10, tmp12); \
  z5 = vec_madds(z5, pw_0382, pw_zero); \
  \
  z2 = vec_madds(tmp10, pw_0541, z5); \
  z4 = vec_madds(tmp12, pw_1306, z5); \
  \
  tmp11 = vec_sl(tmp11, pre_multiply_scale_bits); \
  z3 = vec_madds(tmp11, pw_0707, pw_zero); \
  \
  z11 = vec_add(tmp7, z3); \
  z13 = vec_sub(tmp7, z3); \
  \
  out5 = vec_add(z13, z2); \
  out3 = vec_sub(z13, z2); \
  out1 = vec_add(z11, z4); \
  out7 = vec_sub(z11, z4); \
}


void jsimd_fdct_ifast_altivec(DCTELEM *data)
{
  __vector short row0, row1, row2, row3, row4, row5, row6, row7,
    col0, col1, col2, col3, col4, col5, col6, col7,
    tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp10, tmp11, tmp12, tmp13,
    z1, z2, z3, z4, z5, z11, z13,
    out0, out1, out2, out3, out4, out5, out6, out7;

  /* Constants */
  __vector short pw_zero = { __8X(0) },
    pw_0382 = { __8X(F_0_382 << CONST_SHIFT) },
    pw_0541 = { __8X(F_0_541 << CONST_SHIFT) },
    pw_0707 = { __8X(F_0_707 << CONST_SHIFT) },
    pw_1306 = { __8X(F_1_306 << CONST_SHIFT) };
  __vector unsigned short
    pre_multiply_scale_bits = { __8X(PRE_MULTIPLY_SCALE_BITS) };

  /* Pass 1: process rows */

  row0 = vec_ld(0, data);
  row1 = vec_ld(16, data);
  row2 = vec_ld(32, data);
  row3 = vec_ld(48, data);
  row4 = vec_ld(64, data);
  row5 = vec_ld(80, data);
  row6 = vec_ld(96, data);
  row7 = vec_ld(112, data);

  TRANSPOSE(row, col);

  tmp0 = vec_add(col0, col7);
  tmp7 = vec_sub(col0, col7);
  tmp1 = vec_add(col1, col6);
  tmp6 = vec_sub(col1, col6);
  tmp2 = vec_add(col2, col5);
  tmp5 = vec_sub(col2, col5);
  tmp3 = vec_add(col3, col4);
  tmp4 = vec_sub(col3, col4);

  DO_FDCT();

  /* Pass 2: process columns */

  TRANSPOSE(out, row);

  tmp0 = vec_add(row0, row7);
  tmp7 = vec_sub(row0, row7);
  tmp1 = vec_add(row1, row6);
  tmp6 = vec_sub(row1, row6);
  tmp2 = vec_add(row2, row5);
  tmp5 = vec_sub(row2, row5);
  tmp3 = vec_add(row3, row4);
  tmp4 = vec_sub(row3, row4);

  DO_FDCT();

  vec_st(out0, 0, data);
  vec_st(out1, 16, data);
  vec_st(out2, 32, data);
  vec_st(out3, 48, data);
  vec_st(out4, 64, data);
  vec_st(out5, 80, data);
  vec_st(out6, 96, data);
  vec_st(out7, 112, data);
}
