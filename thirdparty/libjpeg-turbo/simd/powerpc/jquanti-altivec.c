/*
 * AltiVec optimizations for libjpeg-turbo
 *
 * Copyright (C) 2014-2015, 2024, D. R. Commander.  All Rights Reserved.
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

/* INTEGER QUANTIZATION AND SAMPLE CONVERSION */

#include "jsimd_altivec.h"


/* NOTE: The address will either be aligned or offset by 8 bytes, so we can
 * always get the data we want by using a single vector load (although we may
 * have to permute the result.)
 */
#ifdef __BIG_ENDIAN__

#define LOAD_ROW(row) { \
  elemptr = sample_data[row] + start_col; \
  in##row = vec_ld(0, elemptr); \
  if ((size_t)elemptr & 15) \
    in##row = vec_perm(in##row, in##row, vec_lvsl(0, elemptr)); \
}

#else

#define LOAD_ROW(row) { \
  elemptr = sample_data[row] + start_col; \
  in##row = vec_vsx_ld(0, elemptr); \
}

#endif


void jsimd_convsamp_altivec(JSAMPARRAY sample_data, JDIMENSION start_col,
                            DCTELEM *workspace)
{
  JSAMPROW elemptr;

  __vector unsigned char in0, in1, in2, in3, in4, in5, in6, in7;
  __vector short out0, out1, out2, out3, out4, out5, out6, out7;

  /* Constants */
  __vector short pw_centerjsamp = { __8X(CENTERJSAMPLE) };
  __vector unsigned char pb_zero = { __16X(0) };

  LOAD_ROW(0);
  LOAD_ROW(1);
  LOAD_ROW(2);
  LOAD_ROW(3);
  LOAD_ROW(4);
  LOAD_ROW(5);
  LOAD_ROW(6);
  LOAD_ROW(7);

  out0 = (__vector short)VEC_UNPACKHU(in0);
  out1 = (__vector short)VEC_UNPACKHU(in1);
  out2 = (__vector short)VEC_UNPACKHU(in2);
  out3 = (__vector short)VEC_UNPACKHU(in3);
  out4 = (__vector short)VEC_UNPACKHU(in4);
  out5 = (__vector short)VEC_UNPACKHU(in5);
  out6 = (__vector short)VEC_UNPACKHU(in6);
  out7 = (__vector short)VEC_UNPACKHU(in7);

  out0 = vec_sub(out0, pw_centerjsamp);
  out1 = vec_sub(out1, pw_centerjsamp);
  out2 = vec_sub(out2, pw_centerjsamp);
  out3 = vec_sub(out3, pw_centerjsamp);
  out4 = vec_sub(out4, pw_centerjsamp);
  out5 = vec_sub(out5, pw_centerjsamp);
  out6 = vec_sub(out6, pw_centerjsamp);
  out7 = vec_sub(out7, pw_centerjsamp);

  vec_st(out0, 0, workspace);
  vec_st(out1, 16, workspace);
  vec_st(out2, 32, workspace);
  vec_st(out3, 48, workspace);
  vec_st(out4, 64, workspace);
  vec_st(out5, 80, workspace);
  vec_st(out6, 96, workspace);
  vec_st(out7, 112, workspace);
}


#define WORD_BIT  16

/* There is no AltiVec 16-bit unsigned multiply instruction, hence this.
   We basically need an unsigned equivalent of vec_madds(). */

#define MULTIPLY(vs0, vs1, out) { \
  tmpe = vec_mule((__vector unsigned short)vs0, \
                  (__vector unsigned short)vs1); \
  tmpo = vec_mulo((__vector unsigned short)vs0, \
                  (__vector unsigned short)vs1); \
  out = (__vector short)vec_perm((__vector unsigned short)tmpe, \
                                 (__vector unsigned short)tmpo, \
                                 shift_pack_index); \
}

void jsimd_quantize_altivec(JCOEFPTR coef_block, DCTELEM *divisors,
                            DCTELEM *workspace)
{
  __vector short row0, row1, row2, row3, row4, row5, row6, row7,
    row0s, row1s, row2s, row3s, row4s, row5s, row6s, row7s,
    corr0, corr1, corr2, corr3, corr4, corr5, corr6, corr7,
    recip0, recip1, recip2, recip3, recip4, recip5, recip6, recip7,
    scale0, scale1, scale2, scale3, scale4, scale5, scale6, scale7;
  __vector unsigned int tmpe, tmpo;

  /* Constants */
  __vector unsigned short pw_word_bit_m1 = { __8X(WORD_BIT - 1) };
#ifdef __BIG_ENDIAN__
  __vector unsigned char shift_pack_index =
    {  0,  1, 16, 17,  4,  5, 20, 21,  8,  9, 24, 25, 12, 13, 28, 29 };
#else
  __vector unsigned char shift_pack_index =
    {  2,  3, 18, 19,  6,  7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31 };
#endif

  row0 = vec_ld(0, workspace);
  row1 = vec_ld(16, workspace);
  row2 = vec_ld(32, workspace);
  row3 = vec_ld(48, workspace);
  row4 = vec_ld(64, workspace);
  row5 = vec_ld(80, workspace);
  row6 = vec_ld(96, workspace);
  row7 = vec_ld(112, workspace);

  /* Branch-less absolute value */
  row0s = vec_sra(row0, pw_word_bit_m1);
  row1s = vec_sra(row1, pw_word_bit_m1);
  row2s = vec_sra(row2, pw_word_bit_m1);
  row3s = vec_sra(row3, pw_word_bit_m1);
  row4s = vec_sra(row4, pw_word_bit_m1);
  row5s = vec_sra(row5, pw_word_bit_m1);
  row6s = vec_sra(row6, pw_word_bit_m1);
  row7s = vec_sra(row7, pw_word_bit_m1);
  row0 = vec_xor(row0, row0s);
  row1 = vec_xor(row1, row1s);
  row2 = vec_xor(row2, row2s);
  row3 = vec_xor(row3, row3s);
  row4 = vec_xor(row4, row4s);
  row5 = vec_xor(row5, row5s);
  row6 = vec_xor(row6, row6s);
  row7 = vec_xor(row7, row7s);
  row0 = vec_sub(row0, row0s);
  row1 = vec_sub(row1, row1s);
  row2 = vec_sub(row2, row2s);
  row3 = vec_sub(row3, row3s);
  row4 = vec_sub(row4, row4s);
  row5 = vec_sub(row5, row5s);
  row6 = vec_sub(row6, row6s);
  row7 = vec_sub(row7, row7s);

  corr0 = vec_ld(DCTSIZE2 * 2, divisors);
  corr1 = vec_ld(DCTSIZE2 * 2 + 16, divisors);
  corr2 = vec_ld(DCTSIZE2 * 2 + 32, divisors);
  corr3 = vec_ld(DCTSIZE2 * 2 + 48, divisors);
  corr4 = vec_ld(DCTSIZE2 * 2 + 64, divisors);
  corr5 = vec_ld(DCTSIZE2 * 2 + 80, divisors);
  corr6 = vec_ld(DCTSIZE2 * 2 + 96, divisors);
  corr7 = vec_ld(DCTSIZE2 * 2 + 112, divisors);

  row0 = vec_add(row0, corr0);
  row1 = vec_add(row1, corr1);
  row2 = vec_add(row2, corr2);
  row3 = vec_add(row3, corr3);
  row4 = vec_add(row4, corr4);
  row5 = vec_add(row5, corr5);
  row6 = vec_add(row6, corr6);
  row7 = vec_add(row7, corr7);

  recip0 = vec_ld(0, divisors);
  recip1 = vec_ld(16, divisors);
  recip2 = vec_ld(32, divisors);
  recip3 = vec_ld(48, divisors);
  recip4 = vec_ld(64, divisors);
  recip5 = vec_ld(80, divisors);
  recip6 = vec_ld(96, divisors);
  recip7 = vec_ld(112, divisors);

  MULTIPLY(row0, recip0, row0);
  MULTIPLY(row1, recip1, row1);
  MULTIPLY(row2, recip2, row2);
  MULTIPLY(row3, recip3, row3);
  MULTIPLY(row4, recip4, row4);
  MULTIPLY(row5, recip5, row5);
  MULTIPLY(row6, recip6, row6);
  MULTIPLY(row7, recip7, row7);

  scale0 = vec_ld(DCTSIZE2 * 4, divisors);
  scale1 = vec_ld(DCTSIZE2 * 4 + 16, divisors);
  scale2 = vec_ld(DCTSIZE2 * 4 + 32, divisors);
  scale3 = vec_ld(DCTSIZE2 * 4 + 48, divisors);
  scale4 = vec_ld(DCTSIZE2 * 4 + 64, divisors);
  scale5 = vec_ld(DCTSIZE2 * 4 + 80, divisors);
  scale6 = vec_ld(DCTSIZE2 * 4 + 96, divisors);
  scale7 = vec_ld(DCTSIZE2 * 4 + 112, divisors);

  MULTIPLY(row0, scale0, row0);
  MULTIPLY(row1, scale1, row1);
  MULTIPLY(row2, scale2, row2);
  MULTIPLY(row3, scale3, row3);
  MULTIPLY(row4, scale4, row4);
  MULTIPLY(row5, scale5, row5);
  MULTIPLY(row6, scale6, row6);
  MULTIPLY(row7, scale7, row7);

  row0 = vec_xor(row0, row0s);
  row1 = vec_xor(row1, row1s);
  row2 = vec_xor(row2, row2s);
  row3 = vec_xor(row3, row3s);
  row4 = vec_xor(row4, row4s);
  row5 = vec_xor(row5, row5s);
  row6 = vec_xor(row6, row6s);
  row7 = vec_xor(row7, row7s);
  row0 = vec_sub(row0, row0s);
  row1 = vec_sub(row1, row1s);
  row2 = vec_sub(row2, row2s);
  row3 = vec_sub(row3, row3s);
  row4 = vec_sub(row4, row4s);
  row5 = vec_sub(row5, row5s);
  row6 = vec_sub(row6, row6s);
  row7 = vec_sub(row7, row7s);

  vec_st(row0, 0, coef_block);
  vec_st(row1, 16, coef_block);
  vec_st(row2, 32, coef_block);
  vec_st(row3, 48, coef_block);
  vec_st(row4, 64, coef_block);
  vec_st(row5, 80, coef_block);
  vec_st(row6, 96, coef_block);
  vec_st(row7, 112, coef_block);
}
