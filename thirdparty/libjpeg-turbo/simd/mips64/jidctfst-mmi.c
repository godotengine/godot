/*
 * Loongson MMI optimizations for libjpeg-turbo
 *
 * Copyright (C) 2014-2015, 2018-2019, D. R. Commander.  All Rights Reserved.
 * Copyright (C) 2016-2018, Loongson Technology Corporation Limited, BeiJing.
 *                          All Rights Reserved.
 * Authors:  LiuQingfa <liuqingfa-hf@loongson.cn>
 *
 * Based on the x86 SIMD extension for IJG JPEG library
 * Copyright (C) 1999-2006, MIYASAKA Masaru.
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

/* FAST INTEGER INVERSE DCT */

#include "jsimd_mmi.h"


#define CONST_BITS  8
#define PASS1_BITS  2

#define FIX_1_082  ((short)277)                   /* FIX(1.082392200) */
#define FIX_1_414  ((short)362)                   /* FIX(1.414213562) */
#define FIX_1_847  ((short)473)                   /* FIX(1.847759065) */
#define FIX_2_613  ((short)669)                   /* FIX(2.613125930) */
#define FIX_1_613  ((short)(FIX_2_613 - 256 * 3)) /* FIX(2.613125930) - FIX(1) */

#define PRE_MULTIPLY_SCALE_BITS  2
#define CONST_SHIFT  (16 - PRE_MULTIPLY_SCALE_BITS - CONST_BITS)

enum const_index {
  index_PW_F1082,
  index_PW_F1414,
  index_PW_F1847,
  index_PW_MF1613,
  index_PB_CENTERJSAMP
};

static uint64_t const_value[] = {
  _uint64_set1_pi16(FIX_1_082 << CONST_SHIFT),
  _uint64_set1_pi16(FIX_1_414 << CONST_SHIFT),
  _uint64_set1_pi16(FIX_1_847 << CONST_SHIFT),
  _uint64_set1_pi16(-FIX_1_613 << CONST_SHIFT),
  _uint64_set1_pi8(CENTERJSAMPLE)
};

#define PW_F1414        get_const_value(index_PW_F1414)
#define PW_F1847        get_const_value(index_PW_F1847)
#define PW_MF1613       get_const_value(index_PW_MF1613)
#define PW_F1082        get_const_value(index_PW_F1082)
#define PB_CENTERJSAMP  get_const_value(index_PB_CENTERJSAMP)


#define test_m32_zero(mm32)  (!(*(uint32_t *)&mm32))
#define test_m64_zero(mm64)  (!(*(uint64_t *)&mm64))


#define DO_IDCT_COMMON() { \
  tmp7 = _mm_add_pi16(z11, z13); \
  \
  tmp11 = _mm_sub_pi16(z11, z13); \
  tmp11 = _mm_slli_pi16(tmp11, PRE_MULTIPLY_SCALE_BITS); \
  tmp11 = _mm_mulhi_pi16(tmp11, PW_F1414); \
  \
  tmp10 = _mm_slli_pi16(z12, PRE_MULTIPLY_SCALE_BITS); \
  tmp12 = _mm_slli_pi16(z10, PRE_MULTIPLY_SCALE_BITS); \
  \
  /* To avoid overflow... \
   * \
   * (Original) \
   * tmp12 = -2.613125930 * z10 + z5; \
   * \
   * (This implementation) \
   * tmp12 = (-1.613125930 - 1) * z10 + z5; \
   *       = -1.613125930 * z10 - z10 + z5; \
   */ \
  \
  z5 = _mm_add_pi16(tmp10, tmp12); \
  z5 = _mm_mulhi_pi16(z5, PW_F1847); \
  \
  tmp10 = _mm_mulhi_pi16(tmp10, PW_F1082); \
  tmp10 = _mm_sub_pi16(tmp10, z5); \
  tmp12 = _mm_mulhi_pi16(tmp12, PW_MF1613); \
  tmp12 = _mm_sub_pi16(tmp12, z10); \
  tmp12 = _mm_sub_pi16(tmp12, z10); \
  tmp12 = _mm_sub_pi16(tmp12, z10); \
  tmp12 = _mm_add_pi16(tmp12, z5); \
  \
  /* Final output stage */ \
  \
  tmp6 = _mm_sub_pi16(tmp12, tmp7); \
  tmp5 = _mm_sub_pi16(tmp11, tmp6); \
  tmp4 = _mm_add_pi16(tmp10, tmp5); \
  \
  out0 = _mm_add_pi16(tmp0, tmp7); \
  out7 = _mm_sub_pi16(tmp0, tmp7); \
  out1 = _mm_add_pi16(tmp1, tmp6); \
  out6 = _mm_sub_pi16(tmp1, tmp6); \
  \
  out2 = _mm_add_pi16(tmp2, tmp5); \
  out5 = _mm_sub_pi16(tmp2, tmp5); \
  out4 = _mm_add_pi16(tmp3, tmp4); \
  out3 = _mm_sub_pi16(tmp3, tmp4); \
}

#define DO_IDCT_PASS1(iter) { \
  __m64 col0l, col1l, col2l, col3l, col4l, col5l, col6l, col7l; \
  __m64 quant0l, quant1l, quant2l, quant3l; \
  __m64 quant4l, quant5l, quant6l, quant7l; \
  __m64 row01a, row01b, row01c, row01d, row23a, row23b, row23c, row23d; \
  __m64 row0l, row0h, row1l, row1h, row2l, row2h, row3l, row3h; \
  __m32 col0a, col1a, mm0; \
  \
  col0a = _mm_load_si32((__m32 *)&inptr[DCTSIZE * 1]); \
  col1a = _mm_load_si32((__m32 *)&inptr[DCTSIZE * 2]); \
  mm0 = _mm_or_si32(col0a, col1a); \
  \
  if (test_m32_zero(mm0)) { \
    __m64 mm1, mm2; \
    \
    col0l = _mm_load_si64((__m64 *)&inptr[DCTSIZE * 0]); \
    col1l = _mm_load_si64((__m64 *)&inptr[DCTSIZE * 1]); \
    col2l = _mm_load_si64((__m64 *)&inptr[DCTSIZE * 2]); \
    col3l = _mm_load_si64((__m64 *)&inptr[DCTSIZE * 3]); \
    col4l = _mm_load_si64((__m64 *)&inptr[DCTSIZE * 4]); \
    col5l = _mm_load_si64((__m64 *)&inptr[DCTSIZE * 5]); \
    col6l = _mm_load_si64((__m64 *)&inptr[DCTSIZE * 6]); \
    col7l = _mm_load_si64((__m64 *)&inptr[DCTSIZE * 7]); \
    \
    mm1 = _mm_or_si64(col1l, col3l); \
    mm2 = _mm_or_si64(col2l, col4l); \
    mm1 = _mm_or_si64(mm1, col5l); \
    mm2 = _mm_or_si64(mm2, col6l); \
    mm1 = _mm_or_si64(mm1, col7l); \
    mm1 = _mm_or_si64(mm1, mm2); \
    \
    if (test_m64_zero(mm1)) { \
      __m64 dcval, dcvall, dcvalh, row0, row1, row2, row3; \
      \
      /* AC terms all zero */ \
      \
      quant0l = _mm_load_si64((__m64 *)&quantptr[DCTSIZE * 0]); \
      \
      dcval = _mm_mullo_pi16(col0l, quant0l);    /* dcval=(00 10 20 30) */ \
      \
      dcvall = _mm_unpacklo_pi16(dcval, dcval);  /* dcvall=(00 00 10 10) */ \
      dcvalh = _mm_unpackhi_pi16(dcval, dcval);  /* dcvalh=(20 20 30 30) */ \
      \
      row0 = _mm_unpacklo_pi32(dcvall, dcvall);  /* row0=(00 00 00 00) */ \
      row1 = _mm_unpackhi_pi32(dcvall, dcvall);  /* row1=(10 10 10 10) */ \
      row2 = _mm_unpacklo_pi32(dcvalh, dcvalh);  /* row2=(20 20 20 20) */ \
      row3 = _mm_unpackhi_pi32(dcvalh, dcvalh);  /* row3=(30 30 30 30) */ \
      \
      _mm_store_si64((__m64 *)&wsptr[DCTSIZE * 0], row0); \
      _mm_store_si64((__m64 *)&wsptr[DCTSIZE * 0 + 4], row0); \
      _mm_store_si64((__m64 *)&wsptr[DCTSIZE * 1], row1); \
      _mm_store_si64((__m64 *)&wsptr[DCTSIZE * 1 + 4], row1); \
      _mm_store_si64((__m64 *)&wsptr[DCTSIZE * 2], row2); \
      _mm_store_si64((__m64 *)&wsptr[DCTSIZE * 2 + 4], row2); \
      _mm_store_si64((__m64 *)&wsptr[DCTSIZE * 3], row3); \
      _mm_store_si64((__m64 *)&wsptr[DCTSIZE * 3 + 4], row3); \
      \
      goto nextcolumn##iter; \
    } \
  } \
  \
  /* Even part */ \
  \
  col0l = _mm_load_si64((__m64 *)&inptr[DCTSIZE * 0]);  /* (00 10 20 30) */ \
  col2l = _mm_load_si64((__m64 *)&inptr[DCTSIZE * 2]);  /* (02 12 22 32) */ \
  col4l = _mm_load_si64((__m64 *)&inptr[DCTSIZE * 4]);  /* (04 14 24 34) */ \
  col6l = _mm_load_si64((__m64 *)&inptr[DCTSIZE * 6]);  /* (06 16 26 36) */ \
  \
  quant0l = _mm_load_si64((__m64 *)&quantptr[DCTSIZE * 0]); \
  quant2l = _mm_load_si64((__m64 *)&quantptr[DCTSIZE * 2]); \
  quant4l = _mm_load_si64((__m64 *)&quantptr[DCTSIZE * 4]); \
  quant6l = _mm_load_si64((__m64 *)&quantptr[DCTSIZE * 6]); \
  \
  tmp0 = _mm_mullo_pi16(col0l, quant0l); \
  tmp1 = _mm_mullo_pi16(col2l, quant2l); \
  tmp2 = _mm_mullo_pi16(col4l, quant4l); \
  tmp3 = _mm_mullo_pi16(col6l, quant6l); \
  \
  tmp10 = _mm_add_pi16(tmp0, tmp2); \
  tmp11 = _mm_sub_pi16(tmp0, tmp2); \
  tmp13 = _mm_add_pi16(tmp1, tmp3); \
  \
  tmp12 = _mm_sub_pi16(tmp1, tmp3); \
  tmp12 = _mm_slli_pi16(tmp12, PRE_MULTIPLY_SCALE_BITS); \
  tmp12 = _mm_mulhi_pi16(tmp12, PW_F1414); \
  tmp12 = _mm_sub_pi16(tmp12, tmp13); \
  \
  tmp0 = _mm_add_pi16(tmp10, tmp13); \
  tmp3 = _mm_sub_pi16(tmp10, tmp13); \
  tmp1 = _mm_add_pi16(tmp11, tmp12); \
  tmp2 = _mm_sub_pi16(tmp11, tmp12); \
  \
  /* Odd part */ \
  \
  col1l = _mm_load_si64((__m64 *)&inptr[DCTSIZE * 1]);  /* (01 11 21 31) */ \
  col3l = _mm_load_si64((__m64 *)&inptr[DCTSIZE * 3]);  /* (03 13 23 33) */ \
  col5l = _mm_load_si64((__m64 *)&inptr[DCTSIZE * 5]);  /* (05 15 25 35) */ \
  col7l = _mm_load_si64((__m64 *)&inptr[DCTSIZE * 7]);  /* (07 17 27 37) */ \
  \
  quant1l = _mm_load_si64((__m64 *)&quantptr[DCTSIZE * 1]); \
  quant3l = _mm_load_si64((__m64 *)&quantptr[DCTSIZE * 3]); \
  quant5l = _mm_load_si64((__m64 *)&quantptr[DCTSIZE * 5]); \
  quant7l = _mm_load_si64((__m64 *)&quantptr[DCTSIZE * 7]); \
  \
  tmp4 = _mm_mullo_pi16(col1l, quant1l); \
  tmp5 = _mm_mullo_pi16(col3l, quant3l); \
  tmp6 = _mm_mullo_pi16(col5l, quant5l); \
  tmp7 = _mm_mullo_pi16(col7l, quant7l); \
  \
  z13 = _mm_add_pi16(tmp6, tmp5); \
  z10 = _mm_sub_pi16(tmp6, tmp5); \
  z11 = _mm_add_pi16(tmp4, tmp7); \
  z12 = _mm_sub_pi16(tmp4, tmp7); \
  \
  DO_IDCT_COMMON() \
  \
  /* out0=(00 10 20 30), out1=(01 11 21 31) */ \
  /* out2=(02 12 22 32), out3=(03 13 23 33) */ \
  /* out4=(04 14 24 34), out5=(05 15 25 35) */ \
  /* out6=(06 16 26 36), out7=(07 17 27 37) */ \
  \
  /* Transpose coefficients */ \
  \
  row01a = _mm_unpacklo_pi16(out0, out1);     /* row01a=(00 01 10 11) */ \
  row23a = _mm_unpackhi_pi16(out0, out1);     /* row23a=(20 21 30 31) */ \
  row01d = _mm_unpacklo_pi16(out6, out7);     /* row01d=(06 07 16 17) */ \
  row23d = _mm_unpackhi_pi16(out6, out7);     /* row23d=(26 27 36 37) */ \
  \
  row01b = _mm_unpacklo_pi16(out2, out3);     /* row01b=(02 03 12 13) */ \
  row23b = _mm_unpackhi_pi16(out2, out3);     /* row23b=(22 23 32 33) */ \
  row01c = _mm_unpacklo_pi16(out4, out5);     /* row01c=(04 05 14 15) */ \
  row23c = _mm_unpackhi_pi16(out4, out5);     /* row23c=(24 25 34 35) */ \
  \
  row0l = _mm_unpacklo_pi32(row01a, row01b);  /* row0l=(00 01 02 03) */ \
  row1l = _mm_unpackhi_pi32(row01a, row01b);  /* row1l=(10 11 12 13) */ \
  row2l = _mm_unpacklo_pi32(row23a, row23b);  /* row2l=(20 21 22 23) */ \
  row3l = _mm_unpackhi_pi32(row23a, row23b);  /* row3l=(30 31 32 33) */ \
  \
  row0h = _mm_unpacklo_pi32(row01c, row01d);  /* row0h=(04 05 06 07) */ \
  row1h = _mm_unpackhi_pi32(row01c, row01d);  /* row1h=(14 15 16 17) */ \
  row2h = _mm_unpacklo_pi32(row23c, row23d);  /* row2h=(24 25 26 27) */ \
  row3h = _mm_unpackhi_pi32(row23c, row23d);  /* row3h=(34 35 36 37) */ \
  \
  _mm_store_si64((__m64 *)&wsptr[DCTSIZE * 0], row0l); \
  _mm_store_si64((__m64 *)&wsptr[DCTSIZE * 0 + 4], row0h); \
  _mm_store_si64((__m64 *)&wsptr[DCTSIZE * 1], row1l); \
  _mm_store_si64((__m64 *)&wsptr[DCTSIZE * 1 + 4], row1h); \
  _mm_store_si64((__m64 *)&wsptr[DCTSIZE * 2], row2l); \
  _mm_store_si64((__m64 *)&wsptr[DCTSIZE * 2 + 4], row2h); \
  _mm_store_si64((__m64 *)&wsptr[DCTSIZE * 3], row3l); \
  _mm_store_si64((__m64 *)&wsptr[DCTSIZE * 3 + 4], row3h); \
}

#define DO_IDCT_PASS2(ctr) { \
  __m64 row0l, row1l, row2l, row3l, row4l, row5l, row6l, row7l; \
  __m64 col0123a, col0123b, col0123c, col0123d; \
  __m64 col01l, col01h, col23l, col23h; \
  __m64 col0, col1, col2, col3; \
  __m64 row06, row17, row24, row35; \
  \
  row0l = _mm_load_si64((__m64 *)&wsptr[DCTSIZE * 0]);  /* (00 01 02 03) */ \
  row1l = _mm_load_si64((__m64 *)&wsptr[DCTSIZE * 1]);  /* (10 11 12 13) */ \
  row2l = _mm_load_si64((__m64 *)&wsptr[DCTSIZE * 2]);  /* (20 21 22 23) */ \
  row3l = _mm_load_si64((__m64 *)&wsptr[DCTSIZE * 3]);  /* (30 31 32 33) */ \
  row4l = _mm_load_si64((__m64 *)&wsptr[DCTSIZE * 4]);  /* (40 41 42 43) */ \
  row5l = _mm_load_si64((__m64 *)&wsptr[DCTSIZE * 5]);  /* (50 51 52 53) */ \
  row6l = _mm_load_si64((__m64 *)&wsptr[DCTSIZE * 6]);  /* (60 61 62 63) */ \
  row7l = _mm_load_si64((__m64 *)&wsptr[DCTSIZE * 7]);  /* (70 71 72 73) */ \
  \
  /* Even part */ \
  \
  tmp10 = _mm_add_pi16(row0l, row4l); \
  tmp11 = _mm_sub_pi16(row0l, row4l); \
  tmp13 = _mm_add_pi16(row2l, row6l); \
  \
  tmp12 = _mm_sub_pi16(row2l, row6l); \
  tmp12 = _mm_slli_pi16(tmp12, PRE_MULTIPLY_SCALE_BITS); \
  tmp12 = _mm_mulhi_pi16(tmp12, PW_F1414); \
  tmp12 = _mm_sub_pi16(tmp12, tmp13); \
  \
  tmp0 = _mm_add_pi16(tmp10, tmp13); \
  tmp3 = _mm_sub_pi16(tmp10, tmp13); \
  tmp1 = _mm_add_pi16(tmp11, tmp12); \
  tmp2 = _mm_sub_pi16(tmp11, tmp12); \
  \
  /* Odd part */ \
  \
  z13 = _mm_add_pi16(row5l, row3l); \
  z10 = _mm_sub_pi16(row5l, row3l); \
  z11 = _mm_add_pi16(row1l, row7l); \
  z12 = _mm_sub_pi16(row1l, row7l); \
  \
  DO_IDCT_COMMON() \
  \
  /* out0=(00 01 02 03), out1=(10 11 12 13) */ \
  /* out2=(20 21 22 23), out3=(30 31 32 33) */ \
  /* out4=(40 41 42 43), out5=(50 51 52 53) */ \
  /* out6=(60 61 62 63), out7=(70 71 72 73) */ \
  \
  out0 = _mm_srai_pi16(out0, PASS1_BITS + 3); \
  out1 = _mm_srai_pi16(out1, PASS1_BITS + 3); \
  out2 = _mm_srai_pi16(out2, PASS1_BITS + 3); \
  out3 = _mm_srai_pi16(out3, PASS1_BITS + 3); \
  out4 = _mm_srai_pi16(out4, PASS1_BITS + 3); \
  out5 = _mm_srai_pi16(out5, PASS1_BITS + 3); \
  out6 = _mm_srai_pi16(out6, PASS1_BITS + 3); \
  out7 = _mm_srai_pi16(out7, PASS1_BITS + 3); \
  \
  row06 = _mm_packs_pi16(out0, out6);  /* row06=(00 01 02 03 60 61 62 63) */ \
  row17 = _mm_packs_pi16(out1, out7);  /* row17=(10 11 12 13 70 71 72 73) */ \
  row24 = _mm_packs_pi16(out2, out4);  /* row24=(20 21 22 23 40 41 42 43) */ \
  row35 = _mm_packs_pi16(out3, out5);  /* row35=(30 31 32 33 50 51 52 53) */ \
  \
  row06 = _mm_add_pi8(row06, PB_CENTERJSAMP); \
  row17 = _mm_add_pi8(row17, PB_CENTERJSAMP); \
  row24 = _mm_add_pi8(row24, PB_CENTERJSAMP); \
  row35 = _mm_add_pi8(row35, PB_CENTERJSAMP); \
  \
  /* Transpose coefficients */ \
  \
  col0123a = _mm_unpacklo_pi8(row06, row17);  /* col0123a=(00 10 01 11 02 12 03 13) */ \
  col0123d = _mm_unpackhi_pi8(row06, row17);  /* col0123d=(60 70 61 71 62 72 63 73) */ \
  col0123b = _mm_unpacklo_pi8(row24, row35);  /* col0123b=(20 30 21 31 22 32 23 33) */ \
  col0123c = _mm_unpackhi_pi8(row24, row35);  /* col0123c=(40 50 41 51 42 52 43 53) */ \
  \
  col01l = _mm_unpacklo_pi16(col0123a, col0123b);  /* col01l=(00 10 20 30 01 11 21 31) */ \
  col23l = _mm_unpackhi_pi16(col0123a, col0123b);  /* col23l=(02 12 22 32 03 13 23 33) */ \
  col01h = _mm_unpacklo_pi16(col0123c, col0123d);  /* col01h=(40 50 60 70 41 51 61 71) */ \
  col23h = _mm_unpackhi_pi16(col0123c, col0123d);  /* col23h=(42 52 62 72 43 53 63 73) */ \
  \
  col0 = _mm_unpacklo_pi32(col01l, col01h);   /* col0=(00 10 20 30 40 50 60 70) */ \
  col1 = _mm_unpackhi_pi32(col01l, col01h);   /* col1=(01 11 21 31 41 51 61 71) */ \
  col2 = _mm_unpacklo_pi32(col23l, col23h);   /* col2=(02 12 22 32 42 52 62 72) */ \
  col3 = _mm_unpackhi_pi32(col23l, col23h);   /* col3=(03 13 23 33 43 53 63 73) */ \
  \
  _mm_store_si64((__m64 *)(output_buf[ctr + 0] + output_col), col0); \
  _mm_store_si64((__m64 *)(output_buf[ctr + 1] + output_col), col1); \
  _mm_store_si64((__m64 *)(output_buf[ctr + 2] + output_col), col2); \
  _mm_store_si64((__m64 *)(output_buf[ctr + 3] + output_col), col3); \
}

void jsimd_idct_ifast_mmi(void *dct_table, JCOEFPTR coef_block,
                          JSAMPARRAY output_buf, JDIMENSION output_col)
{
  __m64 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
  __m64 tmp10, tmp11, tmp12, tmp13;
  __m64 out0, out1, out2, out3, out4, out5, out6, out7;
  __m64 z5, z10, z11, z12, z13;
  JCOEFPTR inptr;
  ISLOW_MULT_TYPE *quantptr;
  JCOEF *wsptr;
  JCOEF workspace[DCTSIZE2];  /* buffers data between passes */

  /* Pass 1: process columns. */

  inptr = coef_block;
  quantptr = (ISLOW_MULT_TYPE *)dct_table;
  wsptr = workspace;

  DO_IDCT_PASS1(1)
nextcolumn1:
  inptr += 4;
  quantptr += 4;
  wsptr += DCTSIZE * 4;
  DO_IDCT_PASS1(2)
nextcolumn2:

  /* Pass 2: process rows. */

  wsptr = workspace;

  DO_IDCT_PASS2(0)
  wsptr += 4;
  DO_IDCT_PASS2(4)
}
