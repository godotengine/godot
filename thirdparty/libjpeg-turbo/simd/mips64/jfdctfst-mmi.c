/*
 * Loongson MMI optimizations for libjpeg-turbo
 *
 * Copyright (C) 2014, 2018-2019, D. R. Commander.  All Rights Reserved.
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

/* FAST INTEGER FORWARD DCT */

#include "jsimd_mmi.h"


#define CONST_BITS  8

#define F_0_382  ((short)98)   /* FIX(0.382683433) */
#define F_0_541  ((short)139)  /* FIX(0.541196100) */
#define F_0_707  ((short)181)  /* FIX(0.707106781) */
#define F_1_306  ((short)334)  /* FIX(1.306562965) */

#define PRE_MULTIPLY_SCALE_BITS  2
#define CONST_SHIFT  (16 - PRE_MULTIPLY_SCALE_BITS - CONST_BITS)

enum const_index {
  index_PW_F0707,
  index_PW_F0382,
  index_PW_F0541,
  index_PW_F1306
};

static uint64_t const_value[] = {
  _uint64_set1_pi16(F_0_707),
  _uint64_set1_pi16(F_0_382),
  _uint64_set1_pi16(F_0_541),
  _uint64_set1_pi16(F_1_306)
};

#define PW_F0707  get_const_value(index_PW_F0707)
#define PW_F0382  get_const_value(index_PW_F0382)
#define PW_F0541  get_const_value(index_PW_F0541)
#define PW_F1306  get_const_value(index_PW_F1306)


#define DO_FDCT_MULTIPLY(out, in, multiplier) { \
  __m64 mulhi, mullo, mul12, mul34; \
  \
  mullo = _mm_mullo_pi16(in, multiplier); \
  mulhi = _mm_mulhi_pi16(in, multiplier); \
  mul12 = _mm_unpacklo_pi16(mullo, mulhi); \
  mul34 = _mm_unpackhi_pi16(mullo, mulhi); \
  mul12 = _mm_srai_pi32(mul12, CONST_BITS); \
  mul34 = _mm_srai_pi32(mul34, CONST_BITS); \
  out = _mm_packs_pi32(mul12, mul34); \
}

#define DO_FDCT_COMMON() { \
  \
  /* Even part */ \
  \
  tmp10 = _mm_add_pi16(tmp0, tmp3); \
  tmp13 = _mm_sub_pi16(tmp0, tmp3); \
  tmp11 = _mm_add_pi16(tmp1, tmp2); \
  tmp12 = _mm_sub_pi16(tmp1, tmp2); \
  \
  out0 = _mm_add_pi16(tmp10, tmp11); \
  out4 = _mm_sub_pi16(tmp10, tmp11); \
  \
  z1 = _mm_add_pi16(tmp12, tmp13); \
  DO_FDCT_MULTIPLY(z1, z1, PW_F0707) \
  \
  out2 = _mm_add_pi16(tmp13, z1); \
  out6 = _mm_sub_pi16(tmp13, z1); \
  \
  /* Odd part */ \
  \
  tmp10 = _mm_add_pi16(tmp4, tmp5); \
  tmp11 = _mm_add_pi16(tmp5, tmp6); \
  tmp12 = _mm_add_pi16(tmp6, tmp7); \
  \
  z5 = _mm_sub_pi16(tmp10, tmp12); \
  DO_FDCT_MULTIPLY(z5, z5, PW_F0382) \
  \
  DO_FDCT_MULTIPLY(z2, tmp10, PW_F0541) \
  z2 = _mm_add_pi16(z2, z5); \
  \
  DO_FDCT_MULTIPLY(z4, tmp12, PW_F1306) \
  z4 = _mm_add_pi16(z4, z5); \
  \
  DO_FDCT_MULTIPLY(z3, tmp11, PW_F0707) \
  \
  z11 = _mm_add_pi16(tmp7, z3); \
  z13 = _mm_sub_pi16(tmp7, z3); \
  \
  out5 = _mm_add_pi16(z13, z2); \
  out3 = _mm_sub_pi16(z13, z2); \
  out1 = _mm_add_pi16(z11, z4); \
  out7 = _mm_sub_pi16(z11, z4); \
}

#define DO_FDCT_PASS1() { \
  __m64 row0l, row0h, row1l, row1h, row2l, row2h, row3l, row3h; \
  __m64 row01a, row01b, row01c, row01d, row23a, row23b, row23c, row23d; \
  __m64 col0, col1, col2, col3, col4, col5, col6, col7; \
  \
  row0l = _mm_load_si64((__m64 *)&dataptr[DCTSIZE * 0]);     /* (00 01 02 03) */ \
  row0h = _mm_load_si64((__m64 *)&dataptr[DCTSIZE * 0 + 4]); /* (04 05 06 07) */ \
  row1l = _mm_load_si64((__m64 *)&dataptr[DCTSIZE * 1]);     /* (10 11 12 13) */ \
  row1h = _mm_load_si64((__m64 *)&dataptr[DCTSIZE * 1 + 4]); /* (14 15 16 17) */ \
  row2l = _mm_load_si64((__m64 *)&dataptr[DCTSIZE * 2]);     /* (20 21 22 23) */ \
  row2h = _mm_load_si64((__m64 *)&dataptr[DCTSIZE * 2 + 4]); /* (24 25 26 27) */ \
  row3l = _mm_load_si64((__m64 *)&dataptr[DCTSIZE * 3]);     /* (30 31 32 33) */ \
  row3h = _mm_load_si64((__m64 *)&dataptr[DCTSIZE * 3 + 4]); /* (34 35 36 37) */ \
  \
  /* Transpose coefficients */ \
  \
  row23a = _mm_unpacklo_pi16(row2l, row3l);   /* row23a=(20 30 21 31) */ \
  row23b = _mm_unpackhi_pi16(row2l, row3l);   /* row23b=(22 32 23 33) */ \
  row23c = _mm_unpacklo_pi16(row2h, row3h);   /* row23c=(24 34 25 35) */ \
  row23d = _mm_unpackhi_pi16(row2h, row3h);   /* row23d=(26 36 27 37) */ \
  \
  row01a = _mm_unpacklo_pi16(row0l, row1l);   /* row01a=(00 10 01 11) */ \
  row01b = _mm_unpackhi_pi16(row0l, row1l);   /* row01b=(02 12 03 13) */ \
  row01c = _mm_unpacklo_pi16(row0h, row1h);   /* row01c=(04 14 05 15) */ \
  row01d = _mm_unpackhi_pi16(row0h, row1h);   /* row01d=(06 16 07 17) */ \
  \
  col0 = _mm_unpacklo_pi32(row01a, row23a);   /* col0=(00 10 20 30) */ \
  col1 = _mm_unpackhi_pi32(row01a, row23a);   /* col1=(01 11 21 31) */ \
  col6 = _mm_unpacklo_pi32(row01d, row23d);   /* col6=(06 16 26 36) */ \
  col7 = _mm_unpackhi_pi32(row01d, row23d);   /* col7=(07 17 27 37) */ \
  \
  tmp6 = _mm_sub_pi16(col1, col6);            /* tmp6=col1-col6 */ \
  tmp7 = _mm_sub_pi16(col0, col7);            /* tmp7=col0-col7 */ \
  tmp1 = _mm_add_pi16(col1, col6);            /* tmp1=col1+col6 */ \
  tmp0 = _mm_add_pi16(col0, col7);            /* tmp0=col0+col7 */ \
  \
  col2 = _mm_unpacklo_pi32(row01b, row23b);   /* col2=(02 12 22 32) */ \
  col3 = _mm_unpackhi_pi32(row01b, row23b);   /* col3=(03 13 23 33) */ \
  col4 = _mm_unpacklo_pi32(row01c, row23c);   /* col4=(04 14 24 34) */ \
  col5 = _mm_unpackhi_pi32(row01c, row23c);   /* col5=(05 15 25 35) */ \
  \
  tmp3 = _mm_add_pi16(col3, col4);            /* tmp3=col3+col4 */ \
  tmp2 = _mm_add_pi16(col2, col5);            /* tmp2=col2+col5 */ \
  tmp4 = _mm_sub_pi16(col3, col4);            /* tmp4=col3-col4 */ \
  tmp5 = _mm_sub_pi16(col2, col5);            /* tmp5=col2-col5 */ \
  \
  DO_FDCT_COMMON() \
  \
  _mm_store_si64((__m64 *)&dataptr[DCTSIZE * 0], out0); \
  _mm_store_si64((__m64 *)&dataptr[DCTSIZE * 0 + 4], out4); \
  _mm_store_si64((__m64 *)&dataptr[DCTSIZE * 1], out1); \
  _mm_store_si64((__m64 *)&dataptr[DCTSIZE * 1 + 4], out5); \
  _mm_store_si64((__m64 *)&dataptr[DCTSIZE * 2], out2); \
  _mm_store_si64((__m64 *)&dataptr[DCTSIZE * 2 + 4], out6); \
  _mm_store_si64((__m64 *)&dataptr[DCTSIZE * 3], out3); \
  _mm_store_si64((__m64 *)&dataptr[DCTSIZE * 3 + 4], out7); \
}

#define DO_FDCT_PASS2() { \
  __m64 col0l, col0h, col1l, col1h, col2l, col2h, col3l, col3h; \
  __m64 col01a, col01b, col01c, col01d, col23a, col23b, col23c, col23d; \
  __m64 row0, row1, row2, row3, row4, row5, row6, row7; \
  \
  col0l = _mm_load_si64((__m64 *)&dataptr[DCTSIZE * 0]);  /* (00 10 20 30) */ \
  col1l = _mm_load_si64((__m64 *)&dataptr[DCTSIZE * 1]);  /* (01 11 21 31) */ \
  col2l = _mm_load_si64((__m64 *)&dataptr[DCTSIZE * 2]);  /* (02 12 22 32) */ \
  col3l = _mm_load_si64((__m64 *)&dataptr[DCTSIZE * 3]);  /* (03 13 23 33) */ \
  col0h = _mm_load_si64((__m64 *)&dataptr[DCTSIZE * 4]);  /* (40 50 60 70) */ \
  col1h = _mm_load_si64((__m64 *)&dataptr[DCTSIZE * 5]);  /* (41 51 61 71) */ \
  col2h = _mm_load_si64((__m64 *)&dataptr[DCTSIZE * 6]);  /* (42 52 62 72) */ \
  col3h = _mm_load_si64((__m64 *)&dataptr[DCTSIZE * 7]);  /* (43 53 63 73) */ \
  \
  /* Transpose coefficients */ \
  \
  col23a = _mm_unpacklo_pi16(col2l, col3l);   /* col23a=(02 03 12 13) */ \
  col23b = _mm_unpackhi_pi16(col2l, col3l);   /* col23b=(22 23 32 33) */ \
  col23c = _mm_unpacklo_pi16(col2h, col3h);   /* col23c=(42 43 52 53) */ \
  col23d = _mm_unpackhi_pi16(col2h, col3h);   /* col23d=(62 63 72 73) */ \
  \
  col01a = _mm_unpacklo_pi16(col0l, col1l);   /* col01a=(00 01 10 11) */ \
  col01b = _mm_unpackhi_pi16(col0l, col1l);   /* col01b=(20 21 30 31) */ \
  col01c = _mm_unpacklo_pi16(col0h, col1h);   /* col01c=(40 41 50 51) */ \
  col01d = _mm_unpackhi_pi16(col0h, col1h);   /* col01d=(60 61 70 71) */ \
  \
  row0 = _mm_unpacklo_pi32(col01a, col23a);   /* row0=(00 01 02 03) */ \
  row1 = _mm_unpackhi_pi32(col01a, col23a);   /* row1=(10 11 12 13) */ \
  row6 = _mm_unpacklo_pi32(col01d, col23d);   /* row6=(60 61 62 63) */ \
  row7 = _mm_unpackhi_pi32(col01d, col23d);   /* row7=(70 71 72 73) */ \
  \
  tmp6 = _mm_sub_pi16(row1, row6);            /* tmp6=row1-row6 */ \
  tmp7 = _mm_sub_pi16(row0, row7);            /* tmp7=row0-row7 */ \
  tmp1 = _mm_add_pi16(row1, row6);            /* tmp1=row1+row6 */ \
  tmp0 = _mm_add_pi16(row0, row7);            /* tmp0=row0+row7 */ \
  \
  row2 = _mm_unpacklo_pi32(col01b, col23b);   /* row2=(20 21 22 23) */ \
  row3 = _mm_unpackhi_pi32(col01b, col23b);   /* row3=(30 31 32 33) */ \
  row4 = _mm_unpacklo_pi32(col01c, col23c);   /* row4=(40 41 42 43) */ \
  row5 = _mm_unpackhi_pi32(col01c, col23c);   /* row5=(50 51 52 53) */ \
  \
  tmp3 = _mm_add_pi16(row3, row4);            /* tmp3=row3+row4 */ \
  tmp2 = _mm_add_pi16(row2, row5);            /* tmp2=row2+row5 */ \
  tmp4 = _mm_sub_pi16(row3, row4);            /* tmp4=row3-row4 */ \
  tmp5 = _mm_sub_pi16(row2, row5);            /* tmp5=row2-row5 */ \
  \
  DO_FDCT_COMMON() \
  \
  _mm_store_si64((__m64 *)&dataptr[DCTSIZE * 0], out0); \
  _mm_store_si64((__m64 *)&dataptr[DCTSIZE * 1], out1); \
  _mm_store_si64((__m64 *)&dataptr[DCTSIZE * 2], out2); \
  _mm_store_si64((__m64 *)&dataptr[DCTSIZE * 3], out3); \
  _mm_store_si64((__m64 *)&dataptr[DCTSIZE * 4], out4); \
  _mm_store_si64((__m64 *)&dataptr[DCTSIZE * 5], out5); \
  _mm_store_si64((__m64 *)&dataptr[DCTSIZE * 6], out6); \
  _mm_store_si64((__m64 *)&dataptr[DCTSIZE * 7], out7); \
}

void jsimd_fdct_ifast_mmi(DCTELEM *data)
{
  __m64 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
  __m64 out0, out1, out2, out3, out4, out5, out6, out7;
  __m64 tmp10, tmp11, tmp12, tmp13, z1, z2, z3, z4, z5, z11, z13;
  DCTELEM *dataptr = data;

  /* Pass 1: process rows. */

  DO_FDCT_PASS1()
  dataptr += DCTSIZE * 4;
  DO_FDCT_PASS1()

  /* Pass 2: process columns. */

  dataptr = data;
  DO_FDCT_PASS2()
  dataptr += 4;
  DO_FDCT_PASS2()
}
