/*
 * Loongson MMI optimizations for libjpeg-turbo
 *
 * Copyright (C) 2014, 2018, 2020, D. R. Commander.  All Rights Reserved.
 * Copyright (C) 2016-2017, Loongson Technology Corporation Limited, BeiJing.
 *                          All Rights Reserved.
 * Authors:  ZhuChen     <zhuchen@loongson.cn>
 *           CaiWanwei   <caiwanwei@loongson.cn>
 *           SunZhangzhi <sunzhangzhi-cq@loongson.cn>
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

/* ACCURATE INTEGER FORWARD DCT */

#include "jsimd_mmi.h"


#define CONST_BITS  13
#define PASS1_BITS  2
#define DESCALE_P1  (CONST_BITS - PASS1_BITS)
#define DESCALE_P2  (CONST_BITS + PASS1_BITS)

#define FIX_0_298  ((short)2446)   /* FIX(0.298631336) */
#define FIX_0_390  ((short)3196)   /* FIX(0.390180644) */
#define FIX_0_541  ((short)4433)   /* FIX(0.541196100) */
#define FIX_0_765  ((short)6270)   /* FIX(0.765366865) */
#define FIX_0_899  ((short)7373)   /* FIX(0.899976223) */
#define FIX_1_175  ((short)9633)   /* FIX(1.175875602) */
#define FIX_1_501  ((short)12299)  /* FIX(1.501321110) */
#define FIX_1_847  ((short)15137)  /* FIX(1.847759065) */
#define FIX_1_961  ((short)16069)  /* FIX(1.961570560) */
#define FIX_2_053  ((short)16819)  /* FIX(2.053119869) */
#define FIX_2_562  ((short)20995)  /* FIX(2.562915447) */
#define FIX_3_072  ((short)25172)  /* FIX(3.072711026) */

enum const_index {
  index_PW_F130_F054,
  index_PW_F054_MF130,
  index_PW_MF078_F117,
  index_PW_F117_F078,
  index_PW_MF060_MF089,
  index_PW_MF089_F060,
  index_PW_MF050_MF256,
  index_PW_MF256_F050,
  index_PD_DESCALE_P1,
  index_PD_DESCALE_P2,
  index_PW_DESCALE_P2X
};

static uint64_t const_value[] = {
  _uint64_set_pi16(FIX_0_541, (FIX_0_541 + FIX_0_765),
                   FIX_0_541, (FIX_0_541 + FIX_0_765)),
  _uint64_set_pi16((FIX_0_541 - FIX_1_847), FIX_0_541,
                   (FIX_0_541 - FIX_1_847), FIX_0_541),
  _uint64_set_pi16(FIX_1_175, (FIX_1_175 - FIX_1_961),
                   FIX_1_175, (FIX_1_175 - FIX_1_961)),
  _uint64_set_pi16((FIX_1_175 - FIX_0_390), FIX_1_175,
                   (FIX_1_175 - FIX_0_390), FIX_1_175),
  _uint64_set_pi16(-FIX_0_899, (FIX_0_298 - FIX_0_899),
                   -FIX_0_899, (FIX_0_298 - FIX_0_899)),
  _uint64_set_pi16((FIX_1_501 - FIX_0_899), -FIX_0_899,
                   (FIX_1_501 - FIX_0_899), -FIX_0_899),
  _uint64_set_pi16(-FIX_2_562, (FIX_2_053 - FIX_2_562),
                   -FIX_2_562, (FIX_2_053 - FIX_2_562)),
  _uint64_set_pi16((FIX_3_072 - FIX_2_562), -FIX_2_562,
                   (FIX_3_072 - FIX_2_562), -FIX_2_562),
  _uint64_set_pi32((1 << (DESCALE_P1 - 1)), (1 << (DESCALE_P1 - 1))),
  _uint64_set_pi32((1 << (DESCALE_P2 - 1)), (1 << (DESCALE_P2 - 1))),
  _uint64_set_pi16((1 << (PASS1_BITS - 1)), (1 << (PASS1_BITS - 1)),
                   (1 << (PASS1_BITS - 1)), (1 << (PASS1_BITS - 1)))
};

#define PW_F130_F054    get_const_value(index_PW_F130_F054)
#define PW_F054_MF130   get_const_value(index_PW_F054_MF130)
#define PW_MF078_F117   get_const_value(index_PW_MF078_F117)
#define PW_F117_F078    get_const_value(index_PW_F117_F078)
#define PW_MF060_MF089  get_const_value(index_PW_MF060_MF089)
#define PW_MF089_F060   get_const_value(index_PW_MF089_F060)
#define PW_MF050_MF256  get_const_value(index_PW_MF050_MF256)
#define PW_MF256_F050   get_const_value(index_PW_MF256_F050)
#define PD_DESCALE_P1   get_const_value(index_PD_DESCALE_P1)
#define PD_DESCALE_P2   get_const_value(index_PD_DESCALE_P2)
#define PW_DESCALE_P2X  get_const_value(index_PW_DESCALE_P2X)


#define DO_FDCT_COMMON(PASS) { \
  __m64 tmp1312l, tmp1312h, tmp47l, tmp47h, tmp4l, tmp4h, tmp7l, tmp7h; \
  __m64 tmp56l, tmp56h, tmp5l, tmp5h, tmp6l, tmp6h; \
  __m64 out1l, out1h, out2l, out2h, out3l, out3h; \
  __m64 out5l, out5h, out6l, out6h, out7l, out7h; \
  __m64 z34l, z34h, z3l, z3h, z4l, z4h, z3, z4; \
  \
  /* (Original) \
   * z1 = (tmp12 + tmp13) * 0.541196100; \
   * out2 = z1 + tmp13 * 0.765366865; \
   * out6 = z1 + tmp12 * -1.847759065; \
   * \
   * (This implementation) \
   * out2 = tmp13 * (0.541196100 + 0.765366865) + tmp12 * 0.541196100; \
   * out6 = tmp13 * 0.541196100 + tmp12 * (0.541196100 - 1.847759065); \
   */ \
  \
  tmp1312l = _mm_unpacklo_pi16(tmp13, tmp12); \
  tmp1312h = _mm_unpackhi_pi16(tmp13, tmp12); \
  \
  out2l = _mm_madd_pi16(tmp1312l, PW_F130_F054); \
  out2h = _mm_madd_pi16(tmp1312h, PW_F130_F054); \
  out6l = _mm_madd_pi16(tmp1312l, PW_F054_MF130); \
  out6h = _mm_madd_pi16(tmp1312h, PW_F054_MF130); \
  \
  out2l = _mm_add_pi32(out2l, PD_DESCALE_P##PASS); \
  out2h = _mm_add_pi32(out2h, PD_DESCALE_P##PASS); \
  out2l = _mm_srai_pi32(out2l, DESCALE_P##PASS); \
  out2h = _mm_srai_pi32(out2h, DESCALE_P##PASS); \
  \
  out6l = _mm_add_pi32(out6l, PD_DESCALE_P##PASS); \
  out6h = _mm_add_pi32(out6h, PD_DESCALE_P##PASS); \
  out6l = _mm_srai_pi32(out6l, DESCALE_P##PASS); \
  out6h = _mm_srai_pi32(out6h, DESCALE_P##PASS); \
  \
  out2 = _mm_packs_pi32(out2l, out2h); \
  out6 = _mm_packs_pi32(out6l, out6h); \
  \
  /* Odd part */ \
  \
  z3 = _mm_add_pi16(tmp4, tmp6); \
  z4 = _mm_add_pi16(tmp5, tmp7); \
  \
  /* (Original) \
   * z5 = (z3 + z4) * 1.175875602; \
   * z3 = z3 * -1.961570560;  z4 = z4 * -0.390180644; \
   * z3 += z5;  z4 += z5; \
   * \
   * (This implementation) \
   * z3 = z3 * (1.175875602 - 1.961570560) + z4 * 1.175875602; \
   * z4 = z3 * 1.175875602 + z4 * (1.175875602 - 0.390180644); \
   */ \
  \
  z34l = _mm_unpacklo_pi16(z3, z4); \
  z34h = _mm_unpackhi_pi16(z3, z4); \
  z3l = _mm_madd_pi16(z34l, PW_MF078_F117); \
  z3h = _mm_madd_pi16(z34h, PW_MF078_F117); \
  z4l = _mm_madd_pi16(z34l, PW_F117_F078); \
  z4h = _mm_madd_pi16(z34h, PW_F117_F078); \
  \
  /* (Original) \
   * z1 = tmp4 + tmp7;  z2 = tmp5 + tmp6; \
   * tmp4 = tmp4 * 0.298631336;  tmp5 = tmp5 * 2.053119869; \
   * tmp6 = tmp6 * 3.072711026;  tmp7 = tmp7 * 1.501321110; \
   * z1 = z1 * -0.899976223;  z2 = z2 * -2.562915447; \
   * out7 = tmp4 + z1 + z3;  out5 = tmp5 + z2 + z4; \
   * out3 = tmp6 + z2 + z3;  out1 = tmp7 + z1 + z4; \
   * \
   * (This implementation) \
   * tmp4 = tmp4 * (0.298631336 - 0.899976223) + tmp7 * -0.899976223; \
   * tmp5 = tmp5 * (2.053119869 - 2.562915447) + tmp6 * -2.562915447; \
   * tmp6 = tmp5 * -2.562915447 + tmp6 * (3.072711026 - 2.562915447); \
   * tmp7 = tmp4 * -0.899976223 + tmp7 * (1.501321110 - 0.899976223); \
   * out7 = tmp4 + z3;  out5 = tmp5 + z4; \
   * out3 = tmp6 + z3;  out1 = tmp7 + z4; \
   */ \
  \
  tmp47l = _mm_unpacklo_pi16(tmp4, tmp7); \
  tmp47h = _mm_unpackhi_pi16(tmp4, tmp7); \
  \
  tmp4l = _mm_madd_pi16(tmp47l, PW_MF060_MF089); \
  tmp4h = _mm_madd_pi16(tmp47h, PW_MF060_MF089); \
  tmp7l = _mm_madd_pi16(tmp47l, PW_MF089_F060); \
  tmp7h = _mm_madd_pi16(tmp47h, PW_MF089_F060); \
  \
  out7l = _mm_add_pi32(tmp4l, z3l); \
  out7h = _mm_add_pi32(tmp4h, z3h); \
  out1l = _mm_add_pi32(tmp7l, z4l); \
  out1h = _mm_add_pi32(tmp7h, z4h); \
  \
  out7l = _mm_add_pi32(out7l, PD_DESCALE_P##PASS); \
  out7h = _mm_add_pi32(out7h, PD_DESCALE_P##PASS); \
  out7l = _mm_srai_pi32(out7l, DESCALE_P##PASS); \
  out7h = _mm_srai_pi32(out7h, DESCALE_P##PASS); \
  \
  out1l = _mm_add_pi32(out1l, PD_DESCALE_P##PASS); \
  out1h = _mm_add_pi32(out1h, PD_DESCALE_P##PASS); \
  out1l = _mm_srai_pi32(out1l, DESCALE_P##PASS); \
  out1h = _mm_srai_pi32(out1h, DESCALE_P##PASS); \
  \
  out7 = _mm_packs_pi32(out7l, out7h); \
  out1 = _mm_packs_pi32(out1l, out1h); \
  \
  tmp56l = _mm_unpacklo_pi16(tmp5, tmp6); \
  tmp56h = _mm_unpackhi_pi16(tmp5, tmp6); \
  \
  tmp5l = _mm_madd_pi16(tmp56l, PW_MF050_MF256); \
  tmp5h = _mm_madd_pi16(tmp56h, PW_MF050_MF256); \
  tmp6l = _mm_madd_pi16(tmp56l, PW_MF256_F050); \
  tmp6h = _mm_madd_pi16(tmp56h, PW_MF256_F050); \
  \
  out5l = _mm_add_pi32(tmp5l, z4l); \
  out5h = _mm_add_pi32(tmp5h, z4h); \
  out3l = _mm_add_pi32(tmp6l, z3l); \
  out3h = _mm_add_pi32(tmp6h, z3h); \
  \
  out5l = _mm_add_pi32(out5l, PD_DESCALE_P##PASS); \
  out5h = _mm_add_pi32(out5h, PD_DESCALE_P##PASS); \
  out5l = _mm_srai_pi32(out5l, DESCALE_P##PASS); \
  out5h = _mm_srai_pi32(out5h, DESCALE_P##PASS); \
  \
  out3l = _mm_add_pi32(out3l, PD_DESCALE_P##PASS); \
  out3h = _mm_add_pi32(out3h, PD_DESCALE_P##PASS); \
  out3l = _mm_srai_pi32(out3l, DESCALE_P##PASS); \
  out3h = _mm_srai_pi32(out3h, DESCALE_P##PASS); \
  \
  out5 = _mm_packs_pi32(out5l, out5h); \
  out3 = _mm_packs_pi32(out3l, out3h); \
}

#define DO_FDCT_PASS1() { \
  __m64 row0l, row0h, row1l, row1h, row2l, row2h, row3l, row3h; \
  __m64 row01a, row01b, row01c, row01d, row23a, row23b, row23c, row23d; \
  __m64 col0, col1, col2, col3, col4, col5, col6, col7; \
  __m64 tmp10, tmp11; \
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
  /* Even part */ \
  \
  tmp10 = _mm_add_pi16(tmp0, tmp3);           /* tmp10=tmp0+tmp3 */ \
  tmp13 = _mm_sub_pi16(tmp0, tmp3);           /* tmp13=tmp0-tmp3 */ \
  tmp11 = _mm_add_pi16(tmp1, tmp2);           /* tmp11=tmp1+tmp2 */ \
  tmp12 = _mm_sub_pi16(tmp1, tmp2);           /* tmp12=tmp1-tmp2 */ \
  \
  out0 = _mm_add_pi16(tmp10, tmp11);          /* out0=tmp10+tmp11 */ \
  out4 = _mm_sub_pi16(tmp10, tmp11);          /* out4=tmp10-tmp11 */ \
  out0 = _mm_slli_pi16(out0, PASS1_BITS); \
  out4 = _mm_slli_pi16(out4, PASS1_BITS); \
  \
  DO_FDCT_COMMON(1) \
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
  __m64 tmp10, tmp11; \
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
  /* Even part */ \
  \
  tmp10 = _mm_add_pi16(tmp0, tmp3);           /* tmp10=tmp0+tmp3 */ \
  tmp13 = _mm_sub_pi16(tmp0, tmp3);           /* tmp13=tmp0-tmp3 */ \
  tmp11 = _mm_add_pi16(tmp1, tmp2);           /* tmp11=tmp1+tmp2 */ \
  tmp12 = _mm_sub_pi16(tmp1, tmp2);           /* tmp12=tmp1-tmp2 */ \
  \
  out0 = _mm_add_pi16(tmp10, tmp11);          /* out0=tmp10+tmp11 */ \
  out4 = _mm_sub_pi16(tmp10, tmp11);          /* out4=tmp10-tmp11 */ \
  \
  out0 = _mm_add_pi16(out0, PW_DESCALE_P2X); \
  out4 = _mm_add_pi16(out4, PW_DESCALE_P2X); \
  out0 = _mm_srai_pi16(out0, PASS1_BITS); \
  out4 = _mm_srai_pi16(out4, PASS1_BITS); \
  \
  DO_FDCT_COMMON(2) \
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

void jsimd_fdct_islow_mmi(DCTELEM *data)
{
  __m64 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
  __m64 out0, out1, out2, out3, out4, out5, out6, out7;
  __m64 tmp12, tmp13;
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
