/*
 * Loongson MMI optimizations for libjpeg-turbo
 *
 * Copyright (C) 2014-2015, 2018, 2020, D. R. Commander.  All Rights Reserved.
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

/* ACCUATE INTEGER INVERSE DCT */

#include "jsimd_mmi.h"


#define CONST_BITS  13
#define PASS1_BITS  2
#define DESCALE_P1  (CONST_BITS - PASS1_BITS)
#define DESCALE_P2  (CONST_BITS + PASS1_BITS + 3)
#define CENTERJSAMPLE  128

#define FIX_0_298  ((short)2446)  /* FIX(0.298631336) */
#define FIX_0_390  ((short)3196)  /* FIX(0.390180644) */
#define FIX_0_899  ((short)7373)  /* FIX(0.899976223) */
#define FIX_0_541  ((short)4433)  /* FIX(0.541196100) */
#define FIX_0_765  ((short)6270)  /* FIX(0.765366865) */
#define FIX_1_175  ((short)9633)  /* FIX(1.175875602) */
#define FIX_1_501  ((short)12299) /* FIX(1.501321110) */
#define FIX_1_847  ((short)15137) /* FIX(1.847759065) */
#define FIX_1_961  ((short)16069) /* FIX(1.961570560) */
#define FIX_2_053  ((short)16819) /* FIX(2.053119869) */
#define FIX_2_562  ((short)20995) /* FIX(2.562915447) */
#define FIX_3_072  ((short)25172) /* FIX(3.072711026) */

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
  index_PB_CENTERJSAMP
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
  _uint64_set_pi8(CENTERJSAMPLE, CENTERJSAMPLE, CENTERJSAMPLE, CENTERJSAMPLE,
                  CENTERJSAMPLE, CENTERJSAMPLE, CENTERJSAMPLE, CENTERJSAMPLE)
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
#define PB_CENTERJSAMP  get_const_value(index_PB_CENTERJSAMP)


#define test_m32_zero(mm32)  (!(*(uint32_t *)&mm32))
#define test_m64_zero(mm64)  (!(*(uint64_t *)&mm64))


#define DO_IDCT_COMMON(PASS) { \
  __m64 tmp0_3l, tmp0_3h, tmp1_2l, tmp1_2h; \
  __m64 tmp0l, tmp0h, tmp1l, tmp1h, tmp2l, tmp2h, tmp3l, tmp3h; \
  __m64 z34l, z34h, z3l, z3h, z4l, z4h, z3, z4; \
  __m64 out0l, out0h, out1l, out1h, out2l, out2h, out3l, out3h; \
  __m64 out4l, out4h, out5l, out5h, out6l, out6h, out7l, out7h; \
  \
  z3 = _mm_add_pi16(tmp0, tmp2); \
  z4 = _mm_add_pi16(tmp1, tmp3); \
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
   * z1 = tmp0 + tmp3;  z2 = tmp1 + tmp2; \
   * tmp0 = tmp0 * 0.298631336;  tmp1 = tmp1 * 2.053119869; \
   * tmp2 = tmp2 * 3.072711026;  tmp3 = tmp3 * 1.501321110; \
   * z1 = z1 * -0.899976223;  z2 = z2 * -2.562915447; \
   * tmp0 += z1 + z3;  tmp1 += z2 + z4; \
   * tmp2 += z2 + z3;  tmp3 += z1 + z4; \
   * \
   * (This implementation) \
   * tmp0 = tmp0 * (0.298631336 - 0.899976223) + tmp3 * -0.899976223; \
   * tmp1 = tmp1 * (2.053119869 - 2.562915447) + tmp2 * -2.562915447; \
   * tmp2 = tmp1 * -2.562915447 + tmp2 * (3.072711026 - 2.562915447); \
   * tmp3 = tmp0 * -0.899976223 + tmp3 * (1.501321110 - 0.899976223); \
   * tmp0 += z3;  tmp1 += z4; \
   * tmp2 += z3;  tmp3 += z4; \
   */ \
  \
  tmp0_3l = _mm_unpacklo_pi16(tmp0, tmp3); \
  tmp0_3h = _mm_unpackhi_pi16(tmp0, tmp3); \
  \
  tmp0l = _mm_madd_pi16(tmp0_3l, PW_MF060_MF089); \
  tmp0h = _mm_madd_pi16(tmp0_3h, PW_MF060_MF089); \
  tmp3l = _mm_madd_pi16(tmp0_3l, PW_MF089_F060); \
  tmp3h = _mm_madd_pi16(tmp0_3h, PW_MF089_F060); \
  \
  tmp0l = _mm_add_pi32(tmp0l, z3l); \
  tmp0h = _mm_add_pi32(tmp0h, z3h); \
  tmp3l = _mm_add_pi32(tmp3l, z4l); \
  tmp3h = _mm_add_pi32(tmp3h, z4h); \
  \
  tmp1_2l = _mm_unpacklo_pi16(tmp1, tmp2); \
  tmp1_2h = _mm_unpackhi_pi16(tmp1, tmp2); \
  \
  tmp1l = _mm_madd_pi16(tmp1_2l, PW_MF050_MF256); \
  tmp1h = _mm_madd_pi16(tmp1_2h, PW_MF050_MF256); \
  tmp2l = _mm_madd_pi16(tmp1_2l, PW_MF256_F050); \
  tmp2h = _mm_madd_pi16(tmp1_2h, PW_MF256_F050); \
  \
  tmp1l = _mm_add_pi32(tmp1l, z4l); \
  tmp1h = _mm_add_pi32(tmp1h, z4h); \
  tmp2l = _mm_add_pi32(tmp2l, z3l); \
  tmp2h = _mm_add_pi32(tmp2h, z3h); \
  \
  /* Final output stage */ \
  \
  out0l = _mm_add_pi32(tmp10l, tmp3l); \
  out0h = _mm_add_pi32(tmp10h, tmp3h); \
  out7l = _mm_sub_pi32(tmp10l, tmp3l); \
  out7h = _mm_sub_pi32(tmp10h, tmp3h); \
  \
  out0l = _mm_add_pi32(out0l, PD_DESCALE_P##PASS); \
  out0h = _mm_add_pi32(out0h, PD_DESCALE_P##PASS); \
  out0l = _mm_srai_pi32(out0l, DESCALE_P##PASS); \
  out0h = _mm_srai_pi32(out0h, DESCALE_P##PASS); \
  \
  out7l = _mm_add_pi32(out7l, PD_DESCALE_P##PASS); \
  out7h = _mm_add_pi32(out7h, PD_DESCALE_P##PASS); \
  out7l = _mm_srai_pi32(out7l, DESCALE_P##PASS); \
  out7h = _mm_srai_pi32(out7h, DESCALE_P##PASS); \
  \
  out0 = _mm_packs_pi32(out0l, out0h); \
  out7 = _mm_packs_pi32(out7l, out7h); \
  \
  out1l = _mm_add_pi32(tmp11l, tmp2l); \
  out1h = _mm_add_pi32(tmp11h, tmp2h); \
  out6l = _mm_sub_pi32(tmp11l, tmp2l); \
  out6h = _mm_sub_pi32(tmp11h, tmp2h); \
  \
  out1l = _mm_add_pi32(out1l, PD_DESCALE_P##PASS); \
  out1h = _mm_add_pi32(out1h, PD_DESCALE_P##PASS); \
  out1l = _mm_srai_pi32(out1l, DESCALE_P##PASS); \
  out1h = _mm_srai_pi32(out1h, DESCALE_P##PASS); \
  \
  out6l = _mm_add_pi32(out6l, PD_DESCALE_P##PASS); \
  out6h = _mm_add_pi32(out6h, PD_DESCALE_P##PASS); \
  out6l = _mm_srai_pi32(out6l, DESCALE_P##PASS); \
  out6h = _mm_srai_pi32(out6h, DESCALE_P##PASS); \
  \
  out1 = _mm_packs_pi32(out1l, out1h); \
  out6 = _mm_packs_pi32(out6l, out6h); \
  \
  out2l = _mm_add_pi32(tmp12l, tmp1l); \
  out2h = _mm_add_pi32(tmp12h, tmp1h); \
  out5l = _mm_sub_pi32(tmp12l, tmp1l); \
  out5h = _mm_sub_pi32(tmp12h, tmp1h); \
  \
  out2l = _mm_add_pi32(out2l, PD_DESCALE_P##PASS); \
  out2h = _mm_add_pi32(out2h, PD_DESCALE_P##PASS); \
  out2l = _mm_srai_pi32(out2l, DESCALE_P##PASS); \
  out2h = _mm_srai_pi32(out2h, DESCALE_P##PASS); \
  \
  out5l = _mm_add_pi32(out5l, PD_DESCALE_P##PASS); \
  out5h = _mm_add_pi32(out5h, PD_DESCALE_P##PASS); \
  out5l = _mm_srai_pi32(out5l, DESCALE_P##PASS); \
  out5h = _mm_srai_pi32(out5h, DESCALE_P##PASS); \
  \
  out2 = _mm_packs_pi32(out2l, out2h); \
  out5 = _mm_packs_pi32(out5l, out5h); \
  \
  out3l = _mm_add_pi32(tmp13l, tmp0l); \
  out3h = _mm_add_pi32(tmp13h, tmp0h); \
  \
  out4l = _mm_sub_pi32(tmp13l, tmp0l); \
  out4h = _mm_sub_pi32(tmp13h, tmp0h); \
  \
  out3l = _mm_add_pi32(out3l, PD_DESCALE_P##PASS); \
  out3h = _mm_add_pi32(out3h, PD_DESCALE_P##PASS); \
  out3l = _mm_srai_pi32(out3l, DESCALE_P##PASS); \
  out3h = _mm_srai_pi32(out3h, DESCALE_P##PASS); \
  \
  out4l = _mm_add_pi32(out4l, PD_DESCALE_P##PASS); \
  out4h = _mm_add_pi32(out4h, PD_DESCALE_P##PASS); \
  out4l = _mm_srai_pi32(out4l, DESCALE_P##PASS); \
  out4h = _mm_srai_pi32(out4h, DESCALE_P##PASS); \
  \
  out3 = _mm_packs_pi32(out3l, out3h); \
  out4 = _mm_packs_pi32(out4l, out4h); \
}

#define DO_IDCT_PASS1(iter) { \
  __m64 col0l, col1l, col2l, col3l, col4l, col5l, col6l, col7l; \
  __m64 quant0l, quant1l, quant2l, quant3l; \
  __m64 quant4l, quant5l, quant6l, quant7l; \
  __m64 z23, z2, z3, z23l, z23h; \
  __m64 row01a, row01b, row01c, row01d, row23a, row23b, row23c, row23d; \
  __m64 row0l, row0h, row1l, row1h, row2l, row2h, row3l, row3h; \
  __m64 tmp0l, tmp0h, tmp1l, tmp1h, tmp2l, tmp2h, tmp3l, tmp3h; \
  __m64 tmp10l, tmp10h, tmp11l, tmp11h, tmp12l, tmp12h, tmp13l, tmp13h; \
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
      dcval = _mm_mullo_pi16(col0l, quant0l); \
      dcval = _mm_slli_pi16(dcval, PASS1_BITS);  /* dcval=(00 10 20 30) */ \
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
  /* Even part \
   * \
   * (Original) \
   * z1 = (z2 + z3) * 0.541196100; \
   * tmp2 = z1 + z3 * -1.847759065; \
   * tmp3 = z1 + z2 * 0.765366865; \
   * \
   * (This implementation) \
   * tmp2 = z2 * 0.541196100 + z3 * (0.541196100 - 1.847759065); \
   * tmp3 = z2 * (0.541196100 + 0.765366865) + z3 * 0.541196100; \
   */ \
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
  z2 = _mm_mullo_pi16(col2l, quant2l); \
  z3 = _mm_mullo_pi16(col6l, quant6l); \
  \
  z23l = _mm_unpacklo_pi16(z2, z3); \
  z23h = _mm_unpackhi_pi16(z2, z3); \
  tmp3l = _mm_madd_pi16(z23l, PW_F130_F054); \
  tmp3h = _mm_madd_pi16(z23h, PW_F130_F054); \
  tmp2l = _mm_madd_pi16(z23l, PW_F054_MF130); \
  tmp2h = _mm_madd_pi16(z23h, PW_F054_MF130); \
  \
  z2 = _mm_mullo_pi16(col0l, quant0l); \
  z3 = _mm_mullo_pi16(col4l, quant4l); \
  \
  z23 = _mm_add_pi16(z2, z3); \
  tmp0l = _mm_loadlo_pi16_f(z23); \
  tmp0h = _mm_loadhi_pi16_f(z23); \
  tmp0l = _mm_srai_pi32(tmp0l, (16 - CONST_BITS)); \
  tmp0h = _mm_srai_pi32(tmp0h, (16 - CONST_BITS)); \
  \
  tmp10l = _mm_add_pi32(tmp0l, tmp3l); \
  tmp10h = _mm_add_pi32(tmp0h, tmp3h); \
  tmp13l = _mm_sub_pi32(tmp0l, tmp3l); \
  tmp13h = _mm_sub_pi32(tmp0h, tmp3h); \
  \
  z23 = _mm_sub_pi16(z2, z3); \
  tmp1l = _mm_loadlo_pi16_f(z23); \
  tmp1h = _mm_loadhi_pi16_f(z23); \
  tmp1l = _mm_srai_pi32(tmp1l, (16 - CONST_BITS)); \
  tmp1h = _mm_srai_pi32(tmp1h, (16 - CONST_BITS)); \
  \
  tmp11l = _mm_add_pi32(tmp1l, tmp2l); \
  tmp11h = _mm_add_pi32(tmp1h, tmp2h); \
  tmp12l = _mm_sub_pi32(tmp1l, tmp2l); \
  tmp12h = _mm_sub_pi32(tmp1h, tmp2h); \
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
  tmp0 = _mm_mullo_pi16(col7l, quant7l); \
  tmp1 = _mm_mullo_pi16(col5l, quant5l); \
  tmp2 = _mm_mullo_pi16(col3l, quant3l); \
  tmp3 = _mm_mullo_pi16(col1l, quant1l); \
  \
  DO_IDCT_COMMON(1) \
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
  __m64 z23, z23l, z23h; \
  __m64 col0123a, col0123b, col0123c, col0123d; \
  __m64 col01l, col01h, col23l, col23h, row06, row17, row24, row35; \
  __m64 col0, col1, col2, col3; \
  __m64 tmp0l, tmp0h, tmp1l, tmp1h, tmp2l, tmp2h, tmp3l, tmp3h; \
  __m64 tmp10l, tmp10h, tmp11l, tmp11h, tmp12l, tmp12h, tmp13l, tmp13h; \
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
  /* Even part \
   * \
   * (Original) \
   * z1 = (z2 + z3) * 0.541196100; \
   * tmp2 = z1 + z3 * -1.847759065; \
   * tmp3 = z1 + z2 * 0.765366865; \
   * \
   * (This implementation) \
   * tmp2 = z2 * 0.541196100 + z3 * (0.541196100 - 1.847759065); \
   * tmp3 = z2 * (0.541196100 + 0.765366865) + z3 * 0.541196100; \
   */ \
  \
  z23l = _mm_unpacklo_pi16(row2l, row6l); \
  z23h = _mm_unpackhi_pi16(row2l, row6l); \
  \
  tmp3l = _mm_madd_pi16(z23l, PW_F130_F054); \
  tmp3h = _mm_madd_pi16(z23h, PW_F130_F054); \
  tmp2l = _mm_madd_pi16(z23l, PW_F054_MF130); \
  tmp2h = _mm_madd_pi16(z23h, PW_F054_MF130); \
  \
  z23 = _mm_add_pi16(row0l, row4l); \
  tmp0l = _mm_loadlo_pi16_f(z23); \
  tmp0h = _mm_loadhi_pi16_f(z23); \
  tmp0l = _mm_srai_pi32(tmp0l, (16 - CONST_BITS)); \
  tmp0h = _mm_srai_pi32(tmp0h, (16 - CONST_BITS)); \
  \
  tmp10l = _mm_add_pi32(tmp0l, tmp3l); \
  tmp10h = _mm_add_pi32(tmp0h, tmp3h); \
  tmp13l = _mm_sub_pi32(tmp0l, tmp3l); \
  tmp13h = _mm_sub_pi32(tmp0h, tmp3h); \
  \
  z23 = _mm_sub_pi16(row0l, row4l); \
  tmp1l = _mm_loadlo_pi16_f(z23); \
  tmp1h = _mm_loadhi_pi16_f(z23); \
  tmp1l = _mm_srai_pi32(tmp1l, (16 - CONST_BITS)); \
  tmp1h = _mm_srai_pi32(tmp1h, (16 - CONST_BITS)); \
  \
  tmp11l = _mm_add_pi32(tmp1l, tmp2l); \
  tmp11h = _mm_add_pi32(tmp1h, tmp2h); \
  tmp12l = _mm_sub_pi32(tmp1l, tmp2l); \
  tmp12h = _mm_sub_pi32(tmp1h, tmp2h); \
  \
  /* Odd part */ \
  \
  tmp0 = row7l; \
  tmp1 = row5l; \
  tmp2 = row3l; \
  tmp3 = row1l; \
  \
  DO_IDCT_COMMON(2) \
  \
  /* out0=(00 01 02 03), out1=(10 11 12 13) */ \
  /* out2=(20 21 22 23), out3=(30 31 32 33) */ \
  /* out4=(40 41 42 43), out5=(50 51 52 53) */ \
  /* out6=(60 61 62 63), out7=(70 71 72 73) */ \
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

void jsimd_idct_islow_mmi(void *dct_table, JCOEFPTR coef_block,
                          JSAMPARRAY output_buf, JDIMENSION output_col)
{
  __m64 tmp0, tmp1, tmp2, tmp3;
  __m64 out0, out1, out2, out3, out4, out5, out6, out7;
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
