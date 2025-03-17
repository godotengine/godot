/*
 * AltiVec optimizations for libjpeg-turbo
 *
 * Copyright (C) 2014-2015, 2020, 2024, D. R. Commander.  All Rights Reserved.
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

/* ACCURATE INTEGER INVERSE DCT */

#include "jsimd_altivec.h"


#define F_0_298  2446   /* FIX(0.298631336) */
#define F_0_390  3196   /* FIX(0.390180644) */
#define F_0_541  4433   /* FIX(0.541196100) */
#define F_0_765  6270   /* FIX(0.765366865) */
#define F_0_899  7373   /* FIX(0.899976223) */
#define F_1_175  9633   /* FIX(1.175875602) */
#define F_1_501  12299  /* FIX(1.501321110) */
#define F_1_847  15137  /* FIX(1.847759065) */
#define F_1_961  16069  /* FIX(1.961570560) */
#define F_2_053  16819  /* FIX(2.053119869) */
#define F_2_562  20995  /* FIX(2.562915447) */
#define F_3_072  25172  /* FIX(3.072711026) */

#define CONST_BITS  13
#define PASS1_BITS  2
#define DESCALE_P1  (CONST_BITS - PASS1_BITS)
#define DESCALE_P2  (CONST_BITS + PASS1_BITS + 3)


#define DO_IDCT(in, PASS) { \
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
  in##26l = vec_mergeh(in##2, in##6); \
  in##26h = vec_mergel(in##2, in##6); \
  \
  tmp3l = vec_msums(in##26l, pw_f130_f054, pd_zero); \
  tmp3h = vec_msums(in##26h, pw_f130_f054, pd_zero); \
  tmp2l = vec_msums(in##26l, pw_f054_mf130, pd_zero); \
  tmp2h = vec_msums(in##26h, pw_f054_mf130, pd_zero); \
  \
  tmp0 = vec_add(in##0, in##4); \
  tmp1 = vec_sub(in##0, in##4); \
  \
  tmp0l = vec_unpackh(tmp0); \
  tmp0h = vec_unpackl(tmp0); \
  tmp0l = vec_sl(tmp0l, const_bits); \
  tmp0h = vec_sl(tmp0h, const_bits); \
  tmp0l = vec_add(tmp0l, pd_descale_p##PASS); \
  tmp0h = vec_add(tmp0h, pd_descale_p##PASS); \
  \
  tmp10l = vec_add(tmp0l, tmp3l); \
  tmp10h = vec_add(tmp0h, tmp3h); \
  tmp13l = vec_sub(tmp0l, tmp3l); \
  tmp13h = vec_sub(tmp0h, tmp3h); \
  \
  tmp1l = vec_unpackh(tmp1); \
  tmp1h = vec_unpackl(tmp1); \
  tmp1l = vec_sl(tmp1l, const_bits); \
  tmp1h = vec_sl(tmp1h, const_bits); \
  tmp1l = vec_add(tmp1l, pd_descale_p##PASS); \
  tmp1h = vec_add(tmp1h, pd_descale_p##PASS); \
  \
  tmp11l = vec_add(tmp1l, tmp2l); \
  tmp11h = vec_add(tmp1h, tmp2h); \
  tmp12l = vec_sub(tmp1l, tmp2l); \
  tmp12h = vec_sub(tmp1h, tmp2h); \
  \
  /* Odd part */ \
  \
  z3 = vec_add(in##3, in##7); \
  z4 = vec_add(in##1, in##5); \
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
  z34l = vec_mergeh(z3, z4); \
  z34h = vec_mergel(z3, z4); \
  \
  z3l = vec_msums(z34l, pw_mf078_f117, pd_zero); \
  z3h = vec_msums(z34h, pw_mf078_f117, pd_zero); \
  z4l = vec_msums(z34l, pw_f117_f078, pd_zero); \
  z4h = vec_msums(z34h, pw_f117_f078, pd_zero); \
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
  in##71l = vec_mergeh(in##7, in##1); \
  in##71h = vec_mergel(in##7, in##1); \
  \
  tmp0l = vec_msums(in##71l, pw_mf060_mf089, z3l); \
  tmp0h = vec_msums(in##71h, pw_mf060_mf089, z3h); \
  tmp3l = vec_msums(in##71l, pw_mf089_f060, z4l); \
  tmp3h = vec_msums(in##71h, pw_mf089_f060, z4h); \
  \
  in##53l = vec_mergeh(in##5, in##3); \
  in##53h = vec_mergel(in##5, in##3); \
  \
  tmp1l = vec_msums(in##53l, pw_mf050_mf256, z4l); \
  tmp1h = vec_msums(in##53h, pw_mf050_mf256, z4h); \
  tmp2l = vec_msums(in##53l, pw_mf256_f050, z3l); \
  tmp2h = vec_msums(in##53h, pw_mf256_f050, z3h); \
  \
  /* Final output stage */ \
  \
  out0l = vec_add(tmp10l, tmp3l); \
  out0h = vec_add(tmp10h, tmp3h); \
  out7l = vec_sub(tmp10l, tmp3l); \
  out7h = vec_sub(tmp10h, tmp3h); \
  \
  out0l = vec_sra(out0l, descale_p##PASS); \
  out0h = vec_sra(out0h, descale_p##PASS); \
  out7l = vec_sra(out7l, descale_p##PASS); \
  out7h = vec_sra(out7h, descale_p##PASS); \
  \
  out0 = vec_pack(out0l, out0h); \
  out7 = vec_pack(out7l, out7h); \
  \
  out1l = vec_add(tmp11l, tmp2l); \
  out1h = vec_add(tmp11h, tmp2h); \
  out6l = vec_sub(tmp11l, tmp2l); \
  out6h = vec_sub(tmp11h, tmp2h); \
  \
  out1l = vec_sra(out1l, descale_p##PASS); \
  out1h = vec_sra(out1h, descale_p##PASS); \
  out6l = vec_sra(out6l, descale_p##PASS); \
  out6h = vec_sra(out6h, descale_p##PASS); \
  \
  out1 = vec_pack(out1l, out1h); \
  out6 = vec_pack(out6l, out6h); \
  \
  out2l = vec_add(tmp12l, tmp1l); \
  out2h = vec_add(tmp12h, tmp1h); \
  out5l = vec_sub(tmp12l, tmp1l); \
  out5h = vec_sub(tmp12h, tmp1h); \
  \
  out2l = vec_sra(out2l, descale_p##PASS); \
  out2h = vec_sra(out2h, descale_p##PASS); \
  out5l = vec_sra(out5l, descale_p##PASS); \
  out5h = vec_sra(out5h, descale_p##PASS); \
  \
  out2 = vec_pack(out2l, out2h); \
  out5 = vec_pack(out5l, out5h); \
  \
  out3l = vec_add(tmp13l, tmp0l); \
  out3h = vec_add(tmp13h, tmp0h); \
  out4l = vec_sub(tmp13l, tmp0l); \
  out4h = vec_sub(tmp13h, tmp0h); \
  \
  out3l = vec_sra(out3l, descale_p##PASS); \
  out3h = vec_sra(out3h, descale_p##PASS); \
  out4l = vec_sra(out4l, descale_p##PASS); \
  out4h = vec_sra(out4h, descale_p##PASS); \
  \
  out3 = vec_pack(out3l, out3h); \
  out4 = vec_pack(out4l, out4h); \
}


void jsimd_idct_islow_altivec(void *dct_table_, JCOEFPTR coef_block,
                              JSAMPARRAY output_buf, JDIMENSION output_col)
{
  short *dct_table = (short *)dct_table_;
  int *outptr;

  __vector short row0, row1, row2, row3, row4, row5, row6, row7,
    col0, col1, col2, col3, col4, col5, col6, col7,
    quant0, quant1, quant2, quant3, quant4, quant5, quant6, quant7,
    tmp0, tmp1, tmp2, tmp3, z3, z4,
    z34l, z34h, col71l, col71h, col26l, col26h, col53l, col53h,
    row71l, row71h, row26l, row26h, row53l, row53h,
    out0, out1, out2, out3, out4, out5, out6, out7;
  __vector int tmp0l, tmp0h, tmp1l, tmp1h, tmp2l, tmp2h, tmp3l, tmp3h,
    tmp10l, tmp10h, tmp11l, tmp11h, tmp12l, tmp12h, tmp13l, tmp13h,
    z3l, z3h, z4l, z4h,
    out0l, out0h, out1l, out1h, out2l, out2h, out3l, out3h, out4l, out4h,
    out5l, out5h, out6l, out6h, out7l, out7h;
  __vector signed char outb;

  /* Constants */
  __vector short pw_zero = { __8X(0) },
    pw_f130_f054 = { __4X2(F_0_541 + F_0_765, F_0_541) },
    pw_f054_mf130 = { __4X2(F_0_541, F_0_541 - F_1_847) },
    pw_mf078_f117 = { __4X2(F_1_175 - F_1_961, F_1_175) },
    pw_f117_f078 = { __4X2(F_1_175, F_1_175 - F_0_390) },
    pw_mf060_mf089 = { __4X2(F_0_298 - F_0_899, -F_0_899) },
    pw_mf089_f060 = { __4X2(-F_0_899, F_1_501 - F_0_899) },
    pw_mf050_mf256 = { __4X2(F_2_053 - F_2_562, -F_2_562) },
    pw_mf256_f050 = { __4X2(-F_2_562, F_3_072 - F_2_562) };
  __vector unsigned short pass1_bits = { __8X(PASS1_BITS) };
  __vector int pd_zero = { __4X(0) },
    pd_descale_p1 = { __4X(1 << (DESCALE_P1 - 1)) },
    pd_descale_p2 = { __4X(1 << (DESCALE_P2 - 1)) };
  __vector unsigned int descale_p1 = { __4X(DESCALE_P1) },
    descale_p2 = { __4X(DESCALE_P2) },
    const_bits = { __4X(CONST_BITS) };
  __vector signed char pb_centerjsamp = { __16X((signed char)CENTERJSAMPLE) };

  /* Pass 1: process columns */

  col0 = vec_ld(0, coef_block);
  col1 = vec_ld(16, coef_block);
  col2 = vec_ld(32, coef_block);
  col3 = vec_ld(48, coef_block);
  col4 = vec_ld(64, coef_block);
  col5 = vec_ld(80, coef_block);
  col6 = vec_ld(96, coef_block);
  col7 = vec_ld(112, coef_block);

  tmp1 = vec_or(col1, col2);
  tmp2 = vec_or(col3, col4);
  tmp1 = vec_or(tmp1, tmp2);
  tmp3 = vec_or(col5, col6);
  tmp3 = vec_or(tmp3, col7);
  tmp1 = vec_or(tmp1, tmp3);

  quant0 = vec_ld(0, dct_table);
  col0 = vec_mladd(col0, quant0, pw_zero);

  if (vec_all_eq(tmp1, pw_zero)) {
    /* AC terms all zero */

    col0 = vec_sl(col0, pass1_bits);

    row0 = vec_splat(col0, 0);
    row1 = vec_splat(col0, 1);
    row2 = vec_splat(col0, 2);
    row3 = vec_splat(col0, 3);
    row4 = vec_splat(col0, 4);
    row5 = vec_splat(col0, 5);
    row6 = vec_splat(col0, 6);
    row7 = vec_splat(col0, 7);

  } else {

    quant1 = vec_ld(16, dct_table);
    quant2 = vec_ld(32, dct_table);
    quant3 = vec_ld(48, dct_table);
    quant4 = vec_ld(64, dct_table);
    quant5 = vec_ld(80, dct_table);
    quant6 = vec_ld(96, dct_table);
    quant7 = vec_ld(112, dct_table);

    col1 = vec_mladd(col1, quant1, pw_zero);
    col2 = vec_mladd(col2, quant2, pw_zero);
    col3 = vec_mladd(col3, quant3, pw_zero);
    col4 = vec_mladd(col4, quant4, pw_zero);
    col5 = vec_mladd(col5, quant5, pw_zero);
    col6 = vec_mladd(col6, quant6, pw_zero);
    col7 = vec_mladd(col7, quant7, pw_zero);

    DO_IDCT(col, 1);

    TRANSPOSE(out, row);
  }

  /* Pass 2: process rows */

  DO_IDCT(row, 2);

  TRANSPOSE(out, col);

  outb = vec_packs(col0, col0);
  outb = vec_add(outb, pb_centerjsamp);
  outptr = (int *)(output_buf[0] + output_col);
  vec_ste((__vector int)outb, 0, outptr);
  vec_ste((__vector int)outb, 4, outptr);

  outb = vec_packs(col1, col1);
  outb = vec_add(outb, pb_centerjsamp);
  outptr = (int *)(output_buf[1] + output_col);
  vec_ste((__vector int)outb, 0, outptr);
  vec_ste((__vector int)outb, 4, outptr);

  outb = vec_packs(col2, col2);
  outb = vec_add(outb, pb_centerjsamp);
  outptr = (int *)(output_buf[2] + output_col);
  vec_ste((__vector int)outb, 0, outptr);
  vec_ste((__vector int)outb, 4, outptr);

  outb = vec_packs(col3, col3);
  outb = vec_add(outb, pb_centerjsamp);
  outptr = (int *)(output_buf[3] + output_col);
  vec_ste((__vector int)outb, 0, outptr);
  vec_ste((__vector int)outb, 4, outptr);

  outb = vec_packs(col4, col4);
  outb = vec_add(outb, pb_centerjsamp);
  outptr = (int *)(output_buf[4] + output_col);
  vec_ste((__vector int)outb, 0, outptr);
  vec_ste((__vector int)outb, 4, outptr);

  outb = vec_packs(col5, col5);
  outb = vec_add(outb, pb_centerjsamp);
  outptr = (int *)(output_buf[5] + output_col);
  vec_ste((__vector int)outb, 0, outptr);
  vec_ste((__vector int)outb, 4, outptr);

  outb = vec_packs(col6, col6);
  outb = vec_add(outb, pb_centerjsamp);
  outptr = (int *)(output_buf[6] + output_col);
  vec_ste((__vector int)outb, 0, outptr);
  vec_ste((__vector int)outb, 4, outptr);

  outb = vec_packs(col7, col7);
  outb = vec_add(outb, pb_centerjsamp);
  outptr = (int *)(output_buf[7] + output_col);
  vec_ste((__vector int)outb, 0, outptr);
  vec_ste((__vector int)outb, 4, outptr);
}
