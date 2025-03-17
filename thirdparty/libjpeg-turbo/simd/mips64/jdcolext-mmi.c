/*
 * Loongson MMI optimizations for libjpeg-turbo
 *
 * Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
 * Copyright (C) 2015, 2019, D. R. Commander.  All Rights Reserved.
 * Copyright (C) 2016-2018, Loongson Technology Corporation Limited, BeiJing.
 *                          All Rights Reserved.
 * Authors:  ZhuChen     <zhuchen@loongson.cn>
 *           SunZhangzhi <sunzhangzhi-cq@loongson.cn>
 *           CaiWanwei   <caiwanwei@loongson.cn>
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

/* This file is included by jdcolor-mmi.c */


#if RGB_RED == 0
#define mmA  re
#define mmB  ro
#elif RGB_GREEN == 0
#define mmA  ge
#define mmB  go
#elif RGB_BLUE == 0
#define mmA  be
#define mmB  bo
#else
#define mmA  xe
#define mmB  xo
#endif

#if RGB_RED == 1
#define mmC  re
#define mmD  ro
#elif RGB_GREEN == 1
#define mmC  ge
#define mmD  go
#elif RGB_BLUE == 1
#define mmC  be
#define mmD  bo
#else
#define mmC  xe
#define mmD  xo
#endif

#if RGB_RED == 2
#define mmE  re
#define mmF  ro
#elif RGB_GREEN == 2
#define mmE  ge
#define mmF  go
#elif RGB_BLUE == 2
#define mmE  be
#define mmF  bo
#else
#define mmE  xe
#define mmF  xo
#endif

#if RGB_RED == 3
#define mmG  re
#define mmH  ro
#elif RGB_GREEN == 3
#define mmG  ge
#define mmH  go
#elif RGB_BLUE == 3
#define mmG  be
#define mmH  bo
#else
#define mmG  xe
#define mmH  xo
#endif


void jsimd_ycc_rgb_convert_mmi(JDIMENSION out_width, JSAMPIMAGE input_buf,
                               JDIMENSION input_row, JSAMPARRAY output_buf,
                               int num_rows)
{
  JSAMPROW outptr, inptr0, inptr1, inptr2;
  int num_cols, col;
  __m64 ye, yo, y, cbe, cbe2, cbo, cbo2, cb, cre, cre2, cro, cro2, cr;
  __m64 re, ro, gle, ghe, ge, glo, gho, go, be, bo, xe = 0.0, xo = 0.0;
  __m64 decenter, mask;

  while (--num_rows >= 0) {
    inptr0 = input_buf[0][input_row];
    inptr1 = input_buf[1][input_row];
    inptr2 = input_buf[2][input_row];
    input_row++;
    outptr = *output_buf++;

    for (num_cols = out_width; num_cols > 0; num_cols -= 8,
         inptr0 += 8, inptr1 += 8, inptr2 += 8) {

      cb = _mm_load_si64((__m64 *)inptr1);
      cr = _mm_load_si64((__m64 *)inptr2);
      y = _mm_load_si64((__m64 *)inptr0);

      mask = decenter = 0.0;
      mask = _mm_cmpeq_pi16(mask, mask);
      decenter = _mm_cmpeq_pi16(decenter, decenter);
      mask = _mm_srli_pi16(mask, BYTE_BIT);   /* {0xFF 0x00 0xFF 0x00 ..} */
      decenter = _mm_slli_pi16(decenter, 7);  /* {0xFF80 0xFF80 0xFF80 0xFF80} */

      cbe = _mm_and_si64(mask, cb);           /* Cb(0246) */
      cbo = _mm_srli_pi16(cb, BYTE_BIT);      /* Cb(1357) */
      cre = _mm_and_si64(mask, cr);           /* Cr(0246) */
      cro = _mm_srli_pi16(cr, BYTE_BIT);      /* Cr(1357) */
      cbe = _mm_add_pi16(cbe, decenter);
      cbo = _mm_add_pi16(cbo, decenter);
      cre = _mm_add_pi16(cre, decenter);
      cro = _mm_add_pi16(cro, decenter);

      /* (Original)
       * R = Y                + 1.40200 * Cr
       * G = Y - 0.34414 * Cb - 0.71414 * Cr
       * B = Y + 1.77200 * Cb
       *
       * (This implementation)
       * R = Y                + 0.40200 * Cr + Cr
       * G = Y - 0.34414 * Cb + 0.28586 * Cr - Cr
       * B = Y - 0.22800 * Cb + Cb + Cb
       */

      cbe2 = _mm_add_pi16(cbe, cbe);          /* 2*CbE */
      cbo2 = _mm_add_pi16(cbo, cbo);          /* 2*CbO */
      cre2 = _mm_add_pi16(cre, cre);          /* 2*CrE */
      cro2 = _mm_add_pi16(cro, cro);          /* 2*CrO */

      be = _mm_mulhi_pi16(cbe2, PW_MF0228);   /* (2*CbE * -FIX(0.22800) */
      bo = _mm_mulhi_pi16(cbo2, PW_MF0228);   /* (2*CbO * -FIX(0.22800) */
      re = _mm_mulhi_pi16(cre2, PW_F0402);    /* (2*CrE * FIX(0.40200)) */
      ro = _mm_mulhi_pi16(cro2, PW_F0402);    /* (2*CrO * FIX(0.40200)) */

      be = _mm_add_pi16(be, PW_ONE);
      bo = _mm_add_pi16(bo, PW_ONE);
      be = _mm_srai_pi16(be, 1);              /* (CbE * -FIX(0.22800)) */
      bo = _mm_srai_pi16(bo, 1);              /* (CbO * -FIX(0.22800)) */
      re = _mm_add_pi16(re, PW_ONE);
      ro = _mm_add_pi16(ro, PW_ONE);
      re = _mm_srai_pi16(re, 1);              /* (CrE * FIX(0.40200)) */
      ro = _mm_srai_pi16(ro, 1);              /* (CrO * FIX(0.40200)) */

      be = _mm_add_pi16(be, cbe);
      bo = _mm_add_pi16(bo, cbo);
      be = _mm_add_pi16(be, cbe);             /* (CbE * FIX(1.77200))=(B-Y)E */
      bo = _mm_add_pi16(bo, cbo);             /* (CbO * FIX(1.77200))=(B-Y)O */
      re = _mm_add_pi16(re, cre);             /* (CrE * FIX(1.40200))=(R-Y)E */
      ro = _mm_add_pi16(ro, cro);             /* (CrO * FIX(1.40200))=(R-Y)O */

      gle = _mm_unpacklo_pi16(cbe, cre);
      ghe = _mm_unpackhi_pi16(cbe, cre);
      gle = _mm_madd_pi16(gle, PW_MF0344_F0285);
      ghe = _mm_madd_pi16(ghe, PW_MF0344_F0285);
      glo = _mm_unpacklo_pi16(cbo, cro);
      gho = _mm_unpackhi_pi16(cbo, cro);
      glo = _mm_madd_pi16(glo, PW_MF0344_F0285);
      gho = _mm_madd_pi16(gho, PW_MF0344_F0285);

      gle = _mm_add_pi32(gle, PD_ONEHALF);
      ghe = _mm_add_pi32(ghe, PD_ONEHALF);
      gle = _mm_srai_pi32(gle, SCALEBITS);
      ghe = _mm_srai_pi32(ghe, SCALEBITS);
      glo = _mm_add_pi32(glo, PD_ONEHALF);
      gho = _mm_add_pi32(gho, PD_ONEHALF);
      glo = _mm_srai_pi32(glo, SCALEBITS);
      gho = _mm_srai_pi32(gho, SCALEBITS);

      ge = _mm_packs_pi32(gle, ghe);       /* CbE*-FIX(0.344)+CrE*FIX(0.285) */
      go = _mm_packs_pi32(glo, gho);       /* CbO*-FIX(0.344)+CrO*FIX(0.285) */
      ge = _mm_sub_pi16(ge, cre);  /* CbE*-FIX(0.344)+CrE*-FIX(0.714)=(G-Y)E */
      go = _mm_sub_pi16(go, cro);  /* CbO*-FIX(0.344)+CrO*-FIX(0.714)=(G-Y)O */

      ye = _mm_and_si64(mask, y);             /* Y(0246) */
      yo = _mm_srli_pi16(y, BYTE_BIT);        /* Y(1357) */

      re = _mm_add_pi16(re, ye);              /* ((R-Y)E+YE)=(R0 R2 R4 R6) */
      ro = _mm_add_pi16(ro, yo);              /* ((R-Y)O+YO)=(R1 R3 R5 R7) */
      re = _mm_packs_pu16(re, re);            /* (R0 R2 R4 R6 ** ** ** **) */
      ro = _mm_packs_pu16(ro, ro);            /* (R1 R3 R5 R7 ** ** ** **) */

      ge = _mm_add_pi16(ge, ye);              /* ((G-Y)E+YE)=(G0 G2 G4 G6) */
      go = _mm_add_pi16(go, yo);              /* ((G-Y)O+YO)=(G1 G3 G5 G7) */
      ge = _mm_packs_pu16(ge, ge);            /* (G0 G2 G4 G6 ** ** ** **) */
      go = _mm_packs_pu16(go, go);            /* (G1 G3 G5 G7 ** ** ** **) */

      be = _mm_add_pi16(be, ye);              /* (YE+(B-Y)E)=(B0 B2 B4 B6) */
      bo = _mm_add_pi16(bo, yo);              /* (YO+(B-Y)O)=(B1 B3 B5 B7) */
      be = _mm_packs_pu16(be, be);            /* (B0 B2 B4 B6 ** ** ** **) */
      bo = _mm_packs_pu16(bo, bo);            /* (B1 B3 B5 B7 ** ** ** **) */

#if RGB_PIXELSIZE == 3

      /* mmA=(00 02 04 06 ** ** ** **), mmB=(01 03 05 07 ** ** ** **) */
      /* mmC=(10 12 14 16 ** ** ** **), mmD=(11 13 15 17 ** ** ** **) */
      mmA = _mm_unpacklo_pi8(mmA, mmC);       /* (00 10 02 12 04 14 06 16) */
      mmE = _mm_unpacklo_pi8(mmE, mmB);       /* (20 01 22 03 24 05 26 07) */
      mmD = _mm_unpacklo_pi8(mmD, mmF);       /* (11 21 13 23 15 25 17 27) */

      mmH = _mm_srli_si64(mmA, 2 * BYTE_BIT);

      mmG = _mm_unpackhi_pi16(mmA, mmE);      /* (04 14 24 05 06 16 26 07) */
      mmA = _mm_unpacklo_pi16(mmA, mmE);      /* (00 10 20 01 02 12 22 03) */

      mmE = _mm_srli_si64(mmE, 2 * BYTE_BIT);
      mmB = _mm_srli_si64(mmD, 2 * BYTE_BIT);  /* (13 23 15 25 17 27 -- --) */

      mmC = _mm_unpackhi_pi16(mmD, mmH);      /* (15 25 06 16 17 27 -- --) */
      mmD = _mm_unpacklo_pi16(mmD, mmH);      /* (11 21 02 12 13 23 04 14) */

      mmF = _mm_unpackhi_pi16(mmE, mmB);      /* (26 07 17 27 -- -- -- --) */
      mmE = _mm_unpacklo_pi16(mmE, mmB);      /* (22 03 13 23 24 05 15 25) */

      mmA = _mm_unpacklo_pi32(mmA, mmD);      /* (00 10 20 01 11 21 02 12) */
      mmE = _mm_unpacklo_pi32(mmE, mmG);      /* (22 03 13 23 04 14 24 05) */
      mmC = _mm_unpacklo_pi32(mmC, mmF);      /* (15 25 06 16 26 07 17 27) */

      if (num_cols >= 8) {
        if (!(((long)outptr) & 7)) {
          _mm_store_si64((__m64 *)outptr, mmA);
          _mm_store_si64((__m64 *)(outptr + 8), mmE);
          _mm_store_si64((__m64 *)(outptr + 16), mmC);
        } else {
          _mm_storeu_si64((__m64 *)outptr, mmA);
          _mm_storeu_si64((__m64 *)(outptr + 8), mmE);
          _mm_storeu_si64((__m64 *)(outptr + 16), mmC);
        }
        outptr += RGB_PIXELSIZE * 8;
      } else {
        col = num_cols * 3;
        asm(".set noreorder\r\n"

            "li       $8, 16\r\n"
            "move     $9, %4\r\n"
            "mov.s    $f4, %1\r\n"
            "mov.s    $f6, %3\r\n"
            "move     $10, %5\r\n"
            "bltu     $9, $8, 1f\r\n"
            "nop      \r\n"
            "gssdlc1  $f4, 7($10)\r\n"
            "gssdrc1  $f4, 0($10)\r\n"
            "gssdlc1  $f6, 7+8($10)\r\n"
            "gssdrc1  $f6, 8($10)\r\n"
            "mov.s    $f4, %2\r\n"
            "subu     $9, $9, 16\r\n"
            PTR_ADDU  "$10, $10, 16\r\n"
            "b        2f\r\n"
            "nop      \r\n"

            "1:       \r\n"
            "li       $8, 8\r\n"              /* st8 */
            "bltu     $9, $8, 2f\r\n"
            "nop      \r\n"
            "gssdlc1  $f4, 7($10)\r\n"
            "gssdrc1  $f4, 0($10)\r\n"
            "mov.s    $f4, %3\r\n"
            "subu     $9, $9, 8\r\n"
            PTR_ADDU  "$10, $10, 8\r\n"

            "2:       \r\n"
            "li       $8, 4\r\n"              /* st4 */
            "mfc1     $11, $f4\r\n"
            "bltu     $9, $8, 3f\r\n"
            "nop      \r\n"
            "swl      $11, 3($10)\r\n"
            "swr      $11, 0($10)\r\n"
            "li       $8, 32\r\n"
            "mtc1     $8, $f6\r\n"
            "dsrl     $f4, $f4, $f6\r\n"
            "mfc1     $11, $f4\r\n"
            "subu     $9, $9, 4\r\n"
            PTR_ADDU  "$10, $10, 4\r\n"

            "3:       \r\n"
            "li       $8, 2\r\n"              /* st2 */
            "bltu     $9, $8, 4f\r\n"
            "nop      \r\n"
            "ush      $11, 0($10)\r\n"
            "srl      $11, 16\r\n"
            "subu     $9, $9, 2\r\n"
            PTR_ADDU  "$10, $10, 2\r\n"

            "4:       \r\n"
            "li       $8, 1\r\n"              /* st1 */
            "bltu     $9, $8, 5f\r\n"
            "nop      \r\n"
            "sb       $11, 0($10)\r\n"

            "5:       \r\n"
            "nop      \r\n"                   /* end */
            : "=m" (*outptr)
            : "f" (mmA), "f" (mmC), "f" (mmE), "r" (col), "r" (outptr)
            : "$f4", "$f6", "$8", "$9", "$10", "$11", "memory"
           );
      }

#else  /* RGB_PIXELSIZE == 4 */

#ifdef RGBX_FILLER_0XFF
      xe = _mm_cmpeq_pi8(xe, xe);
      xo = _mm_cmpeq_pi8(xo, xo);
#else
      xe = _mm_xor_si64(xe, xe);
      xo = _mm_xor_si64(xo, xo);
#endif
      /* mmA=(00 02 04 06 ** ** ** **), mmB=(01 03 05 07 ** ** ** **) */
      /* mmC=(10 12 14 16 ** ** ** **), mmD=(11 13 15 17 ** ** ** **) */
      /* mmE=(20 22 24 26 ** ** ** **), mmF=(21 23 25 27 ** ** ** **) */
      /* mmG=(30 32 34 36 ** ** ** **), mmH=(31 33 35 37 ** ** ** **) */

      mmA = _mm_unpacklo_pi8(mmA, mmC);       /* (00 10 02 12 04 14 06 16) */
      mmE = _mm_unpacklo_pi8(mmE, mmG);       /* (20 30 22 32 24 34 26 36) */
      mmB = _mm_unpacklo_pi8(mmB, mmD);       /* (01 11 03 13 05 15 07 17) */
      mmF = _mm_unpacklo_pi8(mmF, mmH);       /* (21 31 23 33 25 35 27 37) */

      mmC = _mm_unpackhi_pi16(mmA, mmE);      /* (04 14 24 34 06 16 26 36) */
      mmA = _mm_unpacklo_pi16(mmA, mmE);      /* (00 10 20 30 02 12 22 32) */
      mmG = _mm_unpackhi_pi16(mmB, mmF);      /* (05 15 25 35 07 17 27 37) */
      mmB = _mm_unpacklo_pi16(mmB, mmF);      /* (01 11 21 31 03 13 23 33) */

      mmD = _mm_unpackhi_pi32(mmA, mmB);      /* (02 12 22 32 03 13 23 33) */
      mmA = _mm_unpacklo_pi32(mmA, mmB);      /* (00 10 20 30 01 11 21 31) */
      mmH = _mm_unpackhi_pi32(mmC, mmG);      /* (06 16 26 36 07 17 27 37) */
      mmC = _mm_unpacklo_pi32(mmC, mmG);      /* (04 14 24 34 05 15 25 35) */

      if (num_cols >= 8) {
        if (!(((long)outptr) & 7)) {
          _mm_store_si64((__m64 *)outptr, mmA);
          _mm_store_si64((__m64 *)(outptr + 8), mmD);
          _mm_store_si64((__m64 *)(outptr + 16), mmC);
          _mm_store_si64((__m64 *)(outptr + 24), mmH);
        } else {
          _mm_storeu_si64((__m64 *)outptr, mmA);
          _mm_storeu_si64((__m64 *)(outptr + 8), mmD);
          _mm_storeu_si64((__m64 *)(outptr + 16), mmC);
          _mm_storeu_si64((__m64 *)(outptr + 24), mmH);
        }
        outptr += RGB_PIXELSIZE * 8;
      } else {
        col = num_cols;
        asm(".set noreorder\r\n"              /* st16 */

            "li       $8, 4\r\n"
            "move     $9, %6\r\n"
            "move     $10, %7\r\n"
            "mov.s    $f4, %2\r\n"
            "mov.s    $f6, %4\r\n"
            "bltu     $9, $8, 1f\r\n"
            "nop      \r\n"
            "gssdlc1  $f4, 7($10)\r\n"
            "gssdrc1  $f4, 0($10)\r\n"
            "gssdlc1  $f6, 7+8($10)\r\n"
            "gssdrc1  $f6, 8($10)\r\n"
            "mov.s    $f4, %3\r\n"
            "mov.s    $f6, %5\r\n"
            "subu     $9, $9, 4\r\n"
            PTR_ADDU  "$10, $10, 16\r\n"

            "1:       \r\n"
            "li       $8, 2\r\n"              /* st8 */
            "bltu     $9, $8, 2f\r\n"
            "nop      \r\n"
            "gssdlc1  $f4, 7($10)\r\n"
            "gssdrc1  $f4, 0($10)\r\n"
            "mov.s    $f4, $f6\r\n"
            "subu     $9, $9, 2\r\n"
            PTR_ADDU  "$10, $10, 8\r\n"

            "2:       \r\n"
            "li       $8, 1\r\n"              /* st4 */
            "bltu     $9, $8, 3f\r\n"
            "nop      \r\n"
            "gsswlc1  $f4, 3($10)\r\n"
            "gsswrc1  $f4, 0($10)\r\n"

            "3:       \r\n"
            "li       %1, 0\r\n"              /* end */
            : "=m" (*outptr), "=r" (col)
            : "f" (mmA), "f" (mmC), "f" (mmD), "f" (mmH), "r" (col),
              "r" (outptr)
            : "$f4", "$f6", "$8", "$9", "$10", "memory"
           );
      }

#endif

    }
  }
}

#undef mmA
#undef mmB
#undef mmC
#undef mmD
#undef mmE
#undef mmF
#undef mmG
#undef mmH
