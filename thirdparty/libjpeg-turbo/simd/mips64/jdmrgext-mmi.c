/*
 * Loongson MMI optimizations for libjpeg-turbo
 *
 * Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
 * Copyright (C) 2015, 2019, D. R. Commander.  All Rights Reserved.
 * Copyright (C) 2016-2018, Loongson Technology Corporation Limited, BeiJing.
 *                          All Rights Reserved.
 * Authors:  ZhangLixia <zhanglixia-hf@loongson.cn>
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

/* This file is included by jdmerge-mmi.c */


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


void jsimd_h2v1_merged_upsample_mmi(JDIMENSION output_width,
                                    JSAMPIMAGE input_buf,
                                    JDIMENSION in_row_group_ctr,
                                    JSAMPARRAY output_buf)
{
  JSAMPROW outptr, inptr0, inptr1, inptr2;
  int num_cols, col;
  __m64 ythise, ythiso, ythis, ynexte, ynexto, ynext, yl, y;
  __m64 cbl, cbl2, cbh, cbh2, cb, crl, crl2, crh, crh2, cr;
  __m64 rle, rlo, rl, rhe, rho, rh, re, ro;
  __m64 ga, gb, gle, glo, gl, gc, gd, ghe, gho, gh, ge, go;
  __m64 ble, blo, bl, bhe, bho, bh, be, bo, xe = 0.0, xo = 0.0;
  __m64 decenter, mask, zero = 0.0;
#if RGB_PIXELSIZE == 4
  __m64 mm8, mm9;
#endif

  inptr0 = input_buf[0][in_row_group_ctr];
  inptr1 = input_buf[1][in_row_group_ctr];
  inptr2 = input_buf[2][in_row_group_ctr];
  outptr = output_buf[0];

  for (num_cols = output_width >> 1; num_cols > 0; num_cols -= 8,
       inptr0 += 16, inptr1 += 8, inptr2 += 8) {

    cb = _mm_load_si64((__m64 *)inptr1);
    cr = _mm_load_si64((__m64 *)inptr2);
    ythis = _mm_load_si64((__m64 *)inptr0);
    ynext = _mm_load_si64((__m64 *)inptr0 + 1);

    mask = decenter = 0.0;
    mask = _mm_cmpeq_pi16(mask, mask);
    decenter = _mm_cmpeq_pi16(decenter, decenter);
    mask = _mm_srli_pi16(mask, BYTE_BIT);   /* {0xFF 0x00 0xFF 0x00 ..} */
    decenter = _mm_slli_pi16(decenter, 7);  /* {0xFF80 0xFF80 0xFF80 0xFF80} */

    cbl = _mm_unpacklo_pi8(cb, zero);         /* Cb(0123) */
    cbh = _mm_unpackhi_pi8(cb, zero);         /* Cb(4567) */
    crl = _mm_unpacklo_pi8(cr, zero);         /* Cr(0123) */
    crh = _mm_unpackhi_pi8(cr, zero);         /* Cr(4567) */
    cbl = _mm_add_pi16(cbl, decenter);
    cbh = _mm_add_pi16(cbh, decenter);
    crl = _mm_add_pi16(crl, decenter);
    crh = _mm_add_pi16(crh, decenter);

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

    cbl2 = _mm_add_pi16(cbl, cbl);            /* 2*CbL */
    cbh2 = _mm_add_pi16(cbh, cbh);            /* 2*CbH */
    crl2 = _mm_add_pi16(crl, crl);            /* 2*CrL */
    crh2 = _mm_add_pi16(crh, crh);            /* 2*CrH */

    bl = _mm_mulhi_pi16(cbl2, PW_MF0228);     /* (2*CbL * -FIX(0.22800) */
    bh = _mm_mulhi_pi16(cbh2, PW_MF0228);     /* (2*CbH * -FIX(0.22800) */
    rl = _mm_mulhi_pi16(crl2, PW_F0402);      /* (2*CrL * FIX(0.40200)) */
    rh = _mm_mulhi_pi16(crh2, PW_F0402);      /* (2*CrH * FIX(0.40200)) */

    bl = _mm_add_pi16(bl, PW_ONE);
    bh = _mm_add_pi16(bh, PW_ONE);
    bl = _mm_srai_pi16(bl, 1);                /* (CbL * -FIX(0.22800)) */
    bh = _mm_srai_pi16(bh, 1);                /* (CbH * -FIX(0.22800)) */
    rl = _mm_add_pi16(rl, PW_ONE);
    rh = _mm_add_pi16(rh, PW_ONE);
    rl = _mm_srai_pi16(rl, 1);                /* (CrL * FIX(0.40200)) */
    rh = _mm_srai_pi16(rh, 1);                /* (CrH * FIX(0.40200)) */

    bl = _mm_add_pi16(bl, cbl);
    bh = _mm_add_pi16(bh, cbh);
    bl = _mm_add_pi16(bl, cbl);               /* (CbL * FIX(1.77200))=(B-Y)L */
    bh = _mm_add_pi16(bh, cbh);               /* (CbH * FIX(1.77200))=(B-Y)H */
    rl = _mm_add_pi16(rl, crl);               /* (CrL * FIX(1.40200))=(R-Y)L */
    rh = _mm_add_pi16(rh, crh);               /* (CrH * FIX(1.40200))=(R-Y)H */

    ga = _mm_unpacklo_pi16(cbl, crl);
    gb = _mm_unpackhi_pi16(cbl, crl);
    ga = _mm_madd_pi16(ga, PW_MF0344_F0285);
    gb = _mm_madd_pi16(gb, PW_MF0344_F0285);
    gc = _mm_unpacklo_pi16(cbh, crh);
    gd = _mm_unpackhi_pi16(cbh, crh);
    gc = _mm_madd_pi16(gc, PW_MF0344_F0285);
    gd = _mm_madd_pi16(gd, PW_MF0344_F0285);

    ga = _mm_add_pi32(ga, PD_ONEHALF);
    gb = _mm_add_pi32(gb, PD_ONEHALF);
    ga = _mm_srai_pi32(ga, SCALEBITS);
    gb = _mm_srai_pi32(gb, SCALEBITS);
    gc = _mm_add_pi32(gc, PD_ONEHALF);
    gd = _mm_add_pi32(gd, PD_ONEHALF);
    gc = _mm_srai_pi32(gc, SCALEBITS);
    gd = _mm_srai_pi32(gd, SCALEBITS);

    gl = _mm_packs_pi32(ga, gb);           /* CbL*-FIX(0.344)+CrL*FIX(0.285) */
    gh = _mm_packs_pi32(gc, gd);           /* CbH*-FIX(0.344)+CrH*FIX(0.285) */
    gl = _mm_sub_pi16(gl, crl);    /* CbL*-FIX(0.344)+CrL*-FIX(0.714)=(G-Y)L */
    gh = _mm_sub_pi16(gh, crh);    /* CbH*-FIX(0.344)+CrH*-FIX(0.714)=(G-Y)H */

    ythise = _mm_and_si64(mask, ythis);       /* Y(0246) */
    ythiso = _mm_srli_pi16(ythis, BYTE_BIT);  /* Y(1357) */
    ynexte = _mm_and_si64(mask, ynext);       /* Y(8ACE) */
    ynexto = _mm_srli_pi16(ynext, BYTE_BIT);  /* Y(9BDF) */

    rle = _mm_add_pi16(rl, ythise);           /* (R0 R2 R4 R6) */
    rlo = _mm_add_pi16(rl, ythiso);           /* (R1 R3 R5 R7) */
    rhe = _mm_add_pi16(rh, ynexte);           /* (R8 RA RC RE) */
    rho = _mm_add_pi16(rh, ynexto);           /* (R9 RB RD RF) */
    re = _mm_packs_pu16(rle, rhe);            /* (R0 R2 R4 R6 R8 RA RC RE) */
    ro = _mm_packs_pu16(rlo, rho);            /* (R1 R3 R5 R7 R9 RB RD RF) */

    gle = _mm_add_pi16(gl, ythise);           /* (G0 G2 G4 G6) */
    glo = _mm_add_pi16(gl, ythiso);           /* (G1 G3 G5 G7) */
    ghe = _mm_add_pi16(gh, ynexte);           /* (G8 GA GC GE) */
    gho = _mm_add_pi16(gh, ynexto);           /* (G9 GB GD GF) */
    ge = _mm_packs_pu16(gle, ghe);            /* (G0 G2 G4 G6 G8 GA GC GE) */
    go = _mm_packs_pu16(glo, gho);            /* (G1 G3 G5 G7 G9 GB GD GF) */

    ble = _mm_add_pi16(bl, ythise);           /* (B0 B2 B4 B6) */
    blo = _mm_add_pi16(bl, ythiso);           /* (B1 B3 B5 B7) */
    bhe = _mm_add_pi16(bh, ynexte);           /* (B8 BA BC BE) */
    bho = _mm_add_pi16(bh, ynexto);           /* (B9 BB BD BF) */
    be = _mm_packs_pu16(ble, bhe);            /* (B0 B2 B4 B6 B8 BA BC BE) */
    bo = _mm_packs_pu16(blo, bho);            /* (B1 B3 B5 B7 B9 BB BD BF) */

#if RGB_PIXELSIZE == 3

    /* mmA=(00 02 04 06 08 0A 0C 0E), mmB=(01 03 05 07 09 0B 0D 0F) */
    /* mmC=(10 12 14 16 18 1A 1C 1E), mmD=(11 13 15 17 19 1B 1D 1F) */
    /* mmE=(20 22 24 26 28 2A 2C 2E), mmF=(21 23 25 27 29 2B 2D 2F) */
    mmG = _mm_unpacklo_pi8(mmA, mmC);         /* (00 10 02 12 04 14 06 16) */
    mmA = _mm_unpackhi_pi8(mmA, mmC);         /* (08 18 0A 1A 0C 1C 0E 1E) */
    mmH = _mm_unpacklo_pi8(mmE, mmB);         /* (20 01 22 03 24 05 26 07) */
    mmE = _mm_unpackhi_pi8(mmE, mmB);         /* (28 09 2A 0B 2C 0D 2E 0F) */
    mmC = _mm_unpacklo_pi8(mmD, mmF);         /* (11 21 13 23 15 25 17 27) */
    mmD = _mm_unpackhi_pi8(mmD, mmF);         /* (19 29 1B 2B 1D 2D 1F 2F) */

    mmB = _mm_unpacklo_pi16(mmG, mmA);        /* (00 10 08 18 02 12 0A 1A) */
    mmA = _mm_unpackhi_pi16(mmG, mmA);        /* (04 14 0C 1C 06 16 0E 1E) */
    mmF = _mm_unpacklo_pi16(mmH, mmE);        /* (20 01 28 09 22 03 2A 0B) */
    mmE = _mm_unpackhi_pi16(mmH, mmE);        /* (24 05 2C 0D 26 07 2E 0F) */
    mmH = _mm_unpacklo_pi16(mmC, mmD);        /* (11 21 19 29 13 23 1B 2B) */
    mmG = _mm_unpackhi_pi16(mmC, mmD);        /* (15 25 1D 2D 17 27 1F 2F) */

    mmC = _mm_unpacklo_pi16(mmB, mmF);        /* (00 10 20 01 08 18 28 09) */
    mmB = _mm_srli_si64(mmB, 4 * BYTE_BIT);
    mmB = _mm_unpacklo_pi16(mmH, mmB);        /* (11 21 02 12 19 29 0A 1A) */
    mmD = _mm_unpackhi_pi16(mmF, mmH);        /* (22 03 13 23 2A 0B 1B 2B) */
    mmF = _mm_unpacklo_pi16(mmA, mmE);        /* (04 14 24 05 0C 1C 2C 0D) */
    mmA = _mm_srli_si64(mmA, 4 * BYTE_BIT);
    mmH = _mm_unpacklo_pi16(mmG, mmA);        /* (15 25 06 16 1D 2D 0E 1E) */
    mmG = _mm_unpackhi_pi16(mmE, mmG);        /* (26 07 17 27 2E 0F 1F 2F) */

    mmA = _mm_unpacklo_pi32(mmC, mmB);        /* (00 10 20 01 11 21 02 12) */
    mmE = _mm_unpackhi_pi32(mmC, mmB);        /* (08 18 28 09 19 29 0A 1A) */
    mmB = _mm_unpacklo_pi32(mmD, mmF);        /* (22 03 13 23 04 14 24 05) */
    mmF = _mm_unpackhi_pi32(mmD, mmF);        /* (2A 0B 1B 2B 0C 1C 2C 0D) */
    mmC = _mm_unpacklo_pi32(mmH, mmG);        /* (15 25 06 16 26 07 17 27) */
    mmG = _mm_unpackhi_pi32(mmH, mmG);        /* (1D 2D 0E 1E 2E 0F 1F 2F) */

    if (num_cols >= 8) {
      if (!(((long)outptr) & 7)) {
        _mm_store_si64((__m64 *)outptr, mmA);
        _mm_store_si64((__m64 *)(outptr + 8), mmB);
        _mm_store_si64((__m64 *)(outptr + 16), mmC);
        _mm_store_si64((__m64 *)(outptr + 24), mmE);
        _mm_store_si64((__m64 *)(outptr + 32), mmF);
        _mm_store_si64((__m64 *)(outptr + 40), mmG);
      } else {
        _mm_storeu_si64((__m64 *)outptr, mmA);
        _mm_storeu_si64((__m64 *)(outptr + 8), mmB);
        _mm_storeu_si64((__m64 *)(outptr + 16), mmC);
        _mm_storeu_si64((__m64 *)(outptr + 24), mmE);
        _mm_storeu_si64((__m64 *)(outptr + 32), mmF);
        _mm_storeu_si64((__m64 *)(outptr + 40), mmG);
      }
      outptr += RGB_PIXELSIZE * 16;
    } else {
      if (output_width & 1)
        col = num_cols * 6 + 3;
      else
        col = num_cols * 6;

      asm(".set noreorder\r\n"                /* st24 */

          "li       $8, 24\r\n"
          "move     $9, %7\r\n"
          "mov.s    $f4, %1\r\n"
          "mov.s    $f6, %2\r\n"
          "mov.s    $f8, %3\r\n"
          "move     $10, %8\r\n"
          "bltu     $9, $8, 1f\r\n"
          "nop      \r\n"
          "gssdlc1  $f4, 7($10)\r\n"
          "gssdrc1  $f4, 0($10)\r\n"
          "gssdlc1  $f6, 7+8($10)\r\n"
          "gssdrc1  $f6, 8($10)\r\n"
          "gssdlc1  $f8, 7+16($10)\r\n"
          "gssdrc1  $f8, 16($10)\r\n"
          "mov.s    $f4, %4\r\n"
          "mov.s    $f6, %5\r\n"
          "mov.s    $f8, %6\r\n"
          "subu     $9, $9, 24\r\n"
          PTR_ADDU  "$10, $10, 24\r\n"

          "1:       \r\n"
          "li       $8, 16\r\n"               /* st16 */
          "bltu     $9, $8, 2f\r\n"
          "nop      \r\n"
          "gssdlc1  $f4, 7($10)\r\n"
          "gssdrc1  $f4, 0($10)\r\n"
          "gssdlc1  $f6, 7+8($10)\r\n"
          "gssdrc1  $f6, 8($10)\r\n"
          "mov.s    $f4, $f8\r\n"
          "subu     $9, $9, 16\r\n"
          PTR_ADDU  "$10, $10, 16\r\n"

          "2:       \r\n"
          "li       $8,  8\r\n"               /* st8 */
          "bltu     $9, $8, 3f\r\n"
          "nop      \r\n"
          "gssdlc1  $f4, 7($10)\r\n"
          "gssdrc1  $f4, 0($10)\r\n"
          "mov.s    $f4, $f6\r\n"
          "subu     $9, $9, 8\r\n"
          PTR_ADDU  "$10, $10, 8\r\n"

          "3:       \r\n"
          "li       $8,  4\r\n"               /* st4 */
          "mfc1     $11, $f4\r\n"
          "bltu     $9, $8, 4f\r\n"
          "nop      \r\n"
          "swl      $11, 3($10)\r\n"
          "swr      $11, 0($10)\r\n"
          "li       $8, 32\r\n"
          "mtc1     $8, $f6\r\n"
          "dsrl     $f4, $f4, $f6\r\n"
          "mfc1     $11, $f4\r\n"
          "subu     $9, $9, 4\r\n"
          PTR_ADDU  "$10, $10, 4\r\n"

          "4:       \r\n"
          "li       $8, 2\r\n"                /* st2 */
          "bltu     $9, $8, 5f\r\n"
          "nop      \r\n"
          "ush      $11, 0($10)\r\n"
          "srl      $11, 16\r\n"
          "subu     $9, $9, 2\r\n"
          PTR_ADDU  "$10, $10, 2\r\n"

          "5:       \r\n"
          "li       $8, 1\r\n"                /* st1 */
          "bltu     $9, $8, 6f\r\n"
          "nop      \r\n"
          "sb       $11, 0($10)\r\n"

          "6:       \r\n"
          "nop      \r\n"                     /* end */
          : "=m" (*outptr)
          : "f" (mmA), "f" (mmB), "f" (mmC), "f" (mmE), "f" (mmF),
            "f" (mmG), "r" (col), "r" (outptr)
          : "$f4", "$f6", "$f8", "$8", "$9", "$10", "$11", "memory"
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
    /* mmA=(00 02 04 06 08 0A 0C 0E), mmB=(01 03 05 07 09 0B 0D 0F) */
    /* mmC=(10 12 14 16 18 1A 1C 1E), mmD=(11 13 15 17 19 1B 1D 1F) */
    /* mmE=(20 22 24 26 28 2A 2C 2E), mmF=(21 23 25 27 29 2B 2D 2F) */
    /* mmG=(30 32 34 36 38 3A 3C 3E), mmH=(31 33 35 37 39 3B 3D 3F) */

    mm8 = _mm_unpacklo_pi8(mmA, mmC);         /* (00 10 02 12 04 14 06 16) */
    mm9 = _mm_unpackhi_pi8(mmA, mmC);         /* (08 18 0A 1A 0C 1C 0E 1E) */
    mmA = _mm_unpacklo_pi8(mmE, mmG);         /* (20 30 22 32 24 34 26 36) */
    mmE = _mm_unpackhi_pi8(mmE, mmG);         /* (28 38 2A 3A 2C 3C 2E 3E) */

    mmG = _mm_unpacklo_pi8(mmB, mmD);         /* (01 11 03 13 05 15 07 17) */
    mmB = _mm_unpackhi_pi8(mmB, mmD);         /* (09 19 0B 1B 0D 1D 0F 1F) */
    mmD = _mm_unpacklo_pi8(mmF, mmH);         /* (21 31 23 33 25 35 27 37) */
    mmF = _mm_unpackhi_pi8(mmF, mmH);         /* (29 39 2B 3B 2D 3D 2F 3F) */

    mmH = _mm_unpacklo_pi16(mm8, mmA);        /* (00 10 20 30 02 12 22 32) */
    mm8 = _mm_unpackhi_pi16(mm8, mmA);        /* (04 14 24 34 06 16 26 36) */
    mmA = _mm_unpacklo_pi16(mmG, mmD);        /* (01 11 21 31 03 13 23 33) */
    mmD = _mm_unpackhi_pi16(mmG, mmD);        /* (05 15 25 35 07 17 27 37) */

    mmG = _mm_unpackhi_pi16(mm9, mmE);        /* (0C 1C 2C 3C 0E 1E 2E 3E) */
    mm9 = _mm_unpacklo_pi16(mm9, mmE);        /* (08 18 28 38 0A 1A 2A 3A) */
    mmE = _mm_unpacklo_pi16(mmB, mmF);        /* (09 19 29 39 0B 1B 2B 3B) */
    mmF = _mm_unpackhi_pi16(mmB, mmF);        /* (0D 1D 2D 3D 0F 1F 2F 3F) */

    mmB = _mm_unpackhi_pi32(mmH, mmA);        /* (02 12 22 32 03 13 23 33) */
    mmA = _mm_unpacklo_pi32(mmH, mmA);        /* (00 10 20 30 01 11 21 31) */
    mmC = _mm_unpacklo_pi32(mm8, mmD);        /* (04 14 24 34 05 15 25 35) */
    mmD = _mm_unpackhi_pi32(mm8, mmD);        /* (06 16 26 36 07 17 27 37) */

    mmH = _mm_unpackhi_pi32(mmG, mmF);        /* (0E 1E 2E 3E 0F 1F 2F 3F) */
    mmG = _mm_unpacklo_pi32(mmG, mmF);        /* (0C 1C 2C 3C 0D 1D 2D 3D) */
    mmF = _mm_unpackhi_pi32(mm9, mmE);        /* (0A 1A 2A 3A 0B 1B 2B 3B) */
    mmE = _mm_unpacklo_pi32(mm9, mmE);        /* (08 18 28 38 09 19 29 39) */

    if (num_cols >= 8) {
      if (!(((long)outptr) & 7)) {
        _mm_store_si64((__m64 *)outptr, mmA);
        _mm_store_si64((__m64 *)(outptr + 8), mmB);
        _mm_store_si64((__m64 *)(outptr + 16), mmC);
        _mm_store_si64((__m64 *)(outptr + 24), mmD);
        _mm_store_si64((__m64 *)(outptr + 32), mmE);
        _mm_store_si64((__m64 *)(outptr + 40), mmF);
        _mm_store_si64((__m64 *)(outptr + 48), mmG);
        _mm_store_si64((__m64 *)(outptr + 56), mmH);
      } else {
        _mm_storeu_si64((__m64 *)outptr, mmA);
        _mm_storeu_si64((__m64 *)(outptr + 8), mmB);
        _mm_storeu_si64((__m64 *)(outptr + 16), mmC);
        _mm_storeu_si64((__m64 *)(outptr + 24), mmD);
        _mm_storeu_si64((__m64 *)(outptr + 32), mmE);
        _mm_storeu_si64((__m64 *)(outptr + 40), mmF);
        _mm_storeu_si64((__m64 *)(outptr + 48), mmG);
        _mm_storeu_si64((__m64 *)(outptr + 56), mmH);
      }
      outptr += RGB_PIXELSIZE * 16;
    } else {
      if (output_width & 1)
        col = num_cols * 2 + 1;
      else
        col = num_cols * 2;
      asm(".set noreorder\r\n"                /* st32 */

          "li       $8, 8\r\n"
          "move     $9, %10\r\n"
          "move     $10, %11\r\n"
          "mov.s    $f4, %2\r\n"
          "mov.s    $f6, %3\r\n"
          "mov.s    $f8, %4\r\n"
          "mov.s    $f10, %5\r\n"
          "bltu     $9, $8, 1f\r\n"
          "nop      \r\n"
          "gssdlc1  $f4, 7($10)\r\n"
          "gssdrc1  $f4, 0($10)\r\n"
          "gssdlc1  $f6, 7+8($10)\r\n"
          "gssdrc1  $f6, 8($10)\r\n"
          "gssdlc1  $f8, 7+16($10)\r\n"
          "gssdrc1  $f8, 16($10)\r\n"
          "gssdlc1  $f10, 7+24($10)\r\n"
          "gssdrc1  $f10, 24($10)\r\n"
          "mov.s    $f4, %6\r\n"
          "mov.s    $f6, %7\r\n"
          "mov.s    $f8, %8\r\n"
          "mov.s    $f10, %9\r\n"
          "subu     $9, $9, 8\r\n"
          PTR_ADDU  "$10, $10, 32\r\n"

          "1:       \r\n"
          "li       $8, 4\r\n"                /* st16 */
          "bltu     $9, $8, 2f\r\n"
          "nop      \r\n"
          "gssdlc1  $f4, 7($10)\r\n"
          "gssdrc1  $f4, 0($10)\r\n"
          "gssdlc1  $f6, 7+8($10)\r\n"
          "gssdrc1  $f6, 8($10)\r\n"
          "mov.s    $f4, $f8\r\n"
          "mov.s    $f6, $f10\r\n"
          "subu     $9, $9, 4\r\n"
          PTR_ADDU  "$10, $10, 16\r\n"

          "2:       \r\n"
          "li       $8, 2\r\n"                /* st8 */
          "bltu     $9, $8, 3f\r\n"
          "nop      \r\n"
          "gssdlc1  $f4, 7($10)\r\n"
          "gssdrc1  $f4, 0($10)\r\n"
          "mov.s    $f4, $f6\r\n"
          "subu     $9, $9, 2\r\n"
          PTR_ADDU  "$10, $10, 8\r\n"

          "3:       \r\n"
          "li       $8, 1\r\n"                /* st4 */
          "bltu     $9, $8, 4f\r\n"
          "nop      \r\n"
          "gsswlc1  $f4, 3($10)\r\n"
          "gsswrc1  $f4, 0($10)\r\n"

          "4:       \r\n"
          "li       %1, 0\r\n"                /* end */
          : "=m" (*outptr), "=r" (col)
          : "f" (mmA), "f" (mmB), "f" (mmC), "f" (mmD), "f" (mmE), "f" (mmF),
            "f" (mmG), "f" (mmH), "r" (col), "r" (outptr)
          : "$f4", "$f6", "$f8", "$f10", "$8", "$9", "$10", "memory"
         );
    }

#endif

  }

  if (!((output_width >> 1) & 7)) {
    if (output_width & 1) {
      cb = _mm_load_si64((__m64 *)inptr1);
      cr = _mm_load_si64((__m64 *)inptr2);
      y = _mm_load_si64((__m64 *)inptr0);

      decenter = 0.0;
      decenter = _mm_cmpeq_pi16(decenter, decenter);
      decenter = _mm_slli_pi16(decenter, 7);  /* {0xFF80 0xFF80 0xFF80 0xFF80} */

      cbl = _mm_unpacklo_pi8(cb, zero);       /* Cb(0123) */
      crl = _mm_unpacklo_pi8(cr, zero);       /* Cr(0123) */
      cbl = _mm_add_pi16(cbl, decenter);
      crl = _mm_add_pi16(crl, decenter);

      cbl2 = _mm_add_pi16(cbl, cbl);          /* 2*CbL */
      crl2 = _mm_add_pi16(crl, crl);          /* 2*CrL */
      bl = _mm_mulhi_pi16(cbl2, PW_MF0228);   /* (2*CbL * -FIX(0.22800) */
      rl = _mm_mulhi_pi16(crl2, PW_F0402);    /* (2*CrL * FIX(0.40200)) */

      bl = _mm_add_pi16(bl, PW_ONE);
      bl = _mm_srai_pi16(bl, 1);              /* (CbL * -FIX(0.22800)) */
      rl = _mm_add_pi16(rl, PW_ONE);
      rl = _mm_srai_pi16(rl, 1);              /* (CrL * FIX(0.40200)) */

      bl = _mm_add_pi16(bl, cbl);
      bl = _mm_add_pi16(bl, cbl);             /* (CbL * FIX(1.77200))=(B-Y)L */
      rl = _mm_add_pi16(rl, crl);             /* (CrL * FIX(1.40200))=(R-Y)L */

      gl = _mm_unpacklo_pi16(cbl, crl);
      gl = _mm_madd_pi16(gl, PW_MF0344_F0285);
      gl = _mm_add_pi32(gl, PD_ONEHALF);
      gl = _mm_srai_pi32(gl, SCALEBITS);
      gl = _mm_packs_pi32(gl, zero);       /* CbL*-FIX(0.344)+CrL*FIX(0.285) */
      gl = _mm_sub_pi16(gl, crl);  /* CbL*-FIX(0.344)+CrL*-FIX(0.714)=(G-Y)L */

      yl = _mm_unpacklo_pi8(y, zero);         /* Y(0123) */
      rl = _mm_add_pi16(rl, yl);              /* (R0 R1 R2 R3) */
      gl = _mm_add_pi16(gl, yl);              /* (G0 G1 G2 G3) */
      bl = _mm_add_pi16(bl, yl);              /* (B0 B1 B2 B3) */
      re = _mm_packs_pu16(rl, rl);
      ge = _mm_packs_pu16(gl, gl);
      be = _mm_packs_pu16(bl, bl);
#if RGB_PIXELSIZE == 3
      mmA = _mm_unpacklo_pi8(mmA, mmC);
      mmA = _mm_unpacklo_pi16(mmA, mmE);
      asm(".set noreorder\r\n"

          "move    $8, %2\r\n"
          "mov.s   $f4, %1\r\n"
          "mfc1    $9, $f4\r\n"
          "ush     $9, 0($8)\r\n"
          "srl     $9, 16\r\n"
          "sb      $9, 2($8)\r\n"
          : "=m" (*outptr)
          : "f" (mmA), "r" (outptr)
          : "$f4", "$8", "$9", "memory"
         );
#else  /* RGB_PIXELSIZE == 4 */

#ifdef RGBX_FILLER_0XFF
      xe = _mm_cmpeq_pi8(xe, xe);
#else
      xe = _mm_xor_si64(xe, xe);
#endif
      mmA = _mm_unpacklo_pi8(mmA, mmC);
      mmE = _mm_unpacklo_pi8(mmE, mmG);
      mmA = _mm_unpacklo_pi16(mmA, mmE);
      asm(".set noreorder\r\n"

          "move    $8, %2\r\n"
          "mov.s   $f4, %1\r\n"
          "gsswlc1 $f4, 3($8)\r\n"
          "gsswrc1 $f4, 0($8)\r\n"
          : "=m" (*outptr)
          : "f" (mmA), "r" (outptr)
          : "$f4", "$8", "memory"
         );
#endif
    }
  }
}


void jsimd_h2v2_merged_upsample_mmi(JDIMENSION output_width,
                                    JSAMPIMAGE input_buf,
                                    JDIMENSION in_row_group_ctr,
                                    JSAMPARRAY output_buf)
{
  JSAMPROW inptr, outptr;

  inptr = input_buf[0][in_row_group_ctr];
  outptr = output_buf[0];

  input_buf[0][in_row_group_ctr] = input_buf[0][in_row_group_ctr * 2];
  jsimd_h2v1_merged_upsample_mmi(output_width, input_buf, in_row_group_ctr,
                                 output_buf);

  input_buf[0][in_row_group_ctr] = input_buf[0][in_row_group_ctr * 2 + 1];
  output_buf[0] = output_buf[1];
  jsimd_h2v1_merged_upsample_mmi(output_width, input_buf, in_row_group_ctr,
                                 output_buf);

  input_buf[0][in_row_group_ctr] = inptr;
  output_buf[0] = outptr;
}


#undef mmA
#undef mmB
#undef mmC
#undef mmD
#undef mmE
#undef mmF
#undef mmG
#undef mmH
