/*
 * Loongson MMI optimizations for libjpeg-turbo
 *
 * Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
 * Copyright (C) 2014-2015, 2019, D. R. Commander.  All Rights Reserved.
 * Copyright (C) 2016-2018, Loongson Technology Corporation Limited, BeiJing.
 *                          All Rights Reserved.
 * Authors:  ZhuChen     <zhuchen@loongson.cn>
 *           SunZhangzhi <sunzhangzhi-cq@loongson.cn>
 *           CaiWanwei   <caiwanwei@loongson.cn>
 *           ZhangLixia  <zhanglixia-hf@loongson.cn>
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

/* This file is included by jccolor-mmi.c */


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


void jsimd_rgb_ycc_convert_mmi(JDIMENSION image_width, JSAMPARRAY input_buf,
                               JSAMPIMAGE output_buf, JDIMENSION output_row,
                               int num_rows)
{
  JSAMPROW inptr, outptr0, outptr1, outptr2;
  int num_cols, col;
  __m64 re, ro, ge, go, be, bo, xe;
#if RGB_PIXELSIZE == 4
  __m64 xo;
#endif
  __m64 rgle, rghe, rglo, rgho, bgle, bghe, bglo, bgho;
  __m64 ble, halfble, bhe, halfbhe, blo, halfblo, bho, halfbho;
  __m64 rle, halfrle, rhe, halfrhe, rlo, halfrlo, rho, halfrho;
  __m64 yle_rg, yhe_rg, yle_bg, yhe_bg, yle, yhe, ye;
  __m64 ylo_rg, yho_rg, ylo_bg, yho_bg, ylo, yho, yo, y;
  __m64 cble, cbhe, cbe, cblo, cbho, cbo, cb;
  __m64 crle, crhe, cre, crlo, crho, cro, cr;

  while (--num_rows >= 0) {
    inptr = *input_buf++;
    outptr0 = output_buf[0][output_row];
    outptr1 = output_buf[1][output_row];
    outptr2 = output_buf[2][output_row];
    output_row++;

    for (num_cols = image_width; num_cols > 0; num_cols -= 8,
         outptr0 += 8, outptr1 += 8, outptr2 += 8) {

#if RGB_PIXELSIZE == 3

      if (num_cols < 8) {
        col = num_cols * 3;
        asm(".set noreorder\r\n"

            "li       $8, 1\r\n"
            "move     $9, %3\r\n"
            "and      $10, $9, $8\r\n"
            "beqz     $10, 1f\r\n"
            "nop      \r\n"
            "subu     $9, $9, 1\r\n"
            "xor      $12, $12, $12\r\n"
            "move     $13, %5\r\n"
            PTR_ADDU  "$13, $13, $9\r\n"
            "lbu      $12, 0($13)\r\n"

            "1:       \r\n"
            "li       $8, 2\r\n"
            "and      $10, $9, $8\r\n"
            "beqz     $10, 2f\r\n"
            "nop      \r\n"
            "subu     $9, $9, 2\r\n"
            "xor      $11, $11, $11\r\n"
            "move     $13, %5\r\n"
            PTR_ADDU  "$13, $13, $9\r\n"
            "lhu      $11, 0($13)\r\n"
            "sll      $12, $12, 16\r\n"
            "or       $12, $12, $11\r\n"

            "2:       \r\n"
            "dmtc1    $12, %0\r\n"
            "li       $8, 4\r\n"
            "and      $10, $9, $8\r\n"
            "beqz     $10, 3f\r\n"
            "nop      \r\n"
            "subu     $9, $9, 4\r\n"
            "move     $13, %5\r\n"
            PTR_ADDU  "$13, $13, $9\r\n"
            "lwu      $14, 0($13)\r\n"
            "dmtc1    $14, %1\r\n"
            "dsll32   $12, $12, 0\r\n"
            "or       $12, $12, $14\r\n"
            "dmtc1    $12, %0\r\n"

            "3:       \r\n"
            "li       $8, 8\r\n"
            "and      $10, $9, $8\r\n"
            "beqz     $10, 4f\r\n"
            "nop      \r\n"
            "mov.s    %1, %0\r\n"
            "ldc1     %0, 0(%5)\r\n"
            "li       $9, 8\r\n"
            "j        5f\r\n"
            "nop      \r\n"

            "4:       \r\n"
            "li       $8, 16\r\n"
            "and      $10, $9, $8\r\n"
            "beqz     $10, 5f\r\n"
            "nop      \r\n"
            "mov.s    %2, %0\r\n"
            "ldc1     %0, 0(%5)\r\n"
            "ldc1     %1, 8(%5)\r\n"

            "5:       \r\n"
            "nop      \r\n"
            ".set reorder\r\n"

            : "=f" (mmA), "=f" (mmG), "=f" (mmF)
            : "r" (col), "r" (num_rows), "r" (inptr)
            : "$f0", "$f2", "$f4", "$8", "$9", "$10", "$11", "$12", "$13",
              "$14", "memory"
           );
      } else {
        if (!(((long)inptr) & 7)) {
          mmA = _mm_load_si64((__m64 *)&inptr[0]);
          mmG = _mm_load_si64((__m64 *)&inptr[8]);
          mmF = _mm_load_si64((__m64 *)&inptr[16]);
        } else {
          mmA = _mm_loadu_si64((__m64 *)&inptr[0]);
          mmG = _mm_loadu_si64((__m64 *)&inptr[8]);
          mmF = _mm_loadu_si64((__m64 *)&inptr[16]);
        }
        inptr += RGB_PIXELSIZE * 8;
      }
      mmD = _mm_srli_si64(mmA, 4 * BYTE_BIT);
      mmA = _mm_slli_si64(mmA, 4 * BYTE_BIT);

      mmA = _mm_unpackhi_pi8(mmA, mmG);
      mmG = _mm_slli_si64(mmG, 4 * BYTE_BIT);

      mmD = _mm_unpacklo_pi8(mmD, mmF);
      mmG = _mm_unpackhi_pi8(mmG, mmF);

      mmE = _mm_srli_si64(mmA, 4 * BYTE_BIT);
      mmA = _mm_slli_si64(mmA, 4 * BYTE_BIT);

      mmA = _mm_unpackhi_pi8(mmA, mmD);
      mmD = _mm_slli_si64(mmD, 4 * BYTE_BIT);

      mmE = _mm_unpacklo_pi8(mmE, mmG);
      mmD = _mm_unpackhi_pi8(mmD, mmG);
      mmC = _mm_loadhi_pi8_f(mmA);
      mmA = _mm_loadlo_pi8_f(mmA);

      mmB = _mm_loadhi_pi8_f(mmE);
      mmE = _mm_loadlo_pi8_f(mmE);

      mmF = _mm_loadhi_pi8_f(mmD);
      mmD = _mm_loadlo_pi8_f(mmD);

#else  /* RGB_PIXELSIZE == 4 */

      if (num_cols < 8) {
        col = num_cols;
        asm(".set noreorder\r\n"

            "li       $8, 1\r\n"
            "move     $9, %4\r\n"
            "and      $10, $9, $8\r\n"
            "beqz     $10, 1f\r\n"
            "nop      \r\n"
            "subu     $9, $9, 1\r\n"
            PTR_SLL   "$11, $9, 2\r\n"
            "move     $13, %5\r\n"
            PTR_ADDU  "$13, $13, $11\r\n"
            "lwc1     %0, 0($13)\r\n"

            "1:       \r\n"
            "li       $8, 2\r\n"
            "and      $10, $9, $8\r\n"
            "beqz     $10, 2f\r\n"
            "nop      \r\n"
            "subu     $9, $9, 2\r\n"
            PTR_SLL   "$11, $9, 2\r\n"
            "move     $13, %5\r\n"
            PTR_ADDU  "$13, $13, $11\r\n"
            "mov.s    %1, %0\r\n"
            "ldc1     %0, 0($13)\r\n"

            "2:       \r\n"
            "li       $8, 4\r\n"
            "and      $10, $9, $8\r\n"
            "beqz     $10, 3f\r\n"
            "nop      \r\n"
            "mov.s    %2, %0\r\n"
            "mov.s    %3, %1\r\n"
            "ldc1     %0, 0(%5)\r\n"
            "ldc1     %1, 8(%5)\r\n"

            "3:       \r\n"
            "nop      \r\n"
            ".set reorder\r\n"

            : "=f" (mmA), "=f" (mmF), "=f" (mmD), "=f" (mmC)
            : "r" (col), "r" (inptr)
            : "$f0", "$f2", "$8", "$9", "$10", "$11", "$13", "memory"
           );
      } else {
        if (!(((long)inptr) & 7)) {
          mmA = _mm_load_si64((__m64 *)&inptr[0]);
          mmF = _mm_load_si64((__m64 *)&inptr[8]);
          mmD = _mm_load_si64((__m64 *)&inptr[16]);
          mmC = _mm_load_si64((__m64 *)&inptr[24]);
        } else {
          mmA = _mm_loadu_si64((__m64 *)&inptr[0]);
          mmF = _mm_loadu_si64((__m64 *)&inptr[8]);
          mmD = _mm_loadu_si64((__m64 *)&inptr[16]);
          mmC = _mm_loadu_si64((__m64 *)&inptr[24]);
        }
        inptr += RGB_PIXELSIZE * 8;
      }
      mmB = _mm_unpackhi_pi8(mmA, mmF);
      mmA = _mm_unpacklo_pi8(mmA, mmF);

      mmG = _mm_unpackhi_pi8(mmD, mmC);
      mmD = _mm_unpacklo_pi8(mmD, mmC);

      mmE = _mm_unpackhi_pi16(mmA, mmD);
      mmA = _mm_unpacklo_pi16(mmA, mmD);

      mmH = _mm_unpackhi_pi16(mmB, mmG);
      mmB = _mm_unpacklo_pi16(mmB, mmG);

      mmC = _mm_loadhi_pi8_f(mmA);
      mmA = _mm_loadlo_pi8_f(mmA);

      mmD = _mm_loadhi_pi8_f(mmB);
      mmB = _mm_loadlo_pi8_f(mmB);

      mmG = _mm_loadhi_pi8_f(mmE);
      mmE = _mm_loadlo_pi8_f(mmE);

      mmF = _mm_unpacklo_pi8(mmH, mmH);
      mmH = _mm_unpackhi_pi8(mmH, mmH);
      mmF = _mm_srli_pi16(mmF, BYTE_BIT);
      mmH = _mm_srli_pi16(mmH, BYTE_BIT);

#endif

      /* re=(R0 R2 R4 R6), ge=(G0 G2 G4 G6), be=(B0 B2 B4 B6)
       * ro=(R1 R3 R5 R7), go=(G1 G3 G5 G7), bo=(B1 B3 B5 B7)
       *
       * (Original)
       * Y  =  0.29900 * R + 0.58700 * G + 0.11400 * B
       * Cb = -0.16874 * R - 0.33126 * G + 0.50000 * B + CENTERJSAMPLE
       * Cr =  0.50000 * R - 0.41869 * G - 0.08131 * B + CENTERJSAMPLE
       *
       * (This implementation)
       * Y  =  0.29900 * R + 0.33700 * G + 0.11400 * B + 0.25000 * G
       * Cb = -0.16874 * R - 0.33126 * G + 0.50000 * B + CENTERJSAMPLE
       * Cr =  0.50000 * R - 0.41869 * G - 0.08131 * B + CENTERJSAMPLE
       */

      rglo = _mm_unpacklo_pi16(ro, go);
      rgho = _mm_unpackhi_pi16(ro, go);
      ylo_rg = _mm_madd_pi16(rglo, PW_F0299_F0337);
      yho_rg = _mm_madd_pi16(rgho, PW_F0299_F0337);
      cblo = _mm_madd_pi16(rglo, PW_MF016_MF033);
      cbho = _mm_madd_pi16(rgho, PW_MF016_MF033);

      blo = _mm_loadlo_pi16_f(bo);
      bho = _mm_loadhi_pi16_f(bo);
      halfblo = _mm_srli_pi32(blo, 1);
      halfbho = _mm_srli_pi32(bho, 1);

      cblo = _mm_add_pi32(cblo, halfblo);
      cbho = _mm_add_pi32(cbho, halfbho);
      cblo = _mm_add_pi32(cblo, PD_ONEHALFM1_CJ);
      cbho = _mm_add_pi32(cbho, PD_ONEHALFM1_CJ);
      cblo = _mm_srli_pi32(cblo, SCALEBITS);
      cbho = _mm_srli_pi32(cbho, SCALEBITS);
      cbo = _mm_packs_pi32(cblo, cbho);

      rgle = _mm_unpacklo_pi16(re, ge);
      rghe = _mm_unpackhi_pi16(re, ge);
      yle_rg = _mm_madd_pi16(rgle, PW_F0299_F0337);
      yhe_rg = _mm_madd_pi16(rghe, PW_F0299_F0337);
      cble = _mm_madd_pi16(rgle, PW_MF016_MF033);
      cbhe = _mm_madd_pi16(rghe, PW_MF016_MF033);

      ble = _mm_loadlo_pi16_f(be);
      bhe = _mm_loadhi_pi16_f(be);
      halfble = _mm_srli_pi32(ble, 1);
      halfbhe = _mm_srli_pi32(bhe, 1);

      cble = _mm_add_pi32(cble, halfble);
      cbhe = _mm_add_pi32(cbhe, halfbhe);
      cble = _mm_add_pi32(cble, PD_ONEHALFM1_CJ);
      cbhe = _mm_add_pi32(cbhe, PD_ONEHALFM1_CJ);
      cble = _mm_srli_pi32(cble, SCALEBITS);
      cbhe = _mm_srli_pi32(cbhe, SCALEBITS);
      cbe = _mm_packs_pi32(cble, cbhe);

      cbo = _mm_slli_pi16(cbo, BYTE_BIT);
      cb = _mm_or_si64(cbe, cbo);

      bglo = _mm_unpacklo_pi16(bo, go);
      bgho = _mm_unpackhi_pi16(bo, go);
      ylo_bg = _mm_madd_pi16(bglo, PW_F0114_F0250);
      yho_bg = _mm_madd_pi16(bgho, PW_F0114_F0250);
      crlo = _mm_madd_pi16(bglo, PW_MF008_MF041);
      crho = _mm_madd_pi16(bgho, PW_MF008_MF041);

      ylo = _mm_add_pi32(ylo_bg, ylo_rg);
      yho = _mm_add_pi32(yho_bg, yho_rg);
      ylo = _mm_add_pi32(ylo, PD_ONEHALF);
      yho = _mm_add_pi32(yho, PD_ONEHALF);
      ylo = _mm_srli_pi32(ylo, SCALEBITS);
      yho = _mm_srli_pi32(yho, SCALEBITS);
      yo = _mm_packs_pi32(ylo, yho);

      rlo = _mm_loadlo_pi16_f(ro);
      rho = _mm_loadhi_pi16_f(ro);
      halfrlo = _mm_srli_pi32(rlo, 1);
      halfrho = _mm_srli_pi32(rho, 1);

      crlo = _mm_add_pi32(crlo, halfrlo);
      crho = _mm_add_pi32(crho, halfrho);
      crlo = _mm_add_pi32(crlo, PD_ONEHALFM1_CJ);
      crho = _mm_add_pi32(crho, PD_ONEHALFM1_CJ);
      crlo = _mm_srli_pi32(crlo, SCALEBITS);
      crho = _mm_srli_pi32(crho, SCALEBITS);
      cro = _mm_packs_pi32(crlo, crho);

      bgle = _mm_unpacklo_pi16(be, ge);
      bghe = _mm_unpackhi_pi16(be, ge);
      yle_bg = _mm_madd_pi16(bgle, PW_F0114_F0250);
      yhe_bg = _mm_madd_pi16(bghe, PW_F0114_F0250);
      crle = _mm_madd_pi16(bgle, PW_MF008_MF041);
      crhe = _mm_madd_pi16(bghe, PW_MF008_MF041);

      yle = _mm_add_pi32(yle_bg, yle_rg);
      yhe = _mm_add_pi32(yhe_bg, yhe_rg);
      yle = _mm_add_pi32(yle, PD_ONEHALF);
      yhe = _mm_add_pi32(yhe, PD_ONEHALF);
      yle = _mm_srli_pi32(yle, SCALEBITS);
      yhe = _mm_srli_pi32(yhe, SCALEBITS);
      ye = _mm_packs_pi32(yle, yhe);

      yo = _mm_slli_pi16(yo, BYTE_BIT);
      y = _mm_or_si64(ye, yo);

      rle = _mm_loadlo_pi16_f(re);
      rhe = _mm_loadhi_pi16_f(re);
      halfrle = _mm_srli_pi32(rle, 1);
      halfrhe = _mm_srli_pi32(rhe, 1);

      crle = _mm_add_pi32(crle, halfrle);
      crhe = _mm_add_pi32(crhe, halfrhe);
      crle = _mm_add_pi32(crle, PD_ONEHALFM1_CJ);
      crhe = _mm_add_pi32(crhe, PD_ONEHALFM1_CJ);
      crle = _mm_srli_pi32(crle, SCALEBITS);
      crhe = _mm_srli_pi32(crhe, SCALEBITS);
      cre = _mm_packs_pi32(crle, crhe);

      cro = _mm_slli_pi16(cro, BYTE_BIT);
      cr = _mm_or_si64(cre, cro);

      _mm_store_si64((__m64 *)&outptr0[0], y);
      _mm_store_si64((__m64 *)&outptr1[0], cb);
      _mm_store_si64((__m64 *)&outptr2[0], cr);
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
