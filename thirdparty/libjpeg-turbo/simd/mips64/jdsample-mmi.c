/*
 * Loongson MMI optimizations for libjpeg-turbo
 *
 * Copyright (C) 2015, 2018-2019, D. R. Commander.  All Rights Reserved.
 * Copyright (C) 2016-2018, Loongson Technology Corporation Limited, BeiJing.
 *                          All Rights Reserved.
 * Authors:  ZhuChen     <zhuchen@loongson.cn>
 *           CaiWanwei   <caiwanwei@loongson.cn>
 *           SunZhangzhi <sunzhangzhi-cq@loongson.cn>
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

/* CHROMA UPSAMPLING */

#include "jsimd_mmi.h"


enum const_index {
  index_PW_ONE,
  index_PW_TWO,
  index_PW_THREE,
  index_PW_SEVEN,
  index_PW_EIGHT,
};

static uint64_t const_value[] = {
  _uint64_set_pi16(1, 1, 1, 1),
  _uint64_set_pi16(2, 2, 2, 2),
  _uint64_set_pi16(3, 3, 3, 3),
  _uint64_set_pi16(7, 7, 7, 7),
  _uint64_set_pi16(8, 8, 8, 8),
};

#define PW_ONE    get_const_value(index_PW_ONE)
#define PW_TWO    get_const_value(index_PW_TWO)
#define PW_THREE  get_const_value(index_PW_THREE)
#define PW_SEVEN  get_const_value(index_PW_SEVEN)
#define PW_EIGHT  get_const_value(index_PW_EIGHT)


#define PROCESS_ROW(row, wkoffset, bias1, bias2, shift) { \
  __m64 samp123X, samp3XXX, samp1234, sampX012, samp_1012; \
  __m64 sampXXX4, sampX456, samp3456, samp567X, samp7XXX, samp5678; \
  __m64 outle, outhe, outlo, outho, outl, outh; \
  \
  samp123X = _mm_srli_si64(samp0123, 2 * BYTE_BIT);  /* ( 1 2 3 -) */ \
  sampXXX4 = _mm_slli_si64(samp4567, (SIZEOF_MMWORD - 2) * BYTE_BIT);  /* ( - - - 4) */ \
  samp3XXX = _mm_srli_si64(samp0123, (SIZEOF_MMWORD - 2) * BYTE_BIT);  /* ( 3 - - -) */ \
  sampX456 = _mm_slli_si64(samp4567, 2 * BYTE_BIT);  /* ( - 4 5 6) */ \
  \
  samp1234 = _mm_or_si64(samp123X, sampXXX4);  /* ( 1 2 3 4) */ \
  samp3456 = _mm_or_si64(samp3XXX, sampX456);  /* ( 3 4 5 6) */ \
  \
  sampX012 = _mm_slli_si64(samp0123, 2 * BYTE_BIT);  /* ( - 0 1 2) */ \
  samp567X = _mm_srli_si64(samp4567, 2 * BYTE_BIT);  /* ( 5 6 7 -) */ \
  samp7XXX = _mm_srli_si64(samp4567, (SIZEOF_MMWORD - 2) * BYTE_BIT);  /* ( 7 - - -) */ \
  \
  samp_1012 = _mm_or_si64(sampX012, wk[row]);            /* (-1 0 1 2) */ \
  samp5678 = _mm_or_si64(samp567X, wk[row + wkoffset]);  /* ( 5 6 7 8) */ \
  \
  wk[row] = samp7XXX; \
  \
  samp0123 = _mm_mullo_pi16(samp0123, PW_THREE); \
  samp4567 = _mm_mullo_pi16(samp4567, PW_THREE); \
  samp_1012 = _mm_add_pi16(samp_1012, bias1); \
  samp3456 = _mm_add_pi16(samp3456, bias1); \
  samp1234 = _mm_add_pi16(samp1234, bias2); \
  samp5678 = _mm_add_pi16(samp5678, bias2); \
  \
  outle = _mm_add_pi16(samp_1012, samp0123); \
  outhe = _mm_add_pi16(samp3456, samp4567); \
  outle = _mm_srli_pi16(outle, shift);        /* ( 0  2  4  6) */ \
  outhe = _mm_srli_pi16(outhe, shift);        /* ( 8 10 12 14) */ \
  outlo = _mm_add_pi16(samp1234, samp0123); \
  outho = _mm_add_pi16(samp5678, samp4567); \
  outlo = _mm_srli_pi16(outlo, shift);        /* ( 1  3  5  7) */ \
  outho = _mm_srli_pi16(outho, shift);        /* ( 9 11 13 15) */ \
  \
  outlo = _mm_slli_pi16(outlo, BYTE_BIT); \
  outho = _mm_slli_pi16(outho, BYTE_BIT); \
  outl = _mm_or_si64(outle, outlo);           /* ( 0  1  2  3  4  5  6  7) */ \
  outh = _mm_or_si64(outhe, outho);           /* ( 8  9 10 11 12 13 14 15) */ \
  \
  _mm_store_si64((__m64 *)outptr##row, outl); \
  _mm_store_si64((__m64 *)outptr##row + 1, outh); \
}

void jsimd_h2v2_fancy_upsample_mmi(int max_v_samp_factor,
                                   JDIMENSION downsampled_width,
                                   JSAMPARRAY input_data,
                                   JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output_data = *output_data_ptr;
  JSAMPROW inptr_1, inptr0, inptr1, outptr0, outptr1;
  int inrow, outrow, incol, tmp, tmp1;
  __m64 this_1l, this_1h, this_1, thiscolsum_1l, thiscolsum_1h;
  __m64 this0l, this0h, this0;
  __m64 this1l, this1h, this1, thiscolsum1l, thiscolsum1h;
  __m64 next_1l, next_1h, next_1, nextcolsum_1l, nextcolsum_1h;
  __m64 next0l, next0h, next0;
  __m64 next1l, next1h, next1, nextcolsum1l, nextcolsum1h;
  __m64 mask0 = 0.0, masklast, samp0123, samp4567, wk[4], zero = 0.0;

  mask0 = _mm_cmpeq_pi8(mask0, mask0);
  masklast = _mm_slli_si64(mask0, (SIZEOF_MMWORD - 2) * BYTE_BIT);
  mask0 = _mm_srli_si64(mask0, (SIZEOF_MMWORD - 2) * BYTE_BIT);

  for (inrow = 0, outrow = 0; outrow < max_v_samp_factor; inrow++) {

    inptr_1 = input_data[inrow - 1];
    inptr0 = input_data[inrow];
    inptr1 = input_data[inrow + 1];
    outptr0 = output_data[outrow++];
    outptr1 = output_data[outrow++];

    if (downsampled_width & 7) {
      tmp = (downsampled_width - 1) * sizeof(JSAMPLE);
      tmp1 = downsampled_width * sizeof(JSAMPLE);
      asm(PTR_ADDU  "$8, %3, %6\r\n"
          "lb       $9, ($8)\r\n"
          PTR_ADDU  "$8, %3, %7\r\n"
          "sb       $9, ($8)\r\n"
          PTR_ADDU  "$8, %4, %6\r\n"
          "lb       $9, ($8)\r\n"
          PTR_ADDU  "$8, %4, %7\r\n"
          "sb       $9, ($8)\r\n"
          PTR_ADDU  "$8, %5, %6\r\n"
          "lb       $9, ($8)\r\n"
          PTR_ADDU  "$8, %5, %7\r\n"
          "sb       $9, ($8)\r\n"
          : "=m" (*inptr_1), "=m" (*inptr0), "=m" (*inptr1)
          : "r" (inptr_1), "r" (inptr0), "r" (inptr1), "r" (tmp), "r" (tmp1)
          : "$8", "$9"
         );
    }

    /* process the first column block */
    this0 = _mm_load_si64((__m64 *)inptr0);    /* row[ 0][0] */
    this_1 = _mm_load_si64((__m64 *)inptr_1);  /* row[-1][0] */
    this1 = _mm_load_si64((__m64 *)inptr1);    /* row[ 1][0] */

    this0l = _mm_unpacklo_pi8(this0, zero);    /* row[ 0][0]( 0 1 2 3) */
    this0h = _mm_unpackhi_pi8(this0, zero);    /* row[ 0][0]( 4 5 6 7) */
    this_1l = _mm_unpacklo_pi8(this_1, zero);  /* row[-1][0]( 0 1 2 3) */
    this_1h = _mm_unpackhi_pi8(this_1, zero);  /* row[-1][0]( 4 5 6 7) */
    this1l = _mm_unpacklo_pi8(this1, zero);    /* row[+1][0]( 0 1 2 3) */
    this1h = _mm_unpackhi_pi8(this1, zero);    /* row[+1][0]( 4 5 6 7) */

    this0l = _mm_mullo_pi16(this0l, PW_THREE);
    this0h = _mm_mullo_pi16(this0h, PW_THREE);

    thiscolsum_1l = _mm_add_pi16(this_1l, this0l);  /* ( 0 1 2 3) */
    thiscolsum_1h = _mm_add_pi16(this_1h, this0h);  /* ( 4 5 6 7) */
    thiscolsum1l = _mm_add_pi16(this0l, this1l);    /* ( 0 1 2 3) */
    thiscolsum1h = _mm_add_pi16(this0h, this1h);    /* ( 4 5 6 7) */

    /* temporarily save the intermediate data */
    _mm_store_si64((__m64 *)outptr0, thiscolsum_1l);
    _mm_store_si64((__m64 *)outptr0 + 1, thiscolsum_1h);
    _mm_store_si64((__m64 *)outptr1, thiscolsum1l);
    _mm_store_si64((__m64 *)outptr1 + 1, thiscolsum1h);

    wk[0] = _mm_and_si64(thiscolsum_1l, mask0);  /* ( 0 - - -) */
    wk[1] = _mm_and_si64(thiscolsum1l, mask0);   /* ( 0 - - -) */

    for (incol = downsampled_width; incol > 0;
         incol -= 8, inptr_1 += 8, inptr0 += 8, inptr1 += 8,
         outptr0 += 16, outptr1 += 16) {

      if (incol > 8) {
        /* process the next column block */
        next0 = _mm_load_si64((__m64 *)inptr0 + 1);    /* row[ 0][1] */
        next_1 = _mm_load_si64((__m64 *)inptr_1 + 1);  /* row[-1][1] */
        next1 = _mm_load_si64((__m64 *)inptr1 + 1);    /* row[+1][1] */

        next0l = _mm_unpacklo_pi8(next0, zero);    /* row[ 0][1]( 0 1 2 3) */
        next0h = _mm_unpackhi_pi8(next0, zero);    /* row[ 0][1]( 4 5 6 7) */
        next_1l = _mm_unpacklo_pi8(next_1, zero);  /* row[-1][1]( 0 1 2 3) */
        next_1h = _mm_unpackhi_pi8(next_1, zero);  /* row[-1][1]( 4 5 6 7) */
        next1l = _mm_unpacklo_pi8(next1, zero);    /* row[+1][1]( 0 1 2 3) */
        next1h = _mm_unpackhi_pi8(next1, zero);    /* row[+1][1]( 4 5 6 7) */

        next0l = _mm_mullo_pi16(next0l, PW_THREE);
        next0h = _mm_mullo_pi16(next0h, PW_THREE);

        nextcolsum_1l = _mm_add_pi16(next_1l, next0l);  /* ( 0 1 2 3) */
        nextcolsum_1h = _mm_add_pi16(next_1h, next0h);  /* ( 4 5 6 7) */
        nextcolsum1l = _mm_add_pi16(next0l, next1l);    /* ( 0 1 2 3) */
        nextcolsum1h = _mm_add_pi16(next0h, next1h);    /* ( 4 5 6 7) */

        /* temporarily save the intermediate data */
        _mm_store_si64((__m64 *)outptr0 + 2, nextcolsum_1l);
        _mm_store_si64((__m64 *)outptr0 + 3, nextcolsum_1h);
        _mm_store_si64((__m64 *)outptr1 + 2, nextcolsum1l);
        _mm_store_si64((__m64 *)outptr1 + 3, nextcolsum1h);

        wk[2] = _mm_slli_si64(nextcolsum_1l, (SIZEOF_MMWORD - 2) * BYTE_BIT);  /* ( - - - 0) */
        wk[3] = _mm_slli_si64(nextcolsum1l, (SIZEOF_MMWORD - 2) * BYTE_BIT);   /* ( - - - 0) */
      } else {
        __m64 tmp;

        /* process the last column block */
        tmp = _mm_load_si64((__m64 *)outptr0 + 1);
        wk[2] = _mm_and_si64(masklast, tmp);        /* ( - - - 7) */
        tmp = _mm_load_si64((__m64 *)outptr1 + 1);
        wk[3] = _mm_and_si64(masklast, tmp);        /* ( - - - 7) */
      }

      /* process the upper row */
      samp0123 = _mm_load_si64((__m64 *)outptr0);      /* ( 0 1 2 3) */ \
      samp4567 = _mm_load_si64((__m64 *)outptr0 + 1);  /* ( 4 5 6 7) */ \
      PROCESS_ROW(0, 2, PW_EIGHT, PW_SEVEN, 4)

      /* process the lower row */
      samp0123 = _mm_load_si64((__m64 *)outptr1);      /* ( 0 1 2 3) */ \
      samp4567 = _mm_load_si64((__m64 *)outptr1 + 1);  /* ( 4 5 6 7) */ \
      PROCESS_ROW(1, 2, PW_EIGHT, PW_SEVEN, 4)
    }
  }
}


void jsimd_h2v1_fancy_upsample_mmi(int max_v_samp_factor,
                                   JDIMENSION downsampled_width,
                                   JSAMPARRAY input_data,
                                   JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output_data = *output_data_ptr;
  JSAMPROW inptr0, outptr0;
  int inrow, incol, tmp, tmp1;
  __m64 thisl, this, nextl, next;
  __m64 mask0 = 0.0, masklast, samp0123, samp4567, wk[2], zero = 0.0;

  mask0 = _mm_cmpeq_pi8(mask0, mask0);
  masklast = _mm_slli_si64(mask0, (SIZEOF_MMWORD - 2) * BYTE_BIT);
  mask0 = _mm_srli_si64(mask0, (SIZEOF_MMWORD - 2) * BYTE_BIT);

  for (inrow = 0; inrow < max_v_samp_factor; inrow++) {

    inptr0 = input_data[inrow];
    outptr0 = output_data[inrow];

    if (downsampled_width & 7) {
      tmp = (downsampled_width - 1) * sizeof(JSAMPLE);
      tmp1 = downsampled_width * sizeof(JSAMPLE);
      asm(PTR_ADDU  "$8, %1, %2\r\n"
          "lb       $9, ($8)\r\n"
          PTR_ADDU  "$8, %1, %3\r\n"
          "sb       $9, ($8)\r\n"
          : "=m" (*inptr0)
          : "r" (inptr0), "r" (tmp), "r" (tmp1)
          : "$8", "$9"
         );
    }

    /* process the first column block */
    this = _mm_load_si64((__m64 *)inptr0);    /* row[ 0][0] */
    thisl = _mm_unpacklo_pi8(this, zero);     /* row[ 0][0]( 0 1 2 3) */
    wk[0] = _mm_and_si64(thisl, mask0);       /* ( 0 - - -) */

    for (incol = downsampled_width; incol > 0;
         incol -= 8, inptr0 += 8, outptr0 += 16) {

      if (incol > 8) {
        /* process the next column block */
        next = _mm_load_si64((__m64 *)inptr0 + 1);  /* row[ 0][1] */
        nextl = _mm_unpacklo_pi8(next, zero);       /* row[ 0][1]( 0 1 2 3) */
        wk[1] = _mm_slli_si64(nextl, (SIZEOF_MMWORD - 2) * BYTE_BIT);  /* ( - - - 0) */
      } else {
        __m64 thish;

        /* process the last column block */
        this = _mm_load_si64((__m64 *)inptr0);  /* row[ 0][0] */
        thish = _mm_unpackhi_pi8(this, zero);   /* row[ 0][1]( 4 5 6 7) */
        wk[1] = _mm_and_si64(masklast, thish);  /* ( - - - 7) */
      }

      /* process the row */
      this = _mm_load_si64((__m64 *)inptr0);    /* row[ 0][0] */
      samp0123 = _mm_unpacklo_pi8(this, zero);  /* ( 0 1 2 3) */
      samp4567 = _mm_unpackhi_pi8(this, zero);  /* ( 4 5 6 7) */
      PROCESS_ROW(0, 1, PW_ONE, PW_TWO, 2)
    }
  }
}
