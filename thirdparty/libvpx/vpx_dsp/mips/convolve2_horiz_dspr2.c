/*
 *  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>
#include <stdio.h>

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/mips/convolve_common_dspr2.h"
#include "vpx_dsp/vpx_convolve.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_ports/mem.h"

#if HAVE_DSPR2
static void convolve_bi_horiz_4_dspr2(const uint8_t *src, int32_t src_stride,
                                      uint8_t *dst, int32_t dst_stride,
                                      const int16_t *filter_x0, int32_t h) {
  int32_t y;
  uint8_t *cm = vpx_ff_cropTbl;
  int32_t Temp1, Temp2, Temp3, Temp4;
  uint32_t vector4a = 64;
  uint32_t tp1, tp2;
  uint32_t p1, p2;
  const int16_t *filter = &filter_x0[3];
  uint32_t filter45;

  filter45 = ((const int32_t *)filter)[0];

  for (y = h; y--;) {
    /* prefetch data to cache memory */
    prefetch_load(src + src_stride);
    prefetch_load(src + src_stride + 32);
    prefetch_store(dst + dst_stride);

    __asm__ __volatile__(
        "ulw              %[tp1],      0(%[src])                      \n\t"
        "ulw              %[tp2],      4(%[src])                      \n\t"

        /* even 1. pixel */
        "mtlo             %[vector4a], $ac3                           \n\t"
        "mthi             $zero,       $ac3                           \n\t"
        "preceu.ph.qbr    %[p1],       %[tp1]                         \n\t"
        "preceu.ph.qbl    %[p2],       %[tp1]                         \n\t"
        "dpa.w.ph         $ac3,        %[p1],          %[filter45]    \n\t"
        "extp             %[Temp1],    $ac3,           31             \n\t"

        /* even 2. pixel */
        "mtlo             %[vector4a], $ac2                           \n\t"
        "mthi             $zero,       $ac2                           \n\t"
        "balign           %[tp2],      %[tp1],         3              \n\t"
        "dpa.w.ph         $ac2,        %[p2],          %[filter45]    \n\t"
        "extp             %[Temp3],    $ac2,           31             \n\t"

        /* odd 1. pixel */
        "lbux             %[tp1],      %[Temp1](%[cm])                \n\t"
        "mtlo             %[vector4a], $ac3                           \n\t"
        "mthi             $zero,       $ac3                           \n\t"
        "preceu.ph.qbr    %[p1],       %[tp2]                         \n\t"
        "preceu.ph.qbl    %[p2],       %[tp2]                         \n\t"
        "dpa.w.ph         $ac3,        %[p1],          %[filter45]    \n\t"
        "extp             %[Temp2],    $ac3,           31             \n\t"

        /* odd 2. pixel */
        "lbux             %[tp2],      %[Temp3](%[cm])                \n\t"
        "mtlo             %[vector4a], $ac2                           \n\t"
        "mthi             $zero,       $ac2                           \n\t"
        "dpa.w.ph         $ac2,        %[p2],          %[filter45]    \n\t"
        "extp             %[Temp4],    $ac2,           31             \n\t"

        /* clamp */
        "lbux             %[p1],       %[Temp2](%[cm])                \n\t"
        "lbux             %[p2],       %[Temp4](%[cm])                \n\t"

        /* store bytes */
        "sb               %[tp1],      0(%[dst])                      \n\t"
        "sb               %[p1],       1(%[dst])                      \n\t"
        "sb               %[tp2],      2(%[dst])                      \n\t"
        "sb               %[p2],       3(%[dst])                      \n\t"

        : [tp1] "=&r"(tp1), [tp2] "=&r"(tp2), [p1] "=&r"(p1), [p2] "=&r"(p2),
          [Temp1] "=&r"(Temp1), [Temp2] "=&r"(Temp2), [Temp3] "=&r"(Temp3),
          [Temp4] "=&r"(Temp4)
        : [filter45] "r"(filter45), [vector4a] "r"(vector4a), [cm] "r"(cm),
          [dst] "r"(dst), [src] "r"(src));

    /* Next row... */
    src += src_stride;
    dst += dst_stride;
  }
}

static void convolve_bi_horiz_8_dspr2(const uint8_t *src, int32_t src_stride,
                                      uint8_t *dst, int32_t dst_stride,
                                      const int16_t *filter_x0, int32_t h) {
  int32_t y;
  uint8_t *cm = vpx_ff_cropTbl;
  uint32_t vector4a = 64;
  int32_t Temp1, Temp2, Temp3;
  uint32_t tp1, tp2, tp3;
  uint32_t p1, p2, p3, p4;
  uint32_t st0, st1;
  const int16_t *filter = &filter_x0[3];
  uint32_t filter45;

  filter45 = ((const int32_t *)filter)[0];

  for (y = h; y--;) {
    /* prefetch data to cache memory */
    prefetch_load(src + src_stride);
    prefetch_load(src + src_stride + 32);
    prefetch_store(dst + dst_stride);

    __asm__ __volatile__(
        "ulw              %[tp1],      0(%[src])                      \n\t"
        "ulw              %[tp2],      4(%[src])                      \n\t"

        /* even 1. pixel */
        "mtlo             %[vector4a], $ac3                           \n\t"
        "mthi             $zero,       $ac3                           \n\t"
        "mtlo             %[vector4a], $ac2                           \n\t"
        "mthi             $zero,       $ac2                           \n\t"
        "preceu.ph.qbr    %[p1],       %[tp1]                         \n\t"
        "preceu.ph.qbl    %[p2],       %[tp1]                         \n\t"
        "preceu.ph.qbr    %[p3],       %[tp2]                         \n\t"
        "preceu.ph.qbl    %[p4],       %[tp2]                         \n\t"
        "ulw              %[tp3],      8(%[src])                      \n\t"
        "dpa.w.ph         $ac3,        %[p1],          %[filter45]    \n\t"
        "extp             %[Temp1],    $ac3,           31             \n\t"

        /* even 2. pixel */
        "dpa.w.ph         $ac2,        %[p2],          %[filter45]    \n\t"
        "extp             %[Temp3],    $ac2,           31             \n\t"

        /* even 3. pixel */
        "lbux             %[st0],      %[Temp1](%[cm])                \n\t"
        "mtlo             %[vector4a], $ac1                           \n\t"
        "mthi             $zero,       $ac1                           \n\t"
        "dpa.w.ph         $ac1,        %[p3],          %[filter45]    \n\t"
        "extp             %[Temp1],    $ac1,           31             \n\t"

        /* even 4. pixel */
        "mtlo             %[vector4a], $ac2                           \n\t"
        "mthi             $zero,       $ac2                           \n\t"
        "mtlo             %[vector4a], $ac3                           \n\t"
        "mthi             $zero,       $ac3                           \n\t"
        "sb               %[st0],      0(%[dst])                      \n\t"
        "lbux             %[st1],      %[Temp3](%[cm])                \n\t"

        "balign           %[tp3],      %[tp2],         3              \n\t"
        "balign           %[tp2],      %[tp1],         3              \n\t"

        "dpa.w.ph         $ac2,        %[p4],          %[filter45]    \n\t"
        "extp             %[Temp3],    $ac2,           31             \n\t"

        "lbux             %[st0],      %[Temp1](%[cm])                \n\t"

        /* odd 1. pixel */
        "mtlo             %[vector4a], $ac1                           \n\t"
        "mthi             $zero,       $ac1                           \n\t"
        "sb               %[st1],      2(%[dst])                      \n\t"
        "preceu.ph.qbr    %[p1],       %[tp2]                         \n\t"
        "preceu.ph.qbl    %[p2],       %[tp2]                         \n\t"
        "preceu.ph.qbr    %[p3],       %[tp3]                         \n\t"
        "preceu.ph.qbl    %[p4],       %[tp3]                         \n\t"
        "sb               %[st0],      4(%[dst])                      \n\t"
        "dpa.w.ph         $ac3,        %[p1],          %[filter45]    \n\t"
        "extp             %[Temp2],    $ac3,           31             \n\t"

        /* odd 2. pixel */
        "mtlo             %[vector4a], $ac3                           \n\t"
        "mthi             $zero,       $ac3                           \n\t"
        "mtlo             %[vector4a], $ac2                           \n\t"
        "mthi             $zero,       $ac2                           \n\t"
        "lbux             %[st0],      %[Temp3](%[cm])                \n\t"
        "dpa.w.ph         $ac1,        %[p2],          %[filter45]    \n\t"
        "extp             %[Temp3],    $ac1,           31             \n\t"

        /* odd 3. pixel */
        "lbux             %[st1],      %[Temp2](%[cm])                \n\t"
        "dpa.w.ph         $ac3,        %[p3],          %[filter45]    \n\t"
        "extp             %[Temp2],    $ac3,           31             \n\t"

        /* odd 4. pixel */
        "sb               %[st1],      1(%[dst])                      \n\t"
        "sb               %[st0],      6(%[dst])                      \n\t"
        "dpa.w.ph         $ac2,        %[p4],          %[filter45]    \n\t"
        "extp             %[Temp1],    $ac2,           31             \n\t"

        /* clamp */
        "lbux             %[p4],       %[Temp3](%[cm])                \n\t"
        "lbux             %[p2],       %[Temp2](%[cm])                \n\t"
        "lbux             %[p1],       %[Temp1](%[cm])                \n\t"

        /* store bytes */
        "sb               %[p4],       3(%[dst])                      \n\t"
        "sb               %[p2],       5(%[dst])                      \n\t"
        "sb               %[p1],       7(%[dst])                      \n\t"

        : [tp1] "=&r"(tp1), [tp2] "=&r"(tp2), [tp3] "=&r"(tp3),
          [st0] "=&r"(st0), [st1] "=&r"(st1), [p1] "=&r"(p1), [p2] "=&r"(p2),
          [p3] "=&r"(p3), [p4] "=&r"(p4), [Temp1] "=&r"(Temp1),
          [Temp2] "=&r"(Temp2), [Temp3] "=&r"(Temp3)
        : [filter45] "r"(filter45), [vector4a] "r"(vector4a), [cm] "r"(cm),
          [dst] "r"(dst), [src] "r"(src));

    /* Next row... */
    src += src_stride;
    dst += dst_stride;
  }
}

static void convolve_bi_horiz_16_dspr2(const uint8_t *src_ptr,
                                       int32_t src_stride, uint8_t *dst_ptr,
                                       int32_t dst_stride,
                                       const int16_t *filter_x0, int32_t h,
                                       int32_t count) {
  int32_t y, c;
  const uint8_t *src;
  uint8_t *dst;
  uint8_t *cm = vpx_ff_cropTbl;
  uint32_t vector_64 = 64;
  int32_t Temp1, Temp2, Temp3;
  uint32_t qload1, qload2, qload3;
  uint32_t p1, p2, p3, p4, p5;
  uint32_t st1, st2, st3;
  const int16_t *filter = &filter_x0[3];
  uint32_t filter45;

  filter45 = ((const int32_t *)filter)[0];

  for (y = h; y--;) {
    src = src_ptr;
    dst = dst_ptr;

    /* prefetch data to cache memory */
    prefetch_load(src_ptr + src_stride);
    prefetch_load(src_ptr + src_stride + 32);
    prefetch_store(dst_ptr + dst_stride);

    for (c = 0; c < count; c++) {
      __asm__ __volatile__(
          "ulw              %[qload1],    0(%[src])                    \n\t"
          "ulw              %[qload2],    4(%[src])                    \n\t"

          /* even 1. pixel */
          "mtlo             %[vector_64], $ac1                         \n\t" /* even 1 */
          "mthi             $zero,        $ac1                         \n\t"
          "mtlo             %[vector_64], $ac2                         \n\t" /* even 2 */
          "mthi             $zero,        $ac2                         \n\t"
          "preceu.ph.qbr    %[p1],        %[qload1]                    \n\t"
          "preceu.ph.qbl    %[p2],        %[qload1]                    \n\t"
          "preceu.ph.qbr    %[p3],        %[qload2]                    \n\t"
          "preceu.ph.qbl    %[p4],        %[qload2]                    \n\t"
          "ulw              %[qload3],    8(%[src])                    \n\t"
          "dpa.w.ph         $ac1,         %[p1],          %[filter45]  \n\t" /* even 1 */
          "extp             %[Temp1],     $ac1,           31           \n\t" /* even 1 */

          /* even 2. pixel */
          "mtlo             %[vector_64], $ac3                         \n\t" /* even 3 */
          "mthi             $zero,        $ac3                         \n\t"
          "preceu.ph.qbr    %[p1],        %[qload3]                    \n\t"
          "preceu.ph.qbl    %[p5],        %[qload3]                    \n\t"
          "ulw              %[qload1],    12(%[src])                   \n\t"
          "dpa.w.ph         $ac2,         %[p2],          %[filter45]  \n\t" /* even 1 */
          "extp             %[Temp2],     $ac2,           31           \n\t" /* even 1 */
          "lbux             %[st1],       %[Temp1](%[cm])              \n\t" /* even 1 */

          /* even 3. pixel */
          "mtlo             %[vector_64], $ac1                         \n\t" /* even 4 */
          "mthi             $zero,        $ac1                         \n\t"
          "preceu.ph.qbr    %[p2],        %[qload1]                    \n\t"
          "sb               %[st1],       0(%[dst])                    \n\t" /* even 1 */
          "dpa.w.ph         $ac3,         %[p3],          %[filter45]  \n\t" /* even 3 */
          "extp             %[Temp3],     $ac3,           31           \n\t" /* even 3 */
          "lbux             %[st2],       %[Temp2](%[cm])              \n\t" /* even 1 */

          /* even 4. pixel */
          "mtlo             %[vector_64], $ac2                         \n\t" /* even 5 */
          "mthi             $zero,        $ac2                         \n\t"
          "preceu.ph.qbl    %[p3],        %[qload1]                    \n\t"
          "sb               %[st2],       2(%[dst])                    \n\t" /* even 1 */
          "dpa.w.ph         $ac1,         %[p4],          %[filter45]  \n\t" /* even 4 */
          "extp             %[Temp1],     $ac1,           31           \n\t" /* even 4 */
          "lbux             %[st3],       %[Temp3](%[cm])              \n\t" /* even 3 */

          /* even 5. pixel */
          "mtlo             %[vector_64], $ac3                         \n\t" /* even 6 */
          "mthi             $zero,        $ac3                         \n\t"
          "sb               %[st3],       4(%[dst])                    \n\t" /* even 3 */
          "dpa.w.ph         $ac2,         %[p1],          %[filter45]  \n\t" /* even 5 */
          "extp             %[Temp2],     $ac2,           31           \n\t" /* even 5 */
          "lbux             %[st1],       %[Temp1](%[cm])              \n\t" /* even 4 */

          /* even 6. pixel */
          "mtlo             %[vector_64], $ac1                         \n\t" /* even 7 */
          "mthi             $zero,        $ac1                         \n\t"
          "sb               %[st1],       6(%[dst])                    \n\t" /* even 4 */
          "dpa.w.ph         $ac3,         %[p5],          %[filter45]  \n\t" /* even 6 */
          "extp             %[Temp3],     $ac3,           31           \n\t" /* even 6 */
          "lbux             %[st2],       %[Temp2](%[cm])              \n\t" /* even 5 */

          /* even 7. pixel */
          "mtlo             %[vector_64], $ac2                         \n\t" /* even 8 */
          "mthi             $zero,        $ac2                         \n\t"
          "sb               %[st2],       8(%[dst])                    \n\t" /* even 5 */
          "dpa.w.ph         $ac1,         %[p2],          %[filter45]  \n\t" /* even 7 */
          "extp             %[Temp1],     $ac1,           31           \n\t" /* even 7 */
          "lbux             %[st3],       %[Temp3](%[cm])              \n\t" /* even 6 */

          /* even 8. pixel */
          "mtlo             %[vector_64], $ac3                         \n\t" /* odd 1 */
          "mthi             $zero,        $ac3                         \n\t"
          "dpa.w.ph         $ac2,         %[p3],          %[filter45]  \n\t" /* even 8 */
          "sb               %[st3],       10(%[dst])                   \n\t" /* even 6 */
          "extp             %[Temp2],     $ac2,           31           \n\t" /* even 8 */
          "lbux             %[st1],       %[Temp1](%[cm])              \n\t" /* even 7 */

          /* ODD pixels */
          "ulw              %[qload1],    1(%[src])                    \n\t"
          "ulw              %[qload2],    5(%[src])                    \n\t"

          /* odd 1. pixel */
          "mtlo             %[vector_64], $ac1                         \n\t" /* odd 2 */
          "mthi             $zero,        $ac1                         \n\t"
          "preceu.ph.qbr    %[p1],        %[qload1]                    \n\t"
          "preceu.ph.qbl    %[p2],        %[qload1]                    \n\t"
          "preceu.ph.qbr    %[p3],        %[qload2]                    \n\t"
          "preceu.ph.qbl    %[p4],        %[qload2]                    \n\t"
          "sb               %[st1],       12(%[dst])                   \n\t" /* even 7 */
          "ulw              %[qload3],    9(%[src])                    \n\t"
          "dpa.w.ph         $ac3,         %[p1],          %[filter45]  \n\t" /* odd 1 */
          "extp             %[Temp3],     $ac3,           31           \n\t" /* odd 1 */
          "lbux             %[st2],       %[Temp2](%[cm])              \n\t" /* even 8 */

          /* odd 2. pixel */
          "mtlo             %[vector_64], $ac2                         \n\t" /* odd 3 */
          "mthi             $zero,        $ac2                         \n\t"
          "preceu.ph.qbr    %[p1],        %[qload3]                    \n\t"
          "preceu.ph.qbl    %[p5],        %[qload3]                    \n\t"
          "sb               %[st2],       14(%[dst])                   \n\t" /* even 8 */
          "ulw              %[qload1],    13(%[src])                   \n\t"
          "dpa.w.ph         $ac1,         %[p2],          %[filter45]  \n\t" /* odd 2 */
          "extp             %[Temp1],     $ac1,           31           \n\t" /* odd 2 */
          "lbux             %[st3],       %[Temp3](%[cm])              \n\t" /* odd 1 */

          /* odd 3. pixel */
          "mtlo             %[vector_64], $ac3                         \n\t" /* odd 4 */
          "mthi             $zero,        $ac3                         \n\t"
          "preceu.ph.qbr    %[p2],        %[qload1]                    \n\t"
          "sb               %[st3],       1(%[dst])                    \n\t" /* odd 1 */
          "dpa.w.ph         $ac2,         %[p3],          %[filter45]  \n\t" /* odd 3 */
          "extp             %[Temp2],     $ac2,           31           \n\t" /* odd 3 */
          "lbux             %[st1],       %[Temp1](%[cm])              \n\t" /* odd 2 */

          /* odd 4. pixel */
          "mtlo             %[vector_64], $ac1                         \n\t" /* odd 5 */
          "mthi             $zero,        $ac1                         \n\t"
          "preceu.ph.qbl    %[p3],        %[qload1]                    \n\t"
          "sb               %[st1],       3(%[dst])                    \n\t" /* odd 2 */
          "dpa.w.ph         $ac3,         %[p4],          %[filter45]  \n\t" /* odd 4 */
          "extp             %[Temp3],     $ac3,           31           \n\t" /* odd 4 */
          "lbux             %[st2],       %[Temp2](%[cm])              \n\t" /* odd 3 */

          /* odd 5. pixel */
          "mtlo             %[vector_64], $ac2                         \n\t" /* odd 6 */
          "mthi             $zero,        $ac2                         \n\t"
          "sb               %[st2],       5(%[dst])                    \n\t" /* odd 3 */
          "dpa.w.ph         $ac1,         %[p1],          %[filter45]  \n\t" /* odd 5 */
          "extp             %[Temp1],     $ac1,           31           \n\t" /* odd 5 */
          "lbux             %[st3],       %[Temp3](%[cm])              \n\t" /* odd 4 */

          /* odd 6. pixel */
          "mtlo             %[vector_64], $ac3                         \n\t" /* odd 7 */
          "mthi             $zero,        $ac3                         \n\t"
          "sb               %[st3],       7(%[dst])                    \n\t" /* odd 4 */
          "dpa.w.ph         $ac2,         %[p5],          %[filter45]  \n\t" /* odd 6 */
          "extp             %[Temp2],     $ac2,           31           \n\t" /* odd 6 */
          "lbux             %[st1],       %[Temp1](%[cm])              \n\t" /* odd 5 */

          /* odd 7. pixel */
          "mtlo             %[vector_64], $ac1                         \n\t" /* odd 8 */
          "mthi             $zero,        $ac1                         \n\t"
          "sb               %[st1],       9(%[dst])                    \n\t" /* odd 5 */
          "dpa.w.ph         $ac3,         %[p2],          %[filter45]  \n\t" /* odd 7 */
          "extp             %[Temp3],     $ac3,           31           \n\t" /* odd 7 */

          /* odd 8. pixel */
          "dpa.w.ph         $ac1,         %[p3],          %[filter45]  \n\t" /* odd 8 */
          "extp             %[Temp1],     $ac1,           31           \n\t" /* odd 8 */

          "lbux             %[st2],       %[Temp2](%[cm])              \n\t" /* odd 6 */
          "lbux             %[st3],       %[Temp3](%[cm])              \n\t" /* odd 7 */
          "lbux             %[st1],       %[Temp1](%[cm])              \n\t" /* odd 8 */

          "sb               %[st2],       11(%[dst])                   \n\t" /* odd 6 */
          "sb               %[st3],       13(%[dst])                   \n\t" /* odd 7 */
          "sb               %[st1],       15(%[dst])                   \n\t" /* odd 8 */

          : [qload1] "=&r"(qload1), [qload2] "=&r"(qload2),
            [qload3] "=&r"(qload3), [st1] "=&r"(st1), [st2] "=&r"(st2),
            [st3] "=&r"(st3), [p1] "=&r"(p1), [p2] "=&r"(p2), [p3] "=&r"(p3),
            [p4] "=&r"(p4), [p5] "=&r"(p5), [Temp1] "=&r"(Temp1),
            [Temp2] "=&r"(Temp2), [Temp3] "=&r"(Temp3)
          : [filter45] "r"(filter45), [vector_64] "r"(vector_64), [cm] "r"(cm),
            [dst] "r"(dst), [src] "r"(src));

      src += 16;
      dst += 16;
    }

    /* Next row... */
    src_ptr += src_stride;
    dst_ptr += dst_stride;
  }
}

static void convolve_bi_horiz_64_dspr2(const uint8_t *src_ptr,
                                       int32_t src_stride, uint8_t *dst_ptr,
                                       int32_t dst_stride,
                                       const int16_t *filter_x0, int32_t h) {
  int32_t y, c;
  const uint8_t *src;
  uint8_t *dst;
  uint8_t *cm = vpx_ff_cropTbl;
  uint32_t vector_64 = 64;
  int32_t Temp1, Temp2, Temp3;
  uint32_t qload1, qload2, qload3;
  uint32_t p1, p2, p3, p4, p5;
  uint32_t st1, st2, st3;
  const int16_t *filter = &filter_x0[3];
  uint32_t filter45;

  filter45 = ((const int32_t *)filter)[0];

  for (y = h; y--;) {
    src = src_ptr;
    dst = dst_ptr;

    /* prefetch data to cache memory */
    prefetch_load(src_ptr + src_stride);
    prefetch_load(src_ptr + src_stride + 32);
    prefetch_load(src_ptr + src_stride + 64);
    prefetch_store(dst_ptr + dst_stride);
    prefetch_store(dst_ptr + dst_stride + 32);

    for (c = 0; c < 4; c++) {
      __asm__ __volatile__(
          "ulw              %[qload1],    0(%[src])                    \n\t"
          "ulw              %[qload2],    4(%[src])                    \n\t"

          /* even 1. pixel */
          "mtlo             %[vector_64], $ac1                         \n\t" /* even 1 */
          "mthi             $zero,        $ac1                         \n\t"
          "mtlo             %[vector_64], $ac2                         \n\t" /* even 2 */
          "mthi             $zero,        $ac2                         \n\t"
          "preceu.ph.qbr    %[p1],        %[qload1]                    \n\t"
          "preceu.ph.qbl    %[p2],        %[qload1]                    \n\t"
          "preceu.ph.qbr    %[p3],        %[qload2]                    \n\t"
          "preceu.ph.qbl    %[p4],        %[qload2]                    \n\t"
          "ulw              %[qload3],    8(%[src])                    \n\t"
          "dpa.w.ph         $ac1,         %[p1],          %[filter45]  \n\t" /* even 1 */
          "extp             %[Temp1],     $ac1,           31           \n\t" /* even 1 */

          /* even 2. pixel */
          "mtlo             %[vector_64], $ac3                         \n\t" /* even 3 */
          "mthi             $zero,        $ac3                         \n\t"
          "preceu.ph.qbr    %[p1],        %[qload3]                    \n\t"
          "preceu.ph.qbl    %[p5],        %[qload3]                    \n\t"
          "ulw              %[qload1],    12(%[src])                   \n\t"
          "dpa.w.ph         $ac2,         %[p2],          %[filter45]  \n\t" /* even 1 */
          "extp             %[Temp2],     $ac2,           31           \n\t" /* even 1 */
          "lbux             %[st1],       %[Temp1](%[cm])              \n\t" /* even 1 */

          /* even 3. pixel */
          "mtlo             %[vector_64], $ac1                         \n\t" /* even 4 */
          "mthi             $zero,        $ac1                         \n\t"
          "preceu.ph.qbr    %[p2],        %[qload1]                    \n\t"
          "sb               %[st1],       0(%[dst])                    \n\t" /* even 1 */
          "dpa.w.ph         $ac3,         %[p3],          %[filter45]  \n\t" /* even 3 */
          "extp             %[Temp3],     $ac3,           31           \n\t" /* even 3 */
          "lbux             %[st2],       %[Temp2](%[cm])              \n\t" /* even 1 */

          /* even 4. pixel */
          "mtlo             %[vector_64], $ac2                         \n\t" /* even 5 */
          "mthi             $zero,        $ac2                         \n\t"
          "preceu.ph.qbl    %[p3],        %[qload1]                    \n\t"
          "sb               %[st2],       2(%[dst])                    \n\t" /* even 1 */
          "dpa.w.ph         $ac1,         %[p4],          %[filter45]  \n\t" /* even 4 */
          "extp             %[Temp1],     $ac1,           31           \n\t" /* even 4 */
          "lbux             %[st3],       %[Temp3](%[cm])              \n\t" /* even 3 */

          /* even 5. pixel */
          "mtlo             %[vector_64], $ac3                         \n\t" /* even 6 */
          "mthi             $zero,        $ac3                         \n\t"
          "sb               %[st3],       4(%[dst])                    \n\t" /* even 3 */
          "dpa.w.ph         $ac2,         %[p1],          %[filter45]  \n\t" /* even 5 */
          "extp             %[Temp2],     $ac2,           31           \n\t" /* even 5 */
          "lbux             %[st1],       %[Temp1](%[cm])              \n\t" /* even 4 */

          /* even 6. pixel */
          "mtlo             %[vector_64], $ac1                         \n\t" /* even 7 */
          "mthi             $zero,        $ac1                         \n\t"
          "sb               %[st1],       6(%[dst])                    \n\t" /* even 4 */
          "dpa.w.ph         $ac3,         %[p5],          %[filter45]  \n\t" /* even 6 */
          "extp             %[Temp3],     $ac3,           31           \n\t" /* even 6 */
          "lbux             %[st2],       %[Temp2](%[cm])              \n\t" /* even 5 */

          /* even 7. pixel */
          "mtlo             %[vector_64], $ac2                         \n\t" /* even 8 */
          "mthi             $zero,        $ac2                         \n\t"
          "sb               %[st2],       8(%[dst])                    \n\t" /* even 5 */
          "dpa.w.ph         $ac1,         %[p2],          %[filter45]  \n\t" /* even 7 */
          "extp             %[Temp1],     $ac1,           31           \n\t" /* even 7 */
          "lbux             %[st3],       %[Temp3](%[cm])              \n\t" /* even 6 */

          /* even 8. pixel */
          "mtlo             %[vector_64], $ac3                         \n\t" /* odd 1 */
          "mthi             $zero,        $ac3                         \n\t"
          "dpa.w.ph         $ac2,         %[p3],          %[filter45]  \n\t" /* even 8 */
          "sb               %[st3],       10(%[dst])                   \n\t" /* even 6 */
          "extp             %[Temp2],     $ac2,           31           \n\t" /* even 8 */
          "lbux             %[st1],       %[Temp1](%[cm])              \n\t" /* even 7 */

          /* ODD pixels */
          "ulw              %[qload1],    1(%[src])                    \n\t"
          "ulw              %[qload2],    5(%[src])                    \n\t"

          /* odd 1. pixel */
          "mtlo             %[vector_64], $ac1                         \n\t" /* odd 2 */
          "mthi             $zero,        $ac1                         \n\t"
          "preceu.ph.qbr    %[p1],        %[qload1]                    \n\t"
          "preceu.ph.qbl    %[p2],        %[qload1]                    \n\t"
          "preceu.ph.qbr    %[p3],        %[qload2]                    \n\t"
          "preceu.ph.qbl    %[p4],        %[qload2]                    \n\t"
          "sb               %[st1],       12(%[dst])                   \n\t" /* even 7 */
          "ulw              %[qload3],    9(%[src])                    \n\t"
          "dpa.w.ph         $ac3,         %[p1],          %[filter45]  \n\t" /* odd 1 */
          "extp             %[Temp3],     $ac3,           31           \n\t" /* odd 1 */
          "lbux             %[st2],       %[Temp2](%[cm])              \n\t" /* even 8 */

          /* odd 2. pixel */
          "mtlo             %[vector_64], $ac2                         \n\t" /* odd 3 */
          "mthi             $zero,        $ac2                         \n\t"
          "preceu.ph.qbr    %[p1],        %[qload3]                    \n\t"
          "preceu.ph.qbl    %[p5],        %[qload3]                    \n\t"
          "sb               %[st2],       14(%[dst])                   \n\t" /* even 8 */
          "ulw              %[qload1],    13(%[src])                   \n\t"
          "dpa.w.ph         $ac1,         %[p2],          %[filter45]  \n\t" /* odd 2 */
          "extp             %[Temp1],     $ac1,           31           \n\t" /* odd 2 */
          "lbux             %[st3],       %[Temp3](%[cm])              \n\t" /* odd 1 */

          /* odd 3. pixel */
          "mtlo             %[vector_64], $ac3                         \n\t" /* odd 4 */
          "mthi             $zero,        $ac3                         \n\t"
          "preceu.ph.qbr    %[p2],        %[qload1]                    \n\t"
          "sb               %[st3],       1(%[dst])                    \n\t" /* odd 1 */
          "dpa.w.ph         $ac2,         %[p3],          %[filter45]  \n\t" /* odd 3 */
          "extp             %[Temp2],     $ac2,           31           \n\t" /* odd 3 */
          "lbux             %[st1],       %[Temp1](%[cm])              \n\t" /* odd 2 */

          /* odd 4. pixel */
          "mtlo             %[vector_64], $ac1                         \n\t" /* odd 5 */
          "mthi             $zero,        $ac1                         \n\t"
          "preceu.ph.qbl    %[p3],        %[qload1]                    \n\t"
          "sb               %[st1],       3(%[dst])                    \n\t" /* odd 2 */
          "dpa.w.ph         $ac3,         %[p4],          %[filter45]  \n\t" /* odd 4 */
          "extp             %[Temp3],     $ac3,           31           \n\t" /* odd 4 */
          "lbux             %[st2],       %[Temp2](%[cm])              \n\t" /* odd 3 */

          /* odd 5. pixel */
          "mtlo             %[vector_64], $ac2                         \n\t" /* odd 6 */
          "mthi             $zero,        $ac2                         \n\t"
          "sb               %[st2],       5(%[dst])                    \n\t" /* odd 3 */
          "dpa.w.ph         $ac1,         %[p1],          %[filter45]  \n\t" /* odd 5 */
          "extp             %[Temp1],     $ac1,           31           \n\t" /* odd 5 */
          "lbux             %[st3],       %[Temp3](%[cm])              \n\t" /* odd 4 */

          /* odd 6. pixel */
          "mtlo             %[vector_64], $ac3                         \n\t" /* odd 7 */
          "mthi             $zero,        $ac3                         \n\t"
          "sb               %[st3],       7(%[dst])                    \n\t" /* odd 4 */
          "dpa.w.ph         $ac2,         %[p5],          %[filter45]  \n\t" /* odd 6 */
          "extp             %[Temp2],     $ac2,           31           \n\t" /* odd 6 */
          "lbux             %[st1],       %[Temp1](%[cm])              \n\t" /* odd 5 */

          /* odd 7. pixel */
          "mtlo             %[vector_64], $ac1                         \n\t" /* odd 8 */
          "mthi             $zero,        $ac1                         \n\t"
          "sb               %[st1],       9(%[dst])                    \n\t" /* odd 5 */
          "dpa.w.ph         $ac3,         %[p2],          %[filter45]  \n\t" /* odd 7 */
          "extp             %[Temp3],     $ac3,           31           \n\t" /* odd 7 */

          /* odd 8. pixel */
          "dpa.w.ph         $ac1,         %[p3],          %[filter45]  \n\t" /* odd 8 */
          "extp             %[Temp1],     $ac1,           31           \n\t" /* odd 8 */

          "lbux             %[st2],       %[Temp2](%[cm])              \n\t" /* odd 6 */
          "lbux             %[st3],       %[Temp3](%[cm])              \n\t" /* odd 7 */
          "lbux             %[st1],       %[Temp1](%[cm])              \n\t" /* odd 8 */

          "sb               %[st2],       11(%[dst])                   \n\t" /* odd 6 */
          "sb               %[st3],       13(%[dst])                   \n\t" /* odd 7 */
          "sb               %[st1],       15(%[dst])                   \n\t" /* odd 8 */

          : [qload1] "=&r"(qload1), [qload2] "=&r"(qload2),
            [qload3] "=&r"(qload3), [st1] "=&r"(st1), [st2] "=&r"(st2),
            [st3] "=&r"(st3), [p1] "=&r"(p1), [p2] "=&r"(p2), [p3] "=&r"(p3),
            [p4] "=&r"(p4), [p5] "=&r"(p5), [Temp1] "=&r"(Temp1),
            [Temp2] "=&r"(Temp2), [Temp3] "=&r"(Temp3)
          : [filter45] "r"(filter45), [vector_64] "r"(vector_64), [cm] "r"(cm),
            [dst] "r"(dst), [src] "r"(src));

      src += 16;
      dst += 16;
    }

    /* Next row... */
    src_ptr += src_stride;
    dst_ptr += dst_stride;
  }
}

void vpx_convolve2_horiz_dspr2(const uint8_t *src, ptrdiff_t src_stride,
                               uint8_t *dst, ptrdiff_t dst_stride,
                               const InterpKernel *filter, int x0_q4,
                               int32_t x_step_q4, int y0_q4, int y_step_q4,
                               int w, int h) {
  const int16_t *const filter_x = filter[x0_q4];
  uint32_t pos = 38;

  assert(x_step_q4 == 16);

  prefetch_load((const uint8_t *)filter_x);

  /* bit positon for extract from acc */
  __asm__ __volatile__("wrdsp      %[pos],     1           \n\t"
                       :
                       : [pos] "r"(pos));

  /* prefetch data to cache memory */
  prefetch_load(src);
  prefetch_load(src + 32);
  prefetch_store(dst);

  switch (w) {
    case 4:
      convolve_bi_horiz_4_dspr2(src, (int32_t)src_stride, dst,
                                (int32_t)dst_stride, filter_x, (int32_t)h);
      break;
    case 8:
      convolve_bi_horiz_8_dspr2(src, (int32_t)src_stride, dst,
                                (int32_t)dst_stride, filter_x, (int32_t)h);
      break;
    case 16:
      convolve_bi_horiz_16_dspr2(src, (int32_t)src_stride, dst,
                                 (int32_t)dst_stride, filter_x, (int32_t)h, 1);
      break;
    case 32:
      convolve_bi_horiz_16_dspr2(src, (int32_t)src_stride, dst,
                                 (int32_t)dst_stride, filter_x, (int32_t)h, 2);
      break;
    case 64:
      prefetch_load(src + 64);
      prefetch_store(dst + 32);

      convolve_bi_horiz_64_dspr2(src, (int32_t)src_stride, dst,
                                 (int32_t)dst_stride, filter_x, (int32_t)h);
      break;
    default:
      vpx_convolve8_horiz_c(src, src_stride, dst, dst_stride, filter, x0_q4,
                            x_step_q4, y0_q4, y_step_q4, w, h);
      break;
  }
}
#endif
