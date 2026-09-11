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
#include "vpx_dsp/vpx_filter.h"
#include "vpx_ports/mem.h"

#if HAVE_DSPR2
static void convolve_avg_horiz_4_dspr2(const uint8_t *src, int32_t src_stride,
                                       uint8_t *dst, int32_t dst_stride,
                                       const int16_t *filter_x0, int32_t h) {
  int32_t y;
  uint8_t *cm = vpx_ff_cropTbl;
  int32_t vector1b, vector2b, vector3b, vector4b;
  int32_t Temp1, Temp2, Temp3, Temp4;
  uint32_t vector4a = 64;
  uint32_t tp1, tp2;
  uint32_t p1, p2, p3, p4;
  uint32_t n1, n2, n3, n4;
  uint32_t tn1, tn2;

  vector1b = ((const int32_t *)filter_x0)[0];
  vector2b = ((const int32_t *)filter_x0)[1];
  vector3b = ((const int32_t *)filter_x0)[2];
  vector4b = ((const int32_t *)filter_x0)[3];

  for (y = h; y--;) {
    /* prefetch data to cache memory */
    prefetch_load(src + src_stride);
    prefetch_load(src + src_stride + 32);
    prefetch_store(dst + dst_stride);

    __asm__ __volatile__(
        "ulw              %[tp1],         0(%[src])                      \n\t"
        "ulw              %[tp2],         4(%[src])                      \n\t"

        /* even 1. pixel */
        "mtlo             %[vector4a],    $ac3                           \n\t"
        "mthi             $zero,          $ac3                           \n\t"
        "preceu.ph.qbr    %[p1],          %[tp1]                         \n\t"
        "preceu.ph.qbl    %[p2],          %[tp1]                         \n\t"
        "preceu.ph.qbr    %[p3],          %[tp2]                         \n\t"
        "preceu.ph.qbl    %[p4],          %[tp2]                         \n\t"
        "dpa.w.ph         $ac3,           %[p1],          %[vector1b]    \n\t"
        "dpa.w.ph         $ac3,           %[p2],          %[vector2b]    \n\t"
        "dpa.w.ph         $ac3,           %[p3],          %[vector3b]    \n\t"
        "ulw              %[tn2],         8(%[src])                      \n\t"
        "dpa.w.ph         $ac3,           %[p4],          %[vector4b]    \n\t"
        "extp             %[Temp1],       $ac3,           31             \n\t"

        /* even 2. pixel */
        "mtlo             %[vector4a],    $ac2                           \n\t"
        "mthi             $zero,          $ac2                           \n\t"
        "preceu.ph.qbr    %[p1],          %[tn2]                         \n\t"
        "balign           %[tn1],         %[tn2],         3              \n\t"
        "balign           %[tn2],         %[tp2],         3              \n\t"
        "balign           %[tp2],         %[tp1],         3              \n\t"
        "dpa.w.ph         $ac2,           %[p2],          %[vector1b]    \n\t"
        "dpa.w.ph         $ac2,           %[p3],          %[vector2b]    \n\t"
        "dpa.w.ph         $ac2,           %[p4],          %[vector3b]    \n\t"
        "dpa.w.ph         $ac2,           %[p1],          %[vector4b]    \n\t"
        "extp             %[Temp3],       $ac2,           31             \n\t"

        "lbu              %[p2],          3(%[dst])                      \n\t" /* load odd 2 */

        /* odd 1. pixel */
        "lbux             %[tp1],         %[Temp1](%[cm])                \n\t" /* even 1 */
        "mtlo             %[vector4a],    $ac3                           \n\t"
        "mthi             $zero,          $ac3                           \n\t"
        "lbu              %[Temp1],       1(%[dst])                      \n\t" /* load odd 1 */
        "preceu.ph.qbr    %[n1],          %[tp2]                         \n\t"
        "preceu.ph.qbl    %[n2],          %[tp2]                         \n\t"
        "preceu.ph.qbr    %[n3],          %[tn2]                         \n\t"
        "preceu.ph.qbl    %[n4],          %[tn2]                         \n\t"
        "dpa.w.ph         $ac3,           %[n1],          %[vector1b]    \n\t"
        "dpa.w.ph         $ac3,           %[n2],          %[vector2b]    \n\t"
        "dpa.w.ph         $ac3,           %[n3],          %[vector3b]    \n\t"
        "dpa.w.ph         $ac3,           %[n4],          %[vector4b]    \n\t"
        "extp             %[Temp2],       $ac3,           31             \n\t"

        "lbu              %[tn2],         0(%[dst])                      \n\t" /* load even 1 */

        /* odd 2. pixel */
        "lbux             %[tp2],         %[Temp3](%[cm])                \n\t" /* even 2 */
        "mtlo             %[vector4a],    $ac2                           \n\t"
        "mthi             $zero,          $ac2                           \n\t"
        "preceu.ph.qbr    %[n1],          %[tn1]                         \n\t"
        "lbux             %[tn1],         %[Temp2](%[cm])                \n\t" /* odd 1 */
        "addqh_r.w        %[tn2],         %[tn2],         %[tp1]         \n\t" /* average even 1 */
        "dpa.w.ph         $ac2,           %[n2],          %[vector1b]    \n\t"
        "dpa.w.ph         $ac2,           %[n3],          %[vector2b]    \n\t"
        "dpa.w.ph         $ac2,           %[n4],          %[vector3b]    \n\t"
        "dpa.w.ph         $ac2,           %[n1],          %[vector4b]    \n\t"
        "extp             %[Temp4],       $ac2,           31             \n\t"

        "lbu              %[tp1],         2(%[dst])                      \n\t" /* load even 2 */
        "sb               %[tn2],         0(%[dst])                      \n\t" /* store even 1 */

        /* clamp */
        "addqh_r.w        %[Temp1],       %[Temp1],       %[tn1]         \n\t" /* average odd 1 */
        "lbux             %[n2],          %[Temp4](%[cm])                \n\t" /* odd 2 */
        "sb               %[Temp1],       1(%[dst])                      \n\t" /* store odd 1 */

        "addqh_r.w        %[tp1],         %[tp1],         %[tp2]         \n\t" /* average even 2 */
        "sb               %[tp1],         2(%[dst])                      \n\t" /* store even 2 */

        "addqh_r.w        %[p2],          %[p2],          %[n2]          \n\t" /* average odd 2 */
        "sb               %[p2],          3(%[dst])                      \n\t" /* store odd 2 */

        : [tp1] "=&r"(tp1), [tp2] "=&r"(tp2), [tn1] "=&r"(tn1),
          [tn2] "=&r"(tn2), [p1] "=&r"(p1), [p2] "=&r"(p2), [p3] "=&r"(p3),
          [p4] "=&r"(p4), [n1] "=&r"(n1), [n2] "=&r"(n2), [n3] "=&r"(n3),
          [n4] "=&r"(n4), [Temp1] "=&r"(Temp1), [Temp2] "=&r"(Temp2),
          [Temp3] "=&r"(Temp3), [Temp4] "=&r"(Temp4)
        : [vector1b] "r"(vector1b), [vector2b] "r"(vector2b),
          [vector3b] "r"(vector3b), [vector4b] "r"(vector4b),
          [vector4a] "r"(vector4a), [cm] "r"(cm), [dst] "r"(dst),
          [src] "r"(src));

    /* Next row... */
    src += src_stride;
    dst += dst_stride;
  }
}

static void convolve_avg_horiz_8_dspr2(const uint8_t *src, int32_t src_stride,
                                       uint8_t *dst, int32_t dst_stride,
                                       const int16_t *filter_x0, int32_t h) {
  int32_t y;
  uint8_t *cm = vpx_ff_cropTbl;
  uint32_t vector4a = 64;
  int32_t vector1b, vector2b, vector3b, vector4b;
  int32_t Temp1, Temp2, Temp3;
  uint32_t tp1, tp2;
  uint32_t p1, p2, p3, p4, n1;
  uint32_t tn1, tn2, tn3;
  uint32_t st0, st1;

  vector1b = ((const int32_t *)filter_x0)[0];
  vector2b = ((const int32_t *)filter_x0)[1];
  vector3b = ((const int32_t *)filter_x0)[2];
  vector4b = ((const int32_t *)filter_x0)[3];

  for (y = h; y--;) {
    /* prefetch data to cache memory */
    prefetch_load(src + src_stride);
    prefetch_load(src + src_stride + 32);
    prefetch_store(dst + dst_stride);

    __asm__ __volatile__(
        "ulw              %[tp1],         0(%[src])                      \n\t"
        "ulw              %[tp2],         4(%[src])                      \n\t"

        /* even 1. pixel */
        "mtlo             %[vector4a],    $ac3                           \n\t"
        "mthi             $zero,          $ac3                           \n\t"
        "mtlo             %[vector4a],    $ac2                           \n\t"
        "mthi             $zero,          $ac2                           \n\t"
        "preceu.ph.qbr    %[p1],          %[tp1]                         \n\t"
        "preceu.ph.qbl    %[p2],          %[tp1]                         \n\t"
        "preceu.ph.qbr    %[p3],          %[tp2]                         \n\t"
        "preceu.ph.qbl    %[p4],          %[tp2]                         \n\t"
        "ulw              %[tn2],         8(%[src])                      \n\t"
        "dpa.w.ph         $ac3,           %[p1],          %[vector1b]    \n\t"
        "dpa.w.ph         $ac3,           %[p2],          %[vector2b]    \n\t"
        "dpa.w.ph         $ac3,           %[p3],          %[vector3b]    \n\t"
        "dpa.w.ph         $ac3,           %[p4],          %[vector4b]    \n\t"
        "extp             %[Temp1],       $ac3,           31             \n\t"
        "lbu              %[Temp2],       0(%[dst])                      \n\t"
        "lbu              %[tn3],         2(%[dst])                      \n\t"

        /* even 2. pixel */
        "preceu.ph.qbr    %[p1],          %[tn2]                         \n\t"
        "preceu.ph.qbl    %[n1],          %[tn2]                         \n\t"
        "ulw              %[tn1],         12(%[src])                     \n\t"
        "dpa.w.ph         $ac2,           %[p2],          %[vector1b]    \n\t"
        "dpa.w.ph         $ac2,           %[p3],          %[vector2b]    \n\t"
        "dpa.w.ph         $ac2,           %[p4],          %[vector3b]    \n\t"
        "dpa.w.ph         $ac2,           %[p1],          %[vector4b]    \n\t"
        "extp             %[Temp3],       $ac2,           31             \n\t"

        /* even 3. pixel */
        "lbux             %[st0],         %[Temp1](%[cm])                \n\t"
        "mtlo             %[vector4a],    $ac1                           \n\t"
        "mthi             $zero,          $ac1                           \n\t"
        "preceu.ph.qbr    %[p2],          %[tn1]                         \n\t"
        "lbux             %[st1],         %[Temp3](%[cm])                \n\t"
        "dpa.w.ph         $ac1,           %[p3],          %[vector1b]    \n\t"
        "dpa.w.ph         $ac1,           %[p4],          %[vector2b]    \n\t"
        "dpa.w.ph         $ac1,           %[p1],          %[vector3b]    \n\t"
        "dpa.w.ph         $ac1,           %[n1],          %[vector4b]    \n\t"
        "extp             %[Temp1],       $ac1,           31             \n\t"

        "addqh_r.w        %[Temp2],       %[Temp2],       %[st0]         \n\t"
        "addqh_r.w        %[tn3],         %[tn3],         %[st1]         \n\t"
        "sb               %[Temp2],       0(%[dst])                      \n\t"
        "sb               %[tn3],         2(%[dst])                      \n\t"

        /* even 4. pixel */
        "mtlo             %[vector4a],    $ac2                           \n\t"
        "mthi             $zero,          $ac2                           \n\t"
        "mtlo             %[vector4a],    $ac3                           \n\t"
        "mthi             $zero,          $ac3                           \n\t"

        "balign           %[tn3],         %[tn1],         3              \n\t"
        "balign           %[tn1],         %[tn2],         3              \n\t"
        "balign           %[tn2],         %[tp2],         3              \n\t"
        "balign           %[tp2],         %[tp1],         3              \n\t"

        "lbux             %[st0],         %[Temp1](%[cm])                \n\t"
        "lbu              %[Temp2],       4(%[dst])                      \n\t"
        "addqh_r.w        %[Temp2],       %[Temp2],       %[st0]         \n\t"

        "dpa.w.ph         $ac2,           %[p4],          %[vector1b]    \n\t"
        "dpa.w.ph         $ac2,           %[p1],          %[vector2b]    \n\t"
        "dpa.w.ph         $ac2,           %[n1],          %[vector3b]    \n\t"
        "dpa.w.ph         $ac2,           %[p2],          %[vector4b]    \n\t"
        "extp             %[Temp3],       $ac2,           31             \n\t"

        /* odd 1. pixel */
        "mtlo             %[vector4a],    $ac1                           \n\t"
        "mthi             $zero,          $ac1                           \n\t"
        "sb               %[Temp2],       4(%[dst])                      \n\t"
        "preceu.ph.qbr    %[p1],          %[tp2]                         \n\t"
        "preceu.ph.qbl    %[p2],          %[tp2]                         \n\t"
        "preceu.ph.qbr    %[p3],          %[tn2]                         \n\t"
        "preceu.ph.qbl    %[p4],          %[tn2]                         \n\t"
        "dpa.w.ph         $ac3,           %[p1],          %[vector1b]    \n\t"
        "dpa.w.ph         $ac3,           %[p2],          %[vector2b]    \n\t"
        "dpa.w.ph         $ac3,           %[p3],          %[vector3b]    \n\t"
        "dpa.w.ph         $ac3,           %[p4],          %[vector4b]    \n\t"
        "extp             %[Temp2],       $ac3,           31             \n\t"

        "lbu              %[tp1],         6(%[dst])                      \n\t"

        /* odd 2. pixel */
        "mtlo             %[vector4a],    $ac3                           \n\t"
        "mthi             $zero,          $ac3                           \n\t"
        "mtlo             %[vector4a],    $ac2                           \n\t"
        "mthi             $zero,          $ac2                           \n\t"
        "preceu.ph.qbr    %[p1],          %[tn1]                         \n\t"
        "preceu.ph.qbl    %[n1],          %[tn1]                         \n\t"
        "lbux             %[st0],         %[Temp3](%[cm])                \n\t"
        "dpa.w.ph         $ac1,           %[p2],          %[vector1b]    \n\t"
        "dpa.w.ph         $ac1,           %[p3],          %[vector2b]    \n\t"
        "dpa.w.ph         $ac1,           %[p4],          %[vector3b]    \n\t"
        "dpa.w.ph         $ac1,           %[p1],          %[vector4b]    \n\t"
        "extp             %[Temp3],       $ac1,           31             \n\t"

        "lbu              %[tp2],         1(%[dst])                      \n\t"
        "lbu              %[tn2],         3(%[dst])                      \n\t"
        "addqh_r.w        %[tp1],         %[tp1],         %[st0]         \n\t"

        /* odd 3. pixel */
        "lbux             %[st1],         %[Temp2](%[cm])                \n\t"
        "preceu.ph.qbr    %[p2],          %[tn3]                         \n\t"
        "dpa.w.ph         $ac3,           %[p3],          %[vector1b]    \n\t"
        "dpa.w.ph         $ac3,           %[p4],          %[vector2b]    \n\t"
        "dpa.w.ph         $ac3,           %[p1],          %[vector3b]    \n\t"
        "dpa.w.ph         $ac3,           %[n1],          %[vector4b]    \n\t"
        "addqh_r.w        %[tp2],         %[tp2],         %[st1]         \n\t"
        "extp             %[Temp2],       $ac3,           31             \n\t"

        "lbu              %[tn3],         5(%[dst])                      \n\t"

        /* odd 4. pixel */
        "sb               %[tp2],         1(%[dst])                      \n\t"
        "sb               %[tp1],         6(%[dst])                      \n\t"
        "dpa.w.ph         $ac2,           %[p4],          %[vector1b]    \n\t"
        "dpa.w.ph         $ac2,           %[p1],          %[vector2b]    \n\t"
        "dpa.w.ph         $ac2,           %[n1],          %[vector3b]    \n\t"
        "dpa.w.ph         $ac2,           %[p2],          %[vector4b]    \n\t"
        "extp             %[Temp1],       $ac2,           31             \n\t"

        "lbu              %[tn1],         7(%[dst])                      \n\t"

        /* clamp */
        "lbux             %[p4],          %[Temp3](%[cm])                \n\t"
        "addqh_r.w        %[tn2],         %[tn2],         %[p4]          \n\t"

        "lbux             %[p2],          %[Temp2](%[cm])                \n\t"
        "addqh_r.w        %[tn3],         %[tn3],         %[p2]          \n\t"

        "lbux             %[n1],          %[Temp1](%[cm])                \n\t"
        "addqh_r.w        %[tn1],         %[tn1],         %[n1]          \n\t"

        /* store bytes */
        "sb               %[tn2],         3(%[dst])                      \n\t"
        "sb               %[tn3],         5(%[dst])                      \n\t"
        "sb               %[tn1],         7(%[dst])                      \n\t"

        : [tp1] "=&r"(tp1), [tp2] "=&r"(tp2), [tn1] "=&r"(tn1),
          [tn2] "=&r"(tn2), [tn3] "=&r"(tn3), [st0] "=&r"(st0),
          [st1] "=&r"(st1), [p1] "=&r"(p1), [p2] "=&r"(p2), [p3] "=&r"(p3),
          [p4] "=&r"(p4), [n1] "=&r"(n1), [Temp1] "=&r"(Temp1),
          [Temp2] "=&r"(Temp2), [Temp3] "=&r"(Temp3)
        : [vector1b] "r"(vector1b), [vector2b] "r"(vector2b),
          [vector3b] "r"(vector3b), [vector4b] "r"(vector4b),
          [vector4a] "r"(vector4a), [cm] "r"(cm), [dst] "r"(dst),
          [src] "r"(src));

    /* Next row... */
    src += src_stride;
    dst += dst_stride;
  }
}

static void convolve_avg_horiz_16_dspr2(const uint8_t *src_ptr,
                                        int32_t src_stride, uint8_t *dst_ptr,
                                        int32_t dst_stride,
                                        const int16_t *filter_x0, int32_t h,
                                        int32_t count) {
  int32_t y, c;
  const uint8_t *src;
  uint8_t *dst;
  uint8_t *cm = vpx_ff_cropTbl;
  uint32_t vector_64 = 64;
  int32_t filter12, filter34, filter56, filter78;
  int32_t Temp1, Temp2, Temp3;
  uint32_t qload1, qload2, qload3;
  uint32_t p1, p2, p3, p4, p5;
  uint32_t st1, st2, st3;

  filter12 = ((const int32_t *)filter_x0)[0];
  filter34 = ((const int32_t *)filter_x0)[1];
  filter56 = ((const int32_t *)filter_x0)[2];
  filter78 = ((const int32_t *)filter_x0)[3];

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
          "dpa.w.ph         $ac1,         %[p1],          %[filter12]  \n\t" /* even 1 */
          "dpa.w.ph         $ac1,         %[p2],          %[filter34]  \n\t" /* even 1 */
          "dpa.w.ph         $ac1,         %[p3],          %[filter56]  \n\t" /* even 1 */
          "dpa.w.ph         $ac1,         %[p4],          %[filter78]  \n\t" /* even 1 */
          "extp             %[Temp1],     $ac1,           31           \n\t" /* even 1 */
          "lbu              %[st2],       0(%[dst])                    \n\t" /* load even 1 from dst */

          /* even 2. pixel */
          "mtlo             %[vector_64], $ac3                         \n\t" /* even 3 */
          "mthi             $zero,        $ac3                         \n\t"
          "preceu.ph.qbr    %[p1],        %[qload3]                    \n\t"
          "preceu.ph.qbl    %[p5],        %[qload3]                    \n\t"
          "ulw              %[qload1],    12(%[src])                   \n\t"
          "dpa.w.ph         $ac2,         %[p2],          %[filter12]  \n\t" /* even 1 */
          "dpa.w.ph         $ac2,         %[p3],          %[filter34]  \n\t" /* even 1 */
          "dpa.w.ph         $ac2,         %[p4],          %[filter56]  \n\t" /* even 1 */
          "dpa.w.ph         $ac2,         %[p1],          %[filter78]  \n\t" /* even 1 */
          "extp             %[Temp2],     $ac2,           31           \n\t" /* even 1 */
          "lbux             %[st1],       %[Temp1](%[cm])              \n\t" /* even 1 */

          "lbu              %[qload3],    2(%[dst])                    \n\t" /* load even 2 from dst */

          /* even 3. pixel */
          "mtlo             %[vector_64], $ac1                         \n\t" /* even 4 */
          "mthi             $zero,        $ac1                         \n\t"
          "addqh_r.w        %[st2],       %[st2],         %[st1]       \n\t" /* average even 1 */
          "preceu.ph.qbr    %[p2],        %[qload1]                    \n\t"
          "sb               %[st2],       0(%[dst])                    \n\t" /* store even 1 to dst */
          "dpa.w.ph         $ac3,         %[p3],          %[filter12]  \n\t" /* even 3 */
          "dpa.w.ph         $ac3,         %[p4],          %[filter34]  \n\t" /* even 3 */
          "dpa.w.ph         $ac3,         %[p1],          %[filter56]  \n\t" /* even 3 */
          "dpa.w.ph         $ac3,         %[p5],          %[filter78]  \n\t" /* even 3 */
          "extp             %[Temp3],     $ac3,           31           \n\t" /* even 3 */
          "lbux             %[st2],       %[Temp2](%[cm])              \n\t" /* even 1 */

          /* even 4. pixel */
          "mtlo             %[vector_64], $ac2                         \n\t" /* even 5 */
          "mthi             $zero,        $ac2                         \n\t"
          "addqh_r.w        %[qload3],    %[qload3],      %[st2]       \n\t" /* average even 2 */
          "preceu.ph.qbl    %[p3],        %[qload1]                    \n\t"
          "sb               %[qload3],    2(%[dst])                    \n\t" /* store even 2 to dst */
          "ulw              %[qload2],    16(%[src])                   \n\t"
          "lbu              %[qload3],    4(%[dst])                    \n\t" /* load even 3 from dst */
          "lbu              %[qload1],    6(%[dst])                    \n\t" /* load even 4 from dst */
          "dpa.w.ph         $ac1,         %[p4],          %[filter12]  \n\t" /* even 4 */
          "dpa.w.ph         $ac1,         %[p1],          %[filter34]  \n\t" /* even 4 */
          "dpa.w.ph         $ac1,         %[p5],          %[filter56]  \n\t" /* even 4 */
          "dpa.w.ph         $ac1,         %[p2],          %[filter78]  \n\t" /* even 4 */
          "extp             %[Temp1],     $ac1,           31           \n\t" /* even 4 */
          "lbux             %[st3],       %[Temp3](%[cm])              \n\t" /* even 3 */

          /* even 5. pixel */
          "mtlo             %[vector_64], $ac3                         \n\t" /* even 6 */
          "mthi             $zero,        $ac3                         \n\t"
          "addqh_r.w        %[qload3],    %[qload3],      %[st3]       \n\t" /* average even 3 */
          "preceu.ph.qbr    %[p4],        %[qload2]                    \n\t"
          "sb               %[qload3],    4(%[dst])                    \n\t" /* store even 3 to dst */
          "dpa.w.ph         $ac2,         %[p1],          %[filter12]  \n\t" /* even 5 */
          "dpa.w.ph         $ac2,         %[p5],          %[filter34]  \n\t" /* even 5 */
          "dpa.w.ph         $ac2,         %[p2],          %[filter56]  \n\t" /* even 5 */
          "dpa.w.ph         $ac2,         %[p3],          %[filter78]  \n\t" /* even 5 */
          "extp             %[Temp2],     $ac2,           31           \n\t" /* even 5 */
          "lbux             %[st1],       %[Temp1](%[cm])              \n\t" /* even 4 */

          /* even 6. pixel */
          "mtlo             %[vector_64], $ac1                         \n\t" /* even 7 */
          "mthi             $zero,        $ac1                         \n\t"
          "addqh_r.w        %[qload1],    %[qload1],      %[st1]       \n\t" /* average even 4 */
          "preceu.ph.qbl    %[p1],        %[qload2]                    \n\t"
          "sb               %[qload1],    6(%[dst])                    \n\t" /* store even 4 to dst */
          "ulw              %[qload3],    20(%[src])                   \n\t"
          "dpa.w.ph         $ac3,         %[p5],          %[filter12]  \n\t" /* even 6 */
          "dpa.w.ph         $ac3,         %[p2],          %[filter34]  \n\t" /* even 6 */
          "dpa.w.ph         $ac3,         %[p3],          %[filter56]  \n\t" /* even 6 */
          "dpa.w.ph         $ac3,         %[p4],          %[filter78]  \n\t" /* even 6 */
          "lbu              %[qload2],    8(%[dst])                    \n\t" /* load even 5 from dst */
          "extp             %[Temp3],     $ac3,           31           \n\t" /* even 6 */
          "lbux             %[st2],       %[Temp2](%[cm])              \n\t" /* even 5 */

          /* even 7. pixel */
          "mtlo             %[vector_64], $ac2                         \n\t" /* even 8 */
          "mthi             $zero,        $ac2                         \n\t"
          "addqh_r.w        %[qload2],    %[qload2],      %[st2]       \n\t" /* average even 5 */
          "preceu.ph.qbr    %[p5],        %[qload3]                    \n\t"
          "sb               %[qload2],    8(%[dst])                    \n\t" /* store even 5 to dst */
          "dpa.w.ph         $ac1,         %[p2],          %[filter12]  \n\t" /* even 7 */
          "dpa.w.ph         $ac1,         %[p3],          %[filter34]  \n\t" /* even 7 */
          "dpa.w.ph         $ac1,         %[p4],          %[filter56]  \n\t" /* even 7 */
          "dpa.w.ph         $ac1,         %[p1],          %[filter78]  \n\t" /* even 7 */
          "lbu              %[qload3],    10(%[dst])                   \n\t" /* load even 6 from dst */
          "extp             %[Temp1],     $ac1,           31           \n\t" /* even 7 */
          "lbux             %[st3],       %[Temp3](%[cm])              \n\t" /* even 6 */

          "lbu              %[st2],       12(%[dst])                   \n\t" /* load even 7 from dst */

          /* even 8. pixel */
          "mtlo             %[vector_64], $ac3                         \n\t" /* odd 1 */
          "mthi             $zero,        $ac3                         \n\t"
          "addqh_r.w        %[qload3],    %[qload3],      %[st3]       \n\t" /* average even 6 */
          "dpa.w.ph         $ac2,         %[p3],          %[filter12]  \n\t" /* even 8 */
          "dpa.w.ph         $ac2,         %[p4],          %[filter34]  \n\t" /* even 8 */
          "sb               %[qload3],    10(%[dst])                   \n\t" /* store even 6 to dst */
          "dpa.w.ph         $ac2,         %[p1],          %[filter56]  \n\t" /* even 8 */
          "dpa.w.ph         $ac2,         %[p5],          %[filter78]  \n\t" /* even 8 */
          "extp             %[Temp2],     $ac2,           31           \n\t" /* even 8 */
          "lbux             %[st1],       %[Temp1](%[cm])              \n\t" /* even 7 */

          /* ODD pixels */
          "ulw              %[qload1],    1(%[src])                   \n\t"
          "ulw              %[qload2],    5(%[src])                    \n\t"

          "addqh_r.w        %[st2],       %[st2],         %[st1]       \n\t" /* average even 7 */

          /* odd 1. pixel */
          "mtlo             %[vector_64], $ac1                         \n\t" /* odd 2 */
          "mthi             $zero,        $ac1                         \n\t"
          "preceu.ph.qbr    %[p1],        %[qload1]                    \n\t"
          "preceu.ph.qbl    %[p2],        %[qload1]                    \n\t"
          "preceu.ph.qbr    %[p3],        %[qload2]                    \n\t"
          "preceu.ph.qbl    %[p4],        %[qload2]                    \n\t"
          "sb               %[st2],       12(%[dst])                   \n\t" /* store even 7 to dst */
          "ulw              %[qload3],    9(%[src])                    \n\t"
          "dpa.w.ph         $ac3,         %[p1],          %[filter12]  \n\t" /* odd 1 */
          "dpa.w.ph         $ac3,         %[p2],          %[filter34]  \n\t" /* odd 1 */
          "lbu              %[qload2],    14(%[dst])                   \n\t" /* load even 8 from dst */
          "dpa.w.ph         $ac3,         %[p3],          %[filter56]  \n\t" /* odd 1 */
          "dpa.w.ph         $ac3,         %[p4],          %[filter78]  \n\t" /* odd 1 */
          "extp             %[Temp3],     $ac3,           31           \n\t" /* odd 1 */
          "lbux             %[st2],       %[Temp2](%[cm])              \n\t" /* even 8 */

          "lbu              %[st1],       1(%[dst])                    \n\t" /* load odd 1 from dst */

          /* odd 2. pixel */
          "mtlo             %[vector_64], $ac2                         \n\t" /* odd 3 */
          "mthi             $zero,        $ac2                         \n\t"
          "addqh_r.w        %[qload2],    %[qload2],      %[st2]       \n\t" /* average even 8 */
          "preceu.ph.qbr    %[p1],        %[qload3]                    \n\t"
          "preceu.ph.qbl    %[p5],        %[qload3]                    \n\t"
          "sb               %[qload2],    14(%[dst])                   \n\t" /* store even 8 to dst */
          "ulw              %[qload1],    13(%[src])                   \n\t"
          "dpa.w.ph         $ac1,         %[p2],          %[filter12]  \n\t" /* odd 2 */
          "dpa.w.ph         $ac1,         %[p3],          %[filter34]  \n\t" /* odd 2 */
          "dpa.w.ph         $ac1,         %[p4],          %[filter56]  \n\t" /* odd 2 */
          "dpa.w.ph         $ac1,         %[p1],          %[filter78]  \n\t" /* odd 2 */
          "lbu              %[qload3],    3(%[dst])                    \n\t" /* load odd 2 from dst */
          "extp             %[Temp1],     $ac1,           31           \n\t" /* odd 2 */
          "lbux             %[st3],       %[Temp3](%[cm])              \n\t" /* odd 1 */

          /* odd 3. pixel */
          "mtlo             %[vector_64], $ac3                         \n\t" /* odd 4 */
          "mthi             $zero,        $ac3                         \n\t"
          "addqh_r.w        %[st3],       %[st3],         %[st1]       \n\t" /* average odd 1 */
          "preceu.ph.qbr    %[p2],        %[qload1]                    \n\t"
          "dpa.w.ph         $ac2,         %[p3],          %[filter12]  \n\t" /* odd 3 */
          "dpa.w.ph         $ac2,         %[p4],          %[filter34]  \n\t" /* odd 3 */
          "dpa.w.ph         $ac2,         %[p1],          %[filter56]  \n\t" /* odd 3 */
          "dpa.w.ph         $ac2,         %[p5],          %[filter78]  \n\t" /* odd 3 */
          "sb               %[st3],       1(%[dst])                    \n\t" /* store odd 1 to dst */
          "extp             %[Temp2],     $ac2,           31           \n\t" /* odd 3 */
          "lbux             %[st1],       %[Temp1](%[cm])              \n\t" /* odd 2 */

          /* odd 4. pixel */
          "mtlo             %[vector_64], $ac1                         \n\t" /* odd 5 */
          "mthi             $zero,        $ac1                         \n\t"
          "addqh_r.w        %[qload3],    %[qload3],      %[st1]       \n\t" /* average odd 2 */
          "preceu.ph.qbl    %[p3],        %[qload1]                    \n\t"
          "sb               %[qload3],    3(%[dst])                    \n\t" /* store odd 2 to dst */
          "lbu              %[qload1],    5(%[dst])                    \n\t" /* load odd 3 from dst */
          "ulw              %[qload2],    17(%[src])                   \n\t"
          "dpa.w.ph         $ac3,         %[p4],          %[filter12]  \n\t" /* odd 4 */
          "dpa.w.ph         $ac3,         %[p1],          %[filter34]  \n\t" /* odd 4 */
          "dpa.w.ph         $ac3,         %[p5],          %[filter56]  \n\t" /* odd 4 */
          "dpa.w.ph         $ac3,         %[p2],          %[filter78]  \n\t" /* odd 4 */
          "extp             %[Temp3],     $ac3,           31           \n\t" /* odd 4 */
          "lbux             %[st2],       %[Temp2](%[cm])              \n\t" /* odd 3 */

          "lbu              %[st1],       7(%[dst])                    \n\t" /* load odd 4 from dst */

          /* odd 5. pixel */
          "mtlo             %[vector_64], $ac2                         \n\t" /* odd 6 */
          "mthi             $zero,        $ac2                         \n\t"
          "addqh_r.w        %[qload1],    %[qload1],      %[st2]       \n\t" /* average odd 3 */
          "preceu.ph.qbr    %[p4],        %[qload2]                    \n\t"
          "sb               %[qload1],    5(%[dst])                    \n\t" /* store odd 3 to dst */
          "dpa.w.ph         $ac1,         %[p1],          %[filter12]  \n\t" /* odd 5 */
          "dpa.w.ph         $ac1,         %[p5],          %[filter34]  \n\t" /* odd 5 */
          "dpa.w.ph         $ac1,         %[p2],          %[filter56]  \n\t" /* odd 5 */
          "dpa.w.ph         $ac1,         %[p3],          %[filter78]  \n\t" /* odd 5 */
          "extp             %[Temp1],     $ac1,           31           \n\t" /* odd 5 */
          "lbux             %[st3],       %[Temp3](%[cm])              \n\t" /* odd 4 */

          "lbu              %[qload1],    9(%[dst])                    \n\t" /* load odd 5 from dst */

          /* odd 6. pixel */
          "mtlo             %[vector_64], $ac3                         \n\t" /* odd 7 */
          "mthi             $zero,        $ac3                         \n\t"
          "addqh_r.w        %[st1],       %[st1],         %[st3]       \n\t" /* average odd 4 */
          "preceu.ph.qbl    %[p1],        %[qload2]                    \n\t"
          "sb               %[st1],       7(%[dst])                    \n\t" /* store odd 4 to dst */
          "ulw              %[qload3],    21(%[src])                   \n\t"
          "dpa.w.ph         $ac2,         %[p5],          %[filter12]  \n\t" /* odd 6 */
          "dpa.w.ph         $ac2,         %[p2],          %[filter34]  \n\t" /* odd 6 */
          "dpa.w.ph         $ac2,         %[p3],          %[filter56]  \n\t" /* odd 6 */
          "dpa.w.ph         $ac2,         %[p4],          %[filter78]  \n\t" /* odd 6 */
          "extp             %[Temp2],     $ac2,           31           \n\t" /* odd 6 */
          "lbux             %[st1],       %[Temp1](%[cm])              \n\t" /* odd 5 */

          /* odd 7. pixel */
          "mtlo             %[vector_64], $ac1                         \n\t" /* odd 8 */
          "mthi             $zero,        $ac1                         \n\t"
          "addqh_r.w        %[qload1],    %[qload1],      %[st1]       \n\t" /* average odd 5 */
          "preceu.ph.qbr    %[p5],        %[qload3]                    \n\t"
          "sb               %[qload1],    9(%[dst])                    \n\t" /* store odd 5 to dst */
          "lbu              %[qload2],    11(%[dst])                   \n\t" /* load odd 6 from dst */
          "dpa.w.ph         $ac3,         %[p2],          %[filter12]  \n\t" /* odd 7 */
          "dpa.w.ph         $ac3,         %[p3],          %[filter34]  \n\t" /* odd 7 */
          "dpa.w.ph         $ac3,         %[p4],          %[filter56]  \n\t" /* odd 7 */
          "dpa.w.ph         $ac3,         %[p1],          %[filter78]  \n\t" /* odd 7 */
          "extp             %[Temp3],     $ac3,           31           \n\t" /* odd 7 */

          "lbu              %[qload3],    13(%[dst])                   \n\t" /* load odd 7 from dst */

          /* odd 8. pixel */
          "dpa.w.ph         $ac1,         %[p3],          %[filter12]  \n\t" /* odd 8 */
          "dpa.w.ph         $ac1,         %[p4],          %[filter34]  \n\t" /* odd 8 */
          "dpa.w.ph         $ac1,         %[p1],          %[filter56]  \n\t" /* odd 8 */
          "dpa.w.ph         $ac1,         %[p5],          %[filter78]  \n\t" /* odd 8 */
          "extp             %[Temp1],     $ac1,           31           \n\t" /* odd 8 */

          "lbu              %[qload1],    15(%[dst])                   \n\t" /* load odd 8 from dst */

          "lbux             %[st2],       %[Temp2](%[cm])              \n\t" /* odd 6 */
          "addqh_r.w        %[qload2],    %[qload2],      %[st2]       \n\t" /* average odd 6 */

          "lbux             %[st3],       %[Temp3](%[cm])              \n\t" /* odd 7 */
          "addqh_r.w        %[qload3],    %[qload3],      %[st3]       \n\t" /* average odd 7 */

          "lbux             %[st1],       %[Temp1](%[cm])              \n\t" /* odd 8 */
          "addqh_r.w        %[qload1],    %[qload1],      %[st1]       \n\t" /* average odd 8 */

          "sb               %[qload2],    11(%[dst])                   \n\t" /* store odd 6 to dst */
          "sb               %[qload3],    13(%[dst])                   \n\t" /* store odd 7 to dst */
          "sb               %[qload1],    15(%[dst])                   \n\t" /* store odd 8 to dst */

          : [qload1] "=&r"(qload1), [qload2] "=&r"(qload2), [st1] "=&r"(st1),
            [st2] "=&r"(st2), [st3] "=&r"(st3), [p1] "=&r"(p1), [p2] "=&r"(p2),
            [p3] "=&r"(p3), [p4] "=&r"(p4), [qload3] "=&r"(qload3),
            [p5] "=&r"(p5), [Temp1] "=&r"(Temp1), [Temp2] "=&r"(Temp2),
            [Temp3] "=&r"(Temp3)
          : [filter12] "r"(filter12), [filter34] "r"(filter34),
            [filter56] "r"(filter56), [filter78] "r"(filter78),
            [vector_64] "r"(vector_64), [cm] "r"(cm), [dst] "r"(dst),
            [src] "r"(src));

      src += 16;
      dst += 16;
    }

    /* Next row... */
    src_ptr += src_stride;
    dst_ptr += dst_stride;
  }
}

static void convolve_avg_horiz_64_dspr2(const uint8_t *src_ptr,
                                        int32_t src_stride, uint8_t *dst_ptr,
                                        int32_t dst_stride,
                                        const int16_t *filter_x0, int32_t h) {
  int32_t y, c;
  const uint8_t *src;
  uint8_t *dst;
  uint8_t *cm = vpx_ff_cropTbl;
  uint32_t vector_64 = 64;
  int32_t filter12, filter34, filter56, filter78;
  int32_t Temp1, Temp2, Temp3;
  uint32_t qload1, qload2, qload3;
  uint32_t p1, p2, p3, p4, p5;
  uint32_t st1, st2, st3;

  filter12 = ((const int32_t *)filter_x0)[0];
  filter34 = ((const int32_t *)filter_x0)[1];
  filter56 = ((const int32_t *)filter_x0)[2];
  filter78 = ((const int32_t *)filter_x0)[3];

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
          "dpa.w.ph         $ac1,         %[p1],          %[filter12]  \n\t" /* even 1 */
          "dpa.w.ph         $ac1,         %[p2],          %[filter34]  \n\t" /* even 1 */
          "dpa.w.ph         $ac1,         %[p3],          %[filter56]  \n\t" /* even 1 */
          "dpa.w.ph         $ac1,         %[p4],          %[filter78]  \n\t" /* even 1 */
          "extp             %[Temp1],     $ac1,           31           \n\t" /* even 1 */
          "lbu              %[st2],       0(%[dst])                    \n\t" /* load even 1 from dst */

          /* even 2. pixel */
          "mtlo             %[vector_64], $ac3                         \n\t" /* even 3 */
          "mthi             $zero,        $ac3                         \n\t"
          "preceu.ph.qbr    %[p1],        %[qload3]                    \n\t"
          "preceu.ph.qbl    %[p5],        %[qload3]                    \n\t"
          "ulw              %[qload1],    12(%[src])                   \n\t"
          "dpa.w.ph         $ac2,         %[p2],          %[filter12]  \n\t" /* even 1 */
          "dpa.w.ph         $ac2,         %[p3],          %[filter34]  \n\t" /* even 1 */
          "dpa.w.ph         $ac2,         %[p4],          %[filter56]  \n\t" /* even 1 */
          "dpa.w.ph         $ac2,         %[p1],          %[filter78]  \n\t" /* even 1 */
          "extp             %[Temp2],     $ac2,           31           \n\t" /* even 1 */
          "lbux             %[st1],       %[Temp1](%[cm])              \n\t" /* even 1 */

          "lbu              %[qload3],    2(%[dst])                    \n\t" /* load even 2 from dst */

          /* even 3. pixel */
          "mtlo             %[vector_64], $ac1                         \n\t" /* even 4 */
          "mthi             $zero,        $ac1                         \n\t"
          "addqh_r.w        %[st2],       %[st2],         %[st1]       \n\t" /* average even 1 */
          "preceu.ph.qbr    %[p2],        %[qload1]                    \n\t"
          "sb               %[st2],       0(%[dst])                    \n\t" /* store even 1 to dst */
          "dpa.w.ph         $ac3,         %[p3],          %[filter12]  \n\t" /* even 3 */
          "dpa.w.ph         $ac3,         %[p4],          %[filter34]  \n\t" /* even 3 */
          "dpa.w.ph         $ac3,         %[p1],          %[filter56]  \n\t" /* even 3 */
          "dpa.w.ph         $ac3,         %[p5],          %[filter78]  \n\t" /* even 3 */
          "extp             %[Temp3],     $ac3,           31           \n\t" /* even 3 */
          "lbux             %[st2],       %[Temp2](%[cm])              \n\t" /* even 1 */

          /* even 4. pixel */
          "mtlo             %[vector_64], $ac2                         \n\t" /* even 5 */
          "mthi             $zero,        $ac2                         \n\t"
          "addqh_r.w        %[qload3],    %[qload3],      %[st2]       \n\t" /* average even 2 */
          "preceu.ph.qbl    %[p3],        %[qload1]                    \n\t"
          "sb               %[qload3],    2(%[dst])                    \n\t" /* store even 2 to dst */
          "ulw              %[qload2],    16(%[src])                   \n\t"
          "lbu              %[qload3],    4(%[dst])                    \n\t" /* load even 3 from dst */
          "lbu              %[qload1],    6(%[dst])                    \n\t" /* load even 4 from dst */
          "dpa.w.ph         $ac1,         %[p4],          %[filter12]  \n\t" /* even 4 */
          "dpa.w.ph         $ac1,         %[p1],          %[filter34]  \n\t" /* even 4 */
          "dpa.w.ph         $ac1,         %[p5],          %[filter56]  \n\t" /* even 4 */
          "dpa.w.ph         $ac1,         %[p2],          %[filter78]  \n\t" /* even 4 */
          "extp             %[Temp1],     $ac1,           31           \n\t" /* even 4 */
          "lbux             %[st3],       %[Temp3](%[cm])              \n\t" /* even 3 */

          /* even 5. pixel */
          "mtlo             %[vector_64], $ac3                         \n\t" /* even 6 */
          "mthi             $zero,        $ac3                         \n\t"
          "addqh_r.w        %[qload3],    %[qload3],      %[st3]       \n\t" /* average even 3 */
          "preceu.ph.qbr    %[p4],        %[qload2]                    \n\t"
          "sb               %[qload3],    4(%[dst])                    \n\t" /* store even 3 to dst */
          "dpa.w.ph         $ac2,         %[p1],          %[filter12]  \n\t" /* even 5 */
          "dpa.w.ph         $ac2,         %[p5],          %[filter34]  \n\t" /* even 5 */
          "dpa.w.ph         $ac2,         %[p2],          %[filter56]  \n\t" /* even 5 */
          "dpa.w.ph         $ac2,         %[p3],          %[filter78]  \n\t" /* even 5 */
          "extp             %[Temp2],     $ac2,           31           \n\t" /* even 5 */
          "lbux             %[st1],       %[Temp1](%[cm])              \n\t" /* even 4 */

          /* even 6. pixel */
          "mtlo             %[vector_64], $ac1                         \n\t" /* even 7 */
          "mthi             $zero,        $ac1                         \n\t"
          "addqh_r.w        %[qload1],    %[qload1],      %[st1]       \n\t" /* average even 4 */
          "preceu.ph.qbl    %[p1],        %[qload2]                    \n\t"
          "sb               %[qload1],    6(%[dst])                    \n\t" /* store even 4 to dst */
          "ulw              %[qload3],    20(%[src])                   \n\t"
          "dpa.w.ph         $ac3,         %[p5],          %[filter12]  \n\t" /* even 6 */
          "dpa.w.ph         $ac3,         %[p2],          %[filter34]  \n\t" /* even 6 */
          "dpa.w.ph         $ac3,         %[p3],          %[filter56]  \n\t" /* even 6 */
          "dpa.w.ph         $ac3,         %[p4],          %[filter78]  \n\t" /* even 6 */
          "lbu              %[qload2],    8(%[dst])                    \n\t" /* load even 5 from dst */
          "extp             %[Temp3],     $ac3,           31           \n\t" /* even 6 */
          "lbux             %[st2],       %[Temp2](%[cm])              \n\t" /* even 5 */

          /* even 7. pixel */
          "mtlo             %[vector_64], $ac2                         \n\t" /* even 8 */
          "mthi             $zero,        $ac2                         \n\t"
          "addqh_r.w        %[qload2],    %[qload2],      %[st2]       \n\t" /* average even 5 */
          "preceu.ph.qbr    %[p5],        %[qload3]                    \n\t"
          "sb               %[qload2],    8(%[dst])                    \n\t" /* store even 5 to dst */
          "dpa.w.ph         $ac1,         %[p2],          %[filter12]  \n\t" /* even 7 */
          "dpa.w.ph         $ac1,         %[p3],          %[filter34]  \n\t" /* even 7 */
          "dpa.w.ph         $ac1,         %[p4],          %[filter56]  \n\t" /* even 7 */
          "dpa.w.ph         $ac1,         %[p1],          %[filter78]  \n\t" /* even 7 */
          "lbu              %[qload3],    10(%[dst])                   \n\t" /* load even 6 from dst */
          "extp             %[Temp1],     $ac1,           31           \n\t" /* even 7 */
          "lbux             %[st3],       %[Temp3](%[cm])              \n\t" /* even 6 */

          "lbu              %[st2],       12(%[dst])                   \n\t" /* load even 7 from dst */

          /* even 8. pixel */
          "mtlo             %[vector_64], $ac3                         \n\t" /* odd 1 */
          "mthi             $zero,        $ac3                         \n\t"
          "addqh_r.w        %[qload3],    %[qload3],      %[st3]       \n\t" /* average even 6 */
          "dpa.w.ph         $ac2,         %[p3],          %[filter12]  \n\t" /* even 8 */
          "dpa.w.ph         $ac2,         %[p4],          %[filter34]  \n\t" /* even 8 */
          "sb               %[qload3],    10(%[dst])                   \n\t" /* store even 6 to dst */
          "dpa.w.ph         $ac2,         %[p1],          %[filter56]  \n\t" /* even 8 */
          "dpa.w.ph         $ac2,         %[p5],          %[filter78]  \n\t" /* even 8 */
          "extp             %[Temp2],     $ac2,           31           \n\t" /* even 8 */
          "lbux             %[st1],       %[Temp1](%[cm])              \n\t" /* even 7 */

          /* ODD pixels */
          "ulw              %[qload1],    1(%[src])                   \n\t"
          "ulw              %[qload2],    5(%[src])                    \n\t"

          "addqh_r.w        %[st2],       %[st2],         %[st1]       \n\t" /* average even 7 */

          /* odd 1. pixel */
          "mtlo             %[vector_64], $ac1                         \n\t" /* odd 2 */
          "mthi             $zero,        $ac1                         \n\t"
          "preceu.ph.qbr    %[p1],        %[qload1]                    \n\t"
          "preceu.ph.qbl    %[p2],        %[qload1]                    \n\t"
          "preceu.ph.qbr    %[p3],        %[qload2]                    \n\t"
          "preceu.ph.qbl    %[p4],        %[qload2]                    \n\t"
          "sb               %[st2],       12(%[dst])                   \n\t" /* store even 7 to dst */
          "ulw              %[qload3],    9(%[src])                    \n\t"
          "dpa.w.ph         $ac3,         %[p1],          %[filter12]  \n\t" /* odd 1 */
          "dpa.w.ph         $ac3,         %[p2],          %[filter34]  \n\t" /* odd 1 */
          "lbu              %[qload2],    14(%[dst])                   \n\t" /* load even 8 from dst */
          "dpa.w.ph         $ac3,         %[p3],          %[filter56]  \n\t" /* odd 1 */
          "dpa.w.ph         $ac3,         %[p4],          %[filter78]  \n\t" /* odd 1 */
          "extp             %[Temp3],     $ac3,           31           \n\t" /* odd 1 */
          "lbux             %[st2],       %[Temp2](%[cm])              \n\t" /* even 8 */

          "lbu              %[st1],       1(%[dst])                    \n\t" /* load odd 1 from dst */

          /* odd 2. pixel */
          "mtlo             %[vector_64], $ac2                         \n\t" /* odd 3 */
          "mthi             $zero,        $ac2                         \n\t"
          "addqh_r.w        %[qload2],    %[qload2],      %[st2]       \n\t" /* average even 8 */
          "preceu.ph.qbr    %[p1],        %[qload3]                    \n\t"
          "preceu.ph.qbl    %[p5],        %[qload3]                    \n\t"
          "sb               %[qload2],    14(%[dst])                   \n\t" /* store even 8 to dst */
          "ulw              %[qload1],    13(%[src])                   \n\t"
          "dpa.w.ph         $ac1,         %[p2],          %[filter12]  \n\t" /* odd 2 */
          "dpa.w.ph         $ac1,         %[p3],          %[filter34]  \n\t" /* odd 2 */
          "dpa.w.ph         $ac1,         %[p4],          %[filter56]  \n\t" /* odd 2 */
          "dpa.w.ph         $ac1,         %[p1],          %[filter78]  \n\t" /* odd 2 */
          "lbu              %[qload3],    3(%[dst])                    \n\t" /* load odd 2 from dst */
          "extp             %[Temp1],     $ac1,           31           \n\t" /* odd 2 */
          "lbux             %[st3],       %[Temp3](%[cm])              \n\t" /* odd 1 */

          /* odd 3. pixel */
          "mtlo             %[vector_64], $ac3                         \n\t" /* odd 4 */
          "mthi             $zero,        $ac3                         \n\t"
          "addqh_r.w        %[st3],       %[st3],         %[st1]       \n\t" /* average odd 1 */
          "preceu.ph.qbr    %[p2],        %[qload1]                    \n\t"
          "dpa.w.ph         $ac2,         %[p3],          %[filter12]  \n\t" /* odd 3 */
          "dpa.w.ph         $ac2,         %[p4],          %[filter34]  \n\t" /* odd 3 */
          "dpa.w.ph         $ac2,         %[p1],          %[filter56]  \n\t" /* odd 3 */
          "dpa.w.ph         $ac2,         %[p5],          %[filter78]  \n\t" /* odd 3 */
          "sb               %[st3],       1(%[dst])                    \n\t" /* store odd 1 to dst */
          "extp             %[Temp2],     $ac2,           31           \n\t" /* odd 3 */
          "lbux             %[st1],       %[Temp1](%[cm])              \n\t" /* odd 2 */

          /* odd 4. pixel */
          "mtlo             %[vector_64], $ac1                         \n\t" /* odd 5 */
          "mthi             $zero,        $ac1                         \n\t"
          "addqh_r.w        %[qload3],    %[qload3],      %[st1]       \n\t" /* average odd 2 */
          "preceu.ph.qbl    %[p3],        %[qload1]                    \n\t"
          "sb               %[qload3],    3(%[dst])                    \n\t" /* store odd 2 to dst */
          "lbu              %[qload1],    5(%[dst])                    \n\t" /* load odd 3 from dst */
          "ulw              %[qload2],    17(%[src])                   \n\t"
          "dpa.w.ph         $ac3,         %[p4],          %[filter12]  \n\t" /* odd 4 */
          "dpa.w.ph         $ac3,         %[p1],          %[filter34]  \n\t" /* odd 4 */
          "dpa.w.ph         $ac3,         %[p5],          %[filter56]  \n\t" /* odd 4 */
          "dpa.w.ph         $ac3,         %[p2],          %[filter78]  \n\t" /* odd 4 */
          "extp             %[Temp3],     $ac3,           31           \n\t" /* odd 4 */
          "lbux             %[st2],       %[Temp2](%[cm])              \n\t" /* odd 3 */

          "lbu              %[st1],       7(%[dst])                    \n\t" /* load odd 4 from dst */

          /* odd 5. pixel */
          "mtlo             %[vector_64], $ac2                         \n\t" /* odd 6 */
          "mthi             $zero,        $ac2                         \n\t"
          "addqh_r.w        %[qload1],    %[qload1],      %[st2]       \n\t" /* average odd 3 */
          "preceu.ph.qbr    %[p4],        %[qload2]                    \n\t"
          "sb               %[qload1],    5(%[dst])                    \n\t" /* store odd 3 to dst */
          "dpa.w.ph         $ac1,         %[p1],          %[filter12]  \n\t" /* odd 5 */
          "dpa.w.ph         $ac1,         %[p5],          %[filter34]  \n\t" /* odd 5 */
          "dpa.w.ph         $ac1,         %[p2],          %[filter56]  \n\t" /* odd 5 */
          "dpa.w.ph         $ac1,         %[p3],          %[filter78]  \n\t" /* odd 5 */
          "extp             %[Temp1],     $ac1,           31           \n\t" /* odd 5 */
          "lbux             %[st3],       %[Temp3](%[cm])              \n\t" /* odd 4 */

          "lbu              %[qload1],    9(%[dst])                    \n\t" /* load odd 5 from dst */

          /* odd 6. pixel */
          "mtlo             %[vector_64], $ac3                         \n\t" /* odd 7 */
          "mthi             $zero,        $ac3                         \n\t"
          "addqh_r.w        %[st1],       %[st1],         %[st3]       \n\t" /* average odd 4 */
          "preceu.ph.qbl    %[p1],        %[qload2]                    \n\t"
          "sb               %[st1],       7(%[dst])                    \n\t" /* store odd 4 to dst */
          "ulw              %[qload3],    21(%[src])                   \n\t"
          "dpa.w.ph         $ac2,         %[p5],          %[filter12]  \n\t" /* odd 6 */
          "dpa.w.ph         $ac2,         %[p2],          %[filter34]  \n\t" /* odd 6 */
          "dpa.w.ph         $ac2,         %[p3],          %[filter56]  \n\t" /* odd 6 */
          "dpa.w.ph         $ac2,         %[p4],          %[filter78]  \n\t" /* odd 6 */
          "extp             %[Temp2],     $ac2,           31           \n\t" /* odd 6 */
          "lbux             %[st1],       %[Temp1](%[cm])              \n\t" /* odd 5 */

          /* odd 7. pixel */
          "mtlo             %[vector_64], $ac1                         \n\t" /* odd 8 */
          "mthi             $zero,        $ac1                         \n\t"
          "addqh_r.w        %[qload1],    %[qload1],      %[st1]       \n\t" /* average odd 5 */
          "preceu.ph.qbr    %[p5],        %[qload3]                    \n\t"
          "sb               %[qload1],    9(%[dst])                    \n\t" /* store odd 5 to dst */
          "lbu              %[qload2],    11(%[dst])                   \n\t" /* load odd 6 from dst */
          "dpa.w.ph         $ac3,         %[p2],          %[filter12]  \n\t" /* odd 7 */
          "dpa.w.ph         $ac3,         %[p3],          %[filter34]  \n\t" /* odd 7 */
          "dpa.w.ph         $ac3,         %[p4],          %[filter56]  \n\t" /* odd 7 */
          "dpa.w.ph         $ac3,         %[p1],          %[filter78]  \n\t" /* odd 7 */
          "extp             %[Temp3],     $ac3,           31           \n\t" /* odd 7 */

          "lbu              %[qload3],    13(%[dst])                   \n\t" /* load odd 7 from dst */

          /* odd 8. pixel */
          "dpa.w.ph         $ac1,         %[p3],          %[filter12]  \n\t" /* odd 8 */
          "dpa.w.ph         $ac1,         %[p4],          %[filter34]  \n\t" /* odd 8 */
          "dpa.w.ph         $ac1,         %[p1],          %[filter56]  \n\t" /* odd 8 */
          "dpa.w.ph         $ac1,         %[p5],          %[filter78]  \n\t" /* odd 8 */
          "extp             %[Temp1],     $ac1,           31           \n\t" /* odd 8 */

          "lbu              %[qload1],    15(%[dst])                   \n\t" /* load odd 8 from dst */

          "lbux             %[st2],       %[Temp2](%[cm])              \n\t" /* odd 6 */
          "addqh_r.w        %[qload2],    %[qload2],      %[st2]       \n\t" /* average odd 6 */

          "lbux             %[st3],       %[Temp3](%[cm])              \n\t" /* odd 7 */
          "addqh_r.w        %[qload3],    %[qload3],      %[st3]       \n\t" /* average odd 7 */

          "lbux             %[st1],       %[Temp1](%[cm])              \n\t" /* odd 8 */
          "addqh_r.w        %[qload1],    %[qload1],      %[st1]       \n\t" /* average odd 8 */

          "sb               %[qload2],    11(%[dst])                   \n\t" /* store odd 6 to dst */
          "sb               %[qload3],    13(%[dst])                   \n\t" /* store odd 7 to dst */
          "sb               %[qload1],    15(%[dst])                   \n\t" /* store odd 8 to dst */

          : [qload1] "=&r"(qload1), [qload2] "=&r"(qload2), [st1] "=&r"(st1),
            [st2] "=&r"(st2), [st3] "=&r"(st3), [p1] "=&r"(p1), [p2] "=&r"(p2),
            [p3] "=&r"(p3), [p4] "=&r"(p4), [qload3] "=&r"(qload3),
            [p5] "=&r"(p5), [Temp1] "=&r"(Temp1), [Temp2] "=&r"(Temp2),
            [Temp3] "=&r"(Temp3)
          : [filter12] "r"(filter12), [filter34] "r"(filter34),
            [filter56] "r"(filter56), [filter78] "r"(filter78),
            [vector_64] "r"(vector_64), [cm] "r"(cm), [dst] "r"(dst),
            [src] "r"(src));

      src += 16;
      dst += 16;
    }

    /* Next row... */
    src_ptr += src_stride;
    dst_ptr += dst_stride;
  }
}

void vpx_convolve8_avg_horiz_dspr2(const uint8_t *src, ptrdiff_t src_stride,
                                   uint8_t *dst, ptrdiff_t dst_stride,
                                   const InterpKernel *filter, int x0_q4,
                                   int32_t x_step_q4, int y0_q4, int y_step_q4,
                                   int w, int h) {
  const int16_t *const filter_x = filter[x0_q4];
  assert(x_step_q4 == 16);
  assert(((const int32_t *)filter_x)[1] != 0x800000);

  if (vpx_get_filter_taps(filter_x) == 2) {
    vpx_convolve2_avg_horiz_dspr2(src, src_stride, dst, dst_stride, filter,
                                  x0_q4, x_step_q4, y0_q4, y_step_q4, w, h);
  } else {
    uint32_t pos = 38;

    src -= 3;

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
        convolve_avg_horiz_4_dspr2(src, src_stride, dst, dst_stride, filter_x,
                                   h);
        break;
      case 8:
        convolve_avg_horiz_8_dspr2(src, src_stride, dst, dst_stride, filter_x,
                                   h);
        break;
      case 16:
        convolve_avg_horiz_16_dspr2(src, src_stride, dst, dst_stride, filter_x,
                                    h, 1);
        break;
      case 32:
        convolve_avg_horiz_16_dspr2(src, src_stride, dst, dst_stride, filter_x,
                                    h, 2);
        break;
      case 64:
        prefetch_load(src + 64);
        prefetch_store(dst + 32);

        convolve_avg_horiz_64_dspr2(src, src_stride, dst, dst_stride, filter_x,
                                    h);
        break;
      default:
        vpx_convolve8_avg_horiz_c(src + 3, src_stride, dst, dst_stride, filter,
                                  x0_q4, x_step_q4, y0_q4, y_step_q4, w, h);
        break;
    }
  }
}
#endif
