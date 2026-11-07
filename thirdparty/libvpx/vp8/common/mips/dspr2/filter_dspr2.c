/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <stdlib.h>
#include "vp8_rtcd.h"
#include "vpx_ports/mem.h"

#if HAVE_DSPR2
#define CROP_WIDTH 256
unsigned char ff_cropTbl[256 + 2 * CROP_WIDTH];

static const unsigned short sub_pel_filterss[8][3] = {
  { 0, 0, 0 },
  { 0, 0x0601, 0x7b0c },
  { 0x0201, 0x0b08, 0x6c24 },
  { 0, 0x0906, 0x5d32 },
  { 0x0303, 0x1010, 0x4d4d },
  { 0, 0x0609, 0x325d },
  { 0x0102, 0x080b, 0x246c },
  { 0, 0x0106, 0x0c7b },
};

static const int sub_pel_filters_int[8][3] = {
  { 0, 0, 0 },
  { 0x0000fffa, 0x007b000c, 0xffff0000 },
  { 0x0002fff5, 0x006c0024, 0xfff80001 },
  { 0x0000fff7, 0x005d0032, 0xfffa0000 },
  { 0x0003fff0, 0x004d004d, 0xfff00003 },
  { 0x0000fffa, 0x0032005d, 0xfff70000 },
  { 0x0001fff8, 0x0024006c, 0xfff50002 },
  { 0x0000ffff, 0x000c007b, 0xfffa0000 },
};

static const int sub_pel_filters_inv[8][3] = {
  { 0, 0, 0 },
  { 0xfffa0000, 0x000c007b, 0x0000ffff },
  { 0xfff50002, 0x0024006c, 0x0001fff8 },
  { 0xfff70000, 0x0032005d, 0x0000fffa },
  { 0xfff00003, 0x004d004d, 0x0003fff0 },
  { 0xfffa0000, 0x005d0032, 0x0000fff7 },
  { 0xfff80001, 0x006c0024, 0x0002fff5 },
  { 0xffff0000, 0x007b000c, 0x0000fffa },
};

/* clang-format off */
static const int sub_pel_filters_int_tap_4[8][2] = {
  {          0,          0},
  { 0xfffa007b, 0x000cffff},
  {          0,          0},
  { 0xfff7005d, 0x0032fffa},
  {          0,          0},
  { 0xfffa0032, 0x005dfff7},
  {          0,          0},
  { 0xffff000c, 0x007bfffa},
};


static const int sub_pel_filters_inv_tap_4[8][2] = {
  {          0,          0},
  { 0x007bfffa, 0xffff000c},
  {          0,          0},
  { 0x005dfff7, 0xfffa0032},
  {          0,          0},
  { 0x0032fffa, 0xfff7005d},
  {          0,          0},
  { 0x000cffff, 0xfffa007b},
};
/* clang-format on */

inline void prefetch_load(unsigned char *src) {
  __asm__ __volatile__("pref   0,  0(%[src])   \n\t" : : [src] "r"(src));
}

inline void prefetch_store(unsigned char *dst) {
  __asm__ __volatile__("pref   1,  0(%[dst])   \n\t" : : [dst] "r"(dst));
}

void dsputil_static_init(void) {
  int i;

  for (i = 0; i < 256; ++i) ff_cropTbl[i + CROP_WIDTH] = i;

  for (i = 0; i < CROP_WIDTH; ++i) {
    ff_cropTbl[i] = 0;
    ff_cropTbl[i + CROP_WIDTH + 256] = 255;
  }
}

void vp8_filter_block2d_first_pass_4(unsigned char *RESTRICT src_ptr,
                                     unsigned char *RESTRICT dst_ptr,
                                     unsigned int src_pixels_per_line,
                                     unsigned int output_height, int xoffset,
                                     int pitch) {
  unsigned int i;
  int Temp1, Temp2, Temp3, Temp4;

  unsigned int vector4a = 64;
  int vector1b, vector2b, vector3b;
  unsigned int tp1, tp2, tn1, tn2;
  unsigned int p1, p2, p3;
  unsigned int n1, n2, n3;
  unsigned char *cm = ff_cropTbl + CROP_WIDTH;

  vector3b = sub_pel_filters_inv[xoffset][2];

  /* if (xoffset == 0) we don't need any filtering */
  if (vector3b == 0) {
    for (i = 0; i < output_height; ++i) {
      /* prefetch src_ptr data to cache memory */
      prefetch_load(src_ptr + src_pixels_per_line);
      dst_ptr[0] = src_ptr[0];
      dst_ptr[1] = src_ptr[1];
      dst_ptr[2] = src_ptr[2];
      dst_ptr[3] = src_ptr[3];

      /* next row... */
      src_ptr += src_pixels_per_line;
      dst_ptr += 4;
    }
  } else {
    if (vector3b > 65536) {
      /* 6 tap filter */

      vector1b = sub_pel_filters_inv[xoffset][0];
      vector2b = sub_pel_filters_inv[xoffset][1];

      /* prefetch src_ptr data to cache memory */
      prefetch_load(src_ptr + src_pixels_per_line);

      for (i = output_height; i--;) {
        /* apply filter with vectors pairs */
        __asm__ __volatile__(
            "ulw              %[tp1],      -2(%[src_ptr])                 \n\t"
            "ulw              %[tp2],      2(%[src_ptr])                  \n\t"

            /* even 1. pixel */
            "mtlo             %[vector4a], $ac3                           \n\t"
            "preceu.ph.qbr    %[p1],       %[tp1]                         \n\t"
            "preceu.ph.qbl    %[p2],       %[tp1]                         \n\t"
            "preceu.ph.qbr    %[p3],       %[tp2]                         \n\t"
            "dpa.w.ph         $ac3,        %[p1],          %[vector1b]    \n\t"
            "dpa.w.ph         $ac3,        %[p2],          %[vector2b]    \n\t"
            "dpa.w.ph         $ac3,        %[p3],          %[vector3b]    \n\t"

            /* even 2. pixel */
            "mtlo             %[vector4a], $ac2                           \n\t"
            "preceu.ph.qbl    %[p1],       %[tp2]                         \n\t"
            "balign           %[tp2],      %[tp1],         3              \n\t"
            "extp             %[Temp1],    $ac3,           9              \n\t"
            "dpa.w.ph         $ac2,        %[p2],          %[vector1b]    \n\t"
            "dpa.w.ph         $ac2,        %[p3],          %[vector2b]    \n\t"
            "dpa.w.ph         $ac2,        %[p1],          %[vector3b]    \n\t"

            /* odd 1. pixel */
            "ulw              %[tn2],      3(%[src_ptr])                  \n\t"
            "mtlo             %[vector4a], $ac3                           \n\t"
            "preceu.ph.qbr    %[n1],       %[tp2]                         \n\t"
            "preceu.ph.qbl    %[n2],       %[tp2]                         \n\t"
            "preceu.ph.qbr    %[n3],       %[tn2]                         \n\t"
            "extp             %[Temp3],    $ac2,           9              \n\t"
            "dpa.w.ph         $ac3,        %[n1],          %[vector1b]    \n\t"
            "dpa.w.ph         $ac3,        %[n2],          %[vector2b]    \n\t"
            "dpa.w.ph         $ac3,        %[n3],          %[vector3b]    \n\t"

            /* even 2. pixel */
            "mtlo             %[vector4a], $ac2                           \n\t"
            "preceu.ph.qbl    %[n1],       %[tn2]                         \n\t"
            "extp             %[Temp2],    $ac3,           9              \n\t"
            "dpa.w.ph         $ac2,        %[n2],          %[vector1b]    \n\t"
            "dpa.w.ph         $ac2,        %[n3],          %[vector2b]    \n\t"
            "dpa.w.ph         $ac2,        %[n1],          %[vector3b]    \n\t"
            "extp             %[Temp4],    $ac2,           9              \n\t"

            /* clamp */
            "lbux             %[tp1],      %[Temp1](%[cm])                \n\t"
            "lbux             %[tn1],      %[Temp2](%[cm])                \n\t"
            "lbux             %[tp2],      %[Temp3](%[cm])                \n\t"
            "lbux             %[n2],       %[Temp4](%[cm])                \n\t"

            /* store bytes */
            "sb               %[tp1],      0(%[dst_ptr])                  \n\t"
            "sb               %[tn1],      1(%[dst_ptr])                  \n\t"
            "sb               %[tp2],      2(%[dst_ptr])                  \n\t"
            "sb               %[n2],       3(%[dst_ptr])                  \n\t"

            : [tp1] "=&r"(tp1), [tp2] "=&r"(tp2), [tn1] "=&r"(tn1),
              [tn2] "=&r"(tn2), [p1] "=&r"(p1), [p2] "=&r"(p2), [p3] "=&r"(p3),
              [n1] "=&r"(n1), [n2] "=&r"(n2), [n3] "=&r"(n3),
              [Temp1] "=&r"(Temp1), [Temp2] "=&r"(Temp2), [Temp3] "=&r"(Temp3),
              [Temp4] "=&r"(Temp4)
            : [vector1b] "r"(vector1b), [vector2b] "r"(vector2b),
              [vector4a] "r"(vector4a), [cm] "r"(cm), [dst_ptr] "r"(dst_ptr),
              [vector3b] "r"(vector3b), [src_ptr] "r"(src_ptr));

        /* Next row... */
        src_ptr += src_pixels_per_line;
        dst_ptr += pitch;
      }
    } else {
      /* 4 tap filter */

      vector1b = sub_pel_filters_inv_tap_4[xoffset][0];
      vector2b = sub_pel_filters_inv_tap_4[xoffset][1];

      for (i = output_height; i--;) {
        /* apply filter with vectors pairs */
        __asm__ __volatile__(
            "ulw              %[tp1],      -1(%[src_ptr])                 \n\t"
            "ulw              %[tp2],      3(%[src_ptr])                  \n\t"

            /* even 1. pixel */
            "mtlo             %[vector4a], $ac3                           \n\t"
            "preceu.ph.qbr    %[p1],       %[tp1]                         \n\t"
            "preceu.ph.qbl    %[p2],       %[tp1]                         \n\t"
            "preceu.ph.qbr    %[p3],       %[tp2]                         \n\t"
            "dpa.w.ph         $ac3,        %[p1],          %[vector1b]    \n\t"
            "dpa.w.ph         $ac3,        %[p2],          %[vector2b]    \n\t"

            /* even 2. pixel */
            "mtlo             %[vector4a], $ac2                           \n\t"
            "dpa.w.ph         $ac2,        %[p2],          %[vector1b]    \n\t"
            "dpa.w.ph         $ac2,        %[p3],          %[vector2b]    \n\t"
            "extp             %[Temp1],    $ac3,           9              \n\t"

            /* odd 1. pixel */
            "srl              %[tn1],      %[tp2],         8              \n\t"
            "balign           %[tp2],      %[tp1],         3              \n\t"
            "mtlo             %[vector4a], $ac3                           \n\t"
            "preceu.ph.qbr    %[n1],       %[tp2]                         \n\t"
            "preceu.ph.qbl    %[n2],       %[tp2]                         \n\t"
            "preceu.ph.qbr    %[n3],       %[tn1]                         \n\t"
            "extp             %[Temp3],    $ac2,           9              \n\t"
            "dpa.w.ph         $ac3,        %[n1],          %[vector1b]    \n\t"
            "dpa.w.ph         $ac3,        %[n2],          %[vector2b]    \n\t"

            /* odd 2. pixel */
            "mtlo             %[vector4a], $ac2                           \n\t"
            "extp             %[Temp2],    $ac3,           9              \n\t"
            "dpa.w.ph         $ac2,        %[n2],          %[vector1b]    \n\t"
            "dpa.w.ph         $ac2,        %[n3],          %[vector2b]    \n\t"
            "extp             %[Temp4],    $ac2,           9              \n\t"

            /* clamp and store results */
            "lbux             %[tp1],      %[Temp1](%[cm])                \n\t"
            "lbux             %[tn1],      %[Temp2](%[cm])                \n\t"
            "lbux             %[tp2],      %[Temp3](%[cm])                \n\t"
            "sb               %[tp1],      0(%[dst_ptr])                  \n\t"
            "sb               %[tn1],      1(%[dst_ptr])                  \n\t"
            "lbux             %[n2],       %[Temp4](%[cm])                \n\t"
            "sb               %[tp2],      2(%[dst_ptr])                  \n\t"
            "sb               %[n2],       3(%[dst_ptr])                  \n\t"

            : [tp1] "=&r"(tp1), [tp2] "=&r"(tp2), [tn1] "=&r"(tn1),
              [p1] "=&r"(p1), [p2] "=&r"(p2), [p3] "=&r"(p3), [n1] "=&r"(n1),
              [n2] "=&r"(n2), [n3] "=&r"(n3), [Temp1] "=&r"(Temp1),
              [Temp2] "=&r"(Temp2), [Temp3] "=&r"(Temp3), [Temp4] "=&r"(Temp4)
            : [vector1b] "r"(vector1b), [vector2b] "r"(vector2b),
              [vector4a] "r"(vector4a), [cm] "r"(cm), [dst_ptr] "r"(dst_ptr),
              [src_ptr] "r"(src_ptr));
        /*  Next row... */
        src_ptr += src_pixels_per_line;
        dst_ptr += pitch;
      }
    }
  }
}

void vp8_filter_block2d_first_pass_8_all(unsigned char *RESTRICT src_ptr,
                                         unsigned char *RESTRICT dst_ptr,
                                         unsigned int src_pixels_per_line,
                                         unsigned int output_height,
                                         int xoffset, int pitch) {
  unsigned int i;
  int Temp1, Temp2, Temp3, Temp4;

  unsigned int vector4a = 64;
  unsigned int vector1b, vector2b, vector3b;
  unsigned int tp1, tp2, tn1, tn2;
  unsigned int p1, p2, p3, p4;
  unsigned int n1, n2, n3, n4;

  unsigned char *cm = ff_cropTbl + CROP_WIDTH;

  /* if (xoffset == 0) we don't need any filtering */
  if (xoffset == 0) {
    for (i = 0; i < output_height; ++i) {
      /* prefetch src_ptr data to cache memory */
      prefetch_load(src_ptr + src_pixels_per_line);

      dst_ptr[0] = src_ptr[0];
      dst_ptr[1] = src_ptr[1];
      dst_ptr[2] = src_ptr[2];
      dst_ptr[3] = src_ptr[3];
      dst_ptr[4] = src_ptr[4];
      dst_ptr[5] = src_ptr[5];
      dst_ptr[6] = src_ptr[6];
      dst_ptr[7] = src_ptr[7];

      /* next row... */
      src_ptr += src_pixels_per_line;
      dst_ptr += 8;
    }
  } else {
    vector3b = sub_pel_filters_inv[xoffset][2];

    if (vector3b > 65536) {
      /* 6 tap filter */

      vector1b = sub_pel_filters_inv[xoffset][0];
      vector2b = sub_pel_filters_inv[xoffset][1];

      for (i = output_height; i--;) {
        /* prefetch src_ptr data to cache memory */
        prefetch_load(src_ptr + src_pixels_per_line);

        /* apply filter with vectors pairs */
        __asm__ __volatile__(
            "ulw              %[tp1],      -2(%[src_ptr])                 \n\t"
            "ulw              %[tp2],      2(%[src_ptr])                  \n\t"

            /* even 1. pixel */
            "mtlo             %[vector4a], $ac3                           \n\t"
            "preceu.ph.qbr    %[p1],       %[tp1]                         \n\t"
            "preceu.ph.qbl    %[p2],       %[tp1]                         \n\t"
            "preceu.ph.qbr    %[p3],       %[tp2]                         \n\t"
            "dpa.w.ph         $ac3,        %[p1],          %[vector1b]    \n\t"
            "dpa.w.ph         $ac3,        %[p2],          %[vector2b]    \n\t"
            "dpa.w.ph         $ac3,        %[p3],          %[vector3b]    \n\t"

            /* even 2. pixel */
            "mtlo             %[vector4a], $ac2                           \n\t"
            "preceu.ph.qbl    %[p1],       %[tp2]                         \n\t"
            "dpa.w.ph         $ac2,        %[p2],          %[vector1b]    \n\t"
            "dpa.w.ph         $ac2,        %[p3],          %[vector2b]    \n\t"
            "dpa.w.ph         $ac2,        %[p1],          %[vector3b]    \n\t"

            "balign           %[tp2],      %[tp1],         3              \n\t"
            "extp             %[Temp1],    $ac3,           9              \n\t"
            "ulw              %[tn2],      3(%[src_ptr])                  \n\t"

            /* odd 1. pixel */
            "mtlo             %[vector4a], $ac3                           \n\t"
            "preceu.ph.qbr    %[n1],       %[tp2]                         \n\t"
            "preceu.ph.qbl    %[n2],       %[tp2]                         \n\t"
            "preceu.ph.qbr    %[n3],       %[tn2]                         \n\t"
            "extp             %[Temp3],    $ac2,           9              \n\t"
            "dpa.w.ph         $ac3,        %[n1],          %[vector1b]    \n\t"
            "dpa.w.ph         $ac3,        %[n2],          %[vector2b]    \n\t"
            "dpa.w.ph         $ac3,        %[n3],          %[vector3b]    \n\t"

            /* odd 2. pixel */
            "mtlo             %[vector4a], $ac2                           \n\t"
            "preceu.ph.qbl    %[n1],       %[tn2]                         \n\t"
            "dpa.w.ph         $ac2,        %[n2],          %[vector1b]    \n\t"
            "dpa.w.ph         $ac2,        %[n3],          %[vector2b]    \n\t"
            "dpa.w.ph         $ac2,        %[n1],          %[vector3b]    \n\t"
            "ulw              %[tp1],      6(%[src_ptr])                  \n\t"
            "extp             %[Temp2],    $ac3,           9              \n\t"
            "mtlo             %[vector4a], $ac3                           \n\t"
            "preceu.ph.qbr    %[p2],       %[tp1]                         \n\t"
            "extp             %[Temp4],    $ac2,           9              \n\t"

            : [tp1] "=&r"(tp1), [tp2] "=&r"(tp2), [tn2] "=&r"(tn2),
              [p1] "=&r"(p1), [p2] "=&r"(p2), [p3] "=&r"(p3), [n1] "=&r"(n1),
              [n2] "=&r"(n2), [n3] "=&r"(n3), [Temp1] "=&r"(Temp1),
              [Temp2] "=&r"(Temp2), [Temp3] "=&r"(Temp3), [Temp4] "=r"(Temp4)
            : [vector1b] "r"(vector1b), [vector2b] "r"(vector2b),
              [vector4a] "r"(vector4a), [vector3b] "r"(vector3b),
              [src_ptr] "r"(src_ptr));

        /* clamp and store results */
        dst_ptr[0] = cm[Temp1];
        dst_ptr[1] = cm[Temp2];
        dst_ptr[2] = cm[Temp3];
        dst_ptr[3] = cm[Temp4];

        /* next 4 pixels */
        __asm__ __volatile__(
            /* even 3. pixel */
            "dpa.w.ph         $ac3,        %[p3],          %[vector1b]    \n\t"
            "dpa.w.ph         $ac3,        %[p1],          %[vector2b]    \n\t"
            "dpa.w.ph         $ac3,        %[p2],          %[vector3b]    \n\t"

            /* even 4. pixel */
            "mtlo             %[vector4a], $ac2                           \n\t"
            "preceu.ph.qbl    %[p4],       %[tp1]                         \n\t"
            "dpa.w.ph         $ac2,        %[p1],          %[vector1b]    \n\t"
            "dpa.w.ph         $ac2,        %[p2],          %[vector2b]    \n\t"
            "dpa.w.ph         $ac2,        %[p4],          %[vector3b]    \n\t"

            "ulw              %[tn1],      7(%[src_ptr])                  \n\t"
            "extp             %[Temp1],    $ac3,           9              \n\t"

            /* odd 3. pixel */
            "mtlo             %[vector4a], $ac3                           \n\t"
            "preceu.ph.qbr    %[n2],       %[tn1]                         \n\t"
            "dpa.w.ph         $ac3,        %[n3],          %[vector1b]    \n\t"
            "dpa.w.ph         $ac3,        %[n1],          %[vector2b]    \n\t"
            "dpa.w.ph         $ac3,        %[n2],          %[vector3b]    \n\t"
            "extp             %[Temp3],    $ac2,           9              \n\t"

            /* odd 4. pixel */
            "mtlo             %[vector4a], $ac2                           \n\t"
            "preceu.ph.qbl    %[n4],       %[tn1]                         \n\t"
            "dpa.w.ph         $ac2,        %[n1],          %[vector1b]    \n\t"
            "dpa.w.ph         $ac2,        %[n2],          %[vector2b]    \n\t"
            "dpa.w.ph         $ac2,        %[n4],          %[vector3b]    \n\t"
            "extp             %[Temp2],    $ac3,           9              \n\t"
            "extp             %[Temp4],    $ac2,           9              \n\t"

            : [tn1] "=&r"(tn1), [n2] "=&r"(n2), [p4] "=&r"(p4), [n4] "=&r"(n4),
              [Temp1] "=&r"(Temp1), [Temp2] "=&r"(Temp2), [Temp3] "=&r"(Temp3),
              [Temp4] "=r"(Temp4)
            : [tp1] "r"(tp1), [vector1b] "r"(vector1b), [p2] "r"(p2),
              [vector2b] "r"(vector2b), [n1] "r"(n1), [p1] "r"(p1),
              [vector4a] "r"(vector4a), [vector3b] "r"(vector3b), [p3] "r"(p3),
              [n3] "r"(n3), [src_ptr] "r"(src_ptr));

        /* clamp and store results */
        dst_ptr[4] = cm[Temp1];
        dst_ptr[5] = cm[Temp2];
        dst_ptr[6] = cm[Temp3];
        dst_ptr[7] = cm[Temp4];

        src_ptr += src_pixels_per_line;
        dst_ptr += pitch;
      }
    } else {
      /* 4 tap filter */

      vector1b = sub_pel_filters_inv_tap_4[xoffset][0];
      vector2b = sub_pel_filters_inv_tap_4[xoffset][1];

      for (i = output_height; i--;) {
        /* prefetch src_ptr data to cache memory */
        prefetch_load(src_ptr + src_pixels_per_line);

        /* apply filter with vectors pairs */
        __asm__ __volatile__(
            "ulw              %[tp1],      -1(%[src_ptr])                 \n\t"

            /* even 1. pixel */
            "mtlo             %[vector4a], $ac3                           \n\t"
            "preceu.ph.qbr    %[p1],       %[tp1]                         \n\t"
            "preceu.ph.qbl    %[p2],       %[tp1]                         \n\t"
            "dpa.w.ph         $ac3,        %[p1],          %[vector1b]    \n\t"
            "dpa.w.ph         $ac3,        %[p2],          %[vector2b]    \n\t"

            "ulw              %[tp2],      3(%[src_ptr])                  \n\t"

            /* even 2. pixel  */
            "mtlo             %[vector4a], $ac2                           \n\t"
            "preceu.ph.qbr    %[p3],       %[tp2]                         \n\t"
            "preceu.ph.qbl    %[p4],       %[tp2]                         \n\t"
            "dpa.w.ph         $ac2,        %[p2],          %[vector1b]    \n\t"
            "dpa.w.ph         $ac2,        %[p3],          %[vector2b]    \n\t"
            "extp             %[Temp1],    $ac3,           9              \n\t"

            "balign           %[tp2],      %[tp1],         3              \n\t"

            /* odd 1. pixel */
            "mtlo             %[vector4a], $ac3                           \n\t"
            "preceu.ph.qbr    %[n1],       %[tp2]                         \n\t"
            "preceu.ph.qbl    %[n2],       %[tp2]                         \n\t"
            "dpa.w.ph         $ac3,        %[n1],          %[vector1b]    \n\t"
            "dpa.w.ph         $ac3,        %[n2],          %[vector2b]    \n\t"
            "extp             %[Temp3],    $ac2,           9              \n\t"

            "ulw              %[tn2],      4(%[src_ptr])                  \n\t"

            /* odd 2. pixel */
            "mtlo             %[vector4a], $ac2                           \n\t"
            "preceu.ph.qbr    %[n3],       %[tn2]                         \n\t"
            "preceu.ph.qbl    %[n4],       %[tn2]                         \n\t"
            "dpa.w.ph         $ac2,        %[n2],          %[vector1b]    \n\t"
            "dpa.w.ph         $ac2,        %[n3],          %[vector2b]    \n\t"
            "ulw              %[tp1],      7(%[src_ptr])                  \n\t"
            "extp             %[Temp2],    $ac3,           9              \n\t"
            "mtlo             %[vector4a], $ac3                           \n\t"
            "extp             %[Temp4],    $ac2,           9              \n\t"

            : [tp1] "=&r"(tp1), [tp2] "=&r"(tp2), [tn2] "=&r"(tn2),
              [p1] "=&r"(p1), [p2] "=&r"(p2), [p3] "=&r"(p3), [p4] "=&r"(p4),
              [n1] "=&r"(n1), [n2] "=&r"(n2), [n3] "=&r"(n3), [n4] "=&r"(n4),
              [Temp1] "=&r"(Temp1), [Temp2] "=&r"(Temp2), [Temp3] "=&r"(Temp3),
              [Temp4] "=r"(Temp4)
            : [vector1b] "r"(vector1b), [vector2b] "r"(vector2b),
              [vector4a] "r"(vector4a), [src_ptr] "r"(src_ptr));

        /* clamp and store results */
        dst_ptr[0] = cm[Temp1];
        dst_ptr[1] = cm[Temp2];
        dst_ptr[2] = cm[Temp3];
        dst_ptr[3] = cm[Temp4];

        /* next 4 pixels */
        __asm__ __volatile__(
            /* even 3. pixel */
            "dpa.w.ph         $ac3,        %[p3],          %[vector1b]    \n\t"
            "dpa.w.ph         $ac3,        %[p4],          %[vector2b]    \n\t"

            /* even 4. pixel */
            "mtlo             %[vector4a], $ac2                           \n\t"
            "preceu.ph.qbr    %[p2],       %[tp1]                         \n\t"
            "dpa.w.ph         $ac2,        %[p4],          %[vector1b]    \n\t"
            "dpa.w.ph         $ac2,        %[p2],          %[vector2b]    \n\t"
            "extp             %[Temp1],    $ac3,           9              \n\t"

            /* odd 3. pixel */
            "mtlo             %[vector4a], $ac3                           \n\t"
            "dpa.w.ph         $ac3,        %[n3],          %[vector1b]    \n\t"
            "dpa.w.ph         $ac3,        %[n4],          %[vector2b]    \n\t"
            "ulw              %[tn1],      8(%[src_ptr])                  \n\t"
            "extp             %[Temp3],    $ac2,           9              \n\t"

            /* odd 4. pixel */
            "mtlo             %[vector4a], $ac2                           \n\t"
            "preceu.ph.qbr    %[n2],       %[tn1]                         \n\t"
            "dpa.w.ph         $ac2,        %[n4],          %[vector1b]    \n\t"
            "dpa.w.ph         $ac2,        %[n2],          %[vector2b]    \n\t"
            "extp             %[Temp2],    $ac3,           9              \n\t"
            "extp             %[Temp4],    $ac2,           9              \n\t"

            : [tn1] "=&r"(tn1), [p2] "=&r"(p2), [n2] "=&r"(n2),
              [Temp1] "=&r"(Temp1), [Temp2] "=&r"(Temp2), [Temp3] "=&r"(Temp3),
              [Temp4] "=r"(Temp4)
            : [tp1] "r"(tp1), [p3] "r"(p3), [p4] "r"(p4),
              [vector1b] "r"(vector1b), [vector2b] "r"(vector2b),
              [vector4a] "r"(vector4a), [src_ptr] "r"(src_ptr), [n3] "r"(n3),
              [n4] "r"(n4));

        /* clamp and store results */
        dst_ptr[4] = cm[Temp1];
        dst_ptr[5] = cm[Temp2];
        dst_ptr[6] = cm[Temp3];
        dst_ptr[7] = cm[Temp4];

        /* next row... */
        src_ptr += src_pixels_per_line;
        dst_ptr += pitch;
      }
    }
  }
}

void vp8_filter_block2d_first_pass16_6tap(unsigned char *RESTRICT src_ptr,
                                          unsigned char *RESTRICT dst_ptr,
                                          unsigned int src_pixels_per_line,
                                          unsigned int output_height,
                                          int xoffset, int pitch) {
  unsigned int i;
  int Temp1, Temp2, Temp3, Temp4;

  unsigned int vector4a;
  unsigned int vector1b, vector2b, vector3b;
  unsigned int tp1, tp2, tn1, tn2;
  unsigned int p1, p2, p3, p4;
  unsigned int n1, n2, n3, n4;
  unsigned char *cm = ff_cropTbl + CROP_WIDTH;

  vector1b = sub_pel_filters_inv[xoffset][0];
  vector2b = sub_pel_filters_inv[xoffset][1];
  vector3b = sub_pel_filters_inv[xoffset][2];
  vector4a = 64;

  for (i = output_height; i--;) {
    /* prefetch src_ptr data to cache memory */
    prefetch_load(src_ptr + src_pixels_per_line);

    /* apply filter with vectors pairs */
    __asm__ __volatile__(
        "ulw                %[tp1],      -2(%[src_ptr])                 \n\t"
        "ulw                %[tp2],      2(%[src_ptr])                  \n\t"

        /* even 1. pixel */
        "mtlo               %[vector4a], $ac3                           \n\t"
        "preceu.ph.qbr      %[p1],       %[tp1]                         \n\t"
        "preceu.ph.qbl      %[p2],       %[tp1]                         \n\t"
        "preceu.ph.qbr      %[p3],       %[tp2]                         \n\t"
        "dpa.w.ph           $ac3,        %[p1],           %[vector1b]   \n\t"
        "dpa.w.ph           $ac3,        %[p2],           %[vector2b]   \n\t"
        "dpa.w.ph           $ac3,        %[p3],           %[vector3b]   \n\t"

        /* even 2. pixel */
        "mtlo               %[vector4a], $ac2                           \n\t"
        "preceu.ph.qbl      %[p1],       %[tp2]                         \n\t"
        "dpa.w.ph           $ac2,        %[p2],           %[vector1b]   \n\t"
        "dpa.w.ph           $ac2,        %[p3],           %[vector2b]   \n\t"
        "dpa.w.ph           $ac2,        %[p1],           %[vector3b]   \n\t"

        "balign             %[tp2],      %[tp1],          3             \n\t"
        "ulw                %[tn2],      3(%[src_ptr])                  \n\t"
        "extp               %[Temp1],    $ac3,            9             \n\t"

        /* odd 1. pixel */
        "mtlo               %[vector4a], $ac3                           \n\t"
        "preceu.ph.qbr      %[n1],       %[tp2]                         \n\t"
        "preceu.ph.qbl      %[n2],       %[tp2]                         \n\t"
        "preceu.ph.qbr      %[n3],       %[tn2]                         \n\t"
        "extp               %[Temp3],    $ac2,            9             \n\t"
        "dpa.w.ph           $ac3,        %[n1],           %[vector1b]   \n\t"
        "dpa.w.ph           $ac3,        %[n2],           %[vector2b]   \n\t"
        "dpa.w.ph           $ac3,        %[n3],           %[vector3b]   \n\t"

        /* odd 2. pixel */
        "mtlo               %[vector4a], $ac2                           \n\t"
        "preceu.ph.qbl      %[n1],       %[tn2]                         \n\t"
        "dpa.w.ph           $ac2,        %[n2],           %[vector1b]   \n\t"
        "dpa.w.ph           $ac2,        %[n3],           %[vector2b]   \n\t"
        "dpa.w.ph           $ac2,        %[n1],           %[vector3b]   \n\t"
        "ulw                %[tp1],      6(%[src_ptr])                  \n\t"
        "extp               %[Temp2],    $ac3,            9             \n\t"
        "mtlo               %[vector4a], $ac3                           \n\t"
        "preceu.ph.qbr      %[p2],       %[tp1]                         \n\t"
        "extp               %[Temp4],    $ac2,            9             \n\t"

        : [tp1] "=&r"(tp1), [tp2] "=&r"(tp2), [tn2] "=&r"(tn2), [p1] "=&r"(p1),
          [p2] "=&r"(p2), [p3] "=&r"(p3), [n1] "=&r"(n1), [n2] "=&r"(n2),
          [n3] "=&r"(n3), [Temp1] "=&r"(Temp1), [Temp2] "=&r"(Temp2),
          [Temp3] "=&r"(Temp3), [Temp4] "=r"(Temp4)
        : [vector1b] "r"(vector1b), [vector2b] "r"(vector2b),
          [vector4a] "r"(vector4a), [vector3b] "r"(vector3b),
          [src_ptr] "r"(src_ptr));

    /* clamp and store results */
    dst_ptr[0] = cm[Temp1];
    dst_ptr[1] = cm[Temp2];
    dst_ptr[2] = cm[Temp3];
    dst_ptr[3] = cm[Temp4];

    /* next 4 pixels */
    __asm__ __volatile__(
        /* even 3. pixel */
        "dpa.w.ph           $ac3,        %[p3],           %[vector1b]   \n\t"
        "dpa.w.ph           $ac3,        %[p1],           %[vector2b]   \n\t"
        "dpa.w.ph           $ac3,        %[p2],           %[vector3b]   \n\t"

        /* even 4. pixel */
        "mtlo               %[vector4a], $ac2                           \n\t"
        "preceu.ph.qbl      %[p4],       %[tp1]                         \n\t"
        "dpa.w.ph           $ac2,        %[p1],           %[vector1b]   \n\t"
        "dpa.w.ph           $ac2,        %[p2],           %[vector2b]   \n\t"
        "dpa.w.ph           $ac2,        %[p4],           %[vector3b]   \n\t"
        "ulw                %[tn1],      7(%[src_ptr])                  \n\t"
        "extp               %[Temp1],    $ac3,            9             \n\t"

        /* odd 3. pixel */
        "mtlo               %[vector4a], $ac3                           \n\t"
        "preceu.ph.qbr      %[n2],       %[tn1]                         \n\t"
        "dpa.w.ph           $ac3,        %[n3],           %[vector1b]   \n\t"
        "dpa.w.ph           $ac3,        %[n1],           %[vector2b]   \n\t"
        "dpa.w.ph           $ac3,        %[n2],           %[vector3b]   \n\t"
        "extp               %[Temp3],    $ac2,            9             \n\t"

        /* odd 4. pixel */
        "mtlo               %[vector4a], $ac2                           \n\t"
        "preceu.ph.qbl      %[n4],       %[tn1]                         \n\t"
        "dpa.w.ph           $ac2,        %[n1],           %[vector1b]   \n\t"
        "dpa.w.ph           $ac2,        %[n2],           %[vector2b]   \n\t"
        "dpa.w.ph           $ac2,        %[n4],           %[vector3b]   \n\t"
        "ulw                %[tp2],      10(%[src_ptr])                 \n\t"
        "extp               %[Temp2],    $ac3,            9             \n\t"
        "mtlo               %[vector4a], $ac3                           \n\t"
        "preceu.ph.qbr      %[p1],       %[tp2]                         \n\t"
        "extp               %[Temp4],    $ac2,            9             \n\t"

        : [tn1] "=&r"(tn1), [tp2] "=&r"(tp2), [n2] "=&r"(n2), [p4] "=&r"(p4),
          [n4] "=&r"(n4), [Temp1] "=&r"(Temp1), [Temp2] "=&r"(Temp2),
          [Temp3] "=&r"(Temp3), [Temp4] "=r"(Temp4), [p1] "+r"(p1)
        : [vector1b] "r"(vector1b), [vector2b] "r"(vector2b), [tp1] "r"(tp1),
          [n1] "r"(n1), [vector4a] "r"(vector4a), [p2] "r"(p2),
          [vector3b] "r"(vector3b), [p3] "r"(p3), [n3] "r"(n3),
          [src_ptr] "r"(src_ptr));

    /* clamp and store results */
    dst_ptr[4] = cm[Temp1];
    dst_ptr[5] = cm[Temp2];
    dst_ptr[6] = cm[Temp3];
    dst_ptr[7] = cm[Temp4];

    /* next 4 pixels */
    __asm__ __volatile__(
        /* even 5. pixel */
        "dpa.w.ph           $ac3,        %[p2],           %[vector1b]   \n\t"
        "dpa.w.ph           $ac3,        %[p4],           %[vector2b]   \n\t"
        "dpa.w.ph           $ac3,        %[p1],           %[vector3b]   \n\t"

        /* even 6. pixel */
        "mtlo               %[vector4a], $ac2                           \n\t"
        "preceu.ph.qbl      %[p3],       %[tp2]                         \n\t"
        "dpa.w.ph           $ac2,        %[p4],           %[vector1b]   \n\t"
        "dpa.w.ph           $ac2,        %[p1],           %[vector2b]   \n\t"
        "dpa.w.ph           $ac2,        %[p3],           %[vector3b]   \n\t"

        "ulw                %[tn1],      11(%[src_ptr])                 \n\t"
        "extp               %[Temp1],    $ac3,            9             \n\t"

        /* odd 5. pixel */
        "mtlo               %[vector4a], $ac3                           \n\t"
        "preceu.ph.qbr      %[n1],       %[tn1]                         \n\t"
        "dpa.w.ph           $ac3,        %[n2],           %[vector1b]   \n\t"
        "dpa.w.ph           $ac3,        %[n4],           %[vector2b]   \n\t"
        "dpa.w.ph           $ac3,        %[n1],           %[vector3b]   \n\t"
        "extp               %[Temp3],    $ac2,            9             \n\t"

        /* odd 6. pixel */
        "mtlo               %[vector4a], $ac2                           \n\t"
        "preceu.ph.qbl      %[n3],       %[tn1]                         \n\t"
        "dpa.w.ph           $ac2,        %[n4],           %[vector1b]   \n\t"
        "dpa.w.ph           $ac2,        %[n1],           %[vector2b]   \n\t"
        "dpa.w.ph           $ac2,        %[n3],           %[vector3b]   \n\t"
        "ulw                %[tp1],      14(%[src_ptr])                 \n\t"
        "extp               %[Temp2],    $ac3,            9             \n\t"
        "mtlo               %[vector4a], $ac3                           \n\t"
        "preceu.ph.qbr      %[p4],       %[tp1]                         \n\t"
        "extp               %[Temp4],    $ac2,            9             \n\t"

        : [tn1] "=&r"(tn1), [tp1] "=&r"(tp1), [n1] "=&r"(n1), [p3] "=&r"(p3),
          [n3] "=&r"(n3), [Temp1] "=&r"(Temp1), [Temp2] "=&r"(Temp2),
          [Temp3] "=&r"(Temp3), [Temp4] "=r"(Temp4), [p4] "+r"(p4)
        : [vector1b] "r"(vector1b), [vector2b] "r"(vector2b), [tp2] "r"(tp2),
          [p2] "r"(p2), [n2] "r"(n2), [n4] "r"(n4), [p1] "r"(p1),
          [src_ptr] "r"(src_ptr), [vector4a] "r"(vector4a),
          [vector3b] "r"(vector3b));

    /* clamp and store results */
    dst_ptr[8] = cm[Temp1];
    dst_ptr[9] = cm[Temp2];
    dst_ptr[10] = cm[Temp3];
    dst_ptr[11] = cm[Temp4];

    /* next 4 pixels */
    __asm__ __volatile__(
        /* even 7. pixel */
        "dpa.w.ph           $ac3,        %[p1],           %[vector1b]   \n\t"
        "dpa.w.ph           $ac3,        %[p3],           %[vector2b]   \n\t"
        "dpa.w.ph           $ac3,        %[p4],           %[vector3b]   \n\t"

        /* even 8. pixel */
        "mtlo               %[vector4a], $ac2                           \n\t"
        "preceu.ph.qbl      %[p2],       %[tp1]                         \n\t"
        "dpa.w.ph           $ac2,        %[p3],           %[vector1b]   \n\t"
        "dpa.w.ph           $ac2,        %[p4],           %[vector2b]   \n\t"
        "dpa.w.ph           $ac2,        %[p2],           %[vector3b]   \n\t"
        "ulw                %[tn1],      15(%[src_ptr])                 \n\t"
        "extp               %[Temp1],    $ac3,            9             \n\t"

        /* odd 7. pixel */
        "mtlo               %[vector4a], $ac3                           \n\t"
        "preceu.ph.qbr      %[n4],       %[tn1]                         \n\t"
        "dpa.w.ph           $ac3,        %[n1],           %[vector1b]   \n\t"
        "dpa.w.ph           $ac3,        %[n3],           %[vector2b]   \n\t"
        "dpa.w.ph           $ac3,        %[n4],           %[vector3b]   \n\t"
        "extp               %[Temp3],    $ac2,            9             \n\t"

        /* odd 8. pixel */
        "mtlo               %[vector4a], $ac2                           \n\t"
        "preceu.ph.qbl      %[n2],       %[tn1]                         \n\t"
        "dpa.w.ph           $ac2,        %[n3],           %[vector1b]   \n\t"
        "dpa.w.ph           $ac2,        %[n4],           %[vector2b]   \n\t"
        "dpa.w.ph           $ac2,        %[n2],           %[vector3b]   \n\t"
        "extp               %[Temp2],    $ac3,            9             \n\t"
        "extp               %[Temp4],    $ac2,            9             \n\t"

        /* clamp and store results */
        "lbux               %[tp1],      %[Temp1](%[cm])                \n\t"
        "lbux               %[tn1],      %[Temp2](%[cm])                \n\t"
        "lbux               %[p2],       %[Temp3](%[cm])                \n\t"
        "sb                 %[tp1],      12(%[dst_ptr])                 \n\t"
        "sb                 %[tn1],      13(%[dst_ptr])                 \n\t"
        "lbux               %[n2],       %[Temp4](%[cm])                \n\t"
        "sb                 %[p2],       14(%[dst_ptr])                 \n\t"
        "sb                 %[n2],       15(%[dst_ptr])                 \n\t"

        : [tn1] "=&r"(tn1), [p2] "=&r"(p2), [n2] "=&r"(n2), [n4] "=&r"(n4),
          [Temp1] "=&r"(Temp1), [Temp2] "=&r"(Temp2), [Temp3] "=&r"(Temp3),
          [Temp4] "=r"(Temp4), [tp1] "+r"(tp1)
        : [vector1b] "r"(vector1b), [vector2b] "r"(vector2b), [p4] "r"(p4),
          [n1] "r"(n1), [p1] "r"(p1), [vector4a] "r"(vector4a),
          [vector3b] "r"(vector3b), [p3] "r"(p3), [n3] "r"(n3),
          [src_ptr] "r"(src_ptr), [cm] "r"(cm), [dst_ptr] "r"(dst_ptr));

    src_ptr += src_pixels_per_line;
    dst_ptr += pitch;
  }
}

void vp8_filter_block2d_first_pass16_0(unsigned char *RESTRICT src_ptr,
                                       unsigned char *RESTRICT output_ptr,
                                       unsigned int src_pixels_per_line) {
  int Temp1, Temp2, Temp3, Temp4;
  int i;

  /* prefetch src_ptr data to cache memory */
  prefetch_store(output_ptr + 32);

  /* copy memory from src buffer to dst buffer */
  for (i = 0; i < 7; ++i) {
    __asm__ __volatile__(
        "ulw    %[Temp1],   0(%[src_ptr])                               \n\t"
        "ulw    %[Temp2],   4(%[src_ptr])                               \n\t"
        "ulw    %[Temp3],   8(%[src_ptr])                               \n\t"
        "ulw    %[Temp4],   12(%[src_ptr])                              \n\t"
        "sw     %[Temp1],   0(%[output_ptr])                            \n\t"
        "sw     %[Temp2],   4(%[output_ptr])                            \n\t"
        "sw     %[Temp3],   8(%[output_ptr])                            \n\t"
        "sw     %[Temp4],   12(%[output_ptr])                           \n\t"
        "addu   %[src_ptr], %[src_ptr],        %[src_pixels_per_line]   \n\t"

        : [Temp1] "=&r"(Temp1), [Temp2] "=&r"(Temp2), [Temp3] "=&r"(Temp3),
          [Temp4] "=&r"(Temp4), [src_ptr] "+r"(src_ptr)
        : [src_pixels_per_line] "r"(src_pixels_per_line), [output_ptr] "r"(
                                                              output_ptr));

    __asm__ __volatile__(
        "ulw    %[Temp1],   0(%[src_ptr])                               \n\t"
        "ulw    %[Temp2],   4(%[src_ptr])                               \n\t"
        "ulw    %[Temp3],   8(%[src_ptr])                               \n\t"
        "ulw    %[Temp4],   12(%[src_ptr])                              \n\t"
        "sw     %[Temp1],   16(%[output_ptr])                           \n\t"
        "sw     %[Temp2],   20(%[output_ptr])                           \n\t"
        "sw     %[Temp3],   24(%[output_ptr])                           \n\t"
        "sw     %[Temp4],   28(%[output_ptr])                           \n\t"
        "addu   %[src_ptr], %[src_ptr],        %[src_pixels_per_line]   \n\t"

        : [Temp1] "=&r"(Temp1), [Temp2] "=&r"(Temp2), [Temp3] "=&r"(Temp3),
          [Temp4] "=&r"(Temp4), [src_ptr] "+r"(src_ptr)
        : [src_pixels_per_line] "r"(src_pixels_per_line), [output_ptr] "r"(
                                                              output_ptr));

    __asm__ __volatile__(
        "ulw    %[Temp1],   0(%[src_ptr])                               \n\t"
        "ulw    %[Temp2],   4(%[src_ptr])                               \n\t"
        "ulw    %[Temp3],   8(%[src_ptr])                               \n\t"
        "ulw    %[Temp4],   12(%[src_ptr])                              \n\t"
        "sw     %[Temp1],   32(%[output_ptr])                           \n\t"
        "sw     %[Temp2],   36(%[output_ptr])                           \n\t"
        "sw     %[Temp3],   40(%[output_ptr])                           \n\t"
        "sw     %[Temp4],   44(%[output_ptr])                           \n\t"
        "addu   %[src_ptr], %[src_ptr],        %[src_pixels_per_line]   \n\t"

        : [Temp1] "=&r"(Temp1), [Temp2] "=&r"(Temp2), [Temp3] "=&r"(Temp3),
          [Temp4] "=&r"(Temp4), [src_ptr] "+r"(src_ptr)
        : [src_pixels_per_line] "r"(src_pixels_per_line), [output_ptr] "r"(
                                                              output_ptr));

    output_ptr += 48;
  }
}

void vp8_filter_block2d_first_pass16_4tap(
    unsigned char *RESTRICT src_ptr, unsigned char *RESTRICT output_ptr,
    unsigned int src_pixels_per_line, unsigned int output_width,
    unsigned int output_height, int xoffset, int yoffset,
    unsigned char *RESTRICT dst_ptr, int pitch) {
  unsigned int i, j;
  int Temp1, Temp2, Temp3, Temp4;

  unsigned int vector4a;
  int vector1b, vector2b;
  unsigned int tp1, tp2, tp3, tn1;
  unsigned int p1, p2, p3;
  unsigned int n1, n2, n3;
  unsigned char *cm = ff_cropTbl + CROP_WIDTH;

  vector4a = 64;

  vector1b = sub_pel_filters_inv_tap_4[xoffset][0];
  vector2b = sub_pel_filters_inv_tap_4[xoffset][1];

  /* if (yoffset == 0) don't need temp buffer, data will be stored in dst_ptr */
  if (yoffset == 0) {
    output_height -= 5;
    src_ptr += (src_pixels_per_line + src_pixels_per_line);

    for (i = output_height; i--;) {
      __asm__ __volatile__("ulw     %[tp3],   -1(%[src_ptr])               \n\t"
                           : [tp3] "=&r"(tp3)
                           : [src_ptr] "r"(src_ptr));

      /* processing 4 adjacent pixels */
      for (j = 0; j < 16; j += 4) {
        /* apply filter with vectors pairs */
        __asm__ __volatile__(
            "ulw              %[tp2],      3(%[src_ptr])                    "
            "\n\t"
            "move             %[tp1],      %[tp3]                           "
            "\n\t"

            /* even 1. pixel */
            "mtlo             %[vector4a], $ac3                             "
            "\n\t"
            "mthi             $0,          $ac3                             "
            "\n\t"
            "move             %[tp3],      %[tp2]                           "
            "\n\t"
            "preceu.ph.qbr    %[p1],       %[tp1]                           "
            "\n\t"
            "preceu.ph.qbl    %[p2],       %[tp1]                           "
            "\n\t"
            "preceu.ph.qbr    %[p3],       %[tp2]                           "
            "\n\t"
            "dpa.w.ph         $ac3,        %[p1],           %[vector1b]     "
            "\n\t"
            "dpa.w.ph         $ac3,        %[p2],           %[vector2b]     "
            "\n\t"

            /* even 2. pixel */
            "mtlo             %[vector4a], $ac2                             "
            "\n\t"
            "mthi             $0,          $ac2                             "
            "\n\t"
            "dpa.w.ph         $ac2,        %[p2],           %[vector1b]     "
            "\n\t"
            "dpa.w.ph         $ac2,        %[p3],           %[vector2b]     "
            "\n\t"
            "extr.w           %[Temp1],    $ac3,            7               "
            "\n\t"

            /* odd 1. pixel */
            "ulw              %[tn1],      4(%[src_ptr])                    "
            "\n\t"
            "balign           %[tp2],      %[tp1],          3               "
            "\n\t"
            "mtlo             %[vector4a], $ac3                             "
            "\n\t"
            "mthi             $0,          $ac3                             "
            "\n\t"
            "preceu.ph.qbr    %[n1],       %[tp2]                           "
            "\n\t"
            "preceu.ph.qbl    %[n2],       %[tp2]                           "
            "\n\t"
            "preceu.ph.qbr    %[n3],       %[tn1]                           "
            "\n\t"
            "extr.w           %[Temp3],    $ac2,            7               "
            "\n\t"
            "dpa.w.ph         $ac3,        %[n1],           %[vector1b]     "
            "\n\t"
            "dpa.w.ph         $ac3,        %[n2],           %[vector2b]     "
            "\n\t"

            /* odd 2. pixel */
            "mtlo             %[vector4a], $ac2                             "
            "\n\t"
            "mthi             $0,          $ac2                             "
            "\n\t"
            "extr.w           %[Temp2],    $ac3,            7               "
            "\n\t"
            "dpa.w.ph         $ac2,        %[n2],           %[vector1b]     "
            "\n\t"
            "dpa.w.ph         $ac2,        %[n3],           %[vector2b]     "
            "\n\t"
            "extr.w           %[Temp4],    $ac2,            7               "
            "\n\t"

            /* clamp and store results */
            "lbux             %[tp1],      %[Temp1](%[cm])                  "
            "\n\t"
            "lbux             %[tn1],      %[Temp2](%[cm])                  "
            "\n\t"
            "lbux             %[tp2],      %[Temp3](%[cm])                  "
            "\n\t"
            "sb               %[tp1],      0(%[dst_ptr])                    "
            "\n\t"
            "sb               %[tn1],      1(%[dst_ptr])                    "
            "\n\t"
            "lbux             %[n2],       %[Temp4](%[cm])                  "
            "\n\t"
            "sb               %[tp2],      2(%[dst_ptr])                    "
            "\n\t"
            "sb               %[n2],       3(%[dst_ptr])                    "
            "\n\t"

            : [tp1] "=&r"(tp1), [tp2] "=&r"(tp2), [tp3] "=&r"(tp3),
              [tn1] "=&r"(tn1), [p1] "=&r"(p1), [p2] "=&r"(p2), [n1] "=&r"(n1),
              [n2] "=&r"(n2), [n3] "=&r"(n3), [Temp1] "=&r"(Temp1),
              [Temp2] "=&r"(Temp2), [p3] "=&r"(p3), [Temp3] "=&r"(Temp3),
              [Temp4] "=&r"(Temp4)
            : [vector1b] "r"(vector1b), [vector2b] "r"(vector2b),
              [vector4a] "r"(vector4a), [cm] "r"(cm), [dst_ptr] "r"(dst_ptr),
              [src_ptr] "r"(src_ptr));

        src_ptr += 4;
      }

      /* Next row... */
      src_ptr += src_pixels_per_line - 16;
      dst_ptr += pitch;
    }
  } else {
    for (i = output_height; i--;) {
      /* processing 4 adjacent pixels */
      for (j = 0; j < 16; j += 4) {
        /* apply filter with vectors pairs */
        __asm__ __volatile__(
            "ulw              %[tp1],      -1(%[src_ptr])                   "
            "\n\t"
            "ulw              %[tp2],      3(%[src_ptr])                    "
            "\n\t"

            /* even 1. pixel */
            "mtlo             %[vector4a], $ac3                             "
            "\n\t"
            "mthi             $0,          $ac3                             "
            "\n\t"
            "preceu.ph.qbr    %[p1],       %[tp1]                           "
            "\n\t"
            "preceu.ph.qbl    %[p2],       %[tp1]                           "
            "\n\t"
            "preceu.ph.qbr    %[p3],       %[tp2]                           "
            "\n\t"
            "dpa.w.ph         $ac3,        %[p1],           %[vector1b]     "
            "\n\t"
            "dpa.w.ph         $ac3,        %[p2],           %[vector2b]     "
            "\n\t"

            /* even 2. pixel */
            "mtlo             %[vector4a], $ac2                             "
            "\n\t"
            "mthi             $0,          $ac2                             "
            "\n\t"
            "dpa.w.ph         $ac2,        %[p2],           %[vector1b]     "
            "\n\t"
            "dpa.w.ph         $ac2,        %[p3],           %[vector2b]     "
            "\n\t"
            "extr.w           %[Temp1],    $ac3,            7               "
            "\n\t"

            /* odd 1. pixel */
            "ulw              %[tn1],      4(%[src_ptr])                    "
            "\n\t"
            "balign           %[tp2],      %[tp1],          3               "
            "\n\t"
            "mtlo             %[vector4a], $ac3                             "
            "\n\t"
            "mthi             $0,          $ac3                             "
            "\n\t"
            "preceu.ph.qbr    %[n1],       %[tp2]                           "
            "\n\t"
            "preceu.ph.qbl    %[n2],       %[tp2]                           "
            "\n\t"
            "preceu.ph.qbr    %[n3],       %[tn1]                           "
            "\n\t"
            "extr.w           %[Temp3],    $ac2,            7               "
            "\n\t"
            "dpa.w.ph         $ac3,        %[n1],           %[vector1b]     "
            "\n\t"
            "dpa.w.ph         $ac3,        %[n2],           %[vector2b]     "
            "\n\t"

            /* odd 2. pixel */
            "mtlo             %[vector4a], $ac2                             "
            "\n\t"
            "mthi             $0,          $ac2                             "
            "\n\t"
            "extr.w           %[Temp2],    $ac3,            7               "
            "\n\t"
            "dpa.w.ph         $ac2,        %[n2],           %[vector1b]     "
            "\n\t"
            "dpa.w.ph         $ac2,        %[n3],           %[vector2b]     "
            "\n\t"
            "extr.w           %[Temp4],    $ac2,            7               "
            "\n\t"

            /* clamp and store results */
            "lbux             %[tp1],      %[Temp1](%[cm])                  "
            "\n\t"
            "lbux             %[tn1],      %[Temp2](%[cm])                  "
            "\n\t"
            "lbux             %[tp2],      %[Temp3](%[cm])                  "
            "\n\t"
            "sb               %[tp1],      0(%[output_ptr])                 "
            "\n\t"
            "sb               %[tn1],      1(%[output_ptr])                 "
            "\n\t"
            "lbux             %[n2],       %[Temp4](%[cm])                  "
            "\n\t"
            "sb               %[tp2],      2(%[output_ptr])                 "
            "\n\t"
            "sb               %[n2],       3(%[output_ptr])                 "
            "\n\t"

            : [tp1] "=&r"(tp1), [tp2] "=&r"(tp2), [tn1] "=&r"(tn1),
              [p1] "=&r"(p1), [p2] "=&r"(p2), [p3] "=&r"(p3), [n1] "=&r"(n1),
              [n2] "=&r"(n2), [n3] "=&r"(n3), [Temp1] "=&r"(Temp1),
              [Temp2] "=&r"(Temp2), [Temp3] "=&r"(Temp3), [Temp4] "=&r"(Temp4)
            : [vector1b] "r"(vector1b), [vector2b] "r"(vector2b),
              [vector4a] "r"(vector4a), [cm] "r"(cm),
              [output_ptr] "r"(output_ptr), [src_ptr] "r"(src_ptr));

        src_ptr += 4;
      }

      /* next row... */
      src_ptr += src_pixels_per_line;
      output_ptr += output_width;
    }
  }
}

void vp8_filter_block2d_second_pass4(unsigned char *RESTRICT src_ptr,
                                     unsigned char *RESTRICT output_ptr,
                                     int output_pitch, int yoffset) {
  unsigned int i;

  int Temp1, Temp2, Temp3, Temp4;
  unsigned int vector1b, vector2b, vector3b, vector4a;

  unsigned char src_ptr_l2;
  unsigned char src_ptr_l1;
  unsigned char src_ptr_0;
  unsigned char src_ptr_r1;
  unsigned char src_ptr_r2;
  unsigned char src_ptr_r3;

  unsigned char *cm = ff_cropTbl + CROP_WIDTH;

  vector4a = 64;

  /* load filter coefficients */
  vector1b = sub_pel_filterss[yoffset][0];
  vector2b = sub_pel_filterss[yoffset][2];
  vector3b = sub_pel_filterss[yoffset][1];

  if (vector1b) {
    /* 6 tap filter */

    for (i = 2; i--;) {
      /* prefetch src_ptr data to cache memory */
      prefetch_load(src_ptr);

      /* do not allow compiler to reorder instructions */
      __asm__ __volatile__(
          ".set noreorder                                                 \n\t"
          :
          :);

      /* apply filter with vectors pairs */
      __asm__ __volatile__(
          "lbu            %[src_ptr_l2],  -8(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_l1],  -4(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_0],   0(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r1],  4(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r2],  8(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r3],  12(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac2                            \n\t"

          "append         %[src_ptr_l2],  %[src_ptr_r3],  8               \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac2,           %[src_ptr_l2],  %[vector1b]     \n\t"
          "dpau.h.qbr     $ac2,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac2,           %[src_ptr_l1],  %[vector3b]     \n\t"

          "lbu            %[src_ptr_l2],  -7(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_l1],  -3(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_0],   1(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r1],  5(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r2],  9(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r3],  13(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac3                            \n\t"
          "extp           %[Temp1],       $ac2,           9               \n\t"

          "append         %[src_ptr_l2],  %[src_ptr_r3],  8               \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac3,           %[src_ptr_l2],  %[vector1b]     \n\t"
          "dpau.h.qbr     $ac3,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac3,           %[src_ptr_l1],  %[vector3b]     \n\t"

          "lbu            %[src_ptr_l2],  -6(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_l1],  -2(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_0],   2(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r1],  6(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r2],  10(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r3],  14(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac0                            \n\t"
          "extp           %[Temp2],       $ac3,           9               \n\t"

          "append         %[src_ptr_l2],  %[src_ptr_r3],  8               \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac0,           %[src_ptr_l2],  %[vector1b]     \n\t"
          "dpau.h.qbr     $ac0,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac0,           %[src_ptr_l1],  %[vector3b]     \n\t"

          "lbu            %[src_ptr_l2],  -5(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_l1],  -1(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_0],   3(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r1],  7(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r2],  11(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r3],  15(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac1                            \n\t"
          "extp           %[Temp3],       $ac0,           9               \n\t"

          "append         %[src_ptr_l2],  %[src_ptr_r3],  8               \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac1,           %[src_ptr_l2],  %[vector1b]     \n\t"
          "dpau.h.qbr     $ac1,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac1,           %[src_ptr_l1],  %[vector3b]     \n\t"
          "extp           %[Temp4],       $ac1,           9               \n\t"

          : [Temp1] "=&r"(Temp1), [Temp2] "=&r"(Temp2), [Temp3] "=&r"(Temp3),
            [Temp4] "=r"(Temp4), [src_ptr_l1] "=&r"(src_ptr_l1),
            [src_ptr_0] "=&r"(src_ptr_0), [src_ptr_r1] "=&r"(src_ptr_r1),
            [src_ptr_r2] "=&r"(src_ptr_r2), [src_ptr_l2] "=&r"(src_ptr_l2),
            [src_ptr_r3] "=&r"(src_ptr_r3)
          : [vector1b] "r"(vector1b), [vector2b] "r"(vector2b),
            [vector3b] "r"(vector3b), [vector4a] "r"(vector4a),
            [src_ptr] "r"(src_ptr));

      /* clamp and store results */
      output_ptr[0] = cm[Temp1];
      output_ptr[1] = cm[Temp2];
      output_ptr[2] = cm[Temp3];
      output_ptr[3] = cm[Temp4];

      output_ptr += output_pitch;

      /* apply filter with vectors pairs */
      __asm__ __volatile__(
          "lbu            %[src_ptr_l2],  -4(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_l1],  0(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_0],   4(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r1],  8(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r2],  12(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r3],  16(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac2                            \n\t"
          "append         %[src_ptr_l2],  %[src_ptr_r3],  8               \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac2,           %[src_ptr_l2],  %[vector1b]     \n\t"
          "dpau.h.qbr     $ac2,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac2,           %[src_ptr_l1],  %[vector3b]     \n\t"

          "lbu            %[src_ptr_l2],  -3(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_l1],  1(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_0],   5(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r1],  9(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r2],  13(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r3],  17(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac3                            \n\t"
          "extp           %[Temp1],       $ac2,           9               \n\t"

          "append         %[src_ptr_l2],  %[src_ptr_r3],  8               \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac3,           %[src_ptr_l2],  %[vector1b]     \n\t"
          "dpau.h.qbr     $ac3,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac3,           %[src_ptr_l1],  %[vector3b]     \n\t"

          "lbu            %[src_ptr_l2],  -2(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_l1],  2(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_0],   6(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r1],  10(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r2],  14(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r3],  18(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac0                            \n\t"
          "extp           %[Temp2],       $ac3,           9               \n\t"

          "append         %[src_ptr_l2],  %[src_ptr_r3],  8               \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac0,           %[src_ptr_l2],  %[vector1b]     \n\t"
          "dpau.h.qbr     $ac0,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac0,           %[src_ptr_l1],  %[vector3b]     \n\t"

          "lbu            %[src_ptr_l2],  -1(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_l1],  3(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_0],   7(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r1],  11(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r2],  15(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r3],  19(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac1                            \n\t"
          "extp           %[Temp3],       $ac0,           9               \n\t"

          "append         %[src_ptr_l2],  %[src_ptr_r3],  8               \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac1,           %[src_ptr_l2],  %[vector1b]     \n\t"
          "dpau.h.qbr     $ac1,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac1,           %[src_ptr_l1],  %[vector3b]     \n\t"
          "extp           %[Temp4],       $ac1,           9               \n\t"

          : [Temp1] "=&r"(Temp1), [Temp2] "=&r"(Temp2), [Temp3] "=&r"(Temp3),
            [Temp4] "=r"(Temp4), [src_ptr_l1] "=&r"(src_ptr_l1),
            [src_ptr_0] "=&r"(src_ptr_0), [src_ptr_r1] "=&r"(src_ptr_r1),
            [src_ptr_r2] "=&r"(src_ptr_r2), [src_ptr_l2] "=&r"(src_ptr_l2),
            [src_ptr_r3] "=&r"(src_ptr_r3)
          : [vector1b] "r"(vector1b), [vector2b] "r"(vector2b),
            [vector3b] "r"(vector3b), [vector4a] "r"(vector4a),
            [src_ptr] "r"(src_ptr));

      /* clamp and store results */
      output_ptr[0] = cm[Temp1];
      output_ptr[1] = cm[Temp2];
      output_ptr[2] = cm[Temp3];
      output_ptr[3] = cm[Temp4];

      src_ptr += 8;
      output_ptr += output_pitch;
    }
  } else {
    /* 4 tap filter */

    /* prefetch src_ptr data to cache memory */
    prefetch_load(src_ptr);

    for (i = 2; i--;) {
      /* do not allow compiler to reorder instructions */
      __asm__ __volatile__(
          ".set noreorder                                                 \n\t"
          :
          :);

      /* apply filter with vectors pairs */
      __asm__ __volatile__(
          "lbu            %[src_ptr_l1],  -4(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_0],   0(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r1],  4(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r2],  8(%[src_ptr])                   \n\t"
          "mtlo           %[vector4a],    $ac2                            \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac2,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac2,           %[src_ptr_l1],  %[vector3b]     \n\t"

          "lbu            %[src_ptr_l1],  -3(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_0],   1(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r1],  5(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r2],  9(%[src_ptr])                   \n\t"
          "mtlo           %[vector4a],    $ac3                            \n\t"
          "extp           %[Temp1],       $ac2,           9               \n\t"

          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac3,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac3,           %[src_ptr_l1],  %[vector3b]     \n\t"

          "lbu            %[src_ptr_l1],  -2(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_0],   2(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r1],  6(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r2],  10(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac0                            \n\t"
          "extp           %[Temp2],       $ac3,           9               \n\t"

          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac0,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac0,           %[src_ptr_l1],  %[vector3b]     \n\t"

          "lbu            %[src_ptr_l1],  -1(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_0],   3(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r1],  7(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r2],  11(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac1                            \n\t"
          "extp           %[Temp3],       $ac0,           9               \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac1,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac1,           %[src_ptr_l1],  %[vector3b]     \n\t"
          "extp           %[Temp4],       $ac1,           9               \n\t"

          : [Temp1] "=&r"(Temp1), [Temp2] "=&r"(Temp2), [Temp3] "=&r"(Temp3),
            [Temp4] "=r"(Temp4), [src_ptr_l1] "=&r"(src_ptr_l1),
            [src_ptr_0] "=&r"(src_ptr_0), [src_ptr_r1] "=&r"(src_ptr_r1),
            [src_ptr_r2] "=&r"(src_ptr_r2)
          : [vector2b] "r"(vector2b), [vector3b] "r"(vector3b),
            [vector4a] "r"(vector4a), [src_ptr] "r"(src_ptr));

      /* clamp and store results */
      output_ptr[0] = cm[Temp1];
      output_ptr[1] = cm[Temp2];
      output_ptr[2] = cm[Temp3];
      output_ptr[3] = cm[Temp4];

      output_ptr += output_pitch;

      /* apply filter with vectors pairs */
      __asm__ __volatile__(
          "lbu            %[src_ptr_l1],  0(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_0],   4(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r1],  8(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r2],  12(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac2                            \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac2,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac2,           %[src_ptr_l1],  %[vector3b]     \n\t"

          "lbu            %[src_ptr_l1],  1(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_0],   5(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r1],  9(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r2],  13(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac3                            \n\t"
          "extp           %[Temp1],       $ac2,           9               \n\t"

          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac3,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac3,           %[src_ptr_l1],  %[vector3b]     \n\t"

          "lbu            %[src_ptr_l1],  2(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_0],   6(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r1],  10(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r2],  14(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac0                            \n\t"
          "extp           %[Temp2],       $ac3,           9               \n\t"

          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac0,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac0,           %[src_ptr_l1],  %[vector3b]     \n\t"

          "lbu            %[src_ptr_l1],  3(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_0],   7(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r1],  11(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r2],  15(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac1                            \n\t"
          "extp           %[Temp3],       $ac0,           9               \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac1,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac1,           %[src_ptr_l1],  %[vector3b]     \n\t"
          "extp           %[Temp4],       $ac1,           9               \n\t"

          : [Temp1] "=&r"(Temp1), [Temp2] "=&r"(Temp2), [Temp3] "=&r"(Temp3),
            [Temp4] "=r"(Temp4), [src_ptr_l1] "=&r"(src_ptr_l1),
            [src_ptr_0] "=&r"(src_ptr_0), [src_ptr_r1] "=&r"(src_ptr_r1),
            [src_ptr_r2] "=&r"(src_ptr_r2)
          : [vector2b] "r"(vector2b), [vector3b] "r"(vector3b),
            [vector4a] "r"(vector4a), [src_ptr] "r"(src_ptr));

      /* clamp and store results */
      output_ptr[0] = cm[Temp1];
      output_ptr[1] = cm[Temp2];
      output_ptr[2] = cm[Temp3];
      output_ptr[3] = cm[Temp4];

      src_ptr += 8;
      output_ptr += output_pitch;
    }
  }
}

void vp8_filter_block2d_second_pass_8(unsigned char *RESTRICT src_ptr,
                                      unsigned char *RESTRICT output_ptr,
                                      int output_pitch,
                                      unsigned int output_height,
                                      unsigned int output_width,
                                      unsigned int yoffset) {
  unsigned int i;

  int Temp1, Temp2, Temp3, Temp4, Temp5, Temp6, Temp7, Temp8;
  unsigned int vector1b, vector2b, vector3b, vector4a;

  unsigned char src_ptr_l2;
  unsigned char src_ptr_l1;
  unsigned char src_ptr_0;
  unsigned char src_ptr_r1;
  unsigned char src_ptr_r2;
  unsigned char src_ptr_r3;
  unsigned char *cm = ff_cropTbl + CROP_WIDTH;
  (void)output_width;

  vector4a = 64;

  vector1b = sub_pel_filterss[yoffset][0];
  vector2b = sub_pel_filterss[yoffset][2];
  vector3b = sub_pel_filterss[yoffset][1];

  if (vector1b) {
    /* 6 tap filter */

    /* prefetch src_ptr data to cache memory */
    prefetch_load(src_ptr);

    for (i = output_height; i--;) {
      /* apply filter with vectors pairs */
      __asm__ __volatile__(
          "lbu            %[src_ptr_l2],  -16(%[src_ptr])                 \n\t"
          "lbu            %[src_ptr_l1],  -8(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_0],   0(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r1],  8(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r2],  16(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r3],  24(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac2                            \n\t"

          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "append         %[src_ptr_l2],  %[src_ptr_r3],  8               \n\t"
          "dpau.h.qbr     $ac2,           %[src_ptr_l2],  %[vector1b]     \n\t"
          "dpau.h.qbr     $ac2,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac2,           %[src_ptr_l1],  %[vector3b]     \n\t"

          "lbu            %[src_ptr_l2],  -15(%[src_ptr])                 \n\t"
          "lbu            %[src_ptr_l1],  -7(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_0],   1(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r1],  9(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r2],  17(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r3],  25(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac3                            \n\t"
          "extp           %[Temp1],       $ac2,           9               \n\t"

          "append         %[src_ptr_l2],  %[src_ptr_r3],  8               \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac3,           %[src_ptr_l2],  %[vector1b]     \n\t"
          "dpau.h.qbr     $ac3,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac3,           %[src_ptr_l1],  %[vector3b]     \n\t"

          "lbu            %[src_ptr_l2],  -14(%[src_ptr])                 \n\t"
          "lbu            %[src_ptr_l1],  -6(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_0],   2(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r1],  10(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r2],  18(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r3],  26(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac0                            \n\t"
          "extp           %[Temp2],       $ac3,           9               \n\t"

          "append         %[src_ptr_l2],  %[src_ptr_r3],  8               \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac0,           %[src_ptr_l2],  %[vector1b]     \n\t"
          "dpau.h.qbr     $ac0,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac0,           %[src_ptr_l1],  %[vector3b]     \n\t"

          "lbu            %[src_ptr_l2],  -13(%[src_ptr])                 \n\t"
          "lbu            %[src_ptr_l1],  -5(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_0],   3(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r1],  11(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r2],  19(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r3],  27(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac1                            \n\t"
          "extp           %[Temp3],       $ac0,           9               \n\t"

          "append         %[src_ptr_l2],  %[src_ptr_r3],  8               \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac1,           %[src_ptr_l2],  %[vector1b]     \n\t"
          "dpau.h.qbr     $ac1,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac1,           %[src_ptr_l1],  %[vector3b]     \n\t"

          : [Temp1] "=&r"(Temp1), [Temp2] "=&r"(Temp2), [Temp3] "=&r"(Temp3),
            [src_ptr_l1] "=&r"(src_ptr_l1), [src_ptr_0] "=&r"(src_ptr_0),
            [src_ptr_r1] "=&r"(src_ptr_r1), [src_ptr_r2] "=&r"(src_ptr_r2),
            [src_ptr_l2] "=&r"(src_ptr_l2), [src_ptr_r3] "=&r"(src_ptr_r3)
          : [vector1b] "r"(vector1b), [vector2b] "r"(vector2b),
            [vector3b] "r"(vector3b), [vector4a] "r"(vector4a),
            [src_ptr] "r"(src_ptr));

      /* apply filter with vectors pairs */
      __asm__ __volatile__(
          "lbu            %[src_ptr_l2],  -12(%[src_ptr])                 \n\t"
          "lbu            %[src_ptr_l1],  -4(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_0],   4(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r1],  12(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r2],  20(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r3],  28(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac2                            \n\t"

          "append         %[src_ptr_l2],  %[src_ptr_r3],  8               \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac2,           %[src_ptr_l2],  %[vector1b]     \n\t"
          "dpau.h.qbr     $ac2,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac2,           %[src_ptr_l1],  %[vector3b]     \n\t"
          "extp           %[Temp4],       $ac1,           9               \n\t"

          "lbu            %[src_ptr_l2],  -11(%[src_ptr])                 \n\t"
          "lbu            %[src_ptr_l1],  -3(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_0],   5(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r1],  13(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r2],  21(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r3],  29(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac3                            \n\t"
          "extp           %[Temp5],       $ac2,           9               \n\t"

          "append         %[src_ptr_l2],  %[src_ptr_r3],  8               \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac3,           %[src_ptr_l2],  %[vector1b]     \n\t"
          "dpau.h.qbr     $ac3,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac3,           %[src_ptr_l1],  %[vector3b]     \n\t"

          "lbu            %[src_ptr_l2],  -10(%[src_ptr])                 \n\t"
          "lbu            %[src_ptr_l1],  -2(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_0],   6(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r1],  14(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r2],  22(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r3],  30(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac0                            \n\t"
          "extp           %[Temp6],       $ac3,           9               \n\t"

          "append         %[src_ptr_l2],  %[src_ptr_r3],  8               \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac0,           %[src_ptr_l2],  %[vector1b]     \n\t"
          "dpau.h.qbr     $ac0,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac0,           %[src_ptr_l1],  %[vector3b]     \n\t"

          "lbu            %[src_ptr_l2],  -9(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_l1],  -1(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_0],   7(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r1],  15(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r2],  23(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r3],  31(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac1                            \n\t"
          "extp           %[Temp7],       $ac0,           9               \n\t"

          "append         %[src_ptr_l2],  %[src_ptr_r3],  8               \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac1,           %[src_ptr_l2],  %[vector1b]     \n\t"
          "dpau.h.qbr     $ac1,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac1,           %[src_ptr_l1],  %[vector3b]     \n\t"
          "extp           %[Temp8],       $ac1,           9               \n\t"

          : [Temp4] "=&r"(Temp4), [Temp5] "=&r"(Temp5), [Temp6] "=&r"(Temp6),
            [Temp7] "=&r"(Temp7), [Temp8] "=r"(Temp8),
            [src_ptr_l1] "=&r"(src_ptr_l1), [src_ptr_0] "=&r"(src_ptr_0),
            [src_ptr_r1] "=&r"(src_ptr_r1), [src_ptr_r2] "=&r"(src_ptr_r2),
            [src_ptr_l2] "=&r"(src_ptr_l2), [src_ptr_r3] "=&r"(src_ptr_r3)
          : [vector1b] "r"(vector1b), [vector2b] "r"(vector2b),
            [vector3b] "r"(vector3b), [vector4a] "r"(vector4a),
            [src_ptr] "r"(src_ptr));

      /* clamp and store results */
      output_ptr[0] = cm[Temp1];
      output_ptr[1] = cm[Temp2];
      output_ptr[2] = cm[Temp3];
      output_ptr[3] = cm[Temp4];
      output_ptr[4] = cm[Temp5];
      output_ptr[5] = cm[Temp6];
      output_ptr[6] = cm[Temp7];
      output_ptr[7] = cm[Temp8];

      src_ptr += 8;
      output_ptr += output_pitch;
    }
  } else {
    /* 4 tap filter */

    /* prefetch src_ptr data to cache memory */
    prefetch_load(src_ptr);

    for (i = output_height; i--;) {
      __asm__ __volatile__(
          "lbu            %[src_ptr_l1],  -8(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_0],   0(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r1],  8(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r2],  16(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac2                            \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac2,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac2,           %[src_ptr_l1],  %[vector3b]     \n\t"

          : [src_ptr_l1] "=&r"(src_ptr_l1), [src_ptr_0] "=&r"(src_ptr_0),
            [src_ptr_r1] "=&r"(src_ptr_r1), [src_ptr_r2] "=&r"(src_ptr_r2)
          : [vector2b] "r"(vector2b), [vector3b] "r"(vector3b),
            [vector4a] "r"(vector4a), [src_ptr] "r"(src_ptr));

      __asm__ __volatile__(
          "lbu            %[src_ptr_l1],  -7(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_0],   1(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r1],  9(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r2],  17(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac3                            \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac3,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac3,           %[src_ptr_l1],  %[vector3b]     \n\t"
          "extp           %[Temp1],       $ac2,           9               \n\t"

          : [Temp1] "=r"(Temp1), [src_ptr_l1] "=&r"(src_ptr_l1),
            [src_ptr_0] "=&r"(src_ptr_0), [src_ptr_r1] "=&r"(src_ptr_r1),
            [src_ptr_r2] "=&r"(src_ptr_r2)
          : [vector2b] "r"(vector2b), [vector3b] "r"(vector3b),
            [vector4a] "r"(vector4a), [src_ptr] "r"(src_ptr));

      src_ptr_l1 = src_ptr[-6];
      src_ptr_0 = src_ptr[2];
      src_ptr_r1 = src_ptr[10];
      src_ptr_r2 = src_ptr[18];

      __asm__ __volatile__(
          "mtlo           %[vector4a],    $ac0                            \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac0,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac0,           %[src_ptr_l1],  %[vector3b]     \n\t"
          "extp           %[Temp2],       $ac3,           9               \n\t"

          : [Temp2] "=r"(Temp2)
          : [vector2b] "r"(vector2b), [vector3b] "r"(vector3b),
            [src_ptr_l1] "r"(src_ptr_l1), [src_ptr_0] "r"(src_ptr_0),
            [src_ptr_r1] "r"(src_ptr_r1), [src_ptr_r2] "r"(src_ptr_r2),
            [vector4a] "r"(vector4a));

      src_ptr_l1 = src_ptr[-5];
      src_ptr_0 = src_ptr[3];
      src_ptr_r1 = src_ptr[11];
      src_ptr_r2 = src_ptr[19];

      __asm__ __volatile__(
          "mtlo           %[vector4a],    $ac1                            \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac1,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac1,           %[src_ptr_l1],  %[vector3b]     \n\t"
          "extp           %[Temp3],       $ac0,           9               \n\t"

          : [Temp3] "=r"(Temp3)
          : [vector2b] "r"(vector2b), [vector3b] "r"(vector3b),
            [src_ptr_l1] "r"(src_ptr_l1), [src_ptr_0] "r"(src_ptr_0),
            [src_ptr_r1] "r"(src_ptr_r1), [src_ptr_r2] "r"(src_ptr_r2),
            [vector4a] "r"(vector4a));

      src_ptr_l1 = src_ptr[-4];
      src_ptr_0 = src_ptr[4];
      src_ptr_r1 = src_ptr[12];
      src_ptr_r2 = src_ptr[20];

      __asm__ __volatile__(
          "mtlo           %[vector4a],    $ac2                            \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac2,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac2,           %[src_ptr_l1],  %[vector3b]     \n\t"
          "extp           %[Temp4],       $ac1,           9               \n\t"

          : [Temp4] "=r"(Temp4)
          : [vector2b] "r"(vector2b), [vector3b] "r"(vector3b),
            [src_ptr_l1] "r"(src_ptr_l1), [src_ptr_0] "r"(src_ptr_0),
            [src_ptr_r1] "r"(src_ptr_r1), [src_ptr_r2] "r"(src_ptr_r2),
            [vector4a] "r"(vector4a));

      src_ptr_l1 = src_ptr[-3];
      src_ptr_0 = src_ptr[5];
      src_ptr_r1 = src_ptr[13];
      src_ptr_r2 = src_ptr[21];

      __asm__ __volatile__(
          "mtlo           %[vector4a],    $ac3                            \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac3,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac3,           %[src_ptr_l1],  %[vector3b]     \n\t"
          "extp           %[Temp5],       $ac2,           9               \n\t"

          : [Temp5] "=&r"(Temp5)
          : [vector2b] "r"(vector2b), [vector3b] "r"(vector3b),
            [src_ptr_l1] "r"(src_ptr_l1), [src_ptr_0] "r"(src_ptr_0),
            [src_ptr_r1] "r"(src_ptr_r1), [src_ptr_r2] "r"(src_ptr_r2),
            [vector4a] "r"(vector4a));

      src_ptr_l1 = src_ptr[-2];
      src_ptr_0 = src_ptr[6];
      src_ptr_r1 = src_ptr[14];
      src_ptr_r2 = src_ptr[22];

      __asm__ __volatile__(
          "mtlo           %[vector4a],    $ac0                            \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac0,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac0,           %[src_ptr_l1],  %[vector3b]     \n\t"
          "extp           %[Temp6],       $ac3,           9               \n\t"

          : [Temp6] "=r"(Temp6)
          : [vector2b] "r"(vector2b), [vector3b] "r"(vector3b),
            [src_ptr_l1] "r"(src_ptr_l1), [src_ptr_0] "r"(src_ptr_0),
            [src_ptr_r1] "r"(src_ptr_r1), [src_ptr_r2] "r"(src_ptr_r2),
            [vector4a] "r"(vector4a));

      src_ptr_l1 = src_ptr[-1];
      src_ptr_0 = src_ptr[7];
      src_ptr_r1 = src_ptr[15];
      src_ptr_r2 = src_ptr[23];

      __asm__ __volatile__(
          "mtlo           %[vector4a],    $ac1                            \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac1,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac1,           %[src_ptr_l1],  %[vector3b]     \n\t"
          "extp           %[Temp7],       $ac0,           9               \n\t"
          "extp           %[Temp8],       $ac1,           9               \n\t"

          : [Temp7] "=&r"(Temp7), [Temp8] "=r"(Temp8)
          : [vector2b] "r"(vector2b), [vector3b] "r"(vector3b),
            [src_ptr_l1] "r"(src_ptr_l1), [src_ptr_0] "r"(src_ptr_0),
            [src_ptr_r1] "r"(src_ptr_r1), [src_ptr_r2] "r"(src_ptr_r2),
            [vector4a] "r"(vector4a));

      /* clamp and store results */
      output_ptr[0] = cm[Temp1];
      output_ptr[1] = cm[Temp2];
      output_ptr[2] = cm[Temp3];
      output_ptr[3] = cm[Temp4];
      output_ptr[4] = cm[Temp5];
      output_ptr[5] = cm[Temp6];
      output_ptr[6] = cm[Temp7];
      output_ptr[7] = cm[Temp8];

      src_ptr += 8;
      output_ptr += output_pitch;
    }
  }
}

void vp8_filter_block2d_second_pass161(unsigned char *RESTRICT src_ptr,
                                       unsigned char *RESTRICT output_ptr,
                                       int output_pitch,
                                       const unsigned short *vp8_filter) {
  unsigned int i, j;

  int Temp1, Temp2, Temp3, Temp4, Temp5, Temp6, Temp7, Temp8;
  unsigned int vector4a;
  unsigned int vector1b, vector2b, vector3b;

  unsigned char src_ptr_l2;
  unsigned char src_ptr_l1;
  unsigned char src_ptr_0;
  unsigned char src_ptr_r1;
  unsigned char src_ptr_r2;
  unsigned char src_ptr_r3;
  unsigned char *cm = ff_cropTbl + CROP_WIDTH;

  vector4a = 64;

  vector1b = vp8_filter[0];
  vector2b = vp8_filter[2];
  vector3b = vp8_filter[1];

  if (vector1b == 0) {
    /* 4 tap filter */

    /* prefetch src_ptr data to cache memory */
    prefetch_load(src_ptr + 16);

    for (i = 16; i--;) {
      /* unrolling for loop */
      for (j = 0; j < 16; j += 8) {
        /* apply filter with vectors pairs */
        __asm__ __volatile__(
            "lbu            %[src_ptr_l1],  -16(%[src_ptr])                 "
            "\n\t"
            "lbu            %[src_ptr_0],   0(%[src_ptr])                   "
            "\n\t"
            "lbu            %[src_ptr_r1],  16(%[src_ptr])                  "
            "\n\t"
            "lbu            %[src_ptr_r2],  32(%[src_ptr])                  "
            "\n\t"
            "mtlo           %[vector4a],    $ac2                            "
            "\n\t"
            "append         %[src_ptr_0],   %[src_ptr_r1],  8               "
            "\n\t"
            "append         %[src_ptr_l1],  %[src_ptr_r2],  8               "
            "\n\t"
            "dpau.h.qbr     $ac2,           %[src_ptr_0],   %[vector2b]     "
            "\n\t"
            "dpsu.h.qbr     $ac2,           %[src_ptr_l1],  %[vector3b]     "
            "\n\t"

            "lbu            %[src_ptr_l1],  -15(%[src_ptr])                 "
            "\n\t"
            "lbu            %[src_ptr_0],   1(%[src_ptr])                   "
            "\n\t"
            "lbu            %[src_ptr_r1],  17(%[src_ptr])                  "
            "\n\t"
            "lbu            %[src_ptr_r2],  33(%[src_ptr])                  "
            "\n\t"
            "mtlo           %[vector4a],    $ac3                            "
            "\n\t"
            "extp           %[Temp1],       $ac2,           9               "
            "\n\t"

            "append         %[src_ptr_0],   %[src_ptr_r1],  8               "
            "\n\t"
            "append         %[src_ptr_l1],  %[src_ptr_r2],  8               "
            "\n\t"
            "dpau.h.qbr     $ac3,           %[src_ptr_0],   %[vector2b]     "
            "\n\t"
            "dpsu.h.qbr     $ac3,           %[src_ptr_l1],  %[vector3b]     "
            "\n\t"

            "lbu            %[src_ptr_l1],  -14(%[src_ptr])                 "
            "\n\t"
            "lbu            %[src_ptr_0],   2(%[src_ptr])                   "
            "\n\t"
            "lbu            %[src_ptr_r1],  18(%[src_ptr])                  "
            "\n\t"
            "lbu            %[src_ptr_r2],  34(%[src_ptr])                  "
            "\n\t"
            "mtlo           %[vector4a],    $ac1                            "
            "\n\t"
            "extp           %[Temp2],       $ac3,           9               "
            "\n\t"

            "append         %[src_ptr_0],   %[src_ptr_r1],  8               "
            "\n\t"
            "append         %[src_ptr_l1],  %[src_ptr_r2],  8               "
            "\n\t"
            "dpau.h.qbr     $ac1,           %[src_ptr_0],   %[vector2b]     "
            "\n\t"
            "dpsu.h.qbr     $ac1,           %[src_ptr_l1],  %[vector3b]     "
            "\n\t"

            "lbu            %[src_ptr_l1],  -13(%[src_ptr])                 "
            "\n\t"
            "lbu            %[src_ptr_0],   3(%[src_ptr])                   "
            "\n\t"
            "lbu            %[src_ptr_r1],  19(%[src_ptr])                  "
            "\n\t"
            "lbu            %[src_ptr_r2],  35(%[src_ptr])                  "
            "\n\t"
            "mtlo           %[vector4a],    $ac3                            "
            "\n\t"
            "extp           %[Temp3],       $ac1,           9               "
            "\n\t"

            "append         %[src_ptr_0],   %[src_ptr_r1],  8               "
            "\n\t"
            "append         %[src_ptr_l1],  %[src_ptr_r2],  8               "
            "\n\t"
            "dpau.h.qbr     $ac3,           %[src_ptr_0],   %[vector2b]     "
            "\n\t"
            "dpsu.h.qbr     $ac3,           %[src_ptr_l1],  %[vector3b]     "
            "\n\t"

            "lbu            %[src_ptr_l1],  -12(%[src_ptr])                 "
            "\n\t"
            "lbu            %[src_ptr_0],   4(%[src_ptr])                   "
            "\n\t"
            "lbu            %[src_ptr_r1],  20(%[src_ptr])                  "
            "\n\t"
            "lbu            %[src_ptr_r2],  36(%[src_ptr])                  "
            "\n\t"
            "mtlo           %[vector4a],    $ac2                            "
            "\n\t"
            "extp           %[Temp4],       $ac3,           9               "
            "\n\t"

            "append         %[src_ptr_0],   %[src_ptr_r1],  8               "
            "\n\t"
            "append         %[src_ptr_l1],  %[src_ptr_r2],  8               "
            "\n\t"
            "dpau.h.qbr     $ac2,           %[src_ptr_0],   %[vector2b]     "
            "\n\t"
            "dpsu.h.qbr     $ac2,           %[src_ptr_l1],  %[vector3b]     "
            "\n\t"

            "lbu            %[src_ptr_l1],  -11(%[src_ptr])                 "
            "\n\t"
            "lbu            %[src_ptr_0],   5(%[src_ptr])                   "
            "\n\t"
            "lbu            %[src_ptr_r1],  21(%[src_ptr])                  "
            "\n\t"
            "lbu            %[src_ptr_r2],  37(%[src_ptr])                  "
            "\n\t"
            "mtlo           %[vector4a],    $ac3                            "
            "\n\t"
            "extp           %[Temp5],       $ac2,           9               "
            "\n\t"

            "append         %[src_ptr_0],   %[src_ptr_r1],  8               "
            "\n\t"
            "append         %[src_ptr_l1],  %[src_ptr_r2],  8               "
            "\n\t"
            "dpau.h.qbr     $ac3,           %[src_ptr_0],   %[vector2b]     "
            "\n\t"
            "dpsu.h.qbr     $ac3,           %[src_ptr_l1],  %[vector3b]     "
            "\n\t"

            "lbu            %[src_ptr_l1],  -10(%[src_ptr])                 "
            "\n\t"
            "lbu            %[src_ptr_0],   6(%[src_ptr])                   "
            "\n\t"
            "lbu            %[src_ptr_r1],  22(%[src_ptr])                  "
            "\n\t"
            "lbu            %[src_ptr_r2],  38(%[src_ptr])                  "
            "\n\t"
            "mtlo           %[vector4a],    $ac1                            "
            "\n\t"
            "extp           %[Temp6],       $ac3,           9               "
            "\n\t"

            "append         %[src_ptr_0],   %[src_ptr_r1],  8               "
            "\n\t"
            "append         %[src_ptr_l1],  %[src_ptr_r2],  8               "
            "\n\t"
            "dpau.h.qbr     $ac1,           %[src_ptr_0],   %[vector2b]     "
            "\n\t"
            "dpsu.h.qbr     $ac1,           %[src_ptr_l1],  %[vector3b]     "
            "\n\t"

            "lbu            %[src_ptr_l1],  -9(%[src_ptr])                  "
            "\n\t"
            "lbu            %[src_ptr_0],   7(%[src_ptr])                   "
            "\n\t"
            "lbu            %[src_ptr_r1],  23(%[src_ptr])                  "
            "\n\t"
            "lbu            %[src_ptr_r2],  39(%[src_ptr])                  "
            "\n\t"
            "mtlo           %[vector4a],    $ac3                            "
            "\n\t"
            "extp           %[Temp7],       $ac1,           9               "
            "\n\t"

            "append         %[src_ptr_0],   %[src_ptr_r1],  8               "
            "\n\t"
            "append         %[src_ptr_l1],  %[src_ptr_r2],  8               "
            "\n\t"
            "dpau.h.qbr     $ac3,           %[src_ptr_0],   %[vector2b]     "
            "\n\t"
            "dpsu.h.qbr     $ac3,           %[src_ptr_l1],  %[vector3b]     "
            "\n\t"
            "extp           %[Temp8],       $ac3,           9               "
            "\n\t"

            : [Temp1] "=&r"(Temp1), [Temp2] "=&r"(Temp2), [Temp3] "=&r"(Temp3),
              [Temp4] "=&r"(Temp4), [Temp5] "=&r"(Temp5), [Temp6] "=&r"(Temp6),
              [Temp7] "=&r"(Temp7), [Temp8] "=r"(Temp8),
              [src_ptr_l1] "=&r"(src_ptr_l1), [src_ptr_0] "=&r"(src_ptr_0),
              [src_ptr_r1] "=&r"(src_ptr_r1), [src_ptr_r2] "=&r"(src_ptr_r2)
            : [vector2b] "r"(vector2b), [vector3b] "r"(vector3b),
              [vector4a] "r"(vector4a), [src_ptr] "r"(src_ptr));

        /* clamp and store results */
        output_ptr[j] = cm[Temp1];
        output_ptr[j + 1] = cm[Temp2];
        output_ptr[j + 2] = cm[Temp3];
        output_ptr[j + 3] = cm[Temp4];
        output_ptr[j + 4] = cm[Temp5];
        output_ptr[j + 5] = cm[Temp6];
        output_ptr[j + 6] = cm[Temp7];
        output_ptr[j + 7] = cm[Temp8];

        src_ptr += 8;
      }

      output_ptr += output_pitch;
    }
  } else {
    /* 4 tap filter */

    /* prefetch src_ptr data to cache memory */
    prefetch_load(src_ptr + 16);

    /* unroll for loop */
    for (i = 16; i--;) {
      /* apply filter with vectors pairs */
      __asm__ __volatile__(
          "lbu            %[src_ptr_l2],  -32(%[src_ptr])                 \n\t"
          "lbu            %[src_ptr_l1],  -16(%[src_ptr])                 \n\t"
          "lbu            %[src_ptr_0],   0(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r1],  16(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r2],  32(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r3],  48(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac2                            \n\t"

          "append         %[src_ptr_l2],  %[src_ptr_r3],  8               \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac2,           %[src_ptr_l2],  %[vector1b]     \n\t"
          "dpau.h.qbr     $ac2,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac2,           %[src_ptr_l1],  %[vector3b]     \n\t"

          "lbu            %[src_ptr_l2],  -31(%[src_ptr])                 \n\t"
          "lbu            %[src_ptr_l1],  -15(%[src_ptr])                 \n\t"
          "lbu            %[src_ptr_0],   1(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r1],  17(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r2],  33(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r3],  49(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac0                            \n\t"
          "extp           %[Temp1],       $ac2,           9               \n\t"

          "append         %[src_ptr_l2],  %[src_ptr_r3],  8               \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac0,           %[src_ptr_l2],  %[vector1b]     \n\t"
          "dpau.h.qbr     $ac0,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac0,           %[src_ptr_l1],  %[vector3b]     \n\t"

          "lbu            %[src_ptr_l2],  -30(%[src_ptr])                 \n\t"
          "lbu            %[src_ptr_l1],  -14(%[src_ptr])                 \n\t"
          "lbu            %[src_ptr_0],   2(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r1],  18(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r2],  34(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r3],  50(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac1                            \n\t"
          "extp           %[Temp2],       $ac0,           9               \n\t"

          "append         %[src_ptr_l2],  %[src_ptr_r3],  8               \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac1,           %[src_ptr_l2],  %[vector1b]     \n\t"
          "dpau.h.qbr     $ac1,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac1,           %[src_ptr_l1],  %[vector3b]     \n\t"

          "lbu            %[src_ptr_l2],  -29(%[src_ptr])                 \n\t"
          "lbu            %[src_ptr_l1],  -13(%[src_ptr])                 \n\t"
          "lbu            %[src_ptr_0],   3(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r1],  19(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r2],  35(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r3],  51(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac3                            \n\t"
          "extp           %[Temp3],       $ac1,           9               \n\t"

          "append         %[src_ptr_l2],  %[src_ptr_r3],  8               \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac3,           %[src_ptr_l2],  %[vector1b]     \n\t"
          "dpau.h.qbr     $ac3,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac3,           %[src_ptr_l1],  %[vector3b]     \n\t"

          "lbu            %[src_ptr_l2],  -28(%[src_ptr])                 \n\t"
          "lbu            %[src_ptr_l1],  -12(%[src_ptr])                 \n\t"
          "lbu            %[src_ptr_0],   4(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r1],  20(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r2],  36(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r3],  52(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac2                            \n\t"
          "extp           %[Temp4],       $ac3,           9               \n\t"

          "append         %[src_ptr_l2],  %[src_ptr_r3],  8               \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac2,           %[src_ptr_l2],  %[vector1b]     \n\t"
          "dpau.h.qbr     $ac2,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac2,           %[src_ptr_l1],  %[vector3b]     \n\t"

          "lbu            %[src_ptr_l2],  -27(%[src_ptr])                 \n\t"
          "lbu            %[src_ptr_l1],  -11(%[src_ptr])                 \n\t"
          "lbu            %[src_ptr_0],   5(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r1],  21(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r2],  37(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r3],  53(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac0                            \n\t"
          "extp           %[Temp5],       $ac2,           9               \n\t"

          "append         %[src_ptr_l2],  %[src_ptr_r3],  8               \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac0,           %[src_ptr_l2],  %[vector1b]     \n\t"
          "dpau.h.qbr     $ac0,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac0,           %[src_ptr_l1],  %[vector3b]     \n\t"

          "lbu            %[src_ptr_l2],  -26(%[src_ptr])                 \n\t"
          "lbu            %[src_ptr_l1],  -10(%[src_ptr])                 \n\t"
          "lbu            %[src_ptr_0],   6(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r1],  22(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r2],  38(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r3],  54(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac1                            \n\t"
          "extp           %[Temp6],       $ac0,           9               \n\t"

          "append         %[src_ptr_l2],  %[src_ptr_r3],  8               \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac1,           %[src_ptr_l2],  %[vector1b]     \n\t"
          "dpau.h.qbr     $ac1,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac1,           %[src_ptr_l1],  %[vector3b]     \n\t"

          "lbu            %[src_ptr_l2],  -25(%[src_ptr])                 \n\t"
          "lbu            %[src_ptr_l1],  -9(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_0],   7(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r1],  23(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r2],  39(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r3],  55(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac3                            \n\t"
          "extp           %[Temp7],       $ac1,           9               \n\t"

          "append         %[src_ptr_l2],  %[src_ptr_r3],  8               \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac3,           %[src_ptr_l2],  %[vector1b]     \n\t"
          "dpau.h.qbr     $ac3,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac3,           %[src_ptr_l1],  %[vector3b]     \n\t"
          "extp           %[Temp8],       $ac3,           9               \n\t"

          : [Temp1] "=&r"(Temp1), [Temp2] "=&r"(Temp2), [Temp3] "=&r"(Temp3),
            [Temp4] "=&r"(Temp4), [Temp5] "=&r"(Temp5), [Temp6] "=&r"(Temp6),
            [Temp7] "=&r"(Temp7), [Temp8] "=r"(Temp8),
            [src_ptr_l1] "=&r"(src_ptr_l1), [src_ptr_0] "=&r"(src_ptr_0),
            [src_ptr_r1] "=&r"(src_ptr_r1), [src_ptr_r2] "=&r"(src_ptr_r2),
            [src_ptr_l2] "=&r"(src_ptr_l2), [src_ptr_r3] "=&r"(src_ptr_r3)
          : [vector1b] "r"(vector1b), [vector2b] "r"(vector2b),
            [vector3b] "r"(vector3b), [vector4a] "r"(vector4a),
            [src_ptr] "r"(src_ptr));

      /* clamp and store results */
      output_ptr[0] = cm[Temp1];
      output_ptr[1] = cm[Temp2];
      output_ptr[2] = cm[Temp3];
      output_ptr[3] = cm[Temp4];
      output_ptr[4] = cm[Temp5];
      output_ptr[5] = cm[Temp6];
      output_ptr[6] = cm[Temp7];
      output_ptr[7] = cm[Temp8];

      /* apply filter with vectors pairs */
      __asm__ __volatile__(
          "lbu            %[src_ptr_l2],  -24(%[src_ptr])                 \n\t"
          "lbu            %[src_ptr_l1],  -8(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_0],   8(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r1],  24(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r2],  40(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r3],  56(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac2                            \n\t"

          "append         %[src_ptr_l2],  %[src_ptr_r3],  8               \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac2,           %[src_ptr_l2],  %[vector1b]     \n\t"
          "dpau.h.qbr     $ac2,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac2,           %[src_ptr_l1],  %[vector3b]     \n\t"

          "lbu            %[src_ptr_l2],  -23(%[src_ptr])                 \n\t"
          "lbu            %[src_ptr_l1],  -7(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_0],   9(%[src_ptr])                   \n\t"
          "lbu            %[src_ptr_r1],  25(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r2],  41(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r3],  57(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac0                            \n\t"
          "extp           %[Temp1],       $ac2,           9               \n\t"

          "append         %[src_ptr_l2],  %[src_ptr_r3],  8               \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac0,           %[src_ptr_l2],  %[vector1b]     \n\t"
          "dpau.h.qbr     $ac0,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac0,           %[src_ptr_l1],  %[vector3b]     \n\t"

          "lbu            %[src_ptr_l2],  -22(%[src_ptr])                 \n\t"
          "lbu            %[src_ptr_l1],  -6(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_0],   10(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r1],  26(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r2],  42(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r3],  58(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac1                            \n\t"
          "extp           %[Temp2],       $ac0,           9               \n\t"

          "append         %[src_ptr_l2],  %[src_ptr_r3],  8               \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac1,           %[src_ptr_l2],  %[vector1b]     \n\t"
          "dpau.h.qbr     $ac1,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac1,           %[src_ptr_l1],  %[vector3b]     \n\t"

          "lbu            %[src_ptr_l2],  -21(%[src_ptr])                 \n\t"
          "lbu            %[src_ptr_l1],  -5(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_0],   11(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r1],  27(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r2],  43(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r3],  59(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac3                            \n\t"
          "extp           %[Temp3],       $ac1,           9               \n\t"

          "append         %[src_ptr_l2],  %[src_ptr_r3],  8               \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac3,           %[src_ptr_l2],  %[vector1b]     \n\t"
          "dpau.h.qbr     $ac3,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac3,           %[src_ptr_l1],  %[vector3b]     \n\t"

          "lbu            %[src_ptr_l2],  -20(%[src_ptr])                 \n\t"
          "lbu            %[src_ptr_l1],  -4(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_0],   12(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r1],  28(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r2],  44(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r3],  60(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac2                            \n\t"
          "extp           %[Temp4],       $ac3,           9               \n\t"

          "append         %[src_ptr_l2],  %[src_ptr_r3],  8               \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac2,           %[src_ptr_l2],  %[vector1b]     \n\t"
          "dpau.h.qbr     $ac2,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac2,           %[src_ptr_l1],  %[vector3b]     \n\t"

          "lbu            %[src_ptr_l2],  -19(%[src_ptr])                 \n\t"
          "lbu            %[src_ptr_l1],  -3(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_0],   13(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r1],  29(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r2],  45(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r3],  61(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac0                            \n\t"
          "extp           %[Temp5],       $ac2,           9               \n\t"

          "append         %[src_ptr_l2],  %[src_ptr_r3],  8               \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac0,           %[src_ptr_l2],  %[vector1b]     \n\t"
          "dpau.h.qbr     $ac0,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac0,           %[src_ptr_l1],  %[vector3b]     \n\t"

          "lbu            %[src_ptr_l2],  -18(%[src_ptr])                 \n\t"
          "lbu            %[src_ptr_l1],  -2(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_0],   14(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r1],  30(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r2],  46(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r3],  62(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac1                            \n\t"
          "extp           %[Temp6],       $ac0,           9               \n\t"

          "append         %[src_ptr_l2],  %[src_ptr_r3],  8               \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac1,           %[src_ptr_l2],  %[vector1b]     \n\t"
          "dpau.h.qbr     $ac1,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac1,           %[src_ptr_l1],  %[vector3b]     \n\t"

          "lbu            %[src_ptr_l2],  -17(%[src_ptr])                 \n\t"
          "lbu            %[src_ptr_l1],  -1(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_0],   15(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r1],  31(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r2],  47(%[src_ptr])                  \n\t"
          "lbu            %[src_ptr_r3],  63(%[src_ptr])                  \n\t"
          "mtlo           %[vector4a],    $ac3                            \n\t"
          "extp           %[Temp7],       $ac1,           9               \n\t"

          "append         %[src_ptr_l2],  %[src_ptr_r3],  8               \n\t"
          "append         %[src_ptr_0],   %[src_ptr_r1],  8               \n\t"
          "append         %[src_ptr_l1],  %[src_ptr_r2],  8               \n\t"
          "dpau.h.qbr     $ac3,           %[src_ptr_l2],  %[vector1b]     \n\t"
          "dpau.h.qbr     $ac3,           %[src_ptr_0],   %[vector2b]     \n\t"
          "dpsu.h.qbr     $ac3,           %[src_ptr_l1],  %[vector3b]     \n\t"
          "extp           %[Temp8],       $ac3,           9               \n\t"

          : [Temp1] "=&r"(Temp1), [Temp2] "=&r"(Temp2), [Temp3] "=&r"(Temp3),
            [Temp4] "=&r"(Temp4), [Temp5] "=&r"(Temp5), [Temp6] "=&r"(Temp6),
            [Temp7] "=&r"(Temp7), [Temp8] "=r"(Temp8),
            [src_ptr_l1] "=&r"(src_ptr_l1), [src_ptr_0] "=&r"(src_ptr_0),
            [src_ptr_r1] "=&r"(src_ptr_r1), [src_ptr_r2] "=&r"(src_ptr_r2),
            [src_ptr_l2] "=&r"(src_ptr_l2), [src_ptr_r3] "=&r"(src_ptr_r3)
          : [vector1b] "r"(vector1b), [vector2b] "r"(vector2b),
            [vector3b] "r"(vector3b), [vector4a] "r"(vector4a),
            [src_ptr] "r"(src_ptr));

      src_ptr += 16;
      output_ptr[8] = cm[Temp1];
      output_ptr[9] = cm[Temp2];
      output_ptr[10] = cm[Temp3];
      output_ptr[11] = cm[Temp4];
      output_ptr[12] = cm[Temp5];
      output_ptr[13] = cm[Temp6];
      output_ptr[14] = cm[Temp7];
      output_ptr[15] = cm[Temp8];

      output_ptr += output_pitch;
    }
  }
}

void vp8_sixtap_predict4x4_dspr2(unsigned char *RESTRICT src_ptr,
                                 int src_pixels_per_line, int xoffset,
                                 int yoffset, unsigned char *RESTRICT dst_ptr,
                                 int dst_pitch) {
  unsigned char FData[9 * 4]; /* Temp data bufffer used in filtering */
  unsigned int pos = 16;

  /* bit positon for extract from acc */
  __asm__ __volatile__("wrdsp      %[pos],     1           \n\t"
                       :
                       : [pos] "r"(pos));

  if (yoffset) {
    /* First filter 1-D horizontally... */
    vp8_filter_block2d_first_pass_4(src_ptr - (2 * src_pixels_per_line), FData,
                                    src_pixels_per_line, 9, xoffset, 4);
    /* then filter verticaly... */
    vp8_filter_block2d_second_pass4(FData + 8, dst_ptr, dst_pitch, yoffset);
  } else
    /* if (yoffsset == 0) vp8_filter_block2d_first_pass save data to dst_ptr */
    vp8_filter_block2d_first_pass_4(src_ptr, dst_ptr, src_pixels_per_line, 4,
                                    xoffset, dst_pitch);
}

void vp8_sixtap_predict8x8_dspr2(unsigned char *RESTRICT src_ptr,
                                 int src_pixels_per_line, int xoffset,
                                 int yoffset, unsigned char *RESTRICT dst_ptr,
                                 int dst_pitch) {
  unsigned char FData[13 * 8]; /* Temp data bufffer used in filtering */
  unsigned int pos, Temp1, Temp2;

  pos = 16;

  /* bit positon for extract from acc */
  __asm__ __volatile__("wrdsp      %[pos],     1               \n\t"
                       :
                       : [pos] "r"(pos));

  if (yoffset) {
    src_ptr = src_ptr - (2 * src_pixels_per_line);

    if (xoffset) /* filter 1-D horizontally... */
      vp8_filter_block2d_first_pass_8_all(src_ptr, FData, src_pixels_per_line,
                                          13, xoffset, 8);

    else {
      /* prefetch src_ptr data to cache memory */
      prefetch_load(src_ptr + 2 * src_pixels_per_line);

      __asm__ __volatile__(
          "ulw    %[Temp1],   0(%[src_ptr])                           \n\t"
          "ulw    %[Temp2],   4(%[src_ptr])                           \n\t"
          "sw     %[Temp1],   0(%[FData])                             \n\t"
          "sw     %[Temp2],   4(%[FData])                             \n\t"
          "addu   %[src_ptr], %[src_ptr],    %[src_pixels_per_line]   \n\t"

          "ulw    %[Temp1],   0(%[src_ptr])                           \n\t"
          "ulw    %[Temp2],   4(%[src_ptr])                           \n\t"
          "sw     %[Temp1],   8(%[FData])                             \n\t"
          "sw     %[Temp2],   12(%[FData])                            \n\t"
          "addu   %[src_ptr], %[src_ptr],    %[src_pixels_per_line]   \n\t"

          "ulw    %[Temp1],   0(%[src_ptr])                           \n\t"
          "ulw    %[Temp2],   4(%[src_ptr])                           \n\t"
          "sw     %[Temp1],   16(%[FData])                            \n\t"
          "sw     %[Temp2],   20(%[FData])                            \n\t"
          "addu   %[src_ptr], %[src_ptr],    %[src_pixels_per_line]   \n\t"

          "ulw    %[Temp1],   0(%[src_ptr])                           \n\t"
          "ulw    %[Temp2],   4(%[src_ptr])                           \n\t"
          "sw     %[Temp1],   24(%[FData])                            \n\t"
          "sw     %[Temp2],   28(%[FData])                            \n\t"
          "addu   %[src_ptr], %[src_ptr],    %[src_pixels_per_line]   \n\t"

          "ulw    %[Temp1],   0(%[src_ptr])                           \n\t"
          "ulw    %[Temp2],   4(%[src_ptr])                           \n\t"
          "sw     %[Temp1],   32(%[FData])                            \n\t"
          "sw     %[Temp2],   36(%[FData])                            \n\t"
          "addu   %[src_ptr], %[src_ptr],    %[src_pixels_per_line]   \n\t"

          "ulw    %[Temp1],   0(%[src_ptr])                           \n\t"
          "ulw    %[Temp2],   4(%[src_ptr])                           \n\t"
          "sw     %[Temp1],   40(%[FData])                            \n\t"
          "sw     %[Temp2],   44(%[FData])                            \n\t"
          "addu   %[src_ptr], %[src_ptr],    %[src_pixels_per_line]   \n\t"

          "ulw    %[Temp1],   0(%[src_ptr])                           \n\t"
          "ulw    %[Temp2],   4(%[src_ptr])                           \n\t"
          "sw     %[Temp1],   48(%[FData])                            \n\t"
          "sw     %[Temp2],   52(%[FData])                            \n\t"
          "addu   %[src_ptr], %[src_ptr],    %[src_pixels_per_line]   \n\t"

          "ulw    %[Temp1],   0(%[src_ptr])                           \n\t"
          "ulw    %[Temp2],   4(%[src_ptr])                           \n\t"
          "sw     %[Temp1],   56(%[FData])                            \n\t"
          "sw     %[Temp2],   60(%[FData])                            \n\t"
          "addu   %[src_ptr], %[src_ptr],    %[src_pixels_per_line]   \n\t"

          "ulw    %[Temp1],   0(%[src_ptr])                           \n\t"
          "ulw    %[Temp2],   4(%[src_ptr])                           \n\t"
          "sw     %[Temp1],   64(%[FData])                            \n\t"
          "sw     %[Temp2],   68(%[FData])                            \n\t"
          "addu   %[src_ptr], %[src_ptr],    %[src_pixels_per_line]   \n\t"

          "ulw    %[Temp1],   0(%[src_ptr])                           \n\t"
          "ulw    %[Temp2],   4(%[src_ptr])                           \n\t"
          "sw     %[Temp1],   72(%[FData])                            \n\t"
          "sw     %[Temp2],   76(%[FData])                            \n\t"
          "addu   %[src_ptr], %[src_ptr],    %[src_pixels_per_line]   \n\t"

          "ulw    %[Temp1],   0(%[src_ptr])                           \n\t"
          "ulw    %[Temp2],   4(%[src_ptr])                           \n\t"
          "sw     %[Temp1],   80(%[FData])                            \n\t"
          "sw     %[Temp2],   84(%[FData])                            \n\t"
          "addu   %[src_ptr], %[src_ptr],    %[src_pixels_per_line]   \n\t"

          "ulw    %[Temp1],   0(%[src_ptr])                           \n\t"
          "ulw    %[Temp2],   4(%[src_ptr])                           \n\t"
          "sw     %[Temp1],   88(%[FData])                            \n\t"
          "sw     %[Temp2],   92(%[FData])                            \n\t"
          "addu   %[src_ptr], %[src_ptr],    %[src_pixels_per_line]   \n\t"

          "ulw    %[Temp1],   0(%[src_ptr])                           \n\t"
          "ulw    %[Temp2],   4(%[src_ptr])                           \n\t"
          "sw     %[Temp1],   96(%[FData])                            \n\t"
          "sw     %[Temp2],   100(%[FData])                           \n\t"

          : [Temp1] "=&r"(Temp1), [Temp2] "=&r"(Temp2)
          : [FData] "r"(FData), [src_ptr] "r"(src_ptr),
            [src_pixels_per_line] "r"(src_pixels_per_line));
    }

    /* filter verticaly... */
    vp8_filter_block2d_second_pass_8(FData + 16, dst_ptr, dst_pitch, 8, 8,
                                     yoffset);
  }

  /* if (yoffsset == 0) vp8_filter_block2d_first_pass save data to dst_ptr */
  else {
    if (xoffset)
      vp8_filter_block2d_first_pass_8_all(src_ptr, dst_ptr, src_pixels_per_line,
                                          8, xoffset, dst_pitch);

    else {
      /* copy from src buffer to dst buffer */
      __asm__ __volatile__(
          "ulw    %[Temp1],   0(%[src_ptr])                           \n\t"
          "ulw    %[Temp2],   4(%[src_ptr])                           \n\t"
          "sw     %[Temp1],   0(%[dst_ptr])                           \n\t"
          "sw     %[Temp2],   4(%[dst_ptr])                           \n\t"
          "addu   %[src_ptr], %[src_ptr],    %[src_pixels_per_line]   \n\t"

          "ulw    %[Temp1],   0(%[src_ptr])                           \n\t"
          "ulw    %[Temp2],   4(%[src_ptr])                           \n\t"
          "sw     %[Temp1],   8(%[dst_ptr])                           \n\t"
          "sw     %[Temp2],   12(%[dst_ptr])                          \n\t"
          "addu   %[src_ptr], %[src_ptr],    %[src_pixels_per_line]   \n\t"

          "ulw    %[Temp1],   0(%[src_ptr])                           \n\t"
          "ulw    %[Temp2],   4(%[src_ptr])                           \n\t"
          "sw     %[Temp1],   16(%[dst_ptr])                          \n\t"
          "sw     %[Temp2],   20(%[dst_ptr])                          \n\t"
          "addu   %[src_ptr], %[src_ptr],    %[src_pixels_per_line]   \n\t"

          "ulw    %[Temp1],   0(%[src_ptr])                           \n\t"
          "ulw    %[Temp2],   4(%[src_ptr])                           \n\t"
          "sw     %[Temp1],   24(%[dst_ptr])                          \n\t"
          "sw     %[Temp2],   28(%[dst_ptr])                          \n\t"
          "addu   %[src_ptr], %[src_ptr],   %[src_pixels_per_line]    \n\t"

          "ulw    %[Temp1],   0(%[src_ptr])                           \n\t"
          "ulw    %[Temp2],   4(%[src_ptr])                           \n\t"
          "sw     %[Temp1],   32(%[dst_ptr])                          \n\t"
          "sw     %[Temp2],   36(%[dst_ptr])                          \n\t"
          "addu   %[src_ptr], %[src_ptr],    %[src_pixels_per_line]   \n\t"

          "ulw    %[Temp1],   0(%[src_ptr])                           \n\t"
          "ulw    %[Temp2],   4(%[src_ptr])                           \n\t"
          "sw     %[Temp1],   40(%[dst_ptr])                          \n\t"
          "sw     %[Temp2],   44(%[dst_ptr])                          \n\t"
          "addu   %[src_ptr], %[src_ptr],    %[src_pixels_per_line]   \n\t"

          "ulw    %[Temp1],   0(%[src_ptr])                           \n\t"
          "ulw    %[Temp2],   4(%[src_ptr])                           \n\t"
          "sw     %[Temp1],   48(%[dst_ptr])                          \n\t"
          "sw     %[Temp2],   52(%[dst_ptr])                          \n\t"
          "addu   %[src_ptr], %[src_ptr],    %[src_pixels_per_line]   \n\t"

          "ulw    %[Temp1],   0(%[src_ptr])                           \n\t"
          "ulw    %[Temp2],   4(%[src_ptr])                           \n\t"
          "sw     %[Temp1],   56(%[dst_ptr])                          \n\t"
          "sw     %[Temp2],   60(%[dst_ptr])                          \n\t"

          : [Temp1] "=&r"(Temp1), [Temp2] "=&r"(Temp2)
          : [dst_ptr] "r"(dst_ptr), [src_ptr] "r"(src_ptr),
            [src_pixels_per_line] "r"(src_pixels_per_line));
    }
  }
}

void vp8_sixtap_predict8x4_dspr2(unsigned char *RESTRICT src_ptr,
                                 int src_pixels_per_line, int xoffset,
                                 int yoffset, unsigned char *RESTRICT dst_ptr,
                                 int dst_pitch) {
  unsigned char FData[9 * 8]; /* Temp data bufffer used in filtering */
  unsigned int pos, Temp1, Temp2;

  pos = 16;

  /* bit positon for extract from acc */
  __asm__ __volatile__("wrdsp      %[pos],     1           \n\t"
                       :
                       : [pos] "r"(pos));

  if (yoffset) {
    src_ptr = src_ptr - (2 * src_pixels_per_line);

    if (xoffset) /* filter 1-D horizontally... */
      vp8_filter_block2d_first_pass_8_all(src_ptr, FData, src_pixels_per_line,
                                          9, xoffset, 8);

    else {
      /* prefetch src_ptr data to cache memory */
      prefetch_load(src_ptr + 2 * src_pixels_per_line);

      __asm__ __volatile__(
          "ulw    %[Temp1],   0(%[src_ptr])                           \n\t"
          "ulw    %[Temp2],   4(%[src_ptr])                           \n\t"
          "sw     %[Temp1],   0(%[FData])                             \n\t"
          "sw     %[Temp2],   4(%[FData])                             \n\t"
          "addu   %[src_ptr], %[src_ptr],    %[src_pixels_per_line]   \n\t"

          "ulw    %[Temp1],   0(%[src_ptr])                           \n\t"
          "ulw    %[Temp2],   4(%[src_ptr])                           \n\t"
          "sw     %[Temp1],   8(%[FData])                             \n\t"
          "sw     %[Temp2],   12(%[FData])                            \n\t"
          "addu   %[src_ptr], %[src_ptr],    %[src_pixels_per_line]   \n\t"

          "ulw    %[Temp1],   0(%[src_ptr])                           \n\t"
          "ulw    %[Temp2],   4(%[src_ptr])                           \n\t"
          "sw     %[Temp1],   16(%[FData])                            \n\t"
          "sw     %[Temp2],   20(%[FData])                            \n\t"
          "addu   %[src_ptr], %[src_ptr],    %[src_pixels_per_line]   \n\t"

          "ulw    %[Temp1],   0(%[src_ptr])                           \n\t"
          "ulw    %[Temp2],   4(%[src_ptr])                           \n\t"
          "sw     %[Temp1],   24(%[FData])                            \n\t"
          "sw     %[Temp2],   28(%[FData])                            \n\t"
          "addu   %[src_ptr], %[src_ptr],    %[src_pixels_per_line]   \n\t"

          "ulw    %[Temp1],   0(%[src_ptr])                           \n\t"
          "ulw    %[Temp2],   4(%[src_ptr])                           \n\t"
          "sw     %[Temp1],   32(%[FData])                            \n\t"
          "sw     %[Temp2],   36(%[FData])                            \n\t"
          "addu   %[src_ptr], %[src_ptr],    %[src_pixels_per_line]   \n\t"

          "ulw    %[Temp1],   0(%[src_ptr])                           \n\t"
          "ulw    %[Temp2],   4(%[src_ptr])                           \n\t"
          "sw     %[Temp1],   40(%[FData])                            \n\t"
          "sw     %[Temp2],   44(%[FData])                            \n\t"
          "addu   %[src_ptr], %[src_ptr],    %[src_pixels_per_line]   \n\t"

          "ulw    %[Temp1],   0(%[src_ptr])                           \n\t"
          "ulw    %[Temp2],   4(%[src_ptr])                           \n\t"
          "sw     %[Temp1],   48(%[FData])                            \n\t"
          "sw     %[Temp2],   52(%[FData])                            \n\t"
          "addu   %[src_ptr], %[src_ptr],    %[src_pixels_per_line]   \n\t"

          "ulw    %[Temp1],   0(%[src_ptr])                           \n\t"
          "ulw    %[Temp2],   4(%[src_ptr])                           \n\t"
          "sw     %[Temp1],   56(%[FData])                            \n\t"
          "sw     %[Temp2],   60(%[FData])                            \n\t"
          "addu   %[src_ptr], %[src_ptr],    %[src_pixels_per_line]   \n\t"

          "ulw    %[Temp1],   0(%[src_ptr])                           \n\t"
          "ulw    %[Temp2],   4(%[src_ptr])                           \n\t"
          "sw     %[Temp1],   64(%[FData])                            \n\t"
          "sw     %[Temp2],   68(%[FData])                            \n\t"

          : [Temp1] "=&r"(Temp1), [Temp2] "=&r"(Temp2)
          : [FData] "r"(FData), [src_ptr] "r"(src_ptr),
            [src_pixels_per_line] "r"(src_pixels_per_line));
    }

    /* filter verticaly... */
    vp8_filter_block2d_second_pass_8(FData + 16, dst_ptr, dst_pitch, 4, 8,
                                     yoffset);
  }

  /* if (yoffsset == 0) vp8_filter_block2d_first_pass save data to dst_ptr */
  else {
    if (xoffset)
      vp8_filter_block2d_first_pass_8_all(src_ptr, dst_ptr, src_pixels_per_line,
                                          4, xoffset, dst_pitch);

    else {
      /* copy from src buffer to dst buffer */
      __asm__ __volatile__(
          "ulw    %[Temp1],   0(%[src_ptr])                           \n\t"
          "ulw    %[Temp2],   4(%[src_ptr])                           \n\t"
          "sw     %[Temp1],   0(%[dst_ptr])                           \n\t"
          "sw     %[Temp2],   4(%[dst_ptr])                           \n\t"
          "addu   %[src_ptr], %[src_ptr],    %[src_pixels_per_line]   \n\t"

          "ulw    %[Temp1],   0(%[src_ptr])                           \n\t"
          "ulw    %[Temp2],   4(%[src_ptr])                           \n\t"
          "sw     %[Temp1],   8(%[dst_ptr])                           \n\t"
          "sw     %[Temp2],   12(%[dst_ptr])                          \n\t"
          "addu   %[src_ptr], %[src_ptr],    %[src_pixels_per_line]   \n\t"

          "ulw    %[Temp1],   0(%[src_ptr])                           \n\t"
          "ulw    %[Temp2],   4(%[src_ptr])                           \n\t"
          "sw     %[Temp1],   16(%[dst_ptr])                          \n\t"
          "sw     %[Temp2],   20(%[dst_ptr])                          \n\t"
          "addu   %[src_ptr], %[src_ptr],    %[src_pixels_per_line]   \n\t"

          "ulw    %[Temp1],   0(%[src_ptr])                           \n\t"
          "ulw    %[Temp2],   4(%[src_ptr])                           \n\t"
          "sw     %[Temp1],   24(%[dst_ptr])                          \n\t"
          "sw     %[Temp2],   28(%[dst_ptr])                          \n\t"

          : [Temp1] "=&r"(Temp1), [Temp2] "=&r"(Temp2)
          : [dst_ptr] "r"(dst_ptr), [src_ptr] "r"(src_ptr),
            [src_pixels_per_line] "r"(src_pixels_per_line));
    }
  }
}

void vp8_sixtap_predict16x16_dspr2(unsigned char *RESTRICT src_ptr,
                                   int src_pixels_per_line, int xoffset,
                                   int yoffset, unsigned char *RESTRICT dst_ptr,
                                   int dst_pitch) {
  const unsigned short *VFilter;
  unsigned char FData[21 * 16]; /* Temp data bufffer used in filtering */
  unsigned int pos;

  VFilter = sub_pel_filterss[yoffset];

  pos = 16;

  /* bit positon for extract from acc */
  __asm__ __volatile__("wrdsp      %[pos],     1           \n\t"
                       :
                       : [pos] "r"(pos));

  if (yoffset) {
    src_ptr = src_ptr - (2 * src_pixels_per_line);

    switch (xoffset) {
      /* filter 1-D horizontally... */
      case 2:
      case 4:
      case 6:
        /* 6 tap filter */
        vp8_filter_block2d_first_pass16_6tap(
            src_ptr, FData, src_pixels_per_line, 21, xoffset, 16);
        break;

      case 0:
        /* only copy buffer */
        vp8_filter_block2d_first_pass16_0(src_ptr, FData, src_pixels_per_line);
        break;

      case 1:
      case 3:
      case 5:
      case 7:
        /* 4 tap filter */
        vp8_filter_block2d_first_pass16_4tap(
            src_ptr, FData, src_pixels_per_line, 16, 21, xoffset, yoffset,
            dst_ptr, dst_pitch);
        break;
    }

    /* filter verticaly... */
    vp8_filter_block2d_second_pass161(FData + 32, dst_ptr, dst_pitch, VFilter);
  } else {
    /* if (yoffsset == 0) vp8_filter_block2d_first_pass save data to dst_ptr */
    switch (xoffset) {
      case 2:
      case 4:
      case 6:
        /* 6 tap filter */
        vp8_filter_block2d_first_pass16_6tap(
            src_ptr, dst_ptr, src_pixels_per_line, 16, xoffset, dst_pitch);
        break;

      case 1:
      case 3:
      case 5:
      case 7:
        /* 4 tap filter */
        vp8_filter_block2d_first_pass16_4tap(
            src_ptr, dst_ptr, src_pixels_per_line, 16, 21, xoffset, yoffset,
            dst_ptr, dst_pitch);
        break;
    }
  }
}

#endif
