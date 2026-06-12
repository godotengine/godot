/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "vp8_rtcd.h"

#if HAVE_DSPR2
#define CROP_WIDTH 256

/******************************************************************************
 * Notes:
 *
 * This implementation makes use of 16 bit fixed point version of two multiply
 * constants:
 *         1.   sqrt(2) * cos (pi/8)
 *         2.   sqrt(2) * sin (pi/8)
 * Since the first constant is bigger than 1, to maintain the same 16 bit
 * fixed point precision as the second one, we use a trick of
 *         x * a = x + x*(a-1)
 * so
 *         x * sqrt(2) * cos (pi/8) = x + x * (sqrt(2) *cos(pi/8)-1).
 ****************************************************************************/
extern unsigned char ff_cropTbl[256 + 2 * CROP_WIDTH];
static const int cospi8sqrt2minus1 = 20091;
static const int sinpi8sqrt2 = 35468;

inline void prefetch_load_short(short *src) {
  __asm__ __volatile__("pref   0,  0(%[src])   \n\t" : : [src] "r"(src));
}

void vp8_short_idct4x4llm_dspr2(short *input, unsigned char *pred_ptr,
                                int pred_stride, unsigned char *dst_ptr,
                                int dst_stride) {
  int r, c;
  int a1, b1, c1, d1;
  short output[16];
  short *ip = input;
  short *op = output;
  int temp1, temp2;
  int shortpitch = 4;

  int c2, d2;
  int temp3, temp4;
  unsigned char *cm = ff_cropTbl + CROP_WIDTH;

  /* prepare data for load */
  prefetch_load_short(ip + 8);

  /* first loop is unrolled */
  a1 = ip[0] + ip[8];
  b1 = ip[0] - ip[8];

  temp1 = (ip[4] * sinpi8sqrt2) >> 16;
  temp2 = ip[12] + ((ip[12] * cospi8sqrt2minus1) >> 16);
  c1 = temp1 - temp2;

  temp1 = ip[4] + ((ip[4] * cospi8sqrt2minus1) >> 16);
  temp2 = (ip[12] * sinpi8sqrt2) >> 16;
  d1 = temp1 + temp2;

  temp3 = (ip[5] * sinpi8sqrt2) >> 16;
  temp4 = ip[13] + ((ip[13] * cospi8sqrt2minus1) >> 16);
  c2 = temp3 - temp4;

  temp3 = ip[5] + ((ip[5] * cospi8sqrt2minus1) >> 16);
  temp4 = (ip[13] * sinpi8sqrt2) >> 16;
  d2 = temp3 + temp4;

  op[0] = a1 + d1;
  op[12] = a1 - d1;
  op[4] = b1 + c1;
  op[8] = b1 - c1;

  a1 = ip[1] + ip[9];
  b1 = ip[1] - ip[9];

  op[1] = a1 + d2;
  op[13] = a1 - d2;
  op[5] = b1 + c2;
  op[9] = b1 - c2;

  a1 = ip[2] + ip[10];
  b1 = ip[2] - ip[10];

  temp1 = (ip[6] * sinpi8sqrt2) >> 16;
  temp2 = ip[14] + ((ip[14] * cospi8sqrt2minus1) >> 16);
  c1 = temp1 - temp2;

  temp1 = ip[6] + ((ip[6] * cospi8sqrt2minus1) >> 16);
  temp2 = (ip[14] * sinpi8sqrt2) >> 16;
  d1 = temp1 + temp2;

  temp3 = (ip[7] * sinpi8sqrt2) >> 16;
  temp4 = ip[15] + ((ip[15] * cospi8sqrt2minus1) >> 16);
  c2 = temp3 - temp4;

  temp3 = ip[7] + ((ip[7] * cospi8sqrt2minus1) >> 16);
  temp4 = (ip[15] * sinpi8sqrt2) >> 16;
  d2 = temp3 + temp4;

  op[2] = a1 + d1;
  op[14] = a1 - d1;
  op[6] = b1 + c1;
  op[10] = b1 - c1;

  a1 = ip[3] + ip[11];
  b1 = ip[3] - ip[11];

  op[3] = a1 + d2;
  op[15] = a1 - d2;
  op[7] = b1 + c2;
  op[11] = b1 - c2;

  ip = output;

  /* prepare data for load */
  prefetch_load_short(ip + shortpitch);

  /* second loop is unrolled */
  a1 = ip[0] + ip[2];
  b1 = ip[0] - ip[2];

  temp1 = (ip[1] * sinpi8sqrt2) >> 16;
  temp2 = ip[3] + ((ip[3] * cospi8sqrt2minus1) >> 16);
  c1 = temp1 - temp2;

  temp1 = ip[1] + ((ip[1] * cospi8sqrt2minus1) >> 16);
  temp2 = (ip[3] * sinpi8sqrt2) >> 16;
  d1 = temp1 + temp2;

  temp3 = (ip[5] * sinpi8sqrt2) >> 16;
  temp4 = ip[7] + ((ip[7] * cospi8sqrt2minus1) >> 16);
  c2 = temp3 - temp4;

  temp3 = ip[5] + ((ip[5] * cospi8sqrt2minus1) >> 16);
  temp4 = (ip[7] * sinpi8sqrt2) >> 16;
  d2 = temp3 + temp4;

  op[0] = (a1 + d1 + 4) >> 3;
  op[3] = (a1 - d1 + 4) >> 3;
  op[1] = (b1 + c1 + 4) >> 3;
  op[2] = (b1 - c1 + 4) >> 3;

  a1 = ip[4] + ip[6];
  b1 = ip[4] - ip[6];

  op[4] = (a1 + d2 + 4) >> 3;
  op[7] = (a1 - d2 + 4) >> 3;
  op[5] = (b1 + c2 + 4) >> 3;
  op[6] = (b1 - c2 + 4) >> 3;

  a1 = ip[8] + ip[10];
  b1 = ip[8] - ip[10];

  temp1 = (ip[9] * sinpi8sqrt2) >> 16;
  temp2 = ip[11] + ((ip[11] * cospi8sqrt2minus1) >> 16);
  c1 = temp1 - temp2;

  temp1 = ip[9] + ((ip[9] * cospi8sqrt2minus1) >> 16);
  temp2 = (ip[11] * sinpi8sqrt2) >> 16;
  d1 = temp1 + temp2;

  temp3 = (ip[13] * sinpi8sqrt2) >> 16;
  temp4 = ip[15] + ((ip[15] * cospi8sqrt2minus1) >> 16);
  c2 = temp3 - temp4;

  temp3 = ip[13] + ((ip[13] * cospi8sqrt2minus1) >> 16);
  temp4 = (ip[15] * sinpi8sqrt2) >> 16;
  d2 = temp3 + temp4;

  op[8] = (a1 + d1 + 4) >> 3;
  op[11] = (a1 - d1 + 4) >> 3;
  op[9] = (b1 + c1 + 4) >> 3;
  op[10] = (b1 - c1 + 4) >> 3;

  a1 = ip[12] + ip[14];
  b1 = ip[12] - ip[14];

  op[12] = (a1 + d2 + 4) >> 3;
  op[15] = (a1 - d2 + 4) >> 3;
  op[13] = (b1 + c2 + 4) >> 3;
  op[14] = (b1 - c2 + 4) >> 3;

  ip = output;

  for (r = 0; r < 4; ++r) {
    for (c = 0; c < 4; ++c) {
      short a = ip[c] + pred_ptr[c];
      dst_ptr[c] = cm[a];
    }

    ip += 4;
    dst_ptr += dst_stride;
    pred_ptr += pred_stride;
  }
}

void vp8_dc_only_idct_add_dspr2(short input_dc, unsigned char *pred_ptr,
                                int pred_stride, unsigned char *dst_ptr,
                                int dst_stride) {
  int a1;
  int i, absa1;
  int t2, vector_a1, vector_a;

  /* a1 = ((input_dc + 4) >> 3); */
  __asm__ __volatile__(
      "addi  %[a1], %[input_dc], 4   \n\t"
      "sra   %[a1], %[a1],       3   \n\t"
      : [a1] "=r"(a1)
      : [input_dc] "r"(input_dc));

  if (a1 < 0) {
    /* use quad-byte
     * input and output memory are four byte aligned
     */
    __asm__ __volatile__(
        "abs        %[absa1],     %[a1]         \n\t"
        "replv.qb   %[vector_a1], %[absa1]      \n\t"
        : [absa1] "=r"(absa1), [vector_a1] "=r"(vector_a1)
        : [a1] "r"(a1));

    /* use (a1 - predptr[c]) instead a1 + predptr[c] */
    for (i = 4; i--;) {
      __asm__ __volatile__(
          "lw             %[t2],       0(%[pred_ptr])                     \n\t"
          "add            %[pred_ptr], %[pred_ptr],    %[pred_stride]     \n\t"
          "subu_s.qb      %[vector_a], %[t2],          %[vector_a1]       \n\t"
          "sw             %[vector_a], 0(%[dst_ptr])                      \n\t"
          "add            %[dst_ptr],  %[dst_ptr],     %[dst_stride]      \n\t"
          : [t2] "=&r"(t2), [vector_a] "=&r"(vector_a),
            [dst_ptr] "+&r"(dst_ptr), [pred_ptr] "+&r"(pred_ptr)
          : [dst_stride] "r"(dst_stride), [pred_stride] "r"(pred_stride),
            [vector_a1] "r"(vector_a1));
    }
  } else {
    /* use quad-byte
     * input and output memory are four byte aligned
     */
    __asm__ __volatile__("replv.qb       %[vector_a1], %[a1]     \n\t"
                         : [vector_a1] "=r"(vector_a1)
                         : [a1] "r"(a1));

    for (i = 4; i--;) {
      __asm__ __volatile__(
          "lw             %[t2],       0(%[pred_ptr])                 \n\t"
          "add            %[pred_ptr], %[pred_ptr],    %[pred_stride] \n\t"
          "addu_s.qb      %[vector_a], %[vector_a1],   %[t2]          \n\t"
          "sw             %[vector_a], 0(%[dst_ptr])                  \n\t"
          "add            %[dst_ptr],  %[dst_ptr],     %[dst_stride]  \n\t"
          : [t2] "=&r"(t2), [vector_a] "=&r"(vector_a),
            [dst_ptr] "+&r"(dst_ptr), [pred_ptr] "+&r"(pred_ptr)
          : [dst_stride] "r"(dst_stride), [pred_stride] "r"(pred_stride),
            [vector_a1] "r"(vector_a1));
    }
  }
}

void vp8_short_inv_walsh4x4_dspr2(short *input, short *mb_dqcoeff) {
  short output[16];
  int i;
  int a1, b1, c1, d1;
  int a2, b2, c2, d2;
  short *ip = input;
  short *op = output;

  prefetch_load_short(ip);

  for (i = 4; i--;) {
    a1 = ip[0] + ip[12];
    b1 = ip[4] + ip[8];
    c1 = ip[4] - ip[8];
    d1 = ip[0] - ip[12];

    op[0] = a1 + b1;
    op[4] = c1 + d1;
    op[8] = a1 - b1;
    op[12] = d1 - c1;

    ip++;
    op++;
  }

  ip = output;
  op = output;

  prefetch_load_short(ip);

  for (i = 4; i--;) {
    a1 = ip[0] + ip[3] + 3;
    b1 = ip[1] + ip[2];
    c1 = ip[1] - ip[2];
    d1 = ip[0] - ip[3] + 3;

    a2 = a1 + b1;
    b2 = d1 + c1;
    c2 = a1 - b1;
    d2 = d1 - c1;

    op[0] = a2 >> 3;
    op[1] = b2 >> 3;
    op[2] = c2 >> 3;
    op[3] = d2 >> 3;

    ip += 4;
    op += 4;
  }

  for (i = 0; i < 16; ++i) {
    mb_dqcoeff[i * 16] = output[i];
  }
}

void vp8_short_inv_walsh4x4_1_dspr2(short *input, short *mb_dqcoeff) {
  int a1;

  a1 = ((input[0] + 3) >> 3);

  __asm__ __volatile__(
      "sh             %[a1], 0(%[mb_dqcoeff])                    \n\t"
      "sh             %[a1], 32(%[mb_dqcoeff])                   \n\t"
      "sh             %[a1], 64(%[mb_dqcoeff])                   \n\t"
      "sh             %[a1], 96(%[mb_dqcoeff])                   \n\t"
      "sh             %[a1], 128(%[mb_dqcoeff])                  \n\t"
      "sh             %[a1], 160(%[mb_dqcoeff])                  \n\t"
      "sh             %[a1], 192(%[mb_dqcoeff])                  \n\t"
      "sh             %[a1], 224(%[mb_dqcoeff])                  \n\t"
      "sh             %[a1], 256(%[mb_dqcoeff])                  \n\t"
      "sh             %[a1], 288(%[mb_dqcoeff])                  \n\t"
      "sh             %[a1], 320(%[mb_dqcoeff])                  \n\t"
      "sh             %[a1], 352(%[mb_dqcoeff])                  \n\t"
      "sh             %[a1], 384(%[mb_dqcoeff])                  \n\t"
      "sh             %[a1], 416(%[mb_dqcoeff])                  \n\t"
      "sh             %[a1], 448(%[mb_dqcoeff])                  \n\t"
      "sh             %[a1], 480(%[mb_dqcoeff])                  \n\t"

      :
      : [a1] "r"(a1), [mb_dqcoeff] "r"(mb_dqcoeff));
}

#endif
