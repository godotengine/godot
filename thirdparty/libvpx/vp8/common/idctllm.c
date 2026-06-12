/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vp8_rtcd.h"

/****************************************************************************
 * Notes:
 *
 * This implementation makes use of 16 bit fixed point verio of two multiply
 * constants:
 *         1.   sqrt(2) * cos (pi/8)
 *         2.   sqrt(2) * sin (pi/8)
 * Becuase the first constant is bigger than 1, to maintain the same 16 bit
 * fixed point precision as the second one, we use a trick of
 *         x * a = x + x*(a-1)
 * so
 *         x * sqrt(2) * cos (pi/8) = x + x * (sqrt(2) *cos(pi/8)-1).
 **************************************************************************/
static const int cospi8sqrt2minus1 = 20091;
static const int sinpi8sqrt2 = 35468;

void vp8_short_idct4x4llm_c(short *input, unsigned char *pred_ptr,
                            int pred_stride, unsigned char *dst_ptr,
                            int dst_stride) {
  int i;
  int r, c;
  int a1, b1, c1, d1;
  short output[16];
  short *ip = input;
  short *op = output;
  int temp1, temp2;
  int shortpitch = 4;

  for (i = 0; i < 4; ++i) {
    a1 = ip[0] + ip[8];
    b1 = ip[0] - ip[8];

    temp1 = (ip[4] * sinpi8sqrt2) >> 16;
    temp2 = ip[12] + ((ip[12] * cospi8sqrt2minus1) >> 16);
    c1 = temp1 - temp2;

    temp1 = ip[4] + ((ip[4] * cospi8sqrt2minus1) >> 16);
    temp2 = (ip[12] * sinpi8sqrt2) >> 16;
    d1 = temp1 + temp2;

    op[shortpitch * 0] = a1 + d1;
    op[shortpitch * 3] = a1 - d1;

    op[shortpitch * 1] = b1 + c1;
    op[shortpitch * 2] = b1 - c1;

    ip++;
    op++;
  }

  ip = output;
  op = output;

  for (i = 0; i < 4; ++i) {
    a1 = ip[0] + ip[2];
    b1 = ip[0] - ip[2];

    temp1 = (ip[1] * sinpi8sqrt2) >> 16;
    temp2 = ip[3] + ((ip[3] * cospi8sqrt2minus1) >> 16);
    c1 = temp1 - temp2;

    temp1 = ip[1] + ((ip[1] * cospi8sqrt2minus1) >> 16);
    temp2 = (ip[3] * sinpi8sqrt2) >> 16;
    d1 = temp1 + temp2;

    op[0] = (a1 + d1 + 4) >> 3;
    op[3] = (a1 - d1 + 4) >> 3;

    op[1] = (b1 + c1 + 4) >> 3;
    op[2] = (b1 - c1 + 4) >> 3;

    ip += shortpitch;
    op += shortpitch;
  }

  ip = output;
  for (r = 0; r < 4; ++r) {
    for (c = 0; c < 4; ++c) {
      int a = ip[c] + pred_ptr[c];

      if (a < 0) a = 0;

      if (a > 255) a = 255;

      dst_ptr[c] = (unsigned char)a;
    }
    ip += 4;
    dst_ptr += dst_stride;
    pred_ptr += pred_stride;
  }
}

void vp8_dc_only_idct_add_c(short input_dc, unsigned char *pred_ptr,
                            int pred_stride, unsigned char *dst_ptr,
                            int dst_stride) {
  int a1 = ((input_dc + 4) >> 3);
  int r, c;

  for (r = 0; r < 4; ++r) {
    for (c = 0; c < 4; ++c) {
      int a = a1 + pred_ptr[c];

      if (a < 0) a = 0;

      if (a > 255) a = 255;

      dst_ptr[c] = (unsigned char)a;
    }

    dst_ptr += dst_stride;
    pred_ptr += pred_stride;
  }
}

void vp8_short_inv_walsh4x4_c(short *input, short *mb_dqcoeff) {
  short output[16];
  int i;
  int a1, b1, c1, d1;
  int a2, b2, c2, d2;
  short *ip = input;
  short *op = output;

  for (i = 0; i < 4; ++i) {
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

  for (i = 0; i < 4; ++i) {
    a1 = ip[0] + ip[3];
    b1 = ip[1] + ip[2];
    c1 = ip[1] - ip[2];
    d1 = ip[0] - ip[3];

    a2 = a1 + b1;
    b2 = c1 + d1;
    c2 = a1 - b1;
    d2 = d1 - c1;

    op[0] = (a2 + 3) >> 3;
    op[1] = (b2 + 3) >> 3;
    op[2] = (c2 + 3) >> 3;
    op[3] = (d2 + 3) >> 3;

    ip += 4;
    op += 4;
  }

  for (i = 0; i < 16; ++i) {
    mb_dqcoeff[i * 16] = output[i];
  }
}

void vp8_short_inv_walsh4x4_1_c(short *input, short *mb_dqcoeff) {
  int i;
  int a1;

  a1 = ((input[0] + 3) >> 3);
  for (i = 0; i < 16; ++i) {
    mb_dqcoeff[i * 16] = a1;
  }
}
