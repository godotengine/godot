/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <math.h>

#include "./vp8_rtcd.h"

void vp8_short_fdct4x4_c(short *input, short *output, int pitch) {
  int i;
  int a1, b1, c1, d1;
  short *ip = input;
  short *op = output;

  for (i = 0; i < 4; ++i) {
    a1 = ((ip[0] + ip[3]) * 8);
    b1 = ((ip[1] + ip[2]) * 8);
    c1 = ((ip[1] - ip[2]) * 8);
    d1 = ((ip[0] - ip[3]) * 8);

    op[0] = a1 + b1;
    op[2] = a1 - b1;

    op[1] = (c1 * 2217 + d1 * 5352 + 14500) >> 12;
    op[3] = (d1 * 2217 - c1 * 5352 + 7500) >> 12;

    ip += pitch / 2;
    op += 4;
  }
  ip = output;
  op = output;
  for (i = 0; i < 4; ++i) {
    a1 = ip[0] + ip[12];
    b1 = ip[4] + ip[8];
    c1 = ip[4] - ip[8];
    d1 = ip[0] - ip[12];

    op[0] = (a1 + b1 + 7) >> 4;
    op[8] = (a1 - b1 + 7) >> 4;

    op[4] = ((c1 * 2217 + d1 * 5352 + 12000) >> 16) + (d1 != 0);
    op[12] = (d1 * 2217 - c1 * 5352 + 51000) >> 16;

    ip++;
    op++;
  }
}

void vp8_short_fdct8x4_c(short *input, short *output, int pitch) {
  vp8_short_fdct4x4_c(input, output, pitch);
  vp8_short_fdct4x4_c(input + 4, output + 16, pitch);
}

void vp8_short_walsh4x4_c(short *input, short *output, int pitch) {
  int i;
  int a1, b1, c1, d1;
  int a2, b2, c2, d2;
  short *ip = input;
  short *op = output;

  for (i = 0; i < 4; ++i) {
    a1 = ((ip[0] + ip[2]) * 4);
    d1 = ((ip[1] + ip[3]) * 4);
    c1 = ((ip[1] - ip[3]) * 4);
    b1 = ((ip[0] - ip[2]) * 4);

    op[0] = a1 + d1 + (a1 != 0);
    op[1] = b1 + c1;
    op[2] = b1 - c1;
    op[3] = a1 - d1;
    ip += pitch / 2;
    op += 4;
  }

  ip = output;
  op = output;

  for (i = 0; i < 4; ++i) {
    a1 = ip[0] + ip[8];
    d1 = ip[4] + ip[12];
    c1 = ip[4] - ip[12];
    b1 = ip[0] - ip[8];

    a2 = a1 + d1;
    b2 = b1 + c1;
    c2 = b1 - c1;
    d2 = a1 - d1;

    a2 += a2 < 0;
    b2 += b2 < 0;
    c2 += c2 < 0;
    d2 += d2 < 0;

    op[0] = (a2 + 3) >> 3;
    op[4] = (b2 + 3) >> 3;
    op[8] = (c2 + 3) >> 3;
    op[12] = (d2 + 3) >> 3;

    ip++;
    op++;
  }
}
