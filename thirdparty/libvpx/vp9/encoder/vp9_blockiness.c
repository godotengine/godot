/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#include <stdlib.h>

#include "vpx/vpx_integer.h"
#include "vpx_ports/system_state.h"
#include "vp9/encoder/vp9_blockiness.h"

static int horizontal_filter(const uint8_t *s) {
  return (s[1] - s[-2]) * 2 + (s[-1] - s[0]) * 6;
}

static int vertical_filter(const uint8_t *s, int p) {
  return (s[p] - s[-2 * p]) * 2 + (s[-p] - s[0]) * 6;
}

static int variance(int sum, int sum_squared, int size) {
  return sum_squared / size - (sum / size) * (sum / size);
}
// Calculate a blockiness level for a vertical block edge.
// This function returns a new blockiness metric that's defined as

//              p0 p1 p2 p3
//              q0 q1 q2 q3
// block edge ->
//              r0 r1 r2 r3
//              s0 s1 s2 s3

// blockiness =  p0*-2+q0*6+r0*-6+s0*2 +
//               p1*-2+q1*6+r1*-6+s1*2 +
//               p2*-2+q2*6+r2*-6+s2*2 +
//               p3*-2+q3*6+r3*-6+s3*2 ;

// reconstructed_blockiness = abs(blockiness from reconstructed buffer -
//                                blockiness from source buffer,0)
//
// I make the assumption that flat blocks are much more visible than high
// contrast blocks. As such, I scale the result of the blockiness calc
// by dividing the blockiness by the variance of the pixels on either side
// of the edge as follows:
// var_0 = (q0^2+q1^2+q2^2+q3^2) - ((q0 + q1 + q2 + q3) / 4 )^2
// var_1 = (r0^2+r1^2+r2^2+r3^2) - ((r0 + r1 + r2 + r3) / 4 )^2
// The returned blockiness is the scaled value
// Reconstructed blockiness / ( 1 + var_0 + var_1 ) ;
static int blockiness_vertical(const uint8_t *s, int sp, const uint8_t *r,
                               int rp, int size) {
  int s_blockiness = 0;
  int r_blockiness = 0;
  int sum_0 = 0;
  int sum_sq_0 = 0;
  int sum_1 = 0;
  int sum_sq_1 = 0;
  int i;
  int var_0;
  int var_1;
  for (i = 0; i < size; ++i, s += sp, r += rp) {
    s_blockiness += horizontal_filter(s);
    r_blockiness += horizontal_filter(r);
    sum_0 += s[0];
    sum_sq_0 += s[0] * s[0];
    sum_1 += s[-1];
    sum_sq_1 += s[-1] * s[-1];
  }
  var_0 = variance(sum_0, sum_sq_0, size);
  var_1 = variance(sum_1, sum_sq_1, size);
  r_blockiness = abs(r_blockiness);
  s_blockiness = abs(s_blockiness);

  if (r_blockiness > s_blockiness)
    return (r_blockiness - s_blockiness) / (1 + var_0 + var_1);
  else
    return 0;
}

// Calculate a blockiness level for a horizontal block edge
// same as above.
static int blockiness_horizontal(const uint8_t *s, int sp, const uint8_t *r,
                                 int rp, int size) {
  int s_blockiness = 0;
  int r_blockiness = 0;
  int sum_0 = 0;
  int sum_sq_0 = 0;
  int sum_1 = 0;
  int sum_sq_1 = 0;
  int i;
  int var_0;
  int var_1;
  for (i = 0; i < size; ++i, ++s, ++r) {
    s_blockiness += vertical_filter(s, sp);
    r_blockiness += vertical_filter(r, rp);
    sum_0 += s[0];
    sum_sq_0 += s[0] * s[0];
    sum_1 += s[-sp];
    sum_sq_1 += s[-sp] * s[-sp];
  }
  var_0 = variance(sum_0, sum_sq_0, size);
  var_1 = variance(sum_1, sum_sq_1, size);
  r_blockiness = abs(r_blockiness);
  s_blockiness = abs(s_blockiness);

  if (r_blockiness > s_blockiness)
    return (r_blockiness - s_blockiness) / (1 + var_0 + var_1);
  else
    return 0;
}

// This function returns the blockiness for the entire frame currently by
// looking at all borders in steps of 4.
double vp9_get_blockiness(const uint8_t *img1, int img1_pitch,
                          const uint8_t *img2, int img2_pitch, int width,
                          int height) {
  double blockiness = 0;
  int i, j;
  vpx_clear_system_state();
  for (i = 0; i < height;
       i += 4, img1 += img1_pitch * 4, img2 += img2_pitch * 4) {
    for (j = 0; j < width; j += 4) {
      if (i > 0 && i < height && j > 0 && j < width) {
        blockiness +=
            blockiness_vertical(img1 + j, img1_pitch, img2 + j, img2_pitch, 4);
        blockiness += blockiness_horizontal(img1 + j, img1_pitch, img2 + j,
                                            img2_pitch, 4);
      }
    }
  }
  blockiness /= width * height / 16;
  return blockiness;
}
