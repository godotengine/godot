/*
 *  Copyright (c) 2016 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vpx_dsp_rtcd.h"

uint64_t vpx_sum_squares_2d_i16_c(const int16_t *src, int stride, int size) {
  int r, c;
  uint64_t ss = 0;

  for (r = 0; r < size; r++) {
    for (c = 0; c < size; c++) {
      const int16_t v = src[c];
      ss += v * v;
    }
    src += stride;
  }

  return ss;
}
