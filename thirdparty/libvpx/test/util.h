/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_TEST_UTIL_H_
#define VPX_TEST_UTIL_H_

#include <stdio.h>
#include <math.h>
#include <tuple>

#include "gtest/gtest.h"
#include "vpx/vpx_image.h"

// Macros
#define GET_PARAM(k) std::get<k>(GetParam())

inline double compute_psnr(const vpx_image_t *img1, const vpx_image_t *img2) {
  assert((img1->fmt == img2->fmt) && (img1->d_w == img2->d_w) &&
         (img1->d_h == img2->d_h));

  const unsigned int width_y = img1->d_w;
  const unsigned int height_y = img1->d_h;
  unsigned int i, j;

  int64_t sqrerr = 0;
  for (i = 0; i < height_y; ++i) {
    for (j = 0; j < width_y; ++j) {
      int64_t d = img1->planes[VPX_PLANE_Y][i * img1->stride[VPX_PLANE_Y] + j] -
                  img2->planes[VPX_PLANE_Y][i * img2->stride[VPX_PLANE_Y] + j];
      sqrerr += d * d;
    }
  }
  double mse = static_cast<double>(sqrerr) / (width_y * height_y);
  double psnr = 100.0;
  if (mse > 0.0) {
    psnr = 10 * log10(255.0 * 255.0 / mse);
  }
  return psnr;
}

#endif  // VPX_TEST_UTIL_H_
