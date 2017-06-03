/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <math.h>

#include "vpx/internal/vpx_psnr.h"

#define MAX_PSNR 100.0

double vpx_sse_to_psnr(double samples, double peak, double sse) {
  if (sse > 0.0) {
    const double psnr = 10.0 * log10(samples * peak * peak / sse);
    return psnr > MAX_PSNR ? MAX_PSNR : psnr;
  } else {
    return MAX_PSNR;
  }
}
