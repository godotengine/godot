/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_INTERNAL_VPX_PSNR_H_
#define VPX_INTERNAL_VPX_PSNR_H_

#ifdef __cplusplus
extern "C" {
#endif

// TODO(dkovalev) change vpx_sse_to_psnr signature: double -> int64_t

/*!\brief Converts SSE to PSNR
 *
 * Converts sum of squared errros (SSE) to peak signal-to-noise ratio (PNSR).
 *
 * \param[in]    samples       Number of samples
 * \param[in]    peak          Max sample value
 * \param[in]    sse           Sum of squared errors
 */
double vpx_sse_to_psnr(double samples, double peak, double sse);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_INTERNAL_VPX_PSNR_H_
