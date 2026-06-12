/*
 *  Copyright (c) 2016 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_PSNR_H_
#define VPX_VPX_DSP_PSNR_H_

#include "vpx_scale/yv12config.h"
#include "vpx/vpx_encoder.h"

#define MAX_PSNR 100.0

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vpx_psnr_pkt PSNR_STATS;

// TODO(dkovalev) change vpx_sse_to_psnr signature: double -> int64_t

/*!\brief Converts SSE to PSNR
 *
 * Converts sum of squared errros (SSE) to peak signal-to-noise ratio (PSNR).
 *
 * \param[in]    samples       Number of samples
 * \param[in]    peak          Max sample value
 * \param[in]    sse           Sum of squared errors
 */
double vpx_sse_to_psnr(double samples, double peak, double sse);
int64_t vpx_get_y_sse(const YV12_BUFFER_CONFIG *a, const YV12_BUFFER_CONFIG *b);
#if CONFIG_VP9_HIGHBITDEPTH
int64_t vpx_highbd_get_y_sse(const YV12_BUFFER_CONFIG *a,
                             const YV12_BUFFER_CONFIG *b);
void vpx_calc_highbd_psnr(const YV12_BUFFER_CONFIG *a,
                          const YV12_BUFFER_CONFIG *b, PSNR_STATS *psnr,
                          unsigned int bit_depth, unsigned int in_bit_depth,
                          int spatial_layer_id);
#endif
void vpx_calc_psnr(const YV12_BUFFER_CONFIG *a, const YV12_BUFFER_CONFIG *b,
                   PSNR_STATS *psnr, int spatial_layer_id);

double vpx_psnrhvs(const YV12_BUFFER_CONFIG *source,
                   const YV12_BUFFER_CONFIG *dest, double *phvs_y,
                   double *phvs_u, double *phvs_v, uint32_t bd, uint32_t in_bd);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // VPX_VPX_DSP_PSNR_H_
