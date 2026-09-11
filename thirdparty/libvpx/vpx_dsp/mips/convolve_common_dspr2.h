/*
 *  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_MIPS_CONVOLVE_COMMON_DSPR2_H_
#define VPX_VPX_DSP_MIPS_CONVOLVE_COMMON_DSPR2_H_

#include <assert.h>

#include "./vpx_config.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/mips/common_dspr2.h"

#ifdef __cplusplus
extern "C" {
#endif

#if HAVE_DSPR2
void vpx_convolve2_horiz_dspr2(const uint8_t *src, ptrdiff_t src_stride,
                               uint8_t *dst, ptrdiff_t dst_stride,
                               const InterpKernel *filter, int x0_q4,
                               int32_t x_step_q4, int y0_q4, int y_step_q4,
                               int w, int h);

void vpx_convolve2_avg_horiz_dspr2(const uint8_t *src, ptrdiff_t src_stride,
                                   uint8_t *dst, ptrdiff_t dst_stride,
                                   const InterpKernel *filter, int x0_q4,
                                   int32_t x_step_q4, int y0_q4, int y_step_q4,
                                   int w, int h);

void vpx_convolve2_avg_vert_dspr2(const uint8_t *src, ptrdiff_t src_stride,
                                  uint8_t *dst, ptrdiff_t dst_stride,
                                  const InterpKernel *filter, int x0_q4,
                                  int32_t x_step_q4, int y0_q4, int y_step_q4,
                                  int w, int h);

void vpx_convolve2_dspr2(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst,
                         ptrdiff_t dst_stride, const int16_t *filter, int w,
                         int h);

void vpx_convolve2_vert_dspr2(const uint8_t *src, ptrdiff_t src_stride,
                              uint8_t *dst, ptrdiff_t dst_stride,
                              const InterpKernel *filter, int x0_q4,
                              int32_t x_step_q4, int y0_q4, int y_step_q4,
                              int w, int h);

#endif  // #if HAVE_DSPR2
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VPX_DSP_MIPS_CONVOLVE_COMMON_DSPR2_H_
