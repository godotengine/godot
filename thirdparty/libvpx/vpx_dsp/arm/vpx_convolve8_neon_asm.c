/*
 *  Copyright (c) 2018 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vpx_dsp_rtcd.h"
#include "vp9/common/vp9_filter.h"
#include "vpx_dsp/arm/vpx_convolve8_neon_asm.h"

/* Type1 and Type2 functions are called depending on the position of the
 * negative and positive coefficients in the filter. In type1, the filter kernel
 * used is sub_pel_filters_8lp, in which only the first two and the last two
 * coefficients are negative. In type2, the negative coefficients are 0, 2, 5 &
 * 7.
 */

#define DEFINE_FILTER(dir)                                                   \
  void vpx_convolve8_##dir##_neon(                                           \
      const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst,                \
      ptrdiff_t dst_stride, const InterpKernel *filter, int x0_q4,           \
      int x_step_q4, int y0_q4, int y_step_q4, int w, int h) {               \
    if (filter == vp9_filter_kernels[1]) {                                   \
      vpx_convolve8_##dir##_filter_type1_neon(                               \
          src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4, y0_q4, \
          y_step_q4, w, h);                                                  \
    } else {                                                                 \
      vpx_convolve8_##dir##_filter_type2_neon(                               \
          src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4, y0_q4, \
          y_step_q4, w, h);                                                  \
    }                                                                        \
  }

DEFINE_FILTER(horiz)
DEFINE_FILTER(avg_horiz)
DEFINE_FILTER(vert)
DEFINE_FILTER(avg_vert)
