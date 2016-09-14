/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#ifndef VPX_DSP_X86_CONVOLVE_H_
#define VPX_DSP_X86_CONVOLVE_H_

#include <assert.h>

#include "./vpx_config.h"
#include "vpx/vpx_integer.h"
#include "vpx_ports/mem.h"

typedef void filter8_1dfunction (
  const uint8_t *src_ptr,
  ptrdiff_t src_pitch,
  uint8_t *output_ptr,
  ptrdiff_t out_pitch,
  uint32_t output_height,
  const int16_t *filter
);

#define FUN_CONV_1D(name, step_q4, filter, dir, src_start, avg, opt) \
  void vpx_convolve8_##name##_##opt(const uint8_t *src, ptrdiff_t src_stride, \
                                    uint8_t *dst, ptrdiff_t dst_stride, \
                                    const int16_t *filter_x, int x_step_q4, \
                                    const int16_t *filter_y, int y_step_q4, \
                                    int w, int h) { \
  assert(filter[3] != 128); \
  assert(step_q4 == 16); \
  if (filter[0] | filter[1] | filter[2]) { \
    while (w >= 16) { \
      vpx_filter_block1d16_##dir##8_##avg##opt(src_start, \
                                               src_stride, \
                                               dst, \
                                               dst_stride, \
                                               h, \
                                               filter); \
      src += 16; \
      dst += 16; \
      w -= 16; \
    } \
    if (w == 8) { \
      vpx_filter_block1d8_##dir##8_##avg##opt(src_start, \
                                              src_stride, \
                                              dst, \
                                              dst_stride, \
                                              h, \
                                              filter); \
    } else if (w == 4) { \
      vpx_filter_block1d4_##dir##8_##avg##opt(src_start, \
                                              src_stride, \
                                              dst, \
                                              dst_stride, \
                                              h, \
                                              filter); \
    } \
  } else { \
    while (w >= 16) { \
      vpx_filter_block1d16_##dir##2_##avg##opt(src, \
                                               src_stride, \
                                               dst, \
                                               dst_stride, \
                                               h, \
                                               filter); \
      src += 16; \
      dst += 16; \
      w -= 16; \
    } \
    if (w == 8) { \
      vpx_filter_block1d8_##dir##2_##avg##opt(src, \
                                              src_stride, \
                                              dst, \
                                              dst_stride, \
                                              h, \
                                              filter); \
    } else if (w == 4) { \
      vpx_filter_block1d4_##dir##2_##avg##opt(src, \
                                              src_stride, \
                                              dst, \
                                              dst_stride, \
                                              h, \
                                              filter); \
    } \
  } \
}

#define FUN_CONV_2D(avg, opt) \
void vpx_convolve8_##avg##opt(const uint8_t *src, ptrdiff_t src_stride, \
                              uint8_t *dst, ptrdiff_t dst_stride, \
                              const int16_t *filter_x, int x_step_q4, \
                              const int16_t *filter_y, int y_step_q4, \
                              int w, int h) { \
  assert(filter_x[3] != 128); \
  assert(filter_y[3] != 128); \
  assert(w <= 64); \
  assert(h <= 64); \
  assert(x_step_q4 == 16); \
  assert(y_step_q4 == 16); \
  if (filter_x[0] | filter_x[1] | filter_x[2]) { \
    DECLARE_ALIGNED(16, uint8_t, fdata2[64 * 71]); \
    vpx_convolve8_horiz_##opt(src - 3 * src_stride, src_stride, fdata2, 64, \
                              filter_x, x_step_q4, filter_y, y_step_q4, \
                              w, h + 7); \
    vpx_convolve8_##avg##vert_##opt(fdata2 + 3 * 64, 64, dst, dst_stride, \
                                    filter_x, x_step_q4, filter_y, \
                                    y_step_q4, w, h); \
  } else { \
    DECLARE_ALIGNED(16, uint8_t, fdata2[64 * 65]); \
    vpx_convolve8_horiz_##opt(src, src_stride, fdata2, 64, \
                              filter_x, x_step_q4, filter_y, y_step_q4, \
                              w, h + 1); \
    vpx_convolve8_##avg##vert_##opt(fdata2, 64, dst, dst_stride, \
                                    filter_x, x_step_q4, filter_y, \
                                    y_step_q4, w, h); \
  } \
}

#if CONFIG_VP9_HIGHBITDEPTH

typedef void highbd_filter8_1dfunction (
  const uint16_t *src_ptr,
  const ptrdiff_t src_pitch,
  uint16_t *output_ptr,
  ptrdiff_t out_pitch,
  unsigned int output_height,
  const int16_t *filter,
  int bd
);

#define HIGH_FUN_CONV_1D(name, step_q4, filter, dir, src_start, avg, opt) \
  void vpx_highbd_convolve8_##name##_##opt(const uint8_t *src8, \
                                           ptrdiff_t src_stride, \
                                           uint8_t *dst8, \
                                           ptrdiff_t dst_stride, \
                                           const int16_t *filter_x, \
                                           int x_step_q4, \
                                           const int16_t *filter_y, \
                                           int y_step_q4, \
                                           int w, int h, int bd) { \
  if (step_q4 == 16 && filter[3] != 128) { \
    uint16_t *src = CONVERT_TO_SHORTPTR(src8); \
    uint16_t *dst = CONVERT_TO_SHORTPTR(dst8); \
    if (filter[0] | filter[1] | filter[2]) { \
      while (w >= 16) { \
        vpx_highbd_filter_block1d16_##dir##8_##avg##opt(src_start, \
                                                        src_stride, \
                                                        dst, \
                                                        dst_stride, \
                                                        h, \
                                                        filter, \
                                                        bd); \
        src += 16; \
        dst += 16; \
        w -= 16; \
      } \
      while (w >= 8) { \
        vpx_highbd_filter_block1d8_##dir##8_##avg##opt(src_start, \
                                                       src_stride, \
                                                       dst, \
                                                       dst_stride, \
                                                       h, \
                                                       filter, \
                                                       bd); \
        src += 8; \
        dst += 8; \
        w -= 8; \
      } \
      while (w >= 4) { \
        vpx_highbd_filter_block1d4_##dir##8_##avg##opt(src_start, \
                                                       src_stride, \
                                                       dst, \
                                                       dst_stride, \
                                                       h, \
                                                       filter, \
                                                       bd); \
        src += 4; \
        dst += 4; \
        w -= 4; \
      } \
    } else { \
      while (w >= 16) { \
        vpx_highbd_filter_block1d16_##dir##2_##avg##opt(src, \
                                                        src_stride, \
                                                        dst, \
                                                        dst_stride, \
                                                        h, \
                                                        filter, \
                                                        bd); \
        src += 16; \
        dst += 16; \
        w -= 16; \
      } \
      while (w >= 8) { \
        vpx_highbd_filter_block1d8_##dir##2_##avg##opt(src, \
                                                       src_stride, \
                                                       dst, \
                                                       dst_stride, \
                                                       h, \
                                                       filter, \
                                                       bd); \
        src += 8; \
        dst += 8; \
        w -= 8; \
      } \
      while (w >= 4) { \
        vpx_highbd_filter_block1d4_##dir##2_##avg##opt(src, \
                                                       src_stride, \
                                                       dst, \
                                                       dst_stride, \
                                                       h, \
                                                       filter, \
                                                       bd); \
        src += 4; \
        dst += 4; \
        w -= 4; \
      } \
    } \
  } \
  if (w) { \
    vpx_highbd_convolve8_##name##_c(src8, src_stride, dst8, dst_stride, \
                                    filter_x, x_step_q4, filter_y, y_step_q4, \
                                    w, h, bd); \
  } \
}

#define HIGH_FUN_CONV_2D(avg, opt) \
void vpx_highbd_convolve8_##avg##opt(const uint8_t *src, ptrdiff_t src_stride, \
                                     uint8_t *dst, ptrdiff_t dst_stride, \
                                     const int16_t *filter_x, int x_step_q4, \
                                     const int16_t *filter_y, int y_step_q4, \
                                     int w, int h, int bd) { \
  assert(w <= 64); \
  assert(h <= 64); \
  if (x_step_q4 == 16 && y_step_q4 == 16) { \
    if ((filter_x[0] | filter_x[1] | filter_x[2]) || filter_x[3] == 128) { \
      DECLARE_ALIGNED(16, uint16_t, fdata2[64 * 71]); \
      vpx_highbd_convolve8_horiz_##opt(src - 3 * src_stride, src_stride, \
                                       CONVERT_TO_BYTEPTR(fdata2), 64, \
                                       filter_x, x_step_q4, \
                                       filter_y, y_step_q4, \
                                       w, h + 7, bd); \
      vpx_highbd_convolve8_##avg##vert_##opt(CONVERT_TO_BYTEPTR(fdata2) + 192, \
                                             64, dst, dst_stride, \
                                             filter_x, x_step_q4, \
                                             filter_y, y_step_q4, \
                                             w, h, bd); \
    } else { \
      DECLARE_ALIGNED(16, uint16_t, fdata2[64 * 65]); \
      vpx_highbd_convolve8_horiz_##opt(src, src_stride, \
                                       CONVERT_TO_BYTEPTR(fdata2), 64, \
                                       filter_x, x_step_q4, \
                                       filter_y, y_step_q4, \
                                       w, h + 1, bd); \
      vpx_highbd_convolve8_##avg##vert_##opt(CONVERT_TO_BYTEPTR(fdata2), 64, \
                                             dst, dst_stride, \
                                             filter_x, x_step_q4, \
                                             filter_y, y_step_q4, \
                                             w, h, bd); \
    } \
  } else { \
    vpx_highbd_convolve8_##avg##c(src, src_stride, dst, dst_stride, \
                                  filter_x, x_step_q4, filter_y, y_step_q4, w, \
                                  h, bd); \
  } \
}
#endif  // CONFIG_VP9_HIGHBITDEPTH

#endif  // VPX_DSP_X86_CONVOLVE_H_
