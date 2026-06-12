/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#ifndef VPX_VPX_DSP_X86_CONVOLVE_H_
#define VPX_VPX_DSP_X86_CONVOLVE_H_

#include <assert.h>

#include "./vpx_config.h"
#include "vpx/vpx_integer.h"
#include "vpx_ports/compiler_attributes.h"

// TODO(chiyotsai@google.com): Refactor the code here. Currently this is pretty
// hacky and awful to read. Note that there is a filter_x[3] == 128 check in
// HIGHBD_FUN_CONV_2D to avoid seg fault due to the fact that the c function
// assumes the filter is always 8 tap.
typedef void filter8_1dfunction(const uint8_t *src_ptr, ptrdiff_t src_pitch,
                                uint8_t *output_ptr, ptrdiff_t out_pitch,
                                uint32_t output_height, const int16_t *filter);

// TODO(chiyotsai@google.com): Remove the is_avg argument to the MACROS once we
// have 4-tap vert avg filter.
#define FUN_CONV_1D(name, offset, step_q4, dir, src_start, avg, opt, is_avg) \
  void vpx_convolve8_##name##_##opt(                                         \
      const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst,                \
      ptrdiff_t dst_stride, const InterpKernel *filter, int x0_q4,           \
      int x_step_q4, int y0_q4, int y_step_q4, int w, int h) {               \
    const int16_t *filter_row = filter[offset];                              \
    (void)x0_q4;                                                             \
    (void)x_step_q4;                                                         \
    (void)y0_q4;                                                             \
    (void)y_step_q4;                                                         \
    assert(filter_row[3] != 128);                                            \
    assert(step_q4 == 16);                                                   \
    if (filter_row[0] | filter_row[1] | filter_row[6] | filter_row[7]) {     \
      const int num_taps = 8;                                                \
      while (w >= 16) {                                                      \
        vpx_filter_block1d16_##dir##8_##avg##opt(src_start, src_stride, dst, \
                                                 dst_stride, h, filter_row); \
        src += 16;                                                           \
        dst += 16;                                                           \
        w -= 16;                                                             \
      }                                                                      \
      if (w == 8) {                                                          \
        vpx_filter_block1d8_##dir##8_##avg##opt(src_start, src_stride, dst,  \
                                                dst_stride, h, filter_row);  \
      } else if (w == 4) {                                                   \
        vpx_filter_block1d4_##dir##8_##avg##opt(src_start, src_stride, dst,  \
                                                dst_stride, h, filter_row);  \
      }                                                                      \
      (void)num_taps;                                                        \
    } else if (filter_row[2] | filter_row[5]) {                              \
      const int num_taps = is_avg ? 8 : 4;                                   \
      while (w >= 16) {                                                      \
        vpx_filter_block1d16_##dir##4_##avg##opt(src_start, src_stride, dst, \
                                                 dst_stride, h, filter_row); \
        src += 16;                                                           \
        dst += 16;                                                           \
        w -= 16;                                                             \
      }                                                                      \
      if (w == 8) {                                                          \
        vpx_filter_block1d8_##dir##4_##avg##opt(src_start, src_stride, dst,  \
                                                dst_stride, h, filter_row);  \
      } else if (w == 4) {                                                   \
        vpx_filter_block1d4_##dir##4_##avg##opt(src_start, src_stride, dst,  \
                                                dst_stride, h, filter_row);  \
      }                                                                      \
      (void)num_taps;                                                        \
    } else {                                                                 \
      const int num_taps = 2;                                                \
      while (w >= 16) {                                                      \
        vpx_filter_block1d16_##dir##2_##avg##opt(src_start, src_stride, dst, \
                                                 dst_stride, h, filter_row); \
        src += 16;                                                           \
        dst += 16;                                                           \
        w -= 16;                                                             \
      }                                                                      \
      if (w == 8) {                                                          \
        vpx_filter_block1d8_##dir##2_##avg##opt(src_start, src_stride, dst,  \
                                                dst_stride, h, filter_row);  \
      } else if (w == 4) {                                                   \
        vpx_filter_block1d4_##dir##2_##avg##opt(src_start, src_stride, dst,  \
                                                dst_stride, h, filter_row);  \
      }                                                                      \
      (void)num_taps;                                                        \
    }                                                                        \
  }

#define FUN_CONV_2D(avg, opt, is_avg)                                          \
  void vpx_convolve8_##avg##opt(                                               \
      const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst,                  \
      ptrdiff_t dst_stride, const InterpKernel *filter, int x0_q4,             \
      int x_step_q4, int y0_q4, int y_step_q4, int w, int h) {                 \
    const int16_t *filter_x = filter[x0_q4];                                   \
    const int16_t *filter_y = filter[y0_q4];                                   \
    (void)filter_y;                                                            \
    assert(filter_x[3] != 128);                                                \
    assert(filter_y[3] != 128);                                                \
    assert(w <= 64);                                                           \
    assert(h <= 64);                                                           \
    assert(x_step_q4 == 16);                                                   \
    assert(y_step_q4 == 16);                                                   \
    if (filter_x[0] | filter_x[1] | filter_x[6] | filter_x[7]) {               \
      DECLARE_ALIGNED(16, uint8_t, fdata2[64 * 71] VPX_UNINITIALIZED);         \
      vpx_convolve8_horiz_##opt(src - 3 * src_stride, src_stride, fdata2, 64,  \
                                filter, x0_q4, x_step_q4, y0_q4, y_step_q4, w, \
                                h + 7);                                        \
      vpx_convolve8_##avg##vert_##opt(fdata2 + 3 * 64, 64, dst, dst_stride,    \
                                      filter, x0_q4, x_step_q4, y0_q4,         \
                                      y_step_q4, w, h);                        \
    } else if (filter_x[2] | filter_x[5]) {                                    \
      const int num_taps = is_avg ? 8 : 4;                                     \
      DECLARE_ALIGNED(16, uint8_t, fdata2[64 * 71] VPX_UNINITIALIZED);         \
      vpx_convolve8_horiz_##opt(                                               \
          src - (num_taps / 2 - 1) * src_stride, src_stride, fdata2, 64,       \
          filter, x0_q4, x_step_q4, y0_q4, y_step_q4, w, h + num_taps - 1);    \
      vpx_convolve8_##avg##vert_##opt(fdata2 + 64 * (num_taps / 2 - 1), 64,    \
                                      dst, dst_stride, filter, x0_q4,          \
                                      x_step_q4, y0_q4, y_step_q4, w, h);      \
    } else {                                                                   \
      DECLARE_ALIGNED(16, uint8_t, fdata2[64 * 65] VPX_UNINITIALIZED);         \
      vpx_convolve8_horiz_##opt(src, src_stride, fdata2, 64, filter, x0_q4,    \
                                x_step_q4, y0_q4, y_step_q4, w, h + 1);        \
      vpx_convolve8_##avg##vert_##opt(fdata2, 64, dst, dst_stride, filter,     \
                                      x0_q4, x_step_q4, y0_q4, y_step_q4, w,   \
                                      h);                                      \
    }                                                                          \
  }

#if CONFIG_VP9_HIGHBITDEPTH

typedef void highbd_filter8_1dfunction(const uint16_t *src_ptr,
                                       const ptrdiff_t src_pitch,
                                       uint16_t *output_ptr,
                                       ptrdiff_t out_pitch,
                                       unsigned int output_height,
                                       const int16_t *filter, int bd);

#define HIGH_FUN_CONV_1D(name, offset, step_q4, dir, src_start, avg, opt,     \
                         is_avg)                                              \
  void vpx_highbd_convolve8_##name##_##opt(                                   \
      const uint16_t *src, ptrdiff_t src_stride, uint16_t *dst,               \
      ptrdiff_t dst_stride, const InterpKernel *filter_kernel, int x0_q4,     \
      int x_step_q4, int y0_q4, int y_step_q4, int w, int h, int bd) {        \
    const int16_t *filter_row = filter_kernel[offset];                        \
    if (step_q4 == 16 && filter_row[3] != 128) {                              \
      if (filter_row[0] | filter_row[1] | filter_row[6] | filter_row[7]) {    \
        const int num_taps = 8;                                               \
        while (w >= 16) {                                                     \
          vpx_highbd_filter_block1d16_##dir##8_##avg##opt(                    \
              src_start, src_stride, dst, dst_stride, h, filter_row, bd);     \
          src += 16;                                                          \
          dst += 16;                                                          \
          w -= 16;                                                            \
        }                                                                     \
        while (w >= 8) {                                                      \
          vpx_highbd_filter_block1d8_##dir##8_##avg##opt(                     \
              src_start, src_stride, dst, dst_stride, h, filter_row, bd);     \
          src += 8;                                                           \
          dst += 8;                                                           \
          w -= 8;                                                             \
        }                                                                     \
        while (w >= 4) {                                                      \
          vpx_highbd_filter_block1d4_##dir##8_##avg##opt(                     \
              src_start, src_stride, dst, dst_stride, h, filter_row, bd);     \
          src += 4;                                                           \
          dst += 4;                                                           \
          w -= 4;                                                             \
        }                                                                     \
        (void)num_taps;                                                       \
      } else if (filter_row[2] | filter_row[5]) {                             \
        const int num_taps = is_avg ? 8 : 4;                                  \
        while (w >= 16) {                                                     \
          vpx_highbd_filter_block1d16_##dir##4_##avg##opt(                    \
              src_start, src_stride, dst, dst_stride, h, filter_row, bd);     \
          src += 16;                                                          \
          dst += 16;                                                          \
          w -= 16;                                                            \
        }                                                                     \
        while (w >= 8) {                                                      \
          vpx_highbd_filter_block1d8_##dir##4_##avg##opt(                     \
              src_start, src_stride, dst, dst_stride, h, filter_row, bd);     \
          src += 8;                                                           \
          dst += 8;                                                           \
          w -= 8;                                                             \
        }                                                                     \
        while (w >= 4) {                                                      \
          vpx_highbd_filter_block1d4_##dir##4_##avg##opt(                     \
              src_start, src_stride, dst, dst_stride, h, filter_row, bd);     \
          src += 4;                                                           \
          dst += 4;                                                           \
          w -= 4;                                                             \
        }                                                                     \
        (void)num_taps;                                                       \
      } else {                                                                \
        const int num_taps = 2;                                               \
        while (w >= 16) {                                                     \
          vpx_highbd_filter_block1d16_##dir##2_##avg##opt(                    \
              src_start, src_stride, dst, dst_stride, h, filter_row, bd);     \
          src += 16;                                                          \
          dst += 16;                                                          \
          w -= 16;                                                            \
        }                                                                     \
        while (w >= 8) {                                                      \
          vpx_highbd_filter_block1d8_##dir##2_##avg##opt(                     \
              src_start, src_stride, dst, dst_stride, h, filter_row, bd);     \
          src += 8;                                                           \
          dst += 8;                                                           \
          w -= 8;                                                             \
        }                                                                     \
        while (w >= 4) {                                                      \
          vpx_highbd_filter_block1d4_##dir##2_##avg##opt(                     \
              src_start, src_stride, dst, dst_stride, h, filter_row, bd);     \
          src += 4;                                                           \
          dst += 4;                                                           \
          w -= 4;                                                             \
        }                                                                     \
        (void)num_taps;                                                       \
      }                                                                       \
    }                                                                         \
    if (w) {                                                                  \
      vpx_highbd_convolve8_##name##_c(src, src_stride, dst, dst_stride,       \
                                      filter_kernel, x0_q4, x_step_q4, y0_q4, \
                                      y_step_q4, w, h, bd);                   \
    }                                                                         \
  }

#define HIGH_FUN_CONV_2D(avg, opt, is_avg)                                     \
  void vpx_highbd_convolve8_##avg##opt(                                        \
      const uint16_t *src, ptrdiff_t src_stride, uint16_t *dst,                \
      ptrdiff_t dst_stride, const InterpKernel *filter, int x0_q4,             \
      int x_step_q4, int y0_q4, int y_step_q4, int w, int h, int bd) {         \
    const int16_t *filter_x = filter[x0_q4];                                   \
    assert(w <= 64);                                                           \
    assert(h <= 64);                                                           \
    if (x_step_q4 == 16 && y_step_q4 == 16) {                                  \
      if ((filter_x[0] | filter_x[1] | filter_x[6] | filter_x[7]) ||           \
          filter_x[3] == 128) {                                                \
        DECLARE_ALIGNED(16, uint16_t, fdata2[64 * 71] VPX_UNINITIALIZED);      \
        vpx_highbd_convolve8_horiz_##opt(src - 3 * src_stride, src_stride,     \
                                         fdata2, 64, filter, x0_q4, x_step_q4, \
                                         y0_q4, y_step_q4, w, h + 7, bd);      \
        vpx_highbd_convolve8_##avg##vert_##opt(                                \
            fdata2 + 192, 64, dst, dst_stride, filter, x0_q4, x_step_q4,       \
            y0_q4, y_step_q4, w, h, bd);                                       \
      } else if (filter_x[2] | filter_x[5]) {                                  \
        const int num_taps = is_avg ? 8 : 4;                                   \
        DECLARE_ALIGNED(16, uint16_t, fdata2[64 * 71] VPX_UNINITIALIZED);      \
        vpx_highbd_convolve8_horiz_##opt(                                      \
            src - (num_taps / 2 - 1) * src_stride, src_stride, fdata2, 64,     \
            filter, x0_q4, x_step_q4, y0_q4, y_step_q4, w, h + num_taps - 1,   \
            bd);                                                               \
        vpx_highbd_convolve8_##avg##vert_##opt(                                \
            fdata2 + 64 * (num_taps / 2 - 1), 64, dst, dst_stride, filter,     \
            x0_q4, x_step_q4, y0_q4, y_step_q4, w, h, bd);                     \
      } else {                                                                 \
        DECLARE_ALIGNED(16, uint16_t, fdata2[64 * 65] VPX_UNINITIALIZED);      \
        vpx_highbd_convolve8_horiz_##opt(src, src_stride, fdata2, 64, filter,  \
                                         x0_q4, x_step_q4, y0_q4, y_step_q4,   \
                                         w, h + 1, bd);                        \
        vpx_highbd_convolve8_##avg##vert_##opt(fdata2, 64, dst, dst_stride,    \
                                               filter, x0_q4, x_step_q4,       \
                                               y0_q4, y_step_q4, w, h, bd);    \
      }                                                                        \
    } else {                                                                   \
      vpx_highbd_convolve8_##avg##c(src, src_stride, dst, dst_stride, filter,  \
                                    x0_q4, x_step_q4, y0_q4, y_step_q4, w, h,  \
                                    bd);                                       \
    }                                                                          \
  }

#endif  // CONFIG_VP9_HIGHBITDEPTH
#endif  // VPX_VPX_DSP_X86_CONVOLVE_H_
