/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vp9_rtcd.h"
#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "./vpx_scale_rtcd.h"
#include "vp9/common/vp9_blockd.h"
#include "vp9/encoder/vp9_encoder.h"
#include "vpx/vpx_codec.h"
#include "vpx_dsp/vpx_filter.h"
#include "vpx_scale/yv12config.h"

void vp9_scale_and_extend_frame_c(const YV12_BUFFER_CONFIG *src,
                                  YV12_BUFFER_CONFIG *dst,
                                  INTERP_FILTER filter_type, int phase_scaler) {
  const int src_w = src->y_crop_width;
  const int src_h = src->y_crop_height;
  const uint8_t *const srcs[3] = { src->y_buffer, src->u_buffer,
                                   src->v_buffer };
  const int src_strides[3] = { src->y_stride, src->uv_stride, src->uv_stride };
  uint8_t *const dsts[3] = { dst->y_buffer, dst->u_buffer, dst->v_buffer };
  const int dst_strides[3] = { dst->y_stride, dst->uv_stride, dst->uv_stride };
  const InterpKernel *const kernel = vp9_filter_kernels[filter_type];
  int x, y, i;

#if HAVE_SSSE3 || HAVE_NEON
  // TODO(linfengz): The 4:3 specialized C code is disabled by default since
  // it's much slower than the general version which calls vpx_scaled_2d() even
  // if vpx_scaled_2d() is not optimized. It will only be enabled as a reference
  // for the platforms which have faster optimization.
  if (4 * dst->y_crop_width == 3 * src_w &&
      4 * dst->y_crop_height == 3 * src_h) {
    // Specialize 4 to 3 scaling.
    // Example pixel locations.
    // (O: Original pixel. S: Scaled pixel. X: Overlapped pixel.)
    //      phase_scaler = 0               |      phase_scaler = 8
    //                                     |
    //      X     O S   O   S O     X      |      O     O     O     O     O
    //                                     |
    //                                     |
    //                                     |         S       S       S
    //                                     |
    //                                     |
    //      O     O     O     O     O      |      O     O     O     O     O
    //                                     |
    //      S       S       S       S      |
    //                                     |
    //                                     |
    //                                     |         S       S       S
    //      O     O     O     O     O      |      O     O     O     O     O
    //                                     |
    //                                     |
    //                                     |
    //      S       S       S       S      |
    //                                     |
    //      O     O     O     O     O      |      O     O     O     O     O
    //                                     |         S       S       S
    //                                     |
    //                                     |
    //                                     |
    //                                     |
    //      X     O S   O   S O     X      |      O     O     O     O     O

    const int dst_ws[3] = { dst->y_crop_width, dst->uv_crop_width,
                            dst->uv_crop_width };
    const int dst_hs[3] = { dst->y_crop_height, dst->uv_crop_height,
                            dst->uv_crop_height };
    for (i = 0; i < MAX_MB_PLANE; ++i) {
      const int dst_w = dst_ws[i];
      const int dst_h = dst_hs[i];
      const int src_stride = src_strides[i];
      const int dst_stride = dst_strides[i];
      for (y = 0; y < dst_h; y += 3) {
        for (x = 0; x < dst_w; x += 3) {
          const uint8_t *src_ptr = srcs[i] + 4 * y / 3 * src_stride + 4 * x / 3;
          uint8_t *dst_ptr = dsts[i] + y * dst_stride + x;

          // Must call c function because its optimization doesn't support 3x3.
          vpx_scaled_2d_c(src_ptr, src_stride, dst_ptr, dst_stride, kernel,
                          phase_scaler, 64 / 3, phase_scaler, 64 / 3, 3, 3);
        }
      }
    }
  } else
#endif
  {
    const int dst_w = dst->y_crop_width;
    const int dst_h = dst->y_crop_height;

    // The issue b/311394513 reveals a corner case bug. vpx_scaled_2d() requires
    // both x_step_q4 and y_step_q4 are less than or equal to 64. Otherwise, it
    // needs to call vp9_scale_and_extend_frame_nonnormative() that supports
    // arbitrary scaling.
    const int x_step_q4 = 16 * src_w / dst_w;
    const int y_step_q4 = 16 * src_h / dst_h;
    if (x_step_q4 > 64 || y_step_q4 > 64) {
      // This function is only called while cm->bit_depth is VPX_BITS_8.
#if CONFIG_VP9_HIGHBITDEPTH
      vp9_scale_and_extend_frame_nonnormative(src, dst, (int)VPX_BITS_8);
#else
      vp9_scale_and_extend_frame_nonnormative(src, dst);
#endif  // CONFIG_VP9_HIGHBITDEPTH
      return;
    }

    for (i = 0; i < MAX_MB_PLANE; ++i) {
      const int factor = (i == 0 || i == 3 ? 1 : 2);
      const int src_stride = src_strides[i];
      const int dst_stride = dst_strides[i];
      for (y = 0; y < dst_h; y += 16) {
        const int y_q4 = y * (16 / factor) * src_h / dst_h + phase_scaler;
        for (x = 0; x < dst_w; x += 16) {
          const int x_q4 = x * (16 / factor) * src_w / dst_w + phase_scaler;
          const uint8_t *src_ptr = srcs[i] +
                                   (y / factor) * src_h / dst_h * src_stride +
                                   (x / factor) * src_w / dst_w;
          uint8_t *dst_ptr = dsts[i] + (y / factor) * dst_stride + (x / factor);

          vpx_scaled_2d(src_ptr, src_stride, dst_ptr, dst_stride, kernel,
                        x_q4 & 0xf, 16 * src_w / dst_w, y_q4 & 0xf,
                        16 * src_h / dst_h, 16 / factor, 16 / factor);
        }
      }
    }
  }

  vpx_extend_frame_borders(dst);
}
