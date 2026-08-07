/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_SCALE_YV12CONFIG_H_
#define VPX_VPX_SCALE_YV12CONFIG_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "./vpx_config.h"
#include "vpx/vpx_codec.h"
#include "vpx/vpx_frame_buffer.h"
#include "vpx/vpx_integer.h"

#define VP8BORDERINPIXELS 32
#define VP9INNERBORDERINPIXELS 96
#define VP9_INTERP_EXTEND 4
#define VP9_ENC_BORDER_IN_PIXELS 160
#define VP9_DEC_BORDER_IN_PIXELS 32

typedef struct yv12_buffer_config {
  int y_width;
  int y_height;
  int y_crop_width;
  int y_crop_height;
  int y_stride;

  int uv_width;
  int uv_height;
  int uv_crop_width;
  int uv_crop_height;
  int uv_stride;

  int alpha_width;
  int alpha_height;
  int alpha_stride;

  uint8_t *y_buffer;
  uint8_t *u_buffer;
  uint8_t *v_buffer;
  uint8_t *alpha_buffer;

  uint8_t *buffer_alloc;
  size_t buffer_alloc_sz;
  int border;
  size_t frame_size;
  int subsampling_x;
  int subsampling_y;
  unsigned int bit_depth;
  vpx_color_space_t color_space;
  vpx_color_range_t color_range;
  int render_width;
  int render_height;

  int corrupted;
  int flags;
} YV12_BUFFER_CONFIG;

#define YV12_FLAG_HIGHBITDEPTH 8

int vp8_yv12_alloc_frame_buffer(YV12_BUFFER_CONFIG *ybf, int width, int height,
                                int border);
int vp8_yv12_realloc_frame_buffer(YV12_BUFFER_CONFIG *ybf, int width,
                                  int height, int border);
int vp8_yv12_de_alloc_frame_buffer(YV12_BUFFER_CONFIG *ybf);

int vpx_alloc_frame_buffer(YV12_BUFFER_CONFIG *ybf, int width, int height,
                           int ss_x, int ss_y,
#if CONFIG_VP9_HIGHBITDEPTH
                           int use_highbitdepth,
#endif
                           int border, int byte_alignment);

// Updates the yv12 buffer config with the frame buffer. |byte_alignment| must
// be a power of 2, from 32 to 1024. 0 sets legacy alignment. If cb is not
// NULL, then libvpx is using the frame buffer callbacks to handle memory.
// If cb is not NULL, libvpx will call cb with minimum size in bytes needed
// to decode the current frame. If cb is NULL, libvpx will allocate memory
// internally to decode the current frame. Returns 0 on success. Returns < 0
// on failure.
int vpx_realloc_frame_buffer(YV12_BUFFER_CONFIG *ybf, int width, int height,
                             int ss_x, int ss_y,
#if CONFIG_VP9_HIGHBITDEPTH
                             int use_highbitdepth,
#endif
                             int border, int byte_alignment,
                             vpx_codec_frame_buffer_t *fb,
                             vpx_get_frame_buffer_cb_fn_t cb, void *cb_priv);
int vpx_free_frame_buffer(YV12_BUFFER_CONFIG *ybf);

#ifdef __cplusplus
}
#endif

#endif  // VPX_VPX_SCALE_YV12CONFIG_H_
