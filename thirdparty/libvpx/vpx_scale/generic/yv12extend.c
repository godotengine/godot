/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>
#include "./vpx_config.h"
#include "./vpx_scale_rtcd.h"
#include "vpx/vpx_integer.h"
#include "vpx_mem/vpx_mem.h"
#include "vpx_ports/mem.h"
#include "vpx_scale/yv12config.h"
#if CONFIG_VP9_HIGHBITDEPTH
#include "vp9/common/vp9_common.h"
#endif

static void extend_plane(uint8_t *const src, int src_stride,
                         int width, int height,
                         int extend_top, int extend_left,
                         int extend_bottom, int extend_right) {
  int i;
  const int linesize = extend_left + extend_right + width;

  /* copy the left and right most columns out */
  uint8_t *src_ptr1 = src;
  uint8_t *src_ptr2 = src + width - 1;
  uint8_t *dst_ptr1 = src - extend_left;
  uint8_t *dst_ptr2 = src + width;

  for (i = 0; i < height; ++i) {
    memset(dst_ptr1, src_ptr1[0], extend_left);
    memset(dst_ptr2, src_ptr2[0], extend_right);
    src_ptr1 += src_stride;
    src_ptr2 += src_stride;
    dst_ptr1 += src_stride;
    dst_ptr2 += src_stride;
  }

  /* Now copy the top and bottom lines into each line of the respective
   * borders
   */
  src_ptr1 = src - extend_left;
  src_ptr2 = src + src_stride * (height - 1) - extend_left;
  dst_ptr1 = src + src_stride * -extend_top - extend_left;
  dst_ptr2 = src + src_stride * height - extend_left;

  for (i = 0; i < extend_top; ++i) {
    memcpy(dst_ptr1, src_ptr1, linesize);
    dst_ptr1 += src_stride;
  }

  for (i = 0; i < extend_bottom; ++i) {
    memcpy(dst_ptr2, src_ptr2, linesize);
    dst_ptr2 += src_stride;
  }
}

#if CONFIG_VP9_HIGHBITDEPTH
static void extend_plane_high(uint8_t *const src8, int src_stride,
                              int width, int height,
                              int extend_top, int extend_left,
                              int extend_bottom, int extend_right) {
  int i;
  const int linesize = extend_left + extend_right + width;
  uint16_t *src = CONVERT_TO_SHORTPTR(src8);

  /* copy the left and right most columns out */
  uint16_t *src_ptr1 = src;
  uint16_t *src_ptr2 = src + width - 1;
  uint16_t *dst_ptr1 = src - extend_left;
  uint16_t *dst_ptr2 = src + width;

  for (i = 0; i < height; ++i) {
    vpx_memset16(dst_ptr1, src_ptr1[0], extend_left);
    vpx_memset16(dst_ptr2, src_ptr2[0], extend_right);
    src_ptr1 += src_stride;
    src_ptr2 += src_stride;
    dst_ptr1 += src_stride;
    dst_ptr2 += src_stride;
  }

  /* Now copy the top and bottom lines into each line of the respective
   * borders
   */
  src_ptr1 = src - extend_left;
  src_ptr2 = src + src_stride * (height - 1) - extend_left;
  dst_ptr1 = src + src_stride * -extend_top - extend_left;
  dst_ptr2 = src + src_stride * height - extend_left;

  for (i = 0; i < extend_top; ++i) {
    memcpy(dst_ptr1, src_ptr1, linesize * sizeof(uint16_t));
    dst_ptr1 += src_stride;
  }

  for (i = 0; i < extend_bottom; ++i) {
    memcpy(dst_ptr2, src_ptr2, linesize * sizeof(uint16_t));
    dst_ptr2 += src_stride;
  }
}
#endif

void vp8_yv12_extend_frame_borders_c(YV12_BUFFER_CONFIG *ybf) {
  const int uv_border = ybf->border / 2;

  assert(ybf->border % 2 == 0);
  assert(ybf->y_height - ybf->y_crop_height < 16);
  assert(ybf->y_width - ybf->y_crop_width < 16);
  assert(ybf->y_height - ybf->y_crop_height >= 0);
  assert(ybf->y_width - ybf->y_crop_width >= 0);

#if CONFIG_VP9_HIGHBITDEPTH
  if (ybf->flags & YV12_FLAG_HIGHBITDEPTH) {
    extend_plane_high(
        ybf->y_buffer, ybf->y_stride,
        ybf->y_crop_width, ybf->y_crop_height,
        ybf->border, ybf->border,
        ybf->border + ybf->y_height - ybf->y_crop_height,
        ybf->border + ybf->y_width - ybf->y_crop_width);

    extend_plane_high(
        ybf->u_buffer, ybf->uv_stride,
        ybf->uv_crop_width, ybf->uv_crop_height,
        uv_border, uv_border,
        uv_border + ybf->uv_height - ybf->uv_crop_height,
        uv_border + ybf->uv_width - ybf->uv_crop_width);

    extend_plane_high(
        ybf->v_buffer, ybf->uv_stride,
        ybf->uv_crop_width, ybf->uv_crop_height,
        uv_border, uv_border,
        uv_border + ybf->uv_height - ybf->uv_crop_height,
        uv_border + ybf->uv_width - ybf->uv_crop_width);
    return;
  }
#endif
  extend_plane(ybf->y_buffer, ybf->y_stride,
               ybf->y_crop_width, ybf->y_crop_height,
               ybf->border, ybf->border,
               ybf->border + ybf->y_height - ybf->y_crop_height,
               ybf->border + ybf->y_width - ybf->y_crop_width);

  extend_plane(ybf->u_buffer, ybf->uv_stride,
               ybf->uv_crop_width, ybf->uv_crop_height,
               uv_border, uv_border,
               uv_border + ybf->uv_height - ybf->uv_crop_height,
               uv_border + ybf->uv_width - ybf->uv_crop_width);

  extend_plane(ybf->v_buffer, ybf->uv_stride,
               ybf->uv_crop_width, ybf->uv_crop_height,
               uv_border, uv_border,
               uv_border + ybf->uv_height - ybf->uv_crop_height,
               uv_border + ybf->uv_width - ybf->uv_crop_width);
}

#if CONFIG_VP9
static void extend_frame(YV12_BUFFER_CONFIG *const ybf, int ext_size) {
  const int c_w = ybf->uv_crop_width;
  const int c_h = ybf->uv_crop_height;
  const int ss_x = ybf->uv_width < ybf->y_width;
  const int ss_y = ybf->uv_height < ybf->y_height;
  const int c_et = ext_size >> ss_y;
  const int c_el = ext_size >> ss_x;
  const int c_eb = c_et + ybf->uv_height - ybf->uv_crop_height;
  const int c_er = c_el + ybf->uv_width - ybf->uv_crop_width;

  assert(ybf->y_height - ybf->y_crop_height < 16);
  assert(ybf->y_width - ybf->y_crop_width < 16);
  assert(ybf->y_height - ybf->y_crop_height >= 0);
  assert(ybf->y_width - ybf->y_crop_width >= 0);

#if CONFIG_VP9_HIGHBITDEPTH
  if (ybf->flags & YV12_FLAG_HIGHBITDEPTH) {
    extend_plane_high(ybf->y_buffer, ybf->y_stride,
                      ybf->y_crop_width, ybf->y_crop_height,
                      ext_size, ext_size,
                      ext_size + ybf->y_height - ybf->y_crop_height,
                      ext_size + ybf->y_width - ybf->y_crop_width);
    extend_plane_high(ybf->u_buffer, ybf->uv_stride,
                      c_w, c_h, c_et, c_el, c_eb, c_er);
    extend_plane_high(ybf->v_buffer, ybf->uv_stride,
                      c_w, c_h, c_et, c_el, c_eb, c_er);
    return;
  }
#endif
  extend_plane(ybf->y_buffer, ybf->y_stride,
               ybf->y_crop_width, ybf->y_crop_height,
               ext_size, ext_size,
               ext_size + ybf->y_height - ybf->y_crop_height,
               ext_size + ybf->y_width - ybf->y_crop_width);

  extend_plane(ybf->u_buffer, ybf->uv_stride,
               c_w, c_h, c_et, c_el, c_eb, c_er);

  extend_plane(ybf->v_buffer, ybf->uv_stride,
               c_w, c_h, c_et, c_el, c_eb, c_er);
}

void vpx_extend_frame_borders_c(YV12_BUFFER_CONFIG *ybf) {
  extend_frame(ybf, ybf->border);
}

void vpx_extend_frame_inner_borders_c(YV12_BUFFER_CONFIG *ybf) {
  const int inner_bw = (ybf->border > VP9INNERBORDERINPIXELS) ?
                       VP9INNERBORDERINPIXELS : ybf->border;
  extend_frame(ybf, inner_bw);
}

#if CONFIG_VP9_HIGHBITDEPTH
static void memcpy_short_addr(uint8_t *dst8, const uint8_t *src8, int num) {
  uint16_t *dst = CONVERT_TO_SHORTPTR(dst8);
  uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  memcpy(dst, src, num * sizeof(uint16_t));
}
#endif  // CONFIG_VP9_HIGHBITDEPTH
#endif  // CONFIG_VP9

// Copies the source image into the destination image and updates the
// destination's UMV borders.
// Note: The frames are assumed to be identical in size.
void vp8_yv12_copy_frame_c(const YV12_BUFFER_CONFIG *src_ybc,
                           YV12_BUFFER_CONFIG *dst_ybc) {
  int row;
  const uint8_t *src = src_ybc->y_buffer;
  uint8_t *dst = dst_ybc->y_buffer;

#if 0
  /* These assertions are valid in the codec, but the libvpx-tester uses
   * this code slightly differently.
   */
  assert(src_ybc->y_width == dst_ybc->y_width);
  assert(src_ybc->y_height == dst_ybc->y_height);
#endif

#if CONFIG_VP9_HIGHBITDEPTH
  if (src_ybc->flags & YV12_FLAG_HIGHBITDEPTH) {
    assert(dst_ybc->flags & YV12_FLAG_HIGHBITDEPTH);
    for (row = 0; row < src_ybc->y_height; ++row) {
      memcpy_short_addr(dst, src, src_ybc->y_width);
      src += src_ybc->y_stride;
      dst += dst_ybc->y_stride;
    }

    src = src_ybc->u_buffer;
    dst = dst_ybc->u_buffer;

    for (row = 0; row < src_ybc->uv_height; ++row) {
      memcpy_short_addr(dst, src, src_ybc->uv_width);
      src += src_ybc->uv_stride;
      dst += dst_ybc->uv_stride;
    }

    src = src_ybc->v_buffer;
    dst = dst_ybc->v_buffer;

    for (row = 0; row < src_ybc->uv_height; ++row) {
      memcpy_short_addr(dst, src, src_ybc->uv_width);
      src += src_ybc->uv_stride;
      dst += dst_ybc->uv_stride;
    }

    vp8_yv12_extend_frame_borders_c(dst_ybc);
    return;
  } else {
    assert(!(dst_ybc->flags & YV12_FLAG_HIGHBITDEPTH));
  }
#endif

  for (row = 0; row < src_ybc->y_height; ++row) {
    memcpy(dst, src, src_ybc->y_width);
    src += src_ybc->y_stride;
    dst += dst_ybc->y_stride;
  }

  src = src_ybc->u_buffer;
  dst = dst_ybc->u_buffer;

  for (row = 0; row < src_ybc->uv_height; ++row) {
    memcpy(dst, src, src_ybc->uv_width);
    src += src_ybc->uv_stride;
    dst += dst_ybc->uv_stride;
  }

  src = src_ybc->v_buffer;
  dst = dst_ybc->v_buffer;

  for (row = 0; row < src_ybc->uv_height; ++row) {
    memcpy(dst, src, src_ybc->uv_width);
    src += src_ybc->uv_stride;
    dst += dst_ybc->uv_stride;
  }

  vp8_yv12_extend_frame_borders_c(dst_ybc);
}

void vpx_yv12_copy_y_c(const YV12_BUFFER_CONFIG *src_ybc,
                       YV12_BUFFER_CONFIG *dst_ybc) {
  int row;
  const uint8_t *src = src_ybc->y_buffer;
  uint8_t *dst = dst_ybc->y_buffer;

#if CONFIG_VP9_HIGHBITDEPTH
  if (src_ybc->flags & YV12_FLAG_HIGHBITDEPTH) {
    const uint16_t *src16 = CONVERT_TO_SHORTPTR(src);
    uint16_t *dst16 = CONVERT_TO_SHORTPTR(dst);
    for (row = 0; row < src_ybc->y_height; ++row) {
      memcpy(dst16, src16, src_ybc->y_width * sizeof(uint16_t));
      src16 += src_ybc->y_stride;
      dst16 += dst_ybc->y_stride;
    }
    return;
  }
#endif

  for (row = 0; row < src_ybc->y_height; ++row) {
    memcpy(dst, src, src_ybc->y_width);
    src += src_ybc->y_stride;
    dst += dst_ybc->y_stride;
  }
}
