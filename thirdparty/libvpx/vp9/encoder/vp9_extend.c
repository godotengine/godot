/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_mem/vpx_mem.h"
#include "vpx_ports/mem.h"

#include "vp9/common/vp9_common.h"
#include "vp9/encoder/vp9_extend.h"

static void copy_and_extend_plane(const uint8_t *src, int src_pitch,
                                  uint8_t *dst, int dst_pitch, int w, int h,
                                  int extend_top, int extend_left,
                                  int extend_bottom, int extend_right,
                                  int interleave_step) {
  int i, j, linesize;
  const int step = interleave_step < 1 ? 1 : interleave_step;

  // copy the left and right most columns out
  const uint8_t *src_ptr1 = src;
  const uint8_t *src_ptr2 = src + (w - 1) * step;
  uint8_t *dst_ptr1 = dst - extend_left;
  uint8_t *dst_ptr2 = dst + w;

  for (i = 0; i < h; i++) {
    memset(dst_ptr1, src_ptr1[0], extend_left);
    if (step == 1) {
      memcpy(dst_ptr1 + extend_left, src_ptr1, w);
    } else {
      for (j = 0; j < w; j++) {
        dst_ptr1[extend_left + j] = src_ptr1[step * j];
      }
    }
    memset(dst_ptr2, src_ptr2[0], extend_right);
    src_ptr1 += src_pitch;
    src_ptr2 += src_pitch;
    dst_ptr1 += dst_pitch;
    dst_ptr2 += dst_pitch;
  }

  // Now copy the top and bottom lines into each line of the respective
  // borders
  src_ptr1 = dst - extend_left;
  src_ptr2 = dst + dst_pitch * (h - 1) - extend_left;
  dst_ptr1 = dst + dst_pitch * (-extend_top) - extend_left;
  dst_ptr2 = dst + dst_pitch * (h)-extend_left;
  linesize = extend_left + extend_right + w;

  for (i = 0; i < extend_top; i++) {
    memcpy(dst_ptr1, src_ptr1, linesize);
    dst_ptr1 += dst_pitch;
  }

  for (i = 0; i < extend_bottom; i++) {
    memcpy(dst_ptr2, src_ptr2, linesize);
    dst_ptr2 += dst_pitch;
  }
}

#if CONFIG_VP9_HIGHBITDEPTH
static void highbd_copy_and_extend_plane(const uint8_t *src8, int src_pitch,
                                         uint8_t *dst8, int dst_pitch, int w,
                                         int h, int extend_top, int extend_left,
                                         int extend_bottom, int extend_right) {
  int i, linesize;
  uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  uint16_t *dst = CONVERT_TO_SHORTPTR(dst8);

  // copy the left and right most columns out
  const uint16_t *src_ptr1 = src;
  const uint16_t *src_ptr2 = src + w - 1;
  uint16_t *dst_ptr1 = dst - extend_left;
  uint16_t *dst_ptr2 = dst + w;

  for (i = 0; i < h; i++) {
    vpx_memset16(dst_ptr1, src_ptr1[0], extend_left);
    memcpy(dst_ptr1 + extend_left, src_ptr1, w * sizeof(src_ptr1[0]));
    vpx_memset16(dst_ptr2, src_ptr2[0], extend_right);
    src_ptr1 += src_pitch;
    src_ptr2 += src_pitch;
    dst_ptr1 += dst_pitch;
    dst_ptr2 += dst_pitch;
  }

  // Now copy the top and bottom lines into each line of the respective
  // borders
  src_ptr1 = dst - extend_left;
  src_ptr2 = dst + dst_pitch * (h - 1) - extend_left;
  dst_ptr1 = dst + dst_pitch * (-extend_top) - extend_left;
  dst_ptr2 = dst + dst_pitch * (h)-extend_left;
  linesize = extend_left + extend_right + w;

  for (i = 0; i < extend_top; i++) {
    memcpy(dst_ptr1, src_ptr1, linesize * sizeof(src_ptr1[0]));
    dst_ptr1 += dst_pitch;
  }

  for (i = 0; i < extend_bottom; i++) {
    memcpy(dst_ptr2, src_ptr2, linesize * sizeof(src_ptr2[0]));
    dst_ptr2 += dst_pitch;
  }
}
#endif  // CONFIG_VP9_HIGHBITDEPTH

void vp9_copy_and_extend_frame(const YV12_BUFFER_CONFIG *src,
                               YV12_BUFFER_CONFIG *dst) {
  // Extend src frame in buffer
  // Altref filtering assumes 16 pixel extension
  const int et_y = 16;
  const int el_y = 16;
  // Motion estimation may use src block variance with the block size up
  // to 64x64, so the right and bottom need to be extended to 64 multiple
  // or up to 16, whichever is greater.
  const int er_y =
      VPXMAX(src->y_width + 16, ALIGN_POWER_OF_TWO(src->y_width, 6)) -
      src->y_crop_width;
  const int eb_y =
      VPXMAX(src->y_height + 16, ALIGN_POWER_OF_TWO(src->y_height, 6)) -
      src->y_crop_height;
  const int uv_width_subsampling = (src->uv_width != src->y_width);
  const int uv_height_subsampling = (src->uv_height != src->y_height);
  const int et_uv = et_y >> uv_height_subsampling;
  const int el_uv = el_y >> uv_width_subsampling;
  const int eb_uv = eb_y >> uv_height_subsampling;
  const int er_uv = er_y >> uv_width_subsampling;
  // detect nv12 colorspace
  const int chroma_step = src->v_buffer - src->u_buffer == 1 ? 2 : 1;

#if CONFIG_VP9_HIGHBITDEPTH
  if (src->flags & YV12_FLAG_HIGHBITDEPTH) {
    highbd_copy_and_extend_plane(src->y_buffer, src->y_stride, dst->y_buffer,
                                 dst->y_stride, src->y_crop_width,
                                 src->y_crop_height, et_y, el_y, eb_y, er_y);

    highbd_copy_and_extend_plane(
        src->u_buffer, src->uv_stride, dst->u_buffer, dst->uv_stride,
        src->uv_crop_width, src->uv_crop_height, et_uv, el_uv, eb_uv, er_uv);

    highbd_copy_and_extend_plane(
        src->v_buffer, src->uv_stride, dst->v_buffer, dst->uv_stride,
        src->uv_crop_width, src->uv_crop_height, et_uv, el_uv, eb_uv, er_uv);
    return;
  }
#endif  // CONFIG_VP9_HIGHBITDEPTH

  copy_and_extend_plane(src->y_buffer, src->y_stride, dst->y_buffer,
                        dst->y_stride, src->y_crop_width, src->y_crop_height,
                        et_y, el_y, eb_y, er_y, 1);

  copy_and_extend_plane(src->u_buffer, src->uv_stride, dst->u_buffer,
                        dst->uv_stride, src->uv_crop_width, src->uv_crop_height,
                        et_uv, el_uv, eb_uv, er_uv, chroma_step);

  copy_and_extend_plane(src->v_buffer, src->uv_stride, dst->v_buffer,
                        dst->uv_stride, src->uv_crop_width, src->uv_crop_height,
                        et_uv, el_uv, eb_uv, er_uv, chroma_step);
}
