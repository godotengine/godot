/*
 *  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>

#include "./vpx_config.h"
#include "vpx_scale/yv12config.h"
#include "vpx_mem/vpx_mem.h"
#include "vpx_scale/vpx_scale.h"

#if HAVE_DSPR2
static void extend_plane(uint8_t *const src, int src_stride, int width,
                         int height, int extend_top, int extend_left,
                         int extend_bottom, int extend_right) {
  int i, j;
  uint8_t *left_src, *right_src;
  uint8_t *left_dst_start, *right_dst_start;
  uint8_t *left_dst, *right_dst;
  uint8_t *top_src, *bot_src;
  uint8_t *top_dst, *bot_dst;
  uint32_t left_pix;
  uint32_t right_pix;
  uint32_t linesize;

  /* copy the left and right most columns out */
  left_src = src;
  right_src = src + width - 1;
  left_dst_start = src - extend_left;
  right_dst_start = src + width;

  for (i = height; i--;) {
    left_dst = left_dst_start;
    right_dst = right_dst_start;

    __asm__ __volatile__(
        "lb        %[left_pix],     0(%[left_src])      \n\t"
        "lb        %[right_pix],    0(%[right_src])     \n\t"
        "replv.qb  %[left_pix],     %[left_pix]         \n\t"
        "replv.qb  %[right_pix],    %[right_pix]        \n\t"

        : [left_pix] "=&r"(left_pix), [right_pix] "=&r"(right_pix)
        : [left_src] "r"(left_src), [right_src] "r"(right_src));

    for (j = extend_left / 4; j--;) {
      __asm__ __volatile__(
          "sw     %[left_pix],    0(%[left_dst])     \n\t"
          "sw     %[right_pix],   0(%[right_dst])    \n\t"

          :
          : [left_dst] "r"(left_dst), [left_pix] "r"(left_pix),
            [right_dst] "r"(right_dst), [right_pix] "r"(right_pix));

      left_dst += 4;
      right_dst += 4;
    }

    for (j = extend_left % 4; j--;) {
      __asm__ __volatile__(
          "sb     %[left_pix],    0(%[left_dst])     \n\t"
          "sb     %[right_pix],   0(%[right_dst])     \n\t"

          :
          : [left_dst] "r"(left_dst), [left_pix] "r"(left_pix),
            [right_dst] "r"(right_dst), [right_pix] "r"(right_pix));

      left_dst += 1;
      right_dst += 1;
    }

    left_src += src_stride;
    right_src += src_stride;
    left_dst_start += src_stride;
    right_dst_start += src_stride;
  }

  /* Now copy the top and bottom lines into each line of the respective
   * borders
   */
  top_src = src - extend_left;
  bot_src = src + src_stride * (height - 1) - extend_left;
  top_dst = src + src_stride * (-extend_top) - extend_left;
  bot_dst = src + src_stride * (height)-extend_left;
  linesize = extend_left + extend_right + width;

  for (i = 0; i < extend_top; i++) {
    memcpy(top_dst, top_src, linesize);
    top_dst += src_stride;
  }

  for (i = 0; i < extend_bottom; i++) {
    memcpy(bot_dst, bot_src, linesize);
    bot_dst += src_stride;
  }
}

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

  extend_plane(ybf->y_buffer, ybf->y_stride, ybf->y_crop_width,
               ybf->y_crop_height, ext_size, ext_size,
               ext_size + ybf->y_height - ybf->y_crop_height,
               ext_size + ybf->y_width - ybf->y_crop_width);

  extend_plane(ybf->u_buffer, ybf->uv_stride, c_w, c_h, c_et, c_el, c_eb, c_er);

  extend_plane(ybf->v_buffer, ybf->uv_stride, c_w, c_h, c_et, c_el, c_eb, c_er);
}

void vpx_extend_frame_borders_dspr2(YV12_BUFFER_CONFIG *ybf) {
  extend_frame(ybf, ybf->border);
}

void vpx_extend_frame_inner_borders_dspr2(YV12_BUFFER_CONFIG *ybf) {
  const int inner_bw = (ybf->border > VP9INNERBORDERINPIXELS)
                           ? VP9INNERBORDERINPIXELS
                           : ybf->border;
  extend_frame(ybf, inner_bw);
}
#endif
