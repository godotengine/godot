/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "vp8/common/alloccommon.h"
#include "vp8/common/vp8_skin_detection.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_mem/vpx_mem.h"
#include "vpx_util/vpx_write_yuv_frame.h"

static int avg_2x2(const uint8_t *s, int p) {
  int i, j;
  int sum = 0;
  for (i = 0; i < 2; ++i, s += p) {
    for (j = 0; j < 2; ++j) {
      sum += s[j];
    }
  }
  return (sum + 2) >> 2;
}

int vp8_compute_skin_block(const uint8_t *y, const uint8_t *u, const uint8_t *v,
                           int stride, int strideuv,
                           SKIN_DETECTION_BLOCK_SIZE bsize, int consec_zeromv,
                           int curr_motion_magn) {
  // No skin if block has been zero/small motion for long consecutive time.
  if (consec_zeromv > 60 && curr_motion_magn == 0) {
    return 0;
  } else {
    int motion = 1;
    if (consec_zeromv > 25 && curr_motion_magn == 0) motion = 0;
    if (bsize == SKIN_16X16) {
      // Take the average of center 2x2 pixels.
      const int ysource = avg_2x2(y + 7 * stride + 7, stride);
      const int usource = avg_2x2(u + 3 * strideuv + 3, strideuv);
      const int vsource = avg_2x2(v + 3 * strideuv + 3, strideuv);
      return vpx_skin_pixel(ysource, usource, vsource, motion);
    } else {
      int num_skin = 0;
      int i, j;
      for (i = 0; i < 2; i++) {
        for (j = 0; j < 2; j++) {
          // Take the average of center 2x2 pixels.
          const int ysource = avg_2x2(y + 3 * stride + 3, stride);
          const int usource = avg_2x2(u + strideuv + 1, strideuv);
          const int vsource = avg_2x2(v + strideuv + 1, strideuv);
          num_skin += vpx_skin_pixel(ysource, usource, vsource, motion);
          if (num_skin >= 2) return 1;
          y += 8;
          u += 4;
          v += 4;
        }
        y += (stride << 3) - 16;
        u += (strideuv << 2) - 8;
        v += (strideuv << 2) - 8;
      }

      return 0;
    }
  }
}

#ifdef OUTPUT_YUV_SKINMAP
// For viewing skin map on input source.
void vp8_compute_skin_map(VP8_COMP *const cpi, FILE *yuv_skinmap_file) {
  int i, j, mb_row, mb_col, num_bl;
  VP8_COMMON *const cm = &cpi->common;
  uint8_t *y;
  const uint8_t *src_y = cpi->Source->y_buffer;
  const int src_ystride = cpi->Source->y_stride;
  int offset = 0;

  YV12_BUFFER_CONFIG skinmap;
  memset(&skinmap, 0, sizeof(skinmap));
  if (vp8_yv12_alloc_frame_buffer(&skinmap, cm->Width, cm->Height,
                                  VP8BORDERINPIXELS) < 0) {
    vpx_free_frame_buffer(&skinmap);
    return;
  }
  memset(skinmap.buffer_alloc, 128, skinmap.frame_size);
  y = skinmap.y_buffer;
  // Loop through blocks and set skin map based on center pixel of block.
  // Set y to white for skin block, otherwise set to source with gray scale.
  for (mb_row = 0; mb_row < cm->mb_rows; mb_row += 1) {
    num_bl = 0;
    for (mb_col = 0; mb_col < cm->mb_cols; mb_col += 1) {
      const int is_skin = cpi->skin_map[offset++];
      for (i = 0; i < 16; i++) {
        for (j = 0; j < 16; j++) {
          y[i * src_ystride + j] = is_skin ? 255 : src_y[i * src_ystride + j];
        }
      }
      num_bl++;
      y += 16;
      src_y += 16;
    }
    y += (src_ystride << 4) - (num_bl << 4);
    src_y += (src_ystride << 4) - (num_bl << 4);
  }
  vpx_write_yuv_frame(yuv_skinmap_file, &skinmap);
  vpx_free_frame_buffer(&skinmap);
}
#endif  // OUTPUT_YUV_SKINMAP
