/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <limits.h>
#include <math.h>

#include "vp9/common/vp9_blockd.h"
#include "vp9/encoder/vp9_encoder.h"
#include "vp9/encoder/vp9_skin_detection.h"

int vp9_compute_skin_block(const uint8_t *y, const uint8_t *u, const uint8_t *v,
                           int stride, int strideuv, int bsize,
                           int consec_zeromv, int curr_motion_magn) {
  // No skin if block has been zero/small motion for long consecutive time.
  if (consec_zeromv > 60 && curr_motion_magn == 0) {
    return 0;
  } else {
    int motion = 1;
    // Take center pixel in block to determine is_skin.
    const int y_width_shift = (4 << b_width_log2_lookup[bsize]) >> 1;
    const int y_height_shift = (4 << b_height_log2_lookup[bsize]) >> 1;
    const int uv_width_shift = y_width_shift >> 1;
    const int uv_height_shift = y_height_shift >> 1;
    const uint8_t ysource = y[y_height_shift * stride + y_width_shift];
    const uint8_t usource = u[uv_height_shift * strideuv + uv_width_shift];
    const uint8_t vsource = v[uv_height_shift * strideuv + uv_width_shift];

    if (consec_zeromv > 25 && curr_motion_magn == 0) motion = 0;
    return vpx_skin_pixel(ysource, usource, vsource, motion);
  }
}

void vp9_compute_skin_sb(VP9_COMP *const cpi, BLOCK_SIZE bsize, int mi_row,
                         int mi_col) {
  int i, j, num_bl;
  VP9_COMMON *const cm = &cpi->common;
  const uint8_t *src_y = cpi->Source->y_buffer;
  const uint8_t *src_u = cpi->Source->u_buffer;
  const uint8_t *src_v = cpi->Source->v_buffer;
  const int src_ystride = cpi->Source->y_stride;
  const int src_uvstride = cpi->Source->uv_stride;
  const int y_bsize = 4 << b_width_log2_lookup[bsize];
  const int uv_bsize = y_bsize >> 1;
  const int shy = (y_bsize == 8) ? 3 : 4;
  const int shuv = shy - 1;
  const int fac = y_bsize / 8;
  const int y_shift = src_ystride * (mi_row << 3) + (mi_col << 3);
  const int uv_shift = src_uvstride * (mi_row << 2) + (mi_col << 2);
  const int mi_row_limit = VPXMIN(mi_row + 8, cm->mi_rows - 2);
  const int mi_col_limit = VPXMIN(mi_col + 8, cm->mi_cols - 2);
  src_y += y_shift;
  src_u += uv_shift;
  src_v += uv_shift;

  for (i = mi_row; i < mi_row_limit; i += fac) {
    num_bl = 0;
    for (j = mi_col; j < mi_col_limit; j += fac) {
      int consec_zeromv = 0;
      int bl_index = i * cm->mi_cols + j;
      int bl_index1 = bl_index + 1;
      int bl_index2 = bl_index + cm->mi_cols;
      int bl_index3 = bl_index2 + 1;
      // Don't detect skin on the boundary.
      if (i == 0 || j == 0) continue;
      if (bsize == BLOCK_8X8)
        consec_zeromv = cpi->consec_zero_mv[bl_index];
      else
        consec_zeromv = VPXMIN(cpi->consec_zero_mv[bl_index],
                               VPXMIN(cpi->consec_zero_mv[bl_index1],
                                      VPXMIN(cpi->consec_zero_mv[bl_index2],
                                             cpi->consec_zero_mv[bl_index3])));
      cpi->skin_map[bl_index] =
          vp9_compute_skin_block(src_y, src_u, src_v, src_ystride, src_uvstride,
                                 bsize, consec_zeromv, 0);
      num_bl++;
      src_y += y_bsize;
      src_u += uv_bsize;
      src_v += uv_bsize;
    }
    src_y += (src_ystride << shy) - (num_bl << shy);
    src_u += (src_uvstride << shuv) - (num_bl << shuv);
    src_v += (src_uvstride << shuv) - (num_bl << shuv);
  }

  // Remove isolated skin blocks (none of its neighbors are skin) and isolated
  // non-skin blocks (all of its neighbors are skin).
  // Skip 4 corner blocks which have only 3 neighbors to remove isolated skin
  // blocks. Skip superblock borders to remove isolated non-skin blocks.
  for (i = mi_row; i < mi_row_limit; i += fac) {
    for (j = mi_col; j < mi_col_limit; j += fac) {
      int bl_index = i * cm->mi_cols + j;
      int num_neighbor = 0;
      int mi, mj;
      int non_skin_threshold = 8;
      // Skip 4 corners.
      if ((i == mi_row && (j == mi_col || j == mi_col_limit - fac)) ||
          (i == mi_row_limit - fac && (j == mi_col || j == mi_col_limit - fac)))
        continue;
      // There are only 5 neighbors for non-skin blocks on the border.
      if (i == mi_row || i == mi_row_limit - fac || j == mi_col ||
          j == mi_col_limit - fac)
        non_skin_threshold = 5;

      for (mi = -fac; mi <= fac; mi += fac) {
        for (mj = -fac; mj <= fac; mj += fac) {
          if (i + mi >= mi_row && i + mi < mi_row_limit && j + mj >= mi_col &&
              j + mj < mi_col_limit) {
            int bl_neighbor_index = (i + mi) * cm->mi_cols + j + mj;
            if (cpi->skin_map[bl_neighbor_index]) num_neighbor++;
          }
        }
      }

      if (cpi->skin_map[bl_index] && num_neighbor < 2)
        cpi->skin_map[bl_index] = 0;
      if (!cpi->skin_map[bl_index] && num_neighbor == non_skin_threshold)
        cpi->skin_map[bl_index] = 1;
    }
  }
}

#ifdef OUTPUT_YUV_SKINMAP
// For viewing skin map on input source.
void vp9_output_skin_map(VP9_COMP *const cpi, FILE *yuv_skinmap_file) {
  int i, j, mi_row, mi_col, num_bl;
  VP9_COMMON *const cm = &cpi->common;
  uint8_t *y;
  const uint8_t *src_y = cpi->Source->y_buffer;
  const int src_ystride = cpi->Source->y_stride;
  const int y_bsize = 16;  // Use 8x8 or 16x16.
  const int shy = (y_bsize == 8) ? 3 : 4;
  const int fac = y_bsize / 8;

  YV12_BUFFER_CONFIG skinmap;
  memset(&skinmap, 0, sizeof(YV12_BUFFER_CONFIG));
  if (vpx_alloc_frame_buffer(&skinmap, cm->width, cm->height, cm->subsampling_x,
                             cm->subsampling_y, VP9_ENC_BORDER_IN_PIXELS,
                             cm->byte_alignment)) {
    vpx_free_frame_buffer(&skinmap);
    return;
  }
  memset(skinmap.buffer_alloc, 128, skinmap.frame_size);
  y = skinmap.y_buffer;
  // Loop through blocks and set skin map based on center pixel of block.
  // Set y to white for skin block, otherwise set to source with gray scale.
  // Ignore rightmost/bottom boundary blocks.
  for (mi_row = 0; mi_row < cm->mi_rows - 1; mi_row += fac) {
    num_bl = 0;
    for (mi_col = 0; mi_col < cm->mi_cols - 1; mi_col += fac) {
      const int block_index = mi_row * cm->mi_cols + mi_col;
      const int is_skin = cpi->skin_map[block_index];
      for (i = 0; i < y_bsize; i++) {
        for (j = 0; j < y_bsize; j++) {
          y[i * src_ystride + j] = is_skin ? 255 : src_y[i * src_ystride + j];
        }
      }
      num_bl++;
      y += y_bsize;
      src_y += y_bsize;
    }
    y += (src_ystride << shy) - (num_bl << shy);
    src_y += (src_ystride << shy) - (num_bl << shy);
  }
  vpx_write_yuv_frame(yuv_skinmap_file, &skinmap);
  vpx_free_frame_buffer(&skinmap);
}
#endif
