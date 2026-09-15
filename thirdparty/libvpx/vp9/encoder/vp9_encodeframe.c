/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>

#include "./vp9_rtcd.h"
#include "./vpx_dsp_rtcd.h"
#include "./vpx_config.h"

#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_ports/mem.h"
#include "vpx_ports/vpx_timer.h"
#include "vpx_ports/system_state.h"
#include "vpx_util/vpx_pthread.h"
#if CONFIG_MISMATCH_DEBUG
#include "vpx_util/vpx_debug_util.h"
#endif  // CONFIG_MISMATCH_DEBUG

#include "vp9/common/vp9_common.h"
#include "vp9/common/vp9_entropy.h"
#include "vp9/common/vp9_entropymode.h"
#include "vp9/common/vp9_idct.h"
#include "vp9/common/vp9_mvref_common.h"
#include "vp9/common/vp9_pred_common.h"
#include "vp9/common/vp9_quant_common.h"
#include "vp9/common/vp9_reconintra.h"
#include "vp9/common/vp9_reconinter.h"
#include "vp9/common/vp9_seg_common.h"
#include "vp9/common/vp9_tile_common.h"
#if !CONFIG_REALTIME_ONLY
#include "vp9/encoder/vp9_aq_360.h"
#include "vp9/encoder/vp9_aq_complexity.h"
#endif
#include "vp9/encoder/vp9_aq_cyclicrefresh.h"
#if !CONFIG_REALTIME_ONLY
#include "vp9/encoder/vp9_aq_variance.h"
#endif
#include "vp9/encoder/vp9_encodeframe.h"
#include "vp9/encoder/vp9_encodemb.h"
#include "vp9/encoder/vp9_encodemv.h"
#include "vp9/encoder/vp9_encoder.h"
#include "vp9/encoder/vp9_ethread.h"
#include "vp9/encoder/vp9_extend.h"
#include "vp9/encoder/vp9_multi_thread.h"
#include "vp9/encoder/vp9_partition_models.h"
#include "vp9/encoder/vp9_pickmode.h"
#include "vp9/encoder/vp9_rd.h"
#include "vp9/encoder/vp9_rdopt.h"
#include "vp9/encoder/vp9_segmentation.h"
#include "vp9/encoder/vp9_tokenize.h"

static void encode_superblock(VP9_COMP *cpi, ThreadData *td, TOKENEXTRA **t,
                              int output_enabled, int mi_row, int mi_col,
                              BLOCK_SIZE bsize, PICK_MODE_CONTEXT *ctx);

// This is used as a reference when computing the source variance for the
//  purpose of activity masking.
// Eventually this should be replaced by custom no-reference routines,
//  which will be faster.
static const uint8_t VP9_VAR_OFFS[64] = {
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128
};

#if CONFIG_VP9_HIGHBITDEPTH
static const uint16_t VP9_HIGH_VAR_OFFS_8[64] = {
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128
};

static const uint16_t VP9_HIGH_VAR_OFFS_10[64] = {
  128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4,
  128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4,
  128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4,
  128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4,
  128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4,
  128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4,
  128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4,
  128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4, 128 * 4
};

static const uint16_t VP9_HIGH_VAR_OFFS_12[64] = {
  128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16,
  128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16,
  128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16,
  128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16,
  128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16,
  128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16,
  128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16,
  128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16,
  128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16, 128 * 16,
  128 * 16
};
#endif  // CONFIG_VP9_HIGHBITDEPTH

unsigned int vp9_get_sby_variance(VP9_COMP *cpi, const struct buf_2d *ref,
                                  BLOCK_SIZE bs) {
  unsigned int sse;
  const unsigned int var =
      cpi->fn_ptr[bs].vf(ref->buf, ref->stride, VP9_VAR_OFFS, 0, &sse);
  return var;
}

#if CONFIG_VP9_HIGHBITDEPTH
unsigned int vp9_high_get_sby_variance(VP9_COMP *cpi, const struct buf_2d *ref,
                                       BLOCK_SIZE bs, int bd) {
  unsigned int var, sse;
  switch (bd) {
    case 10:
      var =
          cpi->fn_ptr[bs].vf(ref->buf, ref->stride,
                             CONVERT_TO_BYTEPTR(VP9_HIGH_VAR_OFFS_10), 0, &sse);
      break;
    case 12:
      var =
          cpi->fn_ptr[bs].vf(ref->buf, ref->stride,
                             CONVERT_TO_BYTEPTR(VP9_HIGH_VAR_OFFS_12), 0, &sse);
      break;
    case 8:
    default:
      var =
          cpi->fn_ptr[bs].vf(ref->buf, ref->stride,
                             CONVERT_TO_BYTEPTR(VP9_HIGH_VAR_OFFS_8), 0, &sse);
      break;
  }
  return var;
}
#endif  // CONFIG_VP9_HIGHBITDEPTH

unsigned int vp9_get_sby_perpixel_variance(VP9_COMP *cpi,
                                           const struct buf_2d *ref,
                                           BLOCK_SIZE bs) {
  return ROUND_POWER_OF_TWO(vp9_get_sby_variance(cpi, ref, bs),
                            num_pels_log2_lookup[bs]);
}

#if CONFIG_VP9_HIGHBITDEPTH
unsigned int vp9_high_get_sby_perpixel_variance(VP9_COMP *cpi,
                                                const struct buf_2d *ref,
                                                BLOCK_SIZE bs, int bd) {
  return (unsigned int)ROUND64_POWER_OF_TWO(
      (int64_t)vp9_high_get_sby_variance(cpi, ref, bs, bd),
      num_pels_log2_lookup[bs]);
}
#endif  // CONFIG_VP9_HIGHBITDEPTH

static void set_segment_index(VP9_COMP *cpi, MACROBLOCK *const x, int mi_row,
                              int mi_col, BLOCK_SIZE bsize, int segment_index) {
  VP9_COMMON *const cm = &cpi->common;
  const struct segmentation *const seg = &cm->seg;
  MACROBLOCKD *const xd = &x->e_mbd;
  MODE_INFO *mi = xd->mi[0];

  const AQ_MODE aq_mode = cpi->oxcf.aq_mode;
  const uint8_t *const map =
      seg->update_map ? cpi->segmentation_map : cm->last_frame_seg_map;

  // Initialize the segmentation index as 0.
  mi->segment_id = 0;

  // Skip the rest if AQ mode is disabled.
  if (!seg->enabled) return;

  switch (aq_mode) {
    case CYCLIC_REFRESH_AQ:
      mi->segment_id = get_segment_id(cm, map, bsize, mi_row, mi_col);
      break;
#if !CONFIG_REALTIME_ONLY
    case VARIANCE_AQ:
      if (cm->frame_type == KEY_FRAME || cpi->refresh_alt_ref_frame ||
          cpi->force_update_segmentation ||
          (cpi->refresh_golden_frame && !cpi->rc.is_src_frame_alt_ref)) {
        int min_energy;
        int max_energy;
        // Get sub block energy range
        if (bsize >= BLOCK_32X32) {
          vp9_get_sub_block_energy(cpi, x, mi_row, mi_col, bsize, &min_energy,
                                   &max_energy);
        } else {
          min_energy = bsize <= BLOCK_16X16 ? x->mb_energy
                                            : vp9_block_energy(cpi, x, bsize);
        }
        mi->segment_id = vp9_vaq_segment_id(min_energy);
      } else {
        mi->segment_id = get_segment_id(cm, map, bsize, mi_row, mi_col);
      }
      break;
    case EQUATOR360_AQ:
      if (cm->frame_type == KEY_FRAME || cpi->force_update_segmentation)
        mi->segment_id = vp9_360aq_segment_id(mi_row, cm->mi_rows);
      else
        mi->segment_id = get_segment_id(cm, map, bsize, mi_row, mi_col);
      break;
#endif
    case LOOKAHEAD_AQ:
      mi->segment_id = get_segment_id(cm, map, bsize, mi_row, mi_col);
      break;
    case PSNR_AQ: mi->segment_id = segment_index; break;
    case PERCEPTUAL_AQ: mi->segment_id = x->segment_id; break;
    default:
      // NO_AQ or PSNR_AQ
      break;
  }

  // Set segment index if ROI map or active_map is enabled.
  if (cpi->roi.enabled || cpi->active_map.enabled)
    mi->segment_id = get_segment_id(cm, map, bsize, mi_row, mi_col);

  vp9_init_plane_quantizers(cpi, x);
}

// Lighter version of set_offsets that only sets the mode info
// pointers.
static INLINE void set_mode_info_offsets(VP9_COMMON *const cm,
                                         MACROBLOCK *const x,
                                         MACROBLOCKD *const xd, int mi_row,
                                         int mi_col) {
  const int idx_str = xd->mi_stride * mi_row + mi_col;
  xd->mi = cm->mi_grid_visible + idx_str;
  xd->mi[0] = cm->mi + idx_str;
  x->mbmi_ext = x->mbmi_ext_base + (mi_row * cm->mi_cols + mi_col);
}

static void set_ssim_rdmult(VP9_COMP *const cpi, MACROBLOCK *const x,
                            const BLOCK_SIZE bsize, const int mi_row,
                            const int mi_col, int *const rdmult) {
  const VP9_COMMON *const cm = &cpi->common;

  const int bsize_base = BLOCK_16X16;
  const int num_8x8_w = num_8x8_blocks_wide_lookup[bsize_base];
  const int num_8x8_h = num_8x8_blocks_high_lookup[bsize_base];
  const int num_cols = (cm->mi_cols + num_8x8_w - 1) / num_8x8_w;
  const int num_rows = (cm->mi_rows + num_8x8_h - 1) / num_8x8_h;
  const int num_bcols =
      (num_8x8_blocks_wide_lookup[bsize] + num_8x8_w - 1) / num_8x8_w;
  const int num_brows =
      (num_8x8_blocks_high_lookup[bsize] + num_8x8_h - 1) / num_8x8_h;
  int row, col;
  double num_of_mi = 0.0;
  double geom_mean_of_scale = 0.0;

  assert(cpi->oxcf.tuning == VP8_TUNE_SSIM);

  for (row = mi_row / num_8x8_w;
       row < num_rows && row < mi_row / num_8x8_w + num_brows; ++row) {
    for (col = mi_col / num_8x8_h;
         col < num_cols && col < mi_col / num_8x8_h + num_bcols; ++col) {
      const int index = row * num_cols + col;
      geom_mean_of_scale += log(cpi->mi_ssim_rdmult_scaling_factors[index]);
      num_of_mi += 1.0;
    }
  }
  geom_mean_of_scale = exp(geom_mean_of_scale / num_of_mi);

  *rdmult = (int)((double)(*rdmult) * geom_mean_of_scale);
  *rdmult = VPXMAX(*rdmult, 0);
  set_error_per_bit(x, *rdmult);
  vpx_clear_system_state();
}

static void set_offsets(VP9_COMP *cpi, const TileInfo *const tile,
                        MACROBLOCK *const x, int mi_row, int mi_col,
                        BLOCK_SIZE bsize) {
  VP9_COMMON *const cm = &cpi->common;
  const VP9EncoderConfig *const oxcf = &cpi->oxcf;
  MACROBLOCKD *const xd = &x->e_mbd;
  const int mi_width = num_8x8_blocks_wide_lookup[bsize];
  const int mi_height = num_8x8_blocks_high_lookup[bsize];
  MvLimits *const mv_limits = &x->mv_limits;

  set_skip_context(xd, mi_row, mi_col);

  set_mode_info_offsets(cm, x, xd, mi_row, mi_col);

  // Set up destination pointers.
  vp9_setup_dst_planes(xd->plane, get_frame_new_buffer(cm), mi_row, mi_col);

  // Set up limit values for MV components.
  // Mv beyond the range do not produce new/different prediction block.
  mv_limits->row_min = -(((mi_row + mi_height) * MI_SIZE) + VP9_INTERP_EXTEND);
  mv_limits->col_min = -(((mi_col + mi_width) * MI_SIZE) + VP9_INTERP_EXTEND);
  mv_limits->row_max = (cm->mi_rows - mi_row) * MI_SIZE + VP9_INTERP_EXTEND;
  mv_limits->col_max = (cm->mi_cols - mi_col) * MI_SIZE + VP9_INTERP_EXTEND;

  // Set up distance of MB to edge of frame in 1/8th pel units.
  assert(!(mi_col & (mi_width - 1)) && !(mi_row & (mi_height - 1)));
  set_mi_row_col(xd, tile, mi_row, mi_height, mi_col, mi_width, cm->mi_rows,
                 cm->mi_cols);

  // Set up source buffers.
  vp9_setup_src_planes(x, cpi->Source, mi_row, mi_col);

  // R/D setup.
  x->rddiv = cpi->rd.RDDIV;
  x->rdmult = cpi->rd.RDMULT;
  if (oxcf->tuning == VP8_TUNE_SSIM) {
    set_ssim_rdmult(cpi, x, bsize, mi_row, mi_col, &x->rdmult);
  }

  // required by vp9_append_sub8x8_mvs_for_idx() and vp9_find_best_ref_mvs()
  xd->tile = *tile;
}

static void duplicate_mode_info_in_sb(VP9_COMMON *cm, MACROBLOCKD *xd,
                                      int mi_row, int mi_col,
                                      BLOCK_SIZE bsize) {
  const int block_width =
      VPXMIN(num_8x8_blocks_wide_lookup[bsize], cm->mi_cols - mi_col);
  const int block_height =
      VPXMIN(num_8x8_blocks_high_lookup[bsize], cm->mi_rows - mi_row);
  const int mi_stride = xd->mi_stride;
  MODE_INFO *const src_mi = xd->mi[0];
  int i, j;

  for (j = 0; j < block_height; ++j)
    for (i = 0; i < block_width; ++i) xd->mi[j * mi_stride + i] = src_mi;
}

static void set_block_size(VP9_COMP *const cpi, MACROBLOCK *const x,
                           MACROBLOCKD *const xd, int mi_row, int mi_col,
                           BLOCK_SIZE bsize) {
  if (cpi->common.mi_cols > mi_col && cpi->common.mi_rows > mi_row) {
    set_mode_info_offsets(&cpi->common, x, xd, mi_row, mi_col);
    xd->mi[0]->sb_type = bsize;
  }
}

typedef struct {
  // This struct is used for computing variance in choose_partitioning(), where
  // the max number of samples within a superblock is 16x16 (with 4x4 avg). Even
  // in high bitdepth, uint32_t is enough for sum_square_error (2^12 * 2^12 * 16
  // * 16 = 2^32).
  uint32_t sum_square_error;
  int32_t sum_error;
  int log2_count;
  int variance;
} Var;

typedef struct {
  Var none;
  Var horz[2];
  Var vert[2];
} partition_variance;

typedef struct {
  partition_variance part_variances;
  Var split[4];
} v4x4;

typedef struct {
  partition_variance part_variances;
  v4x4 split[4];
} v8x8;

typedef struct {
  partition_variance part_variances;
  v8x8 split[4];
} v16x16;

typedef struct {
  partition_variance part_variances;
  v16x16 split[4];
} v32x32;

typedef struct {
  partition_variance part_variances;
  v32x32 split[4];
} v64x64;

typedef struct {
  partition_variance *part_variances;
  Var *split[4];
} variance_node;

typedef enum {
  V16X16,
  V32X32,
  V64X64,
} TREE_LEVEL;

static void tree_to_node(void *data, BLOCK_SIZE bsize, variance_node *node) {
  int i;
  node->part_variances = NULL;
  switch (bsize) {
    case BLOCK_64X64: {
      v64x64 *vt = (v64x64 *)data;
      node->part_variances = &vt->part_variances;
      for (i = 0; i < 4; i++)
        node->split[i] = &vt->split[i].part_variances.none;
      break;
    }
    case BLOCK_32X32: {
      v32x32 *vt = (v32x32 *)data;
      node->part_variances = &vt->part_variances;
      for (i = 0; i < 4; i++)
        node->split[i] = &vt->split[i].part_variances.none;
      break;
    }
    case BLOCK_16X16: {
      v16x16 *vt = (v16x16 *)data;
      node->part_variances = &vt->part_variances;
      for (i = 0; i < 4; i++)
        node->split[i] = &vt->split[i].part_variances.none;
      break;
    }
    case BLOCK_8X8: {
      v8x8 *vt = (v8x8 *)data;
      node->part_variances = &vt->part_variances;
      for (i = 0; i < 4; i++)
        node->split[i] = &vt->split[i].part_variances.none;
      break;
    }
    default: {
      v4x4 *vt = (v4x4 *)data;
      assert(bsize == BLOCK_4X4);
      node->part_variances = &vt->part_variances;
      for (i = 0; i < 4; i++) node->split[i] = &vt->split[i];
      break;
    }
  }
}

// Set variance values given sum square error, sum error, count.
static void fill_variance(uint32_t s2, int32_t s, int c, Var *v) {
  v->sum_square_error = s2;
  v->sum_error = s;
  v->log2_count = c;
}

static void get_variance(Var *v) {
  v->variance =
      (int)(256 * (v->sum_square_error -
                   (uint32_t)(((int64_t)v->sum_error * v->sum_error) >>
                              v->log2_count)) >>
            v->log2_count);
}

static void sum_2_variances(const Var *a, const Var *b, Var *r) {
  assert(a->log2_count == b->log2_count);
  fill_variance(a->sum_square_error + b->sum_square_error,
                a->sum_error + b->sum_error, a->log2_count + 1, r);
}

static void fill_variance_tree(void *data, BLOCK_SIZE bsize) {
  variance_node node;
  memset(&node, 0, sizeof(node));
  tree_to_node(data, bsize, &node);
  sum_2_variances(node.split[0], node.split[1], &node.part_variances->horz[0]);
  sum_2_variances(node.split[2], node.split[3], &node.part_variances->horz[1]);
  sum_2_variances(node.split[0], node.split[2], &node.part_variances->vert[0]);
  sum_2_variances(node.split[1], node.split[3], &node.part_variances->vert[1]);
  sum_2_variances(&node.part_variances->vert[0], &node.part_variances->vert[1],
                  &node.part_variances->none);
}

static int set_vt_partitioning(VP9_COMP *cpi, MACROBLOCK *const x,
                               MACROBLOCKD *const xd, void *data,
                               BLOCK_SIZE bsize, int mi_row, int mi_col,
                               int64_t threshold, BLOCK_SIZE bsize_min,
                               int force_split) {
  VP9_COMMON *const cm = &cpi->common;
  variance_node vt;
  const int block_width = num_8x8_blocks_wide_lookup[bsize];
  const int block_height = num_8x8_blocks_high_lookup[bsize];

  assert(block_height == block_width);
  tree_to_node(data, bsize, &vt);

  if (force_split == 1) return 0;

  // For bsize=bsize_min (16x16/8x8 for 8x8/4x4 downsampling), select if
  // variance is below threshold, otherwise split will be selected.
  // No check for vert/horiz split as too few samples for variance.
  if (bsize == bsize_min) {
    // Variance already computed to set the force_split.
    if (frame_is_intra_only(cm)) get_variance(&vt.part_variances->none);
    if (mi_col + block_width / 2 < cm->mi_cols &&
        mi_row + block_height / 2 < cm->mi_rows &&
        vt.part_variances->none.variance < threshold) {
      set_block_size(cpi, x, xd, mi_row, mi_col, bsize);
      return 1;
    }
    return 0;
  } else if (bsize > bsize_min) {
    // Variance already computed to set the force_split.
    if (frame_is_intra_only(cm)) get_variance(&vt.part_variances->none);
    // For key frame: take split for bsize above 32X32 or very high variance.
    if (frame_is_intra_only(cm) &&
        (bsize > BLOCK_32X32 ||
         vt.part_variances->none.variance > (threshold << 4))) {
      return 0;
    }
    // If variance is low, take the bsize (no split).
    if (mi_col + block_width / 2 < cm->mi_cols &&
        mi_row + block_height / 2 < cm->mi_rows &&
        vt.part_variances->none.variance < threshold) {
      set_block_size(cpi, x, xd, mi_row, mi_col, bsize);
      return 1;
    }

    // Check vertical split.
    if (mi_row + block_height / 2 < cm->mi_rows) {
      BLOCK_SIZE subsize = get_subsize(bsize, PARTITION_VERT);
      get_variance(&vt.part_variances->vert[0]);
      get_variance(&vt.part_variances->vert[1]);
      if (vt.part_variances->vert[0].variance < threshold &&
          vt.part_variances->vert[1].variance < threshold &&
          get_plane_block_size(subsize, &xd->plane[1]) < BLOCK_INVALID) {
        set_block_size(cpi, x, xd, mi_row, mi_col, subsize);
        set_block_size(cpi, x, xd, mi_row, mi_col + block_width / 2, subsize);
        return 1;
      }
    }
    // Check horizontal split.
    if (mi_col + block_width / 2 < cm->mi_cols) {
      BLOCK_SIZE subsize = get_subsize(bsize, PARTITION_HORZ);
      get_variance(&vt.part_variances->horz[0]);
      get_variance(&vt.part_variances->horz[1]);
      if (vt.part_variances->horz[0].variance < threshold &&
          vt.part_variances->horz[1].variance < threshold &&
          get_plane_block_size(subsize, &xd->plane[1]) < BLOCK_INVALID) {
        set_block_size(cpi, x, xd, mi_row, mi_col, subsize);
        set_block_size(cpi, x, xd, mi_row + block_height / 2, mi_col, subsize);
        return 1;
      }
    }

    return 0;
  }
  return 0;
}

static int64_t scale_part_thresh_sumdiff(int64_t threshold_base, int speed,
                                         int width, int height,
                                         int content_state) {
  if (speed >= 8) {
    if (width <= 640 && height <= 480)
      return (5 * threshold_base) >> 2;
    else if ((content_state == kLowSadLowSumdiff) ||
             (content_state == kHighSadLowSumdiff) ||
             (content_state == kLowVarHighSumdiff))
      return (5 * threshold_base) >> 2;
  } else if (speed == 7) {
    if ((content_state == kLowSadLowSumdiff) ||
        (content_state == kHighSadLowSumdiff) ||
        (content_state == kLowVarHighSumdiff)) {
      return (5 * threshold_base) >> 2;
    }
  }
  return threshold_base;
}

// Set the variance split thresholds for following the block sizes:
// 0 - threshold_64x64, 1 - threshold_32x32, 2 - threshold_16x16,
// 3 - vbp_threshold_8x8. vbp_threshold_8x8 (to split to 4x4 partition) is
// currently only used on key frame.
static void set_vbp_thresholds(VP9_COMP *cpi, int64_t thresholds[], int q,
                               int content_state) {
  VP9_COMMON *const cm = &cpi->common;
  const int is_key_frame = frame_is_intra_only(cm);
  const int threshold_multiplier =
      is_key_frame ? 20 : cpi->sf.variance_part_thresh_mult;
  int64_t threshold_base =
      (int64_t)(threshold_multiplier * cpi->y_dequant[q][1]);

  if (is_key_frame) {
    thresholds[0] = threshold_base;
    thresholds[1] = threshold_base >> 2;
    thresholds[2] = threshold_base >> 2;
    thresholds[3] = threshold_base << 2;
  } else {
    // Increase base variance threshold based on estimated noise level.
    if (cpi->noise_estimate.enabled && cm->width >= 640 && cm->height >= 480) {
      NOISE_LEVEL noise_level =
          vp9_noise_estimate_extract_level(&cpi->noise_estimate);
      if (noise_level == kHigh)
        threshold_base = 3 * threshold_base;
      else if (noise_level == kMedium)
        threshold_base = threshold_base << 1;
      else if (noise_level < kLow)
        threshold_base = (7 * threshold_base) >> 3;
    }
#if CONFIG_VP9_TEMPORAL_DENOISING
    if (cpi->oxcf.noise_sensitivity > 0 && denoise_svc(cpi) &&
        cpi->oxcf.speed > 5 && cpi->denoiser.denoising_level >= kDenLow)
      threshold_base =
          vp9_scale_part_thresh(threshold_base, cpi->denoiser.denoising_level,
                                content_state, cpi->svc.temporal_layer_id);
    else
      threshold_base =
          scale_part_thresh_sumdiff(threshold_base, cpi->oxcf.speed, cm->width,
                                    cm->height, content_state);
#else
    // Increase base variance threshold based on content_state/sum_diff level.
    threshold_base = scale_part_thresh_sumdiff(
        threshold_base, cpi->oxcf.speed, cm->width, cm->height, content_state);
#endif
    thresholds[0] = threshold_base;
    thresholds[2] = threshold_base << cpi->oxcf.speed;
    if (cm->width >= 1280 && cm->height >= 720 && cpi->oxcf.speed < 7)
      thresholds[2] = thresholds[2] << 1;
    if (cm->width <= 352 && cm->height <= 288) {
      thresholds[0] = threshold_base >> 3;
      thresholds[1] = threshold_base >> 1;
      thresholds[2] = threshold_base << 3;
      if (cpi->rc.avg_frame_qindex[INTER_FRAME] > 220)
        thresholds[2] = thresholds[2] << 2;
      else if (cpi->rc.avg_frame_qindex[INTER_FRAME] > 200)
        thresholds[2] = thresholds[2] << 1;
    } else if (cm->width < 1280 && cm->height < 720) {
      thresholds[1] = (5 * threshold_base) >> 2;
    } else if (cm->width < 1920 && cm->height < 1080) {
      thresholds[1] = threshold_base << 1;
    } else {
      thresholds[1] = (5 * threshold_base) >> 1;
    }
    if (cpi->sf.disable_16x16part_nonkey) thresholds[2] = INT64_MAX;
  }
}

void vp9_set_variance_partition_thresholds(VP9_COMP *cpi, int q,
                                           int content_state) {
  VP9_COMMON *const cm = &cpi->common;
  SPEED_FEATURES *const sf = &cpi->sf;
  const int is_key_frame = frame_is_intra_only(cm);
  if (sf->partition_search_type != VAR_BASED_PARTITION &&
      sf->partition_search_type != REFERENCE_PARTITION) {
    return;
  } else {
    set_vbp_thresholds(cpi, cpi->vbp_thresholds, q, content_state);
    // The thresholds below are not changed locally.
    if (is_key_frame) {
      cpi->vbp_threshold_sad = 0;
      cpi->vbp_threshold_copy = 0;
      cpi->vbp_bsize_min = BLOCK_8X8;
    } else {
      if (cm->width <= 352 && cm->height <= 288)
        cpi->vbp_threshold_sad = 10;
      else
        cpi->vbp_threshold_sad = (cpi->y_dequant[q][1] << 1) > 1000
                                     ? (cpi->y_dequant[q][1] << 1)
                                     : 1000;
      cpi->vbp_bsize_min = BLOCK_16X16;
      if (cm->width <= 352 && cm->height <= 288)
        cpi->vbp_threshold_copy = 4000;
      else if (cm->width <= 640 && cm->height <= 360)
        cpi->vbp_threshold_copy = 8000;
      else
        cpi->vbp_threshold_copy = (cpi->y_dequant[q][1] << 3) > 8000
                                      ? (cpi->y_dequant[q][1] << 3)
                                      : 8000;
      if (cpi->rc.high_source_sad ||
          (cpi->use_svc && cpi->svc.high_source_sad_superframe)) {
        cpi->vbp_threshold_sad = 0;
        cpi->vbp_threshold_copy = 0;
      }
    }
    cpi->vbp_threshold_minmax = 15 + (q >> 3);
  }
}

// Compute the minmax over the 8x8 subblocks.
static int compute_minmax_8x8(const uint8_t *s, int sp, const uint8_t *d,
                              int dp, int x16_idx, int y16_idx,
#if CONFIG_VP9_HIGHBITDEPTH
                              int highbd_flag,
#endif
                              int pixels_wide, int pixels_high) {
  int k;
  int minmax_max = 0;
  int minmax_min = 255;
  // Loop over the 4 8x8 subblocks.
  for (k = 0; k < 4; k++) {
    int x8_idx = x16_idx + ((k & 1) << 3);
    int y8_idx = y16_idx + ((k >> 1) << 3);
    int min = 0;
    int max = 0;
    if (x8_idx < pixels_wide && y8_idx < pixels_high) {
#if CONFIG_VP9_HIGHBITDEPTH
      if (highbd_flag & YV12_FLAG_HIGHBITDEPTH) {
        vpx_highbd_minmax_8x8(s + y8_idx * sp + x8_idx, sp,
                              d + y8_idx * dp + x8_idx, dp, &min, &max);
      } else {
        vpx_minmax_8x8(s + y8_idx * sp + x8_idx, sp, d + y8_idx * dp + x8_idx,
                       dp, &min, &max);
      }
#else
      vpx_minmax_8x8(s + y8_idx * sp + x8_idx, sp, d + y8_idx * dp + x8_idx, dp,
                     &min, &max);
#endif
      if ((max - min) > minmax_max) minmax_max = (max - min);
      if ((max - min) < minmax_min) minmax_min = (max - min);
    }
  }
  return (minmax_max - minmax_min);
}

static void fill_variance_4x4avg(const uint8_t *s, int sp, const uint8_t *d,
                                 int dp, int x8_idx, int y8_idx, v8x8 *vst,
#if CONFIG_VP9_HIGHBITDEPTH
                                 int highbd_flag,
#endif
                                 int pixels_wide, int pixels_high,
                                 int is_key_frame) {
  int k;
  for (k = 0; k < 4; k++) {
    int x4_idx = x8_idx + ((k & 1) << 2);
    int y4_idx = y8_idx + ((k >> 1) << 2);
    unsigned int sse = 0;
    int sum = 0;
    if (x4_idx < pixels_wide && y4_idx < pixels_high) {
      int s_avg;
      int d_avg = 128;
#if CONFIG_VP9_HIGHBITDEPTH
      if (highbd_flag & YV12_FLAG_HIGHBITDEPTH) {
        s_avg = vpx_highbd_avg_4x4(s + y4_idx * sp + x4_idx, sp);
        if (!is_key_frame)
          d_avg = vpx_highbd_avg_4x4(d + y4_idx * dp + x4_idx, dp);
      } else {
        s_avg = vpx_avg_4x4(s + y4_idx * sp + x4_idx, sp);
        if (!is_key_frame) d_avg = vpx_avg_4x4(d + y4_idx * dp + x4_idx, dp);
      }
#else
      s_avg = vpx_avg_4x4(s + y4_idx * sp + x4_idx, sp);
      if (!is_key_frame) d_avg = vpx_avg_4x4(d + y4_idx * dp + x4_idx, dp);
#endif
      sum = s_avg - d_avg;
      sse = sum * sum;
    }
    fill_variance(sse, sum, 0, &vst->split[k].part_variances.none);
  }
}

static void fill_variance_8x8avg(const uint8_t *s, int sp, const uint8_t *d,
                                 int dp, int x16_idx, int y16_idx, v16x16 *vst,
#if CONFIG_VP9_HIGHBITDEPTH
                                 int highbd_flag,
#endif
                                 int pixels_wide, int pixels_high,
                                 int is_key_frame) {
  int k;
  for (k = 0; k < 4; k++) {
    int x8_idx = x16_idx + ((k & 1) << 3);
    int y8_idx = y16_idx + ((k >> 1) << 3);
    unsigned int sse = 0;
    int sum = 0;
    if (x8_idx < pixels_wide && y8_idx < pixels_high) {
      int s_avg;
      int d_avg = 128;
#if CONFIG_VP9_HIGHBITDEPTH
      if (highbd_flag & YV12_FLAG_HIGHBITDEPTH) {
        s_avg = vpx_highbd_avg_8x8(s + y8_idx * sp + x8_idx, sp);
        if (!is_key_frame)
          d_avg = vpx_highbd_avg_8x8(d + y8_idx * dp + x8_idx, dp);
      } else {
        s_avg = vpx_avg_8x8(s + y8_idx * sp + x8_idx, sp);
        if (!is_key_frame) d_avg = vpx_avg_8x8(d + y8_idx * dp + x8_idx, dp);
      }
#else
      s_avg = vpx_avg_8x8(s + y8_idx * sp + x8_idx, sp);
      if (!is_key_frame) d_avg = vpx_avg_8x8(d + y8_idx * dp + x8_idx, dp);
#endif
      sum = s_avg - d_avg;
      sse = sum * sum;
    }
    fill_variance(sse, sum, 0, &vst->split[k].part_variances.none);
  }
}

// Check if most of the superblock is skin content, and if so, force split to
// 32x32, and set x->sb_is_skin for use in mode selection.
static int skin_sb_split(VP9_COMP *cpi, const int low_res, int mi_row,
                         int mi_col, int *force_split) {
  VP9_COMMON *const cm = &cpi->common;
#if CONFIG_VP9_HIGHBITDEPTH
  if (cm->use_highbitdepth) return 0;
#endif
  // Avoid checking superblocks on/near boundary and avoid low resolutions.
  // Note superblock may still pick 64X64 if y_sad is very small
  // (i.e., y_sad < cpi->vbp_threshold_sad) below. For now leave this as is.
  if (!low_res && (mi_col >= 8 && mi_col + 8 < cm->mi_cols && mi_row >= 8 &&
                   mi_row + 8 < cm->mi_rows)) {
    int num_16x16_skin = 0;
    int num_16x16_nonskin = 0;
    const int block_index = mi_row * cm->mi_cols + mi_col;
    const int bw = num_8x8_blocks_wide_lookup[BLOCK_64X64];
    const int bh = num_8x8_blocks_high_lookup[BLOCK_64X64];
    const int xmis = VPXMIN(cm->mi_cols - mi_col, bw);
    const int ymis = VPXMIN(cm->mi_rows - mi_row, bh);
    // Loop through the 16x16 sub-blocks.
    int i, j;
    for (i = 0; i < ymis; i += 2) {
      for (j = 0; j < xmis; j += 2) {
        int bl_index = block_index + i * cm->mi_cols + j;
        int is_skin = cpi->skin_map[bl_index];
        num_16x16_skin += is_skin;
        num_16x16_nonskin += (1 - is_skin);
        if (num_16x16_nonskin > 3) {
          // Exit loop if at least 4 of the 16x16 blocks are not skin.
          i = ymis;
          break;
        }
      }
    }
    if (num_16x16_skin > 12) {
      *force_split = 1;
      return 1;
    }
  }
  return 0;
}

static void set_low_temp_var_flag(VP9_COMP *cpi, MACROBLOCK *x, MACROBLOCKD *xd,
                                  v64x64 *vt, int64_t thresholds[],
                                  MV_REFERENCE_FRAME ref_frame_partition,
                                  int mi_col, int mi_row) {
  int i, j;
  VP9_COMMON *const cm = &cpi->common;
  const int mv_thr = cm->width > 640 ? 8 : 4;
  // Check temporal variance for bsize >= 16x16, if LAST_FRAME was selected and
  // int_pro mv is small. If the temporal variance is small set the flag
  // variance_low for the block. The variance threshold can be adjusted, the
  // higher the more aggressive.
  if (ref_frame_partition == LAST_FRAME &&
      (cpi->sf.short_circuit_low_temp_var == 1 ||
       (xd->mi[0]->mv[0].as_mv.col < mv_thr &&
        xd->mi[0]->mv[0].as_mv.col > -mv_thr &&
        xd->mi[0]->mv[0].as_mv.row < mv_thr &&
        xd->mi[0]->mv[0].as_mv.row > -mv_thr))) {
    if (xd->mi[0]->sb_type == BLOCK_64X64) {
      if ((vt->part_variances).none.variance < (thresholds[0] >> 1))
        x->variance_low[0] = 1;
    } else if (xd->mi[0]->sb_type == BLOCK_64X32) {
      for (i = 0; i < 2; i++) {
        if (vt->part_variances.horz[i].variance < (thresholds[0] >> 2))
          x->variance_low[i + 1] = 1;
      }
    } else if (xd->mi[0]->sb_type == BLOCK_32X64) {
      for (i = 0; i < 2; i++) {
        if (vt->part_variances.vert[i].variance < (thresholds[0] >> 2))
          x->variance_low[i + 3] = 1;
      }
    } else {
      for (i = 0; i < 4; i++) {
        const int idx[4][2] = { { 0, 0 }, { 0, 4 }, { 4, 0 }, { 4, 4 } };
        const int idx_str =
            cm->mi_stride * (mi_row + idx[i][0]) + mi_col + idx[i][1];
        MODE_INFO **this_mi = cm->mi_grid_visible + idx_str;

        if (cm->mi_cols <= mi_col + idx[i][1] ||
            cm->mi_rows <= mi_row + idx[i][0])
          continue;

        if ((*this_mi)->sb_type == BLOCK_32X32) {
          int64_t threshold_32x32 = (cpi->sf.short_circuit_low_temp_var == 1 ||
                                     cpi->sf.short_circuit_low_temp_var == 3)
                                        ? ((5 * thresholds[1]) >> 3)
                                        : (thresholds[1] >> 1);
          if (vt->split[i].part_variances.none.variance < threshold_32x32)
            x->variance_low[i + 5] = 1;
        } else if (cpi->sf.short_circuit_low_temp_var >= 2) {
          // For 32x16 and 16x32 blocks, the flag is set on each 16x16 block
          // inside.
          if ((*this_mi)->sb_type == BLOCK_16X16 ||
              (*this_mi)->sb_type == BLOCK_32X16 ||
              (*this_mi)->sb_type == BLOCK_16X32) {
            for (j = 0; j < 4; j++) {
              if (vt->split[i].split[j].part_variances.none.variance <
                  (thresholds[2] >> 8))
                x->variance_low[(i << 2) + j + 9] = 1;
            }
          }
        }
      }
    }
  }
}

static void copy_partitioning_helper(VP9_COMP *cpi, MACROBLOCK *x,
                                     MACROBLOCKD *xd, BLOCK_SIZE bsize,
                                     int mi_row, int mi_col) {
  VP9_COMMON *const cm = &cpi->common;
  BLOCK_SIZE *prev_part = cpi->prev_partition;
  int start_pos = mi_row * cm->mi_stride + mi_col;

  const int bsl = b_width_log2_lookup[bsize];
  const int bs = (1 << bsl) >> 2;
  BLOCK_SIZE subsize;
  PARTITION_TYPE partition;

  if (mi_row >= cm->mi_rows || mi_col >= cm->mi_cols) return;

  partition = partition_lookup[bsl][prev_part[start_pos]];
  subsize = get_subsize(bsize, partition);

  if (subsize < BLOCK_8X8) {
    set_block_size(cpi, x, xd, mi_row, mi_col, bsize);
  } else {
    switch (partition) {
      case PARTITION_NONE:
        set_block_size(cpi, x, xd, mi_row, mi_col, bsize);
        break;
      case PARTITION_HORZ:
        set_block_size(cpi, x, xd, mi_row, mi_col, subsize);
        set_block_size(cpi, x, xd, mi_row + bs, mi_col, subsize);
        break;
      case PARTITION_VERT:
        set_block_size(cpi, x, xd, mi_row, mi_col, subsize);
        set_block_size(cpi, x, xd, mi_row, mi_col + bs, subsize);
        break;
      default:
        assert(partition == PARTITION_SPLIT);
        copy_partitioning_helper(cpi, x, xd, subsize, mi_row, mi_col);
        copy_partitioning_helper(cpi, x, xd, subsize, mi_row + bs, mi_col);
        copy_partitioning_helper(cpi, x, xd, subsize, mi_row, mi_col + bs);
        copy_partitioning_helper(cpi, x, xd, subsize, mi_row + bs, mi_col + bs);
        break;
    }
  }
}

static int copy_partitioning(VP9_COMP *cpi, MACROBLOCK *x, MACROBLOCKD *xd,
                             int mi_row, int mi_col, int segment_id,
                             int sb_offset) {
  int svc_copy_allowed = 1;
  int frames_since_key_thresh = 1;
  if (cpi->use_svc) {
    // For SVC, don't allow copy if base spatial layer is key frame, or if
    // frame is not a temporal enhancement layer frame.
    int layer = LAYER_IDS_TO_IDX(0, cpi->svc.temporal_layer_id,
                                 cpi->svc.number_temporal_layers);
    const LAYER_CONTEXT *lc = &cpi->svc.layer_context[layer];
    if (lc->is_key_frame || !cpi->svc.non_reference_frame) svc_copy_allowed = 0;
    frames_since_key_thresh = cpi->svc.number_spatial_layers << 1;
  }
  if (cpi->rc.frames_since_key > frames_since_key_thresh && svc_copy_allowed &&
      !cpi->resize_pending && segment_id == CR_SEGMENT_ID_BASE &&
      cpi->prev_segment_id[sb_offset] == CR_SEGMENT_ID_BASE &&
      cpi->copied_frame_cnt[sb_offset] < cpi->max_copied_frame) {
    if (cpi->prev_partition != NULL) {
      copy_partitioning_helper(cpi, x, xd, BLOCK_64X64, mi_row, mi_col);
      cpi->copied_frame_cnt[sb_offset] += 1;
      memcpy(x->variance_low, &(cpi->prev_variance_low[sb_offset * 25]),
             sizeof(x->variance_low));
      return 1;
    }
  }

  return 0;
}

// Set the partition for mi_col/row_high (current resolution) based on
// the previous spatial layer (mi_col/row). Returns 0 if partition is set,
// returns 1 if no scale partitioning is done. Return 1 means the variance
// partitioning will be used.
static int scale_partitioning_svc(VP9_COMP *cpi, MACROBLOCK *x, MACROBLOCKD *xd,
                                  BLOCK_SIZE bsize, int mi_row, int mi_col,
                                  int mi_row_high, int mi_col_high) {
  VP9_COMMON *const cm = &cpi->common;
  SVC *const svc = &cpi->svc;
  BLOCK_SIZE *prev_part = svc->prev_partition_svc;
  // Variables with _high are for higher resolution.
  int bsize_high = 0;
  int subsize_high = 0;
  const int bsl = b_width_log2_lookup[bsize];
  const int bs = (1 << bsl) >> 2;
  const int has_rows = (mi_row_high + bs) < cm->mi_rows;
  const int has_cols = (mi_col_high + bs) < cm->mi_cols;

  int start_pos;
  BLOCK_SIZE bsize_low;
  PARTITION_TYPE partition_high;

  // If the lower layer frame is outside the boundary (this can happen for
  // odd size resolutions) then do not scale partitioning from the lower
  // layer. Do variance based partitioning instead (return 1).
  if (mi_row >= svc->mi_rows[svc->spatial_layer_id - 1] ||
      mi_col >= svc->mi_cols[svc->spatial_layer_id - 1])
    return 1;

  // Do not scale partitioning from lower layers on the boundary. Do
  // variance based partitioning instead (return 1).
  if (!has_rows || !has_cols) return 1;

  // Find corresponding (mi_col/mi_row) block down-scaled by 2x2.
  start_pos = mi_row * (svc->mi_stride[svc->spatial_layer_id - 1]) + mi_col;
  bsize_low = prev_part[start_pos];

  // For reference frames: return 1 (do variance-based partitioning) if the
  // superblock is not low source sad and lower-resoln bsize is below 32x32.
  if (!cpi->svc.non_reference_frame && !x->skip_low_source_sad &&
      bsize_low < BLOCK_32X32)
    return 1;

  // Scale up block size by 2x2. Force 64x64 for size larger than 32x32.
  if (bsize_low < BLOCK_32X32) {
    bsize_high = bsize_low + 3;
  } else if (bsize_low >= BLOCK_32X32) {
    bsize_high = BLOCK_64X64;
  }

  partition_high = partition_lookup[bsl][bsize_high];
  subsize_high = get_subsize(bsize, partition_high);

  if (subsize_high < BLOCK_8X8) {
    set_block_size(cpi, x, xd, mi_row_high, mi_col_high, bsize_high);
  } else {
    switch (partition_high) {
      case PARTITION_NONE:
        set_block_size(cpi, x, xd, mi_row_high, mi_col_high, bsize_high);
        break;
      case PARTITION_HORZ:
        set_block_size(cpi, x, xd, mi_row_high, mi_col_high, subsize_high);
        if (subsize_high < BLOCK_64X64)
          set_block_size(cpi, x, xd, mi_row_high + bs, mi_col_high,
                         subsize_high);
        break;
      case PARTITION_VERT:
        set_block_size(cpi, x, xd, mi_row_high, mi_col_high, subsize_high);
        if (subsize_high < BLOCK_64X64)
          set_block_size(cpi, x, xd, mi_row_high, mi_col_high + bs,
                         subsize_high);
        break;
      default:
        assert(partition_high == PARTITION_SPLIT);
        if (scale_partitioning_svc(cpi, x, xd, subsize_high, mi_row, mi_col,
                                   mi_row_high, mi_col_high))
          return 1;
        if (scale_partitioning_svc(cpi, x, xd, subsize_high, mi_row + (bs >> 1),
                                   mi_col, mi_row_high + bs, mi_col_high))
          return 1;
        if (scale_partitioning_svc(cpi, x, xd, subsize_high, mi_row,
                                   mi_col + (bs >> 1), mi_row_high,
                                   mi_col_high + bs))
          return 1;
        if (scale_partitioning_svc(cpi, x, xd, subsize_high, mi_row + (bs >> 1),
                                   mi_col + (bs >> 1), mi_row_high + bs,
                                   mi_col_high + bs))
          return 1;
        break;
    }
  }

  return 0;
}

static void update_partition_svc(VP9_COMP *cpi, BLOCK_SIZE bsize, int mi_row,
                                 int mi_col) {
  VP9_COMMON *const cm = &cpi->common;
  BLOCK_SIZE *prev_part = cpi->svc.prev_partition_svc;
  int start_pos = mi_row * cm->mi_stride + mi_col;
  const int bsl = b_width_log2_lookup[bsize];
  const int bs = (1 << bsl) >> 2;
  BLOCK_SIZE subsize;
  PARTITION_TYPE partition;
  const MODE_INFO *mi = NULL;
  int xx, yy;

  if (mi_row >= cm->mi_rows || mi_col >= cm->mi_cols) return;

  mi = cm->mi_grid_visible[start_pos];
  partition = partition_lookup[bsl][mi->sb_type];
  subsize = get_subsize(bsize, partition);
  if (subsize < BLOCK_8X8) {
    prev_part[start_pos] = bsize;
  } else {
    switch (partition) {
      case PARTITION_NONE:
        prev_part[start_pos] = bsize;
        if (bsize == BLOCK_64X64) {
          for (xx = 0; xx < 8; xx += 4)
            for (yy = 0; yy < 8; yy += 4) {
              if ((mi_row + xx < cm->mi_rows) && (mi_col + yy < cm->mi_cols))
                prev_part[start_pos + xx * cm->mi_stride + yy] = bsize;
            }
        }
        break;
      case PARTITION_HORZ:
        prev_part[start_pos] = subsize;
        if (mi_row + bs < cm->mi_rows)
          prev_part[start_pos + bs * cm->mi_stride] = subsize;
        break;
      case PARTITION_VERT:
        prev_part[start_pos] = subsize;
        if (mi_col + bs < cm->mi_cols) prev_part[start_pos + bs] = subsize;
        break;
      default:
        assert(partition == PARTITION_SPLIT);
        update_partition_svc(cpi, subsize, mi_row, mi_col);
        update_partition_svc(cpi, subsize, mi_row + bs, mi_col);
        update_partition_svc(cpi, subsize, mi_row, mi_col + bs);
        update_partition_svc(cpi, subsize, mi_row + bs, mi_col + bs);
        break;
    }
  }
}

static void update_prev_partition_helper(VP9_COMP *cpi, BLOCK_SIZE bsize,
                                         int mi_row, int mi_col) {
  VP9_COMMON *const cm = &cpi->common;
  BLOCK_SIZE *prev_part = cpi->prev_partition;
  int start_pos = mi_row * cm->mi_stride + mi_col;
  const int bsl = b_width_log2_lookup[bsize];
  const int bs = (1 << bsl) >> 2;
  BLOCK_SIZE subsize;
  PARTITION_TYPE partition;
  const MODE_INFO *mi = NULL;

  if (mi_row >= cm->mi_rows || mi_col >= cm->mi_cols) return;

  mi = cm->mi_grid_visible[start_pos];
  partition = partition_lookup[bsl][mi->sb_type];
  subsize = get_subsize(bsize, partition);
  if (subsize < BLOCK_8X8) {
    prev_part[start_pos] = bsize;
  } else {
    switch (partition) {
      case PARTITION_NONE: prev_part[start_pos] = bsize; break;
      case PARTITION_HORZ:
        prev_part[start_pos] = subsize;
        if (mi_row + bs < cm->mi_rows)
          prev_part[start_pos + bs * cm->mi_stride] = subsize;
        break;
      case PARTITION_VERT:
        prev_part[start_pos] = subsize;
        if (mi_col + bs < cm->mi_cols) prev_part[start_pos + bs] = subsize;
        break;
      default:
        assert(partition == PARTITION_SPLIT);
        update_prev_partition_helper(cpi, subsize, mi_row, mi_col);
        update_prev_partition_helper(cpi, subsize, mi_row + bs, mi_col);
        update_prev_partition_helper(cpi, subsize, mi_row, mi_col + bs);
        update_prev_partition_helper(cpi, subsize, mi_row + bs, mi_col + bs);
        break;
    }
  }
}

static void update_prev_partition(VP9_COMP *cpi, MACROBLOCK *x, int segment_id,
                                  int mi_row, int mi_col, int sb_offset) {
  update_prev_partition_helper(cpi, BLOCK_64X64, mi_row, mi_col);
  cpi->prev_segment_id[sb_offset] = segment_id;
  memcpy(&(cpi->prev_variance_low[sb_offset * 25]), x->variance_low,
         sizeof(x->variance_low));
  // Reset the counter for copy partitioning
  cpi->copied_frame_cnt[sb_offset] = 0;
}

static void chroma_check(VP9_COMP *cpi, MACROBLOCK *x, int bsize,
                         unsigned int y_sad, int is_key_frame,
                         int scene_change_detected) {
  int i;
  MACROBLOCKD *xd = &x->e_mbd;
  int shift = 2;

  if (is_key_frame) return;

  // For speed > 8, avoid the chroma check if y_sad is above threshold.
  if (cpi->oxcf.speed > 8) {
    if (y_sad > cpi->vbp_thresholds[1] &&
        (!cpi->noise_estimate.enabled ||
         vp9_noise_estimate_extract_level(&cpi->noise_estimate) < kMedium))
      return;
  }

  if (cpi->oxcf.content == VP9E_CONTENT_SCREEN && scene_change_detected)
    shift = 5;

  for (i = 1; i <= 2; ++i) {
    unsigned int uv_sad = UINT_MAX;
    struct macroblock_plane *p = &x->plane[i];
    struct macroblockd_plane *pd = &xd->plane[i];
    const BLOCK_SIZE bs = get_plane_block_size(bsize, pd);

    if (bs != BLOCK_INVALID)
      uv_sad = cpi->fn_ptr[bs].sdf(p->src.buf, p->src.stride, pd->dst.buf,
                                   pd->dst.stride);

    // TODO(marpan): Investigate if we should lower this threshold if
    // superblock is detected as skin.
    x->color_sensitivity[i - 1] = uv_sad > (y_sad >> shift);
  }
}

static uint64_t avg_source_sad(VP9_COMP *cpi, MACROBLOCK *x, int shift,
                               int sb_offset) {
  unsigned int tmp_sse;
  uint64_t tmp_sad;
  unsigned int tmp_variance;
  const BLOCK_SIZE bsize = BLOCK_64X64;
  uint8_t *src_y = cpi->Source->y_buffer;
  int src_ystride = cpi->Source->y_stride;
  uint8_t *last_src_y = cpi->Last_Source->y_buffer;
  int last_src_ystride = cpi->Last_Source->y_stride;
  uint64_t avg_source_sad_threshold = 10000;
  uint64_t avg_source_sad_threshold2 = 12000;
#if CONFIG_VP9_HIGHBITDEPTH
  if (cpi->common.use_highbitdepth) return 0;
#endif
  src_y += shift;
  last_src_y += shift;
  tmp_sad =
      cpi->fn_ptr[bsize].sdf(src_y, src_ystride, last_src_y, last_src_ystride);
  tmp_variance = vpx_variance64x64(src_y, src_ystride, last_src_y,
                                   last_src_ystride, &tmp_sse);
  // Note: tmp_sse - tmp_variance = ((sum * sum) >> 12)
  if (tmp_sad < avg_source_sad_threshold)
    x->content_state_sb = ((tmp_sse - tmp_variance) < 25) ? kLowSadLowSumdiff
                                                          : kLowSadHighSumdiff;
  else
    x->content_state_sb = ((tmp_sse - tmp_variance) < 25) ? kHighSadLowSumdiff
                                                          : kHighSadHighSumdiff;

  // Detect large lighting change.
  if (cpi->oxcf.content != VP9E_CONTENT_SCREEN &&
      cpi->oxcf.rc_mode == VPX_CBR && tmp_variance < (tmp_sse >> 3) &&
      (tmp_sse - tmp_variance) > 10000)
    x->content_state_sb = kLowVarHighSumdiff;
  else if (tmp_sad > (avg_source_sad_threshold << 1))
    x->content_state_sb = kVeryHighSad;

  if (cpi->content_state_sb_fd != NULL) {
    if (tmp_sad < avg_source_sad_threshold2) {
      // Cap the increment to 255.
      if (cpi->content_state_sb_fd[sb_offset] < 255)
        cpi->content_state_sb_fd[sb_offset]++;
    } else {
      cpi->content_state_sb_fd[sb_offset] = 0;
    }
  }
  if (tmp_sad == 0) x->zero_temp_sad_source = 1;
  return tmp_sad;
}

// This function chooses partitioning based on the variance between source and
// reconstructed last, where variance is computed for down-sampled inputs.
static int choose_partitioning(VP9_COMP *cpi, const TileInfo *const tile,
                               MACROBLOCK *x, int mi_row, int mi_col) {
  VP9_COMMON *const cm = &cpi->common;
  MACROBLOCKD *xd = &x->e_mbd;
  int i, j, k, m;
  v64x64 vt;
  v16x16 *vt2 = NULL;
  int force_split[21];
  int avg_32x32;
  int max_var_32x32 = 0;
  int min_var_32x32 = INT_MAX;
  int var_32x32;
  int avg_16x16[4];
  int maxvar_16x16[4];
  int minvar_16x16[4];
  int64_t threshold_4x4avg;
  NOISE_LEVEL noise_level = kLow;
  int content_state = 0;
  uint8_t *s;
  const uint8_t *d;
  int sp;
  int dp;
  int compute_minmax_variance = 1;
  unsigned int y_sad = UINT_MAX;
  BLOCK_SIZE bsize = BLOCK_64X64;
  // Ref frame used in partitioning.
  MV_REFERENCE_FRAME ref_frame_partition = LAST_FRAME;
  int pixels_wide = 64, pixels_high = 64;
  int64_t thresholds[4] = { cpi->vbp_thresholds[0], cpi->vbp_thresholds[1],
                            cpi->vbp_thresholds[2], cpi->vbp_thresholds[3] };
  int scene_change_detected =
      cpi->rc.high_source_sad ||
      (cpi->use_svc && cpi->svc.high_source_sad_superframe);
  int force_64_split = scene_change_detected ||
                       (cpi->oxcf.content == VP9E_CONTENT_SCREEN &&
                        cpi->compute_source_sad_onepass &&
                        cpi->sf.use_source_sad && !x->zero_temp_sad_source);

  // For the variance computation under SVC mode, we treat the frame as key if
  // the reference (base layer frame) is key frame (i.e., is_key_frame == 1).
  int is_key_frame =
      (frame_is_intra_only(cm) ||
       (is_one_pass_svc(cpi) &&
        cpi->svc.layer_context[cpi->svc.temporal_layer_id].is_key_frame));

  if (!is_key_frame) {
    if (cm->frame_refs[LAST_FRAME - 1].sf.x_scale_fp == REF_INVALID_SCALE ||
        cm->frame_refs[LAST_FRAME - 1].sf.y_scale_fp == REF_INVALID_SCALE)
      is_key_frame = 1;
  }

  // Allow for sub8x8 (4x4) partition on key frames, but only for hybrid mode
  // (i.e., sf->nonrd_keyframe = 0), where for small blocks rd intra pickmode
  // (vp9_rd_pick_intra_mode_sb) is used. The nonrd intra pickmode
  // (vp9_pick_intra_mode) does not currently support sub8x8 blocks. This causes
  // the issue: 44166813. Assert is added in vp9_pick_intra_mode to check this.
  const int use_4x4_partition =
      frame_is_intra_only(cm) && !cpi->sf.nonrd_keyframe;
  const int low_res = (cm->width <= 352 && cm->height <= 288);
  int variance4x4downsample[16];
  int segment_id;
  int sb_offset = (cm->mi_stride >> 3) * (mi_row >> 3) + (mi_col >> 3);

  // For SVC: check if LAST frame is NULL or if the resolution of LAST is
  // different than the current frame resolution, and if so, treat this frame
  // as a key frame, for the purpose of the superblock partitioning.
  // LAST == NULL can happen in some cases where enhancement spatial layers are
  // enabled dyanmically in the stream and the only reference is the spatial
  // reference (GOLDEN).
  if (cpi->use_svc) {
    const YV12_BUFFER_CONFIG *const ref = get_ref_frame_buffer(cpi, LAST_FRAME);
    if (ref == NULL || ref->y_crop_height != cm->height ||
        ref->y_crop_width != cm->width)
      is_key_frame = 1;
  }

  set_offsets(cpi, tile, x, mi_row, mi_col, BLOCK_64X64);
  set_segment_index(cpi, x, mi_row, mi_col, BLOCK_64X64, 0);
  segment_id = xd->mi[0]->segment_id;

  if (cpi->oxcf.speed >= 8 || (cpi->use_svc && cpi->svc.non_reference_frame))
    compute_minmax_variance = 0;

  memset(x->variance_low, 0, sizeof(x->variance_low));

  if (cpi->sf.use_source_sad && !is_key_frame) {
    int sb_offset2 = ((cm->mi_cols + 7) >> 3) * (mi_row >> 3) + (mi_col >> 3);
    content_state = x->content_state_sb;
    x->skip_low_source_sad = (content_state == kLowSadLowSumdiff ||
                              content_state == kLowSadHighSumdiff)
                                 ? 1
                                 : 0;
    x->lowvar_highsumdiff = (content_state == kLowVarHighSumdiff) ? 1 : 0;
    if (cpi->content_state_sb_fd != NULL)
      x->last_sb_high_content = cpi->content_state_sb_fd[sb_offset2];

    // For SVC on top spatial layer: use/scale the partition from
    // the lower spatial resolution if svc_use_lowres_part is enabled.
    if (cpi->sf.svc_use_lowres_part &&
        cpi->svc.spatial_layer_id == cpi->svc.number_spatial_layers - 1 &&
        cpi->svc.prev_partition_svc != NULL && content_state != kVeryHighSad) {
      if (!scale_partitioning_svc(cpi, x, xd, BLOCK_64X64, mi_row >> 1,
                                  mi_col >> 1, mi_row, mi_col)) {
        if (cpi->sf.copy_partition_flag) {
          update_prev_partition(cpi, x, segment_id, mi_row, mi_col, sb_offset);
        }
        return 0;
      }
    }
    // If source_sad is low copy the partition without computing the y_sad.
    if (x->skip_low_source_sad && cpi->sf.copy_partition_flag &&
        !force_64_split &&
        copy_partitioning(cpi, x, xd, mi_row, mi_col, segment_id, sb_offset)) {
      x->sb_use_mv_part = 1;
      if (cpi->sf.svc_use_lowres_part &&
          cpi->svc.spatial_layer_id == cpi->svc.number_spatial_layers - 2)
        update_partition_svc(cpi, BLOCK_64X64, mi_row, mi_col);
      return 0;
    }
  }

  if (cpi->oxcf.aq_mode == CYCLIC_REFRESH_AQ && cm->seg.enabled &&
      cyclic_refresh_segment_id_boosted(segment_id)) {
    int q = vp9_get_qindex(&cm->seg, segment_id, cm->base_qindex);
    set_vbp_thresholds(cpi, thresholds, q, content_state);
  } else {
    set_vbp_thresholds(cpi, thresholds, cm->base_qindex, content_state);
  }
  // Decrease 32x32 split threshold for screen on base layer, for scene
  // change/high motion frames.
  if (cpi->oxcf.content == VP9E_CONTENT_SCREEN &&
      cpi->svc.spatial_layer_id == 0 && force_64_split)
    thresholds[1] = 3 * thresholds[1] >> 2;

  // For non keyframes, disable 4x4 average for low resolution when speed = 8
  threshold_4x4avg = (cpi->oxcf.speed < 8) ? thresholds[1] << 1 : INT64_MAX;

  if (xd->mb_to_right_edge < 0) pixels_wide += (xd->mb_to_right_edge >> 3);
  if (xd->mb_to_bottom_edge < 0) pixels_high += (xd->mb_to_bottom_edge >> 3);

  s = x->plane[0].src.buf;
  sp = x->plane[0].src.stride;

  // Index for force_split: 0 for 64x64, 1-4 for 32x32 blocks,
  // 5-20 for the 16x16 blocks.
  force_split[0] = force_64_split;

  if (!is_key_frame) {
    // In the case of spatial/temporal scalable coding, the assumption here is
    // that the temporal reference frame will always be of type LAST_FRAME.
    // TODO(marpan): If that assumption is broken, we need to revisit this code.
    MODE_INFO *mi = xd->mi[0];
    YV12_BUFFER_CONFIG *yv12 = get_ref_frame_buffer(cpi, LAST_FRAME);

    const YV12_BUFFER_CONFIG *yv12_g = NULL;
    unsigned int y_sad_g, y_sad_thr, y_sad_last;
    bsize = BLOCK_32X32 + (mi_col + 4 < cm->mi_cols) * 2 +
            (mi_row + 4 < cm->mi_rows);

    assert(yv12 != NULL);

    if (!(is_one_pass_svc(cpi) && cpi->svc.spatial_layer_id) ||
        cpi->svc.use_gf_temporal_ref_current_layer) {
      // For now, GOLDEN will not be used for non-zero spatial layers, since
      // it may not be a temporal reference.
      yv12_g = get_ref_frame_buffer(cpi, GOLDEN_FRAME);
    }

    // Only compute y_sad_g (sad for golden reference) for speed < 8.
    if (cpi->oxcf.speed < 8 && yv12_g && yv12_g != yv12 &&
        (cpi->ref_frame_flags & VP9_GOLD_FLAG)) {
      vp9_setup_pre_planes(xd, 0, yv12_g, mi_row, mi_col,
                           &cm->frame_refs[GOLDEN_FRAME - 1].sf);
      y_sad_g = cpi->fn_ptr[bsize].sdf(
          x->plane[0].src.buf, x->plane[0].src.stride, xd->plane[0].pre[0].buf,
          xd->plane[0].pre[0].stride);
    } else {
      y_sad_g = UINT_MAX;
    }

    if (cpi->oxcf.lag_in_frames > 0 && cpi->oxcf.rc_mode == VPX_VBR &&
        cpi->rc.is_src_frame_alt_ref) {
      yv12 = get_ref_frame_buffer(cpi, ALTREF_FRAME);
      vp9_setup_pre_planes(xd, 0, yv12, mi_row, mi_col,
                           &cm->frame_refs[ALTREF_FRAME - 1].sf);
      mi->ref_frame[0] = ALTREF_FRAME;
      y_sad_g = UINT_MAX;
    } else {
      vp9_setup_pre_planes(xd, 0, yv12, mi_row, mi_col,
                           &cm->frame_refs[LAST_FRAME - 1].sf);
      mi->ref_frame[0] = LAST_FRAME;
    }
    mi->ref_frame[1] = NO_REF_FRAME;
    mi->sb_type = BLOCK_64X64;
    mi->mv[0].as_int = 0;
    mi->interp_filter = BILINEAR;

    if (cpi->oxcf.speed >= 8 && !low_res &&
        x->content_state_sb != kVeryHighSad) {
      y_sad = cpi->fn_ptr[bsize].sdf(
          x->plane[0].src.buf, x->plane[0].src.stride, xd->plane[0].pre[0].buf,
          xd->plane[0].pre[0].stride);
    } else {
      const MV dummy_mv = { 0, 0 };
      y_sad = vp9_int_pro_motion_estimation(cpi, x, bsize, mi_row, mi_col,
                                            &dummy_mv);
      x->sb_use_mv_part = 1;
      x->sb_mvcol_part = mi->mv[0].as_mv.col;
      x->sb_mvrow_part = mi->mv[0].as_mv.row;
      if (cpi->oxcf.content == VP9E_CONTENT_SCREEN &&
          cpi->svc.spatial_layer_id == cpi->svc.first_spatial_layer_to_encode &&
          cpi->svc.high_num_blocks_with_motion && !x->zero_temp_sad_source &&
          cm->width > 640 && cm->height > 480) {
        // Disable split below 16x16 block size when scroll motion (horz or
        // vert) is detected.
        // TODO(marpan/jianj): Improve this condition: issue is that search
        // range is hard-coded/limited in vp9_int_pro_motion_estimation() so
        // scroll motion may not be detected here.
        if (((abs(x->sb_mvrow_part) >= 48 && abs(x->sb_mvcol_part) <= 8) ||
             (abs(x->sb_mvcol_part) >= 48 && abs(x->sb_mvrow_part) <= 8)) &&
            y_sad < 100000) {
          compute_minmax_variance = 0;
          thresholds[2] = INT64_MAX;
        }
      }
    }

    y_sad_last = y_sad;
    // Pick ref frame for partitioning, bias last frame when y_sad_g and y_sad
    // are close if short_circuit_low_temp_var is on.
    y_sad_thr = cpi->sf.short_circuit_low_temp_var ? (y_sad * 7) >> 3 : y_sad;
    if (y_sad_g < y_sad_thr) {
      vp9_setup_pre_planes(xd, 0, yv12_g, mi_row, mi_col,
                           &cm->frame_refs[GOLDEN_FRAME - 1].sf);
      mi->ref_frame[0] = GOLDEN_FRAME;
      mi->mv[0].as_int = 0;
      y_sad = y_sad_g;
      ref_frame_partition = GOLDEN_FRAME;
    } else {
      x->pred_mv[LAST_FRAME] = mi->mv[0].as_mv;
      ref_frame_partition = LAST_FRAME;
    }

    set_ref_ptrs(cm, xd, mi->ref_frame[0], mi->ref_frame[1]);
    vp9_build_inter_predictors_sb(xd, mi_row, mi_col, BLOCK_64X64);

    if (cpi->use_skin_detection)
      x->sb_is_skin = skin_sb_split(cpi, low_res, mi_row, mi_col, force_split);

    d = xd->plane[0].dst.buf;
    dp = xd->plane[0].dst.stride;

    // If the y_sad is very small, take 64x64 as partition and exit.
    // Don't check on boosted segment for now, as 64x64 is suppressed there.
    if (segment_id == CR_SEGMENT_ID_BASE && y_sad < cpi->vbp_threshold_sad) {
      const int block_width = num_8x8_blocks_wide_lookup[BLOCK_64X64];
      const int block_height = num_8x8_blocks_high_lookup[BLOCK_64X64];
      if (mi_col + block_width / 2 < cm->mi_cols &&
          mi_row + block_height / 2 < cm->mi_rows) {
        set_block_size(cpi, x, xd, mi_row, mi_col, BLOCK_64X64);
        x->variance_low[0] = 1;
        chroma_check(cpi, x, bsize, y_sad, is_key_frame, scene_change_detected);
        if (cpi->sf.svc_use_lowres_part &&
            cpi->svc.spatial_layer_id == cpi->svc.number_spatial_layers - 2)
          update_partition_svc(cpi, BLOCK_64X64, mi_row, mi_col);
        if (cpi->sf.copy_partition_flag) {
          update_prev_partition(cpi, x, segment_id, mi_row, mi_col, sb_offset);
        }
        return 0;
      }
    }

    // If the y_sad is small enough, copy the partition of the superblock in the
    // last frame to current frame only if the last frame is not a keyframe.
    // Stop the copy every cpi->max_copied_frame to refresh the partition.
    // TODO(jianj) : tune the threshold.
    if (cpi->sf.copy_partition_flag && y_sad_last < cpi->vbp_threshold_copy &&
        copy_partitioning(cpi, x, xd, mi_row, mi_col, segment_id, sb_offset)) {
      chroma_check(cpi, x, bsize, y_sad, is_key_frame, scene_change_detected);
      if (cpi->sf.svc_use_lowres_part &&
          cpi->svc.spatial_layer_id == cpi->svc.number_spatial_layers - 2)
        update_partition_svc(cpi, BLOCK_64X64, mi_row, mi_col);
      return 0;
    }
  } else {
    d = VP9_VAR_OFFS;
    dp = 0;
#if CONFIG_VP9_HIGHBITDEPTH
    if (xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH) {
      switch (xd->bd) {
        case 10: d = CONVERT_TO_BYTEPTR(VP9_HIGH_VAR_OFFS_10); break;
        case 12: d = CONVERT_TO_BYTEPTR(VP9_HIGH_VAR_OFFS_12); break;
        case 8:
        default: d = CONVERT_TO_BYTEPTR(VP9_HIGH_VAR_OFFS_8); break;
      }
    }
#endif  // CONFIG_VP9_HIGHBITDEPTH
  }

  if (low_res && threshold_4x4avg < INT64_MAX)
    CHECK_MEM_ERROR(&cm->error, vt2, vpx_calloc(16, sizeof(*vt2)));
  // Fill in the entire tree of 8x8 (or 4x4 under some conditions) variances
  // for splits.
  for (i = 0; i < 4; i++) {
    const int x32_idx = ((i & 1) << 5);
    const int y32_idx = ((i >> 1) << 5);
    const int i2 = i << 2;
    force_split[i + 1] = 0;
    avg_16x16[i] = 0;
    maxvar_16x16[i] = 0;
    minvar_16x16[i] = INT_MAX;
    for (j = 0; j < 4; j++) {
      const int x16_idx = x32_idx + ((j & 1) << 4);
      const int y16_idx = y32_idx + ((j >> 1) << 4);
      const int split_index = 5 + i2 + j;
      v16x16 *vst = &vt.split[i].split[j];
      force_split[split_index] = 0;
      variance4x4downsample[i2 + j] = 0;
      if (!is_key_frame) {
        fill_variance_8x8avg(s, sp, d, dp, x16_idx, y16_idx, vst,
#if CONFIG_VP9_HIGHBITDEPTH
                             xd->cur_buf->flags,
#endif
                             pixels_wide, pixels_high, is_key_frame);
        fill_variance_tree(&vt.split[i].split[j], BLOCK_16X16);
        get_variance(&vt.split[i].split[j].part_variances.none);
        avg_16x16[i] += vt.split[i].split[j].part_variances.none.variance;
        if (vt.split[i].split[j].part_variances.none.variance < minvar_16x16[i])
          minvar_16x16[i] = vt.split[i].split[j].part_variances.none.variance;
        if (vt.split[i].split[j].part_variances.none.variance > maxvar_16x16[i])
          maxvar_16x16[i] = vt.split[i].split[j].part_variances.none.variance;
        if (vt.split[i].split[j].part_variances.none.variance > thresholds[2]) {
          // 16X16 variance is above threshold for split, so force split to 8x8
          // for this 16x16 block (this also forces splits for upper levels).
          force_split[split_index] = 1;
          force_split[i + 1] = 1;
          force_split[0] = 1;
        } else if (compute_minmax_variance &&
                   vt.split[i].split[j].part_variances.none.variance >
                       thresholds[1] &&
                   !cyclic_refresh_segment_id_boosted(segment_id)) {
          // We have some nominal amount of 16x16 variance (based on average),
          // compute the minmax over the 8x8 sub-blocks, and if above threshold,
          // force split to 8x8 block for this 16x16 block.
          int minmax = compute_minmax_8x8(s, sp, d, dp, x16_idx, y16_idx,
#if CONFIG_VP9_HIGHBITDEPTH
                                          xd->cur_buf->flags,
#endif
                                          pixels_wide, pixels_high);
          int thresh_minmax = (int)cpi->vbp_threshold_minmax;
          if (x->content_state_sb == kVeryHighSad)
            thresh_minmax = thresh_minmax << 1;
          if (minmax > thresh_minmax) {
            force_split[split_index] = 1;
            force_split[i + 1] = 1;
            force_split[0] = 1;
          }
        }
      }
      if (is_key_frame ||
          (low_res && vt.split[i].split[j].part_variances.none.variance >
                          threshold_4x4avg)) {
        force_split[split_index] = 0;
        // Go down to 4x4 down-sampling for variance.
        variance4x4downsample[i2 + j] = 1;
        for (k = 0; k < 4; k++) {
          int x8_idx = x16_idx + ((k & 1) << 3);
          int y8_idx = y16_idx + ((k >> 1) << 3);
          v8x8 *vst2 = is_key_frame ? &vst->split[k] : &vt2[i2 + j].split[k];
          fill_variance_4x4avg(s, sp, d, dp, x8_idx, y8_idx, vst2,
#if CONFIG_VP9_HIGHBITDEPTH
                               xd->cur_buf->flags,
#endif
                               pixels_wide, pixels_high, is_key_frame);
        }
      }
    }
  }
  if (cpi->noise_estimate.enabled)
    noise_level = vp9_noise_estimate_extract_level(&cpi->noise_estimate);
  // Fill the rest of the variance tree by summing split partition values.
  avg_32x32 = 0;
  for (i = 0; i < 4; i++) {
    const int i2 = i << 2;
    for (j = 0; j < 4; j++) {
      if (variance4x4downsample[i2 + j] == 1) {
        v16x16 *vtemp = (!is_key_frame) ? &vt2[i2 + j] : &vt.split[i].split[j];
        for (m = 0; m < 4; m++) fill_variance_tree(&vtemp->split[m], BLOCK_8X8);
        fill_variance_tree(vtemp, BLOCK_16X16);
        // If variance of this 16x16 block is above the threshold, force block
        // to split. This also forces a split on the upper levels.
        get_variance(&vtemp->part_variances.none);
        if (vtemp->part_variances.none.variance > thresholds[2]) {
          force_split[5 + i2 + j] = 1;
          force_split[i + 1] = 1;
          force_split[0] = 1;
        }
      }
    }
    fill_variance_tree(&vt.split[i], BLOCK_32X32);
    // If variance of this 32x32 block is above the threshold, or if its above
    // (some threshold of) the average variance over the sub-16x16 blocks, then
    // force this block to split. This also forces a split on the upper
    // (64x64) level.
    if (!force_split[i + 1]) {
      get_variance(&vt.split[i].part_variances.none);
      var_32x32 = vt.split[i].part_variances.none.variance;
      max_var_32x32 = VPXMAX(var_32x32, max_var_32x32);
      min_var_32x32 = VPXMIN(var_32x32, min_var_32x32);
      if (vt.split[i].part_variances.none.variance > thresholds[1] ||
          (!is_key_frame &&
           vt.split[i].part_variances.none.variance > (thresholds[1] >> 1) &&
           vt.split[i].part_variances.none.variance > (avg_16x16[i] >> 1))) {
        force_split[i + 1] = 1;
        force_split[0] = 1;
      } else if (!is_key_frame && noise_level < kLow && cm->height <= 360 &&
                 (maxvar_16x16[i] - minvar_16x16[i]) > (thresholds[1] >> 1) &&
                 maxvar_16x16[i] > thresholds[1]) {
        force_split[i + 1] = 1;
        force_split[0] = 1;
      }
      avg_32x32 += var_32x32;
    }
  }
  if (!force_split[0]) {
    fill_variance_tree(&vt, BLOCK_64X64);
    get_variance(&vt.part_variances.none);
    // If variance of this 64x64 block is above (some threshold of) the average
    // variance over the sub-32x32 blocks, then force this block to split.
    // Only checking this for noise level >= medium for now.
    if (!is_key_frame && noise_level >= kMedium &&
        vt.part_variances.none.variance > (9 * avg_32x32) >> 5)
      force_split[0] = 1;
    // Else if the maximum 32x32 variance minus the miniumum 32x32 variance in
    // a 64x64 block is greater than threshold and the maximum 32x32 variance is
    // above a miniumum threshold, then force the split of a 64x64 block
    // Only check this for low noise.
    else if (!is_key_frame && noise_level < kMedium &&
             (max_var_32x32 - min_var_32x32) > 3 * (thresholds[0] >> 3) &&
             max_var_32x32 > thresholds[0] >> 1)
      force_split[0] = 1;
  }

  // Now go through the entire structure, splitting every block size until
  // we get to one that's got a variance lower than our threshold.
  if (mi_col + 8 > cm->mi_cols || mi_row + 8 > cm->mi_rows ||
      !set_vt_partitioning(cpi, x, xd, &vt, BLOCK_64X64, mi_row, mi_col,
                           thresholds[0], BLOCK_16X16, force_split[0])) {
    for (i = 0; i < 4; ++i) {
      const int x32_idx = ((i & 1) << 2);
      const int y32_idx = ((i >> 1) << 2);
      const int i2 = i << 2;
      if (!set_vt_partitioning(cpi, x, xd, &vt.split[i], BLOCK_32X32,
                               (mi_row + y32_idx), (mi_col + x32_idx),
                               thresholds[1], BLOCK_16X16,
                               force_split[i + 1])) {
        for (j = 0; j < 4; ++j) {
          const int x16_idx = ((j & 1) << 1);
          const int y16_idx = ((j >> 1) << 1);
          // For inter frames: if variance4x4downsample[] == 1 for this 16x16
          // block, then the variance is based on 4x4 down-sampling, so use vt2
          // in set_vt_partitioning(), otherwise use vt.
          v16x16 *vtemp = (!is_key_frame && variance4x4downsample[i2 + j] == 1)
                              ? &vt2[i2 + j]
                              : &vt.split[i].split[j];
          if (!set_vt_partitioning(
                  cpi, x, xd, vtemp, BLOCK_16X16, mi_row + y32_idx + y16_idx,
                  mi_col + x32_idx + x16_idx, thresholds[2], cpi->vbp_bsize_min,
                  force_split[5 + i2 + j])) {
            for (k = 0; k < 4; ++k) {
              const int x8_idx = (k & 1);
              const int y8_idx = (k >> 1);
              if (use_4x4_partition) {
                if (!set_vt_partitioning(cpi, x, xd, &vtemp->split[k],
                                         BLOCK_8X8,
                                         mi_row + y32_idx + y16_idx + y8_idx,
                                         mi_col + x32_idx + x16_idx + x8_idx,
                                         thresholds[3], BLOCK_8X8, 0)) {
                  set_block_size(
                      cpi, x, xd, (mi_row + y32_idx + y16_idx + y8_idx),
                      (mi_col + x32_idx + x16_idx + x8_idx), BLOCK_4X4);
                }
              } else {
                set_block_size(
                    cpi, x, xd, (mi_row + y32_idx + y16_idx + y8_idx),
                    (mi_col + x32_idx + x16_idx + x8_idx), BLOCK_8X8);
              }
            }
          }
        }
      }
    }
  }

  if (!frame_is_intra_only(cm) && cpi->sf.copy_partition_flag) {
    update_prev_partition(cpi, x, segment_id, mi_row, mi_col, sb_offset);
  }

  if (!frame_is_intra_only(cm) && cpi->sf.svc_use_lowres_part &&
      cpi->svc.spatial_layer_id == cpi->svc.number_spatial_layers - 2)
    update_partition_svc(cpi, BLOCK_64X64, mi_row, mi_col);

  if (cpi->sf.short_circuit_low_temp_var) {
    set_low_temp_var_flag(cpi, x, xd, &vt, thresholds, ref_frame_partition,
                          mi_col, mi_row);
  }

  chroma_check(cpi, x, bsize, y_sad, is_key_frame, scene_change_detected);
  if (vt2) vpx_free(vt2);
  return 0;
}

#if !CONFIG_REALTIME_ONLY
static void update_state(VP9_COMP *cpi, ThreadData *td, PICK_MODE_CONTEXT *ctx,
                         int mi_row, int mi_col, BLOCK_SIZE bsize,
                         int output_enabled) {
  int i, x_idx, y;
  VP9_COMMON *const cm = &cpi->common;
  RD_COUNTS *const rdc = &td->rd_counts;
  MACROBLOCK *const x = &td->mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  struct macroblock_plane *const p = x->plane;
  struct macroblockd_plane *const pd = xd->plane;
  MODE_INFO *mi = &ctx->mic;
  MODE_INFO *const xdmi = xd->mi[0];
  MODE_INFO *mi_addr = xd->mi[0];
  const struct segmentation *const seg = &cm->seg;
  const int bw = num_8x8_blocks_wide_lookup[mi->sb_type];
  const int bh = num_8x8_blocks_high_lookup[mi->sb_type];
  const int x_mis = VPXMIN(bw, cm->mi_cols - mi_col);
  const int y_mis = VPXMIN(bh, cm->mi_rows - mi_row);
  MV_REF *const frame_mvs = cm->cur_frame->mvs + mi_row * cm->mi_cols + mi_col;
  int w, h;

  const int mis = cm->mi_stride;
  const int mi_width = num_8x8_blocks_wide_lookup[bsize];
  const int mi_height = num_8x8_blocks_high_lookup[bsize];
  int max_plane;

  assert(mi->sb_type == bsize);

  *mi_addr = *mi;
  *x->mbmi_ext = ctx->mbmi_ext;

  // If segmentation in use
  if (seg->enabled) {
    // For in frame complexity AQ copy the segment id from the segment map.
    if (cpi->oxcf.aq_mode == COMPLEXITY_AQ) {
      const uint8_t *const map =
          seg->update_map ? cpi->segmentation_map : cm->last_frame_seg_map;
      mi_addr->segment_id = get_segment_id(cm, map, bsize, mi_row, mi_col);
    }
    // Else for cyclic refresh mode update the segment map, set the segment id
    // and then update the quantizer.
    if (cpi->oxcf.aq_mode == CYCLIC_REFRESH_AQ &&
        cpi->cyclic_refresh->content_mode) {
      vp9_cyclic_refresh_update_segment(cpi, xd->mi[0], mi_row, mi_col, bsize,
                                        ctx->rate, ctx->dist, x->skip, p);
    }
  }

  max_plane = is_inter_block(xdmi) ? MAX_MB_PLANE : 1;
  for (i = 0; i < max_plane; ++i) {
    p[i].coeff = ctx->coeff_pbuf[i][1];
    p[i].qcoeff = ctx->qcoeff_pbuf[i][1];
    pd[i].dqcoeff = ctx->dqcoeff_pbuf[i][1];
    p[i].eobs = ctx->eobs_pbuf[i][1];
  }

  for (i = max_plane; i < MAX_MB_PLANE; ++i) {
    p[i].coeff = ctx->coeff_pbuf[i][2];
    p[i].qcoeff = ctx->qcoeff_pbuf[i][2];
    pd[i].dqcoeff = ctx->dqcoeff_pbuf[i][2];
    p[i].eobs = ctx->eobs_pbuf[i][2];
  }

  // Restore the coding context of the MB to that that was in place
  // when the mode was picked for it
  for (y = 0; y < mi_height; y++)
    for (x_idx = 0; x_idx < mi_width; x_idx++)
      if ((xd->mb_to_right_edge >> (3 + MI_SIZE_LOG2)) + mi_width > x_idx &&
          (xd->mb_to_bottom_edge >> (3 + MI_SIZE_LOG2)) + mi_height > y) {
        xd->mi[x_idx + y * mis] = mi_addr;
      }

  if (cpi->oxcf.aq_mode != NO_AQ) vp9_init_plane_quantizers(cpi, x);

  if (is_inter_block(xdmi) && xdmi->sb_type < BLOCK_8X8) {
    xdmi->mv[0].as_int = mi->bmi[3].as_mv[0].as_int;
    xdmi->mv[1].as_int = mi->bmi[3].as_mv[1].as_int;
  }

  x->skip = ctx->skip;
  memcpy(x->zcoeff_blk[xdmi->tx_size], ctx->zcoeff_blk,
         sizeof(ctx->zcoeff_blk[0]) * ctx->num_4x4_blk);

  if (!output_enabled) return;

#if CONFIG_INTERNAL_STATS
  if (frame_is_intra_only(cm)) {
    static const int kf_mode_index[] = {
      THR_DC /*DC_PRED*/,          THR_V_PRED /*V_PRED*/,
      THR_H_PRED /*H_PRED*/,       THR_D45_PRED /*D45_PRED*/,
      THR_D135_PRED /*D135_PRED*/, THR_D117_PRED /*D117_PRED*/,
      THR_D153_PRED /*D153_PRED*/, THR_D207_PRED /*D207_PRED*/,
      THR_D63_PRED /*D63_PRED*/,   THR_TM /*TM_PRED*/,
    };
    ++cpi->mode_chosen_counts[kf_mode_index[xdmi->mode]];
  } else {
    // Note how often each mode chosen as best
    ++cpi->mode_chosen_counts[ctx->best_mode_index];
  }
#endif
  if (!frame_is_intra_only(cm)) {
    if (is_inter_block(xdmi)) {
      vp9_update_mv_count(td);

      if (cm->interp_filter == SWITCHABLE) {
        const int ctx_interp = get_pred_context_switchable_interp(xd);
        ++td->counts->switchable_interp[ctx_interp][xdmi->interp_filter];
      }
    }

    rdc->comp_pred_diff[SINGLE_REFERENCE] += ctx->single_pred_diff;
    rdc->comp_pred_diff[COMPOUND_REFERENCE] += ctx->comp_pred_diff;
    rdc->comp_pred_diff[REFERENCE_MODE_SELECT] += ctx->hybrid_pred_diff;

    for (i = 0; i < SWITCHABLE_FILTER_CONTEXTS; ++i)
      rdc->filter_diff[i] += ctx->best_filter_diff[i];
  }

  for (h = 0; h < y_mis; ++h) {
    MV_REF *const frame_mv = frame_mvs + h * cm->mi_cols;
    for (w = 0; w < x_mis; ++w) {
      MV_REF *const mv = frame_mv + w;
      mv->ref_frame[0] = mi->ref_frame[0];
      mv->ref_frame[1] = mi->ref_frame[1];
      mv->mv[0].as_int = mi->mv[0].as_int;
      mv->mv[1].as_int = mi->mv[1].as_int;
    }
  }
}
#endif  // !CONFIG_REALTIME_ONLY

void vp9_setup_src_planes(MACROBLOCK *x, const YV12_BUFFER_CONFIG *src,
                          int mi_row, int mi_col) {
  uint8_t *const buffers[3] = { src->y_buffer, src->u_buffer, src->v_buffer };
  const int strides[3] = { src->y_stride, src->uv_stride, src->uv_stride };
  int i;

  // Set current frame pointer.
  x->e_mbd.cur_buf = src;

  for (i = 0; i < MAX_MB_PLANE; i++)
    setup_pred_plane(&x->plane[i].src, buffers[i], strides[i], mi_row, mi_col,
                     NULL, x->e_mbd.plane[i].subsampling_x,
                     x->e_mbd.plane[i].subsampling_y);
}

static void set_mode_info_seg_skip(MACROBLOCK *x, TX_MODE tx_mode,
                                   INTERP_FILTER interp_filter,
                                   RD_COST *rd_cost, BLOCK_SIZE bsize) {
  MACROBLOCKD *const xd = &x->e_mbd;
  MODE_INFO *const mi = xd->mi[0];
  INTERP_FILTER filter_ref;

  filter_ref = get_pred_context_switchable_interp(xd);
  if (interp_filter == BILINEAR)
    filter_ref = BILINEAR;
  else if (filter_ref == SWITCHABLE_FILTERS)
    filter_ref = EIGHTTAP;

  mi->sb_type = bsize;
  mi->mode = ZEROMV;
  mi->tx_size =
      VPXMIN(max_txsize_lookup[bsize], tx_mode_to_biggest_tx_size[tx_mode]);
  mi->skip = 1;
  mi->uv_mode = DC_PRED;
  mi->ref_frame[0] = LAST_FRAME;
  mi->ref_frame[1] = NO_REF_FRAME;
  mi->mv[0].as_int = 0;
  mi->interp_filter = filter_ref;

  xd->mi[0]->bmi[0].as_mv[0].as_int = 0;
  x->skip = 1;

  vp9_rd_cost_init(rd_cost);
}

#if !CONFIG_REALTIME_ONLY
static void set_segment_rdmult(VP9_COMP *const cpi, MACROBLOCK *const x,
                               int mi_row, int mi_col, BLOCK_SIZE bsize,
                               AQ_MODE aq_mode) {
  VP9_COMMON *const cm = &cpi->common;
  const VP9EncoderConfig *const oxcf = &cpi->oxcf;
  const uint8_t *const map =
      cm->seg.update_map ? cpi->segmentation_map : cm->last_frame_seg_map;

  vp9_init_plane_quantizers(cpi, x);
  vpx_clear_system_state();

  if (aq_mode == NO_AQ || aq_mode == PSNR_AQ) {
    if (cpi->sf.enable_tpl_model) x->rdmult = x->cb_rdmult;
  } else if (aq_mode == PERCEPTUAL_AQ) {
    x->rdmult = x->cb_rdmult;
  } else if (aq_mode == CYCLIC_REFRESH_AQ) {
    // If segment is boosted, use rdmult for that segment.
    if (cyclic_refresh_segment_id_boosted(
            get_segment_id(cm, map, bsize, mi_row, mi_col)))
      x->rdmult = vp9_cyclic_refresh_get_rdmult(cpi->cyclic_refresh);
  } else {
    x->rdmult = vp9_compute_rd_mult(cpi, cm->base_qindex + cm->y_dc_delta_q);
  }

  if (oxcf->tuning == VP8_TUNE_SSIM) {
    set_ssim_rdmult(cpi, x, bsize, mi_row, mi_col, &x->rdmult);
  }
}

static void rd_pick_sb_modes(VP9_COMP *cpi, TileDataEnc *tile_data,
                             MACROBLOCK *const x, int mi_row, int mi_col,
                             RD_COST *rd_cost, BLOCK_SIZE bsize,
                             PICK_MODE_CONTEXT *ctx, int rate_in_best_rd,
                             int64_t dist_in_best_rd) {
  VP9_COMMON *const cm = &cpi->common;
  TileInfo *const tile_info = &tile_data->tile_info;
  MACROBLOCKD *const xd = &x->e_mbd;
  MODE_INFO *mi;
  struct macroblock_plane *const p = x->plane;
  struct macroblockd_plane *const pd = xd->plane;
  const AQ_MODE aq_mode = cpi->oxcf.aq_mode;
  int i, orig_rdmult;
  int64_t best_rd = INT64_MAX;

  vpx_clear_system_state();
#if CONFIG_COLLECT_COMPONENT_TIMING
  start_timing(cpi, rd_pick_sb_modes_time);
#endif

  // Use the lower precision, but faster, 32x32 fdct for mode selection.
  x->use_lp32x32fdct = 1;

  set_offsets(cpi, tile_info, x, mi_row, mi_col, bsize);
  mi = xd->mi[0];
  mi->sb_type = bsize;

  for (i = 0; i < MAX_MB_PLANE; ++i) {
    p[i].coeff = ctx->coeff_pbuf[i][0];
    p[i].qcoeff = ctx->qcoeff_pbuf[i][0];
    pd[i].dqcoeff = ctx->dqcoeff_pbuf[i][0];
    p[i].eobs = ctx->eobs_pbuf[i][0];
  }
  ctx->is_coded = 0;
  ctx->skippable = 0;
  ctx->pred_pixel_ready = 0;
  x->skip_recode = 0;

  // Set to zero to make sure we do not use the previous encoded frame stats
  mi->skip = 0;

#if CONFIG_VP9_HIGHBITDEPTH
  if (xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH) {
    x->source_variance = vp9_high_get_sby_perpixel_variance(
        cpi, &x->plane[0].src, bsize, xd->bd);
  } else {
    x->source_variance =
        vp9_get_sby_perpixel_variance(cpi, &x->plane[0].src, bsize);
  }
#else
  x->source_variance =
      vp9_get_sby_perpixel_variance(cpi, &x->plane[0].src, bsize);
#endif  // CONFIG_VP9_HIGHBITDEPTH

  // Save rdmult before it might be changed, so it can be restored later.
  orig_rdmult = x->rdmult;

  if ((cpi->sf.tx_domain_thresh > 0.0) ||
      (cpi->sf.trellis_opt_tx_rd.thresh > 0.0)) {
    double logvar = vp9_log_block_var(cpi, x, bsize);
    // Check block complexity as part of decision on using pixel or transform
    // domain distortion in rd tests.
    x->block_tx_domain = cpi->sf.allow_txfm_domain_distortion &&
                         (logvar >= cpi->sf.tx_domain_thresh);

    // Store block complexity to decide on using quantized coefficient
    // optimization inside the rd loop.
    x->log_block_src_var = logvar;
  } else {
    x->block_tx_domain = cpi->sf.allow_txfm_domain_distortion;
    x->log_block_src_var = 0.0;
  }

  set_segment_index(cpi, x, mi_row, mi_col, bsize, 0);
  set_segment_rdmult(cpi, x, mi_row, mi_col, bsize, aq_mode);
  if (rate_in_best_rd < INT_MAX && dist_in_best_rd < INT64_MAX) {
    best_rd = vp9_calculate_rd_cost(x->rdmult, x->rddiv, rate_in_best_rd,
                                    dist_in_best_rd);
  }

  // Find best coding mode & reconstruct the MB so it is available
  // as a predictor for MBs that follow in the SB
  if (frame_is_intra_only(cm)) {
    vp9_rd_pick_intra_mode_sb(cpi, x, rd_cost, bsize, ctx, best_rd);
  } else {
    if (bsize >= BLOCK_8X8) {
#if CONFIG_COLLECT_COMPONENT_TIMING
      start_timing(cpi, vp9_rd_pick_inter_mode_sb_time);
#endif
      if (segfeature_active(&cm->seg, mi->segment_id, SEG_LVL_SKIP))
        vp9_rd_pick_inter_mode_sb_seg_skip(cpi, tile_data, x, rd_cost, bsize,
                                           ctx, best_rd);
      else
        vp9_rd_pick_inter_mode_sb(cpi, tile_data, x, mi_row, mi_col, rd_cost,
                                  bsize, ctx, best_rd);
#if CONFIG_COLLECT_COMPONENT_TIMING
      end_timing(cpi, vp9_rd_pick_inter_mode_sb_time);
#endif
    } else {
#if CONFIG_COLLECT_COMPONENT_TIMING
      start_timing(cpi, vp9_rd_pick_inter_mode_sub8x8_time);
#endif
      vp9_rd_pick_inter_mode_sub8x8(cpi, tile_data, x, mi_row, mi_col, rd_cost,
                                    bsize, ctx, best_rd);
#if CONFIG_COLLECT_COMPONENT_TIMING
      end_timing(cpi, vp9_rd_pick_inter_mode_sub8x8_time);
#endif
    }
  }

  // Examine the resulting rate and for AQ mode 2 make a segment choice.
  if ((rd_cost->rate != INT_MAX) && (aq_mode == COMPLEXITY_AQ) &&
      (bsize >= BLOCK_16X16) &&
      (cm->frame_type == KEY_FRAME || cpi->refresh_alt_ref_frame ||
       (cpi->refresh_golden_frame && !cpi->rc.is_src_frame_alt_ref))) {
    vp9_caq_select_segment(cpi, x, bsize, mi_row, mi_col, rd_cost->rate);
  }

  // TODO(jingning) The rate-distortion optimization flow needs to be
  // refactored to provide proper exit/return handle.
  if (rd_cost->rate == INT_MAX || rd_cost->dist == INT64_MAX)
    rd_cost->rdcost = INT64_MAX;
  else
    rd_cost->rdcost = RDCOST(x->rdmult, x->rddiv, rd_cost->rate, rd_cost->dist);

  x->rdmult = orig_rdmult;

  ctx->rate = rd_cost->rate;
  ctx->dist = rd_cost->dist;
#if CONFIG_COLLECT_COMPONENT_TIMING
  end_timing(cpi, rd_pick_sb_modes_time);
#endif
}
#endif  // !CONFIG_REALTIME_ONLY

static void update_stats(VP9_COMMON *cm, ThreadData *td) {
  const MACROBLOCK *x = &td->mb;
  const MACROBLOCKD *const xd = &x->e_mbd;
  const MODE_INFO *const mi = xd->mi[0];
  const MB_MODE_INFO_EXT *const mbmi_ext = x->mbmi_ext;
  const BLOCK_SIZE bsize = mi->sb_type;

  if (!frame_is_intra_only(cm)) {
    FRAME_COUNTS *const counts = td->counts;
    const int inter_block = is_inter_block(mi);
    const int seg_ref_active =
        segfeature_active(&cm->seg, mi->segment_id, SEG_LVL_REF_FRAME);
    if (!seg_ref_active) {
      counts->intra_inter[get_intra_inter_context(xd)][inter_block]++;
      // If the segment reference feature is enabled we have only a single
      // reference frame allowed for the segment so exclude it from
      // the reference frame counts used to work out probabilities.
      if (inter_block) {
        const MV_REFERENCE_FRAME ref0 = mi->ref_frame[0];
        if (cm->reference_mode == REFERENCE_MODE_SELECT)
          counts->comp_inter[vp9_get_reference_mode_context(cm, xd)]
                            [has_second_ref(mi)]++;

        if (has_second_ref(mi)) {
          const int idx = cm->ref_frame_sign_bias[cm->comp_fixed_ref];
          const int ctx = vp9_get_pred_context_comp_ref_p(cm, xd);
          const int bit = mi->ref_frame[!idx] == cm->comp_var_ref[1];
          counts->comp_ref[ctx][bit]++;
        } else {
          counts->single_ref[vp9_get_pred_context_single_ref_p1(xd)][0]
                            [ref0 != LAST_FRAME]++;
          if (ref0 != LAST_FRAME)
            counts->single_ref[vp9_get_pred_context_single_ref_p2(xd)][1]
                              [ref0 != GOLDEN_FRAME]++;
        }
      }
    }
    if (inter_block &&
        !segfeature_active(&cm->seg, mi->segment_id, SEG_LVL_SKIP)) {
      const int mode_ctx = mbmi_ext->mode_context[mi->ref_frame[0]];
      if (bsize >= BLOCK_8X8) {
        const PREDICTION_MODE mode = mi->mode;
        ++counts->inter_mode[mode_ctx][INTER_OFFSET(mode)];
      } else {
        const int num_4x4_w = num_4x4_blocks_wide_lookup[bsize];
        const int num_4x4_h = num_4x4_blocks_high_lookup[bsize];
        int idx, idy;
        for (idy = 0; idy < 2; idy += num_4x4_h) {
          for (idx = 0; idx < 2; idx += num_4x4_w) {
            const int j = idy * 2 + idx;
            const PREDICTION_MODE b_mode = mi->bmi[j].as_mode;
            ++counts->inter_mode[mode_ctx][INTER_OFFSET(b_mode)];
          }
        }
      }
    }
  }
}

#if !CONFIG_REALTIME_ONLY
static void restore_context(MACROBLOCK *const x, int mi_row, int mi_col,
                            ENTROPY_CONTEXT a[16 * MAX_MB_PLANE],
                            ENTROPY_CONTEXT l[16 * MAX_MB_PLANE],
                            PARTITION_CONTEXT sa[8], PARTITION_CONTEXT sl[8],
                            BLOCK_SIZE bsize) {
  MACROBLOCKD *const xd = &x->e_mbd;
  int p;
  const int num_4x4_blocks_wide = num_4x4_blocks_wide_lookup[bsize];
  const int num_4x4_blocks_high = num_4x4_blocks_high_lookup[bsize];
  int mi_width = num_8x8_blocks_wide_lookup[bsize];
  int mi_height = num_8x8_blocks_high_lookup[bsize];
  for (p = 0; p < MAX_MB_PLANE; p++) {
    memcpy(xd->above_context[p] + ((mi_col * 2) >> xd->plane[p].subsampling_x),
           a + num_4x4_blocks_wide * p,
           (sizeof(ENTROPY_CONTEXT) * num_4x4_blocks_wide) >>
               xd->plane[p].subsampling_x);
    memcpy(xd->left_context[p] +
               ((mi_row & MI_MASK) * 2 >> xd->plane[p].subsampling_y),
           l + num_4x4_blocks_high * p,
           (sizeof(ENTROPY_CONTEXT) * num_4x4_blocks_high) >>
               xd->plane[p].subsampling_y);
  }
  memcpy(xd->above_seg_context + mi_col, sa,
         sizeof(*xd->above_seg_context) * mi_width);
  memcpy(xd->left_seg_context + (mi_row & MI_MASK), sl,
         sizeof(xd->left_seg_context[0]) * mi_height);
}

static void save_context(MACROBLOCK *const x, int mi_row, int mi_col,
                         ENTROPY_CONTEXT a[16 * MAX_MB_PLANE],
                         ENTROPY_CONTEXT l[16 * MAX_MB_PLANE],
                         PARTITION_CONTEXT sa[8], PARTITION_CONTEXT sl[8],
                         BLOCK_SIZE bsize) {
  const MACROBLOCKD *const xd = &x->e_mbd;
  int p;
  const int num_4x4_blocks_wide = num_4x4_blocks_wide_lookup[bsize];
  const int num_4x4_blocks_high = num_4x4_blocks_high_lookup[bsize];
  int mi_width = num_8x8_blocks_wide_lookup[bsize];
  int mi_height = num_8x8_blocks_high_lookup[bsize];

  // buffer the above/left context information of the block in search.
  for (p = 0; p < MAX_MB_PLANE; ++p) {
    memcpy(a + num_4x4_blocks_wide * p,
           xd->above_context[p] + (mi_col * 2 >> xd->plane[p].subsampling_x),
           (sizeof(ENTROPY_CONTEXT) * num_4x4_blocks_wide) >>
               xd->plane[p].subsampling_x);
    memcpy(l + num_4x4_blocks_high * p,
           xd->left_context[p] +
               ((mi_row & MI_MASK) * 2 >> xd->plane[p].subsampling_y),
           (sizeof(ENTROPY_CONTEXT) * num_4x4_blocks_high) >>
               xd->plane[p].subsampling_y);
  }
  memcpy(sa, xd->above_seg_context + mi_col,
         sizeof(*xd->above_seg_context) * mi_width);
  memcpy(sl, xd->left_seg_context + (mi_row & MI_MASK),
         sizeof(xd->left_seg_context[0]) * mi_height);
}

static void encode_b(VP9_COMP *cpi, const TileInfo *const tile, ThreadData *td,
                     TOKENEXTRA **tp, int mi_row, int mi_col,
                     int output_enabled, BLOCK_SIZE bsize,
                     PICK_MODE_CONTEXT *ctx) {
  MACROBLOCK *const x = &td->mb;
  set_offsets(cpi, tile, x, mi_row, mi_col, bsize);

  if (cpi->sf.enable_tpl_model &&
      (cpi->oxcf.aq_mode == NO_AQ || cpi->oxcf.aq_mode == PERCEPTUAL_AQ)) {
    const VP9EncoderConfig *const oxcf = &cpi->oxcf;
    x->rdmult = x->cb_rdmult;
    if (oxcf->tuning == VP8_TUNE_SSIM) {
      set_ssim_rdmult(cpi, x, bsize, mi_row, mi_col, &x->rdmult);
    }
  }

  update_state(cpi, td, ctx, mi_row, mi_col, bsize, output_enabled);
  encode_superblock(cpi, td, tp, output_enabled, mi_row, mi_col, bsize, ctx);

  if (output_enabled) {
    update_stats(&cpi->common, td);

    (*tp)->token = EOSB_TOKEN;
    (*tp)++;
  }
}

static void encode_sb(VP9_COMP *cpi, ThreadData *td, const TileInfo *const tile,
                      TOKENEXTRA **tp, int mi_row, int mi_col,
                      int output_enabled, BLOCK_SIZE bsize, PC_TREE *pc_tree) {
  VP9_COMMON *const cm = &cpi->common;
  MACROBLOCK *const x = &td->mb;
  MACROBLOCKD *const xd = &x->e_mbd;

  const int bsl = b_width_log2_lookup[bsize], hbs = (1 << bsl) / 4;
  int ctx;
  PARTITION_TYPE partition;
  BLOCK_SIZE subsize = bsize;

  if (mi_row >= cm->mi_rows || mi_col >= cm->mi_cols) return;

  if (bsize >= BLOCK_8X8) {
    ctx = partition_plane_context(xd, mi_row, mi_col, bsize);
    subsize = get_subsize(bsize, pc_tree->partitioning);
  } else {
    ctx = 0;
    subsize = BLOCK_4X4;
  }

  partition = partition_lookup[bsl][subsize];
  if (output_enabled && bsize != BLOCK_4X4)
    td->counts->partition[ctx][partition]++;

  switch (partition) {
    case PARTITION_NONE:
      encode_b(cpi, tile, td, tp, mi_row, mi_col, output_enabled, subsize,
               &pc_tree->none);
      break;
    case PARTITION_VERT:
      encode_b(cpi, tile, td, tp, mi_row, mi_col, output_enabled, subsize,
               &pc_tree->vertical[0]);
      if (mi_col + hbs < cm->mi_cols && bsize > BLOCK_8X8) {
        encode_b(cpi, tile, td, tp, mi_row, mi_col + hbs, output_enabled,
                 subsize, &pc_tree->vertical[1]);
      }
      break;
    case PARTITION_HORZ:
      encode_b(cpi, tile, td, tp, mi_row, mi_col, output_enabled, subsize,
               &pc_tree->horizontal[0]);
      if (mi_row + hbs < cm->mi_rows && bsize > BLOCK_8X8) {
        encode_b(cpi, tile, td, tp, mi_row + hbs, mi_col, output_enabled,
                 subsize, &pc_tree->horizontal[1]);
      }
      break;
    default:
      assert(partition == PARTITION_SPLIT);
      if (bsize == BLOCK_8X8) {
        encode_b(cpi, tile, td, tp, mi_row, mi_col, output_enabled, subsize,
                 pc_tree->u.leaf_split[0]);
      } else {
        encode_sb(cpi, td, tile, tp, mi_row, mi_col, output_enabled, subsize,
                  pc_tree->u.split[0]);
        encode_sb(cpi, td, tile, tp, mi_row, mi_col + hbs, output_enabled,
                  subsize, pc_tree->u.split[1]);
        encode_sb(cpi, td, tile, tp, mi_row + hbs, mi_col, output_enabled,
                  subsize, pc_tree->u.split[2]);
        encode_sb(cpi, td, tile, tp, mi_row + hbs, mi_col + hbs, output_enabled,
                  subsize, pc_tree->u.split[3]);
      }
      break;
  }

  if (partition != PARTITION_SPLIT || bsize == BLOCK_8X8)
    update_partition_context(xd, mi_row, mi_col, subsize, bsize);
}
#endif  // !CONFIG_REALTIME_ONLY

// Check to see if the given partition size is allowed for a specified number
// of 8x8 block rows and columns remaining in the image.
// If not then return the largest allowed partition size
static BLOCK_SIZE find_partition_size(BLOCK_SIZE bsize, int rows_left,
                                      int cols_left, int *bh, int *bw) {
  if (rows_left <= 0 || cols_left <= 0) {
    return VPXMIN(bsize, BLOCK_8X8);
  } else {
    for (; bsize > 0; bsize -= 3) {
      *bh = num_8x8_blocks_high_lookup[bsize];
      *bw = num_8x8_blocks_wide_lookup[bsize];
      if ((*bh <= rows_left) && (*bw <= cols_left)) {
        break;
      }
    }
  }
  return bsize;
}

static void set_partial_b64x64_partition(MODE_INFO *mi, int mis, int bh_in,
                                         int bw_in, int row8x8_remaining,
                                         int col8x8_remaining, BLOCK_SIZE bsize,
                                         MODE_INFO **mi_8x8) {
  int bh = bh_in;
  int r, c;
  for (r = 0; r < MI_BLOCK_SIZE; r += bh) {
    int bw = bw_in;
    for (c = 0; c < MI_BLOCK_SIZE; c += bw) {
      const int index = r * mis + c;
      mi_8x8[index] = mi + index;
      mi_8x8[index]->sb_type = find_partition_size(
          bsize, row8x8_remaining - r, col8x8_remaining - c, &bh, &bw);
    }
  }
}

// This function attempts to set all mode info entries in a given SB64
// to the same block partition size.
// However, at the bottom and right borders of the image the requested size
// may not be allowed in which case this code attempts to choose the largest
// allowable partition.
static void set_fixed_partitioning(VP9_COMP *cpi, const TileInfo *const tile,
                                   MODE_INFO **mi_8x8, int mi_row, int mi_col,
                                   BLOCK_SIZE bsize) {
  VP9_COMMON *const cm = &cpi->common;
  const int mis = cm->mi_stride;
  const int row8x8_remaining = tile->mi_row_end - mi_row;
  const int col8x8_remaining = tile->mi_col_end - mi_col;
  int block_row, block_col;
  MODE_INFO *mi_upper_left = cm->mi + mi_row * mis + mi_col;
  int bh = num_8x8_blocks_high_lookup[bsize];
  int bw = num_8x8_blocks_wide_lookup[bsize];

  assert((row8x8_remaining > 0) && (col8x8_remaining > 0));

  // Apply the requested partition size to the SB64 if it is all "in image"
  if ((col8x8_remaining >= MI_BLOCK_SIZE) &&
      (row8x8_remaining >= MI_BLOCK_SIZE)) {
    for (block_row = 0; block_row < MI_BLOCK_SIZE; block_row += bh) {
      for (block_col = 0; block_col < MI_BLOCK_SIZE; block_col += bw) {
        int index = block_row * mis + block_col;
        mi_8x8[index] = mi_upper_left + index;
        mi_8x8[index]->sb_type = bsize;
      }
    }
  } else {
    // Else this is a partial SB64.
    set_partial_b64x64_partition(mi_upper_left, mis, bh, bw, row8x8_remaining,
                                 col8x8_remaining, bsize, mi_8x8);
  }
}

static void update_state_rt(VP9_COMP *cpi, ThreadData *td,
                            PICK_MODE_CONTEXT *ctx, int mi_row, int mi_col,
                            int bsize) {
  VP9_COMMON *const cm = &cpi->common;
  MACROBLOCK *const x = &td->mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  MODE_INFO *const mi = xd->mi[0];
  struct macroblock_plane *const p = x->plane;
  const struct segmentation *const seg = &cm->seg;
  const int bw = num_8x8_blocks_wide_lookup[mi->sb_type];
  const int bh = num_8x8_blocks_high_lookup[mi->sb_type];
  const int x_mis = VPXMIN(bw, cm->mi_cols - mi_col);
  const int y_mis = VPXMIN(bh, cm->mi_rows - mi_row);

  *(xd->mi[0]) = ctx->mic;
  *(x->mbmi_ext) = ctx->mbmi_ext;

  if (seg->enabled && (cpi->oxcf.aq_mode != NO_AQ || cpi->roi.enabled ||
                       cpi->active_map.enabled)) {
    // Setting segmentation map for cyclic_refresh.
    if (cpi->oxcf.aq_mode == CYCLIC_REFRESH_AQ &&
        cpi->cyclic_refresh->content_mode) {
      vp9_cyclic_refresh_update_segment(cpi, mi, mi_row, mi_col, bsize,
                                        ctx->rate, ctx->dist, x->skip, p);
    } else {
      const uint8_t *const map =
          seg->update_map ? cpi->segmentation_map : cm->last_frame_seg_map;
      mi->segment_id = get_segment_id(cm, map, bsize, mi_row, mi_col);
    }
    vp9_init_plane_quantizers(cpi, x);
  }

  if (is_inter_block(mi)) {
    vp9_update_mv_count(td);
    if (cm->interp_filter == SWITCHABLE) {
      const int pred_ctx = get_pred_context_switchable_interp(xd);
      ++td->counts->switchable_interp[pred_ctx][mi->interp_filter];
    }

    if (mi->sb_type < BLOCK_8X8) {
      mi->mv[0].as_int = mi->bmi[3].as_mv[0].as_int;
      mi->mv[1].as_int = mi->bmi[3].as_mv[1].as_int;
    }
  }

  if (cm->use_prev_frame_mvs || !cm->error_resilient_mode ||
      (cpi->svc.use_base_mv && cpi->svc.number_spatial_layers > 1 &&
       cpi->svc.spatial_layer_id != cpi->svc.number_spatial_layers - 1)) {
    MV_REF *const frame_mvs =
        cm->cur_frame->mvs + mi_row * cm->mi_cols + mi_col;
    int w, h;

    for (h = 0; h < y_mis; ++h) {
      MV_REF *const frame_mv = frame_mvs + h * cm->mi_cols;
      for (w = 0; w < x_mis; ++w) {
        MV_REF *const mv = frame_mv + w;
        mv->ref_frame[0] = mi->ref_frame[0];
        mv->ref_frame[1] = mi->ref_frame[1];
        mv->mv[0].as_int = mi->mv[0].as_int;
        mv->mv[1].as_int = mi->mv[1].as_int;
      }
    }
  }

  x->skip = ctx->skip;
  x->skip_txfm[0] = (mi->segment_id || xd->lossless) ? 0 : ctx->skip_txfm[0];
}

static void encode_b_rt(VP9_COMP *cpi, ThreadData *td,
                        const TileInfo *const tile, TOKENEXTRA **tp, int mi_row,
                        int mi_col, int output_enabled, BLOCK_SIZE bsize,
                        PICK_MODE_CONTEXT *ctx) {
  MACROBLOCK *const x = &td->mb;
  set_offsets(cpi, tile, x, mi_row, mi_col, bsize);
  update_state_rt(cpi, td, ctx, mi_row, mi_col, bsize);

  encode_superblock(cpi, td, tp, output_enabled, mi_row, mi_col, bsize, ctx);
  update_stats(&cpi->common, td);

  (*tp)->token = EOSB_TOKEN;
  (*tp)++;
}

static void encode_sb_rt(VP9_COMP *cpi, ThreadData *td,
                         const TileInfo *const tile, TOKENEXTRA **tp,
                         int mi_row, int mi_col, int output_enabled,
                         BLOCK_SIZE bsize, PC_TREE *pc_tree) {
  VP9_COMMON *const cm = &cpi->common;
  MACROBLOCK *const x = &td->mb;
  MACROBLOCKD *const xd = &x->e_mbd;

  const int bsl = b_width_log2_lookup[bsize], hbs = (1 << bsl) / 4;
  int ctx;
  PARTITION_TYPE partition;
  BLOCK_SIZE subsize;

  if (mi_row >= cm->mi_rows || mi_col >= cm->mi_cols) return;

  if (bsize >= BLOCK_8X8) {
    const int idx_str = xd->mi_stride * mi_row + mi_col;
    MODE_INFO **mi_8x8 = cm->mi_grid_visible + idx_str;
    ctx = partition_plane_context(xd, mi_row, mi_col, bsize);
    subsize = mi_8x8[0]->sb_type;
  } else {
    ctx = 0;
    subsize = BLOCK_4X4;
  }

  partition = partition_lookup[bsl][subsize];
  if (output_enabled && bsize != BLOCK_4X4)
    td->counts->partition[ctx][partition]++;

  switch (partition) {
    case PARTITION_NONE:
      encode_b_rt(cpi, td, tile, tp, mi_row, mi_col, output_enabled, subsize,
                  &pc_tree->none);
      break;
    case PARTITION_VERT:
      encode_b_rt(cpi, td, tile, tp, mi_row, mi_col, output_enabled, subsize,
                  &pc_tree->vertical[0]);
      if (mi_col + hbs < cm->mi_cols && bsize > BLOCK_8X8) {
        encode_b_rt(cpi, td, tile, tp, mi_row, mi_col + hbs, output_enabled,
                    subsize, &pc_tree->vertical[1]);
      }
      break;
    case PARTITION_HORZ:
      encode_b_rt(cpi, td, tile, tp, mi_row, mi_col, output_enabled, subsize,
                  &pc_tree->horizontal[0]);
      if (mi_row + hbs < cm->mi_rows && bsize > BLOCK_8X8) {
        encode_b_rt(cpi, td, tile, tp, mi_row + hbs, mi_col, output_enabled,
                    subsize, &pc_tree->horizontal[1]);
      }
      break;
    default:
      assert(partition == PARTITION_SPLIT);
      subsize = get_subsize(bsize, PARTITION_SPLIT);
      encode_sb_rt(cpi, td, tile, tp, mi_row, mi_col, output_enabled, subsize,
                   pc_tree->u.split[0]);
      encode_sb_rt(cpi, td, tile, tp, mi_row, mi_col + hbs, output_enabled,
                   subsize, pc_tree->u.split[1]);
      encode_sb_rt(cpi, td, tile, tp, mi_row + hbs, mi_col, output_enabled,
                   subsize, pc_tree->u.split[2]);
      encode_sb_rt(cpi, td, tile, tp, mi_row + hbs, mi_col + hbs,
                   output_enabled, subsize, pc_tree->u.split[3]);
      break;
  }

  if (partition != PARTITION_SPLIT || bsize == BLOCK_8X8)
    update_partition_context(xd, mi_row, mi_col, subsize, bsize);
}

#if !CONFIG_REALTIME_ONLY
static void rd_use_partition(VP9_COMP *cpi, ThreadData *td,
                             TileDataEnc *tile_data, MODE_INFO **mi_8x8,
                             TOKENEXTRA **tp, int mi_row, int mi_col,
                             BLOCK_SIZE bsize, int *rate, int64_t *dist,
                             int do_recon, PC_TREE *pc_tree) {
  VP9_COMMON *const cm = &cpi->common;
  TileInfo *const tile_info = &tile_data->tile_info;
  MACROBLOCK *const x = &td->mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  const int mis = cm->mi_stride;
  const int bsl = b_width_log2_lookup[bsize];
  const int mi_step = num_4x4_blocks_wide_lookup[bsize] / 2;
  const int bss = (1 << bsl) / 4;
  int i, pl;
  PARTITION_TYPE partition = PARTITION_NONE;
  BLOCK_SIZE subsize;
  ENTROPY_CONTEXT l[16 * MAX_MB_PLANE], a[16 * MAX_MB_PLANE];
  PARTITION_CONTEXT sl[8], sa[8];
  RD_COST last_part_rdc, none_rdc, chosen_rdc;
  BLOCK_SIZE sub_subsize = BLOCK_4X4;
  int splits_below = 0;
  BLOCK_SIZE bs_type = mi_8x8[0]->sb_type;
  int do_partition_search = 1;
  PICK_MODE_CONTEXT *ctx = &pc_tree->none;

  if (mi_row >= cm->mi_rows || mi_col >= cm->mi_cols) return;

  assert(num_4x4_blocks_wide_lookup[bsize] ==
         num_4x4_blocks_high_lookup[bsize]);

  vp9_rd_cost_reset(&last_part_rdc);
  vp9_rd_cost_reset(&none_rdc);
  vp9_rd_cost_reset(&chosen_rdc);

  partition = partition_lookup[bsl][bs_type];
  subsize = get_subsize(bsize, partition);

  pc_tree->partitioning = partition;
  save_context(x, mi_row, mi_col, a, l, sa, sl, bsize);

  if (bsize == BLOCK_16X16 && cpi->oxcf.aq_mode != NO_AQ) {
    set_offsets(cpi, tile_info, x, mi_row, mi_col, bsize);
    x->mb_energy = vp9_block_energy(cpi, x, bsize);
  }

  if (do_partition_search &&
      cpi->sf.partition_search_type == SEARCH_PARTITION &&
      cpi->sf.adjust_partitioning_from_last_frame) {
    // Check if any of the sub blocks are further split.
    if (partition == PARTITION_SPLIT && subsize > BLOCK_8X8) {
      sub_subsize = get_subsize(subsize, PARTITION_SPLIT);
      splits_below = 1;
      for (i = 0; i < 4; i++) {
        int jj = i >> 1, ii = i & 0x01;
        MODE_INFO *this_mi = mi_8x8[jj * bss * mis + ii * bss];
        if (this_mi && this_mi->sb_type >= sub_subsize) {
          splits_below = 0;
        }
      }
    }

    // If partition is not none try none unless each of the 4 splits are split
    // even further..
    if (partition != PARTITION_NONE && !splits_below &&
        mi_row + (mi_step >> 1) < cm->mi_rows &&
        mi_col + (mi_step >> 1) < cm->mi_cols) {
      pc_tree->partitioning = PARTITION_NONE;
      rd_pick_sb_modes(cpi, tile_data, x, mi_row, mi_col, &none_rdc, bsize, ctx,
                       INT_MAX, INT64_MAX);

      pl = partition_plane_context(xd, mi_row, mi_col, bsize);

      if (none_rdc.rate < INT_MAX) {
        none_rdc.rate += cpi->partition_cost[pl][PARTITION_NONE];
        none_rdc.rdcost =
            RDCOST(x->rdmult, x->rddiv, none_rdc.rate, none_rdc.dist);
      }

      restore_context(x, mi_row, mi_col, a, l, sa, sl, bsize);
      mi_8x8[0]->sb_type = bs_type;
      pc_tree->partitioning = partition;
    }
  }

  switch (partition) {
    case PARTITION_NONE:
      rd_pick_sb_modes(cpi, tile_data, x, mi_row, mi_col, &last_part_rdc, bsize,
                       ctx, INT_MAX, INT64_MAX);
      break;
    case PARTITION_HORZ:
      pc_tree->horizontal[0].skip_ref_frame_mask = 0;
      rd_pick_sb_modes(cpi, tile_data, x, mi_row, mi_col, &last_part_rdc,
                       subsize, &pc_tree->horizontal[0], INT_MAX, INT64_MAX);
      if (last_part_rdc.rate != INT_MAX && bsize >= BLOCK_8X8 &&
          mi_row + (mi_step >> 1) < cm->mi_rows) {
        RD_COST tmp_rdc;
        PICK_MODE_CONTEXT *hctx = &pc_tree->horizontal[0];
        vp9_rd_cost_init(&tmp_rdc);
        update_state(cpi, td, hctx, mi_row, mi_col, subsize, 0);
        encode_superblock(cpi, td, tp, 0, mi_row, mi_col, subsize, hctx);
        pc_tree->horizontal[1].skip_ref_frame_mask = 0;
        rd_pick_sb_modes(cpi, tile_data, x, mi_row + (mi_step >> 1), mi_col,
                         &tmp_rdc, subsize, &pc_tree->horizontal[1], INT_MAX,
                         INT64_MAX);
        if (tmp_rdc.rate == INT_MAX || tmp_rdc.dist == INT64_MAX) {
          vp9_rd_cost_reset(&last_part_rdc);
          break;
        }
        last_part_rdc.rate += tmp_rdc.rate;
        last_part_rdc.dist += tmp_rdc.dist;
        last_part_rdc.rdcost += tmp_rdc.rdcost;
      }
      break;
    case PARTITION_VERT:
      pc_tree->vertical[0].skip_ref_frame_mask = 0;
      rd_pick_sb_modes(cpi, tile_data, x, mi_row, mi_col, &last_part_rdc,
                       subsize, &pc_tree->vertical[0], INT_MAX, INT64_MAX);
      if (last_part_rdc.rate != INT_MAX && bsize >= BLOCK_8X8 &&
          mi_col + (mi_step >> 1) < cm->mi_cols) {
        RD_COST tmp_rdc;
        PICK_MODE_CONTEXT *vctx = &pc_tree->vertical[0];
        vp9_rd_cost_init(&tmp_rdc);
        update_state(cpi, td, vctx, mi_row, mi_col, subsize, 0);
        encode_superblock(cpi, td, tp, 0, mi_row, mi_col, subsize, vctx);
        pc_tree->vertical[bsize > BLOCK_8X8].skip_ref_frame_mask = 0;
        rd_pick_sb_modes(
            cpi, tile_data, x, mi_row, mi_col + (mi_step >> 1), &tmp_rdc,
            subsize, &pc_tree->vertical[bsize > BLOCK_8X8], INT_MAX, INT64_MAX);
        if (tmp_rdc.rate == INT_MAX || tmp_rdc.dist == INT64_MAX) {
          vp9_rd_cost_reset(&last_part_rdc);
          break;
        }
        last_part_rdc.rate += tmp_rdc.rate;
        last_part_rdc.dist += tmp_rdc.dist;
        last_part_rdc.rdcost += tmp_rdc.rdcost;
      }
      break;
    default:
      assert(partition == PARTITION_SPLIT);
      if (bsize == BLOCK_8X8) {
        rd_pick_sb_modes(cpi, tile_data, x, mi_row, mi_col, &last_part_rdc,
                         subsize, pc_tree->u.leaf_split[0], INT_MAX, INT64_MAX);
        break;
      }
      last_part_rdc.rate = 0;
      last_part_rdc.dist = 0;
      last_part_rdc.rdcost = 0;
      for (i = 0; i < 4; i++) {
        int x_idx = (i & 1) * (mi_step >> 1);
        int y_idx = (i >> 1) * (mi_step >> 1);
        int jj = i >> 1, ii = i & 0x01;
        RD_COST tmp_rdc;
        if ((mi_row + y_idx >= cm->mi_rows) || (mi_col + x_idx >= cm->mi_cols))
          continue;

        vp9_rd_cost_init(&tmp_rdc);
        rd_use_partition(cpi, td, tile_data, mi_8x8 + jj * bss * mis + ii * bss,
                         tp, mi_row + y_idx, mi_col + x_idx, subsize,
                         &tmp_rdc.rate, &tmp_rdc.dist, i != 3,
                         pc_tree->u.split[i]);
        if (tmp_rdc.rate == INT_MAX || tmp_rdc.dist == INT64_MAX) {
          vp9_rd_cost_reset(&last_part_rdc);
          break;
        }
        last_part_rdc.rate += tmp_rdc.rate;
        last_part_rdc.dist += tmp_rdc.dist;
      }
      break;
  }

  pl = partition_plane_context(xd, mi_row, mi_col, bsize);
  if (last_part_rdc.rate < INT_MAX) {
    last_part_rdc.rate += cpi->partition_cost[pl][partition];
    last_part_rdc.rdcost =
        RDCOST(x->rdmult, x->rddiv, last_part_rdc.rate, last_part_rdc.dist);
  }

  if (do_partition_search && cpi->sf.adjust_partitioning_from_last_frame &&
      cpi->sf.partition_search_type == SEARCH_PARTITION &&
      partition != PARTITION_SPLIT && bsize > BLOCK_8X8 &&
      (mi_row + mi_step < cm->mi_rows ||
       mi_row + (mi_step >> 1) == cm->mi_rows) &&
      (mi_col + mi_step < cm->mi_cols ||
       mi_col + (mi_step >> 1) == cm->mi_cols)) {
    BLOCK_SIZE split_subsize = get_subsize(bsize, PARTITION_SPLIT);
    chosen_rdc.rate = 0;
    chosen_rdc.dist = 0;
    restore_context(x, mi_row, mi_col, a, l, sa, sl, bsize);
    pc_tree->partitioning = PARTITION_SPLIT;

    // Split partition.
    for (i = 0; i < 4; i++) {
      int x_idx = (i & 1) * (mi_step >> 1);
      int y_idx = (i >> 1) * (mi_step >> 1);
      RD_COST tmp_rdc;

      if ((mi_row + y_idx >= cm->mi_rows) || (mi_col + x_idx >= cm->mi_cols))
        continue;

      save_context(x, mi_row, mi_col, a, l, sa, sl, bsize);
      pc_tree->u.split[i]->partitioning = PARTITION_NONE;
      rd_pick_sb_modes(cpi, tile_data, x, mi_row + y_idx, mi_col + x_idx,
                       &tmp_rdc, split_subsize, &pc_tree->u.split[i]->none,
                       INT_MAX, INT64_MAX);

      restore_context(x, mi_row, mi_col, a, l, sa, sl, bsize);

      if (tmp_rdc.rate == INT_MAX || tmp_rdc.dist == INT64_MAX) {
        vp9_rd_cost_reset(&chosen_rdc);
        break;
      }

      chosen_rdc.rate += tmp_rdc.rate;
      chosen_rdc.dist += tmp_rdc.dist;

      if (i != 3)
        encode_sb(cpi, td, tile_info, tp, mi_row + y_idx, mi_col + x_idx, 0,
                  split_subsize, pc_tree->u.split[i]);

      pl = partition_plane_context(xd, mi_row + y_idx, mi_col + x_idx,
                                   split_subsize);
      chosen_rdc.rate += cpi->partition_cost[pl][PARTITION_NONE];
    }
    pl = partition_plane_context(xd, mi_row, mi_col, bsize);
    if (chosen_rdc.rate < INT_MAX) {
      chosen_rdc.rate += cpi->partition_cost[pl][PARTITION_SPLIT];
      chosen_rdc.rdcost =
          RDCOST(x->rdmult, x->rddiv, chosen_rdc.rate, chosen_rdc.dist);
    }
  }

  // If last_part is better set the partitioning to that.
  if (last_part_rdc.rdcost < chosen_rdc.rdcost) {
    mi_8x8[0]->sb_type = bsize;
    if (bsize >= BLOCK_8X8) pc_tree->partitioning = partition;
    chosen_rdc = last_part_rdc;
  }
  // If none was better set the partitioning to that.
  if (none_rdc.rdcost < chosen_rdc.rdcost) {
    if (bsize >= BLOCK_8X8) pc_tree->partitioning = PARTITION_NONE;
    chosen_rdc = none_rdc;
  }

  restore_context(x, mi_row, mi_col, a, l, sa, sl, bsize);

  // We must have chosen a partitioning and encoding or we'll fail later on.
  // No other opportunities for success.
  if (bsize == BLOCK_64X64)
    assert(chosen_rdc.rate < INT_MAX && chosen_rdc.dist < INT64_MAX);

  if (do_recon) {
    int output_enabled = (bsize == BLOCK_64X64);
    encode_sb(cpi, td, tile_info, tp, mi_row, mi_col, output_enabled, bsize,
              pc_tree);
  }

  *rate = chosen_rdc.rate;
  *dist = chosen_rdc.dist;
}

static const BLOCK_SIZE min_partition_size[BLOCK_SIZES] = {
  BLOCK_4X4,   BLOCK_4X4,   BLOCK_4X4,  BLOCK_4X4, BLOCK_4X4,
  BLOCK_4X4,   BLOCK_8X8,   BLOCK_8X8,  BLOCK_8X8, BLOCK_16X16,
  BLOCK_16X16, BLOCK_16X16, BLOCK_16X16
};

static const BLOCK_SIZE max_partition_size[BLOCK_SIZES] = {
  BLOCK_8X8,   BLOCK_16X16, BLOCK_16X16, BLOCK_16X16, BLOCK_32X32,
  BLOCK_32X32, BLOCK_32X32, BLOCK_64X64, BLOCK_64X64, BLOCK_64X64,
  BLOCK_64X64, BLOCK_64X64, BLOCK_64X64
};

// Look at all the mode_info entries for blocks that are part of this
// partition and find the min and max values for sb_type.
// At the moment this is designed to work on a 64x64 SB but could be
// adjusted to use a size parameter.
//
// The min and max are assumed to have been initialized prior to calling this
// function so repeat calls can accumulate a min and max of more than one sb64.
static void get_sb_partition_size_range(MACROBLOCKD *xd, MODE_INFO **mi_8x8,
                                        BLOCK_SIZE *min_block_size,
                                        BLOCK_SIZE *max_block_size,
                                        int bs_hist[BLOCK_SIZES]) {
  int sb_width_in_blocks = MI_BLOCK_SIZE;
  int sb_height_in_blocks = MI_BLOCK_SIZE;
  int i, j;
  int index = 0;

  // Check the sb_type for each block that belongs to this region.
  for (i = 0; i < sb_height_in_blocks; ++i) {
    for (j = 0; j < sb_width_in_blocks; ++j) {
      MODE_INFO *mi = mi_8x8[index + j];
      BLOCK_SIZE sb_type = mi ? mi->sb_type : 0;
      bs_hist[sb_type]++;
      *min_block_size = VPXMIN(*min_block_size, sb_type);
      *max_block_size = VPXMAX(*max_block_size, sb_type);
    }
    index += xd->mi_stride;
  }
}

// Next square block size less or equal than current block size.
static const BLOCK_SIZE next_square_size[BLOCK_SIZES] = {
  BLOCK_4X4,   BLOCK_4X4,   BLOCK_4X4,   BLOCK_8X8,   BLOCK_8X8,
  BLOCK_8X8,   BLOCK_16X16, BLOCK_16X16, BLOCK_16X16, BLOCK_32X32,
  BLOCK_32X32, BLOCK_32X32, BLOCK_64X64
};

// Look at neighboring blocks and set a min and max partition size based on
// what they chose.
static void rd_auto_partition_range(VP9_COMP *cpi, const TileInfo *const tile,
                                    MACROBLOCKD *const xd, int mi_row,
                                    int mi_col, BLOCK_SIZE *min_block_size,
                                    BLOCK_SIZE *max_block_size) {
  VP9_COMMON *const cm = &cpi->common;
  MODE_INFO **mi = xd->mi;
  const int left_in_image = !!xd->left_mi;
  const int above_in_image = !!xd->above_mi;
  const int row8x8_remaining = tile->mi_row_end - mi_row;
  const int col8x8_remaining = tile->mi_col_end - mi_col;
  int bh, bw;
  BLOCK_SIZE min_size = BLOCK_4X4;
  BLOCK_SIZE max_size = BLOCK_64X64;
  int bs_hist[BLOCK_SIZES] = { 0 };

  // Trap case where we do not have a prediction.
  if (left_in_image || above_in_image || cm->frame_type != KEY_FRAME) {
    // Default "min to max" and "max to min"
    min_size = BLOCK_64X64;
    max_size = BLOCK_4X4;

    // NOTE: each call to get_sb_partition_size_range() uses the previous
    // passed in values for min and max as a starting point.
    // Find the min and max partition used in previous frame at this location
    if (cm->frame_type != KEY_FRAME) {
      MODE_INFO **prev_mi =
          &cm->prev_mi_grid_visible[mi_row * xd->mi_stride + mi_col];
      get_sb_partition_size_range(xd, prev_mi, &min_size, &max_size, bs_hist);
    }
    // Find the min and max partition sizes used in the left SB64
    if (left_in_image) {
      MODE_INFO **left_sb64_mi = &mi[-MI_BLOCK_SIZE];
      get_sb_partition_size_range(xd, left_sb64_mi, &min_size, &max_size,
                                  bs_hist);
    }
    // Find the min and max partition sizes used in the above SB64.
    if (above_in_image) {
      MODE_INFO **above_sb64_mi = &mi[-xd->mi_stride * MI_BLOCK_SIZE];
      get_sb_partition_size_range(xd, above_sb64_mi, &min_size, &max_size,
                                  bs_hist);
    }

    // Adjust observed min and max for "relaxed" auto partition case.
    if (cpi->sf.auto_min_max_partition_size == RELAXED_NEIGHBORING_MIN_MAX) {
      min_size = min_partition_size[min_size];
      max_size = max_partition_size[max_size];
    }
  }

  // Check border cases where max and min from neighbors may not be legal.
  max_size = find_partition_size(max_size, row8x8_remaining, col8x8_remaining,
                                 &bh, &bw);
  // Test for blocks at the edge of the active image.
  // This may be the actual edge of the image or where there are formatting
  // bars.
  if (vp9_active_edge_sb(cpi, mi_row, mi_col)) {
    min_size = BLOCK_4X4;
  } else {
    min_size =
        VPXMIN(cpi->sf.rd_auto_partition_min_limit, VPXMIN(min_size, max_size));
  }

  // When use_square_partition_only is true, make sure at least one square
  // partition is allowed by selecting the next smaller square size as
  // *min_block_size.
  if (cpi->sf.use_square_partition_only &&
      next_square_size[max_size] < min_size) {
    min_size = next_square_size[max_size];
  }

  *min_block_size = min_size;
  *max_block_size = max_size;
}

// TODO(jingning) refactor functions setting partition search range
static void set_partition_range(VP9_COMMON *cm, MACROBLOCKD *xd, int mi_row,
                                int mi_col, BLOCK_SIZE bsize,
                                BLOCK_SIZE *min_bs, BLOCK_SIZE *max_bs) {
  int mi_width = num_8x8_blocks_wide_lookup[bsize];
  int mi_height = num_8x8_blocks_high_lookup[bsize];
  int idx, idy;

  MODE_INFO *mi;
  const int idx_str = cm->mi_stride * mi_row + mi_col;
  MODE_INFO **prev_mi = &cm->prev_mi_grid_visible[idx_str];
  BLOCK_SIZE bs, min_size, max_size;

  min_size = BLOCK_64X64;
  max_size = BLOCK_4X4;

  for (idy = 0; idy < mi_height; ++idy) {
    for (idx = 0; idx < mi_width; ++idx) {
      mi = prev_mi[idy * cm->mi_stride + idx];
      bs = mi ? mi->sb_type : bsize;
      min_size = VPXMIN(min_size, bs);
      max_size = VPXMAX(max_size, bs);
    }
  }

  if (xd->left_mi) {
    for (idy = 0; idy < mi_height; ++idy) {
      mi = xd->mi[idy * cm->mi_stride - 1];
      bs = mi ? mi->sb_type : bsize;
      min_size = VPXMIN(min_size, bs);
      max_size = VPXMAX(max_size, bs);
    }
  }

  if (xd->above_mi) {
    for (idx = 0; idx < mi_width; ++idx) {
      mi = xd->mi[idx - cm->mi_stride];
      bs = mi ? mi->sb_type : bsize;
      min_size = VPXMIN(min_size, bs);
      max_size = VPXMAX(max_size, bs);
    }
  }

  if (min_size == max_size) {
    min_size = min_partition_size[min_size];
    max_size = max_partition_size[max_size];
  }

  *min_bs = min_size;
  *max_bs = max_size;
}
#endif  // !CONFIG_REALTIME_ONLY

static INLINE void store_pred_mv(MACROBLOCK *x, PICK_MODE_CONTEXT *ctx) {
  memcpy(ctx->pred_mv, x->pred_mv, sizeof(x->pred_mv));
}

static INLINE void load_pred_mv(MACROBLOCK *x, PICK_MODE_CONTEXT *ctx) {
  memcpy(x->pred_mv, ctx->pred_mv, sizeof(x->pred_mv));
}

// Calculate prediction based on the given input features and neural net config.
// Assume there are no more than NN_MAX_NODES_PER_LAYER nodes in each hidden
// layer.
static void nn_predict(const float *features, const NN_CONFIG *nn_config,
                       float *output) {
  int num_input_nodes = nn_config->num_inputs;
  int buf_index = 0;
  float buf[2][NN_MAX_NODES_PER_LAYER];
  const float *input_nodes = features;

  // Propagate hidden layers.
  const int num_layers = nn_config->num_hidden_layers;
  int layer, node, i;
  assert(num_layers <= NN_MAX_HIDDEN_LAYERS);
  for (layer = 0; layer < num_layers; ++layer) {
    const float *weights = nn_config->weights[layer];
    const float *bias = nn_config->bias[layer];
    float *output_nodes = buf[buf_index];
    const int num_output_nodes = nn_config->num_hidden_nodes[layer];
    assert(num_output_nodes < NN_MAX_NODES_PER_LAYER);
    for (node = 0; node < num_output_nodes; ++node) {
      float val = 0.0f;
      for (i = 0; i < num_input_nodes; ++i) val += weights[i] * input_nodes[i];
      val += bias[node];
      // ReLU as activation function.
      val = VPXMAX(val, 0.0f);
      output_nodes[node] = val;
      weights += num_input_nodes;
    }
    num_input_nodes = num_output_nodes;
    input_nodes = output_nodes;
    buf_index = 1 - buf_index;
  }

  // Final output layer.
  {
    const float *weights = nn_config->weights[num_layers];
    for (node = 0; node < nn_config->num_outputs; ++node) {
      const float *bias = nn_config->bias[num_layers];
      float val = 0.0f;
      for (i = 0; i < num_input_nodes; ++i) val += weights[i] * input_nodes[i];
      output[node] = val + bias[node];
      weights += num_input_nodes;
    }
  }
}

#if !CONFIG_REALTIME_ONLY
#define FEATURES 7
// Machine-learning based partition search early termination.
// Return 1 to skip split and rect partitions.
static int ml_pruning_partition(VP9_COMMON *const cm, MACROBLOCKD *const xd,
                                PICK_MODE_CONTEXT *ctx, int mi_row, int mi_col,
                                BLOCK_SIZE bsize) {
  const int mag_mv =
      abs(ctx->mic.mv[0].as_mv.col) + abs(ctx->mic.mv[0].as_mv.row);
  const int left_in_image = !!xd->left_mi;
  const int above_in_image = !!xd->above_mi;
  MODE_INFO **prev_mi =
      &cm->prev_mi_grid_visible[mi_col + cm->mi_stride * mi_row];
  int above_par = 0;  // above_partitioning
  int left_par = 0;   // left_partitioning
  int last_par = 0;   // last_partitioning
  int offset = 0;
  int i;
  BLOCK_SIZE context_size;
  const NN_CONFIG *nn_config = NULL;
  const float *mean, *sd, *linear_weights;
  float nn_score, linear_score;
  float features[FEATURES];

  assert(b_width_log2_lookup[bsize] == b_height_log2_lookup[bsize]);
  vpx_clear_system_state();

  switch (bsize) {
    case BLOCK_64X64:
      offset = 0;
      nn_config = &vp9_partition_nnconfig_64x64;
      break;
    case BLOCK_32X32:
      offset = 8;
      nn_config = &vp9_partition_nnconfig_32x32;
      break;
    case BLOCK_16X16:
      offset = 16;
      nn_config = &vp9_partition_nnconfig_16x16;
      break;
    default: assert(0 && "Unexpected block size."); return 0;
  }

  if (above_in_image) {
    context_size = xd->above_mi->sb_type;
    if (context_size < bsize)
      above_par = 2;
    else if (context_size == bsize)
      above_par = 1;
  }

  if (left_in_image) {
    context_size = xd->left_mi->sb_type;
    if (context_size < bsize)
      left_par = 2;
    else if (context_size == bsize)
      left_par = 1;
  }

  if (prev_mi[0]) {
    context_size = prev_mi[0]->sb_type;
    if (context_size < bsize)
      last_par = 2;
    else if (context_size == bsize)
      last_par = 1;
  }

  mean = &vp9_partition_feature_mean[offset];
  sd = &vp9_partition_feature_std[offset];
  features[0] = ((float)ctx->rate - mean[0]) / sd[0];
  features[1] = ((float)ctx->dist - mean[1]) / sd[1];
  features[2] = ((float)mag_mv / 2 - mean[2]) * sd[2];
  features[3] = ((float)(left_par + above_par) / 2 - mean[3]) * sd[3];
  features[4] = ((float)ctx->sum_y_eobs - mean[4]) / sd[4];
  features[5] = ((float)cm->base_qindex - mean[5]) * sd[5];
  features[6] = ((float)last_par - mean[6]) * sd[6];

  // Predict using linear model.
  linear_weights = &vp9_partition_linear_weights[offset];
  linear_score = linear_weights[FEATURES];
  for (i = 0; i < FEATURES; ++i)
    linear_score += linear_weights[i] * features[i];
  if (linear_score > 0.1f) return 0;

  // Predict using neural net model.
  nn_predict(features, nn_config, &nn_score);

  if (linear_score < -0.0f && nn_score < 0.1f) return 1;
  if (nn_score < -0.0f && linear_score < 0.1f) return 1;
  return 0;
}
#undef FEATURES

#define FEATURES 4
// ML-based partition search breakout.
static int ml_predict_breakout(VP9_COMP *const cpi, BLOCK_SIZE bsize,
                               const MACROBLOCK *const x,
                               const RD_COST *const rd_cost) {
  DECLARE_ALIGNED(16, static const uint8_t, vp9_64_zeros[64]) = { 0 };
  const VP9_COMMON *const cm = &cpi->common;
  float features[FEATURES];
  const float *linear_weights = NULL;  // Linear model weights.
  float linear_score = 0.0f;
  const int qindex = cm->base_qindex;
  const int q_ctx = qindex >= 200 ? 0 : (qindex >= 150 ? 1 : 2);
  const int is_720p_or_larger = VPXMIN(cm->width, cm->height) >= 720;
  const int resolution_ctx = is_720p_or_larger ? 1 : 0;

  switch (bsize) {
    case BLOCK_64X64:
      linear_weights = vp9_partition_breakout_weights_64[resolution_ctx][q_ctx];
      break;
    case BLOCK_32X32:
      linear_weights = vp9_partition_breakout_weights_32[resolution_ctx][q_ctx];
      break;
    case BLOCK_16X16:
      linear_weights = vp9_partition_breakout_weights_16[resolution_ctx][q_ctx];
      break;
    case BLOCK_8X8:
      linear_weights = vp9_partition_breakout_weights_8[resolution_ctx][q_ctx];
      break;
    default: assert(0 && "Unexpected block size."); return 0;
  }
  if (!linear_weights) return 0;

  {  // Generate feature values.
#if CONFIG_VP9_HIGHBITDEPTH
    const int ac_q =
        vp9_ac_quant(cm->base_qindex, 0, cm->bit_depth) >> (x->e_mbd.bd - 8);
#else
    const int ac_q = vp9_ac_quant(qindex, 0, cm->bit_depth);
#endif  // CONFIG_VP9_HIGHBITDEPTH
    const int num_pels_log2 = num_pels_log2_lookup[bsize];
    int feature_index = 0;
    unsigned int var, sse;
    float rate_f, dist_f;

#if CONFIG_VP9_HIGHBITDEPTH
    if (x->e_mbd.cur_buf->flags & YV12_FLAG_HIGHBITDEPTH) {
      var =
          vp9_high_get_sby_variance(cpi, &x->plane[0].src, bsize, x->e_mbd.bd);
    } else {
      var = cpi->fn_ptr[bsize].vf(x->plane[0].src.buf, x->plane[0].src.stride,
                                  vp9_64_zeros, 0, &sse);
    }
#else
    var = cpi->fn_ptr[bsize].vf(x->plane[0].src.buf, x->plane[0].src.stride,
                                vp9_64_zeros, 0, &sse);
#endif
    var = var >> num_pels_log2;

    vpx_clear_system_state();

    rate_f = (float)VPXMIN(rd_cost->rate, INT_MAX);
    dist_f = (float)(VPXMIN(rd_cost->dist, INT_MAX) >> num_pels_log2);
    rate_f =
        ((float)x->rdmult / 128.0f / 512.0f / (float)(1 << num_pels_log2)) *
        rate_f;

    features[feature_index++] = rate_f;
    features[feature_index++] = dist_f;
    features[feature_index++] = (float)var;
    features[feature_index++] = (float)ac_q;
    assert(feature_index == FEATURES);
  }

  {  // Calculate the output score.
    int i;
    linear_score = linear_weights[FEATURES];
    for (i = 0; i < FEATURES; ++i)
      linear_score += linear_weights[i] * features[i];
  }

  return linear_score >= cpi->sf.rd_ml_partition.search_breakout_thresh[q_ctx];
}
#undef FEATURES

#define FEATURES 8
#define LABELS 4
static void ml_prune_rect_partition(VP9_COMP *const cpi, MACROBLOCK *const x,
                                    BLOCK_SIZE bsize,
                                    const PC_TREE *const pc_tree,
                                    int *allow_horz, int *allow_vert,
                                    int64_t ref_rd) {
  const NN_CONFIG *nn_config = NULL;
  float score[LABELS] = {
    0.0f,
  };
  int thresh = -1;
  int i;
  (void)x;

  if (ref_rd <= 0 || ref_rd > 1000000000) return;

  switch (bsize) {
    case BLOCK_8X8: break;
    case BLOCK_16X16:
      nn_config = &vp9_rect_part_nnconfig_16;
      thresh = cpi->sf.rd_ml_partition.prune_rect_thresh[1];
      break;
    case BLOCK_32X32:
      nn_config = &vp9_rect_part_nnconfig_32;
      thresh = cpi->sf.rd_ml_partition.prune_rect_thresh[2];
      break;
    case BLOCK_64X64:
      nn_config = &vp9_rect_part_nnconfig_64;
      thresh = cpi->sf.rd_ml_partition.prune_rect_thresh[3];
      break;
    default: assert(0 && "Unexpected block size."); return;
  }
  if (!nn_config || thresh < 0) return;

  // Feature extraction and model score calculation.
  {
    const VP9_COMMON *const cm = &cpi->common;
#if CONFIG_VP9_HIGHBITDEPTH
    const int dc_q =
        vp9_dc_quant(cm->base_qindex, 0, cm->bit_depth) >> (x->e_mbd.bd - 8);
#else
    const int dc_q = vp9_dc_quant(cm->base_qindex, 0, cm->bit_depth);
#endif  // CONFIG_VP9_HIGHBITDEPTH
    const int bs = 4 * num_4x4_blocks_wide_lookup[bsize];
    int feature_index = 0;
    float features[FEATURES];

    features[feature_index++] = logf((float)dc_q + 1.0f);
    features[feature_index++] =
        (float)(pc_tree->partitioning == PARTITION_NONE);
    features[feature_index++] = logf((float)ref_rd / bs / bs + 1.0f);

    {
      const float norm_factor = 1.0f / ((float)ref_rd + 1.0f);
      const int64_t none_rdcost = pc_tree->none.rdcost;
      float rd_ratio = 2.0f;
      if (none_rdcost > 0 && none_rdcost < 1000000000)
        rd_ratio = (float)none_rdcost * norm_factor;
      features[feature_index++] = VPXMIN(rd_ratio, 2.0f);

      for (i = 0; i < 4; ++i) {
        const int64_t this_rd = pc_tree->u.split[i]->none.rdcost;
        const int rd_valid = this_rd > 0 && this_rd < 1000000000;
        // Ratio between sub-block RD and whole block RD.
        features[feature_index++] =
            rd_valid ? (float)this_rd * norm_factor : 1.0f;
      }
    }

    assert(feature_index == FEATURES);
    nn_predict(features, nn_config, score);
  }

  // Make decisions based on the model score.
  {
    int max_score = -1000;
    int horz = 0, vert = 0;
    int int_score[LABELS];
    for (i = 0; i < LABELS; ++i) {
      int_score[i] = (int)(100 * score[i]);
      max_score = VPXMAX(int_score[i], max_score);
    }
    thresh = max_score - thresh;
    for (i = 0; i < LABELS; ++i) {
      if (int_score[i] >= thresh) {
        if ((i >> 0) & 1) horz = 1;
        if ((i >> 1) & 1) vert = 1;
      }
    }
    *allow_horz = *allow_horz && horz;
    *allow_vert = *allow_vert && vert;
  }
}
#undef FEATURES
#undef LABELS

// Perform fast and coarse motion search for the given block. This is a
// pre-processing step for the ML based partition search speedup.
static void simple_motion_search(const VP9_COMP *const cpi, MACROBLOCK *const x,
                                 BLOCK_SIZE bsize, int mi_row, int mi_col,
                                 MV ref_mv, MV_REFERENCE_FRAME ref,
                                 uint8_t *const pred_buf) {
  const VP9_COMMON *const cm = &cpi->common;
  MACROBLOCKD *const xd = &x->e_mbd;
  MODE_INFO *const mi = xd->mi[0];
  YV12_BUFFER_CONFIG *yv12;
  YV12_BUFFER_CONFIG *scaled_ref_frame = vp9_get_scaled_ref_frame(cpi, ref);
  const int step_param = 1;
  const MvLimits tmp_mv_limits = x->mv_limits;
  const SEARCH_METHODS search_method = NSTEP;
  const int sadpb = x->sadperbit16;
  MV ref_mv_full = { ref_mv.row >> 3, ref_mv.col >> 3 };
  MV best_mv = { 0, 0 };
  int cost_list[5];
  struct buf_2d backup_pre[MAX_MB_PLANE] = { { 0, 0 } };

  if (scaled_ref_frame) {
    yv12 = scaled_ref_frame;
    // As reported in b/311294795, the reference buffer pointer needs to be
    // saved and restored after the search. Otherwise, it causes problems while
    // the reference frame scaling happens.
    for (int i = 0; i < MAX_MB_PLANE; i++) backup_pre[i] = xd->plane[i].pre[0];
  } else {
    yv12 = get_ref_frame_buffer(cpi, ref);
  }

  assert(yv12 != NULL);
  if (!yv12) return;
  vp9_setup_pre_planes(xd, 0, yv12, mi_row, mi_col, NULL);
  mi->ref_frame[0] = ref;
  mi->ref_frame[1] = NO_REF_FRAME;
  mi->sb_type = bsize;
  vp9_set_mv_search_range(&x->mv_limits, &ref_mv);
  vp9_full_pixel_search(cpi, x, bsize, &ref_mv_full, step_param, search_method,
                        sadpb, cond_cost_list(cpi, cost_list), &ref_mv,
                        &best_mv, 0, 0);
  best_mv.row *= 8;
  best_mv.col *= 8;
  x->mv_limits = tmp_mv_limits;
  mi->mv[0].as_mv = best_mv;

  // Restore reference buffer pointer.
  if (scaled_ref_frame) {
    for (int i = 0; i < MAX_MB_PLANE; i++) xd->plane[i].pre[0] = backup_pre[i];
  }

  set_ref_ptrs(cm, xd, mi->ref_frame[0], mi->ref_frame[1]);
  xd->plane[0].dst.buf = pred_buf;
  xd->plane[0].dst.stride = 64;
  vp9_build_inter_predictors_sby(xd, mi_row, mi_col, bsize);
}

// Use a neural net model to prune partition-none and partition-split search.
// Features used: QP; spatial block size contexts; variance of prediction
// residue after simple_motion_search.
#define FEATURES 12
static void ml_predict_var_rd_partitioning(const VP9_COMP *const cpi,
                                           MACROBLOCK *const x,
                                           PC_TREE *const pc_tree,
                                           BLOCK_SIZE bsize, int mi_row,
                                           int mi_col, int *none, int *split) {
  const VP9_COMMON *const cm = &cpi->common;
  const NN_CONFIG *nn_config = NULL;
  const MACROBLOCKD *const xd = &x->e_mbd;
#if CONFIG_VP9_HIGHBITDEPTH
  DECLARE_ALIGNED(16, uint8_t, pred_buffer[64 * 64 * 2]);
  uint8_t *const pred_buf = (xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH)
                                ? (CONVERT_TO_BYTEPTR(pred_buffer))
                                : pred_buffer;
#else
  DECLARE_ALIGNED(16, uint8_t, pred_buffer[64 * 64]);
  uint8_t *const pred_buf = pred_buffer;
#endif  // CONFIG_VP9_HIGHBITDEPTH
  const int speed = cpi->oxcf.speed;
  float thresh = 0.0f;

  switch (bsize) {
    case BLOCK_64X64:
      nn_config = &vp9_part_split_nnconfig_64;
      thresh = speed > 0 ? 2.8f : 3.0f;
      break;
    case BLOCK_32X32:
      nn_config = &vp9_part_split_nnconfig_32;
      thresh = speed > 0 ? 3.5f : 3.0f;
      break;
    case BLOCK_16X16:
      nn_config = &vp9_part_split_nnconfig_16;
      thresh = speed > 0 ? 3.8f : 4.0f;
      break;
    case BLOCK_8X8:
      nn_config = &vp9_part_split_nnconfig_8;
      if (cm->width >= 720 && cm->height >= 720)
        thresh = speed > 0 ? 2.5f : 2.0f;
      else
        thresh = speed > 0 ? 3.8f : 2.0f;
      break;
    default: assert(0 && "Unexpected block size."); return;
  }

  if (!nn_config) return;

  // Do a simple single motion search to find a prediction for current block.
  // The variance of the residue will be used as input features.
  {
    MV ref_mv;
    const MV_REFERENCE_FRAME ref =
        cpi->rc.is_src_frame_alt_ref ? ALTREF_FRAME : LAST_FRAME;
    // If bsize is 64x64, use zero MV as reference; otherwise, use MV result
    // of previous(larger) block as reference.
    if (bsize == BLOCK_64X64)
      ref_mv.row = ref_mv.col = 0;
    else
      ref_mv = pc_tree->mv;
    vp9_setup_src_planes(x, cpi->Source, mi_row, mi_col);
    simple_motion_search(cpi, x, bsize, mi_row, mi_col, ref_mv, ref, pred_buf);
    pc_tree->mv = x->e_mbd.mi[0]->mv[0].as_mv;
  }

  vpx_clear_system_state();

  {
    float features[FEATURES] = { 0.0f };
#if CONFIG_VP9_HIGHBITDEPTH
    const int dc_q =
        vp9_dc_quant(cm->base_qindex, 0, cm->bit_depth) >> (xd->bd - 8);
#else
    const int dc_q = vp9_dc_quant(cm->base_qindex, 0, cm->bit_depth);
#endif  // CONFIG_VP9_HIGHBITDEPTH
    int feature_idx = 0;
    float score;

    // Generate model input features.
    features[feature_idx++] = logf((float)dc_q + 1.0f);

    // Get the variance of the residue as input features.
    {
      const int bs = 4 * num_4x4_blocks_wide_lookup[bsize];
      const BLOCK_SIZE subsize = get_subsize(bsize, PARTITION_SPLIT);
      const uint8_t *pred = pred_buf;
      const uint8_t *src = x->plane[0].src.buf;
      const int src_stride = x->plane[0].src.stride;
      const int pred_stride = 64;
      unsigned int sse;
      // Variance of whole block.
      const unsigned int var =
          cpi->fn_ptr[bsize].vf(src, src_stride, pred, pred_stride, &sse);
      const float factor = (var == 0) ? 1.0f : (1.0f / (float)var);
      const int has_above = !!xd->above_mi;
      const int has_left = !!xd->left_mi;
      const BLOCK_SIZE above_bsize = has_above ? xd->above_mi->sb_type : bsize;
      const BLOCK_SIZE left_bsize = has_left ? xd->left_mi->sb_type : bsize;
      int i;

      features[feature_idx++] = (float)has_above;
      features[feature_idx++] = (float)b_width_log2_lookup[above_bsize];
      features[feature_idx++] = (float)b_height_log2_lookup[above_bsize];
      features[feature_idx++] = (float)has_left;
      features[feature_idx++] = (float)b_width_log2_lookup[left_bsize];
      features[feature_idx++] = (float)b_height_log2_lookup[left_bsize];
      features[feature_idx++] = logf((float)var + 1.0f);
      for (i = 0; i < 4; ++i) {
        const int x_idx = (i & 1) * bs / 2;
        const int y_idx = (i >> 1) * bs / 2;
        const int src_offset = y_idx * src_stride + x_idx;
        const int pred_offset = y_idx * pred_stride + x_idx;
        // Variance of quarter block.
        const unsigned int sub_var =
            cpi->fn_ptr[subsize].vf(src + src_offset, src_stride,
                                    pred + pred_offset, pred_stride, &sse);
        const float var_ratio = (var == 0) ? 1.0f : factor * (float)sub_var;
        features[feature_idx++] = var_ratio;
      }
    }
    assert(feature_idx == FEATURES);

    // Feed the features into the model to get the confidence score.
    nn_predict(features, nn_config, &score);

    // Higher score means that the model has higher confidence that the split
    // partition is better than the non-split partition. So if the score is
    // high enough, we skip the none-split partition search; if the score is
    // low enough, we skip the split partition search.
    if (score > thresh) *none = 0;
    if (score < -thresh) *split = 0;
  }
}
#undef FEATURES
#endif  // !CONFIG_REALTIME_ONLY

static double log_wiener_var(int64_t wiener_variance) {
  return log(1.0 + wiener_variance) / log(2.0);
}

static void build_kmeans_segmentation(VP9_COMP *cpi) {
  VP9_COMMON *cm = &cpi->common;
  BLOCK_SIZE bsize = BLOCK_64X64;
  KMEANS_DATA *kmeans_data;

  vp9_disable_segmentation(&cm->seg);
  if (cm->show_frame) {
    int mi_row, mi_col;
    cpi->kmeans_data_size = 0;
    cpi->kmeans_ctr_num = 8;

    for (mi_row = 0; mi_row < cm->mi_rows; mi_row += MI_BLOCK_SIZE) {
      for (mi_col = 0; mi_col < cm->mi_cols; mi_col += MI_BLOCK_SIZE) {
        int mb_row_start = mi_row >> 1;
        int mb_col_start = mi_col >> 1;
        int mb_row_end = VPXMIN(
            (mi_row + num_8x8_blocks_high_lookup[bsize]) >> 1, cm->mb_rows);
        int mb_col_end = VPXMIN(
            (mi_col + num_8x8_blocks_wide_lookup[bsize]) >> 1, cm->mb_cols);
        int row, col;
        int64_t wiener_variance = 0;

        for (row = mb_row_start; row < mb_row_end; ++row)
          for (col = mb_col_start; col < mb_col_end; ++col)
            wiener_variance += cpi->mb_wiener_variance[row * cm->mb_cols + col];

        wiener_variance /=
            (mb_row_end - mb_row_start) * (mb_col_end - mb_col_start);

#if CONFIG_MULTITHREAD
        pthread_mutex_lock(&cpi->kmeans_mutex);
#endif  // CONFIG_MULTITHREAD

        kmeans_data = &cpi->kmeans_data_arr[cpi->kmeans_data_size++];
        kmeans_data->value = log_wiener_var(wiener_variance);
        kmeans_data->pos = mi_row * cpi->kmeans_data_stride + mi_col;
#if CONFIG_MULTITHREAD
        pthread_mutex_unlock(&cpi->kmeans_mutex);
#endif  // CONFIG_MULTITHREAD
      }
    }

    vp9_kmeans(cpi->kmeans_ctr_ls, cpi->kmeans_boundary_ls,
               cpi->kmeans_count_ls, cpi->kmeans_ctr_num, cpi->kmeans_data_arr,
               cpi->kmeans_data_size);

    vp9_perceptual_aq_mode_setup(cpi, &cm->seg);
  }
}

#if !CONFIG_REALTIME_ONLY
static int wiener_var_segment(VP9_COMP *cpi, BLOCK_SIZE bsize, int mi_row,
                              int mi_col) {
  VP9_COMMON *cm = &cpi->common;
  int mb_row_start = mi_row >> 1;
  int mb_col_start = mi_col >> 1;
  int mb_row_end =
      VPXMIN((mi_row + num_8x8_blocks_high_lookup[bsize]) >> 1, cm->mb_rows);
  int mb_col_end =
      VPXMIN((mi_col + num_8x8_blocks_wide_lookup[bsize]) >> 1, cm->mb_cols);
  int row, col, idx;
  int64_t wiener_variance = 0;
  int segment_id;
  int8_t seg_hist[MAX_SEGMENTS] = { 0 };
  int8_t max_count = 0, max_index = -1;

  vpx_clear_system_state();

  assert(cpi->norm_wiener_variance > 0);

  for (row = mb_row_start; row < mb_row_end; ++row) {
    for (col = mb_col_start; col < mb_col_end; ++col) {
      wiener_variance = cpi->mb_wiener_variance[row * cm->mb_cols + col];
      segment_id =
          vp9_get_group_idx(log_wiener_var(wiener_variance),
                            cpi->kmeans_boundary_ls, cpi->kmeans_ctr_num);
      ++seg_hist[segment_id];
    }
  }

  for (idx = 0; idx < cpi->kmeans_ctr_num; ++idx) {
    if (seg_hist[idx] > max_count) {
      max_count = seg_hist[idx];
      max_index = idx;
    }
  }

  assert(max_index >= 0);
  segment_id = max_index;

  return segment_id;
}

static int get_rdmult_delta(VP9_COMP *cpi, BLOCK_SIZE bsize, int mi_row,
                            int mi_col, int orig_rdmult) {
  const int gf_group_index = cpi->twopass.gf_group.index;
  int64_t intra_cost = 0;
  int64_t mc_dep_cost = 0;
  int mi_wide = num_8x8_blocks_wide_lookup[bsize];
  int mi_high = num_8x8_blocks_high_lookup[bsize];
  int row, col;

  int dr = 0;
  double r0, rk, beta;

  TplDepFrame *tpl_frame;
  TplDepStats *tpl_stats;
  int tpl_stride;

  if (gf_group_index >= MAX_ARF_GOP_SIZE) return orig_rdmult;
  tpl_frame = &cpi->tpl_stats[gf_group_index];

  if (tpl_frame->is_valid == 0) return orig_rdmult;
  tpl_stats = tpl_frame->tpl_stats_ptr;
  tpl_stride = tpl_frame->stride;

  if (cpi->twopass.gf_group.layer_depth[gf_group_index] > 1) return orig_rdmult;

  if (cpi->ext_ratectrl.ready &&
      (cpi->ext_ratectrl.funcs.rc_type & VPX_RC_QP) != 0 &&
      cpi->ext_ratectrl.funcs.get_encodeframe_decision != NULL) {
    int sb_size = num_8x8_blocks_wide_lookup[BLOCK_64X64] * MI_SIZE;
    int sb_stride = (cpi->common.width + sb_size - 1) / sb_size;
    int sby = mi_row / 8;
    int sbx = mi_col / 8;
    return (int)((cpi->sb_mul_scale[sby * sb_stride + sbx] * orig_rdmult) /
                 256);
  }

  for (row = mi_row; row < mi_row + mi_high; ++row) {
    for (col = mi_col; col < mi_col + mi_wide; ++col) {
      TplDepStats *this_stats = &tpl_stats[row * tpl_stride + col];

      if (row >= cpi->common.mi_rows || col >= cpi->common.mi_cols) continue;

      intra_cost += this_stats->intra_cost;
      mc_dep_cost += this_stats->mc_dep_cost;
    }
  }

  vpx_clear_system_state();

  r0 = cpi->rd.r0;
  rk = (double)intra_cost / mc_dep_cost;
  beta = r0 / rk;
  dr = vp9_get_adaptive_rdmult(cpi, beta);

  dr = clamp(dr, orig_rdmult * 1 / 2, orig_rdmult * 3 / 2);
  dr = VPXMAX(1, dr);

  return dr;
}
#endif  // !CONFIG_REALTIME_ONLY

#if !CONFIG_REALTIME_ONLY
// TODO(jingning,jimbankoski,rbultje): properly skip partition types that are
// unlikely to be selected depending on previous rate-distortion optimization
// results, for encoding speed-up.
static int rd_pick_partition(VP9_COMP *cpi, ThreadData *td,
                             TileDataEnc *tile_data, TOKENEXTRA **tp,
                             int mi_row, int mi_col, BLOCK_SIZE bsize,
                             RD_COST *rd_cost, RD_COST best_rdc,
                             PC_TREE *pc_tree) {
  VP9_COMMON *const cm = &cpi->common;
  const VP9EncoderConfig *const oxcf = &cpi->oxcf;
  TileInfo *const tile_info = &tile_data->tile_info;
  MACROBLOCK *const x = &td->mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  const int mi_step = num_8x8_blocks_wide_lookup[bsize] / 2;
  ENTROPY_CONTEXT l[16 * MAX_MB_PLANE], a[16 * MAX_MB_PLANE];
  PARTITION_CONTEXT sl[8], sa[8];
  TOKENEXTRA *tp_orig = *tp;
  PICK_MODE_CONTEXT *const ctx = &pc_tree->none;
  int i;
  const int pl = partition_plane_context(xd, mi_row, mi_col, bsize);
  BLOCK_SIZE subsize;
  RD_COST this_rdc, sum_rdc;
  int do_split = bsize >= BLOCK_8X8;
  int do_rect = 1;
  INTERP_FILTER pred_interp_filter;

  // Override skipping rectangular partition operations for edge blocks
  const int force_horz_split = (mi_row + mi_step >= cm->mi_rows);
  const int force_vert_split = (mi_col + mi_step >= cm->mi_cols);
  const int xss = x->e_mbd.plane[1].subsampling_x;
  const int yss = x->e_mbd.plane[1].subsampling_y;

  BLOCK_SIZE min_size = x->min_partition_size;
  BLOCK_SIZE max_size = x->max_partition_size;

  int partition_none_allowed = !force_horz_split && !force_vert_split;
  int partition_horz_allowed =
      !force_vert_split && yss <= xss && bsize >= BLOCK_8X8;
  int partition_vert_allowed =
      !force_horz_split && xss <= yss && bsize >= BLOCK_8X8;

  int64_t dist_breakout_thr = cpi->sf.partition_search_breakout_thr.dist;
  int rate_breakout_thr = cpi->sf.partition_search_breakout_thr.rate;
  int must_split = 0;
  int should_encode_sb = 0;

  // Ref frames picked in the [i_th] quarter subblock during square partition
  // RD search. It may be used to prune ref frame selection of rect partitions.
  uint8_t ref_frames_used[4] = { 0, 0, 0, 0 };

  int partition_mul = x->cb_rdmult;

  (void)*tp_orig;

  assert(num_8x8_blocks_wide_lookup[bsize] ==
         num_8x8_blocks_high_lookup[bsize]);

  dist_breakout_thr >>=
      8 - (b_width_log2_lookup[bsize] + b_height_log2_lookup[bsize]);

  rate_breakout_thr *= num_pels_log2_lookup[bsize];

  vp9_rd_cost_init(&this_rdc);
  vp9_rd_cost_init(&sum_rdc);

  set_offsets(cpi, tile_info, x, mi_row, mi_col, bsize);

  if (oxcf->tuning == VP8_TUNE_SSIM) {
    set_ssim_rdmult(cpi, x, bsize, mi_row, mi_col, &partition_mul);
  }
  vp9_rd_cost_update(partition_mul, x->rddiv, &best_rdc);

  if (bsize == BLOCK_16X16 && cpi->oxcf.aq_mode != NO_AQ &&
      cpi->oxcf.aq_mode != LOOKAHEAD_AQ)
    x->mb_energy = vp9_block_energy(cpi, x, bsize);

  if (cpi->sf.cb_partition_search && bsize == BLOCK_16X16) {
    int cb_partition_search_ctrl =
        ((pc_tree->index == 0 || pc_tree->index == 3) +
         get_chessboard_index(cm->current_video_frame)) &
        0x1;

    if (cb_partition_search_ctrl && bsize > min_size && bsize < max_size)
      set_partition_range(cm, xd, mi_row, mi_col, bsize, &min_size, &max_size);
  }

  // Get sub block energy range
  if (bsize >= BLOCK_16X16) {
    int min_energy, max_energy;
    vp9_get_sub_block_energy(cpi, x, mi_row, mi_col, bsize, &min_energy,
                             &max_energy);
    must_split = (min_energy < -3) && (max_energy - min_energy > 2);
  }

  // Determine partition types in search according to the speed features.
  // The threshold set here has to be of square block size.
  if (cpi->sf.auto_min_max_partition_size) {
    partition_none_allowed &= (bsize <= max_size);
    partition_horz_allowed &=
        ((bsize <= max_size && bsize > min_size) || force_horz_split);
    partition_vert_allowed &=
        ((bsize <= max_size && bsize > min_size) || force_vert_split);
    do_split &= bsize > min_size;
  }

  if (cpi->sf.use_square_partition_only &&
      (bsize > cpi->sf.use_square_only_thresh_high ||
       bsize < cpi->sf.use_square_only_thresh_low)) {
    if (cpi->use_svc) {
      if (!vp9_active_h_edge(cpi, mi_row, mi_step) || x->e_mbd.lossless)
        partition_horz_allowed &= force_horz_split;
      if (!vp9_active_v_edge(cpi, mi_row, mi_step) || x->e_mbd.lossless)
        partition_vert_allowed &= force_vert_split;
    } else {
      partition_horz_allowed &= force_horz_split;
      partition_vert_allowed &= force_vert_split;
    }
  }

  save_context(x, mi_row, mi_col, a, l, sa, sl, bsize);

  pc_tree->partitioning = PARTITION_NONE;

  if (cpi->sf.rd_ml_partition.var_pruning && !frame_is_intra_only(cm)) {
    const int do_rd_ml_partition_var_pruning =
        partition_none_allowed && do_split &&
        mi_row + num_8x8_blocks_high_lookup[bsize] <= cm->mi_rows &&
        mi_col + num_8x8_blocks_wide_lookup[bsize] <= cm->mi_cols;
    if (do_rd_ml_partition_var_pruning) {
      ml_predict_var_rd_partitioning(cpi, x, pc_tree, bsize, mi_row, mi_col,
                                     &partition_none_allowed, &do_split);
      // ml_predict_var_rd_partitioning() may pruune out either
      // partition_none_allowed or do_split, but we should keep the
      // partition_none_allowed for 8x8 blocks unless disable_split_mask is
      // off (0).
      if (bsize == BLOCK_8X8 && cpi->sf.disable_split_mask &&
          partition_none_allowed == 0) {
        partition_none_allowed = 1;
      }
    } else {
      vp9_zero(pc_tree->mv);
    }
    if (bsize > BLOCK_8X8) {  // Store MV result as reference for subblocks.
      for (i = 0; i < 4; ++i) pc_tree->u.split[i]->mv = pc_tree->mv;
    }
  }

  // PARTITION_NONE
  if (partition_none_allowed) {
    rd_pick_sb_modes(cpi, tile_data, x, mi_row, mi_col, &this_rdc, bsize, ctx,
                     best_rdc.rate, best_rdc.dist);
    ctx->rdcost = this_rdc.rdcost;
    if (this_rdc.rate != INT_MAX) {
      if (cpi->sf.prune_ref_frame_for_rect_partitions) {
        const int ref1 = ctx->mic.ref_frame[0];
        const int ref2 = ctx->mic.ref_frame[1];
        for (i = 0; i < 4; ++i) {
          ref_frames_used[i] |= (1 << ref1);
          if (ref2 > 0) ref_frames_used[i] |= (1 << ref2);
        }
      }
      if (bsize >= BLOCK_8X8) {
        this_rdc.rate += cpi->partition_cost[pl][PARTITION_NONE];
        vp9_rd_cost_update(partition_mul, x->rddiv, &this_rdc);
      }

      if (this_rdc.rdcost < best_rdc.rdcost) {
        MODE_INFO *mi = xd->mi[0];

        best_rdc = this_rdc;
        should_encode_sb = 1;
        if (bsize >= BLOCK_8X8) pc_tree->partitioning = PARTITION_NONE;

        if (cpi->sf.rd_ml_partition.search_early_termination) {
          // Currently, the machine-learning based partition search early
          // termination is only used while bsize is 16x16, 32x32 or 64x64,
          // VPXMIN(cm->width, cm->height) >= 480, and speed = 0.
          if (!x->e_mbd.lossless &&
              !segfeature_active(&cm->seg, mi->segment_id, SEG_LVL_SKIP) &&
              ctx->mic.mode >= INTRA_MODES && bsize >= BLOCK_16X16) {
            if (ml_pruning_partition(cm, xd, ctx, mi_row, mi_col, bsize)) {
              do_split = 0;
              do_rect = 0;
            }
          }
        }

        if ((do_split || do_rect) && !x->e_mbd.lossless && ctx->skippable) {
          const int use_ml_based_breakout =
              cpi->sf.rd_ml_partition.search_breakout && cm->base_qindex >= 100;
          if (use_ml_based_breakout) {
            if (ml_predict_breakout(cpi, bsize, x, &this_rdc)) {
              do_split = 0;
              do_rect = 0;
            }
          } else {
            if (!cpi->sf.rd_ml_partition.search_early_termination) {
              if ((best_rdc.dist < (dist_breakout_thr >> 2)) ||
                  (best_rdc.dist < dist_breakout_thr &&
                   best_rdc.rate < rate_breakout_thr)) {
                do_split = 0;
                do_rect = 0;
              }
            }
          }
        }
      }
    }
    restore_context(x, mi_row, mi_col, a, l, sa, sl, bsize);
  } else {
    vp9_zero(ctx->pred_mv);
    ctx->mic.interp_filter = EIGHTTAP;
  }

  // store estimated motion vector
  store_pred_mv(x, ctx);

  // If the interp_filter is marked as SWITCHABLE_FILTERS, it was for an
  // intra block and used for context purposes.
  if (ctx->mic.interp_filter == SWITCHABLE_FILTERS) {
    pred_interp_filter = EIGHTTAP;
  } else {
    pred_interp_filter = ctx->mic.interp_filter;
  }

  // PARTITION_SPLIT
  // TODO(jingning): use the motion vectors given by the above search as
  // the starting point of motion search in the following partition type check.
  pc_tree->u.split[0]->none.rdcost = 0;
  pc_tree->u.split[1]->none.rdcost = 0;
  pc_tree->u.split[2]->none.rdcost = 0;
  pc_tree->u.split[3]->none.rdcost = 0;
  if (do_split || must_split) {
    subsize = get_subsize(bsize, PARTITION_SPLIT);
    load_pred_mv(x, ctx);
    if (bsize == BLOCK_8X8) {
      i = 4;
      if (cpi->sf.adaptive_pred_interp_filter && partition_none_allowed)
        pc_tree->u.leaf_split[0]->pred_interp_filter = pred_interp_filter;
      rd_pick_sb_modes(cpi, tile_data, x, mi_row, mi_col, &sum_rdc, subsize,
                       pc_tree->u.leaf_split[0], best_rdc.rate, best_rdc.dist);
      if (sum_rdc.rate == INT_MAX) {
        sum_rdc.rdcost = INT64_MAX;
      } else {
        if (cpi->sf.prune_ref_frame_for_rect_partitions) {
          const int ref1 = pc_tree->u.leaf_split[0]->mic.ref_frame[0];
          const int ref2 = pc_tree->u.leaf_split[0]->mic.ref_frame[1];
          for (i = 0; i < 4; ++i) {
            ref_frames_used[i] |= (1 << ref1);
            if (ref2 > 0) ref_frames_used[i] |= (1 << ref2);
          }
        }
      }
    } else {
      for (i = 0; (i < 4) && ((sum_rdc.rdcost < best_rdc.rdcost) || must_split);
           ++i) {
        const int x_idx = (i & 1) * mi_step;
        const int y_idx = (i >> 1) * mi_step;
        int found_best_rd = 0;
        RD_COST best_rdc_split;
        vp9_rd_cost_reset(&best_rdc_split);

        if (best_rdc.rate < INT_MAX && best_rdc.dist < INT64_MAX) {
          // A must split test here increases the number of sub
          // partitions but hurts metrics results quite a bit,
          // so this extra test is commented out pending
          // further tests on whether it adds much in terms of
          // visual quality.
          // (must_split) ? best_rdc.rate
          //              : best_rdc.rate - sum_rdc.rate,
          // (must_split) ? best_rdc.dist
          //              : best_rdc.dist - sum_rdc.dist,
          best_rdc_split.rate = best_rdc.rate - sum_rdc.rate;
          best_rdc_split.dist = best_rdc.dist - sum_rdc.dist;
        }

        if (mi_row + y_idx >= cm->mi_rows || mi_col + x_idx >= cm->mi_cols)
          continue;

        pc_tree->u.split[i]->index = i;
        if (cpi->sf.prune_ref_frame_for_rect_partitions)
          pc_tree->u.split[i]->none.rate = INT_MAX;
        found_best_rd = rd_pick_partition(
            cpi, td, tile_data, tp, mi_row + y_idx, mi_col + x_idx, subsize,
            &this_rdc, best_rdc_split, pc_tree->u.split[i]);

        if (found_best_rd == 0) {
          sum_rdc.rdcost = INT64_MAX;
          break;
        } else {
          if (cpi->sf.prune_ref_frame_for_rect_partitions &&
              pc_tree->u.split[i]->none.rate != INT_MAX) {
            const int ref1 = pc_tree->u.split[i]->none.mic.ref_frame[0];
            const int ref2 = pc_tree->u.split[i]->none.mic.ref_frame[1];
            ref_frames_used[i] |= (1 << ref1);
            if (ref2 > 0) ref_frames_used[i] |= (1 << ref2);
          }
          sum_rdc.rate += this_rdc.rate;
          sum_rdc.dist += this_rdc.dist;
          vp9_rd_cost_update(partition_mul, x->rddiv, &sum_rdc);
        }
      }
    }

    if (((sum_rdc.rdcost < best_rdc.rdcost) || must_split) && i == 4) {
      sum_rdc.rate += cpi->partition_cost[pl][PARTITION_SPLIT];
      vp9_rd_cost_update(partition_mul, x->rddiv, &sum_rdc);

      if ((sum_rdc.rdcost < best_rdc.rdcost) ||
          (must_split && (sum_rdc.dist < best_rdc.dist))) {
        best_rdc = sum_rdc;
        should_encode_sb = 1;
        pc_tree->partitioning = PARTITION_SPLIT;

        // Rate and distortion based partition search termination clause.
        if (!cpi->sf.rd_ml_partition.search_early_termination &&
            !x->e_mbd.lossless &&
            ((best_rdc.dist < (dist_breakout_thr >> 2)) ||
             (best_rdc.dist < dist_breakout_thr &&
              best_rdc.rate < rate_breakout_thr))) {
          do_rect = 0;
        }
      }
    } else {
      // skip rectangular partition test when larger block size
      // gives better rd cost
      if (cpi->sf.less_rectangular_check &&
          (bsize > cpi->sf.use_square_only_thresh_high ||
           best_rdc.dist < dist_breakout_thr))
        do_rect &= !partition_none_allowed;
    }
    restore_context(x, mi_row, mi_col, a, l, sa, sl, bsize);
  }

  pc_tree->horizontal[0].skip_ref_frame_mask = 0;
  pc_tree->horizontal[1].skip_ref_frame_mask = 0;
  pc_tree->vertical[0].skip_ref_frame_mask = 0;
  pc_tree->vertical[1].skip_ref_frame_mask = 0;
  if (cpi->sf.prune_ref_frame_for_rect_partitions) {
    uint8_t used_frames;
    used_frames = ref_frames_used[0] | ref_frames_used[1];
    if (used_frames) {
      pc_tree->horizontal[0].skip_ref_frame_mask = ~used_frames & 0xff;
    }
    used_frames = ref_frames_used[2] | ref_frames_used[3];
    if (used_frames) {
      pc_tree->horizontal[1].skip_ref_frame_mask = ~used_frames & 0xff;
    }
    used_frames = ref_frames_used[0] | ref_frames_used[2];
    if (used_frames) {
      pc_tree->vertical[0].skip_ref_frame_mask = ~used_frames & 0xff;
    }
    used_frames = ref_frames_used[1] | ref_frames_used[3];
    if (used_frames) {
      pc_tree->vertical[1].skip_ref_frame_mask = ~used_frames & 0xff;
    }
  }

  {
    const int do_ml_rect_partition_pruning =
        !frame_is_intra_only(cm) && !force_horz_split && !force_vert_split &&
        (partition_horz_allowed || partition_vert_allowed) && bsize > BLOCK_8X8;
    if (do_ml_rect_partition_pruning) {
      ml_prune_rect_partition(cpi, x, bsize, pc_tree, &partition_horz_allowed,
                              &partition_vert_allowed, best_rdc.rdcost);
    }
  }

  // PARTITION_HORZ
  if (partition_horz_allowed &&
      (do_rect || vp9_active_h_edge(cpi, mi_row, mi_step))) {
    const int part_mode_rate = cpi->partition_cost[pl][PARTITION_HORZ];
    subsize = get_subsize(bsize, PARTITION_HORZ);
    load_pred_mv(x, ctx);
    if (cpi->sf.adaptive_pred_interp_filter && bsize == BLOCK_8X8 &&
        partition_none_allowed)
      pc_tree->horizontal[0].pred_interp_filter = pred_interp_filter;
    rd_pick_sb_modes(cpi, tile_data, x, mi_row, mi_col, &sum_rdc, subsize,
                     &pc_tree->horizontal[0], best_rdc.rate - part_mode_rate,
                     best_rdc.dist);
    if (sum_rdc.rdcost < INT64_MAX) {
      sum_rdc.rate += part_mode_rate;
      vp9_rd_cost_update(partition_mul, x->rddiv, &sum_rdc);
    }

    if (sum_rdc.rdcost < best_rdc.rdcost && mi_row + mi_step < cm->mi_rows &&
        bsize > BLOCK_8X8) {
      PICK_MODE_CONTEXT *hctx = &pc_tree->horizontal[0];
      update_state(cpi, td, hctx, mi_row, mi_col, subsize, 0);
      encode_superblock(cpi, td, tp, 0, mi_row, mi_col, subsize, hctx);
      if (cpi->sf.adaptive_pred_interp_filter && bsize == BLOCK_8X8 &&
          partition_none_allowed)
        pc_tree->horizontal[1].pred_interp_filter = pred_interp_filter;
      rd_pick_sb_modes(cpi, tile_data, x, mi_row + mi_step, mi_col, &this_rdc,
                       subsize, &pc_tree->horizontal[1],
                       best_rdc.rate - sum_rdc.rate,
                       best_rdc.dist - sum_rdc.dist);
      if (this_rdc.rate == INT_MAX) {
        sum_rdc.rdcost = INT64_MAX;
      } else {
        sum_rdc.rate += this_rdc.rate;
        sum_rdc.dist += this_rdc.dist;
        vp9_rd_cost_update(partition_mul, x->rddiv, &sum_rdc);
      }
    }

    if (sum_rdc.rdcost < best_rdc.rdcost) {
      best_rdc = sum_rdc;
      should_encode_sb = 1;
      pc_tree->partitioning = PARTITION_HORZ;

      if (cpi->sf.less_rectangular_check &&
          bsize > cpi->sf.use_square_only_thresh_high)
        do_rect = 0;
    }
    restore_context(x, mi_row, mi_col, a, l, sa, sl, bsize);
  }

  // PARTITION_VERT
  if (partition_vert_allowed &&
      (do_rect || vp9_active_v_edge(cpi, mi_col, mi_step))) {
    const int part_mode_rate = cpi->partition_cost[pl][PARTITION_VERT];
    subsize = get_subsize(bsize, PARTITION_VERT);
    load_pred_mv(x, ctx);
    if (cpi->sf.adaptive_pred_interp_filter && bsize == BLOCK_8X8 &&
        partition_none_allowed)
      pc_tree->vertical[0].pred_interp_filter = pred_interp_filter;
    rd_pick_sb_modes(cpi, tile_data, x, mi_row, mi_col, &sum_rdc, subsize,
                     &pc_tree->vertical[0], best_rdc.rate - part_mode_rate,
                     best_rdc.dist);
    if (sum_rdc.rdcost < INT64_MAX) {
      sum_rdc.rate += part_mode_rate;
      vp9_rd_cost_update(partition_mul, x->rddiv, &sum_rdc);
    }

    if (sum_rdc.rdcost < best_rdc.rdcost && mi_col + mi_step < cm->mi_cols &&
        bsize > BLOCK_8X8) {
      update_state(cpi, td, &pc_tree->vertical[0], mi_row, mi_col, subsize, 0);
      encode_superblock(cpi, td, tp, 0, mi_row, mi_col, subsize,
                        &pc_tree->vertical[0]);
      if (cpi->sf.adaptive_pred_interp_filter && bsize == BLOCK_8X8 &&
          partition_none_allowed)
        pc_tree->vertical[1].pred_interp_filter = pred_interp_filter;
      rd_pick_sb_modes(cpi, tile_data, x, mi_row, mi_col + mi_step, &this_rdc,
                       subsize, &pc_tree->vertical[1],
                       best_rdc.rate - sum_rdc.rate,
                       best_rdc.dist - sum_rdc.dist);
      if (this_rdc.rate == INT_MAX) {
        sum_rdc.rdcost = INT64_MAX;
      } else {
        sum_rdc.rate += this_rdc.rate;
        sum_rdc.dist += this_rdc.dist;
        vp9_rd_cost_update(partition_mul, x->rddiv, &sum_rdc);
      }
    }

    if (sum_rdc.rdcost < best_rdc.rdcost) {
      best_rdc = sum_rdc;
      should_encode_sb = 1;
      pc_tree->partitioning = PARTITION_VERT;
    }
    restore_context(x, mi_row, mi_col, a, l, sa, sl, bsize);
  }

  if (bsize == BLOCK_64X64 && best_rdc.rdcost == INT64_MAX) {
    vp9_rd_cost_reset(&this_rdc);
    rd_pick_sb_modes(cpi, tile_data, x, mi_row, mi_col, &this_rdc, BLOCK_64X64,
                     ctx, INT_MAX, INT64_MAX);
    ctx->rdcost = this_rdc.rdcost;
    vp9_rd_cost_update(partition_mul, x->rddiv, &this_rdc);
    if (this_rdc.rdcost < best_rdc.rdcost) {
      best_rdc = this_rdc;
      should_encode_sb = 1;
      pc_tree->partitioning = PARTITION_NONE;
    }
  }

  *rd_cost = best_rdc;

  if (should_encode_sb && pc_tree->index != 3) {
    int output_enabled = (bsize == BLOCK_64X64);
#if CONFIG_COLLECT_COMPONENT_TIMING
    start_timing(cpi, encode_sb_time);
#endif
    encode_sb(cpi, td, tile_info, tp, mi_row, mi_col, output_enabled, bsize,
              pc_tree);
#if CONFIG_COLLECT_COMPONENT_TIMING
    end_timing(cpi, encode_sb_time);
#endif
  }

  if (bsize == BLOCK_64X64) {
    assert(tp_orig < *tp);
    assert(best_rdc.rate < INT_MAX);
    assert(best_rdc.dist < INT64_MAX);
  } else {
    assert(tp_orig == *tp);
  }

  return should_encode_sb;
}

static void encode_rd_sb_row(VP9_COMP *cpi, ThreadData *td,
                             TileDataEnc *tile_data, int mi_row,
                             TOKENEXTRA **tp) {
  VP9_COMMON *const cm = &cpi->common;
  TileInfo *const tile_info = &tile_data->tile_info;
  MACROBLOCK *const x = &td->mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  SPEED_FEATURES *const sf = &cpi->sf;
  const int mi_col_start = tile_info->mi_col_start;
  const int mi_col_end = tile_info->mi_col_end;
  int mi_col;
  const int sb_row = mi_row >> MI_BLOCK_SIZE_LOG2;
  const int num_sb_cols =
      get_num_cols(tile_data->tile_info, MI_BLOCK_SIZE_LOG2);
  int sb_col_in_tile;

  // Initialize the left context for the new SB row
  memset(&xd->left_context, 0, sizeof(xd->left_context));
  memset(xd->left_seg_context, 0, sizeof(xd->left_seg_context));

  // Code each SB in the row
  for (mi_col = mi_col_start, sb_col_in_tile = 0; mi_col < mi_col_end;
       mi_col += MI_BLOCK_SIZE, sb_col_in_tile++) {
    const struct segmentation *const seg = &cm->seg;
    int dummy_rate;
    int64_t dummy_dist;
    RD_COST dummy_rdc;
    int i;
    int seg_skip = 0;
    int orig_rdmult = cpi->rd.RDMULT;

    const int idx_str = cm->mi_stride * mi_row + mi_col;
    MODE_INFO **mi = cm->mi_grid_visible + idx_str;

    vp9_rd_cost_reset(&dummy_rdc);
    (*(cpi->row_mt_sync_read_ptr))(&tile_data->row_mt_sync, sb_row,
                                   sb_col_in_tile);

    if (sf->adaptive_pred_interp_filter) {
      for (i = 0; i < 64; ++i) td->leaf_tree[i].pred_interp_filter = SWITCHABLE;

      for (i = 0; i < 64; ++i) {
        td->pc_tree[i].vertical[0].pred_interp_filter = SWITCHABLE;
        td->pc_tree[i].vertical[1].pred_interp_filter = SWITCHABLE;
        td->pc_tree[i].horizontal[0].pred_interp_filter = SWITCHABLE;
        td->pc_tree[i].horizontal[1].pred_interp_filter = SWITCHABLE;
      }
    }

    for (i = 0; i < MAX_REF_FRAMES; ++i) {
      x->pred_mv[i].row = INT16_MAX;
      x->pred_mv[i].col = INT16_MAX;
    }
    td->pc_root->index = 0;

    if (seg->enabled) {
      const uint8_t *const map =
          seg->update_map ? cpi->segmentation_map : cm->last_frame_seg_map;
      int segment_id = get_segment_id(cm, map, BLOCK_64X64, mi_row, mi_col);
      seg_skip = segfeature_active(seg, segment_id, SEG_LVL_SKIP);
    }

    x->source_variance = UINT_MAX;

    x->cb_rdmult = orig_rdmult;

    if (sf->partition_search_type == FIXED_PARTITION || seg_skip) {
      const BLOCK_SIZE bsize =
          seg_skip ? BLOCK_64X64 : sf->always_this_block_size;
      set_offsets(cpi, tile_info, x, mi_row, mi_col, BLOCK_64X64);
      set_fixed_partitioning(cpi, tile_info, mi, mi_row, mi_col, bsize);
      rd_use_partition(cpi, td, tile_data, mi, tp, mi_row, mi_col, BLOCK_64X64,
                       &dummy_rate, &dummy_dist, 1, td->pc_root);
    } else if (sf->partition_search_type == VAR_BASED_PARTITION &&
               cm->frame_type != KEY_FRAME) {
      choose_partitioning(cpi, tile_info, x, mi_row, mi_col);
      rd_use_partition(cpi, td, tile_data, mi, tp, mi_row, mi_col, BLOCK_64X64,
                       &dummy_rate, &dummy_dist, 1, td->pc_root);
    } else {
      if (cpi->twopass.gf_group.index > 0 && cpi->sf.enable_tpl_model) {
        int dr =
            get_rdmult_delta(cpi, BLOCK_64X64, mi_row, mi_col, orig_rdmult);
        x->cb_rdmult = dr;
      }

      if (cpi->oxcf.aq_mode == PERCEPTUAL_AQ && cm->show_frame) {
        x->segment_id = wiener_var_segment(cpi, BLOCK_64X64, mi_row, mi_col);
        x->cb_rdmult = vp9_compute_rd_mult(
            cpi, vp9_get_qindex(&cm->seg, x->segment_id, cm->base_qindex));
      }

      // If required set upper and lower partition size limits
      if (sf->auto_min_max_partition_size) {
        set_offsets(cpi, tile_info, x, mi_row, mi_col, BLOCK_64X64);
        rd_auto_partition_range(cpi, tile_info, xd, mi_row, mi_col,
                                &x->min_partition_size, &x->max_partition_size);
      }
      td->pc_root->none.rdcost = 0;

#if CONFIG_COLLECT_COMPONENT_TIMING
      start_timing(cpi, rd_pick_partition_time);
#endif
      rd_pick_partition(cpi, td, tile_data, tp, mi_row, mi_col, BLOCK_64X64,
                        &dummy_rdc, dummy_rdc, td->pc_root);
#if CONFIG_COLLECT_COMPONENT_TIMING
      end_timing(cpi, rd_pick_partition_time);
#endif
    }
    (*(cpi->row_mt_sync_write_ptr))(&tile_data->row_mt_sync, sb_row,
                                    sb_col_in_tile, num_sb_cols);
  }
}
#endif  // !CONFIG_REALTIME_ONLY

static void init_encode_frame_mb_context(VP9_COMP *cpi) {
  MACROBLOCK *const x = &cpi->td.mb;
  VP9_COMMON *const cm = &cpi->common;
  MACROBLOCKD *const xd = &x->e_mbd;
  const int aligned_mi_cols = mi_cols_aligned_to_sb(cm->mi_cols);

  // Copy data over into macro block data structures.
  vp9_setup_src_planes(x, cpi->Source, 0, 0);

  vp9_setup_block_planes(&x->e_mbd, cm->subsampling_x, cm->subsampling_y);

  // Note: this memset assumes above_context[0], [1] and [2]
  // are allocated as part of the same buffer.
  memset(xd->above_context[0], 0,
         sizeof(*xd->above_context[0]) * 2 * aligned_mi_cols * MAX_MB_PLANE);
  memset(xd->above_seg_context, 0,
         sizeof(*xd->above_seg_context) * aligned_mi_cols);
}

static int check_dual_ref_flags(VP9_COMP *cpi) {
  const int ref_flags = cpi->ref_frame_flags;

  if (segfeature_active(&cpi->common.seg, 1, SEG_LVL_REF_FRAME)) {
    return 0;
  } else {
    return (!!(ref_flags & VP9_GOLD_FLAG) + !!(ref_flags & VP9_LAST_FLAG) +
            !!(ref_flags & VP9_ALT_FLAG)) >= 2;
  }
}

static void reset_skip_tx_size(VP9_COMMON *cm, TX_SIZE max_tx_size) {
  int mi_row, mi_col;
  const int mis = cm->mi_stride;
  MODE_INFO **mi_ptr = cm->mi_grid_visible;

  for (mi_row = 0; mi_row < cm->mi_rows; ++mi_row, mi_ptr += mis) {
    for (mi_col = 0; mi_col < cm->mi_cols; ++mi_col) {
      if (mi_ptr[mi_col]->tx_size > max_tx_size)
        mi_ptr[mi_col]->tx_size = max_tx_size;
    }
  }
}

static MV_REFERENCE_FRAME get_frame_type(const VP9_COMP *cpi) {
  if (frame_is_intra_only(&cpi->common))
    return INTRA_FRAME;
  else if (cpi->rc.is_src_frame_alt_ref && cpi->refresh_golden_frame)
    return ALTREF_FRAME;
  else if (cpi->refresh_golden_frame || cpi->refresh_alt_ref_frame)
    return GOLDEN_FRAME;
  else
    return LAST_FRAME;
}

static TX_MODE select_tx_mode(const VP9_COMP *cpi, MACROBLOCKD *const xd) {
  if (xd->lossless) return ONLY_4X4;
  if (cpi->common.frame_type == KEY_FRAME && cpi->sf.use_nonrd_pick_mode)
    return ALLOW_16X16;
  if (cpi->sf.tx_size_search_method == USE_LARGESTALL)
    return ALLOW_32X32;
  else if (cpi->sf.tx_size_search_method == USE_FULL_RD ||
           cpi->sf.tx_size_search_method == USE_TX_8X8)
    return TX_MODE_SELECT;
  else
    return cpi->common.tx_mode;
}

static void hybrid_intra_mode_search(VP9_COMP *cpi, MACROBLOCK *const x,
                                     RD_COST *rd_cost, BLOCK_SIZE bsize,
                                     PICK_MODE_CONTEXT *ctx) {
  if (!cpi->sf.nonrd_keyframe && bsize < BLOCK_16X16)
    vp9_rd_pick_intra_mode_sb(cpi, x, rd_cost, bsize, ctx, INT64_MAX);
  else
    vp9_pick_intra_mode(cpi, x, rd_cost, bsize, ctx);
}

static void hybrid_search_svc_baseiskey(VP9_COMP *cpi, MACROBLOCK *const x,
                                        RD_COST *rd_cost, BLOCK_SIZE bsize,
                                        PICK_MODE_CONTEXT *ctx,
                                        TileDataEnc *tile_data, int mi_row,
                                        int mi_col) {
  if (!cpi->sf.nonrd_keyframe && bsize <= BLOCK_8X8) {
    vp9_rd_pick_intra_mode_sb(cpi, x, rd_cost, bsize, ctx, INT64_MAX);
  } else {
    if (cpi->svc.disable_inter_layer_pred == INTER_LAYER_PRED_OFF)
      vp9_pick_intra_mode(cpi, x, rd_cost, bsize, ctx);
    else if (bsize >= BLOCK_8X8)
      vp9_pick_inter_mode(cpi, x, tile_data, mi_row, mi_col, rd_cost, bsize,
                          ctx);
    else
      vp9_pick_inter_mode_sub8x8(cpi, x, mi_row, mi_col, rd_cost, bsize, ctx);
  }
}

static void hybrid_search_scene_change(VP9_COMP *cpi, MACROBLOCK *const x,
                                       RD_COST *rd_cost, BLOCK_SIZE bsize,
                                       PICK_MODE_CONTEXT *ctx,
                                       TileDataEnc *tile_data, int mi_row,
                                       int mi_col) {
  if (!cpi->sf.nonrd_keyframe && bsize <= BLOCK_8X8) {
    vp9_rd_pick_intra_mode_sb(cpi, x, rd_cost, bsize, ctx, INT64_MAX);
  } else {
    vp9_pick_inter_mode(cpi, x, tile_data, mi_row, mi_col, rd_cost, bsize, ctx);
  }
}

static void nonrd_pick_sb_modes(VP9_COMP *cpi, TileDataEnc *tile_data,
                                MACROBLOCK *const x, int mi_row, int mi_col,
                                RD_COST *rd_cost, BLOCK_SIZE bsize,
                                PICK_MODE_CONTEXT *ctx) {
  VP9_COMMON *const cm = &cpi->common;
  TileInfo *const tile_info = &tile_data->tile_info;
  MACROBLOCKD *const xd = &x->e_mbd;
  MODE_INFO *mi;
  ENTROPY_CONTEXT l[16 * MAX_MB_PLANE], a[16 * MAX_MB_PLANE];
  BLOCK_SIZE bs = VPXMAX(bsize, BLOCK_8X8);  // processing unit block size
  const int num_4x4_blocks_wide = num_4x4_blocks_wide_lookup[bs];
  const int num_4x4_blocks_high = num_4x4_blocks_high_lookup[bs];
  int plane;

  set_offsets(cpi, tile_info, x, mi_row, mi_col, bsize);

  set_segment_index(cpi, x, mi_row, mi_col, bsize, 0);

  x->skip_recode = 0;

  mi = xd->mi[0];
  mi->sb_type = bsize;

  for (plane = 0; plane < MAX_MB_PLANE; ++plane) {
    struct macroblockd_plane *pd = &xd->plane[plane];
    memcpy(a + num_4x4_blocks_wide * plane, pd->above_context,
           (sizeof(a[0]) * num_4x4_blocks_wide) >> pd->subsampling_x);
    memcpy(l + num_4x4_blocks_high * plane, pd->left_context,
           (sizeof(l[0]) * num_4x4_blocks_high) >> pd->subsampling_y);
  }

  if (cpi->oxcf.aq_mode == CYCLIC_REFRESH_AQ && cm->seg.enabled)
    if (cyclic_refresh_segment_id_boosted(mi->segment_id))
      x->rdmult = vp9_cyclic_refresh_get_rdmult(cpi->cyclic_refresh);

  if (frame_is_intra_only(cm))
    hybrid_intra_mode_search(cpi, x, rd_cost, bsize, ctx);
  else if (cpi->svc.layer_context[cpi->svc.temporal_layer_id].is_key_frame)
    hybrid_search_svc_baseiskey(cpi, x, rd_cost, bsize, ctx, tile_data, mi_row,
                                mi_col);
  else if (segfeature_active(&cm->seg, mi->segment_id, SEG_LVL_SKIP))
    set_mode_info_seg_skip(x, cm->tx_mode, cm->interp_filter, rd_cost, bsize);
  else if (bsize >= BLOCK_8X8) {
    if (cpi->rc.hybrid_intra_scene_change)
      hybrid_search_scene_change(cpi, x, rd_cost, bsize, ctx, tile_data, mi_row,
                                 mi_col);
    else
      vp9_pick_inter_mode(cpi, x, tile_data, mi_row, mi_col, rd_cost, bsize,
                          ctx);
  } else {
    vp9_pick_inter_mode_sub8x8(cpi, x, mi_row, mi_col, rd_cost, bsize, ctx);
  }

  duplicate_mode_info_in_sb(cm, xd, mi_row, mi_col, bsize);

  for (plane = 0; plane < MAX_MB_PLANE; ++plane) {
    struct macroblockd_plane *pd = &xd->plane[plane];
    memcpy(pd->above_context, a + num_4x4_blocks_wide * plane,
           (sizeof(a[0]) * num_4x4_blocks_wide) >> pd->subsampling_x);
    memcpy(pd->left_context, l + num_4x4_blocks_high * plane,
           (sizeof(l[0]) * num_4x4_blocks_high) >> pd->subsampling_y);
  }

  if (rd_cost->rate == INT_MAX) vp9_rd_cost_reset(rd_cost);

  ctx->rate = rd_cost->rate;
  ctx->dist = rd_cost->dist;
}

static void fill_mode_info_sb(VP9_COMMON *cm, MACROBLOCK *x, int mi_row,
                              int mi_col, BLOCK_SIZE bsize, PC_TREE *pc_tree) {
  MACROBLOCKD *xd = &x->e_mbd;
  int bsl = b_width_log2_lookup[bsize], hbs = (1 << bsl) / 4;
  PARTITION_TYPE partition = pc_tree->partitioning;
  BLOCK_SIZE subsize = get_subsize(bsize, partition);

  assert(bsize >= BLOCK_8X8);

  if (mi_row >= cm->mi_rows || mi_col >= cm->mi_cols) return;

  switch (partition) {
    case PARTITION_NONE:
      set_mode_info_offsets(cm, x, xd, mi_row, mi_col);
      *(xd->mi[0]) = pc_tree->none.mic;
      *(x->mbmi_ext) = pc_tree->none.mbmi_ext;
      duplicate_mode_info_in_sb(cm, xd, mi_row, mi_col, bsize);
      break;
    case PARTITION_VERT:
      set_mode_info_offsets(cm, x, xd, mi_row, mi_col);
      *(xd->mi[0]) = pc_tree->vertical[0].mic;
      *(x->mbmi_ext) = pc_tree->vertical[0].mbmi_ext;
      duplicate_mode_info_in_sb(cm, xd, mi_row, mi_col, subsize);

      if (mi_col + hbs < cm->mi_cols) {
        set_mode_info_offsets(cm, x, xd, mi_row, mi_col + hbs);
        *(xd->mi[0]) = pc_tree->vertical[1].mic;
        *(x->mbmi_ext) = pc_tree->vertical[1].mbmi_ext;
        duplicate_mode_info_in_sb(cm, xd, mi_row, mi_col + hbs, subsize);
      }
      break;
    case PARTITION_HORZ:
      set_mode_info_offsets(cm, x, xd, mi_row, mi_col);
      *(xd->mi[0]) = pc_tree->horizontal[0].mic;
      *(x->mbmi_ext) = pc_tree->horizontal[0].mbmi_ext;
      duplicate_mode_info_in_sb(cm, xd, mi_row, mi_col, subsize);
      if (mi_row + hbs < cm->mi_rows) {
        set_mode_info_offsets(cm, x, xd, mi_row + hbs, mi_col);
        *(xd->mi[0]) = pc_tree->horizontal[1].mic;
        *(x->mbmi_ext) = pc_tree->horizontal[1].mbmi_ext;
        duplicate_mode_info_in_sb(cm, xd, mi_row + hbs, mi_col, subsize);
      }
      break;
    case PARTITION_SPLIT: {
      fill_mode_info_sb(cm, x, mi_row, mi_col, subsize, pc_tree->u.split[0]);
      fill_mode_info_sb(cm, x, mi_row, mi_col + hbs, subsize,
                        pc_tree->u.split[1]);
      fill_mode_info_sb(cm, x, mi_row + hbs, mi_col, subsize,
                        pc_tree->u.split[2]);
      fill_mode_info_sb(cm, x, mi_row + hbs, mi_col + hbs, subsize,
                        pc_tree->u.split[3]);
      break;
    }
    default: break;
  }
}

// Reset the prediction pixel ready flag recursively.
static void pred_pixel_ready_reset(PC_TREE *pc_tree, BLOCK_SIZE bsize) {
  pc_tree->none.pred_pixel_ready = 0;
  pc_tree->horizontal[0].pred_pixel_ready = 0;
  pc_tree->horizontal[1].pred_pixel_ready = 0;
  pc_tree->vertical[0].pred_pixel_ready = 0;
  pc_tree->vertical[1].pred_pixel_ready = 0;

  if (bsize > BLOCK_8X8) {
    BLOCK_SIZE subsize = get_subsize(bsize, PARTITION_SPLIT);
    int i;
    for (i = 0; i < 4; ++i)
      pred_pixel_ready_reset(pc_tree->u.split[i], subsize);
  }
}

#define FEATURES 6
#define LABELS 2
static int ml_predict_var_partitioning(VP9_COMP *cpi, MACROBLOCK *x,
                                       BLOCK_SIZE bsize, int mi_row,
                                       int mi_col) {
  VP9_COMMON *const cm = &cpi->common;
  const NN_CONFIG *nn_config = NULL;

  switch (bsize) {
    case BLOCK_64X64: nn_config = &vp9_var_part_nnconfig_64; break;
    case BLOCK_32X32: nn_config = &vp9_var_part_nnconfig_32; break;
    case BLOCK_16X16: nn_config = &vp9_var_part_nnconfig_16; break;
    case BLOCK_8X8: break;
    default: assert(0 && "Unexpected block size."); return -1;
  }

  if (!nn_config) return -1;

  vpx_clear_system_state();

  {
    const float thresh = cpi->oxcf.speed <= 5 ? 1.25f : 0.0f;
    float features[FEATURES] = { 0.0f };
    const int dc_q = vp9_dc_quant(cm->base_qindex, 0, cm->bit_depth);
    int feature_idx = 0;
    float score[LABELS];

    features[feature_idx++] = logf((float)(dc_q * dc_q) / 256.0f + 1.0f);
    vp9_setup_src_planes(x, cpi->Source, mi_row, mi_col);
    {
      const int bs = 4 * num_4x4_blocks_wide_lookup[bsize];
      const BLOCK_SIZE subsize = get_subsize(bsize, PARTITION_SPLIT);
      const int sb_offset_row = 8 * (mi_row & 7);
      const int sb_offset_col = 8 * (mi_col & 7);
      const uint8_t *pred = x->est_pred + sb_offset_row * 64 + sb_offset_col;
      const uint8_t *src = x->plane[0].src.buf;
      const int src_stride = x->plane[0].src.stride;
      const int pred_stride = 64;
      unsigned int sse;
      int i;
      // Variance of whole block.
      const unsigned int var =
          cpi->fn_ptr[bsize].vf(src, src_stride, pred, pred_stride, &sse);
      const float factor = (var == 0) ? 1.0f : (1.0f / (float)var);

      features[feature_idx++] = logf((float)var + 1.0f);
      for (i = 0; i < 4; ++i) {
        const int x_idx = (i & 1) * bs / 2;
        const int y_idx = (i >> 1) * bs / 2;
        const int src_offset = y_idx * src_stride + x_idx;
        const int pred_offset = y_idx * pred_stride + x_idx;
        // Variance of quarter block.
        const unsigned int sub_var =
            cpi->fn_ptr[subsize].vf(src + src_offset, src_stride,
                                    pred + pred_offset, pred_stride, &sse);
        const float var_ratio = (var == 0) ? 1.0f : factor * (float)sub_var;
        features[feature_idx++] = var_ratio;
      }
    }

    assert(feature_idx == FEATURES);
    nn_predict(features, nn_config, score);
    if (score[0] > thresh) return PARTITION_SPLIT;
    if (score[0] < -thresh) return PARTITION_NONE;
    return -1;
  }
}
#undef FEATURES
#undef LABELS

static void nonrd_pick_partition(VP9_COMP *cpi, ThreadData *td,
                                 TileDataEnc *tile_data, TOKENEXTRA **tp,
                                 int mi_row, int mi_col, BLOCK_SIZE bsize,
                                 RD_COST *rd_cost, int do_recon,
                                 int64_t best_rd, PC_TREE *pc_tree) {
  const SPEED_FEATURES *const sf = &cpi->sf;
  VP9_COMMON *const cm = &cpi->common;
  TileInfo *const tile_info = &tile_data->tile_info;
  MACROBLOCK *const x = &td->mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  const int ms = num_8x8_blocks_wide_lookup[bsize] / 2;
  TOKENEXTRA *tp_orig = *tp;
  PICK_MODE_CONTEXT *ctx = &pc_tree->none;
  int i;
  BLOCK_SIZE subsize = bsize;
  RD_COST this_rdc, sum_rdc, best_rdc;
  int do_split = bsize >= BLOCK_8X8;
  int do_rect = 1;
  // Override skipping rectangular partition operations for edge blocks
  const int force_horz_split = (mi_row + ms >= cm->mi_rows);
  const int force_vert_split = (mi_col + ms >= cm->mi_cols);
  const int xss = x->e_mbd.plane[1].subsampling_x;
  const int yss = x->e_mbd.plane[1].subsampling_y;

  int partition_none_allowed = !force_horz_split && !force_vert_split;
  int partition_horz_allowed =
      !force_vert_split && yss <= xss && bsize >= BLOCK_8X8;
  int partition_vert_allowed =
      !force_horz_split && xss <= yss && bsize >= BLOCK_8X8;
  const int use_ml_based_partitioning =
      sf->partition_search_type == ML_BASED_PARTITION;

  (void)*tp_orig;

  // Avoid checking for rectangular partitions for speed >= 5.
  if (cpi->oxcf.speed >= 5) do_rect = 0;

  assert(num_8x8_blocks_wide_lookup[bsize] ==
         num_8x8_blocks_high_lookup[bsize]);

  vp9_rd_cost_init(&sum_rdc);
  vp9_rd_cost_reset(&best_rdc);
  best_rdc.rdcost = best_rd;

  // Determine partition types in search according to the speed features.
  // The threshold set here has to be of square block size.
  if (sf->auto_min_max_partition_size) {
    partition_none_allowed &=
        (bsize <= x->max_partition_size && bsize >= x->min_partition_size);
    partition_horz_allowed &=
        ((bsize <= x->max_partition_size && bsize > x->min_partition_size) ||
         force_horz_split);
    partition_vert_allowed &=
        ((bsize <= x->max_partition_size && bsize > x->min_partition_size) ||
         force_vert_split);
    do_split &= bsize > x->min_partition_size;
  }
  if (sf->use_square_partition_only) {
    partition_horz_allowed &= force_horz_split;
    partition_vert_allowed &= force_vert_split;
  }

  if (use_ml_based_partitioning) {
    if (partition_none_allowed || do_split) do_rect = 0;
    if (partition_none_allowed && do_split) {
      const int ml_predicted_partition =
          ml_predict_var_partitioning(cpi, x, bsize, mi_row, mi_col);
      if (ml_predicted_partition == PARTITION_NONE) do_split = 0;
      if (ml_predicted_partition == PARTITION_SPLIT) partition_none_allowed = 0;
    }
  }

  if (!partition_none_allowed && !do_split) do_rect = 1;

  ctx->pred_pixel_ready =
      !(partition_vert_allowed || partition_horz_allowed || do_split);

  // PARTITION_NONE
  if (partition_none_allowed) {
    nonrd_pick_sb_modes(cpi, tile_data, x, mi_row, mi_col, &this_rdc, bsize,
                        ctx);
    ctx->mic = *xd->mi[0];
    ctx->mbmi_ext = *x->mbmi_ext;
    ctx->skip_txfm[0] = x->skip_txfm[0];
    ctx->skip = x->skip;

    if (this_rdc.rate != INT_MAX) {
      const int pl = partition_plane_context(xd, mi_row, mi_col, bsize);
      this_rdc.rate += cpi->partition_cost[pl][PARTITION_NONE];
      this_rdc.rdcost =
          RDCOST(x->rdmult, x->rddiv, this_rdc.rate, this_rdc.dist);
      if (this_rdc.rdcost < best_rdc.rdcost) {
        best_rdc = this_rdc;
        if (bsize >= BLOCK_8X8) pc_tree->partitioning = PARTITION_NONE;

        if (!use_ml_based_partitioning) {
          int64_t dist_breakout_thr = sf->partition_search_breakout_thr.dist;
          int64_t rate_breakout_thr = sf->partition_search_breakout_thr.rate;
          dist_breakout_thr >>=
              8 - (b_width_log2_lookup[bsize] + b_height_log2_lookup[bsize]);
          rate_breakout_thr *= num_pels_log2_lookup[bsize];
          if (!x->e_mbd.lossless && this_rdc.rate < rate_breakout_thr &&
              this_rdc.dist < dist_breakout_thr) {
            do_split = 0;
            do_rect = 0;
          }
        }
      }
    }
  }

  // store estimated motion vector
  store_pred_mv(x, ctx);

  // PARTITION_SPLIT
  if (do_split) {
    int pl = partition_plane_context(xd, mi_row, mi_col, bsize);
    sum_rdc.rate += cpi->partition_cost[pl][PARTITION_SPLIT];
    sum_rdc.rdcost = RDCOST(x->rdmult, x->rddiv, sum_rdc.rate, sum_rdc.dist);
    subsize = get_subsize(bsize, PARTITION_SPLIT);
    for (i = 0; i < 4 && sum_rdc.rdcost < best_rdc.rdcost; ++i) {
      const int x_idx = (i & 1) * ms;
      const int y_idx = (i >> 1) * ms;

      if (mi_row + y_idx >= cm->mi_rows || mi_col + x_idx >= cm->mi_cols)
        continue;
      load_pred_mv(x, ctx);
      nonrd_pick_partition(
          cpi, td, tile_data, tp, mi_row + y_idx, mi_col + x_idx, subsize,
          &this_rdc, 0, best_rdc.rdcost - sum_rdc.rdcost, pc_tree->u.split[i]);

      if (this_rdc.rate == INT_MAX) {
        vp9_rd_cost_reset(&sum_rdc);
      } else {
        sum_rdc.rate += this_rdc.rate;
        sum_rdc.dist += this_rdc.dist;
        sum_rdc.rdcost += this_rdc.rdcost;
      }
    }

    if (sum_rdc.rdcost < best_rdc.rdcost) {
      best_rdc = sum_rdc;
      pc_tree->partitioning = PARTITION_SPLIT;
    } else {
      // skip rectangular partition test when larger block size
      // gives better rd cost
      if (sf->less_rectangular_check) do_rect &= !partition_none_allowed;
    }
  }

  // PARTITION_HORZ
  if (partition_horz_allowed && do_rect) {
    subsize = get_subsize(bsize, PARTITION_HORZ);
    load_pred_mv(x, ctx);
    pc_tree->horizontal[0].pred_pixel_ready = 1;
    nonrd_pick_sb_modes(cpi, tile_data, x, mi_row, mi_col, &sum_rdc, subsize,
                        &pc_tree->horizontal[0]);

    pc_tree->horizontal[0].mic = *xd->mi[0];
    pc_tree->horizontal[0].mbmi_ext = *x->mbmi_ext;
    pc_tree->horizontal[0].skip_txfm[0] = x->skip_txfm[0];
    pc_tree->horizontal[0].skip = x->skip;

    if (sum_rdc.rdcost < best_rdc.rdcost && mi_row + ms < cm->mi_rows) {
      load_pred_mv(x, ctx);
      pc_tree->horizontal[1].pred_pixel_ready = 1;
      nonrd_pick_sb_modes(cpi, tile_data, x, mi_row + ms, mi_col, &this_rdc,
                          subsize, &pc_tree->horizontal[1]);

      pc_tree->horizontal[1].mic = *xd->mi[0];
      pc_tree->horizontal[1].mbmi_ext = *x->mbmi_ext;
      pc_tree->horizontal[1].skip_txfm[0] = x->skip_txfm[0];
      pc_tree->horizontal[1].skip = x->skip;

      if (this_rdc.rate == INT_MAX) {
        vp9_rd_cost_reset(&sum_rdc);
      } else {
        int pl = partition_plane_context(xd, mi_row, mi_col, bsize);
        this_rdc.rate += cpi->partition_cost[pl][PARTITION_HORZ];
        sum_rdc.rate += this_rdc.rate;
        sum_rdc.dist += this_rdc.dist;
        sum_rdc.rdcost =
            RDCOST(x->rdmult, x->rddiv, sum_rdc.rate, sum_rdc.dist);
      }
    }

    if (sum_rdc.rdcost < best_rdc.rdcost) {
      best_rdc = sum_rdc;
      pc_tree->partitioning = PARTITION_HORZ;
    } else {
      pred_pixel_ready_reset(pc_tree, bsize);
    }
  }

  // PARTITION_VERT
  if (partition_vert_allowed && do_rect) {
    subsize = get_subsize(bsize, PARTITION_VERT);
    load_pred_mv(x, ctx);
    pc_tree->vertical[0].pred_pixel_ready = 1;
    nonrd_pick_sb_modes(cpi, tile_data, x, mi_row, mi_col, &sum_rdc, subsize,
                        &pc_tree->vertical[0]);
    pc_tree->vertical[0].mic = *xd->mi[0];
    pc_tree->vertical[0].mbmi_ext = *x->mbmi_ext;
    pc_tree->vertical[0].skip_txfm[0] = x->skip_txfm[0];
    pc_tree->vertical[0].skip = x->skip;

    if (sum_rdc.rdcost < best_rdc.rdcost && mi_col + ms < cm->mi_cols) {
      load_pred_mv(x, ctx);
      pc_tree->vertical[1].pred_pixel_ready = 1;
      nonrd_pick_sb_modes(cpi, tile_data, x, mi_row, mi_col + ms, &this_rdc,
                          subsize, &pc_tree->vertical[1]);
      pc_tree->vertical[1].mic = *xd->mi[0];
      pc_tree->vertical[1].mbmi_ext = *x->mbmi_ext;
      pc_tree->vertical[1].skip_txfm[0] = x->skip_txfm[0];
      pc_tree->vertical[1].skip = x->skip;

      if (this_rdc.rate == INT_MAX) {
        vp9_rd_cost_reset(&sum_rdc);
      } else {
        int pl = partition_plane_context(xd, mi_row, mi_col, bsize);
        sum_rdc.rate += cpi->partition_cost[pl][PARTITION_VERT];
        sum_rdc.rate += this_rdc.rate;
        sum_rdc.dist += this_rdc.dist;
        sum_rdc.rdcost =
            RDCOST(x->rdmult, x->rddiv, sum_rdc.rate, sum_rdc.dist);
      }
    }

    if (sum_rdc.rdcost < best_rdc.rdcost) {
      best_rdc = sum_rdc;
      pc_tree->partitioning = PARTITION_VERT;
    } else {
      pred_pixel_ready_reset(pc_tree, bsize);
    }
  }

  *rd_cost = best_rdc;

  if (best_rdc.rate == INT_MAX) {
    vp9_rd_cost_reset(rd_cost);
    return;
  }

  // update mode info array
  fill_mode_info_sb(cm, x, mi_row, mi_col, bsize, pc_tree);

  if (best_rdc.rate < INT_MAX && best_rdc.dist < INT64_MAX && do_recon) {
    int output_enabled = (bsize == BLOCK_64X64);
    encode_sb_rt(cpi, td, tile_info, tp, mi_row, mi_col, output_enabled, bsize,
                 pc_tree);
  }

  if (bsize == BLOCK_64X64 && do_recon) {
    assert(tp_orig < *tp);
    assert(best_rdc.rate < INT_MAX);
    assert(best_rdc.dist < INT64_MAX);
  } else {
    assert(tp_orig == *tp);
  }
}

static void nonrd_select_partition(VP9_COMP *cpi, ThreadData *td,
                                   TileDataEnc *tile_data, MODE_INFO **mi,
                                   TOKENEXTRA **tp, int mi_row, int mi_col,
                                   BLOCK_SIZE bsize, int output_enabled,
                                   RD_COST *rd_cost, PC_TREE *pc_tree) {
  VP9_COMMON *const cm = &cpi->common;
  TileInfo *const tile_info = &tile_data->tile_info;
  MACROBLOCK *const x = &td->mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  const int bsl = b_width_log2_lookup[bsize], hbs = (1 << bsl) / 4;
  const int mis = cm->mi_stride;
  PARTITION_TYPE partition;
  BLOCK_SIZE subsize;
  RD_COST this_rdc;
  BLOCK_SIZE subsize_ref =
      (cpi->sf.adapt_partition_source_sad) ? BLOCK_8X8 : BLOCK_16X16;

  vp9_rd_cost_reset(&this_rdc);
  if (mi_row >= cm->mi_rows || mi_col >= cm->mi_cols) return;

  subsize = (bsize >= BLOCK_8X8) ? mi[0]->sb_type : BLOCK_4X4;
  partition = partition_lookup[bsl][subsize];

  if (bsize == BLOCK_32X32 && subsize == BLOCK_32X32) {
    x->max_partition_size = BLOCK_32X32;
    x->min_partition_size = BLOCK_16X16;
    nonrd_pick_partition(cpi, td, tile_data, tp, mi_row, mi_col, bsize, rd_cost,
                         0, INT64_MAX, pc_tree);
  } else if (bsize == BLOCK_32X32 && partition != PARTITION_NONE &&
             subsize >= subsize_ref) {
    x->max_partition_size = BLOCK_32X32;
    x->min_partition_size = BLOCK_8X8;
    nonrd_pick_partition(cpi, td, tile_data, tp, mi_row, mi_col, bsize, rd_cost,
                         0, INT64_MAX, pc_tree);
  } else if (bsize == BLOCK_16X16 && partition != PARTITION_NONE) {
    x->max_partition_size = BLOCK_16X16;
    x->min_partition_size = BLOCK_8X8;
    nonrd_pick_partition(cpi, td, tile_data, tp, mi_row, mi_col, bsize, rd_cost,
                         0, INT64_MAX, pc_tree);
  } else {
    switch (partition) {
      case PARTITION_NONE:
        pc_tree->none.pred_pixel_ready = 1;
        nonrd_pick_sb_modes(cpi, tile_data, x, mi_row, mi_col, rd_cost, subsize,
                            &pc_tree->none);
        pc_tree->none.mic = *xd->mi[0];
        pc_tree->none.mbmi_ext = *x->mbmi_ext;
        pc_tree->none.skip_txfm[0] = x->skip_txfm[0];
        pc_tree->none.skip = x->skip;
        break;
      case PARTITION_VERT:
        pc_tree->vertical[0].pred_pixel_ready = 1;
        nonrd_pick_sb_modes(cpi, tile_data, x, mi_row, mi_col, rd_cost, subsize,
                            &pc_tree->vertical[0]);
        pc_tree->vertical[0].mic = *xd->mi[0];
        pc_tree->vertical[0].mbmi_ext = *x->mbmi_ext;
        pc_tree->vertical[0].skip_txfm[0] = x->skip_txfm[0];
        pc_tree->vertical[0].skip = x->skip;
        if (mi_col + hbs < cm->mi_cols) {
          pc_tree->vertical[1].pred_pixel_ready = 1;
          nonrd_pick_sb_modes(cpi, tile_data, x, mi_row, mi_col + hbs,
                              &this_rdc, subsize, &pc_tree->vertical[1]);
          pc_tree->vertical[1].mic = *xd->mi[0];
          pc_tree->vertical[1].mbmi_ext = *x->mbmi_ext;
          pc_tree->vertical[1].skip_txfm[0] = x->skip_txfm[0];
          pc_tree->vertical[1].skip = x->skip;
          if (this_rdc.rate != INT_MAX && this_rdc.dist != INT64_MAX &&
              rd_cost->rate != INT_MAX && rd_cost->dist != INT64_MAX) {
            rd_cost->rate += this_rdc.rate;
            rd_cost->dist += this_rdc.dist;
          }
        }
        break;
      case PARTITION_HORZ:
        pc_tree->horizontal[0].pred_pixel_ready = 1;
        nonrd_pick_sb_modes(cpi, tile_data, x, mi_row, mi_col, rd_cost, subsize,
                            &pc_tree->horizontal[0]);
        pc_tree->horizontal[0].mic = *xd->mi[0];
        pc_tree->horizontal[0].mbmi_ext = *x->mbmi_ext;
        pc_tree->horizontal[0].skip_txfm[0] = x->skip_txfm[0];
        pc_tree->horizontal[0].skip = x->skip;
        if (mi_row + hbs < cm->mi_rows) {
          pc_tree->horizontal[1].pred_pixel_ready = 1;
          nonrd_pick_sb_modes(cpi, tile_data, x, mi_row + hbs, mi_col,
                              &this_rdc, subsize, &pc_tree->horizontal[1]);
          pc_tree->horizontal[1].mic = *xd->mi[0];
          pc_tree->horizontal[1].mbmi_ext = *x->mbmi_ext;
          pc_tree->horizontal[1].skip_txfm[0] = x->skip_txfm[0];
          pc_tree->horizontal[1].skip = x->skip;
          if (this_rdc.rate != INT_MAX && this_rdc.dist != INT64_MAX &&
              rd_cost->rate != INT_MAX && rd_cost->dist != INT64_MAX) {
            rd_cost->rate += this_rdc.rate;
            rd_cost->dist += this_rdc.dist;
          }
        }
        break;
      default:
        assert(partition == PARTITION_SPLIT);
        subsize = get_subsize(bsize, PARTITION_SPLIT);
        nonrd_select_partition(cpi, td, tile_data, mi, tp, mi_row, mi_col,
                               subsize, output_enabled, rd_cost,
                               pc_tree->u.split[0]);
        nonrd_select_partition(cpi, td, tile_data, mi + hbs, tp, mi_row,
                               mi_col + hbs, subsize, output_enabled, &this_rdc,
                               pc_tree->u.split[1]);
        if (this_rdc.rate != INT_MAX && this_rdc.dist != INT64_MAX &&
            rd_cost->rate != INT_MAX && rd_cost->dist != INT64_MAX) {
          rd_cost->rate += this_rdc.rate;
          rd_cost->dist += this_rdc.dist;
        }
        nonrd_select_partition(cpi, td, tile_data, mi + hbs * mis, tp,
                               mi_row + hbs, mi_col, subsize, output_enabled,
                               &this_rdc, pc_tree->u.split[2]);
        if (this_rdc.rate != INT_MAX && this_rdc.dist != INT64_MAX &&
            rd_cost->rate != INT_MAX && rd_cost->dist != INT64_MAX) {
          rd_cost->rate += this_rdc.rate;
          rd_cost->dist += this_rdc.dist;
        }
        nonrd_select_partition(cpi, td, tile_data, mi + hbs * mis + hbs, tp,
                               mi_row + hbs, mi_col + hbs, subsize,
                               output_enabled, &this_rdc, pc_tree->u.split[3]);
        if (this_rdc.rate != INT_MAX && this_rdc.dist != INT64_MAX &&
            rd_cost->rate != INT_MAX && rd_cost->dist != INT64_MAX) {
          rd_cost->rate += this_rdc.rate;
          rd_cost->dist += this_rdc.dist;
        }
        break;
    }
  }

  if (bsize == BLOCK_64X64 && output_enabled)
    encode_sb_rt(cpi, td, tile_info, tp, mi_row, mi_col, 1, bsize, pc_tree);
}

static void nonrd_use_partition(VP9_COMP *cpi, ThreadData *td,
                                TileDataEnc *tile_data, MODE_INFO **mi,
                                TOKENEXTRA **tp, int mi_row, int mi_col,
                                BLOCK_SIZE bsize, int output_enabled,
                                RD_COST *dummy_cost, PC_TREE *pc_tree) {
  VP9_COMMON *const cm = &cpi->common;
  TileInfo *tile_info = &tile_data->tile_info;
  MACROBLOCK *const x = &td->mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  const int bsl = b_width_log2_lookup[bsize], hbs = (1 << bsl) / 4;
  const int mis = cm->mi_stride;
  PARTITION_TYPE partition;
  BLOCK_SIZE subsize;

  if (mi_row >= cm->mi_rows || mi_col >= cm->mi_cols) return;

  subsize = (bsize >= BLOCK_8X8) ? mi[0]->sb_type : BLOCK_4X4;
  partition = partition_lookup[bsl][subsize];

  if (output_enabled && bsize != BLOCK_4X4) {
    int ctx = partition_plane_context(xd, mi_row, mi_col, bsize);
    td->counts->partition[ctx][partition]++;
  }

  switch (partition) {
    case PARTITION_NONE:
      pc_tree->none.pred_pixel_ready = 1;
      nonrd_pick_sb_modes(cpi, tile_data, x, mi_row, mi_col, dummy_cost,
                          subsize, &pc_tree->none);
      pc_tree->none.mic = *xd->mi[0];
      pc_tree->none.mbmi_ext = *x->mbmi_ext;
      pc_tree->none.skip_txfm[0] = x->skip_txfm[0];
      pc_tree->none.skip = x->skip;
      encode_b_rt(cpi, td, tile_info, tp, mi_row, mi_col, output_enabled,
                  subsize, &pc_tree->none);
      break;
    case PARTITION_VERT:
      pc_tree->vertical[0].pred_pixel_ready = 1;
      nonrd_pick_sb_modes(cpi, tile_data, x, mi_row, mi_col, dummy_cost,
                          subsize, &pc_tree->vertical[0]);
      pc_tree->vertical[0].mic = *xd->mi[0];
      pc_tree->vertical[0].mbmi_ext = *x->mbmi_ext;
      pc_tree->vertical[0].skip_txfm[0] = x->skip_txfm[0];
      pc_tree->vertical[0].skip = x->skip;
      encode_b_rt(cpi, td, tile_info, tp, mi_row, mi_col, output_enabled,
                  subsize, &pc_tree->vertical[0]);
      if (mi_col + hbs < cm->mi_cols && bsize > BLOCK_8X8) {
        pc_tree->vertical[1].pred_pixel_ready = 1;
        nonrd_pick_sb_modes(cpi, tile_data, x, mi_row, mi_col + hbs, dummy_cost,
                            subsize, &pc_tree->vertical[1]);
        pc_tree->vertical[1].mic = *xd->mi[0];
        pc_tree->vertical[1].mbmi_ext = *x->mbmi_ext;
        pc_tree->vertical[1].skip_txfm[0] = x->skip_txfm[0];
        pc_tree->vertical[1].skip = x->skip;
        encode_b_rt(cpi, td, tile_info, tp, mi_row, mi_col + hbs,
                    output_enabled, subsize, &pc_tree->vertical[1]);
      }
      break;
    case PARTITION_HORZ:
      pc_tree->horizontal[0].pred_pixel_ready = 1;
      nonrd_pick_sb_modes(cpi, tile_data, x, mi_row, mi_col, dummy_cost,
                          subsize, &pc_tree->horizontal[0]);
      pc_tree->horizontal[0].mic = *xd->mi[0];
      pc_tree->horizontal[0].mbmi_ext = *x->mbmi_ext;
      pc_tree->horizontal[0].skip_txfm[0] = x->skip_txfm[0];
      pc_tree->horizontal[0].skip = x->skip;
      encode_b_rt(cpi, td, tile_info, tp, mi_row, mi_col, output_enabled,
                  subsize, &pc_tree->horizontal[0]);

      if (mi_row + hbs < cm->mi_rows && bsize > BLOCK_8X8) {
        pc_tree->horizontal[1].pred_pixel_ready = 1;
        nonrd_pick_sb_modes(cpi, tile_data, x, mi_row + hbs, mi_col, dummy_cost,
                            subsize, &pc_tree->horizontal[1]);
        pc_tree->horizontal[1].mic = *xd->mi[0];
        pc_tree->horizontal[1].mbmi_ext = *x->mbmi_ext;
        pc_tree->horizontal[1].skip_txfm[0] = x->skip_txfm[0];
        pc_tree->horizontal[1].skip = x->skip;
        encode_b_rt(cpi, td, tile_info, tp, mi_row + hbs, mi_col,
                    output_enabled, subsize, &pc_tree->horizontal[1]);
      }
      break;
    default:
      assert(partition == PARTITION_SPLIT);
      subsize = get_subsize(bsize, PARTITION_SPLIT);
      if (bsize == BLOCK_8X8) {
        nonrd_pick_sb_modes(cpi, tile_data, x, mi_row, mi_col, dummy_cost,
                            subsize, pc_tree->u.leaf_split[0]);
        encode_b_rt(cpi, td, tile_info, tp, mi_row, mi_col, output_enabled,
                    subsize, pc_tree->u.leaf_split[0]);
      } else {
        nonrd_use_partition(cpi, td, tile_data, mi, tp, mi_row, mi_col, subsize,
                            output_enabled, dummy_cost, pc_tree->u.split[0]);
        nonrd_use_partition(cpi, td, tile_data, mi + hbs, tp, mi_row,
                            mi_col + hbs, subsize, output_enabled, dummy_cost,
                            pc_tree->u.split[1]);
        nonrd_use_partition(cpi, td, tile_data, mi + hbs * mis, tp,
                            mi_row + hbs, mi_col, subsize, output_enabled,
                            dummy_cost, pc_tree->u.split[2]);
        nonrd_use_partition(cpi, td, tile_data, mi + hbs * mis + hbs, tp,
                            mi_row + hbs, mi_col + hbs, subsize, output_enabled,
                            dummy_cost, pc_tree->u.split[3]);
      }
      break;
  }

  if (partition != PARTITION_SPLIT || bsize == BLOCK_8X8)
    update_partition_context(xd, mi_row, mi_col, subsize, bsize);
}

// Get a prediction(stored in x->est_pred) for the whole 64x64 superblock.
static void get_estimated_pred(VP9_COMP *cpi, const TileInfo *const tile,
                               MACROBLOCK *x, int mi_row, int mi_col) {
  VP9_COMMON *const cm = &cpi->common;
  const int is_key_frame = frame_is_intra_only(cm);
  MACROBLOCKD *xd = &x->e_mbd;

  set_offsets(cpi, tile, x, mi_row, mi_col, BLOCK_64X64);

  if (!is_key_frame) {
    MODE_INFO *mi = xd->mi[0];
    YV12_BUFFER_CONFIG *yv12 = get_ref_frame_buffer(cpi, LAST_FRAME);
    const YV12_BUFFER_CONFIG *yv12_g = NULL;
    const BLOCK_SIZE bsize = BLOCK_32X32 + (mi_col + 4 < cm->mi_cols) * 2 +
                             (mi_row + 4 < cm->mi_rows);
    unsigned int y_sad_g, y_sad_thr;
    unsigned int y_sad = UINT_MAX;

    assert(yv12 != NULL);

    if (!(is_one_pass_svc(cpi) && cpi->svc.spatial_layer_id) ||
        cpi->svc.use_gf_temporal_ref_current_layer) {
      // For now, GOLDEN will not be used for non-zero spatial layers, since
      // it may not be a temporal reference.
      yv12_g = get_ref_frame_buffer(cpi, GOLDEN_FRAME);
    }

    // Only compute y_sad_g (sad for golden reference) for speed < 8.
    if (cpi->oxcf.speed < 8 && yv12_g && yv12_g != yv12 &&
        (cpi->ref_frame_flags & VP9_GOLD_FLAG)) {
      vp9_setup_pre_planes(xd, 0, yv12_g, mi_row, mi_col,
                           &cm->frame_refs[GOLDEN_FRAME - 1].sf);
      y_sad_g = cpi->fn_ptr[bsize].sdf(
          x->plane[0].src.buf, x->plane[0].src.stride, xd->plane[0].pre[0].buf,
          xd->plane[0].pre[0].stride);
    } else {
      y_sad_g = UINT_MAX;
    }

    if (cpi->oxcf.lag_in_frames > 0 && cpi->oxcf.rc_mode == VPX_VBR &&
        cpi->rc.is_src_frame_alt_ref) {
      yv12 = get_ref_frame_buffer(cpi, ALTREF_FRAME);
      vp9_setup_pre_planes(xd, 0, yv12, mi_row, mi_col,
                           &cm->frame_refs[ALTREF_FRAME - 1].sf);
      mi->ref_frame[0] = ALTREF_FRAME;
      y_sad_g = UINT_MAX;
    } else {
      vp9_setup_pre_planes(xd, 0, yv12, mi_row, mi_col,
                           &cm->frame_refs[LAST_FRAME - 1].sf);
      mi->ref_frame[0] = LAST_FRAME;
    }
    mi->ref_frame[1] = NO_REF_FRAME;
    mi->sb_type = BLOCK_64X64;
    mi->mv[0].as_int = 0;
    mi->interp_filter = BILINEAR;

    {
      const MV dummy_mv = { 0, 0 };
      y_sad = vp9_int_pro_motion_estimation(cpi, x, bsize, mi_row, mi_col,
                                            &dummy_mv);
      x->sb_use_mv_part = 1;
      x->sb_mvcol_part = mi->mv[0].as_mv.col;
      x->sb_mvrow_part = mi->mv[0].as_mv.row;
    }

    // Pick ref frame for partitioning, bias last frame when y_sad_g and y_sad
    // are close if short_circuit_low_temp_var is on.
    y_sad_thr = cpi->sf.short_circuit_low_temp_var ? (y_sad * 7) >> 3 : y_sad;
    if (y_sad_g < y_sad_thr) {
      vp9_setup_pre_planes(xd, 0, yv12_g, mi_row, mi_col,
                           &cm->frame_refs[GOLDEN_FRAME - 1].sf);
      mi->ref_frame[0] = GOLDEN_FRAME;
      mi->mv[0].as_int = 0;
    } else {
      x->pred_mv[LAST_FRAME] = mi->mv[0].as_mv;
    }

    set_ref_ptrs(cm, xd, mi->ref_frame[0], mi->ref_frame[1]);
    xd->plane[0].dst.buf = x->est_pred;
    xd->plane[0].dst.stride = 64;
    vp9_build_inter_predictors_sb(xd, mi_row, mi_col, BLOCK_64X64);
  } else {
#if CONFIG_VP9_HIGHBITDEPTH
    switch (xd->bd) {
      case 8: memset(x->est_pred, 128, 64 * 64 * sizeof(x->est_pred[0])); break;
      case 10:
        memset(x->est_pred, 128 * 4, 64 * 64 * sizeof(x->est_pred[0]));
        break;
      case 12:
        memset(x->est_pred, 128 * 16, 64 * 64 * sizeof(x->est_pred[0]));
        break;
    }
#else
    memset(x->est_pred, 128, 64 * 64 * sizeof(x->est_pred[0]));
#endif  // CONFIG_VP9_HIGHBITDEPTH
  }
}

static void encode_nonrd_sb_row(VP9_COMP *cpi, ThreadData *td,
                                TileDataEnc *tile_data, int mi_row,
                                TOKENEXTRA **tp) {
  SPEED_FEATURES *const sf = &cpi->sf;
  VP9_COMMON *const cm = &cpi->common;
  TileInfo *const tile_info = &tile_data->tile_info;
  MACROBLOCK *const x = &td->mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  const int mi_col_start = tile_info->mi_col_start;
  const int mi_col_end = tile_info->mi_col_end;
  int mi_col;
  const int sb_row = mi_row >> MI_BLOCK_SIZE_LOG2;
  const int num_sb_cols =
      get_num_cols(tile_data->tile_info, MI_BLOCK_SIZE_LOG2);
  int sb_col_in_tile;

  // Initialize the left context for the new SB row
  memset(&xd->left_context, 0, sizeof(xd->left_context));
  memset(xd->left_seg_context, 0, sizeof(xd->left_seg_context));

  // Code each SB in the row
  for (mi_col = mi_col_start, sb_col_in_tile = 0; mi_col < mi_col_end;
       mi_col += MI_BLOCK_SIZE, ++sb_col_in_tile) {
    const struct segmentation *const seg = &cm->seg;
    RD_COST dummy_rdc;
    const int idx_str = cm->mi_stride * mi_row + mi_col;
    MODE_INFO **mi = cm->mi_grid_visible + idx_str;
    PARTITION_SEARCH_TYPE partition_search_type = sf->partition_search_type;
    BLOCK_SIZE bsize = BLOCK_64X64;
    int seg_skip = 0;
    int i;

    (*(cpi->row_mt_sync_read_ptr))(&tile_data->row_mt_sync, sb_row,
                                   sb_col_in_tile);

    if (cpi->use_skin_detection) {
      vp9_compute_skin_sb(cpi, BLOCK_16X16, mi_row, mi_col);
    }

    x->source_variance = UINT_MAX;
    for (i = 0; i < MAX_REF_FRAMES; ++i) {
      x->pred_mv[i].row = INT16_MAX;
      x->pred_mv[i].col = INT16_MAX;
    }
    vp9_rd_cost_init(&dummy_rdc);
    x->color_sensitivity[0] = 0;
    x->color_sensitivity[1] = 0;
    x->sb_is_skin = 0;
    x->skip_low_source_sad = 0;
    x->lowvar_highsumdiff = 0;
    x->content_state_sb = 0;
    x->zero_temp_sad_source = 0;
    x->sb_use_mv_part = 0;
    x->sb_mvcol_part = 0;
    x->sb_mvrow_part = 0;
    x->sb_pickmode_part = 0;
    x->arf_frame_usage = 0;
    x->lastgolden_frame_usage = 0;

    if (cpi->compute_source_sad_onepass && cpi->sf.use_source_sad) {
      int shift = cpi->Source->y_stride * (mi_row << 3) + (mi_col << 3);
      int sb_offset2 = ((cm->mi_cols + 7) >> 3) * (mi_row >> 3) + (mi_col >> 3);
      int64_t source_sad = avg_source_sad(cpi, x, shift, sb_offset2);
      if (sf->adapt_partition_source_sad &&
          (cpi->oxcf.rc_mode == VPX_VBR && !cpi->rc.is_src_frame_alt_ref &&
           source_sad > sf->adapt_partition_thresh &&
           (cpi->refresh_golden_frame || cpi->refresh_alt_ref_frame)))
        partition_search_type = REFERENCE_PARTITION;
    }

    if (seg->enabled) {
      const uint8_t *const map =
          seg->update_map ? cpi->segmentation_map : cm->last_frame_seg_map;
      int segment_id = get_segment_id(cm, map, BLOCK_64X64, mi_row, mi_col);
      seg_skip = segfeature_active(seg, segment_id, SEG_LVL_SKIP);

      if (cpi->roi.enabled && cpi->roi.skip[BACKGROUND_SEG_SKIP_ID] &&
          cpi->rc.frames_since_key > FRAMES_NO_SKIPPING_AFTER_KEY &&
          x->content_state_sb > kLowSadLowSumdiff) {
        // For ROI with skip, force segment = 0 (no skip) over whole
        // superblock to avoid artifacts if temporal change in source_sad is
        // not 0.
        int xi, yi;
        const int bw = num_8x8_blocks_wide_lookup[BLOCK_64X64];
        const int bh = num_8x8_blocks_high_lookup[BLOCK_64X64];
        const int xmis = VPXMIN(cm->mi_cols - mi_col, bw);
        const int ymis = VPXMIN(cm->mi_rows - mi_row, bh);
        const int block_index = mi_row * cm->mi_cols + mi_col;
        set_mode_info_offsets(cm, x, xd, mi_row, mi_col);
        for (yi = 0; yi < ymis; yi++)
          for (xi = 0; xi < xmis; xi++) {
            int map_offset = block_index + yi * cm->mi_cols + xi;
            cpi->segmentation_map[map_offset] = 0;
          }
        set_segment_index(cpi, x, mi_row, mi_col, BLOCK_64X64, 0);
        seg_skip = 0;
      }
      if (seg_skip) {
        partition_search_type = FIXED_PARTITION;
      }
    }

    // Set the partition type of the 64X64 block
    switch (partition_search_type) {
      case VAR_BASED_PARTITION:
        // TODO(jingning, marpan): The mode decision and encoding process
        // support both intra and inter sub8x8 block coding for RTC mode.
        // Tune the thresholds accordingly to use sub8x8 block coding for
        // coding performance improvement.
        choose_partitioning(cpi, tile_info, x, mi_row, mi_col);
        nonrd_use_partition(cpi, td, tile_data, mi, tp, mi_row, mi_col,
                            BLOCK_64X64, 1, &dummy_rdc, td->pc_root);
        break;
      case ML_BASED_PARTITION:
        get_estimated_pred(cpi, tile_info, x, mi_row, mi_col);
        x->max_partition_size = BLOCK_64X64;
        x->min_partition_size = BLOCK_8X8;
        x->sb_pickmode_part = 1;
        nonrd_pick_partition(cpi, td, tile_data, tp, mi_row, mi_col,
                             BLOCK_64X64, &dummy_rdc, 1, INT64_MAX,
                             td->pc_root);
        break;
      case FIXED_PARTITION:
        if (!seg_skip) bsize = sf->always_this_block_size;
        set_fixed_partitioning(cpi, tile_info, mi, mi_row, mi_col, bsize);
        nonrd_use_partition(cpi, td, tile_data, mi, tp, mi_row, mi_col,
                            BLOCK_64X64, 1, &dummy_rdc, td->pc_root);
        break;
      default:
        assert(partition_search_type == REFERENCE_PARTITION);
        x->sb_pickmode_part = 1;
        set_offsets(cpi, tile_info, x, mi_row, mi_col, BLOCK_64X64);
        // Use nonrd_pick_partition on scene-cut for VBR mode.
        // nonrd_pick_partition does not support 4x4 partition, so avoid it
        // on key frame for now.
        if ((cpi->oxcf.rc_mode == VPX_VBR && cpi->rc.high_source_sad &&
             cpi->oxcf.speed < 6 && !frame_is_intra_only(cm) &&
             (cpi->refresh_golden_frame || cpi->refresh_alt_ref_frame))) {
          // Use lower max_partition_size for low resolutions.
          if (cm->width <= 352 && cm->height <= 288)
            x->max_partition_size = BLOCK_32X32;
          else
            x->max_partition_size = BLOCK_64X64;
          x->min_partition_size = BLOCK_8X8;
          nonrd_pick_partition(cpi, td, tile_data, tp, mi_row, mi_col,
                               BLOCK_64X64, &dummy_rdc, 1, INT64_MAX,
                               td->pc_root);
        } else {
          choose_partitioning(cpi, tile_info, x, mi_row, mi_col);
          // TODO(marpan): Seems like nonrd_select_partition does not support
          // 4x4 partition. Since 4x4 is used on key frame, use this switch
          // for now.
          if (frame_is_intra_only(cm))
            nonrd_use_partition(cpi, td, tile_data, mi, tp, mi_row, mi_col,
                                BLOCK_64X64, 1, &dummy_rdc, td->pc_root);
          else
            nonrd_select_partition(cpi, td, tile_data, mi, tp, mi_row, mi_col,
                                   BLOCK_64X64, 1, &dummy_rdc, td->pc_root);
        }

        break;
    }

    // Update ref_frame usage for inter frame if this group is ARF group.
    if (!cpi->rc.is_src_frame_alt_ref && !cpi->refresh_golden_frame &&
        !cpi->refresh_alt_ref_frame && cpi->rc.alt_ref_gf_group &&
        cpi->sf.use_altref_onepass) {
      int sboffset = ((cm->mi_cols + 7) >> 3) * (mi_row >> 3) + (mi_col >> 3);
      if (cpi->count_arf_frame_usage != NULL)
        cpi->count_arf_frame_usage[sboffset] = x->arf_frame_usage;
      if (cpi->count_lastgolden_frame_usage != NULL)
        cpi->count_lastgolden_frame_usage[sboffset] = x->lastgolden_frame_usage;
    }

    (*(cpi->row_mt_sync_write_ptr))(&tile_data->row_mt_sync, sb_row,
                                    sb_col_in_tile, num_sb_cols);
  }
}
// end RTC play code

static int get_skip_encode_frame(const VP9_COMMON *cm, ThreadData *const td) {
  unsigned int intra_count = 0, inter_count = 0;
  int j;

  for (j = 0; j < INTRA_INTER_CONTEXTS; ++j) {
    intra_count += td->counts->intra_inter[j][0];
    inter_count += td->counts->intra_inter[j][1];
  }

  return (intra_count << 2) < inter_count && cm->frame_type != KEY_FRAME &&
         cm->show_frame;
}

void vp9_init_tile_data(VP9_COMP *cpi) {
  VP9_COMMON *const cm = &cpi->common;
  const int tile_cols = 1 << cm->log2_tile_cols;
  const int tile_rows = 1 << cm->log2_tile_rows;
  int tile_col, tile_row;
  TOKENEXTRA *pre_tok = cpi->tile_tok[0][0];
  TOKENLIST *tplist = cpi->tplist[0][0];
  int tile_tok = 0;
  int tplist_count = 0;

  if (cpi->tile_data == NULL || cpi->allocated_tiles < tile_cols * tile_rows) {
    if (cpi->tile_data != NULL) {
      // Free the row mt memory in cpi->tile_data first.
      vp9_row_mt_mem_dealloc(cpi);
      vpx_free(cpi->tile_data);
    }
    cpi->allocated_tiles = 0;
    CHECK_MEM_ERROR(
        &cm->error, cpi->tile_data,
        vpx_malloc(tile_cols * tile_rows * sizeof(*cpi->tile_data)));
    cpi->allocated_tiles = tile_cols * tile_rows;

    for (tile_row = 0; tile_row < tile_rows; ++tile_row)
      for (tile_col = 0; tile_col < tile_cols; ++tile_col) {
        TileDataEnc *tile_data =
            &cpi->tile_data[tile_row * tile_cols + tile_col];
        int i, j;
        const MV zero_mv = { 0, 0 };
        for (i = 0; i < BLOCK_SIZES; ++i) {
          for (j = 0; j < MAX_MODES; ++j) {
            tile_data->thresh_freq_fact[i][j] = RD_THRESH_INIT_FACT;
            tile_data->thresh_freq_fact_prev[i][j] = RD_THRESH_INIT_FACT;
            tile_data->mode_map[i][j] = j;
          }
        }
        tile_data->firstpass_top_mv = zero_mv;
#if CONFIG_MULTITHREAD
        tile_data->row_base_thresh_freq_fact = NULL;
#endif
      }
  }

  for (tile_row = 0; tile_row < tile_rows; ++tile_row) {
    for (tile_col = 0; tile_col < tile_cols; ++tile_col) {
      TileDataEnc *this_tile = &cpi->tile_data[tile_row * tile_cols + tile_col];
      TileInfo *tile_info = &this_tile->tile_info;
      if (cpi->sf.adaptive_rd_thresh_row_mt) {
        vp9_row_mt_alloc_rd_thresh(cpi, this_tile);
      }
      vp9_tile_init(tile_info, cm, tile_row, tile_col);

      cpi->tile_tok[tile_row][tile_col] = pre_tok + tile_tok;
      pre_tok = cpi->tile_tok[tile_row][tile_col];
      tile_tok = allocated_tokens(*tile_info);

      cpi->tplist[tile_row][tile_col] = tplist + tplist_count;
      tplist = cpi->tplist[tile_row][tile_col];
      tplist_count = get_num_vert_units(*tile_info, MI_BLOCK_SIZE_LOG2);
    }
  }
}

void vp9_encode_sb_row(VP9_COMP *cpi, ThreadData *td, int tile_row,
                       int tile_col, int mi_row) {
  VP9_COMMON *const cm = &cpi->common;
  const int tile_cols = 1 << cm->log2_tile_cols;
  TileDataEnc *this_tile = &cpi->tile_data[tile_row * tile_cols + tile_col];
  const TileInfo *const tile_info = &this_tile->tile_info;
  TOKENEXTRA *tok = NULL;
  int tile_sb_row;
  int tile_mb_cols = (tile_info->mi_col_end - tile_info->mi_col_start + 1) >> 1;

  tile_sb_row = mi_cols_aligned_to_sb(mi_row - tile_info->mi_row_start) >>
                MI_BLOCK_SIZE_LOG2;
  get_start_tok(cpi, tile_row, tile_col, mi_row, &tok);
  cpi->tplist[tile_row][tile_col][tile_sb_row].start = tok;

#if CONFIG_REALTIME_ONLY
  assert(cpi->sf.use_nonrd_pick_mode);
  encode_nonrd_sb_row(cpi, td, this_tile, mi_row, &tok);
#else
  if (cpi->sf.use_nonrd_pick_mode)
    encode_nonrd_sb_row(cpi, td, this_tile, mi_row, &tok);
  else
    encode_rd_sb_row(cpi, td, this_tile, mi_row, &tok);
#endif

  cpi->tplist[tile_row][tile_col][tile_sb_row].stop = tok;
  cpi->tplist[tile_row][tile_col][tile_sb_row].count =
      (unsigned int)(cpi->tplist[tile_row][tile_col][tile_sb_row].stop -
                     cpi->tplist[tile_row][tile_col][tile_sb_row].start);
  assert(tok - cpi->tplist[tile_row][tile_col][tile_sb_row].start <=
         get_token_alloc(MI_BLOCK_SIZE >> 1, tile_mb_cols));

  (void)tile_mb_cols;
}

void vp9_encode_tile(VP9_COMP *cpi, ThreadData *td, int tile_row,
                     int tile_col) {
  VP9_COMMON *const cm = &cpi->common;
  const int tile_cols = 1 << cm->log2_tile_cols;
  TileDataEnc *this_tile = &cpi->tile_data[tile_row * tile_cols + tile_col];
  const TileInfo *const tile_info = &this_tile->tile_info;
  const int mi_row_start = tile_info->mi_row_start;
  const int mi_row_end = tile_info->mi_row_end;
  int mi_row;

  for (mi_row = mi_row_start; mi_row < mi_row_end; mi_row += MI_BLOCK_SIZE)
    vp9_encode_sb_row(cpi, td, tile_row, tile_col, mi_row);
}

static void encode_tiles(VP9_COMP *cpi) {
  VP9_COMMON *const cm = &cpi->common;
  const int tile_cols = 1 << cm->log2_tile_cols;
  const int tile_rows = 1 << cm->log2_tile_rows;
  int tile_col, tile_row;

  vp9_init_tile_data(cpi);

  for (tile_row = 0; tile_row < tile_rows; ++tile_row)
    for (tile_col = 0; tile_col < tile_cols; ++tile_col)
      vp9_encode_tile(cpi, &cpi->td, tile_row, tile_col);
}

static int compare_kmeans_data(const void *a, const void *b) {
  if (((const KMEANS_DATA *)a)->value > ((const KMEANS_DATA *)b)->value) {
    return 1;
  } else if (((const KMEANS_DATA *)a)->value <
             ((const KMEANS_DATA *)b)->value) {
    return -1;
  } else {
    return 0;
  }
}

static void compute_boundary_ls(const double *ctr_ls, int k,
                                double *boundary_ls) {
  // boundary_ls[j] is the upper bound of data centered at ctr_ls[j]
  int j;
  for (j = 0; j < k - 1; ++j) {
    boundary_ls[j] = (ctr_ls[j] + ctr_ls[j + 1]) / 2.;
  }
  boundary_ls[k - 1] = DBL_MAX;
}

int vp9_get_group_idx(double value, double *boundary_ls, int k) {
  int group_idx = 0;
  while (value >= boundary_ls[group_idx]) {
    ++group_idx;
    if (group_idx == k - 1) {
      break;
    }
  }
  return group_idx;
}

void vp9_kmeans(double *ctr_ls, double *boundary_ls, int *count_ls, int k,
                KMEANS_DATA *arr, int size) {
  int i, j;
  int itr;
  int group_idx;
  double sum[MAX_KMEANS_GROUPS];
  int count[MAX_KMEANS_GROUPS];

  vpx_clear_system_state();

  assert(k >= 2 && k <= MAX_KMEANS_GROUPS);

  qsort(arr, size, sizeof(*arr), compare_kmeans_data);

  // initialize the center points
  for (j = 0; j < k; ++j) {
    ctr_ls[j] = arr[(size * (2 * j + 1)) / (2 * k)].value;
  }

  for (itr = 0; itr < 10; ++itr) {
    compute_boundary_ls(ctr_ls, k, boundary_ls);
    for (i = 0; i < MAX_KMEANS_GROUPS; ++i) {
      sum[i] = 0;
      count[i] = 0;
    }

    // Both the data and centers are sorted in ascending order.
    // As each data point is processed in order, its corresponding group index
    // can only increase. So we only need to reset the group index to zero here.
    group_idx = 0;
    for (i = 0; i < size; ++i) {
      while (arr[i].value >= boundary_ls[group_idx]) {
        // place samples into clusters
        ++group_idx;
        if (group_idx == k - 1) {
          break;
        }
      }
      sum[group_idx] += arr[i].value;
      ++count[group_idx];
    }

    for (group_idx = 0; group_idx < k; ++group_idx) {
      if (count[group_idx] > 0)
        ctr_ls[group_idx] = sum[group_idx] / count[group_idx];

      sum[group_idx] = 0;
      count[group_idx] = 0;
    }
  }

  // compute group_idx, boundary_ls and count_ls
  for (j = 0; j < k; ++j) {
    count_ls[j] = 0;
  }
  compute_boundary_ls(ctr_ls, k, boundary_ls);
  group_idx = 0;
  for (i = 0; i < size; ++i) {
    while (arr[i].value >= boundary_ls[group_idx]) {
      ++group_idx;
      if (group_idx == k - 1) {
        break;
      }
    }
    arr[i].group_idx = group_idx;
    ++count_ls[group_idx];
  }
}

static void encode_frame_internal(VP9_COMP *cpi) {
  SPEED_FEATURES *const sf = &cpi->sf;
  ThreadData *const td = &cpi->td;
  MACROBLOCK *const x = &td->mb;
  VP9_COMMON *const cm = &cpi->common;
  MACROBLOCKD *const xd = &x->e_mbd;
  const int gf_group_index = cpi->twopass.gf_group.index;

  xd->mi = cm->mi_grid_visible;
  xd->mi[0] = cm->mi;
  vp9_zero(*td->counts);
  vp9_zero(cpi->td.rd_counts);

  xd->lossless = cm->base_qindex == 0 && cm->y_dc_delta_q == 0 &&
                 cm->uv_dc_delta_q == 0 && cm->uv_ac_delta_q == 0;

#if CONFIG_VP9_HIGHBITDEPTH
  if (cm->use_highbitdepth)
    x->fwd_txfm4x4 = xd->lossless ? vp9_highbd_fwht4x4 : vpx_highbd_fdct4x4;
  else
    x->fwd_txfm4x4 = xd->lossless ? vp9_fwht4x4 : vpx_fdct4x4;
  x->highbd_inv_txfm_add =
      xd->lossless ? vp9_highbd_iwht4x4_add : vp9_highbd_idct4x4_add;
#else
  x->fwd_txfm4x4 = xd->lossless ? vp9_fwht4x4 : vpx_fdct4x4;
#endif  // CONFIG_VP9_HIGHBITDEPTH
  x->inv_txfm_add = xd->lossless ? vp9_iwht4x4_add : vp9_idct4x4_add;
  x->optimize = sf->optimize_coefficients == 1 && cpi->oxcf.pass != 1;
  if (xd->lossless) x->optimize = 0;
  x->sharpness = cpi->oxcf.sharpness;
  x->adjust_rdmult_by_segment = (cpi->oxcf.aq_mode == VARIANCE_AQ);

  cm->tx_mode = select_tx_mode(cpi, xd);

  vp9_frame_init_quantizer(cpi);

  vp9_initialize_rd_consts(cpi);
  vp9_initialize_me_consts(cpi, x, cm->base_qindex);
  init_encode_frame_mb_context(cpi);
  cm->use_prev_frame_mvs =
      !cm->error_resilient_mode && cm->width == cm->last_width &&
      cm->height == cm->last_height && !cm->intra_only && cm->last_show_frame;
  // Special case: set prev_mi to NULL when the previous mode info
  // context cannot be used.
  cm->prev_mi =
      cm->use_prev_frame_mvs ? cm->prev_mip + cm->mi_stride + 1 : NULL;

  x->quant_fp = cpi->sf.use_quant_fp;
  vp9_zero(x->skip_txfm);
  if (sf->use_nonrd_pick_mode) {
    // Initialize internal buffer pointers for rtc coding, where non-RD
    // mode decision is used and hence no buffer pointer swap needed.
    int i;
    struct macroblock_plane *const p = x->plane;
    struct macroblockd_plane *const pd = xd->plane;
    PICK_MODE_CONTEXT *ctx = &cpi->td.pc_root->none;

    for (i = 0; i < MAX_MB_PLANE; ++i) {
      p[i].coeff = ctx->coeff_pbuf[i][0];
      p[i].qcoeff = ctx->qcoeff_pbuf[i][0];
      pd[i].dqcoeff = ctx->dqcoeff_pbuf[i][0];
      p[i].eobs = ctx->eobs_pbuf[i][0];
    }
    vp9_zero(x->zcoeff_blk);

    if (cm->frame_type != KEY_FRAME && cpi->rc.frames_since_golden == 0 &&
        !(cpi->oxcf.lag_in_frames > 0 && cpi->oxcf.rc_mode == VPX_VBR) &&
        !cpi->use_svc)
      cpi->ref_frame_flags &= (~VP9_GOLD_FLAG);
  } else if (gf_group_index && gf_group_index < MAX_ARF_GOP_SIZE &&
             cpi->sf.enable_tpl_model) {
    TplDepFrame *tpl_frame = &cpi->tpl_stats[cpi->twopass.gf_group.index];
    TplDepStats *tpl_stats = tpl_frame->tpl_stats_ptr;

    int tpl_stride = tpl_frame->stride;
    int64_t intra_cost_base = 0;
    int64_t mc_dep_cost_base = 0;
    int row, col;

    for (row = 0; row < cm->mi_rows && tpl_frame->is_valid; ++row) {
      for (col = 0; col < cm->mi_cols; ++col) {
        TplDepStats *this_stats = &tpl_stats[row * tpl_stride + col];
        intra_cost_base += this_stats->intra_cost;
        mc_dep_cost_base += this_stats->mc_dep_cost;
      }
    }

    vpx_clear_system_state();

    if (tpl_frame->is_valid)
      cpi->rd.r0 = (double)intra_cost_base / mc_dep_cost_base;
  }

  for (MV_REFERENCE_FRAME ref_frame = LAST_FRAME; ref_frame <= ALTREF_FRAME;
       ++ref_frame) {
    if (cpi->ref_frame_flags & ref_frame_to_flag(ref_frame)) {
      if (cm->frame_refs[ref_frame - 1].sf.x_scale_fp == REF_INVALID_SCALE ||
          cm->frame_refs[ref_frame - 1].sf.y_scale_fp == REF_INVALID_SCALE)
        cpi->ref_frame_flags &= ~ref_frame_to_flag(ref_frame);
    }
  }

  // Frame segmentation
  if (cpi->oxcf.aq_mode == PERCEPTUAL_AQ) build_kmeans_segmentation(cpi);

  {
#if CONFIG_INTERNAL_STATS
    struct vpx_usec_timer emr_timer;
    vpx_usec_timer_start(&emr_timer);
#endif

    if (!cpi->row_mt) {
      cpi->row_mt_sync_read_ptr = vp9_row_mt_sync_read_dummy;
      cpi->row_mt_sync_write_ptr = vp9_row_mt_sync_write_dummy;
      // If allowed, encoding tiles in parallel with one thread handling one
      // tile when row based multi-threading is disabled.
      if (VPXMIN(cpi->oxcf.max_threads, 1 << cm->log2_tile_cols) > 1)
        vp9_encode_tiles_mt(cpi);
      else
        encode_tiles(cpi);
    } else {
      cpi->row_mt_sync_read_ptr = vp9_row_mt_sync_read;
      cpi->row_mt_sync_write_ptr = vp9_row_mt_sync_write;
      vp9_encode_tiles_row_mt(cpi);
    }

#if CONFIG_INTERNAL_STATS
    vpx_usec_timer_mark(&emr_timer);
    cpi->time_encode_sb_row += vpx_usec_timer_elapsed(&emr_timer);
#endif
  }

  sf->skip_encode_frame =
      sf->skip_encode_sb ? get_skip_encode_frame(cm, td) : 0;

#if 0
  // Keep record of the total distortion this time around for future use
  cpi->last_frame_distortion = cpi->frame_distortion;
#endif
}

static INTERP_FILTER get_interp_filter(
    const int64_t threshes[SWITCHABLE_FILTER_CONTEXTS], int is_alt_ref) {
  if (!is_alt_ref && threshes[EIGHTTAP_SMOOTH] > threshes[EIGHTTAP] &&
      threshes[EIGHTTAP_SMOOTH] > threshes[EIGHTTAP_SHARP] &&
      threshes[EIGHTTAP_SMOOTH] > threshes[SWITCHABLE - 1]) {
    return EIGHTTAP_SMOOTH;
  } else if (threshes[EIGHTTAP_SHARP] > threshes[EIGHTTAP] &&
             threshes[EIGHTTAP_SHARP] > threshes[SWITCHABLE - 1]) {
    return EIGHTTAP_SHARP;
  } else if (threshes[EIGHTTAP] > threshes[SWITCHABLE - 1]) {
    return EIGHTTAP;
  } else {
    return SWITCHABLE;
  }
}

static int compute_frame_aq_offset(struct VP9_COMP *cpi) {
  VP9_COMMON *const cm = &cpi->common;
  MODE_INFO **mi_8x8_ptr = cm->mi_grid_visible;
  struct segmentation *const seg = &cm->seg;

  int mi_row, mi_col;
  int sum_delta = 0;
  int qdelta_index;
  int segment_id;

  for (mi_row = 0; mi_row < cm->mi_rows; mi_row++) {
    MODE_INFO **mi_8x8 = mi_8x8_ptr;
    for (mi_col = 0; mi_col < cm->mi_cols; mi_col++, mi_8x8++) {
      segment_id = mi_8x8[0]->segment_id;
      qdelta_index = get_segdata(seg, segment_id, SEG_LVL_ALT_Q);
      sum_delta += qdelta_index;
    }
    mi_8x8_ptr += cm->mi_stride;
  }

  return sum_delta / (cm->mi_rows * cm->mi_cols);
}

static void restore_encode_params(VP9_COMP *cpi) {
  VP9_COMMON *const cm = &cpi->common;
  int tile_idx;
  int i, j;
  TileDataEnc *tile_data;
  RD_OPT *rd_opt = &cpi->rd;
  for (i = 0; i < MAX_REF_FRAMES; i++) {
    for (j = 0; j < REFERENCE_MODES; j++)
      rd_opt->prediction_type_threshes[i][j] =
          rd_opt->prediction_type_threshes_prev[i][j];

    for (j = 0; j < SWITCHABLE_FILTER_CONTEXTS; j++)
      rd_opt->filter_threshes[i][j] = rd_opt->filter_threshes_prev[i][j];
  }

  for (tile_idx = 0; tile_idx < cpi->allocated_tiles; tile_idx++) {
    assert(cpi->tile_data);
    tile_data = &cpi->tile_data[tile_idx];
    vp9_copy(tile_data->thresh_freq_fact, tile_data->thresh_freq_fact_prev);
  }

  cm->interp_filter = cpi->sf.default_interp_filter;
}

void vp9_encode_frame(VP9_COMP *cpi) {
  VP9_COMMON *const cm = &cpi->common;

  restore_encode_params(cpi);

#if CONFIG_MISMATCH_DEBUG
  mismatch_reset_frame(MAX_MB_PLANE);
#endif

  // In the longer term the encoder should be generalized to match the
  // decoder such that we allow compound where one of the 3 buffers has a
  // different sign bias and that buffer is then the fixed ref. However, this
  // requires further work in the rd loop. For now the only supported encoder
  // side behavior is where the ALT ref buffer has opposite sign bias to
  // the other two.
  if (!frame_is_intra_only(cm)) {
    if (vp9_compound_reference_allowed(cm)) {
      cpi->allow_comp_inter_inter = 1;
      vp9_setup_compound_reference_mode(cm);
    } else {
      cpi->allow_comp_inter_inter = 0;
    }
  }

  if (cpi->sf.frame_parameter_update) {
    int i;
    RD_OPT *const rd_opt = &cpi->rd;
    FRAME_COUNTS *counts = cpi->td.counts;
    RD_COUNTS *const rdc = &cpi->td.rd_counts;

    // This code does a single RD pass over the whole frame assuming
    // either compound, single or hybrid prediction as per whatever has
    // worked best for that type of frame in the past.
    // It also predicts whether another coding mode would have worked
    // better than this coding mode. If that is the case, it remembers
    // that for subsequent frames.
    // It also does the same analysis for transform size selection.
    const MV_REFERENCE_FRAME frame_type = get_frame_type(cpi);
    int64_t *const mode_thrs = rd_opt->prediction_type_threshes[frame_type];
    int64_t *const filter_thrs = rd_opt->filter_threshes[frame_type];
    const int is_alt_ref = frame_type == ALTREF_FRAME;

    /* prediction (compound, single or hybrid) mode selection */
    if (is_alt_ref || !cpi->allow_comp_inter_inter)
      cm->reference_mode = SINGLE_REFERENCE;
    else if (mode_thrs[COMPOUND_REFERENCE] > mode_thrs[SINGLE_REFERENCE] &&
             mode_thrs[COMPOUND_REFERENCE] > mode_thrs[REFERENCE_MODE_SELECT] &&
             check_dual_ref_flags(cpi) && cpi->static_mb_pct == 100)
      cm->reference_mode = COMPOUND_REFERENCE;
    else if (mode_thrs[SINGLE_REFERENCE] > mode_thrs[REFERENCE_MODE_SELECT])
      cm->reference_mode = SINGLE_REFERENCE;
    else
      cm->reference_mode = REFERENCE_MODE_SELECT;

    if (cm->interp_filter == SWITCHABLE)
      cm->interp_filter = get_interp_filter(filter_thrs, is_alt_ref);

#if CONFIG_COLLECT_COMPONENT_TIMING
    start_timing(cpi, encode_frame_internal_time);
#endif
    encode_frame_internal(cpi);
#if CONFIG_COLLECT_COMPONENT_TIMING
    end_timing(cpi, encode_frame_internal_time);
#endif

    for (i = 0; i < REFERENCE_MODES; ++i)
      mode_thrs[i] = (mode_thrs[i] + rdc->comp_pred_diff[i] / cm->MBs) / 2;

    for (i = 0; i < SWITCHABLE_FILTER_CONTEXTS; ++i)
      filter_thrs[i] = (filter_thrs[i] + rdc->filter_diff[i] / cm->MBs) / 2;

    if (cm->reference_mode == REFERENCE_MODE_SELECT) {
      int single_count_zero = 0;
      int comp_count_zero = 0;

      for (i = 0; i < COMP_INTER_CONTEXTS; i++) {
        single_count_zero += counts->comp_inter[i][0];
        comp_count_zero += counts->comp_inter[i][1];
      }

      if (comp_count_zero == 0) {
        cm->reference_mode = SINGLE_REFERENCE;
        vp9_zero(counts->comp_inter);
      } else if (single_count_zero == 0) {
        cm->reference_mode = COMPOUND_REFERENCE;
        vp9_zero(counts->comp_inter);
      }
    }

    if (cm->tx_mode == TX_MODE_SELECT) {
      int count4x4 = 0;
      int count8x8_lp = 0, count8x8_8x8p = 0;
      int count16x16_16x16p = 0, count16x16_lp = 0;
      int count32x32 = 0;

      for (i = 0; i < TX_SIZE_CONTEXTS; ++i) {
        count4x4 += counts->tx.p32x32[i][TX_4X4];
        count4x4 += counts->tx.p16x16[i][TX_4X4];
        count4x4 += counts->tx.p8x8[i][TX_4X4];

        count8x8_lp += counts->tx.p32x32[i][TX_8X8];
        count8x8_lp += counts->tx.p16x16[i][TX_8X8];
        count8x8_8x8p += counts->tx.p8x8[i][TX_8X8];

        count16x16_16x16p += counts->tx.p16x16[i][TX_16X16];
        count16x16_lp += counts->tx.p32x32[i][TX_16X16];
        count32x32 += counts->tx.p32x32[i][TX_32X32];
      }
      if (count4x4 == 0 && count16x16_lp == 0 && count16x16_16x16p == 0 &&
          count32x32 == 0) {
        cm->tx_mode = ALLOW_8X8;
        reset_skip_tx_size(cm, TX_8X8);
      } else if (count8x8_8x8p == 0 && count16x16_16x16p == 0 &&
                 count8x8_lp == 0 && count16x16_lp == 0 && count32x32 == 0) {
        cm->tx_mode = ONLY_4X4;
        reset_skip_tx_size(cm, TX_4X4);
      } else if (count8x8_lp == 0 && count16x16_lp == 0 && count4x4 == 0) {
        cm->tx_mode = ALLOW_32X32;
      } else if (count32x32 == 0 && count8x8_lp == 0 && count4x4 == 0) {
        cm->tx_mode = ALLOW_16X16;
        reset_skip_tx_size(cm, TX_16X16);
      }
    }
  } else {
    FRAME_COUNTS *counts = cpi->td.counts;
    cm->reference_mode = SINGLE_REFERENCE;
    if (cpi->allow_comp_inter_inter && cpi->sf.use_compound_nonrd_pickmode &&
        cpi->rc.alt_ref_gf_group && !cpi->rc.is_src_frame_alt_ref &&
        cm->frame_type != KEY_FRAME)
      cm->reference_mode = REFERENCE_MODE_SELECT;

    encode_frame_internal(cpi);

    if (cm->reference_mode == REFERENCE_MODE_SELECT) {
      int single_count_zero = 0;
      int comp_count_zero = 0;
      int i;
      for (i = 0; i < COMP_INTER_CONTEXTS; i++) {
        single_count_zero += counts->comp_inter[i][0];
        comp_count_zero += counts->comp_inter[i][1];
      }
      if (comp_count_zero == 0) {
        cm->reference_mode = SINGLE_REFERENCE;
        vp9_zero(counts->comp_inter);
      } else if (single_count_zero == 0) {
        cm->reference_mode = COMPOUND_REFERENCE;
        vp9_zero(counts->comp_inter);
      }
    }
  }

  // If segmented AQ is enabled compute the average AQ weighting.
  if (cm->seg.enabled && (cpi->oxcf.aq_mode != NO_AQ) &&
      (cm->seg.update_map || cm->seg.update_data)) {
    cm->seg.aq_av_offset = compute_frame_aq_offset(cpi);
  }
}

static void sum_intra_stats(FRAME_COUNTS *counts, const MODE_INFO *mi) {
  const PREDICTION_MODE y_mode = mi->mode;
  const PREDICTION_MODE uv_mode = mi->uv_mode;
  const BLOCK_SIZE bsize = mi->sb_type;

  if (bsize < BLOCK_8X8) {
    int idx, idy;
    const int num_4x4_w = num_4x4_blocks_wide_lookup[bsize];
    const int num_4x4_h = num_4x4_blocks_high_lookup[bsize];
    for (idy = 0; idy < 2; idy += num_4x4_h)
      for (idx = 0; idx < 2; idx += num_4x4_w)
        ++counts->y_mode[0][mi->bmi[idy * 2 + idx].as_mode];
  } else {
    ++counts->y_mode[size_group_lookup[bsize]][y_mode];
  }

  ++counts->uv_mode[y_mode][uv_mode];
}

static void update_zeromv_cnt(VP9_COMP *const cpi, const MODE_INFO *const mi,
                              int mi_row, int mi_col, BLOCK_SIZE bsize) {
  const VP9_COMMON *const cm = &cpi->common;
  MV mv = mi->mv[0].as_mv;
  const int bw = num_8x8_blocks_wide_lookup[bsize];
  const int bh = num_8x8_blocks_high_lookup[bsize];
  const int xmis = VPXMIN(cm->mi_cols - mi_col, bw);
  const int ymis = VPXMIN(cm->mi_rows - mi_row, bh);
  const int block_index = mi_row * cm->mi_cols + mi_col;
  int x, y;
  for (y = 0; y < ymis; y++)
    for (x = 0; x < xmis; x++) {
      int map_offset = block_index + y * cm->mi_cols + x;
      if (mi->ref_frame[0] == LAST_FRAME && is_inter_block(mi) &&
          mi->segment_id <= CR_SEGMENT_ID_BOOST2) {
        if (abs(mv.row) < 8 && abs(mv.col) < 8) {
          if (cpi->consec_zero_mv[map_offset] < 255)
            cpi->consec_zero_mv[map_offset]++;
        } else {
          cpi->consec_zero_mv[map_offset] = 0;
        }
      }
    }
}

static void encode_superblock(VP9_COMP *cpi, ThreadData *td, TOKENEXTRA **t,
                              int output_enabled, int mi_row, int mi_col,
                              BLOCK_SIZE bsize, PICK_MODE_CONTEXT *ctx) {
  VP9_COMMON *const cm = &cpi->common;
  MACROBLOCK *const x = &td->mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  MODE_INFO *mi = xd->mi[0];
  const int seg_skip =
      segfeature_active(&cm->seg, mi->segment_id, SEG_LVL_SKIP);
  x->skip_recode = !x->select_tx_size && mi->sb_type >= BLOCK_8X8 &&
                   cpi->oxcf.aq_mode != COMPLEXITY_AQ &&
                   cpi->oxcf.aq_mode != CYCLIC_REFRESH_AQ &&
                   cpi->sf.allow_skip_recode;

  if (!x->skip_recode && !cpi->sf.use_nonrd_pick_mode)
    memset(x->skip_txfm, 0, sizeof(x->skip_txfm));

  x->skip_optimize = ctx->is_coded;
  ctx->is_coded = 1;
  x->use_lp32x32fdct = cpi->sf.use_lp32x32fdct;
  x->skip_encode = (!output_enabled && cpi->sf.skip_encode_frame &&
                    x->q_index < QIDX_SKIP_THRESH);

  if (x->skip_encode) return;

  if (!is_inter_block(mi)) {
    int plane;
#if CONFIG_BETTER_HW_COMPATIBILITY && CONFIG_VP9_HIGHBITDEPTH
    if ((xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH) &&
        (xd->above_mi == NULL || xd->left_mi == NULL) &&
        need_top_left[mi->uv_mode])
      assert(0);
#endif  // CONFIG_BETTER_HW_COMPATIBILITY && CONFIG_VP9_HIGHBITDEPTH
    mi->skip = 1;
    for (plane = 0; plane < MAX_MB_PLANE; ++plane)
      vp9_encode_intra_block_plane(x, VPXMAX(bsize, BLOCK_8X8), plane, 1);
    if (output_enabled) sum_intra_stats(td->counts, mi);
    vp9_tokenize_sb(cpi, td, t, !output_enabled, seg_skip,
                    VPXMAX(bsize, BLOCK_8X8));
  } else {
    int ref;
    const int is_compound = has_second_ref(mi);
    set_ref_ptrs(cm, xd, mi->ref_frame[0], mi->ref_frame[1]);
    for (ref = 0; ref < 1 + is_compound; ++ref) {
      YV12_BUFFER_CONFIG *cfg = get_ref_frame_buffer(cpi, mi->ref_frame[ref]);
      assert(cfg != NULL);
      vp9_setup_pre_planes(xd, ref, cfg, mi_row, mi_col,
                           &xd->block_refs[ref]->sf);
    }
    if (!(cpi->sf.reuse_inter_pred_sby && ctx->pred_pixel_ready) || seg_skip)
      vp9_build_inter_predictors_sby(xd, mi_row, mi_col,
                                     VPXMAX(bsize, BLOCK_8X8));

    vp9_build_inter_predictors_sbuv(xd, mi_row, mi_col,
                                    VPXMAX(bsize, BLOCK_8X8));

#if CONFIG_MISMATCH_DEBUG
    if (output_enabled) {
      int plane;
      for (plane = 0; plane < MAX_MB_PLANE; ++plane) {
        const struct macroblockd_plane *pd = &xd->plane[plane];
        int pixel_c, pixel_r;
        const BLOCK_SIZE plane_bsize =
            get_plane_block_size(VPXMAX(bsize, BLOCK_8X8), &xd->plane[plane]);
        const int bw = get_block_width(plane_bsize);
        const int bh = get_block_height(plane_bsize);
        mi_to_pixel_loc(&pixel_c, &pixel_r, mi_col, mi_row, 0, 0,
                        pd->subsampling_x, pd->subsampling_y);

        mismatch_record_block_pre(pd->dst.buf, pd->dst.stride, plane, pixel_c,
                                  pixel_r, bw, bh,
                                  xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH);
      }
    }
#endif

    vp9_encode_sb(x, VPXMAX(bsize, BLOCK_8X8), mi_row, mi_col, output_enabled);
    vp9_tokenize_sb(cpi, td, t, !output_enabled, seg_skip,
                    VPXMAX(bsize, BLOCK_8X8));
  }

  if (seg_skip) {
    assert(mi->skip);
  }

  if (output_enabled) {
    if (cm->tx_mode == TX_MODE_SELECT && mi->sb_type >= BLOCK_8X8 &&
        !(is_inter_block(mi) && mi->skip)) {
      ++get_tx_counts(max_txsize_lookup[bsize], get_tx_size_context(xd),
                      &td->counts->tx)[mi->tx_size];
    } else {
      // The new intra coding scheme requires no change of transform size
      if (is_inter_block(mi)) {
        mi->tx_size = VPXMIN(tx_mode_to_biggest_tx_size[cm->tx_mode],
                             max_txsize_lookup[bsize]);
      } else {
        mi->tx_size = (bsize >= BLOCK_8X8) ? mi->tx_size : TX_4X4;
      }
    }

    ++td->counts->tx.tx_totals[mi->tx_size];
    ++td->counts->tx.tx_totals[get_uv_tx_size(mi, &xd->plane[1])];
    if (cm->seg.enabled && cpi->oxcf.aq_mode == CYCLIC_REFRESH_AQ &&
        cpi->cyclic_refresh->content_mode)
      vp9_cyclic_refresh_update_sb_postencode(cpi, mi, mi_row, mi_col, bsize);
    if (cpi->oxcf.pass == 0 && cpi->svc.temporal_layer_id == 0 &&
        (!cpi->use_svc ||
         (cpi->use_svc &&
          !cpi->svc.layer_context[cpi->svc.temporal_layer_id].is_key_frame &&
          cpi->svc.spatial_layer_id == cpi->svc.number_spatial_layers - 1)))
      update_zeromv_cnt(cpi, mi, mi_row, mi_col, bsize);
  }
}
