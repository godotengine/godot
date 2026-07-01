/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "./vp9_rtcd.h"
#include "./vpx_dsp_rtcd.h"

#include "vpx/vpx_codec.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_mem/vpx_mem.h"
#include "vpx_ports/compiler_attributes.h"

#include "vp9/common/vp9_blockd.h"
#include "vp9/common/vp9_common.h"
#include "vp9/common/vp9_mvref_common.h"
#include "vp9/common/vp9_pred_common.h"
#include "vp9/common/vp9_reconinter.h"
#include "vp9/common/vp9_reconintra.h"
#include "vp9/common/vp9_scan.h"

#include "vp9/encoder/vp9_cost.h"
#include "vp9/encoder/vp9_encoder.h"
#include "vp9/encoder/vp9_pickmode.h"
#include "vp9/encoder/vp9_ratectrl.h"
#include "vp9/encoder/vp9_rd.h"

typedef struct {
  uint8_t *data;
  int stride;
  int in_use;
} PRED_BUFFER;

typedef struct {
  PRED_BUFFER *best_pred;
  PREDICTION_MODE best_mode;
  TX_SIZE best_tx_size;
  TX_SIZE best_intra_tx_size;
  MV_REFERENCE_FRAME best_ref_frame;
  MV_REFERENCE_FRAME best_second_ref_frame;
  uint8_t best_mode_skip_txfm;
  INTERP_FILTER best_pred_filter;
} BEST_PICKMODE;

static const int pos_shift_16x16[4][4] = {
  { 9, 10, 13, 14 }, { 11, 12, 15, 16 }, { 17, 18, 21, 22 }, { 19, 20, 23, 24 }
};

static int mv_refs_rt(VP9_COMP *cpi, const VP9_COMMON *cm, const MACROBLOCK *x,
                      const MACROBLOCKD *xd, const TileInfo *const tile,
                      MODE_INFO *mi, MV_REFERENCE_FRAME ref_frame,
                      int_mv *mv_ref_list, int_mv *base_mv, int mi_row,
                      int mi_col, int use_base_mv) {
  const int *ref_sign_bias = cm->ref_frame_sign_bias;
  int i, refmv_count = 0;

  const POSITION *const mv_ref_search = mv_ref_blocks[mi->sb_type];

  int different_ref_found = 0;
  int context_counter = 0;
  int const_motion = 0;

  // Blank the reference vector list
  memset(mv_ref_list, 0, sizeof(*mv_ref_list) * MAX_MV_REF_CANDIDATES);

  // The nearest 2 blocks are treated differently
  // if the size < 8x8 we get the mv from the bmi substructure,
  // and we also need to keep a mode count.
  for (i = 0; i < 2; ++i) {
    const POSITION *const mv_ref = &mv_ref_search[i];
    if (is_inside(tile, mi_col, mi_row, cm->mi_rows, mv_ref)) {
      const MODE_INFO *const candidate_mi =
          xd->mi[mv_ref->col + mv_ref->row * xd->mi_stride];
      // Keep counts for entropy encoding.
      context_counter += mode_2_counter[candidate_mi->mode];
      different_ref_found = 1;

      if (candidate_mi->ref_frame[0] == ref_frame)
        ADD_MV_REF_LIST(get_sub_block_mv(candidate_mi, 0, mv_ref->col, -1),
                        refmv_count, mv_ref_list, Done);
    }
  }

  const_motion = 1;

  // Check the rest of the neighbors in much the same way
  // as before except we don't need to keep track of sub blocks or
  // mode counts.
  for (; i < MVREF_NEIGHBOURS && !refmv_count; ++i) {
    const POSITION *const mv_ref = &mv_ref_search[i];
    if (is_inside(tile, mi_col, mi_row, cm->mi_rows, mv_ref)) {
      const MODE_INFO *const candidate_mi =
          xd->mi[mv_ref->col + mv_ref->row * xd->mi_stride];
      different_ref_found = 1;

      if (candidate_mi->ref_frame[0] == ref_frame)
        ADD_MV_REF_LIST(candidate_mi->mv[0], refmv_count, mv_ref_list, Done);
    }
  }

  // Since we couldn't find 2 mvs from the same reference frame
  // go back through the neighbors and find motion vectors from
  // different reference frames.
  if (different_ref_found && !refmv_count) {
    for (i = 0; i < MVREF_NEIGHBOURS; ++i) {
      const POSITION *mv_ref = &mv_ref_search[i];
      if (is_inside(tile, mi_col, mi_row, cm->mi_rows, mv_ref)) {
        const MODE_INFO *const candidate_mi =
            xd->mi[mv_ref->col + mv_ref->row * xd->mi_stride];

        // If the candidate is INTRA we don't want to consider its mv.
        IF_DIFF_REF_FRAME_ADD_MV(candidate_mi, ref_frame, ref_sign_bias,
                                 refmv_count, mv_ref_list, Done);
      }
    }
  }
  if (use_base_mv &&
      !cpi->svc.layer_context[cpi->svc.temporal_layer_id].is_key_frame &&
      ref_frame == LAST_FRAME) {
    // Get base layer mv.
    const int prev_layer = cpi->svc.spatial_layer_id - 1;
    const int index =
        (mi_col >> 1) + (mi_row >> 1) * cpi->svc.mi_cols[prev_layer];
    // prev_frame->mvs[] is allocated to size mi_cols * mi_rows corresponding
    // to the previous spatial layer, so the index check is against
    // svc.mi_col/rows[prev_layer].
    if (index < cpi->svc.mi_cols[prev_layer] * cpi->svc.mi_rows[prev_layer]) {
      MV_REF *candidate = &cm->prev_frame->mvs[index];
      // Avoid using base_mv if scaled mv is out of range, for either component.
      if (candidate->mv[0].as_int != INVALID_MV &&
          abs(candidate->mv[0].as_mv.row) <= INT16_MAX >> 1 &&
          abs(candidate->mv[0].as_mv.col) <= INT16_MAX >> 1) {
        base_mv->as_mv.row = candidate->mv[0].as_mv.row * 2;
        base_mv->as_mv.col = candidate->mv[0].as_mv.col * 2;
        clamp_mv_ref(&base_mv->as_mv, xd);
      } else {
        base_mv->as_int = INVALID_MV;
      }
    }
  }

Done:

  x->mbmi_ext->mode_context[ref_frame] = counter_to_context[context_counter];

  // Clamp vectors
  for (i = 0; i < MAX_MV_REF_CANDIDATES; ++i)
    clamp_mv_ref(&mv_ref_list[i].as_mv, xd);

  return const_motion;
}

static int combined_motion_search(VP9_COMP *cpi, MACROBLOCK *x,
                                  BLOCK_SIZE bsize, int mi_row, int mi_col,
                                  int_mv *tmp_mv, int *rate_mv,
                                  int64_t best_rd_sofar, int use_base_mv) {
  MACROBLOCKD *xd = &x->e_mbd;
  MODE_INFO *mi = xd->mi[0];
  struct buf_2d backup_yv12[MAX_MB_PLANE] = { { 0, 0 } };
  const int step_param = cpi->sf.mv.fullpel_search_step_param;
  const int sadpb = x->sadperbit16;
  MV mvp_full;
  const int ref = mi->ref_frame[0];
  const MV ref_mv = x->mbmi_ext->ref_mvs[ref][0].as_mv;
  MV center_mv;
  uint32_t dis;
  int rate_mode;
  const MvLimits tmp_mv_limits = x->mv_limits;
  int rv = 0;
  int cost_list[5];
  int search_subpel = 1;
  const YV12_BUFFER_CONFIG *scaled_ref_frame =
      vp9_get_scaled_ref_frame(cpi, ref);
  if (scaled_ref_frame) {
    int i;
    // Swap out the reference frame for a version that's been scaled to
    // match the resolution of the current frame, allowing the existing
    // motion search code to be used without additional modifications.
    for (i = 0; i < MAX_MB_PLANE; i++) backup_yv12[i] = xd->plane[i].pre[0];
    vp9_setup_pre_planes(xd, 0, scaled_ref_frame, mi_row, mi_col, NULL);
  }
  vp9_set_mv_search_range(&x->mv_limits, &ref_mv);

  // Limit motion vector for large lightning change.
  if (cpi->oxcf.speed > 5 && x->lowvar_highsumdiff) {
    x->mv_limits.col_min = VPXMAX(x->mv_limits.col_min, -10);
    x->mv_limits.row_min = VPXMAX(x->mv_limits.row_min, -10);
    x->mv_limits.col_max = VPXMIN(x->mv_limits.col_max, 10);
    x->mv_limits.row_max = VPXMIN(x->mv_limits.row_max, 10);
  }

  assert(x->mv_best_ref_index[ref] <= 2);
  if (x->mv_best_ref_index[ref] < 2)
    mvp_full = x->mbmi_ext->ref_mvs[ref][x->mv_best_ref_index[ref]].as_mv;
  else
    mvp_full = x->pred_mv[ref];

  mvp_full.col >>= 3;
  mvp_full.row >>= 3;

  if (!use_base_mv)
    center_mv = ref_mv;
  else
    center_mv = tmp_mv->as_mv;

  if (x->sb_use_mv_part) {
    tmp_mv->as_mv.row = x->sb_mvrow_part >> 3;
    tmp_mv->as_mv.col = x->sb_mvcol_part >> 3;
  } else {
    vp9_full_pixel_search(
        cpi, x, bsize, &mvp_full, step_param, cpi->sf.mv.search_method, sadpb,
        cond_cost_list(cpi, cost_list), &center_mv, &tmp_mv->as_mv, INT_MAX, 0);
  }

  x->mv_limits = tmp_mv_limits;

  // calculate the bit cost on motion vector
  mvp_full.row = tmp_mv->as_mv.row * 8;
  mvp_full.col = tmp_mv->as_mv.col * 8;

  *rate_mv = vp9_mv_bit_cost(&mvp_full, &ref_mv, x->nmvjointcost, x->mvcost,
                             MV_COST_WEIGHT);

  rate_mode =
      cpi->inter_mode_cost[x->mbmi_ext->mode_context[ref]][INTER_OFFSET(NEWMV)];
  rv =
      !(RDCOST(x->rdmult, x->rddiv, (*rate_mv + rate_mode), 0) > best_rd_sofar);

  // For SVC on non-reference frame, avoid subpel for (0, 0) motion.
  if (cpi->use_svc && cpi->svc.non_reference_frame) {
    if (mvp_full.row == 0 && mvp_full.col == 0) search_subpel = 0;
  }

  if (rv && search_subpel) {
    SUBPEL_FORCE_STOP subpel_force_stop = cpi->sf.mv.subpel_force_stop;
    if (use_base_mv && cpi->sf.base_mv_aggressive) subpel_force_stop = HALF_PEL;
    if (cpi->sf.mv.enable_adaptive_subpel_force_stop) {
      const int mv_thresh = cpi->sf.mv.adapt_subpel_force_stop.mv_thresh;
      if (abs(tmp_mv->as_mv.row) >= mv_thresh ||
          abs(tmp_mv->as_mv.col) >= mv_thresh)
        subpel_force_stop = cpi->sf.mv.adapt_subpel_force_stop.force_stop_above;
      else
        subpel_force_stop = cpi->sf.mv.adapt_subpel_force_stop.force_stop_below;
    }
    cpi->find_fractional_mv_step(
        x, &tmp_mv->as_mv, &ref_mv, cpi->common.allow_high_precision_mv,
        x->errorperbit, &cpi->fn_ptr[bsize], subpel_force_stop,
        cpi->sf.mv.subpel_search_level, cond_cost_list(cpi, cost_list),
        x->nmvjointcost, x->mvcost, &dis, &x->pred_sse[ref], NULL, 0, 0,
        cpi->sf.use_accurate_subpel_search);
    *rate_mv = vp9_mv_bit_cost(&tmp_mv->as_mv, &ref_mv, x->nmvjointcost,
                               x->mvcost, MV_COST_WEIGHT);
  }

  if (scaled_ref_frame) {
    int i;
    for (i = 0; i < MAX_MB_PLANE; i++) xd->plane[i].pre[0] = backup_yv12[i];
  }
  return rv;
}

static void block_variance(const uint8_t *src, int src_stride,
                           const uint8_t *ref, int ref_stride, int w, int h,
                           unsigned int *sse, int *sum, int block_size,
#if CONFIG_VP9_HIGHBITDEPTH
                           int use_highbitdepth, vpx_bit_depth_t bd,
#endif
                           uint32_t *sse8x8, int *sum8x8, uint32_t *var8x8) {
  int i, j, k = 0;
  uint32_t k_sqr = 0;

  *sse = 0;
  *sum = 0;

  for (i = 0; i < h; i += block_size) {
    for (j = 0; j < w; j += block_size) {
#if CONFIG_VP9_HIGHBITDEPTH
      if (use_highbitdepth) {
        switch (bd) {
          case VPX_BITS_8:
            vpx_highbd_8_get8x8var(src + src_stride * i + j, src_stride,
                                   ref + ref_stride * i + j, ref_stride,
                                   &sse8x8[k], &sum8x8[k]);
            break;
          case VPX_BITS_10:
            vpx_highbd_10_get8x8var(src + src_stride * i + j, src_stride,
                                    ref + ref_stride * i + j, ref_stride,
                                    &sse8x8[k], &sum8x8[k]);
            break;
          case VPX_BITS_12:
            vpx_highbd_12_get8x8var(src + src_stride * i + j, src_stride,
                                    ref + ref_stride * i + j, ref_stride,
                                    &sse8x8[k], &sum8x8[k]);
            break;
        }
      } else {
        vpx_get8x8var(src + src_stride * i + j, src_stride,
                      ref + ref_stride * i + j, ref_stride, &sse8x8[k],
                      &sum8x8[k]);
      }
#else
      vpx_get8x8var(src + src_stride * i + j, src_stride,
                    ref + ref_stride * i + j, ref_stride, &sse8x8[k],
                    &sum8x8[k]);
#endif
      *sse += sse8x8[k];
      *sum += sum8x8[k];
      k_sqr = (uint32_t)(((int64_t)sum8x8[k] * sum8x8[k]) >> 6);
      var8x8[k] = sse8x8[k] > k_sqr ? sse8x8[k] - k_sqr : k_sqr - sse8x8[k];
      k++;
    }
  }
}

static void calculate_variance(int bw, int bh, TX_SIZE tx_size,
                               unsigned int *sse_i, int *sum_i,
                               unsigned int *var_o, unsigned int *sse_o,
                               int *sum_o) {
  const BLOCK_SIZE unit_size = txsize_to_bsize[tx_size];
  const int nw = 1 << (bw - b_width_log2_lookup[unit_size]);
  const int nh = 1 << (bh - b_height_log2_lookup[unit_size]);
  int i, j, k = 0;
  uint32_t k_sqr = 0;

  for (i = 0; i < nh; i += 2) {
    for (j = 0; j < nw; j += 2) {
      sse_o[k] = sse_i[i * nw + j] + sse_i[i * nw + j + 1] +
                 sse_i[(i + 1) * nw + j] + sse_i[(i + 1) * nw + j + 1];
      sum_o[k] = sum_i[i * nw + j] + sum_i[i * nw + j + 1] +
                 sum_i[(i + 1) * nw + j] + sum_i[(i + 1) * nw + j + 1];
      k_sqr = (uint32_t)(((int64_t)sum_o[k] * sum_o[k]) >>
                         (b_width_log2_lookup[unit_size] +
                          b_height_log2_lookup[unit_size] + 6));
      var_o[k] = sse_o[k] > k_sqr ? sse_o[k] - k_sqr : k_sqr - sse_o[k];
      k++;
    }
  }
}

// Adjust the ac_thr according to speed, width, height and normalized sum
static int ac_thr_factor(const int speed, const int width, const int height,
                         const int norm_sum) {
  if (speed >= 8 && norm_sum < 5) {
    if (width <= 640 && height <= 480)
      return 4;
    else
      return 2;
  }
  return 1;
}

static TX_SIZE calculate_tx_size(VP9_COMP *const cpi, BLOCK_SIZE bsize,
                                 MACROBLOCKD *const xd, unsigned int var,
                                 unsigned int sse, int64_t ac_thr,
                                 unsigned int source_variance, int is_intra) {
  // TODO(marpan): Tune selection for intra-modes, screen content, etc.
  TX_SIZE tx_size;
  unsigned int var_thresh = is_intra ? (unsigned int)ac_thr : 1;
  int limit_tx = 1;
  if (cpi->oxcf.aq_mode == CYCLIC_REFRESH_AQ &&
      (source_variance == 0 || var < var_thresh))
    limit_tx = 0;
  if (cpi->common.tx_mode == TX_MODE_SELECT) {
    if (sse > (var << 2))
      tx_size = VPXMIN(max_txsize_lookup[bsize],
                       tx_mode_to_biggest_tx_size[cpi->common.tx_mode]);
    else
      tx_size = TX_8X8;
    if (cpi->oxcf.aq_mode == CYCLIC_REFRESH_AQ && limit_tx &&
        cyclic_refresh_segment_id_boosted(xd->mi[0]->segment_id))
      tx_size = TX_8X8;
    else if (tx_size > TX_16X16 && limit_tx)
      tx_size = TX_16X16;
    // For screen-content force 4X4 tx_size over 8X8, for large variance.
    if (cpi->oxcf.content == VP9E_CONTENT_SCREEN && tx_size == TX_8X8 &&
        bsize <= BLOCK_16X16 && ((var >> 5) > (unsigned int)ac_thr))
      tx_size = TX_4X4;
  } else {
    tx_size = VPXMIN(max_txsize_lookup[bsize],
                     tx_mode_to_biggest_tx_size[cpi->common.tx_mode]);
  }
  return tx_size;
}

static void compute_intra_yprediction(PREDICTION_MODE mode, BLOCK_SIZE bsize,
                                      MACROBLOCK *x, MACROBLOCKD *xd) {
  struct macroblockd_plane *const pd = &xd->plane[0];
  struct macroblock_plane *const p = &x->plane[0];
  uint8_t *const src_buf_base = p->src.buf;
  uint8_t *const dst_buf_base = pd->dst.buf;
  const int src_stride = p->src.stride;
  const int dst_stride = pd->dst.stride;
  // block and transform sizes, in number of 4x4 blocks log 2 ("*_b")
  // 4x4=0, 8x8=2, 16x16=4, 32x32=6, 64x64=8
  const TX_SIZE tx_size = max_txsize_lookup[bsize];
  const int num_4x4_w = num_4x4_blocks_wide_lookup[bsize];
  const int num_4x4_h = num_4x4_blocks_high_lookup[bsize];
  int row, col;
  // If mb_to_right_edge is < 0 we are in a situation in which
  // the current block size extends into the UMV and we won't
  // visit the sub blocks that are wholly within the UMV.
  const int max_blocks_wide =
      num_4x4_w + (xd->mb_to_right_edge >= 0
                       ? 0
                       : xd->mb_to_right_edge >> (5 + pd->subsampling_x));
  const int max_blocks_high =
      num_4x4_h + (xd->mb_to_bottom_edge >= 0
                       ? 0
                       : xd->mb_to_bottom_edge >> (5 + pd->subsampling_y));

  // Keep track of the row and column of the blocks we use so that we know
  // if we are in the unrestricted motion border.
  for (row = 0; row < max_blocks_high; row += (1 << tx_size)) {
    // Skip visiting the sub blocks that are wholly within the UMV.
    for (col = 0; col < max_blocks_wide; col += (1 << tx_size)) {
      p->src.buf = &src_buf_base[4 * (row * (int64_t)src_stride + col)];
      pd->dst.buf = &dst_buf_base[4 * (row * (int64_t)dst_stride + col)];
      vp9_predict_intra_block(xd, b_width_log2_lookup[bsize], tx_size, mode,
                              x->skip_encode ? p->src.buf : pd->dst.buf,
                              x->skip_encode ? src_stride : dst_stride,
                              pd->dst.buf, dst_stride, col, row, 0);
    }
  }
  p->src.buf = src_buf_base;
  pd->dst.buf = dst_buf_base;
}

static void model_rd_for_sb_y_large(VP9_COMP *cpi, BLOCK_SIZE bsize,
                                    MACROBLOCK *x, MACROBLOCKD *xd,
                                    int *out_rate_sum, int64_t *out_dist_sum,
                                    unsigned int *var_y, unsigned int *sse_y,
                                    int mi_row, int mi_col, int *early_term,
                                    int *flag_preduv_computed) {
  // Note our transform coeffs are 8 times an orthogonal transform.
  // Hence quantizer step is also 8 times. To get effective quantizer
  // we need to divide by 8 before sending to modeling function.
  unsigned int sse;
  int rate;
  int64_t dist;
  struct macroblock_plane *const p = &x->plane[0];
  struct macroblockd_plane *const pd = &xd->plane[0];
  const uint32_t dc_quant = pd->dequant[0];
  const uint32_t ac_quant = pd->dequant[1];
  int64_t dc_thr = dc_quant * dc_quant >> 6;
  int64_t ac_thr = ac_quant * ac_quant >> 6;
  unsigned int var;
  int sum;
  int skip_dc = 0;

  const int bw = b_width_log2_lookup[bsize];
  const int bh = b_height_log2_lookup[bsize];
  const int num8x8 = 1 << (bw + bh - 2);
  unsigned int sse8x8[64] = { 0 };
  int sum8x8[64] = { 0 };
  unsigned int var8x8[64] = { 0 };
  TX_SIZE tx_size;
  int i, k;
  uint32_t sum_sqr;
#if CONFIG_VP9_HIGHBITDEPTH
  const vpx_bit_depth_t bd = cpi->common.bit_depth;
#endif
  // Calculate variance for whole partition, and also save 8x8 blocks' variance
  // to be used in following transform skipping test.
  block_variance(p->src.buf, p->src.stride, pd->dst.buf, pd->dst.stride,
                 4 << bw, 4 << bh, &sse, &sum, 8,
#if CONFIG_VP9_HIGHBITDEPTH
                 cpi->common.use_highbitdepth, bd,
#endif
                 sse8x8, sum8x8, var8x8);
  sum_sqr = (uint32_t)((int64_t)sum * sum) >> (bw + bh + 4);
  var = sse > sum_sqr ? sse - sum_sqr : sum_sqr - sse;

  *var_y = var;
  *sse_y = sse;

#if CONFIG_VP9_TEMPORAL_DENOISING
  if (cpi->oxcf.noise_sensitivity > 0 && denoise_svc(cpi) &&
      cpi->oxcf.speed > 5)
    ac_thr = vp9_scale_acskip_thresh(ac_thr, cpi->denoiser.denoising_level,
                                     (abs(sum) >> (bw + bh)),
                                     cpi->svc.temporal_layer_id);
  else
    ac_thr *= ac_thr_factor(cpi->oxcf.speed, cpi->common.width,
                            cpi->common.height, abs(sum) >> (bw + bh));
#else
  ac_thr *= ac_thr_factor(cpi->oxcf.speed, cpi->common.width,
                          cpi->common.height, abs(sum) >> (bw + bh));
#endif

  tx_size = calculate_tx_size(cpi, bsize, xd, var, sse, ac_thr,
                              x->source_variance, 0);
  // The code below for setting skip flag assumes tranform size of at least 8x8,
  // so force this lower limit on transform.
  if (tx_size < TX_8X8) tx_size = TX_8X8;
  xd->mi[0]->tx_size = tx_size;

  if (cpi->oxcf.content == VP9E_CONTENT_SCREEN && x->zero_temp_sad_source &&
      x->source_variance == 0)
    dc_thr = dc_thr << 1;

  // Evaluate if the partition block is a skippable block in Y plane.
  {
    unsigned int sse16x16[16] = { 0 };
    int sum16x16[16] = { 0 };
    unsigned int var16x16[16] = { 0 };
    const int num16x16 = num8x8 >> 2;

    unsigned int sse32x32[4] = { 0 };
    int sum32x32[4] = { 0 };
    unsigned int var32x32[4] = { 0 };
    const int num32x32 = num8x8 >> 4;

    int ac_test = 1;
    int dc_test = 1;
    const int num = (tx_size == TX_8X8)
                        ? num8x8
                        : ((tx_size == TX_16X16) ? num16x16 : num32x32);
    const unsigned int *sse_tx =
        (tx_size == TX_8X8) ? sse8x8
                            : ((tx_size == TX_16X16) ? sse16x16 : sse32x32);
    const unsigned int *var_tx =
        (tx_size == TX_8X8) ? var8x8
                            : ((tx_size == TX_16X16) ? var16x16 : var32x32);

    // Calculate variance if tx_size > TX_8X8
    if (tx_size >= TX_16X16)
      calculate_variance(bw, bh, TX_8X8, sse8x8, sum8x8, var16x16, sse16x16,
                         sum16x16);
    if (tx_size == TX_32X32)
      calculate_variance(bw, bh, TX_16X16, sse16x16, sum16x16, var32x32,
                         sse32x32, sum32x32);

    // Skipping test
    x->skip_txfm[0] = SKIP_TXFM_NONE;
    for (k = 0; k < num; k++)
      // Check if all ac coefficients can be quantized to zero.
      if (!(var_tx[k] < ac_thr || var == 0)) {
        ac_test = 0;
        break;
      }

    for (k = 0; k < num; k++)
      // Check if dc coefficient can be quantized to zero.
      if (!(sse_tx[k] - var_tx[k] < dc_thr || sse == var)) {
        dc_test = 0;
        break;
      }

    if (ac_test) {
      x->skip_txfm[0] = SKIP_TXFM_AC_ONLY;

      if (dc_test) x->skip_txfm[0] = SKIP_TXFM_AC_DC;
    } else if (dc_test) {
      skip_dc = 1;
    }
  }

  if (x->skip_txfm[0] == SKIP_TXFM_AC_DC) {
    int skip_uv[2] = { 0 };
    unsigned int var_uv[2];
    unsigned int sse_uv[2];

    *out_rate_sum = 0;
    *out_dist_sum = sse << 4;

    // Transform skipping test in UV planes.
    for (i = 1; i <= 2; i++) {
      struct macroblock_plane *const p_uv = &x->plane[i];
      struct macroblockd_plane *const pd_uv = &xd->plane[i];
      const TX_SIZE uv_tx_size = get_uv_tx_size(xd->mi[0], pd_uv);
      const BLOCK_SIZE unit_size = txsize_to_bsize[uv_tx_size];
      const BLOCK_SIZE uv_bsize = get_plane_block_size(bsize, pd_uv);
      const int uv_bw = b_width_log2_lookup[uv_bsize];
      const int uv_bh = b_height_log2_lookup[uv_bsize];
      const int sf = (uv_bw - b_width_log2_lookup[unit_size]) +
                     (uv_bh - b_height_log2_lookup[unit_size]);
      const uint32_t uv_dc_thr =
          pd_uv->dequant[0] * pd_uv->dequant[0] >> (6 - sf);
      const uint32_t uv_ac_thr =
          pd_uv->dequant[1] * pd_uv->dequant[1] >> (6 - sf);
      int j = i - 1;

      vp9_build_inter_predictors_sbp(xd, mi_row, mi_col, bsize, i);
      flag_preduv_computed[i - 1] = 1;
      var_uv[j] = cpi->fn_ptr[uv_bsize].vf(p_uv->src.buf, p_uv->src.stride,
                                           pd_uv->dst.buf, pd_uv->dst.stride,
                                           &sse_uv[j]);

      if ((var_uv[j] < uv_ac_thr || var_uv[j] == 0) &&
          (sse_uv[j] - var_uv[j] < uv_dc_thr || sse_uv[j] == var_uv[j]))
        skip_uv[j] = 1;
      else
        break;
    }

    // If the transform in YUV planes are skippable, the mode search checks
    // fewer inter modes and doesn't check intra modes.
    if (skip_uv[0] & skip_uv[1]) {
      *early_term = 1;
    }
    return;
  }

  if (!skip_dc) {
#if CONFIG_VP9_HIGHBITDEPTH
    vp9_model_rd_from_var_lapndz(sse - var, num_pels_log2_lookup[bsize],
                                 dc_quant >> (xd->bd - 5), &rate, &dist);
#else
    vp9_model_rd_from_var_lapndz(sse - var, num_pels_log2_lookup[bsize],
                                 dc_quant >> 3, &rate, &dist);
#endif  // CONFIG_VP9_HIGHBITDEPTH
  }

  if (!skip_dc) {
    *out_rate_sum = rate >> 1;
    *out_dist_sum = dist << 3;
  } else {
    *out_rate_sum = 0;
    *out_dist_sum = (sse - var) << 4;
  }

#if CONFIG_VP9_HIGHBITDEPTH
  vp9_model_rd_from_var_lapndz(var, num_pels_log2_lookup[bsize],
                               ac_quant >> (xd->bd - 5), &rate, &dist);
#else
  vp9_model_rd_from_var_lapndz(var, num_pels_log2_lookup[bsize], ac_quant >> 3,
                               &rate, &dist);
#endif  // CONFIG_VP9_HIGHBITDEPTH

  *out_rate_sum += rate;
  *out_dist_sum += dist << 4;
}

static void model_rd_for_sb_y(VP9_COMP *cpi, BLOCK_SIZE bsize, MACROBLOCK *x,
                              MACROBLOCKD *xd, int *out_rate_sum,
                              int64_t *out_dist_sum, unsigned int *var_y,
                              unsigned int *sse_y, int is_intra) {
  // Note our transform coeffs are 8 times an orthogonal transform.
  // Hence quantizer step is also 8 times. To get effective quantizer
  // we need to divide by 8 before sending to modeling function.
  unsigned int sse;
  int rate;
  int64_t dist;
  struct macroblock_plane *const p = &x->plane[0];
  struct macroblockd_plane *const pd = &xd->plane[0];
  const int64_t dc_thr = p->quant_thred[0] >> 6;
  const int64_t ac_thr = p->quant_thred[1] >> 6;
  const uint32_t dc_quant = pd->dequant[0];
  const uint32_t ac_quant = pd->dequant[1];
  unsigned int var = cpi->fn_ptr[bsize].vf(p->src.buf, p->src.stride,
                                           pd->dst.buf, pd->dst.stride, &sse);
  int skip_dc = 0;

  *var_y = var;
  *sse_y = sse;

  xd->mi[0]->tx_size = calculate_tx_size(cpi, bsize, xd, var, sse, ac_thr,
                                         x->source_variance, is_intra);

  // Evaluate if the partition block is a skippable block in Y plane.
  {
    const BLOCK_SIZE unit_size = txsize_to_bsize[xd->mi[0]->tx_size];
    const unsigned int num_blk_log2 =
        (b_width_log2_lookup[bsize] - b_width_log2_lookup[unit_size]) +
        (b_height_log2_lookup[bsize] - b_height_log2_lookup[unit_size]);
    const unsigned int sse_tx = sse >> num_blk_log2;
    const unsigned int var_tx = var >> num_blk_log2;

    x->skip_txfm[0] = SKIP_TXFM_NONE;
    // Check if all ac coefficients can be quantized to zero.
    if (var_tx < ac_thr || var == 0) {
      x->skip_txfm[0] = SKIP_TXFM_AC_ONLY;
      // Check if dc coefficient can be quantized to zero.
      if (sse_tx - var_tx < dc_thr || sse == var)
        x->skip_txfm[0] = SKIP_TXFM_AC_DC;
    } else {
      if (sse_tx - var_tx < dc_thr || sse == var) skip_dc = 1;
    }
  }

  if (x->skip_txfm[0] == SKIP_TXFM_AC_DC) {
    *out_rate_sum = 0;
    *out_dist_sum = sse << 4;
    return;
  }

  if (!skip_dc) {
#if CONFIG_VP9_HIGHBITDEPTH
    vp9_model_rd_from_var_lapndz(sse - var, num_pels_log2_lookup[bsize],
                                 dc_quant >> (xd->bd - 5), &rate, &dist);
#else
    vp9_model_rd_from_var_lapndz(sse - var, num_pels_log2_lookup[bsize],
                                 dc_quant >> 3, &rate, &dist);
#endif  // CONFIG_VP9_HIGHBITDEPTH
  }

  if (!skip_dc) {
    *out_rate_sum = rate >> 1;
    *out_dist_sum = dist << 3;
  } else {
    *out_rate_sum = 0;
    *out_dist_sum = (sse - var) << 4;
  }

#if CONFIG_VP9_HIGHBITDEPTH
  vp9_model_rd_from_var_lapndz(var, num_pels_log2_lookup[bsize],
                               ac_quant >> (xd->bd - 5), &rate, &dist);
#else
  vp9_model_rd_from_var_lapndz(var, num_pels_log2_lookup[bsize], ac_quant >> 3,
                               &rate, &dist);
#endif  // CONFIG_VP9_HIGHBITDEPTH

  *out_rate_sum += rate;
  *out_dist_sum += dist << 4;
}

static void block_yrd(VP9_COMP *cpi, MACROBLOCK *x, RD_COST *this_rdc,
                      int *skippable, int64_t *sse, BLOCK_SIZE bsize,
                      TX_SIZE tx_size, int rd_computed, int is_intra) {
  MACROBLOCKD *xd = &x->e_mbd;
  const struct macroblockd_plane *pd = &xd->plane[0];
  struct macroblock_plane *const p = &x->plane[0];
  const int num_4x4_w = num_4x4_blocks_wide_lookup[bsize];
  const int num_4x4_h = num_4x4_blocks_high_lookup[bsize];
  const int step = 1 << (tx_size << 1);
  const int block_step = (1 << tx_size);
  int block = 0, r, c;
  const int max_blocks_wide =
      num_4x4_w + (xd->mb_to_right_edge >= 0 ? 0 : xd->mb_to_right_edge >> 5);
  const int max_blocks_high =
      num_4x4_h + (xd->mb_to_bottom_edge >= 0 ? 0 : xd->mb_to_bottom_edge >> 5);
  int eob_cost = 0;
  const int bw = 4 * num_4x4_w;
  const int bh = 4 * num_4x4_h;

  if (cpi->sf.use_simple_block_yrd && cpi->common.frame_type != KEY_FRAME &&
      (bsize < BLOCK_32X32 ||
       (cpi->use_svc &&
        (bsize < BLOCK_32X32 || cpi->svc.temporal_layer_id > 0)))) {
    unsigned int var_y, sse_y;
    (void)tx_size;
    if (!rd_computed)
      model_rd_for_sb_y(cpi, bsize, x, xd, &this_rdc->rate, &this_rdc->dist,
                        &var_y, &sse_y, is_intra);
    *sse = INT_MAX;
    *skippable = 0;
    return;
  }

  (void)cpi;

  // The max tx_size passed in is TX_16X16.
  assert(tx_size != TX_32X32);
#if CONFIG_VP9_HIGHBITDEPTH
  if (xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH) {
    vpx_highbd_subtract_block(bh, bw, p->src_diff, bw, p->src.buf,
                              p->src.stride, pd->dst.buf, pd->dst.stride,
                              x->e_mbd.bd);
  } else {
    vpx_subtract_block(bh, bw, p->src_diff, bw, p->src.buf, p->src.stride,
                       pd->dst.buf, pd->dst.stride);
  }
#else
  vpx_subtract_block(bh, bw, p->src_diff, bw, p->src.buf, p->src.stride,
                     pd->dst.buf, pd->dst.stride);
#endif
  *skippable = 1;
  // Keep track of the row and column of the blocks we use so that we know
  // if we are in the unrestricted motion border.
  for (r = 0; r < max_blocks_high; r += block_step) {
    for (c = 0; c < num_4x4_w; c += block_step) {
      if (c < max_blocks_wide) {
        const ScanOrder *const scan_order = &vp9_default_scan_orders[tx_size];
        tran_low_t *const coeff = BLOCK_OFFSET(p->coeff, block);
        tran_low_t *const qcoeff = BLOCK_OFFSET(p->qcoeff, block);
        tran_low_t *const dqcoeff = BLOCK_OFFSET(pd->dqcoeff, block);
        uint16_t *const eob = &p->eobs[block];
        const int diff_stride = bw;
        const int16_t *src_diff;
        src_diff = &p->src_diff[(r * diff_stride + c) << 2];

        // skip block condition should be handled before this is called.
        assert(!x->skip_block);

        switch (tx_size) {
          case TX_16X16:
            vpx_hadamard_16x16(src_diff, diff_stride, coeff);
            vp9_quantize_fp(coeff, 256, p, qcoeff, dqcoeff, pd->dequant, eob,
                            scan_order);
            break;
          case TX_8X8:
            vpx_hadamard_8x8(src_diff, diff_stride, coeff);
            vp9_quantize_fp(coeff, 64, p, qcoeff, dqcoeff, pd->dequant, eob,
                            scan_order);
            break;
          default:
            assert(tx_size == TX_4X4);
            x->fwd_txfm4x4(src_diff, coeff, diff_stride);
            vp9_quantize_fp(coeff, 16, p, qcoeff, dqcoeff, pd->dequant, eob,
                            scan_order);
            break;
        }
        *skippable &= (*eob == 0);
        eob_cost += 1;
      }
      block += step;
    }
  }

  this_rdc->rate = 0;
  if (*sse < INT64_MAX) {
    *sse = (*sse << 6) >> 2;
    if (*skippable) {
      this_rdc->dist = *sse;
      return;
    }
  }

  block = 0;
  this_rdc->dist = 0;
  for (r = 0; r < max_blocks_high; r += block_step) {
    for (c = 0; c < num_4x4_w; c += block_step) {
      if (c < max_blocks_wide) {
        tran_low_t *const coeff = BLOCK_OFFSET(p->coeff, block);
        tran_low_t *const qcoeff = BLOCK_OFFSET(p->qcoeff, block);
        tran_low_t *const dqcoeff = BLOCK_OFFSET(pd->dqcoeff, block);
        uint16_t *const eob = &p->eobs[block];

        if (*eob == 1)
          this_rdc->rate += (int)abs(qcoeff[0]);
        else if (*eob > 1)
          this_rdc->rate += vpx_satd(qcoeff, step << 4);

        this_rdc->dist += vp9_block_error_fp(coeff, dqcoeff, step << 4) >> 2;
      }
      block += step;
    }
  }

  // If skippable is set, rate gets clobbered later.
  this_rdc->rate <<= (2 + VP9_PROB_COST_SHIFT);
  this_rdc->rate += (eob_cost << VP9_PROB_COST_SHIFT);
}

static void model_rd_for_sb_uv(VP9_COMP *cpi, BLOCK_SIZE plane_bsize,
                               MACROBLOCK *x, MACROBLOCKD *xd,
                               RD_COST *this_rdc, unsigned int *var_y,
                               unsigned int *sse_y, int start_plane,
                               int stop_plane) {
  // Note our transform coeffs are 8 times an orthogonal transform.
  // Hence quantizer step is also 8 times. To get effective quantizer
  // we need to divide by 8 before sending to modeling function.
  unsigned int sse;
  int rate;
  int64_t dist;
  int i;
#if CONFIG_VP9_HIGHBITDEPTH
  uint64_t tot_var = *var_y;
  uint64_t tot_sse = *sse_y;
#else
  uint32_t tot_var = *var_y;
  uint32_t tot_sse = *sse_y;
#endif

  this_rdc->rate = 0;
  this_rdc->dist = 0;

  for (i = start_plane; i <= stop_plane; ++i) {
    struct macroblock_plane *const p = &x->plane[i];
    struct macroblockd_plane *const pd = &xd->plane[i];
    const uint32_t dc_quant = pd->dequant[0];
    const uint32_t ac_quant = pd->dequant[1];
    const BLOCK_SIZE bs = plane_bsize;
    unsigned int var;
    if (!x->color_sensitivity[i - 1]) continue;

    var = cpi->fn_ptr[bs].vf(p->src.buf, p->src.stride, pd->dst.buf,
                             pd->dst.stride, &sse);
    assert(sse >= var);
    tot_var += var;
    tot_sse += sse;

#if CONFIG_VP9_HIGHBITDEPTH
    vp9_model_rd_from_var_lapndz(sse - var, num_pels_log2_lookup[bs],
                                 dc_quant >> (xd->bd - 5), &rate, &dist);
#else
    vp9_model_rd_from_var_lapndz(sse - var, num_pels_log2_lookup[bs],
                                 dc_quant >> 3, &rate, &dist);
#endif  // CONFIG_VP9_HIGHBITDEPTH

    this_rdc->rate += rate >> 1;
    this_rdc->dist += dist << 3;

#if CONFIG_VP9_HIGHBITDEPTH
    vp9_model_rd_from_var_lapndz(var, num_pels_log2_lookup[bs],
                                 ac_quant >> (xd->bd - 5), &rate, &dist);
#else
    vp9_model_rd_from_var_lapndz(var, num_pels_log2_lookup[bs], ac_quant >> 3,
                                 &rate, &dist);
#endif  // CONFIG_VP9_HIGHBITDEPTH

    this_rdc->rate += rate;
    this_rdc->dist += dist << 4;
  }

#if CONFIG_VP9_HIGHBITDEPTH
  *var_y = tot_var > UINT32_MAX ? UINT32_MAX : (uint32_t)tot_var;
  *sse_y = tot_sse > UINT32_MAX ? UINT32_MAX : (uint32_t)tot_sse;
#else
  *var_y = tot_var;
  *sse_y = tot_sse;
#endif
}

static int get_pred_buffer(PRED_BUFFER *p, int len) {
  int i;

  for (i = 0; i < len; i++) {
    if (!p[i].in_use) {
      p[i].in_use = 1;
      return i;
    }
  }
  return -1;
}

static void free_pred_buffer(PRED_BUFFER *p) {
  if (p != NULL) p->in_use = 0;
}

static void encode_breakout_test(
    VP9_COMP *cpi, MACROBLOCK *x, BLOCK_SIZE bsize, int mi_row, int mi_col,
    MV_REFERENCE_FRAME ref_frame, PREDICTION_MODE this_mode, unsigned int var_y,
    unsigned int sse_y, struct buf_2d yv12_mb[][MAX_MB_PLANE], int *rate,
    int64_t *dist, int *flag_preduv_computed) {
  MACROBLOCKD *xd = &x->e_mbd;
  MODE_INFO *const mi = xd->mi[0];
  const BLOCK_SIZE uv_size = get_plane_block_size(bsize, &xd->plane[1]);
  unsigned int var = var_y, sse = sse_y;
  // Skipping threshold for ac.
  unsigned int thresh_ac;
  // Skipping threshold for dc.
  unsigned int thresh_dc;
  int motion_low = 1;

  if (cpi->use_svc && ref_frame == GOLDEN_FRAME) return;
  if (mi->mv[0].as_mv.row > 64 || mi->mv[0].as_mv.row < -64 ||
      mi->mv[0].as_mv.col > 64 || mi->mv[0].as_mv.col < -64)
    motion_low = 0;
  if (x->encode_breakout > 0 && motion_low == 1) {
    // Set a maximum for threshold to avoid big PSNR loss in low bit rate
    // case. Use extreme low threshold for static frames to limit
    // skipping.
    const unsigned int max_thresh = 36000;
    // The encode_breakout input
    const unsigned int min_thresh =
        VPXMIN(((unsigned int)x->encode_breakout << 4), max_thresh);
#if CONFIG_VP9_HIGHBITDEPTH
    const int shift = (xd->bd << 1) - 16;
#endif

    // Calculate threshold according to dequant value.
    thresh_ac = (xd->plane[0].dequant[1] * xd->plane[0].dequant[1]) >> 3;
#if CONFIG_VP9_HIGHBITDEPTH
    if ((xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH) && shift > 0) {
      thresh_ac = ROUND_POWER_OF_TWO(thresh_ac, shift);
    }
#endif  // CONFIG_VP9_HIGHBITDEPTH
    thresh_ac = clamp(thresh_ac, min_thresh, max_thresh);

    // Adjust ac threshold according to partition size.
    thresh_ac >>=
        8 - (b_width_log2_lookup[bsize] + b_height_log2_lookup[bsize]);

    thresh_dc = (xd->plane[0].dequant[0] * xd->plane[0].dequant[0] >> 6);
#if CONFIG_VP9_HIGHBITDEPTH
    if ((xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH) && shift > 0) {
      thresh_dc = ROUND_POWER_OF_TWO(thresh_dc, shift);
    }
#endif  // CONFIG_VP9_HIGHBITDEPTH
  } else {
    thresh_ac = 0;
    thresh_dc = 0;
  }

  // Y skipping condition checking for ac and dc.
  if (var <= thresh_ac && (sse - var) <= thresh_dc) {
    unsigned int sse_u, sse_v;
    unsigned int var_u, var_v;
    unsigned int thresh_ac_uv = thresh_ac;
    unsigned int thresh_dc_uv = thresh_dc;
    if (x->sb_is_skin) {
      thresh_ac_uv = 0;
      thresh_dc_uv = 0;
    }

    if (!flag_preduv_computed[0] || !flag_preduv_computed[1]) {
      xd->plane[1].pre[0] = yv12_mb[ref_frame][1];
      xd->plane[2].pre[0] = yv12_mb[ref_frame][2];
      vp9_build_inter_predictors_sbuv(xd, mi_row, mi_col, bsize);
    }

    var_u = cpi->fn_ptr[uv_size].vf(x->plane[1].src.buf, x->plane[1].src.stride,
                                    xd->plane[1].dst.buf,
                                    xd->plane[1].dst.stride, &sse_u);

    // U skipping condition checking
    if (((var_u << 2) <= thresh_ac_uv) && (sse_u - var_u <= thresh_dc_uv)) {
      var_v = cpi->fn_ptr[uv_size].vf(
          x->plane[2].src.buf, x->plane[2].src.stride, xd->plane[2].dst.buf,
          xd->plane[2].dst.stride, &sse_v);

      // V skipping condition checking
      if (((var_v << 2) <= thresh_ac_uv) && (sse_v - var_v <= thresh_dc_uv)) {
        x->skip = 1;

        // The cost of skip bit needs to be added.
        *rate = cpi->inter_mode_cost[x->mbmi_ext->mode_context[ref_frame]]
                                    [INTER_OFFSET(this_mode)];

        // More on this part of rate
        // rate += vp9_cost_bit(vp9_get_skip_prob(cm, xd), 1);

        // Scaling factor for SSE from spatial domain to frequency
        // domain is 16. Adjust distortion accordingly.
        // TODO(yunqingwang): In this function, only y-plane dist is
        // calculated.
        *dist = (sse << 4);  // + ((sse_u + sse_v) << 4);

        // *disable_skip = 1;
      }
    }
  }
}

struct estimate_block_intra_args {
  VP9_COMP *cpi;
  MACROBLOCK *x;
  PREDICTION_MODE mode;
  int skippable;
  RD_COST *rdc;
};

static void estimate_block_intra(int plane, int block, int row, int col,
                                 BLOCK_SIZE plane_bsize, TX_SIZE tx_size,
                                 void *arg) {
  struct estimate_block_intra_args *const args = arg;
  VP9_COMP *const cpi = args->cpi;
  MACROBLOCK *const x = args->x;
  MACROBLOCKD *const xd = &x->e_mbd;
  struct macroblock_plane *const p = &x->plane[plane];
  struct macroblockd_plane *const pd = &xd->plane[plane];
  const BLOCK_SIZE bsize_tx = txsize_to_bsize[tx_size];
  uint8_t *const src_buf_base = p->src.buf;
  uint8_t *const dst_buf_base = pd->dst.buf;
  const int src_stride = p->src.stride;
  const int dst_stride = pd->dst.stride;
  RD_COST this_rdc;

  (void)block;

  p->src.buf = &src_buf_base[4 * (row * (int64_t)src_stride + col)];
  pd->dst.buf = &dst_buf_base[4 * (row * (int64_t)dst_stride + col)];
  // Use source buffer as an approximation for the fully reconstructed buffer.
  vp9_predict_intra_block(xd, b_width_log2_lookup[plane_bsize], tx_size,
                          args->mode, x->skip_encode ? p->src.buf : pd->dst.buf,
                          x->skip_encode ? src_stride : dst_stride, pd->dst.buf,
                          dst_stride, col, row, plane);

  if (plane == 0) {
    int64_t this_sse = INT64_MAX;
    block_yrd(cpi, x, &this_rdc, &args->skippable, &this_sse, bsize_tx,
              VPXMIN(tx_size, TX_16X16), 0, 1);
  } else {
    unsigned int var = 0;
    unsigned int sse = 0;
    model_rd_for_sb_uv(cpi, bsize_tx, x, xd, &this_rdc, &var, &sse, plane,
                       plane);
  }

  p->src.buf = src_buf_base;
  pd->dst.buf = dst_buf_base;
  args->rdc->rate += this_rdc.rate;
  args->rdc->dist += this_rdc.dist;
}

static const THR_MODES mode_idx[MAX_REF_FRAMES][4] = {
  { THR_DC, THR_V_PRED, THR_H_PRED, THR_TM },
  { THR_NEARESTMV, THR_NEARMV, THR_ZEROMV, THR_NEWMV },
  { THR_NEARESTG, THR_NEARG, THR_ZEROG, THR_NEWG },
  { THR_NEARESTA, THR_NEARA, THR_ZEROA, THR_NEWA },
};

static const PREDICTION_MODE intra_mode_list[] = { DC_PRED, V_PRED, H_PRED,
                                                   TM_PRED };

static int mode_offset(const PREDICTION_MODE mode) {
  if (mode >= NEARESTMV) {
    return INTER_OFFSET(mode);
  } else {
    switch (mode) {
      case DC_PRED: return 0;
      case V_PRED: return 1;
      case H_PRED: return 2;
      case TM_PRED: return 3;
      default: return -1;
    }
  }
}

static INLINE int rd_less_than_thresh_row_mt(int64_t best_rd, int thresh,
                                             const int *const thresh_fact) {
  int is_rd_less_than_thresh;
  is_rd_less_than_thresh =
      best_rd < ((int64_t)thresh * (*thresh_fact) >> 5) || thresh == INT_MAX;
  return is_rd_less_than_thresh;
}

static INLINE void update_thresh_freq_fact_row_mt(
    VP9_COMP *cpi, TileDataEnc *tile_data, unsigned int source_variance,
    int thresh_freq_fact_idx, MV_REFERENCE_FRAME ref_frame,
    THR_MODES best_mode_idx, PREDICTION_MODE mode) {
  THR_MODES thr_mode_idx = mode_idx[ref_frame][mode_offset(mode)];
  int freq_fact_idx = thresh_freq_fact_idx + thr_mode_idx;
  int *freq_fact = &tile_data->row_base_thresh_freq_fact[freq_fact_idx];
  if (thr_mode_idx == best_mode_idx)
    *freq_fact -= (*freq_fact >> 4);
  else if (cpi->sf.limit_newmv_early_exit && mode == NEWMV &&
           ref_frame == LAST_FRAME && source_variance < 5) {
    *freq_fact = VPXMIN(*freq_fact + RD_THRESH_INC, 32);
  } else {
    *freq_fact = VPXMIN(*freq_fact + RD_THRESH_INC,
                        cpi->sf.adaptive_rd_thresh * RD_THRESH_MAX_FACT);
  }
}

static INLINE void update_thresh_freq_fact(
    VP9_COMP *cpi, TileDataEnc *tile_data, unsigned int source_variance,
    BLOCK_SIZE bsize, MV_REFERENCE_FRAME ref_frame, THR_MODES best_mode_idx,
    PREDICTION_MODE mode) {
  THR_MODES thr_mode_idx = mode_idx[ref_frame][mode_offset(mode)];
  int *freq_fact = &tile_data->thresh_freq_fact[bsize][thr_mode_idx];
  if (thr_mode_idx == best_mode_idx)
    *freq_fact -= (*freq_fact >> 4);
  else if (cpi->sf.limit_newmv_early_exit && mode == NEWMV &&
           ref_frame == LAST_FRAME && source_variance < 5) {
    *freq_fact = VPXMIN(*freq_fact + RD_THRESH_INC, 32);
  } else {
    *freq_fact = VPXMIN(*freq_fact + RD_THRESH_INC,
                        cpi->sf.adaptive_rd_thresh * RD_THRESH_MAX_FACT);
  }
}

void vp9_pick_intra_mode(VP9_COMP *cpi, MACROBLOCK *x, RD_COST *rd_cost,
                         BLOCK_SIZE bsize, PICK_MODE_CONTEXT *ctx) {
  MACROBLOCKD *const xd = &x->e_mbd;
  MODE_INFO *const mi = xd->mi[0];
  RD_COST this_rdc, best_rdc;
  PREDICTION_MODE this_mode;
  struct estimate_block_intra_args args = { cpi, x, DC_PRED, 1, 0 };
  const TX_SIZE intra_tx_size =
      VPXMIN(max_txsize_lookup[bsize],
             tx_mode_to_biggest_tx_size[cpi->common.tx_mode]);
  MODE_INFO *const mic = xd->mi[0];
  int *bmode_costs;
  const MODE_INFO *above_mi = xd->above_mi;
  const MODE_INFO *left_mi = xd->left_mi;
  const PREDICTION_MODE A = vp9_above_block_mode(mic, above_mi, 0);
  const PREDICTION_MODE L = vp9_left_block_mode(mic, left_mi, 0);
  bmode_costs = cpi->y_mode_costs[A][L];
  assert(bsize >= BLOCK_8X8);

  (void)ctx;
  vp9_rd_cost_reset(&best_rdc);
  vp9_rd_cost_reset(&this_rdc);

  mi->ref_frame[0] = INTRA_FRAME;
  // Initialize interp_filter here so we do not have to check for inter block
  // modes in get_pred_context_switchable_interp()
  mi->interp_filter = SWITCHABLE_FILTERS;

  mi->mv[0].as_int = INVALID_MV;
  mi->uv_mode = DC_PRED;
  memset(x->skip_txfm, 0, sizeof(x->skip_txfm));

  // Change the limit of this loop to add other intra prediction
  // mode tests.
  for (this_mode = DC_PRED; this_mode <= H_PRED; ++this_mode) {
    this_rdc.dist = this_rdc.rate = 0;
    args.mode = this_mode;
    args.skippable = 1;
    args.rdc = &this_rdc;
    mi->tx_size = intra_tx_size;
    vp9_foreach_transformed_block_in_plane(xd, bsize, 0, estimate_block_intra,
                                           &args);
    if (args.skippable) {
      x->skip_txfm[0] = SKIP_TXFM_AC_DC;
      this_rdc.rate = vp9_cost_bit(vp9_get_skip_prob(&cpi->common, xd), 1);
    } else {
      x->skip_txfm[0] = SKIP_TXFM_NONE;
      this_rdc.rate += vp9_cost_bit(vp9_get_skip_prob(&cpi->common, xd), 0);
    }
    this_rdc.rate += bmode_costs[this_mode];
    this_rdc.rdcost = RDCOST(x->rdmult, x->rddiv, this_rdc.rate, this_rdc.dist);

    if (this_rdc.rdcost < best_rdc.rdcost) {
      best_rdc = this_rdc;
      mi->mode = this_mode;
    }
  }

  *rd_cost = best_rdc;
}

static void init_ref_frame_cost(VP9_COMMON *const cm, MACROBLOCKD *const xd,
                                int ref_frame_cost[MAX_REF_FRAMES]) {
  vpx_prob intra_inter_p = vp9_get_intra_inter_prob(cm, xd);
  vpx_prob ref_single_p1 = vp9_get_pred_prob_single_ref_p1(cm, xd);
  vpx_prob ref_single_p2 = vp9_get_pred_prob_single_ref_p2(cm, xd);

  ref_frame_cost[INTRA_FRAME] = vp9_cost_bit(intra_inter_p, 0);
  ref_frame_cost[LAST_FRAME] = ref_frame_cost[GOLDEN_FRAME] =
      ref_frame_cost[ALTREF_FRAME] = vp9_cost_bit(intra_inter_p, 1);

  ref_frame_cost[LAST_FRAME] += vp9_cost_bit(ref_single_p1, 0);
  ref_frame_cost[GOLDEN_FRAME] += vp9_cost_bit(ref_single_p1, 1);
  ref_frame_cost[ALTREF_FRAME] += vp9_cost_bit(ref_single_p1, 1);
  ref_frame_cost[GOLDEN_FRAME] += vp9_cost_bit(ref_single_p2, 0);
  ref_frame_cost[ALTREF_FRAME] += vp9_cost_bit(ref_single_p2, 1);
}

typedef struct {
  MV_REFERENCE_FRAME ref_frame;
  PREDICTION_MODE pred_mode;
} REF_MODE;

#define RT_INTER_MODES 12
static const REF_MODE ref_mode_set[RT_INTER_MODES] = {
  { LAST_FRAME, ZEROMV },   { LAST_FRAME, NEARESTMV },
  { GOLDEN_FRAME, ZEROMV }, { LAST_FRAME, NEARMV },
  { LAST_FRAME, NEWMV },    { GOLDEN_FRAME, NEARESTMV },
  { GOLDEN_FRAME, NEARMV }, { GOLDEN_FRAME, NEWMV },
  { ALTREF_FRAME, ZEROMV }, { ALTREF_FRAME, NEARESTMV },
  { ALTREF_FRAME, NEARMV }, { ALTREF_FRAME, NEWMV }
};

#define RT_INTER_MODES_SVC 8
static const REF_MODE ref_mode_set_svc[RT_INTER_MODES_SVC] = {
  { LAST_FRAME, ZEROMV },      { LAST_FRAME, NEARESTMV },
  { LAST_FRAME, NEARMV },      { GOLDEN_FRAME, ZEROMV },
  { GOLDEN_FRAME, NEARESTMV }, { GOLDEN_FRAME, NEARMV },
  { LAST_FRAME, NEWMV },       { GOLDEN_FRAME, NEWMV }
};

static INLINE void find_predictors(
    VP9_COMP *cpi, MACROBLOCK *x, MV_REFERENCE_FRAME ref_frame,
    int_mv frame_mv[MB_MODE_COUNT][MAX_REF_FRAMES],
    int const_motion[MAX_REF_FRAMES], int *ref_frame_skip_mask,
    TileDataEnc *tile_data, int mi_row, int mi_col,
    struct buf_2d yv12_mb[4][MAX_MB_PLANE], BLOCK_SIZE bsize,
    int force_skip_low_temp_var, int comp_pred_allowed) {
  VP9_COMMON *const cm = &cpi->common;
  MACROBLOCKD *const xd = &x->e_mbd;
  const YV12_BUFFER_CONFIG *yv12 = get_ref_frame_buffer(cpi, ref_frame);
  TileInfo *const tile_info = &tile_data->tile_info;
  // TODO(jingning) placeholder for inter-frame non-RD mode decision.
  x->pred_mv_sad[ref_frame] = INT_MAX;
  frame_mv[NEWMV][ref_frame].as_int = INVALID_MV;
  frame_mv[ZEROMV][ref_frame].as_int = 0;
  // this needs various further optimizations. to be continued..
  if ((cpi->ref_frame_flags & ref_frame_to_flag(ref_frame)) && (yv12 != NULL)) {
    int_mv *const candidates = x->mbmi_ext->ref_mvs[ref_frame];
    const struct scale_factors *const sf = &cm->frame_refs[ref_frame - 1].sf;
    vp9_setup_pred_block(xd, yv12_mb[ref_frame], yv12, mi_row, mi_col, sf, sf);
    if (cm->use_prev_frame_mvs || comp_pred_allowed) {
      vp9_find_mv_refs(cm, xd, xd->mi[0], ref_frame, candidates, mi_row, mi_col,
                       x->mbmi_ext->mode_context);
    } else {
      const_motion[ref_frame] =
          mv_refs_rt(cpi, cm, x, xd, tile_info, xd->mi[0], ref_frame,
                     candidates, &frame_mv[NEWMV][ref_frame], mi_row, mi_col,
                     (int)(cpi->svc.use_base_mv && cpi->svc.spatial_layer_id));
    }
    vp9_find_best_ref_mvs(xd, cm->allow_high_precision_mv, candidates,
                          &frame_mv[NEARESTMV][ref_frame],
                          &frame_mv[NEARMV][ref_frame]);
    // Early exit for golden frame if force_skip_low_temp_var is set.
    if (!vp9_is_scaled(sf) && bsize >= BLOCK_8X8 &&
        !(force_skip_low_temp_var && ref_frame == GOLDEN_FRAME)) {
      vp9_mv_pred(cpi, x, yv12_mb[ref_frame][0].buf, yv12->y_stride, ref_frame,
                  bsize);
    }
  } else {
    *ref_frame_skip_mask |= (1 << ref_frame);
  }
}

static void vp9_NEWMV_diff_bias(const NOISE_ESTIMATE *ne, MACROBLOCKD *xd,
                                PREDICTION_MODE this_mode, RD_COST *this_rdc,
                                BLOCK_SIZE bsize, int mv_row, int mv_col,
                                int is_last_frame, int lowvar_highsumdiff,
                                int is_skin) {
  // Bias against MVs associated with NEWMV mode that are very different from
  // top/left neighbors.
  if (this_mode == NEWMV) {
    int al_mv_average_row;
    int al_mv_average_col;
    int left_row, left_col;
    int row_diff, col_diff;
    int above_mv_valid = 0;
    int left_mv_valid = 0;
    int above_row = 0;
    int above_col = 0;

    if (xd->above_mi) {
      above_mv_valid = xd->above_mi->mv[0].as_int != INVALID_MV;
      above_row = xd->above_mi->mv[0].as_mv.row;
      above_col = xd->above_mi->mv[0].as_mv.col;
    }
    if (xd->left_mi) {
      left_mv_valid = xd->left_mi->mv[0].as_int != INVALID_MV;
      left_row = xd->left_mi->mv[0].as_mv.row;
      left_col = xd->left_mi->mv[0].as_mv.col;
    }
    if (above_mv_valid && left_mv_valid) {
      al_mv_average_row = (above_row + left_row + 1) >> 1;
      al_mv_average_col = (above_col + left_col + 1) >> 1;
    } else if (above_mv_valid) {
      al_mv_average_row = above_row;
      al_mv_average_col = above_col;
    } else if (left_mv_valid) {
      al_mv_average_row = left_row;
      al_mv_average_col = left_col;
    } else {
      al_mv_average_row = al_mv_average_col = 0;
    }
    row_diff = (al_mv_average_row - mv_row);
    col_diff = (al_mv_average_col - mv_col);
    if (row_diff > 48 || row_diff < -48 || col_diff > 48 || col_diff < -48) {
      if (bsize > BLOCK_32X32)
        this_rdc->rdcost = this_rdc->rdcost << 1;
      else
        this_rdc->rdcost = 3 * this_rdc->rdcost >> 1;
    }
  }
  // If noise estimation is enabled, and estimated level is above threshold,
  // add a bias to LAST reference with small motion, for large blocks.
  if (ne->enabled && ne->level >= kMedium && bsize >= BLOCK_32X32 &&
      is_last_frame && mv_row < 8 && mv_row > -8 && mv_col < 8 && mv_col > -8)
    this_rdc->rdcost = 7 * (this_rdc->rdcost >> 3);
  else if (lowvar_highsumdiff && !is_skin && bsize >= BLOCK_16X16 &&
           is_last_frame && mv_row < 16 && mv_row > -16 && mv_col < 16 &&
           mv_col > -16)
    this_rdc->rdcost = 7 * (this_rdc->rdcost >> 3);
}

#if CONFIG_VP9_TEMPORAL_DENOISING
static void vp9_pickmode_ctx_den_update(
    VP9_PICKMODE_CTX_DEN *ctx_den, int64_t zero_last_cost_orig,
    int ref_frame_cost[MAX_REF_FRAMES],
    int_mv frame_mv[MB_MODE_COUNT][MAX_REF_FRAMES], int reuse_inter_pred,
    BEST_PICKMODE *bp) {
  ctx_den->zero_last_cost_orig = zero_last_cost_orig;
  ctx_den->ref_frame_cost = ref_frame_cost;
  ctx_den->frame_mv = frame_mv;
  ctx_den->reuse_inter_pred = reuse_inter_pred;
  ctx_den->best_tx_size = bp->best_tx_size;
  ctx_den->best_mode = bp->best_mode;
  ctx_den->best_ref_frame = bp->best_ref_frame;
  ctx_den->best_pred_filter = bp->best_pred_filter;
  ctx_den->best_mode_skip_txfm = bp->best_mode_skip_txfm;
}

static void recheck_zeromv_after_denoising(
    VP9_COMP *cpi, MODE_INFO *const mi, MACROBLOCK *x, MACROBLOCKD *const xd,
    VP9_DENOISER_DECISION decision, VP9_PICKMODE_CTX_DEN *ctx_den,
    struct buf_2d yv12_mb[4][MAX_MB_PLANE], RD_COST *best_rdc, BLOCK_SIZE bsize,
    int mi_row, int mi_col) {
  // If INTRA or GOLDEN reference was selected, re-evaluate ZEROMV on
  // denoised result. Only do this under noise conditions, and if rdcost of
  // ZEROMV onoriginal source is not significantly higher than rdcost of best
  // mode.
  if (cpi->noise_estimate.enabled && cpi->noise_estimate.level > kLow &&
      ctx_den->zero_last_cost_orig < (best_rdc->rdcost << 3) &&
      ((ctx_den->best_ref_frame == INTRA_FRAME && decision >= FILTER_BLOCK) ||
       (ctx_den->best_ref_frame == GOLDEN_FRAME &&
        cpi->svc.number_spatial_layers == 1 &&
        decision == FILTER_ZEROMV_BLOCK))) {
    // Check if we should pick ZEROMV on denoised signal.
    VP9_COMMON *const cm = &cpi->common;
    int rate = 0;
    int64_t dist = 0;
    uint32_t var_y = UINT_MAX;
    uint32_t sse_y = UINT_MAX;
    RD_COST this_rdc;
    mi->mode = ZEROMV;
    mi->ref_frame[0] = LAST_FRAME;
    mi->ref_frame[1] = NO_REF_FRAME;
    set_ref_ptrs(cm, xd, mi->ref_frame[0], NO_REF_FRAME);
    mi->mv[0].as_int = 0;
    mi->interp_filter = EIGHTTAP;
    if (cpi->sf.default_interp_filter == BILINEAR) mi->interp_filter = BILINEAR;
    xd->plane[0].pre[0] = yv12_mb[LAST_FRAME][0];
    vp9_build_inter_predictors_sby(xd, mi_row, mi_col, bsize);
    model_rd_for_sb_y(cpi, bsize, x, xd, &rate, &dist, &var_y, &sse_y, 0);
    this_rdc.rate = rate + ctx_den->ref_frame_cost[LAST_FRAME] +
                    cpi->inter_mode_cost[x->mbmi_ext->mode_context[LAST_FRAME]]
                                        [INTER_OFFSET(ZEROMV)];
    this_rdc.dist = dist;
    this_rdc.rdcost = RDCOST(x->rdmult, x->rddiv, rate, dist);
    // Don't switch to ZEROMV if the rdcost for ZEROMV on denoised source
    // is higher than best_ref mode (on original source).
    if (this_rdc.rdcost > best_rdc->rdcost) {
      this_rdc = *best_rdc;
      mi->mode = ctx_den->best_mode;
      mi->ref_frame[0] = ctx_den->best_ref_frame;
      set_ref_ptrs(cm, xd, mi->ref_frame[0], NO_REF_FRAME);
      mi->interp_filter = ctx_den->best_pred_filter;
      if (ctx_den->best_ref_frame == INTRA_FRAME) {
        mi->mv[0].as_int = INVALID_MV;
        mi->interp_filter = SWITCHABLE_FILTERS;
      } else if (ctx_den->best_ref_frame == GOLDEN_FRAME) {
        mi->mv[0].as_int =
            ctx_den->frame_mv[ctx_den->best_mode][ctx_den->best_ref_frame]
                .as_int;
        if (ctx_den->reuse_inter_pred) {
          xd->plane[0].pre[0] = yv12_mb[GOLDEN_FRAME][0];
          vp9_build_inter_predictors_sby(xd, mi_row, mi_col, bsize);
        }
      }
      mi->tx_size = ctx_den->best_tx_size;
      x->skip_txfm[0] = ctx_den->best_mode_skip_txfm;
    } else {
      ctx_den->best_ref_frame = LAST_FRAME;
      *best_rdc = this_rdc;
    }
  }
}
#endif  // CONFIG_VP9_TEMPORAL_DENOISING

static INLINE int get_force_skip_low_temp_var(uint8_t *variance_low, int mi_row,
                                              int mi_col, BLOCK_SIZE bsize) {
  const int i = (mi_row & 0x7) >> 1;
  const int j = (mi_col & 0x7) >> 1;
  int force_skip_low_temp_var = 0;
  // Set force_skip_low_temp_var based on the block size and block offset.
  if (bsize == BLOCK_64X64) {
    force_skip_low_temp_var = variance_low[0];
  } else if (bsize == BLOCK_64X32) {
    if (!(mi_col & 0x7) && !(mi_row & 0x7)) {
      force_skip_low_temp_var = variance_low[1];
    } else if (!(mi_col & 0x7) && (mi_row & 0x7)) {
      force_skip_low_temp_var = variance_low[2];
    }
  } else if (bsize == BLOCK_32X64) {
    if (!(mi_col & 0x7) && !(mi_row & 0x7)) {
      force_skip_low_temp_var = variance_low[3];
    } else if ((mi_col & 0x7) && !(mi_row & 0x7)) {
      force_skip_low_temp_var = variance_low[4];
    }
  } else if (bsize == BLOCK_32X32) {
    if (!(mi_col & 0x7) && !(mi_row & 0x7)) {
      force_skip_low_temp_var = variance_low[5];
    } else if ((mi_col & 0x7) && !(mi_row & 0x7)) {
      force_skip_low_temp_var = variance_low[6];
    } else if (!(mi_col & 0x7) && (mi_row & 0x7)) {
      force_skip_low_temp_var = variance_low[7];
    } else if ((mi_col & 0x7) && (mi_row & 0x7)) {
      force_skip_low_temp_var = variance_low[8];
    }
  } else if (bsize == BLOCK_16X16) {
    force_skip_low_temp_var = variance_low[pos_shift_16x16[i][j]];
  } else if (bsize == BLOCK_32X16) {
    // The col shift index for the second 16x16 block.
    const int j2 = ((mi_col + 2) & 0x7) >> 1;
    // Only if each 16x16 block inside has low temporal variance.
    force_skip_low_temp_var = variance_low[pos_shift_16x16[i][j]] &&
                              variance_low[pos_shift_16x16[i][j2]];
  } else if (bsize == BLOCK_16X32) {
    // The row shift index for the second 16x16 block.
    const int i2 = ((mi_row + 2) & 0x7) >> 1;
    force_skip_low_temp_var = variance_low[pos_shift_16x16[i][j]] &&
                              variance_low[pos_shift_16x16[i2][j]];
  }
  return force_skip_low_temp_var;
}

static void search_filter_ref(VP9_COMP *cpi, MACROBLOCK *x, RD_COST *this_rdc,
                              int mi_row, int mi_col, PRED_BUFFER *tmp,
                              BLOCK_SIZE bsize, int reuse_inter_pred,
                              PRED_BUFFER **this_mode_pred, unsigned int *var_y,
                              unsigned int *sse_y, int force_smooth_filter,
                              int *this_early_term, int *flag_preduv_computed,
                              int use_model_yrd_large) {
  MACROBLOCKD *const xd = &x->e_mbd;
  MODE_INFO *const mi = xd->mi[0];
  struct macroblockd_plane *const pd = &xd->plane[0];
  const int bw = num_4x4_blocks_wide_lookup[bsize] << 2;

  int pf_rate[3] = { 0 };
  int64_t pf_dist[3] = { 0 };
  int curr_rate[3] = { 0 };
  unsigned int pf_var[3] = { 0 };
  unsigned int pf_sse[3] = { 0 };
  TX_SIZE pf_tx_size[3] = { 0 };
  int64_t best_cost = INT64_MAX;
  INTERP_FILTER best_filter = SWITCHABLE, filter;
  PRED_BUFFER *current_pred = *this_mode_pred;
  uint8_t skip_txfm = SKIP_TXFM_NONE;
  int best_early_term = 0;
  int best_flag_preduv_computed[2] = { 0 };
  INTERP_FILTER filter_start = force_smooth_filter ? EIGHTTAP_SMOOTH : EIGHTTAP;
  INTERP_FILTER filter_end = EIGHTTAP_SMOOTH;
  for (filter = filter_start; filter <= filter_end; ++filter) {
    int64_t cost;
    mi->interp_filter = filter;
    vp9_build_inter_predictors_sby(xd, mi_row, mi_col, bsize);
    // For large partition blocks, extra testing is done.
    if (use_model_yrd_large)
      model_rd_for_sb_y_large(cpi, bsize, x, xd, &pf_rate[filter],
                              &pf_dist[filter], &pf_var[filter],
                              &pf_sse[filter], mi_row, mi_col, this_early_term,
                              flag_preduv_computed);
    else
      model_rd_for_sb_y(cpi, bsize, x, xd, &pf_rate[filter], &pf_dist[filter],
                        &pf_var[filter], &pf_sse[filter], 0);
    curr_rate[filter] = pf_rate[filter];
    pf_rate[filter] += vp9_get_switchable_rate(cpi, xd);
    cost = RDCOST(x->rdmult, x->rddiv, pf_rate[filter], pf_dist[filter]);
    pf_tx_size[filter] = mi->tx_size;
    if (cost < best_cost) {
      best_filter = filter;
      best_cost = cost;
      skip_txfm = x->skip_txfm[0];
      best_early_term = *this_early_term;
      best_flag_preduv_computed[0] = flag_preduv_computed[0];
      best_flag_preduv_computed[1] = flag_preduv_computed[1];

      if (reuse_inter_pred) {
        if (*this_mode_pred != current_pred) {
          free_pred_buffer(*this_mode_pred);
          *this_mode_pred = current_pred;
        }
        if (filter != filter_end) {
          current_pred = &tmp[get_pred_buffer(tmp, 3)];
          pd->dst.buf = current_pred->data;
          pd->dst.stride = bw;
        }
      }
    }
  }

  if (reuse_inter_pred && *this_mode_pred != current_pred)
    free_pred_buffer(current_pred);

  mi->interp_filter = best_filter;
  mi->tx_size = pf_tx_size[best_filter];
  this_rdc->rate = curr_rate[best_filter];
  this_rdc->dist = pf_dist[best_filter];
  *var_y = pf_var[best_filter];
  *sse_y = pf_sse[best_filter];
  x->skip_txfm[0] = skip_txfm;
  *this_early_term = best_early_term;
  flag_preduv_computed[0] = best_flag_preduv_computed[0];
  flag_preduv_computed[1] = best_flag_preduv_computed[1];
  if (reuse_inter_pred) {
    pd->dst.buf = (*this_mode_pred)->data;
    pd->dst.stride = (*this_mode_pred)->stride;
  } else if (best_filter < filter_end) {
    mi->interp_filter = best_filter;
    vp9_build_inter_predictors_sby(xd, mi_row, mi_col, bsize);
  }
}

static int search_new_mv(VP9_COMP *cpi, MACROBLOCK *x,
                         int_mv frame_mv[][MAX_REF_FRAMES],
                         MV_REFERENCE_FRAME ref_frame, int gf_temporal_ref,
                         BLOCK_SIZE bsize, int mi_row, int mi_col,
                         int best_pred_sad, int *rate_mv,
                         unsigned int best_sse_sofar, RD_COST *best_rdc) {
  SVC *const svc = &cpi->svc;
  MACROBLOCKD *const xd = &x->e_mbd;
  MODE_INFO *const mi = xd->mi[0];
  SPEED_FEATURES *const sf = &cpi->sf;

  if (ref_frame > LAST_FRAME && gf_temporal_ref &&
      cpi->oxcf.rc_mode == VPX_CBR) {
    int tmp_sad;
    uint32_t dis;
    int cost_list[5] = { INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX };

    if (bsize < BLOCK_16X16) return -1;

    tmp_sad = vp9_int_pro_motion_estimation(
        cpi, x, bsize, mi_row, mi_col,
        &x->mbmi_ext->ref_mvs[ref_frame][0].as_mv);

    if (tmp_sad > x->pred_mv_sad[LAST_FRAME]) return -1;
    if (tmp_sad + (num_pels_log2_lookup[bsize] << 4) > best_pred_sad) return -1;

    frame_mv[NEWMV][ref_frame].as_int = mi->mv[0].as_int;
    *rate_mv = vp9_mv_bit_cost(&frame_mv[NEWMV][ref_frame].as_mv,
                               &x->mbmi_ext->ref_mvs[ref_frame][0].as_mv,
                               x->nmvjointcost, x->mvcost, MV_COST_WEIGHT);
    frame_mv[NEWMV][ref_frame].as_mv.row >>= 3;
    frame_mv[NEWMV][ref_frame].as_mv.col >>= 3;

    cpi->find_fractional_mv_step(
        x, &frame_mv[NEWMV][ref_frame].as_mv,
        &x->mbmi_ext->ref_mvs[ref_frame][0].as_mv,
        cpi->common.allow_high_precision_mv, x->errorperbit,
        &cpi->fn_ptr[bsize], cpi->sf.mv.subpel_force_stop,
        cpi->sf.mv.subpel_search_level, cond_cost_list(cpi, cost_list),
        x->nmvjointcost, x->mvcost, &dis, &x->pred_sse[ref_frame], NULL, 0, 0,
        cpi->sf.use_accurate_subpel_search);
  } else if (svc->use_base_mv && svc->spatial_layer_id) {
    if (frame_mv[NEWMV][ref_frame].as_int != INVALID_MV) {
      const int pre_stride = xd->plane[0].pre[0].stride;
      unsigned int base_mv_sse = UINT_MAX;
      int scale = (cpi->rc.avg_frame_low_motion > 60) ? 2 : 4;
      const uint8_t *const pre_buf =
          xd->plane[0].pre[0].buf +
          (frame_mv[NEWMV][ref_frame].as_mv.row >> 3) * pre_stride +
          (frame_mv[NEWMV][ref_frame].as_mv.col >> 3);
      cpi->fn_ptr[bsize].vf(x->plane[0].src.buf, x->plane[0].src.stride,
                            pre_buf, pre_stride, &base_mv_sse);

      // Exit NEWMV search if base_mv is (0,0) && bsize < BLOCK_16x16,
      // for SVC encoding.
      if (cpi->use_svc && svc->use_base_mv && bsize < BLOCK_16X16 &&
          frame_mv[NEWMV][ref_frame].as_mv.row == 0 &&
          frame_mv[NEWMV][ref_frame].as_mv.col == 0)
        return -1;

      // Exit NEWMV search if base_mv_sse is large.
      if (sf->base_mv_aggressive && (base_mv_sse >> scale) > best_sse_sofar)
        return -1;
      if ((base_mv_sse >> 1) < best_sse_sofar) {
        // Base layer mv is good.
        // Exit NEWMV search if the base_mv is (0, 0) and sse is low, since
        // (0, 0) mode is already tested.
        unsigned int base_mv_sse_normalized =
            base_mv_sse >>
            (b_width_log2_lookup[bsize] + b_height_log2_lookup[bsize]);
        if (sf->base_mv_aggressive && base_mv_sse <= best_sse_sofar &&
            base_mv_sse_normalized < 400 &&
            frame_mv[NEWMV][ref_frame].as_mv.row == 0 &&
            frame_mv[NEWMV][ref_frame].as_mv.col == 0)
          return -1;
        if (!combined_motion_search(cpi, x, bsize, mi_row, mi_col,
                                    &frame_mv[NEWMV][ref_frame], rate_mv,
                                    best_rdc->rdcost, 1)) {
          return -1;
        }
      } else if (!combined_motion_search(cpi, x, bsize, mi_row, mi_col,
                                         &frame_mv[NEWMV][ref_frame], rate_mv,
                                         best_rdc->rdcost, 0)) {
        return -1;
      }
    } else if (!combined_motion_search(cpi, x, bsize, mi_row, mi_col,
                                       &frame_mv[NEWMV][ref_frame], rate_mv,
                                       best_rdc->rdcost, 0)) {
      return -1;
    }
  } else if (!combined_motion_search(cpi, x, bsize, mi_row, mi_col,
                                     &frame_mv[NEWMV][ref_frame], rate_mv,
                                     best_rdc->rdcost, 0)) {
    return -1;
  }

  return 0;
}

static INLINE void init_best_pickmode(BEST_PICKMODE *bp) {
  bp->best_mode = ZEROMV;
  bp->best_ref_frame = LAST_FRAME;
  bp->best_tx_size = TX_SIZES;
  bp->best_intra_tx_size = TX_SIZES;
  bp->best_pred_filter = EIGHTTAP;
  bp->best_mode_skip_txfm = SKIP_TXFM_NONE;
  bp->best_second_ref_frame = NO_REF_FRAME;
  bp->best_pred = NULL;
}

void vp9_pick_inter_mode(VP9_COMP *cpi, MACROBLOCK *x, TileDataEnc *tile_data,
                         int mi_row, int mi_col, RD_COST *rd_cost,
                         BLOCK_SIZE bsize, PICK_MODE_CONTEXT *ctx) {
  VP9_COMMON *const cm = &cpi->common;
  SPEED_FEATURES *const sf = &cpi->sf;
  SVC *const svc = &cpi->svc;
  MACROBLOCKD *const xd = &x->e_mbd;
  MODE_INFO *const mi = xd->mi[0];
  struct macroblockd_plane *const pd = &xd->plane[0];

  BEST_PICKMODE best_pickmode;

  MV_REFERENCE_FRAME ref_frame;
  MV_REFERENCE_FRAME usable_ref_frame, second_ref_frame;
  int_mv frame_mv[MB_MODE_COUNT][MAX_REF_FRAMES];
  uint8_t mode_checked[MB_MODE_COUNT][MAX_REF_FRAMES];
  struct buf_2d yv12_mb[4][MAX_MB_PLANE] = { 0 };
  RD_COST this_rdc, best_rdc;
  // var_y and sse_y are saved to be used in skipping checking
  unsigned int var_y = UINT_MAX;
  unsigned int sse_y = UINT_MAX;
  const int intra_cost_penalty =
      vp9_get_intra_cost_penalty(cpi, bsize, cm->base_qindex, cm->y_dc_delta_q);
  int64_t inter_mode_thresh =
      RDCOST(x->rdmult, x->rddiv, intra_cost_penalty, 0);
  const int *const rd_threshes = cpi->rd.threshes[mi->segment_id][bsize];
  const int sb_row = mi_row >> MI_BLOCK_SIZE_LOG2;
  int thresh_freq_fact_idx = (sb_row * BLOCK_SIZES + bsize) * MAX_MODES;
  const int *const rd_thresh_freq_fact =
      (cpi->sf.adaptive_rd_thresh_row_mt)
          ? &(tile_data->row_base_thresh_freq_fact[thresh_freq_fact_idx])
          : tile_data->thresh_freq_fact[bsize];
#if CONFIG_VP9_TEMPORAL_DENOISING
  const int denoise_recheck_zeromv = 1;
#endif
  INTERP_FILTER filter_ref;
  int pred_filter_search = cm->interp_filter == SWITCHABLE;
  int const_motion[MAX_REF_FRAMES] = { 0 };
  const int bh = num_4x4_blocks_high_lookup[bsize] << 2;
  const int bw = num_4x4_blocks_wide_lookup[bsize] << 2;
  // For speed 6, the result of interp filter is reused later in actual encoding
  // process.
  // tmp[3] points to dst buffer, and the other 3 point to allocated buffers.
  PRED_BUFFER tmp[4];
  DECLARE_ALIGNED(16, uint8_t, pred_buf[3 * 64 * 64] VPX_UNINITIALIZED);
#if CONFIG_VP9_HIGHBITDEPTH
  DECLARE_ALIGNED(16, uint16_t, pred_buf_16[3 * 64 * 64] VPX_UNINITIALIZED);
#endif
  struct buf_2d orig_dst = pd->dst;
  PRED_BUFFER *this_mode_pred = NULL;
  const int pixels_in_block = bh * bw;
  int reuse_inter_pred = cpi->sf.reuse_inter_pred_sby && ctx->pred_pixel_ready;
  int ref_frame_skip_mask = 0;
  int idx;
  int best_pred_sad = INT_MAX;
  int best_early_term = 0;
  int ref_frame_cost[MAX_REF_FRAMES];
  int svc_force_zero_mode[3] = { 0 };
  int perform_intra_pred = 1;
  int use_golden_nonzeromv = 1;
  int force_skip_low_temp_var = 0;
  int skip_ref_find_pred[4] = { 0 };
  unsigned int sse_zeromv_normalized = UINT_MAX;
  unsigned int best_sse_sofar = UINT_MAX;
  int gf_temporal_ref = 0;
  int force_test_gf_zeromv = 0;
#if CONFIG_VP9_TEMPORAL_DENOISING
  VP9_PICKMODE_CTX_DEN ctx_den;
  int64_t zero_last_cost_orig = INT64_MAX;
  int denoise_svc_pickmode = 1;
#endif
  INTERP_FILTER filter_gf_svc = EIGHTTAP;
  MV_REFERENCE_FRAME inter_layer_ref = GOLDEN_FRAME;
  const struct segmentation *const seg = &cm->seg;
  int comp_modes = 0;
  int num_inter_modes = (cpi->use_svc) ? RT_INTER_MODES_SVC : RT_INTER_MODES;
  int flag_svc_subpel = 0;
  int svc_mv_col = 0;
  int svc_mv_row = 0;
  int no_scaling = 0;
  int large_block = 0;
  int use_model_yrd_large = 0;
  unsigned int thresh_svc_skip_golden = 500;
  unsigned int thresh_skip_golden = 500;
  int force_smooth_filter = cpi->sf.force_smooth_interpol;
  int scene_change_detected =
      cpi->rc.high_source_sad ||
      (cpi->use_svc && cpi->svc.high_source_sad_superframe);

  init_best_pickmode(&best_pickmode);

  x->encode_breakout = seg->enabled
                           ? cpi->segment_encode_breakout[mi->segment_id]
                           : cpi->encode_breakout;

  x->source_variance = UINT_MAX;
  if (cpi->sf.default_interp_filter == BILINEAR) {
    best_pickmode.best_pred_filter = BILINEAR;
    filter_gf_svc = BILINEAR;
  }
  if (cpi->use_svc && svc->spatial_layer_id > 0) {
    int layer =
        LAYER_IDS_TO_IDX(svc->spatial_layer_id - 1, svc->temporal_layer_id,
                         svc->number_temporal_layers);
    LAYER_CONTEXT *const lc = &svc->layer_context[layer];
    if (lc->scaling_factor_num == lc->scaling_factor_den) no_scaling = 1;
  }
  if (svc->spatial_layer_id > 0 &&
      (svc->high_source_sad_superframe || no_scaling))
    thresh_svc_skip_golden = 0;
  // Lower the skip threshold if lower spatial layer is better quality relative
  // to current layer.
  else if (svc->spatial_layer_id > 0 && cm->base_qindex > 150 &&
           cm->base_qindex > svc->lower_layer_qindex + 15)
    thresh_svc_skip_golden = 100;
  // Increase skip threshold if lower spatial layer is lower quality relative
  // to current layer.
  else if (svc->spatial_layer_id > 0 && cm->base_qindex < 140 &&
           cm->base_qindex < svc->lower_layer_qindex - 20)
    thresh_svc_skip_golden = 1000;

  if (!cpi->use_svc ||
      (svc->use_gf_temporal_ref_current_layer &&
       !svc->layer_context[svc->temporal_layer_id].is_key_frame)) {
    struct scale_factors *const sf_last = &cm->frame_refs[LAST_FRAME - 1].sf;
    struct scale_factors *const sf_golden =
        &cm->frame_refs[GOLDEN_FRAME - 1].sf;
    gf_temporal_ref = 1;
    // For temporal long term prediction, check that the golden reference
    // is same scale as last reference, otherwise disable.
    if ((sf_last->x_scale_fp != sf_golden->x_scale_fp) ||
        (sf_last->y_scale_fp != sf_golden->y_scale_fp)) {
      gf_temporal_ref = 0;
    } else {
      if (cpi->rc.avg_frame_low_motion > 70)
        thresh_svc_skip_golden = 500;
      else
        thresh_svc_skip_golden = 0;
    }
  }

  init_ref_frame_cost(cm, xd, ref_frame_cost);
  memset(&mode_checked[0][0], 0, MB_MODE_COUNT * MAX_REF_FRAMES);

  if (reuse_inter_pred) {
    int i;
    for (i = 0; i < 3; i++) {
#if CONFIG_VP9_HIGHBITDEPTH
      if (cm->use_highbitdepth)
        tmp[i].data = CONVERT_TO_BYTEPTR(&pred_buf_16[pixels_in_block * i]);
      else
        tmp[i].data = &pred_buf[pixels_in_block * i];
#else
      tmp[i].data = &pred_buf[pixels_in_block * i];
#endif  // CONFIG_VP9_HIGHBITDEPTH
      tmp[i].stride = bw;
      tmp[i].in_use = 0;
    }
    tmp[3].data = pd->dst.buf;
    tmp[3].stride = pd->dst.stride;
    tmp[3].in_use = 0;
  }

  x->skip_encode = cpi->sf.skip_encode_frame && x->q_index < QIDX_SKIP_THRESH;
  x->skip = 0;

  if (cpi->sf.cb_pred_filter_search) {
    const int bsl = mi_width_log2_lookup[bsize];
    pred_filter_search = cm->interp_filter == SWITCHABLE
                             ? (((mi_row + mi_col) >> bsl) +
                                get_chessboard_index(cm->current_video_frame)) &
                                   0x1
                             : 0;
  }
  // Instead of using vp9_get_pred_context_switchable_interp(xd) to assign
  // filter_ref, we use a less strict condition on assigning filter_ref.
  // This is to reduce the probabily of entering the flow of not assigning
  // filter_ref and then skip filter search.
  filter_ref = cm->interp_filter;
  if (cpi->sf.default_interp_filter != BILINEAR) {
    if (xd->above_mi && is_inter_block(xd->above_mi))
      filter_ref = xd->above_mi->interp_filter;
    else if (xd->left_mi && is_inter_block(xd->left_mi))
      filter_ref = xd->left_mi->interp_filter;
  }

  // initialize mode decisions
  vp9_rd_cost_reset(&best_rdc);
  vp9_rd_cost_reset(rd_cost);
  mi->sb_type = bsize;
  mi->ref_frame[0] = NO_REF_FRAME;
  mi->ref_frame[1] = NO_REF_FRAME;

  mi->tx_size =
      VPXMIN(max_txsize_lookup[bsize], tx_mode_to_biggest_tx_size[cm->tx_mode]);

  if (sf->short_circuit_flat_blocks || sf->limit_newmv_early_exit) {
#if CONFIG_VP9_HIGHBITDEPTH
    if (xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH)
      x->source_variance = vp9_high_get_sby_perpixel_variance(
          cpi, &x->plane[0].src, bsize, xd->bd);
    else
#endif  // CONFIG_VP9_HIGHBITDEPTH
      x->source_variance =
          vp9_get_sby_perpixel_variance(cpi, &x->plane[0].src, bsize);

    if (cpi->oxcf.content == VP9E_CONTENT_SCREEN &&
        cpi->oxcf.aq_mode == CYCLIC_REFRESH_AQ && mi->segment_id > 0 &&
        x->zero_temp_sad_source && x->source_variance == 0) {
      mi->segment_id = 0;
      vp9_init_plane_quantizers(cpi, x);
    }
  }

#if CONFIG_VP9_TEMPORAL_DENOISING
  if (cpi->oxcf.noise_sensitivity > 0) {
    if (cpi->use_svc) denoise_svc_pickmode = vp9_denoise_svc_non_key(cpi);
    if (cpi->denoiser.denoising_level > kDenLowLow && denoise_svc_pickmode)
      vp9_denoiser_reset_frame_stats(ctx);
  }
#endif

  if (cpi->rc.frames_since_golden == 0 && gf_temporal_ref &&
      !cpi->rc.alt_ref_gf_group && !cpi->rc.last_frame_is_src_altref) {
    usable_ref_frame = LAST_FRAME;
  } else {
    usable_ref_frame = GOLDEN_FRAME;
  }

  if (cpi->oxcf.lag_in_frames > 0 && cpi->oxcf.rc_mode == VPX_VBR) {
    if (cpi->rc.alt_ref_gf_group || cpi->rc.is_src_frame_alt_ref)
      usable_ref_frame = ALTREF_FRAME;

    if (cpi->rc.is_src_frame_alt_ref) {
      skip_ref_find_pred[LAST_FRAME] = 1;
      skip_ref_find_pred[GOLDEN_FRAME] = 1;
    }
    if (!cm->show_frame) {
      if (cpi->rc.frames_since_key == 1) {
        usable_ref_frame = LAST_FRAME;
        skip_ref_find_pred[GOLDEN_FRAME] = 1;
        skip_ref_find_pred[ALTREF_FRAME] = 1;
      }
    }
  }

  // For svc mode, on spatial_layer_id > 0: if the reference has different scale
  // constrain the inter mode to only test zero motion.
  if (cpi->use_svc && svc->force_zero_mode_spatial_ref &&
      svc->spatial_layer_id > 0 && !gf_temporal_ref) {
    if (cpi->ref_frame_flags & VP9_LAST_FLAG) {
      struct scale_factors *const ref_sf = &cm->frame_refs[LAST_FRAME - 1].sf;
      if (vp9_is_scaled(ref_sf)) {
        svc_force_zero_mode[LAST_FRAME - 1] = 1;
        inter_layer_ref = LAST_FRAME;
      }
    }
    if (cpi->ref_frame_flags & VP9_GOLD_FLAG) {
      struct scale_factors *const ref_sf = &cm->frame_refs[GOLDEN_FRAME - 1].sf;
      if (vp9_is_scaled(ref_sf)) {
        svc_force_zero_mode[GOLDEN_FRAME - 1] = 1;
        inter_layer_ref = GOLDEN_FRAME;
      }
    }
  }

  if (cpi->sf.short_circuit_low_temp_var) {
    force_skip_low_temp_var =
        get_force_skip_low_temp_var(&x->variance_low[0], mi_row, mi_col, bsize);
    // If force_skip_low_temp_var is set, and for short circuit mode = 1 and 3,
    // skip golden reference.
    if ((cpi->sf.short_circuit_low_temp_var == 1 ||
         cpi->sf.short_circuit_low_temp_var == 3) &&
        force_skip_low_temp_var) {
      usable_ref_frame = LAST_FRAME;
    }
  }

  if (sf->disable_golden_ref && (x->content_state_sb != kVeryHighSad ||
                                 cpi->rc.avg_frame_low_motion < 60))
    usable_ref_frame = LAST_FRAME;

  if (!((cpi->ref_frame_flags & VP9_GOLD_FLAG) &&
        !svc_force_zero_mode[GOLDEN_FRAME - 1] && !force_skip_low_temp_var))
    use_golden_nonzeromv = 0;

  if (cpi->oxcf.speed >= 8 && !cpi->use_svc &&
      ((cpi->rc.frames_since_golden + 1) < x->last_sb_high_content ||
       x->last_sb_high_content > 40 || cpi->rc.frames_since_golden > 120))
    usable_ref_frame = LAST_FRAME;

  // Compound prediction modes: (0,0) on LAST/GOLDEN and ARF.
  if (cm->reference_mode == REFERENCE_MODE_SELECT &&
      cpi->sf.use_compound_nonrd_pickmode && usable_ref_frame == ALTREF_FRAME)
    comp_modes = 2;

  // If the segment reference frame feature is enabled and it's set to GOLDEN
  // reference, then make sure we don't skip checking GOLDEN, this is to
  // prevent possibility of not picking any mode.
  if (segfeature_active(seg, mi->segment_id, SEG_LVL_REF_FRAME) &&
      get_segdata(seg, mi->segment_id, SEG_LVL_REF_FRAME) == GOLDEN_FRAME) {
    usable_ref_frame = GOLDEN_FRAME;
    skip_ref_find_pred[GOLDEN_FRAME] = 0;
    thresh_svc_skip_golden = 0;
  }

  for (ref_frame = LAST_FRAME; ref_frame <= usable_ref_frame; ++ref_frame) {
    // Skip find_predictor if the reference frame is not in the
    // ref_frame_flags (i.e., not used as a reference for this frame).
    skip_ref_find_pred[ref_frame] =
        !(cpi->ref_frame_flags & ref_frame_to_flag(ref_frame));
    if (!skip_ref_find_pred[ref_frame]) {
      find_predictors(cpi, x, ref_frame, frame_mv, const_motion,
                      &ref_frame_skip_mask, tile_data, mi_row, mi_col, yv12_mb,
                      bsize, force_skip_low_temp_var, comp_modes > 0);
    }
  }

  if (cpi->use_svc || cpi->oxcf.speed <= 7 || bsize < BLOCK_32X32)
    x->sb_use_mv_part = 0;

  // Set the flag_svc_subpel to 1 for SVC if the lower spatial layer used
  // an averaging filter for downsampling (phase = 8). If so, we will test
  // a nonzero motion mode on the spatial reference.
  // The nonzero motion is half pixel shifted to left and top (-4, -4).
  if (cpi->use_svc && svc->spatial_layer_id > 0 &&
      svc_force_zero_mode[inter_layer_ref - 1] &&
      svc->downsample_filter_phase[svc->spatial_layer_id - 1] == 8 &&
      !gf_temporal_ref) {
    svc_mv_col = -4;
    svc_mv_row = -4;
    flag_svc_subpel = 1;
  }

  // For SVC with quality layers, when QP of lower layer is lower
  // than current layer: force check of GF-ZEROMV before early exit
  // due to skip flag.
  if (svc->spatial_layer_id > 0 && no_scaling &&
      (cpi->ref_frame_flags & VP9_GOLD_FLAG) &&
      cm->base_qindex > svc->lower_layer_qindex + 10)
    force_test_gf_zeromv = 1;

  // For low motion content use x->sb_is_skin in addition to VeryHighSad
  // for setting large_block.
  large_block = (x->content_state_sb == kVeryHighSad ||
                 (x->sb_is_skin && cpi->rc.avg_frame_low_motion > 70) ||
                 cpi->oxcf.speed < 7)
                    ? bsize > BLOCK_32X32
                    : bsize >= BLOCK_32X32;
  use_model_yrd_large =
      cpi->oxcf.rc_mode == VPX_CBR && large_block &&
      !cyclic_refresh_segment_id_boosted(xd->mi[0]->segment_id) &&
      cm->base_qindex;

  for (idx = 0; idx < num_inter_modes + comp_modes; ++idx) {
    int rate_mv = 0;
    int mode_rd_thresh;
    int mode_index;
    int i;
    int64_t this_sse;
    int is_skippable;
    int this_early_term = 0;
    int rd_computed = 0;
    int flag_preduv_computed[2] = { 0 };
    int inter_mv_mode = 0;
    int skip_this_mv = 0;
    int comp_pred = 0;
    int force_mv_inter_layer = 0;
    PREDICTION_MODE this_mode;
    second_ref_frame = NO_REF_FRAME;

    if (idx < num_inter_modes) {
      this_mode = ref_mode_set[idx].pred_mode;
      ref_frame = ref_mode_set[idx].ref_frame;

      if (cpi->use_svc) {
        this_mode = ref_mode_set_svc[idx].pred_mode;
        ref_frame = ref_mode_set_svc[idx].ref_frame;
      }
    } else {
      // Add (0,0) compound modes.
      this_mode = ZEROMV;
      ref_frame = LAST_FRAME;
      if (idx == num_inter_modes + comp_modes - 1) ref_frame = GOLDEN_FRAME;
      second_ref_frame = ALTREF_FRAME;
      comp_pred = 1;
    }

    if (ref_frame > usable_ref_frame) continue;
    if (skip_ref_find_pred[ref_frame]) continue;

    if (svc->previous_frame_is_intra_only) {
      if (ref_frame != LAST_FRAME || frame_mv[this_mode][ref_frame].as_int != 0)
        continue;
    }

    // If the segment reference frame feature is enabled then do nothing if the
    // current ref frame is not allowed.
    if (segfeature_active(seg, mi->segment_id, SEG_LVL_REF_FRAME) &&
        get_segdata(seg, mi->segment_id, SEG_LVL_REF_FRAME) != (int)ref_frame)
      continue;

    if (flag_svc_subpel && ref_frame == inter_layer_ref) {
      force_mv_inter_layer = 1;
      // Only test mode if NEARESTMV/NEARMV is (svc_mv_col, svc_mv_row),
      // otherwise set NEWMV to (svc_mv_col, svc_mv_row).
      if (this_mode == NEWMV) {
        frame_mv[this_mode][ref_frame].as_mv.col = svc_mv_col;
        frame_mv[this_mode][ref_frame].as_mv.row = svc_mv_row;
      } else if (frame_mv[this_mode][ref_frame].as_mv.col != svc_mv_col ||
                 frame_mv[this_mode][ref_frame].as_mv.row != svc_mv_row) {
        continue;
      }
    }

    if (comp_pred) {
      if (!cpi->allow_comp_inter_inter) continue;
      // Skip compound inter modes if ARF is not available.
      if (!(cpi->ref_frame_flags & ref_frame_to_flag(second_ref_frame)))
        continue;
      // Do not allow compound prediction if the segment level reference frame
      // feature is in use as in this case there can only be one reference.
      if (segfeature_active(seg, mi->segment_id, SEG_LVL_REF_FRAME)) continue;
    }

    // For CBR mode: skip the golden reference search if sse of zeromv_last is
    // below threshold.
    if (ref_frame == GOLDEN_FRAME && cpi->oxcf.rc_mode == VPX_CBR &&
        ((cpi->use_svc && sse_zeromv_normalized < thresh_svc_skip_golden) ||
         (!cpi->use_svc && sse_zeromv_normalized < thresh_skip_golden)))
      continue;

    if (!(cpi->ref_frame_flags & ref_frame_to_flag(ref_frame))) continue;

    // For screen content. If zero_temp_sad source is computed: skip
    // non-zero motion check for stationary blocks. If the superblock is
    // non-stationary then for flat blocks skip the zero last check (keep golden
    // as it may be inter-layer reference). Otherwise (if zero_temp_sad_source
    // is not computed) skip non-zero motion check for flat blocks.
    // TODO(marpan): Compute zero_temp_sad_source per coding block.
    if (cpi->oxcf.content == VP9E_CONTENT_SCREEN) {
      if (cpi->compute_source_sad_onepass && cpi->sf.use_source_sad) {
        if ((frame_mv[this_mode][ref_frame].as_int != 0 &&
             x->zero_temp_sad_source) ||
            (frame_mv[this_mode][ref_frame].as_int == 0 &&
             x->source_variance == 0 && ref_frame == LAST_FRAME &&
             !x->zero_temp_sad_source))
          continue;
      } else if (frame_mv[this_mode][ref_frame].as_int != 0 &&
                 x->source_variance == 0) {
        continue;
      }
    }

    if (!(cpi->sf.inter_mode_mask[bsize] & (1 << this_mode))) continue;

    if (cpi->oxcf.lag_in_frames > 0 && cpi->oxcf.rc_mode == VPX_VBR) {
      if (cpi->rc.is_src_frame_alt_ref &&
          (ref_frame != ALTREF_FRAME ||
           frame_mv[this_mode][ref_frame].as_int != 0))
        continue;

      if (!cm->show_frame && ref_frame == ALTREF_FRAME &&
          frame_mv[this_mode][ref_frame].as_int != 0)
        continue;

      if (cpi->rc.alt_ref_gf_group && cm->show_frame &&
          cpi->rc.frames_since_golden > (cpi->rc.baseline_gf_interval >> 1) &&
          ref_frame == GOLDEN_FRAME &&
          frame_mv[this_mode][ref_frame].as_int != 0)
        continue;

      if (cpi->rc.alt_ref_gf_group && cm->show_frame &&
          cpi->rc.frames_since_golden > 0 &&
          cpi->rc.frames_since_golden < (cpi->rc.baseline_gf_interval >> 1) &&
          ref_frame == ALTREF_FRAME &&
          frame_mv[this_mode][ref_frame].as_int != 0)
        continue;
    }

    if (const_motion[ref_frame] && this_mode == NEARMV) continue;

    // Skip non-zeromv mode search for golden frame if force_skip_low_temp_var
    // is set. If nearestmv for golden frame is 0, zeromv mode will be skipped
    // later.
    if (!force_mv_inter_layer && force_skip_low_temp_var &&
        ref_frame == GOLDEN_FRAME &&
        frame_mv[this_mode][ref_frame].as_int != 0) {
      continue;
    }

    if (x->content_state_sb != kVeryHighSad &&
        (cpi->sf.short_circuit_low_temp_var >= 2 ||
         (cpi->sf.short_circuit_low_temp_var == 1 && bsize == BLOCK_64X64)) &&
        force_skip_low_temp_var && ref_frame == LAST_FRAME &&
        this_mode == NEWMV) {
      continue;
    }

    if (cpi->use_svc) {
      if (!force_mv_inter_layer && svc_force_zero_mode[ref_frame - 1] &&
          frame_mv[this_mode][ref_frame].as_int != 0)
        continue;
    }

    // Disable this drop out case if the ref frame segment level feature is
    // enabled for this segment. This is to prevent the possibility that we end
    // up unable to pick any mode.
    if (!segfeature_active(seg, mi->segment_id, SEG_LVL_REF_FRAME)) {
      if (sf->reference_masking &&
          !(frame_mv[this_mode][ref_frame].as_int == 0 &&
            ref_frame == LAST_FRAME)) {
        if (usable_ref_frame < ALTREF_FRAME) {
          if (!force_skip_low_temp_var && usable_ref_frame > LAST_FRAME) {
            i = (ref_frame == LAST_FRAME) ? GOLDEN_FRAME : LAST_FRAME;
            if ((cpi->ref_frame_flags & ref_frame_to_flag(i)))
              if (x->pred_mv_sad[ref_frame] > (x->pred_mv_sad[i] << 1))
                ref_frame_skip_mask |= (1 << ref_frame);
          }
        } else if (!cpi->rc.is_src_frame_alt_ref &&
                   !(frame_mv[this_mode][ref_frame].as_int == 0 &&
                     ref_frame == ALTREF_FRAME)) {
          int ref1 = (ref_frame == GOLDEN_FRAME) ? LAST_FRAME : GOLDEN_FRAME;
          int ref2 = (ref_frame == ALTREF_FRAME) ? LAST_FRAME : ALTREF_FRAME;
          if (((cpi->ref_frame_flags & ref_frame_to_flag(ref1)) &&
               (x->pred_mv_sad[ref_frame] > (x->pred_mv_sad[ref1] << 1))) ||
              ((cpi->ref_frame_flags & ref_frame_to_flag(ref2)) &&
               (x->pred_mv_sad[ref_frame] > (x->pred_mv_sad[ref2] << 1))))
            ref_frame_skip_mask |= (1 << ref_frame);
        }
      }
      if (ref_frame_skip_mask & (1 << ref_frame)) continue;
    }

    // Select prediction reference frames.
    for (i = 0; i < MAX_MB_PLANE; i++) {
      xd->plane[i].pre[0] = yv12_mb[ref_frame][i];
      if (comp_pred) xd->plane[i].pre[1] = yv12_mb[second_ref_frame][i];
    }

    mi->ref_frame[0] = ref_frame;
    mi->ref_frame[1] = second_ref_frame;
    set_ref_ptrs(cm, xd, ref_frame, second_ref_frame);

    mode_index = mode_idx[ref_frame][INTER_OFFSET(this_mode)];
    mode_rd_thresh = best_pickmode.best_mode_skip_txfm
                         ? rd_threshes[mode_index] << 1
                         : rd_threshes[mode_index];

    // Increase mode_rd_thresh value for GOLDEN_FRAME for improved encoding
    // speed with little/no subjective quality loss.
    if (cpi->sf.bias_golden && ref_frame == GOLDEN_FRAME &&
        cpi->rc.frames_since_golden > 4)
      mode_rd_thresh = mode_rd_thresh << 3;

    if ((cpi->sf.adaptive_rd_thresh_row_mt &&
         rd_less_than_thresh_row_mt(best_rdc.rdcost, mode_rd_thresh,
                                    &rd_thresh_freq_fact[mode_index])) ||
        (!cpi->sf.adaptive_rd_thresh_row_mt &&
         rd_less_than_thresh(best_rdc.rdcost, mode_rd_thresh,
                             &rd_thresh_freq_fact[mode_index])))
      if (frame_mv[this_mode][ref_frame].as_int != 0) continue;

    if (this_mode == NEWMV && !force_mv_inter_layer) {
      if (search_new_mv(cpi, x, frame_mv, ref_frame, gf_temporal_ref, bsize,
                        mi_row, mi_col, best_pred_sad, &rate_mv, best_sse_sofar,
                        &best_rdc))
        continue;
    }

    // TODO(jianj): Skipping the testing of (duplicate) non-zero motion vector
    // causes some regression, leave it for duplicate zero-mv for now, until
    // regression issue is resolved.
    for (inter_mv_mode = NEARESTMV; inter_mv_mode <= NEWMV; inter_mv_mode++) {
      if (inter_mv_mode == this_mode || comp_pred) continue;
      if (mode_checked[inter_mv_mode][ref_frame] &&
          frame_mv[this_mode][ref_frame].as_int ==
              frame_mv[inter_mv_mode][ref_frame].as_int &&
          frame_mv[inter_mv_mode][ref_frame].as_int == 0) {
        skip_this_mv = 1;
        break;
      }
    }

    if (skip_this_mv) continue;

    // If use_golden_nonzeromv is false, NEWMV mode is skipped for golden, no
    // need to compute best_pred_sad which is only used to skip golden NEWMV.
    if (use_golden_nonzeromv && this_mode == NEWMV && ref_frame == LAST_FRAME &&
        frame_mv[NEWMV][LAST_FRAME].as_int != INVALID_MV) {
      const int pre_stride = xd->plane[0].pre[0].stride;
      const uint8_t *const pre_buf =
          xd->plane[0].pre[0].buf +
          (frame_mv[NEWMV][LAST_FRAME].as_mv.row >> 3) * pre_stride +
          (frame_mv[NEWMV][LAST_FRAME].as_mv.col >> 3);
      best_pred_sad = cpi->fn_ptr[bsize].sdf(
          x->plane[0].src.buf, x->plane[0].src.stride, pre_buf, pre_stride);
      x->pred_mv_sad[LAST_FRAME] = best_pred_sad;
    }

    if (this_mode != NEARESTMV && !comp_pred &&
        frame_mv[this_mode][ref_frame].as_int ==
            frame_mv[NEARESTMV][ref_frame].as_int)
      continue;

    mi->mode = this_mode;
    mi->mv[0].as_int = frame_mv[this_mode][ref_frame].as_int;
    mi->mv[1].as_int = 0;

    // Search for the best prediction filter type, when the resulting
    // motion vector is at sub-pixel accuracy level for luma component, i.e.,
    // the last three bits are all zeros.
    if (reuse_inter_pred) {
      if (!this_mode_pred) {
        this_mode_pred = &tmp[3];
      } else {
        this_mode_pred = &tmp[get_pred_buffer(tmp, 3)];
        pd->dst.buf = this_mode_pred->data;
        pd->dst.stride = bw;
      }
    }

    if ((this_mode == NEWMV || filter_ref == SWITCHABLE) &&
        pred_filter_search &&
        (ref_frame == LAST_FRAME ||
         (ref_frame == GOLDEN_FRAME && !force_mv_inter_layer &&
          (cpi->use_svc || cpi->oxcf.rc_mode == VPX_VBR))) &&
        (((mi->mv[0].as_mv.row | mi->mv[0].as_mv.col) & 0x07) != 0)) {
      rd_computed = 1;
      search_filter_ref(cpi, x, &this_rdc, mi_row, mi_col, tmp, bsize,
                        reuse_inter_pred, &this_mode_pred, &var_y, &sse_y,
                        force_smooth_filter, &this_early_term,
                        flag_preduv_computed, use_model_yrd_large);
    } else {
      mi->interp_filter = (filter_ref == SWITCHABLE) ? EIGHTTAP : filter_ref;

      if (cpi->use_svc && ref_frame == GOLDEN_FRAME &&
          svc_force_zero_mode[ref_frame - 1])
        mi->interp_filter = filter_gf_svc;

      vp9_build_inter_predictors_sby(xd, mi_row, mi_col, bsize);

      // For large partition blocks, extra testing is done.
      if (use_model_yrd_large) {
        rd_computed = 1;
        model_rd_for_sb_y_large(cpi, bsize, x, xd, &this_rdc.rate,
                                &this_rdc.dist, &var_y, &sse_y, mi_row, mi_col,
                                &this_early_term, flag_preduv_computed);
      } else {
        rd_computed = 1;
        model_rd_for_sb_y(cpi, bsize, x, xd, &this_rdc.rate, &this_rdc.dist,
                          &var_y, &sse_y, 0);
      }
      // Save normalized sse (between current and last frame) for (0, 0) motion.
      if (ref_frame == LAST_FRAME &&
          frame_mv[this_mode][ref_frame].as_int == 0) {
        sse_zeromv_normalized =
            sse_y >> (b_width_log2_lookup[bsize] + b_height_log2_lookup[bsize]);
      }
      if (sse_y < best_sse_sofar) best_sse_sofar = sse_y;
    }

    if (!this_early_term) {
      this_sse = (int64_t)sse_y;
      block_yrd(cpi, x, &this_rdc, &is_skippable, &this_sse, bsize,
                VPXMIN(mi->tx_size, TX_16X16), rd_computed, 0);
      x->skip_txfm[0] = is_skippable;
      if (is_skippable) {
        this_rdc.rate = vp9_cost_bit(vp9_get_skip_prob(cm, xd), 1);
      } else {
        if (RDCOST(x->rdmult, x->rddiv, this_rdc.rate, this_rdc.dist) <
            RDCOST(x->rdmult, x->rddiv, 0, this_sse)) {
          this_rdc.rate += vp9_cost_bit(vp9_get_skip_prob(cm, xd), 0);
        } else {
          this_rdc.rate = vp9_cost_bit(vp9_get_skip_prob(cm, xd), 1);
          this_rdc.dist = this_sse;
          x->skip_txfm[0] = SKIP_TXFM_AC_DC;
        }
      }

      if (cm->interp_filter == SWITCHABLE) {
        if ((mi->mv[0].as_mv.row | mi->mv[0].as_mv.col) & 0x07)
          this_rdc.rate += vp9_get_switchable_rate(cpi, xd);
      }
    } else {
      if (cm->interp_filter == SWITCHABLE) {
        if ((mi->mv[0].as_mv.row | mi->mv[0].as_mv.col) & 0x07)
          this_rdc.rate += vp9_get_switchable_rate(cpi, xd);
      }
      this_rdc.rate += vp9_cost_bit(vp9_get_skip_prob(cm, xd), 1);
    }

    if (!this_early_term &&
        (x->color_sensitivity[0] || x->color_sensitivity[1])) {
      RD_COST rdc_uv;
      const BLOCK_SIZE uv_bsize = get_plane_block_size(bsize, &xd->plane[1]);
      if (x->color_sensitivity[0] && !flag_preduv_computed[0]) {
        vp9_build_inter_predictors_sbp(xd, mi_row, mi_col, bsize, 1);
        flag_preduv_computed[0] = 1;
      }
      if (x->color_sensitivity[1] && !flag_preduv_computed[1]) {
        vp9_build_inter_predictors_sbp(xd, mi_row, mi_col, bsize, 2);
        flag_preduv_computed[1] = 1;
      }
      model_rd_for_sb_uv(cpi, uv_bsize, x, xd, &rdc_uv, &var_y, &sse_y, 1, 2);
      this_rdc.rate += rdc_uv.rate;
      this_rdc.dist += rdc_uv.dist;
    }

    this_rdc.rate += rate_mv;
    this_rdc.rate += cpi->inter_mode_cost[x->mbmi_ext->mode_context[ref_frame]]
                                         [INTER_OFFSET(this_mode)];
    // TODO(marpan): Add costing for compound mode.
    this_rdc.rate += ref_frame_cost[ref_frame];
    this_rdc.rdcost = RDCOST(x->rdmult, x->rddiv, this_rdc.rate, this_rdc.dist);

    // Bias against NEWMV that is very different from its neighbors, and bias
    // to small motion-lastref for noisy input.
    if (cpi->oxcf.rc_mode == VPX_CBR && cpi->oxcf.speed >= 5 &&
        cpi->oxcf.content != VP9E_CONTENT_SCREEN) {
      vp9_NEWMV_diff_bias(&cpi->noise_estimate, xd, this_mode, &this_rdc, bsize,
                          frame_mv[this_mode][ref_frame].as_mv.row,
                          frame_mv[this_mode][ref_frame].as_mv.col,
                          ref_frame == LAST_FRAME, x->lowvar_highsumdiff,
                          x->sb_is_skin);
    }

    // Skipping checking: test to see if this block can be reconstructed by
    // prediction only.
    if (cpi->allow_encode_breakout && !xd->lossless && !scene_change_detected &&
        !svc->high_num_blocks_with_motion) {
      encode_breakout_test(cpi, x, bsize, mi_row, mi_col, ref_frame, this_mode,
                           var_y, sse_y, yv12_mb, &this_rdc.rate,
                           &this_rdc.dist, flag_preduv_computed);
      if (x->skip) {
        this_rdc.rate += rate_mv;
        this_rdc.rdcost =
            RDCOST(x->rdmult, x->rddiv, this_rdc.rate, this_rdc.dist);
      }
    }

    // On spatially flat blocks for screne content: bias against zero-last
    // if the sse_y is non-zero. Only on scene change or high motion frames.
    if (cpi->oxcf.content == VP9E_CONTENT_SCREEN &&
        (scene_change_detected || svc->high_num_blocks_with_motion) &&
        ref_frame == LAST_FRAME && frame_mv[this_mode][ref_frame].as_int == 0 &&
        svc->spatial_layer_id == 0 && x->source_variance == 0 && sse_y > 0) {
      this_rdc.rdcost = this_rdc.rdcost << 2;
    }

#if CONFIG_VP9_TEMPORAL_DENOISING
    if (cpi->oxcf.noise_sensitivity > 0 && denoise_svc_pickmode &&
        cpi->denoiser.denoising_level > kDenLowLow) {
      vp9_denoiser_update_frame_stats(mi, sse_y, this_mode, ctx);
      // Keep track of zero_last cost.
      if (ref_frame == LAST_FRAME && frame_mv[this_mode][ref_frame].as_int == 0)
        zero_last_cost_orig = this_rdc.rdcost;
    }
#else
    (void)ctx;
#endif

    mode_checked[this_mode][ref_frame] = 1;

    if (this_rdc.rdcost < best_rdc.rdcost || x->skip) {
      best_rdc = this_rdc;
      best_early_term = this_early_term;
      best_pickmode.best_mode = this_mode;
      best_pickmode.best_pred_filter = mi->interp_filter;
      best_pickmode.best_tx_size = mi->tx_size;
      best_pickmode.best_ref_frame = ref_frame;
      best_pickmode.best_mode_skip_txfm = x->skip_txfm[0];
      best_pickmode.best_second_ref_frame = second_ref_frame;

      if (reuse_inter_pred) {
        free_pred_buffer(best_pickmode.best_pred);
        best_pickmode.best_pred = this_mode_pred;
      }
    } else {
      if (reuse_inter_pred) free_pred_buffer(this_mode_pred);
    }

    if (x->skip &&
        (!force_test_gf_zeromv || mode_checked[ZEROMV][GOLDEN_FRAME]))
      break;

    // If early termination flag is 1 and at least 2 modes are checked,
    // the mode search is terminated.
    if (best_early_term && idx > 0 && !scene_change_detected &&
        (!force_test_gf_zeromv || mode_checked[ZEROMV][GOLDEN_FRAME])) {
      x->skip = 1;
      break;
    }
  }

  mi->mode = best_pickmode.best_mode;
  mi->interp_filter = best_pickmode.best_pred_filter;
  mi->tx_size = best_pickmode.best_tx_size;
  mi->ref_frame[0] = best_pickmode.best_ref_frame;
  mi->mv[0].as_int =
      frame_mv[best_pickmode.best_mode][best_pickmode.best_ref_frame].as_int;
  xd->mi[0]->bmi[0].as_mv[0].as_int = mi->mv[0].as_int;
  x->skip_txfm[0] = best_pickmode.best_mode_skip_txfm;
  mi->ref_frame[1] = best_pickmode.best_second_ref_frame;

  // For spatial enhancemanent layer: perform intra prediction only if base
  // layer is chosen as the reference. Always perform intra prediction if
  // LAST is the only reference, or is_key_frame is set, or on base
  // temporal layer.
  if (svc->spatial_layer_id && !gf_temporal_ref) {
    perform_intra_pred =
        svc->temporal_layer_id == 0 ||
        svc->layer_context[svc->temporal_layer_id].is_key_frame ||
        !(cpi->ref_frame_flags & VP9_GOLD_FLAG) ||
        (!svc->layer_context[svc->temporal_layer_id].is_key_frame &&
         svc_force_zero_mode[best_pickmode.best_ref_frame - 1]);
    inter_mode_thresh = (inter_mode_thresh << 1) + inter_mode_thresh;
  }
  if ((cpi->oxcf.lag_in_frames > 0 && cpi->oxcf.rc_mode == VPX_VBR &&
       cpi->rc.is_src_frame_alt_ref) ||
      svc->previous_frame_is_intra_only)
    perform_intra_pred = 0;

  // If the segment reference frame feature is enabled and set then
  // skip the intra prediction.
  if (segfeature_active(seg, mi->segment_id, SEG_LVL_REF_FRAME) &&
      get_segdata(seg, mi->segment_id, SEG_LVL_REF_FRAME) > 0)
    perform_intra_pred = 0;

  // Perform intra prediction search, if the best SAD is above a certain
  // threshold.
  if (best_rdc.rdcost == INT64_MAX ||
      (cpi->oxcf.content == VP9E_CONTENT_SCREEN && x->source_variance == 0) ||
      (scene_change_detected && perform_intra_pred) ||
      ((!force_skip_low_temp_var || bsize < BLOCK_32X32 ||
        x->content_state_sb == kVeryHighSad) &&
       perform_intra_pred && !x->skip && best_rdc.rdcost > inter_mode_thresh &&
       bsize <= cpi->sf.max_intra_bsize && !x->skip_low_source_sad &&
       !x->lowvar_highsumdiff)) {
    struct estimate_block_intra_args args = { cpi, x, DC_PRED, 1, 0 };
    int64_t this_sse = INT64_MAX;
    int i;
    PRED_BUFFER *const best_pred = best_pickmode.best_pred;
    TX_SIZE intra_tx_size =
        VPXMIN(max_txsize_lookup[bsize],
               tx_mode_to_biggest_tx_size[cpi->common.tx_mode]);

    if (reuse_inter_pred && best_pred != NULL) {
      if (best_pred->data == orig_dst.buf) {
        this_mode_pred = &tmp[get_pred_buffer(tmp, 3)];
#if CONFIG_VP9_HIGHBITDEPTH
        if (cm->use_highbitdepth)
          vpx_highbd_convolve_copy(
              CONVERT_TO_SHORTPTR(best_pred->data), best_pred->stride,
              CONVERT_TO_SHORTPTR(this_mode_pred->data), this_mode_pred->stride,
              NULL, 0, 0, 0, 0, bw, bh, xd->bd);
        else
          vpx_convolve_copy(best_pred->data, best_pred->stride,
                            this_mode_pred->data, this_mode_pred->stride, NULL,
                            0, 0, 0, 0, bw, bh);
#else
        vpx_convolve_copy(best_pred->data, best_pred->stride,
                          this_mode_pred->data, this_mode_pred->stride, NULL, 0,
                          0, 0, 0, bw, bh);
#endif  // CONFIG_VP9_HIGHBITDEPTH
        best_pickmode.best_pred = this_mode_pred;
      }
    }
    pd->dst = orig_dst;

    for (i = 0; i < 4; ++i) {
      const PREDICTION_MODE this_mode = intra_mode_list[i];
      THR_MODES mode_index = mode_idx[INTRA_FRAME][mode_offset(this_mode)];
      int mode_rd_thresh = rd_threshes[mode_index];
      // For spatially flat blocks, under short_circuit_flat_blocks flag:
      // only check DC mode for stationary blocks, otherwise also check
      // H and V mode.
      if (sf->short_circuit_flat_blocks && x->source_variance == 0 &&
          ((x->zero_temp_sad_source && this_mode != DC_PRED) || i > 2)) {
        continue;
      }

      if (!((1 << this_mode) & cpi->sf.intra_y_mode_bsize_mask[bsize]))
        continue;

      if (cpi->sf.rt_intra_dc_only_low_content && this_mode != DC_PRED &&
          x->content_state_sb != kVeryHighSad)
        continue;

      if ((cpi->sf.adaptive_rd_thresh_row_mt &&
           rd_less_than_thresh_row_mt(best_rdc.rdcost, mode_rd_thresh,
                                      &rd_thresh_freq_fact[mode_index])) ||
          (!cpi->sf.adaptive_rd_thresh_row_mt &&
           rd_less_than_thresh(best_rdc.rdcost, mode_rd_thresh,
                               &rd_thresh_freq_fact[mode_index]))) {
        // Avoid this early exit for screen on base layer, for scene
        // changes or high motion frames.
        if (cpi->oxcf.content != VP9E_CONTENT_SCREEN ||
            svc->spatial_layer_id > 0 ||
            (!scene_change_detected && !svc->high_num_blocks_with_motion))
          continue;
      }

      mi->mode = this_mode;
      mi->ref_frame[0] = INTRA_FRAME;
      this_rdc.dist = this_rdc.rate = 0;
      args.mode = this_mode;
      args.skippable = 1;
      args.rdc = &this_rdc;
      mi->tx_size = intra_tx_size;

      compute_intra_yprediction(this_mode, bsize, x, xd);
      model_rd_for_sb_y(cpi, bsize, x, xd, &this_rdc.rate, &this_rdc.dist,
                        &var_y, &sse_y, 1);
      block_yrd(cpi, x, &this_rdc, &args.skippable, &this_sse, bsize,
                VPXMIN(mi->tx_size, TX_16X16), 1, 1);

      // Check skip cost here since skippable is not set for for uv, this
      // mirrors the behavior used by inter
      if (args.skippable) {
        x->skip_txfm[0] = SKIP_TXFM_AC_DC;
        this_rdc.rate = vp9_cost_bit(vp9_get_skip_prob(&cpi->common, xd), 1);
      } else {
        x->skip_txfm[0] = SKIP_TXFM_NONE;
        this_rdc.rate += vp9_cost_bit(vp9_get_skip_prob(&cpi->common, xd), 0);
      }
      // Inter and intra RD will mismatch in scale for non-screen content.
      if (cpi->oxcf.content == VP9E_CONTENT_SCREEN) {
        if (x->color_sensitivity[0])
          vp9_foreach_transformed_block_in_plane(xd, bsize, 1,
                                                 estimate_block_intra, &args);
        if (x->color_sensitivity[1])
          vp9_foreach_transformed_block_in_plane(xd, bsize, 2,
                                                 estimate_block_intra, &args);
      }
      this_rdc.rate += cpi->mbmode_cost[this_mode];
      this_rdc.rate += ref_frame_cost[INTRA_FRAME];
      this_rdc.rate += intra_cost_penalty;
      this_rdc.rdcost =
          RDCOST(x->rdmult, x->rddiv, this_rdc.rate, this_rdc.dist);

      if (this_rdc.rdcost < best_rdc.rdcost) {
        best_rdc = this_rdc;
        best_pickmode.best_mode = this_mode;
        best_pickmode.best_intra_tx_size = mi->tx_size;
        best_pickmode.best_ref_frame = INTRA_FRAME;
        best_pickmode.best_second_ref_frame = NO_REF_FRAME;
        mi->uv_mode = this_mode;
        mi->mv[0].as_int = INVALID_MV;
        mi->mv[1].as_int = INVALID_MV;
        best_pickmode.best_mode_skip_txfm = x->skip_txfm[0];
      }
    }

    // Reset mb_mode_info to the best inter mode.
    if (best_pickmode.best_ref_frame != INTRA_FRAME) {
      mi->tx_size = best_pickmode.best_tx_size;
    } else {
      mi->tx_size = best_pickmode.best_intra_tx_size;
    }
  }

  pd->dst = orig_dst;
  mi->mode = best_pickmode.best_mode;
  mi->ref_frame[0] = best_pickmode.best_ref_frame;
  mi->ref_frame[1] = best_pickmode.best_second_ref_frame;
  x->skip_txfm[0] = best_pickmode.best_mode_skip_txfm;

  if (!is_inter_block(mi)) {
    mi->interp_filter = SWITCHABLE_FILTERS;
  }

  if (reuse_inter_pred && best_pickmode.best_pred != NULL) {
    PRED_BUFFER *const best_pred = best_pickmode.best_pred;
    if (best_pred->data != orig_dst.buf && is_inter_mode(mi->mode)) {
#if CONFIG_VP9_HIGHBITDEPTH
      if (cm->use_highbitdepth)
        vpx_highbd_convolve_copy(
            CONVERT_TO_SHORTPTR(best_pred->data), best_pred->stride,
            CONVERT_TO_SHORTPTR(pd->dst.buf), pd->dst.stride, NULL, 0, 0, 0, 0,
            bw, bh, xd->bd);
      else
        vpx_convolve_copy(best_pred->data, best_pred->stride, pd->dst.buf,
                          pd->dst.stride, NULL, 0, 0, 0, 0, bw, bh);
#else
      vpx_convolve_copy(best_pred->data, best_pred->stride, pd->dst.buf,
                        pd->dst.stride, NULL, 0, 0, 0, 0, bw, bh);
#endif  // CONFIG_VP9_HIGHBITDEPTH
    }
  }

#if CONFIG_VP9_TEMPORAL_DENOISING
  if (cpi->oxcf.noise_sensitivity > 0 && cpi->resize_pending == 0 &&
      denoise_svc_pickmode && cpi->denoiser.denoising_level > kDenLowLow &&
      cpi->denoiser.reset == 0) {
    VP9_DENOISER_DECISION decision = COPY_BLOCK;
    ctx->sb_skip_denoising = 0;
    // TODO(marpan): There is an issue with denoising when the
    // superblock partitioning scheme is based on the pickmode.
    // Remove this condition when the issue is resolved.
    if (x->sb_pickmode_part) ctx->sb_skip_denoising = 1;
    vp9_pickmode_ctx_den_update(&ctx_den, zero_last_cost_orig, ref_frame_cost,
                                frame_mv, reuse_inter_pred, &best_pickmode);
    vp9_denoiser_denoise(cpi, x, mi_row, mi_col, bsize, ctx, &decision,
                         gf_temporal_ref);
    if (denoise_recheck_zeromv)
      recheck_zeromv_after_denoising(cpi, mi, x, xd, decision, &ctx_den,
                                     yv12_mb, &best_rdc, bsize, mi_row, mi_col);
    best_pickmode.best_ref_frame = ctx_den.best_ref_frame;
  }
#endif

  if (best_pickmode.best_ref_frame == ALTREF_FRAME ||
      best_pickmode.best_second_ref_frame == ALTREF_FRAME)
    x->arf_frame_usage++;
  else if (best_pickmode.best_ref_frame != INTRA_FRAME)
    x->lastgolden_frame_usage++;

  if (cpi->sf.adaptive_rd_thresh) {
    THR_MODES best_mode_idx =
        mode_idx[best_pickmode.best_ref_frame][mode_offset(mi->mode)];

    if (best_pickmode.best_ref_frame == INTRA_FRAME) {
      // Only consider the modes that are included in the intra_mode_list.
      int intra_modes = sizeof(intra_mode_list) / sizeof(PREDICTION_MODE);
      int i;

      // TODO(yunqingwang): Check intra mode mask and only update freq_fact
      // for those valid modes.
      for (i = 0; i < intra_modes; i++) {
        if (cpi->sf.adaptive_rd_thresh_row_mt)
          update_thresh_freq_fact_row_mt(cpi, tile_data, x->source_variance,
                                         thresh_freq_fact_idx, INTRA_FRAME,
                                         best_mode_idx, intra_mode_list[i]);
        else
          update_thresh_freq_fact(cpi, tile_data, x->source_variance, bsize,
                                  INTRA_FRAME, best_mode_idx,
                                  intra_mode_list[i]);
      }
    } else {
      for (ref_frame = LAST_FRAME; ref_frame <= GOLDEN_FRAME; ++ref_frame) {
        PREDICTION_MODE this_mode;
        if (best_pickmode.best_ref_frame != ref_frame) continue;
        for (this_mode = NEARESTMV; this_mode <= NEWMV; ++this_mode) {
          if (cpi->sf.adaptive_rd_thresh_row_mt)
            update_thresh_freq_fact_row_mt(cpi, tile_data, x->source_variance,
                                           thresh_freq_fact_idx, ref_frame,
                                           best_mode_idx, this_mode);
          else
            update_thresh_freq_fact(cpi, tile_data, x->source_variance, bsize,
                                    ref_frame, best_mode_idx, this_mode);
        }
      }
    }
  }

  *rd_cost = best_rdc;
}

void vp9_pick_inter_mode_sub8x8(VP9_COMP *cpi, MACROBLOCK *x, int mi_row,
                                int mi_col, RD_COST *rd_cost, BLOCK_SIZE bsize,
                                PICK_MODE_CONTEXT *ctx) {
  VP9_COMMON *const cm = &cpi->common;
  SPEED_FEATURES *const sf = &cpi->sf;
  MACROBLOCKD *const xd = &x->e_mbd;
  MODE_INFO *const mi = xd->mi[0];
  MB_MODE_INFO_EXT *const mbmi_ext = x->mbmi_ext;
  const struct segmentation *const seg = &cm->seg;
  MV_REFERENCE_FRAME ref_frame, second_ref_frame = NO_REF_FRAME;
  MV_REFERENCE_FRAME best_ref_frame = NO_REF_FRAME;
  unsigned char segment_id = mi->segment_id;
  struct buf_2d yv12_mb[4][MAX_MB_PLANE];
  int64_t best_rd = INT64_MAX;
  b_mode_info bsi[MAX_REF_FRAMES][4];
  int ref_frame_skip_mask = 0;
  const int num_4x4_blocks_wide = num_4x4_blocks_wide_lookup[bsize];
  const int num_4x4_blocks_high = num_4x4_blocks_high_lookup[bsize];
  int idx, idy;

  x->skip_encode = sf->skip_encode_frame && x->q_index < QIDX_SKIP_THRESH;
  ctx->pred_pixel_ready = 0;

  for (ref_frame = LAST_FRAME; ref_frame <= GOLDEN_FRAME; ++ref_frame) {
    const YV12_BUFFER_CONFIG *yv12 = get_ref_frame_buffer(cpi, ref_frame);
    int_mv dummy_mv[2];
    x->pred_mv_sad[ref_frame] = INT_MAX;

    if ((cpi->ref_frame_flags & ref_frame_to_flag(ref_frame)) &&
        (yv12 != NULL)) {
      int_mv *const candidates = mbmi_ext->ref_mvs[ref_frame];
      const struct scale_factors *const ref_sf =
          &cm->frame_refs[ref_frame - 1].sf;
      vp9_setup_pred_block(xd, yv12_mb[ref_frame], yv12, mi_row, mi_col, ref_sf,
                           ref_sf);
      vp9_find_mv_refs(cm, xd, xd->mi[0], ref_frame, candidates, mi_row, mi_col,
                       mbmi_ext->mode_context);

      vp9_find_best_ref_mvs(xd, cm->allow_high_precision_mv, candidates,
                            &dummy_mv[0], &dummy_mv[1]);
    } else {
      ref_frame_skip_mask |= (1 << ref_frame);
    }
  }

  mi->sb_type = bsize;
  mi->tx_size = TX_4X4;
  mi->uv_mode = DC_PRED;
  mi->ref_frame[0] = LAST_FRAME;
  mi->ref_frame[1] = NO_REF_FRAME;
  mi->interp_filter =
      cm->interp_filter == SWITCHABLE ? EIGHTTAP : cm->interp_filter;

  for (ref_frame = LAST_FRAME; ref_frame <= GOLDEN_FRAME; ++ref_frame) {
    int64_t this_rd = 0;
    int plane;

    if (ref_frame_skip_mask & (1 << ref_frame)) continue;

#if CONFIG_BETTER_HW_COMPATIBILITY
    if ((bsize == BLOCK_8X4 || bsize == BLOCK_4X8) && ref_frame > INTRA_FRAME &&
        vp9_is_scaled(&cm->frame_refs[ref_frame - 1].sf))
      continue;
#endif

    // TODO(jingning, agrange): Scaling reference frame not supported for
    // sub8x8 blocks. Is this supported now?
    if (ref_frame > INTRA_FRAME &&
        vp9_is_scaled(&cm->frame_refs[ref_frame - 1].sf))
      continue;

    // If the segment reference frame feature is enabled....
    // then do nothing if the current ref frame is not allowed..
    if (segfeature_active(seg, segment_id, SEG_LVL_REF_FRAME) &&
        get_segdata(seg, segment_id, SEG_LVL_REF_FRAME) != (int)ref_frame)
      continue;

    mi->ref_frame[0] = ref_frame;
    x->skip = 0;
    set_ref_ptrs(cm, xd, ref_frame, second_ref_frame);

    // Select prediction reference frames.
    for (plane = 0; plane < MAX_MB_PLANE; plane++)
      xd->plane[plane].pre[0] = yv12_mb[ref_frame][plane];

    for (idy = 0; idy < 2; idy += num_4x4_blocks_high) {
      for (idx = 0; idx < 2; idx += num_4x4_blocks_wide) {
        int_mv b_mv[MB_MODE_COUNT];
        int64_t b_best_rd = INT64_MAX;
        const int i = idy * 2 + idx;
        PREDICTION_MODE this_mode;
        RD_COST this_rdc;
        unsigned int var_y, sse_y;

        struct macroblock_plane *p = &x->plane[0];
        struct macroblockd_plane *pd = &xd->plane[0];

        const struct buf_2d orig_src = p->src;
        const struct buf_2d orig_dst = pd->dst;
        struct buf_2d orig_pre[2];
        memcpy(orig_pre, xd->plane[0].pre, sizeof(orig_pre));

        // set buffer pointers for sub8x8 motion search.
        p->src.buf =
            &p->src.buf[vp9_raster_block_offset(BLOCK_8X8, i, p->src.stride)];
        pd->dst.buf =
            &pd->dst.buf[vp9_raster_block_offset(BLOCK_8X8, i, pd->dst.stride)];
        pd->pre[0].buf =
            &pd->pre[0]
                 .buf[vp9_raster_block_offset(BLOCK_8X8, i, pd->pre[0].stride)];

        b_mv[ZEROMV].as_int = 0;
        b_mv[NEWMV].as_int = INVALID_MV;
        vp9_append_sub8x8_mvs_for_idx(cm, xd, i, 0, mi_row, mi_col,
                                      &b_mv[NEARESTMV], &b_mv[NEARMV],
                                      mbmi_ext->mode_context);

        for (this_mode = NEARESTMV; this_mode <= NEWMV; ++this_mode) {
          int b_rate = 0;
          xd->mi[0]->bmi[i].as_mv[0].as_int = b_mv[this_mode].as_int;

          if (this_mode == NEWMV) {
            const int step_param = cpi->sf.mv.fullpel_search_step_param;
            MV mvp_full;
            MV tmp_mv;
            int cost_list[5];
            const MvLimits tmp_mv_limits = x->mv_limits;
            uint32_t dummy_dist;

            if (i == 0) {
              mvp_full.row = b_mv[NEARESTMV].as_mv.row >> 3;
              mvp_full.col = b_mv[NEARESTMV].as_mv.col >> 3;
            } else {
              mvp_full.row = xd->mi[0]->bmi[0].as_mv[0].as_mv.row >> 3;
              mvp_full.col = xd->mi[0]->bmi[0].as_mv[0].as_mv.col >> 3;
            }

            vp9_set_mv_search_range(&x->mv_limits,
                                    &mbmi_ext->ref_mvs[ref_frame][0].as_mv);

            vp9_full_pixel_search(
                cpi, x, bsize, &mvp_full, step_param, cpi->sf.mv.search_method,
                x->sadperbit4, cond_cost_list(cpi, cost_list),
                &mbmi_ext->ref_mvs[ref_frame][0].as_mv, &tmp_mv, INT_MAX, 0);

            x->mv_limits = tmp_mv_limits;

            // calculate the bit cost on motion vector
            mvp_full.row = tmp_mv.row * 8;
            mvp_full.col = tmp_mv.col * 8;

            b_rate += vp9_mv_bit_cost(
                &mvp_full, &mbmi_ext->ref_mvs[ref_frame][0].as_mv,
                x->nmvjointcost, x->mvcost, MV_COST_WEIGHT);

            b_rate += cpi->inter_mode_cost[x->mbmi_ext->mode_context[ref_frame]]
                                          [INTER_OFFSET(NEWMV)];
            if (RDCOST(x->rdmult, x->rddiv, b_rate, 0) > b_best_rd) continue;

            cpi->find_fractional_mv_step(
                x, &tmp_mv, &mbmi_ext->ref_mvs[ref_frame][0].as_mv,
                cpi->common.allow_high_precision_mv, x->errorperbit,
                &cpi->fn_ptr[bsize], cpi->sf.mv.subpel_force_stop,
                cpi->sf.mv.subpel_search_level, cond_cost_list(cpi, cost_list),
                x->nmvjointcost, x->mvcost, &dummy_dist,
                &x->pred_sse[ref_frame], NULL, 0, 0,
                cpi->sf.use_accurate_subpel_search);

            xd->mi[0]->bmi[i].as_mv[0].as_mv = tmp_mv;
          } else {
            b_rate += cpi->inter_mode_cost[x->mbmi_ext->mode_context[ref_frame]]
                                          [INTER_OFFSET(this_mode)];
          }

#if CONFIG_VP9_HIGHBITDEPTH
          if (xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH) {
            vp9_highbd_build_inter_predictor(
                CONVERT_TO_SHORTPTR(pd->pre[0].buf), pd->pre[0].stride,
                CONVERT_TO_SHORTPTR(pd->dst.buf), pd->dst.stride,
                &xd->mi[0]->bmi[i].as_mv[0].as_mv, &xd->block_refs[0]->sf,
                4 * num_4x4_blocks_wide, 4 * num_4x4_blocks_high, 0,
                vp9_filter_kernels[mi->interp_filter], MV_PRECISION_Q3,
                mi_col * MI_SIZE + 4 * (i & 0x01),
                mi_row * MI_SIZE + 4 * (i >> 1), xd->bd);
          } else {
#endif
            vp9_build_inter_predictor(
                pd->pre[0].buf, pd->pre[0].stride, pd->dst.buf, pd->dst.stride,
                &xd->mi[0]->bmi[i].as_mv[0].as_mv, &xd->block_refs[0]->sf,
                4 * num_4x4_blocks_wide, 4 * num_4x4_blocks_high, 0,
                vp9_filter_kernels[mi->interp_filter], MV_PRECISION_Q3,
                mi_col * MI_SIZE + 4 * (i & 0x01),
                mi_row * MI_SIZE + 4 * (i >> 1));

#if CONFIG_VP9_HIGHBITDEPTH
          }
#endif

          model_rd_for_sb_y(cpi, bsize, x, xd, &this_rdc.rate, &this_rdc.dist,
                            &var_y, &sse_y, 0);

          this_rdc.rate += b_rate;
          this_rdc.rdcost =
              RDCOST(x->rdmult, x->rddiv, this_rdc.rate, this_rdc.dist);
          if (this_rdc.rdcost < b_best_rd) {
            b_best_rd = this_rdc.rdcost;
            bsi[ref_frame][i].as_mode = this_mode;
            bsi[ref_frame][i].as_mv[0].as_mv = xd->mi[0]->bmi[i].as_mv[0].as_mv;
          }
        }  // mode search

        // restore source and prediction buffer pointers.
        p->src = orig_src;
        pd->pre[0] = orig_pre[0];
        pd->dst = orig_dst;
        this_rd += b_best_rd;

        xd->mi[0]->bmi[i] = bsi[ref_frame][i];
        if (num_4x4_blocks_wide > 1) xd->mi[0]->bmi[i + 1] = xd->mi[0]->bmi[i];
        if (num_4x4_blocks_high > 1) xd->mi[0]->bmi[i + 2] = xd->mi[0]->bmi[i];
      }
    }  // loop through sub8x8 blocks

    if (this_rd < best_rd) {
      best_rd = this_rd;
      best_ref_frame = ref_frame;
    }
  }  // reference frames

  mi->tx_size = TX_4X4;
  mi->ref_frame[0] = best_ref_frame;
  for (idy = 0; idy < 2; idy += num_4x4_blocks_high) {
    for (idx = 0; idx < 2; idx += num_4x4_blocks_wide) {
      const int block = idy * 2 + idx;
      xd->mi[0]->bmi[block] = bsi[best_ref_frame][block];
      if (num_4x4_blocks_wide > 1)
        xd->mi[0]->bmi[block + 1] = bsi[best_ref_frame][block];
      if (num_4x4_blocks_high > 1)
        xd->mi[0]->bmi[block + 2] = bsi[best_ref_frame][block];
    }
  }
  mi->mode = xd->mi[0]->bmi[3].as_mode;
  ctx->mic = *(xd->mi[0]);
  ctx->mbmi_ext = *x->mbmi_ext;
  ctx->skip_txfm[0] = SKIP_TXFM_NONE;
  ctx->skip = 0;
  // Dummy assignment for speed -5. No effect in speed -6.
  rd_cost->rdcost = best_rd;
}
