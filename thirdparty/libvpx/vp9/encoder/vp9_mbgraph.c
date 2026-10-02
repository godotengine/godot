/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <limits.h>

#include "./vp9_rtcd.h"
#include "./vpx_dsp_rtcd.h"

#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_mem/vpx_mem.h"
#include "vpx_ports/system_state.h"
#include "vp9/encoder/vp9_segmentation.h"
#include "vp9/encoder/vp9_mcomp.h"
#include "vp9/common/vp9_blockd.h"
#include "vp9/common/vp9_reconinter.h"
#include "vp9/common/vp9_reconintra.h"

static unsigned int do_16x16_motion_iteration(VP9_COMP *cpi, const MV *ref_mv,
                                              MV *dst_mv, int mb_row,
                                              int mb_col) {
  MACROBLOCK *const x = &cpi->td.mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  MV_SPEED_FEATURES *const mv_sf = &cpi->sf.mv;
  const SEARCH_METHODS old_search_method = mv_sf->search_method;
  const vp9_variance_fn_ptr_t v_fn_ptr = cpi->fn_ptr[BLOCK_16X16];
  const MvLimits tmp_mv_limits = x->mv_limits;
  MV ref_full;
  int cost_list[5];

  // Further step/diamond searches as necessary
  int step_param = mv_sf->reduce_first_step_size;
  step_param = VPXMIN(step_param, MAX_MVSEARCH_STEPS - 2);

  vp9_set_mv_search_range(&x->mv_limits, ref_mv);

  ref_full.col = ref_mv->col >> 3;
  ref_full.row = ref_mv->row >> 3;

  mv_sf->search_method = HEX;
  vp9_full_pixel_search(cpi, x, BLOCK_16X16, &ref_full, step_param,
                        cpi->sf.mv.search_method, x->errorperbit,
                        cond_cost_list(cpi, cost_list), ref_mv, dst_mv, 0, 0);
  mv_sf->search_method = old_search_method;

  /* restore UMV window */
  x->mv_limits = tmp_mv_limits;

  // Try sub-pixel MC
  // if (bestsme > error_thresh && bestsme < INT_MAX)
  {
    uint32_t distortion;
    uint32_t sse;
    // TODO(yunqing): may use higher tap interp filter than 2 taps if needed.
    cpi->find_fractional_mv_step(
        x, dst_mv, ref_mv, cpi->common.allow_high_precision_mv, x->errorperbit,
        &v_fn_ptr, 0, mv_sf->subpel_search_level,
        cond_cost_list(cpi, cost_list), NULL, NULL, &distortion, &sse, NULL, 0,
        0, USE_2_TAPS);
  }

  xd->mi[0]->mode = NEWMV;
  xd->mi[0]->mv[0].as_mv = *dst_mv;

  vp9_build_inter_predictors_sby(xd, mb_row, mb_col, BLOCK_16X16);

  return vpx_sad16x16(x->plane[0].src.buf, x->plane[0].src.stride,
                      xd->plane[0].dst.buf, xd->plane[0].dst.stride);
}

static int do_16x16_motion_search(VP9_COMP *cpi, const MV *ref_mv,
                                  int_mv *dst_mv, int mb_row, int mb_col) {
  MACROBLOCK *const x = &cpi->td.mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  unsigned int err, tmp_err;
  MV tmp_mv;

  // Try zero MV first
  // FIXME should really use something like near/nearest MV and/or MV prediction
  err = vpx_sad16x16(x->plane[0].src.buf, x->plane[0].src.stride,
                     xd->plane[0].pre[0].buf, xd->plane[0].pre[0].stride);
  dst_mv->as_int = 0;

  // Test last reference frame using the previous best mv as the
  // starting point (best reference) for the search
  tmp_err = do_16x16_motion_iteration(cpi, ref_mv, &tmp_mv, mb_row, mb_col);
  if (tmp_err < err) {
    err = tmp_err;
    dst_mv->as_mv = tmp_mv;
  }

  // If the current best reference mv is not centered on 0,0 then do a 0,0
  // based search as well.
  if (ref_mv->row != 0 || ref_mv->col != 0) {
    MV zero_ref_mv = { 0, 0 };

    tmp_err =
        do_16x16_motion_iteration(cpi, &zero_ref_mv, &tmp_mv, mb_row, mb_col);
    if (tmp_err < err) {
      dst_mv->as_mv = tmp_mv;
      err = tmp_err;
    }
  }

  return err;
}

static int do_16x16_zerozero_search(VP9_COMP *cpi, int_mv *dst_mv) {
  MACROBLOCK *const x = &cpi->td.mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  unsigned int err;

  // Try zero MV first
  // FIXME should really use something like near/nearest MV and/or MV prediction
  err = vpx_sad16x16(x->plane[0].src.buf, x->plane[0].src.stride,
                     xd->plane[0].pre[0].buf, xd->plane[0].pre[0].stride);

  dst_mv->as_int = 0;

  return err;
}
static int find_best_16x16_intra(VP9_COMP *cpi, PREDICTION_MODE *pbest_mode) {
  MACROBLOCK *const x = &cpi->td.mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  PREDICTION_MODE best_mode = -1, mode;
  unsigned int best_err = INT_MAX;

  // calculate SATD for each intra prediction mode;
  // we're intentionally not doing 4x4, we just want a rough estimate
  for (mode = DC_PRED; mode <= TM_PRED; mode++) {
    unsigned int err;

    xd->mi[0]->mode = mode;
    vp9_predict_intra_block(xd, 2, TX_16X16, mode, x->plane[0].src.buf,
                            x->plane[0].src.stride, xd->plane[0].dst.buf,
                            xd->plane[0].dst.stride, 0, 0, 0);
    err = vpx_sad16x16(x->plane[0].src.buf, x->plane[0].src.stride,
                       xd->plane[0].dst.buf, xd->plane[0].dst.stride);

    // find best
    if (err < best_err) {
      best_err = err;
      best_mode = mode;
    }
  }

  if (pbest_mode) *pbest_mode = best_mode;

  return best_err;
}

static void update_mbgraph_mb_stats(VP9_COMP *cpi, MBGRAPH_MB_STATS *stats,
                                    YV12_BUFFER_CONFIG *buf, int mb_y_offset,
                                    YV12_BUFFER_CONFIG *golden_ref,
                                    const MV *prev_golden_ref_mv,
                                    YV12_BUFFER_CONFIG *alt_ref, int mb_row,
                                    int mb_col) {
  MACROBLOCK *const x = &cpi->td.mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  int intra_error;
  VP9_COMMON *cm = &cpi->common;

  // FIXME in practice we're completely ignoring chroma here
  x->plane[0].src.buf = buf->y_buffer + mb_y_offset;
  x->plane[0].src.stride = buf->y_stride;

  xd->plane[0].dst.buf = get_frame_new_buffer(cm)->y_buffer + mb_y_offset;
  xd->plane[0].dst.stride = get_frame_new_buffer(cm)->y_stride;

  // do intra 16x16 prediction
  intra_error = find_best_16x16_intra(cpi, &stats->ref[INTRA_FRAME].m.mode);
  if (intra_error <= 0) intra_error = 1;
  stats->ref[INTRA_FRAME].err = intra_error;

  // Golden frame MV search, if it exists and is different than last frame
  if (golden_ref) {
    int g_motion_error;
    xd->plane[0].pre[0].buf = golden_ref->y_buffer + mb_y_offset;
    xd->plane[0].pre[0].stride = golden_ref->y_stride;
    g_motion_error =
        do_16x16_motion_search(cpi, prev_golden_ref_mv,
                               &stats->ref[GOLDEN_FRAME].m.mv, mb_row, mb_col);
    stats->ref[GOLDEN_FRAME].err = g_motion_error;
  } else {
    stats->ref[GOLDEN_FRAME].err = INT_MAX;
    stats->ref[GOLDEN_FRAME].m.mv.as_int = 0;
  }

  // Do an Alt-ref frame MV search, if it exists and is different than
  // last/golden frame.
  if (alt_ref) {
    int a_motion_error;
    xd->plane[0].pre[0].buf = alt_ref->y_buffer + mb_y_offset;
    xd->plane[0].pre[0].stride = alt_ref->y_stride;
    a_motion_error =
        do_16x16_zerozero_search(cpi, &stats->ref[ALTREF_FRAME].m.mv);

    stats->ref[ALTREF_FRAME].err = a_motion_error;
  } else {
    stats->ref[ALTREF_FRAME].err = INT_MAX;
    stats->ref[ALTREF_FRAME].m.mv.as_int = 0;
  }
}

static void update_mbgraph_frame_stats(VP9_COMP *cpi,
                                       MBGRAPH_FRAME_STATS *stats,
                                       YV12_BUFFER_CONFIG *buf,
                                       YV12_BUFFER_CONFIG *golden_ref,
                                       YV12_BUFFER_CONFIG *alt_ref) {
  MACROBLOCK *const x = &cpi->td.mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  VP9_COMMON *const cm = &cpi->common;

  int mb_col, mb_row, offset = 0;
  int mb_y_offset = 0;
  MV gld_top_mv = { 0, 0 };
  MODE_INFO mi_local;
  MODE_INFO mi_above, mi_left;

  vp9_zero(mi_local);
  // Set up limit values for motion vectors to prevent them extending outside
  // the UMV borders.
  x->mv_limits.row_min = -BORDER_MV_PIXELS_B16;
  x->mv_limits.row_max = (cm->mb_rows - 1) * 8 + BORDER_MV_PIXELS_B16;
  // Signal to vp9_predict_intra_block() that above is not available
  xd->above_mi = NULL;

  xd->plane[0].dst.stride = buf->y_stride;
  xd->plane[0].pre[0].stride = buf->y_stride;
  xd->plane[1].dst.stride = buf->uv_stride;
  xd->mi[0] = &mi_local;
  mi_local.sb_type = BLOCK_16X16;
  mi_local.ref_frame[0] = LAST_FRAME;
  mi_local.ref_frame[1] = NO_REF_FRAME;

  for (mb_row = 0; mb_row < cm->mb_rows; mb_row++) {
    MV gld_left_mv = gld_top_mv;
    int mb_y_in_offset = mb_y_offset;

    // Set up limit values for motion vectors to prevent them extending outside
    // the UMV borders.
    x->mv_limits.col_min = -BORDER_MV_PIXELS_B16;
    x->mv_limits.col_max = (cm->mb_cols - 1) * 8 + BORDER_MV_PIXELS_B16;
    // Signal to vp9_predict_intra_block() that left is not available
    xd->left_mi = NULL;

    for (mb_col = 0; mb_col < cm->mb_cols; mb_col++) {
      MBGRAPH_MB_STATS *mb_stats = &stats->mb_stats[offset + mb_col];

      update_mbgraph_mb_stats(cpi, mb_stats, buf, mb_y_in_offset, golden_ref,
                              &gld_left_mv, alt_ref, mb_row, mb_col);
      gld_left_mv = mb_stats->ref[GOLDEN_FRAME].m.mv.as_mv;
      if (mb_col == 0) {
        gld_top_mv = gld_left_mv;
      }
      // Signal to vp9_predict_intra_block() that left is available
      xd->left_mi = &mi_left;

      mb_y_in_offset += 16;
      x->mv_limits.col_min -= 16;
      x->mv_limits.col_max -= 16;
    }

    // Signal to vp9_predict_intra_block() that above is available
    xd->above_mi = &mi_above;

    mb_y_offset += buf->y_stride * 16;
    x->mv_limits.row_min -= 16;
    x->mv_limits.row_max -= 16;
    offset += cm->mb_cols;
  }
}

// void separate_arf_mbs_byzz
static void separate_arf_mbs(VP9_COMP *cpi) {
  VP9_COMMON *const cm = &cpi->common;
  int mb_col, mb_row, offset, i;
  int mi_row, mi_col;
  int ncnt[4] = { 0 };
  int n_frames = cpi->mbgraph_n_frames;

  int *arf_not_zz;

  CHECK_MEM_ERROR(
      &cm->error, arf_not_zz,
      vpx_calloc(cm->mb_rows * cm->mb_cols * sizeof(*arf_not_zz), 1));

  // We are not interested in results beyond the alt ref itself.
  if (n_frames > cpi->rc.frames_till_gf_update_due)
    n_frames = cpi->rc.frames_till_gf_update_due;

  // defer cost to reference frames
  for (i = n_frames - 1; i >= 0; i--) {
    MBGRAPH_FRAME_STATS *frame_stats = &cpi->mbgraph_stats[i];

    for (offset = 0, mb_row = 0; mb_row < cm->mb_rows;
         offset += cm->mb_cols, mb_row++) {
      for (mb_col = 0; mb_col < cm->mb_cols; mb_col++) {
        MBGRAPH_MB_STATS *mb_stats = &frame_stats->mb_stats[offset + mb_col];

        int altref_err = mb_stats->ref[ALTREF_FRAME].err;
        int intra_err = mb_stats->ref[INTRA_FRAME].err;
        int golden_err = mb_stats->ref[GOLDEN_FRAME].err;

        // Test for altref vs intra and gf and that its mv was 0,0.
        if (altref_err > 1000 || altref_err > intra_err ||
            altref_err > golden_err) {
          arf_not_zz[offset + mb_col]++;
        }
      }
    }
  }

  // arf_not_zz is indexed by MB, but this loop is indexed by MI to avoid out
  // of bound access in segmentation_map
  for (mi_row = 0; mi_row < cm->mi_rows; mi_row++) {
    for (mi_col = 0; mi_col < cm->mi_cols; mi_col++) {
      // If any of the blocks in the sequence failed then the MB
      // goes in segment 0
      if (arf_not_zz[mi_row / 2 * cm->mb_cols + mi_col / 2]) {
        ncnt[0]++;
        cpi->segmentation_map[mi_row * cm->mi_cols + mi_col] = 0;
      } else {
        cpi->segmentation_map[mi_row * cm->mi_cols + mi_col] = 1;
        ncnt[1]++;
      }
    }
  }

  // Note % of blocks that are marked as static
  if (cm->MBs)
    cpi->static_mb_pct = (ncnt[1] * 100) / (cm->mi_rows * cm->mi_cols);

  // This error case should not be reachable as this function should
  // never be called with the common data structure uninitialized.
  else
    cpi->static_mb_pct = 0;

  vp9_enable_segmentation(&cm->seg);

  // Free localy allocated storage
  vpx_free(arf_not_zz);
}

void vp9_update_mbgraph_stats(VP9_COMP *cpi) {
  VP9_COMMON *const cm = &cpi->common;
  int i, n_frames = vp9_lookahead_depth(cpi->lookahead);
  YV12_BUFFER_CONFIG *golden_ref = get_ref_frame_buffer(cpi, GOLDEN_FRAME);

  assert(golden_ref != NULL);

  // we need to look ahead beyond where the ARF transitions into
  // being a GF - so exit if we don't look ahead beyond that
  if (n_frames <= cpi->rc.frames_till_gf_update_due) return;

  if (n_frames > MAX_LAG_BUFFERS) n_frames = MAX_LAG_BUFFERS;

  cpi->mbgraph_n_frames = n_frames;
  for (i = 0; i < n_frames; i++) {
    MBGRAPH_FRAME_STATS *frame_stats = &cpi->mbgraph_stats[i];
    memset(frame_stats->mb_stats, 0,
           cm->mb_rows * cm->mb_cols * sizeof(*cpi->mbgraph_stats[i].mb_stats));
  }

  // do motion search to find contribution of each reference to data
  // later on in this GF group
  // FIXME really, the GF/last MC search should be done forward, and
  // the ARF MC search backwards, to get optimal results for MV caching
  for (i = 0; i < n_frames; i++) {
    MBGRAPH_FRAME_STATS *frame_stats = &cpi->mbgraph_stats[i];
    struct lookahead_entry *q_cur = vp9_lookahead_peek(cpi->lookahead, i);

    assert(q_cur != NULL);

    update_mbgraph_frame_stats(cpi, frame_stats, &q_cur->img, golden_ref,
                               cpi->Source);
  }

  vpx_clear_system_state();

  separate_arf_mbs(cpi);
}
