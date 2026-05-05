/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <limits.h>
#include <math.h>
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_ports/system_state.h"

#include "vp9/encoder/vp9_aq_complexity.h"
#include "vp9/encoder/vp9_aq_variance.h"
#include "vp9/encoder/vp9_encodeframe.h"
#include "vp9/common/vp9_seg_common.h"
#include "vp9/encoder/vp9_segmentation.h"

#define AQ_C_SEGMENTS 5
#define DEFAULT_AQ2_SEG 3  // Neutral Q segment
#define AQ_C_STRENGTHS 3
static const double aq_c_q_adj_factor[AQ_C_STRENGTHS][AQ_C_SEGMENTS] = {
  { 1.75, 1.25, 1.05, 1.00, 0.90 },
  { 2.00, 1.50, 1.15, 1.00, 0.85 },
  { 2.50, 1.75, 1.25, 1.00, 0.80 }
};
static const double aq_c_transitions[AQ_C_STRENGTHS][AQ_C_SEGMENTS] = {
  { 0.15, 0.30, 0.55, 2.00, 100.0 },
  { 0.20, 0.40, 0.65, 2.00, 100.0 },
  { 0.25, 0.50, 0.75, 2.00, 100.0 }
};
static const double aq_c_var_thresholds[AQ_C_STRENGTHS][AQ_C_SEGMENTS] = {
  { -4.0, -3.0, -2.0, 100.00, 100.0 },
  { -3.5, -2.5, -1.5, 100.00, 100.0 },
  { -3.0, -2.0, -1.0, 100.00, 100.0 }
};

static int get_aq_c_strength(int q_index, vpx_bit_depth_t bit_depth) {
  // Approximate base quatizer (truncated to int)
  const int base_quant = vp9_ac_quant(q_index, 0, bit_depth) / 4;
  return (base_quant > 10) + (base_quant > 25);
}

void vp9_setup_in_frame_q_adj(VP9_COMP *cpi) {
  VP9_COMMON *const cm = &cpi->common;
  struct segmentation *const seg = &cm->seg;

  // Make SURE use of floating point in this function is safe.
  vpx_clear_system_state();

  if (frame_is_intra_only(cm) || cm->error_resilient_mode ||
      cpi->refresh_alt_ref_frame || cpi->force_update_segmentation ||
      (cpi->refresh_golden_frame && !cpi->rc.is_src_frame_alt_ref)) {
    int segment;
    const int aq_strength = get_aq_c_strength(cm->base_qindex, cm->bit_depth);

    // Clear down the segment map.
    memset(cpi->segmentation_map, DEFAULT_AQ2_SEG, cm->mi_rows * cm->mi_cols);

    vp9_clearall_segfeatures(seg);

    // Segmentation only makes sense if the target bits per SB is above a
    // threshold. Below this the overheads will usually outweigh any benefit.
    if (cpi->rc.sb64_target_rate < 256) {
      vp9_disable_segmentation(seg);
      return;
    }

    vp9_enable_segmentation(seg);

    // Select delta coding method.
    seg->abs_delta = SEGMENT_DELTADATA;

    // Default segment "Q" feature is disabled so it defaults to the baseline Q.
    vp9_disable_segfeature(seg, DEFAULT_AQ2_SEG, SEG_LVL_ALT_Q);

    // Use some of the segments for in frame Q adjustment.
    for (segment = 0; segment < AQ_C_SEGMENTS; ++segment) {
      int qindex_delta;

      if (segment == DEFAULT_AQ2_SEG) continue;

      qindex_delta = vp9_compute_qdelta_by_rate(
          &cpi->rc, cm->frame_type, cm->base_qindex,
          aq_c_q_adj_factor[aq_strength][segment], cm->bit_depth);

      // For AQ complexity mode, we don't allow Q0 in a segment if the base
      // Q is not 0. Q0 (lossless) implies 4x4 only and in AQ mode 2 a segment
      // Q delta is sometimes applied without going back around the rd loop.
      // This could lead to an illegal combination of partition size and q.
      if ((cm->base_qindex != 0) && ((cm->base_qindex + qindex_delta) == 0)) {
        qindex_delta = -cm->base_qindex + 1;
      }
      if ((cm->base_qindex + qindex_delta) > 0) {
        vp9_enable_segfeature(seg, segment, SEG_LVL_ALT_Q);
        vp9_set_segdata(seg, segment, SEG_LVL_ALT_Q, qindex_delta);
      }
    }
  }
}

#define DEFAULT_LV_THRESH 10.0
#define MIN_DEFAULT_LV_THRESH 8.0
// Select a segment for the current block.
// The choice of segment for a block depends on the ratio of the projected
// bits for the block vs a target average and its spatial complexity.
void vp9_caq_select_segment(VP9_COMP *cpi, MACROBLOCK *mb, BLOCK_SIZE bs,
                            int mi_row, int mi_col, int projected_rate) {
  VP9_COMMON *const cm = &cpi->common;

  const int mi_offset = mi_row * cm->mi_cols + mi_col;
  const int bw = num_8x8_blocks_wide_lookup[BLOCK_64X64];
  const int bh = num_8x8_blocks_high_lookup[BLOCK_64X64];
  const int xmis = VPXMIN(cm->mi_cols - mi_col, num_8x8_blocks_wide_lookup[bs]);
  const int ymis = VPXMIN(cm->mi_rows - mi_row, num_8x8_blocks_high_lookup[bs]);
  int x, y;
  int i;
  unsigned char segment;

  if (0) {
    segment = DEFAULT_AQ2_SEG;
  } else {
    // Rate depends on fraction of a SB64 in frame (xmis * ymis / bw * bh).
    // It is converted to bits * 256 units.
    const int target_rate =
        (cpi->rc.sb64_target_rate * xmis * ymis * 256) / (bw * bh);
    double logvar;
    double low_var_thresh;
    const int aq_strength = get_aq_c_strength(cm->base_qindex, cm->bit_depth);

    vpx_clear_system_state();
    low_var_thresh = (cpi->oxcf.pass == 2) ? VPXMAX(cpi->twopass.mb_av_energy,
                                                    MIN_DEFAULT_LV_THRESH)
                                           : DEFAULT_LV_THRESH;

    vp9_setup_src_planes(mb, cpi->Source, mi_row, mi_col);
    logvar = vp9_log_block_var(cpi, mb, bs);

    segment = AQ_C_SEGMENTS - 1;  // Just in case no break out below.
    for (i = 0; i < AQ_C_SEGMENTS; ++i) {
      // Test rate against a threshold value and variance against a threshold.
      // Increasing segment number (higher variance and complexity) = higher Q.
      if ((projected_rate < target_rate * aq_c_transitions[aq_strength][i]) &&
          (logvar < (low_var_thresh + aq_c_var_thresholds[aq_strength][i]))) {
        segment = i;
        break;
      }
    }
  }

  // Fill in the entires in the segment map corresponding to this SB64.
  for (y = 0; y < ymis; y++) {
    for (x = 0; x < xmis; x++) {
      cpi->segmentation_map[mi_offset + y * cm->mi_cols + x] = segment;
    }
  }
}
