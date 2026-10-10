/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP9_COMMON_VP9_PRED_COMMON_H_
#define VPX_VP9_COMMON_VP9_PRED_COMMON_H_

#include "vp9/common/vp9_blockd.h"
#include "vp9/common/vp9_onyxc_int.h"
#include "vpx_dsp/vpx_dsp_common.h"

#ifdef __cplusplus
extern "C" {
#endif

static INLINE int get_segment_id(const VP9_COMMON *cm,
                                 const uint8_t *segment_ids, BLOCK_SIZE bsize,
                                 int mi_row, int mi_col) {
  const int mi_offset = mi_row * cm->mi_cols + mi_col;
  const int bw = num_8x8_blocks_wide_lookup[bsize];
  const int bh = num_8x8_blocks_high_lookup[bsize];
  const int xmis = VPXMIN(cm->mi_cols - mi_col, bw);
  const int ymis = VPXMIN(cm->mi_rows - mi_row, bh);
  int x, y, segment_id = MAX_SEGMENTS;

  for (y = 0; y < ymis; ++y)
    for (x = 0; x < xmis; ++x)
      segment_id =
          VPXMIN(segment_id, segment_ids[mi_offset + y * cm->mi_cols + x]);

  assert(segment_id >= 0 && segment_id < MAX_SEGMENTS);
  return segment_id;
}

static INLINE int vp9_get_pred_context_seg_id(const MACROBLOCKD *xd) {
  const MODE_INFO *const above_mi = xd->above_mi;
  const MODE_INFO *const left_mi = xd->left_mi;
  const int above_sip = (above_mi != NULL) ? above_mi->seg_id_predicted : 0;
  const int left_sip = (left_mi != NULL) ? left_mi->seg_id_predicted : 0;

  return above_sip + left_sip;
}

static INLINE vpx_prob vp9_get_pred_prob_seg_id(const struct segmentation *seg,
                                                const MACROBLOCKD *xd) {
  return seg->pred_probs[vp9_get_pred_context_seg_id(xd)];
}

static INLINE int vp9_get_skip_context(const MACROBLOCKD *xd) {
  const MODE_INFO *const above_mi = xd->above_mi;
  const MODE_INFO *const left_mi = xd->left_mi;
  const int above_skip = (above_mi != NULL) ? above_mi->skip : 0;
  const int left_skip = (left_mi != NULL) ? left_mi->skip : 0;
  return above_skip + left_skip;
}

static INLINE vpx_prob vp9_get_skip_prob(const VP9_COMMON *cm,
                                         const MACROBLOCKD *xd) {
  return cm->fc->skip_probs[vp9_get_skip_context(xd)];
}

// Returns a context number for the given MB prediction signal
static INLINE int get_pred_context_switchable_interp(const MACROBLOCKD *xd) {
  // Note:
  // The mode info data structure has a one element border above and to the
  // left of the entries corresponding to real macroblocks.
  // The prediction flags in these dummy entries are initialized to 0.
  const MODE_INFO *const left_mi = xd->left_mi;
  const int left_type = left_mi ? left_mi->interp_filter : SWITCHABLE_FILTERS;
  const MODE_INFO *const above_mi = xd->above_mi;
  const int above_type =
      above_mi ? above_mi->interp_filter : SWITCHABLE_FILTERS;

  if (left_type == above_type)
    return left_type;
  else if (left_type == SWITCHABLE_FILTERS)
    return above_type;
  else if (above_type == SWITCHABLE_FILTERS)
    return left_type;
  else
    return SWITCHABLE_FILTERS;
}

// The mode info data structure has a one element border above and to the
// left of the entries corresponding to real macroblocks.
// The prediction flags in these dummy entries are initialized to 0.
// 0 - inter/inter, inter/--, --/inter, --/--
// 1 - intra/inter, inter/intra
// 2 - intra/--, --/intra
// 3 - intra/intra
static INLINE int get_intra_inter_context(const MACROBLOCKD *xd) {
  const MODE_INFO *const above_mi = xd->above_mi;
  const MODE_INFO *const left_mi = xd->left_mi;
  const int has_above = !!above_mi;
  const int has_left = !!left_mi;

  if (has_above && has_left) {  // both edges available
    const int above_intra = !is_inter_block(above_mi);
    const int left_intra = !is_inter_block(left_mi);
    return left_intra && above_intra ? 3 : left_intra || above_intra;
  } else if (has_above || has_left) {  // one edge available
    return 2 * !is_inter_block(has_above ? above_mi : left_mi);
  }
  return 0;
}

static INLINE vpx_prob vp9_get_intra_inter_prob(const VP9_COMMON *cm,
                                                const MACROBLOCKD *xd) {
  return cm->fc->intra_inter_prob[get_intra_inter_context(xd)];
}

int vp9_get_reference_mode_context(const VP9_COMMON *cm, const MACROBLOCKD *xd);

static INLINE vpx_prob vp9_get_reference_mode_prob(const VP9_COMMON *cm,
                                                   const MACROBLOCKD *xd) {
  return cm->fc->comp_inter_prob[vp9_get_reference_mode_context(cm, xd)];
}

int vp9_get_pred_context_comp_ref_p(const VP9_COMMON *cm,
                                    const MACROBLOCKD *xd);

static INLINE vpx_prob vp9_get_pred_prob_comp_ref_p(const VP9_COMMON *cm,
                                                    const MACROBLOCKD *xd) {
  const int pred_context = vp9_get_pred_context_comp_ref_p(cm, xd);
  return cm->fc->comp_ref_prob[pred_context];
}

int vp9_get_pred_context_single_ref_p1(const MACROBLOCKD *xd);

static INLINE vpx_prob vp9_get_pred_prob_single_ref_p1(const VP9_COMMON *cm,
                                                       const MACROBLOCKD *xd) {
  return cm->fc->single_ref_prob[vp9_get_pred_context_single_ref_p1(xd)][0];
}

int vp9_get_pred_context_single_ref_p2(const MACROBLOCKD *xd);

static INLINE vpx_prob vp9_get_pred_prob_single_ref_p2(const VP9_COMMON *cm,
                                                       const MACROBLOCKD *xd) {
  return cm->fc->single_ref_prob[vp9_get_pred_context_single_ref_p2(xd)][1];
}

int vp9_compound_reference_allowed(const VP9_COMMON *cm);

void vp9_setup_compound_reference_mode(VP9_COMMON *cm);

// Returns a context number for the given MB prediction signal
// The mode info data structure has a one element border above and to the
// left of the entries corresponding to real blocks.
// The prediction flags in these dummy entries are initialized to 0.
static INLINE int get_tx_size_context(const MACROBLOCKD *xd) {
  const int max_tx_size = max_txsize_lookup[xd->mi[0]->sb_type];
  const MODE_INFO *const above_mi = xd->above_mi;
  const MODE_INFO *const left_mi = xd->left_mi;
  const int has_above = !!above_mi;
  const int has_left = !!left_mi;
  int above_ctx =
      (has_above && !above_mi->skip) ? (int)above_mi->tx_size : max_tx_size;
  int left_ctx =
      (has_left && !left_mi->skip) ? (int)left_mi->tx_size : max_tx_size;
  if (!has_left) left_ctx = above_ctx;

  if (!has_above) above_ctx = left_ctx;

  return (above_ctx + left_ctx) > max_tx_size;
}

static INLINE const vpx_prob *get_tx_probs(TX_SIZE max_tx_size, int ctx,
                                           const struct tx_probs *tx_probs) {
  switch (max_tx_size) {
    case TX_8X8: return tx_probs->p8x8[ctx];
    case TX_16X16: return tx_probs->p16x16[ctx];
    case TX_32X32: return tx_probs->p32x32[ctx];
    default: assert(0 && "Invalid max_tx_size."); return NULL;
  }
}

static INLINE unsigned int *get_tx_counts(TX_SIZE max_tx_size, int ctx,
                                          struct tx_counts *tx_counts) {
  switch (max_tx_size) {
    case TX_8X8: return tx_counts->p8x8[ctx];
    case TX_16X16: return tx_counts->p16x16[ctx];
    case TX_32X32: return tx_counts->p32x32[ctx];
    default: assert(0 && "Invalid max_tx_size."); return NULL;
  }
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_COMMON_VP9_PRED_COMMON_H_
