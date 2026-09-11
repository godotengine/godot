/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP8_COMMON_FINDNEARMV_H_
#define VPX_VP8_COMMON_FINDNEARMV_H_

#include "./vpx_config.h"
#include "mv.h"
#include "blockd.h"
#include "modecont.h"
#include "treecoder.h"

#ifdef __cplusplus
extern "C" {
#endif

static INLINE void mv_bias(int refmb_ref_frame_sign_bias, int refframe,
                           int_mv *mvp, const int *ref_frame_sign_bias) {
  if (refmb_ref_frame_sign_bias != ref_frame_sign_bias[refframe]) {
    mvp->as_mv.row *= -1;
    mvp->as_mv.col *= -1;
  }
}

#define LEFT_TOP_MARGIN (16 << 3)
#define RIGHT_BOTTOM_MARGIN (16 << 3)
static INLINE void vp8_clamp_mv2(int_mv *mv, const MACROBLOCKD *xd) {
  if (mv->as_mv.col < (xd->mb_to_left_edge - LEFT_TOP_MARGIN)) {
    mv->as_mv.col = xd->mb_to_left_edge - LEFT_TOP_MARGIN;
  } else if (mv->as_mv.col > xd->mb_to_right_edge + RIGHT_BOTTOM_MARGIN) {
    mv->as_mv.col = xd->mb_to_right_edge + RIGHT_BOTTOM_MARGIN;
  }

  if (mv->as_mv.row < (xd->mb_to_top_edge - LEFT_TOP_MARGIN)) {
    mv->as_mv.row = xd->mb_to_top_edge - LEFT_TOP_MARGIN;
  } else if (mv->as_mv.row > xd->mb_to_bottom_edge + RIGHT_BOTTOM_MARGIN) {
    mv->as_mv.row = xd->mb_to_bottom_edge + RIGHT_BOTTOM_MARGIN;
  }
}

static INLINE void vp8_clamp_mv(int_mv *mv, int mb_to_left_edge,
                                int mb_to_right_edge, int mb_to_top_edge,
                                int mb_to_bottom_edge) {
  mv->as_mv.col =
      (mv->as_mv.col < mb_to_left_edge) ? mb_to_left_edge : mv->as_mv.col;
  mv->as_mv.col =
      (mv->as_mv.col > mb_to_right_edge) ? mb_to_right_edge : mv->as_mv.col;
  mv->as_mv.row =
      (mv->as_mv.row < mb_to_top_edge) ? mb_to_top_edge : mv->as_mv.row;
  mv->as_mv.row =
      (mv->as_mv.row > mb_to_bottom_edge) ? mb_to_bottom_edge : mv->as_mv.row;
}
static INLINE unsigned int vp8_check_mv_bounds(int_mv *mv, int mb_to_left_edge,
                                               int mb_to_right_edge,
                                               int mb_to_top_edge,
                                               int mb_to_bottom_edge) {
  unsigned int need_to_clamp;
  need_to_clamp = (mv->as_mv.col < mb_to_left_edge);
  need_to_clamp |= (mv->as_mv.col > mb_to_right_edge);
  need_to_clamp |= (mv->as_mv.row < mb_to_top_edge);
  need_to_clamp |= (mv->as_mv.row > mb_to_bottom_edge);
  return need_to_clamp;
}

void vp8_find_near_mvs(MACROBLOCKD *xd, const MODE_INFO *here, int_mv *nearest,
                       int_mv *nearby, int_mv *best_mv, int near_mv_ref_cnts[4],
                       int refframe, int *ref_frame_sign_bias);

int vp8_find_near_mvs_bias(MACROBLOCKD *xd, const MODE_INFO *here,
                           int_mv mode_mv_sb[2][MB_MODE_COUNT],
                           int_mv best_mv_sb[2], int cnt[4], int refframe,
                           int *ref_frame_sign_bias);

vp8_prob *vp8_mv_ref_probs(vp8_prob p[VP8_MVREFS - 1],
                           const int near_mv_ref_ct[4]);

extern const unsigned char vp8_mbsplit_offset[4][16];

static INLINE uint32_t left_block_mv(const MODE_INFO *cur_mb, int b) {
  if (!(b & 3)) {
    /* On L edge, get from MB to left of us */
    --cur_mb;

    if (cur_mb->mbmi.mode != SPLITMV) return cur_mb->mbmi.mv.as_int;
    b += 4;
  }

  return (cur_mb->bmi + b - 1)->mv.as_int;
}

static INLINE uint32_t above_block_mv(const MODE_INFO *cur_mb, int b,
                                      int mi_stride) {
  if (!(b >> 2)) {
    /* On top edge, get from MB above us */
    cur_mb -= mi_stride;

    if (cur_mb->mbmi.mode != SPLITMV) return cur_mb->mbmi.mv.as_int;
    b += 16;
  }

  return (cur_mb->bmi + (b - 4))->mv.as_int;
}
static INLINE B_PREDICTION_MODE left_block_mode(const MODE_INFO *cur_mb,
                                                int b) {
  if (!(b & 3)) {
    /* On L edge, get from MB to left of us */
    --cur_mb;
    switch (cur_mb->mbmi.mode) {
      case B_PRED: return (cur_mb->bmi + b + 3)->as_mode;
      case DC_PRED: return B_DC_PRED;
      case V_PRED: return B_VE_PRED;
      case H_PRED: return B_HE_PRED;
      case TM_PRED: return B_TM_PRED;
      default: return B_DC_PRED;
    }
  }

  return (cur_mb->bmi + b - 1)->as_mode;
}

static INLINE B_PREDICTION_MODE above_block_mode(const MODE_INFO *cur_mb, int b,
                                                 int mi_stride) {
  if (!(b >> 2)) {
    /* On top edge, get from MB above us */
    cur_mb -= mi_stride;

    switch (cur_mb->mbmi.mode) {
      case B_PRED: return (cur_mb->bmi + b + 12)->as_mode;
      case DC_PRED: return B_DC_PRED;
      case V_PRED: return B_VE_PRED;
      case H_PRED: return B_HE_PRED;
      case TM_PRED: return B_TM_PRED;
      default: return B_DC_PRED;
    }
  }

  return (cur_mb->bmi + b - 4)->as_mode;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_COMMON_FINDNEARMV_H_
