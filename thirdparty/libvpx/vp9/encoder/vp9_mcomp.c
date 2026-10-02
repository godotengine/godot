/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
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

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"

#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_mem/vpx_mem.h"
#include "vpx_ports/mem.h"

#include "vp9/common/vp9_common.h"
#include "vp9/common/vp9_mvref_common.h"
#include "vp9/common/vp9_reconinter.h"

#include "vp9/encoder/vp9_encoder.h"
#include "vp9/encoder/vp9_mcomp.h"

// #define NEW_DIAMOND_SEARCH

void vp9_set_mv_search_range(MvLimits *mv_limits, const MV *mv) {
  int col_min = (mv->col >> 3) - MAX_FULL_PEL_VAL + (mv->col & 7 ? 1 : 0);
  int row_min = (mv->row >> 3) - MAX_FULL_PEL_VAL + (mv->row & 7 ? 1 : 0);
  int col_max = (mv->col >> 3) + MAX_FULL_PEL_VAL;
  int row_max = (mv->row >> 3) + MAX_FULL_PEL_VAL;

  col_min = VPXMAX(col_min, (MV_LOW >> 3) + 1);
  row_min = VPXMAX(row_min, (MV_LOW >> 3) + 1);
  col_max = VPXMIN(col_max, (MV_UPP >> 3) - 1);
  row_max = VPXMIN(row_max, (MV_UPP >> 3) - 1);

  // Get intersection of UMV window and valid MV window to reduce # of checks
  // in diamond search.
  if (mv_limits->col_min < col_min) mv_limits->col_min = col_min;
  if (mv_limits->col_max > col_max) mv_limits->col_max = col_max;
  if (mv_limits->row_min < row_min) mv_limits->row_min = row_min;
  if (mv_limits->row_max > row_max) mv_limits->row_max = row_max;
}

void vp9_set_subpel_mv_search_range(MvLimits *subpel_mv_limits,
                                    const MvLimits *umv_window_limits,
                                    const MV *ref_mv) {
  subpel_mv_limits->col_min = VPXMAX(umv_window_limits->col_min * 8,
                                     ref_mv->col - MAX_FULL_PEL_VAL * 8);
  subpel_mv_limits->col_max = VPXMIN(umv_window_limits->col_max * 8,
                                     ref_mv->col + MAX_FULL_PEL_VAL * 8);
  subpel_mv_limits->row_min = VPXMAX(umv_window_limits->row_min * 8,
                                     ref_mv->row - MAX_FULL_PEL_VAL * 8);
  subpel_mv_limits->row_max = VPXMIN(umv_window_limits->row_max * 8,
                                     ref_mv->row + MAX_FULL_PEL_VAL * 8);

  subpel_mv_limits->col_min = VPXMAX(MV_LOW + 1, subpel_mv_limits->col_min);
  subpel_mv_limits->col_max = VPXMIN(MV_UPP - 1, subpel_mv_limits->col_max);
  subpel_mv_limits->row_min = VPXMAX(MV_LOW + 1, subpel_mv_limits->row_min);
  subpel_mv_limits->row_max = VPXMIN(MV_UPP - 1, subpel_mv_limits->row_max);
}

int vp9_init_search_range(int size) {
  int sr = 0;
  // Minimum search size no matter what the passed in value.
  size = VPXMAX(16, size);

  while ((size << sr) < MAX_FULL_PEL_VAL) sr++;

  sr = VPXMIN(sr, MAX_MVSEARCH_STEPS - 2);
  return sr;
}

int vp9_mv_bit_cost(const MV *mv, const MV *ref, const int *mvjcost,
                    int *mvcost[2], int weight) {
  const MV diff = { mv->row - ref->row, mv->col - ref->col };
  return ROUND_POWER_OF_TWO(mv_cost(&diff, mvjcost, mvcost) * weight, 7);
}

#define PIXEL_TRANSFORM_ERROR_SCALE 4
static int mv_err_cost(const MV *mv, const MV *ref, const int *mvjcost,
                       int *mvcost[2], int error_per_bit) {
  if (mvcost) {
    const MV diff = { mv->row - ref->row, mv->col - ref->col };
    return (int)ROUND64_POWER_OF_TWO(
        (int64_t)mv_cost(&diff, mvjcost, mvcost) * error_per_bit,
        RDDIV_BITS + VP9_PROB_COST_SHIFT - RD_EPB_SHIFT +
            PIXEL_TRANSFORM_ERROR_SCALE);
  }
  return 0;
}
void vp9_init_dsmotion_compensation(search_site_config *cfg, int stride) {
  int len;
  int ss_count = 0;

  for (len = MAX_FIRST_STEP; len > 0; len /= 2) {
    // Generate offsets for 4 search sites per step.
    const MV ss_mvs[] = { { -len, 0 }, { len, 0 }, { 0, -len }, { 0, len } };
    int i;
    for (i = 0; i < 4; ++i, ++ss_count) {
      cfg->ss_mv[ss_count] = ss_mvs[i];
      cfg->ss_os[ss_count] = ss_mvs[i].row * stride + ss_mvs[i].col;
    }
  }

  cfg->searches_per_step = 4;
  cfg->total_steps = ss_count / cfg->searches_per_step;
}

void vp9_init3smotion_compensation(search_site_config *cfg, int stride) {
  int len;
  int ss_count = 0;

  for (len = MAX_FIRST_STEP; len > 0; len /= 2) {
    // Generate offsets for 8 search sites per step.
    const MV ss_mvs[8] = { { -len, 0 },   { len, 0 },     { 0, -len },
                           { 0, len },    { -len, -len }, { -len, len },
                           { len, -len }, { len, len } };
    int i;
    for (i = 0; i < 8; ++i, ++ss_count) {
      cfg->ss_mv[ss_count] = ss_mvs[i];
      cfg->ss_os[ss_count] = ss_mvs[i].row * stride + ss_mvs[i].col;
    }
  }

  cfg->searches_per_step = 8;
  cfg->total_steps = ss_count / cfg->searches_per_step;
}

// convert motion vector component to offset for sv[a]f calc
static INLINE int sp(int x) { return x & 7; }

static INLINE const uint8_t *pre(const uint8_t *buf, int stride, int r, int c) {
  return &buf[(r >> 3) * stride + (c >> 3)];
}

#if CONFIG_VP9_HIGHBITDEPTH
/* checks if (r, c) has better score than previous best */
#define CHECK_BETTER(v, r, c)                                                  \
  do {                                                                         \
    if (c >= minc && c <= maxc && r >= minr && r <= maxr) {                    \
      int64_t tmpmse;                                                          \
      const MV cb_mv = { r, c };                                               \
      const MV cb_ref_mv = { rr, rc };                                         \
      if (second_pred == NULL) {                                               \
        thismse = vfp->svf(pre(y, y_stride, r, c), y_stride, sp(c), sp(r), z,  \
                           src_stride, &sse);                                  \
      } else {                                                                 \
        thismse = vfp->svaf(pre(y, y_stride, r, c), y_stride, sp(c), sp(r), z, \
                            src_stride, &sse, second_pred);                    \
      }                                                                        \
      tmpmse = thismse;                                                        \
      tmpmse +=                                                                \
          mv_err_cost(&cb_mv, &cb_ref_mv, mvjcost, mvcost, error_per_bit);     \
      if (tmpmse >= INT_MAX) {                                                 \
        v = INT_MAX;                                                           \
      } else if ((v = (uint32_t)tmpmse) < besterr) {                           \
        besterr = v;                                                           \
        br = r;                                                                \
        bc = c;                                                                \
        *distortion = thismse;                                                 \
        *sse1 = sse;                                                           \
      }                                                                        \
    } else {                                                                   \
      v = INT_MAX;                                                             \
    }                                                                          \
  } while (0)
#else
/* checks if (r, c) has better score than previous best */
#define CHECK_BETTER(v, r, c)                                                  \
  do {                                                                         \
    if (c >= minc && c <= maxc && r >= minr && r <= maxr) {                    \
      const MV cb_mv = { r, c };                                               \
      const MV cb_ref_mv = { rr, rc };                                         \
      if (second_pred == NULL)                                                 \
        thismse = vfp->svf(pre(y, y_stride, r, c), y_stride, sp(c), sp(r), z,  \
                           src_stride, &sse);                                  \
      else                                                                     \
        thismse = vfp->svaf(pre(y, y_stride, r, c), y_stride, sp(c), sp(r), z, \
                            src_stride, &sse, second_pred);                    \
      if ((v = mv_err_cost(&cb_mv, &cb_ref_mv, mvjcost, mvcost,                \
                           error_per_bit) +                                    \
               thismse) < besterr) {                                           \
        besterr = v;                                                           \
        br = r;                                                                \
        bc = c;                                                                \
        *distortion = thismse;                                                 \
        *sse1 = sse;                                                           \
      }                                                                        \
    } else {                                                                   \
      v = INT_MAX;                                                             \
    }                                                                          \
  } while (0)

#endif
#define FIRST_LEVEL_CHECKS                                       \
  do {                                                           \
    unsigned int left, right, up, down, diag;                    \
    CHECK_BETTER(left, tr, tc - hstep);                          \
    CHECK_BETTER(right, tr, tc + hstep);                         \
    CHECK_BETTER(up, tr - hstep, tc);                            \
    CHECK_BETTER(down, tr + hstep, tc);                          \
    whichdir = (left < right ? 0 : 1) + (up < down ? 0 : 2);     \
    switch (whichdir) {                                          \
      case 0: CHECK_BETTER(diag, tr - hstep, tc - hstep); break; \
      case 1: CHECK_BETTER(diag, tr - hstep, tc + hstep); break; \
      case 2: CHECK_BETTER(diag, tr + hstep, tc - hstep); break; \
      case 3: CHECK_BETTER(diag, tr + hstep, tc + hstep); break; \
    }                                                            \
  } while (0)

#define SECOND_LEVEL_CHECKS                                       \
  do {                                                            \
    int kr, kc;                                                   \
    unsigned int second;                                          \
    if (tr != br && tc != bc) {                                   \
      kr = br - tr;                                               \
      kc = bc - tc;                                               \
      CHECK_BETTER(second, tr + kr, tc + 2 * kc);                 \
      CHECK_BETTER(second, tr + 2 * kr, tc + kc);                 \
    } else if (tr == br && tc != bc) {                            \
      kc = bc - tc;                                               \
      CHECK_BETTER(second, tr + hstep, tc + 2 * kc);              \
      CHECK_BETTER(second, tr - hstep, tc + 2 * kc);              \
      switch (whichdir) {                                         \
        case 0:                                                   \
        case 1: CHECK_BETTER(second, tr + hstep, tc + kc); break; \
        case 2:                                                   \
        case 3: CHECK_BETTER(second, tr - hstep, tc + kc); break; \
      }                                                           \
    } else if (tr != br && tc == bc) {                            \
      kr = br - tr;                                               \
      CHECK_BETTER(second, tr + 2 * kr, tc + hstep);              \
      CHECK_BETTER(second, tr + 2 * kr, tc - hstep);              \
      switch (whichdir) {                                         \
        case 0:                                                   \
        case 2: CHECK_BETTER(second, tr + kr, tc + hstep); break; \
        case 1:                                                   \
        case 3: CHECK_BETTER(second, tr + kr, tc - hstep); break; \
      }                                                           \
    }                                                             \
  } while (0)

#define SETUP_SUBPEL_SEARCH                                                 \
  const uint8_t *const z = x->plane[0].src.buf;                             \
  const int src_stride = x->plane[0].src.stride;                            \
  const MACROBLOCKD *xd = &x->e_mbd;                                        \
  unsigned int besterr = UINT_MAX;                                          \
  unsigned int sse;                                                         \
  unsigned int whichdir;                                                    \
  int thismse;                                                              \
  const unsigned int halfiters = iters_per_step;                            \
  const unsigned int quarteriters = iters_per_step;                         \
  const unsigned int eighthiters = iters_per_step;                          \
  const int y_stride = xd->plane[0].pre[0].stride;                          \
  const int offset = bestmv->row * y_stride + bestmv->col;                  \
  const uint8_t *const y = xd->plane[0].pre[0].buf;                         \
                                                                            \
  int rr = ref_mv->row;                                                     \
  int rc = ref_mv->col;                                                     \
  int br = bestmv->row * 8;                                                 \
  int bc = bestmv->col * 8;                                                 \
  int hstep = 4;                                                            \
  int minc, maxc, minr, maxr;                                               \
  int tr = br;                                                              \
  int tc = bc;                                                              \
  MvLimits subpel_mv_limits;                                                \
                                                                            \
  vp9_set_subpel_mv_search_range(&subpel_mv_limits, &x->mv_limits, ref_mv); \
  minc = subpel_mv_limits.col_min;                                          \
  maxc = subpel_mv_limits.col_max;                                          \
  minr = subpel_mv_limits.row_min;                                          \
  maxr = subpel_mv_limits.row_max;                                          \
                                                                            \
  bestmv->row *= 8;                                                         \
  bestmv->col *= 8

static unsigned int setup_center_error(
    const MACROBLOCKD *xd, const MV *bestmv, const MV *ref_mv,
    int error_per_bit, const vp9_variance_fn_ptr_t *vfp,
    const uint8_t *const src, const int src_stride, const uint8_t *const y,
    int y_stride, const uint8_t *second_pred, int w, int h, int offset,
    int *mvjcost, int *mvcost[2], uint32_t *sse1, uint32_t *distortion) {
#if CONFIG_VP9_HIGHBITDEPTH
  uint64_t besterr;
  if (second_pred != NULL) {
    if (xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH) {
      DECLARE_ALIGNED(16, uint16_t, comp_pred16[64 * 64]);
      vpx_highbd_comp_avg_pred(comp_pred16, CONVERT_TO_SHORTPTR(second_pred), w,
                               h, CONVERT_TO_SHORTPTR(y + offset), y_stride);
      besterr =
          vfp->vf(CONVERT_TO_BYTEPTR(comp_pred16), w, src, src_stride, sse1);
    } else {
      DECLARE_ALIGNED(32, uint8_t, comp_pred[64 * 64]);
      vpx_comp_avg_pred(comp_pred, second_pred, w, h, y + offset, y_stride);
      besterr = vfp->vf(comp_pred, w, src, src_stride, sse1);
    }
  } else {
    besterr = vfp->vf(y + offset, y_stride, src, src_stride, sse1);
  }
  *distortion = (uint32_t)besterr;
  besterr += mv_err_cost(bestmv, ref_mv, mvjcost, mvcost, error_per_bit);
  if (besterr >= UINT_MAX) return UINT_MAX;
  return (uint32_t)besterr;
#else
  uint32_t besterr;
  (void)xd;
  if (second_pred != NULL) {
    DECLARE_ALIGNED(32, uint8_t, comp_pred[64 * 64]);
    vpx_comp_avg_pred(comp_pred, second_pred, w, h, y + offset, y_stride);
    besterr = vfp->vf(comp_pred, w, src, src_stride, sse1);
  } else {
    besterr = vfp->vf(y + offset, y_stride, src, src_stride, sse1);
  }
  *distortion = besterr;
  besterr += mv_err_cost(bestmv, ref_mv, mvjcost, mvcost, error_per_bit);
  return besterr;
#endif  // CONFIG_VP9_HIGHBITDEPTH
}

static INLINE int64_t divide_and_round(const int64_t n, const int64_t d) {
  return ((n < 0) ^ (d < 0)) ? ((n - d / 2) / d) : ((n + d / 2) / d);
}

static INLINE int is_cost_list_wellbehaved(int *cost_list) {
  return cost_list[0] < cost_list[1] && cost_list[0] < cost_list[2] &&
         cost_list[0] < cost_list[3] && cost_list[0] < cost_list[4];
}

// Returns surface minima estimate at given precision in 1/2^n bits.
// Assume a model for the cost surface: S = A(x - x0)^2 + B(y - y0)^2 + C
// For a given set of costs S0, S1, S2, S3, S4 at points
// (y, x) = (0, 0), (0, -1), (1, 0), (0, 1) and (-1, 0) respectively,
// the solution for the location of the minima (x0, y0) is given by:
// x0 = 1/2 (S1 - S3)/(S1 + S3 - 2*S0),
// y0 = 1/2 (S4 - S2)/(S4 + S2 - 2*S0).
// The code below is an integerized version of that.
static void get_cost_surf_min(int *cost_list, int *ir, int *ic, int bits) {
  const int64_t x0 = (int64_t)cost_list[1] - cost_list[3];
  const int64_t y0 = cost_list[1] - 2 * (int64_t)cost_list[0] + cost_list[3];
  const int64_t x1 = (int64_t)cost_list[4] - cost_list[2];
  const int64_t y1 = cost_list[4] - 2 * (int64_t)cost_list[0] + cost_list[2];
  const int b = 1 << (bits - 1);
  *ic = (int)divide_and_round(x0 * b, y0);
  *ir = (int)divide_and_round(x1 * b, y1);
}

uint32_t vp9_skip_sub_pixel_tree(
    const MACROBLOCK *x, MV *bestmv, const MV *ref_mv, int allow_hp,
    int error_per_bit, const vp9_variance_fn_ptr_t *vfp, int forced_stop,
    int iters_per_step, int *cost_list, int *mvjcost, int *mvcost[2],
    uint32_t *distortion, uint32_t *sse1, const uint8_t *second_pred, int w,
    int h, int use_accurate_subpel_search) {
  SETUP_SUBPEL_SEARCH;
  besterr = setup_center_error(xd, bestmv, ref_mv, error_per_bit, vfp, z,
                               src_stride, y, y_stride, second_pred, w, h,
                               offset, mvjcost, mvcost, sse1, distortion);
  (void)halfiters;
  (void)quarteriters;
  (void)eighthiters;
  (void)whichdir;
  (void)allow_hp;
  (void)forced_stop;
  (void)hstep;
  (void)rr;
  (void)rc;
  (void)minr;
  (void)minc;
  (void)maxr;
  (void)maxc;
  (void)tr;
  (void)tc;
  (void)sse;
  (void)thismse;
  (void)cost_list;
  (void)use_accurate_subpel_search;

  return besterr;
}

uint32_t vp9_find_best_sub_pixel_tree_pruned_evenmore(
    const MACROBLOCK *x, MV *bestmv, const MV *ref_mv, int allow_hp,
    int error_per_bit, const vp9_variance_fn_ptr_t *vfp, int forced_stop,
    int iters_per_step, int *cost_list, int *mvjcost, int *mvcost[2],
    uint32_t *distortion, uint32_t *sse1, const uint8_t *second_pred, int w,
    int h, int use_accurate_subpel_search) {
  SETUP_SUBPEL_SEARCH;
  besterr = setup_center_error(xd, bestmv, ref_mv, error_per_bit, vfp, z,
                               src_stride, y, y_stride, second_pred, w, h,
                               offset, mvjcost, mvcost, sse1, distortion);
  (void)halfiters;
  (void)quarteriters;
  (void)eighthiters;
  (void)whichdir;
  (void)allow_hp;
  (void)forced_stop;
  (void)hstep;
  (void)use_accurate_subpel_search;

  if (cost_list && cost_list[0] != INT_MAX && cost_list[1] != INT_MAX &&
      cost_list[2] != INT_MAX && cost_list[3] != INT_MAX &&
      cost_list[4] != INT_MAX && is_cost_list_wellbehaved(cost_list)) {
    int ir, ic;
    unsigned int minpt = INT_MAX;
    get_cost_surf_min(cost_list, &ir, &ic, 2);
    if (ir != 0 || ic != 0) {
      CHECK_BETTER(minpt, tr + 2 * ir, tc + 2 * ic);
    }
  } else {
    FIRST_LEVEL_CHECKS;
    if (halfiters > 1) {
      SECOND_LEVEL_CHECKS;
    }

    tr = br;
    tc = bc;

    // Each subsequent iteration checks at least one point in common with
    // the last iteration could be 2 ( if diag selected) 1/4 pel
    // Note forced_stop: 0 - full, 1 - qtr only, 2 - half only
    if (forced_stop != 2) {
      hstep >>= 1;
      FIRST_LEVEL_CHECKS;
      if (quarteriters > 1) {
        SECOND_LEVEL_CHECKS;
      }
    }
  }

  tr = br;
  tc = bc;

  if (allow_hp && use_mv_hp(ref_mv) && forced_stop == 0) {
    hstep >>= 1;
    FIRST_LEVEL_CHECKS;
    if (eighthiters > 1) {
      SECOND_LEVEL_CHECKS;
    }
  }

  bestmv->row = br;
  bestmv->col = bc;

  return besterr;
}

uint32_t vp9_find_best_sub_pixel_tree_pruned_more(
    const MACROBLOCK *x, MV *bestmv, const MV *ref_mv, int allow_hp,
    int error_per_bit, const vp9_variance_fn_ptr_t *vfp, int forced_stop,
    int iters_per_step, int *cost_list, int *mvjcost, int *mvcost[2],
    uint32_t *distortion, uint32_t *sse1, const uint8_t *second_pred, int w,
    int h, int use_accurate_subpel_search) {
  SETUP_SUBPEL_SEARCH;
  (void)use_accurate_subpel_search;

  besterr = setup_center_error(xd, bestmv, ref_mv, error_per_bit, vfp, z,
                               src_stride, y, y_stride, second_pred, w, h,
                               offset, mvjcost, mvcost, sse1, distortion);
  if (cost_list && cost_list[0] != INT_MAX && cost_list[1] != INT_MAX &&
      cost_list[2] != INT_MAX && cost_list[3] != INT_MAX &&
      cost_list[4] != INT_MAX && is_cost_list_wellbehaved(cost_list)) {
    unsigned int minpt;
    int ir, ic;
    get_cost_surf_min(cost_list, &ir, &ic, 1);
    if (ir != 0 || ic != 0) {
      CHECK_BETTER(minpt, tr + ir * hstep, tc + ic * hstep);
    }
  } else {
    FIRST_LEVEL_CHECKS;
    if (halfiters > 1) {
      SECOND_LEVEL_CHECKS;
    }
  }

  // Each subsequent iteration checks at least one point in common with
  // the last iteration could be 2 ( if diag selected) 1/4 pel

  // Note forced_stop: 0 - full, 1 - qtr only, 2 - half only
  if (forced_stop != 2) {
    tr = br;
    tc = bc;
    hstep >>= 1;
    FIRST_LEVEL_CHECKS;
    if (quarteriters > 1) {
      SECOND_LEVEL_CHECKS;
    }
  }

  if (allow_hp && use_mv_hp(ref_mv) && forced_stop == 0) {
    tr = br;
    tc = bc;
    hstep >>= 1;
    FIRST_LEVEL_CHECKS;
    if (eighthiters > 1) {
      SECOND_LEVEL_CHECKS;
    }
  }
  // These lines insure static analysis doesn't warn that
  // tr and tc aren't used after the above point.
  (void)tr;
  (void)tc;

  bestmv->row = br;
  bestmv->col = bc;

  return besterr;
}

uint32_t vp9_find_best_sub_pixel_tree_pruned(
    const MACROBLOCK *x, MV *bestmv, const MV *ref_mv, int allow_hp,
    int error_per_bit, const vp9_variance_fn_ptr_t *vfp, int forced_stop,
    int iters_per_step, int *cost_list, int *mvjcost, int *mvcost[2],
    uint32_t *distortion, uint32_t *sse1, const uint8_t *second_pred, int w,
    int h, int use_accurate_subpel_search) {
  SETUP_SUBPEL_SEARCH;
  (void)use_accurate_subpel_search;

  besterr = setup_center_error(xd, bestmv, ref_mv, error_per_bit, vfp, z,
                               src_stride, y, y_stride, second_pred, w, h,
                               offset, mvjcost, mvcost, sse1, distortion);
  if (cost_list && cost_list[0] != INT_MAX && cost_list[1] != INT_MAX &&
      cost_list[2] != INT_MAX && cost_list[3] != INT_MAX &&
      cost_list[4] != INT_MAX) {
    unsigned int left, right, up, down, diag;
    whichdir = (cost_list[1] < cost_list[3] ? 0 : 1) +
               (cost_list[2] < cost_list[4] ? 0 : 2);
    switch (whichdir) {
      case 0:
        CHECK_BETTER(left, tr, tc - hstep);
        CHECK_BETTER(down, tr + hstep, tc);
        CHECK_BETTER(diag, tr + hstep, tc - hstep);
        break;
      case 1:
        CHECK_BETTER(right, tr, tc + hstep);
        CHECK_BETTER(down, tr + hstep, tc);
        CHECK_BETTER(diag, tr + hstep, tc + hstep);
        break;
      case 2:
        CHECK_BETTER(left, tr, tc - hstep);
        CHECK_BETTER(up, tr - hstep, tc);
        CHECK_BETTER(diag, tr - hstep, tc - hstep);
        break;
      case 3:
        CHECK_BETTER(right, tr, tc + hstep);
        CHECK_BETTER(up, tr - hstep, tc);
        CHECK_BETTER(diag, tr - hstep, tc + hstep);
        break;
    }
  } else {
    FIRST_LEVEL_CHECKS;
    if (halfiters > 1) {
      SECOND_LEVEL_CHECKS;
    }
  }

  tr = br;
  tc = bc;

  // Each subsequent iteration checks at least one point in common with
  // the last iteration could be 2 ( if diag selected) 1/4 pel

  // Note forced_stop: 0 - full, 1 - qtr only, 2 - half only
  if (forced_stop != 2) {
    hstep >>= 1;
    FIRST_LEVEL_CHECKS;
    if (quarteriters > 1) {
      SECOND_LEVEL_CHECKS;
    }
    tr = br;
    tc = bc;
  }

  if (allow_hp && use_mv_hp(ref_mv) && forced_stop == 0) {
    hstep >>= 1;
    FIRST_LEVEL_CHECKS;
    if (eighthiters > 1) {
      SECOND_LEVEL_CHECKS;
    }
    tr = br;
    tc = bc;
  }
  // These lines insure static analysis doesn't warn that
  // tr and tc aren't used after the above point.
  (void)tr;
  (void)tc;

  bestmv->row = br;
  bestmv->col = bc;

  return besterr;
}

/* clang-format off */
static const MV search_step_table[12] = {
  // left, right, up, down
  { 0, -4 }, { 0, 4 }, { -4, 0 }, { 4, 0 },
  { 0, -2 }, { 0, 2 }, { -2, 0 }, { 2, 0 },
  { 0, -1 }, { 0, 1 }, { -1, 0 }, { 1, 0 }
};
/* clang-format on */

static int accurate_sub_pel_search(
    const MACROBLOCKD *xd, const MV *this_mv, const struct scale_factors *sf,
    const InterpKernel *kernel, const vp9_variance_fn_ptr_t *vfp,
    const uint8_t *const src_address, const int src_stride,
    const uint8_t *const pre_address, int y_stride, const uint8_t *second_pred,
    int w, int h, uint32_t *sse) {
#if CONFIG_VP9_HIGHBITDEPTH
  uint64_t besterr;
  assert(sf->x_step_q4 == 16 && sf->y_step_q4 == 16);
  assert(w != 0 && h != 0);
  if (xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH) {
    DECLARE_ALIGNED(16, uint16_t, pred16[64 * 64]);
    vp9_highbd_build_inter_predictor(CONVERT_TO_SHORTPTR(pre_address), y_stride,
                                     pred16, w, this_mv, sf, w, h, 0, kernel,
                                     MV_PRECISION_Q3, 0, 0, xd->bd);
    if (second_pred != NULL) {
      DECLARE_ALIGNED(16, uint16_t, comp_pred16[64 * 64]);
      vpx_highbd_comp_avg_pred(comp_pred16, CONVERT_TO_SHORTPTR(second_pred), w,
                               h, pred16, w);
      besterr = vfp->vf(CONVERT_TO_BYTEPTR(comp_pred16), w, src_address,
                        src_stride, sse);
    } else {
      besterr =
          vfp->vf(CONVERT_TO_BYTEPTR(pred16), w, src_address, src_stride, sse);
    }
  } else {
    DECLARE_ALIGNED(16, uint8_t, pred[64 * 64]);
    vp9_build_inter_predictor(pre_address, y_stride, pred, w, this_mv, sf, w, h,
                              0, kernel, MV_PRECISION_Q3, 0, 0);
    if (second_pred != NULL) {
      DECLARE_ALIGNED(32, uint8_t, comp_pred[64 * 64]);
      vpx_comp_avg_pred(comp_pred, second_pred, w, h, pred, w);
      besterr = vfp->vf(comp_pred, w, src_address, src_stride, sse);
    } else {
      besterr = vfp->vf(pred, w, src_address, src_stride, sse);
    }
  }
  if (besterr >= UINT_MAX) return UINT_MAX;
  return (int)besterr;
#else
  int besterr;
  DECLARE_ALIGNED(16, uint8_t, pred[64 * 64]);
  assert(sf->x_step_q4 == 16 && sf->y_step_q4 == 16);
  assert(w != 0 && h != 0);
  (void)xd;

  vp9_build_inter_predictor(pre_address, y_stride, pred, w, this_mv, sf, w, h,
                            0, kernel, MV_PRECISION_Q3, 0, 0);
  if (second_pred != NULL) {
    DECLARE_ALIGNED(32, uint8_t, comp_pred[64 * 64]);
    vpx_comp_avg_pred(comp_pred, second_pred, w, h, pred, w);
    besterr = vfp->vf(comp_pred, w, src_address, src_stride, sse);
  } else {
    besterr = vfp->vf(pred, w, src_address, src_stride, sse);
  }
  return besterr;
#endif  // CONFIG_VP9_HIGHBITDEPTH
}

// TODO(yunqing): this part can be further refactored.
#if CONFIG_VP9_HIGHBITDEPTH
/* checks if (r, c) has better score than previous best */
#define CHECK_BETTER1(v, r, c)                                                \
  do {                                                                        \
    if (c >= minc && c <= maxc && r >= minr && r <= maxr) {                   \
      int64_t tmpmse;                                                         \
      const MV cb_mv = { r, c };                                              \
      const MV cb_ref_mv = { rr, rc };                                        \
      thismse = accurate_sub_pel_search(xd, &cb_mv, x->me_sf, kernel, vfp, z, \
                                        src_stride, y, y_stride, second_pred, \
                                        w, h, &sse);                          \
      tmpmse = thismse;                                                       \
      tmpmse +=                                                               \
          mv_err_cost(&cb_mv, &cb_ref_mv, mvjcost, mvcost, error_per_bit);    \
      if (tmpmse >= INT_MAX) {                                                \
        v = INT_MAX;                                                          \
      } else if ((v = (uint32_t)tmpmse) < besterr) {                          \
        besterr = v;                                                          \
        br = r;                                                               \
        bc = c;                                                               \
        *distortion = thismse;                                                \
        *sse1 = sse;                                                          \
      }                                                                       \
    } else {                                                                  \
      v = INT_MAX;                                                            \
    }                                                                         \
  } while (0)
#else
/* checks if (r, c) has better score than previous best */
#define CHECK_BETTER1(v, r, c)                                                \
  do {                                                                        \
    if (c >= minc && c <= maxc && r >= minr && r <= maxr) {                   \
      const MV cb_mv = { r, c };                                              \
      const MV cb_ref_mv = { rr, rc };                                        \
      thismse = accurate_sub_pel_search(xd, &cb_mv, x->me_sf, kernel, vfp, z, \
                                        src_stride, y, y_stride, second_pred, \
                                        w, h, &sse);                          \
      if ((v = mv_err_cost(&cb_mv, &cb_ref_mv, mvjcost, mvcost,               \
                           error_per_bit) +                                   \
               thismse) < besterr) {                                          \
        besterr = v;                                                          \
        br = r;                                                               \
        bc = c;                                                               \
        *distortion = thismse;                                                \
        *sse1 = sse;                                                          \
      }                                                                       \
    } else {                                                                  \
      v = INT_MAX;                                                            \
    }                                                                         \
  } while (0)

#endif

uint32_t vp9_find_best_sub_pixel_tree(
    const MACROBLOCK *x, MV *bestmv, const MV *ref_mv, int allow_hp,
    int error_per_bit, const vp9_variance_fn_ptr_t *vfp, int forced_stop,
    int iters_per_step, int *cost_list, int *mvjcost, int *mvcost[2],
    uint32_t *distortion, uint32_t *sse1, const uint8_t *second_pred, int w,
    int h, int use_accurate_subpel_search) {
  const uint8_t *const z = x->plane[0].src.buf;
  const uint8_t *const src_address = z;
  const int src_stride = x->plane[0].src.stride;
  const MACROBLOCKD *xd = &x->e_mbd;
  unsigned int besterr = UINT_MAX;
  unsigned int sse;
  int thismse;
  const int y_stride = xd->plane[0].pre[0].stride;
  const int offset = bestmv->row * y_stride + bestmv->col;
  const uint8_t *const y = xd->plane[0].pre[0].buf;

  int rr = ref_mv->row;
  int rc = ref_mv->col;
  int br = bestmv->row * 8;
  int bc = bestmv->col * 8;
  int hstep = 4;
  int iter, round = 3 - forced_stop;

  int minc, maxc, minr, maxr;
  int tr = br;
  int tc = bc;
  const MV *search_step = search_step_table;
  int idx, best_idx = -1;
  unsigned int cost_array[5];
  int kr, kc;
  MvLimits subpel_mv_limits;

  // TODO(yunqing): need to add 4-tap filter optimization to speed up the
  // encoder.
  const InterpKernel *kernel =
      (use_accurate_subpel_search > 0)
          ? ((use_accurate_subpel_search == USE_4_TAPS)
                 ? vp9_filter_kernels[FOURTAP]
                 : ((use_accurate_subpel_search == USE_8_TAPS)
                        ? vp9_filter_kernels[EIGHTTAP]
                        : vp9_filter_kernels[EIGHTTAP_SHARP]))
          : vp9_filter_kernels[BILINEAR];

  vp9_set_subpel_mv_search_range(&subpel_mv_limits, &x->mv_limits, ref_mv);
  minc = subpel_mv_limits.col_min;
  maxc = subpel_mv_limits.col_max;
  minr = subpel_mv_limits.row_min;
  maxr = subpel_mv_limits.row_max;

  if (!(allow_hp && use_mv_hp(ref_mv)))
    if (round == 3) round = 2;

  bestmv->row *= 8;
  bestmv->col *= 8;

  besterr = setup_center_error(xd, bestmv, ref_mv, error_per_bit, vfp, z,
                               src_stride, y, y_stride, second_pred, w, h,
                               offset, mvjcost, mvcost, sse1, distortion);

  (void)cost_list;  // to silence compiler warning

  for (iter = 0; iter < round; ++iter) {
    // Check vertical and horizontal sub-pixel positions.
    for (idx = 0; idx < 4; ++idx) {
      tr = br + search_step[idx].row;
      tc = bc + search_step[idx].col;
      if (tc >= minc && tc <= maxc && tr >= minr && tr <= maxr) {
        MV this_mv;
        this_mv.row = tr;
        this_mv.col = tc;

        if (use_accurate_subpel_search) {
          thismse = accurate_sub_pel_search(xd, &this_mv, x->me_sf, kernel, vfp,
                                            src_address, src_stride, y,
                                            y_stride, second_pred, w, h, &sse);
        } else {
          const uint8_t *const pre_address =
              y + (tr >> 3) * y_stride + (tc >> 3);
          if (second_pred == NULL)
            thismse = vfp->svf(pre_address, y_stride, sp(tc), sp(tr),
                               src_address, src_stride, &sse);
          else
            thismse = vfp->svaf(pre_address, y_stride, sp(tc), sp(tr),
                                src_address, src_stride, &sse, second_pred);
        }

        cost_array[idx] = thismse + mv_err_cost(&this_mv, ref_mv, mvjcost,
                                                mvcost, error_per_bit);

        if (cost_array[idx] < besterr) {
          best_idx = idx;
          besterr = cost_array[idx];
          *distortion = thismse;
          *sse1 = sse;
        }
      } else {
        cost_array[idx] = UINT_MAX;
      }
    }

    // Check diagonal sub-pixel position
    kc = (cost_array[0] <= cost_array[1] ? -hstep : hstep);
    kr = (cost_array[2] <= cost_array[3] ? -hstep : hstep);

    tc = bc + kc;
    tr = br + kr;
    if (tc >= minc && tc <= maxc && tr >= minr && tr <= maxr) {
      MV this_mv = { tr, tc };
      if (use_accurate_subpel_search) {
        thismse = accurate_sub_pel_search(xd, &this_mv, x->me_sf, kernel, vfp,
                                          src_address, src_stride, y, y_stride,
                                          second_pred, w, h, &sse);
      } else {
        const uint8_t *const pre_address = y + (tr >> 3) * y_stride + (tc >> 3);
        if (second_pred == NULL)
          thismse = vfp->svf(pre_address, y_stride, sp(tc), sp(tr), src_address,
                             src_stride, &sse);
        else
          thismse = vfp->svaf(pre_address, y_stride, sp(tc), sp(tr),
                              src_address, src_stride, &sse, second_pred);
      }

      cost_array[4] = thismse + mv_err_cost(&this_mv, ref_mv, mvjcost, mvcost,
                                            error_per_bit);

      if (cost_array[4] < besterr) {
        best_idx = 4;
        besterr = cost_array[4];
        *distortion = thismse;
        *sse1 = sse;
      }
    } else {
      cost_array[idx] = UINT_MAX;
    }

    if (best_idx < 4 && best_idx >= 0) {
      br += search_step[best_idx].row;
      bc += search_step[best_idx].col;
    } else if (best_idx == 4) {
      br = tr;
      bc = tc;
    }

    if (iters_per_step > 0 && best_idx != -1) {
      unsigned int second;
      const int br0 = br;
      const int bc0 = bc;
      assert(tr == br || tc == bc);

      if (tr == br && tc != bc) {
        kc = bc - tc;
        if (iters_per_step == 1) {
          if (use_accurate_subpel_search) {
            CHECK_BETTER1(second, br0, bc0 + kc);
          } else {
            CHECK_BETTER(second, br0, bc0 + kc);
          }
        }
      } else if (tr != br && tc == bc) {
        kr = br - tr;
        if (iters_per_step == 1) {
          if (use_accurate_subpel_search) {
            CHECK_BETTER1(second, br0 + kr, bc0);
          } else {
            CHECK_BETTER(second, br0 + kr, bc0);
          }
        }
      }

      if (iters_per_step > 1) {
        if (use_accurate_subpel_search) {
          CHECK_BETTER1(second, br0 + kr, bc0);
          CHECK_BETTER1(second, br0, bc0 + kc);
          if (br0 != br || bc0 != bc) {
            CHECK_BETTER1(second, br0 + kr, bc0 + kc);
          }
        } else {
          CHECK_BETTER(second, br0 + kr, bc0);
          CHECK_BETTER(second, br0, bc0 + kc);
          if (br0 != br || bc0 != bc) {
            CHECK_BETTER(second, br0 + kr, bc0 + kc);
          }
        }
      }
    }

    search_step += 4;
    hstep >>= 1;
    best_idx = -1;
  }

  // Each subsequent iteration checks at least one point in common with
  // the last iteration could be 2 ( if diag selected) 1/4 pel

  // These lines insure static analysis doesn't warn that
  // tr and tc aren't used after the above point.
  (void)tr;
  (void)tc;

  bestmv->row = br;
  bestmv->col = bc;

  return besterr;
}

#undef CHECK_BETTER
#undef CHECK_BETTER1

static INLINE int check_bounds(const MvLimits *mv_limits, int row, int col,
                               int range) {
  return ((row - range) >= mv_limits->row_min) &
         ((row + range) <= mv_limits->row_max) &
         ((col - range) >= mv_limits->col_min) &
         ((col + range) <= mv_limits->col_max);
}

static INLINE int is_mv_in(const MvLimits *mv_limits, const MV *mv) {
  return (mv->col >= mv_limits->col_min) && (mv->col <= mv_limits->col_max) &&
         (mv->row >= mv_limits->row_min) && (mv->row <= mv_limits->row_max);
}

#define CHECK_BETTER                                                      \
  {                                                                       \
    if (thissad < bestsad) {                                              \
      if (use_mvcost)                                                     \
        thissad += mvsad_err_cost(x, &this_mv, &fcenter_mv, sad_per_bit); \
      if (thissad < bestsad) {                                            \
        bestsad = thissad;                                                \
        best_site = i;                                                    \
      }                                                                   \
    }                                                                     \
  }

#define MAX_PATTERN_SCALES 11
#define MAX_PATTERN_CANDIDATES 8  // max number of candidates per scale
#define PATTERN_CANDIDATES_REF 3  // number of refinement candidates

// Calculate and return a sad+mvcost list around an integer best pel.
static INLINE void calc_int_cost_list(const MACROBLOCK *x, const MV *ref_mv,
                                      int sadpb,
                                      const vp9_variance_fn_ptr_t *fn_ptr,
                                      const MV *best_mv, int *cost_list) {
  static const MV neighbors[4] = { { 0, -1 }, { 1, 0 }, { 0, 1 }, { -1, 0 } };
  const struct buf_2d *const what = &x->plane[0].src;
  const struct buf_2d *const in_what = &x->e_mbd.plane[0].pre[0];
  const MV fcenter_mv = { ref_mv->row >> 3, ref_mv->col >> 3 };
  int br = best_mv->row;
  int bc = best_mv->col;
  const MV mv = { br, bc };
  int i;
  unsigned int sse;

  cost_list[0] =
      fn_ptr->vf(what->buf, what->stride, get_buf_from_mv(in_what, &mv),
                 in_what->stride, &sse) +
      mvsad_err_cost(x, &mv, &fcenter_mv, sadpb);
  if (check_bounds(&x->mv_limits, br, bc, 1)) {
    for (i = 0; i < 4; i++) {
      const MV this_mv = { br + neighbors[i].row, bc + neighbors[i].col };
      cost_list[i + 1] = fn_ptr->vf(what->buf, what->stride,
                                    get_buf_from_mv(in_what, &this_mv),
                                    in_what->stride, &sse) +
                         mv_err_cost(&this_mv, &fcenter_mv, x->nmvjointcost,
                                     x->mvcost, x->errorperbit);
    }
  } else {
    for (i = 0; i < 4; i++) {
      const MV this_mv = { br + neighbors[i].row, bc + neighbors[i].col };
      if (!is_mv_in(&x->mv_limits, &this_mv))
        cost_list[i + 1] = INT_MAX;
      else
        cost_list[i + 1] = fn_ptr->vf(what->buf, what->stride,
                                      get_buf_from_mv(in_what, &this_mv),
                                      in_what->stride, &sse) +
                           mv_err_cost(&this_mv, &fcenter_mv, x->nmvjointcost,
                                       x->mvcost, x->errorperbit);
    }
  }
}

// Generic pattern search function that searches over multiple scales.
// Each scale can have a different number of candidates and shape of
// candidates as indicated in the num_candidates and candidates arrays
// passed into this function
//
static int vp9_pattern_search(
    const MACROBLOCK *x, MV *ref_mv, int search_param, int sad_per_bit,
    int do_init_search, int *cost_list, const vp9_variance_fn_ptr_t *vfp,
    int use_mvcost, const MV *center_mv, MV *best_mv,
    const int num_candidates[MAX_PATTERN_SCALES],
    const MV candidates[MAX_PATTERN_SCALES][MAX_PATTERN_CANDIDATES]) {
  const MACROBLOCKD *const xd = &x->e_mbd;
  static const int search_param_to_steps[MAX_MVSEARCH_STEPS] = {
    10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
  };
  int i, s, t;
  const struct buf_2d *const what = &x->plane[0].src;
  const struct buf_2d *const in_what = &xd->plane[0].pre[0];
  int br, bc;
  int bestsad = INT_MAX;
  int thissad;
  int k = -1;
  const MV fcenter_mv = { center_mv->row >> 3, center_mv->col >> 3 };
  int best_init_s = search_param_to_steps[search_param];
  // adjust ref_mv to make sure it is within MV range
  clamp_mv(ref_mv, x->mv_limits.col_min, x->mv_limits.col_max,
           x->mv_limits.row_min, x->mv_limits.row_max);
  br = ref_mv->row;
  bc = ref_mv->col;

  // Work out the start point for the search
  bestsad = vfp->sdf(what->buf, what->stride, get_buf_from_mv(in_what, ref_mv),
                     in_what->stride) +
            mvsad_err_cost(x, ref_mv, &fcenter_mv, sad_per_bit);

  // Search all possible scales up to the search param around the center point
  // pick the scale of the point that is best as the starting scale of
  // further steps around it.
  if (do_init_search) {
    s = best_init_s;
    best_init_s = -1;
    for (t = 0; t <= s; ++t) {
      int best_site = -1;
      if (check_bounds(&x->mv_limits, br, bc, 1 << t)) {
        for (i = 0; i < num_candidates[t]; i++) {
          const MV this_mv = { br + candidates[t][i].row,
                               bc + candidates[t][i].col };
          thissad =
              vfp->sdf(what->buf, what->stride,
                       get_buf_from_mv(in_what, &this_mv), in_what->stride);
          CHECK_BETTER
        }
      } else {
        for (i = 0; i < num_candidates[t]; i++) {
          const MV this_mv = { br + candidates[t][i].row,
                               bc + candidates[t][i].col };
          if (!is_mv_in(&x->mv_limits, &this_mv)) continue;
          thissad =
              vfp->sdf(what->buf, what->stride,
                       get_buf_from_mv(in_what, &this_mv), in_what->stride);
          CHECK_BETTER
        }
      }
      if (best_site == -1) {
        continue;
      } else {
        best_init_s = t;
        k = best_site;
      }
    }
    if (best_init_s != -1) {
      br += candidates[best_init_s][k].row;
      bc += candidates[best_init_s][k].col;
    }
  }

  // If the center point is still the best, just skip this and move to
  // the refinement step.
  if (best_init_s != -1) {
    int best_site = -1;
    s = best_init_s;

    do {
      // No need to search all 6 points the 1st time if initial search was used
      if (!do_init_search || s != best_init_s) {
        if (check_bounds(&x->mv_limits, br, bc, 1 << s)) {
          for (i = 0; i < num_candidates[s]; i++) {
            const MV this_mv = { br + candidates[s][i].row,
                                 bc + candidates[s][i].col };
            thissad =
                vfp->sdf(what->buf, what->stride,
                         get_buf_from_mv(in_what, &this_mv), in_what->stride);
            CHECK_BETTER
          }
        } else {
          for (i = 0; i < num_candidates[s]; i++) {
            const MV this_mv = { br + candidates[s][i].row,
                                 bc + candidates[s][i].col };
            if (!is_mv_in(&x->mv_limits, &this_mv)) continue;
            thissad =
                vfp->sdf(what->buf, what->stride,
                         get_buf_from_mv(in_what, &this_mv), in_what->stride);
            CHECK_BETTER
          }
        }

        if (best_site == -1) {
          continue;
        } else {
          br += candidates[s][best_site].row;
          bc += candidates[s][best_site].col;
          k = best_site;
        }
      }

      do {
        int next_chkpts_indices[PATTERN_CANDIDATES_REF];
        best_site = -1;
        next_chkpts_indices[0] = (k == 0) ? num_candidates[s] - 1 : k - 1;
        next_chkpts_indices[1] = k;
        next_chkpts_indices[2] = (k == num_candidates[s] - 1) ? 0 : k + 1;

        if (check_bounds(&x->mv_limits, br, bc, 1 << s)) {
          for (i = 0; i < PATTERN_CANDIDATES_REF; i++) {
            const MV this_mv = {
              br + candidates[s][next_chkpts_indices[i]].row,
              bc + candidates[s][next_chkpts_indices[i]].col
            };
            thissad =
                vfp->sdf(what->buf, what->stride,
                         get_buf_from_mv(in_what, &this_mv), in_what->stride);
            CHECK_BETTER
          }
        } else {
          for (i = 0; i < PATTERN_CANDIDATES_REF; i++) {
            const MV this_mv = {
              br + candidates[s][next_chkpts_indices[i]].row,
              bc + candidates[s][next_chkpts_indices[i]].col
            };
            if (!is_mv_in(&x->mv_limits, &this_mv)) continue;
            thissad =
                vfp->sdf(what->buf, what->stride,
                         get_buf_from_mv(in_what, &this_mv), in_what->stride);
            CHECK_BETTER
          }
        }

        if (best_site != -1) {
          k = next_chkpts_indices[best_site];
          br += candidates[s][k].row;
          bc += candidates[s][k].col;
        }
      } while (best_site != -1);
    } while (s--);
  }

  best_mv->row = br;
  best_mv->col = bc;

  // Returns the one-away integer pel sad values around the best as follows:
  // cost_list[0]: cost at the best integer pel
  // cost_list[1]: cost at delta {0, -1} (left)   from the best integer pel
  // cost_list[2]: cost at delta { 1, 0} (bottom) from the best integer pel
  // cost_list[3]: cost at delta { 0, 1} (right)  from the best integer pel
  // cost_list[4]: cost at delta {-1, 0} (top)    from the best integer pel
  if (cost_list) {
    calc_int_cost_list(x, &fcenter_mv, sad_per_bit, vfp, best_mv, cost_list);
  }
  return bestsad;
}

// A specialized function where the smallest scale search candidates
// are 4 1-away neighbors, and cost_list is non-null
// TODO(debargha): Merge this function with the one above. Also remove
// use_mvcost option since it is always 1, to save unnecessary branches.
static int vp9_pattern_search_sad(
    const MACROBLOCK *x, MV *ref_mv, int search_param, int sad_per_bit,
    int do_init_search, int *cost_list, const vp9_variance_fn_ptr_t *vfp,
    int use_mvcost, const MV *center_mv, MV *best_mv,
    const int num_candidates[MAX_PATTERN_SCALES],
    const MV candidates[MAX_PATTERN_SCALES][MAX_PATTERN_CANDIDATES]) {
  const MACROBLOCKD *const xd = &x->e_mbd;
  static const int search_param_to_steps[MAX_MVSEARCH_STEPS] = {
    10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
  };
  int i, s, t;
  const struct buf_2d *const what = &x->plane[0].src;
  const struct buf_2d *const in_what = &xd->plane[0].pre[0];
  int br, bc;
  int bestsad = INT_MAX;
  int thissad;
  int k = -1;
  const MV fcenter_mv = { center_mv->row >> 3, center_mv->col >> 3 };
  int best_init_s = search_param_to_steps[search_param];
  // adjust ref_mv to make sure it is within MV range
  clamp_mv(ref_mv, x->mv_limits.col_min, x->mv_limits.col_max,
           x->mv_limits.row_min, x->mv_limits.row_max);
  br = ref_mv->row;
  bc = ref_mv->col;
  if (cost_list != NULL) {
    cost_list[0] = cost_list[1] = cost_list[2] = cost_list[3] = cost_list[4] =
        INT_MAX;
  }

  // Work out the start point for the search
  bestsad = vfp->sdf(what->buf, what->stride, get_buf_from_mv(in_what, ref_mv),
                     in_what->stride) +
            mvsad_err_cost(x, ref_mv, &fcenter_mv, sad_per_bit);

  // Search all possible scales up to the search param around the center point
  // pick the scale of the point that is best as the starting scale of
  // further steps around it.
  if (do_init_search) {
    s = best_init_s;
    best_init_s = -1;
    for (t = 0; t <= s; ++t) {
      int best_site = -1;
      if (check_bounds(&x->mv_limits, br, bc, 1 << t)) {
        for (i = 0; i < num_candidates[t]; i++) {
          const MV this_mv = { br + candidates[t][i].row,
                               bc + candidates[t][i].col };
          thissad =
              vfp->sdf(what->buf, what->stride,
                       get_buf_from_mv(in_what, &this_mv), in_what->stride);
          CHECK_BETTER
        }
      } else {
        for (i = 0; i < num_candidates[t]; i++) {
          const MV this_mv = { br + candidates[t][i].row,
                               bc + candidates[t][i].col };
          if (!is_mv_in(&x->mv_limits, &this_mv)) continue;
          thissad =
              vfp->sdf(what->buf, what->stride,
                       get_buf_from_mv(in_what, &this_mv), in_what->stride);
          CHECK_BETTER
        }
      }
      if (best_site == -1) {
        continue;
      } else {
        best_init_s = t;
        k = best_site;
      }
    }
    if (best_init_s != -1) {
      br += candidates[best_init_s][k].row;
      bc += candidates[best_init_s][k].col;
    }
  }

  // If the center point is still the best, just skip this and move to
  // the refinement step.
  if (best_init_s != -1) {
    int do_sad = (num_candidates[0] == 4 && cost_list != NULL);
    int best_site = -1;
    s = best_init_s;

    for (; s >= do_sad; s--) {
      if (!do_init_search || s != best_init_s) {
        if (check_bounds(&x->mv_limits, br, bc, 1 << s)) {
          for (i = 0; i < num_candidates[s]; i++) {
            const MV this_mv = { br + candidates[s][i].row,
                                 bc + candidates[s][i].col };
            thissad =
                vfp->sdf(what->buf, what->stride,
                         get_buf_from_mv(in_what, &this_mv), in_what->stride);
            CHECK_BETTER
          }
        } else {
          for (i = 0; i < num_candidates[s]; i++) {
            const MV this_mv = { br + candidates[s][i].row,
                                 bc + candidates[s][i].col };
            if (!is_mv_in(&x->mv_limits, &this_mv)) continue;
            thissad =
                vfp->sdf(what->buf, what->stride,
                         get_buf_from_mv(in_what, &this_mv), in_what->stride);
            CHECK_BETTER
          }
        }

        if (best_site == -1) {
          continue;
        } else {
          br += candidates[s][best_site].row;
          bc += candidates[s][best_site].col;
          k = best_site;
        }
      }

      do {
        int next_chkpts_indices[PATTERN_CANDIDATES_REF];
        best_site = -1;
        next_chkpts_indices[0] = (k == 0) ? num_candidates[s] - 1 : k - 1;
        next_chkpts_indices[1] = k;
        next_chkpts_indices[2] = (k == num_candidates[s] - 1) ? 0 : k + 1;

        if (check_bounds(&x->mv_limits, br, bc, 1 << s)) {
          for (i = 0; i < PATTERN_CANDIDATES_REF; i++) {
            const MV this_mv = {
              br + candidates[s][next_chkpts_indices[i]].row,
              bc + candidates[s][next_chkpts_indices[i]].col
            };
            thissad =
                vfp->sdf(what->buf, what->stride,
                         get_buf_from_mv(in_what, &this_mv), in_what->stride);
            CHECK_BETTER
          }
        } else {
          for (i = 0; i < PATTERN_CANDIDATES_REF; i++) {
            const MV this_mv = {
              br + candidates[s][next_chkpts_indices[i]].row,
              bc + candidates[s][next_chkpts_indices[i]].col
            };
            if (!is_mv_in(&x->mv_limits, &this_mv)) continue;
            thissad =
                vfp->sdf(what->buf, what->stride,
                         get_buf_from_mv(in_what, &this_mv), in_what->stride);
            CHECK_BETTER
          }
        }

        if (best_site != -1) {
          k = next_chkpts_indices[best_site];
          br += candidates[s][k].row;
          bc += candidates[s][k].col;
        }
      } while (best_site != -1);
    }

    // Note: If we enter the if below, then cost_list must be non-NULL.
    if (s == 0) {
      cost_list[0] = bestsad;
      if (!do_init_search || s != best_init_s) {
        if (check_bounds(&x->mv_limits, br, bc, 1 << s)) {
          for (i = 0; i < num_candidates[s]; i++) {
            const MV this_mv = { br + candidates[s][i].row,
                                 bc + candidates[s][i].col };
            cost_list[i + 1] = thissad =
                vfp->sdf(what->buf, what->stride,
                         get_buf_from_mv(in_what, &this_mv), in_what->stride);
            CHECK_BETTER
          }
        } else {
          for (i = 0; i < num_candidates[s]; i++) {
            const MV this_mv = { br + candidates[s][i].row,
                                 bc + candidates[s][i].col };
            if (!is_mv_in(&x->mv_limits, &this_mv)) continue;
            cost_list[i + 1] = thissad =
                vfp->sdf(what->buf, what->stride,
                         get_buf_from_mv(in_what, &this_mv), in_what->stride);
            CHECK_BETTER
          }
        }

        if (best_site != -1) {
          br += candidates[s][best_site].row;
          bc += candidates[s][best_site].col;
          k = best_site;
        }
      }
      while (best_site != -1) {
        int next_chkpts_indices[PATTERN_CANDIDATES_REF];
        best_site = -1;
        next_chkpts_indices[0] = (k == 0) ? num_candidates[s] - 1 : k - 1;
        next_chkpts_indices[1] = k;
        next_chkpts_indices[2] = (k == num_candidates[s] - 1) ? 0 : k + 1;
        cost_list[1] = cost_list[2] = cost_list[3] = cost_list[4] = INT_MAX;
        cost_list[((k + 2) % 4) + 1] = cost_list[0];
        cost_list[0] = bestsad;

        if (check_bounds(&x->mv_limits, br, bc, 1 << s)) {
          for (i = 0; i < PATTERN_CANDIDATES_REF; i++) {
            const MV this_mv = {
              br + candidates[s][next_chkpts_indices[i]].row,
              bc + candidates[s][next_chkpts_indices[i]].col
            };
            cost_list[next_chkpts_indices[i] + 1] = thissad =
                vfp->sdf(what->buf, what->stride,
                         get_buf_from_mv(in_what, &this_mv), in_what->stride);
            CHECK_BETTER
          }
        } else {
          for (i = 0; i < PATTERN_CANDIDATES_REF; i++) {
            const MV this_mv = {
              br + candidates[s][next_chkpts_indices[i]].row,
              bc + candidates[s][next_chkpts_indices[i]].col
            };
            if (!is_mv_in(&x->mv_limits, &this_mv)) {
              cost_list[next_chkpts_indices[i] + 1] = INT_MAX;
              continue;
            }
            cost_list[next_chkpts_indices[i] + 1] = thissad =
                vfp->sdf(what->buf, what->stride,
                         get_buf_from_mv(in_what, &this_mv), in_what->stride);
            CHECK_BETTER
          }
        }

        if (best_site != -1) {
          k = next_chkpts_indices[best_site];
          br += candidates[s][k].row;
          bc += candidates[s][k].col;
        }
      }
    }
  }

  // Returns the one-away integer pel sad values around the best as follows:
  // cost_list[0]: sad at the best integer pel
  // cost_list[1]: sad at delta {0, -1} (left)   from the best integer pel
  // cost_list[2]: sad at delta { 1, 0} (bottom) from the best integer pel
  // cost_list[3]: sad at delta { 0, 1} (right)  from the best integer pel
  // cost_list[4]: sad at delta {-1, 0} (top)    from the best integer pel
  if (cost_list) {
    static const MV neighbors[4] = { { 0, -1 }, { 1, 0 }, { 0, 1 }, { -1, 0 } };
    if (cost_list[0] == INT_MAX) {
      cost_list[0] = bestsad;
      if (check_bounds(&x->mv_limits, br, bc, 1)) {
        for (i = 0; i < 4; i++) {
          const MV this_mv = { br + neighbors[i].row, bc + neighbors[i].col };
          cost_list[i + 1] =
              vfp->sdf(what->buf, what->stride,
                       get_buf_from_mv(in_what, &this_mv), in_what->stride);
        }
      } else {
        for (i = 0; i < 4; i++) {
          const MV this_mv = { br + neighbors[i].row, bc + neighbors[i].col };
          if (!is_mv_in(&x->mv_limits, &this_mv))
            cost_list[i + 1] = INT_MAX;
          else
            cost_list[i + 1] =
                vfp->sdf(what->buf, what->stride,
                         get_buf_from_mv(in_what, &this_mv), in_what->stride);
        }
      }
    } else {
      if (use_mvcost) {
        for (i = 0; i < 4; i++) {
          const MV this_mv = { br + neighbors[i].row, bc + neighbors[i].col };
          if (cost_list[i + 1] != INT_MAX) {
            cost_list[i + 1] +=
                mvsad_err_cost(x, &this_mv, &fcenter_mv, sad_per_bit);
          }
        }
      }
    }
  }
  best_mv->row = br;
  best_mv->col = bc;
  return bestsad;
}

int vp9_get_mvpred_var(const MACROBLOCK *x, const MV *best_mv,
                       const MV *center_mv, const vp9_variance_fn_ptr_t *vfp,
                       int use_mvcost) {
  const MACROBLOCKD *const xd = &x->e_mbd;
  const struct buf_2d *const what = &x->plane[0].src;
  const struct buf_2d *const in_what = &xd->plane[0].pre[0];
  const MV mv = { best_mv->row * 8, best_mv->col * 8 };
  uint32_t unused;
#if CONFIG_VP9_HIGHBITDEPTH
  uint64_t err =
      vfp->vf(what->buf, what->stride, get_buf_from_mv(in_what, best_mv),
              in_what->stride, &unused);
  err += (use_mvcost ? mv_err_cost(&mv, center_mv, x->nmvjointcost, x->mvcost,
                                   x->errorperbit)
                     : 0);
  if (err >= INT_MAX) return INT_MAX;
  return (int)err;
#else
  return vfp->vf(what->buf, what->stride, get_buf_from_mv(in_what, best_mv),
                 in_what->stride, &unused) +
         (use_mvcost ? mv_err_cost(&mv, center_mv, x->nmvjointcost, x->mvcost,
                                   x->errorperbit)
                     : 0);
#endif
}

int vp9_get_mvpred_av_var(const MACROBLOCK *x, const MV *best_mv,
                          const MV *center_mv, const uint8_t *second_pred,
                          const vp9_variance_fn_ptr_t *vfp, int use_mvcost) {
  const MACROBLOCKD *const xd = &x->e_mbd;
  const struct buf_2d *const what = &x->plane[0].src;
  const struct buf_2d *const in_what = &xd->plane[0].pre[0];
  const MV mv = { best_mv->row * 8, best_mv->col * 8 };
  unsigned int unused;

  return vfp->svaf(get_buf_from_mv(in_what, best_mv), in_what->stride, 0, 0,
                   what->buf, what->stride, &unused, second_pred) +
         (use_mvcost ? mv_err_cost(&mv, center_mv, x->nmvjointcost, x->mvcost,
                                   x->errorperbit)
                     : 0);
}

static int hex_search(const MACROBLOCK *x, MV *ref_mv, int search_param,
                      int sad_per_bit, int do_init_search, int *cost_list,
                      const vp9_variance_fn_ptr_t *vfp, int use_mvcost,
                      const MV *center_mv, MV *best_mv) {
  // First scale has 8-closest points, the rest have 6 points in hex shape
  // at increasing scales
  static const int hex_num_candidates[MAX_PATTERN_SCALES] = { 8, 6, 6, 6, 6, 6,
                                                              6, 6, 6, 6, 6 };
  // Note that the largest candidate step at each scale is 2^scale
  /* clang-format off */
  static const MV hex_candidates[MAX_PATTERN_SCALES][MAX_PATTERN_CANDIDATES] = {
    { { -1, -1 }, { 0, -1 }, { 1, -1 }, { 1, 0 }, { 1, 1 }, { 0, 1 }, { -1, 1 },
      { -1, 0 } },
    { { -1, -2 }, { 1, -2 }, { 2, 0 }, { 1, 2 }, { -1, 2 }, { -2, 0 } },
    { { -2, -4 }, { 2, -4 }, { 4, 0 }, { 2, 4 }, { -2, 4 }, { -4, 0 } },
    { { -4, -8 }, { 4, -8 }, { 8, 0 }, { 4, 8 }, { -4, 8 }, { -8, 0 } },
    { { -8, -16 }, { 8, -16 }, { 16, 0 }, { 8, 16 }, { -8, 16 }, { -16, 0 } },
    { { -16, -32 }, { 16, -32 }, { 32, 0 }, { 16, 32 }, { -16, 32 },
      { -32, 0 } },
    { { -32, -64 }, { 32, -64 }, { 64, 0 }, { 32, 64 }, { -32, 64 },
      { -64, 0 } },
    { { -64, -128 }, { 64, -128 }, { 128, 0 }, { 64, 128 }, { -64, 128 },
      { -128, 0 } },
    { { -128, -256 }, { 128, -256 }, { 256, 0 }, { 128, 256 }, { -128, 256 },
      { -256, 0 } },
    { { -256, -512 }, { 256, -512 }, { 512, 0 }, { 256, 512 }, { -256, 512 },
      { -512, 0 } },
    { { -512, -1024 }, { 512, -1024 }, { 1024, 0 }, { 512, 1024 },
      { -512, 1024 }, { -1024, 0 } }
  };
  /* clang-format on */
  return vp9_pattern_search(
      x, ref_mv, search_param, sad_per_bit, do_init_search, cost_list, vfp,
      use_mvcost, center_mv, best_mv, hex_num_candidates, hex_candidates);
}

static int bigdia_search(const MACROBLOCK *x, MV *ref_mv, int search_param,
                         int sad_per_bit, int do_init_search, int *cost_list,
                         const vp9_variance_fn_ptr_t *vfp, int use_mvcost,
                         const MV *center_mv, MV *best_mv) {
  // First scale has 4-closest points, the rest have 8 points in diamond
  // shape at increasing scales
  static const int bigdia_num_candidates[MAX_PATTERN_SCALES] = {
    4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
  };
  // Note that the largest candidate step at each scale is 2^scale
  /* clang-format off */
  static const MV
      bigdia_candidates[MAX_PATTERN_SCALES][MAX_PATTERN_CANDIDATES] = {
        { { 0, -1 }, { 1, 0 }, { 0, 1 }, { -1, 0 } },
        { { -1, -1 }, { 0, -2 }, { 1, -1 }, { 2, 0 }, { 1, 1 }, { 0, 2 },
          { -1, 1 }, { -2, 0 } },
        { { -2, -2 }, { 0, -4 }, { 2, -2 }, { 4, 0 }, { 2, 2 }, { 0, 4 },
          { -2, 2 }, { -4, 0 } },
        { { -4, -4 }, { 0, -8 }, { 4, -4 }, { 8, 0 }, { 4, 4 }, { 0, 8 },
          { -4, 4 }, { -8, 0 } },
        { { -8, -8 }, { 0, -16 }, { 8, -8 }, { 16, 0 }, { 8, 8 }, { 0, 16 },
          { -8, 8 }, { -16, 0 } },
        { { -16, -16 }, { 0, -32 }, { 16, -16 }, { 32, 0 }, { 16, 16 },
          { 0, 32 }, { -16, 16 }, { -32, 0 } },
        { { -32, -32 }, { 0, -64 }, { 32, -32 }, { 64, 0 }, { 32, 32 },
          { 0, 64 }, { -32, 32 }, { -64, 0 } },
        { { -64, -64 }, { 0, -128 }, { 64, -64 }, { 128, 0 }, { 64, 64 },
          { 0, 128 }, { -64, 64 }, { -128, 0 } },
        { { -128, -128 }, { 0, -256 }, { 128, -128 }, { 256, 0 }, { 128, 128 },
          { 0, 256 }, { -128, 128 }, { -256, 0 } },
        { { -256, -256 }, { 0, -512 }, { 256, -256 }, { 512, 0 }, { 256, 256 },
          { 0, 512 }, { -256, 256 }, { -512, 0 } },
        { { -512, -512 }, { 0, -1024 }, { 512, -512 }, { 1024, 0 },
          { 512, 512 }, { 0, 1024 }, { -512, 512 }, { -1024, 0 } }
      };
  /* clang-format on */
  return vp9_pattern_search_sad(
      x, ref_mv, search_param, sad_per_bit, do_init_search, cost_list, vfp,
      use_mvcost, center_mv, best_mv, bigdia_num_candidates, bigdia_candidates);
}

static int square_search(const MACROBLOCK *x, MV *ref_mv, int search_param,
                         int sad_per_bit, int do_init_search, int *cost_list,
                         const vp9_variance_fn_ptr_t *vfp, int use_mvcost,
                         const MV *center_mv, MV *best_mv) {
  // All scales have 8 closest points in square shape
  static const int square_num_candidates[MAX_PATTERN_SCALES] = {
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
  };
  // Note that the largest candidate step at each scale is 2^scale
  /* clang-format off */
  static const MV
      square_candidates[MAX_PATTERN_SCALES][MAX_PATTERN_CANDIDATES] = {
        { { -1, -1 }, { 0, -1 }, { 1, -1 }, { 1, 0 }, { 1, 1 }, { 0, 1 },
          { -1, 1 }, { -1, 0 } },
        { { -2, -2 }, { 0, -2 }, { 2, -2 }, { 2, 0 }, { 2, 2 }, { 0, 2 },
          { -2, 2 }, { -2, 0 } },
        { { -4, -4 }, { 0, -4 }, { 4, -4 }, { 4, 0 }, { 4, 4 }, { 0, 4 },
          { -4, 4 }, { -4, 0 } },
        { { -8, -8 }, { 0, -8 }, { 8, -8 }, { 8, 0 }, { 8, 8 }, { 0, 8 },
          { -8, 8 }, { -8, 0 } },
        { { -16, -16 }, { 0, -16 }, { 16, -16 }, { 16, 0 }, { 16, 16 },
          { 0, 16 }, { -16, 16 }, { -16, 0 } },
        { { -32, -32 }, { 0, -32 }, { 32, -32 }, { 32, 0 }, { 32, 32 },
          { 0, 32 }, { -32, 32 }, { -32, 0 } },
        { { -64, -64 }, { 0, -64 }, { 64, -64 }, { 64, 0 }, { 64, 64 },
          { 0, 64 }, { -64, 64 }, { -64, 0 } },
        { { -128, -128 }, { 0, -128 }, { 128, -128 }, { 128, 0 }, { 128, 128 },
          { 0, 128 }, { -128, 128 }, { -128, 0 } },
        { { -256, -256 }, { 0, -256 }, { 256, -256 }, { 256, 0 }, { 256, 256 },
          { 0, 256 }, { -256, 256 }, { -256, 0 } },
        { { -512, -512 }, { 0, -512 }, { 512, -512 }, { 512, 0 }, { 512, 512 },
          { 0, 512 }, { -512, 512 }, { -512, 0 } },
        { { -1024, -1024 }, { 0, -1024 }, { 1024, -1024 }, { 1024, 0 },
          { 1024, 1024 }, { 0, 1024 }, { -1024, 1024 }, { -1024, 0 } }
      };
  /* clang-format on */
  return vp9_pattern_search(
      x, ref_mv, search_param, sad_per_bit, do_init_search, cost_list, vfp,
      use_mvcost, center_mv, best_mv, square_num_candidates, square_candidates);
}

static int fast_hex_search(const MACROBLOCK *x, MV *ref_mv, int search_param,
                           int sad_per_bit,
                           int do_init_search,  // must be zero for fast_hex
                           int *cost_list, const vp9_variance_fn_ptr_t *vfp,
                           int use_mvcost, const MV *center_mv, MV *best_mv) {
  return hex_search(x, ref_mv, VPXMAX(MAX_MVSEARCH_STEPS - 2, search_param),
                    sad_per_bit, do_init_search, cost_list, vfp, use_mvcost,
                    center_mv, best_mv);
}

static int fast_dia_search(const MACROBLOCK *x, MV *ref_mv, int search_param,
                           int sad_per_bit, int do_init_search, int *cost_list,
                           const vp9_variance_fn_ptr_t *vfp, int use_mvcost,
                           const MV *center_mv, MV *best_mv) {
  return bigdia_search(x, ref_mv, VPXMAX(MAX_MVSEARCH_STEPS - 2, search_param),
                       sad_per_bit, do_init_search, cost_list, vfp, use_mvcost,
                       center_mv, best_mv);
}

#undef CHECK_BETTER

// Exhuastive motion search around a given centre position with a given
// step size.
static int exhaustive_mesh_search(const MACROBLOCK *x, MV *ref_mv, MV *best_mv,
                                  int range, int step, int sad_per_bit,
                                  const vp9_variance_fn_ptr_t *fn_ptr,
                                  const MV *center_mv) {
  const MACROBLOCKD *const xd = &x->e_mbd;
  const struct buf_2d *const what = &x->plane[0].src;
  const struct buf_2d *const in_what = &xd->plane[0].pre[0];
  MV fcenter_mv = { center_mv->row, center_mv->col };
  unsigned int best_sad = INT_MAX;
  int r, c, i;
  int start_col, end_col, start_row, end_row;
  int col_step = (step > 1) ? step : 4;

  assert(step >= 1);

  clamp_mv(&fcenter_mv, x->mv_limits.col_min, x->mv_limits.col_max,
           x->mv_limits.row_min, x->mv_limits.row_max);
  *best_mv = fcenter_mv;
  best_sad =
      fn_ptr->sdf(what->buf, what->stride,
                  get_buf_from_mv(in_what, &fcenter_mv), in_what->stride) +
      mvsad_err_cost(x, &fcenter_mv, ref_mv, sad_per_bit);
  start_row = VPXMAX(-range, x->mv_limits.row_min - fcenter_mv.row);
  start_col = VPXMAX(-range, x->mv_limits.col_min - fcenter_mv.col);
  end_row = VPXMIN(range, x->mv_limits.row_max - fcenter_mv.row);
  end_col = VPXMIN(range, x->mv_limits.col_max - fcenter_mv.col);

  for (r = start_row; r <= end_row; r += step) {
    for (c = start_col; c <= end_col; c += col_step) {
      // Step > 1 means we are not checking every location in this pass.
      if (step > 1) {
        const MV mv = { fcenter_mv.row + r, fcenter_mv.col + c };
        unsigned int sad =
            fn_ptr->sdf(what->buf, what->stride, get_buf_from_mv(in_what, &mv),
                        in_what->stride);
        if (sad < best_sad) {
          sad += mvsad_err_cost(x, &mv, ref_mv, sad_per_bit);
          if (sad < best_sad) {
            best_sad = sad;
            *best_mv = mv;
          }
        }
      } else {
        // 4 sads in a single call if we are checking every location
        if (c + 3 <= end_col) {
          unsigned int sads[4];
          const uint8_t *addrs[4];
          for (i = 0; i < 4; ++i) {
            const MV mv = { fcenter_mv.row + r, fcenter_mv.col + c + i };
            addrs[i] = get_buf_from_mv(in_what, &mv);
          }
          fn_ptr->sdx4df(what->buf, what->stride, addrs, in_what->stride, sads);

          for (i = 0; i < 4; ++i) {
            if (sads[i] < best_sad) {
              const MV mv = { fcenter_mv.row + r, fcenter_mv.col + c + i };
              const unsigned int sad =
                  sads[i] + mvsad_err_cost(x, &mv, ref_mv, sad_per_bit);
              if (sad < best_sad) {
                best_sad = sad;
                *best_mv = mv;
              }
            }
          }
        } else {
          for (i = 0; i < end_col - c; ++i) {
            const MV mv = { fcenter_mv.row + r, fcenter_mv.col + c + i };
            unsigned int sad =
                fn_ptr->sdf(what->buf, what->stride,
                            get_buf_from_mv(in_what, &mv), in_what->stride);
            if (sad < best_sad) {
              sad += mvsad_err_cost(x, &mv, ref_mv, sad_per_bit);
              if (sad < best_sad) {
                best_sad = sad;
                *best_mv = mv;
              }
            }
          }
        }
      }
    }
  }

  return best_sad;
}

#define MIN_RANGE 7
#define MAX_RANGE 256
#define MIN_INTERVAL 1
#if CONFIG_NON_GREEDY_MV
static int64_t exhaustive_mesh_search_multi_step(
    MV *best_mv, const MV *center_mv, int range, int step,
    const struct buf_2d *src, const struct buf_2d *pre, int lambda,
    const int_mv *nb_full_mvs, int full_mv_num, const MvLimits *mv_limits,
    const vp9_variance_fn_ptr_t *fn_ptr) {
  int64_t best_sad;
  int r, c;
  int start_col, end_col, start_row, end_row;
  *best_mv = *center_mv;
  best_sad =
      ((int64_t)fn_ptr->sdf(src->buf, src->stride,
                            get_buf_from_mv(pre, center_mv), pre->stride)
       << LOG2_PRECISION) +
      lambda * vp9_nb_mvs_inconsistency(best_mv, nb_full_mvs, full_mv_num);
  start_row = VPXMAX(center_mv->row - range, mv_limits->row_min);
  start_col = VPXMAX(center_mv->col - range, mv_limits->col_min);
  end_row = VPXMIN(center_mv->row + range, mv_limits->row_max);
  end_col = VPXMIN(center_mv->col + range, mv_limits->col_max);
  for (r = start_row; r <= end_row; r += step) {
    for (c = start_col; c <= end_col; c += step) {
      const MV mv = { r, c };
      int64_t sad = (int64_t)fn_ptr->sdf(src->buf, src->stride,
                                         get_buf_from_mv(pre, &mv), pre->stride)
                    << LOG2_PRECISION;
      if (sad < best_sad) {
        sad += lambda * vp9_nb_mvs_inconsistency(&mv, nb_full_mvs, full_mv_num);
        if (sad < best_sad) {
          best_sad = sad;
          *best_mv = mv;
        }
      }
    }
  }
  return best_sad;
}

static int64_t exhaustive_mesh_search_single_step(
    MV *best_mv, const MV *center_mv, int range, const struct buf_2d *src,
    const struct buf_2d *pre, int lambda, const int_mv *nb_full_mvs,
    int full_mv_num, const MvLimits *mv_limits,
    const vp9_variance_fn_ptr_t *fn_ptr) {
  int64_t best_sad;
  int r, c, i;
  int start_col, end_col, start_row, end_row;

  *best_mv = *center_mv;
  best_sad =
      ((int64_t)fn_ptr->sdf(src->buf, src->stride,
                            get_buf_from_mv(pre, center_mv), pre->stride)
       << LOG2_PRECISION) +
      lambda * vp9_nb_mvs_inconsistency(best_mv, nb_full_mvs, full_mv_num);
  start_row = VPXMAX(center_mv->row - range, mv_limits->row_min);
  start_col = VPXMAX(center_mv->col - range, mv_limits->col_min);
  end_row = VPXMIN(center_mv->row + range, mv_limits->row_max);
  end_col = VPXMIN(center_mv->col + range, mv_limits->col_max);
  for (r = start_row; r <= end_row; r += 1) {
    c = start_col;
    while (c + 3 <= end_col) {
      unsigned int sads[4];
      const uint8_t *addrs[4];
      for (i = 0; i < 4; ++i) {
        const MV mv = { r, c + i };
        addrs[i] = get_buf_from_mv(pre, &mv);
      }
      fn_ptr->sdx4df(src->buf, src->stride, addrs, pre->stride, sads);

      for (i = 0; i < 4; ++i) {
        int64_t sad = (int64_t)sads[i] << LOG2_PRECISION;
        if (sad < best_sad) {
          const MV mv = { r, c + i };
          sad +=
              lambda * vp9_nb_mvs_inconsistency(&mv, nb_full_mvs, full_mv_num);
          if (sad < best_sad) {
            best_sad = sad;
            *best_mv = mv;
          }
        }
      }
      c += 4;
    }
    while (c <= end_col) {
      const MV mv = { r, c };
      int64_t sad = (int64_t)fn_ptr->sdf(src->buf, src->stride,
                                         get_buf_from_mv(pre, &mv), pre->stride)
                    << LOG2_PRECISION;
      if (sad < best_sad) {
        sad += lambda * vp9_nb_mvs_inconsistency(&mv, nb_full_mvs, full_mv_num);
        if (sad < best_sad) {
          best_sad = sad;
          *best_mv = mv;
        }
      }
      c += 1;
    }
  }
  return best_sad;
}

static int64_t exhaustive_mesh_search_new(const MACROBLOCK *x, MV *best_mv,
                                          int range, int step,
                                          const vp9_variance_fn_ptr_t *fn_ptr,
                                          const MV *center_mv, int lambda,
                                          const int_mv *nb_full_mvs,
                                          int full_mv_num) {
  const MACROBLOCKD *const xd = &x->e_mbd;
  const struct buf_2d *src = &x->plane[0].src;
  const struct buf_2d *pre = &xd->plane[0].pre[0];
  assert(step >= 1);
  assert(is_mv_in(&x->mv_limits, center_mv));
  if (step == 1) {
    return exhaustive_mesh_search_single_step(
        best_mv, center_mv, range, src, pre, lambda, nb_full_mvs, full_mv_num,
        &x->mv_limits, fn_ptr);
  }
  return exhaustive_mesh_search_multi_step(best_mv, center_mv, range, step, src,
                                           pre, lambda, nb_full_mvs,
                                           full_mv_num, &x->mv_limits, fn_ptr);
}

static int64_t full_pixel_exhaustive_new(const VP9_COMP *cpi, MACROBLOCK *x,
                                         MV *centre_mv_full,
                                         const vp9_variance_fn_ptr_t *fn_ptr,
                                         MV *dst_mv, int lambda,
                                         const int_mv *nb_full_mvs,
                                         int full_mv_num) {
  const SPEED_FEATURES *const sf = &cpi->sf;
  MV temp_mv = { centre_mv_full->row, centre_mv_full->col };
  int64_t bestsme;
  int i;
  int interval = sf->mesh_patterns[0].interval;
  int range = sf->mesh_patterns[0].range;
  int baseline_interval_divisor;

  // Trap illegal values for interval and range for this function.
  if ((range < MIN_RANGE) || (range > MAX_RANGE) || (interval < MIN_INTERVAL) ||
      (interval > range)) {
    printf("ERROR: invalid range\n");
    assert(0);
  }

  baseline_interval_divisor = range / interval;

  // Check size of proposed first range against magnitude of the centre
  // value used as a starting point.
  range = clamp(range, (5 * VPXMAX(abs(temp_mv.row), abs(temp_mv.col))) / 4,
                MAX_RANGE);
  interval = VPXMAX(interval, range / baseline_interval_divisor);

  // initial search
  bestsme =
      exhaustive_mesh_search_new(x, &temp_mv, range, interval, fn_ptr, &temp_mv,
                                 lambda, nb_full_mvs, full_mv_num);

  if ((interval > MIN_INTERVAL) && (range > MIN_RANGE)) {
    // Progressive searches with range and step size decreasing each time
    // till we reach a step size of 1. Then break out.
    for (i = 1; i < MAX_MESH_STEP; ++i) {
      // First pass with coarser step and longer range
      bestsme = exhaustive_mesh_search_new(
          x, &temp_mv, sf->mesh_patterns[i].range,
          sf->mesh_patterns[i].interval, fn_ptr, &temp_mv, lambda, nb_full_mvs,
          full_mv_num);

      if (sf->mesh_patterns[i].interval == 1) break;
    }
  }

  *dst_mv = temp_mv;

  return bestsme;
}

static int64_t diamond_search_sad_new(const MACROBLOCK *x,
                                      const search_site_config *cfg,
                                      const MV *init_full_mv, MV *best_full_mv,
                                      int search_param, int lambda, int *num00,
                                      const vp9_variance_fn_ptr_t *fn_ptr,
                                      const int_mv *nb_full_mvs,
                                      int full_mv_num) {
  int i, j, step;

  const MACROBLOCKD *const xd = &x->e_mbd;
  uint8_t *what = x->plane[0].src.buf;
  const int what_stride = x->plane[0].src.stride;
  const uint8_t *in_what;
  const int in_what_stride = xd->plane[0].pre[0].stride;
  const uint8_t *best_address;

  int64_t bestsad;
  int best_site = -1;
  int last_site = -1;

  // search_param determines the length of the initial step and hence the number
  // of iterations.
  // 0 = initial step (MAX_FIRST_STEP) pel
  // 1 = (MAX_FIRST_STEP/2) pel,
  // 2 = (MAX_FIRST_STEP/4) pel...
  //  const search_site *ss = &cfg->ss[search_param * cfg->searches_per_step];
  const MV *ss_mv = &cfg->ss_mv[search_param * cfg->searches_per_step];
  const intptr_t *ss_os = &cfg->ss_os[search_param * cfg->searches_per_step];
  const int tot_steps = cfg->total_steps - search_param;
  vpx_clear_system_state();

  *best_full_mv = *init_full_mv;
  clamp_mv(best_full_mv, x->mv_limits.col_min, x->mv_limits.col_max,
           x->mv_limits.row_min, x->mv_limits.row_max);
  *num00 = 0;

  // Work out the start point for the search
  in_what = xd->plane[0].pre[0].buf + best_full_mv->row * in_what_stride +
            best_full_mv->col;
  best_address = in_what;

  // Check the starting position
  {
    const int64_t mv_dist =
        (int64_t)fn_ptr->sdf(what, what_stride, in_what, in_what_stride)
        << LOG2_PRECISION;
    const int64_t mv_cost =
        vp9_nb_mvs_inconsistency(best_full_mv, nb_full_mvs, full_mv_num);
    bestsad = mv_dist + lambda * mv_cost;
  }

  i = 0;

  for (step = 0; step < tot_steps; step++) {
    int all_in = 1, t;

    // All_in is true if every one of the points we are checking are within
    // the bounds of the image.
    all_in &= ((best_full_mv->row + ss_mv[i].row) > x->mv_limits.row_min);
    all_in &= ((best_full_mv->row + ss_mv[i + 1].row) < x->mv_limits.row_max);
    all_in &= ((best_full_mv->col + ss_mv[i + 2].col) > x->mv_limits.col_min);
    all_in &= ((best_full_mv->col + ss_mv[i + 3].col) < x->mv_limits.col_max);

    // If all the pixels are within the bounds we don't check whether the
    // search point is valid in this loop,  otherwise we check each point
    // for validity..
    if (all_in) {
      unsigned int sad_array[4];

      for (j = 0; j < cfg->searches_per_step; j += 4) {
        unsigned char const *block_offset[4];

        for (t = 0; t < 4; t++) block_offset[t] = ss_os[i + t] + best_address;

        fn_ptr->sdx4df(what, what_stride, block_offset, in_what_stride,
                       sad_array);

        for (t = 0; t < 4; t++, i++) {
          const int64_t mv_dist = (int64_t)sad_array[t] << LOG2_PRECISION;
          if (mv_dist < bestsad) {
            const MV this_mv = { best_full_mv->row + ss_mv[i].row,
                                 best_full_mv->col + ss_mv[i].col };
            const int64_t mv_cost =
                vp9_nb_mvs_inconsistency(&this_mv, nb_full_mvs, full_mv_num);
            const int64_t thissad = mv_dist + lambda * mv_cost;
            if (thissad < bestsad) {
              bestsad = thissad;
              best_site = i;
            }
          }
        }
      }
    } else {
      for (j = 0; j < cfg->searches_per_step; j++) {
        // Trap illegal vectors
        const MV this_mv = { best_full_mv->row + ss_mv[i].row,
                             best_full_mv->col + ss_mv[i].col };

        if (is_mv_in(&x->mv_limits, &this_mv)) {
          const uint8_t *const check_here = ss_os[i] + best_address;
          const int64_t mv_dist =
              (int64_t)fn_ptr->sdf(what, what_stride, check_here,
                                   in_what_stride)
              << LOG2_PRECISION;
          if (mv_dist < bestsad) {
            const int64_t mv_cost =
                vp9_nb_mvs_inconsistency(&this_mv, nb_full_mvs, full_mv_num);
            const int64_t thissad = mv_dist + lambda * mv_cost;
            if (thissad < bestsad) {
              bestsad = thissad;
              best_site = i;
            }
          }
        }
        i++;
      }
    }
    if (best_site != last_site) {
      best_full_mv->row += ss_mv[best_site].row;
      best_full_mv->col += ss_mv[best_site].col;
      best_address += ss_os[best_site];
      last_site = best_site;
    } else if (best_address == in_what) {
      (*num00)++;
    }
  }
  return bestsad;
}

int vp9_prepare_nb_full_mvs(const MotionField *motion_field, int mi_row,
                            int mi_col, int_mv *nb_full_mvs) {
  const int mi_width = num_8x8_blocks_wide_lookup[motion_field->bsize];
  const int mi_height = num_8x8_blocks_high_lookup[motion_field->bsize];
  const int dirs[NB_MVS_NUM][2] = { { -1, 0 }, { 0, -1 }, { 1, 0 }, { 0, 1 } };
  int nb_full_mv_num = 0;
  int i;
  assert(mi_row % mi_height == 0);
  assert(mi_col % mi_width == 0);
  for (i = 0; i < NB_MVS_NUM; ++i) {
    int r = dirs[i][0];
    int c = dirs[i][1];
    int brow = mi_row / mi_height + r;
    int bcol = mi_col / mi_width + c;
    if (brow >= 0 && brow < motion_field->block_rows && bcol >= 0 &&
        bcol < motion_field->block_cols) {
      if (vp9_motion_field_is_mv_set(motion_field, brow, bcol)) {
        int_mv mv = vp9_motion_field_get_mv(motion_field, brow, bcol);
        nb_full_mvs[nb_full_mv_num].as_mv = get_full_mv(&mv.as_mv);
        ++nb_full_mv_num;
      }
    }
  }
  return nb_full_mv_num;
}
#endif  // CONFIG_NON_GREEDY_MV

int vp9_diamond_search_sad_c(const MACROBLOCK *x, const search_site_config *cfg,
                             MV *ref_mv, uint32_t start_mv_sad, MV *best_mv,
                             int search_param, int sad_per_bit, int *num00,
                             const vp9_sad_fn_ptr_t *sad_fn_ptr,
                             const MV *center_mv) {
  int i, j, step;

  const MACROBLOCKD *const xd = &x->e_mbd;
  uint8_t *what = x->plane[0].src.buf;
  const int what_stride = x->plane[0].src.stride;
  const uint8_t *in_what;
  const int in_what_stride = xd->plane[0].pre[0].stride;
  const uint8_t *best_address;

  unsigned int bestsad = start_mv_sad;
  int best_site = -1;
  int last_site = -1;

  int ref_row;
  int ref_col;

  // search_param determines the length of the initial step and hence the number
  // of iterations.
  // 0 = initial step (MAX_FIRST_STEP) pel
  // 1 = (MAX_FIRST_STEP/2) pel,
  // 2 = (MAX_FIRST_STEP/4) pel...
  //  const search_site *ss = &cfg->ss[search_param * cfg->searches_per_step];
  const MV *ss_mv = &cfg->ss_mv[search_param * cfg->searches_per_step];
  const intptr_t *ss_os = &cfg->ss_os[search_param * cfg->searches_per_step];
  const int tot_steps = cfg->total_steps - search_param;

  const MV fcenter_mv = { center_mv->row >> 3, center_mv->col >> 3 };
  ref_row = ref_mv->row;
  ref_col = ref_mv->col;
  *num00 = 0;
  best_mv->row = ref_row;
  best_mv->col = ref_col;

  // Work out the start point for the search
  in_what = xd->plane[0].pre[0].buf + ref_row * in_what_stride + ref_col;
  best_address = in_what;

  i = 0;

  for (step = 0; step < tot_steps; step++) {
    int all_in = 1, t;

    // All_in is true if every one of the points we are checking are within
    // the bounds of the image.
    all_in &= ((best_mv->row + ss_mv[i].row) > x->mv_limits.row_min);
    all_in &= ((best_mv->row + ss_mv[i + 1].row) < x->mv_limits.row_max);
    all_in &= ((best_mv->col + ss_mv[i + 2].col) > x->mv_limits.col_min);
    all_in &= ((best_mv->col + ss_mv[i + 3].col) < x->mv_limits.col_max);

    // If all the pixels are within the bounds we don't check whether the
    // search point is valid in this loop,  otherwise we check each point
    // for validity..
    if (all_in) {
      unsigned int sad_array[4];

      for (j = 0; j < cfg->searches_per_step; j += 4) {
        unsigned char const *block_offset[4];

        for (t = 0; t < 4; t++) block_offset[t] = ss_os[i + t] + best_address;

        sad_fn_ptr->sdx4df(what, what_stride, block_offset, in_what_stride,
                           sad_array);

        for (t = 0; t < 4; t++, i++) {
          if (sad_array[t] < bestsad) {
            const MV this_mv = { best_mv->row + ss_mv[i].row,
                                 best_mv->col + ss_mv[i].col };
            sad_array[t] +=
                mvsad_err_cost(x, &this_mv, &fcenter_mv, sad_per_bit);
            if (sad_array[t] < bestsad) {
              bestsad = sad_array[t];
              best_site = i;
            }
          }
        }
      }
    } else {
      for (j = 0; j < cfg->searches_per_step; j++) {
        // Trap illegal vectors
        const MV this_mv = { best_mv->row + ss_mv[i].row,
                             best_mv->col + ss_mv[i].col };

        if (is_mv_in(&x->mv_limits, &this_mv)) {
          const uint8_t *const check_here = ss_os[i] + best_address;
          unsigned int thissad =
              sad_fn_ptr->sdf(what, what_stride, check_here, in_what_stride);

          if (thissad < bestsad) {
            thissad += mvsad_err_cost(x, &this_mv, &fcenter_mv, sad_per_bit);
            if (thissad < bestsad) {
              bestsad = thissad;
              best_site = i;
            }
          }
        }
        i++;
      }
    }
    if (best_site != last_site) {
      best_mv->row += ss_mv[best_site].row;
      best_mv->col += ss_mv[best_site].col;
      best_address += ss_os[best_site];
      last_site = best_site;
#if defined(NEW_DIAMOND_SEARCH)
      while (1) {
        const MV this_mv = { best_mv->row + ss_mv[best_site].row,
                             best_mv->col + ss_mv[best_site].col };
        if (is_mv_in(&x->mv_limits, &this_mv)) {
          const uint8_t *const check_here = ss_os[best_site] + best_address;
          unsigned int thissad =
              fn_ptr->sdf(what, what_stride, check_here, in_what_stride);
          if (thissad < bestsad) {
            thissad += mvsad_err_cost(x, &this_mv, &fcenter_mv, sad_per_bit);
            if (thissad < bestsad) {
              bestsad = thissad;
              best_mv->row += ss_mv[best_site].row;
              best_mv->col += ss_mv[best_site].col;
              best_address += ss_os[best_site];
              continue;
            }
          }
        }
        break;
      }
#endif
    } else if (best_address == in_what) {
      (*num00)++;
    }
  }
  return bestsad;
}

static int vector_match(int16_t *ref, int16_t *src, int bwl) {
  int best_sad = INT_MAX;
  int this_sad;
  int d;
  int center, offset = 0;
  int bw = 4 << bwl;  // redundant variable, to be changed in the experiments.
  for (d = 0; d <= bw; d += 16) {
    this_sad = vpx_vector_var(&ref[d], src, bwl);
    if (this_sad < best_sad) {
      best_sad = this_sad;
      offset = d;
    }
  }
  center = offset;

  for (d = -8; d <= 8; d += 16) {
    int this_pos = offset + d;
    // check limit
    if (this_pos < 0 || this_pos > bw) continue;
    this_sad = vpx_vector_var(&ref[this_pos], src, bwl);
    if (this_sad < best_sad) {
      best_sad = this_sad;
      center = this_pos;
    }
  }
  offset = center;

  for (d = -4; d <= 4; d += 8) {
    int this_pos = offset + d;
    // check limit
    if (this_pos < 0 || this_pos > bw) continue;
    this_sad = vpx_vector_var(&ref[this_pos], src, bwl);
    if (this_sad < best_sad) {
      best_sad = this_sad;
      center = this_pos;
    }
  }
  offset = center;

  for (d = -2; d <= 2; d += 4) {
    int this_pos = offset + d;
    // check limit
    if (this_pos < 0 || this_pos > bw) continue;
    this_sad = vpx_vector_var(&ref[this_pos], src, bwl);
    if (this_sad < best_sad) {
      best_sad = this_sad;
      center = this_pos;
    }
  }
  offset = center;

  for (d = -1; d <= 1; d += 2) {
    int this_pos = offset + d;
    // check limit
    if (this_pos < 0 || this_pos > bw) continue;
    this_sad = vpx_vector_var(&ref[this_pos], src, bwl);
    if (this_sad < best_sad) {
      best_sad = this_sad;
      center = this_pos;
    }
  }

  return (center - (bw >> 1));
}

static const MV search_pos[4] = {
  { -1, 0 },
  { 0, -1 },
  { 0, 1 },
  { 1, 0 },
};

unsigned int vp9_int_pro_motion_estimation(const VP9_COMP *cpi, MACROBLOCK *x,
                                           BLOCK_SIZE bsize, int mi_row,
                                           int mi_col, const MV *ref_mv) {
  MACROBLOCKD *xd = &x->e_mbd;
  MODE_INFO *mi = xd->mi[0];
  struct buf_2d backup_yv12[MAX_MB_PLANE] = { { 0, 0 } };
  DECLARE_ALIGNED(16, int16_t, hbuf[128]);
  DECLARE_ALIGNED(16, int16_t, vbuf[128]);
  DECLARE_ALIGNED(16, int16_t, src_hbuf[64]);
  DECLARE_ALIGNED(16, int16_t, src_vbuf[64]);
  int idx;
  const int bw = 4 << b_width_log2_lookup[bsize];
  const int bh = 4 << b_height_log2_lookup[bsize];
  const int search_width = bw << 1;
  const int search_height = bh << 1;
  const int src_stride = x->plane[0].src.stride;
  const int ref_stride = xd->plane[0].pre[0].stride;
  uint8_t const *ref_buf, *src_buf;
  MV *tmp_mv = &xd->mi[0]->mv[0].as_mv;
  unsigned int best_sad, tmp_sad, this_sad[4];
  MV this_mv;
  const int norm_factor = 3 + (bw >> 5);
  const YV12_BUFFER_CONFIG *scaled_ref_frame =
      vp9_get_scaled_ref_frame(cpi, mi->ref_frame[0]);
  MvLimits subpel_mv_limits;

  if (scaled_ref_frame) {
    int i;
    // Swap out the reference frame for a version that's been scaled to
    // match the resolution of the current frame, allowing the existing
    // motion search code to be used without additional modifications.
    for (i = 0; i < MAX_MB_PLANE; i++) backup_yv12[i] = xd->plane[i].pre[0];
    vp9_setup_pre_planes(xd, 0, scaled_ref_frame, mi_row, mi_col, NULL);
  }

#if CONFIG_VP9_HIGHBITDEPTH
  // TODO(jingning): Implement integral projection functions for high bit-depth
  // setting and remove this part of code.
  if (xd->bd != 8) {
    const unsigned int sad = cpi->fn_ptr[bsize].sdf(
        x->plane[0].src.buf, src_stride, xd->plane[0].pre[0].buf, ref_stride);
    tmp_mv->row = 0;
    tmp_mv->col = 0;

    if (scaled_ref_frame) {
      int i;
      for (i = 0; i < MAX_MB_PLANE; i++) xd->plane[i].pre[0] = backup_yv12[i];
    }
    return sad;
  }
#endif

  // Set up prediction 1-D reference set
  ref_buf = xd->plane[0].pre[0].buf - (bw >> 1);
  for (idx = 0; idx < search_width; idx += 16) {
    vpx_int_pro_row(&hbuf[idx], ref_buf, ref_stride, bh);
    ref_buf += 16;
  }

  ref_buf = xd->plane[0].pre[0].buf - (bh >> 1) * ref_stride;
  for (idx = 0; idx < search_height; ++idx) {
    vbuf[idx] = vpx_int_pro_col(ref_buf, bw) >> norm_factor;
    ref_buf += ref_stride;
  }

  // Set up src 1-D reference set
  for (idx = 0; idx < bw; idx += 16) {
    src_buf = x->plane[0].src.buf + idx;
    vpx_int_pro_row(&src_hbuf[idx], src_buf, src_stride, bh);
  }

  src_buf = x->plane[0].src.buf;
  for (idx = 0; idx < bh; ++idx) {
    src_vbuf[idx] = vpx_int_pro_col(src_buf, bw) >> norm_factor;
    src_buf += src_stride;
  }

  // Find the best match per 1-D search
  tmp_mv->col = vector_match(hbuf, src_hbuf, b_width_log2_lookup[bsize]);
  tmp_mv->row = vector_match(vbuf, src_vbuf, b_height_log2_lookup[bsize]);

  this_mv = *tmp_mv;
  src_buf = x->plane[0].src.buf;
  ref_buf = xd->plane[0].pre[0].buf + this_mv.row * ref_stride + this_mv.col;
  best_sad = cpi->fn_ptr[bsize].sdf(src_buf, src_stride, ref_buf, ref_stride);

  {
    const uint8_t *const pos[4] = {
      ref_buf - ref_stride,
      ref_buf - 1,
      ref_buf + 1,
      ref_buf + ref_stride,
    };

    cpi->fn_ptr[bsize].sdx4df(src_buf, src_stride, pos, ref_stride, this_sad);
  }

  for (idx = 0; idx < 4; ++idx) {
    if (this_sad[idx] < best_sad) {
      best_sad = this_sad[idx];
      tmp_mv->row = search_pos[idx].row + this_mv.row;
      tmp_mv->col = search_pos[idx].col + this_mv.col;
    }
  }

  if (this_sad[0] < this_sad[3])
    this_mv.row -= 1;
  else
    this_mv.row += 1;

  if (this_sad[1] < this_sad[2])
    this_mv.col -= 1;
  else
    this_mv.col += 1;

  ref_buf = xd->plane[0].pre[0].buf + this_mv.row * ref_stride + this_mv.col;

  tmp_sad = cpi->fn_ptr[bsize].sdf(src_buf, src_stride, ref_buf, ref_stride);
  if (best_sad > tmp_sad) {
    *tmp_mv = this_mv;
    best_sad = tmp_sad;
  }

  tmp_mv->row *= 8;
  tmp_mv->col *= 8;

  vp9_set_subpel_mv_search_range(&subpel_mv_limits, &x->mv_limits, ref_mv);
  clamp_mv(tmp_mv, subpel_mv_limits.col_min, subpel_mv_limits.col_max,
           subpel_mv_limits.row_min, subpel_mv_limits.row_max);

  if (scaled_ref_frame) {
    int i;
    for (i = 0; i < MAX_MB_PLANE; i++) xd->plane[i].pre[0] = backup_yv12[i];
  }

  return best_sad;
}

static int get_exhaustive_threshold(int exhaustive_searches_thresh,
                                    BLOCK_SIZE bsize) {
  return exhaustive_searches_thresh >>
         (8 - (b_width_log2_lookup[bsize] + b_height_log2_lookup[bsize]));
}

#if CONFIG_NON_GREEDY_MV
// Runs sequence of diamond searches in smaller steps for RD.
/* do_refine: If last step (1-away) of n-step search doesn't pick the center
              point as the best match, we will do a final 1-away diamond
              refining search  */
int vp9_full_pixel_diamond_new(const VP9_COMP *cpi, MACROBLOCK *x,
                               BLOCK_SIZE bsize, MV *mvp_full, int step_param,
                               int lambda, int do_refine,
                               const int_mv *nb_full_mvs, int full_mv_num,
                               MV *best_mv) {
  const vp9_variance_fn_ptr_t *fn_ptr = &cpi->fn_ptr[bsize];
  const SPEED_FEATURES *const sf = &cpi->sf;
  int n, num00 = 0;
  int thissme;
  int bestsme;
  const int further_steps = MAX_MVSEARCH_STEPS - 1 - step_param;
  const MV center_mv = { 0, 0 };
  vpx_clear_system_state();
  diamond_search_sad_new(x, &cpi->ss_cfg, mvp_full, best_mv, step_param, lambda,
                         &n, fn_ptr, nb_full_mvs, full_mv_num);

  bestsme = vp9_get_mvpred_var(x, best_mv, &center_mv, fn_ptr, 0);

  // If there won't be more n-step search, check to see if refining search is
  // needed.
  if (n > further_steps) do_refine = 0;

  while (n < further_steps) {
    ++n;
    if (num00) {
      num00--;
    } else {
      MV temp_mv;
      diamond_search_sad_new(x, &cpi->ss_cfg, mvp_full, &temp_mv,
                             step_param + n, lambda, &num00, fn_ptr,
                             nb_full_mvs, full_mv_num);
      thissme = vp9_get_mvpred_var(x, &temp_mv, &center_mv, fn_ptr, 0);
      // check to see if refining search is needed.
      if (num00 > further_steps - n) do_refine = 0;

      if (thissme < bestsme) {
        bestsme = thissme;
        *best_mv = temp_mv;
      }
    }
  }

  // final 1-away diamond refining search
  if (do_refine) {
    const int search_range = 8;
    MV temp_mv = *best_mv;
    vp9_refining_search_sad_new(x, &temp_mv, lambda, search_range, fn_ptr,
                                nb_full_mvs, full_mv_num);
    thissme = vp9_get_mvpred_var(x, &temp_mv, &center_mv, fn_ptr, 0);
    if (thissme < bestsme) {
      bestsme = thissme;
      *best_mv = temp_mv;
    }
  }

  if (sf->exhaustive_searches_thresh < INT_MAX &&
      !cpi->rc.is_src_frame_alt_ref) {
    const int64_t exhaustive_thr =
        get_exhaustive_threshold(sf->exhaustive_searches_thresh, bsize);
    if (bestsme > exhaustive_thr) {
      full_pixel_exhaustive_new(cpi, x, best_mv, fn_ptr, best_mv, lambda,
                                nb_full_mvs, full_mv_num);
      bestsme = vp9_get_mvpred_var(x, best_mv, &center_mv, fn_ptr, 0);
    }
  }
  return bestsme;
}
#endif  // CONFIG_NON_GREEDY_MV

// Runs sequence of diamond searches in smaller steps for RD.
/* do_refine: If last step (1-away) of n-step search doesn't pick the center
              point as the best match, we will do a final 1-away diamond
              refining search  */
static int full_pixel_diamond(const VP9_COMP *const cpi,
                              const MACROBLOCK *const x, BLOCK_SIZE bsize,
                              MV *mvp_full, int step_param, int sadpb,
                              int further_steps, int do_refine,
                              int use_downsampled_sad, int *cost_list,
                              const vp9_variance_fn_ptr_t *fn_ptr,
                              const MV *ref_mv, MV *dst_mv) {
  MV temp_mv;
  int thissme, n, num00 = 0;
  int bestsme;
  const int src_buf_stride = x->plane[0].src.stride;
  const uint8_t *const src_buf = x->plane[0].src.buf;
  const MACROBLOCKD *const xd = &x->e_mbd;
  const int pred_buf_stride = xd->plane[0].pre[0].stride;
  uint8_t *pred_buf;
  vp9_sad_fn_ptr_t sad_fn_ptr;
  unsigned int start_mv_sad, start_mv_sad_even_rows, start_mv_sad_odd_rows;
  const MV ref_mv_full = { ref_mv->row >> 3, ref_mv->col >> 3 };
  clamp_mv(mvp_full, x->mv_limits.col_min, x->mv_limits.col_max,
           x->mv_limits.row_min, x->mv_limits.row_max);

  pred_buf =
      xd->plane[0].pre[0].buf + mvp_full->row * pred_buf_stride + mvp_full->col;
  start_mv_sad_even_rows =
      fn_ptr->sdsf(src_buf, src_buf_stride, pred_buf, pred_buf_stride);
  start_mv_sad_odd_rows =
      fn_ptr->sdsf(src_buf + src_buf_stride, src_buf_stride,
                   pred_buf + pred_buf_stride, pred_buf_stride);
  start_mv_sad = (start_mv_sad_even_rows + start_mv_sad_odd_rows) >> 1;
  start_mv_sad += mvsad_err_cost(x, mvp_full, &ref_mv_full, sadpb);

  sad_fn_ptr.sdf = fn_ptr->sdf;
  sad_fn_ptr.sdx4df = fn_ptr->sdx4df;
  if (use_downsampled_sad && num_4x4_blocks_high_lookup[bsize] >= 2) {
    // If the absolute difference between the pred-to-src SAD of even rows and
    // the pred-to-src SAD of odd rows is small, skip every other row in sad
    // computation.
    const int odd_to_even_diff_sad =
        abs((int)start_mv_sad_even_rows - (int)start_mv_sad_odd_rows);
    const int mult_thresh = 10;
    if (odd_to_even_diff_sad * mult_thresh < (int)start_mv_sad_even_rows) {
      sad_fn_ptr.sdf = fn_ptr->sdsf;
      sad_fn_ptr.sdx4df = fn_ptr->sdsx4df;
    }
  }

  bestsme =
      cpi->diamond_search_sad(x, &cpi->ss_cfg, mvp_full, start_mv_sad, &temp_mv,
                              step_param, sadpb, &n, &sad_fn_ptr, ref_mv);
  if (bestsme < INT_MAX)
    bestsme = vp9_get_mvpred_var(x, &temp_mv, ref_mv, fn_ptr, 1);
  *dst_mv = temp_mv;

  // If there won't be more n-step search, check to see if refining search is
  // needed.
  if (n > further_steps) do_refine = 0;

  while (n < further_steps) {
    ++n;

    if (num00) {
      num00--;
    } else {
      thissme = cpi->diamond_search_sad(x, &cpi->ss_cfg, mvp_full, start_mv_sad,
                                        &temp_mv, step_param + n, sadpb, &num00,
                                        &sad_fn_ptr, ref_mv);
      if (thissme < INT_MAX)
        thissme = vp9_get_mvpred_var(x, &temp_mv, ref_mv, fn_ptr, 1);

      // check to see if refining search is needed.
      if (num00 > further_steps - n) do_refine = 0;

      if (thissme < bestsme) {
        bestsme = thissme;
        *dst_mv = temp_mv;
      }
    }
  }

  // final 1-away diamond refining search
  if (do_refine) {
    const int search_range = 8;
    MV best_mv = *dst_mv;
    thissme = vp9_refining_search_sad(x, &best_mv, sadpb, search_range,
                                      &sad_fn_ptr, ref_mv);
    if (thissme < INT_MAX)
      thissme = vp9_get_mvpred_var(x, &best_mv, ref_mv, fn_ptr, 1);
    if (thissme < bestsme) {
      bestsme = thissme;
      *dst_mv = best_mv;
    }
  }

  if (sad_fn_ptr.sdf != fn_ptr->sdf) {
    // If we are skipping rows when we perform the motion search, we need to
    // check the quality of skipping. If it's bad, then we run search with
    // skip row features off.
    const uint8_t *best_address = get_buf_from_mv(&xd->plane[0].pre[0], dst_mv);
    const int sad =
        fn_ptr->sdf(src_buf, src_buf_stride, best_address, pred_buf_stride);
    const int skip_sad =
        fn_ptr->sdsf(src_buf, src_buf_stride, best_address, pred_buf_stride);
    // We will keep the result of skipping rows if it's good enough.
    const int kSADThresh =
        1 << (b_width_log2_lookup[bsize] + b_height_log2_lookup[bsize]);
    if (sad > kSADThresh && abs(skip_sad - sad) * 10 >= VPXMAX(sad, 1) * 9) {
      // There is a large discrepancy between skipping and not skipping, so we
      // need to redo the motion search.
      return full_pixel_diamond(cpi, x, bsize, mvp_full, step_param, sadpb,
                                further_steps, do_refine, 0, cost_list, fn_ptr,
                                ref_mv, dst_mv);
    }
  }

  // Return cost list.
  if (cost_list) {
    calc_int_cost_list(x, ref_mv, sadpb, fn_ptr, dst_mv, cost_list);
  }
  return bestsme;
}

// Runs an limited range exhaustive mesh search using a pattern set
// according to the encode speed profile.
static int full_pixel_exhaustive(const VP9_COMP *const cpi,
                                 const MACROBLOCK *const x, MV *centre_mv_full,
                                 int sadpb, int *cost_list,
                                 const vp9_variance_fn_ptr_t *fn_ptr,
                                 const MV *ref_mv, MV *dst_mv) {
  const SPEED_FEATURES *const sf = &cpi->sf;
  MV temp_mv = { centre_mv_full->row, centre_mv_full->col };
  MV f_ref_mv = { ref_mv->row >> 3, ref_mv->col >> 3 };
  int bestsme;
  int i;
  int interval = sf->mesh_patterns[0].interval;
  int range = sf->mesh_patterns[0].range;
  int baseline_interval_divisor;

  // Trap illegal values for interval and range for this function.
  if ((range < MIN_RANGE) || (range > MAX_RANGE) || (interval < MIN_INTERVAL) ||
      (interval > range))
    return INT_MAX;

  baseline_interval_divisor = range / interval;

  // Check size of proposed first range against magnitude of the centre
  // value used as a starting point.
  range = clamp(range, (5 * VPXMAX(abs(temp_mv.row), abs(temp_mv.col))) / 4,
                MAX_RANGE);
  interval = VPXMAX(interval, range / baseline_interval_divisor);

  // initial search
  bestsme = exhaustive_mesh_search(x, &f_ref_mv, &temp_mv, range, interval,
                                   sadpb, fn_ptr, &temp_mv);

  if ((interval > MIN_INTERVAL) && (range > MIN_RANGE)) {
    // Progressive searches with range and step size decreasing each time
    // till we reach a step size of 1. Then break out.
    for (i = 1; i < MAX_MESH_STEP; ++i) {
      // First pass with coarser step and longer range
      bestsme = exhaustive_mesh_search(
          x, &f_ref_mv, &temp_mv, sf->mesh_patterns[i].range,
          sf->mesh_patterns[i].interval, sadpb, fn_ptr, &temp_mv);

      if (sf->mesh_patterns[i].interval == 1) break;
    }
  }

  if (bestsme < INT_MAX)
    bestsme = vp9_get_mvpred_var(x, &temp_mv, ref_mv, fn_ptr, 1);
  *dst_mv = temp_mv;

  // Return cost list.
  if (cost_list) {
    calc_int_cost_list(x, ref_mv, sadpb, fn_ptr, dst_mv, cost_list);
  }
  return bestsme;
}

#if CONFIG_NON_GREEDY_MV
int64_t vp9_refining_search_sad_new(const MACROBLOCK *x, MV *best_full_mv,
                                    int lambda, int search_range,
                                    const vp9_variance_fn_ptr_t *fn_ptr,
                                    const int_mv *nb_full_mvs,
                                    int full_mv_num) {
  const MACROBLOCKD *const xd = &x->e_mbd;
  const MV neighbors[4] = { { -1, 0 }, { 0, -1 }, { 0, 1 }, { 1, 0 } };
  const struct buf_2d *const what = &x->plane[0].src;
  const struct buf_2d *const in_what = &xd->plane[0].pre[0];
  const uint8_t *best_address = get_buf_from_mv(in_what, best_full_mv);
  int64_t best_sad;
  int i, j;
  vpx_clear_system_state();
  {
    const int64_t mv_dist = (int64_t)fn_ptr->sdf(what->buf, what->stride,
                                                 best_address, in_what->stride)
                            << LOG2_PRECISION;
    const int64_t mv_cost =
        vp9_nb_mvs_inconsistency(best_full_mv, nb_full_mvs, full_mv_num);
    best_sad = mv_dist + lambda * mv_cost;
  }

  for (i = 0; i < search_range; i++) {
    int best_site = -1;
    const int all_in = ((best_full_mv->row - 1) > x->mv_limits.row_min) &
                       ((best_full_mv->row + 1) < x->mv_limits.row_max) &
                       ((best_full_mv->col - 1) > x->mv_limits.col_min) &
                       ((best_full_mv->col + 1) < x->mv_limits.col_max);

    if (all_in) {
      unsigned int sads[4];
      const uint8_t *const positions[4] = { best_address - in_what->stride,
                                            best_address - 1, best_address + 1,
                                            best_address + in_what->stride };

      fn_ptr->sdx4df(what->buf, what->stride, positions, in_what->stride, sads);

      for (j = 0; j < 4; ++j) {
        const MV mv = { best_full_mv->row + neighbors[j].row,
                        best_full_mv->col + neighbors[j].col };
        const int64_t mv_dist = (int64_t)sads[j] << LOG2_PRECISION;
        const int64_t mv_cost =
            vp9_nb_mvs_inconsistency(&mv, nb_full_mvs, full_mv_num);
        const int64_t thissad = mv_dist + lambda * mv_cost;
        if (thissad < best_sad) {
          best_sad = thissad;
          best_site = j;
        }
      }
    } else {
      for (j = 0; j < 4; ++j) {
        const MV mv = { best_full_mv->row + neighbors[j].row,
                        best_full_mv->col + neighbors[j].col };

        if (is_mv_in(&x->mv_limits, &mv)) {
          const int64_t mv_dist =
              (int64_t)fn_ptr->sdf(what->buf, what->stride,
                                   get_buf_from_mv(in_what, &mv),
                                   in_what->stride)
              << LOG2_PRECISION;
          const int64_t mv_cost =
              vp9_nb_mvs_inconsistency(&mv, nb_full_mvs, full_mv_num);
          const int64_t thissad = mv_dist + lambda * mv_cost;
          if (thissad < best_sad) {
            best_sad = thissad;
            best_site = j;
          }
        }
      }
    }

    if (best_site == -1) {
      break;
    } else {
      best_full_mv->row += neighbors[best_site].row;
      best_full_mv->col += neighbors[best_site].col;
      best_address = get_buf_from_mv(in_what, best_full_mv);
    }
  }

  return best_sad;
}
#endif  // CONFIG_NON_GREEDY_MV

int vp9_refining_search_sad(const MACROBLOCK *x, MV *ref_mv, int error_per_bit,
                            int search_range,
                            const vp9_sad_fn_ptr_t *sad_fn_ptr,
                            const MV *center_mv) {
  const MACROBLOCKD *const xd = &x->e_mbd;
  const MV neighbors[4] = { { -1, 0 }, { 0, -1 }, { 0, 1 }, { 1, 0 } };
  const struct buf_2d *const what = &x->plane[0].src;
  const struct buf_2d *const in_what = &xd->plane[0].pre[0];
  const MV fcenter_mv = { center_mv->row >> 3, center_mv->col >> 3 };
  const uint8_t *best_address = get_buf_from_mv(in_what, ref_mv);
  unsigned int best_sad =
      sad_fn_ptr->sdf(what->buf, what->stride, best_address, in_what->stride) +
      mvsad_err_cost(x, ref_mv, &fcenter_mv, error_per_bit);
  int i, j;

  for (i = 0; i < search_range; i++) {
    int best_site = -1;
    const int all_in = ((ref_mv->row - 1) > x->mv_limits.row_min) &
                       ((ref_mv->row + 1) < x->mv_limits.row_max) &
                       ((ref_mv->col - 1) > x->mv_limits.col_min) &
                       ((ref_mv->col + 1) < x->mv_limits.col_max);

    if (all_in) {
      unsigned int sads[4];
      const uint8_t *const positions[4] = { best_address - in_what->stride,
                                            best_address - 1, best_address + 1,
                                            best_address + in_what->stride };

      sad_fn_ptr->sdx4df(what->buf, what->stride, positions, in_what->stride,
                         sads);

      for (j = 0; j < 4; ++j) {
        if (sads[j] < best_sad) {
          const MV mv = { ref_mv->row + neighbors[j].row,
                          ref_mv->col + neighbors[j].col };
          sads[j] += mvsad_err_cost(x, &mv, &fcenter_mv, error_per_bit);
          if (sads[j] < best_sad) {
            best_sad = sads[j];
            best_site = j;
          }
        }
      }
    } else {
      for (j = 0; j < 4; ++j) {
        const MV mv = { ref_mv->row + neighbors[j].row,
                        ref_mv->col + neighbors[j].col };

        if (is_mv_in(&x->mv_limits, &mv)) {
          unsigned int sad =
              sad_fn_ptr->sdf(what->buf, what->stride,
                              get_buf_from_mv(in_what, &mv), in_what->stride);
          if (sad < best_sad) {
            sad += mvsad_err_cost(x, &mv, &fcenter_mv, error_per_bit);
            if (sad < best_sad) {
              best_sad = sad;
              best_site = j;
            }
          }
        }
      }
    }

    if (best_site == -1) {
      break;
    } else {
      ref_mv->row += neighbors[best_site].row;
      ref_mv->col += neighbors[best_site].col;
      best_address = get_buf_from_mv(in_what, ref_mv);
    }
  }

  return best_sad;
}

// This function is called when we do joint motion search in comp_inter_inter
// mode.
int vp9_refining_search_8p_c(const MACROBLOCK *x, MV *ref_mv, int error_per_bit,
                             int search_range,
                             const vp9_variance_fn_ptr_t *fn_ptr,
                             const MV *center_mv, const uint8_t *second_pred) {
  const MV neighbors[8] = { { -1, 0 },  { 0, -1 }, { 0, 1 },  { 1, 0 },
                            { -1, -1 }, { 1, -1 }, { -1, 1 }, { 1, 1 } };
  const MACROBLOCKD *const xd = &x->e_mbd;
  const struct buf_2d *const what = &x->plane[0].src;
  const struct buf_2d *const in_what = &xd->plane[0].pre[0];
  const MV fcenter_mv = { center_mv->row >> 3, center_mv->col >> 3 };
  unsigned int best_sad = INT_MAX;
  int i, j;
  clamp_mv(ref_mv, x->mv_limits.col_min, x->mv_limits.col_max,
           x->mv_limits.row_min, x->mv_limits.row_max);
  best_sad =
      fn_ptr->sdaf(what->buf, what->stride, get_buf_from_mv(in_what, ref_mv),
                   in_what->stride, second_pred) +
      mvsad_err_cost(x, ref_mv, &fcenter_mv, error_per_bit);

  for (i = 0; i < search_range; ++i) {
    int best_site = -1;

    for (j = 0; j < 8; ++j) {
      const MV mv = { ref_mv->row + neighbors[j].row,
                      ref_mv->col + neighbors[j].col };

      if (is_mv_in(&x->mv_limits, &mv)) {
        unsigned int sad =
            fn_ptr->sdaf(what->buf, what->stride, get_buf_from_mv(in_what, &mv),
                         in_what->stride, second_pred);
        if (sad < best_sad) {
          sad += mvsad_err_cost(x, &mv, &fcenter_mv, error_per_bit);
          if (sad < best_sad) {
            best_sad = sad;
            best_site = j;
          }
        }
      }
    }

    if (best_site == -1) {
      break;
    } else {
      ref_mv->row += neighbors[best_site].row;
      ref_mv->col += neighbors[best_site].col;
    }
  }
  return best_sad;
}

int vp9_full_pixel_search(const VP9_COMP *const cpi, const MACROBLOCK *const x,
                          BLOCK_SIZE bsize, MV *mvp_full, int step_param,
                          int search_method, int error_per_bit, int *cost_list,
                          const MV *ref_mv, MV *tmp_mv, int var_max, int rd) {
  const SPEED_FEATURES *const sf = &cpi->sf;
  const SEARCH_METHODS method = (SEARCH_METHODS)search_method;
  const vp9_variance_fn_ptr_t *fn_ptr = &cpi->fn_ptr[bsize];
  int var = 0;
  int run_exhaustive_search = 0;

  if (cost_list) {
    cost_list[0] = INT_MAX;
    cost_list[1] = INT_MAX;
    cost_list[2] = INT_MAX;
    cost_list[3] = INT_MAX;
    cost_list[4] = INT_MAX;
  }

  switch (method) {
    case FAST_DIAMOND:
      var = fast_dia_search(x, mvp_full, step_param, error_per_bit, 0,
                            cost_list, fn_ptr, 1, ref_mv, tmp_mv);
      break;
    case FAST_HEX:
      var = fast_hex_search(x, mvp_full, step_param, error_per_bit, 0,
                            cost_list, fn_ptr, 1, ref_mv, tmp_mv);
      break;
    case HEX:
      var = hex_search(x, mvp_full, step_param, error_per_bit, 1, cost_list,
                       fn_ptr, 1, ref_mv, tmp_mv);
      break;
    case SQUARE:
      var = square_search(x, mvp_full, step_param, error_per_bit, 1, cost_list,
                          fn_ptr, 1, ref_mv, tmp_mv);
      break;
    case BIGDIA:
      var = bigdia_search(x, mvp_full, step_param, error_per_bit, 1, cost_list,
                          fn_ptr, 1, ref_mv, tmp_mv);
      break;
    case NSTEP:
    case MESH:
      var = full_pixel_diamond(
          cpi, x, bsize, mvp_full, step_param, error_per_bit,
          MAX_MVSEARCH_STEPS - 1 - step_param, 1,
          cpi->sf.mv.use_downsampled_sad, cost_list, fn_ptr, ref_mv, tmp_mv);
      break;
    default: assert(0 && "Unknown search method");
  }

  if (method == NSTEP) {
    if (sf->exhaustive_searches_thresh < INT_MAX &&
        !cpi->rc.is_src_frame_alt_ref) {
      const int64_t exhaustive_thr =
          get_exhaustive_threshold(sf->exhaustive_searches_thresh, bsize);
      if (var > exhaustive_thr) {
        run_exhaustive_search = 1;
      }
    }
  } else if (method == MESH) {
    run_exhaustive_search = 1;
  }

  if (run_exhaustive_search) {
    int var_ex;
    MV tmp_mv_ex;
    var_ex = full_pixel_exhaustive(cpi, x, tmp_mv, error_per_bit, cost_list,
                                   fn_ptr, ref_mv, &tmp_mv_ex);
    if (var_ex < var) {
      var = var_ex;
      *tmp_mv = tmp_mv_ex;
    }
  }

  if (method != NSTEP && method != MESH && rd && var < var_max)
    var = vp9_get_mvpred_var(x, tmp_mv, ref_mv, fn_ptr, 1);

  return var;
}

// Note(yunqingwang): The following 2 functions are only used in the motion
// vector unit test, which return extreme motion vectors allowed by the MV
// limits.
#define COMMON_MV_TEST \
  SETUP_SUBPEL_SEARCH; \
                       \
  (void)error_per_bit; \
  (void)vfp;           \
  (void)z;             \
  (void)src_stride;    \
  (void)y;             \
  (void)y_stride;      \
  (void)second_pred;   \
  (void)w;             \
  (void)h;             \
  (void)offset;        \
  (void)mvjcost;       \
  (void)mvcost;        \
  (void)sse1;          \
  (void)distortion;    \
                       \
  (void)halfiters;     \
  (void)quarteriters;  \
  (void)eighthiters;   \
  (void)whichdir;      \
  (void)allow_hp;      \
  (void)forced_stop;   \
  (void)hstep;         \
  (void)rr;            \
  (void)rc;            \
                       \
  (void)tr;            \
  (void)tc;            \
  (void)sse;           \
  (void)thismse;       \
  (void)cost_list;     \
  (void)use_accurate_subpel_search

// Return the maximum MV.
uint32_t vp9_return_max_sub_pixel_mv(
    const MACROBLOCK *x, MV *bestmv, const MV *ref_mv, int allow_hp,
    int error_per_bit, const vp9_variance_fn_ptr_t *vfp, int forced_stop,
    int iters_per_step, int *cost_list, int *mvjcost, int *mvcost[2],
    uint32_t *distortion, uint32_t *sse1, const uint8_t *second_pred, int w,
    int h, int use_accurate_subpel_search) {
  COMMON_MV_TEST;

  (void)minr;
  (void)minc;

  bestmv->row = maxr;
  bestmv->col = maxc;
  besterr = 0;

  // In the sub-pel motion search, if hp is not used, then the last bit of mv
  // has to be 0.
  lower_mv_precision(bestmv, allow_hp && use_mv_hp(ref_mv));

  return besterr;
}
// Return the minimum MV.
uint32_t vp9_return_min_sub_pixel_mv(
    const MACROBLOCK *x, MV *bestmv, const MV *ref_mv, int allow_hp,
    int error_per_bit, const vp9_variance_fn_ptr_t *vfp, int forced_stop,
    int iters_per_step, int *cost_list, int *mvjcost, int *mvcost[2],
    uint32_t *distortion, uint32_t *sse1, const uint8_t *second_pred, int w,
    int h, int use_accurate_subpel_search) {
  COMMON_MV_TEST;

  (void)maxr;
  (void)maxc;

  bestmv->row = minr;
  bestmv->col = minc;
  besterr = 0;

  // In the sub-pel motion search, if hp is not used, then the last bit of mv
  // has to be 0.
  lower_mv_precision(bestmv, allow_hp && use_mv_hp(ref_mv));

  return besterr;
}
