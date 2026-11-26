/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vp8_rtcd.h"
#include "./vpx_dsp_rtcd.h"
#include "onyx_int.h"
#include "mcomp.h"
#include "vpx_mem/vpx_mem.h"
#include "vpx_config.h"
#include <stdio.h>
#include <limits.h>
#include <math.h>
#include "vp8/common/findnearmv.h"
#include "vp8/common/common.h"
#include "vpx_dsp/vpx_dsp_common.h"

int vp8_mv_bit_cost(int_mv *mv, int_mv *ref, int *mvcost[2], int Weight) {
  /* MV costing is based on the distribution of vectors in the previous
   * frame and as such will tend to over state the cost of vectors. In
   * addition coding a new vector can have a knock on effect on the cost
   * of subsequent vectors and the quality of prediction from NEAR and
   * NEAREST for subsequent blocks. The "Weight" parameter allows, to a
   * limited extent, for some account to be taken of these factors.
   */
  const int mv_idx_row =
      clamp((mv->as_mv.row - ref->as_mv.row) >> 1, 0, MVvals);
  const int mv_idx_col =
      clamp((mv->as_mv.col - ref->as_mv.col) >> 1, 0, MVvals);
  return ((mvcost[0][mv_idx_row] + mvcost[1][mv_idx_col]) * Weight) >> 7;
}

static int mv_err_cost(int_mv *mv, int_mv *ref, int *mvcost[2],
                       int error_per_bit) {
  /* Ignore mv costing if mvcost is NULL */
  if (mvcost) {
    const int mv_idx_row =
        clamp((mv->as_mv.row - ref->as_mv.row) >> 1, 0, MVvals);
    const int mv_idx_col =
        clamp((mv->as_mv.col - ref->as_mv.col) >> 1, 0, MVvals);
    return ((mvcost[0][mv_idx_row] + mvcost[1][mv_idx_col]) * error_per_bit +
            128) >>
           8;
  }
  return 0;
}

static int mvsad_err_cost(int_mv *mv, int_mv *ref, int *mvsadcost[2],
                          int error_per_bit) {
  /* Calculate sad error cost on full pixel basis. */
  /* Ignore mv costing if mvsadcost is NULL */
  if (mvsadcost) {
    return ((mvsadcost[0][(mv->as_mv.row - ref->as_mv.row)] +
             mvsadcost[1][(mv->as_mv.col - ref->as_mv.col)]) *
                error_per_bit +
            128) >>
           8;
  }
  return 0;
}

void vp8_init_dsmotion_compensation(MACROBLOCK *x, int stride) {
  int Len;
  int search_site_count = 0;

  /* Generate offsets for 4 search sites per step. */
  Len = MAX_FIRST_STEP;
  x->ss[search_site_count].mv.col = 0;
  x->ss[search_site_count].mv.row = 0;
  x->ss[search_site_count].offset = 0;
  search_site_count++;

  while (Len > 0) {
    /* Compute offsets for search sites. */
    x->ss[search_site_count].mv.col = 0;
    x->ss[search_site_count].mv.row = -Len;
    x->ss[search_site_count].offset = -Len * stride;
    search_site_count++;

    /* Compute offsets for search sites. */
    x->ss[search_site_count].mv.col = 0;
    x->ss[search_site_count].mv.row = Len;
    x->ss[search_site_count].offset = Len * stride;
    search_site_count++;

    /* Compute offsets for search sites. */
    x->ss[search_site_count].mv.col = -Len;
    x->ss[search_site_count].mv.row = 0;
    x->ss[search_site_count].offset = -Len;
    search_site_count++;

    /* Compute offsets for search sites. */
    x->ss[search_site_count].mv.col = Len;
    x->ss[search_site_count].mv.row = 0;
    x->ss[search_site_count].offset = Len;
    search_site_count++;

    /* Contract. */
    Len /= 2;
  }

  x->ss_count = search_site_count;
  x->searches_per_step = 4;
}

void vp8_init3smotion_compensation(MACROBLOCK *x, int stride) {
  int Len;
  int search_site_count = 0;

  /* Generate offsets for 8 search sites per step. */
  Len = MAX_FIRST_STEP;
  x->ss[search_site_count].mv.col = 0;
  x->ss[search_site_count].mv.row = 0;
  x->ss[search_site_count].offset = 0;
  search_site_count++;

  while (Len > 0) {
    /* Compute offsets for search sites. */
    x->ss[search_site_count].mv.col = 0;
    x->ss[search_site_count].mv.row = -Len;
    x->ss[search_site_count].offset = -Len * stride;
    search_site_count++;

    /* Compute offsets for search sites. */
    x->ss[search_site_count].mv.col = 0;
    x->ss[search_site_count].mv.row = Len;
    x->ss[search_site_count].offset = Len * stride;
    search_site_count++;

    /* Compute offsets for search sites. */
    x->ss[search_site_count].mv.col = -Len;
    x->ss[search_site_count].mv.row = 0;
    x->ss[search_site_count].offset = -Len;
    search_site_count++;

    /* Compute offsets for search sites. */
    x->ss[search_site_count].mv.col = Len;
    x->ss[search_site_count].mv.row = 0;
    x->ss[search_site_count].offset = Len;
    search_site_count++;

    /* Compute offsets for search sites. */
    x->ss[search_site_count].mv.col = -Len;
    x->ss[search_site_count].mv.row = -Len;
    x->ss[search_site_count].offset = -Len * stride - Len;
    search_site_count++;

    /* Compute offsets for search sites. */
    x->ss[search_site_count].mv.col = Len;
    x->ss[search_site_count].mv.row = -Len;
    x->ss[search_site_count].offset = -Len * stride + Len;
    search_site_count++;

    /* Compute offsets for search sites. */
    x->ss[search_site_count].mv.col = -Len;
    x->ss[search_site_count].mv.row = Len;
    x->ss[search_site_count].offset = Len * stride - Len;
    search_site_count++;

    /* Compute offsets for search sites. */
    x->ss[search_site_count].mv.col = Len;
    x->ss[search_site_count].mv.row = Len;
    x->ss[search_site_count].offset = Len * stride + Len;
    search_site_count++;

    /* Contract. */
    Len /= 2;
  }

  x->ss_count = search_site_count;
  x->searches_per_step = 8;
}

/*
 * To avoid the penalty for crossing cache-line read, preload the reference
 * area in a small buffer, which is aligned to make sure there won't be crossing
 * cache-line read while reading from this buffer. This reduced the cpu
 * cycles spent on reading ref data in sub-pixel filter functions.
 * TODO: Currently, since sub-pixel search range here is -3 ~ 3, copy 22 rows x
 * 32 cols area that is enough for 16x16 macroblock. Later, for SPLITMV, we
 * could reduce the area.
 */

/* estimated cost of a motion vector (r,c) */
#define MVC(r, c)                                                          \
  (mvcost ? ((mvcost[0][(r) - rr] + mvcost[1][(c) - rc]) * error_per_bit + \
             128) >>                                                       \
                8                                                          \
          : 0)
/* pointer to predictor base of a motionvector */
#define PRE(r, c) (y + (((r) >> 2) * y_stride + ((c) >> 2) - (offset)))
/* convert motion vector component to offset for svf calc */
#define SP(x) (((x) & 3) << 1)
/* returns subpixel variance error function. */
#define DIST(r, c) \
  vfp->svf(PRE(r, c), y_stride, SP(c), SP(r), z, b->src_stride, &sse)
#define IFMVCV(r, c, s, e) \
  if (c >= minc && c <= maxc && r >= minr && r <= maxr) s else e;
/* returns distortion + motion vector cost */
#define ERR(r, c) (MVC(r, c) + DIST(r, c))
/* checks if (r,c) has better score than previous best */
#define CHECK_BETTER(v, r, c)                          \
  do {                                                 \
    IFMVCV(                                            \
        r, c,                                          \
        {                                              \
          thismse = DIST(r, c);                        \
          if ((v = (MVC(r, c) + thismse)) < besterr) { \
            besterr = v;                               \
            br = r;                                    \
            bc = c;                                    \
            *distortion = thismse;                     \
            *sse1 = sse;                               \
          }                                            \
        },                                             \
        v = UINT_MAX;)                                 \
  } while (0)

int vp8_find_best_sub_pixel_step_iteratively(MACROBLOCK *x, BLOCK *b, BLOCKD *d,
                                             int_mv *bestmv, int_mv *ref_mv,
                                             int error_per_bit,
                                             const vp8_variance_fn_ptr_t *vfp,
                                             int *mvcost[2], int *distortion,
                                             unsigned int *sse1) {
  unsigned char *z = (*(b->base_src) + b->src);

  int rr = ref_mv->as_mv.row >> 1, rc = ref_mv->as_mv.col >> 1;
  int br = bestmv->as_mv.row * 4, bc = bestmv->as_mv.col * 4;
  int tr = br, tc = bc;
  unsigned int besterr;
  unsigned int left, right, up, down, diag;
  unsigned int sse;
  unsigned int whichdir;
  unsigned int halfiters = 4;
  unsigned int quarteriters = 4;
  int thismse;

  int minc = VPXMAX(x->mv_col_min * 4,
                    (ref_mv->as_mv.col >> 1) - ((1 << mvlong_width) - 1));
  int maxc = VPXMIN(x->mv_col_max * 4,
                    (ref_mv->as_mv.col >> 1) + ((1 << mvlong_width) - 1));
  int minr = VPXMAX(x->mv_row_min * 4,
                    (ref_mv->as_mv.row >> 1) - ((1 << mvlong_width) - 1));
  int maxr = VPXMIN(x->mv_row_max * 4,
                    (ref_mv->as_mv.row >> 1) + ((1 << mvlong_width) - 1));

  int y_stride;
  int offset;
  int pre_stride = x->e_mbd.pre.y_stride;
  unsigned char *base_pre = x->e_mbd.pre.y_buffer;

#if VPX_ARCH_X86 || VPX_ARCH_X86_64
  MACROBLOCKD *xd = &x->e_mbd;
  unsigned char *y_0 = base_pre + d->offset + (bestmv->as_mv.row) * pre_stride +
                       bestmv->as_mv.col;
  unsigned char *y;
  int buf_r1, buf_r2, buf_c1;

  /* Clamping to avoid out-of-range data access */
  buf_r1 = ((bestmv->as_mv.row - 3) < x->mv_row_min)
               ? (bestmv->as_mv.row - x->mv_row_min)
               : 3;
  buf_r2 = ((bestmv->as_mv.row + 3) > x->mv_row_max)
               ? (x->mv_row_max - bestmv->as_mv.row)
               : 3;
  buf_c1 = ((bestmv->as_mv.col - 3) < x->mv_col_min)
               ? (bestmv->as_mv.col - x->mv_col_min)
               : 3;
  y_stride = 32;

  /* Copy to intermediate buffer before searching. */
  vfp->copymem(y_0 - buf_c1 - pre_stride * buf_r1, pre_stride, xd->y_buf,
               y_stride, 16 + buf_r1 + buf_r2);
  y = xd->y_buf + y_stride * buf_r1 + buf_c1;
#else
  unsigned char *y = base_pre + d->offset + (bestmv->as_mv.row) * pre_stride +
                     bestmv->as_mv.col;
  y_stride = pre_stride;
#endif

  offset = (bestmv->as_mv.row) * y_stride + bestmv->as_mv.col;

  /* central mv */
  bestmv->as_mv.row = clamp(bestmv->as_mv.row * 8, SHRT_MIN, SHRT_MAX);
  bestmv->as_mv.col = clamp(bestmv->as_mv.col * 8, SHRT_MIN, SHRT_MAX);

  /* calculate central point error */
  besterr = vfp->vf(y, y_stride, z, b->src_stride, sse1);
  *distortion = besterr;
  besterr += mv_err_cost(bestmv, ref_mv, mvcost, error_per_bit);

  /* TODO: Each subsequent iteration checks at least one point in common
   * with the last iteration could be 2 ( if diag selected)
   */
  while (--halfiters) {
    /* 1/2 pel */
    CHECK_BETTER(left, tr, tc - 2);
    CHECK_BETTER(right, tr, tc + 2);
    CHECK_BETTER(up, tr - 2, tc);
    CHECK_BETTER(down, tr + 2, tc);

    whichdir = (left < right ? 0 : 1) + (up < down ? 0 : 2);

    switch (whichdir) {
      case 0: CHECK_BETTER(diag, tr - 2, tc - 2); break;
      case 1: CHECK_BETTER(diag, tr - 2, tc + 2); break;
      case 2: CHECK_BETTER(diag, tr + 2, tc - 2); break;
      case 3: CHECK_BETTER(diag, tr + 2, tc + 2); break;
    }

    /* no reason to check the same one again. */
    if (tr == br && tc == bc) break;

    tr = br;
    tc = bc;
  }

  /* TODO: Each subsequent iteration checks at least one point in common
   * with the last iteration could be 2 ( if diag selected)
   */

  /* 1/4 pel */
  while (--quarteriters) {
    CHECK_BETTER(left, tr, tc - 1);
    CHECK_BETTER(right, tr, tc + 1);
    CHECK_BETTER(up, tr - 1, tc);
    CHECK_BETTER(down, tr + 1, tc);

    whichdir = (left < right ? 0 : 1) + (up < down ? 0 : 2);

    switch (whichdir) {
      case 0: CHECK_BETTER(diag, tr - 1, tc - 1); break;
      case 1: CHECK_BETTER(diag, tr - 1, tc + 1); break;
      case 2: CHECK_BETTER(diag, tr + 1, tc - 1); break;
      case 3: CHECK_BETTER(diag, tr + 1, tc + 1); break;
    }

    /* no reason to check the same one again. */
    if (tr == br && tc == bc) break;

    tr = br;
    tc = bc;
  }

  bestmv->as_mv.row = clamp(br * 2, SHRT_MIN, SHRT_MAX);
  bestmv->as_mv.col = clamp(bc * 2, SHRT_MIN, SHRT_MAX);

  if ((abs(bestmv->as_mv.col - ref_mv->as_mv.col) > (MAX_FULL_PEL_VAL << 3)) ||
      (abs(bestmv->as_mv.row - ref_mv->as_mv.row) > (MAX_FULL_PEL_VAL << 3))) {
    return INT_MAX;
  }

  return besterr;
}
#undef MVC
#undef PRE
#undef SP
#undef DIST
#undef IFMVCV
#undef ERR
#undef CHECK_BETTER

int vp8_find_best_sub_pixel_step(MACROBLOCK *x, BLOCK *b, BLOCKD *d,
                                 int_mv *bestmv, int_mv *ref_mv,
                                 int error_per_bit,
                                 const vp8_variance_fn_ptr_t *vfp,
                                 int *mvcost[2], int *distortion,
                                 unsigned int *sse1) {
  int bestmse = INT_MAX;
  int_mv startmv;
  int_mv this_mv;
  unsigned char *z = (*(b->base_src) + b->src);
  int left, right, up, down, diag;
  unsigned int sse;
  int whichdir;
  int thismse;
  int y_stride;
  int pre_stride = x->e_mbd.pre.y_stride;
  unsigned char *base_pre = x->e_mbd.pre.y_buffer;

#if VPX_ARCH_X86 || VPX_ARCH_X86_64
  MACROBLOCKD *xd = &x->e_mbd;
  unsigned char *y_0 = base_pre + d->offset + (bestmv->as_mv.row) * pre_stride +
                       bestmv->as_mv.col;
  unsigned char *y;

  y_stride = 32;
  /* Copy 18 rows x 32 cols area to intermediate buffer before searching. */
  vfp->copymem(y_0 - 1 - pre_stride, pre_stride, xd->y_buf, y_stride, 18);
  y = xd->y_buf + y_stride + 1;
#else
  unsigned char *y = base_pre + d->offset + (bestmv->as_mv.row) * pre_stride +
                     bestmv->as_mv.col;
  y_stride = pre_stride;
#endif

  /* central mv */
  bestmv->as_mv.row = clamp(bestmv->as_mv.row * 8, SHRT_MIN, SHRT_MAX);
  bestmv->as_mv.col = clamp(bestmv->as_mv.col * 8, SHRT_MIN, SHRT_MAX);
  startmv = *bestmv;

  /* calculate central point error */
  bestmse = vfp->vf(y, y_stride, z, b->src_stride, sse1);
  *distortion = bestmse;
  bestmse += mv_err_cost(bestmv, ref_mv, mvcost, error_per_bit);

  /* go left then right and check error */
  this_mv.as_mv.row = startmv.as_mv.row;
  this_mv.as_mv.col = ((startmv.as_mv.col - 8) | 4);
  /* "halfpix" horizontal variance */
  thismse = vfp->svf(y - 1, y_stride, 4, 0, z, b->src_stride, &sse);
  left = thismse + mv_err_cost(&this_mv, ref_mv, mvcost, error_per_bit);

  if (left < bestmse) {
    *bestmv = this_mv;
    bestmse = left;
    *distortion = thismse;
    *sse1 = sse;
  }

  this_mv.as_mv.col += 8;
  /* "halfpix" horizontal variance */
  thismse = vfp->svf(y, y_stride, 4, 0, z, b->src_stride, &sse);
  right = thismse + mv_err_cost(&this_mv, ref_mv, mvcost, error_per_bit);

  if (right < bestmse) {
    *bestmv = this_mv;
    bestmse = right;
    *distortion = thismse;
    *sse1 = sse;
  }

  /* go up then down and check error */
  this_mv.as_mv.col = startmv.as_mv.col;
  this_mv.as_mv.row = ((startmv.as_mv.row - 8) | 4);
  /* "halfpix" vertical variance */
  thismse = vfp->svf(y - y_stride, y_stride, 0, 4, z, b->src_stride, &sse);
  up = thismse + mv_err_cost(&this_mv, ref_mv, mvcost, error_per_bit);

  if (up < bestmse) {
    *bestmv = this_mv;
    bestmse = up;
    *distortion = thismse;
    *sse1 = sse;
  }

  this_mv.as_mv.row += 8;
  /* "halfpix" vertical variance */
  thismse = vfp->svf(y, y_stride, 0, 4, z, b->src_stride, &sse);
  down = thismse + mv_err_cost(&this_mv, ref_mv, mvcost, error_per_bit);

  if (down < bestmse) {
    *bestmv = this_mv;
    bestmse = down;
    *distortion = thismse;
    *sse1 = sse;
  }

  /* now check 1 more diagonal */
  whichdir = (left < right ? 0 : 1) + (up < down ? 0 : 2);
  this_mv = startmv;

  switch (whichdir) {
    case 0:
      this_mv.as_mv.col = (this_mv.as_mv.col - 8) | 4;
      this_mv.as_mv.row = (this_mv.as_mv.row - 8) | 4;
      /* "halfpix" horizontal/vertical variance */
      thismse =
          vfp->svf(y - 1 - y_stride, y_stride, 4, 4, z, b->src_stride, &sse);
      break;
    case 1:
      this_mv.as_mv.col += 4;
      this_mv.as_mv.row = (this_mv.as_mv.row - 8) | 4;
      /* "halfpix" horizontal/vertical variance */
      thismse = vfp->svf(y - y_stride, y_stride, 4, 4, z, b->src_stride, &sse);
      break;
    case 2:
      this_mv.as_mv.col = (this_mv.as_mv.col - 8) | 4;
      this_mv.as_mv.row += 4;
      /* "halfpix" horizontal/vertical variance */
      thismse = vfp->svf(y - 1, y_stride, 4, 4, z, b->src_stride, &sse);
      break;
    case 3:
    default:
      this_mv.as_mv.col += 4;
      this_mv.as_mv.row += 4;
      /* "halfpix" horizontal/vertical variance */
      thismse = vfp->svf(y, y_stride, 4, 4, z, b->src_stride, &sse);
      break;
  }

  diag = thismse + mv_err_cost(&this_mv, ref_mv, mvcost, error_per_bit);

  if (diag < bestmse) {
    *bestmv = this_mv;
    bestmse = diag;
    *distortion = thismse;
    *sse1 = sse;
  }

  /* time to check quarter pels. */
  if (bestmv->as_mv.row < startmv.as_mv.row) y -= y_stride;

  if (bestmv->as_mv.col < startmv.as_mv.col) y--;

  startmv = *bestmv;

  /* go left then right and check error */
  this_mv.as_mv.row = startmv.as_mv.row;

  if (startmv.as_mv.col & 7) {
    this_mv.as_mv.col = startmv.as_mv.col - 2;
    thismse = vfp->svf(y, y_stride, this_mv.as_mv.col & 7,
                       this_mv.as_mv.row & 7, z, b->src_stride, &sse);
  } else {
    this_mv.as_mv.col = (startmv.as_mv.col - 8) | 6;
    thismse = vfp->svf(y - 1, y_stride, 6, this_mv.as_mv.row & 7, z,
                       b->src_stride, &sse);
  }

  left = thismse + mv_err_cost(&this_mv, ref_mv, mvcost, error_per_bit);

  if (left < bestmse) {
    *bestmv = this_mv;
    bestmse = left;
    *distortion = thismse;
    *sse1 = sse;
  }

  this_mv.as_mv.col += 4;
  thismse = vfp->svf(y, y_stride, this_mv.as_mv.col & 7, this_mv.as_mv.row & 7,
                     z, b->src_stride, &sse);
  right = thismse + mv_err_cost(&this_mv, ref_mv, mvcost, error_per_bit);

  if (right < bestmse) {
    *bestmv = this_mv;
    bestmse = right;
    *distortion = thismse;
    *sse1 = sse;
  }

  /* go up then down and check error */
  this_mv.as_mv.col = startmv.as_mv.col;

  if (startmv.as_mv.row & 7) {
    this_mv.as_mv.row = startmv.as_mv.row - 2;
    thismse = vfp->svf(y, y_stride, this_mv.as_mv.col & 7,
                       this_mv.as_mv.row & 7, z, b->src_stride, &sse);
  } else {
    this_mv.as_mv.row = (startmv.as_mv.row - 8) | 6;
    thismse = vfp->svf(y - y_stride, y_stride, this_mv.as_mv.col & 7, 6, z,
                       b->src_stride, &sse);
  }

  up = thismse + mv_err_cost(&this_mv, ref_mv, mvcost, error_per_bit);

  if (up < bestmse) {
    *bestmv = this_mv;
    bestmse = up;
    *distortion = thismse;
    *sse1 = sse;
  }

  this_mv.as_mv.row += 4;
  thismse = vfp->svf(y, y_stride, this_mv.as_mv.col & 7, this_mv.as_mv.row & 7,
                     z, b->src_stride, &sse);
  down = thismse + mv_err_cost(&this_mv, ref_mv, mvcost, error_per_bit);

  if (down < bestmse) {
    *bestmv = this_mv;
    bestmse = down;
    *distortion = thismse;
    *sse1 = sse;
  }

  /* now check 1 more diagonal */
  whichdir = (left < right ? 0 : 1) + (up < down ? 0 : 2);

  this_mv = startmv;

  switch (whichdir) {
    case 0:

      if (startmv.as_mv.row & 7) {
        this_mv.as_mv.row -= 2;

        if (startmv.as_mv.col & 7) {
          this_mv.as_mv.col -= 2;
          thismse = vfp->svf(y, y_stride, this_mv.as_mv.col & 7,
                             this_mv.as_mv.row & 7, z, b->src_stride, &sse);
        } else {
          this_mv.as_mv.col = (startmv.as_mv.col - 8) | 6;
          thismse = vfp->svf(y - 1, y_stride, 6, this_mv.as_mv.row & 7, z,
                             b->src_stride, &sse);
        }
      } else {
        this_mv.as_mv.row = (startmv.as_mv.row - 8) | 6;

        if (startmv.as_mv.col & 7) {
          this_mv.as_mv.col -= 2;
          thismse = vfp->svf(y - y_stride, y_stride, this_mv.as_mv.col & 7, 6,
                             z, b->src_stride, &sse);
        } else {
          this_mv.as_mv.col = (startmv.as_mv.col - 8) | 6;
          thismse = vfp->svf(y - y_stride - 1, y_stride, 6, 6, z, b->src_stride,
                             &sse);
        }
      }

      break;
    case 1:
      this_mv.as_mv.col += 2;

      if (startmv.as_mv.row & 7) {
        this_mv.as_mv.row -= 2;
        thismse = vfp->svf(y, y_stride, this_mv.as_mv.col & 7,
                           this_mv.as_mv.row & 7, z, b->src_stride, &sse);
      } else {
        this_mv.as_mv.row = (startmv.as_mv.row - 8) | 6;
        thismse = vfp->svf(y - y_stride, y_stride, this_mv.as_mv.col & 7, 6, z,
                           b->src_stride, &sse);
      }

      break;
    case 2:
      this_mv.as_mv.row += 2;

      if (startmv.as_mv.col & 7) {
        this_mv.as_mv.col -= 2;
        thismse = vfp->svf(y, y_stride, this_mv.as_mv.col & 7,
                           this_mv.as_mv.row & 7, z, b->src_stride, &sse);
      } else {
        this_mv.as_mv.col = (startmv.as_mv.col - 8) | 6;
        thismse = vfp->svf(y - 1, y_stride, 6, this_mv.as_mv.row & 7, z,
                           b->src_stride, &sse);
      }

      break;
    case 3:
      this_mv.as_mv.col += 2;
      this_mv.as_mv.row += 2;
      thismse = vfp->svf(y, y_stride, this_mv.as_mv.col & 7,
                         this_mv.as_mv.row & 7, z, b->src_stride, &sse);
      break;
  }

  diag = thismse + mv_err_cost(&this_mv, ref_mv, mvcost, error_per_bit);

  if (diag < bestmse) {
    *bestmv = this_mv;
    bestmse = diag;
    *distortion = thismse;
    *sse1 = sse;
  }

  return bestmse;
}

int vp8_find_best_half_pixel_step(MACROBLOCK *x, BLOCK *b, BLOCKD *d,
                                  int_mv *bestmv, int_mv *ref_mv,
                                  int error_per_bit,
                                  const vp8_variance_fn_ptr_t *vfp,
                                  int *mvcost[2], int *distortion,
                                  unsigned int *sse1) {
  int bestmse = INT_MAX;
  int_mv startmv;
  int_mv this_mv;
  unsigned char *z = (*(b->base_src) + b->src);
  int left, right, up, down, diag;
  unsigned int sse;
  int whichdir;
  int thismse;
  int y_stride;
  int pre_stride = x->e_mbd.pre.y_stride;
  unsigned char *base_pre = x->e_mbd.pre.y_buffer;

#if VPX_ARCH_X86 || VPX_ARCH_X86_64
  MACROBLOCKD *xd = &x->e_mbd;
  unsigned char *y_0 = base_pre + d->offset + (bestmv->as_mv.row) * pre_stride +
                       bestmv->as_mv.col;
  unsigned char *y;

  y_stride = 32;
  /* Copy 18 rows x 32 cols area to intermediate buffer before searching. */
  vfp->copymem(y_0 - 1 - pre_stride, pre_stride, xd->y_buf, y_stride, 18);
  y = xd->y_buf + y_stride + 1;
#else
  unsigned char *y = base_pre + d->offset + (bestmv->as_mv.row) * pre_stride +
                     bestmv->as_mv.col;
  y_stride = pre_stride;
#endif

  /* central mv */
  bestmv->as_mv.row = clamp(bestmv->as_mv.row * 8, SHRT_MIN, SHRT_MAX);
  bestmv->as_mv.col = clamp(bestmv->as_mv.col * 8, SHRT_MIN, SHRT_MAX);
  startmv = *bestmv;

  /* calculate central point error */
  bestmse = vfp->vf(y, y_stride, z, b->src_stride, sse1);
  *distortion = bestmse;
  bestmse += mv_err_cost(bestmv, ref_mv, mvcost, error_per_bit);

  /* go left then right and check error */
  this_mv.as_mv.row = startmv.as_mv.row;
  this_mv.as_mv.col = ((startmv.as_mv.col - 8) | 4);
  /* "halfpix" horizontal variance */
  thismse = vfp->svf(y - 1, y_stride, 4, 0, z, b->src_stride, &sse);
  left = thismse + mv_err_cost(&this_mv, ref_mv, mvcost, error_per_bit);

  if (left < bestmse) {
    *bestmv = this_mv;
    bestmse = left;
    *distortion = thismse;
    *sse1 = sse;
  }

  this_mv.as_mv.col += 8;
  /* "halfpix" horizontal variance */
  thismse = vfp->svf(y, y_stride, 4, 0, z, b->src_stride, &sse);
  right = thismse + mv_err_cost(&this_mv, ref_mv, mvcost, error_per_bit);

  if (right < bestmse) {
    *bestmv = this_mv;
    bestmse = right;
    *distortion = thismse;
    *sse1 = sse;
  }

  /* go up then down and check error */
  this_mv.as_mv.col = startmv.as_mv.col;
  this_mv.as_mv.row = ((startmv.as_mv.row - 8) | 4);
  /* "halfpix" vertical variance */
  thismse = vfp->svf(y - y_stride, y_stride, 0, 4, z, b->src_stride, &sse);
  up = thismse + mv_err_cost(&this_mv, ref_mv, mvcost, error_per_bit);

  if (up < bestmse) {
    *bestmv = this_mv;
    bestmse = up;
    *distortion = thismse;
    *sse1 = sse;
  }

  this_mv.as_mv.row += 8;
  /* "halfpix" vertical variance */
  thismse = vfp->svf(y, y_stride, 0, 4, z, b->src_stride, &sse);
  down = thismse + mv_err_cost(&this_mv, ref_mv, mvcost, error_per_bit);

  if (down < bestmse) {
    *bestmv = this_mv;
    bestmse = down;
    *distortion = thismse;
    *sse1 = sse;
  }

  /* now check 1 more diagonal - */
  whichdir = (left < right ? 0 : 1) + (up < down ? 0 : 2);
  this_mv = startmv;

  switch (whichdir) {
    case 0:
      this_mv.as_mv.col = (this_mv.as_mv.col - 8) | 4;
      this_mv.as_mv.row = (this_mv.as_mv.row - 8) | 4;
      /* "halfpix" horizontal/vertical variance */
      thismse =
          vfp->svf(y - 1 - y_stride, y_stride, 4, 4, z, b->src_stride, &sse);
      break;
    case 1:
      this_mv.as_mv.col += 4;
      this_mv.as_mv.row = (this_mv.as_mv.row - 8) | 4;
      /* "halfpix" horizontal/vertical variance */
      thismse = vfp->svf(y - y_stride, y_stride, 4, 4, z, b->src_stride, &sse);
      break;
    case 2:
      this_mv.as_mv.col = (this_mv.as_mv.col - 8) | 4;
      this_mv.as_mv.row += 4;
      /* "halfpix" horizontal/vertical variance */
      thismse = vfp->svf(y - 1, y_stride, 4, 4, z, b->src_stride, &sse);
      break;
    case 3:
    default:
      this_mv.as_mv.col += 4;
      this_mv.as_mv.row += 4;
      /* "halfpix" horizontal/vertical variance */
      thismse = vfp->svf(y, y_stride, 4, 4, z, b->src_stride, &sse);
      break;
  }

  diag = thismse + mv_err_cost(&this_mv, ref_mv, mvcost, error_per_bit);

  if (diag < bestmse) {
    *bestmv = this_mv;
    bestmse = diag;
    *distortion = thismse;
    *sse1 = sse;
  }

  return bestmse;
}

#define CHECK_BOUNDS(range)                    \
  do {                                         \
    all_in = 1;                                \
    all_in &= ((br - range) >= x->mv_row_min); \
    all_in &= ((br + range) <= x->mv_row_max); \
    all_in &= ((bc - range) >= x->mv_col_min); \
    all_in &= ((bc + range) <= x->mv_col_max); \
  } while (0)

#define CHECK_POINT                                  \
  {                                                  \
    if (this_mv.as_mv.col < x->mv_col_min) continue; \
    if (this_mv.as_mv.col > x->mv_col_max) continue; \
    if (this_mv.as_mv.row < x->mv_row_min) continue; \
    if (this_mv.as_mv.row > x->mv_row_max) continue; \
  }

#define CHECK_BETTER                                                     \
  do {                                                                   \
    if (thissad < bestsad) {                                             \
      thissad +=                                                         \
          mvsad_err_cost(&this_mv, &fcenter_mv, mvsadcost, sad_per_bit); \
      if (thissad < bestsad) {                                           \
        bestsad = thissad;                                               \
        best_site = i;                                                   \
      }                                                                  \
    }                                                                    \
  } while (0)

static const MV next_chkpts[6][3] = {
  { { -2, 0 }, { -1, -2 }, { 1, -2 } }, { { -1, -2 }, { 1, -2 }, { 2, 0 } },
  { { 1, -2 }, { 2, 0 }, { 1, 2 } },    { { 2, 0 }, { 1, 2 }, { -1, 2 } },
  { { 1, 2 }, { -1, 2 }, { -2, 0 } },   { { -1, 2 }, { -2, 0 }, { -1, -2 } }
};

int vp8_hex_search(MACROBLOCK *x, BLOCK *b, BLOCKD *d, int_mv *ref_mv,
                   int_mv *best_mv, int search_param, int sad_per_bit,
                   const vp8_variance_fn_ptr_t *vfp, int *mvsadcost[2],
                   int_mv *center_mv) {
  MV hex[6] = {
    { -1, -2 }, { 1, -2 }, { 2, 0 }, { 1, 2 }, { -1, 2 }, { -2, 0 }
  };
  MV neighbors[4] = { { 0, -1 }, { -1, 0 }, { 1, 0 }, { 0, 1 } };
  int i, j;

  unsigned char *what = (*(b->base_src) + b->src);
  int what_stride = b->src_stride;
  int pre_stride = x->e_mbd.pre.y_stride;
  unsigned char *base_pre = x->e_mbd.pre.y_buffer;

  int in_what_stride = pre_stride;
  int br, bc;
  int_mv this_mv;
  unsigned int bestsad;
  unsigned int thissad;
  unsigned char *base_offset;
  unsigned char *this_offset;
  int k = -1;
  int all_in;
  int best_site = -1;
  int hex_range = 127;
  int dia_range = 8;

  int_mv fcenter_mv;
  fcenter_mv.as_mv.row = center_mv->as_mv.row >> 3;
  fcenter_mv.as_mv.col = center_mv->as_mv.col >> 3;

  /* adjust ref_mv to make sure it is within MV range */
  vp8_clamp_mv(ref_mv, x->mv_col_min, x->mv_col_max, x->mv_row_min,
               x->mv_row_max);
  br = ref_mv->as_mv.row;
  bc = ref_mv->as_mv.col;

  /* Work out the start point for the search */
  base_offset = (unsigned char *)(base_pre + d->offset);
  this_offset = base_offset + (br * (pre_stride)) + bc;
  this_mv.as_mv.row = br;
  this_mv.as_mv.col = bc;
  bestsad = vfp->sdf(what, what_stride, this_offset, in_what_stride) +
            mvsad_err_cost(&this_mv, &fcenter_mv, mvsadcost, sad_per_bit);

#if CONFIG_MULTI_RES_ENCODING
  /* Lower search range based on prediction info */
  if (search_param >= 6)
    goto cal_neighbors;
  else if (search_param >= 5)
    hex_range = 4;
  else if (search_param >= 4)
    hex_range = 6;
  else if (search_param >= 3)
    hex_range = 15;
  else if (search_param >= 2)
    hex_range = 31;
  else if (search_param >= 1)
    hex_range = 63;

  dia_range = 8;
#else
  (void)search_param;
#endif

  /* hex search */
  CHECK_BOUNDS(2);

  if (all_in) {
    for (i = 0; i < 6; ++i) {
      this_mv.as_mv.row = br + hex[i].row;
      this_mv.as_mv.col = bc + hex[i].col;
      this_offset = base_offset + (this_mv.as_mv.row * in_what_stride) +
                    this_mv.as_mv.col;
      thissad = vfp->sdf(what, what_stride, this_offset, in_what_stride);
      CHECK_BETTER;
    }
  } else {
    for (i = 0; i < 6; ++i) {
      this_mv.as_mv.row = br + hex[i].row;
      this_mv.as_mv.col = bc + hex[i].col;
      CHECK_POINT
      this_offset = base_offset + (this_mv.as_mv.row * in_what_stride) +
                    this_mv.as_mv.col;
      thissad = vfp->sdf(what, what_stride, this_offset, in_what_stride);
      CHECK_BETTER;
    }
  }

  if (best_site == -1) {
    goto cal_neighbors;
  } else {
    br += hex[best_site].row;
    bc += hex[best_site].col;
    k = best_site;
  }

  for (j = 1; j < hex_range; ++j) {
    best_site = -1;
    CHECK_BOUNDS(2);

    if (all_in) {
      for (i = 0; i < 3; ++i) {
        this_mv.as_mv.row = br + next_chkpts[k][i].row;
        this_mv.as_mv.col = bc + next_chkpts[k][i].col;
        this_offset = base_offset + (this_mv.as_mv.row * (in_what_stride)) +
                      this_mv.as_mv.col;
        thissad = vfp->sdf(what, what_stride, this_offset, in_what_stride);
        CHECK_BETTER;
      }
    } else {
      for (i = 0; i < 3; ++i) {
        this_mv.as_mv.row = br + next_chkpts[k][i].row;
        this_mv.as_mv.col = bc + next_chkpts[k][i].col;
        CHECK_POINT
        this_offset = base_offset + (this_mv.as_mv.row * (in_what_stride)) +
                      this_mv.as_mv.col;
        thissad = vfp->sdf(what, what_stride, this_offset, in_what_stride);
        CHECK_BETTER;
      }
    }

    if (best_site == -1) {
      break;
    } else {
      br += next_chkpts[k][best_site].row;
      bc += next_chkpts[k][best_site].col;
      k += 5 + best_site;
      if (k >= 12) {
        k -= 12;
      } else if (k >= 6) {
        k -= 6;
      }
    }
  }

/* check 4 1-away neighbors */
cal_neighbors:
  for (j = 0; j < dia_range; ++j) {
    best_site = -1;
    CHECK_BOUNDS(1);

    if (all_in) {
      for (i = 0; i < 4; ++i) {
        this_mv.as_mv.row = br + neighbors[i].row;
        this_mv.as_mv.col = bc + neighbors[i].col;
        this_offset = base_offset + (this_mv.as_mv.row * (in_what_stride)) +
                      this_mv.as_mv.col;
        thissad = vfp->sdf(what, what_stride, this_offset, in_what_stride);
        CHECK_BETTER;
      }
    } else {
      for (i = 0; i < 4; ++i) {
        this_mv.as_mv.row = br + neighbors[i].row;
        this_mv.as_mv.col = bc + neighbors[i].col;
        CHECK_POINT
        this_offset = base_offset + (this_mv.as_mv.row * (in_what_stride)) +
                      this_mv.as_mv.col;
        thissad = vfp->sdf(what, what_stride, this_offset, in_what_stride);
        CHECK_BETTER;
      }
    }

    if (best_site == -1) {
      break;
    } else {
      br += neighbors[best_site].row;
      bc += neighbors[best_site].col;
    }
  }

  best_mv->as_mv.row = br;
  best_mv->as_mv.col = bc;

  return bestsad;
}
#undef CHECK_BOUNDS
#undef CHECK_POINT
#undef CHECK_BETTER

int vp8_diamond_search_sad_c(MACROBLOCK *x, BLOCK *b, BLOCKD *d, int_mv *ref_mv,
                             int_mv *best_mv, int search_param, int sad_per_bit,
                             int *num00, vp8_variance_fn_ptr_t *fn_ptr,
                             int *mvcost[2], int_mv *center_mv) {
  int i, j, step;

  unsigned char *what = (*(b->base_src) + b->src);
  int what_stride = b->src_stride;
  unsigned char *in_what;
  int pre_stride = x->e_mbd.pre.y_stride;
  unsigned char *base_pre = x->e_mbd.pre.y_buffer;
  int in_what_stride = pre_stride;
  unsigned char *best_address;

  int tot_steps;
  int_mv this_mv;

  unsigned int bestsad;
  unsigned int thissad;
  int best_site = 0;
  int last_site = 0;

  int ref_row;
  int ref_col;
  int this_row_offset;
  int this_col_offset;
  search_site *ss;

  unsigned char *check_here;

  int *mvsadcost[2];
  int_mv fcenter_mv;

  mvsadcost[0] = x->mvsadcost[0];
  mvsadcost[1] = x->mvsadcost[1];
  fcenter_mv.as_mv.row = center_mv->as_mv.row >> 3;
  fcenter_mv.as_mv.col = center_mv->as_mv.col >> 3;

  vp8_clamp_mv(ref_mv, x->mv_col_min, x->mv_col_max, x->mv_row_min,
               x->mv_row_max);
  ref_row = ref_mv->as_mv.row;
  ref_col = ref_mv->as_mv.col;
  *num00 = 0;
  best_mv->as_mv.row = ref_row;
  best_mv->as_mv.col = ref_col;

  /* Work out the start point for the search */
  in_what = (unsigned char *)(base_pre + d->offset + (ref_row * pre_stride) +
                              ref_col);
  best_address = in_what;

  /* Check the starting position */
  bestsad = fn_ptr->sdf(what, what_stride, in_what, in_what_stride) +
            mvsad_err_cost(best_mv, &fcenter_mv, mvsadcost, sad_per_bit);

  /* search_param determines the length of the initial step and hence
   * the number of iterations 0 = initial step (MAX_FIRST_STEP) pel :
   * 1 = (MAX_FIRST_STEP/2) pel, 2 = (MAX_FIRST_STEP/4) pel... etc.
   */
  ss = &x->ss[search_param * x->searches_per_step];
  tot_steps = (x->ss_count / x->searches_per_step) - search_param;

  i = 1;

  for (step = 0; step < tot_steps; ++step) {
    for (j = 0; j < x->searches_per_step; ++j) {
      /* Trap illegal vectors */
      this_row_offset = best_mv->as_mv.row + ss[i].mv.row;
      this_col_offset = best_mv->as_mv.col + ss[i].mv.col;

      if ((this_col_offset > x->mv_col_min) &&
          (this_col_offset < x->mv_col_max) &&
          (this_row_offset > x->mv_row_min) &&
          (this_row_offset < x->mv_row_max))

      {
        check_here = ss[i].offset + best_address;
        thissad = fn_ptr->sdf(what, what_stride, check_here, in_what_stride);

        if (thissad < bestsad) {
          this_mv.as_mv.row = this_row_offset;
          this_mv.as_mv.col = this_col_offset;
          thissad +=
              mvsad_err_cost(&this_mv, &fcenter_mv, mvsadcost, sad_per_bit);

          if (thissad < bestsad) {
            bestsad = thissad;
            best_site = i;
          }
        }
      }

      i++;
    }

    if (best_site != last_site) {
      best_mv->as_mv.row += ss[best_site].mv.row;
      best_mv->as_mv.col += ss[best_site].mv.col;
      best_address += ss[best_site].offset;
      last_site = best_site;
    } else if (best_address == in_what) {
      (*num00)++;
    }
  }

  this_mv.as_mv.row = clamp(best_mv->as_mv.row * 8, SHRT_MIN, SHRT_MAX);
  this_mv.as_mv.col = clamp(best_mv->as_mv.col * 8, SHRT_MIN, SHRT_MAX);

  return fn_ptr->vf(what, what_stride, best_address, in_what_stride, &thissad) +
         mv_err_cost(&this_mv, center_mv, mvcost, x->errorperbit);
}

#if HAVE_SSE2 || HAVE_MSA || HAVE_LSX
int vp8_diamond_search_sadx4(MACROBLOCK *x, BLOCK *b, BLOCKD *d, int_mv *ref_mv,
                             int_mv *best_mv, int search_param, int sad_per_bit,
                             int *num00, vp8_variance_fn_ptr_t *fn_ptr,
                             int *mvcost[2], int_mv *center_mv) {
  int i, j, step;

  unsigned char *what = (*(b->base_src) + b->src);
  int what_stride = b->src_stride;
  unsigned char *in_what;
  int pre_stride = x->e_mbd.pre.y_stride;
  unsigned char *base_pre = x->e_mbd.pre.y_buffer;
  int in_what_stride = pre_stride;
  unsigned char *best_address;

  int tot_steps;
  int_mv this_mv;

  unsigned int bestsad;
  unsigned int thissad;
  int best_site = 0;
  int last_site = 0;

  int ref_row;
  int ref_col;
  int this_row_offset;
  int this_col_offset;
  search_site *ss;

  unsigned char *check_here;

  int *mvsadcost[2];
  int_mv fcenter_mv;

  mvsadcost[0] = x->mvsadcost[0];
  mvsadcost[1] = x->mvsadcost[1];
  fcenter_mv.as_mv.row = center_mv->as_mv.row >> 3;
  fcenter_mv.as_mv.col = center_mv->as_mv.col >> 3;

  vp8_clamp_mv(ref_mv, x->mv_col_min, x->mv_col_max, x->mv_row_min,
               x->mv_row_max);
  ref_row = ref_mv->as_mv.row;
  ref_col = ref_mv->as_mv.col;
  *num00 = 0;
  best_mv->as_mv.row = ref_row;
  best_mv->as_mv.col = ref_col;

  /* Work out the start point for the search */
  in_what = (unsigned char *)(base_pre + d->offset + (ref_row * pre_stride) +
                              ref_col);
  best_address = in_what;

  /* Check the starting position */
  bestsad = fn_ptr->sdf(what, what_stride, in_what, in_what_stride) +
            mvsad_err_cost(best_mv, &fcenter_mv, mvsadcost, sad_per_bit);

  /* search_param determines the length of the initial step and hence the
   * number of iterations 0 = initial step (MAX_FIRST_STEP) pel : 1 =
   * (MAX_FIRST_STEP/2) pel, 2 = (MAX_FIRST_STEP/4) pel... etc.
   */
  ss = &x->ss[search_param * x->searches_per_step];
  tot_steps = (x->ss_count / x->searches_per_step) - search_param;

  i = 1;

  for (step = 0; step < tot_steps; ++step) {
    int all_in = 1, t;

    /* To know if all neighbor points are within the bounds, 4 bounds
     * checking are enough instead of checking 4 bounds for each
     * points.
     */
    all_in &= ((best_mv->as_mv.row + ss[i].mv.row) > x->mv_row_min);
    all_in &= ((best_mv->as_mv.row + ss[i + 1].mv.row) < x->mv_row_max);
    all_in &= ((best_mv->as_mv.col + ss[i + 2].mv.col) > x->mv_col_min);
    all_in &= ((best_mv->as_mv.col + ss[i + 3].mv.col) < x->mv_col_max);

    if (all_in) {
      unsigned int sad_array[4];

      for (j = 0; j < x->searches_per_step; j += 4) {
        const unsigned char *block_offset[4];

        for (t = 0; t < 4; ++t) {
          block_offset[t] = ss[i + t].offset + best_address;
        }

        fn_ptr->sdx4df(what, what_stride, block_offset, in_what_stride,
                       sad_array);

        for (t = 0; t < 4; t++, i++) {
          if (sad_array[t] < bestsad) {
            this_mv.as_mv.row = best_mv->as_mv.row + ss[i].mv.row;
            this_mv.as_mv.col = best_mv->as_mv.col + ss[i].mv.col;
            sad_array[t] +=
                mvsad_err_cost(&this_mv, &fcenter_mv, mvsadcost, sad_per_bit);

            if (sad_array[t] < bestsad) {
              bestsad = sad_array[t];
              best_site = i;
            }
          }
        }
      }
    } else {
      for (j = 0; j < x->searches_per_step; ++j) {
        /* Trap illegal vectors */
        this_row_offset = best_mv->as_mv.row + ss[i].mv.row;
        this_col_offset = best_mv->as_mv.col + ss[i].mv.col;

        if ((this_col_offset > x->mv_col_min) &&
            (this_col_offset < x->mv_col_max) &&
            (this_row_offset > x->mv_row_min) &&
            (this_row_offset < x->mv_row_max)) {
          check_here = ss[i].offset + best_address;
          thissad = fn_ptr->sdf(what, what_stride, check_here, in_what_stride);

          if (thissad < bestsad) {
            this_mv.as_mv.row = this_row_offset;
            this_mv.as_mv.col = this_col_offset;
            thissad +=
                mvsad_err_cost(&this_mv, &fcenter_mv, mvsadcost, sad_per_bit);

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
      best_mv->as_mv.row += ss[best_site].mv.row;
      best_mv->as_mv.col += ss[best_site].mv.col;
      best_address += ss[best_site].offset;
      last_site = best_site;
    } else if (best_address == in_what) {
      (*num00)++;
    }
  }

  this_mv.as_mv.row = clamp(best_mv->as_mv.row * 8, SHRT_MIN, SHRT_MAX);
  this_mv.as_mv.col = clamp(best_mv->as_mv.col * 8, SHRT_MIN, SHRT_MAX);

  return fn_ptr->vf(what, what_stride, best_address, in_what_stride, &thissad) +
         mv_err_cost(&this_mv, center_mv, mvcost, x->errorperbit);
}
#endif  // HAVE_SSE2 || HAVE_MSA || HAVE_LSX

int vp8_full_search_sad(MACROBLOCK *x, BLOCK *b, BLOCKD *d, int_mv *ref_mv,
                        int sad_per_bit, int distance,
                        vp8_variance_fn_ptr_t *fn_ptr, int *mvcost[2],
                        int_mv *center_mv) {
  unsigned char *what = (*(b->base_src) + b->src);
  int what_stride = b->src_stride;
  unsigned char *in_what;
  int pre_stride = x->e_mbd.pre.y_stride;
  unsigned char *base_pre = x->e_mbd.pre.y_buffer;
  int in_what_stride = pre_stride;
  int mv_stride = pre_stride;
  unsigned char *bestaddress;
  int_mv *best_mv = &d->bmi.mv;
  int_mv this_mv;
  unsigned int bestsad;
  unsigned int thissad;
  int r, c;

  unsigned char *check_here;

  int ref_row = ref_mv->as_mv.row;
  int ref_col = ref_mv->as_mv.col;

  int row_min = ref_row - distance;
  int row_max = ref_row + distance;
  int col_min = ref_col - distance;
  int col_max = ref_col + distance;

  int *mvsadcost[2];
  int_mv fcenter_mv;

  mvsadcost[0] = x->mvsadcost[0];
  mvsadcost[1] = x->mvsadcost[1];
  fcenter_mv.as_mv.row = center_mv->as_mv.row >> 3;
  fcenter_mv.as_mv.col = center_mv->as_mv.col >> 3;

  /* Work out the mid point for the search */
  in_what = base_pre + d->offset;
  bestaddress = in_what + (ref_row * pre_stride) + ref_col;

  best_mv->as_mv.row = ref_row;
  best_mv->as_mv.col = ref_col;

  /* Baseline value at the centre */
  bestsad = fn_ptr->sdf(what, what_stride, bestaddress, in_what_stride) +
            mvsad_err_cost(best_mv, &fcenter_mv, mvsadcost, sad_per_bit);

  /* Apply further limits to prevent us looking using vectors that stretch
   * beyond the UMV border
   */
  if (col_min < x->mv_col_min) col_min = x->mv_col_min;

  if (col_max > x->mv_col_max) col_max = x->mv_col_max;

  if (row_min < x->mv_row_min) row_min = x->mv_row_min;

  if (row_max > x->mv_row_max) row_max = x->mv_row_max;

  for (r = row_min; r < row_max; ++r) {
    this_mv.as_mv.row = r;
    check_here = r * mv_stride + in_what + col_min;

    for (c = col_min; c < col_max; ++c) {
      thissad = fn_ptr->sdf(what, what_stride, check_here, in_what_stride);

      if (thissad < bestsad) {
        this_mv.as_mv.col = c;
        thissad +=
            mvsad_err_cost(&this_mv, &fcenter_mv, mvsadcost, sad_per_bit);

        if (thissad < bestsad) {
          bestsad = thissad;
          best_mv->as_mv.row = r;
          best_mv->as_mv.col = c;
          bestaddress = check_here;
        }
      }

      check_here++;
    }
  }

  this_mv.as_mv.row = clamp(best_mv->as_mv.row * 8, SHRT_MIN, SHRT_MAX);
  this_mv.as_mv.col = clamp(best_mv->as_mv.col * 8, SHRT_MIN, SHRT_MAX);

  return fn_ptr->vf(what, what_stride, bestaddress, in_what_stride, &thissad) +
         mv_err_cost(&this_mv, center_mv, mvcost, x->errorperbit);
}

int vp8_refining_search_sad_c(MACROBLOCK *x, BLOCK *b, BLOCKD *d,
                              int_mv *ref_mv, int error_per_bit,
                              int search_range, vp8_variance_fn_ptr_t *fn_ptr,
                              int *mvcost[2], int_mv *center_mv) {
  MV neighbors[4] = { { -1, 0 }, { 0, -1 }, { 0, 1 }, { 1, 0 } };
  int i, j;
  short this_row_offset, this_col_offset;

  int what_stride = b->src_stride;
  int pre_stride = x->e_mbd.pre.y_stride;
  unsigned char *base_pre = x->e_mbd.pre.y_buffer;
  int in_what_stride = pre_stride;
  unsigned char *what = (*(b->base_src) + b->src);
  unsigned char *best_address =
      (unsigned char *)(base_pre + d->offset +
                        (ref_mv->as_mv.row * pre_stride) + ref_mv->as_mv.col);
  unsigned char *check_here;
  int_mv this_mv;
  unsigned int bestsad;
  unsigned int thissad;

  int *mvsadcost[2];
  int_mv fcenter_mv;

  mvsadcost[0] = x->mvsadcost[0];
  mvsadcost[1] = x->mvsadcost[1];
  fcenter_mv.as_mv.row = center_mv->as_mv.row >> 3;
  fcenter_mv.as_mv.col = center_mv->as_mv.col >> 3;

  bestsad = fn_ptr->sdf(what, what_stride, best_address, in_what_stride) +
            mvsad_err_cost(ref_mv, &fcenter_mv, mvsadcost, error_per_bit);

  for (i = 0; i < search_range; ++i) {
    int best_site = -1;

    for (j = 0; j < 4; ++j) {
      this_row_offset = ref_mv->as_mv.row + neighbors[j].row;
      this_col_offset = ref_mv->as_mv.col + neighbors[j].col;

      if ((this_col_offset > x->mv_col_min) &&
          (this_col_offset < x->mv_col_max) &&
          (this_row_offset > x->mv_row_min) &&
          (this_row_offset < x->mv_row_max)) {
        check_here = (neighbors[j].row) * in_what_stride + neighbors[j].col +
                     best_address;
        thissad = fn_ptr->sdf(what, what_stride, check_here, in_what_stride);

        if (thissad < bestsad) {
          this_mv.as_mv.row = this_row_offset;
          this_mv.as_mv.col = this_col_offset;
          thissad +=
              mvsad_err_cost(&this_mv, &fcenter_mv, mvsadcost, error_per_bit);

          if (thissad < bestsad) {
            bestsad = thissad;
            best_site = j;
          }
        }
      }
    }

    if (best_site == -1) {
      break;
    } else {
      ref_mv->as_mv.row += neighbors[best_site].row;
      ref_mv->as_mv.col += neighbors[best_site].col;
      best_address += (neighbors[best_site].row) * in_what_stride +
                      neighbors[best_site].col;
    }
  }

  this_mv.as_mv.row = clamp(ref_mv->as_mv.row * 8, SHRT_MIN, SHRT_MAX);
  this_mv.as_mv.col = clamp(ref_mv->as_mv.col * 8, SHRT_MIN, SHRT_MAX);

  return fn_ptr->vf(what, what_stride, best_address, in_what_stride, &thissad) +
         mv_err_cost(&this_mv, center_mv, mvcost, x->errorperbit);
}

#if HAVE_SSE2 || HAVE_MSA
int vp8_refining_search_sadx4(MACROBLOCK *x, BLOCK *b, BLOCKD *d,
                              int_mv *ref_mv, int error_per_bit,
                              int search_range, vp8_variance_fn_ptr_t *fn_ptr,
                              int *mvcost[2], int_mv *center_mv) {
  MV neighbors[4] = { { -1, 0 }, { 0, -1 }, { 0, 1 }, { 1, 0 } };
  int i, j;
  short this_row_offset, this_col_offset;

  int what_stride = b->src_stride;
  int pre_stride = x->e_mbd.pre.y_stride;
  unsigned char *base_pre = x->e_mbd.pre.y_buffer;
  int in_what_stride = pre_stride;
  unsigned char *what = (*(b->base_src) + b->src);
  unsigned char *best_address =
      (unsigned char *)(base_pre + d->offset +
                        (ref_mv->as_mv.row * pre_stride) + ref_mv->as_mv.col);
  unsigned char *check_here;
  int_mv this_mv;
  unsigned int bestsad;
  unsigned int thissad;

  int *mvsadcost[2];
  int_mv fcenter_mv;

  mvsadcost[0] = x->mvsadcost[0];
  mvsadcost[1] = x->mvsadcost[1];
  fcenter_mv.as_mv.row = center_mv->as_mv.row >> 3;
  fcenter_mv.as_mv.col = center_mv->as_mv.col >> 3;

  bestsad = fn_ptr->sdf(what, what_stride, best_address, in_what_stride) +
            mvsad_err_cost(ref_mv, &fcenter_mv, mvsadcost, error_per_bit);

  for (i = 0; i < search_range; ++i) {
    int best_site = -1;
    int all_in = 1;

    all_in &= ((ref_mv->as_mv.row - 1) > x->mv_row_min);
    all_in &= ((ref_mv->as_mv.row + 1) < x->mv_row_max);
    all_in &= ((ref_mv->as_mv.col - 1) > x->mv_col_min);
    all_in &= ((ref_mv->as_mv.col + 1) < x->mv_col_max);

    if (all_in) {
      unsigned int sad_array[4];
      const unsigned char *block_offset[4];
      block_offset[0] = best_address - in_what_stride;
      block_offset[1] = best_address - 1;
      block_offset[2] = best_address + 1;
      block_offset[3] = best_address + in_what_stride;

      fn_ptr->sdx4df(what, what_stride, block_offset, in_what_stride,
                     sad_array);

      for (j = 0; j < 4; ++j) {
        if (sad_array[j] < bestsad) {
          this_mv.as_mv.row = ref_mv->as_mv.row + neighbors[j].row;
          this_mv.as_mv.col = ref_mv->as_mv.col + neighbors[j].col;
          sad_array[j] +=
              mvsad_err_cost(&this_mv, &fcenter_mv, mvsadcost, error_per_bit);

          if (sad_array[j] < bestsad) {
            bestsad = sad_array[j];
            best_site = j;
          }
        }
      }
    } else {
      for (j = 0; j < 4; ++j) {
        this_row_offset = ref_mv->as_mv.row + neighbors[j].row;
        this_col_offset = ref_mv->as_mv.col + neighbors[j].col;

        if ((this_col_offset > x->mv_col_min) &&
            (this_col_offset < x->mv_col_max) &&
            (this_row_offset > x->mv_row_min) &&
            (this_row_offset < x->mv_row_max)) {
          check_here = (neighbors[j].row) * in_what_stride + neighbors[j].col +
                       best_address;
          thissad = fn_ptr->sdf(what, what_stride, check_here, in_what_stride);

          if (thissad < bestsad) {
            this_mv.as_mv.row = this_row_offset;
            this_mv.as_mv.col = this_col_offset;
            thissad +=
                mvsad_err_cost(&this_mv, &fcenter_mv, mvsadcost, error_per_bit);

            if (thissad < bestsad) {
              bestsad = thissad;
              best_site = j;
            }
          }
        }
      }
    }

    if (best_site == -1) {
      break;
    } else {
      ref_mv->as_mv.row += neighbors[best_site].row;
      ref_mv->as_mv.col += neighbors[best_site].col;
      best_address += (neighbors[best_site].row) * in_what_stride +
                      neighbors[best_site].col;
    }
  }

  this_mv.as_mv.row = clamp(ref_mv->as_mv.row * 8, SHRT_MIN, SHRT_MAX);
  this_mv.as_mv.col = clamp(ref_mv->as_mv.col * 8, SHRT_MIN, SHRT_MAX);

  return fn_ptr->vf(what, what_stride, best_address, in_what_stride, &thissad) +
         mv_err_cost(&this_mv, center_mv, mvcost, x->errorperbit);
}
#endif  // HAVE_SSE2 || HAVE_MSA
