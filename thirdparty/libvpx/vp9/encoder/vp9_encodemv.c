/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <math.h>

#include "vp9/common/vp9_common.h"
#include "vp9/common/vp9_entropymode.h"

#include "vp9/encoder/vp9_cost.h"
#include "vp9/encoder/vp9_encodemv.h"

#include "vpx_dsp/vpx_dsp_common.h"

static struct vp9_token mv_joint_encodings[MV_JOINTS];
static struct vp9_token mv_class_encodings[MV_CLASSES];
static struct vp9_token mv_fp_encodings[MV_FP_SIZE];

void vp9_entropy_mv_init(void) {
  vp9_tokens_from_tree(mv_joint_encodings, vp9_mv_joint_tree);
  vp9_tokens_from_tree(mv_class_encodings, vp9_mv_class_tree);
  vp9_tokens_from_tree(mv_fp_encodings, vp9_mv_fp_tree);
}

static void encode_mv_component(vpx_writer *w, int comp,
                                const nmv_component *mvcomp, int usehp) {
  int offset;
  const int sign = comp < 0;
  const int mag = sign ? -comp : comp;
  const int mv_class = vp9_get_mv_class(mag - 1, &offset);
  const int d = offset >> 3;         // int mv data
  const int fr = (offset >> 1) & 3;  // fractional mv data
  const int hp = offset & 1;         // high precision mv data

  assert(comp != 0);

  // Sign
  vpx_write(w, sign, mvcomp->sign);

  // Class
  vp9_write_token(w, vp9_mv_class_tree, mvcomp->classes,
                  &mv_class_encodings[mv_class]);

  // Integer bits
  if (mv_class == MV_CLASS_0) {
    vpx_write(w, d, mvcomp->class0[0]);
  } else {
    int i;
    const int n = mv_class + CLASS0_BITS - 1;  // number of bits
    for (i = 0; i < n; ++i) vpx_write(w, (d >> i) & 1, mvcomp->bits[i]);
  }

  // Fractional bits
  vp9_write_token(w, vp9_mv_fp_tree,
                  mv_class == MV_CLASS_0 ? mvcomp->class0_fp[d] : mvcomp->fp,
                  &mv_fp_encodings[fr]);

  // High precision bit
  if (usehp)
    vpx_write(w, hp, mv_class == MV_CLASS_0 ? mvcomp->class0_hp : mvcomp->hp);
}

static void build_nmv_component_cost_table(int *mvcost,
                                           const nmv_component *const mvcomp,
                                           int usehp) {
  int sign_cost[2], class_cost[MV_CLASSES], class0_cost[CLASS0_SIZE];
  int bits_cost[MV_OFFSET_BITS][2];
  int class0_fp_cost[CLASS0_SIZE][MV_FP_SIZE], fp_cost[MV_FP_SIZE];
  int class0_hp_cost[2], hp_cost[2];
  int i;
  int c, o;

  sign_cost[0] = vp9_cost_zero(mvcomp->sign);
  sign_cost[1] = vp9_cost_one(mvcomp->sign);
  vp9_cost_tokens(class_cost, mvcomp->classes, vp9_mv_class_tree);
  vp9_cost_tokens(class0_cost, mvcomp->class0, vp9_mv_class0_tree);
  for (i = 0; i < MV_OFFSET_BITS; ++i) {
    bits_cost[i][0] = vp9_cost_zero(mvcomp->bits[i]);
    bits_cost[i][1] = vp9_cost_one(mvcomp->bits[i]);
  }

  for (i = 0; i < CLASS0_SIZE; ++i)
    vp9_cost_tokens(class0_fp_cost[i], mvcomp->class0_fp[i], vp9_mv_fp_tree);
  vp9_cost_tokens(fp_cost, mvcomp->fp, vp9_mv_fp_tree);

  // Always build the hp costs to avoid an uninitialized warning from gcc
  class0_hp_cost[0] = vp9_cost_zero(mvcomp->class0_hp);
  class0_hp_cost[1] = vp9_cost_one(mvcomp->class0_hp);
  hp_cost[0] = vp9_cost_zero(mvcomp->hp);
  hp_cost[1] = vp9_cost_one(mvcomp->hp);

  mvcost[0] = 0;
  // MV_CLASS_0
  for (o = 0; o < (CLASS0_SIZE << 3); ++o) {
    int d, e, f;
    int cost = class_cost[MV_CLASS_0];
    int v = o + 1;
    d = (o >> 3);     /* int mv data */
    f = (o >> 1) & 3; /* fractional pel mv data */
    cost += class0_cost[d];
    cost += class0_fp_cost[d][f];
    if (usehp) {
      e = (o & 1); /* high precision mv data */
      cost += class0_hp_cost[e];
    }
    mvcost[v] = cost + sign_cost[0];
    mvcost[-v] = cost + sign_cost[1];
  }
  for (c = MV_CLASS_1; c < MV_CLASSES; ++c) {
    int d;
    for (d = 0; d < (1 << c); ++d) {
      int f;
      int whole_cost = class_cost[c];
      int b = c + CLASS0_BITS - 1; /* number of bits */
      for (i = 0; i < b; ++i) whole_cost += bits_cost[i][((d >> i) & 1)];
      for (f = 0; f < 4; ++f) {
        int cost = whole_cost + fp_cost[f];
        int v = (CLASS0_SIZE << (c + 2)) + d * 8 + f * 2 /* + e */ + 1;
        if (usehp) {
          mvcost[v] = cost + hp_cost[0] + sign_cost[0];
          mvcost[-v] = cost + hp_cost[0] + sign_cost[1];
          if (v + 1 > MV_MAX) break;
          mvcost[v + 1] = cost + hp_cost[1] + sign_cost[0];
          mvcost[-v - 1] = cost + hp_cost[1] + sign_cost[1];
        } else {
          mvcost[v] = cost + sign_cost[0];
          mvcost[-v] = cost + sign_cost[1];
          if (v + 1 > MV_MAX) break;
          mvcost[v + 1] = cost + sign_cost[0];
          mvcost[-v - 1] = cost + sign_cost[1];
        }
      }
    }
  }
}

static int update_mv(vpx_writer *w, const unsigned int ct[2], vpx_prob *cur_p,
                     vpx_prob upd_p) {
  const vpx_prob new_p = get_binary_prob(ct[0], ct[1]) | 1;
  const int update = cost_branch256(ct, *cur_p) + vp9_cost_zero(upd_p) >
                     cost_branch256(ct, new_p) + vp9_cost_one(upd_p) +
                         (7 << VP9_PROB_COST_SHIFT);
  vpx_write(w, update, upd_p);
  if (update) {
    *cur_p = new_p;
    vpx_write_literal(w, new_p >> 1, 7);
  }
  return update;
}

static void write_mv_update(const vpx_tree_index *tree,
                            vpx_prob probs[/*n - 1*/],
                            const unsigned int counts[/*n - 1*/], int n,
                            vpx_writer *w) {
  int i;
  unsigned int branch_ct[32][2];

  // Assuming max number of probabilities <= 32
  assert(n <= 32);

  vp9_tree_probs_from_distribution(tree, branch_ct, counts);
  for (i = 0; i < n - 1; ++i)
    update_mv(w, branch_ct[i], &probs[i], MV_UPDATE_PROB);
}

void vp9_write_nmv_probs(VP9_COMMON *cm, int usehp, vpx_writer *w,
                         nmv_context_counts *const counts) {
  int i, j;
  nmv_context *const mvc = &cm->fc->nmvc;

  write_mv_update(vp9_mv_joint_tree, mvc->joints, counts->joints, MV_JOINTS, w);

  for (i = 0; i < 2; ++i) {
    nmv_component *comp = &mvc->comps[i];
    nmv_component_counts *comp_counts = &counts->comps[i];

    update_mv(w, comp_counts->sign, &comp->sign, MV_UPDATE_PROB);
    write_mv_update(vp9_mv_class_tree, comp->classes, comp_counts->classes,
                    MV_CLASSES, w);
    write_mv_update(vp9_mv_class0_tree, comp->class0, comp_counts->class0,
                    CLASS0_SIZE, w);
    for (j = 0; j < MV_OFFSET_BITS; ++j)
      update_mv(w, comp_counts->bits[j], &comp->bits[j], MV_UPDATE_PROB);
  }

  for (i = 0; i < 2; ++i) {
    for (j = 0; j < CLASS0_SIZE; ++j)
      write_mv_update(vp9_mv_fp_tree, mvc->comps[i].class0_fp[j],
                      counts->comps[i].class0_fp[j], MV_FP_SIZE, w);

    write_mv_update(vp9_mv_fp_tree, mvc->comps[i].fp, counts->comps[i].fp,
                    MV_FP_SIZE, w);
  }

  if (usehp) {
    for (i = 0; i < 2; ++i) {
      update_mv(w, counts->comps[i].class0_hp, &mvc->comps[i].class0_hp,
                MV_UPDATE_PROB);
      update_mv(w, counts->comps[i].hp, &mvc->comps[i].hp, MV_UPDATE_PROB);
    }
  }
}

void vp9_encode_mv(VP9_COMP *cpi, vpx_writer *w, const MV *mv, const MV *ref,
                   const nmv_context *mvctx, int usehp,
                   unsigned int *const max_mv_magnitude) {
  const MV diff = { mv->row - ref->row, mv->col - ref->col };
  const MV_JOINT_TYPE j = vp9_get_mv_joint(&diff);
  usehp = usehp && use_mv_hp(ref);

  vp9_write_token(w, vp9_mv_joint_tree, mvctx->joints, &mv_joint_encodings[j]);
  if (mv_joint_vertical(j))
    encode_mv_component(w, diff.row, &mvctx->comps[0], usehp);

  if (mv_joint_horizontal(j))
    encode_mv_component(w, diff.col, &mvctx->comps[1], usehp);

  // If auto_mv_step_size is enabled then keep track of the largest
  // motion vector component used.
  if (cpi->sf.mv.auto_mv_step_size) {
    const unsigned int maxv = VPXMAX(abs(mv->row), abs(mv->col)) >> 3;
    *max_mv_magnitude = VPXMAX(maxv, *max_mv_magnitude);
  }
}

void vp9_build_nmv_cost_table(int *mvjoint, int *mvcost[2],
                              const nmv_context *ctx, int usehp) {
  vp9_cost_tokens(mvjoint, ctx->joints, vp9_mv_joint_tree);
  build_nmv_component_cost_table(mvcost[0], &ctx->comps[0], usehp);
  build_nmv_component_cost_table(mvcost[1], &ctx->comps[1], usehp);
}

static void inc_mvs(const MODE_INFO *mi, const MB_MODE_INFO_EXT *mbmi_ext,
                    const int_mv mvs[2], nmv_context_counts *counts) {
  int i;

  for (i = 0; i < 1 + has_second_ref(mi); ++i) {
    const MV *ref = &mbmi_ext->ref_mvs[mi->ref_frame[i]][0].as_mv;
    const MV diff = { mvs[i].as_mv.row - ref->row,
                      mvs[i].as_mv.col - ref->col };
    vp9_inc_mv(&diff, counts);
  }
}

void vp9_update_mv_count(ThreadData *td) {
  const MACROBLOCKD *xd = &td->mb.e_mbd;
  const MODE_INFO *mi = xd->mi[0];
  const MB_MODE_INFO_EXT *mbmi_ext = td->mb.mbmi_ext;

  if (mi->sb_type < BLOCK_8X8) {
    const int num_4x4_w = num_4x4_blocks_wide_lookup[mi->sb_type];
    const int num_4x4_h = num_4x4_blocks_high_lookup[mi->sb_type];
    int idx, idy;

    for (idy = 0; idy < 2; idy += num_4x4_h) {
      for (idx = 0; idx < 2; idx += num_4x4_w) {
        const int i = idy * 2 + idx;
        if (mi->bmi[i].as_mode == NEWMV)
          inc_mvs(mi, mbmi_ext, mi->bmi[i].as_mv, &td->counts->mv);
      }
    }
  } else {
    if (mi->mode == NEWMV) inc_mvs(mi, mbmi_ext, mi->mv, &td->counts->mv);
  }
}
