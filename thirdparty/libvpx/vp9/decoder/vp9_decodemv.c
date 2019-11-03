/*
  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>

#include "vp9/common/vp9_common.h"
#include "vp9/common/vp9_entropy.h"
#include "vp9/common/vp9_entropymode.h"
#include "vp9/common/vp9_entropymv.h"
#include "vp9/common/vp9_mvref_common.h"
#include "vp9/common/vp9_pred_common.h"
#include "vp9/common/vp9_reconinter.h"
#include "vp9/common/vp9_seg_common.h"

#include "vp9/decoder/vp9_decodemv.h"
#include "vp9/decoder/vp9_decodeframe.h"

#include "vpx_dsp/vpx_dsp_common.h"

static PREDICTION_MODE read_intra_mode(vpx_reader *r, const vpx_prob *p) {
  return (PREDICTION_MODE)vpx_read_tree(r, vp9_intra_mode_tree, p);
}

static PREDICTION_MODE read_intra_mode_y(VP9_COMMON *cm, MACROBLOCKD *xd,
                                         vpx_reader *r, int size_group) {
  const PREDICTION_MODE y_mode =
      read_intra_mode(r, cm->fc->y_mode_prob[size_group]);
  FRAME_COUNTS *counts = xd->counts;
  if (counts)
    ++counts->y_mode[size_group][y_mode];
  return y_mode;
}

static PREDICTION_MODE read_intra_mode_uv(VP9_COMMON *cm, MACROBLOCKD *xd,
                                          vpx_reader *r,
                                          PREDICTION_MODE y_mode) {
  const PREDICTION_MODE uv_mode = read_intra_mode(r,
                                         cm->fc->uv_mode_prob[y_mode]);
  FRAME_COUNTS *counts = xd->counts;
  if (counts)
    ++counts->uv_mode[y_mode][uv_mode];
  return uv_mode;
}

static PREDICTION_MODE read_inter_mode(VP9_COMMON *cm, MACROBLOCKD *xd,
                                       vpx_reader *r, int ctx) {
  const int mode = vpx_read_tree(r, vp9_inter_mode_tree,
                                 cm->fc->inter_mode_probs[ctx]);
  FRAME_COUNTS *counts = xd->counts;
  if (counts)
    ++counts->inter_mode[ctx][mode];

  return NEARESTMV + mode;
}

static int read_segment_id(vpx_reader *r, const struct segmentation *seg) {
  return vpx_read_tree(r, vp9_segment_tree, seg->tree_probs);
}

static TX_SIZE read_selected_tx_size(VP9_COMMON *cm, MACROBLOCKD *xd,
                                     TX_SIZE max_tx_size, vpx_reader *r) {
  FRAME_COUNTS *counts = xd->counts;
  const int ctx = get_tx_size_context(xd);
  const vpx_prob *tx_probs = get_tx_probs(max_tx_size, ctx, &cm->fc->tx_probs);
  int tx_size = vpx_read(r, tx_probs[0]);
  if (tx_size != TX_4X4 && max_tx_size >= TX_16X16) {
    tx_size += vpx_read(r, tx_probs[1]);
    if (tx_size != TX_8X8 && max_tx_size >= TX_32X32)
      tx_size += vpx_read(r, tx_probs[2]);
  }

  if (counts)
    ++get_tx_counts(max_tx_size, ctx, &counts->tx)[tx_size];
  return (TX_SIZE)tx_size;
}

static INLINE TX_SIZE read_tx_size(VP9_COMMON *cm, MACROBLOCKD *xd,
                                   int allow_select, vpx_reader *r) {
  TX_MODE tx_mode = cm->tx_mode;
  BLOCK_SIZE bsize = xd->mi[0]->sb_type;
  const TX_SIZE max_tx_size = max_txsize_lookup[bsize];
  if (allow_select && tx_mode == TX_MODE_SELECT && bsize >= BLOCK_8X8)
    return read_selected_tx_size(cm, xd, max_tx_size, r);
  else
    return VPXMIN(max_tx_size, tx_mode_to_biggest_tx_size[tx_mode]);
}

static int dec_get_segment_id(const VP9_COMMON *cm, const uint8_t *segment_ids,
                              int mi_offset, int x_mis, int y_mis) {
  int x, y, segment_id = INT_MAX;

  for (y = 0; y < y_mis; y++)
    for (x = 0; x < x_mis; x++)
      segment_id =
          VPXMIN(segment_id, segment_ids[mi_offset + y * cm->mi_cols + x]);

  assert(segment_id >= 0 && segment_id < MAX_SEGMENTS);
  return segment_id;
}

static void set_segment_id(VP9_COMMON *cm, int mi_offset,
                           int x_mis, int y_mis, int segment_id) {
  int x, y;

  assert(segment_id >= 0 && segment_id < MAX_SEGMENTS);

  for (y = 0; y < y_mis; y++)
    for (x = 0; x < x_mis; x++)
      cm->current_frame_seg_map[mi_offset + y * cm->mi_cols + x] = segment_id;
}

static void copy_segment_id(const VP9_COMMON *cm,
                           const uint8_t *last_segment_ids,
                           uint8_t *current_segment_ids,
                           int mi_offset, int x_mis, int y_mis) {
  int x, y;

  for (y = 0; y < y_mis; y++)
    for (x = 0; x < x_mis; x++)
      current_segment_ids[mi_offset + y * cm->mi_cols + x] =  last_segment_ids ?
          last_segment_ids[mi_offset + y * cm->mi_cols + x] : 0;
}

static int read_intra_segment_id(VP9_COMMON *const cm, int mi_offset,
                                 int x_mis, int y_mis,
                                 vpx_reader *r) {
  struct segmentation *const seg = &cm->seg;
  int segment_id;

  if (!seg->enabled)
    return 0;  // Default for disabled segmentation

  if (!seg->update_map) {
    copy_segment_id(cm, cm->last_frame_seg_map, cm->current_frame_seg_map,
                    mi_offset, x_mis, y_mis);
    return 0;
  }

  segment_id = read_segment_id(r, seg);
  set_segment_id(cm, mi_offset, x_mis, y_mis, segment_id);
  return segment_id;
}

static int read_inter_segment_id(VP9_COMMON *const cm, MACROBLOCKD *const xd,
                                 int mi_row, int mi_col, vpx_reader *r,
                                 int x_mis, int y_mis) {
  struct segmentation *const seg = &cm->seg;
  MODE_INFO *const mi = xd->mi[0];
  int predicted_segment_id, segment_id;
  const int mi_offset = mi_row * cm->mi_cols + mi_col;

  if (!seg->enabled)
    return 0;  // Default for disabled segmentation

  predicted_segment_id = cm->last_frame_seg_map ?
      dec_get_segment_id(cm, cm->last_frame_seg_map, mi_offset, x_mis, y_mis) :
      0;

  if (!seg->update_map) {
    copy_segment_id(cm, cm->last_frame_seg_map, cm->current_frame_seg_map,
                    mi_offset, x_mis, y_mis);
    return predicted_segment_id;
  }

  if (seg->temporal_update) {
    const vpx_prob pred_prob = vp9_get_pred_prob_seg_id(seg, xd);
    mi->seg_id_predicted = vpx_read(r, pred_prob);
    segment_id = mi->seg_id_predicted ? predicted_segment_id
                                      : read_segment_id(r, seg);
  } else {
    segment_id = read_segment_id(r, seg);
  }
  set_segment_id(cm, mi_offset, x_mis, y_mis, segment_id);
  return segment_id;
}

static int read_skip(VP9_COMMON *cm, const MACROBLOCKD *xd,
                     int segment_id, vpx_reader *r) {
  if (segfeature_active(&cm->seg, segment_id, SEG_LVL_SKIP)) {
    return 1;
  } else {
    const int ctx = vp9_get_skip_context(xd);
    const int skip = vpx_read(r, cm->fc->skip_probs[ctx]);
    FRAME_COUNTS *counts = xd->counts;
    if (counts)
      ++counts->skip[ctx][skip];
    return skip;
  }
}

static void read_intra_frame_mode_info(VP9_COMMON *const cm,
                                       MACROBLOCKD *const xd,
                                       int mi_row, int mi_col, vpx_reader *r,
                                       int x_mis, int y_mis) {
  MODE_INFO *const mi = xd->mi[0];
  const MODE_INFO *above_mi = xd->above_mi;
  const MODE_INFO *left_mi  = xd->left_mi;
  const BLOCK_SIZE bsize = mi->sb_type;
  int i;
  const int mi_offset = mi_row * cm->mi_cols + mi_col;

  mi->segment_id = read_intra_segment_id(cm, mi_offset, x_mis, y_mis, r);
  mi->skip = read_skip(cm, xd, mi->segment_id, r);
  mi->tx_size = read_tx_size(cm, xd, 1, r);
  mi->ref_frame[0] = INTRA_FRAME;
  mi->ref_frame[1] = NONE;

  switch (bsize) {
    case BLOCK_4X4:
      for (i = 0; i < 4; ++i)
        mi->bmi[i].as_mode =
            read_intra_mode(r, get_y_mode_probs(mi, above_mi, left_mi, i));
      mi->mode = mi->bmi[3].as_mode;
      break;
    case BLOCK_4X8:
      mi->bmi[0].as_mode = mi->bmi[2].as_mode =
          read_intra_mode(r, get_y_mode_probs(mi, above_mi, left_mi, 0));
      mi->bmi[1].as_mode = mi->bmi[3].as_mode = mi->mode =
          read_intra_mode(r, get_y_mode_probs(mi, above_mi, left_mi, 1));
      break;
    case BLOCK_8X4:
      mi->bmi[0].as_mode = mi->bmi[1].as_mode =
          read_intra_mode(r, get_y_mode_probs(mi, above_mi, left_mi, 0));
      mi->bmi[2].as_mode = mi->bmi[3].as_mode = mi->mode =
          read_intra_mode(r, get_y_mode_probs(mi, above_mi, left_mi, 2));
      break;
    default:
      mi->mode = read_intra_mode(r,
                                 get_y_mode_probs(mi, above_mi, left_mi, 0));
  }

  mi->uv_mode = read_intra_mode(r, vp9_kf_uv_mode_prob[mi->mode]);
}

static int read_mv_component(vpx_reader *r,
                             const nmv_component *mvcomp, int usehp) {
  int mag, d, fr, hp;
  const int sign = vpx_read(r, mvcomp->sign);
  const int mv_class = vpx_read_tree(r, vp9_mv_class_tree, mvcomp->classes);
  const int class0 = mv_class == MV_CLASS_0;

  // Integer part
  if (class0) {
    d = vpx_read_tree(r, vp9_mv_class0_tree, mvcomp->class0);
    mag = 0;
  } else {
    int i;
    const int n = mv_class + CLASS0_BITS - 1;  // number of bits

    d = 0;
    for (i = 0; i < n; ++i)
      d |= vpx_read(r, mvcomp->bits[i]) << i;
    mag = CLASS0_SIZE << (mv_class + 2);
  }

  // Fractional part
  fr = vpx_read_tree(r, vp9_mv_fp_tree, class0 ? mvcomp->class0_fp[d]
                                               : mvcomp->fp);

  // High precision part (if hp is not used, the default value of the hp is 1)
  hp = usehp ? vpx_read(r, class0 ? mvcomp->class0_hp : mvcomp->hp)
             : 1;

  // Result
  mag += ((d << 3) | (fr << 1) | hp) + 1;
  return sign ? -mag : mag;
}

static INLINE void read_mv(vpx_reader *r, MV *mv, const MV *ref,
                           const nmv_context *ctx,
                           nmv_context_counts *counts, int allow_hp) {
  const MV_JOINT_TYPE joint_type =
      (MV_JOINT_TYPE)vpx_read_tree(r, vp9_mv_joint_tree, ctx->joints);
  const int use_hp = allow_hp && use_mv_hp(ref);
  MV diff = {0, 0};

  if (mv_joint_vertical(joint_type))
    diff.row = read_mv_component(r, &ctx->comps[0], use_hp);

  if (mv_joint_horizontal(joint_type))
    diff.col = read_mv_component(r, &ctx->comps[1], use_hp);

  vp9_inc_mv(&diff, counts);

  mv->row = ref->row + diff.row;
  mv->col = ref->col + diff.col;
}

static REFERENCE_MODE read_block_reference_mode(VP9_COMMON *cm,
                                                const MACROBLOCKD *xd,
                                                vpx_reader *r) {
  if (cm->reference_mode == REFERENCE_MODE_SELECT) {
    const int ctx = vp9_get_reference_mode_context(cm, xd);
    const REFERENCE_MODE mode =
        (REFERENCE_MODE)vpx_read(r, cm->fc->comp_inter_prob[ctx]);
    FRAME_COUNTS *counts = xd->counts;
    if (counts)
      ++counts->comp_inter[ctx][mode];
    return mode;  // SINGLE_REFERENCE or COMPOUND_REFERENCE
  } else {
    return cm->reference_mode;
  }
}

// Read the referncence frame
static void read_ref_frames(VP9_COMMON *const cm, MACROBLOCKD *const xd,
                            vpx_reader *r,
                            int segment_id, MV_REFERENCE_FRAME ref_frame[2]) {
  FRAME_CONTEXT *const fc = cm->fc;
  FRAME_COUNTS *counts = xd->counts;

  if (segfeature_active(&cm->seg, segment_id, SEG_LVL_REF_FRAME)) {
    ref_frame[0] = (MV_REFERENCE_FRAME)get_segdata(&cm->seg, segment_id,
                                                   SEG_LVL_REF_FRAME);
    ref_frame[1] = NONE;
  } else {
    const REFERENCE_MODE mode = read_block_reference_mode(cm, xd, r);
    // FIXME(rbultje) I'm pretty sure this breaks segmentation ref frame coding
    if (mode == COMPOUND_REFERENCE) {
      const int idx = cm->ref_frame_sign_bias[cm->comp_fixed_ref];
      const int ctx = vp9_get_pred_context_comp_ref_p(cm, xd);
      const int bit = vpx_read(r, fc->comp_ref_prob[ctx]);
      if (counts)
        ++counts->comp_ref[ctx][bit];
      ref_frame[idx] = cm->comp_fixed_ref;
      ref_frame[!idx] = cm->comp_var_ref[bit];
    } else if (mode == SINGLE_REFERENCE) {
      const int ctx0 = vp9_get_pred_context_single_ref_p1(xd);
      const int bit0 = vpx_read(r, fc->single_ref_prob[ctx0][0]);
      if (counts)
        ++counts->single_ref[ctx0][0][bit0];
      if (bit0) {
        const int ctx1 = vp9_get_pred_context_single_ref_p2(xd);
        const int bit1 = vpx_read(r, fc->single_ref_prob[ctx1][1]);
        if (counts)
          ++counts->single_ref[ctx1][1][bit1];
        ref_frame[0] = bit1 ? ALTREF_FRAME : GOLDEN_FRAME;
      } else {
        ref_frame[0] = LAST_FRAME;
      }

      ref_frame[1] = NONE;
    } else {
      assert(0 && "Invalid prediction mode.");
    }
  }
}

// TODO(slavarnway): Move this decoder version of
// vp9_get_pred_context_switchable_interp() to vp9_pred_common.h and update the
// encoder.
//
// Returns a context number for the given MB prediction signal
static int dec_get_pred_context_switchable_interp(const MACROBLOCKD *xd) {
  // Note:
  // The mode info data structure has a one element border above and to the
  // left of the entries corresponding to real macroblocks.
  // The prediction flags in these dummy entries are initialized to 0.
  const MODE_INFO *const left_mi = xd->left_mi;
  const int left_type = left_mi ? left_mi->interp_filter : SWITCHABLE_FILTERS;
  const MODE_INFO *const above_mi = xd->above_mi;
  const int above_type = above_mi ? above_mi->interp_filter
                             : SWITCHABLE_FILTERS;

  if (left_type == above_type)
    return left_type;
  else if (left_type == SWITCHABLE_FILTERS)
    return above_type;
  else if (above_type == SWITCHABLE_FILTERS)
    return left_type;
  else
    return SWITCHABLE_FILTERS;
}

static INLINE INTERP_FILTER read_switchable_interp_filter(
    VP9_COMMON *const cm, MACROBLOCKD *const xd,
    vpx_reader *r) {
  const int ctx = dec_get_pred_context_switchable_interp(xd);
  const INTERP_FILTER type =
      (INTERP_FILTER)vpx_read_tree(r, vp9_switchable_interp_tree,
                                   cm->fc->switchable_interp_prob[ctx]);
  FRAME_COUNTS *counts = xd->counts;
  if (counts)
    ++counts->switchable_interp[ctx][type];
  return type;
}

static void read_intra_block_mode_info(VP9_COMMON *const cm,
                                       MACROBLOCKD *const xd, MODE_INFO *mi,
                                       vpx_reader *r) {
  const BLOCK_SIZE bsize = mi->sb_type;
  int i;

  switch (bsize) {
    case BLOCK_4X4:
      for (i = 0; i < 4; ++i)
        mi->bmi[i].as_mode = read_intra_mode_y(cm, xd, r, 0);
      mi->mode = mi->bmi[3].as_mode;
      break;
    case BLOCK_4X8:
      mi->bmi[0].as_mode = mi->bmi[2].as_mode = read_intra_mode_y(cm, xd,
                                                                  r, 0);
      mi->bmi[1].as_mode = mi->bmi[3].as_mode = mi->mode =
          read_intra_mode_y(cm, xd, r, 0);
      break;
    case BLOCK_8X4:
      mi->bmi[0].as_mode = mi->bmi[1].as_mode = read_intra_mode_y(cm, xd,
                                                                  r, 0);
      mi->bmi[2].as_mode = mi->bmi[3].as_mode = mi->mode =
          read_intra_mode_y(cm, xd, r, 0);
      break;
    default:
      mi->mode = read_intra_mode_y(cm, xd, r, size_group_lookup[bsize]);
  }

  mi->uv_mode = read_intra_mode_uv(cm, xd, r, mi->mode);

  // Initialize interp_filter here so we do not have to check for inter block
  // modes in dec_get_pred_context_switchable_interp()
  mi->interp_filter = SWITCHABLE_FILTERS;

  mi->ref_frame[0] = INTRA_FRAME;
  mi->ref_frame[1] = NONE;
}

static INLINE int is_mv_valid(const MV *mv) {
  return mv->row > MV_LOW && mv->row < MV_UPP &&
         mv->col > MV_LOW && mv->col < MV_UPP;
}

static INLINE void copy_mv_pair(int_mv *dst, const int_mv *src) {
  memcpy(dst, src, sizeof(*dst) * 2);
}

static INLINE void zero_mv_pair(int_mv *dst) {
  memset(dst, 0, sizeof(*dst) * 2);
}

static INLINE int assign_mv(VP9_COMMON *cm, MACROBLOCKD *xd,
                            PREDICTION_MODE mode,
                            int_mv mv[2], int_mv ref_mv[2],
                            int_mv near_nearest_mv[2],
                            int is_compound, int allow_hp, vpx_reader *r) {
  int i;
  int ret = 1;

  switch (mode) {
    case NEWMV: {
      FRAME_COUNTS *counts = xd->counts;
      nmv_context_counts *const mv_counts = counts ? &counts->mv : NULL;
      for (i = 0; i < 1 + is_compound; ++i) {
        read_mv(r, &mv[i].as_mv, &ref_mv[i].as_mv, &cm->fc->nmvc, mv_counts,
                allow_hp);
        ret = ret && is_mv_valid(&mv[i].as_mv);
      }
      break;
    }
    case NEARMV:
    case NEARESTMV: {
      copy_mv_pair(mv, near_nearest_mv);
      break;
    }
    case ZEROMV: {
      zero_mv_pair(mv);
      break;
    }
    default: {
      return 0;
    }
  }
  return ret;
}

static int read_is_inter_block(VP9_COMMON *const cm, MACROBLOCKD *const xd,
                               int segment_id, vpx_reader *r) {
  if (segfeature_active(&cm->seg, segment_id, SEG_LVL_REF_FRAME)) {
    return get_segdata(&cm->seg, segment_id, SEG_LVL_REF_FRAME) != INTRA_FRAME;
  } else {
    const int ctx = get_intra_inter_context(xd);
    const int is_inter = vpx_read(r, cm->fc->intra_inter_prob[ctx]);
    FRAME_COUNTS *counts = xd->counts;
    if (counts)
      ++counts->intra_inter[ctx][is_inter];
    return is_inter;
  }
}

static void dec_find_best_ref_mvs(int allow_hp, int_mv *mvlist, int_mv *best_mv,
                                  int refmv_count) {
  int i;

  // Make sure all the candidates are properly clamped etc
  for (i = 0; i < refmv_count; ++i) {
    lower_mv_precision(&mvlist[i].as_mv, allow_hp);
    *best_mv = mvlist[i];
  }
}

static void fpm_sync(void *const data, int mi_row) {
  VP9Decoder *const pbi = (VP9Decoder *)data;
  vp9_frameworker_wait(pbi->frame_worker_owner, pbi->common.prev_frame,
                       mi_row << MI_BLOCK_SIZE_LOG2);
}

// This macro is used to add a motion vector mv_ref list if it isn't
// already in the list.  If it's the second motion vector or early_break
// it will also skip all additional processing and jump to Done!
#define ADD_MV_REF_LIST_EB(mv, refmv_count, mv_ref_list, Done) \
  do { \
    if (refmv_count) { \
      if ((mv).as_int != (mv_ref_list)[0].as_int) { \
        (mv_ref_list)[(refmv_count)] = (mv); \
        refmv_count++; \
        goto Done; \
      } \
    } else { \
      (mv_ref_list)[(refmv_count)++] = (mv); \
      if (early_break) \
        goto Done; \
    } \
  } while (0)

// If either reference frame is different, not INTRA, and they
// are different from each other scale and add the mv to our list.
#define IF_DIFF_REF_FRAME_ADD_MV_EB(mbmi, ref_frame, ref_sign_bias, \
                                    refmv_count, mv_ref_list, Done) \
  do { \
    if (is_inter_block(mbmi)) { \
      if ((mbmi)->ref_frame[0] != ref_frame) \
        ADD_MV_REF_LIST_EB(scale_mv((mbmi), 0, ref_frame, ref_sign_bias), \
                           refmv_count, mv_ref_list, Done); \
      if (has_second_ref(mbmi) && \
          (mbmi)->ref_frame[1] != ref_frame && \
          (mbmi)->mv[1].as_int != (mbmi)->mv[0].as_int) \
        ADD_MV_REF_LIST_EB(scale_mv((mbmi), 1, ref_frame, ref_sign_bias), \
                           refmv_count, mv_ref_list, Done); \
    } \
  } while (0)

// This function searches the neighborhood of a given MB/SB
// to try and find candidate reference vectors.
static int dec_find_mv_refs(const VP9_COMMON *cm, const MACROBLOCKD *xd,
                            PREDICTION_MODE mode, MV_REFERENCE_FRAME ref_frame,
                            const POSITION *const mv_ref_search,
                            int_mv *mv_ref_list,
                            int mi_row, int mi_col, int block, int is_sub8x8,
                            find_mv_refs_sync sync, void *const data) {
  const int *ref_sign_bias = cm->ref_frame_sign_bias;
  int i, refmv_count = 0;
  int different_ref_found = 0;
  const MV_REF *const prev_frame_mvs = cm->use_prev_frame_mvs ?
      cm->prev_frame->mvs + mi_row * cm->mi_cols + mi_col : NULL;
  const TileInfo *const tile = &xd->tile;
  // If mode is nearestmv or newmv (uses nearestmv as a reference) then stop
  // searching after the first mv is found.
  const int early_break = (mode != NEARMV);

  // Blank the reference vector list
  memset(mv_ref_list, 0, sizeof(*mv_ref_list) * MAX_MV_REF_CANDIDATES);

  i = 0;
  if (is_sub8x8) {
    // If the size < 8x8 we get the mv from the bmi substructure for the
    // nearest two blocks.
    for (i = 0; i < 2; ++i) {
      const POSITION *const mv_ref = &mv_ref_search[i];
      if (is_inside(tile, mi_col, mi_row, cm->mi_rows, mv_ref)) {
        const MODE_INFO *const candidate_mi =
            xd->mi[mv_ref->col + mv_ref->row * xd->mi_stride];
        different_ref_found = 1;

        if (candidate_mi->ref_frame[0] == ref_frame)
          ADD_MV_REF_LIST_EB(
              get_sub_block_mv(candidate_mi, 0, mv_ref->col, block),
                  refmv_count, mv_ref_list, Done);
        else if (candidate_mi->ref_frame[1] == ref_frame)
          ADD_MV_REF_LIST_EB(
              get_sub_block_mv(candidate_mi, 1, mv_ref->col, block),
                  refmv_count, mv_ref_list, Done);
      }
    }
  }

  // Check the rest of the neighbors in much the same way
  // as before except we don't need to keep track of sub blocks or
  // mode counts.
  for (; i < MVREF_NEIGHBOURS; ++i) {
    const POSITION *const mv_ref = &mv_ref_search[i];
    if (is_inside(tile, mi_col, mi_row, cm->mi_rows, mv_ref)) {
      const MODE_INFO *const candidate =
          xd->mi[mv_ref->col + mv_ref->row * xd->mi_stride];
      different_ref_found = 1;

      if (candidate->ref_frame[0] == ref_frame)
        ADD_MV_REF_LIST_EB(candidate->mv[0], refmv_count, mv_ref_list, Done);
      else if (candidate->ref_frame[1] == ref_frame)
        ADD_MV_REF_LIST_EB(candidate->mv[1], refmv_count, mv_ref_list, Done);
    }
  }

  // TODO(hkuang): Remove this sync after fixing pthread_cond_broadcast
  // on windows platform. The sync here is unnecessary if use_prev_frame_mvs
  // is 0. But after removing it, there will be hang in the unit test on windows
  // due to several threads waiting for a thread's signal.
#if defined(_WIN32) && !HAVE_PTHREAD_H
    if (cm->frame_parallel_decode && sync != NULL) {
      sync(data, mi_row);
    }
#endif

  // Check the last frame's mode and mv info.
  if (prev_frame_mvs) {
    // Synchronize here for frame parallel decode if sync function is provided.
    if (cm->frame_parallel_decode && sync != NULL) {
      sync(data, mi_row);
    }

    if (prev_frame_mvs->ref_frame[0] == ref_frame) {
      ADD_MV_REF_LIST_EB(prev_frame_mvs->mv[0], refmv_count, mv_ref_list, Done);
    } else if (prev_frame_mvs->ref_frame[1] == ref_frame) {
      ADD_MV_REF_LIST_EB(prev_frame_mvs->mv[1], refmv_count, mv_ref_list, Done);
    }
  }

  // Since we couldn't find 2 mvs from the same reference frame
  // go back through the neighbors and find motion vectors from
  // different reference frames.
  if (different_ref_found) {
    for (i = 0; i < MVREF_NEIGHBOURS; ++i) {
      const POSITION *mv_ref = &mv_ref_search[i];
      if (is_inside(tile, mi_col, mi_row, cm->mi_rows, mv_ref)) {
        const MODE_INFO *const candidate =
            xd->mi[mv_ref->col + mv_ref->row * xd->mi_stride];

        // If the candidate is INTRA we don't want to consider its mv.
        IF_DIFF_REF_FRAME_ADD_MV_EB(candidate, ref_frame, ref_sign_bias,
                                    refmv_count, mv_ref_list, Done);
      }
    }
  }

  // Since we still don't have a candidate we'll try the last frame.
  if (prev_frame_mvs) {
    if (prev_frame_mvs->ref_frame[0] != ref_frame &&
        prev_frame_mvs->ref_frame[0] > INTRA_FRAME) {
      int_mv mv = prev_frame_mvs->mv[0];
      if (ref_sign_bias[prev_frame_mvs->ref_frame[0]] !=
          ref_sign_bias[ref_frame]) {
        mv.as_mv.row *= -1;
        mv.as_mv.col *= -1;
      }
      ADD_MV_REF_LIST_EB(mv, refmv_count, mv_ref_list, Done);
    }

    if (prev_frame_mvs->ref_frame[1] > INTRA_FRAME &&
        prev_frame_mvs->ref_frame[1] != ref_frame &&
        prev_frame_mvs->mv[1].as_int != prev_frame_mvs->mv[0].as_int) {
      int_mv mv = prev_frame_mvs->mv[1];
      if (ref_sign_bias[prev_frame_mvs->ref_frame[1]] !=
          ref_sign_bias[ref_frame]) {
        mv.as_mv.row *= -1;
        mv.as_mv.col *= -1;
      }
      ADD_MV_REF_LIST_EB(mv, refmv_count, mv_ref_list, Done);
    }
  }

  if (mode == NEARMV)
    refmv_count = MAX_MV_REF_CANDIDATES;
  else
    // we only care about the nearestmv for the remaining modes
    refmv_count = 1;

 Done:
  // Clamp vectors
  for (i = 0; i < refmv_count; ++i)
    clamp_mv_ref(&mv_ref_list[i].as_mv, xd);

  return refmv_count;
}

static void append_sub8x8_mvs_for_idx(VP9_COMMON *cm, MACROBLOCKD *xd,
                                      const POSITION *const mv_ref_search,
                                      PREDICTION_MODE b_mode, int block,
                                      int ref, int mi_row, int mi_col,
                                      int_mv *best_sub8x8) {
  int_mv mv_list[MAX_MV_REF_CANDIDATES];
  MODE_INFO *const mi = xd->mi[0];
  b_mode_info *bmi = mi->bmi;
  int n;
  int refmv_count;

  assert(MAX_MV_REF_CANDIDATES == 2);

  refmv_count = dec_find_mv_refs(cm, xd, b_mode, mi->ref_frame[ref],
                                 mv_ref_search, mv_list, mi_row, mi_col, block,
                                 1, NULL, NULL);

  switch (block) {
    case 0:
      best_sub8x8->as_int = mv_list[refmv_count - 1].as_int;
      break;
    case 1:
    case 2:
      if (b_mode == NEARESTMV) {
        best_sub8x8->as_int = bmi[0].as_mv[ref].as_int;
      } else {
        best_sub8x8->as_int = 0;
        for (n = 0; n < refmv_count; ++n)
          if (bmi[0].as_mv[ref].as_int != mv_list[n].as_int) {
            best_sub8x8->as_int = mv_list[n].as_int;
            break;
          }
      }
      break;
    case 3:
      if (b_mode == NEARESTMV) {
        best_sub8x8->as_int = bmi[2].as_mv[ref].as_int;
      } else {
        int_mv candidates[2 + MAX_MV_REF_CANDIDATES];
        candidates[0] = bmi[1].as_mv[ref];
        candidates[1] = bmi[0].as_mv[ref];
        candidates[2] = mv_list[0];
        candidates[3] = mv_list[1];
        best_sub8x8->as_int = 0;
        for (n = 0; n < 2 + MAX_MV_REF_CANDIDATES; ++n)
          if (bmi[2].as_mv[ref].as_int != candidates[n].as_int) {
            best_sub8x8->as_int = candidates[n].as_int;
            break;
          }
      }
      break;
    default:
      assert(0 && "Invalid block index.");
  }
}

static uint8_t get_mode_context(const VP9_COMMON *cm, const MACROBLOCKD *xd,
                                const POSITION *const mv_ref_search,
                                int mi_row, int mi_col) {
  int i;
  int context_counter = 0;
  const TileInfo *const tile = &xd->tile;

  // Get mode count from nearest 2 blocks
  for (i = 0; i < 2; ++i) {
    const POSITION *const mv_ref = &mv_ref_search[i];
    if (is_inside(tile, mi_col, mi_row, cm->mi_rows, mv_ref)) {
      const MODE_INFO *const candidate =
          xd->mi[mv_ref->col + mv_ref->row * xd->mi_stride];
      // Keep counts for entropy encoding.
      context_counter += mode_2_counter[candidate->mode];
    }
  }

  return counter_to_context[context_counter];
}

static void read_inter_block_mode_info(VP9Decoder *const pbi,
                                       MACROBLOCKD *const xd,
                                       MODE_INFO *const mi,
                                       int mi_row, int mi_col, vpx_reader *r) {
  VP9_COMMON *const cm = &pbi->common;
  const BLOCK_SIZE bsize = mi->sb_type;
  const int allow_hp = cm->allow_high_precision_mv;
  int_mv best_ref_mvs[2];
  int ref, is_compound;
  uint8_t inter_mode_ctx;
  const POSITION *const mv_ref_search = mv_ref_blocks[bsize];

  read_ref_frames(cm, xd, r, mi->segment_id, mi->ref_frame);
  is_compound = has_second_ref(mi);
  inter_mode_ctx = get_mode_context(cm, xd, mv_ref_search, mi_row, mi_col);

  if (segfeature_active(&cm->seg, mi->segment_id, SEG_LVL_SKIP)) {
    mi->mode = ZEROMV;
    if (bsize < BLOCK_8X8) {
        vpx_internal_error(xd->error_info, VPX_CODEC_UNSUP_BITSTREAM,
                           "Invalid usage of segement feature on small blocks");
        return;
    }
  } else {
    if (bsize >= BLOCK_8X8)
      mi->mode = read_inter_mode(cm, xd, r, inter_mode_ctx);
    else
      // Sub 8x8 blocks use the nearestmv as a ref_mv if the b_mode is NEWMV.
      // Setting mode to NEARESTMV forces the search to stop after the nearestmv
      // has been found. After b_modes have been read, mode will be overwritten
      // by the last b_mode.
      mi->mode = NEARESTMV;

    if (mi->mode != ZEROMV) {
      for (ref = 0; ref < 1 + is_compound; ++ref) {
        int_mv tmp_mvs[MAX_MV_REF_CANDIDATES];
        const MV_REFERENCE_FRAME frame = mi->ref_frame[ref];
        int refmv_count;

        refmv_count = dec_find_mv_refs(cm, xd, mi->mode, frame, mv_ref_search,
                                       tmp_mvs, mi_row, mi_col, -1, 0,
                                       fpm_sync, (void *)pbi);

        dec_find_best_ref_mvs(allow_hp, tmp_mvs, &best_ref_mvs[ref],
                              refmv_count);
      }
    }
  }

  mi->interp_filter = (cm->interp_filter == SWITCHABLE)
                      ? read_switchable_interp_filter(cm, xd, r)
                      : cm->interp_filter;

  if (bsize < BLOCK_8X8) {
    const int num_4x4_w = 1 << xd->bmode_blocks_wl;
    const int num_4x4_h = 1 << xd->bmode_blocks_hl;
    int idx, idy;
    PREDICTION_MODE b_mode;
    int_mv best_sub8x8[2];
    for (idy = 0; idy < 2; idy += num_4x4_h) {
      for (idx = 0; idx < 2; idx += num_4x4_w) {
        const int j = idy * 2 + idx;
        b_mode = read_inter_mode(cm, xd, r, inter_mode_ctx);

        if (b_mode == NEARESTMV || b_mode == NEARMV) {
          for (ref = 0; ref < 1 + is_compound; ++ref)
            append_sub8x8_mvs_for_idx(cm, xd, mv_ref_search, b_mode, j, ref,
                                      mi_row, mi_col, &best_sub8x8[ref]);
        }

        if (!assign_mv(cm, xd, b_mode, mi->bmi[j].as_mv, best_ref_mvs,
                       best_sub8x8, is_compound, allow_hp, r)) {
          xd->corrupted |= 1;
          break;
        }

        if (num_4x4_h == 2)
          mi->bmi[j + 2] = mi->bmi[j];
        if (num_4x4_w == 2)
          mi->bmi[j + 1] = mi->bmi[j];
      }
    }

    mi->mode = b_mode;

    copy_mv_pair(mi->mv, mi->bmi[3].as_mv);
  } else {
    xd->corrupted |= !assign_mv(cm, xd, mi->mode, mi->mv, best_ref_mvs,
                                best_ref_mvs, is_compound, allow_hp, r);
  }
}

static void read_inter_frame_mode_info(VP9Decoder *const pbi,
                                       MACROBLOCKD *const xd,
                                       int mi_row, int mi_col, vpx_reader *r,
                                       int x_mis, int y_mis) {
  VP9_COMMON *const cm = &pbi->common;
  MODE_INFO *const mi = xd->mi[0];
  int inter_block;

  mi->segment_id = read_inter_segment_id(cm, xd, mi_row, mi_col, r, x_mis,
                                         y_mis);
  mi->skip = read_skip(cm, xd, mi->segment_id, r);
  inter_block = read_is_inter_block(cm, xd, mi->segment_id, r);
  mi->tx_size = read_tx_size(cm, xd, !mi->skip || !inter_block, r);

  if (inter_block)
    read_inter_block_mode_info(pbi, xd, mi, mi_row, mi_col, r);
  else
    read_intra_block_mode_info(cm, xd, mi, r);
}

static INLINE void copy_ref_frame_pair(MV_REFERENCE_FRAME *dst,
                                       const MV_REFERENCE_FRAME *src) {
  memcpy(dst, src, sizeof(*dst) * 2);
}

void vp9_read_mode_info(VP9Decoder *const pbi, MACROBLOCKD *xd,
                        int mi_row, int mi_col, vpx_reader *r,
                        int x_mis, int y_mis) {
  VP9_COMMON *const cm = &pbi->common;
  MODE_INFO *const mi = xd->mi[0];
  MV_REF* frame_mvs = cm->cur_frame->mvs + mi_row * cm->mi_cols + mi_col;
  int w, h;

  if (frame_is_intra_only(cm)) {
    read_intra_frame_mode_info(cm, xd, mi_row, mi_col, r, x_mis, y_mis);
  } else {
    read_inter_frame_mode_info(pbi, xd, mi_row, mi_col, r, x_mis, y_mis);

    for (h = 0; h < y_mis; ++h) {
      for (w = 0; w < x_mis; ++w) {
        MV_REF *const mv = frame_mvs + w;
        copy_ref_frame_pair(mv->ref_frame, mi->ref_frame);
        copy_mv_pair(mv->mv, mi->mv);
      }
      frame_mvs += cm->mi_cols;
    }
  }
#if CONFIG_BETTER_HW_COMPATIBILITY && CONFIG_VP9_HIGHBITDEPTH
    if ((xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH) &&
        (xd->above_mi == NULL || xd->left_mi == NULL) &&
        !is_inter_block(mi) && need_top_left[mi->uv_mode])
      assert(0);
#endif  // CONFIG_BETTER_HW_COMPATIBILITY && CONFIG_VP9_HIGHBITDEPTH
}
