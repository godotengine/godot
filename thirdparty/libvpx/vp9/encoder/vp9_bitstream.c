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
#include <stdint.h>
#include <stdio.h>
#include <limits.h>

#include "vpx/vpx_encoder.h"
#include "vpx_dsp/bitwriter_buffer.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_mem/vpx_mem.h"
#include "vpx_ports/mem_ops.h"
#include "vpx_ports/system_state.h"
#if CONFIG_BITSTREAM_DEBUG
#include "vpx_util/vpx_debug_util.h"
#endif  // CONFIG_BITSTREAM_DEBUG

#include "vp9/common/vp9_entropy.h"
#include "vp9/common/vp9_entropymode.h"
#include "vp9/common/vp9_entropymv.h"
#include "vp9/common/vp9_mvref_common.h"
#include "vp9/common/vp9_pred_common.h"
#include "vp9/common/vp9_seg_common.h"
#include "vp9/common/vp9_tile_common.h"

#include "vp9/encoder/vp9_cost.h"
#include "vp9/encoder/vp9_bitstream.h"
#include "vp9/encoder/vp9_encodemv.h"
#include "vp9/encoder/vp9_mcomp.h"
#include "vp9/encoder/vp9_segmentation.h"
#include "vp9/encoder/vp9_subexp.h"
#include "vp9/encoder/vp9_tokenize.h"

static const struct vp9_token intra_mode_encodings[INTRA_MODES] = {
  { 0, 1 },  { 6, 3 },   { 28, 5 },  { 30, 5 }, { 58, 6 },
  { 59, 6 }, { 126, 7 }, { 127, 7 }, { 62, 6 }, { 2, 2 }
};
static const struct vp9_token
    switchable_interp_encodings[SWITCHABLE_FILTERS] = { { 0, 1 },
                                                        { 2, 2 },
                                                        { 3, 2 } };
static const struct vp9_token partition_encodings[PARTITION_TYPES] = {
  { 0, 1 }, { 2, 2 }, { 6, 3 }, { 7, 3 }
};
static const struct vp9_token inter_mode_encodings[INTER_MODES] = {
  { 2, 2 }, { 6, 3 }, { 0, 1 }, { 7, 3 }
};

static void write_intra_mode(vpx_writer *w, PREDICTION_MODE mode,
                             const vpx_prob *probs) {
  assert(!is_inter_mode(mode));
  vp9_write_token(w, vp9_intra_mode_tree, probs, &intra_mode_encodings[mode]);
}

static void write_inter_mode(vpx_writer *w, PREDICTION_MODE mode,
                             const vpx_prob *probs) {
  assert(is_inter_mode(mode));
  vp9_write_token(w, vp9_inter_mode_tree, probs,
                  &inter_mode_encodings[INTER_OFFSET(mode)]);
}

static void encode_unsigned_max(struct vpx_write_bit_buffer *wb, int data,
                                int max) {
  vpx_wb_write_literal(wb, data, get_unsigned_bits(max));
}

static void prob_diff_update(const vpx_tree_index *tree,
                             vpx_prob probs[/*n - 1*/],
                             const unsigned int counts[/*n - 1*/], int n,
                             vpx_writer *w) {
  int i;
  unsigned int branch_ct[32][2];

  // Assuming max number of probabilities <= 32
  assert(n <= 32);

  vp9_tree_probs_from_distribution(tree, branch_ct, counts);
  for (i = 0; i < n - 1; ++i)
    vp9_cond_prob_diff_update(w, &probs[i], branch_ct[i]);
}

static void write_selected_tx_size(const VP9_COMMON *cm,
                                   const MACROBLOCKD *const xd, vpx_writer *w) {
  TX_SIZE tx_size = xd->mi[0]->tx_size;
  BLOCK_SIZE bsize = xd->mi[0]->sb_type;
  const TX_SIZE max_tx_size = max_txsize_lookup[bsize];
  const vpx_prob *const tx_probs =
      get_tx_probs(max_tx_size, get_tx_size_context(xd), &cm->fc->tx_probs);
  vpx_write(w, tx_size != TX_4X4, tx_probs[0]);
  if (tx_size != TX_4X4 && max_tx_size >= TX_16X16) {
    vpx_write(w, tx_size != TX_8X8, tx_probs[1]);
    if (tx_size != TX_8X8 && max_tx_size >= TX_32X32)
      vpx_write(w, tx_size != TX_16X16, tx_probs[2]);
  }
}

static int write_skip(const VP9_COMMON *cm, const MACROBLOCKD *const xd,
                      int segment_id, const MODE_INFO *mi, vpx_writer *w) {
  if (segfeature_active(&cm->seg, segment_id, SEG_LVL_SKIP)) {
    return 1;
  } else {
    const int skip = mi->skip;
    vpx_write(w, skip, vp9_get_skip_prob(cm, xd));
    return skip;
  }
}

static void update_skip_probs(VP9_COMMON *cm, vpx_writer *w,
                              FRAME_COUNTS *counts) {
  int k;

  for (k = 0; k < SKIP_CONTEXTS; ++k)
    vp9_cond_prob_diff_update(w, &cm->fc->skip_probs[k], counts->skip[k]);
}

static void update_switchable_interp_probs(VP9_COMMON *cm, vpx_writer *w,
                                           FRAME_COUNTS *counts) {
  int j;
  for (j = 0; j < SWITCHABLE_FILTER_CONTEXTS; ++j)
    prob_diff_update(vp9_switchable_interp_tree,
                     cm->fc->switchable_interp_prob[j],
                     counts->switchable_interp[j], SWITCHABLE_FILTERS, w);
}

static void pack_mb_tokens(vpx_writer *w, TOKENEXTRA **tp,
                           const TOKENEXTRA *const stop,
                           vpx_bit_depth_t bit_depth) {
  const TOKENEXTRA *p;
  const vp9_extra_bit *const extra_bits =
#if CONFIG_VP9_HIGHBITDEPTH
      (bit_depth == VPX_BITS_12)   ? vp9_extra_bits_high12
      : (bit_depth == VPX_BITS_10) ? vp9_extra_bits_high10
                                   : vp9_extra_bits;
#else
      vp9_extra_bits;
  (void)bit_depth;
#endif  // CONFIG_VP9_HIGHBITDEPTH

  for (p = *tp; p < stop && p->token != EOSB_TOKEN; ++p) {
    if (p->token == EOB_TOKEN) {
      vpx_write(w, 0, p->context_tree[0]);
      continue;
    }
    vpx_write(w, 1, p->context_tree[0]);
    while (p->token == ZERO_TOKEN) {
      vpx_write(w, 0, p->context_tree[1]);
      ++p;
      if (p == stop || p->token == EOSB_TOKEN) {
        *tp = (TOKENEXTRA *)(uintptr_t)p + (p->token == EOSB_TOKEN);
        return;
      }
    }

    {
      const int t = p->token;
      const vpx_prob *const context_tree = p->context_tree;
      assert(t != ZERO_TOKEN);
      assert(t != EOB_TOKEN);
      assert(t != EOSB_TOKEN);
      vpx_write(w, 1, context_tree[1]);
      if (t == ONE_TOKEN) {
        vpx_write(w, 0, context_tree[2]);
        vpx_write_bit(w, p->extra & 1);
      } else {  // t >= TWO_TOKEN && t < EOB_TOKEN
        const struct vp9_token *const a = &vp9_coef_encodings[t];
        int v = a->value;
        int n = a->len;
        const int e = p->extra;
        vpx_write(w, 1, context_tree[2]);
        vp9_write_tree(w, vp9_coef_con_tree,
                       vp9_pareto8_full[context_tree[PIVOT_NODE] - 1], v,
                       n - UNCONSTRAINED_NODES, 0);
        if (t >= CATEGORY1_TOKEN) {
          const vp9_extra_bit *const b = &extra_bits[t];
          const unsigned char *pb = b->prob;
          v = e >> 1;
          n = b->len;  // number of bits in v, assumed nonzero
          do {
            const int bb = (v >> --n) & 1;
            vpx_write(w, bb, *pb++);
          } while (n);
        }
        vpx_write_bit(w, e & 1);
      }
    }
  }
  *tp = (TOKENEXTRA *)(uintptr_t)p + (p->token == EOSB_TOKEN);
}

static void write_segment_id(vpx_writer *w, const struct segmentation *seg,
                             int segment_id) {
  if (seg->enabled && seg->update_map)
    vp9_write_tree(w, vp9_segment_tree, seg->tree_probs, segment_id, 3, 0);
}

// This function encodes the reference frame
static void write_ref_frames(const VP9_COMMON *cm, const MACROBLOCKD *const xd,
                             vpx_writer *w) {
  const MODE_INFO *const mi = xd->mi[0];
  const int is_compound = has_second_ref(mi);
  const int segment_id = mi->segment_id;

  // If segment level coding of this signal is disabled...
  // or the segment allows multiple reference frame options
  if (segfeature_active(&cm->seg, segment_id, SEG_LVL_REF_FRAME)) {
    assert(!is_compound);
    assert(mi->ref_frame[0] ==
           get_segdata(&cm->seg, segment_id, SEG_LVL_REF_FRAME));
  } else {
    // does the feature use compound prediction or not
    // (if not specified at the frame/segment level)
    if (cm->reference_mode == REFERENCE_MODE_SELECT) {
      vpx_write(w, is_compound, vp9_get_reference_mode_prob(cm, xd));
    } else {
      assert((!is_compound) == (cm->reference_mode == SINGLE_REFERENCE));
    }

    if (is_compound) {
      const int idx = cm->ref_frame_sign_bias[cm->comp_fixed_ref];
      vpx_write(w, mi->ref_frame[!idx] == cm->comp_var_ref[1],
                vp9_get_pred_prob_comp_ref_p(cm, xd));
    } else {
      const int bit0 = mi->ref_frame[0] != LAST_FRAME;
      vpx_write(w, bit0, vp9_get_pred_prob_single_ref_p1(cm, xd));
      if (bit0) {
        const int bit1 = mi->ref_frame[0] != GOLDEN_FRAME;
        vpx_write(w, bit1, vp9_get_pred_prob_single_ref_p2(cm, xd));
      }
    }
  }
}

static void pack_inter_mode_mvs(VP9_COMP *cpi, const MACROBLOCKD *const xd,
                                const MB_MODE_INFO_EXT *const mbmi_ext,
                                vpx_writer *w,
                                unsigned int *const max_mv_magnitude,
                                int interp_filter_selected[][SWITCHABLE]) {
  VP9_COMMON *const cm = &cpi->common;
  const nmv_context *nmvc = &cm->fc->nmvc;
  const struct segmentation *const seg = &cm->seg;
  const MODE_INFO *const mi = xd->mi[0];
  const PREDICTION_MODE mode = mi->mode;
  const int segment_id = mi->segment_id;
  const BLOCK_SIZE bsize = mi->sb_type;
  const int allow_hp = cm->allow_high_precision_mv;
  const int is_inter = is_inter_block(mi);
  const int is_compound = has_second_ref(mi);
  int skip, ref;

  if (seg->update_map) {
    if (seg->temporal_update) {
      const int pred_flag = mi->seg_id_predicted;
      vpx_prob pred_prob = vp9_get_pred_prob_seg_id(seg, xd);
      vpx_write(w, pred_flag, pred_prob);
      if (!pred_flag) write_segment_id(w, seg, segment_id);
    } else {
      write_segment_id(w, seg, segment_id);
    }
  }

  skip = write_skip(cm, xd, segment_id, mi, w);

  if (!segfeature_active(seg, segment_id, SEG_LVL_REF_FRAME))
    vpx_write(w, is_inter, vp9_get_intra_inter_prob(cm, xd));

  if (bsize >= BLOCK_8X8 && cm->tx_mode == TX_MODE_SELECT &&
      !(is_inter && skip)) {
    write_selected_tx_size(cm, xd, w);
  }

  if (!is_inter) {
    if (bsize >= BLOCK_8X8) {
      write_intra_mode(w, mode, cm->fc->y_mode_prob[size_group_lookup[bsize]]);
    } else {
      int idx, idy;
      const int num_4x4_w = num_4x4_blocks_wide_lookup[bsize];
      const int num_4x4_h = num_4x4_blocks_high_lookup[bsize];
      for (idy = 0; idy < 2; idy += num_4x4_h) {
        for (idx = 0; idx < 2; idx += num_4x4_w) {
          const PREDICTION_MODE b_mode = mi->bmi[idy * 2 + idx].as_mode;
          write_intra_mode(w, b_mode, cm->fc->y_mode_prob[0]);
        }
      }
    }
    write_intra_mode(w, mi->uv_mode, cm->fc->uv_mode_prob[mode]);
  } else {
    const int mode_ctx = mbmi_ext->mode_context[mi->ref_frame[0]];
    const vpx_prob *const inter_probs = cm->fc->inter_mode_probs[mode_ctx];
    write_ref_frames(cm, xd, w);

    // If segment skip is not enabled code the mode.
    if (!segfeature_active(seg, segment_id, SEG_LVL_SKIP)) {
      if (bsize >= BLOCK_8X8) {
        write_inter_mode(w, mode, inter_probs);
      }
    }

    if (cm->interp_filter == SWITCHABLE) {
      const int ctx = get_pred_context_switchable_interp(xd);
      vp9_write_token(w, vp9_switchable_interp_tree,
                      cm->fc->switchable_interp_prob[ctx],
                      &switchable_interp_encodings[mi->interp_filter]);
      ++interp_filter_selected[0][mi->interp_filter];
    } else {
      assert(mi->interp_filter == cm->interp_filter);
    }

    if (bsize < BLOCK_8X8) {
      const int num_4x4_w = num_4x4_blocks_wide_lookup[bsize];
      const int num_4x4_h = num_4x4_blocks_high_lookup[bsize];
      int idx, idy;
      for (idy = 0; idy < 2; idy += num_4x4_h) {
        for (idx = 0; idx < 2; idx += num_4x4_w) {
          const int j = idy * 2 + idx;
          const PREDICTION_MODE b_mode = mi->bmi[j].as_mode;
          write_inter_mode(w, b_mode, inter_probs);
          if (b_mode == NEWMV) {
            for (ref = 0; ref < 1 + is_compound; ++ref)
              vp9_encode_mv(cpi, w, &mi->bmi[j].as_mv[ref].as_mv,
                            &mbmi_ext->ref_mvs[mi->ref_frame[ref]][0].as_mv,
                            nmvc, allow_hp, max_mv_magnitude);
          }
        }
      }
    } else {
      if (mode == NEWMV) {
        for (ref = 0; ref < 1 + is_compound; ++ref)
          vp9_encode_mv(cpi, w, &mi->mv[ref].as_mv,
                        &mbmi_ext->ref_mvs[mi->ref_frame[ref]][0].as_mv, nmvc,
                        allow_hp, max_mv_magnitude);
      }
    }
  }
}

static void write_mb_modes_kf(const VP9_COMMON *cm, const MACROBLOCKD *xd,
                              vpx_writer *w) {
  const struct segmentation *const seg = &cm->seg;
  const MODE_INFO *const mi = xd->mi[0];
  const MODE_INFO *const above_mi = xd->above_mi;
  const MODE_INFO *const left_mi = xd->left_mi;
  const BLOCK_SIZE bsize = mi->sb_type;

  if (seg->update_map) write_segment_id(w, seg, mi->segment_id);

  write_skip(cm, xd, mi->segment_id, mi, w);

  if (bsize >= BLOCK_8X8 && cm->tx_mode == TX_MODE_SELECT)
    write_selected_tx_size(cm, xd, w);

  if (bsize >= BLOCK_8X8) {
    write_intra_mode(w, mi->mode, get_y_mode_probs(mi, above_mi, left_mi, 0));
  } else {
    const int num_4x4_w = num_4x4_blocks_wide_lookup[bsize];
    const int num_4x4_h = num_4x4_blocks_high_lookup[bsize];
    int idx, idy;

    for (idy = 0; idy < 2; idy += num_4x4_h) {
      for (idx = 0; idx < 2; idx += num_4x4_w) {
        const int block = idy * 2 + idx;
        write_intra_mode(w, mi->bmi[block].as_mode,
                         get_y_mode_probs(mi, above_mi, left_mi, block));
      }
    }
  }

  write_intra_mode(w, mi->uv_mode, vp9_kf_uv_mode_prob[mi->mode]);
}

static void write_modes_b(VP9_COMP *cpi, MACROBLOCKD *const xd,
                          const TileInfo *const tile, vpx_writer *w,
                          TOKENEXTRA **tok, const TOKENEXTRA *const tok_end,
                          int mi_row, int mi_col,
                          unsigned int *const max_mv_magnitude,
                          int interp_filter_selected[][SWITCHABLE]) {
  const VP9_COMMON *const cm = &cpi->common;
  const MB_MODE_INFO_EXT *const mbmi_ext =
      cpi->td.mb.mbmi_ext_base + (mi_row * cm->mi_cols + mi_col);
  MODE_INFO *m;

  xd->mi = cm->mi_grid_visible + (mi_row * cm->mi_stride + mi_col);
  m = xd->mi[0];

  set_mi_row_col(xd, tile, mi_row, num_8x8_blocks_high_lookup[m->sb_type],
                 mi_col, num_8x8_blocks_wide_lookup[m->sb_type], cm->mi_rows,
                 cm->mi_cols);
  if (frame_is_intra_only(cm)) {
    write_mb_modes_kf(cm, xd, w);
  } else {
    pack_inter_mode_mvs(cpi, xd, mbmi_ext, w, max_mv_magnitude,
                        interp_filter_selected);
  }

  assert(*tok < tok_end);
  pack_mb_tokens(w, tok, tok_end, cm->bit_depth);
}

static void write_partition(const VP9_COMMON *const cm,
                            const MACROBLOCKD *const xd, int hbs, int mi_row,
                            int mi_col, PARTITION_TYPE p, BLOCK_SIZE bsize,
                            vpx_writer *w) {
  const int ctx = partition_plane_context(xd, mi_row, mi_col, bsize);
  const vpx_prob *const probs = xd->partition_probs[ctx];
  const int has_rows = (mi_row + hbs) < cm->mi_rows;
  const int has_cols = (mi_col + hbs) < cm->mi_cols;

  if (has_rows && has_cols) {
    vp9_write_token(w, vp9_partition_tree, probs, &partition_encodings[p]);
  } else if (!has_rows && has_cols) {
    assert(p == PARTITION_SPLIT || p == PARTITION_HORZ);
    vpx_write(w, p == PARTITION_SPLIT, probs[1]);
  } else if (has_rows && !has_cols) {
    assert(p == PARTITION_SPLIT || p == PARTITION_VERT);
    vpx_write(w, p == PARTITION_SPLIT, probs[2]);
  } else {
    assert(p == PARTITION_SPLIT);
  }
}

static void write_modes_sb(VP9_COMP *cpi, MACROBLOCKD *const xd,
                           const TileInfo *const tile, vpx_writer *w,
                           TOKENEXTRA **tok, const TOKENEXTRA *const tok_end,
                           int mi_row, int mi_col, BLOCK_SIZE bsize,
                           unsigned int *const max_mv_magnitude,
                           int interp_filter_selected[][SWITCHABLE]) {
  const VP9_COMMON *const cm = &cpi->common;
  const int bsl = b_width_log2_lookup[bsize];
  const int bs = (1 << bsl) / 4;
  PARTITION_TYPE partition;
  BLOCK_SIZE subsize;
  const MODE_INFO *m = NULL;

  if (mi_row >= cm->mi_rows || mi_col >= cm->mi_cols) return;

  m = cm->mi_grid_visible[mi_row * cm->mi_stride + mi_col];

  partition = partition_lookup[bsl][m->sb_type];
  write_partition(cm, xd, bs, mi_row, mi_col, partition, bsize, w);
  subsize = get_subsize(bsize, partition);
  if (subsize < BLOCK_8X8) {
    write_modes_b(cpi, xd, tile, w, tok, tok_end, mi_row, mi_col,
                  max_mv_magnitude, interp_filter_selected);
  } else {
    switch (partition) {
      case PARTITION_NONE:
        write_modes_b(cpi, xd, tile, w, tok, tok_end, mi_row, mi_col,
                      max_mv_magnitude, interp_filter_selected);
        break;
      case PARTITION_HORZ:
        write_modes_b(cpi, xd, tile, w, tok, tok_end, mi_row, mi_col,
                      max_mv_magnitude, interp_filter_selected);
        if (mi_row + bs < cm->mi_rows)
          write_modes_b(cpi, xd, tile, w, tok, tok_end, mi_row + bs, mi_col,
                        max_mv_magnitude, interp_filter_selected);
        break;
      case PARTITION_VERT:
        write_modes_b(cpi, xd, tile, w, tok, tok_end, mi_row, mi_col,
                      max_mv_magnitude, interp_filter_selected);
        if (mi_col + bs < cm->mi_cols)
          write_modes_b(cpi, xd, tile, w, tok, tok_end, mi_row, mi_col + bs,
                        max_mv_magnitude, interp_filter_selected);
        break;
      default:
        assert(partition == PARTITION_SPLIT);
        write_modes_sb(cpi, xd, tile, w, tok, tok_end, mi_row, mi_col, subsize,
                       max_mv_magnitude, interp_filter_selected);
        write_modes_sb(cpi, xd, tile, w, tok, tok_end, mi_row, mi_col + bs,
                       subsize, max_mv_magnitude, interp_filter_selected);
        write_modes_sb(cpi, xd, tile, w, tok, tok_end, mi_row + bs, mi_col,
                       subsize, max_mv_magnitude, interp_filter_selected);
        write_modes_sb(cpi, xd, tile, w, tok, tok_end, mi_row + bs, mi_col + bs,
                       subsize, max_mv_magnitude, interp_filter_selected);
        break;
    }
  }

  // update partition context
  if (bsize >= BLOCK_8X8 &&
      (bsize == BLOCK_8X8 || partition != PARTITION_SPLIT))
    update_partition_context(xd, mi_row, mi_col, subsize, bsize);
}

static void write_modes(VP9_COMP *cpi, MACROBLOCKD *const xd,
                        const TileInfo *const tile, vpx_writer *w, int tile_row,
                        int tile_col, unsigned int *const max_mv_magnitude,
                        int interp_filter_selected[][SWITCHABLE]) {
  const VP9_COMMON *const cm = &cpi->common;
  int mi_row, mi_col, tile_sb_row;
  TOKENEXTRA *tok = NULL;
  TOKENEXTRA *tok_end = NULL;

  set_partition_probs(cm, xd);

  for (mi_row = tile->mi_row_start; mi_row < tile->mi_row_end;
       mi_row += MI_BLOCK_SIZE) {
    tile_sb_row = mi_cols_aligned_to_sb(mi_row - tile->mi_row_start) >>
                  MI_BLOCK_SIZE_LOG2;
    tok = cpi->tplist[tile_row][tile_col][tile_sb_row].start;
    tok_end = tok + cpi->tplist[tile_row][tile_col][tile_sb_row].count;

    vp9_zero(xd->left_seg_context);
    for (mi_col = tile->mi_col_start; mi_col < tile->mi_col_end;
         mi_col += MI_BLOCK_SIZE)
      write_modes_sb(cpi, xd, tile, w, &tok, tok_end, mi_row, mi_col,
                     BLOCK_64X64, max_mv_magnitude, interp_filter_selected);

    assert(tok == cpi->tplist[tile_row][tile_col][tile_sb_row].stop);
  }
}

static void build_tree_distribution(VP9_COMP *cpi, TX_SIZE tx_size,
                                    vp9_coeff_stats *coef_branch_ct,
                                    vp9_coeff_probs_model *coef_probs) {
  vp9_coeff_count *coef_counts = cpi->td.rd_counts.coef_counts[tx_size];
  unsigned int(*eob_branch_ct)[REF_TYPES][COEF_BANDS][COEFF_CONTEXTS] =
      cpi->common.counts.eob_branch[tx_size];
  int i, j, k, l, m;

  for (i = 0; i < PLANE_TYPES; ++i) {
    for (j = 0; j < REF_TYPES; ++j) {
      for (k = 0; k < COEF_BANDS; ++k) {
        for (l = 0; l < BAND_COEFF_CONTEXTS(k); ++l) {
          vp9_tree_probs_from_distribution(vp9_coef_tree,
                                           coef_branch_ct[i][j][k][l],
                                           coef_counts[i][j][k][l]);
          coef_branch_ct[i][j][k][l][0][1] =
              eob_branch_ct[i][j][k][l] - coef_branch_ct[i][j][k][l][0][0];
          for (m = 0; m < UNCONSTRAINED_NODES; ++m)
            coef_probs[i][j][k][l][m] =
                get_binary_prob(coef_branch_ct[i][j][k][l][m][0],
                                coef_branch_ct[i][j][k][l][m][1]);
        }
      }
    }
  }
}

static void update_coef_probs_common(vpx_writer *const bc, VP9_COMP *cpi,
                                     TX_SIZE tx_size,
                                     vp9_coeff_stats *frame_branch_ct,
                                     vp9_coeff_probs_model *new_coef_probs) {
  vp9_coeff_probs_model *old_coef_probs = cpi->common.fc->coef_probs[tx_size];
  const vpx_prob upd = DIFF_UPDATE_PROB;
  const int entropy_nodes_update = UNCONSTRAINED_NODES;
  int i, j, k, l, t;
  int stepsize = cpi->sf.coeff_prob_appx_step;

  switch (cpi->sf.use_fast_coef_updates) {
    case TWO_LOOP: {
      /* dry run to see if there is any update at all needed */
      int64_t savings = 0;
      int update[2] = { 0, 0 };
      for (i = 0; i < PLANE_TYPES; ++i) {
        for (j = 0; j < REF_TYPES; ++j) {
          for (k = 0; k < COEF_BANDS; ++k) {
            for (l = 0; l < BAND_COEFF_CONTEXTS(k); ++l) {
              for (t = 0; t < entropy_nodes_update; ++t) {
                vpx_prob newp = new_coef_probs[i][j][k][l][t];
                const vpx_prob oldp = old_coef_probs[i][j][k][l][t];
                int64_t s;
                int u = 0;
                if (t == PIVOT_NODE)
                  s = vp9_prob_diff_update_savings_search_model(
                      frame_branch_ct[i][j][k][l][0], oldp, &newp, upd,
                      stepsize);
                else
                  s = vp9_prob_diff_update_savings_search(
                      frame_branch_ct[i][j][k][l][t], oldp, &newp, upd);
                if (s > 0 && newp != oldp) u = 1;
                if (u)
                  savings += s - (int)(vp9_cost_zero(upd));
                else
                  savings -= (int)(vp9_cost_zero(upd));
                update[u]++;
              }
            }
          }
        }
      }

      // printf("Update %d %d, savings %d\n", update[0], update[1], savings);
      /* Is coef updated at all */
      if (update[1] == 0 || savings < 0) {
        vpx_write_bit(bc, 0);
        return;
      }
      vpx_write_bit(bc, 1);
      for (i = 0; i < PLANE_TYPES; ++i) {
        for (j = 0; j < REF_TYPES; ++j) {
          for (k = 0; k < COEF_BANDS; ++k) {
            for (l = 0; l < BAND_COEFF_CONTEXTS(k); ++l) {
              // calc probs and branch cts for this frame only
              for (t = 0; t < entropy_nodes_update; ++t) {
                vpx_prob newp = new_coef_probs[i][j][k][l][t];
                vpx_prob *oldp = old_coef_probs[i][j][k][l] + t;
                int64_t s;
                int u = 0;
                if (t == PIVOT_NODE)
                  s = vp9_prob_diff_update_savings_search_model(
                      frame_branch_ct[i][j][k][l][0], *oldp, &newp, upd,
                      stepsize);
                else
                  s = vp9_prob_diff_update_savings_search(
                      frame_branch_ct[i][j][k][l][t], *oldp, &newp, upd);
                if (s > 0 && newp != *oldp) u = 1;
                vpx_write(bc, u, upd);
                if (u) {
                  /* send/use new probability */
                  vp9_write_prob_diff_update(bc, newp, *oldp);
                  *oldp = newp;
                }
              }
            }
          }
        }
      }
      return;
    }

    default: {
      int updates = 0;
      int noupdates_before_first = 0;
      assert(cpi->sf.use_fast_coef_updates == ONE_LOOP_REDUCED);
      for (i = 0; i < PLANE_TYPES; ++i) {
        for (j = 0; j < REF_TYPES; ++j) {
          for (k = 0; k < COEF_BANDS; ++k) {
            for (l = 0; l < BAND_COEFF_CONTEXTS(k); ++l) {
              // calc probs and branch cts for this frame only
              for (t = 0; t < entropy_nodes_update; ++t) {
                vpx_prob newp = new_coef_probs[i][j][k][l][t];
                vpx_prob *oldp = old_coef_probs[i][j][k][l] + t;
                int64_t s;
                int u = 0;

                if (t == PIVOT_NODE) {
                  s = vp9_prob_diff_update_savings_search_model(
                      frame_branch_ct[i][j][k][l][0], *oldp, &newp, upd,
                      stepsize);
                } else {
                  s = vp9_prob_diff_update_savings_search(
                      frame_branch_ct[i][j][k][l][t], *oldp, &newp, upd);
                }

                if (s > 0 && newp != *oldp) u = 1;
                updates += u;
                if (u == 0 && updates == 0) {
                  noupdates_before_first++;
                  continue;
                }
                if (u == 1 && updates == 1) {
                  int v;
                  // first update
                  vpx_write_bit(bc, 1);
                  for (v = 0; v < noupdates_before_first; ++v)
                    vpx_write(bc, 0, upd);
                }
                vpx_write(bc, u, upd);
                if (u) {
                  /* send/use new probability */
                  vp9_write_prob_diff_update(bc, newp, *oldp);
                  *oldp = newp;
                }
              }
            }
          }
        }
      }
      if (updates == 0) {
        vpx_write_bit(bc, 0);  // no updates
      }
      return;
    }
  }
}

static void update_coef_probs(VP9_COMP *cpi, vpx_writer *w) {
  const TX_MODE tx_mode = cpi->common.tx_mode;
  const TX_SIZE max_tx_size = tx_mode_to_biggest_tx_size[tx_mode];
  TX_SIZE tx_size;
  for (tx_size = TX_4X4; tx_size <= max_tx_size; ++tx_size) {
    vp9_coeff_stats frame_branch_ct[PLANE_TYPES];
    vp9_coeff_probs_model frame_coef_probs[PLANE_TYPES];
    if (cpi->td.counts->tx.tx_totals[tx_size] <= 20 ||
        (tx_size >= TX_16X16 && cpi->sf.tx_size_search_method == USE_TX_8X8)) {
      vpx_write_bit(w, 0);
    } else {
      build_tree_distribution(cpi, tx_size, frame_branch_ct, frame_coef_probs);
      update_coef_probs_common(w, cpi, tx_size, frame_branch_ct,
                               frame_coef_probs);
    }
  }
}

static void encode_loopfilter(struct loopfilter *lf,
                              struct vpx_write_bit_buffer *wb) {
  int i;

  // Encode the loop filter level and type
  vpx_wb_write_literal(wb, lf->filter_level, 6);
  vpx_wb_write_literal(wb, lf->sharpness_level, 3);

  // Write out loop filter deltas applied at the MB level based on mode or
  // ref frame (if they are enabled).
  vpx_wb_write_bit(wb, lf->mode_ref_delta_enabled);

  if (lf->mode_ref_delta_enabled) {
    vpx_wb_write_bit(wb, lf->mode_ref_delta_update);
    if (lf->mode_ref_delta_update) {
      for (i = 0; i < MAX_REF_LF_DELTAS; i++) {
        const int delta = lf->ref_deltas[i];
        const int changed = delta != lf->last_ref_deltas[i];
        vpx_wb_write_bit(wb, changed);
        if (changed) {
          lf->last_ref_deltas[i] = delta;
          vpx_wb_write_literal(wb, abs(delta) & 0x3F, 6);
          vpx_wb_write_bit(wb, delta < 0);
        }
      }

      for (i = 0; i < MAX_MODE_LF_DELTAS; i++) {
        const int delta = lf->mode_deltas[i];
        const int changed = delta != lf->last_mode_deltas[i];
        vpx_wb_write_bit(wb, changed);
        if (changed) {
          lf->last_mode_deltas[i] = delta;
          vpx_wb_write_literal(wb, abs(delta) & 0x3F, 6);
          vpx_wb_write_bit(wb, delta < 0);
        }
      }
    }
  }
}

static void write_delta_q(struct vpx_write_bit_buffer *wb, int delta_q) {
  if (delta_q != 0) {
    vpx_wb_write_bit(wb, 1);
    vpx_wb_write_literal(wb, abs(delta_q), 4);
    vpx_wb_write_bit(wb, delta_q < 0);
  } else {
    vpx_wb_write_bit(wb, 0);
  }
}

static void encode_quantization(const VP9_COMMON *const cm,
                                struct vpx_write_bit_buffer *wb) {
  vpx_wb_write_literal(wb, cm->base_qindex, QINDEX_BITS);
  write_delta_q(wb, cm->y_dc_delta_q);
  write_delta_q(wb, cm->uv_dc_delta_q);
  write_delta_q(wb, cm->uv_ac_delta_q);
}

static void encode_segmentation(VP9_COMMON *cm, MACROBLOCKD *xd,
                                struct vpx_write_bit_buffer *wb) {
  int i, j;

  const struct segmentation *seg = &cm->seg;

  vpx_wb_write_bit(wb, seg->enabled);
  if (!seg->enabled) return;

  // Segmentation map
  vpx_wb_write_bit(wb, seg->update_map);
  if (seg->update_map) {
    // Select the coding strategy (temporal or spatial)
    vp9_choose_segmap_coding_method(cm, xd);
    // Write out probabilities used to decode unpredicted  macro-block segments
    for (i = 0; i < SEG_TREE_PROBS; i++) {
      const int prob = seg->tree_probs[i];
      const int update = prob != MAX_PROB;
      vpx_wb_write_bit(wb, update);
      if (update) vpx_wb_write_literal(wb, prob, 8);
    }

    // Write out the chosen coding method.
    vpx_wb_write_bit(wb, seg->temporal_update);
    if (seg->temporal_update) {
      for (i = 0; i < PREDICTION_PROBS; i++) {
        const int prob = seg->pred_probs[i];
        const int update = prob != MAX_PROB;
        vpx_wb_write_bit(wb, update);
        if (update) vpx_wb_write_literal(wb, prob, 8);
      }
    }
  }

  // Segmentation data
  vpx_wb_write_bit(wb, seg->update_data);
  if (seg->update_data) {
    vpx_wb_write_bit(wb, seg->abs_delta);

    for (i = 0; i < MAX_SEGMENTS; i++) {
      for (j = 0; j < SEG_LVL_MAX; j++) {
        const int active = segfeature_active(seg, i, j);
        vpx_wb_write_bit(wb, active);
        if (active) {
          const int data = get_segdata(seg, i, j);
          const int data_max = vp9_seg_feature_data_max(j);

          if (vp9_is_segfeature_signed(j)) {
            encode_unsigned_max(wb, abs(data), data_max);
            vpx_wb_write_bit(wb, data < 0);
          } else {
            encode_unsigned_max(wb, data, data_max);
          }
        }
      }
    }
  }
}

static void encode_txfm_probs(VP9_COMMON *cm, vpx_writer *w,
                              FRAME_COUNTS *counts) {
  // Mode
  vpx_write_literal(w, VPXMIN(cm->tx_mode, ALLOW_32X32), 2);
  if (cm->tx_mode >= ALLOW_32X32)
    vpx_write_bit(w, cm->tx_mode == TX_MODE_SELECT);

  // Probabilities
  if (cm->tx_mode == TX_MODE_SELECT) {
    int i, j;
    unsigned int ct_8x8p[TX_SIZES - 3][2];
    unsigned int ct_16x16p[TX_SIZES - 2][2];
    unsigned int ct_32x32p[TX_SIZES - 1][2];

    for (i = 0; i < TX_SIZE_CONTEXTS; i++) {
      tx_counts_to_branch_counts_8x8(counts->tx.p8x8[i], ct_8x8p);
      for (j = 0; j < TX_SIZES - 3; j++)
        vp9_cond_prob_diff_update(w, &cm->fc->tx_probs.p8x8[i][j], ct_8x8p[j]);
    }

    for (i = 0; i < TX_SIZE_CONTEXTS; i++) {
      tx_counts_to_branch_counts_16x16(counts->tx.p16x16[i], ct_16x16p);
      for (j = 0; j < TX_SIZES - 2; j++)
        vp9_cond_prob_diff_update(w, &cm->fc->tx_probs.p16x16[i][j],
                                  ct_16x16p[j]);
    }

    for (i = 0; i < TX_SIZE_CONTEXTS; i++) {
      tx_counts_to_branch_counts_32x32(counts->tx.p32x32[i], ct_32x32p);
      for (j = 0; j < TX_SIZES - 1; j++)
        vp9_cond_prob_diff_update(w, &cm->fc->tx_probs.p32x32[i][j],
                                  ct_32x32p[j]);
    }
  }
}

static void write_interp_filter(INTERP_FILTER filter,
                                struct vpx_write_bit_buffer *wb) {
  const int filter_to_literal[] = { 1, 0, 2, 3 };

  vpx_wb_write_bit(wb, filter == SWITCHABLE);
  if (filter != SWITCHABLE)
    vpx_wb_write_literal(wb, filter_to_literal[filter], 2);
}

static void fix_interp_filter(VP9_COMMON *cm, FRAME_COUNTS *counts) {
  if (cm->interp_filter == SWITCHABLE) {
    // Check to see if only one of the filters is actually used
    int count[SWITCHABLE_FILTERS];
    int i, j, c = 0;
    for (i = 0; i < SWITCHABLE_FILTERS; ++i) {
      count[i] = 0;
      for (j = 0; j < SWITCHABLE_FILTER_CONTEXTS; ++j)
        count[i] += counts->switchable_interp[j][i];
      c += (count[i] > 0);
    }
    if (c == 1) {
      // Only one filter is used. So set the filter at frame level
      for (i = 0; i < SWITCHABLE_FILTERS; ++i) {
        if (count[i]) {
          cm->interp_filter = i;
          break;
        }
      }
    }
  }
}

static void write_tile_info(const VP9_COMMON *const cm,
                            struct vpx_write_bit_buffer *wb) {
  int min_log2_tile_cols, max_log2_tile_cols, ones;
  vp9_get_tile_n_bits(cm->mi_cols, &min_log2_tile_cols, &max_log2_tile_cols);

  // columns
  ones = cm->log2_tile_cols - min_log2_tile_cols;
  while (ones--) vpx_wb_write_bit(wb, 1);

  if (cm->log2_tile_cols < max_log2_tile_cols) vpx_wb_write_bit(wb, 0);

  // rows
  vpx_wb_write_bit(wb, cm->log2_tile_rows != 0);
  if (cm->log2_tile_rows != 0) vpx_wb_write_bit(wb, cm->log2_tile_rows != 1);
}

int vp9_get_refresh_mask(VP9_COMP *cpi) {
  if (cpi->ext_ratectrl.ready &&
      (cpi->ext_ratectrl.funcs.rc_type & VPX_RC_GOP) != 0 &&
      cpi->ext_ratectrl.funcs.get_gop_decision != NULL) {
    GF_GROUP *const gf_group = &cpi->twopass.gf_group;
    const int this_gf_index = gf_group->index;
    const int update_ref_idx = gf_group->update_ref_idx[this_gf_index];

    if (update_ref_idx != INVALID_IDX) {
      return (1 << update_ref_idx);
    } else {
      return 0;
    }
  }
  if (vp9_preserve_existing_gf(cpi)) {
    // We have decided to preserve the previously existing golden frame as our
    // new ARF frame. However, in the short term we leave it in the GF slot and,
    // if we're updating the GF with the current decoded frame, we save it
    // instead to the ARF slot.
    // Later, in the function vp9_encoder.c:vp9_update_reference_frames() we
    // will swap gld_fb_idx and alt_fb_idx to achieve our objective. We do it
    // there so that it can be done outside of the recode loop.
    // Note: This is highly specific to the use of ARF as a forward reference,
    // and this needs to be generalized as other uses are implemented
    // (like RTC/temporal scalability).
    return (cpi->refresh_last_frame << cpi->lst_fb_idx) |
           (cpi->refresh_golden_frame << cpi->alt_fb_idx);
  } else {
    int arf_idx = cpi->alt_fb_idx;
    GF_GROUP *const gf_group = &cpi->twopass.gf_group;

    if (cpi->multi_layer_arf) {
      for (arf_idx = 0; arf_idx < REF_FRAMES; ++arf_idx) {
        if (arf_idx != cpi->alt_fb_idx && arf_idx != cpi->lst_fb_idx &&
            arf_idx != cpi->gld_fb_idx) {
          int idx;
          for (idx = 0; idx < gf_group->stack_size; ++idx)
            if (arf_idx == gf_group->arf_index_stack[idx]) break;
          if (idx == gf_group->stack_size) break;
        }
      }
    }
    cpi->twopass.gf_group.top_arf_idx = arf_idx;

    if (cpi->use_svc && cpi->svc.use_set_ref_frame_config &&
        cpi->svc.temporal_layering_mode == VP9E_TEMPORAL_LAYERING_MODE_BYPASS)
      return cpi->svc.update_buffer_slot[cpi->svc.spatial_layer_id];
    return (cpi->refresh_last_frame << cpi->lst_fb_idx) |
           (cpi->refresh_golden_frame << cpi->gld_fb_idx) |
           (cpi->refresh_alt_ref_frame << arf_idx);
  }
}

static int encode_tile_worker(void *arg1, void *arg2) {
  VP9_COMP *cpi = (VP9_COMP *)arg1;
  VP9BitstreamWorkerData *data = (VP9BitstreamWorkerData *)arg2;
  MACROBLOCKD *const xd = &data->xd;
  const int tile_row = 0;
  vpx_start_encode(&data->bit_writer, data->dest, data->dest_size);
  write_modes(cpi, xd, &cpi->tile_data[data->tile_idx].tile_info,
              &data->bit_writer, tile_row, data->tile_idx,
              &data->max_mv_magnitude, data->interp_filter_selected);
  return vpx_stop_encode(&data->bit_writer) == 0;
}

void vp9_bitstream_encode_tiles_buffer_dealloc(VP9_COMP *const cpi) {
  if (cpi->vp9_bitstream_worker_data) {
    int i;
    for (i = 1; i < cpi->num_workers; ++i) {
      vpx_free(cpi->vp9_bitstream_worker_data[i].dest);
    }
    vpx_free(cpi->vp9_bitstream_worker_data);
    cpi->vp9_bitstream_worker_data = NULL;
  }
}

static size_t encode_tiles_buffer_alloc_size(const VP9_COMP *cpi) {
  const VP9_COMMON *cm = &cpi->common;
  const int image_bps =
      (8 + 2 * (8 >> (cm->subsampling_x + cm->subsampling_y))) *
      (1 + (cm->bit_depth > 8));
  const int64_t size =
      (int64_t)cpi->oxcf.width * cpi->oxcf.height * image_bps / 8;
  return (size_t)size;
}

static void encode_tiles_buffer_alloc(VP9_COMP *const cpi,
                                      size_t buffer_alloc_size) {
  VP9_COMMON *const cm = &cpi->common;
  int i;
  const size_t worker_data_size =
      cpi->num_workers * sizeof(*cpi->vp9_bitstream_worker_data);
  CHECK_MEM_ERROR(&cm->error, cpi->vp9_bitstream_worker_data,
                  vpx_memalign(16, worker_data_size));
  memset(cpi->vp9_bitstream_worker_data, 0, worker_data_size);
  for (i = 1; i < cpi->num_workers; ++i) {
    CHECK_MEM_ERROR(&cm->error, cpi->vp9_bitstream_worker_data[i].dest,
                    vpx_malloc(buffer_alloc_size));
    cpi->vp9_bitstream_worker_data[i].dest_size = buffer_alloc_size;
  }
}

static size_t encode_tiles_mt(VP9_COMP *cpi, uint8_t *data_ptr,
                              size_t data_size) {
  const VPxWorkerInterface *const winterface = vpx_get_worker_interface();
  VP9_COMMON *const cm = &cpi->common;
  const int tile_cols = 1 << cm->log2_tile_cols;
  const int num_workers = cpi->num_workers;
  size_t total_size = 0;
  int tile_col = 0;
  int error = 0;

  const size_t buffer_alloc_size = encode_tiles_buffer_alloc_size(cpi);
  if (!cpi->vp9_bitstream_worker_data ||
      cpi->vp9_bitstream_worker_data[1].dest_size != buffer_alloc_size) {
    vp9_bitstream_encode_tiles_buffer_dealloc(cpi);
    encode_tiles_buffer_alloc(cpi, buffer_alloc_size);
  }

  while (tile_col < tile_cols) {
    int i, j;
    for (i = 0; i < num_workers && tile_col < tile_cols; ++i) {
      VPxWorker *const worker = &cpi->workers[i];
      VP9BitstreamWorkerData *const data = &cpi->vp9_bitstream_worker_data[i];

      // Populate the worker data.
      data->xd = cpi->td.mb.e_mbd;
      data->tile_idx = tile_col;
      data->max_mv_magnitude = cpi->max_mv_magnitude;
      memset(data->interp_filter_selected, 0,
             sizeof(data->interp_filter_selected[0][0]) * SWITCHABLE);

      // First thread can directly write into the output buffer.
      if (i == 0) {
        // If this worker happens to be for the last tile, then do not offset it
        // by 4 for the tile size.
        const size_t offset = total_size + (tile_col == tile_cols - 1 ? 0 : 4);
        if (data_size < offset) {
          vpx_internal_error(&cm->error, VPX_CODEC_ERROR,
                             "encode_tiles_mt: output buffer full");
        }
        data->dest = data_ptr + offset;
        data->dest_size = data_size - offset;
      }
      worker->data1 = cpi;
      worker->data2 = data;
      worker->hook = encode_tile_worker;
      worker->had_error = 0;

      if (i < num_workers - 1) {
        winterface->launch(worker);
      } else {
        winterface->execute(worker);
      }
      ++tile_col;
    }
    for (j = 0; j < i; ++j) {
      VPxWorker *const worker = &cpi->workers[j];
      VP9BitstreamWorkerData *const data =
          (VP9BitstreamWorkerData *)worker->data2;
      uint32_t tile_size;
      int k;

      if (!winterface->sync(worker)) {
        error = 1;
        continue;
      }

      tile_size = data->bit_writer.pos;

      // Aggregate per-thread bitstream stats.
      cpi->max_mv_magnitude =
          VPXMAX(cpi->max_mv_magnitude, data->max_mv_magnitude);
      for (k = 0; k < SWITCHABLE; ++k) {
        cpi->interp_filter_selected[0][k] += data->interp_filter_selected[0][k];
      }

      // Prefix the size of the tile on all but the last.
      if (tile_col != tile_cols || j < i - 1) {
        if (data_size - total_size < 4) {
          error = 1;
          continue;
        }
        mem_put_be32(data_ptr + total_size, tile_size);
        total_size += 4;
      }
      if (j > 0) {
        if (data_size - total_size < tile_size) {
          error = 1;
          continue;
        }
        memcpy(data_ptr + total_size, data->dest, tile_size);
      }
      total_size += tile_size;
    }
    if (error) {
      vpx_internal_error(&cm->error, VPX_CODEC_ERROR,
                         "encode_tiles_mt: output buffer full");
    }
  }
  return total_size;
}

static size_t encode_tiles(VP9_COMP *cpi, uint8_t *data_ptr, size_t data_size) {
  VP9_COMMON *const cm = &cpi->common;
  MACROBLOCKD *const xd = &cpi->td.mb.e_mbd;
  vpx_writer residual_bc;
  int tile_row, tile_col;
  size_t total_size = 0;
  const int tile_cols = 1 << cm->log2_tile_cols;
  const int tile_rows = 1 << cm->log2_tile_rows;

  memset(cm->above_seg_context, 0,
         sizeof(*cm->above_seg_context) * mi_cols_aligned_to_sb(cm->mi_cols));

  // Encoding tiles in parallel is done only for realtime mode now. In other
  // modes the speed up is insignificant and requires further testing to ensure
  // that it does not make the overall process worse in any case.
  if (cpi->oxcf.mode == REALTIME && cpi->num_workers > 1 && tile_rows == 1 &&
      tile_cols > 1) {
    return encode_tiles_mt(cpi, data_ptr, data_size);
  }

  for (tile_row = 0; tile_row < tile_rows; tile_row++) {
    for (tile_col = 0; tile_col < tile_cols; tile_col++) {
      int tile_idx = tile_row * tile_cols + tile_col;

      size_t offset;
      if (tile_col < tile_cols - 1 || tile_row < tile_rows - 1)
        offset = total_size + 4;
      else
        offset = total_size;
      if (data_size < offset) {
        vpx_internal_error(&cm->error, VPX_CODEC_ERROR,
                           "encode_tiles: output buffer full");
      }
      vpx_start_encode(&residual_bc, data_ptr + offset, data_size - offset);

      write_modes(cpi, xd, &cpi->tile_data[tile_idx].tile_info, &residual_bc,
                  tile_row, tile_col, &cpi->max_mv_magnitude,
                  cpi->interp_filter_selected);

      if (vpx_stop_encode(&residual_bc)) {
        vpx_internal_error(&cm->error, VPX_CODEC_ERROR,
                           "encode_tiles: output buffer full");
      }
      if (tile_col < tile_cols - 1 || tile_row < tile_rows - 1) {
        // size of this tile
        mem_put_be32(data_ptr + total_size, residual_bc.pos);
        total_size += 4;
      }

      total_size += residual_bc.pos;
    }
  }
  return total_size;
}

static void write_render_size(const VP9_COMMON *cm,
                              struct vpx_write_bit_buffer *wb) {
  const int scaling_active =
      cm->width != cm->render_width || cm->height != cm->render_height;
  vpx_wb_write_bit(wb, scaling_active);
  if (scaling_active) {
    vpx_wb_write_literal(wb, cm->render_width - 1, 16);
    vpx_wb_write_literal(wb, cm->render_height - 1, 16);
  }
}

static void write_frame_size(const VP9_COMMON *cm,
                             struct vpx_write_bit_buffer *wb) {
  vpx_wb_write_literal(wb, cm->width - 1, 16);
  vpx_wb_write_literal(wb, cm->height - 1, 16);

  write_render_size(cm, wb);
}

static void write_frame_size_with_refs(VP9_COMP *cpi,
                                       struct vpx_write_bit_buffer *wb) {
  VP9_COMMON *const cm = &cpi->common;
  int found = 0;

  MV_REFERENCE_FRAME ref_frame;
  for (ref_frame = LAST_FRAME; ref_frame <= ALTREF_FRAME; ++ref_frame) {
    YV12_BUFFER_CONFIG *cfg = get_ref_frame_buffer(cpi, ref_frame);

    // Set "found" to 0 for temporal svc and for spatial svc key frame
    if (cpi->use_svc &&
        ((cpi->svc.number_temporal_layers > 1 &&
          cpi->oxcf.rc_mode == VPX_CBR) ||
         (cpi->svc.number_spatial_layers > 1 &&
          cpi->svc.layer_context[cpi->svc.spatial_layer_id].is_key_frame))) {
      found = 0;
    } else if (cfg != NULL) {
      found =
          cm->width == cfg->y_crop_width && cm->height == cfg->y_crop_height;
    }
    vpx_wb_write_bit(wb, found);
    if (found) {
      break;
    }
  }

  if (!found) {
    vpx_wb_write_literal(wb, cm->width - 1, 16);
    vpx_wb_write_literal(wb, cm->height - 1, 16);
  }

  write_render_size(cm, wb);
}

static void write_sync_code(struct vpx_write_bit_buffer *wb) {
  vpx_wb_write_literal(wb, VP9_SYNC_CODE_0, 8);
  vpx_wb_write_literal(wb, VP9_SYNC_CODE_1, 8);
  vpx_wb_write_literal(wb, VP9_SYNC_CODE_2, 8);
}

static void write_profile(BITSTREAM_PROFILE profile,
                          struct vpx_write_bit_buffer *wb) {
  switch (profile) {
    case PROFILE_0: vpx_wb_write_literal(wb, 0, 2); break;
    case PROFILE_1: vpx_wb_write_literal(wb, 2, 2); break;
    case PROFILE_2: vpx_wb_write_literal(wb, 1, 2); break;
    default:
      assert(profile == PROFILE_3);
      vpx_wb_write_literal(wb, 6, 3);
      break;
  }
}

static void write_bitdepth_colorspace_sampling(
    VP9_COMMON *const cm, struct vpx_write_bit_buffer *wb) {
  if (cm->profile >= PROFILE_2) {
    assert(cm->bit_depth > VPX_BITS_8);
    vpx_wb_write_bit(wb, cm->bit_depth == VPX_BITS_10 ? 0 : 1);
  }
  vpx_wb_write_literal(wb, cm->color_space, 3);
  if (cm->color_space != VPX_CS_SRGB) {
    // 0: [16, 235] (i.e. xvYCC), 1: [0, 255]
    vpx_wb_write_bit(wb, cm->color_range);
    if (cm->profile == PROFILE_1 || cm->profile == PROFILE_3) {
      assert(cm->subsampling_x != 1 || cm->subsampling_y != 1);
      vpx_wb_write_bit(wb, cm->subsampling_x);
      vpx_wb_write_bit(wb, cm->subsampling_y);
      vpx_wb_write_bit(wb, 0);  // unused
    } else {
      assert(cm->subsampling_x == 1 && cm->subsampling_y == 1);
    }
  } else {
    assert(cm->profile == PROFILE_1 || cm->profile == PROFILE_3);
    assert(cm->subsampling_x == 0 && cm->subsampling_y == 0);
    vpx_wb_write_bit(wb, 0);  // unused
  }
}

static void write_uncompressed_header(VP9_COMP *cpi,
                                      struct vpx_write_bit_buffer *wb) {
  VP9_COMMON *const cm = &cpi->common;
  MACROBLOCKD *const xd = &cpi->td.mb.e_mbd;

  vpx_wb_write_literal(wb, VP9_FRAME_MARKER, 2);

  write_profile(cm->profile, wb);

  // If to use show existing frame.
  vpx_wb_write_bit(wb, cm->show_existing_frame);
  if (cm->show_existing_frame) {
    vpx_wb_write_literal(wb, cpi->alt_fb_idx, 3);
    return;
  }

  vpx_wb_write_bit(wb, cm->frame_type);
  vpx_wb_write_bit(wb, cm->show_frame);
  vpx_wb_write_bit(wb, cm->error_resilient_mode);

  if (cm->frame_type == KEY_FRAME) {
    write_sync_code(wb);
    write_bitdepth_colorspace_sampling(cm, wb);
    write_frame_size(cm, wb);
  } else {
    if (!cm->show_frame) vpx_wb_write_bit(wb, cm->intra_only);

    if (!cm->error_resilient_mode)
      vpx_wb_write_literal(wb, cm->reset_frame_context, 2);

    if (cm->intra_only) {
      write_sync_code(wb);

      // Note for profile 0, 420 8bpp is assumed.
      if (cm->profile > PROFILE_0) {
        write_bitdepth_colorspace_sampling(cm, wb);
      }

      vpx_wb_write_literal(wb, vp9_get_refresh_mask(cpi), REF_FRAMES);
      write_frame_size(cm, wb);
    } else {
      MV_REFERENCE_FRAME ref_frame;
      vpx_wb_write_literal(wb, vp9_get_refresh_mask(cpi), REF_FRAMES);
      for (ref_frame = LAST_FRAME; ref_frame <= ALTREF_FRAME; ++ref_frame) {
        assert(get_ref_frame_map_idx(cpi, ref_frame) != INVALID_IDX);
        vpx_wb_write_literal(wb, get_ref_frame_map_idx(cpi, ref_frame),
                             REF_FRAMES_LOG2);
        vpx_wb_write_bit(wb, cm->ref_frame_sign_bias[ref_frame]);
      }

      write_frame_size_with_refs(cpi, wb);

      vpx_wb_write_bit(wb, cm->allow_high_precision_mv);

      fix_interp_filter(cm, cpi->td.counts);
      write_interp_filter(cm->interp_filter, wb);
    }
  }

  if (!cm->error_resilient_mode) {
    vpx_wb_write_bit(wb, cm->refresh_frame_context);
    vpx_wb_write_bit(wb, cm->frame_parallel_decoding_mode);
  }

  vpx_wb_write_literal(wb, cm->frame_context_idx, FRAME_CONTEXTS_LOG2);

  encode_loopfilter(&cm->lf, wb);
  encode_quantization(cm, wb);
  encode_segmentation(cm, xd, wb);

  write_tile_info(cm, wb);
}

static size_t write_compressed_header(VP9_COMP *cpi, uint8_t *data,
                                      size_t data_size) {
  VP9_COMMON *const cm = &cpi->common;
  MACROBLOCKD *const xd = &cpi->td.mb.e_mbd;
  FRAME_CONTEXT *const fc = cm->fc;
  FRAME_COUNTS *counts = cpi->td.counts;
  vpx_writer header_bc;

  vpx_start_encode(&header_bc, data, data_size);

  if (xd->lossless)
    cm->tx_mode = ONLY_4X4;
  else
    encode_txfm_probs(cm, &header_bc, counts);

  update_coef_probs(cpi, &header_bc);
  update_skip_probs(cm, &header_bc, counts);

  if (!frame_is_intra_only(cm)) {
    int i;

    for (i = 0; i < INTER_MODE_CONTEXTS; ++i)
      prob_diff_update(vp9_inter_mode_tree, cm->fc->inter_mode_probs[i],
                       counts->inter_mode[i], INTER_MODES, &header_bc);

    if (cm->interp_filter == SWITCHABLE)
      update_switchable_interp_probs(cm, &header_bc, counts);

    for (i = 0; i < INTRA_INTER_CONTEXTS; i++)
      vp9_cond_prob_diff_update(&header_bc, &fc->intra_inter_prob[i],
                                counts->intra_inter[i]);

    if (cpi->allow_comp_inter_inter) {
      const int use_compound_pred = cm->reference_mode != SINGLE_REFERENCE;
      const int use_hybrid_pred = cm->reference_mode == REFERENCE_MODE_SELECT;

      vpx_write_bit(&header_bc, use_compound_pred);
      if (use_compound_pred) {
        vpx_write_bit(&header_bc, use_hybrid_pred);
        if (use_hybrid_pred)
          for (i = 0; i < COMP_INTER_CONTEXTS; i++)
            vp9_cond_prob_diff_update(&header_bc, &fc->comp_inter_prob[i],
                                      counts->comp_inter[i]);
      }
    }

    if (cm->reference_mode != COMPOUND_REFERENCE) {
      for (i = 0; i < REF_CONTEXTS; i++) {
        vp9_cond_prob_diff_update(&header_bc, &fc->single_ref_prob[i][0],
                                  counts->single_ref[i][0]);
        vp9_cond_prob_diff_update(&header_bc, &fc->single_ref_prob[i][1],
                                  counts->single_ref[i][1]);
      }
    }

    if (cm->reference_mode != SINGLE_REFERENCE)
      for (i = 0; i < REF_CONTEXTS; i++)
        vp9_cond_prob_diff_update(&header_bc, &fc->comp_ref_prob[i],
                                  counts->comp_ref[i]);

    for (i = 0; i < BLOCK_SIZE_GROUPS; ++i)
      prob_diff_update(vp9_intra_mode_tree, cm->fc->y_mode_prob[i],
                       counts->y_mode[i], INTRA_MODES, &header_bc);

    for (i = 0; i < PARTITION_CONTEXTS; ++i)
      prob_diff_update(vp9_partition_tree, fc->partition_prob[i],
                       counts->partition[i], PARTITION_TYPES, &header_bc);

    vp9_write_nmv_probs(cm, cm->allow_high_precision_mv, &header_bc,
                        &counts->mv);
  }

  if (vpx_stop_encode(&header_bc)) {
    vpx_internal_error(&cm->error, VPX_CODEC_ERROR,
                       "write_compressed_header: output buffer full");
  }

  return header_bc.pos;
}

void vp9_pack_bitstream(VP9_COMP *cpi, uint8_t *dest, size_t dest_size,
                        size_t *size) {
  VP9_COMMON *const cm = &cpi->common;
  uint8_t *data = dest;
  size_t data_size = dest_size;
  size_t uncompressed_hdr_size, compressed_hdr_size;
  struct vpx_write_bit_buffer wb;
  struct vpx_write_bit_buffer saved_wb;

#if CONFIG_BITSTREAM_DEBUG
  bitstream_queue_reset_write();
#endif

  vpx_wb_init(&wb, data, data_size);
  write_uncompressed_header(cpi, &wb);
  if (vpx_wb_has_error(&wb)) {
    vpx_internal_error(&cm->error, VPX_CODEC_ERROR,
                       "vp9_pack_bitstream: output buffer full");
  }

  // Skip the rest coding process if use show existing frame.
  if (cm->show_existing_frame) {
    uncompressed_hdr_size = vpx_wb_bytes_written(&wb);
    data += uncompressed_hdr_size;
    *size = data - dest;
    return;
  }

  saved_wb = wb;
  // don't know in advance compressed header size
  vpx_wb_write_literal(&wb, 0, 16);
  if (vpx_wb_has_error(&wb)) {
    vpx_internal_error(&cm->error, VPX_CODEC_ERROR,
                       "vp9_pack_bitstream: output buffer full");
  }

  uncompressed_hdr_size = vpx_wb_bytes_written(&wb);
  data += uncompressed_hdr_size;
  data_size -= uncompressed_hdr_size;

  vpx_clear_system_state();

  compressed_hdr_size = write_compressed_header(cpi, data, data_size);
  data += compressed_hdr_size;
  data_size -= compressed_hdr_size;
  if (compressed_hdr_size > UINT16_MAX) {
    vpx_internal_error(&cm->error, VPX_CODEC_ERROR,
                       "compressed_hdr_size > 16 bits");
  }
  vpx_wb_write_literal(&saved_wb, (int)compressed_hdr_size, 16);
  assert(!vpx_wb_has_error(&saved_wb));

  data += encode_tiles(cpi, data, data_size);

  *size = data - dest;
}
