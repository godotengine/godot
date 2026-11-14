/*
 *  Copyright (c) 2023 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <math.h>

#include "./vpx_dsp_rtcd.h"
#if CONFIG_NON_GREEDY_MV
#include "vp9/common/vp9_mvref_common.h"
#endif
#include "vp9/common/vp9_reconinter.h"
#include "vp9/common/vp9_reconintra.h"
#include "vp9/common/vp9_scan.h"
#include "vp9/encoder/vp9_encoder.h"
#include "vp9/encoder/vp9_ext_ratectrl.h"
#include "vp9/encoder/vp9_firstpass.h"
#include "vp9/encoder/vp9_ratectrl.h"
#include "vp9/encoder/vp9_tpl_model.h"
#include "vpx/internal/vpx_codec_internal.h"
#include "vpx/vpx_codec.h"
#include "vpx/vpx_ext_ratectrl.h"

static int init_gop_frames_rc(VP9_COMP *cpi, GF_PICTURE *gf_picture,
                              const GF_GROUP *gf_group, int *tpl_group_frames) {
  VP9_COMMON *cm = &cpi->common;
  int frame_idx = 0;
  int i;
  int extend_frame_count = 0;
  int pframe_qindex = cpi->tpl_stats[2].base_qindex;
  int frame_gop_offset = 0;

  int added_overlay = 0;

  RefCntBuffer *frame_bufs = cm->buffer_pool->frame_bufs;
  int8_t recon_frame_index[REFS_PER_FRAME + MAX_ARF_LAYERS];

  memset(recon_frame_index, -1, sizeof(recon_frame_index));

  for (i = 0; i < FRAME_BUFFERS; ++i) {
    if (frame_bufs[i].ref_count == 0) {
      alloc_frame_mvs(cm, i);
      if (vpx_realloc_frame_buffer(&frame_bufs[i].buf, cm->width, cm->height,
                                   cm->subsampling_x, cm->subsampling_y,
#if CONFIG_VP9_HIGHBITDEPTH
                                   cm->use_highbitdepth,
#endif
                                   VP9_ENC_BORDER_IN_PIXELS, cm->byte_alignment,
                                   NULL, NULL, NULL))
        vpx_internal_error(&cm->error, VPX_CODEC_MEM_ERROR,
                           "Failed to allocate frame buffer");

      recon_frame_index[frame_idx] = i;
      ++frame_idx;

      if (frame_idx >= REFS_PER_FRAME + cpi->oxcf.enable_auto_arf) break;
    }
  }

  for (i = 0; i < REFS_PER_FRAME + 1; ++i) {
    assert(recon_frame_index[i] >= 0);
    cpi->tpl_recon_frames[i] = &frame_bufs[recon_frame_index[i]].buf;
  }

  *tpl_group_frames = 0;

  int ref_table[3];

  if (gf_group->index == 1 && gf_group->update_type[1] == ARF_UPDATE) {
    if (gf_group->update_type[0] == KF_UPDATE) {
      // This is the only frame in ref buffer. We need it to be on
      // gf_picture[0].
      for (i = 0; i < 3; ++i) ref_table[i] = -REFS_PER_FRAME;

      gf_picture[0].frame =
          &cm->buffer_pool
               ->frame_bufs[cm->ref_frame_map[gf_group->update_ref_idx[0]]]
               .buf;
      ref_table[gf_group->update_ref_idx[0]] = 0;

      for (i = 0; i < 3; ++i) gf_picture[0].ref_frame[i] = -REFS_PER_FRAME;
      gf_picture[0].update_type = gf_group->update_type[0];
    } else {
      for (i = 0; i < REFS_PER_FRAME; i++) {
        if (cm->ref_frame_map[i] != -1) {
          gf_picture[-i].frame =
              &cm->buffer_pool->frame_bufs[cm->ref_frame_map[i]].buf;
          ref_table[i] = -i;
        } else {
          ref_table[i] = -REFS_PER_FRAME;
        }
      }
      for (i = 0; i < 3; ++i) {
        gf_picture[0].ref_frame[i] = ref_table[i];
      }
    }
    ++*tpl_group_frames;

    // Initialize base layer ARF frame
    gf_picture[1].frame = cpi->Source;
    for (i = 0; i < 3; ++i) gf_picture[1].ref_frame[i] = ref_table[i];
    gf_picture[1].update_type = gf_group->update_type[1];
    ref_table[gf_group->update_ref_idx[1]] = 1;

    ++*tpl_group_frames;
  } else {
    assert(gf_group->index == 0);
    if (gf_group->update_type[0] == KF_UPDATE) {
      // This is the only frame in ref buffer. We need it to be on
      // gf_picture[0].
      gf_picture[0].frame = cpi->Source;
      for (i = 0; i < 3; ++i) gf_picture[0].ref_frame[i] = -REFS_PER_FRAME;
      gf_picture[0].update_type = gf_group->update_type[0];

      for (i = 0; i < 3; ++i) ref_table[i] = -REFS_PER_FRAME;
      ref_table[gf_group->update_ref_idx[0]] = 0;
    } else {
      // Initialize ref table
      for (i = 0; i < REFS_PER_FRAME; i++) {
        if (cm->ref_frame_map[i] != -1) {
          gf_picture[-i].frame =
              &cm->buffer_pool->frame_bufs[cm->ref_frame_map[i]].buf;
          ref_table[i] = -i;
        } else {
          ref_table[i] = -REFS_PER_FRAME;
        }
      }
      for (i = 0; i < 3; ++i) {
        gf_picture[0].ref_frame[i] = ref_table[i];
      }
      gf_picture[0].update_type = gf_group->update_type[0];
      if (gf_group->update_type[0] != OVERLAY_UPDATE &&
          gf_group->update_ref_idx[0] != -1) {
        ref_table[gf_group->update_ref_idx[0]] = 0;
      }
    }
    ++*tpl_group_frames;
  }

  int has_arf =
      gf_group->gf_group_size > 1 && gf_group->update_type[1] == ARF_UPDATE &&
      gf_group->update_type[gf_group->gf_group_size] == OVERLAY_UPDATE;

  // Initialize P frames
  for (frame_idx = *tpl_group_frames; frame_idx < MAX_ARF_GOP_SIZE;
       ++frame_idx) {
    if (frame_idx >= gf_group->gf_group_size && !has_arf) break;
    struct lookahead_entry *buf;
    frame_gop_offset = gf_group->frame_gop_index[frame_idx];
    buf = vp9_lookahead_peek(cpi->lookahead, frame_gop_offset - 1);

    if (buf == NULL) break;

    gf_picture[frame_idx].frame = &buf->img;
    for (i = 0; i < 3; ++i) {
      gf_picture[frame_idx].ref_frame[i] = ref_table[i];
    }

    if (gf_group->update_type[frame_idx] != OVERLAY_UPDATE &&
        gf_group->update_ref_idx[frame_idx] != -1) {
      ref_table[gf_group->update_ref_idx[frame_idx]] = frame_idx;
    }

    gf_picture[frame_idx].update_type = gf_group->update_type[frame_idx];

    ++*tpl_group_frames;

    // The length of group of pictures is baseline_gf_interval, plus the
    // beginning golden frame from last GOP, plus the last overlay frame in
    // the same GOP.
    if (frame_idx == gf_group->gf_group_size) {
      added_overlay = 1;

      ++frame_idx;
      ++frame_gop_offset;
      break;
    }

    if (frame_idx == gf_group->gf_group_size - 1 &&
        gf_group->update_type[gf_group->gf_group_size] != OVERLAY_UPDATE) {
      ++frame_idx;
      ++frame_gop_offset;
      break;
    }
  }

  int lst_index = frame_idx - 1;
  // Extend two frames outside the current gf group.
  for (; has_arf && frame_idx < MAX_LAG_BUFFERS && extend_frame_count < 2;
       ++frame_idx) {
    struct lookahead_entry *buf =
        vp9_lookahead_peek(cpi->lookahead, frame_gop_offset - 1);

    if (buf == NULL) break;

    cpi->tpl_stats[frame_idx].base_qindex = pframe_qindex;

    gf_picture[frame_idx].frame = &buf->img;
    gf_picture[frame_idx].ref_frame[0] = gf_picture[lst_index].ref_frame[0];
    gf_picture[frame_idx].ref_frame[1] = gf_picture[lst_index].ref_frame[1];
    gf_picture[frame_idx].ref_frame[2] = gf_picture[lst_index].ref_frame[2];

    if (gf_picture[frame_idx].ref_frame[0] >
            gf_picture[frame_idx].ref_frame[1] &&
        gf_picture[frame_idx].ref_frame[0] >
            gf_picture[frame_idx].ref_frame[2]) {
      gf_picture[frame_idx].ref_frame[0] = lst_index;
    } else if (gf_picture[frame_idx].ref_frame[1] >
                   gf_picture[frame_idx].ref_frame[0] &&
               gf_picture[frame_idx].ref_frame[1] >
                   gf_picture[frame_idx].ref_frame[2]) {
      gf_picture[frame_idx].ref_frame[1] = lst_index;
    } else {
      gf_picture[frame_idx].ref_frame[2] = lst_index;
    }

    gf_picture[frame_idx].update_type = LF_UPDATE;
    lst_index = frame_idx;
    ++*tpl_group_frames;
    ++extend_frame_count;
    ++frame_gop_offset;
  }

  return extend_frame_count + added_overlay;
}

static int init_gop_frames(VP9_COMP *cpi, GF_PICTURE *gf_picture,
                           const GF_GROUP *gf_group, int *tpl_group_frames) {
  if (cpi->ext_ratectrl.ready &&
      (cpi->ext_ratectrl.funcs.rc_type & VPX_RC_GOP) != 0) {
    return init_gop_frames_rc(cpi, gf_picture, gf_group, tpl_group_frames);
  }

  VP9_COMMON *cm = &cpi->common;
  int frame_idx = 0;
  int i;
  int gld_index = -1;
  int alt_index = -2;
  int lst_index = -1;
  int arf_index_stack[MAX_ARF_LAYERS];
  int arf_stack_size = 0;
  int extend_frame_count = 0;
  int pframe_qindex = cpi->tpl_stats[2].base_qindex;
  int frame_gop_offset = 0;

  RefCntBuffer *frame_bufs = cm->buffer_pool->frame_bufs;
  int8_t recon_frame_index[REFS_PER_FRAME + MAX_ARF_LAYERS];

  memset(recon_frame_index, -1, sizeof(recon_frame_index));
  stack_init(arf_index_stack, MAX_ARF_LAYERS);

  for (i = 0; i < FRAME_BUFFERS; ++i) {
    if (frame_bufs[i].ref_count == 0) {
      alloc_frame_mvs(cm, i);
      if (vpx_realloc_frame_buffer(&frame_bufs[i].buf, cm->width, cm->height,
                                   cm->subsampling_x, cm->subsampling_y,
#if CONFIG_VP9_HIGHBITDEPTH
                                   cm->use_highbitdepth,
#endif
                                   VP9_ENC_BORDER_IN_PIXELS, cm->byte_alignment,
                                   NULL, NULL, NULL))
        vpx_internal_error(&cm->error, VPX_CODEC_MEM_ERROR,
                           "Failed to allocate frame buffer");

      recon_frame_index[frame_idx] = i;
      ++frame_idx;

      if (frame_idx >= REFS_PER_FRAME + cpi->oxcf.enable_auto_arf) break;
    }
  }

  for (i = 0; i < REFS_PER_FRAME + 1; ++i) {
    assert(recon_frame_index[i] >= 0);
    cpi->tpl_recon_frames[i] = &frame_bufs[recon_frame_index[i]].buf;
  }

  *tpl_group_frames = 0;

  // Initialize Golden reference frame.
  gf_picture[0].frame = get_ref_frame_buffer(cpi, GOLDEN_FRAME);
  for (i = 0; i < 3; ++i) gf_picture[0].ref_frame[i] = -REFS_PER_FRAME;
  gf_picture[0].update_type = gf_group->update_type[0];
  gld_index = 0;
  ++*tpl_group_frames;

  gf_picture[-1].frame = get_ref_frame_buffer(cpi, LAST_FRAME);
  gf_picture[-2].frame = get_ref_frame_buffer(cpi, ALTREF_FRAME);

  // Initialize base layer ARF frame
  gf_picture[1].frame = cpi->Source;
  gf_picture[1].ref_frame[0] = gld_index;
  gf_picture[1].ref_frame[1] = lst_index;
  gf_picture[1].ref_frame[2] = alt_index;
  gf_picture[1].update_type = gf_group->update_type[1];
  alt_index = 1;
  ++*tpl_group_frames;

  // Initialize P frames
  for (frame_idx = 2; frame_idx < MAX_ARF_GOP_SIZE; ++frame_idx) {
    struct lookahead_entry *buf;
    frame_gop_offset = gf_group->frame_gop_index[frame_idx];
    buf = vp9_lookahead_peek(cpi->lookahead, frame_gop_offset - 1);

    if (buf == NULL) break;

    gf_picture[frame_idx].frame = &buf->img;
    gf_picture[frame_idx].ref_frame[0] = gld_index;
    gf_picture[frame_idx].ref_frame[1] = lst_index;
    gf_picture[frame_idx].ref_frame[2] = alt_index;
    gf_picture[frame_idx].update_type = gf_group->update_type[frame_idx];

    switch (gf_group->update_type[frame_idx]) {
      case ARF_UPDATE:
        stack_push(arf_index_stack, alt_index, arf_stack_size);
        ++arf_stack_size;
        alt_index = frame_idx;
        break;
      case LF_UPDATE: lst_index = frame_idx; break;
      case OVERLAY_UPDATE:
        gld_index = frame_idx;
        alt_index = stack_pop(arf_index_stack, arf_stack_size);
        --arf_stack_size;
        break;
      case USE_BUF_FRAME:
        lst_index = alt_index;
        alt_index = stack_pop(arf_index_stack, arf_stack_size);
        --arf_stack_size;
        break;
      default: break;
    }

    ++*tpl_group_frames;

    // The length of group of pictures is baseline_gf_interval, plus the
    // beginning golden frame from last GOP, plus the last overlay frame in
    // the same GOP.
    if (frame_idx == gf_group->gf_group_size) break;
  }

  alt_index = -1;
  ++frame_idx;
  ++frame_gop_offset;

  // Extend two frames outside the current gf group.
  for (; frame_idx < MAX_LAG_BUFFERS && extend_frame_count < 2; ++frame_idx) {
    struct lookahead_entry *buf =
        vp9_lookahead_peek(cpi->lookahead, frame_gop_offset - 1);

    if (buf == NULL) break;

    cpi->tpl_stats[frame_idx].base_qindex = pframe_qindex;

    gf_picture[frame_idx].frame = &buf->img;
    gf_picture[frame_idx].ref_frame[0] = gld_index;
    gf_picture[frame_idx].ref_frame[1] = lst_index;
    gf_picture[frame_idx].ref_frame[2] = alt_index;
    gf_picture[frame_idx].update_type = LF_UPDATE;
    lst_index = frame_idx;
    ++*tpl_group_frames;
    ++extend_frame_count;
    ++frame_gop_offset;
  }

  return extend_frame_count;
}

static void init_tpl_stats(VP9_COMP *cpi) {
  int frame_idx;
  for (frame_idx = 0; frame_idx < MAX_ARF_GOP_SIZE; ++frame_idx) {
    TplDepFrame *tpl_frame = &cpi->tpl_stats[frame_idx];
    memset(tpl_frame->tpl_stats_ptr, 0,
           tpl_frame->height * tpl_frame->width *
               sizeof(*tpl_frame->tpl_stats_ptr));
    tpl_frame->is_valid = 0;
  }
}

static void free_tpl_frame_stats_list(VpxTplGopStats *tpl_gop_stats) {
  int frame_idx;
  for (frame_idx = 0; frame_idx < tpl_gop_stats->size; ++frame_idx) {
    vpx_free(tpl_gop_stats->frame_stats_list[frame_idx].block_stats_list);
  }
  vpx_free(tpl_gop_stats->frame_stats_list);
}

static void init_tpl_stats_before_propagation(
    struct vpx_internal_error_info *error_info, VpxTplGopStats *tpl_gop_stats,
    TplDepFrame *tpl_stats, int tpl_gop_frames, int frame_width,
    int frame_height) {
  int frame_idx;
  free_tpl_frame_stats_list(tpl_gop_stats);
  CHECK_MEM_ERROR(
      error_info, tpl_gop_stats->frame_stats_list,
      vpx_calloc(tpl_gop_frames, sizeof(*tpl_gop_stats->frame_stats_list)));
  tpl_gop_stats->size = tpl_gop_frames;
  for (frame_idx = 0; frame_idx < tpl_gop_frames; ++frame_idx) {
    const int mi_rows = tpl_stats[frame_idx].mi_rows;
    const int mi_cols = tpl_stats[frame_idx].mi_cols;
    CHECK_MEM_ERROR(
        error_info, tpl_gop_stats->frame_stats_list[frame_idx].block_stats_list,
        vpx_calloc(
            mi_rows * mi_cols,
            sizeof(
                *tpl_gop_stats->frame_stats_list[frame_idx].block_stats_list)));
    tpl_gop_stats->frame_stats_list[frame_idx].num_blocks = mi_rows * mi_cols;
    tpl_gop_stats->frame_stats_list[frame_idx].frame_width = frame_width;
    tpl_gop_stats->frame_stats_list[frame_idx].frame_height = frame_height;
  }
}

#if CONFIG_NON_GREEDY_MV
static uint32_t full_pixel_motion_search(VP9_COMP *cpi, ThreadData *td,
                                         MotionField *motion_field,
                                         int frame_idx, uint8_t *cur_frame_buf,
                                         uint8_t *ref_frame_buf, int stride,
                                         BLOCK_SIZE bsize, int mi_row,
                                         int mi_col, MV *mv) {
  MACROBLOCK *const x = &td->mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  MV_SPEED_FEATURES *const mv_sf = &cpi->sf.mv;
  int step_param;
  uint32_t bestsme = UINT_MAX;
  const MvLimits tmp_mv_limits = x->mv_limits;
  // lambda is used to adjust the importance of motion vector consistency.
  // TODO(angiebird): Figure out lambda's proper value.
  const int lambda = cpi->tpl_stats[frame_idx].lambda;
  int_mv nb_full_mvs[NB_MVS_NUM];
  int nb_full_mv_num;

  MV best_ref_mv1 = { 0, 0 };
  MV best_ref_mv1_full; /* full-pixel value of best_ref_mv1 */

  best_ref_mv1_full.col = best_ref_mv1.col >> 3;
  best_ref_mv1_full.row = best_ref_mv1.row >> 3;

  // Setup frame pointers
  x->plane[0].src.buf = cur_frame_buf;
  x->plane[0].src.stride = stride;
  xd->plane[0].pre[0].buf = ref_frame_buf;
  xd->plane[0].pre[0].stride = stride;

  step_param = mv_sf->reduce_first_step_size;
  step_param = VPXMIN(step_param, MAX_MVSEARCH_STEPS - 2);

  vp9_set_mv_search_range(&x->mv_limits, &best_ref_mv1);

  nb_full_mv_num =
      vp9_prepare_nb_full_mvs(motion_field, mi_row, mi_col, nb_full_mvs);
  vp9_full_pixel_diamond_new(cpi, x, bsize, &best_ref_mv1_full, step_param,
                             lambda, 1, nb_full_mvs, nb_full_mv_num, mv);

  /* restore UMV window */
  x->mv_limits = tmp_mv_limits;

  return bestsme;
}

static uint32_t sub_pixel_motion_search(VP9_COMP *cpi, ThreadData *td,
                                        uint8_t *cur_frame_buf,
                                        uint8_t *ref_frame_buf, int stride,
                                        BLOCK_SIZE bsize, MV *mv) {
  MACROBLOCK *const x = &td->mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  MV_SPEED_FEATURES *const mv_sf = &cpi->sf.mv;
  uint32_t bestsme = UINT_MAX;
  uint32_t distortion;
  uint32_t sse;
  int cost_list[5];

  MV best_ref_mv1 = { 0, 0 };

  // Setup frame pointers
  x->plane[0].src.buf = cur_frame_buf;
  x->plane[0].src.stride = stride;
  xd->plane[0].pre[0].buf = ref_frame_buf;
  xd->plane[0].pre[0].stride = stride;

  // TODO(yunqing): may use higher tap interp filter than 2 taps.
  // Ignore mv costing by sending NULL pointer instead of cost array
  bestsme = cpi->find_fractional_mv_step(
      x, mv, &best_ref_mv1, cpi->common.allow_high_precision_mv, x->errorperbit,
      &cpi->fn_ptr[bsize], 0, mv_sf->subpel_search_level,
      cond_cost_list(cpi, cost_list), NULL, NULL, &distortion, &sse, NULL, 0, 0,
      USE_2_TAPS);

  return bestsme;
}

#else  // CONFIG_NON_GREEDY_MV
static uint32_t motion_compensated_prediction(VP9_COMP *cpi, ThreadData *td,
                                              uint8_t *cur_frame_buf,
                                              uint8_t *ref_frame_buf,
                                              int stride, BLOCK_SIZE bsize,
                                              MV *mv) {
  MACROBLOCK *const x = &td->mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  MV_SPEED_FEATURES *const mv_sf = &cpi->sf.mv;
  const SEARCH_METHODS search_method = NSTEP;
  int step_param;
  int sadpb = x->sadperbit16;
  uint32_t bestsme = UINT_MAX;
  uint32_t distortion;
  uint32_t sse;
  int cost_list[5];
  const MvLimits tmp_mv_limits = x->mv_limits;

  MV best_ref_mv1 = { 0, 0 };
  MV best_ref_mv1_full; /* full-pixel value of best_ref_mv1 */

  best_ref_mv1_full.col = best_ref_mv1.col >> 3;
  best_ref_mv1_full.row = best_ref_mv1.row >> 3;

  // Setup frame pointers
  x->plane[0].src.buf = cur_frame_buf;
  x->plane[0].src.stride = stride;
  xd->plane[0].pre[0].buf = ref_frame_buf;
  xd->plane[0].pre[0].stride = stride;

  step_param = mv_sf->reduce_first_step_size;
  step_param = VPXMIN(step_param, MAX_MVSEARCH_STEPS - 2);

  vp9_set_mv_search_range(&x->mv_limits, &best_ref_mv1);

  vp9_full_pixel_search(cpi, x, bsize, &best_ref_mv1_full, step_param,
                        search_method, sadpb, cond_cost_list(cpi, cost_list),
                        &best_ref_mv1, mv, 0, 0);

  /* restore UMV window */
  x->mv_limits = tmp_mv_limits;

  // TODO(yunqing): may use higher tap interp filter than 2 taps.
  // Ignore mv costing by sending NULL pointer instead of cost array
  bestsme = cpi->find_fractional_mv_step(
      x, mv, &best_ref_mv1, cpi->common.allow_high_precision_mv, x->errorperbit,
      &cpi->fn_ptr[bsize], 0, mv_sf->subpel_search_level,
      cond_cost_list(cpi, cost_list), NULL, NULL, &distortion, &sse, NULL, 0, 0,
      USE_2_TAPS);

  return bestsme;
}
#endif

static int get_overlap_area(int grid_pos_row, int grid_pos_col, int ref_pos_row,
                            int ref_pos_col, int block, BLOCK_SIZE bsize) {
  int width = 0, height = 0;
  int bw = 4 << b_width_log2_lookup[bsize];
  int bh = 4 << b_height_log2_lookup[bsize];

  switch (block) {
    case 0:
      width = grid_pos_col + bw - ref_pos_col;
      height = grid_pos_row + bh - ref_pos_row;
      break;
    case 1:
      width = ref_pos_col + bw - grid_pos_col;
      height = grid_pos_row + bh - ref_pos_row;
      break;
    case 2:
      width = grid_pos_col + bw - ref_pos_col;
      height = ref_pos_row + bh - grid_pos_row;
      break;
    case 3:
      width = ref_pos_col + bw - grid_pos_col;
      height = ref_pos_row + bh - grid_pos_row;
      break;
    default: assert(0);
  }

  return width * height;
}

static int round_floor(int ref_pos, int bsize_pix) {
  int round;
  if (ref_pos < 0)
    round = -(1 + (-ref_pos - 1) / bsize_pix);
  else
    round = ref_pos / bsize_pix;

  return round;
}

static void tpl_model_store(TplDepStats *tpl_stats, int mi_row, int mi_col,
                            BLOCK_SIZE bsize, int stride) {
  const int mi_height = num_8x8_blocks_high_lookup[bsize];
  const int mi_width = num_8x8_blocks_wide_lookup[bsize];
  const TplDepStats *src_stats = &tpl_stats[mi_row * stride + mi_col];
  int idx, idy;

  for (idy = 0; idy < mi_height; ++idy) {
    for (idx = 0; idx < mi_width; ++idx) {
      TplDepStats *tpl_ptr = &tpl_stats[(mi_row + idy) * stride + mi_col + idx];
      const int64_t mc_flow = tpl_ptr->mc_flow;
      const int64_t mc_ref_cost = tpl_ptr->mc_ref_cost;
      *tpl_ptr = *src_stats;
      tpl_ptr->mc_flow = mc_flow;
      tpl_ptr->mc_ref_cost = mc_ref_cost;
      tpl_ptr->mc_dep_cost = tpl_ptr->intra_cost + tpl_ptr->mc_flow;
    }
  }
}

static void tpl_store_before_propagation(VpxTplBlockStats *tpl_block_stats,
                                         TplDepStats *tpl_stats, int mi_row,
                                         int mi_col, BLOCK_SIZE bsize,
                                         int src_stride, int64_t recon_error,
                                         int64_t pred_error, int64_t rate_cost,
                                         int ref_frame_idx, int mi_rows,
                                         int mi_cols) {
  const int mi_height = num_8x8_blocks_high_lookup[bsize];
  const int mi_width = num_8x8_blocks_wide_lookup[bsize];
  const TplDepStats *src_stats = &tpl_stats[mi_row * src_stride + mi_col];
  int idx, idy;

  for (idy = 0; idy < mi_height; ++idy) {
    for (idx = 0; idx < mi_width; ++idx) {
      if (mi_row + idy >= mi_rows || mi_col + idx >= mi_cols) continue;
      VpxTplBlockStats *tpl_block_stats_ptr =
          &tpl_block_stats[(mi_row + idy) * mi_cols + mi_col + idx];
      tpl_block_stats_ptr->row = mi_row * 8 + idy * 8;
      tpl_block_stats_ptr->col = mi_col * 8 + idx * 8;
      tpl_block_stats_ptr->inter_cost = src_stats->inter_cost;
      tpl_block_stats_ptr->intra_cost = src_stats->intra_cost;
      // inter/intra_cost here is calculated with SATD which should be close
      // enough to be used as inter/intra_pred_error
      tpl_block_stats_ptr->inter_pred_err = src_stats->inter_cost;
      tpl_block_stats_ptr->intra_pred_err = src_stats->intra_cost;
      tpl_block_stats_ptr->srcrf_dist = recon_error << TPL_DEP_COST_SCALE_LOG2;
      tpl_block_stats_ptr->srcrf_rate = rate_cost << TPL_DEP_COST_SCALE_LOG2;
      tpl_block_stats_ptr->pred_error = pred_error << TPL_DEP_COST_SCALE_LOG2;
      tpl_block_stats_ptr->mv_r = (src_stats->mv.as_mv.row >= 0 ? 1 : -1) *
                                  (abs(src_stats->mv.as_mv.row) + 4) / 8;
      tpl_block_stats_ptr->mv_c = (src_stats->mv.as_mv.col >= 0 ? 1 : -1) *
                                  (abs(src_stats->mv.as_mv.col) + 4) / 8;
      tpl_block_stats_ptr->ref_frame_index = ref_frame_idx;
    }
  }
}

static void tpl_model_update_b(TplDepFrame *tpl_frame, TplDepStats *tpl_stats,
                               int mi_row, int mi_col, const BLOCK_SIZE bsize) {
  if (tpl_stats->ref_frame_index < 0) return;

  TplDepFrame *ref_tpl_frame = &tpl_frame[tpl_stats->ref_frame_index];
  TplDepStats *ref_stats = ref_tpl_frame->tpl_stats_ptr;
  MV mv = tpl_stats->mv.as_mv;
  int mv_row = mv.row >> 3;
  int mv_col = mv.col >> 3;

  int ref_pos_row = mi_row * MI_SIZE + mv_row;
  int ref_pos_col = mi_col * MI_SIZE + mv_col;

  const int bw = 4 << b_width_log2_lookup[bsize];
  const int bh = 4 << b_height_log2_lookup[bsize];
  const int mi_height = num_8x8_blocks_high_lookup[bsize];
  const int mi_width = num_8x8_blocks_wide_lookup[bsize];
  const int pix_num = bw * bh;

  // top-left on grid block location in pixel
  int grid_pos_row_base = round_floor(ref_pos_row, bh) * bh;
  int grid_pos_col_base = round_floor(ref_pos_col, bw) * bw;
  int block;

  for (block = 0; block < 4; ++block) {
    int grid_pos_row = grid_pos_row_base + bh * (block >> 1);
    int grid_pos_col = grid_pos_col_base + bw * (block & 0x01);

    if (grid_pos_row >= 0 && grid_pos_row < ref_tpl_frame->mi_rows * MI_SIZE &&
        grid_pos_col >= 0 && grid_pos_col < ref_tpl_frame->mi_cols * MI_SIZE) {
      int overlap_area = get_overlap_area(
          grid_pos_row, grid_pos_col, ref_pos_row, ref_pos_col, block, bsize);
      int ref_mi_row = round_floor(grid_pos_row, bh) * mi_height;
      int ref_mi_col = round_floor(grid_pos_col, bw) * mi_width;

      int64_t mc_flow = tpl_stats->mc_dep_cost -
                        (tpl_stats->mc_dep_cost * tpl_stats->inter_cost) /
                            tpl_stats->intra_cost;

      int idx, idy;

      for (idy = 0; idy < mi_height; ++idy) {
        for (idx = 0; idx < mi_width; ++idx) {
          TplDepStats *des_stats =
              &ref_stats[(ref_mi_row + idy) * ref_tpl_frame->stride +
                         (ref_mi_col + idx)];

          des_stats->mc_flow += (mc_flow * overlap_area) / pix_num;
          des_stats->mc_ref_cost +=
              ((tpl_stats->intra_cost - tpl_stats->inter_cost) * overlap_area) /
              pix_num;
          assert(overlap_area >= 0);
        }
      }
    }
  }
}

static void tpl_model_update(TplDepFrame *tpl_frame, TplDepStats *tpl_stats,
                             int mi_row, int mi_col, const BLOCK_SIZE bsize) {
  int idx, idy;
  const int mi_height = num_8x8_blocks_high_lookup[bsize];
  const int mi_width = num_8x8_blocks_wide_lookup[bsize];

  for (idy = 0; idy < mi_height; ++idy) {
    for (idx = 0; idx < mi_width; ++idx) {
      TplDepStats *tpl_ptr =
          &tpl_stats[(mi_row + idy) * tpl_frame->stride + (mi_col + idx)];
      tpl_model_update_b(tpl_frame, tpl_ptr, mi_row + idy, mi_col + idx,
                         BLOCK_8X8);
    }
  }
}

static void get_quantize_error(MACROBLOCK *x, int plane, tran_low_t *coeff,
                               tran_low_t *qcoeff, tran_low_t *dqcoeff,
                               TX_SIZE tx_size, int64_t *recon_error,
                               int64_t *sse, uint16_t *eob) {
  MACROBLOCKD *const xd = &x->e_mbd;
  const struct macroblock_plane *const p = &x->plane[plane];
  const struct macroblockd_plane *const pd = &xd->plane[plane];
  const ScanOrder *const scan_order = &vp9_default_scan_orders[tx_size];
  int pix_num = 1 << num_pels_log2_lookup[txsize_to_bsize[tx_size]];
  const int shift = tx_size == TX_32X32 ? 0 : 2;

  // skip block condition should be handled before this is called.
  assert(!x->skip_block);

#if CONFIG_VP9_HIGHBITDEPTH
  if (xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH) {
    vp9_highbd_quantize_fp_32x32(coeff, pix_num, p, qcoeff, dqcoeff,
                                 pd->dequant, eob, scan_order);
  } else {
    vp9_quantize_fp_32x32(coeff, pix_num, p, qcoeff, dqcoeff, pd->dequant, eob,
                          scan_order);
  }
#else
  vp9_quantize_fp_32x32(coeff, pix_num, p, qcoeff, dqcoeff, pd->dequant, eob,
                        scan_order);
#endif  // CONFIG_VP9_HIGHBITDEPTH

  *recon_error = vp9_block_error(coeff, dqcoeff, pix_num, sse) >> shift;
  *recon_error = VPXMAX(*recon_error, 1);

  *sse = (*sse) >> shift;
  *sse = VPXMAX(*sse, 1);
}

#if CONFIG_VP9_HIGHBITDEPTH
void vp9_highbd_wht_fwd_txfm(int16_t *src_diff, int bw, tran_low_t *coeff,
                             TX_SIZE tx_size) {
  // TODO(sdeng): Implement SIMD based high bit-depth Hadamard transforms.
  switch (tx_size) {
    case TX_8X8: vpx_highbd_hadamard_8x8(src_diff, bw, coeff); break;
    case TX_16X16: vpx_highbd_hadamard_16x16(src_diff, bw, coeff); break;
    case TX_32X32: vpx_highbd_hadamard_32x32(src_diff, bw, coeff); break;
    default: assert(0);
  }
}
#endif  // CONFIG_VP9_HIGHBITDEPTH

void vp9_wht_fwd_txfm(int16_t *src_diff, int bw, tran_low_t *coeff,
                      TX_SIZE tx_size) {
  switch (tx_size) {
    case TX_8X8: vpx_hadamard_8x8(src_diff, bw, coeff); break;
    case TX_16X16: vpx_hadamard_16x16(src_diff, bw, coeff); break;
    case TX_32X32: vpx_hadamard_32x32(src_diff, bw, coeff); break;
    default: assert(0);
  }
}

static void set_mv_limits(const VP9_COMMON *cm, MACROBLOCK *x, int mi_row,
                          int mi_col) {
  x->mv_limits.row_min = -((mi_row * MI_SIZE) + (17 - 2 * VP9_INTERP_EXTEND));
  x->mv_limits.row_max =
      (cm->mi_rows - 1 - mi_row) * MI_SIZE + (17 - 2 * VP9_INTERP_EXTEND);
  x->mv_limits.col_min = -((mi_col * MI_SIZE) + (17 - 2 * VP9_INTERP_EXTEND));
  x->mv_limits.col_max =
      ((cm->mi_cols - 1 - mi_col) * MI_SIZE) + (17 - 2 * VP9_INTERP_EXTEND);
}

static int rate_estimator(const tran_low_t *qcoeff, int eob, TX_SIZE tx_size) {
  const ScanOrder *const scan_order = &vp9_scan_orders[tx_size][DCT_DCT];
  int rate_cost = 1;
  int idx;
  assert((1 << num_pels_log2_lookup[txsize_to_bsize[tx_size]]) >= eob);
  for (idx = 0; idx < eob; ++idx) {
    unsigned int abs_level = abs(qcoeff[scan_order->scan[idx]]);
    rate_cost += get_msb(abs_level + 1) + 1 + (abs_level > 0);
  }

  return (rate_cost << VP9_PROB_COST_SHIFT);
}

static void mode_estimation(VP9_COMP *cpi, MACROBLOCK *x, MACROBLOCKD *xd,
                            struct scale_factors *sf, GF_PICTURE *gf_picture,
                            int frame_idx, TplDepFrame *tpl_frame,
                            int16_t *src_diff, tran_low_t *coeff,
                            tran_low_t *qcoeff, tran_low_t *dqcoeff, int mi_row,
                            int mi_col, BLOCK_SIZE bsize, TX_SIZE tx_size,
                            YV12_BUFFER_CONFIG *ref_frame[], uint8_t *predictor,
                            int64_t *recon_error, int64_t *rate_cost,
                            int64_t *sse, int *ref_frame_idx) {
  VP9_COMMON *cm = &cpi->common;
  ThreadData *td = &cpi->td;

  const int bw = 4 << b_width_log2_lookup[bsize];
  const int bh = 4 << b_height_log2_lookup[bsize];
  const int pix_num = bw * bh;
  int best_rf_idx = -1;
  int_mv best_mv;
  int64_t best_inter_cost = INT64_MAX;
  int64_t inter_cost;
  int rf_idx;
  const InterpKernel *const kernel = vp9_filter_kernels[EIGHTTAP];

  int64_t best_intra_cost = INT64_MAX;
  int64_t intra_cost;
  PREDICTION_MODE mode;
  int mb_y_offset = mi_row * MI_SIZE * xd->cur_buf->y_stride + mi_col * MI_SIZE;
  MODE_INFO mi_above, mi_left;
  const int mi_height = num_8x8_blocks_high_lookup[bsize];
  const int mi_width = num_8x8_blocks_wide_lookup[bsize];
  TplDepStats *tpl_stats =
      &tpl_frame->tpl_stats_ptr[mi_row * tpl_frame->stride + mi_col];

  xd->mb_to_top_edge = -((mi_row * MI_SIZE) * 8);
  xd->mb_to_bottom_edge = ((cm->mi_rows - 1 - mi_row) * MI_SIZE) * 8;
  xd->mb_to_left_edge = -((mi_col * MI_SIZE) * 8);
  xd->mb_to_right_edge = ((cm->mi_cols - 1 - mi_col) * MI_SIZE) * 8;
  xd->above_mi = (mi_row > 0) ? &mi_above : NULL;
  xd->left_mi = (mi_col > 0) ? &mi_left : NULL;

  // Intra prediction search
  for (mode = DC_PRED; mode <= TM_PRED; ++mode) {
    uint8_t *src, *dst;
    int src_stride, dst_stride;

    src = xd->cur_buf->y_buffer + mb_y_offset;
    src_stride = xd->cur_buf->y_stride;

    dst = &predictor[0];
    dst_stride = bw;

    xd->mi[0]->sb_type = bsize;
    xd->mi[0]->ref_frame[0] = INTRA_FRAME;

    vp9_predict_intra_block(xd, b_width_log2_lookup[bsize], tx_size, mode, src,
                            src_stride, dst, dst_stride, 0, 0, 0);

#if CONFIG_VP9_HIGHBITDEPTH
    if (xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH) {
      vpx_highbd_subtract_block(bh, bw, src_diff, bw, src, src_stride, dst,
                                dst_stride, xd->bd);
      vp9_highbd_wht_fwd_txfm(src_diff, bw, coeff, tx_size);
      intra_cost = vpx_highbd_satd(coeff, pix_num);
    } else {
      vpx_subtract_block(bh, bw, src_diff, bw, src, src_stride, dst,
                         dst_stride);
      vp9_wht_fwd_txfm(src_diff, bw, coeff, tx_size);
      intra_cost = vpx_satd(coeff, pix_num);
    }
#else
    vpx_subtract_block(bh, bw, src_diff, bw, src, src_stride, dst, dst_stride);
    vp9_wht_fwd_txfm(src_diff, bw, coeff, tx_size);
    intra_cost = vpx_satd(coeff, pix_num);
#endif  // CONFIG_VP9_HIGHBITDEPTH

    if (intra_cost < best_intra_cost) best_intra_cost = intra_cost;
  }

  // Motion compensated prediction
  best_mv.as_int = 0;

  set_mv_limits(cm, x, mi_row, mi_col);

  for (rf_idx = 0; rf_idx < MAX_INTER_REF_FRAMES; ++rf_idx) {
    int_mv mv;
#if CONFIG_NON_GREEDY_MV
    MotionField *motion_field;
#endif
    if (ref_frame[rf_idx] == NULL) continue;

#if CONFIG_NON_GREEDY_MV
    (void)td;
    motion_field = vp9_motion_field_info_get_motion_field(
        &cpi->motion_field_info, frame_idx, rf_idx, bsize);
    mv = vp9_motion_field_mi_get_mv(motion_field, mi_row, mi_col);
#else
    motion_compensated_prediction(cpi, td, xd->cur_buf->y_buffer + mb_y_offset,
                                  ref_frame[rf_idx]->y_buffer + mb_y_offset,
                                  xd->cur_buf->y_stride, bsize, &mv.as_mv);
#endif

#if CONFIG_VP9_HIGHBITDEPTH
    if (xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH) {
      vp9_highbd_build_inter_predictor(
          CONVERT_TO_SHORTPTR(ref_frame[rf_idx]->y_buffer + mb_y_offset),
          ref_frame[rf_idx]->y_stride, CONVERT_TO_SHORTPTR(&predictor[0]), bw,
          &mv.as_mv, sf, bw, bh, 0, kernel, MV_PRECISION_Q3, mi_col * MI_SIZE,
          mi_row * MI_SIZE, xd->bd);
      vpx_highbd_subtract_block(
          bh, bw, src_diff, bw, xd->cur_buf->y_buffer + mb_y_offset,
          xd->cur_buf->y_stride, &predictor[0], bw, xd->bd);
      vp9_highbd_wht_fwd_txfm(src_diff, bw, coeff, tx_size);
      inter_cost = vpx_highbd_satd(coeff, pix_num);
    } else {
      vp9_build_inter_predictor(
          ref_frame[rf_idx]->y_buffer + mb_y_offset,
          ref_frame[rf_idx]->y_stride, &predictor[0], bw, &mv.as_mv, sf, bw, bh,
          0, kernel, MV_PRECISION_Q3, mi_col * MI_SIZE, mi_row * MI_SIZE);
      vpx_subtract_block(bh, bw, src_diff, bw,
                         xd->cur_buf->y_buffer + mb_y_offset,
                         xd->cur_buf->y_stride, &predictor[0], bw);
      vp9_wht_fwd_txfm(src_diff, bw, coeff, tx_size);
      inter_cost = vpx_satd(coeff, pix_num);
    }
#else
    vp9_build_inter_predictor(ref_frame[rf_idx]->y_buffer + mb_y_offset,
                              ref_frame[rf_idx]->y_stride, &predictor[0], bw,
                              &mv.as_mv, sf, bw, bh, 0, kernel, MV_PRECISION_Q3,
                              mi_col * MI_SIZE, mi_row * MI_SIZE);
    vpx_subtract_block(bh, bw, src_diff, bw,
                       xd->cur_buf->y_buffer + mb_y_offset,
                       xd->cur_buf->y_stride, &predictor[0], bw);
    vp9_wht_fwd_txfm(src_diff, bw, coeff, tx_size);
    inter_cost = vpx_satd(coeff, pix_num);
#endif

    if (inter_cost < best_inter_cost) {
      uint16_t eob = 0;
      best_rf_idx = rf_idx;
      best_inter_cost = inter_cost;
      best_mv.as_int = mv.as_int;
      // Since best_inter_cost is initialized as INT64_MAX, recon_error and
      // rate_cost will be calculated with the best reference frame.
      get_quantize_error(x, 0, coeff, qcoeff, dqcoeff, tx_size, recon_error,
                         sse, &eob);
      *rate_cost = rate_estimator(qcoeff, eob, tx_size);
    }
  }
  best_intra_cost = VPXMAX(best_intra_cost, 1);
  best_inter_cost = VPXMIN(best_intra_cost, best_inter_cost);
  tpl_stats->inter_cost = VPXMAX(
      1, (best_inter_cost << TPL_DEP_COST_SCALE_LOG2) / (mi_height * mi_width));
  tpl_stats->intra_cost = VPXMAX(
      1, (best_intra_cost << TPL_DEP_COST_SCALE_LOG2) / (mi_height * mi_width));
  if (best_rf_idx >= 0) {
    tpl_stats->ref_frame_index = gf_picture[frame_idx].ref_frame[best_rf_idx];
  }
  tpl_stats->mv.as_int = best_mv.as_int;
  *ref_frame_idx = best_rf_idx;
}

#if CONFIG_NON_GREEDY_MV
static int get_block_src_pred_buf(MACROBLOCKD *xd, GF_PICTURE *gf_picture,
                                  int frame_idx, int rf_idx, int mi_row,
                                  int mi_col, struct buf_2d *src,
                                  struct buf_2d *pre) {
  const int mb_y_offset =
      mi_row * MI_SIZE * xd->cur_buf->y_stride + mi_col * MI_SIZE;
  YV12_BUFFER_CONFIG *ref_frame = NULL;
  int ref_frame_idx = gf_picture[frame_idx].ref_frame[rf_idx];
  if (ref_frame_idx != -1) {
    ref_frame = gf_picture[ref_frame_idx].frame;
    src->buf = xd->cur_buf->y_buffer + mb_y_offset;
    src->stride = xd->cur_buf->y_stride;
    pre->buf = ref_frame->y_buffer + mb_y_offset;
    pre->stride = ref_frame->y_stride;
    assert(src->stride == pre->stride);
    return 1;
  } else {
    printf("invalid ref_frame_idx");
    assert(ref_frame_idx != -1);
    return 0;
  }
}

#define kMvPreCheckLines 5
#define kMvPreCheckSize 15

#define MV_REF_POS_NUM 3
POSITION mv_ref_pos[MV_REF_POS_NUM] = {
  { -1, 0 },
  { 0, -1 },
  { -1, -1 },
};

static int_mv *get_select_mv(VP9_COMP *cpi, TplDepFrame *tpl_frame, int mi_row,
                             int mi_col) {
  return &cpi->select_mv_arr[mi_row * tpl_frame->stride + mi_col];
}

static int_mv find_ref_mv(int mv_mode, VP9_COMP *cpi, TplDepFrame *tpl_frame,
                          BLOCK_SIZE bsize, int mi_row, int mi_col) {
  int i;
  const int mi_height = num_8x8_blocks_high_lookup[bsize];
  const int mi_width = num_8x8_blocks_wide_lookup[bsize];
  int_mv nearest_mv, near_mv, invalid_mv;
  nearest_mv.as_int = INVALID_MV;
  near_mv.as_int = INVALID_MV;
  invalid_mv.as_int = INVALID_MV;
  for (i = 0; i < MV_REF_POS_NUM; ++i) {
    int nb_row = mi_row + mv_ref_pos[i].row * mi_height;
    int nb_col = mi_col + mv_ref_pos[i].col * mi_width;
    assert(mv_ref_pos[i].row <= 0);
    assert(mv_ref_pos[i].col <= 0);
    if (nb_row >= 0 && nb_col >= 0) {
      if (nearest_mv.as_int == INVALID_MV) {
        nearest_mv = *get_select_mv(cpi, tpl_frame, nb_row, nb_col);
      } else {
        int_mv mv = *get_select_mv(cpi, tpl_frame, nb_row, nb_col);
        if (mv.as_int == nearest_mv.as_int) {
          continue;
        } else {
          near_mv = mv;
          break;
        }
      }
    }
  }
  if (nearest_mv.as_int == INVALID_MV) {
    nearest_mv.as_mv.row = 0;
    nearest_mv.as_mv.col = 0;
  }
  if (near_mv.as_int == INVALID_MV) {
    near_mv.as_mv.row = 0;
    near_mv.as_mv.col = 0;
  }
  if (mv_mode == NEAREST_MV_MODE) {
    return nearest_mv;
  }
  if (mv_mode == NEAR_MV_MODE) {
    return near_mv;
  }
  assert(0);
  return invalid_mv;
}

static int_mv get_mv_from_mv_mode(int mv_mode, VP9_COMP *cpi,
                                  MotionField *motion_field,
                                  TplDepFrame *tpl_frame, BLOCK_SIZE bsize,
                                  int mi_row, int mi_col) {
  int_mv mv;
  switch (mv_mode) {
    case ZERO_MV_MODE:
      mv.as_mv.row = 0;
      mv.as_mv.col = 0;
      break;
    case NEW_MV_MODE:
      mv = vp9_motion_field_mi_get_mv(motion_field, mi_row, mi_col);
      break;
    case NEAREST_MV_MODE:
      mv = find_ref_mv(mv_mode, cpi, tpl_frame, bsize, mi_row, mi_col);
      break;
    case NEAR_MV_MODE:
      mv = find_ref_mv(mv_mode, cpi, tpl_frame, bsize, mi_row, mi_col);
      break;
    default:
      mv.as_int = INVALID_MV;
      assert(0);
      break;
  }
  return mv;
}

static double get_mv_dist(int mv_mode, VP9_COMP *cpi, MACROBLOCKD *xd,
                          GF_PICTURE *gf_picture, MotionField *motion_field,
                          int frame_idx, TplDepFrame *tpl_frame, int rf_idx,
                          BLOCK_SIZE bsize, int mi_row, int mi_col,
                          int_mv *mv) {
  uint32_t sse;
  struct buf_2d src;
  struct buf_2d pre;
  MV full_mv;
  *mv = get_mv_from_mv_mode(mv_mode, cpi, motion_field, tpl_frame, bsize,
                            mi_row, mi_col);
  full_mv = get_full_mv(&mv->as_mv);
  if (get_block_src_pred_buf(xd, gf_picture, frame_idx, rf_idx, mi_row, mi_col,
                             &src, &pre)) {
    // TODO(angiebird): Consider subpixel when computing the sse.
    cpi->fn_ptr[bsize].vf(src.buf, src.stride, get_buf_from_mv(&pre, &full_mv),
                          pre.stride, &sse);
    return (double)(sse << VP9_DIST_SCALE_LOG2);
  } else {
    assert(0);
    return 0;
  }
}

static int get_mv_mode_cost(int mv_mode) {
  // TODO(angiebird): The probabilities are roughly inferred from
  // default_inter_mode_probs. Check if there is a better way to set the
  // probabilities.
  const int zero_mv_prob = 16;
  const int new_mv_prob = 24 * 1;
  const int ref_mv_prob = 256 - zero_mv_prob - new_mv_prob;
  assert(zero_mv_prob + new_mv_prob + ref_mv_prob == 256);
  switch (mv_mode) {
    case ZERO_MV_MODE: return vp9_prob_cost[zero_mv_prob]; break;
    case NEW_MV_MODE: return vp9_prob_cost[new_mv_prob]; break;
    case NEAREST_MV_MODE: return vp9_prob_cost[ref_mv_prob]; break;
    case NEAR_MV_MODE: return vp9_prob_cost[ref_mv_prob]; break;
    default: assert(0); return -1;
  }
}

static INLINE double get_mv_diff_cost(MV *new_mv, MV *ref_mv) {
  double mv_diff_cost = log2(1 + abs(new_mv->row - ref_mv->row)) +
                        log2(1 + abs(new_mv->col - ref_mv->col));
  mv_diff_cost *= (1 << VP9_PROB_COST_SHIFT);
  return mv_diff_cost;
}
static double get_mv_cost(int mv_mode, VP9_COMP *cpi, MotionField *motion_field,
                          TplDepFrame *tpl_frame, BLOCK_SIZE bsize, int mi_row,
                          int mi_col) {
  double mv_cost = get_mv_mode_cost(mv_mode);
  if (mv_mode == NEW_MV_MODE) {
    MV new_mv = get_mv_from_mv_mode(mv_mode, cpi, motion_field, tpl_frame,
                                    bsize, mi_row, mi_col)
                    .as_mv;
    MV nearest_mv = get_mv_from_mv_mode(NEAREST_MV_MODE, cpi, motion_field,
                                        tpl_frame, bsize, mi_row, mi_col)
                        .as_mv;
    MV near_mv = get_mv_from_mv_mode(NEAR_MV_MODE, cpi, motion_field, tpl_frame,
                                     bsize, mi_row, mi_col)
                     .as_mv;
    double nearest_cost = get_mv_diff_cost(&new_mv, &nearest_mv);
    double near_cost = get_mv_diff_cost(&new_mv, &near_mv);
    mv_cost += nearest_cost < near_cost ? nearest_cost : near_cost;
  }
  return mv_cost;
}

static double eval_mv_mode(int mv_mode, VP9_COMP *cpi, MACROBLOCK *x,
                           GF_PICTURE *gf_picture, MotionField *motion_field,
                           int frame_idx, TplDepFrame *tpl_frame, int rf_idx,
                           BLOCK_SIZE bsize, int mi_row, int mi_col,
                           int_mv *mv) {
  MACROBLOCKD *xd = &x->e_mbd;
  double mv_dist =
      get_mv_dist(mv_mode, cpi, xd, gf_picture, motion_field, frame_idx,
                  tpl_frame, rf_idx, bsize, mi_row, mi_col, mv);
  double mv_cost =
      get_mv_cost(mv_mode, cpi, motion_field, tpl_frame, bsize, mi_row, mi_col);
  double mult = 180;

  return mv_cost + mult * log2f(1 + mv_dist);
}

static int find_best_ref_mv_mode(VP9_COMP *cpi, MACROBLOCK *x,
                                 GF_PICTURE *gf_picture,
                                 MotionField *motion_field, int frame_idx,
                                 TplDepFrame *tpl_frame, int rf_idx,
                                 BLOCK_SIZE bsize, int mi_row, int mi_col,
                                 double *rd, int_mv *mv) {
  int best_mv_mode = ZERO_MV_MODE;
  int update = 0;
  int mv_mode;
  *rd = 0;
  for (mv_mode = 0; mv_mode < MAX_MV_MODE; ++mv_mode) {
    double this_rd;
    int_mv this_mv;
    if (mv_mode == NEW_MV_MODE) {
      continue;
    }
    this_rd = eval_mv_mode(mv_mode, cpi, x, gf_picture, motion_field, frame_idx,
                           tpl_frame, rf_idx, bsize, mi_row, mi_col, &this_mv);
    if (update == 0) {
      *rd = this_rd;
      *mv = this_mv;
      best_mv_mode = mv_mode;
      update = 1;
    } else {
      if (this_rd < *rd) {
        *rd = this_rd;
        *mv = this_mv;
        best_mv_mode = mv_mode;
      }
    }
  }
  return best_mv_mode;
}

static void predict_mv_mode(VP9_COMP *cpi, MACROBLOCK *x,
                            GF_PICTURE *gf_picture, MotionField *motion_field,
                            int frame_idx, TplDepFrame *tpl_frame, int rf_idx,
                            BLOCK_SIZE bsize, int mi_row, int mi_col) {
  const int mi_height = num_8x8_blocks_high_lookup[bsize];
  const int mi_width = num_8x8_blocks_wide_lookup[bsize];
  int tmp_mv_mode_arr[kMvPreCheckSize];
  int *mv_mode_arr = tpl_frame->mv_mode_arr[rf_idx];
  double *rd_diff_arr = tpl_frame->rd_diff_arr[rf_idx];
  int_mv *select_mv_arr = cpi->select_mv_arr;
  int_mv tmp_select_mv_arr[kMvPreCheckSize];
  int stride = tpl_frame->stride;
  double new_mv_rd = 0;
  double no_new_mv_rd = 0;
  double this_new_mv_rd = 0;
  double this_no_new_mv_rd = 0;
  int idx;
  int tmp_idx;
  assert(kMvPreCheckSize == (kMvPreCheckLines * (kMvPreCheckLines + 1)) >> 1);

  // no new mv
  // diagonal scan order
  tmp_idx = 0;
  for (idx = 0; idx < kMvPreCheckLines; ++idx) {
    int r;
    for (r = 0; r <= idx; ++r) {
      int c = idx - r;
      int nb_row = mi_row + r * mi_height;
      int nb_col = mi_col + c * mi_width;
      if (nb_row < tpl_frame->mi_rows && nb_col < tpl_frame->mi_cols) {
        double this_rd;
        int_mv *mv = &select_mv_arr[nb_row * stride + nb_col];
        mv_mode_arr[nb_row * stride + nb_col] = find_best_ref_mv_mode(
            cpi, x, gf_picture, motion_field, frame_idx, tpl_frame, rf_idx,
            bsize, nb_row, nb_col, &this_rd, mv);
        if (r == 0 && c == 0) {
          this_no_new_mv_rd = this_rd;
        }
        no_new_mv_rd += this_rd;
        tmp_mv_mode_arr[tmp_idx] = mv_mode_arr[nb_row * stride + nb_col];
        tmp_select_mv_arr[tmp_idx] = select_mv_arr[nb_row * stride + nb_col];
        ++tmp_idx;
      }
    }
  }

  // new mv
  mv_mode_arr[mi_row * stride + mi_col] = NEW_MV_MODE;
  this_new_mv_rd = eval_mv_mode(
      NEW_MV_MODE, cpi, x, gf_picture, motion_field, frame_idx, tpl_frame,
      rf_idx, bsize, mi_row, mi_col, &select_mv_arr[mi_row * stride + mi_col]);
  new_mv_rd = this_new_mv_rd;
  // We start from idx = 1 because idx = 0 is evaluated as NEW_MV_MODE
  // beforehand.
  for (idx = 1; idx < kMvPreCheckLines; ++idx) {
    int r;
    for (r = 0; r <= idx; ++r) {
      int c = idx - r;
      int nb_row = mi_row + r * mi_height;
      int nb_col = mi_col + c * mi_width;
      if (nb_row < tpl_frame->mi_rows && nb_col < tpl_frame->mi_cols) {
        double this_rd;
        int_mv *mv = &select_mv_arr[nb_row * stride + nb_col];
        mv_mode_arr[nb_row * stride + nb_col] = find_best_ref_mv_mode(
            cpi, x, gf_picture, motion_field, frame_idx, tpl_frame, rf_idx,
            bsize, nb_row, nb_col, &this_rd, mv);
        new_mv_rd += this_rd;
      }
    }
  }

  // update best_mv_mode
  tmp_idx = 0;
  if (no_new_mv_rd < new_mv_rd) {
    for (idx = 0; idx < kMvPreCheckLines; ++idx) {
      int r;
      for (r = 0; r <= idx; ++r) {
        int c = idx - r;
        int nb_row = mi_row + r * mi_height;
        int nb_col = mi_col + c * mi_width;
        if (nb_row < tpl_frame->mi_rows && nb_col < tpl_frame->mi_cols) {
          mv_mode_arr[nb_row * stride + nb_col] = tmp_mv_mode_arr[tmp_idx];
          select_mv_arr[nb_row * stride + nb_col] = tmp_select_mv_arr[tmp_idx];
          ++tmp_idx;
        }
      }
    }
    rd_diff_arr[mi_row * stride + mi_col] = 0;
  } else {
    rd_diff_arr[mi_row * stride + mi_col] =
        (no_new_mv_rd - this_no_new_mv_rd) - (new_mv_rd - this_new_mv_rd);
  }
}

static void predict_mv_mode_arr(VP9_COMP *cpi, MACROBLOCK *x,
                                GF_PICTURE *gf_picture,
                                MotionField *motion_field, int frame_idx,
                                TplDepFrame *tpl_frame, int rf_idx,
                                BLOCK_SIZE bsize) {
  const int mi_height = num_8x8_blocks_high_lookup[bsize];
  const int mi_width = num_8x8_blocks_wide_lookup[bsize];
  const int unit_rows = tpl_frame->mi_rows / mi_height;
  const int unit_cols = tpl_frame->mi_cols / mi_width;
  const int max_diagonal_lines = unit_rows + unit_cols - 1;
  int idx;
  for (idx = 0; idx < max_diagonal_lines; ++idx) {
    int r;
    for (r = VPXMAX(idx - unit_cols + 1, 0); r <= VPXMIN(idx, unit_rows - 1);
         ++r) {
      int c = idx - r;
      int mi_row = r * mi_height;
      int mi_col = c * mi_width;
      assert(c >= 0 && c < unit_cols);
      assert(mi_row >= 0 && mi_row < tpl_frame->mi_rows);
      assert(mi_col >= 0 && mi_col < tpl_frame->mi_cols);
      predict_mv_mode(cpi, x, gf_picture, motion_field, frame_idx, tpl_frame,
                      rf_idx, bsize, mi_row, mi_col);
    }
  }
}

static void do_motion_search(VP9_COMP *cpi, ThreadData *td,
                             MotionField *motion_field, int frame_idx,
                             YV12_BUFFER_CONFIG *ref_frame, BLOCK_SIZE bsize,
                             int mi_row, int mi_col) {
  VP9_COMMON *cm = &cpi->common;
  MACROBLOCK *x = &td->mb;
  MACROBLOCKD *xd = &x->e_mbd;
  const int mb_y_offset =
      mi_row * MI_SIZE * xd->cur_buf->y_stride + mi_col * MI_SIZE;
  assert(ref_frame != NULL);
  set_mv_limits(cm, x, mi_row, mi_col);
  {
    int_mv mv = vp9_motion_field_mi_get_mv(motion_field, mi_row, mi_col);
    uint8_t *cur_frame_buf = xd->cur_buf->y_buffer + mb_y_offset;
    uint8_t *ref_frame_buf = ref_frame->y_buffer + mb_y_offset;
    const int stride = xd->cur_buf->y_stride;
    full_pixel_motion_search(cpi, td, motion_field, frame_idx, cur_frame_buf,
                             ref_frame_buf, stride, bsize, mi_row, mi_col,
                             &mv.as_mv);
    sub_pixel_motion_search(cpi, td, cur_frame_buf, ref_frame_buf, stride,
                            bsize, &mv.as_mv);
    vp9_motion_field_mi_set_mv(motion_field, mi_row, mi_col, mv);
  }
}

static void build_motion_field(
    VP9_COMP *cpi, int frame_idx,
    YV12_BUFFER_CONFIG *ref_frame[MAX_INTER_REF_FRAMES], BLOCK_SIZE bsize) {
  VP9_COMMON *cm = &cpi->common;
  ThreadData *td = &cpi->td;
  TplDepFrame *tpl_frame = &cpi->tpl_stats[frame_idx];
  const int mi_height = num_8x8_blocks_high_lookup[bsize];
  const int mi_width = num_8x8_blocks_wide_lookup[bsize];
  const int pw = num_4x4_blocks_wide_lookup[bsize] << 2;
  const int ph = num_4x4_blocks_high_lookup[bsize] << 2;
  int mi_row, mi_col;
  int rf_idx;

  tpl_frame->lambda = (pw * ph) >> 2;
  assert(pw * ph == tpl_frame->lambda << 2);

  for (rf_idx = 0; rf_idx < MAX_INTER_REF_FRAMES; ++rf_idx) {
    MotionField *motion_field = vp9_motion_field_info_get_motion_field(
        &cpi->motion_field_info, frame_idx, rf_idx, bsize);
    if (ref_frame[rf_idx] == NULL) {
      continue;
    }
    vp9_motion_field_reset_mvs(motion_field);
    for (mi_row = 0; mi_row < cm->mi_rows; mi_row += mi_height) {
      for (mi_col = 0; mi_col < cm->mi_cols; mi_col += mi_width) {
        do_motion_search(cpi, td, motion_field, frame_idx, ref_frame[rf_idx],
                         bsize, mi_row, mi_col);
      }
    }
  }
}
#endif  // CONFIG_NON_GREEDY_MV

static void mc_flow_dispenser(VP9_COMP *cpi, GF_PICTURE *gf_picture,
                              int frame_idx, BLOCK_SIZE bsize) {
  TplDepFrame *tpl_frame = &cpi->tpl_stats[frame_idx];
  VpxTplFrameStats *tpl_frame_stats_before_propagation =
      &cpi->tpl_gop_stats.frame_stats_list[frame_idx];
  YV12_BUFFER_CONFIG *this_frame = gf_picture[frame_idx].frame;
  YV12_BUFFER_CONFIG *ref_frame[MAX_INTER_REF_FRAMES] = { NULL, NULL, NULL };

  VP9_COMMON *cm = &cpi->common;
  struct scale_factors sf;
  int rdmult, idx;
  ThreadData *td = &cpi->td;
  MACROBLOCK *x = &td->mb;
  MACROBLOCKD *xd = &x->e_mbd;
  int mi_row, mi_col;

#if CONFIG_VP9_HIGHBITDEPTH
  DECLARE_ALIGNED(16, uint16_t, predictor16[32 * 32 * 3]);
  DECLARE_ALIGNED(16, uint8_t, predictor8[32 * 32 * 3]);
  uint8_t *predictor;
#else
  DECLARE_ALIGNED(16, uint8_t, predictor[32 * 32 * 3]);
#endif
  DECLARE_ALIGNED(16, int16_t, src_diff[32 * 32]);
  DECLARE_ALIGNED(16, tran_low_t, coeff[32 * 32]);
  DECLARE_ALIGNED(16, tran_low_t, qcoeff[32 * 32]);
  DECLARE_ALIGNED(16, tran_low_t, dqcoeff[32 * 32]);

  const TX_SIZE tx_size = max_txsize_lookup[bsize];
  const int mi_height = num_8x8_blocks_high_lookup[bsize];
  const int mi_width = num_8x8_blocks_wide_lookup[bsize];

  tpl_frame_stats_before_propagation->frame_width = cm->width;
  tpl_frame_stats_before_propagation->frame_height = cm->height;
  // Setup scaling factor
#if CONFIG_VP9_HIGHBITDEPTH
  vp9_setup_scale_factors_for_frame(
      &sf, this_frame->y_crop_width, this_frame->y_crop_height,
      this_frame->y_crop_width, this_frame->y_crop_height,
      cpi->common.use_highbitdepth);

  if (xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH)
    predictor = CONVERT_TO_BYTEPTR(predictor16);
  else
    predictor = predictor8;
#else
  vp9_setup_scale_factors_for_frame(
      &sf, this_frame->y_crop_width, this_frame->y_crop_height,
      this_frame->y_crop_width, this_frame->y_crop_height);
#endif  // CONFIG_VP9_HIGHBITDEPTH

  // Prepare reference frame pointers. If any reference frame slot is
  // unavailable, the pointer will be set to Null.
  for (idx = 0; idx < MAX_INTER_REF_FRAMES; ++idx) {
    int rf_idx = gf_picture[frame_idx].ref_frame[idx];
    if (rf_idx != -REFS_PER_FRAME) ref_frame[idx] = gf_picture[rf_idx].frame;
  }

  xd->mi = cm->mi_grid_visible;
  xd->mi[0] = cm->mi;
  xd->cur_buf = this_frame;

  // Get rd multiplier set up.
  rdmult = vp9_compute_rd_mult_based_on_qindex(cpi, tpl_frame->base_qindex);
  set_error_per_bit(&cpi->td.mb, rdmult);
  vp9_initialize_me_consts(cpi, &cpi->td.mb, tpl_frame->base_qindex);

  tpl_frame->is_valid = 1;

  cm->base_qindex = tpl_frame->base_qindex;
  vp9_frame_init_quantizer(cpi);

#if CONFIG_NON_GREEDY_MV
  {
    int square_block_idx;
    int rf_idx;
    for (square_block_idx = 0; square_block_idx < SQUARE_BLOCK_SIZES;
         ++square_block_idx) {
      BLOCK_SIZE square_bsize = square_block_idx_to_bsize(square_block_idx);
      build_motion_field(cpi, frame_idx, ref_frame, square_bsize);
    }
    for (rf_idx = 0; rf_idx < MAX_INTER_REF_FRAMES; ++rf_idx) {
      int ref_frame_idx = gf_picture[frame_idx].ref_frame[rf_idx];
      if (ref_frame_idx != -1) {
        MotionField *motion_field = vp9_motion_field_info_get_motion_field(
            &cpi->motion_field_info, frame_idx, rf_idx, bsize);
        predict_mv_mode_arr(cpi, x, gf_picture, motion_field, frame_idx,
                            tpl_frame, rf_idx, bsize);
      }
    }
  }
#endif  // CONFIG_NON_GREEDY_MV

  for (mi_row = 0; mi_row < cm->mi_rows; mi_row += mi_height) {
    for (mi_col = 0; mi_col < cm->mi_cols; mi_col += mi_width) {
      int64_t recon_error = 0;
      int64_t rate_cost = 0;
      int64_t sse = 0;
      // Ref frame index in the ref frame buffer.
      int ref_frame_idx = -1;
      mode_estimation(cpi, x, xd, &sf, gf_picture, frame_idx, tpl_frame,
                      src_diff, coeff, qcoeff, dqcoeff, mi_row, mi_col, bsize,
                      tx_size, ref_frame, predictor, &recon_error, &rate_cost,
                      &sse, &ref_frame_idx);
      // Motion flow dependency dispenser.
      tpl_model_store(tpl_frame->tpl_stats_ptr, mi_row, mi_col, bsize,
                      tpl_frame->stride);

      tpl_store_before_propagation(
          tpl_frame_stats_before_propagation->block_stats_list,
          tpl_frame->tpl_stats_ptr, mi_row, mi_col, bsize, tpl_frame->stride,
          recon_error, sse, rate_cost, ref_frame_idx, tpl_frame->mi_rows,
          tpl_frame->mi_cols);

      tpl_model_update(cpi->tpl_stats, tpl_frame->tpl_stats_ptr, mi_row, mi_col,
                       bsize);
    }
  }
}

static void trim_tpl_stats(struct vpx_internal_error_info *error_info,
                           VpxTplGopStats *tpl_gop_stats, int extra_frames) {
  int i;
  VpxTplFrameStats *new_frame_stats;
  const int new_size = tpl_gop_stats->size - extra_frames;
  if (tpl_gop_stats->size <= extra_frames)
    vpx_internal_error(
        error_info, VPX_CODEC_ERROR,
        "The number of frames in VpxTplGopStats is fewer than expected.");
  CHECK_MEM_ERROR(error_info, new_frame_stats,
                  vpx_calloc(new_size, sizeof(*new_frame_stats)));
  for (i = 0; i < new_size; i++) {
    VpxTplFrameStats *frame_stats = &tpl_gop_stats->frame_stats_list[i];
    const int num_blocks = frame_stats->num_blocks;
    new_frame_stats[i].num_blocks = frame_stats->num_blocks;
    new_frame_stats[i].frame_width = frame_stats->frame_width;
    new_frame_stats[i].frame_height = frame_stats->frame_height;
    new_frame_stats[i].num_blocks = num_blocks;
    CHECK_MEM_ERROR(
        error_info, new_frame_stats[i].block_stats_list,
        vpx_calloc(num_blocks, sizeof(*new_frame_stats[i].block_stats_list)));
    memcpy(new_frame_stats[i].block_stats_list, frame_stats->block_stats_list,
           num_blocks * sizeof(*new_frame_stats[i].block_stats_list));
  }
  free_tpl_frame_stats_list(tpl_gop_stats);
  tpl_gop_stats->size = new_size;
  tpl_gop_stats->frame_stats_list = new_frame_stats;
}

#if CONFIG_NON_GREEDY_MV
#define DUMP_TPL_STATS 0
#if DUMP_TPL_STATS
static void dump_buf(uint8_t *buf, int stride, int row, int col, int h, int w) {
  int i, j;
  printf("%d %d\n", h, w);
  for (i = 0; i < h; ++i) {
    for (j = 0; j < w; ++j) {
      printf("%d ", buf[(row + i) * stride + col + j]);
    }
  }
  printf("\n");
}

static void dump_frame_buf(const YV12_BUFFER_CONFIG *frame_buf) {
  dump_buf(frame_buf->y_buffer, frame_buf->y_stride, 0, 0, frame_buf->y_height,
           frame_buf->y_width);
  dump_buf(frame_buf->u_buffer, frame_buf->uv_stride, 0, 0,
           frame_buf->uv_height, frame_buf->uv_width);
  dump_buf(frame_buf->v_buffer, frame_buf->uv_stride, 0, 0,
           frame_buf->uv_height, frame_buf->uv_width);
}

static void dump_tpl_stats(const VP9_COMP *cpi, int tpl_group_frames,
                           const GF_GROUP *gf_group,
                           const GF_PICTURE *gf_picture, BLOCK_SIZE bsize) {
  int frame_idx;
  const VP9_COMMON *cm = &cpi->common;
  int rf_idx;
  for (frame_idx = 1; frame_idx < tpl_group_frames; ++frame_idx) {
    for (rf_idx = 0; rf_idx < MAX_INTER_REF_FRAMES; ++rf_idx) {
      const TplDepFrame *tpl_frame = &cpi->tpl_stats[frame_idx];
      int mi_row, mi_col;
      int ref_frame_idx;
      const int mi_height = num_8x8_blocks_high_lookup[bsize];
      const int mi_width = num_8x8_blocks_wide_lookup[bsize];
      ref_frame_idx = gf_picture[frame_idx].ref_frame[rf_idx];
      if (ref_frame_idx != -1) {
        YV12_BUFFER_CONFIG *ref_frame_buf = gf_picture[ref_frame_idx].frame;
        const int gf_frame_offset = gf_group->frame_gop_index[frame_idx];
        const int ref_gf_frame_offset =
            gf_group->frame_gop_index[ref_frame_idx];
        printf("=\n");
        printf(
            "frame_idx %d mi_rows %d mi_cols %d bsize %d ref_frame_idx %d "
            "rf_idx %d gf_frame_offset %d ref_gf_frame_offset %d\n",
            frame_idx, cm->mi_rows, cm->mi_cols, mi_width * MI_SIZE,
            ref_frame_idx, rf_idx, gf_frame_offset, ref_gf_frame_offset);
        for (mi_row = 0; mi_row < cm->mi_rows; ++mi_row) {
          for (mi_col = 0; mi_col < cm->mi_cols; ++mi_col) {
            if ((mi_row % mi_height) == 0 && (mi_col % mi_width) == 0) {
              int_mv mv = vp9_motion_field_info_get_mv(&cpi->motion_field_info,
                                                       frame_idx, rf_idx, bsize,
                                                       mi_row, mi_col);
              printf("%d %d %d %d\n", mi_row, mi_col, mv.as_mv.row,
                     mv.as_mv.col);
            }
          }
        }
        for (mi_row = 0; mi_row < cm->mi_rows; ++mi_row) {
          for (mi_col = 0; mi_col < cm->mi_cols; ++mi_col) {
            if ((mi_row % mi_height) == 0 && (mi_col % mi_width) == 0) {
              const TplDepStats *tpl_ptr =
                  &tpl_frame
                       ->tpl_stats_ptr[mi_row * tpl_frame->stride + mi_col];
              printf("%f ", tpl_ptr->feature_score);
            }
          }
        }
        printf("\n");

        for (mi_row = 0; mi_row < cm->mi_rows; mi_row += mi_height) {
          for (mi_col = 0; mi_col < cm->mi_cols; mi_col += mi_width) {
            const int mv_mode =
                tpl_frame
                    ->mv_mode_arr[rf_idx][mi_row * tpl_frame->stride + mi_col];
            printf("%d ", mv_mode);
          }
        }
        printf("\n");

        dump_frame_buf(gf_picture[frame_idx].frame);
        dump_frame_buf(ref_frame_buf);
      }
    }
  }
}
#endif  // DUMP_TPL_STATS
#endif  // CONFIG_NON_GREEDY_MV

void vp9_init_tpl_buffer(VP9_COMP *cpi) {
  VP9_COMMON *cm = &cpi->common;
  int frame;

  const int mi_cols = mi_cols_aligned_to_sb(cm->mi_cols);
  const int mi_rows = mi_cols_aligned_to_sb(cm->mi_rows);
#if CONFIG_NON_GREEDY_MV
  int rf_idx;

  vpx_free(cpi->select_mv_arr);
  CHECK_MEM_ERROR(
      &cm->error, cpi->select_mv_arr,
      vpx_calloc(mi_rows * mi_cols * 4, sizeof(*cpi->select_mv_arr)));
#endif

  // TODO(jingning): Reduce the actual memory use for tpl model build up.
  for (frame = 0; frame < MAX_ARF_GOP_SIZE; ++frame) {
    if (cpi->tpl_stats[frame].width >= mi_cols &&
        cpi->tpl_stats[frame].height >= mi_rows &&
        cpi->tpl_stats[frame].tpl_stats_ptr)
      continue;

#if CONFIG_NON_GREEDY_MV
    for (rf_idx = 0; rf_idx < MAX_INTER_REF_FRAMES; ++rf_idx) {
      vpx_free(cpi->tpl_stats[frame].mv_mode_arr[rf_idx]);
      CHECK_MEM_ERROR(
          &cm->error, cpi->tpl_stats[frame].mv_mode_arr[rf_idx],
          vpx_calloc(mi_rows * mi_cols * 4,
                     sizeof(*cpi->tpl_stats[frame].mv_mode_arr[rf_idx])));
      vpx_free(cpi->tpl_stats[frame].rd_diff_arr[rf_idx]);
      CHECK_MEM_ERROR(
          &cm->error, cpi->tpl_stats[frame].rd_diff_arr[rf_idx],
          vpx_calloc(mi_rows * mi_cols * 4,
                     sizeof(*cpi->tpl_stats[frame].rd_diff_arr[rf_idx])));
    }
#endif
    vpx_free(cpi->tpl_stats[frame].tpl_stats_ptr);
    CHECK_MEM_ERROR(&cm->error, cpi->tpl_stats[frame].tpl_stats_ptr,
                    vpx_calloc(mi_rows * mi_cols,
                               sizeof(*cpi->tpl_stats[frame].tpl_stats_ptr)));
    cpi->tpl_stats[frame].is_valid = 0;
    cpi->tpl_stats[frame].width = mi_cols;
    cpi->tpl_stats[frame].height = mi_rows;
    cpi->tpl_stats[frame].stride = mi_cols;
    cpi->tpl_stats[frame].mi_rows = cm->mi_rows;
    cpi->tpl_stats[frame].mi_cols = cm->mi_cols;
  }

  for (frame = 0; frame < REF_FRAMES; ++frame) {
    cpi->enc_frame_buf[frame].mem_valid = 0;
    cpi->enc_frame_buf[frame].released = 1;
  }
}

void vp9_free_tpl_buffer(VP9_COMP *cpi) {
  int frame;
#if CONFIG_NON_GREEDY_MV
  vp9_free_motion_field_info(&cpi->motion_field_info);
  vpx_free(cpi->select_mv_arr);
#endif
  for (frame = 0; frame < MAX_ARF_GOP_SIZE; ++frame) {
#if CONFIG_NON_GREEDY_MV
    int rf_idx;
    for (rf_idx = 0; rf_idx < MAX_INTER_REF_FRAMES; ++rf_idx) {
      vpx_free(cpi->tpl_stats[frame].mv_mode_arr[rf_idx]);
      vpx_free(cpi->tpl_stats[frame].rd_diff_arr[rf_idx]);
    }
#endif
    vpx_free(cpi->tpl_stats[frame].tpl_stats_ptr);
    cpi->tpl_stats[frame].is_valid = 0;
  }
  free_tpl_frame_stats_list(&cpi->tpl_gop_stats);
}

void vp9_estimate_tpl_qp_gop(VP9_COMP *cpi) {
  VP9_COMMON *cm = &cpi->common;
  int gop_length = cpi->twopass.gf_group.gf_group_size;
  int bottom_index, top_index;
  int idx;
  const int gf_index = cpi->twopass.gf_group.index;
  const int is_src_frame_alt_ref = cpi->rc.is_src_frame_alt_ref;
  const int refresh_frame_context = cpi->common.refresh_frame_context;

  const int sb_size = num_8x8_blocks_wide_lookup[BLOCK_64X64] * MI_SIZE;
  const int frame_height_sb = (cm->height + sb_size - 1) / sb_size;
  const int frame_width_sb = (cm->width + sb_size - 1) / sb_size;

  vpx_codec_err_t codec_status;
  const GF_GROUP *gf_group = &cpi->twopass.gf_group;
  vpx_rc_encodeframe_decision_t encode_frame_decision;

  CHECK_MEM_ERROR(
      &cm->error, encode_frame_decision.sb_params_list,
      (sb_params *)vpx_malloc(frame_height_sb * frame_width_sb *
                              sizeof(*encode_frame_decision.sb_params_list)));

  for (idx = gf_index; idx <= gop_length; ++idx) {
    TplDepFrame *tpl_frame = &cpi->tpl_stats[idx];
    int target_rate = cpi->twopass.gf_group.bit_allocation[idx];
    cpi->twopass.gf_group.index = idx;
    vp9_rc_set_frame_target(cpi, target_rate);
    vp9_configure_buffer_updates(cpi, idx);
    if (cpi->ext_ratectrl.ready &&
        (cpi->ext_ratectrl.funcs.rc_type & VPX_RC_QP) != 0 &&
        cpi->ext_ratectrl.funcs.get_encodeframe_decision != NULL) {
      if (idx == gop_length) break;
      memset(encode_frame_decision.sb_params_list, 0,
             sizeof(*encode_frame_decision.sb_params_list) * frame_height_sb *
                 frame_width_sb);
      codec_status = vp9_extrc_get_encodeframe_decision(
          &cpi->ext_ratectrl, gf_group->index, &encode_frame_decision);
      if (codec_status != VPX_CODEC_OK) {
        vpx_internal_error(&cm->error, codec_status,
                           "vp9_extrc_get_encodeframe_decision() failed");
      }
      for (int i = 0; i < frame_height_sb * frame_width_sb; ++i) {
        cpi->sb_mul_scale[i] =
            (((int64_t)encode_frame_decision.sb_params_list[i].rdmult * 256) /
             (encode_frame_decision.rdmult + 1));
      }
      tpl_frame->base_qindex = encode_frame_decision.q_index;
    } else {
      tpl_frame->base_qindex = vp9_rc_pick_q_and_bounds_two_pass(
          cpi, &bottom_index, &top_index, idx);
      tpl_frame->base_qindex = VPXMAX(tpl_frame->base_qindex, 1);
    }
  }
  // Reset the actual index and frame update
  cpi->twopass.gf_group.index = gf_index;
  cpi->rc.is_src_frame_alt_ref = is_src_frame_alt_ref;
  cpi->common.refresh_frame_context = refresh_frame_context;
  vp9_configure_buffer_updates(cpi, gf_index);

  vpx_free(encode_frame_decision.sb_params_list);
}

void vp9_setup_tpl_stats(VP9_COMP *cpi) {
  GF_PICTURE gf_picture_buf[MAX_ARF_GOP_SIZE + REFS_PER_FRAME];
  GF_PICTURE *gf_picture = &gf_picture_buf[REFS_PER_FRAME];
  const GF_GROUP *gf_group = &cpi->twopass.gf_group;
  int tpl_group_frames = 0;
  int frame_idx;
  int extended_frame_count;
  cpi->tpl_bsize = BLOCK_32X32;

  memset(gf_picture_buf, 0, sizeof(gf_picture_buf));
  extended_frame_count =
      init_gop_frames(cpi, gf_picture, gf_group, &tpl_group_frames);

  init_tpl_stats(cpi);

  init_tpl_stats_before_propagation(&cpi->common.error, &cpi->tpl_gop_stats,
                                    cpi->tpl_stats, tpl_group_frames,
                                    cpi->common.width, cpi->common.height);

  // Backward propagation from tpl_group_frames to 1.
  for (frame_idx = tpl_group_frames - 1; frame_idx > 0; --frame_idx) {
    if (gf_picture[frame_idx].update_type == USE_BUF_FRAME) continue;
    mc_flow_dispenser(cpi, gf_picture, frame_idx, cpi->tpl_bsize);
  }

  if (cpi->ext_ratectrl.ready &&
      cpi->ext_ratectrl.funcs.send_tpl_gop_stats != NULL) {
    // Intra search on key frame
    if (gf_group->update_type[0] != OVERLAY_UPDATE) {
      mc_flow_dispenser(cpi, gf_picture, 0, cpi->tpl_bsize);
    }
    // TPL stats has extra frames from next GOP. Trim those extra frames for
    // Qmode.
    trim_tpl_stats(&cpi->common.error, &cpi->tpl_gop_stats,
                   extended_frame_count);
    const vpx_codec_err_t codec_status =
        vp9_extrc_send_tpl_stats(&cpi->ext_ratectrl, &cpi->tpl_gop_stats);
    if (codec_status != VPX_CODEC_OK) {
      vpx_internal_error(&cpi->common.error, codec_status,
                         "vp9_extrc_send_tpl_stats() failed");
    }
  }

#if CONFIG_NON_GREEDY_MV
  cpi->tpl_ready = 1;
#if DUMP_TPL_STATS
  dump_tpl_stats(cpi, tpl_group_frames, gf_group, gf_picture, cpi->tpl_bsize);
#endif  // DUMP_TPL_STATS
#endif  // CONFIG_NON_GREEDY_MV
}
