/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <math.h>

#include "vp9/encoder/vp9_aq_cyclicrefresh.h"
#include "vp9/encoder/vp9_encoder.h"
#include "vp9/encoder/vp9_svc_layercontext.h"
#include "vp9/encoder/vp9_extend.h"
#include "vpx_dsp/vpx_dsp_common.h"

#define SMALL_FRAME_WIDTH 32
#define SMALL_FRAME_HEIGHT 16

static void swap_ptr(void *a, void *b) {
  void **a_p = (void **)a;
  void **b_p = (void **)b;
  void *c = *a_p;
  *a_p = *b_p;
  *b_p = c;
}

void vp9_init_layer_context(VP9_COMP *const cpi) {
  SVC *const svc = &cpi->svc;
  const VP9EncoderConfig *const oxcf = &cpi->oxcf;
  int mi_rows = cpi->common.mi_rows;
  int mi_cols = cpi->common.mi_cols;
  int sl, tl, i;
  int alt_ref_idx = svc->number_spatial_layers;

  svc->spatial_layer_id = 0;
  svc->temporal_layer_id = 0;
  svc->force_zero_mode_spatial_ref = 0;
  svc->use_base_mv = 0;
  svc->use_partition_reuse = 0;
  svc->use_gf_temporal_ref = 1;
  svc->use_gf_temporal_ref_current_layer = 0;
  svc->scaled_temp_is_alloc = 0;
  svc->scaled_one_half = 0;
  svc->current_superframe = 0;
  svc->non_reference_frame = 0;
  svc->skip_enhancement_layer = 0;
  svc->disable_inter_layer_pred = INTER_LAYER_PRED_ON;
  svc->framedrop_mode = CONSTRAINED_LAYER_DROP;
  svc->set_intra_only_frame = 0;
  svc->previous_frame_is_intra_only = 0;
  svc->superframe_has_layer_sync = 0;
  svc->use_set_ref_frame_config = 0;
  svc->num_encoded_top_layer = 0;
  svc->simulcast_mode = 0;
  svc->single_layer_svc = 0;
  svc->resize_set = 0;

  for (i = 0; i < REF_FRAMES; ++i) {
    svc->fb_idx_spatial_layer_id[i] = 0xff;
    svc->fb_idx_temporal_layer_id[i] = 0xff;
    svc->fb_idx_base[i] = 0;
  }
  for (sl = 0; sl < oxcf->ss_number_layers; ++sl) {
    svc->last_layer_dropped[sl] = 0;
    svc->drop_spatial_layer[sl] = 0;
    svc->ext_frame_flags[sl] = 0;
    svc->lst_fb_idx[sl] = 0;
    svc->gld_fb_idx[sl] = 1;
    svc->alt_fb_idx[sl] = 2;
    svc->downsample_filter_type[sl] = BILINEAR;
    svc->downsample_filter_phase[sl] = 8;  // Set to 8 for averaging filter.
    svc->framedrop_thresh[sl] = oxcf->drop_frames_water_mark;
    svc->fb_idx_upd_tl0[sl] = -1;
    svc->drop_count[sl] = 0;
    svc->spatial_layer_sync[sl] = 0;
    svc->force_drop_constrained_from_above[sl] = 0;
  }
  svc->max_consec_drop = INT_MAX;

  svc->buffer_gf_temporal_ref[1].idx = 7;
  svc->buffer_gf_temporal_ref[0].idx = 6;
  svc->buffer_gf_temporal_ref[1].is_used = 0;
  svc->buffer_gf_temporal_ref[0].is_used = 0;

  if (cpi->oxcf.error_resilient_mode == 0 && cpi->oxcf.pass == 2) {
    if (vpx_realloc_frame_buffer(&cpi->svc.empty_frame.img, SMALL_FRAME_WIDTH,
                                 SMALL_FRAME_HEIGHT, cpi->common.subsampling_x,
                                 cpi->common.subsampling_y,
#if CONFIG_VP9_HIGHBITDEPTH
                                 cpi->common.use_highbitdepth,
#endif
                                 VP9_ENC_BORDER_IN_PIXELS,
                                 cpi->common.byte_alignment, NULL, NULL, NULL))
      vpx_internal_error(&cpi->common.error, VPX_CODEC_MEM_ERROR,
                         "Failed to allocate empty frame for multiple frame "
                         "contexts");

    memset(cpi->svc.empty_frame.img.buffer_alloc, 0x80,
           cpi->svc.empty_frame.img.buffer_alloc_sz);
  }

  for (sl = 0; sl < oxcf->ss_number_layers; ++sl) {
    for (tl = 0; tl < oxcf->ts_number_layers; ++tl) {
      int layer = LAYER_IDS_TO_IDX(sl, tl, oxcf->ts_number_layers);
      LAYER_CONTEXT *const lc = &svc->layer_context[layer];
      RATE_CONTROL *const lrc = &lc->rc;
      lc->current_video_frame_in_layer = 0;
      lc->layer_size = 0;
      lc->frames_from_key_frame = 0;
      lc->last_frame_type = FRAME_TYPES;
      lrc->ni_av_qi = oxcf->worst_allowed_q;
      lrc->total_actual_bits = 0;
      lrc->total_target_vs_actual = 0;
      lrc->ni_tot_qi = 0;
      lrc->tot_q = 0.0;
      lrc->avg_q = 0.0;
      lrc->ni_frames = 0;
      lrc->decimation_count = 0;
      lrc->decimation_factor = 0;
      lrc->worst_quality = oxcf->worst_allowed_q;
      lrc->best_quality = oxcf->best_allowed_q;

      for (i = 0; i < RATE_FACTOR_LEVELS; ++i) {
        lrc->rate_correction_factors[i] = 1.0;
      }

      if (cpi->oxcf.rc_mode == VPX_CBR) {
        lc->target_bandwidth = oxcf->layer_target_bitrate[layer];
        lrc->last_q[INTER_FRAME] = oxcf->worst_allowed_q;
        lrc->avg_frame_qindex[INTER_FRAME] = oxcf->worst_allowed_q;
        lrc->avg_frame_qindex[KEY_FRAME] = oxcf->worst_allowed_q;
      } else {
        lc->target_bandwidth = oxcf->layer_target_bitrate[layer];
        lrc->last_q[KEY_FRAME] = oxcf->best_allowed_q;
        lrc->last_q[INTER_FRAME] = oxcf->best_allowed_q;
        lrc->avg_frame_qindex[KEY_FRAME] =
            (oxcf->worst_allowed_q + oxcf->best_allowed_q) / 2;
        lrc->avg_frame_qindex[INTER_FRAME] =
            (oxcf->worst_allowed_q + oxcf->best_allowed_q) / 2;
        if (oxcf->ss_enable_auto_arf[sl])
          lc->alt_ref_idx = alt_ref_idx++;
        else
          lc->alt_ref_idx = INVALID_IDX;
        lc->gold_ref_idx = INVALID_IDX;
      }

      lrc->buffer_level =
          oxcf->starting_buffer_level_ms * lc->target_bandwidth / 1000;
      lrc->bits_off_target = lrc->buffer_level;

      // Initialize the cyclic refresh parameters. If spatial layers are used
      // (i.e., ss_number_layers > 1), these need to be updated per spatial
      // layer.
      // Cyclic refresh is only applied on base temporal layer.
      if (oxcf->ss_number_layers > 1 && tl == 0) {
        size_t last_coded_q_map_size;
        size_t consec_zero_mv_size;
        VP9_COMMON *const cm = &cpi->common;
        lc->sb_index = 0;
        lc->actual_num_seg1_blocks = 0;
        lc->actual_num_seg2_blocks = 0;
        lc->counter_encode_maxq_scene_change = 0;
        CHECK_MEM_ERROR(&cm->error, lc->map,
                        vpx_malloc(mi_rows * mi_cols * sizeof(*lc->map)));
        memset(lc->map, 0, mi_rows * mi_cols);
        last_coded_q_map_size =
            mi_rows * mi_cols * sizeof(*lc->last_coded_q_map);
        CHECK_MEM_ERROR(&cm->error, lc->last_coded_q_map,
                        vpx_malloc(last_coded_q_map_size));
        assert(MAXQ <= 255);
        memset(lc->last_coded_q_map, MAXQ, last_coded_q_map_size);
        consec_zero_mv_size = mi_rows * mi_cols * sizeof(*lc->consec_zero_mv);
        CHECK_MEM_ERROR(&cm->error, lc->consec_zero_mv,
                        vpx_malloc(consec_zero_mv_size));
        memset(lc->consec_zero_mv, 0, consec_zero_mv_size);
      }
    }
  }

  // Still have extra buffer for base layer golden frame
  if (!(svc->number_temporal_layers > 1 && cpi->oxcf.rc_mode == VPX_CBR) &&
      alt_ref_idx < REF_FRAMES)
    svc->layer_context[0].gold_ref_idx = alt_ref_idx;
}

// Update the layer context from a change_config() call.
void vp9_update_layer_context_change_config(VP9_COMP *const cpi,
                                            const int target_bandwidth) {
  SVC *const svc = &cpi->svc;
  const VP9EncoderConfig *const oxcf = &cpi->oxcf;
  const RATE_CONTROL *const rc = &cpi->rc;
  int sl, tl, layer = 0, spatial_layer_target;
  float bitrate_alloc = 1.0;
  int num_spatial_layers_nonzero_rate = 0;

  cpi->svc.temporal_layering_mode = oxcf->temporal_layering_mode;

  if (svc->temporal_layering_mode != VP9E_TEMPORAL_LAYERING_MODE_NOLAYERING) {
    for (sl = 0; sl < oxcf->ss_number_layers; ++sl) {
      for (tl = 0; tl < oxcf->ts_number_layers; ++tl) {
        layer = LAYER_IDS_TO_IDX(sl, tl, oxcf->ts_number_layers);
        svc->layer_context[layer].target_bandwidth =
            oxcf->layer_target_bitrate[layer];
      }

      layer = LAYER_IDS_TO_IDX(
          sl,
          ((oxcf->ts_number_layers - 1) < 0 ? 0 : (oxcf->ts_number_layers - 1)),
          oxcf->ts_number_layers);
      spatial_layer_target = svc->layer_context[layer].target_bandwidth =
          oxcf->layer_target_bitrate[layer];

      for (tl = 0; tl < oxcf->ts_number_layers; ++tl) {
        LAYER_CONTEXT *const lc =
            &svc->layer_context[sl * oxcf->ts_number_layers + tl];
        RATE_CONTROL *const lrc = &lc->rc;

        lc->spatial_layer_target_bandwidth = spatial_layer_target;
        if (target_bandwidth != 0) {
          bitrate_alloc = (float)lc->target_bandwidth / target_bandwidth;
        }
        lrc->starting_buffer_level =
            (int64_t)(rc->starting_buffer_level * bitrate_alloc + 0.5);
        lrc->optimal_buffer_level =
            (int64_t)(rc->optimal_buffer_level * bitrate_alloc + 0.5);
        lrc->maximum_buffer_size =
            (int64_t)(rc->maximum_buffer_size * bitrate_alloc + 0.5);
        lrc->bits_off_target =
            VPXMIN(lrc->bits_off_target, lrc->maximum_buffer_size);
        lrc->buffer_level = VPXMIN(lrc->buffer_level, lrc->maximum_buffer_size);
        lc->framerate = cpi->framerate / oxcf->ts_rate_decimator[tl];
        lrc->avg_frame_bandwidth = saturate_cast_double_to_int(
            round(lc->target_bandwidth / lc->framerate));
        lrc->max_frame_bandwidth = rc->max_frame_bandwidth;
        lrc->worst_quality = rc->worst_quality;
        lrc->best_quality = rc->best_quality;
      }
    }
  } else {
    int layer_end;

    if (svc->number_temporal_layers > 1 && cpi->oxcf.rc_mode == VPX_CBR) {
      layer_end = svc->number_temporal_layers;
    } else {
      layer_end = svc->number_spatial_layers;
    }

    for (layer = 0; layer < layer_end; ++layer) {
      LAYER_CONTEXT *const lc = &svc->layer_context[layer];
      RATE_CONTROL *const lrc = &lc->rc;

      lc->target_bandwidth = oxcf->layer_target_bitrate[layer];

      if (target_bandwidth != 0) {
        bitrate_alloc = (float)lc->target_bandwidth / target_bandwidth;
      }
      // Update buffer-related quantities.
      lrc->starting_buffer_level =
          (int64_t)(rc->starting_buffer_level * bitrate_alloc);
      lrc->optimal_buffer_level =
          (int64_t)(rc->optimal_buffer_level * bitrate_alloc);
      lrc->maximum_buffer_size =
          (int64_t)(rc->maximum_buffer_size * bitrate_alloc);
      lrc->bits_off_target =
          VPXMIN(lrc->bits_off_target, lrc->maximum_buffer_size);
      lrc->buffer_level = VPXMIN(lrc->buffer_level, lrc->maximum_buffer_size);
      // Update framerate-related quantities.
      if (svc->number_temporal_layers > 1 && cpi->oxcf.rc_mode == VPX_CBR) {
        lc->framerate = cpi->framerate / oxcf->ts_rate_decimator[layer];
      } else {
        lc->framerate = cpi->framerate;
      }
      lrc->avg_frame_bandwidth = saturate_cast_double_to_int(
          round(lc->target_bandwidth / lc->framerate));
      lrc->max_frame_bandwidth = rc->max_frame_bandwidth;
      // Update qp-related quantities.
      lrc->worst_quality = rc->worst_quality;
      lrc->best_quality = rc->best_quality;
    }
  }
  for (sl = 0; sl < oxcf->ss_number_layers; ++sl) {
    // Check bitrate of spatia layer.
    layer = LAYER_IDS_TO_IDX(sl, oxcf->ts_number_layers - 1,
                             oxcf->ts_number_layers);
    if (oxcf->layer_target_bitrate[layer] > 0)
      num_spatial_layers_nonzero_rate += 1;
  }
  if (num_spatial_layers_nonzero_rate == 1)
    svc->single_layer_svc = 1;
  else
    svc->single_layer_svc = 0;
}

static LAYER_CONTEXT *get_layer_context(VP9_COMP *const cpi) {
  if (is_one_pass_svc(cpi))
    return &cpi->svc.layer_context[cpi->svc.spatial_layer_id *
                                       cpi->svc.number_temporal_layers +
                                   cpi->svc.temporal_layer_id];
  else
    return (cpi->svc.number_temporal_layers > 1 && cpi->oxcf.rc_mode == VPX_CBR)
               ? &cpi->svc.layer_context[cpi->svc.temporal_layer_id]
               : &cpi->svc.layer_context[cpi->svc.spatial_layer_id];
}

void vp9_update_temporal_layer_framerate(VP9_COMP *const cpi) {
  SVC *const svc = &cpi->svc;
  const VP9EncoderConfig *const oxcf = &cpi->oxcf;
  LAYER_CONTEXT *const lc = get_layer_context(cpi);
  RATE_CONTROL *const lrc = &lc->rc;
  // Index into spatial+temporal arrays.
  const int st_idx = svc->spatial_layer_id * svc->number_temporal_layers +
                     svc->temporal_layer_id;
  const int tl = svc->temporal_layer_id;

  lc->framerate = cpi->framerate / oxcf->ts_rate_decimator[tl];
  lrc->avg_frame_bandwidth =
      saturate_cast_double_to_int(round(lc->target_bandwidth / lc->framerate));
  lrc->max_frame_bandwidth = cpi->rc.max_frame_bandwidth;
  // Update the average layer frame size (non-cumulative per-frame-bw).
  if (tl == 0) {
    lc->avg_frame_size = lrc->avg_frame_bandwidth;
  } else {
    const double prev_layer_framerate =
        cpi->framerate / oxcf->ts_rate_decimator[tl - 1];
    const int prev_layer_target_bandwidth =
        oxcf->layer_target_bitrate[st_idx - 1];
    lc->avg_frame_size =
        (int)round((lc->target_bandwidth - prev_layer_target_bandwidth) /
                   (lc->framerate - prev_layer_framerate));
  }
}

void vp9_update_spatial_layer_framerate(VP9_COMP *const cpi, double framerate) {
  const VP9EncoderConfig *const oxcf = &cpi->oxcf;
  LAYER_CONTEXT *const lc = get_layer_context(cpi);
  RATE_CONTROL *const lrc = &lc->rc;

  lc->framerate = framerate;
  lrc->avg_frame_bandwidth =
      saturate_cast_double_to_int(round(lc->target_bandwidth / lc->framerate));
  const int64_t vbr_min_bits =
      (int64_t)lrc->avg_frame_bandwidth * oxcf->two_pass_vbrmin_section / 100;
  lrc->min_frame_bandwidth = (int)VPXMIN(vbr_min_bits, INT_MAX);
  const int64_t vbr_max_bits =
      (int64_t)lrc->avg_frame_bandwidth * oxcf->two_pass_vbrmax_section / 100;
  lrc->max_frame_bandwidth = (int)VPXMIN(vbr_max_bits, INT_MAX);
  vp9_rc_set_gf_interval_range(cpi, lrc);
}

void vp9_restore_layer_context(VP9_COMP *const cpi) {
  LAYER_CONTEXT *const lc = get_layer_context(cpi);
  const int old_frame_since_key = cpi->rc.frames_since_key;
  const int old_frame_to_key = cpi->rc.frames_to_key;
  const int old_ext_use_post_encode_drop = cpi->rc.ext_use_post_encode_drop;

  cpi->rc = lc->rc;
  cpi->twopass = lc->twopass;
  cpi->oxcf.target_bandwidth = lc->target_bandwidth;
  cpi->alt_ref_source = lc->alt_ref_source;
  // Check if it is one_pass_cbr_svc mode and lc->speed > 0 (real-time mode
  // does not use speed = 0).
  if (is_one_pass_svc(cpi) && lc->speed > 0) {
    cpi->oxcf.speed = lc->speed;
  }
  cpi->loopfilter_ctrl = lc->loopfilter_ctrl;
  // Reset the frames_since_key and frames_to_key counters to their values
  // before the layer restore. Keep these defined for the stream (not layer).
  if (cpi->svc.number_temporal_layers > 1 ||
      cpi->svc.number_spatial_layers > 1) {
    cpi->rc.frames_since_key = old_frame_since_key;
    cpi->rc.frames_to_key = old_frame_to_key;
  }
  cpi->rc.ext_use_post_encode_drop = old_ext_use_post_encode_drop;
  // For spatial-svc, allow cyclic-refresh to be applied on the spatial layers,
  // for the base temporal layer.
  if (cpi->oxcf.aq_mode == CYCLIC_REFRESH_AQ &&
      cpi->svc.number_spatial_layers > 1 && cpi->svc.temporal_layer_id == 0) {
    CYCLIC_REFRESH *const cr = cpi->cyclic_refresh;
    swap_ptr(&cr->map, &lc->map);
    swap_ptr(&cr->last_coded_q_map, &lc->last_coded_q_map);
    swap_ptr(&cpi->consec_zero_mv, &lc->consec_zero_mv);
    cr->sb_index = lc->sb_index;
    cr->actual_num_seg1_blocks = lc->actual_num_seg1_blocks;
    cr->actual_num_seg2_blocks = lc->actual_num_seg2_blocks;
    cr->counter_encode_maxq_scene_change = lc->counter_encode_maxq_scene_change;
  }
}

void vp9_save_layer_context(VP9_COMP *const cpi) {
  const VP9EncoderConfig *const oxcf = &cpi->oxcf;
  LAYER_CONTEXT *const lc = get_layer_context(cpi);

  lc->rc = cpi->rc;
  lc->twopass = cpi->twopass;
  lc->target_bandwidth = (int)oxcf->target_bandwidth;
  lc->alt_ref_source = cpi->alt_ref_source;
  lc->frame_qp = cpi->common.base_qindex;
  lc->MBs = cpi->common.MBs;

  // For spatial-svc, allow cyclic-refresh to be applied on the spatial layers,
  // for the base temporal layer.
  if (cpi->oxcf.aq_mode == CYCLIC_REFRESH_AQ &&
      cpi->svc.number_spatial_layers > 1 && cpi->svc.temporal_layer_id == 0) {
    CYCLIC_REFRESH *const cr = cpi->cyclic_refresh;
    signed char *temp = lc->map;
    uint8_t *temp2 = lc->last_coded_q_map;
    uint8_t *temp3 = lc->consec_zero_mv;
    lc->map = cr->map;
    cr->map = temp;
    lc->last_coded_q_map = cr->last_coded_q_map;
    cr->last_coded_q_map = temp2;
    lc->consec_zero_mv = cpi->consec_zero_mv;
    cpi->consec_zero_mv = temp3;
    lc->sb_index = cr->sb_index;
    lc->actual_num_seg1_blocks = cr->actual_num_seg1_blocks;
    lc->actual_num_seg2_blocks = cr->actual_num_seg2_blocks;
    lc->counter_encode_maxq_scene_change = cr->counter_encode_maxq_scene_change;
    lc->qindex_delta[0] = cr->qindex_delta[0];
    lc->qindex_delta[1] = cr->qindex_delta[1];
    lc->qindex_delta[2] = cr->qindex_delta[2];
  }
}

#if !CONFIG_REALTIME_ONLY
void vp9_init_second_pass_spatial_svc(VP9_COMP *cpi) {
  SVC *const svc = &cpi->svc;
  int i;

  for (i = 0; i < svc->number_spatial_layers; ++i) {
    TWO_PASS *const twopass = &svc->layer_context[i].twopass;

    svc->spatial_layer_id = i;
    vp9_init_second_pass(cpi);

    twopass->total_stats.spatial_layer_id = i;
    twopass->total_left_stats.spatial_layer_id = i;
  }
  svc->spatial_layer_id = 0;
}
#endif  // !CONFIG_REALTIME_ONLY

void vp9_inc_frame_in_layer(VP9_COMP *const cpi) {
  LAYER_CONTEXT *const lc =
      &cpi->svc.layer_context[cpi->svc.spatial_layer_id *
                              cpi->svc.number_temporal_layers];
  ++lc->current_video_frame_in_layer;
  ++lc->frames_from_key_frame;
  if (cpi->svc.spatial_layer_id == cpi->svc.number_spatial_layers - 1)
    ++cpi->svc.current_superframe;
}

void get_layer_resolution(const int width_org, const int height_org,
                          const int num, const int den, int *width_out,
                          int *height_out) {
  int w, h;

  if (width_out == NULL || height_out == NULL) return;

  if (den == 0 || num == 0) {
    *width_out = width_org;
    *height_out = height_org;
    return;
  }

  w = width_org * num / den;
  h = height_org * num / den;

  // make height and width even to make chrome player happy
  w += w % 2;
  h += h % 2;

  *width_out = w;
  *height_out = h;
}

static void reset_fb_idx_unused(VP9_COMP *const cpi) {
  // If a reference frame is not referenced or refreshed, then set the
  // fb_idx for that reference to the first one used/referenced.
  // This is to avoid setting fb_idx for a reference to a slot that is not
  // used/needed (i.e., since that reference is not referenced or refreshed).
  MV_REFERENCE_FRAME ref_frame;
  MV_REFERENCE_FRAME first_ref = 0;
  int first_fb_idx = 0;
  int fb_idx[3] = { cpi->lst_fb_idx, cpi->gld_fb_idx, cpi->alt_fb_idx };
  for (ref_frame = LAST_FRAME; ref_frame <= ALTREF_FRAME; ref_frame++) {
    if (cpi->ref_frame_flags & ref_frame_to_flag(ref_frame)) {
      first_ref = ref_frame;
      first_fb_idx = fb_idx[ref_frame - 1];
      break;
    }
  }
  if (first_ref > 0) {
    if (first_ref != LAST_FRAME && !(cpi->ref_frame_flags & VP9_LAST_FLAG) &&
        !cpi->ext_refresh_last_frame)
      cpi->lst_fb_idx = first_fb_idx;
    else if (first_ref != GOLDEN_FRAME &&
             !(cpi->ref_frame_flags & VP9_GOLD_FLAG) &&
             !cpi->ext_refresh_golden_frame)
      cpi->gld_fb_idx = first_fb_idx;
    else if (first_ref != ALTREF_FRAME &&
             !(cpi->ref_frame_flags & VP9_ALT_FLAG) &&
             !cpi->ext_refresh_alt_ref_frame)
      cpi->alt_fb_idx = first_fb_idx;
  }
}

// Never refresh any reference frame buffers on top temporal layers in
// simulcast mode, which has interlayer prediction disabled.
static void non_reference_frame_simulcast(VP9_COMP *const cpi) {
  if (cpi->svc.temporal_layer_id == cpi->svc.number_temporal_layers - 1 &&
      cpi->svc.temporal_layer_id > 0) {
    cpi->ext_refresh_last_frame = 0;
    cpi->ext_refresh_golden_frame = 0;
    cpi->ext_refresh_alt_ref_frame = 0;
  }
}

// The function sets proper ref_frame_flags, buffer indices, and buffer update
// variables for temporal layering mode 3 - that does 0-2-1-2 temporal layering
// scheme.
static void set_flags_and_fb_idx_for_temporal_mode3(VP9_COMP *const cpi) {
  int frame_num_within_temporal_struct = 0;
  int spatial_id, temporal_id;
  spatial_id = cpi->svc.spatial_layer_id = cpi->svc.spatial_layer_to_encode;
  frame_num_within_temporal_struct =
      cpi->svc
          .layer_context[cpi->svc.spatial_layer_id *
                         cpi->svc.number_temporal_layers]
          .current_video_frame_in_layer %
      4;
  temporal_id = cpi->svc.temporal_layer_id =
      (frame_num_within_temporal_struct & 1)
          ? 2
          : (frame_num_within_temporal_struct >> 1);
  cpi->ext_refresh_last_frame = cpi->ext_refresh_golden_frame =
      cpi->ext_refresh_alt_ref_frame = 0;
  if (!temporal_id) {
    cpi->ext_refresh_frame_flags_pending = 1;
    cpi->ext_refresh_last_frame = 1;
    if (!spatial_id) {
      cpi->ref_frame_flags = VP9_LAST_FLAG;
    } else if (cpi->svc.layer_context[temporal_id].is_key_frame) {
      // base layer is a key frame.
      cpi->ref_frame_flags = VP9_LAST_FLAG;
      cpi->ext_refresh_last_frame = 0;
      cpi->ext_refresh_golden_frame = 1;
    } else {
      cpi->ref_frame_flags = VP9_LAST_FLAG | VP9_GOLD_FLAG;
    }
  } else if (temporal_id == 1) {
    cpi->ext_refresh_frame_flags_pending = 1;
    cpi->ext_refresh_alt_ref_frame = 1;
    if (!spatial_id) {
      cpi->ref_frame_flags = VP9_LAST_FLAG;
    } else {
      cpi->ref_frame_flags = VP9_LAST_FLAG | VP9_GOLD_FLAG;
    }
  } else {
    if (frame_num_within_temporal_struct == 1) {
      // the first tl2 picture
      if (spatial_id == cpi->svc.number_spatial_layers - 1) {  // top layer
        cpi->ext_refresh_frame_flags_pending = 1;
        if (!spatial_id)
          cpi->ref_frame_flags = VP9_LAST_FLAG;
        else
          cpi->ref_frame_flags = VP9_LAST_FLAG | VP9_GOLD_FLAG;
      } else if (!spatial_id) {
        cpi->ext_refresh_frame_flags_pending = 1;
        cpi->ext_refresh_alt_ref_frame = 1;
        cpi->ref_frame_flags = VP9_LAST_FLAG;
      } else if (spatial_id < cpi->svc.number_spatial_layers - 1) {
        cpi->ext_refresh_frame_flags_pending = 1;
        cpi->ext_refresh_alt_ref_frame = 1;
        cpi->ref_frame_flags = VP9_LAST_FLAG | VP9_GOLD_FLAG;
      }
    } else {
      //  The second tl2 picture
      if (spatial_id == cpi->svc.number_spatial_layers - 1) {  // top layer
        cpi->ext_refresh_frame_flags_pending = 1;
        if (!spatial_id)
          cpi->ref_frame_flags = VP9_LAST_FLAG;
        else
          cpi->ref_frame_flags = VP9_LAST_FLAG | VP9_GOLD_FLAG;
      } else if (!spatial_id) {
        cpi->ext_refresh_frame_flags_pending = 1;
        cpi->ref_frame_flags = VP9_LAST_FLAG;
        cpi->ext_refresh_alt_ref_frame = 1;
      } else {  // top layer
        cpi->ext_refresh_frame_flags_pending = 1;
        cpi->ref_frame_flags = VP9_LAST_FLAG | VP9_GOLD_FLAG;
        cpi->ext_refresh_alt_ref_frame = 1;
      }
    }
  }
  if (temporal_id == 0) {
    cpi->lst_fb_idx = spatial_id;
    if (spatial_id) {
      if (cpi->svc.layer_context[temporal_id].is_key_frame) {
        cpi->lst_fb_idx = spatial_id - 1;
        cpi->gld_fb_idx = spatial_id;
      } else {
        cpi->gld_fb_idx = spatial_id - 1;
      }
    } else {
      cpi->gld_fb_idx = 0;
    }
    cpi->alt_fb_idx = 0;
  } else if (temporal_id == 1) {
    cpi->lst_fb_idx = spatial_id;
    cpi->gld_fb_idx = cpi->svc.number_spatial_layers + spatial_id - 1;
    cpi->alt_fb_idx = cpi->svc.number_spatial_layers + spatial_id;
  } else if (frame_num_within_temporal_struct == 1) {
    cpi->lst_fb_idx = spatial_id;
    cpi->gld_fb_idx = cpi->svc.number_spatial_layers + spatial_id - 1;
    cpi->alt_fb_idx = cpi->svc.number_spatial_layers + spatial_id;
  } else {
    cpi->lst_fb_idx = cpi->svc.number_spatial_layers + spatial_id;
    cpi->gld_fb_idx = cpi->svc.number_spatial_layers + spatial_id - 1;
    cpi->alt_fb_idx = cpi->svc.number_spatial_layers + spatial_id;
  }

  if (cpi->svc.simulcast_mode) non_reference_frame_simulcast(cpi);

  reset_fb_idx_unused(cpi);
}

// The function sets proper ref_frame_flags, buffer indices, and buffer update
// variables for temporal layering mode 2 - that does 0-1-0-1 temporal layering
// scheme.
static void set_flags_and_fb_idx_for_temporal_mode2(VP9_COMP *const cpi) {
  int spatial_id, temporal_id;
  spatial_id = cpi->svc.spatial_layer_id = cpi->svc.spatial_layer_to_encode;
  temporal_id = cpi->svc.temporal_layer_id =
      cpi->svc
          .layer_context[cpi->svc.spatial_layer_id *
                         cpi->svc.number_temporal_layers]
          .current_video_frame_in_layer &
      1;
  cpi->ext_refresh_last_frame = cpi->ext_refresh_golden_frame =
      cpi->ext_refresh_alt_ref_frame = 0;
  if (!temporal_id) {
    cpi->ext_refresh_frame_flags_pending = 1;
    cpi->ext_refresh_last_frame = 1;
    if (!spatial_id) {
      cpi->ref_frame_flags = VP9_LAST_FLAG;
    } else if (cpi->svc.layer_context[temporal_id].is_key_frame) {
      // base layer is a key frame.
      cpi->ref_frame_flags = VP9_LAST_FLAG;
      cpi->ext_refresh_last_frame = 0;
      cpi->ext_refresh_golden_frame = 1;
    } else {
      cpi->ref_frame_flags = VP9_LAST_FLAG | VP9_GOLD_FLAG;
    }
  } else if (temporal_id == 1) {
    cpi->ext_refresh_frame_flags_pending = 1;
    cpi->ext_refresh_alt_ref_frame = 1;
    if (!spatial_id) {
      cpi->ref_frame_flags = VP9_LAST_FLAG;
    } else {
      if (spatial_id == cpi->svc.number_spatial_layers - 1)
        cpi->ext_refresh_alt_ref_frame = 0;
      cpi->ref_frame_flags = VP9_LAST_FLAG | VP9_GOLD_FLAG;
    }
  }

  if (temporal_id == 0) {
    cpi->lst_fb_idx = spatial_id;
    if (spatial_id) {
      if (cpi->svc.layer_context[temporal_id].is_key_frame) {
        cpi->lst_fb_idx = spatial_id - 1;
        cpi->gld_fb_idx = spatial_id;
      } else {
        cpi->gld_fb_idx = spatial_id - 1;
      }
    } else {
      cpi->gld_fb_idx = 0;
    }
    cpi->alt_fb_idx = 0;
  } else if (temporal_id == 1) {
    cpi->lst_fb_idx = spatial_id;
    cpi->gld_fb_idx = cpi->svc.number_spatial_layers + spatial_id - 1;
    cpi->alt_fb_idx = cpi->svc.number_spatial_layers + spatial_id;
  }

  if (cpi->svc.simulcast_mode) non_reference_frame_simulcast(cpi);

  reset_fb_idx_unused(cpi);
}

// The function sets proper ref_frame_flags, buffer indices, and buffer update
// variables for temporal layering mode 0 - that has no temporal layering.
static void set_flags_and_fb_idx_for_temporal_mode_noLayering(
    VP9_COMP *const cpi) {
  int spatial_id;
  spatial_id = cpi->svc.spatial_layer_id = cpi->svc.spatial_layer_to_encode;
  cpi->ext_refresh_last_frame = cpi->ext_refresh_golden_frame =
      cpi->ext_refresh_alt_ref_frame = 0;
  cpi->ext_refresh_frame_flags_pending = 1;
  cpi->ext_refresh_last_frame = 1;
  if (!spatial_id) {
    cpi->ref_frame_flags = VP9_LAST_FLAG;
  } else if (cpi->svc.layer_context[0].is_key_frame) {
    cpi->ref_frame_flags = VP9_LAST_FLAG;
    cpi->ext_refresh_last_frame = 0;
    cpi->ext_refresh_golden_frame = 1;
  } else {
    cpi->ref_frame_flags = VP9_LAST_FLAG | VP9_GOLD_FLAG;
  }
  cpi->lst_fb_idx = spatial_id;
  if (spatial_id) {
    if (cpi->svc.layer_context[0].is_key_frame) {
      cpi->lst_fb_idx = spatial_id - 1;
      cpi->gld_fb_idx = spatial_id;
    } else {
      cpi->gld_fb_idx = spatial_id - 1;
    }
  } else {
    cpi->gld_fb_idx = 0;
  }

  if (cpi->svc.simulcast_mode) non_reference_frame_simulcast(cpi);

  reset_fb_idx_unused(cpi);
}

static void set_flags_and_fb_idx_bypass_via_set_ref_frame_config(
    VP9_COMP *const cpi) {
  SVC *const svc = &cpi->svc;
  int sl = svc->spatial_layer_id = svc->spatial_layer_to_encode;
  cpi->svc.temporal_layer_id = cpi->svc.temporal_layer_id_per_spatial[sl];
  cpi->ext_refresh_frame_flags_pending = 1;
  cpi->lst_fb_idx = svc->lst_fb_idx[sl];
  cpi->gld_fb_idx = svc->gld_fb_idx[sl];
  cpi->alt_fb_idx = svc->alt_fb_idx[sl];
  cpi->ext_refresh_last_frame = 0;
  cpi->ext_refresh_golden_frame = 0;
  cpi->ext_refresh_alt_ref_frame = 0;
  cpi->ref_frame_flags = 0;
  if (svc->reference_last[sl]) cpi->ref_frame_flags |= VP9_LAST_FLAG;
  if (svc->reference_golden[sl]) cpi->ref_frame_flags |= VP9_GOLD_FLAG;
  if (svc->reference_altref[sl]) cpi->ref_frame_flags |= VP9_ALT_FLAG;
}

void vp9_copy_flags_ref_update_idx(VP9_COMP *const cpi) {
  SVC *const svc = &cpi->svc;
  int sl = svc->spatial_layer_id;
  svc->lst_fb_idx[sl] = cpi->lst_fb_idx;
  svc->gld_fb_idx[sl] = cpi->gld_fb_idx;
  svc->alt_fb_idx[sl] = cpi->alt_fb_idx;
  // For the fixed SVC mode: pass the refresh_lst/gld/alt_frame flags to the
  // update_buffer_slot, this is needed for the GET_SVC_REF_FRAME_CONFIG api.
  if (svc->temporal_layering_mode != VP9E_TEMPORAL_LAYERING_MODE_BYPASS) {
    int ref;
    for (ref = 0; ref < REF_FRAMES; ++ref) {
      svc->update_buffer_slot[sl] &= ~(1 << ref);
      if ((ref == svc->lst_fb_idx[sl] && cpi->refresh_last_frame) ||
          (ref == svc->gld_fb_idx[sl] && cpi->refresh_golden_frame) ||
          (ref == svc->alt_fb_idx[sl] && cpi->refresh_alt_ref_frame))
        svc->update_buffer_slot[sl] |= (1 << ref);
    }
  }

  // TODO(jianj): Remove these 3, deprecated.
  svc->update_last[sl] = (uint8_t)cpi->refresh_last_frame;
  svc->update_golden[sl] = (uint8_t)cpi->refresh_golden_frame;
  svc->update_altref[sl] = (uint8_t)cpi->refresh_alt_ref_frame;

  svc->reference_last[sl] = (uint8_t)(cpi->ref_frame_flags & VP9_LAST_FLAG);
  svc->reference_golden[sl] = (uint8_t)(cpi->ref_frame_flags & VP9_GOLD_FLAG);
  svc->reference_altref[sl] = (uint8_t)(cpi->ref_frame_flags & VP9_ALT_FLAG);
}

int vp9_one_pass_svc_start_layer(VP9_COMP *const cpi) {
  int width = 0, height = 0;
  SVC *const svc = &cpi->svc;
  LAYER_CONTEXT *lc = NULL;
  int scaling_factor_num = 1;
  int scaling_factor_den = 1;
  svc->skip_enhancement_layer = 0;

  if (svc->disable_inter_layer_pred == INTER_LAYER_PRED_OFF &&
      svc->number_spatial_layers > 1 && svc->number_spatial_layers <= 3 &&
      svc->number_temporal_layers <= 3)
    svc->simulcast_mode = 1;
  else
    svc->simulcast_mode = 0;

  if (svc->number_spatial_layers > 1) {
    svc->use_base_mv = 1;
    svc->use_partition_reuse = 1;
  }
  svc->force_zero_mode_spatial_ref = 1;

  // For constrained_from_above drop mode: before encoding superframe (i.e.,
  // at SL0 frame) check all spatial layers (starting from top) for possible
  // drop, and if so, set a flag to force drop of that layer and all its lower
  // layers.
  if (svc->spatial_layer_to_encode == svc->first_spatial_layer_to_encode) {
    int sl;
    for (sl = 0; sl < svc->number_spatial_layers; sl++)
      svc->force_drop_constrained_from_above[sl] = 0;
    if (svc->framedrop_mode == CONSTRAINED_FROM_ABOVE_DROP) {
      for (sl = svc->number_spatial_layers - 1;
           sl >= svc->first_spatial_layer_to_encode; sl--) {
        int layer = sl * svc->number_temporal_layers + svc->temporal_layer_id;
        LAYER_CONTEXT *const sl_lc = &svc->layer_context[layer];
        cpi->rc = sl_lc->rc;
        cpi->oxcf.target_bandwidth = sl_lc->target_bandwidth;
        if (vp9_test_drop(cpi)) {
          int sl2;
          // Set flag to force drop in encoding for this mode.
          for (sl2 = sl; sl2 >= svc->first_spatial_layer_to_encode; sl2--)
            svc->force_drop_constrained_from_above[sl2] = 1;
          break;
        }
      }
    }
  }

  if (svc->temporal_layering_mode == VP9E_TEMPORAL_LAYERING_MODE_0212) {
    set_flags_and_fb_idx_for_temporal_mode3(cpi);
  } else if (svc->temporal_layering_mode ==
             VP9E_TEMPORAL_LAYERING_MODE_NOLAYERING) {
    set_flags_and_fb_idx_for_temporal_mode_noLayering(cpi);
  } else if (svc->temporal_layering_mode == VP9E_TEMPORAL_LAYERING_MODE_0101) {
    set_flags_and_fb_idx_for_temporal_mode2(cpi);
  } else if (svc->temporal_layering_mode ==
                 VP9E_TEMPORAL_LAYERING_MODE_BYPASS &&
             svc->use_set_ref_frame_config) {
    set_flags_and_fb_idx_bypass_via_set_ref_frame_config(cpi);
  }

  if (cpi->lst_fb_idx == svc->buffer_gf_temporal_ref[0].idx ||
      cpi->gld_fb_idx == svc->buffer_gf_temporal_ref[0].idx ||
      cpi->alt_fb_idx == svc->buffer_gf_temporal_ref[0].idx)
    svc->buffer_gf_temporal_ref[0].is_used = 1;
  if (cpi->lst_fb_idx == svc->buffer_gf_temporal_ref[1].idx ||
      cpi->gld_fb_idx == svc->buffer_gf_temporal_ref[1].idx ||
      cpi->alt_fb_idx == svc->buffer_gf_temporal_ref[1].idx)
    svc->buffer_gf_temporal_ref[1].is_used = 1;

  // For the fixed (non-flexible/bypass) SVC mode:
  // If long term temporal reference is enabled at the sequence level
  // (use_gf_temporal_ref == 1), and inter_layer is disabled (on inter-frames),
  // we can use golden as a second temporal reference
  // (since the spatial/inter-layer reference is disabled).
  // We check that the fb_idx for this reference (buffer_gf_temporal_ref.idx) is
  // unused (slot 7 and 6 should be available for 3-3 layer system).
  // For now usage of this second temporal reference will only be used for
  // highest and next to highest spatial layer (i.e., top and middle layer for
  // 3 spatial layers).
  svc->use_gf_temporal_ref_current_layer = 0;
  if (svc->use_gf_temporal_ref && !svc->buffer_gf_temporal_ref[0].is_used &&
      !svc->buffer_gf_temporal_ref[1].is_used &&
      svc->temporal_layering_mode != VP9E_TEMPORAL_LAYERING_MODE_BYPASS &&
      svc->disable_inter_layer_pred != INTER_LAYER_PRED_ON &&
      svc->number_spatial_layers <= 3 && svc->number_temporal_layers <= 3 &&
      svc->spatial_layer_id >= svc->number_spatial_layers - 2) {
    // Enable the second (long-term) temporal reference at the frame-level.
    svc->use_gf_temporal_ref_current_layer = 1;
  }

  // Check if current superframe has any layer sync, only check once on
  // base layer.
  if (svc->spatial_layer_id == 0) {
    int sl = 0;
    // Default is no sync.
    svc->superframe_has_layer_sync = 0;
    for (sl = 0; sl < svc->number_spatial_layers; ++sl) {
      if (cpi->svc.spatial_layer_sync[sl]) svc->superframe_has_layer_sync = 1;
    }
  }

  // Reset the drop flags for all spatial layers, on the
  // first_spatial_layer_to_encode.
  if (svc->spatial_layer_id == svc->first_spatial_layer_to_encode) {
    vp9_zero(svc->drop_spatial_layer);
    // TODO(jianj/marpan): Investigate why setting svc->lst/gld/alt_fb_idx
    // causes an issue with frame dropping and temporal layers, when the frame
    // flags are passed via the encode call (bypass mode). Issue is that we're
    // resetting ext_refresh_frame_flags_pending to 0 on frame drops.
    if (svc->temporal_layering_mode != VP9E_TEMPORAL_LAYERING_MODE_BYPASS) {
      memset(&svc->lst_fb_idx, -1, sizeof(svc->lst_fb_idx));
      memset(&svc->gld_fb_idx, -1, sizeof(svc->lst_fb_idx));
      memset(&svc->alt_fb_idx, -1, sizeof(svc->lst_fb_idx));
      // These are set by API before the superframe is encoded and they are
      // passed to encoder layer by layer. Don't reset them on layer 0 in bypass
      // mode.
      vp9_zero(svc->update_buffer_slot);
      vp9_zero(svc->reference_last);
      vp9_zero(svc->reference_golden);
      vp9_zero(svc->reference_altref);
      // TODO(jianj): Remove these 3, deprecated.
      vp9_zero(svc->update_last);
      vp9_zero(svc->update_golden);
      vp9_zero(svc->update_altref);
    }
  }

  lc = &svc->layer_context[svc->spatial_layer_id * svc->number_temporal_layers +
                           svc->temporal_layer_id];

  // Setting the worst/best_quality via the encoder control: SET_SVC_PARAMETERS,
  // only for non-BYPASS mode for now.
  if (svc->temporal_layering_mode != VP9E_TEMPORAL_LAYERING_MODE_BYPASS ||
      svc->use_set_ref_frame_config) {
    RATE_CONTROL *const lrc = &lc->rc;
    lrc->worst_quality = vp9_quantizer_to_qindex(lc->max_q);
    lrc->best_quality = vp9_quantizer_to_qindex(lc->min_q);
    if (cpi->fixed_qp_onepass) {
      lrc->worst_quality = cpi->rc.worst_quality;
      lrc->best_quality = cpi->rc.best_quality;
    }
  }

  if (cpi->oxcf.resize_mode == RESIZE_DYNAMIC && svc->single_layer_svc == 1 &&
      svc->spatial_layer_id == svc->first_spatial_layer_to_encode &&
      cpi->resize_state != ORIG) {
    scaling_factor_num = lc->scaling_factor_num_resize;
    scaling_factor_den = lc->scaling_factor_den_resize;
  } else {
    scaling_factor_num = lc->scaling_factor_num;
    scaling_factor_den = lc->scaling_factor_den;
  }

  get_layer_resolution(cpi->oxcf.width, cpi->oxcf.height, scaling_factor_num,
                       scaling_factor_den, &width, &height);

  // Use Eightap_smooth for low resolutions.
  if (width * height <= 320 * 240)
    svc->downsample_filter_type[svc->spatial_layer_id] = EIGHTTAP_SMOOTH;
  // For scale factors > 0.75, set the phase to 0 (aligns decimated pixel
  // to source pixel).
  if (scaling_factor_num > (3 * scaling_factor_den) >> 2)
    svc->downsample_filter_phase[svc->spatial_layer_id] = 0;

  // The usage of use_base_mv or partition_reuse assumes down-scale of 2x2.
  // For now, turn off use of base motion vectors and partition reuse if the
  // spatial scale factors for any layers are not 2,
  // keep the case of 3 spatial layers with scale factor of 4x4 for base layer.
  // TODO(marpan): Fix this to allow for use_base_mv for scale factors != 2.
  if (svc->number_spatial_layers > 1) {
    int sl;
    for (sl = 0; sl < svc->number_spatial_layers - 1; ++sl) {
      lc = &svc->layer_context[sl * svc->number_temporal_layers +
                               svc->temporal_layer_id];
      if ((lc->scaling_factor_num != lc->scaling_factor_den >> 1) &&
          !(lc->scaling_factor_num == lc->scaling_factor_den >> 2 && sl == 0 &&
            svc->number_spatial_layers == 3)) {
        svc->use_base_mv = 0;
        svc->use_partition_reuse = 0;
        break;
      }
    }
    // For non-zero spatial layers: if the previous spatial layer was dropped
    // disable the base_mv and partition_reuse features.
    if (svc->spatial_layer_id > 0 &&
        svc->drop_spatial_layer[svc->spatial_layer_id - 1]) {
      svc->use_base_mv = 0;
      svc->use_partition_reuse = 0;
    }
  }

  svc->non_reference_frame = 0;
  if (cpi->common.frame_type != KEY_FRAME && !cpi->ext_refresh_last_frame &&
      !cpi->ext_refresh_golden_frame && !cpi->ext_refresh_alt_ref_frame)
    svc->non_reference_frame = 1;
  // For flexible mode, where update_buffer_slot is used, need to check if
  // all buffer slots are not refreshed.
  if (svc->temporal_layering_mode == VP9E_TEMPORAL_LAYERING_MODE_BYPASS) {
    if (svc->update_buffer_slot[svc->spatial_layer_id] != 0)
      svc->non_reference_frame = 0;
  }

  if (svc->spatial_layer_id == 0) {
    svc->high_source_sad_superframe = 0;
    svc->high_num_blocks_with_motion = 0;
  }

  if (svc->temporal_layering_mode != VP9E_TEMPORAL_LAYERING_MODE_BYPASS &&
      svc->last_layer_dropped[svc->spatial_layer_id] &&
      svc->fb_idx_upd_tl0[svc->spatial_layer_id] != -1 &&
      !svc->layer_context[svc->temporal_layer_id].is_key_frame) {
    // For fixed/non-flexible mode, if the previous frame (same spatial layer
    // from previous superframe) was dropped, make sure the lst_fb_idx
    // for this frame corresponds to the buffer index updated on (last) encoded
    // TL0 frame (with same spatial layer).
    cpi->lst_fb_idx = svc->fb_idx_upd_tl0[svc->spatial_layer_id];
  }

  if (vp9_set_size_literal(cpi, width, height) != 0)
    return VPX_CODEC_INVALID_PARAM;

  svc->mi_stride[svc->spatial_layer_id] = cpi->common.mi_stride;
  svc->mi_rows[svc->spatial_layer_id] = cpi->common.mi_rows;
  svc->mi_cols[svc->spatial_layer_id] = cpi->common.mi_cols;
  return 0;
}

struct lookahead_entry *vp9_svc_lookahead_pop(VP9_COMP *const cpi,
                                              struct lookahead_ctx *ctx,
                                              int drain) {
  struct lookahead_entry *buf = NULL;
  if (ctx->sz && (drain || ctx->sz == ctx->max_sz - MAX_PRE_FRAMES)) {
    buf = vp9_lookahead_peek(ctx, 0);
    if (buf != NULL) {
      // Only remove the buffer when pop the highest layer.
      if (cpi->svc.spatial_layer_id == cpi->svc.number_spatial_layers - 1) {
        vp9_lookahead_pop(ctx, drain);
      }
    }
  }
  return buf;
}

void vp9_free_svc_cyclic_refresh(VP9_COMP *const cpi) {
  int sl, tl;
  SVC *const svc = &cpi->svc;
  const VP9EncoderConfig *const oxcf = &cpi->oxcf;
  for (sl = 0; sl < oxcf->ss_number_layers; ++sl) {
    for (tl = 0; tl < oxcf->ts_number_layers; ++tl) {
      int layer = LAYER_IDS_TO_IDX(sl, tl, oxcf->ts_number_layers);
      LAYER_CONTEXT *const lc = &svc->layer_context[layer];
      if (lc->map) vpx_free(lc->map);
      if (lc->last_coded_q_map) vpx_free(lc->last_coded_q_map);
      if (lc->consec_zero_mv) vpx_free(lc->consec_zero_mv);
    }
  }
}

// Reset on key frame: reset counters, references and buffer updates.
void vp9_svc_reset_temporal_layers(VP9_COMP *const cpi, int is_key) {
  int sl, tl;
  SVC *const svc = &cpi->svc;
  LAYER_CONTEXT *lc = NULL;
  for (sl = 0; sl < svc->number_spatial_layers; ++sl) {
    for (tl = 0; tl < svc->number_temporal_layers; ++tl) {
      lc = &cpi->svc.layer_context[sl * svc->number_temporal_layers + tl];
      lc->current_video_frame_in_layer = 0;
      if (is_key) lc->frames_from_key_frame = 0;
    }
  }
  if (svc->temporal_layering_mode == VP9E_TEMPORAL_LAYERING_MODE_0212) {
    set_flags_and_fb_idx_for_temporal_mode3(cpi);
  } else if (svc->temporal_layering_mode ==
             VP9E_TEMPORAL_LAYERING_MODE_NOLAYERING) {
    set_flags_and_fb_idx_for_temporal_mode_noLayering(cpi);
  } else if (svc->temporal_layering_mode == VP9E_TEMPORAL_LAYERING_MODE_0101) {
    set_flags_and_fb_idx_for_temporal_mode2(cpi);
  }
  vp9_update_temporal_layer_framerate(cpi);
  vp9_restore_layer_context(cpi);
}

void vp9_svc_check_reset_layer_rc_flag(VP9_COMP *const cpi) {
  SVC *svc = &cpi->svc;
  int sl, tl;
  for (sl = 0; sl < svc->number_spatial_layers; ++sl) {
    // Check for reset based on avg_frame_bandwidth for spatial layer sl.
    const int spatial_layer_idx = LAYER_IDS_TO_IDX(
        sl, svc->number_temporal_layers - 1, svc->number_temporal_layers);
    LAYER_CONTEXT *lc = &svc->layer_context[spatial_layer_idx];
    RATE_CONTROL *lrc = &lc->rc;
    if (lrc->avg_frame_bandwidth / 3 > (lrc->last_avg_frame_bandwidth >> 1) ||
        lrc->avg_frame_bandwidth < (lrc->last_avg_frame_bandwidth >> 1)) {
      // Reset for all temporal layers with spatial layer sl.
      for (tl = 0; tl < svc->number_temporal_layers; ++tl) {
        int temporal_layer_idx =
            LAYER_IDS_TO_IDX(sl, tl, svc->number_temporal_layers);
        lrc = &svc->layer_context[temporal_layer_idx].rc;
        lrc->rc_1_frame = 0;
        lrc->rc_2_frame = 0;
        lrc->bits_off_target = lrc->optimal_buffer_level;
        lrc->buffer_level = lrc->optimal_buffer_level;
      }
    }
  }
}

void vp9_svc_constrain_inter_layer_pred(VP9_COMP *const cpi) {
  VP9_COMMON *const cm = &cpi->common;
  SVC *const svc = &cpi->svc;
  const int sl = svc->spatial_layer_id;
  // Check for disabling inter-layer (spatial) prediction, if
  // svc.disable_inter_layer_pred is set. If the previous spatial layer was
  // dropped then disable the prediction from this (scaled) reference.
  // For INTER_LAYER_PRED_OFF_NONKEY: inter-layer prediction is disabled
  // on key frames or if any spatial layer is a sync layer.
  if ((svc->disable_inter_layer_pred == INTER_LAYER_PRED_OFF_NONKEY &&
       !svc->layer_context[svc->temporal_layer_id].is_key_frame &&
       !svc->superframe_has_layer_sync) ||
      svc->disable_inter_layer_pred == INTER_LAYER_PRED_OFF ||
      svc->drop_spatial_layer[sl - 1]) {
    MV_REFERENCE_FRAME ref_frame;
    for (ref_frame = LAST_FRAME; ref_frame <= ALTREF_FRAME; ++ref_frame) {
      const YV12_BUFFER_CONFIG *yv12 = get_ref_frame_buffer(cpi, ref_frame);
      if (yv12 != NULL &&
          (cpi->ref_frame_flags & ref_frame_to_flag(ref_frame))) {
        const struct scale_factors *const scale_fac =
            &cm->frame_refs[ref_frame - 1].sf;
        if (vp9_is_scaled(scale_fac)) {
          cpi->ref_frame_flags &= (~ref_frame_to_flag(ref_frame));
          // Point golden/altref frame buffer index to last.
          if (!svc->simulcast_mode) {
            if (ref_frame == GOLDEN_FRAME)
              cpi->gld_fb_idx = cpi->lst_fb_idx;
            else if (ref_frame == ALTREF_FRAME)
              cpi->alt_fb_idx = cpi->lst_fb_idx;
          }
        }
      }
    }
  }
  // For fixed/non-flexible SVC: check for disabling inter-layer prediction.
  // If the reference for inter-layer prediction (the reference that is scaled)
  // is not the previous spatial layer from the same superframe, then we disable
  // inter-layer prediction. Only need to check when inter_layer prediction is
  // not set to OFF mode.
  if (svc->temporal_layering_mode != VP9E_TEMPORAL_LAYERING_MODE_BYPASS &&
      svc->disable_inter_layer_pred != INTER_LAYER_PRED_OFF) {
    // We only use LAST and GOLDEN for prediction in real-time mode, so we
    // check both here.
    MV_REFERENCE_FRAME ref_frame;
    for (ref_frame = LAST_FRAME; ref_frame <= GOLDEN_FRAME; ref_frame++) {
      struct scale_factors *scale_fac = &cm->frame_refs[ref_frame - 1].sf;
      if (vp9_is_scaled(scale_fac)) {
        // If this reference  was updated on the previous spatial layer of the
        // current superframe, then we keep this reference (don't disable).
        // Otherwise we disable the inter-layer prediction.
        // This condition is verified by checking if the current frame buffer
        // index is equal to any of the slots for the previous spatial layer,
        // and if so, check if that slot was updated/refreshed. If that is the
        // case, then this reference is valid for inter-layer prediction under
        // the mode INTER_LAYER_PRED_ON_CONSTRAINED.
        int fb_idx =
            ref_frame == LAST_FRAME ? cpi->lst_fb_idx : cpi->gld_fb_idx;
        int ref_flag = ref_frame == LAST_FRAME ? VP9_LAST_FLAG : VP9_GOLD_FLAG;
        int disable = 1;
        if (fb_idx < 0) continue;
        if ((fb_idx == svc->lst_fb_idx[sl - 1] &&
             (svc->update_buffer_slot[sl - 1] & (1 << fb_idx))) ||
            (fb_idx == svc->gld_fb_idx[sl - 1] &&
             (svc->update_buffer_slot[sl - 1] & (1 << fb_idx))) ||
            (fb_idx == svc->alt_fb_idx[sl - 1] &&
             (svc->update_buffer_slot[sl - 1] & (1 << fb_idx))))
          disable = 0;
        if (disable) cpi->ref_frame_flags &= (~ref_flag);
      }
    }
  }
}

void vp9_svc_assert_constraints_pattern(VP9_COMP *const cpi) {
  SVC *const svc = &cpi->svc;
  // For fixed/non-flexible mode, the following constraint are expected,
  // when inter-layer prediction is on (default).
  if (svc->temporal_layering_mode != VP9E_TEMPORAL_LAYERING_MODE_BYPASS &&
      svc->disable_inter_layer_pred == INTER_LAYER_PRED_ON &&
      svc->framedrop_mode != LAYER_DROP) {
    if (!svc->layer_context[svc->temporal_layer_id].is_key_frame) {
      // On non-key frames: LAST is always temporal reference, GOLDEN is
      // spatial reference.
      if (svc->temporal_layer_id == 0)
        // Base temporal only predicts from base temporal.
        assert(svc->fb_idx_temporal_layer_id[cpi->lst_fb_idx] == 0);
      else
        // Non-base temporal only predicts from lower temporal layer.
        assert(svc->fb_idx_temporal_layer_id[cpi->lst_fb_idx] <
               svc->temporal_layer_id);
      if (svc->spatial_layer_id > 0 && cpi->ref_frame_flags & VP9_GOLD_FLAG &&
          svc->spatial_layer_id > svc->first_spatial_layer_to_encode) {
        // Non-base spatial only predicts from lower spatial layer with same
        // temporal_id.
        assert(svc->fb_idx_spatial_layer_id[cpi->gld_fb_idx] ==
               svc->spatial_layer_id - 1);
        assert(svc->fb_idx_temporal_layer_id[cpi->gld_fb_idx] ==
               svc->temporal_layer_id);
      }
    } else if (svc->spatial_layer_id > 0 &&
               svc->spatial_layer_id > svc->first_spatial_layer_to_encode) {
      // Only 1 reference for frame whose base is key; reference may be LAST
      // or GOLDEN, so we check both.
      if (cpi->ref_frame_flags & VP9_LAST_FLAG) {
        assert(svc->fb_idx_spatial_layer_id[cpi->lst_fb_idx] ==
               svc->spatial_layer_id - 1);
        assert(svc->fb_idx_temporal_layer_id[cpi->lst_fb_idx] ==
               svc->temporal_layer_id);
      } else if (cpi->ref_frame_flags & VP9_GOLD_FLAG) {
        assert(svc->fb_idx_spatial_layer_id[cpi->gld_fb_idx] ==
               svc->spatial_layer_id - 1);
        assert(svc->fb_idx_temporal_layer_id[cpi->gld_fb_idx] ==
               svc->temporal_layer_id);
      }
    }
  } else if (svc->use_gf_temporal_ref_current_layer &&
             !svc->layer_context[svc->temporal_layer_id].is_key_frame) {
    // For the usage of golden as second long term reference: the
    // temporal_layer_id of that reference must be base temporal layer 0, and
    // spatial_layer_id of that reference must be same as current
    // spatial_layer_id. If not, disable feature.
    // TODO(marpan): Investigate when this can happen, and maybe put this check
    // and reset in a different place.
    if (svc->fb_idx_spatial_layer_id[cpi->gld_fb_idx] !=
            svc->spatial_layer_id ||
        svc->fb_idx_temporal_layer_id[cpi->gld_fb_idx] != 0)
      svc->use_gf_temporal_ref_current_layer = 0;
  }
}

#if CONFIG_VP9_TEMPORAL_DENOISING
int vp9_denoise_svc_non_key(VP9_COMP *const cpi) {
  int layer =
      LAYER_IDS_TO_IDX(cpi->svc.spatial_layer_id, cpi->svc.temporal_layer_id,
                       cpi->svc.number_temporal_layers);
  LAYER_CONTEXT *lc = &cpi->svc.layer_context[layer];
  return denoise_svc(cpi) && !lc->is_key_frame;
}
#endif

void vp9_svc_check_spatial_layer_sync(VP9_COMP *const cpi) {
  SVC *const svc = &cpi->svc;
  // Only for superframes whose base is not key, as those are
  // already sync frames.
  if (!svc->layer_context[svc->temporal_layer_id].is_key_frame) {
    if (svc->spatial_layer_id == 0) {
      // On base spatial layer: if the current superframe has a layer sync then
      // reset the pattern counters and reset to base temporal layer.
      if (svc->superframe_has_layer_sync)
        vp9_svc_reset_temporal_layers(cpi, cpi->common.frame_type == KEY_FRAME);
    }
    // If the layer sync is set for this current spatial layer then
    // disable the temporal reference.
    if (svc->spatial_layer_id > 0 &&
        svc->spatial_layer_sync[svc->spatial_layer_id]) {
      cpi->ref_frame_flags &= (~VP9_LAST_FLAG);
      if (svc->use_gf_temporal_ref_current_layer) {
        int index = svc->spatial_layer_id;
        // If golden is used as second reference: need to remove it from
        // prediction, reset refresh period to 0, and update the reference.
        svc->use_gf_temporal_ref_current_layer = 0;
        cpi->rc.baseline_gf_interval = 0;
        cpi->rc.frames_till_gf_update_due = 0;
        // On layer sync frame we must update the buffer index used for long
        // term reference. Use the alt_ref since it is not used or updated on
        // sync frames.
        if (svc->number_spatial_layers == 3) index = svc->spatial_layer_id - 1;
        assert(index >= 0);
        cpi->alt_fb_idx = svc->buffer_gf_temporal_ref[index].idx;
        cpi->ext_refresh_alt_ref_frame = 1;
      }
    }
  }
}

void vp9_svc_update_ref_frame_buffer_idx(VP9_COMP *const cpi) {
  SVC *const svc = &cpi->svc;
  int i = 0;
  // Update the usage of frame buffer index for base spatial layers.
  if (svc->spatial_layer_id == 0) {
    if ((cpi->ref_frame_flags & VP9_LAST_FLAG) || cpi->refresh_last_frame)
      svc->fb_idx_base[cpi->lst_fb_idx] = 1;
    if ((cpi->ref_frame_flags & VP9_GOLD_FLAG) || cpi->refresh_golden_frame)
      svc->fb_idx_base[cpi->gld_fb_idx] = 1;
    if ((cpi->ref_frame_flags & VP9_ALT_FLAG) || cpi->refresh_alt_ref_frame)
      svc->fb_idx_base[cpi->alt_fb_idx] = 1;
    // For bypass/flexible mode: check for refresh slots.
    if (svc->temporal_layering_mode == VP9E_TEMPORAL_LAYERING_MODE_BYPASS) {
      for (i = 0; i < REF_FRAMES; ++i)
        if (svc->update_buffer_slot[0] & (1 << i)) svc->fb_idx_base[i] = 1;
    }
  }
}

static void vp9_svc_update_ref_frame_bypass_mode(VP9_COMP *const cpi) {
  // For non-flexible/bypass SVC mode: check for refreshing other buffer
  // slots.
  SVC *const svc = &cpi->svc;
  VP9_COMMON *const cm = &cpi->common;
  BufferPool *const pool = cm->buffer_pool;
  int i;
  for (i = 0; i < REF_FRAMES; i++) {
    if ((cm->frame_type == KEY_FRAME && !svc->simulcast_mode) ||
        svc->update_buffer_slot[svc->spatial_layer_id] & (1 << i)) {
      ref_cnt_fb(pool->frame_bufs, &cm->ref_frame_map[i], cm->new_fb_idx);
      svc->fb_idx_spatial_layer_id[i] = svc->spatial_layer_id;
      svc->fb_idx_temporal_layer_id[i] = svc->temporal_layer_id;
    }
  }
}

void vp9_svc_update_ref_frame(VP9_COMP *const cpi) {
  VP9_COMMON *const cm = &cpi->common;
  SVC *const svc = &cpi->svc;
  BufferPool *const pool = cm->buffer_pool;

  if (svc->temporal_layering_mode == VP9E_TEMPORAL_LAYERING_MODE_BYPASS &&
      svc->use_set_ref_frame_config) {
    vp9_svc_update_ref_frame_bypass_mode(cpi);
  } else if (cm->frame_type == KEY_FRAME && !svc->simulcast_mode) {
    // Keep track of frame index for each reference frame.
    int i;
    // On key frame update all reference frame slots.
    for (i = 0; i < REF_FRAMES; i++) {
      svc->fb_idx_spatial_layer_id[i] = svc->spatial_layer_id;
      svc->fb_idx_temporal_layer_id[i] = svc->temporal_layer_id;
      // LAST/GOLDEN/ALTREF is already updated above.
      if (i != cpi->lst_fb_idx && i != cpi->gld_fb_idx && i != cpi->alt_fb_idx)
        ref_cnt_fb(pool->frame_bufs, &cm->ref_frame_map[i], cm->new_fb_idx);
    }
  } else {
    if (cpi->refresh_last_frame) {
      svc->fb_idx_spatial_layer_id[cpi->lst_fb_idx] = svc->spatial_layer_id;
      svc->fb_idx_temporal_layer_id[cpi->lst_fb_idx] = svc->temporal_layer_id;
    }
    if (cpi->refresh_golden_frame) {
      svc->fb_idx_spatial_layer_id[cpi->gld_fb_idx] = svc->spatial_layer_id;
      svc->fb_idx_temporal_layer_id[cpi->gld_fb_idx] = svc->temporal_layer_id;
    }
    if (cpi->refresh_alt_ref_frame) {
      svc->fb_idx_spatial_layer_id[cpi->alt_fb_idx] = svc->spatial_layer_id;
      svc->fb_idx_temporal_layer_id[cpi->alt_fb_idx] = svc->temporal_layer_id;
    }
  }
  // Copy flags from encoder to SVC struct.
  vp9_copy_flags_ref_update_idx(cpi);
  vp9_svc_update_ref_frame_buffer_idx(cpi);
}

void vp9_svc_adjust_frame_rate(VP9_COMP *const cpi) {
  int64_t this_duration =
      cpi->svc.timebase_fac * cpi->svc.duration[cpi->svc.spatial_layer_id];
  vp9_new_framerate(cpi, 10000000.0 / this_duration);
}

void vp9_svc_adjust_avg_frame_qindex(VP9_COMP *const cpi) {
  VP9_COMMON *const cm = &cpi->common;
  SVC *const svc = &cpi->svc;
  RATE_CONTROL *const rc = &cpi->rc;
  // On key frames in CBR mode: reset the avg_frame_qindex for base layer
  // (to level closer to worst_quality) if the overshoot is significant.
  // Reset it for all temporal layers on base spatial layer.
  if (cm->frame_type == KEY_FRAME && cpi->oxcf.rc_mode == VPX_CBR &&
      !svc->simulcast_mode &&
      rc->projected_frame_size / 3 > rc->avg_frame_bandwidth) {
    int tl;
    rc->avg_frame_qindex[INTER_FRAME] =
        VPXMAX(rc->avg_frame_qindex[INTER_FRAME],
               (cm->base_qindex + rc->worst_quality) >> 1);
    for (tl = 0; tl < svc->number_temporal_layers; ++tl) {
      const int layer = LAYER_IDS_TO_IDX(0, tl, svc->number_temporal_layers);
      LAYER_CONTEXT *lc = &svc->layer_context[layer];
      RATE_CONTROL *lrc = &lc->rc;
      lrc->avg_frame_qindex[INTER_FRAME] = rc->avg_frame_qindex[INTER_FRAME];
    }
  }
}

// SVC: skip encoding of enhancement layer if the layer target bandwidth = 0.
// No need to set svc.skip_enhancement_layer if whole superframe will be
// dropped.
int vp9_svc_check_skip_enhancement_layer(VP9_COMP *const cpi) {
  if (cpi->use_svc && cpi->svc.spatial_layer_id > 0 &&
      cpi->oxcf.target_bandwidth == 0 &&
      !(cpi->svc.framedrop_mode != LAYER_DROP &&
        (cpi->svc.framedrop_mode != CONSTRAINED_FROM_ABOVE_DROP ||
         cpi->svc
             .force_drop_constrained_from_above[cpi->svc.number_spatial_layers -
                                                1]) &&
        cpi->svc.drop_spatial_layer[0])) {
    cpi->svc.skip_enhancement_layer = 1;
    vp9_rc_postencode_update_drop_frame(cpi);
    cpi->ext_refresh_frame_flags_pending = 0;
    cpi->last_frame_dropped = 1;
    cpi->svc.last_layer_dropped[cpi->svc.spatial_layer_id] = 1;
    cpi->svc.drop_spatial_layer[cpi->svc.spatial_layer_id] = 1;
    vp9_inc_frame_in_layer(cpi);
    return 1;
  }
  return 0;
}
