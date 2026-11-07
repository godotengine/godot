/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <limits.h>

#include "vp9/encoder/vp9_encoder.h"
#include "vp9/encoder/vp9_speed_features.h"
#include "vp9/encoder/vp9_rdopt.h"
#include "vpx_dsp/vpx_dsp_common.h"

// Mesh search patters for various speed settings
// Define 2 mesh density levels for FC_GRAPHICS_ANIMATION content type and non
// FC_GRAPHICS_ANIMATION content type.
static MESH_PATTERN best_quality_mesh_pattern[2][MAX_MESH_STEP] = {
  { { 64, 4 }, { 28, 2 }, { 15, 1 }, { 7, 1 } },
  { { 64, 8 }, { 28, 4 }, { 15, 1 }, { 7, 1 } },
};

#if !CONFIG_REALTIME_ONLY
// Define 3 mesh density levels to control the number of searches.
#define MESH_DENSITY_LEVELS 3
static MESH_PATTERN
    good_quality_mesh_patterns[MESH_DENSITY_LEVELS][MAX_MESH_STEP] = {
      { { 64, 8 }, { 28, 4 }, { 15, 1 }, { 7, 1 } },
      { { 64, 8 }, { 14, 2 }, { 7, 1 }, { 7, 1 } },
      { { 64, 16 }, { 24, 8 }, { 12, 4 }, { 7, 1 } },
    };

// Intra only frames, golden frames (except alt ref overlays) and
// alt ref frames tend to be coded at a higher than ambient quality
static int frame_is_boosted(const VP9_COMP *cpi) {
  return frame_is_kf_gf_arf(cpi);
}

// Sets a partition size down to which the auto partition code will always
// search (can go lower), based on the image dimensions. The logic here
// is that the extent to which ringing artefacts are offensive, depends
// partly on the screen area that over which they propagate. Propagation is
// limited by transform block size but the screen area take up by a given block
// size will be larger for a small image format stretched to full screen.
static BLOCK_SIZE set_partition_min_limit(VP9_COMMON *const cm) {
  unsigned int screen_area = (cm->width * cm->height);

  // Select block size based on image format size.
  if (screen_area < 1280 * 720) {
    // Formats smaller in area than 720P
    return BLOCK_4X4;
  } else if (screen_area < 1920 * 1080) {
    // Format >= 720P and < 1080P
    return BLOCK_8X8;
  } else {
    // Formats 1080P and up
    return BLOCK_16X16;
  }
}

static void set_good_speed_feature_framesize_dependent(VP9_COMP *cpi,
                                                       SPEED_FEATURES *sf,
                                                       int speed) {
  VP9_COMMON *const cm = &cpi->common;
  const int min_frame_size = VPXMIN(cm->width, cm->height);
  const int is_480p_or_larger = min_frame_size >= 480;
  const int is_720p_or_larger = min_frame_size >= 720;
  const int is_1080p_or_larger = min_frame_size >= 1080;
  const int is_2160p_or_larger = min_frame_size >= 2160;
  const int boosted = frame_is_boosted(cpi);

  // speed 0 features
  sf->partition_search_breakout_thr.dist = (1 << 20);
  sf->partition_search_breakout_thr.rate = 80;
  sf->use_square_only_thresh_high = BLOCK_SIZES;
  sf->use_square_only_thresh_low = BLOCK_4X4;

  if (is_480p_or_larger) {
    // Currently, the machine-learning based partition search early termination
    // is only used while VPXMIN(cm->width, cm->height) >= 480 and speed = 0.
    sf->rd_ml_partition.search_early_termination = 1;
    sf->recode_tolerance_high = 45;
  } else {
    sf->use_square_only_thresh_high = BLOCK_32X32;
  }
  if (is_720p_or_larger) {
    sf->alt_ref_search_fp = 1;
  }

  if (!is_1080p_or_larger) {
    sf->rd_ml_partition.search_breakout = 1;
    if (is_720p_or_larger) {
      sf->rd_ml_partition.search_breakout_thresh[0] = 0.0f;
      sf->rd_ml_partition.search_breakout_thresh[1] = 0.0f;
      sf->rd_ml_partition.search_breakout_thresh[2] = 0.0f;
    } else {
      sf->rd_ml_partition.search_breakout_thresh[0] = 2.5f;
      sf->rd_ml_partition.search_breakout_thresh[1] = 1.5f;
      sf->rd_ml_partition.search_breakout_thresh[2] = 1.5f;
    }
  }

  if (!is_720p_or_larger) {
    if (is_480p_or_larger)
      sf->prune_single_mode_based_on_mv_diff_mode_rate = boosted ? 0 : 1;
    else
      sf->prune_single_mode_based_on_mv_diff_mode_rate = 1;
  }

  if (speed >= 1) {
    sf->rd_ml_partition.search_early_termination = 0;
    sf->rd_ml_partition.search_breakout = 1;
    if (is_480p_or_larger)
      sf->use_square_only_thresh_high = BLOCK_64X64;
    else
      sf->use_square_only_thresh_high = BLOCK_32X32;
    sf->use_square_only_thresh_low = BLOCK_16X16;
    if (is_720p_or_larger) {
      sf->disable_split_mask =
          cm->show_frame ? DISABLE_ALL_SPLIT : DISABLE_ALL_INTER_SPLIT;
      sf->partition_search_breakout_thr.dist = (1 << 22);
      sf->rd_ml_partition.search_breakout_thresh[0] = -5.0f;
      sf->rd_ml_partition.search_breakout_thresh[1] = -5.0f;
      sf->rd_ml_partition.search_breakout_thresh[2] = -9.0f;
    } else {
      sf->disable_split_mask = DISABLE_COMPOUND_SPLIT;
      sf->partition_search_breakout_thr.dist = (1 << 21);
      sf->rd_ml_partition.search_breakout_thresh[0] = -1.0f;
      sf->rd_ml_partition.search_breakout_thresh[1] = -1.0f;
      sf->rd_ml_partition.search_breakout_thresh[2] = -1.0f;
    }
#if CONFIG_VP9_HIGHBITDEPTH
    if (cpi->Source->flags & YV12_FLAG_HIGHBITDEPTH) {
      sf->rd_ml_partition.search_breakout_thresh[0] -= 1.0f;
      sf->rd_ml_partition.search_breakout_thresh[1] -= 1.0f;
      sf->rd_ml_partition.search_breakout_thresh[2] -= 1.0f;
    }
#endif  // CONFIG_VP9_HIGHBITDEPTH
  }

  if (speed >= 2) {
    sf->use_square_only_thresh_high = BLOCK_4X4;
    sf->use_square_only_thresh_low = BLOCK_SIZES;
    if (is_720p_or_larger) {
      sf->disable_split_mask =
          cm->show_frame ? DISABLE_ALL_SPLIT : DISABLE_ALL_INTER_SPLIT;
      sf->adaptive_pred_interp_filter = 0;
      sf->partition_search_breakout_thr.dist = (1 << 24);
      sf->partition_search_breakout_thr.rate = 120;
      sf->rd_ml_partition.search_breakout = 0;
    } else {
      sf->disable_split_mask = LAST_AND_INTRA_SPLIT_ONLY;
      sf->partition_search_breakout_thr.dist = (1 << 22);
      sf->partition_search_breakout_thr.rate = 100;
      sf->rd_ml_partition.search_breakout_thresh[0] = 0.0f;
      sf->rd_ml_partition.search_breakout_thresh[1] = -1.0f;
      sf->rd_ml_partition.search_breakout_thresh[2] = -4.0f;
    }
    sf->rd_auto_partition_min_limit = set_partition_min_limit(cm);

    // Use a set of speed features for 4k videos.
    if (is_2160p_or_larger) {
      sf->use_square_partition_only = 1;
      sf->intra_y_mode_mask[TX_32X32] = INTRA_DC;
      sf->intra_uv_mode_mask[TX_32X32] = INTRA_DC;
      sf->alt_ref_search_fp = 1;
      sf->cb_pred_filter_search = 2;
      sf->adaptive_interp_filter_search = 1;
      sf->disable_split_mask = DISABLE_ALL_SPLIT;
    }
  }

  if (speed >= 3) {
    sf->rd_ml_partition.search_breakout = 0;
    if (is_720p_or_larger) {
      sf->disable_split_mask = DISABLE_ALL_SPLIT;
      sf->schedule_mode_search = cm->base_qindex < 220 ? 1 : 0;
      sf->partition_search_breakout_thr.dist = (1 << 25);
      sf->partition_search_breakout_thr.rate = 200;
    } else {
      sf->max_intra_bsize = BLOCK_32X32;
      sf->disable_split_mask = DISABLE_ALL_INTER_SPLIT;
      sf->schedule_mode_search = cm->base_qindex < 175 ? 1 : 0;
      sf->partition_search_breakout_thr.dist = (1 << 23);
      sf->partition_search_breakout_thr.rate = 120;
    }
  }

  // If this is a two pass clip that fits the criteria for animated or
  // graphics content then reset disable_split_mask for speeds 1-4.
  // Also if the image edge is internal to the coded area.
  if ((speed >= 1) && (cpi->oxcf.pass == 2) &&
      ((cpi->twopass.fr_content_type == FC_GRAPHICS_ANIMATION) ||
       (vp9_internal_image_edge(cpi)))) {
    sf->disable_split_mask = DISABLE_COMPOUND_SPLIT;
  }

  if (speed >= 4) {
    sf->partition_search_breakout_thr.rate = 300;
    if (is_720p_or_larger) {
      sf->partition_search_breakout_thr.dist = (1 << 26);
    } else {
      sf->partition_search_breakout_thr.dist = (1 << 24);
    }
    sf->disable_split_mask = DISABLE_ALL_SPLIT;
  }

  if (speed >= 5) {
    sf->partition_search_breakout_thr.rate = 500;
  }
}

static double tx_dom_thresholds[6] = { 99.0, 14.0, 12.0, 8.0, 4.0, 0.0 };
static double qopt_thresholds[6] = { 99.0, 12.0, 10.0, 4.0, 2.0, 0.0 };

static void set_good_speed_feature_framesize_independent(VP9_COMP *cpi,
                                                         VP9_COMMON *cm,
                                                         SPEED_FEATURES *sf,
                                                         int speed) {
  const VP9EncoderConfig *const oxcf = &cpi->oxcf;
  const int boosted = frame_is_boosted(cpi);
  int i;

  sf->adaptive_interp_filter_search = 1;
  sf->adaptive_pred_interp_filter = 1;
  sf->adaptive_rd_thresh = 1;
  sf->adaptive_rd_thresh_row_mt = 0;
  sf->allow_skip_recode = 1;
  sf->less_rectangular_check = 1;
  sf->mv.auto_mv_step_size = 1;
  sf->mv.use_downsampled_sad = 1;
  sf->prune_ref_frame_for_rect_partitions = 1;
  sf->temporal_filter_search_method = NSTEP;
  sf->tx_size_search_breakout = 1;
  sf->use_square_partition_only = !boosted;
  sf->early_term_interp_search_plane_rd = 1;
  sf->cb_pred_filter_search = 1;
  sf->trellis_opt_tx_rd.method = sf->optimize_coefficients
                                     ? ENABLE_TRELLIS_OPT_TX_RD_RESIDUAL_MSE
                                     : DISABLE_TRELLIS_OPT;
  sf->trellis_opt_tx_rd.thresh = boosted ? 4.0 : 3.0;

  sf->intra_y_mode_mask[TX_32X32] = INTRA_DC_H_V;
  sf->comp_inter_joint_search_iter_level = 1;

  // Reference masking is not supported in dynamic scaling mode.
  sf->reference_masking = oxcf->resize_mode != RESIZE_DYNAMIC;

  sf->rd_ml_partition.var_pruning = 1;
  sf->rd_ml_partition.prune_rect_thresh[0] = -1;
  sf->rd_ml_partition.prune_rect_thresh[1] = 350;
  sf->rd_ml_partition.prune_rect_thresh[2] = 325;
  sf->rd_ml_partition.prune_rect_thresh[3] = 250;

  if (cpi->twopass.fr_content_type == FC_GRAPHICS_ANIMATION) {
    sf->exhaustive_searches_thresh = (1 << 22);
  } else {
    sf->exhaustive_searches_thresh = INT_MAX;
  }

  for (i = 0; i < MAX_MESH_STEP; ++i) {
    const int mesh_density_level = 0;
    sf->mesh_patterns[i].range =
        good_quality_mesh_patterns[mesh_density_level][i].range;
    sf->mesh_patterns[i].interval =
        good_quality_mesh_patterns[mesh_density_level][i].interval;
  }

  if (speed >= 1) {
    sf->rd_ml_partition.var_pruning = !boosted;
    sf->rd_ml_partition.prune_rect_thresh[1] = 225;
    sf->rd_ml_partition.prune_rect_thresh[2] = 225;
    sf->rd_ml_partition.prune_rect_thresh[3] = 225;

    if (oxcf->pass == 2) {
      TWO_PASS *const twopass = &cpi->twopass;
      if ((twopass->fr_content_type == FC_GRAPHICS_ANIMATION) ||
          vp9_internal_image_edge(cpi)) {
        sf->use_square_partition_only = !boosted;
      } else {
        sf->use_square_partition_only = !frame_is_intra_only(cm);
      }
    } else {
      sf->use_square_partition_only = !frame_is_intra_only(cm);
    }

    sf->allow_txfm_domain_distortion = 1;
    sf->tx_domain_thresh = tx_dom_thresholds[(speed < 6) ? speed : 5];
    sf->trellis_opt_tx_rd.method = sf->optimize_coefficients
                                       ? ENABLE_TRELLIS_OPT_TX_RD_SRC_VAR
                                       : DISABLE_TRELLIS_OPT;
    sf->trellis_opt_tx_rd.thresh = qopt_thresholds[(speed < 6) ? speed : 5];
    sf->less_rectangular_check = 1;
    sf->use_rd_breakout = 1;
    sf->adaptive_motion_search = 1;
    sf->adaptive_rd_thresh = 2;
    sf->mv.subpel_search_level = 1;
    if (cpi->oxcf.content != VP9E_CONTENT_FILM) sf->mode_skip_start = 10;
    sf->allow_acl = 0;

    sf->intra_uv_mode_mask[TX_32X32] = INTRA_DC_H_V;
    if (cpi->oxcf.content != VP9E_CONTENT_FILM) {
      sf->intra_y_mode_mask[TX_16X16] = INTRA_DC_H_V;
      sf->intra_uv_mode_mask[TX_16X16] = INTRA_DC_H_V;
    }

    sf->recode_tolerance_low = 15;
    sf->recode_tolerance_high = 30;

    sf->exhaustive_searches_thresh =
        (cpi->twopass.fr_content_type == FC_GRAPHICS_ANIMATION) ? (1 << 23)
                                                                : INT_MAX;
    sf->use_accurate_subpel_search = USE_4_TAPS;
  }

  if (speed >= 2) {
    sf->rd_ml_partition.var_pruning = 0;
    if (oxcf->vbr_corpus_complexity)
      sf->recode_loop = ALLOW_RECODE_FIRST;
    else
      sf->recode_loop = ALLOW_RECODE_KFARFGF;

    sf->tx_size_search_method =
        frame_is_boosted(cpi) ? USE_FULL_RD : USE_LARGESTALL;

    sf->mode_search_skip_flags =
        (cm->frame_type == KEY_FRAME)
            ? 0
            : FLAG_SKIP_INTRA_DIRMISMATCH | FLAG_SKIP_INTRA_BESTINTER |
                  FLAG_SKIP_COMP_BESTINTRA | FLAG_SKIP_INTRA_LOWVAR;
    sf->disable_filter_search_var_thresh = 100;
    sf->comp_inter_joint_search_iter_level = 2;
    sf->auto_min_max_partition_size = RELAXED_NEIGHBORING_MIN_MAX;
    sf->recode_tolerance_high = 45;
    sf->enhanced_full_pixel_motion_search = 0;
    sf->prune_ref_frame_for_rect_partitions = 0;
    sf->rd_ml_partition.prune_rect_thresh[1] = -1;
    sf->rd_ml_partition.prune_rect_thresh[2] = -1;
    sf->rd_ml_partition.prune_rect_thresh[3] = -1;
    sf->mv.subpel_search_level = 0;

    if (cpi->twopass.fr_content_type == FC_GRAPHICS_ANIMATION) {
      for (i = 0; i < MAX_MESH_STEP; ++i) {
        int mesh_density_level = 1;
        sf->mesh_patterns[i].range =
            good_quality_mesh_patterns[mesh_density_level][i].range;
        sf->mesh_patterns[i].interval =
            good_quality_mesh_patterns[mesh_density_level][i].interval;
      }
    }

    sf->use_accurate_subpel_search = USE_2_TAPS;
  }

  if (speed >= 3) {
    sf->use_square_partition_only = !frame_is_intra_only(cm);
    sf->tx_size_search_method =
        frame_is_intra_only(cm) ? USE_FULL_RD : USE_LARGESTALL;
    sf->mv.subpel_search_method = SUBPEL_TREE_PRUNED;
    sf->adaptive_pred_interp_filter = 0;
    sf->adaptive_mode_search = 1;
    sf->cb_partition_search = !boosted;
    sf->cb_pred_filter_search = 2;
    sf->alt_ref_search_fp = 1;
    sf->recode_loop = ALLOW_RECODE_KFMAXBW;
    sf->adaptive_rd_thresh = 3;
    sf->mode_skip_start = 6;
    sf->intra_y_mode_mask[TX_32X32] = INTRA_DC;
    sf->intra_uv_mode_mask[TX_32X32] = INTRA_DC;

    if (cpi->twopass.fr_content_type == FC_GRAPHICS_ANIMATION) {
      for (i = 0; i < MAX_MESH_STEP; ++i) {
        int mesh_density_level = 2;
        sf->mesh_patterns[i].range =
            good_quality_mesh_patterns[mesh_density_level][i].range;
        sf->mesh_patterns[i].interval =
            good_quality_mesh_patterns[mesh_density_level][i].interval;
      }
    }
  }

  if (speed >= 4) {
    sf->use_square_partition_only = 1;
    sf->tx_size_search_method = USE_LARGESTALL;
    sf->mv.search_method = BIGDIA;
    sf->mv.subpel_search_method = SUBPEL_TREE_PRUNED_MORE;
    sf->adaptive_rd_thresh = 4;
    if (cm->frame_type != KEY_FRAME)
      sf->mode_search_skip_flags |= FLAG_EARLY_TERMINATE;
    sf->disable_filter_search_var_thresh = 200;
    sf->use_lp32x32fdct = 1;
    sf->use_fast_coef_updates = ONE_LOOP_REDUCED;
    sf->use_fast_coef_costing = 1;
    sf->motion_field_mode_search = !boosted;
  }

  if (speed >= 5) {
    sf->optimize_coefficients = 0;
    sf->mv.search_method = HEX;
    sf->disable_filter_search_var_thresh = 500;
    for (i = 0; i < TX_SIZES; ++i) {
      sf->intra_y_mode_mask[i] = INTRA_DC;
      sf->intra_uv_mode_mask[i] = INTRA_DC;
    }
    sf->mv.reduce_first_step_size = 1;
    sf->simple_model_rd_from_var = 1;
  }
}
#endif  // !CONFIG_REALTIME_ONLY

static void set_rt_speed_feature_framesize_dependent(VP9_COMP *cpi,
                                                     SPEED_FEATURES *sf,
                                                     int speed) {
  VP9_COMMON *const cm = &cpi->common;

  if (speed >= 1) {
    if (VPXMIN(cm->width, cm->height) >= 720) {
      sf->disable_split_mask =
          cm->show_frame ? DISABLE_ALL_SPLIT : DISABLE_ALL_INTER_SPLIT;
    } else {
      sf->disable_split_mask = DISABLE_COMPOUND_SPLIT;
    }
  }

  if (speed >= 2) {
    if (VPXMIN(cm->width, cm->height) >= 720) {
      sf->disable_split_mask =
          cm->show_frame ? DISABLE_ALL_SPLIT : DISABLE_ALL_INTER_SPLIT;
    } else {
      sf->disable_split_mask = LAST_AND_INTRA_SPLIT_ONLY;
    }
  }

  if (speed >= 5) {
    sf->partition_search_breakout_thr.rate = 200;
    if (VPXMIN(cm->width, cm->height) >= 720) {
      sf->partition_search_breakout_thr.dist = (1 << 25);
    } else {
      sf->partition_search_breakout_thr.dist = (1 << 23);
    }
  }

  if (speed >= 7) {
    sf->encode_breakout_thresh =
        (VPXMIN(cm->width, cm->height) >= 720) ? 800 : 300;
  }
}

static void set_rt_speed_feature_framesize_independent(
    VP9_COMP *cpi, SPEED_FEATURES *sf, int speed, vp9e_tune_content content) {
  VP9_COMMON *const cm = &cpi->common;
  SVC *const svc = &cpi->svc;
  const int is_keyframe = cm->frame_type == KEY_FRAME;
  const int frames_since_key = is_keyframe ? 0 : cpi->rc.frames_since_key;
  sf->static_segmentation = 0;
  sf->adaptive_rd_thresh = 1;
  sf->adaptive_rd_thresh_row_mt = 0;
  sf->use_fast_coef_costing = 1;
  sf->exhaustive_searches_thresh = INT_MAX;
  sf->allow_acl = 0;
  sf->copy_partition_flag = 0;
  sf->use_source_sad = 0;
  sf->use_simple_block_yrd = 0;
  sf->adapt_partition_source_sad = 0;
  sf->use_altref_onepass = 0;
  sf->use_compound_nonrd_pickmode = 0;
  sf->nonrd_keyframe = 0;
  sf->svc_use_lowres_part = 0;
  sf->overshoot_detection_cbr_rt = NO_DETECTION;
  sf->disable_16x16part_nonkey = 0;
  sf->disable_golden_ref = 0;
  sf->enable_tpl_model = 0;
  sf->enhanced_full_pixel_motion_search = 0;
  sf->use_accurate_subpel_search = USE_2_TAPS;
  sf->nonrd_use_ml_partition = 0;
  sf->variance_part_thresh_mult = 1;
  sf->cb_pred_filter_search = 0;
  sf->force_smooth_interpol = 0;
  sf->rt_intra_dc_only_low_content = 0;
  sf->mv.enable_adaptive_subpel_force_stop = 0;

  if (speed >= 1) {
    sf->allow_txfm_domain_distortion = 1;
    sf->tx_domain_thresh = 0.0;
    sf->trellis_opt_tx_rd.method = DISABLE_TRELLIS_OPT;
    sf->trellis_opt_tx_rd.thresh = 0.0;
    sf->use_square_partition_only = !frame_is_intra_only(cm);
    sf->less_rectangular_check = 1;
    sf->tx_size_search_method =
        frame_is_intra_only(cm) ? USE_FULL_RD : USE_LARGESTALL;

    sf->use_rd_breakout = 1;

    sf->adaptive_motion_search = 1;
    sf->adaptive_pred_interp_filter = 1;
    sf->mv.auto_mv_step_size = 1;
    sf->adaptive_rd_thresh = 2;
    sf->intra_y_mode_mask[TX_32X32] = INTRA_DC_H_V;
    sf->intra_uv_mode_mask[TX_32X32] = INTRA_DC_H_V;
    sf->intra_uv_mode_mask[TX_16X16] = INTRA_DC_H_V;
  }

  if (speed >= 2) {
    sf->mode_search_skip_flags =
        (cm->frame_type == KEY_FRAME)
            ? 0
            : FLAG_SKIP_INTRA_DIRMISMATCH | FLAG_SKIP_INTRA_BESTINTER |
                  FLAG_SKIP_COMP_BESTINTRA | FLAG_SKIP_INTRA_LOWVAR;
    sf->adaptive_pred_interp_filter = 2;

    // Reference masking only enabled for 1 spatial layer, and if none of the
    // references have been scaled. The latter condition needs to be checked
    // for external or internal dynamic resize.
    sf->reference_masking = (svc->number_spatial_layers == 1);
    if (sf->reference_masking == 1 &&
        (cpi->external_resize == 1 ||
         cpi->oxcf.resize_mode == RESIZE_DYNAMIC)) {
      MV_REFERENCE_FRAME ref_frame;
      for (ref_frame = LAST_FRAME; ref_frame <= ALTREF_FRAME; ++ref_frame) {
        const YV12_BUFFER_CONFIG *yv12 = get_ref_frame_buffer(cpi, ref_frame);
        if (yv12 != NULL &&
            (cpi->ref_frame_flags & ref_frame_to_flag(ref_frame))) {
          const struct scale_factors *const scale_fac =
              &cm->frame_refs[ref_frame - 1].sf;
          if (vp9_is_scaled(scale_fac)) sf->reference_masking = 0;
        }
      }
    }

    sf->disable_filter_search_var_thresh = 50;
    sf->comp_inter_joint_search_iter_level = 2;
    sf->auto_min_max_partition_size = RELAXED_NEIGHBORING_MIN_MAX;
    sf->lf_motion_threshold = LOW_MOTION_THRESHOLD;
    sf->adjust_partitioning_from_last_frame = 1;
    sf->last_partitioning_redo_frequency = 3;
    sf->use_lp32x32fdct = 1;
    sf->mode_skip_start = 11;
    sf->intra_y_mode_mask[TX_16X16] = INTRA_DC_H_V;
  }

  if (speed >= 3) {
    sf->use_square_partition_only = 1;
    sf->disable_filter_search_var_thresh = 100;
    sf->use_uv_intra_rd_estimate = 1;
    sf->skip_encode_sb = 1;
    sf->mv.subpel_search_level = 0;
    sf->adaptive_rd_thresh = 4;
    sf->mode_skip_start = 6;
    sf->allow_skip_recode = 0;
    sf->optimize_coefficients = 0;
    sf->disable_split_mask = DISABLE_ALL_SPLIT;
    sf->lpf_pick = LPF_PICK_FROM_Q;
  }

  if (speed >= 4) {
    int i;
    if (cpi->oxcf.rc_mode == VPX_VBR && cpi->oxcf.lag_in_frames > 0)
      sf->use_altref_onepass = 1;
    sf->mv.subpel_force_stop = QUARTER_PEL;
    for (i = 0; i < TX_SIZES; i++) {
      sf->intra_y_mode_mask[i] = INTRA_DC_H_V;
      sf->intra_uv_mode_mask[i] = INTRA_DC;
    }
    sf->intra_y_mode_mask[TX_32X32] = INTRA_DC;
    sf->frame_parameter_update = 0;
    sf->mv.search_method = FAST_HEX;
    sf->allow_skip_recode = 0;
    sf->max_intra_bsize = BLOCK_32X32;
    sf->use_fast_coef_costing = 0;
    sf->use_quant_fp = !is_keyframe;
    sf->inter_mode_mask[BLOCK_32X32] = INTER_NEAREST_NEW_ZERO;
    sf->inter_mode_mask[BLOCK_32X64] = INTER_NEAREST_NEW_ZERO;
    sf->inter_mode_mask[BLOCK_64X32] = INTER_NEAREST_NEW_ZERO;
    sf->inter_mode_mask[BLOCK_64X64] = INTER_NEAREST_NEW_ZERO;
    sf->adaptive_rd_thresh = 2;
    sf->use_fast_coef_updates = is_keyframe ? TWO_LOOP : ONE_LOOP_REDUCED;
    sf->mode_search_skip_flags = FLAG_SKIP_INTRA_DIRMISMATCH;
    sf->tx_size_search_method = is_keyframe ? USE_LARGESTALL : USE_TX_8X8;
    sf->partition_search_type = VAR_BASED_PARTITION;
  }

  if (speed >= 5) {
    sf->use_altref_onepass = 0;
    sf->use_quant_fp = !is_keyframe;
    sf->auto_min_max_partition_size =
        is_keyframe ? RELAXED_NEIGHBORING_MIN_MAX : STRICT_NEIGHBORING_MIN_MAX;
    sf->default_max_partition_size = BLOCK_32X32;
    sf->default_min_partition_size = BLOCK_8X8;
    sf->force_frame_boost =
        is_keyframe ||
        (frames_since_key % (sf->last_partitioning_redo_frequency << 1) == 1);
    sf->max_delta_qindex = is_keyframe ? 20 : 15;
    sf->partition_search_type = REFERENCE_PARTITION;
    if (cpi->oxcf.rc_mode == VPX_VBR && cpi->oxcf.lag_in_frames > 0 &&
        cpi->rc.is_src_frame_alt_ref) {
      sf->partition_search_type = VAR_BASED_PARTITION;
    }
    sf->use_nonrd_pick_mode = 1;
    sf->allow_skip_recode = 0;
    sf->inter_mode_mask[BLOCK_32X32] = INTER_NEAREST_NEW_ZERO;
    sf->inter_mode_mask[BLOCK_32X64] = INTER_NEAREST_NEW_ZERO;
    sf->inter_mode_mask[BLOCK_64X32] = INTER_NEAREST_NEW_ZERO;
    sf->inter_mode_mask[BLOCK_64X64] = INTER_NEAREST_NEW_ZERO;
    sf->adaptive_rd_thresh = 2;
    // This feature is only enabled when partition search is disabled.
    sf->reuse_inter_pred_sby = 1;
    sf->coeff_prob_appx_step = 4;
    sf->use_fast_coef_updates = is_keyframe ? TWO_LOOP : ONE_LOOP_REDUCED;
    sf->mode_search_skip_flags = FLAG_SKIP_INTRA_DIRMISMATCH;
    sf->tx_size_search_method = is_keyframe ? USE_LARGESTALL : USE_TX_8X8;
    sf->simple_model_rd_from_var = 1;
    if (cpi->oxcf.rc_mode == VPX_VBR) sf->mv.search_method = NSTEP;

    if (!is_keyframe) {
      int i;
      if (content == VP9E_CONTENT_SCREEN) {
        for (i = 0; i < BLOCK_SIZES; ++i)
          if (i >= BLOCK_32X32)
            sf->intra_y_mode_bsize_mask[i] = INTRA_DC_H_V;
          else
            sf->intra_y_mode_bsize_mask[i] = INTRA_DC_TM_H_V;
      } else {
        for (i = 0; i < BLOCK_SIZES; ++i)
          if (i > BLOCK_16X16)
            sf->intra_y_mode_bsize_mask[i] = INTRA_DC;
          else
            // Use H and V intra mode for block sizes <= 16X16.
            sf->intra_y_mode_bsize_mask[i] = INTRA_DC_H_V;
      }
    }
    if (content == VP9E_CONTENT_SCREEN) {
      sf->short_circuit_flat_blocks = 1;
    }
    if (cpi->oxcf.rc_mode == VPX_CBR &&
        cpi->oxcf.content != VP9E_CONTENT_SCREEN) {
      sf->limit_newmv_early_exit = 1;
      if (!cpi->use_svc) sf->bias_golden = 1;
    }
    // Keep nonrd_keyframe = 1 for non-base spatial layers to prevent
    // increase in encoding time.
    if (cpi->use_svc && svc->spatial_layer_id > 0) sf->nonrd_keyframe = 1;
    if (cm->frame_type != KEY_FRAME && cpi->resize_state == ORIG &&
        cpi->oxcf.rc_mode == VPX_CBR && !cpi->rc.disable_overshoot_maxq_cbr) {
      if (cm->width * cm->height <= 352 * 288 && !cpi->use_svc &&
          cpi->oxcf.content != VP9E_CONTENT_SCREEN)
        sf->overshoot_detection_cbr_rt = RE_ENCODE_MAXQ;
      else
        sf->overshoot_detection_cbr_rt = FAST_DETECTION_MAXQ;
    }
    if (cpi->oxcf.rc_mode == VPX_VBR && cpi->oxcf.lag_in_frames > 0 &&
        cm->width <= 1280 && cm->height <= 720) {
      sf->use_altref_onepass = 1;
      sf->use_compound_nonrd_pickmode = 1;
    }
    if (cm->width * cm->height > 1280 * 720) sf->cb_pred_filter_search = 2;
    if (!cpi->external_resize) sf->use_source_sad = 1;
  }

  if (speed >= 6) {
    if (cpi->oxcf.rc_mode == VPX_VBR && cpi->oxcf.lag_in_frames > 0) {
      sf->use_altref_onepass = 1;
      sf->use_compound_nonrd_pickmode = 1;
    }
    sf->partition_search_type = VAR_BASED_PARTITION;
    sf->mv.search_method = NSTEP;
    sf->mv.reduce_first_step_size = 1;
    sf->skip_encode_sb = 0;

    if (sf->use_source_sad) {
      sf->adapt_partition_source_sad = 1;
      sf->adapt_partition_thresh =
          (cm->width * cm->height <= 640 * 360) ? 40000 : 60000;
      if (cpi->content_state_sb_fd == NULL &&
          (!cpi->use_svc ||
           svc->spatial_layer_id == svc->number_spatial_layers - 1)) {
        CHECK_MEM_ERROR(&cm->error, cpi->content_state_sb_fd,
                        (uint8_t *)vpx_calloc(
                            (cm->mi_stride >> 3) * ((cm->mi_rows >> 3) + 1),
                            sizeof(uint8_t)));
      }
    }
    if (cpi->oxcf.rc_mode == VPX_CBR && content != VP9E_CONTENT_SCREEN) {
      // Enable short circuit for low temporal variance.
      sf->short_circuit_low_temp_var = 1;
    }
    if (svc->temporal_layer_id > 0) {
      sf->adaptive_rd_thresh = 4;
      sf->limit_newmv_early_exit = 0;
      sf->base_mv_aggressive = 1;
    }
    if (cm->frame_type != KEY_FRAME && cpi->resize_state == ORIG &&
        cpi->oxcf.rc_mode == VPX_CBR && !cpi->rc.disable_overshoot_maxq_cbr)
      sf->overshoot_detection_cbr_rt = FAST_DETECTION_MAXQ;
  }

  if (speed >= 7) {
    sf->adapt_partition_source_sad = 0;
    sf->adaptive_rd_thresh = 3;
    sf->mv.search_method = FAST_DIAMOND;
    sf->mv.fullpel_search_step_param = 10;
    // For SVC: use better mv search on base temporal layer, and only
    // on base spatial layer if highest resolution is above 640x360.
    if (svc->number_temporal_layers > 2 && svc->temporal_layer_id == 0 &&
        (svc->spatial_layer_id == 0 ||
         cpi->oxcf.width * cpi->oxcf.height <= 640 * 360)) {
      sf->mv.search_method = NSTEP;
      sf->mv.fullpel_search_step_param = 6;
    }
    if (svc->temporal_layer_id > 0 || svc->spatial_layer_id > 1) {
      sf->use_simple_block_yrd = 1;
      if (svc->non_reference_frame)
        sf->mv.subpel_search_method = SUBPEL_TREE_PRUNED_EVENMORE;
    }
    if (cpi->use_svc && cpi->row_mt && cpi->oxcf.max_threads > 1)
      sf->adaptive_rd_thresh_row_mt = 1;
    // Enable partition copy. For SVC only enabled for top spatial resolution
    // layer.
    cpi->max_copied_frame = 0;
    if (!cpi->last_frame_dropped && cpi->resize_state == ORIG &&
        !cpi->external_resize &&
        (!cpi->use_svc ||
         (svc->spatial_layer_id == svc->number_spatial_layers - 1 &&
          !svc->last_layer_dropped[svc->number_spatial_layers - 1]))) {
      sf->copy_partition_flag = 1;
      cpi->max_copied_frame = 2;
      // The top temporal enhancement layer (for number of temporal layers > 1)
      // are non-reference frames, so use large/max value for max_copied_frame.
      if (svc->number_temporal_layers > 1 &&
          svc->temporal_layer_id == svc->number_temporal_layers - 1)
        cpi->max_copied_frame = 255;
    }
    // For SVC: enable use of lower resolution partition for higher resolution,
    // only for 3 spatial layers and when config/top resolution is above VGA.
    // Enable only for non-base temporal layer frames.
    if (cpi->use_svc && svc->use_partition_reuse &&
        svc->number_spatial_layers == 3 && svc->temporal_layer_id > 0 &&
        cpi->oxcf.width * cpi->oxcf.height > 640 * 480)
      sf->svc_use_lowres_part = 1;
    // For SVC when golden is used as second temporal reference: to avoid
    // encode time increase only use this feature on base temporal layer.
    // (i.e remove golden flag from frame_flags for temporal_layer_id > 0).
    if (cpi->use_svc && svc->use_gf_temporal_ref_current_layer &&
        svc->temporal_layer_id > 0)
      cpi->ref_frame_flags &= (~VP9_GOLD_FLAG);
    if (cm->width * cm->height > 640 * 480) sf->cb_pred_filter_search = 2;
  }

  if (speed >= 8) {
    sf->adaptive_rd_thresh = 4;
    sf->skip_encode_sb = 1;
    if (cpi->svc.number_spatial_layers > 1 && !cpi->svc.simulcast_mode)
      sf->nonrd_keyframe = 0;
    else
      sf->nonrd_keyframe = 1;
    if (!cpi->use_svc) cpi->max_copied_frame = 4;
    if (cpi->row_mt && cpi->oxcf.max_threads > 1)
      sf->adaptive_rd_thresh_row_mt = 1;
    // Enable ML based partition for low res.
    if (!frame_is_intra_only(cm) && cm->width * cm->height <= 352 * 288) {
      sf->nonrd_use_ml_partition = 1;
    }
#if CONFIG_VP9_HIGHBITDEPTH
    if (cpi->Source->flags & YV12_FLAG_HIGHBITDEPTH)
      sf->nonrd_use_ml_partition = 0;
#endif
    if (content == VP9E_CONTENT_SCREEN) sf->mv.subpel_force_stop = HALF_PEL;
    sf->rt_intra_dc_only_low_content = 1;
    if (!cpi->use_svc && cpi->oxcf.rc_mode == VPX_CBR &&
        content != VP9E_CONTENT_SCREEN) {
      // More aggressive short circuit for speed 8.
      sf->short_circuit_low_temp_var = 3;
      // Use level 2 for noisey cases as there is a regression in some
      // noisy clips with level 3.
      if (cpi->noise_estimate.enabled && cm->width >= 1280 &&
          cm->height >= 720) {
        NOISE_LEVEL noise_level =
            vp9_noise_estimate_extract_level(&cpi->noise_estimate);
        if (noise_level >= kMedium) sf->short_circuit_low_temp_var = 2;
      }
      // Since the short_circuit_low_temp_var is used, reduce the
      // adaptive_rd_thresh level.
      if (cm->width * cm->height > 352 * 288)
        sf->adaptive_rd_thresh = 1;
      else
        sf->adaptive_rd_thresh = 2;
    }
    sf->limit_newmv_early_exit = 0;
    sf->use_simple_block_yrd = 1;
    if (cm->width * cm->height > 352 * 288) sf->cb_pred_filter_search = 2;
  }

  if (speed >= 9) {
    // Only keep INTRA_DC mode for speed 9.
    if (!is_keyframe) {
      int i = 0;
      for (i = 0; i < BLOCK_SIZES; ++i)
        sf->intra_y_mode_bsize_mask[i] = INTRA_DC;
    }
    sf->cb_pred_filter_search = 2;
    sf->mv.enable_adaptive_subpel_force_stop = 1;
    sf->mv.adapt_subpel_force_stop.mv_thresh = 1;
    sf->mv.adapt_subpel_force_stop.force_stop_below = QUARTER_PEL;
    sf->mv.adapt_subpel_force_stop.force_stop_above = HALF_PEL;
    // Disable partition blocks below 16x16, except for low-resolutions.
    if (cm->frame_type != KEY_FRAME && cm->width >= 320 && cm->height >= 240)
      sf->disable_16x16part_nonkey = 1;
    // Allow for disabling GOLDEN reference, for CBR mode.
    if (cpi->oxcf.rc_mode == VPX_CBR) sf->disable_golden_ref = 1;
    if (cpi->rc.avg_frame_low_motion < 70) sf->default_interp_filter = BILINEAR;
    if (cm->width * cm->height >= 640 * 360) sf->variance_part_thresh_mult = 2;
  }

  // Disable split to 8x8 for low-resolution at very high Q.
  // For variance partition (speed >= 6). Ignore the first few frames
  // as avg_frame_qindex starts at max_q (worst_quality).
  if (cm->frame_type != KEY_FRAME && cm->width * cm->height <= 320 * 240 &&
      sf->partition_search_type == VAR_BASED_PARTITION &&
      cpi->rc.avg_frame_qindex[INTER_FRAME] > 208 &&
      cpi->common.current_video_frame > 8)
    sf->disable_16x16part_nonkey = 1;

  if (sf->nonrd_use_ml_partition)
    sf->partition_search_type = ML_BASED_PARTITION;

  if (sf->use_altref_onepass) {
    if (cpi->rc.is_src_frame_alt_ref && cm->frame_type != KEY_FRAME) {
      sf->partition_search_type = FIXED_PARTITION;
      sf->always_this_block_size = BLOCK_64X64;
    }
    if (cpi->count_arf_frame_usage == NULL) {
      CHECK_MEM_ERROR(
          &cm->error, cpi->count_arf_frame_usage,
          (uint8_t *)vpx_calloc((cm->mi_stride >> 3) * ((cm->mi_rows >> 3) + 1),
                                sizeof(*cpi->count_arf_frame_usage)));
    }
    if (cpi->count_lastgolden_frame_usage == NULL)
      CHECK_MEM_ERROR(
          &cm->error, cpi->count_lastgolden_frame_usage,
          (uint8_t *)vpx_calloc((cm->mi_stride >> 3) * ((cm->mi_rows >> 3) + 1),
                                sizeof(*cpi->count_lastgolden_frame_usage)));
  }
  if (svc->previous_frame_is_intra_only) {
    sf->partition_search_type = FIXED_PARTITION;
    sf->always_this_block_size = BLOCK_64X64;
  }
  // Special case for screen content: increase motion search on base spatial
  // layer when high motion is detected or previous SL0 frame was dropped.
  if (cpi->oxcf.content == VP9E_CONTENT_SCREEN && cpi->oxcf.speed >= 5 &&
      (svc->high_num_blocks_with_motion || svc->last_layer_dropped[0])) {
    sf->mv.search_method = NSTEP;
    // TODO(marpan/jianj): Tune this setting for screensharing. For now use
    // small step_param for all spatial layers.
    sf->mv.fullpel_search_step_param = 2;
  }
  // TODO(marpan): There is regression for aq-mode=3 speed <= 4, force it
  // off for now.
  if (speed <= 3 && cpi->oxcf.aq_mode == CYCLIC_REFRESH_AQ)
    cpi->oxcf.aq_mode = 0;
  // For all speeds for rt mode: if the deadline mode changed (was good/best
  // quality on previous frame and now is realtime) set nonrd_keyframe to 1 to
  // avoid entering rd pickmode. This causes issues, such as: b/310663186.
  if (cpi->oxcf.mode != cpi->deadline_mode_previous_frame)
    sf->nonrd_keyframe = 1;

  // TODO(marpan): Force this feature off always, for the issue: 366146260
  // Remove this disabling when underlying issue is resolved.
  sf->svc_use_lowres_part = 0;
}

void vp9_set_speed_features_framesize_dependent(VP9_COMP *cpi, int speed) {
  SPEED_FEATURES *const sf = &cpi->sf;
  const VP9EncoderConfig *const oxcf = &cpi->oxcf;
  RD_OPT *const rd = &cpi->rd;
  int i;

  // best quality defaults
  // Some speed-up features even for best quality as minimal impact on quality.
  sf->partition_search_breakout_thr.dist = (1 << 19);
  sf->partition_search_breakout_thr.rate = 80;
  sf->rd_ml_partition.search_early_termination = 0;
  sf->rd_ml_partition.search_breakout = 0;

  if (oxcf->mode == REALTIME)
    set_rt_speed_feature_framesize_dependent(cpi, sf, speed);
#if !CONFIG_REALTIME_ONLY
  else if (oxcf->mode == GOOD)
    set_good_speed_feature_framesize_dependent(cpi, sf, speed);
#endif

  if (sf->disable_split_mask == DISABLE_ALL_SPLIT) {
    sf->adaptive_pred_interp_filter = 0;
  }

  if (cpi->encode_breakout && oxcf->mode == REALTIME &&
      sf->encode_breakout_thresh > cpi->encode_breakout) {
    cpi->encode_breakout = sf->encode_breakout_thresh;
  }

  // Check for masked out split cases.
  for (i = 0; i < MAX_REFS; ++i) {
    if (sf->disable_split_mask & (1 << i)) {
      rd->thresh_mult_sub8x8[i] = INT_MAX;
    }
  }

  // With row based multi-threading, the following speed features
  // have to be disabled to guarantee that bitstreams encoded with single thread
  // and multiple threads match.
  // It can be used in realtime when adaptive_rd_thresh_row_mt is enabled since
  // adaptive_rd_thresh is defined per-row for non-rd pickmode.
  if (!sf->adaptive_rd_thresh_row_mt && cpi->row_mt_bit_exact &&
      oxcf->max_threads > 1)
    sf->adaptive_rd_thresh = 0;
}

void vp9_set_speed_features_framesize_independent(VP9_COMP *cpi, int speed) {
  SPEED_FEATURES *const sf = &cpi->sf;
#if !CONFIG_REALTIME_ONLY
  VP9_COMMON *const cm = &cpi->common;
#endif
  MACROBLOCK *const x = &cpi->td.mb;
  const VP9EncoderConfig *const oxcf = &cpi->oxcf;
  int i;

  // best quality defaults
  sf->frame_parameter_update = 1;
  sf->mv.search_method = NSTEP;
  sf->recode_loop = ALLOW_RECODE_FIRST;
  sf->mv.subpel_search_method = SUBPEL_TREE;
  sf->mv.subpel_search_level = 2;
  sf->mv.subpel_force_stop = EIGHTH_PEL;
  sf->optimize_coefficients = !is_lossless_requested(&cpi->oxcf);
  sf->mv.reduce_first_step_size = 0;
  sf->coeff_prob_appx_step = 1;
  sf->mv.auto_mv_step_size = 0;
  sf->mv.fullpel_search_step_param = 6;
  sf->mv.use_downsampled_sad = 0;
  sf->comp_inter_joint_search_iter_level = 0;
  sf->tx_size_search_method = USE_FULL_RD;
  sf->use_lp32x32fdct = 0;
  sf->adaptive_motion_search = 0;
  sf->enhanced_full_pixel_motion_search = 1;
  sf->adaptive_pred_interp_filter = 0;
  sf->adaptive_mode_search = 0;
  sf->prune_single_mode_based_on_mv_diff_mode_rate = 0;
  sf->cb_pred_filter_search = 0;
  sf->early_term_interp_search_plane_rd = 0;
  sf->cb_partition_search = 0;
  sf->motion_field_mode_search = 0;
  sf->alt_ref_search_fp = 0;
  sf->use_quant_fp = 0;
  sf->reference_masking = 0;
  sf->partition_search_type = SEARCH_PARTITION;
  sf->less_rectangular_check = 0;
  sf->use_square_partition_only = 0;
  sf->use_square_only_thresh_high = BLOCK_SIZES;
  sf->use_square_only_thresh_low = BLOCK_4X4;
  sf->auto_min_max_partition_size = NOT_IN_USE;
  sf->rd_auto_partition_min_limit = BLOCK_4X4;
  sf->default_max_partition_size = BLOCK_64X64;
  sf->default_min_partition_size = BLOCK_4X4;
  sf->adjust_partitioning_from_last_frame = 0;
  sf->last_partitioning_redo_frequency = 4;
  sf->disable_split_mask = 0;
  sf->mode_search_skip_flags = 0;
  sf->force_frame_boost = 0;
  sf->max_delta_qindex = 0;
  sf->disable_filter_search_var_thresh = 0;
  sf->adaptive_interp_filter_search = 0;
  sf->allow_txfm_domain_distortion = 0;
  sf->tx_domain_thresh = 99.0;
  sf->trellis_opt_tx_rd.method =
      sf->optimize_coefficients ? ENABLE_TRELLIS_OPT : DISABLE_TRELLIS_OPT;
  sf->trellis_opt_tx_rd.thresh = 99.0;
  sf->allow_acl = 1;
  sf->enable_tpl_model = oxcf->enable_tpl_model;
  sf->prune_ref_frame_for_rect_partitions = 0;
  sf->temporal_filter_search_method = MESH;
  sf->allow_skip_txfm_ac_dc = 0;

  for (i = 0; i < TX_SIZES; i++) {
    sf->intra_y_mode_mask[i] = INTRA_ALL;
    sf->intra_uv_mode_mask[i] = INTRA_ALL;
  }
  sf->use_rd_breakout = 0;
  sf->skip_encode_sb = 0;
  sf->use_uv_intra_rd_estimate = 0;
  sf->allow_skip_recode = 0;
  sf->lpf_pick = LPF_PICK_FROM_FULL_IMAGE;
  sf->use_fast_coef_updates = TWO_LOOP;
  sf->use_fast_coef_costing = 0;
  sf->mode_skip_start = MAX_MODES;  // Mode index at which mode skip mask set
  sf->schedule_mode_search = 0;
  sf->use_nonrd_pick_mode = 0;
  for (i = 0; i < BLOCK_SIZES; ++i) sf->inter_mode_mask[i] = INTER_ALL;
  sf->max_intra_bsize = BLOCK_64X64;
  sf->reuse_inter_pred_sby = 0;
  // This setting only takes effect when partition_search_type is set
  // to FIXED_PARTITION.
  sf->always_this_block_size = BLOCK_16X16;
  sf->encode_breakout_thresh = 0;
  // Recode loop tolerance %.
  sf->recode_tolerance_low = 12;
  sf->recode_tolerance_high = 25;
  sf->default_interp_filter = SWITCHABLE;
  sf->simple_model_rd_from_var = 0;
  sf->short_circuit_flat_blocks = 0;
  sf->short_circuit_low_temp_var = 0;
  sf->limit_newmv_early_exit = 0;
  sf->bias_golden = 0;
  sf->base_mv_aggressive = 0;
  sf->rd_ml_partition.prune_rect_thresh[0] = -1;
  sf->rd_ml_partition.prune_rect_thresh[1] = -1;
  sf->rd_ml_partition.prune_rect_thresh[2] = -1;
  sf->rd_ml_partition.prune_rect_thresh[3] = -1;
  sf->rd_ml_partition.var_pruning = 0;
  sf->use_accurate_subpel_search = USE_8_TAPS;

  // Some speed-up features even for best quality as minimal impact on quality.
  sf->adaptive_rd_thresh = 1;
  sf->tx_size_search_breakout = 1;
  sf->tx_size_search_depth = 2;

  sf->exhaustive_searches_thresh =
      (cpi->twopass.fr_content_type == FC_GRAPHICS_ANIMATION) ? (1 << 20)
                                                              : INT_MAX;
  {
    const int mesh_density_level =
        (cpi->twopass.fr_content_type == FC_GRAPHICS_ANIMATION) ? 0 : 1;
    for (i = 0; i < MAX_MESH_STEP; ++i) {
      sf->mesh_patterns[i].range =
          best_quality_mesh_pattern[mesh_density_level][i].range;
      sf->mesh_patterns[i].interval =
          best_quality_mesh_pattern[mesh_density_level][i].interval;
    }
  }

  if (oxcf->mode == REALTIME)
    set_rt_speed_feature_framesize_independent(cpi, sf, speed, oxcf->content);
#if !CONFIG_REALTIME_ONLY
  else if (oxcf->mode == GOOD)
    set_good_speed_feature_framesize_independent(cpi, cm, sf, speed);
#endif

  cpi->diamond_search_sad = vp9_diamond_search_sad;

  // Slow quant, dct and trellis not worthwhile for first pass
  // so make sure they are always turned off.
  if (oxcf->pass == 1) sf->optimize_coefficients = 0;

  // No recode for 1 pass.
  if (oxcf->pass == 0) {
    sf->recode_loop = DISALLOW_RECODE;
    sf->optimize_coefficients = 0;
  }

  if (sf->mv.subpel_force_stop == FULL_PEL) {
    // Whole pel only
    cpi->find_fractional_mv_step = vp9_skip_sub_pixel_tree;
  } else if (sf->mv.subpel_search_method == SUBPEL_TREE) {
    cpi->find_fractional_mv_step = vp9_find_best_sub_pixel_tree;
  } else if (sf->mv.subpel_search_method == SUBPEL_TREE_PRUNED) {
    cpi->find_fractional_mv_step = vp9_find_best_sub_pixel_tree_pruned;
  } else if (sf->mv.subpel_search_method == SUBPEL_TREE_PRUNED_MORE) {
    cpi->find_fractional_mv_step = vp9_find_best_sub_pixel_tree_pruned_more;
  } else if (sf->mv.subpel_search_method == SUBPEL_TREE_PRUNED_EVENMORE) {
    cpi->find_fractional_mv_step = vp9_find_best_sub_pixel_tree_pruned_evenmore;
  }

  // This is only used in motion vector unit test.
  if (cpi->oxcf.motion_vector_unit_test == 1)
    cpi->find_fractional_mv_step = vp9_return_max_sub_pixel_mv;
  else if (cpi->oxcf.motion_vector_unit_test == 2)
    cpi->find_fractional_mv_step = vp9_return_min_sub_pixel_mv;

  x->optimize = sf->optimize_coefficients == 1 && oxcf->pass != 1;

  x->min_partition_size = sf->default_min_partition_size;
  x->max_partition_size = sf->default_max_partition_size;

  if (!cpi->oxcf.frame_periodic_boost) {
    sf->max_delta_qindex = 0;
  }

  // With row based multi-threading, the following speed features
  // have to be disabled to guarantee that bitstreams encoded with single thread
  // and multiple threads match.
  // It can be used in realtime when adaptive_rd_thresh_row_mt is enabled since
  // adaptive_rd_thresh is defined per-row for non-rd pickmode.
  if (!sf->adaptive_rd_thresh_row_mt && cpi->row_mt_bit_exact &&
      oxcf->max_threads > 1)
    sf->adaptive_rd_thresh = 0;
}
