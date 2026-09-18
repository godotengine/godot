/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP9_ENCODER_VP9_FIRSTPASS_H_
#define VPX_VP9_ENCODER_VP9_FIRSTPASS_H_

#include <assert.h>

#include "vp9/common/vp9_onyxc_int.h"
#include "vp9/encoder/vp9_firstpass_stats.h"
#include "vp9/encoder/vp9_lookahead.h"
#include "vp9/encoder/vp9_ratectrl.h"

#ifdef __cplusplus
extern "C" {
#endif

#define INVALID_ROW (-1)

#define MAX_ARF_LAYERS 6
#define SECTION_NOISE_DEF 250.0

typedef struct {
  double frame_mb_intra_factor;
  double frame_mb_brightness_factor;
  double frame_mb_neutral_count;
} FP_MB_FLOAT_STATS;

typedef struct {
  double intra_factor;
  double brightness_factor;
  int64_t coded_error;
  int64_t sr_coded_error;
  int64_t frame_noise_energy;
  int64_t intra_error;
  int intercount;
  int second_ref_count;
  double neutral_count;
  double intra_count_low;   // Coded intra but low variance
  double intra_count_high;  // Coded intra high variance
  int intra_skip_count;
  int image_data_start_row;
  int mvcount;
  int sum_mvr;
  int sum_mvr_abs;
  int sum_mvc;
  int sum_mvc_abs;
  int64_t sum_mvrs;
  int64_t sum_mvcs;
  int sum_in_vectors;
  int intra_smooth_count;
  int new_mv_count;
} FIRSTPASS_DATA;

typedef enum {
  KF_UPDATE = 0,
  LF_UPDATE = 1,
  GF_UPDATE = 2,
  ARF_UPDATE = 3,
  OVERLAY_UPDATE = 4,
  MID_OVERLAY_UPDATE = 5,
  USE_BUF_FRAME = 6,  // Use show existing frame, no ref buffer update
  FRAME_UPDATE_TYPES = 7
} FRAME_UPDATE_TYPE;

#define FC_ANIMATION_THRESH 0.15
typedef enum {
  FC_NORMAL = 0,
  FC_GRAPHICS_ANIMATION = 1,
  FRAME_CONTENT_TYPES = 2
} FRAME_CONTENT_TYPE;

typedef struct ExternalRcReference {
  int last_index;
  int golden_index;
  int altref_index;
} ExternalRcReference;

typedef struct {
  unsigned char index;
  RATE_FACTOR_LEVEL rf_level[MAX_STATIC_GF_GROUP_LENGTH + 2];
  FRAME_UPDATE_TYPE update_type[MAX_STATIC_GF_GROUP_LENGTH + 2];
  unsigned char arf_src_offset[MAX_STATIC_GF_GROUP_LENGTH + 2];
  unsigned char layer_depth[MAX_STATIC_GF_GROUP_LENGTH + 2];
  unsigned char frame_gop_index[MAX_STATIC_GF_GROUP_LENGTH + 2];
  int bit_allocation[MAX_STATIC_GF_GROUP_LENGTH + 2];
  int gfu_boost[MAX_STATIC_GF_GROUP_LENGTH + 2];
  int update_ref_idx[MAX_STATIC_GF_GROUP_LENGTH + 2];

  int frame_start;
  int frame_end;
  // TODO(jingning): The array size of arf_stack could be reduced.
  int arf_index_stack[MAX_LAG_BUFFERS * 2];
  int top_arf_idx;
  int stack_size;
  int gf_group_size;
  int max_layer_depth;
  int allowed_max_layer_depth;
  int group_noise_energy;

  // Structure to store the reference information from external RC.
  // Used to override reference frame decisions in libvpx.
  ExternalRcReference ext_rc_ref[MAX_STATIC_GF_GROUP_LENGTH + 2];
  int ref_frame_list[MAX_STATIC_GF_GROUP_LENGTH + 2][REFS_PER_FRAME];
} GF_GROUP;

typedef struct {
  const FIRSTPASS_STATS *stats;
  int num_frames;
} FIRST_PASS_INFO;

static INLINE void fps_init_first_pass_info(FIRST_PASS_INFO *first_pass_info,
                                            const FIRSTPASS_STATS *stats,
                                            int num_frames) {
  first_pass_info->stats = stats;
  first_pass_info->num_frames = num_frames;
}

static INLINE int fps_get_num_frames(const FIRST_PASS_INFO *first_pass_info) {
  return first_pass_info->num_frames;
}

static INLINE const FIRSTPASS_STATS *fps_get_frame_stats(
    const FIRST_PASS_INFO *first_pass_info, int show_idx) {
  if (show_idx < 0 || show_idx >= first_pass_info->num_frames) {
    return NULL;
  }
  return &first_pass_info->stats[show_idx];
}

typedef struct {
  unsigned int section_intra_rating;
  unsigned int key_frame_section_intra_rating;
  FIRSTPASS_STATS total_stats;
  FIRSTPASS_STATS this_frame_stats;
  const FIRSTPASS_STATS *stats_in;
  const FIRSTPASS_STATS *stats_in_start;
  const FIRSTPASS_STATS *stats_in_end;
  FIRST_PASS_INFO first_pass_info;
  FIRSTPASS_STATS total_left_stats;
  int first_pass_done;
  int64_t bits_left;
  double mean_mod_score;
  double normalized_score_left;
  double mb_av_energy;
  double mb_smooth_pct;

  FP_MB_FLOAT_STATS *fp_mb_float_stats;

  // An indication of the content type of the current frame
  FRAME_CONTENT_TYPE fr_content_type;

  // Projected total bits available for a key frame group of frames
  int64_t kf_group_bits;

  // Error score of frames still to be coded in kf group
  double kf_group_error_left;

  double bpm_factor;
  int rolling_arf_group_target_bits;
  int rolling_arf_group_actual_bits;

  int sr_update_lag;
  int kf_zeromotion_pct;
  int last_kfgroup_zeromotion_pct;
  int active_worst_quality;
  int baseline_active_worst_quality;
  int extend_minq;
  int extend_maxq;
  int extend_minq_fast;
  int arnr_strength_adjustment;
  int last_qindex_of_arf_layer[MAX_ARF_LAYERS];

  GF_GROUP gf_group;

  // Vizeir project experimental two pass rate control parameters.
  // When |use_vizier_rc_params| is 1, the following parameters will
  // be overwritten by pass in values. Otherwise, they are initialized
  // by default values.
  int use_vizier_rc_params;
  double active_wq_factor;
  double err_per_mb;
  double sr_default_decay_limit;
  double sr_diff_factor;
  double kf_err_per_mb;
  double kf_frame_min_boost;
  double kf_frame_max_boost_first;  // Max for first kf in a chunk.
  double kf_frame_max_boost_subs;   // Max for subsequent mid chunk kfs.
  double kf_max_total_boost;
  double gf_max_total_boost;
  double gf_frame_max_boost;
  double zm_factor;
} TWO_PASS;

struct VP9_COMP;
struct ThreadData;
struct TileDataEnc;

void vp9_init_first_pass(struct VP9_COMP *cpi);
void vp9_first_pass(struct VP9_COMP *cpi, const struct lookahead_entry *source);
void vp9_end_first_pass(struct VP9_COMP *cpi);

void vp9_first_pass_encode_tile_mb_row(struct VP9_COMP *cpi,
                                       struct ThreadData *td,
                                       FIRSTPASS_DATA *fp_acc_data,
                                       struct TileDataEnc *tile_data,
                                       MV *best_ref_mv, int mb_row);

void vp9_init_second_pass(struct VP9_COMP *cpi);
void vp9_rc_get_second_pass_params(struct VP9_COMP *cpi);
void vp9_init_vizier_params(TWO_PASS *const twopass, int screen_area);

// Post encode update of the rate control parameters for 2-pass
void vp9_twopass_postencode_update(struct VP9_COMP *cpi);

void calculate_coded_size(struct VP9_COMP *cpi, int *scaled_frame_width,
                          int *scaled_frame_height);

struct VP9EncoderConfig;
int vp9_get_frames_to_next_key(const struct VP9EncoderConfig *oxcf,
                               const TWO_PASS *const twopass, int kf_show_idx,
                               int min_gf_interval);

FIRSTPASS_STATS vp9_get_frame_stats(const TWO_PASS *twopass);
FIRSTPASS_STATS vp9_get_total_stats(const TWO_PASS *twopass);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_ENCODER_VP9_FIRSTPASS_H_
