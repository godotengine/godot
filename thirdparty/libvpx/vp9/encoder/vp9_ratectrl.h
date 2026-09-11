/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP9_ENCODER_VP9_RATECTRL_H_
#define VPX_VP9_ENCODER_VP9_RATECTRL_H_

#include "vpx/vpx_codec.h"
#include "vpx/vpx_integer.h"

#include "vp9/common/vp9_blockd.h"
#include "vp9/encoder/vp9_lookahead.h"

#ifdef __cplusplus
extern "C" {
#endif

// Used to control aggressive VBR mode.
// #define AGGRESSIVE_VBR 1

// Bits Per MB at different Q (Multiplied by 512)
#define BPER_MB_NORMBITS 9

#define DEFAULT_KF_BOOST 2000
#define DEFAULT_GF_BOOST 2000

#define MIN_GF_INTERVAL 4
#define MAX_GF_INTERVAL 16
#define FIXED_GF_INTERVAL 8  // Used in some testing modes only
#define ONEHALFONLY_RESIZE 0

#define FRAME_OVERHEAD_BITS 200

// Threshold used to define a KF group as static (e.g. a slide show).
// Essentially this means that no frame in the group has more than 1% of MBs
// that are not marked as coded with 0,0 motion in the first pass.
#define STATIC_KF_GROUP_THRESH 99

// The maximum duration of a GF group that is static (for example a slide show).
#define MAX_STATIC_GF_GROUP_LENGTH 250

typedef enum {
  INTER_NORMAL = 0,
  INTER_HIGH = 1,
  GF_ARF_LOW = 2,
  GF_ARF_STD = 3,
  KF_STD = 4,
  RATE_FACTOR_LEVELS = 5
} RATE_FACTOR_LEVEL;

// Internal frame scaling level.
typedef enum {
  UNSCALED = 0,     // Frame is unscaled.
  SCALE_STEP1 = 1,  // First-level down-scaling.
  FRAME_SCALE_STEPS
} FRAME_SCALE_LEVEL;

typedef enum {
  NO_RESIZE = 0,
  DOWN_THREEFOUR = 1,  // From orig to 3/4.
  DOWN_ONEHALF = 2,    // From orig or 3/4 to 1/2.
  UP_THREEFOUR = -1,   // From 1/2 to 3/4.
  UP_ORIG = -2,        // From 1/2 or 3/4 to orig.
} RESIZE_ACTION;

typedef enum { ORIG = 0, THREE_QUARTER = 1, ONE_HALF = 2 } RESIZE_STATE;

// Frame dimensions multiplier wrt the native frame size, in 1/16ths,
// specified for the scale-up case.
// e.g. 24 => 16/24 = 2/3 of native size. The restriction to 1/16th is
// intended to match the capabilities of the normative scaling filters,
// giving precedence to the up-scaling accuracy.
static const int frame_scale_factor[FRAME_SCALE_STEPS] = { 16, 24 };

// Multiplier of the target rate to be used as threshold for triggering scaling.
static const double rate_thresh_mult[FRAME_SCALE_STEPS] = { 1.0, 2.0 };

// Scale dependent Rate Correction Factor multipliers. Compensates for the
// greater number of bits per pixel generated in down-scaled frames.
static const double rcf_mult[FRAME_SCALE_STEPS] = { 1.0, 2.0 };

typedef struct {
  // Rate targeting variables
  int base_frame_target;  // A baseline frame target before adjustment
                          // for previous under or over shoot.
  int this_frame_target;  // Actual frame target after rc adjustment.
  int projected_frame_size;
  int sb64_target_rate;
  int last_q[FRAME_TYPES];  // Separate values for Intra/Inter
  int last_boosted_qindex;  // Last boosted GF/KF/ARF q
  int last_kf_qindex;       // Q index of the last key frame coded.

  int gfu_boost;
  int last_boost;
  int kf_boost;

  double rate_correction_factors[RATE_FACTOR_LEVELS];

  int frames_since_golden;
  int frames_till_gf_update_due;
  int min_gf_interval;
  int max_gf_interval;
  int static_scene_max_gf_interval;
  int baseline_gf_interval;
  int constrained_gf_group;
  int frames_to_key;
  int frames_since_key;
  int this_key_frame_forced;
  int next_key_frame_forced;
  int source_alt_ref_pending;
  int source_alt_ref_active;
  int is_src_frame_alt_ref;

  int avg_frame_bandwidth;  // Average frame size target for clip
  int min_frame_bandwidth;  // Minimum allocation used for any frame
  int max_frame_bandwidth;  // Maximum burst rate allowed for a frame.

  int ni_av_qi;
  int ni_tot_qi;
  int ni_frames;
  int avg_frame_qindex[FRAME_TYPES];
  double tot_q;
  double avg_q;

  int64_t buffer_level;
  int64_t bits_off_target;
  int64_t vbr_bits_off_target;
  int64_t vbr_bits_off_target_fast;

  int decimation_factor;
  int decimation_count;

  int rolling_target_bits;
  int rolling_actual_bits;

  int long_rolling_target_bits;
  int long_rolling_actual_bits;

  int rate_error_estimate;

  int64_t total_actual_bits;
  int64_t total_target_bits;
  int64_t total_target_vs_actual;

  int worst_quality;
  int best_quality;

  int64_t starting_buffer_level;
  int64_t optimal_buffer_level;
  int64_t maximum_buffer_size;

  // rate control history for last frame(1) and the frame before(2).
  // -1: undershot
  //  1: overshoot
  //  0: not initialized.
  int rc_1_frame;
  int rc_2_frame;
  int q_1_frame;
  int q_2_frame;
  // Keep track of the last target average frame bandwidth.
  int last_avg_frame_bandwidth;

  // Auto frame-scaling variables.
  FRAME_SCALE_LEVEL frame_size_selector;
  FRAME_SCALE_LEVEL next_frame_size_selector;
  int frame_width[FRAME_SCALE_STEPS];
  int frame_height[FRAME_SCALE_STEPS];
  int rf_level_maxq[RATE_FACTOR_LEVELS];

  int fac_active_worst_inter;
  int fac_active_worst_gf;
  uint64_t avg_source_sad[MAX_LAG_BUFFERS];
  uint64_t prev_avg_source_sad_lag;
  int high_source_sad_lagindex;
  int high_num_blocks_with_motion;
  int alt_ref_gf_group;
  int last_frame_is_src_altref;
  int high_source_sad;
  int count_last_scene_change;
  int hybrid_intra_scene_change;
  int re_encode_maxq_scene_change;
  int avg_frame_low_motion;
  int af_ratio_onepass_vbr;
  int force_qpmin;
  int reset_high_source_sad;
  double perc_arf_usage;
  int force_max_q;
  // Last frame was dropped post encode on scene change.
  int last_post_encode_dropped_scene_change;
  // Enable post encode frame dropping for screen content. Only enabled when
  // ext_use_post_encode_drop is enabled by user.
  int use_post_encode_drop;
  // External flag to enable post encode frame dropping, controlled by user.
  int ext_use_post_encode_drop;
  // Flag to disable CBR feature to increase Q on overshoot detection.
  int disable_overshoot_maxq_cbr;
  int damped_adjustment[RATE_FACTOR_LEVELS];
  double arf_active_best_quality_adjustment_factor;
  int arf_increase_active_best_quality;

  int preserve_arf_as_gld;
  int preserve_next_arf_as_gld;
  int show_arf_as_gld;

  // Flag to constrain golden frame interval on key frame frequency for 1 pass
  // VBR.
  int constrain_gf_key_freq_onepass_vbr;

  // The index of the current GOP. Start from zero.
  // When a key frame is inserted, it resets to zero.
  int gop_global_index;
} RATE_CONTROL;

struct VP9_COMP;
struct VP9EncoderConfig;

void vp9_rc_init(const struct VP9EncoderConfig *oxcf, int pass,
                 RATE_CONTROL *rc);

int vp9_estimate_bits_at_q(FRAME_TYPE frame_type, int q, int mbs,
                           double correction_factor, vpx_bit_depth_t bit_depth);

double vp9_convert_qindex_to_q(int qindex, vpx_bit_depth_t bit_depth);

int vp9_convert_q_to_qindex(double q_val, vpx_bit_depth_t bit_depth);

void vp9_rc_init_minq_luts(void);

int vp9_rc_get_default_min_gf_interval(int width, int height, double framerate);
// Note vp9_rc_get_default_max_gf_interval() requires the min_gf_interval to
// be passed in to ensure that the max_gf_interval returned is at least as big
// as that.
int vp9_rc_get_default_max_gf_interval(double framerate, int min_gf_interval);

// Generally at the high level, the following flow is expected
// to be enforced for rate control:
// First call per frame, one of:
//   vp9_rc_get_one_pass_vbr_params()
//   vp9_rc_get_one_pass_cbr_params()
//   vp9_rc_get_svc_params()
//   vp9_rc_get_first_pass_params()
//   vp9_rc_get_second_pass_params()
// depending on the usage to set the rate control encode parameters desired.
//
// Then, call encode_frame_to_data_rate() to perform the
// actual encode. This function will in turn call encode_frame()
// one or more times, followed by one of:
//   vp9_rc_postencode_update()
//   vp9_rc_postencode_update_drop_frame()
//
// The majority of rate control parameters are only expected
// to be set in the vp9_rc_get_..._params() functions and
// updated during the vp9_rc_postencode_update...() functions.
// The only exceptions are vp9_rc_drop_frame() and
// vp9_rc_update_rate_correction_factors() functions.

// Functions to set parameters for encoding before the actual
// encode_frame_to_data_rate() function.
void vp9_rc_get_one_pass_vbr_params(struct VP9_COMP *cpi);
void vp9_rc_get_one_pass_cbr_params(struct VP9_COMP *cpi);
int vp9_calc_pframe_target_size_one_pass_cbr(const struct VP9_COMP *cpi);
int vp9_calc_iframe_target_size_one_pass_cbr(const struct VP9_COMP *cpi);
int vp9_calc_pframe_target_size_one_pass_vbr(const struct VP9_COMP *cpi);
int vp9_calc_iframe_target_size_one_pass_vbr(const struct VP9_COMP *cpi);
void vp9_set_gf_update_one_pass_vbr(struct VP9_COMP *const cpi);
void vp9_update_buffer_level_preencode(struct VP9_COMP *cpi);
void vp9_rc_get_svc_params(struct VP9_COMP *cpi);

// Post encode update of the rate control parameters based
// on bytes used
void vp9_rc_postencode_update(struct VP9_COMP *cpi, uint64_t bytes_used);
// Post encode update of the rate control parameters for dropped frames
void vp9_rc_postencode_update_drop_frame(struct VP9_COMP *cpi);

// Updates rate correction factors
// Changes only the rate correction factors in the rate control structure.
void vp9_rc_update_rate_correction_factors(struct VP9_COMP *cpi);

// Post encode drop for CBR mode.
int post_encode_drop_cbr(struct VP9_COMP *cpi, size_t *size);

int vp9_test_drop(struct VP9_COMP *cpi);

// Decide if we should drop this frame: For 1-pass CBR.
// Changes only the decimation count in the rate control structure
int vp9_rc_drop_frame(struct VP9_COMP *cpi);

// Computes frame size bounds.
void vp9_rc_compute_frame_size_bounds(const struct VP9_COMP *cpi,
                                      int frame_target,
                                      int *frame_under_shoot_limit,
                                      int *frame_over_shoot_limit);

// Picks q and q bounds given the target for bits
int vp9_rc_pick_q_and_bounds(const struct VP9_COMP *cpi, int *bottom_index,
                             int *top_index);

// Estimates q to achieve a target bits per frame
int vp9_rc_regulate_q(const struct VP9_COMP *cpi, int target_bits_per_frame,
                      int active_best_quality, int active_worst_quality);

// Estimates bits per mb for a given qindex and correction factor.
int vp9_rc_bits_per_mb(FRAME_TYPE frame_type, int qindex,
                       double correction_factor, vpx_bit_depth_t bit_depth);

// Clamping utilities for bitrate targets for iframes and pframes.
int vp9_rc_clamp_iframe_target_size(const struct VP9_COMP *const cpi,
                                    int target);
int vp9_rc_clamp_pframe_target_size(const struct VP9_COMP *const cpi,
                                    int target);
// Utility to set frame_target into the RATE_CONTROL structure
// This function is called only from the vp9_rc_get_..._params() functions.
void vp9_rc_set_frame_target(struct VP9_COMP *cpi, int target);

// Computes a q delta (in "q index" terms) to get from a starting q value
// to a target q value
int vp9_compute_qdelta(const RATE_CONTROL *rc, double qstart, double qtarget,
                       vpx_bit_depth_t bit_depth);

// Computes a q delta (in "q index" terms) to get from a starting q value
// to a value that should equate to the given rate ratio.
int vp9_compute_qdelta_by_rate(const RATE_CONTROL *rc, FRAME_TYPE frame_type,
                               int qindex, double rate_target_ratio,
                               vpx_bit_depth_t bit_depth);

int vp9_frame_type_qdelta(const struct VP9_COMP *cpi, int rf_level, int q);

void vp9_rc_update_framerate(struct VP9_COMP *cpi);

void vp9_rc_set_gf_interval_range(const struct VP9_COMP *const cpi,
                                  RATE_CONTROL *const rc);

void vp9_set_target_rate(struct VP9_COMP *cpi);

int vp9_resize_one_pass_cbr(struct VP9_COMP *cpi);

void vp9_scene_detection_onepass(struct VP9_COMP *cpi);

int vp9_encodedframe_overshoot(struct VP9_COMP *cpi, int frame_size, int *q);

void vp9_configure_buffer_updates(struct VP9_COMP *cpi, int gf_group_index);

void vp9_compute_frame_low_motion(struct VP9_COMP *const cpi);

void vp9_update_buffer_level_svc_preencode(struct VP9_COMP *cpi);

int vp9_rc_pick_q_and_bounds_two_pass(const struct VP9_COMP *cpi,
                                      int *bottom_index, int *top_index,
                                      int gf_group_index);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_ENCODER_VP9_RATECTRL_H_
