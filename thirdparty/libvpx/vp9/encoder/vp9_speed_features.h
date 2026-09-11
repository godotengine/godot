/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP9_ENCODER_VP9_SPEED_FEATURES_H_
#define VPX_VP9_ENCODER_VP9_SPEED_FEATURES_H_

#include "vp9/common/vp9_enums.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  INTRA_ALL = (1 << DC_PRED) | (1 << V_PRED) | (1 << H_PRED) | (1 << D45_PRED) |
              (1 << D135_PRED) | (1 << D117_PRED) | (1 << D153_PRED) |
              (1 << D207_PRED) | (1 << D63_PRED) | (1 << TM_PRED),
  INTRA_DC = (1 << DC_PRED),
  INTRA_DC_TM = (1 << DC_PRED) | (1 << TM_PRED),
  INTRA_DC_H_V = (1 << DC_PRED) | (1 << V_PRED) | (1 << H_PRED),
  INTRA_DC_TM_H_V =
      (1 << DC_PRED) | (1 << TM_PRED) | (1 << V_PRED) | (1 << H_PRED)
};

enum {
  INTER_ALL = (1 << NEARESTMV) | (1 << NEARMV) | (1 << ZEROMV) | (1 << NEWMV),
  INTER_NEAREST = (1 << NEARESTMV),
  INTER_NEAREST_NEW = (1 << NEARESTMV) | (1 << NEWMV),
  INTER_NEAREST_ZERO = (1 << NEARESTMV) | (1 << ZEROMV),
  INTER_NEAREST_NEW_ZERO = (1 << NEARESTMV) | (1 << ZEROMV) | (1 << NEWMV),
  INTER_NEAREST_NEAR_NEW = (1 << NEARESTMV) | (1 << NEARMV) | (1 << NEWMV),
  INTER_NEAREST_NEAR_ZERO = (1 << NEARESTMV) | (1 << NEARMV) | (1 << ZEROMV),
};

enum {
  DISABLE_ALL_INTER_SPLIT = (1 << THR_COMP_GA) | (1 << THR_COMP_LA) |
                            (1 << THR_ALTR) | (1 << THR_GOLD) | (1 << THR_LAST),

  DISABLE_ALL_SPLIT = (1 << THR_INTRA) | DISABLE_ALL_INTER_SPLIT,

  DISABLE_COMPOUND_SPLIT = (1 << THR_COMP_GA) | (1 << THR_COMP_LA),

  LAST_AND_INTRA_SPLIT_ONLY = (1 << THR_COMP_GA) | (1 << THR_COMP_LA) |
                              (1 << THR_ALTR) | (1 << THR_GOLD)
};

typedef enum {
  DIAMOND = 0,
  NSTEP = 1,
  HEX = 2,
  BIGDIA = 3,
  SQUARE = 4,
  FAST_HEX = 5,
  FAST_DIAMOND = 6,
  MESH = 7
} SEARCH_METHODS;

typedef enum {
  // No recode.
  DISALLOW_RECODE = 0,
  // Allow recode for KF and exceeding maximum frame bandwidth.
  ALLOW_RECODE_KFMAXBW = 1,
  // Allow recode only for KF/ARF/GF frames.
  ALLOW_RECODE_KFARFGF = 2,
  // Allow recode for ARF/GF/KF and first normal frame in each group.
  ALLOW_RECODE_FIRST = 3,
  // Allow recode for all frames based on bitrate constraints.
  ALLOW_RECODE = 4,
} RECODE_LOOP_TYPE;

typedef enum {
  SUBPEL_TREE = 0,
  SUBPEL_TREE_PRUNED = 1,           // Prunes 1/2-pel searches
  SUBPEL_TREE_PRUNED_MORE = 2,      // Prunes 1/2-pel searches more aggressively
  SUBPEL_TREE_PRUNED_EVENMORE = 3,  // Prunes 1/2- and 1/4-pel searches
  // Other methods to come
} SUBPEL_SEARCH_METHODS;

typedef enum {
  NO_MOTION_THRESHOLD = 0,
  LOW_MOTION_THRESHOLD = 7
} MOTION_THRESHOLD;

typedef enum {
  USE_FULL_RD = 0,
  USE_LARGESTALL,
  USE_TX_8X8
} TX_SIZE_SEARCH_METHOD;

typedef enum {
  NOT_IN_USE = 0,
  RELAXED_NEIGHBORING_MIN_MAX = 1,
  STRICT_NEIGHBORING_MIN_MAX = 2
} AUTO_MIN_MAX_MODE;

typedef enum {
  // Try the full image with different values.
  LPF_PICK_FROM_FULL_IMAGE,
  // Try a small portion of the image with different values.
  LPF_PICK_FROM_SUBIMAGE,
  // Estimate the level based on quantizer and frame type
  LPF_PICK_FROM_Q,
  // Pick 0 to disable LPF if LPF was enabled last frame
  LPF_PICK_MINIMAL_LPF
} LPF_PICK_METHOD;

typedef enum {
  // Terminate search early based on distortion so far compared to
  // qp step, distortion in the neighborhood of the frame, etc.
  FLAG_EARLY_TERMINATE = 1 << 0,

  // Skips comp inter modes if the best so far is an intra mode.
  FLAG_SKIP_COMP_BESTINTRA = 1 << 1,

  // Skips oblique intra modes if the best so far is an inter mode.
  FLAG_SKIP_INTRA_BESTINTER = 1 << 3,

  // Skips oblique intra modes  at angles 27, 63, 117, 153 if the best
  // intra so far is not one of the neighboring directions.
  FLAG_SKIP_INTRA_DIRMISMATCH = 1 << 4,

  // Skips intra modes other than DC_PRED if the source variance is small
  FLAG_SKIP_INTRA_LOWVAR = 1 << 5,
} MODE_SEARCH_SKIP_LOGIC;

typedef enum {
  FLAG_SKIP_EIGHTTAP = 1 << EIGHTTAP,
  FLAG_SKIP_EIGHTTAP_SMOOTH = 1 << EIGHTTAP_SMOOTH,
  FLAG_SKIP_EIGHTTAP_SHARP = 1 << EIGHTTAP_SHARP,
} INTERP_FILTER_MASK;

typedef enum {
  // Search partitions using RD/NONRD criterion.
  SEARCH_PARTITION,

  // Always use a fixed size partition.
  FIXED_PARTITION,

  REFERENCE_PARTITION,

  // Use an arbitrary partitioning scheme based on source variance within
  // a 64X64 SB.
  VAR_BASED_PARTITION,

  // Make partition decisions with machine learning models.
  ML_BASED_PARTITION
} PARTITION_SEARCH_TYPE;

typedef enum {
  // Does a dry run to see if any of the contexts need to be updated or not,
  // before the final run.
  TWO_LOOP = 0,

  // No dry run, also only half the coef contexts and bands are updated.
  // The rest are not updated at all.
  ONE_LOOP_REDUCED = 1
} FAST_COEFF_UPDATE;

typedef enum { EIGHTH_PEL, QUARTER_PEL, HALF_PEL, FULL_PEL } SUBPEL_FORCE_STOP;

typedef struct ADAPT_SUBPEL_FORCE_STOP {
  // Threshold for full pixel motion vector;
  int mv_thresh;

  // subpel_force_stop if full pixel MV is below the threshold.
  SUBPEL_FORCE_STOP force_stop_below;

  // subpel_force_stop if full pixel MV is equal to or above the threshold.
  SUBPEL_FORCE_STOP force_stop_above;
} ADAPT_SUBPEL_FORCE_STOP;

typedef struct MV_SPEED_FEATURES {
  // Motion search method (Diamond, NSTEP, Hex, Big Diamond, Square, etc).
  SEARCH_METHODS search_method;

  // This parameter controls which step in the n-step process we start at.
  // It's changed adaptively based on circumstances.
  int reduce_first_step_size;

  // If this is set to 1, we limit the motion search range to 2 times the
  // largest motion vector found in the last frame.
  int auto_mv_step_size;

  // Subpel_search_method can only be subpel_tree which does a subpixel
  // logarithmic search that keeps stepping at 1/2 pixel units until
  // you stop getting a gain, and then goes on to 1/4 and repeats
  // the same process. Along the way it skips many diagonals.
  SUBPEL_SEARCH_METHODS subpel_search_method;

  // Subpel MV search level. Can take values 0 - 2. Higher values mean more
  // extensive subpel search.
  int subpel_search_level;

  // When to stop subpel motion search.
  SUBPEL_FORCE_STOP subpel_force_stop;

  // If it's enabled, different subpel_force_stop will be used for different MV.
  int enable_adaptive_subpel_force_stop;

  ADAPT_SUBPEL_FORCE_STOP adapt_subpel_force_stop;

  // This variable sets the step_param used in full pel motion search.
  int fullpel_search_step_param;

  // Whether to downsample the rows in sad calculation during motion search.
  // This is only active when there are at least 8 rows.
  int use_downsampled_sad;
} MV_SPEED_FEATURES;

typedef struct PARTITION_SEARCH_BREAKOUT_THR {
  int64_t dist;
  int rate;
} PARTITION_SEARCH_BREAKOUT_THR;

#define MAX_MESH_STEP 4

typedef struct MESH_PATTERN {
  int range;
  int interval;
} MESH_PATTERN;

typedef enum {
  // No reaction to rate control on a detected slide/scene change.
  NO_DETECTION = 0,

  // Set to larger Q (max_q set by user) based only on the
  // detected slide/scene change and current/past Q.
  FAST_DETECTION_MAXQ = 1,

  // Based on (first pass) encoded frame, if large frame size is detected
  // then set to higher Q for the second re-encode. This involves 2 pass
  // encoding on slide change, so slower than 1, but more accurate for
  // detecting overshoot.
  RE_ENCODE_MAXQ = 2
} OVERSHOOT_DETECTION_CBR_RT;

typedef enum {
  USE_2_TAPS = 0,
  USE_4_TAPS,
  USE_8_TAPS,
  USE_8_TAPS_SHARP,
} SUBPEL_SEARCH_TYPE;

typedef enum {
  // Disable trellis coefficient optimization
  DISABLE_TRELLIS_OPT,
  // Enable trellis coefficient optimization
  ENABLE_TRELLIS_OPT,
  // Enable trellis coefficient optimization based on source variance of the
  // prediction block during transform RD
  ENABLE_TRELLIS_OPT_TX_RD_SRC_VAR,
  // Enable trellis coefficient optimization based on residual mse of the
  // transform block during transform RD
  ENABLE_TRELLIS_OPT_TX_RD_RESIDUAL_MSE,
} ENABLE_TRELLIS_OPT_METHOD;

typedef struct TRELLIS_OPT_CONTROL {
  ENABLE_TRELLIS_OPT_METHOD method;
  double thresh;
} TRELLIS_OPT_CONTROL;

typedef struct SPEED_FEATURES {
  MV_SPEED_FEATURES mv;

  // Frame level coding parameter update
  int frame_parameter_update;

  RECODE_LOOP_TYPE recode_loop;

  // Trellis (dynamic programming) optimization of quantized values (+1, 0).
  int optimize_coefficients;

  // Always set to 0. If on it enables 0 cost background transmission
  // (except for the initial transmission of the segmentation). The feature is
  // disabled because the addition of very large block sizes make the
  // backgrounds very to cheap to encode, and the segmentation we have
  // adds overhead.
  int static_segmentation;

  // The best compound predictor is found using an iterative log search process
  // that searches for best ref0 mv using error of combined predictor and then
  // searches for best ref1 mv. This sf determines the number of iterations of
  // this process based on block size. The sf becomes more aggressive from level
  // 0 to 2. The following table indicates the number of iterations w.r.t bsize:
  //  -----------------------------------------------
  // |sf (level)|bsize < 8X8| [8X8, 16X16] | > 16X16 |
  // |    0     |     4     |      4       |    4    |
  // |    1     |     0     |      2       |    4    |
  // |    2     |     0     |      0       |    0    |
  //  -----------------------------------------------
  // Here, 0 iterations indicate using the best single motion vector selected
  // for each ref frame without any iterative refinement.
  int comp_inter_joint_search_iter_level;

  // This variable is used to cap the maximum number of times we skip testing a
  // mode to be evaluated. A high value means we will be faster.
  // Turned off when (row_mt_bit_exact == 1 && adaptive_rd_thresh_row_mt == 0).
  int adaptive_rd_thresh;

  // Flag to use adaptive_rd_thresh when row-mt is enabled, only for non-rd
  // pickmode.
  int adaptive_rd_thresh_row_mt;

  // Enables skipping the reconstruction step (idct, recon) in the
  // intermediate steps assuming the last frame didn't have too many intra
  // blocks and the q is less than a threshold.
  int skip_encode_sb;
  int skip_encode_frame;
  // Speed feature to allow or disallow skipping of recode at block
  // level within a frame.
  int allow_skip_recode;

  // Coefficient probability model approximation step size
  int coeff_prob_appx_step;

  // Enable uniform quantizer followed by trellis coefficient optimization
  // during transform RD
  TRELLIS_OPT_CONTROL trellis_opt_tx_rd;

  // Enable asymptotic closed-loop encoding decision for key frame and
  // alternate reference frames.
  int allow_acl;

  // Temporal dependency model based encoding mode optimization
  int enable_tpl_model;

  // Use transform domain distortion. Use pixel domain distortion in speed 0
  // and certain situations in higher speed to improve the RD model precision.
  int allow_txfm_domain_distortion;
  double tx_domain_thresh;

  // The threshold is to determine how slow the motino is, it is used when
  // use_lastframe_partitioning is set to LAST_FRAME_PARTITION_LOW_MOTION
  MOTION_THRESHOLD lf_motion_threshold;

  // Determine which method we use to determine transform size. We can choose
  // between options like full rd, largest for prediction size, largest
  // for intra and model coefs for the rest.
  TX_SIZE_SEARCH_METHOD tx_size_search_method;

  // How many levels of tx size to search, starting from the largest.
  int tx_size_search_depth;

  // Low precision 32x32 fdct keeps everything in 16 bits and thus is less
  // precise but significantly faster than the non lp version.
  int use_lp32x32fdct;

  // After looking at the first set of modes (set by index here), skip
  // checking modes for reference frames that don't match the reference frame
  // of the best so far.
  int mode_skip_start;

  // TODO(JBB): Remove this.
  int reference_masking;

  PARTITION_SEARCH_TYPE partition_search_type;

  // Used if partition_search_type = FIXED_PARTITION
  BLOCK_SIZE always_this_block_size;

  // Skip rectangular partition test when partition type none gives better
  // rd than partition type split.
  int less_rectangular_check;

  // Disable testing non square partitions(eg 16x32) for block sizes larger than
  // use_square_only_thresh_high or smaller than use_square_only_thresh_low.
  int use_square_partition_only;
  BLOCK_SIZE use_square_only_thresh_high;
  BLOCK_SIZE use_square_only_thresh_low;

  // Prune reference frames for rectangular partitions.
  int prune_ref_frame_for_rect_partitions;

  // Sets min and max partition sizes for this 64x64 region based on the
  // same 64x64 in last encoded frame, and the left and above neighbor.
  AUTO_MIN_MAX_MODE auto_min_max_partition_size;
  // Ensures the rd based auto partition search will always
  // go down at least to the specified level.
  BLOCK_SIZE rd_auto_partition_min_limit;

  // Min and max partition size we enable (block_size) as per auto
  // min max, but also used by adjust partitioning, and pick_partitioning.
  BLOCK_SIZE default_min_partition_size;
  BLOCK_SIZE default_max_partition_size;

  // Whether or not we allow partitions one smaller or one greater than the last
  // frame's partitioning. Only used if use_lastframe_partitioning is set.
  int adjust_partitioning_from_last_frame;

  // How frequently we re do the partitioning from scratch. Only used if
  // use_lastframe_partitioning is set.
  int last_partitioning_redo_frequency;

  // Disables sub 8x8 blocksizes in different scenarios: Choices are to disable
  // it always, to allow it for only Last frame and Intra, disable it for all
  // inter modes or to enable it always.
  int disable_split_mask;

  // TODO(jingning): combine the related motion search speed features
  // This allows us to use motion search at other sizes as a starting
  // point for this motion search and limits the search range around it.
  int adaptive_motion_search;

  // Do extra full pixel motion search to obtain better motion vector.
  int enhanced_full_pixel_motion_search;

  // Threshold for allowing exhaistive motion search.
  int exhaustive_searches_thresh;

  // Pattern to be used for any exhaustive mesh searches.
  MESH_PATTERN mesh_patterns[MAX_MESH_STEP];

  int schedule_mode_search;

  // Allows sub 8x8 modes to use the prediction filter that was determined
  // best for 8x8 mode. If set to 0 we always re check all the filters for
  // sizes less than 8x8, 1 means we check all filter modes if no 8x8 filter
  // was selected, and 2 means we use 8 tap if no 8x8 filter mode was selected.
  int adaptive_pred_interp_filter;

  // Adaptive prediction mode search
  int adaptive_mode_search;

  // Prune NEAREST and ZEROMV single reference modes based on motion vector
  // difference and mode rate
  int prune_single_mode_based_on_mv_diff_mode_rate;

  // Chessboard pattern prediction for interp filter. Aggressiveness increases
  // with levels.
  // 0: disable
  // 1: cb pattern in eval when filter is not switchable
  // 2: cb pattern prediction for filter search
  int cb_pred_filter_search;

  // This variable enables an early termination of interpolation filter eval
  // based on the current rd cost after processing each plane
  int early_term_interp_search_plane_rd;

  int cb_partition_search;

  int motion_field_mode_search;

  int alt_ref_search_fp;

  // Fast quantization process path
  int use_quant_fp;

  // Use finer quantizer in every other few frames that run variable block
  // partition type search.
  int force_frame_boost;

  // Maximally allowed base quantization index fluctuation.
  int max_delta_qindex;

  // Implements various heuristics to skip searching modes
  // The heuristics selected are based on  flags
  // defined in the MODE_SEARCH_SKIP_HEURISTICS enum
  unsigned int mode_search_skip_flags;

  // A source variance threshold below which filter search is disabled
  // Choose a very large value (UINT_MAX) to use 8-tap always
  unsigned int disable_filter_search_var_thresh;

  // These bit masks allow you to enable or disable intra modes for each
  // transform size separately.
  int intra_y_mode_mask[TX_SIZES];
  int intra_uv_mode_mask[TX_SIZES];

  // These bit masks allow you to enable or disable intra modes for each
  // prediction block size separately.
  int intra_y_mode_bsize_mask[BLOCK_SIZES];

  // This variable enables an early break out of mode testing if the model for
  // rd built from the prediction signal indicates a value that's much
  // higher than the best rd we've seen so far.
  int use_rd_breakout;

  // This enables us to use an estimate for intra rd based on dc mode rather
  // than choosing an actual uv mode in the stage of encoding before the actual
  // final encode.
  int use_uv_intra_rd_estimate;

  // This feature controls how the loop filter level is determined.
  LPF_PICK_METHOD lpf_pick;

  // This feature limits the number of coefficients updates we actually do
  // by only looking at counts from 1/2 the bands.
  FAST_COEFF_UPDATE use_fast_coef_updates;

  // This flag controls the use of non-RD mode decision.
  int use_nonrd_pick_mode;

  // A binary mask indicating if NEARESTMV, NEARMV, ZEROMV, NEWMV
  // modes are used in order from LSB to MSB for each BLOCK_SIZE.
  int inter_mode_mask[BLOCK_SIZES];

  // This feature controls whether we do the expensive context update and
  // calculation in the rd coefficient costing loop.
  int use_fast_coef_costing;

  // This feature controls the tolerence vs target used in deciding whether to
  // recode a frame. It has no meaning if recode is disabled.
  int recode_tolerance_low;
  int recode_tolerance_high;

  // This variable controls the maximum block size where intra blocks can be
  // used in inter frames.
  // TODO(aconverse): Fold this into one of the other many mode skips
  BLOCK_SIZE max_intra_bsize;

  // When partition is pre-set, the inter prediction result from pick_inter_mode
  // can be reused in final block encoding process. It is enabled only for real-
  // time mode speed 6.
  int reuse_inter_pred_sby;

  // This variable sets the encode_breakout threshold. Currently, it is only
  // enabled in real time mode.
  int encode_breakout_thresh;

  // default interp filter choice
  INTERP_FILTER default_interp_filter;

  // Early termination in transform size search, which only applies while
  // tx_size_search_method is USE_FULL_RD.
  int tx_size_search_breakout;

  // adaptive interp_filter search to allow skip of certain filter types.
  int adaptive_interp_filter_search;

  // mask for skip evaluation of certain interp_filter type.
  INTERP_FILTER_MASK interp_filter_search_mask;

  // Partition search early breakout thresholds.
  PARTITION_SEARCH_BREAKOUT_THR partition_search_breakout_thr;

  struct {
    // Use ML-based partition search early breakout.
    int search_breakout;
    // Higher values mean more aggressiveness for partition search breakout that
    // results in better encoding  speed but worse compression performance.
    float search_breakout_thresh[3];

    // Machine-learning based partition search early termination
    int search_early_termination;

    // Machine-learning based partition search pruning using prediction residue
    // variance.
    int var_pruning;

    // Threshold values used for ML based rectangular partition search pruning.
    // If < 0, the feature is turned off.
    // Higher values mean more aggressiveness to skip rectangular partition
    // search that results in better encoding speed but worse coding
    // performance.
    int prune_rect_thresh[4];
  } rd_ml_partition;

  // Fast approximation of vp9_model_rd_from_var_lapndz
  int simple_model_rd_from_var;

  // Skip a number of expensive mode evaluations for blocks with zero source
  // variance.
  int short_circuit_flat_blocks;

  // Skip a number of expensive mode evaluations for blocks with very low
  // temporal variance. If the low temporal variance flag is set for a block,
  // do the following:
  // 1: Skip all golden modes and ALL INTRA for bsize >= 32x32.
  // 2: Skip golden non-zeromv and newmv-last for bsize >= 16x16, skip ALL
  // INTRA for bsize >= 32x32 and vert/horz INTRA for bsize 16x16, 16x32 and
  // 32x16.
  // 3: Same as (2), but also skip golden zeromv.
  int short_circuit_low_temp_var;

  // Limits the rd-threshold update for early exit for the newmv-last mode,
  // for non-rd mode.
  int limit_newmv_early_exit;

  // Adds a bias against golden reference, for non-rd mode.
  int bias_golden;

  // Bias to use base mv and skip 1/4 subpel search when use base mv in
  // enhancement layer.
  int base_mv_aggressive;

  // Global flag to enable partition copy from the previous frame.
  int copy_partition_flag;

  // Compute the source sad for every superblock of the frame,
  // prior to encoding the frame, to be used to bypass some encoder decisions.
  int use_source_sad;

  int use_simple_block_yrd;

  // If source sad of superblock is high (> adapt_partition_thresh), will switch
  // from VARIANCE_PARTITION to REFERENCE_PARTITION (which selects partition
  // based on the nonrd-pickmode).
  int adapt_partition_source_sad;
  int adapt_partition_thresh;

  // Enable use of alt-refs in 1 pass VBR.
  int use_altref_onepass;

  // Enable use of compound prediction, for nonrd_pickmode with nonzero lag.
  int use_compound_nonrd_pickmode;

  // Always use nonrd_pick_intra for all block sizes on keyframes.
  int nonrd_keyframe;

  // For SVC: enables use of partition from lower spatial resolution.
  int svc_use_lowres_part;

  // Flag to indicate process for handling overshoot on slide/scene change,
  // for real-time CBR mode.
  OVERSHOOT_DETECTION_CBR_RT overshoot_detection_cbr_rt;

  // Disable partitioning of 16x16 blocks.
  int disable_16x16part_nonkey;

  // Allow for disabling golden reference.
  int disable_golden_ref;

  // Allow sub-pixel search to use interpolation filters with different taps in
  // order to achieve accurate motion search result.
  SUBPEL_SEARCH_TYPE use_accurate_subpel_search;

  // Search method used by temporal filtering in full_pixel_motion_search.
  SEARCH_METHODS temporal_filter_search_method;

  // Use machine learning based partition search.
  int nonrd_use_ml_partition;

  // Multiplier for base threshold for variance partitioning.
  int variance_part_thresh_mult;

  // Force subpel motion filter to always use SMOOTH_FILTER.
  int force_smooth_interpol;

  // For real-time mode: force DC only under intra search when content
  // does not have high souce SAD.
  int rt_intra_dc_only_low_content;

  // The encoder has a feature that skips forward transform and quantization
  // based on a model rd estimation to reduce encoding time.
  // However, this feature is dangerous since it could lead to bad perceptual
  // quality. This flag is added to guard the feature.
  int allow_skip_txfm_ac_dc;
} SPEED_FEATURES;

struct VP9_COMP;

void vp9_set_speed_features_framesize_independent(struct VP9_COMP *cpi,
                                                  int speed);
void vp9_set_speed_features_framesize_dependent(struct VP9_COMP *cpi,
                                                int speed);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_ENCODER_VP9_SPEED_FEATURES_H_
