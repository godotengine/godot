/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP9_ENCODER_VP9_ENCODER_H_
#define VPX_VP9_ENCODER_VP9_ENCODER_H_

#include <stdio.h>

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "vpx/internal/vpx_codec_internal.h"
#include "vpx/vpx_ext_ratectrl.h"
#include "vpx/vp8cx.h"
#include "vpx/vpx_tpl.h"
#if CONFIG_INTERNAL_STATS
#include "vpx_dsp/ssim.h"
#endif
#include "vpx_dsp/variance.h"
#include "vpx_dsp/psnr.h"
#include "vpx_ports/system_state.h"
#include "vpx_util/vpx_pthread.h"
#include "vpx_util/vpx_thread.h"
#include "vpx_util/vpx_timestamp.h"

#include "vp9/common/vp9_alloccommon.h"
#include "vp9/common/vp9_ppflags.h"
#include "vp9/common/vp9_entropymode.h"
#include "vp9/common/vp9_thread_common.h"
#include "vp9/common/vp9_onyxc_int.h"

#if !CONFIG_REALTIME_ONLY
#include "vp9/encoder/vp9_alt_ref_aq.h"
#endif
#include "vp9/encoder/vp9_aq_cyclicrefresh.h"
#include "vp9/encoder/vp9_context_tree.h"
#include "vp9/encoder/vp9_encodemb.h"
#include "vp9/encoder/vp9_ethread.h"
#include "vp9/encoder/vp9_ext_ratectrl.h"
#include "vp9/encoder/vp9_firstpass.h"
#include "vp9/encoder/vp9_job_queue.h"
#include "vp9/encoder/vp9_lookahead.h"
#include "vp9/encoder/vp9_mbgraph.h"
#include "vp9/encoder/vp9_mcomp.h"
#include "vp9/encoder/vp9_noise_estimate.h"
#include "vp9/encoder/vp9_quantize.h"
#include "vp9/encoder/vp9_ratectrl.h"
#include "vp9/encoder/vp9_rd.h"
#include "vp9/encoder/vp9_speed_features.h"
#include "vp9/encoder/vp9_svc_layercontext.h"
#include "vp9/encoder/vp9_tokenize.h"

#if CONFIG_VP9_TEMPORAL_DENOISING
#include "vp9/encoder/vp9_denoiser.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

// vp9 uses 10,000,000 ticks/second as time stamp
#define TICKS_PER_SEC 10000000

typedef struct {
  int nmvjointcost[MV_JOINTS];
  int nmvcosts[2][MV_VALS];
  int nmvcosts_hp[2][MV_VALS];

  vpx_prob segment_pred_probs[PREDICTION_PROBS];

  unsigned char *last_frame_seg_map_copy;

  // 0 = Intra, Last, GF, ARF
  signed char last_ref_lf_deltas[MAX_REF_LF_DELTAS];
  // 0 = ZERO_MV, MV
  signed char last_mode_lf_deltas[MAX_MODE_LF_DELTAS];

  FRAME_CONTEXT fc;
} CODING_CONTEXT;

typedef enum {
  // encode_breakout is disabled.
  ENCODE_BREAKOUT_DISABLED = 0,
  // encode_breakout is enabled.
  ENCODE_BREAKOUT_ENABLED = 1,
  // encode_breakout is enabled with small max_thresh limit.
  ENCODE_BREAKOUT_LIMITED = 2
} ENCODE_BREAKOUT_TYPE;

typedef enum {
  // Good Quality Fast Encoding. The encoder balances quality with the amount of
  // time it takes to encode the output. Speed setting controls how fast.
  GOOD,

  // The encoder places priority on the quality of the output over encoding
  // speed. The output is compressed at the highest possible quality. This
  // option takes the longest amount of time to encode. Speed setting ignored.
  BEST,

  // Realtime/Live Encoding. This mode is optimized for realtime encoding (for
  // example, capturing a television signal or feed from a live camera). Speed
  // setting controls how fast.
  REALTIME
} MODE;

typedef enum {
  FRAMEFLAGS_KEY = 1 << 0,
  FRAMEFLAGS_GOLDEN = 1 << 1,
  FRAMEFLAGS_ALTREF = 1 << 2,
} FRAMETYPE_FLAGS;

typedef enum {
  NO_AQ = 0,
  VARIANCE_AQ = 1,
  COMPLEXITY_AQ = 2,
  CYCLIC_REFRESH_AQ = 3,
  EQUATOR360_AQ = 4,
  PERCEPTUAL_AQ = 5,
  PSNR_AQ = 6,
  // AQ based on lookahead temporal
  // variance (only valid for altref frames)
  LOOKAHEAD_AQ = 7,
  AQ_MODE_COUNT  // This should always be the last member of the enum
} AQ_MODE;

typedef enum {
  RESIZE_NONE = 0,    // No frame resizing allowed (except for SVC).
  RESIZE_FIXED = 1,   // All frames are coded at the specified dimension.
  RESIZE_DYNAMIC = 2  // Coded size of each frame is determined by the codec.
} RESIZE_TYPE;

typedef enum {
  kInvalid = 0,
  kLowSadLowSumdiff = 1,
  kLowSadHighSumdiff = 2,
  kHighSadLowSumdiff = 3,
  kHighSadHighSumdiff = 4,
  kLowVarHighSumdiff = 5,
  kVeryHighSad = 6,
} CONTENT_STATE_SB;

typedef enum {
  LOOPFILTER_ALL = 0,
  LOOPFILTER_REFERENCE = 1,  // Disable loopfilter on non reference frames.
  NO_LOOPFILTER = 2,         // Disable loopfilter on all frames.
} LOOPFILTER_CONTROL;

typedef struct VP9EncoderConfig {
  BITSTREAM_PROFILE profile;
  vpx_bit_depth_t bit_depth;     // Codec bit-depth.
  int width;                     // width of data passed to the compressor
  int height;                    // height of data passed to the compressor
  unsigned int input_bit_depth;  // Input bit depth.
  double init_framerate;         // set to passed in framerate
  vpx_rational_t g_timebase;  // equivalent to g_timebase in vpx_codec_enc_cfg_t
  vpx_rational64_t g_timebase_in_ts;  // g_timebase * TICKS_PER_SEC

  int64_t target_bandwidth;  // bandwidth to be used in bits per second

  int noise_sensitivity;  // pre processing blur: recommendation 0
  int sharpness;          // sharpening output: recommendation 0:
  int speed;
  // maximum allowed bitrate for any intra frame in % of bitrate target.
  unsigned int rc_max_intra_bitrate_pct;
  // maximum allowed bitrate for any inter frame in % of bitrate target.
  unsigned int rc_max_inter_bitrate_pct;
  // percent of rate boost for golden frame in CBR mode.
  unsigned int gf_cbr_boost_pct;

  MODE mode;
  int pass;

  // Key Framing Operations
  int auto_key;  // autodetect cut scenes and set the keyframes
  int key_freq;  // maximum distance to key frame.

  int lag_in_frames;  // how many frames lag before we start encoding

  // ----------------------------------------------------------------
  // DATARATE CONTROL OPTIONS

  // vbr, cbr, constrained quality or constant quality
  enum vpx_rc_mode rc_mode;

  // buffer targeting aggressiveness
  int under_shoot_pct;
  int over_shoot_pct;

  // buffering parameters
  int64_t starting_buffer_level_ms;
  int64_t optimal_buffer_level_ms;
  int64_t maximum_buffer_size_ms;

  // Frame drop threshold.
  int drop_frames_water_mark;

  // controlling quality
  int fixed_q;
  int worst_allowed_q;
  int best_allowed_q;
  int cq_level;
  AQ_MODE aq_mode;  // Adaptive Quantization mode

  // Special handling of Adaptive Quantization for AltRef frames
  int alt_ref_aq;

  // Internal frame size scaling.
  RESIZE_TYPE resize_mode;
  int scaled_frame_width;
  int scaled_frame_height;

  // Enable feature to reduce the frame quantization every x frames.
  int frame_periodic_boost;

  // two pass datarate control
  int two_pass_vbrbias;  // two pass datarate control tweaks
  int two_pass_vbrmin_section;
  int two_pass_vbrmax_section;
  int vbr_corpus_complexity;  // 0 indicates corpus vbr disabled
  // END DATARATE CONTROL OPTIONS
  // ----------------------------------------------------------------

  // Spatial and temporal scalability.
  int ss_number_layers;  // Number of spatial layers.
  int ts_number_layers;  // Number of temporal layers.
  // Bitrate allocation for spatial layers.
  int layer_target_bitrate[VPX_MAX_LAYERS];
  int ss_target_bitrate[VPX_SS_MAX_LAYERS];
  int ss_enable_auto_arf[VPX_SS_MAX_LAYERS];
  // Bitrate allocation (CBR mode) and framerate factor, for temporal layers.
  int ts_rate_decimator[VPX_TS_MAX_LAYERS];

  int enable_auto_arf;

  int encode_breakout;  // early breakout : for video conf recommend 800

  /* Bitfield defining the error resiliency features to enable.
   * Can provide decodable frames after losses in previous
   * frames and decodable partitions after losses in the same frame.
   */
  unsigned int error_resilient_mode;

  /* Bitfield defining the parallel decoding mode where the
   * decoding in successive frames may be conducted in parallel
   * just by decoding the frame headers.
   */
  unsigned int frame_parallel_decoding_mode;

  int arnr_max_frames;
  int arnr_strength;

  int min_gf_interval;
  int max_gf_interval;

  int tile_columns;
  int tile_rows;

  int enable_tpl_model;

  int enable_keyframe_filtering;

  int max_threads;

  unsigned int target_level;

  vpx_fixed_buf_t two_pass_stats_in;

  vp8e_tuning tuning;
  vp9e_tune_content content;
#if CONFIG_VP9_HIGHBITDEPTH
  int use_highbitdepth;
#endif
  vpx_color_space_t color_space;
  vpx_color_range_t color_range;
  int render_width;
  int render_height;
  VP9E_TEMPORAL_LAYERING_MODE temporal_layering_mode;

  int row_mt;
  unsigned int motion_vector_unit_test;
  int delta_q_uv;
} VP9EncoderConfig;

static INLINE int is_lossless_requested(const VP9EncoderConfig *cfg) {
  return cfg->best_allowed_q == 0 && cfg->worst_allowed_q == 0;
}

typedef struct TplDepStats {
  int64_t intra_cost;
  int64_t inter_cost;
  int64_t mc_flow;
  int64_t mc_dep_cost;
  int64_t mc_ref_cost;

  int ref_frame_index;
  int_mv mv;
} TplDepStats;

#if CONFIG_NON_GREEDY_MV

#define ZERO_MV_MODE 0
#define NEW_MV_MODE 1
#define NEAREST_MV_MODE 2
#define NEAR_MV_MODE 3
#define MAX_MV_MODE 4
#endif

typedef struct TplDepFrame {
  uint8_t is_valid;
  TplDepStats *tpl_stats_ptr;
  int stride;
  int width;
  int height;
  int mi_rows;
  int mi_cols;
  int base_qindex;
#if CONFIG_NON_GREEDY_MV
  int lambda;
  int *mv_mode_arr[3];
  double *rd_diff_arr[3];
#endif
} TplDepFrame;

#define TPL_DEP_COST_SCALE_LOG2 4

// TODO(jingning) All spatially adaptive variables should go to TileDataEnc.
typedef struct TileDataEnc {
  TileInfo tile_info;
  int thresh_freq_fact[BLOCK_SIZES][MAX_MODES];
  int thresh_freq_fact_prev[BLOCK_SIZES][MAX_MODES];
  int8_t mode_map[BLOCK_SIZES][MAX_MODES];
  FIRSTPASS_DATA fp_data;
  VP9RowMTSync row_mt_sync;

  // Used for adaptive_rd_thresh with row multithreading
  int *row_base_thresh_freq_fact;
  // The value of sb_rows when row_base_thresh_freq_fact is allocated.
  // The row_base_thresh_freq_fact array has sb_rows * BLOCK_SIZES * MAX_MODES
  // elements.
  int sb_rows;
  MV firstpass_top_mv;
} TileDataEnc;

typedef struct RowMTInfo {
  JobQueueHandle job_queue_hdl;
#if CONFIG_MULTITHREAD
  pthread_mutex_t job_mutex;
#endif
} RowMTInfo;

typedef struct {
  TOKENEXTRA *start;
  TOKENEXTRA *stop;
  unsigned int count;
} TOKENLIST;

typedef struct MultiThreadHandle {
  int allocated_tile_rows;
  int allocated_tile_cols;
  int allocated_vert_unit_rows;

  // Frame level params
  int num_tile_vert_sbs[MAX_NUM_TILE_ROWS];

  // Job Queue structure and handles
  JobQueue *job_queue;

  int jobs_per_tile_col;

  RowMTInfo row_mt_info[MAX_NUM_TILE_COLS];
  int thread_id_to_tile_id[MAX_NUM_THREADS];  // Mapping of threads to tiles
} MultiThreadHandle;

typedef struct RD_COUNTS {
  vp9_coeff_count coef_counts[TX_SIZES][PLANE_TYPES];
  int64_t comp_pred_diff[REFERENCE_MODES];
  int64_t filter_diff[SWITCHABLE_FILTER_CONTEXTS];
} RD_COUNTS;

typedef struct ThreadData {
  MACROBLOCK mb;
  RD_COUNTS rd_counts;
  FRAME_COUNTS *counts;

  PICK_MODE_CONTEXT *leaf_tree;
  PC_TREE *pc_tree;
  PC_TREE *pc_root;
} ThreadData;

struct EncWorkerData;

typedef struct ActiveMap {
  int enabled;
  int update;
  unsigned char *map;
} ActiveMap;

typedef enum { Y, U, V, ALL } STAT_TYPE;

typedef struct IMAGE_STAT {
  double stat[ALL + 1];
  double worst;
} ImageStat;

// Kf noise filtering currently disabled by default in build.
// #define ENABLE_KF_DENOISE 1

#define CPB_WINDOW_SIZE 4
#define FRAME_WINDOW_SIZE 128
#define SAMPLE_RATE_GRACE_P 0.015
#define VP9_LEVELS 14

typedef enum {
  LEVEL_UNKNOWN = 0,
  LEVEL_AUTO = 1,
  LEVEL_1 = 10,
  LEVEL_1_1 = 11,
  LEVEL_2 = 20,
  LEVEL_2_1 = 21,
  LEVEL_3 = 30,
  LEVEL_3_1 = 31,
  LEVEL_4 = 40,
  LEVEL_4_1 = 41,
  LEVEL_5 = 50,
  LEVEL_5_1 = 51,
  LEVEL_5_2 = 52,
  LEVEL_6 = 60,
  LEVEL_6_1 = 61,
  LEVEL_6_2 = 62,
  LEVEL_MAX = 255
} VP9_LEVEL;

typedef struct {
  VP9_LEVEL level;
  uint64_t max_luma_sample_rate;
  uint32_t max_luma_picture_size;
  uint32_t max_luma_picture_breadth;
  double average_bitrate;  // in kilobits per second
  double max_cpb_size;     // in kilobits
  double compression_ratio;
  uint8_t max_col_tiles;
  uint32_t min_altref_distance;
  uint8_t max_ref_frame_buffers;
} Vp9LevelSpec;

extern const Vp9LevelSpec vp9_level_defs[VP9_LEVELS];

typedef struct {
  int64_t ts;  // timestamp
  uint32_t luma_samples;
  uint32_t size;  // in bytes
} FrameRecord;

typedef struct {
  FrameRecord buf[FRAME_WINDOW_SIZE];
  uint8_t start;
  uint8_t len;
} FrameWindowBuffer;

typedef struct {
  uint8_t seen_first_altref;
  uint32_t frames_since_last_altref;
  uint64_t total_compressed_size;
  uint64_t total_uncompressed_size;
  double time_encoded;  // in seconds
  FrameWindowBuffer frame_window_buffer;
  int ref_refresh_map;
} Vp9LevelStats;

typedef struct {
  Vp9LevelStats level_stats;
  Vp9LevelSpec level_spec;
} Vp9LevelInfo;

typedef enum {
  BITRATE_TOO_LARGE = 0,
  LUMA_PIC_SIZE_TOO_LARGE,
  LUMA_PIC_BREADTH_TOO_LARGE,
  LUMA_SAMPLE_RATE_TOO_LARGE,
  CPB_TOO_LARGE,
  COMPRESSION_RATIO_TOO_SMALL,
  TOO_MANY_COLUMN_TILE,
  ALTREF_DIST_TOO_SMALL,
  TOO_MANY_REF_BUFFER,
  TARGET_LEVEL_FAIL_IDS
} TARGET_LEVEL_FAIL_ID;

typedef struct {
  int8_t level_index;
  uint8_t fail_flag;
  int max_frame_size;   // in bits
  double max_cpb_size;  // in bits
} LevelConstraint;

typedef struct ARNRFilterData {
  YV12_BUFFER_CONFIG *frames[MAX_LAG_BUFFERS];
  int strength;
  int frame_count;
  int alt_ref_index;
  struct scale_factors sf;
  YV12_BUFFER_CONFIG *dst;
} ARNRFilterData;

typedef struct EncFrameBuf {
  int mem_valid;
  int released;
  YV12_BUFFER_CONFIG frame;
} EncFrameBuf;

// Maximum operating frame buffer size needed for a GOP using ARF reference.
// This is used to allocate the memory for TPL stats for a GOP.
#define MAX_ARF_GOP_SIZE (2 * MAX_LAG_BUFFERS)
#define MAX_KMEANS_GROUPS 8

typedef struct KMEANS_DATA {
  double value;
  int pos;
  int group_idx;
} KMEANS_DATA;

#if CONFIG_COLLECT_COMPONENT_TIMING
#include "vpx_ports/vpx_timer.h"
// Adjust the following to add new components.
typedef enum {
  vp9_get_compressed_data_time,
  vp9_temporal_filter_time,
  vp9_rc_get_second_pass_params_time,
  setup_tpl_stats_time,
  Pass2Encode_time,

  encode_with_recode_loop_time,
  loopfilter_frame_time,
  vp9_pack_bitstream_time,

  encode_frame_internal_time,
  rd_pick_partition_time,
  rd_pick_sb_modes_time,
  encode_sb_time,

  vp9_rd_pick_inter_mode_sb_time,
  vp9_rd_pick_inter_mode_sub8x8_time,

  intra_mode_search_time,
  handle_inter_mode_time,
  single_motion_search_time,
  joint_motion_search_time,
  interp_filter_time,

  kTimingComponents,
} TIMING_COMPONENT;

static INLINE char const *get_component_name(int index) {
  switch (index) {
    case vp9_get_compressed_data_time: return "vp9_get_compressed_data_time";
    case vp9_temporal_filter_time: return "vp9_temporal_filter_time";
    case vp9_rc_get_second_pass_params_time:
      return "vp9_rc_get_second_pass_params_time";
    case setup_tpl_stats_time: return "setup_tpl_stats_time";
    case Pass2Encode_time: return "Pass2Encode_time";

    case encode_with_recode_loop_time: return "encode_with_recode_loop_time";
    case loopfilter_frame_time: return "loopfilter_frame_time";
    case vp9_pack_bitstream_time: return "vp9_pack_bitstream_time";

    case encode_frame_internal_time: return "encode_frame_internal_time";
    case rd_pick_partition_time: return "rd_pick_partition_time";
    case rd_pick_sb_modes_time: return "rd_pick_sb_modes_time";
    case encode_sb_time: return "encode_sb_time";

    case vp9_rd_pick_inter_mode_sb_time:
      return "vp9_rd_pick_inter_mode_sb_time";
    case vp9_rd_pick_inter_mode_sub8x8_time:
      return "vp9_rd_pick_inter_mode_sub8x8_time";

    case intra_mode_search_time: return "intra_mode_search_time";
    case handle_inter_mode_time: return "handle_inter_mode_time";
    case single_motion_search_time: return "single_motion_search_time";
    case joint_motion_search_time: return "joint_motion_search_time";
    case interp_filter_time: return "interp_filter_time";

    default: assert(0);
  }
  return "error";
}
#endif

typedef struct VP9_COMP {
  FRAME_INFO frame_info;
  QUANTS quants;
  ThreadData td;
  MB_MODE_INFO_EXT *mbmi_ext_base;
  DECLARE_ALIGNED(16, int16_t, y_dequant[QINDEX_RANGE][8]);
  DECLARE_ALIGNED(16, int16_t, uv_dequant[QINDEX_RANGE][8]);
  VP9_COMMON common;
  VP9EncoderConfig oxcf;
  struct lookahead_ctx *lookahead;
  struct lookahead_entry *alt_ref_source;

  YV12_BUFFER_CONFIG *Source;
  YV12_BUFFER_CONFIG *Last_Source;  // NULL for first frame and alt_ref frames
  YV12_BUFFER_CONFIG *un_scaled_source;
  YV12_BUFFER_CONFIG scaled_source;
  YV12_BUFFER_CONFIG *unscaled_last_source;
  YV12_BUFFER_CONFIG scaled_last_source;
#ifdef ENABLE_KF_DENOISE
  YV12_BUFFER_CONFIG raw_unscaled_source;
  YV12_BUFFER_CONFIG raw_scaled_source;
#endif
  YV12_BUFFER_CONFIG *raw_source_frame;

  BLOCK_SIZE tpl_bsize;
  TplDepFrame tpl_stats[MAX_ARF_GOP_SIZE];
  // Used to store TPL stats before propagation
  VpxTplGopStats tpl_gop_stats;
  YV12_BUFFER_CONFIG *tpl_recon_frames[REF_FRAMES];
  EncFrameBuf enc_frame_buf[REF_FRAMES];
#if CONFIG_MULTITHREAD
  pthread_mutex_t kmeans_mutex;
#endif
  int kmeans_data_arr_alloc;
  KMEANS_DATA *kmeans_data_arr;
  int kmeans_data_size;
  int kmeans_data_stride;
  double kmeans_ctr_ls[MAX_KMEANS_GROUPS];
  double kmeans_boundary_ls[MAX_KMEANS_GROUPS];
  int kmeans_count_ls[MAX_KMEANS_GROUPS];
  int kmeans_ctr_num;
#if CONFIG_NON_GREEDY_MV
  MotionFieldInfo motion_field_info;
  int tpl_ready;
  int_mv *select_mv_arr;
#endif

  TileDataEnc *tile_data;
  int allocated_tiles;  // Keep track of memory allocated for tiles.

  int scaled_ref_idx[REFS_PER_FRAME];
  int lst_fb_idx;
  int gld_fb_idx;
  int alt_fb_idx;

  int ref_fb_idx[REF_FRAMES];

  int refresh_last_frame;
  int refresh_golden_frame;
  int refresh_alt_ref_frame;

  int ext_refresh_frame_flags_pending;
  int ext_refresh_last_frame;
  int ext_refresh_golden_frame;
  int ext_refresh_alt_ref_frame;

  int ext_refresh_frame_context_pending;
  int ext_refresh_frame_context;

  int64_t norm_wiener_variance;
  int64_t *mb_wiener_variance;
  int mb_wiener_var_rows;
  int mb_wiener_var_cols;
  double *mi_ssim_rdmult_scaling_factors;

  int64_t *sb_mul_scale;

  YV12_BUFFER_CONFIG last_frame_uf;

  TOKENEXTRA *tile_tok[4][1 << 6];
  TOKENLIST *tplist[4][1 << 6];

  // Ambient reconstruction err target for force key frames
  int64_t ambient_err;

  RD_CONTROL rd_ctrl;
  RD_OPT rd;

  CODING_CONTEXT coding_context;

  int *nmvcosts[2];
  int *nmvcosts_hp[2];
  int *nmvsadcosts[2];
  int *nmvsadcosts_hp[2];

  int64_t last_time_stamp_seen;
  int64_t last_end_time_stamp_seen;
  int64_t first_time_stamp_ever;

  RATE_CONTROL rc;
  double framerate;

  int interp_filter_selected[REF_FRAMES][SWITCHABLE];

  struct vpx_codec_pkt_list *output_pkt_list;

  MBGRAPH_FRAME_STATS mbgraph_stats[MAX_LAG_BUFFERS];
  int mbgraph_n_frames;  // number of frames filled in the above
  int static_mb_pct;     // % forced skip mbs by segmentation
  int ref_frame_flags;

  SPEED_FEATURES sf;

  uint32_t max_mv_magnitude;
  int mv_step_param;

  int allow_comp_inter_inter;

  // Default value is 1. From first pass stats, encode_breakout may be disabled.
  ENCODE_BREAKOUT_TYPE allow_encode_breakout;

  // Get threshold from external input. A suggested threshold is 800 for HD
  // clips, and 300 for < HD clips.
  int encode_breakout;

  uint8_t *segmentation_map;

  uint8_t *skin_map;

  // segment threshold for encode breakout
  int segment_encode_breakout[MAX_SEGMENTS];

  CYCLIC_REFRESH *cyclic_refresh;
  ActiveMap active_map;

  fractional_mv_step_fp *find_fractional_mv_step;
  struct scale_factors me_sf;
  vp9_diamond_search_fn_t diamond_search_sad;
  vp9_variance_fn_ptr_t fn_ptr[BLOCK_SIZES];
#if CONFIG_INTERNAL_STATS
  uint64_t time_receive_data;
  uint64_t time_compress_data;
  uint64_t time_pick_lpf;
  uint64_t time_encode_sb_row;
#endif

  TWO_PASS twopass;

  // Force recalculation of segment_ids for each mode info
  uint8_t force_update_segmentation;

  YV12_BUFFER_CONFIG tf_buffer;

  // class responsible for adaptive
  // quantization of altref frames
  struct ALT_REF_AQ *alt_ref_aq;

#if CONFIG_INTERNAL_STATS
  unsigned int mode_chosen_counts[MAX_MODES];

  int count;
  uint64_t total_sq_error;
  uint64_t total_samples;
  ImageStat psnr;

  uint64_t totalp_sq_error;
  uint64_t totalp_samples;
  ImageStat psnrp;

  double total_blockiness;
  double worst_blockiness;

  uint64_t bytes;
  double summed_quality;
  double summed_weights;
  double summedp_quality;
  double summedp_weights;
  unsigned int tot_recode_hits;
  double worst_ssim;

  ImageStat ssimg;
  ImageStat fastssim;
  ImageStat psnrhvs;

  int b_calculate_ssimg;
  int b_calculate_blockiness;

  int b_calculate_consistency;

  double total_inconsistency;
  double worst_consistency;
  Ssimv *ssim_vars;
  Metrics metrics;
#endif
  int b_calculate_psnr;

  int droppable;

  int initial_width;
  int initial_height;
  int initial_mbs;  // Number of MBs in the full-size frame; to be used to
                    // normalize the firstpass stats. This will differ from the
                    // number of MBs in the current frame when the frame is
                    // scaled.

  int last_coded_width;
  int last_coded_height;

  int use_svc;

  SVC svc;

  int frame_flags;

  search_site_config ss_cfg;

  int mbmode_cost[INTRA_MODES];
  unsigned int inter_mode_cost[INTER_MODE_CONTEXTS][INTER_MODES];
  int intra_uv_mode_cost[FRAME_TYPES][INTRA_MODES][INTRA_MODES];
  int y_mode_costs[INTRA_MODES][INTRA_MODES][INTRA_MODES];
  int switchable_interp_costs[SWITCHABLE_FILTER_CONTEXTS][SWITCHABLE_FILTERS];
  int partition_cost[PARTITION_CONTEXTS][PARTITION_TYPES];
  // Indices are:  max_tx_size-1,  tx_size_ctx,    tx_size
  int tx_size_cost[TX_SIZES - 1][TX_SIZE_CONTEXTS][TX_SIZES];

#if CONFIG_VP9_TEMPORAL_DENOISING
  VP9_DENOISER denoiser;
#endif

  int resize_pending;
  RESIZE_STATE resize_state;
  int external_resize;
  int resize_scale_num;
  int resize_scale_den;
  int resize_avg_qp;
  int resize_buffer_underflow;
  int resize_count;

  int use_skin_detection;

  int target_level;

  NOISE_ESTIMATE noise_estimate;

  // Count on how many consecutive times a block uses small/zeromv for encoding.
  uint8_t *consec_zero_mv;

  // VAR_BASED_PARTITION thresholds
  // 0 - threshold_64x64; 1 - threshold_32x32;
  // 2 - threshold_16x16; 3 - vbp_threshold_8x8;
  int64_t vbp_thresholds[4];
  int64_t vbp_threshold_minmax;
  int64_t vbp_threshold_sad;
  // Threshold used for partition copy
  int64_t vbp_threshold_copy;
  BLOCK_SIZE vbp_bsize_min;

  // Multi-threading
  int num_workers;
  VPxWorker *workers;
  struct EncWorkerData *tile_thr_data;
  VP9LfSync lf_row_sync;
  struct VP9BitstreamWorkerData *vp9_bitstream_worker_data;

  int keep_level_stats;
  Vp9LevelInfo level_info;
  MultiThreadHandle multi_thread_ctxt;
  void (*row_mt_sync_read_ptr)(VP9RowMTSync *const, int, int);
  void (*row_mt_sync_write_ptr)(VP9RowMTSync *const, int, int, const int);
  ARNRFilterData arnr_filter_data;

  int row_mt;
  unsigned int row_mt_bit_exact;

  // Previous Partition Info
  BLOCK_SIZE *prev_partition;
  int8_t *prev_segment_id;
  // Used to save the status of whether a block has a low variance in
  // choose_partitioning. 0 for 64x64, 1~2 for 64x32, 3~4 for 32x64, 5~8 for
  // 32x32, 9~24 for 16x16.
  // This is for the last frame and is copied to the current frame
  // when partition copy happens.
  uint8_t *prev_variance_low;
  uint8_t *copied_frame_cnt;
  uint8_t max_copied_frame;
  // If the last frame is dropped, we don't copy partition.
  uint8_t last_frame_dropped;

  // For each superblock: keeps track of the last time (in frame distance) the
  // the superblock did not have low source sad.
  uint8_t *content_state_sb_fd;

  int compute_source_sad_onepass;

  int compute_frame_low_motion_onepass;

  LevelConstraint level_constraint;

  uint8_t *count_arf_frame_usage;
  uint8_t *count_lastgolden_frame_usage;

  int multi_layer_arf;
  vpx_roi_map_t roi;

  LOOPFILTER_CONTROL loopfilter_ctrl;
  EXT_RATECTRL ext_ratectrl;

  int fixed_qp_onepass;

  // Flag to keep track of dynamic change in deadline mode
  // (good/best/realtime).
  MODE deadline_mode_previous_frame;

  // Flag to disable scene detection when rtc rate control library is used.
  int disable_scene_detection_rtc_ratectrl;

#if CONFIG_COLLECT_COMPONENT_TIMING
  /*!
   * component_time[] are initialized to zero while encoder starts.
   */
  uint64_t component_time[kTimingComponents];
  /*!
   * Stores timing for individual components between calls of start_timing()
   * and end_timing().
   */
  struct vpx_usec_timer component_timer[kTimingComponents];
  /*!
   * frame_component_time[] are initialized to zero at beginning of each frame.
   */
  uint64_t frame_component_time[kTimingComponents];
#endif
} VP9_COMP;

typedef struct ENCODE_FRAME_RESULT {
  int show_idx;
  FRAME_UPDATE_TYPE update_type;
  int quantize_index;
} ENCODE_FRAME_RESULT;

void vp9_init_encode_frame_result(ENCODE_FRAME_RESULT *encode_frame_result);

void vp9_initialize_enc(void);

struct VP9_COMP *vp9_create_compressor(const VP9EncoderConfig *oxcf,
                                       BufferPool *const pool);
void vp9_remove_compressor(VP9_COMP *cpi);

void vp9_change_config(VP9_COMP *cpi, const VP9EncoderConfig *oxcf);

// receive a frames worth of data. caller can assume that a copy of this
// frame is made and not just a copy of the pointer..
int vp9_receive_raw_frame(VP9_COMP *cpi, vpx_enc_frame_flags_t frame_flags,
                          YV12_BUFFER_CONFIG *sd, int64_t time_stamp,
                          int64_t end_time);

int vp9_get_compressed_data(VP9_COMP *cpi, unsigned int *frame_flags,
                            size_t *size, uint8_t *dest, size_t dest_size,
                            int64_t *time_stamp, int64_t *time_end, int flush,
                            ENCODE_FRAME_RESULT *encode_frame_result);

int vp9_get_preview_raw_frame(VP9_COMP *cpi, YV12_BUFFER_CONFIG *dest,
                              vp9_ppflags_t *flags);

int vp9_use_as_reference(VP9_COMP *cpi, int ref_frame_flags);

void vp9_update_reference(VP9_COMP *cpi, int ref_frame_flags);

int vp9_copy_reference_enc(VP9_COMP *cpi, VP9_REFFRAME ref_frame_flag,
                           YV12_BUFFER_CONFIG *sd);

int vp9_set_reference_enc(VP9_COMP *cpi, VP9_REFFRAME ref_frame_flag,
                          YV12_BUFFER_CONFIG *sd);

int vp9_update_entropy(VP9_COMP *cpi, int update);

int vp9_set_active_map(VP9_COMP *cpi, unsigned char *new_map_16x16, int rows,
                       int cols);

int vp9_get_active_map(VP9_COMP *cpi, unsigned char *new_map_16x16, int rows,
                       int cols);

int vp9_set_internal_size(VP9_COMP *cpi, VPX_SCALING_MODE horiz_mode,
                          VPX_SCALING_MODE vert_mode);

int vp9_set_size_literal(VP9_COMP *cpi, unsigned int width,
                         unsigned int height);

void vp9_set_svc(VP9_COMP *cpi, int use_svc);

// Check for resetting the rc flags (rc_1_frame, rc_2_frame) if the
// configuration change has a large change in avg_frame_bandwidth.
// For SVC check for resetting based on spatial layer average bandwidth.
// Also reset buffer level to optimal level.
void vp9_check_reset_rc_flag(VP9_COMP *cpi);

void vp9_set_rc_buffer_sizes(VP9_COMP *cpi);

static INLINE int stack_pop(int *stack, int stack_size) {
  int idx;
  const int r = stack[0];
  for (idx = 1; idx < stack_size; ++idx) stack[idx - 1] = stack[idx];

  return r;
}

static INLINE int stack_top(const int *stack) { return stack[0]; }

static INLINE void stack_push(int *stack, int new_item, int stack_size) {
  int idx;
  for (idx = stack_size; idx > 0; --idx) stack[idx] = stack[idx - 1];
  stack[0] = new_item;
}

static INLINE void stack_init(int *stack, int length) {
  int idx;
  for (idx = 0; idx < length; ++idx) stack[idx] = -1;
}

int vp9_get_quantizer(const VP9_COMP *cpi);

static INLINE int frame_is_kf_gf_arf(const VP9_COMP *cpi) {
  return frame_is_intra_only(&cpi->common) || cpi->refresh_alt_ref_frame ||
         (cpi->refresh_golden_frame && !cpi->rc.is_src_frame_alt_ref);
}

static INLINE int ref_frame_to_flag(int8_t ref_frame) {
  static const int kVp9RefFlagList[4] = { 0, VP9_LAST_FLAG, VP9_GOLD_FLAG,
                                          VP9_ALT_FLAG };
  assert(ref_frame >= LAST_FRAME && ref_frame <= ALTREF_FRAME);
  return kVp9RefFlagList[ref_frame];
}

static INLINE int get_ref_frame_map_idx(const VP9_COMP *cpi,
                                        MV_REFERENCE_FRAME ref_frame) {
  if (ref_frame == LAST_FRAME) {
    return cpi->lst_fb_idx;
  } else if (ref_frame == GOLDEN_FRAME) {
    return cpi->gld_fb_idx;
  } else {
    return cpi->alt_fb_idx;
  }
}

static INLINE int get_ref_frame_buf_idx(const VP9_COMP *const cpi,
                                        int ref_frame) {
  const VP9_COMMON *const cm = &cpi->common;
  const int map_idx = get_ref_frame_map_idx(cpi, ref_frame);
  return (map_idx != INVALID_IDX) ? cm->ref_frame_map[map_idx] : INVALID_IDX;
}

static INLINE RefCntBuffer *get_ref_cnt_buffer(const VP9_COMMON *cm,
                                               int fb_idx) {
  return fb_idx != INVALID_IDX ? &cm->buffer_pool->frame_bufs[fb_idx] : NULL;
}

static INLINE void get_ref_frame_bufs(
    const VP9_COMP *cpi, RefCntBuffer *ref_frame_bufs[MAX_INTER_REF_FRAMES]) {
  const VP9_COMMON *const cm = &cpi->common;
  MV_REFERENCE_FRAME ref_frame;
  for (ref_frame = LAST_FRAME; ref_frame < MAX_REF_FRAMES; ++ref_frame) {
    int ref_frame_buf_idx = get_ref_frame_buf_idx(cpi, ref_frame);
    int inter_ref_idx = mv_ref_frame_to_inter_ref_idx(ref_frame);
    ref_frame_bufs[inter_ref_idx] = get_ref_cnt_buffer(cm, ref_frame_buf_idx);
  }
}

static INLINE YV12_BUFFER_CONFIG *get_ref_frame_buffer(
    const VP9_COMP *const cpi, MV_REFERENCE_FRAME ref_frame) {
  const VP9_COMMON *const cm = &cpi->common;
  const int buf_idx = get_ref_frame_buf_idx(cpi, ref_frame);
  return buf_idx != INVALID_IDX ? &cm->buffer_pool->frame_bufs[buf_idx].buf
                                : NULL;
}

static INLINE int get_token_alloc(int mb_rows, int mb_cols) {
  // TODO(JBB): double check we can't exceed this token count if we have a
  // 32x32 transform crossing a boundary at a multiple of 16.
  // mb_rows, cols are in units of 16 pixels. We assume 3 planes all at full
  // resolution. We assume up to 1 token per pixel, and then allow
  // a head room of 4.

  // Use aligned mb_rows and mb_cols to better align with actual token sizes.
  const int aligned_mb_rows =
      ALIGN_POWER_OF_TWO(mb_rows, MI_BLOCK_SIZE_LOG2 - 1);
  const int aligned_mb_cols =
      ALIGN_POWER_OF_TWO(mb_cols, MI_BLOCK_SIZE_LOG2 - 1);
  return aligned_mb_rows * aligned_mb_cols * (16 * 16 * 3 + 4);
}

// Get the allocated token size for a tile. It does the same calculation as in
// the frame token allocation.
static INLINE int allocated_tokens(TileInfo tile) {
  int tile_mb_rows = (tile.mi_row_end - tile.mi_row_start + 1) >> 1;
  int tile_mb_cols = (tile.mi_col_end - tile.mi_col_start + 1) >> 1;

  return get_token_alloc(tile_mb_rows, tile_mb_cols);
}

static INLINE void get_start_tok(VP9_COMP *cpi, int tile_row, int tile_col,
                                 int mi_row, TOKENEXTRA **tok) {
  VP9_COMMON *const cm = &cpi->common;
  const int tile_cols = 1 << cm->log2_tile_cols;
  TileDataEnc *this_tile = &cpi->tile_data[tile_row * tile_cols + tile_col];
  const TileInfo *const tile_info = &this_tile->tile_info;

  int tile_mb_cols = (tile_info->mi_col_end - tile_info->mi_col_start + 1) >> 1;
  const int mb_row = (mi_row - tile_info->mi_row_start) >> 1;

  *tok =
      cpi->tile_tok[tile_row][tile_col] + get_token_alloc(mb_row, tile_mb_cols);
}

int64_t vp9_get_y_sse(const YV12_BUFFER_CONFIG *a, const YV12_BUFFER_CONFIG *b);
#if CONFIG_VP9_HIGHBITDEPTH
int64_t vp9_highbd_get_y_sse(const YV12_BUFFER_CONFIG *a,
                             const YV12_BUFFER_CONFIG *b);
#endif  // CONFIG_VP9_HIGHBITDEPTH

void vp9_scale_references(VP9_COMP *cpi);

void vp9_update_reference_frames(VP9_COMP *cpi);

void vp9_get_ref_frame_info(FRAME_UPDATE_TYPE update_type, int ref_frame_flags,
                            RefCntBuffer *ref_frame_bufs[MAX_INTER_REF_FRAMES],
                            int *ref_frame_coding_indexes,
                            int *ref_frame_valid_list);

void vp9_set_high_precision_mv(VP9_COMP *cpi, int allow_high_precision_mv);

#if CONFIG_VP9_HIGHBITDEPTH
void vp9_scale_and_extend_frame_nonnormative(const YV12_BUFFER_CONFIG *src,
                                             YV12_BUFFER_CONFIG *dst, int bd);
#else
void vp9_scale_and_extend_frame_nonnormative(const YV12_BUFFER_CONFIG *src,
                                             YV12_BUFFER_CONFIG *dst);
#endif  // CONFIG_VP9_HIGHBITDEPTH

YV12_BUFFER_CONFIG *vp9_scale_if_required(
    VP9_COMMON *cm, YV12_BUFFER_CONFIG *unscaled, YV12_BUFFER_CONFIG *scaled,
    int use_normative_scaler, INTERP_FILTER filter_type, int phase_scaler);

void vp9_apply_encoding_flags(VP9_COMP *cpi, vpx_enc_frame_flags_t flags);

static INLINE int is_one_pass_svc(const struct VP9_COMP *const cpi) {
  return (cpi->use_svc && cpi->oxcf.pass == 0);
}

#if CONFIG_VP9_TEMPORAL_DENOISING
static INLINE int denoise_svc(const struct VP9_COMP *const cpi) {
  return (!cpi->use_svc || (cpi->use_svc && cpi->svc.spatial_layer_id >=
                                                cpi->svc.first_layer_denoise));
}
#endif

#define MIN_LOOKAHEAD_FOR_ARFS 4
static INLINE int is_altref_enabled(const VP9_COMP *const cpi) {
  return !(cpi->oxcf.mode == REALTIME && cpi->oxcf.rc_mode == VPX_CBR) &&
         cpi->oxcf.lag_in_frames >= MIN_LOOKAHEAD_FOR_ARFS &&
         cpi->oxcf.enable_auto_arf;
}

static INLINE void set_ref_ptrs(const VP9_COMMON *const cm, MACROBLOCKD *xd,
                                MV_REFERENCE_FRAME ref0,
                                MV_REFERENCE_FRAME ref1) {
  xd->block_refs[0] =
      &cm->frame_refs[ref0 >= LAST_FRAME ? ref0 - LAST_FRAME : 0];
  xd->block_refs[1] =
      &cm->frame_refs[ref1 >= LAST_FRAME ? ref1 - LAST_FRAME : 0];
}

static INLINE int get_chessboard_index(const int frame_index) {
  return frame_index & 0x1;
}

static INLINE int *cond_cost_list(const struct VP9_COMP *cpi, int *cost_list) {
  return cpi->sf.mv.subpel_search_method != SUBPEL_TREE ? cost_list : NULL;
}

static INLINE int get_num_vert_units(TileInfo tile, int shift) {
  int num_vert_units =
      (tile.mi_row_end - tile.mi_row_start + (1 << shift) - 1) >> shift;
  return num_vert_units;
}

static INLINE int get_num_cols(TileInfo tile, int shift) {
  int num_cols =
      (tile.mi_col_end - tile.mi_col_start + (1 << shift) - 1) >> shift;
  return num_cols;
}

static INLINE int get_level_index(VP9_LEVEL level) {
  int i;
  for (i = 0; i < VP9_LEVELS; ++i) {
    if (level == vp9_level_defs[i].level) return i;
  }
  return -1;
}

// Return the log2 value of max column tiles corresponding to the level that
// the picture size fits into.
static INLINE int log_tile_cols_from_picsize_level(uint32_t width,
                                                   uint32_t height) {
  int i;
  const uint32_t pic_size = width * height;
  const uint32_t pic_breadth = VPXMAX(width, height);
  for (i = 0; i < VP9_LEVELS; ++i) {
    if (vp9_level_defs[i].max_luma_picture_size >= pic_size &&
        vp9_level_defs[i].max_luma_picture_breadth >= pic_breadth) {
      return get_msb(vp9_level_defs[i].max_col_tiles);
    }
  }
  return INT_MAX;
}

VP9_LEVEL vp9_get_level(const Vp9LevelSpec *const level_spec);

vpx_codec_err_t vp9_set_roi_map(VP9_COMP *cpi, unsigned char *map,
                                unsigned int rows, unsigned int cols,
                                int delta_q[8], int delta_lf[8], int skip[8],
                                int ref_frame[8]);

void vp9_new_framerate(VP9_COMP *cpi, double framerate);

void vp9_set_row_mt(VP9_COMP *cpi);

int vp9_get_psnr(const VP9_COMP *cpi, PSNR_STATS *psnr);

#define LAYER_IDS_TO_IDX(sl, tl, num_tl) ((sl) * (num_tl) + (tl))

static INLINE void alloc_frame_mvs(VP9_COMMON *const cm, int buffer_idx) {
  RefCntBuffer *const new_fb_ptr = &cm->buffer_pool->frame_bufs[buffer_idx];
  if (new_fb_ptr->mvs == NULL || new_fb_ptr->mi_rows < cm->mi_rows ||
      new_fb_ptr->mi_cols < cm->mi_cols) {
    vpx_free(new_fb_ptr->mvs);
    CHECK_MEM_ERROR(&cm->error, new_fb_ptr->mvs,
                    (MV_REF *)vpx_calloc(cm->mi_rows * cm->mi_cols,
                                         sizeof(*new_fb_ptr->mvs)));
    new_fb_ptr->mi_rows = cm->mi_rows;
    new_fb_ptr->mi_cols = cm->mi_cols;
  }
}

static INLINE int mv_cost(const MV *mv, const int *joint_cost,
                          int *const comp_cost[2]) {
  assert(mv->row >= -MV_MAX && mv->row < MV_MAX);
  assert(mv->col >= -MV_MAX && mv->col < MV_MAX);
  return joint_cost[vp9_get_mv_joint(mv)] + comp_cost[0][mv->row] +
         comp_cost[1][mv->col];
}

static INLINE int mvsad_err_cost(const MACROBLOCK *x, const MV *mv,
                                 const MV *ref, int sad_per_bit) {
  MV diff;
  diff.row = mv->row - ref->row;
  diff.col = mv->col - ref->col;
  return ROUND_POWER_OF_TWO(
      (unsigned)mv_cost(&diff, x->nmvjointsadcost, x->nmvsadcost) * sad_per_bit,
      VP9_PROB_COST_SHIFT);
}

static INLINE uint32_t get_start_mv_sad(const MACROBLOCK *x, const MV *mvp_full,
                                        const MV *ref_mv_full,
                                        vpx_sad_fn_t sad_fn_ptr, int sadpb) {
  const int src_buf_stride = x->plane[0].src.stride;
  const uint8_t *const src_buf = x->plane[0].src.buf;
  const MACROBLOCKD *const xd = &x->e_mbd;
  const int pred_buf_stride = xd->plane[0].pre[0].stride;
  const uint8_t *const pred_buf =
      xd->plane[0].pre[0].buf + mvp_full->row * pred_buf_stride + mvp_full->col;
  uint32_t start_mv_sad =
      sad_fn_ptr(src_buf, src_buf_stride, pred_buf, pred_buf_stride);
  start_mv_sad += mvsad_err_cost(x, mvp_full, ref_mv_full, sadpb);

  return start_mv_sad;
}

static INLINE int num_4x4_to_edge(int plane_4x4_dim, int mb_to_edge_dim,
                                  int subsampling_dim, int blk_dim) {
  return plane_4x4_dim + (mb_to_edge_dim >> (5 + subsampling_dim)) - blk_dim;
}

// Compute the sum of squares on all visible 4x4s in the transform block.
static int64_t sum_squares_visible(const MACROBLOCKD *xd,
                                   const struct macroblockd_plane *const pd,
                                   const int16_t *diff, const int diff_stride,
                                   int blk_row, int blk_col,
                                   const BLOCK_SIZE plane_bsize,
                                   const BLOCK_SIZE tx_bsize,
                                   int *visible_width, int *visible_height) {
  int64_t sse;
  const int plane_4x4_w = num_4x4_blocks_wide_lookup[plane_bsize];
  const int plane_4x4_h = num_4x4_blocks_high_lookup[plane_bsize];
  const int tx_4x4_w = num_4x4_blocks_wide_lookup[tx_bsize];
  const int tx_4x4_h = num_4x4_blocks_high_lookup[tx_bsize];
  const int b4x4s_to_right_edge = num_4x4_to_edge(
      plane_4x4_w, xd->mb_to_right_edge, pd->subsampling_x, blk_col);
  const int b4x4s_to_bottom_edge = num_4x4_to_edge(
      plane_4x4_h, xd->mb_to_bottom_edge, pd->subsampling_y, blk_row);
  if (tx_bsize == BLOCK_4X4 ||
      (b4x4s_to_right_edge >= tx_4x4_w && b4x4s_to_bottom_edge >= tx_4x4_h)) {
    assert(tx_4x4_w == tx_4x4_h);
    sse = (int64_t)vpx_sum_squares_2d_i16(diff, diff_stride, tx_4x4_w << 2);
    *visible_width = tx_4x4_w << 2;
    *visible_height = tx_4x4_h << 2;
  } else {
    int r, c;
    const int max_r = VPXMIN(b4x4s_to_bottom_edge, tx_4x4_h);
    const int max_c = VPXMIN(b4x4s_to_right_edge, tx_4x4_w);
    sse = 0;
    // if we are in the unrestricted motion border.
    for (r = 0; r < max_r; ++r) {
      // Skip visiting the sub blocks that are wholly within the UMV.
      for (c = 0; c < max_c; ++c) {
        sse += (int64_t)vpx_sum_squares_2d_i16(
            diff + r * diff_stride * 4 + c * 4, diff_stride, 4);
      }
    }
    *visible_width = max_c << 2;
    *visible_height = max_r << 2;
  }
  return sse;
}

// Check if trellis coefficient optimization of the transform block is enabled.
static INLINE int do_trellis_opt(const struct macroblockd_plane *pd,
                                 const int16_t *src_diff, int diff_stride,
                                 int blk_row, int blk_col,
                                 BLOCK_SIZE plane_bsize, TX_SIZE tx_size,
                                 void *arg) {
  const struct encode_b_args *const args = (struct encode_b_args *)arg;
  const MACROBLOCK *const x = args->x;

  switch (args->enable_trellis_opt) {
    case DISABLE_TRELLIS_OPT: return 0;
    case ENABLE_TRELLIS_OPT: return 1;
    case ENABLE_TRELLIS_OPT_TX_RD_SRC_VAR: {
      vpx_clear_system_state();

      return (args->trellis_opt_thresh > 0.0)
                 ? (x->log_block_src_var <= args->trellis_opt_thresh)
                 : 1;
    }
    case ENABLE_TRELLIS_OPT_TX_RD_RESIDUAL_MSE: {
      const MACROBLOCKD *const xd = &x->e_mbd;
      const BLOCK_SIZE tx_bsize = txsize_to_bsize[tx_size];
#if CONFIG_VP9_HIGHBITDEPTH
      const int dequant_shift =
          (xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH) ? xd->bd - 5 : 3;
#else
      const int dequant_shift = 3;
#endif  // CONFIG_VP9_HIGHBITDEPTH
      const int qstep = pd->dequant[1] >> dequant_shift;
      int *sse_calc_done = args->sse_calc_done;
      int64_t *sse = args->sse;
      int visible_width = 0, visible_height = 0;

      // TODO: Enable the sf for high bit-depth case
      if ((xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH) || !sse ||
          !sse_calc_done)
        return 1;

      *sse = sum_squares_visible(xd, pd, src_diff, diff_stride, blk_row,
                                 blk_col, plane_bsize, tx_bsize, &visible_width,
                                 &visible_height);
      *sse_calc_done = 1;

      vpx_clear_system_state();

      return (*(sse) <= (int64_t)visible_width * visible_height * qstep *
                            qstep * args->trellis_opt_thresh);
    }
    default: assert(0 && "Invalid trellis optimization method."); return 1;
  }
}

#if CONFIG_COLLECT_COMPONENT_TIMING
static INLINE void start_timing(VP9_COMP *cpi, int component) {
  vpx_usec_timer_start(&cpi->component_timer[component]);
}
static INLINE void end_timing(VP9_COMP *cpi, int component) {
  vpx_usec_timer_mark(&cpi->component_timer[component]);
  cpi->frame_component_time[component] +=
      vpx_usec_timer_elapsed(&cpi->component_timer[component]);
}
static INLINE char const *get_frame_type_enum(int type) {
  switch (type) {
    case 0: return "KEY_FRAME";
    case 1: return "INTER_FRAME";
    default: assert(0);
  }
  return "error";
}
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_ENCODER_VP9_ENCODER_H_
