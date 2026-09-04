/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP8_ENCODER_ONYX_INT_H_
#define VPX_VP8_ENCODER_ONYX_INT_H_

#include <assert.h>
#include <stdio.h>

#include "vpx_config.h"
#include "vp8/common/onyx.h"
#include "treewriter.h"
#include "tokenize.h"
#include "vp8/common/onyxc_int.h"
#include "vpx_dsp/variance.h"
#include "vpx_util/vpx_pthread.h"
#include "encodemb.h"
#include "vp8/encoder/quantize.h"
#include "vp8/common/entropy.h"
#include "vp8/common/threading.h"
#include "vpx_ports/mem.h"
#include "vpx/internal/vpx_codec_internal.h"
#include "vpx/vp8.h"
#include "mcomp.h"
#include "vp8/common/findnearmv.h"
#include "lookahead.h"
#if CONFIG_TEMPORAL_DENOISING
#include "vp8/encoder/denoising.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define MIN_GF_INTERVAL 4
#define DEFAULT_GF_INTERVAL 7

#define KEY_FRAME_CONTEXT 5

#define MAX_LAG_BUFFERS (CONFIG_REALTIME_ONLY ? 1 : 25)

#define AF_THRESH 25
#define AF_THRESH2 100
#define ARF_DECAY_THRESH 12

#define MIN_THRESHMULT 32
#define MAX_THRESHMULT 512

#define GF_ZEROMV_ZBIN_BOOST 12
#define LF_ZEROMV_ZBIN_BOOST 6
#define MV_ZBIN_BOOST 4
#define ZBIN_OQ_MAX 192

#define VP8_TEMPORAL_ALT_REF !CONFIG_REALTIME_ONLY

/* vp8 uses 10,000,000 ticks/second as time stamp */
#define TICKS_PER_SEC 10000000

typedef struct {
  int kf_indicated;
  unsigned int frames_since_key;
  unsigned int frames_since_golden;
  int filter_level;
  int frames_till_gf_update_due;
  int recent_ref_frame_usage[MAX_REF_FRAMES];

  MV_CONTEXT mvc[2];
  int mvcosts[2][MVvals + 1];

#ifdef MODE_STATS
  int y_modes[5];
  int uv_modes[4];
  int b_modes[10];
  int inter_y_modes[10];
  int inter_uv_modes[4];
  int inter_b_modes[10];
#endif

  vp8_prob ymode_prob[4], uv_mode_prob[3]; /* interframe intra mode probs */
  vp8_prob kf_ymode_prob[4], kf_uv_mode_prob[3]; /* keyframe "" */

  int ymode_count[5], uv_mode_count[4]; /* intra MB type cts this frame */

  int count_mb_ref_frame_usage[MAX_REF_FRAMES];

  int this_frame_percent_intra;
  int last_frame_percent_intra;

} CODING_CONTEXT;

typedef struct {
  double frame;
  double intra_error;
  double coded_error;
  double ssim_weighted_pred_err;
  double pcnt_inter;
  double pcnt_motion;
  double pcnt_second_ref;
  double pcnt_neutral;
  double MVr;
  double mvr_abs;
  double MVc;
  double mvc_abs;
  double MVrv;
  double MVcv;
  double mv_in_out_count;
  double new_mv_count;
  double duration;
  double count;
} FIRSTPASS_STATS;

typedef struct {
  int frames_so_far;
  double frame_intra_error;
  double frame_coded_error;
  double frame_pcnt_inter;
  double frame_pcnt_motion;
  double frame_mvr;
  double frame_mvr_abs;
  double frame_mvc;
  double frame_mvc_abs;

} ONEPASS_FRAMESTATS;

typedef enum {
  THR_ZERO1 = 0,
  THR_DC = 1,

  THR_NEAREST1 = 2,
  THR_NEAR1 = 3,

  THR_ZERO2 = 4,
  THR_NEAREST2 = 5,

  THR_ZERO3 = 6,
  THR_NEAREST3 = 7,

  THR_NEAR2 = 8,
  THR_NEAR3 = 9,

  THR_V_PRED = 10,
  THR_H_PRED = 11,
  THR_TM = 12,

  THR_NEW1 = 13,
  THR_NEW2 = 14,
  THR_NEW3 = 15,

  THR_SPLIT1 = 16,
  THR_SPLIT2 = 17,
  THR_SPLIT3 = 18,

  THR_B_PRED = 19
} THR_MODES;

typedef enum { DIAMOND = 0, NSTEP = 1, HEX = 2 } SEARCH_METHODS;

typedef struct {
  int RD;
  SEARCH_METHODS search_method;
  int improved_quant;
  int improved_dct;
  int auto_filter;
  int recode_loop;
  int iterative_sub_pixel;
  int half_pixel_search;
  int quarter_pixel_search;
  int thresh_mult[MAX_MODES];
  int max_step_search_steps;
  int first_step;
  int optimize_coefficients;

  int use_fastquant_for_pick;
  int no_skip_block4x4_search;
  int improved_mv_pred;

} SPEED_FEATURES;

typedef struct {
  MACROBLOCK mb;
  int segment_counts[MAX_MB_SEGMENTS];
  int totalrate;
} MB_ROW_COMP;

typedef struct {
  TOKENEXTRA *start;
  TOKENEXTRA *stop;
} TOKENLIST;

typedef struct {
  int ithread;
  void *ptr1;
  void *ptr2;
} ENCODETHREAD_DATA;
typedef struct {
  int ithread;
  void *ptr1;
} LPFTHREAD_DATA;

enum {
  BLOCK_16X8,
  BLOCK_8X16,
  BLOCK_8X8,
  BLOCK_4X4,
  BLOCK_16X16,
  BLOCK_MAX_SEGMENTS
};

typedef struct {
  /* Layer configuration */
  double framerate;
  int target_bandwidth; /* bits per second */

  /* Layer specific coding parameters */
  int64_t starting_buffer_level;
  int64_t optimal_buffer_level;
  int64_t maximum_buffer_size;
  int64_t starting_buffer_level_in_ms;
  int64_t optimal_buffer_level_in_ms;
  int64_t maximum_buffer_size_in_ms;

  int avg_frame_size_for_layer;

  int64_t buffer_level;
  int64_t bits_off_target;

  int64_t total_actual_bits;
  int64_t total_target_vs_actual;

  int worst_quality;
  int active_worst_quality;
  int best_quality;
  int active_best_quality;

  int ni_av_qi;
  int ni_tot_qi;
  int ni_frames;
  int avg_frame_qindex;

  double rate_correction_factor;
  double key_frame_rate_correction_factor;
  double gf_rate_correction_factor;

  int zbin_over_quant;

  int inter_frame_target;
  int64_t total_byte_count;

  int filter_level;

  int frames_since_last_drop_overshoot;

  int force_maxqp;

  int last_frame_percent_intra;

  int count_mb_ref_frame_usage[MAX_REF_FRAMES];

  int last_q[2];
} LAYER_CONTEXT;

typedef struct VP8_COMP {
  DECLARE_ALIGNED(16, short, Y1quant[QINDEX_RANGE][16]);
  DECLARE_ALIGNED(16, short, Y1quant_shift[QINDEX_RANGE][16]);
  DECLARE_ALIGNED(16, short, Y1zbin[QINDEX_RANGE][16]);
  DECLARE_ALIGNED(16, short, Y1round[QINDEX_RANGE][16]);

  DECLARE_ALIGNED(16, short, Y2quant[QINDEX_RANGE][16]);
  DECLARE_ALIGNED(16, short, Y2quant_shift[QINDEX_RANGE][16]);
  DECLARE_ALIGNED(16, short, Y2zbin[QINDEX_RANGE][16]);
  DECLARE_ALIGNED(16, short, Y2round[QINDEX_RANGE][16]);

  DECLARE_ALIGNED(16, short, UVquant[QINDEX_RANGE][16]);
  DECLARE_ALIGNED(16, short, UVquant_shift[QINDEX_RANGE][16]);
  DECLARE_ALIGNED(16, short, UVzbin[QINDEX_RANGE][16]);
  DECLARE_ALIGNED(16, short, UVround[QINDEX_RANGE][16]);

  DECLARE_ALIGNED(16, short, zrun_zbin_boost_y1[QINDEX_RANGE][16]);
  DECLARE_ALIGNED(16, short, zrun_zbin_boost_y2[QINDEX_RANGE][16]);
  DECLARE_ALIGNED(16, short, zrun_zbin_boost_uv[QINDEX_RANGE][16]);
  DECLARE_ALIGNED(16, short, Y1quant_fast[QINDEX_RANGE][16]);
  DECLARE_ALIGNED(16, short, Y2quant_fast[QINDEX_RANGE][16]);
  DECLARE_ALIGNED(16, short, UVquant_fast[QINDEX_RANGE][16]);

  MACROBLOCK mb;
  VP8_COMMON common;
  vp8_writer bc[9]; /* one boolcoder for each partition */

  VP8_CONFIG oxcf;

  struct lookahead_ctx *lookahead;
  struct lookahead_entry *source;
  struct lookahead_entry *alt_ref_source;
  struct lookahead_entry *last_source;

  YV12_BUFFER_CONFIG *Source;
  YV12_BUFFER_CONFIG *un_scaled_source;
  YV12_BUFFER_CONFIG scaled_source;
  YV12_BUFFER_CONFIG *last_frame_unscaled_source;

  unsigned int frames_till_alt_ref_frame;
  /* frame in src_buffers has been identified to be encoded as an alt ref */
  int source_alt_ref_pending;
  /* an alt ref frame has been encoded and is usable */
  int source_alt_ref_active;
  /* source of frame to encode is an exact copy of an alt ref frame */
  int is_src_frame_alt_ref;

  /* golden frame same as last frame ( short circuit gold searches) */
  int gold_is_last;
  /* Alt reference frame same as last ( short circuit altref search) */
  int alt_is_last;
  /* don't do both alt and gold search ( just do gold). */
  int gold_is_alt;

  YV12_BUFFER_CONFIG pick_lf_lvl_frame;

  TOKENEXTRA *tok;
  unsigned int tok_count;

  unsigned int frames_since_key;
  unsigned int key_frame_frequency;
  unsigned int this_key_frame_forced;
  unsigned int next_key_frame_forced;

  /* Ambient reconstruction err target for force key frames */
  int ambient_err;

  unsigned int mode_check_freq[MAX_MODES];

  int rd_baseline_thresh[MAX_MODES];

  int RDMULT;
  int RDDIV;

  CODING_CONTEXT coding_context;

  /* Rate targeting variables */
  int64_t last_prediction_error;
  int64_t last_intra_error;

  int this_frame_target;
  int projected_frame_size;
  int last_q[2]; /* Separate values for Intra/Inter */

  double rate_correction_factor;
  double key_frame_rate_correction_factor;
  double gf_rate_correction_factor;

  int frames_since_golden;
  /* Count down till next GF */
  int frames_till_gf_update_due;

  /* GF interval chosen when we coded the last GF */
  int current_gf_interval;

  /* Total bits overspent because of GF boost (cumulative) */
  int gf_overspend_bits;

  /* Used in the few frames following a GF to recover the extra bits
   * spent in that GF
   */
  int non_gf_bitrate_adjustment;

  /* Extra bits spent on key frames that need to be recovered */
  int kf_overspend_bits;

  /* Current number of bit s to try and recover on each inter frame. */
  int kf_bitrate_adjustment;
  int max_gf_interval;
  int baseline_gf_interval;
  int active_arnr_frames;

  int64_t key_frame_count;
  int prior_key_frame_distance[KEY_FRAME_CONTEXT];
  /* Current section per frame bandwidth target */
  int per_frame_bandwidth;
  /* Average frame size target for clip */
  int av_per_frame_bandwidth;
  /* Minimum allocation that should be used for any frame */
  int min_frame_bandwidth;
  int inter_frame_target;
  double output_framerate;
  int64_t last_time_stamp_seen;
  int64_t last_end_time_stamp_seen;
  int64_t first_time_stamp_ever;

  int ni_av_qi;
  int ni_tot_qi;
  int ni_frames;
  int avg_frame_qindex;

  int64_t total_byte_count;

  int buffered_mode;

  double framerate;
  double ref_framerate;
  int64_t buffer_level;
  int64_t bits_off_target;

  int rolling_target_bits;
  int rolling_actual_bits;

  int long_rolling_target_bits;
  int long_rolling_actual_bits;

  int64_t total_actual_bits;
  int64_t total_target_vs_actual; /* debug stats */

  int worst_quality;
  int active_worst_quality;
  int best_quality;
  int active_best_quality;

  int cq_target_quality;

  int drop_frames_allowed; /* Are we permitted to drop frames? */
  int drop_frame;          /* Drop this frame? */
#if defined(DROP_UNCODED_FRAMES)
  int drop_frame_count;
#endif

  vp8_prob frame_coef_probs[BLOCK_TYPES][COEF_BANDS][PREV_COEF_CONTEXTS]
                           [ENTROPY_NODES];
  char update_probs[BLOCK_TYPES][COEF_BANDS][PREV_COEF_CONTEXTS][ENTROPY_NODES];

  unsigned int frame_branch_ct[BLOCK_TYPES][COEF_BANDS][PREV_COEF_CONTEXTS]
                              [ENTROPY_NODES][2];

  int gfu_boost;
  int kf_boost;
  int last_boost;

  int target_bandwidth; /* bits per second */
  struct vpx_codec_pkt_list *output_pkt_list;

#if 0
    /* Experimental code for lagged and one pass */
    ONEPASS_FRAMESTATS one_pass_frame_stats[MAX_LAG_BUFFERS];
    int one_pass_frame_index;
#endif

  int decimation_factor;
  int decimation_count;

  /* for real time encoding */
  int avg_encode_time;    /* microsecond */
  int avg_pick_mode_time; /* microsecond */
  int Speed;
  int compressor_speed;

  int auto_gold;
  int auto_adjust_gold_quantizer;
  int auto_worst_q;
  int cpu_used;
  int pass;

  int prob_intra_coded;
  int prob_last_coded;
  int prob_gf_coded;
  int prob_skip_false;
  int last_skip_false_probs[3];
  int last_skip_probs_q[3];
  int recent_ref_frame_usage[MAX_REF_FRAMES];

  int this_frame_percent_intra;
  int last_frame_percent_intra;

  int ref_frame_flags;

  SPEED_FEATURES sf;

  /* Count ZEROMV on all reference frames. */
  int zeromv_count;
  int lf_zeromv_pct;

  unsigned char *skin_map;

  unsigned char *segmentation_map;
  signed char segment_feature_data[MB_LVL_MAX][MAX_MB_SEGMENTS];
  unsigned int segment_encode_breakout[MAX_MB_SEGMENTS];

  unsigned char *active_map;
  unsigned int active_map_enabled;

  /* Video conferencing cyclic refresh mode flags. This is a mode
   * designed to clean up the background over time in live encoding
   * scenarious. It uses segmentation.
   */
  int cyclic_refresh_mode_enabled;
  int cyclic_refresh_mode_max_mbs_perframe;
  int cyclic_refresh_mode_index;
  int cyclic_refresh_q;
  signed char *cyclic_refresh_map;
  // Count on how many (consecutive) times a macroblock uses ZER0MV_LAST.
  unsigned char *consec_zero_last;
  // Counter that is reset when a block is checked for a mode-bias against
  // ZEROMV_LASTREF.
  unsigned char *consec_zero_last_mvbias;

  // Frame counter for the temporal pattern. Counter is rest when the temporal
  // layers are changed dynamically (run-time change).
  unsigned int temporal_pattern_counter;
  // Temporal layer id.
  int temporal_layer_id;

  // Measure of average squared difference between source and denoised signal.
  int mse_source_denoised;

  int force_maxqp;
  int frames_since_last_drop_overshoot;
  int last_pred_err_mb;

  // GF update for 1 pass cbr.
  int gf_update_onepass_cbr;
  int gf_interval_onepass_cbr;
  int gf_noboost_onepass_cbr;

#if CONFIG_MULTITHREAD
  /* multithread data */
  vpx_atomic_int *mt_current_mb_col;
  int mt_current_mb_col_size;
  int mt_sync_range;
  vpx_atomic_int b_multi_threaded;
  int encoding_thread_count;
  int b_lpf_running;

  pthread_t *h_encoding_thread;
  pthread_t h_filter_thread;

  MB_ROW_COMP *mb_row_ei;
  ENCODETHREAD_DATA *en_thread_data;
  LPFTHREAD_DATA lpf_thread_data;

  /* events */
  vp8_sem_t *h_event_start_encoding;
  vp8_sem_t *h_event_end_encoding;
  vp8_sem_t h_event_start_lpf;
  vp8_sem_t h_event_end_lpf;
#endif

  TOKENLIST *tplist;
  unsigned int partition_sz[MAX_PARTITIONS];
  unsigned char *partition_d[MAX_PARTITIONS];
  unsigned char *partition_d_end[MAX_PARTITIONS];

  fractional_mv_step_fp *find_fractional_mv_step;
  vp8_refining_search_fn_t refining_search_sad;
  vp8_diamond_search_fn_t diamond_search_sad;
  vp8_variance_fn_ptr_t fn_ptr[BLOCK_MAX_SEGMENTS];
#if CONFIG_INTERNAL_STATS
  uint64_t time_receive_data;
  uint64_t time_compress_data;
  uint64_t time_pick_lpf;
  uint64_t time_encode_mb_row;
#endif

  int base_skip_false_prob[128];

  FRAME_CONTEXT lfc_n; /* last frame entropy */
  FRAME_CONTEXT lfc_a; /* last alt ref entropy */
  FRAME_CONTEXT lfc_g; /* last gold ref entropy */

  struct twopass_rc {
    unsigned int section_intra_rating;
    double section_max_qfactor;
    unsigned int next_iiratio;
    unsigned int this_iiratio;
    FIRSTPASS_STATS total_stats;
    FIRSTPASS_STATS this_frame_stats;
    FIRSTPASS_STATS *stats_in, *stats_in_end, *stats_in_start;
    FIRSTPASS_STATS total_left_stats;
    int first_pass_done;
    int64_t bits_left;
    int64_t clip_bits_total;
    double avg_iiratio;
    double modified_error_total;
    double modified_error_used;
    double modified_error_left;
    double kf_intra_err_min;
    double gf_intra_err_min;
    int frames_to_key;
    int maxq_max_limit;
    int maxq_min_limit;
    int gf_decay_rate;
    int static_scene_max_gf_interval;
    int kf_bits;
    /* Remaining error from uncoded frames in a gf group. */
    int gf_group_error_left;
    /* Projected total bits available for a key frame group of frames */
    int64_t kf_group_bits;
    /* Error score of frames still to be coded in kf group */
    int64_t kf_group_error_left;
    /* Projected Bits available for a group including 1 GF or ARF */
    int64_t gf_group_bits;
    /* Bits for the golden frame or ARF */
    int gf_bits;
    int alt_extra_bits;
    double est_max_qcorrection_factor;
  } twopass;

#if VP8_TEMPORAL_ALT_REF
  YV12_BUFFER_CONFIG alt_ref_buffer;
  YV12_BUFFER_CONFIG *frames[MAX_LAG_BUFFERS];
  int fixed_divide[512];
#endif

#if CONFIG_INTERNAL_STATS
  int count;
  double total_y;
  double total_u;
  double total_v;
  double total;
  double total_sq_error;
  double totalp_y;
  double totalp_u;
  double totalp_v;
  double totalp;
  double total_sq_error2;
  uint64_t bytes;
  double summed_quality;
  double summed_weights;
  unsigned int tot_recode_hits;

  int b_calculate_ssimg;
#endif
  int b_calculate_psnr;

  /* Per MB activity measurement */
  unsigned int activity_avg;
  unsigned int *mb_activity_map;

  /* Record of which MBs still refer to last golden frame either
   * directly or through 0,0
   */
  unsigned char *gf_active_flags;
  int gf_active_count;

  int output_partition;

  /* Store last frame's MV info for next frame MV prediction */
  int_mv *lfmv;
  int *lf_ref_frame_sign_bias;
  int *lf_ref_frame;

  /* force next frame to intra when kf_auto says so */
  int force_next_frame_intra;

  int droppable;

  int initial_width;
  int initial_height;

#if CONFIG_TEMPORAL_DENOISING
  VP8_DENOISER denoiser;
#endif

  /* Coding layer state variables */
  unsigned int current_layer;
  LAYER_CONTEXT layer_context[VPX_TS_MAX_LAYERS];

  int64_t frames_in_layer[VPX_TS_MAX_LAYERS];
  int64_t bytes_in_layer[VPX_TS_MAX_LAYERS];
  double sum_psnr[VPX_TS_MAX_LAYERS];
  double sum_psnr_p[VPX_TS_MAX_LAYERS];
  double total_error2[VPX_TS_MAX_LAYERS];
  double total_error2_p[VPX_TS_MAX_LAYERS];
  double sum_ssim[VPX_TS_MAX_LAYERS];
  double sum_weights[VPX_TS_MAX_LAYERS];

  double total_ssimg_y_in_layer[VPX_TS_MAX_LAYERS];
  double total_ssimg_u_in_layer[VPX_TS_MAX_LAYERS];
  double total_ssimg_v_in_layer[VPX_TS_MAX_LAYERS];
  double total_ssimg_all_in_layer[VPX_TS_MAX_LAYERS];

#if CONFIG_MULTI_RES_ENCODING
  /* Number of MBs per row at lower-resolution level */
  int mr_low_res_mb_cols;
  /* Indicate if lower-res mv info is available */
  unsigned char mr_low_res_mv_avail;
#endif
  /* The frame number of each reference frames */
  unsigned int current_ref_frames[MAX_REF_FRAMES];
  // Closest reference frame to current frame.
  MV_REFERENCE_FRAME closest_reference_frame;

  struct rd_costs_struct {
    int mvcosts[2][MVvals + 1];
    int mvsadcosts[2][MVfpvals + 1];
    int mbmode_cost[2][MB_MODE_COUNT];
    int intra_uv_mode_cost[2][MB_MODE_COUNT];
    int bmode_costs[10][10][10];
    int inter_bmode_costs[B_MODE_COUNT];
    int token_costs[BLOCK_TYPES][COEF_BANDS][PREV_COEF_CONTEXTS]
                   [MAX_ENTROPY_TOKENS];
  } rd_costs;

  // Use the static threshold from ROI settings.
  int use_roi_static_threshold;

  int ext_refresh_frame_flags_pending;

  // Always update correction factor used for rate control after each frame for
  // realtime encoding.
  int rt_always_update_correction_factor;

  // Flag to indicate frame may be dropped due to large expected overshoot,
  // and re-encoded on next frame at max_qp.
  int rt_drop_recode_on_overshoot;
} VP8_COMP;

void vp8_initialize_enc(void);

void vp8_alloc_compressor_data(VP8_COMP *cpi);
int vp8_reverse_trans(int x);
void vp8_reset_temporal_layer_change(VP8_COMP *cpi, const VP8_CONFIG *oxcf,
                                     const int prev_num_layers);
void vp8_init_temporal_layer_context(VP8_COMP *cpi, const VP8_CONFIG *oxcf,
                                     const int layer,
                                     double prev_layer_framerate);
void vp8_update_layer_contexts(VP8_COMP *cpi);
void vp8_save_layer_context(VP8_COMP *cpi);
void vp8_restore_layer_context(VP8_COMP *cpi, const int layer);
void vp8_new_framerate(VP8_COMP *cpi, double framerate);
void vp8_loopfilter_frame(VP8_COMP *cpi, VP8_COMMON *cm);

void vp8_pack_bitstream(VP8_COMP *cpi, unsigned char *dest,
                        unsigned char *dest_end, size_t *size);

void vp8_tokenize_mb(VP8_COMP *, MACROBLOCK *, TOKENEXTRA **);

void vp8_set_speed_features(VP8_COMP *cpi);

int vp8_check_drop_buffer(VP8_COMP *cpi);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_ENCODER_ONYX_INT_H_
