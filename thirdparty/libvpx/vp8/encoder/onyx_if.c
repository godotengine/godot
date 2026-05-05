/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "vpx_config.h"
#include "./vpx_scale_rtcd.h"
#include "./vpx_dsp_rtcd.h"
#include "./vp8_rtcd.h"
#include "bitstream.h"
#include "vp8/common/onyxc_int.h"
#include "vp8/common/blockd.h"
#include "onyx_int.h"
#include "vp8/common/systemdependent.h"
#include "vp8/common/vp8_skin_detection.h"
#include "vp8/encoder/quantize.h"
#include "vp8/common/alloccommon.h"
#include "mcomp.h"
#include "firstpass.h"
#include "vpx_dsp/psnr.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_scale/vpx_scale.h"
#include "vp8/common/extend.h"
#include "ratectrl.h"
#include "vp8/common/quant_common.h"
#include "segmentation.h"
#if CONFIG_POSTPROC
#include "vp8/common/postproc.h"
#endif
#include "vpx_mem/vpx_mem.h"
#include "vp8/common/reconintra.h"
#include "vp8/common/swapyv12buffer.h"
#include "vp8/common/threading.h"
#include "vpx_ports/system_state.h"
#include "vpx_ports/vpx_once.h"
#include "vpx_ports/vpx_timer.h"
#include "vpx_util/vpx_write_yuv_frame.h"
#if VPX_ARCH_ARM
#include "vpx_ports/arm.h"
#endif
#if CONFIG_MULTI_RES_ENCODING
#include "mr_dissim.h"
#endif
#include "encodeframe.h"
#if CONFIG_MULTITHREAD
#include "ethreading.h"
#endif
#include "picklpf.h"
#if !CONFIG_REALTIME_ONLY
#include "temporal_filter.h"
#endif

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <limits.h>

#if CONFIG_REALTIME_ONLY & CONFIG_ONTHEFLY_BITPACKING
extern int vp8_update_coef_context(VP8_COMP *cpi);
#endif

extern unsigned int vp8_get_processor_freq(void);

int vp8_calc_ss_err(YV12_BUFFER_CONFIG *source, YV12_BUFFER_CONFIG *dest);

static void set_default_lf_deltas(VP8_COMP *cpi);

extern const int vp8_gf_interval_table[101];

#if CONFIG_INTERNAL_STATS
#include "math.h"
#include "vpx_dsp/ssim.h"
#endif

#ifdef OUTPUT_YUV_SRC
FILE *yuv_file;
#endif
#ifdef OUTPUT_YUV_DENOISED
FILE *yuv_denoised_file;
#endif
#ifdef OUTPUT_YUV_SKINMAP
static FILE *yuv_skinmap_file = NULL;
#endif

#if 0
FILE *framepsnr;
FILE *kf_list;
FILE *keyfile;
#endif

#if 0
extern int skip_true_count;
extern int skip_false_count;
#endif

#ifdef SPEEDSTATS
unsigned int frames_at_speed[16] = { 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0 };
unsigned int tot_pm = 0;
unsigned int cnt_pm = 0;
unsigned int tot_ef = 0;
unsigned int cnt_ef = 0;
#endif

#ifdef MODE_STATS
extern unsigned __int64 Sectionbits[50];
extern int y_modes[5];
extern int uv_modes[4];
extern int b_modes[10];

extern int inter_y_modes[10];
extern int inter_uv_modes[4];
extern unsigned int inter_b_modes[15];
#endif

extern const int vp8_bits_per_mb[2][QINDEX_RANGE];

extern const int qrounding_factors[129];
extern const int qzbin_factors[129];
extern void vp8cx_init_quantizer(VP8_COMP *cpi);
extern const int vp8cx_base_skip_false_prob[128];

/* Tables relating active max Q to active min Q */
static const unsigned char kf_low_motion_minq[QINDEX_RANGE] = {
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,
  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  4,  4,  4,  5,  5,  5,
  5,  5,  6,  6,  6,  6,  7,  7,  8,  8,  8,  8,  9,  9,  10, 10, 10, 10, 11,
  11, 11, 11, 12, 12, 13, 13, 13, 13, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16,
  17, 17, 18, 18, 18, 18, 19, 20, 20, 21, 21, 22, 23, 23
};
static const unsigned char kf_high_motion_minq[QINDEX_RANGE] = {
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,
  1,  1,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  5,
  5,  5,  5,  5,  5,  6,  6,  6,  6,  7,  7,  8,  8,  8,  8,  9,  9,  10, 10,
  10, 10, 11, 11, 11, 11, 12, 12, 13, 13, 13, 13, 14, 14, 15, 15, 15, 15, 16,
  16, 16, 16, 17, 17, 18, 18, 18, 18, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21,
  22, 22, 23, 23, 24, 25, 25, 26, 26, 27, 28, 28, 29, 30
};
static const unsigned char gf_low_motion_minq[QINDEX_RANGE] = {
  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,
  3,  4,  4,  4,  4,  5,  5,  5,  5,  6,  6,  6,  6,  7,  7,  7,  7,  8,  8,
  8,  8,  9,  9,  9,  9,  10, 10, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15,
  15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24,
  25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 33, 33, 34,
  34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39, 40, 40, 41, 41, 42, 42, 43, 44,
  45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58
};
static const unsigned char gf_mid_motion_minq[QINDEX_RANGE] = {
  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  2,  2,  3,  3,  3,  4,  4,  4,  5,
  5,  5,  6,  6,  6,  7,  7,  7,  8,  8,  8,  9,  9,  9,  10, 10, 10, 10, 11,
  11, 11, 12, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 16, 16, 17, 17, 18,
  18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27,
  28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37,
  37, 38, 39, 39, 40, 40, 41, 41, 42, 42, 43, 43, 44, 45, 46, 47, 48, 49, 50,
  51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64
};
static const unsigned char gf_high_motion_minq[QINDEX_RANGE] = {
  0,  0,  0,  0,  1,  1,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4,  5,
  5,  5,  6,  6,  6,  7,  7,  7,  8,  8,  8,  9,  9,  9,  10, 10, 10, 11, 11,
  12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21,
  21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30,
  31, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39, 40,
  40, 41, 41, 42, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
  57, 58, 59, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80
};
static const unsigned char inter_minq[QINDEX_RANGE] = {
  0,  0,  1,  1,  2,  3,  3,  4,  4,  5,  6,  6,  7,  8,  8,  9,  9,  10, 11,
  11, 12, 13, 13, 14, 15, 15, 16, 17, 17, 18, 19, 20, 20, 21, 22, 22, 23, 24,
  24, 25, 26, 27, 27, 28, 29, 30, 30, 31, 32, 33, 33, 34, 35, 36, 36, 37, 38,
  39, 39, 40, 41, 42, 42, 43, 44, 45, 46, 46, 47, 48, 49, 50, 50, 51, 52, 53,
  54, 55, 55, 56, 57, 58, 59, 60, 60, 61, 62, 63, 64, 65, 66, 67, 67, 68, 69,
  70, 71, 72, 73, 74, 75, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 86,
  87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100
};

#ifdef PACKET_TESTING
extern FILE *vpxlogc;
#endif

void vp8_save_layer_context(VP8_COMP *cpi) {
  LAYER_CONTEXT *lc = &cpi->layer_context[cpi->current_layer];

  /* Save layer dependent coding state */
  lc->target_bandwidth = cpi->target_bandwidth;
  lc->starting_buffer_level = cpi->oxcf.starting_buffer_level;
  lc->optimal_buffer_level = cpi->oxcf.optimal_buffer_level;
  lc->maximum_buffer_size = cpi->oxcf.maximum_buffer_size;
  lc->starting_buffer_level_in_ms = cpi->oxcf.starting_buffer_level_in_ms;
  lc->optimal_buffer_level_in_ms = cpi->oxcf.optimal_buffer_level_in_ms;
  lc->maximum_buffer_size_in_ms = cpi->oxcf.maximum_buffer_size_in_ms;
  lc->buffer_level = cpi->buffer_level;
  lc->bits_off_target = cpi->bits_off_target;
  lc->total_actual_bits = cpi->total_actual_bits;
  lc->worst_quality = cpi->worst_quality;
  lc->active_worst_quality = cpi->active_worst_quality;
  lc->best_quality = cpi->best_quality;
  lc->active_best_quality = cpi->active_best_quality;
  lc->ni_av_qi = cpi->ni_av_qi;
  lc->ni_tot_qi = cpi->ni_tot_qi;
  lc->ni_frames = cpi->ni_frames;
  lc->avg_frame_qindex = cpi->avg_frame_qindex;
  lc->rate_correction_factor = cpi->rate_correction_factor;
  lc->key_frame_rate_correction_factor = cpi->key_frame_rate_correction_factor;
  lc->gf_rate_correction_factor = cpi->gf_rate_correction_factor;
  lc->zbin_over_quant = cpi->mb.zbin_over_quant;
  lc->inter_frame_target = cpi->inter_frame_target;
  lc->total_byte_count = cpi->total_byte_count;
  lc->filter_level = cpi->common.filter_level;
  lc->frames_since_last_drop_overshoot = cpi->frames_since_last_drop_overshoot;
  lc->force_maxqp = cpi->force_maxqp;
  lc->last_frame_percent_intra = cpi->last_frame_percent_intra;
  lc->last_q[0] = cpi->last_q[0];
  lc->last_q[1] = cpi->last_q[1];

  memcpy(lc->count_mb_ref_frame_usage, cpi->mb.count_mb_ref_frame_usage,
         sizeof(cpi->mb.count_mb_ref_frame_usage));
}

void vp8_restore_layer_context(VP8_COMP *cpi, const int layer) {
  LAYER_CONTEXT *lc = &cpi->layer_context[layer];

  /* Restore layer dependent coding state */
  cpi->current_layer = layer;
  cpi->target_bandwidth = lc->target_bandwidth;
  cpi->oxcf.target_bandwidth = lc->target_bandwidth;
  cpi->oxcf.starting_buffer_level = lc->starting_buffer_level;
  cpi->oxcf.optimal_buffer_level = lc->optimal_buffer_level;
  cpi->oxcf.maximum_buffer_size = lc->maximum_buffer_size;
  cpi->oxcf.starting_buffer_level_in_ms = lc->starting_buffer_level_in_ms;
  cpi->oxcf.optimal_buffer_level_in_ms = lc->optimal_buffer_level_in_ms;
  cpi->oxcf.maximum_buffer_size_in_ms = lc->maximum_buffer_size_in_ms;
  cpi->buffer_level = lc->buffer_level;
  cpi->bits_off_target = lc->bits_off_target;
  cpi->total_actual_bits = lc->total_actual_bits;
  cpi->active_worst_quality = lc->active_worst_quality;
  cpi->active_best_quality = lc->active_best_quality;
  cpi->ni_av_qi = lc->ni_av_qi;
  cpi->ni_tot_qi = lc->ni_tot_qi;
  cpi->ni_frames = lc->ni_frames;
  cpi->avg_frame_qindex = lc->avg_frame_qindex;
  cpi->rate_correction_factor = lc->rate_correction_factor;
  cpi->key_frame_rate_correction_factor = lc->key_frame_rate_correction_factor;
  cpi->gf_rate_correction_factor = lc->gf_rate_correction_factor;
  cpi->mb.zbin_over_quant = lc->zbin_over_quant;
  cpi->inter_frame_target = lc->inter_frame_target;
  cpi->total_byte_count = lc->total_byte_count;
  cpi->common.filter_level = lc->filter_level;
  cpi->frames_since_last_drop_overshoot = lc->frames_since_last_drop_overshoot;
  cpi->force_maxqp = lc->force_maxqp;
  cpi->last_frame_percent_intra = lc->last_frame_percent_intra;
  cpi->last_q[0] = lc->last_q[0];
  cpi->last_q[1] = lc->last_q[1];

  memcpy(cpi->mb.count_mb_ref_frame_usage, lc->count_mb_ref_frame_usage,
         sizeof(cpi->mb.count_mb_ref_frame_usage));
}

static int rescale(int val, int num, int denom) {
  int64_t llnum = num;
  int64_t llden = denom;
  int64_t llval = val;

  int64_t result = (llval * llnum / llden);
  if (result <= INT_MAX)
    return (int)result;
  else
    return INT_MAX;
}

void vp8_init_temporal_layer_context(VP8_COMP *cpi, const VP8_CONFIG *oxcf,
                                     const int layer,
                                     double prev_layer_framerate) {
  LAYER_CONTEXT *lc = &cpi->layer_context[layer];

  lc->framerate = cpi->output_framerate / cpi->oxcf.rate_decimator[layer];
  if (cpi->oxcf.target_bitrate[layer] > INT_MAX / 1000)
    lc->target_bandwidth = INT_MAX;
  else
    lc->target_bandwidth = cpi->oxcf.target_bitrate[layer] * 1000;

  lc->starting_buffer_level_in_ms = oxcf->starting_buffer_level;
  lc->optimal_buffer_level_in_ms = oxcf->optimal_buffer_level;
  lc->maximum_buffer_size_in_ms = oxcf->maximum_buffer_size;

  lc->starting_buffer_level =
      rescale((int)(oxcf->starting_buffer_level), lc->target_bandwidth, 1000);

  if (oxcf->optimal_buffer_level == 0) {
    lc->optimal_buffer_level = lc->target_bandwidth / 8;
  } else {
    lc->optimal_buffer_level =
        rescale((int)(oxcf->optimal_buffer_level), lc->target_bandwidth, 1000);
  }

  if (oxcf->maximum_buffer_size == 0) {
    lc->maximum_buffer_size = lc->target_bandwidth / 8;
  } else {
    lc->maximum_buffer_size =
        rescale((int)(oxcf->maximum_buffer_size), lc->target_bandwidth, 1000);
  }

  /* Work out the average size of a frame within this layer */
  if (layer > 0) {
    lc->avg_frame_size_for_layer =
        (int)round((cpi->oxcf.target_bitrate[layer] -
                    cpi->oxcf.target_bitrate[layer - 1]) *
                   1000 / (lc->framerate - prev_layer_framerate));
  }

  lc->active_worst_quality = cpi->oxcf.worst_allowed_q;
  lc->active_best_quality = cpi->oxcf.best_allowed_q;
  lc->avg_frame_qindex = cpi->oxcf.worst_allowed_q;

  lc->buffer_level = lc->starting_buffer_level;
  lc->bits_off_target = lc->starting_buffer_level;

  lc->total_actual_bits = 0;
  lc->ni_av_qi = 0;
  lc->ni_tot_qi = 0;
  lc->ni_frames = 0;
  lc->rate_correction_factor = 1.0;
  lc->key_frame_rate_correction_factor = 1.0;
  lc->gf_rate_correction_factor = 1.0;
  lc->inter_frame_target = 0;
}

// Upon a run-time change in temporal layers, reset the layer context parameters
// for any "new" layers. For "existing" layers, let them inherit the parameters
// from the previous layer state (at the same layer #). In future we may want
// to better map the previous layer state(s) to the "new" ones.
void vp8_reset_temporal_layer_change(VP8_COMP *cpi, const VP8_CONFIG *oxcf,
                                     const int prev_num_layers) {
  int i;
  double prev_layer_framerate = 0;
  const int curr_num_layers = cpi->oxcf.number_of_layers;
  // If the previous state was 1 layer, get current layer context from cpi.
  // We need this to set the layer context for the new layers below.
  if (prev_num_layers == 1) {
    cpi->current_layer = 0;
    vp8_save_layer_context(cpi);
  }
  for (i = 0; i < curr_num_layers; ++i) {
    LAYER_CONTEXT *lc = &cpi->layer_context[i];
    if (i >= prev_num_layers) {
      vp8_init_temporal_layer_context(cpi, oxcf, i, prev_layer_framerate);
    }
    // The initial buffer levels are set based on their starting levels.
    // We could set the buffer levels based on the previous state (normalized
    // properly by the layer bandwidths) but we would need to keep track of
    // the previous set of layer bandwidths (i.e., target_bitrate[i])
    // before the layer change. For now, reset to the starting levels.
    lc->buffer_level =
        cpi->oxcf.starting_buffer_level_in_ms * cpi->oxcf.target_bitrate[i];
    lc->bits_off_target = lc->buffer_level;
    // TDOD(marpan): Should we set the rate_correction_factor and
    // active_worst/best_quality to values derived from the previous layer
    // state (to smooth-out quality dips/rate fluctuation at transition)?

    // We need to treat the 1 layer case separately: oxcf.target_bitrate[i]
    // is not set for 1 layer, and the vp8_restore_layer_context/save_context()
    // are not called in the encoding loop, so we need to call it here to
    // pass the layer context state to |cpi|.
    if (curr_num_layers == 1) {
      lc->target_bandwidth = cpi->oxcf.target_bandwidth;
      lc->buffer_level =
          cpi->oxcf.starting_buffer_level_in_ms * lc->target_bandwidth / 1000;
      lc->bits_off_target = lc->buffer_level;
      vp8_restore_layer_context(cpi, 0);
    }
    prev_layer_framerate = cpi->output_framerate / cpi->oxcf.rate_decimator[i];
  }
}

static void setup_features(VP8_COMP *cpi) {
  // If segmentation enabled set the update flags
  if (cpi->mb.e_mbd.segmentation_enabled) {
    cpi->mb.e_mbd.update_mb_segmentation_map = 1;
    cpi->mb.e_mbd.update_mb_segmentation_data = 1;
  } else {
    cpi->mb.e_mbd.update_mb_segmentation_map = 0;
    cpi->mb.e_mbd.update_mb_segmentation_data = 0;
  }

  cpi->mb.e_mbd.mode_ref_lf_delta_enabled = 0;
  cpi->mb.e_mbd.mode_ref_lf_delta_update = 0;
  memset(cpi->mb.e_mbd.ref_lf_deltas, 0, sizeof(cpi->mb.e_mbd.ref_lf_deltas));
  memset(cpi->mb.e_mbd.mode_lf_deltas, 0, sizeof(cpi->mb.e_mbd.mode_lf_deltas));
  memset(cpi->mb.e_mbd.last_ref_lf_deltas, 0,
         sizeof(cpi->mb.e_mbd.ref_lf_deltas));
  memset(cpi->mb.e_mbd.last_mode_lf_deltas, 0,
         sizeof(cpi->mb.e_mbd.mode_lf_deltas));

  set_default_lf_deltas(cpi);
}

static void dealloc_raw_frame_buffers(VP8_COMP *cpi);

static void initialize_enc(void) {
  vpx_dsp_rtcd();
  vp8_init_intra_predictors();
}

void vp8_initialize_enc(void) { once(initialize_enc); }

static void dealloc_compressor_data(VP8_COMP *cpi) {
  vpx_free(cpi->tplist);
  cpi->tplist = NULL;

  /* Delete last frame MV storage buffers */
  vpx_free(cpi->lfmv);
  cpi->lfmv = 0;

  vpx_free(cpi->lf_ref_frame_sign_bias);
  cpi->lf_ref_frame_sign_bias = 0;

  vpx_free(cpi->lf_ref_frame);
  cpi->lf_ref_frame = 0;

  /* Delete sementation map */
  vpx_free(cpi->segmentation_map);
  cpi->segmentation_map = 0;

  vpx_free(cpi->active_map);
  cpi->active_map = 0;

  vp8_de_alloc_frame_buffers(&cpi->common);

  vp8_yv12_de_alloc_frame_buffer(&cpi->pick_lf_lvl_frame);
  vp8_yv12_de_alloc_frame_buffer(&cpi->scaled_source);
  dealloc_raw_frame_buffers(cpi);

  vpx_free(cpi->tok);
  cpi->tok = 0;

  /* Structure used to monitor GF usage */
  vpx_free(cpi->gf_active_flags);
  cpi->gf_active_flags = 0;

  /* Activity mask based per mb zbin adjustments */
  vpx_free(cpi->mb_activity_map);
  cpi->mb_activity_map = 0;

  vpx_free(cpi->mb.pip);
  cpi->mb.pip = 0;
}

static void enable_segmentation(VP8_COMP *cpi) {
  /* Set the appropriate feature bit */
  cpi->mb.e_mbd.segmentation_enabled = 1;
  cpi->mb.e_mbd.update_mb_segmentation_map = 1;
  cpi->mb.e_mbd.update_mb_segmentation_data = 1;
}
static void disable_segmentation(VP8_COMP *cpi) {
  /* Clear the appropriate feature bit */
  cpi->mb.e_mbd.segmentation_enabled = 0;
}

/* Valid values for a segment are 0 to 3
 * Segmentation map is arrange as [Rows][Columns]
 */
static void set_segmentation_map(VP8_COMP *cpi,
                                 unsigned char *segmentation_map) {
  /* Copy in the new segmentation map */
  memcpy(cpi->segmentation_map, segmentation_map,
         (cpi->common.mb_rows * cpi->common.mb_cols));

  /* Signal that the map should be updated. */
  cpi->mb.e_mbd.update_mb_segmentation_map = 1;
  cpi->mb.e_mbd.update_mb_segmentation_data = 1;
}

/* The values given for each segment can be either deltas (from the default
 * value chosen for the frame) or absolute values.
 *
 * Valid range for abs values is:
 *    (0-127 for MB_LVL_ALT_Q), (0-63 for SEGMENT_ALT_LF)
 * Valid range for delta values are:
 *    (+/-127 for MB_LVL_ALT_Q), (+/-63 for SEGMENT_ALT_LF)
 *
 * abs_delta = SEGMENT_DELTADATA (deltas)
 * abs_delta = SEGMENT_ABSDATA (use the absolute values given).
 *
 */
static void set_segment_data(VP8_COMP *cpi, signed char *feature_data,
                             unsigned char abs_delta) {
  cpi->mb.e_mbd.mb_segment_abs_delta = abs_delta;
  memcpy(cpi->segment_feature_data, feature_data,
         sizeof(cpi->segment_feature_data));
}

/* A simple function to cyclically refresh the background at a lower Q */
static void cyclic_background_refresh(VP8_COMP *cpi, int Q, int lf_adjustment) {
  unsigned char *seg_map = cpi->segmentation_map;
  signed char feature_data[MB_LVL_MAX][MAX_MB_SEGMENTS];
  int i;
  int block_count = cpi->cyclic_refresh_mode_max_mbs_perframe;
  int mbs_in_frame = cpi->common.mb_rows * cpi->common.mb_cols;

  cpi->cyclic_refresh_q = Q / 2;

  if (cpi->oxcf.screen_content_mode) {
    // Modify quality ramp-up based on Q. Above some Q level, increase the
    // number of blocks to be refreshed, and reduce it below the thredhold.
    // Turn-off under certain conditions (i.e., away from key frame, and if
    // we are at good quality (low Q) and most of the blocks were
    // skipped-encoded
    // in previous frame.
    int qp_thresh = (cpi->oxcf.screen_content_mode == 2) ? 80 : 100;
    if (Q >= qp_thresh) {
      cpi->cyclic_refresh_mode_max_mbs_perframe =
          (cpi->common.mb_rows * cpi->common.mb_cols) / 10;
    } else if (cpi->frames_since_key > 250 && Q < 20 &&
               cpi->mb.skip_true_count > (int)(0.95 * mbs_in_frame)) {
      cpi->cyclic_refresh_mode_max_mbs_perframe = 0;
    } else {
      cpi->cyclic_refresh_mode_max_mbs_perframe =
          (cpi->common.mb_rows * cpi->common.mb_cols) / 20;
    }
    block_count = cpi->cyclic_refresh_mode_max_mbs_perframe;
  }

  // Set every macroblock to be eligible for update.
  // For key frame this will reset seg map to 0.
  memset(cpi->segmentation_map, 0, mbs_in_frame);

  if (cpi->common.frame_type != KEY_FRAME && block_count > 0) {
    /* Cycle through the macro_block rows */
    /* MB loop to set local segmentation map */
    i = cpi->cyclic_refresh_mode_index;
    assert(i < mbs_in_frame);
    do {
      /* If the MB is as a candidate for clean up then mark it for
       * possible boost/refresh (segment 1) The segment id may get
       * reset to 0 later if the MB gets coded anything other than
       * last frame 0,0 as only (last frame 0,0) MBs are eligable for
       * refresh : that is to say Mbs likely to be background blocks.
       */
      if (cpi->cyclic_refresh_map[i] == 0) {
        seg_map[i] = 1;
        block_count--;
      } else if (cpi->cyclic_refresh_map[i] < 0) {
        cpi->cyclic_refresh_map[i]++;
      }

      i++;
      if (i == mbs_in_frame) i = 0;

    } while (block_count && i != cpi->cyclic_refresh_mode_index);

    cpi->cyclic_refresh_mode_index = i;

#if CONFIG_TEMPORAL_DENOISING
    if (cpi->oxcf.noise_sensitivity > 0) {
      if (cpi->denoiser.denoiser_mode == kDenoiserOnYUVAggressive &&
          Q < (int)cpi->denoiser.denoise_pars.qp_thresh &&
          (cpi->frames_since_key >
           2 * cpi->denoiser.denoise_pars.consec_zerolast)) {
        // Under aggressive denoising, use segmentation to turn off loop
        // filter below some qp thresh. The filter is reduced for all
        // blocks that have been encoded as ZEROMV LAST x frames in a row,
        // where x is set by cpi->denoiser.denoise_pars.consec_zerolast.
        // This is to avoid "dot" artifacts that can occur from repeated
        // loop filtering on noisy input source.
        cpi->cyclic_refresh_q = Q;
        // lf_adjustment = -MAX_LOOP_FILTER;
        lf_adjustment = -40;
        for (i = 0; i < mbs_in_frame; ++i) {
          seg_map[i] = (cpi->consec_zero_last[i] >
                        cpi->denoiser.denoise_pars.consec_zerolast)
                           ? 1
                           : 0;
        }
      }
    }
#endif
  }

  /* Activate segmentation. */
  cpi->mb.e_mbd.update_mb_segmentation_map = 1;
  cpi->mb.e_mbd.update_mb_segmentation_data = 1;
  enable_segmentation(cpi);

  /* Set up the quant segment data */
  feature_data[MB_LVL_ALT_Q][0] = 0;
  feature_data[MB_LVL_ALT_Q][1] = (cpi->cyclic_refresh_q - Q);
  feature_data[MB_LVL_ALT_Q][2] = 0;
  feature_data[MB_LVL_ALT_Q][3] = 0;

  /* Set up the loop segment data */
  feature_data[MB_LVL_ALT_LF][0] = 0;
  feature_data[MB_LVL_ALT_LF][1] = lf_adjustment;
  feature_data[MB_LVL_ALT_LF][2] = 0;
  feature_data[MB_LVL_ALT_LF][3] = 0;

  /* Initialise the feature data structure */
  set_segment_data(cpi, &feature_data[0][0], SEGMENT_DELTADATA);
}

static void compute_skin_map(VP8_COMP *cpi) {
  int mb_row, mb_col, num_bl;
  VP8_COMMON *cm = &cpi->common;
  const uint8_t *src_y = cpi->Source->y_buffer;
  const uint8_t *src_u = cpi->Source->u_buffer;
  const uint8_t *src_v = cpi->Source->v_buffer;
  const int src_ystride = cpi->Source->y_stride;
  const int src_uvstride = cpi->Source->uv_stride;

  const SKIN_DETECTION_BLOCK_SIZE bsize =
      (cm->Width * cm->Height <= 352 * 288) ? SKIN_8X8 : SKIN_16X16;

  for (mb_row = 0; mb_row < cm->mb_rows; mb_row++) {
    num_bl = 0;
    for (mb_col = 0; mb_col < cm->mb_cols; mb_col++) {
      const int bl_index = mb_row * cm->mb_cols + mb_col;
      cpi->skin_map[bl_index] =
          vp8_compute_skin_block(src_y, src_u, src_v, src_ystride, src_uvstride,
                                 bsize, cpi->consec_zero_last[bl_index], 0);
      num_bl++;
      src_y += 16;
      src_u += 8;
      src_v += 8;
    }
    src_y += (src_ystride << 4) - (num_bl << 4);
    src_u += (src_uvstride << 3) - (num_bl << 3);
    src_v += (src_uvstride << 3) - (num_bl << 3);
  }

  // Remove isolated skin blocks (none of its neighbors are skin) and isolated
  // non-skin blocks (all of its neighbors are skin). Skip the boundary.
  for (mb_row = 1; mb_row < cm->mb_rows - 1; mb_row++) {
    for (mb_col = 1; mb_col < cm->mb_cols - 1; mb_col++) {
      const int bl_index = mb_row * cm->mb_cols + mb_col;
      int num_neighbor = 0;
      int mi, mj;
      int non_skin_threshold = 8;

      for (mi = -1; mi <= 1; mi += 1) {
        for (mj = -1; mj <= 1; mj += 1) {
          int bl_neighbor_index = (mb_row + mi) * cm->mb_cols + mb_col + mj;
          if (cpi->skin_map[bl_neighbor_index]) num_neighbor++;
        }
      }

      if (cpi->skin_map[bl_index] && num_neighbor < 2)
        cpi->skin_map[bl_index] = 0;
      if (!cpi->skin_map[bl_index] && num_neighbor == non_skin_threshold)
        cpi->skin_map[bl_index] = 1;
    }
  }
}

static void set_default_lf_deltas(VP8_COMP *cpi) {
  cpi->mb.e_mbd.mode_ref_lf_delta_enabled = 1;
  cpi->mb.e_mbd.mode_ref_lf_delta_update = 1;

  memset(cpi->mb.e_mbd.ref_lf_deltas, 0, sizeof(cpi->mb.e_mbd.ref_lf_deltas));
  memset(cpi->mb.e_mbd.mode_lf_deltas, 0, sizeof(cpi->mb.e_mbd.mode_lf_deltas));

  /* Test of ref frame deltas */
  cpi->mb.e_mbd.ref_lf_deltas[INTRA_FRAME] = 2;
  cpi->mb.e_mbd.ref_lf_deltas[LAST_FRAME] = 0;
  cpi->mb.e_mbd.ref_lf_deltas[GOLDEN_FRAME] = -2;
  cpi->mb.e_mbd.ref_lf_deltas[ALTREF_FRAME] = -2;

  cpi->mb.e_mbd.mode_lf_deltas[0] = 4; /* BPRED */

  if (cpi->oxcf.Mode == MODE_REALTIME) {
    cpi->mb.e_mbd.mode_lf_deltas[1] = -12; /* Zero */
  } else {
    cpi->mb.e_mbd.mode_lf_deltas[1] = -2; /* Zero */
  }

  cpi->mb.e_mbd.mode_lf_deltas[2] = 2; /* New mv */
  cpi->mb.e_mbd.mode_lf_deltas[3] = 4; /* Split mv */
}

/* Convenience macros for mapping speed and mode into a continuous
 * range
 */
#define GOOD(x) ((x) + 1)
#define RT(x) ((x) + 7)

static int speed_map(int speed, const int *map) {
  int res;

  do {
    res = *map++;
  } while (speed >= *map++);
  return res;
}

static const int thresh_mult_map_znn[] = {
  /* map common to zero, nearest, and near */
  0, GOOD(2), 1500, GOOD(3), 2000, RT(0), 1000, RT(2), 2000, INT_MAX
};

static const int thresh_mult_map_vhpred[] = { 1000, GOOD(2), 1500,    GOOD(3),
                                              2000, RT(0),   1000,    RT(1),
                                              2000, RT(7),   INT_MAX, INT_MAX };

static const int thresh_mult_map_bpred[] = { 2000,    GOOD(0), 2500, GOOD(2),
                                             5000,    GOOD(3), 7500, RT(0),
                                             2500,    RT(1),   5000, RT(6),
                                             INT_MAX, INT_MAX };

static const int thresh_mult_map_tm[] = { 1000,    GOOD(2), 1500, GOOD(3),
                                          2000,    RT(0),   0,    RT(1),
                                          1000,    RT(2),   2000, RT(7),
                                          INT_MAX, INT_MAX };

static const int thresh_mult_map_new1[] = { 1000,  GOOD(2), 2000,
                                            RT(0), 2000,    INT_MAX };

static const int thresh_mult_map_new2[] = { 1000, GOOD(2), 2000, GOOD(3),
                                            2500, GOOD(5), 4000, RT(0),
                                            2000, RT(2),   2500, RT(5),
                                            4000, INT_MAX };

static const int thresh_mult_map_split1[] = {
  2500,  GOOD(0), 1700,  GOOD(2), 10000, GOOD(3), 25000, GOOD(4), INT_MAX,
  RT(0), 5000,    RT(1), 10000,   RT(2), 25000,   RT(3), INT_MAX, INT_MAX
};

static const int thresh_mult_map_split2[] = {
  5000,  GOOD(0), 4500,  GOOD(2), 20000, GOOD(3), 50000, GOOD(4), INT_MAX,
  RT(0), 10000,   RT(1), 20000,   RT(2), 50000,   RT(3), INT_MAX, INT_MAX
};

static const int mode_check_freq_map_zn2[] = {
  /* {zero,nearest}{2,3} */
  0, RT(10), 1 << 1, RT(11), 1 << 2, RT(12), 1 << 3, INT_MAX
};

static const int mode_check_freq_map_vhbpred[] = { 0, GOOD(5), 2, RT(0),
                                                   0, RT(3),   2, RT(5),
                                                   4, INT_MAX };

static const int mode_check_freq_map_near2[] = {
  0,      GOOD(5), 2,      RT(0),  0,      RT(3),  2,
  RT(10), 1 << 2,  RT(11), 1 << 3, RT(12), 1 << 4, INT_MAX
};

static const int mode_check_freq_map_new1[] = {
  0, RT(10), 1 << 1, RT(11), 1 << 2, RT(12), 1 << 3, INT_MAX
};

static const int mode_check_freq_map_new2[] = { 0,      GOOD(5), 4,      RT(0),
                                                0,      RT(3),   4,      RT(10),
                                                1 << 3, RT(11),  1 << 4, RT(12),
                                                1 << 5, INT_MAX };

static const int mode_check_freq_map_split1[] = { 0, GOOD(2), 2, GOOD(3),
                                                  7, RT(1),   2, RT(2),
                                                  7, INT_MAX };

static const int mode_check_freq_map_split2[] = { 0, GOOD(1), 2,  GOOD(2),
                                                  4, GOOD(3), 15, RT(1),
                                                  4, RT(2),   15, INT_MAX };

void vp8_set_speed_features(VP8_COMP *cpi) {
  SPEED_FEATURES *sf = &cpi->sf;
  int Mode = cpi->compressor_speed;
  int Speed = cpi->Speed;
  int Speed2;
  int i;
  VP8_COMMON *cm = &cpi->common;
  int last_improved_quant = sf->improved_quant;
  int ref_frames;

  /* Initialise default mode frequency sampling variables */
  for (i = 0; i < MAX_MODES; ++i) {
    cpi->mode_check_freq[i] = 0;
  }

  cpi->mb.mbs_tested_so_far = 0;
  cpi->mb.mbs_zero_last_dot_suppress = 0;

  /* best quality defaults */
  sf->RD = 1;
  sf->search_method = NSTEP;
  sf->improved_quant = 1;
  sf->improved_dct = 1;
  sf->auto_filter = 1;
  sf->recode_loop = 1;
  sf->quarter_pixel_search = 1;
  sf->half_pixel_search = 1;
  sf->iterative_sub_pixel = 1;
  sf->optimize_coefficients = 1;
  sf->use_fastquant_for_pick = 0;
  sf->no_skip_block4x4_search = 1;

  sf->first_step = 0;
  sf->max_step_search_steps = MAX_MVSEARCH_STEPS;
  sf->improved_mv_pred = 1;

  /* default thresholds to 0 */
  for (i = 0; i < MAX_MODES; ++i) sf->thresh_mult[i] = 0;

  /* Count enabled references */
  ref_frames = 1;
  if (cpi->ref_frame_flags & VP8_LAST_FRAME) ref_frames++;
  if (cpi->ref_frame_flags & VP8_GOLD_FRAME) ref_frames++;
  if (cpi->ref_frame_flags & VP8_ALTR_FRAME) ref_frames++;

  /* Convert speed to continuous range, with clamping */
  if (Mode == 0) {
    Speed = 0;
  } else if (Mode == 2) {
    Speed = RT(Speed);
  } else {
    if (Speed > 5) Speed = 5;
    Speed = GOOD(Speed);
  }

  sf->thresh_mult[THR_ZERO1] = sf->thresh_mult[THR_NEAREST1] =
      sf->thresh_mult[THR_NEAR1] = sf->thresh_mult[THR_DC] = 0; /* always */

  sf->thresh_mult[THR_ZERO2] = sf->thresh_mult[THR_ZERO3] =
      sf->thresh_mult[THR_NEAREST2] = sf->thresh_mult[THR_NEAREST3] =
          sf->thresh_mult[THR_NEAR2] = sf->thresh_mult[THR_NEAR3] =
              speed_map(Speed, thresh_mult_map_znn);

  sf->thresh_mult[THR_V_PRED] = sf->thresh_mult[THR_H_PRED] =
      speed_map(Speed, thresh_mult_map_vhpred);
  sf->thresh_mult[THR_B_PRED] = speed_map(Speed, thresh_mult_map_bpred);
  sf->thresh_mult[THR_TM] = speed_map(Speed, thresh_mult_map_tm);
  sf->thresh_mult[THR_NEW1] = speed_map(Speed, thresh_mult_map_new1);
  sf->thresh_mult[THR_NEW2] = sf->thresh_mult[THR_NEW3] =
      speed_map(Speed, thresh_mult_map_new2);
  sf->thresh_mult[THR_SPLIT1] = speed_map(Speed, thresh_mult_map_split1);
  sf->thresh_mult[THR_SPLIT2] = sf->thresh_mult[THR_SPLIT3] =
      speed_map(Speed, thresh_mult_map_split2);

  // Special case for temporal layers.
  // Reduce the thresholds for zero/nearest/near for GOLDEN, if GOLDEN is
  // used as second reference. We don't modify thresholds for ALTREF case
  // since ALTREF is usually used as long-term reference in temporal layers.
  if ((cpi->Speed <= 6) && (cpi->oxcf.number_of_layers > 1) &&
      (cpi->ref_frame_flags & VP8_LAST_FRAME) &&
      (cpi->ref_frame_flags & VP8_GOLD_FRAME)) {
    if (cpi->closest_reference_frame == GOLDEN_FRAME) {
      sf->thresh_mult[THR_ZERO2] = sf->thresh_mult[THR_ZERO2] >> 3;
      sf->thresh_mult[THR_NEAREST2] = sf->thresh_mult[THR_NEAREST2] >> 3;
      sf->thresh_mult[THR_NEAR2] = sf->thresh_mult[THR_NEAR2] >> 3;
    } else {
      sf->thresh_mult[THR_ZERO2] = sf->thresh_mult[THR_ZERO2] >> 1;
      sf->thresh_mult[THR_NEAREST2] = sf->thresh_mult[THR_NEAREST2] >> 1;
      sf->thresh_mult[THR_NEAR2] = sf->thresh_mult[THR_NEAR2] >> 1;
    }
  }

  cpi->mode_check_freq[THR_ZERO1] = cpi->mode_check_freq[THR_NEAREST1] =
      cpi->mode_check_freq[THR_NEAR1] = cpi->mode_check_freq[THR_TM] =
          cpi->mode_check_freq[THR_DC] = 0; /* always */

  cpi->mode_check_freq[THR_ZERO2] = cpi->mode_check_freq[THR_ZERO3] =
      cpi->mode_check_freq[THR_NEAREST2] = cpi->mode_check_freq[THR_NEAREST3] =
          speed_map(Speed, mode_check_freq_map_zn2);

  cpi->mode_check_freq[THR_NEAR2] = cpi->mode_check_freq[THR_NEAR3] =
      speed_map(Speed, mode_check_freq_map_near2);

  cpi->mode_check_freq[THR_V_PRED] = cpi->mode_check_freq[THR_H_PRED] =
      cpi->mode_check_freq[THR_B_PRED] =
          speed_map(Speed, mode_check_freq_map_vhbpred);

  // For real-time mode at speed 10 keep the mode_check_freq threshold
  // for NEW1 similar to that of speed 9.
  Speed2 = Speed;
  if (cpi->Speed == 10 && Mode == 2) Speed2 = RT(9);
  cpi->mode_check_freq[THR_NEW1] = speed_map(Speed2, mode_check_freq_map_new1);

  cpi->mode_check_freq[THR_NEW2] = cpi->mode_check_freq[THR_NEW3] =
      speed_map(Speed, mode_check_freq_map_new2);

  cpi->mode_check_freq[THR_SPLIT1] =
      speed_map(Speed, mode_check_freq_map_split1);
  cpi->mode_check_freq[THR_SPLIT2] = cpi->mode_check_freq[THR_SPLIT3] =
      speed_map(Speed, mode_check_freq_map_split2);
  Speed = cpi->Speed;
  switch (Mode) {
#if !CONFIG_REALTIME_ONLY
    case 0: /* best quality mode */
      sf->first_step = 0;
      sf->max_step_search_steps = MAX_MVSEARCH_STEPS;
      break;
    case 1:
    case 3:
      if (Speed > 0) {
        /* Disable coefficient optimization above speed 0 */
        sf->optimize_coefficients = 0;
        sf->use_fastquant_for_pick = 1;
        sf->no_skip_block4x4_search = 0;

        sf->first_step = 1;
      }

      if (Speed > 2) {
        sf->improved_quant = 0;
        sf->improved_dct = 0;

        /* Only do recode loop on key frames, golden frames and
         * alt ref frames
         */
        sf->recode_loop = 2;
      }

      if (Speed > 3) {
        sf->auto_filter = 1;
        sf->recode_loop = 0; /* recode loop off */
        sf->RD = 0;          /* Turn rd off */
      }

      if (Speed > 4) {
        sf->auto_filter = 0; /* Faster selection of loop filter */
      }

      break;
#endif
    case 2:
      sf->optimize_coefficients = 0;
      sf->recode_loop = 0;
      sf->auto_filter = 1;
      sf->iterative_sub_pixel = 1;
      sf->search_method = NSTEP;

      if (Speed > 0) {
        sf->improved_quant = 0;
        sf->improved_dct = 0;

        sf->use_fastquant_for_pick = 1;
        sf->no_skip_block4x4_search = 0;
        sf->first_step = 1;
      }

      if (Speed > 2) sf->auto_filter = 0; /* Faster selection of loop filter */

      if (Speed > 3) {
        sf->RD = 0;
        sf->auto_filter = 1;
      }

      if (Speed > 4) {
        sf->auto_filter = 0; /* Faster selection of loop filter */
        sf->search_method = HEX;
        sf->iterative_sub_pixel = 0;
      }

      if (Speed > 6) {
        unsigned int sum = 0;
        unsigned int total_mbs = cm->MBs;
        int thresh;
        unsigned int total_skip;

        int min = 2000;

        if (cpi->oxcf.encode_breakout > 2000) min = cpi->oxcf.encode_breakout;

        min >>= 7;

        for (i = 0; i < min; ++i) {
          sum += cpi->mb.error_bins[i];
        }

        total_skip = sum;
        sum = 0;

        /* i starts from 2 to make sure thresh started from 2048 */
        for (; i < 1024; ++i) {
          sum += cpi->mb.error_bins[i];

          if (10 * sum >=
              (unsigned int)(cpi->Speed - 6) * (total_mbs - total_skip)) {
            break;
          }
        }

        i--;
        thresh = (i << 7);

        if (thresh < 2000) thresh = 2000;

        if (ref_frames > 1) {
          sf->thresh_mult[THR_NEW1] = thresh;
          sf->thresh_mult[THR_NEAREST1] = thresh >> 1;
          sf->thresh_mult[THR_NEAR1] = thresh >> 1;
        }

        if (ref_frames > 2) {
          sf->thresh_mult[THR_NEW2] = thresh << 1;
          sf->thresh_mult[THR_NEAREST2] = thresh;
          sf->thresh_mult[THR_NEAR2] = thresh;
        }

        if (ref_frames > 3) {
          sf->thresh_mult[THR_NEW3] = thresh << 1;
          sf->thresh_mult[THR_NEAREST3] = thresh;
          sf->thresh_mult[THR_NEAR3] = thresh;
        }

        sf->improved_mv_pred = 0;
      }

      if (Speed > 8) sf->quarter_pixel_search = 0;

      if (cm->version == 0) {
        cm->filter_type = NORMAL_LOOPFILTER;

        if (Speed >= 14) cm->filter_type = SIMPLE_LOOPFILTER;
      } else {
        cm->filter_type = SIMPLE_LOOPFILTER;
      }

      /* This has a big hit on quality. Last resort */
      if (Speed >= 15) sf->half_pixel_search = 0;

      memset(cpi->mb.error_bins, 0, sizeof(cpi->mb.error_bins));

  } /* switch */

  /* Slow quant, dct and trellis not worthwhile for first pass
   * so make sure they are always turned off.
   */
  if (cpi->pass == 1) {
    sf->improved_quant = 0;
    sf->optimize_coefficients = 0;
    sf->improved_dct = 0;
  }

  if (cpi->sf.search_method == NSTEP) {
    vp8_init3smotion_compensation(&cpi->mb,
                                  cm->yv12_fb[cm->lst_fb_idx].y_stride);
  } else if (cpi->sf.search_method == DIAMOND) {
    vp8_init_dsmotion_compensation(&cpi->mb,
                                   cm->yv12_fb[cm->lst_fb_idx].y_stride);
  }

  if (cpi->sf.improved_dct) {
    cpi->mb.short_fdct8x4 = vp8_short_fdct8x4;
    cpi->mb.short_fdct4x4 = vp8_short_fdct4x4;
  } else {
    /* No fast FDCT defined for any platform at this time. */
    cpi->mb.short_fdct8x4 = vp8_short_fdct8x4;
    cpi->mb.short_fdct4x4 = vp8_short_fdct4x4;
  }

  cpi->mb.short_walsh4x4 = vp8_short_walsh4x4;

  if (cpi->sf.improved_quant) {
    cpi->mb.quantize_b = vp8_regular_quantize_b;
  } else {
    cpi->mb.quantize_b = vp8_fast_quantize_b;
  }
  if (cpi->sf.improved_quant != last_improved_quant) vp8cx_init_quantizer(cpi);

  if (cpi->sf.iterative_sub_pixel == 1) {
    cpi->find_fractional_mv_step = vp8_find_best_sub_pixel_step_iteratively;
  } else if (cpi->sf.quarter_pixel_search) {
    cpi->find_fractional_mv_step = vp8_find_best_sub_pixel_step;
  } else if (cpi->sf.half_pixel_search) {
    cpi->find_fractional_mv_step = vp8_find_best_half_pixel_step;
  } else {
    cpi->find_fractional_mv_step = vp8_skip_fractional_mv_step;
  }

  if (cpi->sf.optimize_coefficients == 1 && cpi->pass != 1) {
    cpi->mb.optimize = 1;
  } else {
    cpi->mb.optimize = 0;
  }

  if (cpi->common.full_pixel) {
    cpi->find_fractional_mv_step = vp8_skip_fractional_mv_step;
  }

#ifdef SPEEDSTATS
  frames_at_speed[cpi->Speed]++;
#endif
}
#undef GOOD
#undef RT

static void alloc_raw_frame_buffers(VP8_COMP *cpi) {
#if VP8_TEMPORAL_ALT_REF
  int width = (cpi->oxcf.Width + 15) & ~15;
  int height = (cpi->oxcf.Height + 15) & ~15;
#endif

  cpi->lookahead = vp8_lookahead_init(cpi->oxcf.Width, cpi->oxcf.Height,
                                      cpi->oxcf.lag_in_frames);
  if (!cpi->lookahead) {
    vpx_internal_error(&cpi->common.error, VPX_CODEC_MEM_ERROR,
                       "Failed to allocate lag buffers");
  }

#if VP8_TEMPORAL_ALT_REF

  if (vp8_yv12_alloc_frame_buffer(&cpi->alt_ref_buffer, width, height,
                                  VP8BORDERINPIXELS)) {
    vpx_internal_error(&cpi->common.error, VPX_CODEC_MEM_ERROR,
                       "Failed to allocate altref buffer");
  }

#endif
}

static void dealloc_raw_frame_buffers(VP8_COMP *cpi) {
#if VP8_TEMPORAL_ALT_REF
  vp8_yv12_de_alloc_frame_buffer(&cpi->alt_ref_buffer);
#endif
  vp8_lookahead_destroy(cpi->lookahead);
}

static int vp8_alloc_partition_data(VP8_COMP *cpi) {
  vpx_free(cpi->mb.pip);

  cpi->mb.pip =
      vpx_calloc((cpi->common.mb_cols + 1) * (cpi->common.mb_rows + 1),
                 sizeof(PARTITION_INFO));
  if (!cpi->mb.pip) return 1;

  cpi->mb.pi = cpi->mb.pip + cpi->common.mode_info_stride + 1;

  return 0;
}

void vp8_alloc_compressor_data(VP8_COMP *cpi) {
  VP8_COMMON *cm = &cpi->common;

  int width = cm->Width;
  int height = cm->Height;

  if (vp8_alloc_frame_buffers(cm, width, height)) {
    vpx_internal_error(&cpi->common.error, VPX_CODEC_MEM_ERROR,
                       "Failed to allocate frame buffers");
  }

  if (vp8_alloc_partition_data(cpi)) {
    vpx_internal_error(&cpi->common.error, VPX_CODEC_MEM_ERROR,
                       "Failed to allocate partition data");
  }

  if ((width & 0xf) != 0) width += 16 - (width & 0xf);

  if ((height & 0xf) != 0) height += 16 - (height & 0xf);

  if (vp8_yv12_alloc_frame_buffer(&cpi->pick_lf_lvl_frame, width, height,
                                  VP8BORDERINPIXELS)) {
    vpx_internal_error(&cpi->common.error, VPX_CODEC_MEM_ERROR,
                       "Failed to allocate last frame buffer");
  }

  if (vp8_yv12_alloc_frame_buffer(&cpi->scaled_source, width, height,
                                  VP8BORDERINPIXELS)) {
    vpx_internal_error(&cpi->common.error, VPX_CODEC_MEM_ERROR,
                       "Failed to allocate scaled source buffer");
  }

  vpx_free(cpi->tok);

  {
#if CONFIG_REALTIME_ONLY & CONFIG_ONTHEFLY_BITPACKING
    unsigned int tokens = 8 * 24 * 16; /* one MB for each thread */
#else
    unsigned int tokens = cm->mb_rows * cm->mb_cols * 24 * 16;
#endif
    CHECK_MEM_ERROR(&cpi->common.error, cpi->tok,
                    vpx_calloc(tokens, sizeof(*cpi->tok)));
  }

  /* Data used for real time vc mode to see if gf needs refreshing */
  cpi->zeromv_count = 0;

  /* Structures used to monitor GF usage */
  vpx_free(cpi->gf_active_flags);
  CHECK_MEM_ERROR(
      &cpi->common.error, cpi->gf_active_flags,
      vpx_calloc(sizeof(*cpi->gf_active_flags), cm->mb_rows * cm->mb_cols));
  cpi->gf_active_count = cm->mb_rows * cm->mb_cols;

  vpx_free(cpi->mb_activity_map);
  CHECK_MEM_ERROR(
      &cpi->common.error, cpi->mb_activity_map,
      vpx_calloc(sizeof(*cpi->mb_activity_map), cm->mb_rows * cm->mb_cols));

  /* allocate memory for storing last frame's MVs for MV prediction. */
  vpx_free(cpi->lfmv);
  CHECK_MEM_ERROR(
      &cpi->common.error, cpi->lfmv,
      vpx_calloc((cm->mb_rows + 2) * (cm->mb_cols + 2), sizeof(*cpi->lfmv)));
  vpx_free(cpi->lf_ref_frame_sign_bias);
  CHECK_MEM_ERROR(&cpi->common.error, cpi->lf_ref_frame_sign_bias,
                  vpx_calloc((cm->mb_rows + 2) * (cm->mb_cols + 2),
                             sizeof(*cpi->lf_ref_frame_sign_bias)));
  vpx_free(cpi->lf_ref_frame);
  CHECK_MEM_ERROR(&cpi->common.error, cpi->lf_ref_frame,
                  vpx_calloc((cm->mb_rows + 2) * (cm->mb_cols + 2),
                             sizeof(*cpi->lf_ref_frame)));

  /* Create the encoder segmentation map and set all entries to 0 */
  vpx_free(cpi->segmentation_map);
  CHECK_MEM_ERROR(
      &cpi->common.error, cpi->segmentation_map,
      vpx_calloc(cm->mb_rows * cm->mb_cols, sizeof(*cpi->segmentation_map)));
  cpi->cyclic_refresh_mode_index = 0;
  vpx_free(cpi->active_map);
  CHECK_MEM_ERROR(
      &cpi->common.error, cpi->active_map,
      vpx_calloc(cm->mb_rows * cm->mb_cols, sizeof(*cpi->active_map)));
  memset(cpi->active_map, 1, (cm->mb_rows * cm->mb_cols));

#if CONFIG_MULTITHREAD
  if (width < 640) {
    cpi->mt_sync_range = 1;
  } else if (width <= 1280) {
    cpi->mt_sync_range = 4;
  } else if (width <= 2560) {
    cpi->mt_sync_range = 8;
  } else {
    cpi->mt_sync_range = 16;
  }
#endif

  vpx_free(cpi->tplist);
  CHECK_MEM_ERROR(&cpi->common.error, cpi->tplist,
                  vpx_malloc(sizeof(TOKENLIST) * cm->mb_rows));

#if CONFIG_TEMPORAL_DENOISING
  if (cpi->oxcf.noise_sensitivity > 0) {
    vp8_denoiser_free(&cpi->denoiser);
    if (vp8_denoiser_allocate(&cpi->denoiser, width, height, cm->mb_rows,
                              cm->mb_cols, cpi->oxcf.noise_sensitivity)) {
      vpx_internal_error(&cpi->common.error, VPX_CODEC_MEM_ERROR,
                         "Failed to allocate denoiser");
    }
  }
#endif
}

/* Quant MOD */
static const int q_trans[] = {
  0,  1,  2,  3,  4,  5,  7,   8,   9,   10,  12,  13,  15,  17,  18,  19,
  20, 21, 23, 24, 25, 26, 27,  28,  29,  30,  31,  33,  35,  37,  39,  41,
  43, 45, 47, 49, 51, 53, 55,  57,  59,  61,  64,  67,  70,  73,  76,  79,
  82, 85, 88, 91, 94, 97, 100, 103, 106, 109, 112, 115, 118, 121, 124, 127,
};

int vp8_reverse_trans(int x) {
  int i;

  for (i = 0; i < 64; ++i) {
    if (q_trans[i] >= x) return i;
  }

  return 63;
}

static double clamp_framerate(double framerate) {
  if (framerate < .1)
    return 30.0;
  else
    return framerate;
}

void vp8_new_framerate(VP8_COMP *cpi, double framerate) {
  framerate = clamp_framerate(framerate);
  cpi->framerate = framerate;
  cpi->output_framerate = framerate;
  const double per_frame_bandwidth =
      round(cpi->oxcf.target_bandwidth / cpi->output_framerate);
  cpi->per_frame_bandwidth = (int)VPXMIN(per_frame_bandwidth, INT_MAX);
  cpi->av_per_frame_bandwidth = cpi->per_frame_bandwidth;
  const int64_t vbr_min_bits = (int64_t)cpi->av_per_frame_bandwidth *
                               cpi->oxcf.two_pass_vbrmin_section / 100;
  cpi->min_frame_bandwidth = (int)VPXMIN(vbr_min_bits, INT_MAX);

  /* Set Maximum gf/arf interval */
  cpi->max_gf_interval = ((int)(cpi->output_framerate / 2.0) + 2);

  if (cpi->max_gf_interval < 12) cpi->max_gf_interval = 12;

  /* Extended interval for genuinely static scenes */
  cpi->twopass.static_scene_max_gf_interval = cpi->key_frame_frequency >> 1;

  /* Special conditions when altr ref frame enabled in lagged compress mode */
  if (cpi->oxcf.play_alternate && cpi->oxcf.lag_in_frames) {
    if (cpi->max_gf_interval > cpi->oxcf.lag_in_frames - 1) {
      cpi->max_gf_interval = cpi->oxcf.lag_in_frames - 1;
    }

    if (cpi->twopass.static_scene_max_gf_interval >
        cpi->oxcf.lag_in_frames - 1) {
      cpi->twopass.static_scene_max_gf_interval = cpi->oxcf.lag_in_frames - 1;
    }
  }

  if (cpi->max_gf_interval > cpi->twopass.static_scene_max_gf_interval) {
    cpi->max_gf_interval = cpi->twopass.static_scene_max_gf_interval;
  }
}

static void init_config(VP8_COMP *cpi, const VP8_CONFIG *oxcf) {
  VP8_COMMON *cm = &cpi->common;

  cpi->oxcf = *oxcf;

  cpi->auto_gold = 1;
  cpi->auto_adjust_gold_quantizer = 1;

  cm->version = oxcf->Version;
  vp8_setup_version(cm);

  /* Frame rate is not available on the first frame, as it's derived from
   * the observed timestamps. The actual value used here doesn't matter
   * too much, as it will adapt quickly.
   */
  if (oxcf->timebase.num > 0) {
    cpi->framerate =
        (double)(oxcf->timebase.den) / (double)(oxcf->timebase.num);
  } else {
    cpi->framerate = 30;
  }

  /* If the reciprocal of the timebase seems like a reasonable framerate,
   * then use that as a guess, otherwise use 30.
   */
  if (cpi->framerate > 180) cpi->framerate = 30;

  cpi->ref_framerate = cpi->framerate;

  cpi->ref_frame_flags = VP8_ALTR_FRAME | VP8_GOLD_FRAME | VP8_LAST_FRAME;

  cm->refresh_golden_frame = 0;
  cm->refresh_last_frame = 1;
  cm->refresh_entropy_probs = 1;

  /* change includes all joint functionality */
  vp8_change_config(cpi, oxcf);

  /* Initialize active best and worst q and average q values. */
  cpi->active_worst_quality = cpi->oxcf.worst_allowed_q;
  cpi->active_best_quality = cpi->oxcf.best_allowed_q;
  cpi->avg_frame_qindex = cpi->oxcf.worst_allowed_q;

  /* Initialise the starting buffer levels */
  cpi->buffer_level = cpi->oxcf.starting_buffer_level;
  cpi->bits_off_target = cpi->oxcf.starting_buffer_level;

  cpi->rolling_target_bits = cpi->av_per_frame_bandwidth;
  cpi->rolling_actual_bits = cpi->av_per_frame_bandwidth;
  cpi->long_rolling_target_bits = cpi->av_per_frame_bandwidth;
  cpi->long_rolling_actual_bits = cpi->av_per_frame_bandwidth;

  cpi->total_actual_bits = 0;
  cpi->total_target_vs_actual = 0;

  /* Temporal scalabilty */
  if (cpi->oxcf.number_of_layers > 1) {
    unsigned int i;
    double prev_layer_framerate = 0;

    for (i = 0; i < cpi->oxcf.number_of_layers; ++i) {
      vp8_init_temporal_layer_context(cpi, oxcf, i, prev_layer_framerate);
      prev_layer_framerate =
          cpi->output_framerate / cpi->oxcf.rate_decimator[i];
    }
  }

#if VP8_TEMPORAL_ALT_REF
  {
    int i;

    cpi->fixed_divide[0] = 0;

    for (i = 1; i < 512; ++i) cpi->fixed_divide[i] = 0x80000 / i;
  }
#endif
}

void vp8_update_layer_contexts(VP8_COMP *cpi) {
  VP8_CONFIG *oxcf = &cpi->oxcf;

  /* Update snapshots of the layer contexts to reflect new parameters */
  if (oxcf->number_of_layers > 1) {
    unsigned int i;
    double prev_layer_framerate = 0;

    assert(oxcf->number_of_layers <= VPX_TS_MAX_LAYERS);
    for (i = 0; i < oxcf->number_of_layers && i < VPX_TS_MAX_LAYERS; ++i) {
      LAYER_CONTEXT *lc = &cpi->layer_context[i];

      lc->framerate = cpi->ref_framerate / oxcf->rate_decimator[i];
      if (oxcf->target_bitrate[i] > INT_MAX / 1000)
        lc->target_bandwidth = INT_MAX;
      else
        lc->target_bandwidth = oxcf->target_bitrate[i] * 1000;

      lc->starting_buffer_level = rescale(
          (int)oxcf->starting_buffer_level_in_ms, lc->target_bandwidth, 1000);

      if (oxcf->optimal_buffer_level == 0) {
        lc->optimal_buffer_level = lc->target_bandwidth / 8;
      } else {
        lc->optimal_buffer_level = rescale(
            (int)oxcf->optimal_buffer_level_in_ms, lc->target_bandwidth, 1000);
      }

      if (oxcf->maximum_buffer_size == 0) {
        lc->maximum_buffer_size = lc->target_bandwidth / 8;
      } else {
        lc->maximum_buffer_size = rescale((int)oxcf->maximum_buffer_size_in_ms,
                                          lc->target_bandwidth, 1000);
      }

      /* Work out the average size of a frame within this layer */
      if (i > 0) {
        lc->avg_frame_size_for_layer =
            (int)round((oxcf->target_bitrate[i] - oxcf->target_bitrate[i - 1]) *
                       1000 / (lc->framerate - prev_layer_framerate));
      }

      prev_layer_framerate = lc->framerate;
    }
  }
}

void vp8_change_config(VP8_COMP *cpi, const VP8_CONFIG *oxcf) {
  VP8_COMMON *cm = &cpi->common;
  int last_w, last_h;
  unsigned int prev_number_of_layers;
  double raw_target_rate;

  if (!cpi) return;

  if (!oxcf) return;

  if (cm->version != oxcf->Version) {
    cm->version = oxcf->Version;
    vp8_setup_version(cm);
  }

  last_w = cpi->oxcf.Width;
  last_h = cpi->oxcf.Height;
  prev_number_of_layers = cpi->oxcf.number_of_layers;

  cpi->oxcf = *oxcf;

  switch (cpi->oxcf.Mode) {
    case MODE_REALTIME:
      cpi->pass = 0;
      cpi->compressor_speed = 2;

      if (cpi->oxcf.cpu_used < -16) {
        cpi->oxcf.cpu_used = -16;
      }

      if (cpi->oxcf.cpu_used > 16) cpi->oxcf.cpu_used = 16;

      break;

    case MODE_GOODQUALITY:
      cpi->pass = 0;
      cpi->compressor_speed = 1;

      if (cpi->oxcf.cpu_used < -5) {
        cpi->oxcf.cpu_used = -5;
      }

      if (cpi->oxcf.cpu_used > 5) cpi->oxcf.cpu_used = 5;

      break;

    case MODE_BESTQUALITY:
      cpi->pass = 0;
      cpi->compressor_speed = 0;
      break;

    case MODE_FIRSTPASS:
      cpi->pass = 1;
      cpi->compressor_speed = 1;
      break;
    case MODE_SECONDPASS:
      cpi->pass = 2;
      cpi->compressor_speed = 1;

      if (cpi->oxcf.cpu_used < -5) {
        cpi->oxcf.cpu_used = -5;
      }

      if (cpi->oxcf.cpu_used > 5) cpi->oxcf.cpu_used = 5;

      break;
    case MODE_SECONDPASS_BEST:
      cpi->pass = 2;
      cpi->compressor_speed = 0;
      break;
  }

  if (cpi->pass == 0) cpi->auto_worst_q = 1;

  cpi->oxcf.worst_allowed_q = q_trans[oxcf->worst_allowed_q];
  cpi->oxcf.best_allowed_q = q_trans[oxcf->best_allowed_q];
  cpi->oxcf.cq_level = q_trans[cpi->oxcf.cq_level];

  if (oxcf->fixed_q >= 0) {
    if (oxcf->worst_allowed_q < 0) {
      cpi->oxcf.fixed_q = q_trans[0];
    } else {
      cpi->oxcf.fixed_q = q_trans[oxcf->worst_allowed_q];
    }

    if (oxcf->alt_q < 0) {
      cpi->oxcf.alt_q = q_trans[0];
    } else {
      cpi->oxcf.alt_q = q_trans[oxcf->alt_q];
    }

    if (oxcf->key_q < 0) {
      cpi->oxcf.key_q = q_trans[0];
    } else {
      cpi->oxcf.key_q = q_trans[oxcf->key_q];
    }

    if (oxcf->gold_q < 0) {
      cpi->oxcf.gold_q = q_trans[0];
    } else {
      cpi->oxcf.gold_q = q_trans[oxcf->gold_q];
    }
  }

  cpi->ext_refresh_frame_flags_pending = 0;

  cpi->baseline_gf_interval =
      cpi->oxcf.alt_freq ? cpi->oxcf.alt_freq : DEFAULT_GF_INTERVAL;

  // GF behavior for 1 pass CBR, used when error_resilience is off.
  if (!cpi->oxcf.error_resilient_mode &&
      cpi->oxcf.end_usage == USAGE_STREAM_FROM_SERVER &&
      cpi->oxcf.Mode == MODE_REALTIME)
    cpi->baseline_gf_interval = cpi->gf_interval_onepass_cbr;

#if (CONFIG_REALTIME_ONLY & CONFIG_ONTHEFLY_BITPACKING)
  cpi->oxcf.token_partitions = 3;
#endif

  if (cpi->oxcf.token_partitions >= 0 && cpi->oxcf.token_partitions <= 3) {
    cm->multi_token_partition = (TOKEN_PARTITION)cpi->oxcf.token_partitions;
  }

  setup_features(cpi);

  if (!cpi->use_roi_static_threshold) {
    int i;
    for (i = 0; i < MAX_MB_SEGMENTS; ++i) {
      cpi->segment_encode_breakout[i] = cpi->oxcf.encode_breakout;
    }
  }

  /* At the moment the first order values may not be > MAXQ */
  if (cpi->oxcf.fixed_q > MAXQ) cpi->oxcf.fixed_q = MAXQ;

  /* local file playback mode == really big buffer */
  if (cpi->oxcf.end_usage == USAGE_LOCAL_FILE_PLAYBACK) {
    cpi->oxcf.starting_buffer_level = 60000;
    cpi->oxcf.optimal_buffer_level = 60000;
    cpi->oxcf.maximum_buffer_size = 240000;
    cpi->oxcf.starting_buffer_level_in_ms = 60000;
    cpi->oxcf.optimal_buffer_level_in_ms = 60000;
    cpi->oxcf.maximum_buffer_size_in_ms = 240000;
  }

  raw_target_rate = ((int64_t)cpi->oxcf.Width * cpi->oxcf.Height * 8 * 3 *
                     cpi->framerate / 1000.0);
  if (cpi->oxcf.target_bandwidth > raw_target_rate)
    cpi->oxcf.target_bandwidth = (unsigned int)raw_target_rate;
  /* Convert target bandwidth from Kbit/s to Bit/s */
  cpi->oxcf.target_bandwidth *= 1000;

  cpi->oxcf.starting_buffer_level = rescale(
      (int)cpi->oxcf.starting_buffer_level, cpi->oxcf.target_bandwidth, 1000);

  /* Set or reset optimal and maximum buffer levels. */
  if (cpi->oxcf.optimal_buffer_level == 0) {
    cpi->oxcf.optimal_buffer_level = cpi->oxcf.target_bandwidth / 8;
  } else {
    cpi->oxcf.optimal_buffer_level = rescale(
        (int)cpi->oxcf.optimal_buffer_level, cpi->oxcf.target_bandwidth, 1000);
  }

  if (cpi->oxcf.maximum_buffer_size == 0) {
    cpi->oxcf.maximum_buffer_size = cpi->oxcf.target_bandwidth / 8;
  } else {
    cpi->oxcf.maximum_buffer_size = rescale((int)cpi->oxcf.maximum_buffer_size,
                                            cpi->oxcf.target_bandwidth, 1000);
  }
  // Under a configuration change, where maximum_buffer_size may change,
  // keep buffer level clipped to the maximum allowed buffer size.
  if (cpi->bits_off_target > cpi->oxcf.maximum_buffer_size) {
    cpi->bits_off_target = cpi->oxcf.maximum_buffer_size;
    cpi->buffer_level = cpi->bits_off_target;
  }

  /* Set up frame rate and related parameters rate control values. */
  vp8_new_framerate(cpi, cpi->framerate);

  /* Set absolute upper and lower quality limits */
  cpi->worst_quality = cpi->oxcf.worst_allowed_q;
  cpi->best_quality = cpi->oxcf.best_allowed_q;

  /* active values should only be modified if out of new range */
  if (cpi->active_worst_quality > cpi->oxcf.worst_allowed_q) {
    cpi->active_worst_quality = cpi->oxcf.worst_allowed_q;
  }
  /* less likely */
  else if (cpi->active_worst_quality < cpi->oxcf.best_allowed_q) {
    cpi->active_worst_quality = cpi->oxcf.best_allowed_q;
  }
  if (cpi->active_best_quality < cpi->oxcf.best_allowed_q) {
    cpi->active_best_quality = cpi->oxcf.best_allowed_q;
  }
  /* less likely */
  else if (cpi->active_best_quality > cpi->oxcf.worst_allowed_q) {
    cpi->active_best_quality = cpi->oxcf.worst_allowed_q;
  }

  cpi->buffered_mode = cpi->oxcf.optimal_buffer_level > 0;

  cpi->cq_target_quality = cpi->oxcf.cq_level;

  /* Only allow dropped frames in buffered mode */
  cpi->drop_frames_allowed = cpi->oxcf.allow_df && cpi->buffered_mode;

  cpi->target_bandwidth = cpi->oxcf.target_bandwidth;

  // Check if the number of temporal layers has changed, and if so reset the
  // pattern counter and set/initialize the temporal layer context for the
  // new layer configuration.
  if (cpi->oxcf.number_of_layers != prev_number_of_layers) {
    // If the number of temporal layers are changed we must start at the
    // base of the pattern cycle, so set the layer id to 0 and reset
    // the temporal pattern counter.
    if (cpi->temporal_layer_id > 0) {
      cpi->temporal_layer_id = 0;
    }
    cpi->temporal_pattern_counter = 0;
    vp8_reset_temporal_layer_change(cpi, oxcf, prev_number_of_layers);
  }

  if (!cpi->initial_width) {
    cpi->initial_width = cpi->oxcf.Width;
    cpi->initial_height = cpi->oxcf.Height;
  }

  cm->Width = cpi->oxcf.Width;
  cm->Height = cpi->oxcf.Height;
  assert(cm->Width <= cpi->initial_width);
  assert(cm->Height <= cpi->initial_height);

  /* TODO(jkoleszar): if an internal spatial resampling is active,
   * and we downsize the input image, maybe we should clear the
   * internal scale immediately rather than waiting for it to
   * correct.
   */

  /* VP8 sharpness level mapping 0-7 (vs 0-10 in general VPx dialogs) */
  if (cpi->oxcf.Sharpness > 7) cpi->oxcf.Sharpness = 7;

  cm->sharpness_level = cpi->oxcf.Sharpness;

  if (cm->horiz_scale != VP8E_NORMAL || cm->vert_scale != VP8E_NORMAL) {
    int hr, hs, vr, vs;

    Scale2Ratio(cm->horiz_scale, &hr, &hs);
    Scale2Ratio(cm->vert_scale, &vr, &vs);

    /* always go to the next whole number */
    cm->Width = (hs - 1 + cpi->oxcf.Width * hr) / hs;
    cm->Height = (vs - 1 + cpi->oxcf.Height * vr) / vs;
  }

  if (last_w != cpi->oxcf.Width || last_h != cpi->oxcf.Height) {
    cpi->force_next_frame_intra = 1;
  }

  if (((cm->Width + 15) & ~15) != cm->yv12_fb[cm->lst_fb_idx].y_width ||
      ((cm->Height + 15) & ~15) != cm->yv12_fb[cm->lst_fb_idx].y_height ||
      cm->yv12_fb[cm->lst_fb_idx].y_width == 0) {
    dealloc_raw_frame_buffers(cpi);
    alloc_raw_frame_buffers(cpi);
    vp8_alloc_compressor_data(cpi);
  }

  if (cpi->oxcf.fixed_q >= 0) {
    cpi->last_q[0] = cpi->oxcf.fixed_q;
    cpi->last_q[1] = cpi->oxcf.fixed_q;
  }

  cpi->Speed = cpi->oxcf.cpu_used;

  /* force to allowlag to 0 if lag_in_frames is 0; */
  if (cpi->oxcf.lag_in_frames == 0) {
    cpi->oxcf.allow_lag = 0;
  }
  /* Limit on lag buffers as these are not currently dynamically allocated */
  else if (cpi->oxcf.lag_in_frames > MAX_LAG_BUFFERS) {
    cpi->oxcf.lag_in_frames = MAX_LAG_BUFFERS;
  }

  /* YX Temp */
  cpi->alt_ref_source = NULL;
  cpi->is_src_frame_alt_ref = 0;

#if CONFIG_TEMPORAL_DENOISING
  if (cpi->oxcf.noise_sensitivity) {
    if (!cpi->denoiser.yv12_mc_running_avg.buffer_alloc) {
      int width = (cpi->oxcf.Width + 15) & ~15;
      int height = (cpi->oxcf.Height + 15) & ~15;
      if (vp8_denoiser_allocate(&cpi->denoiser, width, height, cm->mb_rows,
                                cm->mb_cols, cpi->oxcf.noise_sensitivity)) {
        vpx_internal_error(&cpi->common.error, VPX_CODEC_MEM_ERROR,
                           "Failed to allocate denoiser");
      }
    }
  }
#endif

#if 0
    /* Experimental RD Code */
    cpi->frame_distortion = 0;
    cpi->last_frame_distortion = 0;
#endif
}

#ifndef M_LOG2_E
#define M_LOG2_E 0.693147180559945309417
#endif
#define log2f(x) (log(x) / (float)M_LOG2_E)

static void cal_mvsadcosts(int *mvsadcost[2]) {
  int i = 1;

  mvsadcost[0][0] = 300;
  mvsadcost[1][0] = 300;

  do {
    double z = 256 * (2 * (log2f(8 * i) + .6));
    mvsadcost[0][i] = (int)z;
    mvsadcost[1][i] = (int)z;
    mvsadcost[0][-i] = (int)z;
    mvsadcost[1][-i] = (int)z;
  } while (++i <= mvfp_max);
}

struct VP8_COMP *vp8_create_compressor(const VP8_CONFIG *oxcf) {
  int i;

  VP8_COMP *cpi;
  VP8_COMMON *cm;

  cpi = vpx_memalign(32, sizeof(VP8_COMP));
  /* Check that the CPI instance is valid */
  if (!cpi) return 0;

  cm = &cpi->common;

  memset(cpi, 0, sizeof(VP8_COMP));

  if (setjmp(cm->error.jmp)) {
    cpi->common.error.setjmp = 0;
    vp8_remove_compressor(&cpi);
    return 0;
  }

  cpi->common.error.setjmp = 1;

  CHECK_MEM_ERROR(
      &cpi->common.error, cpi->mb.ss,
      vpx_calloc(sizeof(search_site), (MAX_MVSEARCH_STEPS * 8) + 1));

  vp8_create_common(&cpi->common);

  init_config(cpi, oxcf);

  memcpy(cpi->base_skip_false_prob, vp8cx_base_skip_false_prob,
         sizeof(vp8cx_base_skip_false_prob));
  cpi->common.current_video_frame = 0;
  cpi->temporal_pattern_counter = 0;
  cpi->temporal_layer_id = -1;
  cpi->kf_overspend_bits = 0;
  cpi->kf_bitrate_adjustment = 0;
  cpi->frames_till_gf_update_due = 0;
  cpi->gf_overspend_bits = 0;
  cpi->non_gf_bitrate_adjustment = 0;
  cpi->prob_last_coded = 128;
  cpi->prob_gf_coded = 128;
  cpi->prob_intra_coded = 63;

  /* Prime the recent reference frame usage counters.
   * Hereafter they will be maintained as a sort of moving average
   */
  cpi->recent_ref_frame_usage[INTRA_FRAME] = 1;
  cpi->recent_ref_frame_usage[LAST_FRAME] = 1;
  cpi->recent_ref_frame_usage[GOLDEN_FRAME] = 1;
  cpi->recent_ref_frame_usage[ALTREF_FRAME] = 1;

  /* Set reference frame sign bias for ALTREF frame to 1 (for now) */
  cpi->common.ref_frame_sign_bias[ALTREF_FRAME] = 1;

  cpi->twopass.gf_decay_rate = 0;
  cpi->baseline_gf_interval = DEFAULT_GF_INTERVAL;

  cpi->gold_is_last = 0;
  cpi->alt_is_last = 0;
  cpi->gold_is_alt = 0;

  cpi->active_map_enabled = 0;

  cpi->use_roi_static_threshold = 0;

#if 0
    /* Experimental code for lagged and one pass */
    /* Initialise one_pass GF frames stats */
    /* Update stats used for GF selection */
    if (cpi->pass == 0)
    {
        cpi->one_pass_frame_index = 0;

        for (i = 0; i < MAX_LAG_BUFFERS; ++i)
        {
            cpi->one_pass_frame_stats[i].frames_so_far = 0;
            cpi->one_pass_frame_stats[i].frame_intra_error = 0.0;
            cpi->one_pass_frame_stats[i].frame_coded_error = 0.0;
            cpi->one_pass_frame_stats[i].frame_pcnt_inter = 0.0;
            cpi->one_pass_frame_stats[i].frame_pcnt_motion = 0.0;
            cpi->one_pass_frame_stats[i].frame_mvr = 0.0;
            cpi->one_pass_frame_stats[i].frame_mvr_abs = 0.0;
            cpi->one_pass_frame_stats[i].frame_mvc = 0.0;
            cpi->one_pass_frame_stats[i].frame_mvc_abs = 0.0;
        }
    }
#endif

  cpi->mse_source_denoised = 0;

  /* Should we use the cyclic refresh method.
   * Currently there is no external control for this.
   * Enable it for error_resilient_mode, or for 1 pass CBR mode.
   */
  cpi->cyclic_refresh_mode_enabled =
      (cpi->oxcf.error_resilient_mode ||
       (cpi->oxcf.end_usage == USAGE_STREAM_FROM_SERVER &&
        cpi->oxcf.Mode <= 2));
  cpi->cyclic_refresh_mode_max_mbs_perframe =
      (cpi->common.mb_rows * cpi->common.mb_cols) / 7;
  if (cpi->oxcf.number_of_layers == 1) {
    cpi->cyclic_refresh_mode_max_mbs_perframe =
        (cpi->common.mb_rows * cpi->common.mb_cols) / 20;
  } else if (cpi->oxcf.number_of_layers == 2) {
    cpi->cyclic_refresh_mode_max_mbs_perframe =
        (cpi->common.mb_rows * cpi->common.mb_cols) / 10;
  }
  cpi->cyclic_refresh_mode_index = 0;
  cpi->cyclic_refresh_q = 32;

  // GF behavior for 1 pass CBR, used when error_resilience is off.
  cpi->gf_update_onepass_cbr = 0;
  cpi->gf_noboost_onepass_cbr = 0;
  if (!cpi->oxcf.error_resilient_mode &&
      cpi->oxcf.end_usage == USAGE_STREAM_FROM_SERVER && cpi->oxcf.Mode <= 2) {
    cpi->gf_update_onepass_cbr = 1;
    cpi->gf_noboost_onepass_cbr = 1;
    cpi->gf_interval_onepass_cbr =
        cpi->cyclic_refresh_mode_max_mbs_perframe > 0
            ? (2 * (cpi->common.mb_rows * cpi->common.mb_cols) /
               cpi->cyclic_refresh_mode_max_mbs_perframe)
            : 10;
    cpi->gf_interval_onepass_cbr = clamp(cpi->gf_interval_onepass_cbr, 6, 40);
    cpi->baseline_gf_interval = cpi->gf_interval_onepass_cbr;
  }

  if (cpi->cyclic_refresh_mode_enabled) {
    CHECK_MEM_ERROR(&cpi->common.error, cpi->cyclic_refresh_map,
                    vpx_calloc((cpi->common.mb_rows * cpi->common.mb_cols), 1));
  } else {
    cpi->cyclic_refresh_map = (signed char *)NULL;
  }

  CHECK_MEM_ERROR(
      &cpi->common.error, cpi->skin_map,
      vpx_calloc(cm->mb_rows * cm->mb_cols, sizeof(cpi->skin_map[0])));

  CHECK_MEM_ERROR(&cpi->common.error, cpi->consec_zero_last,
                  vpx_calloc(cm->mb_rows * cm->mb_cols, 1));
  CHECK_MEM_ERROR(&cpi->common.error, cpi->consec_zero_last_mvbias,
                  vpx_calloc((cpi->common.mb_rows * cpi->common.mb_cols), 1));

  /*Initialize the feed-forward activity masking.*/
  cpi->activity_avg = 90 << 12;

  /* Give a sensible default for the first frame. */
  cpi->frames_since_key = 8;
  cpi->key_frame_frequency = cpi->oxcf.key_freq;
  cpi->this_key_frame_forced = 0;
  cpi->next_key_frame_forced = 0;

  cpi->source_alt_ref_pending = 0;
  cpi->source_alt_ref_active = 0;
  cpi->common.refresh_alt_ref_frame = 0;

  cpi->force_maxqp = 0;
  cpi->frames_since_last_drop_overshoot = 0;
  cpi->rt_always_update_correction_factor = 0;
  cpi->rt_drop_recode_on_overshoot = 1;

  cpi->b_calculate_psnr = CONFIG_INTERNAL_STATS;
#if CONFIG_INTERNAL_STATS
  cpi->b_calculate_ssimg = 0;

  cpi->count = 0;
  cpi->bytes = 0;

  if (cpi->b_calculate_psnr) {
    cpi->total_sq_error = 0.0;
    cpi->total_sq_error2 = 0.0;
    cpi->total_y = 0.0;
    cpi->total_u = 0.0;
    cpi->total_v = 0.0;
    cpi->total = 0.0;
    cpi->totalp_y = 0.0;
    cpi->totalp_u = 0.0;
    cpi->totalp_v = 0.0;
    cpi->totalp = 0.0;
    cpi->tot_recode_hits = 0;
    cpi->summed_quality = 0;
    cpi->summed_weights = 0;
  }

#endif

  cpi->first_time_stamp_ever = 0x7FFFFFFF;

  cpi->frames_till_gf_update_due = 0;
  cpi->key_frame_count = 1;

  cpi->ni_av_qi = cpi->oxcf.worst_allowed_q;
  cpi->ni_tot_qi = 0;
  cpi->ni_frames = 0;
  cpi->total_byte_count = 0;

  cpi->drop_frame = 0;

  cpi->rate_correction_factor = 1.0;
  cpi->key_frame_rate_correction_factor = 1.0;
  cpi->gf_rate_correction_factor = 1.0;
  cpi->twopass.est_max_qcorrection_factor = 1.0;

  for (i = 0; i < KEY_FRAME_CONTEXT; ++i) {
    cpi->prior_key_frame_distance[i] = (int)cpi->output_framerate;
  }

#ifdef OUTPUT_YUV_SRC
  yuv_file = fopen("bd.yuv", "ab");
#endif
#ifdef OUTPUT_YUV_DENOISED
  yuv_denoised_file = fopen("denoised.yuv", "ab");
#endif
#ifdef OUTPUT_YUV_SKINMAP
  yuv_skinmap_file = fopen("skinmap.yuv", "wb");
#endif

#if 0
    framepsnr = fopen("framepsnr.stt", "a");
    kf_list = fopen("kf_list.stt", "w");
#endif

  cpi->output_pkt_list = oxcf->output_pkt_list;

#if !CONFIG_REALTIME_ONLY

  if (cpi->pass == 1) {
    vp8_init_first_pass(cpi);
  } else if (cpi->pass == 2) {
    size_t packet_sz = sizeof(FIRSTPASS_STATS);
    int packets = (int)(oxcf->two_pass_stats_in.sz / packet_sz);

    cpi->twopass.stats_in_start = oxcf->two_pass_stats_in.buf;
    cpi->twopass.stats_in = cpi->twopass.stats_in_start;
    cpi->twopass.stats_in_end =
        (void *)((char *)cpi->twopass.stats_in + (packets - 1) * packet_sz);
    vp8_init_second_pass(cpi);
  }

#endif

  if (cpi->compressor_speed == 2) {
    cpi->avg_encode_time = 0;
    cpi->avg_pick_mode_time = 0;
  }

  vp8_set_speed_features(cpi);

  /* Set starting values of RD threshold multipliers (128 = *1) */
  for (i = 0; i < MAX_MODES; ++i) {
    cpi->mb.rd_thresh_mult[i] = 128;
  }

#if CONFIG_MULTITHREAD
  if (vp8cx_create_encoder_threads(cpi)) {
    cpi->common.error.setjmp = 0;
    vp8_remove_compressor(&cpi);
    return 0;
  }
#endif

  cpi->fn_ptr[BLOCK_16X16].sdf = vpx_sad16x16;
  cpi->fn_ptr[BLOCK_16X16].vf = vpx_variance16x16;
  cpi->fn_ptr[BLOCK_16X16].svf = vpx_sub_pixel_variance16x16;
  cpi->fn_ptr[BLOCK_16X16].sdx4df = vpx_sad16x16x4d;

  cpi->fn_ptr[BLOCK_16X8].sdf = vpx_sad16x8;
  cpi->fn_ptr[BLOCK_16X8].vf = vpx_variance16x8;
  cpi->fn_ptr[BLOCK_16X8].svf = vpx_sub_pixel_variance16x8;
  cpi->fn_ptr[BLOCK_16X8].sdx4df = vpx_sad16x8x4d;

  cpi->fn_ptr[BLOCK_8X16].sdf = vpx_sad8x16;
  cpi->fn_ptr[BLOCK_8X16].vf = vpx_variance8x16;
  cpi->fn_ptr[BLOCK_8X16].svf = vpx_sub_pixel_variance8x16;
  cpi->fn_ptr[BLOCK_8X16].sdx4df = vpx_sad8x16x4d;

  cpi->fn_ptr[BLOCK_8X8].sdf = vpx_sad8x8;
  cpi->fn_ptr[BLOCK_8X8].vf = vpx_variance8x8;
  cpi->fn_ptr[BLOCK_8X8].svf = vpx_sub_pixel_variance8x8;
  cpi->fn_ptr[BLOCK_8X8].sdx4df = vpx_sad8x8x4d;

  cpi->fn_ptr[BLOCK_4X4].sdf = vpx_sad4x4;
  cpi->fn_ptr[BLOCK_4X4].vf = vpx_variance4x4;
  cpi->fn_ptr[BLOCK_4X4].svf = vpx_sub_pixel_variance4x4;
  cpi->fn_ptr[BLOCK_4X4].sdx4df = vpx_sad4x4x4d;

#if VPX_ARCH_X86 || VPX_ARCH_X86_64
  cpi->fn_ptr[BLOCK_16X16].copymem = vp8_copy32xn;
  cpi->fn_ptr[BLOCK_16X8].copymem = vp8_copy32xn;
  cpi->fn_ptr[BLOCK_8X16].copymem = vp8_copy32xn;
  cpi->fn_ptr[BLOCK_8X8].copymem = vp8_copy32xn;
  cpi->fn_ptr[BLOCK_4X4].copymem = vp8_copy32xn;
#endif

  cpi->diamond_search_sad = vp8_diamond_search_sad;
  cpi->refining_search_sad = vp8_refining_search_sad;

  /* make sure frame 1 is okay */
  cpi->mb.error_bins[0] = cpi->common.MBs;

  /* vp8cx_init_quantizer() is first called here. Add check in
   * vp8cx_frame_init_quantizer() so that vp8cx_init_quantizer is only
   * called later when needed. This will avoid unnecessary calls of
   * vp8cx_init_quantizer() for every frame.
   */
  vp8cx_init_quantizer(cpi);

  vp8_loop_filter_init(cm);

#if CONFIG_MULTI_RES_ENCODING

  /* Calculate # of MBs in a row in lower-resolution level image. */
  if (cpi->oxcf.mr_encoder_id > 0) vp8_cal_low_res_mb_cols(cpi);

#endif

  /* setup RD costs to MACROBLOCK struct */

  cpi->mb.mvcost[0] = &cpi->rd_costs.mvcosts[0][mv_max + 1];
  cpi->mb.mvcost[1] = &cpi->rd_costs.mvcosts[1][mv_max + 1];
  cpi->mb.mvsadcost[0] = &cpi->rd_costs.mvsadcosts[0][mvfp_max + 1];
  cpi->mb.mvsadcost[1] = &cpi->rd_costs.mvsadcosts[1][mvfp_max + 1];

  cal_mvsadcosts(cpi->mb.mvsadcost);

  cpi->mb.mbmode_cost = cpi->rd_costs.mbmode_cost;
  cpi->mb.intra_uv_mode_cost = cpi->rd_costs.intra_uv_mode_cost;
  cpi->mb.bmode_costs = cpi->rd_costs.bmode_costs;
  cpi->mb.inter_bmode_costs = cpi->rd_costs.inter_bmode_costs;
  cpi->mb.token_costs = cpi->rd_costs.token_costs;

  /* setup block ptrs & offsets */
  vp8_setup_block_ptrs(&cpi->mb);
  vp8_setup_block_dptrs(&cpi->mb.e_mbd);

  cpi->common.error.setjmp = 0;

  return cpi;
}

void vp8_remove_compressor(VP8_COMP **comp) {
  VP8_COMP *cpi = *comp;

  if (!cpi) return;

  if (cpi && (cpi->common.current_video_frame > 0)) {
#if !CONFIG_REALTIME_ONLY

    if (cpi->pass == 2) {
      vp8_end_second_pass(cpi);
    }

#endif

#if CONFIG_INTERNAL_STATS

    if (cpi->pass != 1) {
      FILE *f = fopen("opsnr.stt", "a");
      double time_encoded =
          (cpi->last_end_time_stamp_seen - cpi->first_time_stamp_ever) /
          10000000.000;

      if (cpi->b_calculate_psnr) {
        if (cpi->oxcf.number_of_layers > 1) {
          int i;

          fprintf(f,
                  "Layer\tBitrate\tAVGPsnr\tGLBPsnr\tAVPsnrP\t"
                  "GLPsnrP\tVPXSSIM\n");
          for (i = 0; i < (int)cpi->oxcf.number_of_layers; ++i) {
            double dr =
                (double)cpi->bytes_in_layer[i] * 8.0 / 1000.0 / time_encoded;
            double samples = 3.0 / 2 * cpi->frames_in_layer[i] *
                             cpi->common.Width * cpi->common.Height;
            double total_psnr =
                vpx_sse_to_psnr(samples, 255.0, cpi->total_error2[i]);
            double total_psnr2 =
                vpx_sse_to_psnr(samples, 255.0, cpi->total_error2_p[i]);
            double total_ssim =
                100 * pow(cpi->sum_ssim[i] / cpi->sum_weights[i], 8.0);

            fprintf(f,
                    "%5d\t%7.3f\t%7.3f\t%7.3f\t%7.3f\t"
                    "%7.3f\t%7.3f\n",
                    i, dr, cpi->sum_psnr[i] / cpi->frames_in_layer[i],
                    total_psnr, cpi->sum_psnr_p[i] / cpi->frames_in_layer[i],
                    total_psnr2, total_ssim);
          }
        } else {
          double dr = (double)cpi->bytes * 8.0 / 1000.0 / time_encoded;
          double samples =
              3.0 / 2 * cpi->count * cpi->common.Width * cpi->common.Height;
          double total_psnr =
              vpx_sse_to_psnr(samples, 255.0, cpi->total_sq_error);
          double total_psnr2 =
              vpx_sse_to_psnr(samples, 255.0, cpi->total_sq_error2);
          double total_ssim =
              100 * pow(cpi->summed_quality / cpi->summed_weights, 8.0);

          fprintf(f,
                  "Bitrate\tAVGPsnr\tGLBPsnr\tAVPsnrP\t"
                  "GLPsnrP\tVPXSSIM\n");
          fprintf(f,
                  "%7.3f\t%7.3f\t%7.3f\t%7.3f\t%7.3f\t"
                  "%7.3f\n",
                  dr, cpi->total / cpi->count, total_psnr,
                  cpi->totalp / cpi->count, total_psnr2, total_ssim);
        }
      }
      fclose(f);
#if 0
            f = fopen("qskip.stt", "a");
            fprintf(f, "minq:%d -maxq:%d skiptrue:skipfalse = %d:%d\n", cpi->oxcf.best_allowed_q, cpi->oxcf.worst_allowed_q, skiptruecount, skipfalsecount);
            fclose(f);
#endif
    }

#endif

#ifdef SPEEDSTATS

    if (cpi->compressor_speed == 2) {
      int i;
      FILE *f = fopen("cxspeed.stt", "a");
      cnt_pm /= cpi->common.MBs;

      for (i = 0; i < 16; ++i) fprintf(f, "%5d", frames_at_speed[i]);

      fprintf(f, "\n");
      fclose(f);
    }

#endif

#ifdef MODE_STATS
    {
      extern int count_mb_seg[4];
      FILE *f = fopen("modes.stt", "a");
      double dr = cpi->framerate * (double)bytes * (double)8 / (double)count /
                  (double)1000;
      fprintf(f, "intra_mode in Intra Frames:\n");
      fprintf(f, "Y: %8d, %8d, %8d, %8d, %8d\n", y_modes[0], y_modes[1],
              y_modes[2], y_modes[3], y_modes[4]);
      fprintf(f, "UV:%8d, %8d, %8d, %8d\n", uv_modes[0], uv_modes[1],
              uv_modes[2], uv_modes[3]);
      fprintf(f, "B: ");
      {
        int i;

        for (i = 0; i < 10; ++i) fprintf(f, "%8d, ", b_modes[i]);

        fprintf(f, "\n");
      }

      fprintf(f, "Modes in Inter Frames:\n");
      fprintf(f, "Y: %8d, %8d, %8d, %8d, %8d, %8d, %8d, %8d, %8d, %8d\n",
              inter_y_modes[0], inter_y_modes[1], inter_y_modes[2],
              inter_y_modes[3], inter_y_modes[4], inter_y_modes[5],
              inter_y_modes[6], inter_y_modes[7], inter_y_modes[8],
              inter_y_modes[9]);
      fprintf(f, "UV:%8d, %8d, %8d, %8d\n", inter_uv_modes[0],
              inter_uv_modes[1], inter_uv_modes[2], inter_uv_modes[3]);
      fprintf(f, "B: ");
      {
        int i;

        for (i = 0; i < 15; ++i) fprintf(f, "%8d, ", inter_b_modes[i]);

        fprintf(f, "\n");
      }
      fprintf(f, "P:%8d, %8d, %8d, %8d\n", count_mb_seg[0], count_mb_seg[1],
              count_mb_seg[2], count_mb_seg[3]);
      fprintf(f, "PB:%8d, %8d, %8d, %8d\n", inter_b_modes[LEFT4X4],
              inter_b_modes[ABOVE4X4], inter_b_modes[ZERO4X4],
              inter_b_modes[NEW4X4]);

      fclose(f);
    }
#endif

#if defined(SECTIONBITS_OUTPUT)

    if (0) {
      int i;
      FILE *f = fopen("tokenbits.stt", "a");

      for (i = 0; i < 28; ++i) fprintf(f, "%8d", (int)(Sectionbits[i] / 256));

      fprintf(f, "\n");
      fclose(f);
    }

#endif

#if 0
        {
            printf("\n_pick_loop_filter_level:%d\n", cpi->time_pick_lpf / 1000);
            printf("\n_frames receive_data encod_mb_row compress_frame  Total\n");
            printf("%6d %10ld %10ld %10ld %10ld\n", cpi->common.current_video_frame, cpi->time_receive_data / 1000, cpi->time_encode_mb_row / 1000, cpi->time_compress_data / 1000, (cpi->time_receive_data + cpi->time_compress_data) / 1000);
        }
#endif
  }

#if CONFIG_MULTITHREAD
  vp8cx_remove_encoder_threads(cpi);
#endif

#if CONFIG_TEMPORAL_DENOISING
  vp8_denoiser_free(&cpi->denoiser);
#endif
  dealloc_compressor_data(cpi);
  vpx_free(cpi->mb.ss);
  vpx_free(cpi->tok);
  vpx_free(cpi->skin_map);
  vpx_free(cpi->cyclic_refresh_map);
  vpx_free(cpi->consec_zero_last);
  vpx_free(cpi->consec_zero_last_mvbias);

  vp8_remove_common(&cpi->common);
  vpx_free(cpi);
  *comp = 0;

#ifdef OUTPUT_YUV_SRC
  fclose(yuv_file);
#endif
#ifdef OUTPUT_YUV_DENOISED
  fclose(yuv_denoised_file);
#endif
#ifdef OUTPUT_YUV_SKINMAP
  fclose(yuv_skinmap_file);
#endif

#if 0

    if (keyfile)
        fclose(keyfile);

    if (framepsnr)
        fclose(framepsnr);

    if (kf_list)
        fclose(kf_list);

#endif
}

static uint64_t calc_plane_error(unsigned char *orig, int orig_stride,
                                 unsigned char *recon, int recon_stride,
                                 unsigned int cols, unsigned int rows) {
  unsigned int row, col;
  uint64_t total_sse = 0;
  int diff;

  for (row = 0; row + 16 <= rows; row += 16) {
    for (col = 0; col + 16 <= cols; col += 16) {
      unsigned int sse;

      vpx_mse16x16(orig + col, orig_stride, recon + col, recon_stride, &sse);
      total_sse += sse;
    }

    /* Handle odd-sized width */
    if (col < cols) {
      unsigned int border_row, border_col;
      unsigned char *border_orig = orig;
      unsigned char *border_recon = recon;

      for (border_row = 0; border_row < 16; ++border_row) {
        for (border_col = col; border_col < cols; ++border_col) {
          diff = border_orig[border_col] - border_recon[border_col];
          total_sse += diff * diff;
        }

        border_orig += orig_stride;
        border_recon += recon_stride;
      }
    }

    orig += orig_stride * 16;
    recon += recon_stride * 16;
  }

  /* Handle odd-sized height */
  for (; row < rows; ++row) {
    for (col = 0; col < cols; ++col) {
      diff = orig[col] - recon[col];
      total_sse += diff * diff;
    }

    orig += orig_stride;
    recon += recon_stride;
  }

  vpx_clear_system_state();
  return total_sse;
}

static void generate_psnr_packet(VP8_COMP *cpi) {
  YV12_BUFFER_CONFIG *orig = cpi->Source;
  YV12_BUFFER_CONFIG *recon = cpi->common.frame_to_show;
  struct vpx_codec_cx_pkt pkt;
  uint64_t sse;
  int i;
  unsigned int width = cpi->common.Width;
  unsigned int height = cpi->common.Height;

  pkt.kind = VPX_CODEC_PSNR_PKT;
  sse = calc_plane_error(orig->y_buffer, orig->y_stride, recon->y_buffer,
                         recon->y_stride, width, height);
  pkt.data.psnr.sse[0] = sse;
  pkt.data.psnr.sse[1] = sse;
  pkt.data.psnr.samples[0] = width * height;
  pkt.data.psnr.samples[1] = width * height;

  width = (width + 1) / 2;
  height = (height + 1) / 2;

  sse = calc_plane_error(orig->u_buffer, orig->uv_stride, recon->u_buffer,
                         recon->uv_stride, width, height);
  pkt.data.psnr.sse[0] += sse;
  pkt.data.psnr.sse[2] = sse;
  pkt.data.psnr.samples[0] += width * height;
  pkt.data.psnr.samples[2] = width * height;

  sse = calc_plane_error(orig->v_buffer, orig->uv_stride, recon->v_buffer,
                         recon->uv_stride, width, height);
  pkt.data.psnr.sse[0] += sse;
  pkt.data.psnr.sse[3] = sse;
  pkt.data.psnr.samples[0] += width * height;
  pkt.data.psnr.samples[3] = width * height;

  for (i = 0; i < 4; ++i) {
    pkt.data.psnr.psnr[i] = vpx_sse_to_psnr(pkt.data.psnr.samples[i], 255.0,
                                            (double)(pkt.data.psnr.sse[i]));
  }

  vpx_codec_pkt_list_add(cpi->output_pkt_list, &pkt);
}

int vp8_use_as_reference(VP8_COMP *cpi, int ref_frame_flags) {
  if (ref_frame_flags > 7) return -1;

  cpi->ref_frame_flags = ref_frame_flags;
  return 0;
}
int vp8_update_reference(VP8_COMP *cpi, int ref_frame_flags) {
  if (ref_frame_flags > 7) return -1;

  cpi->common.refresh_golden_frame = 0;
  cpi->common.refresh_alt_ref_frame = 0;
  cpi->common.refresh_last_frame = 0;

  if (ref_frame_flags & VP8_LAST_FRAME) cpi->common.refresh_last_frame = 1;

  if (ref_frame_flags & VP8_GOLD_FRAME) cpi->common.refresh_golden_frame = 1;

  if (ref_frame_flags & VP8_ALTR_FRAME) cpi->common.refresh_alt_ref_frame = 1;

  cpi->ext_refresh_frame_flags_pending = 1;
  return 0;
}

int vp8_get_reference(VP8_COMP *cpi, enum vpx_ref_frame_type ref_frame_flag,
                      YV12_BUFFER_CONFIG *sd) {
  VP8_COMMON *cm = &cpi->common;
  int ref_fb_idx;

  if (ref_frame_flag == VP8_LAST_FRAME) {
    ref_fb_idx = cm->lst_fb_idx;
  } else if (ref_frame_flag == VP8_GOLD_FRAME) {
    ref_fb_idx = cm->gld_fb_idx;
  } else if (ref_frame_flag == VP8_ALTR_FRAME) {
    ref_fb_idx = cm->alt_fb_idx;
  } else {
    return -1;
  }

  vp8_yv12_copy_frame(&cm->yv12_fb[ref_fb_idx], sd);

  return 0;
}
int vp8_set_reference(VP8_COMP *cpi, enum vpx_ref_frame_type ref_frame_flag,
                      YV12_BUFFER_CONFIG *sd) {
  VP8_COMMON *cm = &cpi->common;

  int ref_fb_idx;

  if (ref_frame_flag == VP8_LAST_FRAME) {
    ref_fb_idx = cm->lst_fb_idx;
  } else if (ref_frame_flag == VP8_GOLD_FRAME) {
    ref_fb_idx = cm->gld_fb_idx;
  } else if (ref_frame_flag == VP8_ALTR_FRAME) {
    ref_fb_idx = cm->alt_fb_idx;
  } else {
    return -1;
  }

  vp8_yv12_copy_frame(sd, &cm->yv12_fb[ref_fb_idx]);

  return 0;
}
int vp8_update_entropy(VP8_COMP *cpi, int update) {
  VP8_COMMON *cm = &cpi->common;
  cm->refresh_entropy_probs = update;

  return 0;
}

static void scale_and_extend_source(YV12_BUFFER_CONFIG *sd, VP8_COMP *cpi) {
  VP8_COMMON *cm = &cpi->common;

  /* are we resizing the image */
  if (cm->horiz_scale != 0 || cm->vert_scale != 0) {
#if CONFIG_SPATIAL_RESAMPLING
    int hr, hs, vr, vs;
    int tmp_height;

    if (cm->vert_scale == 3) {
      tmp_height = 9;
    } else {
      tmp_height = 11;
    }

    Scale2Ratio(cm->horiz_scale, &hr, &hs);
    Scale2Ratio(cm->vert_scale, &vr, &vs);

    vpx_scale_frame(sd, &cpi->scaled_source, cm->temp_scale_frame.y_buffer,
                    tmp_height, hs, hr, vs, vr, 0);

    vp8_yv12_extend_frame_borders(&cpi->scaled_source);
    cpi->Source = &cpi->scaled_source;
#endif
  } else {
    cpi->Source = sd;
  }
}

static int resize_key_frame(VP8_COMP *cpi) {
#if CONFIG_SPATIAL_RESAMPLING
  VP8_COMMON *cm = &cpi->common;

  /* Do we need to apply resampling for one pass cbr.
   * In one pass this is more limited than in two pass cbr.
   * The test and any change is only made once per key frame sequence.
   */
  if (cpi->oxcf.allow_spatial_resampling &&
      (cpi->oxcf.end_usage == USAGE_STREAM_FROM_SERVER)) {
    int hr, hs, vr, vs;
    int new_width, new_height;

    /* If we are below the resample DOWN watermark then scale down a
     * notch.
     */
    if (cpi->buffer_level < (cpi->oxcf.resample_down_water_mark *
                             cpi->oxcf.optimal_buffer_level / 100)) {
      cm->horiz_scale =
          (cm->horiz_scale < VP8E_ONETWO) ? cm->horiz_scale + 1 : VP8E_ONETWO;
      cm->vert_scale =
          (cm->vert_scale < VP8E_ONETWO) ? cm->vert_scale + 1 : VP8E_ONETWO;
    }
    /* Should we now start scaling back up */
    else if (cpi->buffer_level > (cpi->oxcf.resample_up_water_mark *
                                  cpi->oxcf.optimal_buffer_level / 100)) {
      cm->horiz_scale =
          (cm->horiz_scale > VP8E_NORMAL) ? cm->horiz_scale - 1 : VP8E_NORMAL;
      cm->vert_scale =
          (cm->vert_scale > VP8E_NORMAL) ? cm->vert_scale - 1 : VP8E_NORMAL;
    }

    /* Get the new height and width */
    Scale2Ratio(cm->horiz_scale, &hr, &hs);
    Scale2Ratio(cm->vert_scale, &vr, &vs);
    new_width = ((hs - 1) + (cpi->oxcf.Width * hr)) / hs;
    new_height = ((vs - 1) + (cpi->oxcf.Height * vr)) / vs;

    /* If the image size has changed we need to reallocate the buffers
     * and resample the source image
     */
    if ((cm->Width != new_width) || (cm->Height != new_height)) {
      cm->Width = new_width;
      cm->Height = new_height;
      vp8_alloc_compressor_data(cpi);
      scale_and_extend_source(cpi->un_scaled_source, cpi);
      return 1;
    }
  }

#endif
  return 0;
}

static void update_alt_ref_frame_stats(VP8_COMP *cpi) {
  VP8_COMMON *cm = &cpi->common;

  /* Select an interval before next GF or altref */
  if (!cpi->auto_gold) cpi->frames_till_gf_update_due = DEFAULT_GF_INTERVAL;

  if ((cpi->pass != 2) && cpi->frames_till_gf_update_due) {
    cpi->current_gf_interval = cpi->frames_till_gf_update_due;

    /* Set the bits per frame that we should try and recover in
     * subsequent inter frames to account for the extra GF spend...
     * note that his does not apply for GF updates that occur
     * coincident with a key frame as the extra cost of key frames is
     * dealt with elsewhere.
     */
    cpi->gf_overspend_bits += cpi->projected_frame_size;
    cpi->non_gf_bitrate_adjustment =
        cpi->gf_overspend_bits / cpi->frames_till_gf_update_due;
  }

  /* Update data structure that monitors level of reference to last GF */
  memset(cpi->gf_active_flags, 1, (cm->mb_rows * cm->mb_cols));
  cpi->gf_active_count = cm->mb_rows * cm->mb_cols;

  /* this frame refreshes means next frames don't unless specified by user */
  cpi->frames_since_golden = 0;

  /* Clear the alternate reference update pending flag. */
  cpi->source_alt_ref_pending = 0;

  /* Set the alternate reference frame active flag */
  cpi->source_alt_ref_active = 1;
}
static void update_golden_frame_stats(VP8_COMP *cpi) {
  VP8_COMMON *cm = &cpi->common;

  /* Update the Golden frame usage counts. */
  if (cm->refresh_golden_frame) {
    /* Select an interval before next GF */
    if (!cpi->auto_gold) cpi->frames_till_gf_update_due = DEFAULT_GF_INTERVAL;

    if ((cpi->pass != 2) && (cpi->frames_till_gf_update_due > 0)) {
      cpi->current_gf_interval = cpi->frames_till_gf_update_due;

      /* Set the bits per frame that we should try and recover in
       * subsequent inter frames to account for the extra GF spend...
       * note that his does not apply for GF updates that occur
       * coincident with a key frame as the extra cost of key frames
       * is dealt with elsewhere.
       */
      if ((cm->frame_type != KEY_FRAME) && !cpi->source_alt_ref_active) {
        /* Calcluate GF bits to be recovered
         * Projected size - av frame bits available for inter
         * frames for clip as a whole
         */
        cpi->gf_overspend_bits +=
            (cpi->projected_frame_size - cpi->inter_frame_target);
      }

      cpi->non_gf_bitrate_adjustment =
          cpi->gf_overspend_bits / cpi->frames_till_gf_update_due;
    }

    /* Update data structure that monitors level of reference to last GF */
    memset(cpi->gf_active_flags, 1, (cm->mb_rows * cm->mb_cols));
    cpi->gf_active_count = cm->mb_rows * cm->mb_cols;

    /* this frame refreshes means next frames don't unless specified by
     * user
     */
    cm->refresh_golden_frame = 0;
    cpi->frames_since_golden = 0;

    cpi->recent_ref_frame_usage[INTRA_FRAME] = 1;
    cpi->recent_ref_frame_usage[LAST_FRAME] = 1;
    cpi->recent_ref_frame_usage[GOLDEN_FRAME] = 1;
    cpi->recent_ref_frame_usage[ALTREF_FRAME] = 1;

    /* ******** Fixed Q test code only ************ */
    /* If we are going to use the ALT reference for the next group of
     * frames set a flag to say so.
     */
    if (cpi->oxcf.fixed_q >= 0 && cpi->oxcf.play_alternate &&
        !cpi->common.refresh_alt_ref_frame) {
      cpi->source_alt_ref_pending = 1;
      cpi->frames_till_gf_update_due = cpi->baseline_gf_interval;
    }

    if (!cpi->source_alt_ref_pending) cpi->source_alt_ref_active = 0;

    /* Decrement count down till next gf */
    if (cpi->frames_till_gf_update_due > 0) cpi->frames_till_gf_update_due--;

  } else if (!cpi->common.refresh_alt_ref_frame) {
    /* Decrement count down till next gf */
    if (cpi->frames_till_gf_update_due > 0) cpi->frames_till_gf_update_due--;

    if (cpi->frames_till_alt_ref_frame) cpi->frames_till_alt_ref_frame--;

    cpi->frames_since_golden++;

    if (cpi->frames_since_golden > 1) {
      cpi->recent_ref_frame_usage[INTRA_FRAME] +=
          cpi->mb.count_mb_ref_frame_usage[INTRA_FRAME];
      cpi->recent_ref_frame_usage[LAST_FRAME] +=
          cpi->mb.count_mb_ref_frame_usage[LAST_FRAME];
      cpi->recent_ref_frame_usage[GOLDEN_FRAME] +=
          cpi->mb.count_mb_ref_frame_usage[GOLDEN_FRAME];
      cpi->recent_ref_frame_usage[ALTREF_FRAME] +=
          cpi->mb.count_mb_ref_frame_usage[ALTREF_FRAME];
    }
  }
}

/* This function updates the reference frame probability estimates that
 * will be used during mode selection
 */
static void update_rd_ref_frame_probs(VP8_COMP *cpi) {
  VP8_COMMON *cm = &cpi->common;

  const int *const rfct = cpi->mb.count_mb_ref_frame_usage;
  const int rf_intra = rfct[INTRA_FRAME];
  const int rf_inter =
      rfct[LAST_FRAME] + rfct[GOLDEN_FRAME] + rfct[ALTREF_FRAME];

  if (cm->frame_type == KEY_FRAME) {
    cpi->prob_intra_coded = 255;
    cpi->prob_last_coded = 128;
    cpi->prob_gf_coded = 128;
  } else if (!(rf_intra + rf_inter)) {
    cpi->prob_intra_coded = 63;
    cpi->prob_last_coded = 128;
    cpi->prob_gf_coded = 128;
  }

  /* update reference frame costs since we can do better than what we got
   * last frame.
   */
  if (cpi->oxcf.number_of_layers == 1) {
    if (cpi->common.refresh_alt_ref_frame) {
      cpi->prob_intra_coded += 40;
      if (cpi->prob_intra_coded > 255) cpi->prob_intra_coded = 255;
      cpi->prob_last_coded = 200;
      cpi->prob_gf_coded = 1;
    } else if (cpi->frames_since_golden == 0) {
      cpi->prob_last_coded = 214;
    } else if (cpi->frames_since_golden == 1) {
      cpi->prob_last_coded = 192;
      cpi->prob_gf_coded = 220;
    } else if (cpi->source_alt_ref_active) {
      cpi->prob_gf_coded -= 20;

      if (cpi->prob_gf_coded < 10) cpi->prob_gf_coded = 10;
    }
    if (!cpi->source_alt_ref_active) cpi->prob_gf_coded = 255;
  }
}

#if !CONFIG_REALTIME_ONLY
/* 1 = key, 0 = inter */
static int decide_key_frame(VP8_COMP *cpi) {
  VP8_COMMON *cm = &cpi->common;

  int code_key_frame = 0;

  cpi->kf_boost = 0;

  if (cpi->Speed > 11) return 0;

  /* Clear down mmx registers */
  vpx_clear_system_state();

  if ((cpi->compressor_speed == 2) && (cpi->Speed >= 5) && (cpi->sf.RD == 0)) {
    double change = 1.0 *
                    abs((int)(cpi->mb.intra_error - cpi->last_intra_error)) /
                    (1 + cpi->last_intra_error);
    double change2 =
        1.0 *
        abs((int)(cpi->mb.prediction_error - cpi->last_prediction_error)) /
        (1 + cpi->last_prediction_error);
    double minerror = cm->MBs * 256;

    cpi->last_intra_error = cpi->mb.intra_error;
    cpi->last_prediction_error = cpi->mb.prediction_error;

    if (10 * cpi->mb.intra_error / (1 + cpi->mb.prediction_error) < 15 &&
        cpi->mb.prediction_error > minerror &&
        (change > .25 || change2 > .25)) {
      /*(change > 1.4 || change < .75)&& cpi->this_frame_percent_intra >
       * cpi->last_frame_percent_intra + 3*/
      return 1;
    }

    return 0;
  }

  /* If the following are true we might as well code a key frame */
  if (((cpi->this_frame_percent_intra == 100) &&
       (cpi->this_frame_percent_intra > (cpi->last_frame_percent_intra + 2))) ||
      ((cpi->this_frame_percent_intra > 95) &&
       (cpi->this_frame_percent_intra >=
        (cpi->last_frame_percent_intra + 5)))) {
    code_key_frame = 1;
  }
  /* in addition if the following are true and this is not a golden frame
   * then code a key frame Note that on golden frames there often seems
   * to be a pop in intra usage anyway hence this restriction is
   * designed to prevent spurious key frames. The Intra pop needs to be
   * investigated.
   */
  else if (((cpi->this_frame_percent_intra > 60) &&
            (cpi->this_frame_percent_intra >
             (cpi->last_frame_percent_intra * 2))) ||
           ((cpi->this_frame_percent_intra > 75) &&
            (cpi->this_frame_percent_intra >
             (cpi->last_frame_percent_intra * 3 / 2))) ||
           ((cpi->this_frame_percent_intra > 90) &&
            (cpi->this_frame_percent_intra >
             (cpi->last_frame_percent_intra + 10)))) {
    if (!cm->refresh_golden_frame) code_key_frame = 1;
  }

  return code_key_frame;
}

static void Pass1Encode(VP8_COMP *cpi) {
  vp8_set_quantizer(cpi, 26);
  vp8_first_pass(cpi);
}
#endif

#if 0
void write_cx_frame_to_file(YV12_BUFFER_CONFIG *frame, int this_frame)
{

    /* write the frame */
    FILE *yframe;
    int i;
    char filename[255];

    sprintf(filename, "cx\\y%04d.raw", this_frame);
    yframe = fopen(filename, "wb");

    for (i = 0; i < frame->y_height; ++i)
        fwrite(frame->y_buffer + i * frame->y_stride, frame->y_width, 1, yframe);

    fclose(yframe);
    sprintf(filename, "cx\\u%04d.raw", this_frame);
    yframe = fopen(filename, "wb");

    for (i = 0; i < frame->uv_height; ++i)
        fwrite(frame->u_buffer + i * frame->uv_stride, frame->uv_width, 1, yframe);

    fclose(yframe);
    sprintf(filename, "cx\\v%04d.raw", this_frame);
    yframe = fopen(filename, "wb");

    for (i = 0; i < frame->uv_height; ++i)
        fwrite(frame->v_buffer + i * frame->uv_stride, frame->uv_width, 1, yframe);

    fclose(yframe);
}
#endif

#if !CONFIG_REALTIME_ONLY
/* Function to test for conditions that indeicate we should loop
 * back and recode a frame.
 */
static int recode_loop_test(VP8_COMP *cpi, int high_limit, int low_limit, int q,
                            int maxq, int minq) {
  int force_recode = 0;
  VP8_COMMON *cm = &cpi->common;

  /* Is frame recode allowed at all
   * Yes if either recode mode 1 is selected or mode two is selcted
   * and the frame is a key frame. golden frame or alt_ref_frame
   */
  if ((cpi->sf.recode_loop == 1) ||
      ((cpi->sf.recode_loop == 2) &&
       ((cm->frame_type == KEY_FRAME) || cm->refresh_golden_frame ||
        cm->refresh_alt_ref_frame))) {
    /* General over and under shoot tests */
    if (((cpi->projected_frame_size > high_limit) && (q < maxq)) ||
        ((cpi->projected_frame_size < low_limit) && (q > minq))) {
      force_recode = 1;
    }
    /* Special Constrained quality tests */
    else if (cpi->oxcf.end_usage == USAGE_CONSTRAINED_QUALITY) {
      /* Undershoot and below auto cq level */
      if ((q > cpi->cq_target_quality) &&
          (cpi->projected_frame_size < ((cpi->this_frame_target * 7) >> 3))) {
        force_recode = 1;
      }
      /* Severe undershoot and between auto and user cq level */
      else if ((q > cpi->oxcf.cq_level) &&
               (cpi->projected_frame_size < cpi->min_frame_bandwidth) &&
               (cpi->active_best_quality > cpi->oxcf.cq_level)) {
        force_recode = 1;
        cpi->active_best_quality = cpi->oxcf.cq_level;
      }
    }
  }

  return force_recode;
}
#endif  // !CONFIG_REALTIME_ONLY

static void update_reference_frames(VP8_COMP *cpi) {
  VP8_COMMON *cm = &cpi->common;
  YV12_BUFFER_CONFIG *yv12_fb = cm->yv12_fb;

  /* At this point the new frame has been encoded.
   * If any buffer copy / swapping is signaled it should be done here.
   */

  if (cm->frame_type == KEY_FRAME) {
    yv12_fb[cm->new_fb_idx].flags |= VP8_GOLD_FRAME | VP8_ALTR_FRAME;

    yv12_fb[cm->gld_fb_idx].flags &= ~VP8_GOLD_FRAME;
    yv12_fb[cm->alt_fb_idx].flags &= ~VP8_ALTR_FRAME;

    cm->alt_fb_idx = cm->gld_fb_idx = cm->new_fb_idx;

    cpi->current_ref_frames[GOLDEN_FRAME] = cm->current_video_frame;
    cpi->current_ref_frames[ALTREF_FRAME] = cm->current_video_frame;
  } else {
    if (cm->refresh_alt_ref_frame) {
      assert(!cm->copy_buffer_to_arf);

      cm->yv12_fb[cm->new_fb_idx].flags |= VP8_ALTR_FRAME;
      cm->yv12_fb[cm->alt_fb_idx].flags &= ~VP8_ALTR_FRAME;
      cm->alt_fb_idx = cm->new_fb_idx;

      cpi->current_ref_frames[ALTREF_FRAME] = cm->current_video_frame;
    } else if (cm->copy_buffer_to_arf) {
      assert(!(cm->copy_buffer_to_arf & ~0x3));

      if (cm->copy_buffer_to_arf == 1) {
        if (cm->alt_fb_idx != cm->lst_fb_idx) {
          yv12_fb[cm->lst_fb_idx].flags |= VP8_ALTR_FRAME;
          yv12_fb[cm->alt_fb_idx].flags &= ~VP8_ALTR_FRAME;
          cm->alt_fb_idx = cm->lst_fb_idx;

          cpi->current_ref_frames[ALTREF_FRAME] =
              cpi->current_ref_frames[LAST_FRAME];
        }
      } else {
        if (cm->alt_fb_idx != cm->gld_fb_idx) {
          yv12_fb[cm->gld_fb_idx].flags |= VP8_ALTR_FRAME;
          yv12_fb[cm->alt_fb_idx].flags &= ~VP8_ALTR_FRAME;
          cm->alt_fb_idx = cm->gld_fb_idx;

          cpi->current_ref_frames[ALTREF_FRAME] =
              cpi->current_ref_frames[GOLDEN_FRAME];
        }
      }
    }

    if (cm->refresh_golden_frame) {
      assert(!cm->copy_buffer_to_gf);

      cm->yv12_fb[cm->new_fb_idx].flags |= VP8_GOLD_FRAME;
      cm->yv12_fb[cm->gld_fb_idx].flags &= ~VP8_GOLD_FRAME;
      cm->gld_fb_idx = cm->new_fb_idx;

      cpi->current_ref_frames[GOLDEN_FRAME] = cm->current_video_frame;
    } else if (cm->copy_buffer_to_gf) {
      assert(!(cm->copy_buffer_to_arf & ~0x3));

      if (cm->copy_buffer_to_gf == 1) {
        if (cm->gld_fb_idx != cm->lst_fb_idx) {
          yv12_fb[cm->lst_fb_idx].flags |= VP8_GOLD_FRAME;
          yv12_fb[cm->gld_fb_idx].flags &= ~VP8_GOLD_FRAME;
          cm->gld_fb_idx = cm->lst_fb_idx;

          cpi->current_ref_frames[GOLDEN_FRAME] =
              cpi->current_ref_frames[LAST_FRAME];
        }
      } else {
        if (cm->alt_fb_idx != cm->gld_fb_idx) {
          yv12_fb[cm->alt_fb_idx].flags |= VP8_GOLD_FRAME;
          yv12_fb[cm->gld_fb_idx].flags &= ~VP8_GOLD_FRAME;
          cm->gld_fb_idx = cm->alt_fb_idx;

          cpi->current_ref_frames[GOLDEN_FRAME] =
              cpi->current_ref_frames[ALTREF_FRAME];
        }
      }
    }
  }

  if (cm->refresh_last_frame) {
    cm->yv12_fb[cm->new_fb_idx].flags |= VP8_LAST_FRAME;
    cm->yv12_fb[cm->lst_fb_idx].flags &= ~VP8_LAST_FRAME;
    cm->lst_fb_idx = cm->new_fb_idx;

    cpi->current_ref_frames[LAST_FRAME] = cm->current_video_frame;
  }

#if CONFIG_TEMPORAL_DENOISING
  if (cpi->oxcf.noise_sensitivity) {
    /* we shouldn't have to keep multiple copies as we know in advance which
     * buffer we should start - for now to get something up and running
     * I've chosen to copy the buffers
     */
    if (cm->frame_type == KEY_FRAME) {
      int i;
      for (i = LAST_FRAME; i < MAX_REF_FRAMES; ++i)
        vp8_yv12_copy_frame(cpi->Source, &cpi->denoiser.yv12_running_avg[i]);
    } else {
      vp8_yv12_extend_frame_borders(
          &cpi->denoiser.yv12_running_avg[INTRA_FRAME]);

      if (cm->refresh_alt_ref_frame || cm->copy_buffer_to_arf) {
        vp8_yv12_copy_frame(&cpi->denoiser.yv12_running_avg[INTRA_FRAME],
                            &cpi->denoiser.yv12_running_avg[ALTREF_FRAME]);
      }
      if (cm->refresh_golden_frame || cm->copy_buffer_to_gf) {
        vp8_yv12_copy_frame(&cpi->denoiser.yv12_running_avg[INTRA_FRAME],
                            &cpi->denoiser.yv12_running_avg[GOLDEN_FRAME]);
      }
      if (cm->refresh_last_frame) {
        vp8_yv12_copy_frame(&cpi->denoiser.yv12_running_avg[INTRA_FRAME],
                            &cpi->denoiser.yv12_running_avg[LAST_FRAME]);
      }
    }
    if (cpi->oxcf.noise_sensitivity == 4)
      vp8_yv12_copy_frame(cpi->Source, &cpi->denoiser.yv12_last_source);
  }
#endif
}

static int measure_square_diff_partial(YV12_BUFFER_CONFIG *source,
                                       YV12_BUFFER_CONFIG *dest,
                                       VP8_COMP *cpi) {
  int i, j;
  int Total = 0;
  int num_blocks = 0;
  int skip = 2;
  int min_consec_zero_last = 10;
  int tot_num_blocks = (source->y_height * source->y_width) >> 8;
  unsigned char *src = source->y_buffer;
  unsigned char *dst = dest->y_buffer;

  /* Loop through the Y plane, every |skip| blocks along rows and colmumns,
   * summing the square differences, and only for blocks that have been
   * zero_last mode at least |x| frames in a row.
   */
  for (i = 0; i < source->y_height; i += 16 * skip) {
    int block_index_row = (i >> 4) * cpi->common.mb_cols;
    for (j = 0; j < source->y_width; j += 16 * skip) {
      int index = block_index_row + (j >> 4);
      if (cpi->consec_zero_last[index] >= min_consec_zero_last) {
        unsigned int sse;
        Total += vpx_mse16x16(src + j, source->y_stride, dst + j,
                              dest->y_stride, &sse);
        num_blocks++;
      }
    }
    src += 16 * skip * source->y_stride;
    dst += 16 * skip * dest->y_stride;
  }
  // Only return non-zero if we have at least ~1/16 samples for estimate.
  if (num_blocks > (tot_num_blocks >> 4)) {
    assert(num_blocks != 0);
    return (Total / num_blocks);
  } else {
    return 0;
  }
}

#if CONFIG_TEMPORAL_DENOISING
static void process_denoiser_mode_change(VP8_COMP *cpi) {
  const VP8_COMMON *const cm = &cpi->common;
  int i, j;
  int total = 0;
  int num_blocks = 0;
  // Number of blocks skipped along row/column in computing the
  // nmse (normalized mean square error) of source.
  int skip = 2;
  // Only select blocks for computing nmse that have been encoded
  // as ZERO LAST min_consec_zero_last frames in a row.
  // Scale with number of temporal layers.
  int min_consec_zero_last = 12 / cpi->oxcf.number_of_layers;
  // Decision is tested for changing the denoising mode every
  // num_mode_change times this function is called. Note that this
  // function called every 8 frames, so (8 * num_mode_change) is number
  // of frames where denoising mode change is tested for switch.
  int num_mode_change = 20;
  // Framerate factor, to compensate for larger mse at lower framerates.
  // Use ref_framerate, which is full source framerate for temporal layers.
  // TODO(marpan): Adjust this factor.
  int fac_framerate = cpi->ref_framerate < 25.0f ? 80 : 100;
  int tot_num_blocks = cm->mb_rows * cm->mb_cols;
  int ystride = cpi->Source->y_stride;
  unsigned char *src = cpi->Source->y_buffer;
  unsigned char *dst = cpi->denoiser.yv12_last_source.y_buffer;
  static const unsigned char const_source[16] = { 128, 128, 128, 128, 128, 128,
                                                  128, 128, 128, 128, 128, 128,
                                                  128, 128, 128, 128 };
  int bandwidth = (int)(cpi->target_bandwidth);
  // For temporal layers, use full bandwidth (top layer).
  if (cpi->oxcf.number_of_layers > 1) {
    LAYER_CONTEXT *lc = &cpi->layer_context[cpi->oxcf.number_of_layers - 1];
    bandwidth = (int)(lc->target_bandwidth);
  }
  // Loop through the Y plane, every skip blocks along rows and columns,
  // summing the normalized mean square error, only for blocks that have
  // been encoded as ZEROMV LAST at least min_consec_zero_last least frames in
  // a row and have small sum difference between current and previous frame.
  // Normalization here is by the contrast of the current frame block.
  for (i = 0; i < cm->Height; i += 16 * skip) {
    int block_index_row = (i >> 4) * cm->mb_cols;
    for (j = 0; j < cm->Width; j += 16 * skip) {
      int index = block_index_row + (j >> 4);
      if (cpi->consec_zero_last[index] >= min_consec_zero_last) {
        unsigned int sse;
        const unsigned int var =
            vpx_variance16x16(src + j, ystride, dst + j, ystride, &sse);
        // Only consider this block as valid for noise measurement
        // if the sum_diff average of the current and previous frame
        // is small (to avoid effects from lighting change).
        if ((sse - var) < 128) {
          unsigned int sse2;
          const unsigned int act =
              vpx_variance16x16(src + j, ystride, const_source, 0, &sse2);
          if (act > 0) total += sse / act;
          num_blocks++;
        }
      }
    }
    src += 16 * skip * ystride;
    dst += 16 * skip * ystride;
  }
  total = total * fac_framerate / 100;

  // Only consider this frame as valid sample if we have computed nmse over
  // at least ~1/16 blocks, and Total > 0 (Total == 0 can happen if the
  // application inputs duplicate frames, or contrast is all zero).
  if (total > 0 && (num_blocks > (tot_num_blocks >> 4))) {
    // Update the recursive mean square source_diff.
    total = (total << 8) / num_blocks;
    if (cpi->denoiser.nmse_source_diff_count == 0) {
      // First sample in new interval.
      cpi->denoiser.nmse_source_diff = total;
      cpi->denoiser.qp_avg = cm->base_qindex;
    } else {
      // For subsequent samples, use average with weight ~1/4 for new sample.
      cpi->denoiser.nmse_source_diff =
          (int)((total + 3 * cpi->denoiser.nmse_source_diff) >> 2);
      cpi->denoiser.qp_avg =
          (int)((cm->base_qindex + 3 * cpi->denoiser.qp_avg) >> 2);
    }
    cpi->denoiser.nmse_source_diff_count++;
  }
  // Check for changing the denoiser mode, when we have obtained #samples =
  // num_mode_change. Condition the change also on the bitrate and QP.
  if (cpi->denoiser.nmse_source_diff_count == num_mode_change) {
    // Check for going up: from normal to aggressive mode.
    if ((cpi->denoiser.denoiser_mode == kDenoiserOnYUV) &&
        (cpi->denoiser.nmse_source_diff >
         cpi->denoiser.threshold_aggressive_mode) &&
        (cpi->denoiser.qp_avg < cpi->denoiser.qp_threshold_up &&
         bandwidth > cpi->denoiser.bitrate_threshold)) {
      vp8_denoiser_set_parameters(&cpi->denoiser, kDenoiserOnYUVAggressive);
    } else {
      // Check for going down: from aggressive to normal mode.
      if (((cpi->denoiser.denoiser_mode == kDenoiserOnYUVAggressive) &&
           (cpi->denoiser.nmse_source_diff <
            cpi->denoiser.threshold_aggressive_mode)) ||
          ((cpi->denoiser.denoiser_mode == kDenoiserOnYUVAggressive) &&
           (cpi->denoiser.qp_avg > cpi->denoiser.qp_threshold_down ||
            bandwidth < cpi->denoiser.bitrate_threshold))) {
        vp8_denoiser_set_parameters(&cpi->denoiser, kDenoiserOnYUV);
      }
    }
    // Reset metric and counter for next interval.
    cpi->denoiser.nmse_source_diff = 0;
    cpi->denoiser.qp_avg = 0;
    cpi->denoiser.nmse_source_diff_count = 0;
  }
}
#endif

void vp8_loopfilter_frame(VP8_COMP *cpi, VP8_COMMON *cm) {
  const FRAME_TYPE frame_type = cm->frame_type;

  int update_any_ref_buffers = 1;
  if (cpi->common.refresh_last_frame == 0 &&
      cpi->common.refresh_golden_frame == 0 &&
      cpi->common.refresh_alt_ref_frame == 0) {
    update_any_ref_buffers = 0;
  }

  if (cm->no_lpf) {
    cm->filter_level = 0;
  } else {
#if CONFIG_INTERNAL_STATS
    struct vpx_usec_timer timer;
#endif

    vpx_clear_system_state();

#if CONFIG_INTERNAL_STATS
    vpx_usec_timer_start(&timer);
#endif
    if (cpi->sf.auto_filter == 0) {
#if CONFIG_TEMPORAL_DENOISING
      if (cpi->oxcf.noise_sensitivity && cm->frame_type != KEY_FRAME) {
        // Use the denoised buffer for selecting base loop filter level.
        // Denoised signal for current frame is stored in INTRA_FRAME.
        // No denoising on key frames.
        vp8cx_pick_filter_level_fast(
            &cpi->denoiser.yv12_running_avg[INTRA_FRAME], cpi);
      } else {
        vp8cx_pick_filter_level_fast(cpi->Source, cpi);
      }
#else
      vp8cx_pick_filter_level_fast(cpi->Source, cpi);
#endif
    } else {
#if CONFIG_TEMPORAL_DENOISING
      if (cpi->oxcf.noise_sensitivity && cm->frame_type != KEY_FRAME) {
        // Use the denoised buffer for selecting base loop filter level.
        // Denoised signal for current frame is stored in INTRA_FRAME.
        // No denoising on key frames.
        vp8cx_pick_filter_level(&cpi->denoiser.yv12_running_avg[INTRA_FRAME],
                                cpi);
      } else {
        vp8cx_pick_filter_level(cpi->Source, cpi);
      }
#else
      vp8cx_pick_filter_level(cpi->Source, cpi);
#endif
    }

    if (cm->filter_level > 0) {
      vp8cx_set_alt_lf_level(cpi, cm->filter_level);
    }

#if CONFIG_INTERNAL_STATS
    vpx_usec_timer_mark(&timer);
    cpi->time_pick_lpf += vpx_usec_timer_elapsed(&timer);
#endif
  }

#if CONFIG_MULTITHREAD
  if (vpx_atomic_load_acquire(&cpi->b_multi_threaded)) {
    /* signal that we have set filter_level */
    vp8_sem_post(&cpi->h_event_end_lpf);
  }
#endif

  // No need to apply loop-filter if the encoded frame does not update
  // any reference buffers.
  if (cm->filter_level > 0 && update_any_ref_buffers) {
    vp8_loop_filter_frame(cm, &cpi->mb.e_mbd, frame_type);
  }

  vp8_yv12_extend_frame_borders(cm->frame_to_show);
}
// Return 1 if frame is to be dropped. Update frame drop decimation
// counters.
int vp8_check_drop_buffer(VP8_COMP *cpi) {
  VP8_COMMON *cm = &cpi->common;
  int drop_mark = (int)(cpi->oxcf.drop_frames_water_mark *
                        cpi->oxcf.optimal_buffer_level / 100);
  int drop_mark75 = drop_mark * 2 / 3;
  int drop_mark50 = drop_mark / 4;
  int drop_mark25 = drop_mark / 8;
  if (cpi->drop_frames_allowed) {
    /* The reset to decimation 0 is only done here for one pass.
     * Once it is set two pass leaves decimation on till the next kf.
     */
    if (cpi->buffer_level > drop_mark && cpi->decimation_factor > 0) {
      cpi->decimation_factor--;
    }

    if (cpi->buffer_level > drop_mark75 && cpi->decimation_factor > 0) {
      cpi->decimation_factor = 1;

    } else if (cpi->buffer_level < drop_mark25 &&
               (cpi->decimation_factor == 2 || cpi->decimation_factor == 3)) {
      cpi->decimation_factor = 3;
    } else if (cpi->buffer_level < drop_mark50 &&
               (cpi->decimation_factor == 1 || cpi->decimation_factor == 2)) {
      cpi->decimation_factor = 2;
    } else if (cpi->buffer_level < drop_mark75 &&
               (cpi->decimation_factor == 0 || cpi->decimation_factor == 1)) {
      cpi->decimation_factor = 1;
    }
  }

  /* The following decimates the frame rate according to a regular
   * pattern (i.e. to 1/2 or 2/3 frame rate) This can be used to help
   * prevent buffer under-run in CBR mode. Alternatively it might be
   * desirable in some situations to drop frame rate but throw more bits
   * at each frame.
   *
   * Note that dropping a key frame can be problematic if spatial
   * resampling is also active
   */
  if (cpi->decimation_factor > 0 && cpi->drop_frames_allowed) {
    switch (cpi->decimation_factor) {
      case 1:
        cpi->per_frame_bandwidth = cpi->per_frame_bandwidth * 3 / 2;
        break;
      case 2:
        cpi->per_frame_bandwidth = cpi->per_frame_bandwidth * 5 / 4;
        break;
      case 3:
        cpi->per_frame_bandwidth = cpi->per_frame_bandwidth * 5 / 4;
        break;
    }

    /* Note that we should not throw out a key frame (especially when
     * spatial resampling is enabled).
     */
    if (cm->frame_type == KEY_FRAME) {
      cpi->decimation_count = cpi->decimation_factor;
    } else if (cpi->decimation_count > 0) {
      cpi->decimation_count--;

      cpi->bits_off_target += cpi->av_per_frame_bandwidth;
      if (cpi->bits_off_target > cpi->oxcf.maximum_buffer_size) {
        cpi->bits_off_target = cpi->oxcf.maximum_buffer_size;
      }

#if CONFIG_MULTI_RES_ENCODING
      vp8_store_drop_frame_info(cpi);
#endif

      cm->current_video_frame++;
      cpi->frames_since_key++;
      cpi->ext_refresh_frame_flags_pending = 0;
      // We advance the temporal pattern for dropped frames.
      cpi->temporal_pattern_counter++;

#if CONFIG_INTERNAL_STATS
      cpi->count++;
#endif

      cpi->buffer_level = cpi->bits_off_target;

      if (cpi->oxcf.number_of_layers > 1) {
        unsigned int i;

        /* Propagate bits saved by dropping the frame to higher
         * layers
         */
        for (i = cpi->current_layer + 1; i < cpi->oxcf.number_of_layers; ++i) {
          LAYER_CONTEXT *lc = &cpi->layer_context[i];
          lc->bits_off_target += (int)(lc->target_bandwidth / lc->framerate);
          if (lc->bits_off_target > lc->maximum_buffer_size) {
            lc->bits_off_target = lc->maximum_buffer_size;
          }
          lc->buffer_level = lc->bits_off_target;
        }
      }
      return 1;
    } else {
      cpi->decimation_count = cpi->decimation_factor;
    }
  } else {
    cpi->decimation_count = 0;
  }
  return 0;
}

static void encode_frame_to_data_rate(VP8_COMP *cpi, size_t *size,
                                      unsigned char *dest,
                                      unsigned char *dest_end,
                                      unsigned int *frame_flags) {
  int Q;
  int frame_over_shoot_limit;
  int frame_under_shoot_limit;

  int Loop = 0;

  VP8_COMMON *cm = &cpi->common;
  int active_worst_qchanged = 0;

#if !CONFIG_REALTIME_ONLY
  int q_low;
  int q_high;
  int zbin_oq_high;
  int zbin_oq_low = 0;
  int top_index;
  int bottom_index;
  int overshoot_seen = 0;
  int undershoot_seen = 0;
#endif

  /* Clear down mmx registers to allow floating point in what follows */
  vpx_clear_system_state();

  if (cpi->force_next_frame_intra) {
    cm->frame_type = KEY_FRAME; /* delayed intra frame */
    cpi->force_next_frame_intra = 0;
  }

  /* For an alt ref frame in 2 pass we skip the call to the second pass
   * function that sets the target bandwidth
   */
  switch (cpi->pass) {
#if !CONFIG_REALTIME_ONLY
    case 2:
      if (cpi->common.refresh_alt_ref_frame) {
        /* Per frame bit target for the alt ref frame */
        cpi->per_frame_bandwidth = cpi->twopass.gf_bits;
        /* per second target bitrate */
        cpi->target_bandwidth =
            (int)(cpi->twopass.gf_bits * cpi->output_framerate);
      }
      break;
#endif  // !CONFIG_REALTIME_ONLY
    default: {
      const double per_frame_bandwidth =
          round(cpi->target_bandwidth / cpi->output_framerate);
      cpi->per_frame_bandwidth = (int)VPXMIN(per_frame_bandwidth, INT_MAX);
      break;
    }
  }

  /* Default turn off buffer to buffer copying */
  cm->copy_buffer_to_gf = 0;
  cm->copy_buffer_to_arf = 0;

  /* Clear zbin over-quant value and mode boost values. */
  cpi->mb.zbin_over_quant = 0;
  cpi->mb.zbin_mode_boost = 0;

  /* Enable or disable mode based tweaking of the zbin
   * For 2 Pass Only used where GF/ARF prediction quality
   * is above a threshold
   */
  cpi->mb.zbin_mode_boost_enabled = 1;
  if (cpi->pass == 2) {
    if (cpi->gfu_boost <= 400) {
      cpi->mb.zbin_mode_boost_enabled = 0;
    }
  }

  /* Current default encoder behaviour for the altref sign bias */
  if (cpi->source_alt_ref_active) {
    cpi->common.ref_frame_sign_bias[ALTREF_FRAME] = 1;
  } else {
    cpi->common.ref_frame_sign_bias[ALTREF_FRAME] = 0;
  }

  /* Check to see if a key frame is signaled
   * For two pass with auto key frame enabled cm->frame_type may already
   * be set, but not for one pass.
   */
  if ((cm->current_video_frame == 0) || (cm->frame_flags & FRAMEFLAGS_KEY) ||
      (cpi->oxcf.auto_key &&
       (cpi->frames_since_key % cpi->key_frame_frequency == 0))) {
    /* Key frame from VFW/auto-keyframe/first frame */
    cm->frame_type = KEY_FRAME;
#if CONFIG_TEMPORAL_DENOISING
    if (cpi->oxcf.noise_sensitivity == 4) {
      // For adaptive mode, reset denoiser to normal mode on key frame.
      vp8_denoiser_set_parameters(&cpi->denoiser, kDenoiserOnYUV);
    }
#endif
  }

#if CONFIG_MULTI_RES_ENCODING
  if (cpi->oxcf.mr_total_resolutions > 1) {
    LOWER_RES_FRAME_INFO *low_res_frame_info =
        (LOWER_RES_FRAME_INFO *)cpi->oxcf.mr_low_res_mode_info;

    if (cpi->oxcf.mr_encoder_id) {
      // Check if lower resolution is available for motion vector reuse.
      if (cm->frame_type != KEY_FRAME) {
        cpi->mr_low_res_mv_avail = 1;
        cpi->mr_low_res_mv_avail &= !(low_res_frame_info->is_frame_dropped);

        if (cpi->ref_frame_flags & VP8_LAST_FRAME)
          cpi->mr_low_res_mv_avail &=
              (cpi->current_ref_frames[LAST_FRAME] ==
               low_res_frame_info->low_res_ref_frames[LAST_FRAME]);

        if (cpi->ref_frame_flags & VP8_GOLD_FRAME)
          cpi->mr_low_res_mv_avail &=
              (cpi->current_ref_frames[GOLDEN_FRAME] ==
               low_res_frame_info->low_res_ref_frames[GOLDEN_FRAME]);

        // Don't use altref to determine whether low res is available.
        // TODO (marpan): Should we make this type of condition on a
        // per-reference frame basis?
        /*
        if (cpi->ref_frame_flags & VP8_ALTR_FRAME)
            cpi->mr_low_res_mv_avail &= (cpi->current_ref_frames[ALTREF_FRAME]
                     == low_res_frame_info->low_res_ref_frames[ALTREF_FRAME]);
        */
      }
      // Disable motion vector reuse (i.e., disable any usage of the low_res)
      // if the previous lower stream is skipped/disabled.
      if (low_res_frame_info->skip_encoding_prev_stream) {
        cpi->mr_low_res_mv_avail = 0;
      }
    }
    // This stream is not skipped (i.e., it's being encoded), so set this skip
    // flag to 0. This is needed for the next stream (i.e., which is the next
    // frame to be encoded).
    low_res_frame_info->skip_encoding_prev_stream = 0;

    // On a key frame: For the lowest resolution, keep track of the key frame
    // counter value. For the higher resolutions, reset the current video
    // frame counter to that of the lowest resolution.
    // This is done to the handle the case where we may stop/start encoding
    // higher layer(s). The restart-encoding of higher layer is only signaled
    // by a key frame for now.
    // TODO (marpan): Add flag to indicate restart-encoding of higher layer.
    if (cm->frame_type == KEY_FRAME) {
      if (cpi->oxcf.mr_encoder_id) {
        // If the initial starting value of the buffer level is zero (this can
        // happen because we may have not started encoding this higher stream),
        // then reset it to non-zero value based on |starting_buffer_level|.
        if (cpi->common.current_video_frame == 0 && cpi->buffer_level == 0) {
          unsigned int i;
          cpi->bits_off_target = cpi->oxcf.starting_buffer_level;
          cpi->buffer_level = cpi->oxcf.starting_buffer_level;
          for (i = 0; i < cpi->oxcf.number_of_layers; ++i) {
            LAYER_CONTEXT *lc = &cpi->layer_context[i];
            lc->bits_off_target = lc->starting_buffer_level;
            lc->buffer_level = lc->starting_buffer_level;
          }
        }
        cpi->common.current_video_frame =
            low_res_frame_info->key_frame_counter_value;
      } else {
        low_res_frame_info->key_frame_counter_value =
            cpi->common.current_video_frame;
      }
    }
  }
#endif

  // Find the reference frame closest to the current frame.
  cpi->closest_reference_frame = LAST_FRAME;
  if (cm->frame_type != KEY_FRAME) {
    int i;
    MV_REFERENCE_FRAME closest_ref = INTRA_FRAME;
    if (cpi->ref_frame_flags & VP8_LAST_FRAME) {
      closest_ref = LAST_FRAME;
    } else if (cpi->ref_frame_flags & VP8_GOLD_FRAME) {
      closest_ref = GOLDEN_FRAME;
    } else if (cpi->ref_frame_flags & VP8_ALTR_FRAME) {
      closest_ref = ALTREF_FRAME;
    }
    for (i = 1; i <= 3; ++i) {
      vpx_ref_frame_type_t ref_frame_type =
          (vpx_ref_frame_type_t)((i == 3) ? 4 : i);
      if (cpi->ref_frame_flags & ref_frame_type) {
        if ((cm->current_video_frame - cpi->current_ref_frames[i]) <
            (cm->current_video_frame - cpi->current_ref_frames[closest_ref])) {
          closest_ref = i;
        }
      }
    }
    cpi->closest_reference_frame = closest_ref;
  }

  /* Set various flags etc to special state if it is a key frame */
  if (cm->frame_type == KEY_FRAME) {
    int i;

    // Set the loop filter deltas and segmentation map update
    setup_features(cpi);

    /* The alternate reference frame cannot be active for a key frame */
    cpi->source_alt_ref_active = 0;

    /* Reset the RD threshold multipliers to default of * 1 (128) */
    for (i = 0; i < MAX_MODES; ++i) {
      cpi->mb.rd_thresh_mult[i] = 128;
    }

    // Reset the zero_last counter to 0 on key frame.
    memset(cpi->consec_zero_last, 0, cm->mb_rows * cm->mb_cols);
    memset(cpi->consec_zero_last_mvbias, 0,
           (cpi->common.mb_rows * cpi->common.mb_cols));
  }

#if 0
    /* Experimental code for lagged compress and one pass
     * Initialise one_pass GF frames stats
     * Update stats used for GF selection
     */
    {
        cpi->one_pass_frame_index = cm->current_video_frame % MAX_LAG_BUFFERS;

        cpi->one_pass_frame_stats[cpi->one_pass_frame_index ].frames_so_far = 0;
        cpi->one_pass_frame_stats[cpi->one_pass_frame_index ].frame_intra_error = 0.0;
        cpi->one_pass_frame_stats[cpi->one_pass_frame_index ].frame_coded_error = 0.0;
        cpi->one_pass_frame_stats[cpi->one_pass_frame_index ].frame_pcnt_inter = 0.0;
        cpi->one_pass_frame_stats[cpi->one_pass_frame_index ].frame_pcnt_motion = 0.0;
        cpi->one_pass_frame_stats[cpi->one_pass_frame_index ].frame_mvr = 0.0;
        cpi->one_pass_frame_stats[cpi->one_pass_frame_index ].frame_mvr_abs = 0.0;
        cpi->one_pass_frame_stats[cpi->one_pass_frame_index ].frame_mvc = 0.0;
        cpi->one_pass_frame_stats[cpi->one_pass_frame_index ].frame_mvc_abs = 0.0;
    }
#endif

  update_rd_ref_frame_probs(cpi);

  if (vp8_check_drop_buffer(cpi)) {
    return;
  }

  /* Decide how big to make the frame */
  if (!vp8_pick_frame_size(cpi)) {
/*TODO: 2 drop_frame and return code could be put together. */
#if CONFIG_MULTI_RES_ENCODING
    vp8_store_drop_frame_info(cpi);
#endif
    cm->current_video_frame++;
    cpi->frames_since_key++;
    cpi->ext_refresh_frame_flags_pending = 0;
    // We advance the temporal pattern for dropped frames.
    cpi->temporal_pattern_counter++;
    return;
  }

  /* Reduce active_worst_allowed_q for CBR if our buffer is getting too full.
   * This has a knock on effect on active best quality as well.
   * For CBR if the buffer reaches its maximum level then we can no longer
   * save up bits for later frames so we might as well use them up
   * on the current frame.
   */
  if ((cpi->oxcf.end_usage == USAGE_STREAM_FROM_SERVER) &&
      (cpi->buffer_level >= cpi->oxcf.optimal_buffer_level) &&
      cpi->buffered_mode) {
    /* Max adjustment is 1/4 */
    int Adjustment = cpi->active_worst_quality / 4;

    if (Adjustment) {
      int buff_lvl_step;

      if (cpi->buffer_level < cpi->oxcf.maximum_buffer_size) {
        buff_lvl_step = (int)((cpi->oxcf.maximum_buffer_size -
                               cpi->oxcf.optimal_buffer_level) /
                              Adjustment);

        if (buff_lvl_step) {
          Adjustment =
              (int)((cpi->buffer_level - cpi->oxcf.optimal_buffer_level) /
                    buff_lvl_step);
        } else {
          Adjustment = 0;
        }
      }

      cpi->active_worst_quality -= Adjustment;

      if (cpi->active_worst_quality < cpi->active_best_quality) {
        cpi->active_worst_quality = cpi->active_best_quality;
      }
    }
  }

  /* Set an active best quality and if necessary active worst quality
   * There is some odd behavior for one pass here that needs attention.
   */
  if ((cpi->pass == 2) || (cpi->ni_frames > 150)) {
    vpx_clear_system_state();

    Q = cpi->active_worst_quality;

    if (cm->frame_type == KEY_FRAME) {
      if (cpi->pass == 2) {
        if (cpi->gfu_boost > 600) {
          cpi->active_best_quality = kf_low_motion_minq[Q];
        } else {
          cpi->active_best_quality = kf_high_motion_minq[Q];
        }

        /* Special case for key frames forced because we have reached
         * the maximum key frame interval. Here force the Q to a range
         * based on the ambient Q to reduce the risk of popping
         */
        if (cpi->this_key_frame_forced) {
          if (cpi->active_best_quality > cpi->avg_frame_qindex * 7 / 8) {
            cpi->active_best_quality = cpi->avg_frame_qindex * 7 / 8;
          } else if (cpi->active_best_quality < (cpi->avg_frame_qindex >> 2)) {
            cpi->active_best_quality = cpi->avg_frame_qindex >> 2;
          }
        }
      }
      /* One pass more conservative */
      else {
        cpi->active_best_quality = kf_high_motion_minq[Q];
      }
    }

    else if (cpi->oxcf.number_of_layers == 1 &&
             (cm->refresh_golden_frame || cpi->common.refresh_alt_ref_frame)) {
      /* Use the lower of cpi->active_worst_quality and recent
       * average Q as basis for GF/ARF Q limit unless last frame was
       * a key frame.
       */
      if ((cpi->frames_since_key > 1) &&
          (cpi->avg_frame_qindex < cpi->active_worst_quality)) {
        Q = cpi->avg_frame_qindex;
      }

      /* For constrained quality don't allow Q less than the cq level */
      if ((cpi->oxcf.end_usage == USAGE_CONSTRAINED_QUALITY) &&
          (Q < cpi->cq_target_quality)) {
        Q = cpi->cq_target_quality;
      }

      if (cpi->pass == 2) {
        if (cpi->gfu_boost > 1000) {
          cpi->active_best_quality = gf_low_motion_minq[Q];
        } else if (cpi->gfu_boost < 400) {
          cpi->active_best_quality = gf_high_motion_minq[Q];
        } else {
          cpi->active_best_quality = gf_mid_motion_minq[Q];
        }

        /* Constrained quality use slightly lower active best. */
        if (cpi->oxcf.end_usage == USAGE_CONSTRAINED_QUALITY) {
          cpi->active_best_quality = cpi->active_best_quality * 15 / 16;
        }
      }
      /* One pass more conservative */
      else {
        cpi->active_best_quality = gf_high_motion_minq[Q];
      }
    } else {
      cpi->active_best_quality = inter_minq[Q];

      /* For the constant/constrained quality mode we don't want
       * q to fall below the cq level.
       */
      if ((cpi->oxcf.end_usage == USAGE_CONSTRAINED_QUALITY) &&
          (cpi->active_best_quality < cpi->cq_target_quality)) {
        /* If we are strongly undershooting the target rate in the last
         * frames then use the user passed in cq value not the auto
         * cq value.
         */
        if (cpi->rolling_actual_bits < cpi->min_frame_bandwidth) {
          cpi->active_best_quality = cpi->oxcf.cq_level;
        } else {
          cpi->active_best_quality = cpi->cq_target_quality;
        }
      }
    }

    /* If CBR and the buffer is as full then it is reasonable to allow
     * higher quality on the frames to prevent bits just going to waste.
     */
    if (cpi->oxcf.end_usage == USAGE_STREAM_FROM_SERVER) {
      /* Note that the use of >= here elliminates the risk of a divide
       * by 0 error in the else if clause
       */
      if (cpi->buffer_level >= cpi->oxcf.maximum_buffer_size) {
        cpi->active_best_quality = cpi->best_quality;

      } else if (cpi->buffer_level > cpi->oxcf.optimal_buffer_level) {
        int Fraction =
            (int)(((cpi->buffer_level - cpi->oxcf.optimal_buffer_level) * 128) /
                  (cpi->oxcf.maximum_buffer_size -
                   cpi->oxcf.optimal_buffer_level));
        int min_qadjustment =
            ((cpi->active_best_quality - cpi->best_quality) * Fraction) / 128;

        cpi->active_best_quality -= min_qadjustment;
      }
    }
  }
  /* Make sure constrained quality mode limits are adhered to for the first
   * few frames of one pass encodes
   */
  else if (cpi->oxcf.end_usage == USAGE_CONSTRAINED_QUALITY) {
    if ((cm->frame_type == KEY_FRAME) || cm->refresh_golden_frame ||
        cpi->common.refresh_alt_ref_frame) {
      cpi->active_best_quality = cpi->best_quality;
    } else if (cpi->active_best_quality < cpi->cq_target_quality) {
      cpi->active_best_quality = cpi->cq_target_quality;
    }
  }

  /* Clip the active best and worst quality values to limits */
  if (cpi->active_worst_quality > cpi->worst_quality) {
    cpi->active_worst_quality = cpi->worst_quality;
  }

  if (cpi->active_best_quality < cpi->best_quality) {
    cpi->active_best_quality = cpi->best_quality;
  }

  if (cpi->active_worst_quality < cpi->active_best_quality) {
    cpi->active_worst_quality = cpi->active_best_quality;
  }

  /* Determine initial Q to try */
  Q = vp8_regulate_q(cpi, cpi->this_frame_target);

#if !CONFIG_REALTIME_ONLY

  /* Set highest allowed value for Zbin over quant */
  if (cm->frame_type == KEY_FRAME) {
    zbin_oq_high = 0;
  } else if ((cpi->oxcf.number_of_layers == 1) &&
             ((cm->refresh_alt_ref_frame ||
               (cm->refresh_golden_frame && !cpi->source_alt_ref_active)))) {
    zbin_oq_high = 16;
  } else {
    zbin_oq_high = ZBIN_OQ_MAX;
  }
#endif

  compute_skin_map(cpi);

  /* Setup background Q adjustment for error resilient mode.
   * For multi-layer encodes only enable this for the base layer.
   */
  if (cpi->cyclic_refresh_mode_enabled) {
    // Special case for screen_content_mode with golden frame updates.
    int disable_cr_gf =
        (cpi->oxcf.screen_content_mode == 2 && cm->refresh_golden_frame);
    if (cpi->current_layer == 0 && cpi->force_maxqp == 0 && !disable_cr_gf) {
      cyclic_background_refresh(cpi, Q, 0);
    } else {
      disable_segmentation(cpi);
    }
  }

  vp8_compute_frame_size_bounds(cpi, &frame_under_shoot_limit,
                                &frame_over_shoot_limit);

#if !CONFIG_REALTIME_ONLY
  /* Limit Q range for the adaptive loop. */
  bottom_index = cpi->active_best_quality;
  top_index = cpi->active_worst_quality;
  q_low = cpi->active_best_quality;
  q_high = cpi->active_worst_quality;
#endif

  vp8_save_coding_context(cpi);

  scale_and_extend_source(cpi->un_scaled_source, cpi);

#if CONFIG_TEMPORAL_DENOISING && CONFIG_POSTPROC
  // Option to apply spatial blur under the aggressive or adaptive
  // (temporal denoising) mode.
  if (cpi->oxcf.noise_sensitivity >= 3) {
    if (cpi->denoiser.denoise_pars.spatial_blur != 0) {
      vp8_de_noise(cm, cpi->Source, cpi->denoiser.denoise_pars.spatial_blur, 1);
    }
  }
#endif

#if !(CONFIG_REALTIME_ONLY) && CONFIG_POSTPROC && !(CONFIG_TEMPORAL_DENOISING)

  if (cpi->oxcf.noise_sensitivity > 0) {
    unsigned char *src;
    int l = 0;

    switch (cpi->oxcf.noise_sensitivity) {
      case 1: l = 20; break;
      case 2: l = 40; break;
      case 3: l = 60; break;
      case 4: l = 80; break;
      case 5: l = 100; break;
      case 6: l = 150; break;
    }

    if (cm->frame_type == KEY_FRAME) {
      vp8_de_noise(cm, cpi->Source, l, 1);
    } else {
      vp8_de_noise(cm, cpi->Source, l, 1);

      src = cpi->Source->y_buffer;

      if (cpi->Source->y_stride < 0) {
        src += cpi->Source->y_stride * (cpi->Source->y_height - 1);
      }
    }
  }

#endif

#ifdef OUTPUT_YUV_SRC
  vpx_write_yuv_frame(yuv_file, cpi->Source);
#endif

  do {
    vpx_clear_system_state();

    vp8_set_quantizer(cpi, Q);

    /* setup skip prob for costing in mode/mv decision */
    if (cpi->common.mb_no_coeff_skip) {
      cpi->prob_skip_false = cpi->base_skip_false_prob[Q];

      if (cm->frame_type != KEY_FRAME) {
        if (cpi->common.refresh_alt_ref_frame) {
          if (cpi->last_skip_false_probs[2] != 0) {
            cpi->prob_skip_false = cpi->last_skip_false_probs[2];
          }

          /*
                              if(cpi->last_skip_false_probs[2]!=0 && abs(Q-
             cpi->last_skip_probs_q[2])<=16 )
             cpi->prob_skip_false = cpi->last_skip_false_probs[2];
                              else if (cpi->last_skip_false_probs[2]!=0)
             cpi->prob_skip_false = (cpi->last_skip_false_probs[2]  +
             cpi->prob_skip_false ) / 2;
             */
        } else if (cpi->common.refresh_golden_frame) {
          if (cpi->last_skip_false_probs[1] != 0) {
            cpi->prob_skip_false = cpi->last_skip_false_probs[1];
          }

          /*
                              if(cpi->last_skip_false_probs[1]!=0 && abs(Q-
             cpi->last_skip_probs_q[1])<=16 )
             cpi->prob_skip_false = cpi->last_skip_false_probs[1];
                              else if (cpi->last_skip_false_probs[1]!=0)
             cpi->prob_skip_false = (cpi->last_skip_false_probs[1]  +
             cpi->prob_skip_false ) / 2;
             */
        } else {
          if (cpi->last_skip_false_probs[0] != 0) {
            cpi->prob_skip_false = cpi->last_skip_false_probs[0];
          }

          /*
          if(cpi->last_skip_false_probs[0]!=0 && abs(Q-
          cpi->last_skip_probs_q[0])<=16 )
              cpi->prob_skip_false = cpi->last_skip_false_probs[0];
          else if(cpi->last_skip_false_probs[0]!=0)
              cpi->prob_skip_false = (cpi->last_skip_false_probs[0]  +
          cpi->prob_skip_false ) / 2;
              */
        }

        /* as this is for cost estimate, let's make sure it does not
         * go extreme eitehr way
         */
        if (cpi->prob_skip_false < 5) cpi->prob_skip_false = 5;

        if (cpi->prob_skip_false > 250) cpi->prob_skip_false = 250;

        if (cpi->oxcf.number_of_layers == 1 && cpi->is_src_frame_alt_ref) {
          cpi->prob_skip_false = 1;
        }
      }

#if 0

            if (cpi->pass != 1)
            {
                FILE *f = fopen("skip.stt", "a");
                fprintf(f, "%d, %d, %4d ", cpi->common.refresh_golden_frame, cpi->common.refresh_alt_ref_frame, cpi->prob_skip_false);
                fclose(f);
            }

#endif
    }

    if (cm->frame_type == KEY_FRAME) {
      if (resize_key_frame(cpi)) {
        /* If the frame size has changed, need to reset Q, quantizer,
         * and background refresh.
         */
        Q = vp8_regulate_q(cpi, cpi->this_frame_target);
        if (cpi->cyclic_refresh_mode_enabled) {
          if (cpi->current_layer == 0) {
            cyclic_background_refresh(cpi, Q, 0);
          } else {
            disable_segmentation(cpi);
          }
        }
        // Reset the zero_last counter to 0 on key frame.
        memset(cpi->consec_zero_last, 0, cm->mb_rows * cm->mb_cols);
        memset(cpi->consec_zero_last_mvbias, 0,
               (cpi->common.mb_rows * cpi->common.mb_cols));
        vp8_set_quantizer(cpi, Q);
      }

      vp8_setup_key_frame(cpi);
    }

#if CONFIG_REALTIME_ONLY & CONFIG_ONTHEFLY_BITPACKING
    {
      if (cpi->oxcf.error_resilient_mode) cm->refresh_entropy_probs = 0;

      if (cpi->oxcf.error_resilient_mode & VPX_ERROR_RESILIENT_PARTITIONS) {
        if (cm->frame_type == KEY_FRAME) cm->refresh_entropy_probs = 1;
      }

      if (cm->refresh_entropy_probs == 0) {
        /* save a copy for later refresh */
        cm->lfc = cm->fc;
      }

      vp8_update_coef_context(cpi);

      vp8_update_coef_probs(cpi);

      /* transform / motion compensation build reconstruction frame
       * +pack coef partitions
       */
      vp8_encode_frame(cpi);

      /* cpi->projected_frame_size is not needed for RT mode */
    }
#else
    /* transform / motion compensation build reconstruction frame */
    vp8_encode_frame(cpi);

    if (cpi->pass == 0 && cpi->oxcf.end_usage == USAGE_STREAM_FROM_SERVER &&
        cpi->rt_drop_recode_on_overshoot == 1) {
      if (vp8_drop_encodedframe_overshoot(cpi, Q)) {
        vpx_clear_system_state();
        return;
      }
      if (cm->frame_type != KEY_FRAME)
        cpi->last_pred_err_mb =
            (int)(cpi->mb.prediction_error / cpi->common.MBs);
    }

    cpi->projected_frame_size -= vp8_estimate_entropy_savings(cpi);
    cpi->projected_frame_size =
        (cpi->projected_frame_size > 0) ? cpi->projected_frame_size : 0;
#endif
    vpx_clear_system_state();

    /* Test to see if the stats generated for this frame indicate that
     * we should have coded a key frame (assuming that we didn't)!
     */

    if (cpi->pass != 2 && cpi->oxcf.auto_key && cm->frame_type != KEY_FRAME &&
        cpi->compressor_speed != 2) {
#if !CONFIG_REALTIME_ONLY
      if (decide_key_frame(cpi)) {
        /* Reset all our sizing numbers and recode */
        cm->frame_type = KEY_FRAME;

        vp8_pick_frame_size(cpi);

        /* Clear the Alt reference frame active flag when we have
         * a key frame
         */
        cpi->source_alt_ref_active = 0;

        // Set the loop filter deltas and segmentation map update
        setup_features(cpi);

        vp8_restore_coding_context(cpi);

        Q = vp8_regulate_q(cpi, cpi->this_frame_target);

        vp8_compute_frame_size_bounds(cpi, &frame_under_shoot_limit,
                                      &frame_over_shoot_limit);

        /* Limit Q range for the adaptive loop. */
        bottom_index = cpi->active_best_quality;
        top_index = cpi->active_worst_quality;
        q_low = cpi->active_best_quality;
        q_high = cpi->active_worst_quality;

        Loop = 1;

        continue;
      }
#endif
    }

    vpx_clear_system_state();

    if (frame_over_shoot_limit == 0) frame_over_shoot_limit = 1;

    /* Are we are overshooting and up against the limit of active max Q. */
    if (!cpi->rt_always_update_correction_factor &&
        ((cpi->pass != 2) ||
         (cpi->oxcf.end_usage == USAGE_STREAM_FROM_SERVER)) &&
        (Q == cpi->active_worst_quality) &&
        (cpi->active_worst_quality < cpi->worst_quality) &&
        (cpi->projected_frame_size > frame_over_shoot_limit)) {
      int over_size_percent =
          ((cpi->projected_frame_size - frame_over_shoot_limit) * 100) /
          frame_over_shoot_limit;

      /* If so is there any scope for relaxing it */
      while ((cpi->active_worst_quality < cpi->worst_quality) &&
             (over_size_percent > 0)) {
        cpi->active_worst_quality++;
        /* Assume 1 qstep = about 4% on frame size. */
        over_size_percent = (int)(over_size_percent * 0.96);
      }
#if !CONFIG_REALTIME_ONLY
      top_index = cpi->active_worst_quality;
#endif  // !CONFIG_REALTIME_ONLY
      /* If we have updated the active max Q do not call
       * vp8_update_rate_correction_factors() this loop.
       */
      active_worst_qchanged = 1;
    } else {
      active_worst_qchanged = 0;
    }

#if CONFIG_REALTIME_ONLY
    Loop = 0;
#else
    /* Special case handling for forced key frames */
    if ((cm->frame_type == KEY_FRAME) && cpi->this_key_frame_forced) {
      int last_q = Q;
      int kf_err = vp8_calc_ss_err(cpi->Source, &cm->yv12_fb[cm->new_fb_idx]);

      /* The key frame is not good enough */
      if (kf_err > ((cpi->ambient_err * 7) >> 3)) {
        /* Lower q_high */
        q_high = (Q > q_low) ? (Q - 1) : q_low;

        /* Adjust Q */
        Q = (q_high + q_low) >> 1;
      }
      /* The key frame is much better than the previous frame */
      else if (kf_err < (cpi->ambient_err >> 1)) {
        /* Raise q_low */
        q_low = (Q < q_high) ? (Q + 1) : q_high;

        /* Adjust Q */
        Q = (q_high + q_low + 1) >> 1;
      }

      /* Clamp Q to upper and lower limits: */
      if (Q > q_high) {
        Q = q_high;
      } else if (Q < q_low) {
        Q = q_low;
      }

      Loop = Q != last_q;
    }

    /* Is the projected frame size out of range and are we allowed
     * to attempt to recode.
     */
    else if (recode_loop_test(cpi, frame_over_shoot_limit,
                              frame_under_shoot_limit, Q, top_index,
                              bottom_index)) {
      int last_q = Q;
      int Retries = 0;

      /* Frame size out of permitted range. Update correction factor
       * & compute new Q to try...
       */

      /* Frame is too large */
      if (cpi->projected_frame_size > cpi->this_frame_target) {
        /* Raise Qlow as to at least the current value */
        q_low = (Q < q_high) ? (Q + 1) : q_high;

        /* If we are using over quant do the same for zbin_oq_low */
        if (cpi->mb.zbin_over_quant > 0) {
          zbin_oq_low = (cpi->mb.zbin_over_quant < zbin_oq_high)
                            ? (cpi->mb.zbin_over_quant + 1)
                            : zbin_oq_high;
        }

        if (undershoot_seen) {
          /* Update rate_correction_factor unless
           * cpi->active_worst_quality has changed.
           */
          if (!active_worst_qchanged) {
            vp8_update_rate_correction_factors(cpi, 1);
          }

          Q = (q_high + q_low + 1) / 2;

          /* Adjust cpi->zbin_over_quant (only allowed when Q
           * is max)
           */
          if (Q < MAXQ) {
            cpi->mb.zbin_over_quant = 0;
          } else {
            zbin_oq_low = (cpi->mb.zbin_over_quant < zbin_oq_high)
                              ? (cpi->mb.zbin_over_quant + 1)
                              : zbin_oq_high;
            cpi->mb.zbin_over_quant = (zbin_oq_high + zbin_oq_low) / 2;
          }
        } else {
          /* Update rate_correction_factor unless
           * cpi->active_worst_quality has changed.
           */
          if (!active_worst_qchanged) {
            vp8_update_rate_correction_factors(cpi, 0);
          }

          Q = vp8_regulate_q(cpi, cpi->this_frame_target);

          while (((Q < q_low) || (cpi->mb.zbin_over_quant < zbin_oq_low)) &&
                 (Retries < 10)) {
            vp8_update_rate_correction_factors(cpi, 0);
            Q = vp8_regulate_q(cpi, cpi->this_frame_target);
            Retries++;
          }
        }

        overshoot_seen = 1;
      }
      /* Frame is too small */
      else {
        if (cpi->mb.zbin_over_quant == 0) {
          /* Lower q_high if not using over quant */
          q_high = (Q > q_low) ? (Q - 1) : q_low;
        } else {
          /* else lower zbin_oq_high */
          zbin_oq_high = (cpi->mb.zbin_over_quant > zbin_oq_low)
                             ? (cpi->mb.zbin_over_quant - 1)
                             : zbin_oq_low;
        }

        if (overshoot_seen) {
          /* Update rate_correction_factor unless
           * cpi->active_worst_quality has changed.
           */
          if (!active_worst_qchanged) {
            vp8_update_rate_correction_factors(cpi, 1);
          }

          Q = (q_high + q_low) / 2;

          /* Adjust cpi->zbin_over_quant (only allowed when Q
           * is max)
           */
          if (Q < MAXQ) {
            cpi->mb.zbin_over_quant = 0;
          } else {
            cpi->mb.zbin_over_quant = (zbin_oq_high + zbin_oq_low) / 2;
          }
        } else {
          /* Update rate_correction_factor unless
           * cpi->active_worst_quality has changed.
           */
          if (!active_worst_qchanged) {
            vp8_update_rate_correction_factors(cpi, 0);
          }

          Q = vp8_regulate_q(cpi, cpi->this_frame_target);

          /* Special case reset for qlow for constrained quality.
           * This should only trigger where there is very substantial
           * undershoot on a frame and the auto cq level is above
           * the user passsed in value.
           */
          if ((cpi->oxcf.end_usage == USAGE_CONSTRAINED_QUALITY) &&
              (Q < q_low)) {
            q_low = Q;
          }

          while (((Q > q_high) || (cpi->mb.zbin_over_quant > zbin_oq_high)) &&
                 (Retries < 10)) {
            vp8_update_rate_correction_factors(cpi, 0);
            Q = vp8_regulate_q(cpi, cpi->this_frame_target);
            Retries++;
          }
        }

        undershoot_seen = 1;
      }

      /* Clamp Q to upper and lower limits: */
      if (Q > q_high) {
        Q = q_high;
      } else if (Q < q_low) {
        Q = q_low;
      }

      /* Clamp cpi->zbin_over_quant */
      cpi->mb.zbin_over_quant =
          (cpi->mb.zbin_over_quant < zbin_oq_low)    ? zbin_oq_low
          : (cpi->mb.zbin_over_quant > zbin_oq_high) ? zbin_oq_high
                                                     : cpi->mb.zbin_over_quant;

      Loop = Q != last_q;
    } else {
      Loop = 0;
    }
#endif  // CONFIG_REALTIME_ONLY

    if (cpi->is_src_frame_alt_ref) Loop = 0;

    if (Loop == 1) {
      vp8_restore_coding_context(cpi);
#if CONFIG_INTERNAL_STATS
      cpi->tot_recode_hits++;
#endif
    }
  } while (Loop == 1);

#if defined(DROP_UNCODED_FRAMES)
  /* if there are no coded macroblocks at all drop this frame */
  if (cpi->common.MBs == cpi->mb.skip_true_count &&
      (cpi->drop_frame_count & 7) != 7 && cm->frame_type != KEY_FRAME) {
    cpi->common.current_video_frame++;
    cpi->frames_since_key++;
    cpi->drop_frame_count++;
    cpi->ext_refresh_frame_flags_pending = 0;
    // We advance the temporal pattern for dropped frames.
    cpi->temporal_pattern_counter++;
    return;
  }
  cpi->drop_frame_count = 0;
#endif

#if 0
    /* Experimental code for lagged and one pass
     * Update stats used for one pass GF selection
     */
    {
        cpi->one_pass_frame_stats[cpi->one_pass_frame_index].frame_coded_error = (double)cpi->prediction_error;
        cpi->one_pass_frame_stats[cpi->one_pass_frame_index].frame_intra_error = (double)cpi->intra_error;
        cpi->one_pass_frame_stats[cpi->one_pass_frame_index].frame_pcnt_inter = (double)(100 - cpi->this_frame_percent_intra) / 100.0;
    }
#endif

  /* Special case code to reduce pulsing when key frames are forced at a
   * fixed interval. Note the reconstruction error if it is the frame before
   * the force key frame
   */
  if (cpi->next_key_frame_forced && (cpi->twopass.frames_to_key == 0)) {
    cpi->ambient_err =
        vp8_calc_ss_err(cpi->Source, &cm->yv12_fb[cm->new_fb_idx]);
  }

/* This frame's MVs are saved and will be used in next frame's MV predictor.
 * Last frame has one more line(add to bottom) and one more column(add to
 * right) than cm->mip. The edge elements are initialized to 0.
 */
#if CONFIG_MULTI_RES_ENCODING
  if (!cpi->oxcf.mr_encoder_id && cm->show_frame)
#else
  if (cm->show_frame) /* do not save for altref frame */
#endif
  {
    int mb_row;
    int mb_col;
    /* Point to beginning of allocated MODE_INFO arrays. */
    MODE_INFO *tmp = cm->mip;

    if (cm->frame_type != KEY_FRAME) {
      for (mb_row = 0; mb_row < cm->mb_rows + 1; ++mb_row) {
        for (mb_col = 0; mb_col < cm->mb_cols + 1; ++mb_col) {
          if (tmp->mbmi.ref_frame != INTRA_FRAME) {
            cpi->lfmv[mb_col + mb_row * (cm->mode_info_stride + 1)].as_int =
                tmp->mbmi.mv.as_int;
          }

          cpi->lf_ref_frame_sign_bias[mb_col +
                                      mb_row * (cm->mode_info_stride + 1)] =
              cm->ref_frame_sign_bias[tmp->mbmi.ref_frame];
          cpi->lf_ref_frame[mb_col + mb_row * (cm->mode_info_stride + 1)] =
              tmp->mbmi.ref_frame;
          tmp++;
        }
      }
    }
  }

  /* Count last ref frame 0,0 usage on current encoded frame. */
  {
    int mb_row;
    int mb_col;
    /* Point to beginning of MODE_INFO arrays. */
    MODE_INFO *tmp = cm->mi;

    cpi->zeromv_count = 0;

    if (cm->frame_type != KEY_FRAME) {
      for (mb_row = 0; mb_row < cm->mb_rows; ++mb_row) {
        for (mb_col = 0; mb_col < cm->mb_cols; ++mb_col) {
          if (tmp->mbmi.mode == ZEROMV && tmp->mbmi.ref_frame == LAST_FRAME) {
            cpi->zeromv_count++;
          }
          tmp++;
        }
        tmp++;
      }
    }
  }

#if CONFIG_MULTI_RES_ENCODING
  vp8_cal_dissimilarity(cpi);
#endif

  /* Update the GF usage maps.
   * This is done after completing the compression of a frame when all
   * modes etc. are finalized but before loop filter
   */
  if (cpi->oxcf.number_of_layers == 1) {
    vp8_update_gf_usage_maps(cpi, cm, &cpi->mb);
  }

  if (cm->frame_type == KEY_FRAME) cm->refresh_last_frame = 1;

#if 0
    {
        FILE *f = fopen("gfactive.stt", "a");
        fprintf(f, "%8d %8d %8d %8d %8d\n", cm->current_video_frame, (100 * cpi->gf_active_count) / (cpi->common.mb_rows * cpi->common.mb_cols), cpi->this_iiratio, cpi->next_iiratio, cm->refresh_golden_frame);
        fclose(f);
    }
#endif

  /* For inter frames the current default behavior is that when
   * cm->refresh_golden_frame is set we copy the old GF over to the ARF buffer
   * This is purely an encoder decision at present.
   * Avoid this behavior when refresh flags are set by the user.
   */
  if (!cpi->oxcf.error_resilient_mode && cm->refresh_golden_frame &&
      !cpi->ext_refresh_frame_flags_pending) {
    cm->copy_buffer_to_arf = 2;
  } else {
    cm->copy_buffer_to_arf = 0;
  }

  cm->frame_to_show = &cm->yv12_fb[cm->new_fb_idx];

#if CONFIG_TEMPORAL_DENOISING
  // Get some measure of the amount of noise, by measuring the (partial) mse
  // between source and denoised buffer, for y channel. Partial refers to
  // computing the sse for a sub-sample of the frame (i.e., skip x blocks along
  // row/column),
  // and only for blocks in that set that are consecutive ZEROMV_LAST mode.
  // Do this every ~8 frames, to further reduce complexity.
  // TODO(marpan): Keep this for now for the case cpi->oxcf.noise_sensitivity <
  // 4,
  // should be removed in favor of the process_denoiser_mode_change() function
  // below.
  if (cpi->oxcf.noise_sensitivity > 0 && cpi->oxcf.noise_sensitivity < 4 &&
      !cpi->oxcf.screen_content_mode && cpi->frames_since_key % 8 == 0 &&
      cm->frame_type != KEY_FRAME) {
    cpi->mse_source_denoised = measure_square_diff_partial(
        &cpi->denoiser.yv12_running_avg[INTRA_FRAME], cpi->Source, cpi);
  }

  // For the adaptive denoising mode (noise_sensitivity == 4), sample the mse
  // of source diff (between current and previous frame), and determine if we
  // should switch the denoiser mode. Sampling refers to computing the mse for
  // a sub-sample of the frame (i.e., skip x blocks along row/column), and
  // only for blocks in that set that have used ZEROMV LAST, along with some
  // constraint on the sum diff between blocks. This process is called every
  // ~8 frames, to further reduce complexity.
  if (cpi->oxcf.noise_sensitivity == 4 && !cpi->oxcf.screen_content_mode &&
      cpi->frames_since_key % 8 == 0 && cm->frame_type != KEY_FRAME) {
    process_denoiser_mode_change(cpi);
  }
#endif

#ifdef OUTPUT_YUV_SKINMAP
  if (cpi->common.current_video_frame > 1) {
    vp8_compute_skin_map(cpi, yuv_skinmap_file);
  }
#endif

#if CONFIG_MULTITHREAD
  if (vpx_atomic_load_acquire(&cpi->b_multi_threaded)) {
    /* start loopfilter in separate thread */
    vp8_sem_post(&cpi->h_event_start_lpf);
    cpi->b_lpf_running = 1;
    /* wait for the filter_level to be picked so that we can continue with
     * stream packing */
    vp8_sem_wait(&cpi->h_event_end_lpf);
  } else
#endif
  {
    vp8_loopfilter_frame(cpi, cm);
  }

  update_reference_frames(cpi);

#ifdef OUTPUT_YUV_DENOISED
  vpx_write_yuv_frame(yuv_denoised_file,
                      &cpi->denoiser.yv12_running_avg[INTRA_FRAME]);
#endif

#if !(CONFIG_REALTIME_ONLY & CONFIG_ONTHEFLY_BITPACKING)
  if (cpi->oxcf.error_resilient_mode) {
    cm->refresh_entropy_probs = 0;
  }
#endif

  /* build the bitstream */
  vp8_pack_bitstream(cpi, dest, dest_end, size);

  /* Move storing frame_type out of the above loop since it is also
   * needed in motion search besides loopfilter */
  cm->last_frame_type = cm->frame_type;

  /* Update rate control heuristics */
  cpi->total_byte_count += (*size);
  cpi->projected_frame_size = (int)(*size) << 3;

  if (cpi->oxcf.number_of_layers > 1) {
    unsigned int i;
    for (i = cpi->current_layer + 1; i < cpi->oxcf.number_of_layers; ++i) {
      cpi->layer_context[i].total_byte_count += (*size);
    }
  }

  if (!active_worst_qchanged) vp8_update_rate_correction_factors(cpi, 2);

  cpi->last_q[cm->frame_type] = cm->base_qindex;

  if (cm->frame_type == KEY_FRAME) {
    vp8_adjust_key_frame_context(cpi);
  }

  /* Keep a record of ambient average Q. */
  if (cm->frame_type != KEY_FRAME) {
    cpi->avg_frame_qindex =
        (2 + 3 * cpi->avg_frame_qindex + cm->base_qindex) >> 2;
  }

  /* Keep a record from which we can calculate the average Q excluding
   * GF updates and key frames
   */
  if ((cm->frame_type != KEY_FRAME) &&
      ((cpi->oxcf.number_of_layers > 1) ||
       (!cm->refresh_golden_frame && !cm->refresh_alt_ref_frame))) {
    cpi->ni_frames++;

    /* Calculate the average Q for normal inter frames (not key or GFU
     * frames).
     */
    if (cpi->pass == 2) {
      cpi->ni_tot_qi += Q;
      cpi->ni_av_qi = (cpi->ni_tot_qi / cpi->ni_frames);
    } else {
      /* Damp value for first few frames */
      if (cpi->ni_frames > 150) {
        cpi->ni_tot_qi += Q;
        cpi->ni_av_qi = (cpi->ni_tot_qi / cpi->ni_frames);
      }
      /* For one pass, early in the clip ... average the current frame Q
       * value with the worstq entered by the user as a dampening measure
       */
      else {
        cpi->ni_tot_qi += Q;
        cpi->ni_av_qi =
            ((cpi->ni_tot_qi / cpi->ni_frames) + cpi->worst_quality + 1) / 2;
      }

      /* If the average Q is higher than what was used in the last
       * frame (after going through the recode loop to keep the frame
       * size within range) then use the last frame value - 1. The -1
       * is designed to stop Q and hence the data rate, from
       * progressively falling away during difficult sections, but at
       * the same time reduce the number of iterations around the
       * recode loop.
       */
      if (Q > cpi->ni_av_qi) cpi->ni_av_qi = Q - 1;
    }
  }

  /* Update the buffer level variable. */
  /* Non-viewable frames are a special case and are treated as pure overhead. */
  if (!cm->show_frame) {
    cpi->bits_off_target -= cpi->projected_frame_size;
  } else {
    cpi->bits_off_target +=
        cpi->av_per_frame_bandwidth - cpi->projected_frame_size;
  }

  /* Clip the buffer level to the maximum specified buffer size */
  if (cpi->bits_off_target > cpi->oxcf.maximum_buffer_size) {
    cpi->bits_off_target = cpi->oxcf.maximum_buffer_size;
  }

  // Don't let the buffer level go below some threshold, given here
  // by -|maximum_buffer_size|. For now we only do this for
  // screen content input.
  if (cpi->oxcf.screen_content_mode &&
      cpi->bits_off_target < -cpi->oxcf.maximum_buffer_size) {
    cpi->bits_off_target = -cpi->oxcf.maximum_buffer_size;
  }

  /* Rolling monitors of whether we are over or underspending used to
   * help regulate min and Max Q in two pass.
   */
  cpi->rolling_target_bits = (int)ROUND64_POWER_OF_TWO(
      (int64_t)cpi->rolling_target_bits * 3 + cpi->this_frame_target, 2);
  cpi->rolling_actual_bits = (int)ROUND64_POWER_OF_TWO(
      (int64_t)cpi->rolling_actual_bits * 3 + cpi->projected_frame_size, 2);
  cpi->long_rolling_target_bits = (int)ROUND64_POWER_OF_TWO(
      (int64_t)cpi->long_rolling_target_bits * 31 + cpi->this_frame_target, 5);
  cpi->long_rolling_actual_bits = (int)ROUND64_POWER_OF_TWO(
      (int64_t)cpi->long_rolling_actual_bits * 31 + cpi->projected_frame_size,
      5);

  /* Actual bits spent */
  cpi->total_actual_bits += cpi->projected_frame_size;

#if 0 && CONFIG_INTERNAL_STATS
  /* Debug stats */
  cpi->total_target_vs_actual +=
      (cpi->this_frame_target - cpi->projected_frame_size);
#endif

  cpi->buffer_level = cpi->bits_off_target;

  /* Propagate values to higher temporal layers */
  if (cpi->oxcf.number_of_layers > 1) {
    unsigned int i;

    for (i = cpi->current_layer + 1; i < cpi->oxcf.number_of_layers; ++i) {
      LAYER_CONTEXT *lc = &cpi->layer_context[i];
      int bits_off_for_this_layer = (int)round(
          lc->target_bandwidth / lc->framerate - cpi->projected_frame_size);

      lc->bits_off_target += bits_off_for_this_layer;

      /* Clip buffer level to maximum buffer size for the layer */
      if (lc->bits_off_target > lc->maximum_buffer_size) {
        lc->bits_off_target = lc->maximum_buffer_size;
      }

      lc->total_actual_bits += cpi->projected_frame_size;
      lc->total_target_vs_actual += bits_off_for_this_layer;
      lc->buffer_level = lc->bits_off_target;
    }
  }

  /* Update bits left to the kf and gf groups to account for overshoot
   * or undershoot on these frames
   */
  if (cm->frame_type == KEY_FRAME) {
    cpi->twopass.kf_group_bits +=
        cpi->this_frame_target - cpi->projected_frame_size;

    if (cpi->twopass.kf_group_bits < 0) cpi->twopass.kf_group_bits = 0;
  } else if (cm->refresh_golden_frame || cm->refresh_alt_ref_frame) {
    cpi->twopass.gf_group_bits +=
        cpi->this_frame_target - cpi->projected_frame_size;

    if (cpi->twopass.gf_group_bits < 0) cpi->twopass.gf_group_bits = 0;
  }

  if (cm->frame_type != KEY_FRAME) {
    if (cpi->common.refresh_alt_ref_frame) {
      cpi->last_skip_false_probs[2] = cpi->prob_skip_false;
      cpi->last_skip_probs_q[2] = cm->base_qindex;
    } else if (cpi->common.refresh_golden_frame) {
      cpi->last_skip_false_probs[1] = cpi->prob_skip_false;
      cpi->last_skip_probs_q[1] = cm->base_qindex;
    } else {
      cpi->last_skip_false_probs[0] = cpi->prob_skip_false;
      cpi->last_skip_probs_q[0] = cm->base_qindex;

      /* update the baseline */
      cpi->base_skip_false_prob[cm->base_qindex] = cpi->prob_skip_false;
    }
  }

#if 0 && CONFIG_INTERNAL_STATS
    {
        FILE *f = fopen("tmp.stt", "a");

        vpx_clear_system_state();

        if (cpi->twopass.total_left_stats.coded_error != 0.0)
            fprintf(f, "%10d %10d %10d %10d %10d %10"PRId64" %10"PRId64
                       "%10"PRId64" %10d %6d %6d %6d %6d %5d %5d %5d %8d "
                       "%8.2lf %"PRId64" %10.3lf %10"PRId64" %8d\n",
                       cpi->common.current_video_frame, cpi->this_frame_target,
                       cpi->projected_frame_size,
                       (cpi->projected_frame_size - cpi->this_frame_target),
                       cpi->total_target_vs_actual,
                       cpi->buffer_level,
                       (cpi->oxcf.starting_buffer_level-cpi->bits_off_target),
                       cpi->total_actual_bits, cm->base_qindex,
                       cpi->active_best_quality, cpi->active_worst_quality,
                       cpi->ni_av_qi, cpi->cq_target_quality,
                       cm->refresh_golden_frame, cm->refresh_alt_ref_frame,
                       cm->frame_type, cpi->gfu_boost,
                       cpi->twopass.est_max_qcorrection_factor,
                       cpi->twopass.bits_left,
                       cpi->twopass.total_left_stats.coded_error,
                       (double)cpi->twopass.bits_left /
                           cpi->twopass.total_left_stats.coded_error,
                       cpi->tot_recode_hits);
        else
            fprintf(f, "%10d %10d %10d %10d %10d %10"PRId64" %10"PRId64
                       "%10"PRId64" %10d %6d %6d %6d %6d %5d %5d %5d %8d "
                       "%8.2lf %"PRId64" %10.3lf %8d\n",
                       cpi->common.current_video_frame, cpi->this_frame_target,
                       cpi->projected_frame_size,
                       (cpi->projected_frame_size - cpi->this_frame_target),
                       cpi->total_target_vs_actual,
                       cpi->buffer_level,
                       (cpi->oxcf.starting_buffer_level-cpi->bits_off_target),
                       cpi->total_actual_bits, cm->base_qindex,
                       cpi->active_best_quality, cpi->active_worst_quality,
                       cpi->ni_av_qi, cpi->cq_target_quality,
                       cm->refresh_golden_frame, cm->refresh_alt_ref_frame,
                       cm->frame_type, cpi->gfu_boost,
                       cpi->twopass.est_max_qcorrection_factor,
                       cpi->twopass.bits_left,
                       cpi->twopass.total_left_stats.coded_error,
                       cpi->tot_recode_hits);

        fclose(f);

        {
            FILE *fmodes = fopen("Modes.stt", "a");

            fprintf(fmodes, "%6d:%1d:%1d:%1d ",
                        cpi->common.current_video_frame,
                        cm->frame_type, cm->refresh_golden_frame,
                        cm->refresh_alt_ref_frame);

            fprintf(fmodes, "\n");

            fclose(fmodes);
        }
    }

#endif

  cpi->ext_refresh_frame_flags_pending = 0;

  if (cm->refresh_golden_frame == 1) {
    cm->frame_flags = cm->frame_flags | FRAMEFLAGS_GOLDEN;
  } else {
    cm->frame_flags = cm->frame_flags & ~FRAMEFLAGS_GOLDEN;
  }

  if (cm->refresh_alt_ref_frame == 1) {
    cm->frame_flags = cm->frame_flags | FRAMEFLAGS_ALTREF;
  } else {
    cm->frame_flags = cm->frame_flags & ~FRAMEFLAGS_ALTREF;
  }

  if (cm->refresh_last_frame & cm->refresh_golden_frame) { /* both refreshed */
    cpi->gold_is_last = 1;
  } else if (cm->refresh_last_frame ^ cm->refresh_golden_frame) {
    /* 1 refreshed but not the other */
    cpi->gold_is_last = 0;
  }

  if (cm->refresh_last_frame & cm->refresh_alt_ref_frame) { /* both refreshed */
    cpi->alt_is_last = 1;
  } else if (cm->refresh_last_frame ^ cm->refresh_alt_ref_frame) {
    /* 1 refreshed but not the other */
    cpi->alt_is_last = 0;
  }

  if (cm->refresh_alt_ref_frame &
      cm->refresh_golden_frame) { /* both refreshed */
    cpi->gold_is_alt = 1;
  } else if (cm->refresh_alt_ref_frame ^ cm->refresh_golden_frame) {
    /* 1 refreshed but not the other */
    cpi->gold_is_alt = 0;
  }

  cpi->ref_frame_flags = VP8_ALTR_FRAME | VP8_GOLD_FRAME | VP8_LAST_FRAME;

  if (cpi->gold_is_last) cpi->ref_frame_flags &= ~VP8_GOLD_FRAME;

  if (cpi->alt_is_last) cpi->ref_frame_flags &= ~VP8_ALTR_FRAME;

  if (cpi->gold_is_alt) cpi->ref_frame_flags &= ~VP8_ALTR_FRAME;

  if (!cpi->oxcf.error_resilient_mode) {
    if (cpi->oxcf.play_alternate && cm->refresh_alt_ref_frame &&
        (cm->frame_type != KEY_FRAME)) {
      /* Update the alternate reference frame stats as appropriate. */
      update_alt_ref_frame_stats(cpi);
    } else {
      /* Update the Golden frame stats as appropriate. */
      update_golden_frame_stats(cpi);
    }
  }

  if (cm->frame_type == KEY_FRAME) {
    /* Tell the caller that the frame was coded as a key frame */
    *frame_flags = cm->frame_flags | FRAMEFLAGS_KEY;

    /* As this frame is a key frame  the next defaults to an inter frame. */
    cm->frame_type = INTER_FRAME;

    cpi->last_frame_percent_intra = 100;
  } else {
    *frame_flags = cm->frame_flags & ~FRAMEFLAGS_KEY;

    cpi->last_frame_percent_intra = cpi->this_frame_percent_intra;
  }

  /* Clear the one shot update flags for segmentation map and mode/ref
   * loop filter deltas.
   */
  cpi->mb.e_mbd.update_mb_segmentation_map = 0;
  cpi->mb.e_mbd.update_mb_segmentation_data = 0;
  cpi->mb.e_mbd.mode_ref_lf_delta_update = 0;

  /* Don't increment frame counters if this was an altref buffer update
   * not a real frame
   */
  if (cm->show_frame) {
    cm->current_video_frame++;
    cpi->frames_since_key++;
    cpi->temporal_pattern_counter++;
  }

#if 0
    {
        char filename[512];
        FILE *recon_file;
        sprintf(filename, "enc%04d.yuv", (int) cm->current_video_frame);
        recon_file = fopen(filename, "wb");
        fwrite(cm->yv12_fb[cm->lst_fb_idx].buffer_alloc,
               cm->yv12_fb[cm->lst_fb_idx].frame_size, 1, recon_file);
        fclose(recon_file);
    }
#endif

  /* DEBUG */
  /* vpx_write_yuv_frame("encoder_recon.yuv", cm->frame_to_show); */
}
#if !CONFIG_REALTIME_ONLY
static void Pass2Encode(VP8_COMP *cpi, size_t *size, unsigned char *dest,
                        unsigned char *dest_end, unsigned int *frame_flags) {
  if (!cpi->common.refresh_alt_ref_frame) vp8_second_pass(cpi);

  encode_frame_to_data_rate(cpi, size, dest, dest_end, frame_flags);
  cpi->twopass.bits_left -= 8 * (int)(*size);

  if (!cpi->common.refresh_alt_ref_frame) {
    double two_pass_min_rate =
        (double)(cpi->oxcf.target_bandwidth *
                 cpi->oxcf.two_pass_vbrmin_section / 100);
    cpi->twopass.bits_left += (int64_t)(two_pass_min_rate / cpi->framerate);
  }
}
#endif

int vp8_receive_raw_frame(VP8_COMP *cpi, unsigned int frame_flags,
                          YV12_BUFFER_CONFIG *sd, int64_t time_stamp,
                          int64_t end_time) {
#if CONFIG_INTERNAL_STATS
  struct vpx_usec_timer timer;
#endif
  int res = 0;

#if CONFIG_INTERNAL_STATS
  vpx_usec_timer_start(&timer);
#endif

  /* Reinit the lookahead buffer if the frame size changes */
  if (sd->y_width != cpi->oxcf.Width || sd->y_height != cpi->oxcf.Height) {
    assert(cpi->oxcf.lag_in_frames < 2);
    dealloc_raw_frame_buffers(cpi);
    alloc_raw_frame_buffers(cpi);
  }

  if (vp8_lookahead_push(cpi->lookahead, sd, time_stamp, end_time, frame_flags,
                         cpi->active_map_enabled ? cpi->active_map : NULL)) {
    res = -1;
  }
#if CONFIG_INTERNAL_STATS
  vpx_usec_timer_mark(&timer);
  cpi->time_receive_data += vpx_usec_timer_elapsed(&timer);
#endif

  return res;
}

static int frame_is_reference(const VP8_COMP *cpi) {
  const VP8_COMMON *cm = &cpi->common;
  const MACROBLOCKD *xd = &cpi->mb.e_mbd;

  return cm->frame_type == KEY_FRAME || cm->refresh_last_frame ||
         cm->refresh_golden_frame || cm->refresh_alt_ref_frame ||
         cm->copy_buffer_to_gf || cm->copy_buffer_to_arf ||
         cm->refresh_entropy_probs || xd->mode_ref_lf_delta_update ||
         xd->update_mb_segmentation_map || xd->update_mb_segmentation_data;
}

int vp8_get_compressed_data(VP8_COMP *cpi, unsigned int *frame_flags,
                            size_t *size, unsigned char *dest,
                            unsigned char *dest_end, int64_t *time_stamp,
                            int64_t *time_end, int flush) {
  VP8_COMMON *cm;
  struct vpx_usec_timer ticktimer;
#if CONFIG_INTERNAL_STATS
  struct vpx_usec_timer cmptimer;
#endif
  YV12_BUFFER_CONFIG *force_src_buffer = NULL;

  if (!cpi) return -1;

  cm = &cpi->common;

#if CONFIG_INTERNAL_STATS
  vpx_usec_timer_start(&cmptimer);
#endif

  cpi->source = NULL;

#if !CONFIG_REALTIME_ONLY
  /* Should we code an alternate reference frame */
  if (cpi->oxcf.error_resilient_mode == 0 && cpi->oxcf.play_alternate &&
      cpi->source_alt_ref_pending) {
    if ((cpi->source = vp8_lookahead_peek(
             cpi->lookahead, cpi->frames_till_gf_update_due, PEEK_FORWARD))) {
      cpi->alt_ref_source = cpi->source;
      if (cpi->oxcf.arnr_max_frames > 0) {
        vp8_temporal_filter_prepare_c(cpi, cpi->frames_till_gf_update_due);
        force_src_buffer = &cpi->alt_ref_buffer;
      }
      cpi->frames_till_alt_ref_frame = cpi->frames_till_gf_update_due;
      cm->refresh_alt_ref_frame = 1;
      cm->refresh_golden_frame = 0;
      cm->refresh_last_frame = 0;
      cm->show_frame = 0;
      /* Clear Pending alt Ref flag. */
      cpi->source_alt_ref_pending = 0;
      cpi->is_src_frame_alt_ref = 0;
    }
  }
#endif

  if (!cpi->source) {
    /* Read last frame source if we are encoding first pass. */
    if (cpi->pass == 1 && cm->current_video_frame > 0) {
      if ((cpi->last_source =
               vp8_lookahead_peek(cpi->lookahead, 1, PEEK_BACKWARD)) == NULL) {
        return -1;
      }
    }

    if ((cpi->source = vp8_lookahead_pop(cpi->lookahead, flush))) {
      cm->show_frame = 1;

      cpi->is_src_frame_alt_ref =
          cpi->alt_ref_source && (cpi->source == cpi->alt_ref_source);

      if (cpi->is_src_frame_alt_ref) cpi->alt_ref_source = NULL;
    }
  }

  if (cpi->source) {
    cpi->Source = force_src_buffer ? force_src_buffer : &cpi->source->img;
    cpi->un_scaled_source = cpi->Source;
    *time_stamp = cpi->source->ts_start;
    *time_end = cpi->source->ts_end;
    *frame_flags = cpi->source->flags;

    if (cpi->pass == 1 && cm->current_video_frame > 0) {
      cpi->last_frame_unscaled_source = &cpi->last_source->img;
    }
  } else {
    *size = 0;
#if !CONFIG_REALTIME_ONLY

    if (flush && cpi->pass == 1 && !cpi->twopass.first_pass_done) {
      vp8_end_first_pass(cpi); /* get last stats packet */
      cpi->twopass.first_pass_done = 1;
    }

#endif

    return -1;
  }

  if (cpi->source->ts_start < cpi->first_time_stamp_ever) {
    cpi->first_time_stamp_ever = cpi->source->ts_start;
    cpi->last_end_time_stamp_seen = cpi->source->ts_start;
  }

  /* adjust frame rates based on timestamps given */
  if (cm->show_frame) {
    int64_t this_duration;
    int step = 0;

    if (cpi->source->ts_start == cpi->first_time_stamp_ever) {
      this_duration = cpi->source->ts_end - cpi->source->ts_start;
      step = 1;
    } else {
      int64_t last_duration;

      this_duration = cpi->source->ts_end - cpi->last_end_time_stamp_seen;
      last_duration = cpi->last_end_time_stamp_seen - cpi->last_time_stamp_seen;
      // Cap this to avoid overflow of (this_duration - last_duration) * 10
      this_duration = VPXMIN(this_duration, INT64_MAX / 10);
      /* do a step update if the duration changes by 10% */
      if (last_duration) {
        step = (int)(((this_duration - last_duration) * 10 / last_duration));
      }
    }

    if (this_duration) {
      if (step) {
        cpi->ref_framerate = 10000000.0 / this_duration;
      } else {
        double avg_duration, interval;

        /* Average this frame's rate into the last second's average
         * frame rate. If we haven't seen 1 second yet, then average
         * over the whole interval seen.
         */
        interval = (double)(cpi->source->ts_end - cpi->first_time_stamp_ever);
        if (interval > 10000000.0) interval = 10000000;

        avg_duration = 10000000.0 / cpi->ref_framerate;
        avg_duration *= (interval - avg_duration + this_duration);
        avg_duration /= interval;

        cpi->ref_framerate = 10000000.0 / avg_duration;
      }
#if CONFIG_MULTI_RES_ENCODING
      if (cpi->oxcf.mr_total_resolutions > 1) {
        LOWER_RES_FRAME_INFO *low_res_frame_info =
            (LOWER_RES_FRAME_INFO *)cpi->oxcf.mr_low_res_mode_info;
        // Frame rate should be the same for all spatial layers in
        // multi-res-encoding (simulcast), so we constrain the frame for
        // higher layers to be that of lowest resolution. This is needed
        // as he application may decide to skip encoding a high layer and
        // then start again, in which case a big jump in time-stamps will
        // be received for that high layer, which will yield an incorrect
        // frame rate (from time-stamp adjustment in above calculation).
        if (cpi->oxcf.mr_encoder_id) {
          if (!low_res_frame_info->skip_encoding_base_stream)
            cpi->ref_framerate = low_res_frame_info->low_res_framerate;
        } else {
          // Keep track of frame rate for lowest resolution.
          low_res_frame_info->low_res_framerate = cpi->ref_framerate;
          // The base stream is being encoded so set skip flag to 0.
          low_res_frame_info->skip_encoding_base_stream = 0;
        }
      }
#endif
      cpi->ref_framerate = clamp_framerate(cpi->ref_framerate);
      if (cpi->oxcf.number_of_layers > 1) {
        unsigned int i;

        /* Update frame rates for each layer */
        assert(cpi->oxcf.number_of_layers <= VPX_TS_MAX_LAYERS);
        for (i = 0; i < cpi->oxcf.number_of_layers && i < VPX_TS_MAX_LAYERS;
             ++i) {
          LAYER_CONTEXT *lc = &cpi->layer_context[i];
          lc->framerate = cpi->ref_framerate / cpi->oxcf.rate_decimator[i];
        }
      } else {
        vp8_new_framerate(cpi, cpi->ref_framerate);
      }
    }

    cpi->last_time_stamp_seen = cpi->source->ts_start;
    cpi->last_end_time_stamp_seen = cpi->source->ts_end;
  }

  if (cpi->oxcf.number_of_layers > 1) {
    int layer;

    vp8_update_layer_contexts(cpi);

    /* Restore layer specific context & set frame rate */
    if (cpi->temporal_layer_id >= 0) {
      layer = cpi->temporal_layer_id;
    } else {
      layer =
          cpi->oxcf
              .layer_id[cpi->temporal_pattern_counter % cpi->oxcf.periodicity];
    }
    vp8_restore_layer_context(cpi, layer);
    vp8_new_framerate(cpi, cpi->layer_context[layer].framerate);
  }

  if (cpi->compressor_speed == 2) {
    vpx_usec_timer_start(&ticktimer);
  }

  cpi->lf_zeromv_pct = (cpi->zeromv_count * 100) / cm->MBs;

#if CONFIG_REALTIME_ONLY & CONFIG_ONTHEFLY_BITPACKING
  {
    int i;
    const int num_part = (1 << cm->multi_token_partition);
    /* the available bytes in dest */
    const unsigned long dest_size = dest_end - dest;
    const int tok_part_buff_size = (dest_size * 9) / (10 * num_part);

    unsigned char *dp = dest;

    cpi->partition_d[0] = dp;
    dp += dest_size / 10; /* reserve 1/10 for control partition */
    cpi->partition_d_end[0] = dp;

    for (i = 0; i < num_part; ++i) {
      cpi->partition_d[i + 1] = dp;
      dp += tok_part_buff_size;
      cpi->partition_d_end[i + 1] = dp;
    }
  }
#endif

  /* start with a 0 size frame */
  *size = 0;

  /* Clear down mmx registers */
  vpx_clear_system_state();

  cm->frame_type = INTER_FRAME;
  cm->frame_flags = *frame_flags;

#if 0

    if (cm->refresh_alt_ref_frame)
    {
        cm->refresh_golden_frame = 0;
        cm->refresh_last_frame = 0;
    }
    else
    {
        cm->refresh_golden_frame = 0;
        cm->refresh_last_frame = 1;
    }

#endif
  /* find a free buffer for the new frame */
  {
    int i = 0;
    for (; i < NUM_YV12_BUFFERS; ++i) {
      if (!cm->yv12_fb[i].flags) {
        cm->new_fb_idx = i;
        break;
      }
    }

    assert(i < NUM_YV12_BUFFERS);
  }
  switch (cpi->pass) {
#if !CONFIG_REALTIME_ONLY
    case 1: Pass1Encode(cpi); break;
    case 2: Pass2Encode(cpi, size, dest, dest_end, frame_flags); break;
#endif  // !CONFIG_REALTIME_ONLY
    default:
      encode_frame_to_data_rate(cpi, size, dest, dest_end, frame_flags);
      break;
  }

  if (cpi->compressor_speed == 2) {
    unsigned int duration, duration2;
    vpx_usec_timer_mark(&ticktimer);

    duration = (int)(vpx_usec_timer_elapsed(&ticktimer));
    duration2 = (unsigned int)((double)duration / 2);

    if (cm->frame_type != KEY_FRAME) {
      if (cpi->avg_encode_time == 0) {
        cpi->avg_encode_time = duration;
      } else {
        cpi->avg_encode_time = (7 * cpi->avg_encode_time + duration) >> 3;
      }
    }

    if (duration2) {
      {
        if (cpi->avg_pick_mode_time == 0) {
          cpi->avg_pick_mode_time = duration2;
        } else {
          cpi->avg_pick_mode_time =
              (7 * cpi->avg_pick_mode_time + duration2) >> 3;
        }
      }
    }
  }

  if (cm->refresh_entropy_probs == 0) {
    cm->fc = cm->lfc;
  }

  /* Save the contexts separately for alt ref, gold and last. */
  /* (TODO jbb -> Optimize this with pointers to avoid extra copies. ) */
  if (cm->refresh_alt_ref_frame) cpi->lfc_a = cm->fc;

  if (cm->refresh_golden_frame) cpi->lfc_g = cm->fc;

  if (cm->refresh_last_frame) cpi->lfc_n = cm->fc;

  /* if it's a dropped frame honor the requests on subsequent frames */
  if (*size > 0) {
    cpi->droppable = !frame_is_reference(cpi);

    /* return to normal state */
    cm->refresh_entropy_probs = 1;
    cm->refresh_alt_ref_frame = 0;
    cm->refresh_golden_frame = 0;
    cm->refresh_last_frame = 1;
    cm->frame_type = INTER_FRAME;
  }

  /* Save layer specific state */
  if (cpi->oxcf.number_of_layers > 1) vp8_save_layer_context(cpi);

#if CONFIG_INTERNAL_STATS
  vpx_usec_timer_mark(&cmptimer);
  cpi->time_compress_data += vpx_usec_timer_elapsed(&cmptimer);
#endif

#if CONFIG_MULTITHREAD
  /* wait for the lpf thread done */
  if (vpx_atomic_load_acquire(&cpi->b_multi_threaded) && cpi->b_lpf_running) {
    vp8_sem_wait(&cpi->h_event_end_lpf);
    cpi->b_lpf_running = 0;
  }
#endif

  if (cpi->b_calculate_psnr && cpi->pass != 1 && cm->show_frame) {
    generate_psnr_packet(cpi);
  }

#if CONFIG_INTERNAL_STATS

  if (cpi->pass != 1) {
    cpi->bytes += *size;

    if (cm->show_frame) {
      cpi->common.show_frame_mi = cpi->common.mi;
      cpi->count++;

      if (cpi->b_calculate_psnr) {
        uint64_t ye, ue, ve;
        double frame_psnr;
        YV12_BUFFER_CONFIG *orig = cpi->Source;
        YV12_BUFFER_CONFIG *recon = cpi->common.frame_to_show;
        unsigned int y_width = cpi->common.Width;
        unsigned int y_height = cpi->common.Height;
        unsigned int uv_width = (y_width + 1) / 2;
        unsigned int uv_height = (y_height + 1) / 2;
        int y_samples = y_height * y_width;
        int uv_samples = uv_height * uv_width;
        int t_samples = y_samples + 2 * uv_samples;
        double sq_error;

        ye = calc_plane_error(orig->y_buffer, orig->y_stride, recon->y_buffer,
                              recon->y_stride, y_width, y_height);

        ue = calc_plane_error(orig->u_buffer, orig->uv_stride, recon->u_buffer,
                              recon->uv_stride, uv_width, uv_height);

        ve = calc_plane_error(orig->v_buffer, orig->uv_stride, recon->v_buffer,
                              recon->uv_stride, uv_width, uv_height);

        sq_error = (double)(ye + ue + ve);

        frame_psnr = vpx_sse_to_psnr(t_samples, 255.0, sq_error);

        cpi->total_y += vpx_sse_to_psnr(y_samples, 255.0, (double)ye);
        cpi->total_u += vpx_sse_to_psnr(uv_samples, 255.0, (double)ue);
        cpi->total_v += vpx_sse_to_psnr(uv_samples, 255.0, (double)ve);
        cpi->total_sq_error += sq_error;
        cpi->total += frame_psnr;
#if CONFIG_POSTPROC
        {
          YV12_BUFFER_CONFIG *pp = &cm->post_proc_buffer;
          double sq_error2;
          double frame_psnr2, frame_ssim2 = 0;
          double weight = 0;

          vp8_deblock(cm, cm->frame_to_show, &cm->post_proc_buffer,
                      cm->filter_level * 10 / 6);
          vpx_clear_system_state();

          ye = calc_plane_error(orig->y_buffer, orig->y_stride, pp->y_buffer,
                                pp->y_stride, y_width, y_height);

          ue = calc_plane_error(orig->u_buffer, orig->uv_stride, pp->u_buffer,
                                pp->uv_stride, uv_width, uv_height);

          ve = calc_plane_error(orig->v_buffer, orig->uv_stride, pp->v_buffer,
                                pp->uv_stride, uv_width, uv_height);

          sq_error2 = (double)(ye + ue + ve);

          frame_psnr2 = vpx_sse_to_psnr(t_samples, 255.0, sq_error2);

          cpi->totalp_y += vpx_sse_to_psnr(y_samples, 255.0, (double)ye);
          cpi->totalp_u += vpx_sse_to_psnr(uv_samples, 255.0, (double)ue);
          cpi->totalp_v += vpx_sse_to_psnr(uv_samples, 255.0, (double)ve);
          cpi->total_sq_error2 += sq_error2;
          cpi->totalp += frame_psnr2;

          frame_ssim2 =
              vpx_calc_ssim(cpi->Source, &cm->post_proc_buffer, &weight);

          cpi->summed_quality += frame_ssim2 * weight;
          cpi->summed_weights += weight;

          if (cpi->oxcf.number_of_layers > 1) {
            unsigned int i;

            for (i = cpi->current_layer; i < cpi->oxcf.number_of_layers; ++i) {
              cpi->frames_in_layer[i]++;

              cpi->bytes_in_layer[i] += *size;
              cpi->sum_psnr[i] += frame_psnr;
              cpi->sum_psnr_p[i] += frame_psnr2;
              cpi->total_error2[i] += sq_error;
              cpi->total_error2_p[i] += sq_error2;
              cpi->sum_ssim[i] += frame_ssim2 * weight;
              cpi->sum_weights[i] += weight;
            }
          }
        }
#endif
      }
    }
  }

#if 0

    if (cpi->common.frame_type != 0 && cpi->common.base_qindex == cpi->oxcf.worst_allowed_q)
    {
        skiptruecount += cpi->skip_true_count;
        skipfalsecount += cpi->skip_false_count;
    }

#endif
#if 0

    if (cpi->pass != 1)
    {
        FILE *f = fopen("skip.stt", "a");
        fprintf(f, "frame:%4d flags:%4x Q:%4d P:%4d Size:%5d\n", cpi->common.current_video_frame, *frame_flags, cpi->common.base_qindex, cpi->prob_skip_false, *size);

        if (cpi->is_src_frame_alt_ref == 1)
            fprintf(f, "skipcount: %4d framesize: %d\n", cpi->skip_true_count , *size);

        fclose(f);
    }

#endif
#endif

  return 0;
}

int vp8_get_preview_raw_frame(VP8_COMP *cpi, YV12_BUFFER_CONFIG *dest,
                              vp8_ppflags_t *flags) {
  if (cpi->common.refresh_alt_ref_frame) {
    return -1;
  } else {
    int ret;

#if CONFIG_POSTPROC
    cpi->common.show_frame_mi = cpi->common.mi;
    ret = vp8_post_proc_frame(&cpi->common, dest, flags);
#else
    (void)flags;

    if (cpi->common.frame_to_show) {
      *dest = *cpi->common.frame_to_show;
      dest->y_width = cpi->common.Width;
      dest->y_height = cpi->common.Height;
      dest->uv_height = cpi->common.Height / 2;
      ret = 0;
    } else {
      ret = -1;
    }

#endif
    vpx_clear_system_state();
    return ret;
  }
}

int vp8_set_roimap(VP8_COMP *cpi, unsigned char *map, unsigned int rows,
                   unsigned int cols, int delta_q[4], int delta_lf[4],
                   unsigned int threshold[4]) {
  signed char feature_data[MB_LVL_MAX][MAX_MB_SEGMENTS];
  int internal_delta_q[MAX_MB_SEGMENTS];
  const int range = 63;
  int i;

  // Check number of rows and columns match
  if (cpi->common.mb_rows != (int)rows || cpi->common.mb_cols != (int)cols) {
    return -1;
  }

  for (i = 0; i < MAX_MB_SEGMENTS; ++i) {
    // Note abs() alone can't be used as the behavior of abs(INT_MIN) is
    // undefined.
    if (delta_q[i] > range || delta_q[i] < -range || delta_lf[i] > range ||
        delta_lf[i] < -range) {
      return -1;
    }
  }

  // Also disable segmentation if no deltas are specified.
  if (!map || (delta_q[0] == 0 && delta_q[1] == 0 && delta_q[2] == 0 &&
               delta_q[3] == 0 && delta_lf[0] == 0 && delta_lf[1] == 0 &&
               delta_lf[2] == 0 && delta_lf[3] == 0 && threshold[0] == 0 &&
               threshold[1] == 0 && threshold[2] == 0 && threshold[3] == 0)) {
    disable_segmentation(cpi);
    return 0;
  }

  // Translate the external delta q values to internal values.
  for (i = 0; i < MAX_MB_SEGMENTS; ++i) {
    internal_delta_q[i] =
        (delta_q[i] >= 0) ? q_trans[delta_q[i]] : -q_trans[-delta_q[i]];
  }

  /* Set the segmentation Map */
  set_segmentation_map(cpi, map);

  /* Activate segmentation. */
  enable_segmentation(cpi);

  /* Set up the quant segment data */
  feature_data[MB_LVL_ALT_Q][0] = internal_delta_q[0];
  feature_data[MB_LVL_ALT_Q][1] = internal_delta_q[1];
  feature_data[MB_LVL_ALT_Q][2] = internal_delta_q[2];
  feature_data[MB_LVL_ALT_Q][3] = internal_delta_q[3];

  /* Set up the loop segment data s */
  feature_data[MB_LVL_ALT_LF][0] = delta_lf[0];
  feature_data[MB_LVL_ALT_LF][1] = delta_lf[1];
  feature_data[MB_LVL_ALT_LF][2] = delta_lf[2];
  feature_data[MB_LVL_ALT_LF][3] = delta_lf[3];

  cpi->segment_encode_breakout[0] = threshold[0];
  cpi->segment_encode_breakout[1] = threshold[1];
  cpi->segment_encode_breakout[2] = threshold[2];
  cpi->segment_encode_breakout[3] = threshold[3];

  /* Initialise the feature data structure */
  set_segment_data(cpi, &feature_data[0][0], SEGMENT_DELTADATA);

  if (threshold[0] != 0 || threshold[1] != 0 || threshold[2] != 0 ||
      threshold[3] != 0)
    cpi->use_roi_static_threshold = 1;
  cpi->cyclic_refresh_mode_enabled = 0;

  return 0;
}

int vp8_set_active_map(VP8_COMP *cpi, unsigned char *map, unsigned int rows,
                       unsigned int cols) {
  if ((int)rows == cpi->common.mb_rows && (int)cols == cpi->common.mb_cols) {
    if (map) {
      memcpy(cpi->active_map, map, rows * cols);
      cpi->active_map_enabled = 1;
    } else {
      cpi->active_map_enabled = 0;
    }

    return 0;
  } else {
    return -1;
  }
}

int vp8_set_internal_size(VP8_COMP *cpi, VPX_SCALING_MODE horiz_mode,
                          VPX_SCALING_MODE vert_mode) {
  if (horiz_mode <= VP8E_ONETWO) {
    cpi->common.horiz_scale = horiz_mode;
  } else {
    return -1;
  }

  if (vert_mode <= VP8E_ONETWO) {
    cpi->common.vert_scale = vert_mode;
  } else {
    return -1;
  }

  return 0;
}

int vp8_calc_ss_err(YV12_BUFFER_CONFIG *source, YV12_BUFFER_CONFIG *dest) {
  int i, j;
  int Total = 0;

  unsigned char *src = source->y_buffer;
  unsigned char *dst = dest->y_buffer;

  /* Loop through the Y plane raw and reconstruction data summing
   * (square differences)
   */
  for (i = 0; i < source->y_height; i += 16) {
    for (j = 0; j < source->y_width; j += 16) {
      unsigned int sse;
      Total += vpx_mse16x16(src + j, source->y_stride, dst + j, dest->y_stride,
                            &sse);
    }

    src += 16 * source->y_stride;
    dst += 16 * dest->y_stride;
  }

  return Total;
}

int vp8_get_quantizer(VP8_COMP *cpi) { return cpi->common.base_qindex; }
