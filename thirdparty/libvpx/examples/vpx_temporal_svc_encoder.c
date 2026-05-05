/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

//  This is an example demonstrating how to implement a multi-layer VPx
//  encoding scheme based on temporal scalability for video applications
//  that benefit from a scalable bitstream.

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "./vpx_config.h"
#include "./y4minput.h"
#include "../vpx_ports/vpx_timer.h"
#include "vpx/vp8cx.h"
#include "vpx/vpx_encoder.h"
#include "vpx_ports/bitops.h"

#include "../tools_common.h"
#include "../video_writer.h"

#define ROI_MAP 0

#define zero(Dest) memset(&(Dest), 0, sizeof(Dest))

static const char *exec_name;

void usage_exit(void) { exit(EXIT_FAILURE); }

// Denoiser states for vp8, for temporal denoising.
enum denoiserStateVp8 {
  kVp8DenoiserOff,
  kVp8DenoiserOnYOnly,
  kVp8DenoiserOnYUV,
  kVp8DenoiserOnYUVAggressive,
  kVp8DenoiserOnAdaptive
};

// Denoiser states for vp9, for temporal denoising.
enum denoiserStateVp9 {
  kVp9DenoiserOff,
  kVp9DenoiserOnYOnly,
  // For SVC: denoise the top two spatial layers.
  kVp9DenoiserOnYTwoSpatialLayers
};

static int mode_to_num_layers[13] = { 1, 2, 2, 3, 3, 3, 3, 5, 2, 3, 3, 3, 3 };

// For rate control encoding stats.
struct RateControlMetrics {
  // Number of input frames per layer.
  int layer_input_frames[VPX_TS_MAX_LAYERS];
  // Total (cumulative) number of encoded frames per layer.
  int layer_tot_enc_frames[VPX_TS_MAX_LAYERS];
  // Number of encoded non-key frames per layer.
  int layer_enc_frames[VPX_TS_MAX_LAYERS];
  // Framerate per layer layer (cumulative).
  double layer_framerate[VPX_TS_MAX_LAYERS];
  // Target average frame size per layer (per-frame-bandwidth per layer).
  double layer_pfb[VPX_TS_MAX_LAYERS];
  // Actual average frame size per layer.
  double layer_avg_frame_size[VPX_TS_MAX_LAYERS];
  // Average rate mismatch per layer (|target - actual| / target).
  double layer_avg_rate_mismatch[VPX_TS_MAX_LAYERS];
  // Actual encoding bitrate per layer (cumulative).
  double layer_encoding_bitrate[VPX_TS_MAX_LAYERS];
  // Average of the short-time encoder actual bitrate.
  // TODO(marpan): Should we add these short-time stats for each layer?
  double avg_st_encoding_bitrate;
  // Variance of the short-time encoder actual bitrate.
  double variance_st_encoding_bitrate;
  // Window (number of frames) for computing short-timee encoding bitrate.
  int window_size;
  // Number of window measurements.
  int window_count;
  int layer_target_bitrate[VPX_MAX_LAYERS];
};

// Note: these rate control metrics assume only 1 key frame in the
// sequence (i.e., first frame only). So for temporal pattern# 7
// (which has key frame for every frame on base layer), the metrics
// computation will be off/wrong.
// TODO(marpan): Update these metrics to account for multiple key frames
// in the stream.
static void set_rate_control_metrics(struct RateControlMetrics *rc,
                                     vpx_codec_enc_cfg_t *cfg) {
  int i = 0;
  // Set the layer (cumulative) framerate and the target layer (non-cumulative)
  // per-frame-bandwidth, for the rate control encoding stats below.
  const double framerate = cfg->g_timebase.den / cfg->g_timebase.num;
  const int ts_number_layers = cfg->ts_number_layers;
  rc->layer_framerate[0] = framerate / cfg->ts_rate_decimator[0];
  rc->layer_pfb[0] =
      1000.0 * rc->layer_target_bitrate[0] / rc->layer_framerate[0];
  for (i = 0; i < ts_number_layers; ++i) {
    if (i > 0) {
      rc->layer_framerate[i] = framerate / cfg->ts_rate_decimator[i];
      rc->layer_pfb[i] =
          1000.0 *
          (rc->layer_target_bitrate[i] - rc->layer_target_bitrate[i - 1]) /
          (rc->layer_framerate[i] - rc->layer_framerate[i - 1]);
    }
    rc->layer_input_frames[i] = 0;
    rc->layer_enc_frames[i] = 0;
    rc->layer_tot_enc_frames[i] = 0;
    rc->layer_encoding_bitrate[i] = 0.0;
    rc->layer_avg_frame_size[i] = 0.0;
    rc->layer_avg_rate_mismatch[i] = 0.0;
  }
  rc->window_count = 0;
  rc->window_size = 15;
  rc->avg_st_encoding_bitrate = 0.0;
  rc->variance_st_encoding_bitrate = 0.0;
  // Target bandwidth for the whole stream.
  // Set to layer_target_bitrate for highest layer (total bitrate).
  cfg->rc_target_bitrate = rc->layer_target_bitrate[ts_number_layers - 1];
}

static void printout_rate_control_summary(struct RateControlMetrics *rc,
                                          vpx_codec_enc_cfg_t *cfg,
                                          int frame_cnt) {
  unsigned int i = 0;
  int tot_num_frames = 0;
  double perc_fluctuation = 0.0;
  printf("Total number of processed frames: %d\n\n", frame_cnt - 1);
  printf("Rate control layer stats for %d layer(s):\n\n",
         cfg->ts_number_layers);
  for (i = 0; i < cfg->ts_number_layers; ++i) {
    const int num_dropped =
        (i > 0) ? (rc->layer_input_frames[i] - rc->layer_enc_frames[i])
                : (rc->layer_input_frames[i] - rc->layer_enc_frames[i] - 1);
    tot_num_frames += rc->layer_input_frames[i];
    rc->layer_encoding_bitrate[i] = 0.001 * rc->layer_framerate[i] *
                                    rc->layer_encoding_bitrate[i] /
                                    tot_num_frames;
    rc->layer_avg_frame_size[i] =
        rc->layer_avg_frame_size[i] / rc->layer_enc_frames[i];
    rc->layer_avg_rate_mismatch[i] =
        100.0 * rc->layer_avg_rate_mismatch[i] / rc->layer_enc_frames[i];
    printf("For layer#: %d \n", i);
    printf("Bitrate (target vs actual): %d %f \n", rc->layer_target_bitrate[i],
           rc->layer_encoding_bitrate[i]);
    printf("Average frame size (target vs actual): %f %f \n", rc->layer_pfb[i],
           rc->layer_avg_frame_size[i]);
    printf("Average rate_mismatch: %f \n", rc->layer_avg_rate_mismatch[i]);
    printf(
        "Number of input frames, encoded (non-key) frames, "
        "and perc dropped frames: %d %d %f \n",
        rc->layer_input_frames[i], rc->layer_enc_frames[i],
        100.0 * num_dropped / rc->layer_input_frames[i]);
    printf("\n");
  }
  rc->avg_st_encoding_bitrate = rc->avg_st_encoding_bitrate / rc->window_count;
  rc->variance_st_encoding_bitrate =
      rc->variance_st_encoding_bitrate / rc->window_count -
      (rc->avg_st_encoding_bitrate * rc->avg_st_encoding_bitrate);
  perc_fluctuation = 100.0 * sqrt(rc->variance_st_encoding_bitrate) /
                     rc->avg_st_encoding_bitrate;
  printf("Short-time stats, for window of %d frames: \n", rc->window_size);
  printf("Average, rms-variance, and percent-fluct: %f %f %f \n",
         rc->avg_st_encoding_bitrate, sqrt(rc->variance_st_encoding_bitrate),
         perc_fluctuation);
  if ((frame_cnt - 1) != tot_num_frames)
    die("Error: Number of input frames not equal to output! \n");
}

#if ROI_MAP
static void set_roi_map(const char *enc_name, vpx_codec_enc_cfg_t *cfg,
                        vpx_roi_map_t *roi) {
  unsigned int i, j;
  int block_size = 0;
  uint8_t is_vp8 = strncmp(enc_name, "vp8", 3) == 0 ? 1 : 0;
  uint8_t is_vp9 = strncmp(enc_name, "vp9", 3) == 0 ? 1 : 0;
  if (!is_vp8 && !is_vp9) {
    die("unsupported codec.");
  }
  zero(*roi);

  block_size = is_vp9 && !is_vp8 ? 8 : 16;

  // ROI is based on the segments (4 for vp8, 8 for vp9), smallest unit for
  // segment is 16x16 for vp8, 8x8 for vp9.
  roi->rows = (cfg->g_h + block_size - 1) / block_size;
  roi->cols = (cfg->g_w + block_size - 1) / block_size;

  // Applies delta QP on the segment blocks, varies from -63 to 63.
  // Setting to negative means lower QP (better quality).
  // Below we set delta_q to the extreme (-63) to show strong effect.
  // VP8 uses the first 4 segments. VP9 uses all 8 segments.
  zero(roi->delta_q);
  roi->delta_q[1] = -63;

  // Applies delta loopfilter strength on the segment blocks, varies from -63 to
  // 63. Setting to positive means stronger loopfilter. VP8 uses the first 4
  // segments. VP9 uses all 8 segments.
  zero(roi->delta_lf);

  if (is_vp8) {
    // Applies skip encoding threshold on the segment blocks, varies from 0 to
    // UINT_MAX. Larger value means more skipping of encoding is possible.
    // This skip threshold only applies on delta frames.
    zero(roi->static_threshold);
  }

  if (is_vp9) {
    // Apply skip segment. Setting to 1 means this block will be copied from
    // previous frame.
    zero(roi->skip);
  }

  if (is_vp9) {
    // Apply ref frame segment.
    // -1 : Do not apply this segment.
    //  0 : Froce using intra.
    //  1 : Force using last.
    //  2 : Force using golden.
    //  3 : Force using alfref but not used in non-rd pickmode for 0 lag.
    memset(roi->ref_frame, -1, sizeof(roi->ref_frame));
    roi->ref_frame[1] = 1;
  }

  // Use 2 states: 1 is center square, 0 is the rest.
  roi->roi_map =
      (uint8_t *)calloc(roi->rows * roi->cols, sizeof(*roi->roi_map));
  for (i = 0; i < roi->rows; ++i) {
    for (j = 0; j < roi->cols; ++j) {
      if (i > (roi->rows >> 2) && i < ((roi->rows * 3) >> 2) &&
          j > (roi->cols >> 2) && j < ((roi->cols * 3) >> 2)) {
        roi->roi_map[i * roi->cols + j] = 1;
      }
    }
  }
}

static void set_roi_skip_map(vpx_codec_enc_cfg_t *cfg, vpx_roi_map_t *roi,
                             int *skip_map, int *prev_mask_map, int frame_num) {
  const int block_size = 8;
  unsigned int i, j;
  roi->rows = (cfg->g_h + block_size - 1) / block_size;
  roi->cols = (cfg->g_w + block_size - 1) / block_size;
  zero(roi->skip);
  zero(roi->delta_q);
  zero(roi->delta_lf);
  memset(roi->ref_frame, -1, sizeof(roi->ref_frame));
  roi->ref_frame[1] = 1;
  // Use segment 3 for skip.
  roi->skip[3] = 1;
  roi->roi_map =
      (uint8_t *)calloc(roi->rows * roi->cols, sizeof(*roi->roi_map));
  for (i = 0; i < roi->rows; ++i) {
    for (j = 0; j < roi->cols; ++j) {
      const int idx = i * roi->cols + j;
      // Use segment 3 for skip.
      // prev_mask_map keeps track of blocks that have been stably on segment 3
      // for the past 10 frames. Only skip when the block is on segment 3 in
      // both current map and prev_mask_map.
      if (skip_map[idx] == 1 && prev_mask_map[idx] == 1) roi->roi_map[idx] = 3;
      // Reset it every 10 frames so it doesn't propagate for too many frames.
      if (frame_num % 10 == 0)
        prev_mask_map[idx] = skip_map[idx];
      else if (prev_mask_map[idx] == 1 && skip_map[idx] == 0)
        prev_mask_map[idx] = 0;
    }
  }
}
#endif

// Temporal scaling parameters:
// NOTE: The 3 prediction frames cannot be used interchangeably due to
// differences in the way they are handled throughout the code. The
// frames should be allocated to layers in the order LAST, GF, ARF.
// Other combinations work, but may produce slightly inferior results.
static void set_temporal_layer_pattern(int layering_mode,
                                       vpx_codec_enc_cfg_t *cfg,
                                       int *layer_flags,
                                       int *flag_periodicity) {
  switch (layering_mode) {
    case 0: {
      // 1-layer.
      int ids[1] = { 0 };
      cfg->ts_periodicity = 1;
      *flag_periodicity = 1;
      cfg->ts_number_layers = 1;
      cfg->ts_rate_decimator[0] = 1;
      memcpy(cfg->ts_layer_id, ids, sizeof(ids));
      // Update L only.
      layer_flags[0] =
          VPX_EFLAG_FORCE_KF | VP8_EFLAG_NO_UPD_GF | VP8_EFLAG_NO_UPD_ARF;
      break;
    }
    case 1: {
      // 2-layers, 2-frame period.
      int ids[2] = { 0, 1 };
      cfg->ts_periodicity = 2;
      *flag_periodicity = 2;
      cfg->ts_number_layers = 2;
      cfg->ts_rate_decimator[0] = 2;
      cfg->ts_rate_decimator[1] = 1;
      memcpy(cfg->ts_layer_id, ids, sizeof(ids));
#if 1
      // 0=L, 1=GF, Intra-layer prediction enabled.
      layer_flags[0] = VPX_EFLAG_FORCE_KF | VP8_EFLAG_NO_UPD_GF |
                       VP8_EFLAG_NO_UPD_ARF | VP8_EFLAG_NO_REF_GF |
                       VP8_EFLAG_NO_REF_ARF;
      layer_flags[1] =
          VP8_EFLAG_NO_UPD_ARF | VP8_EFLAG_NO_UPD_LAST | VP8_EFLAG_NO_REF_ARF;
#else
      // 0=L, 1=GF, Intra-layer prediction disabled.
      layer_flags[0] = VPX_EFLAG_FORCE_KF | VP8_EFLAG_NO_UPD_GF |
                       VP8_EFLAG_NO_UPD_ARF | VP8_EFLAG_NO_REF_GF |
                       VP8_EFLAG_NO_REF_ARF;
      layer_flags[1] = VP8_EFLAG_NO_UPD_ARF | VP8_EFLAG_NO_UPD_LAST |
                       VP8_EFLAG_NO_REF_ARF | VP8_EFLAG_NO_REF_LAST;
#endif
      break;
    }
    case 2: {
      // 2-layers, 3-frame period.
      int ids[3] = { 0, 1, 1 };
      cfg->ts_periodicity = 3;
      *flag_periodicity = 3;
      cfg->ts_number_layers = 2;
      cfg->ts_rate_decimator[0] = 3;
      cfg->ts_rate_decimator[1] = 1;
      memcpy(cfg->ts_layer_id, ids, sizeof(ids));
      // 0=L, 1=GF, Intra-layer prediction enabled.
      layer_flags[0] = VPX_EFLAG_FORCE_KF | VP8_EFLAG_NO_REF_GF |
                       VP8_EFLAG_NO_REF_ARF | VP8_EFLAG_NO_UPD_GF |
                       VP8_EFLAG_NO_UPD_ARF;
      layer_flags[1] = layer_flags[2] =
          VP8_EFLAG_NO_REF_GF | VP8_EFLAG_NO_REF_ARF | VP8_EFLAG_NO_UPD_ARF |
          VP8_EFLAG_NO_UPD_LAST;
      break;
    }
    case 3: {
      // 3-layers, 6-frame period.
      int ids[6] = { 0, 2, 2, 1, 2, 2 };
      cfg->ts_periodicity = 6;
      *flag_periodicity = 6;
      cfg->ts_number_layers = 3;
      cfg->ts_rate_decimator[0] = 6;
      cfg->ts_rate_decimator[1] = 3;
      cfg->ts_rate_decimator[2] = 1;
      memcpy(cfg->ts_layer_id, ids, sizeof(ids));
      // 0=L, 1=GF, 2=ARF, Intra-layer prediction enabled.
      layer_flags[0] = VPX_EFLAG_FORCE_KF | VP8_EFLAG_NO_REF_GF |
                       VP8_EFLAG_NO_REF_ARF | VP8_EFLAG_NO_UPD_GF |
                       VP8_EFLAG_NO_UPD_ARF;
      layer_flags[3] =
          VP8_EFLAG_NO_REF_ARF | VP8_EFLAG_NO_UPD_ARF | VP8_EFLAG_NO_UPD_LAST;
      layer_flags[1] = layer_flags[2] = layer_flags[4] = layer_flags[5] =
          VP8_EFLAG_NO_UPD_GF | VP8_EFLAG_NO_UPD_LAST;
      break;
    }
    case 4: {
      // 3-layers, 4-frame period.
      int ids[4] = { 0, 2, 1, 2 };
      cfg->ts_periodicity = 4;
      *flag_periodicity = 4;
      cfg->ts_number_layers = 3;
      cfg->ts_rate_decimator[0] = 4;
      cfg->ts_rate_decimator[1] = 2;
      cfg->ts_rate_decimator[2] = 1;
      memcpy(cfg->ts_layer_id, ids, sizeof(ids));
      // 0=L, 1=GF, 2=ARF, Intra-layer prediction disabled.
      layer_flags[0] = VPX_EFLAG_FORCE_KF | VP8_EFLAG_NO_REF_GF |
                       VP8_EFLAG_NO_REF_ARF | VP8_EFLAG_NO_UPD_GF |
                       VP8_EFLAG_NO_UPD_ARF;
      layer_flags[2] = VP8_EFLAG_NO_REF_GF | VP8_EFLAG_NO_REF_ARF |
                       VP8_EFLAG_NO_UPD_ARF | VP8_EFLAG_NO_UPD_LAST;
      layer_flags[1] = layer_flags[3] =
          VP8_EFLAG_NO_REF_ARF | VP8_EFLAG_NO_UPD_LAST | VP8_EFLAG_NO_UPD_GF |
          VP8_EFLAG_NO_UPD_ARF;
      break;
    }
    case 5: {
      // 3-layers, 4-frame period.
      int ids[4] = { 0, 2, 1, 2 };
      cfg->ts_periodicity = 4;
      *flag_periodicity = 4;
      cfg->ts_number_layers = 3;
      cfg->ts_rate_decimator[0] = 4;
      cfg->ts_rate_decimator[1] = 2;
      cfg->ts_rate_decimator[2] = 1;
      memcpy(cfg->ts_layer_id, ids, sizeof(ids));
      // 0=L, 1=GF, 2=ARF, Intra-layer prediction enabled in layer 1, disabled
      // in layer 2.
      layer_flags[0] = VPX_EFLAG_FORCE_KF | VP8_EFLAG_NO_REF_GF |
                       VP8_EFLAG_NO_REF_ARF | VP8_EFLAG_NO_UPD_GF |
                       VP8_EFLAG_NO_UPD_ARF;
      layer_flags[2] =
          VP8_EFLAG_NO_REF_ARF | VP8_EFLAG_NO_UPD_LAST | VP8_EFLAG_NO_UPD_ARF;
      layer_flags[1] = layer_flags[3] =
          VP8_EFLAG_NO_REF_ARF | VP8_EFLAG_NO_UPD_LAST | VP8_EFLAG_NO_UPD_GF |
          VP8_EFLAG_NO_UPD_ARF;
      break;
    }
    case 6: {
      // 3-layers, 4-frame period.
      int ids[4] = { 0, 2, 1, 2 };
      cfg->ts_periodicity = 4;
      *flag_periodicity = 4;
      cfg->ts_number_layers = 3;
      cfg->ts_rate_decimator[0] = 4;
      cfg->ts_rate_decimator[1] = 2;
      cfg->ts_rate_decimator[2] = 1;
      memcpy(cfg->ts_layer_id, ids, sizeof(ids));
      // 0=L, 1=GF, 2=ARF, Intra-layer prediction enabled.
      layer_flags[0] = VPX_EFLAG_FORCE_KF | VP8_EFLAG_NO_REF_GF |
                       VP8_EFLAG_NO_REF_ARF | VP8_EFLAG_NO_UPD_GF |
                       VP8_EFLAG_NO_UPD_ARF;
      layer_flags[2] =
          VP8_EFLAG_NO_REF_ARF | VP8_EFLAG_NO_UPD_LAST | VP8_EFLAG_NO_UPD_ARF;
      layer_flags[1] = layer_flags[3] =
          VP8_EFLAG_NO_UPD_LAST | VP8_EFLAG_NO_UPD_GF;
      break;
    }
    case 7: {
      // NOTE: Probably of academic interest only.
      // 5-layers, 16-frame period.
      int ids[16] = { 0, 4, 3, 4, 2, 4, 3, 4, 1, 4, 3, 4, 2, 4, 3, 4 };
      cfg->ts_periodicity = 16;
      *flag_periodicity = 16;
      cfg->ts_number_layers = 5;
      cfg->ts_rate_decimator[0] = 16;
      cfg->ts_rate_decimator[1] = 8;
      cfg->ts_rate_decimator[2] = 4;
      cfg->ts_rate_decimator[3] = 2;
      cfg->ts_rate_decimator[4] = 1;
      memcpy(cfg->ts_layer_id, ids, sizeof(ids));
      layer_flags[0] = VPX_EFLAG_FORCE_KF;
      layer_flags[1] = layer_flags[3] = layer_flags[5] = layer_flags[7] =
          layer_flags[9] = layer_flags[11] = layer_flags[13] = layer_flags[15] =
              VP8_EFLAG_NO_UPD_LAST | VP8_EFLAG_NO_UPD_GF |
              VP8_EFLAG_NO_UPD_ARF;
      layer_flags[2] = layer_flags[6] = layer_flags[10] = layer_flags[14] =
          VP8_EFLAG_NO_UPD_ARF | VP8_EFLAG_NO_UPD_GF;
      layer_flags[4] = layer_flags[12] =
          VP8_EFLAG_NO_REF_LAST | VP8_EFLAG_NO_UPD_ARF;
      layer_flags[8] = VP8_EFLAG_NO_REF_LAST | VP8_EFLAG_NO_REF_GF;
      break;
    }
    case 8: {
      // 2-layers, with sync point at first frame of layer 1.
      int ids[2] = { 0, 1 };
      cfg->ts_periodicity = 2;
      *flag_periodicity = 8;
      cfg->ts_number_layers = 2;
      cfg->ts_rate_decimator[0] = 2;
      cfg->ts_rate_decimator[1] = 1;
      memcpy(cfg->ts_layer_id, ids, sizeof(ids));
      // 0=L, 1=GF.
      // ARF is used as predictor for all frames, and is only updated on
      // key frame. Sync point every 8 frames.

      // Layer 0: predict from L and ARF, update L and G.
      layer_flags[0] =
          VPX_EFLAG_FORCE_KF | VP8_EFLAG_NO_REF_GF | VP8_EFLAG_NO_UPD_ARF;
      // Layer 1: sync point: predict from L and ARF, and update G.
      layer_flags[1] =
          VP8_EFLAG_NO_REF_GF | VP8_EFLAG_NO_UPD_LAST | VP8_EFLAG_NO_UPD_ARF;
      // Layer 0, predict from L and ARF, update L.
      layer_flags[2] =
          VP8_EFLAG_NO_REF_GF | VP8_EFLAG_NO_UPD_GF | VP8_EFLAG_NO_UPD_ARF;
      // Layer 1: predict from L, G and ARF, and update G.
      layer_flags[3] = VP8_EFLAG_NO_UPD_ARF | VP8_EFLAG_NO_UPD_LAST |
                       VP8_EFLAG_NO_UPD_ENTROPY;
      // Layer 0.
      layer_flags[4] = layer_flags[2];
      // Layer 1.
      layer_flags[5] = layer_flags[3];
      // Layer 0.
      layer_flags[6] = layer_flags[4];
      // Layer 1.
      layer_flags[7] = layer_flags[5];
      break;
    }
    case 9: {
      // 3-layers: Sync points for layer 1 and 2 every 8 frames.
      int ids[4] = { 0, 2, 1, 2 };
      cfg->ts_periodicity = 4;
      *flag_periodicity = 8;
      cfg->ts_number_layers = 3;
      cfg->ts_rate_decimator[0] = 4;
      cfg->ts_rate_decimator[1] = 2;
      cfg->ts_rate_decimator[2] = 1;
      memcpy(cfg->ts_layer_id, ids, sizeof(ids));
      // 0=L, 1=GF, 2=ARF.
      layer_flags[0] = VPX_EFLAG_FORCE_KF | VP8_EFLAG_NO_REF_GF |
                       VP8_EFLAG_NO_REF_ARF | VP8_EFLAG_NO_UPD_GF |
                       VP8_EFLAG_NO_UPD_ARF;
      layer_flags[1] = VP8_EFLAG_NO_REF_GF | VP8_EFLAG_NO_REF_ARF |
                       VP8_EFLAG_NO_UPD_LAST | VP8_EFLAG_NO_UPD_GF;
      layer_flags[2] = VP8_EFLAG_NO_REF_GF | VP8_EFLAG_NO_REF_ARF |
                       VP8_EFLAG_NO_UPD_LAST | VP8_EFLAG_NO_UPD_ARF;
      layer_flags[3] = layer_flags[5] =
          VP8_EFLAG_NO_UPD_LAST | VP8_EFLAG_NO_UPD_GF;
      layer_flags[4] = VP8_EFLAG_NO_REF_GF | VP8_EFLAG_NO_REF_ARF |
                       VP8_EFLAG_NO_UPD_GF | VP8_EFLAG_NO_UPD_ARF;
      layer_flags[6] =
          VP8_EFLAG_NO_REF_ARF | VP8_EFLAG_NO_UPD_LAST | VP8_EFLAG_NO_UPD_ARF;
      layer_flags[7] = VP8_EFLAG_NO_UPD_LAST | VP8_EFLAG_NO_UPD_GF |
                       VP8_EFLAG_NO_UPD_ARF | VP8_EFLAG_NO_UPD_ENTROPY;
      break;
    }
    case 10: {
      // 3-layers structure where ARF is used as predictor for all frames,
      // and is only updated on key frame.
      // Sync points for layer 1 and 2 every 8 frames.

      int ids[4] = { 0, 2, 1, 2 };
      cfg->ts_periodicity = 4;
      *flag_periodicity = 8;
      cfg->ts_number_layers = 3;
      cfg->ts_rate_decimator[0] = 4;
      cfg->ts_rate_decimator[1] = 2;
      cfg->ts_rate_decimator[2] = 1;
      memcpy(cfg->ts_layer_id, ids, sizeof(ids));
      // 0=L, 1=GF, 2=ARF.
      // Layer 0: predict from L and ARF; update L and G.
      layer_flags[0] =
          VPX_EFLAG_FORCE_KF | VP8_EFLAG_NO_UPD_ARF | VP8_EFLAG_NO_REF_GF;
      // Layer 2: sync point: predict from L and ARF; update none.
      layer_flags[1] = VP8_EFLAG_NO_REF_GF | VP8_EFLAG_NO_UPD_GF |
                       VP8_EFLAG_NO_UPD_ARF | VP8_EFLAG_NO_UPD_LAST |
                       VP8_EFLAG_NO_UPD_ENTROPY;
      // Layer 1: sync point: predict from L and ARF; update G.
      layer_flags[2] =
          VP8_EFLAG_NO_REF_GF | VP8_EFLAG_NO_UPD_ARF | VP8_EFLAG_NO_UPD_LAST;
      // Layer 2: predict from L, G, ARF; update none.
      layer_flags[3] = VP8_EFLAG_NO_UPD_GF | VP8_EFLAG_NO_UPD_ARF |
                       VP8_EFLAG_NO_UPD_LAST | VP8_EFLAG_NO_UPD_ENTROPY;
      // Layer 0: predict from L and ARF; update L.
      layer_flags[4] =
          VP8_EFLAG_NO_UPD_GF | VP8_EFLAG_NO_UPD_ARF | VP8_EFLAG_NO_REF_GF;
      // Layer 2: predict from L, G, ARF; update none.
      layer_flags[5] = layer_flags[3];
      // Layer 1: predict from L, G, ARF; update G.
      layer_flags[6] = VP8_EFLAG_NO_UPD_ARF | VP8_EFLAG_NO_UPD_LAST;
      // Layer 2: predict from L, G, ARF; update none.
      layer_flags[7] = layer_flags[3];
      break;
    }
    case 11: {
      // 3-layers structure with one reference frame.
      // This works same as temporal_layering_mode 3.
      // This was added to compare with vp9_spatial_svc_encoder.

      // 3-layers, 4-frame period.
      int ids[4] = { 0, 2, 1, 2 };
      cfg->ts_periodicity = 4;
      *flag_periodicity = 4;
      cfg->ts_number_layers = 3;
      cfg->ts_rate_decimator[0] = 4;
      cfg->ts_rate_decimator[1] = 2;
      cfg->ts_rate_decimator[2] = 1;
      memcpy(cfg->ts_layer_id, ids, sizeof(ids));
      // 0=L, 1=GF, 2=ARF, Intra-layer prediction disabled.
      layer_flags[0] = VP8_EFLAG_NO_REF_GF | VP8_EFLAG_NO_REF_ARF |
                       VP8_EFLAG_NO_UPD_GF | VP8_EFLAG_NO_UPD_ARF;
      layer_flags[2] = VP8_EFLAG_NO_REF_GF | VP8_EFLAG_NO_REF_ARF |
                       VP8_EFLAG_NO_UPD_ARF | VP8_EFLAG_NO_UPD_LAST;
      layer_flags[1] = VP8_EFLAG_NO_REF_GF | VP8_EFLAG_NO_REF_ARF |
                       VP8_EFLAG_NO_UPD_LAST | VP8_EFLAG_NO_UPD_GF;
      layer_flags[3] = VP8_EFLAG_NO_REF_LAST | VP8_EFLAG_NO_REF_ARF |
                       VP8_EFLAG_NO_UPD_LAST | VP8_EFLAG_NO_UPD_GF;
      break;
    }
    case 12:
    default: {
      // 3-layers structure as in case 10, but no sync/refresh points for
      // layer 1 and 2.
      int ids[4] = { 0, 2, 1, 2 };
      cfg->ts_periodicity = 4;
      *flag_periodicity = 8;
      cfg->ts_number_layers = 3;
      cfg->ts_rate_decimator[0] = 4;
      cfg->ts_rate_decimator[1] = 2;
      cfg->ts_rate_decimator[2] = 1;
      memcpy(cfg->ts_layer_id, ids, sizeof(ids));
      // 0=L, 1=GF, 2=ARF.
      // Layer 0: predict from L and ARF; update L.
      layer_flags[0] =
          VP8_EFLAG_NO_UPD_GF | VP8_EFLAG_NO_UPD_ARF | VP8_EFLAG_NO_REF_GF;
      layer_flags[4] = layer_flags[0];
      // Layer 1: predict from L, G, ARF; update G.
      layer_flags[2] = VP8_EFLAG_NO_UPD_ARF | VP8_EFLAG_NO_UPD_LAST;
      layer_flags[6] = layer_flags[2];
      // Layer 2: predict from L, G, ARF; update none.
      layer_flags[1] = VP8_EFLAG_NO_UPD_GF | VP8_EFLAG_NO_UPD_ARF |
                       VP8_EFLAG_NO_UPD_LAST | VP8_EFLAG_NO_UPD_ENTROPY;
      layer_flags[3] = layer_flags[1];
      layer_flags[5] = layer_flags[1];
      layer_flags[7] = layer_flags[1];
      break;
    }
  }
}

#if ROI_MAP
static int read_mask(FILE *mask_file, int *seg_map, int allowed_mask_rows,
                     int allowed_mask_cols) {
  int mask_rows, mask_cols, i, j;
  int *map_start = seg_map;
  if (fscanf(mask_file, "%d %d\n", &mask_cols, &mask_rows) != 2) return 0;
  if (mask_rows != allowed_mask_rows || mask_cols != allowed_mask_cols) {
    return 0;
  }
  for (i = 0; i < mask_rows; i++) {
    for (j = 0; j < mask_cols; j++) {
      if (fscanf(mask_file, "%d ", &seg_map[j]) != 1) return 0;
      // reverse the bit
      seg_map[j] = 1 - seg_map[j];
    }
    seg_map += mask_cols;
  }
  seg_map = map_start;
  return 1;
}
#endif

int main(int argc, char **argv) {
  VpxVideoWriter *outfile[VPX_TS_MAX_LAYERS] = { NULL };
  vpx_codec_ctx_t codec;
  vpx_codec_enc_cfg_t cfg;
  int frame_cnt = 0;
  vpx_image_t raw;
  vpx_codec_err_t res;
  unsigned int width;
  unsigned int height;
  uint32_t error_resilient = 0;
  int speed;
  int frame_avail;
  int got_data;
  int flags = 0;
  unsigned int i;
  int pts = 0;             // PTS starts at 0.
  int frame_duration = 1;  // 1 timebase tick per frame.
  int layering_mode = 0;
  int layer_flags[VPX_TS_MAX_PERIODICITY] = { 0 };
  int flag_periodicity = 1;
#if ROI_MAP
  vpx_roi_map_t roi;
#endif
  vpx_svc_layer_id_t layer_id;
  const VpxInterface *encoder = NULL;
  struct VpxInputContext input_ctx;
  struct RateControlMetrics rc;
  int64_t cx_time = 0;
  const int min_args_base = 13;
#if CONFIG_VP9_HIGHBITDEPTH
  vpx_bit_depth_t bit_depth = VPX_BITS_8;
  int input_bit_depth = 8;
  const int min_args = min_args_base + 1;
#else
  const int min_args = min_args_base;
#endif  // CONFIG_VP9_HIGHBITDEPTH
  double sum_bitrate = 0.0;
  double sum_bitrate2 = 0.0;
  double framerate = 30.0;
#if ROI_MAP
  FILE *mask_file = NULL;
  int block_size = 8;
  int mask_rows = 0;
  int mask_cols = 0;
  int *mask_map;
  int *prev_mask_map;
#endif
  zero(rc.layer_target_bitrate);
  memset(&layer_id, 0, sizeof(vpx_svc_layer_id_t));
  memset(&input_ctx, 0, sizeof(input_ctx));
  /* Setup default input stream settings */
  input_ctx.framerate.numerator = 30;
  input_ctx.framerate.denominator = 1;
  input_ctx.only_i420 = 1;
  input_ctx.bit_depth = 0;

  exec_name = argv[0];
  // Check usage and arguments.
  if (argc < min_args) {
#if CONFIG_VP9_HIGHBITDEPTH
    die("Usage: %s <infile> <outfile> <codec_type(vp8/vp9)> <width> <height> "
        "<rate_num> <rate_den> <speed> <frame_drop_threshold> "
        "<error_resilient> <threads> <mode> "
        "<Rate_0> ... <Rate_nlayers-1> <bit-depth> \n",
        argv[0]);
#else
    die("Usage: %s <infile> <outfile> <codec_type(vp8/vp9)> <width> <height> "
        "<rate_num> <rate_den> <speed> <frame_drop_threshold> "
        "<error_resilient> <threads> <mode> "
        "<Rate_0> ... <Rate_nlayers-1> \n",
        argv[0]);
#endif  // CONFIG_VP9_HIGHBITDEPTH
  }

  encoder = get_vpx_encoder_by_name(argv[3]);
  if (!encoder) die("Unsupported codec.");

  printf("Using %s\n", vpx_codec_iface_name(encoder->codec_interface()));

  width = (unsigned int)strtoul(argv[4], NULL, 0);
  height = (unsigned int)strtoul(argv[5], NULL, 0);
  if (width < 16 || width % 2 || height < 16 || height % 2) {
    die("Invalid resolution: %d x %d", width, height);
  }

  layering_mode = (int)strtol(argv[12], NULL, 0);
  if (layering_mode < 0 || layering_mode > 13) {
    die("Invalid layering mode (0..12) %s", argv[12]);
  }

#if ROI_MAP
  if (argc != min_args + mode_to_num_layers[layering_mode] + 1) {
    die("Invalid number of arguments");
  }
#else
  if (argc != min_args + mode_to_num_layers[layering_mode]) {
    die("Invalid number of arguments");
  }
#endif

  input_ctx.filename = argv[1];
  open_input_file(&input_ctx);

#if CONFIG_VP9_HIGHBITDEPTH
  switch (strtol(argv[argc - 1], NULL, 0)) {
    case 8:
      bit_depth = VPX_BITS_8;
      input_bit_depth = 8;
      break;
    case 10:
      bit_depth = VPX_BITS_10;
      input_bit_depth = 10;
      break;
    case 12:
      bit_depth = VPX_BITS_12;
      input_bit_depth = 12;
      break;
    default: die("Invalid bit depth (8, 10, 12) %s", argv[argc - 1]);
  }

  // Y4M reader has its own allocation.
  if (input_ctx.file_type != FILE_TYPE_Y4M) {
    if (!vpx_img_alloc(
            &raw,
            bit_depth == VPX_BITS_8 ? VPX_IMG_FMT_I420 : VPX_IMG_FMT_I42016,
            width, height, 32)) {
      die("Failed to allocate image (%dx%d)", width, height);
    }
  }
#else
  // Y4M reader has its own allocation.
  if (input_ctx.file_type != FILE_TYPE_Y4M) {
    if (!vpx_img_alloc(&raw, VPX_IMG_FMT_I420, width, height, 32)) {
      die("Failed to allocate image (%dx%d)", width, height);
    }
  }
#endif  // CONFIG_VP9_HIGHBITDEPTH

  // Populate encoder configuration.
  res = vpx_codec_enc_config_default(encoder->codec_interface(), &cfg, 0);
  if (res) {
    printf("Failed to get config: %s\n", vpx_codec_err_to_string(res));
    return EXIT_FAILURE;
  }

  // Update the default configuration with our settings.
  cfg.g_w = width;
  cfg.g_h = height;

#if CONFIG_VP9_HIGHBITDEPTH
  if (bit_depth != VPX_BITS_8) {
    cfg.g_bit_depth = bit_depth;
    cfg.g_input_bit_depth = input_bit_depth;
    cfg.g_profile = 2;
  }
#endif  // CONFIG_VP9_HIGHBITDEPTH

  // Timebase format e.g. 30fps: numerator=1, demoninator = 30.
  cfg.g_timebase.num = (int)strtol(argv[6], NULL, 0);
  cfg.g_timebase.den = (int)strtol(argv[7], NULL, 0);

  speed = (int)strtol(argv[8], NULL, 0);
  if (speed < 0) {
    die("Invalid speed setting: must be positive");
  }
  if (strncmp(encoder->name, "vp9", 3) == 0 && speed > 9) {
    warn("Mapping speed %d to speed 9.\n", speed);
  }

  for (i = min_args_base;
       (int)i < min_args_base + mode_to_num_layers[layering_mode]; ++i) {
    rc.layer_target_bitrate[i - 13] = (int)strtol(argv[i], NULL, 0);
    if (strncmp(encoder->name, "vp8", 3) == 0)
      cfg.ts_target_bitrate[i - 13] = rc.layer_target_bitrate[i - 13];
    else if (strncmp(encoder->name, "vp9", 3) == 0)
      cfg.layer_target_bitrate[i - 13] = rc.layer_target_bitrate[i - 13];
  }

  // Real time parameters.
  cfg.rc_dropframe_thresh = (unsigned int)strtoul(argv[9], NULL, 0);
  cfg.rc_end_usage = VPX_CBR;
  cfg.rc_min_quantizer = 2;
  cfg.rc_max_quantizer = 56;
  if (strncmp(encoder->name, "vp9", 3) == 0) cfg.rc_max_quantizer = 52;
  cfg.rc_undershoot_pct = 50;
  cfg.rc_overshoot_pct = 50;
  cfg.rc_buf_initial_sz = 600;
  cfg.rc_buf_optimal_sz = 600;
  cfg.rc_buf_sz = 1000;

  // Disable dynamic resizing by default.
  cfg.rc_resize_allowed = 0;

  // Use 1 thread as default.
  cfg.g_threads = (unsigned int)strtoul(argv[11], NULL, 0);

  error_resilient = (uint32_t)strtoul(argv[10], NULL, 0);
  if (error_resilient != 0 && error_resilient != 1) {
    die("Invalid value for error resilient (0, 1): %d.", error_resilient);
  }
  // Enable error resilient mode.
  cfg.g_error_resilient = error_resilient;
  cfg.g_lag_in_frames = 0;
  cfg.kf_mode = VPX_KF_AUTO;

  // Disable automatic keyframe placement.
  cfg.kf_min_dist = cfg.kf_max_dist = 3000;

  cfg.temporal_layering_mode = VP9E_TEMPORAL_LAYERING_MODE_BYPASS;

  set_temporal_layer_pattern(layering_mode, &cfg, layer_flags,
                             &flag_periodicity);

  set_rate_control_metrics(&rc, &cfg);

  if (input_ctx.file_type == FILE_TYPE_Y4M) {
    if (input_ctx.width != cfg.g_w || input_ctx.height != cfg.g_h) {
      die("Incorrect width or height: %d x %d", cfg.g_w, cfg.g_h);
    }
    if (input_ctx.framerate.numerator != cfg.g_timebase.den ||
        input_ctx.framerate.denominator != cfg.g_timebase.num) {
      die("Incorrect framerate: numerator %d denominator %d",
          cfg.g_timebase.num, cfg.g_timebase.den);
    }
  }

  framerate = cfg.g_timebase.den / cfg.g_timebase.num;
  // Open an output file for each stream.
  for (i = 0; i < cfg.ts_number_layers; ++i) {
    char file_name[PATH_MAX];
    VpxVideoInfo info;
    info.codec_fourcc = encoder->fourcc;
    info.frame_width = cfg.g_w;
    info.frame_height = cfg.g_h;
    info.time_base.numerator = cfg.g_timebase.num;
    info.time_base.denominator = cfg.g_timebase.den;

    snprintf(file_name, sizeof(file_name), "%s_%d.ivf", argv[2], i);
    outfile[i] = vpx_video_writer_open(file_name, kContainerIVF, &info);
    if (!outfile[i]) die("Failed to open %s for writing", file_name);

    assert(outfile[i] != NULL);
  }
  // No spatial layers in this encoder.
  cfg.ss_number_layers = 1;

// Initialize codec.
#if CONFIG_VP9_HIGHBITDEPTH
  if (vpx_codec_enc_init(
          &codec, encoder->codec_interface(), &cfg,
          bit_depth == VPX_BITS_8 ? 0 : VPX_CODEC_USE_HIGHBITDEPTH))
#else
  if (vpx_codec_enc_init(&codec, encoder->codec_interface(), &cfg, 0))
#endif  // CONFIG_VP9_HIGHBITDEPTH
    die("Failed to initialize encoder");

#if ROI_MAP
  mask_rows = (cfg.g_h + block_size - 1) / block_size;
  mask_cols = (cfg.g_w + block_size - 1) / block_size;
  mask_map = (int *)calloc(mask_rows * mask_cols, sizeof(*mask_map));
  prev_mask_map = (int *)calloc(mask_rows * mask_cols, sizeof(*mask_map));
#endif

  if (strncmp(encoder->name, "vp8", 3) == 0) {
    vpx_codec_control(&codec, VP8E_SET_CPUUSED, -speed);
    vpx_codec_control(&codec, VP8E_SET_NOISE_SENSITIVITY, kVp8DenoiserOff);
    vpx_codec_control(&codec, VP8E_SET_STATIC_THRESHOLD, 1);
    vpx_codec_control(&codec, VP8E_SET_GF_CBR_BOOST_PCT, 0);
#if ROI_MAP
    set_roi_map(encoder->name, &cfg, &roi);
    if (vpx_codec_control(&codec, VP8E_SET_ROI_MAP, &roi))
      die_codec(&codec, "Failed to set ROI map");
#endif
  } else if (strncmp(encoder->name, "vp9", 3) == 0) {
    vpx_svc_extra_cfg_t svc_params;
    memset(&svc_params, 0, sizeof(svc_params));
    vpx_codec_control(&codec, VP9E_SET_POSTENCODE_DROP, 0);
    vpx_codec_control(&codec, VP9E_SET_DISABLE_OVERSHOOT_MAXQ_CBR, 0);
    vpx_codec_control(&codec, VP8E_SET_CPUUSED, speed);
    vpx_codec_control(&codec, VP9E_SET_AQ_MODE, 3);
    vpx_codec_control(&codec, VP9E_SET_GF_CBR_BOOST_PCT, 0);
    vpx_codec_control(&codec, VP9E_SET_FRAME_PARALLEL_DECODING, 0);
    vpx_codec_control(&codec, VP9E_SET_FRAME_PERIODIC_BOOST, 0);
    vpx_codec_control(&codec, VP9E_SET_NOISE_SENSITIVITY, kVp9DenoiserOff);
    vpx_codec_control(&codec, VP8E_SET_STATIC_THRESHOLD, 1);
    vpx_codec_control(&codec, VP9E_SET_TUNE_CONTENT, 0);
    vpx_codec_control(&codec, VP9E_SET_TILE_COLUMNS, get_msb(cfg.g_threads));
    vpx_codec_control(&codec, VP9E_SET_DISABLE_LOOPFILTER, 0);

    if (cfg.g_threads > 1)
      vpx_codec_control(&codec, VP9E_SET_ROW_MT, 1);
    else
      vpx_codec_control(&codec, VP9E_SET_ROW_MT, 0);
    if (vpx_codec_control(&codec, VP9E_SET_SVC, layering_mode > 0 ? 1 : 0))
      die_codec(&codec, "Failed to set SVC");
    for (i = 0; i < cfg.ts_number_layers; ++i) {
      svc_params.max_quantizers[i] = cfg.rc_max_quantizer;
      svc_params.min_quantizers[i] = cfg.rc_min_quantizer;
    }
    svc_params.scaling_factor_num[0] = cfg.g_h;
    svc_params.scaling_factor_den[0] = cfg.g_h;
    vpx_codec_control(&codec, VP9E_SET_SVC_PARAMETERS, &svc_params);
  }
  if (strncmp(encoder->name, "vp8", 3) == 0) {
    vpx_codec_control(&codec, VP8E_SET_SCREEN_CONTENT_MODE, 0);
  }
  vpx_codec_control(&codec, VP8E_SET_TOKEN_PARTITIONS, 1);
  // This controls the maximum target size of the key frame.
  // For generating smaller key frames, use a smaller max_intra_size_pct
  // value, like 100 or 200.
  {
    const int max_intra_size_pct = 1000;
    vpx_codec_control(&codec, VP8E_SET_MAX_INTRA_BITRATE_PCT,
                      max_intra_size_pct);
  }

  frame_avail = 1;
  while (frame_avail || got_data) {
    struct vpx_usec_timer timer;
    vpx_codec_iter_t iter = NULL;
    const vpx_codec_cx_pkt_t *pkt;
#if ROI_MAP
    char mask_file_name[255];
#endif
    // Update the temporal layer_id. No spatial layers in this test.
    layer_id.spatial_layer_id = 0;
    layer_id.temporal_layer_id =
        cfg.ts_layer_id[frame_cnt % cfg.ts_periodicity];
    layer_id.temporal_layer_id_per_spatial[0] = layer_id.temporal_layer_id;
    if (strncmp(encoder->name, "vp9", 3) == 0) {
      vpx_codec_control(&codec, VP9E_SET_SVC_LAYER_ID, &layer_id);
    } else if (strncmp(encoder->name, "vp8", 3) == 0) {
      vpx_codec_control(&codec, VP8E_SET_TEMPORAL_LAYER_ID,
                        layer_id.temporal_layer_id);
    }
    flags = layer_flags[frame_cnt % flag_periodicity];
    if (layering_mode == 0) flags = 0;
#if ROI_MAP
    snprintf(mask_file_name, sizeof(mask_file_name), "%s%05d.txt",
             argv[argc - 1], frame_cnt);
    mask_file = fopen(mask_file_name, "r");
    if (mask_file != NULL) {
      int mask_is_valid = read_mask(mask_file, mask_map, mask_rows, mask_cols);
      fclose(mask_file);
      if (mask_is_valid) {
        // set_roi_map(encoder->name, &cfg, &roi);
        set_roi_skip_map(&cfg, &roi, mask_map, prev_mask_map, frame_cnt);
        if (vpx_codec_control(&codec, VP9E_SET_ROI_MAP, &roi))
          die_codec(&codec, "Failed to set ROI map");
      } else {
        die_codec(&codec, "Mask input is invalid for ROI map");
      }
    }
#endif
    frame_avail = read_frame(&input_ctx, &raw);
    if (frame_avail) ++rc.layer_input_frames[layer_id.temporal_layer_id];
    vpx_usec_timer_start(&timer);
    if (vpx_codec_encode(&codec, frame_avail ? &raw : NULL, pts, 1, flags,
                         VPX_DL_REALTIME)) {
      die_codec(&codec, "Failed to encode frame");
    }
    vpx_usec_timer_mark(&timer);
    cx_time += vpx_usec_timer_elapsed(&timer);
    // Reset KF flag.
    if (layering_mode != 7) {
      layer_flags[0] &= ~VPX_EFLAG_FORCE_KF;
    }
    got_data = 0;
    while ((pkt = vpx_codec_get_cx_data(&codec, &iter))) {
      got_data = 1;
      switch (pkt->kind) {
        case VPX_CODEC_CX_FRAME_PKT:
          for (i = cfg.ts_layer_id[frame_cnt % cfg.ts_periodicity];
               i < cfg.ts_number_layers; ++i) {
            vpx_video_writer_write_frame(outfile[i], pkt->data.frame.buf,
                                         pkt->data.frame.sz, pts);
            ++rc.layer_tot_enc_frames[i];
            rc.layer_encoding_bitrate[i] += 8.0 * pkt->data.frame.sz;
            // Keep count of rate control stats per layer (for non-key frames).
            if (i == cfg.ts_layer_id[frame_cnt % cfg.ts_periodicity] &&
                !(pkt->data.frame.flags & VPX_FRAME_IS_KEY)) {
              rc.layer_avg_frame_size[i] += 8.0 * pkt->data.frame.sz;
              rc.layer_avg_rate_mismatch[i] +=
                  fabs(8.0 * pkt->data.frame.sz - rc.layer_pfb[i]) /
                  rc.layer_pfb[i];
              ++rc.layer_enc_frames[i];
            }
          }
          // Update for short-time encoding bitrate states, for moving window
          // of size rc->window, shifted by rc->window / 2.
          // Ignore first window segment, due to key frame.
          if (rc.window_size == 0) rc.window_size = 15;
          if (frame_cnt > rc.window_size) {
            sum_bitrate += 0.001 * 8.0 * pkt->data.frame.sz * framerate;
            if (frame_cnt % rc.window_size == 0) {
              rc.window_count += 1;
              rc.avg_st_encoding_bitrate += sum_bitrate / rc.window_size;
              rc.variance_st_encoding_bitrate +=
                  (sum_bitrate / rc.window_size) *
                  (sum_bitrate / rc.window_size);
              sum_bitrate = 0.0;
            }
          }
          // Second shifted window.
          if (frame_cnt > rc.window_size + rc.window_size / 2) {
            sum_bitrate2 += 0.001 * 8.0 * pkt->data.frame.sz * framerate;
            if (frame_cnt > 2 * rc.window_size &&
                frame_cnt % rc.window_size == 0) {
              rc.window_count += 1;
              rc.avg_st_encoding_bitrate += sum_bitrate2 / rc.window_size;
              rc.variance_st_encoding_bitrate +=
                  (sum_bitrate2 / rc.window_size) *
                  (sum_bitrate2 / rc.window_size);
              sum_bitrate2 = 0.0;
            }
          }
          break;
        default: break;
      }
    }
    ++frame_cnt;
    pts += frame_duration;
  }
#if ROI_MAP
  free(mask_map);
  free(prev_mask_map);
#endif
  close_input_file(&input_ctx);
  printout_rate_control_summary(&rc, &cfg, frame_cnt);
  printf("\n");
  printf("Frame cnt and encoding time/FPS stats for encoding: %d %f %f \n",
         frame_cnt, 1000 * (float)cx_time / (double)(frame_cnt * 1000000),
         1000000 * (double)frame_cnt / (double)cx_time);

  if (vpx_codec_destroy(&codec)) die_codec(&codec, "Failed to destroy codec");

  // Try to rewrite the output file headers with the actual frame count.
  for (i = 0; i < cfg.ts_number_layers; ++i) vpx_video_writer_close(outfile[i]);

  if (input_ctx.file_type != FILE_TYPE_Y4M) {
    vpx_img_free(&raw);
  }

#if ROI_MAP
  free(roi.roi_map);
#endif
  return EXIT_SUCCESS;
}
