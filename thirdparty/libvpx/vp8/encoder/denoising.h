/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP8_ENCODER_DENOISING_H_
#define VPX_VP8_ENCODER_DENOISING_H_

#include "block.h"
#include "vp8/common/loopfilter.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SUM_DIFF_THRESHOLD 512
#define SUM_DIFF_THRESHOLD_HIGH 600
#define MOTION_MAGNITUDE_THRESHOLD (8 * 3)

#define SUM_DIFF_THRESHOLD_UV (96)  // (8 * 8 * 1.5)
#define SUM_DIFF_THRESHOLD_HIGH_UV (8 * 8 * 2)
#define SUM_DIFF_FROM_AVG_THRESH_UV (8 * 8 * 8)
#define MOTION_MAGNITUDE_THRESHOLD_UV (8 * 3)

#define MAX_GF_ARF_DENOISE_RANGE (8)

enum vp8_denoiser_decision { COPY_BLOCK, FILTER_BLOCK };

enum vp8_denoiser_filter_state { kNoFilter, kFilterZeroMV, kFilterNonZeroMV };

enum vp8_denoiser_mode {
  kDenoiserOff,
  kDenoiserOnYOnly,
  kDenoiserOnYUV,
  kDenoiserOnYUVAggressive,
  kDenoiserOnAdaptive
};

typedef struct {
  // Scale factor on sse threshold above which no denoising is done.
  unsigned int scale_sse_thresh;
  // Scale factor on motion magnitude threshold above which no
  // denoising is done.
  unsigned int scale_motion_thresh;
  // Scale factor on motion magnitude below which we increase the strength of
  // the temporal filter (in function vp8_denoiser_filter).
  unsigned int scale_increase_filter;
  // Scale factor to bias to ZEROMV for denoising.
  unsigned int denoise_mv_bias;
  // Scale factor to bias to ZEROMV for coding mode selection.
  unsigned int pickmode_mv_bias;
  // Quantizer threshold below which we use the segmentation map to switch off
  // loop filter for blocks that have been coded as ZEROMV-LAST a certain number
  // (consec_zerolast) of consecutive frames. Note that the delta-QP is set to
  // 0 when segmentation map is used for shutting off loop filter.
  unsigned int qp_thresh;
  // Threshold for number of consecutive frames for blocks coded as ZEROMV-LAST.
  unsigned int consec_zerolast;
  // Threshold for amount of spatial blur on Y channel. 0 means no spatial blur.
  unsigned int spatial_blur;
} denoise_params;

typedef struct vp8_denoiser {
  YV12_BUFFER_CONFIG yv12_running_avg[MAX_REF_FRAMES];
  YV12_BUFFER_CONFIG yv12_mc_running_avg;
  // TODO(marpan): Should remove yv12_last_source and use vp8_lookahead_peak.
  YV12_BUFFER_CONFIG yv12_last_source;
  unsigned char *denoise_state;
  int num_mb_cols;
  int denoiser_mode;
  int threshold_aggressive_mode;
  int nmse_source_diff;
  int nmse_source_diff_count;
  int qp_avg;
  int qp_threshold_up;
  int qp_threshold_down;
  int bitrate_threshold;
  denoise_params denoise_pars;
} VP8_DENOISER;

int vp8_denoiser_allocate(VP8_DENOISER *denoiser, int width, int height,
                          int num_mb_rows, int num_mb_cols, int mode);

void vp8_denoiser_free(VP8_DENOISER *denoiser);

void vp8_denoiser_set_parameters(VP8_DENOISER *denoiser, int mode);

void vp8_denoiser_denoise_mb(VP8_DENOISER *denoiser, MACROBLOCK *x,
                             unsigned int best_sse, unsigned int zero_mv_sse,
                             int recon_yoffset, int recon_uvoffset,
                             loop_filter_info_n *lfi_n, int mb_row, int mb_col,
                             int block_index, int consec_zero_last);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_ENCODER_DENOISING_H_
