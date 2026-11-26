/*
 *  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

/**
 * SvcContext - input parameters and state to encode a multi-layered
 * spatial SVC frame
 */

#ifndef VPX_EXAMPLES_SVC_CONTEXT_H_
#define VPX_EXAMPLES_SVC_CONTEXT_H_

#include "vpx/vp8cx.h"
#include "vpx/vpx_encoder.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum SVC_LOG_LEVEL {
  SVC_LOG_ERROR,
  SVC_LOG_INFO,
  SVC_LOG_DEBUG
} SVC_LOG_LEVEL;

typedef struct {
  // public interface to svc_command options
  int spatial_layers;   // number of spatial layers
  int temporal_layers;  // number of temporal layers
  int temporal_layering_mode;
  SVC_LOG_LEVEL log_level;  // amount of information to display
  int output_rc_stat;       // for outputting rc stats
  int speed;                // speed setting for codec
  int threads;
  int aqmode;  // turns on aq-mode=3 (cyclic_refresh): 0=off, 1=on.
  int use_psnr;
  // private storage for vpx_svc_encode
  void *internal;
} SvcContext;

#define OPTION_BUFFER_SIZE 1024
#define COMPONENTS 4  // psnr & sse statistics maintained for total, y, u, v

typedef struct SvcInternal {
  char options[OPTION_BUFFER_SIZE];  // set by vpx_svc_set_options

  // values extracted from option, quantizers
  vpx_svc_extra_cfg_t svc_params;
  int enable_auto_alt_ref[VPX_SS_MAX_LAYERS];
  int bitrates[VPX_MAX_LAYERS];

  // accumulated statistics
  double psnr_sum[VPX_SS_MAX_LAYERS][COMPONENTS];  // total/Y/U/V
  uint64_t sse_sum[VPX_SS_MAX_LAYERS][COMPONENTS];
  uint32_t bytes_sum[VPX_SS_MAX_LAYERS];
  int number_of_frames[VPX_SS_MAX_LAYERS];

  // codec encoding values
  int width;    // width of highest layer
  int height;   // height of highest layer
  int kf_dist;  // distance between keyframes

  // state variables
  int layer;
  int use_multiple_frame_contexts;

  vpx_codec_ctx_t *codec_ctx;
} SvcInternal_t;

/**
 * Set SVC options
 * options are supplied as a single string separated by spaces
 * Format: encoding-mode=<i|ip|alt-ip|gf>
 *         layers=<layer_count>
 *         scaling-factors=<n1>/<d1>,<n2>/<d2>,...
 *         quantizers=<q1>,<q2>,...
 */
vpx_codec_err_t vpx_svc_set_options(SvcContext *svc_ctx, const char *options);

/**
 * initialize SVC encoding
 */
vpx_codec_err_t vpx_svc_init(SvcContext *svc_ctx, vpx_codec_ctx_t *codec_ctx,
                             vpx_codec_iface_t *iface,
                             vpx_codec_enc_cfg_t *cfg);
/**
 * encode a frame of video with multiple layers
 */
vpx_codec_err_t vpx_svc_encode(SvcContext *svc_ctx, vpx_codec_ctx_t *codec_ctx,
                               struct vpx_image *rawimg, vpx_codec_pts_t pts,
                               int64_t duration, int deadline);

/**
 * finished with svc encoding, release allocated resources
 */
void vpx_svc_release(SvcContext *svc_ctx);

/**
 * dump accumulated statistics and reset accumulated values
 */
void vpx_svc_dump_statistics(SvcContext *svc_ctx);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_EXAMPLES_SVC_CONTEXT_H_
