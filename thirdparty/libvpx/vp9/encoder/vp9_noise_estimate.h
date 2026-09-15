/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP9_ENCODER_VP9_NOISE_ESTIMATE_H_
#define VPX_VP9_ENCODER_VP9_NOISE_ESTIMATE_H_

#include "vp9/encoder/vp9_block.h"
#include "vp9/encoder/vp9_skin_detection.h"
#include "vpx_scale/yv12config.h"

#if CONFIG_VP9_TEMPORAL_DENOISING
#include "vp9/encoder/vp9_denoiser.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_VAR_HIST_BINS 20

typedef enum noise_level { kLowLow, kLow, kMedium, kHigh } NOISE_LEVEL;

typedef struct noise_estimate {
  int enabled;
  NOISE_LEVEL level;
  int value;
  int thresh;
  int adapt_thresh;
  int count;
  int last_w;
  int last_h;
  int num_frames_estimate;
} NOISE_ESTIMATE;

struct VP9_COMP;

void vp9_noise_estimate_init(NOISE_ESTIMATE *const ne, int width, int height);

NOISE_LEVEL vp9_noise_estimate_extract_level(NOISE_ESTIMATE *const ne);

void vp9_update_noise_estimate(struct VP9_COMP *const cpi);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_ENCODER_VP9_NOISE_ESTIMATE_H_
