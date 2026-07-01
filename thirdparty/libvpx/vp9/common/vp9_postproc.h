/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP9_COMMON_VP9_POSTPROC_H_
#define VPX_VP9_COMMON_VP9_POSTPROC_H_

#include "vpx_ports/mem.h"
#include "vpx_scale/yv12config.h"
#include "vp9/common/vp9_blockd.h"
#include "vp9/common/vp9_mfqe.h"
#include "vp9/common/vp9_ppflags.h"

#ifdef __cplusplus
extern "C" {
#endif

struct postproc_state {
  int last_q;
  int last_noise;
  int last_base_qindex;
  int last_frame_valid;
  MODE_INFO *prev_mip;
  MODE_INFO *prev_mi;
  int clamp;
  uint8_t *limits;
  int limits_size;
  int8_t *generated_noise;
};

struct VP9Common;

#define MFQE_PRECISION 4

int vp9_post_proc_frame(struct VP9Common *cm, YV12_BUFFER_CONFIG *dest,
                        vp9_ppflags_t *ppflags, int unscaled_width);

void vp9_denoise(struct VP9Common *cm, const YV12_BUFFER_CONFIG *src,
                 YV12_BUFFER_CONFIG *dst, int q, uint8_t *limits);

void vp9_deblock(struct VP9Common *cm, const YV12_BUFFER_CONFIG *src,
                 YV12_BUFFER_CONFIG *dst, int q, uint8_t *limits);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_COMMON_VP9_POSTPROC_H_
