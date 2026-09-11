/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP9_ENCODER_VP9_PICKLPF_H_
#define VPX_VP9_ENCODER_VP9_PICKLPF_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "vp9/encoder/vp9_encoder.h"

struct yv12_buffer_config;
struct VP9_COMP;

void vp9_pick_filter_level(const struct yv12_buffer_config *sd,
                           struct VP9_COMP *cpi, LPF_PICK_METHOD method);
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_ENCODER_VP9_PICKLPF_H_
