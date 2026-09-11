/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP9_ENCODER_VP9_EXTEND_H_
#define VPX_VP9_ENCODER_VP9_EXTEND_H_

#include "vpx_scale/yv12config.h"
#include "vpx/vpx_integer.h"

#ifdef __cplusplus
extern "C" {
#endif

void vp9_copy_and_extend_frame(const YV12_BUFFER_CONFIG *src,
                               YV12_BUFFER_CONFIG *dst);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_ENCODER_VP9_EXTEND_H_
