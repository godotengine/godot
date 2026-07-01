/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP8_COMMON_SWAPYV12BUFFER_H_
#define VPX_VP8_COMMON_SWAPYV12BUFFER_H_

#include "vpx_scale/yv12config.h"

#ifdef __cplusplus
extern "C" {
#endif

void vp8_swap_yv12_buffer(YV12_BUFFER_CONFIG *new_frame,
                          YV12_BUFFER_CONFIG *last_frame);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_COMMON_SWAPYV12BUFFER_H_
