/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP8_ENCODER_SEGMENTATION_H_
#define VPX_VP8_ENCODER_SEGMENTATION_H_

#include "string.h"
#include "vp8/common/blockd.h"
#include "onyx_int.h"

#ifdef __cplusplus
extern "C" {
#endif

extern void vp8_update_gf_usage_maps(VP8_COMP *cpi, VP8_COMMON *cm,
                                     MACROBLOCK *x);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_ENCODER_SEGMENTATION_H_
