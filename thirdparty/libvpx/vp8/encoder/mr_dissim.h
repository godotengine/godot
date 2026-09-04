/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP8_ENCODER_MR_DISSIM_H_
#define VPX_VP8_ENCODER_MR_DISSIM_H_
#include "vpx_config.h"

#ifdef __cplusplus
extern "C" {
#endif

extern void vp8_cal_low_res_mb_cols(VP8_COMP *cpi);
extern void vp8_cal_dissimilarity(VP8_COMP *cpi);
extern void vp8_store_drop_frame_info(VP8_COMP *cpi);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_ENCODER_MR_DISSIM_H_
