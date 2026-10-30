/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP8_ENCODER_TEMPORAL_FILTER_H_
#define VPX_VP8_ENCODER_TEMPORAL_FILTER_H_

#ifdef __cplusplus
extern "C" {
#endif

struct VP8_COMP;

void vp8_temporal_filter_prepare_c(struct VP8_COMP *cpi, int distance);

#ifdef __cplusplus
}
#endif

#endif  // VPX_VP8_ENCODER_TEMPORAL_FILTER_H_
