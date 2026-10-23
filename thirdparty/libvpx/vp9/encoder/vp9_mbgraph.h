/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP9_ENCODER_VP9_MBGRAPH_H_
#define VPX_VP9_ENCODER_VP9_MBGRAPH_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  struct {
    int err;
    union {
      int_mv mv;
      PREDICTION_MODE mode;
    } m;
  } ref[MAX_REF_FRAMES];
} MBGRAPH_MB_STATS;

typedef struct {
  MBGRAPH_MB_STATS *mb_stats;
} MBGRAPH_FRAME_STATS;

struct VP9_COMP;

void vp9_update_mbgraph_stats(struct VP9_COMP *cpi);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_ENCODER_VP9_MBGRAPH_H_
