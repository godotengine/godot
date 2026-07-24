/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP8_ENCODER_ENCODEMV_H_
#define VPX_VP8_ENCODER_ENCODEMV_H_

#include "onyx_int.h"

#ifdef __cplusplus
extern "C" {
#endif

void vp8_write_mvprobs(VP8_COMP *);
void vp8_encode_motion_vector(vp8_writer *, const MV *, const MV_CONTEXT *);
void vp8_build_component_cost_table(int *mvcost[2], const MV_CONTEXT *mvc,
                                    int mvc_flag[2]);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_ENCODER_ENCODEMV_H_
