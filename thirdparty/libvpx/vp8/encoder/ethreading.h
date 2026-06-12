/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP8_ENCODER_ETHREADING_H_
#define VPX_VP8_ENCODER_ETHREADING_H_

#include "vp8/encoder/onyx_int.h"

#ifdef __cplusplus
extern "C" {
#endif

struct VP8_COMP;
struct macroblock;

void vp8cx_init_mbrthread_data(struct VP8_COMP *cpi, struct macroblock *x,
                               MB_ROW_COMP *mbr_ei, int count);
int vp8cx_create_encoder_threads(struct VP8_COMP *cpi);
void vp8cx_remove_encoder_threads(struct VP8_COMP *cpi);

#ifdef __cplusplus
}
#endif

#endif  // VPX_VP8_ENCODER_ETHREADING_H_
