/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP8_ENCODER_ENCODEINTRA_H_
#define VPX_VP8_ENCODER_ENCODEINTRA_H_
#include "onyx_int.h"

#ifdef __cplusplus
extern "C" {
#endif

int vp8_encode_intra(MACROBLOCK *x, int use_dc_pred);
void vp8_encode_intra16x16mby(MACROBLOCK *x);
void vp8_encode_intra16x16mbuv(MACROBLOCK *x);
void vp8_encode_intra4x4mby(MACROBLOCK *mb);
void vp8_encode_intra4x4block(MACROBLOCK *x, int ib);
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_ENCODER_ENCODEINTRA_H_
