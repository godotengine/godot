/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP8_ENCODER_ENCODEMB_H_
#define VPX_VP8_ENCODER_ENCODEMB_H_

#include "onyx_int.h"

#ifdef __cplusplus
extern "C" {
#endif
void vp8_encode_inter16x16(MACROBLOCK *x);

void vp8_subtract_b(BLOCK *be, BLOCKD *bd, int pitch);
void vp8_subtract_mbuv(short *diff, unsigned char *usrc, unsigned char *vsrc,
                       int src_stride, unsigned char *upred,
                       unsigned char *vpred, int pred_stride);
void vp8_subtract_mby(short *diff, unsigned char *src, int src_stride,
                      unsigned char *pred, int pred_stride);

void vp8_build_dcblock(MACROBLOCK *b);
void vp8_transform_mb(MACROBLOCK *mb);
void vp8_transform_mbuv(MACROBLOCK *x);
void vp8_transform_intra_mby(MACROBLOCK *x);

void vp8_optimize_mby(MACROBLOCK *x);
void vp8_optimize_mbuv(MACROBLOCK *x);
void vp8_encode_inter16x16y(MACROBLOCK *x);
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_ENCODER_ENCODEMB_H_
