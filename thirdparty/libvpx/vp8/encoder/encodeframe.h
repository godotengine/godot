/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#ifndef VPX_VP8_ENCODER_ENCODEFRAME_H_
#define VPX_VP8_ENCODER_ENCODEFRAME_H_

#include "vp8/encoder/tokenize.h"

#ifdef __cplusplus
extern "C" {
#endif

struct VP8_COMP;
struct macroblock;

void vp8_activity_masking(struct VP8_COMP *cpi, MACROBLOCK *x);

void vp8_build_block_offsets(struct macroblock *x);

void vp8_setup_block_ptrs(struct macroblock *x);

void vp8_encode_frame(struct VP8_COMP *cpi);

int vp8cx_encode_inter_macroblock(struct VP8_COMP *cpi, struct macroblock *x,
                                  TOKENEXTRA **t, int recon_yoffset,
                                  int recon_uvoffset, int mb_row, int mb_col);

int vp8cx_encode_intra_macroblock(struct VP8_COMP *cpi, struct macroblock *x,
                                  TOKENEXTRA **t);
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_ENCODER_ENCODEFRAME_H_
