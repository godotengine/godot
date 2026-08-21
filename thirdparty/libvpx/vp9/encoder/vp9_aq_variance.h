/*
 *  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP9_ENCODER_VP9_AQ_VARIANCE_H_
#define VPX_VP9_ENCODER_VP9_AQ_VARIANCE_H_

#include "vp9/encoder/vp9_encoder.h"

#ifdef __cplusplus
extern "C" {
#endif

unsigned int vp9_vaq_segment_id(int energy);
void vp9_vaq_frame_setup(VP9_COMP *cpi);

void vp9_get_sub_block_energy(VP9_COMP *cpi, MACROBLOCK *mb, int mi_row,
                              int mi_col, BLOCK_SIZE bsize, int *min_e,
                              int *max_e);
int vp9_block_energy(VP9_COMP *cpi, MACROBLOCK *x, BLOCK_SIZE bs);

double vp9_log_block_var(VP9_COMP *cpi, MACROBLOCK *x, BLOCK_SIZE bs);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_ENCODER_VP9_AQ_VARIANCE_H_
