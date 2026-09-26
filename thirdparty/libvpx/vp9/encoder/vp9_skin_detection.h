/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP9_ENCODER_VP9_SKIN_DETECTION_H_
#define VPX_VP9_ENCODER_VP9_SKIN_DETECTION_H_

#include "vp9/common/vp9_blockd.h"
#include "vpx_dsp/skin_detection.h"
#include "vpx_util/vpx_write_yuv_frame.h"

#ifdef __cplusplus
extern "C" {
#endif

struct VP9_COMP;

int vp9_compute_skin_block(const uint8_t *y, const uint8_t *u, const uint8_t *v,
                           int stride, int strideuv, int bsize,
                           int consec_zeromv, int curr_motion_magn);

void vp9_compute_skin_sb(struct VP9_COMP *const cpi, BLOCK_SIZE bsize,
                         int mi_row, int mi_col);

#ifdef OUTPUT_YUV_SKINMAP
// For viewing skin map on input source.
void vp9_output_skin_map(struct VP9_COMP *const cpi, FILE *yuv_skinmap_file);
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_ENCODER_VP9_SKIN_DETECTION_H_
