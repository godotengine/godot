/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP8_ENCODER_BITSTREAM_H_
#define VPX_VP8_ENCODER_BITSTREAM_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "vp8/encoder/treewriter.h"
#include "vp8/encoder/tokenize.h"

void vp8_pack_tokens(vp8_writer *w, const TOKENEXTRA *p, int xcount);
void vp8_convert_rfct_to_prob(struct VP8_COMP *const cpi);
void vp8_calc_ref_frame_costs(int *ref_frame_cost, int prob_intra,
                              int prob_last, int prob_garf);
int vp8_estimate_entropy_savings(struct VP8_COMP *cpi);
void vp8_update_coef_probs(struct VP8_COMP *cpi);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_ENCODER_BITSTREAM_H_
