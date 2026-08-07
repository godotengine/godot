/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP8_ENCODER_PICKINTER_H_
#define VPX_VP8_ENCODER_PICKINTER_H_
#include "vpx_config.h"
#include "vp8/common/onyxc_int.h"

#ifdef __cplusplus
extern "C" {
#endif

extern void vp8_pick_inter_mode(VP8_COMP *cpi, MACROBLOCK *x, int recon_yoffset,
                                int recon_uvoffset, int *returnrate,
                                int *returndistortion, int *returnintra,
                                int mb_row, int mb_col);
extern void vp8_pick_intra_mode(MACROBLOCK *x, int *rate);

extern int vp8_get_inter_mbpred_error(MACROBLOCK *mb,
                                      const vp8_variance_fn_ptr_t *vfp,
                                      unsigned int *sse, int_mv this_mv);
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_ENCODER_PICKINTER_H_
