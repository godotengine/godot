/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP9_ENCODER_VP9_ENCODEMV_H_
#define VPX_VP9_ENCODER_VP9_ENCODEMV_H_

#include "vp9/encoder/vp9_encoder.h"

#ifdef __cplusplus
extern "C" {
#endif

void vp9_entropy_mv_init(void);

void vp9_write_nmv_probs(VP9_COMMON *cm, int usehp, vpx_writer *w,
                         nmv_context_counts *const counts);

void vp9_encode_mv(VP9_COMP *cpi, vpx_writer *w, const MV *mv, const MV *ref,
                   const nmv_context *mvctx, int usehp,
                   unsigned int *const max_mv_magnitude);

void vp9_build_nmv_cost_table(int *mvjoint, int *mvcost[2],
                              const nmv_context *ctx, int usehp);

void vp9_update_mv_count(ThreadData *td);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_ENCODER_VP9_ENCODEMV_H_
