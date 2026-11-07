/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP8_COMMON_RECONINTRA_H_
#define VPX_VP8_COMMON_RECONINTRA_H_

#include "vp8/common/blockd.h"

#ifdef __cplusplus
extern "C" {
#endif

void vp8_build_intra_predictors_mby_s(MACROBLOCKD *x, unsigned char *yabove_row,
                                      unsigned char *yleft, int left_stride,
                                      unsigned char *ypred_ptr, int y_stride);

void vp8_build_intra_predictors_mbuv_s(
    MACROBLOCKD *x, unsigned char *uabove_row, unsigned char *vabove_row,
    unsigned char *uleft, unsigned char *vleft, int left_stride,
    unsigned char *upred_ptr, unsigned char *vpred_ptr, int pred_stride);

void vp8_init_intra_predictors(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_COMMON_RECONINTRA_H_
