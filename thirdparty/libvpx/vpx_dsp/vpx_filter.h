/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_DSP_VPX_FILTER_H_
#define VPX_DSP_VPX_FILTER_H_

#include "vpx/vpx_integer.h"


#ifdef __cplusplus
extern "C" {
#endif

#define FILTER_BITS 7

#define SUBPEL_BITS 4
#define SUBPEL_MASK ((1 << SUBPEL_BITS) - 1)
#define SUBPEL_SHIFTS (1 << SUBPEL_BITS)
#define SUBPEL_TAPS 8

typedef int16_t InterpKernel[SUBPEL_TAPS];

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_DSP_VPX_FILTER_H_
