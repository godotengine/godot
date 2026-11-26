/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP8_COMMON_SYSTEMDEPENDENT_H_
#define VPX_VP8_COMMON_SYSTEMDEPENDENT_H_

#include "vpx_config.h"

#ifdef __cplusplus
extern "C" {
#endif

struct VP8Common;
void vp8_machine_specific_config(struct VP8Common *);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_COMMON_SYSTEMDEPENDENT_H_
