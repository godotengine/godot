/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_PORTS_PPC_H_
#define VPX_VPX_PORTS_PPC_H_
#include <stdlib.h>

#include "./vpx_config.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HAS_VSX 0x01

int ppc_simd_caps(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VPX_PORTS_PPC_H_
