/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */


#ifndef VPX_PORTS_ARM_H_
#define VPX_PORTS_ARM_H_
#include <stdlib.h>
#include "vpx_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/*ARMv5TE "Enhanced DSP" instructions.*/
#define HAS_EDSP  0x01
/*ARMv6 "Parallel" or "Media" instructions.*/
#define HAS_MEDIA 0x02
/*ARMv7 optional NEON instructions.*/
#define HAS_NEON  0x04

int arm_cpu_caps(void);

// Earlier gcc compilers have issues with some neon intrinsics
#if !defined(__clang__) && defined(__GNUC__) && \
    __GNUC__ == 4 && __GNUC_MINOR__ <= 6
#define VPX_INCOMPATIBLE_GCC
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_PORTS_ARM_H_

