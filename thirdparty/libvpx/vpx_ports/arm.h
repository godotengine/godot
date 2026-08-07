/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_PORTS_ARM_H_
#define VPX_VPX_PORTS_ARM_H_
#include <stdlib.h>
#include "vpx_config.h"

#ifdef __cplusplus
extern "C" {
#endif

// Armv7-A optional Neon instructions, mandatory from Armv8.0-A.
#define HAS_NEON (1 << 0)
// Armv8.2-A optional Neon dot-product instructions, mandatory from Armv8.4-A.
#define HAS_NEON_DOTPROD (1 << 1)
// Armv8.2-A optional Neon i8mm instructions, mandatory from Armv8.6-A.
#define HAS_NEON_I8MM (1 << 2)
// Armv8.2-A optional SVE instructions, mandatory from Armv9.0-A.
#define HAS_SVE (1 << 3)
// Armv9.0-A SVE2 instructions.
#define HAS_SVE2 (1 << 4)

int arm_cpu_caps(void);

// Earlier gcc compilers have issues with some neon intrinsics
#if !defined(__clang__) && defined(__GNUC__) && __GNUC__ == 4 && \
    __GNUC_MINOR__ <= 6
#define VPX_INCOMPATIBLE_GCC
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VPX_PORTS_ARM_H_
