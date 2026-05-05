/*
 *  Copyright (c) 2023 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <stdlib.h>
#include <string.h>

#include "vpx_config.h"
#include "vpx_ports/arm.h"

#if defined(_WIN32)
#undef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#undef WIN32_EXTRA_LEAN
#define WIN32_EXTRA_LEAN
#include <windows.h>
#endif

#ifdef WINAPI_FAMILY
#include <winapifamily.h>
#if !WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)
#define getenv(x) NULL
#endif
#endif

#if defined(__ANDROID__) && (__ANDROID_API__ < 18)
#define VPX_USE_ANDROID_CPU_FEATURES 1
// Use getauxval() when targeting (64-bit) Android with API level >= 18.
// getauxval() is supported since Android API level 18 (Android 4.3.)
// First Android version with 64-bit support was Android 5.x (API level 21).
#include <cpu-features.h>
#endif

static INLINE int arm_cpu_env_flags(int *flags) {
  const char *env = getenv("VPX_SIMD_CAPS");
  if (env && *env) {
    *flags = (int)strtol(env, NULL, 0);
    return 1;
  }
  return 0;
}

static INLINE int arm_cpu_env_mask(void) {
  const char *env = getenv("VPX_SIMD_CAPS_MASK");
  return env && *env ? (int)strtol(env, NULL, 0) : ~0;
}
