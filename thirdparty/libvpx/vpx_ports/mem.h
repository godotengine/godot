/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_PORTS_MEM_H_
#define VPX_VPX_PORTS_MEM_H_

#include "vpx_config.h"
#include "vpx/vpx_integer.h"

#if defined(__GNUC__) || defined(__SUNPRO_C)
#define DECLARE_ALIGNED(n, typ, val) typ val __attribute__((aligned(n)))
#elif defined(_MSC_VER)
#define DECLARE_ALIGNED(n, typ, val) __declspec(align(n)) typ val
#else
#warning No alignment directives known for this compiler.
#define DECLARE_ALIGNED(n, typ, val) typ val
#endif

#if defined(__has_builtin)
#define VPX_HAS_BUILTIN(x) __has_builtin(x)
#else
#define VPX_HAS_BUILTIN(x) 0
#endif

#if !VPX_HAS_BUILTIN(__builtin_prefetch) && !defined(__GNUC__)
#define __builtin_prefetch(x)
#endif

/* Shift down with rounding */
#define ROUND_POWER_OF_TWO(value, n) (((value) + (1 << ((n) - 1))) >> (n))
#define ROUND64_POWER_OF_TWO(value, n) (((value) + (1ULL << ((n) - 1))) >> (n))

#define ALIGN_POWER_OF_TWO(value, n) \
  (((value) + ((1 << (n)) - 1)) & ~((1 << (n)) - 1))

#define CONVERT_TO_SHORTPTR(x) ((uint16_t *)(((uintptr_t)(x)) << 1))
#define CAST_TO_SHORTPTR(x) ((uint16_t *)((uintptr_t)(x)))
#if CONFIG_VP9_HIGHBITDEPTH
#define CONVERT_TO_BYTEPTR(x) ((uint8_t *)(((uintptr_t)(x)) >> 1))
#define CAST_TO_BYTEPTR(x) ((uint8_t *)((uintptr_t)(x)))
#endif  // CONFIG_VP9_HIGHBITDEPTH

#endif  // VPX_VPX_PORTS_MEM_H_
