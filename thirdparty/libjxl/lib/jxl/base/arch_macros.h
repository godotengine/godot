// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_BASE_ARCH_MACROS_H_
#define LIB_JXL_BASE_ARCH_MACROS_H_

// Defines the JXL_ARCH_* macros.

namespace jxl {

#if defined(__x86_64__) || defined(_M_X64)
#define JXL_ARCH_X64 1
#else
#define JXL_ARCH_X64 0
#endif

#if defined(__powerpc64__) || defined(_M_PPC)
#define JXL_ARCH_PPC 1
#else
#define JXL_ARCH_PPC 0
#endif

#if defined(__aarch64__) || defined(__arm__)
#define JXL_ARCH_ARM 1
#else
#define JXL_ARCH_ARM 0
#endif

}  // namespace jxl

#endif  // LIB_JXL_BASE_ARCH_MACROS_H_
