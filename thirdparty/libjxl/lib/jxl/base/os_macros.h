// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_BASE_OS_MACROS_H_
#define LIB_JXL_BASE_OS_MACROS_H_

// Defines the JXL_OS_* macros.

#if defined(_WIN32) || defined(_WIN64)
#define JXL_OS_WIN 1
#else
#define JXL_OS_WIN 0
#endif

#ifdef __linux__
#define JXL_OS_LINUX 1
#else
#define JXL_OS_LINUX 0
#endif

#ifdef __APPLE__
#define JXL_OS_MAC 1
#else
#define JXL_OS_MAC 0
#endif

#define JXL_OS_IOS 0
#ifdef __APPLE__
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#undef JXL_OS_IOS
#define JXL_OS_IOS 1
#endif
#endif

#ifdef __FreeBSD__
#define JXL_OS_FREEBSD 1
#else
#define JXL_OS_FREEBSD 0
#endif

#ifdef __HAIKU__
#define JXL_OS_HAIKU 1
#else
#define JXL_OS_HAIKU 0
#endif

#endif  // LIB_JXL_BASE_OS_MACROS_H_
