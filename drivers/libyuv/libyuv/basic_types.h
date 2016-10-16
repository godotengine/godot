/*
 *  Copyright 2011 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef INCLUDE_LIBYUV_BASIC_TYPES_H_  // NOLINT
#define INCLUDE_LIBYUV_BASIC_TYPES_H_

#include <stddef.h>  // for NULL, size_t

#if defined(_MSC_VER) && (_MSC_VER < 1600)
#include <sys/types.h>  // for uintptr_t on x86
#else
#include <stdint.h>  // for uintptr_t
#endif

#ifndef GG_LONGLONG
#ifndef INT_TYPES_DEFINED
#define INT_TYPES_DEFINED
#ifdef COMPILER_MSVC
typedef unsigned __int64 uint64;
typedef __int64 int64;
#ifndef INT64_C
#define INT64_C(x) x ## I64
#endif
#ifndef UINT64_C
#define UINT64_C(x) x ## UI64
#endif
#define INT64_F "I64"
#else  // COMPILER_MSVC
#if defined(__LP64__) && !defined(__OpenBSD__) && !defined(__APPLE__)
typedef unsigned long uint64;  // NOLINT
typedef long int64;  // NOLINT
#ifndef INT64_C
#define INT64_C(x) x ## L
#endif
#ifndef UINT64_C
#define UINT64_C(x) x ## UL
#endif
#define INT64_F "l"
#else  // defined(__LP64__) && !defined(__OpenBSD__) && !defined(__APPLE__)
typedef unsigned long long uint64;  // NOLINT
typedef long long int64;  // NOLINT
#ifndef INT64_C
#define INT64_C(x) x ## LL
#endif
#ifndef UINT64_C
#define UINT64_C(x) x ## ULL
#endif
#define INT64_F "ll"
#endif  // __LP64__
#endif  // COMPILER_MSVC
typedef unsigned int uint32;
typedef int int32;
typedef unsigned short uint16;  // NOLINT
typedef short int16;  // NOLINT
typedef unsigned char uint8;
typedef signed char int8;
#endif  // INT_TYPES_DEFINED
#endif  // GG_LONGLONG

// Detect compiler is for x86 or x64.
#if defined(__x86_64__) || defined(_M_X64) || \
    defined(__i386__) || defined(_M_IX86)
#define CPU_X86 1
#endif
// Detect compiler is for ARM.
#if defined(__arm__) || defined(_M_ARM)
#define CPU_ARM 1
#endif

#ifndef ALIGNP
#ifdef __cplusplus
#define ALIGNP(p, t) \
    (reinterpret_cast<uint8*>(((reinterpret_cast<uintptr_t>(p) + \
    ((t) - 1)) & ~((t) - 1))))
#else
#define ALIGNP(p, t) \
    ((uint8*)((((uintptr_t)(p) + ((t) - 1)) & ~((t) - 1))))  /* NOLINT */
#endif
#endif

#if !defined(LIBYUV_API)
#if defined(_WIN32) || defined(__CYGWIN__)
#if defined(LIBYUV_BUILDING_SHARED_LIBRARY)
#define LIBYUV_API __declspec(dllexport)
#elif defined(LIBYUV_USING_SHARED_LIBRARY)
#define LIBYUV_API __declspec(dllimport)
#else
#define LIBYUV_API
#endif  // LIBYUV_BUILDING_SHARED_LIBRARY
#elif defined(__GNUC__) && (__GNUC__ >= 4) && !defined(__APPLE__) && \
    (defined(LIBYUV_BUILDING_SHARED_LIBRARY) || \
    defined(LIBYUV_USING_SHARED_LIBRARY))
#define LIBYUV_API __attribute__ ((visibility ("default")))
#else
#define LIBYUV_API
#endif  // __GNUC__
#endif  // LIBYUV_API

#define LIBYUV_BOOL int
#define LIBYUV_FALSE 0
#define LIBYUV_TRUE 1

// Visual C x86 or GCC little endian.
#if defined(__x86_64__) || defined(_M_X64) || \
  defined(__i386__) || defined(_M_IX86) || \
  defined(__arm__) || defined(_M_ARM) || \
  (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#define LIBYUV_LITTLE_ENDIAN
#endif

#endif  // INCLUDE_LIBYUV_BASIC_TYPES_H_  NOLINT
