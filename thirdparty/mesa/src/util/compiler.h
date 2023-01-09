/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 1999-2008  Brian Paul   All Rights Reserved.
 * Copyright (C) 2009  VMware, Inc.  All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */


/**
 * \file compiler.h
 * Compiler-related stuff.
 */


#ifndef COMPILER_H
#define COMPILER_H


#include <assert.h>

#include "util/macros.h"
#include "util/u_endian.h"
#include "util/detect_arch.h"

/**
 * Define CPU_TO_LE32
 * Do not use these unless absolutely necessary!
 * Try to use a runtime test instead.
 * For now, only used by some DRI hardware drivers for color/texel packing.
 */
#if UTIL_ARCH_BIG_ENDIAN
#if defined(__linux__)
#include <byteswap.h>
#define CPU_TO_LE32( x ) bswap_32( x )
#elif defined(__APPLE__)
#include <CoreFoundation/CFByteOrder.h>
#define CPU_TO_LE32( x ) CFSwapInt32HostToLittle( x )
#elif defined(__OpenBSD__)
#include <sys/types.h>
#define CPU_TO_LE32( x ) htole32( x )
#else /*__linux__ */
#include <sys/endian.h>
#define CPU_TO_LE32( x ) bswap32( x )
#endif /*__linux__*/
#else
#define CPU_TO_LE32( x ) ( x )
#endif
#define LE32_TO_CPU( x ) CPU_TO_LE32( x )


/* Macro for stack alignment. */
#if defined(__GNUC__) && DETECT_ARCH_X86
#define UTIL_ALIGN_STACK __attribute__((force_align_arg_pointer))
#else
#define UTIL_ALIGN_STACK
#endif

#define IEEE_ONE 0x3f800000

#ifndef __has_attribute
#  define __has_attribute(x) 0
#endif

#if defined(__has_cpp_attribute) && defined(__clang__)
/* We do not do the same trick as __has_attribute because parsing
 * clang::fallthrough in the preprocessor fails in GCC. */
#  define HAS_CLANG_FALLTHROUGH  __has_cpp_attribute(clang::fallthrough)
#else
#  define HAS_CLANG_FALLTHROUGH 0
#endif

#if (defined(__cplusplus) && (__cplusplus >= 201703L)) || \
    (defined(__STDC_VERSION__) && (__STDC_VERSION__ > 201710L))
/* Standard C++17/C23 attribute */
#define FALLTHROUGH [[fallthrough]]
#elif HAS_CLANG_FALLTHROUGH
/* Clang++ specific */
#define FALLTHROUGH [[clang::fallthrough]]
#elif __has_attribute(fallthrough)
/* Non-standard but supported by at least gcc and clang */
#define FALLTHROUGH __attribute__((fallthrough))
#else
#define FALLTHROUGH do { } while(0)
#endif

#endif /* COMPILER_H */
