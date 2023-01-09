/**************************************************************************
 *
 * Copyright 2007-2008 VMware, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 * IN NO EVENT SHALL VMWARE AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/
#ifndef U_ENDIAN_H
#define U_ENDIAN_H

#ifdef HAVE_ENDIAN_H
#include <endian.h>

/* glibc */
#if defined(__BYTE_ORDER) && (__BYTE_ORDER == __LITTLE_ENDIAN)
# define UTIL_ARCH_LITTLE_ENDIAN 1
# define UTIL_ARCH_BIG_ENDIAN 0
#elif defined(__BYTE_ORDER) && (__BYTE_ORDER == __BIG_ENDIAN)
# define UTIL_ARCH_LITTLE_ENDIAN 0
# define UTIL_ARCH_BIG_ENDIAN 1
#endif

#if defined(BYTE_ORDER) && (BYTE_ORDER == LITTLE_ENDIAN)
# define UTIL_ARCH_LITTLE_ENDIAN 1
# define UTIL_ARCH_BIG_ENDIAN 0
#elif defined(BYTE_ORDER) && (BYTE_ORDER == BIG_ENDIAN)
# define UTIL_ARCH_LITTLE_ENDIAN 0
# define UTIL_ARCH_BIG_ENDIAN 1
#endif

#elif defined(__APPLE__)
#include <machine/endian.h>

#if __DARWIN_BYTE_ORDER == __DARWIN_LITTLE_ENDIAN
# define UTIL_ARCH_LITTLE_ENDIAN 1
# define UTIL_ARCH_BIG_ENDIAN 0
#elif __DARWIN_BYTE_ORDER == __DARWIN_BIG_ENDIAN
# define UTIL_ARCH_LITTLE_ENDIAN 0
# define UTIL_ARCH_BIG_ENDIAN 1
#endif

#elif defined(__sun)
#include <sys/isa_defs.h>

#if defined(_LITTLE_ENDIAN)
# define UTIL_ARCH_LITTLE_ENDIAN 1
# define UTIL_ARCH_BIG_ENDIAN 0
#elif defined(_BIG_ENDIAN)
# define UTIL_ARCH_LITTLE_ENDIAN 0
# define UTIL_ARCH_BIG_ENDIAN 1
#endif

#elif defined(__NetBSD__) || defined(__FreeBSD__) || \
      defined(__DragonFly__)
#include <sys/types.h>
#include <machine/endian.h>

#if _BYTE_ORDER == _LITTLE_ENDIAN
# define UTIL_ARCH_LITTLE_ENDIAN 1
# define UTIL_ARCH_BIG_ENDIAN 0
#elif _BYTE_ORDER == _BIG_ENDIAN
# define UTIL_ARCH_LITTLE_ENDIAN 0
# define UTIL_ARCH_BIG_ENDIAN 1
#endif

#elif defined(_WIN32) || defined(ANDROID)

#define UTIL_ARCH_LITTLE_ENDIAN 1
#define UTIL_ARCH_BIG_ENDIAN 0

#endif

#if !defined(UTIL_ARCH_LITTLE_ENDIAN) || !defined(UTIL_ARCH_BIG_ENDIAN)
# error "UTIL_ARCH_LITTLE_ENDIAN and/or UTIL_ARCH_BIG_ENDIAN were unset."
#elif UTIL_ARCH_LITTLE_ENDIAN == UTIL_ARCH_BIG_ENDIAN
# error "UTIL_ARCH_LITTLE_ENDIAN and UTIL_ARCH_BIG_ENDIAN must not both be 1 or 0."
#endif

#endif
