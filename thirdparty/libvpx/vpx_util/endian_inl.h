// Copyright 2014 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Endian related functions.

#ifndef VPX_UTIL_ENDIAN_INL_H_
#define VPX_UTIL_ENDIAN_INL_H_

#include <stdlib.h>
#include "./vpx_config.h"
#include "vpx/vpx_integer.h"

#if defined(__GNUC__)
# define LOCAL_GCC_VERSION ((__GNUC__ << 8) | __GNUC_MINOR__)
# define LOCAL_GCC_PREREQ(maj, min) \
    (LOCAL_GCC_VERSION >= (((maj) << 8) | (min)))
#else
# define LOCAL_GCC_VERSION 0
# define LOCAL_GCC_PREREQ(maj, min) 0
#endif

// handle clang compatibility
#ifndef __has_builtin
# define __has_builtin(x) 0
#endif

// some endian fix (e.g.: mips-gcc doesn't define __BIG_ENDIAN__)
#if !defined(WORDS_BIGENDIAN) && \
    (defined(__BIG_ENDIAN__) || defined(_M_PPC) || \
     (defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)))
#define WORDS_BIGENDIAN
#endif

#if defined(WORDS_BIGENDIAN)
#define HToLE32 BSwap32
#define HToLE16 BSwap16
#define HToBE64(x) (x)
#define HToBE32(x) (x)
#else
#define HToLE32(x) (x)
#define HToLE16(x) (x)
#define HToBE64(X) BSwap64(X)
#define HToBE32(X) BSwap32(X)
#endif

#if LOCAL_GCC_PREREQ(4, 8) || __has_builtin(__builtin_bswap16)
#define HAVE_BUILTIN_BSWAP16
#endif

#if LOCAL_GCC_PREREQ(4, 3) || __has_builtin(__builtin_bswap32)
#define HAVE_BUILTIN_BSWAP32
#endif

#if LOCAL_GCC_PREREQ(4, 3) || __has_builtin(__builtin_bswap64)
#define HAVE_BUILTIN_BSWAP64
#endif

#if HAVE_MIPS32 && defined(__mips__) && !defined(__mips64) && \
    defined(__mips_isa_rev) && (__mips_isa_rev >= 2) && (__mips_isa_rev < 6)
#define VPX_USE_MIPS32_R2
#endif

static INLINE uint16_t BSwap16(uint16_t x) {
#if defined(HAVE_BUILTIN_BSWAP16)
  return __builtin_bswap16(x);
#elif defined(_MSC_VER)
  return _byteswap_ushort(x);
#else
  // gcc will recognize a 'rorw $8, ...' here:
  return (x >> 8) | ((x & 0xff) << 8);
#endif  // HAVE_BUILTIN_BSWAP16
}

static INLINE uint32_t BSwap32(uint32_t x) {
#if defined(VPX_USE_MIPS32_R2)
  uint32_t ret;
  __asm__ volatile (
    "wsbh   %[ret], %[x]          \n\t"
    "rotr   %[ret], %[ret],  16   \n\t"
    : [ret]"=r"(ret)
    : [x]"r"(x)
  );
  return ret;
#elif defined(HAVE_BUILTIN_BSWAP32)
  return __builtin_bswap32(x);
#elif defined(__i386__) || defined(__x86_64__)
  uint32_t swapped_bytes;
  __asm__ volatile("bswap %0" : "=r"(swapped_bytes) : "0"(x));
  return swapped_bytes;
#elif defined(_MSC_VER)
  return (uint32_t)_byteswap_ulong(x);
#else
  return (x >> 24) | ((x >> 8) & 0xff00) | ((x << 8) & 0xff0000) | (x << 24);
#endif  // HAVE_BUILTIN_BSWAP32
}

static INLINE uint64_t BSwap64(uint64_t x) {
#if defined(HAVE_BUILTIN_BSWAP64)
  return __builtin_bswap64(x);
#elif defined(__x86_64__)
  uint64_t swapped_bytes;
  __asm__ volatile("bswapq %0" : "=r"(swapped_bytes) : "0"(x));
  return swapped_bytes;
#elif defined(_MSC_VER)
  return (uint64_t)_byteswap_uint64(x);
#else  // generic code for swapping 64-bit values (suggested by bdb@)
  x = ((x & 0xffffffff00000000ull) >> 32) | ((x & 0x00000000ffffffffull) << 32);
  x = ((x & 0xffff0000ffff0000ull) >> 16) | ((x & 0x0000ffff0000ffffull) << 16);
  x = ((x & 0xff00ff00ff00ff00ull) >>  8) | ((x & 0x00ff00ff00ff00ffull) <<  8);
  return x;
#endif  // HAVE_BUILTIN_BSWAP64
}

#endif  // VPX_UTIL_ENDIAN_INL_H_
