/*
Copyright 2011 Google Inc. All Rights Reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the
distribution.
    * Neither the name of Google Inc. nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Various stubs for the open-source version of Snappy.

File modified by
Zeev Tarantov <zeev.tarantov@gmail.com>

File modified for Sereal by
Steffen Mueller <smueller@cpan.org>
*/

#ifndef CSNAPPY_INTERNAL_USERSPACE_H_
#define CSNAPPY_INTERNAL_USERSPACE_H_

/*note the original version of this file checked for MS version, but MS will *never* support
 * anything but C89, so the version check is bogus. */
#if defined(_MSC_VER)
typedef unsigned __int8  uint8_t;
typedef unsigned __int16 uint16_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
typedef __int32 int32_t; /* Sereal specific change, see csnappy_decompress.c(271) : error C2065: 'int32_t' : undeclared identifier */
/* the following define is Sereal specific, as MS C89 compilers do not know about "inline" */
#define inline __inline
#ifdef _M_X64
#  define __x86_64__
#  define __x86_64
#  define __amd64__
#  define __amd64
#endif
#ifdef _M_IX86
#  define __i386__
#  define __i386
#  define i386
#  define _X86_
#endif
#ifdef _M_IA64
#  define __ia64__
#  define __ia64
#  define __IA64__
#  define __itanium__
#endif

#else
#include <stdint.h>
#endif

#ifdef _GNU_SOURCE
#define min(x, y) (__extension__ ({		\
	typeof(x) _min1 = (x);			\
	typeof(y) _min2 = (y);			\
	(void) (&_min1 == &_min2);		\
	_min1 < _min2 ? _min1 : _min2; }))
#else
#define min(x, y) (((x) < (y)) ? (x) : (y))
#endif

/* Static prediction hints. */
#ifndef __GNUC__
#define __builtin_expect(a,b) a
#endif
#define likely(x)	__builtin_expect(!!(x), 1)
#define unlikely(x)	__builtin_expect(!!(x), 0)


#ifdef DEBUG
#include <assert.h>
#define DCHECK(cond)	assert(cond)
#else
#define DCHECK(cond)
#endif

#include "csnappy_compat.h"

/*
Uses code from http://code.google.com/p/exfat/source/browse/trunk/libexfat/byteorder.h
with 3-clause BSD license instead of GPL, with permission from:
Andrew Nayenko
Albert Lee
*/
#if defined(_MSC_VER)

#include <stdlib.h>
#define bswap_16(x) _byteswap_ushort(x)
#define bswap_32(x) _byteswap_ulong(x)
#define bswap_64(x) _byteswap_uint64(x)
#define __BIG_ENDIAN	4321
#define __LITTLE_ENDIAN	1234
#define __BYTE_ORDER	LITTLE_ENDIAN

#elif defined(__GLIBC__) || defined(__ANDROID__) || defined(__CYGWIN__)

#include <endian.h>
#include <byteswap.h>

#elif defined(__APPLE__)

#include <machine/endian.h>
#include <libkern/OSByteOrder.h>
#define bswap_16(x) OSSwapInt16(x)
#define bswap_32(x) OSSwapInt32(x)
#define bswap_64(x) OSSwapInt64(x)
#define __BYTE_ORDER BYTE_ORDER
#define __LITTLE_ENDIAN LITTLE_ENDIAN
#define __BIG_ENDIAN BIG_ENDIAN

#elif defined(__FreeBSD__) || defined(__DragonFly__) || defined(__NetBSD__)

#include <sys/endian.h>
#define bswap_16(x) bswap16(x)
#define bswap_32(x) bswap32(x)
#define bswap_64(x) bswap64(x)
#define __BYTE_ORDER _BYTE_ORDER
#define __LITTLE_ENDIAN _LITTLE_ENDIAN
#define __BIG_ENDIAN _BIG_ENDIAN

#elif defined(__OpenBSD__)

#include <machine/endian.h>
#define bswap_16(x) swap16(x)
#define bswap_32(x) swap32(x)
#define bswap_64(x) swap64(x)
#define __BYTE_ORDER _BYTE_ORDER
#define __LITTLE_ENDIAN _LITTLE_ENDIAN
#define __BIG_ENDIAN _BIG_ENDIAN

#elif defined(__sun)

#include <sys/byteorder.h>
#define bswap_16(x) BSWAP_16(x)
#define bswap_32(x) BSWAP_32(x)
#define bswap_64(x) BSWAP_64(x)
#define __LITTLE_ENDIAN 1234
#define __BIG_ENDIAN 4321
#ifdef _LITTLE_ENDIAN
#define __BYTE_ORDER __LITTLE_ENDIAN
#else
#define __BYTE_ORDER __BIG_ENDIAN
#endif

#elif defined(__MINGW32__)
#include <sys/param.h>
#define __BYTE_ORDER BYTE_ORDER
#define __LITTLE_ENDIAN LITTLE_ENDIAN
#define __BIG_ENDIAN BIG_ENDIAN

#endif


/* Potentially unaligned loads and stores. */

#if defined(__i386__) || defined(__x86_64__) || defined(__powerpc__)

#define UNALIGNED_LOAD16(_p) (*(const uint16_t*)(_p))
#define UNALIGNED_LOAD32(_p) (*(const uint32_t*)(_p))
#define UNALIGNED_LOAD64(_p) (*(const uint64_t*)(_p))

#define UNALIGNED_STORE16(_p, _val) (*(uint16_t*)(_p) = (_val))
#define UNALIGNED_STORE32(_p, _val) (*(uint32_t*)(_p) = (_val))
#define UNALIGNED_STORE64(_p, _val) (*(uint64_t*)(_p) = (_val))

#elif defined(__arm__) && \
	!defined(__ARM_ARCH_4__) && \
	!defined(__ARM_ARCH_4T__) && /* http://wiki.debian.org/ArmEabiPort#Choice_of_minimum_CPU */ \
	!defined(__MARM_ARMV4__) && \
	!defined(_ARMV4I_) && \
	!defined(__ARM_ARCH_5__) && \
	!defined(__ARM_ARCH_5T__) && \
	!defined(__ARM_ARCH_5E__) && \
	!defined(__ARM_ARCH_5TE__) && \
	!defined(__ARM_ARCH_5TEJ__) && \
	!defined(__MARM_ARMV5__)

#define UNALIGNED_LOAD16(_p) (*(const uint16_t*)(_p))
#define UNALIGNED_LOAD32(_p) (*(const uint32_t*)(_p))
#define UNALIGNED_STORE16(_p, _val) (*(uint16_t*)(_p) = (_val))
#define UNALIGNED_STORE32(_p, _val) (*(uint32_t*)(_p) = (_val))

#pragma pack(1)
struct una_u64 { uint64_t x; };
#pragma pack()

static INLINE uint64_t UNALIGNED_LOAD64(const void *p)
{
	const struct una_u64 *ptr = (const struct una_u64 *)p;
	return ptr->x;
}

static INLINE void UNALIGNED_STORE64(void *p, uint64_t v)
{
	struct una_u64 *ptr = (struct una_u64 *)p;
	ptr->x = v;
}

#else /* !(x86 || powerpc) && !(arm && !(old arm architectures)) */

#pragma pack(1)
struct una_u16 { uint16_t x; };
struct una_u32 { uint32_t x; };
struct una_u64 { uint64_t x; };
#pragma pack()

static INLINE uint16_t UNALIGNED_LOAD16(const void *p)
{
	const struct una_u16 *ptr = (const struct una_u16 *)p;
	return ptr->x;
}

static INLINE uint32_t UNALIGNED_LOAD32(const void *p)
{
	const struct una_u32 *ptr = (const struct una_u32 *)p;
	return ptr->x;
}

static INLINE uint64_t UNALIGNED_LOAD64(const void *p)
{
	const struct una_u64 *ptr = (const struct una_u64 *)p;
	return ptr->x;
}

static INLINE void UNALIGNED_STORE16(void *p, uint16_t v)
{
	struct una_u16 *ptr = (struct una_u16 *)p;
	ptr->x = v;
}

static INLINE void UNALIGNED_STORE32(void *p, uint32_t v)
{
	struct una_u32 *ptr = (struct una_u32 *)p;
	ptr->x = v;
}

static INLINE void UNALIGNED_STORE64(void *p, uint64_t v)
{
	struct una_u64 *ptr = (struct una_u64 *)p;
	ptr->x = v;
}

#endif /* !(x86 || powerpc) && !(arm && !armv5 && !armv6) */


#if __BYTE_ORDER == __LITTLE_ENDIAN
#define get_unaligned_le32(p)           UNALIGNED_LOAD32(p)
#define put_unaligned_le16(v, p)        UNALIGNED_STORE16(p, v)
#elif __BYTE_ORDER == __BIG_ENDIAN
static INLINE uint32_t get_unaligned_le32(const void *p)
{
  return bswap_32(UNALIGNED_LOAD32(p));
}
static INLINE void put_unaligned_le16(uint16_t val, void *p)
{
  UNALIGNED_STORE16(p, bswap_16(val));
}
#else
static INLINE uint32_t get_unaligned_le32(const void *p)
{
  const uint8_t *b = (const uint8_t *)p;
  return b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24);
}
static INLINE void put_unaligned_le16(uint16_t val, void *p)
{
  uint8_t *b = (uint8_t *)p;
  b[0] = val & 255;
  b[1] = val >> 8;
}
#endif


#if defined(HAVE_BUILTIN_CTZ)

static INLINE int FindLSBSetNonZero(uint32_t n)
{
	return __builtin_ctz(n);
}

static INLINE int FindLSBSetNonZero64(uint64_t n)
{
	return __builtin_ctzll(n);
}

#else /* Portable versions. */

static INLINE int FindLSBSetNonZero(uint32_t n)
{
	int rc = 31, i, shift;
	uint32_t x;
	for (i = 4, shift = 1 << 4; i >= 0; --i) {
		x = n << shift;
		if (x != 0) {
			n = x;
			rc -= shift;
		}
		shift >>= 1;
	}
	return rc;
}

/* FindLSBSetNonZero64() is defined in terms of FindLSBSetNonZero(). */
static INLINE int FindLSBSetNonZero64(uint64_t n)
{
	const uint32_t bottombits = (uint32_t)n;
	if (bottombits == 0) {
		/* Bottom bits are zero, so scan in top bits */
		return 32 + FindLSBSetNonZero((uint32_t)(n >> 32));
	} else {
		return FindLSBSetNonZero(bottombits);
	}
}

#endif /* End portable versions. */

#endif  /* CSNAPPY_INTERNAL_USERSPACE_H_ */
