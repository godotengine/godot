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

File modified for the Linux Kernel by
Zeev Tarantov <zeev.tarantov@gmail.com>

File modified for Sereal by
Steffen Mueller <smueller@cpan.org>
*/

#ifndef CSNAPPY_INTERNAL_H_
#define CSNAPPY_INTERNAL_H_

#include "csnappy_compat.h"

#ifndef __KERNEL__
#include "csnappy_internal_userspace.h"
#include <string.h>
#else

#include <linux/types.h>
#include <linux/string.h>
#include <linux/compiler.h>
#include <asm/byteorder.h>
#include <asm/unaligned.h>

#if (defined(__LITTLE_ENDIAN) && defined(__BIG_ENDIAN)) || \
    (!defined(__LITTLE_ENDIAN) && !defined(__BIG_ENDIAN))
#error either __LITTLE_ENDIAN or __BIG_ENDIAN must be defined
#endif
#if defined(__LITTLE_ENDIAN)
#define __BYTE_ORDER __LITTLE_ENDIAN
#else
#define __BYTE_ORDER __BIG_ENDIAN
#endif

#ifdef DEBUG
#define DCHECK(cond)	if (!(cond)) \
			printk(KERN_DEBUG "assert failed @ %s:%i\n", \
				__FILE__, __LINE__)
#else
#define DCHECK(cond)
#endif

#define UNALIGNED_LOAD16(_p)		get_unaligned((const uint16_t *)(_p))
#define UNALIGNED_LOAD32(_p)		get_unaligned((const uint32_t *)(_p))
#define UNALIGNED_LOAD64(_p)		get_unaligned((const uint64_t *)(_p))
#define UNALIGNED_STORE16(_p, _val)	put_unaligned((_val), (uint16_t *)(_p))
#define UNALIGNED_STORE32(_p, _val)	put_unaligned((_val), (uint32_t *)(_p))
#define UNALIGNED_STORE64(_p, _val)	put_unaligned((_val), (uint64_t *)(_p))

#define FindLSBSetNonZero(n)		__builtin_ctz(n)
#define FindLSBSetNonZero64(n)		__builtin_ctzll(n)

#endif /* __KERNEL__ */

#if (!defined(__LITTLE_ENDIAN) && !defined(__BIG_ENDIAN)) || ! defined(__BYTE_ORDER)
#  error either __LITTLE_ENDIAN or __BIG_ENDIAN, plus __BYTE_ORDER must be defined
#endif

#define ARCH_ARM_HAVE_UNALIGNED \
    defined(__ARM_ARCH_6__) || defined(__ARM_ARCH_6J__) || defined(__ARM_ARCH_6K__) || defined(__ARM_ARCH_6Z__) || defined(__ARM_ARCH_6ZK__) || defined(__ARM_ARCH_6T2__) || defined(__ARMV6__) || \
    defined(__ARM_ARCH_7__) || defined(__ARM_ARCH_7A__) || defined(__ARM_ARCH_7R__) || defined(__ARM_ARCH_7M__)

static INLINE void UnalignedCopy64(const void *src, void *dst) {
#if defined(__i386__) || defined(__x86_64__) || defined(__powerpc__) || ARCH_ARM_HAVE_UNALIGNED
  if ((sizeof(void *) == 8) || (sizeof(long) == 8)) {
    UNALIGNED_STORE64(dst, UNALIGNED_LOAD64(src));
  } else {
   /* This can be more efficient than UNALIGNED_LOAD64 + UNALIGNED_STORE64
      on some platforms, in particular ARM. */
    const uint8_t *src_bytep = (const uint8_t *)src;
    uint8_t *dst_bytep = (uint8_t *)dst;

    UNALIGNED_STORE32(dst_bytep, UNALIGNED_LOAD32(src_bytep));
    UNALIGNED_STORE32(dst_bytep + 4, UNALIGNED_LOAD32(src_bytep + 4));
  }
#else
  const uint8_t *src_bytep = (const uint8_t *)src;
  uint8_t *dst_bytep = (uint8_t *)dst;
  dst_bytep[0] = src_bytep[0];
  dst_bytep[1] = src_bytep[1];
  dst_bytep[2] = src_bytep[2];
  dst_bytep[3] = src_bytep[3];
  dst_bytep[4] = src_bytep[4];
  dst_bytep[5] = src_bytep[5];
  dst_bytep[6] = src_bytep[6];
  dst_bytep[7] = src_bytep[7];
#endif
}

#if defined(__arm__)
  #if ARCH_ARM_HAVE_UNALIGNED
     static INLINE uint32_t get_unaligned_le(const void *p, uint32_t n)
     {
       uint32_t wordmask = (1U << (8 * n)) - 1;
       return get_unaligned_le32(p) & wordmask;
     }
  #else
     extern uint32_t get_unaligned_le_armv5(const void *p, uint32_t n);
     #define get_unaligned_le get_unaligned_le_armv5
  #endif
#else
  static INLINE uint32_t get_unaligned_le(const void *p, uint32_t n)
  {
    /* Mapping from i in range [0,4] to a mask to extract the bottom 8*i bits */
    static const uint32_t wordmask[] = {
      0u, 0xffu, 0xffffu, 0xffffffu, 0xffffffffu
    };
    return get_unaligned_le32(p) & wordmask[n];
  }
#endif

#define DCHECK_EQ(a, b)	DCHECK(((a) == (b)))
#define DCHECK_NE(a, b)	DCHECK(((a) != (b)))
#define DCHECK_GT(a, b)	DCHECK(((a) >  (b)))
#define DCHECK_GE(a, b)	DCHECK(((a) >= (b)))
#define DCHECK_LT(a, b)	DCHECK(((a) <  (b)))
#define DCHECK_LE(a, b)	DCHECK(((a) <= (b)))

enum {
	LITERAL = 0,
	COPY_1_BYTE_OFFSET = 1,  /* 3 bit length + 3 bits of offset in opcode */
	COPY_2_BYTE_OFFSET = 2,
	COPY_4_BYTE_OFFSET = 3
};

#endif  /* CSNAPPY_INTERNAL_H_ */
