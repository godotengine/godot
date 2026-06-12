/* Copyright (c) 2003-2008 Timothy B. Terriberry
   Copyright (c) 2008 Xiph.Org Foundation */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*Some common macros for potential platform-specific optimization.*/
#include "opus_types.h"
#include <math.h>
#include <limits.h>
#include "arch.h"
#if !defined(_ecintrin_H)
# define _ecintrin_H (1)

/*Some specific platforms may have optimized intrinsic or OPUS_INLINE assembly
   versions of these functions which can substantially improve performance.
  We define macros for them to allow easy incorporation of these non-ANSI
   features.*/

/*Modern gcc (4.x) can compile the naive versions of min and max with cmov if
   given an appropriate architecture, but the branchless bit-twiddling versions
   are just as fast, and do not require any special target architecture.
  Earlier gcc versions (3.x) compiled both code to the same assembly
   instructions, because of the way they represented ((_b)>(_a)) internally.*/
# define EC_MINI(_a,_b)      ((_a)+(((_b)-(_a))&-((_b)<(_a))))

/*Count leading zeros.
  This macro should only be used for implementing ec_ilog(), if it is defined.
  All other code should use EC_ILOG() instead.*/
#if defined(_MSC_VER) && (_MSC_VER >= 1400)
#if defined(_MSC_VER) && (_MSC_VER >= 1910)
# include <intrin0.h> /* Improve compiler throughput. */
#else
# include <intrin.h>
#endif
/*In _DEBUG mode this is not an intrinsic by default.*/
# pragma intrinsic(_BitScanReverse)

static __inline int ec_bsr(unsigned long _x){
  unsigned long ret;
  _BitScanReverse(&ret,_x);
  return (int)ret;
}
# define EC_CLZ0    (1)
# define EC_CLZ(_x) (-ec_bsr(_x))
#elif defined(ENABLE_TI_DSPLIB)
# include "dsplib.h"
# define EC_CLZ0    (31)
# define EC_CLZ(_x) (_lnorm(_x))
#elif __GNUC_PREREQ(3,4)
# if INT_MAX>=2147483647
#  define EC_CLZ0    ((int)sizeof(unsigned)*CHAR_BIT)
#  define EC_CLZ(_x) (__builtin_clz(_x))
# elif LONG_MAX>=2147483647L
#  define EC_CLZ0    ((int)sizeof(unsigned long)*CHAR_BIT)
#  define EC_CLZ(_x) (__builtin_clzl(_x))
# endif
#endif

#if defined(EC_CLZ)
/*Note that __builtin_clz is not defined when _x==0, according to the gcc
   documentation (and that of the BSR instruction that implements it on x86).
  The majority of the time we can never pass it zero.
  When we need to, it can be special cased.*/
# define EC_ILOG(_x) (EC_CLZ0-EC_CLZ(_x))
#else
int ec_ilog(opus_uint32 _v);
# define EC_ILOG(_x) (ec_ilog(_x))
#endif
#endif
