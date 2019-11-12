/* Copyright (c) 2014, Cisco Systems, INC
   Written by XiangMingZhu WeiZhou MinPeng YanWang

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

#if !defined(X86CPU_H)
# define X86CPU_H

# if defined(OPUS_X86_MAY_HAVE_SSE)
#  define MAY_HAVE_SSE(name) name ## _sse
# else
#  define MAY_HAVE_SSE(name) name ## _c
# endif

# if defined(OPUS_X86_MAY_HAVE_SSE2)
#  define MAY_HAVE_SSE2(name) name ## _sse2
# else
#  define MAY_HAVE_SSE2(name) name ## _c
# endif

# if defined(OPUS_X86_MAY_HAVE_SSE4_1)
#  define MAY_HAVE_SSE4_1(name) name ## _sse4_1
# else
#  define MAY_HAVE_SSE4_1(name) name ## _c
# endif

# if defined(OPUS_X86_MAY_HAVE_AVX)
#  define MAY_HAVE_AVX(name) name ## _avx
# else
#  define MAY_HAVE_AVX(name) name ## _c
# endif

# if defined(OPUS_HAVE_RTCD)
int opus_select_arch(void);
# endif

/*gcc appears to emit MOVDQA's to load the argument of an _mm_cvtepi8_epi32()
  or _mm_cvtepi16_epi32() when optimizations are disabled, even though the
  actual PMOVSXWD instruction takes an m32 or m64. Unlike a normal memory
  reference, these require 16-byte alignment and load a full 16 bytes (instead
  of 4 or 8), possibly reading out of bounds.

  We can insert an explicit MOVD or MOVQ using _mm_cvtsi32_si128() or
  _mm_loadl_epi64(), which should have the same semantics as an m32 or m64
  reference in the PMOVSXWD instruction itself, but gcc is not smart enough to
  optimize this out when optimizations ARE enabled.

  Clang, in contrast, requires us to do this always for _mm_cvtepi8_epi32
  (which is fair, since technically the compiler is always allowed to do the
  dereference before invoking the function implementing the intrinsic).
  However, it is smart enough to eliminate the extra MOVD instruction.
  For _mm_cvtepi16_epi32, it does the right thing, though does *not* optimize out
  the extra MOVQ if it's specified explicitly */

# if defined(__clang__) || !defined(__OPTIMIZE__)
#  define OP_CVTEPI8_EPI32_M32(x) \
 (_mm_cvtepi8_epi32(_mm_cvtsi32_si128(*(int *)(x))))
# else
#  define OP_CVTEPI8_EPI32_M32(x) \
 (_mm_cvtepi8_epi32(*(__m128i *)(x)))
#endif

/* similar reasoning about the instruction sequence as in the 32-bit macro above,
 */
# if defined(__clang__) || !defined(__OPTIMIZE__)
#  define OP_CVTEPI16_EPI32_M64(x) \
 (_mm_cvtepi16_epi32(_mm_loadl_epi64((__m128i *)(x))))
# else
#  define OP_CVTEPI16_EPI32_M64(x) \
 (_mm_cvtepi16_epi32(*(__m128i *)(x)))
# endif

#endif
