/*
 *  Copyright 2012 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "libyuv/basic_types.h"

#include "libyuv/compare_row.h"
#include "libyuv/row.h"

#if defined(_MSC_VER)
#include <intrin.h>  // For __popcnt
#endif

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// This module is for 32 bit Visual C x86
#if !defined(LIBYUV_DISABLE_X86) && defined(_MSC_VER) && defined(_M_IX86) && \
    (!defined(__clang__) || defined(LIBYUV_ENABLE_ROWWIN))

uint32_t HammingDistance_SSE42(const uint8_t* src_a,
                               const uint8_t* src_b,
                               int count) {
  uint32_t diff = 0u;

  int i;
  for (i = 0; i < count - 3; i += 4) {
    uint32_t x = *((uint32_t*)src_a) ^ *((uint32_t*)src_b);  // NOLINT
    src_a += 4;
    src_b += 4;
    diff += __popcnt(x);
  }
  return diff;
}

__declspec(naked) uint32_t
    SumSquareError_SSE2(const uint8_t* src_a, const uint8_t* src_b, int count) {
  __asm {
    mov        eax, [esp + 4]  // src_a
    mov        edx, [esp + 8]  // src_b
    mov        ecx, [esp + 12]  // count
    pxor       xmm0, xmm0
    pxor       xmm5, xmm5

  wloop:
    movdqu     xmm1, [eax]
    lea        eax,  [eax + 16]
    movdqu     xmm2, [edx]
    lea        edx,  [edx + 16]
    movdqa     xmm3, xmm1  // abs trick
    psubusb    xmm1, xmm2
    psubusb    xmm2, xmm3
    por        xmm1, xmm2
    movdqa     xmm2, xmm1
    punpcklbw  xmm1, xmm5
    punpckhbw  xmm2, xmm5
    pmaddwd    xmm1, xmm1
    pmaddwd    xmm2, xmm2
    paddd      xmm0, xmm1
    paddd      xmm0, xmm2
    sub        ecx, 16
    jg         wloop

    pshufd     xmm1, xmm0, 0xee
    paddd      xmm0, xmm1
    pshufd     xmm1, xmm0, 0x01
    paddd      xmm0, xmm1
    movd       eax, xmm0
    ret
  }
}

#ifdef HAS_SUMSQUAREERROR_AVX2
// C4752: found Intel(R) Advanced Vector Extensions; consider using /arch:AVX.
#pragma warning(disable : 4752)
__declspec(naked) uint32_t
    SumSquareError_AVX2(const uint8_t* src_a, const uint8_t* src_b, int count) {
  __asm {
    mov        eax, [esp + 4]  // src_a
    mov        edx, [esp + 8]  // src_b
    mov        ecx, [esp + 12]  // count
    vpxor      ymm0, ymm0, ymm0  // sum
    vpxor      ymm5, ymm5, ymm5  // constant 0 for unpck
    sub        edx, eax

  wloop:
    vmovdqu    ymm1, [eax]
    vmovdqu    ymm2, [eax + edx]
    lea        eax,  [eax + 32]
    vpsubusb   ymm3, ymm1, ymm2  // abs difference trick
    vpsubusb   ymm2, ymm2, ymm1
    vpor       ymm1, ymm2, ymm3
    vpunpcklbw ymm2, ymm1, ymm5  // u16.  mutates order.
    vpunpckhbw ymm1, ymm1, ymm5
    vpmaddwd   ymm2, ymm2, ymm2  // square + hadd to u32.
    vpmaddwd   ymm1, ymm1, ymm1
    vpaddd     ymm0, ymm0, ymm1
    vpaddd     ymm0, ymm0, ymm2
    sub        ecx, 32
    jg         wloop

    vpshufd    ymm1, ymm0, 0xee  // 3, 2 + 1, 0 both lanes.
    vpaddd     ymm0, ymm0, ymm1
    vpshufd    ymm1, ymm0, 0x01  // 1 + 0 both lanes.
    vpaddd     ymm0, ymm0, ymm1
    vpermq     ymm1, ymm0, 0x02  // high + low lane.
    vpaddd     ymm0, ymm0, ymm1
    vmovd      eax, xmm0
    vzeroupper
    ret
  }
}
#endif  // HAS_SUMSQUAREERROR_AVX2

uvec32 kHash16x33 = {0x92d9e201, 0, 0, 0};  // 33 ^ 16
uvec32 kHashMul0 = {
    0x0c3525e1,  // 33 ^ 15
    0xa3476dc1,  // 33 ^ 14
    0x3b4039a1,  // 33 ^ 13
    0x4f5f0981,  // 33 ^ 12
};
uvec32 kHashMul1 = {
    0x30f35d61,  // 33 ^ 11
    0x855cb541,  // 33 ^ 10
    0x040a9121,  // 33 ^ 9
    0x747c7101,  // 33 ^ 8
};
uvec32 kHashMul2 = {
    0xec41d4e1,  // 33 ^ 7
    0x4cfa3cc1,  // 33 ^ 6
    0x025528a1,  // 33 ^ 5
    0x00121881,  // 33 ^ 4
};
uvec32 kHashMul3 = {
    0x00008c61,  // 33 ^ 3
    0x00000441,  // 33 ^ 2
    0x00000021,  // 33 ^ 1
    0x00000001,  // 33 ^ 0
};

__declspec(naked) uint32_t
    HashDjb2_SSE41(const uint8_t* src, int count, uint32_t seed) {
  __asm {
    mov        eax, [esp + 4]  // src
    mov        ecx, [esp + 8]  // count
    movd       xmm0, [esp + 12]  // seed

    pxor       xmm7, xmm7  // constant 0 for unpck
    movdqa     xmm6, xmmword ptr kHash16x33

  wloop:
    movdqu     xmm1, [eax]  // src[0-15]
    lea        eax, [eax + 16]
    pmulld     xmm0, xmm6  // hash *= 33 ^ 16
    movdqa     xmm5, xmmword ptr kHashMul0
    movdqa     xmm2, xmm1
    punpcklbw  xmm2, xmm7  // src[0-7]
    movdqa     xmm3, xmm2
    punpcklwd  xmm3, xmm7  // src[0-3]
    pmulld     xmm3, xmm5
    movdqa     xmm5, xmmword ptr kHashMul1
    movdqa     xmm4, xmm2
    punpckhwd  xmm4, xmm7  // src[4-7]
    pmulld     xmm4, xmm5
    movdqa     xmm5, xmmword ptr kHashMul2
    punpckhbw  xmm1, xmm7  // src[8-15]
    movdqa     xmm2, xmm1
    punpcklwd  xmm2, xmm7  // src[8-11]
    pmulld     xmm2, xmm5
    movdqa     xmm5, xmmword ptr kHashMul3
    punpckhwd  xmm1, xmm7  // src[12-15]
    pmulld     xmm1, xmm5
    paddd      xmm3, xmm4  // add 16 results
    paddd      xmm1, xmm2
    paddd      xmm1, xmm3

    pshufd     xmm2, xmm1, 0x0e  // upper 2 dwords
    paddd      xmm1, xmm2
    pshufd     xmm2, xmm1, 0x01
    paddd      xmm1, xmm2
    paddd      xmm0, xmm1
    sub        ecx, 16
    jg         wloop

    movd       eax, xmm0  // return hash
    ret
  }
}

// Visual C 2012 required for AVX2.
#ifdef HAS_HASHDJB2_AVX2
__declspec(naked) uint32_t
    HashDjb2_AVX2(const uint8_t* src, int count, uint32_t seed) {
  __asm {
    mov        eax, [esp + 4]  // src
    mov        ecx, [esp + 8]  // count
    vmovd      xmm0, [esp + 12]  // seed

  wloop:
    vpmovzxbd  xmm3, [eax]  // src[0-3]
    vpmulld    xmm0, xmm0, xmmword ptr kHash16x33  // hash *= 33 ^ 16
    vpmovzxbd  xmm4, [eax + 4]  // src[4-7]
    vpmulld    xmm3, xmm3, xmmword ptr kHashMul0
    vpmovzxbd  xmm2, [eax + 8]  // src[8-11]
    vpmulld    xmm4, xmm4, xmmword ptr kHashMul1
    vpmovzxbd  xmm1, [eax + 12]  // src[12-15]
    vpmulld    xmm2, xmm2, xmmword ptr kHashMul2
    lea        eax, [eax + 16]
    vpmulld    xmm1, xmm1, xmmword ptr kHashMul3
    vpaddd     xmm3, xmm3, xmm4  // add 16 results
    vpaddd     xmm1, xmm1, xmm2
    vpaddd     xmm1, xmm1, xmm3
    vpshufd    xmm2, xmm1, 0x0e  // upper 2 dwords
    vpaddd     xmm1, xmm1,xmm2
    vpshufd    xmm2, xmm1, 0x01
    vpaddd     xmm1, xmm1, xmm2
    vpaddd     xmm0, xmm0, xmm1
    sub        ecx, 16
    jg         wloop

    vmovd      eax, xmm0  // return hash
    vzeroupper
    ret
  }
}
#endif  // HAS_HASHDJB2_AVX2

#endif  // !defined(LIBYUV_DISABLE_X86) && defined(_M_IX86)

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
