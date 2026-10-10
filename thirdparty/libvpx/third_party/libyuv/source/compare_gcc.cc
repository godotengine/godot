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

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// This module is for GCC x86 and x64.
#if !defined(LIBYUV_DISABLE_X86) && \
    (defined(__x86_64__) || (defined(__i386__) && !defined(_MSC_VER)))

#if defined(__x86_64__)
uint32_t HammingDistance_SSE42(const uint8_t* src_a,
                               const uint8_t* src_b,
                               int count) {
  uint64_t diff = 0u;

  asm volatile(
      "xor        %3,%3                          \n"
      "xor        %%r8,%%r8                      \n"
      "xor        %%r9,%%r9                      \n"
      "xor        %%r10,%%r10                    \n"

      // Process 32 bytes per loop.
      LABELALIGN
      "1:                                        \n"
      "mov        (%0),%%rcx                     \n"
      "mov        0x8(%0),%%rdx                  \n"
      "xor        (%1),%%rcx                     \n"
      "xor        0x8(%1),%%rdx                  \n"
      "popcnt     %%rcx,%%rcx                    \n"
      "popcnt     %%rdx,%%rdx                    \n"
      "mov        0x10(%0),%%rsi                 \n"
      "mov        0x18(%0),%%rdi                 \n"
      "xor        0x10(%1),%%rsi                 \n"
      "xor        0x18(%1),%%rdi                 \n"
      "popcnt     %%rsi,%%rsi                    \n"
      "popcnt     %%rdi,%%rdi                    \n"
      "add        $0x20,%0                       \n"
      "add        $0x20,%1                       \n"
      "add        %%rcx,%3                       \n"
      "add        %%rdx,%%r8                     \n"
      "add        %%rsi,%%r9                     \n"
      "add        %%rdi,%%r10                    \n"
      "sub        $0x20,%2                       \n"
      "jg         1b                             \n"

      "add        %%r8, %3                       \n"
      "add        %%r9, %3                       \n"
      "add        %%r10, %3                      \n"
      : "+r"(src_a),  // %0
        "+r"(src_b),  // %1
        "+r"(count),  // %2
        "=r"(diff)    // %3
      :
      : "memory", "cc", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10");

  return static_cast<uint32_t>(diff);
}
#else
uint32_t HammingDistance_SSE42(const uint8_t* src_a,
                               const uint8_t* src_b,
                               int count) {
  uint32_t diff = 0u;

  asm volatile(
      // Process 16 bytes per loop.
      LABELALIGN
      "1:                                        \n"
      "mov        (%0),%%ecx                     \n"
      "mov        0x4(%0),%%edx                  \n"
      "xor        (%1),%%ecx                     \n"
      "xor        0x4(%1),%%edx                  \n"
      "popcnt     %%ecx,%%ecx                    \n"
      "add        %%ecx,%3                       \n"
      "popcnt     %%edx,%%edx                    \n"
      "add        %%edx,%3                       \n"
      "mov        0x8(%0),%%ecx                  \n"
      "mov        0xc(%0),%%edx                  \n"
      "xor        0x8(%1),%%ecx                  \n"
      "xor        0xc(%1),%%edx                  \n"
      "popcnt     %%ecx,%%ecx                    \n"
      "add        %%ecx,%3                       \n"
      "popcnt     %%edx,%%edx                    \n"
      "add        %%edx,%3                       \n"
      "add        $0x10,%0                       \n"
      "add        $0x10,%1                       \n"
      "sub        $0x10,%2                       \n"
      "jg         1b                             \n"
      : "+r"(src_a),  // %0
        "+r"(src_b),  // %1
        "+r"(count),  // %2
        "+r"(diff)    // %3
      :
      : "memory", "cc", "ecx", "edx");

  return diff;
}
#endif

static const vec8 kNibbleMask = {15, 15, 15, 15, 15, 15, 15, 15,
                                 15, 15, 15, 15, 15, 15, 15, 15};
static const vec8 kBitCount = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};

uint32_t HammingDistance_SSSE3(const uint8_t* src_a,
                               const uint8_t* src_b,
                               int count) {
  uint32_t diff = 0u;

  asm volatile(
      "movdqa     %4,%%xmm2                      \n"
      "movdqa     %5,%%xmm3                      \n"
      "pxor       %%xmm0,%%xmm0                  \n"
      "pxor       %%xmm1,%%xmm1                  \n"
      "sub        %0,%1                          \n"

      LABELALIGN
      "1:                                        \n"
      "movdqa     (%0),%%xmm4                    \n"
      "movdqa     0x10(%0), %%xmm5               \n"
      "pxor       (%0,%1), %%xmm4                \n"
      "movdqa     %%xmm4,%%xmm6                  \n"
      "pand       %%xmm2,%%xmm6                  \n"
      "psrlw      $0x4,%%xmm4                    \n"
      "movdqa     %%xmm3,%%xmm7                  \n"
      "pshufb     %%xmm6,%%xmm7                  \n"
      "pand       %%xmm2,%%xmm4                  \n"
      "movdqa     %%xmm3,%%xmm6                  \n"
      "pshufb     %%xmm4,%%xmm6                  \n"
      "paddb      %%xmm7,%%xmm6                  \n"
      "pxor       0x10(%0,%1),%%xmm5             \n"
      "add        $0x20,%0                       \n"
      "movdqa     %%xmm5,%%xmm4                  \n"
      "pand       %%xmm2,%%xmm5                  \n"
      "psrlw      $0x4,%%xmm4                    \n"
      "movdqa     %%xmm3,%%xmm7                  \n"
      "pshufb     %%xmm5,%%xmm7                  \n"
      "pand       %%xmm2,%%xmm4                  \n"
      "movdqa     %%xmm3,%%xmm5                  \n"
      "pshufb     %%xmm4,%%xmm5                  \n"
      "paddb      %%xmm7,%%xmm5                  \n"
      "paddb      %%xmm5,%%xmm6                  \n"
      "psadbw     %%xmm1,%%xmm6                  \n"
      "paddd      %%xmm6,%%xmm0                  \n"
      "sub        $0x20,%2                       \n"
      "jg         1b                             \n"

      "pshufd     $0xaa,%%xmm0,%%xmm1            \n"
      "paddd      %%xmm1,%%xmm0                  \n"
      "movd       %%xmm0, %3                     \n"
      : "+r"(src_a),       // %0
        "+r"(src_b),       // %1
        "+r"(count),       // %2
        "=r"(diff)         // %3
      : "m"(kNibbleMask),  // %4
        "m"(kBitCount)     // %5
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");

  return diff;
}

#ifdef HAS_HAMMINGDISTANCE_AVX2
uint32_t HammingDistance_AVX2(const uint8_t* src_a,
                              const uint8_t* src_b,
                              int count) {
  uint32_t diff = 0u;

  asm volatile(
      "vbroadcastf128 %4,%%ymm2                  \n"
      "vbroadcastf128 %5,%%ymm3                  \n"
      "vpxor      %%ymm0,%%ymm0,%%ymm0           \n"
      "vpxor      %%ymm1,%%ymm1,%%ymm1           \n"
      "sub        %0,%1                          \n"

      LABELALIGN
      "1:                                        \n"
      "vmovdqa    (%0),%%ymm4                    \n"
      "vmovdqa    0x20(%0), %%ymm5               \n"
      "vpxor      (%0,%1), %%ymm4, %%ymm4        \n"
      "vpand      %%ymm2,%%ymm4,%%ymm6           \n"
      "vpsrlw     $0x4,%%ymm4,%%ymm4             \n"
      "vpshufb    %%ymm6,%%ymm3,%%ymm6           \n"
      "vpand      %%ymm2,%%ymm4,%%ymm4           \n"
      "vpshufb    %%ymm4,%%ymm3,%%ymm4           \n"
      "vpaddb     %%ymm4,%%ymm6,%%ymm6           \n"
      "vpxor      0x20(%0,%1),%%ymm5,%%ymm4      \n"
      "add        $0x40,%0                       \n"
      "vpand      %%ymm2,%%ymm4,%%ymm5           \n"
      "vpsrlw     $0x4,%%ymm4,%%ymm4             \n"
      "vpshufb    %%ymm5,%%ymm3,%%ymm5           \n"
      "vpand      %%ymm2,%%ymm4,%%ymm4           \n"
      "vpshufb    %%ymm4,%%ymm3,%%ymm4           \n"
      "vpaddb     %%ymm5,%%ymm4,%%ymm4           \n"
      "vpaddb     %%ymm6,%%ymm4,%%ymm4           \n"
      "vpsadbw    %%ymm1,%%ymm4,%%ymm4           \n"
      "vpaddd     %%ymm0,%%ymm4,%%ymm0           \n"
      "sub        $0x40,%2                       \n"
      "jg         1b                             \n"

      "vpermq     $0xb1,%%ymm0,%%ymm1            \n"
      "vpaddd     %%ymm1,%%ymm0,%%ymm0           \n"
      "vpermq     $0xaa,%%ymm0,%%ymm1            \n"
      "vpaddd     %%ymm1,%%ymm0,%%ymm0           \n"
      "vmovd      %%xmm0, %3                     \n"
      "vzeroupper                                \n"
      : "+r"(src_a),       // %0
        "+r"(src_b),       // %1
        "+r"(count),       // %2
        "=r"(diff)         // %3
      : "m"(kNibbleMask),  // %4
        "m"(kBitCount)     // %5
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6");

  return diff;
}
#endif  // HAS_HAMMINGDISTANCE_AVX2

uint32_t SumSquareError_SSE2(const uint8_t* src_a,
                             const uint8_t* src_b,
                             int count) {
  uint32_t sse;
  asm volatile(
      "pxor      %%xmm0,%%xmm0                   \n"
      "pxor      %%xmm5,%%xmm5                   \n"

      LABELALIGN
      "1:                                        \n"
      "movdqu    (%0),%%xmm1                     \n"
      "lea       0x10(%0),%0                     \n"
      "movdqu    (%1),%%xmm2                     \n"
      "lea       0x10(%1),%1                     \n"
      "movdqa    %%xmm1,%%xmm3                   \n"
      "psubusb   %%xmm2,%%xmm1                   \n"
      "psubusb   %%xmm3,%%xmm2                   \n"
      "por       %%xmm2,%%xmm1                   \n"
      "movdqa    %%xmm1,%%xmm2                   \n"
      "punpcklbw %%xmm5,%%xmm1                   \n"
      "punpckhbw %%xmm5,%%xmm2                   \n"
      "pmaddwd   %%xmm1,%%xmm1                   \n"
      "pmaddwd   %%xmm2,%%xmm2                   \n"
      "paddd     %%xmm1,%%xmm0                   \n"
      "paddd     %%xmm2,%%xmm0                   \n"
      "sub       $0x10,%2                        \n"
      "jg        1b                              \n"

      "pshufd    $0xee,%%xmm0,%%xmm1             \n"
      "paddd     %%xmm1,%%xmm0                   \n"
      "pshufd    $0x1,%%xmm0,%%xmm1              \n"
      "paddd     %%xmm1,%%xmm0                   \n"
      "movd      %%xmm0,%3                       \n"

      : "+r"(src_a),  // %0
        "+r"(src_b),  // %1
        "+r"(count),  // %2
        "=g"(sse)     // %3
        ::"memory",
        "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm5");
  return sse;
}

static const uvec32 kHash16x33 = {0x92d9e201, 0, 0, 0};  // 33 ^ 16
static const uvec32 kHashMul0 = {
    0x0c3525e1,  // 33 ^ 15
    0xa3476dc1,  // 33 ^ 14
    0x3b4039a1,  // 33 ^ 13
    0x4f5f0981,  // 33 ^ 12
};
static const uvec32 kHashMul1 = {
    0x30f35d61,  // 33 ^ 11
    0x855cb541,  // 33 ^ 10
    0x040a9121,  // 33 ^ 9
    0x747c7101,  // 33 ^ 8
};
static const uvec32 kHashMul2 = {
    0xec41d4e1,  // 33 ^ 7
    0x4cfa3cc1,  // 33 ^ 6
    0x025528a1,  // 33 ^ 5
    0x00121881,  // 33 ^ 4
};
static const uvec32 kHashMul3 = {
    0x00008c61,  // 33 ^ 3
    0x00000441,  // 33 ^ 2
    0x00000021,  // 33 ^ 1
    0x00000001,  // 33 ^ 0
};

uint32_t HashDjb2_SSE41(const uint8_t* src, int count, uint32_t seed) {
  uint32_t hash;
  asm volatile(
      "movd      %2,%%xmm0                       \n"
      "pxor      %%xmm7,%%xmm7                   \n"
      "movdqa    %4,%%xmm6                       \n"

      LABELALIGN
      "1:                                        \n"
      "movdqu    (%0),%%xmm1                     \n"
      "lea       0x10(%0),%0                     \n"
      "pmulld    %%xmm6,%%xmm0                   \n"
      "movdqa    %5,%%xmm5                       \n"
      "movdqa    %%xmm1,%%xmm2                   \n"
      "punpcklbw %%xmm7,%%xmm2                   \n"
      "movdqa    %%xmm2,%%xmm3                   \n"
      "punpcklwd %%xmm7,%%xmm3                   \n"
      "pmulld    %%xmm5,%%xmm3                   \n"
      "movdqa    %6,%%xmm5                       \n"
      "movdqa    %%xmm2,%%xmm4                   \n"
      "punpckhwd %%xmm7,%%xmm4                   \n"
      "pmulld    %%xmm5,%%xmm4                   \n"
      "movdqa    %7,%%xmm5                       \n"
      "punpckhbw %%xmm7,%%xmm1                   \n"
      "movdqa    %%xmm1,%%xmm2                   \n"
      "punpcklwd %%xmm7,%%xmm2                   \n"
      "pmulld    %%xmm5,%%xmm2                   \n"
      "movdqa    %8,%%xmm5                       \n"
      "punpckhwd %%xmm7,%%xmm1                   \n"
      "pmulld    %%xmm5,%%xmm1                   \n"
      "paddd     %%xmm4,%%xmm3                   \n"
      "paddd     %%xmm2,%%xmm1                   \n"
      "paddd     %%xmm3,%%xmm1                   \n"
      "pshufd    $0xe,%%xmm1,%%xmm2              \n"
      "paddd     %%xmm2,%%xmm1                   \n"
      "pshufd    $0x1,%%xmm1,%%xmm2              \n"
      "paddd     %%xmm2,%%xmm1                   \n"
      "paddd     %%xmm1,%%xmm0                   \n"
      "sub       $0x10,%1                        \n"
      "jg        1b                              \n"
      "movd      %%xmm0,%3                       \n"
      : "+r"(src),        // %0
        "+r"(count),      // %1
        "+rm"(seed),      // %2
        "=g"(hash)        // %3
      : "m"(kHash16x33),  // %4
        "m"(kHashMul0),   // %5
        "m"(kHashMul1),   // %6
        "m"(kHashMul2),   // %7
        "m"(kHashMul3)    // %8
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
  return hash;
}
#endif  // defined(__x86_64__) || (defined(__i386__) && !defined(__pic__)))

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
