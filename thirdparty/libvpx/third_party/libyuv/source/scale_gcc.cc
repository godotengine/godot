/*
 *  Copyright 2013 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "libyuv/row.h"
#include "libyuv/scale_row.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// This module is for GCC x86 and x64.
#if !defined(LIBYUV_DISABLE_X86) && \
    (defined(__x86_64__) || (defined(__i386__) && !defined(_MSC_VER)))

// Offsets for source bytes 0 to 9
static const uvec8 kShuf0 = {0,   1,   3,   4,   5,   7,   8,   9,
                             128, 128, 128, 128, 128, 128, 128, 128};

// Offsets for source bytes 11 to 20 with 8 subtracted = 3 to 12.
static const uvec8 kShuf1 = {3,   4,   5,   7,   8,   9,   11,  12,
                             128, 128, 128, 128, 128, 128, 128, 128};

// Offsets for source bytes 21 to 31 with 16 subtracted = 5 to 31.
static const uvec8 kShuf2 = {5,   7,   8,   9,   11,  12,  13,  15,
                             128, 128, 128, 128, 128, 128, 128, 128};

// Offsets for source bytes 0 to 10
static const uvec8 kShuf01 = {0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7, 8, 9, 9, 10};

// Offsets for source bytes 10 to 21 with 8 subtracted = 3 to 13.
static const uvec8 kShuf11 = {2, 3, 4, 5,  5,  6,  6,  7,
                              8, 9, 9, 10, 10, 11, 12, 13};

// Offsets for source bytes 21 to 31 with 16 subtracted = 5 to 31.
static const uvec8 kShuf21 = {5,  6,  6,  7,  8,  9,  9,  10,
                              10, 11, 12, 13, 13, 14, 14, 15};

// Coefficients for source bytes 0 to 10
static const uvec8 kMadd01 = {3, 1, 2, 2, 1, 3, 3, 1, 2, 2, 1, 3, 3, 1, 2, 2};

// Coefficients for source bytes 10 to 21
static const uvec8 kMadd11 = {1, 3, 3, 1, 2, 2, 1, 3, 3, 1, 2, 2, 1, 3, 3, 1};

// Coefficients for source bytes 21 to 31
static const uvec8 kMadd21 = {2, 2, 1, 3, 3, 1, 2, 2, 1, 3, 3, 1, 2, 2, 1, 3};

// Coefficients for source bytes 21 to 31
static const vec16 kRound34 = {2, 2, 2, 2, 2, 2, 2, 2};

static const uvec8 kShuf38a = {0,   3,   6,   8,   11,  14,  128, 128,
                               128, 128, 128, 128, 128, 128, 128, 128};

static const uvec8 kShuf38b = {128, 128, 128, 128, 128, 128, 0,   3,
                               6,   8,   11,  14,  128, 128, 128, 128};

// Arrange words 0,3,6 into 0,1,2
static const uvec8 kShufAc = {0,   1,   6,   7,   12,  13,  128, 128,
                              128, 128, 128, 128, 128, 128, 128, 128};

// Arrange words 0,3,6 into 3,4,5
static const uvec8 kShufAc3 = {128, 128, 128, 128, 128, 128, 0,   1,
                               6,   7,   12,  13,  128, 128, 128, 128};

// Scaling values for boxes of 3x3 and 2x3
static const uvec16 kScaleAc33 = {65536 / 9, 65536 / 9, 65536 / 6, 65536 / 9,
                                  65536 / 9, 65536 / 6, 0,         0};

// Arrange first value for pixels 0,1,2,3,4,5
static const uvec8 kShufAb0 = {0,  128, 3,  128, 6,   128, 8,   128,
                               11, 128, 14, 128, 128, 128, 128, 128};

// Arrange second value for pixels 0,1,2,3,4,5
static const uvec8 kShufAb1 = {1,  128, 4,  128, 7,   128, 9,   128,
                               12, 128, 15, 128, 128, 128, 128, 128};

// Arrange third value for pixels 0,1,2,3,4,5
static const uvec8 kShufAb2 = {2,  128, 5,   128, 128, 128, 10,  128,
                               13, 128, 128, 128, 128, 128, 128, 128};

// Scaling values for boxes of 3x2 and 2x2
static const uvec16 kScaleAb2 = {65536 / 3, 65536 / 3, 65536 / 2, 65536 / 3,
                                 65536 / 3, 65536 / 2, 0,         0};

// GCC versions of row functions are verbatim conversions from Visual C.
// Generated using gcc disassembly on Visual C object file:
// objdump -D yuvscaler.obj >yuvscaler.txt

void ScaleRowDown2_SSSE3(const uint8_t* src_ptr,
                         ptrdiff_t src_stride,
                         uint8_t* dst_ptr,
                         int dst_width) {
  (void)src_stride;
  asm volatile(
      // 16 pixel loop.
      LABELALIGN
      "1:                                        \n"
      "movdqu    (%0),%%xmm0                     \n"
      "movdqu    0x10(%0),%%xmm1                 \n"
      "lea       0x20(%0),%0                     \n"
      "psrlw     $0x8,%%xmm0                     \n"
      "psrlw     $0x8,%%xmm1                     \n"
      "packuswb  %%xmm1,%%xmm0                   \n"
      "movdqu    %%xmm0,(%1)                     \n"
      "lea       0x10(%1),%1                     \n"
      "sub       $0x10,%2                        \n"
      "jg        1b                              \n"
      : "+r"(src_ptr),   // %0
        "+r"(dst_ptr),   // %1
        "+r"(dst_width)  // %2
        ::"memory",
        "cc", "xmm0", "xmm1");
}

void ScaleRowDown2Linear_SSSE3(const uint8_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint8_t* dst_ptr,
                               int dst_width) {
  (void)src_stride;
  asm volatile(
      "pcmpeqb    %%xmm4,%%xmm4                  \n"
      "psrlw      $0xf,%%xmm4                    \n"
      "packuswb   %%xmm4,%%xmm4                  \n"
      "pxor       %%xmm5,%%xmm5                  \n"

      LABELALIGN
      "1:                                        \n"
      "movdqu    (%0),%%xmm0                     \n"
      "movdqu    0x10(%0),%%xmm1                 \n"
      "lea       0x20(%0),%0                     \n"
      "pmaddubsw  %%xmm4,%%xmm0                  \n"
      "pmaddubsw  %%xmm4,%%xmm1                  \n"
      "pavgw      %%xmm5,%%xmm0                  \n"
      "pavgw      %%xmm5,%%xmm1                  \n"
      "packuswb   %%xmm1,%%xmm0                  \n"
      "movdqu    %%xmm0,(%1)                     \n"
      "lea       0x10(%1),%1                     \n"
      "sub       $0x10,%2                        \n"
      "jg        1b                              \n"
      : "+r"(src_ptr),   // %0
        "+r"(dst_ptr),   // %1
        "+r"(dst_width)  // %2
        ::"memory",
        "cc", "xmm0", "xmm1", "xmm4", "xmm5");
}

void ScaleRowDown2Box_SSSE3(const uint8_t* src_ptr,
                            ptrdiff_t src_stride,
                            uint8_t* dst_ptr,
                            int dst_width) {
  asm volatile(
      "pcmpeqb    %%xmm4,%%xmm4                  \n"
      "psrlw      $0xf,%%xmm4                    \n"
      "packuswb   %%xmm4,%%xmm4                  \n"
      "pxor       %%xmm5,%%xmm5                  \n"

      LABELALIGN
      "1:                                        \n"
      "movdqu    (%0),%%xmm0                     \n"
      "movdqu    0x10(%0),%%xmm1                 \n"
      "movdqu    0x00(%0,%3,1),%%xmm2            \n"
      "movdqu    0x10(%0,%3,1),%%xmm3            \n"
      "lea       0x20(%0),%0                     \n"
      "pmaddubsw  %%xmm4,%%xmm0                  \n"
      "pmaddubsw  %%xmm4,%%xmm1                  \n"
      "pmaddubsw  %%xmm4,%%xmm2                  \n"
      "pmaddubsw  %%xmm4,%%xmm3                  \n"
      "paddw      %%xmm2,%%xmm0                  \n"
      "paddw      %%xmm3,%%xmm1                  \n"
      "psrlw      $0x1,%%xmm0                    \n"
      "psrlw      $0x1,%%xmm1                    \n"
      "pavgw      %%xmm5,%%xmm0                  \n"
      "pavgw      %%xmm5,%%xmm1                  \n"
      "packuswb   %%xmm1,%%xmm0                  \n"
      "movdqu    %%xmm0,(%1)                     \n"
      "lea       0x10(%1),%1                     \n"
      "sub       $0x10,%2                        \n"
      "jg        1b                              \n"
      : "+r"(src_ptr),               // %0
        "+r"(dst_ptr),               // %1
        "+r"(dst_width)              // %2
      : "r"((intptr_t)(src_stride))  // %3
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm5");
}

#ifdef HAS_SCALEROWDOWN2_AVX2
void ScaleRowDown2_AVX2(const uint8_t* src_ptr,
                        ptrdiff_t src_stride,
                        uint8_t* dst_ptr,
                        int dst_width) {
  (void)src_stride;
  asm volatile(

      LABELALIGN
      "1:                                        \n"
      "vmovdqu    (%0),%%ymm0                    \n"
      "vmovdqu    0x20(%0),%%ymm1                \n"
      "lea        0x40(%0),%0                    \n"
      "vpsrlw     $0x8,%%ymm0,%%ymm0             \n"
      "vpsrlw     $0x8,%%ymm1,%%ymm1             \n"
      "vpackuswb  %%ymm1,%%ymm0,%%ymm0           \n"
      "vpermq     $0xd8,%%ymm0,%%ymm0            \n"
      "vmovdqu    %%ymm0,(%1)                    \n"
      "lea        0x20(%1),%1                    \n"
      "sub        $0x20,%2                       \n"
      "jg         1b                             \n"
      "vzeroupper                                \n"
      : "+r"(src_ptr),   // %0
        "+r"(dst_ptr),   // %1
        "+r"(dst_width)  // %2
        ::"memory",
        "cc", "xmm0", "xmm1");
}

void ScaleRowDown2Linear_AVX2(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_ptr,
                              int dst_width) {
  (void)src_stride;
  asm volatile(
      "vpcmpeqb   %%ymm4,%%ymm4,%%ymm4           \n"
      "vpsrlw     $0xf,%%ymm4,%%ymm4             \n"
      "vpackuswb  %%ymm4,%%ymm4,%%ymm4           \n"
      "vpxor      %%ymm5,%%ymm5,%%ymm5           \n"

      LABELALIGN
      "1:                                        \n"
      "vmovdqu    (%0),%%ymm0                    \n"
      "vmovdqu    0x20(%0),%%ymm1                \n"
      "lea        0x40(%0),%0                    \n"
      "vpmaddubsw %%ymm4,%%ymm0,%%ymm0           \n"
      "vpmaddubsw %%ymm4,%%ymm1,%%ymm1           \n"
      "vpavgw     %%ymm5,%%ymm0,%%ymm0           \n"
      "vpavgw     %%ymm5,%%ymm1,%%ymm1           \n"
      "vpackuswb  %%ymm1,%%ymm0,%%ymm0           \n"
      "vpermq     $0xd8,%%ymm0,%%ymm0            \n"
      "vmovdqu    %%ymm0,(%1)                    \n"
      "lea        0x20(%1),%1                    \n"
      "sub        $0x20,%2                       \n"
      "jg         1b                             \n"
      "vzeroupper                                \n"
      : "+r"(src_ptr),   // %0
        "+r"(dst_ptr),   // %1
        "+r"(dst_width)  // %2
        ::"memory",
        "cc", "xmm0", "xmm1", "xmm4", "xmm5");
}

void ScaleRowDown2Box_AVX2(const uint8_t* src_ptr,
                           ptrdiff_t src_stride,
                           uint8_t* dst_ptr,
                           int dst_width) {
  asm volatile(
      "vpcmpeqb   %%ymm4,%%ymm4,%%ymm4           \n"
      "vpsrlw     $0xf,%%ymm4,%%ymm4             \n"
      "vpackuswb  %%ymm4,%%ymm4,%%ymm4           \n"
      "vpxor      %%ymm5,%%ymm5,%%ymm5           \n"

      LABELALIGN
      "1:                                        \n"
      "vmovdqu    (%0),%%ymm0                    \n"
      "vmovdqu    0x20(%0),%%ymm1                \n"
      "vmovdqu    0x00(%0,%3,1),%%ymm2           \n"
      "vmovdqu    0x20(%0,%3,1),%%ymm3           \n"
      "lea        0x40(%0),%0                    \n"
      "vpmaddubsw %%ymm4,%%ymm0,%%ymm0           \n"
      "vpmaddubsw %%ymm4,%%ymm1,%%ymm1           \n"
      "vpmaddubsw %%ymm4,%%ymm2,%%ymm2           \n"
      "vpmaddubsw %%ymm4,%%ymm3,%%ymm3           \n"
      "vpaddw     %%ymm2,%%ymm0,%%ymm0           \n"
      "vpaddw     %%ymm3,%%ymm1,%%ymm1           \n"
      "vpsrlw     $0x1,%%ymm0,%%ymm0             \n"
      "vpsrlw     $0x1,%%ymm1,%%ymm1             \n"
      "vpavgw     %%ymm5,%%ymm0,%%ymm0           \n"
      "vpavgw     %%ymm5,%%ymm1,%%ymm1           \n"
      "vpackuswb  %%ymm1,%%ymm0,%%ymm0           \n"
      "vpermq     $0xd8,%%ymm0,%%ymm0            \n"
      "vmovdqu    %%ymm0,(%1)                    \n"
      "lea        0x20(%1),%1                    \n"
      "sub        $0x20,%2                       \n"
      "jg         1b                             \n"
      "vzeroupper                                \n"
      : "+r"(src_ptr),               // %0
        "+r"(dst_ptr),               // %1
        "+r"(dst_width)              // %2
      : "r"((intptr_t)(src_stride))  // %3
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm5");
}
#endif  // HAS_SCALEROWDOWN2_AVX2

void ScaleRowDown4_SSSE3(const uint8_t* src_ptr,
                         ptrdiff_t src_stride,
                         uint8_t* dst_ptr,
                         int dst_width) {
  (void)src_stride;
  asm volatile(
      "pcmpeqb   %%xmm5,%%xmm5                   \n"
      "psrld     $0x18,%%xmm5                    \n"
      "pslld     $0x10,%%xmm5                    \n"

      LABELALIGN
      "1:                                        \n"
      "movdqu    (%0),%%xmm0                     \n"
      "movdqu    0x10(%0),%%xmm1                 \n"
      "lea       0x20(%0),%0                     \n"
      "pand      %%xmm5,%%xmm0                   \n"
      "pand      %%xmm5,%%xmm1                   \n"
      "packuswb  %%xmm1,%%xmm0                   \n"
      "psrlw     $0x8,%%xmm0                     \n"
      "packuswb  %%xmm0,%%xmm0                   \n"
      "movq      %%xmm0,(%1)                     \n"
      "lea       0x8(%1),%1                      \n"
      "sub       $0x8,%2                         \n"
      "jg        1b                              \n"
      : "+r"(src_ptr),   // %0
        "+r"(dst_ptr),   // %1
        "+r"(dst_width)  // %2
        ::"memory",
        "cc", "xmm0", "xmm1", "xmm5");
}

void ScaleRowDown4Box_SSSE3(const uint8_t* src_ptr,
                            ptrdiff_t src_stride,
                            uint8_t* dst_ptr,
                            int dst_width) {
  intptr_t stridex3;
  asm volatile(
      "pcmpeqb    %%xmm4,%%xmm4                  \n"
      "psrlw      $0xf,%%xmm4                    \n"
      "movdqa     %%xmm4,%%xmm5                  \n"
      "packuswb   %%xmm4,%%xmm4                  \n"
      "psllw      $0x3,%%xmm5                    \n"
      "lea       0x00(%4,%4,2),%3                \n"

      LABELALIGN
      "1:                                        \n"
      "movdqu    (%0),%%xmm0                     \n"
      "movdqu    0x10(%0),%%xmm1                 \n"
      "movdqu    0x00(%0,%4,1),%%xmm2            \n"
      "movdqu    0x10(%0,%4,1),%%xmm3            \n"
      "pmaddubsw  %%xmm4,%%xmm0                  \n"
      "pmaddubsw  %%xmm4,%%xmm1                  \n"
      "pmaddubsw  %%xmm4,%%xmm2                  \n"
      "pmaddubsw  %%xmm4,%%xmm3                  \n"
      "paddw      %%xmm2,%%xmm0                  \n"
      "paddw      %%xmm3,%%xmm1                  \n"
      "movdqu    0x00(%0,%4,2),%%xmm2            \n"
      "movdqu    0x10(%0,%4,2),%%xmm3            \n"
      "pmaddubsw  %%xmm4,%%xmm2                  \n"
      "pmaddubsw  %%xmm4,%%xmm3                  \n"
      "paddw      %%xmm2,%%xmm0                  \n"
      "paddw      %%xmm3,%%xmm1                  \n"
      "movdqu    0x00(%0,%3,1),%%xmm2            \n"
      "movdqu    0x10(%0,%3,1),%%xmm3            \n"
      "lea       0x20(%0),%0                     \n"
      "pmaddubsw  %%xmm4,%%xmm2                  \n"
      "pmaddubsw  %%xmm4,%%xmm3                  \n"
      "paddw      %%xmm2,%%xmm0                  \n"
      "paddw      %%xmm3,%%xmm1                  \n"
      "phaddw     %%xmm1,%%xmm0                  \n"
      "paddw      %%xmm5,%%xmm0                  \n"
      "psrlw      $0x4,%%xmm0                    \n"
      "packuswb   %%xmm0,%%xmm0                  \n"
      "movq      %%xmm0,(%1)                     \n"
      "lea       0x8(%1),%1                      \n"
      "sub       $0x8,%2                         \n"
      "jg        1b                              \n"
      : "+r"(src_ptr),               // %0
        "+r"(dst_ptr),               // %1
        "+r"(dst_width),             // %2
        "=&r"(stridex3)              // %3
      : "r"((intptr_t)(src_stride))  // %4
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5");
}

#ifdef HAS_SCALEROWDOWN4_AVX2
void ScaleRowDown4_AVX2(const uint8_t* src_ptr,
                        ptrdiff_t src_stride,
                        uint8_t* dst_ptr,
                        int dst_width) {
  (void)src_stride;
  asm volatile(
      "vpcmpeqb   %%ymm5,%%ymm5,%%ymm5           \n"
      "vpsrld     $0x18,%%ymm5,%%ymm5            \n"
      "vpslld     $0x10,%%ymm5,%%ymm5            \n"

      LABELALIGN
      "1:                                        \n"
      "vmovdqu    (%0),%%ymm0                    \n"
      "vmovdqu    0x20(%0),%%ymm1                \n"
      "lea        0x40(%0),%0                    \n"
      "vpand      %%ymm5,%%ymm0,%%ymm0           \n"
      "vpand      %%ymm5,%%ymm1,%%ymm1           \n"
      "vpackuswb  %%ymm1,%%ymm0,%%ymm0           \n"
      "vpermq     $0xd8,%%ymm0,%%ymm0            \n"
      "vpsrlw     $0x8,%%ymm0,%%ymm0             \n"
      "vpackuswb  %%ymm0,%%ymm0,%%ymm0           \n"
      "vpermq     $0xd8,%%ymm0,%%ymm0            \n"
      "vmovdqu    %%xmm0,(%1)                    \n"
      "lea        0x10(%1),%1                    \n"
      "sub        $0x10,%2                       \n"
      "jg         1b                             \n"
      "vzeroupper                                \n"
      : "+r"(src_ptr),   // %0
        "+r"(dst_ptr),   // %1
        "+r"(dst_width)  // %2
        ::"memory",
        "cc", "xmm0", "xmm1", "xmm5");
}

void ScaleRowDown4Box_AVX2(const uint8_t* src_ptr,
                           ptrdiff_t src_stride,
                           uint8_t* dst_ptr,
                           int dst_width) {
  asm volatile(
      "vpcmpeqb   %%ymm4,%%ymm4,%%ymm4           \n"
      "vpsrlw     $0xf,%%ymm4,%%ymm4             \n"
      "vpsllw     $0x3,%%ymm4,%%ymm5             \n"
      "vpackuswb  %%ymm4,%%ymm4,%%ymm4           \n"

      LABELALIGN
      "1:                                        \n"
      "vmovdqu    (%0),%%ymm0                    \n"
      "vmovdqu    0x20(%0),%%ymm1                \n"
      "vmovdqu    0x00(%0,%3,1),%%ymm2           \n"
      "vmovdqu    0x20(%0,%3,1),%%ymm3           \n"
      "vpmaddubsw %%ymm4,%%ymm0,%%ymm0           \n"
      "vpmaddubsw %%ymm4,%%ymm1,%%ymm1           \n"
      "vpmaddubsw %%ymm4,%%ymm2,%%ymm2           \n"
      "vpmaddubsw %%ymm4,%%ymm3,%%ymm3           \n"
      "vpaddw     %%ymm2,%%ymm0,%%ymm0           \n"
      "vpaddw     %%ymm3,%%ymm1,%%ymm1           \n"
      "vmovdqu    0x00(%0,%3,2),%%ymm2           \n"
      "vmovdqu    0x20(%0,%3,2),%%ymm3           \n"
      "vpmaddubsw %%ymm4,%%ymm2,%%ymm2           \n"
      "vpmaddubsw %%ymm4,%%ymm3,%%ymm3           \n"
      "vpaddw     %%ymm2,%%ymm0,%%ymm0           \n"
      "vpaddw     %%ymm3,%%ymm1,%%ymm1           \n"
      "vmovdqu    0x00(%0,%4,1),%%ymm2           \n"
      "vmovdqu    0x20(%0,%4,1),%%ymm3           \n"
      "lea        0x40(%0),%0                    \n"
      "vpmaddubsw %%ymm4,%%ymm2,%%ymm2           \n"
      "vpmaddubsw %%ymm4,%%ymm3,%%ymm3           \n"
      "vpaddw     %%ymm2,%%ymm0,%%ymm0           \n"
      "vpaddw     %%ymm3,%%ymm1,%%ymm1           \n"
      "vphaddw    %%ymm1,%%ymm0,%%ymm0           \n"
      "vpermq     $0xd8,%%ymm0,%%ymm0            \n"
      "vpaddw     %%ymm5,%%ymm0,%%ymm0           \n"
      "vpsrlw     $0x4,%%ymm0,%%ymm0             \n"
      "vpackuswb  %%ymm0,%%ymm0,%%ymm0           \n"
      "vpermq     $0xd8,%%ymm0,%%ymm0            \n"
      "vmovdqu    %%xmm0,(%1)                    \n"
      "lea        0x10(%1),%1                    \n"
      "sub        $0x10,%2                       \n"
      "jg         1b                             \n"
      "vzeroupper                                \n"
      : "+r"(src_ptr),                   // %0
        "+r"(dst_ptr),                   // %1
        "+r"(dst_width)                  // %2
      : "r"((intptr_t)(src_stride)),     // %3
        "r"((intptr_t)(src_stride * 3))  // %4
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5");
}
#endif  // HAS_SCALEROWDOWN4_AVX2

void ScaleRowDown34_SSSE3(const uint8_t* src_ptr,
                          ptrdiff_t src_stride,
                          uint8_t* dst_ptr,
                          int dst_width) {
  (void)src_stride;
  asm volatile(
      "movdqa    %0,%%xmm3                       \n"
      "movdqa    %1,%%xmm4                       \n"
      "movdqa    %2,%%xmm5                       \n"
      :
      : "m"(kShuf0),  // %0
        "m"(kShuf1),  // %1
        "m"(kShuf2)   // %2
      );
  asm volatile(

      LABELALIGN
      "1:                                        \n"
      "movdqu    (%0),%%xmm0                     \n"
      "movdqu    0x10(%0),%%xmm2                 \n"
      "lea       0x20(%0),%0                     \n"
      "movdqa    %%xmm2,%%xmm1                   \n"
      "palignr   $0x8,%%xmm0,%%xmm1              \n"
      "pshufb    %%xmm3,%%xmm0                   \n"
      "pshufb    %%xmm4,%%xmm1                   \n"
      "pshufb    %%xmm5,%%xmm2                   \n"
      "movq      %%xmm0,(%1)                     \n"
      "movq      %%xmm1,0x8(%1)                  \n"
      "movq      %%xmm2,0x10(%1)                 \n"
      "lea       0x18(%1),%1                     \n"
      "sub       $0x18,%2                        \n"
      "jg        1b                              \n"
      : "+r"(src_ptr),   // %0
        "+r"(dst_ptr),   // %1
        "+r"(dst_width)  // %2
        ::"memory",
        "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5");
}

void ScaleRowDown34_1_Box_SSSE3(const uint8_t* src_ptr,
                                ptrdiff_t src_stride,
                                uint8_t* dst_ptr,
                                int dst_width) {
  asm volatile(
      "movdqa    %0,%%xmm2                       \n"  // kShuf01
      "movdqa    %1,%%xmm3                       \n"  // kShuf11
      "movdqa    %2,%%xmm4                       \n"  // kShuf21
      :
      : "m"(kShuf01),  // %0
        "m"(kShuf11),  // %1
        "m"(kShuf21)   // %2
      );
  asm volatile(
      "movdqa    %0,%%xmm5                       \n"  // kMadd01
      "movdqa    %1,%%xmm0                       \n"  // kMadd11
      "movdqa    %2,%%xmm1                       \n"  // kRound34
      :
      : "m"(kMadd01),  // %0
        "m"(kMadd11),  // %1
        "m"(kRound34)  // %2
      );
  asm volatile(

      LABELALIGN
      "1:                                        \n"
      "movdqu    (%0),%%xmm6                     \n"
      "movdqu    0x00(%0,%3,1),%%xmm7            \n"
      "pavgb     %%xmm7,%%xmm6                   \n"
      "pshufb    %%xmm2,%%xmm6                   \n"
      "pmaddubsw %%xmm5,%%xmm6                   \n"
      "paddsw    %%xmm1,%%xmm6                   \n"
      "psrlw     $0x2,%%xmm6                     \n"
      "packuswb  %%xmm6,%%xmm6                   \n"
      "movq      %%xmm6,(%1)                     \n"
      "movdqu    0x8(%0),%%xmm6                  \n"
      "movdqu    0x8(%0,%3,1),%%xmm7             \n"
      "pavgb     %%xmm7,%%xmm6                   \n"
      "pshufb    %%xmm3,%%xmm6                   \n"
      "pmaddubsw %%xmm0,%%xmm6                   \n"
      "paddsw    %%xmm1,%%xmm6                   \n"
      "psrlw     $0x2,%%xmm6                     \n"
      "packuswb  %%xmm6,%%xmm6                   \n"
      "movq      %%xmm6,0x8(%1)                  \n"
      "movdqu    0x10(%0),%%xmm6                 \n"
      "movdqu    0x10(%0,%3,1),%%xmm7            \n"
      "lea       0x20(%0),%0                     \n"
      "pavgb     %%xmm7,%%xmm6                   \n"
      "pshufb    %%xmm4,%%xmm6                   \n"
      "pmaddubsw %4,%%xmm6                       \n"
      "paddsw    %%xmm1,%%xmm6                   \n"
      "psrlw     $0x2,%%xmm6                     \n"
      "packuswb  %%xmm6,%%xmm6                   \n"
      "movq      %%xmm6,0x10(%1)                 \n"
      "lea       0x18(%1),%1                     \n"
      "sub       $0x18,%2                        \n"
      "jg        1b                              \n"
      : "+r"(src_ptr),                // %0
        "+r"(dst_ptr),                // %1
        "+r"(dst_width)               // %2
      : "r"((intptr_t)(src_stride)),  // %3
        "m"(kMadd21)                  // %4
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}

void ScaleRowDown34_0_Box_SSSE3(const uint8_t* src_ptr,
                                ptrdiff_t src_stride,
                                uint8_t* dst_ptr,
                                int dst_width) {
  asm volatile(
      "movdqa    %0,%%xmm2                       \n"  // kShuf01
      "movdqa    %1,%%xmm3                       \n"  // kShuf11
      "movdqa    %2,%%xmm4                       \n"  // kShuf21
      :
      : "m"(kShuf01),  // %0
        "m"(kShuf11),  // %1
        "m"(kShuf21)   // %2
      );
  asm volatile(
      "movdqa    %0,%%xmm5                       \n"  // kMadd01
      "movdqa    %1,%%xmm0                       \n"  // kMadd11
      "movdqa    %2,%%xmm1                       \n"  // kRound34
      :
      : "m"(kMadd01),  // %0
        "m"(kMadd11),  // %1
        "m"(kRound34)  // %2
      );

  asm volatile(

      LABELALIGN
      "1:                                        \n"
      "movdqu    (%0),%%xmm6                     \n"
      "movdqu    0x00(%0,%3,1),%%xmm7            \n"
      "pavgb     %%xmm6,%%xmm7                   \n"
      "pavgb     %%xmm7,%%xmm6                   \n"
      "pshufb    %%xmm2,%%xmm6                   \n"
      "pmaddubsw %%xmm5,%%xmm6                   \n"
      "paddsw    %%xmm1,%%xmm6                   \n"
      "psrlw     $0x2,%%xmm6                     \n"
      "packuswb  %%xmm6,%%xmm6                   \n"
      "movq      %%xmm6,(%1)                     \n"
      "movdqu    0x8(%0),%%xmm6                  \n"
      "movdqu    0x8(%0,%3,1),%%xmm7             \n"
      "pavgb     %%xmm6,%%xmm7                   \n"
      "pavgb     %%xmm7,%%xmm6                   \n"
      "pshufb    %%xmm3,%%xmm6                   \n"
      "pmaddubsw %%xmm0,%%xmm6                   \n"
      "paddsw    %%xmm1,%%xmm6                   \n"
      "psrlw     $0x2,%%xmm6                     \n"
      "packuswb  %%xmm6,%%xmm6                   \n"
      "movq      %%xmm6,0x8(%1)                  \n"
      "movdqu    0x10(%0),%%xmm6                 \n"
      "movdqu    0x10(%0,%3,1),%%xmm7            \n"
      "lea       0x20(%0),%0                     \n"
      "pavgb     %%xmm6,%%xmm7                   \n"
      "pavgb     %%xmm7,%%xmm6                   \n"
      "pshufb    %%xmm4,%%xmm6                   \n"
      "pmaddubsw %4,%%xmm6                       \n"
      "paddsw    %%xmm1,%%xmm6                   \n"
      "psrlw     $0x2,%%xmm6                     \n"
      "packuswb  %%xmm6,%%xmm6                   \n"
      "movq      %%xmm6,0x10(%1)                 \n"
      "lea       0x18(%1),%1                     \n"
      "sub       $0x18,%2                        \n"
      "jg        1b                              \n"
      : "+r"(src_ptr),                // %0
        "+r"(dst_ptr),                // %1
        "+r"(dst_width)               // %2
      : "r"((intptr_t)(src_stride)),  // %3
        "m"(kMadd21)                  // %4
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}

void ScaleRowDown38_SSSE3(const uint8_t* src_ptr,
                          ptrdiff_t src_stride,
                          uint8_t* dst_ptr,
                          int dst_width) {
  (void)src_stride;
  asm volatile(
      "movdqa    %3,%%xmm4                       \n"
      "movdqa    %4,%%xmm5                       \n"

      LABELALIGN
      "1:                                        \n"
      "movdqu    (%0),%%xmm0                     \n"
      "movdqu    0x10(%0),%%xmm1                 \n"
      "lea       0x20(%0),%0                     \n"
      "pshufb    %%xmm4,%%xmm0                   \n"
      "pshufb    %%xmm5,%%xmm1                   \n"
      "paddusb   %%xmm1,%%xmm0                   \n"
      "movq      %%xmm0,(%1)                     \n"
      "movhlps   %%xmm0,%%xmm1                   \n"
      "movd      %%xmm1,0x8(%1)                  \n"
      "lea       0xc(%1),%1                      \n"
      "sub       $0xc,%2                         \n"
      "jg        1b                              \n"
      : "+r"(src_ptr),   // %0
        "+r"(dst_ptr),   // %1
        "+r"(dst_width)  // %2
      : "m"(kShuf38a),   // %3
        "m"(kShuf38b)    // %4
      : "memory", "cc", "xmm0", "xmm1", "xmm4", "xmm5");
}

void ScaleRowDown38_2_Box_SSSE3(const uint8_t* src_ptr,
                                ptrdiff_t src_stride,
                                uint8_t* dst_ptr,
                                int dst_width) {
  asm volatile(
      "movdqa    %0,%%xmm2                       \n"
      "movdqa    %1,%%xmm3                       \n"
      "movdqa    %2,%%xmm4                       \n"
      "movdqa    %3,%%xmm5                       \n"
      :
      : "m"(kShufAb0),  // %0
        "m"(kShufAb1),  // %1
        "m"(kShufAb2),  // %2
        "m"(kScaleAb2)  // %3
      );
  asm volatile(

      LABELALIGN
      "1:                                        \n"
      "movdqu    (%0),%%xmm0                     \n"
      "movdqu    0x00(%0,%3,1),%%xmm1            \n"
      "lea       0x10(%0),%0                     \n"
      "pavgb     %%xmm1,%%xmm0                   \n"
      "movdqa    %%xmm0,%%xmm1                   \n"
      "pshufb    %%xmm2,%%xmm1                   \n"
      "movdqa    %%xmm0,%%xmm6                   \n"
      "pshufb    %%xmm3,%%xmm6                   \n"
      "paddusw   %%xmm6,%%xmm1                   \n"
      "pshufb    %%xmm4,%%xmm0                   \n"
      "paddusw   %%xmm0,%%xmm1                   \n"
      "pmulhuw   %%xmm5,%%xmm1                   \n"
      "packuswb  %%xmm1,%%xmm1                   \n"
      "movd      %%xmm1,(%1)                     \n"
      "psrlq     $0x10,%%xmm1                    \n"
      "movd      %%xmm1,0x2(%1)                  \n"
      "lea       0x6(%1),%1                      \n"
      "sub       $0x6,%2                         \n"
      "jg        1b                              \n"
      : "+r"(src_ptr),               // %0
        "+r"(dst_ptr),               // %1
        "+r"(dst_width)              // %2
      : "r"((intptr_t)(src_stride))  // %3
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6");
}

void ScaleRowDown38_3_Box_SSSE3(const uint8_t* src_ptr,
                                ptrdiff_t src_stride,
                                uint8_t* dst_ptr,
                                int dst_width) {
  asm volatile(
      "movdqa    %0,%%xmm2                       \n"
      "movdqa    %1,%%xmm3                       \n"
      "movdqa    %2,%%xmm4                       \n"
      "pxor      %%xmm5,%%xmm5                   \n"
      :
      : "m"(kShufAc),    // %0
        "m"(kShufAc3),   // %1
        "m"(kScaleAc33)  // %2
      );
  asm volatile(

      LABELALIGN
      "1:                                        \n"
      "movdqu    (%0),%%xmm0                     \n"
      "movdqu    0x00(%0,%3,1),%%xmm6            \n"
      "movhlps   %%xmm0,%%xmm1                   \n"
      "movhlps   %%xmm6,%%xmm7                   \n"
      "punpcklbw %%xmm5,%%xmm0                   \n"
      "punpcklbw %%xmm5,%%xmm1                   \n"
      "punpcklbw %%xmm5,%%xmm6                   \n"
      "punpcklbw %%xmm5,%%xmm7                   \n"
      "paddusw   %%xmm6,%%xmm0                   \n"
      "paddusw   %%xmm7,%%xmm1                   \n"
      "movdqu    0x00(%0,%3,2),%%xmm6            \n"
      "lea       0x10(%0),%0                     \n"
      "movhlps   %%xmm6,%%xmm7                   \n"
      "punpcklbw %%xmm5,%%xmm6                   \n"
      "punpcklbw %%xmm5,%%xmm7                   \n"
      "paddusw   %%xmm6,%%xmm0                   \n"
      "paddusw   %%xmm7,%%xmm1                   \n"
      "movdqa    %%xmm0,%%xmm6                   \n"
      "psrldq    $0x2,%%xmm0                     \n"
      "paddusw   %%xmm0,%%xmm6                   \n"
      "psrldq    $0x2,%%xmm0                     \n"
      "paddusw   %%xmm0,%%xmm6                   \n"
      "pshufb    %%xmm2,%%xmm6                   \n"
      "movdqa    %%xmm1,%%xmm7                   \n"
      "psrldq    $0x2,%%xmm1                     \n"
      "paddusw   %%xmm1,%%xmm7                   \n"
      "psrldq    $0x2,%%xmm1                     \n"
      "paddusw   %%xmm1,%%xmm7                   \n"
      "pshufb    %%xmm3,%%xmm7                   \n"
      "paddusw   %%xmm7,%%xmm6                   \n"
      "pmulhuw   %%xmm4,%%xmm6                   \n"
      "packuswb  %%xmm6,%%xmm6                   \n"
      "movd      %%xmm6,(%1)                     \n"
      "psrlq     $0x10,%%xmm6                    \n"
      "movd      %%xmm6,0x2(%1)                  \n"
      "lea       0x6(%1),%1                      \n"
      "sub       $0x6,%2                         \n"
      "jg        1b                              \n"
      : "+r"(src_ptr),               // %0
        "+r"(dst_ptr),               // %1
        "+r"(dst_width)              // %2
      : "r"((intptr_t)(src_stride))  // %3
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}

// Reads 16xN bytes and produces 16 shorts at a time.
void ScaleAddRow_SSE2(const uint8_t* src_ptr,
                      uint16_t* dst_ptr,
                      int src_width) {
  asm volatile(

      "pxor      %%xmm5,%%xmm5                   \n"

      // 16 pixel loop.
      LABELALIGN
      "1:                                        \n"
      "movdqu    (%0),%%xmm3                     \n"
      "lea       0x10(%0),%0                     \n"  // src_ptr += 16
      "movdqu    (%1),%%xmm0                     \n"
      "movdqu    0x10(%1),%%xmm1                 \n"
      "movdqa    %%xmm3,%%xmm2                   \n"
      "punpcklbw %%xmm5,%%xmm2                   \n"
      "punpckhbw %%xmm5,%%xmm3                   \n"
      "paddusw   %%xmm2,%%xmm0                   \n"
      "paddusw   %%xmm3,%%xmm1                   \n"
      "movdqu    %%xmm0,(%1)                     \n"
      "movdqu    %%xmm1,0x10(%1)                 \n"
      "lea       0x20(%1),%1                     \n"
      "sub       $0x10,%2                        \n"
      "jg        1b                              \n"
      : "+r"(src_ptr),   // %0
        "+r"(dst_ptr),   // %1
        "+r"(src_width)  // %2
      :
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm5");
}

#ifdef HAS_SCALEADDROW_AVX2
// Reads 32 bytes and accumulates to 32 shorts at a time.
void ScaleAddRow_AVX2(const uint8_t* src_ptr,
                      uint16_t* dst_ptr,
                      int src_width) {
  asm volatile(

      "vpxor      %%ymm5,%%ymm5,%%ymm5           \n"

      LABELALIGN
      "1:                                        \n"
      "vmovdqu    (%0),%%ymm3                    \n"
      "lea        0x20(%0),%0                    \n"  // src_ptr += 32
      "vpermq     $0xd8,%%ymm3,%%ymm3            \n"
      "vpunpcklbw %%ymm5,%%ymm3,%%ymm2           \n"
      "vpunpckhbw %%ymm5,%%ymm3,%%ymm3           \n"
      "vpaddusw   (%1),%%ymm2,%%ymm0             \n"
      "vpaddusw   0x20(%1),%%ymm3,%%ymm1         \n"
      "vmovdqu    %%ymm0,(%1)                    \n"
      "vmovdqu    %%ymm1,0x20(%1)                \n"
      "lea       0x40(%1),%1                     \n"
      "sub       $0x20,%2                        \n"
      "jg        1b                              \n"
      "vzeroupper                                \n"
      : "+r"(src_ptr),   // %0
        "+r"(dst_ptr),   // %1
        "+r"(src_width)  // %2
      :
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm5");
}
#endif  // HAS_SCALEADDROW_AVX2

// Constant for making pixels signed to avoid pmaddubsw
// saturation.
static const uvec8 kFsub80 = {0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
                              0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80};

// Constant for making pixels unsigned and adding .5 for rounding.
static const uvec16 kFadd40 = {0x4040, 0x4040, 0x4040, 0x4040,
                               0x4040, 0x4040, 0x4040, 0x4040};

// Bilinear column filtering. SSSE3 version.
void ScaleFilterCols_SSSE3(uint8_t* dst_ptr,
                           const uint8_t* src_ptr,
                           int dst_width,
                           int x,
                           int dx) {
  intptr_t x0, x1, temp_pixel;
  asm volatile(
      "movd      %6,%%xmm2                       \n"
      "movd      %7,%%xmm3                       \n"
      "movl      $0x04040000,%k2                 \n"
      "movd      %k2,%%xmm5                      \n"
      "pcmpeqb   %%xmm6,%%xmm6                   \n"
      "psrlw     $0x9,%%xmm6                     \n"  // 0x007f007f
      "pcmpeqb   %%xmm7,%%xmm7                   \n"
      "psrlw     $15,%%xmm7                      \n"  // 0x00010001

      "pextrw    $0x1,%%xmm2,%k3                 \n"
      "subl      $0x2,%5                         \n"
      "jl        29f                             \n"
      "movdqa    %%xmm2,%%xmm0                   \n"
      "paddd     %%xmm3,%%xmm0                   \n"
      "punpckldq %%xmm0,%%xmm2                   \n"
      "punpckldq %%xmm3,%%xmm3                   \n"
      "paddd     %%xmm3,%%xmm3                   \n"
      "pextrw    $0x3,%%xmm2,%k4                 \n"

      LABELALIGN
      "2:                                        \n"
      "movdqa    %%xmm2,%%xmm1                   \n"
      "paddd     %%xmm3,%%xmm2                   \n"
      "movzwl    0x00(%1,%3,1),%k2               \n"
      "movd      %k2,%%xmm0                      \n"
      "psrlw     $0x9,%%xmm1                     \n"
      "movzwl    0x00(%1,%4,1),%k2               \n"
      "movd      %k2,%%xmm4                      \n"
      "pshufb    %%xmm5,%%xmm1                   \n"
      "punpcklwd %%xmm4,%%xmm0                   \n"
      "psubb     %8,%%xmm0                       \n"  // make pixels signed.
      "pxor      %%xmm6,%%xmm1                   \n"  // 128 - f = (f ^ 127 ) +
                                                      // 1
      "paddusb   %%xmm7,%%xmm1                   \n"
      "pmaddubsw %%xmm0,%%xmm1                   \n"
      "pextrw    $0x1,%%xmm2,%k3                 \n"
      "pextrw    $0x3,%%xmm2,%k4                 \n"
      "paddw     %9,%%xmm1                       \n"  // make pixels unsigned.
      "psrlw     $0x7,%%xmm1                     \n"
      "packuswb  %%xmm1,%%xmm1                   \n"
      "movd      %%xmm1,%k2                      \n"
      "mov       %w2,(%0)                        \n"
      "lea       0x2(%0),%0                      \n"
      "subl      $0x2,%5                         \n"
      "jge       2b                              \n"

      LABELALIGN
      "29:                                       \n"
      "addl      $0x1,%5                         \n"
      "jl        99f                             \n"
      "movzwl    0x00(%1,%3,1),%k2               \n"
      "movd      %k2,%%xmm0                      \n"
      "psrlw     $0x9,%%xmm2                     \n"
      "pshufb    %%xmm5,%%xmm2                   \n"
      "psubb     %8,%%xmm0                       \n"  // make pixels signed.
      "pxor      %%xmm6,%%xmm2                   \n"
      "paddusb   %%xmm7,%%xmm2                   \n"
      "pmaddubsw %%xmm0,%%xmm2                   \n"
      "paddw     %9,%%xmm2                       \n"  // make pixels unsigned.
      "psrlw     $0x7,%%xmm2                     \n"
      "packuswb  %%xmm2,%%xmm2                   \n"
      "movd      %%xmm2,%k2                      \n"
      "mov       %b2,(%0)                        \n"
      "99:                                       \n"
      : "+r"(dst_ptr),      // %0
        "+r"(src_ptr),      // %1
        "=&a"(temp_pixel),  // %2
        "=&r"(x0),          // %3
        "=&r"(x1),          // %4
#if defined(__x86_64__)
        "+rm"(dst_width)  // %5
#else
        "+m"(dst_width)  // %5
#endif
      : "rm"(x),   // %6
        "rm"(dx),  // %7
#if defined(__x86_64__)
        "x"(kFsub80),  // %8
        "x"(kFadd40)   // %9
#else
        "m"(kFsub80),    // %8
        "m"(kFadd40)     // %9
#endif
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}

// Reads 4 pixels, duplicates them and writes 8 pixels.
// Alignment requirement: src_argb 16 byte aligned, dst_argb 16 byte aligned.
void ScaleColsUp2_SSE2(uint8_t* dst_ptr,
                       const uint8_t* src_ptr,
                       int dst_width,
                       int x,
                       int dx) {
  (void)x;
  (void)dx;
  asm volatile(

      LABELALIGN
      "1:                                        \n"
      "movdqu    (%1),%%xmm0                     \n"
      "lea       0x10(%1),%1                     \n"
      "movdqa    %%xmm0,%%xmm1                   \n"
      "punpcklbw %%xmm0,%%xmm0                   \n"
      "punpckhbw %%xmm1,%%xmm1                   \n"
      "movdqu    %%xmm0,(%0)                     \n"
      "movdqu    %%xmm1,0x10(%0)                 \n"
      "lea       0x20(%0),%0                     \n"
      "sub       $0x20,%2                        \n"
      "jg        1b                              \n"

      : "+r"(dst_ptr),   // %0
        "+r"(src_ptr),   // %1
        "+r"(dst_width)  // %2
        ::"memory",
        "cc", "xmm0", "xmm1");
}

void ScaleARGBRowDown2_SSE2(const uint8_t* src_argb,
                            ptrdiff_t src_stride,
                            uint8_t* dst_argb,
                            int dst_width) {
  (void)src_stride;
  asm volatile(

      LABELALIGN
      "1:                                        \n"
      "movdqu    (%0),%%xmm0                     \n"
      "movdqu    0x10(%0),%%xmm1                 \n"
      "lea       0x20(%0),%0                     \n"
      "shufps    $0xdd,%%xmm1,%%xmm0             \n"
      "movdqu    %%xmm0,(%1)                     \n"
      "lea       0x10(%1),%1                     \n"
      "sub       $0x4,%2                         \n"
      "jg        1b                              \n"
      : "+r"(src_argb),  // %0
        "+r"(dst_argb),  // %1
        "+r"(dst_width)  // %2
        ::"memory",
        "cc", "xmm0", "xmm1");
}

void ScaleARGBRowDown2Linear_SSE2(const uint8_t* src_argb,
                                  ptrdiff_t src_stride,
                                  uint8_t* dst_argb,
                                  int dst_width) {
  (void)src_stride;
  asm volatile(

      LABELALIGN
      "1:                                        \n"
      "movdqu    (%0),%%xmm0                     \n"
      "movdqu    0x10(%0),%%xmm1                 \n"
      "lea       0x20(%0),%0                     \n"
      "movdqa    %%xmm0,%%xmm2                   \n"
      "shufps    $0x88,%%xmm1,%%xmm0             \n"
      "shufps    $0xdd,%%xmm1,%%xmm2             \n"
      "pavgb     %%xmm2,%%xmm0                   \n"
      "movdqu    %%xmm0,(%1)                     \n"
      "lea       0x10(%1),%1                     \n"
      "sub       $0x4,%2                         \n"
      "jg        1b                              \n"
      : "+r"(src_argb),  // %0
        "+r"(dst_argb),  // %1
        "+r"(dst_width)  // %2
        ::"memory",
        "cc", "xmm0", "xmm1");
}

void ScaleARGBRowDown2Box_SSE2(const uint8_t* src_argb,
                               ptrdiff_t src_stride,
                               uint8_t* dst_argb,
                               int dst_width) {
  asm volatile(

      LABELALIGN
      "1:                                        \n"
      "movdqu    (%0),%%xmm0                     \n"
      "movdqu    0x10(%0),%%xmm1                 \n"
      "movdqu    0x00(%0,%3,1),%%xmm2            \n"
      "movdqu    0x10(%0,%3,1),%%xmm3            \n"
      "lea       0x20(%0),%0                     \n"
      "pavgb     %%xmm2,%%xmm0                   \n"
      "pavgb     %%xmm3,%%xmm1                   \n"
      "movdqa    %%xmm0,%%xmm2                   \n"
      "shufps    $0x88,%%xmm1,%%xmm0             \n"
      "shufps    $0xdd,%%xmm1,%%xmm2             \n"
      "pavgb     %%xmm2,%%xmm0                   \n"
      "movdqu    %%xmm0,(%1)                     \n"
      "lea       0x10(%1),%1                     \n"
      "sub       $0x4,%2                         \n"
      "jg        1b                              \n"
      : "+r"(src_argb),              // %0
        "+r"(dst_argb),              // %1
        "+r"(dst_width)              // %2
      : "r"((intptr_t)(src_stride))  // %3
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3");
}

// Reads 4 pixels at a time.
// Alignment requirement: dst_argb 16 byte aligned.
void ScaleARGBRowDownEven_SSE2(const uint8_t* src_argb,
                               ptrdiff_t src_stride,
                               int src_stepx,
                               uint8_t* dst_argb,
                               int dst_width) {
  intptr_t src_stepx_x4 = (intptr_t)(src_stepx);
  intptr_t src_stepx_x12;
  (void)src_stride;
  asm volatile(
      "lea       0x00(,%1,4),%1                  \n"
      "lea       0x00(%1,%1,2),%4                \n"

      LABELALIGN
      "1:                                        \n"
      "movd      (%0),%%xmm0                     \n"
      "movd      0x00(%0,%1,1),%%xmm1            \n"
      "punpckldq %%xmm1,%%xmm0                   \n"
      "movd      0x00(%0,%1,2),%%xmm2            \n"
      "movd      0x00(%0,%4,1),%%xmm3            \n"
      "lea       0x00(%0,%1,4),%0                \n"
      "punpckldq %%xmm3,%%xmm2                   \n"
      "punpcklqdq %%xmm2,%%xmm0                  \n"
      "movdqu    %%xmm0,(%2)                     \n"
      "lea       0x10(%2),%2                     \n"
      "sub       $0x4,%3                         \n"
      "jg        1b                              \n"
      : "+r"(src_argb),       // %0
        "+r"(src_stepx_x4),   // %1
        "+r"(dst_argb),       // %2
        "+r"(dst_width),      // %3
        "=&r"(src_stepx_x12)  // %4
        ::"memory",
        "cc", "xmm0", "xmm1", "xmm2", "xmm3");
}

// Blends four 2x2 to 4x1.
// Alignment requirement: dst_argb 16 byte aligned.
void ScaleARGBRowDownEvenBox_SSE2(const uint8_t* src_argb,
                                  ptrdiff_t src_stride,
                                  int src_stepx,
                                  uint8_t* dst_argb,
                                  int dst_width) {
  intptr_t src_stepx_x4 = (intptr_t)(src_stepx);
  intptr_t src_stepx_x12;
  intptr_t row1 = (intptr_t)(src_stride);
  asm volatile(
      "lea       0x00(,%1,4),%1                  \n"
      "lea       0x00(%1,%1,2),%4                \n"
      "lea       0x00(%0,%5,1),%5                \n"

      LABELALIGN
      "1:                                        \n"
      "movq      (%0),%%xmm0                     \n"
      "movhps    0x00(%0,%1,1),%%xmm0            \n"
      "movq      0x00(%0,%1,2),%%xmm1            \n"
      "movhps    0x00(%0,%4,1),%%xmm1            \n"
      "lea       0x00(%0,%1,4),%0                \n"
      "movq      (%5),%%xmm2                     \n"
      "movhps    0x00(%5,%1,1),%%xmm2            \n"
      "movq      0x00(%5,%1,2),%%xmm3            \n"
      "movhps    0x00(%5,%4,1),%%xmm3            \n"
      "lea       0x00(%5,%1,4),%5                \n"
      "pavgb     %%xmm2,%%xmm0                   \n"
      "pavgb     %%xmm3,%%xmm1                   \n"
      "movdqa    %%xmm0,%%xmm2                   \n"
      "shufps    $0x88,%%xmm1,%%xmm0             \n"
      "shufps    $0xdd,%%xmm1,%%xmm2             \n"
      "pavgb     %%xmm2,%%xmm0                   \n"
      "movdqu    %%xmm0,(%2)                     \n"
      "lea       0x10(%2),%2                     \n"
      "sub       $0x4,%3                         \n"
      "jg        1b                              \n"
      : "+r"(src_argb),        // %0
        "+r"(src_stepx_x4),    // %1
        "+r"(dst_argb),        // %2
        "+rm"(dst_width),      // %3
        "=&r"(src_stepx_x12),  // %4
        "+r"(row1)             // %5
        ::"memory",
        "cc", "xmm0", "xmm1", "xmm2", "xmm3");
}

void ScaleARGBCols_SSE2(uint8_t* dst_argb,
                        const uint8_t* src_argb,
                        int dst_width,
                        int x,
                        int dx) {
  intptr_t x0, x1;
  asm volatile(
      "movd      %5,%%xmm2                       \n"
      "movd      %6,%%xmm3                       \n"
      "pshufd    $0x0,%%xmm2,%%xmm2              \n"
      "pshufd    $0x11,%%xmm3,%%xmm0             \n"
      "paddd     %%xmm0,%%xmm2                   \n"
      "paddd     %%xmm3,%%xmm3                   \n"
      "pshufd    $0x5,%%xmm3,%%xmm0              \n"
      "paddd     %%xmm0,%%xmm2                   \n"
      "paddd     %%xmm3,%%xmm3                   \n"
      "pshufd    $0x0,%%xmm3,%%xmm3              \n"
      "pextrw    $0x1,%%xmm2,%k0                 \n"
      "pextrw    $0x3,%%xmm2,%k1                 \n"
      "cmp       $0x0,%4                         \n"
      "jl        99f                             \n"
      "sub       $0x4,%4                         \n"
      "jl        49f                             \n"

      LABELALIGN
      "40:                                       \n"
      "movd      0x00(%3,%0,4),%%xmm0            \n"
      "movd      0x00(%3,%1,4),%%xmm1            \n"
      "pextrw    $0x5,%%xmm2,%k0                 \n"
      "pextrw    $0x7,%%xmm2,%k1                 \n"
      "paddd     %%xmm3,%%xmm2                   \n"
      "punpckldq %%xmm1,%%xmm0                   \n"
      "movd      0x00(%3,%0,4),%%xmm1            \n"
      "movd      0x00(%3,%1,4),%%xmm4            \n"
      "pextrw    $0x1,%%xmm2,%k0                 \n"
      "pextrw    $0x3,%%xmm2,%k1                 \n"
      "punpckldq %%xmm4,%%xmm1                   \n"
      "punpcklqdq %%xmm1,%%xmm0                  \n"
      "movdqu    %%xmm0,(%2)                     \n"
      "lea       0x10(%2),%2                     \n"
      "sub       $0x4,%4                         \n"
      "jge       40b                             \n"

      "49:                                       \n"
      "test      $0x2,%4                         \n"
      "je        29f                             \n"
      "movd      0x00(%3,%0,4),%%xmm0            \n"
      "movd      0x00(%3,%1,4),%%xmm1            \n"
      "pextrw    $0x5,%%xmm2,%k0                 \n"
      "punpckldq %%xmm1,%%xmm0                   \n"
      "movq      %%xmm0,(%2)                     \n"
      "lea       0x8(%2),%2                      \n"
      "29:                                       \n"
      "test      $0x1,%4                         \n"
      "je        99f                             \n"
      "movd      0x00(%3,%0,4),%%xmm0            \n"
      "movd      %%xmm0,(%2)                     \n"
      "99:                                       \n"
      : "=&a"(x0),       // %0
        "=&d"(x1),       // %1
        "+r"(dst_argb),  // %2
        "+r"(src_argb),  // %3
        "+r"(dst_width)  // %4
      : "rm"(x),         // %5
        "rm"(dx)         // %6
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4");
}

// Reads 4 pixels, duplicates them and writes 8 pixels.
// Alignment requirement: src_argb 16 byte aligned, dst_argb 16 byte aligned.
void ScaleARGBColsUp2_SSE2(uint8_t* dst_argb,
                           const uint8_t* src_argb,
                           int dst_width,
                           int x,
                           int dx) {
  (void)x;
  (void)dx;
  asm volatile(

      LABELALIGN
      "1:                                        \n"
      "movdqu    (%1),%%xmm0                     \n"
      "lea       0x10(%1),%1                     \n"
      "movdqa    %%xmm0,%%xmm1                   \n"
      "punpckldq %%xmm0,%%xmm0                   \n"
      "punpckhdq %%xmm1,%%xmm1                   \n"
      "movdqu    %%xmm0,(%0)                     \n"
      "movdqu    %%xmm1,0x10(%0)                 \n"
      "lea       0x20(%0),%0                     \n"
      "sub       $0x8,%2                         \n"
      "jg        1b                              \n"

      : "+r"(dst_argb),  // %0
        "+r"(src_argb),  // %1
        "+r"(dst_width)  // %2
        ::"memory",
        "cc", "xmm0", "xmm1");
}

// Shuffle table for arranging 2 pixels into pairs for pmaddubsw
static const uvec8 kShuffleColARGB = {
    0u, 4u,  1u, 5u,  2u,  6u,  3u,  7u,  // bbggrraa 1st pixel
    8u, 12u, 9u, 13u, 10u, 14u, 11u, 15u  // bbggrraa 2nd pixel
};

// Shuffle table for duplicating 2 fractions into 8 bytes each
static const uvec8 kShuffleFractions = {
    0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u,
};

// Bilinear row filtering combines 4x2 -> 4x1. SSSE3 version
void ScaleARGBFilterCols_SSSE3(uint8_t* dst_argb,
                               const uint8_t* src_argb,
                               int dst_width,
                               int x,
                               int dx) {
  intptr_t x0, x1;
  asm volatile(
      "movdqa    %0,%%xmm4                       \n"
      "movdqa    %1,%%xmm5                       \n"
      :
      : "m"(kShuffleColARGB),   // %0
        "m"(kShuffleFractions)  // %1
      );

  asm volatile(
      "movd      %5,%%xmm2                       \n"
      "movd      %6,%%xmm3                       \n"
      "pcmpeqb   %%xmm6,%%xmm6                   \n"
      "psrlw     $0x9,%%xmm6                     \n"
      "pextrw    $0x1,%%xmm2,%k3                 \n"
      "sub       $0x2,%2                         \n"
      "jl        29f                             \n"
      "movdqa    %%xmm2,%%xmm0                   \n"
      "paddd     %%xmm3,%%xmm0                   \n"
      "punpckldq %%xmm0,%%xmm2                   \n"
      "punpckldq %%xmm3,%%xmm3                   \n"
      "paddd     %%xmm3,%%xmm3                   \n"
      "pextrw    $0x3,%%xmm2,%k4                 \n"

      LABELALIGN
      "2:                                        \n"
      "movdqa    %%xmm2,%%xmm1                   \n"
      "paddd     %%xmm3,%%xmm2                   \n"
      "movq      0x00(%1,%3,4),%%xmm0            \n"
      "psrlw     $0x9,%%xmm1                     \n"
      "movhps    0x00(%1,%4,4),%%xmm0            \n"
      "pshufb    %%xmm5,%%xmm1                   \n"
      "pshufb    %%xmm4,%%xmm0                   \n"
      "pxor      %%xmm6,%%xmm1                   \n"
      "pmaddubsw %%xmm1,%%xmm0                   \n"
      "psrlw     $0x7,%%xmm0                     \n"
      "pextrw    $0x1,%%xmm2,%k3                 \n"
      "pextrw    $0x3,%%xmm2,%k4                 \n"
      "packuswb  %%xmm0,%%xmm0                   \n"
      "movq      %%xmm0,(%0)                     \n"
      "lea       0x8(%0),%0                      \n"
      "sub       $0x2,%2                         \n"
      "jge       2b                              \n"

      LABELALIGN
      "29:                                       \n"
      "add       $0x1,%2                         \n"
      "jl        99f                             \n"
      "psrlw     $0x9,%%xmm2                     \n"
      "movq      0x00(%1,%3,4),%%xmm0            \n"
      "pshufb    %%xmm5,%%xmm2                   \n"
      "pshufb    %%xmm4,%%xmm0                   \n"
      "pxor      %%xmm6,%%xmm2                   \n"
      "pmaddubsw %%xmm2,%%xmm0                   \n"
      "psrlw     $0x7,%%xmm0                     \n"
      "packuswb  %%xmm0,%%xmm0                   \n"
      "movd      %%xmm0,(%0)                     \n"

      LABELALIGN "99:                            \n"  // clang-format error.

      : "+r"(dst_argb),    // %0
        "+r"(src_argb),    // %1
        "+rm"(dst_width),  // %2
        "=&r"(x0),         // %3
        "=&r"(x1)          // %4
      : "rm"(x),           // %5
        "rm"(dx)           // %6
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6");
}

// Divide num by div and return as 16.16 fixed point result.
int FixedDiv_X86(int num, int div) {
  asm volatile(
      "cdq                                       \n"
      "shld      $0x10,%%eax,%%edx               \n"
      "shl       $0x10,%%eax                     \n"
      "idiv      %1                              \n"
      "mov       %0, %%eax                       \n"
      : "+a"(num)  // %0
      : "c"(div)   // %1
      : "memory", "cc", "edx");
  return num;
}

// Divide num - 1 by div - 1 and return as 16.16 fixed point result.
int FixedDiv1_X86(int num, int div) {
  asm volatile(
      "cdq                                       \n"
      "shld      $0x10,%%eax,%%edx               \n"
      "shl       $0x10,%%eax                     \n"
      "sub       $0x10001,%%eax                  \n"
      "sbb       $0x0,%%edx                      \n"
      "sub       $0x1,%1                         \n"
      "idiv      %1                              \n"
      "mov       %0, %%eax                       \n"
      : "+a"(num)  // %0
      : "c"(div)   // %1
      : "memory", "cc", "edx");
  return num;
}

#endif  // defined(__x86_64__) || defined(__i386__)

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
