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
#if !defined(LIBYUV_DISABLE_X86) &&               \
    (defined(__x86_64__) || defined(__i386__)) && \
    !defined(LIBYUV_ENABLE_ROWWIN)

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
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "lea         0x20(%0),%0                   \n"
      "psrlw       $0x8,%%xmm0                   \n"
      "psrlw       $0x8,%%xmm1                   \n"
      "packuswb    %%xmm1,%%xmm0                 \n"
      "movdqu      %%xmm0,(%1)                   \n"
      "lea         0x10(%1),%1                   \n"
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"
      : "+r"(src_ptr),   // %0
        "+r"(dst_ptr),   // %1
        "+r"(dst_width)  // %2
      :
      : "memory", "cc", "xmm0", "xmm1");
}

void ScaleRowDown2Linear_SSSE3(const uint8_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint8_t* dst_ptr,
                               int dst_width) {
  (void)src_stride;
  asm volatile(
      "pcmpeqb     %%xmm4,%%xmm4                 \n"  // 0x0101
      "pabsb       %%xmm4,%%xmm4                 \n"

      "pxor        %%xmm5,%%xmm5                 \n"

      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "lea         0x20(%0),%0                   \n"
      "pmaddubsw   %%xmm4,%%xmm0                 \n"
      "pmaddubsw   %%xmm4,%%xmm1                 \n"
      "pavgw       %%xmm5,%%xmm0                 \n"
      "pavgw       %%xmm5,%%xmm1                 \n"
      "packuswb    %%xmm1,%%xmm0                 \n"
      "movdqu      %%xmm0,(%1)                   \n"
      "lea         0x10(%1),%1                   \n"
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"
      : "+r"(src_ptr),   // %0
        "+r"(dst_ptr),   // %1
        "+r"(dst_width)  // %2
      :
      : "memory", "cc", "xmm0", "xmm1", "xmm4", "xmm5");
}

void ScaleRowDown2Box_SSSE3(const uint8_t* src_ptr,
                            ptrdiff_t src_stride,
                            uint8_t* dst_ptr,
                            int dst_width) {
  asm volatile(
      "pcmpeqb     %%xmm4,%%xmm4                 \n"  // 0x0101
      "pabsb       %%xmm4,%%xmm4                 \n"
      "pxor        %%xmm5,%%xmm5                 \n"

      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "movdqu      0x00(%0,%3,1),%%xmm2          \n"
      "movdqu      0x10(%0,%3,1),%%xmm3          \n"
      "lea         0x20(%0),%0                   \n"
      "pmaddubsw   %%xmm4,%%xmm0                 \n"
      "pmaddubsw   %%xmm4,%%xmm1                 \n"
      "pmaddubsw   %%xmm4,%%xmm2                 \n"
      "pmaddubsw   %%xmm4,%%xmm3                 \n"
      "paddw       %%xmm2,%%xmm0                 \n"
      "paddw       %%xmm3,%%xmm1                 \n"
      "psrlw       $0x1,%%xmm0                   \n"
      "psrlw       $0x1,%%xmm1                   \n"
      "pavgw       %%xmm5,%%xmm0                 \n"
      "pavgw       %%xmm5,%%xmm1                 \n"
      "packuswb    %%xmm1,%%xmm0                 \n"
      "movdqu      %%xmm0,(%1)                   \n"
      "lea         0x10(%1),%1                   \n"
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"
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
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"
      "vmovdqu     0x20(%0),%%ymm1               \n"
      "lea         0x40(%0),%0                   \n"
      "vpsrlw      $0x8,%%ymm0,%%ymm0            \n"
      "vpsrlw      $0x8,%%ymm1,%%ymm1            \n"
      "vpackuswb   %%ymm1,%%ymm0,%%ymm0          \n"
      "vpermq      $0xd8,%%ymm0,%%ymm0           \n"
      "vmovdqu     %%ymm0,(%1)                   \n"
      "lea         0x20(%1),%1                   \n"
      "sub         $0x20,%2                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_ptr),   // %0
        "+r"(dst_ptr),   // %1
        "+r"(dst_width)  // %2
      :
      : "memory", "cc", "xmm0", "xmm1");
}

void ScaleRowDown2Linear_AVX2(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_ptr,
                              int dst_width) {
  (void)src_stride;
  asm volatile(
      "vpcmpeqb    %%ymm4,%%ymm4,%%ymm4          \n"
      "vpabsb      %%ymm4,%%ymm4                 \n"
      "vpxor       %%ymm5,%%ymm5,%%ymm5          \n"

      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"
      "vmovdqu     0x20(%0),%%ymm1               \n"
      "lea         0x40(%0),%0                   \n"
      "vpmaddubsw  %%ymm4,%%ymm0,%%ymm0          \n"
      "vpmaddubsw  %%ymm4,%%ymm1,%%ymm1          \n"
      "vpavgw      %%ymm5,%%ymm0,%%ymm0          \n"
      "vpavgw      %%ymm5,%%ymm1,%%ymm1          \n"
      "vpackuswb   %%ymm1,%%ymm0,%%ymm0          \n"
      "vpermq      $0xd8,%%ymm0,%%ymm0           \n"
      "vmovdqu     %%ymm0,(%1)                   \n"
      "lea         0x20(%1),%1                   \n"
      "sub         $0x20,%2                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_ptr),   // %0
        "+r"(dst_ptr),   // %1
        "+r"(dst_width)  // %2
      :
      : "memory", "cc", "xmm0", "xmm1", "xmm4", "xmm5");
}

void ScaleRowDown2Box_AVX2(const uint8_t* src_ptr,
                           ptrdiff_t src_stride,
                           uint8_t* dst_ptr,
                           int dst_width) {
  asm volatile(
      "vpcmpeqb    %%ymm4,%%ymm4,%%ymm4          \n"
      "vpabsb      %%ymm4,%%ymm4                 \n"
      "vpxor       %%ymm5,%%ymm5,%%ymm5          \n"

      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"
      "vmovdqu     0x20(%0),%%ymm1               \n"
      "vmovdqu     0x00(%0,%3,1),%%ymm2          \n"
      "vmovdqu     0x20(%0,%3,1),%%ymm3          \n"
      "lea         0x40(%0),%0                   \n"
      "vpmaddubsw  %%ymm4,%%ymm0,%%ymm0          \n"
      "vpmaddubsw  %%ymm4,%%ymm1,%%ymm1          \n"
      "vpmaddubsw  %%ymm4,%%ymm2,%%ymm2          \n"
      "vpmaddubsw  %%ymm4,%%ymm3,%%ymm3          \n"
      "vpaddw      %%ymm2,%%ymm0,%%ymm0          \n"
      "vpaddw      %%ymm3,%%ymm1,%%ymm1          \n"
      "vpsrlw      $0x1,%%ymm0,%%ymm0            \n"
      "vpsrlw      $0x1,%%ymm1,%%ymm1            \n"
      "vpavgw      %%ymm5,%%ymm0,%%ymm0          \n"
      "vpavgw      %%ymm5,%%ymm1,%%ymm1          \n"
      "vpackuswb   %%ymm1,%%ymm0,%%ymm0          \n"
      "vpermq      $0xd8,%%ymm0,%%ymm0           \n"
      "vmovdqu     %%ymm0,(%1)                   \n"
      "lea         0x20(%1),%1                   \n"
      "sub         $0x20,%2                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
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
      "pcmpeqb     %%xmm5,%%xmm5                 \n"
      "psrld       $0x18,%%xmm5                  \n"
      "pslld       $0x10,%%xmm5                  \n"

      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "lea         0x20(%0),%0                   \n"
      "pand        %%xmm5,%%xmm0                 \n"
      "pand        %%xmm5,%%xmm1                 \n"
      "packuswb    %%xmm1,%%xmm0                 \n"
      "psrlw       $0x8,%%xmm0                   \n"
      "packuswb    %%xmm0,%%xmm0                 \n"
      "movq        %%xmm0,(%1)                   \n"
      "lea         0x8(%1),%1                    \n"
      "sub         $0x8,%2                       \n"
      "jg          1b                            \n"
      : "+r"(src_ptr),   // %0
        "+r"(dst_ptr),   // %1
        "+r"(dst_width)  // %2
      :
      : "memory", "cc", "xmm0", "xmm1", "xmm5");
}

void ScaleRowDown4Box_SSSE3(const uint8_t* src_ptr,
                            ptrdiff_t src_stride,
                            uint8_t* dst_ptr,
                            int dst_width) {
  intptr_t stridex3;
  asm volatile(
      "pcmpeqb     %%xmm4,%%xmm4                 \n"
      "pabsw       %%xmm4,%%xmm5                 \n"
      "pabsb       %%xmm4,%%xmm4                 \n"  // 0x0101
      "psllw       $0x3,%%xmm5                   \n"  // 0x0008
      "lea         0x00(%4,%4,2),%3              \n"

      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "movdqu      0x00(%0,%4,1),%%xmm2          \n"
      "movdqu      0x10(%0,%4,1),%%xmm3          \n"
      "pmaddubsw   %%xmm4,%%xmm0                 \n"
      "pmaddubsw   %%xmm4,%%xmm1                 \n"
      "pmaddubsw   %%xmm4,%%xmm2                 \n"
      "pmaddubsw   %%xmm4,%%xmm3                 \n"
      "paddw       %%xmm2,%%xmm0                 \n"
      "paddw       %%xmm3,%%xmm1                 \n"
      "movdqu      0x00(%0,%4,2),%%xmm2          \n"
      "movdqu      0x10(%0,%4,2),%%xmm3          \n"
      "pmaddubsw   %%xmm4,%%xmm2                 \n"
      "pmaddubsw   %%xmm4,%%xmm3                 \n"
      "paddw       %%xmm2,%%xmm0                 \n"
      "paddw       %%xmm3,%%xmm1                 \n"
      "movdqu      0x00(%0,%3,1),%%xmm2          \n"
      "movdqu      0x10(%0,%3,1),%%xmm3          \n"
      "lea         0x20(%0),%0                   \n"
      "pmaddubsw   %%xmm4,%%xmm2                 \n"
      "pmaddubsw   %%xmm4,%%xmm3                 \n"
      "paddw       %%xmm2,%%xmm0                 \n"
      "paddw       %%xmm3,%%xmm1                 \n"
      "phaddw      %%xmm1,%%xmm0                 \n"
      "paddw       %%xmm5,%%xmm0                 \n"
      "psrlw       $0x4,%%xmm0                   \n"
      "packuswb    %%xmm0,%%xmm0                 \n"
      "movq        %%xmm0,(%1)                   \n"
      "lea         0x8(%1),%1                    \n"
      "sub         $0x8,%2                       \n"
      "jg          1b                            \n"
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
      "vpcmpeqb    %%ymm5,%%ymm5,%%ymm5          \n"
      "vpsrld      $0x18,%%ymm5,%%ymm5           \n"
      "vpslld      $0x10,%%ymm5,%%ymm5           \n"

      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"
      "vmovdqu     0x20(%0),%%ymm1               \n"
      "lea         0x40(%0),%0                   \n"
      "vpand       %%ymm5,%%ymm0,%%ymm0          \n"
      "vpand       %%ymm5,%%ymm1,%%ymm1          \n"
      "vpackuswb   %%ymm1,%%ymm0,%%ymm0          \n"
      "vpermq      $0xd8,%%ymm0,%%ymm0           \n"
      "vpsrlw      $0x8,%%ymm0,%%ymm0            \n"
      "vpackuswb   %%ymm0,%%ymm0,%%ymm0          \n"
      "vpermq      $0xd8,%%ymm0,%%ymm0           \n"
      "vmovdqu     %%xmm0,(%1)                   \n"
      "lea         0x10(%1),%1                   \n"
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_ptr),   // %0
        "+r"(dst_ptr),   // %1
        "+r"(dst_width)  // %2
      :
      : "memory", "cc", "xmm0", "xmm1", "xmm5");
}

void ScaleRowDown4Box_AVX2(const uint8_t* src_ptr,
                           ptrdiff_t src_stride,
                           uint8_t* dst_ptr,
                           int dst_width) {
  asm volatile(
      "vpcmpeqb    %%ymm4,%%ymm4,%%ymm4          \n"
      "vpabsw      %%ymm4,%%ymm5                 \n"
      "vpabsb      %%ymm4,%%ymm4                 \n"  // 0x0101
      "vpsllw      $0x3,%%ymm5,%%ymm5            \n"  // 0x0008

      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"
      "vmovdqu     0x20(%0),%%ymm1               \n"
      "vmovdqu     0x00(%0,%3,1),%%ymm2          \n"
      "vmovdqu     0x20(%0,%3,1),%%ymm3          \n"
      "vpmaddubsw  %%ymm4,%%ymm0,%%ymm0          \n"
      "vpmaddubsw  %%ymm4,%%ymm1,%%ymm1          \n"
      "vpmaddubsw  %%ymm4,%%ymm2,%%ymm2          \n"
      "vpmaddubsw  %%ymm4,%%ymm3,%%ymm3          \n"
      "vpaddw      %%ymm2,%%ymm0,%%ymm0          \n"
      "vpaddw      %%ymm3,%%ymm1,%%ymm1          \n"
      "vmovdqu     0x00(%0,%3,2),%%ymm2          \n"
      "vmovdqu     0x20(%0,%3,2),%%ymm3          \n"
      "vpmaddubsw  %%ymm4,%%ymm2,%%ymm2          \n"
      "vpmaddubsw  %%ymm4,%%ymm3,%%ymm3          \n"
      "vpaddw      %%ymm2,%%ymm0,%%ymm0          \n"
      "vpaddw      %%ymm3,%%ymm1,%%ymm1          \n"
      "vmovdqu     0x00(%0,%4,1),%%ymm2          \n"
      "vmovdqu     0x20(%0,%4,1),%%ymm3          \n"
      "lea         0x40(%0),%0                   \n"
      "vpmaddubsw  %%ymm4,%%ymm2,%%ymm2          \n"
      "vpmaddubsw  %%ymm4,%%ymm3,%%ymm3          \n"
      "vpaddw      %%ymm2,%%ymm0,%%ymm0          \n"
      "vpaddw      %%ymm3,%%ymm1,%%ymm1          \n"
      "vphaddw     %%ymm1,%%ymm0,%%ymm0          \n"
      "vpermq      $0xd8,%%ymm0,%%ymm0           \n"
      "vpaddw      %%ymm5,%%ymm0,%%ymm0          \n"
      "vpsrlw      $0x4,%%ymm0,%%ymm0            \n"
      "vpackuswb   %%ymm0,%%ymm0,%%ymm0          \n"
      "vpermq      $0xd8,%%ymm0,%%ymm0           \n"
      "vmovdqu     %%xmm0,(%1)                   \n"
      "lea         0x10(%1),%1                   \n"
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
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
      "movdqa      %0,%%xmm3                     \n"
      "movdqa      %1,%%xmm4                     \n"
      "movdqa      %2,%%xmm5                     \n"
      :
      : "m"(kShuf0),  // %0
        "m"(kShuf1),  // %1
        "m"(kShuf2)   // %2
  );
  asm volatile(
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x10(%0),%%xmm2               \n"
      "lea         0x20(%0),%0                   \n"
      "movdqa      %%xmm2,%%xmm1                 \n"
      "palignr     $0x8,%%xmm0,%%xmm1            \n"
      "pshufb      %%xmm3,%%xmm0                 \n"
      "pshufb      %%xmm4,%%xmm1                 \n"
      "pshufb      %%xmm5,%%xmm2                 \n"
      "movq        %%xmm0,(%1)                   \n"
      "movq        %%xmm1,0x8(%1)                \n"
      "movq        %%xmm2,0x10(%1)               \n"
      "lea         0x18(%1),%1                   \n"
      "sub         $0x18,%2                      \n"
      "jg          1b                            \n"
      : "+r"(src_ptr),   // %0
        "+r"(dst_ptr),   // %1
        "+r"(dst_width)  // %2
      :
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5");
}

void ScaleRowDown34_1_Box_SSSE3(const uint8_t* src_ptr,
                                ptrdiff_t src_stride,
                                uint8_t* dst_ptr,
                                int dst_width) {
  asm volatile(
      "movdqa      %0,%%xmm2                     \n"  // kShuf01
      "movdqa      %1,%%xmm3                     \n"  // kShuf11
      "movdqa      %2,%%xmm4                     \n"  // kShuf21
      :
      : "m"(kShuf01),  // %0
        "m"(kShuf11),  // %1
        "m"(kShuf21)   // %2
  );
  asm volatile(
      "movdqa      %0,%%xmm5                     \n"  // kMadd01
      "movdqa      %1,%%xmm0                     \n"  // kMadd11
      "movdqa      %2,%%xmm1                     \n"  // kRound34
      :
      : "m"(kMadd01),  // %0
        "m"(kMadd11),  // %1
        "m"(kRound34)  // %2
  );
  asm volatile(
      "1:          \n"
      "movdqu      (%0),%%xmm6                   \n"
      "movdqu      0x00(%0,%3,1),%%xmm7          \n"
      "pavgb       %%xmm7,%%xmm6                 \n"
      "pshufb      %%xmm2,%%xmm6                 \n"
      "pmaddubsw   %%xmm5,%%xmm6                 \n"
      "paddsw      %%xmm1,%%xmm6                 \n"
      "psrlw       $0x2,%%xmm6                   \n"
      "packuswb    %%xmm6,%%xmm6                 \n"
      "movq        %%xmm6,(%1)                   \n"
      "movdqu      0x8(%0),%%xmm6                \n"
      "movdqu      0x8(%0,%3,1),%%xmm7           \n"
      "pavgb       %%xmm7,%%xmm6                 \n"
      "pshufb      %%xmm3,%%xmm6                 \n"
      "pmaddubsw   %%xmm0,%%xmm6                 \n"
      "paddsw      %%xmm1,%%xmm6                 \n"
      "psrlw       $0x2,%%xmm6                   \n"
      "packuswb    %%xmm6,%%xmm6                 \n"
      "movq        %%xmm6,0x8(%1)                \n"
      "movdqu      0x10(%0),%%xmm6               \n"
      "movdqu      0x10(%0,%3,1),%%xmm7          \n"
      "lea         0x20(%0),%0                   \n"
      "pavgb       %%xmm7,%%xmm6                 \n"
      "pshufb      %%xmm4,%%xmm6                 \n"
      "pmaddubsw   %4,%%xmm6                     \n"
      "paddsw      %%xmm1,%%xmm6                 \n"
      "psrlw       $0x2,%%xmm6                   \n"
      "packuswb    %%xmm6,%%xmm6                 \n"
      "movq        %%xmm6,0x10(%1)               \n"
      "lea         0x18(%1),%1                   \n"
      "sub         $0x18,%2                      \n"
      "jg          1b                            \n"
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
      "movdqa      %0,%%xmm2                     \n"  // kShuf01
      "movdqa      %1,%%xmm3                     \n"  // kShuf11
      "movdqa      %2,%%xmm4                     \n"  // kShuf21
      :
      : "m"(kShuf01),  // %0
        "m"(kShuf11),  // %1
        "m"(kShuf21)   // %2
  );
  asm volatile(
      "movdqa      %0,%%xmm5                     \n"  // kMadd01
      "movdqa      %1,%%xmm0                     \n"  // kMadd11
      "movdqa      %2,%%xmm1                     \n"  // kRound34
      :
      : "m"(kMadd01),  // %0
        "m"(kMadd11),  // %1
        "m"(kRound34)  // %2
  );

  asm volatile(
      "1:          \n"
      "movdqu      (%0),%%xmm6                   \n"
      "movdqu      0x00(%0,%3,1),%%xmm7          \n"
      "pavgb       %%xmm6,%%xmm7                 \n"
      "pavgb       %%xmm7,%%xmm6                 \n"
      "pshufb      %%xmm2,%%xmm6                 \n"
      "pmaddubsw   %%xmm5,%%xmm6                 \n"
      "paddsw      %%xmm1,%%xmm6                 \n"
      "psrlw       $0x2,%%xmm6                   \n"
      "packuswb    %%xmm6,%%xmm6                 \n"
      "movq        %%xmm6,(%1)                   \n"
      "movdqu      0x8(%0),%%xmm6                \n"
      "movdqu      0x8(%0,%3,1),%%xmm7           \n"
      "pavgb       %%xmm6,%%xmm7                 \n"
      "pavgb       %%xmm7,%%xmm6                 \n"
      "pshufb      %%xmm3,%%xmm6                 \n"
      "pmaddubsw   %%xmm0,%%xmm6                 \n"
      "paddsw      %%xmm1,%%xmm6                 \n"
      "psrlw       $0x2,%%xmm6                   \n"
      "packuswb    %%xmm6,%%xmm6                 \n"
      "movq        %%xmm6,0x8(%1)                \n"
      "movdqu      0x10(%0),%%xmm6               \n"
      "movdqu      0x10(%0,%3,1),%%xmm7          \n"
      "lea         0x20(%0),%0                   \n"
      "pavgb       %%xmm6,%%xmm7                 \n"
      "pavgb       %%xmm7,%%xmm6                 \n"
      "pshufb      %%xmm4,%%xmm6                 \n"
      "pmaddubsw   %4,%%xmm6                     \n"
      "paddsw      %%xmm1,%%xmm6                 \n"
      "psrlw       $0x2,%%xmm6                   \n"
      "packuswb    %%xmm6,%%xmm6                 \n"
      "movq        %%xmm6,0x10(%1)               \n"
      "lea         0x18(%1),%1                   \n"
      "sub         $0x18,%2                      \n"
      "jg          1b                            \n"
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
      "movdqa      %3,%%xmm4                     \n"
      "movdqa      %4,%%xmm5                     \n"

      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "lea         0x20(%0),%0                   \n"
      "pshufb      %%xmm4,%%xmm0                 \n"
      "pshufb      %%xmm5,%%xmm1                 \n"
      "paddusb     %%xmm1,%%xmm0                 \n"
      "movq        %%xmm0,(%1)                   \n"
      "movhlps     %%xmm0,%%xmm1                 \n"
      "movd        %%xmm1,0x8(%1)                \n"
      "lea         0xc(%1),%1                    \n"
      "sub         $0xc,%2                       \n"
      "jg          1b                            \n"
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
      "movdqa      %0,%%xmm2                     \n"
      "movdqa      %1,%%xmm3                     \n"
      "movdqa      %2,%%xmm4                     \n"
      "movdqa      %3,%%xmm5                     \n"
      :
      : "m"(kShufAb0),  // %0
        "m"(kShufAb1),  // %1
        "m"(kShufAb2),  // %2
        "m"(kScaleAb2)  // %3
  );
  asm volatile(
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x00(%0,%3,1),%%xmm1          \n"
      "lea         0x10(%0),%0                   \n"
      "pavgb       %%xmm1,%%xmm0                 \n"
      "movdqa      %%xmm0,%%xmm1                 \n"
      "pshufb      %%xmm2,%%xmm1                 \n"
      "movdqa      %%xmm0,%%xmm6                 \n"
      "pshufb      %%xmm3,%%xmm6                 \n"
      "paddusw     %%xmm6,%%xmm1                 \n"
      "pshufb      %%xmm4,%%xmm0                 \n"
      "paddusw     %%xmm0,%%xmm1                 \n"
      "pmulhuw     %%xmm5,%%xmm1                 \n"
      "packuswb    %%xmm1,%%xmm1                 \n"
      "movd        %%xmm1,(%1)                   \n"
      "psrlq       $0x10,%%xmm1                  \n"
      "movd        %%xmm1,0x2(%1)                \n"
      "lea         0x6(%1),%1                    \n"
      "sub         $0x6,%2                       \n"
      "jg          1b                            \n"
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
      "movdqa      %0,%%xmm2                     \n"
      "movdqa      %1,%%xmm3                     \n"
      "movdqa      %2,%%xmm4                     \n"
      "pxor        %%xmm5,%%xmm5                 \n"
      :
      : "m"(kShufAc),    // %0
        "m"(kShufAc3),   // %1
        "m"(kScaleAc33)  // %2
  );
  asm volatile(
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x00(%0,%3,1),%%xmm6          \n"
      "movhlps     %%xmm0,%%xmm1                 \n"
      "movhlps     %%xmm6,%%xmm7                 \n"
      "punpcklbw   %%xmm5,%%xmm0                 \n"
      "punpcklbw   %%xmm5,%%xmm1                 \n"
      "punpcklbw   %%xmm5,%%xmm6                 \n"
      "punpcklbw   %%xmm5,%%xmm7                 \n"
      "paddusw     %%xmm6,%%xmm0                 \n"
      "paddusw     %%xmm7,%%xmm1                 \n"
      "movdqu      0x00(%0,%3,2),%%xmm6          \n"
      "lea         0x10(%0),%0                   \n"
      "movhlps     %%xmm6,%%xmm7                 \n"
      "punpcklbw   %%xmm5,%%xmm6                 \n"
      "punpcklbw   %%xmm5,%%xmm7                 \n"
      "paddusw     %%xmm6,%%xmm0                 \n"
      "paddusw     %%xmm7,%%xmm1                 \n"
      "movdqa      %%xmm0,%%xmm6                 \n"
      "psrldq      $0x2,%%xmm0                   \n"
      "paddusw     %%xmm0,%%xmm6                 \n"
      "psrldq      $0x2,%%xmm0                   \n"
      "paddusw     %%xmm0,%%xmm6                 \n"
      "pshufb      %%xmm2,%%xmm6                 \n"
      "movdqa      %%xmm1,%%xmm7                 \n"
      "psrldq      $0x2,%%xmm1                   \n"
      "paddusw     %%xmm1,%%xmm7                 \n"
      "psrldq      $0x2,%%xmm1                   \n"
      "paddusw     %%xmm1,%%xmm7                 \n"
      "pshufb      %%xmm3,%%xmm7                 \n"
      "paddusw     %%xmm7,%%xmm6                 \n"
      "pmulhuw     %%xmm4,%%xmm6                 \n"
      "packuswb    %%xmm6,%%xmm6                 \n"
      "movd        %%xmm6,(%1)                   \n"
      "psrlq       $0x10,%%xmm6                  \n"
      "movd        %%xmm6,0x2(%1)                \n"
      "lea         0x6(%1),%1                    \n"
      "sub         $0x6,%2                       \n"
      "jg          1b                            \n"
      : "+r"(src_ptr),               // %0
        "+r"(dst_ptr),               // %1
        "+r"(dst_width)              // %2
      : "r"((intptr_t)(src_stride))  // %3
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}

static const uvec8 kLinearShuffleFar = {2,  3,  0, 1, 6,  7,  4,  5,
                                        10, 11, 8, 9, 14, 15, 12, 13};

static const uvec8 kLinearMadd31 = {3, 1, 1, 3, 3, 1, 1, 3,
                                    3, 1, 1, 3, 3, 1, 1, 3};

#ifdef HAS_SCALEROWUP2_LINEAR_SSE2
void ScaleRowUp2_Linear_SSE2(const uint8_t* src_ptr,
                             uint8_t* dst_ptr,
                             int dst_width) {
  asm volatile(
      "pxor        %%xmm0,%%xmm0                 \n"  // 0
      "pcmpeqw     %%xmm6,%%xmm6                 \n"
      "psrlw       $15,%%xmm6                    \n"
      "psllw       $1,%%xmm6                     \n"  // all 2

      LABELALIGN
      "1:          \n"
      "movq        (%0),%%xmm1                   \n"  // 01234567
      "movq        1(%0),%%xmm2                  \n"  // 12345678
      "movdqa      %%xmm1,%%xmm3                 \n"
      "punpcklbw   %%xmm2,%%xmm3                 \n"  // 0112233445566778
      "punpcklbw   %%xmm1,%%xmm1                 \n"  // 0011223344556677
      "punpcklbw   %%xmm2,%%xmm2                 \n"  // 1122334455667788
      "movdqa      %%xmm1,%%xmm4                 \n"
      "punpcklbw   %%xmm0,%%xmm4                 \n"  // 00112233 (16)
      "movdqa      %%xmm2,%%xmm5                 \n"
      "punpcklbw   %%xmm0,%%xmm5                 \n"  // 11223344 (16)
      "paddw       %%xmm5,%%xmm4                 \n"
      "movdqa      %%xmm3,%%xmm5                 \n"
      "paddw       %%xmm6,%%xmm4                 \n"
      "punpcklbw   %%xmm0,%%xmm5                 \n"  // 01122334 (16)
      "paddw       %%xmm5,%%xmm5                 \n"
      "paddw       %%xmm4,%%xmm5                 \n"  // 3*near+far+2 (lo)
      "psrlw       $2,%%xmm5                     \n"  // 3/4*near+1/4*far (lo)

      "punpckhbw   %%xmm0,%%xmm1                 \n"  // 44556677 (16)
      "punpckhbw   %%xmm0,%%xmm2                 \n"  // 55667788 (16)
      "paddw       %%xmm2,%%xmm1                 \n"
      "punpckhbw   %%xmm0,%%xmm3                 \n"  // 45566778 (16)
      "paddw       %%xmm6,%%xmm1                 \n"
      "paddw       %%xmm3,%%xmm3                 \n"
      "paddw       %%xmm3,%%xmm1                 \n"  // 3*near+far+2 (hi)
      "psrlw       $2,%%xmm1                     \n"  // 3/4*near+1/4*far (hi)

      "packuswb    %%xmm1,%%xmm5                 \n"
      "movdqu      %%xmm5,(%1)                   \n"

      "lea         0x8(%0),%0                    \n"
      "lea         0x10(%1),%1                   \n"  // 8 sample to 16 sample
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"
      : "+r"(src_ptr),   // %0
        "+r"(dst_ptr),   // %1
        "+r"(dst_width)  // %2
      :
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6");
}
#endif

#ifdef HAS_SCALEROWUP2_BILINEAR_SSE2
void ScaleRowUp2_Bilinear_SSE2(const uint8_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint8_t* dst_ptr,
                               ptrdiff_t dst_stride,
                               int dst_width) {
  asm volatile(
      "1:          \n"
      "pxor        %%xmm0,%%xmm0                 \n"  // 0
      // above line
      "movq        (%0),%%xmm1                   \n"  // 01234567
      "movq        1(%0),%%xmm2                  \n"  // 12345678
      "movdqa      %%xmm1,%%xmm3                 \n"
      "punpcklbw   %%xmm2,%%xmm3                 \n"  // 0112233445566778
      "punpcklbw   %%xmm1,%%xmm1                 \n"  // 0011223344556677
      "punpcklbw   %%xmm2,%%xmm2                 \n"  // 1122334455667788

      "movdqa      %%xmm1,%%xmm4                 \n"
      "punpcklbw   %%xmm0,%%xmm4                 \n"  // 00112233 (16)
      "movdqa      %%xmm2,%%xmm5                 \n"
      "punpcklbw   %%xmm0,%%xmm5                 \n"  // 11223344 (16)
      "paddw       %%xmm5,%%xmm4                 \n"  // near+far
      "movdqa      %%xmm3,%%xmm5                 \n"
      "punpcklbw   %%xmm0,%%xmm5                 \n"  // 01122334 (16)
      "paddw       %%xmm5,%%xmm5                 \n"  // 2*near
      "paddw       %%xmm5,%%xmm4                 \n"  // 3*near+far (1, lo)

      "punpckhbw   %%xmm0,%%xmm1                 \n"  // 44556677 (16)
      "punpckhbw   %%xmm0,%%xmm2                 \n"  // 55667788 (16)
      "paddw       %%xmm2,%%xmm1                 \n"
      "punpckhbw   %%xmm0,%%xmm3                 \n"  // 45566778 (16)
      "paddw       %%xmm3,%%xmm3                 \n"  // 2*near
      "paddw       %%xmm3,%%xmm1                 \n"  // 3*near+far (1, hi)

      // below line
      "movq        (%0,%3),%%xmm6                \n"  // 01234567
      "movq        1(%0,%3),%%xmm2               \n"  // 12345678
      "movdqa      %%xmm6,%%xmm3                 \n"
      "punpcklbw   %%xmm2,%%xmm3                 \n"  // 0112233445566778
      "punpcklbw   %%xmm6,%%xmm6                 \n"  // 0011223344556677
      "punpcklbw   %%xmm2,%%xmm2                 \n"  // 1122334455667788

      "movdqa      %%xmm6,%%xmm5                 \n"
      "punpcklbw   %%xmm0,%%xmm5                 \n"  // 00112233 (16)
      "movdqa      %%xmm2,%%xmm7                 \n"
      "punpcklbw   %%xmm0,%%xmm7                 \n"  // 11223344 (16)
      "paddw       %%xmm7,%%xmm5                 \n"  // near+far
      "movdqa      %%xmm3,%%xmm7                 \n"
      "punpcklbw   %%xmm0,%%xmm7                 \n"  // 01122334 (16)
      "paddw       %%xmm7,%%xmm7                 \n"  // 2*near
      "paddw       %%xmm7,%%xmm5                 \n"  // 3*near+far (2, lo)

      "punpckhbw   %%xmm0,%%xmm6                 \n"  // 44556677 (16)
      "punpckhbw   %%xmm0,%%xmm2                 \n"  // 55667788 (16)
      "paddw       %%xmm6,%%xmm2                 \n"  // near+far
      "punpckhbw   %%xmm0,%%xmm3                 \n"  // 45566778 (16)
      "paddw       %%xmm3,%%xmm3                 \n"  // 2*near
      "paddw       %%xmm3,%%xmm2                 \n"  // 3*near+far (2, hi)

      // xmm4 xmm1
      // xmm5 xmm2
      "pcmpeqw     %%xmm0,%%xmm0                 \n"
      "psrlw       $15,%%xmm0                    \n"
      "psllw       $3,%%xmm0                     \n"  // all 8

      "movdqa      %%xmm4,%%xmm3                 \n"
      "movdqa      %%xmm5,%%xmm6                 \n"
      "paddw       %%xmm3,%%xmm3                 \n"  // 6*near+2*far (1, lo)
      "paddw       %%xmm0,%%xmm6                 \n"  // 3*near+far+8 (2, lo)
      "paddw       %%xmm4,%%xmm3                 \n"  // 9*near+3*far (1, lo)
      "paddw       %%xmm6,%%xmm3                 \n"  // 9 3 3 1 + 8 (1, lo)
      "psrlw       $4,%%xmm3                     \n"  // ^ div by 16

      "movdqa      %%xmm1,%%xmm7                 \n"
      "movdqa      %%xmm2,%%xmm6                 \n"
      "paddw       %%xmm7,%%xmm7                 \n"  // 6*near+2*far (1, hi)
      "paddw       %%xmm0,%%xmm6                 \n"  // 3*near+far+8 (2, hi)
      "paddw       %%xmm1,%%xmm7                 \n"  // 9*near+3*far (1, hi)
      "paddw       %%xmm6,%%xmm7                 \n"  // 9 3 3 1 + 8 (1, hi)
      "psrlw       $4,%%xmm7                     \n"  // ^ div by 16

      "packuswb    %%xmm7,%%xmm3                 \n"
      "movdqu      %%xmm3,(%1)                   \n"  // save above line

      "movdqa      %%xmm5,%%xmm3                 \n"
      "paddw       %%xmm0,%%xmm4                 \n"  // 3*near+far+8 (1, lo)
      "paddw       %%xmm3,%%xmm3                 \n"  // 6*near+2*far (2, lo)
      "paddw       %%xmm3,%%xmm5                 \n"  // 9*near+3*far (2, lo)
      "paddw       %%xmm4,%%xmm5                 \n"  // 9 3 3 1 + 8 (lo)
      "psrlw       $4,%%xmm5                     \n"  // ^ div by 16

      "movdqa      %%xmm2,%%xmm3                 \n"
      "paddw       %%xmm0,%%xmm1                 \n"  // 3*near+far+8 (1, hi)
      "paddw       %%xmm3,%%xmm3                 \n"  // 6*near+2*far (2, hi)
      "paddw       %%xmm3,%%xmm2                 \n"  // 9*near+3*far (2, hi)
      "paddw       %%xmm1,%%xmm2                 \n"  // 9 3 3 1 + 8 (hi)
      "psrlw       $4,%%xmm2                     \n"  // ^ div by 16

      "packuswb    %%xmm2,%%xmm5                 \n"
      "movdqu      %%xmm5,(%1,%4)                \n"  // save below line

      "lea         0x8(%0),%0                    \n"
      "lea         0x10(%1),%1                   \n"  // 8 sample to 16 sample
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"
      : "+r"(src_ptr),                // %0
        "+r"(dst_ptr),                // %1
        "+r"(dst_width)               // %2
      : "r"((intptr_t)(src_stride)),  // %3
        "r"((intptr_t)(dst_stride))   // %4
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}
#endif

#ifdef HAS_SCALEROWUP2_LINEAR_12_SSSE3
void ScaleRowUp2_Linear_12_SSSE3(const uint16_t* src_ptr,
                                 uint16_t* dst_ptr,
                                 int dst_width) {
  asm volatile(
      "movdqa      %3,%%xmm5                     \n"
      "pcmpeqw     %%xmm4,%%xmm4                 \n"
      "psrlw       $15,%%xmm4                    \n"
      "psllw       $1,%%xmm4                     \n"  // all 2

      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"  // 01234567 (16)
      "movdqu      2(%0),%%xmm1                  \n"  // 12345678 (16)

      "movdqa      %%xmm0,%%xmm2                 \n"
      "punpckhwd   %%xmm1,%%xmm2                 \n"  // 45566778 (16)
      "punpcklwd   %%xmm1,%%xmm0                 \n"  // 01122334 (16)

      "movdqa      %%xmm2,%%xmm3                 \n"
      "movdqa      %%xmm0,%%xmm1                 \n"
      "pshufb      %%xmm5,%%xmm3                 \n"  // 54657687 (far)
      "pshufb      %%xmm5,%%xmm1                 \n"  // 10213243 (far)

      "paddw       %%xmm4,%%xmm1                 \n"  // far+2
      "paddw       %%xmm4,%%xmm3                 \n"  // far+2
      "paddw       %%xmm0,%%xmm1                 \n"  // near+far+2
      "paddw       %%xmm2,%%xmm3                 \n"  // near+far+2
      "paddw       %%xmm0,%%xmm0                 \n"  // 2*near
      "paddw       %%xmm2,%%xmm2                 \n"  // 2*near
      "paddw       %%xmm1,%%xmm0                 \n"  // 3*near+far+2 (lo)
      "paddw       %%xmm3,%%xmm2                 \n"  // 3*near+far+2 (hi)

      "psrlw       $2,%%xmm0                     \n"  // 3/4*near+1/4*far
      "psrlw       $2,%%xmm2                     \n"  // 3/4*near+1/4*far
      "movdqu      %%xmm0,(%1)                   \n"
      "movdqu      %%xmm2,16(%1)                 \n"

      "lea         0x10(%0),%0                   \n"
      "lea         0x20(%1),%1                   \n"  // 8 sample to 16 sample
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"
      : "+r"(src_ptr),          // %0
        "+r"(dst_ptr),          // %1
        "+r"(dst_width)         // %2
      : "m"(kLinearShuffleFar)  // %3
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5");
}
#endif

#ifdef HAS_SCALEROWUP2_BILINEAR_12_SSSE3
void ScaleRowUp2_Bilinear_12_SSSE3(const uint16_t* src_ptr,
                                   ptrdiff_t src_stride,
                                   uint16_t* dst_ptr,
                                   ptrdiff_t dst_stride,
                                   int dst_width) {
  asm volatile(
      "pcmpeqw     %%xmm7,%%xmm7                 \n"
      "psrlw       $15,%%xmm7                    \n"
      "psllw       $3,%%xmm7                     \n"  // all 8
      "movdqa      %5,%%xmm6                     \n"

      LABELALIGN
      "1:          \n"
      // above line
      "movdqu      (%0),%%xmm0                   \n"  // 01234567 (16)
      "movdqu      2(%0),%%xmm1                  \n"  // 12345678 (16)
      "movdqa      %%xmm0,%%xmm2                 \n"
      "punpckhwd   %%xmm1,%%xmm2                 \n"  // 45566778 (16)
      "punpcklwd   %%xmm1,%%xmm0                 \n"  // 01122334 (16)
      "movdqa      %%xmm2,%%xmm3                 \n"
      "movdqa      %%xmm0,%%xmm1                 \n"
      "pshufb      %%xmm6,%%xmm3                 \n"  // 54657687 (far)
      "pshufb      %%xmm6,%%xmm1                 \n"  // 10213243 (far)
      "paddw       %%xmm0,%%xmm1                 \n"  // near+far
      "paddw       %%xmm2,%%xmm3                 \n"  // near+far
      "paddw       %%xmm0,%%xmm0                 \n"  // 2*near
      "paddw       %%xmm2,%%xmm2                 \n"  // 2*near
      "paddw       %%xmm1,%%xmm0                 \n"  // 3*near+far (1, lo)
      "paddw       %%xmm3,%%xmm2                 \n"  // 3*near+far (1, hi)

      // below line
      "movdqu      (%0,%3,2),%%xmm1              \n"  // 01234567 (16)
      "movdqu      2(%0,%3,2),%%xmm4             \n"  // 12345678 (16)
      "movdqa      %%xmm1,%%xmm3                 \n"
      "punpckhwd   %%xmm4,%%xmm3                 \n"  // 45566778 (16)
      "punpcklwd   %%xmm4,%%xmm1                 \n"  // 01122334 (16)
      "movdqa      %%xmm3,%%xmm5                 \n"
      "movdqa      %%xmm1,%%xmm4                 \n"
      "pshufb      %%xmm6,%%xmm5                 \n"  // 54657687 (far)
      "pshufb      %%xmm6,%%xmm4                 \n"  // 10213243 (far)
      "paddw       %%xmm1,%%xmm4                 \n"  // near+far
      "paddw       %%xmm3,%%xmm5                 \n"  // near+far
      "paddw       %%xmm1,%%xmm1                 \n"  // 2*near
      "paddw       %%xmm3,%%xmm3                 \n"  // 2*near
      "paddw       %%xmm4,%%xmm1                 \n"  // 3*near+far (2, lo)
      "paddw       %%xmm5,%%xmm3                 \n"  // 3*near+far (2, hi)

      // xmm0 xmm2
      // xmm1 xmm3

      "movdqa      %%xmm0,%%xmm4                 \n"
      "movdqa      %%xmm1,%%xmm5                 \n"
      "paddw       %%xmm4,%%xmm4                 \n"  // 6*near+2*far (1, lo)
      "paddw       %%xmm7,%%xmm5                 \n"  // 3*near+far+8 (2, lo)
      "paddw       %%xmm0,%%xmm4                 \n"  // 9*near+3*far (1, lo)
      "paddw       %%xmm5,%%xmm4                 \n"  // 9 3 3 1 + 8 (1, lo)
      "psrlw       $4,%%xmm4                     \n"  // ^ div by 16
      "movdqu      %%xmm4,(%1)                   \n"

      "movdqa      %%xmm2,%%xmm4                 \n"
      "movdqa      %%xmm3,%%xmm5                 \n"
      "paddw       %%xmm4,%%xmm4                 \n"  // 6*near+2*far (1, hi)
      "paddw       %%xmm7,%%xmm5                 \n"  // 3*near+far+8 (2, hi)
      "paddw       %%xmm2,%%xmm4                 \n"  // 9*near+3*far (1, hi)
      "paddw       %%xmm5,%%xmm4                 \n"  // 9 3 3 1 + 8 (1, hi)
      "psrlw       $4,%%xmm4                     \n"  // ^ div by 16
      "movdqu      %%xmm4,0x10(%1)               \n"

      "movdqa      %%xmm1,%%xmm4                 \n"
      "paddw       %%xmm7,%%xmm0                 \n"  // 3*near+far+8 (1, lo)
      "paddw       %%xmm4,%%xmm4                 \n"  // 6*near+2*far (2, lo)
      "paddw       %%xmm4,%%xmm1                 \n"  // 9*near+3*far (2, lo)
      "paddw       %%xmm0,%%xmm1                 \n"  // 9 3 3 1 + 8 (2, lo)
      "psrlw       $4,%%xmm1                     \n"  // ^ div by 16
      "movdqu      %%xmm1,(%1,%4,2)              \n"

      "movdqa      %%xmm3,%%xmm4                 \n"
      "paddw       %%xmm7,%%xmm2                 \n"  // 3*near+far+8 (1, hi)
      "paddw       %%xmm4,%%xmm4                 \n"  // 6*near+2*far (2, hi)
      "paddw       %%xmm4,%%xmm3                 \n"  // 9*near+3*far (2, hi)
      "paddw       %%xmm2,%%xmm3                 \n"  // 9 3 3 1 + 8 (2, hi)
      "psrlw       $4,%%xmm3                     \n"  // ^ div by 16
      "movdqu      %%xmm3,0x10(%1,%4,2)          \n"

      "lea         0x10(%0),%0                   \n"
      "lea         0x20(%1),%1                   \n"  // 8 sample to 16 sample
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"
      : "+r"(src_ptr),                // %0
        "+r"(dst_ptr),                // %1
        "+r"(dst_width)               // %2
      : "r"((intptr_t)(src_stride)),  // %3
        "r"((intptr_t)(dst_stride)),  // %4
        "m"(kLinearShuffleFar)        // %5
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}
#endif

#ifdef HAS_SCALEROWUP2_LINEAR_16_SSE2
void ScaleRowUp2_Linear_16_SSE2(const uint16_t* src_ptr,
                                uint16_t* dst_ptr,
                                int dst_width) {
  asm volatile(
      "pxor        %%xmm5,%%xmm5                 \n"
      "pcmpeqd     %%xmm4,%%xmm4                 \n"
      "psrld       $31,%%xmm4                    \n"
      "pslld       $1,%%xmm4                     \n"  // all 2

      LABELALIGN
      "1:          \n"
      "movq        (%0),%%xmm0                   \n"  // 0123 (16b)
      "movq        2(%0),%%xmm1                  \n"  // 1234 (16b)

      "punpcklwd   %%xmm5,%%xmm0                 \n"  // 0123 (32b)
      "punpcklwd   %%xmm5,%%xmm1                 \n"  // 1234 (32b)

      "movdqa      %%xmm0,%%xmm2                 \n"
      "movdqa      %%xmm1,%%xmm3                 \n"

      "pshufd      $0b10110001,%%xmm2,%%xmm2     \n"  // 1032 (even, far)
      "pshufd      $0b10110001,%%xmm3,%%xmm3     \n"  // 2143 (odd, far)

      "paddd       %%xmm4,%%xmm2                 \n"  // far+2 (lo)
      "paddd       %%xmm4,%%xmm3                 \n"  // far+2 (hi)
      "paddd       %%xmm0,%%xmm2                 \n"  // near+far+2 (lo)
      "paddd       %%xmm1,%%xmm3                 \n"  // near+far+2 (hi)
      "paddd       %%xmm0,%%xmm0                 \n"  // 2*near (lo)
      "paddd       %%xmm1,%%xmm1                 \n"  // 2*near (hi)
      "paddd       %%xmm2,%%xmm0                 \n"  // 3*near+far+2 (lo)
      "paddd       %%xmm3,%%xmm1                 \n"  // 3*near+far+2 (hi)

      "psrld       $2,%%xmm0                     \n"  // 3/4*near+1/4*far (lo)
      "psrld       $2,%%xmm1                     \n"  // 3/4*near+1/4*far (hi)
      "packssdw    %%xmm1,%%xmm0                 \n"
      "pshufd      $0b11011000,%%xmm0,%%xmm0     \n"
      "movdqu      %%xmm0,(%1)                   \n"

      "lea         0x8(%0),%0                    \n"
      "lea         0x10(%1),%1                   \n"  // 4 pixel to 8 pixel
      "sub         $0x8,%2                       \n"
      "jg          1b                            \n"
      : "+r"(src_ptr),   // %0
        "+r"(dst_ptr),   // %1
        "+r"(dst_width)  // %2
      :
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5");
}
#endif

#ifdef HAS_SCALEROWUP2_BILINEAR_16_SSE2
void ScaleRowUp2_Bilinear_16_SSE2(const uint16_t* src_ptr,
                                  ptrdiff_t src_stride,
                                  uint16_t* dst_ptr,
                                  ptrdiff_t dst_stride,
                                  int dst_width) {
  asm volatile(
      "pxor        %%xmm7,%%xmm7                 \n"
      "pcmpeqd     %%xmm6,%%xmm6                 \n"
      "psrld       $31,%%xmm6                    \n"
      "pslld       $3,%%xmm6                     \n"  // all 8

      LABELALIGN
      "1:          \n"
      "movq        (%0),%%xmm0                   \n"  // 0011 (16b, 1u1v)
      "movq        4(%0),%%xmm1                  \n"  // 1122 (16b, 1u1v)
      "punpcklwd   %%xmm7,%%xmm0                 \n"  // 0011 (near) (32b, 1u1v)
      "punpcklwd   %%xmm7,%%xmm1                 \n"  // 1122 (near) (32b, 1u1v)
      "movdqa      %%xmm0,%%xmm2                 \n"
      "movdqa      %%xmm1,%%xmm3                 \n"
      "pshufd      $0b01001110,%%xmm2,%%xmm2     \n"  // 1100 (far) (1, lo)
      "pshufd      $0b01001110,%%xmm3,%%xmm3     \n"  // 2211 (far) (1, hi)
      "paddd       %%xmm0,%%xmm2                 \n"  // near+far (1, lo)
      "paddd       %%xmm1,%%xmm3                 \n"  // near+far (1, hi)
      "paddd       %%xmm0,%%xmm0                 \n"  // 2*near (1, lo)
      "paddd       %%xmm1,%%xmm1                 \n"  // 2*near (1, hi)
      "paddd       %%xmm2,%%xmm0                 \n"  // 3*near+far (1, lo)
      "paddd       %%xmm3,%%xmm1                 \n"  // 3*near+far (1, hi)

      "movq        (%0),%%xmm0                   \n"  // 0123 (16b)
      "movq        2(%0),%%xmm1                  \n"  // 1234 (16b)
      "punpcklwd   %%xmm7,%%xmm0                 \n"  // 0123 (32b)
      "punpcklwd   %%xmm7,%%xmm1                 \n"  // 1234 (32b)
      "movdqa      %%xmm0,%%xmm2                 \n"
      "movdqa      %%xmm1,%%xmm3                 \n"
      "pshufd      $0b10110001,%%xmm2,%%xmm2     \n"  // 1032 (even, far)
      "pshufd      $0b10110001,%%xmm3,%%xmm3     \n"  // 2143 (odd, far)
      "paddd       %%xmm0,%%xmm2                 \n"  // near+far (lo)
      "paddd       %%xmm1,%%xmm3                 \n"  // near+far (hi)
      "paddd       %%xmm0,%%xmm0                 \n"  // 2*near (lo)
      "paddd       %%xmm1,%%xmm1                 \n"  // 2*near (hi)
      "paddd       %%xmm2,%%xmm0                 \n"  // 3*near+far (1, lo)
      "paddd       %%xmm3,%%xmm1                 \n"  // 3*near+far (1, hi)

      "movq        (%0,%3,2),%%xmm2              \n"
      "movq        2(%0,%3,2),%%xmm3             \n"
      "punpcklwd   %%xmm7,%%xmm2                 \n"  // 0123 (32b)
      "punpcklwd   %%xmm7,%%xmm3                 \n"  // 1234 (32b)
      "movdqa      %%xmm2,%%xmm4                 \n"
      "movdqa      %%xmm3,%%xmm5                 \n"
      "pshufd      $0b10110001,%%xmm4,%%xmm4     \n"  // 1032 (even, far)
      "pshufd      $0b10110001,%%xmm5,%%xmm5     \n"  // 2143 (odd, far)
      "paddd       %%xmm2,%%xmm4                 \n"  // near+far (lo)
      "paddd       %%xmm3,%%xmm5                 \n"  // near+far (hi)
      "paddd       %%xmm2,%%xmm2                 \n"  // 2*near (lo)
      "paddd       %%xmm3,%%xmm3                 \n"  // 2*near (hi)
      "paddd       %%xmm4,%%xmm2                 \n"  // 3*near+far (2, lo)
      "paddd       %%xmm5,%%xmm3                 \n"  // 3*near+far (2, hi)

      "movdqa      %%xmm0,%%xmm4                 \n"
      "movdqa      %%xmm2,%%xmm5                 \n"
      "paddd       %%xmm0,%%xmm4                 \n"  // 6*near+2*far (1, lo)
      "paddd       %%xmm6,%%xmm5                 \n"  // 3*near+far+8 (2, lo)
      "paddd       %%xmm0,%%xmm4                 \n"  // 9*near+3*far (1, lo)
      "paddd       %%xmm5,%%xmm4                 \n"  // 9 3 3 1 + 8 (1, lo)
      "psrld       $4,%%xmm4                     \n"  // ^ div by 16 (1, lo)

      "movdqa      %%xmm2,%%xmm5                 \n"
      "paddd       %%xmm2,%%xmm5                 \n"  // 6*near+2*far (2, lo)
      "paddd       %%xmm6,%%xmm0                 \n"  // 3*near+far+8 (1, lo)
      "paddd       %%xmm2,%%xmm5                 \n"  // 9*near+3*far (2, lo)
      "paddd       %%xmm0,%%xmm5                 \n"  // 9 3 3 1 + 8 (2, lo)
      "psrld       $4,%%xmm5                     \n"  // ^ div by 16 (2, lo)

      "movdqa      %%xmm1,%%xmm0                 \n"
      "movdqa      %%xmm3,%%xmm2                 \n"
      "paddd       %%xmm1,%%xmm0                 \n"  // 6*near+2*far (1, hi)
      "paddd       %%xmm6,%%xmm2                 \n"  // 3*near+far+8 (2, hi)
      "paddd       %%xmm1,%%xmm0                 \n"  // 9*near+3*far (1, hi)
      "paddd       %%xmm2,%%xmm0                 \n"  // 9 3 3 1 + 8 (1, hi)
      "psrld       $4,%%xmm0                     \n"  // ^ div by 16 (1, hi)

      "movdqa      %%xmm3,%%xmm2                 \n"
      "paddd       %%xmm3,%%xmm2                 \n"  // 6*near+2*far (2, hi)
      "paddd       %%xmm6,%%xmm1                 \n"  // 3*near+far+8 (1, hi)
      "paddd       %%xmm3,%%xmm2                 \n"  // 9*near+3*far (2, hi)
      "paddd       %%xmm1,%%xmm2                 \n"  // 9 3 3 1 + 8 (2, hi)
      "psrld       $4,%%xmm2                     \n"  // ^ div by 16 (2, hi)

      "packssdw    %%xmm0,%%xmm4                 \n"
      "pshufd      $0b11011000,%%xmm4,%%xmm4     \n"
      "movdqu      %%xmm4,(%1)                   \n"  // store above
      "packssdw    %%xmm2,%%xmm5                 \n"
      "pshufd      $0b11011000,%%xmm5,%%xmm5     \n"
      "movdqu      %%xmm5,(%1,%4,2)              \n"  // store below

      "lea         0x8(%0),%0                    \n"
      "lea         0x10(%1),%1                   \n"  // 4 pixel to 8 pixel
      "sub         $0x8,%2                       \n"
      "jg          1b                            \n"
      : "+r"(src_ptr),                // %0
        "+r"(dst_ptr),                // %1
        "+r"(dst_width)               // %2
      : "r"((intptr_t)(src_stride)),  // %3
        "r"((intptr_t)(dst_stride))   // %4
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}
#endif

#ifdef HAS_SCALEROWUP2_LINEAR_SSSE3
void ScaleRowUp2_Linear_SSSE3(const uint8_t* src_ptr,
                              uint8_t* dst_ptr,
                              int dst_width) {
  asm volatile(
      "pcmpeqw     %%xmm4,%%xmm4                 \n"
      "psrlw       $15,%%xmm4                    \n"
      "psllw       $1,%%xmm4                     \n"  // all 2
      "movdqa      %3,%%xmm3                     \n"

      LABELALIGN
      "1:          \n"
      "movq        (%0),%%xmm0                   \n"  // 01234567
      "movq        1(%0),%%xmm1                  \n"  // 12345678
      "punpcklwd   %%xmm0,%%xmm0                 \n"  // 0101232345456767
      "punpcklwd   %%xmm1,%%xmm1                 \n"  // 1212343456567878
      "movdqa      %%xmm0,%%xmm2                 \n"
      "punpckhdq   %%xmm1,%%xmm2                 \n"  // 4545565667677878
      "punpckldq   %%xmm1,%%xmm0                 \n"  // 0101121223233434
      "pmaddubsw   %%xmm3,%%xmm2                 \n"  // 3*near+far (hi)
      "pmaddubsw   %%xmm3,%%xmm0                 \n"  // 3*near+far (lo)
      "paddw       %%xmm4,%%xmm0                 \n"  // 3*near+far+2 (lo)
      "paddw       %%xmm4,%%xmm2                 \n"  // 3*near+far+2 (hi)
      "psrlw       $2,%%xmm0                     \n"  // 3/4*near+1/4*far (lo)
      "psrlw       $2,%%xmm2                     \n"  // 3/4*near+1/4*far (hi)
      "packuswb    %%xmm2,%%xmm0                 \n"
      "movdqu      %%xmm0,(%1)                   \n"
      "lea         0x8(%0),%0                    \n"
      "lea         0x10(%1),%1                   \n"  // 8 sample to 16 sample
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"
      : "+r"(src_ptr),      // %0
        "+r"(dst_ptr),      // %1
        "+r"(dst_width)     // %2
      : "m"(kLinearMadd31)  // %3
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4");
}
#endif

#ifdef HAS_SCALEROWUP2_BILINEAR_SSSE3
void ScaleRowUp2_Bilinear_SSSE3(const uint8_t* src_ptr,
                                ptrdiff_t src_stride,
                                uint8_t* dst_ptr,
                                ptrdiff_t dst_stride,
                                int dst_width) {
  asm volatile(
      "pcmpeqw     %%xmm6,%%xmm6                 \n"
      "psrlw       $15,%%xmm6                    \n"
      "psllw       $3,%%xmm6                     \n"  // all 8
      "movdqa      %5,%%xmm7                     \n"

      LABELALIGN
      "1:          \n"
      "movq        (%0),%%xmm0                   \n"  // 01234567
      "movq        1(%0),%%xmm1                  \n"  // 12345678
      "punpcklwd   %%xmm0,%%xmm0                 \n"  // 0101232345456767
      "punpcklwd   %%xmm1,%%xmm1                 \n"  // 1212343456567878
      "movdqa      %%xmm0,%%xmm2                 \n"
      "punpckhdq   %%xmm1,%%xmm2                 \n"  // 4545565667677878
      "punpckldq   %%xmm1,%%xmm0                 \n"  // 0101121223233434
      "pmaddubsw   %%xmm7,%%xmm2                 \n"  // 3*near+far (1, hi)
      "pmaddubsw   %%xmm7,%%xmm0                 \n"  // 3*near+far (1, lo)

      "movq        (%0,%3),%%xmm1                \n"
      "movq        1(%0,%3),%%xmm4               \n"
      "punpcklwd   %%xmm1,%%xmm1                 \n"
      "punpcklwd   %%xmm4,%%xmm4                 \n"
      "movdqa      %%xmm1,%%xmm3                 \n"
      "punpckhdq   %%xmm4,%%xmm3                 \n"
      "punpckldq   %%xmm4,%%xmm1                 \n"
      "pmaddubsw   %%xmm7,%%xmm3                 \n"  // 3*near+far (2, hi)
      "pmaddubsw   %%xmm7,%%xmm1                 \n"  // 3*near+far (2, lo)

      // xmm0 xmm2
      // xmm1 xmm3

      "movdqa      %%xmm0,%%xmm4                 \n"
      "movdqa      %%xmm1,%%xmm5                 \n"
      "paddw       %%xmm0,%%xmm4                 \n"  // 6*near+2*far (1, lo)
      "paddw       %%xmm6,%%xmm5                 \n"  // 3*near+far+8 (2, lo)
      "paddw       %%xmm0,%%xmm4                 \n"  // 9*near+3*far (1, lo)
      "paddw       %%xmm5,%%xmm4                 \n"  // 9 3 3 1 + 8 (1, lo)
      "psrlw       $4,%%xmm4                     \n"  // ^ div by 16 (1, lo)

      "movdqa      %%xmm1,%%xmm5                 \n"
      "paddw       %%xmm1,%%xmm5                 \n"  // 6*near+2*far (2, lo)
      "paddw       %%xmm6,%%xmm0                 \n"  // 3*near+far+8 (1, lo)
      "paddw       %%xmm1,%%xmm5                 \n"  // 9*near+3*far (2, lo)
      "paddw       %%xmm0,%%xmm5                 \n"  // 9 3 3 1 + 8 (2, lo)
      "psrlw       $4,%%xmm5                     \n"  // ^ div by 16 (2, lo)

      "movdqa      %%xmm2,%%xmm0                 \n"
      "movdqa      %%xmm3,%%xmm1                 \n"
      "paddw       %%xmm2,%%xmm0                 \n"  // 6*near+2*far (1, hi)
      "paddw       %%xmm6,%%xmm1                 \n"  // 3*near+far+8 (2, hi)
      "paddw       %%xmm2,%%xmm0                 \n"  // 9*near+3*far (1, hi)
      "paddw       %%xmm1,%%xmm0                 \n"  // 9 3 3 1 + 8 (1, hi)
      "psrlw       $4,%%xmm0                     \n"  // ^ div by 16 (1, hi)

      "movdqa      %%xmm3,%%xmm1                 \n"
      "paddw       %%xmm3,%%xmm1                 \n"  // 6*near+2*far (2, hi)
      "paddw       %%xmm6,%%xmm2                 \n"  // 3*near+far+8 (1, hi)
      "paddw       %%xmm3,%%xmm1                 \n"  // 9*near+3*far (2, hi)
      "paddw       %%xmm2,%%xmm1                 \n"  // 9 3 3 1 + 8 (2, hi)
      "psrlw       $4,%%xmm1                     \n"  // ^ div by 16 (2, hi)

      "packuswb    %%xmm0,%%xmm4                 \n"
      "movdqu      %%xmm4,(%1)                   \n"  // store above
      "packuswb    %%xmm1,%%xmm5                 \n"
      "movdqu      %%xmm5,(%1,%4)                \n"  // store below

      "lea         0x8(%0),%0                    \n"
      "lea         0x10(%1),%1                   \n"  // 8 sample to 16 sample
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"
      : "+r"(src_ptr),                // %0
        "+r"(dst_ptr),                // %1
        "+r"(dst_width)               // %2
      : "r"((intptr_t)(src_stride)),  // %3
        "r"((intptr_t)(dst_stride)),  // %4
        "m"(kLinearMadd31)            // %5
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}
#endif

#ifdef HAS_SCALEROWUP2_LINEAR_AVX2
void ScaleRowUp2_Linear_AVX2(const uint8_t* src_ptr,
                             uint8_t* dst_ptr,
                             int dst_width) {
  asm volatile(
      "vpcmpeqw    %%ymm4,%%ymm4,%%ymm4          \n"
      "vpsrlw      $15,%%ymm4,%%ymm4             \n"
      "vpsllw      $1,%%ymm4,%%ymm4              \n"  // all 2
      "vbroadcastf128 %3,%%ymm3                  \n"

      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%xmm0                   \n"  // 0123456789ABCDEF
      "vmovdqu     1(%0),%%xmm1                  \n"  // 123456789ABCDEF0
      "vpermq      $0b11011000,%%ymm0,%%ymm0     \n"
      "vpermq      $0b11011000,%%ymm1,%%ymm1     \n"
      "vpunpcklwd  %%ymm0,%%ymm0,%%ymm0          \n"
      "vpunpcklwd  %%ymm1,%%ymm1,%%ymm1          \n"
      "vpunpckhdq  %%ymm1,%%ymm0,%%ymm2          \n"
      "vpunpckldq  %%ymm1,%%ymm0,%%ymm0          \n"
      "vpmaddubsw  %%ymm3,%%ymm2,%%ymm1          \n"  // 3*near+far (hi)
      "vpmaddubsw  %%ymm3,%%ymm0,%%ymm0          \n"  // 3*near+far (lo)
      "vpaddw      %%ymm4,%%ymm0,%%ymm0          \n"  // 3*near+far+2 (lo)
      "vpaddw      %%ymm4,%%ymm1,%%ymm1          \n"  // 3*near+far+2 (hi)
      "vpsrlw      $2,%%ymm0,%%ymm0              \n"  // 3/4*near+1/4*far (lo)
      "vpsrlw      $2,%%ymm1,%%ymm1              \n"  // 3/4*near+1/4*far (hi)
      "vpackuswb   %%ymm1,%%ymm0,%%ymm0          \n"
      "vmovdqu     %%ymm0,(%1)                   \n"

      "lea         0x10(%0),%0                   \n"
      "lea         0x20(%1),%1                   \n"  // 16 sample to 32 sample
      "sub         $0x20,%2                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_ptr),      // %0
        "+r"(dst_ptr),      // %1
        "+r"(dst_width)     // %2
      : "m"(kLinearMadd31)  // %3
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4");
}
#endif

#ifdef HAS_SCALEROWUP2_BILINEAR_AVX2
void ScaleRowUp2_Bilinear_AVX2(const uint8_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint8_t* dst_ptr,
                               ptrdiff_t dst_stride,
                               int dst_width) {
  asm volatile(
      "vpcmpeqw    %%ymm6,%%ymm6,%%ymm6          \n"
      "vpsrlw      $15,%%ymm6,%%ymm6             \n"
      "vpsllw      $3,%%ymm6,%%ymm6              \n"  // all 8
      "vbroadcastf128 %5,%%ymm7                  \n"

      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%xmm0                   \n"  // 0123456789ABCDEF
      "vmovdqu     1(%0),%%xmm1                  \n"  // 123456789ABCDEF0
      "vpermq      $0b11011000,%%ymm0,%%ymm0     \n"
      "vpermq      $0b11011000,%%ymm1,%%ymm1     \n"
      "vpunpcklwd  %%ymm0,%%ymm0,%%ymm0          \n"
      "vpunpcklwd  %%ymm1,%%ymm1,%%ymm1          \n"
      "vpunpckhdq  %%ymm1,%%ymm0,%%ymm2          \n"
      "vpunpckldq  %%ymm1,%%ymm0,%%ymm0          \n"
      "vpmaddubsw  %%ymm7,%%ymm2,%%ymm1          \n"  // 3*near+far (1, hi)
      "vpmaddubsw  %%ymm7,%%ymm0,%%ymm0          \n"  // 3*near+far (1, lo)

      "vmovdqu     (%0,%3),%%xmm2                \n"  // 0123456789ABCDEF
      "vmovdqu     1(%0,%3),%%xmm3               \n"  // 123456789ABCDEF0
      "vpermq      $0b11011000,%%ymm2,%%ymm2     \n"
      "vpermq      $0b11011000,%%ymm3,%%ymm3     \n"
      "vpunpcklwd  %%ymm2,%%ymm2,%%ymm2          \n"
      "vpunpcklwd  %%ymm3,%%ymm3,%%ymm3          \n"
      "vpunpckhdq  %%ymm3,%%ymm2,%%ymm4          \n"
      "vpunpckldq  %%ymm3,%%ymm2,%%ymm2          \n"
      "vpmaddubsw  %%ymm7,%%ymm4,%%ymm3          \n"  // 3*near+far (2, hi)
      "vpmaddubsw  %%ymm7,%%ymm2,%%ymm2          \n"  // 3*near+far (2, lo)

      // ymm0 ymm1
      // ymm2 ymm3

      "vpaddw      %%ymm0,%%ymm0,%%ymm4          \n"  // 6*near+2*far (1, lo)
      "vpaddw      %%ymm6,%%ymm2,%%ymm5          \n"  // 3*near+far+8 (2, lo)
      "vpaddw      %%ymm4,%%ymm0,%%ymm4          \n"  // 9*near+3*far (1, lo)
      "vpaddw      %%ymm4,%%ymm5,%%ymm4          \n"  // 9 3 3 1 + 8 (1, lo)
      "vpsrlw      $4,%%ymm4,%%ymm4              \n"  // ^ div by 16 (1, lo)

      "vpaddw      %%ymm2,%%ymm2,%%ymm5          \n"  // 6*near+2*far (2, lo)
      "vpaddw      %%ymm6,%%ymm0,%%ymm0          \n"  // 3*near+far+8 (1, lo)
      "vpaddw      %%ymm5,%%ymm2,%%ymm5          \n"  // 9*near+3*far (2, lo)
      "vpaddw      %%ymm5,%%ymm0,%%ymm5          \n"  // 9 3 3 1 + 8 (2, lo)
      "vpsrlw      $4,%%ymm5,%%ymm5              \n"  // ^ div by 16 (2, lo)

      "vpaddw      %%ymm1,%%ymm1,%%ymm0          \n"  // 6*near+2*far (1, hi)
      "vpaddw      %%ymm6,%%ymm3,%%ymm2          \n"  // 3*near+far+8 (2, hi)
      "vpaddw      %%ymm0,%%ymm1,%%ymm0          \n"  // 9*near+3*far (1, hi)
      "vpaddw      %%ymm0,%%ymm2,%%ymm0          \n"  // 9 3 3 1 + 8 (1, hi)
      "vpsrlw      $4,%%ymm0,%%ymm0              \n"  // ^ div by 16 (1, hi)

      "vpaddw      %%ymm3,%%ymm3,%%ymm2          \n"  // 6*near+2*far (2, hi)
      "vpaddw      %%ymm6,%%ymm1,%%ymm1          \n"  // 3*near+far+8 (1, hi)
      "vpaddw      %%ymm2,%%ymm3,%%ymm2          \n"  // 9*near+3*far (2, hi)
      "vpaddw      %%ymm2,%%ymm1,%%ymm2          \n"  // 9 3 3 1 + 8 (2, hi)
      "vpsrlw      $4,%%ymm2,%%ymm2              \n"  // ^ div by 16 (2, hi)

      "vpackuswb   %%ymm0,%%ymm4,%%ymm4          \n"
      "vmovdqu     %%ymm4,(%1)                   \n"  // store above
      "vpackuswb   %%ymm2,%%ymm5,%%ymm5          \n"
      "vmovdqu     %%ymm5,(%1,%4)                \n"  // store below

      "lea         0x10(%0),%0                   \n"
      "lea         0x20(%1),%1                   \n"  // 16 sample to 32 sample
      "sub         $0x20,%2                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_ptr),                // %0
        "+r"(dst_ptr),                // %1
        "+r"(dst_width)               // %2
      : "r"((intptr_t)(src_stride)),  // %3
        "r"((intptr_t)(dst_stride)),  // %4
        "m"(kLinearMadd31)            // %5
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}
#endif

#ifdef HAS_SCALEROWUP2_LINEAR_12_AVX2
void ScaleRowUp2_Linear_12_AVX2(const uint16_t* src_ptr,
                                uint16_t* dst_ptr,
                                int dst_width) {
  asm volatile(
      "vbroadcastf128 %3,%%ymm5                  \n"
      "vpcmpeqw    %%ymm4,%%ymm4,%%ymm4          \n"
      "vpsrlw      $15,%%ymm4,%%ymm4             \n"
      "vpsllw      $1,%%ymm4,%%ymm4              \n"  // all 2

      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"  // 0123456789ABCDEF (16b)
      "vmovdqu     2(%0),%%ymm1                  \n"  // 123456789ABCDEF0 (16b)

      "vpermq      $0b11011000,%%ymm0,%%ymm0     \n"  // 012389AB4567CDEF
      "vpermq      $0b11011000,%%ymm1,%%ymm1     \n"  // 12349ABC5678DEF0

      "vpunpckhwd  %%ymm1,%%ymm0,%%ymm2          \n"  // 899AABBCCDDEEFF0 (near)
      "vpunpcklwd  %%ymm1,%%ymm0,%%ymm0          \n"  // 0112233445566778 (near)
      "vpshufb     %%ymm5,%%ymm2,%%ymm3          \n"  // 98A9BACBDCEDFE0F (far)
      "vpshufb     %%ymm5,%%ymm0,%%ymm1          \n"  // 1021324354657687 (far)

      "vpaddw      %%ymm4,%%ymm1,%%ymm1          \n"  // far+2
      "vpaddw      %%ymm4,%%ymm3,%%ymm3          \n"  // far+2
      "vpaddw      %%ymm0,%%ymm1,%%ymm1          \n"  // near+far+2
      "vpaddw      %%ymm2,%%ymm3,%%ymm3          \n"  // near+far+2
      "vpaddw      %%ymm0,%%ymm0,%%ymm0          \n"  // 2*near
      "vpaddw      %%ymm2,%%ymm2,%%ymm2          \n"  // 2*near
      "vpaddw      %%ymm0,%%ymm1,%%ymm0          \n"  // 3*near+far+2
      "vpaddw      %%ymm2,%%ymm3,%%ymm2          \n"  // 3*near+far+2

      "vpsrlw      $2,%%ymm0,%%ymm0              \n"  // 3/4*near+1/4*far
      "vpsrlw      $2,%%ymm2,%%ymm2              \n"  // 3/4*near+1/4*far
      "vmovdqu     %%ymm0,(%1)                   \n"
      "vmovdqu     %%ymm2,32(%1)                 \n"

      "lea         0x20(%0),%0                   \n"
      "lea         0x40(%1),%1                   \n"  // 16 sample to 32 sample
      "sub         $0x20,%2                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_ptr),          // %0
        "+r"(dst_ptr),          // %1
        "+r"(dst_width)         // %2
      : "m"(kLinearShuffleFar)  // %3
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5");
}
#endif

#ifdef HAS_SCALEROWUP2_BILINEAR_12_AVX2
void ScaleRowUp2_Bilinear_12_AVX2(const uint16_t* src_ptr,
                                  ptrdiff_t src_stride,
                                  uint16_t* dst_ptr,
                                  ptrdiff_t dst_stride,
                                  int dst_width) {
  asm volatile(
      "vbroadcastf128 %5,%%ymm5                  \n"
      "vpcmpeqw    %%ymm4,%%ymm4,%%ymm4          \n"
      "vpsrlw      $15,%%ymm4,%%ymm4             \n"
      "vpsllw      $3,%%ymm4,%%ymm4              \n"  // all 8

      LABELALIGN
      "1:          \n"

      "vmovdqu     (%0),%%xmm0                   \n"  // 01234567 (16b)
      "vmovdqu     2(%0),%%xmm1                  \n"  // 12345678 (16b)
      "vpermq      $0b11011000,%%ymm0,%%ymm0     \n"  // 0123000045670000
      "vpermq      $0b11011000,%%ymm1,%%ymm1     \n"  // 1234000056780000
      "vpunpcklwd  %%ymm1,%%ymm0,%%ymm0          \n"  // 0112233445566778 (near)
      "vpshufb     %%ymm5,%%ymm0,%%ymm1          \n"  // 1021324354657687 (far)
      "vpaddw      %%ymm0,%%ymm1,%%ymm1          \n"  // near+far
      "vpaddw      %%ymm0,%%ymm0,%%ymm0          \n"  // 2*near
      "vpaddw      %%ymm0,%%ymm1,%%ymm2          \n"  // 3*near+far (1)

      "vmovdqu     (%0,%3,2),%%xmm0              \n"  // 01234567 (16b)
      "vmovdqu     2(%0,%3,2),%%xmm1             \n"  // 12345678 (16b)
      "vpermq      $0b11011000,%%ymm0,%%ymm0     \n"  // 0123000045670000
      "vpermq      $0b11011000,%%ymm1,%%ymm1     \n"  // 1234000056780000
      "vpunpcklwd  %%ymm1,%%ymm0,%%ymm0          \n"  // 0112233445566778 (near)
      "vpshufb     %%ymm5,%%ymm0,%%ymm1          \n"  // 1021324354657687 (far)
      "vpaddw      %%ymm0,%%ymm1,%%ymm1          \n"  // near+far
      "vpaddw      %%ymm0,%%ymm0,%%ymm0          \n"  // 2*near
      "vpaddw      %%ymm0,%%ymm1,%%ymm3          \n"  // 3*near+far (2)

      "vpaddw      %%ymm2,%%ymm2,%%ymm0          \n"  // 6*near+2*far (1)
      "vpaddw      %%ymm4,%%ymm3,%%ymm1          \n"  // 3*near+far+8 (2)
      "vpaddw      %%ymm0,%%ymm2,%%ymm0          \n"  // 9*near+3*far (1)
      "vpaddw      %%ymm0,%%ymm1,%%ymm0          \n"  // 9 3 3 1 + 8 (1)
      "vpsrlw      $4,%%ymm0,%%ymm0              \n"  // ^ div by 16
      "vmovdqu     %%ymm0,(%1)                   \n"  // store above

      "vpaddw      %%ymm3,%%ymm3,%%ymm0          \n"  // 6*near+2*far (2)
      "vpaddw      %%ymm4,%%ymm2,%%ymm1          \n"  // 3*near+far+8 (1)
      "vpaddw      %%ymm0,%%ymm3,%%ymm0          \n"  // 9*near+3*far (2)
      "vpaddw      %%ymm0,%%ymm1,%%ymm0          \n"  // 9 3 3 1 + 8 (2)
      "vpsrlw      $4,%%ymm0,%%ymm0              \n"  // ^ div by 16
      "vmovdqu     %%ymm0,(%1,%4,2)              \n"  // store below

      "lea         0x10(%0),%0                   \n"
      "lea         0x20(%1),%1                   \n"  // 8 sample to 16 sample
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_ptr),                // %0
        "+r"(dst_ptr),                // %1
        "+r"(dst_width)               // %2
      : "r"((intptr_t)(src_stride)),  // %3
        "r"((intptr_t)(dst_stride)),  // %4
        "m"(kLinearShuffleFar)        // %5
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5");
}
#endif

#ifdef HAS_SCALEROWUP2_LINEAR_16_AVX2
void ScaleRowUp2_Linear_16_AVX2(const uint16_t* src_ptr,
                                uint16_t* dst_ptr,
                                int dst_width) {
  asm volatile(
      "vpcmpeqd    %%ymm4,%%ymm4,%%ymm4          \n"
      "vpsrld      $31,%%ymm4,%%ymm4             \n"
      "vpslld      $1,%%ymm4,%%ymm4              \n"  // all 2

      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%xmm0                   \n"  // 01234567 (16b, 1u1v)
      "vmovdqu     2(%0),%%xmm1                  \n"  // 12345678 (16b, 1u1v)

      "vpmovzxwd   %%xmm0,%%ymm0                 \n"  // 01234567 (32b, 1u1v)
      "vpmovzxwd   %%xmm1,%%ymm1                 \n"  // 12345678 (32b, 1u1v)

      "vpshufd     $0b10110001,%%ymm0,%%ymm2     \n"  // 10325476 (lo, far)
      "vpshufd     $0b10110001,%%ymm1,%%ymm3     \n"  // 21436587 (hi, far)

      "vpaddd      %%ymm4,%%ymm2,%%ymm2          \n"  // far+2 (lo)
      "vpaddd      %%ymm4,%%ymm3,%%ymm3          \n"  // far+2 (hi)
      "vpaddd      %%ymm0,%%ymm2,%%ymm2          \n"  // near+far+2 (lo)
      "vpaddd      %%ymm1,%%ymm3,%%ymm3          \n"  // near+far+2 (hi)
      "vpaddd      %%ymm0,%%ymm0,%%ymm0          \n"  // 2*near (lo)
      "vpaddd      %%ymm1,%%ymm1,%%ymm1          \n"  // 2*near (hi)
      "vpaddd      %%ymm0,%%ymm2,%%ymm0          \n"  // 3*near+far+2 (lo)
      "vpaddd      %%ymm1,%%ymm3,%%ymm1          \n"  // 3*near+far+2 (hi)

      "vpsrld      $2,%%ymm0,%%ymm0              \n"  // 3/4*near+1/4*far (lo)
      "vpsrld      $2,%%ymm1,%%ymm1              \n"  // 3/4*near+1/4*far (hi)
      "vpackusdw   %%ymm1,%%ymm0,%%ymm0          \n"
      "vpshufd     $0b11011000,%%ymm0,%%ymm0     \n"
      "vmovdqu     %%ymm0,(%1)                   \n"

      "lea         0x10(%0),%0                   \n"
      "lea         0x20(%1),%1                   \n"  // 8 pixel to 16 pixel
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_ptr),   // %0
        "+r"(dst_ptr),   // %1
        "+r"(dst_width)  // %2
      :
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4");
}
#endif

#ifdef HAS_SCALEROWUP2_BILINEAR_16_AVX2
void ScaleRowUp2_Bilinear_16_AVX2(const uint16_t* src_ptr,
                                  ptrdiff_t src_stride,
                                  uint16_t* dst_ptr,
                                  ptrdiff_t dst_stride,
                                  int dst_width) {
  asm volatile(
      "vpcmpeqd    %%ymm6,%%ymm6,%%ymm6          \n"
      "vpsrld      $31,%%ymm6,%%ymm6             \n"
      "vpslld      $3,%%ymm6,%%ymm6              \n"  // all 8

      LABELALIGN
      "1:          \n"

      "vmovdqu     (%0),%%xmm0                   \n"  // 01234567 (16b, 1u1v)
      "vmovdqu     2(%0),%%xmm1                  \n"  // 12345678 (16b, 1u1v)
      "vpmovzxwd   %%xmm0,%%ymm0                 \n"  // 01234567 (32b, 1u1v)
      "vpmovzxwd   %%xmm1,%%ymm1                 \n"  // 12345678 (32b, 1u1v)
      "vpshufd     $0b10110001,%%ymm0,%%ymm2     \n"  // 10325476 (lo, far)
      "vpshufd     $0b10110001,%%ymm1,%%ymm3     \n"  // 21436587 (hi, far)
      "vpaddd      %%ymm0,%%ymm2,%%ymm2          \n"  // near+far (lo)
      "vpaddd      %%ymm1,%%ymm3,%%ymm3          \n"  // near+far (hi)
      "vpaddd      %%ymm0,%%ymm0,%%ymm0          \n"  // 2*near (lo)
      "vpaddd      %%ymm1,%%ymm1,%%ymm1          \n"  // 2*near (hi)
      "vpaddd      %%ymm0,%%ymm2,%%ymm0          \n"  // 3*near+far (1, lo)
      "vpaddd      %%ymm1,%%ymm3,%%ymm1          \n"  // 3*near+far (1, hi)

      "vmovdqu     (%0,%3,2),%%xmm2              \n"  // 01234567 (16b, 1u1v)
      "vmovdqu     2(%0,%3,2),%%xmm3             \n"  // 12345678 (16b, 1u1v)
      "vpmovzxwd   %%xmm2,%%ymm2                 \n"  // 01234567 (32b, 1u1v)
      "vpmovzxwd   %%xmm3,%%ymm3                 \n"  // 12345678 (32b, 1u1v)
      "vpshufd     $0b10110001,%%ymm2,%%ymm4     \n"  // 10325476 (lo, far)
      "vpshufd     $0b10110001,%%ymm3,%%ymm5     \n"  // 21436587 (hi, far)
      "vpaddd      %%ymm2,%%ymm4,%%ymm4          \n"  // near+far (lo)
      "vpaddd      %%ymm3,%%ymm5,%%ymm5          \n"  // near+far (hi)
      "vpaddd      %%ymm2,%%ymm2,%%ymm2          \n"  // 2*near (lo)
      "vpaddd      %%ymm3,%%ymm3,%%ymm3          \n"  // 2*near (hi)
      "vpaddd      %%ymm2,%%ymm4,%%ymm2          \n"  // 3*near+far (2, lo)
      "vpaddd      %%ymm3,%%ymm5,%%ymm3          \n"  // 3*near+far (2, hi)

      "vpaddd      %%ymm0,%%ymm0,%%ymm4          \n"  // 6*near+2*far (1, lo)
      "vpaddd      %%ymm6,%%ymm2,%%ymm5          \n"  // 3*near+far+8 (2, lo)
      "vpaddd      %%ymm4,%%ymm0,%%ymm4          \n"  // 9*near+3*far (1, lo)
      "vpaddd      %%ymm4,%%ymm5,%%ymm4          \n"  // 9 3 3 1 + 8 (1, lo)
      "vpsrld      $4,%%ymm4,%%ymm4              \n"  // ^ div by 16 (1, lo)

      "vpaddd      %%ymm2,%%ymm2,%%ymm5          \n"  // 6*near+2*far (2, lo)
      "vpaddd      %%ymm6,%%ymm0,%%ymm0          \n"  // 3*near+far+8 (1, lo)
      "vpaddd      %%ymm5,%%ymm2,%%ymm5          \n"  // 9*near+3*far (2, lo)
      "vpaddd      %%ymm5,%%ymm0,%%ymm5          \n"  // 9 3 3 1 + 8 (2, lo)
      "vpsrld      $4,%%ymm5,%%ymm5              \n"  // ^ div by 16 (2, lo)

      "vpaddd      %%ymm1,%%ymm1,%%ymm0          \n"  // 6*near+2*far (1, hi)
      "vpaddd      %%ymm6,%%ymm3,%%ymm2          \n"  // 3*near+far+8 (2, hi)
      "vpaddd      %%ymm0,%%ymm1,%%ymm0          \n"  // 9*near+3*far (1, hi)
      "vpaddd      %%ymm0,%%ymm2,%%ymm0          \n"  // 9 3 3 1 + 8 (1, hi)
      "vpsrld      $4,%%ymm0,%%ymm0              \n"  // ^ div by 16 (1, hi)

      "vpaddd      %%ymm3,%%ymm3,%%ymm2          \n"  // 6*near+2*far (2, hi)
      "vpaddd      %%ymm6,%%ymm1,%%ymm1          \n"  // 3*near+far+8 (1, hi)
      "vpaddd      %%ymm2,%%ymm3,%%ymm2          \n"  // 9*near+3*far (2, hi)
      "vpaddd      %%ymm2,%%ymm1,%%ymm2          \n"  // 9 3 3 1 + 8 (2, hi)
      "vpsrld      $4,%%ymm2,%%ymm2              \n"  // ^ div by 16 (2, hi)

      "vpackusdw   %%ymm0,%%ymm4,%%ymm4          \n"
      "vpshufd     $0b11011000,%%ymm4,%%ymm4     \n"
      "vmovdqu     %%ymm4,(%1)                   \n"  // store above
      "vpackusdw   %%ymm2,%%ymm5,%%ymm5          \n"
      "vpshufd     $0b11011000,%%ymm5,%%ymm5     \n"
      "vmovdqu     %%ymm5,(%1,%4,2)              \n"  // store below

      "lea         0x10(%0),%0                   \n"
      "lea         0x20(%1),%1                   \n"  // 8 pixel to 16 pixel
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_ptr),                // %0
        "+r"(dst_ptr),                // %1
        "+r"(dst_width)               // %2
      : "r"((intptr_t)(src_stride)),  // %3
        "r"((intptr_t)(dst_stride))   // %4
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6");
}
#endif

// Reads 16xN bytes and produces 16 shorts at a time.
void ScaleAddRow_SSE2(const uint8_t* src_ptr,
                      uint16_t* dst_ptr,
                      int src_width) {
      asm volatile("pxor        %%xmm5,%%xmm5                 \n"

               // 16 pixel loop.
               LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm3                   \n"
      "lea         0x10(%0),%0                   \n"  // src_ptr += 16
      "movdqu      (%1),%%xmm0                   \n"
      "movdqu      0x10(%1),%%xmm1               \n"
      "movdqa      %%xmm3,%%xmm2                 \n"
      "punpcklbw   %%xmm5,%%xmm2                 \n"
      "punpckhbw   %%xmm5,%%xmm3                 \n"
      "paddusw     %%xmm2,%%xmm0                 \n"
      "paddusw     %%xmm3,%%xmm1                 \n"
      "movdqu      %%xmm0,(%1)                   \n"
      "movdqu      %%xmm1,0x10(%1)               \n"
      "lea         0x20(%1),%1                   \n"
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"
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
      asm volatile("vpxor       %%ymm5,%%ymm5,%%ymm5          \n"

               LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm3                   \n"
      "lea         0x20(%0),%0                   \n"  // src_ptr += 32
      "vpermq      $0xd8,%%ymm3,%%ymm3           \n"
      "vpunpcklbw  %%ymm5,%%ymm3,%%ymm2          \n"
      "vpunpckhbw  %%ymm5,%%ymm3,%%ymm3          \n"
      "vpaddusw    (%1),%%ymm2,%%ymm0            \n"
      "vpaddusw    0x20(%1),%%ymm3,%%ymm1        \n"
      "vmovdqu     %%ymm0,(%1)                   \n"
      "vmovdqu     %%ymm1,0x20(%1)               \n"
      "lea         0x40(%1),%1                   \n"
      "sub         $0x20,%2                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
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
      "movd        %6,%%xmm2                     \n"
      "movd        %7,%%xmm3                     \n"
      "movl        $0x04040000,%k2               \n"
      "movd        %k2,%%xmm5                    \n"
      "pcmpeqb     %%xmm6,%%xmm6                 \n"
      "psrlw       $0x9,%%xmm6                   \n"  // 0x007f007f
      "pcmpeqb     %%xmm7,%%xmm7                 \n"
      "psrlw       $15,%%xmm7                    \n"  // 0x00010001

      "pextrw      $0x1,%%xmm2,%k3               \n"
      "subl        $0x2,%5                       \n"
      "jl          29f                           \n"
      "movdqa      %%xmm2,%%xmm0                 \n"
      "paddd       %%xmm3,%%xmm0                 \n"
      "punpckldq   %%xmm0,%%xmm2                 \n"
      "punpckldq   %%xmm3,%%xmm3                 \n"
      "paddd       %%xmm3,%%xmm3                 \n"
      "pextrw      $0x3,%%xmm2,%k4               \n"

      LABELALIGN
      "2:          \n"
      "movdqa      %%xmm2,%%xmm1                 \n"
      "paddd       %%xmm3,%%xmm2                 \n"
      "movzwl      0x00(%1,%3,1),%k2             \n"
      "movd        %k2,%%xmm0                    \n"
      "psrlw       $0x9,%%xmm1                   \n"
      "movzwl      0x00(%1,%4,1),%k2             \n"
      "movd        %k2,%%xmm4                    \n"
      "pshufb      %%xmm5,%%xmm1                 \n"
      "punpcklwd   %%xmm4,%%xmm0                 \n"
      "psubb       %8,%%xmm0                     \n"  // make pixels signed.
      "pxor        %%xmm6,%%xmm1                 \n"  // 128 - f = (f ^ 127 ) +
                                                      // 1
      "paddusb     %%xmm7,%%xmm1                 \n"
      "pmaddubsw   %%xmm0,%%xmm1                 \n"
      "pextrw      $0x1,%%xmm2,%k3               \n"
      "pextrw      $0x3,%%xmm2,%k4               \n"
      "paddw       %9,%%xmm1                     \n"  // make pixels unsigned.
      "psrlw       $0x7,%%xmm1                   \n"
      "packuswb    %%xmm1,%%xmm1                 \n"
      "movd        %%xmm1,%k2                    \n"
      "mov         %w2,(%0)                      \n"
      "lea         0x2(%0),%0                    \n"
      "subl        $0x2,%5                       \n"
      "jge         2b                            \n"

      LABELALIGN
      "29:         \n"
      "addl        $0x1,%5                       \n"
      "jl          99f                           \n"
      "movzwl      0x00(%1,%3,1),%k2             \n"
      "movd        %k2,%%xmm0                    \n"
      "psrlw       $0x9,%%xmm2                   \n"
      "pshufb      %%xmm5,%%xmm2                 \n"
      "psubb       %8,%%xmm0                     \n"  // make pixels signed.
      "pxor        %%xmm6,%%xmm2                 \n"
      "paddusb     %%xmm7,%%xmm2                 \n"
      "pmaddubsw   %%xmm0,%%xmm2                 \n"
      "paddw       %9,%%xmm2                     \n"  // make pixels unsigned.
      "psrlw       $0x7,%%xmm2                   \n"
      "packuswb    %%xmm2,%%xmm2                 \n"
      "movd        %%xmm2,%k2                    \n"
      "mov         %b2,(%0)                      \n"
      "99:         \n"
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
      "1:          \n"
      "movdqu      (%1),%%xmm0                   \n"
      "lea         0x10(%1),%1                   \n"
      "movdqa      %%xmm0,%%xmm1                 \n"
      "punpcklbw   %%xmm0,%%xmm0                 \n"
      "punpckhbw   %%xmm1,%%xmm1                 \n"
      "movdqu      %%xmm0,(%0)                   \n"
      "movdqu      %%xmm1,0x10(%0)               \n"
      "lea         0x20(%0),%0                   \n"
      "sub         $0x20,%2                      \n"
      "jg          1b                            \n"

      : "+r"(dst_ptr),   // %0
        "+r"(src_ptr),   // %1
        "+r"(dst_width)  // %2
      :
      : "memory", "cc", "xmm0", "xmm1");
}

void ScaleARGBRowDown2_SSE2(const uint8_t* src_argb,
                            ptrdiff_t src_stride,
                            uint8_t* dst_argb,
                            int dst_width) {
  (void)src_stride;
  asm volatile(
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "lea         0x20(%0),%0                   \n"
      "shufps      $0xdd,%%xmm1,%%xmm0           \n"
      "movdqu      %%xmm0,(%1)                   \n"
      "lea         0x10(%1),%1                   \n"
      "sub         $0x4,%2                       \n"
      "jg          1b                            \n"
      : "+r"(src_argb),  // %0
        "+r"(dst_argb),  // %1
        "+r"(dst_width)  // %2
      :
      : "memory", "cc", "xmm0", "xmm1");
}

void ScaleARGBRowDown2Linear_SSE2(const uint8_t* src_argb,
                                  ptrdiff_t src_stride,
                                  uint8_t* dst_argb,
                                  int dst_width) {
  (void)src_stride;
  asm volatile(
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "lea         0x20(%0),%0                   \n"
      "movdqa      %%xmm0,%%xmm2                 \n"
      "shufps      $0x88,%%xmm1,%%xmm0           \n"
      "shufps      $0xdd,%%xmm1,%%xmm2           \n"
      "pavgb       %%xmm2,%%xmm0                 \n"
      "movdqu      %%xmm0,(%1)                   \n"
      "lea         0x10(%1),%1                   \n"
      "sub         $0x4,%2                       \n"
      "jg          1b                            \n"
      : "+r"(src_argb),  // %0
        "+r"(dst_argb),  // %1
        "+r"(dst_width)  // %2
      :
      : "memory", "cc", "xmm0", "xmm1");
}

void ScaleARGBRowDown2Box_SSE2(const uint8_t* src_argb,
                               ptrdiff_t src_stride,
                               uint8_t* dst_argb,
                               int dst_width) {
  asm volatile(
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "movdqu      0x00(%0,%3,1),%%xmm2          \n"
      "movdqu      0x10(%0,%3,1),%%xmm3          \n"
      "lea         0x20(%0),%0                   \n"
      "pavgb       %%xmm2,%%xmm0                 \n"
      "pavgb       %%xmm3,%%xmm1                 \n"
      "movdqa      %%xmm0,%%xmm2                 \n"
      "shufps      $0x88,%%xmm1,%%xmm0           \n"
      "shufps      $0xdd,%%xmm1,%%xmm2           \n"
      "pavgb       %%xmm2,%%xmm0                 \n"
      "movdqu      %%xmm0,(%1)                   \n"
      "lea         0x10(%1),%1                   \n"
      "sub         $0x4,%2                       \n"
      "jg          1b                            \n"
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
      "lea         0x00(,%1,4),%1                \n"
      "lea         0x00(%1,%1,2),%4              \n"

      LABELALIGN
      "1:          \n"
      "movd        (%0),%%xmm0                   \n"
      "movd        0x00(%0,%1,1),%%xmm1          \n"
      "punpckldq   %%xmm1,%%xmm0                 \n"
      "movd        0x00(%0,%1,2),%%xmm2          \n"
      "movd        0x00(%0,%4,1),%%xmm3          \n"
      "lea         0x00(%0,%1,4),%0              \n"
      "punpckldq   %%xmm3,%%xmm2                 \n"
      "punpcklqdq  %%xmm2,%%xmm0                 \n"
      "movdqu      %%xmm0,(%2)                   \n"
      "lea         0x10(%2),%2                   \n"
      "sub         $0x4,%3                       \n"
      "jg          1b                            \n"
      : "+r"(src_argb),       // %0
        "+r"(src_stepx_x4),   // %1
        "+r"(dst_argb),       // %2
        "+r"(dst_width),      // %3
        "=&r"(src_stepx_x12)  // %4
      :
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3");
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
      "lea         0x00(,%1,4),%1                \n"
      "lea         0x00(%1,%1,2),%4              \n"
      "lea         0x00(%0,%5,1),%5              \n"

      LABELALIGN
      "1:          \n"
      "movq        (%0),%%xmm0                   \n"
      "movhps      0x00(%0,%1,1),%%xmm0          \n"
      "movq        0x00(%0,%1,2),%%xmm1          \n"
      "movhps      0x00(%0,%4,1),%%xmm1          \n"
      "lea         0x00(%0,%1,4),%0              \n"
      "movq        (%5),%%xmm2                   \n"
      "movhps      0x00(%5,%1,1),%%xmm2          \n"
      "movq        0x00(%5,%1,2),%%xmm3          \n"
      "movhps      0x00(%5,%4,1),%%xmm3          \n"
      "lea         0x00(%5,%1,4),%5              \n"
      "pavgb       %%xmm2,%%xmm0                 \n"
      "pavgb       %%xmm3,%%xmm1                 \n"
      "movdqa      %%xmm0,%%xmm2                 \n"
      "shufps      $0x88,%%xmm1,%%xmm0           \n"
      "shufps      $0xdd,%%xmm1,%%xmm2           \n"
      "pavgb       %%xmm2,%%xmm0                 \n"
      "movdqu      %%xmm0,(%2)                   \n"
      "lea         0x10(%2),%2                   \n"
      "sub         $0x4,%3                       \n"
      "jg          1b                            \n"
      : "+r"(src_argb),        // %0
        "+r"(src_stepx_x4),    // %1
        "+r"(dst_argb),        // %2
        "+rm"(dst_width),      // %3
        "=&r"(src_stepx_x12),  // %4
        "+r"(row1)             // %5
      :
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3");
}

void ScaleARGBCols_SSE2(uint8_t* dst_argb,
                        const uint8_t* src_argb,
                        int dst_width,
                        int x,
                        int dx) {
  intptr_t x0, x1;
  asm volatile(
      "movd        %5,%%xmm2                     \n"
      "movd        %6,%%xmm3                     \n"
      "pshufd      $0x0,%%xmm2,%%xmm2            \n"
      "pshufd      $0x11,%%xmm3,%%xmm0           \n"
      "paddd       %%xmm0,%%xmm2                 \n"
      "paddd       %%xmm3,%%xmm3                 \n"
      "pshufd      $0x5,%%xmm3,%%xmm0            \n"
      "paddd       %%xmm0,%%xmm2                 \n"
      "paddd       %%xmm3,%%xmm3                 \n"
      "pshufd      $0x0,%%xmm3,%%xmm3            \n"
      "pextrw      $0x1,%%xmm2,%k0               \n"
      "pextrw      $0x3,%%xmm2,%k1               \n"
      "cmp         $0x0,%4                       \n"
      "jl          99f                           \n"
      "sub         $0x4,%4                       \n"
      "jl          49f                           \n"

      LABELALIGN
      "40:         \n"
      "movd        0x00(%3,%0,4),%%xmm0          \n"
      "movd        0x00(%3,%1,4),%%xmm1          \n"
      "pextrw      $0x5,%%xmm2,%k0               \n"
      "pextrw      $0x7,%%xmm2,%k1               \n"
      "paddd       %%xmm3,%%xmm2                 \n"
      "punpckldq   %%xmm1,%%xmm0                 \n"
      "movd        0x00(%3,%0,4),%%xmm1          \n"
      "movd        0x00(%3,%1,4),%%xmm4          \n"
      "pextrw      $0x1,%%xmm2,%k0               \n"
      "pextrw      $0x3,%%xmm2,%k1               \n"
      "punpckldq   %%xmm4,%%xmm1                 \n"
      "punpcklqdq  %%xmm1,%%xmm0                 \n"
      "movdqu      %%xmm0,(%2)                   \n"
      "lea         0x10(%2),%2                   \n"
      "sub         $0x4,%4                       \n"
      "jge         40b                           \n"

      "49:         \n"
      "test        $0x2,%4                       \n"
      "je          29f                           \n"
      "movd        0x00(%3,%0,4),%%xmm0          \n"
      "movd        0x00(%3,%1,4),%%xmm1          \n"
      "pextrw      $0x5,%%xmm2,%k0               \n"
      "punpckldq   %%xmm1,%%xmm0                 \n"
      "movq        %%xmm0,(%2)                   \n"
      "lea         0x8(%2),%2                    \n"
      "29:         \n"
      "test        $0x1,%4                       \n"
      "je          99f                           \n"
      "movd        0x00(%3,%0,4),%%xmm0          \n"
      "movd        %%xmm0,(%2)                   \n"
      "99:         \n"
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
      "1:          \n"
      "movdqu      (%1),%%xmm0                   \n"
      "lea         0x10(%1),%1                   \n"
      "movdqa      %%xmm0,%%xmm1                 \n"
      "punpckldq   %%xmm0,%%xmm0                 \n"
      "punpckhdq   %%xmm1,%%xmm1                 \n"
      "movdqu      %%xmm0,(%0)                   \n"
      "movdqu      %%xmm1,0x10(%0)               \n"
      "lea         0x20(%0),%0                   \n"
      "sub         $0x8,%2                       \n"
      "jg          1b                            \n"

      : "+r"(dst_argb),  // %0
        "+r"(src_argb),  // %1
        "+r"(dst_width)  // %2
      :
      : "memory", "cc", "xmm0", "xmm1");
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
      "movdqa      %0,%%xmm4                     \n"
      "movdqa      %1,%%xmm5                     \n"
      :
      : "m"(kShuffleColARGB),   // %0
        "m"(kShuffleFractions)  // %1
  );

  asm volatile(
      "movd        %5,%%xmm2                     \n"
      "movd        %6,%%xmm3                     \n"
      "pcmpeqb     %%xmm6,%%xmm6                 \n"
      "psrlw       $0x9,%%xmm6                   \n"
      "pextrw      $0x1,%%xmm2,%k3               \n"
      "sub         $0x2,%2                       \n"
      "jl          29f                           \n"
      "movdqa      %%xmm2,%%xmm0                 \n"
      "paddd       %%xmm3,%%xmm0                 \n"
      "punpckldq   %%xmm0,%%xmm2                 \n"
      "punpckldq   %%xmm3,%%xmm3                 \n"
      "paddd       %%xmm3,%%xmm3                 \n"
      "pextrw      $0x3,%%xmm2,%k4               \n"

      LABELALIGN
      "2:          \n"
      "movdqa      %%xmm2,%%xmm1                 \n"
      "paddd       %%xmm3,%%xmm2                 \n"
      "movq        0x00(%1,%3,4),%%xmm0          \n"
      "psrlw       $0x9,%%xmm1                   \n"
      "movhps      0x00(%1,%4,4),%%xmm0          \n"
      "pshufb      %%xmm5,%%xmm1                 \n"
      "pshufb      %%xmm4,%%xmm0                 \n"
      "pxor        %%xmm6,%%xmm1                 \n"
      "pmaddubsw   %%xmm1,%%xmm0                 \n"
      "psrlw       $0x7,%%xmm0                   \n"
      "pextrw      $0x1,%%xmm2,%k3               \n"
      "pextrw      $0x3,%%xmm2,%k4               \n"
      "packuswb    %%xmm0,%%xmm0                 \n"
      "movq        %%xmm0,(%0)                   \n"
      "lea         0x8(%0),%0                    \n"
      "sub         $0x2,%2                       \n"
      "jge         2b                            \n"

      LABELALIGN
      "29:         \n"
      "add         $0x1,%2                       \n"
      "jl          99f                           \n"
      "psrlw       $0x9,%%xmm2                   \n"
      "movq        0x00(%1,%3,4),%%xmm0          \n"
      "pshufb      %%xmm5,%%xmm2                 \n"
      "pshufb      %%xmm4,%%xmm0                 \n"
      "pxor        %%xmm6,%%xmm2                 \n"
      "pmaddubsw   %%xmm2,%%xmm0                 \n"
      "psrlw       $0x7,%%xmm0                   \n"
      "packuswb    %%xmm0,%%xmm0                 \n"
      "movd        %%xmm0,(%0)                   \n"

      LABELALIGN "99:         \n"

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
      "cdq         \n"
      "shld        $0x10,%%eax,%%edx             \n"
      "shl         $0x10,%%eax                   \n"
      "idiv        %1                            \n"
      "mov         %0, %%eax                     \n"
      : "+a"(num)  // %0
      : "c"(div)   // %1
      : "memory", "cc", "edx");
  return num;
}

// Divide num - 1 by div - 1 and return as 16.16 fixed point result.
int FixedDiv1_X86(int num, int div) {
  asm volatile(
      "cdq         \n"
      "shld        $0x10,%%eax,%%edx             \n"
      "shl         $0x10,%%eax                   \n"
      "sub         $0x10001,%%eax                \n"
      "sbb         $0x0,%%edx                    \n"
      "sub         $0x1,%1                       \n"
      "idiv        %1                            \n"
      "mov         %0, %%eax                     \n"
      : "+a"(num)  // %0
      : "c"(div)   // %1
      : "memory", "cc", "edx");
  return num;
}

#if defined(HAS_SCALEUVROWDOWN2BOX_SSSE3) || \
    defined(HAS_SCALEUVROWDOWN2BOX_AVX2)

// Shuffle table for splitting UV into upper and lower part of register.
static const uvec8 kShuffleSplitUV = {0u, 2u, 4u, 6u, 8u, 10u, 12u, 14u,
                                      1u, 3u, 5u, 7u, 9u, 11u, 13u, 15u};
static const uvec8 kShuffleMergeUV = {0u,   8u,   2u,   10u,  4u,   12u,
                                      6u,   14u,  0x80, 0x80, 0x80, 0x80,
                                      0x80, 0x80, 0x80, 0x80};
#endif

#ifdef HAS_SCALEUVROWDOWN2BOX_SSSE3

void ScaleUVRowDown2Box_SSSE3(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_ptr,
                              int dst_width) {
  asm volatile(
      "pcmpeqb     %%xmm4,%%xmm4                 \n"  // 01010101
      "psrlw       $0xf,%%xmm4                   \n"
      "packuswb    %%xmm4,%%xmm4                 \n"
      "pxor        %%xmm5, %%xmm5                \n"  // zero
      "movdqa      %4,%%xmm1                     \n"  // split shuffler
      "movdqa      %5,%%xmm3                     \n"  // merge shuffler

      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"  // 8 UV row 0
      "movdqu      0x00(%0,%3,1),%%xmm2          \n"  // 8 UV row 1
      "lea         0x10(%0),%0                   \n"
      "pshufb      %%xmm1,%%xmm0                 \n"  // uuuuvvvv
      "pshufb      %%xmm1,%%xmm2                 \n"
      "pmaddubsw   %%xmm4,%%xmm0                 \n"  // horizontal add
      "pmaddubsw   %%xmm4,%%xmm2                 \n"
      "paddw       %%xmm2,%%xmm0                 \n"  // vertical add
      "psrlw       $0x1,%%xmm0                   \n"  // round
      "pavgw       %%xmm5,%%xmm0                 \n"
      "pshufb      %%xmm3,%%xmm0                 \n"  // merge uv
      "movq        %%xmm0,(%1)                   \n"
      "lea         0x8(%1),%1                    \n"  // 4 UV
      "sub         $0x4,%2                       \n"
      "jg          1b                            \n"
      : "+r"(src_ptr),                // %0
        "+r"(dst_ptr),                // %1
        "+r"(dst_width)               // %2
      : "r"((intptr_t)(src_stride)),  // %3
        "m"(kShuffleSplitUV),         // %4
        "m"(kShuffleMergeUV)          // %5
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5");
}
#endif  // HAS_SCALEUVROWDOWN2BOX_SSSE3

#ifdef HAS_SCALEUVROWDOWN2BOX_AVX2
void ScaleUVRowDown2Box_AVX2(const uint8_t* src_ptr,
                             ptrdiff_t src_stride,
                             uint8_t* dst_ptr,
                             int dst_width) {
  asm volatile(
      "vpcmpeqb    %%ymm4,%%ymm4,%%ymm4          \n"  // 01010101
      "vpabsb      %%ymm4,%%ymm4                 \n"
      "vpxor       %%ymm5,%%ymm5,%%ymm5          \n"  // zero
      "vbroadcastf128 %4,%%ymm1                  \n"  // split shuffler
      "vbroadcastf128 %5,%%ymm3                  \n"  // merge shuffler

      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"  // 16 UV row 0
      "vmovdqu     0x00(%0,%3,1),%%ymm2          \n"  // 16 UV row 1
      "lea         0x20(%0),%0                   \n"
      "vpshufb     %%ymm1,%%ymm0,%%ymm0          \n"  // uuuuvvvv
      "vpshufb     %%ymm1,%%ymm2,%%ymm2          \n"
      "vpmaddubsw  %%ymm4,%%ymm0,%%ymm0          \n"  // horizontal add
      "vpmaddubsw  %%ymm4,%%ymm2,%%ymm2          \n"
      "vpaddw      %%ymm2,%%ymm0,%%ymm0          \n"  // vertical add
      "vpsrlw      $0x1,%%ymm0,%%ymm0            \n"  // round
      "vpavgw      %%ymm5,%%ymm0,%%ymm0          \n"
      "vpshufb     %%ymm3,%%ymm0,%%ymm0          \n"  // merge uv
      "vpermq      $0xd8,%%ymm0,%%ymm0           \n"  // combine qwords
      "vmovdqu     %%xmm0,(%1)                   \n"
      "lea         0x10(%1),%1                   \n"  // 8 UV
      "sub         $0x8,%2                       \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_ptr),                // %0
        "+r"(dst_ptr),                // %1
        "+r"(dst_width)               // %2
      : "r"((intptr_t)(src_stride)),  // %3
        "m"(kShuffleSplitUV),         // %4
        "m"(kShuffleMergeUV)          // %5
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5");
}
#endif  // HAS_SCALEUVROWDOWN2BOX_AVX2

static const uvec8 kUVLinearMadd31 = {3, 1, 3, 1, 1, 3, 1, 3,
                                      3, 1, 3, 1, 1, 3, 1, 3};

#ifdef HAS_SCALEUVROWUP2_LINEAR_SSSE3
void ScaleUVRowUp2_Linear_SSSE3(const uint8_t* src_ptr,
                                uint8_t* dst_ptr,
                                int dst_width) {
  asm volatile(
      "pcmpeqw     %%xmm4,%%xmm4                 \n"
      "psrlw       $15,%%xmm4                    \n"
      "psllw       $1,%%xmm4                     \n"  // all 2
      "movdqa      %3,%%xmm3                     \n"

      LABELALIGN
      "1:          \n"
      "movq        (%0),%%xmm0                   \n"  // 00112233 (1u1v)
      "movq        2(%0),%%xmm1                  \n"  // 11223344 (1u1v)
      "punpcklbw   %%xmm1,%%xmm0                 \n"  // 0101121223233434 (2u2v)
      "movdqa      %%xmm0,%%xmm2                 \n"
      "punpckhdq   %%xmm0,%%xmm2                 \n"  // 2323232334343434 (2u2v)
      "punpckldq   %%xmm0,%%xmm0                 \n"  // 0101010112121212 (2u2v)
      "pmaddubsw   %%xmm3,%%xmm2                 \n"  // 3*near+far (1u1v16, hi)
      "pmaddubsw   %%xmm3,%%xmm0                 \n"  // 3*near+far (1u1v16, lo)
      "paddw       %%xmm4,%%xmm0                 \n"  // 3*near+far+2 (lo)
      "paddw       %%xmm4,%%xmm2                 \n"  // 3*near+far+2 (hi)
      "psrlw       $2,%%xmm0                     \n"  // 3/4*near+1/4*far (lo)
      "psrlw       $2,%%xmm2                     \n"  // 3/4*near+1/4*far (hi)
      "packuswb    %%xmm2,%%xmm0                 \n"
      "movdqu      %%xmm0,(%1)                   \n"

      "lea         0x8(%0),%0                    \n"
      "lea         0x10(%1),%1                   \n"  // 4 uv to 8 uv
      "sub         $0x8,%2                       \n"
      "jg          1b                            \n"
      : "+r"(src_ptr),        // %0
        "+r"(dst_ptr),        // %1
        "+r"(dst_width)       // %2
      : "m"(kUVLinearMadd31)  // %3
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6");
}
#endif

#ifdef HAS_SCALEUVROWUP2_BILINEAR_SSSE3
void ScaleUVRowUp2_Bilinear_SSSE3(const uint8_t* src_ptr,
                                  ptrdiff_t src_stride,
                                  uint8_t* dst_ptr,
                                  ptrdiff_t dst_stride,
                                  int dst_width) {
  asm volatile(
      "pcmpeqw     %%xmm6,%%xmm6                 \n"
      "psrlw       $15,%%xmm6                    \n"
      "psllw       $3,%%xmm6                     \n"  // all 8
      "movdqa      %5,%%xmm7                     \n"

      LABELALIGN
      "1:          \n"
      "movq        (%0),%%xmm0                   \n"  // 00112233 (1u1v)
      "movq        2(%0),%%xmm1                  \n"  // 11223344 (1u1v)
      "punpcklbw   %%xmm1,%%xmm0                 \n"  // 0101121223233434 (2u2v)
      "movdqa      %%xmm0,%%xmm2                 \n"
      "punpckhdq   %%xmm0,%%xmm2                 \n"  // 2323232334343434 (2u2v)
      "punpckldq   %%xmm0,%%xmm0                 \n"  // 0101010112121212 (2u2v)
      "pmaddubsw   %%xmm7,%%xmm2                 \n"  // 3*near+far (1u1v16, hi)
      "pmaddubsw   %%xmm7,%%xmm0                 \n"  // 3*near+far (1u1v16, lo)

      "movq        (%0,%3),%%xmm1                \n"
      "movq        2(%0,%3),%%xmm4               \n"
      "punpcklbw   %%xmm4,%%xmm1                 \n"
      "movdqa      %%xmm1,%%xmm3                 \n"
      "punpckhdq   %%xmm1,%%xmm3                 \n"
      "punpckldq   %%xmm1,%%xmm1                 \n"
      "pmaddubsw   %%xmm7,%%xmm3                 \n"  // 3*near+far (2, hi)
      "pmaddubsw   %%xmm7,%%xmm1                 \n"  // 3*near+far (2, lo)

      // xmm0 xmm2
      // xmm1 xmm3

      "movdqa      %%xmm0,%%xmm4                 \n"
      "movdqa      %%xmm1,%%xmm5                 \n"
      "paddw       %%xmm0,%%xmm4                 \n"  // 6*near+2*far (1, lo)
      "paddw       %%xmm6,%%xmm5                 \n"  // 3*near+far+8 (2, lo)
      "paddw       %%xmm0,%%xmm4                 \n"  // 9*near+3*far (1, lo)
      "paddw       %%xmm5,%%xmm4                 \n"  // 9 3 3 1 + 8 (1, lo)
      "psrlw       $4,%%xmm4                     \n"  // ^ div by 16 (1, lo)

      "movdqa      %%xmm1,%%xmm5                 \n"
      "paddw       %%xmm1,%%xmm5                 \n"  // 6*near+2*far (2, lo)
      "paddw       %%xmm6,%%xmm0                 \n"  // 3*near+far+8 (1, lo)
      "paddw       %%xmm1,%%xmm5                 \n"  // 9*near+3*far (2, lo)
      "paddw       %%xmm0,%%xmm5                 \n"  // 9 3 3 1 + 8 (2, lo)
      "psrlw       $4,%%xmm5                     \n"  // ^ div by 16 (2, lo)

      "movdqa      %%xmm2,%%xmm0                 \n"
      "movdqa      %%xmm3,%%xmm1                 \n"
      "paddw       %%xmm2,%%xmm0                 \n"  // 6*near+2*far (1, hi)
      "paddw       %%xmm6,%%xmm1                 \n"  // 3*near+far+8 (2, hi)
      "paddw       %%xmm2,%%xmm0                 \n"  // 9*near+3*far (1, hi)
      "paddw       %%xmm1,%%xmm0                 \n"  // 9 3 3 1 + 8 (1, hi)
      "psrlw       $4,%%xmm0                     \n"  // ^ div by 16 (1, hi)

      "movdqa      %%xmm3,%%xmm1                 \n"
      "paddw       %%xmm3,%%xmm1                 \n"  // 6*near+2*far (2, hi)
      "paddw       %%xmm6,%%xmm2                 \n"  // 3*near+far+8 (1, hi)
      "paddw       %%xmm3,%%xmm1                 \n"  // 9*near+3*far (2, hi)
      "paddw       %%xmm2,%%xmm1                 \n"  // 9 3 3 1 + 8 (2, hi)
      "psrlw       $4,%%xmm1                     \n"  // ^ div by 16 (2, hi)

      "packuswb    %%xmm0,%%xmm4                 \n"
      "movdqu      %%xmm4,(%1)                   \n"  // store above
      "packuswb    %%xmm1,%%xmm5                 \n"
      "movdqu      %%xmm5,(%1,%4)                \n"  // store below

      "lea         0x8(%0),%0                    \n"
      "lea         0x10(%1),%1                   \n"  // 4 uv to 8 uv
      "sub         $0x8,%2                       \n"
      "jg          1b                            \n"
      : "+r"(src_ptr),                // %0
        "+r"(dst_ptr),                // %1
        "+r"(dst_width)               // %2
      : "r"((intptr_t)(src_stride)),  // %3
        "r"((intptr_t)(dst_stride)),  // %4
        "m"(kUVLinearMadd31)          // %5
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}
#endif

#ifdef HAS_SCALEUVROWUP2_LINEAR_AVX2

void ScaleUVRowUp2_Linear_AVX2(const uint8_t* src_ptr,
                               uint8_t* dst_ptr,
                               int dst_width) {
  asm volatile(
      "vpcmpeqw    %%ymm4,%%ymm4,%%ymm4          \n"
      "vpsrlw      $15,%%ymm4,%%ymm4             \n"
      "vpsllw      $1,%%ymm4,%%ymm4              \n"  // all 2
      "vbroadcastf128 %3,%%ymm3                  \n"

      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%xmm0                   \n"
      "vmovdqu     2(%0),%%xmm1                  \n"
      "vpermq      $0b11011000,%%ymm0,%%ymm0     \n"
      "vpermq      $0b11011000,%%ymm1,%%ymm1     \n"
      "vpunpcklbw  %%ymm1,%%ymm0,%%ymm0          \n"
      "vpunpckhdq  %%ymm0,%%ymm0,%%ymm2          \n"
      "vpunpckldq  %%ymm0,%%ymm0,%%ymm0          \n"
      "vpmaddubsw  %%ymm3,%%ymm2,%%ymm1          \n"  // 3*near+far (hi)
      "vpmaddubsw  %%ymm3,%%ymm0,%%ymm0          \n"  // 3*near+far (lo)
      "vpaddw      %%ymm4,%%ymm0,%%ymm0          \n"  // 3*near+far+2 (lo)
      "vpaddw      %%ymm4,%%ymm1,%%ymm1          \n"  // 3*near+far+2 (hi)
      "vpsrlw      $2,%%ymm0,%%ymm0              \n"  // 3/4*near+1/4*far (lo)
      "vpsrlw      $2,%%ymm1,%%ymm1              \n"  // 3/4*near+1/4*far (hi)
      "vpackuswb   %%ymm1,%%ymm0,%%ymm0          \n"
      "vmovdqu     %%ymm0,(%1)                   \n"

      "lea         0x10(%0),%0                   \n"
      "lea         0x20(%1),%1                   \n"  // 8 uv to 16 uv
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_ptr),        // %0
        "+r"(dst_ptr),        // %1
        "+r"(dst_width)       // %2
      : "m"(kUVLinearMadd31)  // %3
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4");
}
#endif

#ifdef HAS_SCALEUVROWUP2_BILINEAR_AVX2
void ScaleUVRowUp2_Bilinear_AVX2(const uint8_t* src_ptr,
                                 ptrdiff_t src_stride,
                                 uint8_t* dst_ptr,
                                 ptrdiff_t dst_stride,
                                 int dst_width) {
  asm volatile(
      "vpcmpeqw    %%ymm6,%%ymm6,%%ymm6          \n"
      "vpsrlw      $15,%%ymm6,%%ymm6             \n"
      "vpsllw      $3,%%ymm6,%%ymm6              \n"  // all 8
      "vbroadcastf128 %5,%%ymm7                  \n"

      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%xmm0                   \n"
      "vmovdqu     2(%0),%%xmm1                  \n"
      "vpermq      $0b11011000,%%ymm0,%%ymm0     \n"
      "vpermq      $0b11011000,%%ymm1,%%ymm1     \n"
      "vpunpcklbw  %%ymm1,%%ymm0,%%ymm0          \n"
      "vpunpckhdq  %%ymm0,%%ymm0,%%ymm2          \n"
      "vpunpckldq  %%ymm0,%%ymm0,%%ymm0          \n"
      "vpmaddubsw  %%ymm7,%%ymm2,%%ymm1          \n"  // 3*near+far (1, hi)
      "vpmaddubsw  %%ymm7,%%ymm0,%%ymm0          \n"  // 3*near+far (1, lo)

      "vmovdqu     (%0,%3),%%xmm2                \n"  // 0123456789ABCDEF
      "vmovdqu     2(%0,%3),%%xmm3               \n"  // 123456789ABCDEF0
      "vpermq      $0b11011000,%%ymm2,%%ymm2     \n"
      "vpermq      $0b11011000,%%ymm3,%%ymm3     \n"
      "vpunpcklbw  %%ymm3,%%ymm2,%%ymm2          \n"
      "vpunpckhdq  %%ymm2,%%ymm2,%%ymm4          \n"
      "vpunpckldq  %%ymm2,%%ymm2,%%ymm2          \n"
      "vpmaddubsw  %%ymm7,%%ymm4,%%ymm3          \n"  // 3*near+far (2, hi)
      "vpmaddubsw  %%ymm7,%%ymm2,%%ymm2          \n"  // 3*near+far (2, lo)

      // ymm0 ymm1
      // ymm2 ymm3

      "vpaddw      %%ymm0,%%ymm0,%%ymm4          \n"  // 6*near+2*far (1, lo)
      "vpaddw      %%ymm6,%%ymm2,%%ymm5          \n"  // 3*near+far+8 (2, lo)
      "vpaddw      %%ymm4,%%ymm0,%%ymm4          \n"  // 9*near+3*far (1, lo)
      "vpaddw      %%ymm4,%%ymm5,%%ymm4          \n"  // 9 3 3 1 + 8 (1, lo)
      "vpsrlw      $4,%%ymm4,%%ymm4              \n"  // ^ div by 16 (1, lo)

      "vpaddw      %%ymm2,%%ymm2,%%ymm5          \n"  // 6*near+2*far (2, lo)
      "vpaddw      %%ymm6,%%ymm0,%%ymm0          \n"  // 3*near+far+8 (1, lo)
      "vpaddw      %%ymm5,%%ymm2,%%ymm5          \n"  // 9*near+3*far (2, lo)
      "vpaddw      %%ymm5,%%ymm0,%%ymm5          \n"  // 9 3 3 1 + 8 (2, lo)
      "vpsrlw      $4,%%ymm5,%%ymm5              \n"  // ^ div by 16 (2, lo)

      "vpaddw      %%ymm1,%%ymm1,%%ymm0          \n"  // 6*near+2*far (1, hi)
      "vpaddw      %%ymm6,%%ymm3,%%ymm2          \n"  // 3*near+far+8 (2, hi)
      "vpaddw      %%ymm0,%%ymm1,%%ymm0          \n"  // 9*near+3*far (1, hi)
      "vpaddw      %%ymm0,%%ymm2,%%ymm0          \n"  // 9 3 3 1 + 8 (1, hi)
      "vpsrlw      $4,%%ymm0,%%ymm0              \n"  // ^ div by 16 (1, hi)

      "vpaddw      %%ymm3,%%ymm3,%%ymm2          \n"  // 6*near+2*far (2, hi)
      "vpaddw      %%ymm6,%%ymm1,%%ymm1          \n"  // 3*near+far+8 (1, hi)
      "vpaddw      %%ymm2,%%ymm3,%%ymm2          \n"  // 9*near+3*far (2, hi)
      "vpaddw      %%ymm2,%%ymm1,%%ymm2          \n"  // 9 3 3 1 + 8 (2, hi)
      "vpsrlw      $4,%%ymm2,%%ymm2              \n"  // ^ div by 16 (2, hi)

      "vpackuswb   %%ymm0,%%ymm4,%%ymm4          \n"
      "vmovdqu     %%ymm4,(%1)                   \n"  // store above
      "vpackuswb   %%ymm2,%%ymm5,%%ymm5          \n"
      "vmovdqu     %%ymm5,(%1,%4)                \n"  // store below

      "lea         0x10(%0),%0                   \n"
      "lea         0x20(%1),%1                   \n"  // 8 uv to 16 uv
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_ptr),                // %0
        "+r"(dst_ptr),                // %1
        "+r"(dst_width)               // %2
      : "r"((intptr_t)(src_stride)),  // %3
        "r"((intptr_t)(dst_stride)),  // %4
        "m"(kUVLinearMadd31)          // %5
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}
#endif

#ifdef HAS_SCALEUVROWUP2_LINEAR_16_SSE41
void ScaleUVRowUp2_Linear_16_SSE41(const uint16_t* src_ptr,
                                   uint16_t* dst_ptr,
                                   int dst_width) {
  asm volatile(
      "pxor        %%xmm5,%%xmm5                 \n"
      "pcmpeqd     %%xmm4,%%xmm4                 \n"
      "psrld       $31,%%xmm4                    \n"
      "pslld       $1,%%xmm4                     \n"  // all 2

      LABELALIGN
      "1:          \n"
      "movq        (%0),%%xmm0                   \n"  // 0011 (16b, 1u1v)
      "movq        4(%0),%%xmm1                  \n"  // 1122 (16b, 1u1v)

      "punpcklwd   %%xmm5,%%xmm0                 \n"  // 0011 (32b, 1u1v)
      "punpcklwd   %%xmm5,%%xmm1                 \n"  // 1122 (32b, 1u1v)

      "movdqa      %%xmm0,%%xmm2                 \n"
      "movdqa      %%xmm1,%%xmm3                 \n"

      "pshufd      $0b01001110,%%xmm2,%%xmm2     \n"  // 1100 (lo, far)
      "pshufd      $0b01001110,%%xmm3,%%xmm3     \n"  // 2211 (hi, far)

      "paddd       %%xmm4,%%xmm2                 \n"  // far+2 (lo)
      "paddd       %%xmm4,%%xmm3                 \n"  // far+2 (hi)
      "paddd       %%xmm0,%%xmm2                 \n"  // near+far+2 (lo)
      "paddd       %%xmm1,%%xmm3                 \n"  // near+far+2 (hi)
      "paddd       %%xmm0,%%xmm0                 \n"  // 2*near (lo)
      "paddd       %%xmm1,%%xmm1                 \n"  // 2*near (hi)
      "paddd       %%xmm2,%%xmm0                 \n"  // 3*near+far+2 (lo)
      "paddd       %%xmm3,%%xmm1                 \n"  // 3*near+far+2 (hi)

      "psrld       $2,%%xmm0                     \n"  // 3/4*near+1/4*far (lo)
      "psrld       $2,%%xmm1                     \n"  // 3/4*near+1/4*far (hi)
      "packusdw    %%xmm1,%%xmm0                 \n"
      "movdqu      %%xmm0,(%1)                   \n"

      "lea         0x8(%0),%0                    \n"
      "lea         0x10(%1),%1                   \n"  // 2 uv to 4 uv
      "sub         $0x4,%2                       \n"
      "jg          1b                            \n"
      : "+r"(src_ptr),   // %0
        "+r"(dst_ptr),   // %1
        "+r"(dst_width)  // %2
      :
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5");
}
#endif

#ifdef HAS_SCALEUVROWUP2_BILINEAR_16_SSE41
void ScaleUVRowUp2_Bilinear_16_SSE41(const uint16_t* src_ptr,
                                     ptrdiff_t src_stride,
                                     uint16_t* dst_ptr,
                                     ptrdiff_t dst_stride,
                                     int dst_width) {
  asm volatile(
      "pxor        %%xmm7,%%xmm7                 \n"
      "pcmpeqd     %%xmm6,%%xmm6                 \n"
      "psrld       $31,%%xmm6                    \n"
      "pslld       $3,%%xmm6                     \n"  // all 8

      LABELALIGN
      "1:          \n"
      "movq        (%0),%%xmm0                   \n"  // 0011 (16b, 1u1v)
      "movq        4(%0),%%xmm1                  \n"  // 1122 (16b, 1u1v)
      "punpcklwd   %%xmm7,%%xmm0                 \n"  // 0011 (near) (32b, 1u1v)
      "punpcklwd   %%xmm7,%%xmm1                 \n"  // 1122 (near) (32b, 1u1v)
      "movdqa      %%xmm0,%%xmm2                 \n"
      "movdqa      %%xmm1,%%xmm3                 \n"
      "pshufd      $0b01001110,%%xmm2,%%xmm2     \n"  // 1100 (far) (1, lo)
      "pshufd      $0b01001110,%%xmm3,%%xmm3     \n"  // 2211 (far) (1, hi)
      "paddd       %%xmm0,%%xmm2                 \n"  // near+far (1, lo)
      "paddd       %%xmm1,%%xmm3                 \n"  // near+far (1, hi)
      "paddd       %%xmm0,%%xmm0                 \n"  // 2*near (1, lo)
      "paddd       %%xmm1,%%xmm1                 \n"  // 2*near (1, hi)
      "paddd       %%xmm2,%%xmm0                 \n"  // 3*near+far (1, lo)
      "paddd       %%xmm3,%%xmm1                 \n"  // 3*near+far (1, hi)

      "movq        (%0,%3,2),%%xmm2              \n"
      "movq        4(%0,%3,2),%%xmm3             \n"
      "punpcklwd   %%xmm7,%%xmm2                 \n"
      "punpcklwd   %%xmm7,%%xmm3                 \n"
      "movdqa      %%xmm2,%%xmm4                 \n"
      "movdqa      %%xmm3,%%xmm5                 \n"
      "pshufd      $0b01001110,%%xmm4,%%xmm4     \n"  // 1100 (far) (2, lo)
      "pshufd      $0b01001110,%%xmm5,%%xmm5     \n"  // 2211 (far) (2, hi)
      "paddd       %%xmm2,%%xmm4                 \n"  // near+far (2, lo)
      "paddd       %%xmm3,%%xmm5                 \n"  // near+far (2, hi)
      "paddd       %%xmm2,%%xmm2                 \n"  // 2*near (2, lo)
      "paddd       %%xmm3,%%xmm3                 \n"  // 2*near (2, hi)
      "paddd       %%xmm4,%%xmm2                 \n"  // 3*near+far (2, lo)
      "paddd       %%xmm5,%%xmm3                 \n"  // 3*near+far (2, hi)

      "movdqa      %%xmm0,%%xmm4                 \n"
      "movdqa      %%xmm2,%%xmm5                 \n"
      "paddd       %%xmm0,%%xmm4                 \n"  // 6*near+2*far (1, lo)
      "paddd       %%xmm6,%%xmm5                 \n"  // 3*near+far+8 (2, lo)
      "paddd       %%xmm0,%%xmm4                 \n"  // 9*near+3*far (1, lo)
      "paddd       %%xmm5,%%xmm4                 \n"  // 9 3 3 1 + 8 (1, lo)
      "psrld       $4,%%xmm4                     \n"  // ^ div by 16 (1, lo)

      "movdqa      %%xmm2,%%xmm5                 \n"
      "paddd       %%xmm2,%%xmm5                 \n"  // 6*near+2*far (2, lo)
      "paddd       %%xmm6,%%xmm0                 \n"  // 3*near+far+8 (1, lo)
      "paddd       %%xmm2,%%xmm5                 \n"  // 9*near+3*far (2, lo)
      "paddd       %%xmm0,%%xmm5                 \n"  // 9 3 3 1 + 8 (2, lo)
      "psrld       $4,%%xmm5                     \n"  // ^ div by 16 (2, lo)

      "movdqa      %%xmm1,%%xmm0                 \n"
      "movdqa      %%xmm3,%%xmm2                 \n"
      "paddd       %%xmm1,%%xmm0                 \n"  // 6*near+2*far (1, hi)
      "paddd       %%xmm6,%%xmm2                 \n"  // 3*near+far+8 (2, hi)
      "paddd       %%xmm1,%%xmm0                 \n"  // 9*near+3*far (1, hi)
      "paddd       %%xmm2,%%xmm0                 \n"  // 9 3 3 1 + 8 (1, hi)
      "psrld       $4,%%xmm0                     \n"  // ^ div by 16 (1, hi)

      "movdqa      %%xmm3,%%xmm2                 \n"
      "paddd       %%xmm3,%%xmm2                 \n"  // 6*near+2*far (2, hi)
      "paddd       %%xmm6,%%xmm1                 \n"  // 3*near+far+8 (1, hi)
      "paddd       %%xmm3,%%xmm2                 \n"  // 9*near+3*far (2, hi)
      "paddd       %%xmm1,%%xmm2                 \n"  // 9 3 3 1 + 8 (2, hi)
      "psrld       $4,%%xmm2                     \n"  // ^ div by 16 (2, hi)

      "packusdw    %%xmm0,%%xmm4                 \n"
      "movdqu      %%xmm4,(%1)                   \n"  // store above
      "packusdw    %%xmm2,%%xmm5                 \n"
      "movdqu      %%xmm5,(%1,%4,2)              \n"  // store below

      "lea         0x8(%0),%0                    \n"
      "lea         0x10(%1),%1                   \n"  // 2 uv to 4 uv
      "sub         $0x4,%2                       \n"
      "jg          1b                            \n"
      : "+r"(src_ptr),                // %0
        "+r"(dst_ptr),                // %1
        "+r"(dst_width)               // %2
      : "r"((intptr_t)(src_stride)),  // %3
        "r"((intptr_t)(dst_stride))   // %4
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}
#endif

#ifdef HAS_SCALEUVROWUP2_LINEAR_16_AVX2
void ScaleUVRowUp2_Linear_16_AVX2(const uint16_t* src_ptr,
                                  uint16_t* dst_ptr,
                                  int dst_width) {
  asm volatile(
      "vpcmpeqd    %%ymm4,%%ymm4,%%ymm4          \n"
      "vpsrld      $31,%%ymm4,%%ymm4             \n"
      "vpslld      $1,%%ymm4,%%ymm4              \n"  // all 2

      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%xmm0                   \n"  // 00112233 (16b, 1u1v)
      "vmovdqu     4(%0),%%xmm1                  \n"  // 11223344 (16b, 1u1v)

      "vpmovzxwd   %%xmm0,%%ymm0                 \n"  // 01234567 (32b, 1u1v)
      "vpmovzxwd   %%xmm1,%%ymm1                 \n"  // 12345678 (32b, 1u1v)

      "vpshufd     $0b01001110,%%ymm0,%%ymm2     \n"  // 11003322 (lo, far)
      "vpshufd     $0b01001110,%%ymm1,%%ymm3     \n"  // 22114433 (hi, far)

      "vpaddd      %%ymm4,%%ymm2,%%ymm2          \n"  // far+2 (lo)
      "vpaddd      %%ymm4,%%ymm3,%%ymm3          \n"  // far+2 (hi)
      "vpaddd      %%ymm0,%%ymm2,%%ymm2          \n"  // near+far+2 (lo)
      "vpaddd      %%ymm1,%%ymm3,%%ymm3          \n"  // near+far+2 (hi)
      "vpaddd      %%ymm0,%%ymm0,%%ymm0          \n"  // 2*near (lo)
      "vpaddd      %%ymm1,%%ymm1,%%ymm1          \n"  // 2*near (hi)
      "vpaddd      %%ymm0,%%ymm2,%%ymm0          \n"  // 3*near+far+2 (lo)
      "vpaddd      %%ymm1,%%ymm3,%%ymm1          \n"  // 3*near+far+2 (hi)

      "vpsrld      $2,%%ymm0,%%ymm0              \n"  // 3/4*near+1/4*far (lo)
      "vpsrld      $2,%%ymm1,%%ymm1              \n"  // 3/4*near+1/4*far (hi)
      "vpackusdw   %%ymm1,%%ymm0,%%ymm0          \n"
      "vmovdqu     %%ymm0,(%1)                   \n"

      "lea         0x10(%0),%0                   \n"
      "lea         0x20(%1),%1                   \n"  // 4 uv to 8 uv
      "sub         $0x8,%2                       \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_ptr),   // %0
        "+r"(dst_ptr),   // %1
        "+r"(dst_width)  // %2
      :
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4");
}
#endif

#ifdef HAS_SCALEUVROWUP2_BILINEAR_16_AVX2
void ScaleUVRowUp2_Bilinear_16_AVX2(const uint16_t* src_ptr,
                                    ptrdiff_t src_stride,
                                    uint16_t* dst_ptr,
                                    ptrdiff_t dst_stride,
                                    int dst_width) {
  asm volatile(
      "vpcmpeqd    %%ymm6,%%ymm6,%%ymm6          \n"
      "vpsrld      $31,%%ymm6,%%ymm6             \n"
      "vpslld      $3,%%ymm6,%%ymm6              \n"  // all 8

      LABELALIGN
      "1:          \n"

      "vmovdqu     (%0),%%xmm0                   \n"  // 00112233 (16b, 1u1v)
      "vmovdqu     4(%0),%%xmm1                  \n"  // 11223344 (16b, 1u1v)
      "vpmovzxwd   %%xmm0,%%ymm0                 \n"  // 01234567 (32b, 1u1v)
      "vpmovzxwd   %%xmm1,%%ymm1                 \n"  // 12345678 (32b, 1u1v)
      "vpshufd     $0b01001110,%%ymm0,%%ymm2     \n"  // 11003322 (lo, far)
      "vpshufd     $0b01001110,%%ymm1,%%ymm3     \n"  // 22114433 (hi, far)
      "vpaddd      %%ymm0,%%ymm2,%%ymm2          \n"  // near+far (lo)
      "vpaddd      %%ymm1,%%ymm3,%%ymm3          \n"  // near+far (hi)
      "vpaddd      %%ymm0,%%ymm0,%%ymm0          \n"  // 2*near (lo)
      "vpaddd      %%ymm1,%%ymm1,%%ymm1          \n"  // 2*near (hi)
      "vpaddd      %%ymm0,%%ymm2,%%ymm0          \n"  // 3*near+far (lo)
      "vpaddd      %%ymm1,%%ymm3,%%ymm1          \n"  // 3*near+far (hi)

      "vmovdqu     (%0,%3,2),%%xmm2              \n"  // 00112233 (16b, 1u1v)
      "vmovdqu     4(%0,%3,2),%%xmm3             \n"  // 11223344 (16b, 1u1v)
      "vpmovzxwd   %%xmm2,%%ymm2                 \n"  // 01234567 (32b, 1u1v)
      "vpmovzxwd   %%xmm3,%%ymm3                 \n"  // 12345678 (32b, 1u1v)
      "vpshufd     $0b01001110,%%ymm2,%%ymm4     \n"  // 11003322 (lo, far)
      "vpshufd     $0b01001110,%%ymm3,%%ymm5     \n"  // 22114433 (hi, far)
      "vpaddd      %%ymm2,%%ymm4,%%ymm4          \n"  // near+far (lo)
      "vpaddd      %%ymm3,%%ymm5,%%ymm5          \n"  // near+far (hi)
      "vpaddd      %%ymm2,%%ymm2,%%ymm2          \n"  // 2*near (lo)
      "vpaddd      %%ymm3,%%ymm3,%%ymm3          \n"  // 2*near (hi)
      "vpaddd      %%ymm2,%%ymm4,%%ymm2          \n"  // 3*near+far (lo)
      "vpaddd      %%ymm3,%%ymm5,%%ymm3          \n"  // 3*near+far (hi)

      "vpaddd      %%ymm0,%%ymm0,%%ymm4          \n"  // 6*near+2*far (1, lo)
      "vpaddd      %%ymm6,%%ymm2,%%ymm5          \n"  // 3*near+far+8 (2, lo)
      "vpaddd      %%ymm4,%%ymm0,%%ymm4          \n"  // 9*near+3*far (1, lo)
      "vpaddd      %%ymm4,%%ymm5,%%ymm4          \n"  // 9 3 3 1 + 8 (1, lo)
      "vpsrld      $4,%%ymm4,%%ymm4              \n"  // ^ div by 16 (1, lo)

      "vpaddd      %%ymm2,%%ymm2,%%ymm5          \n"  // 6*near+2*far (2, lo)
      "vpaddd      %%ymm6,%%ymm0,%%ymm0          \n"  // 3*near+far+8 (1, lo)
      "vpaddd      %%ymm5,%%ymm2,%%ymm5          \n"  // 9*near+3*far (2, lo)
      "vpaddd      %%ymm5,%%ymm0,%%ymm5          \n"  // 9 3 3 1 + 8 (2, lo)
      "vpsrld      $4,%%ymm5,%%ymm5              \n"  // ^ div by 16 (2, lo)

      "vpaddd      %%ymm1,%%ymm1,%%ymm0          \n"  // 6*near+2*far (1, hi)
      "vpaddd      %%ymm6,%%ymm3,%%ymm2          \n"  // 3*near+far+8 (2, hi)
      "vpaddd      %%ymm0,%%ymm1,%%ymm0          \n"  // 9*near+3*far (1, hi)
      "vpaddd      %%ymm0,%%ymm2,%%ymm0          \n"  // 9 3 3 1 + 8 (1, hi)
      "vpsrld      $4,%%ymm0,%%ymm0              \n"  // ^ div by 16 (1, hi)

      "vpaddd      %%ymm3,%%ymm3,%%ymm2          \n"  // 6*near+2*far (2, hi)
      "vpaddd      %%ymm6,%%ymm1,%%ymm1          \n"  // 3*near+far+8 (1, hi)
      "vpaddd      %%ymm2,%%ymm3,%%ymm2          \n"  // 9*near+3*far (2, hi)
      "vpaddd      %%ymm2,%%ymm1,%%ymm2          \n"  // 9 3 3 1 + 8 (2, hi)
      "vpsrld      $4,%%ymm2,%%ymm2              \n"  // ^ div by 16 (2, hi)

      "vpackusdw   %%ymm0,%%ymm4,%%ymm4          \n"
      "vmovdqu     %%ymm4,(%1)                   \n"  // store above
      "vpackusdw   %%ymm2,%%ymm5,%%ymm5          \n"
      "vmovdqu     %%ymm5,(%1,%4,2)              \n"  // store below

      "lea         0x10(%0),%0                   \n"
      "lea         0x20(%1),%1                   \n"  // 4 uv to 8 uv
      "sub         $0x8,%2                       \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_ptr),                // %0
        "+r"(dst_ptr),                // %1
        "+r"(dst_width)               // %2
      : "r"((intptr_t)(src_stride)),  // %3
        "r"((intptr_t)(dst_stride))   // %4
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6");
}
#endif

#endif  // defined(__x86_64__) || defined(__i386__)

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
