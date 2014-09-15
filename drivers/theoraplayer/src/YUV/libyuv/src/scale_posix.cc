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

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// This module is for GCC x86 and x64.
#if !defined(LIBYUV_DISABLE_X86) && (defined(__x86_64__) || defined(__i386__))

// Offsets for source bytes 0 to 9
static uvec8 kShuf0 =
  { 0, 1, 3, 4, 5, 7, 8, 9, 128, 128, 128, 128, 128, 128, 128, 128 };

// Offsets for source bytes 11 to 20 with 8 subtracted = 3 to 12.
static uvec8 kShuf1 =
  { 3, 4, 5, 7, 8, 9, 11, 12, 128, 128, 128, 128, 128, 128, 128, 128 };

// Offsets for source bytes 21 to 31 with 16 subtracted = 5 to 31.
static uvec8 kShuf2 =
  { 5, 7, 8, 9, 11, 12, 13, 15, 128, 128, 128, 128, 128, 128, 128, 128 };

// Offsets for source bytes 0 to 10
static uvec8 kShuf01 =
  { 0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7, 8, 9, 9, 10 };

// Offsets for source bytes 10 to 21 with 8 subtracted = 3 to 13.
static uvec8 kShuf11 =
  { 2, 3, 4, 5, 5, 6, 6, 7, 8, 9, 9, 10, 10, 11, 12, 13 };

// Offsets for source bytes 21 to 31 with 16 subtracted = 5 to 31.
static uvec8 kShuf21 =
  { 5, 6, 6, 7, 8, 9, 9, 10, 10, 11, 12, 13, 13, 14, 14, 15 };

// Coefficients for source bytes 0 to 10
static uvec8 kMadd01 =
  { 3, 1, 2, 2, 1, 3, 3, 1, 2, 2, 1, 3, 3, 1, 2, 2 };

// Coefficients for source bytes 10 to 21
static uvec8 kMadd11 =
  { 1, 3, 3, 1, 2, 2, 1, 3, 3, 1, 2, 2, 1, 3, 3, 1 };

// Coefficients for source bytes 21 to 31
static uvec8 kMadd21 =
  { 2, 2, 1, 3, 3, 1, 2, 2, 1, 3, 3, 1, 2, 2, 1, 3 };

// Coefficients for source bytes 21 to 31
static vec16 kRound34 =
  { 2, 2, 2, 2, 2, 2, 2, 2 };

static uvec8 kShuf38a =
  { 0, 3, 6, 8, 11, 14, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128 };

static uvec8 kShuf38b =
  { 128, 128, 128, 128, 128, 128, 0, 3, 6, 8, 11, 14, 128, 128, 128, 128 };

// Arrange words 0,3,6 into 0,1,2
static uvec8 kShufAc =
  { 0, 1, 6, 7, 12, 13, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128 };

// Arrange words 0,3,6 into 3,4,5
static uvec8 kShufAc3 =
  { 128, 128, 128, 128, 128, 128, 0, 1, 6, 7, 12, 13, 128, 128, 128, 128 };

// Scaling values for boxes of 3x3 and 2x3
static uvec16 kScaleAc33 =
  { 65536 / 9, 65536 / 9, 65536 / 6, 65536 / 9, 65536 / 9, 65536 / 6, 0, 0 };

// Arrange first value for pixels 0,1,2,3,4,5
static uvec8 kShufAb0 =
  { 0, 128, 3, 128, 6, 128, 8, 128, 11, 128, 14, 128, 128, 128, 128, 128 };

// Arrange second value for pixels 0,1,2,3,4,5
static uvec8 kShufAb1 =
  { 1, 128, 4, 128, 7, 128, 9, 128, 12, 128, 15, 128, 128, 128, 128, 128 };

// Arrange third value for pixels 0,1,2,3,4,5
static uvec8 kShufAb2 =
  { 2, 128, 5, 128, 128, 128, 10, 128, 13, 128, 128, 128, 128, 128, 128, 128 };

// Scaling values for boxes of 3x2 and 2x2
static uvec16 kScaleAb2 =
  { 65536 / 3, 65536 / 3, 65536 / 2, 65536 / 3, 65536 / 3, 65536 / 2, 0, 0 };

// GCC versions of row functions are verbatim conversions from Visual C.
// Generated using gcc disassembly on Visual C object file:
// objdump -D yuvscaler.obj >yuvscaler.txt

void ScaleRowDown2_SSE2(const uint8* src_ptr, ptrdiff_t src_stride,
                        uint8* dst_ptr, int dst_width) {
  asm volatile (
    LABELALIGN
  "1:                                          \n"
    "movdqa    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqa    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    "lea       " MEMLEA(0x20,0) ",%0           \n"
    "psrlw     $0x8,%%xmm0                     \n"
    "psrlw     $0x8,%%xmm1                     \n"
    "packuswb  %%xmm1,%%xmm0                   \n"
    "movdqa    %%xmm0," MEMACCESS(1) "         \n"
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "sub       $0x10,%2                        \n"
    "jg        1b                              \n"
  : "+r"(src_ptr),    // %0
    "+r"(dst_ptr),    // %1
    "+r"(dst_width)   // %2
  :
  : "memory", "cc"
#if defined(__SSE2__)
    , "xmm0", "xmm1"
#endif
  );
}

void ScaleRowDown2Linear_SSE2(const uint8* src_ptr, ptrdiff_t src_stride,
                              uint8* dst_ptr, int dst_width) {
  asm volatile (
    "pcmpeqb   %%xmm5,%%xmm5                   \n"
    "psrlw     $0x8,%%xmm5                     \n"

    LABELALIGN
  "1:                                          \n"
    "movdqa    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqa    " MEMACCESS2(0x10, 0) ",%%xmm1  \n"
    "lea       " MEMLEA(0x20,0) ",%0           \n"
    "movdqa    %%xmm0,%%xmm2                   \n"
    "psrlw     $0x8,%%xmm0                     \n"
    "movdqa    %%xmm1,%%xmm3                   \n"
    "psrlw     $0x8,%%xmm1                     \n"
    "pand      %%xmm5,%%xmm2                   \n"
    "pand      %%xmm5,%%xmm3                   \n"
    "pavgw     %%xmm2,%%xmm0                   \n"
    "pavgw     %%xmm3,%%xmm1                   \n"
    "packuswb  %%xmm1,%%xmm0                   \n"
    "movdqa    %%xmm0," MEMACCESS(1) "         \n"
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "sub       $0x10,%2                        \n"
    "jg        1b                              \n"
  : "+r"(src_ptr),    // %0
    "+r"(dst_ptr),    // %1
    "+r"(dst_width)   // %2
  :
  : "memory", "cc"
#if defined(__SSE2__)
    , "xmm0", "xmm1", "xmm5"
#endif
  );
}

void ScaleRowDown2Box_SSE2(const uint8* src_ptr, ptrdiff_t src_stride,
                           uint8* dst_ptr, int dst_width) {
  asm volatile (
    "pcmpeqb   %%xmm5,%%xmm5                   \n"
    "psrlw     $0x8,%%xmm5                     \n"

    LABELALIGN
  "1:                                          \n"
    "movdqa    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqa    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    MEMOPREG(movdqa,0x00,0,3,1,xmm2)           //  movdqa  (%0,%3,1),%%xmm2
    BUNDLEALIGN
    MEMOPREG(movdqa,0x10,0,3,1,xmm3)           //  movdqa  0x10(%0,%3,1),%%xmm3
    "lea       " MEMLEA(0x20,0) ",%0           \n"
    "pavgb     %%xmm2,%%xmm0                   \n"
    "pavgb     %%xmm3,%%xmm1                   \n"
    "movdqa    %%xmm0,%%xmm2                   \n"
    "psrlw     $0x8,%%xmm0                     \n"
    "movdqa    %%xmm1,%%xmm3                   \n"
    "psrlw     $0x8,%%xmm1                     \n"
    "pand      %%xmm5,%%xmm2                   \n"
    "pand      %%xmm5,%%xmm3                   \n"
    "pavgw     %%xmm2,%%xmm0                   \n"
    "pavgw     %%xmm3,%%xmm1                   \n"
    "packuswb  %%xmm1,%%xmm0                   \n"
    "movdqa    %%xmm0," MEMACCESS(1) "         \n"
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "sub       $0x10,%2                        \n"
    "jg        1b                              \n"
  : "+r"(src_ptr),    // %0
    "+r"(dst_ptr),    // %1
    "+r"(dst_width)   // %2
  : "r"((intptr_t)(src_stride))   // %3
  : "memory", "cc"
#if defined(__native_client__) && defined(__x86_64__)
    , "r14"
#endif
#if defined(__SSE2__)
    , "xmm0", "xmm1", "xmm2", "xmm3", "xmm5"
#endif
  );
}

void ScaleRowDown2_Unaligned_SSE2(const uint8* src_ptr, ptrdiff_t src_stride,
                                  uint8* dst_ptr, int dst_width) {
  asm volatile (
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    "lea       " MEMLEA(0x20,0) ",%0           \n"
    "psrlw     $0x8,%%xmm0                     \n"
    "psrlw     $0x8,%%xmm1                     \n"
    "packuswb  %%xmm1,%%xmm0                   \n"
    "movdqu    %%xmm0," MEMACCESS(1) "         \n"
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "sub       $0x10,%2                        \n"
    "jg        1b                              \n"
  : "+r"(src_ptr),    // %0
    "+r"(dst_ptr),    // %1
    "+r"(dst_width)   // %2
  :
  : "memory", "cc"
#if defined(__SSE2__)
    , "xmm0", "xmm1"
#endif
  );
}

void ScaleRowDown2Linear_Unaligned_SSE2(const uint8* src_ptr,
                                        ptrdiff_t src_stride,
                                        uint8* dst_ptr, int dst_width) {
  asm volatile (
    "pcmpeqb   %%xmm5,%%xmm5                   \n"
    "psrlw     $0x8,%%xmm5                     \n"

    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    "lea       " MEMLEA(0x20,0) ",%0           \n"
    "movdqa    %%xmm0,%%xmm2                   \n"
    "psrlw     $0x8,%%xmm0                     \n"
    "movdqa    %%xmm1,%%xmm3                   \n"
    "psrlw     $0x8,%%xmm1                     \n"
    "pand      %%xmm5,%%xmm2                   \n"
    "pand      %%xmm5,%%xmm3                   \n"
    "pavgw     %%xmm2,%%xmm0                   \n"
    "pavgw     %%xmm3,%%xmm1                   \n"
    "packuswb  %%xmm1,%%xmm0                   \n"
    "movdqu    %%xmm0," MEMACCESS(1) "         \n"
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "sub       $0x10,%2                        \n"
    "jg        1b                              \n"
  : "+r"(src_ptr),    // %0
    "+r"(dst_ptr),    // %1
    "+r"(dst_width)   // %2
  :
  : "memory", "cc"
#if defined(__SSE2__)
    , "xmm0", "xmm1", "xmm5"
#endif
  );
}

void ScaleRowDown2Box_Unaligned_SSE2(const uint8* src_ptr,
                                     ptrdiff_t src_stride,
                                     uint8* dst_ptr, int dst_width) {
  asm volatile (
    "pcmpeqb   %%xmm5,%%xmm5                   \n"
    "psrlw     $0x8,%%xmm5                     \n"

    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    MEMOPREG(movdqu,0x00,0,3,1,xmm2)           //  movdqu  (%0,%3,1),%%xmm2
    BUNDLEALIGN
    MEMOPREG(movdqu,0x10,0,3,1,xmm3)           //  movdqu  0x10(%0,%3,1),%%xmm3
    "lea       " MEMLEA(0x20,0) ",%0           \n"
    "pavgb     %%xmm2,%%xmm0                   \n"
    "pavgb     %%xmm3,%%xmm1                   \n"
    "movdqa    %%xmm0,%%xmm2                   \n"
    "psrlw     $0x8,%%xmm0                     \n"
    "movdqa    %%xmm1,%%xmm3                   \n"
    "psrlw     $0x8,%%xmm1                     \n"
    "pand      %%xmm5,%%xmm2                   \n"
    "pand      %%xmm5,%%xmm3                   \n"
    "pavgw     %%xmm2,%%xmm0                   \n"
    "pavgw     %%xmm3,%%xmm1                   \n"
    "packuswb  %%xmm1,%%xmm0                   \n"
    "movdqu    %%xmm0," MEMACCESS(1) "         \n"
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "sub       $0x10,%2                        \n"
    "jg        1b                              \n"
  : "+r"(src_ptr),    // %0
    "+r"(dst_ptr),    // %1
    "+r"(dst_width)   // %2
  : "r"((intptr_t)(src_stride))   // %3
  : "memory", "cc"
#if defined(__native_client__) && defined(__x86_64__)
    , "r14"
#endif
#if defined(__SSE2__)
    , "xmm0", "xmm1", "xmm2", "xmm3", "xmm5"
#endif
  );
}

void ScaleRowDown4_SSE2(const uint8* src_ptr, ptrdiff_t src_stride,
                        uint8* dst_ptr, int dst_width) {
  asm volatile (
    "pcmpeqb   %%xmm5,%%xmm5                   \n"
    "psrld     $0x18,%%xmm5                    \n"
    "pslld     $0x10,%%xmm5                    \n"

    LABELALIGN
  "1:                                          \n"
    "movdqa    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqa    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    "lea       " MEMLEA(0x20,0) ",%0           \n"
    "pand      %%xmm5,%%xmm0                   \n"
    "pand      %%xmm5,%%xmm1                   \n"
    "packuswb  %%xmm1,%%xmm0                   \n"
    "psrlw     $0x8,%%xmm0                     \n"
    "packuswb  %%xmm0,%%xmm0                   \n"
    "movq      %%xmm0," MEMACCESS(1) "         \n"
    "lea       " MEMLEA(0x8,1) ",%1            \n"
    "sub       $0x8,%2                         \n"
    "jg        1b                              \n"
  : "+r"(src_ptr),    // %0
    "+r"(dst_ptr),    // %1
    "+r"(dst_width)   // %2
  :
  : "memory", "cc"
#if defined(__SSE2__)
    , "xmm0", "xmm1", "xmm5"
#endif
  );
}

void ScaleRowDown4Box_SSE2(const uint8* src_ptr, ptrdiff_t src_stride,
                           uint8* dst_ptr, int dst_width) {
  intptr_t stridex3 = 0;
  asm volatile (
    "pcmpeqb   %%xmm7,%%xmm7                   \n"
    "psrlw     $0x8,%%xmm7                     \n"
    "lea       " MEMLEA4(0x00,4,4,2) ",%3      \n"

    LABELALIGN
  "1:                                          \n"
    "movdqa    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqa    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    MEMOPREG(movdqa,0x00,0,4,1,xmm2)           //  movdqa  (%0,%4,1),%%xmm2
    BUNDLEALIGN
    MEMOPREG(movdqa,0x10,0,4,1,xmm3)           //  movdqa  0x10(%0,%4,1),%%xmm3
    "pavgb     %%xmm2,%%xmm0                   \n"
    "pavgb     %%xmm3,%%xmm1                   \n"
    MEMOPREG(movdqa,0x00,0,4,2,xmm2)           //  movdqa  (%0,%4,2),%%xmm2
    BUNDLEALIGN
    MEMOPREG(movdqa,0x10,0,4,2,xmm3)           //  movdqa  0x10(%0,%4,2),%%xmm3
    MEMOPREG(movdqa,0x00,0,3,1,xmm4)           //  movdqa  (%0,%3,1),%%xmm4
    MEMOPREG(movdqa,0x10,0,3,1,xmm5)           //  movdqa  0x10(%0,%3,1),%%xmm5
    "lea       " MEMLEA(0x20,0) ",%0           \n"
    "pavgb     %%xmm4,%%xmm2                   \n"
    "pavgb     %%xmm2,%%xmm0                   \n"
    "pavgb     %%xmm5,%%xmm3                   \n"
    "pavgb     %%xmm3,%%xmm1                   \n"
    "movdqa    %%xmm0,%%xmm2                   \n"
    "psrlw     $0x8,%%xmm0                     \n"
    "movdqa    %%xmm1,%%xmm3                   \n"
    "psrlw     $0x8,%%xmm1                     \n"
    "pand      %%xmm7,%%xmm2                   \n"
    "pand      %%xmm7,%%xmm3                   \n"
    "pavgw     %%xmm2,%%xmm0                   \n"
    "pavgw     %%xmm3,%%xmm1                   \n"
    "packuswb  %%xmm1,%%xmm0                   \n"
    "movdqa    %%xmm0,%%xmm2                   \n"
    "psrlw     $0x8,%%xmm0                     \n"
    "pand      %%xmm7,%%xmm2                   \n"
    "pavgw     %%xmm2,%%xmm0                   \n"
    "packuswb  %%xmm0,%%xmm0                   \n"
    "movq      %%xmm0," MEMACCESS(1) "         \n"
    "lea       " MEMLEA(0x8,1) ",%1            \n"
    "sub       $0x8,%2                         \n"
    "jg        1b                              \n"
  : "+r"(src_ptr),     // %0
    "+r"(dst_ptr),     // %1
    "+r"(dst_width),   // %2
    "+r"(stridex3)     // %3
  : "r"((intptr_t)(src_stride))    // %4
  : "memory", "cc"
#if defined(__native_client__) && defined(__x86_64__)
    , "r14"
#endif
#if defined(__SSE2__)
    , "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm7"
#endif
  );
}

void ScaleRowDown34_SSSE3(const uint8* src_ptr, ptrdiff_t src_stride,
                          uint8* dst_ptr, int dst_width) {
  asm volatile (
    "movdqa    %0,%%xmm3                       \n"
    "movdqa    %1,%%xmm4                       \n"
    "movdqa    %2,%%xmm5                       \n"
  :
  : "m"(kShuf0),  // %0
    "m"(kShuf1),  // %1
    "m"(kShuf2)   // %2
  );
  asm volatile (
    LABELALIGN
  "1:                                          \n"
    "movdqa    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqa    " MEMACCESS2(0x10,0) ",%%xmm2   \n"
    "lea       " MEMLEA(0x20,0) ",%0           \n"
    "movdqa    %%xmm2,%%xmm1                   \n"
    "palignr   $0x8,%%xmm0,%%xmm1              \n"
    "pshufb    %%xmm3,%%xmm0                   \n"
    "pshufb    %%xmm4,%%xmm1                   \n"
    "pshufb    %%xmm5,%%xmm2                   \n"
    "movq      %%xmm0," MEMACCESS(1) "         \n"
    "movq      %%xmm1," MEMACCESS2(0x8,1) "    \n"
    "movq      %%xmm2," MEMACCESS2(0x10,1) "   \n"
    "lea       " MEMLEA(0x18,1) ",%1           \n"
    "sub       $0x18,%2                        \n"
    "jg        1b                              \n"
  : "+r"(src_ptr),   // %0
    "+r"(dst_ptr),   // %1
    "+r"(dst_width)  // %2
  :
  : "memory", "cc"
#if defined(__SSE2__)
    , "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
#endif
  );
}

void ScaleRowDown34_1_Box_SSSE3(const uint8* src_ptr,
                                ptrdiff_t src_stride,
                                uint8* dst_ptr, int dst_width) {
  asm volatile (
    "movdqa    %0,%%xmm2                       \n"  // kShuf01
    "movdqa    %1,%%xmm3                       \n"  // kShuf11
    "movdqa    %2,%%xmm4                       \n"  // kShuf21
  :
  : "m"(kShuf01),  // %0
    "m"(kShuf11),  // %1
    "m"(kShuf21)   // %2
  );
  asm volatile (
    "movdqa    %0,%%xmm5                       \n"  // kMadd01
    "movdqa    %1,%%xmm0                       \n"  // kMadd11
    "movdqa    %2,%%xmm1                       \n"  // kRound34
  :
  : "m"(kMadd01),  // %0
    "m"(kMadd11),  // %1
    "m"(kRound34)  // %2
  );
  asm volatile (
    LABELALIGN
  "1:                                          \n"
    "movdqa    " MEMACCESS(0) ",%%xmm6         \n"
    MEMOPREG(movdqa,0x00,0,3,1,xmm7)           //  movdqa  (%0,%3),%%xmm7
    "pavgb     %%xmm7,%%xmm6                   \n"
    "pshufb    %%xmm2,%%xmm6                   \n"
    "pmaddubsw %%xmm5,%%xmm6                   \n"
    "paddsw    %%xmm1,%%xmm6                   \n"
    "psrlw     $0x2,%%xmm6                     \n"
    "packuswb  %%xmm6,%%xmm6                   \n"
    "movq      %%xmm6," MEMACCESS(1) "         \n"
    "movdqu    " MEMACCESS2(0x8,0) ",%%xmm6    \n"
    MEMOPREG(movdqu,0x8,0,3,1,xmm7)            //  movdqu  0x8(%0,%3),%%xmm7
    "pavgb     %%xmm7,%%xmm6                   \n"
    "pshufb    %%xmm3,%%xmm6                   \n"
    "pmaddubsw %%xmm0,%%xmm6                   \n"
    "paddsw    %%xmm1,%%xmm6                   \n"
    "psrlw     $0x2,%%xmm6                     \n"
    "packuswb  %%xmm6,%%xmm6                   \n"
    "movq      %%xmm6," MEMACCESS2(0x8,1) "    \n"
    "movdqa    " MEMACCESS2(0x10,0) ",%%xmm6   \n"
    BUNDLEALIGN
    MEMOPREG(movdqa,0x10,0,3,1,xmm7)           //  movdqa  0x10(%0,%3),%%xmm7
    "lea       " MEMLEA(0x20,0) ",%0           \n"
    "pavgb     %%xmm7,%%xmm6                   \n"
    "pshufb    %%xmm4,%%xmm6                   \n"
    "pmaddubsw %4,%%xmm6                       \n"
    "paddsw    %%xmm1,%%xmm6                   \n"
    "psrlw     $0x2,%%xmm6                     \n"
    "packuswb  %%xmm6,%%xmm6                   \n"
    "movq      %%xmm6," MEMACCESS2(0x10,1) "   \n"
    "lea       " MEMLEA(0x18,1) ",%1           \n"
    "sub       $0x18,%2                        \n"
    "jg        1b                              \n"
  : "+r"(src_ptr),   // %0
    "+r"(dst_ptr),   // %1
    "+r"(dst_width)  // %2
  : "r"((intptr_t)(src_stride)),  // %3
    "m"(kMadd21)     // %4
  : "memory", "cc"
#if defined(__native_client__) && defined(__x86_64__)
    , "r14"
#endif
#if defined(__SSE2__)
    , "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"
#endif
  );
}

void ScaleRowDown34_0_Box_SSSE3(const uint8* src_ptr,
                                ptrdiff_t src_stride,
                                uint8* dst_ptr, int dst_width) {
  asm volatile (
    "movdqa    %0,%%xmm2                       \n"  // kShuf01
    "movdqa    %1,%%xmm3                       \n"  // kShuf11
    "movdqa    %2,%%xmm4                       \n"  // kShuf21
  :
  : "m"(kShuf01),  // %0
    "m"(kShuf11),  // %1
    "m"(kShuf21)   // %2
  );
  asm volatile (
    "movdqa    %0,%%xmm5                       \n"  // kMadd01
    "movdqa    %1,%%xmm0                       \n"  // kMadd11
    "movdqa    %2,%%xmm1                       \n"  // kRound34
  :
  : "m"(kMadd01),  // %0
    "m"(kMadd11),  // %1
    "m"(kRound34)  // %2
  );

  asm volatile (
    LABELALIGN
  "1:                                          \n"
    "movdqa    " MEMACCESS(0) ",%%xmm6         \n"
    MEMOPREG(movdqa,0x00,0,3,1,xmm7)           //  movdqa  (%0,%3,1),%%xmm7
    "pavgb     %%xmm6,%%xmm7                   \n"
    "pavgb     %%xmm7,%%xmm6                   \n"
    "pshufb    %%xmm2,%%xmm6                   \n"
    "pmaddubsw %%xmm5,%%xmm6                   \n"
    "paddsw    %%xmm1,%%xmm6                   \n"
    "psrlw     $0x2,%%xmm6                     \n"
    "packuswb  %%xmm6,%%xmm6                   \n"
    "movq      %%xmm6," MEMACCESS(1) "         \n"
    "movdqu    " MEMACCESS2(0x8,0) ",%%xmm6    \n"
    MEMOPREG(movdqu,0x8,0,3,1,xmm7)            //  movdqu  0x8(%0,%3,1),%%xmm7
    "pavgb     %%xmm6,%%xmm7                   \n"
    "pavgb     %%xmm7,%%xmm6                   \n"
    "pshufb    %%xmm3,%%xmm6                   \n"
    "pmaddubsw %%xmm0,%%xmm6                   \n"
    "paddsw    %%xmm1,%%xmm6                   \n"
    "psrlw     $0x2,%%xmm6                     \n"
    "packuswb  %%xmm6,%%xmm6                   \n"
    "movq      %%xmm6," MEMACCESS2(0x8,1) "    \n"
    "movdqa    " MEMACCESS2(0x10,0) ",%%xmm6   \n"
    MEMOPREG(movdqa,0x10,0,3,1,xmm7)           //  movdqa  0x10(%0,%3,1),%%xmm7
    "lea       " MEMLEA(0x20,0) ",%0           \n"
    "pavgb     %%xmm6,%%xmm7                   \n"
    "pavgb     %%xmm7,%%xmm6                   \n"
    "pshufb    %%xmm4,%%xmm6                   \n"
    "pmaddubsw %4,%%xmm6                       \n"
    "paddsw    %%xmm1,%%xmm6                   \n"
    "psrlw     $0x2,%%xmm6                     \n"
    "packuswb  %%xmm6,%%xmm6                   \n"
    "movq      %%xmm6," MEMACCESS2(0x10,1) "   \n"
    "lea       " MEMLEA(0x18,1) ",%1           \n"
    "sub       $0x18,%2                        \n"
    "jg        1b                              \n"
    : "+r"(src_ptr),   // %0
      "+r"(dst_ptr),   // %1
      "+r"(dst_width)  // %2
    : "r"((intptr_t)(src_stride)),  // %3
      "m"(kMadd21)     // %4
    : "memory", "cc"
#if defined(__native_client__) && defined(__x86_64__)
    , "r14"
#endif
#if defined(__SSE2__)
    , "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"
#endif
  );
}

void ScaleRowDown38_SSSE3(const uint8* src_ptr, ptrdiff_t src_stride,
                          uint8* dst_ptr, int dst_width) {
  asm volatile (
    "movdqa    %3,%%xmm4                       \n"
    "movdqa    %4,%%xmm5                       \n"

    LABELALIGN
  "1:                                          \n"
    "movdqa    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqa    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    "lea       " MEMLEA(0x20,0) ",%0           \n"
    "pshufb    %%xmm4,%%xmm0                   \n"
    "pshufb    %%xmm5,%%xmm1                   \n"
    "paddusb   %%xmm1,%%xmm0                   \n"
    "movq      %%xmm0," MEMACCESS(1) "         \n"
    "movhlps   %%xmm0,%%xmm1                   \n"
    "movd      %%xmm1," MEMACCESS2(0x8,1) "    \n"
    "lea       " MEMLEA(0xc,1) ",%1            \n"
    "sub       $0xc,%2                         \n"
    "jg        1b                              \n"
  : "+r"(src_ptr),   // %0
    "+r"(dst_ptr),   // %1
    "+r"(dst_width)  // %2
  : "m"(kShuf38a),   // %3
    "m"(kShuf38b)    // %4
  : "memory", "cc"
#if defined(__SSE2__)
      , "xmm0", "xmm1", "xmm4", "xmm5"
#endif
  );
}

void ScaleRowDown38_2_Box_SSSE3(const uint8* src_ptr,
                                ptrdiff_t src_stride,
                                uint8* dst_ptr, int dst_width) {
  asm volatile (
    "movdqa    %0,%%xmm2                       \n"
    "movdqa    %1,%%xmm3                       \n"
    "movdqa    %2,%%xmm4                       \n"
    "movdqa    %3,%%xmm5                       \n"
  :
  : "m"(kShufAb0),   // %0
    "m"(kShufAb1),   // %1
    "m"(kShufAb2),   // %2
    "m"(kScaleAb2)   // %3
  );
  asm volatile (
    LABELALIGN
  "1:                                          \n"
    "movdqa    " MEMACCESS(0) ",%%xmm0         \n"
    MEMOPREG(pavgb,0x00,0,3,1,xmm0)            //  pavgb   (%0,%3,1),%%xmm0
    "lea       " MEMLEA(0x10,0) ",%0           \n"
    "movdqa    %%xmm0,%%xmm1                   \n"
    "pshufb    %%xmm2,%%xmm1                   \n"
    "movdqa    %%xmm0,%%xmm6                   \n"
    "pshufb    %%xmm3,%%xmm6                   \n"
    "paddusw   %%xmm6,%%xmm1                   \n"
    "pshufb    %%xmm4,%%xmm0                   \n"
    "paddusw   %%xmm0,%%xmm1                   \n"
    "pmulhuw   %%xmm5,%%xmm1                   \n"
    "packuswb  %%xmm1,%%xmm1                   \n"
    "sub       $0x6,%2                         \n"
    "movd      %%xmm1," MEMACCESS(1) "         \n"
    "psrlq     $0x10,%%xmm1                    \n"
    "movd      %%xmm1," MEMACCESS2(0x2,1) "    \n"
    "lea       " MEMLEA(0x6,1) ",%1            \n"
    "jg        1b                              \n"
  : "+r"(src_ptr),     // %0
    "+r"(dst_ptr),     // %1
    "+r"(dst_width)    // %2
  : "r"((intptr_t)(src_stride))  // %3
  : "memory", "cc"
#if defined(__native_client__) && defined(__x86_64__)
    , "r14"
#endif
#if defined(__SSE2__)
    , "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6"
#endif
  );
}

void ScaleRowDown38_3_Box_SSSE3(const uint8* src_ptr,
                                ptrdiff_t src_stride,
                                uint8* dst_ptr, int dst_width) {
  asm volatile (
    "movdqa    %0,%%xmm2                       \n"
    "movdqa    %1,%%xmm3                       \n"
    "movdqa    %2,%%xmm4                       \n"
    "pxor      %%xmm5,%%xmm5                   \n"
  :
  : "m"(kShufAc),    // %0
    "m"(kShufAc3),   // %1
    "m"(kScaleAc33)  // %2
  );
  asm volatile (
    LABELALIGN
  "1:                                          \n"
    "movdqa    " MEMACCESS(0) ",%%xmm0         \n"
    MEMOPREG(movdqa,0x00,0,3,1,xmm6)           //  movdqa  (%0,%3,1),%%xmm6
    "movhlps   %%xmm0,%%xmm1                   \n"
    "movhlps   %%xmm6,%%xmm7                   \n"
    "punpcklbw %%xmm5,%%xmm0                   \n"
    "punpcklbw %%xmm5,%%xmm1                   \n"
    "punpcklbw %%xmm5,%%xmm6                   \n"
    "punpcklbw %%xmm5,%%xmm7                   \n"
    "paddusw   %%xmm6,%%xmm0                   \n"
    "paddusw   %%xmm7,%%xmm1                   \n"
    MEMOPREG(movdqa,0x00,0,3,2,xmm6)           //  movdqa  (%0,%3,2),%%xmm6
    "lea       " MEMLEA(0x10,0) ",%0           \n"
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
    "sub       $0x6,%2                         \n"
    "movd      %%xmm6," MEMACCESS(1) "         \n"
    "psrlq     $0x10,%%xmm6                    \n"
    "movd      %%xmm6," MEMACCESS2(0x2,1) "    \n"
    "lea       " MEMLEA(0x6,1) ",%1            \n"
    "jg        1b                              \n"
  : "+r"(src_ptr),    // %0
    "+r"(dst_ptr),    // %1
    "+r"(dst_width)   // %2
  : "r"((intptr_t)(src_stride))   // %3
  : "memory", "cc"
#if defined(__native_client__) && defined(__x86_64__)
    , "r14"
#endif
#if defined(__SSE2__)
    , "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"
#endif
  );
}

void ScaleAddRows_SSE2(const uint8* src_ptr, ptrdiff_t src_stride,
                       uint16* dst_ptr, int src_width, int src_height) {
  int tmp_height = 0;
  intptr_t tmp_src = 0;
  asm volatile (
    "pxor      %%xmm4,%%xmm4                   \n"
    "sub       $0x1,%5                         \n"

    LABELALIGN
  "1:                                          \n"
    "movdqa    " MEMACCESS(0) ",%%xmm0         \n"
    "mov       %0,%3                           \n"
    "add       %6,%0                           \n"
    "movdqa    %%xmm0,%%xmm1                   \n"
    "punpcklbw %%xmm4,%%xmm0                   \n"
    "punpckhbw %%xmm4,%%xmm1                   \n"
    "mov       %5,%2                           \n"
    "test      %2,%2                           \n"
    "je        3f                              \n"

    LABELALIGN
  "2:                                          \n"
    "movdqa    " MEMACCESS(0) ",%%xmm2         \n"
    "add       %6,%0                           \n"
    "movdqa    %%xmm2,%%xmm3                   \n"
    "punpcklbw %%xmm4,%%xmm2                   \n"
    "punpckhbw %%xmm4,%%xmm3                   \n"
    "paddusw   %%xmm2,%%xmm0                   \n"
    "paddusw   %%xmm3,%%xmm1                   \n"
    "sub       $0x1,%2                         \n"
    "jg        2b                              \n"

    LABELALIGN
  "3:                                          \n"
    "movdqa    %%xmm0," MEMACCESS(1) "         \n"
    "movdqa    %%xmm1," MEMACCESS2(0x10,1) "   \n"
    "lea       " MEMLEA(0x10,3) ",%0           \n"
    "lea       " MEMLEA(0x20,1) ",%1           \n"
    "sub       $0x10,%4                        \n"
    "jg        1b                              \n"
  : "+r"(src_ptr),     // %0
    "+r"(dst_ptr),     // %1
    "+r"(tmp_height),  // %2
    "+r"(tmp_src),     // %3
    "+r"(src_width),   // %4
    "+rm"(src_height)  // %5
  : "rm"((intptr_t)(src_stride))  // %6
  : "memory", "cc"
#if defined(__SSE2__)
    , "xmm0", "xmm1", "xmm2", "xmm3", "xmm4"
#endif
  );
}

// Bilinear column filtering. SSSE3 version.
void ScaleFilterCols_SSSE3(uint8* dst_ptr, const uint8* src_ptr,
                           int dst_width, int x, int dx) {
  intptr_t x0 = 0, x1 = 0, temp_pixel = 0;
  asm volatile (
    "movd      %6,%%xmm2                       \n"
    "movd      %7,%%xmm3                       \n"
    "movl      $0x04040000,%k2                 \n"
    "movd      %k2,%%xmm5                      \n"
    "pcmpeqb   %%xmm6,%%xmm6                   \n"
    "psrlw     $0x9,%%xmm6                     \n"
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
  "2:                                          \n"
    "movdqa    %%xmm2,%%xmm1                   \n"
    "paddd     %%xmm3,%%xmm2                   \n"
    MEMOPARG(movzwl,0x00,1,3,1,k2)             //  movzwl  (%1,%3,1),%k2
    "movd      %k2,%%xmm0                      \n"
    "psrlw     $0x9,%%xmm1                     \n"
    BUNDLEALIGN
    MEMOPARG(movzwl,0x00,1,4,1,k2)             //  movzwl  (%1,%4,1),%k2
    "movd      %k2,%%xmm4                      \n"
    "pshufb    %%xmm5,%%xmm1                   \n"
    "punpcklwd %%xmm4,%%xmm0                   \n"
    "pxor      %%xmm6,%%xmm1                   \n"
    "pmaddubsw %%xmm1,%%xmm0                   \n"
    "pextrw    $0x1,%%xmm2,%k3                 \n"
    "pextrw    $0x3,%%xmm2,%k4                 \n"
    "psrlw     $0x7,%%xmm0                     \n"
    "packuswb  %%xmm0,%%xmm0                   \n"
    "movd      %%xmm0,%k2                      \n"
    "mov       %w2," MEMACCESS(0) "            \n"
    "lea       " MEMLEA(0x2,0) ",%0            \n"
    "sub       $0x2,%5                         \n"
    "jge       2b                              \n"

    LABELALIGN
  "29:                                         \n"
    "addl      $0x1,%5                         \n"
    "jl        99f                             \n"
    MEMOPARG(movzwl,0x00,1,3,1,k2)             //  movzwl  (%1,%3,1),%k2
    "movd      %k2,%%xmm0                      \n"
    "psrlw     $0x9,%%xmm2                     \n"
    "pshufb    %%xmm5,%%xmm2                   \n"
    "pxor      %%xmm6,%%xmm2                   \n"
    "pmaddubsw %%xmm2,%%xmm0                   \n"
    "psrlw     $0x7,%%xmm0                     \n"
    "packuswb  %%xmm0,%%xmm0                   \n"
    "movd      %%xmm0,%k2                      \n"
    "mov       %b2," MEMACCESS(0) "            \n"
  "99:                                         \n"
  : "+r"(dst_ptr),     // %0
    "+r"(src_ptr),     // %1
    "+a"(temp_pixel),  // %2
    "+r"(x0),          // %3
    "+r"(x1),          // %4
    "+rm"(dst_width)   // %5
  : "rm"(x),           // %6
    "rm"(dx)           // %7
  : "memory", "cc"
#if defined(__native_client__) && defined(__x86_64__)
    , "r14"
#endif
#if defined(__SSE2__)
    , "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6"
#endif
  );
}

// Reads 4 pixels, duplicates them and writes 8 pixels.
// Alignment requirement: src_argb 16 byte aligned, dst_argb 16 byte aligned.
void ScaleColsUp2_SSE2(uint8* dst_ptr, const uint8* src_ptr,
                       int dst_width, int x, int dx) {
  asm volatile (
    LABELALIGN
  "1:                                          \n"
    "movdqa    " MEMACCESS(1) ",%%xmm0         \n"
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "movdqa    %%xmm0,%%xmm1                   \n"
    "punpcklbw %%xmm0,%%xmm0                   \n"
    "punpckhbw %%xmm1,%%xmm1                   \n"
    "sub       $0x20,%2                         \n"
    "movdqa    %%xmm0," MEMACCESS(0) "         \n"
    "movdqa    %%xmm1," MEMACCESS2(0x10,0) "   \n"
    "lea       " MEMLEA(0x20,0) ",%0           \n"
    "jg        1b                              \n"

  : "+r"(dst_ptr),     // %0
    "+r"(src_ptr),     // %1
    "+r"(dst_width)    // %2
  :
  : "memory", "cc"
#if defined(__SSE2__)
    , "xmm0", "xmm1"
#endif
  );
}

void ScaleARGBRowDown2_SSE2(const uint8* src_argb,
                            ptrdiff_t src_stride,
                            uint8* dst_argb, int dst_width) {
  asm volatile (
    LABELALIGN
  "1:                                          \n"
    "movdqa    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqa    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    "lea       " MEMLEA(0x20,0) ",%0           \n"
    "shufps    $0xdd,%%xmm1,%%xmm0             \n"
    "sub       $0x4,%2                         \n"
    "movdqa    %%xmm0," MEMACCESS(1) "         \n"
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "jg        1b                              \n"
  : "+r"(src_argb),  // %0
    "+r"(dst_argb),  // %1
    "+r"(dst_width)  // %2
  :
  : "memory", "cc"
#if defined(__SSE2__)
    , "xmm0", "xmm1"
#endif
  );
}

void ScaleARGBRowDown2Linear_SSE2(const uint8* src_argb,
                                  ptrdiff_t src_stride,
                                  uint8* dst_argb, int dst_width) {
  asm volatile (
    LABELALIGN
  "1:                                          \n"
    "movdqa    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqa    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    "lea       " MEMLEA(0x20,0) ",%0           \n"
    "movdqa    %%xmm0,%%xmm2                   \n"
    "shufps    $0x88,%%xmm1,%%xmm0             \n"
    "shufps    $0xdd,%%xmm1,%%xmm2             \n"
    "pavgb     %%xmm2,%%xmm0                   \n"
    "sub       $0x4,%2                         \n"
    "movdqa    %%xmm0," MEMACCESS(1) "         \n"
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "jg        1b                              \n"
  : "+r"(src_argb),  // %0
    "+r"(dst_argb),  // %1
    "+r"(dst_width)  // %2
  :
  : "memory", "cc"
#if defined(__SSE2__)
    , "xmm0", "xmm1"
#endif
  );
}

void ScaleARGBRowDown2Box_SSE2(const uint8* src_argb,
                               ptrdiff_t src_stride,
                               uint8* dst_argb, int dst_width) {
  asm volatile (
    LABELALIGN
  "1:                                          \n"
    "movdqa    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqa    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    BUNDLEALIGN
    MEMOPREG(movdqa,0x00,0,3,1,xmm2)           //  movdqa   (%0,%3,1),%%xmm2
    MEMOPREG(movdqa,0x10,0,3,1,xmm3)           //  movdqa   0x10(%0,%3,1),%%xmm3
    "lea       " MEMLEA(0x20,0) ",%0           \n"
    "pavgb     %%xmm2,%%xmm0                   \n"
    "pavgb     %%xmm3,%%xmm1                   \n"
    "movdqa    %%xmm0,%%xmm2                   \n"
    "shufps    $0x88,%%xmm1,%%xmm0             \n"
    "shufps    $0xdd,%%xmm1,%%xmm2             \n"
    "pavgb     %%xmm2,%%xmm0                   \n"
    "sub       $0x4,%2                         \n"
    "movdqa    %%xmm0," MEMACCESS(1) "         \n"
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "jg        1b                              \n"
  : "+r"(src_argb),   // %0
    "+r"(dst_argb),   // %1
    "+r"(dst_width)   // %2
  : "r"((intptr_t)(src_stride))   // %3
  : "memory", "cc"
#if defined(__native_client__) && defined(__x86_64__)
    , "r14"
#endif
#if defined(__SSE2__)
    , "xmm0", "xmm1", "xmm2", "xmm3"
#endif
  );
}

// Reads 4 pixels at a time.
// Alignment requirement: dst_argb 16 byte aligned.
void ScaleARGBRowDownEven_SSE2(const uint8* src_argb, ptrdiff_t src_stride,
                               int src_stepx,
                               uint8* dst_argb, int dst_width) {
  intptr_t src_stepx_x4 = (intptr_t)(src_stepx);
  intptr_t src_stepx_x12 = 0;
  asm volatile (
    "lea       " MEMLEA3(0x00,1,4) ",%1        \n"
    "lea       " MEMLEA4(0x00,1,1,2) ",%4      \n"
    LABELALIGN
  "1:                                          \n"
    "movd      " MEMACCESS(0) ",%%xmm0         \n"
    MEMOPREG(movd,0x00,0,1,1,xmm1)             //  movd      (%0,%1,1),%%xmm1
    "punpckldq %%xmm1,%%xmm0                   \n"
    BUNDLEALIGN
    MEMOPREG(movd,0x00,0,1,2,xmm2)             //  movd      (%0,%1,2),%%xmm2
    MEMOPREG(movd,0x00,0,4,1,xmm3)             //  movd      (%0,%4,1),%%xmm3
    "lea       " MEMLEA4(0x00,0,1,4) ",%0      \n"
    "punpckldq %%xmm3,%%xmm2                   \n"
    "punpcklqdq %%xmm2,%%xmm0                  \n"
    "sub       $0x4,%3                         \n"
    "movdqa    %%xmm0," MEMACCESS(2) "         \n"
    "lea       " MEMLEA(0x10,2) ",%2           \n"
    "jg        1b                              \n"
  : "+r"(src_argb),      // %0
    "+r"(src_stepx_x4),  // %1
    "+r"(dst_argb),      // %2
    "+r"(dst_width),     // %3
    "+r"(src_stepx_x12)  // %4
  :
  : "memory", "cc"
#if defined(__native_client__) && defined(__x86_64__)
    , "r14"
#endif
#if defined(__SSE2__)
    , "xmm0", "xmm1", "xmm2", "xmm3"
#endif
  );
}

// Blends four 2x2 to 4x1.
// Alignment requirement: dst_argb 16 byte aligned.
void ScaleARGBRowDownEvenBox_SSE2(const uint8* src_argb,
                                  ptrdiff_t src_stride, int src_stepx,
                                  uint8* dst_argb, int dst_width) {
  intptr_t src_stepx_x4 = (intptr_t)(src_stepx);
  intptr_t src_stepx_x12 = 0;
  intptr_t row1 = (intptr_t)(src_stride);
  asm volatile (
    "lea       " MEMLEA3(0x00,1,4) ",%1        \n"
    "lea       " MEMLEA4(0x00,1,1,2) ",%4      \n"
    "lea       " MEMLEA4(0x00,0,5,1) ",%5      \n"

    LABELALIGN
  "1:                                          \n"
    "movq      " MEMACCESS(0) ",%%xmm0         \n"
    MEMOPREG(movhps,0x00,0,1,1,xmm0)           //  movhps    (%0,%1,1),%%xmm0
    MEMOPREG(movq,0x00,0,1,2,xmm1)             //  movq      (%0,%1,2),%%xmm1
    BUNDLEALIGN
    MEMOPREG(movhps,0x00,0,4,1,xmm1)           //  movhps    (%0,%4,1),%%xmm1
    "lea       " MEMLEA4(0x00,0,1,4) ",%0      \n"
    "movq      " MEMACCESS(5) ",%%xmm2         \n"
    BUNDLEALIGN
    MEMOPREG(movhps,0x00,5,1,1,xmm2)           //  movhps    (%5,%1,1),%%xmm2
    MEMOPREG(movq,0x00,5,1,2,xmm3)             //  movq      (%5,%1,2),%%xmm3
    MEMOPREG(movhps,0x00,5,4,1,xmm3)           //  movhps    (%5,%4,1),%%xmm3
    "lea       " MEMLEA4(0x00,5,1,4) ",%5      \n"
    "pavgb     %%xmm2,%%xmm0                   \n"
    "pavgb     %%xmm3,%%xmm1                   \n"
    "movdqa    %%xmm0,%%xmm2                   \n"
    "shufps    $0x88,%%xmm1,%%xmm0             \n"
    "shufps    $0xdd,%%xmm1,%%xmm2             \n"
    "pavgb     %%xmm2,%%xmm0                   \n"
    "sub       $0x4,%3                         \n"
    "movdqa    %%xmm0," MEMACCESS(2) "         \n"
    "lea       " MEMLEA(0x10,2) ",%2           \n"
    "jg        1b                              \n"
  : "+r"(src_argb),       // %0
    "+r"(src_stepx_x4),   // %1
    "+r"(dst_argb),       // %2
    "+rm"(dst_width),     // %3
    "+r"(src_stepx_x12),  // %4
    "+r"(row1)            // %5
  :
  : "memory", "cc"
#if defined(__native_client__) && defined(__x86_64__)
    , "r14"
#endif
#if defined(__SSE2__)
    , "xmm0", "xmm1", "xmm2", "xmm3"
#endif
  );
}

void ScaleARGBCols_SSE2(uint8* dst_argb, const uint8* src_argb,
                        int dst_width, int x, int dx) {
  intptr_t x0 = 0, x1 = 0;
  asm volatile (
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
  "40:                                         \n"
    MEMOPREG(movd,0x00,3,0,4,xmm0)             //  movd      (%3,%0,4),%%xmm0
    MEMOPREG(movd,0x00,3,1,4,xmm1)             //  movd      (%3,%1,4),%%xmm1
    "pextrw    $0x5,%%xmm2,%k0                 \n"
    "pextrw    $0x7,%%xmm2,%k1                 \n"
    "paddd     %%xmm3,%%xmm2                   \n"
    "punpckldq %%xmm1,%%xmm0                   \n"
    MEMOPREG(movd,0x00,3,0,4,xmm1)             //  movd      (%3,%0,4),%%xmm1
    MEMOPREG(movd,0x00,3,1,4,xmm4)             //  movd      (%3,%1,4),%%xmm4
    "pextrw    $0x1,%%xmm2,%k0                 \n"
    "pextrw    $0x3,%%xmm2,%k1                 \n"
    "punpckldq %%xmm4,%%xmm1                   \n"
    "punpcklqdq %%xmm1,%%xmm0                  \n"
    "sub       $0x4,%4                         \n"
    "movdqu    %%xmm0," MEMACCESS(2) "         \n"
    "lea       " MEMLEA(0x10,2) ",%2           \n"
    "jge       40b                             \n"

  "49:                                         \n"
    "test      $0x2,%4                         \n"
    "je        29f                             \n"
    BUNDLEALIGN
    MEMOPREG(movd,0x00,3,0,4,xmm0)             //  movd      (%3,%0,4),%%xmm0
    MEMOPREG(movd,0x00,3,1,4,xmm1)             //  movd      (%3,%1,4),%%xmm1
    "pextrw    $0x5,%%xmm2,%k0                 \n"
    "punpckldq %%xmm1,%%xmm0                   \n"
    "movq      %%xmm0," MEMACCESS(2) "         \n"
    "lea       " MEMLEA(0x8,2) ",%2            \n"
  "29:                                         \n"
    "test      $0x1,%4                         \n"
    "je        99f                             \n"
    MEMOPREG(movd,0x00,3,0,4,xmm0)             //  movd      (%3,%0,4),%%xmm0
    "movd      %%xmm0," MEMACCESS(2) "         \n"
  "99:                                         \n"
  : "+a"(x0),          // %0
    "+d"(x1),          // %1
    "+r"(dst_argb),    // %2
    "+r"(src_argb),    // %3
    "+r"(dst_width)    // %4
  : "rm"(x),           // %5
    "rm"(dx)           // %6
  : "memory", "cc"
#if defined(__native_client__) && defined(__x86_64__)
    , "r14"
#endif
#if defined(__SSE2__)
    , "xmm0", "xmm1", "xmm2", "xmm3", "xmm4"
#endif
  );
}

// Reads 4 pixels, duplicates them and writes 8 pixels.
// Alignment requirement: src_argb 16 byte aligned, dst_argb 16 byte aligned.
void ScaleARGBColsUp2_SSE2(uint8* dst_argb, const uint8* src_argb,
                           int dst_width, int x, int dx) {
  asm volatile (
    LABELALIGN
  "1:                                          \n"
    "movdqa    " MEMACCESS(1) ",%%xmm0         \n"
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "movdqa    %%xmm0,%%xmm1                   \n"
    "punpckldq %%xmm0,%%xmm0                   \n"
    "punpckhdq %%xmm1,%%xmm1                   \n"
    "sub       $0x8,%2                         \n"
    "movdqa    %%xmm0," MEMACCESS(0) "         \n"
    "movdqa    %%xmm1," MEMACCESS2(0x10,0) "   \n"
    "lea       " MEMLEA(0x20,0) ",%0           \n"
    "jg        1b                              \n"

  : "+r"(dst_argb),    // %0
    "+r"(src_argb),    // %1
    "+r"(dst_width)    // %2
  :
  : "memory", "cc"
#if defined(__native_client__) && defined(__x86_64__)
    , "r14"
#endif
#if defined(__SSE2__)
    , "xmm0", "xmm1"
#endif
  );
}

// Shuffle table for arranging 2 pixels into pairs for pmaddubsw
static uvec8 kShuffleColARGB = {
  0u, 4u, 1u, 5u, 2u, 6u, 3u, 7u,  // bbggrraa 1st pixel
  8u, 12u, 9u, 13u, 10u, 14u, 11u, 15u  // bbggrraa 2nd pixel
};

// Shuffle table for duplicating 2 fractions into 8 bytes each
static uvec8 kShuffleFractions = {
  0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u,
};

// Bilinear row filtering combines 4x2 -> 4x1. SSSE3 version
void ScaleARGBFilterCols_SSSE3(uint8* dst_argb, const uint8* src_argb,
                               int dst_width, int x, int dx) {
  intptr_t x0 = 0, x1 = 0;
  asm volatile (
    "movdqa    %0,%%xmm4                       \n"
    "movdqa    %1,%%xmm5                       \n"
  :
  : "m"(kShuffleColARGB),  // %0
    "m"(kShuffleFractions)  // %1
  );

  asm volatile (
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
  "2:                                          \n"
    "movdqa    %%xmm2,%%xmm1                   \n"
    "paddd     %%xmm3,%%xmm2                   \n"
    MEMOPREG(movq,0x00,1,3,4,xmm0)             //  movq      (%1,%3,4),%%xmm0
    "psrlw     $0x9,%%xmm1                     \n"
    BUNDLEALIGN
    MEMOPREG(movhps,0x00,1,4,4,xmm0)           //  movhps    (%1,%4,4),%%xmm0
    "pshufb    %%xmm5,%%xmm1                   \n"
    "pshufb    %%xmm4,%%xmm0                   \n"
    "pxor      %%xmm6,%%xmm1                   \n"
    "pmaddubsw %%xmm1,%%xmm0                   \n"
    "psrlw     $0x7,%%xmm0                     \n"
    "pextrw    $0x1,%%xmm2,%k3                 \n"
    "pextrw    $0x3,%%xmm2,%k4                 \n"
    "packuswb  %%xmm0,%%xmm0                   \n"
    "movq      %%xmm0," MEMACCESS(0) "         \n"
    "lea       " MEMLEA(0x8,0) ",%0            \n"
    "sub       $0x2,%2                         \n"
    "jge       2b                              \n"

    LABELALIGN
  "29:                                         \n"
    "add       $0x1,%2                         \n"
    "jl        99f                             \n"
    "psrlw     $0x9,%%xmm2                     \n"
    BUNDLEALIGN
    MEMOPREG(movq,0x00,1,3,4,xmm0)             //  movq      (%1,%3,4),%%xmm0
    "pshufb    %%xmm5,%%xmm2                   \n"
    "pshufb    %%xmm4,%%xmm0                   \n"
    "pxor      %%xmm6,%%xmm2                   \n"
    "pmaddubsw %%xmm2,%%xmm0                   \n"
    "psrlw     $0x7,%%xmm0                     \n"
    "packuswb  %%xmm0,%%xmm0                   \n"
    "movd      %%xmm0," MEMACCESS(0) "         \n"

    LABELALIGN
  "99:                                         \n"
  : "+r"(dst_argb),    // %0
    "+r"(src_argb),    // %1
    "+rm"(dst_width),  // %2
    "+r"(x0),          // %3
    "+r"(x1)           // %4
  : "rm"(x),           // %5
    "rm"(dx)           // %6
  : "memory", "cc"
#if defined(__native_client__) && defined(__x86_64__)
    , "r14"
#endif
#if defined(__SSE2__)
    , "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6"
#endif
  );
}

// Divide num by div and return as 16.16 fixed point result.
int FixedDiv_X86(int num, int div) {
  asm volatile (
    "cdq                                       \n"
    "shld      $0x10,%%eax,%%edx               \n"
    "shl       $0x10,%%eax                     \n"
    "idiv      %1                              \n"
    "mov       %0, %%eax                       \n"
    : "+a"(num)  // %0
    : "c"(div)   // %1
    : "memory", "cc", "edx"
  );
  return num;
}

// Divide num - 1 by div - 1 and return as 16.16 fixed point result.
int FixedDiv1_X86(int num, int div) {
  asm volatile (
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
    : "memory", "cc", "edx"
  );
  return num;
}

#endif  // defined(__x86_64__) || defined(__i386__)

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
