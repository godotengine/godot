/*
 *  Copyright 2015 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "libyuv/rotate_row.h"
#include "libyuv/row.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// This module is for GCC x86 and x64.
#if !defined(LIBYUV_DISABLE_X86) &&               \
    (defined(__x86_64__) || defined(__i386__)) && \
    !defined(LIBYUV_ENABLE_ROWWIN)

// Transpose 8x8. 32 or 64 bit, but not NaCL for 64 bit.
#if defined(HAS_TRANSPOSEWX8_SSSE3)
void TransposeWx8_SSSE3(const uint8_t* src,
                        int src_stride,
                        uint8_t* dst,
                        int dst_stride,
                        int width) {
  asm volatile(
      // Read in the data from the source pointer.
      // First round of bit swap.
      LABELALIGN
      "1:          \n"
      "movq        (%0),%%xmm0                   \n"
      "movq        (%0,%3),%%xmm1                \n"
      "lea         (%0,%3,2),%0                  \n"
      "punpcklbw   %%xmm1,%%xmm0                 \n"
      "movq        (%0),%%xmm2                   \n"
      "movdqa      %%xmm0,%%xmm1                 \n"
      "palignr     $0x8,%%xmm1,%%xmm1            \n"
      "movq        (%0,%3),%%xmm3                \n"
      "lea         (%0,%3,2),%0                  \n"
      "punpcklbw   %%xmm3,%%xmm2                 \n"
      "movdqa      %%xmm2,%%xmm3                 \n"
      "movq        (%0),%%xmm4                   \n"
      "palignr     $0x8,%%xmm3,%%xmm3            \n"
      "movq        (%0,%3),%%xmm5                \n"
      "lea         (%0,%3,2),%0                  \n"
      "punpcklbw   %%xmm5,%%xmm4                 \n"
      "movdqa      %%xmm4,%%xmm5                 \n"
      "movq        (%0),%%xmm6                   \n"
      "palignr     $0x8,%%xmm5,%%xmm5            \n"
      "movq        (%0,%3),%%xmm7                \n"
      "lea         (%0,%3,2),%0                  \n"
      "punpcklbw   %%xmm7,%%xmm6                 \n"
      "neg         %3                            \n"
      "movdqa      %%xmm6,%%xmm7                 \n"
      "lea         0x8(%0,%3,8),%0               \n"
      "palignr     $0x8,%%xmm7,%%xmm7            \n"
      "neg         %3                            \n"
      // Second round of bit swap.
      "punpcklwd   %%xmm2,%%xmm0                 \n"
      "punpcklwd   %%xmm3,%%xmm1                 \n"
      "movdqa      %%xmm0,%%xmm2                 \n"
      "movdqa      %%xmm1,%%xmm3                 \n"
      "palignr     $0x8,%%xmm2,%%xmm2            \n"
      "palignr     $0x8,%%xmm3,%%xmm3            \n"
      "punpcklwd   %%xmm6,%%xmm4                 \n"
      "punpcklwd   %%xmm7,%%xmm5                 \n"
      "movdqa      %%xmm4,%%xmm6                 \n"
      "movdqa      %%xmm5,%%xmm7                 \n"
      "palignr     $0x8,%%xmm6,%%xmm6            \n"
      "palignr     $0x8,%%xmm7,%%xmm7            \n"
      // Third round of bit swap.
      // Write to the destination pointer.
      "punpckldq   %%xmm4,%%xmm0                 \n"
      "movq        %%xmm0,(%1)                   \n"
      "movdqa      %%xmm0,%%xmm4                 \n"
      "palignr     $0x8,%%xmm4,%%xmm4            \n"
      "movq        %%xmm4,(%1,%4)                \n"
      "lea         (%1,%4,2),%1                  \n"
      "punpckldq   %%xmm6,%%xmm2                 \n"
      "movdqa      %%xmm2,%%xmm6                 \n"
      "movq        %%xmm2,(%1)                   \n"
      "palignr     $0x8,%%xmm6,%%xmm6            \n"
      "punpckldq   %%xmm5,%%xmm1                 \n"
      "movq        %%xmm6,(%1,%4)                \n"
      "lea         (%1,%4,2),%1                  \n"
      "movdqa      %%xmm1,%%xmm5                 \n"
      "movq        %%xmm1,(%1)                   \n"
      "palignr     $0x8,%%xmm5,%%xmm5            \n"
      "movq        %%xmm5,(%1,%4)                \n"
      "lea         (%1,%4,2),%1                  \n"
      "punpckldq   %%xmm7,%%xmm3                 \n"
      "movq        %%xmm3,(%1)                   \n"
      "movdqa      %%xmm3,%%xmm7                 \n"
      "palignr     $0x8,%%xmm7,%%xmm7            \n"
      "sub         $0x8,%2                       \n"
      "movq        %%xmm7,(%1,%4)                \n"
      "lea         (%1,%4,2),%1                  \n"
      "jg          1b                            \n"
      : "+r"(src),                    // %0
        "+r"(dst),                    // %1
        "+r"(width)                   // %2
      : "r"((intptr_t)(src_stride)),  // %3
        "r"((intptr_t)(dst_stride))   // %4
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}
#endif  // defined(HAS_TRANSPOSEWX8_SSSE3)

// Transpose 16x8. 64 bit
#if defined(HAS_TRANSPOSEWX8_FAST_SSSE3)
void TransposeWx8_Fast_SSSE3(const uint8_t* src,
                             int src_stride,
                             uint8_t* dst,
                             int dst_stride,
                             int width) {
  asm volatile(
      // Read in the data from the source pointer.
      // First round of bit swap.
      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      (%0,%3),%%xmm1                \n"
      "lea         (%0,%3,2),%0                  \n"
      "movdqa      %%xmm0,%%xmm8                 \n"
      "punpcklbw   %%xmm1,%%xmm0                 \n"
      "punpckhbw   %%xmm1,%%xmm8                 \n"
      "movdqu      (%0),%%xmm2                   \n"
      "movdqa      %%xmm0,%%xmm1                 \n"
      "movdqa      %%xmm8,%%xmm9                 \n"
      "palignr     $0x8,%%xmm1,%%xmm1            \n"
      "palignr     $0x8,%%xmm9,%%xmm9            \n"
      "movdqu      (%0,%3),%%xmm3                \n"
      "lea         (%0,%3,2),%0                  \n"
      "movdqa      %%xmm2,%%xmm10                \n"
      "punpcklbw   %%xmm3,%%xmm2                 \n"
      "punpckhbw   %%xmm3,%%xmm10                \n"
      "movdqa      %%xmm2,%%xmm3                 \n"
      "movdqa      %%xmm10,%%xmm11               \n"
      "movdqu      (%0),%%xmm4                   \n"
      "palignr     $0x8,%%xmm3,%%xmm3            \n"
      "palignr     $0x8,%%xmm11,%%xmm11          \n"
      "movdqu      (%0,%3),%%xmm5                \n"
      "lea         (%0,%3,2),%0                  \n"
      "movdqa      %%xmm4,%%xmm12                \n"
      "punpcklbw   %%xmm5,%%xmm4                 \n"
      "punpckhbw   %%xmm5,%%xmm12                \n"
      "movdqa      %%xmm4,%%xmm5                 \n"
      "movdqa      %%xmm12,%%xmm13               \n"
      "movdqu      (%0),%%xmm6                   \n"
      "palignr     $0x8,%%xmm5,%%xmm5            \n"
      "palignr     $0x8,%%xmm13,%%xmm13          \n"
      "movdqu      (%0,%3),%%xmm7                \n"
      "lea         (%0,%3,2),%0                  \n"
      "movdqa      %%xmm6,%%xmm14                \n"
      "punpcklbw   %%xmm7,%%xmm6                 \n"
      "punpckhbw   %%xmm7,%%xmm14                \n"
      "neg         %3                            \n"
      "movdqa      %%xmm6,%%xmm7                 \n"
      "movdqa      %%xmm14,%%xmm15               \n"
      "lea         0x10(%0,%3,8),%0              \n"
      "palignr     $0x8,%%xmm7,%%xmm7            \n"
      "palignr     $0x8,%%xmm15,%%xmm15          \n"
      "neg         %3                            \n"
      // Second round of bit swap.
      "punpcklwd   %%xmm2,%%xmm0                 \n"
      "punpcklwd   %%xmm3,%%xmm1                 \n"
      "movdqa      %%xmm0,%%xmm2                 \n"
      "movdqa      %%xmm1,%%xmm3                 \n"
      "palignr     $0x8,%%xmm2,%%xmm2            \n"
      "palignr     $0x8,%%xmm3,%%xmm3            \n"
      "punpcklwd   %%xmm6,%%xmm4                 \n"
      "punpcklwd   %%xmm7,%%xmm5                 \n"
      "movdqa      %%xmm4,%%xmm6                 \n"
      "movdqa      %%xmm5,%%xmm7                 \n"
      "palignr     $0x8,%%xmm6,%%xmm6            \n"
      "palignr     $0x8,%%xmm7,%%xmm7            \n"
      "punpcklwd   %%xmm10,%%xmm8                \n"
      "punpcklwd   %%xmm11,%%xmm9                \n"
      "movdqa      %%xmm8,%%xmm10                \n"
      "movdqa      %%xmm9,%%xmm11                \n"
      "palignr     $0x8,%%xmm10,%%xmm10          \n"
      "palignr     $0x8,%%xmm11,%%xmm11          \n"
      "punpcklwd   %%xmm14,%%xmm12               \n"
      "punpcklwd   %%xmm15,%%xmm13               \n"
      "movdqa      %%xmm12,%%xmm14               \n"
      "movdqa      %%xmm13,%%xmm15               \n"
      "palignr     $0x8,%%xmm14,%%xmm14          \n"
      "palignr     $0x8,%%xmm15,%%xmm15          \n"
      // Third round of bit swap.
      // Write to the destination pointer.
      "punpckldq   %%xmm4,%%xmm0                 \n"
      "movq        %%xmm0,(%1)                   \n"
      "movdqa      %%xmm0,%%xmm4                 \n"
      "palignr     $0x8,%%xmm4,%%xmm4            \n"
      "movq        %%xmm4,(%1,%4)                \n"
      "lea         (%1,%4,2),%1                  \n"
      "punpckldq   %%xmm6,%%xmm2                 \n"
      "movdqa      %%xmm2,%%xmm6                 \n"
      "movq        %%xmm2,(%1)                   \n"
      "palignr     $0x8,%%xmm6,%%xmm6            \n"
      "punpckldq   %%xmm5,%%xmm1                 \n"
      "movq        %%xmm6,(%1,%4)                \n"
      "lea         (%1,%4,2),%1                  \n"
      "movdqa      %%xmm1,%%xmm5                 \n"
      "movq        %%xmm1,(%1)                   \n"
      "palignr     $0x8,%%xmm5,%%xmm5            \n"
      "movq        %%xmm5,(%1,%4)                \n"
      "lea         (%1,%4,2),%1                  \n"
      "punpckldq   %%xmm7,%%xmm3                 \n"
      "movq        %%xmm3,(%1)                   \n"
      "movdqa      %%xmm3,%%xmm7                 \n"
      "palignr     $0x8,%%xmm7,%%xmm7            \n"
      "movq        %%xmm7,(%1,%4)                \n"
      "lea         (%1,%4,2),%1                  \n"
      "punpckldq   %%xmm12,%%xmm8                \n"
      "movq        %%xmm8,(%1)                   \n"
      "movdqa      %%xmm8,%%xmm12                \n"
      "palignr     $0x8,%%xmm12,%%xmm12          \n"
      "movq        %%xmm12,(%1,%4)               \n"
      "lea         (%1,%4,2),%1                  \n"
      "punpckldq   %%xmm14,%%xmm10               \n"
      "movdqa      %%xmm10,%%xmm14               \n"
      "movq        %%xmm10,(%1)                  \n"
      "palignr     $0x8,%%xmm14,%%xmm14          \n"
      "punpckldq   %%xmm13,%%xmm9                \n"
      "movq        %%xmm14,(%1,%4)               \n"
      "lea         (%1,%4,2),%1                  \n"
      "movdqa      %%xmm9,%%xmm13                \n"
      "movq        %%xmm9,(%1)                   \n"
      "palignr     $0x8,%%xmm13,%%xmm13          \n"
      "movq        %%xmm13,(%1,%4)               \n"
      "lea         (%1,%4,2),%1                  \n"
      "punpckldq   %%xmm15,%%xmm11               \n"
      "movq        %%xmm11,(%1)                  \n"
      "movdqa      %%xmm11,%%xmm15               \n"
      "palignr     $0x8,%%xmm15,%%xmm15          \n"
      "sub         $0x10,%2                      \n"
      "movq        %%xmm15,(%1,%4)               \n"
      "lea         (%1,%4,2),%1                  \n"
      "jg          1b                            \n"
      : "+r"(src),                    // %0
        "+r"(dst),                    // %1
        "+r"(width)                   // %2
      : "r"((intptr_t)(src_stride)),  // %3
        "r"((intptr_t)(dst_stride))   // %4
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7", "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14",
        "xmm15");
}
#endif  // defined(HAS_TRANSPOSEWX8_FAST_SSSE3)

// Transpose UV 8x8.  64 bit.
#if defined(HAS_TRANSPOSEUVWX8_SSE2)
void TransposeUVWx8_SSE2(const uint8_t* src,
                         int src_stride,
                         uint8_t* dst_a,
                         int dst_stride_a,
                         uint8_t* dst_b,
                         int dst_stride_b,
                         int width) {
  asm volatile(
      // Read in the data from the source pointer.
      // First round of bit swap.
      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      (%0,%4),%%xmm1                \n"
      "lea         (%0,%4,2),%0                  \n"
      "movdqa      %%xmm0,%%xmm8                 \n"
      "punpcklbw   %%xmm1,%%xmm0                 \n"
      "punpckhbw   %%xmm1,%%xmm8                 \n"
      "movdqa      %%xmm8,%%xmm1                 \n"
      "movdqu      (%0),%%xmm2                   \n"
      "movdqu      (%0,%4),%%xmm3                \n"
      "lea         (%0,%4,2),%0                  \n"
      "movdqa      %%xmm2,%%xmm8                 \n"
      "punpcklbw   %%xmm3,%%xmm2                 \n"
      "punpckhbw   %%xmm3,%%xmm8                 \n"
      "movdqa      %%xmm8,%%xmm3                 \n"
      "movdqu      (%0),%%xmm4                   \n"
      "movdqu      (%0,%4),%%xmm5                \n"
      "lea         (%0,%4,2),%0                  \n"
      "movdqa      %%xmm4,%%xmm8                 \n"
      "punpcklbw   %%xmm5,%%xmm4                 \n"
      "punpckhbw   %%xmm5,%%xmm8                 \n"
      "movdqa      %%xmm8,%%xmm5                 \n"
      "movdqu      (%0),%%xmm6                   \n"
      "movdqu      (%0,%4),%%xmm7                \n"
      "lea         (%0,%4,2),%0                  \n"
      "movdqa      %%xmm6,%%xmm8                 \n"
      "punpcklbw   %%xmm7,%%xmm6                 \n"
      "neg         %4                            \n"
      "lea         0x10(%0,%4,8),%0              \n"
      "punpckhbw   %%xmm7,%%xmm8                 \n"
      "movdqa      %%xmm8,%%xmm7                 \n"
      "neg         %4                            \n"
      // Second round of bit swap.
      "movdqa      %%xmm0,%%xmm8                 \n"
      "movdqa      %%xmm1,%%xmm9                 \n"
      "punpckhwd   %%xmm2,%%xmm8                 \n"
      "punpckhwd   %%xmm3,%%xmm9                 \n"
      "punpcklwd   %%xmm2,%%xmm0                 \n"
      "punpcklwd   %%xmm3,%%xmm1                 \n"
      "movdqa      %%xmm8,%%xmm2                 \n"
      "movdqa      %%xmm9,%%xmm3                 \n"
      "movdqa      %%xmm4,%%xmm8                 \n"
      "movdqa      %%xmm5,%%xmm9                 \n"
      "punpckhwd   %%xmm6,%%xmm8                 \n"
      "punpckhwd   %%xmm7,%%xmm9                 \n"
      "punpcklwd   %%xmm6,%%xmm4                 \n"
      "punpcklwd   %%xmm7,%%xmm5                 \n"
      "movdqa      %%xmm8,%%xmm6                 \n"
      "movdqa      %%xmm9,%%xmm7                 \n"
      // Third round of bit swap.
      // Write to the destination pointer.
      "movdqa      %%xmm0,%%xmm8                 \n"
      "punpckldq   %%xmm4,%%xmm0                 \n"
      "movlpd      %%xmm0,(%1)                   \n"  // Write back U channel
      "movhpd      %%xmm0,(%2)                   \n"  // Write back V channel
      "punpckhdq   %%xmm4,%%xmm8                 \n"
      "movlpd      %%xmm8,(%1,%5)                \n"
      "lea         (%1,%5,2),%1                  \n"
      "movhpd      %%xmm8,(%2,%6)                \n"
      "lea         (%2,%6,2),%2                  \n"
      "movdqa      %%xmm2,%%xmm8                 \n"
      "punpckldq   %%xmm6,%%xmm2                 \n"
      "movlpd      %%xmm2,(%1)                   \n"
      "movhpd      %%xmm2,(%2)                   \n"
      "punpckhdq   %%xmm6,%%xmm8                 \n"
      "movlpd      %%xmm8,(%1,%5)                \n"
      "lea         (%1,%5,2),%1                  \n"
      "movhpd      %%xmm8,(%2,%6)                \n"
      "lea         (%2,%6,2),%2                  \n"
      "movdqa      %%xmm1,%%xmm8                 \n"
      "punpckldq   %%xmm5,%%xmm1                 \n"
      "movlpd      %%xmm1,(%1)                   \n"
      "movhpd      %%xmm1,(%2)                   \n"
      "punpckhdq   %%xmm5,%%xmm8                 \n"
      "movlpd      %%xmm8,(%1,%5)                \n"
      "lea         (%1,%5,2),%1                  \n"
      "movhpd      %%xmm8,(%2,%6)                \n"
      "lea         (%2,%6,2),%2                  \n"
      "movdqa      %%xmm3,%%xmm8                 \n"
      "punpckldq   %%xmm7,%%xmm3                 \n"
      "movlpd      %%xmm3,(%1)                   \n"
      "movhpd      %%xmm3,(%2)                   \n"
      "punpckhdq   %%xmm7,%%xmm8                 \n"
      "sub         $0x8,%3                       \n"
      "movlpd      %%xmm8,(%1,%5)                \n"
      "lea         (%1,%5,2),%1                  \n"
      "movhpd      %%xmm8,(%2,%6)                \n"
      "lea         (%2,%6,2),%2                  \n"
      "jg          1b                            \n"
      : "+r"(src),                      // %0
        "+r"(dst_a),                    // %1
        "+r"(dst_b),                    // %2
        "+r"(width)                     // %3
      : "r"((intptr_t)(src_stride)),    // %4
        "r"((intptr_t)(dst_stride_a)),  // %5
        "r"((intptr_t)(dst_stride_b))   // %6
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7", "xmm8", "xmm9");
}
#endif  // defined(HAS_TRANSPOSEUVWX8_SSE2)

#if defined(HAS_TRANSPOSE4X4_32_SSE2)
// 4 values, little endian view
// a b c d
// e f g h
// i j k l
// m n o p

// transpose 2x2
// a e b f   from row 0, 1
// i m j n   from row 2, 3
// c g d h   from row 0, 1
// k o l p   from row 2, 3

// transpose 4x4
// a e i m   from row 0, 1
// b f j n   from row 0, 1
// c g k o   from row 2, 3
// d h l p   from row 2, 3

// Transpose 32 bit values (ARGB)
void Transpose4x4_32_SSE2(const uint8_t* src,
                          int src_stride,
                          uint8_t* dst,
                          int dst_stride,
                          int width) {
  asm volatile(
      // Main loop transpose 4x4.  Read a column, write a row.
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"  // a b c d
      "movdqu      (%0,%3),%%xmm1                \n"  // e f g h
      "lea         (%0,%3,2),%0                  \n"  // src += stride * 2
      "movdqu      (%0),%%xmm2                   \n"  // i j k l
      "movdqu      (%0,%3),%%xmm3                \n"  // m n o p
      "lea         (%0,%3,2),%0                  \n"  // src += stride * 2

      // Transpose 2x2
      "movdqa      %%xmm0,%%xmm4                 \n"
      "movdqa      %%xmm2,%%xmm5                 \n"
      "movdqa      %%xmm0,%%xmm6                 \n"
      "movdqa      %%xmm2,%%xmm7                 \n"
      "punpckldq   %%xmm1,%%xmm4                 \n"  // a e b f   from row 0, 1
      "punpckldq   %%xmm3,%%xmm5                 \n"  // i m j n   from row 2, 3
      "punpckhdq   %%xmm1,%%xmm6                 \n"  // c g d h   from row 0, 1
      "punpckhdq   %%xmm3,%%xmm7                 \n"  // k o l p   from row 2, 3

      // Transpose 4x4
      "movdqa      %%xmm4,%%xmm0                 \n"
      "movdqa      %%xmm4,%%xmm1                 \n"
      "movdqa      %%xmm6,%%xmm2                 \n"
      "movdqa      %%xmm6,%%xmm3                 \n"
      "punpcklqdq  %%xmm5,%%xmm0                 \n"  // a e i m   from row 0, 1
      "punpckhqdq  %%xmm5,%%xmm1                 \n"  // b f j n   from row 0, 1
      "punpcklqdq  %%xmm7,%%xmm2                 \n"  // c g k o   from row 2, 3
      "punpckhqdq  %%xmm7,%%xmm3                 \n"  // d h l p   from row 2, 3

      "movdqu      %%xmm0,(%1)                   \n"
      "lea         16(%1,%4),%1                  \n"  // dst += stride + 16
      "movdqu      %%xmm1,-16(%1)                \n"
      "movdqu      %%xmm2,-16(%1,%4)             \n"
      "movdqu      %%xmm3,-16(%1,%4,2)           \n"
      "sub         %4,%1                         \n"
      "sub         $0x4,%2                       \n"
      "jg          1b                            \n"
      : "+r"(src),                     // %0
        "+r"(dst),                     // %1
        "+rm"(width)                   // %2
      : "r"((ptrdiff_t)(src_stride)),  // %3
        "r"((ptrdiff_t)(dst_stride))   // %4
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}
#endif  // defined(HAS_TRANSPOSE4X4_32_SSE2)

#if defined(HAS_TRANSPOSE4X4_32_AVX2)

// Transpose 32 bit values (ARGB)
void Transpose4x4_32_AVX2(const uint8_t* src,
                          int src_stride,
                          uint8_t* dst,
                          int dst_stride,
                          int width) {
  asm volatile(
      // Main loop transpose 2 blocks of 4x4.  Read a column, write a row.
      "1:          \n"
      "vmovdqu     (%0),%%xmm0                   \n"  // a b c d
      "vmovdqu     (%0,%3),%%xmm1                \n"  // e f g h
      "lea         (%0,%3,2),%0                  \n"  // src += stride * 2
      "vmovdqu     (%0),%%xmm2                   \n"  // i j k l
      "vmovdqu     (%0,%3),%%xmm3                \n"  // m n o p
      "lea         (%0,%3,2),%0                  \n"  // src += stride * 2

      "vinserti128 $1,(%0),%%ymm0,%%ymm0         \n"  // a b c d
      "vinserti128 $1,(%0,%3),%%ymm1,%%ymm1      \n"  // e f g h
      "lea         (%0,%3,2),%0                  \n"  // src += stride * 2
      "vinserti128 $1,(%0),%%ymm2,%%ymm2         \n"  // i j k l
      "vinserti128 $1,(%0,%3),%%ymm3,%%ymm3      \n"  // m n o p
      "lea         (%0,%3,2),%0                  \n"  // src += stride * 2

      // Transpose 2x2
      "vpunpckldq  %%ymm1,%%ymm0,%%ymm4          \n"  // a e b f   from row 0, 1
      "vpunpckldq  %%ymm3,%%ymm2,%%ymm5          \n"  // i m j n   from row 2, 3
      "vpunpckhdq  %%ymm1,%%ymm0,%%ymm6          \n"  // c g d h   from row 0, 1
      "vpunpckhdq  %%ymm3,%%ymm2,%%ymm7          \n"  // k o l p   from row 2, 3

      // Transpose 4x4
      "vpunpcklqdq %%ymm5,%%ymm4,%%ymm0          \n"  // a e i m   from row 0, 1
      "vpunpckhqdq %%ymm5,%%ymm4,%%ymm1          \n"  // b f j n   from row 0, 1
      "vpunpcklqdq %%ymm7,%%ymm6,%%ymm2          \n"  // c g k o   from row 2, 3
      "vpunpckhqdq %%ymm7,%%ymm6,%%ymm3          \n"  // d h l p   from row 2, 3

      "vmovdqu     %%ymm0,(%1)                   \n"
      "lea         32(%1,%4),%1                  \n"  // dst += stride + 32
      "vmovdqu     %%ymm1,-32(%1)                \n"
      "vmovdqu     %%ymm2,-32(%1,%4)             \n"
      "vmovdqu     %%ymm3,-32(%1,%4,2)           \n"
      "sub         %4,%1                         \n"
      "sub         $0x8,%2                       \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src),                     // %0
        "+r"(dst),                     // %1
        "+rm"(width)                   // %2
      : "r"((ptrdiff_t)(src_stride)),  // %3
        "r"((ptrdiff_t)(dst_stride))   // %4
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}
#endif  // defined(HAS_TRANSPOSE4X4_32_AVX2)

#endif  // defined(__x86_64__) || defined(__i386__)

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
