/*
 *  Copyright 2011 The LibYuv Project Authors. All rights reserved.
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

// This module is for GCC Neon.
#if !defined(LIBYUV_DISABLE_NEON) && defined(__ARM_NEON__) && \
    !defined(__aarch64__)

// NEON downscalers with interpolation.
// Provided by Fritz Koenig

// Read 32x1 throw away even pixels, and write 16x1.
void ScaleRowDown2_NEON(const uint8* src_ptr, ptrdiff_t src_stride,
                        uint8* dst, int dst_width) {
  asm volatile (
  "1:                                          \n"
    // load even pixels into q0, odd into q1
    MEMACCESS(0)
    "vld2.8     {q0, q1}, [%0]!                \n"
    "subs       %2, %2, #16                    \n"  // 16 processed per loop
    MEMACCESS(1)
    "vst1.8     {q1}, [%1]!                    \n"  // store odd pixels
    "bgt        1b                             \n"
  : "+r"(src_ptr),          // %0
    "+r"(dst),              // %1
    "+r"(dst_width)         // %2
  :
  : "q0", "q1"              // Clobber List
  );
}

// Read 32x1 average down and write 16x1.
void ScaleRowDown2Linear_NEON(const uint8* src_ptr, ptrdiff_t src_stride,
                           uint8* dst, int dst_width) {
  asm volatile (
  "1:                                          \n"
    MEMACCESS(0)
    "vld1.8     {q0, q1}, [%0]!                \n"  // load pixels and post inc
    "subs       %2, %2, #16                    \n"  // 16 processed per loop
    "vpaddl.u8  q0, q0                         \n"  // add adjacent
    "vpaddl.u8  q1, q1                         \n"
    "vrshrn.u16 d0, q0, #1                     \n"  // downshift, round and pack
    "vrshrn.u16 d1, q1, #1                     \n"
    MEMACCESS(1)
    "vst1.8     {q0}, [%1]!                    \n"
    "bgt        1b                             \n"
  : "+r"(src_ptr),          // %0
    "+r"(dst),              // %1
    "+r"(dst_width)         // %2
  :
  : "q0", "q1"     // Clobber List
  );
}

// Read 32x2 average down and write 16x1.
void ScaleRowDown2Box_NEON(const uint8* src_ptr, ptrdiff_t src_stride,
                           uint8* dst, int dst_width) {
  asm volatile (
    // change the stride to row 2 pointer
    "add        %1, %0                         \n"
  "1:                                          \n"
    MEMACCESS(0)
    "vld1.8     {q0, q1}, [%0]!                \n"  // load row 1 and post inc
    MEMACCESS(1)
    "vld1.8     {q2, q3}, [%1]!                \n"  // load row 2 and post inc
    "subs       %3, %3, #16                    \n"  // 16 processed per loop
    "vpaddl.u8  q0, q0                         \n"  // row 1 add adjacent
    "vpaddl.u8  q1, q1                         \n"
    "vpadal.u8  q0, q2                         \n"  // row 2 add adjacent + row1
    "vpadal.u8  q1, q3                         \n"
    "vrshrn.u16 d0, q0, #2                     \n"  // downshift, round and pack
    "vrshrn.u16 d1, q1, #2                     \n"
    MEMACCESS(2)
    "vst1.8     {q0}, [%2]!                    \n"
    "bgt        1b                             \n"
  : "+r"(src_ptr),          // %0
    "+r"(src_stride),       // %1
    "+r"(dst),              // %2
    "+r"(dst_width)         // %3
  :
  : "q0", "q1", "q2", "q3"     // Clobber List
  );
}

void ScaleRowDown4_NEON(const uint8* src_ptr, ptrdiff_t src_stride,
                        uint8* dst_ptr, int dst_width) {
  asm volatile (
  "1:                                          \n"
    MEMACCESS(0)
    "vld4.8     {d0, d1, d2, d3}, [%0]!        \n" // src line 0
    "subs       %2, %2, #8                     \n" // 8 processed per loop
    MEMACCESS(1)
    "vst1.8     {d2}, [%1]!                    \n"
    "bgt        1b                             \n"
  : "+r"(src_ptr),          // %0
    "+r"(dst_ptr),          // %1
    "+r"(dst_width)         // %2
  :
  : "q0", "q1", "memory", "cc"
  );
}

void ScaleRowDown4Box_NEON(const uint8* src_ptr, ptrdiff_t src_stride,
                           uint8* dst_ptr, int dst_width) {
  const uint8* src_ptr1 = src_ptr + src_stride;
  const uint8* src_ptr2 = src_ptr + src_stride * 2;
  const uint8* src_ptr3 = src_ptr + src_stride * 3;
asm volatile (
  "1:                                          \n"
    MEMACCESS(0)
    "vld1.8     {q0}, [%0]!                    \n"   // load up 16x4
    MEMACCESS(3)
    "vld1.8     {q1}, [%3]!                    \n"
    MEMACCESS(4)
    "vld1.8     {q2}, [%4]!                    \n"
    MEMACCESS(5)
    "vld1.8     {q3}, [%5]!                    \n"
    "subs       %2, %2, #4                     \n"
    "vpaddl.u8  q0, q0                         \n"
    "vpadal.u8  q0, q1                         \n"
    "vpadal.u8  q0, q2                         \n"
    "vpadal.u8  q0, q3                         \n"
    "vpaddl.u16 q0, q0                         \n"
    "vrshrn.u32 d0, q0, #4                     \n"   // divide by 16 w/rounding
    "vmovn.u16  d0, q0                         \n"
    MEMACCESS(1)
    "vst1.32    {d0[0]}, [%1]!                 \n"
    "bgt        1b                             \n"
  : "+r"(src_ptr),   // %0
    "+r"(dst_ptr),   // %1
    "+r"(dst_width), // %2
    "+r"(src_ptr1),  // %3
    "+r"(src_ptr2),  // %4
    "+r"(src_ptr3)   // %5
  :
  : "q0", "q1", "q2", "q3", "memory", "cc"
  );
}

// Down scale from 4 to 3 pixels. Use the neon multilane read/write
// to load up the every 4th pixel into a 4 different registers.
// Point samples 32 pixels to 24 pixels.
void ScaleRowDown34_NEON(const uint8* src_ptr,
                         ptrdiff_t src_stride,
                         uint8* dst_ptr, int dst_width) {
  asm volatile (
  "1:                                          \n"
    MEMACCESS(0)
    "vld4.8     {d0, d1, d2, d3}, [%0]!      \n" // src line 0
    "subs       %2, %2, #24                  \n"
    "vmov       d2, d3                       \n" // order d0, d1, d2
    MEMACCESS(1)
    "vst3.8     {d0, d1, d2}, [%1]!          \n"
    "bgt        1b                           \n"
  : "+r"(src_ptr),          // %0
    "+r"(dst_ptr),          // %1
    "+r"(dst_width)         // %2
  :
  : "d0", "d1", "d2", "d3", "memory", "cc"
  );
}

void ScaleRowDown34_0_Box_NEON(const uint8* src_ptr,
                               ptrdiff_t src_stride,
                               uint8* dst_ptr, int dst_width) {
  asm volatile (
    "vmov.u8    d24, #3                        \n"
    "add        %3, %0                         \n"
  "1:                                          \n"
    MEMACCESS(0)
    "vld4.8       {d0, d1, d2, d3}, [%0]!      \n" // src line 0
    MEMACCESS(3)
    "vld4.8       {d4, d5, d6, d7}, [%3]!      \n" // src line 1
    "subs         %2, %2, #24                  \n"

    // filter src line 0 with src line 1
    // expand chars to shorts to allow for room
    // when adding lines together
    "vmovl.u8     q8, d4                       \n"
    "vmovl.u8     q9, d5                       \n"
    "vmovl.u8     q10, d6                      \n"
    "vmovl.u8     q11, d7                      \n"

    // 3 * line_0 + line_1
    "vmlal.u8     q8, d0, d24                  \n"
    "vmlal.u8     q9, d1, d24                  \n"
    "vmlal.u8     q10, d2, d24                 \n"
    "vmlal.u8     q11, d3, d24                 \n"

    // (3 * line_0 + line_1) >> 2
    "vqrshrn.u16  d0, q8, #2                   \n"
    "vqrshrn.u16  d1, q9, #2                   \n"
    "vqrshrn.u16  d2, q10, #2                  \n"
    "vqrshrn.u16  d3, q11, #2                  \n"

    // a0 = (src[0] * 3 + s[1] * 1) >> 2
    "vmovl.u8     q8, d1                       \n"
    "vmlal.u8     q8, d0, d24                  \n"
    "vqrshrn.u16  d0, q8, #2                   \n"

    // a1 = (src[1] * 1 + s[2] * 1) >> 1
    "vrhadd.u8    d1, d1, d2                   \n"

    // a2 = (src[2] * 1 + s[3] * 3) >> 2
    "vmovl.u8     q8, d2                       \n"
    "vmlal.u8     q8, d3, d24                  \n"
    "vqrshrn.u16  d2, q8, #2                   \n"

    MEMACCESS(1)
    "vst3.8       {d0, d1, d2}, [%1]!          \n"

    "bgt          1b                           \n"
  : "+r"(src_ptr),          // %0
    "+r"(dst_ptr),          // %1
    "+r"(dst_width),        // %2
    "+r"(src_stride)        // %3
  :
  : "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "d24", "memory", "cc"
  );
}

void ScaleRowDown34_1_Box_NEON(const uint8* src_ptr,
                               ptrdiff_t src_stride,
                               uint8* dst_ptr, int dst_width) {
  asm volatile (
    "vmov.u8    d24, #3                        \n"
    "add        %3, %0                         \n"
  "1:                                          \n"
    MEMACCESS(0)
    "vld4.8       {d0, d1, d2, d3}, [%0]!      \n" // src line 0
    MEMACCESS(3)
    "vld4.8       {d4, d5, d6, d7}, [%3]!      \n" // src line 1
    "subs         %2, %2, #24                  \n"
    // average src line 0 with src line 1
    "vrhadd.u8    q0, q0, q2                   \n"
    "vrhadd.u8    q1, q1, q3                   \n"

    // a0 = (src[0] * 3 + s[1] * 1) >> 2
    "vmovl.u8     q3, d1                       \n"
    "vmlal.u8     q3, d0, d24                  \n"
    "vqrshrn.u16  d0, q3, #2                   \n"

    // a1 = (src[1] * 1 + s[2] * 1) >> 1
    "vrhadd.u8    d1, d1, d2                   \n"

    // a2 = (src[2] * 1 + s[3] * 3) >> 2
    "vmovl.u8     q3, d2                       \n"
    "vmlal.u8     q3, d3, d24                  \n"
    "vqrshrn.u16  d2, q3, #2                   \n"

    MEMACCESS(1)
    "vst3.8       {d0, d1, d2}, [%1]!          \n"
    "bgt          1b                           \n"
  : "+r"(src_ptr),          // %0
    "+r"(dst_ptr),          // %1
    "+r"(dst_width),        // %2
    "+r"(src_stride)        // %3
  :
  : "r4", "q0", "q1", "q2", "q3", "d24", "memory", "cc"
  );
}

#define HAS_SCALEROWDOWN38_NEON
static uvec8 kShuf38 =
  { 0, 3, 6, 8, 11, 14, 16, 19, 22, 24, 27, 30, 0, 0, 0, 0 };
static uvec8 kShuf38_2 =
  { 0, 8, 16, 2, 10, 17, 4, 12, 18, 6, 14, 19, 0, 0, 0, 0 };
static vec16 kMult38_Div6 =
  { 65536 / 12, 65536 / 12, 65536 / 12, 65536 / 12,
    65536 / 12, 65536 / 12, 65536 / 12, 65536 / 12 };
static vec16 kMult38_Div9 =
  { 65536 / 18, 65536 / 18, 65536 / 18, 65536 / 18,
    65536 / 18, 65536 / 18, 65536 / 18, 65536 / 18 };

// 32 -> 12
void ScaleRowDown38_NEON(const uint8* src_ptr,
                         ptrdiff_t src_stride,
                         uint8* dst_ptr, int dst_width) {
  asm volatile (
    MEMACCESS(3)
    "vld1.8     {q3}, [%3]                     \n"
  "1:                                          \n"
    MEMACCESS(0)
    "vld1.8     {d0, d1, d2, d3}, [%0]!        \n"
    "subs       %2, %2, #12                    \n"
    "vtbl.u8    d4, {d0, d1, d2, d3}, d6       \n"
    "vtbl.u8    d5, {d0, d1, d2, d3}, d7       \n"
    MEMACCESS(1)
    "vst1.8     {d4}, [%1]!                    \n"
    MEMACCESS(1)
    "vst1.32    {d5[0]}, [%1]!                 \n"
    "bgt        1b                             \n"
  : "+r"(src_ptr),          // %0
    "+r"(dst_ptr),          // %1
    "+r"(dst_width)         // %2
  : "r"(&kShuf38)           // %3
  : "d0", "d1", "d2", "d3", "d4", "d5", "memory", "cc"
  );
}

// 32x3 -> 12x1
void OMITFP ScaleRowDown38_3_Box_NEON(const uint8* src_ptr,
                                      ptrdiff_t src_stride,
                                      uint8* dst_ptr, int dst_width) {
  const uint8* src_ptr1 = src_ptr + src_stride * 2;

  asm volatile (
    MEMACCESS(5)
    "vld1.16    {q13}, [%5]                    \n"
    MEMACCESS(6)
    "vld1.8     {q14}, [%6]                    \n"
    MEMACCESS(7)
    "vld1.8     {q15}, [%7]                    \n"
    "add        %3, %0                         \n"
  "1:                                          \n"

    // d0 = 00 40 01 41 02 42 03 43
    // d1 = 10 50 11 51 12 52 13 53
    // d2 = 20 60 21 61 22 62 23 63
    // d3 = 30 70 31 71 32 72 33 73
    MEMACCESS(0)
    "vld4.8       {d0, d1, d2, d3}, [%0]!      \n"
    MEMACCESS(3)
    "vld4.8       {d4, d5, d6, d7}, [%3]!      \n"
    MEMACCESS(4)
    "vld4.8       {d16, d17, d18, d19}, [%4]!  \n"
    "subs         %2, %2, #12                  \n"

    // Shuffle the input data around to get align the data
    //  so adjacent data can be added. 0,1 - 2,3 - 4,5 - 6,7
    // d0 = 00 10 01 11 02 12 03 13
    // d1 = 40 50 41 51 42 52 43 53
    "vtrn.u8      d0, d1                       \n"
    "vtrn.u8      d4, d5                       \n"
    "vtrn.u8      d16, d17                     \n"

    // d2 = 20 30 21 31 22 32 23 33
    // d3 = 60 70 61 71 62 72 63 73
    "vtrn.u8      d2, d3                       \n"
    "vtrn.u8      d6, d7                       \n"
    "vtrn.u8      d18, d19                     \n"

    // d0 = 00+10 01+11 02+12 03+13
    // d2 = 40+50 41+51 42+52 43+53
    "vpaddl.u8    q0, q0                       \n"
    "vpaddl.u8    q2, q2                       \n"
    "vpaddl.u8    q8, q8                       \n"

    // d3 = 60+70 61+71 62+72 63+73
    "vpaddl.u8    d3, d3                       \n"
    "vpaddl.u8    d7, d7                       \n"
    "vpaddl.u8    d19, d19                     \n"

    // combine source lines
    "vadd.u16     q0, q2                       \n"
    "vadd.u16     q0, q8                       \n"
    "vadd.u16     d4, d3, d7                   \n"
    "vadd.u16     d4, d19                      \n"

    // dst_ptr[3] = (s[6 + st * 0] + s[7 + st * 0]
    //             + s[6 + st * 1] + s[7 + st * 1]
    //             + s[6 + st * 2] + s[7 + st * 2]) / 6
    "vqrdmulh.s16 q2, q2, q13                  \n"
    "vmovn.u16    d4, q2                       \n"

    // Shuffle 2,3 reg around so that 2 can be added to the
    //  0,1 reg and 3 can be added to the 4,5 reg. This
    //  requires expanding from u8 to u16 as the 0,1 and 4,5
    //  registers are already expanded. Then do transposes
    //  to get aligned.
    // q2 = xx 20 xx 30 xx 21 xx 31 xx 22 xx 32 xx 23 xx 33
    "vmovl.u8     q1, d2                       \n"
    "vmovl.u8     q3, d6                       \n"
    "vmovl.u8     q9, d18                      \n"

    // combine source lines
    "vadd.u16     q1, q3                       \n"
    "vadd.u16     q1, q9                       \n"

    // d4 = xx 20 xx 30 xx 22 xx 32
    // d5 = xx 21 xx 31 xx 23 xx 33
    "vtrn.u32     d2, d3                       \n"

    // d4 = xx 20 xx 21 xx 22 xx 23
    // d5 = xx 30 xx 31 xx 32 xx 33
    "vtrn.u16     d2, d3                       \n"

    // 0+1+2, 3+4+5
    "vadd.u16     q0, q1                       \n"

    // Need to divide, but can't downshift as the the value
    //  isn't a power of 2. So multiply by 65536 / n
    //  and take the upper 16 bits.
    "vqrdmulh.s16 q0, q0, q15                  \n"

    // Align for table lookup, vtbl requires registers to
    //  be adjacent
    "vmov.u8      d2, d4                       \n"

    "vtbl.u8      d3, {d0, d1, d2}, d28        \n"
    "vtbl.u8      d4, {d0, d1, d2}, d29        \n"

    MEMACCESS(1)
    "vst1.8       {d3}, [%1]!                  \n"
    MEMACCESS(1)
    "vst1.32      {d4[0]}, [%1]!               \n"
    "bgt          1b                           \n"
  : "+r"(src_ptr),          // %0
    "+r"(dst_ptr),          // %1
    "+r"(dst_width),        // %2
    "+r"(src_stride),       // %3
    "+r"(src_ptr1)          // %4
  : "r"(&kMult38_Div6),     // %5
    "r"(&kShuf38_2),        // %6
    "r"(&kMult38_Div9)      // %7
  : "q0", "q1", "q2", "q3", "q8", "q9", "q13", "q14", "q15", "memory", "cc"
  );
}

// 32x2 -> 12x1
void ScaleRowDown38_2_Box_NEON(const uint8* src_ptr,
                               ptrdiff_t src_stride,
                               uint8* dst_ptr, int dst_width) {
  asm volatile (
    MEMACCESS(4)
    "vld1.16    {q13}, [%4]                    \n"
    MEMACCESS(5)
    "vld1.8     {q14}, [%5]                    \n"
    "add        %3, %0                         \n"
  "1:                                          \n"

    // d0 = 00 40 01 41 02 42 03 43
    // d1 = 10 50 11 51 12 52 13 53
    // d2 = 20 60 21 61 22 62 23 63
    // d3 = 30 70 31 71 32 72 33 73
    MEMACCESS(0)
    "vld4.8       {d0, d1, d2, d3}, [%0]!      \n"
    MEMACCESS(3)
    "vld4.8       {d4, d5, d6, d7}, [%3]!      \n"
    "subs         %2, %2, #12                  \n"

    // Shuffle the input data around to get align the data
    //  so adjacent data can be added. 0,1 - 2,3 - 4,5 - 6,7
    // d0 = 00 10 01 11 02 12 03 13
    // d1 = 40 50 41 51 42 52 43 53
    "vtrn.u8      d0, d1                       \n"
    "vtrn.u8      d4, d5                       \n"

    // d2 = 20 30 21 31 22 32 23 33
    // d3 = 60 70 61 71 62 72 63 73
    "vtrn.u8      d2, d3                       \n"
    "vtrn.u8      d6, d7                       \n"

    // d0 = 00+10 01+11 02+12 03+13
    // d2 = 40+50 41+51 42+52 43+53
    "vpaddl.u8    q0, q0                       \n"
    "vpaddl.u8    q2, q2                       \n"

    // d3 = 60+70 61+71 62+72 63+73
    "vpaddl.u8    d3, d3                       \n"
    "vpaddl.u8    d7, d7                       \n"

    // combine source lines
    "vadd.u16     q0, q2                       \n"
    "vadd.u16     d4, d3, d7                   \n"

    // dst_ptr[3] = (s[6] + s[7] + s[6+st] + s[7+st]) / 4
    "vqrshrn.u16  d4, q2, #2                   \n"

    // Shuffle 2,3 reg around so that 2 can be added to the
    //  0,1 reg and 3 can be added to the 4,5 reg. This
    //  requires expanding from u8 to u16 as the 0,1 and 4,5
    //  registers are already expanded. Then do transposes
    //  to get aligned.
    // q2 = xx 20 xx 30 xx 21 xx 31 xx 22 xx 32 xx 23 xx 33
    "vmovl.u8     q1, d2                       \n"
    "vmovl.u8     q3, d6                       \n"

    // combine source lines
    "vadd.u16     q1, q3                       \n"

    // d4 = xx 20 xx 30 xx 22 xx 32
    // d5 = xx 21 xx 31 xx 23 xx 33
    "vtrn.u32     d2, d3                       \n"

    // d4 = xx 20 xx 21 xx 22 xx 23
    // d5 = xx 30 xx 31 xx 32 xx 33
    "vtrn.u16     d2, d3                       \n"

    // 0+1+2, 3+4+5
    "vadd.u16     q0, q1                       \n"

    // Need to divide, but can't downshift as the the value
    //  isn't a power of 2. So multiply by 65536 / n
    //  and take the upper 16 bits.
    "vqrdmulh.s16 q0, q0, q13                  \n"

    // Align for table lookup, vtbl requires registers to
    //  be adjacent
    "vmov.u8      d2, d4                       \n"

    "vtbl.u8      d3, {d0, d1, d2}, d28        \n"
    "vtbl.u8      d4, {d0, d1, d2}, d29        \n"

    MEMACCESS(1)
    "vst1.8       {d3}, [%1]!                  \n"
    MEMACCESS(1)
    "vst1.32      {d4[0]}, [%1]!               \n"
    "bgt          1b                           \n"
  : "+r"(src_ptr),       // %0
    "+r"(dst_ptr),       // %1
    "+r"(dst_width),     // %2
    "+r"(src_stride)     // %3
  : "r"(&kMult38_Div6),  // %4
    "r"(&kShuf38_2)      // %5
  : "q0", "q1", "q2", "q3", "q13", "q14", "memory", "cc"
  );
}

void ScaleAddRows_NEON(const uint8* src_ptr, ptrdiff_t src_stride,
                    uint16* dst_ptr, int src_width, int src_height) {
  const uint8* src_tmp;
  asm volatile (
  "1:                                          \n"
    "mov       %0, %1                          \n"
    "mov       r12, %5                         \n"
    "veor      q2, q2, q2                      \n"
    "veor      q3, q3, q3                      \n"
  "2:                                          \n"
    // load 16 pixels into q0
    MEMACCESS(0)
    "vld1.8     {q0}, [%0], %3                 \n"
    "vaddw.u8   q3, q3, d1                     \n"
    "vaddw.u8   q2, q2, d0                     \n"
    "subs       r12, r12, #1                   \n"
    "bgt        2b                             \n"
    MEMACCESS(2)
    "vst1.16    {q2, q3}, [%2]!                \n"  // store pixels
    "add        %1, %1, #16                    \n"
    "subs       %4, %4, #16                    \n"  // 16 processed per loop
    "bgt        1b                             \n"
  : "=&r"(src_tmp),    // %0
    "+r"(src_ptr),     // %1
    "+r"(dst_ptr),     // %2
    "+r"(src_stride),  // %3
    "+r"(src_width),   // %4
    "+r"(src_height)   // %5
  :
  : "memory", "cc", "r12", "q0", "q1", "q2", "q3"  // Clobber List
  );
}

// TODO(Yang Zhang): Investigate less load instructions for
// the x/dx stepping
#define LOAD2_DATA8_LANE(n)                                    \
    "lsr        %5, %3, #16                    \n"             \
    "add        %6, %1, %5                     \n"             \
    "add        %3, %3, %4                     \n"             \
    MEMACCESS(6)                                               \
    "vld2.8     {d6["#n"], d7["#n"]}, [%6]     \n"

// The NEON version mimics this formula (from row_common.cc):
// #define BLENDER(a, b, f) (uint8)((int)(a) +
//    ((((int)((f)) * ((int)(b) - (int)(a))) + 0x8000) >> 16))

void ScaleFilterCols_NEON(uint8* dst_ptr, const uint8* src_ptr,
                          int dst_width, int x, int dx) {
  int dx_offset[4] = {0, 1, 2, 3};
  int* tmp = dx_offset;
  const uint8* src_tmp = src_ptr;
  asm volatile (
    "vdup.32    q0, %3                         \n"  // x
    "vdup.32    q1, %4                         \n"  // dx
    "vld1.32    {q2}, [%5]                     \n"  // 0 1 2 3
    "vshl.i32   q3, q1, #2                     \n"  // 4 * dx
    "vmul.s32   q1, q1, q2                     \n"
    // x         , x + 1 * dx, x + 2 * dx, x + 3 * dx
    "vadd.s32   q1, q1, q0                     \n"
    // x + 4 * dx, x + 5 * dx, x + 6 * dx, x + 7 * dx
    "vadd.s32   q2, q1, q3                     \n"
    "vshl.i32   q0, q3, #1                     \n"  // 8 * dx
  "1:                                          \n"
    LOAD2_DATA8_LANE(0)
    LOAD2_DATA8_LANE(1)
    LOAD2_DATA8_LANE(2)
    LOAD2_DATA8_LANE(3)
    LOAD2_DATA8_LANE(4)
    LOAD2_DATA8_LANE(5)
    LOAD2_DATA8_LANE(6)
    LOAD2_DATA8_LANE(7)
    "vmov       q10, q1                        \n"
    "vmov       q11, q2                        \n"
    "vuzp.16    q10, q11                       \n"
    "vmovl.u8   q8, d6                         \n"
    "vmovl.u8   q9, d7                         \n"
    "vsubl.s16  q11, d18, d16                  \n"
    "vsubl.s16  q12, d19, d17                  \n"
    "vmovl.u16  q13, d20                       \n"
    "vmovl.u16  q10, d21                       \n"
    "vmul.s32   q11, q11, q13                  \n"
    "vmul.s32   q12, q12, q10                  \n"
    "vrshrn.s32  d18, q11, #16                 \n"
    "vrshrn.s32  d19, q12, #16                 \n"
    "vadd.s16   q8, q8, q9                     \n"
    "vmovn.s16  d6, q8                         \n"

    MEMACCESS(0)
    "vst1.8     {d6}, [%0]!                    \n"  // store pixels
    "vadd.s32   q1, q1, q0                     \n"
    "vadd.s32   q2, q2, q0                     \n"
    "subs       %2, %2, #8                     \n"  // 8 processed per loop
    "bgt        1b                             \n"
  : "+r"(dst_ptr),          // %0
    "+r"(src_ptr),          // %1
    "+r"(dst_width),        // %2
    "+r"(x),                // %3
    "+r"(dx),               // %4
    "+r"(tmp),              // %5
    "+r"(src_tmp)           // %6
  :
  : "memory", "cc", "q0", "q1", "q2", "q3",
    "q8", "q9", "q10", "q11", "q12", "q13"
  );
}

#undef LOAD2_DATA8_LANE

// 16x2 -> 16x1
void ScaleFilterRows_NEON(uint8* dst_ptr,
                          const uint8* src_ptr, ptrdiff_t src_stride,
                          int dst_width, int source_y_fraction) {
  asm volatile (
    "cmp          %4, #0                       \n"
    "beq          100f                         \n"
    "add          %2, %1                       \n"
    "cmp          %4, #64                      \n"
    "beq          75f                          \n"
    "cmp          %4, #128                     \n"
    "beq          50f                          \n"
    "cmp          %4, #192                     \n"
    "beq          25f                          \n"

    "vdup.8       d5, %4                       \n"
    "rsb          %4, #256                     \n"
    "vdup.8       d4, %4                       \n"
    // General purpose row blend.
  "1:                                          \n"
    MEMACCESS(1)
    "vld1.8       {q0}, [%1]!                  \n"
    MEMACCESS(2)
    "vld1.8       {q1}, [%2]!                  \n"
    "subs         %3, %3, #16                  \n"
    "vmull.u8     q13, d0, d4                  \n"
    "vmull.u8     q14, d1, d4                  \n"
    "vmlal.u8     q13, d2, d5                  \n"
    "vmlal.u8     q14, d3, d5                  \n"
    "vrshrn.u16   d0, q13, #8                  \n"
    "vrshrn.u16   d1, q14, #8                  \n"
    MEMACCESS(0)
    "vst1.8       {q0}, [%0]!                  \n"
    "bgt          1b                           \n"
    "b            99f                          \n"

    // Blend 25 / 75.
  "25:                                         \n"
    MEMACCESS(1)
    "vld1.8       {q0}, [%1]!                  \n"
    MEMACCESS(2)
    "vld1.8       {q1}, [%2]!                  \n"
    "subs         %3, %3, #16                  \n"
    "vrhadd.u8    q0, q1                       \n"
    "vrhadd.u8    q0, q1                       \n"
    MEMACCESS(0)
    "vst1.8       {q0}, [%0]!                  \n"
    "bgt          25b                          \n"
    "b            99f                          \n"

    // Blend 50 / 50.
  "50:                                         \n"
    MEMACCESS(1)
    "vld1.8       {q0}, [%1]!                  \n"
    MEMACCESS(2)
    "vld1.8       {q1}, [%2]!                  \n"
    "subs         %3, %3, #16                  \n"
    "vrhadd.u8    q0, q1                       \n"
    MEMACCESS(0)
    "vst1.8       {q0}, [%0]!                  \n"
    "bgt          50b                          \n"
    "b            99f                          \n"

    // Blend 75 / 25.
  "75:                                         \n"
    MEMACCESS(1)
    "vld1.8       {q1}, [%1]!                  \n"
    MEMACCESS(2)
    "vld1.8       {q0}, [%2]!                  \n"
    "subs         %3, %3, #16                  \n"
    "vrhadd.u8    q0, q1                       \n"
    "vrhadd.u8    q0, q1                       \n"
    MEMACCESS(0)
    "vst1.8       {q0}, [%0]!                  \n"
    "bgt          75b                          \n"
    "b            99f                          \n"

    // Blend 100 / 0 - Copy row unchanged.
  "100:                                        \n"
    MEMACCESS(1)
    "vld1.8       {q0}, [%1]!                  \n"
    "subs         %3, %3, #16                  \n"
    MEMACCESS(0)
    "vst1.8       {q0}, [%0]!                  \n"
    "bgt          100b                         \n"

  "99:                                         \n"
    MEMACCESS(0)
    "vst1.8       {d1[7]}, [%0]                \n"
  : "+r"(dst_ptr),          // %0
    "+r"(src_ptr),          // %1
    "+r"(src_stride),       // %2
    "+r"(dst_width),        // %3
    "+r"(source_y_fraction) // %4
  :
  : "q0", "q1", "d4", "d5", "q13", "q14", "memory", "cc"
  );
}

void ScaleARGBRowDown2_NEON(const uint8* src_ptr, ptrdiff_t src_stride,
                            uint8* dst, int dst_width) {
  asm volatile (
  "1:                                          \n"
    // load even pixels into q0, odd into q1
    MEMACCESS(0)
    "vld2.32    {q0, q1}, [%0]!                \n"
    MEMACCESS(0)
    "vld2.32    {q2, q3}, [%0]!                \n"
    "subs       %2, %2, #8                     \n"  // 8 processed per loop
    MEMACCESS(1)
    "vst1.8     {q1}, [%1]!                    \n"  // store odd pixels
    MEMACCESS(1)
    "vst1.8     {q3}, [%1]!                    \n"
    "bgt        1b                             \n"
  : "+r"(src_ptr),          // %0
    "+r"(dst),              // %1
    "+r"(dst_width)         // %2
  :
  : "memory", "cc", "q0", "q1", "q2", "q3"  // Clobber List
  );
}

void ScaleARGBRowDown2Linear_NEON(const uint8* src_argb, ptrdiff_t src_stride,
                                  uint8* dst_argb, int dst_width) {
  asm volatile (
  "1:                                          \n"
    MEMACCESS(0)
    "vld4.8     {d0, d2, d4, d6}, [%0]!        \n"  // load 8 ARGB pixels.
    MEMACCESS(0)
    "vld4.8     {d1, d3, d5, d7}, [%0]!        \n"  // load next 8 ARGB pixels.
    "subs       %2, %2, #8                     \n"  // 8 processed per loop
    "vpaddl.u8  q0, q0                         \n"  // B 16 bytes -> 8 shorts.
    "vpaddl.u8  q1, q1                         \n"  // G 16 bytes -> 8 shorts.
    "vpaddl.u8  q2, q2                         \n"  // R 16 bytes -> 8 shorts.
    "vpaddl.u8  q3, q3                         \n"  // A 16 bytes -> 8 shorts.
    "vrshrn.u16 d0, q0, #1                     \n"  // downshift, round and pack
    "vrshrn.u16 d1, q1, #1                     \n"
    "vrshrn.u16 d2, q2, #1                     \n"
    "vrshrn.u16 d3, q3, #1                     \n"
    MEMACCESS(1)
    "vst4.8     {d0, d1, d2, d3}, [%1]!        \n"
    "bgt       1b                              \n"
  : "+r"(src_argb),         // %0
    "+r"(dst_argb),         // %1
    "+r"(dst_width)         // %2
  :
  : "memory", "cc", "q0", "q1", "q2", "q3"     // Clobber List
  );
}

void ScaleARGBRowDown2Box_NEON(const uint8* src_ptr, ptrdiff_t src_stride,
                               uint8* dst, int dst_width) {
  asm volatile (
    // change the stride to row 2 pointer
    "add        %1, %1, %0                     \n"
  "1:                                          \n"
    MEMACCESS(0)
    "vld4.8     {d0, d2, d4, d6}, [%0]!        \n"  // load 8 ARGB pixels.
    MEMACCESS(0)
    "vld4.8     {d1, d3, d5, d7}, [%0]!        \n"  // load next 8 ARGB pixels.
    "subs       %3, %3, #8                     \n"  // 8 processed per loop.
    "vpaddl.u8  q0, q0                         \n"  // B 16 bytes -> 8 shorts.
    "vpaddl.u8  q1, q1                         \n"  // G 16 bytes -> 8 shorts.
    "vpaddl.u8  q2, q2                         \n"  // R 16 bytes -> 8 shorts.
    "vpaddl.u8  q3, q3                         \n"  // A 16 bytes -> 8 shorts.
    MEMACCESS(1)
    "vld4.8     {d16, d18, d20, d22}, [%1]!    \n"  // load 8 more ARGB pixels.
    MEMACCESS(1)
    "vld4.8     {d17, d19, d21, d23}, [%1]!    \n"  // load last 8 ARGB pixels.
    "vpadal.u8  q0, q8                         \n"  // B 16 bytes -> 8 shorts.
    "vpadal.u8  q1, q9                         \n"  // G 16 bytes -> 8 shorts.
    "vpadal.u8  q2, q10                        \n"  // R 16 bytes -> 8 shorts.
    "vpadal.u8  q3, q11                        \n"  // A 16 bytes -> 8 shorts.
    "vrshrn.u16 d0, q0, #2                     \n"  // downshift, round and pack
    "vrshrn.u16 d1, q1, #2                     \n"
    "vrshrn.u16 d2, q2, #2                     \n"
    "vrshrn.u16 d3, q3, #2                     \n"
    MEMACCESS(2)
    "vst4.8     {d0, d1, d2, d3}, [%2]!        \n"
    "bgt        1b                             \n"
  : "+r"(src_ptr),          // %0
    "+r"(src_stride),       // %1
    "+r"(dst),              // %2
    "+r"(dst_width)         // %3
  :
  : "memory", "cc", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11"
  );
}

// Reads 4 pixels at a time.
// Alignment requirement: src_argb 4 byte aligned.
void ScaleARGBRowDownEven_NEON(const uint8* src_argb,  ptrdiff_t src_stride,
                               int src_stepx, uint8* dst_argb, int dst_width) {
  asm volatile (
    "mov        r12, %3, lsl #2                \n"
  "1:                                          \n"
    MEMACCESS(0)
    "vld1.32    {d0[0]}, [%0], r12             \n"
    MEMACCESS(0)
    "vld1.32    {d0[1]}, [%0], r12             \n"
    MEMACCESS(0)
    "vld1.32    {d1[0]}, [%0], r12             \n"
    MEMACCESS(0)
    "vld1.32    {d1[1]}, [%0], r12             \n"
    "subs       %2, %2, #4                     \n"  // 4 pixels per loop.
    MEMACCESS(1)
    "vst1.8     {q0}, [%1]!                    \n"
    "bgt        1b                             \n"
  : "+r"(src_argb),    // %0
    "+r"(dst_argb),    // %1
    "+r"(dst_width)    // %2
  : "r"(src_stepx)     // %3
  : "memory", "cc", "r12", "q0"
  );
}

// Reads 4 pixels at a time.
// Alignment requirement: src_argb 4 byte aligned.
void ScaleARGBRowDownEvenBox_NEON(const uint8* src_argb, ptrdiff_t src_stride,
                                  int src_stepx,
                                  uint8* dst_argb, int dst_width) {
  asm volatile (
    "mov        r12, %4, lsl #2                \n"
    "add        %1, %1, %0                     \n"
  "1:                                          \n"
    MEMACCESS(0)
    "vld1.8     {d0}, [%0], r12                \n"  // Read 4 2x2 blocks -> 2x1
    MEMACCESS(1)
    "vld1.8     {d1}, [%1], r12                \n"
    MEMACCESS(0)
    "vld1.8     {d2}, [%0], r12                \n"
    MEMACCESS(1)
    "vld1.8     {d3}, [%1], r12                \n"
    MEMACCESS(0)
    "vld1.8     {d4}, [%0], r12                \n"
    MEMACCESS(1)
    "vld1.8     {d5}, [%1], r12                \n"
    MEMACCESS(0)
    "vld1.8     {d6}, [%0], r12                \n"
    MEMACCESS(1)
    "vld1.8     {d7}, [%1], r12                \n"
    "vaddl.u8   q0, d0, d1                     \n"
    "vaddl.u8   q1, d2, d3                     \n"
    "vaddl.u8   q2, d4, d5                     \n"
    "vaddl.u8   q3, d6, d7                     \n"
    "vswp.8     d1, d2                         \n"  // ab_cd -> ac_bd
    "vswp.8     d5, d6                         \n"  // ef_gh -> eg_fh
    "vadd.u16   q0, q0, q1                     \n"  // (a+b)_(c+d)
    "vadd.u16   q2, q2, q3                     \n"  // (e+f)_(g+h)
    "vrshrn.u16 d0, q0, #2                     \n"  // first 2 pixels.
    "vrshrn.u16 d1, q2, #2                     \n"  // next 2 pixels.
    "subs       %3, %3, #4                     \n"  // 4 pixels per loop.
    MEMACCESS(2)
    "vst1.8     {q0}, [%2]!                    \n"
    "bgt        1b                             \n"
  : "+r"(src_argb),    // %0
    "+r"(src_stride),  // %1
    "+r"(dst_argb),    // %2
    "+r"(dst_width)    // %3
  : "r"(src_stepx)     // %4
  : "memory", "cc", "r12", "q0", "q1", "q2", "q3"
  );
}

// TODO(Yang Zhang): Investigate less load instructions for
// the x/dx stepping
#define LOAD1_DATA32_LANE(dn, n)                               \
    "lsr        %5, %3, #16                    \n"             \
    "add        %6, %1, %5, lsl #2             \n"             \
    "add        %3, %3, %4                     \n"             \
    MEMACCESS(6)                                               \
    "vld1.32    {"#dn"["#n"]}, [%6]            \n"

void ScaleARGBCols_NEON(uint8* dst_argb, const uint8* src_argb,
                        int dst_width, int x, int dx) {
  int tmp;
  const uint8* src_tmp = src_argb;
  asm volatile (
  "1:                                          \n"
    LOAD1_DATA32_LANE(d0, 0)
    LOAD1_DATA32_LANE(d0, 1)
    LOAD1_DATA32_LANE(d1, 0)
    LOAD1_DATA32_LANE(d1, 1)
    LOAD1_DATA32_LANE(d2, 0)
    LOAD1_DATA32_LANE(d2, 1)
    LOAD1_DATA32_LANE(d3, 0)
    LOAD1_DATA32_LANE(d3, 1)

    MEMACCESS(0)
    "vst1.32     {q0, q1}, [%0]!               \n"  // store pixels
    "subs       %2, %2, #8                     \n"  // 8 processed per loop
    "bgt        1b                             \n"
  : "+r"(dst_argb),   // %0
    "+r"(src_argb),   // %1
    "+r"(dst_width),  // %2
    "+r"(x),          // %3
    "+r"(dx),         // %4
    "=&r"(tmp),       // %5
    "+r"(src_tmp)     // %6
  :
  : "memory", "cc", "q0", "q1"
  );
}

#undef LOAD1_DATA32_LANE

// TODO(Yang Zhang): Investigate less load instructions for
// the x/dx stepping
#define LOAD2_DATA32_LANE(dn1, dn2, n)                         \
    "lsr        %5, %3, #16                           \n"      \
    "add        %6, %1, %5, lsl #2                    \n"      \
    "add        %3, %3, %4                            \n"      \
    MEMACCESS(6)                                               \
    "vld2.32    {"#dn1"["#n"], "#dn2"["#n"]}, [%6]    \n"

void ScaleARGBFilterCols_NEON(uint8* dst_argb, const uint8* src_argb,
                              int dst_width, int x, int dx) {
  int dx_offset[4] = {0, 1, 2, 3};
  int* tmp = dx_offset;
  const uint8* src_tmp = src_argb;
  asm volatile (
    "vdup.32    q0, %3                         \n"  // x
    "vdup.32    q1, %4                         \n"  // dx
    "vld1.32    {q2}, [%5]                     \n"  // 0 1 2 3
    "vshl.i32   q9, q1, #2                     \n"  // 4 * dx
    "vmul.s32   q1, q1, q2                     \n"
    "vmov.i8    q3, #0x7f                      \n"  // 0x7F
    "vmov.i16   q15, #0x7f                     \n"  // 0x7F
    // x         , x + 1 * dx, x + 2 * dx, x + 3 * dx
    "vadd.s32   q8, q1, q0                     \n"
  "1:                                          \n"
    // d0, d1: a
    // d2, d3: b
    LOAD2_DATA32_LANE(d0, d2, 0)
    LOAD2_DATA32_LANE(d0, d2, 1)
    LOAD2_DATA32_LANE(d1, d3, 0)
    LOAD2_DATA32_LANE(d1, d3, 1)
    "vshrn.i32   d22, q8, #9                   \n"
    "vand.16     d22, d22, d30                 \n"
    "vdup.8      d24, d22[0]                   \n"
    "vdup.8      d25, d22[2]                   \n"
    "vdup.8      d26, d22[4]                   \n"
    "vdup.8      d27, d22[6]                   \n"
    "vext.8      d4, d24, d25, #4              \n"
    "vext.8      d5, d26, d27, #4              \n"  // f
    "veor.8      q10, q2, q3                   \n"  // 0x7f ^ f
    "vmull.u8    q11, d0, d20                  \n"
    "vmull.u8    q12, d1, d21                  \n"
    "vmull.u8    q13, d2, d4                   \n"
    "vmull.u8    q14, d3, d5                   \n"
    "vadd.i16    q11, q11, q13                 \n"
    "vadd.i16    q12, q12, q14                 \n"
    "vshrn.i16   d0, q11, #7                   \n"
    "vshrn.i16   d1, q12, #7                   \n"

    MEMACCESS(0)
    "vst1.32     {d0, d1}, [%0]!               \n"  // store pixels
    "vadd.s32    q8, q8, q9                    \n"
    "subs        %2, %2, #4                    \n"  // 4 processed per loop
    "bgt         1b                            \n"
  : "+r"(dst_argb),         // %0
    "+r"(src_argb),         // %1
    "+r"(dst_width),        // %2
    "+r"(x),                // %3
    "+r"(dx),               // %4
    "+r"(tmp),              // %5
    "+r"(src_tmp)           // %6
  :
  : "memory", "cc", "q0", "q1", "q2", "q3", "q8", "q9",
    "q10", "q11", "q12", "q13", "q14", "q15"
  );
}

#undef LOAD2_DATA32_LANE

#endif  // defined(__ARM_NEON__) && !defined(__aarch64__)

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
