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
#include "libyuv/row.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// This module is for GCC Neon
#if !defined(LIBYUV_DISABLE_NEON) && defined(__ARM_NEON__)

void ScaleARGBRowDown2_NEON(const uint8* src_ptr, ptrdiff_t /* src_stride */,
                            uint8* dst, int dst_width) {
  asm volatile (
#ifdef _ANDROID
	".fpu neon\n"
#endif
  "1:                                          \n"
    // load even pixels into q0, odd into q1
    "vld2.32    {q0, q1}, [%0]!                \n"
    "vld2.32    {q2, q3}, [%0]!                \n"
    "subs       %2, %2, #8                     \n"  // 8 processed per loop
    "vst1.8     {q1}, [%1]!                    \n"  // store odd pixels
    "vst1.8     {q3}, [%1]!                    \n"
    "bgt        1b                             \n"
  : "+r"(src_ptr),          // %0
    "+r"(dst),              // %1
    "+r"(dst_width)         // %2
  :
  : "memory", "cc", "q0", "q1", "q2", "q3"  // Clobber List
  );
}

void ScaleARGBRowDown2Box_NEON(const uint8* src_ptr, ptrdiff_t src_stride,
                               uint8* dst, int dst_width) {
  asm volatile (
    // change the stride to row 2 pointer
    "add        %1, %1, %0                     \n"
  "1:                                          \n"
    "vld4.8     {d0, d2, d4, d6}, [%0]!        \n"  // load 8 ARGB pixels.
    "vld4.8     {d1, d3, d5, d7}, [%0]!        \n"  // load next 8 ARGB pixels.
    "subs       %3, %3, #8                     \n"  // 8 processed per loop.
    "vpaddl.u8  q0, q0                         \n"  // B 16 bytes -> 8 shorts.
    "vpaddl.u8  q1, q1                         \n"  // G 16 bytes -> 8 shorts.
    "vpaddl.u8  q2, q2                         \n"  // R 16 bytes -> 8 shorts.
    "vpaddl.u8  q3, q3                         \n"  // A 16 bytes -> 8 shorts.
    "vld4.8     {d16, d18, d20, d22}, [%1]!    \n"  // load 8 more ARGB pixels.
    "vld4.8     {d17, d19, d21, d23}, [%1]!    \n"  // load last 8 ARGB pixels.
    "vpadal.u8  q0, q8                         \n"  // B 16 bytes -> 8 shorts.
    "vpadal.u8  q1, q9                         \n"  // G 16 bytes -> 8 shorts.
    "vpadal.u8  q2, q10                        \n"  // R 16 bytes -> 8 shorts.
    "vpadal.u8  q3, q11                        \n"  // A 16 bytes -> 8 shorts.
    "vrshrn.u16 d0, q0, #2                     \n"  // downshift, round and pack
    "vrshrn.u16 d1, q1, #2                     \n"
    "vrshrn.u16 d2, q2, #2                     \n"
    "vrshrn.u16 d3, q3, #2                     \n"
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
void ScaleARGBRowDownEven_NEON(const uint8* src_argb, ptrdiff_t, int src_stepx,
                               uint8* dst_argb, int dst_width) {
  asm volatile (
    "mov        r12, %3, lsl #2                \n"
    ".p2align  2                               \n"
  "1:                                          \n"
    "vld1.32    {d0[0]}, [%0], r12             \n"
    "vld1.32    {d0[1]}, [%0], r12             \n"
    "vld1.32    {d1[0]}, [%0], r12             \n"
    "vld1.32    {d1[1]}, [%0], r12             \n"
    "subs       %2, %2, #4                     \n"  // 4 pixels per loop.
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
    "mov       r12, %4, lsl #2                 \n"
    "add       %1, %1, %0                      \n"
    ".p2align  2                               \n"
  "1:                                          \n"
    "vld1.8    {d0}, [%0], r12                 \n"  // Read 4 2x2 blocks -> 2x1
    "vld1.8    {d1}, [%1], r12                 \n"
    "vld1.8    {d2}, [%0], r12                 \n"
    "vld1.8    {d3}, [%1], r12                 \n"
    "vld1.8    {d4}, [%0], r12                 \n"
    "vld1.8    {d5}, [%1], r12                 \n"
    "vld1.8    {d6}, [%0], r12                 \n"
    "vld1.8    {d7}, [%1], r12                 \n"
    "vaddl.u8  q0, d0, d1                      \n"
    "vaddl.u8  q1, d2, d3                      \n"
    "vaddl.u8  q2, d4, d5                      \n"
    "vaddl.u8  q3, d6, d7                      \n"
    "vswp.8    d1, d2                          \n"  // ab_cd -> ac_bd
    "vswp.8    d5, d6                          \n"  // ef_gh -> eg_fh
    "vadd.u16  q0, q0, q1                      \n"  // (a+b)_(c+d)
    "vadd.u16  q2, q2, q3                      \n"  // (e+f)_(g+h)
    "vrshrn.u16 d0, q0, #2                     \n"  // first 2 pixels.
    "vrshrn.u16 d1, q2, #2                     \n"  // next 2 pixels.
    "subs       %3, %3, #4                     \n"  // 4 pixels per loop.
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
#endif  // __ARM_NEON__

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
