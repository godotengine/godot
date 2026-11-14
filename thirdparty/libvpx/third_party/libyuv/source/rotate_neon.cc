/*
 *  Copyright 2011 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "libyuv/rotate_row.h"
#include "libyuv/row.h"

#include "libyuv/basic_types.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

#if !defined(LIBYUV_DISABLE_NEON) && defined(__ARM_NEON__) && \
    !defined(__aarch64__)

static const uvec8 kVTbl4x4Transpose = {0, 4, 8,  12, 1, 5, 9,  13,
                                        2, 6, 10, 14, 3, 7, 11, 15};

void TransposeWx8_NEON(const uint8_t* src,
                       int src_stride,
                       uint8_t* dst,
                       int dst_stride,
                       int width) {
  const uint8_t* src_temp;
  asm volatile(
      // loops are on blocks of 8. loop will stop when
      // counter gets to or below 0. starting the counter
      // at w-8 allow for this
      "sub         %5, #8                        \n"

      // handle 8x8 blocks. this should be the majority of the plane
      "1:                                        \n"
      "mov         %0, %1                      \n"

      "vld1.8      {d0}, [%0], %2              \n"
      "vld1.8      {d1}, [%0], %2              \n"
      "vld1.8      {d2}, [%0], %2              \n"
      "vld1.8      {d3}, [%0], %2              \n"
      "vld1.8      {d4}, [%0], %2              \n"
      "vld1.8      {d5}, [%0], %2              \n"
      "vld1.8      {d6}, [%0], %2              \n"
      "vld1.8      {d7}, [%0]                  \n"

      "vtrn.8      d1, d0                      \n"
      "vtrn.8      d3, d2                      \n"
      "vtrn.8      d5, d4                      \n"
      "vtrn.8      d7, d6                      \n"

      "vtrn.16     d1, d3                      \n"
      "vtrn.16     d0, d2                      \n"
      "vtrn.16     d5, d7                      \n"
      "vtrn.16     d4, d6                      \n"

      "vtrn.32     d1, d5                      \n"
      "vtrn.32     d0, d4                      \n"
      "vtrn.32     d3, d7                      \n"
      "vtrn.32     d2, d6                      \n"

      "vrev16.8    q0, q0                      \n"
      "vrev16.8    q1, q1                      \n"
      "vrev16.8    q2, q2                      \n"
      "vrev16.8    q3, q3                      \n"

      "mov         %0, %3                      \n"

      "vst1.8      {d1}, [%0], %4              \n"
      "vst1.8      {d0}, [%0], %4              \n"
      "vst1.8      {d3}, [%0], %4              \n"
      "vst1.8      {d2}, [%0], %4              \n"
      "vst1.8      {d5}, [%0], %4              \n"
      "vst1.8      {d4}, [%0], %4              \n"
      "vst1.8      {d7}, [%0], %4              \n"
      "vst1.8      {d6}, [%0]                  \n"

      "add         %1, #8                      \n"  // src += 8
      "add         %3, %3, %4, lsl #3          \n"  // dst += 8 * dst_stride
      "subs        %5,  #8                     \n"  // w   -= 8
      "bge         1b                          \n"

      // add 8 back to counter. if the result is 0 there are
      // no residuals.
      "adds        %5, #8                        \n"
      "beq         4f                            \n"

      // some residual, so between 1 and 7 lines left to transpose
      "cmp         %5, #2                        \n"
      "blt         3f                            \n"

      "cmp         %5, #4                        \n"
      "blt         2f                            \n"

      // 4x8 block
      "mov         %0, %1                        \n"
      "vld1.32     {d0[0]}, [%0], %2             \n"
      "vld1.32     {d0[1]}, [%0], %2             \n"
      "vld1.32     {d1[0]}, [%0], %2             \n"
      "vld1.32     {d1[1]}, [%0], %2             \n"
      "vld1.32     {d2[0]}, [%0], %2             \n"
      "vld1.32     {d2[1]}, [%0], %2             \n"
      "vld1.32     {d3[0]}, [%0], %2             \n"
      "vld1.32     {d3[1]}, [%0]                 \n"

      "mov         %0, %3                        \n"

      "vld1.8      {q3}, [%6]                    \n"

      "vtbl.8      d4, {d0, d1}, d6              \n"
      "vtbl.8      d5, {d0, d1}, d7              \n"
      "vtbl.8      d0, {d2, d3}, d6              \n"
      "vtbl.8      d1, {d2, d3}, d7              \n"

      // TODO(frkoenig): Rework shuffle above to
      // write out with 4 instead of 8 writes.
      "vst1.32     {d4[0]}, [%0], %4             \n"
      "vst1.32     {d4[1]}, [%0], %4             \n"
      "vst1.32     {d5[0]}, [%0], %4             \n"
      "vst1.32     {d5[1]}, [%0]                 \n"

      "add         %0, %3, #4                    \n"
      "vst1.32     {d0[0]}, [%0], %4             \n"
      "vst1.32     {d0[1]}, [%0], %4             \n"
      "vst1.32     {d1[0]}, [%0], %4             \n"
      "vst1.32     {d1[1]}, [%0]                 \n"

      "add         %1, #4                        \n"  // src += 4
      "add         %3, %3, %4, lsl #2            \n"  // dst += 4 * dst_stride
      "subs        %5,  #4                       \n"  // w   -= 4
      "beq         4f                            \n"

      // some residual, check to see if it includes a 2x8 block,
      // or less
      "cmp         %5, #2                        \n"
      "blt         3f                            \n"

      // 2x8 block
      "2:                                        \n"
      "mov         %0, %1                        \n"
      "vld1.16     {d0[0]}, [%0], %2             \n"
      "vld1.16     {d1[0]}, [%0], %2             \n"
      "vld1.16     {d0[1]}, [%0], %2             \n"
      "vld1.16     {d1[1]}, [%0], %2             \n"
      "vld1.16     {d0[2]}, [%0], %2             \n"
      "vld1.16     {d1[2]}, [%0], %2             \n"
      "vld1.16     {d0[3]}, [%0], %2             \n"
      "vld1.16     {d1[3]}, [%0]                 \n"

      "vtrn.8      d0, d1                        \n"

      "mov         %0, %3                        \n"

      "vst1.64     {d0}, [%0], %4                \n"
      "vst1.64     {d1}, [%0]                    \n"

      "add         %1, #2                        \n"  // src += 2
      "add         %3, %3, %4, lsl #1            \n"  // dst += 2 * dst_stride
      "subs        %5,  #2                       \n"  // w   -= 2
      "beq         4f                            \n"

      // 1x8 block
      "3:                                        \n"
      "vld1.8      {d0[0]}, [%1], %2             \n"
      "vld1.8      {d0[1]}, [%1], %2             \n"
      "vld1.8      {d0[2]}, [%1], %2             \n"
      "vld1.8      {d0[3]}, [%1], %2             \n"
      "vld1.8      {d0[4]}, [%1], %2             \n"
      "vld1.8      {d0[5]}, [%1], %2             \n"
      "vld1.8      {d0[6]}, [%1], %2             \n"
      "vld1.8      {d0[7]}, [%1]                 \n"

      "vst1.64     {d0}, [%3]                    \n"

      "4:                                        \n"

      : "=&r"(src_temp),         // %0
        "+r"(src),               // %1
        "+r"(src_stride),        // %2
        "+r"(dst),               // %3
        "+r"(dst_stride),        // %4
        "+r"(width)              // %5
      : "r"(&kVTbl4x4Transpose)  // %6
      : "memory", "cc", "q0", "q1", "q2", "q3");
}

static const uvec8 kVTbl4x4TransposeDi = {0, 8,  1, 9,  2, 10, 3, 11,
                                          4, 12, 5, 13, 6, 14, 7, 15};

void TransposeUVWx8_NEON(const uint8_t* src,
                         int src_stride,
                         uint8_t* dst_a,
                         int dst_stride_a,
                         uint8_t* dst_b,
                         int dst_stride_b,
                         int width) {
  const uint8_t* src_temp;
  asm volatile(
      // loops are on blocks of 8. loop will stop when
      // counter gets to or below 0. starting the counter
      // at w-8 allow for this
      "sub         %7, #8                        \n"

      // handle 8x8 blocks. this should be the majority of the plane
      "1:                                        \n"
      "mov         %0, %1                      \n"

      "vld2.8      {d0,  d1},  [%0], %2        \n"
      "vld2.8      {d2,  d3},  [%0], %2        \n"
      "vld2.8      {d4,  d5},  [%0], %2        \n"
      "vld2.8      {d6,  d7},  [%0], %2        \n"
      "vld2.8      {d16, d17}, [%0], %2        \n"
      "vld2.8      {d18, d19}, [%0], %2        \n"
      "vld2.8      {d20, d21}, [%0], %2        \n"
      "vld2.8      {d22, d23}, [%0]            \n"

      "vtrn.8      q1, q0                      \n"
      "vtrn.8      q3, q2                      \n"
      "vtrn.8      q9, q8                      \n"
      "vtrn.8      q11, q10                    \n"

      "vtrn.16     q1, q3                      \n"
      "vtrn.16     q0, q2                      \n"
      "vtrn.16     q9, q11                     \n"
      "vtrn.16     q8, q10                     \n"

      "vtrn.32     q1, q9                      \n"
      "vtrn.32     q0, q8                      \n"
      "vtrn.32     q3, q11                     \n"
      "vtrn.32     q2, q10                     \n"

      "vrev16.8    q0, q0                      \n"
      "vrev16.8    q1, q1                      \n"
      "vrev16.8    q2, q2                      \n"
      "vrev16.8    q3, q3                      \n"
      "vrev16.8    q8, q8                      \n"
      "vrev16.8    q9, q9                      \n"
      "vrev16.8    q10, q10                    \n"
      "vrev16.8    q11, q11                    \n"

      "mov         %0, %3                      \n"

      "vst1.8      {d2},  [%0], %4             \n"
      "vst1.8      {d0},  [%0], %4             \n"
      "vst1.8      {d6},  [%0], %4             \n"
      "vst1.8      {d4},  [%0], %4             \n"
      "vst1.8      {d18}, [%0], %4             \n"
      "vst1.8      {d16}, [%0], %4             \n"
      "vst1.8      {d22}, [%0], %4             \n"
      "vst1.8      {d20}, [%0]                 \n"

      "mov         %0, %5                      \n"

      "vst1.8      {d3},  [%0], %6             \n"
      "vst1.8      {d1},  [%0], %6             \n"
      "vst1.8      {d7},  [%0], %6             \n"
      "vst1.8      {d5},  [%0], %6             \n"
      "vst1.8      {d19}, [%0], %6             \n"
      "vst1.8      {d17}, [%0], %6             \n"
      "vst1.8      {d23}, [%0], %6             \n"
      "vst1.8      {d21}, [%0]                 \n"

      "add         %1, #8*2                    \n"  // src   += 8*2
      "add         %3, %3, %4, lsl #3          \n"  // dst_a += 8 * dst_stride_a
      "add         %5, %5, %6, lsl #3          \n"  // dst_b += 8 * dst_stride_b
      "subs        %7,  #8                     \n"  // w     -= 8
      "bge         1b                          \n"

      // add 8 back to counter. if the result is 0 there are
      // no residuals.
      "adds        %7, #8                        \n"
      "beq         4f                            \n"

      // some residual, so between 1 and 7 lines left to transpose
      "cmp         %7, #2                        \n"
      "blt         3f                            \n"

      "cmp         %7, #4                        \n"
      "blt         2f                            \n"

      // TODO(frkoenig): Clean this up
      // 4x8 block
      "mov         %0, %1                        \n"
      "vld1.64     {d0}, [%0], %2                \n"
      "vld1.64     {d1}, [%0], %2                \n"
      "vld1.64     {d2}, [%0], %2                \n"
      "vld1.64     {d3}, [%0], %2                \n"
      "vld1.64     {d4}, [%0], %2                \n"
      "vld1.64     {d5}, [%0], %2                \n"
      "vld1.64     {d6}, [%0], %2                \n"
      "vld1.64     {d7}, [%0]                    \n"

      "vld1.8      {q15}, [%8]                   \n"

      "vtrn.8      q0, q1                        \n"
      "vtrn.8      q2, q3                        \n"

      "vtbl.8      d16, {d0, d1}, d30            \n"
      "vtbl.8      d17, {d0, d1}, d31            \n"
      "vtbl.8      d18, {d2, d3}, d30            \n"
      "vtbl.8      d19, {d2, d3}, d31            \n"
      "vtbl.8      d20, {d4, d5}, d30            \n"
      "vtbl.8      d21, {d4, d5}, d31            \n"
      "vtbl.8      d22, {d6, d7}, d30            \n"
      "vtbl.8      d23, {d6, d7}, d31            \n"

      "mov         %0, %3                        \n"

      "vst1.32     {d16[0]},  [%0], %4           \n"
      "vst1.32     {d16[1]},  [%0], %4           \n"
      "vst1.32     {d17[0]},  [%0], %4           \n"
      "vst1.32     {d17[1]},  [%0], %4           \n"

      "add         %0, %3, #4                    \n"
      "vst1.32     {d20[0]}, [%0], %4            \n"
      "vst1.32     {d20[1]}, [%0], %4            \n"
      "vst1.32     {d21[0]}, [%0], %4            \n"
      "vst1.32     {d21[1]}, [%0]                \n"

      "mov         %0, %5                        \n"

      "vst1.32     {d18[0]}, [%0], %6            \n"
      "vst1.32     {d18[1]}, [%0], %6            \n"
      "vst1.32     {d19[0]}, [%0], %6            \n"
      "vst1.32     {d19[1]}, [%0], %6            \n"

      "add         %0, %5, #4                    \n"
      "vst1.32     {d22[0]},  [%0], %6           \n"
      "vst1.32     {d22[1]},  [%0], %6           \n"
      "vst1.32     {d23[0]},  [%0], %6           \n"
      "vst1.32     {d23[1]},  [%0]               \n"

      "add         %1, #4*2                      \n"  // src   += 4 * 2
      "add         %3, %3, %4, lsl #2            \n"  // dst_a += 4 *
                                                      // dst_stride_a
      "add         %5, %5, %6, lsl #2            \n"  // dst_b += 4 *
                                                      // dst_stride_b
      "subs        %7,  #4                       \n"  // w     -= 4
      "beq         4f                            \n"

      // some residual, check to see if it includes a 2x8 block,
      // or less
      "cmp         %7, #2                        \n"
      "blt         3f                            \n"

      // 2x8 block
      "2:                                        \n"
      "mov         %0, %1                        \n"
      "vld2.16     {d0[0], d2[0]}, [%0], %2      \n"
      "vld2.16     {d1[0], d3[0]}, [%0], %2      \n"
      "vld2.16     {d0[1], d2[1]}, [%0], %2      \n"
      "vld2.16     {d1[1], d3[1]}, [%0], %2      \n"
      "vld2.16     {d0[2], d2[2]}, [%0], %2      \n"
      "vld2.16     {d1[2], d3[2]}, [%0], %2      \n"
      "vld2.16     {d0[3], d2[3]}, [%0], %2      \n"
      "vld2.16     {d1[3], d3[3]}, [%0]          \n"

      "vtrn.8      d0, d1                        \n"
      "vtrn.8      d2, d3                        \n"

      "mov         %0, %3                        \n"

      "vst1.64     {d0}, [%0], %4                \n"
      "vst1.64     {d2}, [%0]                    \n"

      "mov         %0, %5                        \n"

      "vst1.64     {d1}, [%0], %6                \n"
      "vst1.64     {d3}, [%0]                    \n"

      "add         %1, #2*2                      \n"  // src   += 2 * 2
      "add         %3, %3, %4, lsl #1            \n"  // dst_a += 2 *
                                                      // dst_stride_a
      "add         %5, %5, %6, lsl #1            \n"  // dst_b += 2 *
                                                      // dst_stride_b
      "subs        %7,  #2                       \n"  // w     -= 2
      "beq         4f                            \n"

      // 1x8 block
      "3:                                        \n"
      "vld2.8      {d0[0], d1[0]}, [%1], %2      \n"
      "vld2.8      {d0[1], d1[1]}, [%1], %2      \n"
      "vld2.8      {d0[2], d1[2]}, [%1], %2      \n"
      "vld2.8      {d0[3], d1[3]}, [%1], %2      \n"
      "vld2.8      {d0[4], d1[4]}, [%1], %2      \n"
      "vld2.8      {d0[5], d1[5]}, [%1], %2      \n"
      "vld2.8      {d0[6], d1[6]}, [%1], %2      \n"
      "vld2.8      {d0[7], d1[7]}, [%1]          \n"

      "vst1.64     {d0}, [%3]                    \n"
      "vst1.64     {d1}, [%5]                    \n"

      "4:                                        \n"

      : "=&r"(src_temp),           // %0
        "+r"(src),                 // %1
        "+r"(src_stride),          // %2
        "+r"(dst_a),               // %3
        "+r"(dst_stride_a),        // %4
        "+r"(dst_b),               // %5
        "+r"(dst_stride_b),        // %6
        "+r"(width)                // %7
      : "r"(&kVTbl4x4TransposeDi)  // %8
      : "memory", "cc", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11");
}
#endif  // defined(__ARM_NEON__) && !defined(__aarch64__)

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
