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

void TransposeWx8_NEON(const uint8_t* src,
                       int src_stride,
                       uint8_t* dst,
                       int dst_stride,
                       int width) {
  const uint8_t* temp;
  asm volatile(
      // loops are on blocks of 8. loop will stop when
      // counter gets to or below 0. starting the counter
      // at w-8 allow for this
      "sub         %[width], #8                  \n"

      "1:          \n"
      "mov         %[temp], %[src]               \n"
      "vld1.8      {d0}, [%[temp]], %[src_stride] \n"
      "vld1.8      {d1}, [%[temp]], %[src_stride] \n"
      "vld1.8      {d2}, [%[temp]], %[src_stride] \n"
      "vld1.8      {d3}, [%[temp]], %[src_stride] \n"
      "vld1.8      {d4}, [%[temp]], %[src_stride] \n"
      "vld1.8      {d5}, [%[temp]], %[src_stride] \n"
      "vld1.8      {d6}, [%[temp]], %[src_stride] \n"
      "vld1.8      {d7}, [%[temp]]               \n"
      "add         %[src], #8                    \n"

      "vtrn.8      d1, d0                        \n"
      "vtrn.8      d3, d2                        \n"
      "vtrn.8      d5, d4                        \n"
      "vtrn.8      d7, d6                        \n"
      "subs        %[width], #8                  \n"

      "vtrn.16     d1, d3                        \n"
      "vtrn.16     d0, d2                        \n"
      "vtrn.16     d5, d7                        \n"
      "vtrn.16     d4, d6                        \n"

      "vtrn.32     d1, d5                        \n"
      "vtrn.32     d0, d4                        \n"
      "vtrn.32     d3, d7                        \n"
      "vtrn.32     d2, d6                        \n"

      "vrev16.8    q0, q0                        \n"
      "vrev16.8    q1, q1                        \n"
      "vrev16.8    q2, q2                        \n"
      "vrev16.8    q3, q3                        \n"

      "mov         %[temp], %[dst]               \n"
      "vst1.8      {d1}, [%[temp]], %[dst_stride] \n"
      "vst1.8      {d0}, [%[temp]], %[dst_stride] \n"
      "vst1.8      {d3}, [%[temp]], %[dst_stride] \n"
      "vst1.8      {d2}, [%[temp]], %[dst_stride] \n"
      "vst1.8      {d5}, [%[temp]], %[dst_stride] \n"
      "vst1.8      {d4}, [%[temp]], %[dst_stride] \n"
      "vst1.8      {d7}, [%[temp]], %[dst_stride] \n"
      "vst1.8      {d6}, [%[temp]]               \n"
      "add         %[dst], %[dst], %[dst_stride], lsl #3 \n"

      "bge         1b                            \n"
      : [temp] "=&r"(temp),            // %[temp]
        [src] "+r"(src),               // %[src]
        [dst] "+r"(dst),               // %[dst]
        [width] "+r"(width)            // %[width]
      : [src_stride] "r"(src_stride),  // %[src_stride]
        [dst_stride] "r"(dst_stride)   // %[dst_stride]
      : "memory", "cc", "q0", "q1", "q2", "q3");
}

void TransposeUVWx8_NEON(const uint8_t* src,
                         int src_stride,
                         uint8_t* dst_a,
                         int dst_stride_a,
                         uint8_t* dst_b,
                         int dst_stride_b,
                         int width) {
  const uint8_t* temp;
  asm volatile(
      // loops are on blocks of 8. loop will stop when
      // counter gets to or below 0. starting the counter
      // at w-8 allow for this
      "sub         %[width], #8                  \n"

      "1:          \n"
      "mov         %[temp], %[src]               \n"
      "vld2.8      {d0,  d1},  [%[temp]], %[src_stride] \n"
      "vld2.8      {d2,  d3},  [%[temp]], %[src_stride] \n"
      "vld2.8      {d4,  d5},  [%[temp]], %[src_stride] \n"
      "vld2.8      {d6,  d7},  [%[temp]], %[src_stride] \n"
      "vld2.8      {d16, d17}, [%[temp]], %[src_stride] \n"
      "vld2.8      {d18, d19}, [%[temp]], %[src_stride] \n"
      "vld2.8      {d20, d21}, [%[temp]], %[src_stride] \n"
      "vld2.8      {d22, d23}, [%[temp]]         \n"
      "add         %[src], #8*2                  \n"

      "vtrn.8      q1, q0                        \n"
      "vtrn.8      q3, q2                        \n"
      "vtrn.8      q9, q8                        \n"
      "vtrn.8      q11, q10                      \n"
      "subs        %[width], #8                  \n"

      "vtrn.16     q1, q3                        \n"
      "vtrn.16     q0, q2                        \n"
      "vtrn.16     q9, q11                       \n"
      "vtrn.16     q8, q10                       \n"

      "vtrn.32     q1, q9                        \n"
      "vtrn.32     q0, q8                        \n"
      "vtrn.32     q3, q11                       \n"
      "vtrn.32     q2, q10                       \n"

      "vrev16.8    q0, q0                        \n"
      "vrev16.8    q1, q1                        \n"
      "vrev16.8    q2, q2                        \n"
      "vrev16.8    q3, q3                        \n"
      "vrev16.8    q8, q8                        \n"
      "vrev16.8    q9, q9                        \n"
      "vrev16.8    q10, q10                      \n"
      "vrev16.8    q11, q11                      \n"

      "mov         %[temp], %[dst_a]             \n"
      "vst1.8      {d2},  [%[temp]], %[dst_stride_a] \n"
      "vst1.8      {d0},  [%[temp]], %[dst_stride_a] \n"
      "vst1.8      {d6},  [%[temp]], %[dst_stride_a] \n"
      "vst1.8      {d4},  [%[temp]], %[dst_stride_a] \n"
      "vst1.8      {d18}, [%[temp]], %[dst_stride_a] \n"
      "vst1.8      {d16}, [%[temp]], %[dst_stride_a] \n"
      "vst1.8      {d22}, [%[temp]], %[dst_stride_a] \n"
      "vst1.8      {d20}, [%[temp]]              \n"
      "add         %[dst_a], %[dst_a], %[dst_stride_a], lsl #3 \n"

      "mov         %[temp], %[dst_b]             \n"
      "vst1.8      {d3},  [%[temp]], %[dst_stride_b] \n"
      "vst1.8      {d1},  [%[temp]], %[dst_stride_b] \n"
      "vst1.8      {d7},  [%[temp]], %[dst_stride_b] \n"
      "vst1.8      {d5},  [%[temp]], %[dst_stride_b] \n"
      "vst1.8      {d19}, [%[temp]], %[dst_stride_b] \n"
      "vst1.8      {d17}, [%[temp]], %[dst_stride_b] \n"
      "vst1.8      {d23}, [%[temp]], %[dst_stride_b] \n"
      "vst1.8      {d21}, [%[temp]]              \n"
      "add         %[dst_b], %[dst_b], %[dst_stride_b], lsl #3 \n"

      "bge         1b                            \n"
      : [temp] "=&r"(temp),                // %[temp]
        [src] "+r"(src),                   // %[src]
        [dst_a] "+r"(dst_a),               // %[dst_a]
        [dst_b] "+r"(dst_b),               // %[dst_b]
        [width] "+r"(width)                // %[width]
      : [src_stride] "r"(src_stride),      // %[src_stride]
        [dst_stride_a] "r"(dst_stride_a),  // %[dst_stride_a]
        [dst_stride_b] "r"(dst_stride_b)   // %[dst_stride_b]
      : "memory", "cc", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11");
}

// Transpose 32 bit values (ARGB)
void Transpose4x4_32_NEON(const uint8_t* src,
                          int src_stride,
                          uint8_t* dst,
                          int dst_stride,
                          int width) {
  const uint8_t* src1 = src + src_stride;
  const uint8_t* src2 = src1 + src_stride;
  const uint8_t* src3 = src2 + src_stride;
  uint8_t* dst1 = dst + dst_stride;
  uint8_t* dst2 = dst1 + dst_stride;
  uint8_t* dst3 = dst2 + dst_stride;
  asm volatile(
      // Main loop transpose 4x4.  Read a column, write a row.
      "1:          \n"
      "vld4.32     {d0[0], d2[0], d4[0], d6[0]}, [%0], %9 \n"
      "vld4.32     {d0[1], d2[1], d4[1], d6[1]}, [%1], %9 \n"
      "vld4.32     {d1[0], d3[0], d5[0], d7[0]}, [%2], %9 \n"
      "vld4.32     {d1[1], d3[1], d5[1], d7[1]}, [%3], %9 \n"
      "subs        %8, %8, #4                    \n"  // w -= 4
      "vst1.8      {q0}, [%4]!                   \n"
      "vst1.8      {q1}, [%5]!                   \n"
      "vst1.8      {q2}, [%6]!                   \n"
      "vst1.8      {q3}, [%7]!                   \n"
      "bgt         1b                            \n"

      : "+r"(src),                        // %0
        "+r"(src1),                       // %1
        "+r"(src2),                       // %2
        "+r"(src3),                       // %3
        "+r"(dst),                        // %4
        "+r"(dst1),                       // %5
        "+r"(dst2),                       // %6
        "+r"(dst3),                       // %7
        "+r"(width)                       // %8
      : "r"((ptrdiff_t)(src_stride * 4))  // %9
      : "memory", "cc", "q0", "q1", "q2", "q3");
}

#endif  // defined(__ARM_NEON__) && !defined(__aarch64__)

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
