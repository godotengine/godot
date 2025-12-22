/*
 *  Copyright 2024 The LibYuv Project Authors. All rights reserved.
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

#if !defined(LIBYUV_DISABLE_SME) && defined(CLANG_HAS_SME) && \
    defined(__aarch64__)

__arm_locally_streaming __arm_new("za") void TransposeWxH_SME(
    const uint8_t* src,
    int src_stride,
    uint8_t* dst,
    int dst_stride,
    int width,
    int height) {
  int vl;
  asm("cntb %x0" : "=r"(vl));

  do {
    const uint8_t* src2 = src;
    uint8_t* dst2 = dst;

    // Process up to VL elements per iteration of the inner loop.
    int block_height = height > vl ? vl : height;

    int width2 = width;
    do {
      const uint8_t* src3 = src2;

      // Process up to VL elements per iteration of the inner loop.
      int block_width = width2 > vl ? vl : width2;

      asm volatile(
          "mov     w12, #0                              \n"

          // Create a predicate to handle loading partial rows.
          "whilelt p0.b, wzr, %w[block_width]           \n"

          // Load H <= VL rows into ZA0.
          "1:                                           \n"
          "ld1b    {za0h.b[w12, 0]}, p0/z, [%[src3]]    \n"
          "add     %[src3], %[src3], %[src_stride]      \n"
          "add     w12, w12, #1                         \n"
          "cmp     w12, %w[block_height]                \n"
          "b.ne    1b                                   \n"

          // Create a predicate to handle storing partial columns.
          "whilelt p0.b, wzr, %w[block_height]          \n"
          "mov     w12, #0                              \n"

          // Store W <= VL columns from ZA0.
          "2:                                           \n"
          "st1b    {za0v.b[w12, 0]}, p0, [%[dst2]]      \n"
          "add     %[dst2], %[dst2], %[dst_stride]      \n"
          "add     w12, w12, #1                         \n"
          "cmp     w12, %w[block_width]                 \n"
          "b.ne    2b                                   \n"
          : [src3] "+r"(src3),                        // %[src3]
            [dst2] "+r"(dst2)                         // %[dst2]
          : [src_stride] "r"((ptrdiff_t)src_stride),  // %[src_stride]
            [dst_stride] "r"((ptrdiff_t)dst_stride),  // %[dst_stride]
            [block_width] "r"(block_width),           // %[block_width]
            [block_height] "r"(block_height)          // %[block_height]
          : "cc", "memory", "p0", "w12", "za");

      src2 += vl;
      width2 -= vl;
    } while (width2 > 0);

    src += vl * src_stride;
    dst += vl;
    height -= vl;
  } while (height > 0);
}

__arm_locally_streaming __arm_new("za") void TransposeUVWxH_SME(
    const uint8_t* src,
    int src_stride,
    uint8_t* dst_a,
    int dst_stride_a,
    uint8_t* dst_b,
    int dst_stride_b,
    int width,
    int height) {
  int vl;
  asm("cnth %x0" : "=r"(vl));

  do {
    const uint8_t* src2 = src;
    uint8_t* dst2_a = dst_a;
    uint8_t* dst2_b = dst_b;

    // Process up to VL bytes per iteration of the inner loop.
    int block_height = height > vl * 2 ? vl * 2 : height;

    int width2 = width;
    do {
      const uint8_t* src3 = src2;

      // Process up to VL 16-bit elements per iteration of the inner loop.
      int block_width = width2 > vl ? vl : width2;

      asm volatile(
          "mov      w12, #0                               \n"

          // Create a predicate to handle loading partial rows,
          // %[block_width] is always a multiple of two here.
          "whilelt  p0.b, wzr, %w[block_width]            \n"

          // Load H <= VL rows into ZA0, such that U/V components exist in
          // alternating columns.
          "1:                                             \n"
          "ld1b     {za0h.b[w12, 0]}, p0/z, [%[src]]      \n"
          "add      %[src], %[src], %[src_stride]         \n"
          "add      w12, w12, #1                          \n"
          "cmp      w12, %w[block_height]                 \n"
          "b.ne     1b                                    \n"

          // Create a predicate to handle storing partial columns.
          "whilelt  p0.b, wzr, %w[block_height]           \n"
          "mov      w12, #0                               \n"

          // Store alternating UV data from pairs of ZA0 columns.
          "2:                                             \n"
          "st1b     {za0v.b[w12, 0]}, p0, [%[dst_a]]      \n"
          "st1b     {za0v.b[w12, 1]}, p0, [%[dst_b]]      \n"
          "add      %[dst_a], %[dst_a], %[dst_stride_a]   \n"
          "add      %[dst_b], %[dst_b], %[dst_stride_b]   \n"
          "add      w12, w12, #2                          \n"
          "cmp      w12, %w[block_width]                  \n"
          "b.ne     2b                                    \n"
          : [src] "+r"(src3),                             // %[src]
            [dst_a] "+r"(dst2_a),                         // %[dst_a]
            [dst_b] "+r"(dst2_b)                          // %[dst_b]
          : [src_stride] "r"((ptrdiff_t)src_stride),      // %[src_stride]
            [dst_stride_a] "r"((ptrdiff_t)dst_stride_a),  // %[dst_stride_a]
            [dst_stride_b] "r"((ptrdiff_t)dst_stride_b),  // %[dst_stride_b]
            [block_width] "r"(block_width * 2),           // %[block_width]
            [block_height] "r"(block_height)              // %[block_height]
          : "cc", "memory", "p0", "w12", "za");

      src2 += 2 * vl;
      width2 -= vl;
    } while (width2 > 0);

    src += 2 * vl * src_stride;
    dst_a += 2 * vl;
    dst_b += 2 * vl;
    height -= 2 * vl;
  } while (height > 0);
}

#endif  // !defined(LIBYUV_DISABLE_SME) && defined(CLANG_HAS_SME) &&
        // defined(__aarch64__)

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
