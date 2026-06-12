/*
 *  Copyright 2014 The LibYuv Project Authors. All rights reserved.
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

// This module is for GCC Neon armv8 64 bit.
#if !defined(LIBYUV_DISABLE_NEON) && defined(__aarch64__)

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
      "sub         %w3, %w3, #8                     \n"

      // handle 8x8 blocks. this should be the majority of the plane
      "1:                                          \n"
      "mov         %0, %1                        \n"

      "ld1        {v0.8b}, [%0], %5              \n"
      "ld1        {v1.8b}, [%0], %5              \n"
      "ld1        {v2.8b}, [%0], %5              \n"
      "ld1        {v3.8b}, [%0], %5              \n"
      "ld1        {v4.8b}, [%0], %5              \n"
      "ld1        {v5.8b}, [%0], %5              \n"
      "ld1        {v6.8b}, [%0], %5              \n"
      "ld1        {v7.8b}, [%0]                  \n"

      "trn2     v16.8b, v0.8b, v1.8b             \n"
      "trn1     v17.8b, v0.8b, v1.8b             \n"
      "trn2     v18.8b, v2.8b, v3.8b             \n"
      "trn1     v19.8b, v2.8b, v3.8b             \n"
      "trn2     v20.8b, v4.8b, v5.8b             \n"
      "trn1     v21.8b, v4.8b, v5.8b             \n"
      "trn2     v22.8b, v6.8b, v7.8b             \n"
      "trn1     v23.8b, v6.8b, v7.8b             \n"

      "trn2     v3.4h, v17.4h, v19.4h            \n"
      "trn1     v1.4h, v17.4h, v19.4h            \n"
      "trn2     v2.4h, v16.4h, v18.4h            \n"
      "trn1     v0.4h, v16.4h, v18.4h            \n"
      "trn2     v7.4h, v21.4h, v23.4h            \n"
      "trn1     v5.4h, v21.4h, v23.4h            \n"
      "trn2     v6.4h, v20.4h, v22.4h            \n"
      "trn1     v4.4h, v20.4h, v22.4h            \n"

      "trn2     v21.2s, v1.2s, v5.2s             \n"
      "trn1     v17.2s, v1.2s, v5.2s             \n"
      "trn2     v20.2s, v0.2s, v4.2s             \n"
      "trn1     v16.2s, v0.2s, v4.2s             \n"
      "trn2     v23.2s, v3.2s, v7.2s             \n"
      "trn1     v19.2s, v3.2s, v7.2s             \n"
      "trn2     v22.2s, v2.2s, v6.2s             \n"
      "trn1     v18.2s, v2.2s, v6.2s             \n"

      "mov         %0, %2                        \n"

      "st1      {v17.8b}, [%0], %6               \n"
      "st1      {v16.8b}, [%0], %6               \n"
      "st1      {v19.8b}, [%0], %6               \n"
      "st1      {v18.8b}, [%0], %6               \n"
      "st1      {v21.8b}, [%0], %6               \n"
      "st1      {v20.8b}, [%0], %6               \n"
      "st1      {v23.8b}, [%0], %6               \n"
      "st1      {v22.8b}, [%0]                   \n"

      "add         %1, %1, #8                    \n"  // src += 8
      "add         %2, %2, %6, lsl #3            \n"  // dst += 8 * dst_stride
      "subs        %w3, %w3, #8                  \n"  // w   -= 8
      "b.ge        1b                            \n"

      // add 8 back to counter. if the result is 0 there are
      // no residuals.
      "adds        %w3, %w3, #8                    \n"
      "b.eq        4f                              \n"

      // some residual, so between 1 and 7 lines left to transpose
      "cmp         %w3, #2                          \n"
      "b.lt        3f                              \n"

      "cmp         %w3, #4                          \n"
      "b.lt        2f                              \n"

      // 4x8 block
      "mov         %0, %1                          \n"
      "ld1     {v0.s}[0], [%0], %5                 \n"
      "ld1     {v0.s}[1], [%0], %5                 \n"
      "ld1     {v0.s}[2], [%0], %5                 \n"
      "ld1     {v0.s}[3], [%0], %5                 \n"
      "ld1     {v1.s}[0], [%0], %5                 \n"
      "ld1     {v1.s}[1], [%0], %5                 \n"
      "ld1     {v1.s}[2], [%0], %5                 \n"
      "ld1     {v1.s}[3], [%0]                     \n"

      "mov         %0, %2                          \n"

      "ld1      {v2.16b}, [%4]                     \n"

      "tbl      v3.16b, {v0.16b}, v2.16b           \n"
      "tbl      v0.16b, {v1.16b}, v2.16b           \n"

      // TODO(frkoenig): Rework shuffle above to
      // write out with 4 instead of 8 writes.
      "st1 {v3.s}[0], [%0], %6                     \n"
      "st1 {v3.s}[1], [%0], %6                     \n"
      "st1 {v3.s}[2], [%0], %6                     \n"
      "st1 {v3.s}[3], [%0]                         \n"

      "add         %0, %2, #4                      \n"
      "st1 {v0.s}[0], [%0], %6                     \n"
      "st1 {v0.s}[1], [%0], %6                     \n"
      "st1 {v0.s}[2], [%0], %6                     \n"
      "st1 {v0.s}[3], [%0]                         \n"

      "add         %1, %1, #4                      \n"  // src += 4
      "add         %2, %2, %6, lsl #2              \n"  // dst += 4 * dst_stride
      "subs        %w3, %w3, #4                    \n"  // w   -= 4
      "b.eq        4f                              \n"

      // some residual, check to see if it includes a 2x8 block,
      // or less
      "cmp         %w3, #2                         \n"
      "b.lt        3f                              \n"

      // 2x8 block
      "2:                                          \n"
      "mov         %0, %1                          \n"
      "ld1     {v0.h}[0], [%0], %5                 \n"
      "ld1     {v1.h}[0], [%0], %5                 \n"
      "ld1     {v0.h}[1], [%0], %5                 \n"
      "ld1     {v1.h}[1], [%0], %5                 \n"
      "ld1     {v0.h}[2], [%0], %5                 \n"
      "ld1     {v1.h}[2], [%0], %5                 \n"
      "ld1     {v0.h}[3], [%0], %5                 \n"
      "ld1     {v1.h}[3], [%0]                     \n"

      "trn2    v2.8b, v0.8b, v1.8b                 \n"
      "trn1    v3.8b, v0.8b, v1.8b                 \n"

      "mov         %0, %2                          \n"

      "st1     {v3.8b}, [%0], %6                   \n"
      "st1     {v2.8b}, [%0]                       \n"

      "add         %1, %1, #2                      \n"  // src += 2
      "add         %2, %2, %6, lsl #1              \n"  // dst += 2 * dst_stride
      "subs        %w3, %w3,  #2                   \n"  // w   -= 2
      "b.eq        4f                              \n"

      // 1x8 block
      "3:                                          \n"
      "ld1         {v0.b}[0], [%1], %5             \n"
      "ld1         {v0.b}[1], [%1], %5             \n"
      "ld1         {v0.b}[2], [%1], %5             \n"
      "ld1         {v0.b}[3], [%1], %5             \n"
      "ld1         {v0.b}[4], [%1], %5             \n"
      "ld1         {v0.b}[5], [%1], %5             \n"
      "ld1         {v0.b}[6], [%1], %5             \n"
      "ld1         {v0.b}[7], [%1]                 \n"

      "st1         {v0.8b}, [%2]                   \n"

      "4:                                          \n"

      : "=&r"(src_temp),                          // %0
        "+r"(src),                                // %1
        "+r"(dst),                                // %2
        "+r"(width)                               // %3
      : "r"(&kVTbl4x4Transpose),                  // %4
        "r"(static_cast<ptrdiff_t>(src_stride)),  // %5
        "r"(static_cast<ptrdiff_t>(dst_stride))   // %6
      : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16",
        "v17", "v18", "v19", "v20", "v21", "v22", "v23");
}

static const uint8_t kVTbl4x4TransposeDi[32] = {
    0, 16, 32, 48, 2, 18, 34, 50, 4, 20, 36, 52, 6, 22, 38, 54,
    1, 17, 33, 49, 3, 19, 35, 51, 5, 21, 37, 53, 7, 23, 39, 55};

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
      "sub       %w4, %w4, #8                    \n"

      // handle 8x8 blocks. this should be the majority of the plane
      "1:                                        \n"
      "mov       %0, %1                          \n"

      "ld1       {v0.16b}, [%0], %5              \n"
      "ld1       {v1.16b}, [%0], %5              \n"
      "ld1       {v2.16b}, [%0], %5              \n"
      "ld1       {v3.16b}, [%0], %5              \n"
      "ld1       {v4.16b}, [%0], %5              \n"
      "ld1       {v5.16b}, [%0], %5              \n"
      "ld1       {v6.16b}, [%0], %5              \n"
      "ld1       {v7.16b}, [%0]                  \n"

      "trn1      v16.16b, v0.16b, v1.16b         \n"
      "trn2      v17.16b, v0.16b, v1.16b         \n"
      "trn1      v18.16b, v2.16b, v3.16b         \n"
      "trn2      v19.16b, v2.16b, v3.16b         \n"
      "trn1      v20.16b, v4.16b, v5.16b         \n"
      "trn2      v21.16b, v4.16b, v5.16b         \n"
      "trn1      v22.16b, v6.16b, v7.16b         \n"
      "trn2      v23.16b, v6.16b, v7.16b         \n"

      "trn1      v0.8h, v16.8h, v18.8h           \n"
      "trn2      v1.8h, v16.8h, v18.8h           \n"
      "trn1      v2.8h, v20.8h, v22.8h           \n"
      "trn2      v3.8h, v20.8h, v22.8h           \n"
      "trn1      v4.8h, v17.8h, v19.8h           \n"
      "trn2      v5.8h, v17.8h, v19.8h           \n"
      "trn1      v6.8h, v21.8h, v23.8h           \n"
      "trn2      v7.8h, v21.8h, v23.8h           \n"

      "trn1      v16.4s, v0.4s, v2.4s            \n"
      "trn2      v17.4s, v0.4s, v2.4s            \n"
      "trn1      v18.4s, v1.4s, v3.4s            \n"
      "trn2      v19.4s, v1.4s, v3.4s            \n"
      "trn1      v20.4s, v4.4s, v6.4s            \n"
      "trn2      v21.4s, v4.4s, v6.4s            \n"
      "trn1      v22.4s, v5.4s, v7.4s            \n"
      "trn2      v23.4s, v5.4s, v7.4s            \n"

      "mov       %0, %2                          \n"

      "st1       {v16.d}[0], [%0], %6            \n"
      "st1       {v18.d}[0], [%0], %6            \n"
      "st1       {v17.d}[0], [%0], %6            \n"
      "st1       {v19.d}[0], [%0], %6            \n"
      "st1       {v16.d}[1], [%0], %6            \n"
      "st1       {v18.d}[1], [%0], %6            \n"
      "st1       {v17.d}[1], [%0], %6            \n"
      "st1       {v19.d}[1], [%0]                \n"

      "mov       %0, %3                          \n"

      "st1       {v20.d}[0], [%0], %7            \n"
      "st1       {v22.d}[0], [%0], %7            \n"
      "st1       {v21.d}[0], [%0], %7            \n"
      "st1       {v23.d}[0], [%0], %7            \n"
      "st1       {v20.d}[1], [%0], %7            \n"
      "st1       {v22.d}[1], [%0], %7            \n"
      "st1       {v21.d}[1], [%0], %7            \n"
      "st1       {v23.d}[1], [%0]                \n"

      "add       %1, %1, #16                     \n"  // src   += 8*2
      "add       %2, %2, %6, lsl #3              \n"  // dst_a += 8 *
                                                      // dst_stride_a
      "add       %3, %3, %7, lsl #3              \n"  // dst_b += 8 *
                                                      // dst_stride_b
      "subs      %w4, %w4,  #8                   \n"  // w     -= 8
      "b.ge      1b                              \n"

      // add 8 back to counter. if the result is 0 there are
      // no residuals.
      "adds      %w4, %w4, #8                    \n"
      "b.eq      4f                              \n"

      // some residual, so between 1 and 7 lines left to transpose
      "cmp       %w4, #2                         \n"
      "b.lt      3f                              \n"

      "cmp       %w4, #4                         \n"
      "b.lt      2f                              \n"

      // TODO(frkoenig): Clean this up
      // 4x8 block
      "mov       %0, %1                          \n"
      "ld1       {v0.8b}, [%0], %5               \n"
      "ld1       {v1.8b}, [%0], %5               \n"
      "ld1       {v2.8b}, [%0], %5               \n"
      "ld1       {v3.8b}, [%0], %5               \n"
      "ld1       {v4.8b}, [%0], %5               \n"
      "ld1       {v5.8b}, [%0], %5               \n"
      "ld1       {v6.8b}, [%0], %5               \n"
      "ld1       {v7.8b}, [%0]                   \n"

      "ld1       {v30.16b}, [%8], #16            \n"
      "ld1       {v31.16b}, [%8]                 \n"

      "tbl       v16.16b, {v0.16b, v1.16b, v2.16b, v3.16b}, v30.16b  \n"
      "tbl       v17.16b, {v0.16b, v1.16b, v2.16b, v3.16b}, v31.16b  \n"
      "tbl       v18.16b, {v4.16b, v5.16b, v6.16b, v7.16b}, v30.16b  \n"
      "tbl       v19.16b, {v4.16b, v5.16b, v6.16b, v7.16b}, v31.16b  \n"

      "mov       %0, %2                          \n"

      "st1       {v16.s}[0],  [%0], %6           \n"
      "st1       {v16.s}[1],  [%0], %6           \n"
      "st1       {v16.s}[2],  [%0], %6           \n"
      "st1       {v16.s}[3],  [%0], %6           \n"

      "add       %0, %2, #4                      \n"
      "st1       {v18.s}[0], [%0], %6            \n"
      "st1       {v18.s}[1], [%0], %6            \n"
      "st1       {v18.s}[2], [%0], %6            \n"
      "st1       {v18.s}[3], [%0]                \n"

      "mov       %0, %3                          \n"

      "st1       {v17.s}[0], [%0], %7            \n"
      "st1       {v17.s}[1], [%0], %7            \n"
      "st1       {v17.s}[2], [%0], %7            \n"
      "st1       {v17.s}[3], [%0], %7            \n"

      "add       %0, %3, #4                      \n"
      "st1       {v19.s}[0],  [%0], %7           \n"
      "st1       {v19.s}[1],  [%0], %7           \n"
      "st1       {v19.s}[2],  [%0], %7           \n"
      "st1       {v19.s}[3],  [%0]               \n"

      "add       %1, %1, #8                      \n"  // src   += 4 * 2
      "add       %2, %2, %6, lsl #2              \n"  // dst_a += 4 *
                                                      // dst_stride_a
      "add       %3, %3, %7, lsl #2              \n"  // dst_b += 4 *
                                                      // dst_stride_b
      "subs      %w4,  %w4,  #4                  \n"  // w     -= 4
      "b.eq      4f                              \n"

      // some residual, check to see if it includes a 2x8 block,
      // or less
      "cmp       %w4, #2                         \n"
      "b.lt      3f                              \n"

      // 2x8 block
      "2:                                        \n"
      "mov       %0, %1                          \n"
      "ld2       {v0.h, v1.h}[0], [%0], %5       \n"
      "ld2       {v2.h, v3.h}[0], [%0], %5       \n"
      "ld2       {v0.h, v1.h}[1], [%0], %5       \n"
      "ld2       {v2.h, v3.h}[1], [%0], %5       \n"
      "ld2       {v0.h, v1.h}[2], [%0], %5       \n"
      "ld2       {v2.h, v3.h}[2], [%0], %5       \n"
      "ld2       {v0.h, v1.h}[3], [%0], %5       \n"
      "ld2       {v2.h, v3.h}[3], [%0]           \n"

      "trn1      v4.8b, v0.8b, v2.8b             \n"
      "trn2      v5.8b, v0.8b, v2.8b             \n"
      "trn1      v6.8b, v1.8b, v3.8b             \n"
      "trn2      v7.8b, v1.8b, v3.8b             \n"

      "mov       %0, %2                          \n"

      "st1       {v4.d}[0], [%0], %6             \n"
      "st1       {v6.d}[0], [%0]                 \n"

      "mov       %0, %3                          \n"

      "st1       {v5.d}[0], [%0], %7             \n"
      "st1       {v7.d}[0], [%0]                 \n"

      "add       %1, %1, #4                      \n"  // src   += 2 * 2
      "add       %2, %2, %6, lsl #1              \n"  // dst_a += 2 *
                                                      // dst_stride_a
      "add       %3, %3, %7, lsl #1              \n"  // dst_b += 2 *
                                                      // dst_stride_b
      "subs      %w4,  %w4,  #2                  \n"  // w     -= 2
      "b.eq      4f                              \n"

      // 1x8 block
      "3:                                        \n"
      "ld2       {v0.b, v1.b}[0], [%1], %5       \n"
      "ld2       {v0.b, v1.b}[1], [%1], %5       \n"
      "ld2       {v0.b, v1.b}[2], [%1], %5       \n"
      "ld2       {v0.b, v1.b}[3], [%1], %5       \n"
      "ld2       {v0.b, v1.b}[4], [%1], %5       \n"
      "ld2       {v0.b, v1.b}[5], [%1], %5       \n"
      "ld2       {v0.b, v1.b}[6], [%1], %5       \n"
      "ld2       {v0.b, v1.b}[7], [%1]           \n"

      "st1       {v0.d}[0], [%2]                 \n"
      "st1       {v1.d}[0], [%3]                 \n"

      "4:                                        \n"

      : "=&r"(src_temp),                            // %0
        "+r"(src),                                  // %1
        "+r"(dst_a),                                // %2
        "+r"(dst_b),                                // %3
        "+r"(width)                                 // %4
      : "r"(static_cast<ptrdiff_t>(src_stride)),    // %5
        "r"(static_cast<ptrdiff_t>(dst_stride_a)),  // %6
        "r"(static_cast<ptrdiff_t>(dst_stride_b)),  // %7
        "r"(&kVTbl4x4TransposeDi)                   // %8
      : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16",
        "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v30", "v31");
}
#endif  // !defined(LIBYUV_DISABLE_NEON) && defined(__aarch64__)

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
