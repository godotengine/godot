/*
 *  Copyright 2014 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "libyuv/scale.h"
#include "libyuv/row.h"
#include "libyuv/scale_row.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// This module is for GCC Neon armv8 64 bit.
#if !defined(LIBYUV_DISABLE_NEON) && defined(__aarch64__)

// Read 32x1 throw away even pixels, and write 16x1.
void ScaleRowDown2_NEON(const uint8* src_ptr, ptrdiff_t src_stride,
                        uint8* dst, int dst_width) {
  asm volatile (
  "1:                                          \n"
    // load even pixels into v0, odd into v1
    MEMACCESS(0)
    "ld2        {v0.16b,v1.16b}, [%0], #32     \n"
    "subs       %w2, %w2, #16                  \n"  // 16 processed per loop
    MEMACCESS(1)
    "st1        {v1.16b}, [%1], #16            \n"  // store odd pixels
    "b.gt       1b                             \n"
  : "+r"(src_ptr),          // %0
    "+r"(dst),              // %1
    "+r"(dst_width)         // %2
  :
  : "v0", "v1"              // Clobber List
  );
}

// Read 32x1 average down and write 16x1.
void ScaleRowDown2Linear_NEON(const uint8* src_ptr, ptrdiff_t src_stride,
                           uint8* dst, int dst_width) {
  asm volatile (
  "1:                                          \n"
    MEMACCESS(0)
    "ld1        {v0.16b,v1.16b}, [%0], #32     \n"  // load pixels and post inc
    "subs       %w2, %w2, #16                  \n"  // 16 processed per loop
    "uaddlp     v0.8h, v0.16b                  \n"  // add adjacent
    "uaddlp     v1.8h, v1.16b                  \n"
    "rshrn      v0.8b, v0.8h, #1               \n"  // downshift, round and pack
    "rshrn2     v0.16b, v1.8h, #1              \n"
    MEMACCESS(1)
    "st1        {v0.16b}, [%1], #16            \n"
    "b.gt       1b                             \n"
  : "+r"(src_ptr),          // %0
    "+r"(dst),              // %1
    "+r"(dst_width)         // %2
  :
  : "v0", "v1"     // Clobber List
  );
}

// Read 32x2 average down and write 16x1.
void ScaleRowDown2Box_NEON(const uint8* src_ptr, ptrdiff_t src_stride,
                           uint8* dst, int dst_width) {
  asm volatile (
    // change the stride to row 2 pointer
    "add        %1, %1, %0                     \n"
  "1:                                          \n"
    MEMACCESS(0)
    "ld1        {v0.16b,v1.16b}, [%0], #32    \n"  // load row 1 and post inc
    MEMACCESS(1)
    "ld1        {v2.16b, v3.16b}, [%1], #32    \n"  // load row 2 and post inc
    "subs       %w3, %w3, #16                  \n"  // 16 processed per loop
    "uaddlp     v0.8h, v0.16b                  \n"  // row 1 add adjacent
    "uaddlp     v1.8h, v1.16b                  \n"
    "uadalp     v0.8h, v2.16b                  \n"  // row 2 add adjacent + row1
    "uadalp     v1.8h, v3.16b                  \n"
    "rshrn      v0.8b, v0.8h, #2               \n"  // downshift, round and pack
    "rshrn2     v0.16b, v1.8h, #2              \n"
    MEMACCESS(2)
    "st1        {v0.16b}, [%2], #16            \n"
    "b.gt       1b                             \n"
  : "+r"(src_ptr),          // %0
    "+r"(src_stride),       // %1
    "+r"(dst),              // %2
    "+r"(dst_width)         // %3
  :
  : "v0", "v1", "v2", "v3"     // Clobber List
  );
}

void ScaleRowDown4_NEON(const uint8* src_ptr, ptrdiff_t src_stride,
                        uint8* dst_ptr, int dst_width) {
  asm volatile (
  "1:                                          \n"
    MEMACCESS(0)
    "ld4     {v0.8b,v1.8b,v2.8b,v3.8b}, [%0], #32          \n"  // src line 0
    "subs       %w2, %w2, #8                   \n"  // 8 processed per loop
    MEMACCESS(1)
    "st1     {v2.8b}, [%1], #8                 \n"
    "b.gt       1b                             \n"
  : "+r"(src_ptr),          // %0
    "+r"(dst_ptr),          // %1
    "+r"(dst_width)         // %2
  :
  : "v0", "v1", "v2", "v3", "memory", "cc"
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
    "ld1     {v0.16b}, [%0], #16               \n"   // load up 16x4
    MEMACCESS(3)
    "ld1     {v1.16b}, [%2], #16               \n"
    MEMACCESS(4)
    "ld1     {v2.16b}, [%3], #16               \n"
    MEMACCESS(5)
    "ld1     {v3.16b}, [%4], #16               \n"
    "subs    %w5, %w5, #4                      \n"
    "uaddlp  v0.8h, v0.16b                     \n"
    "uadalp  v0.8h, v1.16b                     \n"
    "uadalp  v0.8h, v2.16b                     \n"
    "uadalp  v0.8h, v3.16b                     \n"
    "addp    v0.8h, v0.8h, v0.8h               \n"
    "rshrn   v0.8b, v0.8h, #4                  \n"   // divide by 16 w/rounding
    MEMACCESS(1)
    "st1    {v0.s}[0], [%1], #4                \n"
    "b.gt       1b                             \n"
  : "+r"(src_ptr),   // %0
    "+r"(dst_ptr),   // %1
    "+r"(src_ptr1),  // %2
    "+r"(src_ptr2),  // %3
    "+r"(src_ptr3),  // %4
    "+r"(dst_width)  // %5
  :
  : "v0", "v1", "v2", "v3", "memory", "cc"
  );
}

// Down scale from 4 to 3 pixels. Use the neon multilane read/write
// to load up the every 4th pixel into a 4 different registers.
// Point samples 32 pixels to 24 pixels.
void ScaleRowDown34_NEON(const uint8* src_ptr,
                         ptrdiff_t src_stride,
                         uint8* dst_ptr, int dst_width) {
  asm volatile (
  "1:                                                  \n"
    MEMACCESS(0)
    "ld4       {v0.8b,v1.8b,v2.8b,v3.8b}, [%0], #32                \n"  // src line 0
    "subs      %w2, %w2, #24                           \n"
    "orr       v2.16b, v3.16b, v3.16b                  \n"  // order v0, v1, v2
    MEMACCESS(1)
    "st3       {v0.8b,v1.8b,v2.8b}, [%1], #24                \n"
    "b.gt      1b                                      \n"
  : "+r"(src_ptr),          // %0
    "+r"(dst_ptr),          // %1
    "+r"(dst_width)         // %2
  :
  : "v0", "v1", "v2", "v3", "memory", "cc"
  );
}

void ScaleRowDown34_0_Box_NEON(const uint8* src_ptr,
                               ptrdiff_t src_stride,
                               uint8* dst_ptr, int dst_width) {
  asm volatile (
    "movi      v20.8b, #3                              \n"
    "add       %3, %3, %0                              \n"
  "1:                                                  \n"
    MEMACCESS(0)
    "ld4       {v0.8b,v1.8b,v2.8b,v3.8b}, [%0], #32                \n"  // src line 0
    MEMACCESS(3)
    "ld4       {v4.8b,v5.8b,v6.8b,v7.8b}, [%3], #32                \n"  // src line 1
    "subs         %w2, %w2, #24                        \n"

    // filter src line 0 with src line 1
    // expand chars to shorts to allow for room
    // when adding lines together
    "ushll     v16.8h, v4.8b, #0                       \n"
    "ushll     v17.8h, v5.8b, #0                       \n"
    "ushll     v18.8h, v6.8b, #0                       \n"
    "ushll     v19.8h, v7.8b, #0                       \n"

    // 3 * line_0 + line_1
    "umlal     v16.8h, v0.8b, v20.8b                   \n"
    "umlal     v17.8h, v1.8b, v20.8b                   \n"
    "umlal     v18.8h, v2.8b, v20.8b                   \n"
    "umlal     v19.8h, v3.8b, v20.8b                   \n"

    // (3 * line_0 + line_1) >> 2
    "uqrshrn   v0.8b, v16.8h, #2                       \n"
    "uqrshrn   v1.8b, v17.8h, #2                       \n"
    "uqrshrn   v2.8b, v18.8h, #2                       \n"
    "uqrshrn   v3.8b, v19.8h, #2                       \n"

    // a0 = (src[0] * 3 + s[1] * 1) >> 2
    "ushll     v16.8h, v1.8b, #0                       \n"
    "umlal     v16.8h, v0.8b, v20.8b                   \n"
    "uqrshrn   v0.8b, v16.8h, #2                       \n"

    // a1 = (src[1] * 1 + s[2] * 1) >> 1
    "urhadd    v1.8b, v1.8b, v2.8b                     \n"

    // a2 = (src[2] * 1 + s[3] * 3) >> 2
    "ushll     v16.8h, v2.8b, #0                       \n"
    "umlal     v16.8h, v3.8b, v20.8b                   \n"
    "uqrshrn   v2.8b, v16.8h, #2                       \n"

    MEMACCESS(1)
    "st3       {v0.8b,v1.8b,v2.8b}, [%1], #24                \n"

    "b.gt      1b                                      \n"
  : "+r"(src_ptr),          // %0
    "+r"(dst_ptr),          // %1
    "+r"(dst_width),        // %2
    "+r"(src_stride)        // %3
  :
  : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19",
    "v20", "memory", "cc"
  );
}

void ScaleRowDown34_1_Box_NEON(const uint8* src_ptr,
                               ptrdiff_t src_stride,
                               uint8* dst_ptr, int dst_width) {
  asm volatile (
    "movi      v20.8b, #3                              \n"
    "add       %3, %3, %0                              \n"
  "1:                                                  \n"
    MEMACCESS(0)
    "ld4       {v0.8b,v1.8b,v2.8b,v3.8b}, [%0], #32                \n"  // src line 0
    MEMACCESS(3)
    "ld4       {v4.8b,v5.8b,v6.8b,v7.8b}, [%3], #32                \n"  // src line 1
    "subs         %w2, %w2, #24                        \n"
    // average src line 0 with src line 1
    "urhadd    v0.8b, v0.8b, v4.8b                     \n"
    "urhadd    v1.8b, v1.8b, v5.8b                     \n"
    "urhadd    v2.8b, v2.8b, v6.8b                     \n"
    "urhadd    v3.8b, v3.8b, v7.8b                     \n"

    // a0 = (src[0] * 3 + s[1] * 1) >> 2
    "ushll     v4.8h, v1.8b, #0                        \n"
    "umlal     v4.8h, v0.8b, v20.8b                    \n"
    "uqrshrn   v0.8b, v4.8h, #2                        \n"

    // a1 = (src[1] * 1 + s[2] * 1) >> 1
    "urhadd    v1.8b, v1.8b, v2.8b                     \n"

    // a2 = (src[2] * 1 + s[3] * 3) >> 2
    "ushll     v4.8h, v2.8b, #0                        \n"
    "umlal     v4.8h, v3.8b, v20.8b                    \n"
    "uqrshrn   v2.8b, v4.8h, #2                        \n"

    MEMACCESS(1)
    "st3       {v0.8b,v1.8b,v2.8b}, [%1], #24                \n"
    "b.gt      1b                                      \n"
  : "+r"(src_ptr),          // %0
    "+r"(dst_ptr),          // %1
    "+r"(dst_width),        // %2
    "+r"(src_stride)        // %3
  :
  : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v20", "memory", "cc"
  );
}

static uvec8 kShuf38 =
  { 0, 3, 6, 8, 11, 14, 16, 19, 22, 24, 27, 30, 0, 0, 0, 0 };
static uvec8 kShuf38_2 =
  { 0, 16, 32, 2, 18, 33, 4, 20, 34, 6, 22, 35, 0, 0, 0, 0 };
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
    "ld1       {v3.16b}, [%3]                          \n"
  "1:                                                  \n"
    MEMACCESS(0)
    "ld1       {v0.16b,v1.16b}, [%0], #32             \n"
    "subs      %w2, %w2, #12                           \n"
    "tbl       v2.16b, {v0.16b,v1.16b}, v3.16b        \n"
    MEMACCESS(1)
    "st1       {v2.8b}, [%1], #8                       \n"
    MEMACCESS(1)
    "st1       {v2.s}[2], [%1], #4                     \n"
    "b.gt      1b                                      \n"
  : "+r"(src_ptr),          // %0
    "+r"(dst_ptr),          // %1
    "+r"(dst_width)         // %2
  : "r"(&kShuf38)           // %3
  : "v0", "v1", "v2", "v3", "memory", "cc"
  );
}

// 32x3 -> 12x1
void OMITFP ScaleRowDown38_3_Box_NEON(const uint8* src_ptr,
                                      ptrdiff_t src_stride,
                                      uint8* dst_ptr, int dst_width) {
  const uint8* src_ptr1 = src_ptr + src_stride * 2;
  ptrdiff_t tmp_src_stride = src_stride;

  asm volatile (
    MEMACCESS(5)
    "ld1       {v29.8h}, [%5]                          \n"
    MEMACCESS(6)
    "ld1       {v30.16b}, [%6]                         \n"
    MEMACCESS(7)
    "ld1       {v31.8h}, [%7]                          \n"
    "add       %2, %2, %0                              \n"
  "1:                                                  \n"

    // 00 40 01 41 02 42 03 43
    // 10 50 11 51 12 52 13 53
    // 20 60 21 61 22 62 23 63
    // 30 70 31 71 32 72 33 73
    MEMACCESS(0)
    "ld4       {v0.8b,v1.8b,v2.8b,v3.8b}, [%0], #32                \n"
    MEMACCESS(3)
    "ld4       {v4.8b,v5.8b,v6.8b,v7.8b}, [%2], #32                \n"
    MEMACCESS(4)
    "ld4       {v16.8b,v17.8b,v18.8b,v19.8b}, [%3], #32              \n"
    "subs      %w4, %w4, #12                           \n"

    // Shuffle the input data around to get align the data
    //  so adjacent data can be added. 0,1 - 2,3 - 4,5 - 6,7
    // 00 10 01 11 02 12 03 13
    // 40 50 41 51 42 52 43 53
    "trn1      v20.8b, v0.8b, v1.8b                    \n"
    "trn2      v21.8b, v0.8b, v1.8b                    \n"
    "trn1      v22.8b, v4.8b, v5.8b                    \n"
    "trn2      v23.8b, v4.8b, v5.8b                    \n"
    "trn1      v24.8b, v16.8b, v17.8b                  \n"
    "trn2      v25.8b, v16.8b, v17.8b                  \n"

    // 20 30 21 31 22 32 23 33
    // 60 70 61 71 62 72 63 73
    "trn1      v0.8b, v2.8b, v3.8b                     \n"
    "trn2      v1.8b, v2.8b, v3.8b                     \n"
    "trn1      v4.8b, v6.8b, v7.8b                     \n"
    "trn2      v5.8b, v6.8b, v7.8b                     \n"
    "trn1      v16.8b, v18.8b, v19.8b                  \n"
    "trn2      v17.8b, v18.8b, v19.8b                  \n"

    // 00+10 01+11 02+12 03+13
    // 40+50 41+51 42+52 43+53
    "uaddlp    v20.4h, v20.8b                          \n"
    "uaddlp    v21.4h, v21.8b                          \n"
    "uaddlp    v22.4h, v22.8b                          \n"
    "uaddlp    v23.4h, v23.8b                          \n"
    "uaddlp    v24.4h, v24.8b                          \n"
    "uaddlp    v25.4h, v25.8b                          \n"

    // 60+70 61+71 62+72 63+73
    "uaddlp    v1.4h, v1.8b                            \n"
    "uaddlp    v5.4h, v5.8b                            \n"
    "uaddlp    v17.4h, v17.8b                          \n"

    // combine source lines
    "add       v20.4h, v20.4h, v22.4h                  \n"
    "add       v21.4h, v21.4h, v23.4h                  \n"
    "add       v20.4h, v20.4h, v24.4h                  \n"
    "add       v21.4h, v21.4h, v25.4h                  \n"
    "add       v2.4h, v1.4h, v5.4h                     \n"
    "add       v2.4h, v2.4h, v17.4h                    \n"

    // dst_ptr[3] = (s[6 + st * 0] + s[7 + st * 0]
    //             + s[6 + st * 1] + s[7 + st * 1]
    //             + s[6 + st * 2] + s[7 + st * 2]) / 6
    "sqrdmulh  v2.8h, v2.8h, v29.8h                    \n"
    "xtn       v2.8b,  v2.8h                           \n"

    // Shuffle 2,3 reg around so that 2 can be added to the
    //  0,1 reg and 3 can be added to the 4,5 reg. This
    //  requires expanding from u8 to u16 as the 0,1 and 4,5
    //  registers are already expanded. Then do transposes
    //  to get aligned.
    // xx 20 xx 30 xx 21 xx 31 xx 22 xx 32 xx 23 xx 33
    "ushll     v16.8h, v16.8b, #0                      \n"
    "uaddl     v0.8h, v0.8b, v4.8b                     \n"

    // combine source lines
    "add       v0.8h, v0.8h, v16.8h                    \n"

    // xx 20 xx 21 xx 22 xx 23
    // xx 30 xx 31 xx 32 xx 33
    "trn1      v1.8h, v0.8h, v0.8h                     \n"
    "trn2      v4.8h, v0.8h, v0.8h                     \n"
    "xtn       v0.4h, v1.4s                            \n"
    "xtn       v4.4h, v4.4s                            \n"

    // 0+1+2, 3+4+5
    "add       v20.8h, v20.8h, v0.8h                   \n"
    "add       v21.8h, v21.8h, v4.8h                   \n"

    // Need to divide, but can't downshift as the the value
    //  isn't a power of 2. So multiply by 65536 / n
    //  and take the upper 16 bits.
    "sqrdmulh  v0.8h, v20.8h, v31.8h                   \n"
    "sqrdmulh  v1.8h, v21.8h, v31.8h                   \n"

    // Align for table lookup, vtbl requires registers to
    //  be adjacent
    "tbl       v3.16b, {v0.16b, v1.16b, v2.16b}, v30.16b \n"

    MEMACCESS(1)
    "st1       {v3.8b}, [%1], #8                       \n"
    MEMACCESS(1)
    "st1       {v3.s}[2], [%1], #4                     \n"
    "b.gt      1b                                      \n"
  : "+r"(src_ptr),          // %0
    "+r"(dst_ptr),          // %1
    "+r"(tmp_src_stride),   // %2
    "+r"(src_ptr1),         // %3
    "+r"(dst_width)         // %4
  : "r"(&kMult38_Div6),     // %5
    "r"(&kShuf38_2),        // %6
    "r"(&kMult38_Div9)      // %7
  : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17",
    "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v29",
    "v30", "v31", "memory", "cc"
  );
}

// 32x2 -> 12x1
void ScaleRowDown38_2_Box_NEON(const uint8* src_ptr,
                               ptrdiff_t src_stride,
                               uint8* dst_ptr, int dst_width) {
  // TODO(fbarchard): use src_stride directly for clang 3.5+.
  ptrdiff_t tmp_src_stride = src_stride;
  asm volatile (
    MEMACCESS(4)
    "ld1       {v30.8h}, [%4]                          \n"
    MEMACCESS(5)
    "ld1       {v31.16b}, [%5]                         \n"
    "add       %2, %2, %0                              \n"
  "1:                                                  \n"

    // 00 40 01 41 02 42 03 43
    // 10 50 11 51 12 52 13 53
    // 20 60 21 61 22 62 23 63
    // 30 70 31 71 32 72 33 73
    MEMACCESS(0)
    "ld4       {v0.8b,v1.8b,v2.8b,v3.8b}, [%0], #32                \n"
    MEMACCESS(3)
    "ld4       {v4.8b,v5.8b,v6.8b,v7.8b}, [%2], #32                \n"
    "subs      %w3, %w3, #12                           \n"

    // Shuffle the input data around to get align the data
    //  so adjacent data can be added. 0,1 - 2,3 - 4,5 - 6,7
    // 00 10 01 11 02 12 03 13
    // 40 50 41 51 42 52 43 53
    "trn1      v16.8b, v0.8b, v1.8b                    \n"
    "trn2      v17.8b, v0.8b, v1.8b                    \n"
    "trn1      v18.8b, v4.8b, v5.8b                    \n"
    "trn2      v19.8b, v4.8b, v5.8b                    \n"

    // 20 30 21 31 22 32 23 33
    // 60 70 61 71 62 72 63 73
    "trn1      v0.8b, v2.8b, v3.8b                     \n"
    "trn2      v1.8b, v2.8b, v3.8b                     \n"
    "trn1      v4.8b, v6.8b, v7.8b                     \n"
    "trn2      v5.8b, v6.8b, v7.8b                     \n"

    // 00+10 01+11 02+12 03+13
    // 40+50 41+51 42+52 43+53
    "uaddlp    v16.4h, v16.8b                          \n"
    "uaddlp    v17.4h, v17.8b                          \n"
    "uaddlp    v18.4h, v18.8b                          \n"
    "uaddlp    v19.4h, v19.8b                          \n"

    // 60+70 61+71 62+72 63+73
    "uaddlp    v1.4h, v1.8b                            \n"
    "uaddlp    v5.4h, v5.8b                            \n"

    // combine source lines
    "add       v16.4h, v16.4h, v18.4h                  \n"
    "add       v17.4h, v17.4h, v19.4h                  \n"
    "add       v2.4h, v1.4h, v5.4h                     \n"

    // dst_ptr[3] = (s[6] + s[7] + s[6+st] + s[7+st]) / 4
    "uqrshrn   v2.8b, v2.8h, #2                        \n"

    // Shuffle 2,3 reg around so that 2 can be added to the
    //  0,1 reg and 3 can be added to the 4,5 reg. This
    //  requires expanding from u8 to u16 as the 0,1 and 4,5
    //  registers are already expanded. Then do transposes
    //  to get aligned.
    // xx 20 xx 30 xx 21 xx 31 xx 22 xx 32 xx 23 xx 33

    // combine source lines
    "uaddl     v0.8h, v0.8b, v4.8b                     \n"

    // xx 20 xx 21 xx 22 xx 23
    // xx 30 xx 31 xx 32 xx 33
    "trn1      v1.8h, v0.8h, v0.8h                     \n"
    "trn2      v4.8h, v0.8h, v0.8h                     \n"
    "xtn       v0.4h, v1.4s                            \n"
    "xtn       v4.4h, v4.4s                            \n"

    // 0+1+2, 3+4+5
    "add       v16.8h, v16.8h, v0.8h                   \n"
    "add       v17.8h, v17.8h, v4.8h                   \n"

    // Need to divide, but can't downshift as the the value
    //  isn't a power of 2. So multiply by 65536 / n
    //  and take the upper 16 bits.
    "sqrdmulh  v0.8h, v16.8h, v30.8h                   \n"
    "sqrdmulh  v1.8h, v17.8h, v30.8h                   \n"

    // Align for table lookup, vtbl requires registers to
    //  be adjacent

    "tbl       v3.16b, {v0.16b, v1.16b, v2.16b}, v31.16b \n"

    MEMACCESS(1)
    "st1       {v3.8b}, [%1], #8                       \n"
    MEMACCESS(1)
    "st1       {v3.s}[2], [%1], #4                     \n"
    "b.gt      1b                                      \n"
  : "+r"(src_ptr),         // %0
    "+r"(dst_ptr),         // %1
    "+r"(tmp_src_stride),  // %2
    "+r"(dst_width)        // %3
  : "r"(&kMult38_Div6),    // %4
    "r"(&kShuf38_2)        // %5
  : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17",
    "v18", "v19", "v30", "v31", "memory", "cc"
  );
}

void ScaleAddRows_NEON(const uint8* src_ptr, ptrdiff_t src_stride,
                    uint16* dst_ptr, int src_width, int src_height) {
  const uint8* src_tmp;
  asm volatile (
  "1:                                          \n"
    "mov       %0, %1                          \n"
    "mov       w12, %w5                        \n"
    "eor       v2.16b, v2.16b, v2.16b          \n"
    "eor       v3.16b, v3.16b, v3.16b          \n"
  "2:                                          \n"
    // load 16 pixels into q0
    MEMACCESS(0)
    "ld1       {v0.16b}, [%0], %3              \n"
    "uaddw2    v3.8h, v3.8h, v0.16b            \n"
    "uaddw     v2.8h, v2.8h, v0.8b             \n"
    "subs      w12, w12, #1                    \n"
    "b.gt      2b                              \n"
    MEMACCESS(2)
    "st1      {v2.8h, v3.8h}, [%2], #32        \n"  // store pixels
    "add      %1, %1, #16                      \n"
    "subs     %w4, %w4, #16                    \n"  // 16 processed per loop
    "b.gt     1b                               \n"
  : "=&r"(src_tmp),    // %0
    "+r"(src_ptr),     // %1
    "+r"(dst_ptr),     // %2
    "+r"(src_stride),  // %3
    "+r"(src_width),   // %4
    "+r"(src_height)   // %5
  :
  : "memory", "cc", "w12", "v0", "v1", "v2", "v3"  // Clobber List
  );
}

// TODO(Yang Zhang): Investigate less load instructions for
// the x/dx stepping
#define LOAD2_DATA8_LANE(n)                                    \
    "lsr        %5, %3, #16                    \n"             \
    "add        %6, %1, %5                    \n"              \
    "add        %3, %3, %4                     \n"             \
    MEMACCESS(6)                                               \
    "ld2        {v4.b, v5.b}["#n"], [%6]      \n"

// The NEON version mimics this formula (from row_common.cc):
// #define BLENDER(a, b, f) (uint8)((int)(a) +
//    ((((int)((f)) * ((int)(b) - (int)(a))) + 0x8000) >> 16))

void ScaleFilterCols_NEON(uint8* dst_ptr, const uint8* src_ptr,
                          int dst_width, int x, int dx) {
  int dx_offset[4] = {0, 1, 2, 3};
  int* tmp = dx_offset;
  const uint8* src_tmp = src_ptr;
  int64 dst_width64 = (int64) dst_width;  // Work around ios 64 bit warning.
  int64 x64 = (int64) x;
  int64 dx64 = (int64) dx;
  asm volatile (
    "dup        v0.4s, %w3                     \n"  // x
    "dup        v1.4s, %w4                     \n"  // dx
    "ld1        {v2.4s}, [%5]                  \n"  // 0 1 2 3
    "shl        v3.4s, v1.4s, #2               \n"  // 4 * dx
    "mul        v1.4s, v1.4s, v2.4s            \n"
    // x         , x + 1 * dx, x + 2 * dx, x + 3 * dx
    "add        v1.4s, v1.4s, v0.4s            \n"
    // x + 4 * dx, x + 5 * dx, x + 6 * dx, x + 7 * dx
    "add        v2.4s, v1.4s, v3.4s            \n"
    "shl        v0.4s, v3.4s, #1               \n"  // 8 * dx
  "1:                                          \n"
    LOAD2_DATA8_LANE(0)
    LOAD2_DATA8_LANE(1)
    LOAD2_DATA8_LANE(2)
    LOAD2_DATA8_LANE(3)
    LOAD2_DATA8_LANE(4)
    LOAD2_DATA8_LANE(5)
    LOAD2_DATA8_LANE(6)
    LOAD2_DATA8_LANE(7)
    "mov       v6.16b, v1.16b                  \n"
    "mov       v7.16b, v2.16b                  \n"
    "uzp1      v6.8h, v6.8h, v7.8h             \n"
    "ushll     v4.8h, v4.8b, #0                \n"
    "ushll     v5.8h, v5.8b, #0                \n"
    "ssubl     v16.4s, v5.4h, v4.4h            \n"
    "ssubl2    v17.4s, v5.8h, v4.8h            \n"
    "ushll     v7.4s, v6.4h, #0                \n"
    "ushll2    v6.4s, v6.8h, #0                \n"
    "mul       v16.4s, v16.4s, v7.4s           \n"
    "mul       v17.4s, v17.4s, v6.4s           \n"
    "rshrn     v6.4h, v16.4s, #16              \n"
    "rshrn2    v6.8h, v17.4s, #16              \n"
    "add       v4.8h, v4.8h, v6.8h             \n"
    "xtn       v4.8b, v4.8h                    \n"

    MEMACCESS(0)
    "st1       {v4.8b}, [%0], #8               \n"  // store pixels
    "add       v1.4s, v1.4s, v0.4s             \n"
    "add       v2.4s, v2.4s, v0.4s             \n"
    "subs      %w2, %w2, #8                    \n"  // 8 processed per loop
    "b.gt      1b                              \n"
  : "+r"(dst_ptr),          // %0
    "+r"(src_ptr),          // %1
    "+r"(dst_width64),      // %2
    "+r"(x64),              // %3
    "+r"(dx64),             // %4
    "+r"(tmp),              // %5
    "+r"(src_tmp)           // %6
  :
  : "memory", "cc", "v0", "v1", "v2", "v3",
    "v4", "v5", "v6", "v7", "v16", "v17"
  );
}

#undef LOAD2_DATA8_LANE

// 16x2 -> 16x1
void ScaleFilterRows_NEON(uint8* dst_ptr,
                          const uint8* src_ptr, ptrdiff_t src_stride,
                          int dst_width, int source_y_fraction) {
    int y_fraction = 256 - source_y_fraction;
  asm volatile (
    "cmp          %w4, #0                      \n"
    "b.eq         100f                         \n"
    "add          %2, %2, %1                   \n"
    "cmp          %w4, #64                     \n"
    "b.eq         75f                          \n"
    "cmp          %w4, #128                    \n"
    "b.eq         50f                          \n"
    "cmp          %w4, #192                    \n"
    "b.eq         25f                          \n"

    "dup          v5.8b, %w4                   \n"
    "dup          v4.8b, %w5                   \n"
    // General purpose row blend.
  "1:                                          \n"
    MEMACCESS(1)
    "ld1          {v0.16b}, [%1], #16          \n"
    MEMACCESS(2)
    "ld1          {v1.16b}, [%2], #16          \n"
    "subs         %w3, %w3, #16                \n"
    "umull        v6.8h, v0.8b, v4.8b          \n"
    "umull2       v7.8h, v0.16b, v4.16b        \n"
    "umlal        v6.8h, v1.8b, v5.8b          \n"
    "umlal2       v7.8h, v1.16b, v5.16b        \n"
    "rshrn        v0.8b, v6.8h, #8             \n"
    "rshrn2       v0.16b, v7.8h, #8            \n"
    MEMACCESS(0)
    "st1          {v0.16b}, [%0], #16          \n"
    "b.gt         1b                           \n"
    "b            99f                          \n"

    // Blend 25 / 75.
  "25:                                         \n"
    MEMACCESS(1)
    "ld1          {v0.16b}, [%1], #16          \n"
    MEMACCESS(2)
    "ld1          {v1.16b}, [%2], #16          \n"
    "subs         %w3, %w3, #16                \n"
    "urhadd       v0.16b, v0.16b, v1.16b       \n"
    "urhadd       v0.16b, v0.16b, v1.16b       \n"
    MEMACCESS(0)
    "st1          {v0.16b}, [%0], #16          \n"
    "b.gt         25b                          \n"
    "b            99f                          \n"

    // Blend 50 / 50.
  "50:                                         \n"
    MEMACCESS(1)
    "ld1          {v0.16b}, [%1], #16          \n"
    MEMACCESS(2)
    "ld1          {v1.16b}, [%2], #16          \n"
    "subs         %w3, %w3, #16                \n"
    "urhadd       v0.16b, v0.16b, v1.16b       \n"
    MEMACCESS(0)
    "st1          {v0.16b}, [%0], #16          \n"
    "b.gt         50b                          \n"
    "b            99f                          \n"

    // Blend 75 / 25.
  "75:                                         \n"
    MEMACCESS(1)
    "ld1          {v1.16b}, [%1], #16          \n"
    MEMACCESS(2)
    "ld1          {v0.16b}, [%2], #16          \n"
    "subs         %w3, %w3, #16                \n"
    "urhadd       v0.16b, v0.16b, v1.16b       \n"
    "urhadd       v0.16b, v0.16b, v1.16b       \n"
    MEMACCESS(0)
    "st1          {v0.16b}, [%0], #16          \n"
    "b.gt         75b                          \n"
    "b            99f                          \n"

    // Blend 100 / 0 - Copy row unchanged.
  "100:                                        \n"
    MEMACCESS(1)
    "ld1          {v0.16b}, [%1], #16          \n"
    "subs         %w3, %w3, #16                \n"
    MEMACCESS(0)
    "st1          {v0.16b}, [%0], #16          \n"
    "b.gt         100b                         \n"

  "99:                                         \n"
    MEMACCESS(0)
    "st1          {v0.b}[15], [%0]             \n"
  : "+r"(dst_ptr),          // %0
    "+r"(src_ptr),          // %1
    "+r"(src_stride),       // %2
    "+r"(dst_width),        // %3
    "+r"(source_y_fraction),// %4
    "+r"(y_fraction)        // %5
  :
  : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "memory", "cc"
  );
}

void ScaleARGBRowDown2_NEON(const uint8* src_ptr, ptrdiff_t src_stride,
                            uint8* dst, int dst_width) {
  asm volatile (
  "1:                                          \n"
    // load even pixels into q0, odd into q1
    MEMACCESS (0)
    "ld2        {v0.4s, v1.4s}, [%0], #32      \n"
    MEMACCESS (0)
    "ld2        {v2.4s, v3.4s}, [%0], #32      \n"
    "subs       %w2, %w2, #8                   \n"  // 8 processed per loop
    MEMACCESS (1)
    "st1        {v1.16b}, [%1], #16            \n"  // store odd pixels
    MEMACCESS (1)
    "st1        {v3.16b}, [%1], #16            \n"
    "b.gt       1b                             \n"
  : "+r" (src_ptr),          // %0
    "+r" (dst),              // %1
    "+r" (dst_width)         // %2
  :
  : "memory", "cc", "v0", "v1", "v2", "v3"  // Clobber List
  );
}

void ScaleARGBRowDown2Linear_NEON(const uint8* src_argb, ptrdiff_t src_stride,
                                  uint8* dst_argb, int dst_width) {
  asm volatile (
  "1:                                          \n"
    MEMACCESS (0)
    // load 8 ARGB pixels.
    "ld4        {v0.16b,v1.16b,v2.16b,v3.16b}, [%0], #64   \n"
    "subs       %w2, %w2, #8                   \n"  // 8 processed per loop.
    "uaddlp     v0.8h, v0.16b                  \n"  // B 16 bytes -> 8 shorts.
    "uaddlp     v1.8h, v1.16b                  \n"  // G 16 bytes -> 8 shorts.
    "uaddlp     v2.8h, v2.16b                  \n"  // R 16 bytes -> 8 shorts.
    "uaddlp     v3.8h, v3.16b                  \n"  // A 16 bytes -> 8 shorts.
    "rshrn      v0.8b, v0.8h, #1               \n"  // downshift, round and pack
    "rshrn      v1.8b, v1.8h, #1               \n"
    "rshrn      v2.8b, v2.8h, #1               \n"
    "rshrn      v3.8b, v3.8h, #1               \n"
    MEMACCESS (1)
    "st4        {v0.8b,v1.8b,v2.8b,v3.8b}, [%1], #32     \n"
    "b.gt       1b                             \n"
  : "+r"(src_argb),         // %0
    "+r"(dst_argb),         // %1
    "+r"(dst_width)         // %2
  :
  : "memory", "cc", "v0", "v1", "v2", "v3"    // Clobber List
  );
}

void ScaleARGBRowDown2Box_NEON(const uint8* src_ptr, ptrdiff_t src_stride,
                               uint8* dst, int dst_width) {
  asm volatile (
    // change the stride to row 2 pointer
    "add        %1, %1, %0                     \n"
  "1:                                          \n"
    MEMACCESS (0)
    "ld4        {v0.16b,v1.16b,v2.16b,v3.16b}, [%0], #64   \n"  // load 8 ARGB pixels.
    "subs       %w3, %w3, #8                   \n"  // 8 processed per loop.
    "uaddlp     v0.8h, v0.16b                  \n"  // B 16 bytes -> 8 shorts.
    "uaddlp     v1.8h, v1.16b                  \n"  // G 16 bytes -> 8 shorts.
    "uaddlp     v2.8h, v2.16b                  \n"  // R 16 bytes -> 8 shorts.
    "uaddlp     v3.8h, v3.16b                  \n"  // A 16 bytes -> 8 shorts.
    MEMACCESS (1)
    "ld4        {v16.16b,v17.16b,v18.16b,v19.16b}, [%1], #64 \n"  // load 8 more ARGB pixels.
    "uadalp     v0.8h, v16.16b                 \n"  // B 16 bytes -> 8 shorts.
    "uadalp     v1.8h, v17.16b                 \n"  // G 16 bytes -> 8 shorts.
    "uadalp     v2.8h, v18.16b                 \n"  // R 16 bytes -> 8 shorts.
    "uadalp     v3.8h, v19.16b                 \n"  // A 16 bytes -> 8 shorts.
    "rshrn      v0.8b, v0.8h, #2               \n"  // downshift, round and pack
    "rshrn      v1.8b, v1.8h, #2               \n"
    "rshrn      v2.8b, v2.8h, #2               \n"
    "rshrn      v3.8b, v3.8h, #2               \n"
    MEMACCESS (2)
    "st4        {v0.8b,v1.8b,v2.8b,v3.8b}, [%2], #32     \n"
    "b.gt       1b                             \n"
  : "+r" (src_ptr),          // %0
    "+r" (src_stride),       // %1
    "+r" (dst),              // %2
    "+r" (dst_width)         // %3
  :
  : "memory", "cc", "v0", "v1", "v2", "v3", "v16", "v17", "v18", "v19"
  );
}

// Reads 4 pixels at a time.
// Alignment requirement: src_argb 4 byte aligned.
void ScaleARGBRowDownEven_NEON(const uint8* src_argb,  ptrdiff_t src_stride,
                               int src_stepx, uint8* dst_argb, int dst_width) {
  asm volatile (
  "1:                                          \n"
    MEMACCESS(0)
    "ld1        {v0.s}[0], [%0], %3            \n"
    MEMACCESS(0)
    "ld1        {v0.s}[1], [%0], %3            \n"
    MEMACCESS(0)
    "ld1        {v0.s}[2], [%0], %3            \n"
    MEMACCESS(0)
    "ld1        {v0.s}[3], [%0], %3            \n"
    "subs       %w2, %w2, #4                   \n"  // 4 pixels per loop.
    MEMACCESS(1)
    "st1        {v0.16b}, [%1], #16            \n"
    "b.gt       1b                             \n"
  : "+r"(src_argb),    // %0
    "+r"(dst_argb),    // %1
    "+r"(dst_width)    // %2
  : "r"((int64)(src_stepx * 4)) // %3
  : "memory", "cc", "v0"
  );
}

// Reads 4 pixels at a time.
// Alignment requirement: src_argb 4 byte aligned.
// TODO(Yang Zhang): Might be worth another optimization pass in future.
// It could be upgraded to 8 pixels at a time to start with.
void ScaleARGBRowDownEvenBox_NEON(const uint8* src_argb, ptrdiff_t src_stride,
                                  int src_stepx,
                                  uint8* dst_argb, int dst_width) {
  asm volatile (
    "add        %1, %1, %0                     \n"
  "1:                                          \n"
    MEMACCESS(0)
    "ld1        {v0.8b}, [%0], %4              \n"  // Read 4 2x2 blocks -> 2x1
    MEMACCESS(1)
    "ld1        {v1.8b}, [%1], %4              \n"
    MEMACCESS(0)
    "ld1        {v2.8b}, [%0], %4              \n"
    MEMACCESS(1)
    "ld1        {v3.8b}, [%1], %4              \n"
    MEMACCESS(0)
    "ld1        {v4.8b}, [%0], %4              \n"
    MEMACCESS(1)
    "ld1        {v5.8b}, [%1], %4              \n"
    MEMACCESS(0)
    "ld1        {v6.8b}, [%0], %4              \n"
    MEMACCESS(1)
    "ld1        {v7.8b}, [%1], %4              \n"
    "uaddl      v0.8h, v0.8b, v1.8b            \n"
    "uaddl      v2.8h, v2.8b, v3.8b            \n"
    "uaddl      v4.8h, v4.8b, v5.8b            \n"
    "uaddl      v6.8h, v6.8b, v7.8b            \n"
    "mov        v16.d[1], v0.d[1]              \n"  // ab_cd -> ac_bd
    "mov        v0.d[1], v2.d[0]               \n"
    "mov        v2.d[0], v16.d[1]              \n"
    "mov        v16.d[1], v4.d[1]              \n"  // ef_gh -> eg_fh
    "mov        v4.d[1], v6.d[0]               \n"
    "mov        v6.d[0], v16.d[1]              \n"
    "add        v0.8h, v0.8h, v2.8h            \n"  // (a+b)_(c+d)
    "add        v4.8h, v4.8h, v6.8h            \n"  // (e+f)_(g+h)
    "rshrn      v0.8b, v0.8h, #2               \n"  // first 2 pixels.
    "rshrn2     v0.16b, v4.8h, #2              \n"  // next 2 pixels.
    "subs       %w3, %w3, #4                   \n"  // 4 pixels per loop.
    MEMACCESS(2)
    "st1     {v0.16b}, [%2], #16               \n"
    "b.gt       1b                             \n"
  : "+r"(src_argb),    // %0
    "+r"(src_stride),  // %1
    "+r"(dst_argb),    // %2
    "+r"(dst_width)    // %3
  : "r"((int64)(src_stepx * 4)) // %4
  : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16"
  );
}

// TODO(Yang Zhang): Investigate less load instructions for
// the x/dx stepping
#define LOAD1_DATA32_LANE(vn, n)                               \
    "lsr        %5, %3, #16                    \n"             \
    "add        %6, %1, %5, lsl #2             \n"             \
    "add        %3, %3, %4                     \n"             \
    MEMACCESS(6)                                               \
    "ld1        {"#vn".s}["#n"], [%6]          \n"

void ScaleARGBCols_NEON(uint8* dst_argb, const uint8* src_argb,
                        int dst_width, int x, int dx) {
  const uint8* src_tmp = src_argb;
  int64 dst_width64 = (int64) dst_width;  // Work around ios 64 bit warning.
  int64 x64 = (int64) x;
  int64 dx64 = (int64) dx;
  int64 tmp64;
  asm volatile (
  "1:                                          \n"
    LOAD1_DATA32_LANE(v0, 0)
    LOAD1_DATA32_LANE(v0, 1)
    LOAD1_DATA32_LANE(v0, 2)
    LOAD1_DATA32_LANE(v0, 3)
    LOAD1_DATA32_LANE(v1, 0)
    LOAD1_DATA32_LANE(v1, 1)
    LOAD1_DATA32_LANE(v1, 2)
    LOAD1_DATA32_LANE(v1, 3)

    MEMACCESS(0)
    "st1        {v0.4s, v1.4s}, [%0], #32      \n"  // store pixels
    "subs       %w2, %w2, #8                   \n"  // 8 processed per loop
    "b.gt        1b                            \n"
  : "+r"(dst_argb),     // %0
    "+r"(src_argb),     // %1
    "+r"(dst_width64),  // %2
    "+r"(x64),          // %3
    "+r"(dx64),         // %4
    "=&r"(tmp64),       // %5
    "+r"(src_tmp)       // %6
  :
  : "memory", "cc", "v0", "v1"
  );
}

#undef LOAD1_DATA32_LANE

// TODO(Yang Zhang): Investigate less load instructions for
// the x/dx stepping
#define LOAD2_DATA32_LANE(vn1, vn2, n)                         \
    "lsr        %5, %3, #16                           \n"      \
    "add        %6, %1, %5, lsl #2                    \n"      \
    "add        %3, %3, %4                            \n"      \
    MEMACCESS(6)                                               \
    "ld2        {"#vn1".s, "#vn2".s}["#n"], [%6]      \n"

void ScaleARGBFilterCols_NEON(uint8* dst_argb, const uint8* src_argb,
                              int dst_width, int x, int dx) {
  int dx_offset[4] = {0, 1, 2, 3};
  int* tmp = dx_offset;
  const uint8* src_tmp = src_argb;
  int64 dst_width64 = (int64) dst_width;  // Work around ios 64 bit warning.
  int64 x64 = (int64) x;
  int64 dx64 = (int64) dx;
  asm volatile (
    "dup        v0.4s, %w3                     \n"  // x
    "dup        v1.4s, %w4                     \n"  // dx
    "ld1        {v2.4s}, [%5]                  \n"  // 0 1 2 3
    "shl        v6.4s, v1.4s, #2               \n"  // 4 * dx
    "mul        v1.4s, v1.4s, v2.4s            \n"
    "movi       v3.16b, #0x7f                  \n"  // 0x7F
    "movi       v4.8h, #0x7f                   \n"  // 0x7F
    // x         , x + 1 * dx, x + 2 * dx, x + 3 * dx
    "add        v5.4s, v1.4s, v0.4s            \n"
  "1:                                          \n"
    // d0, d1: a
    // d2, d3: b
    LOAD2_DATA32_LANE(v0, v1, 0)
    LOAD2_DATA32_LANE(v0, v1, 1)
    LOAD2_DATA32_LANE(v0, v1, 2)
    LOAD2_DATA32_LANE(v0, v1, 3)
    "shrn       v2.4h, v5.4s, #9               \n"
    "and        v2.8b, v2.8b, v4.8b            \n"
    "dup        v16.8b, v2.b[0]                \n"
    "dup        v17.8b, v2.b[2]                \n"
    "dup        v18.8b, v2.b[4]                \n"
    "dup        v19.8b, v2.b[6]                \n"
    "ext        v2.8b, v16.8b, v17.8b, #4      \n"
    "ext        v17.8b, v18.8b, v19.8b, #4     \n"
    "ins        v2.d[1], v17.d[0]              \n"  // f
    "eor        v7.16b, v2.16b, v3.16b         \n"  // 0x7f ^ f
    "umull      v16.8h, v0.8b, v7.8b           \n"
    "umull2     v17.8h, v0.16b, v7.16b         \n"
    "umull      v18.8h, v1.8b, v2.8b           \n"
    "umull2     v19.8h, v1.16b, v2.16b         \n"
    "add        v16.8h, v16.8h, v18.8h         \n"
    "add        v17.8h, v17.8h, v19.8h         \n"
    "shrn       v0.8b, v16.8h, #7              \n"
    "shrn2      v0.16b, v17.8h, #7             \n"

    MEMACCESS(0)
    "st1     {v0.4s}, [%0], #16                \n"  // store pixels
    "add     v5.4s, v5.4s, v6.4s               \n"
    "subs    %w2, %w2, #4                      \n"  // 4 processed per loop
    "b.gt    1b                                \n"
  : "+r"(dst_argb),         // %0
    "+r"(src_argb),         // %1
    "+r"(dst_width64),      // %2
    "+r"(x64),              // %3
    "+r"(dx64),             // %4
    "+r"(tmp),              // %5
    "+r"(src_tmp)           // %6
  :
  : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5",
    "v6", "v7", "v16", "v17", "v18", "v19"
  );
}

#undef LOAD2_DATA32_LANE

#endif  // !defined(LIBYUV_DISABLE_NEON) && defined(__aarch64__)

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
