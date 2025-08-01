/*
 *  Copyright 2014 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "libyuv/row.h"
#include "libyuv/scale.h"
#include "libyuv/scale_row.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// This module is for GCC Neon armv8 64 bit.
#if !defined(LIBYUV_DISABLE_NEON) && defined(__aarch64__)

// Read 32x1 throw away even pixels, and write 16x1.
void ScaleRowDown2_NEON(const uint8_t* src_ptr,
                        ptrdiff_t src_stride,
                        uint8_t* dst,
                        int dst_width) {
  (void)src_stride;
  asm volatile(
      "1:          \n"
      // load even pixels into v0, odd into v1
      "ld2         {v0.16b,v1.16b}, [%0], #32    \n"
      "subs        %w2, %w2, #16                 \n"  // 16 processed per loop
      "prfm        pldl1keep, [%0, 448]          \n"  // prefetch 7 lines ahead
      "st1         {v1.16b}, [%1], #16           \n"  // store odd pixels
      "b.gt        1b                            \n"
      : "+r"(src_ptr),   // %0
        "+r"(dst),       // %1
        "+r"(dst_width)  // %2
      :
      : "memory", "cc", "v0", "v1"  // Clobber List
  );
}

// Read 32x1 average down and write 16x1.
void ScaleRowDown2Linear_NEON(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst,
                              int dst_width) {
  (void)src_stride;
  asm volatile(
      "1:          \n"
      // load even pixels into v0, odd into v1
      "ld2         {v0.16b,v1.16b}, [%0], #32    \n"
      "subs        %w2, %w2, #16                 \n"  // 16 processed per loop
      "urhadd      v0.16b, v0.16b, v1.16b        \n"  // rounding half add
      "prfm        pldl1keep, [%0, 448]          \n"  // prefetch 7 lines ahead
      "st1         {v0.16b}, [%1], #16           \n"
      "b.gt        1b                            \n"
      : "+r"(src_ptr),   // %0
        "+r"(dst),       // %1
        "+r"(dst_width)  // %2
      :
      : "memory", "cc", "v0", "v1"  // Clobber List
  );
}

// Read 32x2 average down and write 16x1.
void ScaleRowDown2Box_NEON(const uint8_t* src_ptr,
                           ptrdiff_t src_stride,
                           uint8_t* dst,
                           int dst_width) {
  asm volatile(
      // change the stride to row 2 pointer
      "add         %1, %1, %0                    \n"
      "1:          \n"
      "ld1         {v0.16b, v1.16b}, [%0], #32   \n"  // load row 1 and post inc
      "ld1         {v2.16b, v3.16b}, [%1], #32   \n"  // load row 2 and post inc
      "subs        %w3, %w3, #16                 \n"  // 16 processed per loop
      "uaddlp      v0.8h, v0.16b                 \n"  // row 1 add adjacent
      "prfm        pldl1keep, [%0, 448]          \n"  // prefetch 7 lines ahead
      "uaddlp      v1.8h, v1.16b                 \n"
      "prfm        pldl1keep, [%1, 448]          \n"
      "uadalp      v0.8h, v2.16b                 \n"  // += row 2 add adjacent
      "uadalp      v1.8h, v3.16b                 \n"
      "rshrn       v0.8b, v0.8h, #2              \n"  // round and pack
      "rshrn2      v0.16b, v1.8h, #2             \n"
      "st1         {v0.16b}, [%2], #16           \n"
      "b.gt        1b                            \n"
      : "+r"(src_ptr),     // %0
        "+r"(src_stride),  // %1
        "+r"(dst),         // %2
        "+r"(dst_width)    // %3
      :
      : "memory", "cc", "v0", "v1", "v2", "v3"  // Clobber List
  );
}

void ScaleRowDown4_NEON(const uint8_t* src_ptr,
                        ptrdiff_t src_stride,
                        uint8_t* dst_ptr,
                        int dst_width) {
  (void)src_stride;
  asm volatile(
      "1:          \n"
      "ld4         {v0.16b,v1.16b,v2.16b,v3.16b}, [%0], #64 \n"  // src line 0
      "subs        %w2, %w2, #16                 \n"  // 16 processed per loop
      "prfm        pldl1keep, [%0, 448]          \n"  // prefetch 7 lines ahead
      "st1         {v2.16b}, [%1], #16           \n"
      "b.gt        1b                            \n"
      : "+r"(src_ptr),   // %0
        "+r"(dst_ptr),   // %1
        "+r"(dst_width)  // %2
      :
      : "memory", "cc", "v0", "v1", "v2", "v3");
}

void ScaleRowDown4Box_NEON(const uint8_t* src_ptr,
                           ptrdiff_t src_stride,
                           uint8_t* dst_ptr,
                           int dst_width) {
  const uint8_t* src_ptr1 = src_ptr + src_stride;
  const uint8_t* src_ptr2 = src_ptr + src_stride * 2;
  const uint8_t* src_ptr3 = src_ptr + src_stride * 3;
  asm volatile(
      "1:          \n"
      "ldp         q0, q4, [%0], #32             \n"  // load up 16x8
      "ldp         q1, q5, [%2], #32             \n"
      "ldp         q2, q6, [%3], #32             \n"
      "ldp         q3, q7, [%4], #32             \n"
      "subs        %w5, %w5, #8                  \n"
      "uaddlp      v0.8h, v0.16b                 \n"
      "uaddlp      v4.8h, v4.16b                 \n"
      "prfm        pldl1keep, [%0, 448]          \n"  // prefetch 7 lines ahead
      "uadalp      v0.8h, v1.16b                 \n"
      "uadalp      v4.8h, v5.16b                 \n"
      "prfm        pldl1keep, [%2, 448]          \n"
      "uadalp      v0.8h, v2.16b                 \n"
      "uadalp      v4.8h, v6.16b                 \n"
      "prfm        pldl1keep, [%3, 448]          \n"
      "uadalp      v0.8h, v3.16b                 \n"
      "uadalp      v4.8h, v7.16b                 \n"
      "prfm        pldl1keep, [%4, 448]          \n"
      "addp        v0.8h, v0.8h, v4.8h           \n"
      "rshrn       v0.8b, v0.8h, #4              \n"  // divide by 16 w/rounding
      "str         d0, [%1], #8                  \n"
      "b.gt        1b                            \n"
      : "+r"(src_ptr),   // %0
        "+r"(dst_ptr),   // %1
        "+r"(src_ptr1),  // %2
        "+r"(src_ptr2),  // %3
        "+r"(src_ptr3),  // %4
        "+r"(dst_width)  // %5
      :
      : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
}

static const uvec8 kShuf34_0 = {
    0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20,
};
static const uvec8 kShuf34_1 = {
    5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20, 21, 23, 24, 25,
};
static const uvec8 kShuf34_2 = {
    11, 12, 13, 15, 16, 17, 19, 20, 21, 23, 24, 25, 27, 28, 29, 31,
};

// Down scale from 4 to 3 pixels. Point samples 64 pixels to 48 pixels.
void ScaleRowDown34_NEON(const uint8_t* src_ptr,
                         ptrdiff_t src_stride,
                         uint8_t* dst_ptr,
                         int dst_width) {
  (void)src_stride;
  asm volatile(
      "ld1         {v29.16b}, [%[kShuf34_0]]     \n"
      "ld1         {v30.16b}, [%[kShuf34_1]]     \n"
      "ld1         {v31.16b}, [%[kShuf34_2]]     \n"
      "1:          \n"
      "ld1         {v0.16b,v1.16b,v2.16b,v3.16b}, [%[src_ptr]], #64 \n"
      "subs        %w[width], %w[width], #48     \n"
      "tbl         v0.16b, {v0.16b, v1.16b}, v29.16b \n"
      "prfm        pldl1keep, [%[src_ptr], 448]  \n"
      "tbl         v1.16b, {v1.16b, v2.16b}, v30.16b \n"
      "tbl         v2.16b, {v2.16b, v3.16b}, v31.16b \n"
      "st1         {v0.16b,v1.16b,v2.16b}, [%[dst_ptr]], #48 \n"
      "b.gt        1b                            \n"
      : [src_ptr] "+r"(src_ptr),      // %[src_ptr]
        [dst_ptr] "+r"(dst_ptr),      // %[dst_ptr]
        [width] "+r"(dst_width)       // %[width]
      : [kShuf34_0] "r"(&kShuf34_0),  // %[kShuf34_0]
        [kShuf34_1] "r"(&kShuf34_1),  // %[kShuf34_1]
        [kShuf34_2] "r"(&kShuf34_2)   // %[kShuf34_2]
      : "memory", "cc", "v0", "v1", "v2", "v3", "v29", "v30", "v31");
}

void ScaleRowDown34_0_Box_NEON(const uint8_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint8_t* dst_ptr,
                               int dst_width) {
  asm volatile(
      "movi        v24.16b, #3                   \n"
      "add         %3, %3, %0                    \n"

      "1:          \n"
      "ld4         {v0.16b,v1.16b,v2.16b,v3.16b}, [%0], #64 \n"  // src line 0
      "ld4         {v4.16b,v5.16b,v6.16b,v7.16b}, [%3], #64 \n"  // src line 1
      "subs        %w2, %w2, #48                 \n"

      // filter src line 0 with src line 1
      // expand chars to shorts to allow for room
      // when adding lines together
      "ushll       v16.8h, v4.8b, #0             \n"
      "ushll       v17.8h, v5.8b, #0             \n"
      "ushll       v18.8h, v6.8b, #0             \n"
      "ushll       v19.8h, v7.8b, #0             \n"
      "ushll2      v20.8h, v4.16b, #0            \n"
      "ushll2      v21.8h, v5.16b, #0            \n"
      "ushll2      v22.8h, v6.16b, #0            \n"
      "ushll2      v23.8h, v7.16b, #0            \n"

      // 3 * line_0 + line_1
      "umlal       v16.8h, v0.8b, v24.8b         \n"
      "umlal       v17.8h, v1.8b, v24.8b         \n"
      "umlal       v18.8h, v2.8b, v24.8b         \n"
      "umlal       v19.8h, v3.8b, v24.8b         \n"
      "umlal2      v20.8h, v0.16b, v24.16b       \n"
      "umlal2      v21.8h, v1.16b, v24.16b       \n"
      "umlal2      v22.8h, v2.16b, v24.16b       \n"
      "umlal2      v23.8h, v3.16b, v24.16b       \n"
      "prfm        pldl1keep, [%0, 448]          \n"  // prefetch 7 lines ahead

      // (3 * line_0 + line_1 + 2) >> 2
      "uqrshrn     v0.8b, v16.8h, #2             \n"
      "uqrshrn     v1.8b, v17.8h, #2             \n"
      "uqrshrn     v2.8b, v18.8h, #2             \n"
      "uqrshrn     v3.8b, v19.8h, #2             \n"
      "uqrshrn2    v0.16b, v20.8h, #2            \n"
      "uqrshrn2    v1.16b, v21.8h, #2            \n"
      "uqrshrn2    v2.16b, v22.8h, #2            \n"
      "uqrshrn2    v3.16b, v23.8h, #2            \n"
      "prfm        pldl1keep, [%3, 448]          \n"

      // a0 = (src[0] * 3 + s[1] * 1 + 2) >> 2
      "ushll       v16.8h, v1.8b, #0             \n"
      "ushll2      v17.8h, v1.16b, #0            \n"
      "umlal       v16.8h, v0.8b, v24.8b         \n"
      "umlal2      v17.8h, v0.16b, v24.16b       \n"
      "uqrshrn     v0.8b, v16.8h, #2             \n"
      "uqrshrn2    v0.16b, v17.8h, #2            \n"

      // a1 = (src[1] * 1 + s[2] * 1 + 1) >> 1
      "urhadd      v1.16b, v1.16b, v2.16b        \n"

      // a2 = (src[2] * 1 + s[3] * 3 + 2) >> 2
      "ushll       v16.8h, v2.8b, #0             \n"
      "ushll2      v17.8h, v2.16b, #0            \n"
      "umlal       v16.8h, v3.8b, v24.8b         \n"
      "umlal2      v17.8h, v3.16b, v24.16b       \n"
      "uqrshrn     v2.8b, v16.8h, #2             \n"
      "uqrshrn2    v2.16b, v17.8h, #2            \n"

      "st3         {v0.16b,v1.16b,v2.16b}, [%1], #48 \n"

      "b.gt        1b                            \n"
      : "+r"(src_ptr),    // %0
        "+r"(dst_ptr),    // %1
        "+r"(dst_width),  // %2
        "+r"(src_stride)  // %3
      :
      : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16",
        "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24");
}

void ScaleRowDown34_1_Box_NEON(const uint8_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint8_t* dst_ptr,
                               int dst_width) {
  asm volatile(
      "movi        v20.16b, #3                   \n"
      "add         %3, %3, %0                    \n"

      "1:          \n"
      "ld4         {v0.16b,v1.16b,v2.16b,v3.16b}, [%0], #64 \n"  // src line 0
      "ld4         {v4.16b,v5.16b,v6.16b,v7.16b}, [%3], #64 \n"  // src line 1
      "subs        %w2, %w2, #48                 \n"
      // average src line 0 with src line 1
      "urhadd      v0.16b, v0.16b, v4.16b        \n"
      "urhadd      v1.16b, v1.16b, v5.16b        \n"
      "urhadd      v2.16b, v2.16b, v6.16b        \n"
      "urhadd      v3.16b, v3.16b, v7.16b        \n"
      "prfm        pldl1keep, [%0, 448]          \n"  // prefetch 7 lines ahead

      // a0 = (src[0] * 3 + s[1] * 1 + 2) >> 2
      "ushll       v4.8h, v1.8b, #0              \n"
      "ushll2      v5.8h, v1.16b, #0             \n"
      "umlal       v4.8h, v0.8b, v20.8b          \n"
      "umlal2      v5.8h, v0.16b, v20.16b        \n"
      "uqrshrn     v0.8b, v4.8h, #2              \n"
      "uqrshrn2    v0.16b, v5.8h, #2             \n"
      "prfm        pldl1keep, [%3, 448]          \n"

      // a1 = (src[1] * 1 + s[2] * 1 + 1) >> 1
      "urhadd      v1.16b, v1.16b, v2.16b        \n"

      // a2 = (src[2] * 1 + s[3] * 3 + 2) >> 2
      "ushll       v4.8h, v2.8b, #0              \n"
      "ushll2      v5.8h, v2.16b, #0             \n"
      "umlal       v4.8h, v3.8b, v20.8b          \n"
      "umlal2      v5.8h, v3.16b, v20.16b        \n"
      "uqrshrn     v2.8b, v4.8h, #2              \n"
      "uqrshrn2    v2.16b, v5.8h, #2             \n"

      "st3         {v0.16b,v1.16b,v2.16b}, [%1], #48 \n"
      "b.gt        1b                            \n"
      : "+r"(src_ptr),    // %0
        "+r"(dst_ptr),    // %1
        "+r"(dst_width),  // %2
        "+r"(src_stride)  // %3
      :
      : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v20");
}

static const uvec8 kShuf38 = {0,  3,  6,  8,  11, 14, 16, 19,
                              22, 24, 27, 30, 0,  0,  0,  0};
static const vec16 kMult38_Div664 = {
    65536 / 12, 65536 / 12, 65536 / 8, 65536 / 12, 65536 / 12, 65536 / 8, 0, 0};
static const vec16 kMult38_Div996 = {65536 / 18, 65536 / 18, 65536 / 12,
                                     65536 / 18, 65536 / 18, 65536 / 12,
                                     0,          0};

// 32 -> 12
void ScaleRowDown38_NEON(const uint8_t* src_ptr,
                         ptrdiff_t src_stride,
                         uint8_t* dst_ptr,
                         int dst_width) {
  (void)src_stride;
  asm volatile(
      "ld1         {v3.16b}, [%[kShuf38]]        \n"
      "subs        %w[width], %w[width], #12     \n"
      "b.eq        2f                            \n"

      "1:          \n"
      "ldp         q0, q1, [%[src_ptr]], #32     \n"
      "subs        %w[width], %w[width], #12     \n"
      "tbl         v2.16b, {v0.16b, v1.16b}, v3.16b \n"
      "prfm        pldl1keep, [%[src_ptr], 448]  \n"  // prefetch 7 lines ahead
      "str         q2, [%[dst_ptr]]              \n"
      "add         %[dst_ptr], %[dst_ptr], #12   \n"
      "b.gt        1b                            \n"

      // Store exactly 12 bytes on the final iteration to avoid writing past
      // the end of the array.
      "2:          \n"
      "ldp         q0, q1, [%[src_ptr]]          \n"
      "tbl         v2.16b, {v0.16b, v1.16b}, v3.16b \n"
      "st1         {v2.8b}, [%[dst_ptr]], #8     \n"
      "st1         {v2.s}[2], [%[dst_ptr]]       \n"
      : [src_ptr] "+r"(src_ptr),  // %[src_ptr]
        [dst_ptr] "+r"(dst_ptr),  // %[dst_ptr]
        [width] "+r"(dst_width)   // %[width]
      : [kShuf38] "r"(&kShuf38)   // %[kShuf38]
      : "memory", "cc", "v0", "v1", "v2", "v3");
}

static const uvec8 kScaleRowDown38_3_BoxIndices1[] = {
    0, 1, 6, 7, 12, 13, 16, 17, 22, 23, 28, 29, 255, 255, 255, 255};
static const uvec8 kScaleRowDown38_3_BoxIndices2[] = {
    2, 3, 8, 9, 14, 15, 18, 19, 24, 25, 30, 31, 255, 255, 255, 255};
static const uvec8 kScaleRowDown38_3_BoxIndices3[] = {
    4, 5, 10, 11, 255, 255, 20, 21, 26, 27, 255, 255, 255, 255, 255, 255};
static const uvec8 kScaleRowDown38_NarrowIndices[] = {
    0, 2, 4, 6, 8, 10, 16, 18, 20, 22, 24, 26, 255, 255, 255, 255};

void ScaleRowDown38_3_Box_NEON(const uint8_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint8_t* dst_ptr,
                               int dst_width) {
  const uint8_t* src_ptr1 = src_ptr + src_stride;
  const uint8_t* src_ptr2 = src_ptr + src_stride * 2;
  asm volatile(
      "ld1         {v27.16b}, [%[tblArray1]]     \n"
      "ld1         {v28.16b}, [%[tblArray2]]     \n"
      "ld1         {v29.16b}, [%[tblArray3]]     \n"
      "ld1         {v31.16b}, [%[tblArray4]]     \n"
      "ld1         {v30.16b}, [%[div996]]        \n"

      "1:          \n"
      "ldp         q20, q0, [%[src_ptr]], #32    \n"
      "ldp         q21, q1, [%[src_ptr1]], #32   \n"
      "ldp         q22, q2, [%[src_ptr2]], #32   \n"

      "subs        %w[width], %w[width], #12     \n"

      // Add across strided rows first.
      "uaddl       v23.8h, v20.8b, v21.8b        \n"
      "uaddl       v3.8h, v0.8b, v1.8b           \n"
      "uaddl2      v24.8h, v20.16b, v21.16b      \n"
      "uaddl2      v4.8h, v0.16b, v1.16b         \n"

      "uaddw       v23.8h, v23.8h, v22.8b        \n"
      "uaddw       v3.8h, v3.8h, v2.8b           \n"
      "uaddw2      v24.8h, v24.8h, v22.16b       \n"  // abcdefgh ...
      "uaddw2      v4.8h, v4.8h, v2.16b          \n"

      // Permute groups of {three,three,two} into separate vectors to sum.
      "tbl         v20.16b, {v23.16b, v24.16b}, v27.16b \n"  // a d g ...
      "tbl         v0.16b, {v3.16b, v4.16b}, v27.16b \n"
      "tbl         v21.16b, {v23.16b, v24.16b}, v28.16b \n"  // b e h ...
      "tbl         v1.16b, {v3.16b, v4.16b}, v28.16b \n"
      "tbl         v22.16b, {v23.16b, v24.16b}, v29.16b \n"  // c f 0...
      "tbl         v2.16b, {v3.16b, v4.16b}, v29.16b \n"

      "add         v23.8h, v20.8h, v21.8h        \n"
      "add         v3.8h, v0.8h, v1.8h           \n"
      "add         v24.8h, v23.8h, v22.8h        \n"  // a+b+c d+e+f g+h
      "add         v4.8h, v3.8h, v2.8h           \n"

      "sqrdmulh    v24.8h, v24.8h, v30.8h        \n"  // v /= {9,9,6}
      "sqrdmulh    v25.8h, v4.8h, v30.8h         \n"
      "tbl         v21.16b, {v24.16b, v25.16b}, v31.16b \n"  // Narrow.
      "st1         {v21.d}[0], [%[dst_ptr]], #8  \n"
      "st1         {v21.s}[2], [%[dst_ptr]], #4  \n"
      "b.gt        1b                            \n"
      : [src_ptr] "+r"(src_ptr),                         // %[src_ptr]
        [dst_ptr] "+r"(dst_ptr),                         // %[dst_ptr]
        [src_ptr1] "+r"(src_ptr1),                       // %[src_ptr1]
        [src_ptr2] "+r"(src_ptr2),                       // %[src_ptr2]
        [width] "+r"(dst_width)                          // %[width]
      : [div996] "r"(&kMult38_Div996),                   // %[div996]
        [tblArray1] "r"(kScaleRowDown38_3_BoxIndices1),  // %[tblArray1]
        [tblArray2] "r"(kScaleRowDown38_3_BoxIndices2),  // %[tblArray2]
        [tblArray3] "r"(kScaleRowDown38_3_BoxIndices3),  // %[tblArray3]
        [tblArray4] "r"(kScaleRowDown38_NarrowIndices)   // %[tblArray4]
      : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v20", "v21", "22", "23",
        "24", "v27", "v28", "v29", "v30", "v31");
}

static const uvec8 kScaleRowDown38_2_BoxIndices1[] = {
    0, 1, 3, 4, 6, 7, 8, 9, 11, 12, 14, 15, 255, 255, 255, 255};
static const uvec8 kScaleRowDown38_2_BoxIndices2[] = {
    2, 18, 5, 21, 255, 255, 10, 26, 13, 29, 255, 255, 255, 255, 255, 255};

void ScaleRowDown38_2_Box_NEON(const uint8_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint8_t* dst_ptr,
                               int dst_width) {
  const uint8_t* src_ptr1 = src_ptr + src_stride;
  asm volatile(
      "ld1         {v28.16b}, [%[tblArray1]]     \n"
      "ld1         {v29.16b}, [%[tblArray2]]     \n"
      "ld1         {v31.16b}, [%[tblArray3]]     \n"
      "ld1         {v30.8h}, [%[div664]]         \n"

      "1:          \n"
      "ldp         q20, q0, [%[src_ptr]], #32    \n"  // abcdefgh ...
      "ldp         q21, q1, [%[src_ptr1]], #32   \n"  // ijklmnop ...
      "subs        %w[width], %w[width], #12     \n"

      // Permute into groups of six values (three pairs) to be summed.
      "tbl         v22.16b, {v20.16b}, v28.16b   \n"  // abdegh ...
      "tbl         v2.16b, {v0.16b}, v28.16b     \n"
      "tbl         v23.16b, {v21.16b}, v28.16b   \n"  // ijlmop ...
      "tbl         v3.16b, {v1.16b}, v28.16b     \n"
      "tbl         v24.16b, {v20.16b, v21.16b}, v29.16b \n"  // ckfn00 ...
      "tbl         v4.16b, {v0.16b, v1.16b}, v29.16b \n"

      "uaddlp      v22.8h, v22.16b               \n"  // a+b d+e g+h ...
      "uaddlp      v2.8h, v2.16b                 \n"
      "uaddlp      v23.8h, v23.16b               \n"  // i+j l+m o+p ...
      "uaddlp      v3.8h, v3.16b                 \n"
      "uaddlp      v24.8h, v24.16b               \n"  // c+k f+n   0 ...
      "uaddlp      v4.8h, v4.16b                 \n"
      "add         v20.8h, v22.8h, v23.8h        \n"
      "add         v0.8h, v2.8h, v3.8h           \n"
      "add         v21.8h, v20.8h, v24.8h        \n"  // a+b+i+j+c+k ...
      "add         v1.8h, v0.8h, v4.8h           \n"

      "sqrdmulh    v21.8h, v21.8h, v30.8h        \n"  // v /= {6,6,4}
      "sqrdmulh    v22.8h, v1.8h, v30.8h         \n"
      "tbl         v21.16b, {v21.16b, v22.16b}, v31.16b \n"  // Narrow.
      "st1         {v21.d}[0], [%[dst_ptr]], #8  \n"
      "st1         {v21.s}[2], [%[dst_ptr]], #4  \n"
      "b.gt        1b                            \n"
      : [src_ptr] "+r"(src_ptr),                         // %[src_ptr]
        [dst_ptr] "+r"(dst_ptr),                         // %[dst_ptr]
        [src_ptr1] "+r"(src_ptr1),                       // %[src_ptr1]
        [width] "+r"(dst_width)                          // %[width]
      : [div664] "r"(&kMult38_Div664),                   // %[div664]
        [tblArray1] "r"(kScaleRowDown38_2_BoxIndices1),  // %[tblArray1]
        [tblArray2] "r"(kScaleRowDown38_2_BoxIndices2),  // %[tblArray2]
        [tblArray3] "r"(kScaleRowDown38_NarrowIndices)   // %[tblArray3]
      : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v20", "v21", "v22",
        "v23", "v24", "v28", "v29", "v30", "v31");
}

void ScaleRowUp2_Linear_NEON(const uint8_t* src_ptr,
                             uint8_t* dst_ptr,
                             int dst_width) {
  const uint8_t* src_temp = src_ptr + 1;
  asm volatile(
      "movi        v31.16b, #3                   \n"

      "1:          \n"
      "ldr         q0, [%0], #16                 \n"  // 0123456789abcdef
      "ldr         q1, [%1], #16                 \n"  // 123456789abcdefg
      "prfm        pldl1keep, [%0, 448]          \n"  // prefetch 7 lines ahead

      "ushll       v2.8h, v0.8b, #0              \n"  // 01234567 (16b)
      "ushll       v3.8h, v1.8b, #0              \n"  // 12345678 (16b)
      "ushll2      v4.8h, v0.16b, #0             \n"  // 89abcdef (16b)
      "ushll2      v5.8h, v1.16b, #0             \n"  // 9abcdefg (16b)

      "umlal       v2.8h, v1.8b, v31.8b          \n"  // 3*near+far (odd)
      "umlal       v3.8h, v0.8b, v31.8b          \n"  // 3*near+far (even)
      "umlal2      v4.8h, v1.16b, v31.16b        \n"  // 3*near+far (odd)
      "umlal2      v5.8h, v0.16b, v31.16b        \n"  // 3*near+far (even)

      "rshrn       v2.8b, v2.8h, #2              \n"  // 3/4*near+1/4*far (odd)
      "rshrn       v1.8b, v3.8h, #2              \n"  // 3/4*near+1/4*far (even)
      "rshrn2      v2.16b, v4.8h, #2             \n"  // 3/4*near+1/4*far (odd)
      "rshrn2      v1.16b, v5.8h, #2             \n"  // 3/4*near+1/4*far (even)

      "st2         {v1.16b, v2.16b}, [%2], #32   \n"
      "subs        %w3, %w3, #32                 \n"
      "b.gt        1b                            \n"
      : "+r"(src_ptr),   // %0
        "+r"(src_temp),  // %1
        "+r"(dst_ptr),   // %2
        "+r"(dst_width)  // %3
      :
      : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5",
        "v31"  // Clobber List
  );
}

void ScaleRowUp2_Bilinear_NEON(const uint8_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint8_t* dst_ptr,
                               ptrdiff_t dst_stride,
                               int dst_width) {
  const uint8_t* src_ptr1 = src_ptr + src_stride;
  uint8_t* dst_ptr1 = dst_ptr + dst_stride;
  const uint8_t* src_temp = src_ptr + 1;
  const uint8_t* src_temp1 = src_ptr1 + 1;

  asm volatile(
      "movi        v31.8b, #3                    \n"
      "movi        v30.8h, #3                    \n"

      "1:          \n"
      "ldr         d0, [%0], #8                  \n"  // 01234567
      "ldr         d1, [%2], #8                  \n"  // 12345678
      "prfm        pldl1keep, [%0, 448]          \n"  // prefetch 7 lines ahead

      "ushll       v2.8h, v0.8b, #0              \n"  // 01234567 (16b)
      "ushll       v3.8h, v1.8b, #0              \n"  // 12345678 (16b)
      "umlal       v2.8h, v1.8b, v31.8b          \n"  // 3*near+far (1, odd)
      "umlal       v3.8h, v0.8b, v31.8b          \n"  // 3*near+far (1, even)

      "ldr         d0, [%1], #8                  \n"
      "ldr         d1, [%3], #8                  \n"
      "prfm        pldl1keep, [%1, 448]          \n"  // prefetch 7 lines ahead

      "ushll       v4.8h, v0.8b, #0              \n"  // 01234567 (16b)
      "ushll       v5.8h, v1.8b, #0              \n"  // 12345678 (16b)
      "umlal       v4.8h, v1.8b, v31.8b          \n"  // 3*near+far (2, odd)
      "umlal       v5.8h, v0.8b, v31.8b          \n"  // 3*near+far (2, even)

      "mov         v0.16b, v4.16b                \n"
      "mov         v1.16b, v5.16b                \n"
      "mla         v4.8h, v2.8h, v30.8h          \n"  // 9 3 3 1 (1, odd)
      "mla         v5.8h, v3.8h, v30.8h          \n"  // 9 3 3 1 (1, even)
      "mla         v2.8h, v0.8h, v30.8h          \n"  // 9 3 3 1 (2, odd)
      "mla         v3.8h, v1.8h, v30.8h          \n"  // 9 3 3 1 (2, even)

      "rshrn       v2.8b, v2.8h, #4              \n"  // 2, odd
      "rshrn       v1.8b, v3.8h, #4              \n"  // 2, even
      "rshrn       v4.8b, v4.8h, #4              \n"  // 1, odd
      "rshrn       v3.8b, v5.8h, #4              \n"  // 1, even

      "st2         {v1.8b, v2.8b}, [%5], #16     \n"  // store 1
      "st2         {v3.8b, v4.8b}, [%4], #16     \n"  // store 2
      "subs        %w6, %w6, #16                 \n"  // 8 sample -> 16 sample
      "b.gt        1b                            \n"
      : "+r"(src_ptr),    // %0
        "+r"(src_ptr1),   // %1
        "+r"(src_temp),   // %2
        "+r"(src_temp1),  // %3
        "+r"(dst_ptr),    // %4
        "+r"(dst_ptr1),   // %5
        "+r"(dst_width)   // %6
      :
      : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v30",
        "v31"  // Clobber List
  );
}

void ScaleRowUp2_Linear_12_NEON(const uint16_t* src_ptr,
                                uint16_t* dst_ptr,
                                int dst_width) {
  const uint16_t* src_temp = src_ptr + 1;
  asm volatile(
      "movi        v31.8h, #3                    \n"

      "1:          \n"
      "ld1         {v0.8h}, [%0], #16            \n"  // 01234567 (16b)
      "ld1         {v1.8h}, [%1], #16            \n"  // 12345678 (16b)
      "prfm        pldl1keep, [%0, 448]          \n"  // prefetch 7 lines ahead

      "mov         v2.16b, v0.16b                \n"
      "mla         v0.8h, v1.8h, v31.8h          \n"  // 3*near+far (odd)
      "mla         v1.8h, v2.8h, v31.8h          \n"  // 3*near+far (even)

      "urshr       v2.8h, v0.8h, #2              \n"  // 3/4*near+1/4*far (odd)
      "urshr       v1.8h, v1.8h, #2              \n"  // 3/4*near+1/4*far (even)

      "st2         {v1.8h, v2.8h}, [%2], #32     \n"  // store
      "subs        %w3, %w3, #16                 \n"  // 8 sample -> 16 sample
      "b.gt        1b                            \n"
      : "+r"(src_ptr),   // %0
        "+r"(src_temp),  // %1
        "+r"(dst_ptr),   // %2
        "+r"(dst_width)  // %3
      :
      : "memory", "cc", "v0", "v1", "v2", "v31"  // Clobber List
  );
}

void ScaleRowUp2_Bilinear_12_NEON(const uint16_t* src_ptr,
                                  ptrdiff_t src_stride,
                                  uint16_t* dst_ptr,
                                  ptrdiff_t dst_stride,
                                  int dst_width) {
  const uint16_t* src_ptr1 = src_ptr + src_stride;
  uint16_t* dst_ptr1 = dst_ptr + dst_stride;
  const uint16_t* src_temp = src_ptr + 1;
  const uint16_t* src_temp1 = src_ptr1 + 1;

  asm volatile(
      "movi        v31.8h, #3                    \n"

      "1:          \n"
      "ld1         {v2.8h}, [%0], #16            \n"  // 01234567 (16b)
      "ld1         {v3.8h}, [%2], #16            \n"  // 12345678 (16b)
      "prfm        pldl1keep, [%0, 448]          \n"  // prefetch 7 lines ahead

      "mov         v0.16b, v2.16b                \n"
      "mla         v2.8h, v3.8h, v31.8h          \n"  // 3*near+far (odd)
      "mla         v3.8h, v0.8h, v31.8h          \n"  // 3*near+far (even)

      "ld1         {v4.8h}, [%1], #16            \n"  // 01234567 (16b)
      "ld1         {v5.8h}, [%3], #16            \n"  // 12345678 (16b)
      "prfm        pldl1keep, [%1, 448]          \n"  // prefetch 7 lines ahead

      "mov         v0.16b, v4.16b                \n"
      "mla         v4.8h, v5.8h, v31.8h          \n"  // 3*near+far (odd)
      "mla         v5.8h, v0.8h, v31.8h          \n"  // 3*near+far (even)

      "mov         v0.16b, v4.16b                \n"
      "mov         v1.16b, v5.16b                \n"
      "mla         v4.8h, v2.8h, v31.8h          \n"  // 9 3 3 1 (1, odd)
      "mla         v5.8h, v3.8h, v31.8h          \n"  // 9 3 3 1 (1, even)
      "mla         v2.8h, v0.8h, v31.8h          \n"  // 9 3 3 1 (2, odd)
      "mla         v3.8h, v1.8h, v31.8h          \n"  // 9 3 3 1 (2, even)

      "urshr       v2.8h, v2.8h, #4              \n"  // 2, odd
      "urshr       v1.8h, v3.8h, #4              \n"  // 2, even
      "urshr       v4.8h, v4.8h, #4              \n"  // 1, odd
      "urshr       v3.8h, v5.8h, #4              \n"  // 1, even

      "st2         {v3.8h, v4.8h}, [%4], #32     \n"  // store 1
      "st2         {v1.8h, v2.8h}, [%5], #32     \n"  // store 2

      "subs        %w6, %w6, #16                 \n"  // 8 sample -> 16 sample
      "b.gt        1b                            \n"
      : "+r"(src_ptr),    // %0
        "+r"(src_ptr1),   // %1
        "+r"(src_temp),   // %2
        "+r"(src_temp1),  // %3
        "+r"(dst_ptr),    // %4
        "+r"(dst_ptr1),   // %5
        "+r"(dst_width)   // %6
      :
      : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5",
        "v31"  // Clobber List
  );
}

void ScaleRowUp2_Linear_16_NEON(const uint16_t* src_ptr,
                                uint16_t* dst_ptr,
                                int dst_width) {
  const uint16_t* src_temp = src_ptr + 1;
  asm volatile(
      "movi        v31.8h, #3                    \n"

      "1:          \n"
      "ld1         {v0.8h}, [%0], #16            \n"  // 01234567 (16b)
      "ld1         {v1.8h}, [%1], #16            \n"  // 12345678 (16b)
      "prfm        pldl1keep, [%0, 448]          \n"  // prefetch 7 lines ahead

      "ushll       v2.4s, v0.4h, #0              \n"  // 0123 (32b)
      "ushll2      v3.4s, v0.8h, #0              \n"  // 4567 (32b)
      "ushll       v4.4s, v1.4h, #0              \n"  // 1234 (32b)
      "ushll2      v5.4s, v1.8h, #0              \n"  // 5678 (32b)

      "umlal       v2.4s, v1.4h, v31.4h          \n"  // 3*near+far (1, odd)
      "umlal2      v3.4s, v1.8h, v31.8h          \n"  // 3*near+far (2, odd)
      "umlal       v4.4s, v0.4h, v31.4h          \n"  // 3*near+far (1, even)
      "umlal2      v5.4s, v0.8h, v31.8h          \n"  // 3*near+far (2, even)

      "rshrn       v0.4h, v4.4s, #2              \n"  // 3/4*near+1/4*far
      "rshrn2      v0.8h, v5.4s, #2              \n"  // 3/4*near+1/4*far (even)
      "rshrn       v1.4h, v2.4s, #2              \n"  // 3/4*near+1/4*far
      "rshrn2      v1.8h, v3.4s, #2              \n"  // 3/4*near+1/4*far (odd)

      "st2         {v0.8h, v1.8h}, [%2], #32     \n"  // store
      "subs        %w3, %w3, #16                 \n"  // 8 sample -> 16 sample
      "b.gt        1b                            \n"
      : "+r"(src_ptr),   // %0
        "+r"(src_temp),  // %1
        "+r"(dst_ptr),   // %2
        "+r"(dst_width)  // %3
      :
      : "memory", "cc", "v0", "v1", "v2", "v31"  // Clobber List
  );
}

void ScaleRowUp2_Bilinear_16_NEON(const uint16_t* src_ptr,
                                  ptrdiff_t src_stride,
                                  uint16_t* dst_ptr,
                                  ptrdiff_t dst_stride,
                                  int dst_width) {
  const uint16_t* src_ptr1 = src_ptr + src_stride;
  uint16_t* dst_ptr1 = dst_ptr + dst_stride;
  const uint16_t* src_temp = src_ptr + 1;
  const uint16_t* src_temp1 = src_ptr1 + 1;

  asm volatile(
      "movi        v31.4h, #3                    \n"
      "movi        v30.4s, #3                    \n"

      "1:          \n"
      "ldr         d0, [%0], #8                  \n"  // 0123 (16b)
      "ldr         d1, [%2], #8                  \n"  // 1234 (16b)
      "prfm        pldl1keep, [%0, 448]          \n"  // prefetch 7 lines ahead
      "ushll       v2.4s, v0.4h, #0              \n"  // 0123 (32b)
      "ushll       v3.4s, v1.4h, #0              \n"  // 1234 (32b)
      "umlal       v2.4s, v1.4h, v31.4h          \n"  // 3*near+far (1, odd)
      "umlal       v3.4s, v0.4h, v31.4h          \n"  // 3*near+far (1, even)

      "ldr         d0, [%1], #8                  \n"  // 0123 (16b)
      "ldr         d1, [%3], #8                  \n"  // 1234 (16b)
      "prfm        pldl1keep, [%1, 448]          \n"  // prefetch 7 lines ahead
      "ushll       v4.4s, v0.4h, #0              \n"  // 0123 (32b)
      "ushll       v5.4s, v1.4h, #0              \n"  // 1234 (32b)
      "umlal       v4.4s, v1.4h, v31.4h          \n"  // 3*near+far (2, odd)
      "umlal       v5.4s, v0.4h, v31.4h          \n"  // 3*near+far (2, even)

      "mov         v0.16b, v4.16b                \n"
      "mov         v1.16b, v5.16b                \n"
      "mla         v4.4s, v2.4s, v30.4s          \n"  // 9 3 3 1 (1, odd)
      "mla         v5.4s, v3.4s, v30.4s          \n"  // 9 3 3 1 (1, even)
      "mla         v2.4s, v0.4s, v30.4s          \n"  // 9 3 3 1 (2, odd)
      "mla         v3.4s, v1.4s, v30.4s          \n"  // 9 3 3 1 (2, even)

      "rshrn       v1.4h, v4.4s, #4              \n"  // 3/4*near+1/4*far
      "rshrn       v0.4h, v5.4s, #4              \n"  // 3/4*near+1/4*far
      "rshrn       v5.4h, v2.4s, #4              \n"  // 3/4*near+1/4*far
      "rshrn       v4.4h, v3.4s, #4              \n"  // 3/4*near+1/4*far

      "st2         {v0.4h, v1.4h}, [%4], #16     \n"  // store 1
      "st2         {v4.4h, v5.4h}, [%5], #16     \n"  // store 2

      "subs        %w6, %w6, #8                  \n"  // 4 sample -> 8 sample
      "b.gt        1b                            \n"
      : "+r"(src_ptr),    // %0
        "+r"(src_ptr1),   // %1
        "+r"(src_temp),   // %2
        "+r"(src_temp1),  // %3
        "+r"(dst_ptr),    // %4
        "+r"(dst_ptr1),   // %5
        "+r"(dst_width)   // %6
      :
      : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v30",
        "v31"  // Clobber List
  );
}

void ScaleUVRowUp2_Linear_NEON(const uint8_t* src_ptr,
                               uint8_t* dst_ptr,
                               int dst_width) {
  const uint8_t* src_temp = src_ptr + 2;
  asm volatile(
      "movi        v31.8b, #3                    \n"

      "1:          \n"
      "ldr         d0, [%0], #8                  \n"  // 00112233 (1u1v)
      "ldr         d1, [%1], #8                  \n"  // 11223344 (1u1v)
      "prfm        pldl1keep, [%0, 448]          \n"  // prefetch 7 lines ahead

      "ushll       v2.8h, v0.8b, #0              \n"  // 00112233 (1u1v, 16b)
      "ushll       v3.8h, v1.8b, #0              \n"  // 11223344 (1u1v, 16b)

      "umlal       v2.8h, v1.8b, v31.8b          \n"  // 3*near+far (odd)
      "umlal       v3.8h, v0.8b, v31.8b          \n"  // 3*near+far (even)

      "rshrn       v2.8b, v2.8h, #2              \n"  // 3/4*near+1/4*far (odd)
      "rshrn       v1.8b, v3.8h, #2              \n"  // 3/4*near+1/4*far (even)

      "st2         {v1.4h, v2.4h}, [%2], #16     \n"  // store
      "subs        %w3, %w3, #8                  \n"  // 4 uv -> 8 uv
      "b.gt        1b                            \n"
      : "+r"(src_ptr),   // %0
        "+r"(src_temp),  // %1
        "+r"(dst_ptr),   // %2
        "+r"(dst_width)  // %3
      :
      : "memory", "cc", "v0", "v1", "v2", "v3", "v31"  // Clobber List
  );
}

void ScaleUVRowUp2_Bilinear_NEON(const uint8_t* src_ptr,
                                 ptrdiff_t src_stride,
                                 uint8_t* dst_ptr,
                                 ptrdiff_t dst_stride,
                                 int dst_width) {
  const uint8_t* src_ptr1 = src_ptr + src_stride;
  uint8_t* dst_ptr1 = dst_ptr + dst_stride;
  const uint8_t* src_temp = src_ptr + 2;
  const uint8_t* src_temp1 = src_ptr1 + 2;

  asm volatile(
      "movi        v31.8b, #3                    \n"
      "movi        v30.8h, #3                    \n"

      "1:          \n"
      "ldr         d0, [%0], #8                  \n"
      "ldr         d1, [%2], #8                  \n"
      "prfm        pldl1keep, [%0, 448]          \n"  // prefetch 7 lines ahead

      "ushll       v2.8h, v0.8b, #0              \n"
      "ushll       v3.8h, v1.8b, #0              \n"
      "umlal       v2.8h, v1.8b, v31.8b          \n"  // 3*near+far (1, odd)
      "umlal       v3.8h, v0.8b, v31.8b          \n"  // 3*near+far (1, even)

      "ldr         d0, [%1], #8                  \n"
      "ldr         d1, [%3], #8                  \n"
      "prfm        pldl1keep, [%1, 448]          \n"  // prefetch 7 lines ahead

      "ushll       v4.8h, v0.8b, #0              \n"
      "ushll       v5.8h, v1.8b, #0              \n"
      "umlal       v4.8h, v1.8b, v31.8b          \n"  // 3*near+far (2, odd)
      "umlal       v5.8h, v0.8b, v31.8b          \n"  // 3*near+far (2, even)

      "mov         v0.16b, v4.16b                \n"
      "mov         v1.16b, v5.16b                \n"
      "mla         v4.8h, v2.8h, v30.8h          \n"  // 9 3 3 1 (1, odd)
      "mla         v5.8h, v3.8h, v30.8h          \n"  // 9 3 3 1 (1, even)
      "mla         v2.8h, v0.8h, v30.8h          \n"  // 9 3 3 1 (2, odd)
      "mla         v3.8h, v1.8h, v30.8h          \n"  // 9 3 3 1 (2, even)

      "rshrn       v2.8b, v2.8h, #4              \n"  // 2, odd
      "rshrn       v1.8b, v3.8h, #4              \n"  // 2, even
      "rshrn       v4.8b, v4.8h, #4              \n"  // 1, odd
      "rshrn       v3.8b, v5.8h, #4              \n"  // 1, even

      "st2         {v1.4h, v2.4h}, [%5], #16     \n"  // store 2
      "st2         {v3.4h, v4.4h}, [%4], #16     \n"  // store 1
      "subs        %w6, %w6, #8                  \n"  // 4 uv -> 8 uv
      "b.gt        1b                            \n"
      : "+r"(src_ptr),    // %0
        "+r"(src_ptr1),   // %1
        "+r"(src_temp),   // %2
        "+r"(src_temp1),  // %3
        "+r"(dst_ptr),    // %4
        "+r"(dst_ptr1),   // %5
        "+r"(dst_width)   // %6
      :
      : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v30",
        "v31"  // Clobber List
  );
}

void ScaleUVRowUp2_Linear_16_NEON(const uint16_t* src_ptr,
                                  uint16_t* dst_ptr,
                                  int dst_width) {
  const uint16_t* src_temp = src_ptr + 2;
  asm volatile(
      "movi        v31.8h, #3                    \n"

      "1:          \n"
      "ld1         {v0.8h}, [%0], #16            \n"  // 01234567 (16b)
      "ld1         {v1.8h}, [%1], #16            \n"  // 12345678 (16b)
      "prfm        pldl1keep, [%0, 448]          \n"  // prefetch 7 lines ahead

      "ushll       v2.4s, v0.4h, #0              \n"  // 0011 (1u1v, 32b)
      "ushll       v3.4s, v1.4h, #0              \n"  // 1122 (1u1v, 32b)
      "ushll2      v4.4s, v0.8h, #0              \n"  // 2233 (1u1v, 32b)
      "ushll2      v5.4s, v1.8h, #0              \n"  // 3344 (1u1v, 32b)

      "umlal       v2.4s, v1.4h, v31.4h          \n"  // 3*near+far (odd)
      "umlal       v3.4s, v0.4h, v31.4h          \n"  // 3*near+far (even)
      "umlal2      v4.4s, v1.8h, v31.8h          \n"  // 3*near+far (odd)
      "umlal2      v5.4s, v0.8h, v31.8h          \n"  // 3*near+far (even)

      "rshrn       v2.4h, v2.4s, #2              \n"  // 3/4*near+1/4*far (odd)
      "rshrn       v1.4h, v3.4s, #2              \n"  // 3/4*near+1/4*far (even)
      "rshrn       v4.4h, v4.4s, #2              \n"  // 3/4*near+1/4*far (odd)
      "rshrn       v3.4h, v5.4s, #2              \n"  // 3/4*near+1/4*far (even)

      "st2         {v1.2s, v2.2s}, [%2], #16     \n"  // store
      "st2         {v3.2s, v4.2s}, [%2], #16     \n"  // store
      "subs        %w3, %w3, #8                  \n"  // 4 uv -> 8 uv
      "b.gt        1b                            \n"
      : "+r"(src_ptr),   // %0
        "+r"(src_temp),  // %1
        "+r"(dst_ptr),   // %2
        "+r"(dst_width)  // %3
      :
      : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5",
        "v31"  // Clobber List
  );
}

void ScaleUVRowUp2_Bilinear_16_NEON(const uint16_t* src_ptr,
                                    ptrdiff_t src_stride,
                                    uint16_t* dst_ptr,
                                    ptrdiff_t dst_stride,
                                    int dst_width) {
  const uint16_t* src_ptr1 = src_ptr + src_stride;
  uint16_t* dst_ptr1 = dst_ptr + dst_stride;
  const uint16_t* src_temp = src_ptr + 2;
  const uint16_t* src_temp1 = src_ptr1 + 2;

  asm volatile(
      "movi        v31.4h, #3                    \n"
      "movi        v30.4s, #3                    \n"

      "1:          \n"
      "ldr         d0, [%0], #8                  \n"
      "ldr         d1, [%2], #8                  \n"
      "prfm        pldl1keep, [%0, 448]          \n"  // prefetch 7 lines ahead
      "ushll       v2.4s, v0.4h, #0              \n"  // 0011 (1u1v, 32b)
      "ushll       v3.4s, v1.4h, #0              \n"  // 1122 (1u1v, 32b)
      "umlal       v2.4s, v1.4h, v31.4h          \n"  // 3*near+far (1, odd)
      "umlal       v3.4s, v0.4h, v31.4h          \n"  // 3*near+far (1, even)

      "ldr         d0, [%1], #8                  \n"
      "ldr         d1, [%3], #8                  \n"
      "prfm        pldl1keep, [%1, 448]          \n"  // prefetch 7 lines ahead
      "ushll       v4.4s, v0.4h, #0              \n"  // 0011 (1u1v, 32b)
      "ushll       v5.4s, v1.4h, #0              \n"  // 1122 (1u1v, 32b)
      "umlal       v4.4s, v1.4h, v31.4h          \n"  // 3*near+far (2, odd)
      "umlal       v5.4s, v0.4h, v31.4h          \n"  // 3*near+far (2, even)

      "mov         v0.16b, v4.16b                \n"
      "mov         v1.16b, v5.16b                \n"
      "mla         v4.4s, v2.4s, v30.4s          \n"  // 9 3 3 1 (1, odd)
      "mla         v5.4s, v3.4s, v30.4s          \n"  // 9 3 3 1 (1, even)
      "mla         v2.4s, v0.4s, v30.4s          \n"  // 9 3 3 1 (2, odd)
      "mla         v3.4s, v1.4s, v30.4s          \n"  // 9 3 3 1 (2, even)

      "rshrn       v1.4h, v2.4s, #4              \n"  // 2, odd
      "rshrn       v0.4h, v3.4s, #4              \n"  // 2, even
      "rshrn       v3.4h, v4.4s, #4              \n"  // 1, odd
      "rshrn       v2.4h, v5.4s, #4              \n"  // 1, even

      "st2         {v0.2s, v1.2s}, [%5], #16     \n"  // store 2
      "st2         {v2.2s, v3.2s}, [%4], #16     \n"  // store 1
      "subs        %w6, %w6, #4                  \n"  // 2 uv -> 4 uv
      "b.gt        1b                            \n"
      : "+r"(src_ptr),    // %0
        "+r"(src_ptr1),   // %1
        "+r"(src_temp),   // %2
        "+r"(src_temp1),  // %3
        "+r"(dst_ptr),    // %4
        "+r"(dst_ptr1),   // %5
        "+r"(dst_width)   // %6
      :
      : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v30",
        "v31"  // Clobber List
  );
}

// Add a row of bytes to a row of shorts.  Used for box filter.
// Reads 16 bytes and accumulates to 16 shorts at a time.
void ScaleAddRow_NEON(const uint8_t* src_ptr,
                      uint16_t* dst_ptr,
                      int src_width) {
  asm volatile(
      "1:          \n"
      "ld1         {v1.8h, v2.8h}, [%1]          \n"  // load accumulator
      "ld1         {v0.16b}, [%0], #16           \n"  // load 16 bytes
      "uaddw2      v2.8h, v2.8h, v0.16b          \n"  // add
      "prfm        pldl1keep, [%0, 448]          \n"  // prefetch 7 lines ahead
      "uaddw       v1.8h, v1.8h, v0.8b           \n"
      "st1         {v1.8h, v2.8h}, [%1], #32     \n"  // store accumulator
      "subs        %w2, %w2, #16                 \n"  // 16 processed per loop
      "b.gt        1b                            \n"
      : "+r"(src_ptr),   // %0
        "+r"(dst_ptr),   // %1
        "+r"(src_width)  // %2
      :
      : "memory", "cc", "v0", "v1", "v2"  // Clobber List
  );
}

#define SCALE_FILTER_COLS_STEP_ADDR                         \
  "lsr        %[tmp_offset], %x[x], #16                 \n" \
  "add        %[tmp_ptr], %[src_ptr], %[tmp_offset]     \n" \
  "add        %x[x], %x[x], %x[dx]                      \n"

// The Neon version mimics this formula (from scale_common.cc):
// #define BLENDER(a, b, f) (uint8_t)((int)(a) +
//    ((((int)((f)) * ((int)(b) - (int)(a))) + 0x8000) >> 16))

void ScaleFilterCols_NEON(uint8_t* dst_ptr,
                          const uint8_t* src_ptr,
                          int dst_width,
                          int x,
                          int dx) {
  int dx_offset[4] = {0, 1, 2, 3};
  int64_t tmp_offset;
  uint8_t* tmp_ptr;
  asm volatile(
      "dup         v0.4s, %w[x]                  \n"
      "dup         v1.4s, %w[dx]                 \n"
      "ld1         {v2.4s}, [%[dx_offset]]       \n"  // 0 1 2 3
      "shl         v3.4s, v1.4s, #2              \n"  // 4 * dx
      "shl         v22.4s, v1.4s, #3             \n"  // 8 * dx

      "mul         v1.4s, v1.4s, v2.4s           \n"
      // x         , x + 1 * dx, x + 2 * dx, x + 3 * dx
      "add         v1.4s, v1.4s, v0.4s           \n"
      // x + 4 * dx, x + 5 * dx, x + 6 * dx, x + 7 * dx
      "add         v2.4s, v1.4s, v3.4s           \n"

      "movi        v0.8h, #0                     \n"

      // truncate to uint16_t
      "trn1        v22.8h, v22.8h, v0.8h         \n"
      "trn1        v20.8h, v1.8h, v0.8h          \n"
      "trn1        v21.8h, v2.8h, v0.8h          \n"

      "1:          \n" SCALE_FILTER_COLS_STEP_ADDR
      "ldr         h6, [%[tmp_ptr]]              \n" SCALE_FILTER_COLS_STEP_ADDR
      "ld1         {v6.h}[1], [%[tmp_ptr]]       \n" SCALE_FILTER_COLS_STEP_ADDR
      "ld1         {v6.h}[2], [%[tmp_ptr]]       \n" SCALE_FILTER_COLS_STEP_ADDR
      "ld1         {v6.h}[3], [%[tmp_ptr]]       \n" SCALE_FILTER_COLS_STEP_ADDR
      "ld1         {v6.h}[4], [%[tmp_ptr]]       \n" SCALE_FILTER_COLS_STEP_ADDR
      "ld1         {v6.h}[5], [%[tmp_ptr]]       \n" SCALE_FILTER_COLS_STEP_ADDR
      "ld1         {v6.h}[6], [%[tmp_ptr]]       \n" SCALE_FILTER_COLS_STEP_ADDR
      "ld1         {v6.h}[7], [%[tmp_ptr]]       \n"

      "subs        %w[width], %w[width], #8      \n"  // 8 processed per loop
      "trn1        v4.16b, v6.16b, v0.16b        \n"
      "trn2        v5.16b, v6.16b, v0.16b        \n"

      "ssubl       v16.4s, v5.4h, v4.4h          \n"
      "ssubl2      v17.4s, v5.8h, v4.8h          \n"
      "mul         v16.4s, v16.4s, v20.4s        \n"
      "mul         v17.4s, v17.4s, v21.4s        \n"
      "rshrn       v6.4h, v16.4s, #16            \n"
      "rshrn2      v6.8h, v17.4s, #16            \n"
      "add         v4.8h, v4.8h, v6.8h           \n"
      "xtn         v4.8b, v4.8h                  \n"

      "add         v20.8h, v20.8h, v22.8h        \n"
      "add         v21.8h, v21.8h, v22.8h        \n"

      "st1         {v4.8b}, [%[dst_ptr]], #8     \n"  // store pixels
      "b.gt        1b                            \n"
      : [src_ptr] "+r"(src_ptr),         // %[src_ptr]
        [dst_ptr] "+r"(dst_ptr),         // %[dst_ptr]
        [width] "+r"(dst_width),         // %[width]
        [x] "+r"(x),                     // %[x]
        [dx] "+r"(dx),                   // %[dx]
        [tmp_offset] "=&r"(tmp_offset),  // %[tmp_offset]
        [tmp_ptr] "=&r"(tmp_ptr)         // %[tmp_ptr]
      : [dx_offset] "r"(dx_offset)       // %[dx_offset]
      : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v16", "v17",
        "v20", "v21", "v22");
}

#undef SCALE_FILTER_COLS_STEP_ADDR

void ScaleARGBRowDown2_NEON(const uint8_t* src_ptr,
                            ptrdiff_t src_stride,
                            uint8_t* dst,
                            int dst_width) {
  (void)src_stride;
  asm volatile(
      "1:          \n"
      "ld1         {v0.4s, v1.4s, v2.4s, v3.4s}, [%[src]], #64 \n"
      "subs        %w[width], %w[width], #8      \n"
      "prfm        pldl1keep, [%[src], 448]      \n"
      "uzp2        v0.4s, v0.4s, v1.4s           \n"
      "uzp2        v1.4s, v2.4s, v3.4s           \n"
      "st1         {v0.4s, v1.4s}, [%[dst]], #32 \n"
      "b.gt        1b                            \n"
      : [src] "+r"(src_ptr),     // %[src]
        [dst] "+r"(dst),         // %[dst]
        [width] "+r"(dst_width)  // %[width]
      :
      : "memory", "cc", "v0", "v1", "v2", "v3");
}

void ScaleARGBRowDown2Linear_NEON(const uint8_t* src_argb,
                                  ptrdiff_t src_stride,
                                  uint8_t* dst_argb,
                                  int dst_width) {
  (void)src_stride;
  const uint8_t* src_argb1 = src_argb + 32;
  asm volatile(
      "1:          \n"
      "ld2         {v0.4s, v1.4s}, [%[src]]      \n"
      "add         %[src], %[src], #64           \n"
      "ld2         {v2.4s, v3.4s}, [%[src1]]     \n"
      "add         %[src1], %[src1], #64         \n"
      "urhadd      v0.16b, v0.16b, v1.16b        \n"
      "urhadd      v1.16b, v2.16b, v3.16b        \n"
      "subs        %w[width], %w[width], #8      \n"
      "st1         {v0.16b, v1.16b}, [%[dst]], #32 \n"
      "b.gt        1b                            \n"
      : [src] "+r"(src_argb),    // %[src]
        [src1] "+r"(src_argb1),  // %[src1]
        [dst] "+r"(dst_argb),    // %[dst]
        [width] "+r"(dst_width)  // %[width]
      :
      : "memory", "cc", "v0", "v1", "v2", "v3");
}

void ScaleARGBRowDown2Box_NEON(const uint8_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint8_t* dst,
                               int dst_width) {
  const uint8_t* src_ptr1 = src_ptr + src_stride;
  asm volatile(
      "1:          \n"
      "ld2         {v0.4s, v1.4s}, [%[src]], #32 \n"
      "ld2         {v20.4s, v21.4s}, [%[src1]], #32 \n"
      "uaddl       v2.8h, v0.8b, v1.8b           \n"
      "uaddl2      v3.8h, v0.16b, v1.16b         \n"
      "uaddl       v22.8h, v20.8b, v21.8b        \n"
      "uaddl2      v23.8h, v20.16b, v21.16b      \n"
      "add         v0.8h, v2.8h, v22.8h          \n"
      "add         v1.8h, v3.8h, v23.8h          \n"
      "rshrn       v0.8b, v0.8h, #2              \n"
      "rshrn       v1.8b, v1.8h, #2              \n"
      "subs        %w[width], %w[width], #4      \n"
      "stp         d0, d1, [%[dst]], #16         \n"
      "b.gt        1b                            \n"
      : [src] "+r"(src_ptr), [src1] "+r"(src_ptr1), [dst] "+r"(dst),
        [width] "+r"(dst_width)
      :
      : "memory", "cc", "v0", "v1", "v2", "v3", "v20", "v21", "v22", "v23");
}

void ScaleARGBRowDownEven_NEON(const uint8_t* src_argb,
                               ptrdiff_t src_stride,
                               int src_stepx,
                               uint8_t* dst_argb,
                               int dst_width) {
  const uint8_t* src_argb1 = src_argb + src_stepx * 4;
  const uint8_t* src_argb2 = src_argb + src_stepx * 8;
  const uint8_t* src_argb3 = src_argb + src_stepx * 12;
  int64_t i = 0;
  (void)src_stride;
  asm volatile(
      "1:          \n"
      "ldr         w10, [%[src], %[i]]           \n"
      "ldr         w11, [%[src1], %[i]]          \n"
      "ldr         w12, [%[src2], %[i]]          \n"
      "ldr         w13, [%[src3], %[i]]          \n"
      "add         %[i], %[i], %[step]           \n"
      "subs        %w[width], %w[width], #4      \n"
      "prfm        pldl1keep, [%[src], 448]      \n"
      "stp         w10, w11, [%[dst]], #8        \n"
      "stp         w12, w13, [%[dst]], #8        \n"
      "b.gt        1b                            \n"
      : [src] "+r"(src_argb), [src1] "+r"(src_argb1), [src2] "+r"(src_argb2),
        [src3] "+r"(src_argb3), [dst] "+r"(dst_argb), [width] "+r"(dst_width),
        [i] "+r"(i)
      : [step] "r"((int64_t)(src_stepx * 16))
      : "memory", "cc", "w10", "w11", "w12", "w13");
}

// Reads 4 pixels at a time.
// Alignment requirement: src_argb 4 byte aligned.
// TODO(Yang Zhang): Might be worth another optimization pass in future.
// It could be upgraded to 8 pixels at a time to start with.
void ScaleARGBRowDownEvenBox_NEON(const uint8_t* src_argb,
                                  ptrdiff_t src_stride,
                                  int src_stepx,
                                  uint8_t* dst_argb,
                                  int dst_width) {
  asm volatile(
      "add         %1, %1, %0                    \n"
      "1:          \n"
      "ld1         {v0.8b}, [%0], %4             \n"  // Read 4 2x2 -> 2x1
      "ld1         {v1.8b}, [%1], %4             \n"
      "ld1         {v2.8b}, [%0], %4             \n"
      "ld1         {v3.8b}, [%1], %4             \n"
      "ld1         {v4.8b}, [%0], %4             \n"
      "ld1         {v5.8b}, [%1], %4             \n"
      "ld1         {v6.8b}, [%0], %4             \n"
      "ld1         {v7.8b}, [%1], %4             \n"
      "uaddl       v0.8h, v0.8b, v1.8b           \n"
      "uaddl       v2.8h, v2.8b, v3.8b           \n"
      "uaddl       v4.8h, v4.8b, v5.8b           \n"
      "uaddl       v6.8h, v6.8b, v7.8b           \n"
      "prfm        pldl1keep, [%0, 448]          \n"  // prefetch 7 lines ahead
      "zip1        v1.2d, v0.2d, v2.2d           \n"
      "zip2        v2.2d, v0.2d, v2.2d           \n"
      "zip1        v5.2d, v4.2d, v6.2d           \n"
      "zip2        v6.2d, v4.2d, v6.2d           \n"
      "prfm        pldl1keep, [%1, 448]          \n"
      "add         v0.8h, v1.8h, v2.8h           \n"  // (a+b)_(c+d)
      "add         v4.8h, v5.8h, v6.8h           \n"  // (e+f)_(g+h)
      "rshrn       v0.8b, v0.8h, #2              \n"  // first 2 pixels.
      "rshrn       v1.8b, v4.8h, #2              \n"  // next 2 pixels.
      "subs        %w3, %w3, #4                  \n"  // 4 pixels per loop.
      "stp         d0, d1, [%2], #16             \n"
      "b.gt        1b                            \n"
      : "+r"(src_argb),                // %0
        "+r"(src_stride),              // %1
        "+r"(dst_argb),                // %2
        "+r"(dst_width)                // %3
      : "r"((int64_t)(src_stepx * 4))  // %4
      : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
}

// TODO(Yang Zhang): Investigate less load instructions for
// the x/dx stepping
#define LOAD1_DATA32_LANE(vn, n)                 \
  "lsr        %5, %3, #16                    \n" \
  "add        %6, %1, %5, lsl #2             \n" \
  "add        %3, %3, %4                     \n" \
  "ld1        {" #vn ".s}[" #n "], [%6]      \n"

void ScaleARGBCols_NEON(uint8_t* dst_argb,
                        const uint8_t* src_argb,
                        int dst_width,
                        int x,
                        int dx) {
  const uint8_t* src_tmp = src_argb;
  int64_t x64 = (int64_t)x;    // NOLINT
  int64_t dx64 = (int64_t)dx;  // NOLINT
  int64_t tmp64;
  asm volatile (
      "1:          \n"
      // clang-format off
      LOAD1_DATA32_LANE(v0, 0)
      LOAD1_DATA32_LANE(v0, 1)
      LOAD1_DATA32_LANE(v0, 2)
      LOAD1_DATA32_LANE(v0, 3)
      LOAD1_DATA32_LANE(v1, 0)
      LOAD1_DATA32_LANE(v1, 1)
      LOAD1_DATA32_LANE(v1, 2)
      LOAD1_DATA32_LANE(v1, 3)
      "prfm        pldl1keep, [%1, 448]          \n"  // prefetch 7 lines ahead
      // clang-format on
      "st1         {v0.4s, v1.4s}, [%0], #32     \n"  // store pixels
      "subs        %w2, %w2, #8                  \n"  // 8 processed per loop
      "b.gt        1b                            \n"
      : "+r"(dst_argb),   // %0
        "+r"(src_argb),   // %1
        "+r"(dst_width),  // %2
        "+r"(x64),        // %3
        "+r"(dx64),       // %4
        "=&r"(tmp64),     // %5
        "+r"(src_tmp)     // %6
      :
      : "memory", "cc", "v0", "v1");
}

#undef LOAD1_DATA32_LANE

static const uvec8 kScaleARGBFilterColsShuffleIndices = {
    0, 0, 0, 0, 2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6,
};

#define SCALE_ARGB_FILTER_COLS_STEP_ADDR         \
  "lsr        %5, %3, #16                    \n" \
  "add        %6, %1, %5, lsl #2             \n" \
  "add        %3, %3, %4                     \n"

void ScaleARGBFilterCols_NEON(uint8_t* dst_argb,
                              const uint8_t* src_argb,
                              int dst_width,
                              int x,
                              int dx) {
  int dx_offset[4] = {0, 1, 2, 3};
  int64_t tmp;
  const uint8_t* src_tmp = src_argb;
  int64_t x64 = (int64_t)x;
  int64_t dx64 = (int64_t)dx;
  asm volatile(
      "dup         v0.4s, %w3                    \n"
      "dup         v1.4s, %w4                    \n"
      "ld1         {v2.4s}, [%[kOffsets]]        \n"
      "shl         v6.4s, v1.4s, #2              \n"
      "mul         v1.4s, v1.4s, v2.4s           \n"
      "movi        v3.16b, #0x7f                 \n"

      "add         v5.4s, v1.4s, v0.4s           \n"
      "ldr         q18, [%[kIndices]]            \n"

      "1:          \n"  //
      SCALE_ARGB_FILTER_COLS_STEP_ADDR
      "ldr         d1, [%6]                      \n"  //
      SCALE_ARGB_FILTER_COLS_STEP_ADDR
      "ldr         d2, [%6]                      \n"
      "shrn        v4.4h, v5.4s, #9              \n"  //
      SCALE_ARGB_FILTER_COLS_STEP_ADDR
      "ld1         {v1.d}[1], [%6]               \n"  //
      SCALE_ARGB_FILTER_COLS_STEP_ADDR
      "ld1         {v2.d}[1], [%6]               \n"

      "subs        %w2, %w2, #4                  \n"  // 4 processed per loop
      "and         v4.8b, v4.8b, v3.8b           \n"
      "trn1        v0.4s, v1.4s, v2.4s           \n"
      "tbl         v4.16b, {v4.16b}, v18.16b     \n"  // f
      "trn2        v1.4s, v1.4s, v2.4s           \n"
      "eor         v7.16b, v4.16b, v3.16b        \n"  // 0x7f ^ f

      "umull       v16.8h, v1.8b, v4.8b          \n"
      "umull2      v17.8h, v1.16b, v4.16b        \n"
      "umlal       v16.8h, v0.8b, v7.8b          \n"
      "umlal2      v17.8h, v0.16b, v7.16b        \n"

      "prfm        pldl1keep, [%1, 448]          \n"  // prefetch 7 lines ahead
      "shrn        v0.8b, v16.8h, #7             \n"
      "shrn        v1.8b, v17.8h, #7             \n"
      "add         v5.4s, v5.4s, v6.4s           \n"
      "stp         d0, d1, [%0], #16             \n"  // store pixels
      "b.gt        1b                            \n"
      : "+r"(dst_argb),                                       // %0
        "+r"(src_argb),                                       // %1
        "+r"(dst_width),                                      // %2
        "+r"(x64),                                            // %3
        "+r"(dx64),                                           // %4
        "=&r"(tmp),                                           // %5
        "+r"(src_tmp)                                         // %6
      : [kIndices] "r"(&kScaleARGBFilterColsShuffleIndices),  // %[kIndices]
        [kOffsets] "r"(dx_offset)                             // %[kOffsets]
      : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16",
        "v17", "v18", "v19");
}

#undef SCALE_ARGB_FILTER_COLS_STEP_ADDR

void ScaleRowDown2_16_NEON(const uint16_t* src_ptr,
                           ptrdiff_t src_stride,
                           uint16_t* dst,
                           int dst_width) {
  (void)src_stride;
  asm volatile(
      "subs        %w[dst_width], %w[dst_width], #32 \n"
      "b.lt        2f                            \n"

      "1:          \n"
      "ldp         q0, q1, [%[src_ptr]]          \n"
      "ldp         q2, q3, [%[src_ptr], #32]     \n"
      "ldp         q4, q5, [%[src_ptr], #64]     \n"
      "ldp         q6, q7, [%[src_ptr], #96]     \n"
      "add         %[src_ptr], %[src_ptr], #128  \n"
      "uzp2        v0.8h, v0.8h, v1.8h           \n"
      "uzp2        v1.8h, v2.8h, v3.8h           \n"
      "uzp2        v2.8h, v4.8h, v5.8h           \n"
      "uzp2        v3.8h, v6.8h, v7.8h           \n"
      "subs        %w[dst_width], %w[dst_width], #32 \n"  // 32 elems per
                                                          // iteration.
      "stp         q0, q1, [%[dst_ptr]]          \n"
      "stp         q2, q3, [%[dst_ptr], #32]     \n"
      "add         %[dst_ptr], %[dst_ptr], #64   \n"
      "b.ge        1b                            \n"

      "2:          \n"
      "adds        %w[dst_width], %w[dst_width], #32 \n"
      "b.eq        99f                           \n"

      "ldp         q0, q1, [%[src_ptr]]          \n"
      "ldp         q2, q3, [%[src_ptr], #32]     \n"
      "uzp2        v0.8h, v0.8h, v1.8h           \n"
      "uzp2        v1.8h, v2.8h, v3.8h           \n"
      "stp         q0, q1, [%[dst_ptr]]          \n"

      "99:         \n"
      : [src_ptr] "+r"(src_ptr),     // %[src_ptr]
        [dst_ptr] "+r"(dst),         // %[dst_ptr]
        [dst_width] "+r"(dst_width)  // %[dst_width]
      :
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
}

void ScaleRowDown2Linear_16_NEON(const uint16_t* src_ptr,
                                 ptrdiff_t src_stride,
                                 uint16_t* dst,
                                 int dst_width) {
  (void)src_stride;
  asm volatile(
      "1:          \n"
      "ld2         {v0.8h, v1.8h}, [%[src_ptr]], #32 \n"
      "ld2         {v2.8h, v3.8h}, [%[src_ptr]], #32 \n"
      "subs        %w[dst_width], %w[dst_width], #16 \n"
      "urhadd      v0.8h, v0.8h, v1.8h           \n"
      "urhadd      v1.8h, v2.8h, v3.8h           \n"
      "prfm        pldl1keep, [%[src_ptr], 448]  \n"
      "stp         q0, q1, [%[dst_ptr]], #32     \n"
      "b.gt        1b                            \n"
      : [src_ptr] "+r"(src_ptr),     // %[src_ptr]
        [dst_ptr] "+r"(dst),         // %[dst_ptr]
        [dst_width] "+r"(dst_width)  // %[dst_width]
      :
      : "memory", "cc", "v0", "v1", "v2", "v3");
}

// Read 16x2 average down and write 8x1.
void ScaleRowDown2Box_16_NEON(const uint16_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint16_t* dst,
                              int dst_width) {
  asm volatile(
      // change the stride to row 2 pointer
      "add         %1, %0, %1, lsl #1            \n"  // ptr + stide * 2
      "1:          \n"
      "ld1         {v0.8h, v1.8h}, [%0], #32     \n"  // load row 1 and post inc
      "ld1         {v2.8h, v3.8h}, [%1], #32     \n"  // load row 2 and post inc
      "subs        %w3, %w3, #8                  \n"  // 8 processed per loop
      "uaddlp      v0.4s, v0.8h                  \n"  // row 1 add adjacent
      "uaddlp      v1.4s, v1.8h                  \n"
      "prfm        pldl1keep, [%0, 448]          \n"  // prefetch 7 lines ahead
      "uadalp      v0.4s, v2.8h                  \n"  // +row 2 add adjacent
      "uadalp      v1.4s, v3.8h                  \n"
      "prfm        pldl1keep, [%1, 448]          \n"
      "rshrn       v0.4h, v0.4s, #2              \n"  // round and pack
      "rshrn2      v0.8h, v1.4s, #2              \n"
      "st1         {v0.8h}, [%2], #16            \n"
      "b.gt        1b                            \n"
      : "+r"(src_ptr),     // %0
        "+r"(src_stride),  // %1
        "+r"(dst),         // %2
        "+r"(dst_width)    // %3
      :
      : "memory", "cc", "v0", "v1", "v2", "v3"  // Clobber List
  );
}

void ScaleUVRowDown2_NEON(const uint8_t* src_ptr,
                          ptrdiff_t src_stride,
                          uint8_t* dst,
                          int dst_width) {
  (void)src_stride;
  asm volatile(
      "1:          \n"
      "ld2         {v0.8h,v1.8h}, [%0], #32      \n"  // load 16 UV
      "subs        %w2, %w2, #8                  \n"  // 8 processed per loop.
      "prfm        pldl1keep, [%0, 448]          \n"  // prefetch 7 lines ahead
      "st1         {v1.8h}, [%1], #16            \n"  // store 8 UV
      "b.gt        1b                            \n"
      : "+r"(src_ptr),   // %0
        "+r"(dst),       // %1
        "+r"(dst_width)  // %2
      :
      : "memory", "cc", "v0", "v1");
}

void ScaleUVRowDown2Linear_NEON(const uint8_t* src_ptr,
                                ptrdiff_t src_stride,
                                uint8_t* dst,
                                int dst_width) {
  (void)src_stride;
  asm volatile(
      "1:          \n"
      "ld2         {v0.8h,v1.8h}, [%0], #32      \n"  // load 16 UV
      "subs        %w2, %w2, #8                  \n"  // 8 processed per loop.
      "urhadd      v0.16b, v0.16b, v1.16b        \n"  // rounding half add
      "prfm        pldl1keep, [%0, 448]          \n"  // prefetch 7 lines ahead
      "st1         {v0.8h}, [%1], #16            \n"  // store 8 UV
      "b.gt        1b                            \n"
      : "+r"(src_ptr),   // %0
        "+r"(dst),       // %1
        "+r"(dst_width)  // %2
      :
      : "memory", "cc", "v0", "v1");
}

void ScaleUVRowDown2Box_NEON(const uint8_t* src_ptr,
                             ptrdiff_t src_stride,
                             uint8_t* dst,
                             int dst_width) {
  asm volatile(
      // change the stride to row 2 pointer
      "add         %1, %1, %0                    \n"
      "1:          \n"
      "ld2         {v0.16b,v1.16b}, [%0], #32    \n"  // load 16 UV
      "subs        %w3, %w3, #8                  \n"  // 8 processed per loop.
      "uaddlp      v0.8h, v0.16b                 \n"  // U 16 bytes -> 8 shorts.
      "uaddlp      v1.8h, v1.16b                 \n"  // V 16 bytes -> 8 shorts.
      "ld2         {v16.16b,v17.16b}, [%1], #32  \n"  // load 16
      "uadalp      v0.8h, v16.16b                \n"  // U 16 bytes -> 8 shorts.
      "uadalp      v1.8h, v17.16b                \n"  // V 16 bytes -> 8 shorts.
      "prfm        pldl1keep, [%0, 448]          \n"  // prefetch 7 lines ahead
      "rshrn       v0.8b, v0.8h, #2              \n"  // round and pack
      "prfm        pldl1keep, [%1, 448]          \n"
      "rshrn       v1.8b, v1.8h, #2              \n"
      "st2         {v0.8b,v1.8b}, [%2], #16      \n"
      "b.gt        1b                            \n"
      : "+r"(src_ptr),     // %0
        "+r"(src_stride),  // %1
        "+r"(dst),         // %2
        "+r"(dst_width)    // %3
      :
      : "memory", "cc", "v0", "v1", "v16", "v17");
}

// Reads 4 pixels at a time.
void ScaleUVRowDownEven_NEON(const uint8_t* src_ptr,
                             ptrdiff_t src_stride,
                             int src_stepx,  // pixel step
                             uint8_t* dst_ptr,
                             int dst_width) {
  const uint8_t* src1_ptr = src_ptr + src_stepx * 2;
  const uint8_t* src2_ptr = src_ptr + src_stepx * 4;
  const uint8_t* src3_ptr = src_ptr + src_stepx * 6;
  (void)src_stride;
  asm volatile(
      "1:          \n"
      "ld1         {v0.h}[0], [%0], %6           \n"
      "ld1         {v1.h}[0], [%1], %6           \n"
      "ld1         {v2.h}[0], [%2], %6           \n"
      "ld1         {v3.h}[0], [%3], %6           \n"
      "subs        %w5, %w5, #4                  \n"  // 4 pixels per loop.
      "st4         {v0.h, v1.h, v2.h, v3.h}[0], [%4], #8 \n"
      "b.gt        1b                            \n"
      : "+r"(src_ptr),                 // %0
        "+r"(src1_ptr),                // %1
        "+r"(src2_ptr),                // %2
        "+r"(src3_ptr),                // %3
        "+r"(dst_ptr),                 // %4
        "+r"(dst_width)                // %5
      : "r"((int64_t)(src_stepx * 8))  // %6
      : "memory", "cc", "v0", "v1", "v2", "v3");
}

#endif  // !defined(LIBYUV_DISABLE_NEON) && defined(__aarch64__)

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
