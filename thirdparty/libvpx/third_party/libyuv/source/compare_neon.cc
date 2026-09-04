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

#include "libyuv/compare_row.h"
#include "libyuv/row.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

#if !defined(LIBYUV_DISABLE_NEON) && defined(__ARM_NEON__) && \
    !defined(__aarch64__)

// 256 bits at a time
// uses short accumulator which restricts count to 131 KB
uint32_t HammingDistance_NEON(const uint8_t* src_a,
                              const uint8_t* src_b,
                              int count) {
  uint32_t diff;

  asm volatile(
      "vmov.u16   q4, #0                         \n"  // accumulator

      "1:                                        \n"
      "vld1.8     {q0, q1}, [%0]!                \n"
      "vld1.8     {q2, q3}, [%1]!                \n"
      "veor.32    q0, q0, q2                     \n"
      "veor.32    q1, q1, q3                     \n"
      "vcnt.i8    q0, q0                         \n"
      "vcnt.i8    q1, q1                         \n"
      "subs       %2, %2, #32                    \n"
      "vadd.u8    q0, q0, q1                     \n"  // 16 byte counts
      "vpadal.u8  q4, q0                         \n"  // 8 shorts
      "bgt        1b                             \n"

      "vpaddl.u16 q0, q4                         \n"  // 4 ints
      "vpadd.u32  d0, d0, d1                     \n"
      "vpadd.u32  d0, d0, d0                     \n"
      "vmov.32    %3, d0[0]                      \n"

      : "+r"(src_a), "+r"(src_b), "+r"(count), "=r"(diff)
      :
      : "cc", "q0", "q1", "q2", "q3", "q4");
  return diff;
}

uint32_t SumSquareError_NEON(const uint8_t* src_a,
                             const uint8_t* src_b,
                             int count) {
  uint32_t sse;
  asm volatile(
      "vmov.u8    q8, #0                         \n"
      "vmov.u8    q10, #0                        \n"
      "vmov.u8    q9, #0                         \n"
      "vmov.u8    q11, #0                        \n"

      "1:                                        \n"
      "vld1.8     {q0}, [%0]!                    \n"
      "vld1.8     {q1}, [%1]!                    \n"
      "subs       %2, %2, #16                    \n"
      "vsubl.u8   q2, d0, d2                     \n"
      "vsubl.u8   q3, d1, d3                     \n"
      "vmlal.s16  q8, d4, d4                     \n"
      "vmlal.s16  q9, d6, d6                     \n"
      "vmlal.s16  q10, d5, d5                    \n"
      "vmlal.s16  q11, d7, d7                    \n"
      "bgt        1b                             \n"

      "vadd.u32   q8, q8, q9                     \n"
      "vadd.u32   q10, q10, q11                  \n"
      "vadd.u32   q11, q8, q10                   \n"
      "vpaddl.u32 q1, q11                        \n"
      "vadd.u64   d0, d2, d3                     \n"
      "vmov.32    %3, d0[0]                      \n"
      : "+r"(src_a), "+r"(src_b), "+r"(count), "=r"(sse)
      :
      : "memory", "cc", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11");
  return sse;
}

#endif  // defined(__ARM_NEON__) && !defined(__aarch64__)

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
