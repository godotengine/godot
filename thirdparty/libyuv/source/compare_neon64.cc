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

#if !defined(LIBYUV_DISABLE_NEON) && defined(__aarch64__)

// 256 bits at a time
// uses short accumulator which restricts count to 131 KB
uint32_t HammingDistance_NEON(const uint8_t* src_a,
                              const uint8_t* src_b,
                              int count) {
  uint32_t diff;
  asm volatile(
      "movi        v4.8h, #0                     \n"

      "1:          \n"
      "ld1         {v0.16b, v1.16b}, [%0], #32   \n"
      "ld1         {v2.16b, v3.16b}, [%1], #32   \n"
      "eor         v0.16b, v0.16b, v2.16b        \n"
      "prfm        pldl1keep, [%0, 448]          \n"  // prefetch 7 lines ahead
      "eor         v1.16b, v1.16b, v3.16b        \n"
      "cnt         v0.16b, v0.16b                \n"
      "prfm        pldl1keep, [%1, 448]          \n"
      "cnt         v1.16b, v1.16b                \n"
      "subs        %w2, %w2, #32                 \n"
      "add         v0.16b, v0.16b, v1.16b        \n"
      "uadalp      v4.8h, v0.16b                 \n"
      "b.gt        1b                            \n"

      "uaddlv      s4, v4.8h                     \n"
      "fmov        %w3, s4                       \n"
      : "+r"(src_a), "+r"(src_b), "+r"(count), "=r"(diff)
      :
      : "memory", "cc", "v0", "v1", "v2", "v3", "v4");
  return diff;
}

uint32_t SumSquareError_NEON(const uint8_t* src_a,
                             const uint8_t* src_b,
                             int count) {
  uint32_t sse;
  asm volatile(
      "movi        v16.16b, #0                   \n"
      "movi        v17.16b, #0                   \n"
      "movi        v18.16b, #0                   \n"
      "movi        v19.16b, #0                   \n"

      "1:          \n"
      "ld1         {v0.16b}, [%0], #16           \n"
      "ld1         {v1.16b}, [%1], #16           \n"
      "subs        %w2, %w2, #16                 \n"
      "usubl       v2.8h, v0.8b, v1.8b           \n"
      "usubl2      v3.8h, v0.16b, v1.16b         \n"
      "prfm        pldl1keep, [%0, 448]          \n"  // prefetch 7 lines ahead
      "smlal       v16.4s, v2.4h, v2.4h          \n"
      "smlal       v17.4s, v3.4h, v3.4h          \n"
      "prfm        pldl1keep, [%1, 448]          \n"
      "smlal2      v18.4s, v2.8h, v2.8h          \n"
      "smlal2      v19.4s, v3.8h, v3.8h          \n"
      "b.gt        1b                            \n"

      "add         v16.4s, v16.4s, v17.4s        \n"
      "add         v18.4s, v18.4s, v19.4s        \n"
      "add         v19.4s, v16.4s, v18.4s        \n"
      "addv        s0, v19.4s                    \n"
      "fmov        %w3, s0                       \n"
      : "+r"(src_a), "+r"(src_b), "+r"(count), "=r"(sse)
      :
      : "memory", "cc", "v0", "v1", "v2", "v3", "v16", "v17", "v18", "v19");
  return sse;
}

static const uvec32 kDjb2Multiplicands[] = {
    {0x0c3525e1,   // 33^15
     0xa3476dc1,   // 33^14
     0x3b4039a1,   // 33^13
     0x4f5f0981},  // 33^12
    {0x30f35d61,   // 33^11
     0x855cb541,   // 33^10
     0x040a9121,   // 33^9
     0x747c7101},  // 33^8
    {0xec41d4e1,   // 33^7
     0x4cfa3cc1,   // 33^6
     0x025528a1,   // 33^5
     0x00121881},  // 33^4
    {0x00008c61,   // 33^3
     0x00000441,   // 33^2
     0x00000021,   // 33^1
     0x00000001},  // 33^0
};

static const uvec32 kDjb2WidenIndices[] = {
    {0xffffff00U, 0xffffff01U, 0xffffff02U, 0xffffff03U},
    {0xffffff04U, 0xffffff05U, 0xffffff06U, 0xffffff07U},
    {0xffffff08U, 0xffffff09U, 0xffffff0aU, 0xffffff0bU},
    {0xffffff0cU, 0xffffff0dU, 0xffffff0eU, 0xffffff0fU},
};

uint32_t HashDjb2_NEON(const uint8_t* src, int count, uint32_t seed) {
  uint32_t hash = seed;
  const uint32_t c16 = 0x92d9e201;  // 33^16
  uint32_t tmp, tmp2;
      asm("ld1         {v16.4s, v17.4s, v18.4s, v19.4s}, [%[kIdx]] \n"
      "ld1         {v4.4s, v5.4s, v6.4s, v7.4s}, [%[kMuls]] \n"

      // count is always a multiple of 16.
      // maintain two accumulators, reduce and then final sum in scalar since
      // this has better performance on little cores.
      "1:          \n"
      "ldr         q0, [%[src]], #16             \n"
      "subs        %w[count], %w[count], #16     \n"
      "tbl         v3.16b, {v0.16b}, v19.16b     \n"
      "tbl         v2.16b, {v0.16b}, v18.16b     \n"
      "tbl         v1.16b, {v0.16b}, v17.16b     \n"
      "tbl         v0.16b, {v0.16b}, v16.16b     \n"
      "mul         v3.4s, v3.4s, v7.4s           \n"
      "mul         v2.4s, v2.4s, v6.4s           \n"
      "mla         v3.4s, v1.4s, v5.4s           \n"
      "mla         v2.4s, v0.4s, v4.4s           \n"
      "addv        s1, v3.4s                     \n"
      "addv        s0, v2.4s                     \n"
      "fmov        %w[tmp2], s1                  \n"
      "fmov        %w[tmp], s0                   \n"
      "add         %w[tmp], %w[tmp], %w[tmp2]    \n"
      "madd        %w[hash], %w[hash], %w[c16], %w[tmp] \n"
      "b.gt        1b                            \n"
      : [hash] "+r"(hash),                // %[hash]
        [count] "+r"(count),              // %[count]
        [tmp] "=&r"(tmp),                 // %[tmp]
        [tmp2] "=&r"(tmp2)                // %[tmp2]
      : [src] "r"(src),                   // %[src]
        [kMuls] "r"(kDjb2Multiplicands),  // %[kMuls]
        [kIdx] "r"(kDjb2WidenIndices),    // %[kIdx]
        [c16] "r"(c16)                    // %[c16]
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16",
        "v17", "v18", "v19");
  return hash;
}

uint32_t HammingDistance_NEON_DotProd(const uint8_t* src_a,
                                      const uint8_t* src_b,
                                      int count) {
  uint32_t diff;
  asm volatile(
      "movi        v4.4s, #0                     \n"
      "movi        v5.4s, #0                     \n"
      "movi        v6.16b, #1                    \n"

      "1:          \n"
      "ldp         q0, q1, [%0], #32             \n"
      "ldp         q2, q3, [%1], #32             \n"
      "eor         v0.16b, v0.16b, v2.16b        \n"
      "prfm        pldl1keep, [%0, 448]          \n"  // prefetch 7 lines ahead
      "eor         v1.16b, v1.16b, v3.16b        \n"
      "cnt         v0.16b, v0.16b                \n"
      "prfm        pldl1keep, [%1, 448]          \n"
      "cnt         v1.16b, v1.16b                \n"
      "subs        %w2, %w2, #32                 \n"
      "udot        v4.4s, v0.16b, v6.16b         \n"
      "udot        v5.4s, v1.16b, v6.16b         \n"
      "b.gt        1b                            \n"

      "add         v0.4s, v4.4s, v5.4s           \n"
      "addv        s0, v0.4s                     \n"
      "fmov        %w3, s0                       \n"
      : "+r"(src_a), "+r"(src_b), "+r"(count), "=r"(diff)
      :
      : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6");
  return diff;
}

uint32_t SumSquareError_NEON_DotProd(const uint8_t* src_a,
                                     const uint8_t* src_b,
                                     int count) {
  // count is guaranteed to be a multiple of 32.
  uint32_t sse;
  asm volatile(
      "movi        v4.4s, #0                     \n"
      "movi        v5.4s, #0                     \n"

      "1:          \n"
      "ldp         q0, q2, [%0], #32             \n"
      "ldp         q1, q3, [%1], #32             \n"
      "subs        %w2, %w2, #32                 \n"
      "uabd        v0.16b, v0.16b, v1.16b        \n"
      "uabd        v1.16b, v2.16b, v3.16b        \n"
      "prfm        pldl1keep, [%0, 448]          \n"  // prefetch 7 lines ahead
      "udot        v4.4s, v0.16b, v0.16b         \n"
      "udot        v5.4s, v1.16b, v1.16b         \n"
      "prfm        pldl1keep, [%1, 448]          \n"
      "b.gt        1b                            \n"

      "add         v0.4s, v4.4s, v5.4s           \n"
      "addv        s0, v0.4s                     \n"
      "fmov        %w3, s0                       \n"
      : "+r"(src_a), "+r"(src_b), "+r"(count), "=r"(sse)
      :
      : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5");
  return sse;
}

#endif  // !defined(LIBYUV_DISABLE_NEON) && defined(__aarch64__)

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
