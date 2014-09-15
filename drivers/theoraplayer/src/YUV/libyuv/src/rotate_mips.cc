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

#include "libyuv/basic_types.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

#if !defined(LIBYUV_DISABLE_MIPS) && \
    defined(__mips_dsp) && (__mips_dsp_rev >= 2)

void TransposeWx8_MIPS_DSPR2(const uint8* src, int src_stride,
                             uint8* dst, int dst_stride,
                             int width) {
   __asm__ __volatile__ (
      ".set push                                         \n"
      ".set noreorder                                    \n"
      "sll              $t2, %[src_stride], 0x1          \n" // src_stride x 2
      "sll              $t4, %[src_stride], 0x2          \n" // src_stride x 4
      "sll              $t9, %[src_stride], 0x3          \n" // src_stride x 8
      "addu             $t3, $t2, %[src_stride]          \n"
      "addu             $t5, $t4, %[src_stride]          \n"
      "addu             $t6, $t2, $t4                    \n"
      "andi             $t0, %[dst], 0x3                 \n"
      "andi             $t1, %[dst_stride], 0x3          \n"
      "or               $t0, $t0, $t1                    \n"
      "bnez             $t0, 11f                         \n"
      " subu            $t7, $t9, %[src_stride]          \n"
//dst + dst_stride word aligned
    "1:                                                  \n"
      "lbu              $t0, 0(%[src])                   \n"
      "lbux             $t1, %[src_stride](%[src])       \n"
      "lbux             $t8, $t2(%[src])                 \n"
      "lbux             $t9, $t3(%[src])                 \n"
      "sll              $t1, $t1, 16                     \n"
      "sll              $t9, $t9, 16                     \n"
      "or               $t0, $t0, $t1                    \n"
      "or               $t8, $t8, $t9                    \n"
      "precr.qb.ph      $s0, $t8, $t0                    \n"
      "lbux             $t0, $t4(%[src])                 \n"
      "lbux             $t1, $t5(%[src])                 \n"
      "lbux             $t8, $t6(%[src])                 \n"
      "lbux             $t9, $t7(%[src])                 \n"
      "sll              $t1, $t1, 16                     \n"
      "sll              $t9, $t9, 16                     \n"
      "or               $t0, $t0, $t1                    \n"
      "or               $t8, $t8, $t9                    \n"
      "precr.qb.ph      $s1, $t8, $t0                    \n"
      "sw               $s0, 0(%[dst])                   \n"
      "addiu            %[width], -1                     \n"
      "addiu            %[src], 1                        \n"
      "sw               $s1, 4(%[dst])                   \n"
      "bnez             %[width], 1b                     \n"
      " addu            %[dst], %[dst], %[dst_stride]    \n"
      "b                2f                               \n"
//dst + dst_stride unaligned
   "11:                                                  \n"
      "lbu              $t0, 0(%[src])                   \n"
      "lbux             $t1, %[src_stride](%[src])       \n"
      "lbux             $t8, $t2(%[src])                 \n"
      "lbux             $t9, $t3(%[src])                 \n"
      "sll              $t1, $t1, 16                     \n"
      "sll              $t9, $t9, 16                     \n"
      "or               $t0, $t0, $t1                    \n"
      "or               $t8, $t8, $t9                    \n"
      "precr.qb.ph      $s0, $t8, $t0                    \n"
      "lbux             $t0, $t4(%[src])                 \n"
      "lbux             $t1, $t5(%[src])                 \n"
      "lbux             $t8, $t6(%[src])                 \n"
      "lbux             $t9, $t7(%[src])                 \n"
      "sll              $t1, $t1, 16                     \n"
      "sll              $t9, $t9, 16                     \n"
      "or               $t0, $t0, $t1                    \n"
      "or               $t8, $t8, $t9                    \n"
      "precr.qb.ph      $s1, $t8, $t0                    \n"
      "swr              $s0, 0(%[dst])                   \n"
      "swl              $s0, 3(%[dst])                   \n"
      "addiu            %[width], -1                     \n"
      "addiu            %[src], 1                        \n"
      "swr              $s1, 4(%[dst])                   \n"
      "swl              $s1, 7(%[dst])                   \n"
      "bnez             %[width], 11b                    \n"
       "addu             %[dst], %[dst], %[dst_stride]   \n"
    "2:                                                  \n"
      ".set pop                                          \n"
      :[src] "+r" (src),
       [dst] "+r" (dst),
       [width] "+r" (width)
      :[src_stride] "r" (src_stride),
       [dst_stride] "r" (dst_stride)
      : "t0", "t1",  "t2", "t3", "t4", "t5",
        "t6", "t7", "t8", "t9",
        "s0", "s1"
  );
}

void TransposeWx8_FAST_MIPS_DSPR2(const uint8* src, int src_stride,
                                  uint8* dst, int dst_stride,
                                  int width) {
  __asm__ __volatile__ (
      ".set noat                                         \n"
      ".set push                                         \n"
      ".set noreorder                                    \n"
      "beqz             %[width], 2f                     \n"
      " sll             $t2, %[src_stride], 0x1          \n"  // src_stride x 2
      "sll              $t4, %[src_stride], 0x2          \n"  // src_stride x 4
      "sll              $t9, %[src_stride], 0x3          \n"  // src_stride x 8
      "addu             $t3, $t2, %[src_stride]          \n"
      "addu             $t5, $t4, %[src_stride]          \n"
      "addu             $t6, $t2, $t4                    \n"

      "srl              $AT, %[width], 0x2               \n"
      "andi             $t0, %[dst], 0x3                 \n"
      "andi             $t1, %[dst_stride], 0x3          \n"
      "or               $t0, $t0, $t1                    \n"
      "bnez             $t0, 11f                         \n"
      " subu            $t7, $t9, %[src_stride]          \n"
//dst + dst_stride word aligned
      "1:                                                \n"
      "lw               $t0, 0(%[src])                   \n"
      "lwx              $t1, %[src_stride](%[src])       \n"
      "lwx              $t8, $t2(%[src])                 \n"
      "lwx              $t9, $t3(%[src])                 \n"

// t0 = | 30 | 20 | 10 | 00 |
// t1 = | 31 | 21 | 11 | 01 |
// t8 = | 32 | 22 | 12 | 02 |
// t9 = | 33 | 23 | 13 | 03 |

      "precr.qb.ph     $s0, $t1, $t0                     \n"
      "precr.qb.ph     $s1, $t9, $t8                     \n"
      "precrq.qb.ph    $s2, $t1, $t0                     \n"
      "precrq.qb.ph    $s3, $t9, $t8                     \n"

  // s0 = | 21 | 01 | 20 | 00 |
  // s1 = | 23 | 03 | 22 | 02 |
  // s2 = | 31 | 11 | 30 | 10 |
  // s3 = | 33 | 13 | 32 | 12 |

      "precr.qb.ph     $s4, $s1, $s0                     \n"
      "precrq.qb.ph    $s5, $s1, $s0                     \n"
      "precr.qb.ph     $s6, $s3, $s2                     \n"
      "precrq.qb.ph    $s7, $s3, $s2                     \n"

  // s4 = | 03 | 02 | 01 | 00 |
  // s5 = | 23 | 22 | 21 | 20 |
  // s6 = | 13 | 12 | 11 | 10 |
  // s7 = | 33 | 32 | 31 | 30 |

      "lwx              $t0, $t4(%[src])                 \n"
      "lwx              $t1, $t5(%[src])                 \n"
      "lwx              $t8, $t6(%[src])                 \n"
      "lwx              $t9, $t7(%[src])                 \n"

// t0 = | 34 | 24 | 14 | 04 |
// t1 = | 35 | 25 | 15 | 05 |
// t8 = | 36 | 26 | 16 | 06 |
// t9 = | 37 | 27 | 17 | 07 |

      "precr.qb.ph     $s0, $t1, $t0                     \n"
      "precr.qb.ph     $s1, $t9, $t8                     \n"
      "precrq.qb.ph    $s2, $t1, $t0                     \n"
      "precrq.qb.ph    $s3, $t9, $t8                     \n"

  // s0 = | 25 | 05 | 24 | 04 |
  // s1 = | 27 | 07 | 26 | 06 |
  // s2 = | 35 | 15 | 34 | 14 |
  // s3 = | 37 | 17 | 36 | 16 |

      "precr.qb.ph     $t0, $s1, $s0                     \n"
      "precrq.qb.ph    $t1, $s1, $s0                     \n"
      "precr.qb.ph     $t8, $s3, $s2                     \n"
      "precrq.qb.ph    $t9, $s3, $s2                     \n"

  // t0 = | 07 | 06 | 05 | 04 |
  // t1 = | 27 | 26 | 25 | 24 |
  // t8 = | 17 | 16 | 15 | 14 |
  // t9 = | 37 | 36 | 35 | 34 |

      "addu            $s0, %[dst], %[dst_stride]        \n"
      "addu            $s1, $s0, %[dst_stride]           \n"
      "addu            $s2, $s1, %[dst_stride]           \n"

      "sw              $s4, 0(%[dst])                    \n"
      "sw              $t0, 4(%[dst])                    \n"
      "sw              $s6, 0($s0)                       \n"
      "sw              $t8, 4($s0)                       \n"
      "sw              $s5, 0($s1)                       \n"
      "sw              $t1, 4($s1)                       \n"
      "sw              $s7, 0($s2)                       \n"
      "sw              $t9, 4($s2)                       \n"

      "addiu            $AT, -1                          \n"
      "addiu            %[src], 4                        \n"

      "bnez             $AT, 1b                          \n"
      " addu            %[dst], $s2, %[dst_stride]       \n"
      "b                2f                               \n"
//dst + dst_stride unaligned
      "11:                                               \n"
      "lw               $t0, 0(%[src])                   \n"
      "lwx              $t1, %[src_stride](%[src])       \n"
      "lwx              $t8, $t2(%[src])                 \n"
      "lwx              $t9, $t3(%[src])                 \n"

// t0 = | 30 | 20 | 10 | 00 |
// t1 = | 31 | 21 | 11 | 01 |
// t8 = | 32 | 22 | 12 | 02 |
// t9 = | 33 | 23 | 13 | 03 |

      "precr.qb.ph     $s0, $t1, $t0                     \n"
      "precr.qb.ph     $s1, $t9, $t8                     \n"
      "precrq.qb.ph    $s2, $t1, $t0                     \n"
      "precrq.qb.ph    $s3, $t9, $t8                     \n"

  // s0 = | 21 | 01 | 20 | 00 |
  // s1 = | 23 | 03 | 22 | 02 |
  // s2 = | 31 | 11 | 30 | 10 |
  // s3 = | 33 | 13 | 32 | 12 |

      "precr.qb.ph     $s4, $s1, $s0                     \n"
      "precrq.qb.ph    $s5, $s1, $s0                     \n"
      "precr.qb.ph     $s6, $s3, $s2                     \n"
      "precrq.qb.ph    $s7, $s3, $s2                     \n"

  // s4 = | 03 | 02 | 01 | 00 |
  // s5 = | 23 | 22 | 21 | 20 |
  // s6 = | 13 | 12 | 11 | 10 |
  // s7 = | 33 | 32 | 31 | 30 |

      "lwx              $t0, $t4(%[src])                 \n"
      "lwx              $t1, $t5(%[src])                 \n"
      "lwx              $t8, $t6(%[src])                 \n"
      "lwx              $t9, $t7(%[src])                 \n"

// t0 = | 34 | 24 | 14 | 04 |
// t1 = | 35 | 25 | 15 | 05 |
// t8 = | 36 | 26 | 16 | 06 |
// t9 = | 37 | 27 | 17 | 07 |

      "precr.qb.ph     $s0, $t1, $t0                     \n"
      "precr.qb.ph     $s1, $t9, $t8                     \n"
      "precrq.qb.ph    $s2, $t1, $t0                     \n"
      "precrq.qb.ph    $s3, $t9, $t8                     \n"

  // s0 = | 25 | 05 | 24 | 04 |
  // s1 = | 27 | 07 | 26 | 06 |
  // s2 = | 35 | 15 | 34 | 14 |
  // s3 = | 37 | 17 | 36 | 16 |

      "precr.qb.ph     $t0, $s1, $s0                     \n"
      "precrq.qb.ph    $t1, $s1, $s0                     \n"
      "precr.qb.ph     $t8, $s3, $s2                     \n"
      "precrq.qb.ph    $t9, $s3, $s2                     \n"

  // t0 = | 07 | 06 | 05 | 04 |
  // t1 = | 27 | 26 | 25 | 24 |
  // t8 = | 17 | 16 | 15 | 14 |
  // t9 = | 37 | 36 | 35 | 34 |

      "addu            $s0, %[dst], %[dst_stride]        \n"
      "addu            $s1, $s0, %[dst_stride]           \n"
      "addu            $s2, $s1, %[dst_stride]           \n"

      "swr              $s4, 0(%[dst])                   \n"
      "swl              $s4, 3(%[dst])                   \n"
      "swr              $t0, 4(%[dst])                   \n"
      "swl              $t0, 7(%[dst])                   \n"
      "swr              $s6, 0($s0)                      \n"
      "swl              $s6, 3($s0)                      \n"
      "swr              $t8, 4($s0)                      \n"
      "swl              $t8, 7($s0)                      \n"
      "swr              $s5, 0($s1)                      \n"
      "swl              $s5, 3($s1)                      \n"
      "swr              $t1, 4($s1)                      \n"
      "swl              $t1, 7($s1)                      \n"
      "swr              $s7, 0($s2)                      \n"
      "swl              $s7, 3($s2)                      \n"
      "swr              $t9, 4($s2)                      \n"
      "swl              $t9, 7($s2)                      \n"

      "addiu            $AT, -1                          \n"
      "addiu            %[src], 4                        \n"

      "bnez             $AT, 11b                         \n"
      " addu            %[dst], $s2, %[dst_stride]       \n"
      "2:                                                \n"
      ".set pop                                          \n"
      ".set at                                           \n"
      :[src] "+r" (src),
       [dst] "+r" (dst),
       [width] "+r" (width)
      :[src_stride] "r" (src_stride),
       [dst_stride] "r" (dst_stride)
      : "t0", "t1",  "t2", "t3",  "t4", "t5",
        "t6", "t7", "t8", "t9",
        "s0", "s1", "s2", "s3", "s4",
        "s5", "s6", "s7"
  );
}

void TransposeUVWx8_MIPS_DSPR2(const uint8* src, int src_stride,
                               uint8* dst_a, int dst_stride_a,
                               uint8* dst_b, int dst_stride_b,
                               int width) {
  __asm__ __volatile__ (
      ".set push                                         \n"
      ".set noreorder                                    \n"
      "beqz            %[width], 2f                      \n"
      " sll            $t2, %[src_stride], 0x1           \n" // src_stride x 2
      "sll             $t4, %[src_stride], 0x2           \n" // src_stride x 4
      "sll             $t9, %[src_stride], 0x3           \n" // src_stride x 8
      "addu            $t3, $t2, %[src_stride]           \n"
      "addu            $t5, $t4, %[src_stride]           \n"
      "addu            $t6, $t2, $t4                     \n"
      "subu            $t7, $t9, %[src_stride]           \n"
      "srl             $t1, %[width], 1                  \n"

// check word aligment for dst_a, dst_b, dst_stride_a and dst_stride_b
      "andi            $t0, %[dst_a], 0x3                \n"
      "andi            $t8, %[dst_b], 0x3                \n"
      "or              $t0, $t0, $t8                     \n"
      "andi            $t8, %[dst_stride_a], 0x3         \n"
      "andi            $s5, %[dst_stride_b], 0x3         \n"
      "or              $t8, $t8, $s5                     \n"
      "or              $t0, $t0, $t8                     \n"
      "bnez            $t0, 11f                          \n"
      " nop                                              \n"
// dst + dst_stride word aligned (both, a & b dst addresses)
    "1:                                                  \n"
      "lw              $t0, 0(%[src])                    \n" // |B0|A0|b0|a0|
      "lwx             $t8, %[src_stride](%[src])        \n" // |B1|A1|b1|a1|
      "addu            $s5, %[dst_a], %[dst_stride_a]    \n"
      "lwx             $t9, $t2(%[src])                  \n" // |B2|A2|b2|a2|
      "lwx             $s0, $t3(%[src])                  \n" // |B3|A3|b3|a3|
      "addu            $s6, %[dst_b], %[dst_stride_b]    \n"

      "precrq.ph.w     $s1, $t8, $t0                     \n" // |B1|A1|B0|A0|
      "precrq.ph.w     $s2, $s0, $t9                     \n" // |B3|A3|B2|A2|
      "precr.qb.ph     $s3, $s2, $s1                     \n" // |A3|A2|A1|A0|
      "precrq.qb.ph    $s4, $s2, $s1                     \n" // |B3|B2|B1|B0|

      "sll             $t0, $t0, 16                      \n"
      "packrl.ph       $s1, $t8, $t0                     \n" // |b1|a1|b0|a0|
      "sll             $t9, $t9, 16                      \n"
      "packrl.ph       $s2, $s0, $t9                     \n" // |b3|a3|b2|a2|

      "sw              $s3, 0($s5)                       \n"
      "sw              $s4, 0($s6)                       \n"

      "precr.qb.ph     $s3, $s2, $s1                     \n" // |a3|a2|a1|a0|
      "precrq.qb.ph    $s4, $s2, $s1                     \n" // |b3|b2|b1|b0|

      "lwx             $t0, $t4(%[src])                  \n" // |B4|A4|b4|a4|
      "lwx             $t8, $t5(%[src])                  \n" // |B5|A5|b5|a5|
      "lwx             $t9, $t6(%[src])                  \n" // |B6|A6|b6|a6|
      "lwx             $s0, $t7(%[src])                  \n" // |B7|A7|b7|a7|
      "sw              $s3, 0(%[dst_a])                  \n"
      "sw              $s4, 0(%[dst_b])                  \n"

      "precrq.ph.w     $s1, $t8, $t0                     \n" // |B5|A5|B4|A4|
      "precrq.ph.w     $s2, $s0, $t9                     \n" // |B6|A6|B7|A7|
      "precr.qb.ph     $s3, $s2, $s1                     \n" // |A7|A6|A5|A4|
      "precrq.qb.ph    $s4, $s2, $s1                     \n" // |B7|B6|B5|B4|

      "sll             $t0, $t0, 16                      \n"
      "packrl.ph       $s1, $t8, $t0                     \n" // |b5|a5|b4|a4|
      "sll             $t9, $t9, 16                      \n"
      "packrl.ph       $s2, $s0, $t9                     \n" // |b7|a7|b6|a6|
      "sw              $s3, 4($s5)                       \n"
      "sw              $s4, 4($s6)                       \n"

      "precr.qb.ph     $s3, $s2, $s1                     \n" // |a7|a6|a5|a4|
      "precrq.qb.ph    $s4, $s2, $s1                     \n" // |b7|b6|b5|b4|

      "addiu           %[src], 4                         \n"
      "addiu           $t1, -1                           \n"
      "sll             $t0, %[dst_stride_a], 1           \n"
      "sll             $t8, %[dst_stride_b], 1           \n"
      "sw              $s3, 4(%[dst_a])                  \n"
      "sw              $s4, 4(%[dst_b])                  \n"
      "addu            %[dst_a], %[dst_a], $t0           \n"
      "bnez            $t1, 1b                           \n"
      " addu           %[dst_b], %[dst_b], $t8           \n"
      "b               2f                                \n"
      " nop                                              \n"

// dst_a or dst_b or dst_stride_a or dst_stride_b not word aligned
   "11:                                                  \n"
      "lw              $t0, 0(%[src])                    \n" // |B0|A0|b0|a0|
      "lwx             $t8, %[src_stride](%[src])        \n" // |B1|A1|b1|a1|
      "addu            $s5, %[dst_a], %[dst_stride_a]    \n"
      "lwx             $t9, $t2(%[src])                  \n" // |B2|A2|b2|a2|
      "lwx             $s0, $t3(%[src])                  \n" // |B3|A3|b3|a3|
      "addu            $s6, %[dst_b], %[dst_stride_b]    \n"

      "precrq.ph.w     $s1, $t8, $t0                     \n" // |B1|A1|B0|A0|
      "precrq.ph.w     $s2, $s0, $t9                     \n" // |B3|A3|B2|A2|
      "precr.qb.ph     $s3, $s2, $s1                     \n" // |A3|A2|A1|A0|
      "precrq.qb.ph    $s4, $s2, $s1                     \n" // |B3|B2|B1|B0|

      "sll             $t0, $t0, 16                      \n"
      "packrl.ph       $s1, $t8, $t0                     \n" // |b1|a1|b0|a0|
      "sll             $t9, $t9, 16                      \n"
      "packrl.ph       $s2, $s0, $t9                     \n" // |b3|a3|b2|a2|

      "swr             $s3, 0($s5)                       \n"
      "swl             $s3, 3($s5)                       \n"
      "swr             $s4, 0($s6)                       \n"
      "swl             $s4, 3($s6)                       \n"

      "precr.qb.ph     $s3, $s2, $s1                     \n" // |a3|a2|a1|a0|
      "precrq.qb.ph    $s4, $s2, $s1                     \n" // |b3|b2|b1|b0|

      "lwx             $t0, $t4(%[src])                  \n" // |B4|A4|b4|a4|
      "lwx             $t8, $t5(%[src])                  \n" // |B5|A5|b5|a5|
      "lwx             $t9, $t6(%[src])                  \n" // |B6|A6|b6|a6|
      "lwx             $s0, $t7(%[src])                  \n" // |B7|A7|b7|a7|
      "swr             $s3, 0(%[dst_a])                  \n"
      "swl             $s3, 3(%[dst_a])                  \n"
      "swr             $s4, 0(%[dst_b])                  \n"
      "swl             $s4, 3(%[dst_b])                  \n"

      "precrq.ph.w     $s1, $t8, $t0                     \n" // |B5|A5|B4|A4|
      "precrq.ph.w     $s2, $s0, $t9                     \n" // |B6|A6|B7|A7|
      "precr.qb.ph     $s3, $s2, $s1                     \n" // |A7|A6|A5|A4|
      "precrq.qb.ph    $s4, $s2, $s1                     \n" // |B7|B6|B5|B4|

      "sll             $t0, $t0, 16                      \n"
      "packrl.ph       $s1, $t8, $t0                     \n" // |b5|a5|b4|a4|
      "sll             $t9, $t9, 16                      \n"
      "packrl.ph       $s2, $s0, $t9                     \n" // |b7|a7|b6|a6|

      "swr             $s3, 4($s5)                       \n"
      "swl             $s3, 7($s5)                       \n"
      "swr             $s4, 4($s6)                       \n"
      "swl             $s4, 7($s6)                       \n"

      "precr.qb.ph     $s3, $s2, $s1                     \n" // |a7|a6|a5|a4|
      "precrq.qb.ph    $s4, $s2, $s1                     \n" // |b7|b6|b5|b4|

      "addiu           %[src], 4                         \n"
      "addiu           $t1, -1                           \n"
      "sll             $t0, %[dst_stride_a], 1           \n"
      "sll             $t8, %[dst_stride_b], 1           \n"
      "swr             $s3, 4(%[dst_a])                  \n"
      "swl             $s3, 7(%[dst_a])                  \n"
      "swr             $s4, 4(%[dst_b])                  \n"
      "swl             $s4, 7(%[dst_b])                  \n"
      "addu            %[dst_a], %[dst_a], $t0           \n"
      "bnez            $t1, 11b                          \n"
      " addu           %[dst_b], %[dst_b], $t8           \n"

      "2:                                                \n"
      ".set pop                                          \n"
      : [src] "+r" (src),
        [dst_a] "+r" (dst_a),
        [dst_b] "+r" (dst_b),
        [width] "+r" (width),
        [src_stride] "+r" (src_stride)
      : [dst_stride_a] "r" (dst_stride_a),
        [dst_stride_b] "r" (dst_stride_b)
      : "t0", "t1",  "t2", "t3",  "t4", "t5",
        "t6", "t7", "t8", "t9",
        "s0", "s1", "s2", "s3",
        "s4", "s5", "s6"
  );
}

#endif  // defined(__mips_dsp) && (__mips_dsp_rev >= 2)

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
