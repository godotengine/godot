/*
 *  Copyright (c) 2012 The LibYuv project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "libyuv/row.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// The following are available on Mips platforms:
#if !defined(LIBYUV_DISABLE_MIPS) && defined(__mips__) && \
    (_MIPS_SIM == _MIPS_SIM_ABI32)

#ifdef HAS_COPYROW_MIPS
void CopyRow_MIPS(const uint8* src, uint8* dst, int count) {
  __asm__ __volatile__ (
    ".set      noreorder                         \n"
    ".set      noat                              \n"
    "slti      $at, %[count], 8                  \n"
    "bne       $at ,$zero, $last8                \n"
    "xor       $t8, %[src], %[dst]               \n"
    "andi      $t8, $t8, 0x3                     \n"

    "bne       $t8, $zero, unaligned             \n"
    "negu      $a3, %[dst]                       \n"
    // make dst/src aligned
    "andi      $a3, $a3, 0x3                     \n"
    "beq       $a3, $zero, $chk16w               \n"
    // word-aligned now count is the remining bytes count
    "subu     %[count], %[count], $a3            \n"

    "lwr       $t8, 0(%[src])                    \n"
    "addu      %[src], %[src], $a3               \n"
    "swr       $t8, 0(%[dst])                    \n"
    "addu      %[dst], %[dst], $a3               \n"

    // Now the dst/src are mutually word-aligned with word-aligned addresses
    "$chk16w:                                    \n"
    "andi      $t8, %[count], 0x3f               \n"  // whole 64-B chunks?
    // t8 is the byte count after 64-byte chunks
    "beq       %[count], $t8, chk8w              \n"
    // There will be at most 1 32-byte chunk after it
    "subu      $a3, %[count], $t8                \n"  // the reminder
    // Here a3 counts bytes in 16w chunks
    "addu      $a3, %[dst], $a3                  \n"
    // Now a3 is the final dst after 64-byte chunks
    "addu      $t0, %[dst], %[count]             \n"
    // t0 is the "past the end" address

    // When in the loop we exercise "pref 30,x(a1)", the a1+x should not be past
    // the "t0-32" address
    // This means: for x=128 the last "safe" a1 address is "t0-160"
    // Alternatively, for x=64 the last "safe" a1 address is "t0-96"
    // we will use "pref 30,128(a1)", so "t0-160" is the limit
    "subu      $t9, $t0, 160                     \n"
    // t9 is the "last safe pref 30,128(a1)" address
    "pref      0, 0(%[src])                      \n"  // first line of src
    "pref      0, 32(%[src])                     \n"  // second line of src
    "pref      0, 64(%[src])                     \n"
    "pref      30, 32(%[dst])                    \n"
    // In case the a1 > t9 don't use "pref 30" at all
    "sgtu      $v1, %[dst], $t9                  \n"
    "bgtz      $v1, $loop16w                     \n"
    "nop                                         \n"
    // otherwise, start with using pref30
    "pref      30, 64(%[dst])                    \n"
    "$loop16w:                                    \n"
    "pref      0, 96(%[src])                     \n"
    "lw        $t0, 0(%[src])                    \n"
    "bgtz      $v1, $skip_pref30_96              \n"  // skip
    "lw        $t1, 4(%[src])                    \n"
    "pref      30, 96(%[dst])                    \n"  // continue
    "$skip_pref30_96:                            \n"
    "lw        $t2, 8(%[src])                    \n"
    "lw        $t3, 12(%[src])                   \n"
    "lw        $t4, 16(%[src])                   \n"
    "lw        $t5, 20(%[src])                   \n"
    "lw        $t6, 24(%[src])                   \n"
    "lw        $t7, 28(%[src])                   \n"
    "pref      0, 128(%[src])                    \n"
    //  bring the next lines of src, addr 128
    "sw        $t0, 0(%[dst])                    \n"
    "sw        $t1, 4(%[dst])                    \n"
    "sw        $t2, 8(%[dst])                    \n"
    "sw        $t3, 12(%[dst])                   \n"
    "sw        $t4, 16(%[dst])                   \n"
    "sw        $t5, 20(%[dst])                   \n"
    "sw        $t6, 24(%[dst])                   \n"
    "sw        $t7, 28(%[dst])                   \n"
    "lw        $t0, 32(%[src])                   \n"
    "bgtz      $v1, $skip_pref30_128             \n"  // skip pref 30,128(a1)
    "lw        $t1, 36(%[src])                   \n"
    "pref      30, 128(%[dst])                   \n"  // set dest, addr 128
    "$skip_pref30_128:                           \n"
    "lw        $t2, 40(%[src])                   \n"
    "lw        $t3, 44(%[src])                   \n"
    "lw        $t4, 48(%[src])                   \n"
    "lw        $t5, 52(%[src])                   \n"
    "lw        $t6, 56(%[src])                   \n"
    "lw        $t7, 60(%[src])                   \n"
    "pref      0, 160(%[src])                    \n"
    // bring the next lines of src, addr 160
    "sw        $t0, 32(%[dst])                   \n"
    "sw        $t1, 36(%[dst])                   \n"
    "sw        $t2, 40(%[dst])                   \n"
    "sw        $t3, 44(%[dst])                   \n"
    "sw        $t4, 48(%[dst])                   \n"
    "sw        $t5, 52(%[dst])                   \n"
    "sw        $t6, 56(%[dst])                   \n"
    "sw        $t7, 60(%[dst])                   \n"

    "addiu     %[dst], %[dst], 64                \n"  // adding 64 to dest
    "sgtu      $v1, %[dst], $t9                  \n"
    "bne       %[dst], $a3, $loop16w             \n"
    " addiu    %[src], %[src], 64                \n"  // adding 64 to src
    "move      %[count], $t8                     \n"

    // Here we have src and dest word-aligned but less than 64-bytes to go

    "chk8w:                                      \n"
    "pref      0, 0x0(%[src])                    \n"
    "andi      $t8, %[count], 0x1f               \n"  // 32-byte chunk?
    // the t8 is the reminder count past 32-bytes
    "beq       %[count], $t8, chk1w              \n"
    // count=t8,no 32-byte chunk
    " nop                                        \n"

    "lw        $t0, 0(%[src])                    \n"
    "lw        $t1, 4(%[src])                    \n"
    "lw        $t2, 8(%[src])                    \n"
    "lw        $t3, 12(%[src])                   \n"
    "lw        $t4, 16(%[src])                   \n"
    "lw        $t5, 20(%[src])                   \n"
    "lw        $t6, 24(%[src])                   \n"
    "lw        $t7, 28(%[src])                   \n"
    "addiu     %[src], %[src], 32                \n"

    "sw        $t0, 0(%[dst])                    \n"
    "sw        $t1, 4(%[dst])                    \n"
    "sw        $t2, 8(%[dst])                    \n"
    "sw        $t3, 12(%[dst])                   \n"
    "sw        $t4, 16(%[dst])                   \n"
    "sw        $t5, 20(%[dst])                   \n"
    "sw        $t6, 24(%[dst])                   \n"
    "sw        $t7, 28(%[dst])                   \n"
    "addiu     %[dst], %[dst], 32                \n"

    "chk1w:                                      \n"
    "andi      %[count], $t8, 0x3                \n"
    // now count is the reminder past 1w chunks
    "beq       %[count], $t8, $last8             \n"
    " subu     $a3, $t8, %[count]                \n"
    // a3 is count of bytes in 1w chunks
    "addu      $a3, %[dst], $a3                  \n"
    // now a3 is the dst address past the 1w chunks
    // copying in words (4-byte chunks)
    "$wordCopy_loop:                             \n"
    "lw        $t3, 0(%[src])                    \n"
    // the first t3 may be equal t0 ... optimize?
    "addiu     %[src], %[src],4                  \n"
    "addiu     %[dst], %[dst],4                  \n"
    "bne       %[dst], $a3,$wordCopy_loop        \n"
    " sw       $t3, -4(%[dst])                   \n"

    // For the last (<8) bytes
    "$last8:                                     \n"
    "blez      %[count], leave                   \n"
    " addu     $a3, %[dst], %[count]             \n"  // a3 -last dst address
    "$last8loop:                                 \n"
    "lb        $v1, 0(%[src])                    \n"
    "addiu     %[src], %[src], 1                 \n"
    "addiu     %[dst], %[dst], 1                 \n"
    "bne       %[dst], $a3, $last8loop           \n"
    " sb       $v1, -1(%[dst])                   \n"

    "leave:                                      \n"
    "  j       $ra                               \n"
    "  nop                                       \n"

    //
    // UNALIGNED case
    //

    "unaligned:                                  \n"
    // got here with a3="negu a1"
    "andi      $a3, $a3, 0x3                     \n"  // a1 is word aligned?
    "beqz      $a3, $ua_chk16w                   \n"
    " subu     %[count], %[count], $a3           \n"
    // bytes left after initial a3 bytes
    "lwr       $v1, 0(%[src])                    \n"
    "lwl       $v1, 3(%[src])                    \n"
    "addu      %[src], %[src], $a3               \n"  // a3 may be 1, 2 or 3
    "swr       $v1, 0(%[dst])                    \n"
    "addu      %[dst], %[dst], $a3               \n"
    // below the dst will be word aligned (NOTE1)
    "$ua_chk16w:                                 \n"
    "andi      $t8, %[count], 0x3f               \n"  // whole 64-B chunks?
    // t8 is the byte count after 64-byte chunks
    "beq       %[count], $t8, ua_chk8w           \n"
    // if a2==t8, no 64-byte chunks
    // There will be at most 1 32-byte chunk after it
    "subu      $a3, %[count], $t8                \n"  // the reminder
    // Here a3 counts bytes in 16w chunks
    "addu      $a3, %[dst], $a3                  \n"
    // Now a3 is the final dst after 64-byte chunks
    "addu      $t0, %[dst], %[count]             \n"  // t0 "past the end"
    "subu      $t9, $t0, 160                     \n"
    // t9 is the "last safe pref 30,128(a1)" address
    "pref      0, 0(%[src])                      \n"  // first line of src
    "pref      0, 32(%[src])                     \n"  // second line  addr 32
    "pref      0, 64(%[src])                     \n"
    "pref      30, 32(%[dst])                    \n"
    // safe, as we have at least 64 bytes ahead
    // In case the a1 > t9 don't use "pref 30" at all
    "sgtu      $v1, %[dst], $t9                  \n"
    "bgtz      $v1, $ua_loop16w                  \n"
    // skip "pref 30,64(a1)" for too short arrays
    " nop                                        \n"
    // otherwise, start with using pref30
    "pref      30, 64(%[dst])                    \n"
    "$ua_loop16w:                                \n"
    "pref      0, 96(%[src])                     \n"
    "lwr       $t0, 0(%[src])                    \n"
    "lwl       $t0, 3(%[src])                    \n"
    "lwr       $t1, 4(%[src])                    \n"
    "bgtz      $v1, $ua_skip_pref30_96           \n"
    " lwl      $t1, 7(%[src])                    \n"
    "pref      30, 96(%[dst])                    \n"
    // continue setting up the dest, addr 96
    "$ua_skip_pref30_96:                         \n"
    "lwr       $t2, 8(%[src])                    \n"
    "lwl       $t2, 11(%[src])                   \n"
    "lwr       $t3, 12(%[src])                   \n"
    "lwl       $t3, 15(%[src])                   \n"
    "lwr       $t4, 16(%[src])                   \n"
    "lwl       $t4, 19(%[src])                   \n"
    "lwr       $t5, 20(%[src])                   \n"
    "lwl       $t5, 23(%[src])                   \n"
    "lwr       $t6, 24(%[src])                   \n"
    "lwl       $t6, 27(%[src])                   \n"
    "lwr       $t7, 28(%[src])                   \n"
    "lwl       $t7, 31(%[src])                   \n"
    "pref      0, 128(%[src])                    \n"
    // bring the next lines of src, addr 128
    "sw        $t0, 0(%[dst])                    \n"
    "sw        $t1, 4(%[dst])                    \n"
    "sw        $t2, 8(%[dst])                    \n"
    "sw        $t3, 12(%[dst])                   \n"
    "sw        $t4, 16(%[dst])                   \n"
    "sw        $t5, 20(%[dst])                   \n"
    "sw        $t6, 24(%[dst])                   \n"
    "sw        $t7, 28(%[dst])                   \n"
    "lwr       $t0, 32(%[src])                   \n"
    "lwl       $t0, 35(%[src])                   \n"
    "lwr       $t1, 36(%[src])                   \n"
    "bgtz      $v1, ua_skip_pref30_128           \n"
    " lwl      $t1, 39(%[src])                   \n"
    "pref      30, 128(%[dst])                   \n"
    // continue setting up the dest, addr 128
    "ua_skip_pref30_128:                         \n"

    "lwr       $t2, 40(%[src])                   \n"
    "lwl       $t2, 43(%[src])                   \n"
    "lwr       $t3, 44(%[src])                   \n"
    "lwl       $t3, 47(%[src])                   \n"
    "lwr       $t4, 48(%[src])                   \n"
    "lwl       $t4, 51(%[src])                   \n"
    "lwr       $t5, 52(%[src])                   \n"
    "lwl       $t5, 55(%[src])                   \n"
    "lwr       $t6, 56(%[src])                   \n"
    "lwl       $t6, 59(%[src])                   \n"
    "lwr       $t7, 60(%[src])                   \n"
    "lwl       $t7, 63(%[src])                   \n"
    "pref      0, 160(%[src])                    \n"
    // bring the next lines of src, addr 160
    "sw        $t0, 32(%[dst])                   \n"
    "sw        $t1, 36(%[dst])                   \n"
    "sw        $t2, 40(%[dst])                   \n"
    "sw        $t3, 44(%[dst])                   \n"
    "sw        $t4, 48(%[dst])                   \n"
    "sw        $t5, 52(%[dst])                   \n"
    "sw        $t6, 56(%[dst])                   \n"
    "sw        $t7, 60(%[dst])                   \n"

    "addiu     %[dst],%[dst],64                  \n"  // adding 64 to dest
    "sgtu      $v1,%[dst],$t9                    \n"
    "bne       %[dst],$a3,$ua_loop16w            \n"
    " addiu    %[src],%[src],64                  \n"  // adding 64 to src
    "move      %[count],$t8                      \n"

    // Here we have src and dest word-aligned but less than 64-bytes to go

    "ua_chk8w:                                   \n"
    "pref      0, 0x0(%[src])                    \n"
    "andi      $t8, %[count], 0x1f               \n"  // 32-byte chunk?
    // the t8 is the reminder count
    "beq       %[count], $t8, $ua_chk1w          \n"
    // when count==t8, no 32-byte chunk

    "lwr       $t0, 0(%[src])                    \n"
    "lwl       $t0, 3(%[src])                    \n"
    "lwr       $t1, 4(%[src])                    \n"
    "lwl       $t1, 7(%[src])                    \n"
    "lwr       $t2, 8(%[src])                    \n"
    "lwl       $t2, 11(%[src])                   \n"
    "lwr       $t3, 12(%[src])                   \n"
    "lwl       $t3, 15(%[src])                   \n"
    "lwr       $t4, 16(%[src])                   \n"
    "lwl       $t4, 19(%[src])                   \n"
    "lwr       $t5, 20(%[src])                   \n"
    "lwl       $t5, 23(%[src])                   \n"
    "lwr       $t6, 24(%[src])                   \n"
    "lwl       $t6, 27(%[src])                   \n"
    "lwr       $t7, 28(%[src])                   \n"
    "lwl       $t7, 31(%[src])                   \n"
    "addiu     %[src], %[src], 32                \n"

    "sw        $t0, 0(%[dst])                    \n"
    "sw        $t1, 4(%[dst])                    \n"
    "sw        $t2, 8(%[dst])                    \n"
    "sw        $t3, 12(%[dst])                   \n"
    "sw        $t4, 16(%[dst])                   \n"
    "sw        $t5, 20(%[dst])                   \n"
    "sw        $t6, 24(%[dst])                   \n"
    "sw        $t7, 28(%[dst])                   \n"
    "addiu     %[dst], %[dst], 32                \n"

    "$ua_chk1w:                                  \n"
    "andi      %[count], $t8, 0x3                \n"
    // now count is the reminder past 1w chunks
    "beq       %[count], $t8, ua_smallCopy       \n"
    "subu      $a3, $t8, %[count]                \n"
    // a3 is count of bytes in 1w chunks
    "addu      $a3, %[dst], $a3                  \n"
    // now a3 is the dst address past the 1w chunks

    // copying in words (4-byte chunks)
    "$ua_wordCopy_loop:                          \n"
    "lwr       $v1, 0(%[src])                    \n"
    "lwl       $v1, 3(%[src])                    \n"
    "addiu     %[src], %[src], 4                 \n"
    "addiu     %[dst], %[dst], 4                 \n"
    // note: dst=a1 is word aligned here, see NOTE1
    "bne       %[dst], $a3, $ua_wordCopy_loop    \n"
    " sw       $v1,-4(%[dst])                    \n"

    // Now less than 4 bytes (value in count) left to copy
    "ua_smallCopy:                               \n"
    "beqz      %[count], leave                   \n"
    " addu     $a3, %[dst], %[count]             \n" // a3 = last dst address
    "$ua_smallCopy_loop:                         \n"
    "lb        $v1, 0(%[src])                    \n"
    "addiu     %[src], %[src], 1                 \n"
    "addiu     %[dst], %[dst], 1                 \n"
    "bne       %[dst],$a3,$ua_smallCopy_loop     \n"
    " sb       $v1, -1(%[dst])                   \n"

    "j         $ra                               \n"
    " nop                                        \n"
    ".set      at                                \n"
    ".set      reorder                           \n"
       : [dst] "+r" (dst), [src] "+r" (src)
       : [count] "r" (count)
       : "t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7",
       "t8", "t9", "a3", "v1", "at"
  );
}
#endif  // HAS_COPYROW_MIPS

// DSPR2 functions
#if !defined(LIBYUV_DISABLE_MIPS) && defined(__mips_dsp) && \
    (__mips_dsp_rev >= 2) && \
    (_MIPS_SIM == _MIPS_SIM_ABI32) && (__mips_isa_rev < 6)

void SplitUVRow_DSPR2(const uint8* src_uv, uint8* dst_u, uint8* dst_v,
                      int width) {
  __asm__ __volatile__ (
    ".set push                                     \n"
    ".set noreorder                                \n"
    "srl             $t4, %[width], 4              \n"  // multiplies of 16
    "blez            $t4, 2f                       \n"
    " andi           %[width], %[width], 0xf       \n"  // residual

  "1:                                              \n"
    "addiu           $t4, $t4, -1                  \n"
    "lw              $t0, 0(%[src_uv])             \n"  // V1 | U1 | V0 | U0
    "lw              $t1, 4(%[src_uv])             \n"  // V3 | U3 | V2 | U2
    "lw              $t2, 8(%[src_uv])             \n"  // V5 | U5 | V4 | U4
    "lw              $t3, 12(%[src_uv])            \n"  // V7 | U7 | V6 | U6
    "lw              $t5, 16(%[src_uv])            \n"  // V9 | U9 | V8 | U8
    "lw              $t6, 20(%[src_uv])            \n"  // V11 | U11 | V10 | U10
    "lw              $t7, 24(%[src_uv])            \n"  // V13 | U13 | V12 | U12
    "lw              $t8, 28(%[src_uv])            \n"  // V15 | U15 | V14 | U14
    "addiu           %[src_uv], %[src_uv], 32      \n"
    "precrq.qb.ph    $t9, $t1, $t0                 \n"  // V3 | V2 | V1 | V0
    "precr.qb.ph     $t0, $t1, $t0                 \n"  // U3 | U2 | U1 | U0
    "precrq.qb.ph    $t1, $t3, $t2                 \n"  // V7 | V6 | V5 | V4
    "precr.qb.ph     $t2, $t3, $t2                 \n"  // U7 | U6 | U5 | U4
    "precrq.qb.ph    $t3, $t6, $t5                 \n"  // V11 | V10 | V9 | V8
    "precr.qb.ph     $t5, $t6, $t5                 \n"  // U11 | U10 | U9 | U8
    "precrq.qb.ph    $t6, $t8, $t7                 \n"  // V15 | V14 | V13 | V12
    "precr.qb.ph     $t7, $t8, $t7                 \n"  // U15 | U14 | U13 | U12
    "sw              $t9, 0(%[dst_v])              \n"
    "sw              $t0, 0(%[dst_u])              \n"
    "sw              $t1, 4(%[dst_v])              \n"
    "sw              $t2, 4(%[dst_u])              \n"
    "sw              $t3, 8(%[dst_v])              \n"
    "sw              $t5, 8(%[dst_u])              \n"
    "sw              $t6, 12(%[dst_v])             \n"
    "sw              $t7, 12(%[dst_u])             \n"
    "addiu           %[dst_v], %[dst_v], 16        \n"
    "bgtz            $t4, 1b                       \n"
    " addiu          %[dst_u], %[dst_u], 16        \n"

    "beqz            %[width], 3f                  \n"
    " nop                                          \n"

  "2:                                              \n"
    "lbu             $t0, 0(%[src_uv])             \n"
    "lbu             $t1, 1(%[src_uv])             \n"
    "addiu           %[src_uv], %[src_uv], 2       \n"
    "addiu           %[width], %[width], -1        \n"
    "sb              $t0, 0(%[dst_u])              \n"
    "sb              $t1, 0(%[dst_v])              \n"
    "addiu           %[dst_u], %[dst_u], 1         \n"
    "bgtz            %[width], 2b                  \n"
    " addiu          %[dst_v], %[dst_v], 1         \n"

  "3:                                              \n"
    ".set pop                                      \n"
     : [src_uv] "+r" (src_uv),
       [width] "+r" (width),
       [dst_u] "+r" (dst_u),
       [dst_v] "+r" (dst_v)
     :
     : "t0", "t1", "t2", "t3",
     "t4", "t5", "t6", "t7", "t8", "t9"
  );
}

void MirrorRow_DSPR2(const uint8* src, uint8* dst, int width) {
  __asm__ __volatile__ (
    ".set push                             \n"
    ".set noreorder                        \n"

    "srl       $t4, %[width], 4            \n"  // multiplies of 16
    "andi      $t5, %[width], 0xf          \n"
    "blez      $t4, 2f                     \n"
    " addu     %[src], %[src], %[width]    \n"  // src += width

   "1:                                     \n"
    "lw        $t0, -16(%[src])            \n"  // |3|2|1|0|
    "lw        $t1, -12(%[src])            \n"  // |7|6|5|4|
    "lw        $t2, -8(%[src])             \n"  // |11|10|9|8|
    "lw        $t3, -4(%[src])             \n"  // |15|14|13|12|
    "wsbh      $t0, $t0                    \n"  // |2|3|0|1|
    "wsbh      $t1, $t1                    \n"  // |6|7|4|5|
    "wsbh      $t2, $t2                    \n"  // |10|11|8|9|
    "wsbh      $t3, $t3                    \n"  // |14|15|12|13|
    "rotr      $t0, $t0, 16                \n"  // |0|1|2|3|
    "rotr      $t1, $t1, 16                \n"  // |4|5|6|7|
    "rotr      $t2, $t2, 16                \n"  // |8|9|10|11|
    "rotr      $t3, $t3, 16                \n"  // |12|13|14|15|
    "addiu     %[src], %[src], -16         \n"
    "addiu     $t4, $t4, -1                \n"
    "sw        $t3, 0(%[dst])              \n"  // |15|14|13|12|
    "sw        $t2, 4(%[dst])              \n"  // |11|10|9|8|
    "sw        $t1, 8(%[dst])              \n"  // |7|6|5|4|
    "sw        $t0, 12(%[dst])             \n"  // |3|2|1|0|
    "bgtz      $t4, 1b                     \n"
    " addiu    %[dst], %[dst], 16          \n"
    "beqz      $t5, 3f                     \n"
    " nop                                  \n"

   "2:                                     \n"
    "lbu       $t0, -1(%[src])             \n"
    "addiu     $t5, $t5, -1                \n"
    "addiu     %[src], %[src], -1          \n"
    "sb        $t0, 0(%[dst])              \n"
    "bgez      $t5, 2b                     \n"
    " addiu    %[dst], %[dst], 1           \n"

   "3:                                     \n"
    ".set pop                              \n"
      : [src] "+r" (src), [dst] "+r" (dst)
      : [width] "r" (width)
      : "t0", "t1", "t2", "t3", "t4", "t5"
  );
}

void MirrorUVRow_DSPR2(const uint8* src_uv, uint8* dst_u, uint8* dst_v,
                       int width) {
  int x;
  int y;
  __asm__ __volatile__ (
    ".set push                                    \n"
    ".set noreorder                               \n"

    "addu            $t4, %[width], %[width]      \n"
    "srl             %[x], %[width], 4            \n"
    "andi            %[y], %[width], 0xf          \n"
    "blez            %[x], 2f                     \n"
    " addu           %[src_uv], %[src_uv], $t4    \n"

   "1:                                            \n"
    "lw              $t0, -32(%[src_uv])          \n"  // |3|2|1|0|
    "lw              $t1, -28(%[src_uv])          \n"  // |7|6|5|4|
    "lw              $t2, -24(%[src_uv])          \n"  // |11|10|9|8|
    "lw              $t3, -20(%[src_uv])          \n"  // |15|14|13|12|
    "lw              $t4, -16(%[src_uv])          \n"  // |19|18|17|16|
    "lw              $t6, -12(%[src_uv])          \n"  // |23|22|21|20|
    "lw              $t7, -8(%[src_uv])           \n"  // |27|26|25|24|
    "lw              $t8, -4(%[src_uv])           \n"  // |31|30|29|28|

    "rotr            $t0, $t0, 16                 \n"  // |1|0|3|2|
    "rotr            $t1, $t1, 16                 \n"  // |5|4|7|6|
    "rotr            $t2, $t2, 16                 \n"  // |9|8|11|10|
    "rotr            $t3, $t3, 16                 \n"  // |13|12|15|14|
    "rotr            $t4, $t4, 16                 \n"  // |17|16|19|18|
    "rotr            $t6, $t6, 16                 \n"  // |21|20|23|22|
    "rotr            $t7, $t7, 16                 \n"  // |25|24|27|26|
    "rotr            $t8, $t8, 16                 \n"  // |29|28|31|30|
    "precr.qb.ph     $t9, $t0, $t1                \n"  // |0|2|4|6|
    "precrq.qb.ph    $t5, $t0, $t1                \n"  // |1|3|5|7|
    "precr.qb.ph     $t0, $t2, $t3                \n"  // |8|10|12|14|
    "precrq.qb.ph    $t1, $t2, $t3                \n"  // |9|11|13|15|
    "precr.qb.ph     $t2, $t4, $t6                \n"  // |16|18|20|22|
    "precrq.qb.ph    $t3, $t4, $t6                \n"  // |17|19|21|23|
    "precr.qb.ph     $t4, $t7, $t8                \n"  // |24|26|28|30|
    "precrq.qb.ph    $t6, $t7, $t8                \n"  // |25|27|29|31|
    "addiu           %[src_uv], %[src_uv], -32    \n"
    "addiu           %[x], %[x], -1               \n"
    "swr             $t4, 0(%[dst_u])             \n"
    "swl             $t4, 3(%[dst_u])             \n"  // |30|28|26|24|
    "swr             $t6, 0(%[dst_v])             \n"
    "swl             $t6, 3(%[dst_v])             \n"  // |31|29|27|25|
    "swr             $t2, 4(%[dst_u])             \n"
    "swl             $t2, 7(%[dst_u])             \n"  // |22|20|18|16|
    "swr             $t3, 4(%[dst_v])             \n"
    "swl             $t3, 7(%[dst_v])             \n"  // |23|21|19|17|
    "swr             $t0, 8(%[dst_u])             \n"
    "swl             $t0, 11(%[dst_u])            \n"  // |14|12|10|8|
    "swr             $t1, 8(%[dst_v])             \n"
    "swl             $t1, 11(%[dst_v])            \n"  // |15|13|11|9|
    "swr             $t9, 12(%[dst_u])            \n"
    "swl             $t9, 15(%[dst_u])            \n"  // |6|4|2|0|
    "swr             $t5, 12(%[dst_v])            \n"
    "swl             $t5, 15(%[dst_v])            \n"  // |7|5|3|1|
    "addiu           %[dst_v], %[dst_v], 16       \n"
    "bgtz            %[x], 1b                     \n"
    " addiu          %[dst_u], %[dst_u], 16       \n"
    "beqz            %[y], 3f                     \n"
    " nop                                         \n"
    "b               2f                           \n"
    " nop                                         \n"

   "2:                                            \n"
    "lbu             $t0, -2(%[src_uv])           \n"
    "lbu             $t1, -1(%[src_uv])           \n"
    "addiu           %[src_uv], %[src_uv], -2     \n"
    "addiu           %[y], %[y], -1               \n"
    "sb              $t0, 0(%[dst_u])             \n"
    "sb              $t1, 0(%[dst_v])             \n"
    "addiu           %[dst_u], %[dst_u], 1        \n"
    "bgtz            %[y], 2b                     \n"
    " addiu          %[dst_v], %[dst_v], 1        \n"

   "3:                                            \n"
    ".set pop                                     \n"
      : [src_uv] "+r" (src_uv),
        [dst_u] "+r" (dst_u),
        [dst_v] "+r" (dst_v),
        [x] "=&r" (x),
        [y] "=&r" (y)
      : [width] "r" (width)
      : "t0", "t1", "t2", "t3", "t4",
      "t5", "t7", "t8", "t9"
  );
}

// Convert (4 Y and 2 VU) I422 and arrange RGB values into
// t5 = | 0 | B0 | 0 | b0 |
// t4 = | 0 | B1 | 0 | b1 |
// t9 = | 0 | G0 | 0 | g0 |
// t8 = | 0 | G1 | 0 | g1 |
// t2 = | 0 | R0 | 0 | r0 |
// t1 = | 0 | R1 | 0 | r1 |
#define YUVTORGB                                                               \
      "lw                $t0, 0(%[y_buf])       \n"                            \
      "lhu               $t1, 0(%[u_buf])       \n"                            \
      "lhu               $t2, 0(%[v_buf])       \n"                            \
      "preceu.ph.qbr     $t1, $t1               \n"                            \
      "preceu.ph.qbr     $t2, $t2               \n"                            \
      "preceu.ph.qbra    $t3, $t0               \n"                            \
      "preceu.ph.qbla    $t0, $t0               \n"                            \
      "subu.ph           $t1, $t1, $s5          \n"                            \
      "subu.ph           $t2, $t2, $s5          \n"                            \
      "subu.ph           $t3, $t3, $s4          \n"                            \
      "subu.ph           $t0, $t0, $s4          \n"                            \
      "mul.ph            $t3, $t3, $s0          \n"                            \
      "mul.ph            $t0, $t0, $s0          \n"                            \
      "shll.ph           $t4, $t1, 0x7          \n"                            \
      "subu.ph           $t4, $t4, $t1          \n"                            \
      "mul.ph            $t6, $t1, $s1          \n"                            \
      "mul.ph            $t1, $t2, $s2          \n"                            \
      "addq_s.ph         $t5, $t4, $t3          \n"                            \
      "addq_s.ph         $t4, $t4, $t0          \n"                            \
      "shra.ph           $t5, $t5, 6            \n"                            \
      "shra.ph           $t4, $t4, 6            \n"                            \
      "addiu             %[u_buf], 2            \n"                            \
      "addiu             %[v_buf], 2            \n"                            \
      "addu.ph           $t6, $t6, $t1          \n"                            \
      "mul.ph            $t1, $t2, $s3          \n"                            \
      "addu.ph           $t9, $t6, $t3          \n"                            \
      "addu.ph           $t8, $t6, $t0          \n"                            \
      "shra.ph           $t9, $t9, 6            \n"                            \
      "shra.ph           $t8, $t8, 6            \n"                            \
      "addu.ph           $t2, $t1, $t3          \n"                            \
      "addu.ph           $t1, $t1, $t0          \n"                            \
      "shra.ph           $t2, $t2, 6            \n"                            \
      "shra.ph           $t1, $t1, 6            \n"                            \
      "subu.ph           $t5, $t5, $s5          \n"                            \
      "subu.ph           $t4, $t4, $s5          \n"                            \
      "subu.ph           $t9, $t9, $s5          \n"                            \
      "subu.ph           $t8, $t8, $s5          \n"                            \
      "subu.ph           $t2, $t2, $s5          \n"                            \
      "subu.ph           $t1, $t1, $s5          \n"                            \
      "shll_s.ph         $t5, $t5, 8            \n"                            \
      "shll_s.ph         $t4, $t4, 8            \n"                            \
      "shll_s.ph         $t9, $t9, 8            \n"                            \
      "shll_s.ph         $t8, $t8, 8            \n"                            \
      "shll_s.ph         $t2, $t2, 8            \n"                            \
      "shll_s.ph         $t1, $t1, 8            \n"                            \
      "shra.ph           $t5, $t5, 8            \n"                            \
      "shra.ph           $t4, $t4, 8            \n"                            \
      "shra.ph           $t9, $t9, 8            \n"                            \
      "shra.ph           $t8, $t8, 8            \n"                            \
      "shra.ph           $t2, $t2, 8            \n"                            \
      "shra.ph           $t1, $t1, 8            \n"                            \
      "addu.ph           $t5, $t5, $s5          \n"                            \
      "addu.ph           $t4, $t4, $s5          \n"                            \
      "addu.ph           $t9, $t9, $s5          \n"                            \
      "addu.ph           $t8, $t8, $s5          \n"                            \
      "addu.ph           $t2, $t2, $s5          \n"                            \
      "addu.ph           $t1, $t1, $s5          \n"

// TODO(fbarchard): accept yuv conversion constants.
void I422ToARGBRow_DSPR2(const uint8* y_buf,
                         const uint8* u_buf,
                         const uint8* v_buf,
                         uint8* rgb_buf,
                         const struct YuvConstants* yuvconstants,
                         int width) {
  __asm__ __volatile__ (
    ".set push                                \n"
    ".set noreorder                           \n"
    "beqz              %[width], 2f           \n"
    " repl.ph          $s0, 74                \n"  // |YG|YG| = |74|74|
    "repl.ph           $s1, -25               \n"  // |UG|UG| = |-25|-25|
    "repl.ph           $s2, -52               \n"  // |VG|VG| = |-52|-52|
    "repl.ph           $s3, 102               \n"  // |VR|VR| = |102|102|
    "repl.ph           $s4, 16                \n"  // |0|16|0|16|
    "repl.ph           $s5, 128               \n"  // |128|128| // clipping
    "lui               $s6, 0xff00            \n"
    "ori               $s6, 0xff00            \n"  // |ff|00|ff|00|ff|

   "1:                                        \n"
      YUVTORGB
// Arranging into argb format
    "precr.qb.ph       $t4, $t8, $t4          \n"  // |G1|g1|B1|b1|
    "precr.qb.ph       $t5, $t9, $t5          \n"  // |G0|g0|B0|b0|
    "addiu             %[width], -4           \n"
    "precrq.qb.ph      $t8, $t4, $t5          \n"  // |G1|B1|G0|B0|
    "precr.qb.ph       $t9, $t4, $t5          \n"  // |g1|b1|g0|b0|
    "precr.qb.ph       $t2, $t1, $t2          \n"  // |R1|r1|R0|r0|

    "addiu             %[y_buf], 4            \n"
    "preceu.ph.qbla    $t1, $t2               \n"  // |0 |R1|0 |R0|
    "preceu.ph.qbra    $t2, $t2               \n"  // |0 |r1|0 |r0|
    "or                $t1, $t1, $s6          \n"  // |ff|R1|ff|R0|
    "or                $t2, $t2, $s6          \n"  // |ff|r1|ff|r0|
    "precrq.ph.w       $t0, $t2, $t9          \n"  // |ff|r1|g1|b1|
    "precrq.ph.w       $t3, $t1, $t8          \n"  // |ff|R1|G1|B1|
    "sll               $t9, $t9, 16           \n"
    "sll               $t8, $t8, 16           \n"
    "packrl.ph         $t2, $t2, $t9          \n"  // |ff|r0|g0|b0|
    "packrl.ph         $t1, $t1, $t8          \n"  // |ff|R0|G0|B0|
// Store results.
    "sw                $t2, 0(%[rgb_buf])     \n"
    "sw                $t0, 4(%[rgb_buf])     \n"
    "sw                $t1, 8(%[rgb_buf])     \n"
    "sw                $t3, 12(%[rgb_buf])    \n"
    "bnez              %[width], 1b           \n"
    " addiu            %[rgb_buf], 16         \n"
   "2:                                        \n"
    ".set pop                                 \n"
      :[y_buf] "+r" (y_buf),
       [u_buf] "+r" (u_buf),
       [v_buf] "+r" (v_buf),
       [width] "+r" (width),
       [rgb_buf] "+r" (rgb_buf)
      :
      : "t0", "t1",  "t2", "t3",  "t4", "t5",
      "t6", "t7", "t8", "t9",
      "s0", "s1", "s2", "s3",
      "s4", "s5", "s6"
  );
}

// Bilinear filter 8x2 -> 8x1
void InterpolateRow_DSPR2(uint8* dst_ptr, const uint8* src_ptr,
                          ptrdiff_t src_stride, int dst_width,
                          int source_y_fraction) {
    int y0_fraction = 256 - source_y_fraction;
    const uint8* src_ptr1 = src_ptr + src_stride;

  __asm__ __volatile__ (
     ".set push                                           \n"
     ".set noreorder                                      \n"

     "replv.ph          $t0, %[y0_fraction]               \n"
     "replv.ph          $t1, %[source_y_fraction]         \n"

   "1:                                                    \n"
     "lw                $t2, 0(%[src_ptr])                \n"
     "lw                $t3, 0(%[src_ptr1])               \n"
     "lw                $t4, 4(%[src_ptr])                \n"
     "lw                $t5, 4(%[src_ptr1])               \n"
     "muleu_s.ph.qbl    $t6, $t2, $t0                     \n"
     "muleu_s.ph.qbr    $t7, $t2, $t0                     \n"
     "muleu_s.ph.qbl    $t8, $t3, $t1                     \n"
     "muleu_s.ph.qbr    $t9, $t3, $t1                     \n"
     "muleu_s.ph.qbl    $t2, $t4, $t0                     \n"
     "muleu_s.ph.qbr    $t3, $t4, $t0                     \n"
     "muleu_s.ph.qbl    $t4, $t5, $t1                     \n"
     "muleu_s.ph.qbr    $t5, $t5, $t1                     \n"
     "addq.ph           $t6, $t6, $t8                     \n"
     "addq.ph           $t7, $t7, $t9                     \n"
     "addq.ph           $t2, $t2, $t4                     \n"
     "addq.ph           $t3, $t3, $t5                     \n"
     "shra.ph           $t6, $t6, 8                       \n"
     "shra.ph           $t7, $t7, 8                       \n"
     "shra.ph           $t2, $t2, 8                       \n"
     "shra.ph           $t3, $t3, 8                       \n"
     "precr.qb.ph       $t6, $t6, $t7                     \n"
     "precr.qb.ph       $t2, $t2, $t3                     \n"
     "addiu             %[src_ptr], %[src_ptr], 8         \n"
     "addiu             %[src_ptr1], %[src_ptr1], 8       \n"
     "addiu             %[dst_width], %[dst_width], -8    \n"
     "sw                $t6, 0(%[dst_ptr])                \n"
     "sw                $t2, 4(%[dst_ptr])                \n"
     "bgtz              %[dst_width], 1b                  \n"
     " addiu            %[dst_ptr], %[dst_ptr], 8         \n"

     ".set pop                                            \n"
  : [dst_ptr] "+r" (dst_ptr),
    [src_ptr1] "+r" (src_ptr1),
    [src_ptr] "+r" (src_ptr),
    [dst_width] "+r" (dst_width)
  : [source_y_fraction] "r" (source_y_fraction),
    [y0_fraction] "r" (y0_fraction),
    [src_stride] "r" (src_stride)
  : "t0", "t1", "t2", "t3", "t4", "t5",
    "t6", "t7", "t8", "t9"
  );
}
#endif  // __mips_dsp_rev >= 2

#endif  // defined(__mips__)

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
