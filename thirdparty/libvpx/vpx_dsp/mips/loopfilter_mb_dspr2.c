/*
 *  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <stdlib.h>

#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/mips/common_dspr2.h"
#include "vpx_dsp/mips/loopfilter_filters_dspr2.h"
#include "vpx_dsp/mips/loopfilter_macros_dspr2.h"
#include "vpx_dsp/mips/loopfilter_masks_dspr2.h"
#include "vpx_mem/vpx_mem.h"

#if HAVE_DSPR2
void vpx_lpf_horizontal_8_dspr2(unsigned char *s, int pitch,
                                const uint8_t *blimit, const uint8_t *limit,
                                const uint8_t *thresh) {
  uint32_t mask;
  uint32_t hev, flat;
  uint8_t i;
  uint8_t *sp3, *sp2, *sp1, *sp0, *sq0, *sq1, *sq2, *sq3;
  uint32_t thresh_vec, flimit_vec, limit_vec;
  uint32_t uflimit, ulimit, uthresh;
  uint32_t p1_f0, p0_f0, q0_f0, q1_f0;
  uint32_t p3, p2, p1, p0, q0, q1, q2, q3;
  uint32_t p0_l, p1_l, p2_l, p3_l, q0_l, q1_l, q2_l, q3_l;
  uint32_t p0_r, p1_r, p2_r, p3_r, q0_r, q1_r, q2_r, q3_r;

  uflimit = *blimit;
  ulimit = *limit;
  uthresh = *thresh;

  /* create quad-byte */
  __asm__ __volatile__(
      "replv.qb       %[thresh_vec],    %[uthresh]    \n\t"
      "replv.qb       %[flimit_vec],    %[uflimit]    \n\t"
      "replv.qb       %[limit_vec],     %[ulimit]     \n\t"

      : [thresh_vec] "=&r"(thresh_vec), [flimit_vec] "=&r"(flimit_vec),
        [limit_vec] "=r"(limit_vec)
      : [uthresh] "r"(uthresh), [uflimit] "r"(uflimit), [ulimit] "r"(ulimit));

  /* prefetch data for store */
  prefetch_store(s);

  for (i = 0; i < 2; i++) {
    sp3 = s - (pitch << 2);
    sp2 = sp3 + pitch;
    sp1 = sp2 + pitch;
    sp0 = sp1 + pitch;
    sq0 = s;
    sq1 = s + pitch;
    sq2 = sq1 + pitch;
    sq3 = sq2 + pitch;

    __asm__ __volatile__(
        "lw     %[p3],      (%[sp3])    \n\t"
        "lw     %[p2],      (%[sp2])    \n\t"
        "lw     %[p1],      (%[sp1])    \n\t"
        "lw     %[p0],      (%[sp0])    \n\t"
        "lw     %[q0],      (%[sq0])    \n\t"
        "lw     %[q1],      (%[sq1])    \n\t"
        "lw     %[q2],      (%[sq2])    \n\t"
        "lw     %[q3],      (%[sq3])    \n\t"

        : [p3] "=&r"(p3), [p2] "=&r"(p2), [p1] "=&r"(p1), [p0] "=&r"(p0),
          [q3] "=&r"(q3), [q2] "=&r"(q2), [q1] "=&r"(q1), [q0] "=&r"(q0)
        : [sp3] "r"(sp3), [sp2] "r"(sp2), [sp1] "r"(sp1), [sp0] "r"(sp0),
          [sq3] "r"(sq3), [sq2] "r"(sq2), [sq1] "r"(sq1), [sq0] "r"(sq0));

    filter_hev_mask_flatmask4_dspr2(limit_vec, flimit_vec, thresh_vec, p1, p0,
                                    p3, p2, q0, q1, q2, q3, &hev, &mask, &flat);

    if ((flat == 0) && (mask != 0)) {
      filter1_dspr2(mask, hev, p1, p0, q0, q1, &p1_f0, &p0_f0, &q0_f0, &q1_f0);

      __asm__ __volatile__(
          "sw       %[p1_f0],   (%[sp1])    \n\t"
          "sw       %[p0_f0],   (%[sp0])    \n\t"
          "sw       %[q0_f0],   (%[sq0])    \n\t"
          "sw       %[q1_f0],   (%[sq1])    \n\t"

          :
          : [p1_f0] "r"(p1_f0), [p0_f0] "r"(p0_f0), [q0_f0] "r"(q0_f0),
            [q1_f0] "r"(q1_f0), [sp1] "r"(sp1), [sp0] "r"(sp0), [sq0] "r"(sq0),
            [sq1] "r"(sq1));
    } else if ((mask & flat) == 0xFFFFFFFF) {
      /* left 2 element operation */
      PACK_LEFT_0TO3()
      mbfilter_dspr2(&p3_l, &p2_l, &p1_l, &p0_l, &q0_l, &q1_l, &q2_l, &q3_l);

      /* right 2 element operation */
      PACK_RIGHT_0TO3()
      mbfilter_dspr2(&p3_r, &p2_r, &p1_r, &p0_r, &q0_r, &q1_r, &q2_r, &q3_r);

      COMBINE_LEFT_RIGHT_0TO2()

      __asm__ __volatile__(
          "sw       %[p2],      (%[sp2])    \n\t"
          "sw       %[p1],      (%[sp1])    \n\t"
          "sw       %[p0],      (%[sp0])    \n\t"
          "sw       %[q0],      (%[sq0])    \n\t"
          "sw       %[q1],      (%[sq1])    \n\t"
          "sw       %[q2],      (%[sq2])    \n\t"

          :
          : [p2] "r"(p2), [p1] "r"(p1), [p0] "r"(p0), [q0] "r"(q0),
            [q1] "r"(q1), [q2] "r"(q2), [sp2] "r"(sp2), [sp1] "r"(sp1),
            [sp0] "r"(sp0), [sq0] "r"(sq0), [sq1] "r"(sq1), [sq2] "r"(sq2));
    } else if ((flat != 0) && (mask != 0)) {
      /* filtering */
      filter1_dspr2(mask, hev, p1, p0, q0, q1, &p1_f0, &p0_f0, &q0_f0, &q1_f0);

      /* left 2 element operation */
      PACK_LEFT_0TO3()
      mbfilter_dspr2(&p3_l, &p2_l, &p1_l, &p0_l, &q0_l, &q1_l, &q2_l, &q3_l);

      /* right 2 element operation */
      PACK_RIGHT_0TO3()
      mbfilter_dspr2(&p3_r, &p2_r, &p1_r, &p0_r, &q0_r, &q1_r, &q2_r, &q3_r);

      if (mask & flat & 0x000000FF) {
        __asm__ __volatile__(
            "sb     %[p2_r],    (%[sp2])    \n\t"
            "sb     %[p1_r],    (%[sp1])    \n\t"
            "sb     %[p0_r],    (%[sp0])    \n\t"
            "sb     %[q0_r],    (%[sq0])    \n\t"
            "sb     %[q1_r],    (%[sq1])    \n\t"
            "sb     %[q2_r],    (%[sq2])    \n\t"

            :
            : [p2_r] "r"(p2_r), [p1_r] "r"(p1_r), [p0_r] "r"(p0_r),
              [q0_r] "r"(q0_r), [q1_r] "r"(q1_r), [q2_r] "r"(q2_r),
              [sp2] "r"(sp2), [sp1] "r"(sp1), [sp0] "r"(sp0), [sq0] "r"(sq0),
              [sq1] "r"(sq1), [sq2] "r"(sq2));
      } else if (mask & 0x000000FF) {
        __asm__ __volatile__(
            "sb         %[p1_f0],  (%[sp1])    \n\t"
            "sb         %[p0_f0],  (%[sp0])    \n\t"
            "sb         %[q0_f0],  (%[sq0])    \n\t"
            "sb         %[q1_f0],  (%[sq1])    \n\t"

            :
            : [p1_f0] "r"(p1_f0), [p0_f0] "r"(p0_f0), [q0_f0] "r"(q0_f0),
              [q1_f0] "r"(q1_f0), [sp1] "r"(sp1), [sp0] "r"(sp0),
              [sq0] "r"(sq0), [sq1] "r"(sq1));
      }

      __asm__ __volatile__(
          "srl      %[p2_r],    %[p2_r],    16      \n\t"
          "srl      %[p1_r],    %[p1_r],    16      \n\t"
          "srl      %[p0_r],    %[p0_r],    16      \n\t"
          "srl      %[q0_r],    %[q0_r],    16      \n\t"
          "srl      %[q1_r],    %[q1_r],    16      \n\t"
          "srl      %[q2_r],    %[q2_r],    16      \n\t"
          "srl      %[p1_f0],   %[p1_f0],   8       \n\t"
          "srl      %[p0_f0],   %[p0_f0],   8       \n\t"
          "srl      %[q0_f0],   %[q0_f0],   8       \n\t"
          "srl      %[q1_f0],   %[q1_f0],   8       \n\t"

          : [p2_r] "+r"(p2_r), [p1_r] "+r"(p1_r), [p0_r] "+r"(p0_r),
            [q0_r] "+r"(q0_r), [q1_r] "+r"(q1_r), [q2_r] "+r"(q2_r),
            [p1_f0] "+r"(p1_f0), [p0_f0] "+r"(p0_f0), [q0_f0] "+r"(q0_f0),
            [q1_f0] "+r"(q1_f0)
          :);

      if (mask & flat & 0x0000FF00) {
        __asm__ __volatile__(
            "sb     %[p2_r],    +1(%[sp2])    \n\t"
            "sb     %[p1_r],    +1(%[sp1])    \n\t"
            "sb     %[p0_r],    +1(%[sp0])    \n\t"
            "sb     %[q0_r],    +1(%[sq0])    \n\t"
            "sb     %[q1_r],    +1(%[sq1])    \n\t"
            "sb     %[q2_r],    +1(%[sq2])    \n\t"

            :
            : [p2_r] "r"(p2_r), [p1_r] "r"(p1_r), [p0_r] "r"(p0_r),
              [q0_r] "r"(q0_r), [q1_r] "r"(q1_r), [q2_r] "r"(q2_r),
              [sp2] "r"(sp2), [sp1] "r"(sp1), [sp0] "r"(sp0), [sq0] "r"(sq0),
              [sq1] "r"(sq1), [sq2] "r"(sq2));
      } else if (mask & 0x0000FF00) {
        __asm__ __volatile__(
            "sb     %[p1_f0],   +1(%[sp1])    \n\t"
            "sb     %[p0_f0],   +1(%[sp0])    \n\t"
            "sb     %[q0_f0],   +1(%[sq0])    \n\t"
            "sb     %[q1_f0],   +1(%[sq1])    \n\t"

            :
            : [p1_f0] "r"(p1_f0), [p0_f0] "r"(p0_f0), [q0_f0] "r"(q0_f0),
              [q1_f0] "r"(q1_f0), [sp1] "r"(sp1), [sp0] "r"(sp0),
              [sq0] "r"(sq0), [sq1] "r"(sq1));
      }

      __asm__ __volatile__(
          "srl      %[p1_f0],   %[p1_f0],   8     \n\t"
          "srl      %[p0_f0],   %[p0_f0],   8     \n\t"
          "srl      %[q0_f0],   %[q0_f0],   8     \n\t"
          "srl      %[q1_f0],   %[q1_f0],   8     \n\t"

          : [p2] "+r"(p2), [p1] "+r"(p1), [p0] "+r"(p0), [q0] "+r"(q0),
            [q1] "+r"(q1), [q2] "+r"(q2), [p1_f0] "+r"(p1_f0),
            [p0_f0] "+r"(p0_f0), [q0_f0] "+r"(q0_f0), [q1_f0] "+r"(q1_f0)
          :);

      if (mask & flat & 0x00FF0000) {
        __asm__ __volatile__(
            "sb     %[p2_l],    +2(%[sp2])    \n\t"
            "sb     %[p1_l],    +2(%[sp1])    \n\t"
            "sb     %[p0_l],    +2(%[sp0])    \n\t"
            "sb     %[q0_l],    +2(%[sq0])    \n\t"
            "sb     %[q1_l],    +2(%[sq1])    \n\t"
            "sb     %[q2_l],    +2(%[sq2])    \n\t"

            :
            : [p2_l] "r"(p2_l), [p1_l] "r"(p1_l), [p0_l] "r"(p0_l),
              [q0_l] "r"(q0_l), [q1_l] "r"(q1_l), [q2_l] "r"(q2_l),
              [sp2] "r"(sp2), [sp1] "r"(sp1), [sp0] "r"(sp0), [sq0] "r"(sq0),
              [sq1] "r"(sq1), [sq2] "r"(sq2));
      } else if (mask & 0x00FF0000) {
        __asm__ __volatile__(
            "sb     %[p1_f0],   +2(%[sp1])    \n\t"
            "sb     %[p0_f0],   +2(%[sp0])    \n\t"
            "sb     %[q0_f0],   +2(%[sq0])    \n\t"
            "sb     %[q1_f0],   +2(%[sq1])    \n\t"

            :
            : [p1_f0] "r"(p1_f0), [p0_f0] "r"(p0_f0), [q0_f0] "r"(q0_f0),
              [q1_f0] "r"(q1_f0), [sp1] "r"(sp1), [sp0] "r"(sp0),
              [sq0] "r"(sq0), [sq1] "r"(sq1));
      }

      __asm__ __volatile__(
          "srl      %[p2_l],    %[p2_l],    16      \n\t"
          "srl      %[p1_l],    %[p1_l],    16      \n\t"
          "srl      %[p0_l],    %[p0_l],    16      \n\t"
          "srl      %[q0_l],    %[q0_l],    16      \n\t"
          "srl      %[q1_l],    %[q1_l],    16      \n\t"
          "srl      %[q2_l],    %[q2_l],    16      \n\t"
          "srl      %[p1_f0],   %[p1_f0],   8       \n\t"
          "srl      %[p0_f0],   %[p0_f0],   8       \n\t"
          "srl      %[q0_f0],   %[q0_f0],   8       \n\t"
          "srl      %[q1_f0],   %[q1_f0],   8       \n\t"

          : [p2_l] "+r"(p2_l), [p1_l] "+r"(p1_l), [p0_l] "+r"(p0_l),
            [q0_l] "+r"(q0_l), [q1_l] "+r"(q1_l), [q2_l] "+r"(q2_l),
            [p1_f0] "+r"(p1_f0), [p0_f0] "+r"(p0_f0), [q0_f0] "+r"(q0_f0),
            [q1_f0] "+r"(q1_f0)
          :);

      if (mask & flat & 0xFF000000) {
        __asm__ __volatile__(
            "sb     %[p2_l],    +3(%[sp2])    \n\t"
            "sb     %[p1_l],    +3(%[sp1])    \n\t"
            "sb     %[p0_l],    +3(%[sp0])    \n\t"
            "sb     %[q0_l],    +3(%[sq0])    \n\t"
            "sb     %[q1_l],    +3(%[sq1])    \n\t"
            "sb     %[q2_l],    +3(%[sq2])    \n\t"

            :
            : [p2_l] "r"(p2_l), [p1_l] "r"(p1_l), [p0_l] "r"(p0_l),
              [q0_l] "r"(q0_l), [q1_l] "r"(q1_l), [q2_l] "r"(q2_l),
              [sp2] "r"(sp2), [sp1] "r"(sp1), [sp0] "r"(sp0), [sq0] "r"(sq0),
              [sq1] "r"(sq1), [sq2] "r"(sq2));
      } else if (mask & 0xFF000000) {
        __asm__ __volatile__(
            "sb     %[p1_f0],   +3(%[sp1])    \n\t"
            "sb     %[p0_f0],   +3(%[sp0])    \n\t"
            "sb     %[q0_f0],   +3(%[sq0])    \n\t"
            "sb     %[q1_f0],   +3(%[sq1])    \n\t"

            :
            : [p1_f0] "r"(p1_f0), [p0_f0] "r"(p0_f0), [q0_f0] "r"(q0_f0),
              [q1_f0] "r"(q1_f0), [sp1] "r"(sp1), [sp0] "r"(sp0),
              [sq0] "r"(sq0), [sq1] "r"(sq1));
      }
    }

    s = s + 4;
  }
}

void vpx_lpf_vertical_8_dspr2(unsigned char *s, int pitch,
                              const uint8_t *blimit, const uint8_t *limit,
                              const uint8_t *thresh) {
  uint8_t i;
  uint32_t mask, hev, flat;
  uint8_t *s1, *s2, *s3, *s4;
  uint32_t prim1, prim2, sec3, sec4, prim3, prim4;
  uint32_t thresh_vec, flimit_vec, limit_vec;
  uint32_t uflimit, ulimit, uthresh;
  uint32_t p3, p2, p1, p0, q3, q2, q1, q0;
  uint32_t p1_f0, p0_f0, q0_f0, q1_f0;
  uint32_t p0_l, p1_l, p2_l, p3_l, q0_l, q1_l, q2_l, q3_l;
  uint32_t p0_r, p1_r, p2_r, p3_r, q0_r, q1_r, q2_r, q3_r;

  uflimit = *blimit;
  ulimit = *limit;
  uthresh = *thresh;

  /* create quad-byte */
  __asm__ __volatile__(
      "replv.qb     %[thresh_vec],  %[uthresh]    \n\t"
      "replv.qb     %[flimit_vec],  %[uflimit]    \n\t"
      "replv.qb     %[limit_vec],   %[ulimit]     \n\t"

      : [thresh_vec] "=&r"(thresh_vec), [flimit_vec] "=&r"(flimit_vec),
        [limit_vec] "=r"(limit_vec)
      : [uthresh] "r"(uthresh), [uflimit] "r"(uflimit), [ulimit] "r"(ulimit));

  prefetch_store(s + pitch);

  for (i = 0; i < 2; i++) {
    s1 = s;
    s2 = s + pitch;
    s3 = s2 + pitch;
    s4 = s3 + pitch;
    s = s4 + pitch;

    __asm__ __volatile__(
        "lw     %[p0],  -4(%[s1])    \n\t"
        "lw     %[p1],  -4(%[s2])    \n\t"
        "lw     %[p2],  -4(%[s3])    \n\t"
        "lw     %[p3],  -4(%[s4])    \n\t"
        "lw     %[q3],    (%[s1])    \n\t"
        "lw     %[q2],    (%[s2])    \n\t"
        "lw     %[q1],    (%[s3])    \n\t"
        "lw     %[q0],    (%[s4])    \n\t"

        : [p3] "=&r"(p3), [p2] "=&r"(p2), [p1] "=&r"(p1), [p0] "=&r"(p0),
          [q0] "=&r"(q0), [q1] "=&r"(q1), [q2] "=&r"(q2), [q3] "=&r"(q3)
        : [s1] "r"(s1), [s2] "r"(s2), [s3] "r"(s3), [s4] "r"(s4));

    /* transpose p3, p2, p1, p0
       original (when loaded from memory)
       register       -4    -3   -2     -1
         p0         p0_0  p0_1  p0_2  p0_3
         p1         p1_0  p1_1  p1_2  p1_3
         p2         p2_0  p2_1  p2_2  p2_3
         p3         p3_0  p3_1  p3_2  p3_3

       after transpose
       register
         p0         p3_3  p2_3  p1_3  p0_3
         p1         p3_2  p2_2  p1_2  p0_2
         p2         p3_1  p2_1  p1_1  p0_1
         p3         p3_0  p2_0  p1_0  p0_0
    */
    __asm__ __volatile__(
        "precrq.qb.ph   %[prim1],   %[p0],      %[p1]       \n\t"
        "precr.qb.ph    %[prim2],   %[p0],      %[p1]       \n\t"
        "precrq.qb.ph   %[prim3],   %[p2],      %[p3]       \n\t"
        "precr.qb.ph    %[prim4],   %[p2],      %[p3]       \n\t"

        "precrq.qb.ph   %[p1],      %[prim1],   %[prim2]    \n\t"
        "precr.qb.ph    %[p3],      %[prim1],   %[prim2]    \n\t"
        "precrq.qb.ph   %[sec3],    %[prim3],   %[prim4]    \n\t"
        "precr.qb.ph    %[sec4],    %[prim3],   %[prim4]    \n\t"

        "precrq.ph.w    %[p0],      %[p1],      %[sec3]     \n\t"
        "precrq.ph.w    %[p2],      %[p3],      %[sec4]     \n\t"
        "append         %[p1],      %[sec3],    16          \n\t"
        "append         %[p3],      %[sec4],    16          \n\t"

        : [prim1] "=&r"(prim1), [prim2] "=&r"(prim2), [prim3] "=&r"(prim3),
          [prim4] "=&r"(prim4), [p0] "+r"(p0), [p1] "+r"(p1), [p2] "+r"(p2),
          [p3] "+r"(p3), [sec3] "=&r"(sec3), [sec4] "=&r"(sec4)
        :);

    /* transpose q0, q1, q2, q3
       original (when loaded from memory)
       register       +1    +2    +3    +4
         q3         q3_0  q3_1  q3_2  q3_3
         q2         q2_0  q2_1  q2_2  q2_3
         q1         q1_0  q1_1  q1_2  q1_3
         q0         q0_0  q0_1  q0_2  q0_3

       after transpose
       register
         q3         q0_3  q1_3  q2_3  q3_3
         q2         q0_2  q1_2  q2_2  q3_2
         q1         q0_1  q1_1  q2_1  q3_1
         q0         q0_0  q1_0  q2_0  q3_0
    */
    __asm__ __volatile__(
        "precrq.qb.ph   %[prim1],   %[q3],      %[q2]       \n\t"
        "precr.qb.ph    %[prim2],   %[q3],      %[q2]       \n\t"
        "precrq.qb.ph   %[prim3],   %[q1],      %[q0]       \n\t"
        "precr.qb.ph    %[prim4],   %[q1],      %[q0]       \n\t"

        "precrq.qb.ph   %[q2],      %[prim1],   %[prim2]    \n\t"
        "precr.qb.ph    %[q0],      %[prim1],   %[prim2]    \n\t"
        "precrq.qb.ph   %[sec3],    %[prim3],   %[prim4]    \n\t"
        "precr.qb.ph    %[sec4],    %[prim3],   %[prim4]    \n\t"

        "precrq.ph.w    %[q3],      %[q2],      %[sec3]     \n\t"
        "precrq.ph.w    %[q1],      %[q0],      %[sec4]     \n\t"
        "append         %[q2],      %[sec3],    16          \n\t"
        "append         %[q0],      %[sec4],    16          \n\t"

        : [prim1] "=&r"(prim1), [prim2] "=&r"(prim2), [prim3] "=&r"(prim3),
          [prim4] "=&r"(prim4), [q3] "+r"(q3), [q2] "+r"(q2), [q1] "+r"(q1),
          [q0] "+r"(q0), [sec3] "=&r"(sec3), [sec4] "=&r"(sec4)
        :);

    filter_hev_mask_flatmask4_dspr2(limit_vec, flimit_vec, thresh_vec, p1, p0,
                                    p3, p2, q0, q1, q2, q3, &hev, &mask, &flat);

    if ((flat == 0) && (mask != 0)) {
      filter1_dspr2(mask, hev, p1, p0, q0, q1, &p1_f0, &p0_f0, &q0_f0, &q1_f0);
      STORE_F0()
    } else if ((mask & flat) == 0xFFFFFFFF) {
      /* left 2 element operation */
      PACK_LEFT_0TO3()
      mbfilter_dspr2(&p3_l, &p2_l, &p1_l, &p0_l, &q0_l, &q1_l, &q2_l, &q3_l);

      /* right 2 element operation */
      PACK_RIGHT_0TO3()
      mbfilter_dspr2(&p3_r, &p2_r, &p1_r, &p0_r, &q0_r, &q1_r, &q2_r, &q3_r);

      STORE_F1()
    } else if ((flat != 0) && (mask != 0)) {
      filter1_dspr2(mask, hev, p1, p0, q0, q1, &p1_f0, &p0_f0, &q0_f0, &q1_f0);

      /* left 2 element operation */
      PACK_LEFT_0TO3()
      mbfilter_dspr2(&p3_l, &p2_l, &p1_l, &p0_l, &q0_l, &q1_l, &q2_l, &q3_l);

      /* right 2 element operation */
      PACK_RIGHT_0TO3()
      mbfilter_dspr2(&p3_r, &p2_r, &p1_r, &p0_r, &q0_r, &q1_r, &q2_r, &q3_r);

      if (mask & flat & 0x000000FF) {
        __asm__ __volatile__(
            "sb         %[p2_r],  -3(%[s4])    \n\t"
            "sb         %[p1_r],  -2(%[s4])    \n\t"
            "sb         %[p0_r],  -1(%[s4])    \n\t"
            "sb         %[q0_r],    (%[s4])    \n\t"
            "sb         %[q1_r],  +1(%[s4])    \n\t"
            "sb         %[q2_r],  +2(%[s4])    \n\t"

            :
            : [p2_r] "r"(p2_r), [p1_r] "r"(p1_r), [p0_r] "r"(p0_r),
              [q0_r] "r"(q0_r), [q1_r] "r"(q1_r), [q2_r] "r"(q2_r),
              [s4] "r"(s4));
      } else if (mask & 0x000000FF) {
        __asm__ __volatile__(
            "sb         %[p1_f0],  -2(%[s4])    \n\t"
            "sb         %[p0_f0],  -1(%[s4])    \n\t"
            "sb         %[q0_f0],    (%[s4])    \n\t"
            "sb         %[q1_f0],  +1(%[s4])    \n\t"

            :
            : [p1_f0] "r"(p1_f0), [p0_f0] "r"(p0_f0), [q0_f0] "r"(q0_f0),
              [q1_f0] "r"(q1_f0), [s4] "r"(s4));
      }

      __asm__ __volatile__(
          "srl      %[p2_r],    %[p2_r],    16      \n\t"
          "srl      %[p1_r],    %[p1_r],    16      \n\t"
          "srl      %[p0_r],    %[p0_r],    16      \n\t"
          "srl      %[q0_r],    %[q0_r],    16      \n\t"
          "srl      %[q1_r],    %[q1_r],    16      \n\t"
          "srl      %[q2_r],    %[q2_r],    16      \n\t"
          "srl      %[p1_f0],   %[p1_f0],   8       \n\t"
          "srl      %[p0_f0],   %[p0_f0],   8       \n\t"
          "srl      %[q0_f0],   %[q0_f0],   8       \n\t"
          "srl      %[q1_f0],   %[q1_f0],   8       \n\t"

          : [p2_r] "+r"(p2_r), [p1_r] "+r"(p1_r), [p0_r] "+r"(p0_r),
            [q0_r] "+r"(q0_r), [q1_r] "+r"(q1_r), [q2_r] "+r"(q2_r),
            [p1_f0] "+r"(p1_f0), [p0_f0] "+r"(p0_f0), [q0_f0] "+r"(q0_f0),
            [q1_f0] "+r"(q1_f0)
          :);

      if (mask & flat & 0x0000FF00) {
        __asm__ __volatile__(
            "sb         %[p2_r],  -3(%[s3])    \n\t"
            "sb         %[p1_r],  -2(%[s3])    \n\t"
            "sb         %[p0_r],  -1(%[s3])    \n\t"
            "sb         %[q0_r],    (%[s3])    \n\t"
            "sb         %[q1_r],  +1(%[s3])    \n\t"
            "sb         %[q2_r],  +2(%[s3])    \n\t"

            :
            : [p2_r] "r"(p2_r), [p1_r] "r"(p1_r), [p0_r] "r"(p0_r),
              [q0_r] "r"(q0_r), [q1_r] "r"(q1_r), [q2_r] "r"(q2_r),
              [s3] "r"(s3));
      } else if (mask & 0x0000FF00) {
        __asm__ __volatile__(
            "sb         %[p1_f0],  -2(%[s3])    \n\t"
            "sb         %[p0_f0],  -1(%[s3])    \n\t"
            "sb         %[q0_f0],    (%[s3])    \n\t"
            "sb         %[q1_f0],  +1(%[s3])    \n\t"

            :
            : [p1_f0] "r"(p1_f0), [p0_f0] "r"(p0_f0), [q0_f0] "r"(q0_f0),
              [q1_f0] "r"(q1_f0), [s3] "r"(s3));
      }

      __asm__ __volatile__(
          "srl      %[p1_f0],   %[p1_f0],   8     \n\t"
          "srl      %[p0_f0],   %[p0_f0],   8     \n\t"
          "srl      %[q0_f0],   %[q0_f0],   8     \n\t"
          "srl      %[q1_f0],   %[q1_f0],   8     \n\t"

          : [p2] "+r"(p2), [p1] "+r"(p1), [p0] "+r"(p0), [q0] "+r"(q0),
            [q1] "+r"(q1), [q2] "+r"(q2), [p1_f0] "+r"(p1_f0),
            [p0_f0] "+r"(p0_f0), [q0_f0] "+r"(q0_f0), [q1_f0] "+r"(q1_f0)
          :);

      if (mask & flat & 0x00FF0000) {
        __asm__ __volatile__(
            "sb         %[p2_l],  -3(%[s2])    \n\t"
            "sb         %[p1_l],  -2(%[s2])    \n\t"
            "sb         %[p0_l],  -1(%[s2])    \n\t"
            "sb         %[q0_l],    (%[s2])    \n\t"
            "sb         %[q1_l],  +1(%[s2])    \n\t"
            "sb         %[q2_l],  +2(%[s2])    \n\t"

            :
            : [p2_l] "r"(p2_l), [p1_l] "r"(p1_l), [p0_l] "r"(p0_l),
              [q0_l] "r"(q0_l), [q1_l] "r"(q1_l), [q2_l] "r"(q2_l),
              [s2] "r"(s2));
      } else if (mask & 0x00FF0000) {
        __asm__ __volatile__(
            "sb         %[p1_f0],  -2(%[s2])    \n\t"
            "sb         %[p0_f0],  -1(%[s2])    \n\t"
            "sb         %[q0_f0],    (%[s2])    \n\t"
            "sb         %[q1_f0],  +1(%[s2])    \n\t"

            :
            : [p1_f0] "r"(p1_f0), [p0_f0] "r"(p0_f0), [q0_f0] "r"(q0_f0),
              [q1_f0] "r"(q1_f0), [s2] "r"(s2));
      }

      __asm__ __volatile__(
          "srl      %[p2_l],    %[p2_l],    16      \n\t"
          "srl      %[p1_l],    %[p1_l],    16      \n\t"
          "srl      %[p0_l],    %[p0_l],    16      \n\t"
          "srl      %[q0_l],    %[q0_l],    16      \n\t"
          "srl      %[q1_l],    %[q1_l],    16      \n\t"
          "srl      %[q2_l],    %[q2_l],    16      \n\t"
          "srl      %[p1_f0],   %[p1_f0],   8       \n\t"
          "srl      %[p0_f0],   %[p0_f0],   8       \n\t"
          "srl      %[q0_f0],   %[q0_f0],   8       \n\t"
          "srl      %[q1_f0],   %[q1_f0],   8       \n\t"

          : [p2_l] "+r"(p2_l), [p1_l] "+r"(p1_l), [p0_l] "+r"(p0_l),
            [q0_l] "+r"(q0_l), [q1_l] "+r"(q1_l), [q2_l] "+r"(q2_l),
            [p1_f0] "+r"(p1_f0), [p0_f0] "+r"(p0_f0), [q0_f0] "+r"(q0_f0),
            [q1_f0] "+r"(q1_f0)
          :);

      if (mask & flat & 0xFF000000) {
        __asm__ __volatile__(
            "sb         %[p2_l],  -3(%[s1])    \n\t"
            "sb         %[p1_l],  -2(%[s1])    \n\t"
            "sb         %[p0_l],  -1(%[s1])    \n\t"
            "sb         %[q0_l],    (%[s1])    \n\t"
            "sb         %[q1_l],  +1(%[s1])    \n\t"
            "sb         %[q2_l],  +2(%[s1])    \n\t"

            :
            : [p2_l] "r"(p2_l), [p1_l] "r"(p1_l), [p0_l] "r"(p0_l),
              [q0_l] "r"(q0_l), [q1_l] "r"(q1_l), [q2_l] "r"(q2_l),
              [s1] "r"(s1));
      } else if (mask & 0xFF000000) {
        __asm__ __volatile__(
            "sb         %[p1_f0],  -2(%[s1])    \n\t"
            "sb         %[p0_f0],  -1(%[s1])    \n\t"
            "sb         %[q0_f0],    (%[s1])    \n\t"
            "sb         %[q1_f0],  +1(%[s1])    \n\t"

            :
            : [p1_f0] "r"(p1_f0), [p0_f0] "r"(p0_f0), [q0_f0] "r"(q0_f0),
              [q1_f0] "r"(q1_f0), [s1] "r"(s1));
      }
    }
  }
}
#endif  // #if HAVE_DSPR2
