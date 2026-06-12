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
static void mb_lpf_horizontal_edge(unsigned char *s, int pitch,
                                   const uint8_t *blimit, const uint8_t *limit,
                                   const uint8_t *thresh, int count) {
  uint32_t mask;
  uint32_t hev, flat, flat2;
  uint8_t i;
  uint8_t *sp7, *sp6, *sp5, *sp4, *sp3, *sp2, *sp1, *sp0;
  uint8_t *sq0, *sq1, *sq2, *sq3, *sq4, *sq5, *sq6, *sq7;
  uint32_t thresh_vec, flimit_vec, limit_vec;
  uint32_t uflimit, ulimit, uthresh;
  uint32_t p7, p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6, q7;
  uint32_t p1_f0, p0_f0, q0_f0, q1_f0;
  uint32_t p7_l, p6_l, p5_l, p4_l, p3_l, p2_l, p1_l, p0_l;
  uint32_t q0_l, q1_l, q2_l, q3_l, q4_l, q5_l, q6_l, q7_l;
  uint32_t p7_r, p6_r, p5_r, p4_r, p3_r, p2_r, p1_r, p0_r;
  uint32_t q0_r, q1_r, q2_r, q3_r, q4_r, q5_r, q6_r, q7_r;
  uint32_t p2_l_f1, p1_l_f1, p0_l_f1, p2_r_f1, p1_r_f1, p0_r_f1;
  uint32_t q0_l_f1, q1_l_f1, q2_l_f1, q0_r_f1, q1_r_f1, q2_r_f1;

  uflimit = *blimit;
  ulimit = *limit;
  uthresh = *thresh;

  /* create quad-byte */
  __asm__ __volatile__(
      "replv.qb       %[thresh_vec],    %[uthresh]      \n\t"
      "replv.qb       %[flimit_vec],    %[uflimit]      \n\t"
      "replv.qb       %[limit_vec],     %[ulimit]       \n\t"

      : [thresh_vec] "=&r"(thresh_vec), [flimit_vec] "=&r"(flimit_vec),
        [limit_vec] "=r"(limit_vec)
      : [uthresh] "r"(uthresh), [uflimit] "r"(uflimit), [ulimit] "r"(ulimit));

  /* prefetch data for store */
  prefetch_store(s);

  for (i = 0; i < (2 * count); i++) {
    sp7 = s - (pitch << 3);
    sp6 = sp7 + pitch;
    sp5 = sp6 + pitch;
    sp4 = sp5 + pitch;
    sp3 = sp4 + pitch;
    sp2 = sp3 + pitch;
    sp1 = sp2 + pitch;
    sp0 = sp1 + pitch;
    sq0 = s;
    sq1 = s + pitch;
    sq2 = sq1 + pitch;
    sq3 = sq2 + pitch;
    sq4 = sq3 + pitch;
    sq5 = sq4 + pitch;
    sq6 = sq5 + pitch;
    sq7 = sq6 + pitch;

    __asm__ __volatile__(
        "lw     %[p7],      (%[sp7])            \n\t"
        "lw     %[p6],      (%[sp6])            \n\t"
        "lw     %[p5],      (%[sp5])            \n\t"
        "lw     %[p4],      (%[sp4])            \n\t"
        "lw     %[p3],      (%[sp3])            \n\t"
        "lw     %[p2],      (%[sp2])            \n\t"
        "lw     %[p1],      (%[sp1])            \n\t"
        "lw     %[p0],      (%[sp0])            \n\t"

        : [p3] "=&r"(p3), [p2] "=&r"(p2), [p1] "=&r"(p1), [p0] "=&r"(p0),
          [p7] "=&r"(p7), [p6] "=&r"(p6), [p5] "=&r"(p5), [p4] "=&r"(p4)
        : [sp3] "r"(sp3), [sp2] "r"(sp2), [sp1] "r"(sp1), [sp0] "r"(sp0),
          [sp4] "r"(sp4), [sp5] "r"(sp5), [sp6] "r"(sp6), [sp7] "r"(sp7));

    __asm__ __volatile__(
        "lw     %[q0],      (%[sq0])            \n\t"
        "lw     %[q1],      (%[sq1])            \n\t"
        "lw     %[q2],      (%[sq2])            \n\t"
        "lw     %[q3],      (%[sq3])            \n\t"
        "lw     %[q4],      (%[sq4])            \n\t"
        "lw     %[q5],      (%[sq5])            \n\t"
        "lw     %[q6],      (%[sq6])            \n\t"
        "lw     %[q7],      (%[sq7])            \n\t"

        : [q3] "=&r"(q3), [q2] "=&r"(q2), [q1] "=&r"(q1), [q0] "=&r"(q0),
          [q7] "=&r"(q7), [q6] "=&r"(q6), [q5] "=&r"(q5), [q4] "=&r"(q4)
        : [sq3] "r"(sq3), [sq2] "r"(sq2), [sq1] "r"(sq1), [sq0] "r"(sq0),
          [sq4] "r"(sq4), [sq5] "r"(sq5), [sq6] "r"(sq6), [sq7] "r"(sq7));

    filter_hev_mask_flatmask4_dspr2(limit_vec, flimit_vec, thresh_vec, p1, p0,
                                    p3, p2, q0, q1, q2, q3, &hev, &mask, &flat);

    flatmask5(p7, p6, p5, p4, p0, q0, q4, q5, q6, q7, &flat2);

    /* f0 */
    if (((flat2 == 0) && (flat == 0) && (mask != 0)) ||
        ((flat2 != 0) && (flat == 0) && (mask != 0))) {
      filter1_dspr2(mask, hev, p1, p0, q0, q1, &p1_f0, &p0_f0, &q0_f0, &q1_f0);

      __asm__ __volatile__(
          "sw       %[p1_f0],   (%[sp1])            \n\t"
          "sw       %[p0_f0],   (%[sp0])            \n\t"
          "sw       %[q0_f0],   (%[sq0])            \n\t"
          "sw       %[q1_f0],   (%[sq1])            \n\t"

          :
          : [p1_f0] "r"(p1_f0), [p0_f0] "r"(p0_f0), [q0_f0] "r"(q0_f0),
            [q1_f0] "r"(q1_f0), [sp1] "r"(sp1), [sp0] "r"(sp0), [sq0] "r"(sq0),
            [sq1] "r"(sq1));
    } else if ((flat2 == 0XFFFFFFFF) && (flat == 0xFFFFFFFF) &&
               (mask == 0xFFFFFFFF)) {
      /* f2 */
      PACK_LEFT_0TO3()
      PACK_LEFT_4TO7()
      wide_mbfilter_dspr2(&p7_l, &p6_l, &p5_l, &p4_l, &p3_l, &p2_l, &p1_l,
                          &p0_l, &q0_l, &q1_l, &q2_l, &q3_l, &q4_l, &q5_l,
                          &q6_l, &q7_l);

      PACK_RIGHT_0TO3()
      PACK_RIGHT_4TO7()
      wide_mbfilter_dspr2(&p7_r, &p6_r, &p5_r, &p4_r, &p3_r, &p2_r, &p1_r,
                          &p0_r, &q0_r, &q1_r, &q2_r, &q3_r, &q4_r, &q5_r,
                          &q6_r, &q7_r);

      COMBINE_LEFT_RIGHT_0TO2()
      COMBINE_LEFT_RIGHT_3TO6()

      __asm__ __volatile__(
          "sw         %[p6], (%[sp6])    \n\t"
          "sw         %[p5], (%[sp5])    \n\t"
          "sw         %[p4], (%[sp4])    \n\t"
          "sw         %[p3], (%[sp3])    \n\t"
          "sw         %[p2], (%[sp2])    \n\t"
          "sw         %[p1], (%[sp1])    \n\t"
          "sw         %[p0], (%[sp0])    \n\t"

          :
          : [p6] "r"(p6), [p5] "r"(p5), [p4] "r"(p4), [p3] "r"(p3),
            [p2] "r"(p2), [p1] "r"(p1), [p0] "r"(p0), [sp6] "r"(sp6),
            [sp5] "r"(sp5), [sp4] "r"(sp4), [sp3] "r"(sp3), [sp2] "r"(sp2),
            [sp1] "r"(sp1), [sp0] "r"(sp0));

      __asm__ __volatile__(
          "sw         %[q6], (%[sq6])    \n\t"
          "sw         %[q5], (%[sq5])    \n\t"
          "sw         %[q4], (%[sq4])    \n\t"
          "sw         %[q3], (%[sq3])    \n\t"
          "sw         %[q2], (%[sq2])    \n\t"
          "sw         %[q1], (%[sq1])    \n\t"
          "sw         %[q0], (%[sq0])    \n\t"

          :
          : [q6] "r"(q6), [q5] "r"(q5), [q4] "r"(q4), [q3] "r"(q3),
            [q2] "r"(q2), [q1] "r"(q1), [q0] "r"(q0), [sq6] "r"(sq6),
            [sq5] "r"(sq5), [sq4] "r"(sq4), [sq3] "r"(sq3), [sq2] "r"(sq2),
            [sq1] "r"(sq1), [sq0] "r"(sq0));
    } else if ((flat2 == 0) && (flat == 0xFFFFFFFF) && (mask == 0xFFFFFFFF)) {
      /* f1 */
      /* left 2 element operation */
      PACK_LEFT_0TO3()
      mbfilter_dspr2(&p3_l, &p2_l, &p1_l, &p0_l, &q0_l, &q1_l, &q2_l, &q3_l);

      /* right 2 element operation */
      PACK_RIGHT_0TO3()
      mbfilter_dspr2(&p3_r, &p2_r, &p1_r, &p0_r, &q0_r, &q1_r, &q2_r, &q3_r);

      COMBINE_LEFT_RIGHT_0TO2()

      __asm__ __volatile__(
          "sw         %[p2], (%[sp2])    \n\t"
          "sw         %[p1], (%[sp1])    \n\t"
          "sw         %[p0], (%[sp0])    \n\t"
          "sw         %[q0], (%[sq0])    \n\t"
          "sw         %[q1], (%[sq1])    \n\t"
          "sw         %[q2], (%[sq2])    \n\t"

          :
          : [p2] "r"(p2), [p1] "r"(p1), [p0] "r"(p0), [q0] "r"(q0),
            [q1] "r"(q1), [q2] "r"(q2), [sp2] "r"(sp2), [sp1] "r"(sp1),
            [sp0] "r"(sp0), [sq0] "r"(sq0), [sq1] "r"(sq1), [sq2] "r"(sq2));
    } else if ((flat2 == 0) && (flat != 0) && (mask != 0)) {
      /* f0+f1 */
      filter1_dspr2(mask, hev, p1, p0, q0, q1, &p1_f0, &p0_f0, &q0_f0, &q1_f0);

      /* left 2 element operation */
      PACK_LEFT_0TO3()
      mbfilter_dspr2(&p3_l, &p2_l, &p1_l, &p0_l, &q0_l, &q1_l, &q2_l, &q3_l);

      /* right 2 element operation */
      PACK_RIGHT_0TO3()
      mbfilter_dspr2(&p3_r, &p2_r, &p1_r, &p0_r, &q0_r, &q1_r, &q2_r, &q3_r);

      if (mask & flat & 0x000000FF) {
        __asm__ __volatile__(
            "sb         %[p2_r],  (%[sp2])    \n\t"
            "sb         %[p1_r],  (%[sp1])    \n\t"
            "sb         %[p0_r],  (%[sp0])    \n\t"
            "sb         %[q0_r],  (%[sq0])    \n\t"
            "sb         %[q1_r],  (%[sq1])    \n\t"
            "sb         %[q2_r],  (%[sq2])    \n\t"

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
            "sb         %[p2_r],  +1(%[sp2])    \n\t"
            "sb         %[p1_r],  +1(%[sp1])    \n\t"
            "sb         %[p0_r],  +1(%[sp0])    \n\t"
            "sb         %[q0_r],  +1(%[sq0])    \n\t"
            "sb         %[q1_r],  +1(%[sq1])    \n\t"
            "sb         %[q2_r],  +1(%[sq2])    \n\t"

            :
            : [p2_r] "r"(p2_r), [p1_r] "r"(p1_r), [p0_r] "r"(p0_r),
              [q0_r] "r"(q0_r), [q1_r] "r"(q1_r), [q2_r] "r"(q2_r),
              [sp2] "r"(sp2), [sp1] "r"(sp1), [sp0] "r"(sp0), [sq0] "r"(sq0),
              [sq1] "r"(sq1), [sq2] "r"(sq2));
      } else if (mask & 0x0000FF00) {
        __asm__ __volatile__(
            "sb         %[p1_f0],  +1(%[sp1])    \n\t"
            "sb         %[p0_f0],  +1(%[sp0])    \n\t"
            "sb         %[q0_f0],  +1(%[sq0])    \n\t"
            "sb         %[q1_f0],  +1(%[sq1])    \n\t"

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

          : [p1_f0] "+r"(p1_f0), [p0_f0] "+r"(p0_f0), [q0_f0] "+r"(q0_f0),
            [q1_f0] "+r"(q1_f0)
          :);

      if (mask & flat & 0x00FF0000) {
        __asm__ __volatile__(
            "sb         %[p2_l],  +2(%[sp2])    \n\t"
            "sb         %[p1_l],  +2(%[sp1])    \n\t"
            "sb         %[p0_l],  +2(%[sp0])    \n\t"
            "sb         %[q0_l],  +2(%[sq0])    \n\t"
            "sb         %[q1_l],  +2(%[sq1])    \n\t"
            "sb         %[q2_l],  +2(%[sq2])    \n\t"

            :
            : [p2_l] "r"(p2_l), [p1_l] "r"(p1_l), [p0_l] "r"(p0_l),
              [q0_l] "r"(q0_l), [q1_l] "r"(q1_l), [q2_l] "r"(q2_l),
              [sp2] "r"(sp2), [sp1] "r"(sp1), [sp0] "r"(sp0), [sq0] "r"(sq0),
              [sq1] "r"(sq1), [sq2] "r"(sq2));
      } else if (mask & 0x00FF0000) {
        __asm__ __volatile__(
            "sb         %[p1_f0],  +2(%[sp1])    \n\t"
            "sb         %[p0_f0],  +2(%[sp0])    \n\t"
            "sb         %[q0_f0],  +2(%[sq0])    \n\t"
            "sb         %[q1_f0],  +2(%[sq1])    \n\t"

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
            "sb         %[p2_l],  +3(%[sp2])    \n\t"
            "sb         %[p1_l],  +3(%[sp1])    \n\t"
            "sb         %[p0_l],  +3(%[sp0])    \n\t"
            "sb         %[q0_l],  +3(%[sq0])    \n\t"
            "sb         %[q1_l],  +3(%[sq1])    \n\t"
            "sb         %[q2_l],  +3(%[sq2])    \n\t"

            :
            : [p2_l] "r"(p2_l), [p1_l] "r"(p1_l), [p0_l] "r"(p0_l),
              [q0_l] "r"(q0_l), [q1_l] "r"(q1_l), [q2_l] "r"(q2_l),
              [sp2] "r"(sp2), [sp1] "r"(sp1), [sp0] "r"(sp0), [sq0] "r"(sq0),
              [sq1] "r"(sq1), [sq2] "r"(sq2));
      } else if (mask & 0xFF000000) {
        __asm__ __volatile__(
            "sb         %[p1_f0],  +3(%[sp1])    \n\t"
            "sb         %[p0_f0],  +3(%[sp0])    \n\t"
            "sb         %[q0_f0],  +3(%[sq0])    \n\t"
            "sb         %[q1_f0],  +3(%[sq1])    \n\t"

            :
            : [p1_f0] "r"(p1_f0), [p0_f0] "r"(p0_f0), [q0_f0] "r"(q0_f0),
              [q1_f0] "r"(q1_f0), [sp1] "r"(sp1), [sp0] "r"(sp0),
              [sq0] "r"(sq0), [sq1] "r"(sq1));
      }
    } else if ((flat2 != 0) && (flat != 0) && (mask != 0)) {
      /* f0 + f1 + f2 */
      /* f0  function */
      filter1_dspr2(mask, hev, p1, p0, q0, q1, &p1_f0, &p0_f0, &q0_f0, &q1_f0);

      /* f1  function */
      /* left 2 element operation */
      PACK_LEFT_0TO3()
      mbfilter1_dspr2(p3_l, p2_l, p1_l, p0_l, q0_l, q1_l, q2_l, q3_l, &p2_l_f1,
                      &p1_l_f1, &p0_l_f1, &q0_l_f1, &q1_l_f1, &q2_l_f1);

      /* right 2 element operation */
      PACK_RIGHT_0TO3()
      mbfilter1_dspr2(p3_r, p2_r, p1_r, p0_r, q0_r, q1_r, q2_r, q3_r, &p2_r_f1,
                      &p1_r_f1, &p0_r_f1, &q0_r_f1, &q1_r_f1, &q2_r_f1);

      /* f2  function */
      PACK_LEFT_4TO7()
      wide_mbfilter_dspr2(&p7_l, &p6_l, &p5_l, &p4_l, &p3_l, &p2_l, &p1_l,
                          &p0_l, &q0_l, &q1_l, &q2_l, &q3_l, &q4_l, &q5_l,
                          &q6_l, &q7_l);

      PACK_RIGHT_4TO7()
      wide_mbfilter_dspr2(&p7_r, &p6_r, &p5_r, &p4_r, &p3_r, &p2_r, &p1_r,
                          &p0_r, &q0_r, &q1_r, &q2_r, &q3_r, &q4_r, &q5_r,
                          &q6_r, &q7_r);

      if (mask & flat & flat2 & 0x000000FF) {
        __asm__ __volatile__(
            "sb         %[p6_r],  (%[sp6])    \n\t"
            "sb         %[p5_r],  (%[sp5])    \n\t"
            "sb         %[p4_r],  (%[sp4])    \n\t"
            "sb         %[p3_r],  (%[sp3])    \n\t"
            "sb         %[p2_r],  (%[sp2])    \n\t"
            "sb         %[p1_r],  (%[sp1])    \n\t"
            "sb         %[p0_r],  (%[sp0])    \n\t"

            :
            : [p6_r] "r"(p6_r), [p5_r] "r"(p5_r), [p4_r] "r"(p4_r),
              [p3_r] "r"(p3_r), [p2_r] "r"(p2_r), [p1_r] "r"(p1_r),
              [sp6] "r"(sp6), [sp5] "r"(sp5), [sp4] "r"(sp4), [sp3] "r"(sp3),
              [sp2] "r"(sp2), [sp1] "r"(sp1), [p0_r] "r"(p0_r), [sp0] "r"(sp0));

        __asm__ __volatile__(
            "sb         %[q0_r],  (%[sq0])    \n\t"
            "sb         %[q1_r],  (%[sq1])    \n\t"
            "sb         %[q2_r],  (%[sq2])    \n\t"
            "sb         %[q3_r],  (%[sq3])    \n\t"
            "sb         %[q4_r],  (%[sq4])    \n\t"
            "sb         %[q5_r],  (%[sq5])    \n\t"
            "sb         %[q6_r],  (%[sq6])    \n\t"

            :
            : [q0_r] "r"(q0_r), [q1_r] "r"(q1_r), [q2_r] "r"(q2_r),
              [q3_r] "r"(q3_r), [q4_r] "r"(q4_r), [q5_r] "r"(q5_r),
              [q6_r] "r"(q6_r), [sq0] "r"(sq0), [sq1] "r"(sq1), [sq2] "r"(sq2),
              [sq3] "r"(sq3), [sq4] "r"(sq4), [sq5] "r"(sq5), [sq6] "r"(sq6));
      } else if (mask & flat & 0x000000FF) {
        __asm__ __volatile__(
            "sb         %[p2_r_f1],  (%[sp2])    \n\t"
            "sb         %[p1_r_f1],  (%[sp1])    \n\t"
            "sb         %[p0_r_f1],  (%[sp0])    \n\t"
            "sb         %[q0_r_f1],  (%[sq0])    \n\t"
            "sb         %[q1_r_f1],  (%[sq1])    \n\t"
            "sb         %[q2_r_f1],  (%[sq2])    \n\t"

            :
            : [p2_r_f1] "r"(p2_r_f1), [p1_r_f1] "r"(p1_r_f1),
              [p0_r_f1] "r"(p0_r_f1), [q0_r_f1] "r"(q0_r_f1),
              [q1_r_f1] "r"(q1_r_f1), [q2_r_f1] "r"(q2_r_f1), [sp2] "r"(sp2),
              [sp1] "r"(sp1), [sp0] "r"(sp0), [sq0] "r"(sq0), [sq1] "r"(sq1),
              [sq2] "r"(sq2));
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
          "srl        %[p6_r], %[p6_r], 16     \n\t"
          "srl        %[p5_r], %[p5_r], 16     \n\t"
          "srl        %[p4_r], %[p4_r], 16     \n\t"
          "srl        %[p3_r], %[p3_r], 16     \n\t"
          "srl        %[p2_r], %[p2_r], 16     \n\t"
          "srl        %[p1_r], %[p1_r], 16     \n\t"
          "srl        %[p0_r], %[p0_r], 16     \n\t"
          "srl        %[q0_r], %[q0_r], 16     \n\t"
          "srl        %[q1_r], %[q1_r], 16     \n\t"
          "srl        %[q2_r], %[q2_r], 16     \n\t"
          "srl        %[q3_r], %[q3_r], 16     \n\t"
          "srl        %[q4_r], %[q4_r], 16     \n\t"
          "srl        %[q5_r], %[q5_r], 16     \n\t"
          "srl        %[q6_r], %[q6_r], 16     \n\t"

          : [q0_r] "+r"(q0_r), [q1_r] "+r"(q1_r), [q2_r] "+r"(q2_r),
            [q3_r] "+r"(q3_r), [q4_r] "+r"(q4_r), [q5_r] "+r"(q5_r),
            [p6_r] "+r"(p6_r), [p5_r] "+r"(p5_r), [p4_r] "+r"(p4_r),
            [p3_r] "+r"(p3_r), [p2_r] "+r"(p2_r), [p1_r] "+r"(p1_r),
            [q6_r] "+r"(q6_r), [p0_r] "+r"(p0_r)
          :);

      __asm__ __volatile__(
          "srl        %[p2_r_f1], %[p2_r_f1], 16     \n\t"
          "srl        %[p1_r_f1], %[p1_r_f1], 16     \n\t"
          "srl        %[p0_r_f1], %[p0_r_f1], 16     \n\t"
          "srl        %[q0_r_f1], %[q0_r_f1], 16     \n\t"
          "srl        %[q1_r_f1], %[q1_r_f1], 16     \n\t"
          "srl        %[q2_r_f1], %[q2_r_f1], 16     \n\t"
          "srl        %[p1_f0],   %[p1_f0],   8      \n\t"
          "srl        %[p0_f0],   %[p0_f0],   8      \n\t"
          "srl        %[q0_f0],   %[q0_f0],   8      \n\t"
          "srl        %[q1_f0],   %[q1_f0],   8      \n\t"

          : [p2_r_f1] "+r"(p2_r_f1), [p1_r_f1] "+r"(p1_r_f1),
            [p0_r_f1] "+r"(p0_r_f1), [q0_r_f1] "+r"(q0_r_f1),
            [q1_r_f1] "+r"(q1_r_f1), [q2_r_f1] "+r"(q2_r_f1),
            [p1_f0] "+r"(p1_f0), [p0_f0] "+r"(p0_f0), [q0_f0] "+r"(q0_f0),
            [q1_f0] "+r"(q1_f0)
          :);

      if (mask & flat & flat2 & 0x0000FF00) {
        __asm__ __volatile__(
            "sb         %[p6_r],  +1(%[sp6])    \n\t"
            "sb         %[p5_r],  +1(%[sp5])    \n\t"
            "sb         %[p4_r],  +1(%[sp4])    \n\t"
            "sb         %[p3_r],  +1(%[sp3])    \n\t"
            "sb         %[p2_r],  +1(%[sp2])    \n\t"
            "sb         %[p1_r],  +1(%[sp1])    \n\t"
            "sb         %[p0_r],  +1(%[sp0])    \n\t"

            :
            : [p6_r] "r"(p6_r), [p5_r] "r"(p5_r), [p4_r] "r"(p4_r),
              [p3_r] "r"(p3_r), [p2_r] "r"(p2_r), [p1_r] "r"(p1_r),
              [p0_r] "r"(p0_r), [sp6] "r"(sp6), [sp5] "r"(sp5), [sp4] "r"(sp4),
              [sp3] "r"(sp3), [sp2] "r"(sp2), [sp1] "r"(sp1), [sp0] "r"(sp0));

        __asm__ __volatile__(
            "sb         %[q0_r],  +1(%[sq0])    \n\t"
            "sb         %[q1_r],  +1(%[sq1])    \n\t"
            "sb         %[q2_r],  +1(%[sq2])    \n\t"
            "sb         %[q3_r],  +1(%[sq3])    \n\t"
            "sb         %[q4_r],  +1(%[sq4])    \n\t"
            "sb         %[q5_r],  +1(%[sq5])    \n\t"
            "sb         %[q6_r],  +1(%[sq6])    \n\t"

            :
            : [q0_r] "r"(q0_r), [q1_r] "r"(q1_r), [q2_r] "r"(q2_r),
              [q3_r] "r"(q3_r), [q4_r] "r"(q4_r), [q5_r] "r"(q5_r),
              [q6_r] "r"(q6_r), [sq0] "r"(sq0), [sq1] "r"(sq1), [sq2] "r"(sq2),
              [sq3] "r"(sq3), [sq4] "r"(sq4), [sq5] "r"(sq5), [sq6] "r"(sq6));
      } else if (mask & flat & 0x0000FF00) {
        __asm__ __volatile__(
            "sb         %[p2_r_f1],  +1(%[sp2])    \n\t"
            "sb         %[p1_r_f1],  +1(%[sp1])    \n\t"
            "sb         %[p0_r_f1],  +1(%[sp0])    \n\t"
            "sb         %[q0_r_f1],  +1(%[sq0])    \n\t"
            "sb         %[q1_r_f1],  +1(%[sq1])    \n\t"
            "sb         %[q2_r_f1],  +1(%[sq2])    \n\t"

            :
            : [p2_r_f1] "r"(p2_r_f1), [p1_r_f1] "r"(p1_r_f1),
              [p0_r_f1] "r"(p0_r_f1), [q0_r_f1] "r"(q0_r_f1),
              [q1_r_f1] "r"(q1_r_f1), [q2_r_f1] "r"(q2_r_f1), [sp2] "r"(sp2),
              [sp1] "r"(sp1), [sp0] "r"(sp0), [sq0] "r"(sq0), [sq1] "r"(sq1),
              [sq2] "r"(sq2));
      } else if (mask & 0x0000FF00) {
        __asm__ __volatile__(
            "sb         %[p1_f0],  +1(%[sp1])    \n\t"
            "sb         %[p0_f0],  +1(%[sp0])    \n\t"
            "sb         %[q0_f0],  +1(%[sq0])    \n\t"
            "sb         %[q1_f0],  +1(%[sq1])    \n\t"

            :
            : [p1_f0] "r"(p1_f0), [p0_f0] "r"(p0_f0), [q0_f0] "r"(q0_f0),
              [q1_f0] "r"(q1_f0), [sp1] "r"(sp1), [sp0] "r"(sp0),
              [sq0] "r"(sq0), [sq1] "r"(sq1));
      }

      __asm__ __volatile__(
          "srl        %[p1_f0], %[p1_f0], 8     \n\t"
          "srl        %[p0_f0], %[p0_f0], 8     \n\t"
          "srl        %[q0_f0], %[q0_f0], 8     \n\t"
          "srl        %[q1_f0], %[q1_f0], 8     \n\t"

          : [p1_f0] "+r"(p1_f0), [p0_f0] "+r"(p0_f0), [q0_f0] "+r"(q0_f0),
            [q1_f0] "+r"(q1_f0)
          :);

      if (mask & flat & flat2 & 0x00FF0000) {
        __asm__ __volatile__(
            "sb         %[p6_l],  +2(%[sp6])    \n\t"
            "sb         %[p5_l],  +2(%[sp5])    \n\t"
            "sb         %[p4_l],  +2(%[sp4])    \n\t"
            "sb         %[p3_l],  +2(%[sp3])    \n\t"
            "sb         %[p2_l],  +2(%[sp2])    \n\t"
            "sb         %[p1_l],  +2(%[sp1])    \n\t"
            "sb         %[p0_l],  +2(%[sp0])    \n\t"

            :
            : [p6_l] "r"(p6_l), [p5_l] "r"(p5_l), [p4_l] "r"(p4_l),
              [p3_l] "r"(p3_l), [p2_l] "r"(p2_l), [p1_l] "r"(p1_l),
              [p0_l] "r"(p0_l), [sp6] "r"(sp6), [sp5] "r"(sp5), [sp4] "r"(sp4),
              [sp3] "r"(sp3), [sp2] "r"(sp2), [sp1] "r"(sp1), [sp0] "r"(sp0));

        __asm__ __volatile__(
            "sb         %[q0_l],  +2(%[sq0])    \n\t"
            "sb         %[q1_l],  +2(%[sq1])    \n\t"
            "sb         %[q2_l],  +2(%[sq2])    \n\t"
            "sb         %[q3_l],  +2(%[sq3])    \n\t"
            "sb         %[q4_l],  +2(%[sq4])    \n\t"
            "sb         %[q5_l],  +2(%[sq5])    \n\t"
            "sb         %[q6_l],  +2(%[sq6])    \n\t"

            :
            : [q0_l] "r"(q0_l), [q1_l] "r"(q1_l), [q2_l] "r"(q2_l),
              [q3_l] "r"(q3_l), [q4_l] "r"(q4_l), [q5_l] "r"(q5_l),
              [q6_l] "r"(q6_l), [sq0] "r"(sq0), [sq1] "r"(sq1), [sq2] "r"(sq2),
              [sq3] "r"(sq3), [sq4] "r"(sq4), [sq5] "r"(sq5), [sq6] "r"(sq6));
      } else if (mask & flat & 0x00FF0000) {
        __asm__ __volatile__(
            "sb         %[p2_l_f1],  +2(%[sp2])    \n\t"
            "sb         %[p1_l_f1],  +2(%[sp1])    \n\t"
            "sb         %[p0_l_f1],  +2(%[sp0])    \n\t"
            "sb         %[q0_l_f1],  +2(%[sq0])    \n\t"
            "sb         %[q1_l_f1],  +2(%[sq1])    \n\t"
            "sb         %[q2_l_f1],  +2(%[sq2])    \n\t"

            :
            : [p2_l_f1] "r"(p2_l_f1), [p1_l_f1] "r"(p1_l_f1),
              [p0_l_f1] "r"(p0_l_f1), [q0_l_f1] "r"(q0_l_f1),
              [q1_l_f1] "r"(q1_l_f1), [q2_l_f1] "r"(q2_l_f1), [sp2] "r"(sp2),
              [sp1] "r"(sp1), [sp0] "r"(sp0), [sq0] "r"(sq0), [sq1] "r"(sq1),
              [sq2] "r"(sq2));
      } else if (mask & 0x00FF0000) {
        __asm__ __volatile__(
            "sb         %[p1_f0],  +2(%[sp1])    \n\t"
            "sb         %[p0_f0],  +2(%[sp0])    \n\t"
            "sb         %[q0_f0],  +2(%[sq0])    \n\t"
            "sb         %[q1_f0],  +2(%[sq1])    \n\t"

            :
            : [p1_f0] "r"(p1_f0), [p0_f0] "r"(p0_f0), [q0_f0] "r"(q0_f0),
              [q1_f0] "r"(q1_f0), [sp1] "r"(sp1), [sp0] "r"(sp0),
              [sq0] "r"(sq0), [sq1] "r"(sq1));
      }

      __asm__ __volatile__(
          "srl      %[p6_l],    %[p6_l],    16   \n\t"
          "srl      %[p5_l],    %[p5_l],    16   \n\t"
          "srl      %[p4_l],    %[p4_l],    16   \n\t"
          "srl      %[p3_l],    %[p3_l],    16   \n\t"
          "srl      %[p2_l],    %[p2_l],    16   \n\t"
          "srl      %[p1_l],    %[p1_l],    16   \n\t"
          "srl      %[p0_l],    %[p0_l],    16   \n\t"
          "srl      %[q0_l],    %[q0_l],    16   \n\t"
          "srl      %[q1_l],    %[q1_l],    16   \n\t"
          "srl      %[q2_l],    %[q2_l],    16   \n\t"
          "srl      %[q3_l],    %[q3_l],    16   \n\t"
          "srl      %[q4_l],    %[q4_l],    16   \n\t"
          "srl      %[q5_l],    %[q5_l],    16   \n\t"
          "srl      %[q6_l],    %[q6_l],    16   \n\t"

          : [q0_l] "+r"(q0_l), [q1_l] "+r"(q1_l), [q2_l] "+r"(q2_l),
            [q3_l] "+r"(q3_l), [q4_l] "+r"(q4_l), [q5_l] "+r"(q5_l),
            [q6_l] "+r"(q6_l), [p6_l] "+r"(p6_l), [p5_l] "+r"(p5_l),
            [p4_l] "+r"(p4_l), [p3_l] "+r"(p3_l), [p2_l] "+r"(p2_l),
            [p1_l] "+r"(p1_l), [p0_l] "+r"(p0_l)
          :);

      __asm__ __volatile__(
          "srl      %[p2_l_f1],   %[p2_l_f1],   16   \n\t"
          "srl      %[p1_l_f1],   %[p1_l_f1],   16   \n\t"
          "srl      %[p0_l_f1],   %[p0_l_f1],   16   \n\t"
          "srl      %[q0_l_f1],   %[q0_l_f1],   16   \n\t"
          "srl      %[q1_l_f1],   %[q1_l_f1],   16   \n\t"
          "srl      %[q2_l_f1],   %[q2_l_f1],   16   \n\t"
          "srl      %[p1_f0],     %[p1_f0],     8    \n\t"
          "srl      %[p0_f0],     %[p0_f0],     8    \n\t"
          "srl      %[q0_f0],     %[q0_f0],     8    \n\t"
          "srl      %[q1_f0],     %[q1_f0],     8    \n\t"

          : [p2_l_f1] "+r"(p2_l_f1), [p1_l_f1] "+r"(p1_l_f1),
            [p0_l_f1] "+r"(p0_l_f1), [q0_l_f1] "+r"(q0_l_f1),
            [q1_l_f1] "+r"(q1_l_f1), [q2_l_f1] "+r"(q2_l_f1),
            [p1_f0] "+r"(p1_f0), [p0_f0] "+r"(p0_f0), [q0_f0] "+r"(q0_f0),
            [q1_f0] "+r"(q1_f0)
          :);

      if (mask & flat & flat2 & 0xFF000000) {
        __asm__ __volatile__(
            "sb     %[p6_l],    +3(%[sp6])    \n\t"
            "sb     %[p5_l],    +3(%[sp5])    \n\t"
            "sb     %[p4_l],    +3(%[sp4])    \n\t"
            "sb     %[p3_l],    +3(%[sp3])    \n\t"
            "sb     %[p2_l],    +3(%[sp2])    \n\t"
            "sb     %[p1_l],    +3(%[sp1])    \n\t"
            "sb     %[p0_l],    +3(%[sp0])    \n\t"

            :
            : [p6_l] "r"(p6_l), [p5_l] "r"(p5_l), [p4_l] "r"(p4_l),
              [p3_l] "r"(p3_l), [p2_l] "r"(p2_l), [p1_l] "r"(p1_l),
              [p0_l] "r"(p0_l), [sp6] "r"(sp6), [sp5] "r"(sp5), [sp4] "r"(sp4),
              [sp3] "r"(sp3), [sp2] "r"(sp2), [sp1] "r"(sp1), [sp0] "r"(sp0));

        __asm__ __volatile__(
            "sb     %[q0_l],    +3(%[sq0])    \n\t"
            "sb     %[q1_l],    +3(%[sq1])    \n\t"
            "sb     %[q2_l],    +3(%[sq2])    \n\t"
            "sb     %[q3_l],    +3(%[sq3])    \n\t"
            "sb     %[q4_l],    +3(%[sq4])    \n\t"
            "sb     %[q5_l],    +3(%[sq5])    \n\t"
            "sb     %[q6_l],    +3(%[sq6])    \n\t"

            :
            : [q0_l] "r"(q0_l), [q1_l] "r"(q1_l), [q2_l] "r"(q2_l),
              [q3_l] "r"(q3_l), [q4_l] "r"(q4_l), [q5_l] "r"(q5_l),
              [sq0] "r"(sq0), [sq1] "r"(sq1), [sq2] "r"(sq2), [sq3] "r"(sq3),
              [sq4] "r"(sq4), [sq5] "r"(sq5), [q6_l] "r"(q6_l), [sq6] "r"(sq6));
      } else if (mask & flat & 0xFF000000) {
        __asm__ __volatile__(
            "sb     %[p2_l_f1],     +3(%[sp2])    \n\t"
            "sb     %[p1_l_f1],     +3(%[sp1])    \n\t"
            "sb     %[p0_l_f1],     +3(%[sp0])    \n\t"
            "sb     %[q0_l_f1],     +3(%[sq0])    \n\t"
            "sb     %[q1_l_f1],     +3(%[sq1])    \n\t"
            "sb     %[q2_l_f1],     +3(%[sq2])    \n\t"

            :
            : [p2_l_f1] "r"(p2_l_f1), [p1_l_f1] "r"(p1_l_f1),
              [p0_l_f1] "r"(p0_l_f1), [q0_l_f1] "r"(q0_l_f1),
              [q1_l_f1] "r"(q1_l_f1), [q2_l_f1] "r"(q2_l_f1), [sp2] "r"(sp2),
              [sp1] "r"(sp1), [sp0] "r"(sp0), [sq0] "r"(sq0), [sq1] "r"(sq1),
              [sq2] "r"(sq2));
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

void vpx_lpf_horizontal_16_dspr2(unsigned char *s, int pitch,
                                 const uint8_t *blimit, const uint8_t *limit,
                                 const uint8_t *thresh) {
  mb_lpf_horizontal_edge(s, pitch, blimit, limit, thresh, 1);
}

void vpx_lpf_horizontal_16_dual_dspr2(unsigned char *s, int pitch,
                                      const uint8_t *blimit,
                                      const uint8_t *limit,
                                      const uint8_t *thresh) {
  mb_lpf_horizontal_edge(s, pitch, blimit, limit, thresh, 2);
}
#endif  // #if HAVE_DSPR2
