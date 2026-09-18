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
void vpx_lpf_horizontal_4_dspr2(unsigned char *s, int pitch,
                                const uint8_t *blimit, const uint8_t *limit,
                                const uint8_t *thresh) {
  uint8_t i;
  uint32_t mask;
  uint32_t hev;
  uint32_t pm1, p0, p1, p2, p3, p4, p5, p6;
  uint8_t *sm1, *s0, *s1, *s2, *s3, *s4, *s5, *s6;
  uint32_t thresh_vec, flimit_vec, limit_vec;
  uint32_t uflimit, ulimit, uthresh;

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

  /* loop filter designed to work using chars so that we can make maximum use
     of 8 bit simd instructions. */
  for (i = 0; i < 2; i++) {
    sm1 = s - (pitch << 2);
    s0 = sm1 + pitch;
    s1 = s0 + pitch;
    s2 = s - pitch;
    s3 = s;
    s4 = s + pitch;
    s5 = s4 + pitch;
    s6 = s5 + pitch;

    __asm__ __volatile__(
        "lw     %[p1],  (%[s1])    \n\t"
        "lw     %[p2],  (%[s2])    \n\t"
        "lw     %[p3],  (%[s3])    \n\t"
        "lw     %[p4],  (%[s4])    \n\t"

        : [p1] "=&r"(p1), [p2] "=&r"(p2), [p3] "=&r"(p3), [p4] "=&r"(p4)
        : [s1] "r"(s1), [s2] "r"(s2), [s3] "r"(s3), [s4] "r"(s4));

    /* if (p1 - p4 == 0) and (p2 - p3 == 0)
       mask will be zero and filtering is not needed */
    if (!(((p1 - p4) == 0) && ((p2 - p3) == 0))) {
      __asm__ __volatile__(
          "lw       %[pm1], (%[sm1])   \n\t"
          "lw       %[p0],  (%[s0])    \n\t"
          "lw       %[p5],  (%[s5])    \n\t"
          "lw       %[p6],  (%[s6])    \n\t"

          : [pm1] "=&r"(pm1), [p0] "=&r"(p0), [p5] "=&r"(p5), [p6] "=&r"(p6)
          : [sm1] "r"(sm1), [s0] "r"(s0), [s5] "r"(s5), [s6] "r"(s6));

      filter_hev_mask_dspr2(limit_vec, flimit_vec, p1, p2, pm1, p0, p3, p4, p5,
                            p6, thresh_vec, &hev, &mask);

      /* if mask == 0 do filtering is not needed */
      if (mask) {
        /* filtering */
        filter_dspr2(mask, hev, &p1, &p2, &p3, &p4);

        __asm__ __volatile__(
            "sw     %[p1],  (%[s1])    \n\t"
            "sw     %[p2],  (%[s2])    \n\t"
            "sw     %[p3],  (%[s3])    \n\t"
            "sw     %[p4],  (%[s4])    \n\t"

            :
            : [p1] "r"(p1), [p2] "r"(p2), [p3] "r"(p3), [p4] "r"(p4),
              [s1] "r"(s1), [s2] "r"(s2), [s3] "r"(s3), [s4] "r"(s4));
      }
    }

    s = s + 4;
  }
}

void vpx_lpf_vertical_4_dspr2(unsigned char *s, int pitch,
                              const uint8_t *blimit, const uint8_t *limit,
                              const uint8_t *thresh) {
  uint8_t i;
  uint32_t mask, hev;
  uint32_t pm1, p0, p1, p2, p3, p4, p5, p6;
  uint8_t *s1, *s2, *s3, *s4;
  uint32_t prim1, prim2, sec3, sec4, prim3, prim4;
  uint32_t thresh_vec, flimit_vec, limit_vec;
  uint32_t uflimit, ulimit, uthresh;

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
  prefetch_store(s + pitch);

  for (i = 0; i < 2; i++) {
    s1 = s;
    s2 = s + pitch;
    s3 = s2 + pitch;
    s4 = s3 + pitch;
    s = s4 + pitch;

    /* load quad-byte vectors
     * memory is 4 byte aligned
     */
    p2 = *((uint32_t *)(s1 - 4));
    p6 = *((uint32_t *)(s1));
    p1 = *((uint32_t *)(s2 - 4));
    p5 = *((uint32_t *)(s2));
    p0 = *((uint32_t *)(s3 - 4));
    p4 = *((uint32_t *)(s3));
    pm1 = *((uint32_t *)(s4 - 4));
    p3 = *((uint32_t *)(s4));

    /* transpose pm1, p0, p1, p2 */
    __asm__ __volatile__(
        "precrq.qb.ph   %[prim1],   %[p2],      %[p1]       \n\t"
        "precr.qb.ph    %[prim2],   %[p2],      %[p1]       \n\t"
        "precrq.qb.ph   %[prim3],   %[p0],      %[pm1]      \n\t"
        "precr.qb.ph    %[prim4],   %[p0],      %[pm1]      \n\t"

        "precrq.qb.ph   %[p1],      %[prim1],   %[prim2]    \n\t"
        "precr.qb.ph    %[pm1],     %[prim1],   %[prim2]    \n\t"
        "precrq.qb.ph   %[sec3],    %[prim3],   %[prim4]    \n\t"
        "precr.qb.ph    %[sec4],    %[prim3],   %[prim4]    \n\t"

        "precrq.ph.w    %[p2],      %[p1],      %[sec3]     \n\t"
        "precrq.ph.w    %[p0],      %[pm1],     %[sec4]     \n\t"
        "append         %[p1],      %[sec3],    16          \n\t"
        "append         %[pm1],     %[sec4],    16          \n\t"

        : [prim1] "=&r"(prim1), [prim2] "=&r"(prim2), [prim3] "=&r"(prim3),
          [prim4] "=&r"(prim4), [p2] "+r"(p2), [p1] "+r"(p1), [p0] "+r"(p0),
          [pm1] "+r"(pm1), [sec3] "=&r"(sec3), [sec4] "=&r"(sec4)
        :);

    /* transpose p3, p4, p5, p6 */
    __asm__ __volatile__(
        "precrq.qb.ph   %[prim1],   %[p6],      %[p5]       \n\t"
        "precr.qb.ph    %[prim2],   %[p6],      %[p5]       \n\t"
        "precrq.qb.ph   %[prim3],   %[p4],      %[p3]       \n\t"
        "precr.qb.ph    %[prim4],   %[p4],      %[p3]       \n\t"

        "precrq.qb.ph   %[p5],      %[prim1],   %[prim2]    \n\t"
        "precr.qb.ph    %[p3],      %[prim1],   %[prim2]    \n\t"
        "precrq.qb.ph   %[sec3],    %[prim3],   %[prim4]    \n\t"
        "precr.qb.ph    %[sec4],    %[prim3],   %[prim4]    \n\t"

        "precrq.ph.w    %[p6],      %[p5],      %[sec3]     \n\t"
        "precrq.ph.w    %[p4],      %[p3],      %[sec4]     \n\t"
        "append         %[p5],      %[sec3],    16          \n\t"
        "append         %[p3],      %[sec4],    16          \n\t"

        : [prim1] "=&r"(prim1), [prim2] "=&r"(prim2), [prim3] "=&r"(prim3),
          [prim4] "=&r"(prim4), [p6] "+r"(p6), [p5] "+r"(p5), [p4] "+r"(p4),
          [p3] "+r"(p3), [sec3] "=&r"(sec3), [sec4] "=&r"(sec4)
        :);

    /* if (p1 - p4 == 0) and (p2 - p3 == 0)
     * mask will be zero and filtering is not needed
     */
    if (!(((p1 - p4) == 0) && ((p2 - p3) == 0))) {
      filter_hev_mask_dspr2(limit_vec, flimit_vec, p1, p2, pm1, p0, p3, p4, p5,
                            p6, thresh_vec, &hev, &mask);

      /* if mask == 0 do filtering is not needed */
      if (mask) {
        /* filtering */
        filter_dspr2(mask, hev, &p1, &p2, &p3, &p4);

        /* unpack processed 4x4 neighborhood
         * don't use transpose on output data
         * because memory isn't aligned
         */
        __asm__ __volatile__(
            "sb     %[p4],   1(%[s4])    \n\t"
            "sb     %[p3],   0(%[s4])    \n\t"
            "sb     %[p2],  -1(%[s4])    \n\t"
            "sb     %[p1],  -2(%[s4])    \n\t"

            :
            : [p4] "r"(p4), [p3] "r"(p3), [p2] "r"(p2), [p1] "r"(p1),
              [s4] "r"(s4));

        __asm__ __volatile__(
            "srl    %[p4],  %[p4],  8     \n\t"
            "srl    %[p3],  %[p3],  8     \n\t"
            "srl    %[p2],  %[p2],  8     \n\t"
            "srl    %[p1],  %[p1],  8     \n\t"

            : [p4] "+r"(p4), [p3] "+r"(p3), [p2] "+r"(p2), [p1] "+r"(p1)
            :);

        __asm__ __volatile__(
            "sb     %[p4],   1(%[s3])    \n\t"
            "sb     %[p3],   0(%[s3])    \n\t"
            "sb     %[p2],  -1(%[s3])    \n\t"
            "sb     %[p1],  -2(%[s3])    \n\t"

            : [p1] "+r"(p1)
            : [p4] "r"(p4), [p3] "r"(p3), [p2] "r"(p2), [s3] "r"(s3));

        __asm__ __volatile__(
            "srl    %[p4],  %[p4],  8     \n\t"
            "srl    %[p3],  %[p3],  8     \n\t"
            "srl    %[p2],  %[p2],  8     \n\t"
            "srl    %[p1],  %[p1],  8     \n\t"

            : [p4] "+r"(p4), [p3] "+r"(p3), [p2] "+r"(p2), [p1] "+r"(p1)
            :);

        __asm__ __volatile__(
            "sb     %[p4],   1(%[s2])    \n\t"
            "sb     %[p3],   0(%[s2])    \n\t"
            "sb     %[p2],  -1(%[s2])    \n\t"
            "sb     %[p1],  -2(%[s2])    \n\t"

            :
            : [p4] "r"(p4), [p3] "r"(p3), [p2] "r"(p2), [p1] "r"(p1),
              [s2] "r"(s2));

        __asm__ __volatile__(
            "srl    %[p4],  %[p4],  8     \n\t"
            "srl    %[p3],  %[p3],  8     \n\t"
            "srl    %[p2],  %[p2],  8     \n\t"
            "srl    %[p1],  %[p1],  8     \n\t"

            : [p4] "+r"(p4), [p3] "+r"(p3), [p2] "+r"(p2), [p1] "+r"(p1)
            :);

        __asm__ __volatile__(
            "sb     %[p4],   1(%[s1])    \n\t"
            "sb     %[p3],   0(%[s1])    \n\t"
            "sb     %[p2],  -1(%[s1])    \n\t"
            "sb     %[p1],  -2(%[s1])    \n\t"

            :
            : [p4] "r"(p4), [p3] "r"(p3), [p2] "r"(p2), [p1] "r"(p1),
              [s1] "r"(s1));
      }
    }
  }
}

void vpx_lpf_horizontal_4_dual_dspr2(
    uint8_t *s, int p /* pitch */, const uint8_t *blimit0,
    const uint8_t *limit0, const uint8_t *thresh0, const uint8_t *blimit1,
    const uint8_t *limit1, const uint8_t *thresh1) {
  vpx_lpf_horizontal_4_dspr2(s, p, blimit0, limit0, thresh0);
  vpx_lpf_horizontal_4_dspr2(s + 8, p, blimit1, limit1, thresh1);
}

void vpx_lpf_horizontal_8_dual_dspr2(
    uint8_t *s, int p /* pitch */, const uint8_t *blimit0,
    const uint8_t *limit0, const uint8_t *thresh0, const uint8_t *blimit1,
    const uint8_t *limit1, const uint8_t *thresh1) {
  vpx_lpf_horizontal_8_dspr2(s, p, blimit0, limit0, thresh0);
  vpx_lpf_horizontal_8_dspr2(s + 8, p, blimit1, limit1, thresh1);
}

void vpx_lpf_vertical_4_dual_dspr2(uint8_t *s, int p, const uint8_t *blimit0,
                                   const uint8_t *limit0,
                                   const uint8_t *thresh0,
                                   const uint8_t *blimit1,
                                   const uint8_t *limit1,
                                   const uint8_t *thresh1) {
  vpx_lpf_vertical_4_dspr2(s, p, blimit0, limit0, thresh0);
  vpx_lpf_vertical_4_dspr2(s + 8 * p, p, blimit1, limit1, thresh1);
}

void vpx_lpf_vertical_8_dual_dspr2(uint8_t *s, int p, const uint8_t *blimit0,
                                   const uint8_t *limit0,
                                   const uint8_t *thresh0,
                                   const uint8_t *blimit1,
                                   const uint8_t *limit1,
                                   const uint8_t *thresh1) {
  vpx_lpf_vertical_8_dspr2(s, p, blimit0, limit0, thresh0);
  vpx_lpf_vertical_8_dspr2(s + 8 * p, p, blimit1, limit1, thresh1);
}

void vpx_lpf_vertical_16_dual_dspr2(uint8_t *s, int p, const uint8_t *blimit,
                                    const uint8_t *limit,
                                    const uint8_t *thresh) {
  vpx_lpf_vertical_16_dspr2(s, p, blimit, limit, thresh);
  vpx_lpf_vertical_16_dspr2(s + 8 * p, p, blimit, limit, thresh);
}
#endif  // #if HAVE_DSPR2
