/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <stdlib.h>
#include "vp8_rtcd.h"
#include "vp8/common/onyxc_int.h"

#if HAVE_DSPR2
typedef unsigned char uc;

/* prefetch data for load */
inline void prefetch_load_lf(unsigned char *src) {
  __asm__ __volatile__("pref   0,  0(%[src])   \n\t" : : [src] "r"(src));
}

/* prefetch data for store */
inline void prefetch_store_lf(unsigned char *dst) {
  __asm__ __volatile__("pref   1,  0(%[dst])   \n\t" : : [dst] "r"(dst));
}

/* processing 4 pixels at the same time
 * compute hev and mask in the same function
 */
static __inline void vp8_filter_mask_vec_mips(
    uint32_t limit, uint32_t flimit, uint32_t p1, uint32_t p0, uint32_t p3,
    uint32_t p2, uint32_t q0, uint32_t q1, uint32_t q2, uint32_t q3,
    uint32_t thresh, uint32_t *hev, uint32_t *mask) {
  uint32_t c, r, r3, r_k;
  uint32_t s1, s2, s3;
  uint32_t ones = 0xFFFFFFFF;
  uint32_t hev1;

  __asm__ __volatile__(
      /* mask |= (abs(p3 - p2) > limit) */
      "subu_s.qb      %[c],   %[p3],     %[p2]        \n\t"
      "subu_s.qb      %[r_k], %[p2],     %[p3]        \n\t"
      "or             %[r_k], %[r_k],    %[c]         \n\t"
      "cmpgu.lt.qb    %[c],   %[limit],  %[r_k]       \n\t"
      "or             %[r],   $0,        %[c]         \n\t"

      /* mask |= (abs(p2 - p1) > limit) */
      "subu_s.qb      %[c],   %[p2],     %[p1]        \n\t"
      "subu_s.qb      %[r_k], %[p1],     %[p2]        \n\t"
      "or             %[r_k], %[r_k],    %[c]         \n\t"
      "cmpgu.lt.qb    %[c],   %[limit],  %[r_k]       \n\t"
      "or             %[r],   %[r],      %[c]         \n\t"

      /* mask |= (abs(p1 - p0) > limit)
       * hev  |= (abs(p1 - p0) > thresh)
       */
      "subu_s.qb      %[c],   %[p1],     %[p0]        \n\t"
      "subu_s.qb      %[r_k], %[p0],     %[p1]        \n\t"
      "or             %[r_k], %[r_k],    %[c]         \n\t"
      "cmpgu.lt.qb    %[c],   %[thresh], %[r_k]       \n\t"
      "or             %[r3],  $0,        %[c]         \n\t"
      "cmpgu.lt.qb    %[c],   %[limit],  %[r_k]       \n\t"
      "or             %[r],   %[r],      %[c]         \n\t"

      /* mask |= (abs(q1 - q0) > limit)
       * hev  |= (abs(q1 - q0) > thresh)
       */
      "subu_s.qb      %[c],   %[q1],     %[q0]        \n\t"
      "subu_s.qb      %[r_k], %[q0],     %[q1]        \n\t"
      "or             %[r_k], %[r_k],    %[c]         \n\t"
      "cmpgu.lt.qb    %[c],   %[thresh], %[r_k]       \n\t"
      "or             %[r3],  %[r3],     %[c]         \n\t"
      "cmpgu.lt.qb    %[c],   %[limit],  %[r_k]       \n\t"
      "or             %[r],   %[r],      %[c]         \n\t"

      /* mask |= (abs(q2 - q1) > limit) */
      "subu_s.qb      %[c],   %[q2],     %[q1]        \n\t"
      "subu_s.qb      %[r_k], %[q1],     %[q2]        \n\t"
      "or             %[r_k], %[r_k],    %[c]         \n\t"
      "cmpgu.lt.qb    %[c],   %[limit],  %[r_k]       \n\t"
      "or             %[r],   %[r],      %[c]         \n\t"
      "sll            %[r3],    %[r3],    24          \n\t"

      /* mask |= (abs(q3 - q2) > limit) */
      "subu_s.qb      %[c],   %[q3],     %[q2]        \n\t"
      "subu_s.qb      %[r_k], %[q2],     %[q3]        \n\t"
      "or             %[r_k], %[r_k],    %[c]         \n\t"
      "cmpgu.lt.qb    %[c],   %[limit],  %[r_k]       \n\t"
      "or             %[r],   %[r],      %[c]         \n\t"

      : [c] "=&r"(c), [r_k] "=&r"(r_k), [r] "=&r"(r), [r3] "=&r"(r3)
      : [limit] "r"(limit), [p3] "r"(p3), [p2] "r"(p2), [p1] "r"(p1),
        [p0] "r"(p0), [q1] "r"(q1), [q0] "r"(q0), [q2] "r"(q2), [q3] "r"(q3),
        [thresh] "r"(thresh));

  __asm__ __volatile__(
      /* abs(p0 - q0) */
      "subu_s.qb      %[c],   %[p0],     %[q0]        \n\t"
      "subu_s.qb      %[r_k], %[q0],     %[p0]        \n\t"
      "wrdsp          %[r3]                           \n\t"
      "or             %[s1],  %[r_k],    %[c]         \n\t"

      /* abs(p1 - q1) */
      "subu_s.qb      %[c],    %[p1],    %[q1]        \n\t"
      "addu_s.qb      %[s3],   %[s1],    %[s1]        \n\t"
      "pick.qb        %[hev1], %[ones],  $0           \n\t"
      "subu_s.qb      %[r_k],  %[q1],    %[p1]        \n\t"
      "or             %[s2],   %[r_k],   %[c]         \n\t"

      /* abs(p0 - q0) * 2 + abs(p1 - q1) / 2  > flimit * 2 + limit */
      "shrl.qb        %[s2],   %[s2],     1           \n\t"
      "addu_s.qb      %[s1],   %[s2],     %[s3]       \n\t"
      "cmpgu.lt.qb    %[c],    %[flimit], %[s1]       \n\t"
      "or             %[r],    %[r],      %[c]        \n\t"
      "sll            %[r],    %[r],      24          \n\t"

      "wrdsp          %[r]                            \n\t"
      "pick.qb        %[s2],  $0,         %[ones]     \n\t"

      : [c] "=&r"(c), [r_k] "=&r"(r_k), [s1] "=&r"(s1), [hev1] "=&r"(hev1),
        [s2] "=&r"(s2), [r] "+r"(r), [s3] "=&r"(s3)
      : [p0] "r"(p0), [q0] "r"(q0), [p1] "r"(p1), [r3] "r"(r3), [q1] "r"(q1),
        [ones] "r"(ones), [flimit] "r"(flimit));

  *hev = hev1;
  *mask = s2;
}

/* inputs & outputs are quad-byte vectors */
static __inline void vp8_filter_mips(uint32_t mask, uint32_t hev, uint32_t *ps1,
                                     uint32_t *ps0, uint32_t *qs0,
                                     uint32_t *qs1) {
  int32_t vp8_filter_l, vp8_filter_r;
  int32_t Filter1_l, Filter1_r, Filter2_l, Filter2_r;
  int32_t subr_r, subr_l;
  uint32_t t1, t2, HWM, t3;
  uint32_t hev_l, hev_r, mask_l, mask_r, invhev_l, invhev_r;

  int32_t vps1, vps0, vqs0, vqs1;
  int32_t vps1_l, vps1_r, vps0_l, vps0_r, vqs0_l, vqs0_r, vqs1_l, vqs1_r;
  uint32_t N128;

  N128 = 0x80808080;
  t1 = 0x03000300;
  t2 = 0x04000400;
  t3 = 0x01000100;
  HWM = 0xFF00FF00;

  vps0 = (*ps0) ^ N128;
  vps1 = (*ps1) ^ N128;
  vqs0 = (*qs0) ^ N128;
  vqs1 = (*qs1) ^ N128;

  /* use halfword pairs instead quad-bytes because of accuracy */
  vps0_l = vps0 & HWM;
  vps0_r = vps0 << 8;
  vps0_r = vps0_r & HWM;

  vps1_l = vps1 & HWM;
  vps1_r = vps1 << 8;
  vps1_r = vps1_r & HWM;

  vqs0_l = vqs0 & HWM;
  vqs0_r = vqs0 << 8;
  vqs0_r = vqs0_r & HWM;

  vqs1_l = vqs1 & HWM;
  vqs1_r = vqs1 << 8;
  vqs1_r = vqs1_r & HWM;

  mask_l = mask & HWM;
  mask_r = mask << 8;
  mask_r = mask_r & HWM;

  hev_l = hev & HWM;
  hev_r = hev << 8;
  hev_r = hev_r & HWM;

  __asm__ __volatile__(
      /* vp8_filter = vp8_signed_char_clamp(ps1 - qs1); */
      "subq_s.ph    %[vp8_filter_l], %[vps1_l],       %[vqs1_l]       \n\t"
      "subq_s.ph    %[vp8_filter_r], %[vps1_r],       %[vqs1_r]       \n\t"

      /* qs0 - ps0 */
      "subq_s.ph    %[subr_l],       %[vqs0_l],       %[vps0_l]       \n\t"
      "subq_s.ph    %[subr_r],       %[vqs0_r],       %[vps0_r]       \n\t"

      /* vp8_filter &= hev; */
      "and          %[vp8_filter_l], %[vp8_filter_l], %[hev_l]        \n\t"
      "and          %[vp8_filter_r], %[vp8_filter_r], %[hev_r]        \n\t"

      /* vp8_filter = vp8_signed_char_clamp(vp8_filter + 3 * (qs0 - ps0)); */
      "addq_s.ph    %[vp8_filter_l], %[vp8_filter_l], %[subr_l]       \n\t"
      "addq_s.ph    %[vp8_filter_r], %[vp8_filter_r], %[subr_r]       \n\t"
      "xor          %[invhev_l],     %[hev_l],        %[HWM]          \n\t"
      "addq_s.ph    %[vp8_filter_l], %[vp8_filter_l], %[subr_l]       \n\t"
      "addq_s.ph    %[vp8_filter_r], %[vp8_filter_r], %[subr_r]       \n\t"
      "xor          %[invhev_r],     %[hev_r],        %[HWM]          \n\t"
      "addq_s.ph    %[vp8_filter_l], %[vp8_filter_l], %[subr_l]       \n\t"
      "addq_s.ph    %[vp8_filter_r], %[vp8_filter_r], %[subr_r]       \n\t"

      /* vp8_filter &= mask; */
      "and          %[vp8_filter_l], %[vp8_filter_l], %[mask_l]       \n\t"
      "and          %[vp8_filter_r], %[vp8_filter_r], %[mask_r]       \n\t"

      : [vp8_filter_l] "=&r"(vp8_filter_l), [vp8_filter_r] "=&r"(vp8_filter_r),
        [subr_l] "=&r"(subr_l), [subr_r] "=&r"(subr_r),
        [invhev_l] "=&r"(invhev_l), [invhev_r] "=&r"(invhev_r)

      : [vps0_l] "r"(vps0_l), [vps0_r] "r"(vps0_r), [vps1_l] "r"(vps1_l),
        [vps1_r] "r"(vps1_r), [vqs0_l] "r"(vqs0_l), [vqs0_r] "r"(vqs0_r),
        [vqs1_l] "r"(vqs1_l), [vqs1_r] "r"(vqs1_r), [mask_l] "r"(mask_l),
        [mask_r] "r"(mask_r), [hev_l] "r"(hev_l), [hev_r] "r"(hev_r),
        [HWM] "r"(HWM));

  /* save bottom 3 bits so that we round one side +4 and the other +3 */
  __asm__ __volatile__(
      /* Filter2 = vp8_signed_char_clamp(vp8_filter + 3) >>= 3; */
      "addq_s.ph    %[Filter1_l],    %[vp8_filter_l], %[t2]           \n\t"
      "addq_s.ph    %[Filter1_r],    %[vp8_filter_r], %[t2]           \n\t"

      /* Filter1 = vp8_signed_char_clamp(vp8_filter + 4) >>= 3; */
      "addq_s.ph    %[Filter2_l],    %[vp8_filter_l], %[t1]           \n\t"
      "addq_s.ph    %[Filter2_r],    %[vp8_filter_r], %[t1]           \n\t"
      "shra.ph      %[Filter1_r],    %[Filter1_r],    3               \n\t"
      "shra.ph      %[Filter1_l],    %[Filter1_l],    3               \n\t"

      "shra.ph      %[Filter2_l],    %[Filter2_l],    3               \n\t"
      "shra.ph      %[Filter2_r],    %[Filter2_r],    3               \n\t"

      "and          %[Filter1_l],    %[Filter1_l],    %[HWM]          \n\t"
      "and          %[Filter1_r],    %[Filter1_r],    %[HWM]          \n\t"

      /* vps0 = vp8_signed_char_clamp(ps0 + Filter2); */
      "addq_s.ph    %[vps0_l],       %[vps0_l],       %[Filter2_l]    \n\t"
      "addq_s.ph    %[vps0_r],       %[vps0_r],       %[Filter2_r]    \n\t"

      /* vqs0 = vp8_signed_char_clamp(qs0 - Filter1); */
      "subq_s.ph    %[vqs0_l],       %[vqs0_l],       %[Filter1_l]    \n\t"
      "subq_s.ph    %[vqs0_r],       %[vqs0_r],       %[Filter1_r]    \n\t"

      : [Filter1_l] "=&r"(Filter1_l), [Filter1_r] "=&r"(Filter1_r),
        [Filter2_l] "=&r"(Filter2_l), [Filter2_r] "=&r"(Filter2_r),
        [vps0_l] "+r"(vps0_l), [vps0_r] "+r"(vps0_r), [vqs0_l] "+r"(vqs0_l),
        [vqs0_r] "+r"(vqs0_r)

      : [t1] "r"(t1), [t2] "r"(t2), [vp8_filter_l] "r"(vp8_filter_l),
        [vp8_filter_r] "r"(vp8_filter_r), [HWM] "r"(HWM));

  __asm__ __volatile__(
      /* (vp8_filter += 1) >>= 1 */
      "addqh.ph    %[Filter1_l],    %[Filter1_l],     %[t3]           \n\t"
      "addqh.ph    %[Filter1_r],    %[Filter1_r],     %[t3]           \n\t"

      /* vp8_filter &= ~hev; */
      "and          %[Filter1_l],    %[Filter1_l],    %[invhev_l]     \n\t"
      "and          %[Filter1_r],    %[Filter1_r],    %[invhev_r]     \n\t"

      /* vps1 = vp8_signed_char_clamp(ps1 + vp8_filter); */
      "addq_s.ph    %[vps1_l],       %[vps1_l],       %[Filter1_l]    \n\t"
      "addq_s.ph    %[vps1_r],       %[vps1_r],       %[Filter1_r]    \n\t"

      /* vqs1 = vp8_signed_char_clamp(qs1 - vp8_filter); */
      "subq_s.ph    %[vqs1_l],       %[vqs1_l],       %[Filter1_l]    \n\t"
      "subq_s.ph    %[vqs1_r],       %[vqs1_r],       %[Filter1_r]    \n\t"

      : [Filter1_l] "+r"(Filter1_l), [Filter1_r] "+r"(Filter1_r),
        [vps1_l] "+r"(vps1_l), [vps1_r] "+r"(vps1_r), [vqs1_l] "+r"(vqs1_l),
        [vqs1_r] "+r"(vqs1_r)

      : [t3] "r"(t3), [invhev_l] "r"(invhev_l), [invhev_r] "r"(invhev_r));

  /* Create quad-bytes from halfword pairs */
  vqs0_l = vqs0_l & HWM;
  vqs1_l = vqs1_l & HWM;
  vps0_l = vps0_l & HWM;
  vps1_l = vps1_l & HWM;

  __asm__ __volatile__(
      "shrl.ph      %[vqs0_r],       %[vqs0_r],       8               \n\t"
      "shrl.ph      %[vps0_r],       %[vps0_r],       8               \n\t"
      "shrl.ph      %[vqs1_r],       %[vqs1_r],       8               \n\t"
      "shrl.ph      %[vps1_r],       %[vps1_r],       8               \n\t"

      : [vps1_r] "+r"(vps1_r), [vqs1_r] "+r"(vqs1_r), [vps0_r] "+r"(vps0_r),
        [vqs0_r] "+r"(vqs0_r)
      :);

  vqs0 = vqs0_l | vqs0_r;
  vqs1 = vqs1_l | vqs1_r;
  vps0 = vps0_l | vps0_r;
  vps1 = vps1_l | vps1_r;

  *ps0 = vps0 ^ N128;
  *ps1 = vps1 ^ N128;
  *qs0 = vqs0 ^ N128;
  *qs1 = vqs1 ^ N128;
}

void vp8_loop_filter_horizontal_edge_mips(unsigned char *s, int p,
                                          unsigned int flimit,
                                          unsigned int limit,
                                          unsigned int thresh, int count) {
  uint32_t mask;
  uint32_t hev;
  uint32_t pm1, p0, p1, p2, p3, p4, p5, p6;
  unsigned char *sm1, *s0, *s1, *s2, *s3, *s4, *s5, *s6;
  (void)count;

  mask = 0;
  hev = 0;
  p1 = 0;
  p2 = 0;
  p3 = 0;
  p4 = 0;

  /* prefetch data for store */
  prefetch_store_lf(s);

  /* loop filter designed to work using chars so that we can make maximum use
   * of 8 bit simd instructions.
   */

  sm1 = s - (p << 2);
  s0 = s - p - p - p;
  s1 = s - p - p;
  s2 = s - p;
  s3 = s;
  s4 = s + p;
  s5 = s + p + p;
  s6 = s + p + p + p;

  /* load quad-byte vectors
   * memory is 4 byte aligned
   */
  p1 = *((uint32_t *)(s1));
  p2 = *((uint32_t *)(s2));
  p3 = *((uint32_t *)(s3));
  p4 = *((uint32_t *)(s4));

  /* if (p1 - p4 == 0) and (p2 - p3 == 0)
   * mask will be zero and filtering is not needed
   */
  if (!(((p1 - p4) == 0) && ((p2 - p3) == 0))) {
    pm1 = *((uint32_t *)(sm1));
    p0 = *((uint32_t *)(s0));
    p5 = *((uint32_t *)(s5));
    p6 = *((uint32_t *)(s6));

    vp8_filter_mask_vec_mips(limit, flimit, p1, p2, pm1, p0, p3, p4, p5, p6,
                             thresh, &hev, &mask);

    /* if mask == 0 do filtering is not needed */
    if (mask) {
      /* filtering */
      vp8_filter_mips(mask, hev, &p1, &p2, &p3, &p4);

      /* unpack processed 4x4 neighborhood */
      *((uint32_t *)s1) = p1;
      *((uint32_t *)s2) = p2;
      *((uint32_t *)s3) = p3;
      *((uint32_t *)s4) = p4;
    }
  }

  sm1 += 4;
  s0 += 4;
  s1 += 4;
  s2 += 4;
  s3 += 4;
  s4 += 4;
  s5 += 4;
  s6 += 4;

  /* load quad-byte vectors
   * memory is 4 byte aligned
   */
  p1 = *((uint32_t *)(s1));
  p2 = *((uint32_t *)(s2));
  p3 = *((uint32_t *)(s3));
  p4 = *((uint32_t *)(s4));

  /* if (p1 - p4 == 0) and (p2 - p3 == 0)
   * mask will be zero and filtering is not needed
   */
  if (!(((p1 - p4) == 0) && ((p2 - p3) == 0))) {
    pm1 = *((uint32_t *)(sm1));
    p0 = *((uint32_t *)(s0));
    p5 = *((uint32_t *)(s5));
    p6 = *((uint32_t *)(s6));

    vp8_filter_mask_vec_mips(limit, flimit, p1, p2, pm1, p0, p3, p4, p5, p6,
                             thresh, &hev, &mask);

    /* if mask == 0 do filtering is not needed */
    if (mask) {
      /* filtering */
      vp8_filter_mips(mask, hev, &p1, &p2, &p3, &p4);

      /* unpack processed 4x4 neighborhood */
      *((uint32_t *)s1) = p1;
      *((uint32_t *)s2) = p2;
      *((uint32_t *)s3) = p3;
      *((uint32_t *)s4) = p4;
    }
  }

  sm1 += 4;
  s0 += 4;
  s1 += 4;
  s2 += 4;
  s3 += 4;
  s4 += 4;
  s5 += 4;
  s6 += 4;

  /* load quad-byte vectors
   * memory is 4 byte aligned
   */
  p1 = *((uint32_t *)(s1));
  p2 = *((uint32_t *)(s2));
  p3 = *((uint32_t *)(s3));
  p4 = *((uint32_t *)(s4));

  /* if (p1 - p4 == 0) and (p2 - p3 == 0)
   * mask will be zero and filtering is not needed
   */
  if (!(((p1 - p4) == 0) && ((p2 - p3) == 0))) {
    pm1 = *((uint32_t *)(sm1));
    p0 = *((uint32_t *)(s0));
    p5 = *((uint32_t *)(s5));
    p6 = *((uint32_t *)(s6));

    vp8_filter_mask_vec_mips(limit, flimit, p1, p2, pm1, p0, p3, p4, p5, p6,
                             thresh, &hev, &mask);

    /* if mask == 0 do filtering is not needed */
    if (mask) {
      /* filtering */
      vp8_filter_mips(mask, hev, &p1, &p2, &p3, &p4);

      /* unpack processed 4x4 neighborhood */
      *((uint32_t *)s1) = p1;
      *((uint32_t *)s2) = p2;
      *((uint32_t *)s3) = p3;
      *((uint32_t *)s4) = p4;
    }
  }

  sm1 += 4;
  s0 += 4;
  s1 += 4;
  s2 += 4;
  s3 += 4;
  s4 += 4;
  s5 += 4;
  s6 += 4;

  /* load quad-byte vectors
   * memory is 4 byte aligned
   */
  p1 = *((uint32_t *)(s1));
  p2 = *((uint32_t *)(s2));
  p3 = *((uint32_t *)(s3));
  p4 = *((uint32_t *)(s4));

  /* if (p1 - p4 == 0) and (p2 - p3 == 0)
   * mask will be zero and filtering is not needed
   */
  if (!(((p1 - p4) == 0) && ((p2 - p3) == 0))) {
    pm1 = *((uint32_t *)(sm1));
    p0 = *((uint32_t *)(s0));
    p5 = *((uint32_t *)(s5));
    p6 = *((uint32_t *)(s6));

    vp8_filter_mask_vec_mips(limit, flimit, p1, p2, pm1, p0, p3, p4, p5, p6,
                             thresh, &hev, &mask);

    /* if mask == 0 do filtering is not needed */
    if (mask) {
      /* filtering */
      vp8_filter_mips(mask, hev, &p1, &p2, &p3, &p4);

      /* unpack processed 4x4 neighborhood */
      *((uint32_t *)s1) = p1;
      *((uint32_t *)s2) = p2;
      *((uint32_t *)s3) = p3;
      *((uint32_t *)s4) = p4;
    }
  }
}

void vp8_loop_filter_uvhorizontal_edge_mips(unsigned char *s, int p,
                                            unsigned int flimit,
                                            unsigned int limit,
                                            unsigned int thresh, int count) {
  uint32_t mask;
  uint32_t hev;
  uint32_t pm1, p0, p1, p2, p3, p4, p5, p6;
  unsigned char *sm1, *s0, *s1, *s2, *s3, *s4, *s5, *s6;
  (void)count;

  mask = 0;
  hev = 0;
  p1 = 0;
  p2 = 0;
  p3 = 0;
  p4 = 0;

  /* loop filter designed to work using chars so that we can make maximum use
   * of 8 bit simd instructions.
   */

  sm1 = s - (p << 2);
  s0 = s - p - p - p;
  s1 = s - p - p;
  s2 = s - p;
  s3 = s;
  s4 = s + p;
  s5 = s + p + p;
  s6 = s + p + p + p;

  /* load quad-byte vectors
   * memory is 4 byte aligned
   */
  p1 = *((uint32_t *)(s1));
  p2 = *((uint32_t *)(s2));
  p3 = *((uint32_t *)(s3));
  p4 = *((uint32_t *)(s4));

  /* if (p1 - p4 == 0) and (p2 - p3 == 0)
   * mask will be zero and filtering is not needed
   */
  if (!(((p1 - p4) == 0) && ((p2 - p3) == 0))) {
    pm1 = *((uint32_t *)(sm1));
    p0 = *((uint32_t *)(s0));
    p5 = *((uint32_t *)(s5));
    p6 = *((uint32_t *)(s6));

    vp8_filter_mask_vec_mips(limit, flimit, p1, p2, pm1, p0, p3, p4, p5, p6,
                             thresh, &hev, &mask);

    /* if mask == 0 do filtering is not needed */
    if (mask) {
      /* filtering */
      vp8_filter_mips(mask, hev, &p1, &p2, &p3, &p4);

      /* unpack processed 4x4 neighborhood */
      *((uint32_t *)s1) = p1;
      *((uint32_t *)s2) = p2;
      *((uint32_t *)s3) = p3;
      *((uint32_t *)s4) = p4;
    }
  }

  sm1 += 4;
  s0 += 4;
  s1 += 4;
  s2 += 4;
  s3 += 4;
  s4 += 4;
  s5 += 4;
  s6 += 4;

  /* load quad-byte vectors
   * memory is 4 byte aligned
   */
  p1 = *((uint32_t *)(s1));
  p2 = *((uint32_t *)(s2));
  p3 = *((uint32_t *)(s3));
  p4 = *((uint32_t *)(s4));

  /* if (p1 - p4 == 0) and (p2 - p3 == 0)
   * mask will be zero and filtering is not needed
   */
  if (!(((p1 - p4) == 0) && ((p2 - p3) == 0))) {
    pm1 = *((uint32_t *)(sm1));
    p0 = *((uint32_t *)(s0));
    p5 = *((uint32_t *)(s5));
    p6 = *((uint32_t *)(s6));

    vp8_filter_mask_vec_mips(limit, flimit, p1, p2, pm1, p0, p3, p4, p5, p6,
                             thresh, &hev, &mask);

    /* if mask == 0 do filtering is not needed */
    if (mask) {
      /* filtering */
      vp8_filter_mips(mask, hev, &p1, &p2, &p3, &p4);

      /* unpack processed 4x4 neighborhood */
      *((uint32_t *)s1) = p1;
      *((uint32_t *)s2) = p2;
      *((uint32_t *)s3) = p3;
      *((uint32_t *)s4) = p4;
    }
  }
}

void vp8_loop_filter_vertical_edge_mips(unsigned char *s, int p,
                                        const unsigned int flimit,
                                        const unsigned int limit,
                                        const unsigned int thresh, int count) {
  int i;
  uint32_t mask, hev;
  uint32_t pm1, p0, p1, p2, p3, p4, p5, p6;
  unsigned char *s1, *s2, *s3, *s4;
  uint32_t prim1, prim2, sec3, sec4, prim3, prim4;

  hev = 0;
  mask = 0;
  i = 0;
  pm1 = 0;
  p0 = 0;
  p1 = 0;
  p2 = 0;
  p3 = 0;
  p4 = 0;
  p5 = 0;
  p6 = 0;

  /* loop filter designed to work using chars so that we can make maximum use
   * of 8 bit simd instructions.
   */

  /* apply filter on 4 pixesl at the same time */
  do {
    /* prefetch data for store */
    prefetch_store_lf(s + p);

    s1 = s;
    s2 = s + p;
    s3 = s2 + p;
    s4 = s3 + p;
    s = s4 + p;

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
      vp8_filter_mask_vec_mips(limit, flimit, p1, p2, pm1, p0, p3, p4, p5, p6,
                               thresh, &hev, &mask);

      /* if mask == 0 do filtering is not needed */
      if (mask) {
        /* filtering */
        vp8_filter_mips(mask, hev, &p1, &p2, &p3, &p4);

        /* unpack processed 4x4 neighborhood
         * don't use transpose on output data
         * because memory isn't aligned
         */
        __asm__ __volatile__(
            "sb         %[p4],  1(%[s4])    \n\t"
            "sb         %[p3],  0(%[s4])    \n\t"
            "sb         %[p2], -1(%[s4])    \n\t"
            "sb         %[p1], -2(%[s4])    \n\t"
            :
            : [p4] "r"(p4), [p3] "r"(p3), [s4] "r"(s4), [p2] "r"(p2),
              [p1] "r"(p1));

        __asm__ __volatile__(
            "srl        %[p4], %[p4], 8     \n\t"
            "srl        %[p3], %[p3], 8     \n\t"
            "srl        %[p2], %[p2], 8     \n\t"
            "srl        %[p1], %[p1], 8     \n\t"
            : [p4] "+r"(p4), [p3] "+r"(p3), [p2] "+r"(p2), [p1] "+r"(p1)
            :);

        __asm__ __volatile__(
            "sb         %[p4],  1(%[s3])    \n\t"
            "sb         %[p3],  0(%[s3])    \n\t"
            "sb         %[p2], -1(%[s3])    \n\t"
            "sb         %[p1], -2(%[s3])    \n\t"
            : [p1] "+r"(p1)
            : [p4] "r"(p4), [p3] "r"(p3), [s3] "r"(s3), [p2] "r"(p2));

        __asm__ __volatile__(
            "srl        %[p4], %[p4], 8     \n\t"
            "srl        %[p3], %[p3], 8     \n\t"
            "srl        %[p2], %[p2], 8     \n\t"
            "srl        %[p1], %[p1], 8     \n\t"
            : [p4] "+r"(p4), [p3] "+r"(p3), [p2] "+r"(p2), [p1] "+r"(p1)
            :);

        __asm__ __volatile__(
            "sb         %[p4],  1(%[s2])    \n\t"
            "sb         %[p3],  0(%[s2])    \n\t"
            "sb         %[p2], -1(%[s2])    \n\t"
            "sb         %[p1], -2(%[s2])    \n\t"
            :
            : [p4] "r"(p4), [p3] "r"(p3), [s2] "r"(s2), [p2] "r"(p2),
              [p1] "r"(p1));

        __asm__ __volatile__(
            "srl        %[p4], %[p4], 8     \n\t"
            "srl        %[p3], %[p3], 8     \n\t"
            "srl        %[p2], %[p2], 8     \n\t"
            "srl        %[p1], %[p1], 8     \n\t"
            : [p4] "+r"(p4), [p3] "+r"(p3), [p2] "+r"(p2), [p1] "+r"(p1)
            :);

        __asm__ __volatile__(
            "sb         %[p4],  1(%[s1])    \n\t"
            "sb         %[p3],  0(%[s1])    \n\t"
            "sb         %[p2], -1(%[s1])    \n\t"
            "sb         %[p1], -2(%[s1])    \n\t"
            :
            : [p4] "r"(p4), [p3] "r"(p3), [s1] "r"(s1), [p2] "r"(p2),
              [p1] "r"(p1));
      }
    }

    s1 = s;
    s2 = s + p;
    s3 = s2 + p;
    s4 = s3 + p;
    s = s4 + p;

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
      vp8_filter_mask_vec_mips(limit, flimit, p1, p2, pm1, p0, p3, p4, p5, p6,
                               thresh, &hev, &mask);

      /* if mask == 0 do filtering is not needed */
      if (mask) {
        /* filtering */
        vp8_filter_mips(mask, hev, &p1, &p2, &p3, &p4);

        /* unpack processed 4x4 neighborhood
         * don't use transpose on output data
         * because memory isn't aligned
         */
        __asm__ __volatile__(
            "sb         %[p4],  1(%[s4])    \n\t"
            "sb         %[p3],  0(%[s4])    \n\t"
            "sb         %[p2], -1(%[s4])    \n\t"
            "sb         %[p1], -2(%[s4])    \n\t"
            :
            : [p4] "r"(p4), [p3] "r"(p3), [s4] "r"(s4), [p2] "r"(p2),
              [p1] "r"(p1));

        __asm__ __volatile__(
            "srl        %[p4], %[p4], 8     \n\t"
            "srl        %[p3], %[p3], 8     \n\t"
            "srl        %[p2], %[p2], 8     \n\t"
            "srl        %[p1], %[p1], 8     \n\t"
            : [p4] "+r"(p4), [p3] "+r"(p3), [p2] "+r"(p2), [p1] "+r"(p1)
            :);

        __asm__ __volatile__(
            "sb         %[p4],  1(%[s3])    \n\t"
            "sb         %[p3],  0(%[s3])    \n\t"
            "sb         %[p2], -1(%[s3])    \n\t"
            "sb         %[p1], -2(%[s3])    \n\t"
            : [p1] "+r"(p1)
            : [p4] "r"(p4), [p3] "r"(p3), [s3] "r"(s3), [p2] "r"(p2));

        __asm__ __volatile__(
            "srl        %[p4], %[p4], 8     \n\t"
            "srl        %[p3], %[p3], 8     \n\t"
            "srl        %[p2], %[p2], 8     \n\t"
            "srl        %[p1], %[p1], 8     \n\t"
            : [p4] "+r"(p4), [p3] "+r"(p3), [p2] "+r"(p2), [p1] "+r"(p1)
            :);

        __asm__ __volatile__(
            "sb         %[p4],  1(%[s2])    \n\t"
            "sb         %[p3],  0(%[s2])    \n\t"
            "sb         %[p2], -1(%[s2])    \n\t"
            "sb         %[p1], -2(%[s2])    \n\t"
            :
            : [p4] "r"(p4), [p3] "r"(p3), [s2] "r"(s2), [p2] "r"(p2),
              [p1] "r"(p1));

        __asm__ __volatile__(
            "srl        %[p4], %[p4], 8     \n\t"
            "srl        %[p3], %[p3], 8     \n\t"
            "srl        %[p2], %[p2], 8     \n\t"
            "srl        %[p1], %[p1], 8     \n\t"
            : [p4] "+r"(p4), [p3] "+r"(p3), [p2] "+r"(p2), [p1] "+r"(p1)
            :);

        __asm__ __volatile__(
            "sb         %[p4],  1(%[s1])    \n\t"
            "sb         %[p3],  0(%[s1])    \n\t"
            "sb         %[p2], -1(%[s1])    \n\t"
            "sb         %[p1], -2(%[s1])    \n\t"
            :
            : [p4] "r"(p4), [p3] "r"(p3), [s1] "r"(s1), [p2] "r"(p2),
              [p1] "r"(p1));
      }
    }

    i += 8;
  }

  while (i < count);
}

void vp8_loop_filter_uvvertical_edge_mips(unsigned char *s, int p,
                                          unsigned int flimit,
                                          unsigned int limit,
                                          unsigned int thresh, int count) {
  uint32_t mask, hev;
  uint32_t pm1, p0, p1, p2, p3, p4, p5, p6;
  unsigned char *s1, *s2, *s3, *s4;
  uint32_t prim1, prim2, sec3, sec4, prim3, prim4;
  (void)count;

  /* loop filter designed to work using chars so that we can make maximum use
   * of 8 bit simd instructions.
   */

  /* apply filter on 4 pixesl at the same time */

  s1 = s;
  s2 = s + p;
  s3 = s2 + p;
  s4 = s3 + p;

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
    vp8_filter_mask_vec_mips(limit, flimit, p1, p2, pm1, p0, p3, p4, p5, p6,
                             thresh, &hev, &mask);

    /* if mask == 0 do filtering is not needed */
    if (mask) {
      /* filtering */
      vp8_filter_mips(mask, hev, &p1, &p2, &p3, &p4);

      /* unpack processed 4x4 neighborhood
       * don't use transpose on output data
       * because memory isn't aligned
       */
      __asm__ __volatile__(
          "sb         %[p4],  1(%[s4])    \n\t"
          "sb         %[p3],  0(%[s4])    \n\t"
          "sb         %[p2], -1(%[s4])    \n\t"
          "sb         %[p1], -2(%[s4])    \n\t"
          :
          :
          [p4] "r"(p4), [p3] "r"(p3), [s4] "r"(s4), [p2] "r"(p2), [p1] "r"(p1));

      __asm__ __volatile__(
          "srl        %[p4], %[p4], 8     \n\t"
          "srl        %[p3], %[p3], 8     \n\t"
          "srl        %[p2], %[p2], 8     \n\t"
          "srl        %[p1], %[p1], 8     \n\t"
          : [p4] "+r"(p4), [p3] "+r"(p3), [p2] "+r"(p2), [p1] "+r"(p1)
          :);

      __asm__ __volatile__(
          "sb         %[p4],  1(%[s3])    \n\t"
          "sb         %[p3],  0(%[s3])    \n\t"
          "sb         %[p2], -1(%[s3])    \n\t"
          "sb         %[p1], -2(%[s3])    \n\t"
          : [p1] "+r"(p1)
          : [p4] "r"(p4), [p3] "r"(p3), [s3] "r"(s3), [p2] "r"(p2));

      __asm__ __volatile__(
          "srl        %[p4], %[p4], 8     \n\t"
          "srl        %[p3], %[p3], 8     \n\t"
          "srl        %[p2], %[p2], 8     \n\t"
          "srl        %[p1], %[p1], 8     \n\t"
          : [p4] "+r"(p4), [p3] "+r"(p3), [p2] "+r"(p2), [p1] "+r"(p1)
          :);

      __asm__ __volatile__(
          "sb         %[p4],  1(%[s2])    \n\t"
          "sb         %[p3],  0(%[s2])    \n\t"
          "sb         %[p2], -1(%[s2])    \n\t"
          "sb         %[p1], -2(%[s2])    \n\t"
          :
          :
          [p4] "r"(p4), [p3] "r"(p3), [s2] "r"(s2), [p2] "r"(p2), [p1] "r"(p1));

      __asm__ __volatile__(
          "srl        %[p4], %[p4], 8     \n\t"
          "srl        %[p3], %[p3], 8     \n\t"
          "srl        %[p2], %[p2], 8     \n\t"
          "srl        %[p1], %[p1], 8     \n\t"
          : [p4] "+r"(p4), [p3] "+r"(p3), [p2] "+r"(p2), [p1] "+r"(p1)
          :);

      __asm__ __volatile__(
          "sb         %[p4],  1(%[s1])    \n\t"
          "sb         %[p3],  0(%[s1])    \n\t"
          "sb         %[p2], -1(%[s1])    \n\t"
          "sb         %[p1], -2(%[s1])    \n\t"
          :
          :
          [p4] "r"(p4), [p3] "r"(p3), [s1] "r"(s1), [p2] "r"(p2), [p1] "r"(p1));
    }
  }

  s1 = s4 + p;
  s2 = s1 + p;
  s3 = s2 + p;
  s4 = s3 + p;

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
    vp8_filter_mask_vec_mips(limit, flimit, p1, p2, pm1, p0, p3, p4, p5, p6,
                             thresh, &hev, &mask);

    /* if mask == 0 do filtering is not needed */
    if (mask) {
      /* filtering */
      vp8_filter_mips(mask, hev, &p1, &p2, &p3, &p4);

      /* unpack processed 4x4 neighborhood
       * don't use transpose on output data
       * because memory isn't aligned
       */
      __asm__ __volatile__(
          "sb         %[p4],  1(%[s4])    \n\t"
          "sb         %[p3],  0(%[s4])    \n\t"
          "sb         %[p2], -1(%[s4])    \n\t"
          "sb         %[p1], -2(%[s4])    \n\t"
          :
          :
          [p4] "r"(p4), [p3] "r"(p3), [s4] "r"(s4), [p2] "r"(p2), [p1] "r"(p1));

      __asm__ __volatile__(
          "srl        %[p4], %[p4], 8     \n\t"
          "srl        %[p3], %[p3], 8     \n\t"
          "srl        %[p2], %[p2], 8     \n\t"
          "srl        %[p1], %[p1], 8     \n\t"
          : [p4] "+r"(p4), [p3] "+r"(p3), [p2] "+r"(p2), [p1] "+r"(p1)
          :);

      __asm__ __volatile__(
          "sb         %[p4],  1(%[s3])    \n\t"
          "sb         %[p3],  0(%[s3])    \n\t"
          "sb         %[p2], -1(%[s3])    \n\t"
          "sb         %[p1], -2(%[s3])    \n\t"
          : [p1] "+r"(p1)
          : [p4] "r"(p4), [p3] "r"(p3), [s3] "r"(s3), [p2] "r"(p2));

      __asm__ __volatile__(
          "srl        %[p4], %[p4], 8     \n\t"
          "srl        %[p3], %[p3], 8     \n\t"
          "srl        %[p2], %[p2], 8     \n\t"
          "srl        %[p1], %[p1], 8     \n\t"
          : [p4] "+r"(p4), [p3] "+r"(p3), [p2] "+r"(p2), [p1] "+r"(p1)
          :);

      __asm__ __volatile__(
          "sb         %[p4],  1(%[s2])    \n\t"
          "sb         %[p3],  0(%[s2])    \n\t"
          "sb         %[p2], -1(%[s2])    \n\t"
          "sb         %[p1], -2(%[s2])    \n\t"
          :
          :
          [p4] "r"(p4), [p3] "r"(p3), [s2] "r"(s2), [p2] "r"(p2), [p1] "r"(p1));

      __asm__ __volatile__(
          "srl        %[p4], %[p4], 8     \n\t"
          "srl        %[p3], %[p3], 8     \n\t"
          "srl        %[p2], %[p2], 8     \n\t"
          "srl        %[p1], %[p1], 8     \n\t"
          : [p4] "+r"(p4), [p3] "+r"(p3), [p2] "+r"(p2), [p1] "+r"(p1)
          :);

      __asm__ __volatile__(
          "sb         %[p4],  1(%[s1])    \n\t"
          "sb         %[p3],  0(%[s1])    \n\t"
          "sb         %[p2], -1(%[s1])    \n\t"
          "sb         %[p1], -2(%[s1])    \n\t"
          :
          :
          [p4] "r"(p4), [p3] "r"(p3), [s1] "r"(s1), [p2] "r"(p2), [p1] "r"(p1));
    }
  }
}

/* inputs & outputs are quad-byte vectors */
static __inline void vp8_mbfilter_mips(uint32_t mask, uint32_t hev,
                                       uint32_t *ps2, uint32_t *ps1,
                                       uint32_t *ps0, uint32_t *qs0,
                                       uint32_t *qs1, uint32_t *qs2) {
  int32_t vps2, vps1, vps0, vqs0, vqs1, vqs2;
  int32_t vps2_l, vps1_l, vps0_l, vqs0_l, vqs1_l, vqs2_l;
  int32_t vps2_r, vps1_r, vps0_r, vqs0_r, vqs1_r, vqs2_r;
  uint32_t HWM, vp8_filter_l, vp8_filter_r, mask_l, mask_r, hev_l, hev_r,
      subr_r, subr_l;
  uint32_t Filter2_l, Filter2_r, t1, t2, Filter1_l, Filter1_r, invhev_l,
      invhev_r;
  uint32_t N128, R63;
  uint32_t u1_l, u1_r, u2_l, u2_r, u3_l, u3_r;

  R63 = 0x003F003F;
  HWM = 0xFF00FF00;
  N128 = 0x80808080;
  t1 = 0x03000300;
  t2 = 0x04000400;

  vps0 = (*ps0) ^ N128;
  vps1 = (*ps1) ^ N128;
  vps2 = (*ps2) ^ N128;
  vqs0 = (*qs0) ^ N128;
  vqs1 = (*qs1) ^ N128;
  vqs2 = (*qs2) ^ N128;

  /* use halfword pairs instead quad-bytes because of accuracy */
  vps0_l = vps0 & HWM;
  vps0_r = vps0 << 8;
  vps0_r = vps0_r & HWM;

  vqs0_l = vqs0 & HWM;
  vqs0_r = vqs0 << 8;
  vqs0_r = vqs0_r & HWM;

  vps1_l = vps1 & HWM;
  vps1_r = vps1 << 8;
  vps1_r = vps1_r & HWM;

  vqs1_l = vqs1 & HWM;
  vqs1_r = vqs1 << 8;
  vqs1_r = vqs1_r & HWM;

  vqs2_l = vqs2 & HWM;
  vqs2_r = vqs2 << 8;
  vqs2_r = vqs2_r & HWM;

  __asm__ __volatile__(
      /* qs0 - ps0 */
      "subq_s.ph    %[subr_l],       %[vqs0_l],       %[vps0_l]       \n\t"
      "subq_s.ph    %[subr_r],       %[vqs0_r],       %[vps0_r]       \n\t"

      /* vp8_filter = vp8_signed_char_clamp(ps1 - qs1); */
      "subq_s.ph    %[vp8_filter_l], %[vps1_l],       %[vqs1_l]       \n\t"
      "subq_s.ph    %[vp8_filter_r], %[vps1_r],       %[vqs1_r]       \n\t"

      : [vp8_filter_l] "=&r"(vp8_filter_l), [vp8_filter_r] "=r"(vp8_filter_r),
        [subr_l] "=&r"(subr_l), [subr_r] "=&r"(subr_r)
      : [vps0_l] "r"(vps0_l), [vps0_r] "r"(vps0_r), [vps1_l] "r"(vps1_l),
        [vps1_r] "r"(vps1_r), [vqs0_l] "r"(vqs0_l), [vqs0_r] "r"(vqs0_r),
        [vqs1_l] "r"(vqs1_l), [vqs1_r] "r"(vqs1_r));

  vps2_l = vps2 & HWM;
  vps2_r = vps2 << 8;
  vps2_r = vps2_r & HWM;

  /* add outer taps if we have high edge variance */
  __asm__ __volatile__(
      /* vp8_filter = vp8_signed_char_clamp(vp8_filter + 3 * (qs0 - ps0)); */
      "addq_s.ph    %[vp8_filter_l], %[vp8_filter_l], %[subr_l]       \n\t"
      "addq_s.ph    %[vp8_filter_r], %[vp8_filter_r], %[subr_r]       \n\t"
      "and          %[mask_l],       %[HWM],          %[mask]         \n\t"
      "sll          %[mask_r],       %[mask],         8               \n\t"
      "and          %[mask_r],       %[HWM],          %[mask_r]       \n\t"
      "addq_s.ph    %[vp8_filter_l], %[vp8_filter_l], %[subr_l]       \n\t"
      "addq_s.ph    %[vp8_filter_r], %[vp8_filter_r], %[subr_r]       \n\t"
      "and          %[hev_l],        %[HWM],          %[hev]          \n\t"
      "sll          %[hev_r],        %[hev],          8               \n\t"
      "and          %[hev_r],        %[HWM],          %[hev_r]        \n\t"
      "addq_s.ph    %[vp8_filter_l], %[vp8_filter_l], %[subr_l]       \n\t"
      "addq_s.ph    %[vp8_filter_r], %[vp8_filter_r], %[subr_r]       \n\t"

      /* vp8_filter &= mask; */
      "and          %[vp8_filter_l], %[vp8_filter_l], %[mask_l]       \n\t"
      "and          %[vp8_filter_r], %[vp8_filter_r], %[mask_r]       \n\t"

      /* Filter2 = vp8_filter & hev; */
      "and          %[Filter2_l],    %[vp8_filter_l], %[hev_l]        \n\t"
      "and          %[Filter2_r],    %[vp8_filter_r], %[hev_r]        \n\t"

      : [vp8_filter_l] "+r"(vp8_filter_l), [vp8_filter_r] "+r"(vp8_filter_r),
        [hev_l] "=&r"(hev_l), [hev_r] "=&r"(hev_r), [mask_l] "=&r"(mask_l),
        [mask_r] "=&r"(mask_r), [Filter2_l] "=&r"(Filter2_l),
        [Filter2_r] "=&r"(Filter2_r)
      : [subr_l] "r"(subr_l), [subr_r] "r"(subr_r), [HWM] "r"(HWM),
        [hev] "r"(hev), [mask] "r"(mask));

  /* save bottom 3 bits so that we round one side +4 and the other +3 */
  __asm__ __volatile__(
      /* Filter1 = vp8_signed_char_clamp(Filter2 + 4) >>= 3; */
      "addq_s.ph    %[Filter1_l],    %[Filter2_l],    %[t2]           \n\t"
      "xor          %[invhev_l],     %[hev_l],        %[HWM]          \n\t"
      "addq_s.ph    %[Filter1_r],    %[Filter2_r],    %[t2]           \n\t"

      /* Filter2 = vp8_signed_char_clamp(Filter2 + 3) >>= 3; */
      "addq_s.ph    %[Filter2_l],    %[Filter2_l],    %[t1]           \n\t"
      "addq_s.ph    %[Filter2_r],    %[Filter2_r],    %[t1]           \n\t"

      "shra.ph      %[Filter1_l],    %[Filter1_l],    3               \n\t"
      "shra.ph      %[Filter1_r],    %[Filter1_r],    3               \n\t"

      "shra.ph      %[Filter2_l],    %[Filter2_l],    3               \n\t"
      "shra.ph      %[Filter2_r],    %[Filter2_r],    3               \n\t"
      "and          %[Filter1_l],    %[Filter1_l],    %[HWM]          \n\t"
      "and          %[Filter1_r],    %[Filter1_r],    %[HWM]          \n\t"
      "xor          %[invhev_r],     %[hev_r],        %[HWM]          \n\t"

      /* qs0 = vp8_signed_char_clamp(qs0 - Filter1); */
      "subq_s.ph    %[vqs0_l],       %[vqs0_l],       %[Filter1_l]    \n\t"
      "subq_s.ph    %[vqs0_r],       %[vqs0_r],       %[Filter1_r]    \n\t"

      /* ps0 = vp8_signed_char_clamp(ps0 + Filter2); */
      "addq_s.ph    %[vps0_l],       %[vps0_l],       %[Filter2_l]    \n\t"
      "addq_s.ph    %[vps0_r],       %[vps0_r],       %[Filter2_r]    \n\t"

      : [invhev_l] "=&r"(invhev_l), [invhev_r] "=&r"(invhev_r),
        [Filter1_l] "=&r"(Filter1_l), [Filter1_r] "=&r"(Filter1_r),
        [Filter2_l] "+r"(Filter2_l), [Filter2_r] "+r"(Filter2_r),
        [vps0_l] "+r"(vps0_l), [vps0_r] "+r"(vps0_r), [vqs0_l] "+r"(vqs0_l),
        [vqs0_r] "+r"(vqs0_r)
      : [t1] "r"(t1), [t2] "r"(t2), [HWM] "r"(HWM), [hev_l] "r"(hev_l),
        [hev_r] "r"(hev_r));

  /* only apply wider filter if not high edge variance */
  __asm__ __volatile__(
      /* vp8_filter &= ~hev; */
      "and          %[Filter2_l],    %[vp8_filter_l], %[invhev_l]     \n\t"
      "and          %[Filter2_r],    %[vp8_filter_r], %[invhev_r]     \n\t"

      "shra.ph      %[Filter2_l],    %[Filter2_l],    8               \n\t"
      "shra.ph      %[Filter2_r],    %[Filter2_r],    8               \n\t"

      : [Filter2_l] "=&r"(Filter2_l), [Filter2_r] "=&r"(Filter2_r)
      : [vp8_filter_l] "r"(vp8_filter_l), [vp8_filter_r] "r"(vp8_filter_r),
        [invhev_l] "r"(invhev_l), [invhev_r] "r"(invhev_r));

  /* roughly 3/7th difference across boundary */
  __asm__ __volatile__(
      "shll.ph      %[u3_l],         %[Filter2_l],    3               \n\t"
      "shll.ph      %[u3_r],         %[Filter2_r],    3               \n\t"

      "addq.ph      %[u3_l],         %[u3_l],         %[Filter2_l]    \n\t"
      "addq.ph      %[u3_r],         %[u3_r],         %[Filter2_r]    \n\t"

      "shll.ph      %[u2_l],         %[u3_l],         1               \n\t"
      "shll.ph      %[u2_r],         %[u3_r],         1               \n\t"

      "addq.ph      %[u1_l],         %[u3_l],         %[u2_l]         \n\t"
      "addq.ph      %[u1_r],         %[u3_r],         %[u2_r]         \n\t"

      "addq.ph      %[u2_l],         %[u2_l],         %[R63]          \n\t"
      "addq.ph      %[u2_r],         %[u2_r],         %[R63]          \n\t"

      "addq.ph      %[u3_l],         %[u3_l],         %[R63]          \n\t"
      "addq.ph      %[u3_r],         %[u3_r],         %[R63]          \n\t"

      /* vp8_signed_char_clamp((63 + Filter2 * 27) >> 7)
       * vp8_signed_char_clamp((63 + Filter2 * 18) >> 7)
       */
      "addq.ph      %[u1_l],         %[u1_l],         %[R63]          \n\t"
      "addq.ph      %[u1_r],         %[u1_r],         %[R63]          \n\t"
      "shra.ph      %[u1_l],         %[u1_l],         7               \n\t"
      "shra.ph      %[u1_r],         %[u1_r],         7               \n\t"
      "shra.ph      %[u2_l],         %[u2_l],         7               \n\t"
      "shra.ph      %[u2_r],         %[u2_r],         7               \n\t"
      "shll.ph      %[u1_l],         %[u1_l],         8               \n\t"
      "shll.ph      %[u1_r],         %[u1_r],         8               \n\t"
      "shll.ph      %[u2_l],         %[u2_l],         8               \n\t"
      "shll.ph      %[u2_r],         %[u2_r],         8               \n\t"

      /* vqs0 = vp8_signed_char_clamp(qs0 - u); */
      "subq_s.ph    %[vqs0_l],       %[vqs0_l],       %[u1_l]         \n\t"
      "subq_s.ph    %[vqs0_r],       %[vqs0_r],       %[u1_r]         \n\t"

      /* vps0 = vp8_signed_char_clamp(ps0 + u); */
      "addq_s.ph    %[vps0_l],       %[vps0_l],       %[u1_l]         \n\t"
      "addq_s.ph    %[vps0_r],       %[vps0_r],       %[u1_r]         \n\t"

      : [u1_l] "=&r"(u1_l), [u1_r] "=&r"(u1_r), [u2_l] "=&r"(u2_l),
        [u2_r] "=&r"(u2_r), [u3_l] "=&r"(u3_l), [u3_r] "=&r"(u3_r),
        [vps0_l] "+r"(vps0_l), [vps0_r] "+r"(vps0_r), [vqs0_l] "+r"(vqs0_l),
        [vqs0_r] "+r"(vqs0_r)
      : [R63] "r"(R63), [Filter2_l] "r"(Filter2_l), [Filter2_r] "r"(Filter2_r));

  __asm__ __volatile__(
      /* vqs1 = vp8_signed_char_clamp(qs1 - u); */
      "subq_s.ph    %[vqs1_l],       %[vqs1_l],       %[u2_l]         \n\t"
      "addq_s.ph    %[vps1_l],       %[vps1_l],       %[u2_l]         \n\t"

      /* vps1 = vp8_signed_char_clamp(ps1 + u); */
      "addq_s.ph    %[vps1_r],       %[vps1_r],       %[u2_r]         \n\t"
      "subq_s.ph    %[vqs1_r],       %[vqs1_r],       %[u2_r]         \n\t"

      : [vps1_l] "+r"(vps1_l), [vps1_r] "+r"(vps1_r), [vqs1_l] "+r"(vqs1_l),
        [vqs1_r] "+r"(vqs1_r)
      : [u2_l] "r"(u2_l), [u2_r] "r"(u2_r));

  /* roughly 1/7th difference across boundary */
  __asm__ __volatile__(
      /* u = vp8_signed_char_clamp((63 + Filter2 * 9) >> 7); */
      "shra.ph      %[u3_l],         %[u3_l],         7               \n\t"
      "shra.ph      %[u3_r],         %[u3_r],         7               \n\t"
      "shll.ph      %[u3_l],         %[u3_l],         8               \n\t"
      "shll.ph      %[u3_r],         %[u3_r],         8               \n\t"

      /* vqs2 = vp8_signed_char_clamp(qs2 - u); */
      "subq_s.ph    %[vqs2_l],       %[vqs2_l],       %[u3_l]         \n\t"
      "subq_s.ph    %[vqs2_r],       %[vqs2_r],       %[u3_r]         \n\t"

      /* vps2 = vp8_signed_char_clamp(ps2 + u); */
      "addq_s.ph    %[vps2_l],       %[vps2_l],       %[u3_l]         \n\t"
      "addq_s.ph    %[vps2_r],       %[vps2_r],       %[u3_r]         \n\t"

      : [u3_l] "+r"(u3_l), [u3_r] "+r"(u3_r), [vps2_l] "+r"(vps2_l),
        [vps2_r] "+r"(vps2_r), [vqs2_l] "+r"(vqs2_l), [vqs2_r] "+r"(vqs2_r)
      :);

  /* Create quad-bytes from halfword pairs */
  __asm__ __volatile__(
      "and          %[vqs0_l],       %[vqs0_l],       %[HWM]          \n\t"
      "shrl.ph      %[vqs0_r],       %[vqs0_r],       8               \n\t"

      "and          %[vps0_l],       %[vps0_l],       %[HWM]          \n\t"
      "shrl.ph      %[vps0_r],       %[vps0_r],       8               \n\t"

      "and          %[vqs1_l],       %[vqs1_l],       %[HWM]          \n\t"
      "shrl.ph      %[vqs1_r],       %[vqs1_r],       8               \n\t"

      "and          %[vps1_l],       %[vps1_l],       %[HWM]          \n\t"
      "shrl.ph      %[vps1_r],       %[vps1_r],       8               \n\t"

      "and          %[vqs2_l],       %[vqs2_l],       %[HWM]          \n\t"
      "shrl.ph      %[vqs2_r],       %[vqs2_r],       8               \n\t"

      "and          %[vps2_l],       %[vps2_l],       %[HWM]          \n\t"
      "shrl.ph      %[vps2_r],       %[vps2_r],       8               \n\t"

      "or           %[vqs0_r],       %[vqs0_l],       %[vqs0_r]       \n\t"
      "or           %[vps0_r],       %[vps0_l],       %[vps0_r]       \n\t"
      "or           %[vqs1_r],       %[vqs1_l],       %[vqs1_r]       \n\t"
      "or           %[vps1_r],       %[vps1_l],       %[vps1_r]       \n\t"
      "or           %[vqs2_r],       %[vqs2_l],       %[vqs2_r]       \n\t"
      "or           %[vps2_r],       %[vps2_l],       %[vps2_r]       \n\t"

      : [vps1_l] "+r"(vps1_l), [vps1_r] "+r"(vps1_r), [vqs1_l] "+r"(vqs1_l),
        [vqs1_r] "+r"(vqs1_r), [vps0_l] "+r"(vps0_l), [vps0_r] "+r"(vps0_r),
        [vqs0_l] "+r"(vqs0_l), [vqs0_r] "+r"(vqs0_r), [vqs2_l] "+r"(vqs2_l),
        [vqs2_r] "+r"(vqs2_r), [vps2_r] "+r"(vps2_r), [vps2_l] "+r"(vps2_l)
      : [HWM] "r"(HWM));

  *ps0 = vps0_r ^ N128;
  *ps1 = vps1_r ^ N128;
  *ps2 = vps2_r ^ N128;
  *qs0 = vqs0_r ^ N128;
  *qs1 = vqs1_r ^ N128;
  *qs2 = vqs2_r ^ N128;
}

void vp8_mbloop_filter_horizontal_edge_mips(unsigned char *s, int p,
                                            unsigned int flimit,
                                            unsigned int limit,
                                            unsigned int thresh, int count) {
  int i;
  uint32_t mask, hev;
  uint32_t pm1, p0, p1, p2, p3, p4, p5, p6;
  unsigned char *sm1, *s0, *s1, *s2, *s3, *s4, *s5, *s6;

  mask = 0;
  hev = 0;
  i = 0;
  p1 = 0;
  p2 = 0;
  p3 = 0;
  p4 = 0;

  /* loop filter designed to work using chars so that we can make maximum use
   * of 8 bit simd instructions.
   */

  sm1 = s - (p << 2);
  s0 = s - p - p - p;
  s1 = s - p - p;
  s2 = s - p;
  s3 = s;
  s4 = s + p;
  s5 = s + p + p;
  s6 = s + p + p + p;

  /* prefetch data for load */
  prefetch_load_lf(s + p);

  /* apply filter on 4 pixesl at the same time */
  do {
    /* load quad-byte vectors
     * memory is 4 byte aligned
     */
    p1 = *((uint32_t *)(s1));
    p2 = *((uint32_t *)(s2));
    p3 = *((uint32_t *)(s3));
    p4 = *((uint32_t *)(s4));

    /* if (p1 - p4 == 0) and (p2 - p3 == 0)
     * mask will be zero and filtering is not needed
     */
    if (!(((p1 - p4) == 0) && ((p2 - p3) == 0))) {
      pm1 = *((uint32_t *)(sm1));
      p0 = *((uint32_t *)(s0));
      p5 = *((uint32_t *)(s5));
      p6 = *((uint32_t *)(s6));

      vp8_filter_mask_vec_mips(limit, flimit, p1, p2, pm1, p0, p3, p4, p5, p6,
                               thresh, &hev, &mask);

      /* if mask == 0 do filtering is not needed */
      if (mask) {
        /* filtering */
        vp8_mbfilter_mips(mask, hev, &p0, &p1, &p2, &p3, &p4, &p5);

        /* unpack processed 4x4 neighborhood
         * memory is 4 byte aligned
         */
        *((uint32_t *)s0) = p0;
        *((uint32_t *)s1) = p1;
        *((uint32_t *)s2) = p2;
        *((uint32_t *)s3) = p3;
        *((uint32_t *)s4) = p4;
        *((uint32_t *)s5) = p5;
      }
    }

    sm1 += 4;
    s0 += 4;
    s1 += 4;
    s2 += 4;
    s3 += 4;
    s4 += 4;
    s5 += 4;
    s6 += 4;

    /* load quad-byte vectors
     * memory is 4 byte aligned
     */
    p1 = *((uint32_t *)(s1));
    p2 = *((uint32_t *)(s2));
    p3 = *((uint32_t *)(s3));
    p4 = *((uint32_t *)(s4));

    /* if (p1 - p4 == 0) and (p2 - p3 == 0)
     * mask will be zero and filtering is not needed
     */
    if (!(((p1 - p4) == 0) && ((p2 - p3) == 0))) {
      pm1 = *((uint32_t *)(sm1));
      p0 = *((uint32_t *)(s0));
      p5 = *((uint32_t *)(s5));
      p6 = *((uint32_t *)(s6));

      vp8_filter_mask_vec_mips(limit, flimit, p1, p2, pm1, p0, p3, p4, p5, p6,
                               thresh, &hev, &mask);

      /* if mask == 0 do filtering is not needed */
      if (mask) {
        /* filtering */
        vp8_mbfilter_mips(mask, hev, &p0, &p1, &p2, &p3, &p4, &p5);

        /* unpack processed 4x4 neighborhood
         * memory is 4 byte aligned
         */
        *((uint32_t *)s0) = p0;
        *((uint32_t *)s1) = p1;
        *((uint32_t *)s2) = p2;
        *((uint32_t *)s3) = p3;
        *((uint32_t *)s4) = p4;
        *((uint32_t *)s5) = p5;
      }
    }

    sm1 += 4;
    s0 += 4;
    s1 += 4;
    s2 += 4;
    s3 += 4;
    s4 += 4;
    s5 += 4;
    s6 += 4;

    i += 8;
  }

  while (i < count);
}

void vp8_mbloop_filter_uvhorizontal_edge_mips(unsigned char *s, int p,
                                              unsigned int flimit,
                                              unsigned int limit,
                                              unsigned int thresh, int count) {
  uint32_t mask, hev;
  uint32_t pm1, p0, p1, p2, p3, p4, p5, p6;
  unsigned char *sm1, *s0, *s1, *s2, *s3, *s4, *s5, *s6;
  (void)count;

  mask = 0;
  hev = 0;
  p1 = 0;
  p2 = 0;
  p3 = 0;
  p4 = 0;

  /* loop filter designed to work using chars so that we can make maximum use
   * of 8 bit simd instructions.
   */

  sm1 = s - (p << 2);
  s0 = s - p - p - p;
  s1 = s - p - p;
  s2 = s - p;
  s3 = s;
  s4 = s + p;
  s5 = s + p + p;
  s6 = s + p + p + p;

  /* load quad-byte vectors
   * memory is 4 byte aligned
   */
  p1 = *((uint32_t *)(s1));
  p2 = *((uint32_t *)(s2));
  p3 = *((uint32_t *)(s3));
  p4 = *((uint32_t *)(s4));

  /* if (p1 - p4 == 0) and (p2 - p3 == 0)
   * mask will be zero and filtering is not needed
   */
  if (!(((p1 - p4) == 0) && ((p2 - p3) == 0))) {
    pm1 = *((uint32_t *)(sm1));
    p0 = *((uint32_t *)(s0));
    p5 = *((uint32_t *)(s5));
    p6 = *((uint32_t *)(s6));

    /* if mask == 0 do filtering is not needed */
    vp8_filter_mask_vec_mips(limit, flimit, p1, p2, pm1, p0, p3, p4, p5, p6,
                             thresh, &hev, &mask);

    if (mask) {
      /* filtering */
      vp8_mbfilter_mips(mask, hev, &p0, &p1, &p2, &p3, &p4, &p5);

      /* unpack processed 4x4 neighborhood
       * memory is 4 byte aligned
       */
      *((uint32_t *)s0) = p0;
      *((uint32_t *)s1) = p1;
      *((uint32_t *)s2) = p2;
      *((uint32_t *)s3) = p3;
      *((uint32_t *)s4) = p4;
      *((uint32_t *)s5) = p5;
    }
  }

  sm1 += 4;
  s0 += 4;
  s1 += 4;
  s2 += 4;
  s3 += 4;
  s4 += 4;
  s5 += 4;
  s6 += 4;

  /* load quad-byte vectors
   * memory is 4 byte aligned
   */
  p1 = *((uint32_t *)(s1));
  p2 = *((uint32_t *)(s2));
  p3 = *((uint32_t *)(s3));
  p4 = *((uint32_t *)(s4));

  /* if (p1 - p4 == 0) and (p2 - p3 == 0)
   * mask will be zero and filtering is not needed
   */
  if (!(((p1 - p4) == 0) && ((p2 - p3) == 0))) {
    pm1 = *((uint32_t *)(sm1));
    p0 = *((uint32_t *)(s0));
    p5 = *((uint32_t *)(s5));
    p6 = *((uint32_t *)(s6));

    vp8_filter_mask_vec_mips(limit, flimit, p1, p2, pm1, p0, p3, p4, p5, p6,
                             thresh, &hev, &mask);

    /* if mask == 0 do filtering is not needed */
    if (mask) {
      /* filtering */
      vp8_mbfilter_mips(mask, hev, &p0, &p1, &p2, &p3, &p4, &p5);

      /* unpack processed 4x4 neighborhood
       * memory is 4 byte aligned
       */
      *((uint32_t *)s0) = p0;
      *((uint32_t *)s1) = p1;
      *((uint32_t *)s2) = p2;
      *((uint32_t *)s3) = p3;
      *((uint32_t *)s4) = p4;
      *((uint32_t *)s5) = p5;
    }
  }
}

void vp8_mbloop_filter_vertical_edge_mips(unsigned char *s, int p,
                                          unsigned int flimit,
                                          unsigned int limit,
                                          unsigned int thresh, int count) {
  int i;
  uint32_t mask, hev;
  uint32_t pm1, p0, p1, p2, p3, p4, p5, p6;
  unsigned char *s1, *s2, *s3, *s4;
  uint32_t prim1, prim2, sec3, sec4, prim3, prim4;

  mask = 0;
  hev = 0;
  i = 0;
  pm1 = 0;
  p0 = 0;
  p1 = 0;
  p2 = 0;
  p3 = 0;
  p4 = 0;
  p5 = 0;
  p6 = 0;

  /* loop filter designed to work using chars so that we can make maximum use
   * of 8 bit simd instructions.
   */

  /* apply filter on 4 pixesl at the same time */
  do {
    s1 = s;
    s2 = s + p;
    s3 = s2 + p;
    s4 = s3 + p;
    s = s4 + p;

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
      vp8_filter_mask_vec_mips(limit, flimit, p1, p2, pm1, p0, p3, p4, p5, p6,
                               thresh, &hev, &mask);

      /* if mask == 0 do filtering is not needed */
      if (mask) {
        /* filtering */
        vp8_mbfilter_mips(mask, hev, &p0, &p1, &p2, &p3, &p4, &p5);

        /* don't use transpose on output data
         * because memory isn't aligned
         */
        __asm__ __volatile__(
            "sb         %[p5],  2(%[s4])        \n\t"
            "sb         %[p4],  1(%[s4])        \n\t"
            "sb         %[p3],  0(%[s4])        \n\t"
            "sb         %[p2], -1(%[s4])        \n\t"
            "sb         %[p1], -2(%[s4])        \n\t"
            "sb         %[p0], -3(%[s4])        \n\t"
            :
            : [p5] "r"(p5), [p4] "r"(p4), [p3] "r"(p3), [s4] "r"(s4),
              [p2] "r"(p2), [p1] "r"(p1), [p0] "r"(p0));

        __asm__ __volatile__(
            "srl        %[p5], %[p5], 8         \n\t"
            "srl        %[p4], %[p4], 8         \n\t"
            "srl        %[p3], %[p3], 8         \n\t"
            "srl        %[p2], %[p2], 8         \n\t"
            "srl        %[p1], %[p1], 8         \n\t"
            "srl        %[p0], %[p0], 8         \n\t"
            : [p5] "+r"(p5), [p4] "+r"(p4), [p3] "+r"(p3), [p2] "+r"(p2),
              [p1] "+r"(p1), [p0] "+r"(p0)
            :);

        __asm__ __volatile__(
            "sb         %[p5],  2(%[s3])        \n\t"
            "sb         %[p4],  1(%[s3])        \n\t"
            "sb         %[p3],  0(%[s3])        \n\t"
            "sb         %[p2], -1(%[s3])        \n\t"
            "sb         %[p1], -2(%[s3])        \n\t"
            "sb         %[p0], -3(%[s3])        \n\t"
            :
            : [p5] "r"(p5), [p4] "r"(p4), [p3] "r"(p3), [s3] "r"(s3),
              [p2] "r"(p2), [p1] "r"(p1), [p0] "r"(p0));

        __asm__ __volatile__(
            "srl        %[p5], %[p5], 8         \n\t"
            "srl        %[p4], %[p4], 8         \n\t"
            "srl        %[p3], %[p3], 8         \n\t"
            "srl        %[p2], %[p2], 8         \n\t"
            "srl        %[p1], %[p1], 8         \n\t"
            "srl        %[p0], %[p0], 8         \n\t"
            : [p5] "+r"(p5), [p4] "+r"(p4), [p3] "+r"(p3), [p2] "+r"(p2),
              [p1] "+r"(p1), [p0] "+r"(p0)
            :);

        __asm__ __volatile__(
            "sb         %[p5],  2(%[s2])        \n\t"
            "sb         %[p4],  1(%[s2])        \n\t"
            "sb         %[p3],  0(%[s2])        \n\t"
            "sb         %[p2], -1(%[s2])        \n\t"
            "sb         %[p1], -2(%[s2])        \n\t"
            "sb         %[p0], -3(%[s2])        \n\t"
            :
            : [p5] "r"(p5), [p4] "r"(p4), [p3] "r"(p3), [s2] "r"(s2),
              [p2] "r"(p2), [p1] "r"(p1), [p0] "r"(p0));

        __asm__ __volatile__(
            "srl        %[p5], %[p5], 8         \n\t"
            "srl        %[p4], %[p4], 8         \n\t"
            "srl        %[p3], %[p3], 8         \n\t"
            "srl        %[p2], %[p2], 8         \n\t"
            "srl        %[p1], %[p1], 8         \n\t"
            "srl        %[p0], %[p0], 8         \n\t"
            : [p5] "+r"(p5), [p4] "+r"(p4), [p3] "+r"(p3), [p2] "+r"(p2),
              [p1] "+r"(p1), [p0] "+r"(p0)
            :);

        __asm__ __volatile__(
            "sb         %[p5],  2(%[s1])        \n\t"
            "sb         %[p4],  1(%[s1])        \n\t"
            "sb         %[p3],  0(%[s1])        \n\t"
            "sb         %[p2], -1(%[s1])        \n\t"
            "sb         %[p1], -2(%[s1])        \n\t"
            "sb         %[p0], -3(%[s1])        \n\t"
            :
            : [p5] "r"(p5), [p4] "r"(p4), [p3] "r"(p3), [s1] "r"(s1),
              [p2] "r"(p2), [p1] "r"(p1), [p0] "r"(p0));
      }
    }

    i += 4;
  }

  while (i < count);
}

void vp8_mbloop_filter_uvvertical_edge_mips(unsigned char *s, int p,
                                            unsigned int flimit,
                                            unsigned int limit,
                                            unsigned int thresh, int count) {
  uint32_t mask, hev;
  uint32_t pm1, p0, p1, p2, p3, p4, p5, p6;
  unsigned char *s1, *s2, *s3, *s4;
  uint32_t prim1, prim2, sec3, sec4, prim3, prim4;
  (void)count;

  mask = 0;
  hev = 0;
  pm1 = 0;
  p0 = 0;
  p1 = 0;
  p2 = 0;
  p3 = 0;
  p4 = 0;
  p5 = 0;
  p6 = 0;

  /* loop filter designed to work using chars so that we can make maximum use
   * of 8 bit simd instructions.
   */

  /* apply filter on 4 pixesl at the same time */

  s1 = s;
  s2 = s + p;
  s3 = s2 + p;
  s4 = s3 + p;

  /* prefetch data for load */
  prefetch_load_lf(s + 2 * p);

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
    vp8_filter_mask_vec_mips(limit, flimit, p1, p2, pm1, p0, p3, p4, p5, p6,
                             thresh, &hev, &mask);

    /* if mask == 0 do filtering is not needed */
    if (mask) {
      /* filtering */
      vp8_mbfilter_mips(mask, hev, &p0, &p1, &p2, &p3, &p4, &p5);

      /* don't use transpose on output data
       * because memory isn't aligned
       */
      __asm__ __volatile__(
          "sb         %[p5],  2(%[s4])        \n\t"
          "sb         %[p4],  1(%[s4])        \n\t"
          "sb         %[p3],  0(%[s4])        \n\t"
          "sb         %[p2], -1(%[s4])        \n\t"
          "sb         %[p1], -2(%[s4])        \n\t"
          "sb         %[p0], -3(%[s4])        \n\t"
          :
          : [p5] "r"(p5), [p4] "r"(p4), [p3] "r"(p3), [s4] "r"(s4),
            [p2] "r"(p2), [p1] "r"(p1), [p0] "r"(p0));

      __asm__ __volatile__(
          "srl        %[p5], %[p5], 8         \n\t"
          "srl        %[p4], %[p4], 8         \n\t"
          "srl        %[p3], %[p3], 8         \n\t"
          "srl        %[p2], %[p2], 8         \n\t"
          "srl        %[p1], %[p1], 8         \n\t"
          "srl        %[p0], %[p0], 8         \n\t"
          : [p5] "+r"(p5), [p4] "+r"(p4), [p3] "+r"(p3), [p2] "+r"(p2),
            [p1] "+r"(p1), [p0] "+r"(p0)
          :);

      __asm__ __volatile__(
          "sb         %[p5],  2(%[s3])        \n\t"
          "sb         %[p4],  1(%[s3])        \n\t"
          "sb         %[p3],  0(%[s3])        \n\t"
          "sb         %[p2], -1(%[s3])        \n\t"
          "sb         %[p1], -2(%[s3])        \n\t"
          "sb         %[p0], -3(%[s3])        \n\t"
          :
          : [p5] "r"(p5), [p4] "r"(p4), [p3] "r"(p3), [s3] "r"(s3),
            [p2] "r"(p2), [p1] "r"(p1), [p0] "r"(p0));

      __asm__ __volatile__(
          "srl        %[p5], %[p5], 8         \n\t"
          "srl        %[p4], %[p4], 8         \n\t"
          "srl        %[p3], %[p3], 8         \n\t"
          "srl        %[p2], %[p2], 8         \n\t"
          "srl        %[p1], %[p1], 8         \n\t"
          "srl        %[p0], %[p0], 8         \n\t"
          : [p5] "+r"(p5), [p4] "+r"(p4), [p3] "+r"(p3), [p2] "+r"(p2),
            [p1] "+r"(p1), [p0] "+r"(p0)
          :);

      __asm__ __volatile__(
          "sb         %[p5],  2(%[s2])        \n\t"
          "sb         %[p4],  1(%[s2])        \n\t"
          "sb         %[p3],  0(%[s2])        \n\t"
          "sb         %[p2], -1(%[s2])        \n\t"
          "sb         %[p1], -2(%[s2])        \n\t"
          "sb         %[p0], -3(%[s2])        \n\t"
          :
          : [p5] "r"(p5), [p4] "r"(p4), [p3] "r"(p3), [s2] "r"(s2),
            [p2] "r"(p2), [p1] "r"(p1), [p0] "r"(p0));

      __asm__ __volatile__(
          "srl        %[p5], %[p5], 8         \n\t"
          "srl        %[p4], %[p4], 8         \n\t"
          "srl        %[p3], %[p3], 8         \n\t"
          "srl        %[p2], %[p2], 8         \n\t"
          "srl        %[p1], %[p1], 8         \n\t"
          "srl        %[p0], %[p0], 8         \n\t"
          : [p5] "+r"(p5), [p4] "+r"(p4), [p3] "+r"(p3), [p2] "+r"(p2),
            [p1] "+r"(p1), [p0] "+r"(p0)
          :);

      __asm__ __volatile__(
          "sb         %[p5],  2(%[s1])        \n\t"
          "sb         %[p4],  1(%[s1])        \n\t"
          "sb         %[p3],  0(%[s1])        \n\t"
          "sb         %[p2], -1(%[s1])        \n\t"
          "sb         %[p1], -2(%[s1])        \n\t"
          "sb         %[p0], -3(%[s1])        \n\t"
          :
          : [p5] "r"(p5), [p4] "r"(p4), [p3] "r"(p3), [s1] "r"(s1),
            [p2] "r"(p2), [p1] "r"(p1), [p0] "r"(p0));
    }
  }

  s1 = s4 + p;
  s2 = s1 + p;
  s3 = s2 + p;
  s4 = s3 + p;

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
    vp8_filter_mask_vec_mips(limit, flimit, p1, p2, pm1, p0, p3, p4, p5, p6,
                             thresh, &hev, &mask);

    /* if mask == 0 do filtering is not needed */
    if (mask) {
      /* filtering */
      vp8_mbfilter_mips(mask, hev, &p0, &p1, &p2, &p3, &p4, &p5);

      /* don't use transpose on output data
       * because memory isn't aligned
       */
      __asm__ __volatile__(
          "sb         %[p5],  2(%[s4])        \n\t"
          "sb         %[p4],  1(%[s4])        \n\t"
          "sb         %[p3],  0(%[s4])        \n\t"
          "sb         %[p2], -1(%[s4])        \n\t"
          "sb         %[p1], -2(%[s4])        \n\t"
          "sb         %[p0], -3(%[s4])        \n\t"
          :
          : [p5] "r"(p5), [p4] "r"(p4), [p3] "r"(p3), [s4] "r"(s4),
            [p2] "r"(p2), [p1] "r"(p1), [p0] "r"(p0));

      __asm__ __volatile__(
          "srl        %[p5], %[p5], 8         \n\t"
          "srl        %[p4], %[p4], 8         \n\t"
          "srl        %[p3], %[p3], 8         \n\t"
          "srl        %[p2], %[p2], 8         \n\t"
          "srl        %[p1], %[p1], 8         \n\t"
          "srl        %[p0], %[p0], 8         \n\t"
          : [p5] "+r"(p5), [p4] "+r"(p4), [p3] "+r"(p3), [p2] "+r"(p2),
            [p1] "+r"(p1), [p0] "+r"(p0)
          :);

      __asm__ __volatile__(
          "sb         %[p5],  2(%[s3])        \n\t"
          "sb         %[p4],  1(%[s3])        \n\t"
          "sb         %[p3],  0(%[s3])        \n\t"
          "sb         %[p2], -1(%[s3])        \n\t"
          "sb         %[p1], -2(%[s3])        \n\t"
          "sb         %[p0], -3(%[s3])        \n\t"
          :
          : [p5] "r"(p5), [p4] "r"(p4), [p3] "r"(p3), [s3] "r"(s3),
            [p2] "r"(p2), [p1] "r"(p1), [p0] "r"(p0));

      __asm__ __volatile__(
          "srl        %[p5], %[p5], 8         \n\t"
          "srl        %[p4], %[p4], 8         \n\t"
          "srl        %[p3], %[p3], 8         \n\t"
          "srl        %[p2], %[p2], 8         \n\t"
          "srl        %[p1], %[p1], 8         \n\t"
          "srl        %[p0], %[p0], 8         \n\t"
          : [p5] "+r"(p5), [p4] "+r"(p4), [p3] "+r"(p3), [p2] "+r"(p2),
            [p1] "+r"(p1), [p0] "+r"(p0)
          :);

      __asm__ __volatile__(
          "sb         %[p5],  2(%[s2])        \n\t"
          "sb         %[p4],  1(%[s2])        \n\t"
          "sb         %[p3],  0(%[s2])        \n\t"
          "sb         %[p2], -1(%[s2])        \n\t"
          "sb         %[p1], -2(%[s2])        \n\t"
          "sb         %[p0], -3(%[s2])        \n\t"
          :
          : [p5] "r"(p5), [p4] "r"(p4), [p3] "r"(p3), [s2] "r"(s2),
            [p2] "r"(p2), [p1] "r"(p1), [p0] "r"(p0));

      __asm__ __volatile__(
          "srl        %[p5], %[p5], 8         \n\t"
          "srl        %[p4], %[p4], 8         \n\t"
          "srl        %[p3], %[p3], 8         \n\t"
          "srl        %[p2], %[p2], 8         \n\t"
          "srl        %[p1], %[p1], 8         \n\t"
          "srl        %[p0], %[p0], 8         \n\t"
          : [p5] "+r"(p5), [p4] "+r"(p4), [p3] "+r"(p3), [p2] "+r"(p2),
            [p1] "+r"(p1), [p0] "+r"(p0)
          :);

      __asm__ __volatile__(
          "sb         %[p5],  2(%[s1])        \n\t"
          "sb         %[p4],  1(%[s1])        \n\t"
          "sb         %[p3],  0(%[s1])        \n\t"
          "sb         %[p2], -1(%[s1])        \n\t"
          "sb         %[p1], -2(%[s1])        \n\t"
          "sb         %[p0], -3(%[s1])        \n\t"
          :
          : [p5] "r"(p5), [p4] "r"(p4), [p3] "r"(p3), [s1] "r"(s1),
            [p2] "r"(p2), [p1] "r"(p1), [p0] "r"(p0));
    }
  }
}

/* Horizontal MB filtering */
void vp8_loop_filter_mbh_dspr2(unsigned char *y_ptr, unsigned char *u_ptr,
                               unsigned char *v_ptr, int y_stride,
                               int uv_stride, loop_filter_info *lfi) {
  unsigned int thresh_vec, flimit_vec, limit_vec;
  unsigned char thresh, flimit, limit, flimit_temp;

  /* use direct value instead pointers */
  limit = *(lfi->lim);
  flimit_temp = *(lfi->mblim);
  thresh = *(lfi->hev_thr);
  flimit = flimit_temp;

  /* create quad-byte */
  __asm__ __volatile__(
      "replv.qb       %[thresh_vec], %[thresh]    \n\t"
      "replv.qb       %[flimit_vec], %[flimit]    \n\t"
      "replv.qb       %[limit_vec],  %[limit]     \n\t"
      : [thresh_vec] "=&r"(thresh_vec), [flimit_vec] "=&r"(flimit_vec),
        [limit_vec] "=r"(limit_vec)
      : [thresh] "r"(thresh), [flimit] "r"(flimit), [limit] "r"(limit));

  vp8_mbloop_filter_horizontal_edge_mips(y_ptr, y_stride, flimit_vec, limit_vec,
                                         thresh_vec, 16);

  if (u_ptr) {
    vp8_mbloop_filter_uvhorizontal_edge_mips(u_ptr, uv_stride, flimit_vec,
                                             limit_vec, thresh_vec, 0);
  }

  if (v_ptr) {
    vp8_mbloop_filter_uvhorizontal_edge_mips(v_ptr, uv_stride, flimit_vec,
                                             limit_vec, thresh_vec, 0);
  }
}

/* Vertical MB Filtering */
void vp8_loop_filter_mbv_dspr2(unsigned char *y_ptr, unsigned char *u_ptr,
                               unsigned char *v_ptr, int y_stride,
                               int uv_stride, loop_filter_info *lfi) {
  unsigned int thresh_vec, flimit_vec, limit_vec;
  unsigned char thresh, flimit, limit, flimit_temp;

  /* use direct value instead pointers */
  limit = *(lfi->lim);
  flimit_temp = *(lfi->mblim);
  thresh = *(lfi->hev_thr);
  flimit = flimit_temp;

  /* create quad-byte */
  __asm__ __volatile__(
      "replv.qb       %[thresh_vec], %[thresh]    \n\t"
      "replv.qb       %[flimit_vec], %[flimit]    \n\t"
      "replv.qb       %[limit_vec],  %[limit]     \n\t"
      : [thresh_vec] "=&r"(thresh_vec), [flimit_vec] "=&r"(flimit_vec),
        [limit_vec] "=r"(limit_vec)
      : [thresh] "r"(thresh), [flimit] "r"(flimit), [limit] "r"(limit));

  vp8_mbloop_filter_vertical_edge_mips(y_ptr, y_stride, flimit_vec, limit_vec,
                                       thresh_vec, 16);

  if (u_ptr)
    vp8_mbloop_filter_uvvertical_edge_mips(u_ptr, uv_stride, flimit_vec,
                                           limit_vec, thresh_vec, 0);

  if (v_ptr)
    vp8_mbloop_filter_uvvertical_edge_mips(v_ptr, uv_stride, flimit_vec,
                                           limit_vec, thresh_vec, 0);
}

/* Horizontal B Filtering */
void vp8_loop_filter_bh_dspr2(unsigned char *y_ptr, unsigned char *u_ptr,
                              unsigned char *v_ptr, int y_stride, int uv_stride,
                              loop_filter_info *lfi) {
  unsigned int thresh_vec, flimit_vec, limit_vec;
  unsigned char thresh, flimit, limit, flimit_temp;

  /* use direct value instead pointers */
  limit = *(lfi->lim);
  flimit_temp = *(lfi->blim);
  thresh = *(lfi->hev_thr);
  flimit = flimit_temp;

  /* create quad-byte */
  __asm__ __volatile__(
      "replv.qb       %[thresh_vec], %[thresh]    \n\t"
      "replv.qb       %[flimit_vec], %[flimit]    \n\t"
      "replv.qb       %[limit_vec],  %[limit]     \n\t"
      : [thresh_vec] "=&r"(thresh_vec), [flimit_vec] "=&r"(flimit_vec),
        [limit_vec] "=r"(limit_vec)
      : [thresh] "r"(thresh), [flimit] "r"(flimit), [limit] "r"(limit));

  vp8_loop_filter_horizontal_edge_mips(y_ptr + 4 * y_stride, y_stride,
                                       flimit_vec, limit_vec, thresh_vec, 16);
  vp8_loop_filter_horizontal_edge_mips(y_ptr + 8 * y_stride, y_stride,
                                       flimit_vec, limit_vec, thresh_vec, 16);
  vp8_loop_filter_horizontal_edge_mips(y_ptr + 12 * y_stride, y_stride,
                                       flimit_vec, limit_vec, thresh_vec, 16);

  if (u_ptr)
    vp8_loop_filter_uvhorizontal_edge_mips(
        u_ptr + 4 * uv_stride, uv_stride, flimit_vec, limit_vec, thresh_vec, 0);

  if (v_ptr)
    vp8_loop_filter_uvhorizontal_edge_mips(
        v_ptr + 4 * uv_stride, uv_stride, flimit_vec, limit_vec, thresh_vec, 0);
}

/* Vertical B Filtering */
void vp8_loop_filter_bv_dspr2(unsigned char *y_ptr, unsigned char *u_ptr,
                              unsigned char *v_ptr, int y_stride, int uv_stride,
                              loop_filter_info *lfi) {
  unsigned int thresh_vec, flimit_vec, limit_vec;
  unsigned char thresh, flimit, limit, flimit_temp;

  /* use direct value instead pointers */
  limit = *(lfi->lim);
  flimit_temp = *(lfi->blim);
  thresh = *(lfi->hev_thr);
  flimit = flimit_temp;

  /* create quad-byte */
  __asm__ __volatile__(
      "replv.qb       %[thresh_vec], %[thresh]    \n\t"
      "replv.qb       %[flimit_vec], %[flimit]    \n\t"
      "replv.qb       %[limit_vec],  %[limit]     \n\t"
      : [thresh_vec] "=&r"(thresh_vec), [flimit_vec] "=&r"(flimit_vec),
        [limit_vec] "=r"(limit_vec)
      : [thresh] "r"(thresh), [flimit] "r"(flimit), [limit] "r"(limit));

  vp8_loop_filter_vertical_edge_mips(y_ptr + 4, y_stride, flimit_vec, limit_vec,
                                     thresh_vec, 16);
  vp8_loop_filter_vertical_edge_mips(y_ptr + 8, y_stride, flimit_vec, limit_vec,
                                     thresh_vec, 16);
  vp8_loop_filter_vertical_edge_mips(y_ptr + 12, y_stride, flimit_vec,
                                     limit_vec, thresh_vec, 16);

  if (u_ptr)
    vp8_loop_filter_uvvertical_edge_mips(u_ptr + 4, uv_stride, flimit_vec,
                                         limit_vec, thresh_vec, 0);

  if (v_ptr)
    vp8_loop_filter_uvvertical_edge_mips(v_ptr + 4, uv_stride, flimit_vec,
                                         limit_vec, thresh_vec, 0);
}

#endif
