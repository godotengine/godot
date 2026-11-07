/*
 *  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_MIPS_LOOPFILTER_FILTERS_DSPR2_H_
#define VPX_VPX_DSP_MIPS_LOOPFILTER_FILTERS_DSPR2_H_

#include <stdlib.h>

#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"
#include "vpx_mem/vpx_mem.h"
#include "vpx_ports/mem.h"

#ifdef __cplusplus
extern "C" {
#endif

#if HAVE_DSPR2
/* inputs & outputs are quad-byte vectors */
static INLINE void filter_dspr2(uint32_t mask, uint32_t hev, uint32_t *ps1,
                                uint32_t *ps0, uint32_t *qs0, uint32_t *qs1) {
  int32_t vpx_filter_l, vpx_filter_r;
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
      /* vpx_filter = vp8_signed_char_clamp(ps1 - qs1); */
      "subq_s.ph    %[vpx_filter_l], %[vps1_l],       %[vqs1_l]       \n\t"
      "subq_s.ph    %[vpx_filter_r], %[vps1_r],       %[vqs1_r]       \n\t"

      /* qs0 - ps0 */
      "subq_s.ph    %[subr_l],       %[vqs0_l],       %[vps0_l]       \n\t"
      "subq_s.ph    %[subr_r],       %[vqs0_r],       %[vps0_r]       \n\t"

      /* vpx_filter &= hev; */
      "and          %[vpx_filter_l], %[vpx_filter_l], %[hev_l]        \n\t"
      "and          %[vpx_filter_r], %[vpx_filter_r], %[hev_r]        \n\t"

      /* vpx_filter = vp8_signed_char_clamp(vpx_filter + 3 * (qs0 - ps0)); */
      "addq_s.ph    %[vpx_filter_l], %[vpx_filter_l], %[subr_l]       \n\t"
      "addq_s.ph    %[vpx_filter_r], %[vpx_filter_r], %[subr_r]       \n\t"
      "xor          %[invhev_l],     %[hev_l],        %[HWM]          \n\t"
      "addq_s.ph    %[vpx_filter_l], %[vpx_filter_l], %[subr_l]       \n\t"
      "addq_s.ph    %[vpx_filter_r], %[vpx_filter_r], %[subr_r]       \n\t"
      "xor          %[invhev_r],     %[hev_r],        %[HWM]          \n\t"
      "addq_s.ph    %[vpx_filter_l], %[vpx_filter_l], %[subr_l]       \n\t"
      "addq_s.ph    %[vpx_filter_r], %[vpx_filter_r], %[subr_r]       \n\t"

      /* vpx_filter &= mask; */
      "and          %[vpx_filter_l], %[vpx_filter_l], %[mask_l]       \n\t"
      "and          %[vpx_filter_r], %[vpx_filter_r], %[mask_r]       \n\t"

      : [vpx_filter_l] "=&r"(vpx_filter_l), [vpx_filter_r] "=&r"(vpx_filter_r),
        [subr_l] "=&r"(subr_l), [subr_r] "=&r"(subr_r),
        [invhev_l] "=&r"(invhev_l), [invhev_r] "=&r"(invhev_r)
      : [vps0_l] "r"(vps0_l), [vps0_r] "r"(vps0_r), [vps1_l] "r"(vps1_l),
        [vps1_r] "r"(vps1_r), [vqs0_l] "r"(vqs0_l), [vqs0_r] "r"(vqs0_r),
        [vqs1_l] "r"(vqs1_l), [vqs1_r] "r"(vqs1_r), [mask_l] "r"(mask_l),
        [mask_r] "r"(mask_r), [hev_l] "r"(hev_l), [hev_r] "r"(hev_r),
        [HWM] "r"(HWM));

  /* save bottom 3 bits so that we round one side +4 and the other +3 */
  __asm__ __volatile__(
      /* Filter2 = vp8_signed_char_clamp(vpx_filter + 3) >>= 3; */
      "addq_s.ph    %[Filter1_l],    %[vpx_filter_l], %[t2]           \n\t"
      "addq_s.ph    %[Filter1_r],    %[vpx_filter_r], %[t2]           \n\t"

      /* Filter1 = vp8_signed_char_clamp(vpx_filter + 4) >>= 3; */
      "addq_s.ph    %[Filter2_l],    %[vpx_filter_l], %[t1]           \n\t"
      "addq_s.ph    %[Filter2_r],    %[vpx_filter_r], %[t1]           \n\t"
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
      : [t1] "r"(t1), [t2] "r"(t2), [HWM] "r"(HWM),
        [vpx_filter_l] "r"(vpx_filter_l), [vpx_filter_r] "r"(vpx_filter_r));

  __asm__ __volatile__(
      /* (vpx_filter += 1) >>= 1 */
      "addqh.ph    %[Filter1_l],    %[Filter1_l],     %[t3]           \n\t"
      "addqh.ph    %[Filter1_r],    %[Filter1_r],     %[t3]           \n\t"

      /* vpx_filter &= ~hev; */
      "and          %[Filter1_l],    %[Filter1_l],    %[invhev_l]     \n\t"
      "and          %[Filter1_r],    %[Filter1_r],    %[invhev_r]     \n\t"

      /* vps1 = vp8_signed_char_clamp(ps1 + vpx_filter); */
      "addq_s.ph    %[vps1_l],       %[vps1_l],       %[Filter1_l]    \n\t"
      "addq_s.ph    %[vps1_r],       %[vps1_r],       %[Filter1_r]    \n\t"

      /* vqs1 = vp8_signed_char_clamp(qs1 - vpx_filter); */
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
      "shrl.ph      %[vqs0_r],       %[vqs0_r],       8   \n\t"
      "shrl.ph      %[vps0_r],       %[vps0_r],       8   \n\t"
      "shrl.ph      %[vqs1_r],       %[vqs1_r],       8   \n\t"
      "shrl.ph      %[vps1_r],       %[vps1_r],       8   \n\t"

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

static INLINE void filter1_dspr2(uint32_t mask, uint32_t hev, uint32_t ps1,
                                 uint32_t ps0, uint32_t qs0, uint32_t qs1,
                                 uint32_t *p1_f0, uint32_t *p0_f0,
                                 uint32_t *q0_f0, uint32_t *q1_f0) {
  int32_t vpx_filter_l, vpx_filter_r;
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

  vps0 = (ps0) ^ N128;
  vps1 = (ps1) ^ N128;
  vqs0 = (qs0) ^ N128;
  vqs1 = (qs1) ^ N128;

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
      /* vpx_filter = vp8_signed_char_clamp(ps1 - qs1); */
      "subq_s.ph    %[vpx_filter_l], %[vps1_l],       %[vqs1_l]       \n\t"
      "subq_s.ph    %[vpx_filter_r], %[vps1_r],       %[vqs1_r]       \n\t"

      /* qs0 - ps0 */
      "subq_s.ph    %[subr_l],       %[vqs0_l],       %[vps0_l]       \n\t"
      "subq_s.ph    %[subr_r],       %[vqs0_r],       %[vps0_r]       \n\t"

      /* vpx_filter &= hev; */
      "and          %[vpx_filter_l], %[vpx_filter_l], %[hev_l]        \n\t"
      "and          %[vpx_filter_r], %[vpx_filter_r], %[hev_r]        \n\t"

      /* vpx_filter = vp8_signed_char_clamp(vpx_filter + 3 * (qs0 - ps0)); */
      "addq_s.ph    %[vpx_filter_l], %[vpx_filter_l], %[subr_l]       \n\t"
      "addq_s.ph    %[vpx_filter_r], %[vpx_filter_r], %[subr_r]       \n\t"
      "xor          %[invhev_l],     %[hev_l],        %[HWM]          \n\t"
      "addq_s.ph    %[vpx_filter_l], %[vpx_filter_l], %[subr_l]       \n\t"
      "addq_s.ph    %[vpx_filter_r], %[vpx_filter_r], %[subr_r]       \n\t"
      "xor          %[invhev_r],     %[hev_r],        %[HWM]          \n\t"
      "addq_s.ph    %[vpx_filter_l], %[vpx_filter_l], %[subr_l]       \n\t"
      "addq_s.ph    %[vpx_filter_r], %[vpx_filter_r], %[subr_r]       \n\t"

      /* vpx_filter &= mask; */
      "and          %[vpx_filter_l], %[vpx_filter_l], %[mask_l]       \n\t"
      "and          %[vpx_filter_r], %[vpx_filter_r], %[mask_r]       \n\t"

      : [vpx_filter_l] "=&r"(vpx_filter_l), [vpx_filter_r] "=&r"(vpx_filter_r),
        [subr_l] "=&r"(subr_l), [subr_r] "=&r"(subr_r),
        [invhev_l] "=&r"(invhev_l), [invhev_r] "=&r"(invhev_r)
      : [vps0_l] "r"(vps0_l), [vps0_r] "r"(vps0_r), [vps1_l] "r"(vps1_l),
        [vps1_r] "r"(vps1_r), [vqs0_l] "r"(vqs0_l), [vqs0_r] "r"(vqs0_r),
        [vqs1_l] "r"(vqs1_l), [vqs1_r] "r"(vqs1_r), [mask_l] "r"(mask_l),
        [mask_r] "r"(mask_r), [hev_l] "r"(hev_l), [hev_r] "r"(hev_r),
        [HWM] "r"(HWM));

  /* save bottom 3 bits so that we round one side +4 and the other +3 */
  __asm__ __volatile__(
      /* Filter2 = vp8_signed_char_clamp(vpx_filter + 3) >>= 3; */
      "addq_s.ph    %[Filter1_l],    %[vpx_filter_l], %[t2]           \n\t"
      "addq_s.ph    %[Filter1_r],    %[vpx_filter_r], %[t2]           \n\t"

      /* Filter1 = vp8_signed_char_clamp(vpx_filter + 4) >>= 3; */
      "addq_s.ph    %[Filter2_l],    %[vpx_filter_l], %[t1]           \n\t"
      "addq_s.ph    %[Filter2_r],    %[vpx_filter_r], %[t1]           \n\t"
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
      : [t1] "r"(t1), [t2] "r"(t2), [HWM] "r"(HWM),
        [vpx_filter_l] "r"(vpx_filter_l), [vpx_filter_r] "r"(vpx_filter_r));

  __asm__ __volatile__(
      /* (vpx_filter += 1) >>= 1 */
      "addqh.ph    %[Filter1_l],    %[Filter1_l],     %[t3]           \n\t"
      "addqh.ph    %[Filter1_r],    %[Filter1_r],     %[t3]           \n\t"

      /* vpx_filter &= ~hev; */
      "and          %[Filter1_l],    %[Filter1_l],    %[invhev_l]     \n\t"
      "and          %[Filter1_r],    %[Filter1_r],    %[invhev_r]     \n\t"

      /* vps1 = vp8_signed_char_clamp(ps1 + vpx_filter); */
      "addq_s.ph    %[vps1_l],       %[vps1_l],       %[Filter1_l]    \n\t"
      "addq_s.ph    %[vps1_r],       %[vps1_r],       %[Filter1_r]    \n\t"

      /* vqs1 = vp8_signed_char_clamp(qs1 - vpx_filter); */
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
      "shrl.ph      %[vqs0_r],       %[vqs0_r],       8   \n\t"
      "shrl.ph      %[vps0_r],       %[vps0_r],       8   \n\t"
      "shrl.ph      %[vqs1_r],       %[vqs1_r],       8   \n\t"
      "shrl.ph      %[vps1_r],       %[vps1_r],       8   \n\t"

      : [vps1_r] "+r"(vps1_r), [vqs1_r] "+r"(vqs1_r), [vps0_r] "+r"(vps0_r),
        [vqs0_r] "+r"(vqs0_r)
      :);

  vqs0 = vqs0_l | vqs0_r;
  vqs1 = vqs1_l | vqs1_r;
  vps0 = vps0_l | vps0_r;
  vps1 = vps1_l | vps1_r;

  *p0_f0 = vps0 ^ N128;
  *p1_f0 = vps1 ^ N128;
  *q0_f0 = vqs0 ^ N128;
  *q1_f0 = vqs1 ^ N128;
}

static INLINE void mbfilter_dspr2(uint32_t *op3, uint32_t *op2, uint32_t *op1,
                                  uint32_t *op0, uint32_t *oq0, uint32_t *oq1,
                                  uint32_t *oq2, uint32_t *oq3) {
  /* use a 7 tap filter [1, 1, 1, 2, 1, 1, 1] for flat line */
  const uint32_t p3 = *op3, p2 = *op2, p1 = *op1, p0 = *op0;
  const uint32_t q0 = *oq0, q1 = *oq1, q2 = *oq2, q3 = *oq3;
  uint32_t res_op2, res_op1, res_op0;
  uint32_t res_oq0, res_oq1, res_oq2;
  uint32_t tmp;
  uint32_t add_p210_q012;
  uint32_t u32Four = 0x00040004;

  /* *op2 = ROUND_POWER_OF_TWO(p3 + p3 + p3 + p2 + p2 + p1 + p0 + q0, 3)  1 */
  /* *op1 = ROUND_POWER_OF_TWO(p3 + p3 + p2 + p1 + p1 + p0 + q0 + q1, 3)  2 */
  /* *op0 = ROUND_POWER_OF_TWO(p3 + p2 + p1 + p0 + p0 + q0 + q1 + q2, 3)  3 */
  /* *oq0 = ROUND_POWER_OF_TWO(p2 + p1 + p0 + q0 + q0 + q1 + q2 + q3, 3)  4 */
  /* *oq1 = ROUND_POWER_OF_TWO(p1 + p0 + q0 + q1 + q1 + q2 + q3 + q3, 3)  5 */
  /* *oq2 = ROUND_POWER_OF_TWO(p0 + q0 + q1 + q2 + q2 + q3 + q3 + q3, 3)  6 */

  __asm__ __volatile__(
      "addu.ph    %[add_p210_q012],  %[p2],             %[p1]            \n\t"
      "addu.ph    %[add_p210_q012],  %[add_p210_q012],  %[p0]            \n\t"
      "addu.ph    %[add_p210_q012],  %[add_p210_q012],  %[q0]            \n\t"
      "addu.ph    %[add_p210_q012],  %[add_p210_q012],  %[q1]            \n\t"
      "addu.ph    %[add_p210_q012],  %[add_p210_q012],  %[q2]            \n\t"
      "addu.ph    %[add_p210_q012],  %[add_p210_q012],  %[u32Four]       \n\t"

      "shll.ph    %[tmp],            %[p3],             1                \n\t"
      "addu.ph    %[res_op2],        %[tmp],            %[p3]            \n\t"
      "addu.ph    %[res_op1],        %[p3],             %[p3]            \n\t"
      "addu.ph    %[res_op2],        %[res_op2],        %[p2]            \n\t"
      "addu.ph    %[res_op1],        %[res_op1],        %[p1]            \n\t"
      "addu.ph    %[res_op2],        %[res_op2],        %[add_p210_q012] \n\t"
      "addu.ph    %[res_op1],        %[res_op1],        %[add_p210_q012] \n\t"
      "subu.ph    %[res_op2],        %[res_op2],        %[q1]            \n\t"
      "subu.ph    %[res_op1],        %[res_op1],        %[q2]            \n\t"
      "subu.ph    %[res_op2],        %[res_op2],        %[q2]            \n\t"
      "shrl.ph    %[res_op1],        %[res_op1],        3                \n\t"
      "shrl.ph    %[res_op2],        %[res_op2],        3                \n\t"
      "addu.ph    %[res_op0],        %[p3],             %[p0]            \n\t"
      "addu.ph    %[res_oq0],        %[q0],             %[q3]            \n\t"
      "addu.ph    %[res_op0],        %[res_op0],        %[add_p210_q012] \n\t"
      "addu.ph    %[res_oq0],        %[res_oq0],        %[add_p210_q012] \n\t"
      "addu.ph    %[res_oq1],        %[q3],             %[q3]            \n\t"
      "shll.ph    %[tmp],            %[q3],             1                \n\t"
      "addu.ph    %[res_oq1],        %[res_oq1],        %[q1]            \n\t"
      "addu.ph    %[res_oq2],        %[tmp],            %[q3]            \n\t"
      "addu.ph    %[res_oq1],        %[res_oq1],        %[add_p210_q012] \n\t"
      "addu.ph    %[res_oq2],        %[res_oq2],        %[add_p210_q012] \n\t"
      "subu.ph    %[res_oq1],        %[res_oq1],        %[p2]            \n\t"
      "addu.ph    %[res_oq2],        %[res_oq2],        %[q2]            \n\t"
      "shrl.ph    %[res_oq1],        %[res_oq1],        3                \n\t"
      "subu.ph    %[res_oq2],        %[res_oq2],        %[p2]            \n\t"
      "shrl.ph    %[res_oq0],        %[res_oq0],        3                \n\t"
      "subu.ph    %[res_oq2],        %[res_oq2],        %[p1]            \n\t"
      "shrl.ph    %[res_op0],        %[res_op0],        3                \n\t"
      "shrl.ph    %[res_oq2],        %[res_oq2],        3                \n\t"

      : [add_p210_q012] "=&r"(add_p210_q012), [tmp] "=&r"(tmp),
        [res_op2] "=&r"(res_op2), [res_op1] "=&r"(res_op1),
        [res_op0] "=&r"(res_op0), [res_oq0] "=&r"(res_oq0),
        [res_oq1] "=&r"(res_oq1), [res_oq2] "=&r"(res_oq2)
      : [p0] "r"(p0), [q0] "r"(q0), [p1] "r"(p1), [q1] "r"(q1), [p2] "r"(p2),
        [q2] "r"(q2), [p3] "r"(p3), [q3] "r"(q3), [u32Four] "r"(u32Four));

  *op2 = res_op2;
  *op1 = res_op1;
  *op0 = res_op0;
  *oq0 = res_oq0;
  *oq1 = res_oq1;
  *oq2 = res_oq2;
}

static INLINE void mbfilter1_dspr2(uint32_t p3, uint32_t p2, uint32_t p1,
                                   uint32_t p0, uint32_t q0, uint32_t q1,
                                   uint32_t q2, uint32_t q3, uint32_t *op2_f1,
                                   uint32_t *op1_f1, uint32_t *op0_f1,
                                   uint32_t *oq0_f1, uint32_t *oq1_f1,
                                   uint32_t *oq2_f1) {
  /* use a 7 tap filter [1, 1, 1, 2, 1, 1, 1] for flat line */
  uint32_t res_op2, res_op1, res_op0;
  uint32_t res_oq0, res_oq1, res_oq2;
  uint32_t tmp;
  uint32_t add_p210_q012;
  uint32_t u32Four = 0x00040004;

  /* *op2 = ROUND_POWER_OF_TWO(p3 + p3 + p3 + p2 + p2 + p1 + p0 + q0, 3)   1 */
  /* *op1 = ROUND_POWER_OF_TWO(p3 + p3 + p2 + p1 + p1 + p0 + q0 + q1, 3)   2 */
  /* *op0 = ROUND_POWER_OF_TWO(p3 + p2 + p1 + p0 + p0 + q0 + q1 + q2, 3)   3 */
  /* *oq0 = ROUND_POWER_OF_TWO(p2 + p1 + p0 + q0 + q0 + q1 + q2 + q3, 3)   4 */
  /* *oq1 = ROUND_POWER_OF_TWO(p1 + p0 + q0 + q1 + q1 + q2 + q3 + q3, 3)   5 */
  /* *oq2 = ROUND_POWER_OF_TWO(p0 + q0 + q1 + q2 + q2 + q3 + q3 + q3, 3)   6 */

  __asm__ __volatile__(
      "addu.ph    %[add_p210_q012],  %[p2],             %[p1]             \n\t"
      "addu.ph    %[add_p210_q012],  %[add_p210_q012],  %[p0]             \n\t"
      "addu.ph    %[add_p210_q012],  %[add_p210_q012],  %[q0]             \n\t"
      "addu.ph    %[add_p210_q012],  %[add_p210_q012],  %[q1]             \n\t"
      "addu.ph    %[add_p210_q012],  %[add_p210_q012],  %[q2]             \n\t"
      "addu.ph    %[add_p210_q012],  %[add_p210_q012],  %[u32Four]        \n\t"

      "shll.ph    %[tmp],            %[p3],             1                 \n\t"
      "addu.ph    %[res_op2],        %[tmp],            %[p3]             \n\t"
      "addu.ph    %[res_op1],        %[p3],             %[p3]             \n\t"
      "addu.ph    %[res_op2],        %[res_op2],        %[p2]             \n\t"
      "addu.ph    %[res_op1],        %[res_op1],        %[p1]             \n\t"
      "addu.ph    %[res_op2],        %[res_op2],        %[add_p210_q012]  \n\t"
      "addu.ph    %[res_op1],        %[res_op1],        %[add_p210_q012]  \n\t"
      "subu.ph    %[res_op2],        %[res_op2],        %[q1]             \n\t"
      "subu.ph    %[res_op1],        %[res_op1],        %[q2]             \n\t"
      "subu.ph    %[res_op2],        %[res_op2],        %[q2]             \n\t"
      "shrl.ph    %[res_op1],        %[res_op1],        3                 \n\t"
      "shrl.ph    %[res_op2],        %[res_op2],        3                 \n\t"
      "addu.ph    %[res_op0],        %[p3],             %[p0]             \n\t"
      "addu.ph    %[res_oq0],        %[q0],             %[q3]             \n\t"
      "addu.ph    %[res_op0],        %[res_op0],        %[add_p210_q012]  \n\t"
      "addu.ph    %[res_oq0],        %[res_oq0],        %[add_p210_q012]  \n\t"
      "addu.ph    %[res_oq1],        %[q3],             %[q3]             \n\t"
      "shll.ph    %[tmp],            %[q3],             1                 \n\t"
      "addu.ph    %[res_oq1],        %[res_oq1],        %[q1]             \n\t"
      "addu.ph    %[res_oq2],        %[tmp],            %[q3]             \n\t"
      "addu.ph    %[res_oq1],        %[res_oq1],        %[add_p210_q012]  \n\t"
      "addu.ph    %[res_oq2],        %[res_oq2],        %[add_p210_q012]  \n\t"
      "subu.ph    %[res_oq1],        %[res_oq1],        %[p2]             \n\t"
      "addu.ph    %[res_oq2],        %[res_oq2],        %[q2]             \n\t"
      "shrl.ph    %[res_oq1],        %[res_oq1],        3                 \n\t"
      "subu.ph    %[res_oq2],        %[res_oq2],        %[p2]             \n\t"
      "shrl.ph    %[res_oq0],        %[res_oq0],        3                 \n\t"
      "subu.ph    %[res_oq2],        %[res_oq2],        %[p1]             \n\t"
      "shrl.ph    %[res_op0],        %[res_op0],        3                 \n\t"
      "shrl.ph    %[res_oq2],        %[res_oq2],        3                 \n\t"

      : [add_p210_q012] "=&r"(add_p210_q012), [tmp] "=&r"(tmp),
        [res_op2] "=&r"(res_op2), [res_op1] "=&r"(res_op1),
        [res_op0] "=&r"(res_op0), [res_oq0] "=&r"(res_oq0),
        [res_oq1] "=&r"(res_oq1), [res_oq2] "=&r"(res_oq2)
      : [p0] "r"(p0), [q0] "r"(q0), [p1] "r"(p1), [q1] "r"(q1), [p2] "r"(p2),
        [q2] "r"(q2), [p3] "r"(p3), [q3] "r"(q3), [u32Four] "r"(u32Four));

  *op2_f1 = res_op2;
  *op1_f1 = res_op1;
  *op0_f1 = res_op0;
  *oq0_f1 = res_oq0;
  *oq1_f1 = res_oq1;
  *oq2_f1 = res_oq2;
}

static INLINE void wide_mbfilter_dspr2(
    uint32_t *op7, uint32_t *op6, uint32_t *op5, uint32_t *op4, uint32_t *op3,
    uint32_t *op2, uint32_t *op1, uint32_t *op0, uint32_t *oq0, uint32_t *oq1,
    uint32_t *oq2, uint32_t *oq3, uint32_t *oq4, uint32_t *oq5, uint32_t *oq6,
    uint32_t *oq7) {
  const uint32_t p7 = *op7, p6 = *op6, p5 = *op5, p4 = *op4;
  const uint32_t p3 = *op3, p2 = *op2, p1 = *op1, p0 = *op0;
  const uint32_t q0 = *oq0, q1 = *oq1, q2 = *oq2, q3 = *oq3;
  const uint32_t q4 = *oq4, q5 = *oq5, q6 = *oq6, q7 = *oq7;
  uint32_t res_op6, res_op5, res_op4, res_op3, res_op2, res_op1, res_op0;
  uint32_t res_oq0, res_oq1, res_oq2, res_oq3, res_oq4, res_oq5, res_oq6;
  uint32_t tmp;
  uint32_t add_p6toq6;
  uint32_t u32Eight = 0x00080008;

  __asm__ __volatile__(
      /* addition of p6,p5,p4,p3,p2,p1,p0,q0,q1,q2,q3,q4,q5,q6
         which is used most of the time */
      "addu.ph      %[add_p6toq6],     %[p6],              %[p5]         \n\t"
      "addu.ph      %[add_p6toq6],     %[add_p6toq6],      %[p4]         \n\t"
      "addu.ph      %[add_p6toq6],     %[add_p6toq6],      %[p3]         \n\t"
      "addu.ph      %[add_p6toq6],     %[add_p6toq6],      %[p2]         \n\t"
      "addu.ph      %[add_p6toq6],     %[add_p6toq6],      %[p1]         \n\t"
      "addu.ph      %[add_p6toq6],     %[add_p6toq6],      %[p0]         \n\t"
      "addu.ph      %[add_p6toq6],     %[add_p6toq6],      %[q0]         \n\t"
      "addu.ph      %[add_p6toq6],     %[add_p6toq6],      %[q1]         \n\t"
      "addu.ph      %[add_p6toq6],     %[add_p6toq6],      %[q2]         \n\t"
      "addu.ph      %[add_p6toq6],     %[add_p6toq6],      %[q3]         \n\t"
      "addu.ph      %[add_p6toq6],     %[add_p6toq6],      %[q4]         \n\t"
      "addu.ph      %[add_p6toq6],     %[add_p6toq6],      %[q5]         \n\t"
      "addu.ph      %[add_p6toq6],     %[add_p6toq6],      %[q6]         \n\t"
      "addu.ph      %[add_p6toq6],     %[add_p6toq6],      %[u32Eight]   \n\t"

      : [add_p6toq6] "=&r"(add_p6toq6)
      : [p6] "r"(p6), [p5] "r"(p5), [p4] "r"(p4), [p3] "r"(p3), [p2] "r"(p2),
        [p1] "r"(p1), [p0] "r"(p0), [q0] "r"(q0), [q1] "r"(q1), [q2] "r"(q2),
        [q3] "r"(q3), [q4] "r"(q4), [q5] "r"(q5), [q6] "r"(q6),
        [u32Eight] "r"(u32Eight));

  __asm__ __volatile__(
      /* *op6 = ROUND_POWER_OF_TWO(p7 * 7 + p6 * 2 + p5 + p4 +
                                   p3 + p2 + p1 + p0 + q0, 4) */
      "shll.ph       %[tmp],            %[p7],            3               \n\t"
      "subu.ph       %[res_op6],        %[tmp],           %[p7]           \n\t"
      "addu.ph       %[res_op6],        %[res_op6],       %[p6]           \n\t"
      "addu.ph       %[res_op6],        %[res_op6],       %[add_p6toq6]   \n\t"
      "subu.ph       %[res_op6],        %[res_op6],       %[q1]           \n\t"
      "subu.ph       %[res_op6],        %[res_op6],       %[q2]           \n\t"
      "subu.ph       %[res_op6],        %[res_op6],       %[q3]           \n\t"
      "subu.ph       %[res_op6],        %[res_op6],       %[q4]           \n\t"
      "subu.ph       %[res_op6],        %[res_op6],       %[q5]           \n\t"
      "subu.ph       %[res_op6],        %[res_op6],       %[q6]           \n\t"
      "shrl.ph       %[res_op6],        %[res_op6],       4               \n\t"

      /* *op5 = ROUND_POWER_OF_TWO(p7 * 6 + p6 + p5 * 2 + p4 + p3 +
                                   p2 + p1 + p0 + q0 + q1, 4) */
      "shll.ph       %[tmp],            %[p7],            2               \n\t"
      "addu.ph       %[res_op5],        %[tmp],           %[p7]           \n\t"
      "addu.ph       %[res_op5],        %[res_op5],       %[p7]           \n\t"
      "addu.ph       %[res_op5],        %[res_op5],       %[p5]           \n\t"
      "addu.ph       %[res_op5],        %[res_op5],       %[add_p6toq6]   \n\t"
      "subu.ph       %[res_op5],        %[res_op5],       %[q2]           \n\t"
      "subu.ph       %[res_op5],        %[res_op5],       %[q3]           \n\t"
      "subu.ph       %[res_op5],        %[res_op5],       %[q4]           \n\t"
      "subu.ph       %[res_op5],        %[res_op5],       %[q5]           \n\t"
      "subu.ph       %[res_op5],        %[res_op5],       %[q6]           \n\t"
      "shrl.ph       %[res_op5],        %[res_op5],       4               \n\t"

      /* *op4 = ROUND_POWER_OF_TWO(p7 * 5 + p6 + p5 + p4 * 2 + p3 + p2 +
                                   p1 + p0 + q0 + q1 + q2, 4) */
      "shll.ph       %[tmp],            %[p7],            2               \n\t"
      "addu.ph       %[res_op4],        %[tmp],           %[p7]           \n\t"
      "addu.ph       %[res_op4],        %[res_op4],       %[p4]           \n\t"
      "addu.ph       %[res_op4],        %[res_op4],       %[add_p6toq6]   \n\t"
      "subu.ph       %[res_op4],        %[res_op4],       %[q3]           \n\t"
      "subu.ph       %[res_op4],        %[res_op4],       %[q4]           \n\t"
      "subu.ph       %[res_op4],        %[res_op4],       %[q5]           \n\t"
      "subu.ph       %[res_op4],        %[res_op4],       %[q6]           \n\t"
      "shrl.ph       %[res_op4],        %[res_op4],       4               \n\t"

      /* *op3 = ROUND_POWER_OF_TWO(p7 * 4 + p6 + p5 + p4 + p3 * 2 + p2 +
                                   p1 + p0 + q0 + q1 + q2 + q3, 4) */
      "shll.ph       %[tmp],            %[p7],            2               \n\t"
      "addu.ph       %[res_op3],        %[tmp],           %[p3]           \n\t"
      "addu.ph       %[res_op3],        %[res_op3],       %[add_p6toq6]   \n\t"
      "subu.ph       %[res_op3],        %[res_op3],       %[q4]           \n\t"
      "subu.ph       %[res_op3],        %[res_op3],       %[q5]           \n\t"
      "subu.ph       %[res_op3],        %[res_op3],       %[q6]           \n\t"
      "shrl.ph       %[res_op3],        %[res_op3],       4               \n\t"

      /* *op2 = ROUND_POWER_OF_TWO(p7 * 3 + p6 + p5 + p4 + p3 + p2 * 2 + p1 +
                                   p0 + q0 + q1 + q2 + q3 + q4, 4) */
      "shll.ph       %[tmp],            %[p7],            1               \n\t"
      "addu.ph       %[res_op2],        %[tmp],           %[p7]           \n\t"
      "addu.ph       %[res_op2],        %[res_op2],       %[p2]           \n\t"
      "addu.ph       %[res_op2],        %[res_op2],       %[add_p6toq6]   \n\t"
      "subu.ph       %[res_op2],        %[res_op2],       %[q5]           \n\t"
      "subu.ph       %[res_op2],        %[res_op2],       %[q6]           \n\t"
      "shrl.ph       %[res_op2],        %[res_op2],       4               \n\t"

      /* *op1 = ROUND_POWER_OF_TWO(p7 * 2 + p6 + p5 + p4 + p3 + p2 + p1 * 2 +
                                   p0 + q0 + q1 + q2 + q3 + q4 + q5, 4); */
      "shll.ph       %[tmp],            %[p7],            1               \n\t"
      "addu.ph       %[res_op1],        %[tmp],           %[p1]           \n\t"
      "addu.ph       %[res_op1],        %[res_op1],       %[add_p6toq6]   \n\t"
      "subu.ph       %[res_op1],        %[res_op1],       %[q6]           \n\t"
      "shrl.ph       %[res_op1],        %[res_op1],       4               \n\t"

      /* *op0 = ROUND_POWER_OF_TWO(p7 + p6 + p5 + p4 + p3 + p2 + p1 + p0 * 2 +
                                  q0 + q1 + q2 + q3 + q4 + q5 + q6, 4) */
      "addu.ph       %[res_op0],        %[p7],            %[p0]           \n\t"
      "addu.ph       %[res_op0],        %[res_op0],       %[add_p6toq6]   \n\t"
      "shrl.ph       %[res_op0],        %[res_op0],       4               \n\t"

      : [res_op6] "=&r"(res_op6), [res_op5] "=&r"(res_op5),
        [res_op4] "=&r"(res_op4), [res_op3] "=&r"(res_op3),
        [res_op2] "=&r"(res_op2), [res_op1] "=&r"(res_op1),
        [res_op0] "=&r"(res_op0), [tmp] "=&r"(tmp)
      : [p7] "r"(p7), [p6] "r"(p6), [p5] "r"(p5), [p4] "r"(p4), [p3] "r"(p3),
        [p2] "r"(p2), [p1] "r"(p1), [p0] "r"(p0), [q2] "r"(q2), [q1] "r"(q1),
        [q3] "r"(q3), [q4] "r"(q4), [q5] "r"(q5), [q6] "r"(q6),
        [add_p6toq6] "r"(add_p6toq6));

  *op6 = res_op6;
  *op5 = res_op5;
  *op4 = res_op4;
  *op3 = res_op3;
  *op2 = res_op2;
  *op1 = res_op1;
  *op0 = res_op0;

  __asm__ __volatile__(
      /* *oq0 = ROUND_POWER_OF_TWO(p6 + p5 + p4 + p3 + p2 + p1 + p0 + q0 * 2 +
                                   q1 + q2 + q3 + q4 + q5 + q6 + q7, 4); */
      "addu.ph       %[res_oq0],        %[q7],            %[q0]           \n\t"
      "addu.ph       %[res_oq0],        %[res_oq0],       %[add_p6toq6]   \n\t"
      "shrl.ph       %[res_oq0],        %[res_oq0],       4               \n\t"

      /* *oq1 = ROUND_POWER_OF_TWO(p5 + p4 + p3 + p2 + p1 + p0 + q0 + q1 * 2 +
                                   q2 + q3 + q4 + q5 + q6 + q7 * 2, 4) */
      "shll.ph       %[tmp],            %[q7],            1               \n\t"
      "addu.ph       %[res_oq1],        %[tmp],           %[q1]           \n\t"
      "addu.ph       %[res_oq1],        %[res_oq1],       %[add_p6toq6]   \n\t"
      "subu.ph       %[res_oq1],        %[res_oq1],       %[p6]           \n\t"
      "shrl.ph       %[res_oq1],        %[res_oq1],       4               \n\t"

      /* *oq2 = ROUND_POWER_OF_TWO(p4 + p3 + p2 + p1 + p0 + q0 + q1 + q2 * 2 +
                                   q3 + q4 + q5 + q6 + q7 * 3, 4) */
      "shll.ph       %[tmp],            %[q7],            1               \n\t"
      "addu.ph       %[res_oq2],        %[tmp],           %[q7]           \n\t"
      "addu.ph       %[res_oq2],        %[res_oq2],       %[q2]           \n\t"
      "addu.ph       %[res_oq2],        %[res_oq2],       %[add_p6toq6]   \n\t"
      "subu.ph       %[res_oq2],        %[res_oq2],       %[p5]           \n\t"
      "subu.ph       %[res_oq2],        %[res_oq2],       %[p6]           \n\t"
      "shrl.ph       %[res_oq2],        %[res_oq2],       4               \n\t"

      /* *oq3 = ROUND_POWER_OF_TWO(p3 + p2 + p1 + p0 + q0 + q1 + q2 +
                                   q3 * 2 + q4 + q5 + q6 + q7 * 4, 4) */
      "shll.ph       %[tmp],            %[q7],            2               \n\t"
      "addu.ph       %[res_oq3],        %[tmp],           %[q3]           \n\t"
      "addu.ph       %[res_oq3],        %[res_oq3],       %[add_p6toq6]   \n\t"
      "subu.ph       %[res_oq3],        %[res_oq3],       %[p4]           \n\t"
      "subu.ph       %[res_oq3],        %[res_oq3],       %[p5]           \n\t"
      "subu.ph       %[res_oq3],        %[res_oq3],       %[p6]           \n\t"
      "shrl.ph       %[res_oq3],        %[res_oq3],       4               \n\t"

      /* *oq4 = ROUND_POWER_OF_TWO(p2 + p1 + p0 + q0 + q1 + q2 + q3 +
                                   q4 * 2 + q5 + q6 + q7 * 5, 4) */
      "shll.ph       %[tmp],            %[q7],            2               \n\t"
      "addu.ph       %[res_oq4],        %[tmp],           %[q7]           \n\t"
      "addu.ph       %[res_oq4],        %[res_oq4],       %[q4]           \n\t"
      "addu.ph       %[res_oq4],        %[res_oq4],       %[add_p6toq6]   \n\t"
      "subu.ph       %[res_oq4],        %[res_oq4],       %[p3]           \n\t"
      "subu.ph       %[res_oq4],        %[res_oq4],       %[p4]           \n\t"
      "subu.ph       %[res_oq4],        %[res_oq4],       %[p5]           \n\t"
      "subu.ph       %[res_oq4],        %[res_oq4],       %[p6]           \n\t"
      "shrl.ph       %[res_oq4],        %[res_oq4],       4               \n\t"

      /* *oq5 = ROUND_POWER_OF_TWO(p1 + p0 + q0 + q1 + q2 + q3 + q4 +
                                   q5 * 2 + q6 + q7 * 6, 4) */
      "shll.ph       %[tmp],            %[q7],            2               \n\t"
      "addu.ph       %[res_oq5],        %[tmp],           %[q7]           \n\t"
      "addu.ph       %[res_oq5],        %[res_oq5],       %[q7]           \n\t"
      "addu.ph       %[res_oq5],        %[res_oq5],       %[q5]           \n\t"
      "addu.ph       %[res_oq5],        %[res_oq5],       %[add_p6toq6]   \n\t"
      "subu.ph       %[res_oq5],        %[res_oq5],       %[p2]           \n\t"
      "subu.ph       %[res_oq5],        %[res_oq5],       %[p3]           \n\t"
      "subu.ph       %[res_oq5],        %[res_oq5],       %[p4]           \n\t"
      "subu.ph       %[res_oq5],        %[res_oq5],       %[p5]           \n\t"
      "subu.ph       %[res_oq5],        %[res_oq5],       %[p6]           \n\t"
      "shrl.ph       %[res_oq5],        %[res_oq5],       4               \n\t"

      /* *oq6 = ROUND_POWER_OF_TWO(p0 + q0 + q1 + q2 + q3 +
                                   q4 + q5 + q6 * 2 + q7 * 7, 4) */
      "shll.ph       %[tmp],            %[q7],            3               \n\t"
      "subu.ph       %[res_oq6],        %[tmp],           %[q7]           \n\t"
      "addu.ph       %[res_oq6],        %[res_oq6],       %[q6]           \n\t"
      "addu.ph       %[res_oq6],        %[res_oq6],       %[add_p6toq6]   \n\t"
      "subu.ph       %[res_oq6],        %[res_oq6],       %[p1]           \n\t"
      "subu.ph       %[res_oq6],        %[res_oq6],       %[p2]           \n\t"
      "subu.ph       %[res_oq6],        %[res_oq6],       %[p3]           \n\t"
      "subu.ph       %[res_oq6],        %[res_oq6],       %[p4]           \n\t"
      "subu.ph       %[res_oq6],        %[res_oq6],       %[p5]           \n\t"
      "subu.ph       %[res_oq6],        %[res_oq6],       %[p6]           \n\t"
      "shrl.ph       %[res_oq6],        %[res_oq6],       4               \n\t"

      : [res_oq6] "=&r"(res_oq6), [res_oq5] "=&r"(res_oq5),
        [res_oq4] "=&r"(res_oq4), [res_oq3] "=&r"(res_oq3),
        [res_oq2] "=&r"(res_oq2), [res_oq1] "=&r"(res_oq1),
        [res_oq0] "=&r"(res_oq0), [tmp] "=&r"(tmp)
      : [q7] "r"(q7), [q6] "r"(q6), [q5] "r"(q5), [q4] "r"(q4), [q3] "r"(q3),
        [q2] "r"(q2), [q1] "r"(q1), [q0] "r"(q0), [p1] "r"(p1), [p2] "r"(p2),
        [p3] "r"(p3), [p4] "r"(p4), [p5] "r"(p5), [p6] "r"(p6),
        [add_p6toq6] "r"(add_p6toq6));

  *oq0 = res_oq0;
  *oq1 = res_oq1;
  *oq2 = res_oq2;
  *oq3 = res_oq3;
  *oq4 = res_oq4;
  *oq5 = res_oq5;
  *oq6 = res_oq6;
}
#endif  // #if HAVE_DSPR2
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VPX_DSP_MIPS_LOOPFILTER_FILTERS_DSPR2_H_
