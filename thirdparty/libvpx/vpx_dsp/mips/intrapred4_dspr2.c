/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "vpx_dsp/mips/common_dspr2.h"

#if HAVE_DSPR2
void vpx_h_predictor_4x4_dspr2(uint8_t *dst, ptrdiff_t stride,
                               const uint8_t *above, const uint8_t *left) {
  int32_t tmp1, tmp2, tmp3, tmp4;
  (void)above;

  __asm__ __volatile__(
      "lb         %[tmp1],      (%[left])                    \n\t"
      "lb         %[tmp2],      1(%[left])                   \n\t"
      "lb         %[tmp3],      2(%[left])                   \n\t"
      "lb         %[tmp4],      3(%[left])                   \n\t"
      "replv.qb   %[tmp1],      %[tmp1]                      \n\t"
      "replv.qb   %[tmp2],      %[tmp2]                      \n\t"
      "replv.qb   %[tmp3],      %[tmp3]                      \n\t"
      "replv.qb   %[tmp4],      %[tmp4]                      \n\t"
      "sw         %[tmp1],      (%[dst])                     \n\t"
      "add        %[dst],       %[dst],         %[stride]    \n\t"
      "sw         %[tmp2],      (%[dst])                     \n\t"
      "add        %[dst],       %[dst],         %[stride]    \n\t"
      "sw         %[tmp3],      (%[dst])                     \n\t"
      "add        %[dst],       %[dst],         %[stride]    \n\t"
      "sw         %[tmp4],      (%[dst])                     \n\t"

      : [tmp1] "=&r"(tmp1), [tmp2] "=&r"(tmp2), [tmp3] "=&r"(tmp3),
        [tmp4] "=&r"(tmp4)
      : [left] "r"(left), [dst] "r"(dst), [stride] "r"(stride));
}

void vpx_dc_predictor_4x4_dspr2(uint8_t *dst, ptrdiff_t stride,
                                const uint8_t *above, const uint8_t *left) {
  int32_t expected_dc;
  int32_t average;
  int32_t tmp, above_c, above_l, above_r, left_c, left_r, left_l;

  __asm__ __volatile__(
      "lw              %[above_c],         (%[above])                    \n\t"
      "lw              %[left_c],          (%[left])                     \n\t"

      "preceu.ph.qbl   %[above_l],         %[above_c]                    \n\t"
      "preceu.ph.qbr   %[above_r],         %[above_c]                    \n\t"
      "preceu.ph.qbl   %[left_l],          %[left_c]                     \n\t"
      "preceu.ph.qbr   %[left_r],          %[left_c]                     \n\t"

      "addu.ph         %[average],         %[above_r],       %[above_l]  \n\t"
      "addu.ph         %[average],         %[average],       %[left_l]   \n\t"
      "addu.ph         %[average],         %[average],       %[left_r]   \n\t"
      "addiu           %[average],         %[average],       4           \n\t"
      "srl             %[tmp],             %[average],       16          \n\t"
      "addu.ph         %[average],         %[tmp],           %[average]  \n\t"
      "srl             %[expected_dc],     %[average],       3           \n\t"
      "replv.qb        %[expected_dc],     %[expected_dc]                \n\t"

      "sw              %[expected_dc],     (%[dst])                      \n\t"
      "add             %[dst],              %[dst],          %[stride]   \n\t"
      "sw              %[expected_dc],     (%[dst])                      \n\t"
      "add             %[dst],              %[dst],          %[stride]   \n\t"
      "sw              %[expected_dc],     (%[dst])                      \n\t"
      "add             %[dst],              %[dst],          %[stride]   \n\t"
      "sw              %[expected_dc],     (%[dst])                      \n\t"

      : [above_c] "=&r"(above_c), [above_l] "=&r"(above_l),
        [above_r] "=&r"(above_r), [left_c] "=&r"(left_c),
        [left_l] "=&r"(left_l), [left_r] "=&r"(left_r),
        [average] "=&r"(average), [tmp] "=&r"(tmp),
        [expected_dc] "=&r"(expected_dc)
      : [above] "r"(above), [left] "r"(left), [dst] "r"(dst),
        [stride] "r"(stride));
}

void vpx_tm_predictor_4x4_dspr2(uint8_t *dst, ptrdiff_t stride,
                                const uint8_t *above, const uint8_t *left) {
  int32_t abovel, abover;
  int32_t left0, left1, left2, left3;
  int32_t res0, res1;
  int32_t resl;
  int32_t resr;
  int32_t top_left;
  uint8_t *cm = vpx_ff_cropTbl;

  __asm__ __volatile__(
      "ulw             %[resl],       (%[above])                         \n\t"

      "lbu             %[left0],       (%[left])                         \n\t"
      "lbu             %[left1],       1(%[left])                        \n\t"
      "lbu             %[left2],       2(%[left])                        \n\t"
      "lbu             %[left3],       3(%[left])                        \n\t"

      "lbu             %[top_left],    -1(%[above])                      \n\t"

      "preceu.ph.qbl   %[abovel],      %[resl]                           \n\t"
      "preceu.ph.qbr   %[abover],      %[resl]                           \n\t"

      "replv.ph        %[left0],       %[left0]                          \n\t"
      "replv.ph        %[left1],       %[left1]                          \n\t"
      "replv.ph        %[left2],       %[left2]                          \n\t"
      "replv.ph        %[left3],       %[left3]                          \n\t"

      "replv.ph        %[top_left],    %[top_left]                       \n\t"

      "addu.ph         %[resl],        %[abovel],         %[left0]       \n\t"
      "subu.ph         %[resl],        %[resl],           %[top_left]    \n\t"

      "addu.ph         %[resr],        %[abover],         %[left0]       \n\t"
      "subu.ph         %[resr],        %[resr],           %[top_left]    \n\t"

      "sll             %[res0],        %[resr],           16             \n\t"
      "sra             %[res0],        %[res0],           16             \n\t"
      "lbux            %[res0],        %[res0](%[cm])                    \n\t"

      "sra             %[res1],        %[resr],           16             \n\t"
      "lbux            %[res1],        %[res1](%[cm])                    \n\t"
      "sb              %[res0],        (%[dst])                          \n\t"

      "sll             %[res0],        %[resl],           16             \n\t"
      "sra             %[res0],        %[res0],           16             \n\t"
      "lbux            %[res0],        %[res0](%[cm])                    \n\t"
      "sb              %[res1],        1(%[dst])                         \n\t"

      "sra             %[res1],        %[resl],           16             \n\t"
      "lbux            %[res1],        %[res1](%[cm])                    \n\t"

      "addu.ph         %[resl],        %[abovel],         %[left1]       \n\t"
      "subu.ph         %[resl],        %[resl],           %[top_left]    \n\t"

      "addu.ph         %[resr],        %[abover],         %[left1]       \n\t"
      "subu.ph         %[resr],        %[resr],           %[top_left]    \n\t"

      "sb              %[res0],        2(%[dst])                         \n\t"
      "sb              %[res1],        3(%[dst])                         \n\t"

      "add             %[dst],          %[dst],           %[stride]      \n\t"

      "sll             %[res0],        %[resr],           16             \n\t"
      "sra             %[res0],        %[res0],           16             \n\t"
      "lbux            %[res0],        %[res0](%[cm])                    \n\t"

      "sra             %[res1],        %[resr],           16             \n\t"
      "lbux            %[res1],        %[res1](%[cm])                    \n\t"
      "sb              %[res0],        (%[dst])                          \n\t"

      "sll             %[res0],        %[resl],           16             \n\t"
      "sra             %[res0],        %[res0],           16             \n\t"
      "lbux            %[res0],        %[res0](%[cm])                    \n\t"

      "sb              %[res1],        1(%[dst])                         \n\t"
      "sra             %[res1],        %[resl],           16             \n\t"
      "lbux            %[res1],        %[res1](%[cm])                    \n\t"

      "addu.ph         %[resl],        %[abovel],         %[left2]       \n\t"
      "subu.ph         %[resl],        %[resl],           %[top_left]    \n\t"

      "addu.ph         %[resr],        %[abover],         %[left2]       \n\t"
      "subu.ph         %[resr],        %[resr],           %[top_left]    \n\t"

      "sb              %[res0],        2(%[dst])                         \n\t"
      "sb              %[res1],        3(%[dst])                         \n\t"

      "add             %[dst],          %[dst],           %[stride]      \n\t"

      "sll             %[res0],        %[resr],           16             \n\t"
      "sra             %[res0],        %[res0],           16             \n\t"
      "lbux            %[res0],        %[res0](%[cm])                    \n\t"

      "sra             %[res1],        %[resr],           16             \n\t"
      "lbux            %[res1],        %[res1](%[cm])                    \n\t"
      "sb              %[res0],        (%[dst])                          \n\t"

      "sll             %[res0],        %[resl],           16             \n\t"
      "sra             %[res0],        %[res0],           16             \n\t"
      "lbux            %[res0],        %[res0](%[cm])                    \n\t"

      "sb              %[res1],        1(%[dst])                         \n\t"
      "sra             %[res1],        %[resl],           16             \n\t"
      "lbux            %[res1],        %[res1](%[cm])                    \n\t"

      "addu.ph         %[resl],        %[abovel],        %[left3]        \n\t"
      "subu.ph         %[resl],        %[resl],          %[top_left]     \n\t"

      "addu.ph         %[resr],        %[abover],        %[left3]        \n\t"
      "subu.ph         %[resr],        %[resr],          %[top_left]     \n\t"

      "sb              %[res0],        2(%[dst])                         \n\t"
      "sb              %[res1],        3(%[dst])                         \n\t"

      "add             %[dst],          %[dst],          %[stride]       \n\t"

      "sll             %[res0],        %[resr],           16             \n\t"
      "sra             %[res0],        %[res0],           16             \n\t"
      "lbux            %[res0],        %[res0](%[cm])                    \n\t"

      "sra             %[res1],        %[resr],           16             \n\t"
      "lbux            %[res1],        %[res1](%[cm])                    \n\t"
      "sb              %[res0],        (%[dst])                          \n\t"

      "sll             %[res0],        %[resl],           16             \n\t"
      "sra             %[res0],        %[res0],           16             \n\t"
      "lbux            %[res0],        %[res0](%[cm])                    \n\t"
      "sb              %[res1],        1(%[dst])                         \n\t"

      "sra             %[res1],        %[resl],           16             \n\t"
      "lbux            %[res1],        %[res1](%[cm])                    \n\t"

      "sb              %[res0],        2(%[dst])                         \n\t"
      "sb              %[res1],        3(%[dst])                         \n\t"

      : [abovel] "=&r"(abovel), [abover] "=&r"(abover), [left0] "=&r"(left0),
        [left1] "=&r"(left1), [left2] "=&r"(left2), [res0] "=&r"(res0),
        [res1] "=&r"(res1), [left3] "=&r"(left3), [resl] "=&r"(resl),
        [resr] "=&r"(resr), [top_left] "=&r"(top_left)
      : [above] "r"(above), [left] "r"(left), [dst] "r"(dst),
        [stride] "r"(stride), [cm] "r"(cm));
}
#endif  // #if HAVE_DSPR2
