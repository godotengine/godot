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
void vpx_h_predictor_8x8_dspr2(uint8_t *dst, ptrdiff_t stride,
                               const uint8_t *above, const uint8_t *left) {
  int32_t tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8;
  (void)above;

  __asm__ __volatile__(
      "lb         %[tmp1],      (%[left])                   \n\t"
      "lb         %[tmp2],      1(%[left])                  \n\t"
      "lb         %[tmp3],      2(%[left])                  \n\t"
      "lb         %[tmp4],      3(%[left])                  \n\t"
      "lb         %[tmp5],      4(%[left])                  \n\t"
      "lb         %[tmp6],      5(%[left])                  \n\t"
      "lb         %[tmp7],      6(%[left])                  \n\t"
      "lb         %[tmp8],      7(%[left])                  \n\t"

      "replv.qb   %[tmp1],      %[tmp1]                     \n\t"
      "replv.qb   %[tmp2],      %[tmp2]                     \n\t"
      "replv.qb   %[tmp3],      %[tmp3]                     \n\t"
      "replv.qb   %[tmp4],      %[tmp4]                     \n\t"
      "replv.qb   %[tmp5],      %[tmp5]                     \n\t"
      "replv.qb   %[tmp6],      %[tmp6]                     \n\t"
      "replv.qb   %[tmp7],      %[tmp7]                     \n\t"
      "replv.qb   %[tmp8],      %[tmp8]                     \n\t"

      "sw         %[tmp1],      (%[dst])                    \n\t"
      "sw         %[tmp1],      4(%[dst])                   \n\t"
      "add        %[dst],       %[dst],         %[stride]   \n\t"
      "sw         %[tmp2],      (%[dst])                    \n\t"
      "sw         %[tmp2],      4(%[dst])                   \n\t"
      "add        %[dst],       %[dst],         %[stride]   \n\t"
      "sw         %[tmp3],      (%[dst])                    \n\t"
      "sw         %[tmp3],      4(%[dst])                   \n\t"
      "add        %[dst],       %[dst],         %[stride]   \n\t"
      "sw         %[tmp4],      (%[dst])                    \n\t"
      "sw         %[tmp4],      4(%[dst])                   \n\t"
      "add        %[dst],       %[dst],         %[stride]   \n\t"
      "sw         %[tmp5],      (%[dst])                    \n\t"
      "sw         %[tmp5],      4(%[dst])                   \n\t"
      "add        %[dst],       %[dst],         %[stride]   \n\t"
      "sw         %[tmp6],      (%[dst])                    \n\t"
      "sw         %[tmp6],      4(%[dst])                   \n\t"
      "add        %[dst],       %[dst],         %[stride]   \n\t"
      "sw         %[tmp7],      (%[dst])                    \n\t"
      "sw         %[tmp7],      4(%[dst])                   \n\t"
      "add        %[dst],       %[dst],         %[stride]   \n\t"
      "sw         %[tmp8],      (%[dst])                    \n\t"
      "sw         %[tmp8],      4(%[dst])                   \n\t"

      : [tmp1] "=&r"(tmp1), [tmp2] "=&r"(tmp2), [tmp3] "=&r"(tmp3),
        [tmp4] "=&r"(tmp4), [tmp5] "=&r"(tmp5), [tmp7] "=&r"(tmp7),
        [tmp6] "=&r"(tmp6), [tmp8] "=&r"(tmp8)
      : [left] "r"(left), [dst] "r"(dst), [stride] "r"(stride));
}

void vpx_dc_predictor_8x8_dspr2(uint8_t *dst, ptrdiff_t stride,
                                const uint8_t *above, const uint8_t *left) {
  int32_t expected_dc;
  int32_t average;
  int32_t tmp, above1, above_l1, above_r1, left1, left_r1, left_l1;
  int32_t above2, above_l2, above_r2, left2, left_r2, left_l2;

  __asm__ __volatile__(
      "lw              %[above1],         (%[above])                      \n\t"
      "lw              %[above2],         4(%[above])                     \n\t"
      "lw              %[left1],          (%[left])                       \n\t"
      "lw              %[left2],          4(%[left])                      \n\t"

      "preceu.ph.qbl   %[above_l1],       %[above1]                       \n\t"
      "preceu.ph.qbr   %[above_r1],       %[above1]                       \n\t"
      "preceu.ph.qbl   %[left_l1],        %[left1]                        \n\t"
      "preceu.ph.qbr   %[left_r1],        %[left1]                        \n\t"

      "preceu.ph.qbl   %[above_l2],       %[above2]                       \n\t"
      "preceu.ph.qbr   %[above_r2],       %[above2]                       \n\t"
      "preceu.ph.qbl   %[left_l2],        %[left2]                        \n\t"
      "preceu.ph.qbr   %[left_r2],        %[left2]                        \n\t"

      "addu.ph         %[average],        %[above_r1],      %[above_l1]   \n\t"
      "addu.ph         %[average],        %[average],       %[left_l1]    \n\t"
      "addu.ph         %[average],        %[average],       %[left_r1]    \n\t"

      "addu.ph         %[average],        %[average],       %[above_l2]   \n\t"
      "addu.ph         %[average],        %[average],       %[above_r2]   \n\t"
      "addu.ph         %[average],        %[average],       %[left_l2]    \n\t"
      "addu.ph         %[average],        %[average],       %[left_r2]    \n\t"

      "addiu           %[average],        %[average],       8             \n\t"

      "srl             %[tmp],            %[average],       16            \n\t"
      "addu.ph         %[average],        %[tmp],           %[average]    \n\t"
      "srl             %[expected_dc],    %[average],       4             \n\t"
      "replv.qb        %[expected_dc],    %[expected_dc]                  \n\t"

      "sw              %[expected_dc],    (%[dst])                        \n\t"
      "sw              %[expected_dc],    4(%[dst])                       \n\t"

      "add             %[dst],             %[dst],          %[stride]     \n\t"
      "sw              %[expected_dc],    (%[dst])                        \n\t"
      "sw              %[expected_dc],    4(%[dst])                       \n\t"

      "add             %[dst],             %[dst],          %[stride]     \n\t"
      "sw              %[expected_dc],    (%[dst])                        \n\t"
      "sw              %[expected_dc],    4(%[dst])                       \n\t"

      "add             %[dst],             %[dst],          %[stride]     \n\t"
      "sw              %[expected_dc],    (%[dst])                        \n\t"
      "sw              %[expected_dc],    4(%[dst])                       \n\t"

      "add             %[dst],             %[dst],          %[stride]     \n\t"
      "sw              %[expected_dc],    (%[dst])                        \n\t"
      "sw              %[expected_dc],    4(%[dst])                       \n\t"

      "add             %[dst],             %[dst],          %[stride]     \n\t"
      "sw              %[expected_dc],    (%[dst])                        \n\t"
      "sw              %[expected_dc],    4(%[dst])                       \n\t"

      "add             %[dst],             %[dst],          %[stride]     \n\t"
      "sw              %[expected_dc],    (%[dst])                        \n\t"
      "sw              %[expected_dc],    4(%[dst])                       \n\t"

      "add             %[dst],             %[dst],          %[stride]     \n\t"
      "sw              %[expected_dc],    (%[dst])                        \n\t"
      "sw              %[expected_dc],    4(%[dst])                       \n\t"

      : [above1] "=&r"(above1), [above_l1] "=&r"(above_l1),
        [above_r1] "=&r"(above_r1), [left1] "=&r"(left1),
        [left_l1] "=&r"(left_l1), [left_r1] "=&r"(left_r1),
        [above2] "=&r"(above2), [above_l2] "=&r"(above_l2),
        [above_r2] "=&r"(above_r2), [left2] "=&r"(left2),
        [left_l2] "=&r"(left_l2), [left_r2] "=&r"(left_r2),
        [average] "=&r"(average), [tmp] "=&r"(tmp),
        [expected_dc] "=&r"(expected_dc)
      : [above] "r"(above), [left] "r"(left), [dst] "r"(dst),
        [stride] "r"(stride));
}

void vpx_tm_predictor_8x8_dspr2(uint8_t *dst, ptrdiff_t stride,
                                const uint8_t *above, const uint8_t *left) {
  int32_t abovel, abover;
  int32_t abovel_1, abover_1;
  int32_t left0;
  int32_t res0, res1, res2, res3;
  int32_t reshw;
  int32_t top_left;
  uint8_t *cm = vpx_ff_cropTbl;

  __asm__ __volatile__(
      "ulw             %[reshw],       (%[above])                         \n\t"
      "ulw             %[top_left],    4(%[above])                        \n\t"

      "lbu             %[left0],       (%[left])                          \n\t"

      "preceu.ph.qbl   %[abovel],      %[reshw]                           \n\t"
      "preceu.ph.qbr   %[abover],      %[reshw]                           \n\t"
      "preceu.ph.qbl   %[abovel_1],    %[top_left]                        \n\t"
      "preceu.ph.qbr   %[abover_1],    %[top_left]                        \n\t"

      "lbu             %[top_left],    -1(%[above])                       \n\t"
      "replv.ph        %[left0],       %[left0]                           \n\t"

      "replv.ph        %[top_left],    %[top_left]                        \n\t"

      "addu.ph         %[reshw],       %[abovel],           %[left0]      \n\t"
      "subu.ph         %[reshw],       %[reshw],            %[top_left]   \n\t"

      "sll             %[res2],        %[reshw],            16            \n\t"
      "sra             %[res2],        %[res2],             16            \n\t"
      "sra             %[res3],        %[reshw],            16            \n\t"

      "addu.ph         %[reshw],       %[abover],           %[left0]      \n\t"
      "subu.ph         %[reshw],       %[reshw],            %[top_left]   \n\t"

      "sll             %[res0],        %[reshw],            16            \n\t"
      "sra             %[res0],        %[res0],             16            \n\t"
      "sra             %[res1],        %[reshw],            16            \n\t"

      "lbux            %[res0],        %[res0](%[cm])                     \n\t"
      "lbux            %[res1],        %[res1](%[cm])                     \n\t"
      "lbux            %[res2],        %[res2](%[cm])                     \n\t"
      "lbux            %[res3],        %[res3](%[cm])                     \n\t"

      "sb              %[res0],        (%[dst])                           \n\t"
      "sb              %[res1],        1(%[dst])                          \n\t"
      "sb              %[res2],        2(%[dst])                          \n\t"
      "sb              %[res3],        3(%[dst])                          \n\t"

      "addu.ph         %[reshw],       %[abovel_1],         %[left0]      \n\t"
      "subu.ph         %[reshw],       %[reshw],            %[top_left]   \n\t"

      "sll             %[res2],        %[reshw],            16            \n\t"
      "sra             %[res2],        %[res2],             16            \n\t"
      "sra             %[res3],        %[reshw],            16            \n\t"

      "addu.ph         %[reshw],       %[abover_1],         %[left0]      \n\t"
      "subu.ph         %[reshw],       %[reshw],            %[top_left]   \n\t"

      "sll             %[res0],        %[reshw],            16            \n\t"
      "sra             %[res0],        %[res0],             16            \n\t"
      "sra             %[res1],        %[reshw],            16            \n\t"

      "lbu             %[left0],       1(%[left])                         \n\t"

      "lbux            %[res0],        %[res0](%[cm])                     \n\t"
      "lbux            %[res1],        %[res1](%[cm])                     \n\t"
      "lbux            %[res2],        %[res2](%[cm])                     \n\t"
      "lbux            %[res3],        %[res3](%[cm])                     \n\t"

      "sb              %[res0],        4(%[dst])                          \n\t"
      "sb              %[res1],        5(%[dst])                          \n\t"
      "sb              %[res2],        6(%[dst])                          \n\t"
      "sb              %[res3],        7(%[dst])                          \n\t"

      "replv.ph        %[left0],       %[left0]                           \n\t"
      "add             %[dst],          %[dst],             %[stride]     \n\t"

      "addu.ph         %[reshw],       %[abovel],           %[left0]      \n\t"
      "subu.ph         %[reshw],       %[reshw],            %[top_left]   \n\t"

      "sll             %[res2],        %[reshw],            16            \n\t"
      "sra             %[res2],        %[res2],             16            \n\t"
      "sra             %[res3],        %[reshw],            16            \n\t"

      "addu.ph         %[reshw],       %[abover],           %[left0]      \n\t"
      "subu.ph         %[reshw],       %[reshw],            %[top_left]   \n\t"

      "sll             %[res0],        %[reshw],            16            \n\t"
      "sra             %[res0],        %[res0],             16            \n\t"
      "sra             %[res1],        %[reshw],            16            \n\t"

      "lbux            %[res0],        %[res0](%[cm])                     \n\t"
      "lbux            %[res1],        %[res1](%[cm])                     \n\t"
      "lbux            %[res2],        %[res2](%[cm])                     \n\t"
      "lbux            %[res3],        %[res3](%[cm])                     \n\t"

      "sb              %[res0],        (%[dst])                           \n\t"
      "sb              %[res1],        1(%[dst])                          \n\t"
      "sb              %[res2],        2(%[dst])                          \n\t"
      "sb              %[res3],        3(%[dst])                          \n\t"

      "addu.ph         %[reshw],       %[abovel_1],         %[left0]      \n\t"
      "subu.ph         %[reshw],       %[reshw],            %[top_left]   \n\t"

      "sll             %[res2],        %[reshw],            16            \n\t"
      "sra             %[res2],        %[res2],             16            \n\t"
      "sra             %[res3],        %[reshw],            16            \n\t"

      "addu.ph         %[reshw],       %[abover_1],         %[left0]      \n\t"
      "subu.ph         %[reshw],       %[reshw],            %[top_left]   \n\t"

      "sll             %[res0],        %[reshw],            16            \n\t"
      "sra             %[res0],        %[res0],             16            \n\t"
      "sra             %[res1],        %[reshw],            16            \n\t"

      "lbu             %[left0],       2(%[left])                         \n\t"

      "lbux            %[res0],        %[res0](%[cm])                     \n\t"
      "lbux            %[res1],        %[res1](%[cm])                     \n\t"
      "lbux            %[res2],        %[res2](%[cm])                     \n\t"
      "lbux            %[res3],        %[res3](%[cm])                     \n\t"

      "sb              %[res0],        4(%[dst])                          \n\t"
      "sb              %[res1],        5(%[dst])                          \n\t"
      "sb              %[res2],        6(%[dst])                          \n\t"
      "sb              %[res3],        7(%[dst])                          \n\t"

      "replv.ph        %[left0],       %[left0]                           \n\t"
      "add             %[dst],          %[dst],             %[stride]     \n\t"

      "addu.ph         %[reshw],       %[abovel],           %[left0]      \n\t"
      "subu.ph         %[reshw],       %[reshw],            %[top_left]   \n\t"

      "sll             %[res2],        %[reshw],            16            \n\t"
      "sra             %[res2],        %[res2],             16            \n\t"
      "sra             %[res3],        %[reshw],            16            \n\t"

      "addu.ph         %[reshw],       %[abover],           %[left0]      \n\t"
      "subu.ph         %[reshw],       %[reshw],            %[top_left]   \n\t"

      "sll             %[res0],        %[reshw],            16            \n\t"
      "sra             %[res0],        %[res0],             16            \n\t"
      "sra             %[res1],        %[reshw],            16            \n\t"

      "lbux            %[res0],        %[res0](%[cm])                     \n\t"
      "lbux            %[res1],        %[res1](%[cm])                     \n\t"
      "lbux            %[res2],        %[res2](%[cm])                     \n\t"
      "lbux            %[res3],        %[res3](%[cm])                     \n\t"

      "sb              %[res0],        (%[dst])                           \n\t"
      "sb              %[res1],        1(%[dst])                          \n\t"
      "sb              %[res2],        2(%[dst])                          \n\t"
      "sb              %[res3],        3(%[dst])                          \n\t"

      "addu.ph         %[reshw],       %[abovel_1],         %[left0]      \n\t"
      "subu.ph         %[reshw],       %[reshw],            %[top_left]   \n\t"

      "sll             %[res2],        %[reshw],            16            \n\t"
      "sra             %[res2],        %[res2],             16            \n\t"
      "sra             %[res3],        %[reshw],            16            \n\t"

      "addu.ph         %[reshw],       %[abover_1],         %[left0]      \n\t"
      "subu.ph         %[reshw],       %[reshw],            %[top_left]   \n\t"

      "sll             %[res0],        %[reshw],            16            \n\t"
      "sra             %[res0],        %[res0],             16            \n\t"
      "sra             %[res1],        %[reshw],            16            \n\t"

      "lbu             %[left0],       3(%[left])                         \n\t"

      "lbux            %[res0],        %[res0](%[cm])                     \n\t"
      "lbux            %[res1],        %[res1](%[cm])                     \n\t"
      "lbux            %[res2],        %[res2](%[cm])                     \n\t"
      "lbux            %[res3],        %[res3](%[cm])                     \n\t"

      "sb              %[res0],        4(%[dst])                          \n\t"
      "sb              %[res1],        5(%[dst])                          \n\t"
      "sb              %[res2],        6(%[dst])                          \n\t"
      "sb              %[res3],        7(%[dst])                          \n\t"

      "replv.ph        %[left0],       %[left0]                           \n\t"
      "add             %[dst],          %[dst],             %[stride]     \n\t"

      "addu.ph         %[reshw],       %[abovel],           %[left0]      \n\t"
      "subu.ph         %[reshw],       %[reshw],            %[top_left]   \n\t"

      "sll             %[res2],        %[reshw],            16            \n\t"
      "sra             %[res2],        %[res2],             16            \n\t"
      "sra             %[res3],        %[reshw],            16            \n\t"

      "addu.ph         %[reshw],       %[abover],           %[left0]      \n\t"
      "subu.ph         %[reshw],       %[reshw],            %[top_left]   \n\t"

      "sll             %[res0],        %[reshw],            16            \n\t"
      "sra             %[res0],        %[res0],             16            \n\t"
      "sra             %[res1],        %[reshw],            16            \n\t"

      "lbux            %[res0],        %[res0](%[cm])                     \n\t"
      "lbux            %[res1],        %[res1](%[cm])                     \n\t"
      "lbux            %[res2],        %[res2](%[cm])                     \n\t"
      "lbux            %[res3],        %[res3](%[cm])                     \n\t"

      "sb              %[res0],        (%[dst])                           \n\t"
      "sb              %[res1],        1(%[dst])                          \n\t"
      "sb              %[res2],        2(%[dst])                          \n\t"
      "sb              %[res3],        3(%[dst])                          \n\t"

      "addu.ph         %[reshw],       %[abovel_1],         %[left0]      \n\t"
      "subu.ph         %[reshw],       %[reshw],            %[top_left]   \n\t"

      "sll             %[res2],        %[reshw],            16            \n\t"
      "sra             %[res2],        %[res2],             16            \n\t"
      "sra             %[res3],        %[reshw],            16            \n\t"

      "addu.ph         %[reshw],       %[abover_1],         %[left0]      \n\t"
      "subu.ph         %[reshw],       %[reshw],            %[top_left]   \n\t"

      "sll             %[res0],        %[reshw],            16            \n\t"
      "sra             %[res0],        %[res0],             16            \n\t"
      "sra             %[res1],        %[reshw],            16            \n\t"

      "lbu             %[left0],       4(%[left])                         \n\t"

      "lbux            %[res0],        %[res0](%[cm])                     \n\t"
      "lbux            %[res1],        %[res1](%[cm])                     \n\t"
      "lbux            %[res2],        %[res2](%[cm])                     \n\t"
      "lbux            %[res3],        %[res3](%[cm])                     \n\t"

      "sb              %[res0],        4(%[dst])                          \n\t"
      "sb              %[res1],        5(%[dst])                          \n\t"
      "sb              %[res2],        6(%[dst])                          \n\t"
      "sb              %[res3],        7(%[dst])                          \n\t"

      "replv.ph        %[left0],       %[left0]                           \n\t"
      "add             %[dst],          %[dst],             %[stride]     \n\t"

      "addu.ph         %[reshw],       %[abovel],           %[left0]      \n\t"
      "subu.ph         %[reshw],       %[reshw],            %[top_left]   \n\t"

      "sll             %[res2],        %[reshw],            16            \n\t"
      "sra             %[res2],        %[res2],             16            \n\t"
      "sra             %[res3],        %[reshw],            16            \n\t"

      "addu.ph         %[reshw],       %[abover],           %[left0]      \n\t"
      "subu.ph         %[reshw],       %[reshw],            %[top_left]   \n\t"

      "sll             %[res0],        %[reshw],            16            \n\t"
      "sra             %[res0],        %[res0],             16            \n\t"
      "sra             %[res1],        %[reshw],            16            \n\t"

      "lbux            %[res0],        %[res0](%[cm])                     \n\t"
      "lbux            %[res1],        %[res1](%[cm])                     \n\t"
      "lbux            %[res2],        %[res2](%[cm])                     \n\t"
      "lbux            %[res3],        %[res3](%[cm])                     \n\t"

      "sb              %[res0],        (%[dst])                           \n\t"
      "sb              %[res1],        1(%[dst])                          \n\t"
      "sb              %[res2],        2(%[dst])                          \n\t"
      "sb              %[res3],        3(%[dst])                          \n\t"

      "addu.ph         %[reshw],       %[abovel_1],         %[left0]      \n\t"
      "subu.ph         %[reshw],       %[reshw],            %[top_left]   \n\t"

      "sll             %[res2],        %[reshw],            16            \n\t"
      "sra             %[res2],        %[res2],             16            \n\t"
      "sra             %[res3],        %[reshw],            16            \n\t"

      "addu.ph         %[reshw],       %[abover_1],         %[left0]      \n\t"
      "subu.ph         %[reshw],       %[reshw],            %[top_left]   \n\t"

      "sll             %[res0],        %[reshw],            16            \n\t"
      "sra             %[res0],        %[res0],             16            \n\t"
      "sra             %[res1],        %[reshw],            16            \n\t"

      "lbu             %[left0],       5(%[left])                         \n\t"

      "lbux            %[res0],        %[res0](%[cm])                     \n\t"
      "lbux            %[res1],        %[res1](%[cm])                     \n\t"
      "lbux            %[res2],        %[res2](%[cm])                     \n\t"
      "lbux            %[res3],        %[res3](%[cm])                     \n\t"

      "sb              %[res0],        4(%[dst])                          \n\t"
      "sb              %[res1],        5(%[dst])                          \n\t"
      "sb              %[res2],        6(%[dst])                          \n\t"
      "sb              %[res3],        7(%[dst])                          \n\t"

      "replv.ph        %[left0],       %[left0]                           \n\t"
      "add             %[dst],          %[dst],             %[stride]     \n\t"

      "addu.ph         %[reshw],       %[abovel],           %[left0]      \n\t"
      "subu.ph         %[reshw],       %[reshw],            %[top_left]   \n\t"

      "sll             %[res2],        %[reshw],            16            \n\t"
      "sra             %[res2],        %[res2],             16            \n\t"
      "sra             %[res3],        %[reshw],            16            \n\t"

      "addu.ph         %[reshw],       %[abover],           %[left0]      \n\t"
      "subu.ph         %[reshw],       %[reshw],            %[top_left]   \n\t"

      "sll             %[res0],        %[reshw],            16            \n\t"
      "sra             %[res0],        %[res0],             16            \n\t"
      "sra             %[res1],        %[reshw],            16            \n\t"

      "lbux            %[res0],        %[res0](%[cm])                     \n\t"
      "lbux            %[res1],        %[res1](%[cm])                     \n\t"
      "lbux            %[res2],        %[res2](%[cm])                     \n\t"
      "lbux            %[res3],        %[res3](%[cm])                     \n\t"

      "sb              %[res0],        (%[dst])                           \n\t"
      "sb              %[res1],        1(%[dst])                          \n\t"
      "sb              %[res2],        2(%[dst])                          \n\t"
      "sb              %[res3],        3(%[dst])                          \n\t"

      "addu.ph         %[reshw],       %[abovel_1],         %[left0]      \n\t"
      "subu.ph         %[reshw],       %[reshw],            %[top_left]   \n\t"

      "sll             %[res2],        %[reshw],            16            \n\t"
      "sra             %[res2],        %[res2],             16            \n\t"
      "sra             %[res3],        %[reshw],            16            \n\t"

      "addu.ph         %[reshw],       %[abover_1],         %[left0]      \n\t"
      "subu.ph         %[reshw],       %[reshw],            %[top_left]   \n\t"

      "sll             %[res0],        %[reshw],            16            \n\t"
      "sra             %[res0],        %[res0],             16            \n\t"
      "sra             %[res1],        %[reshw],            16            \n\t"

      "lbu             %[left0],       6(%[left])                         \n\t"

      "lbux            %[res0],        %[res0](%[cm])                     \n\t"
      "lbux            %[res1],        %[res1](%[cm])                     \n\t"
      "lbux            %[res2],        %[res2](%[cm])                     \n\t"
      "lbux            %[res3],        %[res3](%[cm])                     \n\t"

      "sb              %[res0],        4(%[dst])                          \n\t"
      "sb              %[res1],        5(%[dst])                          \n\t"
      "sb              %[res2],        6(%[dst])                          \n\t"
      "sb              %[res3],        7(%[dst])                          \n\t"

      "replv.ph        %[left0],       %[left0]                           \n\t"
      "add             %[dst],          %[dst],             %[stride]     \n\t"

      "addu.ph         %[reshw],       %[abovel],           %[left0]      \n\t"
      "subu.ph         %[reshw],       %[reshw],            %[top_left]   \n\t"

      "sll             %[res2],        %[reshw],            16            \n\t"
      "sra             %[res2],        %[res2],             16            \n\t"
      "sra             %[res3],        %[reshw],            16            \n\t"

      "addu.ph         %[reshw],       %[abover],           %[left0]      \n\t"
      "subu.ph         %[reshw],       %[reshw],            %[top_left]   \n\t"

      "sll             %[res0],        %[reshw],            16            \n\t"
      "sra             %[res0],        %[res0],             16            \n\t"
      "sra             %[res1],        %[reshw],            16            \n\t"

      "lbux            %[res0],        %[res0](%[cm])                     \n\t"
      "lbux            %[res1],        %[res1](%[cm])                     \n\t"
      "lbux            %[res2],        %[res2](%[cm])                     \n\t"
      "lbux            %[res3],        %[res3](%[cm])                     \n\t"

      "sb              %[res0],        (%[dst])                           \n\t"
      "sb              %[res1],        1(%[dst])                          \n\t"
      "sb              %[res2],        2(%[dst])                          \n\t"
      "sb              %[res3],        3(%[dst])                          \n\t"

      "addu.ph         %[reshw],       %[abovel_1],         %[left0]      \n\t"
      "subu.ph         %[reshw],       %[reshw],            %[top_left]   \n\t"

      "sll             %[res2],        %[reshw],            16            \n\t"
      "sra             %[res2],        %[res2],             16            \n\t"
      "sra             %[res3],        %[reshw],            16            \n\t"

      "addu.ph         %[reshw],       %[abover_1],         %[left0]      \n\t"
      "subu.ph         %[reshw],       %[reshw],            %[top_left]   \n\t"

      "sll             %[res0],        %[reshw],            16            \n\t"
      "sra             %[res0],        %[res0],             16            \n\t"
      "sra             %[res1],        %[reshw],            16            \n\t"

      "lbu             %[left0],       7(%[left])                         \n\t"

      "lbux            %[res0],        %[res0](%[cm])                     \n\t"
      "lbux            %[res1],        %[res1](%[cm])                     \n\t"
      "lbux            %[res2],        %[res2](%[cm])                     \n\t"
      "lbux            %[res3],        %[res3](%[cm])                     \n\t"

      "sb              %[res0],        4(%[dst])                          \n\t"
      "sb              %[res1],        5(%[dst])                          \n\t"
      "sb              %[res2],        6(%[dst])                          \n\t"
      "sb              %[res3],        7(%[dst])                          \n\t"

      "replv.ph        %[left0],       %[left0]                           \n\t"
      "add             %[dst],          %[dst],             %[stride]     \n\t"

      "addu.ph         %[reshw],       %[abovel],           %[left0]      \n\t"
      "subu.ph         %[reshw],       %[reshw],            %[top_left]   \n\t"

      "sll             %[res2],        %[reshw],            16            \n\t"
      "sra             %[res2],        %[res2],             16            \n\t"
      "sra             %[res3],        %[reshw],            16            \n\t"

      "addu.ph         %[reshw],       %[abover],           %[left0]      \n\t"
      "subu.ph         %[reshw],       %[reshw],            %[top_left]   \n\t"

      "sll             %[res0],        %[reshw],            16            \n\t"
      "sra             %[res0],        %[res0],             16            \n\t"
      "sra             %[res1],        %[reshw],            16            \n\t"

      "lbux            %[res0],        %[res0](%[cm])                     \n\t"
      "lbux            %[res1],        %[res1](%[cm])                     \n\t"
      "lbux            %[res2],        %[res2](%[cm])                     \n\t"
      "lbux            %[res3],        %[res3](%[cm])                     \n\t"

      "sb              %[res0],        (%[dst])                           \n\t"
      "sb              %[res1],        1(%[dst])                          \n\t"
      "sb              %[res2],        2(%[dst])                          \n\t"
      "sb              %[res3],        3(%[dst])                          \n\t"

      "addu.ph         %[reshw],       %[abovel_1],         %[left0]      \n\t"
      "subu.ph         %[reshw],       %[reshw],            %[top_left]   \n\t"

      "sll             %[res2],        %[reshw],            16            \n\t"
      "sra             %[res2],        %[res2],             16            \n\t"
      "sra             %[res3],        %[reshw],            16            \n\t"

      "addu.ph         %[reshw],       %[abover_1],         %[left0]      \n\t"
      "subu.ph         %[reshw],       %[reshw],            %[top_left]   \n\t"

      "sll             %[res0],        %[reshw],            16            \n\t"
      "sra             %[res0],        %[res0],             16            \n\t"
      "sra             %[res1],        %[reshw],            16            \n\t"

      "lbux            %[res0],        %[res0](%[cm])                     \n\t"
      "lbux            %[res1],        %[res1](%[cm])                     \n\t"
      "lbux            %[res2],        %[res2](%[cm])                     \n\t"
      "lbux            %[res3],        %[res3](%[cm])                     \n\t"

      "sb              %[res0],        4(%[dst])                          \n\t"
      "sb              %[res1],        5(%[dst])                          \n\t"
      "sb              %[res2],        6(%[dst])                          \n\t"
      "sb              %[res3],        7(%[dst])                          \n\t"

      : [abovel] "=&r"(abovel), [abover] "=&r"(abover),
        [abovel_1] "=&r"(abovel_1), [abover_1] "=&r"(abover_1),
        [left0] "=&r"(left0), [res2] "=&r"(res2), [res3] "=&r"(res3),
        [res0] "=&r"(res0), [res1] "=&r"(res1), [reshw] "=&r"(reshw),
        [top_left] "=&r"(top_left)
      : [above] "r"(above), [left] "r"(left), [dst] "r"(dst),
        [stride] "r"(stride), [cm] "r"(cm));
}
#endif  // #if HAVE_DSPR2
