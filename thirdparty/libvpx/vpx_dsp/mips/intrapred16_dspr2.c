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
void vpx_h_predictor_16x16_dspr2(uint8_t *dst, ptrdiff_t stride,
                                 const uint8_t *above, const uint8_t *left) {
  int32_t tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8;
  int32_t tmp9, tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, tmp16;
  (void)above;

  __asm__ __volatile__(
      "lb         %[tmp1],      (%[left])                    \n\t"
      "lb         %[tmp2],      1(%[left])                   \n\t"
      "lb         %[tmp3],      2(%[left])                   \n\t"
      "lb         %[tmp4],      3(%[left])                   \n\t"
      "lb         %[tmp5],      4(%[left])                   \n\t"
      "lb         %[tmp6],      5(%[left])                   \n\t"
      "lb         %[tmp7],      6(%[left])                   \n\t"
      "lb         %[tmp8],      7(%[left])                   \n\t"
      "lb         %[tmp9],      8(%[left])                   \n\t"
      "lb         %[tmp10],     9(%[left])                   \n\t"
      "lb         %[tmp11],     10(%[left])                  \n\t"
      "lb         %[tmp12],     11(%[left])                  \n\t"
      "lb         %[tmp13],     12(%[left])                  \n\t"
      "lb         %[tmp14],     13(%[left])                  \n\t"
      "lb         %[tmp15],     14(%[left])                  \n\t"
      "lb         %[tmp16],     15(%[left])                  \n\t"

      "replv.qb   %[tmp1],      %[tmp1]                      \n\t"
      "replv.qb   %[tmp2],      %[tmp2]                      \n\t"
      "replv.qb   %[tmp3],      %[tmp3]                      \n\t"
      "replv.qb   %[tmp4],      %[tmp4]                      \n\t"
      "replv.qb   %[tmp5],      %[tmp5]                      \n\t"
      "replv.qb   %[tmp6],      %[tmp6]                      \n\t"
      "replv.qb   %[tmp7],      %[tmp7]                      \n\t"
      "replv.qb   %[tmp8],      %[tmp8]                      \n\t"
      "replv.qb   %[tmp9],      %[tmp9]                      \n\t"
      "replv.qb   %[tmp10],     %[tmp10]                     \n\t"
      "replv.qb   %[tmp11],     %[tmp11]                     \n\t"
      "replv.qb   %[tmp12],     %[tmp12]                     \n\t"
      "replv.qb   %[tmp13],     %[tmp13]                     \n\t"
      "replv.qb   %[tmp14],     %[tmp14]                     \n\t"
      "replv.qb   %[tmp15],     %[tmp15]                     \n\t"
      "replv.qb   %[tmp16],     %[tmp16]                     \n\t"

      "sw         %[tmp1],      (%[dst])                     \n\t"
      "sw         %[tmp1],      4(%[dst])                    \n\t"
      "sw         %[tmp1],      8(%[dst])                    \n\t"
      "sw         %[tmp1],      12(%[dst])                   \n\t"

      "add        %[dst],       %[dst],         %[stride]    \n\t"
      "sw         %[tmp2],      (%[dst])                     \n\t"
      "sw         %[tmp2],      4(%[dst])                    \n\t"
      "sw         %[tmp2],      8(%[dst])                    \n\t"
      "sw         %[tmp2],      12(%[dst])                   \n\t"

      "add        %[dst],       %[dst],         %[stride]    \n\t"
      "sw         %[tmp3],      (%[dst])                     \n\t"
      "sw         %[tmp3],      4(%[dst])                    \n\t"
      "sw         %[tmp3],      8(%[dst])                    \n\t"
      "sw         %[tmp3],      12(%[dst])                   \n\t"

      "add        %[dst],       %[dst],         %[stride]    \n\t"
      "sw         %[tmp4],      (%[dst])                     \n\t"
      "sw         %[tmp4],      4(%[dst])                    \n\t"
      "sw         %[tmp4],      8(%[dst])                    \n\t"
      "sw         %[tmp4],      12(%[dst])                   \n\t"

      "add        %[dst],       %[dst],         %[stride]    \n\t"
      "sw         %[tmp5],      (%[dst])                     \n\t"
      "sw         %[tmp5],      4(%[dst])                    \n\t"
      "sw         %[tmp5],      8(%[dst])                    \n\t"
      "sw         %[tmp5],      12(%[dst])                   \n\t"

      "add        %[dst],       %[dst],         %[stride]    \n\t"
      "sw         %[tmp6],      (%[dst])                     \n\t"
      "sw         %[tmp6],      4(%[dst])                    \n\t"
      "sw         %[tmp6],      8(%[dst])                    \n\t"
      "sw         %[tmp6],      12(%[dst])                   \n\t"

      "add        %[dst],       %[dst],         %[stride]    \n\t"
      "sw         %[tmp7],      (%[dst])                     \n\t"
      "sw         %[tmp7],      4(%[dst])                    \n\t"
      "sw         %[tmp7],      8(%[dst])                    \n\t"
      "sw         %[tmp7],      12(%[dst])                   \n\t"

      "add        %[dst],       %[dst],         %[stride]    \n\t"
      "sw         %[tmp8],      (%[dst])                     \n\t"
      "sw         %[tmp8],      4(%[dst])                    \n\t"
      "sw         %[tmp8],      8(%[dst])                    \n\t"
      "sw         %[tmp8],      12(%[dst])                   \n\t"

      "add        %[dst],       %[dst],         %[stride]    \n\t"
      "sw         %[tmp9],      (%[dst])                     \n\t"
      "sw         %[tmp9],      4(%[dst])                    \n\t"
      "sw         %[tmp9],      8(%[dst])                    \n\t"
      "sw         %[tmp9],      12(%[dst])                   \n\t"

      "add        %[dst],       %[dst],         %[stride]    \n\t"
      "sw         %[tmp10],     (%[dst])                     \n\t"
      "sw         %[tmp10],     4(%[dst])                    \n\t"
      "sw         %[tmp10],     8(%[dst])                    \n\t"
      "sw         %[tmp10],     12(%[dst])                   \n\t"

      "add        %[dst],       %[dst],         %[stride]    \n\t"
      "sw         %[tmp11],     (%[dst])                     \n\t"
      "sw         %[tmp11],     4(%[dst])                    \n\t"
      "sw         %[tmp11],     8(%[dst])                    \n\t"
      "sw         %[tmp11],     12(%[dst])                   \n\t"

      "add        %[dst],       %[dst],         %[stride]    \n\t"
      "sw         %[tmp12],     (%[dst])                     \n\t"
      "sw         %[tmp12],     4(%[dst])                    \n\t"
      "sw         %[tmp12],     8(%[dst])                    \n\t"
      "sw         %[tmp12],     12(%[dst])                   \n\t"

      "add        %[dst],       %[dst],         %[stride]    \n\t"
      "sw         %[tmp13],     (%[dst])                     \n\t"
      "sw         %[tmp13],     4(%[dst])                    \n\t"
      "sw         %[tmp13],     8(%[dst])                    \n\t"
      "sw         %[tmp13],     12(%[dst])                   \n\t"

      "add        %[dst],       %[dst],         %[stride]    \n\t"
      "sw         %[tmp14],     (%[dst])                     \n\t"
      "sw         %[tmp14],     4(%[dst])                    \n\t"
      "sw         %[tmp14],     8(%[dst])                    \n\t"
      "sw         %[tmp14],     12(%[dst])                   \n\t"

      "add        %[dst],       %[dst],         %[stride]    \n\t"
      "sw         %[tmp15],     (%[dst])                     \n\t"
      "sw         %[tmp15],     4(%[dst])                    \n\t"
      "sw         %[tmp15],     8(%[dst])                    \n\t"
      "sw         %[tmp15],     12(%[dst])                   \n\t"

      "add        %[dst],       %[dst],         %[stride]    \n\t"
      "sw         %[tmp16],     (%[dst])                     \n\t"
      "sw         %[tmp16],     4(%[dst])                    \n\t"
      "sw         %[tmp16],     8(%[dst])                    \n\t"
      "sw         %[tmp16],     12(%[dst])                   \n\t"

      : [tmp1] "=&r"(tmp1), [tmp2] "=&r"(tmp2), [tmp3] "=&r"(tmp3),
        [tmp4] "=&r"(tmp4), [tmp5] "=&r"(tmp5), [tmp7] "=&r"(tmp7),
        [tmp6] "=&r"(tmp6), [tmp8] "=&r"(tmp8), [tmp9] "=&r"(tmp9),
        [tmp10] "=&r"(tmp10), [tmp11] "=&r"(tmp11), [tmp12] "=&r"(tmp12),
        [tmp13] "=&r"(tmp13), [tmp14] "=&r"(tmp14), [tmp15] "=&r"(tmp15),
        [tmp16] "=&r"(tmp16)
      : [left] "r"(left), [dst] "r"(dst), [stride] "r"(stride));
}

void vpx_dc_predictor_16x16_dspr2(uint8_t *dst, ptrdiff_t stride,
                                  const uint8_t *above, const uint8_t *left) {
  int32_t expected_dc;
  int32_t average;
  int32_t tmp, above1, above_l1, above_r1, left1, left_r1, left_l1;
  int32_t above2, left2;

  __asm__ __volatile__(
      "lw              %[above1],           (%[above])                    \n\t"
      "lw              %[above2],           4(%[above])                   \n\t"
      "lw              %[left1],            (%[left])                     \n\t"
      "lw              %[left2],            4(%[left])                    \n\t"

      "preceu.ph.qbl   %[above_l1],         %[above1]                     \n\t"
      "preceu.ph.qbr   %[above_r1],         %[above1]                     \n\t"
      "preceu.ph.qbl   %[left_l1],          %[left1]                      \n\t"
      "preceu.ph.qbr   %[left_r1],          %[left1]                      \n\t"

      "addu.ph         %[average],          %[above_r1],     %[above_l1]  \n\t"
      "addu.ph         %[average],          %[average],      %[left_l1]   \n\t"
      "addu.ph         %[average],          %[average],      %[left_r1]   \n\t"

      "preceu.ph.qbl   %[above_l1],         %[above2]                     \n\t"
      "preceu.ph.qbr   %[above_r1],         %[above2]                     \n\t"
      "preceu.ph.qbl   %[left_l1],          %[left2]                      \n\t"
      "preceu.ph.qbr   %[left_r1],          %[left2]                      \n\t"

      "addu.ph         %[average],          %[average],      %[above_l1]  \n\t"
      "addu.ph         %[average],          %[average],      %[above_r1]  \n\t"
      "addu.ph         %[average],          %[average],      %[left_l1]   \n\t"
      "addu.ph         %[average],          %[average],      %[left_r1]   \n\t"

      "lw              %[above1],           8(%[above])                   \n\t"
      "lw              %[above2],           12(%[above])                  \n\t"
      "lw              %[left1],            8(%[left])                    \n\t"
      "lw              %[left2],            12(%[left])                   \n\t"

      "preceu.ph.qbl   %[above_l1],         %[above1]                     \n\t"
      "preceu.ph.qbr   %[above_r1],         %[above1]                     \n\t"
      "preceu.ph.qbl   %[left_l1],          %[left1]                      \n\t"
      "preceu.ph.qbr   %[left_r1],          %[left1]                      \n\t"

      "addu.ph         %[average],          %[average],      %[above_l1]  \n\t"
      "addu.ph         %[average],          %[average],      %[above_r1]  \n\t"
      "addu.ph         %[average],          %[average],      %[left_l1]   \n\t"
      "addu.ph         %[average],          %[average],      %[left_r1]   \n\t"

      "preceu.ph.qbl   %[above_l1],         %[above2]                     \n\t"
      "preceu.ph.qbr   %[above_r1],         %[above2]                     \n\t"
      "preceu.ph.qbl   %[left_l1],          %[left2]                      \n\t"
      "preceu.ph.qbr   %[left_r1],          %[left2]                      \n\t"

      "addu.ph         %[average],          %[average],      %[above_l1]  \n\t"
      "addu.ph         %[average],          %[average],      %[above_r1]  \n\t"
      "addu.ph         %[average],          %[average],      %[left_l1]   \n\t"
      "addu.ph         %[average],          %[average],      %[left_r1]   \n\t"

      "addiu           %[average],          %[average],      16           \n\t"
      "srl             %[tmp],              %[average],      16           \n\t"
      "addu.ph         %[average],          %[tmp],          %[average]   \n\t"
      "srl             %[expected_dc],      %[average],      5            \n\t"
      "replv.qb        %[expected_dc],      %[expected_dc]                \n\t"

      "sw              %[expected_dc],      (%[dst])                      \n\t"
      "sw              %[expected_dc],      4(%[dst])                     \n\t"
      "sw              %[expected_dc],      8(%[dst])                     \n\t"
      "sw              %[expected_dc],      12(%[dst])                    \n\t"

      "add             %[dst],              %[dst],          %[stride]    \n\t"
      "sw              %[expected_dc],      (%[dst])                      \n\t"
      "sw              %[expected_dc],      4(%[dst])                     \n\t"
      "sw              %[expected_dc],      8(%[dst])                     \n\t"
      "sw              %[expected_dc],      12(%[dst])                    \n\t"

      "add             %[dst],              %[dst],          %[stride]    \n\t"
      "sw              %[expected_dc],      (%[dst])                      \n\t"
      "sw              %[expected_dc],      4(%[dst])                     \n\t"
      "sw              %[expected_dc],      8(%[dst])                     \n\t"
      "sw              %[expected_dc],      12(%[dst])                    \n\t"

      "add             %[dst],              %[dst],          %[stride]    \n\t"
      "sw              %[expected_dc],      (%[dst])                      \n\t"
      "sw              %[expected_dc],      4(%[dst])                     \n\t"
      "sw              %[expected_dc],      8(%[dst])                     \n\t"
      "sw              %[expected_dc],      12(%[dst])                    \n\t"

      "add             %[dst],              %[dst],          %[stride]    \n\t"
      "sw              %[expected_dc],      (%[dst])                      \n\t"
      "sw              %[expected_dc],      4(%[dst])                     \n\t"
      "sw              %[expected_dc],      8(%[dst])                     \n\t"
      "sw              %[expected_dc],      12(%[dst])                    \n\t"

      "add             %[dst],              %[dst],          %[stride]    \n\t"
      "sw              %[expected_dc],      (%[dst])                      \n\t"
      "sw              %[expected_dc],      4(%[dst])                     \n\t"
      "sw              %[expected_dc],      8(%[dst])                     \n\t"
      "sw              %[expected_dc],      12(%[dst])                    \n\t"

      "add             %[dst],              %[dst],          %[stride]    \n\t"
      "sw              %[expected_dc],      (%[dst])                      \n\t"
      "sw              %[expected_dc],      4(%[dst])                     \n\t"
      "sw              %[expected_dc],      8(%[dst])                     \n\t"
      "sw              %[expected_dc],      12(%[dst])                    \n\t"

      "add             %[dst],              %[dst],          %[stride]    \n\t"
      "sw              %[expected_dc],      (%[dst])                      \n\t"
      "sw              %[expected_dc],      4(%[dst])                     \n\t"
      "sw              %[expected_dc],      8(%[dst])                     \n\t"
      "sw              %[expected_dc],      12(%[dst])                    \n\t"

      "add             %[dst],              %[dst],          %[stride]    \n\t"
      "sw              %[expected_dc],      (%[dst])                      \n\t"
      "sw              %[expected_dc],      4(%[dst])                     \n\t"
      "sw              %[expected_dc],      8(%[dst])                     \n\t"
      "sw              %[expected_dc],      12(%[dst])                    \n\t"

      "add             %[dst],              %[dst],          %[stride]    \n\t"
      "sw              %[expected_dc],      (%[dst])                      \n\t"
      "sw              %[expected_dc],      4(%[dst])                     \n\t"
      "sw              %[expected_dc],      8(%[dst])                     \n\t"
      "sw              %[expected_dc],      12(%[dst])                    \n\t"

      "add             %[dst],              %[dst],          %[stride]    \n\t"
      "sw              %[expected_dc],      (%[dst])                      \n\t"
      "sw              %[expected_dc],      4(%[dst])                     \n\t"
      "sw              %[expected_dc],      8(%[dst])                     \n\t"
      "sw              %[expected_dc],      12(%[dst])                    \n\t"

      "add             %[dst],              %[dst],          %[stride]    \n\t"
      "sw              %[expected_dc],      (%[dst])                      \n\t"
      "sw              %[expected_dc],      4(%[dst])                     \n\t"
      "sw              %[expected_dc],      8(%[dst])                     \n\t"
      "sw              %[expected_dc],      12(%[dst])                    \n\t"

      "add             %[dst],              %[dst],          %[stride]    \n\t"
      "sw              %[expected_dc],      (%[dst])                      \n\t"
      "sw              %[expected_dc],      4(%[dst])                     \n\t"
      "sw              %[expected_dc],      8(%[dst])                     \n\t"
      "sw              %[expected_dc],      12(%[dst])                    \n\t"

      "add             %[dst],              %[dst],          %[stride]    \n\t"
      "sw              %[expected_dc],      (%[dst])                      \n\t"
      "sw              %[expected_dc],      4(%[dst])                     \n\t"
      "sw              %[expected_dc],      8(%[dst])                     \n\t"
      "sw              %[expected_dc],      12(%[dst])                    \n\t"

      "add             %[dst],              %[dst],          %[stride]    \n\t"
      "sw              %[expected_dc],      (%[dst])                      \n\t"
      "sw              %[expected_dc],      4(%[dst])                     \n\t"
      "sw              %[expected_dc],      8(%[dst])                     \n\t"
      "sw              %[expected_dc],      12(%[dst])                    \n\t"

      "add             %[dst],              %[dst],          %[stride]    \n\t"
      "sw              %[expected_dc],      (%[dst])                      \n\t"
      "sw              %[expected_dc],      4(%[dst])                     \n\t"
      "sw              %[expected_dc],      8(%[dst])                     \n\t"
      "sw              %[expected_dc],      12(%[dst])                    \n\t"

      : [left1] "=&r"(left1), [above1] "=&r"(above1), [left_l1] "=&r"(left_l1),
        [above_l1] "=&r"(above_l1), [left_r1] "=&r"(left_r1),
        [above_r1] "=&r"(above_r1), [above2] "=&r"(above2),
        [left2] "=&r"(left2), [average] "=&r"(average), [tmp] "=&r"(tmp),
        [expected_dc] "=&r"(expected_dc)
      : [above] "r"(above), [left] "r"(left), [dst] "r"(dst),
        [stride] "r"(stride));
}
#endif  // #if HAVE_DSPR2
