/*
 *  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>
#include <stdio.h>

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/mips/convolve_common_dspr2.h"
#include "vpx_dsp/vpx_convolve.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_ports/mem.h"

#if HAVE_DSPR2
static void convolve_bi_avg_vert_4_dspr2(const uint8_t *src, int32_t src_stride,
                                         uint8_t *dst, int32_t dst_stride,
                                         const int16_t *filter_y, int32_t w,
                                         int32_t h) {
  int32_t x, y;
  const uint8_t *src_ptr;
  uint8_t *dst_ptr;
  uint8_t *cm = vpx_ff_cropTbl;
  uint32_t vector4a = 64;
  uint32_t load1, load2;
  uint32_t p1, p2;
  uint32_t scratch1, scratch2;
  uint32_t store1, store2;
  int32_t Temp1, Temp2;
  const int16_t *filter = &filter_y[3];
  uint32_t filter45;

  filter45 = ((const int32_t *)filter)[0];

  for (y = h; y--;) {
    /* prefetch data to cache memory */
    prefetch_store(dst + dst_stride);

    for (x = 0; x < w; x += 4) {
      src_ptr = src + x;
      dst_ptr = dst + x;

      __asm__ __volatile__(
          "ulw              %[load1],     0(%[src_ptr])                   \n\t"
          "add              %[src_ptr],   %[src_ptr],     %[src_stride]   \n\t"
          "ulw              %[load2],     0(%[src_ptr])                   \n\t"

          "mtlo             %[vector4a],  $ac0                            \n\t"
          "mtlo             %[vector4a],  $ac1                            \n\t"
          "mtlo             %[vector4a],  $ac2                            \n\t"
          "mtlo             %[vector4a],  $ac3                            \n\t"
          "mthi             $zero,        $ac0                            \n\t"
          "mthi             $zero,        $ac1                            \n\t"
          "mthi             $zero,        $ac2                            \n\t"
          "mthi             $zero,        $ac3                            \n\t"

          "preceu.ph.qbr    %[scratch1],  %[load1]                        \n\t"
          "preceu.ph.qbr    %[p1],        %[load2]                        \n\t"
          "precrq.ph.w      %[p2],        %[p1],          %[scratch1]     \n\t" /* pixel 2 */
          "append           %[p1],        %[scratch1],    16              \n\t" /* pixel 1 */

          "dpa.w.ph         $ac0,         %[p1],          %[filter45]     \n\t"
          "dpa.w.ph         $ac1,         %[p2],          %[filter45]     \n\t"

          "preceu.ph.qbl    %[scratch1],  %[load1]                        \n\t"
          "preceu.ph.qbl    %[p1],        %[load2]                        \n\t"
          "precrq.ph.w      %[p2],        %[p1],          %[scratch1]     \n\t" /* pixel 2 */
          "append           %[p1],        %[scratch1],    16              \n\t" /* pixel 1 */

          "dpa.w.ph         $ac2,         %[p1],          %[filter45]     \n\t"
          "dpa.w.ph         $ac3,         %[p2],          %[filter45]     \n\t"

          "extp             %[Temp1],     $ac0,           31              \n\t"
          "extp             %[Temp2],     $ac1,           31              \n\t"

          "lbu              %[scratch1],  0(%[dst_ptr])                   \n\t"
          "lbu              %[scratch2],  1(%[dst_ptr])                   \n\t"

          "lbux             %[store1],    %[Temp1](%[cm])                 \n\t"
          "addqh_r.w        %[store1],    %[store1],      %[scratch1]     \n\t" /* pixel 1 */
          "extp             %[Temp1],     $ac2,           31              \n\t"

          "lbux             %[store2],    %[Temp2](%[cm])                 \n\t"
          "addqh_r.w        %[store2],    %[store2],      %[scratch2]     \n\t" /* pixel 2 */
          "extp             %[Temp2],     $ac3,           31              \n\t"
          "lbu              %[scratch1],  2(%[dst_ptr])                   \n\t"

          "sb               %[store1],    0(%[dst_ptr])                   \n\t"
          "sb               %[store2],    1(%[dst_ptr])                   \n\t"
          "lbu              %[scratch2],  3(%[dst_ptr])                   \n\t"

          "lbux             %[store1],    %[Temp1](%[cm])                 \n\t"
          "lbux             %[store2],    %[Temp2](%[cm])                 \n\t"
          "addqh_r.w        %[store1],    %[store1],      %[scratch1]     \n\t" /* pixel 3 */
          "addqh_r.w        %[store2],    %[store2],      %[scratch2]     \n\t" /* pixel 4 */

          "sb               %[store1],    2(%[dst_ptr])                   \n\t"
          "sb               %[store2],    3(%[dst_ptr])                   \n\t"

          : [load1] "=&r"(load1), [load2] "=&r"(load2), [p1] "=&r"(p1),
            [p2] "=&r"(p2), [scratch1] "=&r"(scratch1),
            [scratch2] "=&r"(scratch2), [Temp1] "=&r"(Temp1),
            [Temp2] "=&r"(Temp2), [store1] "=&r"(store1),
            [store2] "=&r"(store2), [src_ptr] "+r"(src_ptr)
          : [filter45] "r"(filter45), [vector4a] "r"(vector4a),
            [src_stride] "r"(src_stride), [cm] "r"(cm), [dst_ptr] "r"(dst_ptr));
    }

    /* Next row... */
    src += src_stride;
    dst += dst_stride;
  }
}

static void convolve_bi_avg_vert_64_dspr2(const uint8_t *src,
                                          int32_t src_stride, uint8_t *dst,
                                          int32_t dst_stride,
                                          const int16_t *filter_y, int32_t h) {
  int32_t x, y;
  const uint8_t *src_ptr;
  uint8_t *dst_ptr;
  uint8_t *cm = vpx_ff_cropTbl;
  uint32_t vector4a = 64;
  uint32_t load1, load2;
  uint32_t p1, p2;
  uint32_t scratch1, scratch2;
  uint32_t store1, store2;
  int32_t Temp1, Temp2;
  const int16_t *filter = &filter_y[3];
  uint32_t filter45;

  filter45 = ((const int32_t *)filter)[0];

  for (y = h; y--;) {
    /* prefetch data to cache memory */
    prefetch_store(dst + dst_stride);
    prefetch_store(dst + dst_stride + 32);

    for (x = 0; x < 64; x += 4) {
      src_ptr = src + x;
      dst_ptr = dst + x;

      __asm__ __volatile__(
          "ulw              %[load1],     0(%[src_ptr])                   \n\t"
          "add              %[src_ptr],   %[src_ptr],     %[src_stride]   \n\t"
          "ulw              %[load2],     0(%[src_ptr])                   \n\t"

          "mtlo             %[vector4a],  $ac0                            \n\t"
          "mtlo             %[vector4a],  $ac1                            \n\t"
          "mtlo             %[vector4a],  $ac2                            \n\t"
          "mtlo             %[vector4a],  $ac3                            \n\t"
          "mthi             $zero,        $ac0                            \n\t"
          "mthi             $zero,        $ac1                            \n\t"
          "mthi             $zero,        $ac2                            \n\t"
          "mthi             $zero,        $ac3                            \n\t"

          "preceu.ph.qbr    %[scratch1],  %[load1]                        \n\t"
          "preceu.ph.qbr    %[p1],        %[load2]                        \n\t"
          "precrq.ph.w      %[p2],        %[p1],          %[scratch1]     \n\t" /* pixel 2 */
          "append           %[p1],        %[scratch1],    16              \n\t" /* pixel 1 */

          "dpa.w.ph         $ac0,         %[p1],          %[filter45]     \n\t"
          "dpa.w.ph         $ac1,         %[p2],          %[filter45]     \n\t"

          "preceu.ph.qbl    %[scratch1],  %[load1]                        \n\t"
          "preceu.ph.qbl    %[p1],        %[load2]                        \n\t"
          "precrq.ph.w      %[p2],        %[p1],          %[scratch1]     \n\t" /* pixel 2 */
          "append           %[p1],        %[scratch1],    16              \n\t" /* pixel 1 */

          "dpa.w.ph         $ac2,         %[p1],          %[filter45]     \n\t"
          "dpa.w.ph         $ac3,         %[p2],          %[filter45]     \n\t"

          "extp             %[Temp1],     $ac0,           31              \n\t"
          "extp             %[Temp2],     $ac1,           31              \n\t"

          "lbu              %[scratch1],  0(%[dst_ptr])                   \n\t"
          "lbu              %[scratch2],  1(%[dst_ptr])                   \n\t"

          "lbux             %[store1],    %[Temp1](%[cm])                 \n\t"
          "addqh_r.w        %[store1],    %[store1],      %[scratch1]     \n\t" /* pixel 1 */
          "extp             %[Temp1],     $ac2,           31              \n\t"

          "lbux             %[store2],    %[Temp2](%[cm])                 \n\t"
          "addqh_r.w        %[store2],    %[store2],      %[scratch2]     \n\t" /* pixel 2 */
          "extp             %[Temp2],     $ac3,           31              \n\t"
          "lbu              %[scratch1],  2(%[dst_ptr])                   \n\t"

          "sb               %[store1],    0(%[dst_ptr])                   \n\t"
          "sb               %[store2],    1(%[dst_ptr])                   \n\t"
          "lbu              %[scratch2],  3(%[dst_ptr])                   \n\t"

          "lbux             %[store1],    %[Temp1](%[cm])                 \n\t"
          "lbux             %[store2],    %[Temp2](%[cm])                 \n\t"
          "addqh_r.w        %[store1],    %[store1],      %[scratch1]     \n\t" /* pixel 3 */
          "addqh_r.w        %[store2],    %[store2],      %[scratch2]     \n\t" /* pixel 4 */

          "sb               %[store1],    2(%[dst_ptr])                   \n\t"
          "sb               %[store2],    3(%[dst_ptr])                   \n\t"

          : [load1] "=&r"(load1), [load2] "=&r"(load2), [p1] "=&r"(p1),
            [p2] "=&r"(p2), [scratch1] "=&r"(scratch1),
            [scratch2] "=&r"(scratch2), [Temp1] "=&r"(Temp1),
            [Temp2] "=&r"(Temp2), [store1] "=&r"(store1),
            [store2] "=&r"(store2), [src_ptr] "+r"(src_ptr)
          : [filter45] "r"(filter45), [vector4a] "r"(vector4a),
            [src_stride] "r"(src_stride), [cm] "r"(cm), [dst_ptr] "r"(dst_ptr));
    }

    /* Next row... */
    src += src_stride;
    dst += dst_stride;
  }
}

void vpx_convolve2_avg_vert_dspr2(const uint8_t *src, ptrdiff_t src_stride,
                                  uint8_t *dst, ptrdiff_t dst_stride,
                                  const InterpKernel *filter, int x0_q4,
                                  int32_t x_step_q4, int y0_q4, int y_step_q4,
                                  int w, int h) {
  const int16_t *const filter_y = filter[y0_q4];
  uint32_t pos = 38;

  assert(y_step_q4 == 16);

  /* bit positon for extract from acc */
  __asm__ __volatile__("wrdsp      %[pos],     1           \n\t"
                       :
                       : [pos] "r"(pos));

  prefetch_store(dst);

  switch (w) {
    case 4:
    case 8:
    case 16:
    case 32:
      convolve_bi_avg_vert_4_dspr2(src, src_stride, dst, dst_stride, filter_y,
                                   w, h);
      break;
    case 64:
      prefetch_store(dst + 32);
      convolve_bi_avg_vert_64_dspr2(src, src_stride, dst, dst_stride, filter_y,
                                    h);
      break;
    default:
      vpx_convolve8_avg_vert_c(src, src_stride, dst, dst_stride, filter, x0_q4,
                               x_step_q4, y0_q4, y_step_q4, w, h);
      break;
  }
}
#endif
