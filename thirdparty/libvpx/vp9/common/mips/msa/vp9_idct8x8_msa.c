/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>

#include "./vp9_rtcd.h"
#include "vp9/common/vp9_enums.h"
#include "vpx_dsp/mips/inv_txfm_msa.h"

void vp9_iht8x8_64_add_msa(const int16_t *input, uint8_t *dst,
                           int32_t dst_stride, int32_t tx_type) {
  v8i16 in0, in1, in2, in3, in4, in5, in6, in7;

  /* load vector elements of 8x8 block */
  LD_SH8(input, 8, in0, in1, in2, in3, in4, in5, in6, in7);

  TRANSPOSE8x8_SH_SH(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3,
                     in4, in5, in6, in7);

  switch (tx_type) {
    case DCT_DCT:
      /* DCT in horizontal */
      VP9_IDCT8x8_1D(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3,
                     in4, in5, in6, in7);
      /* DCT in vertical */
      TRANSPOSE8x8_SH_SH(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2,
                         in3, in4, in5, in6, in7);
      VP9_IDCT8x8_1D(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3,
                     in4, in5, in6, in7);
      break;
    case ADST_DCT:
      /* DCT in horizontal */
      VP9_IDCT8x8_1D(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3,
                     in4, in5, in6, in7);
      /* ADST in vertical */
      TRANSPOSE8x8_SH_SH(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2,
                         in3, in4, in5, in6, in7);
      VP9_ADST8(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3, in4,
                in5, in6, in7);
      break;
    case DCT_ADST:
      /* ADST in horizontal */
      VP9_ADST8(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3, in4,
                in5, in6, in7);
      /* DCT in vertical */
      TRANSPOSE8x8_SH_SH(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2,
                         in3, in4, in5, in6, in7);
      VP9_IDCT8x8_1D(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3,
                     in4, in5, in6, in7);
      break;
    case ADST_ADST:
      /* ADST in horizontal */
      VP9_ADST8(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3, in4,
                in5, in6, in7);
      /* ADST in vertical */
      TRANSPOSE8x8_SH_SH(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2,
                         in3, in4, in5, in6, in7);
      VP9_ADST8(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3, in4,
                in5, in6, in7);
      break;
    default: assert(0); break;
  }

  /* final rounding (add 2^4, divide by 2^5) and shift */
  SRARI_H4_SH(in0, in1, in2, in3, 5);
  SRARI_H4_SH(in4, in5, in6, in7, 5);

  /* add block and store 8x8 */
  VP9_ADDBLK_ST8x4_UB(dst, dst_stride, in0, in1, in2, in3);
  dst += (4 * dst_stride);
  VP9_ADDBLK_ST8x4_UB(dst, dst_stride, in4, in5, in6, in7);
}
