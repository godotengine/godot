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

void vp9_iht4x4_16_add_msa(const int16_t *input, uint8_t *dst,
                           int32_t dst_stride, int32_t tx_type) {
  v8i16 in0, in1, in2, in3;

  /* load vector elements of 4x4 block */
  LD4x4_SH(input, in0, in1, in2, in3);
  TRANSPOSE4x4_SH_SH(in0, in1, in2, in3, in0, in1, in2, in3);

  switch (tx_type) {
    case DCT_DCT:
      /* DCT in horizontal */
      VP9_IDCT4x4(in0, in1, in2, in3, in0, in1, in2, in3);
      /* DCT in vertical */
      TRANSPOSE4x4_SH_SH(in0, in1, in2, in3, in0, in1, in2, in3);
      VP9_IDCT4x4(in0, in1, in2, in3, in0, in1, in2, in3);
      break;
    case ADST_DCT:
      /* DCT in horizontal */
      VP9_IDCT4x4(in0, in1, in2, in3, in0, in1, in2, in3);
      /* ADST in vertical */
      TRANSPOSE4x4_SH_SH(in0, in1, in2, in3, in0, in1, in2, in3);
      VP9_IADST4x4(in0, in1, in2, in3, in0, in1, in2, in3);
      break;
    case DCT_ADST:
      /* ADST in horizontal */
      VP9_IADST4x4(in0, in1, in2, in3, in0, in1, in2, in3);
      /* DCT in vertical */
      TRANSPOSE4x4_SH_SH(in0, in1, in2, in3, in0, in1, in2, in3);
      VP9_IDCT4x4(in0, in1, in2, in3, in0, in1, in2, in3);
      break;
    case ADST_ADST:
      /* ADST in horizontal */
      VP9_IADST4x4(in0, in1, in2, in3, in0, in1, in2, in3);
      /* ADST in vertical */
      TRANSPOSE4x4_SH_SH(in0, in1, in2, in3, in0, in1, in2, in3);
      VP9_IADST4x4(in0, in1, in2, in3, in0, in1, in2, in3);
      break;
    default: assert(0); break;
  }

  /* final rounding (add 2^3, divide by 2^4) and shift */
  SRARI_H4_SH(in0, in1, in2, in3, 4);
  /* add block and store 4x4 */
  ADDBLK_ST4x4_UB(in0, in1, in2, in3, dst, dst_stride);
}
