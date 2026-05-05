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

void vp9_iht16x16_256_add_msa(const int16_t *input, uint8_t *dst,
                              int32_t dst_stride, int32_t tx_type) {
  int32_t i;
  DECLARE_ALIGNED(32, int16_t, out[16 * 16]);
  int16_t *out_ptr = &out[0];

  switch (tx_type) {
    case DCT_DCT:
      /* transform rows */
      for (i = 0; i < 2; ++i) {
        /* process 16 * 8 block */
        vpx_idct16_1d_rows_msa((input + (i << 7)), (out_ptr + (i << 7)));
      }

      /* transform columns */
      for (i = 0; i < 2; ++i) {
        /* process 8 * 16 block */
        vpx_idct16_1d_columns_addblk_msa((out_ptr + (i << 3)), (dst + (i << 3)),
                                         dst_stride);
      }
      break;
    case ADST_DCT:
      /* transform rows */
      for (i = 0; i < 2; ++i) {
        /* process 16 * 8 block */
        vpx_idct16_1d_rows_msa((input + (i << 7)), (out_ptr + (i << 7)));
      }

      /* transform columns */
      for (i = 0; i < 2; ++i) {
        vpx_iadst16_1d_columns_addblk_msa((out_ptr + (i << 3)),
                                          (dst + (i << 3)), dst_stride);
      }
      break;
    case DCT_ADST:
      /* transform rows */
      for (i = 0; i < 2; ++i) {
        /* process 16 * 8 block */
        vpx_iadst16_1d_rows_msa((input + (i << 7)), (out_ptr + (i << 7)));
      }

      /* transform columns */
      for (i = 0; i < 2; ++i) {
        /* process 8 * 16 block */
        vpx_idct16_1d_columns_addblk_msa((out_ptr + (i << 3)), (dst + (i << 3)),
                                         dst_stride);
      }
      break;
    case ADST_ADST:
      /* transform rows */
      for (i = 0; i < 2; ++i) {
        /* process 16 * 8 block */
        vpx_iadst16_1d_rows_msa((input + (i << 7)), (out_ptr + (i << 7)));
      }

      /* transform columns */
      for (i = 0; i < 2; ++i) {
        vpx_iadst16_1d_columns_addblk_msa((out_ptr + (i << 3)),
                                          (dst + (i << 3)), dst_stride);
      }
      break;
    default: assert(0); break;
  }
}
