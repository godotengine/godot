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

#include "./vpx_config.h"
#include "./vp9_rtcd.h"
#include "vp9/common/vp9_common.h"
#include "vp9/common/vp9_blockd.h"
#include "vp9/common/vp9_idct.h"
#include "vpx_dsp/mips/inv_txfm_dspr2.h"
#include "vpx_dsp/txfm_common.h"
#include "vpx_ports/mem.h"

#if HAVE_DSPR2
void vp9_iht4x4_16_add_dspr2(const int16_t *input, uint8_t *dest, int stride,
                             int tx_type) {
  int i, j;
  DECLARE_ALIGNED(32, int16_t, out[4 * 4]);
  int16_t *outptr = out;
  int16_t temp_in[4 * 4], temp_out[4];
  uint32_t pos = 45;

  /* bit positon for extract from acc */
  __asm__ __volatile__("wrdsp      %[pos],     1           \n\t"
                       :
                       : [pos] "r"(pos));

  switch (tx_type) {
    case DCT_DCT:  // DCT in both horizontal and vertical
      vpx_idct4_rows_dspr2(input, outptr);
      vpx_idct4_columns_add_blk_dspr2(&out[0], dest, stride);
      break;
    case ADST_DCT:  // ADST in vertical, DCT in horizontal
      vpx_idct4_rows_dspr2(input, outptr);

      outptr = out;

      for (i = 0; i < 4; ++i) {
        iadst4_dspr2(outptr, temp_out);

        for (j = 0; j < 4; ++j)
          dest[j * stride + i] = clip_pixel(ROUND_POWER_OF_TWO(temp_out[j], 4) +
                                            dest[j * stride + i]);

        outptr += 4;
      }
      break;
    case DCT_ADST:  // DCT in vertical, ADST in horizontal
      for (i = 0; i < 4; ++i) {
        iadst4_dspr2(input, outptr);
        input += 4;
        outptr += 4;
      }

      for (i = 0; i < 4; ++i) {
        for (j = 0; j < 4; ++j) {
          temp_in[i * 4 + j] = out[j * 4 + i];
        }
      }
      vpx_idct4_columns_add_blk_dspr2(&temp_in[0], dest, stride);
      break;
    case ADST_ADST:  // ADST in both directions
      for (i = 0; i < 4; ++i) {
        iadst4_dspr2(input, outptr);
        input += 4;
        outptr += 4;
      }

      for (i = 0; i < 4; ++i) {
        for (j = 0; j < 4; ++j) temp_in[j] = out[j * 4 + i];
        iadst4_dspr2(temp_in, temp_out);

        for (j = 0; j < 4; ++j)
          dest[j * stride + i] = clip_pixel(ROUND_POWER_OF_TWO(temp_out[j], 4) +
                                            dest[j * stride + i]);
      }
      break;
    default: printf("vp9_short_iht4x4_add_dspr2 : Invalid tx_type\n"); break;
  }
}
#endif  // #if HAVE_DSPR2
