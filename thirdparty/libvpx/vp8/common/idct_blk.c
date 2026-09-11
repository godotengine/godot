/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "vpx_config.h"
#include "vp8_rtcd.h"
#include "vpx_mem/vpx_mem.h"

void vp8_dequant_idct_add_y_block_c(short *q, short *dq, unsigned char *dst,
                                    int stride, char *eobs) {
  int i, j;

  for (i = 0; i < 4; ++i) {
    for (j = 0; j < 4; ++j) {
      if (*eobs++ > 1) {
        vp8_dequant_idct_add_c(q, dq, dst, stride);
      } else {
        vp8_dc_only_idct_add_c(q[0] * dq[0], dst, stride, dst, stride);
        memset(q, 0, 2 * sizeof(q[0]));
      }

      q += 16;
      dst += 4;
    }

    dst += 4 * stride - 16;
  }
}

void vp8_dequant_idct_add_uv_block_c(short *q, short *dq, unsigned char *dst_u,
                                     unsigned char *dst_v, int stride,
                                     char *eobs) {
  int i, j;

  for (i = 0; i < 2; ++i) {
    for (j = 0; j < 2; ++j) {
      if (*eobs++ > 1) {
        vp8_dequant_idct_add_c(q, dq, dst_u, stride);
      } else {
        vp8_dc_only_idct_add_c(q[0] * dq[0], dst_u, stride, dst_u, stride);
        memset(q, 0, 2 * sizeof(q[0]));
      }

      q += 16;
      dst_u += 4;
    }

    dst_u += 4 * stride - 8;
  }

  for (i = 0; i < 2; ++i) {
    for (j = 0; j < 2; ++j) {
      if (*eobs++ > 1) {
        vp8_dequant_idct_add_c(q, dq, dst_v, stride);
      } else {
        vp8_dc_only_idct_add_c(q[0] * dq[0], dst_v, stride, dst_v, stride);
        memset(q, 0, 2 * sizeof(q[0]));
      }

      q += 16;
      dst_v += 4;
    }

    dst_v += 4 * stride - 8;
  }
}
