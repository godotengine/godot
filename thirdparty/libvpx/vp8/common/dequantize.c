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
#include "vp8/common/blockd.h"
#include "vpx_mem/vpx_mem.h"

void vp8_dequantize_b_c(BLOCKD *d, short *DQC) {
  int i;
  short *DQ = d->dqcoeff;
  short *Q = d->qcoeff;

  for (i = 0; i < 16; ++i) {
    DQ[i] = Q[i] * DQC[i];
  }
}

void vp8_dequant_idct_add_c(short *input, short *dq, unsigned char *dest,
                            int stride) {
  int i;

  for (i = 0; i < 16; ++i) {
    input[i] = dq[i] * input[i];
  }

  vp8_short_idct4x4llm_c(input, dest, stride, dest, stride);

  memset(input, 0, 32);
}
