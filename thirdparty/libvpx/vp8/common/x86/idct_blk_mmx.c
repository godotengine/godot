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

extern void vp8_dequantize_b_impl_mmx(short *sq, short *dq, short *q);

void vp8_dequantize_b_mmx(BLOCKD *d, short *DQC) {
  short *sq = (short *)d->qcoeff;
  short *dq = (short *)d->dqcoeff;

  vp8_dequantize_b_impl_mmx(sq, dq, DQC);
}
