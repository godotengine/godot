/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP8_COMMON_INVTRANS_H_
#define VPX_VP8_COMMON_INVTRANS_H_

#include "./vpx_config.h"
#include "vp8_rtcd.h"
#include "blockd.h"
#include "onyxc_int.h"

#if CONFIG_MULTITHREAD
#include "vpx_mem/vpx_mem.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

static void eob_adjust(char *eobs, short *diff) {
  /* eob adjust.... the idct can only skip if both the dc and eob are zero */
  int js;
  for (js = 0; js < 16; ++js) {
    if ((eobs[js] == 0) && (diff[0] != 0)) eobs[js]++;
    diff += 16;
  }
}

static INLINE void vp8_inverse_transform_mby(MACROBLOCKD *xd) {
  short *DQC = xd->dequant_y1;

  if (xd->mode_info_context->mbmi.mode != SPLITMV) {
    /* do 2nd order transform on the dc block */
    if (xd->eobs[24] > 1) {
      vp8_short_inv_walsh4x4(&xd->block[24].dqcoeff[0], xd->qcoeff);
    } else {
      vp8_short_inv_walsh4x4_1(&xd->block[24].dqcoeff[0], xd->qcoeff);
    }
    eob_adjust(xd->eobs, xd->qcoeff);

    DQC = xd->dequant_y1_dc;
  }
  vp8_dequant_idct_add_y_block(xd->qcoeff, DQC, xd->dst.y_buffer,
                               xd->dst.y_stride, xd->eobs);
}
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_COMMON_INVTRANS_H_
