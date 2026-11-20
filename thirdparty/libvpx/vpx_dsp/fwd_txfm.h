/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_FWD_TXFM_H_
#define VPX_VPX_DSP_FWD_TXFM_H_

#include "vpx_dsp/txfm_common.h"

static INLINE tran_high_t fdct_round_shift(tran_high_t input) {
  tran_high_t rv = ROUND_POWER_OF_TWO(input, DCT_CONST_BITS);
  // TODO(debargha, peter.derivaz): Find new bounds for this assert
  // and make the bounds consts.
  // assert(INT16_MIN <= rv && rv <= INT16_MAX);
  return rv;
}

void vpx_fdct32(const tran_high_t *input, tran_high_t *output, int round);
#endif  // VPX_VPX_DSP_FWD_TXFM_H_
