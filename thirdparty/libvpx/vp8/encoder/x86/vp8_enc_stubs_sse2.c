/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "vpx_config.h"
#include "vp8_rtcd.h"
#include "vpx_ports/x86.h"
#include "vp8/encoder/block.h"

int vp8_mbblock_error_sse2_impl(short *coeff_ptr, short *dcoef_ptr, int dc);
int vp8_mbblock_error_sse2(MACROBLOCK *mb, int dc) {
  short *coeff_ptr = mb->block[0].coeff;
  short *dcoef_ptr = mb->e_mbd.block[0].dqcoeff;
  return vp8_mbblock_error_sse2_impl(coeff_ptr, dcoef_ptr, dc);
}

int vp8_mbuverror_sse2_impl(short *s_ptr, short *d_ptr);
int vp8_mbuverror_sse2(MACROBLOCK *mb) {
  short *s_ptr = &mb->coeff[256];
  short *d_ptr = &mb->e_mbd.dqcoeff[256];
  return vp8_mbuverror_sse2_impl(s_ptr, d_ptr);
}
