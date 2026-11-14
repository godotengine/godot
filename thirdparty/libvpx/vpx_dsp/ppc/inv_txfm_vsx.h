/*
 *  Copyright (c) 2018 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_PPC_INV_TXFM_VSX_H_
#define VPX_VPX_DSP_PPC_INV_TXFM_VSX_H_

#include "vpx_dsp/ppc/types_vsx.h"

void vpx_round_store4x4_vsx(int16x8_t *in, int16x8_t *out, uint8_t *dest,
                            int stride);
void vpx_idct4_vsx(int16x8_t *in, int16x8_t *out);
void vp9_iadst4_vsx(int16x8_t *in, int16x8_t *out);

void vpx_round_store8x8_vsx(int16x8_t *in, uint8_t *dest, int stride);
void vpx_idct8_vsx(int16x8_t *in, int16x8_t *out);
void vp9_iadst8_vsx(int16x8_t *in, int16x8_t *out);

#define LOAD_INPUT16(load, source, offset, step, in) \
  in[0] = load(offset, source);                      \
  in[1] = load((step) + (offset), source);           \
  in[2] = load(2 * (step) + (offset), source);       \
  in[3] = load(3 * (step) + (offset), source);       \
  in[4] = load(4 * (step) + (offset), source);       \
  in[5] = load(5 * (step) + (offset), source);       \
  in[6] = load(6 * (step) + (offset), source);       \
  in[7] = load(7 * (step) + (offset), source);       \
  in[8] = load(8 * (step) + (offset), source);       \
  in[9] = load(9 * (step) + (offset), source);       \
  in[10] = load(10 * (step) + (offset), source);     \
  in[11] = load(11 * (step) + (offset), source);     \
  in[12] = load(12 * (step) + (offset), source);     \
  in[13] = load(13 * (step) + (offset), source);     \
  in[14] = load(14 * (step) + (offset), source);     \
  in[15] = load(15 * (step) + (offset), source);

void vpx_round_store16x16_vsx(int16x8_t *src0, int16x8_t *src1, uint8_t *dest,
                              int stride);
void vpx_idct16_vsx(int16x8_t *src0, int16x8_t *src1);
void vpx_iadst16_vsx(int16x8_t *src0, int16x8_t *src1);

#endif  // VPX_VPX_DSP_PPC_INV_TXFM_VSX_H_
