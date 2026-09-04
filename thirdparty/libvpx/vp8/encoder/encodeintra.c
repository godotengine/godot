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
#include "./vpx_dsp_rtcd.h"
#include "vp8/encoder/quantize.h"
#include "vp8/common/reconintra.h"
#include "vp8/common/reconintra4x4.h"
#include "encodemb.h"
#include "vp8/common/invtrans.h"
#include "encodeintra.h"

int vp8_encode_intra(MACROBLOCK *x, int use_dc_pred) {
  int i;
  int intra_pred_var = 0;

  if (use_dc_pred) {
    x->e_mbd.mode_info_context->mbmi.mode = DC_PRED;
    x->e_mbd.mode_info_context->mbmi.uv_mode = DC_PRED;
    x->e_mbd.mode_info_context->mbmi.ref_frame = INTRA_FRAME;

    vp8_encode_intra16x16mby(x);

    vp8_inverse_transform_mby(&x->e_mbd);
  } else {
    for (i = 0; i < 16; ++i) {
      x->e_mbd.block[i].bmi.as_mode = B_DC_PRED;
      vp8_encode_intra4x4block(x, i);
    }
  }

  intra_pred_var = vpx_get_mb_ss(x->src_diff);

  return intra_pred_var;
}

void vp8_encode_intra4x4block(MACROBLOCK *x, int ib) {
  BLOCKD *b = &x->e_mbd.block[ib];
  BLOCK *be = &x->block[ib];
  int dst_stride = x->e_mbd.dst.y_stride;
  unsigned char *dst = x->e_mbd.dst.y_buffer + b->offset;
  unsigned char *Above = dst - dst_stride;
  unsigned char *yleft = dst - 1;
  unsigned char top_left = Above[-1];

  vp8_intra4x4_predict(Above, yleft, dst_stride, b->bmi.as_mode, b->predictor,
                       16, top_left);

  vp8_subtract_b(be, b, 16);

  x->short_fdct4x4(be->src_diff, be->coeff, 32);

  x->quantize_b(be, b);

  if (*b->eob > 1) {
    vp8_short_idct4x4llm(b->dqcoeff, b->predictor, 16, dst, dst_stride);
  } else {
    vp8_dc_only_idct_add(b->dqcoeff[0], b->predictor, 16, dst, dst_stride);
  }
}

void vp8_encode_intra4x4mby(MACROBLOCK *mb) {
  int i;

  MACROBLOCKD *xd = &mb->e_mbd;
  intra_prediction_down_copy(xd, xd->dst.y_buffer - xd->dst.y_stride + 16);

  for (i = 0; i < 16; ++i) vp8_encode_intra4x4block(mb, i);
  return;
}

void vp8_encode_intra16x16mby(MACROBLOCK *x) {
  BLOCK *b = &x->block[0];
  MACROBLOCKD *xd = &x->e_mbd;

  vp8_build_intra_predictors_mby_s(xd, xd->dst.y_buffer - xd->dst.y_stride,
                                   xd->dst.y_buffer - 1, xd->dst.y_stride,
                                   xd->dst.y_buffer, xd->dst.y_stride);

  vp8_subtract_mby(x->src_diff, *(b->base_src), b->src_stride, xd->dst.y_buffer,
                   xd->dst.y_stride);

  vp8_transform_intra_mby(x);

  vp8_quantize_mby(x);

  if (x->optimize) vp8_optimize_mby(x);
}

void vp8_encode_intra16x16mbuv(MACROBLOCK *x) {
  MACROBLOCKD *xd = &x->e_mbd;

  vp8_build_intra_predictors_mbuv_s(xd, xd->dst.u_buffer - xd->dst.uv_stride,
                                    xd->dst.v_buffer - xd->dst.uv_stride,
                                    xd->dst.u_buffer - 1, xd->dst.v_buffer - 1,
                                    xd->dst.uv_stride, xd->dst.u_buffer,
                                    xd->dst.v_buffer, xd->dst.uv_stride);

  vp8_subtract_mbuv(x->src_diff, x->src.u_buffer, x->src.v_buffer,
                    x->src.uv_stride, xd->dst.u_buffer, xd->dst.v_buffer,
                    xd->dst.uv_stride);

  vp8_transform_mbuv(x);

  vp8_quantize_mbuv(x);

  if (x->optimize) vp8_optimize_mbuv(x);
}
