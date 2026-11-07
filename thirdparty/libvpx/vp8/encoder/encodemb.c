/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vpx_dsp_rtcd.h"

#include "vpx_config.h"
#include "vp8_rtcd.h"
#include "encodemb.h"
#include "vp8/common/reconinter.h"
#include "vp8/encoder/quantize.h"
#include "tokenize.h"
#include "vp8/common/invtrans.h"
#include "vpx_mem/vpx_mem.h"
#include "rdopt.h"

void vp8_subtract_b(BLOCK *be, BLOCKD *bd, int pitch) {
  unsigned char *src_ptr = (*(be->base_src) + be->src);
  short *diff_ptr = be->src_diff;
  unsigned char *pred_ptr = bd->predictor;
  int src_stride = be->src_stride;

  vpx_subtract_block(4, 4, diff_ptr, pitch, src_ptr, src_stride, pred_ptr,
                     pitch);
}

void vp8_subtract_mbuv(short *diff, unsigned char *usrc, unsigned char *vsrc,
                       int src_stride, unsigned char *upred,
                       unsigned char *vpred, int pred_stride) {
  short *udiff = diff + 256;
  short *vdiff = diff + 320;

  vpx_subtract_block(8, 8, udiff, 8, usrc, src_stride, upred, pred_stride);
  vpx_subtract_block(8, 8, vdiff, 8, vsrc, src_stride, vpred, pred_stride);
}

void vp8_subtract_mby(short *diff, unsigned char *src, int src_stride,
                      unsigned char *pred, int pred_stride) {
  vpx_subtract_block(16, 16, diff, 16, src, src_stride, pred, pred_stride);
}

static void vp8_subtract_mb(MACROBLOCK *x) {
  BLOCK *b = &x->block[0];

  vp8_subtract_mby(x->src_diff, *(b->base_src), b->src_stride,
                   x->e_mbd.dst.y_buffer, x->e_mbd.dst.y_stride);
  vp8_subtract_mbuv(x->src_diff, x->src.u_buffer, x->src.v_buffer,
                    x->src.uv_stride, x->e_mbd.dst.u_buffer,
                    x->e_mbd.dst.v_buffer, x->e_mbd.dst.uv_stride);
}

static void build_dcblock(MACROBLOCK *x) {
  short *src_diff_ptr = &x->src_diff[384];
  int i;

  for (i = 0; i < 16; ++i) {
    src_diff_ptr[i] = x->coeff[i * 16];
  }
}

void vp8_transform_mbuv(MACROBLOCK *x) {
  int i;

  for (i = 16; i < 24; i += 2) {
    x->short_fdct8x4(&x->block[i].src_diff[0], &x->block[i].coeff[0], 16);
  }
}

void vp8_transform_intra_mby(MACROBLOCK *x) {
  int i;

  for (i = 0; i < 16; i += 2) {
    x->short_fdct8x4(&x->block[i].src_diff[0], &x->block[i].coeff[0], 32);
  }

  /* build dc block from 16 y dc values */
  build_dcblock(x);

  /* do 2nd order transform on the dc block */
  x->short_walsh4x4(&x->block[24].src_diff[0], &x->block[24].coeff[0], 8);
}

static void transform_mb(MACROBLOCK *x) {
  int i;

  for (i = 0; i < 16; i += 2) {
    x->short_fdct8x4(&x->block[i].src_diff[0], &x->block[i].coeff[0], 32);
  }

  /* build dc block from 16 y dc values */
  if (x->e_mbd.mode_info_context->mbmi.mode != SPLITMV) build_dcblock(x);

  for (i = 16; i < 24; i += 2) {
    x->short_fdct8x4(&x->block[i].src_diff[0], &x->block[i].coeff[0], 16);
  }

  /* do 2nd order transform on the dc block */
  if (x->e_mbd.mode_info_context->mbmi.mode != SPLITMV) {
    x->short_walsh4x4(&x->block[24].src_diff[0], &x->block[24].coeff[0], 8);
  }
}

static void transform_mby(MACROBLOCK *x) {
  int i;

  for (i = 0; i < 16; i += 2) {
    x->short_fdct8x4(&x->block[i].src_diff[0], &x->block[i].coeff[0], 32);
  }

  /* build dc block from 16 y dc values */
  if (x->e_mbd.mode_info_context->mbmi.mode != SPLITMV) {
    build_dcblock(x);
    x->short_walsh4x4(&x->block[24].src_diff[0], &x->block[24].coeff[0], 8);
  }
}

#define RDTRUNC(RM, DM, R, D) ((128 + (R) * (RM)) & 0xFF)

typedef struct vp8_token_state vp8_token_state;

struct vp8_token_state {
  int rate;
  int error;
  signed char next;
  signed char token;
  short qc;
};

/* TODO: experiments to find optimal multiple numbers */
#define Y1_RD_MULT 4
#define UV_RD_MULT 2
#define Y2_RD_MULT 16

static const int plane_rd_mult[4] = { Y1_RD_MULT, Y2_RD_MULT, UV_RD_MULT,
                                      Y1_RD_MULT };

static void optimize_b(MACROBLOCK *mb, int ib, int type, ENTROPY_CONTEXT *a,
                       ENTROPY_CONTEXT *l) {
  BLOCK *b;
  BLOCKD *d;
  vp8_token_state tokens[17][2];
  unsigned best_mask[2];
  const short *dequant_ptr;
  const short *coeff_ptr;
  short *qcoeff_ptr;
  short *dqcoeff_ptr;
  int eob;
  int i0;
  int rc;
  int x;
  int sz = 0;
  int next;
  int rdmult;
  int rddiv;
  int final_eob;
  int rd_cost0;
  int rd_cost1;
  int rate0;
  int rate1;
  int error0;
  int error1;
  int t0;
  int t1;
  int best;
  int band;
  int pt;
  int i;
  int err_mult = plane_rd_mult[type];

  b = &mb->block[ib];
  d = &mb->e_mbd.block[ib];

  dequant_ptr = d->dequant;
  coeff_ptr = b->coeff;
  qcoeff_ptr = d->qcoeff;
  dqcoeff_ptr = d->dqcoeff;
  i0 = !type;
  eob = *d->eob;

  /* Now set up a Viterbi trellis to evaluate alternative roundings. */
  rdmult = mb->rdmult * err_mult;
  if (mb->e_mbd.mode_info_context->mbmi.ref_frame == INTRA_FRAME) {
    rdmult = (rdmult * 9) >> 4;
  }

  rddiv = mb->rddiv;
  best_mask[0] = best_mask[1] = 0;
  /* Initialize the sentinel node of the trellis. */
  tokens[eob][0].rate = 0;
  tokens[eob][0].error = 0;
  tokens[eob][0].next = 16;
  tokens[eob][0].token = DCT_EOB_TOKEN;
  tokens[eob][0].qc = 0;
  *(tokens[eob] + 1) = *(tokens[eob] + 0);
  next = eob;
  for (i = eob; i-- > i0;) {
    int base_bits;
    int d2;
    int dx;

    rc = vp8_default_zig_zag1d[i];
    x = qcoeff_ptr[rc];
    /* Only add a trellis state for non-zero coefficients. */
    if (x) {
      int shortcut = 0;
      error0 = tokens[next][0].error;
      error1 = tokens[next][1].error;
      /* Evaluate the first possibility for this state. */
      rate0 = tokens[next][0].rate;
      rate1 = tokens[next][1].rate;
      t0 = (vp8_dct_value_tokens_ptr + x)->Token;
      /* Consider both possible successor states. */
      if (next < 16) {
        band = vp8_coef_bands[i + 1];
        pt = vp8_prev_token_class[t0];
        rate0 += mb->token_costs[type][band][pt][tokens[next][0].token];
        rate1 += mb->token_costs[type][band][pt][tokens[next][1].token];
      }
      rd_cost0 = RDCOST(rdmult, rddiv, rate0, error0);
      rd_cost1 = RDCOST(rdmult, rddiv, rate1, error1);
      if (rd_cost0 == rd_cost1) {
        rd_cost0 = RDTRUNC(rdmult, rddiv, rate0, error0);
        rd_cost1 = RDTRUNC(rdmult, rddiv, rate1, error1);
      }
      /* And pick the best. */
      best = rd_cost1 < rd_cost0;
      base_bits = *(vp8_dct_value_cost_ptr + x);
      dx = dqcoeff_ptr[rc] - coeff_ptr[rc];
      d2 = dx * dx;
      tokens[i][0].rate = base_bits + (best ? rate1 : rate0);
      tokens[i][0].error = d2 + (best ? error1 : error0);
      tokens[i][0].next = next;
      tokens[i][0].token = t0;
      tokens[i][0].qc = x;
      best_mask[0] |= best << i;
      /* Evaluate the second possibility for this state. */
      rate0 = tokens[next][0].rate;
      rate1 = tokens[next][1].rate;

      if ((abs(x) * dequant_ptr[rc] > abs(coeff_ptr[rc])) &&
          (abs(x) * dequant_ptr[rc] < abs(coeff_ptr[rc]) + dequant_ptr[rc])) {
        shortcut = 1;
      } else {
        shortcut = 0;
      }

      if (shortcut) {
        sz = -(x < 0);
        x -= 2 * sz + 1;
      }

      /* Consider both possible successor states. */
      if (!x) {
        /* If we reduced this coefficient to zero, check to see if
         *  we need to move the EOB back here.
         */
        t0 =
            tokens[next][0].token == DCT_EOB_TOKEN ? DCT_EOB_TOKEN : ZERO_TOKEN;
        t1 =
            tokens[next][1].token == DCT_EOB_TOKEN ? DCT_EOB_TOKEN : ZERO_TOKEN;
      } else {
        t0 = t1 = (vp8_dct_value_tokens_ptr + x)->Token;
      }
      if (next < 16) {
        band = vp8_coef_bands[i + 1];
        if (t0 != DCT_EOB_TOKEN) {
          pt = vp8_prev_token_class[t0];
          rate0 += mb->token_costs[type][band][pt][tokens[next][0].token];
        }
        if (t1 != DCT_EOB_TOKEN) {
          pt = vp8_prev_token_class[t1];
          rate1 += mb->token_costs[type][band][pt][tokens[next][1].token];
        }
      }

      rd_cost0 = RDCOST(rdmult, rddiv, rate0, error0);
      rd_cost1 = RDCOST(rdmult, rddiv, rate1, error1);
      if (rd_cost0 == rd_cost1) {
        rd_cost0 = RDTRUNC(rdmult, rddiv, rate0, error0);
        rd_cost1 = RDTRUNC(rdmult, rddiv, rate1, error1);
      }
      /* And pick the best. */
      best = rd_cost1 < rd_cost0;
      base_bits = *(vp8_dct_value_cost_ptr + x);

      if (shortcut) {
        dx -= (dequant_ptr[rc] + sz) ^ sz;
        d2 = dx * dx;
      }
      tokens[i][1].rate = base_bits + (best ? rate1 : rate0);
      tokens[i][1].error = d2 + (best ? error1 : error0);
      tokens[i][1].next = next;
      tokens[i][1].token = best ? t1 : t0;
      tokens[i][1].qc = x;
      best_mask[1] |= best << i;
      /* Finally, make this the new head of the trellis. */
      next = i;
    }
    /* There's no choice to make for a zero coefficient, so we don't
     *  add a new trellis node, but we do need to update the costs.
     */
    else {
      band = vp8_coef_bands[i + 1];
      t0 = tokens[next][0].token;
      t1 = tokens[next][1].token;
      /* Update the cost of each path if we're past the EOB token. */
      if (t0 != DCT_EOB_TOKEN) {
        tokens[next][0].rate += mb->token_costs[type][band][0][t0];
        tokens[next][0].token = ZERO_TOKEN;
      }
      if (t1 != DCT_EOB_TOKEN) {
        tokens[next][1].rate += mb->token_costs[type][band][0][t1];
        tokens[next][1].token = ZERO_TOKEN;
      }
      /* Don't update next, because we didn't add a new node. */
    }
  }

  /* Now pick the best path through the whole trellis. */
  band = vp8_coef_bands[i + 1];
  VP8_COMBINEENTROPYCONTEXTS(pt, *a, *l);
  rate0 = tokens[next][0].rate;
  rate1 = tokens[next][1].rate;
  error0 = tokens[next][0].error;
  error1 = tokens[next][1].error;
  t0 = tokens[next][0].token;
  t1 = tokens[next][1].token;
  rate0 += mb->token_costs[type][band][pt][t0];
  rate1 += mb->token_costs[type][band][pt][t1];
  rd_cost0 = RDCOST(rdmult, rddiv, rate0, error0);
  rd_cost1 = RDCOST(rdmult, rddiv, rate1, error1);
  if (rd_cost0 == rd_cost1) {
    rd_cost0 = RDTRUNC(rdmult, rddiv, rate0, error0);
    rd_cost1 = RDTRUNC(rdmult, rddiv, rate1, error1);
  }
  best = rd_cost1 < rd_cost0;
  final_eob = i0 - 1;
  for (i = next; i < eob; i = next) {
    x = tokens[i][best].qc;
    if (x) final_eob = i;
    rc = vp8_default_zig_zag1d[i];
    qcoeff_ptr[rc] = x;
    dqcoeff_ptr[rc] = x * dequant_ptr[rc];
    next = tokens[i][best].next;
    best = (best_mask[best] >> i) & 1;
  }
  final_eob++;

  *a = *l = (final_eob != !type);
  *d->eob = (char)final_eob;
}
static void check_reset_2nd_coeffs(MACROBLOCKD *x, int type, ENTROPY_CONTEXT *a,
                                   ENTROPY_CONTEXT *l) {
  int sum = 0;
  int i;
  BLOCKD *bd = &x->block[24];

  if (bd->dequant[0] >= 35 && bd->dequant[1] >= 35) return;

  for (i = 0; i < (*bd->eob); ++i) {
    int coef = bd->dqcoeff[vp8_default_zig_zag1d[i]];
    sum += (coef >= 0) ? coef : -coef;
    if (sum >= 35) return;
  }
  /**************************************************************************
  our inverse hadamard transform effectively is weighted sum of all 16 inputs
  with weight either 1 or -1. It has a last stage scaling of (sum+3)>>3. And
  dc only idct is (dc+4)>>3. So if all the sums are between -35 and 29, the
  output after inverse wht and idct will be all zero. A sum of absolute value
  smaller than 35 guarantees all 16 different (+1/-1) weighted sums in wht
  fall between -35 and +35.
  **************************************************************************/
  if (sum < 35) {
    for (i = 0; i < (*bd->eob); ++i) {
      int rc = vp8_default_zig_zag1d[i];
      bd->qcoeff[rc] = 0;
      bd->dqcoeff[rc] = 0;
    }
    *bd->eob = 0;
    *a = *l = (*bd->eob != !type);
  }
}

static void optimize_mb(MACROBLOCK *x) {
  int b;
  int type;
  int has_2nd_order;

  ENTROPY_CONTEXT_PLANES t_above, t_left;
  ENTROPY_CONTEXT *ta;
  ENTROPY_CONTEXT *tl;

  t_above = *x->e_mbd.above_context;
  t_left = *x->e_mbd.left_context;

  ta = (ENTROPY_CONTEXT *)&t_above;
  tl = (ENTROPY_CONTEXT *)&t_left;

  has_2nd_order = (x->e_mbd.mode_info_context->mbmi.mode != B_PRED &&
                   x->e_mbd.mode_info_context->mbmi.mode != SPLITMV);
  type = has_2nd_order ? PLANE_TYPE_Y_NO_DC : PLANE_TYPE_Y_WITH_DC;

  for (b = 0; b < 16; ++b) {
    optimize_b(x, b, type, ta + vp8_block2above[b], tl + vp8_block2left[b]);
  }

  for (b = 16; b < 24; ++b) {
    optimize_b(x, b, PLANE_TYPE_UV, ta + vp8_block2above[b],
               tl + vp8_block2left[b]);
  }

  if (has_2nd_order) {
    b = 24;
    optimize_b(x, b, PLANE_TYPE_Y2, ta + vp8_block2above[b],
               tl + vp8_block2left[b]);
    check_reset_2nd_coeffs(&x->e_mbd, PLANE_TYPE_Y2, ta + vp8_block2above[b],
                           tl + vp8_block2left[b]);
  }
}

void vp8_optimize_mby(MACROBLOCK *x) {
  int b;
  int type;
  int has_2nd_order;

  ENTROPY_CONTEXT_PLANES t_above, t_left;
  ENTROPY_CONTEXT *ta;
  ENTROPY_CONTEXT *tl;

  if (!x->e_mbd.above_context) return;

  if (!x->e_mbd.left_context) return;

  t_above = *x->e_mbd.above_context;
  t_left = *x->e_mbd.left_context;

  ta = (ENTROPY_CONTEXT *)&t_above;
  tl = (ENTROPY_CONTEXT *)&t_left;

  has_2nd_order = (x->e_mbd.mode_info_context->mbmi.mode != B_PRED &&
                   x->e_mbd.mode_info_context->mbmi.mode != SPLITMV);
  type = has_2nd_order ? PLANE_TYPE_Y_NO_DC : PLANE_TYPE_Y_WITH_DC;

  for (b = 0; b < 16; ++b) {
    optimize_b(x, b, type, ta + vp8_block2above[b], tl + vp8_block2left[b]);
  }

  if (has_2nd_order) {
    b = 24;
    optimize_b(x, b, PLANE_TYPE_Y2, ta + vp8_block2above[b],
               tl + vp8_block2left[b]);
    check_reset_2nd_coeffs(&x->e_mbd, PLANE_TYPE_Y2, ta + vp8_block2above[b],
                           tl + vp8_block2left[b]);
  }
}

void vp8_optimize_mbuv(MACROBLOCK *x) {
  int b;
  ENTROPY_CONTEXT_PLANES t_above, t_left;
  ENTROPY_CONTEXT *ta;
  ENTROPY_CONTEXT *tl;

  if (!x->e_mbd.above_context) return;

  if (!x->e_mbd.left_context) return;

  t_above = *x->e_mbd.above_context;
  t_left = *x->e_mbd.left_context;

  ta = (ENTROPY_CONTEXT *)&t_above;
  tl = (ENTROPY_CONTEXT *)&t_left;

  for (b = 16; b < 24; ++b) {
    optimize_b(x, b, PLANE_TYPE_UV, ta + vp8_block2above[b],
               tl + vp8_block2left[b]);
  }
}

void vp8_encode_inter16x16(MACROBLOCK *x) {
  vp8_build_inter_predictors_mb(&x->e_mbd);

  vp8_subtract_mb(x);

  transform_mb(x);

  vp8_quantize_mb(x);

  if (x->optimize) optimize_mb(x);
}

/* this funciton is used by first pass only */
void vp8_encode_inter16x16y(MACROBLOCK *x) {
  BLOCK *b = &x->block[0];

  vp8_build_inter16x16_predictors_mby(&x->e_mbd, x->e_mbd.dst.y_buffer,
                                      x->e_mbd.dst.y_stride);

  vp8_subtract_mby(x->src_diff, *(b->base_src), b->src_stride,
                   x->e_mbd.dst.y_buffer, x->e_mbd.dst.y_stride);

  transform_mby(x);

  vp8_quantize_mby(x);

  vp8_inverse_transform_mby(&x->e_mbd);
}
