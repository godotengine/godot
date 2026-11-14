/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "onyx_int.h"
#include "tokenize.h"
#include "vpx_mem/vpx_mem.h"

/* Global event counters used for accumulating statistics across several
   compressions, then generating context.c = initial stats. */

void vp8_stuff_mb(VP8_COMP *cpi, MACROBLOCK *x, TOKENEXTRA **t);
void vp8_fix_contexts(MACROBLOCKD *x);

#include "dct_value_tokens.h"
#include "dct_value_cost.h"

const TOKENVALUE *const vp8_dct_value_tokens_ptr =
    dct_value_tokens + DCT_MAX_VALUE;
const short *const vp8_dct_value_cost_ptr = dct_value_cost + DCT_MAX_VALUE;

#if 0
int skip_true_count = 0;
int skip_false_count = 0;
#endif

/* function used to generate dct_value_tokens and dct_value_cost tables */
/*
static void fill_value_tokens()
{

    TOKENVALUE *t = dct_value_tokens + DCT_MAX_VALUE;
    const vp8_extra_bit_struct *e = vp8_extra_bits;

    int i = -DCT_MAX_VALUE;
    int sign = 1;

    do
    {
        if (!i)
            sign = 0;

        {
            const int a = sign ? -i : i;
            int eb = sign;

            if (a > 4)
            {
                int j = 4;

                while (++j < 11  &&  e[j].base_val <= a) {}

                t[i].Token = --j;
                eb |= (a - e[j].base_val) << 1;
            }
            else
                t[i].Token = a;

            t[i].Extra = eb;
        }

        // initialize the cost for extra bits for all possible coefficient
value.
        {
            int cost = 0;
            const vp8_extra_bit_struct *p = vp8_extra_bits + t[i].Token;

            if (p->base_val)
            {
                const int extra = t[i].Extra;
                const int Length = p->Len;

                if (Length)
                    cost += vp8_treed_cost(p->tree, p->prob, extra >> 1,
Length);

                cost += vp8_cost_bit(vp8_prob_half, extra & 1); // sign
                dct_value_cost[i + DCT_MAX_VALUE] = cost;
            }

        }

    }
    while (++i < DCT_MAX_VALUE);

    vp8_dct_value_tokens_ptr = dct_value_tokens + DCT_MAX_VALUE;
    vp8_dct_value_cost_ptr   = dct_value_cost + DCT_MAX_VALUE;
}
*/

static void tokenize2nd_order_b(MACROBLOCK *x, TOKENEXTRA **tp, VP8_COMP *cpi) {
  MACROBLOCKD *xd = &x->e_mbd;
  int pt;              /* near block/prev token context index */
  int c;               /* start at DC */
  TOKENEXTRA *t = *tp; /* store tokens starting here */
  const BLOCKD *b;
  const short *qcoeff_ptr;
  ENTROPY_CONTEXT *a;
  ENTROPY_CONTEXT *l;
  int band, rc, v, token;
  int eob;

  b = xd->block + 24;
  qcoeff_ptr = b->qcoeff;
  a = (ENTROPY_CONTEXT *)xd->above_context + 8;
  l = (ENTROPY_CONTEXT *)xd->left_context + 8;
  eob = xd->eobs[24];
  VP8_COMBINEENTROPYCONTEXTS(pt, *a, *l);

  if (!eob) {
    /* c = band for this case */
    t->Token = DCT_EOB_TOKEN;
    t->context_tree = cpi->common.fc.coef_probs[1][0][pt];
    t->skip_eob_node = 0;

    ++x->coef_counts[1][0][pt][DCT_EOB_TOKEN];
    t++;
    *tp = t;
    *a = *l = 0;
    return;
  }

  v = qcoeff_ptr[0];
  t->Extra = vp8_dct_value_tokens_ptr[v].Extra;
  token = vp8_dct_value_tokens_ptr[v].Token;
  t->Token = token;

  t->context_tree = cpi->common.fc.coef_probs[1][0][pt];
  t->skip_eob_node = 0;
  ++x->coef_counts[1][0][pt][token];
  pt = vp8_prev_token_class[token];
  t++;
  c = 1;

  for (; c < eob; ++c) {
    rc = vp8_default_zig_zag1d[c];
    band = vp8_coef_bands[c];
    v = qcoeff_ptr[rc];

    t->Extra = vp8_dct_value_tokens_ptr[v].Extra;
    token = vp8_dct_value_tokens_ptr[v].Token;

    t->Token = token;
    t->context_tree = cpi->common.fc.coef_probs[1][band][pt];

    t->skip_eob_node = ((pt == 0));

    ++x->coef_counts[1][band][pt][token];

    pt = vp8_prev_token_class[token];
    t++;
  }
  if (c < 16) {
    band = vp8_coef_bands[c];
    t->Token = DCT_EOB_TOKEN;
    t->context_tree = cpi->common.fc.coef_probs[1][band][pt];

    t->skip_eob_node = 0;

    ++x->coef_counts[1][band][pt][DCT_EOB_TOKEN];

    t++;
  }

  *tp = t;
  *a = *l = 1;
}

static void tokenize1st_order_b(
    MACROBLOCK *x, TOKENEXTRA **tp,
    int type, /* which plane: 0=Y no DC, 1=Y2, 2=UV, 3=Y with DC */
    VP8_COMP *cpi) {
  MACROBLOCKD *xd = &x->e_mbd;
  unsigned int block;
  const BLOCKD *b;
  int pt; /* near block/prev token context index */
  int c;
  int token;
  TOKENEXTRA *t = *tp; /* store tokens starting here */
  const short *qcoeff_ptr;
  ENTROPY_CONTEXT *a;
  ENTROPY_CONTEXT *l;
  int band, rc, v;
  int tmp1, tmp2;

  b = xd->block;
  /* Luma */
  for (block = 0; block < 16; block++, b++) {
    const int eob = *b->eob;
    tmp1 = vp8_block2above[block];
    tmp2 = vp8_block2left[block];
    qcoeff_ptr = b->qcoeff;
    a = (ENTROPY_CONTEXT *)xd->above_context + tmp1;
    l = (ENTROPY_CONTEXT *)xd->left_context + tmp2;

    VP8_COMBINEENTROPYCONTEXTS(pt, *a, *l);

    c = type ? 0 : 1;

    if (c >= eob) {
      /* c = band for this case */
      t->Token = DCT_EOB_TOKEN;
      t->context_tree = cpi->common.fc.coef_probs[type][c][pt];
      t->skip_eob_node = 0;

      ++x->coef_counts[type][c][pt][DCT_EOB_TOKEN];
      t++;
      *tp = t;
      *a = *l = 0;
      continue;
    }

    v = qcoeff_ptr[c];

    t->Extra = vp8_dct_value_tokens_ptr[v].Extra;
    token = vp8_dct_value_tokens_ptr[v].Token;
    t->Token = token;

    t->context_tree = cpi->common.fc.coef_probs[type][c][pt];
    t->skip_eob_node = 0;
    ++x->coef_counts[type][c][pt][token];
    pt = vp8_prev_token_class[token];
    t++;
    c++;

    assert(eob <= 16);
    for (; c < eob; ++c) {
      rc = vp8_default_zig_zag1d[c];
      band = vp8_coef_bands[c];
      v = qcoeff_ptr[rc];

      t->Extra = vp8_dct_value_tokens_ptr[v].Extra;
      token = vp8_dct_value_tokens_ptr[v].Token;

      t->Token = token;
      t->context_tree = cpi->common.fc.coef_probs[type][band][pt];

      t->skip_eob_node = (pt == 0);
      ++x->coef_counts[type][band][pt][token];

      pt = vp8_prev_token_class[token];
      t++;
    }
    if (c < 16) {
      band = vp8_coef_bands[c];
      t->Token = DCT_EOB_TOKEN;
      t->context_tree = cpi->common.fc.coef_probs[type][band][pt];

      t->skip_eob_node = 0;
      ++x->coef_counts[type][band][pt][DCT_EOB_TOKEN];

      t++;
    }
    *tp = t;
    *a = *l = 1;
  }

  /* Chroma */
  for (block = 16; block < 24; block++, b++) {
    const int eob = *b->eob;
    tmp1 = vp8_block2above[block];
    tmp2 = vp8_block2left[block];
    qcoeff_ptr = b->qcoeff;
    a = (ENTROPY_CONTEXT *)xd->above_context + tmp1;
    l = (ENTROPY_CONTEXT *)xd->left_context + tmp2;

    VP8_COMBINEENTROPYCONTEXTS(pt, *a, *l);

    if (!eob) {
      /* c = band for this case */
      t->Token = DCT_EOB_TOKEN;
      t->context_tree = cpi->common.fc.coef_probs[2][0][pt];
      t->skip_eob_node = 0;

      ++x->coef_counts[2][0][pt][DCT_EOB_TOKEN];
      t++;
      *tp = t;
      *a = *l = 0;
      continue;
    }

    v = qcoeff_ptr[0];

    t->Extra = vp8_dct_value_tokens_ptr[v].Extra;
    token = vp8_dct_value_tokens_ptr[v].Token;
    t->Token = token;

    t->context_tree = cpi->common.fc.coef_probs[2][0][pt];
    t->skip_eob_node = 0;
    ++x->coef_counts[2][0][pt][token];
    pt = vp8_prev_token_class[token];
    t++;
    c = 1;

    assert(eob <= 16);
    for (; c < eob; ++c) {
      rc = vp8_default_zig_zag1d[c];
      band = vp8_coef_bands[c];
      v = qcoeff_ptr[rc];

      t->Extra = vp8_dct_value_tokens_ptr[v].Extra;
      token = vp8_dct_value_tokens_ptr[v].Token;

      t->Token = token;
      t->context_tree = cpi->common.fc.coef_probs[2][band][pt];

      t->skip_eob_node = (pt == 0);

      ++x->coef_counts[2][band][pt][token];

      pt = vp8_prev_token_class[token];
      t++;
    }
    if (c < 16) {
      band = vp8_coef_bands[c];
      t->Token = DCT_EOB_TOKEN;
      t->context_tree = cpi->common.fc.coef_probs[2][band][pt];

      t->skip_eob_node = 0;

      ++x->coef_counts[2][band][pt][DCT_EOB_TOKEN];

      t++;
    }
    *tp = t;
    *a = *l = 1;
  }
}

static int mb_is_skippable(MACROBLOCKD *x, int has_y2_block) {
  int skip = 1;
  int i = 0;

  if (has_y2_block) {
    for (i = 0; i < 16; ++i) skip &= (x->eobs[i] < 2);
  }

  for (; i < 24 + has_y2_block; ++i) skip &= (!x->eobs[i]);

  return skip;
}

void vp8_tokenize_mb(VP8_COMP *cpi, MACROBLOCK *x, TOKENEXTRA **t) {
  MACROBLOCKD *xd = &x->e_mbd;
  int plane_type;
  int has_y2_block;

  has_y2_block = (xd->mode_info_context->mbmi.mode != B_PRED &&
                  xd->mode_info_context->mbmi.mode != SPLITMV);

  xd->mode_info_context->mbmi.mb_skip_coeff = mb_is_skippable(xd, has_y2_block);
  if (xd->mode_info_context->mbmi.mb_skip_coeff) {
    if (!cpi->common.mb_no_coeff_skip) {
      vp8_stuff_mb(cpi, x, t);
    } else {
      vp8_fix_contexts(xd);
      x->skip_true_count++;
    }

    return;
  }

  plane_type = 3;
  if (has_y2_block) {
    tokenize2nd_order_b(x, t, cpi);
    plane_type = 0;
  }

  tokenize1st_order_b(x, t, plane_type, cpi);
}

static void stuff2nd_order_b(TOKENEXTRA **tp, ENTROPY_CONTEXT *a,
                             ENTROPY_CONTEXT *l, VP8_COMP *cpi, MACROBLOCK *x) {
  int pt;              /* near block/prev token context index */
  TOKENEXTRA *t = *tp; /* store tokens starting here */
  VP8_COMBINEENTROPYCONTEXTS(pt, *a, *l);

  t->Token = DCT_EOB_TOKEN;
  t->context_tree = cpi->common.fc.coef_probs[1][0][pt];
  t->skip_eob_node = 0;
  ++x->coef_counts[1][0][pt][DCT_EOB_TOKEN];
  ++t;

  *tp = t;
  pt = 0;
  *a = *l = pt;
}

static void stuff1st_order_b(TOKENEXTRA **tp, ENTROPY_CONTEXT *a,
                             ENTROPY_CONTEXT *l, int type, VP8_COMP *cpi,
                             MACROBLOCK *x) {
  int pt; /* near block/prev token context index */
  int band;
  TOKENEXTRA *t = *tp; /* store tokens starting here */
  VP8_COMBINEENTROPYCONTEXTS(pt, *a, *l);
  band = type ? 0 : 1;
  t->Token = DCT_EOB_TOKEN;
  t->context_tree = cpi->common.fc.coef_probs[type][band][pt];
  t->skip_eob_node = 0;
  ++x->coef_counts[type][band][pt][DCT_EOB_TOKEN];
  ++t;
  *tp = t;
  pt = 0; /* 0 <-> all coeff data is zero */
  *a = *l = pt;
}

static void stuff1st_order_buv(TOKENEXTRA **tp, ENTROPY_CONTEXT *a,
                               ENTROPY_CONTEXT *l, VP8_COMP *cpi,
                               MACROBLOCK *x) {
  int pt;              /* near block/prev token context index */
  TOKENEXTRA *t = *tp; /* store tokens starting here */
  VP8_COMBINEENTROPYCONTEXTS(pt, *a, *l);

  t->Token = DCT_EOB_TOKEN;
  t->context_tree = cpi->common.fc.coef_probs[2][0][pt];
  t->skip_eob_node = 0;
  ++x->coef_counts[2][0][pt][DCT_EOB_TOKEN];
  ++t;
  *tp = t;
  pt = 0; /* 0 <-> all coeff data is zero */
  *a = *l = pt;
}

void vp8_stuff_mb(VP8_COMP *cpi, MACROBLOCK *x, TOKENEXTRA **t) {
  MACROBLOCKD *xd = &x->e_mbd;
  ENTROPY_CONTEXT *A = (ENTROPY_CONTEXT *)xd->above_context;
  ENTROPY_CONTEXT *L = (ENTROPY_CONTEXT *)xd->left_context;
  int plane_type;
  int b;
  plane_type = 3;
  if ((xd->mode_info_context->mbmi.mode != B_PRED &&
       xd->mode_info_context->mbmi.mode != SPLITMV)) {
    stuff2nd_order_b(t, A + vp8_block2above[24], L + vp8_block2left[24], cpi,
                     x);
    plane_type = 0;
  }

  for (b = 0; b < 16; ++b) {
    stuff1st_order_b(t, A + vp8_block2above[b], L + vp8_block2left[b],
                     plane_type, cpi, x);
  }

  for (b = 16; b < 24; ++b) {
    stuff1st_order_buv(t, A + vp8_block2above[b], L + vp8_block2left[b], cpi,
                       x);
  }
}
void vp8_fix_contexts(MACROBLOCKD *x) {
  /* Clear entropy contexts for Y2 blocks */
  if (x->mode_info_context->mbmi.mode != B_PRED &&
      x->mode_info_context->mbmi.mode != SPLITMV) {
    memset(x->above_context, 0, sizeof(ENTROPY_CONTEXT_PLANES));
    memset(x->left_context, 0, sizeof(ENTROPY_CONTEXT_PLANES));
  } else {
    memset(x->above_context, 0, sizeof(ENTROPY_CONTEXT_PLANES) - 1);
    memset(x->left_context, 0, sizeof(ENTROPY_CONTEXT_PLANES) - 1);
  }
}
