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

#include "./vpx_config.h"
#include "vpx_ports/bitops.h"
#include "vpx_mem/vpx_mem.h"

#include "onyx_int.h"
#include "vp8/encoder/quantize.h"
#include "vp8/common/quant_common.h"

void vp8_fast_quantize_b_c(BLOCK *b, BLOCKD *d) {
  int i, rc, eob;
  int x, y, z, sz;
  short *coeff_ptr = b->coeff;
  short *round_ptr = b->round;
  short *quant_ptr = b->quant_fast;
  short *qcoeff_ptr = d->qcoeff;
  short *dqcoeff_ptr = d->dqcoeff;
  short *dequant_ptr = d->dequant;

  eob = -1;
  for (i = 0; i < 16; ++i) {
    rc = vp8_default_zig_zag1d[i];
    z = coeff_ptr[rc];

    sz = (z >> 31);    /* sign of z */
    x = (z ^ sz) - sz; /* x = abs(z) */

    y = ((x + round_ptr[rc]) * quant_ptr[rc]) >> 16; /* quantize (x) */
    x = (y ^ sz) - sz;                               /* get the sign back */
    qcoeff_ptr[rc] = x;                              /* write to destination */
    dqcoeff_ptr[rc] = x * dequant_ptr[rc];           /* dequantized value */

    if (y) {
      eob = i; /* last nonzero coeffs */
    }
  }
  *d->eob = (char)(eob + 1);
}

void vp8_regular_quantize_b_c(BLOCK *b, BLOCKD *d) {
  int i, rc, eob;
  int zbin;
  int x, y, z, sz;
  short *zbin_boost_ptr = b->zrun_zbin_boost;
  short *coeff_ptr = b->coeff;
  short *zbin_ptr = b->zbin;
  short *round_ptr = b->round;
  short *quant_ptr = b->quant;
  short *quant_shift_ptr = b->quant_shift;
  short *qcoeff_ptr = d->qcoeff;
  short *dqcoeff_ptr = d->dqcoeff;
  short *dequant_ptr = d->dequant;
  short zbin_oq_value = b->zbin_extra;

  memset(qcoeff_ptr, 0, 32);
  memset(dqcoeff_ptr, 0, 32);

  eob = -1;

  for (i = 0; i < 16; ++i) {
    rc = vp8_default_zig_zag1d[i];
    z = coeff_ptr[rc];

    zbin = zbin_ptr[rc] + *zbin_boost_ptr + zbin_oq_value;

    zbin_boost_ptr++;
    sz = (z >> 31);    /* sign of z */
    x = (z ^ sz) - sz; /* x = abs(z) */

    if (x >= zbin) {
      x += round_ptr[rc];
      y = ((((x * quant_ptr[rc]) >> 16) + x) * quant_shift_ptr[rc]) >>
          16;                                /* quantize (x) */
      x = (y ^ sz) - sz;                     /* get the sign back */
      qcoeff_ptr[rc] = x;                    /* write to destination */
      dqcoeff_ptr[rc] = x * dequant_ptr[rc]; /* dequantized value */

      if (y) {
        eob = i;                             /* last nonzero coeffs */
        zbin_boost_ptr = b->zrun_zbin_boost; /* reset zero runlength */
      }
    }
  }

  *d->eob = (char)(eob + 1);
}

void vp8_quantize_mby(MACROBLOCK *x) {
  int i;
  int has_2nd_order = (x->e_mbd.mode_info_context->mbmi.mode != B_PRED &&
                       x->e_mbd.mode_info_context->mbmi.mode != SPLITMV);

  for (i = 0; i < 16; ++i) x->quantize_b(&x->block[i], &x->e_mbd.block[i]);

  if (has_2nd_order) x->quantize_b(&x->block[24], &x->e_mbd.block[24]);
}

void vp8_quantize_mb(MACROBLOCK *x) {
  int i;
  int has_2nd_order = (x->e_mbd.mode_info_context->mbmi.mode != B_PRED &&
                       x->e_mbd.mode_info_context->mbmi.mode != SPLITMV);

  for (i = 0; i < 24 + has_2nd_order; ++i) {
    x->quantize_b(&x->block[i], &x->e_mbd.block[i]);
  }
}

void vp8_quantize_mbuv(MACROBLOCK *x) {
  int i;

  for (i = 16; i < 24; ++i) x->quantize_b(&x->block[i], &x->e_mbd.block[i]);
}

static const int qrounding_factors[129] = {
  48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48,
  48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48,
  48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48,
  48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48,
  48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48,
  48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48,
  48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48
};

static const int qzbin_factors[129] = {
  84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84,
  84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84,
  84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 80, 80, 80, 80, 80, 80, 80, 80, 80,
  80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80,
  80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80,
  80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80,
  80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80
};

static const int qrounding_factors_y2[129] = {
  48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48,
  48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48,
  48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48,
  48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48,
  48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48,
  48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48,
  48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48
};

static const int qzbin_factors_y2[129] = {
  84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84,
  84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84,
  84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 80, 80, 80, 80, 80, 80, 80, 80, 80,
  80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80,
  80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80,
  80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80,
  80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80
};

static void invert_quant(int improved_quant, short *quant, short *shift,
                         short d) {
  if (improved_quant) {
    unsigned int t;
    int l, m;
    t = (unsigned int)d;
    l = get_msb(t);
    m = 1 + (1 << (16 + l)) / d;
    *quant = (short)(m - (1 << 16));
    *shift = l;
    /* use multiplication and constant shift by 16 */
    *shift = 1 << (16 - *shift);
  } else {
    *quant = (1 << 16) / d;
    *shift = 0;
  }
}

void vp8cx_init_quantizer(VP8_COMP *cpi) {
  int i;
  int quant_val;
  int Q;

  int zbin_boost[16] = { 0,  0,  8,  10, 12, 14, 16, 20,
                         24, 28, 32, 36, 40, 44, 44, 44 };

  for (Q = 0; Q < QINDEX_RANGE; ++Q) {
    /* dc values */
    quant_val = vp8_dc_quant(Q, cpi->common.y1dc_delta_q);
    cpi->Y1quant_fast[Q][0] = (1 << 16) / quant_val;
    invert_quant(cpi->sf.improved_quant, cpi->Y1quant[Q] + 0,
                 cpi->Y1quant_shift[Q] + 0, quant_val);
    cpi->Y1zbin[Q][0] = ((qzbin_factors[Q] * quant_val) + 64) >> 7;
    cpi->Y1round[Q][0] = (qrounding_factors[Q] * quant_val) >> 7;
    cpi->common.Y1dequant[Q][0] = quant_val;
    cpi->zrun_zbin_boost_y1[Q][0] = (quant_val * zbin_boost[0]) >> 7;

    quant_val = vp8_dc2quant(Q, cpi->common.y2dc_delta_q);
    cpi->Y2quant_fast[Q][0] = (1 << 16) / quant_val;
    invert_quant(cpi->sf.improved_quant, cpi->Y2quant[Q] + 0,
                 cpi->Y2quant_shift[Q] + 0, quant_val);
    cpi->Y2zbin[Q][0] = ((qzbin_factors_y2[Q] * quant_val) + 64) >> 7;
    cpi->Y2round[Q][0] = (qrounding_factors_y2[Q] * quant_val) >> 7;
    cpi->common.Y2dequant[Q][0] = quant_val;
    cpi->zrun_zbin_boost_y2[Q][0] = (quant_val * zbin_boost[0]) >> 7;

    quant_val = vp8_dc_uv_quant(Q, cpi->common.uvdc_delta_q);
    cpi->UVquant_fast[Q][0] = (1 << 16) / quant_val;
    invert_quant(cpi->sf.improved_quant, cpi->UVquant[Q] + 0,
                 cpi->UVquant_shift[Q] + 0, quant_val);
    cpi->UVzbin[Q][0] = ((qzbin_factors[Q] * quant_val) + 64) >> 7;
    cpi->UVround[Q][0] = (qrounding_factors[Q] * quant_val) >> 7;
    cpi->common.UVdequant[Q][0] = quant_val;
    cpi->zrun_zbin_boost_uv[Q][0] = (quant_val * zbin_boost[0]) >> 7;

    /* all the ac values = ; */
    quant_val = vp8_ac_yquant(Q);
    cpi->Y1quant_fast[Q][1] = (1 << 16) / quant_val;
    invert_quant(cpi->sf.improved_quant, cpi->Y1quant[Q] + 1,
                 cpi->Y1quant_shift[Q] + 1, quant_val);
    cpi->Y1zbin[Q][1] = ((qzbin_factors[Q] * quant_val) + 64) >> 7;
    cpi->Y1round[Q][1] = (qrounding_factors[Q] * quant_val) >> 7;
    cpi->common.Y1dequant[Q][1] = quant_val;
    cpi->zrun_zbin_boost_y1[Q][1] = (quant_val * zbin_boost[1]) >> 7;

    quant_val = vp8_ac2quant(Q, cpi->common.y2ac_delta_q);
    cpi->Y2quant_fast[Q][1] = (1 << 16) / quant_val;
    invert_quant(cpi->sf.improved_quant, cpi->Y2quant[Q] + 1,
                 cpi->Y2quant_shift[Q] + 1, quant_val);
    cpi->Y2zbin[Q][1] = ((qzbin_factors_y2[Q] * quant_val) + 64) >> 7;
    cpi->Y2round[Q][1] = (qrounding_factors_y2[Q] * quant_val) >> 7;
    cpi->common.Y2dequant[Q][1] = quant_val;
    cpi->zrun_zbin_boost_y2[Q][1] = (quant_val * zbin_boost[1]) >> 7;

    quant_val = vp8_ac_uv_quant(Q, cpi->common.uvac_delta_q);
    cpi->UVquant_fast[Q][1] = (1 << 16) / quant_val;
    invert_quant(cpi->sf.improved_quant, cpi->UVquant[Q] + 1,
                 cpi->UVquant_shift[Q] + 1, quant_val);
    cpi->UVzbin[Q][1] = ((qzbin_factors[Q] * quant_val) + 64) >> 7;
    cpi->UVround[Q][1] = (qrounding_factors[Q] * quant_val) >> 7;
    cpi->common.UVdequant[Q][1] = quant_val;
    cpi->zrun_zbin_boost_uv[Q][1] = (quant_val * zbin_boost[1]) >> 7;

    for (i = 2; i < 16; ++i) {
      cpi->Y1quant_fast[Q][i] = cpi->Y1quant_fast[Q][1];
      cpi->Y1quant[Q][i] = cpi->Y1quant[Q][1];
      cpi->Y1quant_shift[Q][i] = cpi->Y1quant_shift[Q][1];
      cpi->Y1zbin[Q][i] = cpi->Y1zbin[Q][1];
      cpi->Y1round[Q][i] = cpi->Y1round[Q][1];
      cpi->zrun_zbin_boost_y1[Q][i] =
          (cpi->common.Y1dequant[Q][1] * zbin_boost[i]) >> 7;

      cpi->Y2quant_fast[Q][i] = cpi->Y2quant_fast[Q][1];
      cpi->Y2quant[Q][i] = cpi->Y2quant[Q][1];
      cpi->Y2quant_shift[Q][i] = cpi->Y2quant_shift[Q][1];
      cpi->Y2zbin[Q][i] = cpi->Y2zbin[Q][1];
      cpi->Y2round[Q][i] = cpi->Y2round[Q][1];
      cpi->zrun_zbin_boost_y2[Q][i] =
          (cpi->common.Y2dequant[Q][1] * zbin_boost[i]) >> 7;

      cpi->UVquant_fast[Q][i] = cpi->UVquant_fast[Q][1];
      cpi->UVquant[Q][i] = cpi->UVquant[Q][1];
      cpi->UVquant_shift[Q][i] = cpi->UVquant_shift[Q][1];
      cpi->UVzbin[Q][i] = cpi->UVzbin[Q][1];
      cpi->UVround[Q][i] = cpi->UVround[Q][1];
      cpi->zrun_zbin_boost_uv[Q][i] =
          (cpi->common.UVdequant[Q][1] * zbin_boost[i]) >> 7;
    }
  }
}

#define ZBIN_EXTRA_Y                                                \
  ((cpi->common.Y1dequant[QIndex][1] *                              \
    (x->zbin_over_quant + x->zbin_mode_boost + x->act_zbin_adj)) >> \
   7)

#define ZBIN_EXTRA_UV                                               \
  ((cpi->common.UVdequant[QIndex][1] *                              \
    (x->zbin_over_quant + x->zbin_mode_boost + x->act_zbin_adj)) >> \
   7)

#define ZBIN_EXTRA_Y2                                                     \
  ((cpi->common.Y2dequant[QIndex][1] *                                    \
    ((x->zbin_over_quant / 2) + x->zbin_mode_boost + x->act_zbin_adj)) >> \
   7)

void vp8cx_mb_init_quantizer(VP8_COMP *cpi, MACROBLOCK *x, int ok_to_skip) {
  int i;
  int QIndex;
  MACROBLOCKD *xd = &x->e_mbd;
  int zbin_extra;

  /* Select the baseline MB Q index. */
  if (xd->segmentation_enabled) {
    /* Abs Value */
    if (xd->mb_segment_abs_delta == SEGMENT_ABSDATA) {
      QIndex = xd->segment_feature_data[MB_LVL_ALT_Q]
                                       [xd->mode_info_context->mbmi.segment_id];
      /* Delta Value */
    } else {
      QIndex = cpi->common.base_qindex +
               xd->segment_feature_data[MB_LVL_ALT_Q]
                                       [xd->mode_info_context->mbmi.segment_id];
      /* Clamp to valid range */
      QIndex = (QIndex >= 0) ? ((QIndex <= MAXQ) ? QIndex : MAXQ) : 0;
    }
  } else {
    QIndex = cpi->common.base_qindex;
  }

  /* This initialization should be called at least once. Use ok_to_skip to
   * decide if it is ok to skip.
   * Before encoding a frame, this function is always called with ok_to_skip
   * =0, which means no skiping of calculations. The "last" values are
   * initialized at that time.
   */
  if (!ok_to_skip || QIndex != x->q_index) {
    xd->dequant_y1_dc[0] = 1;
    xd->dequant_y1[0] = cpi->common.Y1dequant[QIndex][0];
    xd->dequant_y2[0] = cpi->common.Y2dequant[QIndex][0];
    xd->dequant_uv[0] = cpi->common.UVdequant[QIndex][0];

    for (i = 1; i < 16; ++i) {
      xd->dequant_y1_dc[i] = xd->dequant_y1[i] =
          cpi->common.Y1dequant[QIndex][1];
      xd->dequant_y2[i] = cpi->common.Y2dequant[QIndex][1];
      xd->dequant_uv[i] = cpi->common.UVdequant[QIndex][1];
    }
#if 1
    /*TODO:  Remove dequant from BLOCKD.  This is a temporary solution until
     * the quantizer code uses a passed in pointer to the dequant constants.
     * This will also require modifications to the x86 and neon assembly.
     * */
    for (i = 0; i < 16; ++i) x->e_mbd.block[i].dequant = xd->dequant_y1;
    for (i = 16; i < 24; ++i) x->e_mbd.block[i].dequant = xd->dequant_uv;
    x->e_mbd.block[24].dequant = xd->dequant_y2;
#endif

    /* Y */
    zbin_extra = ZBIN_EXTRA_Y;

    for (i = 0; i < 16; ++i) {
      x->block[i].quant = cpi->Y1quant[QIndex];
      x->block[i].quant_fast = cpi->Y1quant_fast[QIndex];
      x->block[i].quant_shift = cpi->Y1quant_shift[QIndex];
      x->block[i].zbin = cpi->Y1zbin[QIndex];
      x->block[i].round = cpi->Y1round[QIndex];
      x->block[i].zrun_zbin_boost = cpi->zrun_zbin_boost_y1[QIndex];
      x->block[i].zbin_extra = (short)zbin_extra;
    }

    /* UV */
    zbin_extra = ZBIN_EXTRA_UV;

    for (i = 16; i < 24; ++i) {
      x->block[i].quant = cpi->UVquant[QIndex];
      x->block[i].quant_fast = cpi->UVquant_fast[QIndex];
      x->block[i].quant_shift = cpi->UVquant_shift[QIndex];
      x->block[i].zbin = cpi->UVzbin[QIndex];
      x->block[i].round = cpi->UVround[QIndex];
      x->block[i].zrun_zbin_boost = cpi->zrun_zbin_boost_uv[QIndex];
      x->block[i].zbin_extra = (short)zbin_extra;
    }

    /* Y2 */
    zbin_extra = ZBIN_EXTRA_Y2;

    x->block[24].quant_fast = cpi->Y2quant_fast[QIndex];
    x->block[24].quant = cpi->Y2quant[QIndex];
    x->block[24].quant_shift = cpi->Y2quant_shift[QIndex];
    x->block[24].zbin = cpi->Y2zbin[QIndex];
    x->block[24].round = cpi->Y2round[QIndex];
    x->block[24].zrun_zbin_boost = cpi->zrun_zbin_boost_y2[QIndex];
    x->block[24].zbin_extra = (short)zbin_extra;

    /* save this macroblock QIndex for vp8_update_zbin_extra() */
    x->q_index = QIndex;

    x->last_zbin_over_quant = x->zbin_over_quant;
    x->last_zbin_mode_boost = x->zbin_mode_boost;
    x->last_act_zbin_adj = x->act_zbin_adj;

  } else if (x->last_zbin_over_quant != x->zbin_over_quant ||
             x->last_zbin_mode_boost != x->zbin_mode_boost ||
             x->last_act_zbin_adj != x->act_zbin_adj) {
    /* Y */
    zbin_extra = ZBIN_EXTRA_Y;

    for (i = 0; i < 16; ++i) x->block[i].zbin_extra = (short)zbin_extra;

    /* UV */
    zbin_extra = ZBIN_EXTRA_UV;

    for (i = 16; i < 24; ++i) x->block[i].zbin_extra = (short)zbin_extra;

    /* Y2 */
    zbin_extra = ZBIN_EXTRA_Y2;
    x->block[24].zbin_extra = (short)zbin_extra;

    x->last_zbin_over_quant = x->zbin_over_quant;
    x->last_zbin_mode_boost = x->zbin_mode_boost;
    x->last_act_zbin_adj = x->act_zbin_adj;
  }
}

void vp8_update_zbin_extra(VP8_COMP *cpi, MACROBLOCK *x) {
  int i;
  int QIndex = x->q_index;
  int zbin_extra;

  /* Y */
  zbin_extra = ZBIN_EXTRA_Y;

  for (i = 0; i < 16; ++i) x->block[i].zbin_extra = (short)zbin_extra;

  /* UV */
  zbin_extra = ZBIN_EXTRA_UV;

  for (i = 16; i < 24; ++i) x->block[i].zbin_extra = (short)zbin_extra;

  /* Y2 */
  zbin_extra = ZBIN_EXTRA_Y2;
  x->block[24].zbin_extra = (short)zbin_extra;
}
#undef ZBIN_EXTRA_Y
#undef ZBIN_EXTRA_UV
#undef ZBIN_EXTRA_Y2

void vp8cx_frame_init_quantizer(VP8_COMP *cpi) {
  /* Clear Zbin mode boost for default case */
  cpi->mb.zbin_mode_boost = 0;

  /* MB level quantizer setup */
  vp8cx_mb_init_quantizer(cpi, &cpi->mb, 0);
}

void vp8_set_quantizer(struct VP8_COMP *cpi, int Q) {
  VP8_COMMON *cm = &cpi->common;
  MACROBLOCKD *mbd = &cpi->mb.e_mbd;
  int update = 0;
  int new_delta_q;
  int new_uv_delta_q;
  cm->base_qindex = Q;

  /* if any of the delta_q values are changing update flag has to be set */
  /* currently only y2dc_delta_q may change */

  cm->y1dc_delta_q = 0;
  cm->y2ac_delta_q = 0;

  if (Q < 4) {
    new_delta_q = 4 - Q;
  } else {
    new_delta_q = 0;
  }

  update |= cm->y2dc_delta_q != new_delta_q;
  cm->y2dc_delta_q = new_delta_q;

  new_uv_delta_q = 0;
  // For screen content, lower the q value for UV channel. For now, select
  // conservative delta; same delta for dc and ac, and decrease it with lower
  // Q, and set to 0 below some threshold. May want to condition this in
  // future on the variance/energy in UV channel.
  if (cpi->oxcf.screen_content_mode && Q > 40) {
    new_uv_delta_q = -(int)(0.15 * Q);
    // Check range: magnitude of delta is 4 bits.
    if (new_uv_delta_q < -15) {
      new_uv_delta_q = -15;
    }
  }
  update |= cm->uvdc_delta_q != new_uv_delta_q;
  cm->uvdc_delta_q = new_uv_delta_q;
  cm->uvac_delta_q = new_uv_delta_q;

  /* Set Segment specific quatizers */
  mbd->segment_feature_data[MB_LVL_ALT_Q][0] =
      cpi->segment_feature_data[MB_LVL_ALT_Q][0];
  mbd->segment_feature_data[MB_LVL_ALT_Q][1] =
      cpi->segment_feature_data[MB_LVL_ALT_Q][1];
  mbd->segment_feature_data[MB_LVL_ALT_Q][2] =
      cpi->segment_feature_data[MB_LVL_ALT_Q][2];
  mbd->segment_feature_data[MB_LVL_ALT_Q][3] =
      cpi->segment_feature_data[MB_LVL_ALT_Q][3];

  /* quantizer has to be reinitialized for any delta_q changes */
  if (update) vp8cx_init_quantizer(cpi);
}
