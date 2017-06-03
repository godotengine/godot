/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */


#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "./vp8_rtcd.h"
#include "vpx_mem/vpx_mem.h"
#include "vpx_ports/vpx_once.h"
#include "blockd.h"
#include "vp8/common/reconintra.h"
#include "vp8/common/reconintra4x4.h"

enum {
    SIZE_16,
    SIZE_8,
    NUM_SIZES,
};

typedef void (*intra_pred_fn)(uint8_t *dst, ptrdiff_t stride,
                              const uint8_t *above, const uint8_t *left);

static intra_pred_fn pred[4][NUM_SIZES];
static intra_pred_fn dc_pred[2][2][NUM_SIZES];

static void vp8_init_intra_predictors_internal(void)
{
#define INIT_SIZE(sz) \
    pred[V_PRED][SIZE_##sz] = vpx_v_predictor_##sz##x##sz; \
    pred[H_PRED][SIZE_##sz] = vpx_h_predictor_##sz##x##sz; \
    pred[TM_PRED][SIZE_##sz] = vpx_tm_predictor_##sz##x##sz; \
 \
    dc_pred[0][0][SIZE_##sz] = vpx_dc_128_predictor_##sz##x##sz; \
    dc_pred[0][1][SIZE_##sz] = vpx_dc_top_predictor_##sz##x##sz; \
    dc_pred[1][0][SIZE_##sz] = vpx_dc_left_predictor_##sz##x##sz; \
    dc_pred[1][1][SIZE_##sz] = vpx_dc_predictor_##sz##x##sz

    INIT_SIZE(16);
    INIT_SIZE(8);
    vp8_init_intra4x4_predictors_internal();
}

void vp8_build_intra_predictors_mby_s(MACROBLOCKD *x,
                                      unsigned char * yabove_row,
                                      unsigned char * yleft,
                                      int left_stride,
                                      unsigned char * ypred_ptr,
                                      int y_stride)
{
    MB_PREDICTION_MODE mode = x->mode_info_context->mbmi.mode;
    DECLARE_ALIGNED(16, uint8_t, yleft_col[16]);
    int i;
    intra_pred_fn fn;

    for (i = 0; i < 16; i++)
    {
        yleft_col[i] = yleft[i* left_stride];
    }

    if (mode == DC_PRED)
    {
        fn = dc_pred[x->left_available][x->up_available][SIZE_16];
    }
    else
    {
        fn = pred[mode][SIZE_16];
    }

    fn(ypred_ptr, y_stride, yabove_row, yleft_col);
}

void vp8_build_intra_predictors_mbuv_s(MACROBLOCKD *x,
                                       unsigned char * uabove_row,
                                       unsigned char * vabove_row,
                                       unsigned char * uleft,
                                       unsigned char * vleft,
                                       int left_stride,
                                       unsigned char * upred_ptr,
                                       unsigned char * vpred_ptr,
                                       int pred_stride)
{
    MB_PREDICTION_MODE uvmode = x->mode_info_context->mbmi.uv_mode;
    unsigned char uleft_col[8];
    unsigned char vleft_col[8];
    int i;
    intra_pred_fn fn;

    for (i = 0; i < 8; i++)
    {
        uleft_col[i] = uleft[i * left_stride];
        vleft_col[i] = vleft[i * left_stride];
    }

    if (uvmode == DC_PRED)
    {
        fn = dc_pred[x->left_available][x->up_available][SIZE_8];
    }
    else
    {
        fn = pred[uvmode][SIZE_8];
    }

    fn(upred_ptr, pred_stride, uabove_row, uleft_col);
    fn(vpred_ptr, pred_stride, vabove_row, vleft_col);
}

void vp8_init_intra_predictors(void)
{
    once(vp8_init_intra_predictors_internal);
}
