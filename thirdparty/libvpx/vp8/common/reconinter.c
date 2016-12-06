/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */


#include <limits.h>
#include <string.h>

#include "vpx_config.h"
#include "vp8_rtcd.h"
#include "vpx/vpx_integer.h"
#include "blockd.h"
#include "reconinter.h"
#if CONFIG_RUNTIME_CPU_DETECT
#include "onyxc_int.h"
#endif

void vp8_copy_mem16x16_c(
    unsigned char *src,
    int src_stride,
    unsigned char *dst,
    int dst_stride)
{

    int r;

    for (r = 0; r < 16; r++)
    {
        memcpy(dst, src, 16);

        src += src_stride;
        dst += dst_stride;

    }

}

void vp8_copy_mem8x8_c(
    unsigned char *src,
    int src_stride,
    unsigned char *dst,
    int dst_stride)
{
    int r;

    for (r = 0; r < 8; r++)
    {
        memcpy(dst, src, 8);

        src += src_stride;
        dst += dst_stride;

    }

}

void vp8_copy_mem8x4_c(
    unsigned char *src,
    int src_stride,
    unsigned char *dst,
    int dst_stride)
{
    int r;

    for (r = 0; r < 4; r++)
    {
        memcpy(dst, src, 8);

        src += src_stride;
        dst += dst_stride;

    }

}


void vp8_build_inter_predictors_b(BLOCKD *d, int pitch, unsigned char *base_pre, int pre_stride, vp8_subpix_fn_t sppf)
{
    int r;
    unsigned char *pred_ptr = d->predictor;
    unsigned char *ptr;
    ptr = base_pre + d->offset + (d->bmi.mv.as_mv.row >> 3) * pre_stride + (d->bmi.mv.as_mv.col >> 3);

    if (d->bmi.mv.as_mv.row & 7 || d->bmi.mv.as_mv.col & 7)
    {
        sppf(ptr, pre_stride, d->bmi.mv.as_mv.col & 7, d->bmi.mv.as_mv.row & 7, pred_ptr, pitch);
    }
    else
    {
        for (r = 0; r < 4; r++)
        {
            pred_ptr[0]  = ptr[0];
            pred_ptr[1]  = ptr[1];
            pred_ptr[2]  = ptr[2];
            pred_ptr[3]  = ptr[3];
            pred_ptr     += pitch;
            ptr         += pre_stride;
        }
    }
}

static void build_inter_predictors4b(MACROBLOCKD *x, BLOCKD *d, unsigned char *dst, int dst_stride, unsigned char *base_pre, int pre_stride)
{
    unsigned char *ptr;
    ptr = base_pre + d->offset + (d->bmi.mv.as_mv.row >> 3) * pre_stride + (d->bmi.mv.as_mv.col >> 3);

    if (d->bmi.mv.as_mv.row & 7 || d->bmi.mv.as_mv.col & 7)
    {
        x->subpixel_predict8x8(ptr, pre_stride, d->bmi.mv.as_mv.col & 7, d->bmi.mv.as_mv.row & 7, dst, dst_stride);
    }
    else
    {
        vp8_copy_mem8x8(ptr, pre_stride, dst, dst_stride);
    }
}

static void build_inter_predictors2b(MACROBLOCKD *x, BLOCKD *d, unsigned char *dst, int dst_stride, unsigned char *base_pre, int pre_stride)
{
    unsigned char *ptr;
    ptr = base_pre + d->offset + (d->bmi.mv.as_mv.row >> 3) * pre_stride + (d->bmi.mv.as_mv.col >> 3);

    if (d->bmi.mv.as_mv.row & 7 || d->bmi.mv.as_mv.col & 7)
    {
        x->subpixel_predict8x4(ptr, pre_stride, d->bmi.mv.as_mv.col & 7, d->bmi.mv.as_mv.row & 7, dst, dst_stride);
    }
    else
    {
        vp8_copy_mem8x4(ptr, pre_stride, dst, dst_stride);
    }
}

static void build_inter_predictors_b(BLOCKD *d, unsigned char *dst, int dst_stride, unsigned char *base_pre, int pre_stride, vp8_subpix_fn_t sppf)
{
    int r;
    unsigned char *ptr;
    ptr = base_pre + d->offset + (d->bmi.mv.as_mv.row >> 3) * pre_stride + (d->bmi.mv.as_mv.col >> 3);

    if (d->bmi.mv.as_mv.row & 7 || d->bmi.mv.as_mv.col & 7)
    {
        sppf(ptr, pre_stride, d->bmi.mv.as_mv.col & 7, d->bmi.mv.as_mv.row & 7, dst, dst_stride);
    }
    else
    {
        for (r = 0; r < 4; r++)
        {
          dst[0]  = ptr[0];
          dst[1]  = ptr[1];
          dst[2]  = ptr[2];
          dst[3]  = ptr[3];
          dst     += dst_stride;
          ptr     += pre_stride;
        }
    }
}


/*encoder only*/
void vp8_build_inter16x16_predictors_mbuv(MACROBLOCKD *x)
{
    unsigned char *uptr, *vptr;
    unsigned char *upred_ptr = &x->predictor[256];
    unsigned char *vpred_ptr = &x->predictor[320];

    int mv_row = x->mode_info_context->mbmi.mv.as_mv.row;
    int mv_col = x->mode_info_context->mbmi.mv.as_mv.col;
    int offset;
    int pre_stride = x->pre.uv_stride;

    /* calc uv motion vectors */
    mv_row += 1 | (mv_row >> (sizeof(int) * CHAR_BIT - 1));
    mv_col += 1 | (mv_col >> (sizeof(int) * CHAR_BIT - 1));
    mv_row /= 2;
    mv_col /= 2;
    mv_row &= x->fullpixel_mask;
    mv_col &= x->fullpixel_mask;

    offset = (mv_row >> 3) * pre_stride + (mv_col >> 3);
    uptr = x->pre.u_buffer + offset;
    vptr = x->pre.v_buffer + offset;

    if ((mv_row | mv_col) & 7)
    {
        x->subpixel_predict8x8(uptr, pre_stride, mv_col & 7, mv_row & 7, upred_ptr, 8);
        x->subpixel_predict8x8(vptr, pre_stride, mv_col & 7, mv_row & 7, vpred_ptr, 8);
    }
    else
    {
        vp8_copy_mem8x8(uptr, pre_stride, upred_ptr, 8);
        vp8_copy_mem8x8(vptr, pre_stride, vpred_ptr, 8);
    }
}

/*encoder only*/
void vp8_build_inter4x4_predictors_mbuv(MACROBLOCKD *x)
{
    int i, j;
    int pre_stride = x->pre.uv_stride;
    unsigned char *base_pre;

    /* build uv mvs */
    for (i = 0; i < 2; i++)
    {
        for (j = 0; j < 2; j++)
        {
            int yoffset = i * 8 + j * 2;
            int uoffset = 16 + i * 2 + j;
            int voffset = 20 + i * 2 + j;

            int temp;

            temp = x->block[yoffset  ].bmi.mv.as_mv.row
                   + x->block[yoffset+1].bmi.mv.as_mv.row
                   + x->block[yoffset+4].bmi.mv.as_mv.row
                   + x->block[yoffset+5].bmi.mv.as_mv.row;

            temp += 4 + ((temp >> (sizeof(temp) * CHAR_BIT - 1)) * 8);

            x->block[uoffset].bmi.mv.as_mv.row = (temp / 8) & x->fullpixel_mask;

            temp = x->block[yoffset  ].bmi.mv.as_mv.col
                   + x->block[yoffset+1].bmi.mv.as_mv.col
                   + x->block[yoffset+4].bmi.mv.as_mv.col
                   + x->block[yoffset+5].bmi.mv.as_mv.col;

            temp += 4 + ((temp >> (sizeof(temp) * CHAR_BIT - 1)) * 8);

            x->block[uoffset].bmi.mv.as_mv.col = (temp / 8) & x->fullpixel_mask;

            x->block[voffset].bmi.mv.as_int = x->block[uoffset].bmi.mv.as_int;
        }
    }

    base_pre = x->pre.u_buffer;
    for (i = 16; i < 20; i += 2)
    {
        BLOCKD *d0 = &x->block[i];
        BLOCKD *d1 = &x->block[i+1];

        if (d0->bmi.mv.as_int == d1->bmi.mv.as_int)
            build_inter_predictors2b(x, d0, d0->predictor, 8, base_pre, pre_stride);
        else
        {
            vp8_build_inter_predictors_b(d0, 8, base_pre, pre_stride, x->subpixel_predict);
            vp8_build_inter_predictors_b(d1, 8, base_pre, pre_stride, x->subpixel_predict);
        }
    }

    base_pre = x->pre.v_buffer;
    for (i = 20; i < 24; i += 2)
    {
        BLOCKD *d0 = &x->block[i];
        BLOCKD *d1 = &x->block[i+1];

        if (d0->bmi.mv.as_int == d1->bmi.mv.as_int)
            build_inter_predictors2b(x, d0, d0->predictor, 8, base_pre, pre_stride);
        else
        {
            vp8_build_inter_predictors_b(d0, 8, base_pre, pre_stride, x->subpixel_predict);
            vp8_build_inter_predictors_b(d1, 8, base_pre, pre_stride, x->subpixel_predict);
        }
    }
}


/*encoder only*/
void vp8_build_inter16x16_predictors_mby(MACROBLOCKD *x,
                                         unsigned char *dst_y,
                                         int dst_ystride)
{
    unsigned char *ptr_base;
    unsigned char *ptr;
    int mv_row = x->mode_info_context->mbmi.mv.as_mv.row;
    int mv_col = x->mode_info_context->mbmi.mv.as_mv.col;
    int pre_stride = x->pre.y_stride;

    ptr_base = x->pre.y_buffer;
    ptr = ptr_base + (mv_row >> 3) * pre_stride + (mv_col >> 3);

    if ((mv_row | mv_col) & 7)
    {
        x->subpixel_predict16x16(ptr, pre_stride, mv_col & 7, mv_row & 7,
                                 dst_y, dst_ystride);
    }
    else
    {
        vp8_copy_mem16x16(ptr, pre_stride, dst_y,
            dst_ystride);
    }
}

static void clamp_mv_to_umv_border(MV *mv, const MACROBLOCKD *xd)
{
    /* If the MV points so far into the UMV border that no visible pixels
     * are used for reconstruction, the subpel part of the MV can be
     * discarded and the MV limited to 16 pixels with equivalent results.
     *
     * This limit kicks in at 19 pixels for the top and left edges, for
     * the 16 pixels plus 3 taps right of the central pixel when subpel
     * filtering. The bottom and right edges use 16 pixels plus 2 pixels
     * left of the central pixel when filtering.
     */
    if (mv->col < (xd->mb_to_left_edge - (19 << 3)))
        mv->col = xd->mb_to_left_edge - (16 << 3);
    else if (mv->col > xd->mb_to_right_edge + (18 << 3))
        mv->col = xd->mb_to_right_edge + (16 << 3);

    if (mv->row < (xd->mb_to_top_edge - (19 << 3)))
        mv->row = xd->mb_to_top_edge - (16 << 3);
    else if (mv->row > xd->mb_to_bottom_edge + (18 << 3))
        mv->row = xd->mb_to_bottom_edge + (16 << 3);
}

/* A version of the above function for chroma block MVs.*/
static void clamp_uvmv_to_umv_border(MV *mv, const MACROBLOCKD *xd)
{
    mv->col = (2*mv->col < (xd->mb_to_left_edge - (19 << 3))) ?
        (xd->mb_to_left_edge - (16 << 3)) >> 1 : mv->col;
    mv->col = (2*mv->col > xd->mb_to_right_edge + (18 << 3)) ?
        (xd->mb_to_right_edge + (16 << 3)) >> 1 : mv->col;

    mv->row = (2*mv->row < (xd->mb_to_top_edge - (19 << 3))) ?
        (xd->mb_to_top_edge - (16 << 3)) >> 1 : mv->row;
    mv->row = (2*mv->row > xd->mb_to_bottom_edge + (18 << 3)) ?
        (xd->mb_to_bottom_edge + (16 << 3)) >> 1 : mv->row;
}

void vp8_build_inter16x16_predictors_mb(MACROBLOCKD *x,
                                        unsigned char *dst_y,
                                        unsigned char *dst_u,
                                        unsigned char *dst_v,
                                        int dst_ystride,
                                        int dst_uvstride)
{
    int offset;
    unsigned char *ptr;
    unsigned char *uptr, *vptr;

    int_mv _16x16mv;

    unsigned char *ptr_base = x->pre.y_buffer;
    int pre_stride = x->pre.y_stride;

    _16x16mv.as_int = x->mode_info_context->mbmi.mv.as_int;

    if (x->mode_info_context->mbmi.need_to_clamp_mvs)
    {
        clamp_mv_to_umv_border(&_16x16mv.as_mv, x);
    }

    ptr = ptr_base + ( _16x16mv.as_mv.row >> 3) * pre_stride + (_16x16mv.as_mv.col >> 3);

    if ( _16x16mv.as_int & 0x00070007)
    {
        x->subpixel_predict16x16(ptr, pre_stride, _16x16mv.as_mv.col & 7,  _16x16mv.as_mv.row & 7, dst_y, dst_ystride);
    }
    else
    {
        vp8_copy_mem16x16(ptr, pre_stride, dst_y, dst_ystride);
    }

    /* calc uv motion vectors */
    _16x16mv.as_mv.row += 1 | (_16x16mv.as_mv.row >> (sizeof(int) * CHAR_BIT - 1));
    _16x16mv.as_mv.col += 1 | (_16x16mv.as_mv.col >> (sizeof(int) * CHAR_BIT - 1));
    _16x16mv.as_mv.row /= 2;
    _16x16mv.as_mv.col /= 2;
    _16x16mv.as_mv.row &= x->fullpixel_mask;
    _16x16mv.as_mv.col &= x->fullpixel_mask;

    pre_stride >>= 1;
    offset = ( _16x16mv.as_mv.row >> 3) * pre_stride + (_16x16mv.as_mv.col >> 3);
    uptr = x->pre.u_buffer + offset;
    vptr = x->pre.v_buffer + offset;

    if ( _16x16mv.as_int & 0x00070007)
    {
        x->subpixel_predict8x8(uptr, pre_stride, _16x16mv.as_mv.col & 7,  _16x16mv.as_mv.row & 7, dst_u, dst_uvstride);
        x->subpixel_predict8x8(vptr, pre_stride, _16x16mv.as_mv.col & 7,  _16x16mv.as_mv.row & 7, dst_v, dst_uvstride);
    }
    else
    {
        vp8_copy_mem8x8(uptr, pre_stride, dst_u, dst_uvstride);
        vp8_copy_mem8x8(vptr, pre_stride, dst_v, dst_uvstride);
    }
}

static void build_inter4x4_predictors_mb(MACROBLOCKD *x)
{
    int i;
    unsigned char *base_dst = x->dst.y_buffer;
    unsigned char *base_pre = x->pre.y_buffer;

    if (x->mode_info_context->mbmi.partitioning < 3)
    {
        BLOCKD *b;
        int dst_stride = x->dst.y_stride;

        x->block[ 0].bmi = x->mode_info_context->bmi[ 0];
        x->block[ 2].bmi = x->mode_info_context->bmi[ 2];
        x->block[ 8].bmi = x->mode_info_context->bmi[ 8];
        x->block[10].bmi = x->mode_info_context->bmi[10];
        if (x->mode_info_context->mbmi.need_to_clamp_mvs)
        {
            clamp_mv_to_umv_border(&x->block[ 0].bmi.mv.as_mv, x);
            clamp_mv_to_umv_border(&x->block[ 2].bmi.mv.as_mv, x);
            clamp_mv_to_umv_border(&x->block[ 8].bmi.mv.as_mv, x);
            clamp_mv_to_umv_border(&x->block[10].bmi.mv.as_mv, x);
        }

        b = &x->block[ 0];
        build_inter_predictors4b(x, b, base_dst + b->offset, dst_stride, base_pre, dst_stride);
        b = &x->block[ 2];
        build_inter_predictors4b(x, b, base_dst + b->offset, dst_stride, base_pre, dst_stride);
        b = &x->block[ 8];
        build_inter_predictors4b(x, b, base_dst + b->offset, dst_stride, base_pre, dst_stride);
        b = &x->block[10];
        build_inter_predictors4b(x, b, base_dst + b->offset, dst_stride, base_pre, dst_stride);
    }
    else
    {
        for (i = 0; i < 16; i += 2)
        {
            BLOCKD *d0 = &x->block[i];
            BLOCKD *d1 = &x->block[i+1];
            int dst_stride = x->dst.y_stride;

            x->block[i+0].bmi = x->mode_info_context->bmi[i+0];
            x->block[i+1].bmi = x->mode_info_context->bmi[i+1];
            if (x->mode_info_context->mbmi.need_to_clamp_mvs)
            {
                clamp_mv_to_umv_border(&x->block[i+0].bmi.mv.as_mv, x);
                clamp_mv_to_umv_border(&x->block[i+1].bmi.mv.as_mv, x);
            }

            if (d0->bmi.mv.as_int == d1->bmi.mv.as_int)
                build_inter_predictors2b(x, d0, base_dst + d0->offset, dst_stride, base_pre, dst_stride);
            else
            {
                build_inter_predictors_b(d0, base_dst + d0->offset, dst_stride, base_pre, dst_stride, x->subpixel_predict);
                build_inter_predictors_b(d1, base_dst + d1->offset, dst_stride, base_pre, dst_stride, x->subpixel_predict);
            }

        }

    }
    base_dst = x->dst.u_buffer;
    base_pre = x->pre.u_buffer;
    for (i = 16; i < 20; i += 2)
    {
        BLOCKD *d0 = &x->block[i];
        BLOCKD *d1 = &x->block[i+1];
        int dst_stride = x->dst.uv_stride;

        /* Note: uv mvs already clamped in build_4x4uvmvs() */

        if (d0->bmi.mv.as_int == d1->bmi.mv.as_int)
            build_inter_predictors2b(x, d0, base_dst + d0->offset, dst_stride, base_pre, dst_stride);
        else
        {
            build_inter_predictors_b(d0, base_dst + d0->offset, dst_stride, base_pre, dst_stride, x->subpixel_predict);
            build_inter_predictors_b(d1, base_dst + d1->offset, dst_stride, base_pre, dst_stride, x->subpixel_predict);
        }
    }

    base_dst = x->dst.v_buffer;
    base_pre = x->pre.v_buffer;
    for (i = 20; i < 24; i += 2)
    {
        BLOCKD *d0 = &x->block[i];
        BLOCKD *d1 = &x->block[i+1];
        int dst_stride = x->dst.uv_stride;

        /* Note: uv mvs already clamped in build_4x4uvmvs() */

        if (d0->bmi.mv.as_int == d1->bmi.mv.as_int)
            build_inter_predictors2b(x, d0, base_dst + d0->offset, dst_stride, base_pre, dst_stride);
        else
        {
            build_inter_predictors_b(d0, base_dst + d0->offset, dst_stride, base_pre, dst_stride, x->subpixel_predict);
            build_inter_predictors_b(d1, base_dst + d1->offset, dst_stride, base_pre, dst_stride, x->subpixel_predict);
        }
    }
}

static
void build_4x4uvmvs(MACROBLOCKD *x)
{
    int i, j;

    for (i = 0; i < 2; i++)
    {
        for (j = 0; j < 2; j++)
        {
            int yoffset = i * 8 + j * 2;
            int uoffset = 16 + i * 2 + j;
            int voffset = 20 + i * 2 + j;

            int temp;

            temp = x->mode_info_context->bmi[yoffset + 0].mv.as_mv.row
                 + x->mode_info_context->bmi[yoffset + 1].mv.as_mv.row
                 + x->mode_info_context->bmi[yoffset + 4].mv.as_mv.row
                 + x->mode_info_context->bmi[yoffset + 5].mv.as_mv.row;

            temp += 4 + ((temp >> (sizeof(temp) * CHAR_BIT - 1)) * 8);

            x->block[uoffset].bmi.mv.as_mv.row = (temp / 8) & x->fullpixel_mask;

            temp = x->mode_info_context->bmi[yoffset + 0].mv.as_mv.col
                 + x->mode_info_context->bmi[yoffset + 1].mv.as_mv.col
                 + x->mode_info_context->bmi[yoffset + 4].mv.as_mv.col
                 + x->mode_info_context->bmi[yoffset + 5].mv.as_mv.col;

            temp += 4 + ((temp >> (sizeof(temp) * CHAR_BIT - 1)) * 8);

            x->block[uoffset].bmi.mv.as_mv.col = (temp / 8) & x->fullpixel_mask;

            if (x->mode_info_context->mbmi.need_to_clamp_mvs)
                clamp_uvmv_to_umv_border(&x->block[uoffset].bmi.mv.as_mv, x);

            x->block[voffset].bmi.mv.as_int = x->block[uoffset].bmi.mv.as_int;
        }
    }
}

void vp8_build_inter_predictors_mb(MACROBLOCKD *xd)
{
    if (xd->mode_info_context->mbmi.mode != SPLITMV)
    {
        vp8_build_inter16x16_predictors_mb(xd, xd->dst.y_buffer,
                                           xd->dst.u_buffer, xd->dst.v_buffer,
                                           xd->dst.y_stride, xd->dst.uv_stride);
    }
    else
    {
        build_4x4uvmvs(xd);
        build_inter4x4_predictors_mb(xd);
    }
}
