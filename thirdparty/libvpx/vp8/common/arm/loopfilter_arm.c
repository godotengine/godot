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
#include "vp8/common/loopfilter.h"
#include "vp8/common/onyxc_int.h"

#define prototype_loopfilter(sym) \
    void sym(unsigned char *src, int pitch, const unsigned char *blimit,\
             const unsigned char *limit, const unsigned char *thresh, int count)

#if HAVE_MEDIA
extern prototype_loopfilter(vp8_loop_filter_horizontal_edge_armv6);
extern prototype_loopfilter(vp8_loop_filter_vertical_edge_armv6);
extern prototype_loopfilter(vp8_mbloop_filter_horizontal_edge_armv6);
extern prototype_loopfilter(vp8_mbloop_filter_vertical_edge_armv6);
#endif

#if HAVE_NEON
typedef void loopfilter_y_neon(unsigned char *src, int pitch,
        unsigned char blimit, unsigned char limit, unsigned char thresh);
typedef void loopfilter_uv_neon(unsigned char *u, int pitch,
        unsigned char blimit, unsigned char limit, unsigned char thresh,
        unsigned char *v);

extern loopfilter_y_neon vp8_loop_filter_horizontal_edge_y_neon;
extern loopfilter_y_neon vp8_loop_filter_vertical_edge_y_neon;
extern loopfilter_uv_neon vp8_loop_filter_horizontal_edge_uv_neon;
extern loopfilter_uv_neon vp8_loop_filter_vertical_edge_uv_neon;

extern loopfilter_y_neon vp8_mbloop_filter_horizontal_edge_y_neon;
extern loopfilter_y_neon vp8_mbloop_filter_vertical_edge_y_neon;
extern loopfilter_uv_neon vp8_mbloop_filter_horizontal_edge_uv_neon;
extern loopfilter_uv_neon vp8_mbloop_filter_vertical_edge_uv_neon;
#endif

#if HAVE_MEDIA
/* ARMV6/MEDIA loopfilter functions*/
/* Horizontal MB filtering */
void vp8_loop_filter_mbh_armv6(unsigned char *y_ptr, unsigned char *u_ptr, unsigned char *v_ptr,
                               int y_stride, int uv_stride, loop_filter_info *lfi)
{
    vp8_mbloop_filter_horizontal_edge_armv6(y_ptr, y_stride, lfi->mblim, lfi->lim, lfi->hev_thr, 2);

    if (u_ptr)
        vp8_mbloop_filter_horizontal_edge_armv6(u_ptr, uv_stride, lfi->mblim, lfi->lim, lfi->hev_thr, 1);

    if (v_ptr)
        vp8_mbloop_filter_horizontal_edge_armv6(v_ptr, uv_stride, lfi->mblim, lfi->lim, lfi->hev_thr, 1);
}

/* Vertical MB Filtering */
void vp8_loop_filter_mbv_armv6(unsigned char *y_ptr, unsigned char *u_ptr, unsigned char *v_ptr,
                               int y_stride, int uv_stride, loop_filter_info *lfi)
{
    vp8_mbloop_filter_vertical_edge_armv6(y_ptr, y_stride, lfi->mblim, lfi->lim, lfi->hev_thr, 2);

    if (u_ptr)
        vp8_mbloop_filter_vertical_edge_armv6(u_ptr, uv_stride, lfi->mblim, lfi->lim, lfi->hev_thr, 1);

    if (v_ptr)
        vp8_mbloop_filter_vertical_edge_armv6(v_ptr, uv_stride, lfi->mblim, lfi->lim, lfi->hev_thr, 1);
}

/* Horizontal B Filtering */
void vp8_loop_filter_bh_armv6(unsigned char *y_ptr, unsigned char *u_ptr, unsigned char *v_ptr,
                              int y_stride, int uv_stride, loop_filter_info *lfi)
{
    vp8_loop_filter_horizontal_edge_armv6(y_ptr + 4 * y_stride, y_stride, lfi->blim, lfi->lim, lfi->hev_thr, 2);
    vp8_loop_filter_horizontal_edge_armv6(y_ptr + 8 * y_stride, y_stride, lfi->blim, lfi->lim, lfi->hev_thr, 2);
    vp8_loop_filter_horizontal_edge_armv6(y_ptr + 12 * y_stride, y_stride, lfi->blim, lfi->lim, lfi->hev_thr, 2);

    if (u_ptr)
        vp8_loop_filter_horizontal_edge_armv6(u_ptr + 4 * uv_stride, uv_stride, lfi->blim, lfi->lim, lfi->hev_thr, 1);

    if (v_ptr)
        vp8_loop_filter_horizontal_edge_armv6(v_ptr + 4 * uv_stride, uv_stride, lfi->blim, lfi->lim, lfi->hev_thr, 1);
}

void vp8_loop_filter_bhs_armv6(unsigned char *y_ptr, int y_stride,
                               const unsigned char *blimit)
{
    vp8_loop_filter_simple_horizontal_edge_armv6(y_ptr + 4 * y_stride, y_stride, blimit);
    vp8_loop_filter_simple_horizontal_edge_armv6(y_ptr + 8 * y_stride, y_stride, blimit);
    vp8_loop_filter_simple_horizontal_edge_armv6(y_ptr + 12 * y_stride, y_stride, blimit);
}

/* Vertical B Filtering */
void vp8_loop_filter_bv_armv6(unsigned char *y_ptr, unsigned char *u_ptr, unsigned char *v_ptr,
                              int y_stride, int uv_stride, loop_filter_info *lfi)
{
    vp8_loop_filter_vertical_edge_armv6(y_ptr + 4, y_stride, lfi->blim, lfi->lim, lfi->hev_thr, 2);
    vp8_loop_filter_vertical_edge_armv6(y_ptr + 8, y_stride, lfi->blim, lfi->lim, lfi->hev_thr, 2);
    vp8_loop_filter_vertical_edge_armv6(y_ptr + 12, y_stride, lfi->blim, lfi->lim, lfi->hev_thr, 2);

    if (u_ptr)
        vp8_loop_filter_vertical_edge_armv6(u_ptr + 4, uv_stride, lfi->blim, lfi->lim, lfi->hev_thr, 1);

    if (v_ptr)
        vp8_loop_filter_vertical_edge_armv6(v_ptr + 4, uv_stride, lfi->blim, lfi->lim, lfi->hev_thr, 1);
}

void vp8_loop_filter_bvs_armv6(unsigned char *y_ptr, int y_stride,
                               const unsigned char *blimit)
{
    vp8_loop_filter_simple_vertical_edge_armv6(y_ptr + 4, y_stride, blimit);
    vp8_loop_filter_simple_vertical_edge_armv6(y_ptr + 8, y_stride, blimit);
    vp8_loop_filter_simple_vertical_edge_armv6(y_ptr + 12, y_stride, blimit);
}
#endif

#if HAVE_NEON
/* NEON loopfilter functions */
/* Horizontal MB filtering */
void vp8_loop_filter_mbh_neon(unsigned char *y_ptr, unsigned char *u_ptr, unsigned char *v_ptr,
                              int y_stride, int uv_stride, loop_filter_info *lfi)
{
    unsigned char mblim = *lfi->mblim;
    unsigned char lim = *lfi->lim;
    unsigned char hev_thr = *lfi->hev_thr;
    vp8_mbloop_filter_horizontal_edge_y_neon(y_ptr, y_stride, mblim, lim, hev_thr);

    if (u_ptr)
        vp8_mbloop_filter_horizontal_edge_uv_neon(u_ptr, uv_stride, mblim, lim, hev_thr, v_ptr);
}

/* Vertical MB Filtering */
void vp8_loop_filter_mbv_neon(unsigned char *y_ptr, unsigned char *u_ptr, unsigned char *v_ptr,
                              int y_stride, int uv_stride, loop_filter_info *lfi)
{
    unsigned char mblim = *lfi->mblim;
    unsigned char lim = *lfi->lim;
    unsigned char hev_thr = *lfi->hev_thr;

    vp8_mbloop_filter_vertical_edge_y_neon(y_ptr, y_stride, mblim, lim, hev_thr);

    if (u_ptr)
        vp8_mbloop_filter_vertical_edge_uv_neon(u_ptr, uv_stride, mblim, lim, hev_thr, v_ptr);
}

/* Horizontal B Filtering */
void vp8_loop_filter_bh_neon(unsigned char *y_ptr, unsigned char *u_ptr, unsigned char *v_ptr,
                             int y_stride, int uv_stride, loop_filter_info *lfi)
{
    unsigned char blim = *lfi->blim;
    unsigned char lim = *lfi->lim;
    unsigned char hev_thr = *lfi->hev_thr;

    vp8_loop_filter_horizontal_edge_y_neon(y_ptr + 4 * y_stride, y_stride, blim, lim, hev_thr);
    vp8_loop_filter_horizontal_edge_y_neon(y_ptr + 8 * y_stride, y_stride, blim, lim, hev_thr);
    vp8_loop_filter_horizontal_edge_y_neon(y_ptr + 12 * y_stride, y_stride, blim, lim, hev_thr);

    if (u_ptr)
        vp8_loop_filter_horizontal_edge_uv_neon(u_ptr + 4 * uv_stride, uv_stride, blim, lim, hev_thr, v_ptr + 4 * uv_stride);
}

/* Vertical B Filtering */
void vp8_loop_filter_bv_neon(unsigned char *y_ptr, unsigned char *u_ptr, unsigned char *v_ptr,
                             int y_stride, int uv_stride, loop_filter_info *lfi)
{
    unsigned char blim = *lfi->blim;
    unsigned char lim = *lfi->lim;
    unsigned char hev_thr = *lfi->hev_thr;

    vp8_loop_filter_vertical_edge_y_neon(y_ptr + 4, y_stride, blim, lim, hev_thr);
    vp8_loop_filter_vertical_edge_y_neon(y_ptr + 8, y_stride, blim, lim, hev_thr);
    vp8_loop_filter_vertical_edge_y_neon(y_ptr + 12, y_stride, blim, lim, hev_thr);

    if (u_ptr)
        vp8_loop_filter_vertical_edge_uv_neon(u_ptr + 4, uv_stride, blim, lim, hev_thr, v_ptr + 4);
}
#endif
