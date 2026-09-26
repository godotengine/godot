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
#include "./vp8_rtcd.h"
#include "vp8/common/arm/loopfilter_arm.h"
#include "vp8/common/loopfilter.h"
#include "vp8/common/onyxc_int.h"

/* NEON loopfilter functions */
/* Horizontal MB filtering */
void vp8_loop_filter_mbh_neon(unsigned char *y_ptr, unsigned char *u_ptr,
                              unsigned char *v_ptr, int y_stride, int uv_stride,
                              loop_filter_info *lfi) {
  unsigned char mblim = *lfi->mblim;
  unsigned char lim = *lfi->lim;
  unsigned char hev_thr = *lfi->hev_thr;
  vp8_mbloop_filter_horizontal_edge_y_neon(y_ptr, y_stride, mblim, lim,
                                           hev_thr);

  if (u_ptr)
    vp8_mbloop_filter_horizontal_edge_uv_neon(u_ptr, uv_stride, mblim, lim,
                                              hev_thr, v_ptr);
}

/* Vertical MB Filtering */
void vp8_loop_filter_mbv_neon(unsigned char *y_ptr, unsigned char *u_ptr,
                              unsigned char *v_ptr, int y_stride, int uv_stride,
                              loop_filter_info *lfi) {
  unsigned char mblim = *lfi->mblim;
  unsigned char lim = *lfi->lim;
  unsigned char hev_thr = *lfi->hev_thr;

  vp8_mbloop_filter_vertical_edge_y_neon(y_ptr, y_stride, mblim, lim, hev_thr);

  if (u_ptr)
    vp8_mbloop_filter_vertical_edge_uv_neon(u_ptr, uv_stride, mblim, lim,
                                            hev_thr, v_ptr);
}

/* Horizontal B Filtering */
void vp8_loop_filter_bh_neon(unsigned char *y_ptr, unsigned char *u_ptr,
                             unsigned char *v_ptr, int y_stride, int uv_stride,
                             loop_filter_info *lfi) {
  unsigned char blim = *lfi->blim;
  unsigned char lim = *lfi->lim;
  unsigned char hev_thr = *lfi->hev_thr;

  vp8_loop_filter_horizontal_edge_y_neon(y_ptr + 4 * y_stride, y_stride, blim,
                                         lim, hev_thr);
  vp8_loop_filter_horizontal_edge_y_neon(y_ptr + 8 * y_stride, y_stride, blim,
                                         lim, hev_thr);
  vp8_loop_filter_horizontal_edge_y_neon(y_ptr + 12 * y_stride, y_stride, blim,
                                         lim, hev_thr);

  if (u_ptr)
    vp8_loop_filter_horizontal_edge_uv_neon(u_ptr + 4 * uv_stride, uv_stride,
                                            blim, lim, hev_thr,
                                            v_ptr + 4 * uv_stride);
}

/* Vertical B Filtering */
void vp8_loop_filter_bv_neon(unsigned char *y_ptr, unsigned char *u_ptr,
                             unsigned char *v_ptr, int y_stride, int uv_stride,
                             loop_filter_info *lfi) {
  unsigned char blim = *lfi->blim;
  unsigned char lim = *lfi->lim;
  unsigned char hev_thr = *lfi->hev_thr;

  vp8_loop_filter_vertical_edge_y_neon(y_ptr + 4, y_stride, blim, lim, hev_thr);
  vp8_loop_filter_vertical_edge_y_neon(y_ptr + 8, y_stride, blim, lim, hev_thr);
  vp8_loop_filter_vertical_edge_y_neon(y_ptr + 12, y_stride, blim, lim,
                                       hev_thr);

  if (u_ptr)
    vp8_loop_filter_vertical_edge_uv_neon(u_ptr + 4, uv_stride, blim, lim,
                                          hev_thr, v_ptr + 4);
}
