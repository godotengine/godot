/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

/* MFQE: Multiframe Quality Enhancement
 * In rate limited situations keyframes may cause significant visual artifacts
 * commonly referred to as "popping." This file implements a postproccesing
 * algorithm which blends data from the preceeding frame when there is no
 * motion and the q from the previous frame is lower which indicates that it is
 * higher quality.
 */

#include "./vp8_rtcd.h"
#include "./vpx_dsp_rtcd.h"
#include "vp8/common/common.h"
#include "vp8/common/postproc.h"
#include "vpx_dsp/variance.h"
#include "vpx_mem/vpx_mem.h"
#include "vpx_scale/yv12config.h"

#include <limits.h>
#include <stdlib.h>

static void filter_by_weight(unsigned char *src, int src_stride,
                             unsigned char *dst, int dst_stride, int block_size,
                             int src_weight) {
  int dst_weight = (1 << MFQE_PRECISION) - src_weight;
  int rounding_bit = 1 << (MFQE_PRECISION - 1);
  int r, c;

  for (r = 0; r < block_size; ++r) {
    for (c = 0; c < block_size; ++c) {
      dst[c] = (src[c] * src_weight + dst[c] * dst_weight + rounding_bit) >>
               MFQE_PRECISION;
    }
    src += src_stride;
    dst += dst_stride;
  }
}

void vp8_filter_by_weight16x16_c(unsigned char *src, int src_stride,
                                 unsigned char *dst, int dst_stride,
                                 int src_weight) {
  filter_by_weight(src, src_stride, dst, dst_stride, 16, src_weight);
}

void vp8_filter_by_weight8x8_c(unsigned char *src, int src_stride,
                               unsigned char *dst, int dst_stride,
                               int src_weight) {
  filter_by_weight(src, src_stride, dst, dst_stride, 8, src_weight);
}

void vp8_filter_by_weight4x4_c(unsigned char *src, int src_stride,
                               unsigned char *dst, int dst_stride,
                               int src_weight) {
  filter_by_weight(src, src_stride, dst, dst_stride, 4, src_weight);
}

static void apply_ifactor(unsigned char *y_src, int y_src_stride,
                          unsigned char *y_dst, int y_dst_stride,
                          unsigned char *u_src, unsigned char *v_src,
                          int uv_src_stride, unsigned char *u_dst,
                          unsigned char *v_dst, int uv_dst_stride,
                          int block_size, int src_weight) {
  if (block_size == 16) {
    vp8_filter_by_weight16x16(y_src, y_src_stride, y_dst, y_dst_stride,
                              src_weight);
    vp8_filter_by_weight8x8(u_src, uv_src_stride, u_dst, uv_dst_stride,
                            src_weight);
    vp8_filter_by_weight8x8(v_src, uv_src_stride, v_dst, uv_dst_stride,
                            src_weight);
  } else {
    vp8_filter_by_weight8x8(y_src, y_src_stride, y_dst, y_dst_stride,
                            src_weight);
    vp8_filter_by_weight4x4(u_src, uv_src_stride, u_dst, uv_dst_stride,
                            src_weight);
    vp8_filter_by_weight4x4(v_src, uv_src_stride, v_dst, uv_dst_stride,
                            src_weight);
  }
}

static unsigned int int_sqrt(unsigned int x) {
  unsigned int y = x;
  unsigned int guess;
  int p = 1;
  while (y >>= 1) p++;
  p >>= 1;

  guess = 0;
  while (p >= 0) {
    guess |= (1 << p);
    if (x < guess * guess) guess -= (1 << p);
    p--;
  }
  /* choose between guess or guess+1 */
  return guess + (guess * guess + guess + 1 <= x);
}

#define USE_SSD
static void multiframe_quality_enhance_block(
    int blksize, /* Currently only values supported are 16, 8 */
    int qcurr, int qprev, unsigned char *y, unsigned char *u, unsigned char *v,
    int y_stride, int uv_stride, unsigned char *yd, unsigned char *ud,
    unsigned char *vd, int yd_stride, int uvd_stride) {
  static const unsigned char VP8_ZEROS[16] = { 0, 0, 0, 0, 0, 0, 0, 0,
                                               0, 0, 0, 0, 0, 0, 0, 0 };
  int uvblksize = blksize >> 1;
  int qdiff = qcurr - qprev;

  int i;
  unsigned char *up;
  unsigned char *udp;
  unsigned char *vp;
  unsigned char *vdp;

  unsigned int act, actd, sad, usad, vsad, sse, thr, thrsq, actrisk;

  if (blksize == 16) {
    actd = (vpx_variance16x16(yd, yd_stride, VP8_ZEROS, 0, &sse) + 128) >> 8;
    act = (vpx_variance16x16(y, y_stride, VP8_ZEROS, 0, &sse) + 128) >> 8;
#ifdef USE_SSD
    vpx_variance16x16(y, y_stride, yd, yd_stride, &sse);
    sad = (sse + 128) >> 8;
    vpx_variance8x8(u, uv_stride, ud, uvd_stride, &sse);
    usad = (sse + 32) >> 6;
    vpx_variance8x8(v, uv_stride, vd, uvd_stride, &sse);
    vsad = (sse + 32) >> 6;
#else
    sad = (vpx_sad16x16(y, y_stride, yd, yd_stride) + 128) >> 8;
    usad = (vpx_sad8x8(u, uv_stride, ud, uvd_stride) + 32) >> 6;
    vsad = (vpx_sad8x8(v, uv_stride, vd, uvd_stride) + 32) >> 6;
#endif
  } else {
    actd = (vpx_variance8x8(yd, yd_stride, VP8_ZEROS, 0, &sse) + 32) >> 6;
    act = (vpx_variance8x8(y, y_stride, VP8_ZEROS, 0, &sse) + 32) >> 6;
#ifdef USE_SSD
    vpx_variance8x8(y, y_stride, yd, yd_stride, &sse);
    sad = (sse + 32) >> 6;
    vpx_variance4x4(u, uv_stride, ud, uvd_stride, &sse);
    usad = (sse + 8) >> 4;
    vpx_variance4x4(v, uv_stride, vd, uvd_stride, &sse);
    vsad = (sse + 8) >> 4;
#else
    sad = (vpx_sad8x8(y, y_stride, yd, yd_stride) + 32) >> 6;
    usad = (vpx_sad4x4(u, uv_stride, ud, uvd_stride) + 8) >> 4;
    vsad = (vpx_sad4x4(v, uv_stride, vd, uvd_stride) + 8) >> 4;
#endif
  }

  actrisk = (actd > act * 5);

  /* thr = qdiff/16 + log2(act) + log4(qprev) */
  thr = (qdiff >> 4);
  while (actd >>= 1) thr++;
  while (qprev >>= 2) thr++;

#ifdef USE_SSD
  thrsq = thr * thr;
  if (sad < thrsq &&
      /* additional checks for color mismatch and excessive addition of
       * high-frequencies */
      4 * usad < thrsq && 4 * vsad < thrsq && !actrisk)
#else
  if (sad < thr &&
      /* additional checks for color mismatch and excessive addition of
       * high-frequencies */
      2 * usad < thr && 2 * vsad < thr && !actrisk)
#endif
  {
    int ifactor;
#ifdef USE_SSD
    /* TODO: optimize this later to not need sqr root */
    sad = int_sqrt(sad);
#endif
    ifactor = (sad << MFQE_PRECISION) / thr;
    ifactor >>= (qdiff >> 5);

    if (ifactor) {
      apply_ifactor(y, y_stride, yd, yd_stride, u, v, uv_stride, ud, vd,
                    uvd_stride, blksize, ifactor);
    }
  } else { /* else implicitly copy from previous frame */
    if (blksize == 16) {
      vp8_copy_mem16x16(y, y_stride, yd, yd_stride);
      vp8_copy_mem8x8(u, uv_stride, ud, uvd_stride);
      vp8_copy_mem8x8(v, uv_stride, vd, uvd_stride);
    } else {
      vp8_copy_mem8x8(y, y_stride, yd, yd_stride);
      for (up = u, udp = ud, i = 0; i < uvblksize;
           ++i, up += uv_stride, udp += uvd_stride) {
        memcpy(udp, up, uvblksize);
      }
      for (vp = v, vdp = vd, i = 0; i < uvblksize;
           ++i, vp += uv_stride, vdp += uvd_stride) {
        memcpy(vdp, vp, uvblksize);
      }
    }
  }
}

static int qualify_inter_mb(const MODE_INFO *mode_info_context, int *map) {
  if (mode_info_context->mbmi.mb_skip_coeff) {
    map[0] = map[1] = map[2] = map[3] = 1;
  } else if (mode_info_context->mbmi.mode == SPLITMV) {
    static int ndx[4][4] = {
      { 0, 1, 4, 5 }, { 2, 3, 6, 7 }, { 8, 9, 12, 13 }, { 10, 11, 14, 15 }
    };
    int i, j;
    vp8_zero_array(map, 4);
    for (i = 0; i < 4; ++i) {
      map[i] = 1;
      for (j = 0; j < 4 && map[j]; ++j) {
        map[i] &= (mode_info_context->bmi[ndx[i][j]].mv.as_mv.row <= 2 &&
                   mode_info_context->bmi[ndx[i][j]].mv.as_mv.col <= 2);
      }
    }
  } else {
    map[0] = map[1] = map[2] = map[3] =
        (mode_info_context->mbmi.mode > B_PRED &&
         abs(mode_info_context->mbmi.mv.as_mv.row) <= 2 &&
         abs(mode_info_context->mbmi.mv.as_mv.col) <= 2);
  }
  return (map[0] + map[1] + map[2] + map[3]);
}

void vp8_multiframe_quality_enhance(VP8_COMMON *cm) {
  YV12_BUFFER_CONFIG *show = cm->frame_to_show;
  YV12_BUFFER_CONFIG *dest = &cm->post_proc_buffer;

  FRAME_TYPE frame_type = cm->frame_type;
  /* Point at base of Mb MODE_INFO list has motion vectors etc */
  const MODE_INFO *mode_info_context = cm->mi;
  int mb_row;
  int mb_col;
  int totmap, map[4];
  int qcurr = cm->base_qindex;
  int qprev = cm->postproc_state.last_base_qindex;

  unsigned char *y_ptr, *u_ptr, *v_ptr;
  unsigned char *yd_ptr, *ud_ptr, *vd_ptr;

  /* Set up the buffer pointers */
  y_ptr = show->y_buffer;
  u_ptr = show->u_buffer;
  v_ptr = show->v_buffer;
  yd_ptr = dest->y_buffer;
  ud_ptr = dest->u_buffer;
  vd_ptr = dest->v_buffer;

  /* postprocess each macro block */
  for (mb_row = 0; mb_row < cm->mb_rows; ++mb_row) {
    for (mb_col = 0; mb_col < cm->mb_cols; ++mb_col) {
      /* if motion is high there will likely be no benefit */
      if (frame_type == INTER_FRAME) {
        totmap = qualify_inter_mb(mode_info_context, map);
      } else {
        totmap = (frame_type == KEY_FRAME ? 4 : 0);
      }
      if (totmap) {
        if (totmap < 4) {
          int i, j;
          for (i = 0; i < 2; ++i) {
            for (j = 0; j < 2; ++j) {
              if (map[i * 2 + j]) {
                multiframe_quality_enhance_block(
                    8, qcurr, qprev, y_ptr + 8 * (i * show->y_stride + j),
                    u_ptr + 4 * (i * show->uv_stride + j),
                    v_ptr + 4 * (i * show->uv_stride + j), show->y_stride,
                    show->uv_stride, yd_ptr + 8 * (i * dest->y_stride + j),
                    ud_ptr + 4 * (i * dest->uv_stride + j),
                    vd_ptr + 4 * (i * dest->uv_stride + j), dest->y_stride,
                    dest->uv_stride);
              } else {
                /* copy a 8x8 block */
                int k;
                unsigned char *up = u_ptr + 4 * (i * show->uv_stride + j);
                unsigned char *udp = ud_ptr + 4 * (i * dest->uv_stride + j);
                unsigned char *vp = v_ptr + 4 * (i * show->uv_stride + j);
                unsigned char *vdp = vd_ptr + 4 * (i * dest->uv_stride + j);
                vp8_copy_mem8x8(
                    y_ptr + 8 * (i * show->y_stride + j), show->y_stride,
                    yd_ptr + 8 * (i * dest->y_stride + j), dest->y_stride);
                for (k = 0; k < 4; ++k, up += show->uv_stride,
                    udp += dest->uv_stride, vp += show->uv_stride,
                    vdp += dest->uv_stride) {
                  memcpy(udp, up, 4);
                  memcpy(vdp, vp, 4);
                }
              }
            }
          }
        } else { /* totmap = 4 */
          multiframe_quality_enhance_block(
              16, qcurr, qprev, y_ptr, u_ptr, v_ptr, show->y_stride,
              show->uv_stride, yd_ptr, ud_ptr, vd_ptr, dest->y_stride,
              dest->uv_stride);
        }
      } else {
        vp8_copy_mem16x16(y_ptr, show->y_stride, yd_ptr, dest->y_stride);
        vp8_copy_mem8x8(u_ptr, show->uv_stride, ud_ptr, dest->uv_stride);
        vp8_copy_mem8x8(v_ptr, show->uv_stride, vd_ptr, dest->uv_stride);
      }
      y_ptr += 16;
      u_ptr += 8;
      v_ptr += 8;
      yd_ptr += 16;
      ud_ptr += 8;
      vd_ptr += 8;
      mode_info_context++; /* step to next MB */
    }

    y_ptr += show->y_stride * 16 - 16 * cm->mb_cols;
    u_ptr += show->uv_stride * 8 - 8 * cm->mb_cols;
    v_ptr += show->uv_stride * 8 - 8 * cm->mb_cols;
    yd_ptr += dest->y_stride * 16 - 16 * cm->mb_cols;
    ud_ptr += dest->uv_stride * 8 - 8 * cm->mb_cols;
    vd_ptr += dest->uv_stride * 8 - 8 * cm->mb_cols;

    mode_info_context++; /* Skip border mb */
  }
}
