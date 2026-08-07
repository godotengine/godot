/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP8_COMMON_LOOPFILTER_H_
#define VPX_VP8_COMMON_LOOPFILTER_H_

#include "vpx_ports/mem.h"
#include "vpx_config.h"
#include "vp8_rtcd.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_LOOP_FILTER 63
/* fraction of total macroblock rows to be used in fast filter level picking */
/* has to be > 2 */
#define PARTIAL_FRAME_FRACTION 8

typedef enum { NORMAL_LOOPFILTER = 0, SIMPLE_LOOPFILTER = 1 } LOOPFILTERTYPE;

#if VPX_ARCH_ARM
#define SIMD_WIDTH 1
#else
#define SIMD_WIDTH 16
#endif

/* Need to align this structure so when it is declared and
 * passed it can be loaded into vector registers.
 */
typedef struct {
  DECLARE_ALIGNED(SIMD_WIDTH, unsigned char,
                  mblim[MAX_LOOP_FILTER + 1][SIMD_WIDTH]);
  DECLARE_ALIGNED(SIMD_WIDTH, unsigned char,
                  blim[MAX_LOOP_FILTER + 1][SIMD_WIDTH]);
  DECLARE_ALIGNED(SIMD_WIDTH, unsigned char,
                  lim[MAX_LOOP_FILTER + 1][SIMD_WIDTH]);
  DECLARE_ALIGNED(SIMD_WIDTH, unsigned char, hev_thr[4][SIMD_WIDTH]);
  unsigned char lvl[4][4][4];
  unsigned char hev_thr_lut[2][MAX_LOOP_FILTER + 1];
  unsigned char mode_lf_lut[10];
} loop_filter_info_n;

typedef struct loop_filter_info {
  const unsigned char *mblim;
  const unsigned char *blim;
  const unsigned char *lim;
  const unsigned char *hev_thr;
} loop_filter_info;

typedef void loop_filter_uvfunction(unsigned char *u, /* source pointer */
                                    int p,            /* pitch */
                                    const unsigned char *blimit,
                                    const unsigned char *limit,
                                    const unsigned char *thresh,
                                    unsigned char *v);

/* assorted loopfilter functions which get used elsewhere */
struct VP8Common;
struct macroblockd;
struct modeinfo;

void vp8_loop_filter_init(struct VP8Common *cm);

void vp8_loop_filter_frame_init(struct VP8Common *cm, struct macroblockd *mbd,
                                int default_filt_lvl);

void vp8_loop_filter_frame(struct VP8Common *cm, struct macroblockd *mbd,
                           int frame_type);

void vp8_loop_filter_partial_frame(struct VP8Common *cm,
                                   struct macroblockd *mbd,
                                   int default_filt_lvl);

void vp8_loop_filter_frame_yonly(struct VP8Common *cm, struct macroblockd *mbd,
                                 int default_filt_lvl);

void vp8_loop_filter_update_sharpness(loop_filter_info_n *lfi,
                                      int sharpness_lvl);

void vp8_loop_filter_row_normal(struct VP8Common *cm,
                                struct modeinfo *mode_info_context, int mb_row,
                                int post_ystride, int post_uvstride,
                                unsigned char *y_ptr, unsigned char *u_ptr,
                                unsigned char *v_ptr);

void vp8_loop_filter_row_simple(struct VP8Common *cm,
                                struct modeinfo *mode_info_context, int mb_row,
                                int post_ystride, unsigned char *y_ptr);
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_COMMON_LOOPFILTER_H_
