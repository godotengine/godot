/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP8_ENCODER_RDOPT_H_
#define VPX_VP8_ENCODER_RDOPT_H_

#include "./vpx_config.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RDCOST(RM, DM, R, D) (((128 + (R) * (RM)) >> 8) + (DM) * (D))

void vp8cx_initialize_me_consts(VP8_COMP *cpi, int QIndex);
void vp8_auto_select_speed(VP8_COMP *cpi);

static INLINE void insertsortmv(int arr[], int len) {
  int i, j, k;

  for (i = 1; i <= len - 1; ++i) {
    for (j = 0; j < i; ++j) {
      if (arr[j] > arr[i]) {
        int temp;

        temp = arr[i];

        for (k = i; k > j; k--) arr[k] = arr[k - 1];

        arr[j] = temp;
      }
    }
  }
}

static INLINE void insertsortsad(int arr[], int idx[], int len) {
  int i, j, k;

  for (i = 1; i <= len - 1; ++i) {
    for (j = 0; j < i; ++j) {
      if (arr[j] > arr[i]) {
        int temp, tempi;

        temp = arr[i];
        tempi = idx[i];

        for (k = i; k > j; k--) {
          arr[k] = arr[k - 1];
          idx[k] = idx[k - 1];
        }

        arr[j] = temp;
        idx[j] = tempi;
      }
    }
  }
}

void vp8_initialize_rd_consts(VP8_COMP *cpi, MACROBLOCK *x, int Qvalue);
void vp8_rd_pick_inter_mode(VP8_COMP *cpi, MACROBLOCK *x, int recon_yoffset,
                            int recon_uvoffset, int *returnrate,
                            int *returndistortion, int *returnintra, int mb_row,
                            int mb_col);
void vp8_rd_pick_intra_mode(MACROBLOCK *x, int *rate);

static INLINE void get_plane_pointers(const YV12_BUFFER_CONFIG *fb,
                                      unsigned char *plane[3],
                                      unsigned int recon_yoffset,
                                      unsigned int recon_uvoffset) {
  plane[0] = fb->y_buffer + recon_yoffset;
  plane[1] = fb->u_buffer + recon_uvoffset;
  plane[2] = fb->v_buffer + recon_uvoffset;
}

static INLINE void get_predictor_pointers(const VP8_COMP *cpi,
                                          unsigned char *plane[4][3],
                                          unsigned int recon_yoffset,
                                          unsigned int recon_uvoffset) {
  if (cpi->ref_frame_flags & VP8_LAST_FRAME) {
    get_plane_pointers(&cpi->common.yv12_fb[cpi->common.lst_fb_idx],
                       plane[LAST_FRAME], recon_yoffset, recon_uvoffset);
  }

  if (cpi->ref_frame_flags & VP8_GOLD_FRAME) {
    get_plane_pointers(&cpi->common.yv12_fb[cpi->common.gld_fb_idx],
                       plane[GOLDEN_FRAME], recon_yoffset, recon_uvoffset);
  }

  if (cpi->ref_frame_flags & VP8_ALTR_FRAME) {
    get_plane_pointers(&cpi->common.yv12_fb[cpi->common.alt_fb_idx],
                       plane[ALTREF_FRAME], recon_yoffset, recon_uvoffset);
  }
}

static INLINE void get_reference_search_order(const VP8_COMP *cpi,
                                              int ref_frame_map[4]) {
  int i = 0;

  ref_frame_map[i++] = INTRA_FRAME;
  if (cpi->ref_frame_flags & VP8_LAST_FRAME) ref_frame_map[i++] = LAST_FRAME;
  if (cpi->ref_frame_flags & VP8_GOLD_FRAME) ref_frame_map[i++] = GOLDEN_FRAME;
  if (cpi->ref_frame_flags & VP8_ALTR_FRAME) ref_frame_map[i++] = ALTREF_FRAME;
  for (; i < 4; ++i) ref_frame_map[i] = -1;
}

void vp8_mv_pred(VP8_COMP *cpi, MACROBLOCKD *xd, const MODE_INFO *here,
                 int_mv *mvp, int refframe, int *ref_frame_sign_bias, int *sr,
                 int near_sadidx[]);
void vp8_cal_sad(VP8_COMP *cpi, MACROBLOCKD *xd, MACROBLOCK *x,
                 int recon_yoffset, int near_sadidx[]);
int VP8_UVSSE(MACROBLOCK *x);
int vp8_cost_mv_ref(MB_PREDICTION_MODE m, const int near_mv_ref_ct[4]);
void vp8_set_mbmode_and_mvs(MACROBLOCK *x, MB_PREDICTION_MODE mb, int_mv *mv);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_ENCODER_RDOPT_H_
