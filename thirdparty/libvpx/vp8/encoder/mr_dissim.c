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
#include "vpx_config.h"
#include "onyx_int.h"
#include "mr_dissim.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_mem/vpx_mem.h"
#include "rdopt.h"
#include "vp8/common/common.h"

void vp8_cal_low_res_mb_cols(VP8_COMP *cpi) {
  int low_res_w;

  /* Support arbitrary down-sampling factor */
  unsigned int iw = cpi->oxcf.Width * cpi->oxcf.mr_down_sampling_factor.den +
                    cpi->oxcf.mr_down_sampling_factor.num - 1;

  low_res_w = iw / cpi->oxcf.mr_down_sampling_factor.num;
  cpi->mr_low_res_mb_cols = ((low_res_w + 15) >> 4);
}

#define GET_MV(x)                         \
  if (x->mbmi.ref_frame != INTRA_FRAME) { \
    mvx[cnt] = x->mbmi.mv.as_mv.row;      \
    mvy[cnt] = x->mbmi.mv.as_mv.col;      \
    cnt++;                                \
  }

#define GET_MV_SIGN(x)                                  \
  if (x->mbmi.ref_frame != INTRA_FRAME) {               \
    mvx[cnt] = x->mbmi.mv.as_mv.row;                    \
    mvy[cnt] = x->mbmi.mv.as_mv.col;                    \
    if (cm->ref_frame_sign_bias[x->mbmi.ref_frame] !=   \
        cm->ref_frame_sign_bias[tmp->mbmi.ref_frame]) { \
      mvx[cnt] *= -1;                                   \
      mvy[cnt] *= -1;                                   \
    }                                                   \
    cnt++;                                              \
  }

void vp8_cal_dissimilarity(VP8_COMP *cpi) {
  VP8_COMMON *cm = &cpi->common;

  /* Note: The first row & first column in mip are outside the frame, which
   * were initialized to all 0.(ref_frame, mode, mv...)
   * Their ref_frame = 0 means they won't be counted in the following
   * calculation.
   */
  if (cpi->oxcf.mr_total_resolutions > 1 &&
      cpi->oxcf.mr_encoder_id < (cpi->oxcf.mr_total_resolutions - 1)) {
    /* Store info for show/no-show frames for supporting alt_ref.
     * If parent frame is alt_ref, child has one too.
     */
    LOWER_RES_FRAME_INFO *store_info =
        (LOWER_RES_FRAME_INFO *)cpi->oxcf.mr_low_res_mode_info;

    store_info->frame_type = cm->frame_type;

    if (cm->frame_type != KEY_FRAME) {
      int i;
      store_info->is_frame_dropped = 0;
      for (i = 1; i < MAX_REF_FRAMES; ++i)
        store_info->low_res_ref_frames[i] = cpi->current_ref_frames[i];
    }

    if (cm->frame_type != KEY_FRAME) {
      int mb_row;
      int mb_col;
      /* Point to beginning of allocated MODE_INFO arrays. */
      MODE_INFO *tmp = cm->mip + cm->mode_info_stride;
      LOWER_RES_MB_INFO *store_mode_info = store_info->mb_info;

      for (mb_row = 0; mb_row < cm->mb_rows; ++mb_row) {
        tmp++;
        for (mb_col = 0; mb_col < cm->mb_cols; ++mb_col) {
          int dissim = INT_MAX;

          if (tmp->mbmi.ref_frame != INTRA_FRAME) {
            int mvx[8];
            int mvy[8];
            int mmvx;
            int mmvy;
            int cnt = 0;
            const MODE_INFO *here = tmp;
            const MODE_INFO *above = here - cm->mode_info_stride;
            const MODE_INFO *left = here - 1;
            const MODE_INFO *aboveleft = above - 1;
            const MODE_INFO *aboveright = NULL;
            const MODE_INFO *right = NULL;
            const MODE_INFO *belowleft = NULL;
            const MODE_INFO *below = NULL;
            const MODE_INFO *belowright = NULL;

            /* If alternate reference frame is used, we have to
             * check sign of MV. */
            if (cpi->oxcf.play_alternate) {
              /* Gather mv of neighboring MBs */
              GET_MV_SIGN(above)
              GET_MV_SIGN(left)
              GET_MV_SIGN(aboveleft)

              if (mb_col < (cm->mb_cols - 1)) {
                right = here + 1;
                aboveright = above + 1;
                GET_MV_SIGN(right)
                GET_MV_SIGN(aboveright)
              }

              if (mb_row < (cm->mb_rows - 1)) {
                below = here + cm->mode_info_stride;
                belowleft = below - 1;
                GET_MV_SIGN(below)
                GET_MV_SIGN(belowleft)
              }

              if (mb_col < (cm->mb_cols - 1) && mb_row < (cm->mb_rows - 1)) {
                belowright = below + 1;
                GET_MV_SIGN(belowright)
              }
            } else {
              /* No alt_ref and gather mv of neighboring MBs */
              GET_MV(above)
              GET_MV(left)
              GET_MV(aboveleft)

              if (mb_col < (cm->mb_cols - 1)) {
                right = here + 1;
                aboveright = above + 1;
                GET_MV(right)
                GET_MV(aboveright)
              }

              if (mb_row < (cm->mb_rows - 1)) {
                below = here + cm->mode_info_stride;
                belowleft = below - 1;
                GET_MV(below)
                GET_MV(belowleft)
              }

              if (mb_col < (cm->mb_cols - 1) && mb_row < (cm->mb_rows - 1)) {
                belowright = below + 1;
                GET_MV(belowright)
              }
            }

            if (cnt > 0) {
              int max_mvx = mvx[0];
              int min_mvx = mvx[0];
              int max_mvy = mvy[0];
              int min_mvy = mvy[0];
              int i;

              if (cnt > 1) {
                for (i = 1; i < cnt; ++i) {
                  if (mvx[i] > max_mvx)
                    max_mvx = mvx[i];
                  else if (mvx[i] < min_mvx)
                    min_mvx = mvx[i];
                  if (mvy[i] > max_mvy)
                    max_mvy = mvy[i];
                  else if (mvy[i] < min_mvy)
                    min_mvy = mvy[i];
                }
              }

              mmvx = VPXMAX(abs(min_mvx - here->mbmi.mv.as_mv.row),
                            abs(max_mvx - here->mbmi.mv.as_mv.row));
              mmvy = VPXMAX(abs(min_mvy - here->mbmi.mv.as_mv.col),
                            abs(max_mvy - here->mbmi.mv.as_mv.col));
              dissim = VPXMAX(mmvx, mmvy);
            }
          }

          /* Store mode info for next resolution encoding */
          store_mode_info->mode = tmp->mbmi.mode;
          store_mode_info->ref_frame = tmp->mbmi.ref_frame;
          store_mode_info->mv.as_int = tmp->mbmi.mv.as_int;
          store_mode_info->dissim = dissim;
          tmp++;
          store_mode_info++;
        }
      }
    }
  }
}

/* This function is called only when this frame is dropped at current
   resolution level. */
void vp8_store_drop_frame_info(VP8_COMP *cpi) {
  /* If the frame is dropped in lower-resolution encoding, this information
     is passed to higher resolution level so that the encoder knows there
     is no mode & motion info available.
   */
  if (cpi->oxcf.mr_total_resolutions > 1 &&
      cpi->oxcf.mr_encoder_id < (cpi->oxcf.mr_total_resolutions - 1)) {
    /* Store info for show/no-show frames for supporting alt_ref.
     * If parent frame is alt_ref, child has one too.
     */
    LOWER_RES_FRAME_INFO *store_info =
        (LOWER_RES_FRAME_INFO *)cpi->oxcf.mr_low_res_mode_info;

    /* Set frame_type to be INTER_FRAME since we won't drop key frame. */
    store_info->frame_type = INTER_FRAME;
    store_info->is_frame_dropped = 1;
  }
}
