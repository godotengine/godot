/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "segmentation.h"
#include "vpx_mem/vpx_mem.h"

void vp8_update_gf_usage_maps(VP8_COMP *cpi, VP8_COMMON *cm, MACROBLOCK *x) {
  int mb_row, mb_col;

  MODE_INFO *this_mb_mode_info = cm->mi;

  x->gf_active_ptr = (signed char *)cpi->gf_active_flags;

  if ((cm->frame_type == KEY_FRAME) || (cm->refresh_golden_frame)) {
    /* Reset Gf usage monitors */
    memset(cpi->gf_active_flags, 1, (cm->mb_rows * cm->mb_cols));
    cpi->gf_active_count = cm->mb_rows * cm->mb_cols;
  } else {
    /* for each macroblock row in image */
    for (mb_row = 0; mb_row < cm->mb_rows; ++mb_row) {
      /* for each macroblock col in image */
      for (mb_col = 0; mb_col < cm->mb_cols; ++mb_col) {
        /* If using golden then set GF active flag if not already set.
         * If using last frame 0,0 mode then leave flag as it is
         * else if using non 0,0 motion or intra modes then clear
         * flag if it is currently set
         */
        if ((this_mb_mode_info->mbmi.ref_frame == GOLDEN_FRAME) ||
            (this_mb_mode_info->mbmi.ref_frame == ALTREF_FRAME)) {
          if (*(x->gf_active_ptr) == 0) {
            *(x->gf_active_ptr) = 1;
            cpi->gf_active_count++;
          }
        } else if ((this_mb_mode_info->mbmi.mode != ZEROMV) &&
                   *(x->gf_active_ptr)) {
          *(x->gf_active_ptr) = 0;
          cpi->gf_active_count--;
        }

        x->gf_active_ptr++;  /* Step onto next entry */
        this_mb_mode_info++; /* skip to next mb */
      }

      /* this is to account for the border */
      this_mb_mode_info++;
    }
  }
}
