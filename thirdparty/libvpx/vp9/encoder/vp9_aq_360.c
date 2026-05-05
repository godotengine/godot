/*
 *  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <math.h>

#include "vpx_ports/mem.h"
#include "vpx_ports/system_state.h"

#include "vp9/encoder/vp9_aq_360.h"
#include "vp9/encoder/vp9_aq_variance.h"

#include "vp9/common/vp9_seg_common.h"

#include "vp9/encoder/vp9_ratectrl.h"
#include "vp9/encoder/vp9_rd.h"
#include "vp9/encoder/vp9_segmentation.h"

static const double rate_ratio[MAX_SEGMENTS] = { 1.0, 0.75, 0.6, 0.5,
                                                 0.4, 0.3,  0.25 };

// Sets segment id 0 for the equatorial region, 1 for temperate region
// and 2 for the polar regions
unsigned int vp9_360aq_segment_id(int mi_row, int mi_rows) {
  if (mi_row < mi_rows / 8 || mi_row > mi_rows - mi_rows / 8)
    return 2;
  else if (mi_row < mi_rows / 4 || mi_row > mi_rows - mi_rows / 4)
    return 1;
  else
    return 0;
}

void vp9_360aq_frame_setup(VP9_COMP *cpi) {
  VP9_COMMON *cm = &cpi->common;
  struct segmentation *seg = &cm->seg;
  int i;

  if (frame_is_intra_only(cm) || cpi->force_update_segmentation ||
      cm->error_resilient_mode) {
    vp9_enable_segmentation(seg);
    vp9_clearall_segfeatures(seg);

    seg->abs_delta = SEGMENT_DELTADATA;

    vpx_clear_system_state();

    for (i = 0; i < MAX_SEGMENTS; ++i) {
      int qindex_delta =
          vp9_compute_qdelta_by_rate(&cpi->rc, cm->frame_type, cm->base_qindex,
                                     rate_ratio[i], cm->bit_depth);

      // We don't allow qindex 0 in a segment if the base value is not 0.
      // Q index 0 (lossless) implies 4x4 encoding only and in AQ mode a segment
      // Q delta is sometimes applied without going back around the rd loop.
      // This could lead to an illegal combination of partition size and q.
      if ((cm->base_qindex != 0) && ((cm->base_qindex + qindex_delta) == 0)) {
        qindex_delta = -cm->base_qindex + 1;
      }

      // No need to enable SEG_LVL_ALT_Q for this segment.
      if (rate_ratio[i] == 1.0) {
        continue;
      }

      vp9_set_segdata(seg, i, SEG_LVL_ALT_Q, qindex_delta);
      vp9_enable_segfeature(seg, i, SEG_LVL_ALT_Q);
    }
  }
}
