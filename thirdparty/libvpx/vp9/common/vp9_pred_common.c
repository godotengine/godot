
/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "vp9/common/vp9_common.h"
#include "vp9/common/vp9_pred_common.h"
#include "vp9/common/vp9_seg_common.h"

int vp9_compound_reference_allowed(const VP9_COMMON *cm) {
  int i;
  for (i = 1; i < REFS_PER_FRAME; ++i)
    if (cm->ref_frame_sign_bias[i + 1] != cm->ref_frame_sign_bias[1]) return 1;

  return 0;
}

void vp9_setup_compound_reference_mode(VP9_COMMON *cm) {
  if (cm->ref_frame_sign_bias[LAST_FRAME] ==
      cm->ref_frame_sign_bias[GOLDEN_FRAME]) {
    cm->comp_fixed_ref = ALTREF_FRAME;
    cm->comp_var_ref[0] = LAST_FRAME;
    cm->comp_var_ref[1] = GOLDEN_FRAME;
  } else if (cm->ref_frame_sign_bias[LAST_FRAME] ==
             cm->ref_frame_sign_bias[ALTREF_FRAME]) {
    cm->comp_fixed_ref = GOLDEN_FRAME;
    cm->comp_var_ref[0] = LAST_FRAME;
    cm->comp_var_ref[1] = ALTREF_FRAME;
  } else {
    cm->comp_fixed_ref = LAST_FRAME;
    cm->comp_var_ref[0] = GOLDEN_FRAME;
    cm->comp_var_ref[1] = ALTREF_FRAME;
  }
}

int vp9_get_reference_mode_context(const VP9_COMMON *cm,
                                   const MACROBLOCKD *xd) {
  int ctx;
  const MODE_INFO *const above_mi = xd->above_mi;
  const MODE_INFO *const left_mi = xd->left_mi;
  const int has_above = !!above_mi;
  const int has_left = !!left_mi;
  // Note:
  // The mode info data structure has a one element border above and to the
  // left of the entries corresponding to real macroblocks.
  // The prediction flags in these dummy entries are initialized to 0.
  if (has_above && has_left) {  // both edges available
    if (!has_second_ref(above_mi) && !has_second_ref(left_mi))
      // neither edge uses comp pred (0/1)
      ctx = (above_mi->ref_frame[0] == cm->comp_fixed_ref) ^
            (left_mi->ref_frame[0] == cm->comp_fixed_ref);
    else if (!has_second_ref(above_mi))
      // one of two edges uses comp pred (2/3)
      ctx = 2 + (above_mi->ref_frame[0] == cm->comp_fixed_ref ||
                 !is_inter_block(above_mi));
    else if (!has_second_ref(left_mi))
      // one of two edges uses comp pred (2/3)
      ctx = 2 + (left_mi->ref_frame[0] == cm->comp_fixed_ref ||
                 !is_inter_block(left_mi));
    else  // both edges use comp pred (4)
      ctx = 4;
  } else if (has_above || has_left) {  // one edge available
    const MODE_INFO *edge_mi = has_above ? above_mi : left_mi;

    if (!has_second_ref(edge_mi))
      // edge does not use comp pred (0/1)
      ctx = edge_mi->ref_frame[0] == cm->comp_fixed_ref;
    else
      // edge uses comp pred (3)
      ctx = 3;
  } else {  // no edges available (1)
    ctx = 1;
  }
  assert(ctx >= 0 && ctx < COMP_INTER_CONTEXTS);
  return ctx;
}

// Returns a context number for the given MB prediction signal
int vp9_get_pred_context_comp_ref_p(const VP9_COMMON *cm,
                                    const MACROBLOCKD *xd) {
  int pred_context;
  const MODE_INFO *const above_mi = xd->above_mi;
  const MODE_INFO *const left_mi = xd->left_mi;
  const int above_in_image = !!above_mi;
  const int left_in_image = !!left_mi;

  // Note:
  // The mode info data structure has a one element border above and to the
  // left of the entries corresponding to real macroblocks.
  // The prediction flags in these dummy entries are initialized to 0.
  const int fix_ref_idx = cm->ref_frame_sign_bias[cm->comp_fixed_ref];
  const int var_ref_idx = !fix_ref_idx;

  if (above_in_image && left_in_image) {  // both edges available
    const int above_intra = !is_inter_block(above_mi);
    const int left_intra = !is_inter_block(left_mi);

    if (above_intra && left_intra) {  // intra/intra (2)
      pred_context = 2;
    } else if (above_intra || left_intra) {  // intra/inter
      const MODE_INFO *edge_mi = above_intra ? left_mi : above_mi;

      if (!has_second_ref(edge_mi))  // single pred (1/3)
        pred_context = 1 + 2 * (edge_mi->ref_frame[0] != cm->comp_var_ref[1]);
      else  // comp pred (1/3)
        pred_context =
            1 + 2 * (edge_mi->ref_frame[var_ref_idx] != cm->comp_var_ref[1]);
    } else {  // inter/inter
      const int l_sg = !has_second_ref(left_mi);
      const int a_sg = !has_second_ref(above_mi);
      const MV_REFERENCE_FRAME vrfa =
          a_sg ? above_mi->ref_frame[0] : above_mi->ref_frame[var_ref_idx];
      const MV_REFERENCE_FRAME vrfl =
          l_sg ? left_mi->ref_frame[0] : left_mi->ref_frame[var_ref_idx];

      if (vrfa == vrfl && cm->comp_var_ref[1] == vrfa) {
        pred_context = 0;
      } else if (l_sg && a_sg) {  // single/single
        if ((vrfa == cm->comp_fixed_ref && vrfl == cm->comp_var_ref[0]) ||
            (vrfl == cm->comp_fixed_ref && vrfa == cm->comp_var_ref[0]))
          pred_context = 4;
        else if (vrfa == vrfl)
          pred_context = 3;
        else
          pred_context = 1;
      } else if (l_sg || a_sg) {  // single/comp
        const MV_REFERENCE_FRAME vrfc = l_sg ? vrfa : vrfl;
        const MV_REFERENCE_FRAME rfs = a_sg ? vrfa : vrfl;
        if (vrfc == cm->comp_var_ref[1] && rfs != cm->comp_var_ref[1])
          pred_context = 1;
        else if (rfs == cm->comp_var_ref[1] && vrfc != cm->comp_var_ref[1])
          pred_context = 2;
        else
          pred_context = 4;
      } else if (vrfa == vrfl) {  // comp/comp
        pred_context = 4;
      } else {
        pred_context = 2;
      }
    }
  } else if (above_in_image || left_in_image) {  // one edge available
    const MODE_INFO *edge_mi = above_in_image ? above_mi : left_mi;

    if (!is_inter_block(edge_mi)) {
      pred_context = 2;
    } else {
      if (has_second_ref(edge_mi))
        pred_context =
            4 * (edge_mi->ref_frame[var_ref_idx] != cm->comp_var_ref[1]);
      else
        pred_context = 3 * (edge_mi->ref_frame[0] != cm->comp_var_ref[1]);
    }
  } else {  // no edges available (2)
    pred_context = 2;
  }
  assert(pred_context >= 0 && pred_context < REF_CONTEXTS);

  return pred_context;
}

int vp9_get_pred_context_single_ref_p1(const MACROBLOCKD *xd) {
  int pred_context;
  const MODE_INFO *const above_mi = xd->above_mi;
  const MODE_INFO *const left_mi = xd->left_mi;
  const int has_above = !!above_mi;
  const int has_left = !!left_mi;
  // Note:
  // The mode info data structure has a one element border above and to the
  // left of the entries corresponding to real macroblocks.
  // The prediction flags in these dummy entries are initialized to 0.
  if (has_above && has_left) {  // both edges available
    const int above_intra = !is_inter_block(above_mi);
    const int left_intra = !is_inter_block(left_mi);

    if (above_intra && left_intra) {  // intra/intra
      pred_context = 2;
    } else if (above_intra || left_intra) {  // intra/inter or inter/intra
      const MODE_INFO *edge_mi = above_intra ? left_mi : above_mi;
      if (!has_second_ref(edge_mi))
        pred_context = 4 * (edge_mi->ref_frame[0] == LAST_FRAME);
      else
        pred_context = 1 + (edge_mi->ref_frame[0] == LAST_FRAME ||
                            edge_mi->ref_frame[1] == LAST_FRAME);
    } else {  // inter/inter
      const int above_has_second = has_second_ref(above_mi);
      const int left_has_second = has_second_ref(left_mi);
      const MV_REFERENCE_FRAME above0 = above_mi->ref_frame[0];
      const MV_REFERENCE_FRAME above1 = above_mi->ref_frame[1];
      const MV_REFERENCE_FRAME left0 = left_mi->ref_frame[0];
      const MV_REFERENCE_FRAME left1 = left_mi->ref_frame[1];

      if (above_has_second && left_has_second) {
        pred_context = 1 + (above0 == LAST_FRAME || above1 == LAST_FRAME ||
                            left0 == LAST_FRAME || left1 == LAST_FRAME);
      } else if (above_has_second || left_has_second) {
        const MV_REFERENCE_FRAME rfs = !above_has_second ? above0 : left0;
        const MV_REFERENCE_FRAME crf1 = above_has_second ? above0 : left0;
        const MV_REFERENCE_FRAME crf2 = above_has_second ? above1 : left1;

        if (rfs == LAST_FRAME)
          pred_context = 3 + (crf1 == LAST_FRAME || crf2 == LAST_FRAME);
        else
          pred_context = (crf1 == LAST_FRAME || crf2 == LAST_FRAME);
      } else {
        pred_context = 2 * (above0 == LAST_FRAME) + 2 * (left0 == LAST_FRAME);
      }
    }
  } else if (has_above || has_left) {  // one edge available
    const MODE_INFO *edge_mi = has_above ? above_mi : left_mi;
    if (!is_inter_block(edge_mi)) {  // intra
      pred_context = 2;
    } else {  // inter
      if (!has_second_ref(edge_mi))
        pred_context = 4 * (edge_mi->ref_frame[0] == LAST_FRAME);
      else
        pred_context = 1 + (edge_mi->ref_frame[0] == LAST_FRAME ||
                            edge_mi->ref_frame[1] == LAST_FRAME);
    }
  } else {  // no edges available
    pred_context = 2;
  }

  assert(pred_context >= 0 && pred_context < REF_CONTEXTS);
  return pred_context;
}

int vp9_get_pred_context_single_ref_p2(const MACROBLOCKD *xd) {
  int pred_context;
  const MODE_INFO *const above_mi = xd->above_mi;
  const MODE_INFO *const left_mi = xd->left_mi;
  const int has_above = !!above_mi;
  const int has_left = !!left_mi;

  // Note:
  // The mode info data structure has a one element border above and to the
  // left of the entries corresponding to real macroblocks.
  // The prediction flags in these dummy entries are initialized to 0.
  if (has_above && has_left) {  // both edges available
    const int above_intra = !is_inter_block(above_mi);
    const int left_intra = !is_inter_block(left_mi);

    if (above_intra && left_intra) {  // intra/intra
      pred_context = 2;
    } else if (above_intra || left_intra) {  // intra/inter or inter/intra
      const MODE_INFO *edge_mi = above_intra ? left_mi : above_mi;
      if (!has_second_ref(edge_mi)) {
        if (edge_mi->ref_frame[0] == LAST_FRAME)
          pred_context = 3;
        else
          pred_context = 4 * (edge_mi->ref_frame[0] == GOLDEN_FRAME);
      } else {
        pred_context = 1 + 2 * (edge_mi->ref_frame[0] == GOLDEN_FRAME ||
                                edge_mi->ref_frame[1] == GOLDEN_FRAME);
      }
    } else {  // inter/inter
      const int above_has_second = has_second_ref(above_mi);
      const int left_has_second = has_second_ref(left_mi);
      const MV_REFERENCE_FRAME above0 = above_mi->ref_frame[0];
      const MV_REFERENCE_FRAME above1 = above_mi->ref_frame[1];
      const MV_REFERENCE_FRAME left0 = left_mi->ref_frame[0];
      const MV_REFERENCE_FRAME left1 = left_mi->ref_frame[1];

      if (above_has_second && left_has_second) {
        if (above0 == left0 && above1 == left1)
          pred_context =
              3 * (above0 == GOLDEN_FRAME || above1 == GOLDEN_FRAME ||
                   left0 == GOLDEN_FRAME || left1 == GOLDEN_FRAME);
        else
          pred_context = 2;
      } else if (above_has_second || left_has_second) {
        const MV_REFERENCE_FRAME rfs = !above_has_second ? above0 : left0;
        const MV_REFERENCE_FRAME crf1 = above_has_second ? above0 : left0;
        const MV_REFERENCE_FRAME crf2 = above_has_second ? above1 : left1;

        if (rfs == GOLDEN_FRAME)
          pred_context = 3 + (crf1 == GOLDEN_FRAME || crf2 == GOLDEN_FRAME);
        else if (rfs == ALTREF_FRAME)
          pred_context = crf1 == GOLDEN_FRAME || crf2 == GOLDEN_FRAME;
        else
          pred_context = 1 + 2 * (crf1 == GOLDEN_FRAME || crf2 == GOLDEN_FRAME);
      } else {
        if (above0 == LAST_FRAME && left0 == LAST_FRAME) {
          pred_context = 3;
        } else if (above0 == LAST_FRAME || left0 == LAST_FRAME) {
          const MV_REFERENCE_FRAME edge0 =
              (above0 == LAST_FRAME) ? left0 : above0;
          pred_context = 4 * (edge0 == GOLDEN_FRAME);
        } else {
          pred_context =
              2 * (above0 == GOLDEN_FRAME) + 2 * (left0 == GOLDEN_FRAME);
        }
      }
    }
  } else if (has_above || has_left) {  // one edge available
    const MODE_INFO *edge_mi = has_above ? above_mi : left_mi;

    if (!is_inter_block(edge_mi) ||
        (edge_mi->ref_frame[0] == LAST_FRAME && !has_second_ref(edge_mi)))
      pred_context = 2;
    else if (!has_second_ref(edge_mi))
      pred_context = 4 * (edge_mi->ref_frame[0] == GOLDEN_FRAME);
    else
      pred_context = 3 * (edge_mi->ref_frame[0] == GOLDEN_FRAME ||
                          edge_mi->ref_frame[1] == GOLDEN_FRAME);
  } else {  // no edges available (2)
    pred_context = 2;
  }
  assert(pred_context >= 0 && pred_context < REF_CONTEXTS);
  return pred_context;
}
