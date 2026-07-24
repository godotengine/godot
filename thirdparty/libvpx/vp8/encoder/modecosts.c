/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "vp8/common/blockd.h"
#include "modecosts.h"
#include "onyx_int.h"
#include "treewriter.h"
#include "vp8/common/entropymode.h"

void vp8_init_mode_costs(VP8_COMP *c) {
  VP8_COMMON *x = &c->common;
  struct rd_costs_struct *rd_costs = &c->rd_costs;

  {
    const vp8_tree_p T = vp8_bmode_tree;

    int i = 0;

    do {
      int j = 0;

      do {
        vp8_cost_tokens(rd_costs->bmode_costs[i][j], vp8_kf_bmode_prob[i][j],
                        T);
      } while (++j < VP8_BINTRAMODES);
    } while (++i < VP8_BINTRAMODES);

    vp8_cost_tokens(rd_costs->inter_bmode_costs, x->fc.bmode_prob, T);
  }
  vp8_cost_tokens(rd_costs->inter_bmode_costs, x->fc.sub_mv_ref_prob,
                  vp8_sub_mv_ref_tree);

  vp8_cost_tokens(rd_costs->mbmode_cost[1], x->fc.ymode_prob, vp8_ymode_tree);
  vp8_cost_tokens(rd_costs->mbmode_cost[0], vp8_kf_ymode_prob,
                  vp8_kf_ymode_tree);

  vp8_cost_tokens(rd_costs->intra_uv_mode_cost[1], x->fc.uv_mode_prob,
                  vp8_uv_mode_tree);
  vp8_cost_tokens(rd_costs->intra_uv_mode_cost[0], vp8_kf_uv_mode_prob,
                  vp8_uv_mode_tree);
}
