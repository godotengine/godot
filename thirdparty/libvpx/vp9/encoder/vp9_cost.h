/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP9_ENCODER_VP9_COST_H_
#define VPX_VP9_ENCODER_VP9_COST_H_

#include "vpx_dsp/prob.h"
#include "vpx/vpx_integer.h"

#ifdef __cplusplus
extern "C" {
#endif

extern const uint16_t vp9_prob_cost[256];

// The factor to scale from cost in bits to cost in vp9_prob_cost units.
#define VP9_PROB_COST_SHIFT 9

#define vp9_cost_zero(prob) (vp9_prob_cost[prob])

#define vp9_cost_one(prob) vp9_cost_zero(256 - (prob))

#define vp9_cost_bit(prob, bit) vp9_cost_zero((bit) ? 256 - (prob) : (prob))

static INLINE uint64_t cost_branch256(const unsigned int ct[2], vpx_prob p) {
  return (uint64_t)ct[0] * vp9_cost_zero(p) + (uint64_t)ct[1] * vp9_cost_one(p);
}

static INLINE int treed_cost(vpx_tree tree, const vpx_prob *probs, int bits,
                             int len) {
  int cost = 0;
  vpx_tree_index i = 0;

  do {
    const int bit = (bits >> --len) & 1;
    cost += vp9_cost_bit(probs[i >> 1], bit);
    i = tree[i + bit];
  } while (len);

  return cost;
}

void vp9_cost_tokens(int *costs, const vpx_prob *probs, vpx_tree tree);
void vp9_cost_tokens_skip(int *costs, const vpx_prob *probs, vpx_tree tree);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_ENCODER_VP9_COST_H_
