/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP9_ENCODER_VP9_TREEWRITER_H_
#define VPX_VP9_ENCODER_VP9_TREEWRITER_H_

#include "vpx_dsp/bitwriter.h"

#ifdef __cplusplus
extern "C" {
#endif

void vp9_tree_probs_from_distribution(vpx_tree tree,
                                      unsigned int branch_ct[/* n - 1 */][2],
                                      const unsigned int num_events[/* n */]);

struct vp9_token {
  int value;
  int len;
};

void vp9_tokens_from_tree(struct vp9_token *, const vpx_tree_index *);

static INLINE void vp9_write_tree(vpx_writer *w, const vpx_tree_index *tree,
                                  const vpx_prob *probs, int bits, int len,
                                  vpx_tree_index i) {
  do {
    const int bit = (bits >> --len) & 1;
    vpx_write(w, bit, probs[i >> 1]);
    i = tree[i + bit];
  } while (len);
}

static INLINE void vp9_write_token(vpx_writer *w, const vpx_tree_index *tree,
                                   const vpx_prob *probs,
                                   const struct vp9_token *token) {
  vp9_write_tree(w, tree, probs, token->value, token->len, 0);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_ENCODER_VP9_TREEWRITER_H_
