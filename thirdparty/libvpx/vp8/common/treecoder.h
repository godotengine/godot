/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP8_COMMON_TREECODER_H_
#define VPX_VP8_COMMON_TREECODER_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned char vp8bc_index_t; /* probability index */

typedef unsigned char vp8_prob;

#define vp8_prob_half ((vp8_prob)128)

typedef signed char vp8_tree_index;
struct bool_coder_spec;

typedef struct bool_coder_spec bool_coder_spec;
typedef struct bool_writer bool_writer;
typedef struct bool_reader bool_reader;

typedef const bool_coder_spec c_bool_coder_spec;
typedef const bool_writer c_bool_writer;
typedef const bool_reader c_bool_reader;

#define vp8_complement(x) (255 - (x))

/* We build coding trees compactly in arrays.
   Each node of the tree is a pair of vp8_tree_indices.
   Array index often references a corresponding probability table.
   Index <= 0 means done encoding/decoding and value = -Index,
   Index > 0 means need another bit, specification at index.
   Nonnegative indices are always even;  processing begins at node 0. */

typedef const vp8_tree_index vp8_tree[], *vp8_tree_p;

typedef const struct vp8_token_struct {
  int value;
  int Len;
} vp8_token;

/* Construct encoding array from tree. */

void vp8_tokens_from_tree(struct vp8_token_struct *, vp8_tree);
void vp8_tokens_from_tree_offset(struct vp8_token_struct *, vp8_tree,
                                 int offset);

/* Convert array of token occurrence counts into a table of probabilities
   for the associated binary encoding tree.  Also writes count of branches
   taken for each node on the tree; this facilitiates decisions as to
   probability updates. */

void vp8_tree_probs_from_distribution(int n, /* n = size of alphabet */
                                      vp8_token tok[/* n */], vp8_tree tree,
                                      vp8_prob probs[/* n-1 */],
                                      unsigned int branch_ct[/* n-1 */][2],
                                      const unsigned int num_events[/* n */],
                                      unsigned int Pfactor, int Round);

/* Variant of above using coder spec rather than hardwired 8-bit probs. */

void vp8bc_tree_probs_from_distribution(int n, /* n = size of alphabet */
                                        vp8_token tok[/* n */], vp8_tree tree,
                                        vp8_prob probs[/* n-1 */],
                                        unsigned int branch_ct[/* n-1 */][2],
                                        const unsigned int num_events[/* n */],
                                        c_bool_coder_spec *s);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_COMMON_TREECODER_H_
