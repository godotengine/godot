// Copyright 2012 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Utilities for building and looking up Huffman trees.
//
// Author: Urvang Joshi (urvang@google.com)

#ifndef WEBP_UTILS_HUFFMAN_H_
#define WEBP_UTILS_HUFFMAN_H_

#include <assert.h>
#include "../webp/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// A node of a Huffman tree.
typedef struct {
  int symbol_;
  int children_;  // delta offset to both children (contiguous) or 0 if leaf.
} HuffmanTreeNode;

// Huffman Tree.
#define HUFF_LUT_BITS 7
#define HUFF_LUT (1U << HUFF_LUT_BITS)
typedef struct HuffmanTree HuffmanTree;
struct HuffmanTree {
  // Fast lookup for short bit lengths.
  uint8_t lut_bits_[HUFF_LUT];
  int16_t lut_symbol_[HUFF_LUT];
  int16_t lut_jump_[HUFF_LUT];
  // Complete tree for lookups.
  HuffmanTreeNode* root_;   // all the nodes, starting at root.
  int max_nodes_;           // max number of nodes
  int num_nodes_;           // number of currently occupied nodes
};

// Returns true if the given node is not a leaf of the Huffman tree.
static WEBP_INLINE int HuffmanTreeNodeIsNotLeaf(
    const HuffmanTreeNode* const node) {
  return node->children_;
}

// Go down one level. Most critical function. 'right_child' must be 0 or 1.
static WEBP_INLINE const HuffmanTreeNode* HuffmanTreeNextNode(
    const HuffmanTreeNode* node, int right_child) {
  return node + node->children_ + right_child;
}

// Releases the nodes of the Huffman tree.
// Note: It does NOT free 'tree' itself.
void HuffmanTreeRelease(HuffmanTree* const tree);

// Builds Huffman tree assuming code lengths are implicitly in symbol order.
// Returns false in case of error (invalid tree or memory error).
int HuffmanTreeBuildImplicit(HuffmanTree* const tree,
                             const int* const code_lengths,
                             int code_lengths_size);

// Build a Huffman tree with explicitly given lists of code lengths, codes
// and symbols. Verifies that all symbols added are smaller than max_symbol.
// Returns false in case of an invalid symbol, invalid tree or memory error.
int HuffmanTreeBuildExplicit(HuffmanTree* const tree,
                             const int* const code_lengths,
                             const int* const codes,
                             const int* const symbols, int max_symbol,
                             int num_symbols);

// Utility: converts Huffman code lengths to corresponding Huffman codes.
// 'huff_codes' should be pre-allocated.
// Returns false in case of error (memory allocation, invalid codes).
int HuffmanCodeLengthsToCodes(const int* const code_lengths,
                              int code_lengths_size, int* const huff_codes);


#ifdef __cplusplus
}    // extern "C"
#endif

#endif  // WEBP_UTILS_HUFFMAN_H_
