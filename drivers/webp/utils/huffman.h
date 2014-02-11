// Copyright 2012 Google Inc. All Rights Reserved.
//
// This code is licensed under the same terms as WebM:
//  Software License Agreement:  http://www.webmproject.org/license/software/
//  Additional IP Rights Grant:  http://www.webmproject.org/license/additional/
// -----------------------------------------------------------------------------
//
// Utilities for building and looking up Huffman trees.
//
// Author: Urvang Joshi (urvang@google.com)

#ifndef WEBP_UTILS_HUFFMAN_H_
#define WEBP_UTILS_HUFFMAN_H_

#include <assert.h>
#include "../webp/types.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

// A node of a Huffman tree.
typedef struct {
  int symbol_;
  int children_;  // delta offset to both children (contiguous) or 0 if leaf.
} HuffmanTreeNode;

// Huffman Tree.
typedef struct HuffmanTree HuffmanTree;
struct HuffmanTree {
  HuffmanTreeNode* root_;   // all the nodes, starting at root.
  int max_nodes_;           // max number of nodes
  int num_nodes_;           // number of currently occupied nodes
};

// Returns true if the given node is a leaf of the Huffman tree.
static WEBP_INLINE int HuffmanTreeNodeIsLeaf(
    const HuffmanTreeNode* const node) {
  return (node->children_ == 0);
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


#if defined(__cplusplus) || defined(c_plusplus)
}    // extern "C"
#endif

#endif  // WEBP_UTILS_HUFFMAN_H_
