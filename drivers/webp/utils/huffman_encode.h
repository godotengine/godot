// Copyright 2011 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Author: Jyrki Alakuijala (jyrki@google.com)
//
// Entropy encoding (Huffman) for webp lossless

#ifndef WEBP_UTILS_HUFFMAN_ENCODE_H_
#define WEBP_UTILS_HUFFMAN_ENCODE_H_

#include "../webp/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Struct for holding the tree header in coded form.
typedef struct {
  uint8_t code;         // value (0..15) or escape code (16,17,18)
  uint8_t extra_bits;   // extra bits for escape codes
} HuffmanTreeToken;

// Struct to represent the tree codes (depth and bits array).
typedef struct {
  int       num_symbols;   // Number of symbols.
  uint8_t*  code_lengths;  // Code lengths of the symbols.
  uint16_t* codes;         // Symbol Codes.
} HuffmanTreeCode;

// Struct to represent the Huffman tree.
typedef struct {
  uint32_t total_count_;   // Symbol frequency.
  int value_;              // Symbol value.
  int pool_index_left_;    // Index for the left sub-tree.
  int pool_index_right_;   // Index for the right sub-tree.
} HuffmanTree;

// Turn the Huffman tree into a token sequence.
// Returns the number of tokens used.
int VP8LCreateCompressedHuffmanTree(const HuffmanTreeCode* const tree,
                                    HuffmanTreeToken* tokens, int max_tokens);

// Create an optimized tree, and tokenize it.
// 'buf_rle' and 'huff_tree' are pre-allocated and the 'tree' is the constructed
// huffman code tree.
void VP8LCreateHuffmanTree(uint32_t* const histogram, int tree_depth_limit,
                           uint8_t* const buf_rle, HuffmanTree* const huff_tree,
                           HuffmanTreeCode* const tree);

#ifdef __cplusplus
}
#endif

#endif  // WEBP_UTILS_HUFFMAN_ENCODE_H_
