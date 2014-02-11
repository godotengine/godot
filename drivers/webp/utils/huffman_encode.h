// Copyright 2011 Google Inc. All Rights Reserved.
//
// This code is licensed under the same terms as WebM:
//  Software License Agreement:  http://www.webmproject.org/license/software/
//  Additional IP Rights Grant:  http://www.webmproject.org/license/additional/
// -----------------------------------------------------------------------------
//
// Author: Jyrki Alakuijala (jyrki@google.com)
//
// Entropy encoding (Huffman) for webp lossless

#ifndef WEBP_UTILS_HUFFMAN_ENCODE_H_
#define WEBP_UTILS_HUFFMAN_ENCODE_H_

#include "../webp/types.h"

#if defined(__cplusplus) || defined(c_plusplus)
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

// Turn the Huffman tree into a token sequence.
// Returns the number of tokens used.
int VP8LCreateCompressedHuffmanTree(const HuffmanTreeCode* const tree,
                                    HuffmanTreeToken* tokens, int max_tokens);

// Create an optimized tree, and tokenize it.
int VP8LCreateHuffmanTree(int* const histogram, int tree_depth_limit,
                          HuffmanTreeCode* const tree);

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif

#endif  // WEBP_UTILS_HUFFMAN_ENCODE_H_
