// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Library for creating Huffman codes from population counts.

#ifndef LIB_JXL_HUFFMAN_TREE_H_
#define LIB_JXL_HUFFMAN_TREE_H_

#include <stdint.h>
#include <stdlib.h>

namespace jxl {

// A node of a Huffman tree.
struct HuffmanTree {
  HuffmanTree(uint32_t count, int16_t left, int16_t right)
      : total_count(count), index_left(left), index_right_or_value(right) {}
  uint32_t total_count;
  int16_t index_left;
  int16_t index_right_or_value;
};

void SetDepth(const HuffmanTree& p, HuffmanTree* pool, uint8_t* depth,
              uint8_t level);

// This function will create a Huffman tree.
//
// The (data,length) contains the population counts.
// The tree_limit is the maximum bit depth of the Huffman codes.
//
// The depth contains the tree, i.e., how many bits are used for
// the symbol.
//
// See http://en.wikipedia.org/wiki/Huffman_coding
void CreateHuffmanTree(const uint32_t* data, size_t length, int tree_limit,
                       uint8_t* depth);

// Write a Huffman tree from bit depths into the bitstream representation
// of a Huffman tree. The generated Huffman tree is to be compressed once
// more using a Huffman tree
void WriteHuffmanTree(const uint8_t* depth, size_t length, size_t* tree_size,
                      uint8_t* tree, uint8_t* extra_bits_data);

// Get the actual bit values for a tree of bit depths.
void ConvertBitDepthsToSymbols(const uint8_t* depth, size_t len,
                               uint16_t* bits);

}  // namespace jxl

#endif  // LIB_JXL_HUFFMAN_TREE_H_
