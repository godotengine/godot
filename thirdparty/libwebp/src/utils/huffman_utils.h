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

#ifndef WEBP_UTILS_HUFFMAN_UTILS_H_
#define WEBP_UTILS_HUFFMAN_UTILS_H_

#include <assert.h>
#include "src/webp/format_constants.h"
#include "src/webp/types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HUFFMAN_TABLE_BITS      8
#define HUFFMAN_TABLE_MASK      ((1 << HUFFMAN_TABLE_BITS) - 1)

#define LENGTHS_TABLE_BITS      7
#define LENGTHS_TABLE_MASK      ((1 << LENGTHS_TABLE_BITS) - 1)


// Huffman lookup table entry
typedef struct {
  uint8_t bits;     // number of bits used for this symbol
  uint16_t value;   // symbol value or table offset
} HuffmanCode;

// long version for holding 32b values
typedef struct {
  int bits;         // number of bits used for this symbol,
                    // or an impossible value if not a literal code.
  uint32_t value;   // 32b packed ARGB value if literal,
                    // or non-literal symbol otherwise
} HuffmanCode32;

#define HUFFMAN_PACKED_BITS 6
#define HUFFMAN_PACKED_TABLE_SIZE (1u << HUFFMAN_PACKED_BITS)

// Huffman table group.
// Includes special handling for the following cases:
//  - is_trivial_literal: one common literal base for RED/BLUE/ALPHA (not GREEN)
//  - is_trivial_code: only 1 code (no bit is read from bitstream)
//  - use_packed_table: few enough literal symbols, so all the bit codes
//    can fit into a small look-up table packed_table[]
// The common literal base, if applicable, is stored in 'literal_arb'.
typedef struct HTreeGroup HTreeGroup;
struct HTreeGroup {
  HuffmanCode* htrees[HUFFMAN_CODES_PER_META_CODE];
  int      is_trivial_literal;  // True, if huffman trees for Red, Blue & Alpha
                                // Symbols are trivial (have a single code).
  uint32_t literal_arb;         // If is_trivial_literal is true, this is the
                                // ARGB value of the pixel, with Green channel
                                // being set to zero.
  int is_trivial_code;          // true if is_trivial_literal with only one code
  int use_packed_table;         // use packed table below for short literal code
  // table mapping input bits to a packed values, or escape case to literal code
  HuffmanCode32 packed_table[HUFFMAN_PACKED_TABLE_SIZE];
};

// Creates the instance of HTreeGroup with specified number of tree-groups.
HTreeGroup* VP8LHtreeGroupsNew(int num_htree_groups);

// Releases the memory allocated for HTreeGroup.
void VP8LHtreeGroupsFree(HTreeGroup* const htree_groups);

// Builds Huffman lookup table assuming code lengths are in symbol order.
// The 'code_lengths' is pre-allocated temporary memory buffer used for creating
// the huffman table.
// Returns built table size or 0 in case of error (invalid tree or
// memory error).
int VP8LBuildHuffmanTable(HuffmanCode* const root_table, int root_bits,
                          const int code_lengths[], int code_lengths_size);

#ifdef __cplusplus
}    // extern "C"
#endif

#endif  // WEBP_UTILS_HUFFMAN_UTILS_H_
