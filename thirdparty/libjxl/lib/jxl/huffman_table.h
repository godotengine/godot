// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_HUFFMAN_TABLE_H_
#define LIB_JXL_HUFFMAN_TABLE_H_

#include <stdint.h>
#include <stdlib.h>

namespace jxl {

struct HuffmanCode {
  uint8_t bits;   /* number of bits used for this symbol */
  uint16_t value; /* symbol value or table offset */
};

/* Builds Huffman lookup table assuming code lengths are in symbol order. */
/* Returns 0 in case of error (invalid tree or memory error), otherwise
   populated size of table. */
uint32_t BuildHuffmanTable(HuffmanCode* root_table, int root_bits,
                           const uint8_t* code_lengths,
                           size_t code_lengths_size, uint16_t* count);

}  // namespace jxl

#endif  // LIB_JXL_HUFFMAN_TABLE_H_
