// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_DEC_HUFFMAN_H_
#define LIB_JXL_DEC_HUFFMAN_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/huffman_table.h"

namespace jxl {

static constexpr size_t kHuffmanTableBits = 8u;

struct HuffmanDecodingData {
  // Decodes the Huffman code lengths from the bit-stream and fills in the
  // pre-allocated table with the corresponding 2-level Huffman decoding table.
  // Returns false if the Huffman code lengths can not be decoded.
  bool ReadFromBitStream(size_t alphabet_size, BitReader* br);

  // Decodes the next Huffman coded symbol from the bit-stream.
  JXL_INLINE uint16_t ReadSymbol(BitReader* br) const {
    size_t n_bits;
    const HuffmanCode* table = table_.data();
    table += br->PeekBits(kHuffmanTableBits);
    n_bits = table->bits;
    if (n_bits > kHuffmanTableBits) {
      br->Consume(kHuffmanTableBits);
      n_bits -= kHuffmanTableBits;
      table += table->value;
      table += br->PeekBits(n_bits);
    }
    br->Consume(table->bits);
    return table->value;
  }

  std::vector<HuffmanCode> table_;
};

}  // namespace jxl

#endif  // LIB_JXL_DEC_HUFFMAN_H_
