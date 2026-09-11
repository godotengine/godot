// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_DEC_HUFFMAN_H_
#define LIB_JXL_DEC_HUFFMAN_H_

#include <memory>
#include <vector>

#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/huffman_table.h"

namespace jxl {

static constexpr size_t kHuffmanTableBits = 8u;

struct HuffmanDecodingData {
  // Decodes the Huffman code lengths from the bit-stream and fills in the
  // pre-allocated table with the corresponding 2-level Huffman decoding table.
  // Returns false if the Huffman code lengths can not de decoded.
  bool ReadFromBitStream(size_t alphabet_size, BitReader* br);

  uint16_t ReadSymbol(BitReader* br) const;

  std::vector<HuffmanCode> table_;
};

}  // namespace jxl

#endif  // LIB_JXL_DEC_HUFFMAN_H_
