// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_huffman.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/enc_huffman_tree.h"

namespace jxl {

namespace {

constexpr int kCodeLengthCodes = 18;

void StoreHuffmanTreeOfHuffmanTreeToBitMask(const int num_codes,
                                            const uint8_t* code_length_bitdepth,
                                            BitWriter* writer) {
  static const uint8_t kStorageOrder[kCodeLengthCodes] = {
      1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  // The bit lengths of the Huffman code over the code length alphabet
  // are compressed with the following static Huffman code:
  //   Symbol   Code
  //   ------   ----
  //   0          00
  //   1        1110
  //   2         110
  //   3          01
  //   4          10
  //   5        1111
  static const uint8_t kHuffmanBitLengthHuffmanCodeSymbols[6] = {0, 7, 3,
                                                                 2, 1, 15};
  static const uint8_t kHuffmanBitLengthHuffmanCodeBitLengths[6] = {2, 4, 3,
                                                                    2, 2, 4};

  // Throw away trailing zeros:
  size_t codes_to_store = kCodeLengthCodes;
  if (num_codes > 1) {
    for (; codes_to_store > 0; --codes_to_store) {
      if (code_length_bitdepth[kStorageOrder[codes_to_store - 1]] != 0) {
        break;
      }
    }
  }
  size_t skip_some = 0;  // skips none.
  if (code_length_bitdepth[kStorageOrder[0]] == 0 &&
      code_length_bitdepth[kStorageOrder[1]] == 0) {
    skip_some = 2;  // skips two.
    if (code_length_bitdepth[kStorageOrder[2]] == 0) {
      skip_some = 3;  // skips three.
    }
  }
  writer->Write(2, skip_some);
  for (size_t i = skip_some; i < codes_to_store; ++i) {
    size_t l = code_length_bitdepth[kStorageOrder[i]];
    writer->Write(kHuffmanBitLengthHuffmanCodeBitLengths[l],
                  kHuffmanBitLengthHuffmanCodeSymbols[l]);
  }
}

Status StoreHuffmanTreeToBitMask(const size_t huffman_tree_size,
                                 const uint8_t* huffman_tree,
                                 const uint8_t* huffman_tree_extra_bits,
                                 const uint8_t* code_length_bitdepth,
                                 const uint16_t* code_length_bitdepth_symbols,
                                 BitWriter* writer) {
  for (size_t i = 0; i < huffman_tree_size; ++i) {
    size_t ix = huffman_tree[i];
    writer->Write(code_length_bitdepth[ix], code_length_bitdepth_symbols[ix]);
    JXL_ENSURE(ix <= 17);
    // Extra bits
    switch (ix) {
      case 16:
        writer->Write(2, huffman_tree_extra_bits[i]);
        break;
      case 17:
        writer->Write(3, huffman_tree_extra_bits[i]);
        break;
      default:
        // no-op
        break;
    }
  }
  return true;
}

void StoreSimpleHuffmanTree(const uint8_t* depths, size_t symbols[4],
                            size_t num_symbols, size_t max_bits,
                            BitWriter* writer) {
  // value of 1 indicates a simple Huffman code
  writer->Write(2, 1);
  writer->Write(2, num_symbols - 1);  // NSYM - 1

  // Sort
  for (size_t i = 0; i < num_symbols; i++) {
    for (size_t j = i + 1; j < num_symbols; j++) {
      if (depths[symbols[j]] < depths[symbols[i]]) {
        std::swap(symbols[j], symbols[i]);
      }
    }
  }

  if (num_symbols == 2) {
    writer->Write(max_bits, symbols[0]);
    writer->Write(max_bits, symbols[1]);
  } else if (num_symbols == 3) {
    writer->Write(max_bits, symbols[0]);
    writer->Write(max_bits, symbols[1]);
    writer->Write(max_bits, symbols[2]);
  } else {
    writer->Write(max_bits, symbols[0]);
    writer->Write(max_bits, symbols[1]);
    writer->Write(max_bits, symbols[2]);
    writer->Write(max_bits, symbols[3]);
    // tree-select
    writer->Write(1, depths[symbols[0]] == 1 ? 1 : 0);
  }
}

// num = alphabet size
// depths = symbol depths
Status StoreHuffmanTree(const uint8_t* depths, size_t num, BitWriter* writer) {
  // Write the Huffman tree into the compact representation.
  auto arena = jxl::make_uninitialized_vector<uint8_t>(2 * num);
  uint8_t* huffman_tree = arena.data();
  uint8_t* huffman_tree_extra_bits = arena.data() + num;
  size_t huffman_tree_size = 0;
  WriteHuffmanTree(depths, num, &huffman_tree_size, huffman_tree,
                   huffman_tree_extra_bits);

  // Calculate the statistics of the Huffman tree in the compact representation.
  uint32_t huffman_tree_histogram[kCodeLengthCodes] = {0};
  for (size_t i = 0; i < huffman_tree_size; ++i) {
    ++huffman_tree_histogram[huffman_tree[i]];
  }

  int num_codes = 0;
  int code = 0;
  for (int i = 0; i < kCodeLengthCodes; ++i) {
    if (huffman_tree_histogram[i]) {
      if (num_codes == 0) {
        code = i;
        num_codes = 1;
      } else if (num_codes == 1) {
        num_codes = 2;
        break;
      }
    }
  }

  // Calculate another Huffman tree to use for compressing both the
  // earlier Huffman tree with.
  uint8_t code_length_bitdepth[kCodeLengthCodes] = {0};
  uint16_t code_length_bitdepth_symbols[kCodeLengthCodes] = {0};
  CreateHuffmanTree(&huffman_tree_histogram[0], kCodeLengthCodes, 5,
                    &code_length_bitdepth[0]);
  ConvertBitDepthsToSymbols(code_length_bitdepth, kCodeLengthCodes,
                            &code_length_bitdepth_symbols[0]);

  // Now, we have all the data, let's start storing it
  StoreHuffmanTreeOfHuffmanTreeToBitMask(num_codes, code_length_bitdepth,
                                         writer);

  if (num_codes == 1) {
    code_length_bitdepth[code] = 0;
  }

  // Store the real huffman tree now.
  JXL_RETURN_IF_ERROR(StoreHuffmanTreeToBitMask(
      huffman_tree_size, huffman_tree, huffman_tree_extra_bits,
      &code_length_bitdepth[0], code_length_bitdepth_symbols, writer));
  return true;
}

}  // namespace

Status BuildAndStoreHuffmanTree(const uint32_t* histogram, const size_t length,
                                uint8_t* depth, uint16_t* bits,
                                BitWriter* writer) {
  size_t count = 0;
  size_t s4[4] = {0};
  for (size_t i = 0; i < length; i++) {
    if (histogram[i]) {
      if (count < 4) {
        s4[count] = i;
      } else if (count > 4) {
        break;
      }
      count++;
    }
  }

  size_t max_bits_counter = length - 1;
  size_t max_bits = 0;
  while (max_bits_counter) {
    max_bits_counter >>= 1;
    ++max_bits;
  }

  if (count <= 1) {
    // Output symbol bits and depths are initialized with 0, nothing to do.
    writer->Write(4, 1);
    writer->Write(max_bits, s4[0]);
    return true;
  }

  CreateHuffmanTree(histogram, length, 15, depth);
  ConvertBitDepthsToSymbols(depth, length, bits);

  if (count <= 4) {
    StoreSimpleHuffmanTree(depth, s4, count, max_bits, writer);
  } else {
    JXL_RETURN_IF_ERROR(StoreHuffmanTree(depth, length, writer));
  }
  return true;
}

}  // namespace jxl
