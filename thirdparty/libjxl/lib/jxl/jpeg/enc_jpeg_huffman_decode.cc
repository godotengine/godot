// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/jpeg/enc_jpeg_huffman_decode.h"

#include <cstdint>

#include "lib/jxl/jpeg/jpeg_data.h"

namespace jxl {
namespace jpeg {

// Returns the table width of the next 2nd level table, count is the histogram
// of bit lengths for the remaining symbols, len is the code length of the next
// processed symbol.
static inline int NextTableBitSize(const int* count, int len) {
  int left = 1 << (len - kJpegHuffmanRootTableBits);
  while (len < static_cast<int>(kJpegHuffmanMaxBitLength)) {
    left -= count[len];
    if (left <= 0) break;
    ++len;
    left <<= 1;
  }
  return len - kJpegHuffmanRootTableBits;
}

void BuildJpegHuffmanTable(const uint32_t* count, const uint32_t* symbols,
                           HuffmanTableEntry* lut) {
  HuffmanTableEntry code;    // current table entry
  HuffmanTableEntry* table;  // next available space in table
  int len;                   // current code length
  int idx;                   // symbol index
  int key;                   // prefix code
  int reps;                  // number of replicate key values in current table
  int low;                   // low bits for current root entry
  int table_bits;            // key length of current table
  int table_size;            // size of current table

  // Make a local copy of the input bit length histogram.
  int tmp_count[kJpegHuffmanMaxBitLength + 1] = {0};
  int total_count = 0;
  for (len = 1; len <= static_cast<int>(kJpegHuffmanMaxBitLength); ++len) {
    tmp_count[len] = count[len];
    total_count += tmp_count[len];
  }

  table = lut;
  table_bits = kJpegHuffmanRootTableBits;
  table_size = 1 << table_bits;

  // Special case code with only one value.
  if (total_count == 1) {
    code.bits = 0;
    code.value = symbols[0];
    for (key = 0; key < table_size; ++key) {
      table[key] = code;
    }
    return;
  }

  // Fill in root table.
  key = 0;
  idx = 0;
  for (len = 1; len <= kJpegHuffmanRootTableBits; ++len) {
    for (; tmp_count[len] > 0; --tmp_count[len]) {
      code.bits = len;
      code.value = symbols[idx++];
      reps = 1 << (kJpegHuffmanRootTableBits - len);
      while (reps--) {
        table[key++] = code;
      }
    }
  }

  // Fill in 2nd level tables and add pointers to root table.
  table += table_size;
  table_size = 0;
  low = 0;
  for (len = kJpegHuffmanRootTableBits + 1;
       len <= static_cast<int>(kJpegHuffmanMaxBitLength); ++len) {
    for (; tmp_count[len] > 0; --tmp_count[len]) {
      // Start a new sub-table if the previous one is full.
      if (low >= table_size) {
        table += table_size;
        table_bits = NextTableBitSize(tmp_count, len);
        table_size = 1 << table_bits;
        low = 0;
        lut[key].bits = table_bits + kJpegHuffmanRootTableBits;
        lut[key].value = (table - lut) - key;
        ++key;
      }
      code.bits = len - kJpegHuffmanRootTableBits;
      code.value = symbols[idx++];
      reps = 1 << (table_bits - code.bits);
      while (reps--) {
        table[low++] = code;
      }
    }
  }
}

}  // namespace jpeg
}  // namespace jxl
