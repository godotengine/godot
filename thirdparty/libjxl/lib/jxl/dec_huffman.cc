// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/dec_huffman.h"

#include <jxl/types.h>
#include <string.h> /* for memset */

#include <vector>

#include "lib/jxl/ans_params.h"
#include "lib/jxl/base/bits.h"
#include "lib/jxl/huffman_table.h"

namespace jxl {

static const int kCodeLengthCodes = 18;
static const uint8_t kCodeLengthCodeOrder[kCodeLengthCodes] = {
    1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15,
};
static const uint8_t kDefaultCodeLength = 8;
static const uint8_t kCodeLengthRepeatCode = 16;

JXL_BOOL ReadHuffmanCodeLengths(const uint8_t* code_length_code_lengths,
                                int num_symbols, uint8_t* code_lengths,
                                BitReader* br) {
  int symbol = 0;
  uint8_t prev_code_len = kDefaultCodeLength;
  int repeat = 0;
  uint8_t repeat_code_len = 0;
  int space = 32768;
  HuffmanCode table[32];

  uint16_t counts[16] = {0};
  for (int i = 0; i < kCodeLengthCodes; ++i) {
    ++counts[code_length_code_lengths[i]];
  }
  if (!BuildHuffmanTable(table, 5, code_length_code_lengths, kCodeLengthCodes,
                         &counts[0])) {
    return JXL_FALSE;
  }

  while (symbol < num_symbols && space > 0) {
    const HuffmanCode* p = table;
    uint8_t code_len;
    br->Refill();
    p += br->PeekFixedBits<5>();
    br->Consume(p->bits);
    code_len = static_cast<uint8_t>(p->value);
    if (code_len < kCodeLengthRepeatCode) {
      repeat = 0;
      code_lengths[symbol++] = code_len;
      if (code_len != 0) {
        prev_code_len = code_len;
        space -= 32768u >> code_len;
      }
    } else {
      const int extra_bits = code_len - 14;
      int old_repeat;
      int repeat_delta;
      uint8_t new_len = 0;
      if (code_len == kCodeLengthRepeatCode) {
        new_len = prev_code_len;
      }
      if (repeat_code_len != new_len) {
        repeat = 0;
        repeat_code_len = new_len;
      }
      old_repeat = repeat;
      if (repeat > 0) {
        repeat -= 2;
        repeat <<= extra_bits;
      }
      repeat += static_cast<int>(br->ReadBits(extra_bits) + 3);
      repeat_delta = repeat - old_repeat;
      if (symbol + repeat_delta > num_symbols) {
        return 0;
      }
      memset(&code_lengths[symbol], repeat_code_len,
             static_cast<size_t>(repeat_delta));
      symbol += repeat_delta;
      if (repeat_code_len != 0) {
        space -= repeat_delta << (15 - repeat_code_len);
      }
    }
  }
  if (space != 0) {
    return JXL_FALSE;
  }
  memset(&code_lengths[symbol], 0, static_cast<size_t>(num_symbols - symbol));
  return JXL_TRUE;
}

static JXL_INLINE bool ReadSimpleCode(size_t alphabet_size, BitReader* br,
                                      HuffmanCode* table) {
  size_t max_bits =
      (alphabet_size > 1u) ? FloorLog2Nonzero(alphabet_size - 1u) + 1 : 0;

  size_t num_symbols = br->ReadFixedBits<2>() + 1;

  uint16_t symbols[4] = {0};
  for (size_t i = 0; i < num_symbols; ++i) {
    uint16_t symbol = br->ReadBits(max_bits);
    if (symbol >= alphabet_size) {
      return false;
    }
    symbols[i] = symbol;
  }

  for (size_t i = 0; i < num_symbols - 1; ++i) {
    for (size_t j = i + 1; j < num_symbols; ++j) {
      if (symbols[i] == symbols[j]) return false;
    }
  }

  // 4 symbols have to option to encode.
  if (num_symbols == 4) num_symbols += br->ReadFixedBits<1>();

  const auto swap_symbols = [&symbols](size_t i, size_t j) {
    uint16_t t = symbols[j];
    symbols[j] = symbols[i];
    symbols[i] = t;
  };

  size_t table_size = 1;
  switch (num_symbols) {
    case 1:
      table[0] = {0, symbols[0]};
      break;
    case 2:
      if (symbols[0] > symbols[1]) swap_symbols(0, 1);
      table[0] = {1, symbols[0]};
      table[1] = {1, symbols[1]};
      table_size = 2;
      break;
    case 3:
      if (symbols[1] > symbols[2]) swap_symbols(1, 2);
      table[0] = {1, symbols[0]};
      table[2] = {1, symbols[0]};
      table[1] = {2, symbols[1]};
      table[3] = {2, symbols[2]};
      table_size = 4;
      break;
    case 4: {
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = i + 1; j < 4; ++j) {
          if (symbols[i] > symbols[j]) swap_symbols(i, j);
        }
      }
      table[0] = {2, symbols[0]};
      table[2] = {2, symbols[1]};
      table[1] = {2, symbols[2]};
      table[3] = {2, symbols[3]};
      table_size = 4;
      break;
    }
    case 5: {
      if (symbols[2] > symbols[3]) swap_symbols(2, 3);
      table[0] = {1, symbols[0]};
      table[1] = {2, symbols[1]};
      table[2] = {1, symbols[0]};
      table[3] = {3, symbols[2]};
      table[4] = {1, symbols[0]};
      table[5] = {2, symbols[1]};
      table[6] = {1, symbols[0]};
      table[7] = {3, symbols[3]};
      table_size = 8;
      break;
    }
    default: {
      // Unreachable.
      return false;
    }
  }

  const uint32_t goal_size = 1u << kHuffmanTableBits;
  while (table_size != goal_size) {
    memcpy(&table[table_size], &table[0],
           static_cast<size_t>(table_size) * sizeof(table[0]));
    table_size <<= 1;
  }

  return true;
}

bool HuffmanDecodingData::ReadFromBitStream(size_t alphabet_size,
                                            BitReader* br) {
  if (alphabet_size > (1 << PREFIX_MAX_BITS)) return false;

  /* simple_code_or_skip is used as follows:
     1 for simple code;
     0 for no skipping, 2 skips 2 code lengths, 3 skips 3 code lengths */
  uint32_t simple_code_or_skip = br->ReadFixedBits<2>();
  if (simple_code_or_skip == 1u) {
    table_.resize(1u << kHuffmanTableBits);
    return ReadSimpleCode(alphabet_size, br, table_.data());
  }

  std::vector<uint8_t> code_lengths(alphabet_size, 0);
  uint8_t code_length_code_lengths[kCodeLengthCodes] = {0};
  int space = 32;
  int num_codes = 0;
  /* Static Huffman code for the code length code lengths */
  static const HuffmanCode huff[16] = {
      {2, 0}, {2, 4}, {2, 3}, {3, 2}, {2, 0}, {2, 4}, {2, 3}, {4, 1},
      {2, 0}, {2, 4}, {2, 3}, {3, 2}, {2, 0}, {2, 4}, {2, 3}, {4, 5},
  };
  for (size_t i = simple_code_or_skip; i < kCodeLengthCodes && space > 0; ++i) {
    const int code_len_idx = kCodeLengthCodeOrder[i];
    const HuffmanCode* p = huff;
    uint8_t v;
    br->Refill();
    p += br->PeekFixedBits<4>();
    br->Consume(p->bits);
    v = static_cast<uint8_t>(p->value);
    code_length_code_lengths[code_len_idx] = v;
    if (v != 0) {
      space -= (32u >> v);
      ++num_codes;
    }
  }
  bool ok =
      (num_codes == 1 || space == 0) &&
      FROM_JXL_BOOL(ReadHuffmanCodeLengths(
          code_length_code_lengths, alphabet_size, code_lengths.data(), br));

  if (!ok) return false;
  uint16_t counts[16] = {0};
  for (size_t i = 0; i < alphabet_size; ++i) {
    ++counts[code_lengths[i]];
  }
  table_.resize(alphabet_size + 376);
  uint32_t table_size =
      BuildHuffmanTable(table_.data(), kHuffmanTableBits, code_lengths.data(),
                        alphabet_size, &counts[0]);
  table_.resize(table_size);
  return (table_size > 0);
}

// Decodes the next Huffman coded symbol from the bit-stream.
uint16_t HuffmanDecodingData::ReadSymbol(BitReader* br) const {
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

}  // namespace jxl
