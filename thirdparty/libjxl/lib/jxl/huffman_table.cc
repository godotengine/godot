// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/huffman_table.h"

#include <cstring> /* for memcpy */
#include <vector>

#include "lib/jxl/ans_params.h"
#include "lib/jxl/dec_huffman.h"

namespace jxl {

/* Returns reverse(reverse(key, len) + 1, len), where reverse(key, len) is the
   bit-wise reversal of the len least significant bits of key. */
static inline int GetNextKey(int key, int len) {
  int step = 1u << (len - 1);
  while (key & step) {
    step >>= 1;
  }
  return (key & (step - 1)) + step;
}

/* Stores code in table[0], table[step], table[2*step], ..., table[end] */
/* Assumes that end is an integer multiple of step */
static inline void ReplicateValue(HuffmanCode* table, int step, int end,
                                  HuffmanCode code) {
  do {
    end -= step;
    table[end] = code;
  } while (end > 0);
}

/* Returns the table width of the next 2nd level table. count is the histogram
   of bit lengths for the remaining symbols, len is the code length of the next
   processed symbol */
static inline size_t NextTableBitSize(const uint16_t* const count, size_t len,
                                      int root_bits) {
  size_t left = 1u << (len - root_bits);
  while (len < PREFIX_MAX_BITS) {
    if (left <= count[len]) break;
    left -= count[len];
    ++len;
    left <<= 1;
  }
  return len - root_bits;
}

uint32_t BuildHuffmanTable(HuffmanCode* root_table, int root_bits,
                           const uint8_t* const code_lengths,
                           size_t code_lengths_size, uint16_t* count) {
  HuffmanCode code;   /* current table entry */
  HuffmanCode* table; /* next available space in table */
  size_t len;         /* current code length */
  size_t symbol;      /* symbol index in original or sorted table */
  int key;            /* reversed prefix code */
  int step;           /* step size to replicate values in current table */
  int low;            /* low bits for current root entry */
  int mask;           /* mask for low bits */
  size_t table_bits;  /* key length of current table */
  int table_size;     /* size of current table */
  int total_size;     /* sum of root table size and 2nd level table sizes */
  /* offsets in sorted table for each length */
  uint16_t offset[PREFIX_MAX_BITS + 1];
  size_t max_length = 1;

  if (code_lengths_size > 1u << PREFIX_MAX_BITS) return 0;

  /* symbols sorted by code length */
  std::vector<uint16_t> sorted_storage(code_lengths_size);
  uint16_t* sorted = sorted_storage.data();

  /* generate offsets into sorted symbol table by code length */
  {
    uint16_t sum = 0;
    for (len = 1; len <= PREFIX_MAX_BITS; len++) {
      offset[len] = sum;
      if (count[len]) {
        sum = static_cast<uint16_t>(sum + count[len]);
        max_length = len;
      }
    }
  }

  /* sort symbols by length, by symbol order within each length */
  for (symbol = 0; symbol < code_lengths_size; symbol++) {
    if (code_lengths[symbol] != 0) {
      sorted[offset[code_lengths[symbol]]++] = symbol;
    }
  }

  table = root_table;
  table_bits = root_bits;
  table_size = 1u << table_bits;
  total_size = table_size;

  /* special case code with only one value */
  if (offset[PREFIX_MAX_BITS] == 1) {
    code.bits = 0;
    code.value = static_cast<uint16_t>(sorted[0]);
    for (key = 0; key < total_size; ++key) {
      table[key] = code;
    }
    return total_size;
  }

  /* fill in root table */
  /* let's reduce the table size to a smaller size if possible, and */
  /* create the repetitions by memcpy if possible in the coming loop */
  if (table_bits > max_length) {
    table_bits = max_length;
    table_size = 1u << table_bits;
  }
  key = 0;
  symbol = 0;
  code.bits = 1;
  step = 2;
  do {
    for (; count[code.bits] != 0; --count[code.bits]) {
      code.value = static_cast<uint16_t>(sorted[symbol++]);
      ReplicateValue(&table[key], step, table_size, code);
      key = GetNextKey(key, code.bits);
    }
    step <<= 1;
  } while (++code.bits <= table_bits);

  /* if root_bits != table_bits we only created one fraction of the */
  /* table, and we need to replicate it now. */
  while (total_size != table_size) {
    memcpy(&table[table_size], &table[0], table_size * sizeof(table[0]));
    table_size <<= 1;
  }

  /* fill in 2nd level tables and add pointers to root table */
  mask = total_size - 1;
  low = -1;
  for (len = root_bits + 1, step = 2; len <= max_length; ++len, step <<= 1) {
    for (; count[len] != 0; --count[len]) {
      if ((key & mask) != low) {
        table += table_size;
        table_bits = NextTableBitSize(count, len, root_bits);
        table_size = 1u << table_bits;
        total_size += table_size;
        low = key & mask;
        root_table[low].bits = static_cast<uint8_t>(table_bits + root_bits);
        root_table[low].value =
            static_cast<uint16_t>((table - root_table) - low);
      }
      code.bits = static_cast<uint8_t>(len - root_bits);
      code.value = static_cast<uint16_t>(sorted[symbol++]);
      ReplicateValue(&table[key >> root_bits], step, table_size, code);
      key = GetNextKey(key, len);
    }
  }

  return total_size;
}

}  // namespace jxl
