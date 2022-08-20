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

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "src/utils/huffman_utils.h"
#include "src/utils/utils.h"
#include "src/webp/format_constants.h"

// Huffman data read via DecodeImageStream is represented in two (red and green)
// bytes.
#define MAX_HTREE_GROUPS    0x10000

HTreeGroup* VP8LHtreeGroupsNew(int num_htree_groups) {
  HTreeGroup* const htree_groups =
      (HTreeGroup*)WebPSafeMalloc(num_htree_groups, sizeof(*htree_groups));
  if (htree_groups == NULL) {
    return NULL;
  }
  assert(num_htree_groups <= MAX_HTREE_GROUPS);
  return htree_groups;
}

void VP8LHtreeGroupsFree(HTreeGroup* const htree_groups) {
  if (htree_groups != NULL) {
    WebPSafeFree(htree_groups);
  }
}

// Returns reverse(reverse(key, len) + 1, len), where reverse(key, len) is the
// bit-wise reversal of the len least significant bits of key.
static WEBP_INLINE uint32_t GetNextKey(uint32_t key, int len) {
  uint32_t step = 1 << (len - 1);
  while (key & step) {
    step >>= 1;
  }
  return step ? (key & (step - 1)) + step : key;
}

// Stores code in table[0], table[step], table[2*step], ..., table[end].
// Assumes that end is an integer multiple of step.
static WEBP_INLINE void ReplicateValue(HuffmanCode* table,
                                       int step, int end,
                                       HuffmanCode code) {
  assert(end % step == 0);
  do {
    end -= step;
    table[end] = code;
  } while (end > 0);
}

// Returns the table width of the next 2nd level table. count is the histogram
// of bit lengths for the remaining symbols, len is the code length of the next
// processed symbol
static WEBP_INLINE int NextTableBitSize(const int* const count,
                                        int len, int root_bits) {
  int left = 1 << (len - root_bits);
  while (len < MAX_ALLOWED_CODE_LENGTH) {
    left -= count[len];
    if (left <= 0) break;
    ++len;
    left <<= 1;
  }
  return len - root_bits;
}

// sorted[code_lengths_size] is a pre-allocated array for sorting symbols
// by code length.
static int BuildHuffmanTable(HuffmanCode* const root_table, int root_bits,
                             const int code_lengths[], int code_lengths_size,
                             uint16_t sorted[]) {
  HuffmanCode* table = root_table;  // next available space in table
  int total_size = 1 << root_bits;  // total size root table + 2nd level table
  int len;                          // current code length
  int symbol;                       // symbol index in original or sorted table
  // number of codes of each length:
  int count[MAX_ALLOWED_CODE_LENGTH + 1] = { 0 };
  // offsets in sorted table for each length:
  int offset[MAX_ALLOWED_CODE_LENGTH + 1];

  assert(code_lengths_size != 0);
  assert(code_lengths != NULL);
  assert((root_table != NULL && sorted != NULL) ||
         (root_table == NULL && sorted == NULL));
  assert(root_bits > 0);

  // Build histogram of code lengths.
  for (symbol = 0; symbol < code_lengths_size; ++symbol) {
    if (code_lengths[symbol] > MAX_ALLOWED_CODE_LENGTH) {
      return 0;
    }
    ++count[code_lengths[symbol]];
  }

  // Error, all code lengths are zeros.
  if (count[0] == code_lengths_size) {
    return 0;
  }

  // Generate offsets into sorted symbol table by code length.
  offset[1] = 0;
  for (len = 1; len < MAX_ALLOWED_CODE_LENGTH; ++len) {
    if (count[len] > (1 << len)) {
      return 0;
    }
    offset[len + 1] = offset[len] + count[len];
  }

  // Sort symbols by length, by symbol order within each length.
  for (symbol = 0; symbol < code_lengths_size; ++symbol) {
    const int symbol_code_length = code_lengths[symbol];
    if (code_lengths[symbol] > 0) {
      if (sorted != NULL) {
        sorted[offset[symbol_code_length]++] = symbol;
      } else {
        offset[symbol_code_length]++;
      }
    }
  }

  // Special case code with only one value.
  if (offset[MAX_ALLOWED_CODE_LENGTH] == 1) {
    if (sorted != NULL) {
      HuffmanCode code;
      code.bits = 0;
      code.value = (uint16_t)sorted[0];
      ReplicateValue(table, 1, total_size, code);
    }
    return total_size;
  }

  {
    int step;              // step size to replicate values in current table
    uint32_t low = -1;     // low bits for current root entry
    uint32_t mask = total_size - 1;    // mask for low bits
    uint32_t key = 0;      // reversed prefix code
    int num_nodes = 1;     // number of Huffman tree nodes
    int num_open = 1;      // number of open branches in current tree level
    int table_bits = root_bits;        // key length of current table
    int table_size = 1 << table_bits;  // size of current table
    symbol = 0;
    // Fill in root table.
    for (len = 1, step = 2; len <= root_bits; ++len, step <<= 1) {
      num_open <<= 1;
      num_nodes += num_open;
      num_open -= count[len];
      if (num_open < 0) {
        return 0;
      }
      if (root_table == NULL) continue;
      for (; count[len] > 0; --count[len]) {
        HuffmanCode code;
        code.bits = (uint8_t)len;
        code.value = (uint16_t)sorted[symbol++];
        ReplicateValue(&table[key], step, table_size, code);
        key = GetNextKey(key, len);
      }
    }

    // Fill in 2nd level tables and add pointers to root table.
    for (len = root_bits + 1, step = 2; len <= MAX_ALLOWED_CODE_LENGTH;
         ++len, step <<= 1) {
      num_open <<= 1;
      num_nodes += num_open;
      num_open -= count[len];
      if (num_open < 0) {
        return 0;
      }
      if (root_table == NULL) continue;
      for (; count[len] > 0; --count[len]) {
        HuffmanCode code;
        if ((key & mask) != low) {
          table += table_size;
          table_bits = NextTableBitSize(count, len, root_bits);
          table_size = 1 << table_bits;
          total_size += table_size;
          low = key & mask;
          root_table[low].bits = (uint8_t)(table_bits + root_bits);
          root_table[low].value = (uint16_t)((table - root_table) - low);
        }
        code.bits = (uint8_t)(len - root_bits);
        code.value = (uint16_t)sorted[symbol++];
        ReplicateValue(&table[key >> root_bits], step, table_size, code);
        key = GetNextKey(key, len);
      }
    }

    // Check if tree is full.
    if (num_nodes != 2 * offset[MAX_ALLOWED_CODE_LENGTH] - 1) {
      return 0;
    }
  }

  return total_size;
}

// Maximum code_lengths_size is 2328 (reached for 11-bit color_cache_bits).
// More commonly, the value is around ~280.
#define MAX_CODE_LENGTHS_SIZE \
  ((1 << MAX_CACHE_BITS) + NUM_LITERAL_CODES + NUM_LENGTH_CODES)
// Cut-off value for switching between heap and stack allocation.
#define SORTED_SIZE_CUTOFF 512
int VP8LBuildHuffmanTable(HuffmanCode* const root_table, int root_bits,
                          const int code_lengths[], int code_lengths_size) {
  int total_size;
  assert(code_lengths_size <= MAX_CODE_LENGTHS_SIZE);
  if (root_table == NULL) {
    total_size = BuildHuffmanTable(NULL, root_bits,
                                   code_lengths, code_lengths_size, NULL);
  } else if (code_lengths_size <= SORTED_SIZE_CUTOFF) {
    // use local stack-allocated array.
    uint16_t sorted[SORTED_SIZE_CUTOFF];
    total_size = BuildHuffmanTable(root_table, root_bits,
                                   code_lengths, code_lengths_size, sorted);
  } else {   // rare case. Use heap allocation.
    uint16_t* const sorted =
        (uint16_t*)WebPSafeMalloc(code_lengths_size, sizeof(*sorted));
    if (sorted == NULL) return 0;
    total_size = BuildHuffmanTable(root_table, root_bits,
                                   code_lengths, code_lengths_size, sorted);
    WebPSafeFree(sorted);
  }
  return total_size;
}
