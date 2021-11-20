// Copyright 2011 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Author: Jyrki Alakuijala (jyrki@google.com)
//
// Entropy encoding (Huffman) for webp lossless.

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "src/utils/huffman_encode_utils.h"
#include "src/utils/utils.h"
#include "src/webp/format_constants.h"

// -----------------------------------------------------------------------------
// Util function to optimize the symbol map for RLE coding

// Heuristics for selecting the stride ranges to collapse.
static int ValuesShouldBeCollapsedToStrideAverage(int a, int b) {
  return abs(a - b) < 4;
}

// Change the population counts in a way that the consequent
// Huffman tree compression, especially its RLE-part, give smaller output.
static void OptimizeHuffmanForRle(int length, uint8_t* const good_for_rle,
                                  uint32_t* const counts) {
  // 1) Let's make the Huffman code more compatible with rle encoding.
  int i;
  for (; length >= 0; --length) {
    if (length == 0) {
      return;  // All zeros.
    }
    if (counts[length - 1] != 0) {
      // Now counts[0..length - 1] does not have trailing zeros.
      break;
    }
  }
  // 2) Let's mark all population counts that already can be encoded
  // with an rle code.
  {
    // Let's not spoil any of the existing good rle codes.
    // Mark any seq of 0's that is longer as 5 as a good_for_rle.
    // Mark any seq of non-0's that is longer as 7 as a good_for_rle.
    uint32_t symbol = counts[0];
    int stride = 0;
    for (i = 0; i < length + 1; ++i) {
      if (i == length || counts[i] != symbol) {
        if ((symbol == 0 && stride >= 5) ||
            (symbol != 0 && stride >= 7)) {
          int k;
          for (k = 0; k < stride; ++k) {
            good_for_rle[i - k - 1] = 1;
          }
        }
        stride = 1;
        if (i != length) {
          symbol = counts[i];
        }
      } else {
        ++stride;
      }
    }
  }
  // 3) Let's replace those population counts that lead to more rle codes.
  {
    uint32_t stride = 0;
    uint32_t limit = counts[0];
    uint32_t sum = 0;
    for (i = 0; i < length + 1; ++i) {
      if (i == length || good_for_rle[i] ||
          (i != 0 && good_for_rle[i - 1]) ||
          !ValuesShouldBeCollapsedToStrideAverage(counts[i], limit)) {
        if (stride >= 4 || (stride >= 3 && sum == 0)) {
          uint32_t k;
          // The stride must end, collapse what we have, if we have enough (4).
          uint32_t count = (sum + stride / 2) / stride;
          if (count < 1) {
            count = 1;
          }
          if (sum == 0) {
            // Don't make an all zeros stride to be upgraded to ones.
            count = 0;
          }
          for (k = 0; k < stride; ++k) {
            // We don't want to change value at counts[i],
            // that is already belonging to the next stride. Thus - 1.
            counts[i - k - 1] = count;
          }
        }
        stride = 0;
        sum = 0;
        if (i < length - 3) {
          // All interesting strides have a count of at least 4,
          // at least when non-zeros.
          limit = (counts[i] + counts[i + 1] +
                   counts[i + 2] + counts[i + 3] + 2) / 4;
        } else if (i < length) {
          limit = counts[i];
        } else {
          limit = 0;
        }
      }
      ++stride;
      if (i != length) {
        sum += counts[i];
        if (stride >= 4) {
          limit = (sum + stride / 2) / stride;
        }
      }
    }
  }
}

// A comparer function for two Huffman trees: sorts first by 'total count'
// (more comes first), and then by 'value' (more comes first).
static int CompareHuffmanTrees(const void* ptr1, const void* ptr2) {
  const HuffmanTree* const t1 = (const HuffmanTree*)ptr1;
  const HuffmanTree* const t2 = (const HuffmanTree*)ptr2;
  if (t1->total_count_ > t2->total_count_) {
    return -1;
  } else if (t1->total_count_ < t2->total_count_) {
    return 1;
  } else {
    assert(t1->value_ != t2->value_);
    return (t1->value_ < t2->value_) ? -1 : 1;
  }
}

static void SetBitDepths(const HuffmanTree* const tree,
                         const HuffmanTree* const pool,
                         uint8_t* const bit_depths, int level) {
  if (tree->pool_index_left_ >= 0) {
    SetBitDepths(&pool[tree->pool_index_left_], pool, bit_depths, level + 1);
    SetBitDepths(&pool[tree->pool_index_right_], pool, bit_depths, level + 1);
  } else {
    bit_depths[tree->value_] = level;
  }
}

// Create an optimal Huffman tree.
//
// (data,length): population counts.
// tree_limit: maximum bit depth (inclusive) of the codes.
// bit_depths[]: how many bits are used for the symbol.
//
// Returns 0 when an error has occurred.
//
// The catch here is that the tree cannot be arbitrarily deep
//
// count_limit is the value that is to be faked as the minimum value
// and this minimum value is raised until the tree matches the
// maximum length requirement.
//
// This algorithm is not of excellent performance for very long data blocks,
// especially when population counts are longer than 2**tree_limit, but
// we are not planning to use this with extremely long blocks.
//
// See http://en.wikipedia.org/wiki/Huffman_coding
static void GenerateOptimalTree(const uint32_t* const histogram,
                                int histogram_size,
                                HuffmanTree* tree, int tree_depth_limit,
                                uint8_t* const bit_depths) {
  uint32_t count_min;
  HuffmanTree* tree_pool;
  int tree_size_orig = 0;
  int i;

  for (i = 0; i < histogram_size; ++i) {
    if (histogram[i] != 0) {
      ++tree_size_orig;
    }
  }

  if (tree_size_orig == 0) {   // pretty optimal already!
    return;
  }

  tree_pool = tree + tree_size_orig;

  // For block sizes with less than 64k symbols we never need to do a
  // second iteration of this loop.
  // If we actually start running inside this loop a lot, we would perhaps
  // be better off with the Katajainen algorithm.
  assert(tree_size_orig <= (1 << (tree_depth_limit - 1)));
  for (count_min = 1; ; count_min *= 2) {
    int tree_size = tree_size_orig;
    // We need to pack the Huffman tree in tree_depth_limit bits.
    // So, we try by faking histogram entries to be at least 'count_min'.
    int idx = 0;
    int j;
    for (j = 0; j < histogram_size; ++j) {
      if (histogram[j] != 0) {
        const uint32_t count =
            (histogram[j] < count_min) ? count_min : histogram[j];
        tree[idx].total_count_ = count;
        tree[idx].value_ = j;
        tree[idx].pool_index_left_ = -1;
        tree[idx].pool_index_right_ = -1;
        ++idx;
      }
    }

    // Build the Huffman tree.
    qsort(tree, tree_size, sizeof(*tree), CompareHuffmanTrees);

    if (tree_size > 1) {  // Normal case.
      int tree_pool_size = 0;
      while (tree_size > 1) {  // Finish when we have only one root.
        uint32_t count;
        tree_pool[tree_pool_size++] = tree[tree_size - 1];
        tree_pool[tree_pool_size++] = tree[tree_size - 2];
        count = tree_pool[tree_pool_size - 1].total_count_ +
                tree_pool[tree_pool_size - 2].total_count_;
        tree_size -= 2;
        {
          // Search for the insertion point.
          int k;
          for (k = 0; k < tree_size; ++k) {
            if (tree[k].total_count_ <= count) {
              break;
            }
          }
          memmove(tree + (k + 1), tree + k, (tree_size - k) * sizeof(*tree));
          tree[k].total_count_ = count;
          tree[k].value_ = -1;

          tree[k].pool_index_left_ = tree_pool_size - 1;
          tree[k].pool_index_right_ = tree_pool_size - 2;
          tree_size = tree_size + 1;
        }
      }
      SetBitDepths(&tree[0], tree_pool, bit_depths, 0);
    } else if (tree_size == 1) {  // Trivial case: only one element.
      bit_depths[tree[0].value_] = 1;
    }

    {
      // Test if this Huffman tree satisfies our 'tree_depth_limit' criteria.
      int max_depth = bit_depths[0];
      for (j = 1; j < histogram_size; ++j) {
        if (max_depth < bit_depths[j]) {
          max_depth = bit_depths[j];
        }
      }
      if (max_depth <= tree_depth_limit) {
        break;
      }
    }
  }
}

// -----------------------------------------------------------------------------
// Coding of the Huffman tree values

static HuffmanTreeToken* CodeRepeatedValues(int repetitions,
                                            HuffmanTreeToken* tokens,
                                            int value, int prev_value) {
  assert(value <= MAX_ALLOWED_CODE_LENGTH);
  if (value != prev_value) {
    tokens->code = value;
    tokens->extra_bits = 0;
    ++tokens;
    --repetitions;
  }
  while (repetitions >= 1) {
    if (repetitions < 3) {
      int i;
      for (i = 0; i < repetitions; ++i) {
        tokens->code = value;
        tokens->extra_bits = 0;
        ++tokens;
      }
      break;
    } else if (repetitions < 7) {
      tokens->code = 16;
      tokens->extra_bits = repetitions - 3;
      ++tokens;
      break;
    } else {
      tokens->code = 16;
      tokens->extra_bits = 3;
      ++tokens;
      repetitions -= 6;
    }
  }
  return tokens;
}

static HuffmanTreeToken* CodeRepeatedZeros(int repetitions,
                                           HuffmanTreeToken* tokens) {
  while (repetitions >= 1) {
    if (repetitions < 3) {
      int i;
      for (i = 0; i < repetitions; ++i) {
        tokens->code = 0;   // 0-value
        tokens->extra_bits = 0;
        ++tokens;
      }
      break;
    } else if (repetitions < 11) {
      tokens->code = 17;
      tokens->extra_bits = repetitions - 3;
      ++tokens;
      break;
    } else if (repetitions < 139) {
      tokens->code = 18;
      tokens->extra_bits = repetitions - 11;
      ++tokens;
      break;
    } else {
      tokens->code = 18;
      tokens->extra_bits = 0x7f;  // 138 repeated 0s
      ++tokens;
      repetitions -= 138;
    }
  }
  return tokens;
}

int VP8LCreateCompressedHuffmanTree(const HuffmanTreeCode* const tree,
                                    HuffmanTreeToken* tokens, int max_tokens) {
  HuffmanTreeToken* const starting_token = tokens;
  HuffmanTreeToken* const ending_token = tokens + max_tokens;
  const int depth_size = tree->num_symbols;
  int prev_value = 8;  // 8 is the initial value for rle.
  int i = 0;
  assert(tokens != NULL);
  while (i < depth_size) {
    const int value = tree->code_lengths[i];
    int k = i + 1;
    int runs;
    while (k < depth_size && tree->code_lengths[k] == value) ++k;
    runs = k - i;
    if (value == 0) {
      tokens = CodeRepeatedZeros(runs, tokens);
    } else {
      tokens = CodeRepeatedValues(runs, tokens, value, prev_value);
      prev_value = value;
    }
    i += runs;
    assert(tokens <= ending_token);
  }
  (void)ending_token;    // suppress 'unused variable' warning
  return (int)(tokens - starting_token);
}

// -----------------------------------------------------------------------------

// Pre-reversed 4-bit values.
static const uint8_t kReversedBits[16] = {
  0x0, 0x8, 0x4, 0xc, 0x2, 0xa, 0x6, 0xe,
  0x1, 0x9, 0x5, 0xd, 0x3, 0xb, 0x7, 0xf
};

static uint32_t ReverseBits(int num_bits, uint32_t bits) {
  uint32_t retval = 0;
  int i = 0;
  while (i < num_bits) {
    i += 4;
    retval |= kReversedBits[bits & 0xf] << (MAX_ALLOWED_CODE_LENGTH + 1 - i);
    bits >>= 4;
  }
  retval >>= (MAX_ALLOWED_CODE_LENGTH + 1 - num_bits);
  return retval;
}

// Get the actual bit values for a tree of bit depths.
static void ConvertBitDepthsToSymbols(HuffmanTreeCode* const tree) {
  // 0 bit-depth means that the symbol does not exist.
  int i;
  int len;
  uint32_t next_code[MAX_ALLOWED_CODE_LENGTH + 1];
  int depth_count[MAX_ALLOWED_CODE_LENGTH + 1] = { 0 };

  assert(tree != NULL);
  len = tree->num_symbols;
  for (i = 0; i < len; ++i) {
    const int code_length = tree->code_lengths[i];
    assert(code_length <= MAX_ALLOWED_CODE_LENGTH);
    ++depth_count[code_length];
  }
  depth_count[0] = 0;  // ignore unused symbol
  next_code[0] = 0;
  {
    uint32_t code = 0;
    for (i = 1; i <= MAX_ALLOWED_CODE_LENGTH; ++i) {
      code = (code + depth_count[i - 1]) << 1;
      next_code[i] = code;
    }
  }
  for (i = 0; i < len; ++i) {
    const int code_length = tree->code_lengths[i];
    tree->codes[i] = ReverseBits(code_length, next_code[code_length]++);
  }
}

// -----------------------------------------------------------------------------
// Main entry point

void VP8LCreateHuffmanTree(uint32_t* const histogram, int tree_depth_limit,
                           uint8_t* const buf_rle, HuffmanTree* const huff_tree,
                           HuffmanTreeCode* const huff_code) {
  const int num_symbols = huff_code->num_symbols;
  memset(buf_rle, 0, num_symbols * sizeof(*buf_rle));
  OptimizeHuffmanForRle(num_symbols, buf_rle, histogram);
  GenerateOptimalTree(histogram, num_symbols, huff_tree, tree_depth_limit,
                      huff_code->code_lengths);
  // Create the actual bit codes for the bit lengths.
  ConvertBitDepthsToSymbols(huff_code);
}
