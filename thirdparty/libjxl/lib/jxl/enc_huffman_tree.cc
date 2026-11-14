// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_huffman_tree.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"

namespace jxl {

void SetDepth(const HuffmanTree& p, HuffmanTree* pool, uint8_t* depth,
              uint8_t level) {
  if (p.index_left >= 0) {
    ++level;
    SetDepth(pool[p.index_left], pool, depth, level);
    SetDepth(pool[p.index_right_or_value], pool, depth, level);
  } else {
    depth[p.index_right_or_value] = level;
  }
}

// Compare the root nodes, least popular first; indices are in decreasing order
// before sorting is applied.
static JXL_INLINE bool Compare(const HuffmanTree& v0, const HuffmanTree& v1) {
  return v0.total_count != v1.total_count
             ? v0.total_count < v1.total_count
             : v0.index_right_or_value > v1.index_right_or_value;
}

// This function will create a Huffman tree.
//
// The catch here is that the tree cannot be arbitrarily deep.
// Brotli specifies a maximum depth of 15 bits for "code trees"
// and 7 bits for "code length code trees."
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
void CreateHuffmanTree(const uint32_t* data, const size_t length,
                       const int tree_limit, uint8_t* depth) {
  // For block sizes below 64 kB, we never need to do a second iteration
  // of this loop. Probably all of our block sizes will be smaller than
  // that, so this loop is mostly of academic interest. If we actually
  // would need this, we would be better off with the Katajainen algorithm.
  for (uint32_t count_limit = 1;; count_limit *= 2) {
    std::vector<HuffmanTree> tree;
    tree.reserve(2 * length + 1);

    for (size_t i = length; i != 0;) {
      --i;
      if (data[i]) {
        const uint32_t count = std::max(data[i], count_limit - 1);
        tree.emplace_back(count, -1, static_cast<int16_t>(i));
      }
    }

    const size_t n = tree.size();
    if (n == 1) {
      // Fake value; will be fixed on upper level.
      depth[tree[0].index_right_or_value] = 1;
      break;
    }

    std::sort(tree.begin(), tree.end(), Compare);

    // The nodes are:
    // [0, n): the sorted leaf nodes that we start with.
    // [n]: we add a sentinel here.
    // [n + 1, 2n): new parent nodes are added here, starting from
    //              (n+1). These are naturally in ascending order.
    // [2n]: we add a sentinel at the end as well.
    // There will be (2n+1) elements at the end.
    const HuffmanTree sentinel(std::numeric_limits<uint32_t>::max(), -1, -1);
    tree.push_back(sentinel);
    tree.push_back(sentinel);

    size_t i = 0;      // Points to the next leaf node.
    size_t j = n + 1;  // Points to the next non-leaf node.
    for (size_t k = n - 1; k != 0; --k) {
      size_t left;
      size_t right;
      if (tree[i].total_count <= tree[j].total_count) {
        left = i;
        ++i;
      } else {
        left = j;
        ++j;
      }
      if (tree[i].total_count <= tree[j].total_count) {
        right = i;
        ++i;
      } else {
        right = j;
        ++j;
      }

      // The sentinel node becomes the parent node.
      size_t j_end = tree.size() - 1;
      tree[j_end].total_count =
          tree[left].total_count + tree[right].total_count;
      tree[j_end].index_left = static_cast<int16_t>(left);
      tree[j_end].index_right_or_value = static_cast<int16_t>(right);

      // Add back the last sentinel node.
      tree.push_back(sentinel);
    }
    JXL_DASSERT(tree.size() == 2 * n + 1);
    SetDepth(tree[2 * n - 1], tree.data(), depth, 0);

    // We need to pack the Huffman tree in tree_limit bits.
    // If this was not successful, add fake entities to the lowest values
    // and retry.
    if (*std::max_element(&depth[0], &depth[length]) <= tree_limit) {
      break;
    }
  }
}

void Reverse(uint8_t* v, size_t start, size_t end) {
  --end;
  while (start < end) {
    uint8_t tmp = v[start];
    v[start] = v[end];
    v[end] = tmp;
    ++start;
    --end;
  }
}

void WriteHuffmanTreeRepetitions(const uint8_t previous_value,
                                 const uint8_t value, size_t repetitions,
                                 size_t* tree_size, uint8_t* tree,
                                 uint8_t* extra_bits_data) {
  JXL_DASSERT(repetitions > 0);
  if (previous_value != value) {
    tree[*tree_size] = value;
    extra_bits_data[*tree_size] = 0;
    ++(*tree_size);
    --repetitions;
  }
  if (repetitions == 7) {
    tree[*tree_size] = value;
    extra_bits_data[*tree_size] = 0;
    ++(*tree_size);
    --repetitions;
  }
  if (repetitions < 3) {
    for (size_t i = 0; i < repetitions; ++i) {
      tree[*tree_size] = value;
      extra_bits_data[*tree_size] = 0;
      ++(*tree_size);
    }
  } else {
    repetitions -= 3;
    size_t start = *tree_size;
    while (true) {
      tree[*tree_size] = 16;
      extra_bits_data[*tree_size] = repetitions & 0x3;
      ++(*tree_size);
      repetitions >>= 2;
      if (repetitions == 0) {
        break;
      }
      --repetitions;
    }
    Reverse(tree, start, *tree_size);
    Reverse(extra_bits_data, start, *tree_size);
  }
}

void WriteHuffmanTreeRepetitionsZeros(size_t repetitions, size_t* tree_size,
                                      uint8_t* tree, uint8_t* extra_bits_data) {
  if (repetitions == 11) {
    tree[*tree_size] = 0;
    extra_bits_data[*tree_size] = 0;
    ++(*tree_size);
    --repetitions;
  }
  if (repetitions < 3) {
    for (size_t i = 0; i < repetitions; ++i) {
      tree[*tree_size] = 0;
      extra_bits_data[*tree_size] = 0;
      ++(*tree_size);
    }
  } else {
    repetitions -= 3;
    size_t start = *tree_size;
    while (true) {
      tree[*tree_size] = 17;
      extra_bits_data[*tree_size] = repetitions & 0x7;
      ++(*tree_size);
      repetitions >>= 3;
      if (repetitions == 0) {
        break;
      }
      --repetitions;
    }
    Reverse(tree, start, *tree_size);
    Reverse(extra_bits_data, start, *tree_size);
  }
}

static void DecideOverRleUse(const uint8_t* depth, const size_t length,
                             bool* use_rle_for_non_zero,
                             bool* use_rle_for_zero) {
  size_t total_reps_zero = 0;
  size_t total_reps_non_zero = 0;
  size_t count_reps_zero = 1;
  size_t count_reps_non_zero = 1;
  for (size_t i = 0; i < length;) {
    const uint8_t value = depth[i];
    size_t reps = 1;
    for (size_t k = i + 1; k < length && depth[k] == value; ++k) {
      ++reps;
    }
    if (reps >= 3 && value == 0) {
      total_reps_zero += reps;
      ++count_reps_zero;
    }
    if (reps >= 4 && value != 0) {
      total_reps_non_zero += reps;
      ++count_reps_non_zero;
    }
    i += reps;
  }
  *use_rle_for_non_zero = total_reps_non_zero > count_reps_non_zero * 2;
  *use_rle_for_zero = total_reps_zero > count_reps_zero * 2;
}

void WriteHuffmanTree(const uint8_t* depth, size_t length, size_t* tree_size,
                      uint8_t* tree, uint8_t* extra_bits_data) {
  uint8_t previous_value = 8;

  // Throw away trailing zeros.
  size_t new_length = length;
  for (size_t i = 0; i < length; ++i) {
    if (depth[length - i - 1] == 0) {
      --new_length;
    } else {
      break;
    }
  }

  // First gather statistics on if it is a good idea to do rle.
  bool use_rle_for_non_zero = false;
  bool use_rle_for_zero = false;
  if (length > 50) {
    // Find rle coding for longer codes.
    // Shorter codes seem not to benefit from rle.
    DecideOverRleUse(depth, new_length, &use_rle_for_non_zero,
                     &use_rle_for_zero);
  }

  // Actual rle coding.
  for (size_t i = 0; i < new_length;) {
    const uint8_t value = depth[i];
    size_t reps = 1;
    if ((value != 0 && use_rle_for_non_zero) ||
        (value == 0 && use_rle_for_zero)) {
      for (size_t k = i + 1; k < new_length && depth[k] == value; ++k) {
        ++reps;
      }
    }
    if (value == 0) {
      WriteHuffmanTreeRepetitionsZeros(reps, tree_size, tree, extra_bits_data);
    } else {
      WriteHuffmanTreeRepetitions(previous_value, value, reps, tree_size, tree,
                                  extra_bits_data);
      previous_value = value;
    }
    i += reps;
  }
}

namespace {

uint16_t ReverseBits(int num_bits, uint16_t bits) {
  static const size_t kLut[16] = {// Pre-reversed 4-bit values.
                                  0x0, 0x8, 0x4, 0xc, 0x2, 0xa, 0x6, 0xe,
                                  0x1, 0x9, 0x5, 0xd, 0x3, 0xb, 0x7, 0xf};
  size_t retval = kLut[bits & 0xf];
  for (int i = 4; i < num_bits; i += 4) {
    retval <<= 4;
    bits = static_cast<uint16_t>(bits >> 4);
    retval |= kLut[bits & 0xf];
  }
  retval >>= (-num_bits & 0x3);
  return static_cast<uint16_t>(retval);
}

}  // namespace

void ConvertBitDepthsToSymbols(const uint8_t* depth, size_t len,
                               uint16_t* bits) {
  // In Brotli, all bit depths are [1..15]
  // 0 bit depth means that the symbol does not exist.
  const int kMaxBits = 16;  // 0..15 are values for bits
  uint16_t bl_count[kMaxBits] = {0};
  {
    for (size_t i = 0; i < len; ++i) {
      ++bl_count[depth[i]];
    }
    bl_count[0] = 0;
  }
  uint16_t next_code[kMaxBits];
  next_code[0] = 0;
  {
    int code = 0;
    for (size_t i = 1; i < kMaxBits; ++i) {
      code = (code + bl_count[i - 1]) << 1;
      next_code[i] = static_cast<uint16_t>(code);
    }
  }
  for (size_t i = 0; i < len; ++i) {
    if (depth[i]) {
      bits[i] = ReverseBits(depth[i], next_code[depth[i]]++);
    }
  }
}

}  // namespace jxl
