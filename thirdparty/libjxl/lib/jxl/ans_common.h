// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ANS_COMMON_H_
#define LIB_JXL_ANS_COMMON_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <hwy/base.h>
#include <hwy/cache_control.h>  // Prefetch
#include <vector>

#include "lib/jxl/ans_params.h"
#include "lib/jxl/base/byte_order.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"

namespace jxl {

// Returns the precision (number of bits) that should be used to store
// a histogram count such that Log2Floor(count) == logcount.
static JXL_MAYBE_UNUSED JXL_INLINE uint32_t
GetPopulationCountPrecision(uint32_t logcount, uint32_t shift) {
  int32_t r = std::min<int>(
      logcount, static_cast<int>(shift) -
                    static_cast<int>((ANS_LOG_TAB_SIZE - logcount) >> 1));
  if (r < 0) return 0;
  return r;
}

// Returns a histogram where the counts are positive, differ by at most 1,
// and add up to total_count. The bigger counts (if any) are at the beginning
// of the histogram.
std::vector<int32_t> CreateFlatHistogram(int length, int total_count);

// An alias table implements a mapping from the [0, ANS_TAB_SIZE) range into
// the [0, ANS_MAX_ALPHABET_SIZE) range, satisfying the following conditions:
// - each symbol occurs as many times as specified by any valid distribution
//   of frequencies of the symbols. A valid distribution here is an array of
//   ANS_MAX_ALPHABET_SIZE that contains numbers in the range [0, ANS_TAB_SIZE],
//   and whose sum is ANS_TAB_SIZE.
// - lookups can be done in constant time, and also return how many smaller
//   input values map into the same symbol, according to some well-defined order
//   of input values.
// - the space used by the alias table is given by a small constant times the
//   index of the largest symbol with nonzero probability in the distribution.
// Each of the entries in the table covers a range of `entry_size` values in the
// [0, ANS_TAB_SIZE) range; consecutive entries represent consecutive
// sub-ranges. In the range covered by entry `i`, the first `cutoff` values map
// to symbol `i`, while the others map to symbol `right_value`.
//
// TODO(veluca): consider making the order used for computing offsets easier to
// define - it is currently defined by the algorithm to compute the alias table.
// Beware of breaking the implicit assumption that symbols that come after the
// cutoff value should have an offset at least as big as the cutoff.

struct AliasTable {
  struct Symbol {
    size_t value;
    size_t offset;
    size_t freq;
  };

// Working set size matters here (~64 tables x 256 entries).
// offsets0 is always zero (beginning of [0] side among the same symbol).
// offsets1 is an offset of (pos >= cutoff) side decremented by cutoff.
#pragma pack(push, 1)
  struct Entry {
    uint8_t cutoff;       // < kEntrySizeMinus1 when used by ANS.
    uint8_t right_value;  // < alphabet size.
    uint16_t freq0;

    // Only used if `greater` (see Lookup)
    uint16_t offsets1;         // <= ANS_TAB_SIZE
    uint16_t freq1_xor_freq0;  // for branchless ternary in Lookup
  };
#pragma pack(pop)

  // Dividing `value` by `entry_size` determines `i`, the entry which is
  // responsible for the input. If the remainder is below `cutoff`, then the
  // mapped symbol is `i`; since `offsets[0]` stores the number of occurrences
  // of `i` "before" the start of this entry, the offset of the input will be
  // `offsets[0] + remainder`. If the remainder is above cutoff, the mapped
  // symbol is `right_value`; since `offsets[1]` stores the number of
  // occurrences of `right_value` "before" this entry, minus the `cutoff` value,
  // the input offset is then `remainder + offsets[1]`.
  static JXL_INLINE Symbol Lookup(const Entry* JXL_RESTRICT table, size_t value,
                                  size_t log_entry_size,
                                  size_t entry_size_minus_1) {
    const size_t i = value >> log_entry_size;
    const size_t pos = value & entry_size_minus_1;

#if JXL_BYTE_ORDER_LITTLE
    uint64_t entry;
    memcpy(&entry, &table[i].cutoff, sizeof(entry));
    const size_t cutoff = entry & 0xFF;              // = MOVZX
    const size_t right_value = (entry >> 8) & 0xFF;  // = MOVZX
    const size_t freq0 = (entry >> 16) & 0xFFFF;
#else
    // Generates multiple loads with complex addressing.
    const size_t cutoff = table[i].cutoff;
    const size_t right_value = table[i].right_value;
    const size_t freq0 = table[i].freq0;
#endif

    const bool greater = pos >= cutoff;

#if JXL_BYTE_ORDER_LITTLE
    const uint64_t conditional = greater ? entry : 0;  // = CMOV
    const size_t offsets1_or_0 = (conditional >> 32) & 0xFFFF;
    const size_t freq1_xor_freq0_or_0 = conditional >> 48;
#else
    const size_t offsets1_or_0 = greater ? table[i].offsets1 : 0;
    const size_t freq1_xor_freq0_or_0 = greater ? table[i].freq1_xor_freq0 : 0;
#endif

    // WARNING: moving this code may interfere with CMOV heuristics.
    Symbol s;
    s.value = greater ? right_value : i;
    s.offset = offsets1_or_0 + pos;
    s.freq = freq0 ^ freq1_xor_freq0_or_0;  // = greater ? freq1 : freq0
    // XOR avoids implementation-defined conversion from unsigned to signed.
    // Alternatives considered: BEXTR is 2 cycles on HSW, SET+shift causes
    // spills, simple ternary has a long dependency chain.

    return s;
  }

  static HWY_INLINE void Prefetch(const Entry* JXL_RESTRICT table, size_t value,
                                  size_t log_entry_size) {
    const size_t i = value >> log_entry_size;
    hwy::Prefetch(table + i);
  }
};

// Computes an alias table for a given distribution.
Status InitAliasTable(std::vector<int32_t> distribution, uint32_t log_range,
                      size_t log_alpha_size, AliasTable::Entry* JXL_RESTRICT a);

}  // namespace jxl

#endif  // LIB_JXL_ANS_COMMON_H_
