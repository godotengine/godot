// Copyright (c) 2022 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SOURCE_DIFF_LCS_H_
#define SOURCE_DIFF_LCS_H_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stack>
#include <vector>

namespace spvtools {
namespace diff {

// The result of a diff.
using DiffMatch = std::vector<bool>;

// Helper class to find the longest common subsequence between two function
// bodies.
template <typename Sequence>
class LongestCommonSubsequence {
 public:
  LongestCommonSubsequence(const Sequence& src, const Sequence& dst)
      : src_(src),
        dst_(dst),
        table_(src.size(), std::vector<DiffMatchEntry>(dst.size())) {}

  // Given two sequences, it creates a matching between them.  The elements are
  // simply marked as matched in src and dst, with any unmatched element in src
  // implying a removal and any unmatched element in dst implying an addition.
  //
  // Returns the length of the longest common subsequence.
  template <typename T>
  uint32_t Get(std::function<bool(T src_elem, T dst_elem)> match,
               DiffMatch* src_match_result, DiffMatch* dst_match_result);

 private:
  struct DiffMatchIndex {
    uint32_t src_offset;
    uint32_t dst_offset;
  };

  template <typename T>
  void CalculateLCS(std::function<bool(T src_elem, T dst_elem)> match);
  void RetrieveMatch(DiffMatch* src_match_result, DiffMatch* dst_match_result);
  bool IsInBound(DiffMatchIndex index) {
    return index.src_offset < src_.size() && index.dst_offset < dst_.size();
  }
  bool IsCalculated(DiffMatchIndex index) {
    assert(IsInBound(index));
    return table_[index.src_offset][index.dst_offset].valid;
  }
  bool IsCalculatedOrOutOfBound(DiffMatchIndex index) {
    return !IsInBound(index) || IsCalculated(index);
  }
  uint32_t GetMemoizedLength(DiffMatchIndex index) {
    if (!IsInBound(index)) {
      return 0;
    }
    assert(IsCalculated(index));
    return table_[index.src_offset][index.dst_offset].best_match_length;
  }
  bool IsMatched(DiffMatchIndex index) {
    assert(IsCalculated(index));
    return table_[index.src_offset][index.dst_offset].matched;
  }
  void MarkMatched(DiffMatchIndex index, uint32_t best_match_length,
                   bool matched) {
    assert(IsInBound(index));
    DiffMatchEntry& entry = table_[index.src_offset][index.dst_offset];
    assert(!entry.valid);

    entry.best_match_length = best_match_length & 0x3FFFFFFF;
    assert(entry.best_match_length == best_match_length);
    entry.matched = matched;
    entry.valid = true;
  }

  const Sequence& src_;
  const Sequence& dst_;

  struct DiffMatchEntry {
    DiffMatchEntry() : best_match_length(0), matched(false), valid(false) {}

    uint32_t best_match_length : 30;
    // Whether src[i] and dst[j] matched.  This is an optimization to avoid
    // calling the `match` function again when walking the LCS table.
    uint32_t matched : 1;
    // Use for the recursive algorithm to know if the contents of this entry are
    // valid.
    uint32_t valid : 1;
  };

  std::vector<std::vector<DiffMatchEntry>> table_;
};

template <typename Sequence>
template <typename T>
uint32_t LongestCommonSubsequence<Sequence>::Get(
    std::function<bool(T src_elem, T dst_elem)> match,
    DiffMatch* src_match_result, DiffMatch* dst_match_result) {
  CalculateLCS(match);
  RetrieveMatch(src_match_result, dst_match_result);
  return GetMemoizedLength({0, 0});
}

template <typename Sequence>
template <typename T>
void LongestCommonSubsequence<Sequence>::CalculateLCS(
    std::function<bool(T src_elem, T dst_elem)> match) {
  // The LCS algorithm is simple.  Given sequences s and d, with a:b depicting a
  // range in python syntax:
  //
  //     lcs(s[i:], d[j:]) =
  //         lcs(s[i+1:], d[j+1:]) + 1                        if s[i] == d[j]
  //         max(lcs(s[i+1:], d[j:]), lcs(s[i:], d[j+1:]))               o.w.
  //
  // Once the LCS table is filled according to the above, it can be walked and
  // the best match retrieved.
  //
  // This is a recursive function with memoization, which avoids filling table
  // entries where unnecessary.  This makes the best case O(N) instead of
  // O(N^2).  The implemention uses a std::stack to avoid stack overflow on long
  // sequences.

  if (src_.empty() || dst_.empty()) {
    return;
  }

  std::stack<DiffMatchIndex> to_calculate;
  to_calculate.push({0, 0});

  while (!to_calculate.empty()) {
    DiffMatchIndex current = to_calculate.top();
    to_calculate.pop();
    assert(IsInBound(current));

    // If already calculated through another path, ignore it.
    if (IsCalculated(current)) {
      continue;
    }

    if (match(src_[current.src_offset], dst_[current.dst_offset])) {
      // If the current elements match, advance both indices and calculate the
      // LCS if not already.  Visit `current` again afterwards, so its
      // corresponding entry will be updated.
      DiffMatchIndex next = {current.src_offset + 1, current.dst_offset + 1};
      if (IsCalculatedOrOutOfBound(next)) {
        MarkMatched(current, GetMemoizedLength(next) + 1, true);
      } else {
        to_calculate.push(current);
        to_calculate.push(next);
      }
      continue;
    }

    // We've reached a pair of elements that don't match.  Calculate the LCS for
    // both cases of either being left unmatched and take the max.  Visit
    // `current` again afterwards, so its corresponding entry will be updated.
    DiffMatchIndex next_src = {current.src_offset + 1, current.dst_offset};
    DiffMatchIndex next_dst = {current.src_offset, current.dst_offset + 1};

    if (IsCalculatedOrOutOfBound(next_src) &&
        IsCalculatedOrOutOfBound(next_dst)) {
      uint32_t best_match_length =
          std::max(GetMemoizedLength(next_src), GetMemoizedLength(next_dst));
      MarkMatched(current, best_match_length, false);
      continue;
    }

    to_calculate.push(current);
    if (!IsCalculatedOrOutOfBound(next_src)) {
      to_calculate.push(next_src);
    }
    if (!IsCalculatedOrOutOfBound(next_dst)) {
      to_calculate.push(next_dst);
    }
  }
}

template <typename Sequence>
void LongestCommonSubsequence<Sequence>::RetrieveMatch(
    DiffMatch* src_match_result, DiffMatch* dst_match_result) {
  src_match_result->clear();
  dst_match_result->clear();

  src_match_result->resize(src_.size(), false);
  dst_match_result->resize(dst_.size(), false);

  DiffMatchIndex current = {0, 0};
  while (IsInBound(current)) {
    if (IsMatched(current)) {
      (*src_match_result)[current.src_offset++] = true;
      (*dst_match_result)[current.dst_offset++] = true;
      continue;
    }

    if (GetMemoizedLength({current.src_offset + 1, current.dst_offset}) >=
        GetMemoizedLength({current.src_offset, current.dst_offset + 1})) {
      ++current.src_offset;
    } else {
      ++current.dst_offset;
    }
  }
}

}  // namespace diff
}  // namespace spvtools

#endif  // SOURCE_DIFF_LCS_H_
