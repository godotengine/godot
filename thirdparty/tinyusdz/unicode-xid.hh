// SPDX-License-Identifier: MIT
// Copyright 2024 - Present, Light Transport Entertainment Inc.
//
// UTF-8 Unicode identifier XID_Start and XID_Continue validation utility.
//
// Based on UAX31 Default Identifier and Unicode 5.1.0
#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>
#include <limits>

namespace unicode_xid {

constexpr uint32_t kMaxCodepoint = 0x10FFFF;

namespace detail {

// Assume table is sorted by the first key(range_begin)
#include "unicode-xid-table.inc"

}

inline bool is_xid_start(uint32_t codepoint) {
  if (codepoint > kMaxCodepoint) {
    return false;
  }

  // first a range(range_begin <= codepoint <= range_end) by comparing the second key(range end).
  auto it = std::lower_bound(detail::kXID_StartTable.begin(), detail::kXID_StartTable.end(), int(codepoint), [](const std::pair<int, int> &a, const int b) {
    return a.second < b;
  });

  if (it != detail::kXID_StartTable.end()) {
    if ((int(codepoint) >= it->first) && (int(codepoint) <= it->second)) { // range end is inclusive.
      return true;
    }
  }

  return false;
}

inline bool is_xid_continue(uint32_t codepoint) {
  if (codepoint > kMaxCodepoint) {
    return false;
  }

  auto it = std::lower_bound(detail::kXID_ContinueTable.begin(), detail::kXID_ContinueTable.end(), int(codepoint), [](const std::pair<int, int> &a, const int b) {
    return a.second < b;
  });

  if (it != detail::kXID_ContinueTable.end()) {
    if ((int(codepoint) >= it->first) && (int(codepoint) <= it->second)) { // range end is inclusive.
      return true;
    }
  }

  return false;
}

} // namespace unicode_xid


