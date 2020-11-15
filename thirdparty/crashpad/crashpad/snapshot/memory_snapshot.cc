// Copyright 2018 The Crashpad Authors. All rights reserved.
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

#include "snapshot/memory_snapshot.h"

#include <algorithm>

#include "base/format_macros.h"
#include "base/strings/stringprintf.h"
#include "util/numeric/checked_range.h"

namespace crashpad {
namespace {

bool DetermineMergedRangeImpl(bool log,
                              const MemorySnapshot* a,
                              const MemorySnapshot* b,
                              CheckedRange<uint64_t, size_t>* merged) {
  if (a->Size() == 0) {
    LOG_IF(ERROR, log) << base::StringPrintf(
        "invalid empty range at 0x%" PRIx64, a->Address());
    return false;
  }

  if (b->Size() == 0) {
    LOG_IF(ERROR, log) << base::StringPrintf(
        "invalid empty range at 0x%" PRIx64, b->Address());
    return false;
  }

  CheckedRange<uint64_t, size_t> range_a(a->Address(), a->Size());
  if (!range_a.IsValid()) {
    LOG_IF(ERROR, log) << base::StringPrintf("invalid range at 0x%" PRIx64
                                             ", size %" PRIuS,
                                             range_a.base(),
                                             range_a.size());
    return false;
  }

  CheckedRange<uint64_t, size_t> range_b(b->Address(), b->Size());
  if (!range_b.IsValid()) {
    LOG_IF(ERROR, log) << base::StringPrintf("invalid range at 0x%" PRIx64
                                             ", size %" PRIuS,
                                             range_b.base(),
                                             range_b.size());
    return false;
  }

  if (!range_a.OverlapsRange(range_b) && range_a.end() != range_b.base() &&
      range_b.end() != range_a.base()) {
    LOG_IF(ERROR, log) << base::StringPrintf(
        "ranges not overlapping or abutting: (0x%" PRIx64 ", size %" PRIuS
        ") and (0x%" PRIx64 ", size %" PRIuS ")",
        range_a.base(),
        range_a.size(),
        range_b.base(),
        range_b.size());
    return false;
  }

  if (merged) {
    uint64_t base = std::min(range_a.base(), range_b.base());
    uint64_t end = std::max(range_a.end(), range_b.end());
    size_t size = static_cast<size_t>(end - base);
    merged->SetRange(base, size);
  }
  return true;
}

}  // namespace

bool LoggingDetermineMergedRange(const MemorySnapshot* a,
                                 const MemorySnapshot* b,
                                 CheckedRange<uint64_t, size_t>* merged) {
  return DetermineMergedRangeImpl(true, a, b, merged);
}

bool DetermineMergedRange(const MemorySnapshot* a,
                          const MemorySnapshot* b,
                          CheckedRange<uint64_t, size_t>* merged) {
  return DetermineMergedRangeImpl(false, a, b, merged);
}

}  // namespace crashpad
