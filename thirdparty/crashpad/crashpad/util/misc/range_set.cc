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

#include "util/misc/range_set.h"

#include <algorithm>

namespace crashpad {

RangeSet::RangeSet() = default;

RangeSet::~RangeSet() = default;

void RangeSet::Insert(VMAddress base, VMSize size) {
  if (!size) {
    return;
  }

  VMAddress last = base + size - 1;

  auto overlapping_range = ranges_.lower_bound(base);
#define OVERLAPPING_RANGES_BASE overlapping_range->second
#define OVERLAPPING_RANGES_LAST overlapping_range->first
  while (overlapping_range != ranges_.end() &&
         OVERLAPPING_RANGES_BASE <= last) {
    base = std::min(base, OVERLAPPING_RANGES_BASE);
    last = std::max(last, OVERLAPPING_RANGES_LAST);
    auto tmp = overlapping_range;
    ++overlapping_range;
    ranges_.erase(tmp);
  }
#undef OVERLAPPING_RANGES_BASE
#undef OVERLAPPING_RANGES_LAST

  ranges_[last] = base;
}

bool RangeSet::Contains(VMAddress address) const {
  auto range_above_address = ranges_.lower_bound(address);
  return range_above_address != ranges_.end() &&
         range_above_address->second <= address;
}

}  // namespace crashpad
