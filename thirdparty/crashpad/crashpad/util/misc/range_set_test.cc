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

#include <sys/types.h>

#include <memory>

#include "base/format_macros.h"
#include "base/strings/stringprintf.h"
#include "gtest/gtest.h"
#include "util/misc/address_types.h"
#include "util/misc/from_pointer_cast.h"

namespace crashpad {
namespace test {
namespace {

void ExpectRangeIsContained(const RangeSet& ranges,
                            VMAddress base,
                            VMSize size) {
  for (VMAddress addr = base; addr < base + size; ++addr) {
    SCOPED_TRACE(base::StringPrintf("0x%" PRIx64 " in range 0x%" PRIx64
                                    ":0x%" PRIx64,
                                    addr,
                                    base,
                                    base + size));
    EXPECT_TRUE(ranges.Contains(addr));
  }
}

TEST(RangeSet, Basic) {
  RangeSet ranges;
  auto base = FromPointerCast<VMAddress>(&ranges);
  VMSize size = sizeof(ranges);
  ranges.Insert(base, size);
  ExpectRangeIsContained(ranges, base, size);
  EXPECT_FALSE(ranges.Contains(base - 1));
  EXPECT_FALSE(ranges.Contains(base + size));
}

TEST(RangeSet, ZeroSizedRange) {
  RangeSet ranges;
  auto addr = FromPointerCast<VMAddress>(&ranges);
  ranges.Insert(addr, 0);
  EXPECT_FALSE(ranges.Contains(addr));
}

TEST(RangeSet, DuplicateRanges) {
  RangeSet ranges;
  auto base = FromPointerCast<VMAddress>(&ranges);
  VMSize size = sizeof(ranges);
  ranges.Insert(base, size);
  ranges.Insert(base, size);
  ExpectRangeIsContained(ranges, base, size);
}

TEST(RangeSet, OverlappingRanges) {
  RangeSet ranges;
  ranges.Insert(37, 16);
  ranges.Insert(9, 9);
  ranges.Insert(17, 42);

  EXPECT_TRUE(ranges.Contains(9));
  EXPECT_TRUE(ranges.Contains(17));
  EXPECT_TRUE(ranges.Contains(36));
  EXPECT_TRUE(ranges.Contains(37));
  EXPECT_TRUE(ranges.Contains(52));
  EXPECT_TRUE(ranges.Contains(58));
}

TEST(RangeSet, SubRangeInLargeRange) {
  constexpr size_t kBufferSize = 2 << 22;
  auto buf = std::make_unique<char[]>(kBufferSize);

  RangeSet ranges;
  auto addr = FromPointerCast<VMAddress>(buf.get());

  ranges.Insert(addr, kBufferSize);
  EXPECT_TRUE(ranges.Contains(addr));
  EXPECT_TRUE(ranges.Contains(addr + kBufferSize - 1));

  ranges.Insert(addr, kBufferSize / 2);
  EXPECT_TRUE(ranges.Contains(addr));
  EXPECT_TRUE(ranges.Contains(addr + kBufferSize / 2 - 1));
  EXPECT_TRUE(ranges.Contains(addr + kBufferSize - 1));
}

TEST(RangeSet, LargeOverlappingRanges) {
  constexpr size_t kBufferSize = 2 << 23;
  auto buf = std::make_unique<char[]>(kBufferSize);

  RangeSet ranges;
  auto addr = FromPointerCast<VMAddress>(buf.get());

  ranges.Insert(addr, 3 * kBufferSize / 4);
  ranges.Insert(addr + kBufferSize / 4, 3 * kBufferSize / 4);
  EXPECT_TRUE(ranges.Contains(addr));
  EXPECT_TRUE(ranges.Contains(addr + kBufferSize - 1));
}

}  // namespace
}  // namespace test
}  // namespace crashpad
