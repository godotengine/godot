// Copyright 2014 The Crashpad Authors. All rights reserved.
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

#include "util/numeric/checked_range.h"

#include <stdint.h>
#include <sys/types.h>

#include <limits>

#include "base/format_macros.h"
#include "base/macros.h"
#include "base/strings/stringprintf.h"
#include "gtest/gtest.h"

namespace crashpad {
namespace test {
namespace {

TEST(CheckedRange, IsValid) {
  static constexpr struct {
    uint32_t base;
    uint32_t size;
    bool valid;
  } kUnsignedTestData[] = {
      {0, 0, true},
      {0, 1, true},
      {0, 2, true},
      {0, 0x7fffffff, true},
      {0, 0x80000000, true},
      {0, 0xfffffffe, true},
      {0, 0xffffffff, true},
      {1, 0, true},
      {1, 1, true},
      {1, 2, true},
      {1, 0x7fffffff, true},
      {1, 0x80000000, true},
      {1, 0xfffffffe, true},
      {1, 0xffffffff, false},
      {0x7fffffff, 0, true},
      {0x7fffffff, 1, true},
      {0x7fffffff, 2, true},
      {0x7fffffff, 0x7fffffff, true},
      {0x7fffffff, 0x80000000, true},
      {0x7fffffff, 0xfffffffe, false},
      {0x7fffffff, 0xffffffff, false},
      {0x80000000, 0, true},
      {0x80000000, 1, true},
      {0x80000000, 2, true},
      {0x80000000, 0x7fffffff, true},
      {0x80000000, 0x80000000, false},
      {0x80000000, 0xfffffffe, false},
      {0x80000000, 0xffffffff, false},
      {0xfffffffe, 0, true},
      {0xfffffffe, 1, true},
      {0xfffffffe, 2, false},
      {0xfffffffe, 0x7fffffff, false},
      {0xfffffffe, 0x80000000, false},
      {0xfffffffe, 0xfffffffe, false},
      {0xfffffffe, 0xffffffff, false},
      {0xffffffff, 0, true},
      {0xffffffff, 1, false},
      {0xffffffff, 2, false},
      {0xffffffff, 0x7fffffff, false},
      {0xffffffff, 0x80000000, false},
      {0xffffffff, 0xfffffffe, false},
      {0xffffffff, 0xffffffff, false},
  };

  for (size_t index = 0; index < arraysize(kUnsignedTestData); ++index) {
    const auto& testcase = kUnsignedTestData[index];
    SCOPED_TRACE(base::StringPrintf("unsigned index %" PRIuS
                                    ", base 0x%x, size 0x%x",
                                    index,
                                    testcase.base,
                                    testcase.size));

    CheckedRange<uint32_t> range(testcase.base, testcase.size);
    EXPECT_EQ(range.IsValid(), testcase.valid);
  }

  const int32_t kMinInt32 = std::numeric_limits<int32_t>::min();
  static constexpr struct {
    int32_t base;
    uint32_t size;
    bool valid;
  } kSignedTestData[] = {
      {0, 0, true},
      {0, 1, true},
      {0, 2, true},
      {0, 0x7fffffff, true},
      {0, 0x80000000, false},
      {0, 0xfffffffe, false},
      {0, 0xffffffff, false},
      {1, 0, true},
      {1, 1, true},
      {1, 2, true},
      {1, 0x7fffffff, false},
      {1, 0x80000000, false},
      {1, 0xfffffffe, false},
      {1, 0xffffffff, false},
      {0x7fffffff, 0, true},
      {0x7fffffff, 1, false},
      {0x7fffffff, 2, false},
      {0x7fffffff, 0x7fffffff, false},
      {0x7fffffff, 0x80000000, false},
      {0x7fffffff, 0xfffffffe, false},
      {0x7fffffff, 0xffffffff, false},
      {kMinInt32, 0, true},
      {kMinInt32, 1, true},
      {kMinInt32, 2, true},
      {kMinInt32, 0x7fffffff, true},
      {kMinInt32, 0x80000000, false},
      {kMinInt32, 0xfffffffe, false},
      {kMinInt32, 0xffffffff, false},
      {-2, 0, true},
      {-2, 1, true},
      {-2, 2, true},
      {-2, 0x7fffffff, true},
      {-2, 0x80000000, false},
      {-2, 0xfffffffe, false},
      {-2, 0xffffffff, false},
      {-1, 0, true},
      {-1, 1, true},
      {-1, 2, true},
      {-1, 0x7fffffff, true},
      {-1, 0x80000000, false},
      {-1, 0xfffffffe, false},
      {-1, 0xffffffff, false},
  };

  for (size_t index = 0; index < arraysize(kSignedTestData); ++index) {
    const auto& testcase = kSignedTestData[index];
    SCOPED_TRACE(base::StringPrintf("signed index %" PRIuS
                                    ", base 0x%x, size 0x%x",
                                    index,
                                    testcase.base,
                                    testcase.size));

    CheckedRange<int32_t, uint32_t> range(testcase.base, testcase.size);
    EXPECT_EQ(range.IsValid(), testcase.valid);
  }
}

TEST(CheckedRange, ContainsValue) {
  static constexpr struct {
    uint32_t value;
    bool contains;
  } kTestData[] = {
      {0, false},
      {1, false},
      {0x1fff, false},
      {0x2000, true},
      {0x2001, true},
      {0x2ffe, true},
      {0x2fff, true},
      {0x3000, false},
      {0x3001, false},
      {0x7fffffff, false},
      {0x80000000, false},
      {0x80000001, false},
      {0x80001fff, false},
      {0x80002000, false},
      {0x80002001, false},
      {0x80002ffe, false},
      {0x80002fff, false},
      {0x80003000, false},
      {0x80003001, false},
      {0xffffcfff, false},
      {0xffffdfff, false},
      {0xffffefff, false},
      {0xffffffff, false},
  };

  CheckedRange<uint32_t> parent_range(0x2000, 0x1000);
  ASSERT_TRUE(parent_range.IsValid());

  for (size_t index = 0; index < arraysize(kTestData); ++index) {
    const auto& testcase = kTestData[index];
    SCOPED_TRACE(base::StringPrintf(
        "index %" PRIuS ", value 0x%x", index, testcase.value));

    EXPECT_EQ(parent_range.ContainsValue(testcase.value), testcase.contains);
  }
}

TEST(CheckedRange, ContainsRange) {
  static constexpr struct {
    uint32_t base;
    uint32_t size;
    bool contains;
  } kTestData[] = {
      {0, 0, false},
      {0, 1, false},
      {0x2000, 0x1000, true},
      {0, 0x2000, false},
      {0x3000, 0x1000, false},
      {0x1800, 0x1000, false},
      {0x2800, 0x1000, false},
      {0x2000, 0x800, true},
      {0x2800, 0x800, true},
      {0x2400, 0x800, true},
      {0x2800, 0, true},
      {0x2000, 0xffffdfff, false},
      {0x2800, 0xffffd7ff, false},
      {0x3000, 0xffffcfff, false},
      {0xfffffffe, 1, false},
      {0xffffffff, 0, false},
      {0x1fff, 0, false},
      {0x2000, 0, true},
      {0x2001, 0, true},
      {0x2fff, 0, true},
      {0x3000, 0, true},
      {0x3001, 0, false},
      {0x1fff, 1, false},
      {0x2000, 1, true},
      {0x2001, 1, true},
      {0x2fff, 1, true},
      {0x3000, 1, false},
      {0x3001, 1, false},
  };

  CheckedRange<uint32_t> parent_range(0x2000, 0x1000);
  ASSERT_TRUE(parent_range.IsValid());

  for (size_t index = 0; index < arraysize(kTestData); ++index) {
    const auto& testcase = kTestData[index];
    SCOPED_TRACE(base::StringPrintf("index %" PRIuS ", base 0x%x, size 0x%x",
                                    index,
                                    testcase.base,
                                    testcase.size));

    CheckedRange<uint32_t> child_range(testcase.base, testcase.size);
    ASSERT_TRUE(child_range.IsValid());
    EXPECT_EQ(parent_range.ContainsRange(child_range), testcase.contains);
  }
}

TEST(CheckedRange, OverlapsRange) {
  static constexpr struct {
    uint32_t base;
    uint32_t size;
    bool overlaps;
  } kTestData[] = {
      {0, 0, false},
      {0, 1, false},
      {0x2000, 0x1000, true},
      {0, 0x2000, false},
      {0x3000, 0x1000, false},
      {0x1800, 0x1000, true},
      {0x1800, 0x2000, true},
      {0x2800, 0x1000, true},
      {0x2000, 0x800, true},
      {0x2800, 0x800, true},
      {0x2400, 0x800, true},
      {0x2800, 0, false},
      {0x2000, 0xffffdfff, true},
      {0x2800, 0xffffd7ff, true},
      {0x3000, 0xffffcfff, false},
      {0xfffffffe, 1, false},
      {0xffffffff, 0, false},
      {0x1fff, 0, false},
      {0x2000, 0, false},
      {0x2001, 0, false},
      {0x2fff, 0, false},
      {0x3000, 0, false},
      {0x3001, 0, false},
      {0x1fff, 1, false},
      {0x2000, 1, true},
      {0x2001, 1, true},
      {0x2fff, 1, true},
      {0x3000, 1, false},
      {0x3001, 1, false},
  };

  CheckedRange<uint32_t> first_range(0x2000, 0x1000);
  ASSERT_TRUE(first_range.IsValid());

  for (size_t index = 0; index < arraysize(kTestData); ++index) {
    const auto& testcase = kTestData[index];
    SCOPED_TRACE(base::StringPrintf("index %" PRIuS ", base 0x%x, size 0x%x",
                                    index,
                                    testcase.base,
                                    testcase.size));

    CheckedRange<uint32_t> second_range(testcase.base, testcase.size);
    ASSERT_TRUE(second_range.IsValid());
    EXPECT_EQ(first_range.OverlapsRange(second_range), testcase.overlaps);
  }
}

}  // namespace
}  // namespace test
}  // namespace crashpad
