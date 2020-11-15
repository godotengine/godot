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

#include "util/mac/checked_mach_address_range.h"

#include <mach/mach.h>
#include <sys/types.h>

#include <limits>

#include "base/macros.h"
#include "base/strings/stringprintf.h"
#include "build/build_config.h"
#include "gtest/gtest.h"

namespace crashpad {
namespace test {
namespace {

enum Validity {
  kInvalid = false,
  kValid,
  kValid64Invalid32,
};

bool ExpectationForValidity32(Validity validity) {
  return validity == kValid;
}

bool ExpectationForValidity64(Validity validity) {
  return validity == kValid || validity == kValid64Invalid32;
}

TEST(CheckedMachAddressRange, IsValid) {
  static constexpr struct {
    mach_vm_address_t base;
    mach_vm_size_t size;
    Validity validity;
  } kTestData[] = {
      {0, 0, kValid},
      {0, 1, kValid},
      {0, 2, kValid},
      {0, 0x7fffffff, kValid},
      {0, 0x80000000, kValid},
      {0, 0xfffffffe, kValid},
      {0, 0xffffffff, kValid},
      {0, 0xffffffffffffffff, kValid64Invalid32},
      {1, 0, kValid},
      {1, 1, kValid},
      {1, 2, kValid},
      {1, 0x7fffffff, kValid},
      {1, 0x80000000, kValid},
      {1, 0xfffffffe, kValid},
      {1, 0xffffffff, kValid64Invalid32},
      {1, 0xfffffffffffffffe, kValid64Invalid32},
      {1, 0xffffffffffffffff, kInvalid},
      {0x7fffffff, 0, kValid},
      {0x7fffffff, 1, kValid},
      {0x7fffffff, 2, kValid},
      {0x7fffffff, 0x7fffffff, kValid},
      {0x7fffffff, 0x80000000, kValid},
      {0x7fffffff, 0xfffffffe, kValid64Invalid32},
      {0x7fffffff, 0xffffffff, kValid64Invalid32},
      {0x80000000, 0, kValid},
      {0x80000000, 1, kValid},
      {0x80000000, 2, kValid},
      {0x80000000, 0x7fffffff, kValid},
      {0x80000000, 0x80000000, kValid64Invalid32},
      {0x80000000, 0xfffffffe, kValid64Invalid32},
      {0x80000000, 0xffffffff, kValid64Invalid32},
      {0xfffffffe, 0, kValid},
      {0xfffffffe, 1, kValid},
      {0xfffffffe, 2, kValid64Invalid32},
      {0xfffffffe, 0x7fffffff, kValid64Invalid32},
      {0xfffffffe, 0x80000000, kValid64Invalid32},
      {0xfffffffe, 0xfffffffe, kValid64Invalid32},
      {0xfffffffe, 0xffffffff, kValid64Invalid32},
      {0xffffffff, 0, kValid},
      {0xffffffff, 1, kValid64Invalid32},
      {0xffffffff, 2, kValid64Invalid32},
      {0xffffffff, 0x7fffffff, kValid64Invalid32},
      {0xffffffff, 0x80000000, kValid64Invalid32},
      {0xffffffff, 0xfffffffe, kValid64Invalid32},
      {0xffffffff, 0xffffffff, kValid64Invalid32},
      {0x7fffffffffffffff, 0, kValid64Invalid32},
      {0x7fffffffffffffff, 1, kValid64Invalid32},
      {0x7fffffffffffffff, 2, kValid64Invalid32},
      {0x7fffffffffffffff, 0x7fffffffffffffff, kValid64Invalid32},
      {0x7fffffffffffffff, 0x8000000000000000, kValid64Invalid32},
      {0x7fffffffffffffff, 0x8000000000000001, kInvalid},
      {0x7fffffffffffffff, 0xfffffffffffffffe, kInvalid},
      {0x7fffffffffffffff, 0xffffffffffffffff, kInvalid},
      {0x8000000000000000, 0, kValid64Invalid32},
      {0x8000000000000000, 1, kValid64Invalid32},
      {0x8000000000000000, 2, kValid64Invalid32},
      {0x8000000000000000, 0x7fffffffffffffff, kValid64Invalid32},
      {0x8000000000000000, 0x8000000000000000, kInvalid},
      {0x8000000000000000, 0x8000000000000001, kInvalid},
      {0x8000000000000000, 0xfffffffffffffffe, kInvalid},
      {0x8000000000000000, 0xffffffffffffffff, kInvalid},
      {0xfffffffffffffffe, 0, kValid64Invalid32},
      {0xfffffffffffffffe, 1, kValid64Invalid32},
      {0xfffffffffffffffe, 2, kInvalid},
      {0xffffffffffffffff, 0, kValid64Invalid32},
      {0xffffffffffffffff, 1, kInvalid},
  };

  for (size_t index = 0; index < arraysize(kTestData); ++index) {
    const auto& testcase = kTestData[index];
    SCOPED_TRACE(base::StringPrintf("index %zu, base 0x%llx, size 0x%llx",
                                    index,
                                    testcase.base,
                                    testcase.size));

    CheckedMachAddressRange range_32(false, testcase.base, testcase.size);
    EXPECT_EQ(range_32.IsValid(), ExpectationForValidity32(testcase.validity));

    CheckedMachAddressRange range_64(true, testcase.base, testcase.size);
    EXPECT_EQ(range_64.IsValid(), ExpectationForValidity64(testcase.validity));
  }
}

TEST(CheckedMachAddressRange, ContainsValue) {
  static constexpr struct {
    mach_vm_address_t value;
    bool expectation;
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
      {0x100000000, false},
      {0xffffffffffffffff, false},
  };

  CheckedMachAddressRange parent_range_32(false, 0x2000, 0x1000);
  ASSERT_TRUE(parent_range_32.IsValid());

  for (size_t index = 0; index < arraysize(kTestData); ++index) {
    const auto& testcase = kTestData[index];
    SCOPED_TRACE(
        base::StringPrintf("index %zu, value 0x%llx", index, testcase.value));

    EXPECT_EQ(parent_range_32.ContainsValue(testcase.value),
              testcase.expectation);
  }

  CheckedMachAddressRange parent_range_64(true, 0x100000000, 0x1000);
  ASSERT_TRUE(parent_range_64.IsValid());
  EXPECT_FALSE(parent_range_64.ContainsValue(0xffffffff));
  EXPECT_TRUE(parent_range_64.ContainsValue(0x100000000));
  EXPECT_TRUE(parent_range_64.ContainsValue(0x100000001));
  EXPECT_TRUE(parent_range_64.ContainsValue(0x100000fff));
  EXPECT_FALSE(parent_range_64.ContainsValue(0x100001000));
}

TEST(CheckedMachAddressRange, ContainsRange) {
  static constexpr struct {
    mach_vm_address_t base;
    mach_vm_size_t size;
    bool expectation;
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

  CheckedMachAddressRange parent_range_32(false, 0x2000, 0x1000);
  ASSERT_TRUE(parent_range_32.IsValid());

  for (size_t index = 0; index < arraysize(kTestData); ++index) {
    const auto& testcase = kTestData[index];
    SCOPED_TRACE(base::StringPrintf("index %zu, base 0x%llx, size 0x%llx",
                                    index,
                                    testcase.base,
                                    testcase.size));

    CheckedMachAddressRange child_range_32(false, testcase.base, testcase.size);
    ASSERT_TRUE(child_range_32.IsValid());
    EXPECT_EQ(parent_range_32.ContainsRange(child_range_32),
              testcase.expectation);
  }

  CheckedMachAddressRange parent_range_64(true, 0x100000000, 0x1000);
  ASSERT_TRUE(parent_range_64.IsValid());

  CheckedMachAddressRange child_range_64(true, 0xffffffff, 2);
  EXPECT_FALSE(parent_range_64.ContainsRange(child_range_64));

  child_range_64.SetRange(true, 0x100000000, 2);
  EXPECT_TRUE(parent_range_64.ContainsRange(child_range_64));

  child_range_64.SetRange(true, 0x100000ffe, 2);
  EXPECT_TRUE(parent_range_64.ContainsRange(child_range_64));

  child_range_64.SetRange(true, 0x100000fff, 2);
  EXPECT_FALSE(parent_range_64.ContainsRange(child_range_64));
}

}  // namespace
}  // namespace test
}  // namespace crashpad
