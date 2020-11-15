// Copyright 2015 The Crashpad Authors. All rights reserved.
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

#include "util/stdlib/aligned_allocator.h"

#include <stdint.h>

#include "base/compiler_specific.h"
#include "gtest/gtest.h"
#include "test/gtest_death.h"

#if defined(OS_WIN)
#include <crtdbg.h>
#endif

namespace crashpad {
namespace test {
namespace {

bool IsAligned(void* pointer, size_t alignment) {
  uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
  return (address & (alignment - 1)) == 0;
}

TEST(AlignedAllocator, AlignedVector) {
  // Test a structure with natural alignment.
  struct NaturalAlignedStruct {
    int i;
  };

  AlignedVector<NaturalAlignedStruct> natural_aligned_vector;
  natural_aligned_vector.push_back(NaturalAlignedStruct());
  EXPECT_TRUE(
      IsAligned(&natural_aligned_vector[0], alignof(NaturalAlignedStruct)));

  natural_aligned_vector.resize(3);
  EXPECT_TRUE(
      IsAligned(&natural_aligned_vector[0], alignof(NaturalAlignedStruct)));
  EXPECT_TRUE(
      IsAligned(&natural_aligned_vector[1], alignof(NaturalAlignedStruct)));
  EXPECT_TRUE(
      IsAligned(&natural_aligned_vector[2], alignof(NaturalAlignedStruct)));

  // Test a structure that declares its own alignment.
  struct alignas(16) AlignedStruct {
    int i;
  };
  ASSERT_EQ(alignof(AlignedStruct), 16u);

  AlignedVector<AlignedStruct> aligned_vector;
  aligned_vector.push_back(AlignedStruct());
  EXPECT_TRUE(IsAligned(&aligned_vector[0], alignof(AlignedStruct)));

  aligned_vector.resize(3);
  EXPECT_TRUE(IsAligned(&aligned_vector[0], alignof(AlignedStruct)));
  EXPECT_TRUE(IsAligned(&aligned_vector[1], alignof(AlignedStruct)));
  EXPECT_TRUE(IsAligned(&aligned_vector[2], alignof(AlignedStruct)));

  // Try a custom alignment. Since the structure itself doesn’t specify an
  // alignment constraint, only the base address will be aligned to the
  // requested boundary.
  AlignedVector<NaturalAlignedStruct, 32> custom_aligned_vector;
  custom_aligned_vector.push_back(NaturalAlignedStruct());
  EXPECT_TRUE(IsAligned(&custom_aligned_vector[0], 32));

  // Try a structure with a pretty big alignment request.
  struct alignas(64) BigAlignedStruct {
    int i;
  };
  ASSERT_EQ(alignof(BigAlignedStruct), 64u);

  AlignedVector<BigAlignedStruct> big_aligned_vector;
  big_aligned_vector.push_back(BigAlignedStruct());
  EXPECT_TRUE(IsAligned(&big_aligned_vector[0], alignof(BigAlignedStruct)));

  big_aligned_vector.resize(3);
  EXPECT_TRUE(IsAligned(&big_aligned_vector[0], alignof(BigAlignedStruct)));
  EXPECT_TRUE(IsAligned(&big_aligned_vector[1], alignof(BigAlignedStruct)));
  EXPECT_TRUE(IsAligned(&big_aligned_vector[2], alignof(BigAlignedStruct)));
}

void BadAlignmentTest() {
#if defined(OS_WIN)
  // Suppress the assertion MessageBox() normally displayed by the CRT in debug
  // mode.
  int previous = _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_DEBUG);

  // In release mode, _CrtSetReportMode() is #defined to ((int)0), so |previous|
  // would appear unused.
  ALLOW_UNUSED_LOCAL(previous);
#endif

  // Alignment constraints must be powers of 2. 7 is not valid.
  AlignedVector<int, 7> bad_aligned_vector;
  bad_aligned_vector.push_back(0);

#if defined(OS_WIN)
  _CrtSetReportMode(_CRT_ASSERT, previous);
#endif
}

TEST(AlignedAllocatorDeathTest, BadAlignment) {
  ASSERT_DEATH_CRASH(BadAlignmentTest(), "");
}

}  // namespace
}  // namespace test
}  // namespace crashpad
