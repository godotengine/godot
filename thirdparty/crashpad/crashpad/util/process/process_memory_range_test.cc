// Copyright 2017 The Crashpad Authors. All rights reserved.
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

#include "util/process/process_memory_range.h"

#include <unistd.h>

#include <limits>

#include "base/logging.h"
#include "build/build_config.h"
#include "gtest/gtest.h"
#include "util/misc/from_pointer_cast.h"

#if defined(OS_FUCHSIA)
#include <lib/zx/process.h>

#include "util/process/process_memory_fuchsia.h"
#else
#include "util/process/process_memory_linux.h"
#endif

namespace crashpad {
namespace test {
namespace {

struct TestObject {
  char string1[16];
  char string2[16];
} kTestObject = {"string1", "string2"};

TEST(ProcessMemoryRange, Basic) {
#if defined(OS_FUCHSIA)
  ProcessMemoryFuchsia memory;
  ASSERT_TRUE(memory.Initialize(*zx::process::self()));
  constexpr bool is_64_bit = true;
#else
  pid_t pid = getpid();
#if defined(ARCH_CPU_64_BITS)
  constexpr bool is_64_bit = true;
#else
  constexpr bool is_64_bit = false;
#endif  // ARCH_CPU_64_BITS

  ProcessMemoryLinux memory;
  ASSERT_TRUE(memory.Initialize(pid));
#endif  // OS_FUCHSIA

  ProcessMemoryRange range;
  ASSERT_TRUE(range.Initialize(&memory, is_64_bit));
  EXPECT_EQ(range.Is64Bit(), is_64_bit);

  // Both strings are accessible within the object's range.
  auto object_addr = FromPointerCast<VMAddress>(&kTestObject);
  EXPECT_TRUE(range.RestrictRange(object_addr, sizeof(kTestObject)));

  TestObject object;
  ASSERT_TRUE(range.Read(object_addr, sizeof(object), &object));
  EXPECT_EQ(memcmp(&object, &kTestObject, sizeof(object)), 0);

  std::string string;
  auto string1_addr = FromPointerCast<VMAddress>(kTestObject.string1);
  auto string2_addr = FromPointerCast<VMAddress>(kTestObject.string2);
  ASSERT_TRUE(range.ReadCStringSizeLimited(
      string1_addr, arraysize(kTestObject.string1), &string));
  EXPECT_STREQ(string.c_str(), kTestObject.string1);

  ASSERT_TRUE(range.ReadCStringSizeLimited(
      string2_addr, arraysize(kTestObject.string2), &string));
  EXPECT_STREQ(string.c_str(), kTestObject.string2);

  // Limit the range to remove access to string2.
  ProcessMemoryRange range2;
  ASSERT_TRUE(range2.Initialize(range));
  ASSERT_TRUE(
      range2.RestrictRange(string1_addr, arraysize(kTestObject.string1)));
  EXPECT_TRUE(range2.ReadCStringSizeLimited(
      string1_addr, arraysize(kTestObject.string1), &string));
  EXPECT_FALSE(range2.ReadCStringSizeLimited(
      string2_addr, arraysize(kTestObject.string2), &string));
  EXPECT_FALSE(range2.Read(object_addr, sizeof(object), &object));

  // String reads fail if the NUL terminator is outside the range.
  ASSERT_TRUE(range2.RestrictRange(string1_addr, strlen(kTestObject.string1)));
  EXPECT_FALSE(range2.ReadCStringSizeLimited(
      string1_addr, arraysize(kTestObject.string1), &string));

  // New range outside the old range.
  EXPECT_FALSE(range2.RestrictRange(string1_addr - 1, 1));
}

}  // namespace
}  // namespace test
}  // namespace crashpad
