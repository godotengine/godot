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

#include "util/mach/task_memory.h"

#include <mach/mach.h>
#include <string.h>

#include <algorithm>
#include <memory>
#include <string>

#include "base/mac/scoped_mach_port.h"
#include "base/mac/scoped_mach_vm.h"
#include "gtest/gtest.h"
#include "test/mac/mach_errors.h"
#include "util/misc/from_pointer_cast.h"

namespace crashpad {
namespace test {
namespace {

TEST(TaskMemory, ReadSelf) {
  vm_address_t address = 0;
  constexpr vm_size_t kSize = 4 * PAGE_SIZE;
  kern_return_t kr =
      vm_allocate(mach_task_self(), &address, kSize, VM_FLAGS_ANYWHERE);
  ASSERT_EQ(kr, KERN_SUCCESS) << MachErrorMessage(kr, "vm_allocate");
  base::mac::ScopedMachVM vm_owner(address, mach_vm_round_page(kSize));

  char* region = reinterpret_cast<char*>(address);
  for (size_t index = 0; index < kSize; ++index) {
    region[index] = (index % 256) ^ ((index >> 8) % 256);
  }

  TaskMemory memory(mach_task_self());

  // This tests using both the Read() and ReadMapped() interfaces.
  std::string result(kSize, '\0');
  std::unique_ptr<TaskMemory::MappedMemory> mapped;

  // Ensure that the entire region can be read.
  ASSERT_TRUE(memory.Read(address, kSize, &result[0]));
  EXPECT_EQ(memcmp(region, &result[0], kSize), 0);
  ASSERT_TRUE((mapped = memory.ReadMapped(address, kSize)));
  EXPECT_EQ(memcmp(region, mapped->data(), kSize), 0);

  // Ensure that a read of length 0 succeeds and doesn’t touch the result.
  result.assign(kSize, '\0');
  std::string zeroes = result;
  ASSERT_TRUE(memory.Read(address, 0, &result[0]));
  EXPECT_EQ(result, zeroes);
  ASSERT_TRUE((mapped = memory.ReadMapped(address, 0)));

  // Ensure that a read starting at an unaligned address works.
  ASSERT_TRUE(memory.Read(address + 1, kSize - 1, &result[0]));
  EXPECT_EQ(memcmp(region + 1, &result[0], kSize - 1), 0);
  ASSERT_TRUE((mapped = memory.ReadMapped(address + 1, kSize - 1)));
  EXPECT_EQ(memcmp(region + 1, mapped->data(), kSize - 1), 0);

  // Ensure that a read ending at an unaligned address works.
  ASSERT_TRUE(memory.Read(address, kSize - 1, &result[0]));
  EXPECT_EQ(memcmp(region, &result[0], kSize - 1), 0);
  ASSERT_TRUE((mapped = memory.ReadMapped(address, kSize - 1)));
  EXPECT_EQ(memcmp(region, mapped->data(), kSize - 1), 0);

  // Ensure that a read starting and ending at unaligned addresses works.
  ASSERT_TRUE(memory.Read(address + 1, kSize - 2, &result[0]));
  EXPECT_EQ(memcmp(region + 1, &result[0], kSize - 2), 0);
  ASSERT_TRUE((mapped = memory.ReadMapped(address + 1, kSize - 2)));
  EXPECT_EQ(memcmp(region + 1, mapped->data(), kSize - 2), 0);

  // Ensure that a read of exactly one page works.
  ASSERT_TRUE(memory.Read(address + PAGE_SIZE, PAGE_SIZE, &result[0]));
  EXPECT_EQ(memcmp(region + PAGE_SIZE, &result[0], PAGE_SIZE), 0);
  ASSERT_TRUE((mapped = memory.ReadMapped(address + PAGE_SIZE, PAGE_SIZE)));
  EXPECT_EQ(memcmp(region + PAGE_SIZE, mapped->data(), PAGE_SIZE), 0);

  // Ensure that a read of a single byte works.
  ASSERT_TRUE(memory.Read(address + 2, 1, &result[0]));
  EXPECT_EQ(result[0], region[2]);
  ASSERT_TRUE((mapped = memory.ReadMapped(address + 2, 1)));
  EXPECT_EQ(reinterpret_cast<const char*>(mapped->data())[0], region[2]);

  // Ensure that a read of length zero works and doesn’t touch the data.
  result[0] = 'M';
  ASSERT_TRUE(memory.Read(address + 3, 0, &result[0]));
  EXPECT_EQ(result[0], 'M');
  ASSERT_TRUE((mapped = memory.ReadMapped(address + 3, 0)));
}

TEST(TaskMemory, ReadSelfUnmapped) {
  vm_address_t address = 0;
  constexpr vm_size_t kSize = 2 * PAGE_SIZE;
  kern_return_t kr =
      vm_allocate(mach_task_self(), &address, kSize, VM_FLAGS_ANYWHERE);
  ASSERT_EQ(kr, KERN_SUCCESS) << MachErrorMessage(kr, "vm_allocate");
  base::mac::ScopedMachVM vm_owner(address, mach_vm_round_page(kSize));

  char* region = reinterpret_cast<char*>(address);
  for (size_t index = 0; index < kSize; ++index) {
    // Don’t include any NUL bytes, because ReadCString stops when it encounters
    // a NUL.
    region[index] = (index % 255) + 1;
  }

  kr = vm_protect(
      mach_task_self(), address + PAGE_SIZE, PAGE_SIZE, FALSE, VM_PROT_NONE);
  ASSERT_EQ(kr, KERN_SUCCESS) << MachErrorMessage(kr, "vm_protect");

  TaskMemory memory(mach_task_self());
  std::string result(kSize, '\0');

  EXPECT_FALSE(memory.Read(address, kSize, &result[0]));
  EXPECT_FALSE(memory.Read(address + 1, kSize - 1, &result[0]));
  EXPECT_FALSE(memory.Read(address + PAGE_SIZE, 1, &result[0]));
  EXPECT_FALSE(memory.Read(address + PAGE_SIZE - 1, 2, &result[0]));
  EXPECT_TRUE(memory.Read(address, PAGE_SIZE, &result[0]));
  EXPECT_TRUE(memory.Read(address + PAGE_SIZE - 1, 1, &result[0]));

  // Do the same thing with the ReadMapped() interface.
  std::unique_ptr<TaskMemory::MappedMemory> mapped;
  EXPECT_FALSE((mapped = memory.ReadMapped(address, kSize)));
  EXPECT_FALSE((mapped = memory.ReadMapped(address + 1, kSize - 1)));
  EXPECT_FALSE((mapped = memory.ReadMapped(address + PAGE_SIZE, 1)));
  EXPECT_FALSE((mapped = memory.ReadMapped(address + PAGE_SIZE - 1, 2)));
  EXPECT_TRUE((mapped = memory.ReadMapped(address, PAGE_SIZE)));
  EXPECT_TRUE((mapped = memory.ReadMapped(address + PAGE_SIZE - 1, 1)));

  // Repeat the test with an unmapped page instead of an unreadable one. This
  // portion of the test may be flaky in the presence of other threads, if
  // another thread maps something in the region that is deallocated here.
  kr = vm_deallocate(mach_task_self(), address + PAGE_SIZE, PAGE_SIZE);
  ASSERT_EQ(kr, KERN_SUCCESS) << MachErrorMessage(kr, "vm_deallocate");
  vm_owner.reset(address, PAGE_SIZE);

  EXPECT_FALSE(memory.Read(address, kSize, &result[0]));
  EXPECT_FALSE(memory.Read(address + 1, kSize - 1, &result[0]));
  EXPECT_FALSE(memory.Read(address + PAGE_SIZE, 1, &result[0]));
  EXPECT_FALSE(memory.Read(address + PAGE_SIZE - 1, 2, &result[0]));
  EXPECT_TRUE(memory.Read(address, PAGE_SIZE, &result[0]));
  EXPECT_TRUE(memory.Read(address + PAGE_SIZE - 1, 1, &result[0]));

  // Do the same thing with the ReadMapped() interface.
  EXPECT_FALSE((mapped = memory.ReadMapped(address, kSize)));
  EXPECT_FALSE((mapped = memory.ReadMapped(address + 1, kSize - 1)));
  EXPECT_FALSE((mapped = memory.ReadMapped(address + PAGE_SIZE, 1)));
  EXPECT_FALSE((mapped = memory.ReadMapped(address + PAGE_SIZE - 1, 2)));
  EXPECT_TRUE((mapped = memory.ReadMapped(address, PAGE_SIZE)));
  EXPECT_TRUE((mapped = memory.ReadMapped(address + PAGE_SIZE - 1, 1)));
}

// This function consolidates the cast from a char* to mach_vm_address_t in one
// location when reading from the current task.
bool ReadCStringSelf(TaskMemory* memory,
                     const char* pointer,
                     std::string* result) {
  return memory->ReadCString(FromPointerCast<mach_vm_address_t>(pointer),
                             result);
}

TEST(TaskMemory, ReadCStringSelf) {
  TaskMemory memory(mach_task_self());
  std::string result;

  const char kConstCharEmpty[] = "";
  ASSERT_TRUE(ReadCStringSelf(&memory, kConstCharEmpty, &result));
  EXPECT_TRUE(result.empty());
  EXPECT_EQ(result, kConstCharEmpty);

  const char kConstCharShort[] = "A short const char[]";
  ASSERT_TRUE(ReadCStringSelf(&memory, kConstCharShort, &result));
  EXPECT_FALSE(result.empty());
  EXPECT_EQ(result, kConstCharShort);

  static const char kStaticConstCharEmpty[] = "";
  ASSERT_TRUE(ReadCStringSelf(&memory, kStaticConstCharEmpty, &result));
  EXPECT_TRUE(result.empty());
  EXPECT_EQ(result, kStaticConstCharEmpty);

  static const char kStaticConstCharShort[] = "A short static const char[]";
  ASSERT_TRUE(ReadCStringSelf(&memory, kStaticConstCharShort, &result));
  EXPECT_FALSE(result.empty());
  EXPECT_EQ(result, kStaticConstCharShort);

  constexpr char kConstexprCharEmpty[] = "";
  ASSERT_TRUE(ReadCStringSelf(&memory, kConstexprCharEmpty, &result));
  EXPECT_TRUE(result.empty());
  EXPECT_EQ(result, kConstexprCharEmpty);

  constexpr char kConstexprCharShort[] = "A short constexpr char[]";
  ASSERT_TRUE(ReadCStringSelf(&memory, kConstexprCharShort, &result));
  EXPECT_FALSE(result.empty());
  EXPECT_EQ(result, kConstexprCharShort);

  static constexpr char kStaticConstexprCharEmpty[] = "";
  ASSERT_TRUE(ReadCStringSelf(&memory, kStaticConstexprCharEmpty, &result));
  EXPECT_TRUE(result.empty());
  EXPECT_EQ(result, kStaticConstexprCharEmpty);

  static constexpr char kStaticConstexprCharShort[] =
      "A short static constexpr char[]";
  ASSERT_TRUE(ReadCStringSelf(&memory, kStaticConstexprCharShort, &result));
  EXPECT_FALSE(result.empty());
  EXPECT_EQ(result, kStaticConstexprCharShort);

  std::string string_short("A short std::string in a function");
  ASSERT_TRUE(ReadCStringSelf(&memory, &string_short[0], &result));
  EXPECT_FALSE(result.empty());
  EXPECT_EQ(result, string_short);

  std::string string_long;
  constexpr size_t kStringLongSize = 4 * PAGE_SIZE;
  for (size_t index = 0; index < kStringLongSize; ++index) {
    // Don’t include any NUL bytes, because ReadCString stops when it encounters
    // a NUL.
    string_long.append(1, (index % 255) + 1);
  }
  ASSERT_EQ(string_long.size(), kStringLongSize);
  ASSERT_TRUE(ReadCStringSelf(&memory, &string_long[0], &result));
  EXPECT_FALSE(result.empty());
  EXPECT_EQ(result.size(), kStringLongSize);
  EXPECT_EQ(result, string_long);
}

TEST(TaskMemory, ReadCStringSelfUnmapped) {
  vm_address_t address = 0;
  constexpr vm_size_t kSize = 2 * PAGE_SIZE;
  kern_return_t kr =
      vm_allocate(mach_task_self(), &address, kSize, VM_FLAGS_ANYWHERE);
  ASSERT_EQ(kr, KERN_SUCCESS) << MachErrorMessage(kr, "vm_allocate");
  base::mac::ScopedMachVM vm_owner(address, mach_vm_round_page(kSize));

  char* region = reinterpret_cast<char*>(address);
  for (size_t index = 0; index < kSize; ++index) {
    // Don’t include any NUL bytes, because ReadCString stops when it encounters
    // a NUL.
    region[index] = (index % 255) + 1;
  }

  kr = vm_protect(
      mach_task_self(), address + PAGE_SIZE, PAGE_SIZE, FALSE, VM_PROT_NONE);
  ASSERT_EQ(kr, KERN_SUCCESS) << MachErrorMessage(kr, "vm_protect");

  TaskMemory memory(mach_task_self());
  std::string result;
  EXPECT_FALSE(memory.ReadCString(address, &result));

  // Make sure that if the string is NUL-terminated within the mapped memory
  // region, it can be read properly.
  char terminator_or_not = '\0';
  std::swap(region[PAGE_SIZE - 1], terminator_or_not);
  ASSERT_TRUE(memory.ReadCString(address, &result));
  EXPECT_FALSE(result.empty());
  EXPECT_EQ(result.size(), PAGE_SIZE - 1u);
  EXPECT_EQ(result, region);

  // Repeat the test with an unmapped page instead of an unreadable one. This
  // portion of the test may be flaky in the presence of other threads, if
  // another thread maps something in the region that is deallocated here.
  std::swap(region[PAGE_SIZE - 1], terminator_or_not);
  kr = vm_deallocate(mach_task_self(), address + PAGE_SIZE, PAGE_SIZE);
  ASSERT_EQ(kr, KERN_SUCCESS) << MachErrorMessage(kr, "vm_deallocate");
  vm_owner.reset(address, PAGE_SIZE);

  EXPECT_FALSE(memory.ReadCString(address, &result));

  // Clear the result before testing that the string can be read. This makes
  // sure that the result is actually filled in, because it already contains the
  // expected value from the tests above.
  result.clear();
  std::swap(region[PAGE_SIZE - 1], terminator_or_not);
  ASSERT_TRUE(memory.ReadCString(address, &result));
  EXPECT_FALSE(result.empty());
  EXPECT_EQ(result.size(), PAGE_SIZE - 1u);
  EXPECT_EQ(result, region);
}

// This function consolidates the cast from a char* to mach_vm_address_t in one
// location when reading from the current task.
bool ReadCStringSizeLimitedSelf(TaskMemory* memory,
                                const char* pointer,
                                size_t size,
                                std::string* result) {
  return memory->ReadCStringSizeLimited(
      FromPointerCast<mach_vm_address_t>(pointer), size, result);
}

TEST(TaskMemory, ReadCStringSizeLimited_ConstCharEmpty) {
  TaskMemory memory(mach_task_self());
  std::string result;

  static constexpr char kConstCharEmpty[] = "";
  ASSERT_TRUE(ReadCStringSizeLimitedSelf(
      &memory, kConstCharEmpty, arraysize(kConstCharEmpty), &result));
  EXPECT_TRUE(result.empty());
  EXPECT_EQ(result, kConstCharEmpty);

  result.clear();
  ASSERT_TRUE(ReadCStringSizeLimitedSelf(
      &memory, kConstCharEmpty, arraysize(kConstCharEmpty) + 1, &result));
  EXPECT_TRUE(result.empty());
  EXPECT_EQ(result, kConstCharEmpty);

  result.clear();
  ASSERT_TRUE(ReadCStringSizeLimitedSelf(&memory, kConstCharEmpty, 0, &result));
  EXPECT_TRUE(result.empty());
  EXPECT_EQ(result, kConstCharEmpty);
}

TEST(TaskMemory, ReadCStringSizeLimited_ConstCharShort) {
  TaskMemory memory(mach_task_self());
  std::string result;

  static constexpr char kConstCharShort[] = "A short const char[]";
  ASSERT_TRUE(ReadCStringSizeLimitedSelf(
      &memory, kConstCharShort, arraysize(kConstCharShort), &result));
  EXPECT_FALSE(result.empty());
  EXPECT_EQ(result, kConstCharShort);

  result.clear();
  ASSERT_TRUE(ReadCStringSizeLimitedSelf(
      &memory, kConstCharShort, arraysize(kConstCharShort) + 1, &result));
  EXPECT_FALSE(result.empty());
  EXPECT_EQ(result, kConstCharShort);

  ASSERT_FALSE(ReadCStringSizeLimitedSelf(
      &memory, kConstCharShort, arraysize(kConstCharShort) - 1, &result));
}

TEST(TaskMemory, ReadCStringSizeLimited_StaticConstCharEmpty) {
  TaskMemory memory(mach_task_self());
  std::string result;

  static constexpr char kStaticConstCharEmpty[] = "";
  ASSERT_TRUE(ReadCStringSizeLimitedSelf(&memory,
                                         kStaticConstCharEmpty,
                                         arraysize(kStaticConstCharEmpty),
                                         &result));
  EXPECT_TRUE(result.empty());
  EXPECT_EQ(result, kStaticConstCharEmpty);

  result.clear();
  ASSERT_TRUE(ReadCStringSizeLimitedSelf(&memory,
                                         kStaticConstCharEmpty,
                                         arraysize(kStaticConstCharEmpty) + 1,
                                         &result));
  EXPECT_TRUE(result.empty());
  EXPECT_EQ(result, kStaticConstCharEmpty);

  result.clear();
  ASSERT_TRUE(
      ReadCStringSizeLimitedSelf(&memory, kStaticConstCharEmpty, 0, &result));
  EXPECT_TRUE(result.empty());
  EXPECT_EQ(result, kStaticConstCharEmpty);
}

TEST(TaskMemory, ReadCStringSizeLimited_StaticConstCharShort) {
  TaskMemory memory(mach_task_self());
  std::string result;

  static constexpr char kStaticConstCharShort[] =
      "A short static constexpr char[]";
  ASSERT_TRUE(ReadCStringSizeLimitedSelf(&memory,
                                         kStaticConstCharShort,
                                         arraysize(kStaticConstCharShort),
                                         &result));
  EXPECT_FALSE(result.empty());
  EXPECT_EQ(result, kStaticConstCharShort);

  result.clear();
  ASSERT_TRUE(ReadCStringSizeLimitedSelf(&memory,
                                         kStaticConstCharShort,
                                         arraysize(kStaticConstCharShort) + 1,
                                         &result));
  EXPECT_FALSE(result.empty());
  EXPECT_EQ(result, kStaticConstCharShort);

  ASSERT_FALSE(ReadCStringSizeLimitedSelf(&memory,
                                          kStaticConstCharShort,
                                          arraysize(kStaticConstCharShort) - 1,
                                          &result));
}

TEST(TaskMemory, ReadCStringSizeLimited_StringShort) {
  TaskMemory memory(mach_task_self());
  std::string result;

  std::string string_short("A short std::string in a function");
  ASSERT_TRUE(ReadCStringSizeLimitedSelf(
      &memory, &string_short[0], string_short.size() + 1, &result));
  EXPECT_FALSE(result.empty());
  EXPECT_EQ(result, string_short);

  result.clear();
  ASSERT_TRUE(ReadCStringSizeLimitedSelf(
      &memory, &string_short[0], string_short.size() + 2, &result));
  EXPECT_FALSE(result.empty());
  EXPECT_EQ(result, string_short);

  ASSERT_FALSE(ReadCStringSizeLimitedSelf(
      &memory, &string_short[0], string_short.size(), &result));
}

TEST(TaskMemory, ReadCStringSizeLimited_StringLong) {
  TaskMemory memory(mach_task_self());
  std::string result;

  std::string string_long;
  constexpr size_t kStringLongSize = 4 * PAGE_SIZE;
  for (size_t index = 0; index < kStringLongSize; ++index) {
    // Don’t include any NUL bytes, because ReadCString stops when it encounters
    // a NUL.
    string_long.append(1, (index % 255) + 1);
  }
  ASSERT_EQ(string_long.size(), kStringLongSize);
  ASSERT_TRUE(ReadCStringSizeLimitedSelf(
      &memory, &string_long[0], string_long.size() + 1, &result));
  EXPECT_FALSE(result.empty());
  EXPECT_EQ(result.size(), kStringLongSize);
  EXPECT_EQ(result, string_long);

  result.clear();
  ASSERT_TRUE(ReadCStringSizeLimitedSelf(
      &memory, &string_long[0], string_long.size() + 2, &result));
  EXPECT_FALSE(result.empty());
  EXPECT_EQ(result.size(), kStringLongSize);
  EXPECT_EQ(result, string_long);

  ASSERT_FALSE(ReadCStringSizeLimitedSelf(
      &memory, &string_long[0], string_long.size(), &result));
}

bool IsAddressMapped(vm_address_t address) {
  vm_address_t region_address = address;
  vm_size_t region_size;
  mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
  vm_region_basic_info_64 info;
  mach_port_t object;
  kern_return_t kr = vm_region_64(mach_task_self(),
                                  &region_address,
                                  &region_size,
                                  VM_REGION_BASIC_INFO_64,
                                  reinterpret_cast<vm_region_info_t>(&info),
                                  &count,
                                  &object);
  if (kr == KERN_SUCCESS) {
    // |object| will be MACH_PORT_NULL (10.9.4 xnu-2422.110.17/osfmk/vm/vm_map.c
    // vm_map_region()), but the interface acts as if it might carry a send
    // right, so treat it as documented.
    base::mac::ScopedMachSendRight object_owner(object);

    return address >= region_address && address <= region_address + region_size;
  }

  if (kr == KERN_INVALID_ADDRESS) {
    return false;
  }

  ADD_FAILURE() << MachErrorMessage(kr, "vm_region_64");
  return false;
}

TEST(TaskMemory, MappedMemoryDeallocates) {
  // This tests that once a TaskMemory::MappedMemory object is destroyed, it
  // releases the mapped memory that it owned. Technically, this test is not
  // valid because after the mapping is released, something else (on another
  // thread) might wind up mapped in the same address. In the test environment,
  // hopefully there are either no other threads or they’re all quiescent, so
  // nothing else should wind up mapped in the address.

  TaskMemory memory(mach_task_self());
  std::unique_ptr<TaskMemory::MappedMemory> mapped;

  static constexpr char kTestBuffer[] = "hello!";
  mach_vm_address_t test_address =
      FromPointerCast<mach_vm_address_t>(&kTestBuffer);
  ASSERT_TRUE((mapped = memory.ReadMapped(test_address, sizeof(kTestBuffer))));
  EXPECT_EQ(memcmp(kTestBuffer, mapped->data(), sizeof(kTestBuffer)), 0);

  vm_address_t mapped_address = reinterpret_cast<vm_address_t>(mapped->data());
  EXPECT_TRUE(IsAddressMapped(mapped_address));

  mapped.reset();
  EXPECT_FALSE(IsAddressMapped(mapped_address));

  // This is the same but with a big buffer that’s definitely larger than a
  // single page. This makes sure that the whole mapped region winds up being
  // deallocated.
  constexpr size_t kBigSize = 4 * PAGE_SIZE;
  std::unique_ptr<char[]> big_buffer(new char[kBigSize]);
  test_address = FromPointerCast<mach_vm_address_t>(&big_buffer[0]);
  ASSERT_TRUE((mapped = memory.ReadMapped(test_address, kBigSize)));

  mapped_address = reinterpret_cast<vm_address_t>(mapped->data());
  vm_address_t mapped_last_address = mapped_address + kBigSize - 1;
  EXPECT_TRUE(IsAddressMapped(mapped_address));
  EXPECT_TRUE(IsAddressMapped(mapped_address + PAGE_SIZE));
  EXPECT_TRUE(IsAddressMapped(mapped_last_address));

  mapped.reset();
  EXPECT_FALSE(IsAddressMapped(mapped_address));
  EXPECT_FALSE(IsAddressMapped(mapped_address + PAGE_SIZE));
  EXPECT_FALSE(IsAddressMapped(mapped_last_address));
}

TEST(TaskMemory, MappedMemoryReadCString) {
  // This tests the behavior of TaskMemory::MappedMemory::ReadCString().
  TaskMemory memory(mach_task_self());
  std::unique_ptr<TaskMemory::MappedMemory> mapped;

  static constexpr char kTestBuffer[] = "0\0" "2\0" "45\0" "789";
  const mach_vm_address_t kTestAddress =
      FromPointerCast<mach_vm_address_t>(&kTestBuffer);
  ASSERT_TRUE((mapped = memory.ReadMapped(kTestAddress, 10)));

  std::string string;
  ASSERT_TRUE(mapped->ReadCString(0, &string));
  EXPECT_EQ(string, "0");
  ASSERT_TRUE(mapped->ReadCString(1, &string));
  EXPECT_EQ(string, "");
  ASSERT_TRUE(mapped->ReadCString(2, &string));
  EXPECT_EQ(string, "2");
  ASSERT_TRUE(mapped->ReadCString(3, &string));
  EXPECT_EQ(string, "");
  ASSERT_TRUE(mapped->ReadCString(4, &string));
  EXPECT_EQ(string, "45");
  ASSERT_TRUE(mapped->ReadCString(5, &string));
  EXPECT_EQ(string, "5");
  ASSERT_TRUE(mapped->ReadCString(6, &string));
  EXPECT_EQ(string, "");

  // kTestBuffer’s NUL terminator was not read, so these will see an
  // unterminated string and fail.
  EXPECT_FALSE(mapped->ReadCString(7, &string));
  EXPECT_FALSE(mapped->ReadCString(8, &string));
  EXPECT_FALSE(mapped->ReadCString(9, &string));

  // This is out of the range of what was read, so it will fail.
  EXPECT_FALSE(mapped->ReadCString(10, &string));
  EXPECT_FALSE(mapped->ReadCString(11, &string));

  // Read it again, this time with a length long enough to include the NUL
  // terminator.
  ASSERT_TRUE((mapped = memory.ReadMapped(kTestAddress, 11)));

  ASSERT_TRUE(mapped->ReadCString(6, &string));
  EXPECT_EQ(string, "");

  // These should now succeed.
  ASSERT_TRUE(mapped->ReadCString(7, &string));
  EXPECT_EQ(string, "789");
  ASSERT_TRUE(mapped->ReadCString(8, &string));
  EXPECT_EQ(string, "89");
  ASSERT_TRUE(mapped->ReadCString(9, &string));
  EXPECT_EQ(string, "9");
  EXPECT_TRUE(mapped->ReadCString(10, &string));
  EXPECT_EQ(string, "");

  // These are still out of range.
  EXPECT_FALSE(mapped->ReadCString(11, &string));
  EXPECT_FALSE(mapped->ReadCString(12, &string));
}

}  // namespace
}  // namespace test
}  // namespace crashpad
