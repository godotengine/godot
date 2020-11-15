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

#include "util/process/process_memory.h"

#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include <memory>

#include "gtest/gtest.h"
#include "test/errors.h"
#include "test/multiprocess.h"
#include "test/multiprocess_exec.h"
#include "test/process_type.h"
#include "util/file/file_io.h"
#include "util/misc/from_pointer_cast.h"
#include "util/posix/scoped_mmap.h"
#include "util/process/process_memory_native.h"

namespace crashpad {
namespace test {
namespace {

void DoChildReadTestSetup(size_t* region_size,
                          std::unique_ptr<char[]>* region) {
  *region_size = 4 * getpagesize();
  region->reset(new char[*region_size]);
  for (size_t index = 0; index < *region_size; ++index) {
    (*region)[index] = index % 256;
  }
}

CRASHPAD_CHILD_TEST_MAIN(ReadTestChild) {
  size_t region_size;
  std::unique_ptr<char[]> region;
  DoChildReadTestSetup(&region_size, &region);
  FileHandle out = StdioFileHandle(StdioStream::kStandardOutput);
  CheckedWriteFile(out, &region_size, sizeof(region_size));
  VMAddress address = FromPointerCast<VMAddress>(region.get());
  CheckedWriteFile(out, &address, sizeof(address));
  CheckedReadFileAtEOF(StdioFileHandle(StdioStream::kStandardInput));
  return 0;
}

class ReadTest : public MultiprocessExec {
 public:
  ReadTest() : MultiprocessExec() {
    SetChildTestMainFunction("ReadTestChild");
  }

  void RunAgainstSelf() {
    size_t region_size;
    std::unique_ptr<char[]> region;
    DoChildReadTestSetup(&region_size, &region);
    DoTest(GetSelfProcess(),
           region_size,
           FromPointerCast<VMAddress>(region.get()));
  }

  void RunAgainstChild() { Run(); }

 private:
  void MultiprocessParent() override {
    size_t region_size;
    VMAddress region;
    ASSERT_TRUE(
        ReadFileExactly(ReadPipeHandle(), &region_size, sizeof(region_size)));
    ASSERT_TRUE(ReadFileExactly(ReadPipeHandle(), &region, sizeof(region)));
    DoTest(ChildProcess(), region_size, region);
  }

  void DoTest(ProcessType process, size_t region_size, VMAddress address) {
    ProcessMemoryNative memory;
    ASSERT_TRUE(memory.Initialize(process));

    std::unique_ptr<char[]> result(new char[region_size]);

    // Ensure that the entire region can be read.
    ASSERT_TRUE(memory.Read(address, region_size, result.get()));
    for (size_t i = 0; i < region_size; ++i) {
      EXPECT_EQ(result[i], static_cast<char>(i % 256));
    }

    // Ensure that a read of length 0 succeeds and doesn’t touch the result.
    memset(result.get(), '\0', region_size);
    ASSERT_TRUE(memory.Read(address, 0, result.get()));
    for (size_t i = 0; i < region_size; ++i) {
      EXPECT_EQ(result[i], 0);
    }

    // Ensure that a read starting at an unaligned address works.
    ASSERT_TRUE(memory.Read(address + 1, region_size - 1, result.get()));
    for (size_t i = 0; i < region_size - 1; ++i) {
      EXPECT_EQ(result[i], static_cast<char>((i + 1) % 256));
    }

    // Ensure that a read ending at an unaligned address works.
    ASSERT_TRUE(memory.Read(address, region_size - 1, result.get()));
    for (size_t i = 0; i < region_size - 1; ++i) {
      EXPECT_EQ(result[i], static_cast<char>(i % 256));
    }

    // Ensure that a read starting and ending at unaligned addresses works.
    ASSERT_TRUE(memory.Read(address + 1, region_size - 2, result.get()));
    for (size_t i = 0; i < region_size - 2; ++i) {
      EXPECT_EQ(result[i], static_cast<char>((i + 1) % 256));
    }

    // Ensure that a read of exactly one page works.
    size_t page_size = getpagesize();
    ASSERT_GE(region_size, page_size + page_size);
    ASSERT_TRUE(memory.Read(address + page_size, page_size, result.get()));
    for (size_t i = 0; i < page_size; ++i) {
      EXPECT_EQ(result[i], static_cast<char>((i + page_size) % 256));
    }

    // Ensure that reading exactly a single byte works.
    result[1] = 'J';
    ASSERT_TRUE(memory.Read(address + 2, 1, result.get()));
    EXPECT_EQ(result[0], 2);
    EXPECT_EQ(result[1], 'J');
  }

  DISALLOW_COPY_AND_ASSIGN(ReadTest);
};

TEST(ProcessMemory, ReadSelf) {
  ReadTest test;
  test.RunAgainstSelf();
}

TEST(ProcessMemory, ReadChild) {
  ReadTest test;
  test.RunAgainstChild();
}

constexpr char kConstCharEmpty[] = "";
constexpr char kConstCharShort[] = "A short const char[]";

#define SHORT_LOCAL_STRING "A short local variable char[]"

std::string MakeLongString() {
  std::string long_string;
  const size_t kStringLongSize = 4 * getpagesize();
  for (size_t index = 0; index < kStringLongSize; ++index) {
    long_string.push_back((index % 255) + 1);
  }
  EXPECT_EQ(long_string.size(), kStringLongSize);
  return long_string;
}

void DoChildCStringReadTestSetup(const char** const_empty,
                                 const char** const_short,
                                 const char** local_empty,
                                 const char** local_short,
                                 std::string* long_string) {
  *const_empty = kConstCharEmpty;
  *const_short = kConstCharShort;
  *local_empty = "";
  *local_short = SHORT_LOCAL_STRING;
  *long_string = MakeLongString();
}

CRASHPAD_CHILD_TEST_MAIN(ReadCStringTestChild) {
  const char* const_empty;
  const char* const_short;
  const char* local_empty;
  const char* local_short;
  std::string long_string;
  DoChildCStringReadTestSetup(
      &const_empty, &const_short, &local_empty, &local_short, &long_string);
  const auto write_address = [](const char* p) {
    VMAddress address = FromPointerCast<VMAddress>(p);
    CheckedWriteFile(StdioFileHandle(StdioStream::kStandardOutput),
                     &address,
                     sizeof(address));
  };
  write_address(const_empty);
  write_address(const_short);
  write_address(local_empty);
  write_address(local_short);
  write_address(long_string.c_str());
  CheckedReadFileAtEOF(StdioFileHandle(StdioStream::kStandardInput));
  return 0;
}

class ReadCStringTest : public MultiprocessExec {
 public:
  ReadCStringTest(bool limit_size)
      : MultiprocessExec(), limit_size_(limit_size) {
    SetChildTestMainFunction("ReadCStringTestChild");
  }

  void RunAgainstSelf() {
    const char* const_empty;
    const char* const_short;
    const char* local_empty;
    const char* local_short;
    std::string long_string;
    DoChildCStringReadTestSetup(
        &const_empty, &const_short, &local_empty, &local_short, &long_string);
    DoTest(GetSelfProcess(),
           FromPointerCast<VMAddress>(const_empty),
           FromPointerCast<VMAddress>(const_short),
           FromPointerCast<VMAddress>(local_empty),
           FromPointerCast<VMAddress>(local_short),
           FromPointerCast<VMAddress>(long_string.c_str()));
  }
  void RunAgainstChild() { Run(); }

 private:
  void MultiprocessParent() override {
#define DECLARE_AND_READ_ADDRESS(name) \
  VMAddress name;                      \
  ASSERT_TRUE(ReadFileExactly(ReadPipeHandle(), &name, sizeof(name)));
    DECLARE_AND_READ_ADDRESS(const_empty_address);
    DECLARE_AND_READ_ADDRESS(const_short_address);
    DECLARE_AND_READ_ADDRESS(local_empty_address);
    DECLARE_AND_READ_ADDRESS(local_short_address);
    DECLARE_AND_READ_ADDRESS(long_string_address);
#undef DECLARE_AND_READ_ADDRESS

    DoTest(ChildProcess(),
           const_empty_address,
           const_short_address,
           local_empty_address,
           local_short_address,
           long_string_address);
  }

  void DoTest(ProcessType process,
              VMAddress const_empty_address,
              VMAddress const_short_address,
              VMAddress local_empty_address,
              VMAddress local_short_address,
              VMAddress long_string_address) {
    ProcessMemoryNative memory;
    ASSERT_TRUE(memory.Initialize(process));

    std::string result;

    if (limit_size_) {
      ASSERT_TRUE(memory.ReadCStringSizeLimited(
          const_empty_address, arraysize(kConstCharEmpty), &result));
      EXPECT_EQ(result, kConstCharEmpty);

      ASSERT_TRUE(memory.ReadCStringSizeLimited(
          const_short_address, arraysize(kConstCharShort), &result));
      EXPECT_EQ(result, kConstCharShort);
      EXPECT_FALSE(memory.ReadCStringSizeLimited(
          const_short_address, arraysize(kConstCharShort) - 1, &result));

      ASSERT_TRUE(
          memory.ReadCStringSizeLimited(local_empty_address, 1, &result));
      EXPECT_EQ(result, "");

      ASSERT_TRUE(memory.ReadCStringSizeLimited(
          local_short_address, strlen(SHORT_LOCAL_STRING) + 1, &result));
      EXPECT_EQ(result, SHORT_LOCAL_STRING);
      EXPECT_FALSE(memory.ReadCStringSizeLimited(
          local_short_address, strlen(SHORT_LOCAL_STRING), &result));

      std::string long_string_for_comparison = MakeLongString();
      ASSERT_TRUE(memory.ReadCStringSizeLimited(
          long_string_address, long_string_for_comparison.size() + 1, &result));
      EXPECT_EQ(result, long_string_for_comparison);
      EXPECT_FALSE(memory.ReadCStringSizeLimited(
          long_string_address, long_string_for_comparison.size(), &result));
    } else {
      ASSERT_TRUE(memory.ReadCString(const_empty_address, &result));
      EXPECT_EQ(result, kConstCharEmpty);

      ASSERT_TRUE(memory.ReadCString(const_short_address, &result));
      EXPECT_EQ(result, kConstCharShort);

      ASSERT_TRUE(memory.ReadCString(local_empty_address, &result));
      EXPECT_EQ(result, "");

      ASSERT_TRUE(memory.ReadCString(local_short_address, &result));
      EXPECT_EQ(result, SHORT_LOCAL_STRING);

      ASSERT_TRUE(memory.ReadCString(long_string_address, &result));
      EXPECT_EQ(result, MakeLongString());
    }
  }

  const bool limit_size_;

  DISALLOW_COPY_AND_ASSIGN(ReadCStringTest);
};

TEST(ProcessMemory, ReadCStringSelf) {
  ReadCStringTest test(/* limit_size= */ false);
  test.RunAgainstSelf();
}

TEST(ProcessMemory, ReadCStringChild) {
  ReadCStringTest test(/* limit_size= */ false);
  test.RunAgainstChild();
}

TEST(ProcessMemory, ReadCStringSizeLimitedSelf) {
  ReadCStringTest test(/* limit_size= */ true);
  test.RunAgainstSelf();
}

TEST(ProcessMemory, ReadCStringSizeLimitedChild) {
  ReadCStringTest test(/* limit_size= */ true);
  test.RunAgainstChild();
}

void DoReadUnmappedChildMainSetup(ScopedMmap* pages,
                                  VMAddress* address,
                                  size_t* page_size,
                                  size_t* region_size) {
  *page_size = getpagesize();
  *region_size = 2 * (*page_size);
  if (!pages->ResetMmap(nullptr,
                        *region_size,
                        PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS,
                        -1,
                        0)) {
    ADD_FAILURE();
    return;
  }

  *address = pages->addr_as<VMAddress>();

  char* region = pages->addr_as<char*>();
  for (size_t index = 0; index < *region_size; ++index) {
    region[index] = index % 256;
  }

  EXPECT_TRUE(pages->ResetAddrLen(region, *page_size));
}

CRASHPAD_CHILD_TEST_MAIN(ReadUnmappedChildMain) {
  ScopedMmap pages;
  VMAddress address = 0;
  size_t page_size, region_size;
  DoReadUnmappedChildMainSetup(&pages, &address, &page_size, &region_size);
  FileHandle out = StdioFileHandle(StdioStream::kStandardOutput);
  CheckedWriteFile(out, &address, sizeof(address));
  CheckedWriteFile(out, &page_size, sizeof(page_size));
  CheckedWriteFile(out, &region_size, sizeof(region_size));
  CheckedReadFileAtEOF(StdioFileHandle(StdioStream::kStandardInput));
  return 0;
}

class ReadUnmappedTest : public MultiprocessExec {
 public:
  ReadUnmappedTest() : MultiprocessExec() {
    SetChildTestMainFunction("ReadUnmappedChildMain");
  }

  void RunAgainstSelf() {
    ScopedMmap pages;
    VMAddress address = 0;
    size_t page_size, region_size;
    DoReadUnmappedChildMainSetup(&pages, &address, &page_size, &region_size);
    DoTest(GetSelfProcess(), address, page_size, region_size);
  }

  void RunAgainstChild() { Run(); }

 private:
  void MultiprocessParent() override {
    VMAddress address = 0;
    size_t page_size, region_size;
    ASSERT_TRUE(ReadFileExactly(ReadPipeHandle(), &address, sizeof(address)));
    ASSERT_TRUE(
        ReadFileExactly(ReadPipeHandle(), &page_size, sizeof(page_size)));
    ASSERT_TRUE(
        ReadFileExactly(ReadPipeHandle(), &region_size, sizeof(region_size)));
    DoTest(ChildProcess(), address, page_size, region_size);
  }

  void DoTest(ProcessType process,
              VMAddress address,
              size_t page_size,
              size_t region_size) {
    ProcessMemoryNative memory;
    ASSERT_TRUE(memory.Initialize(process));

    VMAddress page_addr1 = address;
    VMAddress page_addr2 = page_addr1 + page_size;

    std::unique_ptr<char[]> result(new char[region_size]);
    EXPECT_TRUE(memory.Read(page_addr1, page_size, result.get()));
    EXPECT_TRUE(memory.Read(page_addr2 - 1, 1, result.get()));

    EXPECT_FALSE(memory.Read(page_addr1, region_size, result.get()));
    EXPECT_FALSE(memory.Read(page_addr2, page_size, result.get()));
    EXPECT_FALSE(memory.Read(page_addr2 - 1, 2, result.get()));
  }

  DISALLOW_COPY_AND_ASSIGN(ReadUnmappedTest);
};

TEST(ProcessMemory, ReadUnmappedSelf) {
  ReadUnmappedTest test;
  ASSERT_FALSE(testing::Test::HasFailure());
  test.RunAgainstSelf();
}

TEST(ProcessMemory, ReadUnmappedChild) {
  ReadUnmappedTest test;
  ASSERT_FALSE(testing::Test::HasFailure());
  test.RunAgainstChild();
}

constexpr size_t kChildProcessStringLength = 10;

class StringDataInChildProcess {
 public:
  // This constructor only makes sense in the child process.
  explicit StringDataInChildProcess(const char* cstring)
      : address_(FromPointerCast<VMAddress>(cstring)) {
    memcpy(expected_value_, cstring, kChildProcessStringLength + 1);
  }

  void Write(FileHandle out) {
    CheckedWriteFile(out, &address_, sizeof(address_));
    CheckedWriteFile(out, &expected_value_, sizeof(expected_value_));
  }

  static StringDataInChildProcess Read(FileHandle in) {
    StringDataInChildProcess str;
    EXPECT_TRUE(ReadFileExactly(in, &str.address_, sizeof(str.address_)));
    EXPECT_TRUE(
        ReadFileExactly(in, &str.expected_value_, sizeof(str.expected_value_)));
    return str;
  }

  VMAddress address() const { return address_; }
  std::string expected_value() const { return expected_value_; }

  private:
   StringDataInChildProcess() : address_(0), expected_value_() {}

   VMAddress address_;
   char expected_value_[kChildProcessStringLength + 1];
};

void DoCStringUnmappedTestSetup(
    ScopedMmap* pages,
    std::vector<StringDataInChildProcess>* strings) {
  const size_t page_size = getpagesize();
  const size_t region_size = 2 * page_size;
  if (!pages->ResetMmap(nullptr,
                        region_size,
                        PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS,
                        -1,
                        0)) {
    ADD_FAILURE();
    return;
  }

  char* region = pages->addr_as<char*>();
  for (size_t index = 0; index < region_size; ++index) {
    region[index] = 1 + index % 255;
  }

  // A string at the start of the mapped region
  char* string1 = region;
  string1[kChildProcessStringLength] = '\0';

  // A string near the end of the mapped region
  char* string2 = region + page_size - kChildProcessStringLength * 2;
  string2[kChildProcessStringLength] = '\0';

  // A string that crosses from the mapped into the unmapped region
  char* string3 = region + page_size - kChildProcessStringLength + 1;
  string3[kChildProcessStringLength] = '\0';

  // A string entirely in the unmapped region
  char* string4 = region + page_size + 10;
  string4[kChildProcessStringLength] = '\0';

  strings->push_back(StringDataInChildProcess(string1));
  strings->push_back(StringDataInChildProcess(string2));
  strings->push_back(StringDataInChildProcess(string3));
  strings->push_back(StringDataInChildProcess(string4));

  EXPECT_TRUE(pages->ResetAddrLen(region, page_size));
}

CRASHPAD_CHILD_TEST_MAIN(ReadCStringUnmappedChildMain) {
  ScopedMmap pages;
  std::vector<StringDataInChildProcess> strings;
  DoCStringUnmappedTestSetup(&pages, &strings);
  FileHandle out = StdioFileHandle(StdioStream::kStandardOutput);
  strings[0].Write(out);
  strings[1].Write(out);
  strings[2].Write(out);
  strings[3].Write(out);
  CheckedReadFileAtEOF(StdioFileHandle(StdioStream::kStandardInput));
  return 0;
}

class ReadCStringUnmappedTest : public MultiprocessExec {
 public:
  ReadCStringUnmappedTest(bool limit_size)
      : MultiprocessExec(), limit_size_(limit_size) {
    SetChildTestMainFunction("ReadCStringUnmappedChildMain");
  }

  void RunAgainstSelf() {
    ScopedMmap pages;
    std::vector<StringDataInChildProcess> strings;
    DoCStringUnmappedTestSetup(&pages, &strings);
    DoTest(GetSelfProcess(), strings);
  }

  void RunAgainstChild() { Run(); }

 private:
  void MultiprocessParent() override {
    std::vector<StringDataInChildProcess> strings;
    strings.push_back(StringDataInChildProcess::Read(ReadPipeHandle()));
    strings.push_back(StringDataInChildProcess::Read(ReadPipeHandle()));
    strings.push_back(StringDataInChildProcess::Read(ReadPipeHandle()));
    strings.push_back(StringDataInChildProcess::Read(ReadPipeHandle()));
    ASSERT_NO_FATAL_FAILURE();
    DoTest(ChildProcess(), strings);
  }

  void DoTest(ProcessType process,
              const std::vector<StringDataInChildProcess>& strings) {
    ProcessMemoryNative memory;
    ASSERT_TRUE(memory.Initialize(process));

    std::string result;
    result.reserve(kChildProcessStringLength + 1);

    if (limit_size_) {
      ASSERT_TRUE(memory.ReadCStringSizeLimited(
          strings[0].address(), kChildProcessStringLength + 1, &result));
      EXPECT_EQ(result, strings[0].expected_value());
      ASSERT_TRUE(memory.ReadCStringSizeLimited(
          strings[1].address(), kChildProcessStringLength + 1, &result));
      EXPECT_EQ(result, strings[1].expected_value());
      EXPECT_FALSE(memory.ReadCStringSizeLimited(
          strings[2].address(), kChildProcessStringLength + 1, &result));
      EXPECT_FALSE(memory.ReadCStringSizeLimited(
          strings[3].address(), kChildProcessStringLength + 1, &result));
    } else {
      ASSERT_TRUE(memory.ReadCString(strings[0].address(), &result));
      EXPECT_EQ(result, strings[0].expected_value());
      ASSERT_TRUE(memory.ReadCString(strings[1].address(), &result));
      EXPECT_EQ(result, strings[1].expected_value());
      EXPECT_FALSE(memory.ReadCString(strings[2].address(), &result));
      EXPECT_FALSE(memory.ReadCString(strings[3].address(), &result));
    }
  }

  const bool limit_size_;

  DISALLOW_COPY_AND_ASSIGN(ReadCStringUnmappedTest);
};

TEST(ProcessMemory, ReadCStringUnmappedSelf) {
  ReadCStringUnmappedTest test(/* limit_size= */ false);
  ASSERT_FALSE(testing::Test::HasFailure());
  test.RunAgainstSelf();
}

TEST(ProcessMemory, ReadCStringUnmappedChild) {
  ReadCStringUnmappedTest test(/* limit_size= */ false);
  ASSERT_FALSE(testing::Test::HasFailure());
  test.RunAgainstChild();
}

TEST(ProcessMemory, ReadCStringSizeLimitedUnmappedSelf) {
  ReadCStringUnmappedTest test(/* limit_size= */ true);
  ASSERT_FALSE(testing::Test::HasFailure());
  test.RunAgainstSelf();
}

TEST(ProcessMemory, ReadCStringSizeLimitedUnmappedChild) {
  ReadCStringUnmappedTest test(/* limit_size= */ true);
  ASSERT_FALSE(testing::Test::HasFailure());
  test.RunAgainstChild();
}

}  // namespace
}  // namespace test
}  // namespace crashpad
