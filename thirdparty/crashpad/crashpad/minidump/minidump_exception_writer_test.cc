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

#include "minidump/minidump_exception_writer.h"

#include <string>
#include <utility>

#include "gtest/gtest.h"
#include "minidump/minidump_context.h"
#include "minidump/minidump_context_writer.h"
#include "minidump/minidump_extensions.h"
#include "minidump/minidump_file_writer.h"
#include "minidump/test/minidump_context_test_util.h"
#include "minidump/test/minidump_file_writer_test_util.h"
#include "minidump/test/minidump_writable_test_util.h"
#include "snapshot/test/test_cpu_context.h"
#include "snapshot/test/test_exception_snapshot.h"
#include "test/gtest_death.h"
#include "util/file/string_file.h"

namespace crashpad {
namespace test {
namespace {

// This returns the MINIDUMP_EXCEPTION_STREAM stream in |exception_stream|.
void GetExceptionStream(const std::string& file_contents,
                        const MINIDUMP_EXCEPTION_STREAM** exception_stream) {
  constexpr size_t kDirectoryOffset = sizeof(MINIDUMP_HEADER);
  constexpr size_t kExceptionStreamOffset =
      kDirectoryOffset + sizeof(MINIDUMP_DIRECTORY);
  constexpr size_t kContextOffset =
      kExceptionStreamOffset + sizeof(MINIDUMP_EXCEPTION_STREAM);
  constexpr size_t kFileSize = kContextOffset + sizeof(MinidumpContextX86);
  ASSERT_EQ(kFileSize, file_contents.size());

  const MINIDUMP_DIRECTORY* directory;
  const MINIDUMP_HEADER* header =
      MinidumpHeaderAtStart(file_contents, &directory);
  ASSERT_NO_FATAL_FAILURE(VerifyMinidumpHeader(header, 1, 0));

  ASSERT_EQ(directory[0].StreamType, kMinidumpStreamTypeException);
  EXPECT_EQ(directory[0].Location.Rva, kExceptionStreamOffset);

  *exception_stream =
      MinidumpWritableAtLocationDescriptor<MINIDUMP_EXCEPTION_STREAM>(
          file_contents, directory[0].Location);
  ASSERT_TRUE(exception_stream);
}

// The MINIDUMP_EXCEPTION_STREAMs |expected| and |observed| are compared against
// each other using gtest assertions. The context will be recovered from
// |file_contents| and stored in |context|.
void ExpectExceptionStream(const MINIDUMP_EXCEPTION_STREAM* expected,
                           const MINIDUMP_EXCEPTION_STREAM* observed,
                           const std::string& file_contents,
                           const MinidumpContextX86** context) {
  EXPECT_EQ(observed->ThreadId, expected->ThreadId);
  EXPECT_EQ(observed->__alignment, 0u);
  EXPECT_EQ(observed->ExceptionRecord.ExceptionCode,
            expected->ExceptionRecord.ExceptionCode);
  EXPECT_EQ(observed->ExceptionRecord.ExceptionFlags,
            expected->ExceptionRecord.ExceptionFlags);
  EXPECT_EQ(observed->ExceptionRecord.ExceptionRecord,
            expected->ExceptionRecord.ExceptionRecord);
  EXPECT_EQ(observed->ExceptionRecord.ExceptionAddress,
            expected->ExceptionRecord.ExceptionAddress);
  EXPECT_EQ(observed->ExceptionRecord.NumberParameters,
            expected->ExceptionRecord.NumberParameters);
  EXPECT_EQ(observed->ExceptionRecord.__unusedAlignment, 0u);
  for (size_t index = 0;
       index < arraysize(observed->ExceptionRecord.ExceptionInformation);
       ++index) {
    EXPECT_EQ(observed->ExceptionRecord.ExceptionInformation[index],
              expected->ExceptionRecord.ExceptionInformation[index]);
  }
  *context = MinidumpWritableAtLocationDescriptor<MinidumpContextX86>(
      file_contents, observed->ThreadContext);
  ASSERT_TRUE(context);
}

TEST(MinidumpExceptionWriter, Minimal) {
  MinidumpFileWriter minidump_file_writer;
  auto exception_writer = std::make_unique<MinidumpExceptionWriter>();

  constexpr uint32_t kSeed = 100;

  auto context_x86_writer = std::make_unique<MinidumpContextX86Writer>();
  InitializeMinidumpContextX86(context_x86_writer->context(), kSeed);
  exception_writer->SetContext(std::move(context_x86_writer));

  ASSERT_TRUE(minidump_file_writer.AddStream(std::move(exception_writer)));

  StringFile string_file;
  ASSERT_TRUE(minidump_file_writer.WriteEverything(&string_file));

  const MINIDUMP_EXCEPTION_STREAM* observed_exception_stream = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      GetExceptionStream(string_file.string(), &observed_exception_stream));

  MINIDUMP_EXCEPTION_STREAM expected_exception_stream = {};
  expected_exception_stream.ThreadContext.DataSize = sizeof(MinidumpContextX86);

  const MinidumpContextX86* observed_context = nullptr;
  ASSERT_NO_FATAL_FAILURE(ExpectExceptionStream(&expected_exception_stream,
                                                observed_exception_stream,
                                                string_file.string(),
                                                &observed_context));

  ASSERT_NO_FATAL_FAILURE(
      ExpectMinidumpContextX86(kSeed, observed_context, false));
}

TEST(MinidumpExceptionWriter, Standard) {
  MinidumpFileWriter minidump_file_writer;
  auto exception_writer = std::make_unique<MinidumpExceptionWriter>();

  constexpr uint32_t kSeed = 200;
  constexpr uint32_t kThreadID = 1;
  constexpr uint32_t kExceptionCode = 2;
  constexpr uint32_t kExceptionFlags = 3;
  constexpr uint32_t kExceptionRecord = 4;
  constexpr uint32_t kExceptionAddress = 5;
  constexpr uint64_t kExceptionInformation0 = 6;
  constexpr uint64_t kExceptionInformation1 = 7;
  constexpr uint64_t kExceptionInformation2 = 7;

  auto context_x86_writer = std::make_unique<MinidumpContextX86Writer>();
  InitializeMinidumpContextX86(context_x86_writer->context(), kSeed);
  exception_writer->SetContext(std::move(context_x86_writer));

  exception_writer->SetThreadID(kThreadID);
  exception_writer->SetExceptionCode(kExceptionCode);
  exception_writer->SetExceptionFlags(kExceptionFlags);
  exception_writer->SetExceptionRecord(kExceptionRecord);
  exception_writer->SetExceptionAddress(kExceptionAddress);

  // Set a lot of exception information at first, and then replace it with less.
  // This tests that the exception that is written does not contain the
  // “garbage” from the initial SetExceptionInformation() call.
  std::vector<uint64_t> exception_information(EXCEPTION_MAXIMUM_PARAMETERS,
                                              0x5a5a5a5a5a5a5a5a);
  exception_writer->SetExceptionInformation(exception_information);

  exception_information.clear();
  exception_information.push_back(kExceptionInformation0);
  exception_information.push_back(kExceptionInformation1);
  exception_information.push_back(kExceptionInformation2);
  exception_writer->SetExceptionInformation(exception_information);

  ASSERT_TRUE(minidump_file_writer.AddStream(std::move(exception_writer)));

  StringFile string_file;
  ASSERT_TRUE(minidump_file_writer.WriteEverything(&string_file));

  const MINIDUMP_EXCEPTION_STREAM* observed_exception_stream = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      GetExceptionStream(string_file.string(), &observed_exception_stream));

  MINIDUMP_EXCEPTION_STREAM expected_exception_stream = {};
  expected_exception_stream.ThreadId = kThreadID;
  expected_exception_stream.ExceptionRecord.ExceptionCode = kExceptionCode;
  expected_exception_stream.ExceptionRecord.ExceptionFlags = kExceptionFlags;
  expected_exception_stream.ExceptionRecord.ExceptionRecord = kExceptionRecord;
  expected_exception_stream.ExceptionRecord.ExceptionAddress =
      kExceptionAddress;
  expected_exception_stream.ExceptionRecord.NumberParameters =
      static_cast<uint32_t>(exception_information.size());
  for (size_t index = 0; index < exception_information.size(); ++index) {
    expected_exception_stream.ExceptionRecord.ExceptionInformation[index] =
        exception_information[index];
  }
  expected_exception_stream.ThreadContext.DataSize = sizeof(MinidumpContextX86);

  const MinidumpContextX86* observed_context = nullptr;
  ASSERT_NO_FATAL_FAILURE(ExpectExceptionStream(&expected_exception_stream,
                                                observed_exception_stream,
                                                string_file.string(),
                                                &observed_context));

  ASSERT_NO_FATAL_FAILURE(
      ExpectMinidumpContextX86(kSeed, observed_context, false));
}

TEST(MinidumpExceptionWriter, InitializeFromSnapshot) {
  std::vector<uint64_t> exception_codes;
  exception_codes.push_back(0x1000000000000000);
  exception_codes.push_back(0x5555555555555555);

  MINIDUMP_EXCEPTION_STREAM expect_exception = {};

  expect_exception.ThreadId = 123;
  expect_exception.ExceptionRecord.ExceptionCode = 100;
  expect_exception.ExceptionRecord.ExceptionFlags = 1;
  expect_exception.ExceptionRecord.ExceptionAddress = 0xfedcba9876543210;
  expect_exception.ExceptionRecord.NumberParameters =
      static_cast<uint32_t>(exception_codes.size());
  for (size_t index = 0; index < exception_codes.size(); ++index) {
    expect_exception.ExceptionRecord.ExceptionInformation[index] =
        exception_codes[index];
  }
  constexpr uint64_t kThreadID = 0xaaaaaaaaaaaaaaaa;
  constexpr uint32_t kSeed = 65;

  TestExceptionSnapshot exception_snapshot;
  exception_snapshot.SetThreadID(kThreadID);
  exception_snapshot.SetException(
      expect_exception.ExceptionRecord.ExceptionCode);
  exception_snapshot.SetExceptionInfo(
      expect_exception.ExceptionRecord.ExceptionFlags);
  exception_snapshot.SetExceptionAddress(
      expect_exception.ExceptionRecord.ExceptionAddress);
  exception_snapshot.SetCodes(exception_codes);

  InitializeCPUContextX86(exception_snapshot.MutableContext(), kSeed);

  MinidumpThreadIDMap thread_id_map;
  thread_id_map[kThreadID] = expect_exception.ThreadId;

  auto exception_writer = std::make_unique<MinidumpExceptionWriter>();
  exception_writer->InitializeFromSnapshot(&exception_snapshot, thread_id_map);

  MinidumpFileWriter minidump_file_writer;
  ASSERT_TRUE(minidump_file_writer.AddStream(std::move(exception_writer)));

  StringFile string_file;
  ASSERT_TRUE(minidump_file_writer.WriteEverything(&string_file));

  const MINIDUMP_EXCEPTION_STREAM* exception = nullptr;
  ASSERT_NO_FATAL_FAILURE(GetExceptionStream(string_file.string(), &exception));

  const MinidumpContextX86* observed_context = nullptr;
  ASSERT_NO_FATAL_FAILURE(ExpectExceptionStream(&expect_exception,
                                                exception,
                                                string_file.string(),
                                                &observed_context));

  ASSERT_NO_FATAL_FAILURE(
      ExpectMinidumpContextX86(kSeed, observed_context, true));
}

TEST(MinidumpExceptionWriterDeathTest, NoContext) {
  MinidumpFileWriter minidump_file_writer;
  auto exception_writer = std::make_unique<MinidumpExceptionWriter>();

  ASSERT_TRUE(minidump_file_writer.AddStream(std::move(exception_writer)));

  StringFile string_file;
  ASSERT_DEATH_CHECK(minidump_file_writer.WriteEverything(&string_file),
                     "context_");
}

TEST(MinidumpExceptionWriterDeathTest, TooMuchInformation) {
  MinidumpExceptionWriter exception_writer;
  std::vector<uint64_t> exception_information(EXCEPTION_MAXIMUM_PARAMETERS + 1,
                                              0x5a5a5a5a5a5a5a5a);
  ASSERT_DEATH_CHECK(
      exception_writer.SetExceptionInformation(exception_information),
      "kMaxParameters");
}

}  // namespace
}  // namespace test
}  // namespace crashpad
