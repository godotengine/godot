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

#include "minidump/minidump_file_writer.h"

#include <stdint.h>
#include <string.h>

#include <string>
#include <utility>

#include "base/compiler_specific.h"
#include "gtest/gtest.h"
#include "minidump/minidump_stream_writer.h"
#include "minidump/minidump_user_extension_stream_data_source.h"
#include "minidump/test/minidump_file_writer_test_util.h"
#include "minidump/test/minidump_user_extension_stream_util.h"
#include "minidump/test/minidump_writable_test_util.h"
#include "snapshot/test/test_cpu_context.h"
#include "snapshot/test/test_exception_snapshot.h"
#include "snapshot/test/test_memory_snapshot.h"
#include "snapshot/test/test_module_snapshot.h"
#include "snapshot/test/test_process_snapshot.h"
#include "snapshot/test/test_system_snapshot.h"
#include "snapshot/test/test_thread_snapshot.h"
#include "test/gtest_death.h"
#include "util/file/string_file.h"

namespace crashpad {
namespace test {
namespace {

TEST(MinidumpFileWriter, Empty) {
  MinidumpFileWriter minidump_file;
  StringFile string_file;
  ASSERT_TRUE(minidump_file.WriteEverything(&string_file));
  ASSERT_EQ(string_file.string().size(), sizeof(MINIDUMP_HEADER));

  const MINIDUMP_DIRECTORY* directory;
  const MINIDUMP_HEADER* header =
      MinidumpHeaderAtStart(string_file.string(), &directory);
  ASSERT_NO_FATAL_FAILURE(VerifyMinidumpHeader(header, 0, 0));
  EXPECT_FALSE(directory);
}

class TestStream final : public internal::MinidumpStreamWriter {
 public:
  TestStream(MinidumpStreamType stream_type,
             size_t stream_size,
             uint8_t stream_value)
      : stream_data_(stream_size, stream_value), stream_type_(stream_type) {}

  ~TestStream() override {}

  // MinidumpStreamWriter:
  MinidumpStreamType StreamType() const override {
    return stream_type_;
  }

 protected:
  // MinidumpWritable:
  size_t SizeOfObject() override {
    EXPECT_GE(state(), kStateFrozen);
    return stream_data_.size();
  }

  bool WriteObject(FileWriterInterface* file_writer) override {
    EXPECT_EQ(kStateWritable, state());
    return file_writer->Write(&stream_data_[0], stream_data_.size());
  }

 private:
  std::string stream_data_;
  MinidumpStreamType stream_type_;

  DISALLOW_COPY_AND_ASSIGN(TestStream);
};

TEST(MinidumpFileWriter, OneStream) {
  MinidumpFileWriter minidump_file;
  constexpr time_t kTimestamp = 0x155d2fb8;
  minidump_file.SetTimestamp(kTimestamp);

  constexpr size_t kStreamSize = 5;
  constexpr MinidumpStreamType kStreamType =
      static_cast<MinidumpStreamType>(0x4d);
  constexpr uint8_t kStreamValue = 0x5a;
  auto stream =
      std::make_unique<TestStream>(kStreamType, kStreamSize, kStreamValue);
  ASSERT_TRUE(minidump_file.AddStream(std::move(stream)));

  StringFile string_file;
  ASSERT_TRUE(minidump_file.WriteEverything(&string_file));

  constexpr size_t kDirectoryOffset = sizeof(MINIDUMP_HEADER);
  constexpr size_t kStreamOffset =
      kDirectoryOffset + sizeof(MINIDUMP_DIRECTORY);
  constexpr size_t kFileSize = kStreamOffset + kStreamSize;

  ASSERT_EQ(string_file.string().size(), kFileSize);

  const MINIDUMP_DIRECTORY* directory;
  const MINIDUMP_HEADER* header =
      MinidumpHeaderAtStart(string_file.string(), &directory);
  ASSERT_NO_FATAL_FAILURE(VerifyMinidumpHeader(header, 1, kTimestamp));
  ASSERT_TRUE(directory);

  EXPECT_EQ(directory[0].StreamType, kStreamType);
  EXPECT_EQ(directory[0].Location.DataSize, kStreamSize);
  EXPECT_EQ(directory[0].Location.Rva, kStreamOffset);

  const uint8_t* stream_data = MinidumpWritableAtLocationDescriptor<uint8_t>(
      string_file.string(), directory[0].Location);
  ASSERT_TRUE(stream_data);

  std::string expected_stream(kStreamSize, kStreamValue);
  EXPECT_EQ(memcmp(stream_data, expected_stream.c_str(), kStreamSize), 0);
}

TEST(MinidumpFileWriter, AddUserExtensionStream) {
  MinidumpFileWriter minidump_file;
  constexpr time_t kTimestamp = 0x155d2fb8;
  minidump_file.SetTimestamp(kTimestamp);

  static constexpr uint8_t kStreamData[] = "Hello World!";
  constexpr size_t kStreamSize = arraysize(kStreamData);
  constexpr MinidumpStreamType kStreamType =
      static_cast<MinidumpStreamType>(0x4d);

  auto data_source = std::make_unique<test::BufferExtensionStreamDataSource>(
      kStreamType, kStreamData, kStreamSize);
  ASSERT_TRUE(minidump_file.AddUserExtensionStream(std::move(data_source)));

  // Adding the same stream type a second time should fail.
  data_source = std::make_unique<test::BufferExtensionStreamDataSource>(
      kStreamType, kStreamData, kStreamSize);
  ASSERT_FALSE(minidump_file.AddUserExtensionStream(std::move(data_source)));

  StringFile string_file;
  ASSERT_TRUE(minidump_file.WriteEverything(&string_file));

  constexpr size_t kDirectoryOffset = sizeof(MINIDUMP_HEADER);
  constexpr size_t kStreamOffset =
      kDirectoryOffset + sizeof(MINIDUMP_DIRECTORY);
  constexpr size_t kFileSize = kStreamOffset + kStreamSize;

  ASSERT_EQ(string_file.string().size(), kFileSize);

  const MINIDUMP_DIRECTORY* directory;
  const MINIDUMP_HEADER* header =
      MinidumpHeaderAtStart(string_file.string(), &directory);
  ASSERT_NO_FATAL_FAILURE(VerifyMinidumpHeader(header, 1, kTimestamp));
  ASSERT_TRUE(directory);

  EXPECT_EQ(directory[0].StreamType, kStreamType);
  EXPECT_EQ(directory[0].Location.DataSize, kStreamSize);
  EXPECT_EQ(directory[0].Location.Rva, kStreamOffset);

  const uint8_t* stream_data = MinidumpWritableAtLocationDescriptor<uint8_t>(
      string_file.string(), directory[0].Location);
  ASSERT_TRUE(stream_data);

  EXPECT_EQ(memcmp(stream_data, kStreamData, kStreamSize), 0);
}

TEST(MinidumpFileWriter, AddEmptyUserExtensionStream) {
  MinidumpFileWriter minidump_file;
  constexpr time_t kTimestamp = 0x155d2fb8;
  minidump_file.SetTimestamp(kTimestamp);

  constexpr MinidumpStreamType kStreamType =
      static_cast<MinidumpStreamType>(0x4d);

  auto data_source = std::make_unique<test::BufferExtensionStreamDataSource>(
      kStreamType, nullptr, 0);
  ASSERT_TRUE(minidump_file.AddUserExtensionStream(std::move(data_source)));

  StringFile string_file;
  ASSERT_TRUE(minidump_file.WriteEverything(&string_file));

  constexpr size_t kDirectoryOffset = sizeof(MINIDUMP_HEADER);
  constexpr size_t kStreamOffset =
      kDirectoryOffset + sizeof(MINIDUMP_DIRECTORY);
  constexpr size_t kFileSize = kStreamOffset;

  ASSERT_EQ(string_file.string().size(), kFileSize);

  const MINIDUMP_DIRECTORY* directory;
  const MINIDUMP_HEADER* header =
      MinidumpHeaderAtStart(string_file.string(), &directory);
  ASSERT_NO_FATAL_FAILURE(VerifyMinidumpHeader(header, 1, kTimestamp));
  ASSERT_TRUE(directory);

  EXPECT_EQ(directory[0].StreamType, kStreamType);
  EXPECT_EQ(directory[0].Location.DataSize, 0u);
  EXPECT_EQ(directory[0].Location.Rva, kStreamOffset);
}

TEST(MinidumpFileWriter, ThreeStreams) {
  MinidumpFileWriter minidump_file;
  constexpr time_t kTimestamp = 0x155d2fb8;
  minidump_file.SetTimestamp(kTimestamp);

  constexpr size_t kStream0Size = 5;
  constexpr MinidumpStreamType kStream0Type =
      static_cast<MinidumpStreamType>(0x6d);
  constexpr uint8_t kStream0Value = 0x5a;
  auto stream0 =
      std::make_unique<TestStream>(kStream0Type, kStream0Size, kStream0Value);
  ASSERT_TRUE(minidump_file.AddStream(std::move(stream0)));

  // Make the second stream’s type be a smaller quantity than the first stream’s
  // to test that the streams show up in the order that they were added, not in
  // numeric order.
  constexpr size_t kStream1Size = 3;
  constexpr MinidumpStreamType kStream1Type =
      static_cast<MinidumpStreamType>(0x4d);
  constexpr uint8_t kStream1Value = 0xa5;
  auto stream1 =
      std::make_unique<TestStream>(kStream1Type, kStream1Size, kStream1Value);
  ASSERT_TRUE(minidump_file.AddStream(std::move(stream1)));

  constexpr size_t kStream2Size = 1;
  constexpr MinidumpStreamType kStream2Type =
      static_cast<MinidumpStreamType>(0x7e);
  constexpr uint8_t kStream2Value = 0x36;
  auto stream2 =
      std::make_unique<TestStream>(kStream2Type, kStream2Size, kStream2Value);
  ASSERT_TRUE(minidump_file.AddStream(std::move(stream2)));

  StringFile string_file;
  ASSERT_TRUE(minidump_file.WriteEverything(&string_file));

  constexpr size_t kDirectoryOffset = sizeof(MINIDUMP_HEADER);
  constexpr size_t kStream0Offset =
      kDirectoryOffset + 3 * sizeof(MINIDUMP_DIRECTORY);
  constexpr size_t kStream1Padding = 3;
  constexpr size_t kStream1Offset =
      kStream0Offset + kStream0Size + kStream1Padding;
  constexpr size_t kStream2Padding = 1;
  constexpr size_t kStream2Offset =
      kStream1Offset + kStream1Size + kStream2Padding;
  constexpr size_t kFileSize = kStream2Offset + kStream2Size;

  ASSERT_EQ(string_file.string().size(), kFileSize);

  const MINIDUMP_DIRECTORY* directory;
  const MINIDUMP_HEADER* header =
      MinidumpHeaderAtStart(string_file.string(), &directory);
  ASSERT_NO_FATAL_FAILURE(VerifyMinidumpHeader(header, 3, kTimestamp));
  ASSERT_TRUE(directory);

  EXPECT_EQ(directory[0].StreamType, kStream0Type);
  EXPECT_EQ(directory[0].Location.DataSize, kStream0Size);
  EXPECT_EQ(directory[0].Location.Rva, kStream0Offset);
  EXPECT_EQ(directory[1].StreamType, kStream1Type);
  EXPECT_EQ(directory[1].Location.DataSize, kStream1Size);
  EXPECT_EQ(directory[1].Location.Rva, kStream1Offset);
  EXPECT_EQ(directory[2].StreamType, kStream2Type);
  EXPECT_EQ(directory[2].Location.DataSize, kStream2Size);
  EXPECT_EQ(directory[2].Location.Rva, kStream2Offset);

  const uint8_t* stream0_data = MinidumpWritableAtLocationDescriptor<uint8_t>(
      string_file.string(), directory[0].Location);
  ASSERT_TRUE(stream0_data);

  std::string expected_stream0(kStream0Size, kStream0Value);
  EXPECT_EQ(memcmp(stream0_data, expected_stream0.c_str(), kStream0Size), 0);

  static constexpr int kZeroes[16] = {};
  ASSERT_GE(sizeof(kZeroes), kStream1Padding);
  EXPECT_EQ(memcmp(stream0_data + kStream0Size, kZeroes, kStream1Padding), 0);

  const uint8_t* stream1_data = MinidumpWritableAtLocationDescriptor<uint8_t>(
      string_file.string(), directory[1].Location);
  ASSERT_TRUE(stream1_data);

  std::string expected_stream1(kStream1Size, kStream1Value);
  EXPECT_EQ(memcmp(stream1_data, expected_stream1.c_str(), kStream1Size), 0);

  ASSERT_GE(sizeof(kZeroes), kStream2Padding);
  EXPECT_EQ(memcmp(stream1_data + kStream1Size, kZeroes, kStream2Padding), 0);

  const uint8_t* stream2_data = MinidumpWritableAtLocationDescriptor<uint8_t>(
      string_file.string(), directory[2].Location);
  ASSERT_TRUE(stream2_data);

  std::string expected_stream2(kStream2Size, kStream2Value);
  EXPECT_EQ(memcmp(stream2_data, expected_stream2.c_str(), kStream2Size), 0);
}

TEST(MinidumpFileWriter, ZeroLengthStream) {
  MinidumpFileWriter minidump_file;

  constexpr size_t kStreamSize = 0;
  constexpr MinidumpStreamType kStreamType =
      static_cast<MinidumpStreamType>(0x4d);
  auto stream = std::make_unique<TestStream>(
      kStreamType, kStreamSize, static_cast<uint8_t>(0));
  ASSERT_TRUE(minidump_file.AddStream(std::move(stream)));

  StringFile string_file;
  ASSERT_TRUE(minidump_file.WriteEverything(&string_file));

  constexpr size_t kDirectoryOffset = sizeof(MINIDUMP_HEADER);
  constexpr size_t kStreamOffset =
      kDirectoryOffset + sizeof(MINIDUMP_DIRECTORY);
  constexpr size_t kFileSize = kStreamOffset + kStreamSize;

  ASSERT_EQ(string_file.string().size(), kFileSize);

  const MINIDUMP_DIRECTORY* directory;
  const MINIDUMP_HEADER* header =
      MinidumpHeaderAtStart(string_file.string(), &directory);
  ASSERT_NO_FATAL_FAILURE(VerifyMinidumpHeader(header, 1, 0));
  ASSERT_TRUE(directory);

  EXPECT_EQ(directory[0].StreamType, kStreamType);
  EXPECT_EQ(directory[0].Location.DataSize, kStreamSize);
  EXPECT_EQ(directory[0].Location.Rva, kStreamOffset);
}

TEST(MinidumpFileWriter, InitializeFromSnapshot_Basic) {
  constexpr uint32_t kSnapshotTime = 0x4976043c;
  constexpr timeval kSnapshotTimeval = {static_cast<time_t>(kSnapshotTime), 0};

  TestProcessSnapshot process_snapshot;
  process_snapshot.SetSnapshotTime(kSnapshotTimeval);

  auto system_snapshot = std::make_unique<TestSystemSnapshot>();
  system_snapshot->SetCPUArchitecture(kCPUArchitectureX86_64);
  system_snapshot->SetOperatingSystem(SystemSnapshot::kOperatingSystemMacOSX);
  process_snapshot.SetSystem(std::move(system_snapshot));

  auto peb_snapshot = std::make_unique<TestMemorySnapshot>();
  constexpr uint64_t kPebAddress = 0x07f90000;
  peb_snapshot->SetAddress(kPebAddress);
  constexpr size_t kPebSize = 0x280;
  peb_snapshot->SetSize(kPebSize);
  peb_snapshot->SetValue('p');
  process_snapshot.AddExtraMemory(std::move(peb_snapshot));

  MinidumpFileWriter minidump_file_writer;
  minidump_file_writer.InitializeFromSnapshot(&process_snapshot);

  StringFile string_file;
  ASSERT_TRUE(minidump_file_writer.WriteEverything(&string_file));

  const MINIDUMP_DIRECTORY* directory;
  const MINIDUMP_HEADER* header =
      MinidumpHeaderAtStart(string_file.string(), &directory);
  ASSERT_NO_FATAL_FAILURE(VerifyMinidumpHeader(header, 5, kSnapshotTime));
  ASSERT_TRUE(directory);

  EXPECT_EQ(directory[0].StreamType, kMinidumpStreamTypeSystemInfo);
  EXPECT_TRUE(MinidumpWritableAtLocationDescriptor<MINIDUMP_SYSTEM_INFO>(
                  string_file.string(), directory[0].Location));

  EXPECT_EQ(directory[1].StreamType, kMinidumpStreamTypeMiscInfo);
  EXPECT_TRUE(MinidumpWritableAtLocationDescriptor<MINIDUMP_MISC_INFO_4>(
                  string_file.string(), directory[1].Location));

  EXPECT_EQ(directory[2].StreamType, kMinidumpStreamTypeThreadList);
  EXPECT_TRUE(MinidumpWritableAtLocationDescriptor<MINIDUMP_THREAD_LIST>(
                  string_file.string(), directory[2].Location));

  EXPECT_EQ(directory[3].StreamType, kMinidumpStreamTypeModuleList);
  EXPECT_TRUE(MinidumpWritableAtLocationDescriptor<MINIDUMP_MODULE_LIST>(
                  string_file.string(), directory[3].Location));

  EXPECT_EQ(directory[4].StreamType, kMinidumpStreamTypeMemoryList);
  EXPECT_TRUE(MinidumpWritableAtLocationDescriptor<MINIDUMP_MEMORY_LIST>(
                  string_file.string(), directory[4].Location));

  const MINIDUMP_MEMORY_LIST* memory_list =
      MinidumpWritableAtLocationDescriptor<MINIDUMP_MEMORY_LIST>(
          string_file.string(), directory[4].Location);
  EXPECT_EQ(memory_list->NumberOfMemoryRanges, 1u);
  EXPECT_EQ(memory_list->MemoryRanges[0].StartOfMemoryRange, kPebAddress);
  EXPECT_EQ(memory_list->MemoryRanges[0].Memory.DataSize, kPebSize);
}

TEST(MinidumpFileWriter, InitializeFromSnapshot_Exception) {
  // In a 32-bit environment, this will give a “timestamp out of range” warning,
  // but the test should complete without failure.
  constexpr uint32_t kSnapshotTime = 0xfd469ab8;
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconstant-conversion"
#define DISABLED_WCONSTANT_CONVERSION
#endif  // __clang__
  MSVC_SUPPRESS_WARNING(4309);  // Truncation of constant value.
  MSVC_SUPPRESS_WARNING(4838);  // Narrowing conversion.
  constexpr timeval kSnapshotTimeval = {static_cast<time_t>(kSnapshotTime), 0};
#if defined(DISABLED_WCONSTANT_CONVERSION)
#pragma clang diagnostic pop
#undef DISABLED_WCONSTANT_CONVERSION
#endif  // DISABLED_WCONSTANT_CONVERSION

  TestProcessSnapshot process_snapshot;
  process_snapshot.SetSnapshotTime(kSnapshotTimeval);

  auto system_snapshot = std::make_unique<TestSystemSnapshot>();
  system_snapshot->SetCPUArchitecture(kCPUArchitectureX86_64);
  system_snapshot->SetOperatingSystem(SystemSnapshot::kOperatingSystemMacOSX);
  process_snapshot.SetSystem(std::move(system_snapshot));

  auto thread_snapshot = std::make_unique<TestThreadSnapshot>();
  InitializeCPUContextX86_64(thread_snapshot->MutableContext(), 5);
  process_snapshot.AddThread(std::move(thread_snapshot));

  auto exception_snapshot = std::make_unique<TestExceptionSnapshot>();
  InitializeCPUContextX86_64(exception_snapshot->MutableContext(), 11);
  process_snapshot.SetException(std::move(exception_snapshot));

  // The module does not have anything that needs to be represented in a
  // MinidumpModuleCrashpadInfo structure, so no such structure is expected to
  // be present, which will in turn suppress the addition of a
  // MinidumpCrashpadInfo stream.
  auto module_snapshot = std::make_unique<TestModuleSnapshot>();
  process_snapshot.AddModule(std::move(module_snapshot));

  MinidumpFileWriter minidump_file_writer;
  minidump_file_writer.InitializeFromSnapshot(&process_snapshot);

  StringFile string_file;
  ASSERT_TRUE(minidump_file_writer.WriteEverything(&string_file));

  const MINIDUMP_DIRECTORY* directory;
  const MINIDUMP_HEADER* header =
      MinidumpHeaderAtStart(string_file.string(), &directory);
  ASSERT_NO_FATAL_FAILURE(VerifyMinidumpHeader(header, 6, kSnapshotTime));
  ASSERT_TRUE(directory);

  EXPECT_EQ(directory[0].StreamType, kMinidumpStreamTypeSystemInfo);
  EXPECT_TRUE(MinidumpWritableAtLocationDescriptor<MINIDUMP_SYSTEM_INFO>(
                  string_file.string(), directory[0].Location));

  EXPECT_EQ(directory[1].StreamType, kMinidumpStreamTypeMiscInfo);
  EXPECT_TRUE(MinidumpWritableAtLocationDescriptor<MINIDUMP_MISC_INFO_4>(
                  string_file.string(), directory[1].Location));

  EXPECT_EQ(directory[2].StreamType, kMinidumpStreamTypeThreadList);
  EXPECT_TRUE(MinidumpWritableAtLocationDescriptor<MINIDUMP_THREAD_LIST>(
                  string_file.string(), directory[2].Location));

  EXPECT_EQ(directory[3].StreamType, kMinidumpStreamTypeException);
  EXPECT_TRUE(MinidumpWritableAtLocationDescriptor<MINIDUMP_EXCEPTION_STREAM>(
                  string_file.string(), directory[3].Location));

  EXPECT_EQ(directory[4].StreamType, kMinidumpStreamTypeModuleList);
  EXPECT_TRUE(MinidumpWritableAtLocationDescriptor<MINIDUMP_MODULE_LIST>(
                  string_file.string(), directory[4].Location));

  EXPECT_EQ(directory[5].StreamType, kMinidumpStreamTypeMemoryList);
  EXPECT_TRUE(MinidumpWritableAtLocationDescriptor<MINIDUMP_MEMORY_LIST>(
                  string_file.string(), directory[5].Location));
}

TEST(MinidumpFileWriter, InitializeFromSnapshot_CrashpadInfo) {
  constexpr uint32_t kSnapshotTime = 0x15393bd3;
  constexpr timeval kSnapshotTimeval = {static_cast<time_t>(kSnapshotTime), 0};

  TestProcessSnapshot process_snapshot;
  process_snapshot.SetSnapshotTime(kSnapshotTimeval);

  auto system_snapshot = std::make_unique<TestSystemSnapshot>();
  system_snapshot->SetCPUArchitecture(kCPUArchitectureX86_64);
  system_snapshot->SetOperatingSystem(SystemSnapshot::kOperatingSystemMacOSX);
  process_snapshot.SetSystem(std::move(system_snapshot));

  auto thread_snapshot = std::make_unique<TestThreadSnapshot>();
  InitializeCPUContextX86_64(thread_snapshot->MutableContext(), 5);
  process_snapshot.AddThread(std::move(thread_snapshot));

  auto exception_snapshot = std::make_unique<TestExceptionSnapshot>();
  InitializeCPUContextX86_64(exception_snapshot->MutableContext(), 11);
  process_snapshot.SetException(std::move(exception_snapshot));

  // The module needs an annotation for the MinidumpCrashpadInfo stream to be
  // considered useful and be included.
  auto module_snapshot = std::make_unique<TestModuleSnapshot>();
  std::vector<std::string> annotations_list(1, std::string("annotation"));
  module_snapshot->SetAnnotationsVector(annotations_list);
  process_snapshot.AddModule(std::move(module_snapshot));

  MinidumpFileWriter minidump_file_writer;
  minidump_file_writer.InitializeFromSnapshot(&process_snapshot);

  StringFile string_file;
  ASSERT_TRUE(minidump_file_writer.WriteEverything(&string_file));

  const MINIDUMP_DIRECTORY* directory;
  const MINIDUMP_HEADER* header =
      MinidumpHeaderAtStart(string_file.string(), &directory);
  ASSERT_NO_FATAL_FAILURE(VerifyMinidumpHeader(header, 7, kSnapshotTime));
  ASSERT_TRUE(directory);

  EXPECT_EQ(directory[0].StreamType, kMinidumpStreamTypeSystemInfo);
  EXPECT_TRUE(MinidumpWritableAtLocationDescriptor<MINIDUMP_SYSTEM_INFO>(
                  string_file.string(), directory[0].Location));

  EXPECT_EQ(directory[1].StreamType, kMinidumpStreamTypeMiscInfo);
  EXPECT_TRUE(MinidumpWritableAtLocationDescriptor<MINIDUMP_MISC_INFO_4>(
                  string_file.string(), directory[1].Location));

  EXPECT_EQ(directory[2].StreamType, kMinidumpStreamTypeThreadList);
  EXPECT_TRUE(MinidumpWritableAtLocationDescriptor<MINIDUMP_THREAD_LIST>(
                  string_file.string(), directory[2].Location));

  EXPECT_EQ(directory[3].StreamType, kMinidumpStreamTypeException);
  EXPECT_TRUE(MinidumpWritableAtLocationDescriptor<MINIDUMP_EXCEPTION_STREAM>(
                  string_file.string(), directory[3].Location));

  EXPECT_EQ(directory[4].StreamType, kMinidumpStreamTypeModuleList);
  EXPECT_TRUE(MinidumpWritableAtLocationDescriptor<MINIDUMP_MODULE_LIST>(
                  string_file.string(), directory[4].Location));

  EXPECT_EQ(directory[5].StreamType, kMinidumpStreamTypeCrashpadInfo);
  EXPECT_TRUE(MinidumpWritableAtLocationDescriptor<MinidumpCrashpadInfo>(
                  string_file.string(), directory[5].Location));

  EXPECT_EQ(directory[6].StreamType, kMinidumpStreamTypeMemoryList);
  EXPECT_TRUE(MinidumpWritableAtLocationDescriptor<MINIDUMP_MEMORY_LIST>(
                  string_file.string(), directory[6].Location));
}

TEST(MinidumpFileWriter, SameStreamType) {
  MinidumpFileWriter minidump_file;

  constexpr size_t kStream0Size = 3;
  constexpr MinidumpStreamType kStreamType =
      static_cast<MinidumpStreamType>(0x4d);
  constexpr uint8_t kStream0Value = 0x5a;
  auto stream0 =
      std::make_unique<TestStream>(kStreamType, kStream0Size, kStream0Value);
  ASSERT_TRUE(minidump_file.AddStream(std::move(stream0)));

  // An attempt to add a second stream of the same type should fail.
  constexpr size_t kStream1Size = 5;
  constexpr uint8_t kStream1Value = 0xa5;
  auto stream1 =
      std::make_unique<TestStream>(kStreamType, kStream1Size, kStream1Value);
  ASSERT_FALSE(minidump_file.AddStream(std::move(stream1)));

  StringFile string_file;
  ASSERT_TRUE(minidump_file.WriteEverything(&string_file));

  constexpr size_t kDirectoryOffset = sizeof(MINIDUMP_HEADER);
  constexpr size_t kStream0Offset =
      kDirectoryOffset + sizeof(MINIDUMP_DIRECTORY);
  constexpr size_t kFileSize = kStream0Offset + kStream0Size;

  ASSERT_EQ(string_file.string().size(), kFileSize);

  const MINIDUMP_DIRECTORY* directory;
  const MINIDUMP_HEADER* header =
      MinidumpHeaderAtStart(string_file.string(), &directory);
  ASSERT_NO_FATAL_FAILURE(VerifyMinidumpHeader(header, 1, 0));
  ASSERT_TRUE(directory);

  EXPECT_EQ(directory[0].StreamType, kStreamType);
  EXPECT_EQ(directory[0].Location.DataSize, kStream0Size);
  EXPECT_EQ(directory[0].Location.Rva, kStream0Offset);

  const uint8_t* stream_data = MinidumpWritableAtLocationDescriptor<uint8_t>(
      string_file.string(), directory[0].Location);
  ASSERT_TRUE(stream_data);

  std::string expected_stream(kStream0Size, kStream0Value);
  EXPECT_EQ(memcmp(stream_data, expected_stream.c_str(), kStream0Size), 0);
}

}  // namespace
}  // namespace test
}  // namespace crashpad
