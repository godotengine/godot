// Copyright 2016 The Crashpad Authors. All rights reserved.
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

#include "minidump/minidump_user_stream_writer.h"

#include <string>
#include <utility>

#include "gtest/gtest.h"
#include "minidump/minidump_file_writer.h"
#include "minidump/test/minidump_file_writer_test_util.h"
#include "minidump/test/minidump_user_extension_stream_util.h"
#include "minidump/test/minidump_writable_test_util.h"
#include "snapshot/test/test_memory_snapshot.h"
#include "util/file/string_file.h"

namespace crashpad {
namespace test {
namespace {

// The user stream is expected to be the only stream.
void GetUserStream(const std::string& file_contents,
                   MINIDUMP_LOCATION_DESCRIPTOR* user_stream_location,
                   uint32_t stream_type,
                   size_t stream_size) {
  constexpr size_t kDirectoryOffset = sizeof(MINIDUMP_HEADER);
  constexpr size_t kUserStreamOffset =
      kDirectoryOffset + sizeof(MINIDUMP_DIRECTORY);

  const MINIDUMP_DIRECTORY* directory;
  const MINIDUMP_HEADER* header =
      MinidumpHeaderAtStart(file_contents, &directory);
  ASSERT_NO_FATAL_FAILURE(VerifyMinidumpHeader(header, 1, 0));
  ASSERT_TRUE(directory);

  constexpr size_t kDirectoryIndex = 0;

  ASSERT_EQ(directory[kDirectoryIndex].StreamType, stream_type);
  EXPECT_EQ(directory[kDirectoryIndex].Location.Rva, kUserStreamOffset);
  EXPECT_EQ(directory[kDirectoryIndex].Location.DataSize, stream_size);
  *user_stream_location = directory[kDirectoryIndex].Location;
}

constexpr MinidumpStreamType kTestStreamId =
    static_cast<MinidumpStreamType>(0x123456);

TEST(MinidumpUserStreamWriter, InitializeFromSnapshotNoData) {
  MinidumpFileWriter minidump_file_writer;
  auto user_stream_writer = std::make_unique<MinidumpUserStreamWriter>();
  auto stream = std::make_unique<UserMinidumpStream>(kTestStreamId, nullptr);
  user_stream_writer->InitializeFromSnapshot(stream.get());
  ASSERT_TRUE(minidump_file_writer.AddStream(std::move(user_stream_writer)));

  StringFile string_file;
  ASSERT_TRUE(minidump_file_writer.WriteEverything(&string_file));

  ASSERT_EQ(string_file.string().size(),
            sizeof(MINIDUMP_HEADER) + sizeof(MINIDUMP_DIRECTORY));

  MINIDUMP_LOCATION_DESCRIPTOR user_stream_location;
  ASSERT_NO_FATAL_FAILURE(GetUserStream(
      string_file.string(), &user_stream_location, kTestStreamId, 0u));
}

TEST(MinidumpUserStreamWriter, InitializeFromUserExtensionStreamNoData) {
  MinidumpFileWriter minidump_file_writer;
  auto data_source = std::make_unique<test::BufferExtensionStreamDataSource>(
      kTestStreamId, nullptr, 0);
  auto user_stream_writer = std::make_unique<MinidumpUserStreamWriter>();
  user_stream_writer->InitializeFromUserExtensionStream(std::move(data_source));
  minidump_file_writer.AddStream(std::move(user_stream_writer));

  StringFile string_file;
  ASSERT_TRUE(minidump_file_writer.WriteEverything(&string_file));

  ASSERT_EQ(string_file.string().size(),
            sizeof(MINIDUMP_HEADER) + sizeof(MINIDUMP_DIRECTORY));

  MINIDUMP_LOCATION_DESCRIPTOR user_stream_location;
  ASSERT_NO_FATAL_FAILURE(GetUserStream(
      string_file.string(), &user_stream_location, kTestStreamId, 0u));
}

TEST(MinidumpUserStreamWriter, InitializeFromSnapshotOneStream) {
  MinidumpFileWriter minidump_file_writer;
  auto user_stream_writer = std::make_unique<MinidumpUserStreamWriter>();

  TestMemorySnapshot* test_data = new TestMemorySnapshot();
  test_data->SetAddress(97865);
  constexpr size_t kStreamSize = 128;
  test_data->SetSize(kStreamSize);
  test_data->SetValue('c');
  auto stream = std::make_unique<UserMinidumpStream>(kTestStreamId, test_data);
  user_stream_writer->InitializeFromSnapshot(stream.get());
  ASSERT_TRUE(minidump_file_writer.AddStream(std::move(user_stream_writer)));

  StringFile string_file;
  ASSERT_TRUE(minidump_file_writer.WriteEverything(&string_file));

  ASSERT_EQ(string_file.string().size(),
            sizeof(MINIDUMP_HEADER) + sizeof(MINIDUMP_DIRECTORY) + kStreamSize);

  MINIDUMP_LOCATION_DESCRIPTOR user_stream_location = {};
  ASSERT_NO_FATAL_FAILURE(GetUserStream(
      string_file.string(), &user_stream_location, kTestStreamId, kStreamSize));
  const std::string stream_data = string_file.string().substr(
      user_stream_location.Rva, user_stream_location.DataSize);
  EXPECT_EQ(stream_data, std::string(kStreamSize, 'c'));
}

TEST(MinidumpUserStreamWriter, InitializeFromBufferOneStream) {
  MinidumpFileWriter minidump_file_writer;

  constexpr size_t kStreamSize = 128;
  std::vector<uint8_t> data(kStreamSize, 'c');
  auto data_source = std::make_unique<test::BufferExtensionStreamDataSource>(
      kTestStreamId, &data[0], data.size());
  auto user_stream_writer = std::make_unique<MinidumpUserStreamWriter>();
  user_stream_writer->InitializeFromUserExtensionStream(std::move(data_source));
  minidump_file_writer.AddStream(std::move(user_stream_writer));

  StringFile string_file;
  ASSERT_TRUE(minidump_file_writer.WriteEverything(&string_file));

  ASSERT_EQ(string_file.string().size(),
            sizeof(MINIDUMP_HEADER) + sizeof(MINIDUMP_DIRECTORY) + kStreamSize);

  MINIDUMP_LOCATION_DESCRIPTOR user_stream_location = {};
  ASSERT_NO_FATAL_FAILURE(GetUserStream(
      string_file.string(), &user_stream_location, kTestStreamId, kStreamSize));
  const std::string stream_data = string_file.string().substr(
      user_stream_location.Rva, user_stream_location.DataSize);
  EXPECT_EQ(stream_data, std::string(kStreamSize, 'c'));
}

}  // namespace
}  // namespace test
}  // namespace crashpad
