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

#include "util/file/file_reader.h"

#include <stdint.h>

#include <limits>
#include <type_traits>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace crashpad {
namespace test {
namespace {

using testing::_;
using testing::InSequence;
using testing::Return;

class MockFileReader : public FileReaderInterface {
 public:
  MockFileReader() : FileReaderInterface() {}
  ~MockFileReader() override {}

  // Since it’s more convenient for the test to use uintptr_t than void*,
  // ReadExactlyInt() and ReadInt() adapt the types.

  bool ReadExactlyInt(uintptr_t data, size_t size) {
    return ReadExactly(reinterpret_cast<void*>(data), size);
  }

  MOCK_METHOD2(ReadInt, FileOperationResult(uintptr_t, size_t));

  // FileReaderInterface:
  FileOperationResult Read(void* data, size_t size) override {
    return ReadInt(reinterpret_cast<uintptr_t>(data), size);
  }

  // FileSeekerInterface:
  MOCK_METHOD2(Seek, FileOffset(FileOffset, int));

 private:
  DISALLOW_COPY_AND_ASSIGN(MockFileReader);
};

TEST(FileReader, ReadExactly_Zero) {
  MockFileReader file_reader;
  InSequence in_sequence;
  EXPECT_CALL(file_reader, ReadInt(_, _)).Times(0);
  EXPECT_CALL(file_reader, Seek(_, _)).Times(0);
  EXPECT_TRUE(file_reader.ReadExactlyInt(100, 0));
}

TEST(FileReader, ReadExactly_SingleSmallSuccess) {
  MockFileReader file_reader;
  InSequence in_sequence;
  EXPECT_CALL(file_reader, ReadInt(1000, 1)).WillOnce(Return(1));
  EXPECT_CALL(file_reader, Seek(_, _)).Times(0);
  EXPECT_TRUE(file_reader.ReadExactlyInt(1000, 1));
}

TEST(FileReader, ReadExactly_SingleSmallFailure) {
  MockFileReader file_reader;
  InSequence in_sequence;
  EXPECT_CALL(file_reader, ReadInt(1000, 1)).WillOnce(Return(-1));
  EXPECT_CALL(file_reader, Seek(_, _)).Times(0);
  EXPECT_FALSE(file_reader.ReadExactlyInt(1000, 1));
}

TEST(FileReader, ReadExactly_DoubleSmallSuccess) {
  MockFileReader file_reader;
  InSequence in_sequence;
  EXPECT_CALL(file_reader, ReadInt(0x1000, 2)).WillOnce(Return(1));
  EXPECT_CALL(file_reader, ReadInt(0x1001, 1)).WillOnce(Return(1));
  EXPECT_CALL(file_reader, Seek(_, _)).Times(0);
  EXPECT_TRUE(file_reader.ReadExactlyInt(0x1000, 2));
}

TEST(FileReader, ReadExactly_DoubleSmallShort) {
  MockFileReader file_reader;
  InSequence in_sequence;
  EXPECT_CALL(file_reader, ReadInt(0x20000, 2)).WillOnce(Return(1));
  EXPECT_CALL(file_reader, ReadInt(0x20001, 1)).WillOnce(Return(0));
  EXPECT_CALL(file_reader, Seek(_, _)).Times(0);
  EXPECT_FALSE(file_reader.ReadExactlyInt(0x20000, 2));
}

TEST(FileReader, ReadExactly_Medium) {
  MockFileReader file_reader;
  InSequence in_sequence;
  EXPECT_CALL(file_reader, ReadInt(0x80000000, 0x20000000))
      .WillOnce(Return(0x10000000));
  EXPECT_CALL(file_reader, ReadInt(0x90000000, 0x10000000))
      .WillOnce(Return(0x8000000));
  EXPECT_CALL(file_reader, ReadInt(0x98000000, 0x8000000))
      .WillOnce(Return(0x4000000));
  EXPECT_CALL(file_reader, ReadInt(0x9c000000, 0x4000000))
      .WillOnce(Return(0x2000000));
  EXPECT_CALL(file_reader, ReadInt(0x9e000000, 0x2000000))
      .WillOnce(Return(0x1000000));
  EXPECT_CALL(file_reader, ReadInt(0x9f000000, 0x1000000))
      .WillOnce(Return(0x800000));
  EXPECT_CALL(file_reader, ReadInt(0x9f800000, 0x800000))
      .WillOnce(Return(0x400000));
  EXPECT_CALL(file_reader, ReadInt(0x9fc00000, 0x400000))
      .WillOnce(Return(0x200000));
  EXPECT_CALL(file_reader, ReadInt(0x9fe00000, 0x200000))
      .WillOnce(Return(0x100000));
  EXPECT_CALL(file_reader, ReadInt(0x9ff00000, 0x100000))
      .WillOnce(Return(0x80000));
  EXPECT_CALL(file_reader, ReadInt(0x9ff80000, 0x80000))
      .WillOnce(Return(0x40000));
  EXPECT_CALL(file_reader, ReadInt(0x9ffc0000, 0x40000))
      .WillOnce(Return(0x20000));
  EXPECT_CALL(file_reader, ReadInt(0x9ffe0000, 0x20000))
      .WillOnce(Return(0x10000));
  EXPECT_CALL(file_reader, ReadInt(0x9fff0000, 0x10000))
      .WillOnce(Return(0x8000));
  EXPECT_CALL(file_reader, ReadInt(0x9fff8000, 0x8000))
      .WillOnce(Return(0x4000));
  EXPECT_CALL(file_reader, ReadInt(0x9fffc000, 0x4000))
      .WillOnce(Return(0x2000));
  EXPECT_CALL(file_reader, ReadInt(0x9fffe000, 0x2000))
      .WillOnce(Return(0x1000));
  EXPECT_CALL(file_reader, ReadInt(0x9ffff000, 0x1000)).WillOnce(Return(0x800));
  EXPECT_CALL(file_reader, ReadInt(0x9ffff800, 0x800)).WillOnce(Return(0x400));
  EXPECT_CALL(file_reader, ReadInt(0x9ffffc00, 0x400)).WillOnce(Return(0x200));
  EXPECT_CALL(file_reader, ReadInt(0x9ffffe00, 0x200)).WillOnce(Return(0x100));
  EXPECT_CALL(file_reader, ReadInt(0x9fffff00, 0x100)).WillOnce(Return(0x80));
  EXPECT_CALL(file_reader, ReadInt(0x9fffff80, 0x80)).WillOnce(Return(0x40));
  EXPECT_CALL(file_reader, ReadInt(0x9fffffc0, 0x40)).WillOnce(Return(0x20));
  EXPECT_CALL(file_reader, ReadInt(0x9fffffe0, 0x20)).WillOnce(Return(0x10));
  EXPECT_CALL(file_reader, ReadInt(0x9ffffff0, 0x10)).WillOnce(Return(0x8));
  EXPECT_CALL(file_reader, ReadInt(0x9ffffff8, 0x8)).WillOnce(Return(0x4));
  EXPECT_CALL(file_reader, ReadInt(0x9ffffffc, 0x4)).WillOnce(Return(0x2));
  EXPECT_CALL(file_reader, ReadInt(0x9ffffffe, 0x2)).WillOnce(Return(0x1));
  EXPECT_CALL(file_reader, ReadInt(0x9fffffff, 0x1)).WillOnce(Return(0x1));
  EXPECT_CALL(file_reader, Seek(_, _)).Times(0);
  EXPECT_TRUE(file_reader.ReadExactlyInt(0x80000000, 0x20000000));
}

TEST(FileReader, ReadExactly_LargeSuccess) {
  MockFileReader file_reader;
  InSequence in_sequence;
  constexpr size_t max = std::numeric_limits<uint32_t>::max();
  constexpr size_t increment = std::numeric_limits<int32_t>::max();
  EXPECT_CALL(file_reader, ReadInt(0, max)).WillOnce(Return(increment));
  EXPECT_CALL(file_reader, ReadInt(increment, max - increment))
      .WillOnce(Return(increment));
  EXPECT_CALL(file_reader, ReadInt(2 * increment, 1)).WillOnce(Return(1));
  EXPECT_CALL(file_reader, Seek(_, _)).Times(0);
  EXPECT_TRUE(file_reader.ReadExactlyInt(0, max));
}

TEST(FileReader, ReadExactly_LargeShort) {
  MockFileReader file_reader;
  InSequence in_sequence;
  EXPECT_CALL(file_reader, ReadInt(0, 0xffffffff)).WillOnce(Return(0x7fffffff));
  EXPECT_CALL(file_reader, ReadInt(0x7fffffff, 0x80000000))
      .WillOnce(Return(0x10000000));
  EXPECT_CALL(file_reader, ReadInt(0x8fffffff, 0x70000000)).WillOnce(Return(0));
  EXPECT_CALL(file_reader, Seek(_, _)).Times(0);
  EXPECT_FALSE(file_reader.ReadExactlyInt(0, 0xffffffff));
}

TEST(FileReader, ReadExactly_LargeFailure) {
  MockFileReader file_reader;
  InSequence in_sequence;
  EXPECT_CALL(file_reader, ReadInt(0, 0xffffffff)).WillOnce(Return(0x7fffffff));
  EXPECT_CALL(file_reader, ReadInt(0x7fffffff, 0x80000000))
      .WillOnce(Return(-1));
  EXPECT_CALL(file_reader, Seek(_, _)).Times(0);
  EXPECT_FALSE(file_reader.ReadExactlyInt(0, 0xffffffff));
}

TEST(FileReader, ReadExactly_TripleMax) {
  MockFileReader file_reader;
  InSequence in_sequence;
  constexpr size_t max = std::numeric_limits<size_t>::max();
  constexpr size_t increment =
      std::numeric_limits<std::make_signed<size_t>::type>::max();
  EXPECT_CALL(file_reader, ReadInt(0, max)).WillOnce(Return(increment));
  EXPECT_CALL(file_reader, ReadInt(increment, max - increment))
      .WillOnce(Return(increment));
  EXPECT_CALL(file_reader, ReadInt(2 * increment, 1)).WillOnce(Return(1));
  EXPECT_CALL(file_reader, Seek(_, _)).Times(0);
  EXPECT_TRUE(file_reader.ReadExactlyInt(0, max));
}

}  // namespace
}  // namespace test
}  // namespace crashpad
