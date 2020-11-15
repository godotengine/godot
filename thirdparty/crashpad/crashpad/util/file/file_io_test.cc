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

#include "util/file/file_io.h"

#include <stdio.h>

#include <limits>
#include <type_traits>

#include "base/atomicops.h"
#include "base/files/file_path.h"
#include "base/macros.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "test/errors.h"
#include "test/file.h"
#include "test/scoped_temp_dir.h"
#include "util/misc/implicit_cast.h"
#include "util/thread/thread.h"

namespace crashpad {
namespace test {
namespace {

using testing::_;
using testing::InSequence;
using testing::Return;

class MockReadExactly : public internal::ReadExactlyInternal {
 public:
  MockReadExactly() : ReadExactlyInternal() {}
  ~MockReadExactly() {}

  // Since it’s more convenient for the test to use uintptr_t than void*,
  // ReadExactlyInt() and ReadInt() adapt the types.

  bool ReadExactlyInt(uintptr_t data, size_t size, bool can_log) {
    return ReadExactly(reinterpret_cast<void*>(data), size, can_log);
  }

  MOCK_METHOD3(ReadInt, FileOperationResult(uintptr_t, size_t, bool));

  // ReadExactlyInternal:
  FileOperationResult Read(void* data, size_t size, bool can_log) {
    return ReadInt(reinterpret_cast<uintptr_t>(data), size, can_log);
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(MockReadExactly);
};

TEST(FileIO, ReadExactly_Zero) {
  MockReadExactly read_exactly;
  InSequence in_sequence;
  EXPECT_CALL(read_exactly, ReadInt(_, _, false)).Times(0);
  EXPECT_TRUE(read_exactly.ReadExactlyInt(100, 0, false));
}

TEST(FileIO, ReadExactly_SingleSmallSuccess) {
  MockReadExactly read_exactly;
  InSequence in_sequence;
  EXPECT_CALL(read_exactly, ReadInt(1000, 1, false)).WillOnce(Return(1));
  EXPECT_TRUE(read_exactly.ReadExactlyInt(1000, 1, false));
}

TEST(FileIO, ReadExactly_SingleSmallSuccessCanLog) {
  MockReadExactly read_exactly;
  InSequence in_sequence;
  EXPECT_CALL(read_exactly, ReadInt(1000, 1, true)).WillOnce(Return(1));
  EXPECT_TRUE(read_exactly.ReadExactlyInt(1000, 1, true));
}

TEST(FileIO, ReadExactly_SingleSmallFailure) {
  MockReadExactly read_exactly;
  InSequence in_sequence;
  EXPECT_CALL(read_exactly, ReadInt(1000, 1, false)).WillOnce(Return(-1));
  EXPECT_FALSE(read_exactly.ReadExactlyInt(1000, 1, false));
}

TEST(FileIO, ReadExactly_SingleSmallFailureCanLog) {
  MockReadExactly read_exactly;
  InSequence in_sequence;
  EXPECT_CALL(read_exactly, ReadInt(1000, 1, true)).WillOnce(Return(-1));
  EXPECT_FALSE(read_exactly.ReadExactlyInt(1000, 1, true));
}

TEST(FileIO, ReadExactly_DoubleSmallSuccess) {
  MockReadExactly read_exactly;
  InSequence in_sequence;
  EXPECT_CALL(read_exactly, ReadInt(0x1000, 2, false)).WillOnce(Return(1));
  EXPECT_CALL(read_exactly, ReadInt(0x1001, 1, false)).WillOnce(Return(1));
  EXPECT_TRUE(read_exactly.ReadExactlyInt(0x1000, 2, false));
}

TEST(FileIO, ReadExactly_DoubleSmallShort) {
  MockReadExactly read_exactly;
  InSequence in_sequence;
  EXPECT_CALL(read_exactly, ReadInt(0x20000, 2, false)).WillOnce(Return(1));
  EXPECT_CALL(read_exactly, ReadInt(0x20001, 1, false)).WillOnce(Return(0));
  EXPECT_FALSE(read_exactly.ReadExactlyInt(0x20000, 2, false));
}

TEST(FileIO, ReadExactly_DoubleSmallShortCanLog) {
  MockReadExactly read_exactly;
  InSequence in_sequence;
  EXPECT_CALL(read_exactly, ReadInt(0x20000, 2, true)).WillOnce(Return(1));
  EXPECT_CALL(read_exactly, ReadInt(0x20001, 1, true)).WillOnce(Return(0));
  EXPECT_FALSE(read_exactly.ReadExactlyInt(0x20000, 2, true));
}

TEST(FileIO, ReadExactly_Medium) {
  MockReadExactly read_exactly;
  InSequence in_sequence;
  EXPECT_CALL(read_exactly, ReadInt(0x80000000, 0x20000000, false))
      .WillOnce(Return(0x10000000));
  EXPECT_CALL(read_exactly, ReadInt(0x90000000, 0x10000000, false))
      .WillOnce(Return(0x8000000));
  EXPECT_CALL(read_exactly, ReadInt(0x98000000, 0x8000000, false))
      .WillOnce(Return(0x4000000));
  EXPECT_CALL(read_exactly, ReadInt(0x9c000000, 0x4000000, false))
      .WillOnce(Return(0x2000000));
  EXPECT_CALL(read_exactly, ReadInt(0x9e000000, 0x2000000, false))
      .WillOnce(Return(0x1000000));
  EXPECT_CALL(read_exactly, ReadInt(0x9f000000, 0x1000000, false))
      .WillOnce(Return(0x800000));
  EXPECT_CALL(read_exactly, ReadInt(0x9f800000, 0x800000, false))
      .WillOnce(Return(0x400000));
  EXPECT_CALL(read_exactly, ReadInt(0x9fc00000, 0x400000, false))
      .WillOnce(Return(0x200000));
  EXPECT_CALL(read_exactly, ReadInt(0x9fe00000, 0x200000, false))
      .WillOnce(Return(0x100000));
  EXPECT_CALL(read_exactly, ReadInt(0x9ff00000, 0x100000, false))
      .WillOnce(Return(0x80000));
  EXPECT_CALL(read_exactly, ReadInt(0x9ff80000, 0x80000, false))
      .WillOnce(Return(0x40000));
  EXPECT_CALL(read_exactly, ReadInt(0x9ffc0000, 0x40000, false))
      .WillOnce(Return(0x20000));
  EXPECT_CALL(read_exactly, ReadInt(0x9ffe0000, 0x20000, false))
      .WillOnce(Return(0x10000));
  EXPECT_CALL(read_exactly, ReadInt(0x9fff0000, 0x10000, false))
      .WillOnce(Return(0x8000));
  EXPECT_CALL(read_exactly, ReadInt(0x9fff8000, 0x8000, false))
      .WillOnce(Return(0x4000));
  EXPECT_CALL(read_exactly, ReadInt(0x9fffc000, 0x4000, false))
      .WillOnce(Return(0x2000));
  EXPECT_CALL(read_exactly, ReadInt(0x9fffe000, 0x2000, false))
      .WillOnce(Return(0x1000));
  EXPECT_CALL(read_exactly, ReadInt(0x9ffff000, 0x1000, false))
      .WillOnce(Return(0x800));
  EXPECT_CALL(read_exactly, ReadInt(0x9ffff800, 0x800, false))
      .WillOnce(Return(0x400));
  EXPECT_CALL(read_exactly, ReadInt(0x9ffffc00, 0x400, false))
      .WillOnce(Return(0x200));
  EXPECT_CALL(read_exactly, ReadInt(0x9ffffe00, 0x200, false))
      .WillOnce(Return(0x100));
  EXPECT_CALL(read_exactly, ReadInt(0x9fffff00, 0x100, false))
      .WillOnce(Return(0x80));
  EXPECT_CALL(read_exactly, ReadInt(0x9fffff80, 0x80, false))
      .WillOnce(Return(0x40));
  EXPECT_CALL(read_exactly, ReadInt(0x9fffffc0, 0x40, false))
      .WillOnce(Return(0x20));
  EXPECT_CALL(read_exactly, ReadInt(0x9fffffe0, 0x20, false))
      .WillOnce(Return(0x10));
  EXPECT_CALL(read_exactly, ReadInt(0x9ffffff0, 0x10, false))
      .WillOnce(Return(0x8));
  EXPECT_CALL(read_exactly, ReadInt(0x9ffffff8, 0x8, false))
      .WillOnce(Return(0x4));
  EXPECT_CALL(read_exactly, ReadInt(0x9ffffffc, 0x4, false))
      .WillOnce(Return(0x2));
  EXPECT_CALL(read_exactly, ReadInt(0x9ffffffe, 0x2, false))
      .WillOnce(Return(0x1));
  EXPECT_CALL(read_exactly, ReadInt(0x9fffffff, 0x1, false))
      .WillOnce(Return(0x1));
  EXPECT_TRUE(read_exactly.ReadExactlyInt(0x80000000, 0x20000000, false));
}

TEST(FileIO, ReadExactly_LargeSuccess) {
  MockReadExactly read_exactly;
  InSequence in_sequence;
  constexpr size_t max = std::numeric_limits<uint32_t>::max();
  constexpr size_t increment = std::numeric_limits<int32_t>::max();
  EXPECT_CALL(read_exactly, ReadInt(0, max, false)).WillOnce(Return(increment));
  EXPECT_CALL(read_exactly, ReadInt(increment, max - increment, false))
      .WillOnce(Return(increment));
  EXPECT_CALL(read_exactly, ReadInt(2 * increment, 1, false))
      .WillOnce(Return(1));
  EXPECT_TRUE(read_exactly.ReadExactlyInt(0, max, false));
}

TEST(FileIO, ReadExactly_LargeShort) {
  MockReadExactly read_exactly;
  InSequence in_sequence;
  EXPECT_CALL(read_exactly, ReadInt(0, 0xffffffff, false))
      .WillOnce(Return(0x7fffffff));
  EXPECT_CALL(read_exactly, ReadInt(0x7fffffff, 0x80000000, false))
      .WillOnce(Return(0x10000000));
  EXPECT_CALL(read_exactly, ReadInt(0x8fffffff, 0x70000000, false))
      .WillOnce(Return(0));
  EXPECT_FALSE(read_exactly.ReadExactlyInt(0, 0xffffffff, false));
}

TEST(FileIO, ReadExactly_LargeFailure) {
  MockReadExactly read_exactly;
  InSequence in_sequence;
  EXPECT_CALL(read_exactly, ReadInt(0, 0xffffffff, false))
      .WillOnce(Return(0x7fffffff));
  EXPECT_CALL(read_exactly, ReadInt(0x7fffffff, 0x80000000, false))
      .WillOnce(Return(-1));
  EXPECT_FALSE(read_exactly.ReadExactlyInt(0, 0xffffffff, false));
}

TEST(FileIO, ReadExactly_TripleMax) {
  MockReadExactly read_exactly;
  InSequence in_sequence;
  constexpr size_t max = std::numeric_limits<size_t>::max();
  constexpr size_t increment =
      std::numeric_limits<std::make_signed<size_t>::type>::max();
  EXPECT_CALL(read_exactly, ReadInt(0, max, false)).WillOnce(Return(increment));
  EXPECT_CALL(read_exactly, ReadInt(increment, max - increment, false))
      .WillOnce(Return(increment));
  EXPECT_CALL(read_exactly, ReadInt(2 * increment, 1, false))
      .WillOnce(Return(1));
  EXPECT_TRUE(read_exactly.ReadExactlyInt(0, max, false));
}

class MockWriteAll : public internal::WriteAllInternal {
 public:
  MockWriteAll() : WriteAllInternal() {}
  ~MockWriteAll() {}

  // Since it’s more convenient for the test to use uintptr_t than const void*,
  // WriteAllInt() and WriteInt() adapt the types.

  bool WriteAllInt(uintptr_t data, size_t size) {
    return WriteAll(reinterpret_cast<const void*>(data), size);
  }

  MOCK_METHOD2(WriteInt, FileOperationResult(uintptr_t, size_t));

  // WriteAllInternal:
  FileOperationResult Write(const void* data, size_t size) {
    return WriteInt(reinterpret_cast<uintptr_t>(data), size);
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(MockWriteAll);
};

TEST(FileIO, WriteAll_Zero) {
  MockWriteAll write_all;
  InSequence in_sequence;
  EXPECT_CALL(write_all, WriteInt(_, _)).Times(0);
  EXPECT_TRUE(write_all.WriteAllInt(100, 0));
}

TEST(FileIO, WriteAll_SingleSmallSuccess) {
  MockWriteAll write_all;
  InSequence in_sequence;
  EXPECT_CALL(write_all, WriteInt(1000, 1)).WillOnce(Return(1));
  EXPECT_TRUE(write_all.WriteAllInt(1000, 1));
}

TEST(FileIO, WriteAll_SingleSmallFailure) {
  MockWriteAll write_all;
  InSequence in_sequence;
  EXPECT_CALL(write_all, WriteInt(1000, 1)).WillOnce(Return(-1));
  EXPECT_FALSE(write_all.WriteAllInt(1000, 1));
}

TEST(FileIO, WriteAll_DoubleSmall) {
  MockWriteAll write_all;
  InSequence in_sequence;
  EXPECT_CALL(write_all, WriteInt(0x1000, 2)).WillOnce(Return(1));
  EXPECT_CALL(write_all, WriteInt(0x1001, 1)).WillOnce(Return(1));
  EXPECT_TRUE(write_all.WriteAllInt(0x1000, 2));
}

TEST(FileIO, WriteAll_Medium) {
  MockWriteAll write_all;
  InSequence in_sequence;
  EXPECT_CALL(write_all, WriteInt(0x80000000, 0x20000000))
      .WillOnce(Return(0x10000000));
  EXPECT_CALL(write_all, WriteInt(0x90000000, 0x10000000))
      .WillOnce(Return(0x8000000));
  EXPECT_CALL(write_all, WriteInt(0x98000000, 0x8000000))
      .WillOnce(Return(0x4000000));
  EXPECT_CALL(write_all, WriteInt(0x9c000000, 0x4000000))
      .WillOnce(Return(0x2000000));
  EXPECT_CALL(write_all, WriteInt(0x9e000000, 0x2000000))
      .WillOnce(Return(0x1000000));
  EXPECT_CALL(write_all, WriteInt(0x9f000000, 0x1000000))
      .WillOnce(Return(0x800000));
  EXPECT_CALL(write_all, WriteInt(0x9f800000, 0x800000))
      .WillOnce(Return(0x400000));
  EXPECT_CALL(write_all, WriteInt(0x9fc00000, 0x400000))
      .WillOnce(Return(0x200000));
  EXPECT_CALL(write_all, WriteInt(0x9fe00000, 0x200000))
      .WillOnce(Return(0x100000));
  EXPECT_CALL(write_all, WriteInt(0x9ff00000, 0x100000))
      .WillOnce(Return(0x80000));
  EXPECT_CALL(write_all, WriteInt(0x9ff80000, 0x80000))
      .WillOnce(Return(0x40000));
  EXPECT_CALL(write_all, WriteInt(0x9ffc0000, 0x40000))
      .WillOnce(Return(0x20000));
  EXPECT_CALL(write_all, WriteInt(0x9ffe0000, 0x20000))
      .WillOnce(Return(0x10000));
  EXPECT_CALL(write_all, WriteInt(0x9fff0000, 0x10000))
      .WillOnce(Return(0x8000));
  EXPECT_CALL(write_all, WriteInt(0x9fff8000, 0x8000)).WillOnce(Return(0x4000));
  EXPECT_CALL(write_all, WriteInt(0x9fffc000, 0x4000)).WillOnce(Return(0x2000));
  EXPECT_CALL(write_all, WriteInt(0x9fffe000, 0x2000)).WillOnce(Return(0x1000));
  EXPECT_CALL(write_all, WriteInt(0x9ffff000, 0x1000)).WillOnce(Return(0x800));
  EXPECT_CALL(write_all, WriteInt(0x9ffff800, 0x800)).WillOnce(Return(0x400));
  EXPECT_CALL(write_all, WriteInt(0x9ffffc00, 0x400)).WillOnce(Return(0x200));
  EXPECT_CALL(write_all, WriteInt(0x9ffffe00, 0x200)).WillOnce(Return(0x100));
  EXPECT_CALL(write_all, WriteInt(0x9fffff00, 0x100)).WillOnce(Return(0x80));
  EXPECT_CALL(write_all, WriteInt(0x9fffff80, 0x80)).WillOnce(Return(0x40));
  EXPECT_CALL(write_all, WriteInt(0x9fffffc0, 0x40)).WillOnce(Return(0x20));
  EXPECT_CALL(write_all, WriteInt(0x9fffffe0, 0x20)).WillOnce(Return(0x10));
  EXPECT_CALL(write_all, WriteInt(0x9ffffff0, 0x10)).WillOnce(Return(0x8));
  EXPECT_CALL(write_all, WriteInt(0x9ffffff8, 0x8)).WillOnce(Return(0x4));
  EXPECT_CALL(write_all, WriteInt(0x9ffffffc, 0x4)).WillOnce(Return(0x2));
  EXPECT_CALL(write_all, WriteInt(0x9ffffffe, 0x2)).WillOnce(Return(0x1));
  EXPECT_CALL(write_all, WriteInt(0x9fffffff, 0x1)).WillOnce(Return(0x1));
  EXPECT_TRUE(write_all.WriteAllInt(0x80000000, 0x20000000));
}

TEST(FileIO, WriteAll_LargeSuccess) {
  MockWriteAll write_all;
  InSequence in_sequence;
  constexpr size_t max = std::numeric_limits<uint32_t>::max();
  constexpr size_t increment = std::numeric_limits<int32_t>::max();
  EXPECT_CALL(write_all, WriteInt(0, max)).WillOnce(Return(increment));
  EXPECT_CALL(write_all, WriteInt(increment, max - increment))
      .WillOnce(Return(increment));
  EXPECT_CALL(write_all, WriteInt(2 * increment, 1)).WillOnce(Return(1));
  EXPECT_TRUE(write_all.WriteAllInt(0, max));
}

TEST(FileIO, WriteAll_LargeFailure) {
  MockWriteAll write_all;
  InSequence in_sequence;
  EXPECT_CALL(write_all, WriteInt(0, 0xffffffff)).WillOnce(Return(0x7fffffff));
  EXPECT_CALL(write_all, WriteInt(0x7fffffff, 0x80000000)).WillOnce(Return(-1));
  EXPECT_FALSE(write_all.WriteAllInt(0, 0xffffffff));
}

TEST(FileIO, WriteAll_TripleMax) {
  MockWriteAll write_all;
  InSequence in_sequence;
  constexpr size_t max = std::numeric_limits<size_t>::max();
  constexpr size_t increment =
      std::numeric_limits<std::make_signed<size_t>::type>::max();
  EXPECT_CALL(write_all, WriteInt(0, max)).WillOnce(Return(increment));
  EXPECT_CALL(write_all, WriteInt(increment, max - increment))
      .WillOnce(Return(increment));
  EXPECT_CALL(write_all, WriteInt(2 * increment, 1)).WillOnce(Return(1));
  EXPECT_TRUE(write_all.WriteAllInt(0, max));
}

void TestOpenFileForWrite(FileHandle (*opener)(const base::FilePath&,
                                               FileWriteMode,
                                               FilePermissions)) {
  ScopedTempDir temp_dir;
  base::FilePath file_path_1 =
      temp_dir.path().Append(FILE_PATH_LITERAL("file_1"));
  ASSERT_FALSE(FileExists(file_path_1));

  ScopedFileHandle file_handle(opener(file_path_1,
                                      FileWriteMode::kReuseOrFail,
                                      FilePermissions::kWorldReadable));
  EXPECT_EQ(file_handle, kInvalidFileHandle);
  EXPECT_FALSE(FileExists(file_path_1));

  file_handle.reset(opener(file_path_1,
                           FileWriteMode::kCreateOrFail,
                           FilePermissions::kWorldReadable));
  EXPECT_NE(file_handle, kInvalidFileHandle);
  EXPECT_TRUE(FileExists(file_path_1));
  EXPECT_EQ(FileSize(file_path_1), 0);

  file_handle.reset(opener(file_path_1,
                           FileWriteMode::kReuseOrCreate,
                           FilePermissions::kWorldReadable));
  EXPECT_NE(file_handle, kInvalidFileHandle);
  EXPECT_TRUE(FileExists(file_path_1));
  EXPECT_EQ(FileSize(file_path_1), 0);

  constexpr char data = '%';
  EXPECT_TRUE(LoggingWriteFile(file_handle.get(), &data, sizeof(data)));

  // Close file_handle to ensure that the write is flushed to disk.
  file_handle.reset();
  EXPECT_EQ(FileSize(file_path_1), implicit_cast<FileOffset>(sizeof(data)));

  file_handle.reset(opener(file_path_1,
                           FileWriteMode::kReuseOrCreate,
                           FilePermissions::kWorldReadable));
  EXPECT_NE(file_handle, kInvalidFileHandle);
  EXPECT_TRUE(FileExists(file_path_1));
  EXPECT_EQ(FileSize(file_path_1), implicit_cast<FileOffset>(sizeof(data)));

  file_handle.reset(opener(file_path_1,
                           FileWriteMode::kCreateOrFail,
                           FilePermissions::kWorldReadable));
  EXPECT_EQ(file_handle, kInvalidFileHandle);
  EXPECT_TRUE(FileExists(file_path_1));
  EXPECT_EQ(FileSize(file_path_1), implicit_cast<FileOffset>(sizeof(data)));

  file_handle.reset(opener(file_path_1,
                           FileWriteMode::kReuseOrFail,
                           FilePermissions::kWorldReadable));
  EXPECT_NE(file_handle, kInvalidFileHandle);
  EXPECT_TRUE(FileExists(file_path_1));
  EXPECT_EQ(FileSize(file_path_1), implicit_cast<FileOffset>(sizeof(data)));

  file_handle.reset(opener(file_path_1,
                           FileWriteMode::kTruncateOrCreate,
                           FilePermissions::kWorldReadable));
  EXPECT_NE(file_handle, kInvalidFileHandle);
  EXPECT_TRUE(FileExists(file_path_1));
  EXPECT_EQ(FileSize(file_path_1), 0);

  base::FilePath file_path_2 =
      temp_dir.path().Append(FILE_PATH_LITERAL("file_2"));
  ASSERT_FALSE(FileExists(file_path_2));

  file_handle.reset(opener(file_path_2,
                           FileWriteMode::kTruncateOrCreate,
                           FilePermissions::kWorldReadable));
  EXPECT_NE(file_handle, kInvalidFileHandle);
  EXPECT_TRUE(FileExists(file_path_2));
  EXPECT_EQ(FileSize(file_path_2), 0);

  base::FilePath file_path_3 =
      temp_dir.path().Append(FILE_PATH_LITERAL("file_3"));
  ASSERT_FALSE(FileExists(file_path_3));

  file_handle.reset(opener(file_path_3,
                           FileWriteMode::kReuseOrCreate,
                           FilePermissions::kWorldReadable));
  EXPECT_NE(file_handle, kInvalidFileHandle);
  EXPECT_TRUE(FileExists(file_path_3));
  EXPECT_EQ(FileSize(file_path_3), 0);
}

TEST(FileIO, OpenFileForWrite) {
  TestOpenFileForWrite(OpenFileForWrite);
}

TEST(FileIO, OpenFileForReadAndWrite) {
  TestOpenFileForWrite(OpenFileForReadAndWrite);
}

TEST(FileIO, LoggingOpenFileForWrite) {
  TestOpenFileForWrite(LoggingOpenFileForWrite);
}

TEST(FileIO, LoggingOpenFileForReadAndWrite) {
  TestOpenFileForWrite(LoggingOpenFileForReadAndWrite);
}

enum class ReadOrWrite : bool {
  kRead,
  kWrite,
};

void FileShareModeTest(ReadOrWrite first, ReadOrWrite second) {
  ScopedTempDir temp_dir;
  base::FilePath shared_file =
      temp_dir.path().Append(FILE_PATH_LITERAL("shared_file"));
  {
    // Create an empty file to work on.
    ScopedFileHandle create(
        LoggingOpenFileForWrite(shared_file,
                                FileWriteMode::kCreateOrFail,
                                FilePermissions::kOwnerOnly));
  }

  auto handle1 = ScopedFileHandle(
      (first == ReadOrWrite::kRead)
          ? LoggingOpenFileForRead(shared_file)
          : LoggingOpenFileForWrite(shared_file,
                                    FileWriteMode::kReuseOrCreate,
                                    FilePermissions::kOwnerOnly));
  ASSERT_NE(handle1, kInvalidFileHandle);
  auto handle2 = ScopedFileHandle(
      (second == ReadOrWrite::kRead)
          ? LoggingOpenFileForRead(shared_file)
          : LoggingOpenFileForWrite(shared_file,
                                    FileWriteMode::kReuseOrCreate,
                                    FilePermissions::kOwnerOnly));
  EXPECT_NE(handle2, kInvalidFileHandle);

  EXPECT_NE(handle1.get(), handle2.get());
}

TEST(FileIO, FileShareMode_Read_Read) {
  FileShareModeTest(ReadOrWrite::kRead, ReadOrWrite::kRead);
}

TEST(FileIO, FileShareMode_Read_Write) {
  FileShareModeTest(ReadOrWrite::kRead, ReadOrWrite::kWrite);
}

TEST(FileIO, FileShareMode_Write_Read) {
  FileShareModeTest(ReadOrWrite::kWrite, ReadOrWrite::kRead);
}

TEST(FileIO, FileShareMode_Write_Write) {
  FileShareModeTest(ReadOrWrite::kWrite, ReadOrWrite::kWrite);
}

// Fuchsia does not currently support any sort of file locking. See
// https://crashpad.chromium.org/bug/196 and
// https://crashpad.chromium.org/bug/217.
#if !defined(OS_FUCHSIA)

TEST(FileIO, MultipleSharedLocks) {
  ScopedTempDir temp_dir;
  base::FilePath shared_file =
      temp_dir.path().Append(FILE_PATH_LITERAL("file_to_lock"));

  {
    // Create an empty file to lock.
    ScopedFileHandle create(
        LoggingOpenFileForWrite(shared_file,
                                FileWriteMode::kCreateOrFail,
                                FilePermissions::kOwnerOnly));
  }

  auto handle1 = ScopedFileHandle(LoggingOpenFileForRead(shared_file));
  ASSERT_NE(handle1, kInvalidFileHandle);
  EXPECT_TRUE(LoggingLockFile(handle1.get(), FileLocking::kShared));

  auto handle2 = ScopedFileHandle(LoggingOpenFileForRead(shared_file));
  ASSERT_NE(handle1, kInvalidFileHandle);
  EXPECT_TRUE(LoggingLockFile(handle2.get(), FileLocking::kShared));

  EXPECT_TRUE(LoggingUnlockFile(handle1.get()));
  EXPECT_TRUE(LoggingUnlockFile(handle2.get()));
}

class LockingTestThread : public Thread {
 public:
  LockingTestThread()
      : file_(), lock_type_(), iterations_(), actual_iterations_() {}

  void Init(FileHandle file,
            FileLocking lock_type,
            int iterations,
            base::subtle::Atomic32* actual_iterations) {
    ASSERT_NE(file, kInvalidFileHandle);
    file_ = ScopedFileHandle(file);
    lock_type_ = lock_type;
    iterations_ = iterations;
    actual_iterations_ = actual_iterations;
  }

 private:
  void ThreadMain() override {
    for (int i = 0; i < iterations_; ++i) {
      EXPECT_TRUE(LoggingLockFile(file_.get(), lock_type_));
      base::subtle::NoBarrier_AtomicIncrement(actual_iterations_, 1);
      EXPECT_TRUE(LoggingUnlockFile(file_.get()));
    }
  }

  ScopedFileHandle file_;
  FileLocking lock_type_;
  int iterations_;
  base::subtle::Atomic32* actual_iterations_;

  DISALLOW_COPY_AND_ASSIGN(LockingTestThread);
};

void LockingTest(FileLocking main_lock, FileLocking other_locks) {
  ScopedTempDir temp_dir;
  base::FilePath shared_file =
      temp_dir.path().Append(FILE_PATH_LITERAL("file_to_lock"));

  {
    // Create an empty file to lock.
    ScopedFileHandle create(
        LoggingOpenFileForWrite(shared_file,
                                FileWriteMode::kCreateOrFail,
                                FilePermissions::kOwnerOnly));
  }

  auto initial = ScopedFileHandle(
      (main_lock == FileLocking::kShared)
          ? LoggingOpenFileForRead(shared_file)
          : LoggingOpenFileForWrite(shared_file,
                                    FileWriteMode::kReuseOrCreate,
                                    FilePermissions::kOwnerOnly));
  ASSERT_NE(initial, kInvalidFileHandle);
  ASSERT_TRUE(LoggingLockFile(initial.get(), main_lock));

  base::subtle::Atomic32 actual_iterations = 0;

  LockingTestThread threads[20];
  int expected_iterations = 0;
  for (size_t index = 0; index < arraysize(threads); ++index) {
    int iterations_for_this_thread = static_cast<int>(index * 10);
    threads[index].Init(
        (other_locks == FileLocking::kShared)
            ? LoggingOpenFileForRead(shared_file)
            : LoggingOpenFileForWrite(shared_file,
                                      FileWriteMode::kReuseOrCreate,
                                      FilePermissions::kOwnerOnly),
        other_locks,
        iterations_for_this_thread,
        &actual_iterations);
    expected_iterations += iterations_for_this_thread;

    ASSERT_NO_FATAL_FAILURE(threads[index].Start());
  }

  base::subtle::Atomic32 result =
      base::subtle::NoBarrier_Load(&actual_iterations);
  EXPECT_EQ(result, 0);

  ASSERT_TRUE(LoggingUnlockFile(initial.get()));

  for (auto& t : threads)
    t.Join();

  result = base::subtle::NoBarrier_Load(&actual_iterations);
  EXPECT_EQ(result, expected_iterations);
}

TEST(FileIO, ExclusiveVsExclusives) {
  LockingTest(FileLocking::kExclusive, FileLocking::kExclusive);
}

TEST(FileIO, ExclusiveVsShareds) {
  LockingTest(FileLocking::kExclusive, FileLocking::kShared);
}

TEST(FileIO, SharedVsExclusives) {
  LockingTest(FileLocking::kShared, FileLocking::kExclusive);
}

#endif  // !OS_FUCHSIA

TEST(FileIO, FileSizeByHandle) {
  EXPECT_EQ(LoggingFileSizeByHandle(kInvalidFileHandle), -1);

  ScopedTempDir temp_dir;
  base::FilePath file_path =
      temp_dir.path().Append(FILE_PATH_LITERAL("file_size"));

  ScopedFileHandle file_handle(LoggingOpenFileForWrite(
      file_path, FileWriteMode::kCreateOrFail, FilePermissions::kOwnerOnly));
  ASSERT_NE(file_handle.get(), kInvalidFileHandle);
  EXPECT_EQ(LoggingFileSizeByHandle(file_handle.get()), 0);

  static constexpr char data[] = "zippyzap";
  ASSERT_TRUE(LoggingWriteFile(file_handle.get(), &data, sizeof(data)));

  EXPECT_EQ(LoggingFileSizeByHandle(file_handle.get()), 9);
}

FileHandle FileHandleForFILE(FILE* file) {
  int fd = fileno(file);
#if defined(OS_POSIX)
  return fd;
#elif defined(OS_WIN)
  return reinterpret_cast<HANDLE>(_get_osfhandle(fd));
#else
#error Port
#endif
}

TEST(FileIO, StdioFileHandle) {
  EXPECT_EQ(StdioFileHandle(StdioStream::kStandardInput),
            FileHandleForFILE(stdin));
  EXPECT_EQ(StdioFileHandle(StdioStream::kStandardOutput),
            FileHandleForFILE(stdout));
  EXPECT_EQ(StdioFileHandle(StdioStream::kStandardError),
            FileHandleForFILE(stderr));
}

}  // namespace
}  // namespace test
}  // namespace crashpad
