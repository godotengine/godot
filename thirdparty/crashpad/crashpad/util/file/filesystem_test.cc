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

#include "util/file/filesystem.h"

#include <sys/time.h>

#include "base/logging.h"
#include "build/build_config.h"
#include "gtest/gtest.h"
#include "test/errors.h"
#include "test/filesystem.h"
#include "test/gtest_disabled.h"
#include "test/scoped_temp_dir.h"
#include "util/misc/time.h"

namespace crashpad {
namespace test {
namespace {

bool CurrentTime(timespec* now) {
#if defined(OS_MACOSX)
  timeval now_tv;
  int res = gettimeofday(&now_tv, nullptr);
  if (res != 0) {
    EXPECT_EQ(res, 0) << ErrnoMessage("gettimeofday");
    return false;
  }
  TimevalToTimespec(now_tv, now);
  return true;
#elif defined(OS_POSIX)
  int res = clock_gettime(CLOCK_REALTIME, now);
  if (res != 0) {
    EXPECT_EQ(res, 0) << ErrnoMessage("clock_gettime");
    return false;
  }
  return true;
#else
  int res = timespec_get(now, TIME_UTC);
  if (res != TIME_UTC) {
    EXPECT_EQ(res, TIME_UTC);
    return false;
  }
  return true;
#endif
}

TEST(Filesystem, FileModificationTime) {
  timespec expected_time_start, expected_time_end;

  ASSERT_TRUE(CurrentTime(&expected_time_start));
  ScopedTempDir temp_dir;
  ASSERT_TRUE(CurrentTime(&expected_time_end));

  timespec dir_mtime;
  ASSERT_TRUE(FileModificationTime(temp_dir.path(), &dir_mtime));
  EXPECT_GE(dir_mtime.tv_sec, expected_time_start.tv_sec - 2);
  EXPECT_LE(dir_mtime.tv_sec, expected_time_end.tv_sec + 2);

  base::FilePath file(temp_dir.path().Append(FILE_PATH_LITERAL("file")));
  ASSERT_TRUE(CurrentTime(&expected_time_start));
  ASSERT_TRUE(CreateFile(file));
  ASSERT_TRUE(CurrentTime(&expected_time_end));

  timespec file_mtime;
  ASSERT_TRUE(FileModificationTime(file, &file_mtime));
  EXPECT_GE(file_mtime.tv_sec, expected_time_start.tv_sec - 2);
  EXPECT_LE(file_mtime.tv_sec, expected_time_end.tv_sec + 2);

  timespec file_mtime_again;
  ASSERT_TRUE(FileModificationTime(file, &file_mtime_again));
  EXPECT_EQ(file_mtime.tv_sec, file_mtime_again.tv_sec);
  EXPECT_EQ(file_mtime.tv_nsec, file_mtime_again.tv_nsec);

  timespec mtime;
  EXPECT_FALSE(FileModificationTime(base::FilePath(), &mtime));
  EXPECT_FALSE(FileModificationTime(
      temp_dir.path().Append(FILE_PATH_LITERAL("notafile")), &mtime));
}

#if !defined(OS_FUCHSIA)

TEST(Filesystem, FileModificationTime_SymbolicLinks) {
  if (!CanCreateSymbolicLinks()) {
    DISABLED_TEST();
  }

  ScopedTempDir temp_dir;

  const base::FilePath dir_link(
      temp_dir.path().Append(FILE_PATH_LITERAL("dir_link")));

  timespec expected_time_start, expected_time_end;
  ASSERT_TRUE(CurrentTime(&expected_time_start));
  ASSERT_TRUE(CreateSymbolicLink(temp_dir.path(), dir_link));
  ASSERT_TRUE(CurrentTime(&expected_time_end));

  timespec mtime, mtime2;
  ASSERT_TRUE(FileModificationTime(temp_dir.path(), &mtime));
  mtime.tv_sec -= 100;
  ASSERT_TRUE(SetFileModificationTime(temp_dir.path(), mtime));
  ASSERT_TRUE(FileModificationTime(temp_dir.path(), &mtime2));
  EXPECT_GE(mtime2.tv_sec, mtime.tv_sec - 2);
  EXPECT_LE(mtime2.tv_sec, mtime.tv_sec + 2);

  ASSERT_TRUE(FileModificationTime(dir_link, &mtime));
  EXPECT_GE(mtime.tv_sec, expected_time_start.tv_sec - 2);
  EXPECT_LE(mtime.tv_sec, expected_time_end.tv_sec + 2);

  base::FilePath file(temp_dir.path().Append(FILE_PATH_LITERAL("file")));
  ASSERT_TRUE(CreateFile(file));

  base::FilePath link(temp_dir.path().Append(FILE_PATH_LITERAL("link")));

  ASSERT_TRUE(CurrentTime(&expected_time_start));
  ASSERT_TRUE(CreateSymbolicLink(file, link));
  ASSERT_TRUE(CurrentTime(&expected_time_end));

  ASSERT_TRUE(FileModificationTime(file, &mtime));
  mtime.tv_sec -= 100;
  ASSERT_TRUE(SetFileModificationTime(file, mtime));
  ASSERT_TRUE(FileModificationTime(file, &mtime2));
  EXPECT_GE(mtime2.tv_sec, mtime.tv_sec - 2);
  EXPECT_LE(mtime2.tv_sec, mtime.tv_sec + 2);

  ASSERT_TRUE(FileModificationTime(link, &mtime));
  EXPECT_GE(mtime.tv_sec, expected_time_start.tv_sec - 2);
  EXPECT_LE(mtime.tv_sec, expected_time_end.tv_sec + 2);
}

#endif  // !OS_FUCHSIA

TEST(Filesystem, CreateDirectory) {
  ScopedTempDir temp_dir;

  base::FilePath dir(temp_dir.path().Append(FILE_PATH_LITERAL("dir")));
  EXPECT_FALSE(IsDirectory(dir, false));

  ASSERT_TRUE(
      LoggingCreateDirectory(dir, FilePermissions::kWorldReadable, false));
  EXPECT_TRUE(IsDirectory(dir, false));

  EXPECT_FALSE(
      LoggingCreateDirectory(dir, FilePermissions::kWorldReadable, false));

  base::FilePath file(dir.Append(FILE_PATH_LITERAL("file")));
  ASSERT_TRUE(CreateFile(file));

  EXPECT_TRUE(
      LoggingCreateDirectory(dir, FilePermissions::kWorldReadable, true));
  EXPECT_TRUE(IsRegularFile(file));
}

TEST(Filesystem, MoveFileOrDirectory) {
  ScopedTempDir temp_dir;

  base::FilePath file(temp_dir.path().Append(FILE_PATH_LITERAL("file")));
  ASSERT_TRUE(CreateFile(file));

  // empty paths
  EXPECT_FALSE(MoveFileOrDirectory(base::FilePath(), base::FilePath()));
  EXPECT_FALSE(MoveFileOrDirectory(base::FilePath(), file));
  EXPECT_FALSE(MoveFileOrDirectory(file, base::FilePath()));
  EXPECT_TRUE(IsRegularFile(file));

  // files
  base::FilePath file2(temp_dir.path().Append(FILE_PATH_LITERAL("file2")));
  EXPECT_TRUE(MoveFileOrDirectory(file, file2));
  EXPECT_TRUE(IsRegularFile(file2));
  EXPECT_FALSE(PathExists(file));

  EXPECT_FALSE(MoveFileOrDirectory(file, file2));
  EXPECT_TRUE(IsRegularFile(file2));
  EXPECT_FALSE(PathExists(file));

  ASSERT_TRUE(CreateFile(file));
  EXPECT_TRUE(MoveFileOrDirectory(file2, file));
  EXPECT_TRUE(IsRegularFile(file));
  EXPECT_FALSE(PathExists(file2));

  // directories
  base::FilePath dir(temp_dir.path().Append(FILE_PATH_LITERAL("dir")));
  ASSERT_TRUE(
      LoggingCreateDirectory(dir, FilePermissions::kWorldReadable, false));

  base::FilePath nested(dir.Append(FILE_PATH_LITERAL("nested")));
  ASSERT_TRUE(CreateFile(nested));

  base::FilePath dir2(temp_dir.path().Append(FILE_PATH_LITERAL("dir2")));
  EXPECT_TRUE(MoveFileOrDirectory(dir, dir2));
  EXPECT_FALSE(IsDirectory(dir, true));
  EXPECT_TRUE(IsDirectory(dir2, false));
  EXPECT_TRUE(IsRegularFile(dir2.Append(nested.BaseName())));

  ASSERT_TRUE(
      LoggingCreateDirectory(dir, FilePermissions::kWorldReadable, false));
  EXPECT_FALSE(MoveFileOrDirectory(dir, dir2));
  EXPECT_TRUE(IsDirectory(dir, false));
  EXPECT_TRUE(IsDirectory(dir2, false));
  EXPECT_TRUE(IsRegularFile(dir2.Append(nested.BaseName())));

  // files <-> directories
  EXPECT_FALSE(MoveFileOrDirectory(file, dir2));
  EXPECT_TRUE(IsDirectory(dir2, false));
  EXPECT_TRUE(IsRegularFile(file));

  EXPECT_FALSE(MoveFileOrDirectory(dir2, file));
  EXPECT_TRUE(IsDirectory(dir2, false));
  EXPECT_TRUE(IsRegularFile(file));
}

#if !defined(OS_FUCHSIA)

TEST(Filesystem, MoveFileOrDirectory_SymbolicLinks) {
  if (!CanCreateSymbolicLinks()) {
    DISABLED_TEST();
  }

  ScopedTempDir temp_dir;

  base::FilePath file(temp_dir.path().Append(FILE_PATH_LITERAL("file")));
  ASSERT_TRUE(CreateFile(file));

  // file links
  base::FilePath link(temp_dir.path().Append(FILE_PATH_LITERAL("link")));
  ASSERT_TRUE(CreateSymbolicLink(file, link));

  base::FilePath link2(temp_dir.path().Append(FILE_PATH_LITERAL("link2")));
  EXPECT_TRUE(MoveFileOrDirectory(link, link2));
  EXPECT_TRUE(PathExists(link2));
  EXPECT_FALSE(PathExists(link));

  ASSERT_TRUE(CreateSymbolicLink(file, link));
  EXPECT_TRUE(MoveFileOrDirectory(link, link2));
  EXPECT_TRUE(PathExists(link2));
  EXPECT_FALSE(PathExists(link));

  // file links <-> files
  EXPECT_TRUE(MoveFileOrDirectory(file, link2));
  EXPECT_TRUE(IsRegularFile(link2));
  EXPECT_FALSE(PathExists(file));

  ASSERT_TRUE(MoveFileOrDirectory(link2, file));
  ASSERT_TRUE(CreateSymbolicLink(file, link));
  EXPECT_TRUE(MoveFileOrDirectory(link, file));
  EXPECT_TRUE(PathExists(file));
  EXPECT_FALSE(IsRegularFile(file));
  EXPECT_FALSE(PathExists(link));
  EXPECT_TRUE(LoggingRemoveFile(file));

  // dangling file links
  ASSERT_TRUE(CreateSymbolicLink(file, link));
  EXPECT_TRUE(MoveFileOrDirectory(link, link2));
  EXPECT_TRUE(PathExists(link2));
  EXPECT_FALSE(PathExists(link));

  // directory links
  base::FilePath dir(temp_dir.path().Append(FILE_PATH_LITERAL("dir")));
  ASSERT_TRUE(
      LoggingCreateDirectory(dir, FilePermissions::kWorldReadable, false));

  ASSERT_TRUE(CreateSymbolicLink(dir, link));

  EXPECT_TRUE(MoveFileOrDirectory(link, link2));
  EXPECT_TRUE(PathExists(link2));
  EXPECT_FALSE(PathExists(link));

  // dangling directory links
  ASSERT_TRUE(LoggingRemoveDirectory(dir));
  EXPECT_TRUE(MoveFileOrDirectory(link2, link));
  EXPECT_TRUE(PathExists(link));
  EXPECT_FALSE(PathExists(link2));
}

#endif  // !OS_FUCHSIA

TEST(Filesystem, IsRegularFile) {
  EXPECT_FALSE(IsRegularFile(base::FilePath()));

  ScopedTempDir temp_dir;
  EXPECT_FALSE(IsRegularFile(temp_dir.path()));

  base::FilePath file(temp_dir.path().Append(FILE_PATH_LITERAL("file")));
  EXPECT_FALSE(IsRegularFile(file));

  ASSERT_TRUE(CreateFile(file));
  EXPECT_TRUE(IsRegularFile(file));
}

#if !defined(OS_FUCHSIA)

TEST(Filesystem, IsRegularFile_SymbolicLinks) {
  if (!CanCreateSymbolicLinks()) {
    DISABLED_TEST();
  }

  ScopedTempDir temp_dir;
  base::FilePath file(temp_dir.path().Append(FILE_PATH_LITERAL("file")));
  ASSERT_TRUE(CreateFile(file));

  base::FilePath link(temp_dir.path().Append(FILE_PATH_LITERAL("link")));
  ASSERT_TRUE(CreateSymbolicLink(file, link));
  EXPECT_FALSE(IsRegularFile(link));

  ASSERT_TRUE(LoggingRemoveFile(file));
  EXPECT_FALSE(IsRegularFile(link));

  base::FilePath dir_link(
      temp_dir.path().Append((FILE_PATH_LITERAL("dir_link"))));
  ASSERT_TRUE(CreateSymbolicLink(temp_dir.path(), dir_link));
  EXPECT_FALSE(IsRegularFile(dir_link));
}

#endif  // !OS_FUCHSIA

TEST(Filesystem, IsDirectory) {
  EXPECT_FALSE(IsDirectory(base::FilePath(), false));
  EXPECT_FALSE(IsDirectory(base::FilePath(), true));

  ScopedTempDir temp_dir;
  EXPECT_TRUE(IsDirectory(temp_dir.path(), false));

  base::FilePath file(temp_dir.path().Append(FILE_PATH_LITERAL("file")));
  EXPECT_FALSE(IsDirectory(file, false));
  EXPECT_FALSE(IsDirectory(file, true));

  ASSERT_TRUE(CreateFile(file));
  EXPECT_FALSE(IsDirectory(file, false));
  EXPECT_FALSE(IsDirectory(file, true));
}

#if !defined(OS_FUCHSIA)

TEST(Filesystem, IsDirectory_SymbolicLinks) {
  if (!CanCreateSymbolicLinks()) {
    DISABLED_TEST();
  }

  ScopedTempDir temp_dir;
  base::FilePath file(temp_dir.path().Append(FILE_PATH_LITERAL("file")));
  ASSERT_TRUE(CreateFile(file));

  base::FilePath link(temp_dir.path().Append(FILE_PATH_LITERAL("link")));
  ASSERT_TRUE(CreateSymbolicLink(file, link));
  EXPECT_FALSE(IsDirectory(link, false));
  EXPECT_FALSE(IsDirectory(link, true));

  ASSERT_TRUE(LoggingRemoveFile(file));
  EXPECT_FALSE(IsDirectory(link, false));
  EXPECT_FALSE(IsDirectory(link, true));

  base::FilePath dir_link(
      temp_dir.path().Append(FILE_PATH_LITERAL("dir_link")));
  ASSERT_TRUE(CreateSymbolicLink(temp_dir.path(), dir_link));
  EXPECT_FALSE(IsDirectory(dir_link, false));
  EXPECT_TRUE(IsDirectory(dir_link, true));
}

#endif  // !OS_FUCHSIA

TEST(Filesystem, RemoveFile) {
  EXPECT_FALSE(LoggingRemoveFile(base::FilePath()));

  ScopedTempDir temp_dir;

  base::FilePath file(temp_dir.path().Append(FILE_PATH_LITERAL("file")));
  EXPECT_FALSE(LoggingRemoveFile(file));

  ASSERT_TRUE(CreateFile(file));
  EXPECT_TRUE(IsRegularFile(file));

  base::FilePath dir(temp_dir.path().Append(FILE_PATH_LITERAL("dir")));
  ASSERT_TRUE(
      LoggingCreateDirectory(dir, FilePermissions::kWorldReadable, false));

  EXPECT_TRUE(LoggingRemoveFile(file));
  EXPECT_FALSE(PathExists(file));
  EXPECT_FALSE(LoggingRemoveFile(file));
}

#if !defined(OS_FUCHSIA)

TEST(Filesystem, RemoveFile_SymbolicLinks) {
  if (!CanCreateSymbolicLinks()) {
    DISABLED_TEST();
  }

  ScopedTempDir temp_dir;
  base::FilePath file(temp_dir.path().Append(FILE_PATH_LITERAL("file")));
  ASSERT_TRUE(CreateFile(file));

  base::FilePath dir(temp_dir.path().Append(FILE_PATH_LITERAL("dir")));
  ASSERT_TRUE(
      LoggingCreateDirectory(dir, FilePermissions::kWorldReadable, false));

  base::FilePath link(temp_dir.path().Append(FILE_PATH_LITERAL("link")));
  ASSERT_TRUE(CreateSymbolicLink(file, link));
  EXPECT_TRUE(LoggingRemoveFile(link));
  EXPECT_FALSE(PathExists(link));
  EXPECT_TRUE(PathExists(file));

  ASSERT_TRUE(CreateSymbolicLink(dir, link));
  EXPECT_TRUE(LoggingRemoveFile(link));
  EXPECT_FALSE(PathExists(link));
  EXPECT_TRUE(PathExists(dir));
}

#endif  // !OS_FUCHSIA

TEST(Filesystem, RemoveDirectory) {
  EXPECT_FALSE(LoggingRemoveDirectory(base::FilePath()));

  ScopedTempDir temp_dir;

  base::FilePath dir(temp_dir.path().Append(FILE_PATH_LITERAL("dir")));
  ASSERT_TRUE(
      LoggingCreateDirectory(dir, FilePermissions::kWorldReadable, false));

  base::FilePath file(dir.Append(FILE_PATH_LITERAL("file")));
  EXPECT_FALSE(LoggingRemoveDirectory(file));

  ASSERT_TRUE(CreateFile(file));
#if !defined(OS_FUCHSIA)
  // The current implementation of Fuchsia's rmdir() simply calls unlink(), and
  // unlink() works on all FS objects. This is incorrect as
  // http://pubs.opengroup.org/onlinepubs/9699919799/functions/rmdir.html says
  // "The directory shall be removed only if it is an empty directory." and "If
  // the directory is not an empty directory, rmdir() shall fail and set errno
  // to [EEXIST] or [ENOTEMPTY]." Upstream bug: US-400.
  EXPECT_FALSE(LoggingRemoveDirectory(file));
  EXPECT_FALSE(LoggingRemoveDirectory(dir));
#endif

  ASSERT_TRUE(LoggingRemoveFile(file));
  EXPECT_TRUE(LoggingRemoveDirectory(dir));
}

#if !defined(OS_FUCHSIA)

TEST(Filesystem, RemoveDirectory_SymbolicLinks) {
  if (!CanCreateSymbolicLinks()) {
    DISABLED_TEST();
  }

  ScopedTempDir temp_dir;
  base::FilePath dir(temp_dir.path().Append(FILE_PATH_LITERAL("dir")));
  ASSERT_TRUE(
      LoggingCreateDirectory(dir, FilePermissions::kWorldReadable, false));

  base::FilePath file(dir.Append(FILE_PATH_LITERAL("file")));
  EXPECT_FALSE(LoggingRemoveDirectory(file));

  base::FilePath link(temp_dir.path().Append(FILE_PATH_LITERAL("link")));
  ASSERT_TRUE(CreateSymbolicLink(file, link));
  EXPECT_FALSE(LoggingRemoveDirectory(link));
  EXPECT_TRUE(LoggingRemoveFile(link));

  ASSERT_TRUE(CreateSymbolicLink(dir, link));
  EXPECT_FALSE(LoggingRemoveDirectory(link));
  EXPECT_TRUE(LoggingRemoveFile(link));
}

#endif  // !OS_FUCHSIA

}  // namespace
}  // namespace test
}  // namespace crashpad
