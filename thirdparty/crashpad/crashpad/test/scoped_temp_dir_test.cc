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

#include "test/scoped_temp_dir.h"

#include <fcntl.h>
#include <string.h>

#include "base/posix/eintr_wrapper.h"
#include "build/build_config.h"
#include "gtest/gtest.h"
#include "test/errors.h"
#include "test/file.h"

#if defined(OS_POSIX)
#include <unistd.h>
#elif defined(OS_WIN)
#include <direct.h>
#include <io.h>
#endif  // OS_POSIX

namespace crashpad {
namespace test {
namespace {

void CreateFile(const base::FilePath& path) {
#if defined(OS_POSIX)
  int fd = HANDLE_EINTR(creat(path.value().c_str(), 0644));
  ASSERT_GE(fd, 0) << ErrnoMessage("creat") << " " << path.value();

  // gcc refuses to compile ASSERT_EQ(IGNORE_EINTR(close(fd)), 0).
  int close_rv = IGNORE_EINTR(close(fd));
  ASSERT_EQ(close_rv, 0) << ErrnoMessage("close") << " " << path.value();
#elif defined(OS_WIN)
  int fd = _wcreat(path.value().c_str(), _S_IREAD | _S_IWRITE);
  ASSERT_GE(fd, 0) << ErrnoMessage("_wcreat") << " " << path.value();
  ASSERT_EQ(_close(fd), 0) << ErrnoMessage("_close") << " " << path.value();
#else
#error "Not implemented"
#endif
  EXPECT_TRUE(FileExists(path));
}

void CreateDirectory(const base::FilePath& path) {
#if defined(OS_POSIX)
  ASSERT_EQ(mkdir(path.value().c_str(), 0755), 0) << ErrnoMessage("mkdir")
                                                  << " " << path.value();
#elif defined(OS_WIN)
  ASSERT_EQ(_wmkdir(path.value().c_str()), 0) << ErrnoMessage("_wmkdir") << " "
                                              << path.value();
#else
#error "Not implemented"
#endif
  ASSERT_TRUE(FileExists(path));
}

TEST(ScopedTempDir, Empty) {
  base::FilePath path;
  {
    ScopedTempDir dir;
    path = dir.path();
    EXPECT_TRUE(FileExists(path));
  }
  EXPECT_FALSE(FileExists(path));
}

TEST(ScopedTempDir, WithTwoFiles) {
  base::FilePath parent, file1, file2;

  {
    ScopedTempDir dir;
    parent = dir.path();
    ASSERT_TRUE(FileExists(parent));

    file1 = parent.Append(FILE_PATH_LITERAL("test1"));
    CreateFile(file1);

    file2 = parent.Append(FILE_PATH_LITERAL("test 2"));
    CreateFile(file2);
  }

  EXPECT_FALSE(FileExists(file1));
  EXPECT_FALSE(FileExists(file2));
  EXPECT_FALSE(FileExists(parent));
}

TEST(ScopedTempDir, WithRecursiveDirectory) {
  base::FilePath parent, file1, child_dir, file2;

  {
    ScopedTempDir dir;
    parent = dir.path();
    ASSERT_TRUE(FileExists(parent));

    file1 = parent.Append(FILE_PATH_LITERAL(".first-level file"));
    CreateFile(file1);

    child_dir = parent.Append(FILE_PATH_LITERAL("subdir"));
    CreateDirectory(child_dir);

    file2 = child_dir.Append(FILE_PATH_LITERAL("second level file"));
    CreateFile(file2);
  }

  EXPECT_FALSE(FileExists(file1));
  EXPECT_FALSE(FileExists(file2));
  EXPECT_FALSE(FileExists(child_dir));
  EXPECT_FALSE(FileExists(parent));
}

}  // namespace
}  // namespace test
}  // namespace crashpad
