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

#include "test/file.h"

#include <errno.h>
#include <sys/stat.h>

#include "build/build_config.h"
#include "gtest/gtest.h"
#include "test/errors.h"

namespace crashpad {
namespace test {

bool FileExists(const base::FilePath& path) {
#if defined(OS_POSIX)
  struct stat st;
  int rv = lstat(path.value().c_str(), &st);
  static constexpr char stat_function[] = "lstat";
#elif defined(OS_WIN)
  struct _stat st;
  int rv = _wstat(path.value().c_str(), &st);
  static constexpr char stat_function[] = "_wstat";
#else
#error "Not implemented"
#endif
  if (rv < 0) {
    EXPECT_EQ(errno, ENOENT) << ErrnoMessage(stat_function) << " "
                             << path.value();
    return false;
  }
  return true;
}

FileOffset FileSize(const base::FilePath& path) {
#if defined(OS_POSIX)
  struct stat st;
  int rv = lstat(path.value().c_str(), &st);
  static constexpr char stat_function[] = "lstat";
#elif defined(OS_WIN)
  struct _stati64 st;
  int rv = _wstati64(path.value().c_str(), &st);
  static constexpr char stat_function[] = "_wstati64";
#else
#error "Not implemented"
#endif
  if (rv < 0) {
    ADD_FAILURE() << ErrnoMessage(stat_function) << " " << path.value();
    return -1;
  }
  return st.st_size;
}

}  // namespace test
}  // namespace crashpad
