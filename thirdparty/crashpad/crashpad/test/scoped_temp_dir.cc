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

#include "gtest/gtest.h"
#include "util/file/directory_reader.h"
#include "util/file/file_io.h"
#include "util/file/filesystem.h"

namespace crashpad {
namespace test {

ScopedTempDir::ScopedTempDir() : path_(CreateTemporaryDirectory()) {
}

ScopedTempDir::~ScopedTempDir() {
  RecursivelyDeleteTemporaryDirectory(path());
}

// static
void ScopedTempDir::RecursivelyDeleteTemporaryDirectory(
    const base::FilePath& path) {
  DirectoryReader reader;
  ASSERT_TRUE(reader.Open(path));
  DirectoryReader::Result result;
  base::FilePath entry;
  while ((result = reader.NextFile(&entry)) ==
         DirectoryReader::Result::kSuccess) {
    const base::FilePath entry_path(path.Append(entry));
    if (IsDirectory(entry_path, false)) {
      RecursivelyDeleteTemporaryDirectory(entry_path);
    } else {
      EXPECT_TRUE(LoggingRemoveFile(entry_path));
    }
  }
  EXPECT_EQ(result, DirectoryReader::Result::kNoMoreFiles);
  EXPECT_TRUE(LoggingRemoveDirectory(path));
}

}  // namespace test
}  // namespace crashpad
