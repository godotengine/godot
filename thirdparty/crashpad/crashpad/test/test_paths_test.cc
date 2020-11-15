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

#include "test/test_paths.h"

#include "base/files/file_path.h"
#include "gtest/gtest.h"
#include "util/file/file_io.h"

namespace crashpad {
namespace test {
namespace {

TEST(TestPaths, TestDataRoot) {
  base::FilePath test_data_root = TestPaths::TestDataRoot();
  ScopedFileHandle file(LoggingOpenFileForRead(
      test_data_root.Append(FILE_PATH_LITERAL("test"))
          .Append(FILE_PATH_LITERAL("test_paths_test_data_root.txt"))));
  EXPECT_TRUE(file.is_valid());
}

}  // namespace
}  // namespace test
}  // namespace crashpad
