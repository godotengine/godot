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

#include "util/file/directory_reader.h"

#include <set>

#include "base/files/file_path.h"
#include "base/logging.h"
#include "base/strings/stringprintf.h"
#include "base/strings/utf_string_conversions.h"
#include "gtest/gtest.h"
#include "test/filesystem.h"
#include "test/gtest_disabled.h"
#include "test/scoped_temp_dir.h"
#include "util/file/file_io.h"
#include "util/file/filesystem.h"

namespace crashpad {
namespace test {
namespace {

TEST(DirectoryReader, BadPaths) {
  DirectoryReader reader;
  EXPECT_FALSE(reader.Open(base::FilePath()));

  ScopedTempDir temp_dir;
  base::FilePath file(temp_dir.path().Append(FILE_PATH_LITERAL("file")));
  ASSERT_TRUE(CreateFile(file));
  EXPECT_FALSE(reader.Open(file));

  EXPECT_FALSE(
      reader.Open(temp_dir.path().Append(FILE_PATH_LITERAL("doesntexist"))));
}

#if !defined(OS_FUCHSIA)

TEST(DirectoryReader, BadPaths_SymbolicLinks) {
  if (!CanCreateSymbolicLinks()) {
    DISABLED_TEST();
  }

  ScopedTempDir temp_dir;
  base::FilePath file(temp_dir.path().Append(FILE_PATH_LITERAL("file")));
  ASSERT_TRUE(CreateFile(file));

  base::FilePath link(temp_dir.path().Append(FILE_PATH_LITERAL("link")));
  ASSERT_TRUE(CreateSymbolicLink(file, link));

  DirectoryReader reader;
  EXPECT_FALSE(reader.Open(link));

  ASSERT_TRUE(LoggingRemoveFile(file));
  EXPECT_FALSE(reader.Open(link));
}

#endif  // !OS_FUCHSIA

TEST(DirectoryReader, EmptyDirectory) {
  ScopedTempDir temp_dir;
  DirectoryReader reader;

  ASSERT_TRUE(reader.Open(temp_dir.path()));
  base::FilePath filename;
  EXPECT_EQ(reader.NextFile(&filename), DirectoryReader::Result::kNoMoreFiles);
}

void ExpectFiles(const std::set<base::FilePath>& files,
                 const std::set<base::FilePath>& expected) {
  EXPECT_EQ(files.size(), expected.size());

  for (const auto& filename : expected) {
    SCOPED_TRACE(
        base::StringPrintf("Filename: %" PRFilePath, filename.value().c_str()));
    EXPECT_NE(files.find(filename), files.end());
  }
}

void TestFilesAndDirectories(bool symbolic_links) {
  ScopedTempDir temp_dir;
  std::set<base::FilePath> expected_files;

  base::FilePath file(FILE_PATH_LITERAL("file"));
  ASSERT_TRUE(CreateFile(temp_dir.path().Append(file)));
  EXPECT_TRUE(expected_files.insert(file).second);

  base::FilePath directory(FILE_PATH_LITERAL("directory"));
  ASSERT_TRUE(LoggingCreateDirectory(temp_dir.path().Append(directory),
                                     FilePermissions::kWorldReadable,
                                     false));
  EXPECT_TRUE(expected_files.insert(directory).second);

  base::FilePath nested_file(FILE_PATH_LITERAL("nested_file"));
  ASSERT_TRUE(
      CreateFile(temp_dir.path().Append(directory).Append(nested_file)));

#if !defined(OS_FUCHSIA)

  if (symbolic_links) {
    base::FilePath link(FILE_PATH_LITERAL("link"));
    ASSERT_TRUE(CreateSymbolicLink(temp_dir.path().Append(file),
                                   temp_dir.path().Append(link)));
    EXPECT_TRUE(expected_files.insert(link).second);

    base::FilePath dangling(FILE_PATH_LITERAL("dangling"));
    ASSERT_TRUE(
        CreateSymbolicLink(base::FilePath(FILE_PATH_LITERAL("not_a_file")),
                           temp_dir.path().Append(dangling)));
    EXPECT_TRUE(expected_files.insert(dangling).second);
  }

#endif  // !OS_FUCHSIA

  std::set<base::FilePath> files;
  DirectoryReader reader;
  ASSERT_TRUE(reader.Open(temp_dir.path()));
  DirectoryReader::Result result;
  base::FilePath filename;
  while ((result = reader.NextFile(&filename)) ==
         DirectoryReader::Result::kSuccess) {
    EXPECT_TRUE(files.insert(filename).second);
  }
  EXPECT_EQ(result, DirectoryReader::Result::kNoMoreFiles);
  EXPECT_EQ(reader.NextFile(&filename), DirectoryReader::Result::kNoMoreFiles);
  ExpectFiles(files, expected_files);
}

TEST(DirectoryReader, FilesAndDirectories) {
  TestFilesAndDirectories(false);
}

#if !defined(OS_FUCHSIA)

TEST(DirectoryReader, FilesAndDirectories_SymbolicLinks) {
  if (!CanCreateSymbolicLinks()) {
    DISABLED_TEST();
  }

  TestFilesAndDirectories(true);
}

#endif  // !OS_FUCHSIA

}  // namespace
}  // namespace test
}  // namespace crashpad
