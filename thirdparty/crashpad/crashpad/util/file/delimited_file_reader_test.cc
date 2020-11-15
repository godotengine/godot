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

#include "util/file/delimited_file_reader.h"

#include <vector>

#include "base/format_macros.h"
#include "base/strings/stringprintf.h"
#include "gtest/gtest.h"
#include "util/file/string_file.h"

namespace crashpad {
namespace test {
namespace {

TEST(DelimitedFileReader, EmptyFile) {
  StringFile string_file;
  DelimitedFileReader delimited_file_reader(&string_file);

  std::string line;
  EXPECT_EQ(delimited_file_reader.GetLine(&line),
            DelimitedFileReader::Result::kEndOfFile);

  // The file is still at EOF.
  EXPECT_EQ(delimited_file_reader.GetLine(&line),
            DelimitedFileReader::Result::kEndOfFile);
}

TEST(DelimitedFileReader, EmptyOneLineFile) {
  StringFile string_file;
  string_file.SetString("\n");
  DelimitedFileReader delimited_file_reader(&string_file);

  std::string line;
  ASSERT_EQ(delimited_file_reader.GetLine(&line),
            DelimitedFileReader::Result::kSuccess);
  EXPECT_EQ(line, string_file.string());
  EXPECT_EQ(delimited_file_reader.GetLine(&line),
            DelimitedFileReader::Result::kEndOfFile);

  // The file is still at EOF.
  EXPECT_EQ(delimited_file_reader.GetLine(&line),
            DelimitedFileReader::Result::kEndOfFile);
}

TEST(DelimitedFileReader, SmallOneLineFile) {
  StringFile string_file;
  string_file.SetString("one line\n");
  DelimitedFileReader delimited_file_reader(&string_file);

  std::string line;
  ASSERT_EQ(delimited_file_reader.GetLine(&line),
            DelimitedFileReader::Result::kSuccess);
  EXPECT_EQ(line, string_file.string());
  EXPECT_EQ(delimited_file_reader.GetLine(&line),
            DelimitedFileReader::Result::kEndOfFile);

  // The file is still at EOF.
  EXPECT_EQ(delimited_file_reader.GetLine(&line),
            DelimitedFileReader::Result::kEndOfFile);
}

TEST(DelimitedFileReader, SmallOneLineFileWithoutNewline) {
  StringFile string_file;
  string_file.SetString("no newline");
  DelimitedFileReader delimited_file_reader(&string_file);

  std::string line;
  ASSERT_EQ(delimited_file_reader.GetLine(&line),
            DelimitedFileReader::Result::kSuccess);
  EXPECT_EQ(line, string_file.string());
  EXPECT_EQ(delimited_file_reader.GetLine(&line),
            DelimitedFileReader::Result::kEndOfFile);

  // The file is still at EOF.
  EXPECT_EQ(delimited_file_reader.GetLine(&line),
            DelimitedFileReader::Result::kEndOfFile);
}

TEST(DelimitedFileReader, SmallMultiLineFile) {
  StringFile string_file;
  string_file.SetString("first\nsecond line\n3rd\n");
  DelimitedFileReader delimited_file_reader(&string_file);

  std::string line;
  ASSERT_EQ(delimited_file_reader.GetLine(&line),
            DelimitedFileReader::Result::kSuccess);
  EXPECT_EQ(line, "first\n");
  ASSERT_EQ(delimited_file_reader.GetLine(&line),
            DelimitedFileReader::Result::kSuccess);
  EXPECT_EQ(line, "second line\n");
  ASSERT_EQ(delimited_file_reader.GetLine(&line),
            DelimitedFileReader::Result::kSuccess);
  EXPECT_EQ(line, "3rd\n");
  EXPECT_EQ(delimited_file_reader.GetLine(&line),
            DelimitedFileReader::Result::kEndOfFile);

  // The file is still at EOF.
  EXPECT_EQ(delimited_file_reader.GetLine(&line),
            DelimitedFileReader::Result::kEndOfFile);
}

TEST(DelimitedFileReader, SmallMultiFieldFile) {
  StringFile string_file;
  string_file.SetString("first,second field\ntwo lines,3rd,");
  DelimitedFileReader delimited_file_reader(&string_file);

  std::string field;
  ASSERT_EQ(delimited_file_reader.GetDelim(',', &field),
            DelimitedFileReader::Result::kSuccess);
  EXPECT_EQ(field, "first,");
  ASSERT_EQ(delimited_file_reader.GetDelim(',', &field),
            DelimitedFileReader::Result::kSuccess);
  EXPECT_EQ(field, "second field\ntwo lines,");
  ASSERT_EQ(delimited_file_reader.GetDelim(',', &field),
            DelimitedFileReader::Result::kSuccess);
  EXPECT_EQ(field, "3rd,");
  EXPECT_EQ(delimited_file_reader.GetDelim(',', &field),
            DelimitedFileReader::Result::kEndOfFile);

  // The file is still at EOF.
  EXPECT_EQ(delimited_file_reader.GetDelim(',', &field),
            DelimitedFileReader::Result::kEndOfFile);
}

TEST(DelimitedFileReader, SmallMultiFieldFile_MixedDelimiters) {
  StringFile string_file;
  string_file.SetString("first,second, still 2nd\t3rd\nalso\tnewline\n55555$");
  DelimitedFileReader delimited_file_reader(&string_file);

  std::string field;
  ASSERT_EQ(delimited_file_reader.GetDelim(',', &field),
            DelimitedFileReader::Result::kSuccess);
  EXPECT_EQ(field, "first,");
  ASSERT_EQ(delimited_file_reader.GetDelim('\t', &field),
            DelimitedFileReader::Result::kSuccess);
  EXPECT_EQ(field, "second, still 2nd\t");
  ASSERT_EQ(delimited_file_reader.GetLine(&field),
            DelimitedFileReader::Result::kSuccess);
  EXPECT_EQ(field, "3rd\n");
  ASSERT_EQ(delimited_file_reader.GetDelim('\n', &field),
            DelimitedFileReader::Result::kSuccess);
  EXPECT_EQ(field, "also\tnewline\n");
  ASSERT_EQ(delimited_file_reader.GetDelim('$', &field),
            DelimitedFileReader::Result::kSuccess);
  EXPECT_EQ(field, "55555$");
  EXPECT_EQ(delimited_file_reader.GetDelim('?', &field),
            DelimitedFileReader::Result::kEndOfFile);

  // The file is still at EOF.
  EXPECT_EQ(delimited_file_reader.GetLine(&field),
            DelimitedFileReader::Result::kEndOfFile);
}

TEST(DelimitedFileReader, EmptyLineMultiLineFile) {
  StringFile string_file;
  string_file.SetString("first\n\n\n4444\n");
  DelimitedFileReader delimited_file_reader(&string_file);

  std::string line;
  ASSERT_EQ(delimited_file_reader.GetLine(&line),
            DelimitedFileReader::Result::kSuccess);
  EXPECT_EQ(line, "first\n");
  ASSERT_EQ(delimited_file_reader.GetLine(&line),
            DelimitedFileReader::Result::kSuccess);
  EXPECT_EQ(line, "\n");
  ASSERT_EQ(delimited_file_reader.GetLine(&line),
            DelimitedFileReader::Result::kSuccess);
  EXPECT_EQ(line, "\n");
  ASSERT_EQ(delimited_file_reader.GetLine(&line),
            DelimitedFileReader::Result::kSuccess);
  EXPECT_EQ(line, "4444\n");
  EXPECT_EQ(delimited_file_reader.GetLine(&line),
            DelimitedFileReader::Result::kEndOfFile);

  // The file is still at EOF.
  EXPECT_EQ(delimited_file_reader.GetLine(&line),
            DelimitedFileReader::Result::kEndOfFile);
}

TEST(DelimitedFileReader, LongOneLineFile) {
  std::string contents(50000, '!');
  contents[1] = '?';
  contents.push_back('\n');

  StringFile string_file;
  string_file.SetString(contents);
  DelimitedFileReader delimited_file_reader(&string_file);

  std::string line;
  ASSERT_EQ(delimited_file_reader.GetLine(&line),
            DelimitedFileReader::Result::kSuccess);
  EXPECT_EQ(line, contents);
  EXPECT_EQ(delimited_file_reader.GetLine(&line),
            DelimitedFileReader::Result::kEndOfFile);

  // The file is still at EOF.
  EXPECT_EQ(delimited_file_reader.GetLine(&line),
            DelimitedFileReader::Result::kEndOfFile);
}

void TestLongMultiLineFile(int base_length) {
  std::vector<std::string> lines;
  std::string contents;
  for (size_t line_index = 0; line_index <= 'z' - 'a'; ++line_index) {
    char c = 'a' + static_cast<char>(line_index);

    // Mix up the lengths a little.
    std::string line(base_length + line_index * ((line_index % 3) - 1), c);

    // Mix up the data a little too.
    ASSERT_LT(line_index, line.size());
    line[line_index] -= ('a' - 'A');

    line.push_back('\n');
    contents.append(line);
    lines.push_back(line);
  }

  StringFile string_file;
  string_file.SetString(contents);
  DelimitedFileReader delimited_file_reader(&string_file);

  std::string line;
  for (size_t line_index = 0; line_index < lines.size(); ++line_index) {
    SCOPED_TRACE(base::StringPrintf("line_index %" PRIuS, line_index));
    ASSERT_EQ(delimited_file_reader.GetLine(&line),
              DelimitedFileReader::Result::kSuccess);
    EXPECT_EQ(line, lines[line_index]);
  }
  EXPECT_EQ(delimited_file_reader.GetLine(&line),
            DelimitedFileReader::Result::kEndOfFile);

  // The file is still at EOF.
  EXPECT_EQ(delimited_file_reader.GetLine(&line),
            DelimitedFileReader::Result::kEndOfFile);
}

TEST(DelimitedFileReader, LongMultiLineFile) {
  TestLongMultiLineFile(500);
}

TEST(DelimitedFileReader, ReallyLongMultiLineFile) {
  TestLongMultiLineFile(5000);
}

TEST(DelimitedFileReader, EmbeddedNUL) {
  static constexpr char kString[] = "embedded\0NUL\n";
  StringFile string_file;
  string_file.SetString(std::string(kString, arraysize(kString) - 1));
  DelimitedFileReader delimited_file_reader(&string_file);

  std::string line;
  ASSERT_EQ(delimited_file_reader.GetLine(&line),
            DelimitedFileReader::Result::kSuccess);
  EXPECT_EQ(line, string_file.string());
  EXPECT_EQ(delimited_file_reader.GetLine(&line),
            DelimitedFileReader::Result::kEndOfFile);

  // The file is still at EOF.
  EXPECT_EQ(delimited_file_reader.GetLine(&line),
            DelimitedFileReader::Result::kEndOfFile);
}

TEST(DelimitedFileReader, NULDelimiter) {
  static constexpr char kString[] = "aa\0b\0ccc\0";
  StringFile string_file;
  string_file.SetString(std::string(kString, arraysize(kString) - 1));
  DelimitedFileReader delimited_file_reader(&string_file);

  std::string field;
  ASSERT_EQ(delimited_file_reader.GetDelim('\0', &field),
            DelimitedFileReader::Result::kSuccess);
  EXPECT_EQ(field, std::string("aa\0", 3));
  ASSERT_EQ(delimited_file_reader.GetDelim('\0', &field),
            DelimitedFileReader::Result::kSuccess);
  EXPECT_EQ(field, std::string("b\0", 2));
  ASSERT_EQ(delimited_file_reader.GetDelim('\0', &field),
            DelimitedFileReader::Result::kSuccess);
  EXPECT_EQ(field, std::string("ccc\0", 4));
  EXPECT_EQ(delimited_file_reader.GetDelim('\0', &field),
            DelimitedFileReader::Result::kEndOfFile);

  // The file is still at EOF.
  EXPECT_EQ(delimited_file_reader.GetDelim('\0', &field),
            DelimitedFileReader::Result::kEndOfFile);
}

TEST(DelimitedFileReader, EdgeCases) {
  static constexpr size_t kSizes[] =
      {4094, 4095, 4096, 4097, 8190, 8191, 8192, 8193};
  for (size_t index = 0; index < arraysize(kSizes); ++index) {
    size_t size = kSizes[index];
    SCOPED_TRACE(
        base::StringPrintf("index %" PRIuS ", size %" PRIuS, index, size));

    std::string line_0(size, '$');
    line_0.push_back('\n');

    StringFile string_file;
    string_file.SetString(line_0);
    DelimitedFileReader delimited_file_reader(&string_file);

    std::string line;
    ASSERT_EQ(delimited_file_reader.GetLine(&line),
              DelimitedFileReader::Result::kSuccess);
    EXPECT_EQ(line, line_0);
    EXPECT_EQ(delimited_file_reader.GetLine(&line),
              DelimitedFileReader::Result::kEndOfFile);

    // The file is still at EOF.
    EXPECT_EQ(delimited_file_reader.GetLine(&line),
              DelimitedFileReader::Result::kEndOfFile);

    std::string line_1(size, '@');
    line_1.push_back('\n');

    string_file.SetString(line_0 + line_1);
    ASSERT_EQ(delimited_file_reader.GetLine(&line),
              DelimitedFileReader::Result::kSuccess);
    EXPECT_EQ(line, line_0);
    ASSERT_EQ(delimited_file_reader.GetLine(&line),
              DelimitedFileReader::Result::kSuccess);
    EXPECT_EQ(line, line_1);
    EXPECT_EQ(delimited_file_reader.GetLine(&line),
              DelimitedFileReader::Result::kEndOfFile);

    // The file is still at EOF.
    EXPECT_EQ(delimited_file_reader.GetLine(&line),
              DelimitedFileReader::Result::kEndOfFile);

    line_1[size] = '?';

    string_file.SetString(line_0 + line_1);
    ASSERT_EQ(delimited_file_reader.GetLine(&line),
              DelimitedFileReader::Result::kSuccess);
    EXPECT_EQ(line, line_0);
    ASSERT_EQ(delimited_file_reader.GetLine(&line),
              DelimitedFileReader::Result::kSuccess);
    EXPECT_EQ(line, line_1);
    EXPECT_EQ(delimited_file_reader.GetLine(&line),
              DelimitedFileReader::Result::kEndOfFile);

    // The file is still at EOF.
    EXPECT_EQ(delimited_file_reader.GetLine(&line),
              DelimitedFileReader::Result::kEndOfFile);
  }
}

}  // namespace
}  // namespace test
}  // namespace crashpad
