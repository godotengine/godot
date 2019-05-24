// Copyright 2015 The Shaderc Authors. All rights reserved.
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

#include "libshaderc_util/io.h"

#include <fstream>

#include <gmock/gmock.h>

namespace {

using shaderc_util::IsAbsolutePath;
using shaderc_util::ReadFile;
using shaderc_util::WriteFile;
using shaderc_util::GetOutputStream;
using shaderc_util::GetBaseFileName;
using testing::Eq;
using testing::HasSubstr;

std::string ToString(const std::vector<char>& v) {
  return std::string(v.data(), v.size());
}

class ReadFileTest : public testing::Test {
 protected:
  // A vector to pass to ReadFile.
  std::vector<char> read_data;
};

TEST(IsAbsolutePathTest, Linux) {
  EXPECT_FALSE(IsAbsolutePath(""));
  EXPECT_TRUE(IsAbsolutePath("/"));
  EXPECT_FALSE(IsAbsolutePath("."));
  EXPECT_FALSE(IsAbsolutePath(".."));
  EXPECT_TRUE(IsAbsolutePath("/bin/echo"));
  EXPECT_TRUE(IsAbsolutePath("//etc/shadow"));
  EXPECT_TRUE(IsAbsolutePath("/../../../lib"));
  EXPECT_FALSE(IsAbsolutePath("./something"));
  EXPECT_FALSE(IsAbsolutePath("input"));
  EXPECT_FALSE(IsAbsolutePath("../test"));
  EXPECT_FALSE(IsAbsolutePath(" /abc"));
  EXPECT_TRUE(IsAbsolutePath("/abc def/ttt"));
}

TEST(IsAbsolutePathTest, Windows) {
  EXPECT_TRUE(IsAbsolutePath(R"(\\Server1000\superuser\file)"));
  EXPECT_TRUE(IsAbsolutePath(R"(\\zzzz 1000\user with space\file with space)"));
  EXPECT_TRUE(
      IsAbsolutePath(R"(C:\Program Files (x86)\Windows Folder\shader.glsl)"));
  EXPECT_FALSE(IsAbsolutePath(R"(third_party\gmock)"));
  EXPECT_FALSE(IsAbsolutePath(R"(C:..\File.txt)"));
}

TEST(GetBaseFileName, Linux) {
  EXPECT_EQ("", GetBaseFileName(""));
  EXPECT_EQ("", GetBaseFileName("/"));
  EXPECT_EQ("", GetBaseFileName("."));
  EXPECT_EQ("", GetBaseFileName(".."));
  EXPECT_EQ("echo", GetBaseFileName("/bin/echo"));
  EXPECT_EQ("shadow", GetBaseFileName("//etc/shadow"));
  EXPECT_EQ("lib", GetBaseFileName("/../../../lib"));
  EXPECT_EQ("something", GetBaseFileName("./something"));
  EXPECT_EQ("input", GetBaseFileName("input"));
  EXPECT_EQ("test", GetBaseFileName("../test"));
  EXPECT_EQ("abc", GetBaseFileName(" /abc"));
  EXPECT_EQ("ttt", GetBaseFileName("/abc def/ttt"));
}

TEST(GetBaseFileName, Windows) {
  EXPECT_EQ("file", GetBaseFileName(R"(\\Server1000\superuser\file)"));
  EXPECT_EQ("file with space",
            GetBaseFileName(R"(\\zzzz 1000\user with space\file with space)"));
  EXPECT_EQ(
      "shader.glsl",
      GetBaseFileName(R"(C:\Program Files (x86)\Windows Folder\shader.glsl)"));
  EXPECT_EQ("gmock", GetBaseFileName(R"(third_party\gmock)"));
  EXPECT_EQ("File.txt", GetBaseFileName(R"(C:..\File.txt)"));
}

TEST_F(ReadFileTest, CorrectContent) {
  ASSERT_TRUE(ReadFile("include_file.1", &read_data));
  EXPECT_EQ("The quick brown fox jumps over a lazy dog.",
            ToString(read_data));
}

TEST_F(ReadFileTest, EmptyContent) {
  ASSERT_TRUE(ReadFile("dir/subdir/include_file.2", &read_data));
  EXPECT_TRUE(read_data.empty());
}

TEST_F(ReadFileTest, FileNotFound) {
  EXPECT_FALSE(ReadFile("garbage garbage vjoiarhiupo hrfewi", &read_data));
}

TEST_F(ReadFileTest, EmptyFilename) { EXPECT_FALSE(ReadFile("", &read_data)); }

TEST(WriteFiletest, BadStream) {
  std::ofstream fstream;
  std::ostringstream err;
  std::ostream* output_stream = GetOutputStream(
      "/this/should/not/be/writable/asdfasdfasdfasdf", &fstream, &err);
  EXPECT_EQ(nullptr, output_stream);
  EXPECT_TRUE(fstream.fail());
  EXPECT_EQ(nullptr, output_stream);
  EXPECT_THAT(err.str(), HasSubstr("cannot open output file"));
}

TEST(WriteFileTest, Roundtrip) {
  const std::string content = "random content 12345";
  const std::string filename = "WriteFileTestOutput.tmp";
  std::ofstream fstream;
  std::ostringstream err;
  std::ostream* output_stream = GetOutputStream(filename, &fstream, &err);
  ASSERT_EQ(output_stream, &fstream);
  EXPECT_THAT(err.str(), Eq(""));
  ASSERT_TRUE(WriteFile(output_stream, content));
  std::vector<char> read_data;
  ASSERT_TRUE(ReadFile(filename, &read_data));
  EXPECT_EQ(content, ToString(read_data));
}

TEST(OutputStreamTest, Stdout) {
  std::ofstream fstream;
  std::ostringstream err;
  std::ostream* output_stream = GetOutputStream("-", &fstream, &err);
  EXPECT_EQ(&std::cout, output_stream);
  EXPECT_THAT(err.str(), Eq(""));
}
}  // anonymous namespace
